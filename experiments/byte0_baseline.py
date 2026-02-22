from __future__ import annotations

import argparse
import csv
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Make matplotlib usable in environments where ~/.matplotlib isn't writable.
_MPLCONFIGDIR = _REPO_ROOT / "results" / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.makedirs(_MPLCONFIGDIR, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _repo_root() -> Path:
    return _REPO_ROOT


def _setup_imports() -> None:
    # Allow: python experiments/byte0_baseline.py (run from repo root or elsewhere)
    import sys

    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


@dataclass(frozen=True)
class ExperimentPaths:
    out_dir: Path
    ge_plot: Path
    sr_plot: Path
    metrics_csv: Path


def _resolve_dataset_path(user_path: str) -> Path:
    p = Path(user_path)
    if p.exists():
        return p
    if not p.is_absolute():
        p2 = _repo_root() / p
        if p2.exists():
            return p2

    # Common repo-local location (also appears in .gitignore).
    candidate = _repo_root() / "ASCAD_data" / "ASCAD_databases" / "ASCAD_desync100.h5"
    if user_path == "ASCAD_desync100.h5" and candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Dataset file not found at '{user_path}'. "
        f"Tried '{candidate}' as a fallback for ASCAD desync100."
    )


def _make_trace_steps(n_attack: int, num_points: int) -> np.ndarray:
    num_points = int(num_points)
    if num_points < 2:
        raise ValueError("--num-points must be >= 2")

    steps = np.linspace(1, int(n_attack), num=num_points, dtype=int)
    steps[-1] = int(n_attack)  # ensure we include the full set
    steps = np.unique(steps)
    if steps[0] != 1:
        steps = np.insert(steps, 0, 1)
    if steps[-1] != int(n_attack):
        steps = np.append(steps, int(n_attack))
    return steps.astype(int, copy=False)


def _true_key_byte(keys: np.ndarray, byte_index: int) -> int:
    k = np.asarray(keys)
    if k.ndim != 2 or k.shape[1] != 16:
        raise ValueError(f"Expected key shape [N,16], got {k.shape}")
    b = k[:, int(byte_index)].astype(np.uint8, copy=False)
    first = int(b[0])
    if not np.all(b == first):
        # ASCAD uses a fixed key; if this happens, still proceed deterministically.
        uniq = np.unique(b)
        logging.warning(
            "Attack key byte %d is not constant (found %d unique values). Using first value %d.",
            int(byte_index),
            int(uniq.size),
            first,
        )
    return first


def _paths(tag: str | None) -> ExperimentPaths:
    out_dir = _repo_root() / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "" if not tag else f"_{tag}"
    return ExperimentPaths(
        out_dir=out_dir,
        ge_plot=out_dir / f"byte0_baseline_ge{suffix}.png",
        sr_plot=out_dir / f"byte0_baseline_sr{suffix}.png",
        metrics_csv=out_dir / f"byte0_baseline_ge_sr{suffix}.csv",
    )


def _save_csv(path: Path, steps: np.ndarray, ge: np.ndarray, sr: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_attack_traces", "GE", "SR"])
        for n, ge_i, sr_i in zip(steps.tolist(), ge.tolist(), sr.tolist()):
            w.writerow([int(n), float(ge_i), float(sr_i)])


def _plot_ge(path: Path, steps: np.ndarray, ge: np.ndarray) -> None:
    fig = plt.figure(figsize=(7.2, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(steps, ge, label="GE", linewidth=2.0)
    ax.set_xlabel("Number of attack traces")
    ax.set_ylabel("Guessing entropy (rank)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_sr(path: Path, steps: np.ndarray, sr: np.ndarray) -> None:
    fig = plt.figure(figsize=(7.2, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(steps, sr, label="SR", linewidth=2.0)
    ax.set_xlabel("Number of attack traces")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Byte-0 baseline: train CNN on ASCAD desync100 and evaluate GE/SR.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ASCAD_desync100.h5",
        help="Path to ASCAD_desync100.h5 (or repo-local ASCAD_data/... path).",
    )
    parser.add_argument("--byte-index", type=int, default=0, help="AES byte index (0-15).")
    parser.add_argument("--num-runs", type=int, default=20, help="Ranking runs (random perms).")
    parser.add_argument(
        "--num-points",
        type=int,
        default=100,
        help="Number of x-axis points for GE/SR curves (includes N_attack).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional suffix tag for output filenames.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    _setup_imports()
    try:
        from src.dataset import load_ascad, normalize_traces
        from src.labels import compute_labels
        from src.models import SmallCNN
        from src.rank import compute_ge_sr
        from src.train import train_model
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Missing dependency/import while starting the experiment: {e}. "
            "Make sure your environment has the required packages installed "
            "(notably: numpy, matplotlib, torch, h5py)."
        ) from e

    byte_index = int(args.byte_index)
    if not (0 <= byte_index <= 15):
        raise ValueError("--byte-index must be in [0, 15]")

    paths = _paths(args.tag)

    dataset_path = _resolve_dataset_path(args.dataset)
    logging.info("Loading dataset: %s", dataset_path)
    data = load_ascad(str(dataset_path))

    profiling_traces = data["profiling_traces"]
    attack_traces = data["attack_traces"]
    logging.info(
        "Loaded traces: profiling=%s attack=%s",
        tuple(profiling_traces.shape),
        tuple(attack_traces.shape),
    )

    logging.info("Normalizing traces (fit on profiling set).")
    profiling_traces, attack_traces = normalize_traces(profiling_traces, attack_traces)
    profiling_traces = profiling_traces.astype(np.float32, copy=False)
    attack_traces = attack_traces.astype(np.float32, copy=False)

    logging.info("Computing labels for byte_index=%d (unmasked SBox output).", byte_index)
    profiling_labels = compute_labels(
        data["profiling_plaintext"],
        data["profiling_key"],
        byte_index=byte_index,
    ).astype(np.int64, copy=False)

    logging.info("Training CNN classifier.")
    model = SmallCNN()
    model = train_model(model, profiling_traces, profiling_labels)

    logging.info("Running key ranking evaluation (GE/SR).")
    true_kb = _true_key_byte(data["attack_key"], byte_index=byte_index)
    steps = _make_trace_steps(int(attack_traces.shape[0]), int(args.num_points))

    ge, sr = compute_ge_sr(
        model,
        attack_traces,
        data["attack_plaintext"],
        true_key_byte=int(true_kb),
        byte_index=byte_index,
        trace_steps=steps,
        num_runs=int(args.num_runs),
    )

    logging.info("Saving metrics CSV: %s", paths.metrics_csv)
    _save_csv(paths.metrics_csv, steps, ge, sr)

    logging.info("Plotting GE: %s", paths.ge_plot)
    _plot_ge(paths.ge_plot, steps, ge)

    logging.info("Plotting SR: %s", paths.sr_plot)
    _plot_sr(paths.sr_plot, steps, sr)

    logging.info(
        "Done. Final point: N=%d GE=%.3f SR=%.3f",
        int(steps[-1]),
        float(ge[-1]),
        float(sr[-1]),
    )


if __name__ == "__main__":
    main()

