from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import torch
from torch import nn

from .labels import AES_SBOX


def _get_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def _batched_log_probs(
    model: nn.Module,
    traces: np.ndarray,
    *,
    batch_size: int = 2048,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Run model inference in batches and return log-probabilities [N, 256].
    """
    x_np = np.asarray(traces)
    if x_np.ndim != 2:
        raise ValueError(f"attack_traces must have shape [N,L], got {x_np.shape}")

    dev = _get_device() if device is None else device
    model = model.to(dev)
    model.eval()

    n, l = x_np.shape
    out = np.empty((n, 256), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = torch.from_numpy(x_np[start:end]).to(dtype=torch.float32)
            xb = xb.unsqueeze(1).to(dev)  # [B, 1, L]

            logits = model(xb)  # [B, 256]
            logp = torch.log_softmax(logits, dim=1)
            out[start:end] = logp.detach().cpu().to(torch.float32).numpy()

    return out


def _rank_of_true_key(scores: np.ndarray, true_key_byte: int) -> int:
    """
    Rank is 1-based: 1 means best (highest score).
    """
    order = np.argsort(-scores, kind="mergesort")
    return int(np.where(order == int(true_key_byte))[0][0]) + 1


def compute_ge_sr(
    model: nn.Module,
    attack_traces: np.ndarray,
    attack_plaintext: np.ndarray,
    true_key_byte: int,
    byte_index: int,
    trace_steps: Iterable[int],
    num_runs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Guessing Entropy (GE) and Success Rate (SR) for one AES key byte.

    For each run, the attack traces are randomly permuted. For each step N in
    trace_steps, we accumulate:

        score[kg] += log P_model(z_guess | trace_i)
        where z_guess = SBox(plaintext[i, byte_index] XOR kg)

    Rank is the 1-based position of the true key in descending score order.
    GE is the mean rank over runs, SR is the fraction of runs with rank == 1.
    """
    traces = np.asarray(attack_traces)
    pt = np.asarray(attack_plaintext)

    if traces.ndim != 2:
        raise ValueError(f"attack_traces must have shape [N,L], got {traces.shape}")
    if pt.ndim != 2 or pt.shape[1] != 16:
        raise ValueError(f"attack_plaintext must have shape [N,16], got {pt.shape}")
    if pt.shape[0] != traces.shape[0]:
        raise ValueError("attack_traces and attack_plaintext must have the same N.")
    if not (0 <= int(byte_index) <= 15):
        raise ValueError("byte_index must be in [0, 15].")
    if not (0 <= int(true_key_byte) <= 255):
        raise ValueError("true_key_byte must be in [0, 255].")
    if int(num_runs) <= 0:
        raise ValueError("num_runs must be > 0.")

    steps = np.array(list(trace_steps), dtype=int)
    if steps.ndim != 1 or steps.size == 0:
        raise ValueError("trace_steps must be a non-empty 1D iterable of integers.")
    if np.any(steps <= 0):
        raise ValueError("trace_steps must contain only positive integers.")
    if np.any(steps > traces.shape[0]):
        raise ValueError("trace_steps cannot exceed the number of attack traces (N).")

    # Precompute log-probabilities for all traces once (runs only permute order).
    log_probs = _batched_log_probs(model, traces)  # [N, 256], float32

    # Precompute z = SBox(pt_byte XOR kg) for all pt_byte and kg.
    sbox = np.asarray(AES_SBOX, dtype=np.uint8)
    kg = np.arange(256, dtype=np.uint8)
    z_table = sbox[np.bitwise_xor.outer(np.arange(256, dtype=np.uint8), kg)]  # [256,256]

    pt_byte = pt[:, int(byte_index)].astype(np.uint8, copy=False)  # [N]

    max_step = int(steps.max())
    steps_sorted = np.sort(steps)
    step_set = set(int(x) for x in steps_sorted.tolist())

    ranks = np.empty((int(num_runs), steps.size), dtype=np.int32)

    # Fixed seed for reproducible evaluation across calls.
    rng = np.random.default_rng(1337)

    for run in range(int(num_runs)):
        perm = rng.permutation(traces.shape[0])

        scores = np.zeros(256, dtype=np.float64)
        step_to_rank: dict[int, int] = {}

        for t in range(max_step):
            idx = int(perm[t])
            z_idx = z_table[int(pt_byte[idx])]  # [256] uint8
            scores += log_probs[idx, z_idx].astype(np.float64, copy=False)

            n_traces = t + 1
            if n_traces in step_set:
                step_to_rank[n_traces] = _rank_of_true_key(scores, int(true_key_byte))

        # Map ranks back to original trace_steps order.
        for j, step in enumerate(steps):
            ranks[run, j] = step_to_rank[int(step)]

    ge = ranks.mean(axis=0, dtype=np.float64)
    sr = (ranks == 1).mean(axis=0, dtype=np.float64)
    return ge, sr


