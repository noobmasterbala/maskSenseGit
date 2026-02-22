from __future__ import annotations

from typing import Any, Dict

import h5py
import numpy as np


_CHUNK_SIZE = 4096


def _read_in_chunks(ds: Any, *, chunk_size: int = _CHUNK_SIZE) -> np.ndarray:
    """
    Read an HDF5 dataset (or h5py FieldsView) into a NumPy array using chunked
    slicing along axis 0 to keep peak memory usage low.
    """
    shape = tuple(ds.shape)
    out = np.empty(shape, dtype=ds.dtype)

    if len(shape) == 0:
        out[...] = ds[()]
        return out

    n = shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        out[start:end, ...] = ds[start:end, ...]

    return out


def _read_plaintext_and_key(meta: h5py.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract plaintext and key from ASCAD metadata.
    """
    if meta.dtype.fields is None:
        raise ValueError("ASCAD metadata is expected to be a compound dataset with fields.")

    missing = [name for name in ("plaintext", "key") if name not in meta.dtype.fields]
    if missing:
        raise KeyError(f"Metadata missing required fields: {missing}")

    def _field_base_and_shape(field: str) -> tuple[np.dtype, tuple[int, ...]]:
        dt = meta.dtype.fields[field][0]
        if dt.subdtype is None:
            return dt, ()
        base, shape = dt.subdtype
        return base, tuple(shape)

    pt_base, pt_shape = _field_base_and_shape("plaintext")
    key_base, key_shape = _field_base_and_shape("key")

    if pt_shape != (16,) or key_shape != (16,):
        raise ValueError(
            "Expected 'plaintext' and 'key' to have sub-shape (16,) in metadata. "
            f"Got plaintext={pt_shape}, key={key_shape}."
        )

    n = int(meta.shape[0])
    plaintext = np.empty((n, 16), dtype=pt_base)
    key = np.empty((n, 16), dtype=key_base)

    for start in range(0, n, _CHUNK_SIZE):
        end = min(start + _CHUNK_SIZE, n)
        chunk = meta[start:end]
        plaintext[start:end] = chunk["plaintext"]
        key[start:end] = chunk["key"]

    return plaintext, key


def load_ascad(path: str) -> Dict[str, np.ndarray]:
    """
    Load the ASCAD masked AES dataset from an HDF5 file (e.g. ASCAD_desync100.h5).

    Loads only:
    - Profiling_traces/traces
    - Profiling_traces/metadata (plaintext, key only)
    - Attack_traces/traces
    - Attack_traces/metadata (plaintext, key only)

    Mask-related metadata fields are intentionally not read or used.
    """
    with h5py.File(path, "r") as f:
        profiling_group = f["Profiling_traces"]
        attack_group = f["Attack_traces"]

        profiling_traces_ds = profiling_group["traces"]
        attack_traces_ds = attack_group["traces"]

        profiling_meta_ds = profiling_group["metadata"]
        attack_meta_ds = attack_group["metadata"]

        profiling_traces = _read_in_chunks(profiling_traces_ds)
        attack_traces = _read_in_chunks(attack_traces_ds)

        profiling_plaintext, profiling_key = _read_plaintext_and_key(profiling_meta_ds)
        attack_plaintext, attack_key = _read_plaintext_and_key(attack_meta_ds)

    print(profiling_traces.shape)
    print(attack_traces.shape)

    return {
        "profiling_traces": profiling_traces,
        "profiling_plaintext": profiling_plaintext,
        "profiling_key": profiling_key,
        "attack_traces": attack_traces,
        "attack_plaintext": attack_plaintext,
        "attack_key": attack_key,
    }


def normalize_traces(profiling_traces, attack_traces):
    profiling_traces = np.asarray(profiling_traces)
    attack_traces = np.asarray(attack_traces)

    if profiling_traces.ndim != 2 or attack_traces.ndim != 2:
        raise ValueError("Expected 2D arrays: [num_traces, num_samples].")
    if profiling_traces.shape[1] != attack_traces.shape[1]:
        raise ValueError(
            "profiling_traces and attack_traces must have the same number of samples per trace."
        )

    mean = np.mean(profiling_traces, axis=0, dtype=np.float64)
    std = np.sqrt(np.var(profiling_traces, axis=0, dtype=np.float64))

    denom = np.where(std > 0.0, std, 1.0).astype(np.float32, copy=False)
    mean = mean.astype(np.float32, copy=False)

    profiling_norm = (profiling_traces - mean) / denom
    attack_norm = (attack_traces - mean) / denom

    return profiling_norm, attack_norm


