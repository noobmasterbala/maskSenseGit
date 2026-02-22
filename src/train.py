from __future__ import annotations

import os
import random
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


def _get_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Best-effort determinism. Some ops on some backends (incl. MPS) may still
    # be nondeterministic; we keep this minimal and stable.
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _make_dataloaders(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    dataset = TensorDataset(x, y)
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train

    split_gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=split_gen)

    loader_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=loader_gen,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return train_loader, val_loader


def train_model(
    model: nn.Module,
    profiling_traces: np.ndarray,
    profiling_labels: np.ndarray,
) -> nn.Module:
    """
    Train a side-channel classifier to predict Z = SBox(P ⊕ K) from ASCAD traces.

    - Uses only profiling traces/labels (no attack traces).
    - Deterministic 80/20 train/validation split with seed=1337.
    - Early stopping on validation loss with patience=10.
    - Saves best weights to results/model_best.pt and restores them before return.
    """
    seed = 1337
    _set_determinism(seed)

    traces = np.asarray(profiling_traces)
    labels = np.asarray(profiling_labels)

    if traces.ndim != 2:
        raise ValueError(f"profiling_traces must have shape [N,L], got {traces.shape}")
    if labels.ndim != 1:
        raise ValueError(f"profiling_labels must have shape [N], got {labels.shape}")
    if traces.shape[0] != labels.shape[0]:
        raise ValueError("profiling_traces and profiling_labels must have the same N.")

    x = torch.from_numpy(traces).to(dtype=torch.float32)
    x = x.unsqueeze(1)  # [N, 1, L]
    y = torch.from_numpy(labels).to(dtype=torch.long)

    train_loader, val_loader = _make_dataloaders(x, y, batch_size=256, seed=seed)

    device = _get_device()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    best_state = None
    patience = 10
    epochs_without_improve = 0

    os.makedirs("results", exist_ok=True)
    best_path = os.path.join("results", "model_best.pt")

    for epoch in range(1, 101):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = int(xb.shape[0])
            train_loss_sum += float(loss.detach().cpu().item()) * bs
            train_count += bs

        train_loss = train_loss_sum / max(train_count, 1)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)

                bs = int(xb.shape[0])
                val_loss_sum += float(loss.detach().cpu().item()) * bs
                val_count += bs

        val_loss = val_loss_sum / max(val_count, 1)

        print(f"{epoch} {train_loss:.6f} {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, best_path)
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                break

    if best_state is None and os.path.exists(best_path):
        best_state = torch.load(best_path, map_location="cpu")
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


