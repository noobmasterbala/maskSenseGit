from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """
    Lightweight 1D CNN classifier for side-channel traces.

    Input:  [batch, 1, trace_length]
    Output: [batch, 256] raw logits
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=11, padding=5),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=11, padding=5),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(64, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)  # [B, 64, L]
        x = x.mean(dim=-1)  # global average pooling over time -> [B, 64]
        x = self.classifier(x)  # [B, 256]
        return x


