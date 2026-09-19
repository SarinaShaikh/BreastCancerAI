"""Phase 6 B1: Simple CNN image-level baseline.

Roadmap: PROGRESS.md Phase 6 (Baseline Deep Learning Models). Owner-approved
scope 2026-09-18: B1 simple CNN; B2 ResNet/transfer learning DEFERRED (so no
pretrained weights exist anywhere in this file); CPU-only.

Architecture (exactly the owner instruction):
    Input 1x224x224 (Phase 5.5 loader output; Phase 5 normalization already
    applied — this model NEVER re-normalizes, resizes, or re-preprocesses).
    4 × [Conv3x3(s=1, pad=1) -> BatchNorm -> ReLU -> MaxPool2x2]
    channels 16 -> 32 -> 64 -> 128  (spatial 224→112→56→28→14)
    GlobalAveragePooling -> Dropout(p=0.3) -> Linear(128 -> 1)

One raw binary logit per IMAGE. Loss BCEWithLogitsLoss (applied by the
trainer). Parameter count 163,665 (~164k; inside the owner's 150k–250k
target — channels were widened from an initial 16/32/64/128 draft after
measuring 97,761 params, below the floor).

The convolutional trunk is exposed as ``features`` and the post-GAP dropout
plus classifier head as ``head``: B4 (mean-pooling MIL) reuses the same
block layout through its own independently-initialized module — B4 never
loads B1 weights (owner rule: B4 is trained from scratch as an independent
bag-level model).
"""
from __future__ import annotations

import torch
import torch.nn as nn

INPUT_CHANNELS = 1
INPUT_SIZE = 224
DROPOUT_P = 0.3
CHANNELS = (24, 48, 96, 128)
FEATURE_DIM = CHANNELS[-1]  # 128


class SimpleCNN(nn.Module):
    """B1 — compact image-level binary CNN baseline."""

    def __init__(self, in_channels: int = INPUT_CHANNELS) -> None:
        super().__init__()
        c = CHANNELS

        def block(ci: int, co: int) -> list:
            return [
                nn.Conv2d(ci, co, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ]

        self.features = nn.Sequential(
            *block(in_channels, c[0]),
            *block(c[0], c[1]),
            *block(c[1], c[2]),
            *block(c[2], c[3]),
        )
        self.dropout = nn.Dropout(DROPOUT_P)
        self.fc = nn.Linear(c[3], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)                    # (N, 128, 14, 14)
        z = z.mean(dim=(2, 3))                  # Global Average Pooling -> (N, 128)
        z = self.dropout(z)
        return self.fc(z)                       # (N, 1) raw logit

    @staticmethod
    def param_count(model: "SimpleCNN") -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
