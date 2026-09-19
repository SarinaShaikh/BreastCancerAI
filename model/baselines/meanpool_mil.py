"""Phase 6 B4: Mean-Pooling MIL baseline (attention-free).

Roadmap: PROGRESS.md Phase 6. This is deliberately NOT Dual Attention MIL:
there are NO attention weights, NO gated attention, NO transformer pooling,
NO top-k/max/learned pooling. The defining aggregation is MEAN across the
instances of a bag.

Architecture (exactly the owner instruction):
    instance: 1x224x224 -> B1-style conv trunk (4 × Conv-BN-ReLU-Pool,
              channels 16/32/64/128) -> GAP -> embedding 128
    bag:      mean over instance embeddings -> Linear(128 -> 1) -> bag logit

Variable bag sizes are handled WITHOUT padding duplication: each bag's images
are stacked (n_i, 224, 224) and processed in one forward; the mean is taken
over the true n_i instances. (A masked-collate path is also provided for
later framework integration; padding positions are excluded via the mask so
padded zeros can never bias the mean.)

Training unit: BAG. Loss BCEWithLogitsLoss. This model is trained from
scratch with its own independently-initialized parameters — it never loads
B1 weights or B1 predictions (owner rule: B4 is an independent model).
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from model.baselines.simple_cnn import CHANNELS, DROPOUT_P, INPUT_CHANNELS

EMBED_DIM = CHANNELS[-1]  # 128


class MeanPoolMIL(nn.Module):
    """B4 — attention-free mean-pooling MIL baseline (bag-level)."""

    def __init__(self, in_channels: int = INPUT_CHANNELS,
                 embed_dim: int = EMBED_DIM) -> None:
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
        self.classifier = nn.Linear(embed_dim, 1)

    # ------------------------------------------------------------ helpers
    def embed_instances(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 1, 224, 224) -> instance embeddings (N, 128)."""
        z = self.features(x)
        return z.mean(dim=(2, 3))               # GAP -> (N, 128)

    def forward_stacked(self, images: torch.Tensor) -> torch.Tensor:
        """One variable-size bag: images (n_i, 1, 224, 224) -> logit (1,).

        MEAN pooling over the bag's true instances (no padding exists here).
        """
        if images.ndim != 4 or images.size(0) < 1:
            raise ValueError(
                f"forward_stacked expects (n_i, 1, 224, 224), got {tuple(images.shape)}")
        emb = self.embed_instances(images)                  # (n_i, 128)
        bag_emb = emb.mean(dim=0, keepdim=True)             # MEAN over instances
        return self.classifier(self.dropout(bag_emb))       # (1, 1)

    def forward_padded(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Padded batch: images (B, N_max, 1, 224, 224), mask (B, N_max) bool.

        The mask is the single source of validity: padded positions are
        excluded from the mean so padded zeros can never act as instances.
        Raises if any bag would be fully masked (undefined mean).
        """
        if images.shape[:2] != mask.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} != image leading dims "
                f"{tuple(images.shape[:2])}")
        n_real = mask.sum(dim=1)
        if int((n_real == 0).sum()) != 0:
            raise ValueError("forward_padded received a bag with zero real "
                             "instances (mean over empty set is undefined)")
        emb = self.embed_instances(images.flatten(0, 1))    # (B*N, 128)
        emb = emb.view(images.shape[0], images.shape[1], -1)
        m = mask.unsqueeze(-1).to(emb.dtype)                # (B, N_max, 1)
        bag_emb = (emb * m).sum(dim=1) / m.sum(dim=1)       # MASKED MEAN
        return self.classifier(self.dropout(bag_emb))       # (B, 1)

    def forward(self, images: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Unified entry: stacked single bag (4-D) or padded batch (5-D+mask)."""
        if images.ndim == 4:
            return self.forward_stacked(images)
        if images.ndim == 5:
            if mask is None:
                mask = torch.ones(images.shape[:2], dtype=torch.bool,
                                  device=images.device)
            return self.forward_padded(images, mask)
        raise ValueError(f"unsupported input rank {images.ndim}")

    @staticmethod
    def param_count(model: "MeanPoolMIL") -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
