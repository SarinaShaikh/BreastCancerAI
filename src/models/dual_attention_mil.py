"""Phase 8: Dual Attention MIL — full bag-level model.

Locked design (owner-approved 2026-09-19). Composition:

    Bag
      ↓  Phase 7 InstanceFeatureExtractor (fresh init, seed 20260918,
      |  freeze_trunk True/False; NO B1/B4/pretrained weights)
      ↓  (N, 128) instance embeddings
      ↓  Stage 1  ChannelAttention (SE, r=16): c = σ(W2(ReLU(W1(h))))
      ↓  h̃ = c ⊙ h
      ↓  Stage 2  Phase 7 SingleAttentionAggregator (ABMIL, reused
      ↓           unchanged): a = softmax(V,w over h̃) within each bag
      ↓  z = Σ a_i h̃_i                                  (128,)
      ↓  BagClassifier Linear(128→1)                    (raw logit)
    bag logit

Paths:
    forward_bag    one bag (n_i, 1, 224, 224) -> dict of intermediates
    forward_list   PRIMARY dynamic path — list of per-bag tensors; bags
                   never interact; logits (B,)
    forward_padded convenience — (B, N_max, 1, 224, 224) + mask (B, N_max);
                   numerically equivalent to forward_list (TDA-05)

Outputs always expose: logits, per-bag instance attention (stage 2),
channel gates (stage 1), embeddings, h̃, and z — both attention weight
families are model attributions, NOT clinical explanations.

Parameter accounting (trainable, owner-approved erratum applied):
    trunk 163,536 · stage-1 4,240 · stage-2 16,640 · classifier 129
    → head excluding trunk 21,009 · full model 184,545
    state-dict elements (+596 BN buffers): 185,141 — reported separately.

Training (recorded for the LATER authorized experiment phase; NOT executed
in Phase 8): unweighted BCEWithLogitsLoss, Adam lr 1e-3, max 20 epochs,
patience 5 on validation bag ROC-AUC, threshold 0.5 fixed, seed 20260918.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from model.baselines.simple_cnn import CHANNELS
from src.mil.aggregation import BagClassifier
from src.mil.dual_attention import BOTTLENECK_R, DualAttentionAggregator
from src.mil.feature_extractor import InstanceFeatureExtractor

EMBED_DIM = CHANNELS[-1]          # 128

# Locked, owner-verified trainable parameter counts (TDA-10 asserts these).
EXPECTED_PARAMS = {
    "trunk": 163_536,
    "channel_attention": 4_240,
    "instance_attention": 16_640,
    "classifier": 129,
    "head_excluding_trunk": 21_009,
    "full_model": 184_545,
}
EXPECTED_STATE_DICT_ELEMENTS = 185_141   # trainable + 596 BN buffer elements


class DualAttentionMIL(nn.Module):
    """Bag-level Dual Attention MIL (two-stage: channel → instance)."""

    def __init__(self, extractor: InstanceFeatureExtractor | None = None,
                 embed_dim: int = EMBED_DIM,
                 r: int = BOTTLENECK_R) -> None:
        super().__init__()
        self.extractor = extractor or InstanceFeatureExtractor()
        self.dual_attention = DualAttentionAggregator(embed_dim, r)
        self.classifier = BagClassifier(embed_dim)

    # ------------------------------------------------------------ one bag
    def forward_bag(self, images: torch.Tensor) -> dict:
        """One bag: (n_i, 1, 224, 224) -> dict of intermediates."""
        emb = self.extractor(images)                          # (n_i, 128)
        out = self.dual_attention.forward_stacked(emb)
        logit = self.classifier(out["z"].unsqueeze(0))        # (1, 1)
        return {"logit": logit, "z": out["z"],
                "attention": out["attention"], "gates": out["gates"],
                "h_tilde": out["h_tilde"], "embeddings": emb,
                "n_instances": int(images.shape[0])}

    # ------------------------------------------------- dynamic/list (prim)
    def forward_list(self, image_tensors: Sequence[torch.Tensor]) -> dict:
        """Dynamic path: per-bag tensors of varying n_i. Bags never interact.

        Returns logits (B,), stacked z (B, 128), and per-bag lists of
        attention vectors / gate matrices.
        """
        if not image_tensors:
            raise ValueError("forward_list received no bags")
        logits, zs, atts, gates = [], [], [], []
        for x in image_tensors:
            out = self.forward_bag(x)
            logits.append(out["logit"].squeeze(0))
            zs.append(out["z"])
            atts.append(out["attention"])
            gates.append(out["gates"])
        return {"logits": torch.stack(logits).view(-1),
                "z": torch.stack(zs),
                "attention": atts, "gates": gates}

    # ------------------------------------------------- padded convenience
    def forward_padded(self, images: torch.Tensor,
                       mask: torch.Tensor) -> dict:
        """Padded path: (B, N_max, 1, 224, 224) + mask (B, N_max) bool.

        Numerically equivalent to forward_list over the same real instances
        (masked positions: exactly-zero attention, no contribution to z).
        """
        if images.ndim != 5 or images.shape[:2] != mask.shape:
            raise ValueError(
                f"expected images (B, N_max, 1, 224, 224) with matching "
                f"mask; got images {tuple(images.shape)}, "
                f"mask {tuple(mask.shape)}")
        b, n_max = images.shape[:2]
        emb = self.extractor(images.flatten(0, 1)).view(b, n_max, -1)
        out = self.dual_attention.forward_padded(emb, mask)
        logits = self.classifier(out["z"])                    # (B, 1)
        return {"logits": logits.view(-1), "z": out["z"],
                "attention": out["attention"], "gates": out["gates"],
                "h_tilde": out["h_tilde"], "embeddings": emb}

    # ---------------------------------------------------- parameter counts
    def param_counts(self) -> dict:
        """Trainable parameter counts per component (requires_grad aware)."""
        def _n(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters()
                       if p.requires_grad)
        trunk = _n(self.extractor)
        stage1 = _n(self.dual_attention.channel_attention)
        stage2 = _n(self.dual_attention.instance_attention)
        head = stage1 + stage2 + _n(self.classifier)
        return {"trunk": trunk,
                "channel_attention": stage1,
                "instance_attention": stage2,
                "classifier": _n(self.classifier),
                "head_excluding_trunk": head,
                "full_model": trunk + head}

    @staticmethod
    def state_dict_element_count(model: "DualAttentionMIL") -> int:
        """All state-dict tensor elements (trainable params + BN buffers).

        Reported SEPARATELY from trainable parameters — never conflated.
        """
        return sum(t.numel() for t in model.state_dict().values())
