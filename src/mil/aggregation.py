"""Phase 7: single-attention ABMIL aggregation + bag classifier + pipeline.

Scope boundary (owner instruction §5/§15 and Phase 8 boundary): this module
implements exactly ONE attention mechanism — the standard non-gated
instance-level attention (Ilse et al. 2018 form). It does NOT implement dual
attention, gated attention, transformer/self-attention, cross-bag attention,
top-k pooling, or max-pooling aggregation. (MaxPool2d inside the CNN feature
extractor is spatial downsampling, not MIL aggregation, and lives in
``feature_extractor.py``, not here.)

Mathematical specification (per bag of n instance embeddings h_i ∈ R^128):
    u_i = tanh(V h_i + b)          V ∈ R^{128×128}, b ∈ R^{128}
    s_i = wᵀ u_i                   w ∈ R^{128}      (one score per instance)
    a_i = softmax_i(s_i)           normalized WITHIN the bag only
    z   = Σ_i a_i h_i              bag representation ∈ R^{128}
    logit = Linear(128 → 1)(z)     raw logit (BCEWithLogitsLoss-compatible;
                                   no sigmoid inside the model)

Two numerically equivalent paths:
    stacked (dynamic/list): per-bag tensor (n_i, 128) — the primary path,
        matching the B4 pattern; no padding, no duplication, any n_i ≥ 1.
    padded (convenience):   (B, N_max, 128) + mask (B, N_max) bool — masked
        positions receive logits −inf before the softmax, so their attention
        weights are EXACTLY 0 and padded values can never contribute.
        A bag with zero real instances raises (softmax over all −inf is
        undefined) rather than silently returning garbage.
Both paths normalize attention within each bag — weights NEVER mix across
bags — and a one-instance bag yields weight exactly 1.0.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from model.baselines.simple_cnn import CHANNELS

EMBED_DIM = CHANNELS[-1]          # 128


class SingleAttentionAggregator(nn.Module):
    """Non-gated ABMIL attention: (instances) -> attention weights + z.

    Parameter count: V (128·128 + 128) + w (128) = 16,512 + 128 = 16,640.
    """

    def __init__(self, embed_dim: int = EMBED_DIM,
                 hidden_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.V = nn.Linear(embed_dim, hidden_dim, bias=True)
        self.w = nn.Linear(hidden_dim, 1, bias=False)
        self.embed_dim = embed_dim

    # ------------------------------------------------------------ stacked
    def forward_stacked(self, emb: torch.Tensor
                        ) -> "tuple[torch.Tensor, torch.Tensor]":
        """One bag: emb (n_i, 128) -> (attention (n_i,), z (128,))."""
        if emb.ndim != 2 or emb.size(0) < 1:
            raise ValueError(
                f"forward_stacked expects (n_i, {self.embed_dim}), "
                f"got {tuple(emb.shape)}")
        s = self.w(torch.tanh(self.V(emb))).squeeze(-1)      # (n_i,)
        a = torch.softmax(s, dim=0)                          # sums to 1
        z = (a.unsqueeze(-1) * emb).sum(dim=0)               # (128,)
        return a, z

    # ------------------------------------------------------------- padded
    def forward_padded(self, emb: torch.Tensor, mask: torch.Tensor
                       ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Batch: emb (B, N_max, 128), mask (B, N_max) bool
        -> (attention (B, N_max), z (B, 128)). Masked positions get logits
        −inf pre-softmax -> weights exactly 0. Fully-masked bags raise."""
        if emb.shape[:2] != mask.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} != embedding leading dims "
                f"{tuple(emb.shape[:2])}")
        n_real = mask.sum(dim=1)
        if int((n_real == 0).sum()) != 0:
            raise ValueError("attention received a bag with zero real "
                             "instances (softmax over all -inf is undefined)")
        s = self.w(torch.tanh(self.V(emb))).squeeze(-1)      # (B, N_max)
        s = s.masked_fill(~mask, float("-inf"))
        a = torch.softmax(s, dim=1)                          # rows sum to 1
        z = (a.unsqueeze(-1) * emb).sum(dim=1)               # (B, 128)
        return a, z


class BagClassifier(nn.Module):
    """Bag-level head: z ∈ R^128 -> Linear(128 → 1) -> raw logit.

    No sigmoid inside the model (BCEWithLogitsLoss applies it). No
    threshold logic lives here; threshold-dependent metrics use the fixed
    0.5 threshold and belong to evaluation code, not the model.
    Parameter count: 128 + 1 = 129.
    """

    def __init__(self, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.size(1) != self.head.in_features:
            raise ValueError(
                f"expected (B, {self.head.in_features}), got {tuple(z.shape)}")
        return self.head(z)


class AttentionMILPipeline(nn.Module):
    """End-to-end Phase 7 pipeline (composable; clear module boundaries).

    Bag → instances → CNN extractor → 128-d embeddings → single attention
    → 128-d bag representation → Linear(128→1) → bag logit.

    The pipeline exposes intermediates (embeddings, attention weights, bag
    representations, logits) for testing and documentation. It owns NO
    data loading (use ``src.mil.instance_generation``) and NO loss
    (apply ``BCEWithLogitsLoss`` to the returned logits).
    """

    def __init__(self, extractor, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.extractor = extractor
        self.attention = SingleAttentionAggregator(embed_dim)
        self.classifier = BagClassifier(embed_dim)

    # ------------------------------------------------------------ stacked
    def forward_bag(self, images: torch.Tensor) -> dict:
        """One bag: images (n_i, 1, 224, 224) -> dict of intermediates."""
        emb = self.extractor(images)                          # (n_i, 128)
        a, z = self.attention.forward_stacked(emb)
        logit = self.classifier(z.unsqueeze(0))               # (1, 1)
        return {"logit": logit, "bag_repr": z, "attention": a,
                "embeddings": emb, "n_instances": int(images.shape[0])}

    def forward_list(self, image_tensors: Sequence[torch.Tensor]) -> dict:
        """Dynamic/list path: per-bag tensors of varying n_i.

        Returns logits (B,) plus per-bag lists of attention vectors and bag
        representations. Bags never interact: each is processed
        independently.
        """
        if not image_tensors:
            raise ValueError("forward_list received no bags")
        logits, zs, atts = [], [], []
        for x in image_tensors:
            out = self.forward_bag(x)
            logits.append(out["logit"].squeeze(0))
            zs.append(out["bag_repr"])
            atts.append(out["attention"])
        return {"logits": torch.stack(logits).view(-1),
                "bag_reprs": torch.stack(zs),
                "attention": atts}

    # ------------------------------------------------------------- padded
    def forward_padded(self, images: torch.Tensor, mask: torch.Tensor) -> dict:
        """Padded path: images (B, N_max, 1, 224, 224), mask (B, N_max) bool.

        Numerically equivalent to forward_list over the same real instances
        (masked positions contribute exactly zero attention).
        """
        if images.ndim != 5 or images.shape[:2] != mask.shape:
            raise ValueError(
                f"expected images (B, N_max, 1, 224, 224) with matching mask; "
                f"got images {tuple(images.shape)}, mask {tuple(mask.shape)}")
        b, n_max = images.shape[:2]
        emb = self.extractor(images.flatten(0, 1)).view(b, n_max, -1)
        a, z = self.attention.forward_padded(emb, mask)
        logits = self.classifier(z).view(-1)                  # (B,)
        return {"logits": logits, "bag_reprs": z, "attention": a,
                "embeddings": emb}

    @staticmethod
    def param_count(model: "AttentionMILPipeline") -> int:
        return sum(p.numel() for p in model.parameters()
                   if p.requires_grad)
