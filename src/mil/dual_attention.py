"""Phase 8: dual attention — channel attention (stage 1) × instance attention (stage 2).

Locked design (owner-approved 2026-09-19; planning report + sanity check):

    Stage 1 — channel/feature attention (SE-style, per instance, no
    cross-instance interaction):
        c_i = sigmoid(W2(ReLU(W1(h_i))))     W1: 128→16, W2: 16→128 (biases)
        h̃_i = c_i ⊙ h_i                      elementwise channel gate
    Stage 2 — instance attention (the UNMODIFIED Phase 7 non-gated ABMIL,
    reused by composition — not reimplemented):
        u_i = tanh(V h̃_i + b)
        s_i = wᵀ u_i
        a_i = softmax_i(s_i)                 normalized WITHIN each bag only
    Aggregation + head:
        z = Σ_i a_i h̃_i
        logit = Linear(128 → 1)(z)           raw logit (BCEWithLogitsLoss)

This is the project's approved definition of "Dual Attention": two
mechanisms on genuinely different axes (feature/channel vs instance),
computed by different parameterizations, composed multiplicatively —
NOT two sequential instance-attention modules, NOT gated/transformer/
spatial/self/cross attention, NOT top-k/max pooling.

Interaction: stage 1 re-weights FEATURES within every instance; stage 2
re-weights INSTANCES within the bag over the channel-refined embeddings.
Because c ≡ 1 (uniform gates) makes h̃ = h, the dual model reduces EXACTLY
to the Phase 7 single-attention ABMIL — asserted by TDA-12; conversely,
neutralizing stage 2 must change outputs — asserted by TDA-11.

Parameter accounting (trainable):
    Stage 1  W1 2048+16, W2 2048+128  → 4,240   (owner-approved erratum:
                                                 planning figure 4,256 was
                                                 arithmetic bookkeeping;
                                                 corrected 2026-09-19)
    Stage 2  (Phase 7 aggregator)     → 16,640
    Classifier                        → 129
    Head excluding trunk              → 21,009
    Full jointly trainable model      → 184,545  (trunk 163,536)
State-dict elements (trainable + 596 BN buffer elements): 185,141 — reported
separately, never conflated with trainable parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model.baselines.simple_cnn import CHANNELS
from src.mil.aggregation import SingleAttentionAggregator

EMBED_DIM = CHANNELS[-1]          # 128
BOTTLENECK_R = 16                 # owner decision D-D


class ChannelAttention(nn.Module):
    """Stage 1: per-instance SE-style channel attention (squeeze-free).

    Input/output (N, 128). Gates c ∈ (0,1)^128 per instance; operates on
    each instance independently — there is NO cross-instance interaction in
    stage 1 (cross-instance normalization happens only in stage 2's
    within-bag softmax).
    """

    def __init__(self, embed_dim: int = EMBED_DIM,
                 r: int = BOTTLENECK_R) -> None:
        super().__init__()
        self.w1 = nn.Linear(embed_dim, r, bias=True)   # 128→16
        self.w2 = nn.Linear(r, embed_dim, bias=True)   # 16→128
        self.embed_dim = embed_dim
        self.r = r

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """(N, 128) -> gates c (N, 128) in (0, 1)."""
        if emb.ndim != 2 or emb.size(-1) != self.embed_dim:
            raise ValueError(
                f"ChannelAttention expects (N, {self.embed_dim}), "
                f"got {tuple(emb.shape)}")
        return torch.sigmoid(self.w2(torch.relu(self.w1(emb))))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class DualAttentionAggregator(nn.Module):
    """Stage-1 gate → h̃ → stage-2 ABMIL (Phase 7, reused) → (a, z).

    Two numerically equivalent paths (softmax invariance under −inf
    masking), mirroring the Phase 7 aggregator contract:

    stacked (primary):  emb (n_i, 128) -> (a (n_i,), gates (n_i, 128),
                        h_tilde (n_i, 128), z (128,))
    padded (convenience): emb (B, N_max, 128) + mask (B, N_max) bool ->
                        (a (B, N_max), gates (B, N_max, 128),
                        h_tilde (B, N_max, 128), z (B, 128))

    Masked positions: stage-2 logits are −inf before the softmax, so their
    attention weights are EXACTLY 0 and they contribute nothing to z.
    Stage 1 is elementwise per instance (no bag interaction), so masked rows
    simply never influence other instances. A fully-masked bag raises
    (softmax over all −inf is undefined) rather than returning garbage.
    """

    def __init__(self, embed_dim: int = EMBED_DIM,
                 r: int = BOTTLENECK_R) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(embed_dim, r)
        # Stage 2 IS the Phase 7 single-attention mechanism, reused
        # unchanged by composition — the dual model is built BESIDE the
        # Phase 7 reference implementation, not by reimplementing it.
        self.instance_attention = SingleAttentionAggregator(embed_dim)

    # ------------------------------------------------------------ stacked
    def forward_stacked(self, emb: torch.Tensor) -> dict:
        """One bag: emb (n_i, 128) -> dict(a, gates, h_tilde, z)."""
        if emb.ndim != 2 or emb.size(0) < 1:
            raise ValueError(
                f"forward_stacked expects (n_i, {self.embed_dim}), "
                f"got {tuple(emb.shape)}")
        gates = self.channel_attention(emb)                   # (n_i, 128)
        h_tilde = gates * emb                                 # (n_i, 128)
        a, z = self.instance_attention.forward_stacked(h_tilde)
        return {"attention": a, "gates": gates,
                "h_tilde": h_tilde, "z": z}

    # ------------------------------------------------------------- padded
    def forward_padded(self, emb: torch.Tensor, mask: torch.Tensor) -> dict:
        """Batch: emb (B, N_max, 128), mask (B, N_max) bool -> dict(...).

        Masked positions receive −inf stage-2 logits (exactly-zero
        attention). Raises if any bag is fully masked.
        """
        if emb.shape[:2] != mask.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} != embedding leading dims "
                f"{tuple(emb.shape[:2])}")
        n_real = mask.sum(dim=1)
        if int((n_real == 0).sum()) != 0:
            raise ValueError("attention received a bag with zero real "
                             "instances (softmax over all -inf is undefined)")
        gates = self.channel_attention(
            emb.flatten(0, 1)).view(emb.shape)                # (B, N, 128)
        h_tilde = gates * emb                                 # (B, N, 128)
        a, z = self.instance_attention.forward_padded(h_tilde, mask)
        return {"attention": a, "gates": gates,
                "h_tilde": h_tilde, "z": z}

    def param_counts(self) -> dict:
        """Separate trainable parameter counts per component."""
        return {"channel_attention": self.channel_attention.param_count(),
                "instance_attention": sum(
                    p.numel() for p in self.instance_attention.parameters())}
