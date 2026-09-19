"""Phase 8 validation suite (TDA-01…TDA-14) for the Dual Attention MIL.

Runs standalone (``python tests/test_phase8_dual_attention.py``) or under
pytest. Locked-design assertions (owner-approved 2026-09-19, erratum
applied): trainable parameters trunk 163,536 · stage-1 4,240 · stage-2
16,640 · classifier 129 · head 21,009 · full 184,545; state-dict elements
185,141 (reported separately).

  TDA-01  variable bag sizes {1,2,3,14,16,21,53} without shape errors
  TDA-02  exactly one raw bag logit per bag
  TDA-03  one-instance bag: attention weight exactly 1.0, z = h̃₁
  TDA-04  stage-2 attention sums to 1 within each bag (1e-6), never across
  TDA-05  padded/masked path ≡ dynamic path
  TDA-06  masked instances receive exactly zero attention
  TDA-07  no NaN/Inf anywhere
  TDA-08  deterministic inference under fixed seed (bitwise)
  TDA-09  gradients flow through BOTH attention stages; trunk per freeze mode
  TDA-10  parameter counts match the locked table exactly (no weakening)
  TDA-11  dual-attention ablation guard: neutralizing stage 2 MUST change
          outputs (fails if the second attention is removed/bypassed)
  TDA-12  uniform gates (c ≡ 1) reduce the dual model EXACTLY to the Phase 7
          single-attention ABMIL path
  TDA-13  boundary scan: forbidden mechanisms absent; CNN MaxPool2d allowed;
          test-split guards present; no MRI/cross-modal identifiers
  TDA-14  authorized train-only infrastructure smoke: deterministic sample,
          joint forward→loss→backward→step, frozen-path temp-cache
          round-trip (deleted afterward); NO performance claim

Scope guard: architecture/infrastructure validation only. No training, no
test access, no persistent cache, no performance claims.
"""
from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.baselines.simple_cnn import CHANNELS          # noqa: E402
from src.mil.aggregation import (                        # noqa: E402
    SingleAttentionAggregator)
from src.mil.dual_attention import (                     # noqa: E402
    BOTTLENECK_R, ChannelAttention, DualAttentionAggregator)
from src.mil.feature_extractor import (                  # noqa: E402
    InstanceFeatureExtractor, extract_embeddings, load_embedding_cache,
    seed_extractor, write_embedding_cache)
from src.mil.instance_generation import (                # noqa: E402
    deterministic_sample, load_bags, open_split)
from src.models.dual_attention_mil import (              # noqa: E402
    EXPECTED_PARAMS, EXPECTED_STATE_DICT_ELEMENTS, DualAttentionMIL)

SEED = 20260918
RUNNERS = []


def check(name):
    def deco(fn):
        def wrapper():
            try:
                fn()
                print(f"PASS  {name}")
                return True
            except Exception as e:  # noqa: BLE001
                print(f"FAIL  {name}: {e!r}")
                return False
        wrapper._label = name
        RUNNERS.append(wrapper)
        return wrapper
    return deco


def _model(seed: int = SEED) -> DualAttentionMIL:
    seed_extractor(seed)
    return DualAttentionMIL()


def _make(n: int, seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 1, 224, 224, generator=g)


def _pads(sizes, seed: int = 7, h: int = 28, w: int = 28):
    """Padded batch + mask + per-bag stacked tensors (small spatial size for
    CPU speed; architecture is size-agnostic thanks to GAP)."""
    n_max = max(sizes)
    g = torch.Generator().manual_seed(seed)
    imgs = torch.zeros(len(sizes), n_max, 1, h, w)
    mask = torch.zeros(len(sizes), n_max, dtype=torch.bool)
    stacked = []
    for b, n in enumerate(sizes):
        x = torch.randn(n, 1, h, w, generator=g)
        imgs[b, :n] = x
        mask[b, :n] = True
        stacked.append(x)
    return imgs, mask, stacked


# ---------------------------------------------------------------- TDA-01
@check("TDA-01 variable bag sizes {1,2,3,14,16,21,53} processed without shape errors")
def t01():
    m = _model()
    for n in (1, 2, 3, 14, 16, 21, 53):
        out = m.forward_bag(_make(n))
        assert out["logit"].shape == (1, 1), (n, out["logit"].shape)
        assert out["embeddings"].shape == (n, 128)
        assert out["attention"].shape == (n,)
        assert out["gates"].shape == (n, 128)
    sizes = (1, 2, 3, 14, 16, 21, 53)
    imgs, mask, _ = _pads(sizes)
    op = m.forward_padded(imgs, mask)
    assert op["logits"].shape == (len(sizes),)
    assert op["gates"].shape == (len(sizes), max(sizes), 128)


# ---------------------------------------------------------------- TDA-02
@check("TDA-02 exactly one raw bag logit per bag (dynamic + padded)")
def t02():
    m = _model().eval()   # eval: BN uses fixed running stats (determinism)
    sizes = (14, 16, 21, 53, 1, 2, 3)
    o = m.forward_list([_make(n) for n in sizes])
    assert o["logits"].shape == (len(sizes),) and o["logits"].ndim == 1
    imgs, mask, _ = _pads(sizes)
    op = m.forward_padded(imgs, mask)
    assert op["logits"].shape == (len(sizes),)
    # logits are the classifier applied to the returned representations
    assert torch.equal(m.classifier(o["z"]).view(-1), o["logits"])
    # RAW logit proof (structural): the model contains no Sigmoid module...
    assert not any(isinstance(mod, torch.nn.Sigmoid) for mod in m.modules())
    # ...and an amplified input drives |logit| > 1 (unbounded output;
    # eval-mode BN keeps the fixed running stats so the scale survives)
    with torch.no_grad():
        big = float(m.forward_bag(_make(21, seed=5) * 200.0)["logit"])
    assert abs(big) > 1.0, big


# ---------------------------------------------------------------- TDA-03
@check("TDA-03 one-instance bag: attention weight exactly 1.0, z = h_tilde")
def t03():
    m = _model()
    out = m.forward_bag(_make(1))
    assert float(out["attention"][0].detach()) == 1.0
    assert torch.allclose(out["z"], out["h_tilde"][0], atol=1e-6)
    # robustness only: NO single-instance bag exists in the dataset (min 14)


# ---------------------------------------------------------------- TDA-04
@check("TDA-04 stage-2 attention sums to 1 within each bag (1e-6), never across bags")
def t04():
    m = _model()
    sizes = (14, 16, 21, 53, 3)
    o = m.forward_list([_make(n) for n in sizes])
    for n, a in zip(sizes, o["attention"]):
        assert abs(float(a.detach().sum()) - 1.0) < 1e-6, (n, float(a.sum()))
        assert (a >= 0).all() and (a <= 1).all()
    imgs, mask, _ = _pads(sizes)
    op = m.forward_padded(imgs, mask)
    sums = op["attention"].sum(dim=1)
    assert torch.allclose(sums, torch.ones(len(sizes)), atol=1e-6)


# ---------------------------------------------------------------- TDA-05
@check("TDA-05 padded/masked path == dynamic path (numerical equivalence)")
def t05():
    m = _model().eval()
    sizes = (14, 16, 21)
    imgs, mask, stacked = _pads(sizes)
    with torch.no_grad():
        op = m.forward_padded(imgs, mask)
        ol = m.forward_list(stacked)
    assert torch.allclose(op["logits"], ol["logits"], atol=1e-5), \
        (op["logits"], ol["logits"])
    for b in range(len(sizes)):
        assert torch.allclose(op["z"][b], ol["z"][b], atol=1e-5)
        assert torch.allclose(op["attention"][b, :sizes[b]],
                              ol["attention"][b], atol=1e-5)
        assert torch.allclose(op["gates"][b, :sizes[b]],
                              ol["gates"][b], atol=1e-5)


# ---------------------------------------------------------------- TDA-06
@check("TDA-06 masked instances receive exactly zero attention and never affect z")
def t06():
    agg = DualAttentionAggregator()
    mask = torch.tensor([[True] * 5 + [False] * 3,
                         [True] * 3 + [False] * 5])
    emb = torch.randn(2, 8, 128)
    out = agg.forward_padded(emb, mask)
    a, z = out["attention"], out["z"]
    assert float(a[0, 5:].abs().sum()) == 0.0
    assert float(a[1, 3:].abs().sum()) == 0.0
    # z must equal the stacked-path z over the real instances only
    z0 = agg.forward_stacked(emb[0, :5])["z"]
    z1 = agg.forward_stacked(emb[1, :3])["z"]
    assert torch.allclose(z[0], z0, atol=1e-6)
    assert torch.allclose(z[1], z1, atol=1e-6)


# ---------------------------------------------------------------- TDA-07
@check("TDA-07 no NaN/Inf in embeddings, gates, h_tilde, attention, z, logits")
def t07():
    m = _model().eval()
    sizes = (1, 3, 14, 53)
    with torch.no_grad():
        o = m.forward_list([_make(n) for n in sizes])
        imgs, mask, _ = _pads(sizes)
        op = m.forward_padded(imgs, mask)
    for t in (o["logits"], o["z"]):
        assert torch.isfinite(t).all()
    for a in o["attention"]:
        assert torch.isfinite(a).all()
    for t in (op["logits"], op["z"], op["embeddings"], op["h_tilde"],
              op["attention"], op["gates"]):
        assert torch.isfinite(t).all()


# ---------------------------------------------------------------- TDA-08
@check("TDA-08 deterministic inference under fixed seed (bitwise)")
def t08():
    outs = []
    for _ in range(2):
        m = _model().eval()
        with torch.no_grad():
            outs.append(m.forward_bag(_make(21))["logit"].clone())
    assert torch.equal(outs[0], outs[1]), "bitwise determinism violated"


# ---------------------------------------------------------------- TDA-09
@check("TDA-09 gradients through BOTH stages; trunk grads per freeze mode")
def t09():
    # frozen trunk: no trunk grads; both attention stages + head still learn
    m = _model()
    m.extractor.freeze_trunk(True)
    out = m.forward_bag(_make(6))
    out["logit"].sum().backward()
    assert all(p.grad is None for p in m.extractor.parameters()), \
        "frozen trunk must have no gradients"
    assert all(p.grad is not None for p in
               m.dual_attention.channel_attention.parameters()), \
        "stage-1 channel attention must receive gradients in frozen mode"
    assert all(p.grad is not None for p in
               m.dual_attention.instance_attention.parameters()), \
        "stage-2 instance attention must receive gradients in frozen mode"
    assert all(p.grad is not None for p in m.classifier.parameters())

    # joint trunk: gradients reach the trunk as well
    m2 = _model()
    m2.extractor.freeze_trunk(False)
    out2 = m2.forward_bag(_make(6))
    out2["logit"].sum().backward()
    assert any(g is not None for g in
               [p.grad for p in m2.extractor.parameters()]), \
        "joint mode must propagate gradients into the trunk"
    assert all(p.grad is not None for p in
               m2.dual_attention.channel_attention.parameters())


# ---------------------------------------------------------------- TDA-10
@check("TDA-10 locked parameter counts: 184,545 trainable; 185,141 state-dict elements")
def t10():
    m = _model()
    counts = m.param_counts()
    for key, expected in EXPECTED_PARAMS.items():
        assert counts[key] == expected, \
            f"{key}: {counts[key]} != locked {expected}"
    assert m.param_counts()["full_model"] == 184_545
    sd = DualAttentionMIL.state_dict_element_count(m)
    assert sd == EXPECTED_STATE_DICT_ELEMENTS, \
        f"state-dict elements {sd} != locked {EXPECTED_STATE_DICT_ELEMENTS}"
    assert sum(1 for p in m.parameters() if p.requires_grad) > 0


# ---------------------------------------------------------------- TDA-11
@check("TDA-11 ablation guard: neutralizing stage 2 MUST change outputs")
def t11():
    # Stage-2 observable: the ATTENTION VECTOR itself (the logit is an
    # insensitive linear readout at init — measured z-diff ~1e-7, attention
    # diff ~3e-5, float noise ~1e-9 — so thresholds are set from MEASURED
    # margins, not guesses).
    m = _model().eval()
    x = _make(16, seed=123)
    agg = m.dual_attention
    with torch.no_grad():
        a_full = m.forward_bag(x)["attention"].detach().clone()
        # (a) neutralize stage 2: w = 0 makes every s_i equal -> uniform
        # attention (mean pooling over h_tilde). If stage 2 were dead or
        # bypassed, the attention vector would be bitwise unchanged.
        w_backup = agg.instance_attention.w.weight.clone()
        agg.instance_attention.w.weight.zero_()
        o_neutral = m.forward_bag(x)
        a_neutral, z_neutral = o_neutral["attention"], o_neutral["z"]
        # (b) amplified contrast: w x50 must move attention and z far from
        # uniform (each observable captured under its own weights)
        agg.instance_attention.w.weight.copy_(w_backup * 50.0)
        o_amp = m.forward_bag(x)
        a_amp, z_amp = o_amp["attention"], o_amp["z"]
        agg.instance_attention.w.weight.copy_(w_backup)
    assert not torch.allclose(a_full, a_neutral, atol=1e-8), \
        "attention unchanged with stage-2 neutralized — stage 2 is dead/bypassed"
    assert not torch.allclose(a_amp, a_neutral, atol=1e-6), \
        "amplified stage-2 does not change attention — stage 2 is ineffective"
    assert not torch.allclose(z_amp, z_neutral, atol=1e-6), \
        "z unchanged between amplified and neutralized stage 2"
    assert torch.equal(agg.instance_attention.w.weight, w_backup)
    # Stage-1 ablation: driving all channel gates to ~0 must collapse z.
    with torch.no_grad():
        z_full = m.forward_bag(x)["z"].detach().clone()
        w2w = agg.channel_attention.w2.weight.clone()
        w2b = agg.channel_attention.w2.bias.clone()
        agg.channel_attention.w2.weight.zero_()
        agg.channel_attention.w2.bias.fill_(-10.0)  # sigmoid(-10) ~ 4.5e-5
        out_ab = m.forward_bag(x)
        agg.channel_attention.w2.weight.copy_(w2w)
        agg.channel_attention.w2.bias.copy_(w2b)
    assert float(out_ab["gates"].abs().max()) < 1e-4, \
        "gates should be ~0 under ablation"
    assert not torch.allclose(z_full, out_ab["z"], atol=1e-4), \
        "z unchanged with stage-1 gates ~0 — stage 1 is dead"
    assert torch.equal(agg.channel_attention.w2.weight, w2w) \
        and torch.equal(agg.channel_attention.w2.bias, w2b)


# ---------------------------------------------------------------- TDA-12
@check("TDA-12 uniform gates (c == 1) reduce the dual model exactly to Phase 7 ABMIL")
def t12():
    m = _model().eval()
    x = _make(21, seed=42)
    # Phase 7 reference: same extractor weights, plain ABMIL over h
    seed_extractor(SEED)
    ref_extractor = InstanceFeatureExtractor().eval()  # eval: BN running stats
    ref = SingleAttentionAggregator()
    ref_head = torch.nn.Linear(128, 1)
    # copy the dual model's stage-2 + head weights into the reference path
    ref.load_state_dict(m.dual_attention.instance_attention.state_dict())
    # BagClassifier wraps its Linear as `.head` — load that submodule's state
    ref_head.load_state_dict(m.classifier.head.state_dict())
    ref_extractor.load_state_dict(m.extractor.state_dict())
    with torch.no_grad():
        h = ref_extractor(x)
        a7, z7 = ref.forward_stacked(h)
        logit7 = ref_head(z7.unsqueeze(0))
        # dual model with gates forced to exactly 1.0 (fp32 sigmoid(20) == 1.0)
        agg = m.dual_attention
        agg.channel_attention.w1.weight.zero_()
        agg.channel_attention.w1.bias.zero_()
        agg.channel_attention.w2.weight.zero_()
        agg.channel_attention.w2.bias.fill_(20.0)
        out = m.forward_bag(x)
    assert torch.all(out["gates"] == 1.0), "gates must be exactly 1 here"
    assert torch.allclose(out["attention"], a7, atol=1e-6), \
        (out["attention"] - a7).abs().max()
    assert torch.allclose(out["z"], z7, atol=1e-5)
    assert torch.allclose(out["logit"], logit7, atol=1e-4), \
        (out["logit"], logit7)


# ---------------------------------------------------------------- TDA-13
@check("TDA-13 boundary scan: forbidden mechanisms absent; guards present")
def t13_boundary():
    targets = ["src/mil/dual_attention.py",
               "src/models/dual_attention_mil.py"]
    banned_names = {"gated_attention", "GatedAttention", "MultiheadAttention",
                    "Transformer", "transformer", "cross_bag", "CrossBag",
                    "topk", "top_k", "amax", "self_attention", "SelfAttention"}
    for rel in targets:
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id not in banned_names, (rel, node.id)
            if isinstance(node, ast.Attribute):
                assert node.attr not in banned_names, (rel, node.attr)
    for rel in targets:
        src = (REPO / rel).read_text(encoding="utf-8")
        for token in ("GatedAttention", "MultiheadAttention", "torch.topk",
                      ".max(dim", ".amax(", "nn.MaxPool1d", "nn.MultiheadAttention"):
            assert token not in src, (rel, token)
        for token in ("MRI", "mri", "cross_modal", "crossmodal"):
            assert token not in src, (rel, token)
    # CNN spatial MaxPool2d is legitimate and must exist in the extractor
    ex_src = (REPO / "src/mil/feature_extractor.py").read_text(encoding="utf-8")
    assert "nn.MaxPool2d(2, 2)" in ex_src
    # aggregation module stays pooling-free; dual module adds no pooling
    agg_src = (REPO / "src/mil/aggregation.py").read_text(encoding="utf-8")
    da_src = (REPO / "src/mil/dual_attention.py").read_text(encoding="utf-8")
    assert "nn.MaxPool" not in agg_src and "nn.MaxPool" not in da_src
    # test-split guards still authoritative in the reused Phase 7 modules
    ig = (REPO / "src/mil/instance_generation.py").read_text(encoding="utf-8")
    assert 'PHASE7_SPLITS = ("train", "val")' in ig
    # stage-2 must reuse (not reimplement) the Phase 7 aggregator
    assert "SingleAttentionAggregator" in da_src


# ---------------------------------------------------------------- TDA-14
@check("TDA-14 authorized train-only smoke: joint step + frozen temp-cache round-trip (no persistent cache)")
def t14():
    import torch.nn as _nn
    ds = open_split("train")
    sample_ids = deterministic_sample(ds, 8, SEED)
    sub = load_bags(ds, sample_ids)
    sizes = [b.n_instances for b in sub]
    assert len(sub) == 8, f"smoke sample must be exactly 8 bags, got {len(sub)}"
    assert 14 in sizes, \
        f"deterministic sample must include the 14-instance train bag; got {sorted(sizes)}"

    # ---- joint mode: forward -> BCEWithLogitsLoss -> backward -> step ----
    seed_extractor(SEED)
    m = DualAttentionMIL()
    m.extractor.freeze_trunk(False)
    m.train()
    labels = torch.tensor([b.numeric_label for b in sub], dtype=torch.float32)
    out = m.forward_list([b.images for b in sub])
    loss = _nn.functional.binary_cross_entropy_with_logits(out["logits"],
                                                           labels)
    loss.backward()
    assert any(p.grad is not None for p in m.extractor.parameters())
    assert all(p.grad is not None for p in
               m.dual_attention.channel_attention.parameters())
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad],
                           lr=1e-3)
    opt.step()

    # ---- frozen mode: temp-cache write -> read (bitwise) -> head forward --
    m.eval()
    m.extractor.freeze_trunk(True)
    emb_ref = extract_embeddings(m.extractor, sub)
    with tempfile.TemporaryDirectory() as td:
        write_embedding_cache(td, m.extractor, "train", sub, seed=SEED)
        emb2, rows, _ = load_embedding_cache(td, m.extractor, "train",
                                             seed=SEED)
    assert np.array_equal(emb_ref, emb2), "cache round-trip not bitwise"
    assert len(rows) == emb2.shape[0]
    logits = []
    off = 0
    for b in sub:
        n = b.n_instances
        e = torch.from_numpy(emb2[off:off + n])
        off += n
        o = m.dual_attention.forward_stacked(e)
        logits.append(m.classifier(o["z"].unsqueeze(0)).item())
    assert len(logits) == 8 and all(np.isfinite(v) for v in logits)
    # smoke values are descriptive only — recorded, never claimed
    print(f"      [smoke] 8 bags, sizes {sorted(sizes)}, instances "
          f"{sum(sizes)}, joint loss {float(loss):.4f}; frozen-path logits "
          f"finite: OK (temp cache deleted)")


# ------------------------------------------------------------------ run
def main() -> int:
    results = [(w._label, w()) for w in RUNNERS]
    npass = sum(1 for _, ok in results if ok)
    print(f"\n{npass}/{len(results)} checks passed")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
