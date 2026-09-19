"""Phase 7 validation suite (T1–T15) for the MIL pipeline deliverables.

Runs standalone (``python tests/test_phase7_mil_pipeline.py``) or under
pytest. Covers the owner instruction's minimum test list:

  T1  variable bag sizes {1,2,3,14,16,21,53} without shape errors
  T2  exactly one binary bag logit per bag
  T3  one-instance bag attention weight == 1.0 (robustness case; note that
      NO single-instance bag exists in the actual dataset)
  T4  attention weights sum to 1 within each bag (tol 1e-6)
  T5  padded/masked path == dynamic path (numerical equivalence)
  T6  masked instances receive exactly zero attention
  T7  no NaN/Inf anywhere
  T8  embedding dimensionality exactly 128
  T9  deterministic inference under fixed seed (bitwise)
  T10 frozen trunk: no trunk grads, attention/head grads present;
      joint mode: gradients reach the trunk
  T11 embedding cache write->load->compare reproduces bitwise
  T12 cache write REJECTS a trainable extractor
  T13 every cached row has valid provenance (loader-traceable)
  T14 end-to-end smoke on a deterministic training-only sample (the
      roadmap exit criterion), incl. one joint backward step and the
      frozen cache round-trip
  T15 boundary scan: no dual/gated/transformer/cross-bag attention, no
      top-k pooling, no max-pooling AGGREGATION — while explicitly
      allowing the CNN trunk's ordinary spatial MaxPool2d

Scope guard: single-attention ABMIL infrastructure only. No Dual Attention
MIL, no MRI/cross-modal code, no test-split access, no pretrained weights,
no committed checkpoints, no full training experiment.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.baselines.simple_cnn import CHANNELS          # noqa: E402
from src.mil.aggregation import (                         # noqa: E402
    AttentionMILPipeline, BagClassifier, EMBED_DIM,
    SingleAttentionAggregator)
from src.mil.feature_extractor import (                   # noqa: E402
    InstanceFeatureExtractor, build_cache_manifest, extract_embeddings,
    load_embedding_cache, seed_extractor, write_embedding_cache)
from src.mil.instance_generation import (                 # noqa: E402
    deterministic_sample, load_bags, open_split)

TEST_SPLIT_SHA256 = "959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74"
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


def _fresh_extractor(seed: int = SEED) -> InstanceFeatureExtractor:
    seed_extractor(seed)
    return InstanceFeatureExtractor()


def _pipeline(seed: int = SEED) -> AttentionMILPipeline:
    seed_extractor(seed)
    return AttentionMILPipeline(InstanceFeatureExtractor())


def _make(n: int, seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 1, 224, 224, generator=g)


def _pads(sizes, seed: int = 7, channels: int = 1, h: int = 28, w: int = 28):
    """Padded batch + mask + per-bag stacked tensors (small spatial size for
    CPU speed; architecture is size-agnostic thanks to GAP)."""
    n_max = max(sizes)
    g = torch.Generator().manual_seed(seed)
    imgs = torch.zeros(len(sizes), n_max, channels, h, w)
    mask = torch.zeros(len(sizes), n_max, dtype=torch.bool)
    stacked = []
    for b, n in enumerate(sizes):
        x = torch.randn(n, channels, h, w, generator=g)
        imgs[b, :n] = x
        mask[b, :n] = True
        stacked.append(x)
    return imgs, mask, stacked


# ------------------------------------------------------------------ T1
@check("T1 variable bag sizes {1,2,3,14,16,21,53} processed without shape errors")
def t01():
    pipe = _pipeline()
    for n in (1, 2, 3, 14, 16, 21, 53):
        out = pipe.forward_bag(_make(n))
        assert out["logit"].shape == (1, 1), (n, out["logit"].shape)
        assert out["embeddings"].shape == (n, 128)
    sizes = (1, 2, 3, 14, 16, 21, 53)
    imgs, mask, _ = _pads(sizes)
    out = pipe.forward_padded(imgs, mask)
    assert out["logits"].shape == (len(sizes),)


# ------------------------------------------------------------------ T2
@check("T2 exactly one binary bag logit per bag (stacked + padded)")
def t02():
    pipe = _pipeline()
    sizes = (14, 16, 21, 53, 1, 2, 3)
    o = pipe.forward_list([_make(n) for n in sizes])
    assert o["logits"].shape == (len(sizes),), o["logits"].shape
    assert o["logits"].ndim == 1
    imgs, mask, _ = _pads(sizes)
    op = pipe.forward_padded(imgs, mask)
    assert op["logits"].shape == (len(sizes),)


# ------------------------------------------------------------------ T3
@check("T3 one-instance bag attention weight == exactly 1.0 (robustness)")
def t03():
    agg = SingleAttentionAggregator()
    h = torch.randn(1, 128)
    a, z = agg.forward_stacked(h)
    assert float(a[0]) == 1.0 and a.shape == (1,)
    # z must equal the single instance embedding when the weight is 1
    assert torch.allclose(z, h[0], atol=1e-6)


# ------------------------------------------------------------------ T4
@check("T4 attention weights sum to 1 within each bag (tol 1e-6), never across bags")
def t04():
    pipe = _pipeline()
    sizes = (14, 16, 21, 53, 3)
    o = pipe.forward_list([_make(n) for n in sizes])
    for n, a in zip(sizes, o["attention"]):
        assert abs(float(a.sum()) - 1.0) < 1e-6, (n, float(a.sum()))
        assert (a >= 0).all() and (a <= 1).all()
    imgs, mask, _ = _pads(sizes)
    op = pipe.forward_padded(imgs, mask)
    sums = op["attention"].sum(dim=1)
    assert torch.allclose(sums, torch.ones(len(sizes)), atol=1e-6)


# ------------------------------------------------------------------ T5
@check("T5 padded/masked path == dynamic path (numerical equivalence)")
def t05():
    pipe = _pipeline().eval()
    sizes = (14, 16, 21)
    imgs, mask, stacked = _pads(sizes)
    with torch.no_grad():
        op = pipe.forward_padded(imgs, mask)
        ol = pipe.forward_list(stacked)
    assert torch.allclose(op["logits"], ol["logits"], atol=1e-5), \
        (op["logits"], ol["logits"])
    for b in range(len(sizes)):
        assert torch.allclose(op["bag_reprs"][b], ol["bag_reprs"][b],
                              atol=1e-5)


# ------------------------------------------------------------------ T6
@check("T6 masked instances receive exactly zero attention")
def t06():
    agg = SingleAttentionAggregator()
    imgs, mask, _ = _pads([5, 3], h=4, w=4)
    emb = torch.randn(2, 5, 128)
    a, _ = agg.forward_padded(emb, mask)
    assert float(a[0, 5:].abs().sum()) == 0.0
    assert float(a[1, 3:].abs().sum()) == 0.0
    assert float(a[0, :5].abs().sum()) > 0 and float(a[1, :3].abs().sum()) > 0


# ------------------------------------------------------------------ T7
@check("T7 no NaN/Inf in embeddings, attention, representations, logits")
def t07():
    pipe = _pipeline().eval()
    sizes = (1, 3, 14, 53)
    with torch.no_grad():
        o = pipe.forward_list([_make(n) for n in sizes])
        imgs, mask, _ = _pads(sizes)
        op = pipe.forward_padded(imgs, mask)
    for t in (o["logits"], o["bag_reprs"]):
        assert torch.isfinite(t).all()
    for t in (op["logits"], op["bag_reprs"], op["embeddings"],
              op["attention"]):
        assert torch.isfinite(t).all()


# ------------------------------------------------------------------ T8
@check("T8 embedding dimensionality is exactly 128")
def t08():
    ex = _fresh_extractor()
    assert EMBED_DIM == 128 == CHANNELS[-1]
    assert ex(_make(4)).shape == (4, 128)


# ------------------------------------------------------------------ T9
@check("T9 deterministic inference under fixed seed (bitwise)")
def t09():
    outs = []
    for _ in range(2):
        pipe = _pipeline().eval()
        with torch.no_grad():
            outs.append(pipe.forward_bag(_make(21))["logit"].clone())
    assert torch.equal(outs[0], outs[1]), "bitwise determinism violated"


# ----------------------------------------------------------------- T10
@check("T10 frozen trunk: no trunk grads; joint mode: trunk grads exist")
def t10():
    # frozen mode
    pipe = _pipeline()
    pipe.extractor.freeze_trunk(True)
    assert pipe.extractor.is_frozen
    out = pipe.forward_bag(_make(6))
    loss = out["logit"].sum()
    loss.backward()
    assert all(p.grad is None for p in pipe.extractor.parameters()), \
        "frozen trunk must have no gradients"
    assert all(p.grad is not None for p in pipe.attention.parameters()), \
        "attention must receive gradients in frozen mode"
    assert all(p.grad is not None for p in pipe.classifier.parameters())

    # joint mode
    pipe2 = _pipeline()
    pipe2.extractor.freeze_trunk(False)
    assert not pipe2.extractor.is_frozen
    out2 = pipe2.forward_bag(_make(6))
    out2["logit"].sum().backward()
    trunk_grads = [p.grad for p in pipe2.extractor.parameters()]
    assert any(g is not None for g in trunk_grads), \
        "joint mode must propagate gradients into the trunk"
    assert all(p.grad is not None for p in pipe2.attention.parameters())


# ----------------------------------------------------------------- T11
@check("T11 embedding cache write->load->compare reproduces bitwise")
def t11():
    ex = _fresh_extractor().freeze_trunk(True)
    ds = open_split("train")
    sample_ids = deterministic_sample(ds, 3, SEED)
    sub = load_bags(ds, sample_ids)
    emb_ref = extract_embeddings(ex, sub)
    with tempfile.TemporaryDirectory() as td:
        key = write_embedding_cache(td, ex, "train", sub, seed=SEED)
        emb2, rows, key2 = load_embedding_cache(td, ex, "train", seed=SEED)
        assert key == key2
        assert np.array_equal(emb_ref, emb2), "cache round-trip not bitwise"
        assert emb2.dtype == np.float32 and emb2.shape[1] == 128
        assert len(rows) == emb2.shape[0]


# ----------------------------------------------------------------- T12
@check("T12 cache write REJECTS a trainable extractor")
def t12():
    ex = _fresh_extractor().freeze_trunk(False)
    assert not ex.is_frozen
    ds = open_split("train")
    sub = load_bags(ds, [ds.bags[0].bag_id])
    try:
        write_embedding_cache(tempfile.gettempdir(), ex, "train", sub,
                              seed=SEED)
        raise AssertionError("trainable extractor must be refused by the cache")
    except RuntimeError:
        pass


# ----------------------------------------------------------------- T13
@check("T13 every cached embedding row has valid provenance")
def t13():
    ex = _fresh_extractor().freeze_trunk(True)
    ds = open_split("train")
    sample_ids = deterministic_sample(ds, 3, SEED)
    sub = load_bags(ds, sample_ids)
    with tempfile.TemporaryDirectory() as td:
        write_embedding_cache(td, ex, "train", sub, seed=SEED)
        _, rows, _ = load_embedding_cache(td, ex, "train", seed=SEED)
    loader_index = {}
    for b in sub:
        for r in b.provenance:
            loader_index[(b.bag_id, r["image_path"])] = r["md5"]
    for row in rows:
        k = (row["bag_id"], row["image_path"])
        assert k in loader_index, f"untraceable cache row {k}"
        assert loader_index[k] == row["md5"], f"md5 drift for {k}"
        assert row["source_key_status"] == \
            "inferred_from_filename_not_verified_identifier"
        assert int(row["row_index"]) == rows.index(row)


# ----------------------------------------------------------------- T14
@check("T14 end-to-end smoke: train-only sample, joint backward + frozen cache round-trip")
def t14():
    import torch.nn as _nn
    # ---- deterministic 8-bag TRAIN-ONLY sample (includes smallest bag) --
    ds = open_split("train")
    sample_ids = deterministic_sample(ds, 8, SEED)
    sub = load_bags(ds, sample_ids)
    sizes = [b.n_instances for b in sub]
    assert len(sub) == 8, f"smoke sample must be exactly 8 bags, got {len(sub)}"
    assert 14 in sizes, \
        f"deterministic sample must include the 14-instance train bag; got {sorted(sizes)}"

    # ---- joint mode: forward -> BCEWithLogitsLoss -> backward -> step ---
    seed_extractor(SEED)
    pipe = AttentionMILPipeline(InstanceFeatureExtractor())
    pipe.extractor.freeze_trunk(False)
    pipe.train()
    imgs = [b.images for b in sub]
    labels = torch.tensor([b.numeric_label for b in sub], dtype=torch.float32)
    out = pipe.forward_list(imgs)
    loss = _nn.functional.binary_cross_entropy_with_logits(
        out["logits"], labels)
    loss.backward()
    trunk_had_grads = any(p.grad is not None
                          for p in pipe.extractor.parameters())
    assert trunk_had_grads
    opt = torch.optim.Adam(
        [p for p in pipe.parameters() if p.requires_grad], lr=1e-3)
    opt.step()

    # ---- frozen mode: cache write -> read -> attention/head forward -----
    pipe.eval()
    pipe.extractor.freeze_trunk(True)
    emb_ref = extract_embeddings(pipe.extractor, sub)
    with tempfile.TemporaryDirectory() as td:
        write_embedding_cache(td, pipe.extractor, "train", sub, seed=SEED)
        emb2, rows, _ = load_embedding_cache(td, pipe.extractor, "train",
                                             seed=SEED)
    assert np.array_equal(emb_ref, emb2)
    att = pipe.attention
    logits = []
    off = 0
    for b in sub:
        n = b.n_instances
        e = torch.from_numpy(emb2[off:off + n])
        off += n
        _, z = att.forward_stacked(e)
        logits.append(pipe.classifier(z.unsqueeze(0)).item())
    assert len(logits) == 8 and all(np.isfinite(v) for v in logits)
    # smoke metrics are descriptive only — recorded, never claimed
    print(f"      [smoke] 8 bags, sizes {sorted(sizes)}, "
          f"instances {sum(sizes)}, joint loss {float(loss):.4f}; "
          f"frozen-path logits finite: OK")


# ----------------------------------------------------------------- T15
@check("T15 boundary scan: no dual/gated/transformer/cross-bag attention, "
       "no top-k, no max-pool AGGREGATION (CNN spatial MaxPool2d allowed)")
def t15():
    targets = ["src/mil/instance_generation.py",
               "src/mil/feature_extractor.py",
               "src/mil/aggregation.py"]
    # AST scan: banned identifiers (aggregation-level max pooling would
    # appear as a Name/Attribute like 'amax'/'max' used over instance dim;
    # CNN MaxPool2d is a CONSTRUCTOR call and therefore not a Name/Attr hit).
    banned_names = {"dual_attention", "DualAttention", "gated_attention",
                    "GatedAttention", "MultiheadAttention", "Transformer",
                    "transformer", "cross_bag", "CrossBag", "topk",
                    "top_k", "amax"}
    for rel in targets:
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id not in banned_names, (rel, node.id)
            if isinstance(node, ast.Attribute):
                assert node.attr not in banned_names, (rel, node.attr)
    # textual scan: no dual/gated/transformer/cross-bag mechanisms at all;
    # 'max pooling' as MIL aggregation must not appear in code constructs
    # (docstring mentions are the documented boundary statement itself).
    for rel in targets:
        src = (REPO / rel).read_text(encoding="utf-8")
        for token in ("DualAttention", "GatedAttention",
                      "MultiheadAttention", "torch.topk", ".max(dim",
                      ".amax(", "nn.MaxPool1d"):
            assert token not in src, (rel, token)
    # the only MaxPool in Phase 7 code must be the CNN's spatial MaxPool2d
    src_agg = (REPO / "src/mil/aggregation.py").read_text(encoding="utf-8")
    assert "nn.MaxPool" not in src_agg, \
        "aggregation module must not contain pooling layers"
    for rel in ("src/mil/feature_extractor.py",):
        src = (REPO / rel).read_text(encoding="utf-8")
        assert "nn.MaxPool2d(2, 2)" in src, "CNN spatial MaxPool2d expected"
    # no MRI / cross-modal identifiers anywhere in Phase 7 modules
    for rel in targets:
        src = (REPO / rel).read_text(encoding="utf-8")
        for token in ("MRI", "mri", "cross_modal", "crossmodal"):
            assert token not in src, (rel, token)
    # no test-split access helper defaults to test
    ig = (REPO / "src/mil/instance_generation.py").read_text(encoding="utf-8")
    assert 'PHASE7_SPLITS = ("train", "val")' in ig


# ------------------------------------------------------------------ run
def main() -> int:
    results = [(w._label, w()) for w in RUNNERS]
    npass = sum(1 for _, ok in results if ok)
    print(f"\n{npass}/{len(results)} checks passed")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
