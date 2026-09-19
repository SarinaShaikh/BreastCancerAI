"""Phase 7: fresh CNN feature extractor + leakage-safe embedding cache.

Owner decisions D1 + D3 (Phase 7 planning, 2026-09-19):

D1 — the extractor is FRESHLY INITIALIZED (seed 20260918) with the same
trunk architecture as the approved B1/B4 baselines: 4 × Conv3×3-BN-ReLU-
MaxPool then Global Average Pooling → 128-d instance embedding. Channels
are (24, 48, 96, 128) — imported from ``simple_cnn.CHANNELS`` so this
module is architecturally IDENTICAL to the implemented Phase 6 trunk. (The
16/32/64/128 channel figures from the original draft were widened during
Phase 6 to satisfy the owner's 150k–250k parameter floor — measured
97,761 params at 16/32/64/128; see simple_cnn.py. Embedding dim stays 128.)
It does NOT load B1/B4 checkpoints, ImageNet weights,
or any pretrained features; it does NOT modify ``model/baselines/`` or any
Phase 6 checkpoint. The frozen Phase 6 baselines stay frozen: this module
redeclares the trunk (importing only the hyperparameter constants) so the
Phase 7 model is an independent object.

D3 — a leakage-safe embedding cache stores 128-d float32 embeddings of the
UNAUGMENTED Phase 5 images, produced ONLY by an extractor in evaluation
mode with ``requires_grad`` disabled (frozen mode). The cache writer
REJECTS trainable extractors (a trainable trunk must be trained from
pixels; caching would silently freeze it). Caches are per-split, keyed by
an exact content hash of a manifest that records extractor/weights/seed/
preprocessing/split identities — invalidation is hash-based, never
mtime-based. Augmented images are never cached (augmentation is a
train-time-on-the-fly pixel transform; it has no deterministic embedding).
Test-split embeddings are never built in Phase 7.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util as _ilu
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from model.baselines.simple_cnn import CHANNELS, INPUT_CHANNELS

EMBED_DIM = CHANNELS[-1]          # 128
SEED = 20260918


def _load_module(name: str, relpath: str):
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    spec = _ilu.spec_from_file_location(name, os.path.join(root, relpath))
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_tb = _load_module("phase7_train_baseline", "src/training/train_baseline.py")
PREPROCESSING_VERSION = _tb.PREPROCESSING_VERSION     # "1.0.0"
PIPELINE_VERSION = _tb.PIPELINE_VERSION               # "1.0.1"
TEST_SPLIT_SHA256 = _tb.TEST_SPLIT_SHA256
CACHE_VERSION = "1.0.0"


def seed_extractor(seed: int = SEED) -> None:
    """Seed every RNG stream before extractor construction (D1)."""
    import random
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


class InstanceFeatureExtractor(nn.Module):
    """Fresh B1/B4-style trunk: (N, 1, 224, 224) -> (N, 128) embeddings.

    Same block topology and hyperparameters as the approved Phase 6
    baselines (constants imported from ``simple_cnn``), independently
    initialized. The classifier is intentionally absent: embedding and
    classification are separate modules (owner instruction §4).
    """

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
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 1, 224, 224) -> GAP -> (N, 128)."""
        if x.ndim != 4 or x.size(1) != 1:
            raise ValueError(
                f"expected (N, 1, 224, 224), got {tuple(x.shape)}")
        z = self.features(x)
        return z.mean(dim=(2, 3))

    # ------------------------------------------------------------ freeze
    def freeze_trunk(self, freeze: bool = True) -> "InstanceFeatureExtractor":
        """D1/D2: freeze=True -> requires_grad=False on every parameter
        (no trunk gradients; used with the cache path). freeze=False keeps
        the trunk trainable (joint path)."""
        for p in self.parameters():
            p.requires_grad = not freeze
        self.eval() if freeze else self.train()
        return self

    @property
    def is_frozen(self) -> bool:
        return all(not p.requires_grad for p in self.parameters())

    def weights_hash(self) -> str:
        """Deterministic content hash of the parameter state (and the BN
        buffers, which encode normalization state) — part of the cache key."""
        h = hashlib.sha256()
        with torch.no_grad():
            for k in sorted(self.state_dict().keys()):
                v = self.state_dict()[k].detach().cpu().numpy()
                h.update(k.encode("utf-8"))
                h.update(np.ascontiguousarray(v).tobytes())
        return h.hexdigest()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------------------------ cache
def _split_manifest_sha(root: str, split: str) -> str:
    p = os.path.join(root, "data", "manifests", f"{split}_split.csv")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_cache_manifest(extractor: InstanceFeatureExtractor, split: str,
                         seed: int = SEED) -> dict:
    """Exact identity manifest for one split's embedding cache (D3)."""
    if split == "test":
        raise ValueError("Phase 7 never builds test-split embeddings")
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    return {
        "cache_version": CACHE_VERSION,
        "component": "phase7_instance_embeddings",
        "architecture": ("Conv3x3-BN-ReLU-MaxPool x4, channels "
                         "1->24->48->96->128, GAP"),
        "embed_dim": EMBED_DIM,
        "weights_sha256": extractor.weights_hash(),
        "extractor_trainable_at_cache_time": False,
        "seed": int(seed),
        "preprocessing_version": PREPROCESSING_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "split": split,
        "split_manifest_sha256": _split_manifest_sha(root, split),
        "frozen_test_sha256": TEST_SPLIT_SHA256,
    }


def _manifest_key(manifest: dict) -> str:
    """Exact content hash of the manifest (canonical JSON) — the cache key.

    No modification times, no filesystem state: any identity change in the
    manifest changes the key and forces a rebuild.
    """
    blob = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def extract_embeddings(extractor: InstanceFeatureExtractor,
                       batches: Sequence,
                       batch_size: int = 64) -> np.ndarray:
    """Run extractor over InstanceBatch list -> float32 (total_n, 128).

    Caller is responsible for extractor mode: the cache writer refuses
    anything not in frozen/eval state; this function itself is mode-agnostic
    (trainable mode is legitimate for the joint path, which never caches).
    """
    outs: List[np.ndarray] = []
    for b in batches:
        x = b.images
        for s in range(0, x.shape[0], batch_size):
            with torch.no_grad() if extractor.is_frozen else _null_ctx():
                outs.append(extractor(x[s:s + batch_size]).cpu().numpy())
    if not outs:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    return np.concatenate(outs, axis=0).astype(np.float32, copy=False)


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _provenance_rows(batches: Sequence) -> List[dict]:
    rows: List[dict] = []
    for b in batches:
        for i, prov in enumerate(b.provenance):
            rows.append({"row_index": len(rows),
                         "bag_id": b.bag_id,
                         "source_group_id": b.source_group_id,
                         "label": b.label,
                         "numeric_label": b.numeric_label,
                         "instance_index_in_bag": i,
                         **prov})
    return rows


def write_embedding_cache(dest_root: str, extractor: InstanceFeatureExtractor,
                          split: str, batches: Sequence,
                          seed: int = SEED) -> str:
    """Write one split's embedding cache. REFUSES trainable extractors.

    Layout: <dest_root>/<cache_key>/ with embeddings.npy (float32 matrix,
    rows in loader instance order), provenance.csv (one row per instance,
    full Phase 5 provenance), manifest.json (identity) and manifest_key.txt
    (the exact hash). Returns the cache key directory.
    """
    if not extractor.is_frozen:
        raise RuntimeError(
            "embedding cache refused: the extractor is trainable. "
            "Caching while the trunk is trainable would silently freeze it; "
            "the joint path must train from pixels (owner rule §10).")
    extractor.eval()
    manifest = build_cache_manifest(extractor, split, seed=seed)
    key = _manifest_key(manifest)
    out_dir = os.path.join(dest_root, key)
    os.makedirs(out_dir, exist_ok=True)

    emb = extract_embeddings(extractor, batches)
    if emb.shape[0] != sum(b.n_instances for b in batches):
        raise RuntimeError("embedding count != instance count")
    np.save(os.path.join(out_dir, "embeddings.npy"), emb)
    with open(os.path.join(out_dir, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(out_dir, "manifest_key.txt"), "w",
              encoding="utf-8") as f:
        f.write(key + "\n")
    with open(os.path.join(out_dir, "provenance.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(_provenance_rows(batches)[0]
                                             .keys()))
        w.writeheader()
        w.writerows(_provenance_rows(batches))
    return key


def load_embedding_cache(cache_root: str, extractor: InstanceFeatureExtractor,
                         split: str, seed: int = SEED
                         ) -> Tuple[np.ndarray, List[dict], str]:
    """Load a cache iff its manifest exactly matches the caller's identity.

    Verifies: manifest.json content == expected manifest (bit-exact key),
    embeddings.npy shape == (n_instances, 128), dtype float32, and no
    NaN/Inf. Returns (embeddings, provenance_rows, key).
    """
    manifest = build_cache_manifest(extractor, split, seed=seed)
    key = _manifest_key(manifest)
    d = os.path.join(cache_root, key)
    if not os.path.isfile(os.path.join(d, "manifest.json")):
        raise FileNotFoundError(
            f"embedding cache miss for key {key[:12]}… under {cache_root}")
    stored = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
    if stored != manifest:
        raise RuntimeError("embedding cache manifest drift — rebuild required")
    emb = np.load(os.path.join(d, "embeddings.npy"))
    if emb.dtype != np.float32 or emb.ndim != 2 or emb.shape[1] != EMBED_DIM:
        raise RuntimeError(f"embedding cache shape/dtype violation: "
                           f"{emb.shape} {emb.dtype}")
    rows = list(csv.DictReader(open(os.path.join(d, "provenance.csv"),
                                    encoding="utf-8")))
    if len(rows) != emb.shape[0]:
        raise RuntimeError("provenance rows != embedding rows")
    if not np.isfinite(emb).all():
        raise RuntimeError("embedding cache contains NaN/Inf")
    return emb, rows, key
