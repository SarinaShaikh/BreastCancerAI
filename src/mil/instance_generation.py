"""Phase 7: thin instance-generation adapter over the Phase 5.5 loader.

Owner decision D0 (Phase 7 planning, 2026-09-19): this module is a THIN
ADAPTER over the authoritative ``src.mil.bag_dataset.UltrasoundBagDataset``
— it does NOT define bags, does NOT discover groups, does NOT reconstruct
``source_group_id``, does NOT create patient/study/lesion identifiers, does
NOT perform duplicate grouping, and does NOT alter labels or split
membership. Every bag/instance/label/split fact comes from the Phase 3/4/5.5
artifacts via the loader; this module only reshapes what the loader already
provides into tensors ready for the Phase 7 feature extractor:

    BagRecord
        ↓ (UltrasoundBagDataset.load_bag_images — Phase 5 exact decode)
    ordered instance tensors (n_i, 1, 224, 224) float32
        + per-instance provenance (image_path / md5 / bag_id / group / label)

Deterministic ordering is inherited from the loader (bags ascending
``bag_id``; instances ascending ``image_path`` — identical to the frozen
split-manifest row order and Phase 3 ``instance_list``). No hidden
normalization, no augmentation, no re-grouping, no test-split access: the
split is chosen explicitly by the caller and only ``train``/``val`` are
legitimate Phase 7 inputs (the loader can construct ``test``, but Phase 7
code must not ask it to — see the leakage note in the Phase 7 report).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

import importlib.util as _ilu


def _load_phase55_loader():
    """Import-by-path the authoritative Phase 5.5 loader (project convention).

    The module is registered in ``sys.modules`` BEFORE exec_module — the
    dataclasses inside ``bag_dataset.py`` resolve their own module during
    class creation, so registration must precede execution.
    """
    import sys
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    spec = _ilu.spec_from_file_location(
        "phase7_bag_dataset", os.path.join(root, "src", "mil", "bag_dataset.py"))
    mod = _ilu.module_from_spec(spec)
    sys.modules["phase7_bag_dataset"] = mod
    spec.loader.exec_module(mod)
    return mod


_bd = _load_phase55_loader()

# Splits Phase 7 infrastructure may touch (the frozen test split is OFF
# LIMITS for Phase 7: no test embeddings, no test forward passes).
PHASE7_SPLITS = ("train", "val")


@dataclass(frozen=True)
class InstanceBatch:
    """One bag rendered as extractor-ready tensors + provenance.

    images     : float32 (n_i, 1, 224, 224) — Phase 5 exact decode, channels
                 added for the Conv2d trunk; NO normalization is applied here
                 (the Phase 5 cache already stores normalized values).
    labels     : numeric bag label (benign=0, malignant=1) from the loader.
    bag_id / source_group_id / label : verbatim loader metadata.
    provenance : tuple of per-instance dicts (image_path, processed_relpath,
                 md5, bag_id, source_group_id, label, split, source_key,
                 source_key_status) — every field the Phase 5.5
                 InstanceRecord carries, so each tensor row stays traceable.
    """
    bag_id: str
    source_group_id: str
    label: str
    numeric_label: int
    images: torch.Tensor
    provenance: Tuple[dict, ...]

    @property
    def n_instances(self) -> int:
        return int(self.images.shape[0])


def load_bag_batch(bag, dataset) -> InstanceBatch:
    """Render ONE loader BagRecord as an InstanceBatch (thin adapter)."""
    arr = dataset.load_bag_images(bag)              # (n_i, 224, 224) float32
    images = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(1)
    prov = tuple({
        "image_path": r.image_path,
        "processed_relpath": r.processed_relpath,
        "md5": r.md5,
        "bag_id": r.bag_id,
        "source_group_id": r.source_group_id,
        "label": r.label,
        "split": r.split,
        "source_key": r.source_key,
        "source_key_status": r.source_key_status,
    } for r in bag.instances)
    return InstanceBatch(
        bag_id=bag.bag_id,
        source_group_id=bag.source_group_id,
        label=bag.label,
        numeric_label=int(bag.numeric_label),
        images=images,
        provenance=prov,
    )


def open_split(split: str, root: Optional[str] = None):
    """Build the authoritative loader for ``split`` WITHOUT decoding pixels
    (construction is manifest/validation only). Pair with
    ``deterministic_sample`` + ``load_bag_batch`` to load only the bags a
    test/run actually needs — full-split materialization goes through
    ``iter_split_batches``. ``test`` is rejected (Phase 7 leakage rule)."""
    if split not in PHASE7_SPLITS:
        raise ValueError(
            f"Phase 7 may only load {PHASE7_SPLITS}; got {split!r} "
            f"(the frozen test split is out of scope for Phase 7)")
    return _bd.UltrasoundBagDataset(split, root=root)


def load_bags(dataset, bag_ids: Sequence[str]) -> List[InstanceBatch]:
    """Load ONLY the named bags (in loader order) as InstanceBatches."""
    wanted = set(bag_ids)
    unknown = wanted - {b.bag_id for b in dataset.bags}
    if unknown:
        raise KeyError(f"bag_ids not in split {dataset.split}: "
                       f"{sorted(unknown)[:3]}")
    return [load_bag_batch(b, dataset) for b in dataset.bags
            if b.bag_id in wanted]


def iter_split_batches(split: str, root: Optional[str] = None,
                       bag_ids: Optional[Sequence[str]] = None
                       ) -> "tuple[List[InstanceBatch], object]":
    """Build the authoritative loader for ``split`` and yield all its bags
    (or the requested subset, by bag_id) in the loader's deterministic order.

    ``test`` is rejected: Phase 7 infrastructure never touches the frozen
    test split (owner leakage rule). The returned dataset object is also
    handed back so callers can reuse its summary()/ordering guarantees
    without constructing a second loader.
    """
    if split not in PHASE7_SPLITS:
        raise ValueError(
            f"Phase 7 may only load {PHASE7_SPLITS}; got {split!r} "
            f"(the frozen test split is out of scope for Phase 7)")
    ds = _bd.UltrasoundBagDataset(split, root=root)
    if bag_ids is None:
        batches = [load_bag_batch(b, ds) for b in ds.bags]
    else:
        wanted = set(bag_ids)
        unknown = wanted - {b.bag_id for b in ds.bags}
        if unknown:
            raise KeyError(f"bag_ids not in split {split}: {sorted(unknown)[:3]}")
        batches = [load_bag_batch(b, ds) for b in ds.bags if b.bag_id in wanted]
    return batches, ds


def deterministic_sample(dataset, n_bags: int, seed: int,
                         rng=None) -> List[str]:
    """Seeded sample of bag_ids from a split loader (smoke-test helper).

    Uses a dedicated ``random.Random(seed)`` stream (never the global RNG),
    so sampling is reproducible and does not disturb model-init randomness.
    Prefers including the split's smallest bag(s) when the deterministic
    draw leaves room, so edge sizes (e.g. the train split's single
    14-instance bag) are exercised without special-casing.
    """
    import random
    ids = [b.bag_id for b in dataset.bags]          # loader order (sorted)
    if n_bags > len(ids):
        raise ValueError(f"requested {n_bags} bags, split has {len(ids)}")
    r = rng if rng is not None else random.Random(seed)
    picked = set(r.sample(ids, n_bags))
    # Deterministically swap in the smallest bag(s) if not already picked.
    smallest = min(dataset.bags, key=lambda b: (b.instance_count, b.bag_id))
    if smallest.bag_id not in picked:
        dropped = sorted(picked)[0]
        picked.discard(dropped)
        picked.add(smallest.bag_id)
    return sorted(picked)
