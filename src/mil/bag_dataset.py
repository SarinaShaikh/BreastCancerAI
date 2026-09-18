"""Phase 5.5: ultrasound bag dataset loader & dataset interface.

Roadmap context: NEW roadmap section (added 2026-09-18 by owner decision,
between Phase 5 preprocessing and roadmap-Phase 6 "Baseline Deep Learning
Models"). Decision D-1 maps MIL infrastructure to ``src/mil/`` (this module
sits beside the Phase 3 ``bag_definition.py`` whose artifacts it consumes).

Purpose — a clean, deterministic, testable INPUT/DATA INTERFACE through which
later model phases (roadmap Phases 6-8) request

    split -> bags -> instances -> labels -> metadata

WITHOUT redefining bag membership or labels. This module implements NO model,
NO MIL, NO attention, NO training, NO evaluation, NO feature extraction
(scope guard enforced by tests/test_phase6_dataset_loader.py).

Authoritative definitions (consumed, never changed):
    BAG      = one Phase 2 ``source_group_id`` (source-image/augmentation
               family) per Phase 3. NOT a verified patient/study/lesion ID.
    INSTANCE = one image file of that source group.
    LABEL    = the group's directory-encoded class (``benign``/``malignant``).
    SPLIT    = the frozen Phase 4 assignment (``data/manifests/{split}_split.csv``).

Data sources (all frozen/committed):
    data/manifests/bag_manifest.csv        Phase 3 bags (496) + instance lists
    data/manifests/{train,val,test}_split.csv  Phase 4 instance-level split
    data/processed/{split}/provenance.csv  Phase 5 processed-cache mapping
    data/processed/{split}/*.png           Phase 5 16-bit PNG cache (gitignored;
                                           regenerable via
                                           ``python -m src.preprocessing.pipeline``)
    configs/preprocessing_config.yaml      Phase 5 contract (size, decode constants)

Ordering rule (deterministic, documented):
    bags      -> ascending ``bag_id`` (lowercase hex string sort)
    instances -> ascending ``image_path`` (ASCII lexicographic on the
                 relative raw path), which equals both the frozen split
                 manifests' row order and the Phase 3 ``instance_list`` order.
    No filesystem enumeration order and no Python hash randomization is used.

Memory safety: image pixel data is loaded LAZILY, one image (or one bag) at a
time. Construction performs only manifest/CSV parsing plus file-existence
checks; it never decodes image pixels and never loads all 9,016 images.

Train/val/test behaviour: the loader NEVER augments, NEVER recomputes
normalization statistics, and NEVER applies split-specific transforms. The
training-time-only augmentation remains the Phase 5 ``pipeline.augment()``
API (which hard-raises for val/test). Validation/test construction is
byte-for-byte the same code path as train; only the input manifests differ.

Fresh-clone behaviour: if the Phase 5 PNG cache is absent, construction
fails with an actionable error telling the user to regenerate it locally
(no automatic regeneration, no downloads).
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Reuse the Phase 5 module for the frozen-contract helpers (freeze
# verification, config loading, decode constants). Import-by-path keeps this
# module importable both as ``src.mil.bag_dataset`` and standalone.
import importlib.util as _ilu


def _load_phase5_pipeline():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    spec = _ilu.spec_from_file_location(
        "phase5_pipeline", os.path.join(root, "src", "preprocessing", "pipeline.py"))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_p5 = _load_phase5_pipeline()

SPLITS = ("train", "val", "test")
EXPECTED_INSTANCE_COUNTS = {"train": 6327, "val": 1339, "test": 1350}
EXPECTED_BAG_COUNTS = {"train": 349, "val": 74, "test": 73}  # Phase 4 record
TOTAL_INSTANCES = 9016
VALID_LABELS = ("benign", "malignant")
# Numeric encoding for later model phases. The STRING label is authoritative;
# this mapping is a documented convention (benign=0, malignant=1), consistent
# with the directory-encoded class names verified in Phase 1.
LABEL_MAP = {"benign": 0, "malignant": 1}

TEST_MANIFEST_SHA256 = _p5.TEST_MANIFEST_SHA256  # single source of truth (Phase 5)


class DatasetError(RuntimeError):
    """Raised on any dataset-interface failure (missing/ambiguous/drifted data)."""


@dataclass(frozen=True)
class InstanceRecord:
    """One ultrasound image instance (one file) with full provenance."""
    image_path: str            # relative raw path, e.g. train/benign/benign (103).png
    processed_relpath: str     # relative processed path under data/processed/
    md5: str                   # Phase 1 content digest (chain to raw file)
    bag_id: str                # Phase 3 bag
    source_group_id: str       # Phase 2 source-image family
    label: str                 # directory-encoded class: benign | malignant
    split: str                 # frozen Phase 4 split
    source_key: str            # filename-derived key (NOT a verified identifier)
    source_key_status: str     # always inferred_from_filename_not_verified_identifier
    original_supplied_split: str  # the dataset's original train/val directory

    @property
    def numeric_label(self) -> int:
        return LABEL_MAP[self.label]


@dataclass(frozen=True)
class BagRecord:
    """One MIL bag (one Phase 2/3 source-image family) in one frozen split."""
    bag_id: str
    source_group_id: str
    label: str
    split: str
    source_key: str
    source_key_status: str
    instances: Tuple[InstanceRecord, ...]

    @property
    def instance_count(self) -> int:
        return len(self.instances)

    @property
    def numeric_label(self) -> int:
        return LABEL_MAP[self.label]

    @property
    def instance_paths(self) -> Tuple[str, ...]:
        return tuple(i.image_path for i in self.instances)


class UltrasoundBagDataset:
    """Deterministic bag-level dataset over one frozen Phase 4 split.

    Construction validates the frozen Phase 4 state and the Phase 5 cache
    mapping, then exposes bag/instance metadata and LAZY image loading that
    preserves the Phase 5 preprocessing contract exactly.
    """

    def __init__(self, split: str, root: Optional[str] = None,
                 processed_root: Optional[str] = None,
                 verify_cache_files: bool = True) -> None:
        if split not in SPLITS:
            raise DatasetError(
                f"unknown split {split!r}; expected one of {SPLITS}")
        self.split = split
        self.root = root or _p5.repo_root()
        self.processed_root = processed_root or os.path.join(
            self.root, "data", "processed")

        # --- frozen Phase 4 verification (fail fast, never repair) ---------
        digest = _p5.verify_freeze(self.root)
        self.frozen_test_sha256 = digest
        counts = _p5.verify_split_counts(self.root)
        if counts[split] != EXPECTED_INSTANCE_COUNTS[split]:
            raise DatasetError(
                f"split {split} count drift: {counts[split]} != "
                f"{EXPECTED_INSTANCE_COUNTS[split]}")

        # --- Phase 5 contract (config) --------------------------------------
        self.config = _p5.load_config(os.path.join(
            self.root, "configs", "preprocessing_config.yaml"))
        self.image_size = (int(self.config["image"]["size"]["height"]),
                           int(self.config["image"]["size"]["width"]))
        self.raw_root = _p5.raw_image_root(self.root)

        # --- load authoritative manifests -----------------------------------
        split_rows = self._read_csv(os.path.join(
            self.root, "data", "manifests", f"{split}_split.csv"))
        bag_rows = self._read_csv(os.path.join(
            self.root, "data", "manifests", "bag_manifest.csv"))
        prov_rows = self._read_csv(os.path.join(
            self.processed_root, split, "provenance.csv"))

        # Phase 5 provenance: image_path -> output_file (must be a bijection).
        prov_by_path: Dict[str, str] = {}
        for r in prov_rows:
            p = r["image_path"]
            if p in prov_by_path:
                raise DatasetError(
                    f"ambiguous processed representation: {p!r} appears "
                    f"multiple times in {split} provenance")
            prov_by_path[p] = r["output_file"]
        out_to_path: Dict[str, str] = {}
        for p, o in prov_by_path.items():
            if o in out_to_path:
                raise DatasetError(
                    f"processed output {o!r} maps to multiple instances "
                    f"({out_to_path[o]!r}, {p!r})")
            out_to_path[o] = p

        # Phase 4 split manifest for this split: instance-level authority.
        split_by_path: Dict[str, dict] = {r["image_path"]: r for r in split_rows}
        if len(split_by_path) != len(split_rows):
            raise DatasetError(f"duplicate image_path rows in {split}_split.csv")

        # Bags of this split = source groups present in this split's manifest.
        bag_ids_here = {r["bag_id"] for r in split_rows}

        # Build bag_id -> BagRecord using Phase 3 membership, intersected with
        # the frozen split manifest (the intersection is validated to be the
        # FULL bag for every bag — Phase 4 atomicity — below in _assemble).
        bag_meta: Dict[str, dict] = {}
        for r in bag_rows:
            bag_meta[r["bag_id"]] = r

        unknown = bag_ids_here - set(bag_meta)
        if unknown:
            raise DatasetError(
                f"{split} manifest references unknown bag ids: {sorted(unknown)[:3]}")

        # Assemble instances per bag (deterministic: sorted by image_path).
        per_bag: Dict[str, List[InstanceRecord]] = {b: [] for b in bag_ids_here}
        for path, m in split_by_path.items():
            if path not in prov_by_path:
                raise DatasetError(
                    f"instance {path!r} has no Phase 5 processed representation "
                    f"in {split} provenance.csv — regenerate the cache with "
                    f"`python -m src.preprocessing.pipeline`")
            o = prov_by_path[path]
            if m["md5"] != self._prov_md5(prov_rows, path):
                raise DatasetError(f"md5 drift between split manifest and "
                                   f"provenance for {path!r}")
            rec = InstanceRecord(
                image_path=path,
                # POSIX-style relative path (forward slashes) for cross-platform
                # deterministic metadata, matching every other manifest path
                # field; os.path.join(root, <posix path>) opens fine on Windows.
                processed_relpath=f"data/processed/{split}/{o}",
                md5=m["md5"],
                bag_id=m["bag_id"],
                source_group_id=m["source_group_id"],
                label=m["bag_label"],
                split=m["split"],
                source_key=m["source_key"],
                source_key_status=m["source_key_status"],
                original_supplied_split=m["original_supplied_split"],
            )
            per_bag[m["bag_id"]].append(rec)

        # Validate every bag referenced by the split manifest is complete
        # (Phase 4 atomicity: the split contains the WHOLE bag) and coherent.
        self._bags: List[BagRecord] = []
        for bag_id in sorted(per_bag):
            meta = bag_meta[bag_id]
            insts = tuple(sorted(per_bag[bag_id], key=lambda r: r.image_path))
            listed = tuple(meta["instance_list"].split("|"))
            listed_here = tuple(p for p in listed)
            if tuple(i.image_path for i in insts) != tuple(
                    sorted(listed_here)):
                raise DatasetError(
                    f"bag {bag_id}: instance membership/order mismatch between "
                    f"frozen {split} manifest and Phase 3 instance_list")
            if insts and len({i.label for i in insts}) != 1:
                raise DatasetError(f"bag {bag_id}: inconsistent labels within bag")
            if insts and insts[0].label != meta["bag_label"]:
                raise DatasetError(
                    f"bag {bag_id}: label {meta['bag_label']!r} != instance "
                    f"label {insts[0].label!r}")
            if insts and len({i.source_group_id for i in insts}) != 1:
                raise DatasetError(f"bag {bag_id}: mixed source_group_id")
            if insts and any(i.source_key_status !=
                             "inferred_from_filename_not_verified_identifier"
                             for i in insts):
                raise DatasetError(
                    f"bag {bag_id}: source_key_status marker drifted")
            rec0 = insts[0]
            self._bags.append(BagRecord(
                bag_id=bag_id,
                source_group_id=meta["source_group_id"],
                label=meta["bag_label"],
                split=split,
                source_key=meta["source_key"],
                source_key_status=meta["source_key_status"],
                instances=insts,
            ))

        # Cross-split integrity: no bag and no instance may appear in another
        # split's frozen manifest (Phase 4 guarantees this; re-verified here).
        for other in SPLITS:
            if other == split:
                continue
            op = os.path.join(self.root, "data", "manifests", f"{other}_split.csv")
            other_paths = {r["image_path"] for r in self._read_csv(op)}
            mine = {i.image_path for b in self._bags for i in b.instances}
            clash = mine & other_paths
            if clash:
                raise DatasetError(
                    f"leakage: {len(clash)} instances of split {split} also in "
                    f"{other}: {sorted(clash)[:3]}")
            other_bag_ids = {r["bag_id"] for r in self._read_csv(op)}
            bag_clash = {b.bag_id for b in self._bags} & other_bag_ids
            if bag_clash:
                raise DatasetError(
                    f"leakage: bags {sorted(bag_clash)[:3]} appear in both "
                    f"{split} and {other}")

        # Bag count sanity vs the Phase 4 record.
        if len(self._bags) != EXPECTED_BAG_COUNTS[split]:
            raise DatasetError(
                f"bag count drift for {split}: {len(self._bags)} != "
                f"{EXPECTED_BAG_COUNTS[split]}")

        # Optional file-existence validation (stat calls only; no pixel reads).
        if verify_cache_files:
            for b in self._bags:
                for i in b.instances:
                    if not os.path.isfile(os.path.join(self.root,
                                                       i.processed_relpath)):
                        raise DatasetError(
                            f"missing processed image {i.processed_relpath!r} — "
                            f"the Phase 5 cache must be regenerated locally "
                            f"with `python -m src.preprocessing.pipeline` "
                            f"(cache PNGs are gitignored by policy)")

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _read_csv(path: str) -> List[dict]:
        if not os.path.isfile(path):
            raise DatasetError(
                f"required manifest missing: {path} (fresh clones must first "
                f"regenerate the Phase 5 cache with "
                f"`python -m src.preprocessing.pipeline`)")
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _prov_md5(prov_rows: List[dict], image_path: str) -> str:
        # provenance.csv is small enough; called only on rows already matched.
        for r in prov_rows:
            if r["image_path"] == image_path:
                return r["md5"]
        raise DatasetError(f"{image_path!r} absent from provenance")

    # ------------------------------------------------------------- accessors
    @property
    def bags(self) -> Tuple[BagRecord, ...]:
        """All bags of this split, ordered by ascending bag_id."""
        return tuple(self._bags)

    @property
    def bag_ids(self) -> Tuple[str, ...]:
        return tuple(b.bag_id for b in self._bags)

    @property
    def instances(self) -> Tuple[InstanceRecord, ...]:
        """All instances, ordered by ascending image_path (deterministic)."""
        return tuple(sorted((i for b in self._bags for i in b.instances),
                            key=lambda r: r.image_path))

    @property
    def bag_sizes(self) -> Tuple[int, ...]:
        return tuple(b.instance_count for b in self._bags)

    def bag(self, bag_id: str) -> BagRecord:
        for b in self._bags:
            if b.bag_id == bag_id:
                return b
        raise DatasetError(f"no bag {bag_id!r} in split {self.split!r}")

    def __len__(self) -> int:
        return len(self._bags)

    def __getitem__(self, idx: int) -> BagRecord:
        return self._bags[idx]

    def __iter__(self):
        return iter(self._bags)

    # ------------------------------------------------------- image loading
    def load_instance_image(self, record: InstanceRecord) -> np.ndarray:
        """Decode one cached Phase 5 image -> float32 (224, 224).

        Preserves the Phase 5 contract: this is the EXACT integer inverse of
        the Phase 5 storage transform (uint16 -> normalized float). No
        re-normalization, no augmentation, no additional preprocessing.
        """
        path = os.path.join(self.root, record.processed_relpath)
        if not os.path.isfile(path):
            raise DatasetError(
                f"processed image missing: {record.processed_relpath} — "
                f"regenerate the Phase 5 cache with "
                f"`python -m src.preprocessing.pipeline`")
        try:
            with Image.open(path) as im:
                if im.mode != "I;16" or im.size != (self.image_size[1],
                                                    self.image_size[0]):
                    raise DatasetError(
                        f"Phase 5 contract violation in "
                        f"{record.processed_relpath}: mode={im.mode} "
                        f"size={im.size} (expected I;16, "
                        f"{self.image_size[1]}x{self.image_size[0]})")
                arr = np.asarray(im)
        except DatasetError:
            raise
        except Exception as e:  # noqa: BLE001 - explicit, never silent
            raise DatasetError(
                f"corrupt processed image {record.processed_relpath}: {e}") from e
        norm = _p5.decode_cached(arr, self.config)
        return norm.astype(np.float32)

    def load_bag_images(self, bag: BagRecord) -> np.ndarray:
        """Load all instances of one bag -> float32 (n_i, 224, 224).

        Variable bag size is preserved; no padding, no resizing, no
        duplication, no discarding.
        """
        return np.stack([self.load_instance_image(i) for i in bag.instances])

    def load_bag_with_label(self, bag: BagRecord):
        """Convenience: (images (n_i,224,224) float32, numeric_label int)."""
        return self.load_bag_images(bag), bag.numeric_label

    # ----------------------------------------------------------- collate
    @staticmethod
    def collate_bags(batches: Sequence[Tuple[np.ndarray, int, str]]) -> dict:
        """Variable-length bag collation with an explicit validity mask.

        Input : sequence of (images (n_i,224,224) float32, numeric_label,
                bag_id).
        Output: dict with
            images           (B, N_max, 224, 224) float32 — zero-padded
            mask             (B, N_max) bool — True ONLY at real-instance
                             positions; padded positions are False and MUST
                             be excluded by downstream models
            labels           (B,) int64
            bag_ids          tuple[str, ...]
            instance_counts  (B,) int64 (== mask.sum(axis=1))

        Padding is 0.0 and is never a real ultrasound instance; the mask is
        the single source of truth for validity.
        """
        if not batches:
            raise DatasetError("collate_bags received an empty batch")
        n_max = max(im.shape[0] for im, _, _ in batches)
        h = batches[0][0].shape[1]
        w = batches[0][0].shape[2]
        images = np.zeros((len(batches), n_max, h, w), dtype=np.float32)
        mask = np.zeros((len(batches), n_max), dtype=bool)
        labels = np.zeros(len(batches), dtype=np.int64)
        counts = np.zeros(len(batches), dtype=np.int64)
        bag_ids: List[str] = []
        for b, (im, lab, bid) in enumerate(batches):
            n = im.shape[0]
            if im.shape[1:] != (h, w):
                raise DatasetError(
                    f"inconsistent image shape in batch: {im.shape[1:]} vs "
                    f"{(h, w)}")
            images[b, :n] = im
            mask[b, :n] = True
            labels[b] = lab
            counts[b] = n
            bag_ids.append(bid)
        return {"images": images, "mask": mask, "labels": labels,
                "bag_ids": tuple(bag_ids), "instance_counts": counts}

    # ------------------------------------------------------------ summary
    def summary(self) -> dict:
        sizes = self.bag_sizes
        return {
            "split": self.split,
            "bags": len(self._bags),
            "instances": int(sum(sizes)),
            "bag_size_min": int(min(sizes)),
            "bag_size_max": int(max(sizes)),
            "bag_size_mean": round(float(np.mean(sizes)), 4),
            "bag_size_median": float(np.median(sizes)),
            "bag_size_histogram": {
                str(s): int(sizes.count(s)) for s in sorted(set(sizes))},
            "label_counts": {lab: int(sum(1 for b in self._bags
                                          if b.label == lab))
                             for lab in VALID_LABELS},
            "frozen_test_sha256": self.frozen_test_sha256,
            "ordering_rule": "bags ascending bag_id; instances ascending image_path",
            "augmentation": "none (training-time augmentation is Phase 5 "
                            "pipeline.augment(), train split only, never here)",
        }


def build_all_summaries(root: Optional[str] = None) -> dict:
    """Convenience for reports/tests: summaries of all three split loaders."""
    return {s: UltrasoundBagDataset(s, root=root).summary() for s in SPLITS}
