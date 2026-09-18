"""Phase 4: deterministic, leakage-aware train/val/test splitting.

Roadmap: PROGRESS.md Phase 4 — Patient/Study-Level Data Splitting &
Leakage Prevention. Decision documented in reports/phase4_split_report.md.

Split unit
----------
The highest defensible grouping level available. Phase 1 verified that NO
patient, study, or lesion identifiers exist in the dataset, so:

    split unit = one complete Phase 3 ``source_group_id`` /
                 source-image family (BAG)

Instances of one source group are NEVER distributed across splits.

Near-duplicate constraint policy (conservative, per Phase 4 instructions)
-------------------------------------------------------------------------
Every pair in the committed Phase 2 candidate manifest
(near_duplicate_candidates.csv, 25,607 pairs) is treated as a split-level
CONSTRAINT: its two endpoints must land in the same Phase 4 split. These
remain CANDIDATE relationships — this is a leakage-prevention policy, NOT
a reclassification of candidates as confirmed duplicates, and it does NOT
alter the authoritative Phase 2 grouping artifact. Allocation units are
connected components of this constraint graph over source groups; a unit
is a Phase 4 bookkeeping construct, not an identity claim. Some units
contain both benign and malignant groups (371 candidate pairs are
cross-class); co-splitting them is the conservative choice and does not
affect any bag's label.

Determinism
-----------
Fixed seed and explicit ordering everywhere; no timestamps inside the
manifests; LF line endings. Running this module twice in fresh processes
produces byte-identical train/val/test manifests.

This module performs splitting ONLY: no preprocessing, augmentation,
features, training, attention, checkpoints, or evaluation (Phase 5+).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

SPLIT_SEED = 20260918          # fixed, documented deterministic seed
TARGETS = {"train": 0.70, "val": 0.15, "test": 0.15}   # of instances, per class
SPLIT_ORDER = ["train", "val", "test"]                 # tie-break order
TEST_FREEZE_DATE = "2026-09-18"

SPLIT_COLUMNS = [
    "split",
    "source_group_id",
    "bag_id",
    "bag_label",
    "image_path",
    "md5",
    "source_key",
    "source_key_status",
    "allocation_unit_id",
    "unit_bag_count",
    "unit_instance_count",
    "original_supplied_split",
]


class SplittingError(RuntimeError):
    """Raised when inputs violate the Phase 4 splitting invariants."""


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_bags(bag_manifest_csv: str) -> List[dict]:
    with open(bag_manifest_csv, newline="", encoding="utf-8") as f:
        bags = list(csv.DictReader(f))
    if not bags:
        raise SplittingError("empty bag manifest")
    return bags


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_units(bags: List[dict], candidates_csv: str) -> List[dict]:
    """Union source groups over ALL candidate pairs -> allocation units.

    Returns units sorted deterministically by (instance_count desc,
    min_bag_id asc). Each unit carries its instances with full traceability.
    """
    bag_by_id = {b["bag_id"]: b for b in bags}
    path_to_bag: Dict[str, dict] = {}
    for b in bags:
        for p in b["instance_list"].split("|"):
            if p in path_to_bag:
                raise SplittingError(f"image in two bags: {p}")
            path_to_bag[p] = b

    # union-find over bag ids (1 bag == 1 source group, per Phase 3)
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    n_pairs = 0
    with open(candidates_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ba = path_to_bag.get(row["file_a"])
            bb = path_to_bag.get(row["file_b"])
            if ba is None or bb is None:
                raise SplittingError(
                    f"candidate endpoint not in bag manifest: {row}")
            n_pairs += 1
            union(ba["bag_id"], bb["bag_id"])

    comps: Dict[str, List[str]] = defaultdict(list)
    for b in bags:
        comps[find(b["bag_id"])].append(b["bag_id"])

    units: List[dict] = []
    for members in comps.values():
        member_bags = [bag_by_id[m] for m in members]
        unit_id = "unit-" + hashlib.sha256(
            ",".join(sorted(members)).encode("utf-8")).hexdigest()[:12]
        instances = []
        inst_by_class = {"benign": 0, "malignant": 0}
        for b in sorted(member_bags, key=lambda x: x["bag_id"]):
            for p, m in zip(b["instance_list"].split("|"),
                            b["instance_md5_list"].split("|")):
                instances.append({
                    "source_group_id": b["source_group_id"],
                    "bag_id": b["bag_id"],
                    "bag_label": b["bag_label"],
                    "image_path": p,
                    "md5": m,
                    "source_key": b["source_key"],
                    "source_key_status": b["source_key_status"],
                    "original_supplied_split": p.split("/")[0],
                })
                inst_by_class[b["bag_label"]] += 1
        units.append({
            "unit_id": unit_id,
            "bag_ids": sorted(members),
            "group_ids": sorted({b["source_group_id"] for b in member_bags}),
            "instances": sorted(instances, key=lambda r: r["image_path"]),
            "instance_count": len(instances),
            "inst_by_class": inst_by_class,
            "min_bag_id": min(members),
        })
    if n_pairs == 0:
        raise SplittingError("no candidate pairs loaded")
    units.sort(key=lambda u: (-u["instance_count"], u["min_bag_id"]))
    return units


def allocate(units: List[dict]) -> Dict[str, str]:
    """Deterministic stratified greedy allocation of units to splits.

    Units are processed largest-first; each is assigned to the split whose
    class-wise fill (relative to its proportional target) has the most room,
    computed as max over classes of (fill+unit)/target — minimising the
    worst-class ratio keeps per-class proportions close to 70/15/15.
    Ties resolve in SPLIT_ORDER. Group membership is never altered.
    """
    total_by_class = {"benign": 0, "malignant": 0}
    for u in units:
        for c, n in u["inst_by_class"].items():
            total_by_class[c] += n
    target_inst = {
        s: {c: total_by_class[c] * TARGETS[s] for c in ("benign", "malignant")}
        for s in SPLIT_ORDER
    }
    fill = {s: {"benign": 0, "malignant": 0} for s in SPLIT_ORDER}
    assignment: Dict[str, str] = {}
    for u in units:  # already ordered deterministically
        best, best_gap = None, None
        for s in SPLIT_ORDER:
            gap = max((fill[s][c] + u["inst_by_class"][c]) / target_inst[s][c]
                      for c in ("benign", "malignant"))
            if best_gap is None or gap < best_gap - 1e-12:
                best, best_gap = s, gap
        assignment[u["unit_id"]] = best
        for c in ("benign", "malignant"):
            fill[best][c] += u["inst_by_class"][c]
    return assignment


def write_split_csvs(units: List[dict], assignment: Dict[str, str],
                     out_dir: str) -> Dict[str, str]:
    """Write the three split manifests; returns {split: sha256}."""
    rows: Dict[str, List[dict]] = {s: [] for s in SPLIT_ORDER}
    for u in units:
        s = assignment[u["unit_id"]]
        for r in u["instances"]:
            rows[s].append({
                "split": s,
                "source_group_id": r["source_group_id"],
                "bag_id": r["bag_id"],
                "bag_label": r["bag_label"],
                "image_path": r["image_path"],
                "md5": r["md5"],
                "source_key": r["source_key"],
                "source_key_status": r["source_key_status"],
                "allocation_unit_id": u["unit_id"],
                "unit_bag_count": len(u["bag_ids"]),
                "unit_instance_count": u["instance_count"],
                "original_supplied_split": r["original_supplied_split"],
            })
    os.makedirs(out_dir, exist_ok=True)
    hashes: Dict[str, str] = {}
    for s in SPLIT_ORDER:
        out = os.path.join(out_dir, f"{s}_split.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SPLIT_COLUMNS,
                               lineterminator="\n")
            w.writeheader()
            for row in sorted(rows[s], key=lambda r: r["image_path"]):
                w.writerow(row)
        hashes[s] = sha256_of_file(out)
    return hashes


def split_statistics(units: List[dict], assignment: Dict[str, str]) -> dict:
    """Per-split counts recomputed from the units (used by report/tests)."""
    stats = {s: {"groups": 0, "bags": 0, "instances": 0,
                 "benign_instances": 0, "malignant_instances": 0,
                 "units": 0, "bag_sizes": []} for s in SPLIT_ORDER}
    for u in units:
        s = assignment[u["unit_id"]]
        st = stats[s]
        st["units"] += 1
        st["groups"] += len(u["group_ids"])
        st["bags"] += len(u["bag_ids"])
        st["instances"] += u["instance_count"]
        st["benign_instances"] += u["inst_by_class"]["benign"]
        st["malignant_instances"] += u["inst_by_class"]["malignant"]
    # exact bag sizes from unit instance rows
    for u in units:
        s = assignment[u["unit_id"]]
        per_bag = defaultdict(int)
        for r in u["instances"]:
            per_bag[r["bag_id"]] += 1
        stats[s]["bag_sizes"].extend(per_bag[b] for b in sorted(per_bag))
        stats[s]["bag_sizes"].sort()
    for s in SPLIT_ORDER:
        st = stats[s]
        sizes = st.pop("bag_sizes")
        st["bag_size_min"] = min(sizes) if sizes else 0
        st["bag_size_max"] = max(sizes) if sizes else 0
        st["bag_size_mean"] = round(statistics.fmean(sizes), 4) if sizes else 0.0
        st["bag_size_median"] = statistics.median(sizes) if sizes else 0
    return stats


def main() -> int:  # pragma: no cover - orchestration wrapper
    root = repo_root()
    mdir = os.path.join(root, "data", "manifests")
    bags = load_bags(os.path.join(mdir, "bag_manifest.csv"))
    units = build_units(bags, os.path.join(mdir, "near_duplicate_candidates.csv"))
    assignment = allocate(units)
    hashes = write_split_csvs(units, assignment, mdir)
    stats = split_statistics(units, assignment)

    multi_dir_units = [u["unit_id"] for u in units
                       if len({r["original_supplied_split"]
                               for r in u["instances"]}) > 1]
    cross_class_units = [u["unit_id"] for u in units
                         if len({r["bag_label"] for r in u["instances"]}) > 1]

    print(json.dumps({
        "phase4_version": "1.0.0",
        "split_seed": SPLIT_SEED,
        "targets": TARGETS,
        "allocation_units": len(units),
        "units_with_mixed_labels": len(cross_class_units),
        "units_spanning_original_supplied_dirs": len(multi_dir_units),
        "stats": stats,
        "sha256": hashes,
        "test_freeze_date": TEST_FREEZE_DATE,
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
