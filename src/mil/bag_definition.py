"""Phase 3: MIL bag definition and bag-manifest construction.

Roadmap: PROGRESS.md Phase 3 — Data Organization & MIL Bag Definition.
Decision documented in reports/phase3_bag_definition.md:

    BAG      = one Phase 2 ``source_group_id`` (one source-image family)
    INSTANCE = one image file belonging to that source group
    LABEL    = the group's directory-encoded class (``benign`` / ``malignant``)

Patient/study/lesion bag levels were evaluated and REJECTED: Phase 1 found
no verified patient, study, or lesion identifiers (the dataset has no
metadata files; filename-derived source keys are explicitly labelled
``inferred_from_filename_not_verified_identifier`` and are NOT patient or
study IDs).

Scientific-integrity note (documented, not a methodology change): instances
of a bag are related augmented image variants of a single source-image
family. They are NOT independent patients, studies, lesions, or independent
clinical observations.

This module deliberately implements ONLY bag definition: no features, no
CNN, no attention, no splitting (Phase 4), no preprocessing for training
(Phase 5), no training/checkpoints/experiments.

Outputs are deterministic: no timestamps inside the manifest; bags ordered
by ``bag_id``; instance lists ordered by image path; LF line endings.
"""
from __future__ import annotations

import csv
import json
import os
import statistics
import sys
from typing import Dict, List

BAG_ID_PREFIX = "bag-"
GRP_PREFIX = "grp-"

MANIFEST_COLUMNS = [
    "bag_id",
    "source_group_id",
    "bag_label",
    "instance_count",
    "instance_list",
    "instance_md5_list",
    "source_key",
    "source_key_status",
    "splits",
    "multi_split",
    "original_candidate_path",
]

RAW_MANIFEST_COLUMNS = [
    "bag_id",
    "source_group_id",
    "bag_label",
    "instance_count",
    "instance_list",
]


class BagManifestError(RuntimeError):
    """Raised when inputs violate the Phase 3 bag-definition invariants."""


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_source_groups(augmentation_groups_csv: str) -> Dict[str, List[dict]]:
    """Load the committed Phase 2 grouping manifest, grouped by source_group_id.

    Enforces the invariants the bag definition relies on:
      * every row carries a non-empty ``source_group_id``;
      * every group is label-coherent (``group_label`` == ``class_dir`` for
        every member, and one label per group).
    """
    groups: Dict[str, List[dict]] = {}
    with open(augmentation_groups_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gid = (row.get("source_group_id") or "").strip()
            if not gid:
                raise BagManifestError(f"row without source_group_id: {row.get('path')!r}")
            label = (row.get("group_label") or "").strip()
            class_dir = (row.get("class_dir") or "").strip()
            if label != class_dir or label not in ("benign", "malignant"):
                raise BagManifestError(
                    f"row {row.get('path')!r} has ambiguous label "
                    f"(group_label={label!r}, class_dir={class_dir!r})"
                )
            groups.setdefault(gid, []).append(row)
    for gid, rows in groups.items():
        labels = {r["group_label"] for r in rows}
        if len(labels) != 1:
            raise BagManifestError(f"group {gid} contains conflicting labels: {sorted(labels)}")
    return groups


def derive_bag_id(source_group_id: str) -> str:
    """Deterministic 1:1 mapping grp-XXXXXXXXXXXX -> bag-XXXXXXXXXXXX."""
    if not source_group_id.startswith(GRP_PREFIX):
        raise BagManifestError(f"unexpected source_group_id format: {source_group_id!r}")
    return BAG_ID_PREFIX + source_group_id[len(GRP_PREFIX):]


def build_bags(groups: Dict[str, List[dict]]) -> List[dict]:
    """Build bag records; deterministic ordering (bags by bag_id, instances by path)."""
    bags: List[dict] = []
    for gid in sorted(groups):
        rows = sorted(groups[gid], key=lambda r: r["path"])
        labels = {r["group_label"] for r in rows}
        if len(labels) != 1:
            raise BagManifestError(f"group {gid} is not label-coherent")
        originals = [r["path"] for r in rows if r.get("role") == "original_candidate"]
        if len(originals) > 1:
            raise BagManifestError(f"group {gid} has {len(originals)} original candidates")
        splits = sorted({r["split"] for r in rows})
        keys = sorted({r["source_key"] for r in rows})
        if len(keys) != 1:
            raise BagManifestError(f"group {gid} spans multiple source keys: {keys}")
        statuses = sorted({r.get("source_key_status", "") for r in rows})
        bags.append({
            "bag_id": derive_bag_id(gid),
            "source_group_id": gid,
            "bag_label": labels.pop(),
            "instance_count": len(rows),
            "instance_list": "|".join(r["path"] for r in rows),
            "instance_md5_list": "|".join(r["md5"] for r in rows),
            "source_key": keys[0],
            "source_key_status": statuses[0] if len(statuses) == 1 else "|".join(statuses),
            "splits": "|".join(splits),
            "multi_split": "True" if len(splits) > 1 else "False",
            "original_candidate_path": originals[0] if originals else "",
        })
    return bags


def bag_size_statistics(bags: List[dict]) -> dict:
    sizes = [b["instance_count"] for b in bags]
    return {
        "bags": len(bags),
        "min": min(sizes),
        "max": max(sizes),
        "mean": round(statistics.fmean(sizes), 4),
        "median": statistics.median(sizes),
        "total_instances": sum(sizes),
    }


def verify_against_phase1(bags: List[dict], dataset_manifest_csv: str) -> None:
    """Every bag instance must exist in the Phase 1 manifest with the same md5."""
    p1: Dict[str, str] = {}
    with open(dataset_manifest_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p1[row["path"]] = row["md5"]
    seen: Dict[str, str] = {}
    for bag in bags:
        for path, md5 in zip(bag["instance_list"].split("|"),
                             bag["instance_md5_list"].split("|")):
            if path in seen:
                raise BagManifestError(f"image in two bags: {path}")
            if p1.get(path) != md5:
                raise BagManifestError(f"instance not traceable to Phase 1 manifest: {path}")
            seen[path] = md5
    if len(seen) != len(p1):
        raise BagManifestError(
            f"instance coverage mismatch: bags cover {len(seen)} of {len(p1)} Phase 1 rows"
        )


def write_manifest(bags: List[dict], out_csv: str, columns: List[str] = None) -> None:
    cols = columns or MANIFEST_COLUMNS
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n",
                           extrasaction="ignore")
        w.writeheader()
        for bag in bags:
            w.writerow(bag)


def main() -> int:  # pragma: no cover - thin orchestration wrapper
    root = repo_root()
    groups_csv = os.path.join(root, "data", "manifests", "augmentation_groups.csv")
    p1_csv = os.path.join(root, "data", "manifests", "dataset_manifest.csv")
    out_csv = os.path.join(root, "data", "manifests", "bag_manifest.csv")

    groups = load_source_groups(groups_csv)
    bags = build_bags(groups)
    verify_against_phase1(bags, p1_csv)
    write_manifest(bags, out_csv)
    stats = bag_size_statistics(bags)

    label_counts: Dict[str, int] = {}
    multi_split = [b["bag_id"] for b in bags if b["multi_split"] == "True"]
    for bag in bags:
        label_counts[bag["bag_label"]] = label_counts.get(bag["bag_label"], 0) + 1

    print(json.dumps({
        "phase3_version": "1.0.0",
        "bag_definition": {
            "BAG": "one Phase 2 source_group_id (source-image family)",
            "INSTANCE": "one image file in that source group",
            "LABEL": "group directory-encoded class (benign|malignant)",
            "identifier_basis": "source_key_status="
                                "inferred_from_filename_not_verified_identifier"
                                " (NOT patient/study/lesion IDs)",
        },
        "inputs": {"source_groups": len(groups),
                   "phase2_manifest": os.path.relpath(groups_csv, root)},
        "stats": stats,
        "bags_by_label": label_counts,
        "multi_split_bags": multi_split,
        "output": os.path.relpath(out_csv, root),
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
