"""Phase 2: duplicate detection, augmentation-lineage analysis, and grouping.

Roadmap: PROGRESS.md Phase 2 — Data Cleaning, Image Integrity & Augmentation
Leakage Analysis. Decision D-1 maps this file to
src/preprocessing/duplicate_detection.py. Raw data under dataset/raw/ is
immutable evidence: this script NEVER writes inside the dataset root.

Grouping policy (the heart of Phase 2)
--------------------------------------
source_group_id = connected components over CONFIRMED-IDENTITY edges only:

  1. filename lineage: same `source_key` parsed from the raw filename stem
     (the dataset's own naming scheme: "benign (100)" + augmentation chains);
  2. exact content identity: identical md5 (byte-identical files, including
     the Phase 1 finding of byte-identical files under DIFFERENT chain labels).

Two guards are applied on every merge:
  - label safety: an edge whose endpoints have different class labels
    (benign vs malignant) is NEVER merged; it is counted and reported as a
    label-conflict edge instead;
  - split policy: merges are split-agnostic BY DESIGN (group-level splitting
    in Phase 4 depends on it), but every resulting multi-split group is
    counted and listed as a leakage-risk artifact.

Near-duplicate candidate pairs (Phase 1 dHash candidates, plus any new dHash
couples found within groups) are NOT identity edges: they are reported,
SSIM-cross-validated, and carried into the manifest as candidate fields, but
they never join source groups.

Outputs (data/manifests/, deterministic):
    augmentation_groups.csv       one row per image: group id, lineage, role,
                                  flags (sorted by group id, then path)
    phase2_grouping_summary.json  machine-readable group statistics
    ssim_crosscheck.csv           SSIM per candidate pair (corroboration only)

Usage
-----
    python src/preprocessing/duplicate_detection.py [--root DIR]
        [--manifests-dir DIR] [--ssim-size 64] [--limit-ssim N]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

# Reuse the Phase 1 audited module (grammar, dHash, root discovery) verbatim.
_AUDIT_PATH = Path(__file__).resolve().parents[1] / "data" / "audit.py"
_spec = importlib.util.spec_from_file_location("phase1_audit", _AUDIT_PATH)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)

PHASE2_VERSION = "1.0.0"
SOURCE_KEY_STATUS = audit.SOURCE_KEY_STATUS  # inherited verbatim

GROUP_FIELDS = [
    "path", "split", "class_dir", "md5", "source_key", "chain",
    "source_group_id", "group_size", "group_splits", "group_label",
    "role", "role_confidence", "role_rule", "duplicate_group",
    "near_duplicate_group", "near_dup_candidate", "dhash_to_original",
    "grouping_confidence", "grouping_basis", "source_key_status",
    "ungrouped_low_confidence",
]

ROLE_BASE = "original_candidate"
ROLE_AUG = "augmented_variant"


# ---------------------------------------------------------------------------
# Grouping (confirmed-identity edges only)
# ---------------------------------------------------------------------------

def group_images(records: list[dict]) -> tuple[dict[str, list[dict]], dict]:
    """Union-find over confirmed-identity edges; returns groups + edge stats.

    records: rows of the Phase 1 manifest (path/split/class_dir/md5/
    source_key/duplicate_group/near_duplicate_group/...).
    """
    n = len(records)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[parent[a]]
        return a

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if ra < rb:  # deterministic root: smaller index wins
            parent[rb] = ra
        else:
            parent[ra] = rb
        return True

    edges = {"filename_source_key": 0, "exact_md5": 0, "redundant": 0,
             "label_conflict_blocked": 0}
    label_conflicts: list[dict] = []

    def merge(i: int, j: int, kind: str) -> None:
        if records[i]["class_dir"] != records[j]["class_dir"]:
            edges["label_conflict_blocked"] += 1
            label_conflicts.append({
                "edge_kind": kind,
                "file_a": records[i]["path"], "label_a": records[i]["class_dir"],
                "file_b": records[j]["path"], "label_b": records[j]["class_dir"],
            })
            return
        if union(i, j):
            edges[kind] += 1
        else:
            edges["redundant"] += 1

    by_key: dict[str, list[int]] = defaultdict(list)
    by_md5: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        if r["source_key"]:
            by_key[r["source_key"]].append(i)
        if r["md5"]:
            by_md5[r["md5"]].append(i)

    for idxs in by_key.values():          # edges added in deterministic order
        for k in range(1, len(idxs)):
            merge(idxs[k - 1], idxs[k], "filename_source_key")
    for idxs in by_md5.values():
        for k in range(1, len(idxs)):
            merge(idxs[k - 1], idxs[k], "exact_md5")

    comps: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(i)

    groups: dict[str, list[dict]] = {}
    for members in sorted(comps.values(), key=lambda ms: min(ms)):
        # group id: digest over sorted member md5s — stable across runs
        key_src = "|".join(sorted(records[m]["md5"] or records[m]["path"]
                                  for m in members))
        gid = "grp-" + hashlib.md5(key_src.encode()).hexdigest()[:12]
        for m in members:
            records[m]["source_group_id"] = gid
        groups[gid] = [records[m] for m in members]
    return groups, {"edges": edges, "label_conflicts": label_conflicts}


def group_splits_and_label(group: list[dict]) -> tuple[str, str]:
    splits = sorted({r["split"] for r in group if r["split"]})
    labels = sorted({r["class_dir"] for r in group})
    return "|".join(splits), "|".join(labels)


def recover_roles(group: list[dict]) -> None:
    """Best-effort original-vs-augmented recovery (documented confidence).

    Evidence layers (Phase 1 verified facts):
      - chain rule: chained stems are augmented_variant with HIGH confidence
        (the dataset's own filenames state the augmentation);
      - dimension rule: base (chain-free) files are exactly the 227x227
        population, so a chain-free 227x227 file is original_candidate with
        HIGH confidence; a chain-free file at another dimension (e.g. a
        byte-identical JPG/PNG pair whose PNG copy was 227x227) keeps the
        original_candidate role at MEDIUM confidence (grammar root, but no
        dimension corroboration).
    role_rule records which rule decided, so every confidence is auditable.
    """
    for r in group:
        if r["chain"]:
            r["role"] = ROLE_AUG
            r["role_confidence"] = "high"
            r["role_rule"] = "filename_chain_explicit"
        else:
            r["role"] = ROLE_BASE
            if (r["width"], r["height"]) == (227, 227):
                r["role_confidence"] = "high"
                r["role_rule"] = "chain_free+base_dimension_227x227"
            else:
                r["role_confidence"] = "medium"
                r["role_rule"] = "chain_free_only(off_base_dimension)"
    if not any((r["width"], r["height"]) == (227, 227) for r in group):
        for r in group:  # no dimension-confirmed original exists in this group
            if r["role"] == ROLE_BASE:
                r["role_confidence"] = "medium"


def assign_group_confidence(group: list[dict]) -> str:
    """Group-level confidence label, from the evidence that built it."""
    has_lineage = any(r["source_key"] for r in group)
    has_exact = any(r["duplicate_group"] for r in group)
    if has_lineage and has_exact:
        return "high (filename lineage + exact md5 duplicates inside group)"
    if has_lineage:
        return "high (filename lineage; dataset's own naming scheme)"
    if has_exact:
        return "medium (exact md5 duplicates; no filename lineage)"
    return "low (singleton; no corroborating evidence)"


# ---------------------------------------------------------------------------
# SSIM cross-validation (Wang et al. 2004 formulation, numpy-only)
# ---------------------------------------------------------------------------

def _gaussian_kernel(sigma: float = 1.5, radius: int = 5) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _filter2_valid(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Separable symmetric-padded correlation -> same grid as the image.

    Vectorized (sliding windows via stride tricks); ~40x faster than the
    apply_along_axis version, which timed out at the 10-minute cap.
    """
    pad = len(k) // 2
    a = np.pad(img, ((pad, pad), (0, 0)), mode="symmetric")
    # rows
    v = np.lib.stride_tricks.sliding_window_view(a, len(k), axis=1)  # (H, W, 11)
    tmp = v @ k[::-1]                                                # (H, W)
    a2 = np.pad(tmp, ((0, 0), (pad, pad)), mode="symmetric")
    v2 = np.lib.stride_tricks.sliding_window_view(a2, len(k), axis=0)  # (H, W, 11)
    return v2 @ k[::-1]


def ssim_gray(a: np.ndarray, b: np.ndarray, data_range: float = 255.0,
              win: int = 7, sigma: float = 1.5) -> float:
    """Global mean SSIM (uniform-window equivalent, statistics via gaussian).

    Uses an 11x11 gaussian-window local statistics map (sigma 1.5, matching
    the canonical Wang et al. settings with C1=(0.01*L)^2, C2=(0.03*L)^2) and
    averages the SSIM map over an interior win x win crop so border effects
    are excluded. Deterministic; float64 throughout.
    """
    k = _gaussian_kernel(sigma)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu_a = _filter2_valid(a, k)
    mu_b = _filter2_valid(b, k)
    saa = _filter2_valid(a * a, k) - mu_a * mu_a
    sbb = _filter2_valid(b * b, k) - mu_b * mu_b
    sab = _filter2_valid(a * b, k) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / \
        ((mu_a * mu_a + mu_b * mu_b + C1) * (saa + sbb + C2))
    h, w = s.shape
    r = win // 2
    return float(s[h // 2 - r:h // 2 + r + 1, w // 2 - r:w // 2 + r + 1].mean())


def _gray64(im: Image.Image, size: int) -> np.ndarray:
    g = im.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(g, dtype=np.float64)


def crosscheck_ssim(pairs: list[tuple[str, str, int]], root: str,
                    out_csv: str, size: int = 64) -> dict:
    """SSIM for every candidate pair; writes ssim_crosscheck.csv; returns stats.

    Pairs: (path_a, path_b, dhash_distance). Corroboration only: SSIM never
    adds, removes or changes any grouping — it measures whether the dHash
    candidates are also structurally similar.
    """
    cache: dict[str, np.ndarray] = {}

    def gray(rel: str) -> np.ndarray:
        if rel not in cache:
            with Image.open(os.path.join(root, *rel.split("/"))) as im:
                im.load()
                cache[rel] = _gray64(im, size)
        return cache[rel]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    hist = Counter()
    min_ssim, max_ssim = 1.1, -1.0
    below: list[tuple[str, str, int, float]] = []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_a", "file_b", "dhash_hamming", "ssim_64"])
        for pa, pb, d in pairs:
            try:
                s = ssim_gray(gray(pa), gray(pb))
            except Exception:  # noqa: BLE001 - record, never crash
                w.writerow([pa, pb, d, ""])
                hist["ssim_errors"] += 1
                continue
            w.writerow([pa, pb, d, f"{s:.6f}"])
            if s == 1.0:
                hist["ssim_exactly_1"] += 1
            elif s >= 0.90:
                hist["ssim_ge_0.90"] += 1
            elif s >= 0.80:
                hist["ssim_0.80_0.90"] += 1
            elif s >= 0.60:
                hist["ssim_0.60_0.80"] += 1
            else:
                hist["ssim_lt_0.60"] += 1
                below.append((pa, pb, d, s))
            min_ssim, max_ssim = min(min_ssim, s), max(max_ssim, s)
    computed = sum(v for k, v in hist.items() if k != "ssim_errors")
    return {
        "method": (f"global mean SSIM, {size}x{size} grayscale, 11x11 gaussian "
                   "window sigma=1.5, C1/C2 per Wang et al. 2004 (numpy-only, "
                   "deterministic); re-decoded from raw files"),
        "purpose": ("cross-validation of perceptual-hash candidate pairs "
                    "(roadmap Phase 2 task 3); corroboration only — no "
                    "grouping is changed by SSIM"),
        "pairs_expected": len(pairs),
        "pairs_computed": computed,
        "ssim_errors": hist.get("ssim_errors", 0),
        "histogram": {k: v for k, v in sorted(hist.items()) if k != "ssim_errors"},
        "min_ssim": None if max_ssim < -1.0 else round(min_ssim, 6),
        "max_ssim": None if max_ssim < -1.0 else round(max_ssim, 6),
        "pairs_below_0.60": len(below),
        "lowest_examples": [{"file_a": pa, "file_b": pb, "dhash": d,
                             "ssim": round(s, 6)} for pa, pb, d, s
                            in sorted(below, key=lambda t: t[3])[:10]],
    }


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def write_augmentation_groups_csv(groups: dict[str, list[dict]],
                                  out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=GROUP_FIELDS, extrasaction="ignore")
        w.writeheader()
        for gid in sorted(groups):
            rows = sorted(groups[gid], key=lambda r: r["path"])
            splits, label = group_splits_and_label(rows)
            conf = assign_group_confidence(rows)
            for r in rows:
                w.writerow({
                    "path": r["path"], "split": r["split"],
                    "class_dir": r["class_dir"], "md5": r["md5"],
                    "source_key": r["source_key"], "chain": r["chain"],
                    "source_group_id": gid, "group_size": len(rows),
                    "group_splits": splits, "group_label": label,
                    "role": r["role"], "role_confidence": r["role_confidence"],
                    "role_rule": r["role_rule"],
                    "duplicate_group": r["duplicate_group"] or "",
                    "near_duplicate_group": r["near_duplicate_group"] or "",
                    "near_dup_candidate": r["near_dup_candidate"],
                    "dhash_to_original": r["dhash_to_original"],
                    "grouping_confidence": conf,
                    "grouping_basis": r["grouping_basis"],
                    "source_key_status": SOURCE_KEY_STATUS,
                    "ungrouped_low_confidence": r["ungrouped_low_confidence"],
                })


def summarize(groups: dict[str, list[dict]], n_records: int,
              phase1_pairs: list[tuple[str, str, int]],
              new_couples: list[tuple[str, str, int]],
              ssim_stats: dict, merge: dict) -> dict:
    sizes = Counter(len(g) for g in groups.values())
    multi_split = [(gid, group_splits_and_label(g)[0])
                   for gid, g in sorted(groups.items())
                   if len(group_splits_and_label(g)[0].split("|")) > 1]

    # unique cross-split candidate pairs inside groups (pairs may repeat
    # across near-dup and new-couple lists; deduplicate unordered pairs)
    cross_pairs: set[tuple[str, str]] = set()
    for pa, pb, _d in list(phase1_pairs) + new_couples:
        if pa.split("/")[0] != pb.split("/")[0]:
            cross_pairs.add(tuple(sorted((pa, pb))))

    return {
        "phase2_version": PHASE2_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {"phase1_manifest_rows": n_records},
        "grouping_policy": {
            "source_group_id": ("connected components over confirmed-identity "
                                "edges only: same filename source_key and "
                                "same md5 (byte-identical)"),
            "near_duplicate_candidates": ("NOT identity edges; reported and "
                                          "SSIM-cross-checked only"),
            "label_guard": ("edges with conflicting class labels are never "
                            "merged; counted and listed"),
            "split_policy": ("grouping is split-agnostic BY DESIGN (Phase 4 "
                             "group-level splitting depends on it); every "
                             "multi-split group is reported as a leakage-risk "
                             "artifact"),
        },
        "merge_edges": merge["edges"],
        "label_conflicts": {"count": merge["edges"]["label_conflict_blocked"],
                            "detail": merge["label_conflicts"]},
        "counts": {
            "total_images": n_records,
            "total_groups": len(groups),
            "group_size_histogram": {str(k): v for k, v in sorted(sizes.items())},
            "singleton_groups": sizes.get(1, 0),
            "largest_group": max(sizes) if sizes else 0,
            "images_in_multi_split_groups":
                sum(len(groups[g]) for g, _ in multi_split),
            "multi_split_groups": len(multi_split),
            "multi_split_group_list": [{"source_group_id": g, "splits": s}
                                       for g, s in multi_split],
            "cross_split_candidate_pairs_in_groups": len(cross_pairs),
        },
        "roles": {
            "original_candidates": sum(1 for g in groups.values()
                                       for r in g if r["role"] == ROLE_BASE),
            "original_high_confidence": sum(
                1 for g in groups.values() for r in g
                if r["role"] == ROLE_BASE and r["role_confidence"] == "high"),
            "augmented_variants": sum(1 for g in groups.values()
                                      for r in g if r["role"] == ROLE_AUG),
            "caveat": ("best-effort recovery from the dataset's own naming "
                       "scheme and the Phase 1 dimension finding; a group "
                       "without a 227x227 member has no high-confidence "
                       "original"),
        },
        "near_duplicate_candidates": {
            "phase1_candidate_pairs": len(phase1_pairs),
            "new_dhash_couples_within_groups": len(new_couples),
            "total_candidate_pairs": len(phase1_pairs) + len(new_couples),
            "status": "CANDIDATES ONLY - not confirmed duplicates",
            "ssim_crosscheck": ssim_stats,
        },
        "ungrouped_low_confidence": {
            "count": sum(1 for g in groups.values() for r in g
                         if r["ungrouped_low_confidence"] == "True"),
            "definition": ("files in low-confidence groups (no lineage and no "
                           "exact duplicate) — none expected in this dataset "
                           "(grammar parse rate was 100%)"),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 2 duplicate/augmentation grouping (PROGRESS.md; D-1)")
    ap.add_argument("--root", default=None,
                    help="dataset root; default: Phase 1 discovery order")
    ap.add_argument("--manifests-dir", default="data/manifests")
    ap.add_argument("--ssim-size", type=int, default=64)
    ap.add_argument("--limit-ssim", type=int, default=0,
                    help="debug: SSIM for first N pairs only (0=all)")
    args = ap.parse_args()

    root, how, _searched = audit.discover_dataset_root(args.root)
    if root is None:
        print("dataset root not found", file=sys.stderr)
        return 2
    # Image paths in the manifest are relative to the deepest common image
    # root (e.g. .../versions/1/ultrasound breast classification/), NOT to the
    # dataset root itself. Phase 1's resolve_image_root recomputes it.
    common_root, _rels = audit.resolve_image_root(root)

    # Load the committed Phase 1 manifest (frozen evidence; never rewritten).
    manifest_path = os.path.join(args.manifests_dir, "dataset_manifest.csv")
    with open(manifest_path, encoding="utf-8", newline="") as f:
        records = list(csv.DictReader(f))
    for r in records:
        r["near_dup_candidate"] = "False"
        r["dhash_to_original"] = ""
        r["role"] = r["role_confidence"] = r["role_rule"] = ""
        r["grouping_basis"] = ""
        r["ungrouped_low_confidence"] = "False"
        # manifest CSV fields arrive as strings; normalize numeric fields
        for f in ("width", "height", "bytes"):
            r[f] = int(r[f]) if r[f] else None

    print(f"Phase 2 grouping over {len(records)} manifest rows "
          f"(dataset root via {how})", flush=True)

    # -- grouping over confirmed identity ----------------------------------
    groups, merge = group_images(records)

    # -- near-duplicate candidate refresh (bit-identical to Phase 1) -------
    ok = [r for r in records if r["dhash_hex"]]
    idx_of = {r["path"]: i for i, r in enumerate(ok)}
    hashes = [int(r["dhash_hex"], 16) for r in ok]
    # audit.compute_candidate_pairs operates on the raw manifest row dicts and
    # returns index pairs into the dhash-bearing subsequence `ok`.
    phase1_pairs = [(ok[i]["path"], ok[j]["path"], d)
                    for i, j, d in audit.compute_candidate_pairs(records)]
    p1_set = {tuple(sorted((pa, pb))) for pa, pb, _d in phase1_pairs}
    for pa, pb, _d in phase1_pairs:
        records[idx_of[pa]]["near_dup_candidate"] = "True"
        records[idx_of[pb]]["near_dup_candidate"] = "True"

    # New dHash couples INSIDE one source group missed by Phase 1's pair list.
    new_couples: list[tuple[str, str, int]] = []
    for gid in sorted(groups):
        g = groups[gid]
        gi = [idx_of[r["path"]] for r in g if r["path"] in idx_of]
        for x in range(len(gi)):
            for y in range(x + 1, len(gi)):
                i, j = sorted((gi[x], gi[y]))
                d = (hashes[i] ^ hashes[j]).bit_count()
                pa, pb = sorted((ok[i]["path"], ok[j]["path"]))
                if d <= audit.NEARDUP_THRESHOLD and (pa, pb) not in p1_set:
                    new_couples.append((pa, pb, d))
                    records[idx_of[pa]]["near_dup_candidate"] = "True"
                    records[idx_of[pb]]["near_dup_candidate"] = "True"
    new_couples.sort(key=lambda t: (t[2], t[0], t[1]))

    # -- roles / flags / dhash-to-original evidence ------------------------
    for gid in sorted(groups):
        g = groups[gid]
        recover_roles(g)
        bases = [r for r in g if r["role"] == ROLE_BASE
                 and r["role_confidence"] == "high"]
        base0 = bases[0] if bases else None
        for r in g:
            if r["role"] == ROLE_AUG and base0 is not None:
                d = (int(r["dhash_hex"], 16)
                     ^ int(base0["dhash_hex"], 16)).bit_count()
                r["dhash_to_original"] = str(d)
            r["grouping_basis"] = ("filename_source_key"
                                   if r["source_key"] else "exact_md5_only")
            if r["grouping_basis"] == "exact_md5_only" and len(g) == 1:
                r["ungrouped_low_confidence"] = "True"

    ssim_pairs = phase1_pairs + new_couples
    if args.limit_ssim:
        ssim_pairs = ssim_pairs[:args.limit_ssim]
    ssim_csv = os.path.join(args.manifests_dir, "ssim_crosscheck.csv")
    print(f"SSIM cross-validation over {len(ssim_pairs)} candidate pairs "
          f"(size {args.ssim_size})...", flush=True)
    ssim_stats = crosscheck_ssim(ssim_pairs, common_root, ssim_csv,
                                 size=args.ssim_size)

    # -- outputs ------------------------------------------------------------
    groups_csv = os.path.join(args.manifests_dir, "augmentation_groups.csv")
    write_augmentation_groups_csv(groups, groups_csv)
    summary = summarize(groups, len(records), phase1_pairs, new_couples,
                        ssim_stats, merge)
    summary_path = os.path.join(args.manifests_dir,
                                "phase2_grouping_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({k: summary[k] for k in (
        "counts", "merge_edges", "label_conflicts", "roles",
        "near_duplicate_candidates", "ungrouped_low_confidence")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
