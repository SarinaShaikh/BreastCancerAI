"""Phase 1 dataset audit for 'Ultrasound Breast Images for Breast Cancer' (Kaggle).

Authoritative roadmap: PROGRESS.md (Phase 1). Path mapping: decision D-1
(reports/phase0_decisions.md section 1). Raw data is treated as immutable
evidence: this script NEVER writes inside the dataset root and NEVER modifies,
renames, converts or deletes raw files.

Design notes
------------
- Every reported number is computed from the files on disk; nothing is
  hard-coded. Counts, hashes and dimensions come from an EXHAUSTIVE pass over
  all discovered image files (no sampling anywhere in v1.1.0).
- The filename grammar is DISCOVERED by inspection, then applied and reported;
  files that do not parse are listed, not silently dropped.
- Identity basis: the raw filename stem equals a canonical augmentation chain,
  so (source_key, chain) equality across the pre-existing splits is
  content-identity evidence at the filename layer. md5 is recorded for every
  file and used as independent corroboration.
- Perceptual hashing (dHash, 64-bit) generates NEAR-DUPLICATE CANDIDATES only.
  The chunked index is complete for Hamming distance <= 7; candidates are
  never automatically confirmed duplicates; raw data is untouched.
- MRI/cross-modal status is determined by a filename/directory-name scan
  (D-2 vocabulary, reports/phase0_decisions.md section 2). Absence of MRI
  files is evidence about files only, not about patients (recorded limitation).
- Outputs (all under data/manifests/ by default):
    dataset_manifest.csv          one row per image file, 100% coverage
    near_duplicate_candidates.csv candidate pairs with Hamming distances
    source_key_overlap.csv        source keys present in >1 pre-existing split
    dataset_source.json           dataset identity/provenance record
    audit_summary.json            full machine-readable audit summary

Usage
-----
    python src/data/audit.py [--root DIR] [--manifests-dir DIR] [--json OUT.json]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # dataset images are small; no decompression-bomb risk

AUDIT_VERSION = "1.1.2"

# Canonical dataset handle (roadmap Dataset section / kagglehub handle).
KAGGLE_HANDLE = "vuppalaadithyasairam/ultrasound-breast-images-for-breast-cancer"
KAGGLE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_HANDLE}"

# ---------------------------------------------------------------------------
# Filename grammar (discovered 2026-09-16 during Phase 1 inspection and
# verified against 100% of files; see reports/phase1_dataset_audit.md):
#   "<class> (<idx>)[-<op1>[-<op2>[...]]].<ext>"  with op in
#   {rotated1, rotated2, rotated32, sharpened}
# ---------------------------------------------------------------------------
NAME_RE = re.compile(r"^(?P<cls>[A-Za-z]+) \((?P<idx>\d+)\)(?P<chain>(?:-[A-Za-z0-9]+)*)$")
KNOWN_OPS = ("rotated1", "rotated2", "rotated32", "sharpened")

IMG_EXTS = {".png", ".jpg", ".jpeg"}
# MRI/cross-modal scan vocabulary (D-2; see audit_mri_status docstring).
MRI_HINTS = ("mri", "mr_", "_mr", "-mr", "dce", "t1", "t2", "flair", "dicom")

# Near-duplicate detection: 64-bit dHash split into 8 chunks of 8 bits.
# Indexing by chunk identity is COMPLETE for Hamming distance <= 7 (pigeonhole:
# 7 differing bits across 8 chunks leave at least one chunk intact).
NEARDUP_BITS = 64
NEARDUP_CHUNKS = 8
NEARDUP_THRESHOLD = 7

MANIFEST_FIELDS = [
    "path", "split", "class_dir", "ext", "bytes", "read_ok",
    "width", "height", "channels", "mode", "format", "bit_depth",
    "md5", "sha256", "dhash_hex",
    "source_key", "chain", "lineage_class", "source_key_status",
    "duplicate_group", "near_duplicate_group", "stem_parsed",
]
SOURCE_KEY_STATUS = "inferred_from_filename_not_verified_identifier"


# ---------------------------------------------------------------------------
# Hashing helpers (streaming; deterministic)
# ---------------------------------------------------------------------------

def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def md5_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dhash64(im: Image.Image) -> int:
    """64-bit difference hash: grayscale, resize to 9x8, horizontal gradients."""
    g = im.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    px = g.tobytes()  # row-major bytes, mode 'L' -> one byte per pixel
    bits = 0
    for row in range(8):
        base = row * 9
        for col in range(8):
            bits = (bits << 1) | (1 if px[base + col] > px[base + col + 1] else 0)
    return bits


def dhash_hex(v: int) -> str:
    return f"{v:016x}"


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# ---------------------------------------------------------------------------
# Dataset root discovery (Issue A fix: supports arg, env, repo-local cache)
# ---------------------------------------------------------------------------

def _kagglehub_versions_dirs(base: str) -> list[str]:
    """Return sorted existing version dirs for KAGGLE_HANDLE under base, if any."""
    cand = os.path.join(base, "datasets", *KAGGLE_HANDLE.split("/"), "versions")
    if not os.path.isdir(cand):
        return []
    return sorted(os.path.join(cand, d) for d in os.listdir(cand)
                  if os.path.isdir(os.path.join(cand, d)))


def discover_dataset_root(explicit: str | None) -> tuple[str | None, str, list[str]]:
    """Return (root, how_found, searched).

    Preference order: --root arg > BREASTCANCERAI_DATASET_ROOT env var >
    repo-local kagglehub cache (dataset/raw/.kagglehub_cache, per decision D-1)
    > user-home kagglehub cache. No machine-specific absolute paths are
    hard-coded; repo-local paths are derived from this file's location.
    """
    searched: list[str] = []
    if explicit:
        if os.path.isdir(explicit):
            return explicit, "--root argument", searched
        raise SystemExit(f"--root given but not a directory: {explicit}")

    env = os.environ.get("BREASTCANCERAI_DATASET_ROOT")
    if env:
        searched.append(f"env:{env}")
        if os.path.isdir(env):
            return env, "BREASTCANCERAI_DATASET_ROOT env var", searched

    repo_root = Path(__file__).resolve().parents[2]
    local_base = repo_root / "dataset" / "raw" / ".kagglehub_cache"
    for v in _kagglehub_versions_dirs(str(local_base)):
        searched.append(str(v))
        if any(Path(v).rglob("*")):  # non-empty
            return v, "repo-local kagglehub cache (dataset/raw/.kagglehub_cache)", searched

    home_base = os.path.join(os.path.expanduser("~"), ".cache", "kagglehub")
    for v in _kagglehub_versions_dirs(home_base):
        searched.append(str(v))
        if any(Path(v).rglob("*")):
            return v, "user-home kagglehub cache", searched

    return None, "not found", searched


def resolve_image_root(dataset_root: str) -> tuple[str, list[str]]:
    """Return (common_ancestor_dir, sorted relative image paths).

    v1.0.0 bug fix: the old find_image_root descended into the FIRST
    image-containing subdirectory (which here would be train/), silently
    auditing only one split. Instead we take every image file under the
    dataset root and compute the deepest common ancestor directory, so all
    splits are always included.
    """
    img_paths: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(dataset_root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                img_paths.append(os.path.join(dirpath, fn))
    if not img_paths:
        raise SystemExit(f"no image files found under dataset root: {dataset_root}")
    common = os.path.commonpath(img_paths)
    rels = sorted(os.path.relpath(p, common).replace("\\", "/") for p in img_paths)
    return common, rels


# ---------------------------------------------------------------------------
# Filename grammar
# ---------------------------------------------------------------------------

def parse_filename(stem: str) -> dict | None:
    m = NAME_RE.match(stem)
    if not m:
        return None
    chain = m.group("chain").lstrip("-").split("-") if m.group("chain") else []
    unknown = [op for op in chain if op not in KNOWN_OPS]
    return {
        "cls": m.group("cls").lower(),
        "idx": int(m.group("idx")),
        "source_key": f"{m.group('cls')} ({m.group('idx')})",
        "chain": tuple(chain),
        "unknown_ops": unknown,
        "depth": len(chain),
    }


def assign_lineage_labels(records: list[dict]) -> None:
    """confirmed  - chain non-empty (augmentation provenance explicit in name)
    strong     - chain empty AND at least one augmented variant of the same
                 source key exists in the audited pool
    candidate  - chain empty AND no augmented variant observed (base may live
                 outside the dataset; NOT evidence of absence)
    unparsed   - filename did not match the discovered grammar
    """
    keys_with_chain = {r["source_key"] for r in records
                       if r["chain"] not in (None, "")}
    for r in records:
        if r["chain"] is None:
            r["lineage_class"] = "unparsed"
        elif r["chain"]:
            r["lineage_class"] = "confirmed"
        else:
            r["lineage_class"] = "strong" if r["source_key"] in keys_with_chain else "candidate"


# ---------------------------------------------------------------------------
# Audit stages
# ---------------------------------------------------------------------------

def collect_records(common_root: str, rels: list[str], warnings: list[str]) -> list[dict]:
    records = []
    for rel in rels:
        full = os.path.join(common_root, *rel.split("/"))
        parts = rel.split("/")
        # Expected layout: <split>/<class>/<file> (3 components). Anything else
        # is recorded, never forced into the assumption.
        if len(parts) == 3:
            split, class_dir = parts[0], parts[1].lower()
        else:
            split, class_dir = "", parts[0].lower() if len(parts) == 2 else ""
            warnings.append(f"unexpected depth {len(parts)} (split derived as ''): {rel}")
        stem = os.path.splitext(parts[-1])[0]
        parsed = parse_filename(stem)
        if parsed is None:
            warnings.append(f"unparsed filename: {rel}")
        elif parsed["unknown_ops"]:
            warnings.append(f"unknown chain op {parsed['unknown_ops']} in {rel}")
        records.append({
            "path": rel,
            "full_path": full,
            "split": split,
            "class_dir": class_dir,
            "ext": os.path.splitext(parts[-1])[1].lower().lstrip("."),
            "bytes": os.path.getsize(full),
            "read_ok": None,
            "width": None, "height": None, "channels": None, "mode": None,
            "format": None, "bit_depth": None,
            "md5": None, "sha256": None, "dhash_hex": None,
            "stem_parsed": parsed is not None,
            "source_key": parsed["source_key"] if parsed else None,
            "chain": "|".join(parsed["chain"]) if parsed else None,
            "lineage_class": None,
            "source_key_status": SOURCE_KEY_STATUS if parsed else None,
            "duplicate_group": None,
            "near_duplicate_group": None,
        })
    return records


def exhaustive_pass(records: list[dict], warnings: list[str]) -> None:
    """Exhaustive for ALL files: md5 + sha256; decode + properties + dHash.

    No sampling: ~9k small images is cheap, and exact-duplicate detection must
    be exhaustive (Issue B fix).
    """
    mode_bits = {"L": 8, "P": 8, "I;16": 16, "I": 32, "RGB": 8, "RGBA": 8, "LA": 8}
    for r in records:
        full = r["full_path"]
        try:
            r["md5"] = md5_of(full)
            r["sha256"] = sha256_of(full)
            with Image.open(full) as im:
                im.load()
                r.update(width=im.width, height=im.height, mode=im.mode,
                         channels=len(im.getbands()), format=im.format,
                         bit_depth=mode_bits.get(im.mode, None))
                r["dhash_hex"] = dhash_hex(dhash64(im))
            r["read_ok"] = True
        except Exception as e:  # noqa: BLE001 - audit records, never crashes
            r["read_ok"] = False
            warnings.append(f"UNREADABLE {r['path']}: {type(e).__name__}: {e}")


def audit_duplicates(records: list[dict]) -> dict:
    """EXHAUSTIVE exact-duplicate detection over md5 (all files)."""
    exact: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["md5"]:
            exact[r["md5"]].append(r)
    groups = {h: rs for h, rs in exact.items() if len(rs) > 1}
    for h, rs in groups.items():
        gid = f"exact-{h[:12]}"
        for r in rs:
            r["duplicate_group"] = gid
    return {
        "method": "md5 over 100% of image files (exhaustive, no sampling)",
        "files_hashed": sum(1 for r in records if r["md5"]),
        "unique_md5": len(exact),
        "exact_duplicate_groups": len(groups),
        "files_in_exact_groups": sum(len(rs) for rs in groups.values()),
        "group_size_histogram": dict(Counter(len(rs) for rs in groups.values())),
        "cross_split_exact_groups": sum(
            1 for rs in groups.values() if len({r["split"] for r in rs}) > 1),
    }


def compute_candidate_pairs(records: list[dict]) -> list[tuple[int, int, int]]:
    """Deterministic (i, j, hamming) candidate pairs, i < j, d <= threshold.

    Single source of truth for BOTH the summary statistics and the CSV output
    (v1.1.0 regression fix: the first implementation had two divergent pair
    computations, and the chunked one emitted self-pairs (i, i) when an
    image's two chunk keys collided, inflating the count by 8). The chunk
    index (chunk_value | chunk_position as key) is complete for Hamming
    distance <= 7 by pigeonhole; self-pairs are excluded by construction.
    """
    ok = [r for r in records if r["dhash_hex"]]
    hashes = [int(r["dhash_hex"], 16) for r in ok]
    index: dict[int, list[int]] = defaultdict(list)
    for i, h in enumerate(hashes):
        for c in range(NEARDUP_CHUNKS):
            index[h & (0xFF << (8 * c)) | c].append(i)  # chunk value + position
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int, int]] = []
    for bucket in index.values():
        for x in range(len(bucket)):
            for y in range(x + 1, len(bucket)):
                i, j = bucket[x], bucket[y]
                if i == j:  # v1.1.0 regression fix: chunk-key collision self-pair
                    continue
                if i > j:
                    i, j = j, i
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                d = hamming(hashes[i], hashes[j])
                if d <= NEARDUP_THRESHOLD:
                    pairs.append((i, j, d))
    pairs.sort(key=lambda t: (t[2], t[0], t[1]))  # deterministic order
    return pairs


def audit_near_duplicates(records: list[dict],
                          pairs: list[tuple[int, int, int]]) -> dict:
    """dHash candidates. Candidates are NOT confirmed duplicates.

    Union-find over the candidate pairs yields candidate groups; group ids are
    stable digests of member md5s so repeated runs produce identical ids.
    """
    ok = [r for r in records if r["dhash_hex"]]
    parent = list(range(len(ok)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j, _d in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    comps: dict[int, list[int]] = defaultdict(list)
    for i in range(len(ok)):
        comps[find(i)].append(i)
    for root_idx, members in comps.items():
        if len(members) >= 2:
            key = "|".join(sorted(ok[m]["md5"] or ok[m]["path"] for m in members))
            gid = "near-" + hashlib.md5(key.encode()).hexdigest()[:12]
            for m in members:
                ok[m]["near_duplicate_group"] = gid

    dist_hist = Counter(d for _i, _j, d in pairs)
    group_list = [ms for ms in comps.values() if len(ms) >= 2]
    return {
        "method": "64-bit dHash, 8x8-bit chunk index, exact Hamming on candidates",
        "threshold_hamming": NEARDUP_THRESHOLD,
        "completeness": "complete for Hamming distance <= 7; pairs at exactly 8 "
                        "could in principle be missed (documented limitation)",
        "status": "CANDIDATES ONLY - not confirmed duplicates; raw data untouched",
        "images_hashed": len(ok),
        "candidate_pairs": len(pairs),
        "candidate_distance_histogram": {str(d): n for d, n in sorted(dist_hist.items())},
        "candidate_groups": len(group_list),
        "files_in_candidate_groups": sum(len(ms) for ms in group_list),
        "cross_split_candidate_groups": sum(
            1 for ms in group_list
            if len({ok[m]["split"] for m in ms}) > 1),
    }


def audit_lineage_and_split(records: list[dict]) -> dict:
    """Full-pool source-key indexing; cross-split overlap = leakage evidence."""
    keys = defaultdict(list)
    chains: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in records:
        if r["source_key"]:
            keys[r["source_key"]].append(r)
            chains[(r["source_key"], r["chain"] or "")].add(r["split"])
        else:
            keys[f"<unparsed>:{r['path']}"].append(r)

    shared = {k: v for k, v in keys.items()
              if not k.startswith("<unparsed>:")
              and len({r["split"] for r in v}) > 1}
    shared_detail = []
    for k in sorted(shared):
        files = sorted(shared[k], key=lambda r: r["path"])
        shared_chains = sorted(c for (key, c), sp in chains.items()
                               if key == k and len(sp) > 1)
        shared_detail.append({
            "source_key": k,
            "splits": sorted({r["split"] for r in files}),
            "train_files": [r["path"] for r in files if r["split"] == "train"],
            "val_files": [r["path"] for r in files if r["split"] == "val"],
            "shared_chains_across_splits": shared_chains,
            "has_exact_duplicate_member": any(
                r["duplicate_group"] for r in files),
            "exact_duplicate_member_files": [r["path"] for r in files
                                             if r["duplicate_group"]],
        })

    per_key_counts = {k: len(v) for k, v in keys.items() if not k.startswith("<unparsed>:")}
    bases = {k for k in per_key_counts
             if any(r["chain"] == "" for r in keys[k])}
    return {
        "identity_basis": "raw filename stem == canonical augmentation chain; md5 "
                          "recorded as corroboration. Source keys are FILENAME-DERIVED "
                          "and are NOT verified patient/study identifiers.",
        "source_key_status": SOURCE_KEY_STATUS,
        "distinct_source_keys": len(per_key_counts),
        "source_keys_by_split": {
            s: len({r["source_key"] for r in records
                    if r["split"] == s and r["source_key"]})
            for s in sorted({r["split"] for r in records})},
        "source_keys_with_base_variant_in_pool": len(bases),
        "source_keys_without_base_variant": len(per_key_counts) - len(bases),
        "cross_split_identity_groups": sum(
            1 for sp in chains.values() if len(sp) > 1),
        "shared_source_keys": len(shared),
        "shared_source_key_detail": shared_detail,
        "lineage_class_counts": dict(Counter(r["lineage_class"] for r in records)),
    }


def audit_mri_status(dataset_root: str) -> dict:
    """D-2 classification from the FILE layer only.

    Scans every file and directory name for MRI/DICOM vocabulary. Absence of
    hits establishes the absence of MRI FILES in this download; it cannot
    establish anything about patients or pairing in the real world (limitation
    recorded in the Phase 1 report).
    """
    hits = []
    for dirpath, dirnames, filenames in os.walk(dataset_root):
        for name in list(dirnames) + list(filenames):
            low = name.lower()
            if low.endswith(".dcm") or any(h in low for h in MRI_HINTS):
                hits.append(os.path.relpath(os.path.join(dirpath, name),
                                            dataset_root).replace("\\", "/"))
    if hits:
        state = ("B (MRI-suggestive files found; pairing/linkage UNVERIFIED -> "
                 "NOT T-1 Paired Cross-Modal Validation)")
        t_label = "not T-1 (T-2/T-3 at most, and only with explicit approval)"
    else:
        state = ("C (no MRI files found in the primary dataset -> T-5 Absence of "
                 "MRI validation)")
        t_label = "T-5"
    return {
        "scan_scope": "all file and directory names under the dataset root",
        "vocabulary": list(MRI_HINTS) + ["*.dcm"],
        "mri_hint_files": hits,
        "d2_evidence_state": state,
        "d2_label": t_label,
        "caveat": "File-layer evidence only; establishes no patient-level claims "
                  "in either direction (D-2, reports/phase0_decisions.md section 2).",
    }


def dataset_fingerprint(records: list[dict]) -> str:
    """Stable digest over (path, size, md5) for all image files."""
    h = hashlib.sha256()
    for r in sorted(records, key=lambda x: x["path"]):
        h.update(f"{r['path']}:{r['bytes']}:{r['md5'] or 'NA'}\n".encode())
    return h.hexdigest()


def build_provenance(dataset_root: str, records: list[dict]) -> dict:
    """Dataset identity record (Phase 1 task: version/date/checksum info)."""
    marker = Path(dataset_root).parent / "1.complete"
    completed_at = None
    if marker.exists():
        completed_at = datetime.fromtimestamp(
            marker.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
    return {
        "kaggle_handle": KAGGLE_HANDLE,
        "kaggle_url": KAGGLE_URL,
        "version_dir": os.path.basename(dataset_root),
        "dataset_root_relative": "dataset/raw/.kagglehub_cache/datasets/"
                                 + KAGGLE_HANDLE + "/versions/"
                                 + os.path.basename(dataset_root),
        "acquisition_note": "kagglehub anonymous download of a PUBLIC dataset "
                            "(no credentials used or required; no access controls "
                            "bypassed); recorded by the Phase 1 session",
        "download_completed_at_utc_marker_mtime": completed_at,
        "marker_mtime_caveat": "filesystem mtime of the kagglehub '1.complete' "
                               "marker; used as acquisition-time evidence, not a "
                               "Kaggle-published timestamp",
        "image_file_count": len(records),
        "total_image_bytes": sum(r["bytes"] for r in records),
        "manifest_digest_sha256": dataset_fingerprint(records),
        "digest_definition": "sha256 over 'path:bytes:md5\\n' lines, sorted by path",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "audit_version": AUDIT_VERSION,
    }


def write_outputs(records: list[dict], lineage: dict,
                  pairs: list[tuple[int, int, int]],
                  manifests_dir: str) -> list[str]:
    os.makedirs(manifests_dir, exist_ok=True)
    written = []

    # 1. dataset_manifest.csv - one row per image, 100% coverage
    p = os.path.join(manifests_dir, "dataset_manifest.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(records, key=lambda x: x["path"]):
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in MANIFEST_FIELDS})
    written.append(p)

    # 2. near_duplicate_candidates.csv - candidate pairs with distances
    # (single pair computation shared with the summary: see compute_candidate_pairs)
    ok = [r for r in records if r["dhash_hex"]]
    p = os.path.join(manifests_dir, "near_duplicate_candidates.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file_a", "file_b", "hamming",
                                          "same_split", "same_source_key"])
        w.writeheader()
        for i, j, d in pairs:  # pairs are already deterministically ordered
            w.writerow({"file_a": ok[i]["path"], "file_b": ok[j]["path"],
                        "hamming": d,
                        "same_split": ok[i]["split"] == ok[j]["split"],
                        "same_source_key": ok[i]["source_key"] == ok[j]["source_key"]})
    written.append(p)

    # 3. source_key_overlap.csv - keys present in >1 pre-existing split
    p = os.path.join(manifests_dir, "source_key_overlap.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source_key", "splits", "train_files",
                                          "val_files", "shared_chains_across_splits",
                                          "has_exact_duplicate_member"])
        w.writeheader()
        for d in lineage["shared_source_key_detail"]:
            w.writerow({
                "source_key": d["source_key"],
                "splits": "|".join(d["splits"]),
                "train_files": "|".join(d["train_files"]),
                "val_files": "|".join(d["val_files"]),
                "shared_chains_across_splits": "|".join(d["shared_chains_across_splits"]),
                "has_exact_duplicate_member": str(d["has_exact_duplicate_member"]).lower(),
            })
    written.append(p)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 1 dataset audit (PROGRESS.md; D-1 paths)")
    ap.add_argument("--root", default=None,
                    help="dataset root; default: env var, repo-local kagglehub "
                         "cache, then user-home cache")
    ap.add_argument("--manifests-dir", default="data/manifests")
    ap.add_argument("--json", default="data/manifests/audit_summary.json")
    args = ap.parse_args()

    root, how, searched = discover_dataset_root(args.root)
    if root is None:
        print(json.dumps({"audit_version": AUDIT_VERSION, "dataset_available": False,
                          "root_search_order": searched}, indent=2))
        return 2

    warnings: list[str] = []
    common_root, rels = resolve_image_root(root)
    records = collect_records(common_root, rels, warnings)
    exhaustive_pass(records, warnings)
    assign_lineage_labels(records)
    dups = audit_duplicates(records)
    pairs = compute_candidate_pairs(records)
    near = audit_near_duplicates(records, pairs)
    lineage = audit_lineage_and_split(records)
    mri = audit_mri_status(root)

    # Record machine-independent paths in committed artifacts (privacy fix):
    # if the dataset root lies inside the repository, store it repo-relative;
    # never persist usernames or absolute local paths.
    repo_root = Path(__file__).resolve().parents[2]
    try:
        root_rel = Path(root).resolve().relative_to(repo_root).as_posix()
        common_rel = Path(common_root).resolve().relative_to(repo_root).as_posix()
    except ValueError:  # root outside the repo (e.g. user-home cache)
        root_rel = f"<outside-repo>:{Path(root).resolve().as_posix()}"
        common_rel = f"<outside-repo>:{Path(common_root).resolve().as_posix()}"

    splits = sorted({r["split"] for r in records})
    classes = sorted({r["class_dir"] for r in records})
    per_split_cls = Counter((r["split"], r["class_dir"]) for r in records)
    dims = Counter((r["width"], r["height"]) for r in records)
    corrupt = [r["path"] for r in records if r["read_ok"] is False]
    unparsed = [r["path"] for r in records if not r["stem_parsed"]]

    summary = {
        "audit_version": AUDIT_VERSION,
        "scan_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_handle": KAGGLE_HANDLE,
        "dataset_root": root_rel,
        "image_common_root": common_rel,
        "root_found_via": how,
        "raw_data_immutable": True,
        "coverage": "EXHAUSTIVE: every image file hashed (md5+sha256) and decoded "
                    "(properties + dHash); no sampling",
        "structure": {
            "discovered_splits": splits,
            "discovered_class_dirs": classes,
            "split_is_directory_encoded": True,
        },
        "counts": {
            "total_image_files": len(records),
            "per_split_class": {f"{s}/{c}": n for (s, c), n in sorted(per_split_cls.items())},
            "by_extension": dict(Counter(r["ext"] for r in records)),
            "unreadable": len(corrupt),
            "unreadable_files": corrupt,
            "unparsed_filenames": len(unparsed),
        },
        "image_properties": {
            "distinct_dimensions": len(dims),
            "top_dimensions": [{"w": w, "h": h, "n": n} for (w, h), n in dims.most_common(10)],
            "modes": dict(Counter(r["mode"] for r in records)),
            "formats": dict(Counter(r["format"] for r in records)),
            "channel_counts": dict(Counter(r["channels"] for r in records)),
        },
        "filename_grammar": {
            "regex": NAME_RE.pattern,
            "known_ops": list(KNOWN_OPS),
            "unparsed_count": len(unparsed),
            "unparsed_examples": unparsed[:20],
        },
        "duplicates": dups,
        "near_duplicates": near,
        "lineage": lineage,
        "preexisting_split": {
            "detected": len(splits) > 1,
            "splits": splits,
            "creation_documentation_found": False,
            "source_keys_shared_across_splits": lineage["shared_source_keys"],
            "shared_source_key_detail": lineage["shared_source_key_detail"],
        },
        "mri": mri,
        "manifest_digest_sha256": dataset_fingerprint(records),
        "warnings": warnings[:100],
        "warning_count": len(warnings),
        "outputs": write_outputs(records, lineage, pairs, args.manifests_dir),
    }

    prov = build_provenance(root, records)
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    prov_path = os.path.join(args.manifests_dir, "dataset_source.json")
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
    summary["outputs"].append(prov_path)

    print(json.dumps({k: summary[k] for k in (
        "audit_version", "coverage", "counts", "image_properties", "duplicates",
        "preexisting_split", "mri", "manifest_digest_sha256", "warning_count")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
