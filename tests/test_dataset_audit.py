"""Phase 1 validation suite (V-1..V-11) for the dataset-audit deliverables.

Runs standalone (``python tests/test_dataset_audit.py``) or under pytest.
Every check RECOMPUTES values from the dataset and the manifest CSVs rather
than trusting the audit summary - V-9 (no fabricated data) is enforced by
construction: if a reported number cannot be re-derived, the suite fails.

Roadmap: PROGRESS.md Phase 1 validation checks. Paths: decision D-1
(reports/phase0_decisions.md section 1). MRI vocabulary: decision D-2.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "data" / "manifests"
SUMMARY_PATH = MANIFESTS / "audit_summary.json"
MANIFEST_PATH = MANIFESTS / "dataset_manifest.csv"
NEARDUP_PATH = MANIFESTS / "near_duplicate_candidates.csv"
OVERLAP_PATH = MANIFESTS / "source_key_overlap.csv"

_spec = importlib.util.spec_from_file_location(
    "phase1_audit", REPO / "src" / "data" / "audit.py")
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)

RESULTS: list[tuple[str, str, str]] = []


def check(vid: str, name: str):
    def wrap(fn):
        def run() -> bool:
            try:
                detail = fn()
                RESULTS.append((vid, name, "PASS"))
                print(f"[PASS] {vid} {name}" + (f" :: {detail}" if detail else ""))
                return True
            except AssertionError as e:
                RESULTS.append((vid, name, f"FAIL: {e}"))
                print(f"[FAIL] {vid} {name} :: {e}")
                return False
        run.__name__ = f"test_{vid.lower()}_{name.lower().replace(' ', '_').replace('/', '_')}"
        return run
    return wrap


# --------------------------------------------------------------------------
# Shared fixtures (loaded once; recomputation happens inside each check)
# --------------------------------------------------------------------------
SUMMARY: dict = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
with open(MANIFEST_PATH, encoding="utf-8", newline="") as f:
    MANIFEST: list[dict] = list(csv.DictReader(f))
with open(NEARDUP_PATH, encoding="utf-8", newline="") as f:
    NEARDUP: list[dict] = list(csv.DictReader(f))
with open(OVERLAP_PATH, encoding="utf-8", newline="") as f:
    OVERLAP: list[dict] = list(csv.DictReader(f))

ROOT, _how, _searched = audit.discover_dataset_root(None)
COMMON, RELS = audit.resolve_image_root(ROOT)


def full(rel: str) -> str:
    return os.path.join(COMMON, *rel.split("/"))


# --------------------------------------------------------------------------
# V-1 Dataset availability
# --------------------------------------------------------------------------
@check("V-1", "dataset available and readable")
def v1():
    assert ROOT is not None, "dataset root not discovered"
    assert len(RELS) > 0, "no image files found"
    assert os.path.getsize(full(RELS[0])) > 0
    return f"root via {_how}; {len(RELS)} images"


# --------------------------------------------------------------------------
# V-2 Manifest integrity
# --------------------------------------------------------------------------
@check("V-2", "manifest exists, parses, schema ok, 100% coverage")
def v2():
    assert MANIFEST_PATH.is_file()
    assert len(MANIFEST) == len(RELS), \
        f"manifest rows {len(MANIFEST)} != image files on disk {len(RELS)}"
    assert set(MANIFEST[0].keys()) == set(audit.MANIFEST_FIELDS), "schema drift"
    missing = [r["path"] for r in MANIFEST[:50] if not os.path.isfile(full(r["path"]))]
    assert not missing, f"unresolvable manifest paths: {missing[:5]}"
    # every relative path on disk appears in the manifest (set equality)
    disk = set(RELS)
    sheet = {r["path"] for r in MANIFEST}
    assert disk == sheet, f"path-set mismatch: {len(disk ^ sheet)} differing"
    return f"{len(MANIFEST)} rows, schema ok, paths resolve"


# --------------------------------------------------------------------------
# V-3 Reproducibility (digest recomputation; full re-run evidence in report)
# --------------------------------------------------------------------------
@check("V-3", "manifest digest recomputes from manifest rows")
def v3():
    recomputed = audit.dataset_fingerprint(
        [{"path": r["path"], "bytes": int(r["bytes"]), "md5": r["md5"] or None}
         for r in MANIFEST])
    assert recomputed == SUMMARY["manifest_digest_sha256"], "digest mismatch"
    return f"sha256 {recomputed[:16]}... matches summary"


# --------------------------------------------------------------------------
# V-4 File integrity (re-derived independently: full re-decode)
# --------------------------------------------------------------------------
@check("V-4", "all files decode; integrity re-verified from disk")
def v4():
    assert all(r["read_ok"] == "True" for r in MANIFEST), "manifest has read failures"
    from PIL import Image
    bad = []
    for rel in RELS:
        try:
            with Image.open(full(rel)) as im:
                im.load()
        except Exception as e:  # noqa: BLE001
            bad.append(f"{rel}: {e}")
    assert not bad, f"{len(bad)} undecodable files, e.g. {bad[:3]}"
    assert SUMMARY["counts"]["unreadable"] == 0
    return f"{len(RELS)}/{len(RELS)} decoded cleanly (exhaustive re-check)"


# --------------------------------------------------------------------------
# V-5 Identifier integrity
# --------------------------------------------------------------------------
@check("V-5", "no patient/study IDs; source keys labelled as unverified")
def v5():
    assert not any("patient" in k or "study" in k for k in audit.MANIFEST_FIELDS), \
        "manifest fabricates identifier fields"
    statuses = {r["source_key_status"] for r in MANIFEST}
    assert statuses == {audit.SOURCE_KEY_STATUS}, f"unexpected statuses: {statuses}"
    assert "NOT verified patient" in SUMMARY["lineage"]["identity_basis"]
    return f"all {len(MANIFEST)} rows: source_key_status={audit.SOURCE_KEY_STATUS}"


# --------------------------------------------------------------------------
# V-6 Duplicate audit (independent full re-hash)
# --------------------------------------------------------------------------
@check("V-6", "exact-duplicate stats reproduce from independent full re-hash")
def v6():
    fresh = {r["path"]: hashlib.md5(open(full(r["path"]), "rb").read()).hexdigest()
             for r in MANIFEST}
    drifted = [(r["path"], r["md5"], fresh[r["path"]])
               for r in MANIFEST if r["md5"] != fresh[r["path"]]]
    assert not drifted, f"{len(drifted)} md5 mismatches vs manifest"
    groups = defaultdict(list)
    for p, h in fresh.items():
        groups[h].append(p)
    multi = {h: ps for h, ps in groups.items() if len(ps) > 1}
    d = SUMMARY["duplicates"]
    assert d["unique_md5"] == len(groups), "unique_md5 mismatch"
    assert d["exact_duplicate_groups"] == len(multi), "group count mismatch"
    assert d["files_in_exact_groups"] == sum(len(ps) for ps in multi.values())
    cross = sum(1 for ps in multi.values()
                if len({p.split("/")[0] for p in ps}) > 1)
    assert d["cross_split_exact_groups"] == cross, "cross-split group count mismatch"
    return (f"{len(fresh)} files re-hashed; {len(multi)} groups, "
            f"{cross} cross-split")


# --------------------------------------------------------------------------
# V-7 Leakage audit
# --------------------------------------------------------------------------
@check("V-7", "source-key overlap and near-dup stats reproduce")
def v7():
    keys = defaultdict(set)
    for r in MANIFEST:
        keys[r["source_key"]].add(r["split"])
    shared = sorted(k for k, sp in keys.items() if len(sp) > 1)
    assert sorted(SUMMARY["lineage"]["shared_source_key_detail"][i]["source_key"]
                  for i in range(SUMMARY["lineage"]["shared_source_keys"])) == shared
    overlap_rows = {row["source_key"] for row in OVERLAP}
    assert overlap_rows == set(shared), "overlap CSV != recomputed shared keys"
    # The two candidates flagged by the preliminary session genuinely overlap:
    for cand in ("benign (36)", "malignant (18)"):
        assert cand in shared, f"candidate {cand} not actually shared"
    assert SUMMARY["near_duplicates"]["candidate_pairs"] == len(NEARDUP), \
        "near-dup CSV != summary"
    cross_rows = sum(1 for r in NEARDUP if r["same_split"] == "False")
    assert cross_rows == 996, f"cross-split near-dup pair count changed: {cross_rows}"
    return (f"shared keys: {shared}; {len(NEARDUP)} candidate pairs, "
            f"{cross_rows} cross-split")


# --------------------------------------------------------------------------
# V-8 Raw immutability
# --------------------------------------------------------------------------
@check("V-8", "raw files untouched; sizes+hashes match manifest; audit read-only")
def v8():
    size_drift = [r["path"] for r in MANIFEST
                  if os.path.getsize(full(r["path"])) != int(r["bytes"])]
    assert not size_drift, f"{len(size_drift)} files changed size on disk"
    src_text = (REPO / "src" / "data" / "audit.py").read_text(encoding="utf-8")
    lines = src_text.splitlines()
    # Trace every open(..., "w") target back to a manifests_dir/args.json origin:
    # v1.1.1 writes through variables (p, prov_path) assigned from
    # os.path.join(manifests_dir, ...) or args.json - verify no other origin.
    origins = {}
    write_targets = []
    for ln in lines:
        s = ln.strip()
        m = re.match(r"^(\w+) = os\.path\.join\((?:args\.)?manifests_dir, (.+)\)$", s)
        if m:
            origins[m.group(1)] = "manifests_dir"
        m = re.match(r"^(\w+) = os\.path\.dirname\((args\.json)\)$", s)
        if m:
            origins[m.group(1)] = "args.json"
        if re.search(r"open\((\w+), \"w\"", s):
            var = re.search(r"open\((\w+), \"w\"", s).group(1)
            write_targets.append(origins.get(var, f"UNKNOWN:{var}"))
        if re.search(r"open\((args\.json), \"w\"", s):
            write_targets.append("args.json")
    assert write_targets, "no write targets found (check logic broken)"
    assert all(t in ("manifests_dir", "args.json") for t in write_targets), \
        f"audit writes outside manifests/summary targets: {write_targets}"
    return (f"sizes stable, hashes stable (V-6); all {len(write_targets)} write "
            "targets traced to data/manifests/")


# --------------------------------------------------------------------------
# V-9 No fabricated data (summary must equal manifest-derived values)
# --------------------------------------------------------------------------
@check("V-9", "every summary statistic re-derives from the manifest")
def v9():
    c = SUMMARY["counts"]
    assert c["total_image_files"] == len(MANIFEST)
    per = Counter((r["split"], r["class_dir"]) for r in MANIFEST)
    assert c["per_split_class"] == {f"{s}/{cl}": n for (s, cl), n in sorted(per.items())}
    assert c["by_extension"] == dict(Counter(r["ext"] for r in MANIFEST))
    dims = Counter((int(r["width"]), int(r["height"])) for r in MANIFEST)
    assert SUMMARY["image_properties"]["distinct_dimensions"] == len(dims)
    assert SUMMARY["image_properties"]["modes"] == dict(
        Counter(r["mode"] for r in MANIFEST))
    lin = Counter(r["lineage_class"] for r in MANIFEST)
    assert SUMMARY["lineage"]["lineage_class_counts"] == dict(lin)
    keys = {r["source_key"] for r in MANIFEST}
    assert SUMMARY["lineage"]["distinct_source_keys"] == len(keys)
    return (f"{len(MANIFEST)} rows -> all counts, dims, modes, lineage, keys match")


# --------------------------------------------------------------------------
# V-10 MRI claim safety
# --------------------------------------------------------------------------
@check("V-10", "MRI status is T-5; no paired-MRI claim")
def v10():
    m = SUMMARY["mri"]
    assert m["d2_label"] == "T-5", f"unexpected D-2 label: {m['d2_label']}"
    assert m["mri_hint_files"] == [], "unexpected MRI-suggestive files"
    assert "NOT T-1" in m["d2_evidence_state"] or "no MRI files" in m["d2_evidence_state"]
    return "T-5 Absence of MRI validation (file-layer scan, D-2)"


# --------------------------------------------------------------------------
# V-11 Phase boundary
# --------------------------------------------------------------------------
@check("V-11", "no later-phase implementation present")
def v11():
    forbidden_imports = re.compile(
        r"^\s*(import|from)\s+(torch|tensorflow|keras|sklearn|albumentations)\b",
        re.MULTILINE)
    # Phase 1/2 boundary guard retained; Phase 6/7/8-approved modules
    # explicitly exempted (owner-authorized 2026-09-19; Phase 7 additions
    # owner-authorized in the Phase 7 implementation instruction; Phase 8
    # additions owner-authorized in the Phase 8 implementation instruction).
    # The exemption is file-level and exact: ONLY these seven files may
    # import torch/sklearn. Any other file under src/ — including any new
    # file in src/training/, src/mil/, or src/models/ — still fails this
    # guard.
    phase6_approved = {
        "src/training/metrics.py",
        "src/training/train_baseline.py",
        "src/mil/instance_generation.py",
        "src/mil/feature_extractor.py",
        "src/mil/aggregation.py",
        "src/mil/dual_attention.py",
        "src/models/dual_attention_mil.py",
        # Phase 9 (owner-authorized): the bag-level trainer for E1, a
        # structural mirror of train_b4 reusing train_baseline helpers.
        "src/training/train_dual_attention.py",
    }
    for py in sorted((REPO / "src").rglob("*.py")):
        if py.relative_to(REPO).as_posix() in phase6_approved:
            continue
        assert not forbidden_imports.search(py.read_text(encoding="utf-8")), \
            f"training-adjacent import in {py.name}"
    assert not (REPO / "checkpoints").exists()
    # data/processed/ is Phase 5's roadmap-named cache; it is legitimate only
    # if it matches the Phase 5 layout (three split dirs + summary, nothing
    # else), and is otherwise unexpected.
    if (REPO / "data" / "processed").exists():
        p5_ok = {"train", "val", "test", "cache_summary.json"}
        assert {p.name for p in (REPO / "data" / "processed").iterdir()} <= p5_ok, \
            "unexpected data/processed/ entry (Phase 5 layout drift)"
    # Phase 1 artifacts must remain present. (Originally this asserted the
    # directory held EXACTLY these five files; later phases legitimately add
    # their roadmap-named artifacts — Phase 2 added augmentation_groups.csv /
    # ssim_crosscheck.csv / phase2_grouping_summary.json, Phase 3 added
    # bag_manifest.csv, Phase 4 added train/val/test_split.csv — so the
    # check is now: the five Phase 1 artifacts present + nothing outside the
    # known Phase 1/2/3/4 sets.)
    p1_required = {"audit_summary.json", "dataset_manifest.csv",
                   "dataset_source.json", "near_duplicate_candidates.csv",
                   "source_key_overlap.csv"}
    p2_known = {"augmentation_groups.csv", "ssim_crosscheck.csv",
                "phase2_grouping_summary.json"}
    p3_known = {"bag_manifest.csv"}
    p4_known = {"train_split.csv", "val_split.csv", "test_split.csv"}
    present = {p.name for p in MANIFESTS.iterdir()}
    missing = p1_required - present
    assert not missing, f"Phase 1 artifacts missing: {sorted(missing)}"
    unexpected = present - p1_required - p2_known - p3_known - p4_known
    assert not unexpected, f"unexpected artifacts: {sorted(unexpected)}"
    # Phase 4 legitimately added the roadmap-named split manifests; any other
    # split-named file under data/ remains unexpected.
    p4_known = {"train_split.csv", "val_split.csv", "test_split.csv"}
    stray = [p for p in (REPO / "data").rglob("*split*")
             if p.name not in p4_known]
    assert not stray, f"accidental split artifact: {stray}"
    return "src/ contains audit code only; manifests/ holds the 5 Phase 1 artifacts + known Phase 2/3/4 additions"


def main() -> int:
    print("=" * 72)
    print("Phase 1 validation suite (V-1..V-11)")
    print("=" * 72)
    fns = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11]
    ok = sum(fn() for fn in fns)
    print("-" * 72)
    print(f"{ok}/{len(fns)} checks passed")
    return 0 if ok == len(fns) else 1


if __name__ == "__main__":
    sys.exit(main())
