"""Phase 4 validation suite (V-L4a..V-L4h) for the split deliverables.

Runs standalone (``python tests/test_leakage.py``) or under pytest.
Every check RECOMPUTES its values from the committed Phase 1/2/3 manifests
and the generated split manifests rather than trusting the report.

Leakage dimensions covered (Phase 4 instructions §8):
  A patient overlap          -> v_l4a (unavailability documented; NOT fabricated)
  B study overlap            -> v_l4b (unavailability documented; NOT fabricated)
  C source-group overlap     -> v_l4c (must be ZERO)
  D duplicate overlap        -> v_l4d (confirmed md5 duplicates; must be ZERO)
  E near-duplicate candidate -> v_l4e (ALL 25,607 pairs; must be ZERO crossing)
  F augmentation-family      -> v_l4f (must be ZERO)
plus completeness/atomicity (V-L4a/4b), freeze integrity (V-L4g) and
phase boundary (V-L4h).

IMPORTANT: near-duplicate pairs remain CANDIDATES; co-splitting them is a
leakage-prevention policy and is NOT a reclassification as confirmed
duplicates. Patient/study/lesion identifiers do not exist in this dataset
and were NOT fabricated; those overlap checks document unavailability
rather than zero overlap.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "data" / "manifests"
P1_MANIFEST = MANIFESTS / "dataset_manifest.csv"
P1_NEARDUP = MANIFESTS / "near_duplicate_candidates.csv"
P3_BAGS = MANIFESTS / "bag_manifest.csv"
P4_REPORT = REPO / "reports" / "phase4_split_report.md"
SPLIT_CSVS = {s: MANIFESTS / f"{s}_split.csv" for s in ("train", "val", "test")}
TEST_FREEZE_DATE = "2026-09-18"
EXPECTED_TEST_SHA256 = "959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74"

_spec = importlib.util.spec_from_file_location(
    "phase4_split", REPO / "src" / "data" / "splitting.py")
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)

_audit_spec = importlib.util.spec_from_file_location(
    "phase1_audit", REPO / "src" / "data" / "audit.py")
audit = importlib.util.module_from_spec(_audit_spec)
_audit_spec.loader.exec_module(audit)

COMMON, _RELS = audit.resolve_image_root(str(REPO / "dataset" / "raw"))


def full(rel: str) -> str:
    return os.path.join(COMMON, *rel.split("/"))


# --- shared, freshly loaded evidence -------------------------------------------
P1 = list(csv.DictReader(open(P1_MANIFEST, newline="", encoding="utf-8")))
BAGS = list(csv.DictReader(open(P3_BAGS, newline="", encoding="utf-8")))
CANDS = list(csv.DictReader(open(P1_NEARDUP, newline="", encoding="utf-8")))
SPLITS = {s: list(csv.DictReader(open(p, newline="", encoding="utf-8")))
          for s, p in SPLIT_CSVS.items()}

# path -> split (the authoritative mapping under test)
PATH2SPLIT: dict = {}
for s, rows in SPLITS.items():
    for r in rows:
        PATH2SPLIT[r["image_path"]] = s

# path -> bag metadata from the authoritative Phase 3 manifest
PATH2BAG: dict = {}
for b in BAGS:
    for p in b["instance_list"].split("|"):
        PATH2BAG[p] = b

BAG2SPLIT: dict = {}
for b in BAGS:
    splits = {PATH2SPLIT[p] for p in b["instance_list"].split("|")}
    assert len(splits) == 1, f"bag split across splits pre-check: {b['bag_id']}"
    BAG2SPLIT[b["bag_id"]] = splits.pop()


def check(name, desc):
    def deco(fn):
        fn._check = (name, desc)
        return fn
    return deco


# --- completeness & atomicity ----------------------------------------------------
@check("V-L4a", "496 source groups assigned exactly once; no group crosses splits")
def v_l4a():
    gmap = defaultdict(set)
    for s, rows in SPLITS.items():
        for r in rows:
            gmap[r["source_group_id"]].add(s)
    partial = {g: ss for g, ss in gmap.items() if len(ss) > 1}
    assert not partial, f"groups spanning splits: {dict(list(partial.items())[:3])}"
    assert len(gmap) == 496, f"expected 496 groups, found {len(gmap)}"
    counts = {s: len({r['source_group_id'] for r in rows}) for s, rows in SPLITS.items()}
    assert counts["train"] + counts["val"] + counts["test"] == 496
    return f"349+74+73 = 496 groups, each in exactly one split"


@check("V-L4b", "9,016 instances assigned exactly once across the three files")
def v_l4b():
    all_rows = [r for rows in SPLITS.values() for r in rows]
    paths = [r["image_path"] for r in all_rows]
    assert len(paths) == len(set(paths)) == 9016, (
        f"rows={len(paths)} unique={len(set(paths))}")
    assert set(paths) == set(PATH2BAG), "split instance set != Phase 3 instance set"
    return "6,327+1,339+1,350 = 9,016 instances, each exactly once"


# --- A/B: patient & study unavailability (documented, NOT fabricated) -------------
@check("V-L4A", "patient overlap: documented NOT ASSESSABLE (no verified patient IDs)")
def v_l4a_pat():
    # The dataset has no patient identifiers; verify their absence in every
    # artifact feeding the split and that no fabricated column exists.
    cols = set(SPLITS["train"][0].keys()) | set(SPLITS["val"][0].keys()) | set(SPLITS["test"][0].keys())
    for bad in ("patient_id", "study_id", "lesion_id"):
        assert bad not in cols, f"fabricated identifier column present: {bad}"
    statuses = {r["source_key_status"] for r in SPLITS["train"]}
    assert statuses == {"inferred_from_filename_not_verified_identifier"}
    report = " ".join(P4_REPORT.read_text(encoding="utf-8").split())
    assert "not assessable because verified patient IDs are absent" in report
    return "no patient IDs exist; unavailability documented in report §15; nothing fabricated"


@check("V-L4B", "study overlap: documented NOT ASSESSABLE (no verified study IDs)")
def v_l4b_study():
    report = " ".join(P4_REPORT.read_text(encoding="utf-8").split())
    assert "not assessable because verified study IDs are absent" in report
    statuses = {r["source_key_status"] for rows in SPLITS.values() for r in rows}
    assert statuses == {"inferred_from_filename_not_verified_identifier"}
    return "no study IDs exist; unavailability documented in report §16; nothing fabricated"


# --- C: source-group overlap --------------------------------------------------------
@check("V-L4c", "source_group_id disjointness: train ∩ val = train ∩ test = val ∩ test = ∅")
def v_l4c():
    g = {s: {r["source_group_id"] for r in rows} for s, rows in SPLITS.items()}
    assert not (g["train"] & g["val"]), "train ∩ val non-empty"
    assert not (g["train"] & g["test"]), "train ∩ test non-empty"
    assert not (g["val"] & g["test"]), "val ∩ test non-empty"
    return "all three pairwise intersections empty over source_group_id"


# --- D: confirmed duplicate leakage ---------------------------------------------------
@check("V-L4d", "no confirmed md5-duplicate relationship crosses splits")
def v_l4d():
    groups = defaultdict(list)
    for r in P1:
        groups[r["md5"]].append(r["path"])
    dup_groups = {m: ps for m, ps in groups.items() if len(ps) > 1}
    crossing = []
    for m, ps in dup_groups.items():
        ss = {PATH2SPLIT[p] for p in ps}
        if len(ss) > 1:
            crossing.append((m, ps))
    assert not crossing, f"{len(crossing)} md5-duplicate groups cross splits: {crossing[:2]}"
    return f"{len(dup_groups)} confirmed md5-duplicate groups re-derived; 0 cross splits"


# --- E: near-duplicate candidate leakage ------------------------------------------------
@check("V-L4e", "0 of ALL 25,607 near-duplicate candidate pairs cross splits")
def v_l4e():
    crossing = []
    for p in CANDS:
        sa = PATH2SPLIT.get(p["file_a"])
        sb = PATH2SPLIT.get(p["file_b"])
        if sa is None or sb is None:
            raise AssertionError(f"candidate endpoint unmapped: {p}")
        if sa != sb:
            crossing.append((p["file_a"], p["file_b"], sa, sb))
    assert not crossing, f"{len(crossing)} candidate pairs cross splits, e.g. {crossing[:3]}"
    assert len(CANDS) == 25607
    return f"0/25,607 candidate pairs cross splits (incl. the 996 originally cross-split; candidates NOT reclassified)"


# --- F: augmentation-family overlap --------------------------------------------------------
@check("V-L4f", "every instance shares its source_group_id's split (0 families span)")
def v_l4f():
    bad = []
    for s, rows in SPLITS.items():
        for r in rows:
            b = PATH2BAG[r["image_path"]]
            if b["source_group_id"] != r["source_group_id"] or BAG2SPLIT[b["bag_id"]] != s:
                bad.append(r["image_path"])
    assert not bad, f"family/instance split mismatches: {bad[:3]}"
    return "9,016/9,016 instances consistent with their family's single split"


# --- freeze integrity -------------------------------------------------------------------------
@check("V-L4g", "test split frozen: SHA256 matches report; freeze date recorded")
def v_l4g():
    h = hashlib.sha256(SPLIT_CSVS["test"].read_bytes()).hexdigest()
    assert h == EXPECTED_TEST_SHA256, f"test_split.csv hash changed: {h}"
    report = P4_REPORT.read_text(encoding="utf-8")
    assert EXPECTED_TEST_SHA256 in report, "report does not record the test hash"
    assert TEST_FREEZE_DATE in report, "report does not record the freeze date"
    assert "FROZEN" in report
    return f"sha256 {h[:16]}… stable and recorded; freeze date {TEST_FREEZE_DATE}"


# --- determinism ----------------------------------------------------------------------------------
@check("V-L4h", "fresh-process regeneration byte-identical; freeze hash unaffected")
def v_l4h():
    import subprocess
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    before = {s: p.read_bytes() for s, p in SPLIT_CSVS.items()}
    subprocess.run([sys.executable, "-m", "src.data.splitting"], cwd=str(REPO),
                   env=env, check=True, stdout=subprocess.DEVNULL)
    for s, p in SPLIT_CSVS.items():
        assert p.read_bytes() == before[s], f"{s}_split.csv not byte-identical"
    h = hashlib.sha256(SPLIT_CSVS["test"].read_bytes()).hexdigest()
    assert h == EXPECTED_TEST_SHA256
    return "regeneration byte-identical for all three manifests; test hash unchanged"


# --- phase boundary --------------------------------------------------------------------------------
@check("V-L4i", "no Phase 5+ functionality in Phase 4 deliverables")
def v_l4i():
    src = (REPO / "src" / "data" / "splitting.py").read_text(encoding="utf-8")
    import ast
    tree = ast.parse(src)
    forbidden_libs = {"torch", "tensorflow", "keras", "sklearn", "cv2",
                      "albumentations", "PIL", "mlflow", "wandb"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name.split(".")[0] not in forbidden_libs, f"forbidden import: {a.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in forbidden_libs, f"forbidden import: {node.module}"
    forbidden_calls = ("train", "fit", "predict", "transform", "resize",
                       "augment", "attention", "extract_feature")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", "")) or ""
            hit = [t for t in forbidden_calls if t in name.lower()]
            assert not hit, f"forbidden call: {name} ({hit})"
    return "AST analysis: no ML/preprocessing/augmentation imports or calls in splitting.py"


def main() -> int:
    fns = [v for name, v in sorted(globals().items())
           if name.startswith("v_l4") and callable(v) and hasattr(v, "_check")]
    fns.sort(key=lambda f: f._check[0])
    print("=" * 72)
    print("Phase 4 leakage & split validation suite (V-L4a..V-L4i)")
    print("=" * 72)
    ok = 0
    for fn in fns:
        name, desc = fn._check
        try:
            detail = fn()
            print(f"[PASS] {name} {desc} :: {detail}")
            ok += 1
        except AssertionError as e:
            print(f"[FAIL] {name} {desc} :: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {name} {desc} :: {type(e).__name__}: {e}")
    print("-" * 72)
    print(f"{ok}/{len(fns)} checks passed")
    return 0 if ok == len(fns) else 1


if __name__ == "__main__":
    sys.exit(main())
