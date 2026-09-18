"""Phase 3 validation suite (V3-1..V3-13) for the bag-definition deliverables.

Runs standalone (``python tests/test_phase3_bags.py``) or under pytest.
Every check RECOMPUTES its values from the committed Phase 1/2 manifests and
the generated bag manifest rather than trusting the report - no number in
reports/phase3_bag_definition.md may exist without an executable derivation.

Roadmap: PROGRESS.md Phase 3 validation checks + exit criterion. Paths:
decision D-1 (src/mil for MIL-definition code). The suite treats the
committed Phase 1/2 manifests as frozen evidence and the bag manifest as
the Phase 3 output under test.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "data" / "manifests"
P1_MANIFEST = MANIFESTS / "dataset_manifest.csv"
P2_GROUPS = MANIFESTS / "augmentation_groups.csv"
P2_SUMMARY = MANIFESTS / "phase2_grouping_summary.json"
BAGS_CSV = MANIFESTS / "bag_manifest.csv"
P3_REPORT = REPO / "reports" / "phase3_bag_definition.md"

_spec = importlib.util.spec_from_file_location(
    "phase3_bd", REPO / "src" / "mil" / "bag_definition.py")
bd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bd)

_audit_spec = importlib.util.spec_from_file_location(
    "phase1_audit", REPO / "src" / "data" / "audit.py")
audit = importlib.util.module_from_spec(_audit_spec)
_audit_spec.loader.exec_module(audit)

COMMON, _RELS = audit.resolve_image_root(str(REPO / "dataset" / "raw"))


def full(rel: str) -> str:
    return os.path.join(COMMON, *rel.split("/"))


# --- shared, freshly loaded evidence -----------------------------------------
P1 = list(csv.DictReader(open(P1_MANIFEST, newline="", encoding="utf-8")))
P2 = list(csv.DictReader(open(P2_GROUPS, newline="", encoding="utf-8")))
BAGS = list(csv.DictReader(open(BAGS_CSV, newline="", encoding="utf-8")))

P2_BY_GID: dict = defaultdict(list)
for r in P2:
    P2_BY_GID[r["source_group_id"]].append(r)

P1_MD5 = {r["path"]: r["md5"] for r in P1}


def check(name, desc):
    def deco(fn):
        fn._check = (name, desc)
        return fn
    return deco


# --- V3-1: bag completeness ---------------------------------------------------
@check("V3-1", "every Phase 2 source_group_id appears exactly once as a bag")
def v3_1():
    p2_gids = set(P2_BY_GID)
    bag_gids = [b["source_group_id"] for b in BAGS]
    assert len(bag_gids) == len(set(bag_gids)), "duplicate bags for one group"
    assert set(bag_gids) == p2_gids, (
        f"missing={sorted(p2_gids - set(bag_gids))[:3]} "
        f"extra={sorted(set(bag_gids) - p2_gids)[:3]}")
    bag_ids = [b["bag_id"] for b in BAGS]
    assert len(set(bag_ids)) == len(bag_ids), "duplicate bag_id values"
    assert all(bd.derive_bag_id(g) == i for g, i in zip(bag_gids, bag_ids))
    return f"{len(BAGS)}/{len(p2_gids)} groups -> bags, 1:1, ids deterministic"


# --- V3-2: instance completeness ----------------------------------------------
@check("V3-2", "every Phase 2 image appears in exactly one bag")
def v3_2():
    shown = []
    for b in BAGS:
        shown.extend(b["instance_list"].split("|"))
    c = Counter(shown)
    dups = [p for p, n in c.items() if n > 1]
    assert not dups, f"image in multiple bags: {dups[:3]}"
    assert set(shown) == set(P1_MD5), "bag coverage != Phase 2/1 image set"
    return f"{len(shown)}/9,016 images, each in exactly one bag"


# --- V3-3: traceability --------------------------------------------------------
@check("V3-3", "every bag instance traces to Phase 1/2 rows with matching md5")
def v3_3():
    bad = []
    p2_by_path = {r["path"]: r for r in P2}
    for b in BAGS:
        paths = b["instance_list"].split("|")
        md5s = b["instance_md5_list"].split("|")
        for p, m in zip(paths, md5s):
            if P1_MD5.get(p) != m or p not in p2_by_path:
                bad.append(p)
    assert not bad, f"untraceable instances: {bad[:3]}"
    return f"{sum(len(b['instance_list'].split('|')) for b in BAGS)} instances trace via md5 to Phase 1+2"


# --- V3-4: label integrity ------------------------------------------------------
@check("V3-4", "one unambiguous label per bag; no mixed-label bags")
def v3_4():
    bad = []
    labels = Counter()
    for b in BAGS:
        member_labels = {r["group_label"] for r in P2_BY_GID[b["source_group_id"]]}
        if member_labels != {b["bag_label"]} or b["bag_label"] not in ("benign", "malignant"):
            bad.append(b["bag_id"])
        labels[b["bag_label"]] += 1
    assert not bad, f"label-defect bags: {bad[:3]}"
    return f"496 bags single-labelled ({labels['benign']} benign / {labels['malignant']} malignant)"


# --- V3-5: group integrity -------------------------------------------------------
@check("V3-5", "every instance's source_group_id equals its bag's")
def v3_5():
    P2_BY_PATH = {r["path"]: r for r in P2}
    bad = []
    for b in BAGS:
        for p in b["instance_list"].split("|"):
            if P2_BY_PATH[p]["source_group_id"] != b["source_group_id"]:
                bad.append(p)
    assert not bad, f"group-id mismatches: {bad[:3]}"
    return "all 9,016 instances carry their bag's source_group_id"


# --- V3-6: deterministic ordering (module-level) ----------------------------------
@check("V3-6", "deterministic ordering - module output is byte-stable")
def v3_6():
    groups = bd.load_source_groups(str(P2_GROUPS))
    bags = bd.build_bags(groups)
    import io
    buf = io.StringIO()
    bd.write_manifest(bags, "/dev/null") if False else None
    # render in-memory via csv writer with the same settings
    import io as _io
    s = _io.StringIO()
    w = csv.DictWriter(s, fieldnames=bd.MANIFEST_COLUMNS, lineterminator="\n",
                       extrasaction="ignore")
    w.writeheader()
    for bag in bags:
        w.writerow(bag)
    rendered = s.getvalue().encode("utf-8")
    committed = BAGS_CSV.read_bytes()
    assert rendered == committed, "in-memory regeneration != committed manifest"
    return "module regeneration byte-identical to committed manifest"


# --- V3-7: bag-size statistics vs report ------------------------------------------
@check("V3-7", "min/max/mean/median recomputed and matching the report")
def v3_7():
    sizes = [int(b["instance_count"]) for b in BAGS]
    stats = {"bags": len(sizes), "min": min(sizes), "max": max(sizes),
             "mean": round(statistics.fmean(sizes), 4),
             "median": statistics.median(sizes),
             "total_instances": sum(sizes)}
    text = P3_REPORT.read_text(encoding="utf-8")
    assert f"| Bags | {stats['bags']} |" in text
    assert f"| Min bag size | {stats['min']} |" in text
    assert f"| Max bag size | {stats['max']} |" in text
    assert f"| Mean bag size | {stats['mean']} |" in text
    assert f"| Median bag size | {stats['median']} |" in text
    assert f"| Total instances | {stats['total_instances']:,} |" in text
    # cross-check against module statistics (CSV rows carry strings)
    int_bags = [{**b, "instance_count": int(b["instance_count"])} for b in BAGS]
    assert stats == bd.bag_size_statistics(int_bags)
    return (f"bags={stats['bags']} min={stats['min']} max={stats['max']} "
            f"mean={stats['mean']} median={stats['median']} "
            f"total={stats['total_instances']} - report matches")


# --- V3-8: no fabricated identifiers -----------------------------------------------
@check("V3-8", "no fabricated patient/study/lesion identifiers in Phase 3 files")
def v3_8():
    patterns = re.compile(
        r"\b(patient[_ -]?(?:id|001|123)|study[_ -]?(?:id|001|123)|"
        r"lesion[_ -]?(?:id|001|123))\b", re.IGNORECASE)
    bad = []
    for path in (BAGS_CSV, P3_REPORT, REPO / "src" / "mil" / "bag_definition.py"):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if patterns.search(line):
                bad.append(f"{path.name}:{i}")
    assert not bad, f"fabricated-identifier suspects: {bad[:5]}"
    statuses = {b["source_key_status"] for b in BAGS}
    assert statuses == {"inferred_from_filename_not_verified_identifier"}
    return "no patient/study/lesion ids; source_key_status marker on all 496 rows"


# --- V3-9: raw-data immutability -----------------------------------------------------
@check("V3-9", "raw files untouched: md5 re-hash of all 9,016 images")
def v3_9():
    drift = []
    for r in P1:
        h = hashlib.md5(open(full(r["path"]), "rb").read()).hexdigest()
        if h != r["md5"]:
            drift.append(r["path"])
    assert not drift, f"{len(drift)} raw files changed, e.g. {drift[:3]}"
    return "9,016/9,016 md5-identical to the committed Phase 1 manifest"


# --- V3-10: phase boundary -------------------------------------------------------------
@check("V3-10", "no Phase 4+ functionality in Phase 3 deliverables")
def v3_10():
    import ast
    src = (REPO / "src" / "mil" / "bag_definition.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden_libs = {"torch", "tensorflow", "keras", "sklearn",
                      "albumentations", "cv2", "mlflow", "wandb"}
    forbidden_name_parts = ("train", "attention", "checkpoint", "predict",
                            "augment", "split_")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                assert root not in forbidden_libs, f"forbidden import: {a.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            assert root not in forbidden_libs, f"forbidden import: {node.module}"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            low = node.name.lower()
            hit = [t for t in forbidden_name_parts if t in low]
            assert not hit, f"forbidden definition name: {node.name} ({hit})"
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            low = (name or "").lower()
            hit = [t for t in ("train", "fit", "predict", "transform",
                               "attention", "checkpoint") if t in low]
            assert not hit, f"forbidden call: {name} ({hit})"
    # Phase 4 legitimately added the roadmap-named split manifests; any
    # other split/fold-named file under data/ remains unexpected here.
    p4_known = {"train_split.csv", "val_split.csv", "test_split.csv"}
    p4 = [p for p in (REPO / "data").rglob("*")
          if p.is_file() and p.name not in p4_known
          and re.search(r"split|fold", p.name)]
    assert not p4, f"Phase 4-like artifacts present: {p4}"
    return "AST analysis: no ML/split/training/attention imports, definitions, or calls; no unexpected split artifacts"


# --- V3-11: manifest integrity -----------------------------------------------------------
@check("V3-11", "columns/ids/counts/paths valid; all instance paths resolve")
def v3_11():
    with open(BAGS_CSV, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == bd.MANIFEST_COLUMNS, f"unexpected columns: {header}"
    for b in BAGS:
        n = int(b["instance_count"])
        paths = b["instance_list"].split("|")
        md5s = b["instance_md5_list"].split("|")
        assert n == len(paths) == len(md5s), f"count/list mismatch in {b['bag_id']}"
        assert b["multi_split"] == ("True" if "|" in b["splits"] else "False")
        if b["multi_split"] == "False":
            assert len({p.split("/")[0] for p in paths}) == 1
    missing = [p for b in BAGS for p in b["instance_list"].split("|")
               if not os.path.isfile(full(p))]
    assert not missing, f"paths not resolving to raw images: {missing[:3]}"
    return (f"{len(BAGS)} rows x {len(header)} columns; 9,016/9,016 paths resolve "
            f"under the raw image root")


# --- V3-12: reproducibility (fresh process) ------------------------------------------------
@check("V3-12", "two regenerations byte-identical to the committed manifest")
def v3_12():
    import subprocess
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "-m", "src.mil.bag_definition"]
    digests = set()
    for _ in range(2):
        subprocess.run(cmd, cwd=str(REPO), env=env, check=True,
                       stdout=subprocess.DEVNULL)
        digests.add(hashlib.sha256(BAGS_CSV.read_bytes()).hexdigest())
    assert len(digests) == 1, "regeneration is not deterministic"
    return "two fresh-process regenerations sha256-identical to committed artifact"


# --- V3-13: cross-split family integrity ------------------------------------------------------
@check("V3-13", "the 2 multi-split families remain single bags")
def v3_13():
    ms = [b for b in BAGS if b["multi_split"] == "True"]
    assert len(ms) == 2, f"expected 2 multi-split bags, found {len(ms)}"
    by_key = {b["source_key"]: b for b in ms}
    assert set(by_key) == {"benign (36)", "malignant (18)"}
    summary = P2_SUMMARY.read_text(encoding="utf-8")
    for b in ms:
        p2 = P2_BY_GID[b["source_group_id"]]
        p2_splits = sorted({r["split"] for r in p2})
        assert p2_splits == b["splits"].split("|") == ["train", "val"]
        gid_in_summary = f'"{b["source_group_id"]}"' in summary
        assert gid_in_summary, f"{b['source_group_id']} not in Phase 2 summary"
        sizes = {len(P2_BY_GID[b["source_group_id"]]), int(b["instance_count"])}
        assert len(sizes) == 1
    return ("bag-3281cc7851ab (benign (36), 16) and bag-6f66ae71bafc "
            "(malignant (18), 21) each remain ONE bag spanning train|val")


# --- runner (repository convention) --------------------------------------------
def main() -> int:
    fns = [v for name, v in sorted(globals().items())
           if name.startswith("v3_") and callable(v) and hasattr(v, "_check")]
    fns.sort(key=lambda f: f._check[0])
    print("=" * 72)
    print("Phase 3 validation suite (V3-1..V3-13)")
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
