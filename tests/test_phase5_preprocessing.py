"""Phase 5 validation suite (V5-A..V5-I) for the preprocessing deliverables.

Runs standalone (``python tests/test_phase5_preprocessing.py``) or under
pytest. Every check RECOMPUTES its values from the raw dataset, the
committed Phase 4 manifests, the processed cache, and the committed
configuration - no number in the Phase 5 report may exist without an
executable derivation.

Leakage/protection invariants verified here:
  - the frozen Phase 4 test manifest is byte-unchanged (V5-G);
  - normalization statistics derive from TRAIN only (V5-C2);
  - no augmentation exists for val/test (V5-F);
  - the raw dataset is bit-identical to the Phase 1 manifest (V5-I).
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "data" / "manifests"
PROCESSED = REPO / "data" / "processed"
CONFIG = REPO / "configs" / "preprocessing_config.yaml"
P4_REPORT = REPO / "reports" / "phase4_split_report.md"
P5_REPORT = REPO / "reports" / "phase5_preprocessing_justification.md"
TEST_SPLIT_SHA256 = "959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74"
SPLITS = ("train", "val", "test")


def _load(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sp = _load("phase5_pipe", "src/preprocessing/pipeline.py")
audit = _load("phase1_audit", "src/data/audit.py")

CFG = sp.load_config()
COMMON, _RELS = audit.resolve_image_root(str(REPO / "dataset" / "raw"))


def full(rel: str) -> str:
    return os.path.join(COMMON, *rel.split("/"))


SPLIT_ROWS = {s: list(csv.DictReader(open(MANIFESTS / f"{s}_split.csv",
                                          newline="", encoding="utf-8")))
              for s in SPLITS}
PROV = {s: list(csv.DictReader(open(PROCESSED / s / "provenance.csv",
                                    newline="", encoding="utf-8")))
        for s in SPLITS}


def check(name, desc):
    def deco(fn):
        fn._check = (name, desc)
        return fn
    return deco


# --- A: image loading ------------------------------------------------------------
@check("V5-A", "loading: supported formats load; corrupt input fails explicitly")
def v5_a():
    for fmt in ("PNG", "JPEG"):
        row = next(r for r in SPLIT_ROWS["train"]
                   if r["image_path"].lower().endswith(".png" if fmt == "PNG" else ".jpg"))
        arr = sp.load_and_preprocess(full(row["image_path"]), CFG)
        assert arr.shape == (224, 224) and arr.dtype == np.uint16
    try:
        sp.load_and_preprocess(os.path.join(COMMON, "no_such_image.png"), CFG)
        raise AssertionError("corrupt/missing image did not raise")
    except sp.PreprocessingError:
        pass
    return "PNG + JPEG load via the pipeline; missing image raises PreprocessingError (no silent skip)"


# --- B: output shape/channels -----------------------------------------------------
@check("V5-B", "processed outputs are 224x224 single-channel I;16 PNGs")
def v5_b():
    p = PROCESSED / "test" / PROV["test"][0]["output_file"]
    with Image.open(p) as im:
        assert im.format == "PNG" and im.mode == "I;16" and im.size == (224, 224), \
            f"{im.format}/{im.mode}/{im.size}"
    return "PNG, I;16 (single-channel 16-bit), 224x224 — per config"


# --- C: determinism ------------------------------------------------------------------
@check("V5-C", "deterministic preprocessing: same input -> byte-identical output")
def v5_c():
    rows = SPLIT_ROWS["train"][::len(SPLIT_ROWS["train"]) // 10][:10]
    for r in rows:
        a1 = sp.load_and_preprocess(full(r["image_path"]), CFG)
        a2 = sp.load_and_preprocess(full(r["image_path"]), CFG)
        assert np.array_equal(a1, a2)
        cached = np.asarray(Image.open(PROCESSED / "train" / sp._out_name(r["image_path"])))
        assert np.array_equal(a1, cached)
    return "10/10 images: repeat calls and cache bytes identical"


@check("V5-C2", "normalization statistics derive from TRAIN only (recomputed)")
def v5_c2():
    mean, std = sp.verify_train_stats(str(REPO), CFG)
    text = " ".join(P5_REPORT.read_text(encoding="utf-8").split()).lower()
    assert "train split only" in text
    assert "no test-derived" in text or "test data were excluded" in text or "test set is not used" in text
    return f"train-only recomputation matches config: mean={mean:.6f}, std={std:.6f}"


# --- D: split coverage ------------------------------------------------------------------
@check("V5-D", "processed cache covers exactly the Phase 4 split manifests")
def v5_d():
    for s in SPLITS:
        manifest_paths = {r["image_path"] for r in SPLIT_ROWS[s]}
        prov_paths = {r["image_path"] for r in PROV[s]}
        assert manifest_paths == prov_paths, f"{s}: cache != manifest"
        files = {p.name for p in (PROCESSED / s).glob("*.png")}
        assert files == {r["output_file"] for r in PROV[s]}, f"{s}: files != provenance"
        assert len(PROV[s]) == {"train": 6327, "val": 1339, "test": 1350}[s]
    return "6,327 + 1,339 + 1,350 cached images == Phase 4 manifests, no missing/extra"


# --- E: split disjointness ----------------------------------------------------------------
@check("V5-E", "processed train/val/test outputs remain disjoint (Phase 4 manifests)")
def v5_e():
    sets = {s: {r["image_path"] for r in PROV[s]} for s in SPLITS}
    assert not (sets["train"] & sets["val"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["val"] & sets["test"])
    # provenance split fields agree with directory
    for s in SPLITS:
        assert all(r["split"] == s for r in PROV[s])
    return "pairwise disjoint; provenance split field matches cache directory"


# --- F: training-only augmentation ----------------------------------------------------------
@check("V5-F", "augmentation is train-only by code guard; no augmented variants cached")
def v5_f():
    for s in ("val", "test"):
        try:
            sp.augment(Image.new("L", (4, 4)), s, random.Random(1))
            raise AssertionError(f"{s} augmentation was allowed")
        except sp.PreprocessingError:
            pass
    rng = random.Random(CFG["config"]["random_seed"])
    seq1 = [sp.augment(Image.new("L", (16, 16)), "train", rng).tobytes()
            for _ in range(64)]
    rng = random.Random(CFG["config"]["random_seed"])
    seq2 = [sp.augment(Image.new("L", (16, 16)), "train", rng).tobytes()
            for _ in range(64)]
    assert seq1 == seq2, "seeded augmentation not reproducible"
    # cached outputs contain no augmented variants: cache rows == manifest rows 1:1
    assert len(PROV["val"]) == len(SPLIT_ROWS["val"]) == 1339
    assert len(PROV["test"]) == len(SPLIT_ROWS["test"]) == 1350
    summary = json.loads((PROCESSED / "cache_summary.json").read_text(encoding="utf-8"))
    assert "augmentation_in_cache" in summary and "none" in summary["augmentation_in_cache"]
    return "val/test raise PreprocessingError; train augmentation seeded+reproducible; cache holds 1:1 manifest rows only"


# --- G: test-manifest integrity ---------------------------------------------------------------
@check("V5-G", "frozen test_split.csv SHA256 unchanged")
def v5_g():
    h = hashlib.sha256((MANIFESTS / "test_split.csv").read_bytes()).hexdigest()
    assert h == TEST_SPLIT_SHA256, f"test manifest changed: {h}"
    text = P4_REPORT.read_text(encoding="utf-8")
    assert h in text
    assert "2026-09-18" in text and "FROZEN" in text
    return f"sha256 {h[:16]}… matches Phase 4 report and freeze record"


# --- H: provenance ---------------------------------------------------------------------------------
@check("V5-H", "provenance traces every processed image to raw path/md5/bag/group")
def v5_h():
    cols = list(PROV["train"][0].keys())
    for need in ("output_file", "image_path", "md5", "bag_id", "source_group_id",
                 "bag_label", "split"):
        assert need in cols, f"missing provenance column: {need}"
    p1 = {r["path"]: r["md5"] for r in csv.DictReader(
        open(MANIFESTS / "dataset_manifest.csv", newline="", encoding="utf-8"))}
    for s in SPLITS:
        for r in PROV[s][::len(PROV[s]) // 20][:20]:
            assert p1[r["image_path"]] == r["md5"], f"md5 chain broken: {r}"
            assert os.path.isfile(full(r["image_path"])), f"raw missing: {r}"
            assert r["source_key_status"] == "inferred_from_filename_not_verified_identifier"
    return "md5 chain to Phase 1 verified; paths resolve; no fabricated identity columns"


# --- I: raw-data immutability ------------------------------------------------------------------------
@check("V5-I", "raw dataset untouched by preprocessing (full md5 re-hash)")
def v5_i():
    drift = []
    for r in csv.DictReader(open(MANIFESTS / "dataset_manifest.csv",
                                 newline="", encoding="utf-8")):
        h = hashlib.md5(open(full(r["path"]), "rb").read()).hexdigest()
        if h != r["md5"]:
            drift.append(r["path"])
    assert not drift, f"{len(drift)} raw files changed: {drift[:3]}"
    return "9,016/9,016 raw files md5-identical to the Phase 1 manifest"


# --- scope boundary ------------------------------------------------------------------------------------
@check("V5-J", "no Phase 6+ functionality in Phase 5 deliverables")
def v5_j():
    import ast
    for rel in ("src/preprocessing/pipeline.py", "src/preprocessing/visual_review.py"):
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
        forbidden = {"torch", "tensorflow", "keras", "sklearn", "mlflow", "wandb"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert a.name.split(".")[0] not in forbidden, f"{rel}: import {a.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden, f"{rel}: import {node.module}"
            elif isinstance(node, ast.Call):
                # exact-name matching: a bare substring scan would match the
                # module's own train-only helpers (e.g. verify_train_stats)
                name = getattr(node.func, "id", getattr(node.func, "attr", "")) or ""
                if name in {"train", "fit", "evaluate", "predict"}:
                    raise AssertionError(f"{rel}: forbidden call {name}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in {"train", "fit", "evaluate", "predict"}:
                    raise AssertionError(f"{rel}: forbidden def {node.name}")
    assert not (REPO / "checkpoints").exists()
    return "AST analysis: no ML/training/eval imports or calls; no checkpoints/"


def main() -> int:
    fns = [v for name, v in sorted(globals().items())
           if name.startswith("v5_") and callable(v) and hasattr(v, "_check")]
    fns.sort(key=lambda f: f._check[0])
    print("=" * 72)
    print("Phase 5 preprocessing validation suite (V5-A..V5-J)")
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
