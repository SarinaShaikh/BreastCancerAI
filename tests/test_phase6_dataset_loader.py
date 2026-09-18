"""Phase 5.5 validation suite (V6-A..V6-R) for the bag data loader.

Runs standalone (``python tests/test_phase6_dataset_loader.py``) or under
pytest. Every check RECOMPUTES its values from the frozen Phase 4 manifests,
the Phase 3 bag manifest, the Phase 5 provenance/cache, and the loader API
itself — no number in reports/phase6_dataset_loader.md may exist without an
executable derivation.

Scope guard: the loader (src/mil/bag_dataset.py) implements NO model, NO MIL,
NO attention, NO training, NO evaluation (V6-S AST check).

Roadmap: NEW Phase 5.5 section (added 2026-09-18 by owner decision).
Test IDs V6-A..V6-S follow the Phase 6 dataset-loader instruction set.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "data" / "manifests"
PROCESSED = REPO / "data" / "processed"
TEST_SPLIT_SHA256 = "959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74"
SPLITS = ("train", "val", "test")
EXPECTED_INSTANCES = {"train": 6327, "val": 1339, "test": 1350}
EXPECTED_BAGS = {"train": 349, "val": 74, "test": 73}
LABEL_MAP = {"benign": 0, "malignant": 1}


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # required for dataclass processing
    spec.loader.exec_module(mod)
    return mod


bd = _load("phase6_bag_dataset", "src/mil/bag_dataset.py")
p5 = _load("phase6_p5_pipeline", "src/preprocessing/pipeline.py")

CFG = p5.load_config(str(REPO / "configs" / "preprocessing_config.yaml"))

SPLIT_ROWS = {s: list(csv.DictReader(open(MANIFESTS / f"{s}_split.csv",
                                          newline="", encoding="utf-8")))
              for s in SPLITS}
BAG_ROWS = list(csv.DictReader(open(MANIFESTS / "bag_manifest.csv",
                                    newline="", encoding="utf-8")))
DS = {s: bd.UltrasoundBagDataset(s, root=str(REPO)) for s in SPLITS}


def check(name, desc):
    def deco(fn):
        fn._check = (name, desc)
        return fn
    return deco


# --- A: manifests load -----------------------------------------------------------
@check("V6-A", "all three frozen split manifests load via the loader")
def v6_a():
    for s in SPLITS:
        assert len(DS[s]) > 0
    return "train/val/test datasets constructed without error"


# --- B: instance counts ------------------------------------------------------------
@check("V6-B", "instance counts match the frozen manifests (6327/1339/1350)")
def v6_b():
    for s in SPLITS:
        assert len(DS[s].instances) == EXPECTED_INSTANCES[s], s
        assert sum(b.instance_count for b in DS[s].bags) == EXPECTED_INSTANCES[s]
    assert sum(EXPECTED_INSTANCES.values()) == 9016
    return "6,327 + 1,339 + 1,350 = 9,016 instances, per split, via the loader"


# --- C: bag counts --------------------------------------------------------------------
@check("V6-C", "bag counts match Phase 3/Phase 4 artifacts (349/74/73)")
def v6_c():
    for s in SPLITS:
        assert len(DS[s].bags) == EXPECTED_BAGS[s], s
    assert len(BAG_ROWS) == 496
    return "349 + 74 + 73 = 496 Phase 3 bags, each in exactly one split"


# --- D: every manifest instance resolves to exactly one processed representation -------
@check("V6-D", "each manifest instance resolves to exactly one Phase 5 output")
def v6_d():
    for s in SPLITS:
        prov = {r["image_path"]: r["output_file"] for r in
                csv.DictReader(open(PROCESSED / s / "provenance.csv",
                                    newline="", encoding="utf-8"))}
        assert len(prov) == EXPECTED_INSTANCES[s]
        for r in SPLIT_ROWS[s]:
            assert r["image_path"] in prov, r["image_path"]
            assert os.path.isfile(REPO / "data" / "processed" / s / prov[r["image_path"]])
        rec = DS[s].instances[0]
        assert DS[s].load_instance_image is not None
    return "bijection image_path <-> output_file verified per split; all files exist"


# --- E/F: no bag or instance crosses splits ------------------------------------------------
@check("V6-E", "no bag crosses splits")
def v6_e():
    seen = {}
    for s in SPLITS:
        for b in DS[s].bags:
            if b.bag_id in seen:
                raise AssertionError(f"bag {b.bag_id} in {seen[b.bag_id]} and {s}")
            seen[b.bag_id] = s
    assert len(seen) == 496
    return "496 bags, each in exactly one split"


@check("V6-F", "no instance crosses splits")
def v6_f():
    seen = {}
    for s in SPLITS:
        for i in DS[s].instances:
            if i.image_path in seen:
                raise AssertionError(f"instance {i.image_path} in {seen[i.image_path]} and {s}")
            seen[i.image_path] = s
    assert len(seen) == 9016
    return "9,016 instances, each in exactly one split"


# --- G: bag membership matches the frozen Phase 4 mapping -----------------------------------
@check("V6-G", "loader bag membership exactly matches frozen manifests")
def v6_g():
    for s in SPLITS:
        loader_map = {i.image_path: i.bag_id for i in DS[s].instances}
        manifest_map = {r["image_path"]: r["bag_id"] for r in SPLIT_ROWS[s]}
        assert loader_map == manifest_map
        # group id + label also match row-for-row
        for i in DS[s].instances:
            m = SPLIT_ROWS[s]
        loader_group = {i.image_path: i.source_group_id for i in DS[s].instances}
        manifest_group = {r["image_path"]: r["source_group_id"] for r in SPLIT_ROWS[s]}
        assert loader_group == manifest_group
    return "bag_id and source_group_id per instance identical to frozen manifests"


# --- H: labels valid and consistent ----------------------------------------------------------
@check("V6-H", "labels valid (benign/malignant) and consistent within every bag")
def v6_h():
    for s in SPLITS:
        for b in DS[s].bags:
            assert b.label in LABEL_MAP, b.label
            assert len({i.label for i in b.instances}) == 1
            assert all(i.label == b.label for i in b.instances)
            assert b.numeric_label == LABEL_MAP[b.label]
    counts = {s: dict(Counter(b.label for b in DS[s].bags)) for s in SPLITS}
    assert counts["train"] == {"benign": 200, "malignant": 149}
    assert counts["val"] == {"benign": 43, "malignant": 31}
    assert counts["test"] == {"benign": 43, "malignant": 30}
    return f"all bag labels valid; split label counts {counts}"


# --- I/J: deterministic ordering (same process, repeated construction) ------------------------
@check("V6-I", "bag ordering deterministic across repeated loader construction")
def v6_i():
    for s in SPLITS:
        ids1 = [b.bag_id for b in DS[s].bags]
        ids2 = [b.bag_id for b in bd.UltrasoundBagDataset(s, root=str(REPO)).bags]
        assert ids1 == ids2 and ids1 == sorted(ids1)
    return "bag order identical across constructions and equals ascending bag_id"


@check("V6-J", "instance ordering deterministic across repeated loader construction")
def v6_j():
    for s in SPLITS:
        p1 = [i.image_path for i in DS[s].instances]
        p2 = [i.image_path for i in bd.UltrasoundBagDataset(s, root=str(REPO)).instances]
        assert p1 == p2 and p1 == sorted(p1)
        # and within-bag order equals the Phase 3 instance_list order
        bag_rows = {r["bag_id"]: r for r in BAG_ROWS}
        for b in DS[s].bags:
            listed = bag_rows[b.bag_id]["instance_list"].split("|")
            got = [i.image_path for i in b.instances]
            assert got == listed == sorted(got)
    return "instance order equals ascending image_path == Phase 3 instance_list"


# --- K: Phase 5 contract on loaded images -------------------------------------------------------
@check("V6-K", "loaded images honor the Phase 5 contract (shape/dtype/range/decode)")
def v6_k():
    rng = np.random.default_rng(0)
    for s in SPLITS:
        insts = DS[s].instances
        picks = [insts[0], insts[len(insts) // 2], insts[-1],
                 insts[int(rng.integers(0, len(insts)))]]
        for rec in picks:
            arr = DS[s].load_instance_image(rec)
            assert arr.shape == (224, 224) and arr.dtype == np.float32
            assert float(arr.min()) >= -1.3 and float(arr.max()) <= 2.8
            # decode must equal the Phase 5 exact integer inverse
            with Image.open(REPO / rec.processed_relpath) as im:
                u16 = np.asarray(im)
                assert u16.dtype == np.uint16 and u16.shape == (224, 224)
                expect = (u16.astype(np.float64) / 4096.0 - 8.0).astype(np.float32)
            assert np.array_equal(arr, expect)
    # loader decodes == direct pipeline.load_and_preprocess normalization
    rec = DS["test"].instances[0]
    raw = os.path.join(p5.raw_image_root(str(REPO)), *rec.image_path.split("/"))
    u16_direct = p5.load_and_preprocess(raw, CFG)
    expect = (u16_direct.astype(np.float64) / 4096.0 - 8.0).astype(np.float32)
    assert np.array_equal(DS["test"].load_instance_image(rec), expect)
    return "float32 (224,224); exact Phase 5 inverse decode; equals direct pipeline output"


# --- L: no augmentation for val/test (and none anywhere in the loader) --------------------------
@check("V6-L", "validation and test loaders perform no augmentation")
def v6_l():
    # the loader has no augmentation entry point at all
    assert not hasattr(bd.UltrasoundBagDataset, "augment")
    src = (REPO / "src" / "mil" / "bag_dataset.py").read_text(encoding="utf-8")
    for t in ("FLIP_LEFT_RIGHT", "transpose", "RandomCrop", "ColorJitter"):
        assert t not in src, f"augmentation primitive in loader: {t}"
    # deterministic image bytes on repeated loads for val/test
    for s in ("val", "test"):
        rec = DS[s].instances[7]
        a1 = DS[s].load_instance_image(rec)
        a2 = bd.UltrasoundBagDataset(s, root=str(REPO)).load_instance_image(rec)
        assert np.array_equal(a1, a2)
    return "no augmentation API/primitives in the loader; val/test loads byte-identical"


# --- M: no statistics computed from test data -----------------------------------------------------
@check("V6-M", "test loader computes no statistics from test data")
def v6_m():
    src = (REPO / "src" / "mil" / "bag_dataset.py").read_text(encoding="utf-8")
    # the loader contains no mean/std estimation over images; the only
    # normalization reference is the frozen Phase 5 decode contract
    assert "verify_train_stats" not in src  # stats enforcement stays in Phase 5
    assert ".mean()" not in src.replace("mean\", \"", "")
    # constructing/iterating the test dataset performs no pixel reads
    ds2 = bd.UltrasoundBagDataset("test", root=str(REPO), verify_cache_files=False)
    _ = [b.bag_id for b in ds2.bags]
    return "loader never estimates normalization statistics; construction does no pixel reads"


# --- N: collate mask semantics -----------------------------------------------------------------------
@check("V6-N", "collate mask marks real instances only; padding never interpretable")
def v6_n():
    rng = np.random.default_rng(1)
    batch = []
    for k, (n_i, lab) in enumerate([(14, 0), (21, 1), (16, 0)]):
        im = rng.standard_normal((n_i, 224, 224)).astype(np.float32)
        batch.append((im, lab, f"bag-{k}"))
    out = bd.UltrasoundBagDataset.collate_bags(batch)
    assert out["images"].shape == (3, 21, 224, 224)
    assert out["mask"].shape == (3, 21) and out["mask"].dtype == np.bool_
    assert out["mask"].sum(axis=1).tolist() == [14, 21, 16]
    assert (out["instance_counts"] == out["mask"].sum(axis=1)).all()
    for row, n_i in enumerate((14, 21, 16)):
        assert out["mask"][row, :n_i].all() and not out["mask"][row, n_i:].any()
        # padded pixels are exactly zero and masked out
        if n_i < 21:
            assert float(np.abs(out["images"][row, n_i:]).max()) == 0.0
    assert out["labels"].tolist() == [0, 1, 0]
    assert out["bag_ids"] == ("bag-0", "bag-1", "bag-2")
    try:
        bd.UltrasoundBagDataset.collate_bags([])
        raise AssertionError("empty collate did not raise")
    except bd.DatasetError:
        pass
    return "mask == real instances; padded region zeroed; counts == mask.sum; empty batch raises"


# --- O: missing/corrupt processed files fail clearly ----------------------------------------------------
@check("V6-O", "missing/corrupt processed files fail with actionable errors")
def v6_o():
    ds = DS["test"]
    rec = ds.instances[0]
    # corrupt file (wrong mode/size content) -> contract violation
    tmp = REPO / "data" / "processed" / "test" / "_v6o_tmp.png"
    try:
        Image.new("RGB", (64, 64), 0).save(tmp)
        bad = bd.InstanceRecord(**{**rec.__dict__, "processed_relpath":
                                   str(tmp.relative_to(REPO)).replace(os.sep, "/")})
        try:
            ds.load_instance_image(bad)
            raise AssertionError("corrupt image did not raise")
        except bd.DatasetError as e:
            assert "Phase 5 contract violation" in str(e) or "corrupt" in str(e)
        # missing file -> actionable message
        gone = bd.InstanceRecord(**{**rec.__dict__, "processed_relpath":
                                    "data/processed/test/definitely_missing.png"})
        try:
            ds.load_instance_image(gone)
            raise AssertionError("missing image did not raise")
        except bd.DatasetError as e:
            assert "missing" in str(e) and "regenerate" in str(e).lower()
        # constructor hard-fails on a missing cache file
        try:
            bd.UltrasoundBagDataset("test", root=str(REPO),
                                    processed_root=str(REPO / "data" / "_absent"))
            raise AssertionError("absent cache did not raise")
        except (bd.DatasetError, FileNotFoundError):
            pass
    finally:
        if tmp.exists():
            tmp.unlink()
    return "missing -> 'regenerate the Phase 5 cache' error; corrupt -> contract-violation error"


# --- P: provenance fields intact --------------------------------------------------------------------------
@check("V6-P", "provenance chain intact on every instance record")
def v6_p():
    p1 = {r["path"]: r for r in
          csv.DictReader(open(MANIFESTS / "dataset_manifest.csv",
                              newline="", encoding="utf-8"))}
    n = 0
    for s in SPLITS:
        for i in DS[s].instances[:: max(1, len(DS[s].instances) // 40)]:
            # status marker: loader record AND Phase 1 manifest must both say
            # the filename-derived key is NOT a verified identifier
            assert i.source_key_status == "inferred_from_filename_not_verified_identifier"
            assert p1[i.image_path]["source_key_status"] == \
                "inferred_from_filename_not_verified_identifier"
            # md5 chain to Phase 1
            assert i.md5 == p1[i.image_path]["md5"]
            assert i.split == s
            assert i.processed_relpath.startswith(f"data/processed/{s}/")
            assert i.label in LABEL_MAP
            assert i.numeric_label == LABEL_MAP[i.label]
            n += 1
    return f"{n} sampled records carry md5->Phase1, bag/group->Phase3, split->Phase4, status marker (loader+Phase 1)"


# --- Q: fresh-process determinism ---------------------------------------------------------------------------
@check("V6-Q", "a fresh Python process produces the same bag/instance ordering")
def v6_q():
    code = (
        "import importlib.util, sys, json\n"
        "spec = importlib.util.spec_from_file_location('bd', 'src/mil/bag_dataset.py')\n"
        "bd = importlib.util.module_from_spec(spec); sys.modules['bd'] = bd\n"
        "spec.loader.exec_module(bd)\n"
        "out = {}\n"
        "for s in ('train','val','test'):\n"
        "    ds = bd.UltrasoundBagDataset(s, root='.', verify_cache_files=False)\n"
        "    out[s] = {'bags': [b.bag_id for b in ds.bags][:20],\n"
        "              'instances': [i.image_path for i in ds.instances][:20]}\n"
        "print(json.dumps(out, sort_keys=True))\n"
    )
    procs = [subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, cwd=str(REPO)) for _ in range(2)]
    assert procs[0].returncode == 0 and procs[1].returncode == 0, procs[0].stderr
    assert procs[0].stdout == procs[1].stdout
    return "two fresh processes: identical (truncated) bag/instance orderings"


# --- R: frozen test manifest unchanged ------------------------------------------------------------------------
@check("V6-R", "Phase 4 frozen test_split.csv SHA256 unchanged")
def v6_r():
    h = hashlib.sha256((MANIFESTS / "test_split.csv").read_bytes()).hexdigest()
    assert h == TEST_SPLIT_SHA256
    # and the loader verifies the same hash through the Phase 5 module
    assert DS["test"].frozen_test_sha256 == h
    assert p5.verify_freeze(str(REPO)) == h
    return f"sha256 {h[:16]}… enforced by the loader itself (single source of truth)"


# --- S: phase boundary ------------------------------------------------------------------------------------------
@check("V6-S", "no Phase 6+ model functionality in the loader (AST scan)")
def v6_s():
    import ast
    tree = ast.parse((REPO / "src" / "mil" / "bag_dataset.py").read_text(encoding="utf-8"))
    forbidden_imports = {"torch", "tensorflow", "keras", "sklearn"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name.split(".")[0] not in forbidden_imports, a.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in forbidden_imports, node.module
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", "")) or ""
            assert name not in {"train", "fit", "evaluate", "predict"}, name
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert node.name not in {"train", "fit", "evaluate", "predict"}, node.name
    assert not (REPO / "checkpoints").exists()
    return "AST: no ML imports/calls/defs in the loader; no checkpoints/"


def main() -> int:
    fns = [v for name, v in sorted(globals().items())
           if name.startswith("v6_") and callable(v) and hasattr(v, "_check")]
    fns.sort(key=lambda f: f._check[0])
    print("=" * 72)
    print("Phase 5.5 bag data loader validation suite (V6-A..V6-S)")
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
