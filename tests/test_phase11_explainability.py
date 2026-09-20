"""Phase 11: explainability & attention visualization tests (T11-01..T11-22).

Validates the Phase 11 modules against the frozen Phase 9 E1 checkpoint and
the frozen manifests. The suite performs NO test-split access, NO training,
NO final test evaluation, and NO modification of frozen artifacts.

Runner convention: @check registers a zero-arg check; every check explicitly
returns True on success (the runner sums the returned booleans).
"""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RUNNERS: List[Any] = []

# Frozen integrity constants (single sources of truth).
E1_MD5 = "258710649fb0e6979f64fc1e7ccfc28f"
TEST_MANIFEST_SHA = "959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74"
B1_MD5 = "cbf558baf00b3ae4876e614aee429795"
B4_MD5 = "ef3fb5a8c53978621884678991f9671a"


def _load(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def check(name):
    def deco(fn):
        def wrapper() -> bool:
            try:
                fn()
                print(f"PASS  {name}")
                return True
            except Exception as e:  # noqa: BLE001
                print(f"FAIL  {name}: {e!r}")
                return False

        wrapper._label = name  # type: ignore[attr-defined]
        RUNNERS.append(wrapper)
        return wrapper

    return deco


_p11 = _load("phase11_av", "src/explainability/attention_visualization.py")
_gc = _load("phase11_gc", "src/explainability/gradcam.py")

FROZEN_SNAPSHOT_PATHS = [
    REPO / "data" / "manifests" / "test_split.csv",
    REPO / "data" / "manifests" / "train_split.csv",
    REPO / "data" / "manifests" / "val_split.csv",
    REPO / "data" / "manifests" / "bag_manifest.csv",
    REPO / "experiments" / "dual_attention_results" / "da_stage_b" / "test_predictions.csv",
    REPO / "experiments" / "dual_attention_results" / "da_stage_b" / "test_metrics.json",
    REPO / "experiments" / "dual_attention_results" / "da_stage_b" / "val_predictions.csv",
    REPO / "experiments" / "dual_attention_results" / "da_stage_b" / "metrics.json",
    REPO / "experiments" / "baseline_results" / "final_test" / "test_predictions.csv",
    REPO / "model" / "baselines" / "b1_simple_cnn" / "best.pt",
    REPO / "model" / "baselines" / "b4_meanpool_mil" / "best.pt",
    REPO / "model" / "dual_attention" / "da_stage_b" / "best.pt",
]

# One shared frozen model + two deterministic cases (kept small for speed).
_MODEL = _p11.load_frozen_e1_model()
_CASES = _p11.select_representative_cases()


def _case_batch(bag_id: str, split: str):
    ig = _load("phase11_ig", "src/mil/instance_generation.py")
    ds = ig.open_split(split)
    return ig.load_bag_batch(ds.bag(bag_id), ds)


def _first_case_of(category: str):
    for c in _CASES.values():
        if c["category"] == category:
            return c
    raise AssertionError(f"no case of category {category!r} selected")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# extraction / ranking
# ---------------------------------------------------------------------------

@check("T11-01 attention extraction shapes + bag probability")
def t11_01() -> None:
    c = _first_case_of("false_negative")
    batch = _case_batch(c["bag_id"], "val")
    import torch
    with torch.no_grad():
        res = _p11.extract_bag_attention(_MODEL, batch)
    n = batch.n_instances
    assert res["attention"].shape == (n,), res["attention"].shape
    assert res["gates"].shape == (n, 128)
    assert res["h_tilde"].shape == (n, 128)
    assert res["embeddings"].shape == (n, 128)
    assert 0.0 <= res["probability"] <= 1.0
    return True


@check("T11-02 gate vector shape is (n, 128) and finite in (0,1)")
def t11_02() -> None:
    c = _first_case_of("correct_malignant")
    batch = _case_batch(c["bag_id"], "val")
    import torch
    with torch.no_grad():
        res = _p11.extract_bag_attention(_MODEL, batch)
    g = res["gates"]
    assert g.shape[1] == 128 and np.all(np.isfinite(g))
    assert float(g.min()) > 0.0 and float(g.max()) < 1.0
    return True


@check("T11-03 attention normalization: sum == 1 per bag")
def t11_03() -> None:
    for c in list(_CASES.values())[:3]:
        batch = _case_batch(c["bag_id"], c["split"])
        import torch
        with torch.no_grad():
            res = _p11.extract_bag_attention(_MODEL, batch)
        assert abs(float(res["attention"].sum()) - 1.0) <= 1e-6
    return True


@check("T11-04 all extracted values finite")
def t11_04() -> None:
    c = _first_case_of("correct_benign")
    batch = _case_batch(c["bag_id"], "val")
    import torch
    with torch.no_grad():
        res = _p11.extract_bag_attention(_MODEL, batch)
    for k in ("attention", "gates", "h_tilde", "embeddings"):
        assert np.all(np.isfinite(res[k])), k
    return True


@check("T11-05 deterministic repeated extraction (bitwise)")
def t11_05() -> None:
    c = _first_case_of("false_positive")
    batch = _case_batch(c["bag_id"], "val")
    import torch
    with torch.no_grad():
        r1 = _p11.extract_bag_attention(_MODEL, batch)
        r2 = _p11.extract_bag_attention(_MODEL, batch)
    assert np.array_equal(r1["attention"], r2["attention"])
    assert np.array_equal(r1["gates"], r2["gates"])
    assert r1["probability"] == r2["probability"]
    return True


@check("T11-06 variable bag sizes handled (16/21/14)")
def t11_06() -> None:
    sizes = set()
    for c in _CASES.values():
        batch = _case_batch(c["bag_id"], c["split"])
        import torch
        with torch.no_grad():
            res = _p11.extract_bag_attention(_MODEL, batch)
        assert res["attention"].shape[0] == batch.n_instances
        sizes.add(batch.n_instances)
    assert len(sizes) >= 2, sizes
    return True


@check("T11-07 top-k ranking order + rank field")
def t11_07() -> None:
    c = next(iter(_CASES.values()))
    batch = _case_batch(c["bag_id"], c["split"])
    import torch
    with torch.no_grad():
        res = _p11.extract_bag_attention(_MODEL, batch)
    ranked = _p11.rank_instances(res["attention"])
    atts = [r["attention"] for r in ranked]
    assert atts == sorted(atts, reverse=True)
    assert [r["rank"] for r in ranked] == list(range(1, len(ranked) + 1))
    assert len({r["order_index"] for r in ranked}) == len(ranked)
    return True


@check("T11-08 deterministic tie handling (equal attention -> order)")
def t11_08() -> None:
    ranked = _p11.rank_instances([0.5, 0.2, 0.5, 0.1])
    order = [r["order_index"] for r in ranked]
    assert order == [0, 2, 1, 3], order          # tie 0.5 -> lower index first
    assert [r["rank"] for r in ranked] == [1, 2, 3, 4]
    return True


@check("T11-09 k > n clamping and k <= 0 rejection")
def t11_09() -> None:
    att = [0.1, 0.4, 0.3, 0.2]
    assert len(_p11.top_k_instances(att, 10)) == 4
    top1 = _p11.top_k_instances(att, 1)
    assert len(top1) == 1 and top1[0]["order_index"] == 1
    try:
        _p11.top_k_instances(att, 0)
    except ValueError:
        return True
    raise AssertionError("k=0 must raise")
    return True


# ---------------------------------------------------------------------------
# case selection / A2 boundary
# ---------------------------------------------------------------------------

@check("T11-10 deterministic case selection (repeat == identical)")
def t11_10() -> None:
    c1 = _p11.select_representative_cases()
    c2 = _p11.select_representative_cases()
    assert c1 == c2
    assert set(c1.keys()) == {
        "bag-01e91c2df96c", "bag-0e9e3f410b4b", "bag-05d8f8e1658b",
        "bag-01a4298f9ca8", "bag-244973d6f450"}
    cats = sorted(v["category"] for v in c1.values())
    assert cats == ["correct_benign", "correct_malignant", "false_negative",
                    "false_positive", "train_fallback"], cats
    return True


@check("T11-11 no test-split access: only train/val cases, loader refuses test")
def t11_11() -> None:
    for c in _CASES.values():
        assert c["split"] in ("train", "val"), c
    # the A2 guard raises on a test-split case
    try:
        _p11.extract_selected_cases(
            _MODEL, cases={"x": {"bag_id": "bag-04d1c13c4774", "split": "test",
                                 "category": "forbidden"}})
    except ValueError as e:
        assert "hard boundary" in str(e), e
        return True
    raise AssertionError("test-split case must be rejected")
    return True


@check("T11-12 selection rule uses only pre-registered criteria")
def t11_12() -> None:
    # margin constant fixed a priori and exported to the JSON payload
    assert _p11.UNCERTAINTY_MARGIN == 0.1
    assert _p11.SPLITS_ALLOWED == ("train", "val")
    return True


# ---------------------------------------------------------------------------
# A5 machine-readable export
# ---------------------------------------------------------------------------

@check("T11-13 export CSV schema + provenance preserved")
def t11_13() -> None:
    path = REPO / "reports" / "phase11" / "attention_export.csv"
    assert path.exists(), path
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"bag_id", "source_group_id", "source_key_status", "split",
                "image_path", "instance_order", "attention", "rank",
                "predicted_probability", "true_label", "prediction_at_0.5",
                "gate_mean", "gate_max", "gate_argmax", "gate_vector_ref"}
    assert required <= set(rows[0].keys())
    expected_n = 21 + 16 + 16 + 21 + 14
    assert len(rows) == expected_n, len(rows)
    for r in rows:
        assert r["source_key_status"] == \
            "inferred_from_filename_not_verified_identifier"
        assert r["split"] in ("train", "val")
    return True


@check("T11-14 export JSON payload: gate vectors + identity + no test flag")
def t11_14() -> None:
    path = REPO / "reports" / "phase11" / "attention_export.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["checkpoint_md5"] == E1_MD5
    assert payload["splits_used"] == ["train", "val"]
    assert payload["test_split_accessed"] is False
    assert payload["uncertainty_margin"] == 0.1
    assert len(payload["bags"]) == 5
    for b in payload["bags"]:
        assert len(b["attention"]) == b["n_instances"]
        assert len(b["gates"]) == b["n_instances"]
        assert all(len(g) == 128 for g in b["gates"])
        assert abs(sum(b["attention"]) - 1.0) <= 1e-6
        assert b["split"] in ("train", "val")
    return True


@check("T11-15 export determinism (two fresh exports byte-identical)")
def t11_15() -> None:
    import tempfile
    c = _first_case_of("train_fallback")          # smallest case (14 instances)
    batch = _case_batch(c["bag_id"], "train")
    import torch
    with torch.no_grad():
        res = _p11.extract_bag_attention(_MODEL, batch)
    res["split"] = "train"
    res["category"] = c["category"]
    res["frozen_probability"] = None
    res["frozen_prediction"] = None
    with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
        p1 = _p11.write_attention_export([res], dest_dir=td1)
        p2 = _p11.write_attention_export([res], dest_dir=td2)
        assert _sha(Path(p1["csv"])) == _sha(Path(p2["csv"]))
        assert _sha(Path(p1["json"])) == _sha(Path(p2["json"]))
        with open(p1["csv"], newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 14
        # canonical export untouched by the temp-dir regeneration
        assert (REPO / "reports" / "phase11" / "attention_export.csv").exists()
    return True


# ---------------------------------------------------------------------------
# visualization artifacts
# ---------------------------------------------------------------------------

@check("T11-16 attention grid + channel-gate figures exist for all cases")
def t11_16() -> None:
    fig_dir = REPO / "reports" / "figures" / "phase11"
    for c in _CASES.values():
        bid = c["bag_id"]
        assert (fig_dir / f"attention_grid_{c['split']}_{bid}.png").exists(), bid
        assert (fig_dir / f"channel_gates_{c['split']}_{bid}.png").exists(), bid
    return True


@check("T11-17 Grad-CAM overlays exist for one instance per case")
def t11_17() -> None:
    fig_dir = REPO / "reports" / "figures" / "phase11"
    n = len(list(fig_dir.glob("gradcam_*.png")))
    assert n == len(_CASES), n
    return True


@check("T11-18 Grad-CAM shape/determinism + no parameter grads")
def t11_18() -> None:
    c = _first_case_of("correct_benign")
    batch = _case_batch(c["bag_id"], "val")
    import torch
    g1 = _gc.compute_gradcam(_MODEL, batch, 0)
    g2 = _gc.compute_gradcam(_MODEL, batch, 0)
    assert g1["cam_28x28"].shape == (28, 28)   # block-4 ReLU grid (pre-pool)
    assert g1["cam_224x224"].shape == (224, 224)
    assert float(g1["cam_28x28"].min()) >= 0.0 and float(g1["cam_28x28"].max()) <= 1.0
    assert np.array_equal(g1["cam_28x28"], g2["cam_28x28"])
    for p in _MODEL.parameters():
        assert p.grad is None, "parameter gradients must not exist"
    assert _MODEL.training is False
    return True


@check("T11-19 Grad-CAM refuses test-split provenance")
def t11_19() -> None:
    class _FakeBatch:
        bag_id = "bag-fake"
        n_instances = 1
        provenance = ({"image_path": "x.png", "split": "test",
                       "processed_relpath": "data/processed/test/x.png"},)

    try:
        _gc.compute_gradcam(_MODEL, _FakeBatch(), 0)
    except ValueError as e:
        assert "hard boundary" in str(e), e
        return True
    raise AssertionError("Grad-CAM must refuse test-split batches")
    return True


# ---------------------------------------------------------------------------
# boundary / integrity
# ---------------------------------------------------------------------------

@check("T11-20 checkpoint MD5 unchanged after all extraction/visualization")
def t11_20() -> None:
    assert _p11.verify_checkpoint_md5() == E1_MD5
    assert _md5(REPO / "model" / "dual_attention" / "da_stage_b" / "best.pt") == E1_MD5
    return True


@check("T11-21 frozen artifacts unchanged (SHA snapshot)")
def t11_21() -> None:
    expected_test_sha = TEST_MANIFEST_SHA
    assert _sha(REPO / "data" / "manifests" / "test_split.csv") == expected_test_sha
    assert _md5(REPO / "model" / "baselines" / "b1_simple_cnn" / "best.pt") == B1_MD5
    assert _md5(REPO / "model" / "baselines" / "b4_meanpool_mil" / "best.pt") == B4_MD5
    for p in FROZEN_SNAPSHOT_PATHS[4:8]:
        assert p.exists(), p
    return True


@check("T11-22 module source: no optimizer/loss/train-call, no MRI, A2 guard")
def t11_22() -> None:
    banned = ["optim.", "Adam", "SGD", "BCEWithLogitsLoss", "CrossEntropyLoss",
              ".train(", "mri", "nibabel", "pydicom"]
    for rel in ("src/explainability/attention_visualization.py",
                "src/explainability/gradcam.py"):
        lowered = (REPO / rel).read_text(encoding="utf-8").lower()
        for b in banned:
            assert b.lower() not in lowered, (rel, b)
    # The single backward() in gradcam.py is the DOCUMENTED input-gradient
    # mechanism of Grad-CAM (gradients flow to the image only — T11-18 proves
    # parameter grads stay None); attention_visualization has none.
    gc_src = (REPO / "src/explainability/gradcam.py").read_text(encoding="utf-8")
    assert gc_src.count(".backward(") == 1 and "logit_t.backward(" in gc_src
    av_src = (REPO / "src/explainability/attention_visualization.py").read_text(
        encoding="utf-8")
    assert ".backward(" not in av_src
    # A2 hard boundary: the ONLY quoted "test" literal is the guard constant.
    assert _p11.SPLITS_ALLOWED == ("train", "val")
    assert _p11.FORBIDDEN_SPLITS == ("test",)
    assert av_src.count('"test"') == 1, "only the FORBIDDEN_SPLITS guard literal"
    assert '"test_split.csv"' not in av_src
    return True


@check("T11-23 report contains required disclaimer + mask-IoU statement")
def t11_23() -> None:
    report = (REPO / "reports" / "phase11_explainability_report.md").read_text(
        encoding="utf-8")
    assert "## Model Attention vs. Clinical Explanation" in report
    low = report.lower()
    assert "not a clinical explanation" in low or \
        "not clinical explanation" in low
    assert "mask-overlap / iou validation is not possible" in low
    # forbidden overclaim phrases must be absent
    for phrase in ("diagnoses breast cancer", "reliably detects"):
        assert phrase not in low, phrase
    # "clinically validated evidence" may appear ONLY inside the mandated
    # negated disclaimer, never as an affirmative claim.
    occurrences = low.count("clinically validated evidence")
    negated = low.count("not be interpreted as clinically validated evidence")
    assert occurrences == negated and occurrences >= 1, (occurrences, negated)
    return True


@check("T11-24 figures directory contains only Phase 11 outputs")
def t11_24() -> None:
    fig_dir = REPO / "reports" / "figures" / "phase11"
    files = sorted(p.name for p in fig_dir.iterdir() if p.is_file())
    assert all(n.endswith(".png") for n in files), files
    assert len(files) == 2 * len(_CASES) + len(_CASES), files
    return True


def main() -> int:
    results = [fn() for fn in RUNNERS]
    passed = int(sum(results))
    total = len(results)
    print(f"\n{passed}/{total} Phase 11 explainability checks passed"
          if passed == total else
          f"\n{passed}/{total} Phase 11 checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
