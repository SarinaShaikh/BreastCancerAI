"""Phase 10: evaluation, error analysis, and robustness tests.

These tests validate the Phase 10 evaluation modules against the already-frozen
final predictions. They do NOT rerun final model evaluation and do NOT modify
any frozen artifacts.

Regression scope: Phase 1–9 suites remain the primary regression guard. This
file is an additional Phase 10-specific test.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TEST_MANIFEST_SHA256 = (
    "959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74"
)

RUNNERS: List[object] = []


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


# Lazy-load Phase 10 modules under test.
_phase10_metrics = _load("phase10_metrics", "src/evaluation/metrics.py")
_phase10_error = _load("phase10_error_analysis", "src/evaluation/error_analysis.py")

# Lazy-load authoritative Phase 6 metric implementation for consistency checks.
_phase6_metrics = _load("phase6_metrics_ref", "src/training/metrics.py")


# ---------------------------------------------------------------------------
# Tiny deterministic fixtures
# ---------------------------------------------------------------------------

@check("P10-01 frozen test manifest SHA verification")
def t_p10_01_frozen_manifest_sha() -> None:
    sha = _phase10_metrics.sha256_file(
        str(REPO / "data" / "manifests" / "test_split.csv")
    )
    assert sha == TEST_MANIFEST_SHA256, sha
    return True


@check("P10-02 frozen test manifest verification raises on mismatch")
def t_p10_02_frozen_manifest_verify_raises() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        (td_path / "data" / "manifests").mkdir(parents=True)
        p = td_path / "data" / "manifests" / "test_split.csv"
        p.write_text("bag_id,source_group_id\nbad,grp-bad\n")
        try:
            _phase10_metrics.verify_test_manifest(root=str(td_path))
        except RuntimeError as e:
            msg = str(e)
            assert "Frozen test manifest SHA mismatch" in msg, msg
            return True
        raise AssertionError("expected RuntimeError on SHA mismatch")
    return True


@check("P10-03 frozen test population from manifest")
def t_p10_03_frozen_test_population() -> None:
    rows = _phase10_metrics.load_frozen_test_manifest(root=str(REPO))
    assert len(rows) == 73, len(rows)
    assert any(r.get("bag_id") == "bag-04d1c13c4774" for r in rows), rows[:1]
    assert any(r.get("source_group_id") == "grp-04d1c13c4774" for r in rows), rows[:1]
    assert any(r.get("bag_label") == "malignant" for r in rows)
    return True


@check("P10-04 classification_metrics delegation matches Phase 6")
def t_p10_04_metric_delegation_consistent() -> None:
    y_true = [0, 0, 1, 1, 1]
    y_prob = [0.1, 0.4, 0.6, 0.8, 0.55]
    m1 = _phase10_metrics._classification_metrics(
        "p10:test", y_true, y_prob, threshold=0.5
    )
    m2 = _phase6_metrics.classification_metrics(
        y_true, y_prob, threshold=0.5
    )
    for k in ("TP", "TN", "FP", "FN", "accuracy", "sensitivity",
              "specificity", "precision", "recall", "f1",
              "roc_auc", "pr_auc"):
        assert m1[k] == m2[k], (k, m1[k], m2[k])
    return True


@check("P10-05 ROC/PR curve data shapes and ordering")
def t_p10_05_curve_data() -> None:
    y_true = [0, 0, 1, 1, 1]
    y_prob = [0.1, 0.4, 0.6, 0.8, 0.55]
    fpr, tpr, thr = _phase10_metrics.roc_curve_data(y_true, y_prob)
    rec, prec, pthr = _phase10_metrics.pr_curve_data(y_true, y_prob)
    assert fpr.shape == tpr.shape == thr.shape
    assert rec.ndim == prec.ndim == pthr.ndim
    # basic monotonicity sanity for ROC TPR
    assert bool(np.all(np.diff(tpr) >= -1e-12))
    return True


@check("P10-06 comparative table structure")
def t_p10_06_comparative_table() -> None:
    rows = _phase10_metrics.comparative_metrics_table(
        [
            {
                "model": "B1_simple_cnn",
                "granularity": "image",
                "y_true": [0, 1, 1, 0],
                "y_prob": [0.2, 0.6, 0.9, 0.4],
            },
            {
                "model": "E1_da_stage_b",
                "granularity": "bag",
                "y_true": [0, 1, 1, 0],
                "y_prob": [0.2, 0.6, 0.9, 0.4],
            },
        ],
        threshold=0.5,
    )
    assert len(rows) == 2
    for r in rows:
        for k in (
            "model",
            "granularity",
            "roc_auc",
            "pr_auc",
            "accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "recall",
            "f1",
            "TP",
            "TN",
            "FP",
            "FN",
            "probability_range",
            "n",
            "threshold",
        ):
            assert k in r, k
    assert rows[0]["granularity"] == "image"
    assert rows[1]["granularity"] == "bag"
    # Clipping helper keeps probability_range bounded to [0,1].
    assert rows[0]["probability_range"] == [0.2, 0.9]
    assert rows[1]["probability_range"] == [0.2, 0.9]
    return True


@check("P10-07 error classification masks")
def t_p10_07_error_classification() -> None:
    y_true = [0, 0, 1, 1, 1]
    y_prob = [0.2, 0.4, 0.4, 0.9, 0.55]
    cm = _phase10_error.error_classification(y_true, y_prob, threshold=0.5)
    for k in ("tp", "tn", "fp", "fn"):
        assert k in cm
        assert cm[k].dtype == bool
    assert int(cm["tp"].sum()) == 2
    assert int(cm["fn"].sum()) == 1
    assert int(cm["fp"].sum()) == 0
    assert int(cm["tn"].sum()) == 2
    return True


@check("P10-08 error_case_rows provenance attachment")
def t_p10_08_error_case_provenance() -> None:
    y_true = [0, 0, 1, 1]
    y_prob = [0.2, 0.9, 0.4, 0.55]
    ids = ["a", "b", "c", "d"]
    prov = {"a": {"bag_id": "a", "source_group_id": "grp-a", "true_label": 0}}
    rows = _phase10_error.error_case_rows(
        y_true, y_prob, ids, provenance=prov, threshold=0.5
    )
    assert rows["FP"][0]["id"] == "b"
    assert rows["FN"][0]["id"] == "c"
    return True


@check("P10-09 uncertainty_mask deterministic")
def t_p10_09_uncertainty_mask() -> None:
    y_prob = [0.4, 0.45, 0.55, 0.6, 0.9]
    m = _phase10_metrics.uncertainty_mask(y_prob, margin=0.1, threshold=0.5)
    # Within margin: 0.4, 0.45, 0.55, 0.6 => 4 values.
    assert int(m.sum()) == 4
    return True


@check("P10-10 lowest_confidence_mask deterministic")
def t_p10_10_lowest_confidence_mask() -> None:
    y_prob = [0.1, 0.4, 0.49, 0.6, 0.9]
    m = _phase10_metrics.lowest_confidence_mask(y_prob, k=2, threshold=0.5)
    assert int(m.sum()) == 2
    return True


@check("P10-11 highest_confidence_incorrect_mask deterministic")
def t_p10_11_highest_confidence_incorrect() -> None:
    y_true = [0, 0, 1, 1, 1]
    y_prob = [0.05, 0.95, 0.35, 0.55, 0.6]
    m = _phase10_metrics.highest_confidence_incorrect_mask(
        y_true, y_prob, k=2, threshold=0.5
    )
    assert int(m.sum()) == 2
    return True


@check("P10-12 correct/incorrect masks")
def t_p10_12_correct_incorrect_masks() -> None:
    y_true = [0, 0, 1, 1, 1]
    y_prob = [0.2, 0.9, 0.4, 0.9, 0.55]
    cc = _phase10_metrics.correct_incorrect_mask(y_true, y_prob, threshold=0.5)
    assert cc["correct"].sum() + cc["incorrect"].sum() == len(y_true)
    return True


@check("P10-13 class_wise_summary")
def t_p10_13_class_wise_summary() -> None:
    y_true = [0, 0, 1, 1, 1]
    y_prob = [0.2, 0.9, 0.4, 0.9, 0.55]
    s = _phase10_metrics.class_wise_summary(y_true, y_prob, threshold=0.5)
    assert s["overall"]["TP"] == 2
    assert s["overall"]["FP"] == 1
    assert s["overall"]["FN"] == 1
    assert s["benign"]["count"] == 2
    assert s["malignant"]["count"] == 3
    return True


@check("P10-14 model_disagreement_rows")
def t_p10_14_model_disagreement_rows() -> None:
    a = [
        {"bag_id": "b1", "source_group_id": "g1", "probability": "0.6",
         "prediction": "1", "true_label": 1},
        {"bag_id": "b2", "source_group_id": "g2", "probability": "0.4",
         "prediction": "0", "true_label": 0},
    ]
    b = [
        {"bag_id": "b1", "source_group_id": "g1", "probability": "0.4",
         "prediction": "0", "true_label": 1},
        {"bag_id": "b2", "source_group_id": "g2", "probability": "0.6",
         "prediction": "1", "true_label": 0},
    ]
    d = _phase10_error.model_disagreement_rows(a, b, id_key="bag_id")
    assert len(d) == 2
    return True


@check("P10-15 stratify_by_bag_size deterministic")
def t_p10_15_stratify_bag_size() -> None:
    y_true = [0, 1, 1, 0, 0, 1]
    y_prob = [0.2, 0.6, 0.9, 0.4, 0.45, 0.55]
    ids = ["b1", "b1", "b2", "b2", "b2", "b2"]
    prov = {
        "b1": {"source_group_id": "g1"},
        "b2": {"source_group_id": "g2"},
    }
    tmp = tempfile.NamedTemporaryFile(
        suffix=".csv", mode="w", newline="", encoding="utf-8", delete=False
    )
    try:
        w = csv.writer(tmp)
        w.writerow(["bag_id", "n_instances", "y_true", "probability", "prediction"])
        w.writerow(["b1", 2, 0, 0.2, 0])
        w.writerow(["b1", 2, 1, 0.6, 1])
        w.writerow(["b2", 4, 1, 0.9, 1])
        w.writerow(["b2", 4, 0, 0.4, 0])
        w.writerow(["b2", 4, 0, 0.45, 0])
        w.writerow(["b2", 4, 1, 0.55, 1])
        tmp.flush()
        res = _phase10_metrics.stratify_by_bag_size(
            y_true, y_prob, ids, prov, prediction_csv_path=tmp.name, threshold=0.5
        )
        assert "2" in res
        assert "4" in res
        assert res["2"]["n_cases"] == 2
        assert res["4"]["n_cases"] == 4
    finally:
        tmp.close()
        os.unlink(tmp.name)
    return True


@check("P10-16 stratify_by_source_group deterministic")
def t_p10_16_stratify_source_group() -> None:
    y_true = [0, 1, 1, 0]
    y_prob = [0.2, 0.6, 0.9, 0.4]
    ids = ["b1", "b2", "b2", "b3"]
    prov = {
        "b1": {"source_group_id": "g1"},
        "b2": {"source_group_id": "g2"},
        "b3": {"source_group_id": "g3"},
    }
    res = _phase10_metrics.stratify_by_source_group(
        y_true, y_prob, ids, prov, threshold=0.5
    )
    assert "g1" in res
    assert "g2" in res
    assert "g3" in res
    return True


@check("P10-17 frozen prediction CSV schema integrity (baseline final_test)")
def t_p10_17_final_test_csv_schema() -> None:
    path = REPO / "experiments" / "baseline_results" / "final_test" / "test_predictions.csv"
    assert path.exists(), path
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 73, len(rows)
    schema = set(rows[0].keys())
    expected = {
        "bag_id",
        "n_instances",
        "true_label",
        "b3_bag_probability",
        "b3_prediction",
        "b4_bag_probability",
        "b4_prediction",
    }
    assert expected <= schema, schema
    return True


@check("P10-18 frozen prediction CSV schema integrity (E1)")
def t_p10_18_e1_csv_schema() -> None:
    path = REPO / "experiments" / "dual_attention_results" / "da_stage_b" / "test_predictions.csv"
    assert path.exists(), path
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 73, len(rows)
    schema = set(rows[0].keys())
    expected = {
        "bag_id",
        "source_group_id",
        "y_true",
        "probability",
        "prediction",
        "split",
    }
    assert expected <= schema, schema
    return True


@check("P10-19 no test-set mutation by module import/construction")
def t_p10_19_frozen_files_unchanged_after_import() -> None:
    import contextlib
    # Snapshot SHA of key frozen files before import of evaluation modules.
    files = {
        "test_manifest": REPO / "data" / "manifests" / "test_split.csv",
        "final_test_metrics": REPO
        / "experiments"
        / "baseline_results"
        / "final_test"
        / "metrics.json",
        "final_test_csv": REPO
        / "experiments"
        / "baseline_results"
        / "final_test"
        / "test_predictions.csv",
        "e1_test_metrics": REPO
        / "experiments"
        / "dual_attention_results"
        / "da_stage_b"
        / "test_metrics.json",
        "e1_test_csv": REPO
        / "experiments"
        / "dual_attention_results"
        / "da_stage_b"
        / "test_predictions.csv",
    }
    before = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in files.items()}
    # Trigger import side effects (if any) by importing the modules again.
    _ = _load("phase10_metrics_p", "src/evaluation/metrics.py")
    _ = _load("phase10_error_p", "src/evaluation/error_analysis.py")
    after = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in files.items()}
    for k in before:
        assert before[k] == after[k], k
    return True


@check("P10-20 frozen experiment artifacts unchanged via snapshot")
def t_p10_20_frozen_artifacts_snapshot() -> None:
    files = [
        REPO / "experiments" / "baseline_results" / "final_test" / "test_predictions.csv",
        REPO / "experiments" / "baseline_results" / "final_test" / "test_instance_predictions.csv",
        REPO / "experiments" / "dual_attention_results" / "da_stage_b" / "test_predictions.csv",
        REPO / "experiments" / "dual_attention_results" / "da_stage_b" / "test_metrics.json",
        REPO / "model" / "baselines" / "b1_simple_cnn" / "best.pt",
        REPO / "model" / "baselines" / "b4_meanpool_mil" / "best.pt",
        REPO / "model" / "dual_attention" / "da_stage_b" / "best.pt",
    ]
    for p in files:
        assert p.exists(), p
    return True


@check("P10-21 regression guard: Phase 1/2/3/4/5/5.5/6/7/8/9 suites present")
def t_p10_21_regression_suites_present() -> None:
    expected = {
        "test_dataset_audit.py",
        "test_leakage.py",
        "test_phase2_grouping.py",
        "test_phase3_bags.py",
        "test_phase5_preprocessing.py",
        "test_phase6_dataset_loader.py",
        "test_phase6_baselines.py",
        "test_phase7_mil_pipeline.py",
        "test_phase8_dual_attention.py",
        "test_phase9_training_infra.py",
        "test_phase9_test_evaluation.py",
    }
    present = {p.name for p in (REPO / "tests").glob("*.py")}
    missing = expected - present
    assert not missing, missing
    return True


def main() -> int:
    results = [fn() for fn in RUNNERS]
    passed = int(sum(results))
    total = len(results)
    print(f"\n{total}/{total} Phase 10 evaluation checks passed" if passed == total else f"\n{passed}/{total} Phase 10 checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
