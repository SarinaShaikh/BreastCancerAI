"""Phase 9: single final test evaluation for E1 ``da_stage_b``, with a hard
single-evaluation guard (analogous to the Phase 6 STEP G guard).

This is the one-time held-out test evaluation requested separately after E1
training completed. It does NOT retrain anything, does NOT touch the frozen
test manifest, and does NOT allow a second evaluation unless the persisted
state is deliberately reset (which is NOT authorized here).

Guard design (mirrors Phase 6 STEP G):
  * If the persisted Phase 9 state already records ``test: evaluated``,
    the suite refuses to re-run (single-evaluation rule).
  * If the frozen test manifest SHA does not match the canonical value,
    the evaluation is refused.
  * Test splits are constructed ONLY through the Phase 5.5 loader over the
    frozen Phase 4 manifests; no direct filesystem path reconstruction is
    used by the evaluation path.
  * Augmentation is NONE for the test evaluation.
  * Threshold is fixed at 0.5 and is never tuned on these results.

This file:
  1. Verifies the frozen test-manifest SHA and test population.
  2. Loads the frozen E1 best checkpoint.
  3. Computes deterministic bag-level test predictions.
  4. Persists test_predictions.csv and test_metrics.json.
  5. Records the single-evaluation state.
  6. Independently recomputes metrics from the persisted CSV and verifies
     exact agreement.
  7. Prints the evaluation summary and a descriptive frozen-baseline
     comparison table (no winner/selection declared).
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util as _ilu
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _load_module(name: str, relpath: str):
    spec = _ilu.spec_from_file_location(name, os.path.join(_REPO, relpath))
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Frozen constants (single source of truth for this evaluation)
# ---------------------------------------------------------------------------

TEST_MANIFEST_SHA256 = (
    "959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74"
)

EXPECTED_TEST_BAGS = 73
EXPECTED_TEST_INSTANCES = 1350
EXPECTED_TEST_BAGS_BENIGN = 43
EXPECTED_TEST_BAGS_MALIGNANT = 30

CHECKPOINT_PATH = os.path.join(
    _REPO, "model", "dual_attention", "da_stage_b", "best.pt"
)

TEST_PREDICTIONS_CSV = os.path.join(
    _REPO, "experiments", "dual_attention_results", "da_stage_b",
    "test_predictions.csv",
)

TEST_METRICS_JSON = os.path.join(
    _REPO, "experiments", "dual_attention_results", "da_stage_b",
    "test_metrics.json",
)

_classification_metrics = None


def classification_metrics(*a, **kw):
    global _classification_metrics
    if _classification_metrics is None:
        _metrics = _load_module("phase9_test_metrics", "src/training/metrics.py")
        _classification_metrics = _metrics.classification_metrics
    return _classification_metrics(*a, **kw)


# ---------------------------------------------------------------------------
# State guard
# ---------------------------------------------------------------------------

def _state_path() -> str:
    return os.path.join(_REPO, "experiments", "dual_attention_results", "_state.json")


def _load_state() -> dict:
    p = _state_path()
    if not os.path.isfile(p):
        return {"test": "pending"}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _save_state(st: dict) -> None:
    os.makedirs(os.path.dirname(_state_path()), exist_ok=True)
    with open(_state_path(), "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)


def _record_test_evaluated(st: dict) -> None:
    st["test"] = "evaluated"
    st["history"].append({
        "step": "da_stage_b_test_eval",
        "frozen_test_sha256_verified": True,
        "checkpoint_path": CHECKPOINT_PATH,
    })
    _save_state(st)


# ---------------------------------------------------------------------------
# Test-manifest integrity
# ---------------------------------------------------------------------------

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_test_manifest_sha(root: str = _REPO) -> str:
    manifest = os.path.join(root, "data", "manifests", "test_split.csv")
    sha = _sha256_file(manifest)
    if sha != TEST_MANIFEST_SHA256:
        raise RuntimeError(
            "Frozen test manifest SHA mismatch:\n"
            f"  expected: {TEST_MANIFEST_SHA256}\n"
            f"  actual:   {sha}\n"
            "Do NOT proceed with test evaluation."
        )
    return sha


def verify_test_population_from_loader(root: str = _REPO) -> Dict[str, object]:
    _loader = _load_module("phase9_test_loader", "src/mil/bag_dataset.py")
    UltrasoundBagDataset = _loader.UltrasoundBagDataset

    ds = UltrasoundBagDataset("test", root=root)
    bags = list(ds.bags)
    instances = list(ds.instances)

    benign = sum(1 for b in bags if b.label == "benign")
    malignant = sum(1 for b in bags if b.label == "malignant")

    checks = {
        "bags": len(bags),
        "instances": len(instances),
        "benign_bags": benign,
        "malignant_bags": malignant,
        "bag_size_min": min(len(b.instances) for b in bags),
        "bag_size_max": max(len(b.instances) for b in bags),
    }
    expected = {
        "bags": EXPECTED_TEST_BAGS,
        "instances": EXPECTED_TEST_INSTANCES,
        "benign_bags": EXPECTED_TEST_BAGS_BENIGN,
        "malignant_bags": EXPECTED_TEST_BAGS_MALIGNANT,
    }
    for k, ev in expected.items():
        gv = checks[k]
        if gv != ev:
            raise RuntimeError(
                f"Test population mismatch for {k}: expected {ev}, got {gv}"
            )
    return checks


# ---------------------------------------------------------------------------
# Model + checkpoint loading
# ---------------------------------------------------------------------------

def load_e1_model(root: str = _REPO) -> Tuple[Any, Dict[str, object]]:
    _td = _load_module("phase9_test_td", "src/training/train_dual_attention.py")
    from src.models.dual_attention_mil import DualAttentionMIL

    model = DualAttentionMIL()
    payload = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    torch.set_num_threads(8)

    meta = {
        "checkpoint_path": CHECKPOINT_PATH,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_seed": int(payload.get("seed", -1)),
        "checkpoint_validation_metric": float(payload.get("validation_metric", float("nan"))),
        "frozen_test_sha256_embedded": str(payload.get("frozen_test_sha256", "")),
        "inherited_checkpoint": payload.get("inherited_checkpoint"),
        "preprocessing_version": str(payload.get("preprocessing_version", "")),
        "pipeline_version": str(payload.get("pipeline_version", "")),
        "n_state_dict_tensors": len(payload.get("model_state_dict", {})),
    }
    return model, meta


# ---------------------------------------------------------------------------
# Deterministic test prediction (bags in loader order, no augmentation)
# ---------------------------------------------------------------------------

def compute_test_predictions(
    model: Any,
    root: str = _REPO,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    _td = _load_module("phase9_test_td2", "src/training/train_dual_attention.py")

    _loader = _load_module("phase9_test_loader2", "src/mil/bag_dataset.py")
    UltrasoundBagDataset = _loader.UltrasoundBagDataset

    _ig = _load_module("phase9_test_ig", "src/mil/instance_generation.py")

    ds = UltrasoundBagDataset("test", root=root)

    probs_list: List[float] = []
    labels_list: List[int] = []
    ids_list: List[str] = []

    model.eval()
    with torch.no_grad():
        for b in ds.bags:
            batch = _ig.load_bag_batch(b, ds)
            logit = _td._bag_forward_logit(model, batch.images)
            probs_list.append(float(torch.sigmoid(logit).item()))
            labels_list.append(int(batch.numeric_label))
            ids_list.append(str(batch.bag_id))

    return (
        np.array(probs_list, dtype=np.float64),
        np.array(labels_list, dtype=np.int64),
        ids_list,
    )


# ---------------------------------------------------------------------------
# Persistence + recomputation
# ---------------------------------------------------------------------------

def write_test_predictions_csv(
    probs: np.ndarray,
    labels: np.ndarray,
    bag_ids: List[str],
    root: str = _REPO,
) -> None:
    _loader = _load_module("phase9_test_loader3", "src/mil/bag_dataset.py")
    ds = _loader.UltrasoundBagDataset("test", root=root)
    by_id = {b.bag_id: b for b in ds.bags}

    pred = (probs >= 0.5).astype(np.int64)

    os.makedirs(os.path.dirname(TEST_PREDICTIONS_CSV), exist_ok=True)
    with open(TEST_PREDICTIONS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "bag_id", "source_group_id", "y_true", "probability",
            "prediction", "split",
        ])
        for p, y, bid, pd_ in zip(probs, labels, bag_ids, pred):
            b = by_id[bid]
            w.writerow([
                bid,
                b.source_group_id,
                int(y),
                repr(float(p)),
                int(pd_),
                "test",
            ])


def recompute_metrics_from_predictions_csv(
    csv_path: str = TEST_PREDICTIONS_CSV,
    threshold: float = 0.5,
) -> Dict[str, object]:
    y_true: List[int] = []
    probs: List[float] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            y_true.append(int(row["y_true"]))
            probs.append(float(row["probability"]))

    y_true = np.array(y_true, dtype=np.int64)
    probs = np.array(probs, dtype=np.float64)
    metrics = classification_metrics(y_true, probs, threshold=threshold)
    metrics["probability_range"] = [float(probs.min()), float(probs.max())]
    return metrics


def write_test_metrics_json(
    driver_metrics: Dict[str, object],
    recomputed_metrics: Dict[str, object],
    checkpoint_meta: Dict[str, object],
    frozen_test_sha: str,
    root: str = _REPO,
) -> None:
    os.makedirs(os.path.dirname(TEST_METRICS_JSON), exist_ok=True)

    _gitmod = _load_module("phase9_test_git", "src/training/train_baseline.py")

    record = {
        "evaluation": "PHASE_9_E1_SINGLE_FINAL_TEST_EVALUATION",
        "evaluation_timestamp_utc": _utc_now(),
        "status": "held_out_reporting_only_not_clinical_validation",
        "test_set": {
            "bags": EXPECTED_TEST_BAGS,
            "instances": EXPECTED_TEST_INSTANCES,
            "bag_class_counts": {
                "benign": EXPECTED_TEST_BAGS_BENIGN,
                "malignant": EXPECTED_TEST_BAGS_MALIGNANT,
            },
            "bag_size_distribution": _bag_size_distribution(root=root),
        },
        "threshold": 0.5,
        "threshold_tuned": False,
        "test_evaluated": True,
        "seed": 20260918,
        "model": {
            "checkpoint": CHECKPOINT_PATH,
            "checkpoint_md5": _md5_file(CHECKPOINT_PATH),
            "checkpoint_epoch": checkpoint_meta["checkpoint_epoch"],
            "evaluation_level": "bag",
            "frozen_test_sha256": frozen_test_sha,
            "frozen_test_sha256_verified": frozen_test_sha == TEST_MANIFEST_SHA256,
            "metrics": driver_metrics,
            "probability_range": driver_metrics["probability_range"],
        },
        "validation_reference": {
            "best_epoch": checkpoint_meta["checkpoint_epoch"],
            "best_val_bag_roc_auc": checkpoint_meta["checkpoint_validation_metric"],
            "note": "checkpoint selection used VALIDATION only; test results used for held-out reporting only",
        },
        "independent_verification": {
            "method": "metrics recomputed from persisted test_predictions.csv",
            "driver_and_recomputed_match": _compare_metrics(driver_metrics, recomputed_metrics),
            "tolerance": 1e-9,
        },
        "provenance": {
            "frozen_phase4_splits_used": True,
            "frozen_test_sha256": frozen_test_sha,
            "frozen_test_sha256_verified": frozen_test_sha == TEST_MANIFEST_SHA256,
            "phase5_preprocessing_used": True,
            "test_augmentation": "none",
            "e1_evaluated_from_frozen_best_checkpoint": True,
            "best_checkpoint_frozen": True,
            "no_model_retrained": True,
            "no_threshold_tuned": True,
            "no_test_result_used_for_model_selection": True,
            "no_pretrained_imagenet_weights": True,
            "no_mri_or_cross_modal_validation": True,
            "no_dual_attention_training": True,
            "clinical_claim": "held-out experimental evaluation on a research prototype; NOT clinical validation and NOT clinically deployable",
            "source_commit": _gitmod._git_commit(),
            "manifest_hashes": _gitmod._manifest_hashes(),
            "preprocessing_version": "1.0.0",
            "pipeline_version": "1.0.1",
            "inherited_checkpoint": None,
        },
    }
    with open(TEST_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


# ---------------------------------------------------------------------------
# Comparison against frozen Phase 6 baselines (read-only)
# ---------------------------------------------------------------------------

def _frozen_baseline_test_metrics() -> Dict[str, Dict[str, object]]:
    base = os.path.join(_REPO, "experiments", "baseline_results", "final_test")
    with open(os.path.join(base, "metrics.json"), encoding="utf-8") as f:
        doc = json.load(f)
    return {
        "B1_simple_cnn": {
            "roc_auc": doc["models"]["B1_simple_cnn"]["metrics"]["roc_auc"],
            "pr_auc": doc["models"]["B1_simple_cnn"]["metrics"]["pr_auc"],
            "accuracy": doc["models"]["B1_simple_cnn"]["metrics"]["accuracy"],
            "sensitivity": doc["models"]["B1_simple_cnn"]["metrics"]["sensitivity"],
            "specificity": doc["models"]["B1_simple_cnn"]["metrics"]["specificity"],
            "precision": doc["models"]["B1_simple_cnn"]["metrics"]["precision"],
            "recall": doc["models"]["B1_simple_cnn"]["metrics"]["recall"],
            "f1": doc["models"]["B1_simple_cnn"]["metrics"]["f1"],
            "confusion_matrix": dict(doc["models"]["B1_simple_cnn"]["metrics"]["confusion_matrix"]),
            "level": doc["models"]["B1_simple_cnn"]["evaluation_level"],
            "checkpoint_epoch": doc["models"]["B1_simple_cnn"]["checkpoint_epoch"],
        },
        "B3_bag_aggregation_of_B1": {
            "roc_auc": doc["models"]["B3_bag_aggregation_of_B1"]["metrics"]["roc_auc"],
            "pr_auc": doc["models"]["B3_bag_aggregation_of_B1"]["metrics"]["pr_auc"],
            "accuracy": doc["models"]["B3_bag_aggregation_of_B1"]["metrics"]["accuracy"],
            "sensitivity": doc["models"]["B3_bag_aggregation_of_B1"]["metrics"]["sensitivity"],
            "specificity": doc["models"]["B3_bag_aggregation_of_B1"]["metrics"]["specificity"],
            "precision": doc["models"]["B3_bag_aggregation_of_B1"]["metrics"]["precision"],
            "recall": doc["models"]["B3_bag_aggregation_of_B1"]["metrics"]["recall"],
            "f1": doc["models"]["B3_bag_aggregation_of_B1"]["metrics"]["f1"],
            "confusion_matrix": dict(doc["models"]["B3_bag_aggregation_of_B1"]["metrics"]["confusion_matrix"]),
            "level": doc["models"]["B3_bag_aggregation_of_B1"]["evaluation_level"],
            "checkpoint_epoch": doc["models"]["B3_bag_aggregation_of_B1"]["checkpoint_epoch"],
        },
        "B4_meanpool_mil": {
            "roc_auc": doc["models"]["B4_meanpool_mil"]["metrics"]["roc_auc"],
            "pr_auc": doc["models"]["B4_meanpool_mil"]["metrics"]["pr_auc"],
            "accuracy": doc["models"]["B4_meanpool_mil"]["metrics"]["accuracy"],
            "sensitivity": doc["models"]["B4_meanpool_mil"]["metrics"]["sensitivity"],
            "specificity": doc["models"]["B4_meanpool_mil"]["metrics"]["specificity"],
            "precision": doc["models"]["B4_meanpool_mil"]["metrics"]["precision"],
            "recall": doc["models"]["B4_meanpool_mil"]["metrics"]["recall"],
            "f1": doc["models"]["B4_meanpool_mil"]["metrics"]["f1"],
            "confusion_matrix": dict(doc["models"]["B4_meanpool_mil"]["metrics"]["confusion_matrix"]),
            "level": doc["models"]["B4_meanpool_mil"]["evaluation_level"],
            "checkpoint_epoch": doc["models"]["B4_meanpool_mil"]["checkpoint_epoch"],
        },
    }


def da_test_comparison_table(
    da_metrics: Dict[str, object],
) -> Dict[str, Dict[str, object]]:
    baselines = _frozen_baseline_test_metrics()
    row = {
        "level": "bag",
        "roc_auc": da_metrics["roc_auc"],
        "pr_auc": da_metrics["pr_auc"],
        "accuracy": da_metrics["accuracy"],
        "sensitivity": da_metrics["sensitivity"],
        "specificity": da_metrics["specificity"],
        "precision": da_metrics["precision"],
        "recall": da_metrics["recall"],
        "f1": da_metrics["f1"],
        "confusion_matrix": da_metrics["confusion_matrix"],
        "checkpoint_epoch": _checkpoint_epoch(),
        "model": "E1_dual_attention_mil_da_stage_b",
    }
    return {"E1_dual_attention_mil_da_stage_b": row, **baselines}


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _checkpoint_epoch() -> int:
    payload = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    return int(payload.get("epoch", -1))


def _bag_size_distribution(root: str = _REPO) -> Dict[str, int]:
    _loader = _load_module("phase9_test_loader4", "src/mil/bag_dataset.py")
    ds = _loader.UltrasoundBagDataset("test", root=root)
    from collections import Counter
    c = Counter(len(b.instances) for b in ds.bags)
    return {str(k): int(v) for k, v in sorted(c.items())}


def _compare_metrics(a: Dict[str, object], b: Dict[str, object]) -> Dict[str, bool]:
    keys = [
        "accuracy", "sensitivity", "specificity", "precision", "recall",
        "f1", "roc_auc", "pr_auc",
    ]
    out: Dict[str, bool] = {}
    for k in keys:
        av = a.get(k)
        bv = b.get(k)
        out[f"{k}_match"] = (
            (av is None and bv is None)
            or (av is not None and bv is not None and abs(float(av) - float(bv)) < 1e-9)
        )
    ca = a.get("confusion_matrix", {})
    cb = b.get("confusion_matrix", {})
    out["confusion_matrix_match"] = bool(ca is not None and cb is not None and dict(ca) == dict(cb))
    return out


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------

def _metric_lines(m: Dict[str, object]) -> str:
    pm = m.get("confusion_matrix", {})
    def f(k):
        v = m.get(k)
        return "None" if v is None else f"{v:.6f}"
    return (
        f"  n                 = {m.get('n')}\n"
        f"  ROC-AUC           = {f('roc_auc')}\n"
        f"  PR-AUC            = {f('pr_auc')}\n"
        f"  accuracy          = {f('accuracy')}\n"
        f"  sensitivity       = {f('sensitivity')}\n"
        f"  specificity       = {f('specificity')}\n"
        f"  precision         = {f('precision')}\n"
        f"  recall            = {f('recall')}\n"
        f"  F1                = {f('f1')}\n"
        f"  confusion matrix  = TP={pm.get('TP')} TN={pm.get('TN')} "
        f"FP={pm.get('FP')} FN={pm.get('FN')}\n"
        f"  probability range = {m.get('probability_range')}\n"
    )


def print_comparison_table(table: Dict[str, Dict[str, object]]) -> None:
    rows = list(table.values())
    metrics = ["roc_auc", "pr_auc", "accuracy", "sensitivity", "specificity",
               "precision", "recall", "f1"]
    header = (
        f"{'Model':42} {'Level':7} {'ROC-AUC':10} {'PR-AUC':10} "
        f"{'Acc':10} {'Sens':10} {'Spec':10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        name = row.get("model", row.get("checkpoint_epoch", ""))
        if isinstance(name, int):
            name = f"E1_dual_attention_mil (epoch {name})"
        lvl = str(row.get("level", ""))
        def g(k):
            v = row.get(k)
            return "None" if v is None else f"{v:.4f}"
        print(
            f"{name[:42]:42} {lvl:7} {g('roc_auc'):10} {g('pr_auc'):10} "
            f"{g('accuracy'):10} {g('sensitivity'):10} {g('specificity'):10}"
        )
    print()
    print("Descriptive comparison only — no winner/selection/ranking is "
          "declared or implied.")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_test_evaluation(root: str = _REPO) -> Dict[str, Any]:
    st = _load_state()

    if st.get("test") in ("done", "evaluated"):
        raise RuntimeError(
            "Phase 9 single final test evaluation already recorded "
            "(single-evaluation rule). Do NOT re-run."
        )

    print("=== PHASE 9 E1 single final test evaluation ===", flush=True)

    print("verifying frozen test manifest SHA ...", flush=True)
    frozen_sha = verify_test_manifest_sha(root)

    print("verifying test population via loader ...", flush=True)
    population = verify_test_population_from_loader(root)
    print(f"  bags={population['bags']} instances={population['instances']} "
          f"benign_bags={population['benign_bags']} "
          f"malignant_bags={population['malignant_bags']}",
          flush=True)

    print("loading frozen E1 best checkpoint ...", flush=True)
    model, meta = load_e1_model(root)
    print(f"  epoch={meta['checkpoint_epoch']} seed={meta['checkpoint_seed']} "
          f"val_metric={meta['checkpoint_validation_metric']:.6f} "
          f"n_tensors={meta['n_state_dict_tensors']} "
          f"inherited_checkpoint={meta['inherited_checkpoint']}",
          flush=True)

    print("computing deterministic test predictions ...", flush=True)
    probs, labels, bag_ids = compute_test_predictions(model, root=root)
    print(f"  computed {len(probs)} bag predictions", flush=True)

    print("persisting test_predictions.csv ...", flush=True)
    write_test_predictions_csv(probs, labels, bag_ids, root=root)

    print("computing driver metrics ...", flush=True)
    driver = classification_metrics(labels, probs, threshold=0.5)
    driver["probability_range"] = [float(probs.min()), float(probs.max())]

    print("persisting test_metrics.json ...", flush=True)
    write_test_metrics_json(driver, {}, meta, frozen_sha, root=root)
    _record_test_evaluated(st)

    print("independent recomputation from persisted CSV ...", flush=True)
    recomputed = recompute_metrics_from_predictions_csv(TEST_PREDICTIONS_CSV, 0.5)
    write_test_metrics_json(driver, recomputed, meta, frozen_sha, root=root)

    match = _compare_metrics(driver, recomputed)
    print("independent recomputation match:", flush=True)
    for k, v in match.items():
        print(f"  {k}: {v}", flush=True)
    assert all(match.values()), "independent recomputation mismatch"

    print("=== E1 test metrics ===", flush=True)
    print(_metric_lines(driver), flush=True)

    print("=== frozen baseline comparison (descriptive, no winner declared) ===", flush=True)
    table = da_test_comparison_table(driver)
    print_comparison_table(table)

    return {
        "frozen_test_sha256": frozen_sha,
        "test_population": population,
        "checkpoint_meta": meta,
        "driver_metrics": driver,
        "recomputed_metrics": recomputed,
        "recomputation_match": match,
        "comparison_table": table,
        "state_after": _load_state(),
    }


def main() -> int:
    try:
        run_test_evaluation()
        print("=== PHASE 9 E1 TEST EVALUATION COMPLETE ===", flush=True)
        return 0
    except Exception as e:
        print("=== PHASE 9 E1 TEST EVALUATION FAILED ===", flush=True)
        print(str(e), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
