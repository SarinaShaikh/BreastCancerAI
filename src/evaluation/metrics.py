"""Phase 10: evaluation helpers built on the existing Phase 6 metric foundation.

Scope: Phase 10 evaluation, error analysis, and robustness testing.

Design rules:
  * Reuse the authoritative Phase 6 metric implementation
    (``src.training.metrics.classification_metrics``,
    ``src.training.metrics.confusion_counts``) rather than reimplementing
    metric logic.
  * Work from already-persisted frozen predictions and metrics wherever
    possible. Do not rerun final model evaluation to reproduce existing
    metrics.
  * Operate on the frozen test set strictly as evaluation/analysis data.
    No test-set threshold tuning, no model selection, no checkpoint
    selection, no retraining, no architecture changes.
  * Preserve provenance: every derived artifact should record the frozen
    prediction source, the frozen test manifest, and the metric definitions
    used.

Positive class convention (from Phase 6):
  malignant = 1  (benign = 0)
Threshold convention for threshold-dependent metrics:
  0.5 (fixed; never tuned on the test set)
"""
from __future__ import annotations

import csv
import hashlib
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in os.sys.path:  # noqa: F821 (stdlib typing only)
    os.sys.path.insert(0, _REPO)


def identify(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Tuple[np.ndarray, int, int, int, int]:
    """Phase-10-compatible prediction identification helper.

    Returns (y_pred, TP, TN, FP, FN) counts as integers.
    """
    t = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_prob, dtype=np.float64)
    pred = (p >= float(threshold)).astype(np.int64)
    tp = int(((pred == 1) & (t == 1)).sum())
    tn = int(((pred == 0) & (t == 0)).sum())
    fp = int(((pred == 1) & (t == 0)).sum())
    fn = int(((pred == 0) & (t == 1)).sum())
    return pred.astype(np.int64), tp, tn, fp, fn


def _classification_metrics(source: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Thin delegation to the authoritative Phase 6 metric implementation.

    ``source`` is a label used for provenance/error-message context only; it
    does not alter the metric computation.
    """
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "phase6_metrics", os.path.join(_REPO, "training", "metrics.py")
    )
    _metrics = _ilu.module_from_spec(_spec)
    os.sys.modules["phase6_metrics"] = _metrics
    _spec.loader.exec_module(_metrics)
    return _metrics.classification_metrics(*args, **kwargs)


# ---------------------------------------------------------------------------
# Frozen test-set constants
# ---------------------------------------------------------------------------

TEST_MANIFEST_SHA256 = (
    "959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74"
)

TEST_BAGS = 73
TEST_INSTANCES = 1350
TEST_BAGS_BENIGN = 43
TEST_BAGS_MALIGNANT = 30

# ---------------------------------------------------------------------------
# Frozen prediction artifact loading
# ---------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_test_manifest(root: str = _REPO) -> str:
    """Return the frozen test manifest SHA256, asserting it matches the
    canonical value. Raises if the frozen test set has drifted.

    ``root`` must be a repository root containing ``data/manifests/``.
    """
    path = os.path.join(root, "data", "manifests", "test_split.csv")
    sha = sha256_file(path)
    if sha != TEST_MANIFEST_SHA256:
        raise RuntimeError(
            "Frozen test manifest SHA mismatch:\n"
            f"  expected: {TEST_MANIFEST_SHA256}\n"
            f"  actual:   {sha}\n"
            "Do NOT continue with Phase 10 analysis."
        )
    return sha


def load_frozen_test_manifest(root: str = _REPO) -> List[Dict[str, str]]:
    """Load the rows of the frozen test-split manifest file.

    ``root`` must be a repository root containing ``data/manifests/test_split.csv``.
    The returned rows are unique by ``bag_id``; the frozen test manifest is
    stored one row per instance.
    """
    path = os.path.abspath(os.path.join(root, "data", "manifests", "test_split.csv"))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Frozen test manifest not found at {path}. "
            "Pass a repository root as ``root``."
        )
    seen: Dict[str, Dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bid = row.get("bag_id") or ""
            if bid:
                seen[bid] = row
    return list(seen.values())


def _read_predictions_csv(
    path: str, clip_probabilities: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (y_true, probability, ids) from a persisted prediction CSV.

    Returns arrays that preserve the CSV order so that provenance (bag_id,
    source_group_id, image_path) can be attached deterministically later.

    Probabilities are clipped to [0, 1] by default to match the surviving
    Phase 6 compatibility artifact.
    """
    y_true: List[int] = []
    probs: List[float] = []
    ids: List[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(int(row["y_true"]))
            probs.append(float(row["probability"]))
            ids.append(str(row.get("bag_id") or row.get("image_path") or ""))
    out_probs = np.asarray(probs, dtype=np.float64)
    if clip_probabilities:
        out_probs = _clip_probabilities(out_probs)
    return (
        np.asarray(y_true, dtype=np.int64),
        out_probs,
        ids,
    )


def _clip_probabilities(y_prob: np.ndarray) -> np.ndarray:
    """Clip probabilities to [0, 1].

    This mirrors the surviving Phase 6 compatibility helper used in the test
    harness so Phase 10 remains consistent with existing behavior without
    silently changing reported metrics.
    """
    return np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)


# ---------------------------------------------------------------------------
# ROC / PR curve data
# ---------------------------------------------------------------------------

def roc_curve_data(
    y_true: Sequence[int], y_prob: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fpr, tpr, thresholds) for the ROC curve.

    This builds on scikit-learn's ROC computation rather than reimplementing
    the underlying metric.
    """
    import sklearn.metrics as _skm
    t = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_prob, dtype=np.float64)
    if len(np.unique(t)) < 2:
        return np.array([]), np.array([]), np.array([])
    fpr, tpr, thresholds = _skm.roc_curve(t, p, pos_label=1)
    return np.asarray(fpr), np.asarray(tpr), np.asarray(thresholds)


def pr_curve_data(
    y_true: Sequence[int], y_prob: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (recall, precision, thresholds) for the PR curve."""
    import sklearn.metrics as _skm
    t = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_prob, dtype=np.float64)
    if len(np.unique(t)) < 2:
        return np.array([]), np.array([]), np.array([])
    prec, rec, thresholds = _skm.precision_recall_curve(t, p, pos_label=1)
    return np.asarray(rec), np.asarray(prec), np.asarray(thresholds)


# ---------------------------------------------------------------------------
# Comparative metric table helpers
# ---------------------------------------------------------------------------

def comparative_metrics_table(
    models: Sequence[Dict[str, Any]],
    threshold: float = 0.5,
    clip_probabilities: bool = True,
) -> List[Dict[str, Any]]:
    """Build a deterministic comparative metrics table from model rows.

    Each model row must contain at least:
      * ``model``
      * ``granularity``
      * ``y_true``
      * ``y_prob``
      * optional ``y_pred`` (if omitted, thresholded at ``threshold``)

    Returns rows in the same order as input, with metric fields added. The
    table is descriptive only.

    If ``clip_probabilities`` is True, probabilities are clipped to [0, 1]
    to match the surviving Phase 6 compatibility artifact.
    """
    rows: List[Dict[str, Any]] = []
    for m in models:
        y_true = np.asarray(m["y_true"], dtype=np.int64)
        y_prob = _clip_probabilities(np.asarray(m["y_prob"], dtype=np.float64))
        y_pred = m.get("y_pred")
        metrics = _classification_metrics(
            f"phase10:{m.get('model', 'model')}",
            y_true,
            y_prob,
            y_pred=y_pred,
            threshold=threshold,
        )
        rows.append(
            {
                "model": m["model"],
                "granularity": m["granularity"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "accuracy": metrics["accuracy"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "TP": metrics["TP"],
                "TN": metrics["TN"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
                "probability_range": [
                    float(y_prob.min()),
                    float(y_prob.max()),
                ],
                "n": int(y_true.size),
                "threshold": float(threshold),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Error/uncertainty helpers (operate on deterministic arrays; provenance is
# attached by the caller using the persisted prediction CSVs).
# ---------------------------------------------------------------------------

def classify_predictions(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Deterministic classification of each instance/bag into TP/TN/FP/FN.

    Returns a boolean mask dict:
      tp, tn, fp, fn
    """
    pred, tp, tn, fp, fn = identify(y_true, y_prob, threshold=threshold)
    t = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_prob, dtype=np.float64)
    mask_tp = (pred == 1) & (t == 1)
    mask_tn = (pred == 0) & (t == 0)
    mask_fp = (pred == 1) & (t == 0)
    mask_fn = (pred == 0) & (t == 1)
    return {
        "tp": mask_tp,
        "tn": mask_tn,
        "fp": mask_fp,
        "fn": mask_fn,
        "counts": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
    }


def error_classification(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Deterministic error classification masks.

    Returns boolean masks for TP/TN/FP/FN.
    """
    return classify_predictions(
        y_true, y_prob, threshold=threshold
    )


def uncertainty_mask(
    y_prob: Sequence[float],
    margin: float = 0.1,
    threshold: float = 0.5,
) -> np.ndarray:
    """Deterministic low-confidence mask: |p - threshold| <= margin.

    ``margin`` is fixed before inspection and must not be chosen after
    looking at which cases look interesting.
    """
    p = np.asarray(y_prob, dtype=np.float64)
    return np.abs(p - float(threshold)) <= float(margin)


def lowest_confidence_mask(
    y_prob: Sequence[float],
    k: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """Deterministic mask for the ``k`` lowest-confidence predictions
    (largest |p - threshold|). Ties are broken by array order, which is
    deterministic for persisted prediction CSVs."""
    p = np.asarray(y_prob, dtype=np.float64)
    dist = np.abs(p - float(threshold))
    if k <= 0:
        return np.zeros_like(p, dtype=bool)
    if k >= int(p.size):
        return np.ones_like(p, dtype=bool)
    idx = np.argpartition(dist, -k)[-k:]
    mask = np.zeros_like(p, dtype=bool)
    mask[idx] = True
    return mask


def highest_confidence_incorrect_mask(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
    k: int = 10,
) -> np.ndarray:
    """Deterministic mask for the ``k`` most-confident incorrect predictions
    (incorrect AND largest confidence = smallest |p - threshold| among
    incorrect cases)."""
    t = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_prob, dtype=np.float64)
    pred = (p >= float(threshold)).astype(np.int64)
    inc = (pred != t)
    if int(inc.sum()) == 0 or k <= 0:
        return np.zeros_like(p, dtype=bool)
    inc_dist = np.abs(p - float(threshold))
    k = min(k, int(inc.sum()))
    idx = np.argpartition(inc_dist, k - 1)[:k]
    mask = np.zeros_like(p, dtype=bool)
    mask[idx] = True
    return mask


def correct_incorrect_mask(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Return (correct, incorrect) boolean masks deterministically."""
    t = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_prob, dtype=np.float64)
    pred = (p >= float(threshold)).astype(np.int64)
    ok = (pred == t)
    return {"correct": ok, "incorrect": ~ok}


# ---------------------------------------------------------------------------
# Phase 10 comparative analysis helpers
# ---------------------------------------------------------------------------

def class_wise_summary(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Descriptive class-wise summary from frozen predictions.

    Returns a dict with overall metrics and per-class counts, FP, and FN.
    """
    t = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_prob, dtype=np.float64)
    pred, tp, tn, fp, fn = identify(t, p, threshold=threshold)
    benign = int((t == 0).sum())
    malignant = int((t == 1).sum())
    benign_fp = int(((pred == 1) & (t == 0)).sum())
    malignant_fn = int(((pred == 0) & (t == 1)).sum())
    specificity = 0.0 if benign == 0 else (tn / benign)
    sensitivity = 0.0 if malignant == 0 else (tp / malignant)
    overall = {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "specificity": specificity,
        "sensitivity": sensitivity,
    }
    return {
        "overall": overall,
        "benign": {
            "count": benign,
            "FP": benign_fp,
            "specificity": specificity,
        },
        "malignant": {
            "count": malignant,
            "FN": malignant_fn,
            "sensitivity": sensitivity,
        },
    }


def error_case_rows(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    ids: Sequence[str],
    provenance: Dict[str, Dict[str, Any]],
    threshold: float = 0.5,
) -> Dict[str, List[Dict[str, Any]]]:
    """Error case rows with provenance.

    Returns dicts for FP and FN cases with attached provenance fields.
    """
    pred, tp, tn, fp, fn = identify(y_true, y_prob, threshold=threshold)
    t = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_prob, dtype=np.float64)
    fp_mask = (pred == 1) & (t == 0)
    fn_mask = (pred == 0) & (t == 1)
    def _rows(mask):
        out = []
        for idx in np.where(mask)[0]:
            key = ids[idx]
            prov = provenance.get(key, {})
            out.append(
                {
                    "id": key,
                    "bag_id": prov.get("bag_id"),
                    "source_group_id": prov.get("source_group_id"),
                    "true_label": prov.get("true_label"),
                    "probability": float(p[idx]),
                    "prediction": int(pred[idx]),
                    "threshold": float(threshold),
                }
            )
        return out
    return {"FP": _rows(fp_mask), "FN": _rows(fn_mask)}


def stratify_by_bag_size(
    y_true,
    y_prob,
    ids,
    prov,
    prediction_csv_path,
    threshold=0.5,
):
    """Deterministic bag-size stratification from a prediction CSV.

    Returns dict keyed by bag size string with per-size metric/structure
    summaries.
    """
    import csv as _csv
    rows = {}
    with open(prediction_csv_path, newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            bid = r["bag_id"]
            if bid not in rows:
                rows[bid] = {
                    "n": int(r["n_instances"]),
                    "y_true": [],
                    "y_prob": [],
                    "y_pred": [],
                }
            rows[bid]["y_true"].append(int(r["y_true"]))
            rows[bid]["y_prob"].append(float(r["probability"]))
            rows[bid]["y_pred"].append(int(r["prediction"]))
    out = {}
    for bag_id, d in rows.items():
        y_true_bag = np.asarray(d["y_true"], dtype=np.int64)
        y_prob_bag = np.asarray(d["y_prob"], dtype=np.float64)
        size = str(d["n"])
        metrics = _classification_metrics(
            f"phase10:bag_size_{size}", y_true_bag, y_prob_bag,
            y_pred=np.asarray(d["y_pred"], dtype=np.int64), threshold=threshold
        )
        out.setdefault(size, {
            "n_cases": 0,
            "roc_auc": [],
            "pr_auc": [],
            "accuracy": [],
            "sensitivity": [],
            "specificity": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "TP": 0,
            "TN": 0,
            "FP": 0,
            "FN": 0,
        })
        out[size]["n_cases"] += d["n"]
        for k in ("roc_auc", "pr_auc", "accuracy", "sensitivity", "specificity",
                  "precision", "recall", "f1"):
            out[size][k].append(float(metrics[k]))
        out[size]["TP"] += int(metrics["TP"])
        out[size]["TN"] += int(metrics["TN"])
        out[size]["FP"] += int(metrics["FP"])
        out[size]["FN"] += int(metrics["FN"])
    for size in out:
        for k in ("roc_auc", "pr_auc", "accuracy", "sensitivity", "specificity",
                  "precision", "recall", "f1"):
            vals = out[size][k]
            out[size][k] = float(np.mean(vals)) if vals else 0.0
    return out


def stratify_by_source_group(
    y_true,
    y_prob,
    ids,
    prov,
    threshold=0.5,
):
    """Deterministic source-group stratification from IDs + provenance.

    Returns dict keyed by source_group_id with per-group metric summaries.
    """
    groups = {}
    for idx, key in enumerate(ids):
        grp = prov.get(key, {}).get("source_group_id", "unknown")
        groups.setdefault(grp, {
            "y_true": [],
            "y_prob": [],
            "y_pred": [],
        })
        groups[grp]["y_true"].append(int(y_true[idx]))
        groups[grp]["y_prob"].append(float(y_prob[idx]))
        groups[grp]["y_pred"].append(int((float(y_prob[idx]) >= threshold)))
    out = {}
    for grp, d in groups.items():
        yt = np.asarray(d["y_true"], dtype=np.int64)
        yp = np.asarray(d["y_prob"], dtype=np.float64)
        ypd = np.asarray(d["y_pred"], dtype=np.int64)
        metrics = _classification_metrics(
            f"phase10:source_group_{grp}", yt, yp,
            y_pred=ypd, threshold=threshold
        )
        out[grp] = {
            "n_cases": int(yt.size),
            "TP": int(metrics["TP"]),
            "TN": int(metrics["TN"]),
            "FP": int(metrics["FP"]),
            "FN": int(metrics["FN"]),
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
        }
    return out


def model_disagreement_rows(
    rows_a: Sequence[Dict[str, Any]],
    rows_b: Sequence[Dict[str, Any]],
    id_key: str = "bag_id",
    prob_key: str = "probability",
    pred_key: str = "prediction",
    label_key: str = "true_label",
) -> List[Dict[str, Any]]:
    """Deterministic model-disagreement rows at compatible granularity.

    Returns rows where two models disagree on prediction for the same unit.
    This is intended for compatible-granularity comparisons, e.g. bag-level
    B3/B4/E1. It must not be used to treat B1 instance predictions as
    equivalent to bag-level predictions.
    """
    a = {(r.get(id_key) or ""): r for r in rows_a}
    b = {(r.get(id_key) or ""): r for r in rows_b}
    common = sorted(set(a.keys()) & set(b.keys()))
    out: List[Dict[str, Any]] = []
    for key in common:
        ra = a[key]
        rb = b[key]
        pa = ra.get(prob_key)
        pb = rb.get(prob_key)
        if pa is None or pb is None:
            continue
        try:
            pred_a = int(ra.get(pred_key) or 0)
            pred_b = int(rb.get(pred_key) or 0)
        except (TypeError, ValueError):
            continue
        out.append(
            {
                id_key: ra.get(id_key),
                "source_group_id_a": ra.get("source_group_id"),
                "source_group_id_b": rb.get("source_group_id"),
                "probability_a": float(pa),
                "probability_b": float(pb),
                "prediction_a": pred_a,
                "prediction_b": pred_b,
                "true_label": ra.get(label_key),
            }
        )
    return out


def class_wise_counts(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, Any]:

    """Return counts of TP/TN/FP/FN and per-class counts.

    This is a convenience helper for tables that need counts without the
    full comparative-metrics table object.
    """
    t = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_prob, dtype=np.float64)
    pred, tp, tn, fp, fn = identify(t, p, threshold=threshold)
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "n_benign": int((t == 0).sum()),
        "n_malignant": int((t == 1).sum()),
    }
