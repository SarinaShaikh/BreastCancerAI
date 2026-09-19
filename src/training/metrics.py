"""Phase 6: reusable baseline metrics (scikit-learn based).

Roadmap: PROGRESS.md Phase 6 "Baseline Deep Learning Models". Every metric
requested by the Phase 6 instruction is exposed here and reused by
src/training/train_baseline.py so B1/B3/B4 report in one consistent format.

Discipline:
  - Sensitivity is computed explicitly as TP / (TP + FN).
  - Specificity is computed explicitly as TN / (TN + FP).
  - Undefined cases (zero denominators, single-class input) are NEVER silently
    replaced by a misleading number: they are recorded as None and surfaced in
    the returned dict under "undefined" so no downstream report can mistake
    them for real values.
  - Positive class = malignant (1), per the documented LABEL_MAP convention
    {benign: 0, malignant: 1}.

No clinical claims are made by or for these metrics (research prototype).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

POSITIVE_LABEL = 1  # malignant


def _safe_div(num: float, den: float) -> Optional[float]:
    """Deterministic division guard; None when the denominator is zero."""
    return float(num) / float(den) if den else None


def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, int]:
    """TP/TN/FP/FN from binary labels (positive class = 1)."""
    t = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_pred, dtype=np.int64)
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: {t.shape} vs {p.shape}")
    tn, fp, fn, tp = confusion_matrix(t, p, labels=[0, 1]).ravel().tolist()
    return {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}


def classification_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    y_pred: Optional[Sequence[int]] = None,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Full metric dict at one evaluation level (image OR bag).

    y_true : binary ground truth (0/1; positive = malignant)
    y_prob : predicted probability of the positive class
    y_pred : optional pre-computed binary predictions; if omitted they are
             thresholded from y_prob at `threshold`.

    Undefined ROC-AUC / PR-AUC (single-class truth) are recorded as None and
    listed in the "undefined" key — never faked.
    """
    t = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_prob, dtype=np.float64)
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: {t.shape} vs {p.shape}")
    if y_pred is None:
        pred = (p >= float(threshold)).astype(np.int64)
    else:
        pred = np.asarray(y_pred, dtype=np.int64)
        if pred.shape != t.shape:
            raise ValueError(f"shape mismatch: {pred.shape} vs {t.shape}")

    c = confusion_counts(t, pred)
    undefined: Dict[str, str] = []

    sens = _safe_div(c["TP"], c["TP"] + c["FN"])          # TP / (TP + FN)
    if sens is None:
        undefined.append("sensitivity (TP + FN == 0)")
    spec = _safe_div(c["TN"], c["TN"] + c["FP"])          # TN / (TN + FP)
    if spec is None:
        undefined.append("specificity (TN + FP == 0)")
    prec = _safe_div(c["TP"], c["TP"] + c["FP"])
    if prec is None:
        undefined.append("precision (TP + FP == 0)")
    rec = _safe_div(c["TP"], c["TP"] + c["FN"])
    if rec is None:
        undefined.append("recall (TP + FN == 0)")

    # F1 from explicit P/R so its undefined-ness matches the components.
    f1 = (_safe_div(2 * prec * rec, prec + rec)
          if (prec is not None and rec is not None) else None)
    if f1 is None:
        undefined.append("f1 (precision/recall undefined or 0 denominator)")

    roc = pr = None
    if len(np.unique(t)) < 2:
        undefined.append("roc_auc (single-class truth)")
        undefined.append("pr_auc (single-class truth)")
    else:
        roc = float(roc_auc_score(t, p))
        pr = float(average_precision_score(t, p))

    return {
        "n": int(t.size),
        "threshold": float(threshold),
        "TP": c["TP"], "TN": c["TN"], "FP": c["FP"], "FN": c["FN"],
        "confusion_matrix": {"TP": c["TP"], "TN": c["TN"],
                             "FP": c["FP"], "FN": c["FN"]},
        "accuracy": float(accuracy_score(t, pred)),
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": pr,
        "undefined": undefined,
    }
