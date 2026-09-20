"""Phase 10: deterministic error analysis for the frozen final predictions.

Scope: analysis only. No retraining, no threshold tuning, no model selection,
no checkpoint modification, and no new final evaluation.

This module operates on already-persisted frozen predictions and attaches
provenance from the persisted prediction CSVs so that every identified case
can be traced back to bag_id / source_group_id / image_path where
applicable.

Positive class convention (from Phase 6):
  malignant = 1, benign = 0
Threshold convention for classification metrics:
  0.5 (fixed; never tuned on the test set)
"""
from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.evaluation.metrics import (
    classify_predictions,
    uncertainty_mask,
    lowest_confidence_mask,
    highest_confidence_incorrect_mask,
    class_wise_summary,
    error_case_rows,
    stratify_by_bag_size,
    stratify_by_source_group,
    model_disagreement_rows as _model_disagreement_rows,
)


def error_classification(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Deterministic error classification masks.

    Returns boolean masks for TP/TN/FP/FN.
    """
    return classify_predictions(y_true, y_prob, threshold=threshold)


def error_case_rows_full(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    ids: Sequence[str],
    provenance: Optional[Dict[str, Dict[str, Any]]] = None,
    threshold: float = 0.5,
) -> Dict[str, List[Dict[str, Any]]]:
    """Deterministic FP/FN case rows with provenance attached.

    Returns TP/TN/FP/FN rows. Each row contains the available provenance
    fields for the case.
    """
    t = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_prob, dtype=np.float64)
    pred = (p >= float(threshold)).astype(np.int64)
    pos = (t == 1)
    neg = (t == 0)

    def _rows(mask):
        out: List[Dict[str, Any]] = []
        for i in np.where(mask)[0]:
            prov: Dict[str, Any] = {
                "index": int(i),
                "id": str(ids[i]) if i < len(ids) else "",
                "true_label": int(t[i]),
                "pred_label": int(pred[i]),
                "probability": float(p[i]),
                "threshold": float(threshold),
            }
            if provenance is not None:
                key = str(ids[i]) if i < len(ids) else ""
                prov.update(provenance.get(key, {}))
            out.append(prov)
        return out

    tp_mask = (pred == 1) & pos
    tn_mask = (pred == 0) & neg
    fp_mask = (pred == 1) & neg
    fn_mask = (pred == 0) & pos

    return {
        "TP": _rows(tp_mask),
        "TN": _rows(tn_mask),
        "FP": _rows(fp_mask),
        "FN": _rows(fn_mask),
        "counts": {
            "TP": int(tp_mask.sum()),
            "TN": int(tn_mask.sum()),
            "FP": int(fp_mask.sum()),
            "FN": int(fn_mask.sum()),
        },
    }


def model_disagreement_rows(
    rows_a: Sequence[Dict[str, Any]],
    rows_b: Sequence[Dict[str, Any]],
    id_key: str = "bag_id",
) -> List[Dict[str, Any]]:
    """Deterministic disagreement identification between two bag-level models.

    Both inputs must be aligned by the same id_key and evaluated at the same
    granularity. B1 image-level predictions must NOT be compared directly to
    bag-level predictions as if they were the same unit.
    """
    return _model_disagreement_rows(rows_a, rows_b, id_key=id_key)