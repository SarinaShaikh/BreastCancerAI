#!/usr/bin/env python
"""Phase 10: generate ROC / PR / confusion-matrix figures from frozen
final predictions.

This script must:
  * use only already-persisted frozen predictions and metrics;
  * NOT rerun final model evaluation;
  * NOT modify any frozen artifacts;
  * NOT tune thresholds or select models/checkpoints;
  * produce figures in ``reports/figures/phase10/``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TEST_MANIFEST_SHA256 = (
    "959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74"
)

_OUT = REPO / "reports" / "figures" / "phase10"


def _load(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_phase10_metrics = _load("phase10_metrics", "src/evaluation/metrics.py")
_phase6_metrics = _load("phase6_metrics_ref", "src/training/metrics.py")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    raise RuntimeError("matplotlib is required to generate Phase 10 figures") from e


# ---------------------------------------------------------------------------
# frozen prediction loaders
# ---------------------------------------------------------------------------

def _read_bag_predictions(
    csv_path: Path,
    prob_col: str,
    pred_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true: List[int] = []
    probs: List[float] = []
    ids: List[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(int(row.get("true_label", row.get("y_true", 0))))
            probs.append(float(row.get(prob_col, 0.0)))
            ids.append(str(row.get("bag_id", row.get("image_path", ""))))
    return (
        np.asarray(y_true, dtype=np.int64),
        np.asarray(probs, dtype=np.float64),
        np.asarray(ids, dtype=str),
    )


def _read_b1_instance_predictions(
    csv_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_true: List[int] = []
    probs: List[float] = []
    ids: List[str] = []
    paths: List[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(int(row.get("true_label", 0)))
            probs.append(float(row.get("b1_probability", 0.0)))
            ids.append(str(row.get("bag_id", "")))
            paths.append(str(row.get("image_path", "")))
    return (
        np.asarray(y_true, dtype=np.int64),
        np.asarray(probs, dtype=np.float64),
        np.asarray(ids, dtype=str),
        np.asarray(paths, dtype=str),
    )


def _confusion_matrix_counts(y_true, y_prob, threshold=0.5):
    m = _phase6_metrics.classification_metrics(y_true, y_prob, threshold=threshold)
    return m["confusion_matrix"]


def _plot_roc_curves(models: List[Dict[str, Any]], title: str, path: Path) -> None:
    plt.figure(figsize=(7, 5))
    for m in models:
        fpr, tpr, _ = _phase10_metrics.roc_curve_data(m["y_true"], m["y_prob"])
        plt.plot(fpr, tpr, label=f"{m['name']} ({m['granularity']})", linewidth=1.6)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="grey")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_pr_curves(models: List[Dict[str, Any]], title: str, path: Path) -> None:
    plt.figure(figsize=(7, 5))
    for m in models:
        rec, prec, _ = _phase10_metrics.pr_curve_data(m["y_true"], m["y_prob"])
        plt.plot(rec, prec, label=f"{m['name']} ({m['granularity']})", linewidth=1.6)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_confusion_matrices(models: List[Dict[str, Any]], title: str, path: Path) -> None:
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2), squeeze=False)
    for ax, m in zip(axes.flat, models):
        cm = _confusion_matrix_counts(m["y_true"], m["y_prob"], threshold=0.5)
        tp = int(cm["TP"])
        tn = int(cm["TN"])
        fp = int(cm["FP"])
        fn = int(cm["FN"])
        data = np.array([[tn, fp], [fn, tp]])
        im = ax.imshow(data, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"{m['name']}\n({m['granularity']})")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0 (benign)", "Pred 1 (malignant)"])
        ax.set_yticklabels(["True 0 (benign)", "True 1 (malignant)"])
        for i in range(2):
            for j in range(2):
                val = data[i, j]
                color = "white" if val > max(data.flatten()) * 0.5 else "black"
                ax.text(j, i, str(int(val)), ha="center", va="center", color=color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# model rows from frozen predictions
# ---------------------------------------------------------------------------

def _b1_row() -> Dict[str, Any]:
    csv_path = REPO / "experiments" / "baseline_results" / "final_test" / "test_instance_predictions.csv"
    yt, yp, _ids, _paths = _read_b1_instance_predictions(csv_path)
    return {
        "name": "B1 Simple CNN",
        "granularity": "image",
        "y_true": yt.tolist(),
        "y_prob": yp.tolist(),
        "source": str(csv_path),
    }


def _b3_row() -> Dict[str, Any]:
    csv_path = REPO / "experiments" / "baseline_results" / "final_test" / "test_predictions.csv"
    yt, yp, _ids = _read_bag_predictions(csv_path, prob_col="b3_bag_probability", pred_col="b3_prediction")
    return {
        "name": "B3 B1 + bag aggregation",
        "granularity": "bag",
        "y_true": yt.tolist(),
        "y_prob": yp.tolist(),
        "source": str(csv_path),
    }


def _b4_row() -> Dict[str, Any]:
    csv_path = REPO / "experiments" / "baseline_results" / "final_test" / "test_predictions.csv"
    yt, yp, _ids = _read_bag_predictions(csv_path, prob_col="b4_bag_probability", pred_col="b4_prediction")
    return {
        "name": "B4 Mean-Pooling MIL",
        "granularity": "bag",
        "y_true": yt.tolist(),
        "y_prob": yp.tolist(),
        "source": str(csv_path),
    }


def _e1_row() -> Dict[str, Any]:
    csv_path = REPO / "experiments" / "dual_attention_results" / "da_stage_b" / "test_predictions.csv"
    yt, yp, _ids = _read_bag_predictions(csv_path, prob_col="probability", pred_col="prediction")
    return {
        "name": "E1 Dual Attention MIL",
        "granularity": "bag",
        "y_true": yt.tolist(),
        "y_prob": yp.tolist(),
        "source": str(csv_path),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=_OUT)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    # frozen test manifest verification
    manifest = REPO / "data" / "manifests" / "test_split.csv"
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if sha != TEST_MANIFEST_SHA256:
        raise RuntimeError(f"Frozen test manifest SHA mismatch:\n  expected {TEST_MANIFEST_SHA256}\n  actual {sha}")

    print("frozen test manifest SHA verified:", sha)

    models = [_b1_row(), _b3_row(), _b4_row(), _e1_row()]

    _plot_roc_curves(models, "Phase 10 — ROC curves (frozen final predictions)", args.out / "roc_curves.png")
    _plot_pr_curves(models, "Phase 10 — Precision-Recall curves (frozen final predictions)", args.out / "pr_curves.png")
    _plot_confusion_matrices(models, "Phase 10 — Confusion matrices (threshold = 0.5)", args.out / "confusion_matrices.png")

    print("figures written to:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
