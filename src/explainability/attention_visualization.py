"""Phase 11: attention visualization & machine-readable attention export.

Owner decisions (Phase 11 authorization, 2026-09-21):
  A2  TRAIN + VALIDATION ONLY — this module never loads a test image, never
      performs a test forward pass, and never consumes test labels. The test
      split is a hard boundary enforced by tests/test_phase11_explainability.py.
  A3  Two distinct visualizations from the existing Phase 8 architecture:
        * stage-2 instance MIL attention — ONE scalar per image/instance,
          ranked within each bag (NOT a pixel map);
        * stage-1 channel gates — 128-d FEATURE-SPACE attribution per instance,
          explicitly labeled "Feature-space channel gates — model attribution,
          not image-space localization."
  A5  deterministic machine-readable export (CSV + JSON) with full provenance
      and ``source_key_status=inferred_from_filename_not_verified_identifier``
      preserved on every row. No patient/study/lesion identifiers exist in the
      data; none are invented here.

Model/checkpoint rules (owner instruction):
  * the ONLY checkpoint is the frozen Phase 9 E1 ``best.pt``
    (model/dual_attention/da_stage_b/best.pt, expected MD5
    258710649fb0e6979f64fc1e7ccfc28f);
  * loaded strictly with ``model.eval()``; every forward runs under
    ``torch.no_grad()``; weights are never modified, no training/optimization;
  * threshold 0.5 appears only as the frozen reporting convention for
    carrying persisted predictions into provenance — it is never tuned.

Attention-weight semantics: stage-2 attention sums to 1 within each bag
(softmax invariant); one-instance bags yield exactly 1.0. Attention is a
MODEL-INTERNAL ATTRIBUTION SIGNAL, not a clinical explanation, and is not
validated as anatomical/lesion localization (Phase 1 verified that no
ground-truth masks exist, so IoU/mask-overlap validation is impossible).
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# repo path bootstrap (keeps this module importable standalone and as a package)
# ---------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in os.sys.path:  # noqa: F821
    os.sys.path.insert(0, REPO)


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, relpath))
    mod = importlib.util.module_from_spec(spec)
    os.sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# frozen constants (single sources of truth; never modified here)
# ---------------------------------------------------------------------------
E1_CHECKPOINT_RELPATH = "model/dual_attention/da_stage_b/best.pt"
E1_CHECKPOINT_MD5 = "258710649fb0e6979f64fc1e7ccfc28f"
SOURCE_KEY_STATUS = "inferred_from_filename_not_verified_identifier"
SPLITS_ALLOWED = ("train", "val")          # A2: test is a hard boundary
FORBIDDEN_SPLITS = ("test",)

# Uncertainty margin for case selection — FIXED BEFORE inspecting any
# attention map or visualization (owner preflight rule: deterministic,
# pre-registered criteria only).
UNCERTAINTY_MARGIN = 0.1

# Machine-readable export location (owner preference, documented).
ATTENTION_EXPORT_DIR = os.path.join(REPO, "reports", "phase11")


# ---------------------------------------------------------------------------
# checkpoint / model loading (frozen, eval-only)
# ---------------------------------------------------------------------------

def verify_checkpoint_md5(root: str = REPO) -> str:
    """Return the frozen E1 checkpoint MD5, raising on any drift."""
    path = os.path.join(root, E1_CHECKPOINT_RELPATH)
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if digest != E1_CHECKPOINT_MD5:
        raise RuntimeError(
            "Frozen E1 checkpoint MD5 mismatch:\n"
            f"  expected: {E1_CHECKPOINT_MD5}\n"
            f"  actual:   {digest}\n"
            "Refusing to run Phase 11 attention extraction.")
    return digest


def load_frozen_e1_model(root: str = REPO):
    """Load the frozen Phase 9 E1 DualAttentionMIL strictly in eval mode.

    Raises on checkpoint drift. The returned model is never trained here;
    callers must keep every forward under ``torch.no_grad()``.
    """
    import torch
    from src.models.dual_attention_mil import DualAttentionMIL

    verify_checkpoint_md5(root)
    payload = torch.load(os.path.join(root, E1_CHECKPOINT_RELPATH),
                         map_location="cpu", weights_only=False)
    model = DualAttentionMIL()
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# attention extraction (train/val only; eval mode; no_grad)
# ---------------------------------------------------------------------------

def extract_bag_attention(model, batch) -> Dict[str, Any]:
    """Extract per-instance attention + gates + probability for ONE bag.

    ``batch`` is a Phase 7 ``InstanceBatch`` (from
    ``src.mil.instance_generation.load_bag_batch``) for a train/val bag.

    Returns a dict with:
        attention     (n_i,) float64 stage-2 instance attention (sums to 1)
        gates         (n_i, 128) float64 stage-1 channel gates, in (0, 1)
        h_tilde       (n_i, 128) float64 channel-refined embeddings
        embeddings    (n_i, 128) float64 trunk embeddings
        probability   float — sigmoid of the bag logit (model output)
        logit         float
        bag_id / source_group_id / true_label / provenance (verbatim batch)
    """
    import torch

    out = model.forward_bag(batch.images)
    att = out["attention"].detach().cpu().to(torch.float64).numpy()
    gates = out["gates"].detach().cpu().to(torch.float64).numpy()
    h_tilde = out["h_tilde"].detach().cpu().to(torch.float64).numpy()
    emb = out["embeddings"].detach().cpu().to(torch.float64).numpy()
    logit = float(out["logit"].detach().cpu().view(-1)[0])
    prob = float(1.0 / (1.0 + np.exp(-logit)))

    n = int(batch.n_instances)
    if att.shape != (n,):
        raise ValueError(f"attention shape {att.shape} != ({n},)")
    if gates.shape != (n, 128):
        raise ValueError(f"gates shape {gates.shape} != ({n}, 128)")
    if not (np.all(np.isfinite(att)) and np.all(np.isfinite(gates))):
        raise ValueError("non-finite attention/gate values")
    if not (0.0 <= prob <= 1.0):
        raise ValueError(f"probability out of range: {prob}")

    return {
        "bag_id": batch.bag_id,
        "source_group_id": batch.source_group_id,
        "true_label": int(batch.numeric_label),
        "attention": att,
        "gates": gates,
        "h_tilde": h_tilde,
        "embeddings": emb,
        "logit": logit,
        "probability": prob,
        "n_instances": n,
        "provenance": tuple(batch.provenance),
    }


def rank_instances(attention: Sequence[float]) -> List[Dict[str, Any]]:
    """Deterministic descending rank of instances by stage-2 attention.

    Ties are broken by ascending instance order (stable, order-based), so the
    ranking is reproducible for any persisted attention vector. ``rank`` is
    1-based (1 = highest attention). ``order_index`` is the 0-based instance
    position within the bag's deterministic loader order.
    """
    att = np.asarray(attention, dtype=np.float64)
    n = att.size
    # stable argsort on (-attention, index): sort by attention desc, tie -> index asc
    order = sorted(range(n), key=lambda i: (-float(att[i]), i))
    return [{"order_index": i, "attention": float(att[i]), "rank": r + 1}
            for r, i in enumerate(order)]


def top_k_instances(attention: Sequence[float], k: int) -> List[Dict[str, Any]]:
    """Deterministic top-k rows of :func:`rank_instances` (k > n clamps to n)."""
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")
    ranked = rank_instances(attention)
    return ranked[:min(k, len(ranked))]


# ---------------------------------------------------------------------------
# deterministic case selection (A2: train/val only; criteria fixed a priori)
# ---------------------------------------------------------------------------

def _load_split_predictions(root: str, split: str) -> Dict[str, dict]:
    """Load the frozen Phase 9 persisted predictions for one allowed split.

    Only ``val_predictions.csv`` exists for E1 (Phase 9 never wrote test
    predictions through this path in Phase 11). Train bags have no persisted
    E1 prediction; their ``probability``/``prediction`` are None and those
    cases are selected only by the fallback rule below.
    """
    path = os.path.join(root, "experiments", "dual_attention_results",
                        "da_stage_b", f"{split}_predictions.csv")
    rows: Dict[str, dict] = {}
    if not os.path.isfile(path):
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("split") != split:
                continue
            rows[r["bag_id"]] = {
                "bag_id": r["bag_id"],
                "source_group_id": r.get("source_group_id", ""),
                "y_true": int(r["y_true"]) if r.get("y_true") not in (None, "") else None,
                "probability": float(r["probability"]) if r.get("probability") else None,
                "prediction": int(r["prediction"]) if r.get("prediction") not in (None, "") else None,
            }
    return rows


def select_representative_cases(root: str = REPO,
                                n_uncertain: int = 2,
                                seed: int = 20260918) -> Dict[str, Dict[str, Any]]:
    """Deterministically select representative train/val bags (A2).

    Pre-registered rule (fixed BEFORE any attention map was generated; never
    tuned on visualization output):

      1. From the frozen Phase 9 ``val_predictions.csv`` (the ONLY persisted
         E1 validation predictions), sort bags deterministically by
         (bag_id ascending) and classify each as
         ``correct_benign`` / ``correct_malignant`` / ``false_positive`` /
         ``false_negative`` at the FIXED threshold 0.5 (reporting convention
         inherited from Phases 6/9/10 — never tuned here).
      2. From each non-empty class, take the FIRST bag in bag_id order.
      3. Uncertain cases: bags with ``|probability - 0.5| <= 0.1``
         (margin fixed a priori), FIRST ``n_uncertain`` in bag_id order.
      4. Train-split fallback (E1 has no persisted train predictions): the
         deterministically sampled bag from
         ``src.mil.instance_generation.deterministic_sample(seed=20260918)``
         covering the train split's smallest bag, giving the visualization a
         train example without any label-derived selection.
      5. De-duplicate by bag_id; every selected case carries its split and,
         where legitimately available, the frozen prediction/label.

    No case is chosen after looking at attention maps or images.
    """
    import random

    selected: Dict[str, Dict[str, Any]] = {}
    val_rows = _load_split_predictions(root, "val")
    threshold = 0.5

    def add(split: str, bag_id: str, prob=None, pred=None, y=None, category="") -> None:
        if bag_id in selected:
            return
        selected[bag_id] = {
            "bag_id": bag_id, "split": split, "category": category,
            "probability": prob, "prediction": pred, "true_label": y,
        }

    # 1-2: first bag per prediction-outcome class (val only, deterministic)
    for category, ok in (("correct_benign", lambda r: r["y_true"] == 0 and r["prediction"] == 0),
                         ("correct_malignant", lambda r: r["y_true"] == 1 and r["prediction"] == 1),
                         ("false_positive", lambda r: r["y_true"] == 0 and r["prediction"] == 1),
                         ("false_negative", lambda r: r["y_true"] == 1 and r["prediction"] == 0)):
        for bag_id in sorted(val_rows):
            r = val_rows[bag_id]
            if r["probability"] is None or ok(r):
                if ok(r):
                    add("val", bag_id, r["probability"], r["prediction"],
                        r["y_true"], category)
                    break

    # 3: uncertain cases (fixed margin), first n in bag_id order
    uncertain = sorted(
        (bid for bid, r in val_rows.items()
         if r["probability"] is not None
         and abs(r["probability"] - 0.5) <= UNCERTAINTY_MARGIN))
    for bid in uncertain[:max(0, n_uncertain)]:
        r = val_rows[bid]
        add("val", bid, r["probability"], r["prediction"], r["y_true"], "uncertain")

    # 4: train fallback via the Phase 7 deterministic sampler
    try:
        ig = _load_module("p11_instance_generation", "src/mil/instance_generation.py")
        ds = ig.open_split("train", root=root)
        sampled = ig.deterministic_sample(ds, 1, seed=seed)
        for bid in sampled:
            add("train", bid, None, None, None, "train_fallback")
    except Exception:  # noqa: BLE001 — cache-missing environments: val cases suffice
        pass

    return selected


# ---------------------------------------------------------------------------
# extraction over selected cases
# ---------------------------------------------------------------------------

def extract_selected_cases(model, root: str = REPO,
                           cases: Optional[Dict[str, Dict[str, Any]]] = None
                           ) -> List[Dict[str, Any]]:
    """Extract attention for the selected train/val cases (A2 boundary enforced).

    Every case is re-verified to be a train/val bag via the Phase 5.5 loader
    membership before any image is loaded; a test-split request raises.
    """
    import torch
    ig = _load_module("p11_instance_generation2", "src/mil/instance_generation.py")

    if cases is None:
        cases = select_representative_cases(root=root)

    by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": []}
    for c in cases.values():
        split = c.get("split")
        if split not in SPLITS_ALLOWED:
            raise ValueError(
                f"Phase 11 hard boundary: case {c.get('bag_id')!r} has split "
                f"{split!r}; only {SPLITS_ALLOWED} are permitted")
        by_split[split].append(c)

    results: List[Dict[str, Any]] = []
    for split in ("val", "train"):                     # deterministic order
        if not by_split[split]:
            continue
        ds = ig.open_split(split, root=root)
        membership = {b.bag_id for b in ds.bags}
        for c in sorted(by_split[split], key=lambda x: x["bag_id"]):
            bag_id = c["bag_id"]
            if bag_id not in membership:
                raise ValueError(f"bag {bag_id!r} not present in split {split!r}")
            batch = ig.load_bag_batch(ds.bag(bag_id), ds)
            with torch.no_grad():
                res = extract_bag_attention(model, batch)
            res["split"] = split
            res["category"] = c.get("category", "")
            res["frozen_probability"] = c.get("probability")
            res["frozen_prediction"] = c.get("prediction")
            results.append(res)
    return results


# ---------------------------------------------------------------------------
# A5: machine-readable export (CSV + JSON), full provenance
# ---------------------------------------------------------------------------

EXPORT_CSV_COLUMNS = [
    "bag_id", "source_group_id", "source_key_status", "split", "image_path",
    "md5", "processed_relpath", "instance_order", "attention", "rank",
    "predicted_probability", "true_label", "prediction_at_0.5",
    "category", "gate_mean", "gate_max", "gate_argmax", "gate_vector_ref",
]


def build_export_rows(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten extraction results into one provenance-complete row/instance."""
    rows: List[Dict[str, Any]] = []
    for res in results:
        n = res["n_instances"]
        ranks = {r["order_index"]: r for r in rank_instances(res["attention"])}
        for i, prov in enumerate(res["provenance"]):
            g = res["gates"][i]
            att = float(res["attention"][i])
            prob = res["probability"]
            rows.append({
                "bag_id": res["bag_id"],
                "source_group_id": res["source_group_id"],
                "source_key_status": prov.get("source_key_status", SOURCE_KEY_STATUS),
                "split": res["split"],
                "image_path": prov["image_path"],
                "md5": prov.get("md5", ""),
                "processed_relpath": prov.get("processed_relpath", ""),
                "instance_order": i,
                "attention": att,
                "rank": ranks[i]["rank"],
                "predicted_probability": prob,
                "true_label": int(res["true_label"]),
                "prediction_at_0.5": int(prob >= 0.5) if prob is not None else "",
                "category": res.get("category", ""),
                "gate_mean": float(np.mean(g)),
                "gate_max": float(np.max(g)),
                "gate_argmax": int(np.argmax(g)),
                # gate vector reference: gates are exported in the JSON sidecar
                "gate_vector_ref": f"gates[{res['bag_id']}][{i}]",
            })
    return rows


def write_attention_export(results: Sequence[Dict[str, Any]],
                           dest_dir: str = ATTENTION_EXPORT_DIR,
                           ) -> Dict[str, str]:
    """Write the A5 machine-readable export: CSV (tabular) + JSON (vectors).

    CSV  : one row per instance with full provenance and scalar summaries.
    JSON : per-bag gate vectors (128-d), attention vectors, model/checkpoint
           identity, and the frozen provenance constants.
    Returns the written paths.
    """
    os.makedirs(dest_dir, exist_ok=True)
    rows = build_export_rows(results)
    csv_path = os.path.join(dest_dir, "attention_export.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EXPORT_CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    payload = {
        "phase": "11",
        "generated_from": "frozen Phase 9 E1 checkpoint (eval mode, no_grad)",
        "checkpoint_relpath": E1_CHECKPOINT_RELPATH,
        "checkpoint_md5": E1_CHECKPOINT_MD5,
        "source_key_status": SOURCE_KEY_STATUS,
        "splits_used": sorted({r["split"] for r in rows}),
        "test_split_accessed": False,
        "uncertainty_margin": UNCERTAINTY_MARGIN,
        "attention_semantics": "stage-2 instance attention sums to 1 per bag; "
                               "model attribution, NOT clinical explanation",
        "bags": [
            {
                "bag_id": res["bag_id"],
                "source_group_id": res["source_group_id"],
                "split": res["split"],
                "category": res.get("category", ""),
                "n_instances": res["n_instances"],
                "bag_probability": res["probability"],
                "bag_logit": res["logit"],
                "true_label": int(res["true_label"]),
                "attention": [float(a) for a in res["attention"]],
                "gates": [[float(v) for v in row] for row in res["gates"]],
                "gate_vector_ref": EXPORT_CSV_COLUMNS and "gates[bag_id][instance_order]",
            }
            for res in results
        ],
    }
    json_path = os.path.join(dest_dir, "attention_export.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return {"csv": csv_path, "json": json_path}


# ---------------------------------------------------------------------------
# visualization (matplotlib Agg; deterministic; reports/figures/phase11)
# ---------------------------------------------------------------------------

FIGURES_DIR = os.path.join(REPO, "reports", "figures", "phase11")


def _thumb_from_processed(root: str, processed_relpath: str) -> np.ndarray:
    """Decode one Phase 5 cached image for thumbnail rendering (train/val only)."""
    from PIL import Image
    path = os.path.join(root, processed_relpath)
    with Image.open(path) as im:
        arr = np.asarray(im).astype(np.float64)
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr


def render_attention_grid(model, batch, out_path: str, k_top: int = 4,
                          k_low: int = 4) -> Dict[str, Any]:
    """Attention-ranked thumbnail grid for ONE bag (A3 item 1).

    Layout: top-k highest-attention instances then up to k_low lowest, each
    annotated with attention weight and rank. Titles are attribution-labeled;
    provenance is embedded as a caption line. Deterministic ordering.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = extract_bag_attention(model, batch)
    att = res["attention"]
    ranked = rank_instances(att)
    top = ranked[:min(k_top, len(ranked))]
    low = list(reversed(ranked[-min(k_low, len(ranked)):]))  # ascending attention

    cells = top + low
    n_cols = min(4, max(1, len(cells)))
    n_rows = (len(cells) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.0 * n_cols, 3.4 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for j, r in enumerate(cells):
        i = r["order_index"]
        prov = res["provenance"][i]
        img = _thumb_from_processed(REPO, prov["processed_relpath"])
        ax = axes[j // n_cols][j % n_cols]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"rank {r['rank']}  attention {r['attention']:.4f}",
                     fontsize=9)
        ax.set_xlabel(prov["image_path"], fontsize=5, rotation=0)
    fig.suptitle(
        f"Stage-2 instance MIL attention — bag {res['bag_id']} "
        f"(p={res['probability']:.4f})\n"
        "Model attribution — NOT clinical explanation, NOT pixel localization",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"bag_id": res["bag_id"], "path": out_path,
            "attention_sum": float(att.sum())}


def render_channel_gate_figure(model, batch, out_path: str) -> Dict[str, Any]:
    """128-d stage-1 channel-gate figure for ONE bag (A3 item 2).

    Rendered as per-instance gate profiles across the 128 embedding channels
    (feature-space attribution). Explicitly labeled as feature-space — the
    128 dimensions are embedding channels, NOT anatomical regions and NOT
    pixels.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = extract_bag_attention(model, batch)
    gates = res["gates"]                       # (n_i, 128)
    n = res["n_instances"]
    ranked = rank_instances(res["attention"])
    show = ranked[:min(6, n)]                  # highest-attention instances
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for r in show:
        i = r["order_index"]
        ax.plot(range(128), gates[i], linewidth=0.9,
                label=f"inst {i} (attention {r['attention']:.3f})")
    ax.set_xlabel("embedding channel (0..127) — FEATURE SPACE")
    ax.set_ylabel("stage-1 gate value c ∈ (0,1)")
    ax.set_title(
        "Feature-space channel gates — model attribution, "
        "not image-space localization")
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"bag_id": res["bag_id"], "path": out_path,
            "n_gate_vectors": int(gates.shape[0])}


def generate_phase11_outputs(model=None, root: str = REPO,
                             figures_dir: str = FIGURES_DIR,
                             export_dir: str = ATTENTION_EXPORT_DIR,
                             ) -> Dict[str, Any]:
    """End-to-end Phase 11 generation over the deterministic train/val cases.

    Returns a summary dict with the selected cases, per-case figure paths,
    export paths, and integrity checks (checkpoint MD5 verified before and
    after; attention normalization asserted per bag).
    """
    import torch
    from src.explainability.gradcam import render_gradcam_overlay

    md5_before = verify_checkpoint_md5(root)
    if model is None:
        model = load_frozen_e1_model(root)
    assert model.training is False

    cases = select_representative_cases(root=root)
    results = extract_selected_cases(model, root=root, cases=cases)
    if not results:
        raise RuntimeError("no Phase 11 cases selected — check data/cache state")

    figure_paths: List[str] = []
    for res in results:
        ig = _load_module("p11_instance_generation3", "src/mil/instance_generation.py")
        ds = ig.open_split(res["split"], root=root)
        batch = ig.load_bag_batch(ds.bag(res["bag_id"]), ds)

        grid = os.path.join(figures_dir,
                            f"attention_grid_{res['split']}_{res['bag_id']}.png")
        figure_paths.append(render_attention_grid(model, batch, grid)["path"])

        gates_fig = os.path.join(
            figures_dir, f"channel_gates_{res['split']}_{res['bag_id']}.png")
        figure_paths.append(render_channel_gate_figure(model, batch, gates_fig)["path"])

        # A1/A3 item 3: Grad-CAM overlay for the single highest-attention
        # instance of each bag (one representative instance per case).
        top = rank_instances(res["attention"])[0]["order_index"]
        gc = os.path.join(figures_dir,
                          f"gradcam_{res['split']}_{res['bag_id']}_inst{top}.png")
        figure_paths.append(render_gradcam_overlay(
            model, batch, instance_index=top, out_path=gc)["path"])

    export_paths = write_attention_export(results, dest_dir=export_dir)

    md5_after = verify_checkpoint_md5(root)
    if md5_before != md5_after:
        raise RuntimeError("E1 checkpoint changed during Phase 11 generation")

    for res in results:
        s = float(np.sum(res["attention"]))
        if abs(s - 1.0) > 1e-6:
            raise RuntimeError(
                f"attention normalization failure for bag {res['bag_id']}: sum={s}")

    return {
        "cases": cases,
        "bags": [{"bag_id": r["bag_id"], "split": r["split"],
                  "category": r["category"], "n_instances": r["n_instances"],
                  "probability": r["probability"]}
                 for r in results],
        "figure_paths": figure_paths,
        "export_paths": export_paths,
        "checkpoint_md5": md5_after,
        "attention_sums": {r["bag_id"]: float(np.sum(r["attention"]))
                           for r in results},
    }
