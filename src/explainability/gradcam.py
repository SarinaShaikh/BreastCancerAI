"""Phase 11: Grad-CAM on the frozen CNN feature extractor (owner decision A1).

Purpose: a per-pixel SPATIAL cross-check of what the Phase 8 trunk responds
to, complementing — never replacing — the MIL instance attention. Grad-CAM
output is a MODEL ATTRIBUTION VISUALIZATION; it is NOT a clinical explanation
and has NOT been validated as an anatomical/lesion localization method (the
dataset has no ground-truth masks, so IoU/mask-overlap validation is
impossible — stated explicitly in the Phase 11 report).

Method (Selvaraju et al., 2017 — standard Grad-CAM, documented per the
owner instruction "document the target layer used" / "document the method"):

    target layer : model.extractor.features[14] — the FINAL ReLU of the
                   fourth Conv3×3-BN-ReLU-MaxPool block (128 channels,
                   28×28 for 224×224 input; the block-4 MaxPool at index 15
                   follows it). Architecture is NOT modified; the layer is
                   only observed via forward/backward hooks.
    forward      : A^k = target-layer activations (1, 128, 28, 28)
    backward     : dY/dA with Y = the bag logit for ONE instance
    weights      : α_k = GAP(dY/dA) over spatial positions
    raw CAM      : ReLU( Σ_k α_k A^k ) → (28, 28)
    normalization: min-max of the positive CAM to [0, 1] (all-zero CAM maps
                   to all-zero — never fabricated signal)
    upsampling   : bilinear 28×28 → 224×224 (PIL), overlay alpha-blended on
                   the Phase 5 processed image.

Frozen-weights discipline (mirrors the owner instruction "use model.eval()
and torch.no_grad() for attention extraction"; Grad-CAM additionally requires
one backward pass, which is safe under the same discipline):

  * the model is ALWAYS in ``eval()`` mode (BatchNorm running statistics are
    used; no batch statistics are updated);
  * every parameter has ``requires_grad=False`` (see
    ``attention_visualization.load_frozen_e1_model``) so gradients flow ONLY
    to the input image tensor — no weight gradients exist, no optimizer is
    ever constructed, no weight can change;
  * the checkpoint is verified by MD5 before and after every run;
  * execution is deterministic on CPU (eval-mode BN, no dropout in the trunk,
    fixed overlay parameters).

Hard boundary: train/val instances ONLY. Every entry point validates the
instance's split provenance and raises on any test-split input.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

from src.explainability.attention_visualization import (
    REPO,
    SPLITS_ALLOWED,
    verify_checkpoint_md5,
)

# Documented target layer: index into InstanceFeatureExtractor.features
# (16 modules = 4 blocks × [Conv2d, BatchNorm2d, ReLU, MaxPool2d]).
# Index 14 = block-4 ReLU → activations (1, 128, 28, 28); the block-4
# MaxPool2d at index 15 → (1, 128, 14, 14) feeding GAP.
TARGET_LAYER_INDEX = 14
TARGET_LAYER_DESCRIPTION = (
    "model.extractor.features[14] — final ReLU of conv block 4 "
    "(128 channels, 28x28 spatial grid for 224x224 input; the block-4 "
    "MaxPool2d at index 15 then yields 14x14 before GAP); "
    "standard Grad-CAM (Selvaraju et al., 2017); architecture unmodified"
)


def _assert_train_val_batch(batch) -> None:
    """A2 hard boundary: refuse any batch whose provenance is not train/val."""
    for prov in batch.provenance:
        split = prov.get("split")
        if split not in SPLITS_ALLOWED:
            raise ValueError(
                "Phase 11 Grad-CAM hard boundary: instance "
                f"{prov.get('image_path')!r} has split {split!r}; "
                f"only {SPLITS_ALLOWED} are permitted (no test access)")


def compute_gradcam(model, batch, instance_index: int) -> Dict[str, Any]:
    """Compute the raw Grad-CAM heatmap for ONE instance of a train/val bag.

    Returns a dict with:
        cam_28x28        float64 (28, 28) normalized ReLU CAM in [0, 1]
        cam_224x224      float64 (224, 224) bilinear upsample of cam_28x28
        logit            float — bag logit for the instance input
        target_layer     description string
        n_target_channels int (128)

    Deterministic: eval-mode BN, frozen weights, fixed normalization.
    The input tensor is the ONLY tensor with requires_grad=True; parameter
    grads remain None (asserted by the Phase 11 test suite).
    """
    import torch

    _assert_train_val_batch(batch)
    n = int(batch.n_instances)
    if not 0 <= instance_index < n:
        raise IndexError(f"instance_index {instance_index} out of range for "
                         f"{n}-instance bag {batch.bag_id!r}")
    if model.training:
        raise RuntimeError("Grad-CAM requires model.eval() mode")

    target_layer = model.extractor.features[TARGET_LAYER_INDEX]
    captured: Dict[str, torch.Tensor] = {}

    def _fwd(_module, _inp, out):
        captured["A"] = out
        out.retain_grad()

    handle = target_layer.register_forward_hook(_fwd)
    try:
        x = batch.images[instance_index:instance_index + 1].clone()  # (1,1,224,224)
        x.requires_grad_(True)                                       # input-only grads

        # Forward through the exact Phase 8 composition (extractor → dual
        # attention → classifier); architecture untouched, eval mode.
        emb = model.extractor(x)                                     # (1, 128)
        out = model.dual_attention.forward_stacked(emb)              # a, gates, h̃, z
        logit_t = model.classifier(out["z"].unsqueeze(0))            # (1, 1)

        model.zero_grad(set_to_none=True)                            # safety no-op
        logit_t.backward(torch.ones_like(logit_t))                   # dY/dA via graph
        A = captured["A"].detach()                                   # (1, 128, 14, 14)
        dYdA = captured["A"].grad.detach()                           # (1, 128, 14, 14)
    finally:
        handle.remove()

    alpha = dYdA.mean(dim=(2, 3), keepdim=True)                      # GAP → (1,128,1,1)
    raw_cam = torch.relu((alpha * A).sum(dim=1)).squeeze(0)          # (28, 28)
    cam = raw_cam.detach().cpu().to(torch.float64).numpy()
    span = float(cam.max() - cam.min())
    cam = (cam - float(cam.min())) / span if span > 0 else np.zeros_like(cam)

    cam_up = _resize_bilinear(cam, 224, 224)
    return {
        "cam_28x28": cam,
        "cam_224x224": cam_up,
        "logit": float(logit_t.detach().view(-1)[0]),
        "target_layer": TARGET_LAYER_DESCRIPTION,
        "n_target_channels": int(A.shape[1]),
    }


def _resize_bilinear(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    """Deterministic bilinear resize (PIL) of a 2-D float array."""
    from PIL import Image
    lo, hi = float(arr.min()), float(arr.max())
    scaled = ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8) if hi > lo \
        else np.zeros_like(arr, dtype=np.uint8)
    im = Image.fromarray(scaled, mode="L").resize((width, height),
                                                  Image.BILINEAR)
    out = np.asarray(im).astype(np.float64) / 255.0
    return out if hi > lo else np.zeros_like(out)


def render_gradcam_overlay(model, batch, instance_index: int, out_path: str,
                           alpha: float = 0.45) -> Dict[str, Any]:
    """Grad-CAM overlay on the Phase 5 processed image (A3 item 3).

    The overlay is alpha-blended with the fixed 'jet' colormap and clearly
    labeled as a model attribution. Deterministic parameters only.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _assert_train_val_batch(batch)
    prov = batch.provenance[instance_index]
    from src.explainability.attention_visualization import _thumb_from_processed
    base = _thumb_from_processed(REPO, prov["processed_relpath"])

    gc = compute_gradcam(model, batch, instance_index)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    axes[0].imshow(base, cmap="gray")
    axes[0].set_title("instance (Phase 5 cache)", fontsize=9)
    im1 = axes[1].imshow(gc["cam_224x224"], cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title("Grad-CAM (28×28 → 224×224)\nmodel attribution", fontsize=9)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(base, cmap="gray")
    axes[2].imshow(gc["cam_224x224"], cmap="jet", vmin=0.0, vmax=1.0,
                   alpha=alpha)
    axes[2].set_title("overlay\nmodel attribution — NOT clinical explanation",
                      fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"Grad-CAM @ features[{TARGET_LAYER_INDEX}] — bag {batch.bag_id} "
        f"instance {instance_index} ({prov['image_path']})\n"
        "Model attribution visualization — not validated as lesion "
        "localization; not a clinical explanation", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"bag_id": batch.bag_id, "path": out_path,
            "instance_index": instance_index,
            "target_layer": TARGET_LAYER_DESCRIPTION,
            "cam_range": [float(gc["cam_28x28"].min()),
                          float(gc["cam_28x28"].max())]}
