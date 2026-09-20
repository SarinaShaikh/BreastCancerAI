"""Phase 9: Dual Attention MIL training (E1 ``da_stage_b``).

Owner-authorized (2026-09-19): E1 ONLY — E2/``da_stage_a`` is NOT authorized,
no persistent embedding cache (O2: the trunk is jointly trainable, so pixels
flow through the extractor every step), fresh initialization with seed
20260918 (O3), and NO test-split access (O4: test evaluation is separately
authorized later; this module cannot construct the test split at all).

The training loop is a structural mirror of the proven Phase 6 ``train_b4``
(``src/training/train_baseline.py``) with only the model swapped to the
locked Phase 8 ``DualAttentionMIL``: identical unweighted BCEWithLogitsLoss,
Adam(lr=1e-3), one optimizer step per bag, seeded bag permutation, train-only
horizontal flip stream (seed+2, one draw per train instance in dataset
order), validation materialized once (deterministic, unaugmented), strict
bag-ROC-AUC best-checkpoint selection, patience-5 early stopping, and the
bit-exact chunk-resume machinery (model/optimizer/all-RNG persistence) reused
directly from Phase 6. No Phase 6 file is modified.
"""
from __future__ import annotations

import csv
import math
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.mil.instance_generation import InstanceBatch, load_bag_batch, open_split
from src.models.dual_attention_mil import DualAttentionMIL
from src.training import train_baseline as tb

SEED = tb.SEED
DEVICE = tb.DEVICE
_REPO = tb._REPO
THRESHOLD_DEFAULT = 0.5

# Splits this module may ever materialize. The frozen test split is NOT in
# this tuple, and no code path below constructs it (leakage guard; O4).
PHASE9_SPLITS = ("train", "val")


# ------------------------------------------------------------- split access
def materialize_train_val() -> Tuple[List[InstanceBatch], List[InstanceBatch]]:
    """Load the frozen train/val splits through the Phase 7 adapter.

    The Phase 7 adapter rejects ``test`` by construction, so the trainer
    cannot accidentally touch the frozen test split.
    """
    train_ds = open_split("train", root=_REPO)
    val_ds = open_split("val", root=_REPO)
    train_bags = [load_bag_batch(b, train_ds) for b in train_ds.bags]
    val_bags = [load_bag_batch(b, val_ds) for b in val_ds.bags]
    return train_bags, val_bags


# ------------------------------------------------------------ flip policy
def _bag_flip_slices(n_instances_total: int, bag_sizes: Sequence[int],
                     flip_rng: random.Random) -> List[Sequence[bool]]:
    """Phase 6 flip discipline: one draw per train INSTANCE in dataset order;
    each bag consumes its contiguous slice (mirrors train_b4 exactly)."""
    flags = tb._epoch_flip_mask(n_instances_total, flip_rng)
    slices, off = [], 0
    for n in bag_sizes:
        slices.append(flags[off:off + n])
        off += n
    assert off == n_instances_total
    return slices


def _apply_flips(x: torch.Tensor, flags: Sequence[bool]) -> torch.Tensor:
    """Bitwise-identical Phase 5 horizontal flip on float32 rows."""
    if not any(flags):
        return x
    rows = [torch.flip(x[k], dims=(2,)) if f else x[k]
            for k, f in enumerate(flags)]
    return torch.stack(rows)


def _is_improvement(auc: Optional[float], best_auc: float) -> bool:
    """Strict-improvement checkpoint rule (equal AUC never updates)."""
    return auc is not None and auc > best_auc


def _should_stop(best: dict, epoch: int, patience: int) -> bool:
    """Phase 6 early-stopping rule, verbatim in intent."""
    return best["epoch"] >= 0 and (epoch - best["epoch"]) >= patience


# ------------------------------------------------------------- core loop
def _bag_forward_logit(model: DualAttentionMIL, x: torch.Tensor) -> torch.Tensor:
    """(n_i, 1, 224, 224) -> raw bag logit shape (1,)."""
    return model.forward_bag(x)["logit"].view(-1)


def _train_loop(train_bags: List[InstanceBatch],
                val_bags: List[InstanceBatch],
                model: DualAttentionMIL,
                opt: torch.optim.Optimizer,
                seed: int,
                max_epochs: int,
                patience: int,
                checkpoint_path: str,
                cfg_payload: dict,
                progress_cb=None,
                chunk_epochs: Optional[int] = None,
                chunk_state_path: Optional[str] = None,
                log: Optional[List[dict]] = None,
                best: Optional[dict] = None,
                start_epoch: int = 1) -> Tuple[DualAttentionMIL, List[dict], dict]:
    """The train_b4 loop with the Dual Attention model. Bit-exact resumable."""
    lossf = nn.BCEWithLogitsLoss()          # unweighted (locked protocol)

    epoch_rng = random.Random(seed)         # explicit seeded bag shuffle
    flip_rng = random.Random(seed + 2)      # train-only augmentation stream

    log = log if log is not None else []
    best = best if best is not None else {"val_bag_roc_auc": -math.inf,
                                          "epoch": -1}
    if chunk_state_path and os.path.isfile(chunk_state_path):
        log, best, start_epoch = tb._load_chunk_state(
            chunk_state_path, model, opt, epoch_rng, flip_rng)

    limit = max_epochs
    if chunk_epochs is not None:
        limit = min(max_epochs, start_epoch + int(chunk_epochs) - 1)

    # Validation materialized ONCE (deterministic, unaugmented).
    val_x = [b.images for b in val_bags]
    vy = np.array([b.numeric_label for b in val_bags], dtype=np.int64)

    # Dataset-order instance offsets for the shared flip-decision stream.
    offsets, off = [], 0
    for b in train_bags:
        offsets.append(off)
        off += b.n_instances
    n_instances = off

    stop = False
    for epoch in range(start_epoch, limit + 1):
        model.train()
        t0 = time.time()
        order = list(range(len(train_bags)))
        epoch_rng.shuffle(order)            # explicit seeded bag permutation
        flip_flags = tb._epoch_flip_mask(n_instances, flip_rng)

        total_loss, nb = 0.0, 0
        for bi in order:
            bag = train_bags[bi]
            flags = flip_flags[offsets[bi]:offsets[bi] + bag.n_instances]
            x = _apply_flips(bag.images, flags)
            opt.zero_grad()
            logit = _bag_forward_logit(model, x)
            loss = lossf(logit, torch.tensor([float(bag.numeric_label)]))
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            nb += 1

        model.eval()
        with torch.no_grad():
            v_logits = torch.cat([_bag_forward_logit(model, xb)
                                  for xb in val_x])
        v_probs = torch.sigmoid(v_logits).numpy()
        bag_auc = tb._auc(vy, v_probs)

        log.append({
            "epoch": epoch,
            "train_loss": round(total_loss / max(nb, 1), 6),
            "val_bag_roc_auc": None if bag_auc is None else round(bag_auc, 6),
            "epoch_seconds": round(time.time() - t0, 1),
        })
        if progress_cb:
            progress_cb(epoch, log[-1])

        if _is_improvement(bag_auc, best["val_bag_roc_auc"]):
            best["val_bag_roc_auc"] = bag_auc
            best["epoch"] = epoch
            tb.save_checkpoint(checkpoint_path, model, epoch, seed,
                               bag_auc, cfg_payload)

        stop = _should_stop(best, epoch, patience)
        if stop or epoch == limit:
            break

    finished = bool(stop) or limit >= max_epochs
    if not finished and chunk_state_path:
        tb._save_chunk_state(chunk_state_path, model, opt, epoch_rng, flip_rng,
                             log, best, limit + 1)
        info = {"paused": True, "epochs_run": len(log),
                "next_epoch": limit + 1, "total_epochs": max_epochs}
        return model, log, info
    if chunk_state_path and os.path.isfile(chunk_state_path):
        os.remove(chunk_state_path)   # finished: no resume state may linger

    payload = tb.load_checkpoint(checkpoint_path, model)   # restore best
    info = {
        "best_epoch": payload["epoch"],
        "best_val_bag_roc_auc": payload["validation_metric"],
        "epochs_run": len(log),
        "stopped_early": len(log) < max_epochs,
    }
    return model, log, info


# ------------------------------------------------------------ official E1
def train_da_stage_b(cfg: dict, seed: int = SEED,
                     max_epochs_override: Optional[int] = None,
                     progress_cb=None,
                     chunk_epochs: Optional[int] = None,
                     chunk_state_path: Optional[str] = None,
                     bags: Optional[List[InstanceBatch]] = None,
                     val_bags: Optional[List[InstanceBatch]] = None
                     ) -> Tuple[DualAttentionMIL, List[dict], dict]:
    """Train E1 ``da_stage_b`` under the locked protocol.

    ``bags``/``val_bags`` default to the frozen train/val splits loaded via
    the Phase 7 adapter; tests may inject synthetic bags to exercise the loop
    without touching dataset pixels. Test-split access does not exist.
    """
    tr = cfg["training"]
    max_epochs = int(max_epochs_override or tr["max_epochs"])
    patience = int(tr["early_stopping"]["patience"])
    lr = float(tr["learning_rate"])
    checkpoint = os.path.join(_REPO, cfg["experiment"]["checkpoint"])
    if chunk_state_path is None:
        chunk_state_path = os.path.join(_REPO,
                                        cfg["experiment"]["chunk_state"])

    tb.set_seed(seed)          # before any model init (3-stream discipline)

    if bags is None or val_bags is None:
        real_bags, real_val = materialize_train_val()
        bags = bags if bags is not None else real_bags
        val_bags = val_bags if val_bags is not None else real_val

    model = DualAttentionMIL()                     # fresh init (O3)
    model.extractor.freeze_trunk(False)            # E1: trunk jointly trainable
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    cfg_payload = {
        "experiment": dict(cfg["experiment"]),
        "training": dict(tr),
        "model": dict(cfg["model"]),
        "seed": seed,
    }
    return _train_loop(
        bags, val_bags, model, opt, seed, max_epochs, patience,
        checkpoint, cfg_payload, progress_cb=progress_cb,
        chunk_epochs=chunk_epochs, chunk_state_path=chunk_state_path,
    )


# ------------------------------------------------- deterministic predictions
def da_bag_predictions(model: DualAttentionMIL, dataset,
                       bags: Optional[List[InstanceBatch]] = None
                       ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Deterministic bag-level probabilities for a trained Dual Attention
    model over ``dataset`` (unaugmented, loader bag order). The caller is
    responsible for split legality — Phase 9 training/selection only ever
    passes the validation split."""
    if bags is None:
        bags = [load_bag_batch(b, dataset) for b in dataset.bags]
    model.eval()
    probs: List[float] = []
    labels: List[int] = []
    ids: List[str] = []
    with torch.no_grad():
        for b in bags:
            logit = _bag_forward_logit(model, b.images)
            probs.append(float(torch.sigmoid(logit).item()))
            labels.append(b.numeric_label)
            ids.append(b.bag_id)
    return (np.array(probs, dtype=np.float64),
            np.array(labels, dtype=np.int64), ids)


def write_val_predictions(dest_csv: str, probs: np.ndarray,
                          labels: np.ndarray, bag_ids: List[str],
                          val_bags: List[InstanceBatch],
                          threshold: float = THRESHOLD_DEFAULT) -> None:
    """Validation predictions artifact (bag order, provenance attached)."""
    os.makedirs(os.path.dirname(dest_csv), exist_ok=True)
    by_id = {b.bag_id: b for b in val_bags}
    with open(dest_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bag_id", "source_group_id", "y_true", "probability",
                    "prediction", "split"])
        for p, y, bid in zip(probs, labels, bag_ids):
            b = by_id[bid]
            w.writerow([bid, b.source_group_id, int(y), repr(float(p)),
                        int(float(p) >= threshold), "val"])
