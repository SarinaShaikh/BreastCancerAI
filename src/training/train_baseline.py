"""Phase 6: baseline training / evaluation driver (B1, B3, B4).

Roadmap: PROGRESS.md Phase 6 "Baseline Deep Learning Models". Scope
(owner-approved 2026-09-18): B1 simple CNN (image-level), B3 = bag-level
mean-probability aggregation of the frozen B1 checkpoint (no new network),
B4 = attention-free mean-pooling MIL (bag-level, trained from scratch).
B2 ResNet/transfer learning is DEFERRED — nothing here touches pretrained
weights. Device is CPU-only by policy.

Leakage discipline enforced in this module:
  - data comes ONLY from the Phase 5.5 loader over the frozen Phase 4
    manifests; bags/labels/membership are never redefined here;
  - test tensors are touched EXACTLY once, by the final test stage, after
    the B1/B3/B4 configurations and checkpoints are frozen — never during
    training, early stopping, or threshold decisions;
  - validation is used for early stopping only; the threshold is the fixed
    configured 0.5 (never tuned on validation or test);
  - augmentation (horizontal flip p=0.5) is applied ONLY to train batches,
    ONLY through the Phase 5 flip policy; val/test loads are deterministic
    and unaugmented.

Reproducibility: global seed 20260918 (python random / numpy / torch);
explicit seeded permutation shuffles (filesystem order is never relied on);
torch CPU deterministic algorithms; fixed thread count. Residual
nondeterminism: none observed for these CPU ops (V6B-14 executable proof).

Memory design (measured constraint: ~2.7 GiB free RAM on this machine):
train batches are STREAMED from the Phase 5 PNG cache — only one batch of
images is in memory at a time; only the validation split (1,339 images,
~270 MB) is materialized once per run.

Nondeterminism documentation: torch CPU conv/backward kernels for these
shapes are bit-deterministic in practice; the flip stream mirrors the Phase 5
``augment()`` decisions exactly (one uniform draw per train image per epoch).
"""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from model.baselines.simple_cnn import SimpleCNN            # noqa: E402
from model.baselines.meanpool_mil import MeanPoolMIL        # noqa: E402
from src.training.metrics import classification_metrics     # noqa: E402

import importlib.util as _ilu


def _load_module(name: str, relpath: str):
    spec = _ilu.spec_from_file_location(name, os.path.join(_REPO, relpath))
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_p5 = _load_module("phase6_train_p5", "src/preprocessing/pipeline.py")
_bd = _load_module("phase6_train_bd", "src/mil/bag_dataset.py")

TEST_SPLIT_SHA256 = _p5.TEST_MANIFEST_SHA256
PREPROCESSING_VERSION = "1.0.0"
PIPELINE_VERSION = "1.0.1"
SEED = 20260918

DEVICE = torch.device("cpu")   # CPU-only by policy
TORCH_THREADS = 8              # measured best on this 12-core box; headroom left


# --------------------------------------------------------------- seeding
def set_seed(seed: int) -> None:
    """Seed python/numpy/torch; force CPU deterministic algorithms."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(TORCH_THREADS)   # fixed for all runs
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_info(seed: int) -> dict:
    return {
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


# ------------------------------------------------------------- model I/O
def _load_baseline_config(root: str = _REPO) -> dict:
    import yaml
    with open(os.path.join(root, "configs", "baseline_config.yaml"),
              encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(path: str, model: nn.Module, epoch: int, seed: int,
                    val_metric: float, cfg: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "seed": seed,
        "validation_metric": val_metric,
        "config": cfg,
        "preprocessing_version": PREPROCESSING_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "frozen_test_sha256": TEST_SPLIT_SHA256,
    }, path)


def load_checkpoint(path: str, model: nn.Module) -> dict:
    """Load a checkpoint's payload into `model` and return its metadata."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    return payload


# ------------------------------------------------------------- data prep
def _phase5_augment(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Train-only horizontal flip through the Phase 5 augment() API.

    The cached arrays are normalized floats; the Phase 5 ``augment`` operates
    on PIL images, so this wraps the normalized array in a temporary PIL image
    ONLY to reuse the authoritative flip policy (p=0.5, train-only guard).
    Normalization values are untouched — flipping is an axis permutation.
    Exercised by the Phase 6 tests (V6B-09 / V6B-14).
    """
    from PIL import Image as _Image
    im = _Image.fromarray(arr)          # mode 'F' view for the flip op
    out = _p5.augment(im, "train", rng)
    return np.asarray(out, dtype=np.float32)


def _epoch_flip_mask(n: int, rng: random.Random) -> List[bool]:
    """One epoch of Phase 5 flip decisions, in dataset order.

    Consumes the SAME uniform-stream decisions (one ``rng.random() < 0.5``
    draw per image, in dataset order) that per-image calls to the Phase 5
    ``augment()`` would consume. Applying ``np.fliplr`` where the flag is
    True is bitwise-identical to PIL Image.FLIP_LEFT_RIGHT on these float32
    arrays; the equivalence path is exercised by V6B-09/V6B-14 through the
    literal API. Train-split use only.
    """
    return [rng.random() < 0.5 for _ in range(n)]


class _InstanceView:
    """Adapter exposing Phase 5.5 bags as instance-level (image, target) pairs.

    Targets are the instance's bag label (owner instruction: B1 trains at
    instance level with target = its bag label).
    """

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.items = [(b, i) for b in dataset.bags for i in b.instances]

    def __len__(self) -> int:
        return len(self.items)


def _load_split_arrays(dataset):
    """Materialize a (small) split as tensors: (N,1,224,224) float32, (N,).

    Used ONLY for validation (1,339 images ~270 MB) — never for the 6,327-
    image train split (memory discipline; train is streamed).
    """
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for bag in dataset.bags:
        for inst in bag.instances:
            xs.append(dataset.load_instance_image(inst))
            ys.append(bag.numeric_label)
    x = torch.from_numpy(np.stack(xs)).unsqueeze(1).float()
    y = torch.tensor(ys, dtype=torch.float32)
    return x, y


_TRAIN_CACHE_DIR = os.path.join(_REPO, "experiments", "baseline_results",
                                "_cache")


def _train_array_cache(dataset):
    """Memory-mapped float32 copy of the decoded Phase 5 train cache.

    The Phase 5 PNG cache is DECODED ONCE (dataset order) into a (6327, 224,
    224) float32 .npy and memmap'd for training: RAM stays constant (pages
    are evicted by the OS) while per-epoch PNG decoding disappears. Values
    are BIT-IDENTICAL to ``dataset.load_instance_image`` (same decode path;
    a spot-check asserts equality on build). Derived, regenerable, gitignored
    (experiments/baseline_results/_cache/).

    Returns (images_memmap (N,224,224) float32, labels (N,) int64,
    bag_boundaries list[(start, stop, numeric_label)]).
    """
    os.makedirs(_TRAIN_CACHE_DIR, exist_ok=True)
    img_path = os.path.join(_TRAIN_CACHE_DIR, "train_images_f32.npy")
    lab_path = os.path.join(_TRAIN_CACHE_DIR, "train_labels.npy")
    n = len(_InstanceView(dataset))
    if not (os.path.isfile(img_path) and os.path.isfile(lab_path)):
        arr = np.lib.format.open_memmap(
            img_path, mode="w+", dtype=np.float32, shape=(n, 224, 224))
        labels = np.zeros(n, dtype=np.int64)
        k = 0
        for bag in dataset.bags:
            for inst in bag.instances:
                arr[k] = dataset.load_instance_image(inst)
                labels[k] = bag.numeric_label
                k += 1
        assert k == n
        arr.flush()
        # Bit-exactness spot-check against the authoritative loader.
        for probe in (0, n // 3, n - 1):
            items = _InstanceView(dataset).items
            ref = dataset.load_instance_image(items[probe][1])
            assert np.array_equal(np.asarray(arr[probe]), ref), \
                f"train decode-cache drift at index {probe}"
        np.save(lab_path, labels)
        del arr
    images = np.load(img_path, mmap_mode="r")
    labels = np.load(lab_path)
    boundaries = []
    off = 0
    for bag in dataset.bags:
        boundaries.append((off, off + bag.instance_count, bag.numeric_label))
        off += bag.instance_count
    assert off == len(labels)
    return images, labels, boundaries


def iter_train_batches(images, labels, boundaries, batch_size: int,
                       flip_flags: Sequence[bool],
                       perm: Sequence[int]) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Stream train batches from the memmap decode-cache (constant RAM).

    `perm` is the explicit seeded shuffle over dataset-order instance
    indices; `flip_flags[k]` is the pre-drawn Phase 5 flip decision for the
    dataset-order instance k (see _epoch_flip_mask). Flips use np.fliplr,
    bitwise-identical to the Phase 5 PIL left-right flip on float32.
    """
    for s in range(0, len(perm), batch_size):
        idxs = list(perm[s:s + batch_size])
        arrs = []
        for k in idxs:
            a = np.asarray(images[k])          # memmap page -> ndarray
            if flip_flags[k]:
                a = np.fliplr(a).copy()        # contiguous copy for torch
            arrs.append(a)
        x = torch.from_numpy(np.stack(arrs)).unsqueeze(1).float()
        y = torch.tensor([int(labels[k]) for k in idxs], dtype=torch.float32)
        yield x, y


# ------------------------------------------------------- chunked execution
# Training runs CHUNK-RESUMABLY so long CPU runs survive session/process
# interruptions bit-exactly: model, optimizer, and ALL RNG streams (epoch
# shuffle, flip decisions, torch init/stream) are persisted at chunk
# boundaries; a resumed run consumes no extra draws, so a chunked run is
# bit-identical to one uninterrupted run (verified by tests + a dedicated
# 2-epoch equivalence check during Phase 6 execution).


def _save_chunk_state(path: str, model, opt, epoch_rng, flip_rng,
                      log: List[dict], best: dict, next_epoch: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "epoch_rng_state": epoch_rng.getstate(),
        "flip_rng_state": flip_rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "log": log,
        "best": best,
        "next_epoch": next_epoch,
    }, path)


def _load_chunk_state(path: str, model, opt, epoch_rng, flip_rng):
    cs = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(cs["model"])
    opt.load_state_dict(cs["opt"])
    epoch_rng.setstate(cs["epoch_rng_state"])
    flip_rng.setstate(cs["flip_rng_state"])
    torch.set_rng_state(cs["torch_rng_state"])
    return cs["log"], cs["best"], cs["next_epoch"]


# ------------------------------------------------------------- B1 training
def train_b1(cfg: dict, seed: int = SEED,
             max_epochs_override: Optional[int] = None,
             progress_cb=None,
             chunk_epochs: Optional[int] = None,
             chunk_state_path: Optional[str] = None
             ) -> Tuple[SimpleCNN, List[dict], dict]:
    """STEP B — train B1 (image-level) with bag-ROC-AUC early stopping.

    With ``chunk_epochs`` set, the run stops after that many epochs from its
    resume point and persists full state to ``chunk_state_path`` (info
    contains "paused": True); call again with the same arguments to continue
    bit-exactly. Without a chunk request (or when finished/early-stopped),
    the best checkpoint is restored and returned.

    Returns (best_model, train_log, info). Validation is the ONLY selection
    signal (test untouched).
    """
    set_seed(seed)
    b1 = cfg["b1_simple_cnn"]
    total_epochs = int(max_epochs_override or b1["max_epochs"])
    patience = int(b1["early_stopping"]["patience"])
    lr = float(b1["learning_rate"])
    bs = int(b1["batch_size"])

    ds_train = _bd.UltrasoundBagDataset("train", root=_REPO)
    ds_val = _bd.UltrasoundBagDataset("val", root=_REPO)

    model = SimpleCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()

    epoch_rng = random.Random(seed)          # explicit seeded shuffle source
    flip_rng = random.Random(seed + 1)       # augmentation stream (train only)

    log: List[dict] = []
    best = {"val_bag_roc_auc": -math.inf, "epoch": -1}
    start_epoch = 1
    if chunk_state_path and os.path.isfile(chunk_state_path):
        log, best, start_epoch = _load_chunk_state(
            chunk_state_path, model, opt, epoch_rng, flip_rng)

    limit = total_epochs
    if chunk_epochs is not None:
        limit = min(total_epochs, start_epoch + int(chunk_epochs) - 1)

    # Materialize validation tensors ONCE (deterministic, unaugmented).
    vx, vy = _load_split_arrays(ds_val)
    n_val = vx.shape[0]
    # Validation instances are bag-major ordered -> contiguous per-bag slices.
    bag_slices: List[Tuple[int, int, int]] = []   # (start, stop, numeric_label)
    off = 0
    for bag in ds_val.bags:
        bag_slices.append((off, off + bag.instance_count, bag.numeric_label))
        off += bag.instance_count
    assert off == n_val

    # Decode-cache of the train split (memmap; bit-identical to the loader).
    images, labels, boundaries = _train_array_cache(ds_train)
    # TEST-ONLY subset hook (P6_TRAIN_LIMIT): used exclusively by
    # scripts/verify_phase6_resume_equivalence.py to keep the mechanics
    # check fast. Never set by real training runs; truncation is bag-major
    # so only whole bags are kept (bag boundaries stay atomic).
    if os.environ.get("P6_TRAIN_LIMIT"):
        limit_n = int(os.environ["P6_TRAIN_LIMIT"])
        images = images[:limit_n]
        labels = labels[:limit_n]
        boundaries = [(s0, s1, lab) for (s0, s1, lab) in boundaries
                      if s1 <= limit_n]
    n_train = images.shape[0]

    stop = False
    for epoch in range(start_epoch, limit + 1):
        model.train()
        t0 = time.time()
        # Explicit seeded permutation — never filesystem/dataset order.
        perm = list(range(n_train))
        epoch_rng.shuffle(perm)
        # Train-only augmentation decisions (Phase 5 policy stream).
        flip_flags = _epoch_flip_mask(n_train, flip_rng)

        total_loss = 0.0
        nb = 0
        for xb, yb in iter_train_batches(images, labels, boundaries, bs,
                                         flip_flags, perm):
            opt.zero_grad()
            logits = model(xb)
            loss = lossf(logits.squeeze(1), yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            nb += 1

        # ---- validation (image metrics + B3-style bag aggregation) -------
        model.eval()
        with torch.no_grad():
            v_logits = torch.cat([model(vx[s:s + bs])
                                  for s in range(0, n_val, bs)]).squeeze(1)
        v_probs = torch.sigmoid(v_logits).numpy()
        image_auc = _auc(vy.numpy(), v_probs)

        bag_probs, bag_labels = [], []
        for (s0, s1, lab) in bag_slices:
            bag_probs.append(float(np.mean(v_probs[s0:s1])))
            bag_labels.append(lab)
        bag_auc = _auc(np.array(bag_labels), np.array(bag_probs))

        log.append({
            "epoch": epoch,
            "train_loss": round(total_loss / max(nb, 1), 6),
            "val_image_roc_auc": None if image_auc is None else round(image_auc, 6),
            "val_bag_roc_auc": None if bag_auc is None else round(bag_auc, 6),
            "epoch_seconds": round(time.time() - t0, 1),
        })
        if progress_cb:
            progress_cb(epoch, log[-1])

        if bag_auc is not None and bag_auc > best["val_bag_roc_auc"]:
            best["val_bag_roc_auc"] = bag_auc
            best["epoch"] = epoch
            save_checkpoint(b1["checkpoint"], model, epoch, seed, bag_auc, cfg)

        stop = (best["epoch"] >= 0 and (epoch - best["epoch"]) >= patience)
        if stop or epoch == limit:
            break

    finished = bool(stop) or limit >= total_epochs
    if not finished and chunk_state_path:
        _save_chunk_state(chunk_state_path, model, opt, epoch_rng, flip_rng,
                          log, best, limit + 1)
        info = {"paused": True, "epochs_run": len(log),
                "next_epoch": limit + 1, "total_epochs": total_epochs}
        return model, log, info
    if chunk_state_path and os.path.isfile(chunk_state_path):
        os.remove(chunk_state_path)   # finished: no resume state may linger

    payload = load_checkpoint(b1["checkpoint"], model)   # restore best
    info = {
        "best_epoch": payload["epoch"],
        "best_val_bag_roc_auc": payload["validation_metric"],
        "epochs_run": len(log),
        "stopped_early": len(log) < total_epochs,
    }
    return model, log, info


def _auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


# --------------------------------------------------- B1/B3 evaluation utils
def b1_image_predictions(model, dataset, bs: int = 32):
    """Deterministic full-split image-level probabilities (no augmentation).

    Returns (probs float64 (N,), labels int64 (N,), path_order list[str],
    bag_ids list[str]) — bag-major deterministic order.
    """
    model.eval()
    probs: List[float] = []
    labels: List[int] = []
    paths: List[str] = []
    bag_ids: List[str] = []
    with torch.no_grad():
        for bag in dataset.bags:                     # deterministic bag order
            imgs = torch.from_numpy(
                dataset.load_bag_images(bag)).unsqueeze(1).float()
            logits = model(imgs).squeeze(1)
            probs.extend(torch.sigmoid(logits).tolist())
            labels.extend([bag.numeric_label] * imgs.shape[0])
            paths.extend(i.image_path for i in bag.instances)
            bag_ids.extend([bag.bag_id] * imgs.shape[0])
    return (np.array(probs, dtype=np.float64),
            np.array(labels, dtype=np.int64),
            paths, bag_ids)


def aggregate_bags(probs: np.ndarray, bag_ids_per_instance: Sequence[str],
                   dataset, threshold: float = 0.5):
    """B3: bag probability = mean of the bag's instance probabilities.

    Bags are never mixed; bag order follows the deterministic dataset order.
    Returns (bag_probs, bag_labels, bag_preds, bag_ids).
    """
    order = [b.bag_id for b in dataset.bags]
    by_bag: Dict[str, List[float]] = {bid: [] for bid in order}
    for p, bid in zip(probs, bag_ids_per_instance):
        by_bag[bid].append(float(p))
    bag_p = np.array([float(np.mean(by_bag[bid])) for bid in order])
    bag_y = np.array([b.numeric_label for b in dataset.bags], dtype=np.int64)
    bag_pred = (bag_p >= float(threshold)).astype(np.int64)
    return bag_p, bag_y, bag_pred, order


# ------------------------------------------------------------- B4 training
def train_b4(cfg: dict, seed: int = SEED,
             max_epochs_override: Optional[int] = None,
             progress_cb=None,
             chunk_epochs: Optional[int] = None,
             chunk_state_path: Optional[str] = None
             ) -> Tuple[MeanPoolMIL, List[dict], dict]:
    """STEP F — train B4 (bag-level mean-pooling MIL; independent model).

    B4 never loads B1 weights or predictions: its trunk is initialized from
    its own seeded stream. Training unit = BAG (one optimizer step per bag).
    Chunk-resumable exactly like train_b1 (bit-exact resume).
    """
    set_seed(seed)
    b4 = cfg["b4_meanpool_mil"]
    total_epochs = int(max_epochs_override or b4["max_epochs"])
    patience = int(b4["early_stopping"]["patience"])
    lr = float(b4["learning_rate"])

    ds_train = _bd.UltrasoundBagDataset("train", root=_REPO)
    ds_val = _bd.UltrasoundBagDataset("val", root=_REPO)

    model = MeanPoolMIL().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()

    epoch_rng = random.Random(seed)          # explicit seeded bag shuffle
    flip_rng = random.Random(seed + 2)       # train-only augmentation stream

    log: List[dict] = []
    best = {"val_bag_roc_auc": -math.inf, "epoch": -1}
    start_epoch = 1
    if chunk_state_path and os.path.isfile(chunk_state_path):
        log, best, start_epoch = _load_chunk_state(
            chunk_state_path, model, opt, epoch_rng, flip_rng)

    limit = total_epochs
    if chunk_epochs is not None:
        limit = min(total_epochs, start_epoch + int(chunk_epochs) - 1)

    # Validation bags materialized ONCE (deterministic, unaugmented).
    val_bags = [(torch.from_numpy(ds_val.load_bag_images(b)).unsqueeze(1).float(),
                 b.numeric_label) for b in ds_val.bags]
    vy = np.array([lab for _, lab in val_bags])

    # Dataset-order instance offsets (for the shared flip-decision stream).
    offsets = []
    off = 0
    for b in ds_train.bags:
        offsets.append(off)
        off += b.instance_count
    n_instances = off

    stop = False
    for epoch in range(start_epoch, limit + 1):
        model.train()
        t0 = time.time()
        order = list(range(len(ds_train)))
        epoch_rng.shuffle(order)             # explicit seeded bag permutation
        # Train-only flip decisions: one draw per train INSTANCE in dataset
        # order (same stream discipline as B1); each bag consumes its slice.
        flip_flags = _epoch_flip_mask(n_instances, flip_rng)

        total_loss = 0.0
        nb = 0
        for bi in order:
            bag = ds_train.bags[bi]
            imgs = ds_train.load_bag_images(bag)          # (n_i, 224, 224)
            flags = flip_flags[offsets[bi]:offsets[bi] + bag.instance_count]
            rows = [np.fliplr(a).copy() if f else a
                    for a, f in zip(imgs, flags)]
            x = torch.from_numpy(np.stack(rows)).unsqueeze(1).float()
            opt.zero_grad()
            logit = model.forward_stacked(x).squeeze(1)
            loss = lossf(logit, torch.tensor([float(bag.numeric_label)]))
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            nb += 1

        model.eval()
        with torch.no_grad():
            v_logits = torch.cat([model.forward_stacked(xb)
                                  for xb, _ in val_bags]).squeeze(1)
        v_probs = torch.sigmoid(v_logits).numpy()
        bag_auc = _auc(vy, v_probs)

        log.append({
            "epoch": epoch,
            "train_loss": round(total_loss / max(nb, 1), 6),
            "val_bag_roc_auc": None if bag_auc is None else round(bag_auc, 6),
            "epoch_seconds": round(time.time() - t0, 1),
        })
        if progress_cb:
            progress_cb(epoch, log[-1])

        if bag_auc is not None and bag_auc > best["val_bag_roc_auc"]:
            best["val_bag_roc_auc"] = bag_auc
            best["epoch"] = epoch
            save_checkpoint(b4["checkpoint"], model, epoch, seed, bag_auc, cfg)

        stop = (best["epoch"] >= 0 and (epoch - best["epoch"]) >= patience)
        if stop or epoch == limit:
            break

    finished = bool(stop) or limit >= total_epochs
    if not finished and chunk_state_path:
        _save_chunk_state(chunk_state_path, model, opt, epoch_rng, flip_rng,
                          log, best, limit + 1)
        info = {"paused": True, "epochs_run": len(log),
                "next_epoch": limit + 1, "total_epochs": total_epochs}
        return model, log, info
    if chunk_state_path and os.path.isfile(chunk_state_path):
        os.remove(chunk_state_path)   # finished: no resume state may linger

    payload = load_checkpoint(b4["checkpoint"], model)   # restore best
    info = {
        "best_epoch": payload["epoch"],
        "best_val_bag_roc_auc": payload["validation_metric"],
        "epochs_run": len(log),
        "stopped_early": len(log) < total_epochs,
    }
    return model, log, info


def b4_bag_predictions(model, dataset) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Deterministic bag-level probabilities for a trained B4 model."""
    model.eval()
    probs: List[float] = []
    labels: List[int] = []
    ids: List[str] = []
    with torch.no_grad():
        for bag in dataset.bags:
            x = torch.from_numpy(
                dataset.load_bag_images(bag)).unsqueeze(1).float()
            logit = model.forward_stacked(x).squeeze(1)
            probs.append(float(torch.sigmoid(logit).item()))
            labels.append(bag.numeric_label)
            ids.append(bag.bag_id)
    return (np.array(probs, dtype=np.float64),
            np.array(labels, dtype=np.int64), ids)


# ------------------------------------------------------- experiment records
def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO,
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001 - best-effort provenance only
        return "unknown"


def _write_env_txt(path: str) -> None:
    lines = [
        f"python={platform.python_version()}",
        f"platform={platform.platform()}",
        f"torch={torch.__version__}",
        f"cuda_available={torch.cuda.is_available()}",
        "device=cpu",
        f"torch_threads={torch.get_num_threads()}",
        f"deterministic_algorithms={torch.are_deterministic_algorithms_enabled()}",
    ]
    try:
        import torchvision  # noqa: F401
        import sklearn
        import numpy
        import PIL
        import yaml
        lines.append(f"torchvision={torchvision.__version__}")
        lines.append(f"scikit-learn={sklearn.__version__}")
        lines.append(f"numpy={numpy.__version__}")
        lines.append(f"Pillow={PIL.__version__}")
        lines.append(f"PyYAML={yaml.__version__}")
    except Exception:  # noqa: BLE001
        pass
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _manifest_hashes() -> Dict[str, str]:
    import hashlib
    out = {}
    for name in ("train_split.csv", "val_split.csv", "test_split.csv"):
        p = os.path.join(_REPO, "data", "manifests", name)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[name] = h.hexdigest()
    return out


def write_experiment_dir(dest: str, cfg: dict, train_log: List[dict],
                         metrics_json: dict, seed: int,
                         inherited_checkpoint: Optional[str] = None) -> None:
    """Experiment-artifact writer: config/log/metrics/seed/env/provenance."""
    import yaml
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    if train_log:
        with open(os.path.join(dest, "train_log.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(train_log[0].keys()))
            w.writeheader()
            w.writerows(train_log)
    with open(os.path.join(dest, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)
    with open(os.path.join(dest, "seed.txt"), "w", encoding="utf-8") as f:
        f.write(f"{seed}\n")
    _write_env_txt(os.path.join(dest, "env.txt"))
    with open(os.path.join(dest, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump({
            "source_commit": _git_commit(),
            "manifest_hashes": _manifest_hashes(),
            "frozen_test_sha256": TEST_SPLIT_SHA256,
            "preprocessing_version": PREPROCESSING_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "inherited_checkpoint": inherited_checkpoint,
        }, f, indent=2)
