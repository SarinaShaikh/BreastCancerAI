"""Phase 5: ultrasound image preprocessing pipeline (deterministic).

Roadmap: PROGRESS.md Phase 5 — Ultrasound Image Preprocessing &
Augmentation Pipeline. Configuration: configs/preprocessing_config.yaml
(v1.0.0). Decision record: reports/phase5_preprocessing_justification.md.

Pipeline (finalized, evidence-grounded):

    raw image (PNG/JPEG, RGB containers, verified on disk)
      -> PIL open (read-only; failures are explicit, never silent)
      -> convert('L')                       canonical single-channel grayscale
      -> resize (224, 224) BILINEAR         sources are square; no distortion
      -> x = pixel / 255
      -> normalized = (x - 0.316515) / 0.245010
           (statistics derived from the TRAIN split ONLY; val/test are
            transformed with the fixed constants and contribute nothing)
      -> cached as 16-bit grayscale PNG:
             uint16 = round((normalized + 8) * 4096)   (lossless round-trip
             at precision 1/4096 normalized units; exact integer inverse)

Split policies (enforced in code, verified by tests):
    train: deterministic preprocessing; OPTIONAL training-time-only
           augmentation is provided as an API (horizontal flip, seeded)
           and is NEVER applied during cache generation.
    val:   deterministic preprocessing only.
    test:  deterministic preprocessing only; the test cache is a
           transformed representation of the frozen Phase 4 test set.

Scope guard: this module performs preprocessing ONLY — no CNNs, no
training, no evaluation, no checkpoints (Phase 6+). Raw data under
dataset/raw/ is never modified.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
import yaml
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

TEST_MANIFEST_SHA256 = "959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74"
SPLITS = ("train", "val", "test")
CONFIG_RELPATH = os.path.join("configs", "preprocessing_config.yaml")

PROC_COLUMNS = ["output_file", "image_path", "md5", "bag_id",
                "source_group_id", "bag_label", "split",
                "source_key_status", "original_supplied_split"]


class PreprocessingError(RuntimeError):
    """Raised on any explicit preprocessing failure (never silent)."""


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path: str = None) -> dict:
    root = repo_root()
    path = config_path or os.path.join(root, CONFIG_RELPATH)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    expected = {"version": "1.0.0", "pipeline_version": "1.0.1"}
    for key, want in expected.items():
        got = cfg["config"].get(key)
        if got != want:
            raise PreprocessingError(f"config {key} mismatch: {got!r}")
    return cfg


def raw_image_root(root: str = None) -> str:
    """Deepest common image directory under the immutable raw dataset."""
    if root is None:
        root = repo_root()
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "phase1_audit", os.path.join(root, "src", "data", "audit.py"))
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    common, _ = audit.resolve_image_root(os.path.join(root, "dataset", "raw"))
    return common


def verify_freeze(root: str = None) -> str:
    """Independently verify the Phase 4 frozen test manifest before any work."""
    if root is None:
        root = repo_root()
    p = os.path.join(root, "data", "manifests", "test_split.csv")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if digest != TEST_MANIFEST_SHA256:
        raise PreprocessingError(
            f"FROZEN test manifest hash mismatch: {digest} — STOP, do not process")
    return digest


def verify_split_counts(root: str = None) -> Dict[str, int]:
    if root is None:
        root = repo_root()
    expected = {"train": 6327, "val": 1339, "test": 1350}
    counts = {}
    for s in SPLITS:
        with open(os.path.join(root, "data", "manifests", f"{s}_split.csv"),
                  newline="", encoding="utf-8") as f:
            counts[s] = sum(1 for _ in csv.DictReader(f))
    if counts != expected:
        raise PreprocessingError(f"split counts drifted from Phase 4: {counts}")
    return counts


def verify_train_stats(root: str = None, cfg: dict = None) -> Tuple[float, float]:
    """Recompute train-split intensity statistics and check config constants.

    Guarantees the normalization statistics truly derive from TRAIN only
    and match the committed configuration (no val/test contribution).
    """
    if root is None:
        root = repo_root()
    if cfg is None:
        cfg = load_config()
    mean_cfg = float(cfg["normalization"]["statistics"]["mean"])
    std_cfg = float(cfg["normalization"]["statistics"]["std"])
    common = raw_image_root(root)
    acc = acc_sq = 0.0
    n = 0
    with open(os.path.join(root, "data", "manifests", "train_split.csv"),
              newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            with Image.open(os.path.join(common, *r["image_path"].split("/"))) as im:
                g = np.asarray(im.convert("L"), dtype=np.float64) / 255.0
            acc += float(g.sum())
            acc_sq += float((g ** 2).sum())
            n += g.size
    mean = acc / n
    std = float(np.sqrt(acc_sq / n - mean ** 2))
    if abs(mean - mean_cfg) > 1e-6 or abs(std - std_cfg) > 1e-6:
        raise PreprocessingError(
            f"train statistics drift: computed ({mean:.6f}, {std:.6f}) "
            f"vs config ({mean_cfg:.6f}, {std_cfg:.6f})")
    return mean, std


def load_and_preprocess(raw_path: str, cfg: dict) -> np.ndarray:
    """Deterministic preprocessing of one raw image -> uint16 (H, W).

    load -> convert('L') -> resize BILINEAR (224,224) -> /255 ->
    (x - mean)/std -> offset/scale -> uint16. Raises explicitly on
    unreadable input.
    """
    size = (int(cfg["image"]["size"]["width"]), int(cfg["image"]["size"]["height"]))
    mean = float(cfg["normalization"]["statistics"]["mean"])
    std = float(cfg["normalization"]["statistics"]["std"])
    offset = float(cfg["cache"]["storage_transform"]["offset"])
    scale = float(cfg["cache"]["storage_transform"]["scale"])
    try:
        with Image.open(raw_path) as im:
            im.load()
            g = im.convert("L")
    except Exception as e:  # noqa: BLE001 - explicit failure required
        raise PreprocessingError(f"unreadable image {raw_path!r}: {e}") from e
    if g.size != size:
        g = g.resize(size, Image.BILINEAR)
    x = np.asarray(g, dtype=np.float64) / 255.0
    norm = (x - mean) / std
    q = np.rint((norm + offset) * scale)
    if q.min() < 0 or q.max() > 65535:
        raise PreprocessingError(f"quantization overflow for {raw_path!r}")
    return q.astype(np.uint16)


def decode_cached(arr: np.ndarray, cfg: dict) -> np.ndarray:
    """Exact integer inverse of the storage transform -> normalized floats."""
    offset = float(cfg["cache"]["storage_transform"]["offset"])
    scale = float(cfg["cache"]["storage_transform"]["scale"])
    return arr.astype(np.float64) / scale - offset


def augment(image_l: Image.Image, split: str, rng: random.Random) -> Image.Image:
    """Training-time-only augmentation (horizontal flip, p=0.5).

    NOT used during cache generation. Raises for any split other than
    'train' so validation/test can never be augmented by accident.
    """
    if split != "train":
        raise PreprocessingError(
            f"augmentation attempted for split {split!r} — training-only")
    if rng.random() < 0.5:
        return image_l.transpose(Image.FLIP_LEFT_RIGHT)
    return image_l


def _out_name(image_path: str) -> str:
    """Deterministic, collision-free cache file name for a raw path.

    v1.0.1: '<path with / -> __>' + '.png'. The ORIGINAL extension is kept
    inside the name because the dataset contains same-stem .png/.jpg pairs
    (e.g. `benign (100)-rotated1.png` and `.jpg`); stripping it caused
    cache-name collisions. Every cache file is therefore a real PNG file
    (16-bit grayscale) whose name embeds the source format.
    """
    return image_path.replace("/", "__") + ".png"


def build_split_cache(split: str, root: str, cfg: dict,
                      common: str) -> Dict[str, object]:
    manifest = os.path.join(root, "data", "manifests", f"{split}_split.csv")
    out_dir = os.path.join(root, "data", "processed", split)
    os.makedirs(out_dir, exist_ok=True)
    rows = list(csv.DictReader(open(manifest, newline="", encoding="utf-8")))
    rows.sort(key=lambda r: r["image_path"])

    prov_rows: List[dict] = []
    names = set()
    qa = {"n": 0, "blank": 0, "clip_low": 0, "clip_high": 0,
          "sum": 0.0, "sum_sq": 0.0, "n_px": 0, "min": 1e9, "max": -1e9}
    for r in rows:
        out_name = _out_name(r["image_path"])
        if out_name in names:
            raise PreprocessingError(f"cache name collision: {out_name}")
        names.add(out_name)
        arr = load_and_preprocess(os.path.join(common, *r["image_path"].split("/")), cfg)
        if arr.shape != (int(cfg["image"]["size"]["height"]),
                         int(cfg["image"]["size"]["width"])):
            raise PreprocessingError(f"shape mismatch for {r['image_path']}")
        out_path = os.path.join(out_dir, out_name)
        img = Image.fromarray(arr)  # uint16 2-D -> mode 'I;16' (native mapping)
        if img.mode != "I;16":
            raise PreprocessingError(
                f"unexpected cached mode {img.mode!r} (want I;16)")
        img.save(out_path, format="PNG")
        # automated QA on the encoded values (all images, cheap ops)
        f = arr.astype(np.float64)
        qa["n"] += 1
        qa["blank"] += 1 if float(f.std()) < 1e-9 else 0
        qa["clip_low"] += 1 if int(f.min()) == 0 else 0
        qa["clip_high"] += 1 if int(f.max()) == 65535 else 0
        qa["sum"] += float(f.sum())
        qa["sum_sq"] += float((f ** 2).sum())
        qa["n_px"] += f.size
        qa["min"] = min(qa["min"], float(f.min()))
        qa["max"] = max(qa["max"], float(f.max()))
        prov_rows.append({
            "output_file": out_name,
            "image_path": r["image_path"],
            "md5": r["md5"],
            "bag_id": r["bag_id"],
            "source_group_id": r["source_group_id"],
            "bag_label": r["bag_label"],
            "split": split,
            "source_key_status": r["source_key_status"],
            "original_supplied_split": r["original_supplied_split"],
        })

    prov_path = os.path.join(out_dir, "provenance.csv")
    with open(prov_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PROC_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(prov_rows)

    mean_q = qa["sum"] / qa["n_px"]
    std_q = float(np.sqrt(qa["sum_sq"] / qa["n_px"] - mean_q ** 2))
    return {
        "split": split,
        "images": qa["n"],
        "blank_images": qa["blank"],
        "images_clipped_low": qa["clip_low"],
        "images_clipped_high": qa["clip_high"],
        "uint16_min": qa["min"],
        "uint16_max": qa["max"],
        "uint16_mean": round(mean_q, 4),
        "uint16_std": round(std_q, 4),
        "provenance_sha256": hashlib.sha256(
            open(prov_path, "rb").read()).hexdigest(),
        "rows": len(prov_rows),
    }


def build_all(root: str = None) -> dict:
    if root is None:
        root = repo_root()
    digest = verify_freeze(root)
    counts = verify_split_counts(root)
    cfg = load_config()
    verify_train_stats(root, cfg)
    common = raw_image_root(root)
    results = [build_split_cache(s, root, cfg, common) for s in SPLITS]
    summary = {
        "phase5_version": cfg["config"]["version"],
        "config": CONFIG_RELPATH.replace(os.sep, "/"),
        "pipeline_version": cfg["config"]["pipeline_version"],
        "frozen_test_sha256_verified": digest,
        "split_instance_counts": counts,
        "results": results,
        "augmentation_in_cache": "none (train augmentation is training-time only, never cached)",
    }
    with open(os.path.join(root, "data", "processed", "cache_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    return summary


def main() -> int:  # pragma: no cover
    root = repo_root()
    summary = build_all(root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
