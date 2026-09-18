"""Phase 2: perceptual hashing helpers (dHash / aHash) for duplicate detection.

Roadmap: PROGRESS.md Phase 2 ("Implement perceptual hashing (e.g., pHash/
aHash/dHash) to detect near-duplicates and rotated/sharpened variants").
Decision D-1 maps this file to src/preprocessing/perceptual_hash.py. Raw data
under dataset/raw/ is immutable evidence: this module never writes inside the
dataset root.

Design notes
------------
- The Phase 1 audited dHash (64-bit difference hash, audit.py v1.1.2) is the
  primary perceptual hash. It is REUSED, not reimplemented, so Phase 2
  near-duplicate candidates are bit-identical to the committed Phase 1
  candidates (data/manifests/near_duplicate_candidates.csv); any change to the
  dHash definition would silently break that equivalence.
- aHash (64-bit average hash) is provided as a secondary hash for
  corroboration. Both are deterministic; Phase 2 computes them exhaustively
  over all images (no sampling).
- Perceptual hashes produce CANDIDATES, not confirmed duplicates. Confirmed
  identity in Phase 2 comes from content (md5) or the dataset's own filename
  grammar — see duplicate_detection.py for the grouping policy.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
from PIL import Image

# Reuse the Phase 1 audited implementation (D-1: src/data/audit.py).
_AUDIT_PATH = Path(__file__).resolve().parents[1] / "data" / "audit.py"
_spec = importlib.util.spec_from_file_location("phase1_audit", _AUDIT_PATH)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)

IMG_EXTS = audit.IMG_EXTS


def dhash64_image(im: Image.Image) -> int:
    """64-bit dHash of a PIL image (Phase 1 definition, reused verbatim)."""
    return audit.dhash64(im)


def dhash_hex(v: int) -> str:
    return audit.dhash_hex(v)


def ahash64(im: Image.Image) -> int:
    """64-bit average hash: grayscale, 8x8, threshold at the mean.

    Deterministic; corroboration only, never used for grouping decisions.
    """
    g = im.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    arr = np.asarray(g, dtype=np.uint8)
    bits = 0
    mean = float(arr.mean())
    for px in arr.ravel():
        bits = (bits << 1) | (1 if float(px) >= mean else 0)
    return bits


def ahash_hex(v: int) -> str:
    return f"{v:016x}"


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def list_images(root: str) -> list[str]:
    """Deterministic sorted list of image relative paths under root."""
    rels: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                rels.append(os.path.relpath(os.path.join(dirpath, fn), root)
                            .replace("\\", "/"))
    return sorted(rels)


def hash_image_file(path: str) -> dict:
    """Decode one image; return its dHash/aHash hex (exhaustive-pass helper)."""
    with Image.open(path) as im:
        im.load()
        d = dhash64_image(im)
        a = ahash64(im)
    return {"dhash_hex": dhash_hex(d), "ahash_hex": ahash_hex(a), "read_ok": True}
