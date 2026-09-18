"""Phase 5: deterministic visual-sample review artifact.

Roadmap: PROGRESS.md Phase 5 validation check 1 — "Visual sample review
of preprocessed images per class". Produces a reviewable grid PDF (12
samples per split x class = 72 tiles) from the PROCESSED cache, plus
automated image-quality metrics per displayed image (min/max/mean/std,
blank/clipping flags). Deterministic: fixed stride sampling from the
provenance CSV, no timestamps in the PDF, LF text.

The review is a data-quality inspection, NOT a clinical assessment: no
diagnostic validity or superiority may be inferred from it.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SPLITS = ("train", "val", "test")
CLASSES = ("benign", "malignant")
PER_GRID = 12          # 12 samples per split x class
QA_LIMITS = {"blank_std": 1.0, "clip_low": 2, "clip_high": 65533}


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def qa_on_cached(path: str) -> dict:
    """Decode a cached I;16 PNG to normalized units and compute QA metrics."""
    a = np.asarray(Image.open(path)).astype(np.float64)
    norm = a / 4096.0 - 8.0
    return {
        "nmin": round(float(norm.min()), 4),
        "nmax": round(float(norm.max()), 4),
        "nmean": round(float(norm.mean()), 4),
        "nstd": round(float(norm.std()), 4),
        "blank": bool(norm.std() < 1e-6),
        "clip_low": bool(a.min() <= QA_LIMITS["clip_low"]),
        "clip_high": bool(a.max() >= QA_LIMITS["clip_high"]),
    }


def select_samples(split: str, cls: str, n: int = PER_GRID) -> list:
    """Deterministic even stride over the provenance rows for split+class."""
    rows = [r for r in csv.DictReader(
                open(os.path.join(repo_root(), "data", "processed", split,
                                  "provenance.csv"), newline="", encoding="utf-8"))
            if r["bag_label"] == cls]
    rows.sort(key=lambda r: r["output_file"])
    if len(rows) <= n:
        return rows
    stride = len(rows) / n
    return [rows[min(len(rows) - 1, int(i * stride))] for i in range(n)]


def build(out_dir: str = None) -> dict:
    root = repo_root()
    out_dir = out_dir or os.path.join(root, "reports")
    os.makedirs(out_dir, exist_ok=True)

    grids = []
    index_rows = []
    anomalies = []
    for split in SPLITS:
        for cls in CLASSES:
            samples = select_samples(split, cls)
            cell = 140
            label_h = 26
            grid = Image.new("L", (cell * 6, (cell + label_h) * 2), 235)
            draw = ImageDraw.Draw(grid)
            for i, r in enumerate(samples):
                qa = qa_on_cached(os.path.join(root, "data", "processed", split,
                                               r["output_file"]))
                col, row = i % 6, i // 6
                x0, y0 = col * cell, row * (cell + label_h)
                tile = Image.fromarray(
                    (np.asarray(Image.open(os.path.join(
                        root, "data", "processed", split, r["output_file"])))
                     .astype(np.float64) / 4096.0 - 8.0)
                    .clip(-1.3, 2.8) / 4.1 * 255).convert("L")
                grid.paste(tile.resize((cell - 4, cell - 4)), (x0 + 2, y0 + 2))
                draw.text((x0 + 3, y0 + cell - 2),
                          f"{i+1}:{r['output_file'][:22]}", fill=0, font=_font(8))
                flags = [k for k in ("blank", "clip_low", "clip_high") if qa[k]]
                if flags:
                    anomalies.append({"split": split, "class": cls,
                                      "file": r["output_file"], "flags": flags,
                                      "qa": qa})
                index_rows.append({
                    "split": split, "class": cls, "position": i + 1,
                    "output_file": r["output_file"],
                    "image_path": r["image_path"], "md5": r["md5"],
                    "bag_id": r["bag_id"], "source_group_id": r["source_group_id"],
                    "qa_min": qa["nmin"], "qa_max": qa["nmax"],
                    "qa_mean": qa["nmean"], "qa_std": qa["nstd"],
                    "flags": "|".join(flags),
                })
            grids.append((f"{split} - {cls} (12 deterministic samples)", grid))

    from reportlab.pdfgen import canvas as pdfcanvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    pdf_path = os.path.join(out_dir, "phase5_visual_review.pdf")
    c = pdfcanvas.Canvas(pdf_path, pagesize=A4, invariant=1)
    c.setTitle("Phase 5 visual sample review (processed cache)")
    W, H = A4
    margin = 14 * mm
    c.setFont("Helvetica", 9)
    y = H - margin
    for title, grid in grids:
        gw = W - 2 * margin
        gh = gw * grid.height / grid.width
        if y - gh - 8 * mm < margin:
            c.showPage()
            c.setFont("Helvetica", 9)
            y = H - margin
        c.drawString(margin, y - 8, title)
        c.drawImage(ImageReader(grid), margin, y - 8 - gh, width=gw, height=gh)
        y -= gh + 12 * mm
    c.save()

    index_path = os.path.join(out_dir, "phase5_visual_review_index.csv")
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()),
                           lineterminator="\n")
        w.writeheader()
        w.writerows(index_rows)

    return {
        "pdf": os.path.relpath(pdf_path, root),
        "index": os.path.relpath(index_path, root),
        "tiles": len(index_rows),
        "grids": len(grids),
        "anomalies": anomalies,
        "qa_note": "min/max/mean/std in normalized units; blank/clipping flags per displayed image",
    }


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
