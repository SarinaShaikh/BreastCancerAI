"""Phase 2 human visual spot-check: deterministic contact-sheet generator.

Roadmap context: PROGRESS.md Phase 2 validation check 2 requires a HUMAN
visual spot-check of grouped images ("manual/visual spot-checking is
required"). The machine-checkable V2-9 content-purity test does NOT satisfy
that requirement; this tool only PREPARES the human review. It renders a
deterministic PDF contact sheet plus a companion index CSV from the already
committed Phase 1/Phase 2 manifests — no new methodology, thresholds, or
grouping decisions; selection uses only documented Phase 2 evidence.

Relationship labels are strict:
  CONFIRMED IDENTITY      - md5-equal (exact content) or same filename source
                            key (dataset's own naming scheme)
  NEAR-DUP CANDIDATE      - dHash distance <= 7 (Phase 1 candidate set);
                            NEVER presented as confirmed identity
  CROSS-SPLIT (candidate) - the pair spans the supplied train/val split

Outputs (reports/):
  phase2_spotcheck_contact_sheet.pdf   the visual sheet (A4 portrait)
  phase2_spotcheck_index.csv           one row per displayed image/pair
  phase2_spotcheck_guide.md            how to review; selection criteria

Raw data is read-only; nothing is written inside dataset/raw/.

Usage:
    python src/preprocessing/contact_sheet.py [--manifests-dir DIR]
        [--out-dir DIR] [--max-per-section N]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageFont

# Reuse the Phase 1 audited module (root discovery, dHash) verbatim.
_AUDIT_PATH = Path(__file__).resolve().parents[1] / "data" / "audit.py"
_spec = importlib.util.spec_from_file_location("phase1_audit", _AUDIT_PATH)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)

CONTACT_VERSION = "1.0.0"

# SSIM lookup for candidate pairs, loaded from the committed Phase 2 CSV.
_ssim_lookup: dict[tuple[str, str], float] = {}

# ---------------------------------------------------------------------------
# deterministic selection from committed manifests
# ---------------------------------------------------------------------------


def load_manifests(manifests_dir: str) -> tuple[list[dict], list[dict], list[dict]]:
    p1 = list(csv.DictReader(open(os.path.join(manifests_dir, "dataset_manifest.csv"),
                                  encoding="utf-8", newline="")))
    groups = list(csv.DictReader(open(os.path.join(manifests_dir, "augmentation_groups.csv"),
                                      encoding="utf-8", newline="")))
    nd = list(csv.DictReader(open(os.path.join(manifests_dir, "near_duplicate_candidates.csv"),
                                  encoding="utf-8", newline="")))
    return p1, groups, nd


def select_sections(p1: list[dict], groups: list[dict], nd: list[dict],
                    max_per_section: int = 12) -> list[dict]:
    """Deterministic sections; each item = one rendered tile or pair.

    Categories are exactly those documented in reports/phase2_leakage_analysis.md;
    no new thresholds or categories are introduced.
    """
    by_path = {r["path"]: r for r in p1}
    groups_by_id = defaultdict(list)
    for r in groups:
        groups_by_id[r["source_group_id"]].append(r)

    sections: list[dict] = []

    # --- S1: three whole augmentation/source-lineage families -------------
    # (deterministic exemplars: first key alphabetically, median-size family,
    # and the smallest family - all documented group structure)
    sizes = {g: (len(ms), ms[0]["source_key"]) for g, ms in groups_by_id.items()}
    sorted_gids = sorted(groups_by_id)
    smallest = min(sorted_gids, key=lambda g: (sizes[g][0], g))
    median = sorted(sorted_gids, key=lambda g: (sizes[g][0], g))[len(sorted_gids) // 2]
    first = sorted_gids[0]
    fam = []
    for gid in [first, median, smallest]:
        for r in sorted(groups_by_id[gid], key=lambda x: x["path"]):
            fam.append({"img": r["path"],
                        "rel": ("CONFIRMED same source key ("
                                + r["source_key"] + "; chain "
                                + (r["chain"] or "(base)") + ")"),
                        "group_id": gid})
    sections.append({"id": "S1", "title": "Augmentation/source-lineage families (whole groups)",
                     "relationship": "CONFIRMED identity (same filename source key)",
                     "kind": "group", "items": fam[:max_per_section]})

    # --- S2: exact-duplicate groups (CONFIRMED md5 identity) --------------
    dup_by_group = defaultdict(list)
    for r in p1:
        if r["duplicate_group"]:
            dup_by_group[r["duplicate_group"]].append(r)
    dup_gids = sorted(dup_by_group)[:max_per_section]
    dup_items = []
    for dg in dup_gids:
        for r in sorted(dup_by_group[dg], key=lambda x: x["path"]):
            dup_items.append({"img": r["path"],
                              "rel": "CONFIRMED md5-identical (group " + dg + ")",
                              "group_id": next(m["source_group_id"] for m in groups
                                               if m["path"] == r["path"])})
    sections.append({"id": "S2", "title": "Exact-duplicate groups (first 10 of 228, sorted)",
                     "relationship": "CONFIRMED identity (identical md5)",
                     "kind": "group", "items": dup_items})

    # --- S3: augmentation chains inside one family (median family) --------
    chain_items = []
    for r in sorted(groups_by_id[median], key=lambda x: (len(x["chain"]), x["path"])):
        chain_items.append({"img": r["path"],
                            "rel": ("CONFIRMED same source key; chain depth "
                                    + str(len(r["chain"])) + " ("
                                    + (r["chain"] or "base/original candidate")
                                    + "); role=" + r["role"] + "/" + r["role_confidence"]),
                            "group_id": median})
    sections.append({"id": "S3", "title": "Rotated/sharpened augmentation chains (family " + median + ")",
                     "relationship": "CONFIRMED identity (same source key; chain order per filename)",
                     "kind": "group", "items": chain_items[:max_per_section]})

    # --- S4: largest family (malignant (1), 53 members) -------------------
    largest = max(sorted_gids, key=lambda g: (sizes[g][0], g))
    large_items = [{"img": r["path"],
                    "rel": ("CONFIRMED same source key; largest family; chain "
                            + (r["chain"] or "(base)")),
                    "group_id": largest}
                   for r in sorted(groups_by_id[largest], key=lambda x: x["path"])]
    sections.append({"id": "S4", "title": "Largest family (53 members, first 12)",
                     "relationship": "CONFIRMED identity (same source key)",
                     "kind": "group", "items": large_items[:max_per_section]})

    # --- S5: original-vs-augmented decision (one family, all roles) -------
    orig = [r for r in groups_by_id[smallest] if r["role"] == "original_candidate"]
    orig_items = [{"img": r["path"],
                   "rel": "CONFIRMED same source key; role=" + r["role"] + "/"
                          + r["role_confidence"] + " (rule " + r["role_rule"] + ")",
                   "group_id": smallest}
                  for r in sorted(orig, key=lambda x: x["path"])]
    sections.append({"id": "S5", "title": "Original-vs-augmented decision (smallest family, originals marked)",
                     "relationship": "CONFIRMED identity (same source key); role decision per Phase 2 rules",
                     "kind": "group", "items": orig_items})

    # --- S6: cross-split SAME-KEY pairs (candidates; family-level leak) ---
    # All 19 cross-split candidate pairs whose endpoints share a source key.
    samekey_cross = []
    for x in nd:
        if x["same_split"] == "False" and x["same_source_key"] == "True":
            samekey_cross.append(x)
    pair_items = []
    for x in sorted(samekey_cross, key=lambda r: (r["hamming"], r["file_a"], r["file_b"])):
        s = _ssim_lookup.get((x["file_a"], x["file_b"]))
        pair_items.append({"pair": (x["file_a"], x["file_b"]), "d": int(x["hamming"]),
                           "ssim": s,
                           "rel": ("NEAR-DUP CANDIDATE, CROSS-SPLIT, same source key "
                                   "(family-level leak)")})
    sections.append({"id": "S6", "title": "Cross-split same-source-key candidate pairs (all 19)",
                     "relationship": "NEAR-DUP CANDIDATE (NOT confirmed) + CROSS-SPLIT",
                     "kind": "pair", "items": pair_items})

    # --- S7: cross-split CROSS-KEY pairs (the 977-signal; 19 key-pairs) ---
    kp = {}
    for x in nd:
        if x["same_split"] == "False" and x["same_source_key"] == "False":
            k = tuple(sorted((by_path[x["file_a"]]["source_key"],
                              by_path[x["file_b"]]["source_key"])))
            cur = kp.get(k)
            if cur is None or (int(x["hamming"]), x["file_a"], x["file_b"]) < \
                    (int(cur["hamming"]), cur["file_a"], cur["file_b"]):
                kp[k] = x
    crosskey_items = []
    for k in sorted(kp):
        x = kp[k]
        s = _ssim_lookup.get((x["file_a"], x["file_b"]))
        crosskey_items.append({"pair": (x["file_a"], x["file_b"]), "d": int(x["hamming"]),
                               "ssim": s,
                               "rel": ("NEAR-DUP CANDIDATE, CROSS-SPLIT, CROSS-KEY ("
                                       + k[0] + " vs " + k[1] + ")")})
    sections.append({"id": "S7", "title": "Cross-split cross-key relationships (best pair of all 19 key-pairs)",
                     "relationship": "NEAR-DUP CANDIDATE (NOT confirmed) + CROSS-SPLIT + CROSS-KEY",
                     "kind": "pair", "items": crosskey_items})

    # --- S8: dHash=0 cross-key examples (strongest candidate signal) ------
    d0 = []
    for x in nd:
        if x["same_source_key"] == "False" and x["hamming"] == "0":
            s = _ssim_lookup.get((x["file_a"], x["file_b"]))
            d0.append({"pair": (x["file_a"], x["file_b"]), "d": 0, "ssim": s,
                       "rel": ("NEAR-DUP CANDIDATE at dHash 0, CROSS-KEY "
                               + ("+ CROSS-SPLIT" if x["same_split"] == "False" else "same-split")
                               + " - plausibility of dHash-0 across distinct keys")})
    d0.sort(key=lambda r: (r["pair"][0], r["pair"][1]))
    step = max(1, len(d0) // max_per_section)
    sections.append({"id": "S8", "title": "dHash=0 cross-key examples (deterministic sample of "
                                           + str(len(d0)) + ")",
                     "relationship": "NEAR-DUP CANDIDATE (NOT confirmed); dHash identical, bytes differ",
                     "kind": "pair", "items": d0[::step][:max_per_section]})

    # --- S9: high-distance / low-SSIM candidates (over-grouping risk) -----
    low = []
    for x in nd:
        s = _ssim_lookup.get((x["file_a"], x["file_b"]))
        if s is not None and s < 0.60 and int(x["hamming"]) >= 5:
            low.append({"pair": (x["file_a"], x["file_b"]), "d": int(x["hamming"]),
                        "ssim": s,
                        "rel": "NEAR-DUP CANDIDATE with SSIM<0.60 - dHash over-grouping risk"})
    low.sort(key=lambda r: (r["ssim"], r["pair"][0], r["pair"][1]))
    sections.append({"id": "S9", "title": "Low-SSIM candidates (dHash>=5 & SSIM<0.60; 12 weakest)",
                     "relationship": "NEAR-DUP CANDIDATE (NOT confirmed); SSIM contradicts dHash",
                     "kind": "pair", "items": low[:max_per_section]})

    # --- S10: the two multi-split families in full (37 images) ------------
    ms_gids = sorted(g for g, ms in groups_by_id.items()
                     if len({m["split"] for m in ms}) > 1)
    ms_items = []
    for gid in ms_gids:
        for r in sorted(groups_by_id[gid], key=lambda x: x["path"]):
            ms_items.append({"img": r["path"],
                             "rel": ("CONFIRMED same source key; multi-split family ("
                                     + r["source_key"] + "); split=" + r["split"]),
                             "group_id": gid})
    sections.append({"id": "S10", "title": "The 2 families spanning the supplied train/val split (all 37 images)",
                     "relationship": "CONFIRMED same source key + CROSS-SPLIT family membership",
                     "kind": "group", "items": ms_items})
    return sections


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

TILE = 220          # thumbnail box (px at 150 dpi)
PAD = 10
LABEL_H = 56


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def _load_thumb(common: str, rel: str) -> Image.Image:
    with Image.open(os.path.join(common, *rel.split("/"))) as im:
        im.load()
        t = im.convert("RGB").copy()
    t.thumbnail((TILE, TILE))
    return t


def build_pdf(sections: list[dict], common: str, out_pdf: str,
              title: str) -> list[dict]:
    """Render sections; returns index rows (one per displayed image)."""
    from reportlab.pdfgen import canvas as pdfcanvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader

    index_rows: list[dict] = []
    W, H = A4
    margin = 12 * mm
    cols = 4
    tile_w = (W - 2 * margin - (cols - 1) * 6) / cols
    tile_h = tile_w * 1.18
    c = pdfcanvas.Canvas(out_pdf, pagesize=A4, invariant=1)
    c.setTitle(title)
    x0, y = margin, H - margin - 16

    def header(text: str, sub: str | None = None) -> None:
        nonlocal y
        c.setFont("Helvetica-Bold", 9)
        c.drawString(x0, y, text)
        y -= 11
        if sub:
            c.setFont("Helvetica-Oblique", 7)
            c.drawString(x0, y, sub)
            y -= 9
        y -= 2

    header(title, "Phase 2 human visual spot-check - generated " +
           datetime.now(timezone.utc).date().isoformat() +
           " (deterministic selection from committed manifests)")

    for sec in sections:
        if y < 3 * tile_h + 40:
            c.showPage()
            y = H - margin - 16
        header(f"{sec['id']}. {sec['title']}",
               "Relationship: " + sec["relationship"])
        for i, item in enumerate(sec["items"]):
            col = i % cols
            if col == 0 and i and y < tile_h + 30:
                c.showPage()
                y = H - margin - 16
            x = x0 + col * (tile_w + 6)
            top = y
            if "pair" in item:
                (pa, pb), d, s = item["pair"], item["d"], item["ssim"]
                t1, t2 = _load_thumb(common, pa), _load_thumb(common, pb)
                half = (tile_w - 4) / 2
                for t, rel in ((t1, pa), (t2, pb)):
                    tw, thh = t.size
                    scale = min(half / tw, (tile_h * 0.62) / thh)
                    c.drawImage(ImageReader(t), x + 2 if t is t1 else x + 2 + half + 2,
                                top - tile_h * 0.62 + (tile_h * 0.62 - thh * scale) / 2,
                                width=tw * scale, height=thh * scale)
                c.setFont("Courier", 5.4)
                c.drawString(x + 2, top - tile_h * 0.62 - 8, os.path.basename(pa)[:38])
                c.drawString(x + 2, top - tile_h * 0.62 - 14, os.path.basename(pb)[:38])
                c.setFont("Helvetica-Bold", 6)
                c.setFillColorRGB(0.6, 0, 0)
                c.drawString(x + 2, top - tile_h * 0.62 - 21,
                             f"dHash d={d}  SSIM={s if s is not None else 'n/a'}")
                c.setFillColorRGB(0, 0, 0)
                index_rows.append({"section": sec["id"], "relationship": item["rel"],
                                   "file_a": pa, "file_b": pb, "dhash_d": d,
                                   "ssim_64": s, "group_id": "", "split_a":
                                   pa.split("/")[0], "split_b": pb.split("/")[0]})
            else:
                rel_path = item["img"]
                t = _load_thumb(common, rel_path)
                tw, thh = t.size
                scale = min(tile_w / tw, (tile_h * 0.72) / thh)
                c.drawImage(ImageReader(t), x + 2,
                            top - tile_h * 0.72 + (tile_h * 0.72 - thh * scale) / 2,
                            width=tw * scale, height=thh * scale)
                c.setFont("Courier", 5.2)
                c.drawString(x + 2, top - tile_h * 0.72 - 8, os.path.basename(rel_path)[:40])
                c.setFont("Helvetica", 5.6)
                c.drawString(x + 2, top - tile_h * 0.72 - 15,
                             (item["group_id"] + " | " + rel_path.split("/")[0]
                              + "/" + rel_path.split("/")[1])[:44])
                c.setFont("Helvetica-Oblique", 5.6)
                c.drawString(x + 2, top - tile_h * 0.72 - 22, item["rel"][:60])
                index_rows.append({"section": sec["id"], "relationship": item["rel"],
                                   "file_a": rel_path, "file_b": "", "dhash_d": "",
                                   "ssim_64": "", "group_id": item["group_id"],
                                   "split_a": rel_path.split("/")[0], "split_b": ""})
            if col == cols - 1 or i == len(sec["items"]) - 1:
                y -= tile_h + 10
        y -= 8
    c.save()
    return index_rows


def write_guide(sections: list[dict], out_md: str, total_imgs: int) -> None:
    lines = [
        "# Phase 2 - Human Visual Spot-Check Guide",
        "",
        "Roadmap requirement (PROGRESS.md, Phase 2, Validation Checks):",
        "> Spot-check a random sample of grouped images **visually** to confirm",
        "> grouping plausibility - manual/visual spot-checking is required.",
        "",
        "The machine-checkable V2-9 content-purity test does **not** satisfy this",
        "requirement; this contact sheet + index exist so a human can perform it.",
        "",
        "## How to review",
        "For each tile/pair, judge whether the stated relationship is plausible",
        "from the visible image content. Record your verdict (plausible / not",
        "plausible / unsure) per section or per tile; the review is complete when",
        "every section has a verdict.",
        "",
        "## Label legend (strict)",
        "- **CONFIRMED identity** - md5-equal files, or same filename source key",
        "  (the dataset's own naming scheme). Not a patient/study identifier.",
        "- **NEAR-DUP CANDIDATE** - dHash distance <= 7 from the Phase 1 candidate",
        "  set. Candidates are NOT confirmed duplicates.",
        "- **CROSS-SPLIT** - endpoints lie in the supplied train/val split.",
        "- **CROSS-KEY** - endpoints belong to different source keys.",
        "",
        "## Sections",
    ]
    for s in sections:
        lines.append(f"- **{s['id']} - {s['title']}** ({len(s['items'])} item(s); "
                     f"relationship: {s['relationship']})")
    lines += [
        "",
        f"Total displayed: {total_imgs} images/pairs across {len(sections)} sections",
        "(selection is deterministic from the committed manifests; regenerate with",
        "`python src/preprocessing/contact_sheet.py`).",
        "",
        "## Provenance",
        "- Images: immutable raw dataset (`dataset/raw/`, unmodified; re-verified).",
        "- Relationships: `data/manifests/dataset_manifest.csv`,",
        "  `augmentation_groups.csv`, `near_duplicate_candidates.csv`,",
        "  `ssim_crosscheck.csv` (all committed Phase 1/2 artifacts).",
        "- Companion index: `reports/phase2_spotcheck_index.csv`.",
    ]
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 2 spot-check contact sheet")
    ap.add_argument("--manifests-dir", default="data/manifests")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--max-per-section", type=int, default=12)
    args = ap.parse_args()

    p1, groups, nd = load_manifests(args.manifests_dir)
    _ssim_lookup.clear()
    ssim_csv = os.path.join(args.manifests_dir, "ssim_crosscheck.csv")
    with open(ssim_csv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            _ssim_lookup[(r["file_a"], r["file_b"])] = float(r["ssim_64"])

    root, how, _s = audit.discover_dataset_root(None)
    common, rels = audit.resolve_image_root(root)
    assert set(rels) == {r["path"] for r in p1}, "manifest/dataset drift"

    sections = select_sections(p1, groups, nd, args.max_per_section)
    out_pdf = os.path.join(args.out_dir, "phase2_spotcheck_contact_sheet.pdf")
    index_rows = build_pdf(sections, common, out_pdf,
                           "Phase 2 Visual Spot-Check (human review)")
    write_guide(sections, os.path.join(args.out_dir, "phase2_spotcheck_guide.md"),
                len(index_rows))

    idx_csv = os.path.join(args.out_dir, "phase2_spotcheck_index.csv")
    with open(idx_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["section", "relationship", "file_a",
                                          "file_b", "dhash_d", "ssim_64",
                                          "group_id", "split_a", "split_b"])
        w.writeheader()
        for r in index_rows:
            w.writerow(r)

    n_imgs = sum(2 if r["file_b"] else 1 for r in index_rows)
    print(json.dumps({"sections": len(sections),
                      "index_rows": len(index_rows),
                      "images_shown": n_imgs,
                      "pdf": out_pdf,
                      "index": idx_csv}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
