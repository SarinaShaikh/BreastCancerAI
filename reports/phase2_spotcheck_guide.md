# Phase 2 - Human Visual Spot-Check Guide

Roadmap requirement (PROGRESS.md, Phase 2, Validation Checks):
> Spot-check a random sample of grouped images **visually** to confirm
> grouping plausibility - manual/visual spot-checking is required.

The machine-checkable V2-9 content-purity test does **not** satisfy this
requirement; this contact sheet + index exist so a human can perform it.

## How to review
For each tile/pair, judge whether the stated relationship is plausible
from the visible image content. Record your verdict (plausible / not
plausible / unsure) per section or per tile; the review is complete when
every section has a verdict.

## Label legend (strict)
- **CONFIRMED identity** - md5-equal files, or same filename source key
  (the dataset's own naming scheme). Not a patient/study identifier.
- **NEAR-DUP CANDIDATE** - dHash distance <= 7 from the Phase 1 candidate
  set. Candidates are NOT confirmed duplicates.
- **CROSS-SPLIT** - endpoints lie in the supplied train/val split.
- **CROSS-KEY** - endpoints belong to different source keys.

## Sections
- **S1 - Augmentation/source-lineage families (whole groups)** (12 item(s); relationship: CONFIRMED identity (same filename source key))
- **S2 - Exact-duplicate groups (first 10 of 228, sorted)** (24 item(s); relationship: CONFIRMED identity (identical md5))
- **S3 - Rotated/sharpened augmentation chains (family grp-e35c859a925c)** (12 item(s); relationship: CONFIRMED identity (same source key; chain order per filename))
- **S4 - Largest family (53 members, first 12)** (12 item(s); relationship: CONFIRMED identity (same source key))
- **S5 - Original-vs-augmented decision (smallest family, originals marked)** (1 item(s); relationship: CONFIRMED identity (same source key); role decision per Phase 2 rules)
- **S6 - Cross-split same-source-key candidate pairs (all 19)** (19 item(s); relationship: NEAR-DUP CANDIDATE (NOT confirmed) + CROSS-SPLIT)
- **S7 - Cross-split cross-key relationships (best pair of all 19 key-pairs)** (19 item(s); relationship: NEAR-DUP CANDIDATE (NOT confirmed) + CROSS-SPLIT + CROSS-KEY)
- **S8 - dHash=0 cross-key examples (deterministic sample of 329)** (12 item(s); relationship: NEAR-DUP CANDIDATE (NOT confirmed); dHash identical, bytes differ)
- **S9 - Low-SSIM candidates (dHash>=5 & SSIM<0.60; 12 weakest)** (12 item(s); relationship: NEAR-DUP CANDIDATE (NOT confirmed); SSIM contradicts dHash)
- **S10 - The 2 families spanning the supplied train/val split (all 37 images)** (37 item(s); relationship: CONFIRMED same source key + CROSS-SPLIT family membership)

Total displayed: 160 images/pairs across 10 sections
(selection is deterministic from the committed manifests; regenerate with
`python src/preprocessing/contact_sheet.py`).

## Provenance
- Images: immutable raw dataset (`dataset/raw/`, unmodified; re-verified).
- Relationships: `data/manifests/dataset_manifest.csv`,
  `augmentation_groups.csv`, `near_duplicate_candidates.csv`,
  `ssim_crosscheck.csv` (all committed Phase 1/2 artifacts).
- Companion index: `reports/phase2_spotcheck_index.csv`.
