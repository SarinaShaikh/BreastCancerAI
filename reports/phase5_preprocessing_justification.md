# Phase 5 — Preprocessing Justification Report

**Project:** Predicting Breast Cancer from Ultrasound Images with Cross-Modal Validation using Dual Attention Multiple Instance Learning
**Phase:** 5 — Ultrasound Image Preprocessing & Augmentation Pipeline
**Status:** Complete — all Phase 5 tasks executed; all validation checks pass (§26)
**Date:** 2026-09-18

---

## 1. Objective

Implement a reproducible, leakage-safe preprocessing pipeline for the ultrasound dataset, respecting the frozen Phase 4 split, and document every preprocessing decision with executed evidence rather than generic computer-vision convention. Roadmap: `PROGRESS.md` Phase 5. Configuration: `configs/preprocessing_config.yaml` (v1.0.0; pipeline v1.0.1). Implementation: `src/preprocessing/pipeline.py`.

---

## 2. Phase 4 dependency and freeze

All Phase 5 work consumes the committed Phase 4 artifacts and never modifies them:

- `data/manifests/{train,val,test}_split.csv` — the authoritative split (6,327 / 1,339 / 1,350 instances; 496 source groups; 9,016 total).
- Frozen test manifest SHA256 **verified at pipeline start, mid-phase, and at suite time**: `959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74` (exact hash also recorded in §27); every Phase 5 entry point calls `verify_freeze()`, which hard-fails on any mismatch before processing.
- The split unit (one Phase 2/3 `source_group_id` family) and all Phase 4 assignments are untouched.

---

## 3. Observed image formats (raw dataset, verified on disk this phase)

Phase 1 established PNG + JPEG; Phase 5 re-verified the formats across the full dataset: **8,520 PNG and 496 JPEG files** (the 496 JPEGs correspond to the 496 family originals; the 8,520 PNGs are the embedded augmentation variants). All inspected images open as **RGB containers**; no grayscale (`L`) or palette (`P`) containers were observed.

---

## 4. Observed dimensions

Executed full-dataset scan: **8,520 instances are exactly 224×224 (94.5%)** and the **496 originals are exactly 227×227**. No other dimensions were observed. The dataset is uniformly square.

---

## 5–7. Selected target resolution, resize method, and channel analysis

**Target: 224×224, PIL `Image.BILINEAR`, aspect ratio preserved (no distortion — sources are square).**

Rationale (grounded in §4):
- 224×224 reproduces the dataset's dominant native grid **exactly** — 94.5% of instances undergo zero resampling; the 496 originals change by only 3 px per side (uniform scale 0.9887).
- Computationally practical for the intended MIL prototype; identical treatment for all splits.
- Bilinear was chosen over sharper kernels (e.g. Lanczos) to avoid introducing new sharpening-ringing that could interact with the dataset's existing sharpened-variant lineage (Phase 1/2 evidence). No claim of clinical optimality is made.

**Channel analysis (executed 902-image systematic sample, every ~10th file):** 14/902 images (1.55%, all PNG) contain any chroma deviation >16/255; the worst affected pixel fraction in a single image was **6.35%** (`train/benign/benign (277)-sharpened-sharpened.png`). The remaining ~98.5% of sampled images are pure grayscale inside RGB containers. The chroma-bearing regions are consistent with burned-in ultrasound-machine overlays/markers (documented observation, per-image filenames preserved in the sample records). No claim is made that the colored pixels are universally artifacts; no evidence exists that overlay color carries diagnostic information.

**Channel decision: canonical single-channel grayscale (`convert('L')` for every image, before resize).**

---

## 8. Normalization

Per-pixel affine: `(x/255 - mean) / std`, applied identically to every split.

- `mean = 0.316515`, `std = 0.245010` (6 d.p. in the config; the pipeline recomputes full-precision train statistics at every run and hard-fails if they disagree with the config constants by >1e-6 — `verify_train_stats()`).
- Output range for this dataset ≈ **[−1.29, 2.79]**.

---

## 9. Normalization-statistics source

**TRAIN split ONLY (6,327 instances), computed from the raw files in L mode.** No validation or test image contributed to the statistics. The pipeline recomputes the statistics from the train split at every run and aborts if the config constants drift.

---

## 10. Test-data exclusion from parameter fitting/selection

- Normalization statistics: train-only (§9), recomputed and enforced in code.
- Despeckling evaluation (§13): **train-derived samples only** (60 images, deterministic stride selection).
- Target size, channel handling, and augmentation policy decisions used train/full-dataset structural properties only (dimensions, format counts, channel composition) — structural facts, not performance signals.
- No preprocessing parameter was selected or tuned using validation or test data, and no test-set performance signal exists in Phase 5 (no model has been built).

---

## 11. Ultrasound intensity/noise observations

From the executed QA metrics on the full cached dataset and the 902-image chroma sample:
- Pure grayscale content in RGB containers; intensity is the entire diagnostic signal carried in the cache.
- No blank images and no saturated/clipped images detected (automated QA on all 9,016 cached images: `blank_images = 0`, no low/high clipping).
- Speckle noise is inherent to ultrasound B-mode imaging and is present, but the executed despeckling measurements (§13) showed the relevant trade-off: suppression of texture/edge energy affects speckle and tissue texture alike.

---

## 12–14. Despeckling evaluation and decision

**Decision: EXCLUDED from the finalized preprocessing pipeline.**

Executed bounded evaluation on **train-derived samples only** (60 images, deterministic stride selection, gradient-magnitude energy after filtering vs before, on the 224×224 grayscale pipeline representation):

| Filter | Texture/edge energy retained | Suppressed |
|---|---|---|
| 3×3 median (PIL `MedianFilter`) | 79.1% | 20.9% |
| Gaussian σ=0.5 (PIL `GaussianBlur`) | 91.7% | 8.3% |

Speckle and tissue texture are not separable here without downstream model evidence, so any suppression risks the diagnostically relevant texture the roadmap's risk note warns about. **No Phase 5 evidence justifies including a despeckling filter**; exclusion is the conservative, evidence-grounded choice. A later phase may revisit this with model evidence, without touching the frozen test set. No claim of diagnostic benefit is made or permitted for any filter.

---

## 15–16. Cropping evaluation and lesion-annotation limitation

- **No lesion-aware cropping is performed because lesion annotations/masks are unavailable** (Phase 1 exhaustively verified: no lesion masks, annotations, or verified coordinates exist; none were fabricated).
- Non-lesion-specific cropping was also evaluated and **excluded**: no evidence justifies content-derived cropping, and cropping away image regions merely to make images look cleaner is prohibited by the roadmap. Full-frame processing after resize is the finalized policy.

---

## 17. Existing dataset augmentation lineage (scientific integrity)

The **source dataset already contains augmented image files** (rotation/sharpening chains documented by Phase 1/2; 8,520 variants + 496 originals). Those files are part of the supplied dataset, are represented via Phase 2 `source_group_id` lineage, and are **NOT independent patients, studies, lesions, or independent clinical observations**. Phase 3 definitions are preserved untouched: BAG = one `source_group_id`; INSTANCE = one image file of the group; LABEL = directory-encoded benign/malignant class. Source keys remain `inferred_from_filename_not_verified_identifier` in every provenance row.

---

## 18–21. New augmentation evaluation and final split policies

Evaluation: the dataset's lineage already covers rotation and sharpening variants; duplicating those transforms would add nothing new. The one transform absent from the lineage is a **horizontal flip**, so the pipeline provides **horizontal flip (p=0.5), training-time only, seeded (20260918)** — a genuinely new invariance, applied **on the fly at training time and NEVER cached**. Phase 4's group-level splitting keeps all variants of a family in one split, so no cross-split leakage is possible; Phase 3 bag definitions are preserved.

- **Training augmentation policy:** horizontal flip p=0.5, train split only, seeded/deterministic under the documented seed, applied at training time (provided by `pipeline.augment()`; never invoked during cache generation).
- **Validation augmentation policy: NONE** — deterministic preprocessing only.
- **Test augmentation policy: NONE** — deterministic preprocessing only; frozen until Phase 10.
- Enforcement: `augment()` raises `PreprocessingError` for any split ≠ `train`; automated check V5-F verifies the guard, seed reproducibility, and that the caches contain exactly 1:1 manifest rows (no augmented variants cached).

---

## 22. Versioned configuration

`configs/preprocessing_config.yaml` — `config.version = 1.0.0`, `pipeline_version = 1.0.1`, `random_seed = 20260918`. Every field contains a finalized executed value (no placeholders/TODOs). The pipeline validates the config version and hard-fails on mismatch. Contents: input formats, target size, resize method, channel handling, normalization statistics + source, storage transform, cache layout, crop policy, despeckling policy, train/val/test augmentation policies, test-set protection rules, reproducibility instructions.

---

## 23. Processed-cache structure

```
data/processed/
├── cache_summary.json        # deterministic build summary (versions, counts, QA, provenance hashes)
├── train/  6,327 images + provenance.csv
├── val/      1,339 images + provenance.csv
└── test/     1,350 images + provenance.csv
```

Split membership is explicit by directory; splits are never mixed. Every cached image is a **16-bit grayscale PNG (`I;16`), 224×224**, encoding `uint16 = round((normalized + 8) × 4096)` (lossless round-trip at 1/4096 normalized-unit precision; exact integer inverse `decode_cached()`).

---

## 24. Cache naming and provenance strategy

**Naming (v1.0.1):** `<raw relative path with / → __>.png` — the **original extension is preserved inside the name** (then `.png` appended as the real format extension). The dataset contains same-stem `.png`/`.jpg` pairs (e.g. `benign (100)-rotated1.png` and `benign (100)-rotated1.jpg`); **858 same-stem multi-format groups exist and 0 collide** under this rule (executed full-dataset verification). Every cache file is a real PNG whose name embeds its source format; v1.0.0's extension-stripping collision defect was found by this phase's own validation and fixed.

**Provenance** (`data/processed/{split}/provenance.csv`, LF, sorted by `image_path`): columns `output_file, image_path, md5, bag_id, source_group_id, bag_label, split, source_key_status, original_supplied_split` — every processed image traces to its raw source path + Phase 1 md5 + Phase 3 bag/source-group + Phase 4 split + label. **No patient/study/lesion/MRI columns exist**; `source_key_status = inferred_from_filename_not_verified_identifier` on every row, preserving the established distinction that filename-derived source keys are not verified identifiers.

---

## 25. Visual sample review

`reports/phase5_visual_review.pdf` + `reports/phase5_visual_review_index.csv` (deterministic; regenerated after the naming fix; byte-identical across repeated runs): 6 grids (benign/malignant × train/val/test) × 12 tiles = **72 tiles**, even-stride sampled from the provenance rows. Per-tile QA metrics in normalized units, plus blank/clipping flags: **0 anomalies detected** (no blank, no clipped images among displayed samples; consistent with the full-dataset QA in §11).

**Limitations of this review:** it is a technical data-quality inspection only — orientation, distortion, clipping, blanking, channel handling, and gross artifacts. It is **not a clinical assessment**; no diagnostic validity, superiority, or clinical benefit may be inferred from it.

---

## 26. Automated validation results

Phase 5 suite `tests/test_phase5_preprocessing.py` (V5-A…V5-J): **11/11 PASS** (see below for per-check results). Regressions after all Phase 5 changes: Phase 1 **11/11**, Phase 2 **11/11**, Phase 3 **13/13**, Phase 4 **11/11**.

| Check | Result | Evidence |
|---|---|---|
| V5-A image loading | PASS | PNG + JPEG load via the pipeline; missing image raises `PreprocessingError` (no silent skip) |
| V5-B output shape/format | PASS | PNG, `I;16` single-channel, 224×224 per config |
| V5-C deterministic preprocessing | PASS | 10 sampled images: repeat calls + cache bytes identical |
| V5-C2 train-only normalization statistics | PASS | full train-split recomputation matches config (mean 0.316515, std 0.245010); report wording verified |
| V5-D split coverage | PASS | 6,327 + 1,339 + 1,350 cached images == Phase 4 manifests; no missing/extra; no duplicate outputs |
| V5-E split disjointness | PASS | pairwise disjoint by provenance `image_path`; `split` field matches directory |
| V5-F training-only augmentation | PASS | val/test raise; train flip seeded+reproducible; cache = 1:1 manifest rows; `augmentation_in_cache: none` |
| V5-G frozen manifest integrity | PASS | SHA256 `959f1cd3d9f71919…` matches Phase 4 report + freeze record |
| V5-H provenance/traceability | PASS | md5 chain to Phase 1 manifest on 20 rows/split; paths resolve; status marker on every row |
| V5-I raw-data immutability | PASS | 9,016/9,016 raw files md5-identical to the Phase 1 manifest (full re-hash) |
| V5-J phase boundary | PASS | AST analysis of Phase 5 modules: no ML/training/eval imports or exact-name calls; no `checkpoints/` |

Additional executed verifications beyond the suite: full cache re-run determinism (byte-identical rebuild across fresh runs); cached-vs-direct byte identity (30/30); cache-mapping audit (9,016 unique raw→output, 0 collisions incl. 858 same-stem groups, all fields vs manifests); flip-rate 107/200 ≈ 0.5 on a non-constant probe; independent `sha256sum` cross-verification of the frozen manifest.

---

## 27. Test protection

- **Frozen test manifest unchanged** — `data/manifests/test_split.csv` SHA256 = `959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74`, independently re-verified with `sha256sum` at phase start, mid-phase, and suite time; row count 1,350; group assignments unchanged.
- **No test-derived normalization statistics** — train-only, enforced by recomputation in `verify_train_stats()`.
- **No test augmentation** — code guard + V5-F.
- **No test-based preprocessing selection** — despeckling/size/channel decisions used train or full-dataset structural evidence only (§10).
- **No Phase 4 split changes** — split manifests and `src/data/splitting.py` untouched (git confirms zero diff).

---

## 28. Reproducibility instructions

```
python -m src.preprocessing.pipeline    # rebuilds all three split caches byte-identically
python src/preprocessing/visual_review.py  # regenerates the visual-review artifacts byte-identically
python tests/test_phase5_preprocessing.py  # full Phase 5 validation suite
```

Fixed seed (20260918); explicit ordering (sorted `image_path`); no filesystem-order or hash-randomization dependence; no timestamps inside cache artifacts; `cache_summary.json` is deterministic. The build hard-fails if the frozen test hash, split counts, config version, or train statistics drift.

---

## 29. Limitations

- **Despeckling excluded**: measured texture-energy suppression (20.9% median / 8.3% Gaussian) cannot be shown to spare diagnostically relevant texture without downstream model evidence; exclusion is the evidence-grounded choice. A later phase may revisit with model evidence.
- **No lesion-aware cropping**: impossible without lesion annotations/masks (Phase 1); background/non-lesion regions remain in frame.
- **Overlay chroma discarded** (~1.5% of sampled images): intensity contrast is retained; color is not represented in the cache; no evidence indicates diagnostic color information exists.
- **Filename-derived source keys are not verified identifiers**; patient/study-level leakage remains structurally unassessable on this dataset (Phase 4 limitation, preserved).
- **224×224 from 227×227 originals** slightly resamples the 496 originals (uniform scale 0.9887); no clinical-optimality claim is made.
- **Visual review is not clinical validation**; it inspects technical image quality only.
- Existing dataset augmentation lineage means instance counts ≠ independent observations; Phase 3 bag semantics preserve this.

---

## 30. Phase 5 completion status

**Phase 5: COMPLETE** — all roadmap tasks executed (loading, resizing, normalization, ultrasound intensity/noise evaluation, despeckling evaluation + justified exclusion, cropping evaluation + justified exclusion, training-only augmentation policy, versioned config, cached preprocessing for all splits, visual sample review, automated training-only-augmentation verification), all validation checks pass (11/11 + regressions green), test-set protection verified, raw data untouched. Phase 6+ has **not** been started. Phase 5 awaits owner review and commit authorization.

---

*Clinical disclaimer: this report documents technical data preprocessing only. No clinical validation, diagnostic claim, or benefit of any preprocessing choice is claimed or implied.*
