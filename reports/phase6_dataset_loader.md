# Phase 5.5 — Ultrasound Bag Data Loader & Dataset Interface Report

**Project:** Predicting Breast Cancer from Ultrasound Images with Cross-Modal Validation using Dual Attention Multiple Instance Learning
**Phase:** 5.5 — Ultrasound Bag Data Loader & Dataset Interface (NEW section added 2026-09-18 by owner decision)
**Status:** Complete — implementation finished, all validation checks pass (§13)
**Date:** 2026-09-18

---

## 1. Objective

Provide a clean, deterministic, testable **input/data interface** through which later model phases (roadmap Phases 6–8) request `split → bags → instances → labels → metadata` **without redefining bag membership or labels**. This is a data-interface phase only: no model, no MIL, no attention, no training, no evaluation, no feature extraction.

**Roadmap reconciliation (owner decision 2026-09-18):** the existing roadmap defines Phase 6 as "Baseline Deep Learning Models" (unchanged, still Not Started). The requested loader phase was recorded as a **new Phase 5.5 section** in `PROGRESS.md` rather than rewriting the roadmap's phase sequence. Implementation lives in `src/mil/bag_dataset.py` per the owner's D-1 mapping decision (MIL infrastructure → `src/mil/`, beside Phase 3's `bag_definition.py`).

## 2. Phase 4 dependency and freeze

The loader consumes the frozen Phase 4 manifests and **enforces the freeze at every construction**: it calls the Phase 5 `verify_freeze()` (single source of truth for the hash), verifies split counts (6,327 / 1,339 / 1,350 = 9,016), and hard-fails on any drift before loading anything. **Phase 4 manifests were not modified.**

**Discrepancy note (transparency):** the Phase 6 instruction text quoted the frozen hash as `959f1cd3d9f719195ac31af1a23698ec7616c376714112b74` — missing the `586d7c84` segment relative to the authoritative hash `959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74` recorded by Phase 4 (report §19, Phase 5 config/test/report, PROGRESS.md). The on-disk manifest matches the authoritative hash exactly; the instruction string was a transcription truncation, not a data change. All Phase 5.5 artifacts use the authoritative hash.

## 3. Dataset interface (public API)

`src/mil/bag_dataset.py` — module `src.mil.bag_dataset`:

| API | Returns | Notes |
|---|---|---|
| `UltrasoundBagDataset(split, root=None, processed_root=None, verify_cache_files=True)` | dataset | `split ∈ {train, val, test}`; validates freeze, counts, manifests, provenance bijection, cross-split disjointness, bag completeness/coherence at construction |
| `.bags` | `tuple[BagRecord]` | ordered by ascending `bag_id` |
| `.instances` | `tuple[InstanceRecord]` | ordered by ascending `image_path` |
| `.bag_ids`, `.bag_sizes`, `len()`, `__getitem__`, `__iter__` | — | deterministic accessors |
| `.bag(bag_id)` | `BagRecord` | raises `DatasetError` if absent from split |
| `.load_instance_image(record)` | `float32 (224, 224)` | Phase 5 exact inverse decode; no re-normalization, no augmentation |
| `.load_bag_images(bag)` | `float32 (n_i, 224, 224)` | variable-size, no padding/duplication/discard |
| `.load_bag_with_label(bag)` | `(images, numeric_label)` | convenience for later phases |
| `UltrasoundBagDataset.collate_bags(batches)` | dict | `images (B,N_max,224,224)` zero-padded + `mask (B,N_max)` bool + `labels` + `bag_ids` + `instance_counts` |
| `.summary()` | dict | counts, bag-size stats, label counts, ordering rule |
| `build_all_summaries()` | dict | summaries for all three splits |

`BagRecord` exposes: `bag_id`, `source_group_id`, `label`, `split`, `source_key`, `source_key_status`, `instances`, `instance_count`, `numeric_label`, `instance_paths`.
`InstanceRecord` exposes: `image_path` (raw reference), `processed_relpath`, `md5`, `bag_id`, `source_group_id`, `label`, `split`, `source_key`, `source_key_status`, `original_supplied_split`, `numeric_label`.

Errors raise `DatasetError` with actionable messages (e.g. missing cache → "regenerate the Phase 5 cache with `python -m src.preprocessing.pipeline`").

## 4. Bag definition (consumed, not changed)

- **BAG** = one Phase 2 `source_group_id` (source-image/augmentation family) per Phase 3 — **not** a verified patient/study/lesion identifier.
- **INSTANCE** = one image file of that source group.
- **LABEL** = the group's directory-encoded class.
- No new grouping logic; loader consumes Phase 3 `instance_list` membership exactly (verified row-for-row by V6-J/V6-G).

## 5. Instance definition

One image file, fully provenance-traced: raw reference (`image_path` + Phase 1 `md5`) → Phase 2 `source_group_id` → Phase 3 `bag_id` → Phase 4 frozen split → Phase 5 processed path → Phase 6 loader. The processed file name embeds the original file name **including its source extension** (Phase 5 v1.0.1 collision-free naming; 858 same-stem `.png`/`.jpg` groups remain distinguished).

## 6. Label mapping (verified from artifacts)

Directory-encoded class verified across Phase 1 (audit), Phase 3 (`bag_manifest.bag_label`), Phase 4 (split-manifest `bag_label`) and Phase 5 (`provenance.bag_label`): **`benign` / `malignant`**, exactly one label per bag, identical across all four artifact layers.

- Authoritative string labels: `benign`, `malignant`.
- Documented numeric convention: **`LABEL_MAP = {benign: 0, malignant: 1}`** (benign = negative class). Provided for later model phases; the string label remains authoritative. No label is ever inferred from pixels or predictions.

## 7. Train/val/test counts (loaded via the interface)

| Split | Bags | Instances | benign bags | malignant bags | Instances benign/malignant share |
|---|---|---|---|---|---|
| train | 349 | 6,327 | 200 | 149 | 50.5% / 49.5% |
| val | 74 | 1,339 | 43 | 31 | 51.4% / 48.6% |
| test | 73 | 1,350 | 43 | 30 | 51.0% / 49.0% |
| **total** | **496** | **9,016** | **286** | **210** | — |

## 8. Bag-size statistics (per split; variable sizes preserved)

| Split | min | max | mean | median | histogram |
|---|---|---|---|---|---|
| train | 14 | 21 | 18.1289 | 16.0 | {14×1, 16×199, 21×149} |
| val | 16 | 21 | 18.0946 | 16.0 | {16×43, 21×31} |
| test | 16 | 53 | 18.4932 | 16.0 | {16×43, 21×29, 53×1} |

(The 53-instance and 14-instance bags are the two originally cross-split families `benign (36)` and `malignant (18)`, resolved whole by Phase 4.)

## 9. Deterministic ordering rule

- **Bags:** ascending `bag_id` (lowercase-hex ASCII sort).
- **Instances:** ascending `image_path` (ASCII lexicographic on the relative raw path) — identical to both the frozen split manifests' row order and the Phase 3 `instance_list` order (all three verified equal, V6-J).
- No filesystem enumeration order, no Python hash randomization. Same-process reconstruction (V6-I/J) and two fresh processes (V6-Q) produce identical orderings.

## 10. Provenance chain and leakage validation

Full chain preserved on every record: raw image → Phase 2 `source_group_id` → Phase 3 bag → Phase 4 split → Phase 5 processed representation → Phase 6 loader. `source_key_status = inferred_from_filename_not_verified_identifier` is carried and verified on every record (loader records **and** the Phase 1 manifest); source keys are never rewritten as patient/study/lesion identifiers.

Validated (executed in V6 suite + loader construction, which re-validates on every instantiation):
- no bag occurs in multiple splits; no instance occurs in multiple splits (V6-E/F);
- loader split membership exactly matches the frozen manifests, field-for-field (V6-G);
- every manifest instance resolves to exactly one processed representation and vice versa (bijection; V6-D);
- labels valid and consistent within every bag (V6-H);
- Phase 4 frozen SHA256 unchanged (V6-R, enforced inside the loader).

## 11. Phase 5 compatibility (contract preserved)

- Consumes the Phase 5 cache rather than re-preprocessing: `load_instance_image` performs the **exact integer inverse** of Phase 5's storage transform (`uint16 → /4096 − 8`), verified to equal a direct `pipeline.load_and_preprocess` decode (V6-K).
- 224×224, single-channel, float32 output; mode/size validated per file (`I;16`).
- **No augmentation anywhere in the loader** (the training-time flip remains Phase 5's `pipeline.augment()` API); val/test loads are byte-identical (V6-L).
- No statistics recomputation; construction performs no pixel reads at all (V6-M).
- No despeckling, no cropping, no additional preprocessing introduced.
- Phase 5 config v1.0.0 / pipeline v1.0.1 untouched; the Phase 5 Git policy is respected (cache PNGs stay gitignored; the loader documents local regeneration).

## 12. Variable-bag handling, memory, and fresh-clone behavior

- Variable bag sizes are **preserved end-to-end** (14/16/21/53); no resizing, duplication, or discarding.
- `collate_bags` provides the only padded representation: explicit bool `mask`, `True` **only** at real-instance positions; padded pixels are exactly 0.0 and must be excluded via the mask (verified V6-N); `instance_counts == mask.sum(axis=1)`.
- Memory: lazy, on-demand loading; construction parses manifests and stats files only — never decodes all images (verified V6-M).
- Fresh clone without cache PNGs: construction/loading fails with an actionable error directing to `python -m src.preprocessing.pipeline` (no auto-download, no silent regeneration).

## 13. Validation results

Phase 5.5 suite `tests/test_phase6_dataset_loader.py` — **19/19 PASS** (V6-A…V6-S):

| Check | Result |
|---|---|
| V6-A manifests load · V6-B instance counts · V6-C bag counts · V6-D one processed representation per instance · V6-E bag disjointness · V6-F instance disjointness · V6-G membership == frozen manifests · V6-H label validity/consistency · V6-I bag-order determinism · V6-J instance-order determinism · V6-K Phase 5 contract (shape/dtype/range/decode) · V6-L no val/test augmentation · V6-M no test statistics · V6-N collate mask semantics · V6-O missing/corrupt error behavior · V6-P provenance chain · V6-Q fresh-process determinism · V6-R frozen SHA unchanged · V6-S phase boundary (AST) | **all PASS** |

Regressions after all Phase 5.5 changes: Phase 1 **11/11** · Phase 2 **11/11** · Phase 3 **13/13** · Phase 4 **11/11** · Phase 5 **11/11**.

One defect found and fixed during validation: `InstanceRecord.processed_relpath` was built with `os.path.join` (Windows backslashes in metadata). Fixed to deterministic POSIX-style relative paths; a check covering this field (V6-P) was corrected in the same pass to assert the marker on both the loader record and the Phase 1 manifest. No test was weakened.

## 14. Limitations

- `source_group_id`/`bag_id` are source-image family constructs, **not** verified patient/study/lesion identifiers — patient-level leakage remains structurally unassessable on this dataset (Phase 1/4 limitation, preserved).
- `LABEL_MAP` (benign=0, malignant=1) is a documented convention, not a clinical encoding.
- The loader does not shuffle; later phases should use their own seeded shuffling for training (documented interface decision — determinism is this phase's contract).
- Bag-size stats are split-level properties of the frozen Phase 4 allocation; the loader never alters them.
- The frozen hash transcription discrepancy in the instruction text is documented in §2; the authoritative hash is unchanged.

## 15. Phase 5.5 completion status

**Phase 5.5: COMPLETE** — interface implemented, 19/19 checks pass, all regression suites green, Phase 4 freeze and Phase 5 contract preserved, no Phase 6+ functionality introduced. Awaiting owner review and commit authorization. Roadmap Phase 6 ("Baseline Deep Learning Models") remains **Not Started**.

---

*No clinical claims are made or implied by this phase. This is a data-interface deliverable only.*
