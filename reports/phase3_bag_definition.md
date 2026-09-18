# Phase 3 — Data Organization & MIL Bag Definition

**Project:** Predicting Breast Cancer from Ultrasound Images with Cross-Modal
Validation using Dual Attention Multiple Instance Learning
**Phase:** 3 — Data Organization & MIL Bag Definition
**Status:** Complete (pending owner review / commit authorization)
**Tool:** `src/mil/bag_definition.py` v1.0.0 (deterministic; no timestamps in output)
**Generated manifest:** `data/manifests/bag_manifest.csv` (496 rows, LF)
**Date:** 2026-09-18

## 1. Phase 3 objective

Define, based only on verified Phase 1/2 findings, what constitutes a MIL
"bag", an "instance", and a bag "label" — and build the bag manifest that
downstream splitting (Phase 4) and training (Phases 6–8) will consume. No
features, CNNs, attention, splitting, preprocessing, training, checkpoints,
or experiments are implemented in this phase.

## 2. Phase 1 evidence relevant to identifiers

From `reports/phase1_dataset_audit.md` (owner-approved baseline):

* **9,016 images**; exhaustive filename/metadata search found **0 metadata,
  annotation, or README files** — the dataset is a bare image tree.
* **No verified patient IDs, study IDs, or lesion IDs exist.** The only
  per-image identity-like token is the filename-derived `source_key`
  (e.g. `benign (36)`), which is labelled in every manifest row as
  `inferred_from_filename_not_verified_identifier` and must never be
  represented as a patient or study identifier.
* **No lesion annotations or masks** exist anywhere in the dataset.
* Labels are **directory-encoded** (`benign` / `malignant`) with no label
  files to cross-check; no medical meaning beyond the directory names is
  claimed.

## 3. Phase 2 evidence relevant to source groups

From `reports/phase2_leakage_analysis.md` and
`data/manifests/augmentation_groups.csv`:

* **496 `source_group_id` families** cover all 9,016 images exactly once
  (V2-1); groups are connected components over **confirmed-identity edges
  only** (same filename source key ∪ same md5).
* Every group is **label-coherent**: `group_label` == `class_dir` for all
  9,016 rows; 0 label conflicts (Phase 2 label guard).
* Group sizes: 14×1, 16×285, 21×209, 53×1 (min 14, max 53).
* **2 groups span the supplied train/val directories**
  (`grp-3281cc7851ab` = `benign (36)`, `grp-6f66ae71bafc` =
  `malignant (18)`; 37 images). These are source-lineage families, **not**
  cross-split exact-duplicate groups (`cross_split_exact_groups = 0`).
* Near-duplicate candidates (25,607 pairs) remain **candidates only**;
  they never merged groups and do not affect bag construction.
* Grouping plausibility was confirmed by the owner's human visual
  spot-check (all 10 sections Plausible, 2026-09-18).

## 4. Candidate bag definitions considered

Roadmap Phase 3 lists five candidates. Evaluation against verified evidence:

| Candidate | Requires | Evidence available | Verdict |
|---|---|---|---|
| A. Patient → images | verified patient IDs | **None** (Phase 1 exhaustive search) | **Rejected — unsupported** |
| B. Study → images | verified study IDs | **None** | **Rejected — unsupported** |
| C. Lesion → image/patch instances | verified lesion IDs/annotations | **None** (no annotations/masks) | **Rejected — unsupported** |
| D. Source-image-group → augmented variants | reliable source grouping | **496 verified, label-coherent, visually spot-checked groups** | **Selected** |
| E. Single image → tiled patches | none (fallback if grouping unreliable) | grouping is reliable, so the fallback condition does not hold | Evaluated; **not selected** (see §9) |

## 5. Why patient-level grouping is unsupported

Phase 1 performed an exhaustive audit (all 9,016 filenames parsed; full
dataset tree inspected): no patient identifiers, no directory levels above
class, no metadata files. Constructing patient bags would require fabricating
IDs — prohibited by the project rules (rule 9) and by the owner-approved
Phase 1 sign-off ("filename-derived source keys must never be represented as
verified patient identifiers"). Candidate A is therefore unsupported and
rejected.

## 6. Why study-level grouping is unsupported

The dataset contains no study identifiers, timestamps, or session markers of
any kind. There is no evidence that images sharing a source key were acquired
in the same or different imaging sessions. Candidate B is unsupported and
rejected.

## 7. Why lesion-level grouping is unsupported

No lesion IDs, lesion annotations, masks, or region-of-interest files exist
(Phase 1, exhaustive search). A lesion-level bag would require inventing
clinical structure the dataset does not contain. Candidate C is unsupported
and rejected.

## 8. Evaluation of source-image-group bags (selected)

Candidate D was verified against the actual manifests rather than assumed:

* **Coherent labels:** every one of the 496 groups has exactly one label,
  equal to its directory class (re-verified in V3-4); Phase 2's label guard
  recorded 0 conflicts.
* **Membership:** every member carries the group's single `source_group_id`
  and single `source_key` (re-verified in V3-5); membership derives from
  Phase 1 filename grammar + md5 evidence, unchanged in Phase 3.
* **Traceability:** every group member is a Phase 1 manifest row with
  matching md5 and resolves to a raw image path (V3-3, V3-11).
* **Size manageability:** sizes 14–53 (mean 18.18, median 16) are within the
  range standard attention-MIL implementations handle; no empty or singleton
  bags exist. Very small/large bags (14; 53) are flagged for Phase 7/8
  attention, as the roadmap's risk note requires.
* **Cross-split integrity:** the 2 multi-split families remain **single
  groups/bags** — grouping is split-agnostic by the approved Phase 2 policy,
  which is exactly what Phase 4 group-level splitting requires (V3-13).

## 9. Evaluation of image-as-bag-of-patches alternative (not selected)

Candidate E (one image = bag of tiled patches) is the roadmap's fallback for
the case where grouping is unreliable. The verified evidence does not meet
that condition: the 496-family structure is filename- and content-corroborated
(228 md5 groups fully absorbed by lineage; content-purity check 0 mixed-class;
owner visual spot-check all-plausible). Patch-tiles would additionally
(a) discard the dataset's own family structure that Phase 2 established,
(b) treat pixels of one image as if they were separate observations, and
(c) compound, not reduce, the augmentation-correlation problem. Candidate E
is therefore documented as evaluated-but-not-selected. **No tiling or
preprocessing pipeline was implemented** (that machinery belongs to Phase 5
if ever needed).

## 10. Final selected definition

* **BAG = one Phase 2 `source_group_id`** — one source-image family
  (one filename-derived source key and its confirmed-identity variants).
* **INSTANCE = one image file** belonging to that source group (the family's
  original candidate plus its augmented variants, as recorded in
  `augmentation_groups.csv`).
* **LABEL = the group's directory-encoded class** (`benign` / `malignant`),
  identical for every member of the group.

## 11. Source-group / augmentation relationship (MIL integrity note)

A bag's instances are **related image variants of a single source-image
family** — one underlying ultrasound source image (per the dataset's own
naming scheme) plus its rotation/sharpening derivatives. They are **NOT**
independent patients, studies, lesions, or independent clinical observations,
and must never be presented or used as such. Consequences recorded for later
phases (no methodology change made here):

* MIL attention over such a bag aggregates evidence across **correlated**
  variants of one source image.
* The bag label is a **family-level** label inherited from the dataset's
  directory structure, not an independently verified per-observation
  diagnosis.
* Near-duplicate candidates across bags (25,607 pairs) remain candidates;
  no bag was merged or split on their basis.

## 12. Limitation: source groups are not verified identities

The 496 bags operate at the **highest grouping level the data verifiably
supports**. A source group is a filename/bytes-derived family, not a verified
patient, study, or lesion. If real patient identity matters for a future
claim (e.g. patient-level generalisation), that claim is **not supported**
by this dataset. This decision — using the highest valid available grouping
level instead of fabricating identifiers — is recorded explicitly, per the
Phase 3 roadmap task.

## 13. Bag-size statistics (computed from `bag_manifest.csv`)

| Statistic | Value |
|---|---|
| Bags | 496 |
| Total instances | 9,016 |
| Min bag size | 14 |
| Max bag size | 53 |
| Mean bag size | 18.1774 |
| Median bag size | 16.0 |

Size histogram: 14×1, 16×285, 21×209, 53×1.
Bags by label: 286 benign, 210 malignant.
Multi-split bags (single bags spanning the supplied train/val directories):
`bag-3281cc7851ab` (`benign (36)`, 16 instances), `bag-6f66ae71bafc`
(`malignant (18)`, 21 instances) — 2 of 496.

## 14. Traceability / provenance

`bag_id` is the deterministic 1:1 mapping `grp-XXXXXXXXXXXX` →
`bag-XXXXXXXXXXXX` of the Phase 2 `source_group_id`; no random components.
Chain of custody for every instance:

```
bag_manifest.instance_list
  → augmentation_groups.csv row (source_group_id, md5, source_key, role)
    → dataset_manifest.csv row (Phase 1: md5, dimensions, dhash)
      → immutable raw image under dataset/raw/.../ultrasound breast classification/
```

Instance lists are ordered by image path; bags are ordered by `bag_id`;
the manifest uses LF line endings and contains no timestamps, so repeated
generation is byte-identical (V3-6, V3-12). No images were discarded: all
9,016 Phase 2 rows appear in exactly one bag (V3-2).

## 15. Exact manifest generated

`data/manifests/bag_manifest.csv` — 496 data rows, columns:

`bag_id, source_group_id, bag_label, instance_count, instance_list,
instance_md5_list, source_key, source_key_status, splits, multi_split,
original_candidate_path`

* `instance_list` / `instance_md5_list` — path / md5 joined with `|`
  (CSV-safe deterministic representation of membership).
* `source_key_status` — always
  `inferred_from_filename_not_verified_identifier` (anti-fabrication marker
  carried through from Phase 1).
* `splits` / `multi_split` — descriptive record of where the family's files
  happen to sit in the **supplied** directory layout; this is **not** a
  Phase 4 split. No train/val/test assignment is made in Phase 3.

## 16. Conceptual MIL hierarchy (documentation only — NOT implemented here)

```
BAG        one source_group_id (source-image family)          [IMPLEMENTED Phase 3]
 └──       INSTANCE  one image file of the family             [IMPLEMENTED Phase 3]
      └──   FEATURES  per-instance CNN/embedding extraction    [conceptual — Phase 6+]
       └──  ATTENTION instance weighting / evidence aggregation [conceptual — Phase 7/8]
        └── PREDICTION bag-level benign/malignant              [conceptual — Phase 6+]
```

Phases 6–8 must supply the feature extractor, attention mechanism, and
prediction head; nothing in `src/mil/bag_definition.py` performs any of
these steps (verified by V3-10).

## 17. Validation results

Independent suite `tests/test_phase3_bags.py` (V3-1…V3-13), executed
2026-09-18 — **13/13 PASS**:

| Check | Verifies | Result |
|---|---|---|
| V3-1 | every Phase 2 group appears exactly once as a bag | PASS |
| V3-2 | every Phase 2 image appears in exactly one bag | PASS |
| V3-3 | every instance traces to Phase 1/2 rows (md5 chain) | PASS |
| V3-4 | one unambiguous label per bag; no mixed-label bags | PASS |
| V3-5 | every instance's `source_group_id` equals its bag's | PASS |
| V3-6 | deterministic ordering — byte-identical regeneration | PASS |
| V3-7 | min/max/mean/median recomputed and match this report | PASS |
| V3-8 | no fabricated patient/study/lesion identifiers | PASS |
| V3-9 | raw dataset unchanged (full re-hash digest) | PASS |
| V3-10 | no Phase 4+ functionality introduced | PASS |
| V3-11 | columns/IDs/counts/paths all valid; 9,016 paths resolve | PASS |
| V3-12 | two regenerations byte-identical to committed manifest | PASS |
| V3-13 | the 2 multi-split families remain single bags | PASS |

Phase 1 (11/11) and Phase 2 (11/11) suites re-run after Phase 3 changes —
both still PASS.

## 18. Data-driven limitations

1. **Family-level, not patient-level:** bags are the finest defensible
   unit; no patient-level claim is possible with this dataset.
2. **Correlated instances:** instances are augmented variants of one source
   image; attention evidence within a bag is not independent evidence.
3. **Label provenance:** labels are directory names, not histopathology-
   verified ground truth (dataset-level limitation inherited from Phase 1).
4. **Two bags span the supplied split:** Phase 4 must split by bag; any
   image-level splitting would leak these families.
5. **Bag-size spread (14–53)** may require size-aware batching/attention in
   Phases 7/8 (roadmap risk note).
6. **Near-duplicate candidates across bags** (996 cross-split candidate
   pairs) are untouched by bag definition and remain Phase 4 leakage-test
   inputs.

---

*Report based on executed Phase 3 output (bag_definition.py v1.0.0, run
twice, byte-identical) and the committed Phase 1/2 manifests. No number in
this document is hand-computed; every figure traces to
`data/manifests/bag_manifest.csv`, `phase2_grouping_summary.json`, or an
executed validation check.*
