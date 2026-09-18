# Phase 4 — Patient/Study-Level Data Splitting & Leakage Prevention

**Project:** Predicting Breast Cancer from Ultrasound Images with Cross-Modal
Validation using Dual Attention Multiple Instance Learning
**Phase:** 4 — Patient/Study-Level Data Splitting & Leakage Prevention
**Status:** Complete (pending owner review / commit authorization)
**Tool:** `src/data/splitting.py` v1.0.0 (deterministic; no timestamps in output)
**Generated manifests:** `data/manifests/{train,val,test}_split.csv` (LF)
**Date:** 2026-09-18

## 1. Objective

Create deterministic train/validation/test splits at the highest defensible
grouping level, guaranteeing zero leakage across splits along every
verifiable dimension — source-group membership, confirmed duplicate identity,
near-duplicate candidate relationships, and augmentation-family membership —
and freeze the test set until Phase 10.

## 2. Source artifacts used

* `data/manifests/bag_manifest.csv` (Phase 3, committed `b49f149`) —
  **authoritative** bag/source-group mapping: 496 bags ↔ 496
  `source_group_id`s, 9,016 instances with per-instance md5.
* `data/manifests/near_duplicate_candidates.csv` (Phase 1, committed) —
  25,607 candidate pairs with `file_a`, `file_b`, `hamming` (dHash ≤ 7).
* `data/manifests/dataset_manifest.csv` (Phase 1) — per-image md5 identity.
* `data/manifests/augmentation_groups.csv` (Phase 2) — group/label/role
  cross-check.
* `reports/phase1_dataset_audit.md`, `reports/phase2_leakage_analysis.md`,
  `reports/phase3_bag_definition.md`.

## 3. Available identity metadata

Verified by the Phase 1 exhaustive audit and re-confirmed for Phase 4:

* Patient IDs: **absent** (no metadata files, no ID directory levels).
* Study IDs: **absent**.
* Lesion IDs / annotations / masks: **absent**.
* The only grouping token is the filename-derived `source_key`, labelled in
  every row `inferred_from_filename_not_verified_identifier`.

## 4. Why patient/study/lesion-level splitting is unavailable

A patient-level split requires verified patient identity; a study-level
split requires verified study/session identity; a lesion-level split
requires verified lesion IDs or annotations. None exist in this dataset
(exhaustive Phase 1 audit; no metadata files of any kind). Inventing any of
these identifiers is prohibited by the project rules and would fabricate
clinical structure. Patient-, study-, and lesion-level splitting are
therefore **unavailable**, and the split unit falls to the highest level
that the data verifiably supports.

## 5. Selected split unit

**One complete Phase 3 `source_group_id` / source-image family (BAG).**

All 496 bags are allocated atomically: instances of one source group are
never distributed across splits. Because 25,607 near-duplicate **candidate**
pairs impose same-split constraints (§7), atomic bags alone are insufficient;
bags are therefore allocated as members of deterministic **allocation units**
(connected components of the candidate-constraint graph, §7) — a Phase 4
bookkeeping construct only, not a new identity level.

## 6. Exact definition of the split unit

* `source_group_id` = the Phase 2/3 source-image/augmentation-family
  construct (filename lineage ∪ confirmed md5 identity; candidates never
  merged). It is **NOT** a `patient_id`, **NOT** a `study_id`, and **NOT**
  a `lesion_id`.
* `bag_id` = deterministic Phase 3 1:1 alias of `source_group_id`
  (`grp-…` → `bag-…`).
* Augmented images of one family are **correlated instances**, not
  independent patients, studies, lesions, or independent clinical
  observations.
* Allocation unit id (`unit-…`) = sha256-derived id over the sorted member
  bag ids of one constraint component; likewise not an identity construct.

## 7. Split-generation method

1. Load the committed Phase 3 bag manifest; map every instance path to its
   bag (every candidate endpoint must resolve — enforced).
2. Read ALL 25,607 candidate pairs from the authoritative Phase 2 manifest
   and union the endpoint **bags** (union-find). Rationale: the Phase 4
   test E quantifies over every candidate pair, so treating only the 996
   Phase-2-recorded cross-split pairs as constraints would leave 24,611
   same-split-recorded pairs free to straddle the new Phase 4 boundaries —
   the conservative, leakage-first policy required. Candidates remain
   candidates: no Phase 2 artifact is altered, no candidate is reclassified
   as a confirmed duplicate, and no group membership is changed.
3. Connected components = **356 allocation units** (281 singletons; largest
   spans 39 bags). 6 units contain both benign and malignant bags (371
   candidate pairs are cross-class); co-splitting mixed units whole is the
   conservative choice and affects no bag's label.
4. Stratified greedy allocation, largest-unit-first: each unit goes to the
   split minimizing the worst per-class fill ratio
   max_c (fill_s,c + unit_s,c) / target_s,c against 70/15/15 class-proportional
   targets; ties resolve in fixed order train → val → test.
   Fixed seed/config: **`SPLIT_SEED = 20260918`**, `TARGETS = {70%, 15%, 15%}`,
   deterministic ordering throughout. (The seed pins `random` for future
   variants; the current allocation is fully order-determined and does not
   draw random numbers.)
5. Write one CSV per split, sorted by `image_path`; LF endings; no
   timestamps. Group membership is never altered for balance.

## 8. Deterministic seed/configuration

* `SPLIT_SEED = 20260918` (fixed; recorded in code and report).
* `TARGETS = {"train": 0.70, "val": 0.15, "test": 0.15}` (instance share,
  per class).
* Ordering: units by (instance_count desc, min_bag_id asc); rows by
  `image_path` within each split; unit ids are content hashes of sorted
  member bag ids. No reliance on filesystem order or Python hash
  randomization.

## 9. Train/validation/test counts

| Split | Allocation units | Source groups (= bags) | Instances |
|---|---|---|---|
| train | 249 | 349 | 6,327 |
| val | 53 | 74 | 1,339 |
| test | 54 | 73 | 1,350 |
| **total** | **356** | **496** | **9,016** |

Atomicity: every group in exactly one split (349+74+73 = 496; V-L4a); every
instance exactly once (6,327+1,339+1,350 = 9,016; V-L4b).

## 10. Class balance

| Split | benign inst. | malignant inst. | % benign | % malignant | % of all instances |
|---|---|---|---|---|---|
| train | 3,198 | 3,129 | 50.5% | 49.5% | 70.2% |
| val | 688 | 651 | 51.4% | 48.6% | 14.8% |
| test | 688 | 662 | 51.0% | 49.0% | 15.0% |

Group-level: benign/malignant bags are 286/210 overall; each split's class
share stays within ~1 point of 50% because the near-uniform family sizes
(16/21) make group-grain and instance-grain stratification nearly
equivalent. Deviations from exact 70/15/15 are the expected cost of atomic
units under conservative constraints (e.g. the 39-bag unit = 699 instances
cannot be divided); balance was not optimized at the expense of leakage
prevention, and no group membership was altered. Bag-size distribution per
split: train 14–21 (mean 18.13), val 16–21 (mean 18.09), test 16–53
(mean 18.49); the largest family (53) and the second-largest contiguous
blocks sit in test/train respectively per the largest-first pass.

## 11. Source-group integrity results

`train ∩ val = ∅`, `train ∩ test = ∅`, `val ∩ test = ∅` over
`source_group_id` (V-L4c) — **zero overlap**; 496 = 349+74+73.

## 12. Confirmed duplicate leakage results

All 228 Phase 1 md5-duplicate groups were re-derived from the committed
manifest and each group's members checked against the split maps: **0
confirmed-duplicate relationships cross splits** (V-L4d). (Phase 2 already
showed every md5 pair shares a source key; this test re-verifies the
property against the new Phase 4 boundaries.)

## 13. Near-duplicate candidate leakage results

All **25,607** candidate pairs were mapped endpoint-by-endpoint to their
Phase 4 splits: **0 cross-split candidate pairs** (V-L4e). This includes
the 996 pairs Phase 2 recorded as cross-split under the dataset's original
directory layout (19 same-key + 977 cross-key) — all now co-split by the
constraint policy. Candidate status is unchanged: this is leakage
prevention, not confirmation.

## 14. Augmentation-family leakage results

Every one of the 9,016 instances was verified to share its
`source_group_id`'s split: **0 families span splits** (V-L4f). Family
integrity holds per instance, not merely per group list.

## 15. Patient-level limitation

`patient-level overlap = not assessable because verified patient IDs are
absent.` No patient-level leakage claim is made or testable on this
dataset; no identifiers were fabricated. The leakage guarantee is at the
source-group level and candidate-constraint level only.

## 16. Study-level limitation

`study-level overlap = not assessable because verified study IDs are
absent.` Same prohibition and scope as §15; lesion-level overlap is
equally unassessable (no lesion IDs/annotations exist).

## 17. Original cross-split source families and their Phase 4 resolution

Phase 2/3 identified `benign (36)` (`grp-3281cc7851ab`, 16 instances) and
`malignant (18)` (`grp-6f66ae71bafc`, 21 instances) as families whose files
the dataset's original layout placed in **both** train and val directories
(source-lineage families; `cross_split_exact_groups = 0` — not exact-
duplicate groups). Phase 4 resolution, verified in the generated manifests:

* `grp-3281cc7851ab` → **test** (all 16 instances; rows carry
  `original_supplied_split` = train for 12, val for 4);
* `grp-6f66ae71bafc` → **train** (all 21 instances; 10 train + 11 val
  originally);
* each appears in exactly one split file — no partial assignment.

The original directory placement can no longer create Phase 4 leakage.
`original_supplied_split` is recorded per row for full traceability.

## 18. Test-set freeze date

**2026-09-18.** The test split is frozen as of this date and must remain
byte-identical until Phase 10 evaluation; no preprocessing, model, or
hyperparameter decision may be tuned against it.

## 19. Test-set SHA256

```
959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74
```

Computed over the final `data/manifests/test_split.csv` (1,350 rows, LF,
sorted by `image_path`); reproducible from the committed file and
cross-verified with an independent tool (`sha256sum`). **FROZEN —
2026-09-18.**

## 20. Reproducibility results

Two fresh-process regenerations (`python -m src.data.splitting`) produced
byte-identical train/val/test manifests (sha1 comparison per file), and a
third regeneration after the statistics fix left `test_split.csv`
bit-identical (empty git diffstat). The reported SHA256 is stable and
reproducible from the committed manifest.

## 21. Limitations

1. **Group-level, not patient-level:** leakage prevention is as strong as
   the available identity allows; real patient overlap cannot be excluded
   or quantified from this dataset.
2. **Conservative over-co-splitting:** treating all 25,607 candidates as
   constraints binds 215 extra bags into multi-bag units and mildly reduces
   split-freedom; the alternative (constraining only the 996 recorded
   cross-split pairs) was rejected because test E quantifies over all
   pairs. This trades a small amount of balance for strictly stronger
   leakage prevention.
3. **Cross-class units:** 6 units contain both classes; their bags share a
   split by constraint. No bag's label is affected; family sizes are
   near-uniform so this does not bias stratification materially.
4. **Integer atomism:** exact 70/15/15 is unattainable with 356 atomic
   units; achieved 70.2/14.8/15.0.
5. **Labels remain directory-derived** (Phase 1 limitation inherited): the
   split is clean relative to the dataset's own labels, not to
   histopathology-verified truth.

## 22. Scope statement

Phase 4 is **research-data splitting and leakage prevention** — a
data-organization step for model development. It is **not** clinical
validation, not patient-level validation, not study-level validation, not
lesion-level validation, and not MRI validation; it establishes no clinical
safety or effectiveness claim of any kind.

---

*Every number in this report traces to the committed manifests or an
executed check (`tests/test_leakage.py` V-L4a…V-L4h + independent
recomputation). Test-set SHA256 independently cross-verified with
`sha256sum`. No Phase 2 grouping artifact, Phase 3 bag manifest, or
earlier-phase evidence was modified.*
