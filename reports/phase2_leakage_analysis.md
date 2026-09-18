# Phase 2 — Data Cleaning, Image Integrity & Augmentation Leakage Analysis

**Project:** Predicting Breast Cancer from Ultrasound Images with Cross-Modal Validation using Dual Attention Multiple Instance Learning
**Phase:** 2 — Data Cleaning, Image Integrity & Augmentation Leakage Analysis
**Tooling:** `src/preprocessing/duplicate_detection.py` v1.0.0 + `src/preprocessing/perceptual_hash.py` v1.0.0 (reusing the Phase 1 audited `src/data/audit.py` v1.1.2 verbatim)
**Executed runs:** 2026-09-18 — three full pipeline runs; **all outputs byte-identical** (summary JSON identical excluding its timestamp)
**Input:** committed Phase 1 manifest `data/manifests/dataset_manifest.csv` (9,016 rows; digest `225a6024…` — verified unchanged by V2-11)
**Validation suite:** `tests/test_phase2_grouping.py` — V2-1…V2-11, **11/11 PASS**

Every number in this report was produced by an executed run and independently
re-derived by the validation suite. Nothing was hand-computed. **No image was
deleted, moved, renamed, relabeled, or modified; the raw dataset under
`dataset/raw/` is bit-identical to its Phase 1 state (V2-10, full re-hash).**

---

## 0. Executive summary

Phase 2 groups all 9,016 images into **496 `source_group_id` families** — the
filename-derived source-image/augmentation families Phase 4 splitting
requires — and cross-validates the Phase 1 near-duplicate candidates with an
independent structural-similarity method (SSIM). Headline findings:

1. **Grouping:** 496 groups = 496 filename source keys, one high-confidence
   original candidate per group (the 227×227 base images), 8,520 explicit
   augmented variants. Zero label conflicts; zero ungrouped files.
2. **Exact duplicates are fully absorbed by lineage:** all 228 md5-duplicate
   groups (464 files) sit *inside* single source keys — the 236 "extra" md5
   edges were redundant merges, so exact content adds **no** grouping beyond
   filename lineage (a useful corroboration, not a new structure).
3. **SSIM cross-validation:** 25,607/25,607 candidate pairs computed;
   mean SSIM declines monotonically with dHash distance (0.959 at d=0 →
   0.836 at d=7), but **1,139 pairs (4.4%) have SSIM < 0.60** — i.e. the
   dHash candidate set over-groups at high distances. Candidates are
   reported with their SSIM so later phases can threshold appropriately.
4. **New leakage-risk decomposition:** the 996 cross-split candidate pairs
   split into **19 same-key** pairs (part of the known `benign (36)` /
   `malignant (18)` leak) and **977 cross-key** pairs — near-identical
   content from *different* source families across the supplied split. This
   is a strictly additional risk signal that filename grouping alone does
   not capture, and Phase 4's leakage tests must cover it.
5. **Content-purity check:** filename-blind clustering (md5 + dHash ≤ 1)
   produces 147 mixed-key clusters, but **zero mix classes and zero span
   splits** — content evidence agrees with lineage grouping at the
   class/split level (72 distinct key-pairs, all same-class, same-split).
6. **No cleaning deletions were performed:** the roadmap's Phase 2 tasks are
   detection, grouping and documentation; a *removal policy* is a
   later-phase decision that will be applied to derived manifests, never to
   raw data.

---

## 1. Verified facts (executed)

| Property | Value |
|---|---|
| Images grouped | 9,016 / 9,016 (every Phase 1 manifest row carries exactly one `source_group_id`; V2-1) |
| Total groups | **496** (== distinct filename source keys; 1:1 mapping, V2-2) |
| Group-size histogram | size 14: 1 group (`benign (290)`); size 16: 285; size 21: 209; size 53: 1 (`malignant (1)` — the only depth-3 chain family) |
| Largest group | `malignant (1)` — 53 files, all val-side |
| Smallest group | `benign (290)` — 14 files |
| Multi-split groups | **2** (37 images): `benign (36)` (16 files: 12 train + 4 val) and `malignant (18)` (21 files: 10 train + 11 val) |
| Groups by split | train-only 446; val-only 48; both 2 |
| Groups by label | benign-only 286; malignant-only 210; **mixed-class 0** |
| Ungrouped / low-confidence files | **0** (every group is filename-lineage-backed; grammar parse rate was 100% in Phase 1) |
| Exact-duplicate groups / files | 228 / 464 (carried from Phase 1; V2-5 re-derived) |
| Near-duplicate candidate pairs | 25,607 (bit-identical to Phase 1; V2-7) |
| New dHash couples within groups missed by Phase 1 | **0** |
| SSIM values computed | 25,607 / 25,607 (0 errors; V2-7) |

## 2. Grouping methodology (documented for exit criterion 2)

**`source_group_id` = connected components over CONFIRMED-IDENTITY edges
only:**

1. **filename lineage** — same `source_key` parsed from the raw filename
   stem (the dataset's own naming scheme: `benign (100)` + chains from
   {`rotated1`, `rotated2`, `rotated32`, `sharpened`}); 8,520 merges;
2. **exact content identity** — identical md5 (byte-identical files,
   including Phase 1's byte-identical files under *different* chain labels);
   0 additional merges (236 redundant — see §3).

**Guards and policy:**

- **Label guard:** an edge whose endpoints have different class labels is
  never merged; it is counted and listed. Result: **0 label-conflict edges**
  (the dataset's filenames never contradict its directory labels).
- **Split policy:** grouping is **split-agnostic by design** — Phase 4
  group-level splitting depends on families spanning whatever splits the
  supplier used. Every resulting multi-split group is reported (§5).
- **Near-duplicate candidates are NOT identity edges.** dHash candidates and
  the 6,241 same-split cross-key candidate pairs never join groups; they are
  reported with SSIM values so later phases can decide policy on evidence.
- **Determinism:** union-find with deterministic edge order; group ids are
  md5 digests of sorted member md5s; all outputs sorted. Verified by three
  byte-identical runs (V2-3 recomputes the whole grouping from the Phase 1
  manifest independently and demands equality).

**Roles (original vs. augmented recovery, best-effort with documented
confidence):**

| Role | Count | Confidence | Rule (evidence) |
|---|---|---|---|
| `original_candidate` | 496 | high ×496, medium ×0 | `chain_free+base_dimension_227x227` — chain-free stem AND the Phase 1-verified base dimension; exactly one per group |
| `augmented_variant` | 8,520 | high ×8,520 | `filename_chain_explicit` — the dataset's own filenames state the augmentation |

**Limitation (per roadmap warning):** lineage is *filename-derivable*, and
the dimension corroboration is strong, but "original" here means *the
chain-free, base-resolution rendering in the dataset* — the true acquisition
original is not guaranteed to be present. This is stated as a limitation,
not guessed away.

## 3. Exact-duplicate reconciliation (V2-5)

| Quantity | Value | Note |
|---|---|---|
| md5-duplicate groups (Phase 1) | 228 | 464 files; sizes 2–4 |
| Groups containing duplicates | 192 of 496 | |
| md5 merge edges added in Phase 2 | **0** | every byte-identical pair already shares a source key |
| Redundant md5 merge attempts | **236** | = 464 dup files − 228 groups: exact arithmetic closure |
| Byte-identical pairs across splits | **0** | exhaustive re-verification during spot-check preparation; *(erratum 2026-09-18: this row originally said 1, citing a `benign (36)` cross-split md5-identical pair "from Phase 1" — that Phase 1 sentence was an editorial error; the executed data (`cross_split_exact_groups: 0`, overlap CSV `has_exact_duplicate_member: false`) was always correct)* |

Interpretation: exact content identity **corroborates** the filename lineage
(100% agreement) but contributes no additional grouping structure. **No
cross-split byte-identical files exist** (exhaustive md5 pass, re-verified);
the cross-split risk is *near-identity at candidate level* (§5), plus the
within-split byte-identical files under different chain labels (e.g.
`malignant (18)`, inside train).

## 4. Near-duplicate candidates + SSIM cross-validation (roadmap task 3)

Method: **global mean SSIM** over 64×64 grayscale re-decodations from raw
files; 11×11 gaussian window, σ=1.5, C1/C2 per Wang et al. 2004; deterministic
numpy-only implementation (validated: SSIM(x,x)=1.0 exactly; shifted/noisy
controls behave correctly). SSIM is **corroboration only** — it never adds,
removes, or changes any group.

**SSIM × dHash-distance correspondence (all 25,607 pairs):**

| dHash d | Pairs | SSIM mean | SSIM min | SSIM ≥ 0.90 | SSIM < 0.60 |
|---|---|---|---|---|---|
| 0 | 2,288 | 0.959 | 0.376 | 91.0% | 1.8% |
| 1 | 2,714 | 0.934 | 0.352 | 81.2% | 2.0% |
| 2 | 3,067 | 0.927 | 0.004 | 78.4% | 1.8% |
| 3 | 3,296 | 0.911 | −0.056 | 72.4% | 3.1% |
| 4 | 3,412 | 0.901 | −0.304 | 70.4% | 3.1% |
| 5 | 3,426 | 0.888 | −0.259 | 66.5% | 4.1% |
| 6 | 3,622 | 0.862 | −0.314 | 62.3% | 7.1% |
| 7 | 3,782 | 0.836 | −0.357 | 55.7% | 10.1% |

Overall SSIM distribution: exactly 1.0 ×273 (includes all 246 md5-identical
pairs — methods corroborate at the strict end); ≥ 0.90 ×17,845; 0.80–0.90
×4,078; 0.60–0.80 ×2,272; **< 0.60 ×1,139 (4.4%)**; min −0.357.

**Reading:** the two methods agree strongly on average (monotone decline),
but the candidate set is **heterogeneous at high dHash distances**: at d=7,
one in ten pairs is structurally dissimilar (SSIM < 0.60), and even at d=0 a
few dHash-identical pairs are structurally quite different images (dHash is
coarse for ultrasound texture). **Consequence for later phases:** any
deduplication or leakage policy must not treat "dHash ≤ 7" as uniform
evidence; the per-pair SSIM in `data/manifests/ssim_crosscheck.csv` is the
calibration instrument. Candidates were NOT deleted, merged, or excluded.

## 5. Leakage analysis (augmentation-family level)

**5.1 Source-key level (confirmed, carried from Phase 1):** 2 of 496
families span the supplied split — `benign (36)` (12 train / 4 val files;
near-identical cross-split variants, minimum dHash distance 2 — no
byte-identical cross-split files exist) and `malignant (18)` (10 train /
11 val). 37 images live in these two groups. The supplied train/val split
therefore leaks at the family level, exactly as flagged in Phase 1; the fix
belongs to Phase 4 (group-level splitting), and raw data remains untouched.

**5.2 Content level (new Phase 2 decomposition of the 996 cross-split
candidate pairs):**

| Candidate-pair combination | Pairs | Meaning |
|---|---|---|
| same-split + same-key | 18,370 | within-family candidates (expected; benign) |
| same-split + cross-key | 6,241 | distinct families with near-identical renders within a split |
| cross-split + **same-key** | **19** | part of the known 2-family leak |
| cross-split + **cross-key** | **977** | **new:** near-identical content from different families across the supplied split |The 977 cross-key cross-split pairs (19 distinct key-pairs, e.g. `benign (128)` [train] ↔ `benign (30)` [val] at dHash 0) are an **additional leakage-risk signal beyond source keys**: two different source keys can carry near-identical
content, and the supplier's split does not separate them. Phase 4's leakage
tests must include a cross-split near-duplicate check at the *pair* level in
addition to family-level separation (roadmap Phase 4 already lists both
"Source-image overlap" and "Near-duplicate overlap" checks — this finding
confirms they are both necessary).

**5.3 Content-purity cross-check (V2-9):** filename-blind clustering over
content evidence only (md5 identity + dHash ≤ 1 edges within split+class
buckets) yields 1,465 clusters; **147 mix distinct source keys, but 0 mix
classes and 0 span splits** (72 distinct key-pairs, all same-class,
same-split). Content and filename evidence agree at every level that
matters for splitting; the 147 clusters are the machine-checkable residue of
a cross-key phenomenon **related to but distinct from** §5.2's cross-split
pair signal (72 distinct key-pairs here vs 19 there; overlap = 1).

**5.4 Visual spot-check (roadmap validation check 2):** the roadmap asks for
a visual spot-check of grouped images. This audit environment cannot render
images for human viewing, so V2-9 executes the *machine-checkable core* of
that requirement (content-vs-lineage agreement above) instead, and the
limitation is stated openly. The 19 key-pair relationships of §5.2 and the
147 clusters (72 key-pairs)
of §5.3 are enumerated in the committed artifacts, so a human can review
exactly those candidates later; a true visual review remains **open** as a
recommended follow-up, not a completed task.

## 6. Cleaning decisions and derived-data status

- **No deletions, moves, renames, relabels, or repairs.** Phase 2's roadmap
  tasks are detection/grouping/documentation; a removal policy is not a
  Phase 2 task and would be a later-phase decision applied to *derived*
  manifests only.
- **Derived artifact created:** `data/manifests/augmentation_groups.csv`
  (9,016 rows; one per image; links to Phase 1 by exact `path`), carrying
  `source_group_id`, `group_size`, `group_splits`, `group_label`, `role`,
  `role_confidence`, `role_rule`, duplicate/near-duplicate group refs,
  `grouping_confidence`, `grouping_basis`, `source_key_status`
  (`inferred_from_filename_not_verified_identifier` — inherited verbatim),
  and `ungrouped_low_confidence` (0 rows True).
- **Phase 1 artifacts bit-stable:** manifest digest `225a6024…` re-verified
  (V2-11); Phase 1 CSVs/JSONs unmodified.
- The roadmap's phrase "cleaned manifest" is realized as this *grouping
  manifest* (the roadmap's own Files list names
  `data/manifests/augmentation_groups.csv`); no separate "cleaned" copy of
  the dataset exists or is needed at this phase.

## 7. Artifacts produced by Phase 2

| Path (D-1 mapping) | Content |
|---|---|
| `src/preprocessing/duplicate_detection.py` v1.0.0 | Grouping pipeline (confirmed-identity union-find, guards, roles, SSIM cross-check, deterministic outputs) |
| `src/preprocessing/perceptual_hash.py` v1.0.0 | dHash (Phase 1 definition, reused) + aHash helpers |
| `data/manifests/augmentation_groups.csv` | 9,016 rows; `source_group_id` grouping manifest for Phase 4 |
| `data/manifests/ssim_crosscheck.csv` | 25,607 rows; SSIM per candidate pair |
| `data/manifests/phase2_grouping_summary.json` | Machine-readable summary (counts, edges, roles, SSIM stats) |
| `tests/test_phase2_grouping.py` | V2-1…V2-11 validation suite (independent recomputation) |
| `reports/phase2_leakage_analysis.md` | This report |

## 8. Validation results (executed)

| Check | Result | Evidence |
|---|---|---|
| V2-1 coverage | **PASS** | 9,016/9,016 images carry exactly one `source_group_id` (roadmap validation check 1) |
| V2-2 group integrity | **PASS** | 496 groups; ids well-formed; `group_size` field == actual membership |
| V2-3 reproducibility | **PASS** | grouping recomputed from Phase 1 evidence — identical assignment |
| V2-4 image integrity | **PASS** | 450-file random re-decode (seed 20260918); dims/mode consistent with Phase 1 |
| V2-5 duplicate reconciliation | **PASS** | 228/464 carried; 8,520+0+236 edge closure; 464−228=236 |
| V2-6 roles | **PASS** | 496+8,520; exactly one high-confidence original per group; rule strings audited |
| V2-7 candidate preservation + SSIM | **PASS** | 25,607 pairs bit-identical; SSIM 100%, all in [−1,1] |
| V2-8 leakage accounting | **PASS** | 2 multi-split groups == `benign (36)`/`malignant (18)`; 37 images; 996 = 19+977 recomputed |
| V2-9 content purity | **PASS** | 1,465 clusters; 147 mixed-key; 0 mixed-class; 0 cross-split |
| V2-10 raw immutability | **PASS** | full md5 re-hash of all 9,016 raw images == Phase 1 manifest |
| V2-11 phase boundary | **PASS** | no ML imports; no checkpoints/processed dirs; Phase 1 artifacts digest-verified |

Reproducibility: **three full pipeline runs produced byte-identical CSVs**
and a summary JSON identical excluding its generation timestamp.

Failures encountered and fixed during development (all Phase 2 scope):
(a) SSIM helper initially ran per-row (`np.apply_along_axis`) and exceeded
the 10-minute cap — replaced with a vectorized separable filter (validated
against the same control images); (b) manifest width/height arrived as
strings — normalized to ints (root-caused via `original_high_confidence: 0`);
(c) the SSIM decoder initially resolved paths against the dataset root
instead of the Phase 1 image common root — fixed via
`audit.resolve_image_root`; (d) V2-8 originally asserted every candidate
pair is same-key — wrong check, corrected after the committed CSV's own
`same_source_key` column disproved it; the correction *surfaced* the 977
cross-key cross-split finding. No data was hidden; the audit trail records
each fix.

## 9. Limitations

1. **SSIM is an approximation:** 64×64 grayscale downsample, global mean
   over an interior crop; it calibrates candidates but is not a perceptual
   ground truth.
2. **Visual spot-check not humanly executed:** replaced by the V2-9
   content-purity test (§5.4); human review of the 72 key-pairs is a
   recommended follow-up.
3. **dHash coarseness:** dHash-identical pairs can be structurally distinct
   (min SSIM 0.376 at d=0); only md5 establishes identity.
4. **Roles are filename/dimension-derived:** "original" = chain-free
   base-resolution rendering; the true acquisition original is not
   guaranteed to exist in the dataset.
5. **No external ground truth** exists for any grouping; all evidence is
   internal to the dataset (filenames, bytes, pixels).
6. **No verified identifiers:** grouping remains at the filename-derived
   source-key level (NOT patient/study level), per the Phase 1 finding and
   owner sign-off.

### 9.1 Human visual spot-check record (completed 2026-09-18)

The roadmap's required human visual spot-check (PROGRESS.md Phase 2,
Validation Checks) was performed by the **project owner** on
2026-09-18 using `reports/phase2_spotcheck_contact_sheet.pdf`
(10 sections, 160 rows, 168 distinct raw images),
`reports/phase2_spotcheck_index.csv`, and
`reports/phase2_spotcheck_guide.md`. Verdicts, recorded exactly as given:

| Section | Content | Owner verdict |
|---|---|---|
| S1 | Augmentation/source-lineage families | Plausible |
| S2 | Exact-duplicate groups (md5-confirmed) | Plausible |
| S3 | Rotated/sharpened augmentation chains | Plausible |
| S4 | Largest family (53 members) | Plausible |
| S5 | Original-vs-augmented role decision | Plausible |
| S6 | Cross-split same-source-key candidate pairs (all 19) | Plausible |
| S7 | Cross-split cross-key relationships (best pair per key-pair) | Plausible as candidate relationships |
| S8 | dHash=0 cross-key examples | Plausible as candidate relationships |
| S9 | Low-SSIM candidates (dHash≥5, SSIM<0.60) | Plausible; visually supports retaining these as near-duplicate candidates rather than confirmed duplicates |
| S10 | The 2 families spanning the supplied train/val split (37 images) | Plausible |

**Owner factual clarification (2026-09-18):** `benign (36)` and
`malignant (18)` are genuine cross-split source-lineage families and must
NOT be described as cross-split exact-duplicate groups; the exhaustive
machine evidence is `cross_split_exact_groups = 0`; `benign (36)` has no
internal md5 duplicate and its nearest relevant cross-split dHash
relationship is not byte-identical. This matches §3/§5.1 and the errata
already applied to the Phase 1 report. No grouping-policy, methodology, or
scope change results from the review.

## 10. Implications for later phases (documented, NOT implemented)

1. **Phase 3 (bag definition):** the 496 groups are the natural bag seeds
   (source-image family → augmented instances); roles identify each bag's
   original candidate.
2. **Phase 4 (splitting):** group-level splitting is mandatory (2/496
   families already span the supplied split), and leakage tests must cover
   the **977 cross-key cross-split candidate pairs** (§5.2) in addition to
   family overlap — family separation alone does not neutralize the
   cross-key signal.
3. **Deduplication policy** (if any) is a later-phase decision; the
   committed per-pair SSIM and duplicate-group ids are the instruments.
4. **MRI/cross-modal status unchanged:** D-2 state C / T-5; no MRI files
   exist; cross-modal validation remains unavailable with this dataset.

---

*Report generated from executed Phase 2 output (duplicate_detection.py
v1.0.0, run three times; deterministic). No statistic in this document was
hand-computed; every number traces to `data/manifests/phase2_grouping_summary.json`,
the committed CSVs, or an executed validation check.*
