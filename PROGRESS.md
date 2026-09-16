# Project Progress
# AI DEVELOPMENT INSTRUCTIONS

This file is the authoritative development roadmap for this project.

Any AI coding assistant working on this repository MUST:

1. Read this file before making changes.
2. Work only on the currently authorized phase.
3. Never skip phases.
4. Never assume unverified dataset metadata.
5. Never fabricate medical data.
6. Never fabricate patient/study IDs.
7. Never fabricate MRI data or reports.
8. Never claim clinical validation.
9. Never use test data for tuning.
10. Never introduce data leakage.
11. Run relevant validation/tests before declaring a phase complete.
12. Update PROGRESS.md only when the phase exit criteria have genuinely been satisfied.
13. Stop after the requested phase and wait for further instructions.
14. Preserve existing repository structure unless a change is necessary.

**Project Title:** Predicting Breast Cancer from Ultrasound Images with Cross-Modal Validation using Dual Attention Multiple Instance Learning

**Problem Statement:** To create an AI-powered breast cancer prediction system that utilizes ultrasound imaging and Dual Attention Multiple Instance Learning to improve early diagnosis, enhance clinical decision-making, and provide personalized treatment recommendations. The system will cross-validate ultrasound prediction accuracy against MRI reports of the same patient to ensure diagnostic consistency and accuracy.

**Classification:** Research Prototype (NOT a clinically validated diagnostic system)

---

## Dataset

**Primary Dataset (mandatory, fixed):** "Ultrasound Breast Images for Breast Cancer"
**Source:** Kaggle — https://www.kaggle.com/datasets/vuppalaadithyasairam/ultrasound-breast-images-for-breast-cancer
**Author:** Vuppala Adithya Sairam
**Known description (per Kaggle listing, unverified until inspected):** Contains ultrasound images labeled benign/malignant; images have been augmented using rotation and sharpening.

> ⚠️ **Dataset Assumption Policy:** No claim about patient IDs, study IDs, MRI reports, lesion masks, or any metadata beyond raw images + benign/malignant class folders is to be treated as true until PHASE 1 dataset audit confirms it by direct inspection of the downloaded files. All such fields are UNVERIFIED until then.
>
> **Audit status (2026-09-16):** dataset downloaded and exhaustively audited — see `reports/phase1_dataset_audit.md` and `data/manifests/audit_summary.json`. Headline: 9,016 images; the listing's augmentation note is confirmed by filenames, but the dataset also ships a pre-existing train/val split, and it contains **no** metadata files, **no** patient/study/lesion IDs, **no** masks/annotations and **no** MRI data (D-2 state C → T-5).

---

## Progress Table

| Phase | Description | Status | Dependencies | Deliverable |
|-------|-------------|--------|---------------|-------------|
| 0 | Project Definition, Scope & Research Requirements | [x] Complete (2026-09-16 — deliverables approved by project owner; D-1/D-2 in effect) | None | Scope document `reports/phase0_scope.md` + `reports/research_questions.md` + `reports/phase0_literature_review.md` + `reports/phase0_decisions.md` |
| 1 | Dataset Acquisition & Complete Dataset Audit | [x] Complete (2026-09-16 — deliverables complete; manifests committed in `cec00cd`; audit report reviewed and **approved by the project owner**; validation 11/11 PASS) | Phase 0 | Dataset manifest + audit report (+ audit artifacts in `data/manifests/`, `tests/test_dataset_audit.py`) |
| 2 | Data Cleaning, Image Integrity & Augmentation Leakage Analysis | [ ] Not Started | Phase 1 | Duplicate/near-duplicate report, cleaned manifest |
| 3 | Data Organization & MIL Bag Definition | [ ] Not Started | Phase 2 | Bag definition spec + bag manifest |
| 4 | Patient/Study-Level Data Splitting & Leakage Prevention | [ ] Not Started | Phase 3 | Train/val/test split files + leakage test results |
| 5 | Ultrasound Image Preprocessing & Augmentation Pipeline | [ ] Not Started | Phase 4 | Preprocessing pipeline + config file |
| 6 | Baseline Deep Learning Models | [ ] Not Started | Phase 5 | Baseline model results |
| 7 | Multiple Instance Learning Pipeline | [ ] Not Started | Phase 5 | MIL pipeline implementation |
| 8 | Dual Attention MIL Model | [ ] Not Started | Phase 7 | Dual Attention MIL architecture + docs |
| 9 | Model Training, Optimization & Experiment Tracking | [ ] Not Started | Phase 6, 8 | Trained models + experiment logs |
| 10 | Evaluation, Error Analysis & Robustness Testing | [ ] Not Started | Phase 9 | Evaluation report |
| 11 | Explainability & Ultrasound Attention Visualization | [ ] Not Started | Phase 8, 9 | Attention visualizations + explainability report |
| 12 | MRI Cross-Modal Validation | [ ] Not Started | Phase 1, 10 | MRI feasibility report + (if feasible) concordance module |
| 13 | Clinical Decision Support / Risk Stratification | [ ] Not Started | Phase 10, 11, 12 | Decision-support output spec + module |
| 14 | Application Integration, Testing, Documentation & Final Demonstration | [ ] Not Started | All prior phases | End-to-end application + final report |

---

## Overall Progress

Phase 0: 100% — Complete (2026-09-16; deliverables approved by project owner)
Phase 1: 100% — Complete (2026-09-16; audit baseline approved by project owner)
Phase 2: 0%
Phase 3: 0%
Phase 4: 0%
Phase 5: 0%
Phase 6: 0%
Phase 7: 0%
Phase 8: 0%
Phase 9: 0%
Phase 10: 0%
Phase 11: 0%
Phase 12: 0%
Phase 13: 0%
Phase 14: 0%

**Overall: ~13%** (Phases 0–1 complete: 2/15 phases closed; no later phase started)

---

## Current Phase

Phase 1 — Dataset Acquisition & Complete Dataset Audit

**Status:** [x] Complete (2026-09-16) — deliverables reviewed and approved by the project owner; both exit criteria satisfied (manifests committed in `cec00cd`; owner sign-off recorded after the Phase 1 exit criteria). Phase 0 remains Complete (2026-09-16).

---

## Completed Milestones

- **2026-09-16 — Phase 0 Complete.** Deliverables: `reports/phase0_scope.md` (objectives, boundaries, assumptions A-1…A-17, Phase 1 gate V-1…V-11, limitations, safety boundaries, success criteria), `reports/research_questions.md` (RQ-1…RQ-10, falsifiable, unanswered), `reports/phase0_literature_review.md` (tracking framework; surveys ongoing), `reports/phase0_decisions.md` (D-1 repository mapping, D-2 cross-modal taxonomy — both approved by the project owner and in effect). Both Phase 0 validation checks passed; exit criterion satisfied by project-owner approval of the deliverables as a whole. No dataset downloaded; no code written; Phases 1–14 untouched.
- **2026-09-16 — Phase 1 Complete (owner-approved audit baseline).** Authorized by the owner; anonymous kagglehub acquisition of the authorized public dataset into gitignored `dataset/raw/` (D-1); exhaustive audit v1.1.2 (`src/data/audit.py`): 9,016 images (8,158 PNG / 858 JPG; 224×224 ×8,520, 227×227 ×496 = base images; RGB; 0 unreadable; 0 unparsed); labels directory-encoded (benign/malignant); **no** patient/study/lesion IDs, masks, annotations, metadata files or MRI data (D-2 state C → **T-5**); 496 filename-derived source keys (NOT verified identifiers); 8,520 files are explicit augmentation variants; 228 exact-duplicate groups (0 cross-split; but 1 byte-identical pair across splits via `benign (36)`); **2/496 source keys span the pre-existing train/val split** (`benign (36)`, `malignant (18)`) — a confirmed source-key-level leakage risk; 25,607 near-duplicate candidate pairs (candidates only; 996 cross-split pairs across 40 candidate groups). Manifests committed in `cec00cd`; report `reports/phase1_dataset_audit.md`; validation `tests/test_dataset_audit.py` V-1…V-11 **11/11 PASS**. **The project owner reviewed and approved the audit findings as the project's factual baseline (dataset accepted as-is, NOT as leakage-free); see the sign-off record in the Phase 1 section.** Phase 2–14 untouched; no model/MIL/training/split code created.

---

## Current Blockers

- **Phase 1 is Complete (2026-09-16)** — deliverables committed (`cec00cd`) and the audit report approved by the project owner; both exit criteria satisfied. Next: Phase 2 starts only on the owner's explicit instruction.
- Dataset facts now VERIFIED by the Phase 1 audit (2026-09-16): 9,016 images; directory-encoded labels (benign/malignant); explicit augmentation lineage in filenames; **no** patient/study/lesion identifiers; **no** masks/annotations; **no** MRI data (D-2 state C → T-5). Bag definition, split unit, preprocessing, model configuration, MRI functionality and decision-support features must be based on the audit report, not on the former A-1…A-17 assumptions.
- Repository layout: D-1 physically realized (2026-09-16) — `src/data/audit.py`, `data/manifests/`, `tests/`, gitignored `dataset/raw/`.
- **New leakage finding requiring later-phase handling:** the dataset's pre-existing train/val split shares 2 of 496 filename-derived source keys (`benign (36)`, `malignant (18)`), including a byte-identical image pair across splits; 228 exact-duplicate groups exist. Split correction is later-phase work (Phase 2/4); raw data untouched.
- Definition of "cross-modal validation": resolved and closed by owner-approved decision **D-2** (`reports/phase0_decisions.md` §2; T-1…T-5 taxonomy, evidence states A–E; expected primary-dataset outcome T-5 pending V-4).
- Still open: Q-3 (empty scaffold files), Q-5 (README title), Q-6 (risk-estimation scope) — none blocks Phase 1; Q-7/Q-8 deliberately deferred until Phase 12/14 relevance.
- RESOLVED (2026-09-16) — Kaggle dataset structure inspected: pre-existing train/val split; classes benign/malignant; 9,016 images (8,158 PNG / 858 JPG).
- RESOLVED (2026-09-16) — no reliable patient/study/lesion identifiers exist anywhere in the dataset; the finest defensible grouping unit is the filename-derived **source key** (496 keys; explicitly NOT a verified patient ID).
- RESOLVED (2026-09-16) — augmentation lineage is identifiable: filenames encode rotated1/rotated2/rotated32/sharpened chains (8,520 augmented variants of 496 base images); 228 exact md5-duplicate groups; 25,607 near-duplicate candidate pairs (candidates only).
- Need to determine how MIL bags can legitimately be constructed.
- Need to determine availability of paired MRI data.
- Need to determine whether an external MRI dataset/source is required.
- Need to verify whether treatment-support functionality is appropriate for the available data.

---

## Non-Negotiable Project Rules

1. Never fabricate medical data.
2. Never fabricate MRI reports.
3. Never fabricate patient IDs.
4. Never claim MRI validation without actual paired MRI evidence.
5. Never allow augmented versions of the same source image across different data splits.
6. Never use the final test set for model tuning.
7. Never report accuracy alone.
8. Always report clinically relevant metrics such as sensitivity and specificity.
9. Never claim that model attention is automatically a clinical explanation.
10. Never present the system as a replacement for a radiologist or oncologist.
11. Never provide autonomous treatment prescriptions.
12. Clearly distinguish research results from clinical validation.
13. Record all assumptions.
14. Make experiments reproducible.
15. Never hide dataset limitations.

---

## Potential Novelty — Requires Literature Verification

The following are **candidate** research contributions. None are to be claimed as novel, publishable, or state-of-the-art until a structured literature review (task defined below) is completed.

1. Dual Attention MIL applied specifically to breast ultrasound classification.
2. Instance-level evidence aggregation rather than simple image classification.
3. Attention-guided ultrasound evidence visualization.
4. Cross-modal validation against MRI findings.
5. Ultrasound/MRI diagnostic concordance analysis.
6. Patient/study-level risk estimation.
7. Explainable clinical decision support.

**Literature Review Task (tracked under Phase 0 tasks):**

> Tracking begun 2026-09-16: `reports/phase0_literature_review.md` (topic dossiers, search protocol, evidence schema, unfilled novelty-positioning matrix). **No search has been performed and no prior work has been assessed**, so every box below remains unticked and every item above remains "to be verified". Absence of a hit in a future search is not evidence of novelty, and no item may be presented as novel or state-of-the-art.

- [ ] Survey CNN-based breast ultrasound classifiers.
- [ ] Survey transfer-learning approaches for breast ultrasound.
- [ ] Survey MIL approaches in medical imaging.
- [ ] Survey attention-based MIL approaches (e.g., ABMIL-style architectures) and identify what "dual attention" already exists in prior work.
- [ ] Survey multimodal breast imaging systems (ultrasound + MRI, ultrasound + mammography).
- [ ] Survey existing ultrasound + MRI concordance/consistency systems.
- [ ] Survey explainable breast cancer AI systems (Grad-CAM, attention maps, saliency methods).
- [ ] Produce a novelty-positioning summary comparing this project's design choices against the above.

---

## Repository Structure

```
project/
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── manifests/
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   ├── mil/
│   ├── training/
│   ├── evaluation/
│   ├── explainability/
│   ├── mri/
│   ├── clinical_support/
│   └── utils/
│
├── configs/
├── experiments/
├── checkpoints/
├── reports/
├── tests/
├── app/
└── PROGRESS.md
```

> Note: Adapt to any pre-existing repository structure. Do not overwrite or reorganize existing files unnecessarily.

---

## Testing Requirements (Global — tracked per relevant phase)

Mandatory automated tests across the project:

- [ ] Dataset discovery test
- [ ] Image loading test
- [ ] Image validation test
- [ ] Duplicate detection test
- [ ] Augmentation leakage detection test
- [ ] Dataset splitting test
- [ ] Patient/study leakage test
- [ ] MIL bag creation test
- [ ] Variable bag size handling test
- [ ] Attention output dimension test
- [ ] Model forward-pass test
- [ ] Training pipeline test
- [ ] Evaluation metrics correctness test
- [ ] MRI matching test (if MRI data available)
- [ ] MRI report extraction test (if MRI data available)
- [ ] Cross-modal validation test (if MRI data available)
- [ ] End-to-end inference test

> Leakage tests (patient/study/source-image/duplicate/near-duplicate/augmentation-family) are **mandatory and non-optional**, given the confirmed use of rotation/sharpening augmentation in the source dataset.

---

# Phase 0 — Project Definition, Scope & Research Requirements

**Status:** [x] **Complete** (2026-09-16) — deliverables drafted, both validation checks passed, all tasks done, and the **project owner approved the Phase 0 deliverables as a whole** on 2026-09-16 (decisions D-1/D-2 approved separately the same day). See the Phase 0 Progress Note and the Exit Criteria record below. Literature surveys remain deliberately open as ongoing work; Phase 1 may begin only on the owner's explicit instruction.

### Phase 0 Progress Note (recorded 2026-09-16)

**Deliverables drafted:**
- `reports/phase0_scope.md` — objectives (technical + research), scope boundaries, intended input/output, prediction targets, research-prototype vs. clinically-validated distinction, assumptions (A-1…A-17, all UNVERIFIED), Phase 1 verification requirements (V-1…V-11), known limitations, MRI uncertainty, identifier uncertainty, bag-construction uncertainty, medical AI safety boundaries, non-negotiable rules, technical/project success criteria, completion criteria, literature requirements, and open decisions Q-1…Q-8.
- `reports/research_questions.md` — 10 falsifiable research questions (RQ-1…RQ-10) with null hypotheses, refutation conditions, required evidence and threats to validity. **No question is answered.**
- `reports/phase0_literature_review.md` — literature-review tracking framework (topic dossiers, search protocol, evidence schema, unfilled novelty-positioning matrix). **No literature search has been performed.**

**Validation checks:**
- Scope document contains no clinical deployment claims — PASS `reports/phase0_scope.md` §3, §4, §12, §16.
- Research questions are falsifiable/testable — PASS `reports/research_questions.md` §3.

**Exit criteria: SATISFIED (2026-09-16).** "Documented and approved by project team/advisor" — documented via `reports/phase0_scope.md`, `reports/research_questions.md`, `reports/phase0_literature_review.md`, `reports/phase0_decisions.md`; approved by the **project owner** (approval authority confirmed sufficient — no external advisor/reviewer approval required; see `reports/phase0_decisions.md` §4). Both validation checks passed; all 13 tasks complete. **Phase 0 is Complete.** Per the roadmap's stop-after-phase rule and the owner's instruction, Phase 1 has NOT been started and awaits the owner's explicit prompt.

**Phase 1 gate:** every dataset assumption recorded in Phase 0 (A-1…A-17) is UNVERIFIED. Phase 1 must verify V-1…V-11 **before** any bag definition, split unit, preprocessing, model configuration, MRI functionality or decision-support feature is finalised. Where a verification fails, the dependent capability must be documented as unavailable — never approximated, simulated or assumed.

**Decision status (updated 2026-09-16):** Q-1 (D-1) and Q-4 (D-2) — **approved by the project owner**; in effect. Q-2 — resolved: project-owner approval is sufficient for project-level technical decisions, and the owner approved the Phase 0 deliverables as a whole on 2026-09-16. Q-7 and Q-8 — **deliberately deferred by owner instruction** until synthetic MRI or an external MRI dataset actually becomes relevant (expected Phase 12/14). Q-3, Q-5, Q-6 — still open. See `reports/phase0_scope.md` §24 and `reports/phase0_decisions.md` §4.

**Blocker-resolution record (2026-09-16):** Q-1 and Q-4 have been investigated and resolved at specification level in `reports/phase0_decisions.md` — D-1 (repository structure mapping; raw data → gitignored `dataset/raw/`, research code → `src/`, committed manifests → `data/manifests/`, ML model code → existing `model/`, app/UI → existing `backend/`+`frontend/`) and D-2 (cross-modal validation defined per evidence state A–E under the T-1…T-5 taxonomy; expected primary-dataset outcome is T-5, pending Phase 1 verification V-4). **Both decisions were approved by the project owner on 2026-09-16 and are now in effect as the project's working decisions** (approval recorded in `reports/phase0_decisions.md` §4). No directory, file, manifest or dataset change has been made for either decision yet — D-1 takes physical effect when Phase 1 creates files under the approved mapping, and all roadmap paths in this document remain as written (D-1 is a mapping, not a rename).

**Phase 0 closure record (2026-09-16):** with D-1/D-2 approved, both Phase 0 validation checks passed, all 13 Phase 0 tasks complete, and the **project owner's approval of the Phase 0 deliverables as a whole** received the same day, the exit criterion ("documented and approved by project team/advisor") is satisfied with the project owner as the approval authority. **Phase 0 is Complete.** Literature surveys remain deliberately open as ongoing work. Per the roadmap's stop-after-phase rule and the owner's instruction, **Phase 1 has not been started** and awaits the owner's explicit prompt.

### Objective
Establish a precise, honest, and bounded definition of what this project will and will not do, before any code or data work begins.

### Why This Phase Matters
Medical AI projects fail credibility when scope is vague or overstated. Defining hard boundaries now prevents scope creep into unsupported clinical claims later (especially around MRI validation and treatment recommendations).

### Prerequisites
None.

### Tasks
- [x] Write formal objectives (technical + research). → Done: `reports/phase0_scope.md` §5 (TO-1…TO-13), §6 (RO-1…RO-7).
- [x] Write research questions (e.g., "Does dual attention improve bag-level classification over single-attention MIL and non-MIL baselines on this dataset?"). → Done: `reports/research_questions.md` RQ-1…RQ-10; operationalised as RQ-1/RQ-2. No question answered.
- [x] Define input: ultrasound image(s), optionally grouped by patient/study. → Done: `reports/phase0_scope.md` §7 (grouping key UNVERIFIED, see §18).
- [x] Define output: benign/malignant prediction, confidence score, attention-based evidence, optional risk category, optional MRI concordance status. → Done: `reports/phase0_scope.md` §8 (risk category and concordance status are conditional outputs).
- [x] Define prediction target(s): image-level, and bag-level (patient/study) where dataset structure supports it. → Done: `reports/phase0_scope.md` §9 (bag-level conditional on Phase 1/2/3; patient/study-level may be impossible).
- [x] Define scope boundaries: research prototype only, not a diagnostic device, not a treatment engine. → Done: `reports/phase0_scope.md` §11, §12.
- [x] Document assumptions (to be revisited/verified in Phase 1): e.g., dataset is image-only until proven otherwise. → Done: `reports/phase0_scope.md` §13 (A-1…A-17, all UNVERIFIED), §14 (V-1…V-11).
- [x] Document known limitations: single public dataset, likely no MRI pairing, likely no patient IDs, augmented images already embedded in the dataset. → Done: `reports/phase0_scope.md` §15.
- [x] Define medical AI safety boundaries (see Non-Negotiable Rules above). → Done: `reports/phase0_scope.md` §16; rules reproduced in §20.
- [x] Define success criteria (technical): e.g., MIL model outperforms non-MIL baseline on sensitivity/specificity/AUC on a leakage-free test set. → Done: `reports/phase0_scope.md` §21 (TS-1…TS-14; note TS-9 makes a null result acceptable — the criterion is honest comparison, not beating baselines).
- [x] Define success criteria (project): completed phases, reproducibility, documented limitations, working demo. → Done: `reports/phase0_scope.md` §22.
- [x] Explicitly write the "Research Prototype vs. Clinically Validated System" distinction into the report/README. → Done: `reports/phase0_scope.md` §4 (README alignment itself deferred to Phase 14 documentation, per Q-5).
- [x] Begin literature review tracking (see Novelty section tasks above). → Tracking begun 2026-09-16 in `reports/phase0_literature_review.md`; the surveys themselves remain NOT STARTED and continue as ongoing work feeding Phases 6–12.

### Files / Modules
- `reports/phase0_scope.md`
- `reports/research_questions.md`
- `reports/phase0_literature_review.md` (Phase 0 literature-review tracking)

### Expected Outputs
A scope document that clearly separates prototype claims from clinical claims, plus an initial research question list.

### Validation Checks
- [x] Scope document reviewed and does not contain clinical deployment claims. → PASS (2026-09-16): `reports/phase0_scope.md` §3, §4, §12, §16, §22 contain no clinical deployment claim; every capability is conditional and any capability whose precondition is unverified is marked unavailable rather than promised.
- [x] Research questions are falsifiable/testable. → PASS (2026-09-16): `reports/research_questions.md` §3 gives every question a null hypothesis, a refutation condition and named observable evidence; RQ-4, RQ-8(d), RQ-9 and RQ-10 are explicitly flagged as partially answerable or unanswerable at the current data stage.

### Exit Criteria
- [x] Scope, objectives, and boundaries documented and approved by project team/advisor. → **SATISFIED 2026-09-16.** Documented: `reports/phase0_scope.md` (25 sections), `reports/research_questions.md`, `reports/phase0_literature_review.md`, `reports/phase0_decisions.md`. Approved: the **project owner approved the Phase 0 deliverables as a whole on 2026-09-16** (owner approval confirmed sufficient per `reports/phase0_decisions.md` §4 — no external advisor/reviewer approval required for project-level technical decisions). Both validation checks above had already passed. Phase 0 is therefore **Complete**; Phase 1 may begin only on the owner's explicit instruction.

### Potential Issues / Risks
- Scope may need revision after Phase 1 dataset audit reveals actual data limitations (e.g., no patient IDs at all).

---

# Phase 1 — Dataset Acquisition & Complete Dataset Audit

**Status:** [x] **Complete** (2026-09-16) — deliverables committed (`cec00cd`), validation suite 11/11 PASS, both exit criteria satisfied, and the **project owner reviewed and approved the audit findings as the project's factual baseline** (sign-off record below). The dataset is accepted as audited — NOT as a leakage-free dataset.

### Objective
Download the exact Kaggle dataset and produce a factual, verified, machine-readable account of its real structure — with zero assumptions.

### Why This Phase Matters
All downstream decisions (bag definition, splitting strategy, MRI feasibility) depend entirely on what this dataset actually contains, not on what a project brief hopes it contains.

### Prerequisites
Phase 0 complete.

### Tasks
- [x] Download dataset from https://www.kaggle.com/datasets/vuppalaadithyasairam/ultrasound-breast-images-for-breast-cancer into `data/raw/`. → Done: anonymous kagglehub download of the **public** dataset into gitignored `dataset/raw/` (D-1 mapping of `data/raw/`); no credentials used or required; no access controls bypassed.
- [x] Record dataset version/download date/checksum information. → Done: `data/manifests/dataset_source.json` (handle, URL, version dir, acquisition note, manifest digest `225a6024…`); the only local timestamp is kagglehub's `1.complete` marker mtime, recorded explicitly as filesystem evidence, not a Kaggle-published date.
- [x] Enumerate complete directory structure (all folders/subfolders). → Done: `.../versions/1/ultrasound breast classification/{train,val}/{benign,malignant}`; no other content (report §2).
- [x] Count total files, broken down by class folder (benign/malignant, or whatever classes actually exist). → Done: 9,016 images — train/benign 4,074; train/malignant 4,042; val/benign 500; val/malignant 400.
- [x] Identify image formats present (jpg/png/etc.). → Done: PNG 8,158, JPEG 858; no other formats; no non-image files inside the image root.
- [x] Extract image dimensions and color channel info (grayscale vs RGB) for a full or sampled pass. → Done: **exhaustive** (no sampling needed): 224×224 ×8,520; 227×227 ×496 (exactly the 496 base images); RGB, 3 channels, all files.
- [x] Record exact class names as found (do not assume "benign"/"malignant" spelling/casing without checking). → Done: exactly `benign` and `malignant`, directory-encoded.
- [x] Compute class distribution (benign vs malignant vs any other class found). → Done: counts above; train near-balanced (4,074/4,042), val 500/400; no other classes exist.
- [x] Analyze filename patterns for embedded metadata (IDs, augmentation tags, sequence numbers). → Done: grammar `"<class> (<idx>)[-<op>…]"`, ops ∈ {rotated1, rotated2, rotated32, sharpened}; 100% parse; 496 source keys; **no IDs of any kind**; source keys labelled `inferred_from_filename_not_verified_identifier`.
- [x] Check for duplicate files (exact byte-level duplicates). → Done: exhaustive md5+sha256 — 8,780 unique md5; 228 exact-duplicate groups (464 files); 0 groups crossing the pre-existing split; md5-dup pairs have dHash distance 0 (cross-check).
- [x] Check for corrupted/unreadable image files. → Done: **0 unreadable** (exhaustive decode of all 9,016).
- [x] Search for any accompanying metadata files (CSV, JSON, XML, README, license file) within the dataset download. → Done: **none found**.
- [x] Determine whether patient IDs are present anywhere (explicitly search; do not assume absence or presence). → Done: **confirmed absent** (report §5.1 explicit record).
- [x] Determine whether study IDs are present. → Done: **confirmed absent**.
- [x] Determine whether lesion IDs are present. → Done: **confirmed absent**.
- [x] Determine whether masks/annotations are present. → Done: **confirmed absent** (and **MRI-related data confirmed absent** → D-2 state C / T-5).
- [x] Document explicitly what information is MISSING (this list is expected to be long given the dataset's known scope). → Done: report §5.1 (explicit Confirmed-Present / Confirmed-Absent table), §10 (assumption outcomes), §11 (limitations).
- [x] Build a machine-readable dataset manifest (CSV/Parquet/JSON) with one row per image recording: filepath, class label, format, dimensions, channels, file hash, and any recoverable naming-pattern metadata. → Done: `data/manifests/dataset_manifest.csv` — 9,016 rows (100% coverage), 22 fields incl. md5+sha256, dims, source_key, chain, lineage_class, source_key_status, duplicate/near-dup groups; byte-identical across two independent runs.

### Files / Modules
- `src/data/audit.py` (v1.1.2 — created; v1.1.2 = pre-commit privacy fix recording dataset root repo-relative in committed artifacts)
- `data/manifests/dataset_manifest.csv` (9,016 rows — created; **committed in `cec00cd`**)
- `reports/phase1_dataset_audit.md` (created)
- `data/manifests/near_duplicate_candidates.csv`, `data/manifests/source_key_overlap.csv`, `data/manifests/audit_summary.json`, `data/manifests/dataset_source.json` (created — additional audit artifacts)
- `tests/test_dataset_audit.py` (created — V-1…V-11 validation suite, standalone or pytest)

### Expected Outputs
- Complete, factual dataset audit report.
- Dataset manifest covering 100% of discovered image files.

### Validation Checks
- [x] Manifest row count equals actual file count on disk. → Verified: 9,016 == 9,016 (V-2; also path-set equality).
- [x] All class labels in manifest match actual folder/label structure. → Verified: label = directory; recomputed from manifest in V-9.
- [x] Audit report contains an explicit "Confirmed Present" and "Confirmed Absent / Unknown" section for patient IDs, study IDs, lesion IDs, masks, and MRI-related data. → Verified: report §5.1 table.
- Additional (beyond roadmap minimum): full suite `tests/test_dataset_audit.py` — V-1…V-11, **11/11 PASS** (report §14).

### Exit Criteria
- [x] Audit report finalized and reviewed. → **SATISFIED 2026-09-16**: report finalized and validated (11/11 PASS), then reviewed and **approved by the project owner** the same day (sign-off record below).
- [x] Manifest committed to `data/manifests/`. → **SATISFIED 2026-09-16**: all five manifests + report + audit code + tests committed in `cec00cd` ("Phase 1: dataset audit and leakage analysis").

### Project-Owner Sign-off (2026-09-16) — Phase 1 audit baseline
The project owner reviewed `reports/phase1_dataset_audit.md` and approved the Phase 1 dataset acquisition and audit findings **as the factual baseline for the project**, explicitly recording:

- The dataset is accepted as the audited research starting point, **NOT as a leakage-free dataset**.
- The two cross-split inferred source-key overlaps remain documented: `benign (36)` and `malignant (18)`.
- The 996 cross-split near-duplicate candidate pairs across 40 candidate groups remain documented (candidates, not confirmed duplicates).
- No verified patient/study/lesion identifiers were found; filename-derived source keys must **never** be represented as verified patient identifiers.
- MRI validation remains unavailable under **D-2 State C / T-5**.
- No raw images are deleted, moved, relabeled, or silently repaired as part of Phase 1.
- Any future leakage correction or split redesign must occur in its designated later phase and **preserve the original audit evidence** (manifests, report, raw data).

### Potential Issues / Risks
- Kaggle dataset structure may differ from its description; do not trust the Kaggle description text over direct inspection.
- Large image counts may require sampling for dimension/channel checks — document sampling methodology if used.

---

# Phase 2 — Data Cleaning, Image Integrity & Augmentation Leakage Analysis

**Status:** [ ] Not Started

### Objective
Detect and document duplicate, near-duplicate, and augmented (rotated/sharpened) relationships between images, since the dataset is known to contain rotation- and sharpening-based augmentation.

### Why This Phase Matters
If augmented copies of the same source image land in both train and test sets, evaluation metrics become invalid due to data leakage. This is a critical integrity risk given the dataset's documented augmentation.

### Prerequisites
Phase 1 complete, manifest available.

### Tasks
- [ ] Implement exact-duplicate detection (hash-based) across the full dataset.
- [ ] Implement perceptual hashing (e.g., pHash/aHash/dHash) to detect near-duplicates and rotated/sharpened variants.
- [ ] Implement additional similarity check (e.g., structural similarity or embedding-based similarity) to cross-validate perceptual hash groupings, where computationally feasible.
- [ ] Analyze filenames for augmentation-related naming patterns (if any exist).
- [ ] Cluster images into "duplicate groups" and "near-duplicate/augmentation-family groups."
- [ ] Attempt to recover original vs. augmented relationships within each group (best-effort; document confidence level).
- [ ] Document counts: total images, estimated unique source images, estimated augmented images, duplicate groups, near-duplicate groups.
- [ ] Flag any images that cannot be confidently grouped (log as "ungrouped / low-confidence").
- [ ] Update manifest with a `source_group_id` (or equivalent) field representing the augmentation-family/source-image grouping, to be used in Phase 4 splitting.

### Files / Modules
- `src/preprocessing/duplicate_detection.py`
- `src/preprocessing/perceptual_hash.py`
- `data/manifests/augmentation_groups.csv`
- `reports/phase2_leakage_analysis.md`

### Expected Outputs
- Augmentation/duplicate group manifest linked to the Phase 1 manifest via image path or ID.
- Leakage analysis report with quantified group statistics.

### Validation Checks
- [ ] Every image in the manifest has an assigned `source_group_id` (even if it is a singleton group).
- [ ] Spot-check a random sample of grouped images visually to confirm grouping plausibility.

### Exit Criteria
- [ ] Grouping manifest finalized and validated.
- [ ] Report documents grouping confidence level and methodology.

### Potential Issues / Risks
- Perceptual hashing may over- or under-group images; manual/visual spot-checking is required.
- True original-vs-augmented lineage may be unrecoverable — this must be stated as a limitation rather than guessed.

---

# Phase 3 — Data Organization & MIL Bag Definition

**Status:** [ ] Not Started

### Objective
Define, based only on verified Phase 1/2 findings, what constitutes a "bag" and an "instance" for the MIL formulation.

### Why This Phase Matters
MIL is meaningless without a scientifically defensible bag definition. This must follow from actual dataset structure, not from a generic template.

### Prerequisites
Phase 1 and Phase 2 complete.

### Tasks
- [ ] Review Phase 1 audit findings on available identifiers (patient/study/lesion) and Phase 2 grouping findings (source-image/augmentation families).
- [ ] Evaluate candidate bag definitions:
  - [ ] Patient → images (only if patient IDs confirmed present)
  - [ ] Study → images (only if study IDs confirmed present)
  - [ ] Lesion → image/patch instances (only if lesion IDs/annotations confirmed present)
  - [ ] Source-image-group → augmented instance variants (fallback if no higher-level identifiers exist)
  - [ ] Single image → tiled patches as instances (fallback/alternative if grouping is unreliable)
- [ ] Select and justify the bag definition actually supported by the data (expect this to likely be the source-image-group or image-as-bag-of-patches definition unless Phase 1 proves otherwise).
- [ ] Explicitly document: BAG = ?, INSTANCE = ?, LABEL = ?
- [ ] Diagram the chosen hierarchy (bag → instances → features → attention → prediction).
- [ ] If no reliable patient/study grouping exists, explicitly document this limitation and record the decision to use the highest valid available grouping level instead of fabricating identifiers.
- [ ] Build the bag manifest (bag ID, instance list, bag label, instance count).

### Files / Modules
- `src/mil/bag_definition.py`
- `data/manifests/bag_manifest.csv`
- `reports/phase3_bag_definition.md`

### Expected Outputs
- Documented, justified bag/instance/label definition.
- Bag manifest usable by downstream splitting and training phases.

### Validation Checks
- [ ] Every instance in the bag manifest traces back to a valid entry in the Phase 1/2 manifests.
- [ ] Every bag has a well-defined, non-ambiguous label.
- [ ] Bag sizes (instance counts) are documented with summary statistics (min/max/mean/median).

### Exit Criteria
- [ ] Bag definition formally documented and justified against actual dataset evidence.

### Potential Issues / Risks
- Temptation to assume patient-level bags exists even without IDs — must be actively resisted per project rules.
- Very small or very large bag sizes may require special handling in Phase 7/8.

---

# Phase 4 — Patient/Study-Level Data Splitting & Leakage Prevention

**Status:** [ ] Not Started

### Objective
Create train/validation/test splits at the highest defensible grouping level, guaranteeing no leakage across splits.

### Why This Phase Matters
Even without patient IDs, failing to split at the source-image/augmentation-group level would let near-identical images appear in both training and test data, inflating performance metrics artificially.

### Prerequisites
Phase 3 complete (bag manifest available).

### Tasks
- [ ] Determine the split unit: patient (if available) → study (if available) → source-image group (fallback, expected default given known dataset constraints).
- [ ] Implement stratified splitting by class label at the chosen split-unit level.
- [ ] Generate train/validation/test split files (lists of bag IDs / group IDs per split).
- [ ] Implement automated leakage tests:
  - [ ] Patient overlap check (if applicable)
  - [ ] Study overlap check (if applicable)
  - [ ] Source-image overlap check
  - [ ] Duplicate overlap check
  - [ ] Near-duplicate overlap check
  - [ ] Augmentation-family overlap check
- [ ] Run all leakage tests and confirm zero overlaps across splits.
- [ ] Freeze and lock the test set; document the freeze date and hash of the test split file.
- [ ] Document class balance within each split.

### Files / Modules
- `src/data/splitting.py`
- `tests/test_leakage.py`
- `data/manifests/train_split.csv`, `data/manifests/val_split.csv`, `data/manifests/test_split.csv`
- `reports/phase4_split_report.md`

### Expected Outputs
- Verified, leakage-free train/val/test splits.
- Automated test suite proving zero overlap.

### Validation Checks
- [ ] All leakage tests pass with zero overlaps.
- [ ] Class distribution per split documented and reasonably balanced or explicitly noted as imbalanced.

### Exit Criteria
- [ ] Test set frozen, hashed, and marked as untouched until Phase 10.

### Potential Issues / Risks
- Small dataset size may make patient/study-level (or group-level) splitting reduce test set size significantly — document trade-off.
- Class imbalance may require stratification adjustments.

---

# Phase 5 — Ultrasound Image Preprocessing & Augmentation Pipeline

**Status:** [ ] Not Started

### Objective
Build a reproducible, justified preprocessing pipeline applied consistently across the pipeline, with new augmentation (if any) applied only to training data.

### Why This Phase Matters
Since the dataset already contains embedded augmentation, this phase must be careful not to compound leakage risk with further indiscriminate augmentation, and must justify every preprocessing step rather than applying a generic recipe.

### Prerequisites
Phase 4 complete (splits frozen).

### Tasks
- [ ] Implement image loading utilities with consistent format handling.
- [ ] Implement resizing strategy (document target resolution and justification).
- [ ] Implement normalization (document statistics used — dataset-specific vs. pretrained-backbone-specific).
- [ ] Evaluate and document need for intensity/noise handling specific to ultrasound (e.g., speckle noise).
- [ ] Evaluate optional despeckling filters; justify inclusion/exclusion experimentally.
- [ ] Evaluate cropping strategy; only crop to lesion regions if annotations were confirmed present in Phase 1.
- [ ] Define any additional training-only augmentation (e.g., flips) — must not duplicate the dataset's existing rotation/sharpening augmentation in a way that risks further leakage confusion.
- [ ] Ensure augmentation is applied only within the training split, never validation/test.
- [ ] Build and save a preprocessing configuration file (parameters, versioned).
- [ ] Run preprocessing pipeline on all splits and cache processed outputs to `data/processed/`.

### Files / Modules
- `src/preprocessing/pipeline.py`
- `configs/preprocessing_config.yaml`
- `reports/phase5_preprocessing_justification.md`

### Expected Outputs
- Reproducible preprocessing pipeline with versioned configuration.
- Processed image cache for all splits.

### Validation Checks
- [ ] Visual sample review of preprocessed images per class.
- [ ] Confirm augmentation is applied exclusively to training split (automated check).

### Exit Criteria
- [ ] Preprocessing config finalized and preprocessing applied consistently across all splits.

### Potential Issues / Risks
- Over-aggressive preprocessing (e.g., excessive despeckling) may remove diagnostically relevant texture information — must be validated experimentally, not assumed beneficial.

---

# Phase 6 — Baseline Deep Learning Models

**Status:** [ ] Not Started

### Objective
Establish non-MIL and simple-MIL baselines to provide a fair comparison point for the Dual Attention MIL model.

### Why This Phase Matters
Without baselines, any claimed improvement from Dual Attention MIL is unsubstantiated. This directly supports the novelty-verification requirement.

### Prerequisites
Phase 5 complete.

### Tasks
- [ ] Implement a simple CNN classifier (image-level).
- [ ] Implement a transfer-learning classifier using a pretrained backbone (ResNet/EfficientNet/DenseNet — choose based on available compute; document choice).
- [ ] Implement a standard image-level classifier baseline (majority-vote or single-instance aggregation at bag level, for comparability with MIL results).
- [ ] Implement a standard (single-attention or mean-pooling) MIL baseline for direct comparison with Dual Attention MIL in Phase 8.
- [ ] Train all baselines using the same splits and preprocessing as will be used for the main model.
- [ ] Record baseline metrics (accuracy, sensitivity, specificity, precision, recall, F1, ROC-AUC, PR-AUC).
- [ ] Document compute budget and training time per baseline.

### Files / Modules
- `src/models/baseline_cnn.py`
- `src/models/transfer_learning.py`
- `src/mil/standard_mil_baseline.py`
- `experiments/baseline_results/`
- `reports/phase6_baseline_results.md`

### Expected Outputs
- Baseline performance table to be referenced in Phase 10 comparative evaluation.

### Validation Checks
- [ ] All baselines trained and evaluated on identical splits.
- [ ] Metrics logged in a consistent, comparable format.

### Exit Criteria
- [ ] Baseline results documented and stored for later comparison.

### Potential Issues / Risks
- Limited compute may restrict backbone choice — document any resource-driven compromises.

---

# Phase 7 — Multiple Instance Learning Pipeline

**Status:** [ ] Not Started

### Objective
Implement the general MIL pipeline infrastructure (feature extraction → instance embeddings → aggregation → classification) that Phase 8's Dual Attention model will build upon.

### Why This Phase Matters
A robust, correctly implemented MIL infrastructure — especially variable bag-size handling — is a prerequisite for a valid Dual Attention MIL implementation.

### Prerequisites
Phase 3 (bag definition) and Phase 6 (baseline feature extractor candidates) complete.

### Tasks
- [ ] Implement instance generation from bags per the Phase 3 definition.
- [ ] Implement/select a feature extractor (CNN backbone) for instance embeddings.
- [ ] Implement embedding storage/caching for efficiency.
- [ ] Implement a basic attention-based aggregation mechanism (single attention, as a stepping stone toward dual attention).
- [ ] Implement bag representation construction from aggregated instance embeddings.
- [ ] Implement classification head for bag-level prediction.
- [ ] Implement correct handling of variable numbers of instances per bag (padding/masking or dynamic batching).
- [ ] Document: bag construction method, instances-per-bag statistics, feature extractor used, embedding dimensionality, aggregation method, classification head architecture.

### Files / Modules
- `src/mil/instance_generation.py`
- `src/mil/feature_extractor.py`
- `src/mil/aggregation.py`
- `reports/phase7_mil_pipeline.md`

### Expected Outputs
- Working, tested MIL pipeline capable of handling the dataset's actual bag-size distribution.

### Validation Checks
- [ ] Unit test: variable bag sizes processed without shape errors.
- [ ] Unit test: forward pass produces correctly shaped bag-level output.

### Exit Criteria
- [ ] MIL pipeline runs end-to-end on a sample of the training split without errors.

### Potential Issues / Risks
- Very small bags (e.g., single-instance) may degrade the value of attention-based aggregation — document if this occurs.

---

# Phase 8 — Dual Attention MIL Model

**Status:** [ ] Not Started

### Objective
Implement the core research contribution: a genuine Dual Attention MIL architecture with two complementary attention mechanisms/stages.

### Why This Phase Matters
This is the central novel component of the project and must be implemented rigorously, not as a superficial addition of an attention layer to an existing CNN.

### Prerequisites
Phase 7 complete.

### Tasks
- [ ] Design the first attention mechanism (e.g., instance-level attention scoring within a bag).
- [ ] Design the second, complementary attention mechanism (e.g., channel/feature-level attention, or a second-stage refinement attention over the first attention's output — architecture choice must be documented and justified).
- [ ] Implement combination/fusion of the two attention stages into a final instance weighting.
- [ ] Implement bag-level feature aggregation using the dual attention weights.
- [ ] Implement the classification head on top of the aggregated bag representation.
- [ ] Ensure the model can return raw attention weights (both stages) for later explainability use (Phase 11).
- [ ] Produce an architecture diagram covering: instance feature extraction → attention stage 1 → attention stage 2 → weighting → aggregation → bag representation → classification head.
- [ ] Write architecture documentation explaining the mathematical formulation of both attention mechanisms.
- [ ] Implement unit tests confirming attention weight output shapes match instance counts per bag.

### Files / Modules
- `src/mil/dual_attention.py`
- `src/models/dual_attention_mil.py`
- `reports/phase8_architecture.md` (with diagram)

### Expected Outputs
- Fully implemented, documented Dual Attention MIL model returning both predictions and attention weights.

### Validation Checks
- [ ] Forward pass test on sample bags of varying sizes.
- [ ] Attention weight dimension test (weights sum appropriately / match instance count).

### Exit Criteria
- [ ] Model architecture implemented, documented, and passing all unit tests.

### Potential Issues / Risks
- Risk of implementing a superficial "attention-in-name-only" model — mitigate via the mandatory architecture documentation and diagram review.

---

# Phase 9 — Model Training, Optimization & Experiment Tracking

**Status:** [ ] Not Started

### Objective
Train the Dual Attention MIL model (and re-confirm baselines under identical conditions) using a reproducible, properly tracked training process.

### Why This Phase Matters
Rigorous, tracked, reproducible training is required for the eventual comparison against baselines to be scientifically valid.

### Prerequisites
Phase 6 and Phase 8 complete.

### Tasks
- [ ] Fix and document reproducible random seeds across all libraries used.
- [ ] Define training configuration (batch size, epochs, image size, etc.) in a config file.
- [ ] Select and document optimizer and learning-rate scheduler.
- [ ] Define batching strategy compatible with variable bag sizes.
- [ ] Implement class-imbalance handling (e.g., class weighting, weighted sampling) informed by Phase 1's actual class distribution.
- [ ] Implement early stopping based on validation metrics.
- [ ] Implement checkpointing and best-model selection based on validation performance only (never test set).
- [ ] Implement experiment logging (e.g., structured logs/CSV, or a tracking tool) capturing all runs.
- [ ] Track per-epoch: training loss, validation loss, accuracy, precision, recall, sensitivity, specificity, F1, ROC-AUC, PR-AUC.
- [ ] Run training for baselines (re-confirm under final pipeline) and the Dual Attention MIL model.

### Files / Modules
- `src/training/train.py`
- `configs/training_config.yaml`
- `experiments/logs/`
- `checkpoints/`
- `reports/phase9_training_report.md`

### Expected Outputs
- Trained model checkpoints for all baselines and the Dual Attention MIL model.
- Full experiment logs.

### Validation Checks
- [ ] Confirm best-model selection logic never references test-set metrics.
- [ ] Confirm reproducibility by re-running one experiment and comparing results within expected variance.

### Exit Criteria
- [ ] All models trained, logged, and checkpointed; best models selected via validation performance.

### Potential Issues / Risks
- Class imbalance severity (to be known from Phase 1) may require more aggressive handling than initially planned.

---

# Phase 10 — Evaluation, Error Analysis & Robustness Testing

**Status:** [ ] Not Started

### Objective
Evaluate all trained models on the frozen test set at the correct level(s) of granularity, using a comprehensive metric suite — not accuracy alone.

### Why This Phase Matters
Clinically meaningful evaluation requires sensitivity/specificity and error analysis, not just headline accuracy, especially in a class-imbalanced medical imaging context.

### Prerequisites
Phase 9 complete; test set still frozen and unused until now.

### Tasks
- [ ] Determine evaluation granularity supported by the dataset (instance-level, image-level, and bag/study/patient-level as applicable per Phase 3's bag definition).
- [ ] Run final evaluation of all baselines and the Dual Attention MIL model on the frozen test set.
- [ ] Generate confusion matrices for each model.
- [ ] Generate ROC curves and compute ROC-AUC.
- [ ] Generate Precision-Recall curves and compute PR-AUC.
- [ ] Compute sensitivity, specificity, precision, recall, F1-score for each model.
- [ ] Perform calibration analysis where feasible (e.g., reliability diagrams).
- [ ] Perform false-positive case analysis (qualitative review of misclassified benign-as-malignant cases).
- [ ] Perform false-negative case analysis (qualitative review of misclassified malignant-as-benign cases — highest clinical priority).
- [ ] Perform difficult-case analysis (cases with high uncertainty/low confidence).
- [ ] Perform class-wise performance analysis.
- [ ] Perform robustness analysis (e.g., performance under mild input perturbations, or performance stratified by image quality/source group if recoverable).
- [ ] Produce comparative results table: baselines vs. Dual Attention MIL, across all metrics.

### Files / Modules
- `src/evaluation/metrics.py`
- `src/evaluation/error_analysis.py`
- `reports/phase10_evaluation_report.md`

### Expected Outputs
- Comprehensive evaluation report with comparative results and error analysis.

### Validation Checks
- [ ] Confirm test set was used exactly once for final evaluation (no iterative tuning against it).
- [ ] Confirm all required metrics are reported for every model.

### Exit Criteria
- [ ] Evaluation report finalized, including honest discussion of whether Dual Attention MIL outperforms baselines.

### Potential Issues / Risks
- Small test set size (dependent on Phase 4 splitting) may limit statistical confidence — report confidence intervals or note this limitation explicitly.

---

# Phase 11 — Explainability & Ultrasound Attention Visualization

**Status:** [ ] Not Started

### Objective
Use the Dual Attention MIL model's attention weights to visualize influential instances/regions, while clearly distinguishing model attention from clinical explanation.

### Why This Phase Matters
Explainability is central to clinical decision support credibility, but overclaiming attention-as-diagnosis would violate medical AI safety rules.

### Prerequisites
Phase 8 (model) and Phase 9 (trained weights) complete.

### Tasks
- [ ] Implement extraction and visualization of attention weights per instance within a bag.
- [ ] Implement top-k important instance identification per prediction.
- [ ] Implement ultrasound evidence visualization (e.g., overlaying attention scores on instance thumbnails).
- [ ] Implement heatmap generation where spatially meaningful (e.g., for patch-based instances).
- [ ] Optionally implement Grad-CAM (or equivalent) on the CNN feature extractor for cross-checking attention-based highlights.
- [ ] Write an explicit "Model Attention vs. Clinical Explanation" disclaimer section.
- [ ] If lesion ground-truth masks were confirmed present in Phase 1, quantitatively evaluate overlap between attention highlights and annotated regions (e.g., IoU-style comparison); otherwise, explicitly state this analysis is not possible with the available data.

### Files / Modules
- `src/explainability/attention_visualization.py`
- `src/explainability/gradcam.py` (optional)
- `reports/phase11_explainability_report.md`

### Expected Outputs
- Visualized attention evidence for representative predictions (correct and incorrect).
- Explicit clinical-explanation disclaimer documentation.

### Validation Checks
- [ ] Visualizations render correctly for bags of varying sizes.
- [ ] Report explicitly states whether mask-overlap validation was possible.

### Exit Criteria
- [ ] Explainability report and visualization tooling completed.

### Potential Issues / Risks
- Without ground-truth masks, explainability validation remains qualitative only — must be stated as a limitation, not glossed over.

---

# Phase 12 — MRI Cross-Modal Validation

**Status:** [ ] Not Started

### Objective
Rigorously determine whether reliable paired MRI data exists for cross-modal validation, and only build the concordance module if such data is genuinely available.

### Why This Phase Matters
This is explicitly flagged as a high-risk area for fabrication. The primary Kaggle dataset is an ultrasound-only dataset; MRI validation must not be faked or inferred.

### Prerequisites
Phase 1 audit results (confirming presence/absence of patient/study identifiers) and Phase 10 (ultrasound predictions available for comparison).

### Tasks
- [ ] Formally re-confirm from Phase 1 audit whether the primary Kaggle dataset contains any patient IDs, MRI identifiers, MRI reports, or paired MRI examinations.
- [ ] **Feasibility Gate:** If none of the above are confirmed present:
  - [ ] Document explicitly: "Cross-modal MRI validation cannot be clinically demonstrated using the primary Kaggle dataset alone."
  - [ ] Research and document candidate external/supplementary MRI datasets or sources that could provide compatible patient/study-level information, without integrating them without further verification and explicit approval.
  - [ ] Mark the MRI cross-modal module as a "research prototype design," including architecture and interface design, without any executed patient-level validation.
- [ ] **If paired MRI data is found/obtained and verified:**
  - [ ] Define patient matching method between ultrasound and MRI records.
  - [ ] Define study matching method.
  - [ ] Implement MRI report/finding extraction (structured or NLP-based, depending on report format).
  - [ ] Define terminology normalization scheme (e.g., BI-RADS or equivalent standardized categories) with explicit sourcing.
  - [ ] Define what constitutes benign/malignant evidence from MRI findings.
  - [ ] Define concordance and discordance criteria precisely.
  - [ ] Define missing-MRI handling (must not fabricate substitute data).
  - [ ] Implement the comparison pipeline: ultrasound AI prediction vs. structured MRI evidence → concordance/discordance output.
  - [ ] Ensure all outputs use "cross-modal diagnostic consistency" / "cross-modal concordance" language — never language implying MRI "proves" the model correct.
- [ ] Write the MRI feasibility report regardless of outcome.

### Files / Modules
- `src/mri/feasibility_check.py`
- `src/mri/matching.py` (only if data available)
- `src/mri/report_extraction.py` (only if data available)
- `src/mri/concordance.py` (only if data available)
- `reports/phase12_mri_feasibility.md`

### Expected Outputs
- A definitive feasibility statement.
- Either: (a) a documented "not clinically demonstrable with current data" conclusion with a designed-but-unexecuted module spec, or (b) a working, verified concordance module using confirmed real paired data.

### Validation Checks
- [ ] No MRI record, report, or finding used in the project is fabricated or inferred without source verification.
- [ ] Feasibility conclusion is explicitly stated in the final report and README.

### Exit Criteria
- [ ] MRI feasibility formally documented; module either implemented against verified real data or explicitly marked as unexecuted/prototype-only.

### Potential Issues / Risks
- Strong temptation exists to simulate MRI data for demo purposes — this is explicitly forbidden and must be resisted; any simulated/demo-only MRI data used for illustration must be labeled unmistakably as "SYNTHETIC / ILLUSTRATIVE ONLY — NOT REAL PATIENT DATA" and never mixed into evaluation.

---

# Phase 13 — Clinical Decision Support / Risk Stratification

**Status:** [ ] Not Started

### Objective
Build a decision-support output layer that presents predictions, risk, evidence, and (if available) cross-modal concordance — without prescribing treatment or replacing clinical judgment.

### Why This Phase Matters
This phase turns raw model output into a clinically-framed but safely-bounded presentation, which is where overclaiming risk is highest.

### Prerequisites
Phase 10 (evaluation), Phase 11 (explainability), Phase 12 (MRI feasibility outcome) complete.

### Tasks
- [ ] Define the decision-support output schema:
  - Prediction: Benign / Malignant
  - Confidence/probability score
  - Risk category: Low / Moderate / High (with explicit, documented thresholding methodology tied to model probability and/or calibration from Phase 10)
  - Evidence: top-k important ultrasound instances (from Phase 11)
  - Cross-modal evidence: MRI concordant / discordant / unavailable (from Phase 12 outcome)
  - Suggested action: Clinical review / further investigation / follow-up consideration
- [ ] Ensure risk thresholds are derived from validation-set calibration, not arbitrary cutoffs.
- [ ] If any treatment-support consideration is included, source it from a verifiable, cited clinical guideline and label it explicitly: "Treatment-support consideration requiring qualified clinician review."
- [ ] Implement the decision-support module producing the above structured output per case.
- [ ] Add mandatory disclaimers to every output instance.

### Files / Modules
- `src/clinical_support/decision_support.py`
- `reports/phase13_decision_support_spec.md`

### Expected Outputs
- Structured, safely-bounded decision-support output module.

### Validation Checks
- [ ] No output includes an autonomous treatment prescription.
- [ ] Every output includes a disclaimer and risk-category derivation is traceable to documented thresholds.

### Exit Criteria
- [ ] Decision-support schema and module implemented and reviewed for safety-language compliance.

### Potential Issues / Risks
- Risk-category thresholds may need recalibration if class imbalance or small test-set size limits reliable probability calibration (link back to Phase 10 findings).

---

# Phase 14 — Application Integration, Testing, Documentation & Final Demonstration

**Status:** [ ] Not Started

### Objective
Integrate all components into a single end-to-end application, complete all remaining tests, finalize documentation, and prepare the final demonstration.

### Why This Phase Matters
This is the deliverable phase — all prior research and engineering must be assembled into a coherent, testable, presentable system with honest, clearly labeled limitations.

### Prerequisites
All prior phases (0–13) completed.

### Tasks
- [ ] Build the end-to-end pipeline: upload ultrasound image(s) → preprocessing → MIL bag creation → Dual Attention MIL inference → prediction → risk score → attention visualization → MRI validation (if available) → cross-modal consistency → decision-support output.
- [ ] Build the application UI/interface layer displaying: prediction, confidence/probability, important image regions/instances, MRI validation status, limitations section, and medical disclaimer.
- [ ] Ensure UI never displays unsupported claims (e.g., "100% diagnosis," "guaranteed cancer detection").
- [ ] Complete remaining items from the Global Testing Requirements checklist (end-to-end inference test, etc.).
- [ ] Write final project report consolidating all phase reports.
- [ ] Write user-facing README with setup instructions, limitations, and disclaimers.
- [ ] Prepare final demonstration materials (slides/demo script) that accurately reflect what was and was not achieved (e.g., honestly state MRI validation outcome from Phase 12).
- [ ] Conduct an internal review pass against the Non-Negotiable Project Rules checklist before final submission.

### Files / Modules
- `app/` (full application)
- `reports/final_project_report.md`
- `README.md`

### Expected Outputs
- Fully functioning end-to-end demo application.
- Final consolidated report and README.

### Validation Checks
- [ ] End-to-end test passes on sample inputs.
- [ ] UI reviewed for absence of overclaiming language.
- [ ] Final report cross-checked against all 15 Non-Negotiable Project Rules.

### Exit Criteria
- [ ] Application demonstrated end-to-end; final documentation complete and internally reviewed for safety-language and accuracy compliance.

### Potential Issues / Risks
- Time pressure near submission deadlines is the highest-risk point for accidentally overstating results (e.g., calling the prototype "clinically validated") — final review pass exists specifically to catch this.

---

*This PROGRESS.md is a living document. Update phase statuses, the Overall Progress table, Completed Milestones, and Current Blockers as work proceeds. Do not mark any phase or task complete until its exit criteria and validation checks are genuinely satisfied.*
