# Phase 12 — MRI Feasibility & Data-Availability Report

**Status:** Feasibility investigation complete — **NOT CURRENTLY FEASIBLE (patient-level paired cross-modal validation)** — audit-only preflight; uncommitted, awaiting owner review.
**Date:** 2026-09-21
**Scope of this document:** Preflight/feasibility ONLY. No MRI code, no loaders, no models, no pairing tables, no acquisition, no evaluation. Repository HEAD at investigation time: `73210016bfd869493be724ee94a85b068546e0cf` (`Phase 11: add explainability and attribution`), working tree clean, `main = origin/main`.

---

## 1. Objective

Determine, from the actual repository data and a documented external investigation, whether the project's intended objective —

> "cross-validate ultrasound prediction accuracy against MRI reports of the same patient"

— is achievable with data that can be **legitimately linked**, without fabricating patient/study/lesion identity. This is the roadmap's Phase 12 feasibility gate (PROGRESS.md §Phase 12, "Feasibility Gate"), which explicitly states that if no patient IDs / MRI identifiers / MRI reports / paired examinations are confirmed present, the project must document that cross-modal MRI validation cannot be clinically demonstrated using the primary dataset alone, and must research (not integrate) external sources pending explicit owner approval.

---

## 2. Current ultrasound dataset status (verified on disk, 2026-09-21)

Inspected live at `dataset/raw/.kagglehub_cache/datasets/vuppalaadithyasairam/ultrasound-breast-images-for-breast-cancer/versions/1/` (the download backing the frozen Phase 1 audit; the audit's findings re-verified by direct inspection):

| Property | Verified value |
| --- | --- |
| Total images | 9,016 (8,158 PNG / 858 JPG) — matches Phase 1 audit exactly |
| Layout | `ultrasound breast classification/train/{benign,malignant}` (8,116 files: 4,074 benign / 4,042 malignant); `…/val/{benign,malignant}` (900 files: 500 benign / 400 malignant) |
| Labels | Directory-encoded only (`benign` / `malignant`) |
| Filenames | `benign (36)-rotated1-sharpened.png`, `malignant (99).jpg`, … — class + integer + explicit augmentation chain (`rotated1`, `rotated2`, `rotated32`, `sharpened`) |
| Metadata files | **None** (exhaustive scan: zero non-image files in the dataset tree) |
| Patient / study / lesion / accession identifiers | **None** (Phase 1 audit §5, owner-approved baseline; re-confirmed here — filename integers are unsourced sequence numbers, not patient IDs) |
| Grouping artifact | 496 filename-derived "source keys" — `inferred_from_filename_not_verified_identifier`; NOT patient/study/lesion identifiers |
| MRI content | **None** — `audit_summary.json` MRI scan: vocabulary {mri, mr_, _mr, -mr, dce, t1, t2, flair, dicom} → `mri_hint_files: []`, D-2 state **C** ("no MRI files found") → **T-5** (absence of MRI validation) |

Live re-inspection of filenames and directory structure in this phase reproduces every one of these findings. **The ultrasound dataset contains no verified clinical identifier of any kind — it therefore cannot anchor a cross-modal linkage on its own side, independent of whatever MRI data exists elsewhere.**

---

## 3. Existing MRI-data status (repository search)

Case-insensitive repository-wide searches for `MRI / mri / DICOM / .dcm / NIfTI / .nii(.gz) / radiology / report / patient_id / study_id / accession / lesion_id / case_id / subject_id`:

- **Zero MRI files anywhere**: no `.dcm`, `.nii`, `.nii.gz`, or MRI-named image file exists (filesystem search excluding `venv/`, `.git/`).
- The only MRI-related repository item remains the **0-byte placeholder** `backend/services/mri_validation.py` (documented in Phase 0 as F-9; left untouched, no significance).
- All MRI references in `src/`, `tests/`, and reports are **negative/boundary statements** (guards that *forbid* MRI code in earlier phases, the T-1…T-4 taxonomy, and D-2/T-5 state records) — not data.
- `data/manifests/` contains only ultrasound split/audit manifests. No MRI manifest, no pairing table, no linkage artifact exists.

**Conclusion: no MRI data of any kind exists locally, and no linkage artifact exists.**

---

## 4. External candidate datasets/sources investigated

Investigated via public documentation only (no download, no registration, no acquisition — acquisition requires owner authorization per Q-8/A-4c, `reports/phase0_decisions.md` §252). All candidates are **MRI-only cohorts**; none contains ultrasound, and none has any documented relationship to the Kaggle ultrasound dataset's cases.

| Candidate | Identity / source | Content | Identifiers / linkage mechanism | MRI reports? | Access | Pairable with the ultrasound data? |
| --- | --- | --- | --- | --- | --- | --- |
| **MAMA-MIA** (Garrucho et al., *Scientific Data* 2025) | Multi-center breast DCE-MRI benchmark; imaging integrated from 4 TCIA collections; distributed via Synapse | 1,506 DCE-MRI cases, expert tumor segmentations, clinical/imaging tables, train/test splits | Case ID = collection acronym + patient ID (within-MRI-cohort linkage only) | NO radiology reports (structured clinical tables) | Public, registration (Synapse) required; academic use subject to underlying TCIA/data-source terms | **NO** — independent cohort; no shared identifier space; different population (neoadjuvant-chemotherapy trial patients) |
| **BreastDCEDL / BreastDCEDL_ISPY2** (Zenodo 15627233; TCIA analysis result) | DL-ready 3D DCE-MRI from 2,070 patients across 3 clinical trials (incl. I-SPY1/2; 982 in ISPY2 curation) | 3D DCE-MRI + labels | Patient/study IDs within each trial | UNKNOWN / not documented as free-text reports | Open (Zenodo) / TCIA terms | **NO** — clinical-trial population, unlinked to the ultrasound data |
| **DUKE-BREAST-CANCER-MRI** (TCIA) | Single-institution (Duke) DCE-MRI collection | 922 patients, invasive breast cancer, pre-treatment DCE-MRI + clinical data | TCIA patient IDs (within-cohort) | Clinical/pathology fields, not documented as public radiology reports | Public, TCIA data-usage policy | **NO** — independent cohort, unlinked |
| **NYU Breast MRI v1.0** | NYU (Geras lab) research dataset | 21,537 MRI studies / 13,463 patients (2008–2020) | Patient/study IDs within-cohort; research-use request process | UNKNOWN — report availability not verified | Restricted/request-based | **NO** — independent cohort, unlinked |

A 2026 systematic review of public breast-ultrasound AI datasets (PMC13206445) independently corroborates the linkage finding: public ultrasound datasets provide **no cross-dataset patient identifier**, so even dataset-to-dataset matching is not supportable. Public literature reporting genuinely paired US+MRI cohorts (e.g., dual-centre fusion studies) is based on **private institutional data** — no public paired resource was found in either search direction.

---

## 5. Identifier / linkage audit (the central Phase 12 rule)

Valid cross-modal pairing requires a **defensible, source-documented linkage mechanism**. The audit finds linkage impossible at every level:

1. **Ultrasound side anchor: absent.** The ultrasound dataset has zero verified identifiers (§2). There is no key on the ultrasound side to match *anything* against — this alone is decisive and cannot be remedied by any external dataset.
2. **MRI side candidates: independent populations.** Every public MRI candidate (§4) identifies its own patients internally but has no documented overlap or mapping to the Kaggle dataset's cases.
3. **Prohibited linkage bases — all absent or invalid.** No shared filenames, no sequential correspondence, no accession numbers, no demographics, no explicit cross-modality mapping exists. Per the owner's Phase 12 rule, filename similarity, sequential numbering, visual similarity, label agreement, or demographic approximation are **not** acceptable as pairing evidence — and none is even present here to misuse.
4. **No linkage table was fabricated.** None will be produced at any point unless verified paired data is obtained under owner authorization.

---

## 6. Label compatibility audit

| Dimension | Ultrasound side (verified) | MRI-side candidates (documented) | Compatible? |
| --- | --- | --- | --- |
| Class definition | Directory-encoded `benign`/`malignant` (basis undocumented; unknown pathology grounding) | Trial/institutional clinical labels (e.g., pCR/response, pathology-confirmed invasive cancer in Duke) | **UNKNOWN → NO** — mapping would require a documented, sourced justification that does not exist |
| Granularity | Image-level labels; bags are filename-derived groups, NOT patients | Patient/study-level | **NO** — no common unit |
| Population | Unknown provenance, unknown acquisition protocol | Clinical trials (neoadjuvant) / single-institution diagnostic cohorts | **NO documented overlap** — different populations entirely |

Even if identifiers existed, benign/malignant definitions could not be collapsed across these cohorts without an explicitly documented mapping — which no candidate provides.

## 7. Modality compatibility audit

- **Ultrasound**: 2D grayscale B-mode stills, 224×224 after preprocessing.
- **MRI candidates**: 3D/4D DCE-MRI volumes (multi-sequence, contrast-enhanced), completely different acquisition, dimensionality, and radiology workflow (BI-RADS-MRI, kinetic curves).
- The intended "MRI **reports**" component is weaker still: none of the accessible public candidates ships radiology reports (structured clinical tables only, where anything exists). **Report-based cross-modal validation has no available public data path.**
- A concordance comparison between a 2D-US bag classifier and 3D DCE-MRI findings would additionally require modality-specific pipelines that do not exist and are out of scope for this phase.

## 8. Access / licensing notes

- **MAMA-MIA**: public via Synapse with registration; governed by the underlying TCIA/source collections' terms — per-use confirmation required before any acquisition.
- **TCIA collections (Duke, ISPY curations)**: public under TCIA Data Usage Policy (attribution/acknowledgement obligations).
- **BreastDCEDL**: open Zenodo deposit.
- **NYU**: request-based research access; terms not verified here.
- **The Kaggle ultrasound dataset**: public Kaggle terms; already in use under Phase 0 D-1.
- No legal-permission conclusion is claimed beyond "appears accessible for academic use subject to each source's own terms, with registration/attribution requirements." No dataset was downloaded or registered for in this phase.

## 9. Leakage / overlap considerations

- Because **neither side has shared identifiers**, overlap between the ultrasound dataset and any MRI candidate **cannot be established or excluded**. Independence must therefore be stated as "unverifiable," not "confirmed."
- The ultrasound dataset's own known issues compound this: 8,520 explicit augmentation variants, 228 exact-duplicate groups, and 2 source keys spanning the pre-existing train/val split (Phase 1, owner-accepted as-is). Any future cohort-level comparison would inherit these limitations.
- No datasets were merged; the frozen train/validation/test manifests were not touched.

## 10. Feasibility matrix

| Requirement | Available? | Evidence | Consequence |
| --- | --- | --- | --- |
| Ultrasound images | **YES** | 9,016 images on disk; frozen pipeline/manifests (Phases 1–5) | Existing project basis intact |
| MRI images | **NO** (locally) / YES as unlinked public cohorts | Repo scan: zero MRI files; §4 candidates | No MRI data tied to this project's cases |
| MRI reports | **NO** | No report files anywhere; public candidates ship structured tables, not reports | "Report-based" validation has no data path |
| Patient identifiers | **NO** (ultrasound side) / within-cohort only (MRI side) | Phase 1 audit §5 re-verified; §4 | No shared identifier space |
| Study identifiers | **NO** | Same evidence | Study-level matching impossible |
| Lesion identifiers | **NO** | Same evidence | Lesion-level matching impossible |
| Cross-modal linkage key | **NO** | §5 audit | **Pairing impossible — decisive** |
| Compatible labels | **UNKNOWN → NO** | §6 | Cross-cohort class comparison unjustified without a documented mapping |
| Paired cases | **NO** | §5 | True paired validation cannot be performed |
| Academic-access availability | **YES (unlinked cohorts only)** | §8 | A *cohort-level* external MRI study is *obtainable in principle* (owner authorization required), but does not enable paired validation |

## 11. What can and cannot legitimately be claimed

**A. True paired cross-modal validation (same patient/study/lesion, explicit linkage):** **NOT possible** with any data available to this project. No public paired US+MRI dataset was found; the ultrasound side has no identifiers; no candidate MRI cohort shares any linkage with it.

**B. Cohort-level external validation (independent MRI cohort, unlinked):** *Conceptually obtainable in principle* (e.g., MAMA-MIA/BreastDCEDL/Duke), but it is **not cross-modal validation**, requires new modality pipelines, external-dataset acquisition authorization (Q-8/A-4c), and would only support "external validity of an MRI-side model" — it cannot validate the ultrasound model on the same cases. Overlap with the ultrasound data would remain unverifiable.

**C. Label-level comparison (aggregate class distributions):** Trivially possible in principle, but statistically and semantically weak; it is **not** validation of the model on any case.

**D. Literature-level comparison:** Possible (descriptive comparison with published results); **not** validation on the same cases.

Per the binding T-1…T-4 taxonomy (`reports/phase0_decisions.md` §2.4): this project has data support **only for T-5 (absence of MRI validation)**. T-2/T-3-style descriptive extensions would require owner-authorized acquisition and are not validated on the same cases.

## 12. Original intended objective vs. what the data supports

| | Statement |
| --- | --- |
| **Original intended objective** (project description) | "cross-validate ultrasound prediction accuracy against MRI reports of **the same patient**" |
| **What the available data actually supports** | Ultrasound-only research-prototype evaluation (Phases 6/9/10) on an identifier-free, augmentation-heavy Kaggle dataset. **Cross-modal MRI validation cannot be clinically demonstrated using the primary Kaggle dataset alone** — the Phase 12 gate condition is met. No paired data, no reports, no identifiers exist; none can be legitimately inferred. |

This distinction must be preserved explicitly in all future documentation (including Phase 13's "Cross-modal evidence: MRI concordant / discordant / **unavailable**" field, which should default to **unavailable**, and Phase 14's demonstration materials). The original objective is **not** quietly rewritten as achieved; it is recorded as **unsupported by available data**.

## 13. Recommended next technical step (descriptive, not evaluative)

Per the roadmap's own Feasibility Gate, the remaining Phase 12 path consistent with this evidence is to document the MRI cross-modal module as a **"research prototype design"** — architecture and interface design only, explicitly unexecuted, with no patient-level validation — while leaving all acquisition/integration decisions (external datasets Q-8, synthetic MRI Q-7) to explicit owner decisions. Any future change to this conclusion requires *verified paired data with a documented linkage mechanism*, which no currently identified public source provides.

## 14. Explicit Phase 12 boundary

This phase delivered **this report only**. NOT performed: MRI implementation (`src/mri/*`), MRI loaders/preprocessing/models, pairing scripts, linkage tables, concordance metrics, multimodal/fusion networks, cross-modal inference, MRI acquisition/registration, manifest/split modification, training, test-set evaluation, or any modification to frozen artifacts or prior-phase code. The empty `backend/services/mri_validation.py` placeholder remains untouched (F-9).

---

### Feasibility classification (§17)

**NOT CURRENTLY FEASIBLE** — paired patient-level ultrasound↔MRI cross-modal validation, and a fortiori report-based cross-modal validation, cannot be established: the ultrasound dataset carries zero verified identifiers (no anchor), no publicly accessible paired US+MRI dataset with a documented linkage mechanism suitable for this project was identified during this investigation, and all identified public MRI cohorts are independent populations with no linkage mechanism and no radiology reports. The gate condition defined in PROGRESS.md ("Cross-modal MRI validation cannot be clinically demonstrated using the primary Kaggle dataset alone") is formally met and documented here.
