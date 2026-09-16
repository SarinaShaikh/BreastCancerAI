# Phase 0 — Project Definition, Scope & Research Requirements

**Deliverable:** `reports/phase0_scope.md`
**Phase:** 0 (per `PROGRESS.md`)
**Status:** APPROVED — **Phase 0 Complete** (2026-09-16). The project owner approved the Phase 0 deliverables as a whole on 2026-09-16 (approval authority: project owner; see `reports/phase0_decisions.md` §4). Validation record: §25.
**Author of record:** AI development assistant (working under `PROGRESS.md` rules)
**Date recorded:** 2026-09-16
**Authority:** `PROGRESS.md` is the authoritative specification. Where this document conflicts with `PROGRESS.md`, `PROGRESS.md` wins.

> **Document purpose.** Establish a precise, honest and bounded definition of what this project will and will not do *before* any data, model or pipeline work begins (Phase 0 objective, `PROGRESS.md`).
>
> **Nothing in this document is a result.** It contains no experimental numbers, no dataset findings, no patient-level information and no verified metadata. Every statement about the dataset is an *assumption or uncertainty*, explicitly labelled as such, and must be confirmed or refuted by direct inspection in Phase 1.

---

## 1. Project Title

**Predicting Breast Cancer from Ultrasound Images with Cross-Modal Validation using Dual Attention Multiple Instance Learning**

*(Title as recorded in `PROGRESS.md`. Note: `README.md` currently uses the variant "Predicting Breast Cancer from Multi-Modal Images using Dual Attention Multiple Instance Learning" — a title inconsistency flagged for resolution in §24, not silently changed here.)*

---

## 2. Problem Statement

### 2.1 Original brief (verbatim from `PROGRESS.md`)

> To create an AI-powered breast cancer prediction system that utilizes ultrasound imaging and Dual Attention Multiple Instance Learning to improve early diagnosis, enhance clinical decision-making, and provide personalized treatment recommendations. The system will cross-validate ultrasound prediction accuracy against MRI reports of the same patient to ensure diagnostic consistency and accuracy.

### 2.2 Bounded restatement (what this project actually commits to)

This project is a **research prototype** that investigates whether a Dual Attention Multiple Instance Learning (MIL) model can classify breast ultrasound images as benign or malignant, and whether instance-level attention aggregation produces *model evidence* that can be visualised and inspected. It seeks to establish whether that approach differs measurably from non-MIL and single-attention MIL baselines under leakage-free evaluation.

### 2.3 Why the brief must be bounded

Three commitments in the original brief have preconditions that are **UNVERIFIED** and must not be treated as available:

| Brief commitment | Precondition | Status |
|---|---|---|
| "cross-validate ultrasound prediction accuracy against MRI reports of the same patient" | Paired MRI examinations or MRI reports linked to the *same* patients | **UNVERIFIED.** The primary dataset is described by its Kaggle listing as an ultrasound image set. No MRI linkage has been confirmed. See §17. |
| "provide personalized treatment recommendations" | A clinically valid, guideline-sourced basis for treatment support | **NOT AVAILABLE in this project.** No treatment recommendation capability is implemented or promised. See §13 and §16. |
| "enhance clinical decision-making" / "improve early diagnosis" | Prospective clinical evaluation against radiologist/oncologist performance | **OUT OF SCOPE.** This project performs no clinical evaluation. See §6. |

The brief describes an *aspiration*. This document records the *bounded commitment*: ultrasound classification, honest comparative evaluation, and—only if the data genuinely supports it—a documented MRI feasibility outcome.

---

## 3. Project Classification

**Classification: Research Prototype (NOT a clinically validated diagnostic system)** — per `PROGRESS.md` and the Non-Negotiable Project Rules.

Implications that bind every later phase:

- Outputs are research artifacts, not diagnostic results.
- No output may be presented as usable for patient care.
- Performance figures are dataset-specific and carry no clinical generalisability claim.
- The prototype has not been reviewed, cleared, certified or registered by any regulatory body, and no such process is in scope.

---

## 4. Research Prototype vs. Clinically Validated System

`PROGRESS.md` Phase 0 requires this distinction to be written explicitly into the report.

| Dimension | This project (research prototype) | Clinically validated diagnostic system |
|---|---|---|
| Regulatory status | None. No clearance, certification or registration sought or obtained. | Requires regulatory review/clearance and a defined intended-use statement. |
| Intended use | Research investigation of a modelling approach on a public dataset. | Diagnosis or triage of real patients, with defined clinical indication. |
| Evidence basis | Offline evaluation on a single public dataset; leakage-free splits within that dataset. | Prospective and/or multi-site studies, reader studies, comparison to standard of care. |
| Population validity | Unknown beyond the source dataset; source population characteristics are not established. | Defined and characterised target population, with subgroup analysis. |
| Failure consequences | Misclassification harms no patient; it degrades research validity. | Misclassification can delay diagnosis or cause unnecessary intervention. |
| Explainability claim | Attention weights are *model evidence*, never a clinical explanation. | Explanation must be clinically actionable and validated as such. |
| Deployment | Local demonstration only. | Governed deployment, monitoring, drift management, incident handling. |
| Language permitted in artifacts | "prototype", "research result", "model attention", "cross-modal concordance" (if applicable), "not clinically validated". | "diagnostic", "detects", "rules out", only within a cleared intended-use statement. |

**Standing statement to be reproduced in later reports and the final README:** this prototype is not a diagnostic device, has not been clinically validated, and must not be used to make or influence patient-care decisions.

---

## 5. Technical Objectives

| ID | Technical objective | Owning phase |
|---|---|---|
| TO-1 | Produce a factual, machine-readable audit and manifest of the dataset with zero assumed metadata. | Phase 1 |
| TO-2 | Quantify duplicate / near-duplicate / augmented (rotation, sharpening) relationships and assign a source-image group to every image. | Phase 2 |
| TO-3 | Define a scientifically defensible MIL bag/instance/label formulation derived from verified dataset structure. | Phase 3 |
| TO-4 | Produce train/validation/test splits at the highest defensible grouping level with automated proof of zero leakage. | Phase 4 |
| TO-5 | Implement a reproducible, justified ultrasound preprocessing pipeline with training-only augmentation. | Phase 5 |
| TO-6 | Implement non-MIL baselines (simple CNN; transfer-learning backbone) and a standard/single-attention MIL baseline on identical splits. | Phase 6, 7 |
| TO-7 | Implement a genuine Dual Attention MIL architecture with two complementary attention stages, returning both predictions and attention weights, with architecture documentation. | Phase 8 |
| TO-8 | Train all models reproducibly; select best models on validation performance only; log all runs. | Phase 9 |
| TO-9 | Evaluate all models once on the frozen test set across a full metric suite (accuracy, sensitivity, specificity, precision, recall, F1, ROC-AUC, PR-AUC) plus confusion matrices, ROC/PR curves, calibration and error analysis. | Phase 10 |
| TO-10 | Visualise instance-level attention evidence (and optionally Grad-CAM cross-checks) with an explicit "model attention ≠ clinical explanation" disclaimer. | Phase 11 |
| TO-11 | Determine, by verified evidence only, whether MRI cross-modal validation is feasible; build the concordance module **only** if real paired data is confirmed. | Phase 12 |
| TO-12 | Produce a safely-bounded decision-support *output schema* (prediction, confidence, risk category, evidence, cross-modal status, non-prescriptive suggested action) with mandatory disclaimers. | Phase 13 |
| TO-13 | Integrate an end-to-end demonstration application, complete the global test checklist, and finalise documentation with honest limitations. | Phase 14 |

*No technical objective above is implemented in Phase 0.*

---

## 6. Research Objectives

| ID | Research objective | Owning phase |
|---|---|---|
| RO-1 | Determine empirically whether Dual Attention MIL differs from non-MIL image-level baselines in bag-level classification on leakage-free splits. | Phase 6, 9, 10 |
| RO-2 | Determine empirically whether Dual Attention MIL differs from a standard/single-attention MIL baseline. | Phase 7, 9, 10 |
| RO-3 | Determine whether instance-level evidence aggregation yields inspectable model evidence beyond an image-level score, and characterise how far that evidence can be trusted. | Phase 7, 11 |
| RO-4 | Quantify how much apparent performance depends on evaluation design, by comparing leakage-free source-image/augmentation-family splitting against naive image-level splitting. | Phase 4, 10 |
| RO-5 | Establish a defensible, evidence-based feasibility verdict on ultrasound/MRI cross-modal validation with the data actually available. | Phase 1, 12 |
| RO-6 | Document precisely which conclusions the available dataset can and cannot support, and which limitations bound every reported result. | Phase 1–14 |
| RO-7 | Verify (through structured literature review) the novelty status of the seven candidate contributions listed in `PROGRESS.md`. No novelty is claimed until that review supports it. | Phase 0 (tracking), ongoing |

*The research questions operationalising these objectives are defined in `reports/research_questions.md`. They are defined, not answered, in Phase 0.*

---

## 7. Intended Input

| Input | Description | Availability status |
|---|---|---|
| Breast ultrasound images | Grayscale or RGB ultrasound images labelled benign/malignant, from the fixed primary dataset ("Ultrasound Breast Images for Breast Cancer", Kaggle, author Vuppala Adithya Sairam). | **ASSUMED.** Existence and structure unverified until Phase 1. Exact class folder names/casing must not be assumed. |
| Image-level label | Benign/malignant per image. | **ASSUMED** per Kaggle listing. |
| Bag/grouping key (patient ID, study ID, lesion ID, or source-image group) | Required for MIL bags and leakage-free splitting. | **UNVERIFIED.** Presence of patient/study/lesion IDs is unknown. A source-image/augmentation-family grouping will be derived in Phase 2 regardless. See §18, §19. |
| Lesion masks / annotations | Ground-truth regions for quantitative explainability validation. | **UNVERIFIED**, and expected absent. Must not be assumed. |
| Paired MRI data or MRI reports | Basis for cross-modal validation. | **UNVERIFIED** and expected absent. See §17. |
| Clinical metadata (age, BI-RADS, histopathology) | Would enable risk-factor or staging analysis. | **UNVERIFIED** and expected absent. No clinical analysis is promised. |

**Input the project must never invent:** patient IDs, study IDs, lesion IDs, MRI records, MRI reports, MRI findings, masks, annotations, clinical findings, class balance figures, or dataset sizes.

---

## 8. Intended Outputs

**Definitely in scope (data permitting):**

- A binary per-case prediction: **benign / malignant** (Phase 10, 14).
- A **probability/confidence score**, with calibration reported where feasible (Phase 10).
- **Model evidence:** top-k influential ultrasound instances/regions per prediction, and attention visualisations (Phase 11).
- A **comparative results table**: non-MIL baselines vs. single-attention MIL vs. Dual Attention MIL across all required metrics (Phase 10).
- **Error analysis**: false-positive, false-negative and difficult-case review, and class-wise analysis (Phase 10).
- A **documentable limitations statement** and a full audit/leakage/manifest trail (Phase 1–4).
- An **end-to-end demonstration application** operating on local inputs (Phase 14).

**Conditional outputs (only if preconditions are verified):**

- A **risk category** (Low/Moderate/High) with thresholds derived from validation-set calibration, not arbitrary cutoffs (Phase 13, 10).
- A **cross-modal concordance status** (concordant / discordant / unavailable) — produced **only** if verified real paired data exists; otherwise the honest output is "unavailable" plus a documented feasibility conclusion (Phase 12).
- A **suggested action** field framed as clinical review / further investigation consideration — never a treatment prescription (Phase 13).

**Explicitly not an output:** a diagnosis, a treatment plan, a prescription, an MRI-derived confirmation of model correctness, or any statement of clinical validation.

---

## 9. Prediction Targets

| Target level | Definition | Status |
|---|---|---|
| **Image-level** | Benign/malignant probability for a single ultrasound image. | **Intended primary target.** Supported if image-level labels are confirmed in Phase 1. |
| **Bag-level** | Benign/malignant probability for a group of instances (bag), where the bag definition follows from verified dataset structure. | **Conditional.** Depends entirely on Phase 1/2/3 findings. The bag construction is unresolved (§19). |
| **Patient/study-level** | Risk estimate at the patient or study level. | **Conditional and may be impossible.** Requires confirmed patient/study identifiers. Must not be assumed. |
| **Lesion-level** | Prediction localised to a lesion. | **Not targetable** unless lesion IDs/annotations are confirmed present. |
| **MRI-derived target** | Any label, finding or ground truth derived from MRI. | **Not a prediction target.** No MRI data is confirmed to exist; none will be invented. |

`PROGRESS.md` requires that evaluation granularity (instance/image/bag) be determined from the dataset in Phase 10; the above is therefore a set of candidate targets, not a commitment.

---

## 10. Proposed High-Level System Workflow

*Provisional design only. Nothing below is implemented, and no component is represented as available.*

```
[Ultrasound image(s)]                                            INPUT (Phase 0: described only)
        │
        ▼
Phase 1   Dataset audit + manifest                 ── verify what actually exists
        ▼
Phase 2   Duplicate / near-duplicate / augmentation-family grouping → source_group_id
        ▼
Phase 3   Bag + instance + label definition (BAG=?, INSTANCE=?, LABEL=?)
        ▼
Phase 4   Group-level train/val/test split + mandatory leakage tests (test frozen)
        ▼
Phase 5   Preprocessing pipeline (training-only augmentation)
        ▼
Phase 6   Baselines: simple CNN · transfer-learning CNN · standard/single-attention MIL
        ▼
Phase 7   MIL infrastructure: instance generation → embeddings → aggregation head
        ▼
Phase 8   Dual Attention MIL: instance features → attention stage 1 → attention stage 2
                                            → weighted aggregation → bag representation → classifier
        ▼
Phase 9   Training / tuning / experiment tracking (best model by validation only)
        ▼
Phase 10  Frozen-test evaluation + full metric suite + error analysis
        ▼
Phase 11  Attention visualisation + "model attention ≠ clinical explanation" disclaimer
        ▼
Phase 12  MRI feasibility gate ──► (a) not demonstrable → documented, module unexecuted
                              └──► (b) verified real paired data → concordance module
        ▼
Phase 13  Bounded decision-support output schema (no prescriptions)
        ▼
Phase 14  Application integration + final report + demo
```

**Design note (research integrity):** the pipeline is deliberately ordered so that grouping and leakage control (Phases 2–4) precede modelling (Phases 6–9). If bags or splits are built after seeing model behaviour, every subsequent metric is compromised.

---

## 11. Scope Boundaries

**In scope (Phase 0–14 as written in `PROGRESS.md`):**

- Auditing and documenting the fixed public Kaggle ultrasound dataset.
- Leakage-aware grouping, splitting and preprocessing of that dataset.
- Non-MIL and MIL baselines, and a Dual Attention MIL architecture.
- Reproducible training and honest, leakage-free evaluation with clinically relevant metrics.
- Attention-based evidence visualisation, explicitly framed as model evidence.
- A **feasibility assessment** of MRI cross-modal validation, with implementation only against verified real data.
- A bounded decision-support output schema without treatment prescriptions.
- Documentation of limitations, and a demonstration application.

**Out of scope:**

- Clinical trials, prospective studies, reader studies, or any clinical evaluation.
- Regulatory clearance, certification, or claims of approved intended use.
- Any diagnostic use with real patients.
- Autonomous treatment recommendation or prescription.
- Acquiring or purchasing additional datasets without explicit approval (Phase 12 may *research* candidate external MRI sources; it may not integrate them unapproved).
- Treating the Kaggle listing description as verified fact.
- Building any model, pipeline, split, bag or UI in Phase 0.

---

## 12. Explicitly NOT Intended To Do

This system is explicitly **not** intended to:

1. Diagnose breast cancer in any patient.
2. Replace, or be presented as a replacement for, a radiologist, oncologist, pathologist or any clinician.
3. Recommend, select, dose or withhold any treatment.
4. Declare a lesion benign or malignant as a clinical conclusion.
5. Rule out malignancy, or be used to justify avoiding biopsy or follow-up.
6. Fabricate, simulate or infer MRI evidence, MRI reports, or cross-modal agreement.
7. Claim cross-modal "confirmation", "proof" or "validation" of model correctness from MRI.
8. Claim clinical validation, clinical utility, or superiority to clinicians.
9. Present attention weights as a clinical explanation or as a localisation of disease.
10. Use patient-identifying or fabricated identifiers to construct patient-level analysis.
11. Report a single headline metric (e.g., accuracy) as evidence of success.
12. Present any performance figure as generalisable beyond the evaluated dataset.
13. Operate in real time on real patient data, or be connected to any clinical system.
14. Be cited as a medical device or as evidence of clinical efficacy.

---

## 13. Current Assumptions (ALL UNVERIFIED)

Every row below is an **assumption about the dataset or domain**, never a verified fact. Phase 1 must confirm or refute each one. Assumptions are recorded rather than relied upon.

| ID | Assumption | Why it matters | Verification phase | Status |
|---|---|---|---|---|
| A-1 | The primary dataset consists of 2D ultrasound images. | Determines the entire modelling approach. | Phase 1 | UNVERIFIED |
| A-2 | Images carry a binary benign/malignant label. | Defines the prediction target. | Phase 1 | UNVERIFIED (per Kaggle listing only) |
| A-3 | Class folder names, spelling and casing may differ from "benign"/"malignant". | Wrong assumptions silently corrupt labels. | Phase 1 | UNVERIFIED — must read exact names as found |
| A-4 | The dataset contains images already augmented by rotation and sharpening. | Creates a leakage risk requiring group-level splitting. | Phase 1, 2 | UNVERIFIED (per Kaggle listing only) |
| A-5 | Multiple images may derive from the same source image. | Determines whether bags and splits are defensible. | Phase 2 | UNVERIFIED |
| A-6 | Patient IDs are absent. | If true, patient-level bags are impossible and must not be fabricated. | Phase 1 | UNVERIFIED — must be explicitly searched for, not assumed either way |
| A-7 | Study IDs are absent. | Same as A-6. | Phase 1 | UNVERIFIED |
| A-8 | Lesion IDs / masks / annotations are absent. | Blocks lesion-level targets and quantitative attention validation. | Phase 1 | UNVERIFIED |
| A-9 | No MRI data, MRI reports or MRI linkage exists in the dataset. | Determines whether Phase 12 can be executed at all. | Phase 1, 12 | UNVERIFIED |
| A-10 | No clinical metadata (age, BI-RADS, histopathology) accompanies the images. | Blocks risk-factor and staging analysis. | Phase 1 | UNVERIFIED |
| A-11 | Image colour space and dimensions are consistent, or are documented if not. | Affects preprocessing and backbone input layers. | Phase 1 | UNVERIFIED |
| A-12 | The dataset is small by deep-learning standards. | Affects statistical confidence and split trade-offs. | Phase 1, 4 | UNVERIFIED — magnitude unknown |
| A-13 | Class distribution may be imbalanced. | Requires imbalance handling in Phases 9–10. | Phase 1 | UNVERIFIED |
| A-14 | Image quality is heterogeneous (probe, gain, speckle, annotations burned into pixels). | May confound results and short-cut learning. | Phase 1, 10 | UNVERIFIED |
| A-15 | Filenames may encode recoverable metadata (IDs, augmentation tags, sequence numbers). | Could enable grouping; must be checked, not presumed. | Phase 1 | UNVERIFIED |
| A-16 | Source population, acquisition protocol and labelling provenance are unknown. | Bounds external validity — a limitation regardless of the audit outcome. | Phase 1 (likely unresolvable) | UNVERIFIED / possibly unverifiable |
| A-17 | The licensed use of the dataset permits this research use. | Governance and publication requirements. | Phase 1 (record licence) | UNVERIFIED |

---

## 14. Assumptions That MUST Be Verified in Phase 1

**Gate rule:** no downstream implementation decision (bag definition, split unit, preprocessing, model configuration, MRI functionality, or any risk/decision-support feature) may be finalised before the corresponding Phase 1 verification is complete. Where verification fails, the affected capability is documented as **not available** rather than approximated.

Mandatory Phase 1 verifications:

1. **V-1 — Image-level labels:** exact class folders, exact names/casing, count per class. (A-2, A-3)
2. **V-2 — Identifiers:** explicit, evidence-based yes/no for patient IDs, study IDs, lesion IDs, applied by active search. (A-6, A-7)
3. **V-3 — Masks/annotations:** presence or absence, by inspection. (A-8)
4. **V-4 — MRI linkage:** presence of any MRI file, report, or identifying key. (A-9) — feeds the Phase 12 feasibility gate.
5. **V-5 — Augmentation lineage:** whether rotated/sharpened variants are identifiable and groupable. (A-4, A-5)
6. **V-6 — Duplicates:** exact byte-level duplicates and corrupted/unreadable files. (A-5)
7. **V-7 — Metadata files:** any CSV/JSON/XML/README/licence inside the download. (A-10, A-17)
8. **V-8 — Image properties:** formats, dimensions, channels, colour space. (A-11)
9. **V-9 — Distribution:** class balance and total file counts. (A-12, A-13)
10. **V-10 — Filename patterns:** any recoverable structure. (A-15)
11. **V-11 — Explicit missing-information list:** a documented account of what the dataset does *not* contain. (A-16, A-10, A-8, A-9)

Phase 1's own validation checks require exactly this: a manifest whose row count equals the on-disk file count, labels matching actual structure, and an explicit "Confirmed Present" / "Confirmed Absent / Unknown" section for patient IDs, study IDs, lesion IDs, masks and MRI-related data.

---

## 15. Known Limitations (accepted up front)

1. **Single public dataset.** All results are dataset-specific; no external validation is planned or possible within scope.
2. **No confirmed MRI pairing (expected).** Cross-modal validation is expected to be non-demonstrable; that outcome will be documented rather than worked around.
3. **No confirmed patient/study identifiers (expected).** Patient-level grouping, and therefore a literal patient-level MIL bag, may be impossible. The fallback is the highest valid available grouping (source-image/augmentation family), which is *weaker* than a patient-level bag.
4. **Pre-augmented source data.** Rotation/sharpening variants inside the dataset mean naive splitting can leak. Leakage control is mandatory, not optional.
5. **Possible small sample size.** Limits statistical power; confidence intervals and explicit small-sample caveats are required rather than point estimates alone.
6. **Possible class imbalance.** Sensitivity/specificity balance must be reported honestly, not hidden behind accuracy.
7. **Label provenance unknown.** It is unverified whether labels are histopathology-confirmed or reader-assigned; label noise is therefore unquantified.
8. **Attention is not explanation.** Visualisations show where the model attended, not what is clinically relevant.
9. **No mask-based validation of attention (expected).** Without masks, explainability validation is qualitative only.
10. **No clinical metadata.** No risk stratification by clinical factors; no comparison to clinical standard of care.
11. **Bag-size dependence.** MIL benefit may be limited if bags are small or singleton; this is a documented risk, not a hidden one.
12. **No treatment support.** Any treatment-related content is out of scope (`PROGRESS.md` Phase 13 permits only guideline-sourced, clinician-review-labelled consideration text).
13. **Reproducibility ceiling.** Compute limits may restrict backbone choice; compromises must be documented.
14. **Publication/novelty risk.** Candidate contributions may already exist in the literature; novelty remains unverified and unclaimed.

---

## 16. Medical AI Safety Boundaries

The following boundaries are binding on every phase, artifact, report, UI, and demonstration. They derive from the `PROGRESS.md` Non-Negotiable Project Rules.

**The system must remain:**

- A **research prototype** — offline, experimental, and clearly labelled as such.
- **Not a clinically validated diagnostic device** — no clearance, no certification, no clinical claim.
- **Not a replacement for a radiologist or oncologist** — it does not diagnose and does not decide.
- **Not an autonomous treatment recommendation system** — it never prescribes, selects, doses, or withholds treatment.
- **Not permitted to fabricate MRI evidence** — no simulated MRI may enter any result, metric, or claim; any illustrative synthetic MRI must be unmistakably labelled "SYNTHETIC / ILLUSTRATIVE ONLY — NOT REAL PATIENT DATA" and kept out of evaluation.
- **Not permitted to fabricate patient information** — no invented IDs, demographics, histories or clinical findings.
- **Not permitted to claim clinical validation without real verified evidence** — and no such evidence is expected from this dataset.

**Additional enforced rules:**

- Attention weights must never be described as a clinical explanation or as a disease localisation.
- MRI-derived information, if it ever exists, may only be described as "cross-modal diagnostic consistency"/"concordance" — never as MRI "proving" the model correct.
- Every reported output must include a disclaimer, and risk categories must trace to documented thresholds.
- All outputs must state the limitations that apply to them.
- If a capability's precondition is unverified, the capability is **documented as unavailable**, not simulated.

---

## 17. Current MRI Uncertainty

**Status: MRI functionality is NOT available. Nothing MRI-related is implemented, and no MRI evidence exists in this repository.**

What is known:

- `PROGRESS.md` names MRI cross-modal validation as a project element (Phase 12).
- `PROGRESS.md` explicitly flags Phase 12 as "a high-risk area for fabrication" and states the primary Kaggle dataset is an ultrasound-only dataset.
- Phase 12's own feasibility gate states: if no patient IDs / MRI identifiers / MRI reports / paired examinations are confirmed, the project must document that "Cross-modal MRI validation cannot be clinically demonstrated using the primary Kaggle dataset alone."

What is unknown and must not be assumed:

- Whether any MRI file, report, or linking key exists in the dataset. (A-9, V-4)
- Whether any external MRI dataset could be *compatibly* linked — which would require shared patient/study identifiers that a public ultrasound dataset almost certainly does not provide.
- Whether a *statistical* (not patient-level) comparison against a separate, unrelated MRI cohort could ever be meaningful. This is an open methodological ambiguity (§24, Q-4); it must not be resolved by inventing linkage.

**Repository note:** `backend/services/mri_validation.py` exists as an empty (0-byte) placeholder from the initial scaffold. Its presence is **not** evidence of MRI capability, and it must not be treated as such.

**Consequence for the research question set:** the MRI question (`reports/research_questions.md`, RQ-6) is a *feasibility* question. Its honest Phase 0 answer is "undetermined"; a definitive answer is only possible after Phase 1 (data audit) and Phase 12.

---

## 18. Current Uncertainty About Patient / Study Identifiers

**Status: UNVERIFIED. No patient IDs, study IDs or lesion IDs are present in any artifact of this project, and none will be invented.**

- The dataset's known description covers images plus benign/malignant class folders. It is **not** evidence of identifiers.
- The dataset is described by its listing as pre-augmented (rotation, sharpening), which suggests image-level (not patient-level) provenance — but this is an inference, not a finding, and must not be relied on before inspection.
- Phase 1 must actively *search* for identifiers rather than assume absence; assuming absence is itself a violation of the dataset assumption policy.
- Consequences if identifiers are absent (expected): no patient-level bags, no patient-level risk estimate, no patient-level leakage test (only source-image/augmentation-family leakage tests), split unit falls back to the Phase 2 source-image group, and any downstream "patient-level" claim must be re-labelled or dropped.
- Consequence for language: without identifiers, results must be described at image- or group-level, never as patient-level performance.

---

## 19. Current Uncertainty About MIL Bag Construction

**Status: UNRESOLVED BY DESIGN. No bags exist and none will be created in Phase 0.**

`PROGRESS.md` Phase 3 defines the candidate bag formulations and requires that the chosen one be justified by actual Phase 1/2 evidence. At Phase 0, all candidates remain open:

| Candidate bag definition | Precondition | Phase 0 status |
|---|---|---|
| Patient → images | Confirmed patient IDs | Blocked pending V-2; expected unavailable |
| Study → images | Confirmed study IDs | Blocked pending V-2; expected unavailable |
| Lesion → image/patch instances | Confirmed lesion IDs/annotations | Blocked pending V-3; expected unavailable |
| Source-image group → augmented instance variants | Recoverable augmentation lineage / grouping | Plausible fallback; depends on V-5, V-6 |
| Single image → tiled patches as instances | No cross-image grouping needed | Alternative fallback; requires justifying that patches, not images, are instances |

Open questions carried forward (recorded, not answered):

1. If groupings are only probabilistic, what confidence threshold makes a bag defensible?
2. If bags are singleton or very small, does attention-based aggregation still carry meaning, or does the MIL framing become nominal? (A research-relevant risk, not a result.)
3. If instances are image patches, is the resulting "attention evidence" a lesion localisation or merely a texture preference? (Cannot be validated without masks.)
4. Is a bag's label guaranteed to be well-defined under the chosen definition? `PROGRESS.md` requires every bag to have a non-ambiguous label.

**Rule:** the MIL formulation must be derived from verified data structure. It will not be chosen for convenience, and labels will not be propagated from a fabricated grouping key.

---

## 20. Non-Negotiable Project Rules (from `PROGRESS.md`)

Reproduced here so every report is self-contained. These are binding and may not be relaxed.

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

Plus the file-level AI instructions in `PROGRESS.md` (read before changes; work only on the authorized phase; never skip phases; never assume unverified metadata; never introduce leakage; run validation before declaring a phase complete; update `PROGRESS.md` only when exit criteria are genuinely satisfied; stop after the requested phase; preserve existing repository structure).

---

## 21. Technical Success Criteria

These are **criteria**, not results. None is satisfied yet.

| ID | Criterion | Evidence required |
|---|---|---|
| TS-1 | Dataset audit complete and honest: manifest covers 100% of discovered images; row count equals on-disk file count. | Phase 1 manifest + audit report |
| TS-2 | Every image is assigned to a source-image/augmentation group (singletons permitted). | Phase 2 grouping manifest |
| TS-3 | Bag definition documented, traced to the manifests, with a non-ambiguous label for every bag and bag-size statistics reported. | Phase 3 report + bag manifest |
| TS-4 | All leakage tests pass with **zero** overlap across splits (source-image, duplicate, near-duplicate, augmentation-family; patient/study only if applicable). | Phase 4 automated test output |
| TS-5 | Test set frozen (date + hash) and not used for tuning at any point. | Phase 4 freeze record; Phase 9/10 audit |
| TS-6 | Baseline suite implemented and evaluated on identical splits and preprocessing as the main model. | Phase 6 results |
| TS-7 | Dual Attention MIL implemented with two genuinely distinct attention stages, mathematically documented, returning attention weights, and passing shape/forward-pass unit tests. | Phase 8 docs + tests |
| TS-8 | All models evaluated on the frozen test set, once, reporting accuracy, sensitivity, specificity, precision, recall, F1, ROC-AUC, PR-AUC — never accuracy alone. | Phase 10 report |
| TS-9 | Comparative result reported honestly, **including the case where Dual Attention MIL does not outperform baselines**. A negative or inconclusive result satisfies this criterion. | Phase 10 discussion |
| TS-10 | Error analysis complete: false-positive, false-negative, difficult-case and class-wise analysis. | Phase 10 report |
| TS-11 | Attention visualisations produced for bags of varying size, with the "attention ≠ clinical explanation" disclaimer; explicit statement whether mask-based validation was possible. | Phase 11 report |
| TS-12 | MRI feasibility stated definitively, with no fabricated MRI evidence anywhere in the project. | Phase 12 report |
| TS-13 | Training reproducible: seeds fixed and documented; one experiment re-run compared within expected variance; best-model selection provably never references test metrics. | Phase 9 logs + checks |
| TS-14 | End-to-end inference runs on sample inputs through the demo application. | Phase 14 test |

**Non-criteria (explicitly not required for success):** achieving any particular AUC, beating baselines, achieving SOTA, or producing a clinically usable tool.

---

## 22. Project Success Criteria & Completion Criteria

**Project success criteria (from `PROGRESS.md` Phase 0 task list and the roadmap):**

1. All phases 0–14 completed with their exit criteria satisfied, or explicitly documented as not achievable with the available data.
2. A working end-to-end demonstration application (Phase 14).
3. Full reproducibility: documented configs, seeds, splits, manifests, and experiment logs.
4. Documented limitations, disclosed prominently rather than buried.
5. Every result clearly distinguished from clinical validation.
6. A final report reviewed against all 15 Non-Negotiable Project Rules.

**Project completion criteria (definition of "done"):**

- Phases 1–11, 13, 14 have satisfied exit criteria; **and**
- Phase 12 has produced either a verified-executed concordance module (only against real paired data) **or** a documented "not clinically demonstrable with current data" conclusion with an unexecuted design spec; **and**
- The final report and README contain no unsupported clinical claim and state the research-prototype classification; **and**
- The demonstration materials accurately reflect what was and was not achieved — including the MRI outcome.

**Honesty clause:** completing the roadmap with a null or negative finding counts as project success. Fabricating a positive finding, or overstating a result, is project failure.

---

## 23. Literature Review Requirements (Phase 0)

Phase 0 requires literature review to be **begun and tracked**. The tracking document is `reports/phase0_literature_review.md`.

**Required coverage (from `PROGRESS.md` and the Phase 0 brief):**

1. CNN-based breast ultrasound classifiers.
2. Transfer-learning approaches for breast ultrasound.
3. MIL in medical imaging.
4. Attention-based MIL (ABMIL-style architectures) — and specifically **what "dual attention" already exists in prior work**.
5. Existing dual-attention approaches (medical and general vision).
6. Multimodal breast imaging systems (ultrasound + MRI, ultrasound + mammography).
7. Ultrasound + MRI systems.
8. Ultrasound/MRI concordance or consistency systems.
9. Explainable breast cancer AI (Grad-CAM, attention maps, saliency methods).
10. Grad-CAM and related visualisation methods.

**Required output:** a novelty-positioning summary comparing this project's design choices against the above.

**Standing rule on novelty:** none of the seven candidate contributions in `PROGRESS.md` — Dual Attention MIL for breast ultrasound classification; instance-level evidence aggregation; attention-guided ultrasound evidence visualization; ultrasound/MRI cross-modal validation; ultrasound/MRI concordance analysis; patient/study-level risk estimation; explainable clinical decision support — may be described as novel, publishable, or state-of-the-art. They remain **candidate contributions with novelty "to be verified"**. A search returning no direct hit is **not** evidence of novelty.

---

## 24. Ambiguities and Open Decisions (documented, NOT silently resolved)

These require human decisions. Per the Q-2 authority resolution (2026-09-16; `reports/phase0_decisions.md` §4), the decision authority for project-level technical decisions is the **project owner** (external advisor/reviewer approval is not required unless a specific item explicitly demands it). They are recorded here rather than assumed away.

| ID | Ambiguity | Why it blocks | Options | Owner |
|---|---|---|---|---|
| Q-1 | **Repository layout mismatch.** `PROGRESS.md`'s illustrative structure uses `data/`, `src/data`, `src/models`, `src/mil`, `configs/`, `app/`, `tests/`. The actual repository uses `backend/`, `model/da_mil/`, `frontend/`, `notebooks/`, `docs/`, `reports/`, `dataset/`. Later phases of `PROGRESS.md` reference paths that do not exist (e.g. `src/data/audit.py`, `data/manifests/`). | Phase 1 cannot create its files without a decision. `PROGRESS.md` says "Adapt to any pre-existing repository structure. Do not overwrite or reorganize existing files unnecessarily." | (a) Adopt `PROGRESS.md`'s `src/…` + `data/…` layout wholesale; (b) map `PROGRESS.md`'s modules onto the existing `model/`/`backend/` layout and record the mapping; (c) hybrid — use `src/` for research code and keep `backend/` for the eventual API. → **Investigated 2026-09-16: resolution D-1 = option (c)-hybrid with full mapping table** (`reports/phase0_decisions.md` §1) — **APPROVED by the project owner 2026-09-16, in effect** (physical creation deferred to Phase 1). | Project owner |
| Q-2 | **Who approves Phase 0?** The exit criterion is "documented and approved by project team/advisor". | Phase 0 cannot be marked complete without it. | Name the approver and record approval in `PROGRESS.md`. → **Resolved and closed 2026-09-16: project-owner approval suffices for project-level technical decisions, and the owner approved the Phase 0 deliverables as a whole** (`reports/phase0_decisions.md` §4); Phase 0 marked Complete accordingly. | Project owner |
| Q-3 | **Pre-existing empty scaffold files.** `backend/services/mri_validation.py`, `backend/services/report_service.py`, `backend/services/ultrasound_prediction.py`, `backend/router/*.py`, `backend/schemas.py`, `backend/main.py`, `backend/database.py`, `model/da_mil/{model,attention,dataset}.py` and `frontend/package.json` exist but are empty (0 bytes). | They hint at an intended backend/frontend architecture that the roadmap does not describe. An empty `mri_validation.py` in particular risks being misread as MRI capability. | (a) Keep as placeholders and document their intended role; (b) delete them and follow the roadmap; (c) flesh them out in later phases under the roadmap's own file names. | Project owner |
| Q-4 | **Meaning of "cross-modal validation" given unlinkable data.** If the ultrasound dataset has no MRI linkage, is a *statistical comparison* against an unrelated external MRI cohort ever acceptable, and what would it establish? | Determines whether Phase 12 has any executable path at all. | (a) Treat any unlinked comparison as out of scope and report "not demonstrable"; (b) permit an explicitly-labelled, non-paired, population-level comparison as a discussion-only exercise, clearly marked as not patient-level validation. → **Investigated 2026-09-16: resolution D-2 = five-label taxonomy (T-1 paired cross-modal validation / T-2 independent external validation / T-3 modality-transfer & domain-shift analysis / T-4 synthetic-data experimentation / T-5 absence of MRI validation) with per-state definitions for evidence states A–E** (`reports/phase0_decisions.md` §2) — **APPROVED by the project owner 2026-09-16, in effect as binding terminology**. | Project owner |
| Q-5 | **Title inconsistency.** `README.md` says "Multi-Modal Images"; `PROGRESS.md` says "Ultrasound Images with Cross-Modal Validation". | Final deliverables must be titled consistently. | Align `README.md` to the `PROGRESS.md` title (a later-phase documentation task). | Project owner |
| Q-6 | **"Patient/study-level risk estimation" as a candidate contribution** while patient identifiers may not exist. | Contradiction between candidate contribution #6 and expected data. | (a) Restrict to group/label-level risk framing; (b) re-scope after Phase 1; (c) retain as aspirational and mark unavailable. | Project owner |
| Q-7 | **Synthetic/demo data policy.** `PROGRESS.md` forbids simulated MRI in evaluation but contemplates unmistakably-labelled illustrative synthetic data. | Determines whether the Phase 14 UI can show any placeholder MRI panel at all. | (a) No synthetic MRI anywhere; UI shows "MRI: unavailable"; (b) permit clearly-labelled synthetic illustration strictly outside evaluation. | Project owner |
| Q-8 | **External dataset approval process.** Phase 12 may research external MRI sources but not integrate them without approval. | Determines whether Phase 12 can ever exceed "not demonstrable". | Define who approves, and what evidence a candidate source must meet (licence, linkage key, ethics). | Project owner |

**Status summary (2026-09-16):** Q-1 and Q-4 — **resolved and approved by the project owner** (D-1, D-2 in effect; see `reports/phase0_decisions.md` §4). Q-2 — **resolved and closed**: project-owner approval suffices for project-level technical decisions, and the owner approved the Phase 0 deliverables as a whole the same day (Phase 0 marked Complete). Q-7 and Q-8 — **deliberately deferred by owner instruction** until synthetic MRI or an external MRI dataset actually becomes relevant (expected Phase 12/14); standing roadmap rules apply until then. Q-3, Q-5, Q-6 — still open (non-blocking for Phase 1). No assumption has been made in place of any unresolved item.

---

## 25. Phase 0 Validation Record and Status

### 25.1 Validation checks required by `PROGRESS.md`

| Check | Result | Evidence |
|---|---|---|
| Scope document does not contain clinical deployment claims. | **PASS** | §3, §4, §13, §16 and §22 explicitly classify the project as a research prototype and prohibit deployment/diagnostic language; §12 enumerates prohibited uses. |
| Research questions are falsifiable/testable. | **PASS** | `reports/research_questions.md` gives each question a null hypothesis, refutation condition and required evidence; §"Falsifiability review" in that file maps each one. RQ-6 (MRI feasibility) is testable as a definitive feasibility determination even though its Phase 0 answer is undetermined. |

### 25.2 Phase 0 tasks (`PROGRESS.md`) — completion record

| Task | State |
|---|---|
| Write formal objectives (technical + research) | Done — §5, §6 |
| Write research questions | Done — `reports/research_questions.md` |
| Define input | Done — §7 |
| Define output | Done — §8 |
| Define prediction target(s) | Done — §9 |
| Define scope boundaries | Done — §11 |
| Document assumptions | Done — §13, §14 |
| Document known limitations | Done — §15 |
| Define medical AI safety boundaries | Done — §16 |
| Define technical success criteria | Done — §21 |
| Define project success criteria | Done — §22 |
| Write the "Research Prototype vs. Clinically Validated System" distinction into the report | Done — §4 |
| Begin literature review tracking | Done — `reports/phase0_literature_review.md` (tracking begun; **surveys not yet performed**) |

### 25.3 Exit criteria — SATISFIED (2026-09-16)

`PROGRESS.md` Phase 0 exit criterion: *"Scope, objectives, and boundaries documented and approved by project team/advisor."*

- **Documented:** yes — `reports/phase0_scope.md` (this document), `reports/research_questions.md`, `reports/phase0_literature_review.md`, `reports/phase0_decisions.md`.
- **Approved:** **yes — the project owner approved the Phase 0 deliverables as a whole on 2026-09-16** (approval authority confirmed: project-owner approval suffices for project-level technical decisions; no external advisor/reviewer approval required — see `reports/phase0_decisions.md` §4). Decisions D-1 and D-2 were approved separately the same day and are in effect.

Therefore, per `PROGRESS.md` rule 12 ("Update PROGRESS.md only when the phase exit criteria have genuinely been satisfied") and its final instruction ("Do not mark any phase or task complete until its exit criteria and validation checks are genuinely satisfied"):

> **Phase 0 is COMPLETE** (2026-09-16): both validation checks PASS, all 13 tasks done, exit criterion satisfied by project-owner approval. Phase 1 has NOT been started and awaits the owner's explicit instruction. Open items carried forward deliberately: Q-3, Q-5, Q-6 (open, non-blocking) and Q-7/Q-8 (deferred to Phase 12/14 relevance); literature surveys ongoing.

### 25.4 What Phase 1 must resolve before downstream decisions

Phase 0 defines scope but **cannot** confirm any dataset property. Phase 1 must verify §14 (V-1 … V-11) before any decision on bag definition (Phase 3), split unit (Phase 4), preprocessing (Phase 5), model configuration (Phases 6–8), MRI functionality (Phase 12), or risk/decision-support features (Phase 13) is finalised. Where a verification fails, the dependent capability must be documented as unavailable rather than approximated. Expected outcome: several capabilities in the original brief will be re-scoped or dropped.

---

*End of Phase 0 scope document. APPROVED by the project owner (2026-09-16); Phase 0 Complete. No clinical validation is claimed or implied anywhere in this document.*
