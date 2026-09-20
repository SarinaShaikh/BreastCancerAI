# Phase 13 — Research Decision-Support Layer: Specification & Implementation Report

**Status:** Implementation complete per the owner-approved margin-based scope — uncommitted, awaiting final audit and separate documentation/commit authorizations.
**Date:** 2026-09-21
**Verdict carried forward:** Phase 12 — **NOT CURRENTLY FEASIBLE** for patient-level paired ultrasound↔MRI validation with current data; therefore every Phase 13 output carries `cross_modal_evidence: UNAVAILABLE`.
**Boundary headline:** This is a **research-oriented presentation layer over a frozen research model**. It is **NOT a clinical diagnostic system** and **NOT clinical risk stratification**.

---

## 1. Objective

Per the owner-approved scope: build a research-oriented ultrasound decision-support layer around the frozen E1 Dual Attention MIL model providing (a) benign/malignant model prediction, (b) malignant-class model score, (c) explicit prediction-confidence information, (d) a margin-based uncertainty flag, (e) top-k model-attribution evidence, (f) a research-only suggested action, and (g) an explicit cross-modal evidence status — **never representing these outputs as clinically validated risk stratification**.

## 2. Approved scope (owner authorization, 2026-09-21)

Margin-based uncertainty path only: fixed threshold 0.5, pre-registered margin 0.1, no calibration of any kind, no new data sources, no dashboard/UI integration (Phase 14), no clinical-capability claims. All of §15's exclusions (clinical risk scores, BI-RADS, staging, grading, prognosis, recurrence, survival, treatment recommendation, personalized/patient-specific outputs, triage, clinical decision automation, MRI validation, cross-modal fusion, concordance/discordance, Platt/temperature/isotonic calibration, conformal prediction, external clinical metadata, patient-level risk profiles, longitudinal tracking, dashboard/UI) are **implemented nowhere** and are enforced by the Phase 13 test suite (T13-12, T13-15).

## 3. Exact score terminology (owner decision §4)

| Approved term | Meaning |
| --- | --- |
| `malignant_class_model_score` | The raw sigmoid output of frozen E1 for the malignant class. **Not calibrated.** |
| `model_estimated_malignancy_probability (not calibrated)` | Permitted descriptive paraphrase of the same quantity. |
| `prediction_confidence` | Model-confidence wording derived from the margin rule (`high` / `borderline (within uncertainty margin)`). Model confidence only — **never clinical certainty**. |
| `uncertainty_flag` | Boolean: model-score proximity to the decision threshold ONLY. |

Prohibited as capability claims (allowed only inside negative/disclaimer statements): clinical risk score, cancer risk score, calibrated clinical probability, patient risk, clinical risk stratification. The implemented code contains **zero** occurrences of the prohibited terms outside the sanctioned disclaimer/negation contexts (T13-12, AST-verified over identifiers and string constants).

## 4. Prediction rule

`score < 0.5 → Benign model prediction`; `score ≥ 0.5 → Malignant model prediction`. The threshold is the **pre-registered frozen Phase 9 constant 0.5** — never fitted, tuned, or re-derived from any split (explicitly not from the test set). Labels are "Benign/Malignant **model prediction**" — a model output, not a diagnosis (T13-01..T13-03; T13-03 additionally verifies the frozen persisted Phase 9 predictions agree with this mapping exactly).

## 5. Uncertainty rule — and why this is NOT calibration

**Rule (pre-registered):** `uncertain = abs(model_score − 0.5) ≤ 0.1` (margin `0.1`, the established Phase 10/11 convention; `metrics.uncertainty_mask` default margin and Phase 11's `UNCERTAINTY_MARGIN` are both 0.1 — T13-07 verifies the heritage). Boundary behaviour is documented and tested: 0.5 → uncertain; 0.4 and 0.6 (exact margin edges, handled with a 1e-9 exact-decimal tolerance) → uncertain; strictly beyond → not uncertain (T13-04..T13-06).

**Why this is not calibration:** no reliability diagram, no Brier score, no expected calibration error, no temperature/Platt/isotonic scaling, no conformal method is computed or applied anywhere in Phase 13 (T13-15 verifies no calibration machinery is even imported). The margin rule is a **fixed, pre-registered presentation flag** derived from the Phase 10/11 convention — not a statistical calibration procedure, not fitted to any data, and carrying no probabilistic interpretation. Consequently the module's provenance states `score_is_calibrated: false`, and the disclaimer states the score is uncalibrated.

**Why this is not clinical risk stratification:** the score comes from a research model trained on 9,016 identifier-free, augmentation-heavy ultrasound images with directory-encoded benign/malignant labels only; it is underperforming on the frozen test set (Phase 10: E1 bag-level ROC-AUC 0.530); its probabilities are severely compressed (48/74 validation bags — 65% — and 51/73 test bags lie within ±0.1 of 0.5); and no clinical outcome, demographic, or follow-up variable exists to anchor any risk semantics. A "Low/Moderate/High" banding of such scores would be presentation, not risk. The uncertainty flag expresses **model prediction proximity to a decision threshold only** — it is NOT clinical uncertainty, NOT diagnostic uncertainty, and NOT patient/cancer risk.

## 6. Top-k attribution semantics

Evidence rows are **model-attributed image instances**: deterministic descending stage-2 MIL attention order, ties broken by ascending instance order, `k > n` clamping — all delegated verbatim to the Phase 11 helpers (`rank_instances` / `top_k_instances`; T13-11). Default `top_k = 3` (fixed a priori). Every row is annotated: *"model-attributed image instance (attention weight); not a tumor/cancer/lesion region and not a clinical explanation"*. Phase 11 semantics are preserved untouched: **Model Attention ≠ Clinical Explanation** — attention/gates/Grad-CAM are model attribution mechanisms, not clinical explanations, and are not validated as lesion localization (no ground-truth masks exist).

## 7. Cross-modal evidence field

`cross_modal_evidence: "UNAVAILABLE"` — a hard-coded module constant (T13-14). Concordant/discordant values are never produced; MRI is never imported (T13-15); no MRI data, synthetic MRI, inferred MRI findings, or unrelated-cohort "evidence" exists. This encodes the authoritative Phase 12 outcome.

## 8. Suggested-action semantics

The only produced value is the fixed research safety flag **"Clinical review recommended"**. It is NOT treatment recommendation, medical advice, triage, referral recommendation, diagnosis, or an emergency recommendation; the disclaimer explains this distinction explicitly.

## 9. Mandatory disclaimer (every output)

`research_only_disclaimer` is embedded verbatim in every record (T13-13 asserts every required element): research prototype; not clinically validated; score is raw and **uncalibrated**; uncertainty flag = score proximity to the fixed 0.5 threshold only (not clinical/diagnostic uncertainty, not patient/cancer risk); attention/Grad-CAM are model attributions, not clinical explanations (**Model Attention ≠ Clinical Explanation**), not validated as lesion localization; no MRI/cross-modal evidence available (Phase 12); must not be used as diagnosis, clinical risk assessment, or treatment recommendation; does not replace radiologist or pathologist judgment.

## 10. Output schema

Fixed field set (`OUTPUT_FIELDS`, exact-set asserted by T13-08): `prediction`, `malignant_class_model_score`, `prediction_confidence`, `uncertainty_flag`, `uncertainty_margin`, `top_k_evidence`, `cross_modal_evidence`, `suggested_action`, `research_only_disclaimer`, `provenance`. Provenance carries: E1 checkpoint relpath + MD5, decision threshold, `score_is_calibrated: false`, the uncertainty rule text, attribution source + note, `source_key_status = inferred_from_filename_not_verified_identifier`, and (when supplied) `bag_id`, `source_group_id` (annotated as a filename-derived grouping identifier, NOT a patient/study/lesion identifier), `split`, and `attention_available`. Records are pure functions of their inputs — byte-identical on reconstruction (T13-10).

## 11. Inputs and frozen identity

Inputs are frozen artifacts only: the frozen E1 checkpoint `model/dual_attention/da_stage_b/best.pt` (**MD5 `258710649fb0e6979f64fc1e7ccfc28f`**, verified through the Phase 11 helper — T13-16), the Phase 5.5 bag dataset interface, the Phase 11 attribution infrastructure and attention export (88 rows, integrity re-asserted — T13-17), and the frozen Phase 9 persisted predictions (`val_predictions.csv` / `test_predictions.csv`; test artifacts used for provenance verification only — never re-evaluated). No new clinical data, no fabricated metadata. The frozen test manifest sha256 `959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74` is re-verified by T13-18. No new inference is required: scores come from the frozen persisted predictions; attention comes from the Phase 11 export or the frozen model via Phase 11 helpers.

## 12. Dataset / model limitations (must accompany any use)

Identifier-free dataset (no verified patient/study/lesion/accession identifiers; bags are filename-derived groups) · benign/malignant directory labels with undocumented pathology grounding · augmentation-heavy, near-duplicate-prone images with two source keys spanning the pre-existing train/val split (Phase 1, owner-accepted as-is) · E1 test underperformance (ROC-AUC 0.530; CM TP 14/TN 29/FP 14/FN 16) · compressed probabilities (65% of val bags within ±0.1 of 0.5) · no calibration · small test set (73 bags) → wide uncertainty on all reported metrics · research prototype only, NOT clinical validation.

## 13. Test strategy

`tests/test_phase13_decision_support.py` — **18 checks, 18/18 PASS** (runner convention identical to the earlier phase suites; CPU-only, deterministic; no test-set inference, no training, no final test evaluation; the only writes are none — the suite is read-only). Coverage: frozen threshold + determinism + agreement with persisted predictions (T13-01..03); margin boundary behaviour incl. exact 0.4/0.6 edges (T13-04..07); schema exactness, provenance identity, end-to-end records over all 74 frozen val bags, byte-identical reconstruction (T13-08..10); deterministic top-k with ties + k>n clamping (T13-11); clinical-claim guards — AST scan of identifiers/string literals for banned capability terms and disclaimer completeness (T13-12..13); cross-modal UNAVAILABLE + no MRI/calibration imports (T13-14..15); frozen integrity — checkpoint MD5, Phase 11 export, Phase 9 artifacts, manifest SHA (T13-16..18).

## 14. Explicit out-of-scope capabilities

All items in §2's exclusion list. Additionally deferred by design: `PROGRESS.md`/README updates (separate authorization), dashboard/UI integration (Phase 14, whose "MRI validation (if available)" pipeline step resolves to the documented `UNAVAILABLE` branch), and any calibration layer (would require a separate owner-authorized, validation-only protocol; the current 74-bag validation set limits calibration reliability and any such step must state so).

## 15. Boundary statement and next gates

The Phase 12 MRI feasibility verdict (NOT CURRENTLY FEASIBLE; `reports/phase12_mri_feasibility_report.md`) is the authoritative cross-modal boundary and is encoded operationally in every Phase 13 output via `cross_modal_evidence: UNAVAILABLE`; it is not restated or re-derived here. This report is the Phase 13 specification/implementation record only. Remaining owner-gated steps, none executed: read-only implementation audit → documentation update (`PROGRESS.md`/README) → commit authorization → push authorization. No clinical validation is claimed anywhere in this document; the implemented layer is a research prototype presentation over frozen research artifacts.
