# Phase 0 — Research Questions

**Deliverable:** `reports/research_questions.md`
**Phase:** 0 (per `PROGRESS.md`)
**Status:** DRAFT — questions defined, **none answered**. Phase 0 defines questions; it does not produce findings.
**Date recorded:** 2026-09-16
**Authority:** `PROGRESS.md`. Where this document conflicts with `PROGRESS.md`, `PROGRESS.md` wins.

> **Scope of this document.** Define clear, testable and **falsifiable** research questions for the project, together with null hypotheses, refutation conditions and required evidence. It contains **no results, no metrics, no dataset findings and no novelty claims.**

---

## 0. Rules Governing These Questions

1. **No question is answered here.** Answers are produced only in Phases 6–12, and only from leakage-free evaluation.
2. **Hypotheses are labelled.** Every directional expectation below is marked `HYPOTHESIS (untested)`. None is a fact, an expectation of success, or a promise of improvement.
3. **Null results are legitimate outcomes.** A finding that Dual Attention MIL does *not* outperform baselines is a valid, reportable project result (`reports/phase0_scope.md` §21, TS-9).
4. **No novelty claim.** Nothing here asserts that any method is novel or state-of-the-art. Novelty for all seven candidate contributions in `PROGRESS.md` remains **"to be verified"** by the literature review (`reports/phase0_literature_review.md`).
5. **Nothing may be assumed about the data.** Every question that depends on dataset properties is gated by Phase 1 verification (`reports/phase0_scope.md` §14, V-1…V-11).
6. **Pre-registration.** The primary metric and the operating point for the headline comparison must be fixed *before* the frozen test set is evaluated (Phase 10), so that the headline result cannot be selected post hoc.
7. **Metrics discipline.** Accuracy alone never answers a question. Sensitivity, specificity, ROC-AUC, PR-AUC and F1 — with confidence intervals where sample size allows — are required.
8. **Leakage discipline.** No question may be answered using splits that have not passed the Phase 4 leakage tests.

---

## 1. Question Register (overview)

| ID | Question (short form) | Type | Primary phase |
|---|---|---|---|
| RQ-1 | Does Dual Attention MIL differ from non-MIL image-level baselines in bag-level classification? | Comparative | 6, 9, 10 |
| RQ-2 | Does Dual Attention MIL differ from a standard / single-attention MIL baseline? | Comparative | 7, 9, 10 |
| RQ-3 | How do sensitivity, specificity, ROC-AUC, PR-AUC and F1 compare across all models? | Comparative / descriptive | 10 |
| RQ-4 | Does instance-level evidence aggregation provide useful information beyond an image-level classification? | Methodological | 7, 11 |
| RQ-5 | How does performance change under leakage-free source-image / augmentation-family splitting versus naive image-level splitting? | Methodological / integrity | 4, 10 |
| RQ-6 | Is reliable MRI cross-modal validation feasible with the available data — and what can and cannot be established about ultrasound/MRI concordance? | Feasibility | 1, 12 |
| RQ-7 | What limitations arise from the available dataset, and which conclusions can the data support? | Scoping / validity | 1–14 |
| RQ-8 | Can attention-based evidence be visualized meaningfully, and what would falsify that claim? | Methodological | 11 |
| RQ-9 | How sensitive is Dual Attention MIL to bag size and variable instance counts? | Robustness | 7, 8, 10 |
| RQ-10 | Do the model's probabilities support calibrated, threshold-derived risk categories? | Exploratory | 10, 13 |

This register operationalises `PROGRESS.md`'s Phase 0 example question — *"Does dual attention improve bag-level classification over single-attention MIL and non-MIL baselines on this dataset?"* — as the separable, individually falsifiable questions RQ-1 and RQ-2.

---

## 2. Detailed Questions

Every entry states: the question, why it matters, the untested hypothesis, the null hypothesis, what would refute the hypothesis, what evidence is required, which data it needs, and the threats to its validity.

---

### RQ-1 — Dual Attention MIL vs. non-MIL baselines

**Question.** On leakage-free splits of this dataset, does a Dual Attention MIL model differ in bag-level classification performance from non-MIL image-level baselines (a simple CNN, and a transfer-learning backbone classifier, with image-to-bag aggregation by majority vote or equivalent)?

**Why it matters.** This is the project's central comparative claim. Without a non-MIL baseline trained on identical splits and preprocessing, any MIL result is uninterpretable.

**H1-1 — `HYPOTHESIS (untested)`:** Dual Attention MIL will achieve higher ROC-AUC than non-MIL baselines at the bag level.
**H0-1:** There is no difference in bag-level ROC-AUC between Dual Attention MIL and non-MIL baselines.

**What would refute H1-1.** Overlapping confidence intervals, or a difference whose sign is not consistent across seeds, or a Dual Attention MIL deficit — any of these refutes H1-1 for this dataset. A null result answers the question: it will be reported as such.

**Required evidence.** All models trained and evaluated on identical splits, identical preprocessing, identical bag definitions, and the same frozen test set, with ROC-AUC, PR-AUC, sensitivity, specificity and F1 reported alongside confidence intervals; multiple seeds where compute allows.

**Data required.** Existing class labels; a valid bag definition from Phase 3. *(Gated by Phase 1 V-1, Phase 3.)*

**Threats to validity.** Small test-set size limiting statistical power; class imbalance distorting threshold-based metrics; uncontrolled image-quality confounds; differences arising from hyperparameter effort rather than architecture (the main model must not receive materially more tuning budget than baselines).

---

### RQ-2 — Dual Attention MIL vs. standard / single-attention MIL

**Question.** Does the second attention stage of the Dual Attention architecture produce a measurable difference in bag-level classification compared with a standard single-attention MIL baseline (and, where implemented, a mean/attention-pooling baseline), when everything else is held constant?

**Why it matters.** This isolates the architectural contribution. Without it, an improvement could be attributable to MIL in general rather than to dual attention specifically.

**H1-2 — `HYPOTHESIS (untested)`:** Dual Attention MIL will differ measurably from the single-attention MIL baseline.
**H0-2:** Dual Attention MIL performs equivalently to the single-attention MIL baseline.

**What would refute H1-2.** Equivalent performance within confidence intervals, or a single-attention advantage, refutes the added value of the second stage under these conditions. That is a reportable, informative result — and it must be reported rather than tuned away.

**Required evidence.** The two models matched on backbone, feature dimensionality, bag definition, splits, preprocessing, optimizer family, training budget and early-stopping rule; ablations removing each attention stage; attention-weight outputs from both stages.

**Data required.** As RQ-1.

**Threats to validity.** Extra parameters alone can shift performance (a capacity effect, not an attention effect) — parameter-matched controls are required to interpret any difference. Also: the second attention stage may be under-trained on small bags, confounding RQ-9 with RQ-2.

---

### RQ-3 — Metric comparison across models

**Question.** How do sensitivity, specificity, ROC-AUC, PR-AUC, F1 (plus precision, recall, accuracy, PPV/NPV where derivable) compare across the simple CNN, transfer-learning CNN, single-attention MIL and Dual Attention MIL, both at a common operating point and across the full threshold range?

**Why it matters.** `PROGRESS.md` rule 7 forbids reporting accuracy alone, and in a possibly-imbalanced dataset accuracy is the least informative metric. Sensitivity (missed malignancy) and specificity (unnecessary intervention) carry asymmetric clinical weight and must both be shown.

**H1-3 — `HYPOTHESIS (untested)`:** ROC-AUC and PR-AUC will rank the models consistently with the sensitivity/specificity trade-off at the pre-registered operating point.
**H0-3:** Model rankings will not be consistent across metrics and thresholds.

**What would refute H1-3.** Rank reversal between ROC-AUC and PR-AUC, or a model that wins on AUC but loses on sensitivity at the clinically relevant operating point. Such a discordance refutes any single-metric success claim and must be surfaced.

**Required evidence.** Full metric suite per model with confidence intervals, confusion matrices, ROC and PR curves, and an explicitly documented, pre-registered operating-point rule.

**Data required.** Frozen test set (Phase 4), trained models (Phase 9). *(Gated by Phase 1 V-9 for class balance.)*

**Threats to validity.** Imbalance making PR-AUC unstable; small-sample CIs that overlap across all models (which would mean the question is *unresolved* at this sample size — a limitation to state, not to hide); threshold choice materially changing the ranking.

---

### RQ-4 — Value of instance-level evidence aggregation

**Question.** Does aggregating instance-level evidence provide information that an image-level classification does not — specifically: (a) is bag-level performance at least comparable to image-level performance, (b) are instance attention weights per bag non-degenerate (i.e., not uniform or collapsed onto a single instance without reason), and (c) do the top-k attended instances carry inspectable image content?

**Why it matters.** Instance-level evidence aggregation is candidate contribution #2 in `PROGRESS.md` and is the substrate of the explainability claim. If aggregation adds nothing beyond an image-level score, that must be stated plainly.

**H1-4 — `HYPOTHESIS (untested)`:** bag-level aggregation will be non-inferior to image-level classification, and attention distributions will be non-degenerate.
**H0-4:** Aggregation offers no measurable benefit, and/or attention distributions are degenerate.

**What would refute H1-4.** Uniform attention across all instances (aggregation collapses to mean pooling in practice), attention that is unstable across seeds for the same bag, or performance degradation at bag level. Any of these refutes the "useful evidence" claim for this dataset.

**Required evidence.** Attention-weight distributions per bag with diversity statistics; stability across seeds; agreement between top-k instance sets; bag-level vs. image-level metric comparison; documented bag-size statistics.

**Data required.** Bag definition (Phase 3), models (Phase 9). **Mask-based quantitative validation is only possible if Phase 1 confirms masks exist (V-3) — expected absent.** Without masks, this question can only be *partially* answered, and that limitation must be stated.

**Threats to validity.** Small or singleton bags make aggregation meaningless (see RQ-9), so a null result here may reflect bag structure rather than the method; this confound must be documented.

---

### RQ-5 — Effect of leakage-free splitting on measured performance

**Question.** How does measured model performance differ between (a) leakage-free splitting at the source-image/augmentation-family level and (b) naive image-level splitting that permits augmented variants of the same source image to cross splits?

**Why it matters.** The dataset is described as containing rotation/sharpening-augmented images. If augmented siblings cross splits, reported metrics are inflated and every comparative conclusion (RQ-1, RQ-2) is invalid. This question quantifies the size of that risk rather than asserting it. It is also a methodological contribution candidate with a clear, honest finding regardless of direction.

**H1-5 — `HYPOTHESIS (untested)`:** naive image-level splitting will produce optimistically biased performance relative to group-level splitting, with the gap largest for models with the most capacity for memorisation.
**H0-5:** Split design has no measurable effect on reported performance.

**What would refute H1-5.** A negligible or absent gap, or a gap with the opposite sign. Either outcome refutes the stated hypothesis — and remains a valuable, reportable finding about this dataset.

**Required evidence.** Phase 2 grouping manifest; Phase 4 leakage test results (zero overlap in the leakage-free condition); identical models and seeds evaluated under both split designs; the same metric suite as RQ-3.

**Data required.** Grouping must be recoverable enough to form groups. **If lineage is unrecoverable, this question becomes unanswerable in its strong form and must be downgraded to a documented limitation, not estimated with invented groups.**

**Threats to validity.** Imperfect grouping (perceptual hashing over- or under-grouping) makes the "leakage-free" condition only as good as the grouping; grouping confidence must be reported. Any doubt about grouping weakens both arms of the comparison.

---

### RQ-6 — MRI cross-modal validation feasibility

**Question.** Is reliable MRI cross-modal validation feasible with the available data? Concretely: (a) does the primary dataset contain any patient IDs, MRI identifiers, MRI reports, or paired MRI examinations; (b) can any external MRI source be **verifiably linked** at the patient/study level to these ultrasound images; and (c) what can and cannot be established about ultrasound/MRI concordance given the verified answer to (a) and (b)?

**Why it matters.** `PROGRESS.md` names Phase 12 "a high-risk area for fabrication" and mandates a feasibility gate. This question determines whether cross-modal validation is a real capability or a documented limitation.

**H1-6 — `HYPOTHESIS (untested, and expected to be refuted):`** the primary dataset will provide no patient-level or MRI-level linkage sufficient for paired cross-modal validation.
**H0-6 / alternative:** sufficient linkage exists to support paired ultrasound/MRI comparison.

**What would refute H1-6.** Direct inspection evidence of shared patient/study keys plus real MRI data or reports for the same cases. **What would NOT refute it:** a similar-sounding external dataset, a dataset whose patients merely overlap in demographics, a synthetic/illustrative MRI, or an assumption that "the same patient" must exist.

**Required evidence.** Phase 1 audit output with explicit "Confirmed Present / Confirmed Absent / Unknown" findings for patient IDs, study IDs, MRI identifiers and MRI reports; a documented review of candidate external sources with licence and linkage-key requirements; and a definitive written feasibility verdict.

**Data required.** Dataset audit (Phase 1), ultrasound predictions (Phase 10), verified MRI data (Phase 12 — currently none).

**Threats to validity.** The dominant risk is *invention*: simulating MRI, inventing matching keys, or reporting a statistical comparison against unrelated data as if it were patient-level validation. All are prohibited.

**Answerable sub-questions regardless of the feasibility verdict:**

- What is the correct, non-overclaiming vocabulary for any cross-modal statement ("concordant/discordant/unavailable", never MRI "proving" the model right)?
- What would a concordance metric even mean without paired data, and is the answer "nothing interpretable" (which must be stated)?
- What minimal data-linkage requirements would a future study need? (A design output, not an executed validation.)

**Unanswerable sub-questions:** whether MRI findings would *confirm* ultrasound AI predictions in this cohort; whether cross-modal concordance improves diagnostic accuracy; whether either modality is superior — none of these can be established without verified paired data.

---

### RQ-7 — Dataset-derived limitations

**Question.** Which limitations arise from the available dataset — identifier absence, augmentation lineage, licence/provenance, class balance, label quality, image-quality heterogeneity, size — and which conclusions do those limitations permit versus preclude?

**Why it matters.** `PROGRESS.md` rules 13 and 15 require all assumptions to be recorded and dataset limitations never to be hidden. This is the question that keeps every other answer honest.

**H1-7 — `HYPOTHESIS (untested)`:** the dataset will support image-level and source-image-group-level conclusions but not patient-level, lesion-level or cross-modal conclusions.
**H0-7:** the dataset supports finer-grained conclusions than the above.

**What would refute H1-7.** Phase 1 evidence of identifiers, masks or MRI linkage. Refutation here is *good news* for scope, but only when backed by inspection evidence.

**Required evidence.** Phase 1 audit report with explicit present/absent/unknown findings; provenance/licence record; label-quality assessment; class distribution; image property survey; and a written statement of what is missing.

**Data required.** Phase 1 (all of V-1…V-11).

**Threats to validity.** Optimism bias — treating "probably present" as "present". Mitigation: every claim must cite a specific Phase 1 observation, not an inference.

---

### RQ-8 — Meaningfulness of attention-based visualisation

**Question.** Can attention weights be visualized meaningfully — that is, (a) do visualisations render correctly for bags of varying size, (b) are attention patterns stable across seeds/retraining, (c) are they non-degenerate, and (d) if ground-truth masks exist, do attended regions overlap annotated regions above chance?

**Why it matters.** Attention visualisation is candidate contribution #3, and `PROGRESS.md` rule 9 forbids equating model attention with clinical explanation. "Meaningful" must be defined operationally, or the claim is unfalsifiable.

**Operational definition (required for falsifiability):** a visualisation is *meaningful* only if it renders correctly, is stable across seeds for equivalent inputs, is non-degenerate (not uniform), and — where masks exist — aligns with annotated regions above a chance baseline. Otherwise it is at most a qualitative depiction of model behaviour.

**H1-8 — `HYPOTHESIS (untested)`:** attention visualisations will be non-degenerate and reasonably stable across seeds.
**H0-8:** attention patterns are degenerate and/or unstable, adding no inspectable information.

**What would refute H1-8.** Uniform or single-instance collapse, large seed-to-seed instability for the same bag, or mask overlap at chance level (if masks exist). Any of these refutes the "meaningful visualisation" claim — a required finding, not a failure to hide.

**Required evidence.** Per-bag attention statistics; multi-seed stability measurements; rendered visualisations for bags of varying size; and an explicit statement of whether mask-based overlap analysis was possible.

**Data required.** Trained model with attention outputs (Phases 8–9); masks for quantitative validation (Phase 1 V-3 — **expected absent**, in which case (d) is unanswerable and must be reported as such).

**Threats to validity.** Visual plausibility is not evidence: a convincing heatmap can be diagnostically meaningless. Also, any Grad-CAM cross-check must be compared against attention deliberately (agreement/disagreement reported), not used to validate it rhetorically.

---

### RQ-9 — Sensitivity to bag size / variable instance counts

**Question.** How does Dual Attention MIL performance and attention behaviour vary with bag size (singleton, small, large), and does variable-bag-size handling introduce artefactual behaviour?

**Why it matters.** If bags are singleton or very small — plausible given the expected absence of patient/study identifiers — the MIL framing may be nominal and attention aggregation near-meaningless. This directly conditions how RQ-1, RQ-2, RQ-4 and RQ-8 may be interpreted.

**H1-9 — `HYPOTHESIS (untested)`:** performance and attention stability will degrade as bag size approaches one.
**H0-9:** performance is insensitive to bag size.

**What would refute H1-9.** Stable, size-insensitive performance — or the absence of enough bag-size variation to test the question at all, which must be documented as an inability to answer rather than silently omitted.

**Required evidence.** Documented bag-size distribution (min/max/mean/median); stratified performance and attention metrics by bag-size bucket; the variable-bag-size unit tests from Phase 7/8.

**Data required.** Phase 3 bag manifest. If bag sizes do not vary meaningfully, the question is unanswerable and must be recorded as a limitation.

**Threats to validity.** Buckets may be too small for reliable per-bucket estimates; confounding with class label or source group.

---

### RQ-10 — Calibration and threshold-derived risk categories (exploratory)

**Question.** Are the model's output probabilities calibrated well enough that documented thresholds can legitimately define Low/Moderate/High risk categories (as `PROGRESS.md` Phase 13 requires, with thresholds derived from validation-set calibration rather than arbitrary cutoffs)?

**Why it matters.** Phase 13's risk categories and error analysis depend on calibration. If probabilities are poorly calibrated, risk categories must be labelled unreliable rather than presented as meaningful — and thresholds must not be invented.

**H1-10 — `HYPOTHESIS (untested)`:** calibration will be adequate enough to define ordered risk bands with documented uncertainty.
**H0-10:** calibration is inadequate, making probability-derived risk categories unreliable.

**What would refute H1-10.** Reliability diagrams showing large systematic over/under-confidence, or insufficient test-set size to estimate calibration at all. Refutation means risk categories are reported as unreliable — never silently redefined.

**Required evidence.** Reliability diagrams / calibration error where feasible; clearly documented threshold methodology; explicit statement when sample size precludes reliable calibration.

**Data required.** Phase 10 outputs (validation-set calibration; frozen-test evaluation). *(Gated by Phase 1 V-9.)*

**Threats to validity.** Small samples make calibration estimates noisy; class imbalance makes threshold placement sensitive; calibration on an unrepresentative split does not transfer.

---

## 3. Falsifiability Review

Required by `PROGRESS.md` Phase 0 validation check: *"Research questions are falsifiable/testable."*

| ID | Testable? | Falsifiable? | Refutation condition exists? | Observable evidence named? | Phase that answers it |
|---|---|---|---|---|---|
| RQ-1 | Yes — comparative experiment on fixed splits | Yes | Non-overlapping CIs favouring the null, or inconsistent sign across seeds | Metrics per model on the frozen test set | 10 |
| RQ-2 | Yes — matched-architecture comparison + ablations | Yes | Equivalence within CIs or single-attention advantage | Metrics + per-stage attention outputs | 10 |
| RQ-3 | Yes — descriptive across thresholds | Yes | Rank reversal between metrics | Full metric suite, CIs, curves | 10 |
| RQ-4 | Partially — (a)–(c) testable; the mask-dependent part is RQ-8(d), answerable only if masks exist | Yes for (a)–(c) | Uniform/unstable attention, or no bag-level benefit | Attention distributions, stability, bag vs image metrics | 7, 11 |
| RQ-5 | Yes — two split designs, same models/seeds | Yes | Negligible or reversed gap | Metrics under both designs + leakage test output | 4, 10 |
| RQ-6 | Yes — a definitive feasibility determination is always possible | Yes | Any verified paired MRI linkage refutes the expected negative verdict | Phase 1 audit + source review + written verdict | 1, 12 |
| RQ-7 | Yes — every limitation traceable to a Phase 1 observation | Yes | Phase 1 evidence of identifiers/masks/MRI | Audit's present/absent/unknown findings | 1–14 |
| RQ-8 | Yes — with the operational definition in §2 | Yes | Degenerate or unstable attention; chance-level mask overlap | Attention statistics, multi-seed stability, renders | 11 |
| RQ-9 | Yes, if bag sizes vary; otherwise unanswerable (documented) | Yes | Size-insensitive performance | Bag-size distribution + stratified metrics | 10 |
| RQ-10 | Yes, where sample size permits | Yes | Poor calibration | Reliability diagrams, calibration error | 10, 13 |

**Note on RQ-6:** "is it feasible?" is a decision question, and its answer can be definitively *no*. A feasibility verdict is therefore testable and falsifiable: it is refuted by finding genuinely linked paired MRI data.

---

## 4. Questions This Project Explicitly Does NOT Answer

Recorded so no later report accidentally implies otherwise:

1. Does the system improve patient outcomes, survival, or time-to-diagnosis? *(Requires clinical study — out of scope.)*
2. Is the system as accurate as a radiologist or oncologist? *(Requires reader study — out of scope.)*
3. Can the system be used to rule out malignancy or avoid biopsy? *(Clinical decision — prohibited.)*
4. Which treatment should a patient receive? *(Prohibited; not an intended capability.)*
5. Does the model generalise to other scanners, populations, or institutions? *(No external validation planned; single dataset.)*
6. Does MRI confirm the model's predictions? *(No paired MRI evidence expected; such a claim is prohibited outright.)*
7. Is model attention clinically explanatory or lesion-localising? *(Attention is model evidence only; without masks, unverifiable.)*
8. Is the approach novel or state-of-the-art? *(Requires literature review; novelty remains "to be verified".)*
9. Why does the model succeed or fail mechanistically? *(Not pursued at this scope; attention is not a mechanistic explanation.)*

---

## 5. Cross-References and Gating

- **Phase 1 gate.** RQ-1, RQ-3, RQ-5, RQ-6, RQ-7, RQ-9 and RQ-10 depend on dataset facts that are **UNVERIFIED**. Per `reports/phase0_scope.md` §14, no downstream decision may be finalised before Phase 1 verifies them (V-1 … V-11). Questions whose preconditions fail — most likely RQ-6's paired-MRI arm and RQ-8(d)'s mask validation — must be recorded as **unanswerable with the available data**, not approximated.
- **Phase 3 gate.** RQ-1, RQ-2, RQ-4 and RQ-9 depend on a justified bag definition that does not yet exist.
- **Pre-registration gate.** Before Phase 10, the primary metric and operating point for RQ-1/RQ-2 must be fixed so the headline comparison cannot be chosen post hoc.
- **Literature gate.** RQ-1, RQ-2, RQ-4 and RQ-8 must be positioned against prior work via `reports/phase0_literature_review.md`. If prior work already reports these comparisons, these questions become replication/extension questions rather than novel ones — a possibility that must be surfaced, not suppressed.
- **No answer has been produced in Phase 0.** Every question above remains open.

---

*End of Phase 0 research questions. APPROVED by the project owner (2026-09-16); Phase 0 Complete. No findings, metrics, novelty claims or clinical claims are made in this document; every question remains open.*
