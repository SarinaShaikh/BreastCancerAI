# Phase 0 — Literature Review Tracker

**Deliverable:** `reports/phase0_literature_review.md` (Phase 0 literature-review tracking — `PROGRESS.md` Phase 0 task: "Begin literature review tracking")
**Phase:** 0 (tracking begins here; the review itself continues across later phases)
**Status:** **NOT STARTED — tracking framework only.** No papers have been searched, read or assessed at the time of writing.
**Date recorded:** 2026-09-16
**Authority:** `PROGRESS.md`. Where this document conflicts with `PROGRESS.md`, `PROGRESS.md` wins.

> **What this document is.** A structured, auditable plan and checklist for the literature review required by Phase 0. It defines the topics to cover, the questions each topic must answer, the search approach, the evidence-recording format, and the novelty-positioning output.
>
> **What this document is not.** It is **not** a literature review. It contains no findings, no citations, no assessments of prior work, and no novelty conclusions — because no searches have been performed. Nothing here may be cited as evidence about prior art.

---

## 1. Rules Governing This Review

1. **No novelty is claimed.** All seven candidate contributions listed in `PROGRESS.md` remain **"to be verified"** until this review is complete and documented. They must not be described as novel, publishable, inventive, or state-of-the-art in any report, README, slide or demo — including inside this project's own documents.
2. **Absence of a hit is not evidence of novelty.** A failed or narrow search establishes only that the search was narrow. A claim of novelty requires a documented search across multiple databases, venues and query formulations, and an affirmative comparison against the closest prior work found.
3. **"Uncommon in our small sample of results" ≠ "novel."** Volume of literature is not the criterion; the criterion is whether the specific design choice and its evaluation have already been reported.
4. **Record the search, not just the result.** Every claim must be traceable to a recorded query, database, date and screening decision (§3).
5. **Distinguish search-proven from assumption.** Nothing may be asserted about prior work from memory or intuition.
6. **The term "dual attention" is a hazard, not a given.** It is used inconsistently across the literature (e.g. channel+spatial attention in CNNs; attention over two feature streams; two sequential MIL attention stages). This review must determine what prior work means by it, and must state precisely what *this* project means by it before any comparison is drawn. Divergent usage must be reported explicitly rather than silently equated.
7. **Negative positioning is required.** The review must actively look for prior work that would *reduce* the project's claimed contribution, not only for work that supports it.
8. **Feeds decisions, not decoration.** This review is an input to baseline selection (Phase 6), architecture justification (Phase 8), visualization methodology (Phase 11) and cross-modal vocabulary and feasibility framing (Phase 12). It is not a formality to complete after implementation.

---

## 2. Required Coverage (from `PROGRESS.md` Phase 0)

The eight items below are the checklist defined in `PROGRESS.md`'s "Potential Novelty — Requires Literature Verification" section. Each maps to a topic dossier in §4.

| # | Required topic | Dossier | Status |
|---|---|---|---|
| 1 | Survey CNN-based breast ultrasound classifiers | T1 | Not started |
| 2 | Survey transfer-learning approaches for breast ultrasound | T2 | Not started |
| 3 | Survey MIL approaches in medical imaging | T3 | Not started |
| 4 | Survey attention-based MIL approaches (e.g. ABMIL-style architectures) and identify what "dual attention" already exists in prior work | T4, T5 | Not started |
| 5 | Survey multimodal breast imaging systems (ultrasound + MRI, ultrasound + mammography) | T6 | Not started |
| 6 | Survey existing ultrasound + MRI concordance/consistency systems | T7, T8 | Not started |
| 7 | Survey explainable breast cancer AI systems (Grad-CAM, attention maps, saliency) | T9, T10 | Not started |
| 8 | Produce a novelty-positioning summary comparing this project's design choices against the above | §5, §6 | Not started |

**Additional coverage required by the Phase 0 brief and by project risk:**

| # | Additional topic | Dossier | Why it is needed |
|---|---|---|---|
| 9 | Ultrasound + MRI paired/multimodal systems (as distinct from concordance systems) | T8 | Separates "multimodal input" from "cross-modal verification of a prediction" |
| 10 | Grad-CAM and related visualization methods (including method limitations) | T10 | Needed for the Phase 11 cross-check and to define what a heatmap can and cannot support |
| 11 | MIL applied specifically to breast ultrasound | T3/T4 | The exact intersection that determines novelty of candidate contribution #1 |
| 12 | Provenance and known limitations of the specific Kaggle dataset | T11 | Establishes what is already publicly known about this dataset before Phase 1 re-derives it |

---

## 3. Search and Evidence-Recording Protocol

Each search must be logged before its results are interpreted, so the process is auditable:

| Field | Requirement |
|---|---|
| Query ID | Sequential (e.g. `T4-Q3`) |
| Dossier | Topic it belongs to |
| Database / source | E.g. Google Scholar, PubMed, IEEE Xplore, arXiv, Semantic Scholar, DBLP |
| Exact query string | Verbatim, including filters and date range |
| Date run | ISO date |
| Results screened | Count retrieved / count screened by title+abstract |
| Included | Citations recorded in the evidence table |
| Excluded (why) | Reason codes: off-topic, non-medical, no evaluation, preprint-only, superseded, inaccessible |
| Notes / limitations | Known gaps, e.g. paywalled venues, non-English literature not searched |

**Evidence table schema (used for every included paper):**

| Ref ID | Citation | Year | Task | Dataset(s) | Modality | Method family | Attention type | MIL? | Evaluation design (leakage-aware?) | Reported metrics | Relevance to our candidate contributions | Threat to our novelty claim |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

The **"Evaluation design (leakage-aware?)"** column is mandatory: prior work that splits at image level on a pre-augmented dataset is directly relevant to RQ-5 and must be flagged, since it indicates a known-inflated-metrics risk in the surrounding literature.

---

## 4. Topic Dossiers

Each dossier states: why it matters for this project, the questions it must answer, candidate search strings, and what a hit would mean. Search strings are starting points, not exhaustive queries.

---

### T1 — CNN-based breast ultrasound classifiers

**Why it matters.** This is the immediate baseline family for the project (Phase 6). It defines the performance context against which any MIL result will be judged, and it establishes whether any claimed improvement is even meaningful relative to established CNN performance on breast ultrasound.

**Questions to answer.** What CNN architectures have been applied to breast ultrasound classification? On which datasets, at what scale, with what evaluation design (single dataset? leakage-controlled? external validation?)? What metric ranges are typically reported, and with what sensitivity/specificity trade-off? What is known about class imbalance handling?

**Candidate search strings.** `breast ultrasound classification CNN`; `BUS benign malignant deep learning`; `breast ultrasound convolutional neural network diagnosis`.

**A hit means.** Non-MIL baselines are well established — which makes Phase 6 baselines mandatory and non-negotiable, and reduces the novelty value of "a CNN on breast ultrasound" to essentially zero.

---

### T2 — Transfer-learning approaches for breast ultrasound

**Why it matters.** Directly determines the Phase 6 backbone choice and the Phase 5 normalization strategy (dataset-specific vs. pretrained-backbone statistics). Also documents a known confound: ImageNet-pretrained features on grayscale ultrasound may or may not transfer well.

**Questions to answer.** Which pretrained backbones are commonly used for ultrasound? Is ImageNet pretraining reported as beneficial, neutral or harmful for ultrasound? How is grayscale-to-RGB handling usually addressed? How much fine-tuning depth is typical? Are dataset-specific normalization statistics standard practice?

**Candidate search strings.** `transfer learning breast ultrasound`; `ImageNet pretrained ultrasound classification breast`; `fine-tuning DenseNet breast ultrasound`.

**A hit means.** Backbone and normalization choices must be justified by precedent, not preference — and any deviation must be documented.

---

### T3 — MIL in medical imaging

**Why it matters.** Provides the methodological foundation for Phase 7. MIL in medical imaging is a mature area (pathology WSI, CT, mammography), so the general technique is not novel; what matters is its application to breast ultrasound under this dataset's constraints.

**Questions to answer.** How is MIL formulated in medical imaging (bag/instance/label conventions)? What are the established pooling strategies (max, mean, attention, mixed)? How are variable bag sizes and small bags handled in practice? What bag definitions are considered defensible when patient identifiers are unavailable? How is MIL evaluated to avoid leakage?

**Candidate search strings.** `multiple instance learning medical imaging survey`; `MIL whole slide image classification attention`; `weak supervision multiple instance learning diagnosis`.

**A hit means.** MIL infrastructure (Phase 7) must be implemented as a known, cited technique, and candidate contribution #2 (instance-level evidence aggregation) must be positioned against an extensive existing literature rather than assumed distinctive.

---

### T4 — Attention-based MIL (including ABMIL-style architectures)

**Why it matters.** The single-attention baseline in Phase 7 and the "first attention stage" in Phase 8 derive from this literature. This dossier also must determine an uncomfortable possibility: that attention-based MIL is so standard that "attention" alone adds no contribution.

**Questions to answer.** What are the canonical attention-based MIL formulations and their gating mechanisms? How is attention reported and visualized in prior work, and how do those works describe it (model evidence vs. explanation)? Do prior works validate attention against ground-truth regions, and what do they find when they do? How is attention stability assessed, if at all? What is the standard parameter-matched comparison protocol?

**Candidate search strings.** `attention based multiple instance learning`; `ABMIL gated attention deep learning`; `attention MIL interpretability validation`.

**A hit means.** The single-attention baseline and the "attention ≠ explanation" caveat have strong precedent; our explainability claims must be framed as measurement and stability analysis, not as a new capability.

---

### T5 — Existing dual-attention approaches

**Why it matters.** This is the highest-risk dossier for the project's central claim. If dual attention in MIL or in ultrasound already exists, candidate contribution #1 shrinks or disappears; the honest response is to reposition the work as comparative/extension study rather than to obscure the overlap.

**Questions to answer.** What does "dual attention" mean in existing work (channel+spatial attention; two-stream fusion; two sequential attention stages; dual cross-attention for multimodality)? Has any of these been applied to MIL aggregation over instances? Has any been applied to breast ultrasound? Is there prior art on two-stage instance-weight refinement in MIL specifically? What evaluation did it use? Are there negative/neutral replications of second-stage attention?

**Candidate search strings.** `dual attention network medical image`; `dual attention multiple instance learning`; `two stage attention aggregation MIL`; `channel spatial attention ultrasound classification`.

**A hit means.** Any hit must be documented and its overlap with our design stated precisely (same mechanism? same domain? same comparison?). A hit does **not** invalidate the project — it repositions it, and the repositioning must be written down. **A search that finds nothing must not be recorded as "no prior art exists."**

---

### T6 — Multimodal breast imaging systems

**Why it matters.** Frames what "multimodal" already means in breast imaging (ultrasound + mammography, MRI + mammography, multimodal fusion), and separates *fusing modalities as input* from *verifying one modality's prediction against another*. This distinction is central to how Phase 12 may be framed.

**Questions to answer.** How are breast imaging modalities fused in prior work (early/late/joint fusion)? What data linkages do multimodal systems require, and how do they obtain them? What do they report when one modality is missing? Is "cross-modal validation" as we intend it (using MRI to check a model's ultrasound prediction) an established concept, and under what name?

**Candidate search strings.** `multimodal breast cancer deep learning ultrasound mammography`; `multimodal fusion breast imaging diagnosis`; `missing modality medical imaging fusion`.

**A hit means.** Prior work establishes both the vocabulary and the data-linkage requirements the project would need — which sharpens exactly how far short of them the current dataset falls.

---

### T7 — Ultrasound + MRI systems

**Why it matters.** Determines whether ultrasound+MRI integration is a well-trodden research area, and what minimum identifiers and ethics approvals existing systems assume. This directly informs the Phase 12 feasibility verdict.

**Questions to answer.** Are there ultrasound+MRI breast studies, and how are patients matched across modalities? What linkage keys are used, and where does the data come from (institutional, registry, public)? Are there public datasets containing **paired** breast ultrasound and MRI? What accuracy does multimodal ultrasound+MRI report relative to single-modality?

**Candidate search strings.** `breast ultrasound MRI combined deep learning`; `paired ultrasound MRI breast dataset`; `multimodal breast cancer ultrasound magnetic resonance`.

**A hit means.** A public paired dataset, if one exists, is a lead for Phase 12 — but it must be verified for licence and true patient-level linkage before it can be used, and it may not be integrated without explicit approval (`PROGRESS.md` Phase 12; `reports/phase0_scope.md` Q-8).

---

### T8 — Ultrasound/MRI concordance systems

**Why it matters.** Defines whether "concordance between modalities" is an established evaluation concept, and with what semantics — because `PROGRESS.md` forbids language implying MRI *proves* an AI prediction correct. The review must locate the sanctioned vocabulary, not invent it.

**Questions to answer.** How is inter-modality concordance defined and measured in prior work (agreement statistics, kappa, correlation)? Is concordance ever used as a *verification* mechanism for an AI prediction, and with what caveats? How is discordance reported and interpreted? What statistical framing is used when modalities disagree?

**Candidate search strings.** `ultrasound MRI concordance breast lesion`; `inter modality agreement breast imaging kappa`; `diagnostic consistency modalities breast`.

**A hit means.** If concordance-as-verification has precedent, its limitations almost certainly do too — and those limitations become mandatory caveats in Phase 12 and Phase 13 language.

---

### T9 — Explainable breast cancer AI

**Why it matters.** Underpins Phase 11 and the constraint that attention must never be equated with clinical explanation. The critical question is whether explainability in this domain has been *validated*, or merely *displayed*.

**Questions to answer.** What explainability methods dominate in breast cancer AI (saliency, attention, concept-based, prototype)? How is explainability evaluated — human reader studies, mask overlap, faithfulness/sanity checks? What is the reported evidence that attention or saliency in breast imaging reflects clinically meaningful regions? What are the documented failure modes (e.g. shortcuts to burned-in annotation marks, laterality markers, device text)?

**Candidate search strings.** `explainable AI breast cancer ultrasound attention`; `saliency map validation medical imaging faithfulness`; `explainability evaluation reader study radiology`.

**A hit means.** Prior failure modes — particularly shortcut learning on scanner annotations or markers — become mandatory checks in Phase 11, and any claim beyond "depicts model behaviour" must be dropped unless validated.

---

### T10 — Grad-CAM and related visualization methods

**Why it matters.** Phase 11 optionally implements Grad-CAM as a cross-check on attention. Its known weaknesses (resolution, class-discriminativeness caveats, sensitivity to implementation details) determine what such a cross-check can legitimately prove. The "Sanity Checks for Saliency Maps" line of work is directly relevant.

**Questions to answer.** What are canonical gradient-based visualization methods and their documented limitations? Which sanity-check criteria apply (model-parameter randomization, data randomization, faithfulness)? What is the standard way to compare two saliency methods without circularity? What resolution/upsampling artefacts should be expected on small inputs?

**Candidate search strings.** `Grad-CAM breast ultrasound`; `sanity checks for saliency maps`; `faithfulness evaluation saliency medical imaging`.

**A hit means.** The Phase 11 cross-check must be framed as *agreement measurement between two fallible depictions*, never as mutual validation; agreement must be quantified.

---

### T11 — The specific dataset's provenance and known limitations

**Why it matters.** Distinguishes what is publicly *documented* about this Kaggle dataset (licence, stated provenance, stated augmentation, known critiques) from what Phase 1 must *verify by inspection*. This prevents re-asserting listing claims as findings and prevents Phase 1 from overlooking a documented caveat.

**Questions to answer.** What is the dataset's stated licence and citation requirement? Who built it, from what source, and with what labelling process? Is any of its provenance documented in the peer-reviewed literature? Are there published critiques, known label-quality issues, or benchmark results on it? Has it appeared in any paper with an evaluation design we should learn from (or avoid)?

**Candidate search strings.** `"Ultrasound Breast Images for Breast Cancer" Kaggle dataset`; dataset author name; `breast ultrasound augmented dataset rotation sharpening`.

**A hit means.** Any documented label-quality or provenance issue must be carried into `reports/phase0_scope.md` §15 (Known Limitations) and Phase 1's audit scope. **Absence of published critique must not be read as evidence of dataset quality.**

---

## 5. Novelty-Positioning Matrix (template — unfilled)

Required Phase 0 output. The candidate contributions are taken verbatim from `PROGRESS.md`. **Every cell is empty because no review has been performed. Novelty status is "to be verified" across the board, and must remain so until evidence is recorded.**

| # | Candidate contribution (`PROGRESS.md`) | Closest prior work found | Design overlap | Domain overlap | Evaluation overlap | Verdict (novel / partially overlapping / already established / inconclusive) | Evidence ref IDs |
|---|---|---|---|---|---|---|---|
| 1 | Dual Attention MIL applied specifically to breast ultrasound classification | — | — | — | — | **To be verified** | — |
| 2 | Instance-level evidence aggregation rather than simple image classification | — | — | — | — | **To be verified** | — |
| 3 | Attention-guided ultrasound evidence visualization | — | — | — | — | **To be verified** | — |
| 4 | Cross-modal validation against MRI findings | — | — | — | — | **To be verified** | — |
| 5 | Ultrasound/MRI diagnostic concordance analysis | — | — | — | — | **To be verified** | — |
| 6 | Patient/study-level risk estimation | — | — | — | — | **To be verified** | — |
| 7 | Explainable clinical decision support | — | — | — | — | **To be verified** | — |

**Additional required statement for each verdict:** whether the contribution is *methodological* (a new mechanism), *applied* (an existing mechanism in a new domain), *comparative* (an evaluation comparing known methods), or *engineering* (integration of existing components). `PROGRESS.md` does not permit any of these to be presented as novel by default — and per `reports/phase0_scope.md` §24 (Q-6), contribution #6 has an unresolved internal contradiction with the expected absence of patient identifiers.

---

## 6. Notes on Contributions #4–#7 Relative to Data Reality

Cross-referenced with `reports/phase0_scope.md` §17 and `reports/research_questions.md` RQ-6, and recorded here because the literature review must not compensate for a data gap:

- **#4 Cross-modal validation against MRI findings** and **#5 Ultrasound/MRI concordance analysis** require paired data that is currently **unverified and expected absent**. Prior work describing such systems does **not** make them feasible here, and must never be used to imply that this project implements them.
- **#6 Patient/study-level risk estimation** presumes patient or study identifiers whose existence is **unverified and expected absent**.
- **#7 Explainable clinical decision support** is constrained by `PROGRESS.md` rules 9–11: attention is not explanation, the system does not replace a clinician, and no autonomous treatment prescription. The literature may describe clinical decision support systems; that does not authorise this prototype to claim clinical decision support capability.

These four contributions must be presented in any novelty-positioning summary together with their data preconditions, and marked as **conditional** where those preconditions are unmet.

---

## 7. Required Outputs of the Literature Review

1. A completed evidence table per dossier (§3 schema).
2. A filled novelty-positioning matrix (§5) with explicit verdicts and reference IDs.
3. A written statement of what is **already established** in prior work, so the project stops treating it as a contribution.
4. A written statement of what appears **distinctive**, with the search basis for that statement — and the explicit caveat that absence of a hit is not proof of novelty.
5. A "closest prior work" paragraph for the final report, naming the nearest existing approach and stating precisely how this project's design compares.
6. Any **negative** findings: known replication failures, documented dataset caveats, or prior work showing attention/saliency is misleading in this domain.

**Review completion criteria:** the eight required topics in §2 are each marked complete with recorded searches; every candidate contribution has a verdict with supporting reference IDs; and no claim of novelty appears anywhere in the project's documents without a matching entry in the matrix.

---

## 8. How This Review Constrains Later Phases

| Phase | Constraint from this review |
|---|---|
| 6 | Baseline choice and backbone must be justified against T1/T2 precedent; evaluation design must be compared with prior work's leakage handling. |
| 7 | MIL formulation must cite established practice (T3); variable bag-size handling should follow documented approaches. |
| 8 | The Dual Attention design must be documented against T4/T5: what exists, what we changed, and why — with no novelty assertion until T5 is complete. |
| 9–10 | Metric suite must be comparable to prior work (T1/T9) so results are interpretable in context; evaluation design must be defensible given documented leakage risks. |
| 11 | Visualization and cross-check methodology must follow T9/T10, including sanity checks and shortcut-learning checks. |
| 12 | Feasibility framing and concordance vocabulary must follow T6/T7/T8 — using established, non-overclaiming terminology. |
| 14 | Final report must include the closest-prior-work statement and must not claim novelty unsupported by the matrix. |

---

*End of Phase 0 literature review tracker. Tracking framework only — no searches performed, no prior work assessed, no novelty claim made. All seven candidate contributions remain "to be verified".*
