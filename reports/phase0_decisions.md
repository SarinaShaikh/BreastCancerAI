# Phase 0 — Decision Record (Q-1, Q-4)

**Deliverable:** `reports/phase0_decisions.md`
**Phase:** 0 (blocker resolution at specification/architecture level only)
**Date recorded:** 2026-09-16
**Status of this document:** SPECIFICATION ONLY. Nothing in it has been implemented. No directories, code, manifests, splits, or data were created, moved, renamed or downloaded for this record.
**Authority:** `PROGRESS.md` remains authoritative. This record resolves two open decisions **at specification level**. Both were **approved by the project owner on 2026-09-16** and are in effect as the project's working decisions (see §4). Where this record conflicts with `PROGRESS.md`, `PROGRESS.md` wins and the conflict must be surfaced.

> **How to read this record.** Q-1 and Q-4 were logged as open decisions in `reports/phase0_scope.md` §24. This document investigates each, documents the technically valid alternatives, records the recommended resolution, and states precisely what remains a human decision. It deliberately does **not** quietly convert an investigation into an irreversible choice.

---

## 0. Verified Repository Facts (inputs to both decisions)

Established by direct inspection on 2026-09-16 (`git check-ignore`, `ls`, `find`, `git status`) — these are observations of the repository, not assumptions:

| # | Fact | Evidence |
|---|---|---|
| F-1 | The actual top-level structure is `backend/`, `model/`, `frontend/`, `notebooks/`, `dataset/`, `docs/`, `reports/`, `README.md`, `PROGRESS.md`, `.gitignore`. | `git ls-files`, directory listing |
| F-2 | `dataset/` is **gitignored** (rule `dataset/` at `.gitignore` line 10) and currently contains only `.gitkeep`. No dataset is present; nothing has been downloaded. | `git check-ignore -v dataset/x.jpg`; `ls -A dataset/` |
| F-3 | `data/` does not exist and is **not** gitignored — files created there would be committable. | `git check-ignore -v data/manifests/test.csv` → not ignored |
| F-4 | `src/`, `tests/`, `configs/`, `app/`, `experiments/`, `checkpoints/` do not exist and are not gitignored. | directory existence check |
| F-5 | Every `.py` file under `backend/` and `model/da_mil/` is empty (0 bytes); `backend/requirements.txt` is the only non-empty backend file. No code of any kind exists yet. | `find . -name '*.py' -size +0c` → 0 results |
| F-6 | The roadmap's "Repository Structure" section is explicitly illustrative and self-limiting: *"Adapt to any pre-existing repository structure. Do not overwrite or reorganize existing files unnecessarily."* | `PROGRESS.md`, Repository Structure note |
| F-7 | Later phases reference concrete roadmap paths 20+ times (`src/data/audit.py`, `data/manifests/*.csv`, `src/preprocessing/*`, `src/mil/*`, `src/models/*`, `src/training/train.py`, `src/evaluation/*`, `src/explainability/*`, `src/mri/*`, `src/clinical_support/*`, `configs/*.yaml`, `experiments/`, `checkpoints/`, `tests/test_leakage.py`, `app/`). | `PROGRESS.md` Files/Modules sections, Phases 1–14 |
| F-8 | The only dataset rule in `.gitignore` is `dataset/` (plus `*.zip`, `*.pth`, `*.pt`, `model/da_mil/weights/`). There is no rule covering `data/`, `data/processed/`, or `experiments/`. | `.gitignore` |
| F-9 | No MRI file, report or MRI-named artifact exists anywhere in the repository. The only MRI-related item is the empty placeholder `backend/services/mri_validation.py` (0 bytes). | full-file inventory; `find` for MRI-named files |

---

# 1. Q-1 — Repository Structure Resolution

## 1.1 Existing structure (verified, unchanged)

```
BreastCancerAI/
├── backend/            # application/API scaffold — all .py files empty placeholders (F-5)
│   ├── main.py, database.py, schemas.py, requirements.txt
│   ├── router/         (prediction.py, report.py, validation.py — empty)
│   └── services/       (mri_validation.py, report_service.py, ultrasound_prediction.py — empty)
├── model/
│   └── da_mil/         (model.py, attention.py, dataset.py — empty placeholders)
├── frontend/           (empty package.json)
├── notebooks/          (5 stub notebooks)
├── dataset/            # gitignored raw-data root (F-2) — EMPTY
├── docs/               (empty architecture.md, methodology.md)
├── reports/            # Phase 0 deliverables live here
└── PROGRESS.md, README.md, .gitignore
```

The existing architecture separates **application code** (`backend/`, `frontend/`) from **ML model code** (`model/da_mil/`) — a reasonable split that Q-1 must preserve.

## 1.2 Problem with the current roadmap paths

1. **The roadmap tree does not exist and overlaps the real one.** It prescribes `src/` + `data/` + `configs/` + `tests/` + `app/`, none of which exist (F-4), while the real repo's `model/`, `backend/`, `dataset/` appear nowhere in the roadmap tree.
2. **`data/raw/` collides with the existing, gitignored `dataset/`.** Phase 1 says "Download dataset … into `data/raw/`", but `dataset/` already *is* the designated raw-data root — creating `data/raw/` would produce two parallel raw-data locations, exactly the duplication Q-1 is instructed to avoid. It would also put raw images under a path that is **not** gitignored (F-3), risking accidental commits of large binary data.
3. **Generated artifacts vs. raw data are not distinguished.** `data/processed/` (Phase 5 cache) and `data/manifests/` (committed tables) have opposite commit requirements, but the roadmap treats `data/` uniformly and `.gitignore` covers neither (F-8). Without a rule, generated audit artifacts could be confused with raw data — a named requirement of this task.
4. **Model-code placement is ambiguous.** The roadmap puts Phase 6–8 models in `src/models/` and `src/mil/`, while the existing scaffold pre-creates `model/da_mil/{model,attention,dataset}.py` — apparently the intended home for exactly that code. Unresolved, Phase 8 would face two candidate homes.

## 1.3 Technically valid options (documented, not silently chosen)

| Option | Description | Advantages | Disadvantages |
|---|---|---|---|
| **O-1 — Adopt the roadmap tree wholesale** | Create `src/…`, `data/…`, `configs/`, `tests/`, `app/` exactly as written; leave existing dirs untouched. | Zero edits to `PROGRESS.md`'s path references; roadmap stays literally true. | Creates a parallel structure next to `model/` and `backend/` (the roadmap's `src/models/` vs existing `model/da_mil/`); leaves the `data/raw/` vs `dataset/` collision unresolved; largest new-tree footprint. |
| **O-2 — Map the roadmap onto the existing structure** | Keep `model/`, `backend/`, `dataset/` as canonical homes; treat roadmap paths as logical names mapped onto real ones via a recorded table. | Preserves existing architecture (requirement 1); no duplicate structures (requirement 2); no file moves (requirement 3). | The 20+ literal roadmap paths no longer match the filesystem; every future reader/agent must consult the mapping table; `PROGRESS.md` text would mislead unless annotated. |
| **O-3 — Hybrid (recommended)** | Create the roadmap's `src/`, `data/`, `configs/`, `tests/` for the research/data pipeline; keep existing `model/` (ML model code), `backend/`+`frontend/` (application), `dataset/` (raw data) in their existing roles; record the full mapping. | Preserves the existing app/model/data separation; no duplication (nothing currently lives in `src/`/`data/`/`tests/` — F-4, F-5); honors the roadmap's own "adapt" note (F-6); keeps 12 of the roadmap's path families literally true. | Leaves exactly two intentional divergences (raw-data root; Phase 6–8 model-code home) that must be recorded and ratified. |

## 1.4 Adopted resolution — D-1 (**APPROVED by the project owner 2026-09-16; in effect**)

**Adopt O-3, the hybrid structure.** Specifically:

- **Research & data-audit code → `src/`** (new, not yet created). All Phases 1–5, 7, 9–13 module paths from the roadmap remain literally valid: `src/data/audit.py`, `src/data/splitting.py`, `src/preprocessing/*`, `src/mil/*` (infrastructure + baselines), `src/training/train.py`, `src/evaluation/*`, `src/explainability/*`, `src/mri/*`, `src/clinical_support/*`, plus `src/utils/`.
- **Raw dataset → `dataset/`** (existing, gitignored). Phase 1's "`data/raw/`" is mapped to **`dataset/raw/`** (a subfolder to be created *during* Phase 1, not now). Raw images stay uncommitted; integrity is guaranteed by recording checksums in the committed manifest instead.
- **Manifests and committed tables → `data/manifests/`** (new at Phase 1; not gitignored — F-3 — which is exactly right: the Phase 1 exit criterion requires the manifest to be *committed*).
- **Generated/reproducible caches → `data/interim/`, `data/processed/`, `experiments/`, `checkpoints/`** (new at Phases 5/9). These must be **added to `.gitignore` when created** (see §1.8) so generated artifacts can never be confused with raw or source material.
- **ML model code → `model/`** (existing). The pre-created placeholders are recognized as the intended homes: Phase 8's roadmap files `src/models/dual_attention_mil.py` and `src/mil/dual_attention.py` map to **`model/da_mil/model.py`** and **`model/da_mil/attention.py`**; `src/models/baseline_cnn.py` / `transfer_learning.py` map under **`model/`** (e.g. `model/baselines/`). Model weights remain ignored via the existing `*.pth`/`*.pt`/`model/da_mil/weights/` rules.
- **Application/API/UI → `backend/` + `frontend/`** (existing, untouched; realized in Phase 14 under the roadmap's `app/` concept). The empty `backend/services/mri_validation.py` placeholder stays untouched and gains no significance (F-9).
- **Tests → `tests/`** (new at Phase 4: `tests/test_leakage.py`, then the global checklist).
- **Configs → `configs/`** (new at Phase 5: `preprocessing_config.yaml`; Phase 9: `training_config.yaml`).
- **Reports → `reports/`** (existing; all `reports/phaseN_*.md`).

## 1.5 Phase 1 artifact locations under D-1

| Phase 1 artifact | Roadmap path | Adopted path | Committed? |
|---|---|---|---|
| Audit code | `src/data/audit.py` | `src/data/audit.py` (unchanged) | Yes |
| Dataset manifest | `data/manifests/dataset_manifest.csv` | `data/manifests/dataset_manifest.csv` (unchanged) | Yes (exit criterion requires it) |
| Audit report | `reports/phase1_dataset_audit.md` | `reports/phase1_dataset_audit.md` (unchanged) | Yes |
| Raw dataset | `data/raw/` | **`dataset/raw/`** (mapped; subfolder created in Phase 1) | **No** — gitignored (F-2) |
| Download record (version/date/checksums) | — | `data/manifests/dataset_source.json` | Yes |
| Tests (audit/discovery/image-validation) | Testing Requirements section | `tests/test_dataset_audit.py` (naming convention: one file per phase) | Yes |

## 1.6 Why this structure is preferable (against the stated requirements)

1. **Preserves existing architecture** — `backend/`, `model/`, `frontend/`, `dataset/` keep their roles; nothing moves or is renamed (requirements 1, 3).
2. **No duplicate parallel structures** — `src/`, `data/`, `tests/` currently contain *nothing* (F-4, F-5), so creating them duplicates no code; the one true collision (`data/raw/` vs `dataset/`) is resolved by mapping rather than duplication (requirement 2).
3. **Separation of concerns** — application (`backend/`, `frontend/`) | ML models (`model/`) | research/data-audit code (`src/`) | raw data (`dataset/`, ignored) | committed manifests (`data/manifests/`) | reports (`reports/`) | tests (`tests/`) | configs (`configs/`) (requirement 4).
4. **Reproducibility & artifact hygiene** — everything needed to reproduce (code, configs, committed manifests, reports) is versionable; everything generated in bulk (raw images, processed caches, logs, checkpoints, weights) is gitignored, so generated audit artifacts cannot masquerade as raw data or source (requirement 5).
5. **Later-phase compatibility** — 12 roadmap path families remain literally valid; only two divergences exist, both recorded in the mapping table (§1.7) and both *reducing* risk (requirement 6).

## 1.7 How later phases reference these locations

Mapping table (roadmap path → adopted path). All unmapped roadmap paths are **unchanged**.

| Phase | Roadmap reference | Adopted location |
|---|---|---|
| 1 | `data/raw/` (download target) | `dataset/raw/` |
| 1–4 | `data/manifests/*.csv` | unchanged — `data/manifests/` |
| 1–13 | `src/data/*`, `src/preprocessing/*`, `src/mil/*`, `src/training/*`, `src/evaluation/*`, `src/explainability/*`, `src/mri/*`, `src/clinical_support/*` | unchanged — `src/…` |
| 5, 9 | `configs/*.yaml` | unchanged — `configs/` |
| 5 | `data/processed/` cache | unchanged — `data/processed/` (gitignore rule to be added when created) |
| 6 | `src/models/baseline_cnn.py`, `src/models/transfer_learning.py` | `model/baselines/baseline_cnn.py`, `model/baselines/transfer_learning.py` |
| 6 | `src/mil/standard_mil_baseline.py` | unchanged — `src/mil/standard_mil_baseline.py` |
| 8 | `src/mil/dual_attention.py`, `src/models/dual_attention_mil.py` | `model/da_mil/attention.py`, `model/da_mil/model.py` (existing placeholders) |
| 9 | `experiments/logs/`, `checkpoints/` | unchanged (gitignore rules to be added when created) |
| 4+ | `tests/test_leakage.py` + global checklist | unchanged — `tests/` |
| 14 | `app/` | realized as `backend/` (API) + `frontend/` (UI) |

Under D-1, `PROGRESS.md`'s "Repository Structure" tree remains the **logical** map; the table above is the **physical** map. Any future agent must treat the physical table as binding when creating files.

## 1.8 Remaining concerns (recorded, not silently resolved)

1. **`dataset/` vs `data/` visual confusion.** Two similarly named top-level directories with opposite commit semantics is a standing foot-gun. Mitigation (documentation-level, done here): raw images only ever under `dataset/` (ignored); only small, committed, human-readable artifacts under `data/` (manifests). Mitigation (Phase 1 action, not now): add a one-line comment to `.gitignore` when `dataset/raw/` is created.
2. **`.gitignore` must grow later.** `data/processed/`, `data/interim/`, `experiments/`, `checkpoints/` are not currently ignored (F-8). The rules must be added **at the moment those directories are created** (Phases 5/9), not retroactively. Flagged as a Phase 5/9 task; intentionally not changed now because this task is documentation-only.
3. **Model-code split spans two roots.** MIL *infrastructure* lives in `src/mil/` while the Dual Attention *model* lives in `model/da_mil/`. This mirrors the existing scaffold but imports will cross roots; acceptable, but the project owner should confirm this is intended rather than accidental.
4. **Interplay with Q-3.** If the owner instead chooses to *delete* the empty `model/da_mil/` and `backend/` placeholders (Q-3 option b), the Phase 8/14 rows of the mapping table change to the literal roadmap paths. The two decisions should be ratified together.

## 1.9 Q-1 decision status

- **A-1a (primary): APPROVED by the project owner (2026-09-16).** D-1 (O-3 hybrid) and the mapping table in §1.7 are the **binding physical structure**. Alternatives O-1 and O-2 are closed.
- **A-1b: still open** (owned by the project owner, per Q-3) — the fate of the empty `model/da_mil/` + `backend/` placeholders. D-1 was ratified on the keep-and-map reading; if the owner later chooses delete-and-literalize under Q-3, the Phase 6/8/14 rows of §1.7 change accordingly.
- **A-1c: closed by ratification of the hybrid** — `dataset/` is not renamed; the §1.7 mapping stands.

D-1 takes **physical** effect only when Phase 1 creates files under the approved mapping; no directory or file is created now.

---

# 2. Q-4 — Cross-Modal Validation Definition

## 2.1 What `PROGRESS.md` actually says (extracted, not paraphrased)

| Location | Statement |
|---|---|
| Problem Statement | "cross-validate ultrasound prediction accuracy against MRI reports of **the same patient**" |
| Dataset policy | "No claim about … MRI reports … is to be treated as true until PHASE 1 dataset audit confirms it" |
| Rules 2, 4 | "Never fabricate MRI reports." "Never claim MRI validation without actual paired MRI evidence." |
| Phase 12 Objective | "Rigorously determine whether reliable **paired** MRI data exists … only build the concordance module if such data is genuinely available." |
| Phase 12 Why | "explicitly flagged as a high-risk area for fabrication. The primary Kaggle dataset is an **ultrasound-only** dataset; MRI validation must not be faked or inferred." |
| Phase 12 gate | If no patient IDs / MRI identifiers / MRI reports / paired exams confirmed → document: *"Cross-modal MRI validation cannot be clinically demonstrated using the primary Kaggle dataset alone."* |
| Phase 12 risk | Simulated MRI forbidden in evaluation; any illustrative synthetic MRI must be labelled "SYNTHETIC / ILLUSTRATIVE ONLY — NOT REAL PATIENT DATA" and never mixed into evaluation. |
| Phase 12 language | Outputs must use "cross-modal diagnostic consistency" / "cross-modal concordance" — **never** language implying MRI "proves" the model correct. |
| Phase 13 schema | "Cross-modal evidence: MRI concordant / discordant / **unavailable**" |
| Phase 14 | Demo must honestly state the Phase 12 outcome. |
| Testing | MRI tests exist only "(if MRI data available)". |

Two structural facts follow directly: the roadmap's only sanctioned validation concept is **paired** validation (same patient/study), and Phase 12 is designed as a **feasibility gate with a legitimate negative outcome** — not as a module that must succeed.

## 2.2 Canonical taxonomy (the five, and only five, admissible concepts)

Henceforth, every project artifact must use exactly one of these labels. No other framing is permitted.

| Label | Definition | What it can establish | What it can never establish |
|---|---|---|---|
| **T-1. Paired cross-modal validation** | Ultrasound cases and MRI examinations/reports are linked **per patient/study by verified identifiers**, and the ultrasound model's prediction is compared against MRI-derived findings **on the same cases**. | Concordance/discordance at case level; cross-modal consistency statistics. | Diagnostic correctness of either modality (MRI is a comparison, not ground truth, absent histopathology); clinical validation. |
| **T-2. Independent external validation** | The model (or the method) is evaluated on a **different, unlinked** dataset — possibly MRI-based — as a separate population. | External validity of *that* evaluation; robustness across populations/acquisition protocols. | Anything about ultrasound/MRI agreement on the same case; it is **not** cross-modal validation at all. |
| **T-3. Modality-transfer / domain-shift analysis** | Quantification of distributional differences between the ultrasound data and a separate MRI cohort (intensity statistics, texture, artifacts), or cross-modality feature analysis. | That the modalities differ, and how; hypotheses for future paired studies. | Any claim of diagnostic agreement or model validation; a distribution gap is not a diagnostic statement. |
| **T-4. Synthetic-data experimentation** | Generated/fake MRI used strictly for pipeline engineering or illustration. | That the *code paths* run. | Anything scientific or clinical; excluded from all evaluation by roadmap rule. |
| **T-5. Absence of MRI validation** | The verified outcome that no usable MRI evidence exists. | An honest, complete feasibility verdict — which the roadmap explicitly accommodates. | Nothing about cross-modal agreement — and it must be reported as "not established", never omitted. |

**Forbidden equivalences (binding):** an external MRI dataset containing breast images does **not** become paired data because of topical overlap (D → T-2/T-3, never T-1); synthetic MRI is never T-1 (E → T-4 only); patient-level multimodal claims require verified linkage, full stop; and "cross-modal consistency" language must never be upgraded to "MRI confirms/proves/validates the model".

## 2.3 Evidence states A–E

### State A — Verifiable paired ultrasound + MRI data exists

Linkage is verified (shared patient/study identifiers confirmed by direct inspection, with documented matching method), and real MRI examinations/reports cover the linked cases.

- **Can legitimately claim:** case-level **T-1** — concordance/discordance per case; cross-modal consistency statistics; the Phase 12 concordance module executed on real paired data. All outputs must still use "concordance/consistency" language, never "MRI proves the model right" (roadmap Phase 12).
- **Cannot claim:** that MRI establishes diagnostic correctness (unless a verified histopathology ground truth exists for those cases, which is a separate, stronger evidence state); clinical validation; that concordance implies accuracy.
- **Phase 12 executability:** fully executable as designed (matching → report/finding extraction → concordance criteria → comparison pipeline → feasibility report).
- **Scientifically meaningful validation:** T-1 only. Reporting must still include the full ultrasound-only metric suite (rules 7–8); concordance is supplementary evidence, not a headline.
- **Insufficient evidence:** a shared filename *pattern* without verified identifier semantics; an asserted (undocumented) matching key; matching by class label or demographics; "similar-looking" cohorts; an external paired dataset whose linkage key was never inspected.
- **Human approval:** not required beyond standard phase gates — the data itself satisfies the roadmap's condition. Matching-method and BI-RADS-equivalent terminology choices still need documentation and review.

### State B — MRI exists but pairing/linkage cannot be verified

MRI-related material is present in (or alongside) the primary dataset, but no verifiable patient/study linkage key exists.

- **Can legitimately claim:** that MRI-related data exists but is **unusable for T-1**; T-3 analyses on the unlinked material are permissible if explicitly labelled "unlinked exploratory analysis".
- **Cannot claim:** any paired cross-modal result; any patient-level multimodal statement. This state must **never** be narrated as "cross-modal validation performed with caveats".
- **Phase 12 executability:** only the feasibility-gate branch — the verdict is "linkage unverifiable ⇒ paired validation not demonstrable", plus a design-only module spec. That verdict is a **successful Phase 12 outcome**, not a failure to complete the phase.
- **Scientifically meaningful validation:** none beyond T-3 description; optionally T-2-style evaluation of an MRI classifier as a separate study, clearly labelled as not validating the ultrasound model.
- **Insufficient evidence:** probabilistic or fuzzy matching ("probably the same patient"); linkage inferred from acquisition date/sequence numbers; demographic-only matching.
- **Human approval:** **required** before any use of the unlinked MRI material (even T-3), since the roadmap's gate condition ("confirmed present") is not met — the project owner must decide whether unlinked exploratory analysis is in scope at all. Recommended default: **no**, keep Phase 12 at the documented-infeasibility branch.

### State C — No MRI data exists in the primary dataset

The expected state per the Kaggle listing (ultrasound-only) — to be confirmed or refuted by Phase 1's V-4 audit, not assumed.

- **Can legitimately claim:** **T-5** — the roadmap's own sentence, verbatim: "Cross-modal MRI validation cannot be clinically demonstrated using the primary Kaggle dataset alone." Plus the designed-but-unexecuted concordance-module spec, and a documented review of candidate external sources (research only, no integration).
- **Cannot claim:** anything in T-1's column; any MRI-related result; any implication that the limitation is provisional or being "worked around".
- **Phase 12 executability:** executable as the feasibility-gate branch — feasibility report + unexecuted module design. This is the roadmap's explicitly sanctioned negative outcome.
- **Scientifically meaningful validation:** none cross-modal; all validation remains ultrasound-only (Phases 4–11).
- **Insufficient evidence:** *any* substitute — simulating MRI, borrowing an unrelated cohort's statistics as if paired, or reporting a T-2/T-3 exercise under the words "cross-modal validation".
- **Human approval:** not required to *document* the negative verdict (it is the roadmap's own gate outcome). Approval **is** required before acquiring any external MRI dataset (which would move the project toward State D) — see A-4c/Q-8.

### State D — Approved external MRI dataset, not paired with the ultrasound dataset

A separate MRI dataset is approved for use but shares no verifiable patient/study identifiers with the ultrasound data.

- **Can legitimately claim:** **T-2** and/or **T-3** only — e.g., an MRI classifier trained/evaluated on the external cohort as an independent study; distributional/domain-shift comparison between cohorts. Every artifact must state "independent external data — **not paired**; no patient-level linkage exists".
- **Cannot claim:** paired cross-modal validation, concordance, "cross-modal validation" in any sense, or any patient-level multimodal statement. Combining an external MRI set with the ultrasound set does **not** manufacture pairing — topical overlap of two breast-imaging cohorts is explicitly ruled out by this decision.
- **Phase 12 executability:** the feasibility gate still resolves to "paired validation not demonstrable with linked data"; Phase 12 may *additionally* document the T-2/T-3 analyses as clearly-labelled supplementary studies. The Phase 12 deliverable remains the feasibility report; the concordance module stays unexecuted.
- **Scientifically meaningful validation:** T-2 (external validity for whatever model is evaluated on the external cohort) and T-3 (domain characterization). Neither validates the ultrasound model against MRI.
- **Insufficient evidence:** citing the external dataset's own accuracy as if it contextualized the ultrasound model; merging the cohorts into one evaluation; describing T-2/T-3 outputs as "cross-modal".
- **Human approval:** **mandatory before acquisition or integration** (roadmap Phase 12: "without integrating them without further verification and explicit approval"; scope §24 Q-8). Approval must cover licence, ethics/provenance, linkage-key assessment, and the exact permitted analyses.

### State E — Synthetic MRI data is used

Generated or simulated MRI material exists in the project in any form.

- **Can legitimately claim:** **T-4** only — that engineering/demo code paths execute on synthetic input. Every synthetic artifact must carry the roadmap's unmistakable label: "SYNTHETIC / ILLUSTRATIVE ONLY — NOT REAL PATIENT DATA".
- **Cannot claim:** anything scientific: no validation, no concordance, no feasibility improvement, no performance number. Synthetic data must **never** enter training, validation, test, or any reported metric (roadmap Phase 12 risk note).
- **Phase 12 executability:** Phase 12's feasibility verdict is unaffected — synthetic data cannot satisfy the gate. At most, a synthetic fixture may exercise unexecuted module code inside the design-only deliverable, labelled as such.
- **Scientifically meaningful validation:** none. T-4 is an engineering aid, not evidence.
- **Insufficient evidence:** n/a — by construction nothing synthetic counts as evidence; the risk to guard is *contamination* (synthetic content leaking into evaluation or into unlabelled artifacts), which the Phase 14 UI review and the rules-9–11 review must check.
- **Human approval:** **mandatory** before any synthetic MRI is introduced at all (scope §24 Q-7 — option (a) "no synthetic MRI anywhere" vs option (b) "clearly-labelled illustration outside evaluation" is an unresolved owner choice). Recommended default: **(a)**, which keeps the repository free of synthetic medical content entirely.

## 2.4 Single-paragraph canonical definition (for reuse in later reports)

> "Cross-modal validation" in this project means **paired cross-modal validation (T-1)**: comparing this project's ultrasound-model predictions against MRI-derived findings **for the same cases**, where the same-case linkage is established by **verified patient/study identifiers**. Every other arrangement is named differently and carries a different claim: independent external validation (T-2), modality-transfer/domain-shift analysis (T-3), synthetic-data experimentation (T-4), or absence of MRI validation (T-5). The expected outcome on the fixed primary dataset is **T-5**, pending Phase 1 verification (V-4); the roadmap's Phase 12 feasibility gate exists precisely to make that outcome explicit and honest rather than approximated.

## 2.5 Relation to existing Phase 0 artifacts

- **`reports/research_questions.md` RQ-6** is unchanged and consistent: its refutation condition ("verified paired MRI linkage") corresponds exactly to **State A**; its expected outcome corresponds to **State C**. No edit needed.
- **`reports/phase0_scope.md` §17** (Current MRI Uncertainty) remains accurate: MRI functionality is NOT available; the empty `backend/services/mri_validation.py` placeholder is not evidence of capability (F-9). The state taxonomy above now gives §17 a precise vocabulary, which later phases must adopt.
- **`PROGRESS.md` Phase 12** requires no text change: the feasibility gate already implements States A/C, and the synthetic-MRI rule already implements the T-4 restriction of State E. This decision record **interprets** the roadmap; it does not override it.
- **Phase 13 dependency:** the decision-support schema's "Cross-modal evidence: MRI concordant / discordant / **unavailable**" maps to States A / A / B·C·D·E respectively — i.e. "unavailable" is the schema's honest default unless State A is verified.

## 2.6 Insufficiency summary (what would NOT count as paired evidence)

For every state: class-label agreement; demographic overlap; acquisition-date or sequence-number inference; filename similarity without verified identifier semantics; an external dataset with breast MRI "of the same population"; expert assertion without documented linkage; any synthetic artifact. This list is binding for Phase 12's feasibility verdict.

## 2.7 Contradiction check against the roadmap

Deliberately verified (see §3 validation): the taxonomy above does not contradict any `PROGRESS.md` statement — Problem Statement and candidate contribution #4 retain their original wording (aspirational, gated by Phase 12), Phase 12's paired-only gate matches T-1, the Phase 13 "unavailable" default matches States B–E, and the synthetic-data rule matches T-4. The one tension — the Problem Statement's "same patient" language describing an outcome the data will likely not support — is already recorded as scope §2.3 and remains the project owner's accepted risk, not something this record may edit away.

## 2.8 Q-4 decision status

- **A-4a: APPROVED by the project owner (2026-09-16).** The five-label taxonomy (§2.2) and the canonical definition (§2.4) are **binding terminology for all later artifacts**.
- **A-4b: deferred by owner instruction** — the permissive T-3 reading of State B remains available but the recommended default ("no unlinked exploratory analysis") stands unless and until the owner reopens it at Phase 12. State B is not expected to arise (V-4 audit pending).
- **A-4c: deliberately deferred by owner instruction** (= scope §24 Q-8) — until the project reaches the point where an external MRI dataset is actually relevant (expected Phase 12). Until then, the roadmap's standing rule holds: no external dataset may be acquired or integrated without explicit owner approval.
- **A-4d: deliberately deferred by owner instruction** (= scope §24 Q-7) — until the project reaches the point where synthetic MRI is actually relevant (expected Phase 12/14). Until then, the recommended default applies: **no synthetic MRI anywhere**; any Phase 14 UI shows "MRI: unavailable".

The taxonomy is now the binding vocabulary; the deferred items affect only States D and E, which cannot arise before their Phase 12/14 relevance point.

---

## 4. Approval Record

| Decision | Scope of approval | Approved by | Date | Status |
|---|---|---|---|---|
| **D-1** — Repository structure mapping (§1.4, §1.7) | Hybrid layout: roadmap `src/` + `data/` + `configs/` + `tests/` for the research pipeline; existing `model/`, `backend/`, `frontend/`, `dataset/` keep their roles; raw data → gitignored `dataset/raw/`; committed manifests → `data/manifests/` | **Project owner** (in their capacity as project owner; not an advisor or external reviewer) | 2026-09-16 | **In effect as a working decision.** Alternatives O-1/O-2 closed. Physical structure creation deferred to Phase 1. |
| **D-2** — Cross-modal validation taxonomy (§2.2, §2.4) and evidence states A–E (§2.3) | Five-label taxonomy T-1…T-5 as binding terminology; per-state claims/limits as specified | **Project owner** (same capacity) | 2026-09-16 | **In effect as binding terminology.** Sub-items A-4b/A-4c/A-4d deferred as recorded in §2.8. |
| Q-2 (approval authority) | Resolution of *who must approve*: project-owner approval is sufficient; no external advisor/reviewer approval is required for project-level technical decisions | **Project owner** (self-determination of authority) | 2026-09-16 | Authority resolved, and the owner subsequently approved the Phase 0 **deliverables as a whole** the same day — Phase 0 marked Complete. |
| Q-7 / Q-8 (A-4d / A-4c) | Deferred until synthetic MRI or an external MRI dataset actually becomes relevant | **Project owner** (deferral instruction) | 2026-09-16 | **Deliberately open**; roadmap standing rules apply until reopened (expected Phase 12/14). |

**Attribution note (binding):** these approvals were made by the **project owner** and must be recorded and cited as such. They must **not** be stated or implied to be advisor, committee, or external-reviewer approval. This distinction matters because `PROGRESS.md`'s Phase 0 exit criterion is phrased around project team/advisor approval; as of this record the operative approval authority for project-level technical decisions is the project owner, whose approval of the deliverables as a whole (2026-09-16) closed Phase 0.

---

## 5. Self-validation of this record

Performed after writing; results recorded in the conversation transcript and summarized in the final response: re-read of all modified sections; D-1/D-2 cross-reference consistency; `PROGRESS.md` path-contradiction scan; cross-modal-definition contradiction scan across all four Phase 0 documents; verification that no Phase 1 implementation exists; verification that no dataset was downloaded or modified; verification that Phases 1–14 statuses are untouched; fabrication and clinical-claim scans over new/changed text.

---

*End of decision record. D-1 and D-2 approved by the project owner (2026-09-16) and in effect as working decisions; Q-3, Q-5, Q-6 remain open; Q-7/Q-8 deliberately deferred. No structure created, no data touched, no phase statuses changed.*
