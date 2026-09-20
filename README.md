# BreastCancerAI
Predicting Breast Cancer from Multi-Modal Images using Dual Attention Multiple Instance Learning

## Project Status (Phase 12 feasibility outcome, 2026-09-21)

- **Implemented capability:** ultrasound-only breast classification (public Kaggle ultrasound dataset) with a Dual Attention Multiple Instance Learning model — a research prototype, not clinically validated.
- **Intended research objective (unchanged):** cross-validate ultrasound prediction accuracy against MRI reports of the same patient.
- **Current feasibility (Phase 12 audit — NOT CURRENTLY FEASIBLE with available data):** the ultrasound dataset contains no verified patient/study/lesion identifiers (`source_group_id`/`source_key` are filename-derived grouping identifiers, not clinical identifiers), no local MRI data exists, and no publicly accessible paired US+MRI dataset with a documented linkage mechanism suitable for this project was identified during this investigation.
- **No MRI or cross-modal validation has been implemented or performed**, and no patient-level linkage has been fabricated. Future paired validation would require an appropriate paired dataset or a verified linkage mechanism.
- Details: `reports/phase12_mri_feasibility_report.md` (authoritative evidence record) and `PROGRESS.md` (Phase 12 section).
- **Research decision-support layer (Phase 13, 2026-09-21):** deterministic, structured presentation of the frozen E1 model outputs — benign/malignant **model prediction** (fixed threshold 0.5), malignant-class model score (**not calibrated**), margin-based uncertainty flag (`|score − 0.5| ≤ 0.1`, threshold-proximity only), top-k **model-attributed image instances** (Model Attention ≠ Clinical Explanation), and the research safety flag "Clinical review recommended". This is **model-level decision support only**: it is NOT a diagnosis, clinical risk score/stratification, BI-RADS, prognosis, treatment recommendation, or clinical validation of any kind, and no MRI/cross-modal evidence exists (`cross_modal_evidence: UNAVAILABLE`). See `reports/phase13_decision_support_spec.md` and `PROGRESS.md` (Phase 13 section).
