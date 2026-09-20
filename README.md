# BreastCancerAI
Predicting Breast Cancer from Multi-Modal Images using Dual Attention Multiple Instance Learning

## Project Status (Phase 12 feasibility outcome, 2026-09-21)

- **Implemented capability:** ultrasound-only breast classification (public Kaggle ultrasound dataset) with a Dual Attention Multiple Instance Learning model — a research prototype, not clinically validated.
- **Intended research objective (unchanged):** cross-validate ultrasound prediction accuracy against MRI reports of the same patient.
- **Current feasibility (Phase 12 audit — NOT CURRENTLY FEASIBLE with available data):** the ultrasound dataset contains no verified patient/study/lesion identifiers (`source_group_id`/`source_key` are filename-derived grouping identifiers, not clinical identifiers), no local MRI data exists, and no publicly accessible paired US+MRI dataset with a documented linkage mechanism suitable for this project was identified during this investigation.
- **No MRI or cross-modal validation has been implemented or performed**, and no patient-level linkage has been fabricated. Future paired validation would require an appropriate paired dataset or a verified linkage mechanism.
- Details: `reports/phase12_mri_feasibility_report.md` (authoritative evidence record) and `PROGRESS.md` (Phase 12 section).
