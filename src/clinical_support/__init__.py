"""Phase 13: research decision-support layer (owner-approved margin-based path).

Module:
    decision_support — deterministic, structured, research-only presentation of
                       frozen E1 Dual Attention MIL outputs.

Hard boundaries (enforced by tests/test_phase13_decision_support.py):
    * NOT a clinical diagnostic system: outputs are research artifacts only
    * no clinical risk score, no BI-RADS, no staging/grading, no prognosis,
      no recurrence/survival prediction, no treatment recommendation,
      no triage, no calibrated-clinical-probability claim
    * uncertainty flag = model-score proximity to the fixed 0.5 decision
      threshold ONLY (margin 0.1, pre-registered Phase 10/11 convention) —
      it is NOT clinical/diagnostic uncertainty and NOT patient risk
    * no calibration is performed and the model score is never presented as
      a calibrated clinical probability
    * cross-modal evidence is hard-coded UNAVAILABLE (Phase 12 outcome);
      no MRI imports, no MRI data, no synthetic MRI
    * every output carries the mandatory research-only disclaimer and
      preserves the Phase 11 boundary: Model Attention != Clinical Explanation
    * reads frozen artifacts only; no training, no checkpoint modification,
      no manifest modification, no test-set evaluation
"""
