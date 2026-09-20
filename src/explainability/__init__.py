"""Phase 11: explainability — attention visualization & Grad-CAM (A1, owner-approved).

Submodules:
    attention_visualization  — MIL attention extraction/visualization/export
                               (train/val ONLY; the test split is a hard boundary)
    gradcam                  — Grad-CAM on the frozen CNN trunk (optional task,
                               approved by owner decision A1)

Hard boundaries (enforced by tests/test_phase11_explainability.py):
    * no training, no optimizer/loss code, no checkpoint modification
    * no test-split access: no test images, no new test forward passes,
      no test attention/Grad-CAM maps, no test-label use
    * no threshold logic, no MRI imports
    * attention weights and Grad-CAM are MODEL ATTRIBUTIONS, never clinical
      explanations, and neither is validated as lesion localization
      (no ground-truth masks exist — IoU/mask-overlap validation is impossible).
"""
