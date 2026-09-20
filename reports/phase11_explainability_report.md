# Phase 11 — Explainability & Ultrasound Attention Visualization

**Status:** Research-prototype explainability report (not clinically validated)

This report documents the Phase 11 attention-visualization and Grad-CAM work performed with the frozen Phase 9 E1 Dual Attention MIL checkpoint. It adds **no** new evaluation, **no** training, and **no** performance claims.

---

## 1. Executive Summary

Phase 11 visualizes the two attention mechanisms of the approved Phase 8 Dual Attention MIL model — stage-2 **instance MIL attention** and stage-1 **channel/feature gates** — and adds an owner-approved **Grad-CAM** cross-check of the frozen CNN feature extractor. All extraction ran on **train/validation bags only**; the test split was never accessed. Outputs are: attention-ranked thumbnail grids, channel-gate feature-space figures, Grad-CAM overlays, and a machine-readable attention export (CSV + JSON).

Every visualization in this report is a **model attribution**. None of them is a clinical explanation, and none has been validated against anatomical ground truth.

---

## 2. Phase 11 Scope

In scope (per `PROGRESS.md` §Phase 11 and the owner authorization A1–A5):

- Per-instance attention-weight extraction and visualization within bags
- Deterministic top-k important-instance identification per prediction
- Attention overlays on ultrasound instance thumbnails (attention-ranked grids)
- Feature-space channel-gate visualization (128-d), explicitly labeled as feature-space attribution
- Grad-CAM on the CNN feature extractor (owner decision A1) as a per-pixel cross-check
- Machine-readable attention export (owner decision A5)
- Explicit "Model Attention vs. Clinical Explanation" disclaimer
- Explicit statement that mask-overlap/IoU validation is not possible

Out of scope: MRI/cross-modal work (Phase 12), clinical decision support (Phase 13), application integration (Phase 14), any new evaluation metrics, any threshold or preprocessing change, any training.

---

## 3. Inputs and Frozen Provenance

| Input | Role | Status |
| --- | --- | --- |
| `model/dual_attention/da_stage_b/best.pt` | Frozen E1 checkpoint (epoch 2, seed 20260918) | MD5 `258710649fb0e6979f64fc1e7ccfc28f`, verified before and after every run |
| `src/models/dual_attention_mil.py` + `src/mil/` | Phase 7/8 architecture, used unchanged | Frozen; zero modifications in Phase 11 |
| `data/manifests/{bag_manifest,train_split,val_split}.csv` | Bag/split membership + provenance | Frozen; test manifest SHA `959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74` unchanged |
| `experiments/dual_attention_results/da_stage_b/val_predictions.csv` | Frozen validation predictions for deterministic case selection | Read-only |
| `data/processed/{train,val}/` | Phase 5 processed thumbnails | Read-only |

`source_group_id` / `bag_id` are bag boundaries derived from filenames. They are **not** verified patient, study, or lesion identifiers: every exported row carries `source_key_status=inferred_from_filename_not_verified_identifier`. No patient/study/lesion identity is asserted, implied, or fabricated anywhere in Phase 11.

---

## 4. Case Selection (Deterministic, Pre-Registered)

The selection rule was fixed **before** any attention map was generated and was never tuned on visualization output. All cases come from train/validation only (owner decision A2 — the test split is a hard boundary).

1. From the frozen `val_predictions.csv`, classify each bag at the fixed reporting threshold 0.5 (inherited from Phases 6/9/10; never tuned) into `correct_benign`, `correct_malignant`, `false_positive`, `false_negative`; take the **first** bag in ascending `bag_id` order from each non-empty class.
2. Uncertain cases: `|probability − 0.5| ≤ 0.1` (margin fixed a priori); first 2 bags in `bag_id` order.
3. Train fallback (E1 has no persisted train predictions): the Phase 7 `deterministic_sample(seed=20260918)` bag, which also covers the train split's smallest bag size.
4. De-duplicate by `bag_id`.

Selected cases (5):

| Bag | Split | Category | Instances | Bag probability |
| --- | --- | --- | ---: | ---: |
| `bag-01e91c2df96c` | val | correct_benign | 16 | 0.4478 |
| `bag-0e9e3f410b4b` | val | correct_malignant | 21 | 0.5847 |
| `bag-05d8f8e1658b` | val | false_positive | 16 | 0.6963 |
| `bag-01a4298f9ca8` | val | false_negative | 21 | 0.4498 |
| `bag-244973d6f450` | train | train_fallback | 14 | 0.3877 |

Correctly classified, incorrectly classified (FP and FN), and low-confidence cases are all represented, as required. Probabilities shown are the model outputs recomputed identically during extraction (validation values match the frozen `val_predictions.csv`).

---

## 5. Attention Extraction Details

- Loader: the authoritative Phase 5.5 `UltrasoundBagDataset` via the Phase 7 thin adapter; bags in ascending `bag_id`, instances in ascending `image_path` (the loader's deterministic order).
- Model: frozen E1 `DualAttentionMIL`, `model.eval()`, `requires_grad=False` on every parameter, every forward under `torch.no_grad()`, CPU.
- Extracted per bag: stage-2 instance attention `(n_i,)`, stage-1 gates `(n_i, 128)`, channel-refined embeddings `h̃ (n_i, 128)`, trunk embeddings `(n_i, 128)`, bag logit and sigmoid probability.
- Verification: attention normalization `Σ attention = 1` asserted per bag (tolerance 1e-6; observed exactly 1.0); all values finite; repeated extraction bitwise-identical; checkpoint MD5 unchanged before/after.

Ranking is deterministic: descending attention, ties broken by ascending instance order (stable, order-based), ranks 1-based.

---

## 6. Grad-CAM Implementation Details

- **Method:** standard Grad-CAM (Selvaraju et al., 2017).
- **Target layer:** `model.extractor.features[14]` — the final ReLU of conv block 4 (128 channels, 28×28 grid for 224×224 input; the block-4 MaxPool2d at index 15 then yields 14×14 before GAP). The architecture is **not modified**; the layer is only observed via a forward hook.
- **Computation:** forward activations `A` (1, 128, 28, 28); backward gradient `∂Y/∂A` with `Y` = the bag logit for one instance; channel weights `α_k = GAP(∂Y/∂A)`; CAM = min-max-normalized `ReLU(Σ_k α_k A^k)`; bilinear upsample 28×28 → 224×224; alpha-blended overlay (α = 0.45, fixed) on the Phase 5 processed image.
- **Frozen-weights discipline:** gradients flow **only to the input image** (all parameters `requires_grad=False`, eval-mode BatchNorm running statistics, no optimizer exists anywhere in Phase 11). The single `.backward()` call in `src/explainability/gradcam.py` is this documented input-gradient mechanism; the test suite proves parameter gradients remain `None` (T11-18) and the checkpoint MD5 is unchanged (T11-20).
- **Determinism:** bitwise-identical CAMs across repeated runs (T11-18).
- **Boundary:** every Grad-CAM entry point validates instance provenance against train/val and raises on any test-split input (T11-19).

---

## 7. Visualization Outputs

Generated under `reports/figures/phase11/` (15 figures):

- `attention_grid_{split}_{bag_id}.png` — attention-ranked thumbnail grids: top-4 highest-attention and up to 4 lowest-attention instances per bag, each annotated with attention weight and rank. This is **instance-level** evidence: one scalar per image, not a pixel map.
- `channel_gates_{split}_{bag_id}.png` — per-instance profiles of the 128 stage-1 gates for the highest-attention instances, titled *"Feature-space channel gates — model attribution, not image-space localization."* The 128 axes are embedding channels, not pixels and not anatomical regions.
- `gradcam_{split}_{bag_id}_inst{i}.png` — instance image, Grad-CAM heatmap, and overlay for the single highest-attention instance per bag, titled as model attribution, not clinical explanation.

---

## 8. Machine-Readable Attention Export (A5)

Under `reports/phase11/`:

- **`attention_export.csv`** — 88 rows (one per instance: 21+16+16+21+14), columns: `bag_id, source_group_id, source_key_status, split, image_path, md5, processed_relpath, instance_order, attention, rank, predicted_probability, true_label, prediction_at_0.5, category, gate_mean, gate_max, gate_argmax, gate_vector_ref`.
- **`attention_export.json`** — per-bag attention vectors, full 128-d gate vectors, checkpoint identity (path + MD5), `splits_used=["train","val"]`, `test_split_accessed=false`, `uncertainty_margin=0.1`, and the attention-semantics statement.

Labels and probabilities are included only where legitimately available (validation predictions from the frozen Phase 9 artifact; the train fallback carries the loader-derived label and the recomputed model probability, with no persisted-prediction coupling). Two independent exports of the same extraction are byte-identical (T11-15).

---

## Model Attention vs. Clinical Explanation

**This section is the required disclaimer.**

- Attention weights (stage-1 gates and stage-2 instance attention) are **model-internal attribution signals**: they describe which instances and which learned feature channels influenced this particular model's bag representation.
- Grad-CAM is a **model attribution visualization**: it shows where the frozen CNN's logit-gradient mass concentrates on one input image. It is computed from the same model, the same frozen weights, and the same task.
- **Neither constitutes a clinical explanation.** None of these visualizations explains disease, anatomy, or biology, and none has been reviewed or endorsed by any clinical authority.
- **Neither has been validated as an anatomical or lesion-localization method.** The dataset contains no ground-truth masks or annotations (verified in Phase 1), so no localization accuracy, overlap, or correctness of these highlights can be established or claimed.
- Attention mechanisms are known to be imperfect and sometimes unstable proxies of "what the model looked at"; their agreement or disagreement with Grad-CAM on this dataset is **not** evidence of correctness.
- This system is a **research prototype**. It is not clinically validated, not clinically deployable, and its outputs must not be interpreted as clinically validated evidence or used for any diagnostic purpose.

---

## 10. Mask-Overlap / Localization Validation

**Mask-overlap / IoU validation is not possible with the available data.**

Phase 1 exhaustively verified that the dataset contains no ground-truth segmentation masks, lesion annotations, or region labels. No masks were fabricated and no pseudo-masks were substituted. Consequently no IoU, Dice, or any other localization-accuracy metric is reported anywhere in Phase 11. Explainability validation therefore remains **qualitative and descriptive only**, and this limitation is stated explicitly rather than glossed over.

---

## 11. Test-Set Boundary (A2)

The test split was never accessed by Phase 11: no test images were loaded, no test forward passes were performed, no test attention or Grad-CAM maps were generated, and no test labels or test predictions were used for case selection. Case selection used only the frozen validation predictions plus the deterministic train fallback. The frozen test manifest remains byte-identical (`959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74`), and the Phase 9 single-evaluation guard was not touched.

---

## 12. Leakage Controls

- No training, no optimizer, no loss, no weight update anywhere in `src/explainability/` (asserted by T11-22).
- No threshold tuning: 0.5 appears only as the fixed reporting convention for carrying persisted predictions.
- No checkpoint selection: the single frozen E1 checkpoint is MD5-verified before and after every operation (T11-20).
- No test-split access by any Phase 11 code path (T11-11, T11-19, T11-22).
- No preprocessing modification: the Phase 5 cache and loader are consumed read-only.
- Frozen artifacts (`data/manifests/`, `experiments/`, `model/`) unchanged (T11-21).
- No patient/study/lesion identifiers created or implied; `source_key_status` carried on every exported row (T11-13).

---

## 13. Limitations

- No ground-truth masks → localization validity of attention/Grad-CAM is **unestablished**; only qualitative review is possible.
- Very small case set (5 bags) chosen for deterministic demonstration, not statistical coverage; the single 14-instance train bag is the split's minimum and is not a representative-size subgroup by itself.
- Stage-2 attention is one scalar per instance; it cannot attribute within an image. Only Grad-CAM provides a per-pixel view, and it inherits all Grad-CAM limitations (resolution bounded by the 28×28 grid upsampled; sensitive to the chosen target layer; highlights model-relevant regions, not necessarily diagnostically relevant ones).
- Stage-1 gates describe 128 learned embedding channels whose semantics are not labeled; profiles are shown for transparency, not interpretation.
- The model's observed test performance is modest (Phase 9/10 reports); visualizing the attributions of a weak model does not improve the model.
- All statements above are research-prototype observations on a small ultrasound dataset; external validation would be required for any claim beyond this report.

---

## 14. Relationship to Later Phases

- **Phase 12 (MRI cross-modal validation):** untouched. Phase 11 creates no MRI code, data, or claims.
- **Phase 13 (clinical decision support):** must not treat these visualizations as clinical evidence; any decision-support layer is bounded by the disclaimer in §9.
- **Phase 14 (application integration):** may surface these figures only with the same attribution-vs-explanation labeling preserved.

---

## 15. Final Phase 11 Status

- Deterministic attention extraction, ranking, and export implemented and tested (Phase 11 suite: see `tests/test_phase11_explainability.py`).
- Visualizations generated for representative train/val cases covering correct, incorrect (FP/FN), and uncertain categories.
- Grad-CAM cross-check implemented per owner decision A1 without architecture modification.
- Test split untouched; frozen artifacts unchanged; no training; no performance claims added.

Phase 11 explainability tooling is implemented as a research prototype with the explicit limitation that no ground-truth masks exist for quantitative localization validation.
