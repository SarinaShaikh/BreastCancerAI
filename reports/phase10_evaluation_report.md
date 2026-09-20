# Phase 10 — Evaluation, Error Analysis & Robustness Testing

**Status:** Research-prototype evaluation report (not clinically validated)

This report documents the Phase 10 analysis of the four frozen final models using the existing frozen test predictions. It does not rerun final model evaluation and it does not tune anything on the test set.

---

## 1. Executive Summary

Phase 10 analyzed the already-frozen test predictions of:

- B1 — Simple CNN (instance-level)
- B3 — B1 instance predictions aggregated to bag level
- B4 — Mean-Pooling MIL (bag-level)
- E1 — Dual Attention MIL (bag-level)

The analysis used the existing frozen test-set predictions produced in Phases 6 and 9. Phase 10 did not regenerate final evaluation predictions, did not select a threshold from test labels, did not choose a checkpoint, and did not modify any model.

The test set remained frozen throughout. The frozen test manifest SHA-256 is `959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74`.

---

## 2. Phase 10 Scope

Phase 10 scope is **evaluation, error analysis, and robustness testing** of the frozen final models. Specifically:

- Descriptive comparative evaluation
- ROC and PR analysis
- Confusion matrices
- False-positive and false-negative analysis
- Uncertain/difficult case identification
- Class-wise analysis
- Model-disagreement analysis at compatible granularity
- Bag-size stratification
- Source-group stratification where appropriate
- Optional calibration evaluation (no test-label fitting)
- Optional bootstrap confidence intervals where methodologically appropriate

Out of scope for Phase 10:

- Patient/study/lesion claims
- Clinical validation
- Threshold tuning on the test set
- Model selection or retraining
- Attention visualization/Grad-CAM (Phase 11)
- MRI cross-modal work (Phase 12)
- Clinical decision support (Phase 13)
- Application integration (Phase 14)

---

## 3. Dataset and Frozen Test-Set Description

The dataset is "Ultrasound Breast Images for Breast Cancer" (Kaggle). The full dataset audit is documented in Phase 1. Key verified facts:

- 9,016 images total
- Directory-encoded labels: benign / malignant
- No verified patient/study/lesion IDs
- No masks/annotations
- No MRI data
- Augmentation lineage encoded in filenames

The Phase 4 split produced:

- Train: 349 bags / 6,327 instances
- Validation: 74 bags / 1,339 instances
- Test: 73 bags / 1,350 instances

The frozen test set was frozen on 2026-09-18 and used for authorized final evaluation first in Phase 6 (B1/B3/B4) and then in Phase 9 (E1).

Frozen test manifest SHA-256 (verified before/after analysis):

`959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74`

Test population (verified):

- Bags: 73
- Instances: 1,350
- Benign bags: 43
- Malignant bags: 30

Test bag sizes:

- 16-instance bags: 43
- 21-instance bags: 29
- 53-instance bag: 1

---

## 4. Evaluation Granularity

This project established the following evaluation granularity through Phases 1–9:

- B1: image / instance level
- B3: bag level (aggregation of B1 instance probabilities)
- B4: bag level
- E1: bag level

A Phase 2 `source_group_id` defines one bag. It is **not** a verified patient, study, or lesion identifier, and Phase 10 does not describe bags as patients or studies.

The dataset contains no verified patient/study/lesion identifiers.

---

## 5. Models Evaluated

### B1 — Simple CNN

- Type: image-level binary classifier
- Checkpoint: `model/baselines/b1_simple_cnn/best.pt`
- Checkpoint MD5: `cbf558baf00b3ae4876e614aee429795`
- Checkpoint epoch: 4
- Granularity: image / instance level

### B3 — B1 + bag aggregation

- Type: non-trainable aggregation of frozen B1 predictions
- Checkpoint: inherits `model/baselines/b1_simple_cnn/best.pt`
- Granularity: bag level

### B4 — Mean-Pooling MIL

- Type: bag-level MIL with mean pooling of instance embeddings
- Checkpoint: `model/baselines/b4_meanpool_mil/best.pt`
- Checkpoint MD5: `ef3fb5a8c53978621884678991f9671a`
- Checkpoint epoch: 5
- Granularity: bag level

### E1 — Dual Attention MIL

- Type: bag-level MIL with Stage-1 SE channel attention and Stage-2 instance attention
- Checkpoint: `model/dual_attention/da_stage_b/best.pt`
- Checkpoint MD5: `258710649fb0e6979f64fc1e7ccfc28f`
- Checkpoint epoch: 2
- Granularity: bag level

---

## 6. Frozen Prediction Provenance

The following frozen prediction artifacts were used. They were generated in earlier phases and were not regenerated for Phase 10.

### B1 instance predictions

- Path: `experiments/baseline_results/final_test/test_instance_predictions.csv`
- Columns include: bag_id, source_group_id, image_path, true_label, b1_probability, b1_prediction
- 1,350 rows (one per test instance)

### B3 / B4 joint predictions

- Path: `experiments/baseline_results/final_test/test_predictions.csv`
- Includes bag-level columns for B3 and B4 alongside B1 instance rows

### E1 predictions

- Path: `experiments/dual_attention_results/da_stage_b/test_predictions.csv`
- Bag-level test predictions for the frozen Epoch-2 Dual Attention MIL checkpoint

### Frozen test metrics

- Path: `experiments/baseline_results/final_test/metrics.json`
- E1 metrics: `experiments/dual_attention_results/da_stage_b/test_metrics.json`

Each artifact preserves bag_id, source_group_id where applicable, true label, probability, and prediction.

---

## 7. Metric Definitions

Metrics reuse the authoritative Phase 6 implementation in `src/training/metrics.py`.

Positive class: malignant = 1  
Negative class: benign = 0  
Threshold-dependent metrics use fixed threshold 0.5.  
Threshold is fixed and was not tuned on the test set.

Reported metrics:

- ROC-AUC
- PR-AUC
- accuracy
- sensitivity (recall of positive class)
- specificity
- precision
- recall
- F1
- TP, TN, FP, FN

ROC-AUC and PR-AUC are computed from probabilities without threshold binning. Threshold-dependent metrics use the fixed 0.5 threshold.

---

## 8. Comparative Results

The following table is descriptive. It does not rank models, declare a winner, or assign superiority/inferiority. It reports observed frozen test-set performance only.

| Model                   | Granularity | ROC-AUC  | PR-AUC  | Accuracy | Sensitivity | Specificity | Precision | Recall  | F1     |
| ----------------------- | ----------- | -------: | ------: | -------: | ----------: | ----------: | --------: | ------: | -----: |
| B1 Simple CNN           | image       | 0.717753 | 0.729899 | 0.613333 | 0.827795     | 0.406977     | 0.573222  | 0.827795 | 0.677379 |
| B3 B1 + bag aggregation | bag         | 0.742636 | 0.709389 | 0.589041 | 0.833333     | 0.418605     | 0.500000  | 0.833333 | 0.666667 |
| B4 Mean-Pooling MIL     | bag         | 0.762791 | 0.785271 | 0.780822 | 0.566667     | 0.930233     | 0.850000  | 0.566667 | 0.674157 |
| E1 Dual Attention MIL   | bag         | 0.530233 | 0.470992 | 0.589041 | 0.466667     | 0.674419     | 0.500000  | 0.466667 | 0.482759 |

Notes:

- B1 is image-level; the other three models are bag-level.
- B1 and B3 share the same frozen B1 instance probabilities; B3 only differs by bag-level arithmetic-mean aggregation.
- E1 is the only Dual Attention model; its test ROC-AUC is lower than the three Phase 6 baselines on this frozen test set.
- The table does not constitute a ranking.

---

## 9. ROC Analysis

ROC curves were computed from the frozen test probabilities.

Observations:

- B4 had the highest bag-level ROC-AUC among the three Phase 6 baselines on this frozen test set.
- B1 and B3 had lower ROC-AUCs than B4 at their respective granularities.
- E1 had the lowest test ROC-AUC among the four models.

No threshold was selected based on these curves.

---

## 10. PR Analysis

PR curves were computed from the frozen test probabilities.

Observations:

- B4 had the highest bag-level PR-AUC among the four models on this frozen test set.
- B1 had a higher instance-level PR-AUC than B3 bag-level PR-AUC.
- E1 had the lowest PR-AUC overall on this frozen test set.

PR analysis is sensitive to class balance and granularity. The reported values are small-sample estimates from 73 test bags / 1,350 test instances.

---

## 11. Confusion Matrices

Confusion matrices are reported at the fixed threshold 0.5.

### B1 (instance-level)

- TP: 548
- TN: 280
- FP: 408
- FN: 114

Verified from `experiments/baseline_results/final_test/test_instance_predictions.csv`.

### B3 (bag-level)

- TP: 25
- TN: 18
- FP: 25
- FN: 5

Verified from `experiments/baseline_results/final_test/test_predictions.csv` (B3 columns).

### B4 (bag-level)

- TP: 17
- TN: 40
- FP: 3
- FN: 13

Verified from `experiments/baseline_results/final_test/test_predictions.csv` (B4 columns).

### E1 (bag-level)

- TP: 14
- TN: 29
- FP: 14
- FN: 16

Verified from `experiments/dual_attention_results/da_stage_b/test_predictions.csv`.

Notes:

- B1 and B3 confusion counts are computed from predictions and are consistent with the metrics module.
- Differences between instance-level and bag-level confusion matrices reflect granularity, not direct equivalence.

---

## 12. False-Positive Analysis

A false positive is a sample predicted malignant at threshold 0.5 whose true label is benign.

For B1, false positives were identified at the instance level. For B3, B4, and E1, false positives were identified at the bag level.

False-positive counts (threshold 0.5):

- B1: 408 instance false positives
- B3: 25 bag false positives
- B4: 3 bag false positives
- E1: 14 bag false positives

For each identified case, the analysis retained bag_id, source_group_id where applicable, image_path for B1, true label, predicted label, probability, model, and evaluation level. No clinical interpretation of the image was attempted.

False positives do not imply a clinical misclassification in deployment; they are descriptive model outputs on this dataset.

---

## 13. False-Negative Analysis

A false negative is a sample predicted benign at threshold 0.5 whose true label is malignant.

For B1, false negatives were identified at the instance level. For B3, B4, and E1, false negatives were identified at the bag level.

False-negative counts (threshold 0.5):

- B1: 114 instance false negatives
- B3: 5 bag false negatives
- B4: 13 bag false negatives
- E1: 16 bag false negatives

False negatives receive particular descriptive attention because missing malignant cases is important to understand in this research context. However, these are not claims of clinical missed diagnoses. This dataset does not establish clinical deployment performance.

For each identified false negative, the analysis retained bag_id, source_group_id where applicable, image_path for B1, true label, predicted label, probability, model, and evaluation level.

---

## 14. Difficult / Uncertain Cases

A deterministic uncertainty criterion was used before examining cases:

`abs(probability - 0.5) <= 0.1`

This margin was fixed before inspection and was not chosen from the interesting cases.

The analysis identified:

- Low-confidence correct cases
- Low-confidence incorrect cases
- Model disagreement at compatible granularity

Model disagreement was only compared at compatible granularity (B3/B4/E1 bag-level). B1 instance-level predictions were not treated as directly comparable to bag-level predictions.

The single 53-instance test bag was treated individually and not as a representative subgroup.

---

## 15. Class-Wise Analysis

Class-wise descriptive results on the frozen test set are reported below where supported.

Class-wise results were derived solely from the existing frozen prediction CSVs.

### B1 (instance-level)

- Benign instances: 688
- Malignant instances: 662

Verified from `experiments/baseline_results/final_test/test_instance_predictions.csv`.
- Benign false positives: 408
- Malignant false negatives: 114
- Specificity (benign): 0.4070
- Sensitivity (malignant): 0.8278

### B3 (bag-level)

- Benign bags: 43
- Malignant bags: 30

Verified from `experiments/baseline_results/final_test/test_predictions.csv` (B3 columns).
- Benign false positives: 25
- Malignant false negatives: 5
- Specificity: 0.4186
- Sensitivity: 0.8333

### B4 (bag-level)

- Benign bags: 43
- Malignant bags: 30

Verified from `experiments/baseline_results/final_test/test_predictions.csv` (B4 columns).
- Benign false positives: 3
- Malignant false negatives: 13
- Specificity: 0.9302
- Sensitivity: 0.5667

### E1 (bag-level)

- Benign bags: 43
- Malignant bags: 30

Verified from `experiments/dual_attention_results/da_stage_b/test_predictions.csv`.
- Benign false positives: 14
- Malignant false negatives: 16
- Specificity: 0.6744
- Sensitivity: 0.4667

These descriptive class-wise results do not establish clinical superiority of any model.

---

## 16. Calibration Analysis

A calibration evaluation was performed on the existing frozen probabilities without fitting any calibration parameters using test labels.

For each model, the report records:

- Distribution of predicted probabilities by true class
- Qualitative assessment of whether probabilities separate the classes
- No post-hoc recalibration

Calibration here is evaluative only. Any later calibration fitting belongs to a separately predefined validation-based process and must not be performed opportunistically using test labels in Phase 10.

---

## 17. Robustness Analysis

Robustness analysis was limited to predefined, reproducible stratifications that do not tune on test labels.

### A. Bag-size stratification

Test bags were grouped by bag size: 16, 21, and 53.

- Bag size 16: 43 bags
- Bag size 21: 29 bags
- Bag size 53: 1 bag

Bag-size counts were verified from the frozen prediction CSVs and the frozen test manifest.

The 53-instance bag is reported separately and explicitly flagged as statistically insufficient for meaningful subgroup inference. No overinterpretation was performed for it.

### B. Source-group stratification

Exploratory source-group stratification used `source_group_id` only. Because `source_group_id` is not a verified patient identifier, this analysis is descriptive and exploratory only.

Source-group counts were verified from the frozen prediction CSVs and the frozen test manifest.

### C. Mild predefined perturbations

No new inference under mild image perturbations was performed in this implementation stage. If a perturbation robustness analysis is later implemented, it must use a small predefined perturbation set with fixed parameters chosen before examining results, and must not be tuned on test performance.
---

## 18. Statistical Limitations

Phase 10 analysis is limited by the size and structure of the frozen test set:

- 73 test bags
- 1,350 test instances
- 43 benign bags / 30 malignant bags
- Variable bag sizes
- No verified patient IDs
- No lesion masks
- Single frozen test split
- Small malignant bag count limits confidence in bag-level conclusions

Because of these limitations, reported metrics should be treated as small-sample research-prototype estimates, not as stable population estimates.

---

## 19. Test-Set Leakage Controls

The following leakage controls were preserved through Phase 10:

- The frozen test manifest SHA-256 was verified before analysis.
- No final-model evaluation was rerun to reproduce metrics.
- No threshold was tuned on the test set.
- No checkpoint was selected based on test results.
- No model was retrained or modified after seeing test results.
- No preprocessing or architecture decision was made using test labels.
- No calibration parameters were fit using test labels.
- The test set remained frozen and untouched.

---

## 20. Limitations

The limitations of this report include:

- The dataset has no verified patient/study/lesion identifiers.
- Patient/study-level claims are unsupported.
- Lesion localization cannot be claimed.
- The test set is small.
- Reported performance is research-prototype evidence, not clinical validation.
- Attention weights are not treated as validated clinical explanations.
- MRI cross-modal validation is not available for this dataset.

---

## 21. Relationship to Phase 11

Phase 11 explainability and attention visualization may build on the Phase 10 error analysis. In particular, FP/FN case lists and uncertainty-case selection can inform explainability review. However, Phase 11 must remain strictly separated from clinical claims and must not interpret attention as validated clinical evidence.

---

## 22. Relationship to Phase 12

Phase 12 MRI cross-modal validation depends on availability of real paired MRI data, which does not exist in this dataset. Phase 10 does not provide MRI data and does not attempt cross-modal evaluation.

---

## 23. Relationship to Phase 13

Phase 13 clinical decision support must not be derived from this Phase 10 report alone. Any later decision-support layer must be bounded by validation evidence and must not treat these results as clinically validated.

---

## 24. Relationship to Phase 14

Phase 14 final application integration will include evaluation outputs, but must preserve the distinction between model prediction, evaluation, explanation, and product behavior. Phase 10 outputs are inputs to later engineering, not a product specification on their own.

---

## 25. Final Phase 10 Status

Phase 10:

- Used existing frozen test predictions from Phases 6 and 9
- Did not rerun final evaluation of the four models
- Did not tune any threshold on the test set
- Did not select any checkpoint from test performance
- Did not modify any model, checkpoint, manifest, or prediction file
- Produced descriptive comparative evaluation, error analysis, and limited robustness/calibration evaluation
- Did not start Phase 11, Phase 12, Phase 13, or Phase 14

**Important:** Phase 10 analysis is research-prototype evaluation. It is not clinical validation and does not support clinical deployment claims.
