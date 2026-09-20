# Phase 9 — E1 (`da_stage_b`) Dual Attention MIL Training Report

**Status:** Training complete (early-stopped). Test evaluation performed once (epoch-2 best checkpoint, held-out reporting only, threshold fixed at 0.5 — not tuned).

**Scope note:** This report now contains training/validation AND the single final frozen test evaluation. The test evaluation was performed exactly once after E1 training completed, using the frozen test manifest, frozen preprocessing, and the frozen epoch-2 best checkpoint. No threshold tuning occurred; test results were used for held-out reporting only, never for model selection or training. No performance ranking or verdict is made; comparisons with frozen Phase 6 baselines are descriptive reference values only.

---

## 1. Experiment identity

| Field | Value |
| --- | --- |
| Experiment ID | `da_stage_b` (E1) |
| Phase | 9 (Model Training, Optimization & Experiment Tracking) |
| Source commit | `f39b7cd0bb7d7acb149ba7ba6bf6f8c6b299b5f7` (Phase 8, pushed) |
| Seed | `20260918` |
| Initialization | Fresh random (owner decision O3/D-C); no B1/B3/B4/E2/pretrained weights |
| Artifacts | `experiments/dual_attention_results/da_stage_b/` |
| Driver state | `experiments/dual_attention_results/_state.json` |

## 2. Locked training protocol (as executed)

* Model: Phase 8 `DualAttentionMIL` (locked design) — trunk 24→48→96→128 + GAP → 128-d; Stage-1 SE channel attention r=16; Stage-2 Phase 7 ABMIL (reused unchanged); `Linear(128→1)` raw logit.
* Trainable parameters: **184,545** (state-dict elements 185,141; reported separately).
* Optimizer: Adam, lr `1e-3`. Loss: unweighted `BCEWithLogitsLoss` (no `pos_weight`, per D-E).
* Batching: one optimizer step per training bag (349 steps/epoch); seeded bag permutation.
* Augmentation: train-only horizontal flip p=0.5 (per-instance stream, seed+2). Validation: deterministic, unaugmented, materialized once.
* Max epochs 20; early stopping patience 5 on validation bag ROC-AUC; strict improvement (`>`) for best-checkpoint replacement; best checkpoint restored at completion.
* Threshold: fixed 0.5 (reporting only; never tuned).
* Execution: detached 1-epoch chunks with bit-exact resume (persisted model/optimizer/`epoch_rng`/`flip_rng`/torch RNG per chunk); chunk state removed by the finishing path.

## 3. Execution and resume history

Training ran across multiple Freebuff session interruptions using the chunk/resume mechanism. Every interruption resumed exactly from the persisted `next_epoch`; no epoch was rerun or skipped. The final interruption occurred after the epoch-7 launch; that detached process survived and completed the run.

Persisted per-epoch history (from `train_log.csv`, verbatim):

| Epoch | Train loss | Val bag ROC-AUC | Duration (s) | Note |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.676031 | 0.559640 | 367.0 | — |
| 2 | 0.642757 | **0.572393** | 366.7 | **Best checkpoint saved** |
| 3 | 0.605997 | 0.510128 | 327.3 | patience 1/5 |
| 4 | 0.563471 | 0.489122 | 3113.9 | patience 2/5 (heavy machine load) |
| 5 | 0.521543 | 0.540135 | 699.3 | patience 3/5 |
| 6 | 0.484012 | 0.540135 | 496.8 | patience 4/5; **exact tie did not replace best** (strict-improvement rule verified live) |
| 7 | 0.435423 | 0.507877 | 1081.0 | patience 5/5 → **early stopping triggered** |

Total recorded training time: ≈ 6452 s (~107.5 min) across 7 epochs.

Descriptive observation (not a verdict): train loss decreased monotonically (0.676 → 0.435) while validation bag ROC-AUC peaked at epoch 2 and did not improve thereafter — the early-stopping rule stopped the run 5 epochs after the last strict improvement, as designed.

## 4. Best checkpoint verification (read from disk)

`model/dual_attention/da_stage_b/best.pt` contains:

* `epoch: 2`
* `seed: 20260918`
* `validation_metric: 0.5723930982745686` (= `best_validation_bag_roc_auc` in `metrics.json`)
* Embedded config payload matching the locked protocol (fresh init; no inherited checkpoint: `inherited_checkpoint: null`)
* `preprocessing_version: 1.0.0`, `pipeline_version: 1.0.1`
* `frozen_test_sha256: 959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74` (matches the frozen Phase 4 test manifest on disk)

`_chunk_state.pt` is absent — removed by the finishing path, which also restored the best (epoch-2) weights before writing final artifacts.

## 5. Best-checkpoint validation metrics

Recomputed **from the persisted `val_predictions.csv`** (74 bags, all `split=val`) using the Phase 6 `classification_metrics` implementation:

| Metric | Value |
| --- | ---: |
| Validation bags | 74 |
| Bag ROC-AUC | 0.5723930982745686 (**exact match** to training-time value) |
| Bag PR-AUC | 0.449457583406735 |
| Accuracy @0.5 | 0.581081081081081 |
| Sensitivity / Recall @0.5 | 0.4838709677419355 |
| Specificity @0.5 | 0.6511627906976745 |
| Precision @0.5 | 0.5 |
| F1 @0.5 | 0.4918032786885246 |
| Confusion matrix @0.5 | TP=15, TN=28, FP=15, FN=16 |
| Probability range | [0.314190, 0.797325] |
| Threshold | 0.5 (fixed; never tuned) |

Independent ROC-AUC recomputation from the persisted epoch-2 predictions exactly matches the recorded training-time metric.

## 6. Descriptive reference comparison (frozen Phase 6 baselines)

Historical frozen reference values, recorded without ranking or verdict:

* B4 (standard mean-pooling MIL) best validation bag ROC-AUC: 0.833458 (frozen Phase 6 result)
* B1 best validation bag ROC-AUC: 0.867217 (frozen Phase 6 result)
* E1 (`da_stage_b`) best validation bag ROC-AUC: 0.572393 (this experiment)

The measured values differed. No "better/worse/best/winner" characterization is made. Any interpretation (including the near-chance level of the E1 validation result and the train-loss/val-divergence pattern) is hypothesis-generating only and must not be treated as a conclusion about the architecture.

## 7. Leakage and test isolation

* Test split was never loaded, evaluated, or predicted during E1; the Phase 9 driver's test stage was not invoked (`test: pending` in the persisted driver state).
* No test predictions or test metrics exist anywhere under `experiments/dual_attention_results/` (verified by file scan).
* Frozen test manifest SHA unchanged: `959f1cd3…714112b74`.
* Model selection / early stopping used validation bag ROC-AUC only; threshold fixed at 0.5.
* Train-only augmentation remained train-only; validation was unaugmented.
* No persistent Phase 9 embedding cache was created (O2); no `_cache/*.npy` artifacts exist.

## 8. Frozen-artifact integrity (verified after training)

* B1 checkpoint MD5: `cbf558baf00b3ae4876e614aee429795` — unchanged
* B4 checkpoint MD5: `ef3fb5a8c53978621884678991f9671a` — unchanged
* Frozen test manifest SHA-256: `959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74` — unchanged
* Phase 5.5 loader, Phase 6 trainer/driver, Phase 7 modules, Phase 8 architecture: unmodified (no source changes during training)
* Phase 6 experiment artifacts: untouched

## 9. Artifact inventory

```
experiments/dual_attention_results/
  _state.json                 driver state (da_stage_b: frozen, test: pending, full chunk history)
  da_stage_b/
    config.yaml               locked protocol snapshot (written at first chunk)
    train_log.csv             epochs 1–7 (train loss, val bag ROC-AUC, seconds)
    metrics.json              best_epoch=2, best ROC-AUC, epochs_run=7, stopped_early=true,
                              threshold=0.5, threshold_tuned=false, test_evaluated=false,
                              parameter counts, seed
    val_predictions.csv       74 validation bags: bag_id, source_group_id, y_true,
                              probability, prediction, split
    seed.txt / env.txt / provenance.json
model/dual_attention/da_stage_b/best.pt   epoch-2 weights (restored best)
```

## 10. Limitations (pre-registered, unchanged from Phase 9 planning)

* Small dataset (496 bags / 9,016 instances; single frozen split — no cross-validation or external validation in scope).
* `source_group_id` defines bag boundaries, **not** verified patient/study/lesion identity; patient-level overlap remains unassessable.
* Candidate near-duplicate grouping and augmentation lineage are handled only by the frozen Phase 1–4 constraints.
* CPU-only environment; per-epoch runtime varied with machine load (327–3114 s observed).
* The E1 validation result is near chance level on this split; this is a descriptive outcome of one pre-registered run, not evidence about dual attention in general.
* No MRI/cross-modal validation; research prototype with no clinical validity.

## 11. Final test evaluation (performed once, after training completion)

The one-time Dual Attention MIL test evaluation was performed after E1 training completion using the frozen epoch-2 best checkpoint and the frozen test manifest (SHA256 `959f1cd3…714112b74`). It was executed exactly once under a single-evaluation guard (the guard refuses a second run; `experiments/dual_attention_results/_state.json` now records `test: evaluated`).

**E1 best checkpoint:** `model/dual_attention/da_stage_b/best.pt` — epoch 2, seed 20260918, validation_metric 0.5723930982745686, inherited_checkpoint None, preprocessing_version 1.0.0, pipeline_version 1.0.1, frozen_test_sha256 `959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74`. MD5 `258710649fb0e6979f64fc1e7ccfc28f`, SHA256 `5cc9883a60b7317b3a3baa72cd08464d95a97d59b6fe88ac2c9d9ea5d649f94a`, 37 state-dict tensors.

**Test-set facts (frozen):** 73 bags / 1,350 instances; 43 benign bags / 30 malignant bags; bag sizes {16×43, 21×29, 53×1}; threshold 0.5 fixed; no augmentation.

**Test metrics (bag-level, independently recomputed from the persisted `test_predictions.csv` and verified to match the driver output):**

| Metric | Value |
| --- | ---: |
| bags | 73 |
| ROC-AUC | 0.5302325581395348 |
| PR-AUC | 0.4709920889816891 |
| accuracy @0.5 | 0.589041095890411 |
| sensitivity / recall @0.5 | 0.4666666666666667 |
| specificity @0.5 | 0.6744186046511628 |
| precision @0.5 | 0.5 |
| F1 @0.5 | 0.4827586206896552 |
| confusion matrix @0.5 | TP=14, TN=29, FP=14, FN=16 |
| probability range | [0.3443276286125183, 0.6819376945495605] |

**Persistent artifacts:** `experiments/dual_attention_results/da_stage_b/test_predictions.csv` (73 test bags) and `experiments/dual_attention_results/da_stage_b/test_metrics.json`.

**Descriptive frozen-baseline comparison (read-only; no winner/selection declared):**

| Model | Level | ROC-AUC | PR-AUC | Acc | Sens | Spec |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E1 Dual Attention MIL (`da_stage_b`, epoch 2) | bag | 0.5302 | 0.4710 | 0.5890 | 0.4667 | 0.6744 |
| B1 Simple CNN (frozen) | image | 0.7178 | 0.7299 | 0.6133 | 0.8278 | 0.4070 |
| B3 bag aggregation of B1 (frozen) | bag | 0.7426 | 0.7094 | 0.5890 | 0.8333 | 0.4186 |
| B4 Mean-Pooling MIL (frozen) | bag | 0.7628 | 0.7853 | 0.7808 | 0.5667 | 0.9302 |

The E1 test ROC-AUC was lower than the three frozen Phase 6 baselines on this held-out split. No winner, ranking, score, or overall superiority is declared or implied.

**Frozen-artifact integrity after the test evaluation:**

* frozen test manifest SHA unchanged: `959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74`
* B1 checkpoint MD5 unchanged: `cbf558baf00b3ae4876e614aee429795`
* B4 checkpoint MD5 unchanged: `ef3fb5a8c53978621884678991f9671a`
* E1 Dual Attention best checkpoint MD5 unchanged: `258710649fb0e6979f64fc1e7ccfc28f`

**Test isolation:** the test split was accessed only by this single final evaluation; it was never used during E1 training or model selection. No E2 artifacts exist; no MRI/cross-modal artifacts exist; no persistent Phase 9 embedding cache exists.

**Current driver state:** `da_stage_b: frozen`, `test: evaluated`.

No further Phase 9 test evaluation is authorized (single-evaluation rule).
