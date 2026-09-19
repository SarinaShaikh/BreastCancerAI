# Phase 7 — Multiple Instance Learning Pipeline (Implementation Report)

**Status:** Implemented and validated (uncommitted — awaiting owner review).
**Date:** 2026-09-19. **Scope:** general MIL infrastructure + single-attention
ABMIL only. No Dual Attention MIL, no MRI/cross-modal functionality, no
full training experiment, no test-split access, no performance claims.

---

## 1. Phase objective

Implement the reusable MIL pipeline — feature extraction → instance
embeddings → aggregation → classification — that Phase 8's Dual Attention
model will build upon, with correct variable-bag-size handling, per the
approved roadmap (PROGRESS.md, Phase 7 section).

## 2. Relationship to Phases 3–6

Phase 7 **consumes** the existing stack; it never redefines it:

| Layer | Source of truth | Phase 7 relationship |
|---|---|---|
| Bag definition | Phase 3 (`bag_manifest.csv`): one Phase 2 `source_group_id` = one bag | consumed verbatim; no regrouping |
| Instance | one image file | consumed verbatim |
| Labels | directory-encoded, `benign=0`, `malignant=1` (Phase 1/3) | consumed verbatim |
| Splits | frozen Phase 4 manifests (test SHA `959f1cd3…714112b74`) | read-only; test split never accessed |
| Preprocessing | Phase 5 pipeline v1.0.1 / config v1.0.0 (224×224 grayscale, train-derived normalization baked into cache) | consumed via the Phase 5.5 loader's exact inverse decode; no re-normalization |
| Loader | Phase 5.5 `src/mil/bag_dataset.UltrasoundBagDataset` | the only data path; wrapped by a thin adapter (owner decision D0) |
| Baselines | Phase 6 B1/B3/B4 (frozen; test evaluated once) | untouched; trunk constants reused, weights never loaded |

## 3. Authoritative bag definition

One bag = one Phase 2 `source_group_id` (source-image/augmentation family),
per Phase 3. **`source_group_id` is NOT a verified patient/study/lesion
identifier** — Phase 1 verified no such identifiers exist and none are
fabricated. `source_key` remains `inferred_from_filename_not_verified_identifier`
on every provenance row.

## 4. Instance definition

One instance = one image file of that source group (one row of the frozen
split manifests, one entry of the Phase 3 `instance_list`).

## 5. Actual bag-size statistics (dataset ground truth)

496 bags / 9,016 instances: **{14 × 1, 16 × 285, 21 × 209, 53 × 1}**
(min 14, max 53, mean ≈ 18.18, median 16). **No single-instance bag exists
in this dataset**; size-1 handling is a software-robustness property, not an
observed data property.

## 6. Per-split statistics

| Split | Bags | Instances | Size distribution |
|---|---|---|---|
| train | 349 | 6,327 | {14 × 1, 16 × 199, 21 × 149} |
| val | 74 | 1,339 | {16 × 43, 21 × 31} |
| test | 73 | 1,350 | {16 × 43, 21 × 29, 53 × 1} |

(Splits themselves were never rebuilt; numbers quoted from the Phase 3/4
records and re-asserted by the Phase 5.5 loader on every construction.)

## 7. Feature extractor architecture

`src/mil/feature_extractor.InstanceFeatureExtractor` — exactly the
implemented Phase 6 trunk topology: 4 × [Conv3×3(s=1,p=1) → BatchNorm →
ReLU → MaxPool2×2], channels **(24, 48, 96, 128)**, spatial 224→112→56→28→14,
then Global Average Pooling. Input `(N, 1, 224, 224)` → output `(N, 128)`.
Channels are **imported from `simple_cnn.CHANNELS`** so the Phase 7 extractor
is architecturally identical to the implemented baselines.

> **Documented deviation from the instruction's "channels 1→16→32→64→128":**
> the implemented Phase 6 trunk uses 24/48/96/128 — `simple_cnn.py` records
> that the original 16/32/64/128 draft measured 97,761 params, below the
> owner's 150k–250k floor, and was widened accordingly during Phase 6.
> Embedding dimensionality (128) and block topology match the instruction.
> Phase 7 follows the *implemented* architecture to remain a faithful
> pipeline for Phase 8; the instruction text predates that Phase 6 widening.

**Parameter counts (measured):** extractor 163,536 · attention 16,640 ·
head 129 · full pipeline **180,305** trainable params.

## 8. Initialization

Fresh random initialization only (owner decision D1): `seed_extractor(20260918)`
seeds Python `random`, NumPy, and Torch before construction. No B1/B4
checkpoint loading, no ImageNet/pretrained weights, no downloads. The frozen
Phase 6 baselines are never touched (verified by hash post-implementation).

## 9. Freeze / train modes

`freeze_trunk(True)` sets `requires_grad=False` on all extractor parameters
and switches to eval mode (deterministic BN); `freeze_trunk(False)` keeps
the trunk trainable. Verified by T10: frozen mode produces **no** trunk
gradients while attention/head gradients flow; joint mode propagates
gradients into the trunk.

## 10. 128-D embedding representation

Post-GAP instance embedding `h_i ∈ R^128` per image; a bag of n instances
yields `H ∈ R^{n×128}`. Embeddings from a *frozen* extractor on unaugmented
Phase 5 images are the only cacheable form (§11).

## 11. Cache architecture

`<dest_root>/<cache_key>/` (the cache **key** — a SHA-256 of the manifest —
is the only directory segment; `cache_version` is recorded *inside*
`manifest.json` as part of the cache identity, not as a path segment) with
`embeddings.npy` (float32, rows in loader instance order), `provenance.csv`
(one row per instance), `manifest.json`, `manifest_key.txt`. Full-split cache
(9,016 × 128 × float32) ≈ **4.4 MB**. Current implementation status: the
cache is implemented and fully tested (T11–T13) against temporary
directories; **no persistent cache directory was created during Phase 7**
(the smoke test writes only to a temp dir), because Phase 7 authorizes no
training run that would consume it. It becomes valuable when a future
frozen-trunk attention experiment is authorized.

## 12. Cache provenance

`provenance.csv` records per row: `row_index`, `bag_id`, `source_group_id`,
`label`, `numeric_label`, `instance_index_in_bag`, `image_path`,
`processed_relpath`, `md5`, `split`, `source_key`,
`source_key_status` — every cached vector is traceable to its source
instance (T13 cross-checks `md5` against the loader).

## 13. Cache invalidation

The cache key is the **SHA-256 of the canonical-JSON manifest** containing:
cache version, component, architecture, embed dim, **weights SHA-256**
(content hash of all parameters + BN buffers), trainable-state flag, seed,
preprocessing version (1.0.0), pipeline version (1.0.1), split name, split
manifest SHA-256, frozen test SHA. Any identity change ⇒ different key ⇒
cache miss ⇒ rebuild. **No modification times are used anywhere.** Loading
re-verifies the stored manifest bit-exactly, plus shape/dtype/finiteness.

## 14. Single-attention mathematical specification

Non-gated ABMIL (Ilse et al. 2018 form), one mechanism only:

```
u_i = tanh(V h_i + b)        V ∈ R^{128×128}, b ∈ R^{128}
s_i = wᵀ u_i                 w ∈ R^{128}        (scalar per instance)
a_i = softmax_i(s_i)         normalized WITHIN each bag only
z   = Σ_i a_i h_i            bag representation ∈ R^{128}
logit = Linear(128 → 1)(z)   raw logit; BCEWithLogitsLoss applies the sigmoid
```

Attention weights sum to 1 **within each bag** (T4, tol 1e-6); they never
mix across bags (each bag is scored and normalized independently). With
uniform weights this reduces exactly to B4's mean pooling (useful sanity
anchor). With 16,640 attention + 129 head parameters, the new trainable
surface beyond the trunk is 16,769.

## 15. Variable-bag handling

Primary path: **dynamic/list processing** (`forward_list`) — each bag's
instances are stacked `(n_i, 1, 224, 224)` and processed in one forward;
no padding, no duplication, any `n_i ≥ 1`. This matches the proven B4
pattern and the CPU/memory constraints (largest bag = 53 instances ≈ 10.6 MB
as float32 tensors).

## 16. Padding/masking behavior (convenience path)

`forward_padded` accepts `(B, N_max, 1, 224, 224)` + bool mask `(B, N_max)`.
Masked positions receive attention logits `−inf` before the softmax, so
their weights are **exactly 0** and padded values can never contribute.
A fully-masked bag raises `ValueError` (softmax over all-−inf is undefined)
rather than returning garbage. The padded and dynamic paths are
**numerically equivalent** on identical inputs (T5, atol 1e-5; equivalence
stems from softmax invariance under −inf masking). Inference determinism is
bitwise under fixed seed (T9).

## 17. Bag representation

`z = Σ a_i h_i ∈ R^128` — the attention-weighted instance mean; fed to the
head (§18). Exposed by the pipeline together with attention weights,
embeddings, and logits for testing/documentation.

## 18. Classification head

`BagClassifier`: `Linear(128 → 1)`, **raw logit**, no sigmoid inside the
model — directly compatible with `BCEWithLogitsLoss`. 129 parameters.
No threshold logic in the model; the fixed 0.5 threshold belongs to
evaluation code, and **no threshold tuning occurred or is authorized in
Phase 7**.

## 19. Smoke-test methodology

Owner decision D4/D2: infrastructure smoke only — **no performance
experiment**. Deterministic seeded sample (`random.Random(20260918)` stream;
the sample deterministically includes the smallest bag) of **8 bags from the
train split only** (sizes {14, 16, 16, 16, 21, 21, 21, 21} = 146 instances —
the 14-instance train bag was captured). Executed in T14:
1. joint mode: forward → `BCEWithLogitsLoss` → backward → one Adam step;
   trunk gradients verified present;
2. frozen mode: frozen extraction → cache write → cache read (bitwise
   round-trip) → attention/head forward → finite logits.

No epochs, no validation split, no test split, no metrics claim. Sample
selection, bag loading, instance tensors, extraction, attention, bag
representation, classification, backward, and variable-size handling are all
exercised — the roadmap exit criterion.

## 20. Smoke-test result

**PASS.** 8/8 bags (146 instances) processed end-to-end; joint loss ≈ 0.6937
(recorded descriptively — a one-step value on an untrained model, **not** a
performance result); trunk gradients present; frozen-path logits finite;
cache round-trip bitwise-identical.

## 21. Tests and outcomes

`tests/test_phase7_mil_pipeline.py` — standalone convention (V-suite style):

| # | Check | Result |
|---|---|---|
| T1 | sizes {1,2,3,14,16,21,53} without shape errors (stacked + padded) | PASS |
| T2 | exactly one binary logit per bag | PASS |
| T3 | one-instance bag attention ≡ 1.0 (robustness; no such bags in data) | PASS |
| T4 | per-bag attention sums to 1 (tol 1e-6) | PASS |
| T5 | padded ≡ dynamic path | PASS |
| T6 | masked positions get exactly zero attention | PASS |
| T7 | no NaN/Inf anywhere | PASS |
| T8 | embedding dim exactly 128 | PASS |
| T9 | bitwise-deterministic inference (fixed seed) | PASS |
| T10 | freeze modes: trunk-grad semantics | PASS |
| T11 | cache write→load→compare bitwise | PASS |
| T12 | cache refuses trainable extractor | PASS |
| T13 | every cache row provenance-traceable (md5 match) | PASS |
| T14 | end-to-end smoke (train-only, joint backward + frozen cache) | PASS |
| T15 | boundary scan (no Phase 8 mechanisms; CNN MaxPool2d allowed) | PASS |

**15/15 PASS.** Full historical regressions re-run: Phase 1 **11/11** ·
Phase 2 **11/11** · Phase 3 **13/13** · Phase 4 **11/11** · Phase 5 **11/11**
· Phase 5.5 **19/19** · Phase 6 **16/16**.

T15 distinguishes CNN spatial max-pooling (legitimate trunk component,
asserted present) from max-pooling-as-MIL-aggregation (banned identifiers/
constructs: `.max(dim`, `.amax(`, `nn.MaxPool1d`, `topk`, etc., asserted
absent from the aggregation module and banned as identifiers everywhere in
Phase 7 code).

## 22. Leakage controls

- Split manifests read-only; frozen test SHA re-verified unchanged
  (`959f1cd3…714112b74`).
- **No test access:** `instance_generation.PHASE7_SPLITS = ("train", "val")`;
  `iter_split_batches`/`open_split` raise on `"test"`; the cache builder
  raises on `"test"`. Verified by source scan (the only `"test"` literals in
  Phase 7 modules are rejection guards and identity constants).
- No regrouping, no identifier fabrication, no cross-split relationships
  introduced; Phase 4 duplicate/candidate constraints untouched.
- Cache: frozen-extractor-only writes (T12), split-separated manifests
  (split name + split-manifest SHA inside the key), no augmented embeddings
  (augmentation is a train-time pixel transform; the deterministic cache
  stores embeddings of unaugmented Phase 5 images only), no test cache.
- Phase 6 baselines/checkpoints untouched (md5-verified below, §24).

## 23. Reproducibility controls

Seed 20260918 on all three RNG streams before init and per re-run;
deterministic loader ordering (ascending `bag_id`/`image_path`) inherited;
dedicated `random.Random(seed)` sampling stream (never the global RNG);
fixed-seed inference verified **bitwise** (T9); cache keys are exact content
hashes; provenance chains every embedding to its instance md5. Honest
limitation: bitwise determinism is claimed only for the tested CPU inference
path (T9); training-step determinism across machines is not asserted.

## 24. Frozen-artifact verification (post-implementation)

- B1 `best.pt` md5 `cbf558baf00b3ae4876e614aee429795` — unchanged ✓
- B4 `best.pt` md5 `ef3fb5a8c53978621884678991f9671a` — unchanged ✓
- `git diff` over `experiments/`, `model/`, `data/`, `src/training/`,
  `src/mil/bag_dataset.py`, `src/preprocessing/` — **empty** ✓
- State machine: `b1/b3/b4: frozen, test: evaluated` — unchanged ✓

## 25. Runtime considerations

Measured during the smoke test on this CPU-only machine (2.7 GiB RAM): the
8-bag/146-instance joint forward+backward+step completed in seconds. Scaling
to full-split epochs is bounded by the trunk, i.e. Phase 6's measured
~340–410 s/epoch idle (up to ~2.2 h under external load) applies to a future
joint-training run; a frozen-trunk run would amortize extraction via the
4.4 MB cache and train the 16.8k attention/head parameters at bag-step cost.
No long-running job was started during Phase 7 (D4).

## 26. Limitations

- Infrastructure phase: **no performance evidence** of any kind is claimed;
  the smoke loss (≈0.6937, one step, untrained) is not a metric.
- The attention benefit over mean pooling is **untested** (D4 defers any
  training comparison to a separate authorized phase).
- The one-instance-bag case is robustness-only; the dataset has none.
- Cache is implemented/tested but not persisted at scale (no authorized
  consumer yet).
- Bitwise determinism asserted for CPU inference under fixed seed only.

## 27. Single-instance hypothetical robustness case

If a size-1 bag appeared: softmax over one score yields attention weight
exactly 1.0 and `z = h_1` (T3) — degenerate but well-defined; the padded
path handles it identically (mask row with a single True). The roadmap's
"very small bags may degrade attention value" risk therefore does not
materialize in this dataset (min size 14) but is handled by construction.

## 28. Actual 14/16/21/53 bag-size implications

All observed sizes flow through the identical dynamic path (T1); the
14-instance train bag is deterministically included in the smoke sample; the
53-instance test bag exists only in the frozen test split and was **not**
executed against (its size class is exercised synthetically in T1/T2).
Largest-bag memory ≈ 10.6 MB float32 — no batching pressure at this scale.

## 29. Phase 8 boundary

Phase 7 contains exactly **one** attention mechanism (instance-level
non-gated ABMIL). Dual/gated/transformer/cross-bag attention, top-k pooling,
and max-pooling aggregation are absent from the implementation and banned by
the T15 boundary scan. Phase 8 (Dual Attention MIL) will be a separate,
individually authorized phase building on these modules.

## 30. Exit criteria checklist

- [x] MIL pipeline runs end-to-end on a sample of the training split
      without errors (T14, 8 bags / 146 instances, joint + frozen modes)
- [x] Unit test: variable bag sizes processed without shape errors (T1)
- [x] Unit test: forward pass produces correctly shaped bag-level output (T2)
- [x] Documentation complete (this report)
- [x] All historical regression suites remain passing

## Deliverables

- `src/mil/instance_generation.py` — thin adapter (D0) over the Phase 5.5
  loader: bag → ordered instance tensors + per-instance provenance;
  test-split guard; seeded deterministic sampling helper.
- `src/mil/feature_extractor.py` — fresh trunk extractor (D1) with
  freeze/train modes + leakage-safe hash-keyed embedding cache (D3) that
  refuses trainable extractors.
- `src/mil/aggregation.py` — single-attention ABMIL aggregator
  (stacked + masked-padded paths), `Linear(128→1)` head, composable
  `AttentionMILPipeline`.
- `tests/test_phase7_mil_pipeline.py` — T1–T15 (15/15 PASS).
- This report.

*Research-prototype infrastructure only. No clinical validity, no diagnostic
performance claim, and no generalization claim is made or implied.*
