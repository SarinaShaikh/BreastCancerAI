# Phase 8 — Dual Attention MIL Architecture (Implementation Report)

**Status:** Implemented and validated (uncommitted — awaiting owner review).
**Date:** 2026-09-19. **Scope:** architecture, implementation, tests, and an
infrastructure smoke test ONLY. No training, no performance evaluation, no
test-set access, no persistent cache, no clinical claims.

---

## 1. Objective

Implement the project's core research contribution per the owner-locked
design (2026-09-19): a genuine **two-stage Dual Attention MIL** — Stage 1
channel/feature attention, Stage 2 instance attention — built BESIDE the
frozen Phase 7 single-attention reference implementation, with both
attention weight families exposed for later explainability (Phase 11).

## 2. Approved decisions implemented

| Decision | Value implemented |
|---|---|
| D-A | Two-stage Channel → Instance attention (exactly as specified below) |
| D-B | Staged strategy recorded; **no stage executed** (Phase 9 scope) |
| D-C | Fresh random init, seed 20260918; no B1/B3/B4/pretrained weights |
| D-D | Bottleneck r = 16 |
| D-E | Unweighted `BCEWithLogitsLoss` (recorded; not run) |
| D-F | No persistent cache created; Phase 7 cache rules authoritative |
| D-G | Phase 8 = architecture + tests + smoke only |

## 3. Mathematical formulation (locked)

```
Stage 1 — channel attention (per instance, NO cross-instance interaction):
    c_i   = sigmoid( W2( ReLU( W1 h_i ) ) )     W1 ∈ R^{16×128}, W2 ∈ R^{128×16}
    h̃_i  = c_i ⊙ h_i                            elementwise gate
Stage 2 — instance attention (the UNMODIFIED Phase 7 ABMIL, reused by
composition — not reimplemented):
    u_i   = tanh( V h̃_i + b )                   V ∈ R^{128×128}, b ∈ R^{128}
    s_i   = wᵀ u_i                               w ∈ R^{128}
    a_i   = softmax_i(s_i)                       normalized WITHIN each bag only
Aggregation + head:
    z     = Σ_i a_i h̃_i                          z ∈ R^{128}
    logit = Linear(128 → 1)(z)                   raw logit (BCEWithLogitsLoss)
```

**Why genuinely dual:** two mechanisms on different axes (feature/channel vs
instance), different parameterizations (SE-MLP vs ABMIL scoring), composed
multiplicatively — stage 1 changes *what stage 2 sees*. Verified empirically:
TDA-11 proves neutralizing either stage changes outputs (so neither is
dead/bypassed); TDA-12 proves uniform gates (c ≡ 1) reduce the model EXACTLY
to the Phase 7 single-attention ABMIL (atol 1e-4 logit / 1e-6 attention) —
i.e., Phase 7 is the c≡1 special case of Phase 8.

**Interpretability:** a_i = per-instance attention within the bag; c_i =
per-instance, per-channel importance (128 values/instance). Both are model
attributions for later analysis — **NOT** validated clinical explanations.

## 4. Tensor shapes

| Stage | Shape |
|---|---|
| One image | (1, 224, 224) |
| Trunk output | (N, 128, 14, 14) |
| Instance embedding (GAP) | (N, 128) |
| Channel gates c / h̃ | (N, 128) / (N, 128) |
| Attention a | (N,) — sums to 1 within the bag |
| Bag representation z | (128,) |
| Logit | (1, 1) raw; sigmoid applied only at inference for probability |
| Dynamic batch | list of per-bag tensors → logits (B,) |
| Padded batch | (B, N_max, 1, 224, 224) + mask (B, N_max) → logits (B,), attention (B, N_max), gates (B, N_max, 128) |

## 5. Parameter accounting (locked counts, erratum applied)

| Component | Trainable params |
|---|---|
| CNN trunk (Phase 7 extractor, 24→48→96→128) | 163,536 |
| Stage-1 channel attention (W1 2,048+16; W2 2,048+128) | **4,240** |
| Stage-2 instance attention (Phase 7 ABMIL) | 16,640 |
| Classifier Linear(128→1) | 129 |
| Head excluding trunk | **21,009** |
| **Full jointly trainable model** | **184,545** |
| State-dict elements (+596 BN buffer elements) | **185,141** (reported separately, never conflated) |

Erratum note (owner-approved): the planning report's stage-1 figure 4,256
was arithmetic bookkeeping; the locked equations yield 4,240 (+16 exactly).
TDA-10 asserts the corrected table mechanically; no count was weakened.

## 6. Architecture diagram

```
                 bag (n_i images, 1×224×224 each)
                              │
              Phase 7 InstanceFeatureExtractor
              4 × [Conv3×3 → BN → ReLU → MaxPool2×2]
              channels 24 → 48 → 96 → 128  (spatial 224→14)
                              │ GAP
                    h_i ∈ R^128  (per instance)
                              │
        ┌─────────────────────┴─────────────────────┐
        │ STAGE 1 — channel attention (SE, r=16)    │
        │ c_i = σ(W2 ReLU(W1 h_i))   (per instance) │
        │ h̃_i = c_i ⊙ h_i                           │
        └─────────────────────┬─────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │ STAGE 2 — instance attention (Phase 7     │
        │ ABMIL, reused unchanged)                  │
        │ u_i = tanh(V h̃_i + b); s_i = wᵀu_i;       │
        │ a_i = softmax_i(s_i)   (within bag only)  │
        └─────────────────────┬─────────────────────┘
                              │  z = Σ a_i h̃_i   ∈ R^128
                    Linear(128 → 1)   → raw bag logit
```

## 7. Implementation map

- `src/mil/dual_attention.py` — `ChannelAttention` (stage 1),
  `DualAttentionAggregator` (stage-1 gate → h̃ → stage-2 via the **reused**
  `SingleAttentionAggregator`; stacked + masked-padded paths; both weight
  families returned; −inf masking; fully-masked bag raises).
- `src/models/dual_attention_mil.py` — `DualAttentionMIL` (Phase 7
  extractor + aggregator + `BagClassifier`); `forward_bag`/`forward_list`
  (primary)/`forward_padded` (convenience, equivalence-test-mandated);
  `param_counts()` (requires_grad-aware, per component) and
  `state_dict_element_count()` kept strictly separate;
  `EXPECTED_PARAMS`/`EXPECTED_STATE_DICT_ELEMENTS` constants asserted by
  TDA-10.
- `configs/dual_attention_config.yaml` — locked design, recorded (not
  executed) training settings, cache/leakage policy, forbidden-formulation
  list.
- Phase 7 modules: **zero modifications** (verified: `git diff` over
  `src/mil/{aggregation,feature_extractor,instance_generation}.py` empty).

## 8. Variable bag-size handling

Primary dynamic/list path (no padding/duplication, any n_i ≥ 1); padded
convenience path with −inf stage-2 logits (exactly-zero masked attention,
zero contribution to z; fully-masked bag raises). Sizes {1, 2, 3, 14, 16,
21, 53} exercised in TDA-01/02 (1–3 synthetic robustness only — the dataset
has sizes {14×1, 16×285, 21×209, 53×1}; min 14, max 53, mean ≈ 18.18, no
hard-coded sizes anywhere). Padded ≡ dynamic verified numerically (TDA-05:
logits, z, attention, and gates all within 1e-5).

## 9. Tests and outcomes (TDA-01…TDA-14 — 14/14 PASS)

| # | Check | Result |
|---|---|---|
| TDA-01 | sizes {1,2,3,14,16,21,53}, both paths | PASS |
| TDA-02 | one raw logit/bag; structural no-sigmoid proof; \|logit\|>1 at amplified input | PASS |
| TDA-03 | one-instance bag: a = 1.0 exactly, z = h̃₁ (robustness) | PASS |
| TDA-04 | per-bag attention sums to 1 (1e-6), never across bags | PASS |
| TDA-05 | padded ≡ dynamic (logits, z, attention, gates) | PASS |
| TDA-06 | masked instances: exactly-zero attention; z equals stacked-path z over real instances | PASS |
| TDA-07 | no NaN/Inf anywhere | PASS |
| TDA-08 | bitwise-deterministic inference (fixed seed) | PASS |
| TDA-09 | gradients through BOTH stages; frozen trunk ⇒ no trunk grads, joint ⇒ trunk grads | PASS |
| TDA-10 | locked counts: 184,545 trainable / 185,141 state-dict elements | PASS |
| TDA-11 | ablation guard: neutralizing stage 2 changes attention (1e-8 threshold vs measured ~3e-5 effect); ×50-amplified stage 2 changes attention & z; stage-1 gates→~0 collapses z; weights restored bitwise | PASS |
| TDA-12 | c ≡ 1 reduces EXACTLY to Phase 7 ABMIL (gates exactly 1.0; attention 1e-6, z 1e-5, logit 1e-4) | PASS |
| TDA-13 | boundary scan: no gated/transformer/self/cross attention, no top-k/max MIL aggregation, no MRI identifiers; CNN MaxPool2d allowed; stage-2 reuses `SingleAttentionAggregator`; test-split guards present | PASS |
| TDA-14 | authorized train-only smoke: 8 bags / 146 instances (sizes {14,16,16,16,21,21,21,21}, includes the 14-instance bag); joint forward→unweighted BCEWithLogitsLoss→backward→one Adam step (trunk + stage-1 grads verified); frozen-path temp-cache write→read bitwise→finite logits; temp artifacts deleted | PASS |

TDA-11 thresholds are set from **measured margins**, not guesses: at random
init the stage-2 effect on the scalar logit is ~1e-7 (a linear readout of a
near-uniform attention vector) while the attention vector itself moves ~3e-5
(float noise ~1e-9) — the test therefore asserts on the attention vector and
on amplified contrast, so it would fail if stage 2 were removed, bypassed, or
neutralized by construction.

Full regression re-run: Phase 1 **11/11** · Phase 2 **11/11** · Phase 3
**13/13** · Phase 4 **11/11** · Phase 5 **11/11** · Phase 5.5 **19/19** ·
Phase 6 **16/16** · Phase 7 **15/15**.

## 10. Regression-guard note (owner-authorized)

Phase 6's V6B-17 contained a Phase-6-era future-guard `assert not
src/models/ exists`, which collided with the owner-approved Phase 8 roadmap
placement `src/models/dual_attention_mil.py`. Per the owner's explicit
decision (2026-09-19) it was updated minimally: exactly
`src/models/__init__.py` and `src/models/dual_attention_mil.py` are exempt;
**any other file under `src/models/` still fails the guard**; all other
V6B-17 protections (no ResNet/B2, no pretrained downloads, no
transfer_learning.py, gitignore policy) are untouched. Same file-level
pattern as the V-11/V2-11 precedents.

## 11. Leakage controls

- Frozen Phase 4 manifests untouched; test SHA
  `959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74`
  re-verified.
- **No test access**: the smoke test and all tests load `train` only via the
  Phase 7 adapter, whose `PHASE7_SPLITS=("train","val")` guard rejects
  `test` (TDA-13 asserts the guard is present).
- Bag semantics unchanged: one Phase 2 `source_group_id` = one bag; one
  image = one instance; labels directory-encoded; no new grouping; no
  patient/study/lesion identifiers fabricated (`source_key_status` remains
  `inferred_from_filename_not_verified_identifier`).
- No persistent embedding cache created (`experiments/mil_embeddings/` does
  not exist); TDA-14 uses only a temp dir, deleted on exit; frozen-only,
  never-test, never-augmented cache rules inherited unchanged.
- Phase 6 artifacts byte-identical: B1 md5 `cbf558ba…`, B4 md5 `ef3fb5a8…`,
  state machine `b1/b3/b4: frozen, test: evaluated`.

## 12. Reproducibility controls

Seed 20260918 on python/NumPy/Torch before every construction
(`seed_extractor`); dedicated seeded sampling stream; fresh init from locked
constants only; bitwise-deterministic CPU inference verified (TDA-08);
parameter identities pinned by TDA-10; config records seed, versions
(preprocessing v1.0.0, pipeline v1.0.1), and the frozen test SHA. Bitwise
determinism is claimed only for the tested CPU inference path.

## 13. Runtime considerations

Smoke test (8 bags / 146 instances, joint forward+backward+step) completes
in seconds on the CPU-only machine (2.7 GiB RAM). A future joint epoch is
bounded by the trunk (Phase 6 measured ~340–410 s/epoch idle; ~2 h+ under
load); a frozen Stage-A run would train only the 21,009 head parameters on
cached embeddings. **No long-running job was started in Phase 8.**

## 14. Limitations

- Infrastructure phase: **no performance evidence exists** for the Dual
  Attention model; the smoke loss (≈0.6966, one step, untrained) is
  descriptive only, not a metric.
- Attention-vs-mean-pooling benefit untested (deferred to the authorized
  training phase, D-B/D-G).
- Stage-1 channel gates are per-instance; no spatial localization is
  provided by this design (spatial attention was explicitly rejected in D-A).
- The scalar-logit insensitivity at init (TDA-11 note) is expected behavior
  of a linear readout on a near-uniform attention vector, not a defect;
  training will re-scale attention as learning proceeds.
- Single-instance bags do not exist in this dataset; size-1 handling is
  robustness-only.

## 15. Phase 9 boundary

Phase 8 contains exactly the approved two-stage dual attention and no
training. Stage A (frozen trunk, may use the Phase 7 cache) and Stage B
(joint, cache forbidden) belong to the separately authorized training
phase, which will define its own protocol on top of this locked
architecture. No MRI/cross-modal functionality exists.

## 16. Exit criteria checklist

- [x] Model architecture implemented, documented, and passing all unit
      tests (roadmap exit criterion; TDA-01…14, 14/14)
- [x] Forward pass on sample bags of varying sizes (TDA-01/02/14)
- [x] Attention weight dimensions match instance counts (TDA-01/04)
- [x] Both attention weight families returned (TDA-01/05/11)
- [x] Architecture documentation + diagram (this report, §3–§6)
- [x] All Phase 1–7 regression suites remain passing
- [x] Genuine dual attention demonstrated (TDA-11 ablation + TDA-12
      reduction), not "attention in name only"

*Research-prototype infrastructure only. No clinical validity, no
diagnostic performance claim, and no generalization claim is made or
implied. Attention weights are model attributions, not clinical
explanations.*
