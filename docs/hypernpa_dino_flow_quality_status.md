# HyperNPA DINO/Flow Quality Status

Current status: broad 1k/10k generalized DINO/flow HyperNPA quality is not
proven. The previous evidence pointed to a modeling/objective gap, not a
scale-only issue. The code now has the first real adapter rectified-flow path
and DINO token-grid conditioning, but quality-scale flow runs still need to be
completed before making a generalization claim.

## Completed Evidence

### 10k DINO CLS + Patch Mean

Artifact:
`artifacts/hyper2d_adapter_bank_omnisvg_10k_dino_canonical_h1024_valselect`

- Train/holdout split: 9000 / 1000.
- Objective: `supervised-static-lora-vector-mse`, not rectified flow.
- Status: `conditioning_not_rectified_flow`, `quality_ready=false`,
  `hypernet_generalization_validated=false`.
- Best validation step: 0.
- Train vector: nRMSE 1.000934, cosine -0.000923.
- Holdout vector: nRMSE 1.001770, cosine 0.000628.
- Low-count rollout only: 64 particles, 256 target samples.
- Peak memory stayed bounded in the memory-safe WGPU path.

Interpretation: the learned mapping collapses near zero adapter output under
validation selection. Short low-particle rollout ratios look close because this
pilot rollout is too weak to prove adapter-vector quality.

### 1k DINO Patch Stats

Artifact:
`artifacts/hyper2d_adapter_bank_omnisvg_1k_dino_patch_stats_h1024_valselect`

- Train/holdout split: 900 / 100.
- Condition feature: DINO ViT-S CLS plus patch mean/std/min/max, 1920 values.
- Objective: `supervised-static-lora-vector-mse`, not rectified flow.
- Status: `conditioning_not_rectified_flow`, `quality_ready=false`,
  `hypernet_generalization_validated=false`.
- Best validation step: 0.
- Train vector: nRMSE 1.000798, cosine -0.000151.
- Holdout vector: nRMSE 1.001463, cosine 0.000792.
- Training overfits train loss while validation loss worsens from 0.00008129
  to 0.00015911 before best-step restoration.
- Low-count rollout only: 64 particles, 256 target samples.
- WGPU training throughput: median 10.9M adapter values/sec, peak RSS about
  0.74 GiB for the completed run.
- DINO WGPU extraction needed batch size 1 on this host; batch size 4 repeatedly
  hit a 180 MB cubecl-wgpu allocation failure.

Interpretation: richer global DINO patch statistics do not fix the adapter
prediction failure.

### 1k Patch-Stats kNN Probe

Read-only diagnostic over the same 900/100 split and target adapter vectors:

| k | Holdout nRMSE | Mean Cosine |
| --- | ---: | ---: |
| 1 | 1.71593 | 0.01152 |
| 4 | 1.21684 | 0.01782 |
| 16 | 1.05966 | 0.03023 |
| 64 | 1.01732 | 0.04222 |
| 128 | 1.00958 | 0.04983 |
| 256 | 1.00578 | 0.05643 |

Interpretation: nonparametric lookup over DINO patch-stat features also fails
to recover target adapters. This argues against simply scaling the same static
condition-to-vector target.

## Current Bottleneck

The bottleneck was the training target and generator architecture:

- The previous scalable Burn/WGPU path was only a static adapter-vector MSE
  baseline.
- It did not implement a conditioned rectified-flow LoRA generator.
- The target adapters are low-rank factorizations with residual
non-identifiability even after canonicalization, so vector MSE is a weak proxy
for image/dynamics quality.
- The current low-count rollout validation is useful as a smoke check but is
not paper-quality evidence.

## Implemented Course Correction

- Added `condition.encoder = "dino-vits-token-grid"` so DINO conditioning can
  preserve spatial patch-token structure instead of reducing everything to a
  single global mean. At `dino_image_size = 518`, `token_grid_width = 37` and
  `token_grid_height = 37` stores CLS plus every ViT-S/14 patch token.
- Added a Burn/WGPU adapter-flow trainer. The model input is condition features,
  timestep, and noisy/interpolated adapter state; the target is rectified-flow
  velocity from noise to the stored adapter vector.
- Added TOML recipes for a small token-grid flow smoke, a 1k 8x8 token-grid flow
  run, and a tiny full-token 37x37 structural smoke.

## Current Smoke Evidence

These are implementation smokes, not broad quality claims.

### 20-sample 8x8 token-grid flow smoke

Artifact:
`artifacts/hyper2d_adapter_bank_omnisvg_20_dino_token_grid_flow_smoke`

- Condition encoder: `dino-vits-token-grid-8x8-v1`.
- Feature dimensions: 24,960 = CLS plus 8x8 pooled DINO ViT-S tokens.
- Objective: `rectified-flow-lora-vector`.
- Backend: `burn_wgpu_rectified_flow_lora_vector`.
- Train/holdout split: 16 / 4.
- Final holdout flow-velocity MSE: 0.0029103998.
- Best step: 2.
- Peak RSS in report: 727,044,096 bytes.

### 3-sample 37x37 full-token flow smoke

Artifact:
`artifacts/hyper2d_adapter_bank_omnisvg_3_dino_full_tokens_flow_smoke`

- Condition encoder: `dino-vits-token-grid-37x37-v1`.
- Feature dimensions: 526,080 = CLS plus all 37x37 DINO ViT-S/14 patch tokens.
- Objective: `rectified-flow-lora-vector`.
- Backend: `burn_wgpu_rectified_flow_lora_vector`.
- Train/holdout split: 2 / 1.
- Final holdout flow-velocity MSE: 0.0018538993.
- Best step: 0.
- Peak RSS in report: 608,251,904 bytes.

The first full-token attempt with `flow_hidden = 64` hit a cubecl-wgpu buffer
allocation failure around 130 MiB for the flat first-layer matrix. The smoke was
changed to `flow_hidden = 16`, and the WGPU trainer now preflights oversized
matrix allocations with a targeted error before calling into cubecl-wgpu.

## 1k Token-Grid Flow Evidence

Artifact:
`artifacts/hyper2d_adapter_bank_omnisvg_1k_dino_token_grid_flow_h512`

- Condition encoder: `dino-vits-token-grid-8x8-v1`.
- Train/holdout split: 900 / 100.
- Source-noise scale: 0.1653492, inherited from max-range output scale.
- Best holdout flow-velocity MSE: 0.009204323.
- Holdout vector nRMSE: 10.650944.
- Holdout vector cosine: 0.0030307965.
- 2048-particle holdout rollout ratio to direct stored LoRA: 42.465538.

Interpretation: this run was dominated by oversized random source noise. It was
not a meaningful HyperNPA quality result.

Artifact:
`artifacts/hyper2d_adapter_bank_omnisvg_1k_dino_token_grid_flow_h512_rms_noise`

- Condition encoder: `dino-vits-token-grid-8x8-v1`.
- Train/holdout split: 900 / 100.
- Source-noise scale: 0.012375862, the adapter target RMS.
- Best holdout flow-velocity MSE: 0.00013371115.
- Holdout vector nRMSE: 1.2820431.
- Holdout vector cosine: 0.022384403.
- Peak RSS: 1,880,059,904 bytes.
- WGPU throughput during training: roughly 4.6M to 7.1M adapter values/sec.
- 2048-particle train rollout summary over 8 samples:
  - zero adapter mean loss: 4.971189
  - direct stored LoRA mean loss: 11.55598
  - HyperNPA mean loss: 5.521879
  - HyperNPA/static ratio: 0.5332336
  - HyperNPA/zero ratio: 1.1079336
- 2048-particle holdout rollout summary over 8 samples:
  - zero adapter mean loss: 3.9184957
  - direct stored LoRA mean loss: 16.994625
  - HyperNPA mean loss: 4.817649
  - HyperNPA/static ratio: 0.3688099
  - HyperNPA/zero ratio: 1.2251998

Interpretation: RMS source noise fixes the pathological random-adapter output,
but this is still not broad generalized HyperNPA quality. The generated adapter
does not match sample LoRAs by vector metrics, and it is worse than zero adapter
on average. The direct stored LoRA targets are also worse than zero adapter on
the sampled 2048-particle rollout validation, so static adapter-vector
distillation is currently a malformed quality target.

## Remaining Validation

Do not claim broad generalized HyperNPA quality until these are complete:

1. Fix the direct shared-base plus per-sample LoRA quality target so stored
   LoRAs beat the zero-adapter baseline at quality scale. `report-hyper2d` now
   blocks direct-basis readiness unless shared-vs-zero max ratio is `<= 1.00x`
   on both train and holdout oracle splits.
2. Add image/dynamics-loss fine-tuning or guidance for the conditioned flow so
   selection is based on rollout quality, not only factorized adapter-vector MSE.
   `report-hyper2d` now also blocks conditioning readiness unless generated
   adapters beat the zero-adapter rollout baseline as well as matching stored
   direct LoRAs.
3. Scale to 10k only if the 1k quality-scale run closes the vector and rollout
   gaps.
4. Validate generated adapters against direct stored LoRAs and 2D overfit
   oracles at quality scale: at least 2048 rollout particles and 2048 target
   samples.
5. Replace JSON full-token caches with a compact binary/sharded cache before
   broad 37x37 full-token training; full-token JSON is acceptable for the tiny
   smoke only.
