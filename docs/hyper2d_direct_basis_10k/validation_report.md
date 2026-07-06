# Hyper2D Direct-Basis 10k Validation Report

## Status

The evaluated artifact is **not yet an image-conditioned hypernetwork**. It is a shared NPA base with directly optimized, stored per-sample LoRA adapters. The intended conditional system is `image encoder features -> rectified-flow LoRA generator -> shared-base NPA rollout`; that final image-to-LoRA generator remains unvalidated in this artifact set.

The evaluated 10k artifact is also **not quality-ready** for paper claims. It was trained and compared at low rollout/sample scale: 64 rollout particles, 8 rollout steps, and as few as 190 target samples in the stored adapter bank. Current quality gates require at least 2048 rollout particles, at least 2048 target samples, and at least 8 oracle examples per split.

## Main 10k Run

- Artifact: `artifacts/hyper2d_omnisvg_10k_burn_wgpu_adamw_batch64_600_300_300_eval1000_oracle2x2`
- Train / holdout examples: 9000 / 1000
- Adapter: rank 16, alpha 16.0
- Training: 600 shared+adapter steps, 300 train refine steps, 300 holdout adapter steps
- Rollout objective: 64 particles, 8 training steps
- Minimum stored target samples: 190
- Backend: burn_wgpu_autodiff_dense_direct_basis

## Quality Gate Audit

| gate | required | observed | status |
| --- | ---: | ---: | --- |
| rollout particles | >= 2048 | 64 | fail |
| target samples | >= 2048 | 190 min | fail |
| oracle train examples | >= 8 | 8 | pass |
| oracle holdout examples | >= 8 | 8 | pass |
| direct-basis oracle max ratio | <= 1.20x | 1.141 train / 1.126 holdout | pass |

Overall status: `quality_particle_count_too_low`, `quality_ready=false`.

## Loss Summary

| split | initial mean loss | final mean loss | reduction |
| --- | ---: | ---: | ---: |
| train sample eval | 12.112 | 6.735 | 44.4% |
| holdout sample eval | 11.656 | 6.473 | 44.5% |

## Oracle Comparison

| split | examples | shared loss | oracle overfit loss | mean ratio | max ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 8 | 7.830 | 7.188 | 1.079 | 1.141 |
| holdout | 8 | 7.243 | 6.676 | 1.075 | 1.126 |

## Rollout Figures

See `figures/rollout_grid.png` for target thumbnails and long-rollout panels from selected stored LoRA adapters. Selected rows include oracle-overfit rollout panels where the validation report persisted oracle checkpoints. These renders evaluate stability of the shared-base+LoRA baseline, not a learned image-conditioned hypernet.

## Interpretation

The 10k result is useful as a low-count pilot: train and holdout losses improve similarly, and the sampled low-count oracle gap is moderate rather than catastrophic. It does **not** establish paper-quality shared-basis parity because the rollout and target-sampling scale are far below the current 2048-particle / 2048-target-sample gate. It also does **not** establish that image features can predict the LoRA weights. The next experiment must rerun direct-basis training and oracle validation at quality scale, then train and evaluate the conditional LoRA generator against that adapter bank.
