# Hyper2D Direct-Basis 10k Validation Report

## Status

The evaluated artifact is **not yet an image-conditioned hypernetwork**. It is a shared NPA base with directly optimized, stored per-sample LoRA adapters. The intended conditional system is `image encoder features -> rectified-flow LoRA generator -> shared-base NPA rollout`; that final image-to-LoRA generator remains unvalidated in this artifact set.

## Main 10k Run

- Artifact: `artifacts/hyper2d_omnisvg_10k_burn_wgpu_adamw_batch64_600_300_300_eval1000_oracle2x2`
- Train / holdout examples: 9000 / 1000
- Adapter: rank 16, alpha 16.0
- Training: 600 shared+adapter steps, 300 train refine steps, 300 holdout adapter steps
- Rollout objective: 64 particles, 8 training steps
- Backend: burn_wgpu_autodiff_dense_direct_basis

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

The 10k result supports the claim that a shared NPA basis plus per-sample LoRA adapters is a viable intermediate representation: train and holdout losses improve similarly, and the sampled oracle gap is moderate rather than catastrophic. It does **not** yet establish that image features can predict the LoRA weights. The next experiment must train and evaluate the conditional LoRA generator against this adapter bank and compare generated adapters to direct adapters and oracle overfits on the same rollout metrics.
