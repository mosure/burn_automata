# HyperNPA 2D E2E Rollout Experiments

These configs describe the Burn-first HyperNPA path that trains from image
condition to online DINO tokens to a learned token-attention rectified-flow LoRA
generator against the `target2d` rollout/image loss.

This directory is intentionally separate from `hyper2d_adapter_bank`: adapter
bank and oracle LoRA configs are diagnostics or warm-start sources, not the
primary generalized HyperNPA objective.

## Smoke Configs

`smoke_lizard_steps1_dino_online_joint.toml` is a single-image execution smoke.
It runs online DINO, one generated-adapter rollout training step, and bounded
32-particle quality validation.

`smoke_omnisvg_4_steps2_dino_online_joint.toml` is a cached OmniSVG
train/holdout smoke. It uses three train examples and one holdout example with
batch size two, so it exercises the generated LoRA batch path and reports
holdout render PSNR.

`smoke_omnisvg_8_dino_online_joint.toml` is a preflight-only config for
validating source resolution, split accounting, and full-token DINO dimensions.

`bench_omnisvg_8_steps80_b{1,2,4}_p64s4.toml` are throughput benchmark configs
over the same cached 8-image OmniSVG slice. They vary only the training
`example_batch_size` so GPU batching effects are directly comparable. Each run
also performs bounded generated-adapter inference validation and reports
`particle_steps_per_sec` in `training_result.quality_validation`.

`bench_omnisvg_8_steps200_b4_p128s4.toml` is the heavier utilization probe. It
keeps the same 8-image slice and batch size four, doubles the particle count,
and runs long enough to sample GPU power/utilization externally.

`preflight_omnisvg_1k_dino_online_joint.toml` validates 1k OmniSVG source
resolution, train/holdout split accounting, full-token DINO dimensionality, and
memory estimates without running DINO or training.

`stability_omnisvg_1k_steps200_b8_p64s4_cuda.toml` is the first online
1k-scale learning probe. It uses full-token DINO in batches of eight, CUDA Burn
training, 64 particles, four rollout steps, and a 950/50 train/holdout split so
loss trends can be checked before moving to the heavier particle count.

`train_omnisvg_1k_steps300_b8_p128s4_cuda.toml` is the current bounded 1k
online HyperNPA training target. It keeps 2048-particle quality runs out of the
dense backward path and trains the shared NPA trunk plus generated LoRA path at
128 particles with bounded holdout validation.

`train_omnisvg_1k_steps500_b8_p128s4_lr1e4_cuda.toml` repeats the 128-particle
1k target with a lower learning rate and more steps. Use it when the 3e-4 run
shows stochastic late-step loss regression.

`train_omnisvg_1k_steps1000_b8_p128s4_rank16_cosine_cuda.toml` is a
quality-push ablation for the online path. It doubles LoRA rank to 16 and uses
cosine LR decay with best-reported checkpoint restoration, while keeping the
token conditioner hidden width at 128 to avoid making the DINO-token projection
the dominant ablation variable.

`preflight_omnisvg_10k_dino_online_joint.toml` validates the 10k source split
and memory budget. Full-token DINO for 10k examples stores about 19.6 GiB of
f32 condition features. The trainer uses a host-row-streamed condition cache
above the resident-device threshold, so 10k does not keep the full token bank as
one long-lived GPU tensor and does not flatten a duplicate 10k feature bank
during cache construction.

`stability_omnisvg_10k_steps300_b16_p64s4_lr1e4_cuda.toml` is the bounded 10k
stability probe. It uses DINO batches of 16 and generated-adapter training
batches of 16 at 64 particles.

`train_omnisvg_10k_steps300_b16_p128s4_attention_lr1e4_cuda.toml` is the
bounded 10k 128-particle proof for the learned token-attention condition
pooler. It is shorter than the 1000-step target and is meant to confirm that
the more expressive full-token conditioner remains memory-stable and learns the
fixed holdout objective.

`train_omnisvg_10k_steps1000_b16_p128s4_lr1e4_cuda.toml` is the current 10k
online HyperNPA learning run. It keeps dense training at 128 particles, records
initial and final fixed holdout validation, and is intended to prove objective
learning rather than oracle-quality parity.

`train_omnisvg_10k_steps1500_b16_p128s4_rank16_cosine_cuda.toml` is the
stronger 10k quality-push run. It uses the same full-token online DINO
conditioning and 128-particle dense rollout limit as the 1000-step baseline, but
uses rank-16 generated LoRA, cosine LR decay, and best-reported checkpoint
selection to reduce late-step quality regression.

`train_omnisvg_10k_steps3000_b16_p128s4_rank16_cosine_cuda.toml` extends that
quality push. The trainer evaluates the fixed holdout set at report checkpoints
and exports the checkpoint with best mean holdout PSNR, so the final artifacts
are selected against the same quality metric reported in validation rather than
against a noisy sampled training batch.

These smoke configs are not quality claims. Their validation sections are kept
small so CI/development runs do not accidentally launch 2048-particle dense
training or validation. Use explicit quality-scale configs for 2048-particle
PSNR gates once the sparse/tiled rollout path is active.
