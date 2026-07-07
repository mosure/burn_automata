# HyperNPA: Current Paper Status

This document mirrors the current `docs/hyper_npa.tex` revision.

## Current Paradigm

The current HyperNPA path is end-to-end and image-conditioned:

```text
image -> DINO ViT-S full tokens -> token-aware rectified-flow LoRA generator
      -> shared trainable NPA base -> rollout image loss
```

This replaces the older primary emphasis on per-sample oracle LoRA training plus
raw adapter-vector regression. Oracle adapters are now treated as baselines and
controls, not the main training target.

## Main Artifact

The current default E2E artifact is:

- `artifacts/hyper2d_e2e_rollout_train_omnisvg_10k_steps3000_b16_p128s4_rank16_cosine_cuda/shared_base.bpk`
- `artifacts/hyper2d_e2e_rollout_train_omnisvg_10k_steps3000_b16_p128s4_rank16_cosine_cuda/hyper_2d.json`
- `models/dino/dino_vits.mpk`

Training config:

- 10k OmniSVG selected examples.
- 9500 train / 500 holdout.
- Online DINO ViT-S full tokens: CLS + 37x37 patch grid, 1370 tokens x 384 dims.
- Trainable shared growing-2D NPA base from step zero.
- Generated LoRA rank 16, alpha 16.
- Token-aware rectified-flow generator, hidden 128, two sample steps.
- Burn/CUDA autodiff.
- 128 rollout particles, 4 rollout steps, TBPTT chunk size 2.
- Target-2D rollout image loss.

## Latest E2E Results

The selected checkpoint is step 2700, chosen by best holdout PSNR.

| checkpoint | step | train loss | holdout loss | holdout PSNR |
| --- | ---: | ---: | ---: | ---: |
| first report | 100 | 7.642 | 5.398 | 4.83 dB |
| best holdout PSNR | 2700 | 7.305 | 4.592 | 8.56 dB |
| best holdout loss | 2900 | 6.364 | 4.578 | 8.35 dB |
| final | 3000 | 4.896 | 4.583 | 8.42 dB |

The 32-sample holdout validation at training scale is:

| metric | value |
| --- | ---: |
| particles / steps | 128 / 4 |
| mean target-image PSNR | 8.56 dB |
| min target-image PSNR | 0.56 dB |
| max target-image PSNR | 16.75 dB |
| mean total validation loss | 4.592 |
| passes 8 dB threshold | false |

This is not oracle parity. It is a functional E2E training run with low current
quality.

## New Bevy Renderer Figure

New paper figure:

- `docs/hyper_npa_figures/e2e_bevy_rollouts/e2e_hypernpa_bevy_rollout_panel.png`

Supporting generated artifacts:

- `docs/hyper_npa_figures/e2e_bevy_rollouts/e2e_bevy_rollout_summary.json`
- per-sample PNGs and reports under `docs/hyper_npa_figures/e2e_bevy_rollouts/<slug>/`

The figure uses the current E2E HyperNPA inference path through `bevy_automata`
headless export:

```sh
cargo run -p bevy_automata -- export \
  --hyper-image <condition.png> \
  --output-dir docs/hyper_npa_figures/e2e_bevy_rollouts/<slug> \
  --output-prefix <slug> \
  --steps 128 \
  --capture-steps 4,16,64,128 \
  --width 192 \
  --height 192 \
  --particles 2048
```

Rows are the top eight held-out validation samples from the E2E report. The
panel shows actual `bevy_gaussian_splatting` output at rollout steps 4, 16, 64,
and 128. It also shows the current weakness clearly: many rollouts decay or go
near-empty by long horizons.

## Oracle Comparison

Available high-particle oracle comparisons remain separate from the current E2E
10k holdout slice.

| system | examples | particles | steps | mean PSNR | min PSNR | status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Direct exact LoRA vs oracle NPA | 16 | 2048 | 32 | 99.00 | 99.00 | materialization sanity pass |
| DINO-flow zero-source control vs oracle NPA | 16 | 2048 | 32 | 30.44 | 27.31 | passes 26 dB gate |
| Trained WGPU DINO-flow adapter model vs oracle NPA | 16 | 2048 | 32 | 9.10 | 0.53 | fails oracle gate |
| Current E2E HyperNPA vs target image | 32 | 128 | 4 | 8.56 | 0.56 | not oracle-comparable |

The direct/zero-source rows prove the NPA + LoRA representation and sampling
path can match oracle-trained per-sample NPAs. The current E2E model has not yet
matched that quality.

## Claim Boundary

What is proven:

- Feed-forward image -> DINO -> generated LoRA -> NPA rollout works in the
  actual Bevy viewer/export path.
- The 10k E2E training objective learns a measurable target-image signal.
- The repo has high-particle oracle controls showing the representational upper
  bound.

What is not proven:

- Broad 1k/10k generalized oracle-quality HyperNPA.
- Matched 2048-particle oracle PSNR for the current E2E holdout slice.
- Long-rollout stability.
- Parity with per-sample overfit NPA oracles.

## Next Work

1. Generate matched oracle NPAs for the E2E validation slice.
2. Report current E2E HyperNPA vs oracle PSNR on the same images shown in the
   Bevy panel.
3. Train with longer rollout horizons and stability penalties.
4. Add a curriculum up to 2048-particle validation without dense all-pairs memory
   blowups.
5. Compare trainable shared-base E2E training against frozen-base and direct
   oracle variants on identical samples.
