# HyperNPA: Latest Paper Status

This document mirrors the current `docs/hyper_npa.tex` paper revision. The
paper has been reframed around the latest high-particle oracle comparison rather
than the older low-particle 10k pilot.

## Claim Boundary

We have **not** proven broad generalized image -> DINO -> rectified-flow LoRA ->
NPA quality.

What is now proven:

- Direct exact LoRA adapters can materialize persisted per-sample oracle NPAs.
- DINO ViT-S 8x8 token-grid conditions can key an exact 16-sample
  condition-to-adapter control.
- The rectified-flow sampling path can emit nearly exact adapter vectors when
  constructed with the deterministic zero-source control.
- The 2048-particle PSNR gate and DINO feature-cache validation path are
  functioning.

What is still failing:

- The WGPU-trained DINO rectified-flow adapter generator does not learn the exact
  adapter bank yet.
- The passing control is train-only over 16 samples, not holdout or 1k/10k
  generalized quality.

## Latest Results

All metrics below use persisted oracle NPA models as the baseline, 2048 rollout
particles, 32 rollout steps, update probability 0.5, 128x128 Gaussian raster
metrics, and seed 42.

| system | examples | vector NRMSE | cosine | mean render PSNR | min render PSNR | status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Direct exact LoRA vs oracle | 16 | 0.0 | 1.000 | 99.00 dB | 99.00 dB | passes materialization sanity gate |
| DINO-flow zero-source control vs oracle | 16 | 1.18e-7 | 1.000 | 30.44 dB | 27.31 dB | passes narrow overfit gate |
| WGPU-trained DINO-flow vs oracle | 16 | 1.112 | 0.151 | 9.10 dB | 0.53 dB | fails |

The zero-source control is intentionally deterministic: `flow_source_scale = 0.0`.
It proves the adapter representation and flow sampling path can express the
targets. It does not prove that WGPU training can learn them.

## Per-Sample HyperNPA Control PSNR

Direct exact LoRA is 99 dB for every row. HyperNPA values below are the
DINO-flow zero-source control sorted from worst to best.

| sample | render PSNR | sample | render PSNR |
| --- | ---: | --- | ---: |
| `00009725` | 27.31 | `00006072` | 29.91 |
| `00001643` | 27.91 | `00005000` | 30.28 |
| `00005840` | 28.29 | `00007095` | 30.48 |
| `00005598` | 28.71 | `00005512` | 31.37 |
| `00000371` | 28.83 | `00009530` | 31.99 |
| `00002120` | 28.89 | `00003130` | 33.22 |
| `00006450` | 29.24 | `00005081` | 33.71 |
| `00006500` | 29.47 | `00009750` | 37.44 |

## Figures

New high-particle paper figures:

- `docs/hyper_npa_figures/exact_dino_flow_psnr/exact_dino_flow_16sample_panel_a.png`
- `docs/hyper_npa_figures/exact_dino_flow_psnr/exact_dino_flow_16sample_panel_b.png`
- `docs/hyper_npa_figures/exact_dino_flow_psnr/exact_dino_flow_16sample_summary.json`

Each row shows:

1. condition thumbnail;
2. DINO-flow zero-source HyperNPA adapter rollout;
3. direct exact LoRA rollout;
4. persisted oracle NPA rollout.

All rollouts in those panels use 2048 particles and 32 steps. They are dense
particle rollout rasterizations, not low-count sparse thumbnails.

The separate WGPU viewer renderer reference remains:

- `docs/hyper_npa_figures/lizard_wgpu_gaussian_4096_rollout.png`

That lizard figure validates the WGPU/Bevy Gaussian renderer path for an
imported catalog model. It is not a generated HyperNPA result.

## Reproduction

Regenerate the underlying validation artifacts with the Rust CLIs:

```sh
cargo run -p burn_automata --features "cli dino backend_wgpu" --bin burn_automata -- \
  validate-hyper2d-psnr-gate --config configs/hyper2d_adapter_bank/psnr_gate_exact_oracle_10k256x64_2048_rank132_dino_flow_sampled.toml
cargo run -p burn_automata --features "cli dino backend_wgpu" --bin burn_automata -- \
  report-hyper2d --input artifacts/hyper2d_adapter_bank_exact_oracle_10k256x64_dino_token_grid_flow_zero_source_h384_sampled/report.json
```

The validation consumes:

- `artifacts/hyper2d_adapter_bank_exact_oracle_10k8x8_dino_token_grid_flow_linear_solve_overfit_train_all/psnr_gate_report.json`
- `artifacts/hyper2d_adapter_bank_exact_oracle_10k8x8_dino_token_grid_flow_overfit_train_all/psnr_gate_report.json`

Rollout panels should be generated from renderer output artifacts referenced by
the PSNR-gate report; Python paper renderers are no longer part of the canonical
pipeline.

## Interpretation

The main bottleneck is no longer validation plumbing. The current bottleneck is
the learned WGPU rectified-flow objective/optimizer/architecture. The next
meaningful milestone is to make WGPU training match the zero-source control on
the same 16-sample exact bank, then add holdout samples, then scale to 128 and
1k examples while retaining the 2048-particle oracle gate.
