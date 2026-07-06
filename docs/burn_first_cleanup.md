# Burn-First Cleanup Status

This repository now treats the Rust/Burn CLI as the primary implementation for
2D direct-basis and Hyper2D adapter-bank experiments. Python remains only where
it still provides parity checks, historical imports, or rich paper rendering.

## Primary Paths

| Workflow | Primary command | Status |
| --- | --- | --- |
| Single-sample 2D target training | `train-target2d` | Burn CLI entrypoint with upstream parity checks retained. |
| Shared 2D base plus per-sample LoRA bank | `train-hyper2d-direct-basis` | Defaults to `burn-wgpu` in TOML recipes. |
| Image condition to LoRA generator | `train-hyper2d-adapter-bank` | Burn/WGPU training path; CPU is for smoke correctness only. |
| Hyper2D validation summary | `report-hyper2d` | Rust JSON, Markdown, and LaTeX summary path with optional quality-gate failure. |
| Single-target 3D oracle overfit | `train-render3d --config` | Burn-native TOML recipes under `configs/render3d/`. |
| Shared 3D base plus per-target adapters | `train-render3d-adapters --config` | Burn-native TOML recipes under `configs/render3d_adapters/`. |

## Legacy Python Inventory

| Path | Classification | Retirement condition |
| --- | --- | --- |
| `scripts/train_target2d_upstream_gpu.py` | Legacy parity trainer | Remove after Burn 2D parity gates cover target loss, rollout quality, and throughput on catalog samples. |
| `scripts/train_hyper2d_direct_basis_gpu.py` | Legacy parity trainer | Remove after `train-hyper2d-direct-basis --gpu-backend burn-wgpu` is the only needed backend for 1k/10k experiments. |
| `scripts/render_hyper2d_direct_basis_paper.py` | Legacy paper renderer | Keep only for rollout-grid figures and PDF assembly until Rust reporting renders those assets. |
| `scripts/render_kernel_ablation_paper.py` | Paper renderer | Keep until kernel ablation paper generation moves into Rust or `xtask`. |
| `scripts/import_selforg_catalog.py`, `scripts/export_npa_checkpoint.py` | Import/export utilities | Keep while external checkpoint/catalog interchange is supported. |
| `scripts/validate_import_parity.py`, `scripts/validate_catalog_parity.py`, `scripts/validate_3d_catalog.py` | Parity validation | Keep while imported upstream catalogs remain a compatibility target. |
| `scripts/bench_*.py` | Benchmark helpers | Migrate once equivalent Rust `bench-*` commands cover the same matrix outputs. |

## Required Gates Before Removing Upstream Python Training

1. `train-target2d` Burn training reproduces single-sample 2D oracle quality for
   representative catalog targets and OmniSVG thumbnails.
2. `train-hyper2d-direct-basis` Burn/WGPU produces a shared base and persistent
   per-sample LoRA bank whose oracle-validation ratios are within the documented
   `report-hyper2d` gate.
3. `train-hyper2d-adapter-bank` Burn/WGPU reports generated LoRA vector metrics
   and rollout-vs-static-adapter metrics that pass `report-hyper2d`.
4. CI or local validation includes Rust tests for config parsing, report
   interpretation, adapter persistence, and GPU backend selection.

## Report Gates

`report-hyper2d` writes `validation_summary.json`, `validation_report.md`, and
`validation_report.tex` by default. Add `--require-quality-ready` when a local
or CI run should fail if the summarized artifact is below threshold.

| Report kind | Ready status | Gate |
| --- | --- | --- |
| Direct-basis shared base plus stored LoRA | `direct_basis_oracle_ready` | Train and holdout oracle max ratio must be `<= 1.20x`. |
| Condition image to LoRA generator | `conditioning_quality_ready` | Train and holdout adapter-vector normalized RMSE must be `<= 0.35`. |
| Condition image to LoRA generator | `conditioning_quality_ready` | Train and holdout adapter-vector mean cosine must be `>= 0.80`. |
| Condition image to LoRA generator | `conditioning_quality_ready` | Train and holdout rollout max ratio to static LoRA must be `<= 1.15x`. |

Reports also summarize available throughput history: direct-basis reports expose
particle-steps/sec when present, and adapter-bank reports expose
adapter-values/sec when present.

## Current Experimental Reading

The direct-basis 10k artifact is a shared-base plus stored-adapter result, not a
conditioned hypernet. The adapter-bank command trains the conditioned HyperNPA
stage from that stored bank. Current summary-token pilot reports show that the
pipeline runs on Burn/WGPU, but generated adapter vectors underfit the stored
LoRAs. That makes DINO features, a stronger adapter decoder, and residual/flow
adapter generation the next 2D HyperNPA priorities before scaling claims.

The direct-basis command still accepts `upstream-python`, `python`, and
`torch-cuda` as compatibility aliases, but the canonical backend name is now
`legacy-upstream-python`. New experiment TOMLs should use `burn-wgpu`.

## 3D Cleanup Boundary

The 3D Burn-native oracle path has TOML experiment loading through
`train-render3d --config configs/render3d/torus_oracle_smoke.toml`. The
`configs/render3d/torus_oracle_quality.toml` recipe is the first parity-oriented
3D overfit target.

The 3D adapter-suite path now has TOML experiment loading through
`train-render3d-adapters --config configs/render3d_adapters/torus_smoke.toml`.
The smoke recipe validates report and adapter-bank plumbing; the next quality
step is running `configs/render3d_adapters/many_slice_quality.toml` after a
3D oracle baseline is accepted, then training a conditioned 3D adapter generator
against that bank.
