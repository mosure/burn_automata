# Burn-First Cleanup Status

This repository now treats the Rust/Burn CLI as the primary implementation for
2D direct-basis and Hyper2D adapter-bank experiments. Python remains only where
it provides reference validation or external checkpoint/model interchange.

## Primary Paths

| Workflow | Primary command | Status |
| --- | --- | --- |
| Single-sample 2D target training | `train-target2d` | Rust CPU entrypoint; the former upstream Python/CUDA trainer is removed. |
| Shared 2D base plus per-sample LoRA bank | `train-hyper2d-direct-basis` | Defaults to `burn-wgpu` in TOML recipes. |
| Image condition to LoRA generator | `train-hyper2d-adapter-bank` | Burn/WGPU training path; CPU is for smoke correctness only. |
| Hyper2D validation summary | `report-hyper2d` | Rust JSON, Markdown, and LaTeX summary path with optional quality-gate failure. |
| Single-target 3D oracle overfit | `train-render3d --config` | Burn-native TOML recipes under `configs/render3d/`. |
| Shared 3D base plus per-target adapters | `train-render3d-adapters --config` | Burn-native TOML recipes under `configs/render3d_adapters/`. |

## Retained Python Inventory

| Path | Classification | Boundary |
| --- | --- | --- |
| `scripts/import_selforg_catalog.py`, `scripts/export_npa_checkpoint.py` | Import/export utilities | Allowed for external SelfOrg/PyTorch checkpoint interchange. |
| `scripts/setup_dino_vits.py` | Model export utility | Allowed while Burn DINO model-pack generation still depends on a Torch checkpoint. |
| `scripts/validate_import_parity.py`, `scripts/validate_catalog_parity.py`, `scripts/validate_3d_catalog.py`, `scripts/compare_3d_candidate.py`, `scripts/catalog3d_validation/` | Reference/parity validation | Allowed for imported-catalog and renderer parity checks. |

## Removed Python Surfaces

- Python/Torch training backends for target2d and Hyper2D direct-basis training.
- Python report/paper renderers; use Rust `report-hyper2d` and checked-in report
  artifacts instead.
- Python benchmark matrix wrappers; use Rust `bench`, `bench-spatial`, and
  `bench-training` commands or TOML/Rust experiment bundles.

## Report Gates

`report-hyper2d` writes `validation_summary.json`, `validation_report.md`, and
`validation_report.tex` by default. Add `--require-quality-ready` when a local
or CI run should fail if the summarized artifact is below threshold.

| Report kind | Ready status | Gate |
| --- | --- | --- |
| Direct-basis shared base plus stored LoRA | `direct_basis_oracle_ready` | Train and holdout oracle max ratio must be `<= 1.20x`. |
| Direct-basis shared base plus stored LoRA | `direct_basis_oracle_ready` | Train and holdout shared-vs-zero max ratio must be `<= 1.00x`. |
| Condition image to LoRA generator | `conditioning_quality_ready` | Train and holdout adapter-vector normalized RMSE must be `<= 0.35`. |
| Condition image to LoRA generator | `conditioning_quality_ready` | Train and holdout adapter-vector mean cosine must be `>= 0.80`. |
| Condition image to LoRA generator | `conditioning_quality_ready` | Train and holdout rollout max ratio to static LoRA must be `<= 1.15x`. |
| Condition image to LoRA generator | `conditioning_quality_ready` | Train and holdout rollout max ratio to zero adapter must be `<= 1.00x`. |
| Condition image to LoRA generator | `conditioning_quality_ready` | Oracle render RGB PSNR from `validate-hyper2d-psnr-gate` must be `>= 26.0 dB`. |

Reports also summarize available throughput history: direct-basis reports expose
particle-steps/sec when present, and adapter-bank reports expose
adapter-values/sec when present.

The direct-basis oracle validator and PSNR gate both accept TOML configs, so
quality runs can be bundled and reproduced without long CLI argument lists:
`configs/hyper2d_direct_basis/oracle_validate_10k_quality_2048.toml` and
`configs/hyper2d_adapter_bank/psnr_gate_1k_dino_token_grid_flow_h512_rms_noise.toml`
are the current reference validation recipes.

## Current Experimental Reading

The direct-basis 10k artifact is a shared-base plus stored-adapter result, not a
conditioned hypernet. The adapter-bank command trains the conditioned HyperNPA
stage from that stored bank. Current summary-token pilot reports show that the
pipeline runs on Burn/WGPU, but generated adapter vectors underfit the stored
LoRAs. That makes DINO features, a stronger adapter decoder, and residual/flow
adapter generation the next 2D HyperNPA priorities before scaling claims.

The direct-basis command now accepts only Burn backends. Removed aliases include
`upstream-python`, `python`, `torch-cuda`, and `legacy-upstream-python`; old
experiment TOMLs using them should be migrated to `burn-wgpu` or `burn-cuda`.

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
