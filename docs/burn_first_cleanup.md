# Burn-First Cleanup Status

The 2D codebase is being reset around a small number of maintained paths. The
official SelfOrg-NPA release remains the oracle reference until local Burn
training passes strict parity.

## Canonical 2D Paths

| Workflow | Command/config | Status |
| --- | --- | --- |
| Official 2D lizard reference | `scripts/fetch_selforg_npa.sh`, `scripts/export_selforg_npa_fixture.py`, `validate-npa2d-parity` | Canonical parity gate; upstream code is cached externally under `.cache/`. |
| Imported 2D inference | `.bpk` catalog models plus CPU/WGPU rollout commands | Maintained inference hot path. |
| Online HyperNPA training | `train-hyper2d-e2e-rollout --config configs/verified/2d/hyper_e2e/*.toml` | Maintained Burn/WGPU/CUDA HyperNPA path. |
| Single-sample Burn target training | `train-target2d --experimental` | Experimental diagnostic only; not an accepted oracle baseline. |

## Config Policy

Tracked, supported configs live in `configs/verified/`. Current verified 2D
configs are:

- `configs/verified/2d/parity/lizard_smoke.toml`
- `configs/verified/2d/parity/lizard_full.toml`
- `configs/verified/2d/hyper_e2e/smoke_lizard_dino_online.toml`
- `configs/verified/2d/hyper_e2e/bench_omnisvg_8_b4_p128.toml`
- `configs/verified/2d/hyper_e2e/bench_omnisvg_8_b4_p128_tiled.toml`

Exploratory configs, failed sweeps, and local one-offs belong in
`configs/sandbox/`, which is gitignored. Direct-basis LoRA banks, static
adapter reconstruction, and dense Burn target-image trainers stay in sandbox or
artifacts until they pass the official parity gate.

No production-quality 1k/10k HyperNPA configuration is currently verified.
Promoting one requires a high-particle held-out quality report and an explicit
oracle-relative parity result, not only a successful scale run.

## Retained Python Inventory

Python is restricted to external reference/export work:

- `scripts/fetch_selforg_npa.sh` fetches the official upstream repo into
  `.cache/selforg_npa/NPA`.
- `scripts/export_selforg_npa_fixture.py` exports official target/checkpoint
  metadata for Rust parity checks.
- `scripts/import_selforg_catalog.py` and `scripts/export_npa_checkpoint.py`
  handle external SelfOrg/PyTorch checkpoint interchange.
- `scripts/setup_dino_vits.py` creates the Burn DINO model pack from a Torch
  checkpoint.
- `scripts/validate_*`, `scripts/compare_3d_candidate.py`, and
  `scripts/catalog3d_validation/` remain reference/parity utilities.

Do not add Python training, benchmark-matrix, or paper-rendering entrypoints.
New experiments should be Rust/Burn CLI commands with TOML configs.

## Parity Requirements Before Promotion

Burn 2D oracle training is not promoted until a strict harness shows agreement
with upstream for:

- target extraction and adaptive target point count,
- seeded model initialization and imported pretrained weights,
- one-step SPH perception/update semantics,
- upstream splat loss without extra background, Chamfer, or altered weights,
- gradient and optimizer update behavior,
- bounded 4096-particle rollout quality against the official pretrained lizard.

Until those gates pass, any model produced by `train-target2d --experimental` is
a diagnostic artifact, not a baseline for HyperNPA papers or quality claims.

## 3D Boundary

This cleanup pass intentionally does not reorganize 3D training. Existing
`configs/render3d/` and `configs/render3d_adapters/` remain in place for the
separate 3D cleanup/parity effort.
