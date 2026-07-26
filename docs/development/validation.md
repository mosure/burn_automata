# Validation

This document is the maintained validation contract. Historical run narratives
belong in experiment artifacts, not here.

## Repository Gate

Run the cheap structural checks first:

```bash
scripts/ci/check_repository_layout.sh
scripts/ci/check_inference_features.sh
cargo fmt --all --check
```

The layout gate prevents tracked sandbox experiments, root-level operational
scripts/docs, legacy configuration roots, generated LaTeX/cache files, and
missing publication PDFs.

## Test Workflow

The GitHub `test` workflow executes this matrix:

```bash
cargo check -p burn_automata_kernels -p burn_automata --all-targets
cargo check -p burn_automata --examples --benches
cargo check -p burn_automata --no-default-features --features backend_wgpu
cargo check -p burn_automata --no-default-features --features gpu_wgpu

cargo test -p burn_automata_kernels -p bevy_burn
cargo test -p burn_automata --lib \
  --no-default-features --features "backend_ndarray cli import"
cargo test -p burn_automata --test core \
  --no-default-features --features "backend_ndarray cli import" \
  -- --skip catalog_growth::
cargo test -p burn_automata --test gpu_wgpu \
  --no-default-features --features gpu_wgpu

cargo test -p bevy_automata \
  --no-default-features --features "splatting gpu_wgpu" \
  --test gaussian_gpu_link --no-run
cargo check -p bevy_automata --no-default-features --features viewer
cargo check -p bevy_automata
```

The hosted runner links GPU integration tests but does not execute
adapter-dependent renderer checks.

## Clippy Workflow

All retained feature surfaces are warning-free:

```bash
cargo clippy \
  -p burn_automata_kernels -p burn_automata -p bevy_burn \
  --all-targets -- -D warnings
cargo clippy -p burn_automata --examples --benches -- -D warnings
cargo clippy -p burn_automata \
  --no-default-features --features "backend_wgpu gpu_wgpu" \
  -- -D warnings
cargo clippy -p bevy_automata \
  --no-default-features --features viewer \
  -- -D warnings
cargo clippy -p bevy_automata \
  --no-default-features --features "splatting gpu_wgpu" \
  --test gaussian_gpu_link -- -D warnings
```

## Fixed 2D Reference

Provision and export the pinned upstream fixture:

```bash
scripts/reference/selforg/fetch_selforg_npa.sh
scripts/reference/selforg/fetch_selforg_npa_targets.sh
python3 scripts/reference/selforg/export_selforg_npa_fixture.py \
  --training-step
```

Run the strict bounded and full parity contracts:

```bash
cargo run --release -p burn_automata -- \
  validate-npa2d-parity \
  --config configs/verified/2d/parity/lizard_smoke.toml

cargo run --release -p burn_automata -- \
  validate-npa2d-parity \
  --config configs/verified/2d/parity/lizard_full.toml
```

Independent imported-model parity:

```bash
python3 scripts/reference/selforg/validate_import_parity.py \
  --model models/catalog/growing/lizard.bpk \
  --particles 64 \
  --preset growing-2d \
  --seed-scale 0.2
```

The Python oracle is intentionally dependency-free and bounded to small
particle counts. It is an independent formula/layout check, not a training or
throughput path.

## GPU Inference

Execute device tests and imported-model parity on a GPU host:

```bash
REQUIRE_BPK=1 scripts/validation/validate_gpu_e2e.sh
```

For throughput, use the synchronized Rust benchmark:

```bash
PARTICLES=4096 STEPS=128 PRESET=growing-2d \
  scripts/ci/bench_rollout.sh
```

Any nonzero hash-grid overflow invalidates a performance result.

## HyperNPA

The bounded structured-flow smoke is:

```bash
cargo run --release -p burn_automata \
  --no-default-features \
  --features cli,backend_ndarray,backend_wgpu,dino -- \
  train-hypernpa2d \
  --config \
  configs/verified/2d/hypernpa/flow/smoke_conditional_row_flow.toml
```

Teacher-free, published-control, and throughput configs are classified in
[`../../configs/verified/2d/hypernpa/README.md`](../../configs/verified/2d/hypernpa/README.md).
A quality claim additionally requires held-out p10 PSNR, base-only and shuffled
condition controls, matched particles/seeds/horizons, and long-horizon drift.

## Adaptive NPA

CPU topology conservation:

```bash
cargo run -p burn_automata -- \
  audit-adaptive-topology \
  --config \
  configs/verified/2d/adaptive/audits/continuous_topology_smoke.toml
```

Current bounded continuous-scale evaluation:

```bash
cargo run --release -p burn_automata \
  --features cli,backend_ndarray,backend_wgpu -- \
  eval-adaptive-target2d \
  --config \
  configs/verified/2d/adaptive/evaluation/recurrent_target2d_lizard_continuous_ratio4_smoke_3070_2d_wgpu.toml
```

Promotion requires conservation, zero overflow, dynamic-versus-static
allocation benefit, matched fixed-oracle quality, bounded worst-seed drift,
and synchronized wall-time evidence.

## 3D Research

The bounded command surfaces are:

```bash
cargo run --release -p burn_automata -- \
  train-render3d \
  --config configs/verified/3d/oracle/torus_oracle_smoke.toml

cargo run --release -p burn_automata -- \
  train-render3d-adapters \
  --config configs/verified/3d/adapters/torus_smoke.toml

python3 scripts/validation/3d/validate_3d_catalog.py --help
```

No local 3D model currently passes catalog promotion. See
[`../research/3d_status.md`](../research/3d_status.md).

## Web

Build the viewer and isolated browser training worker:

```bash
scripts/web/build_wasm.sh
python3 -m http.server 4173 --directory www
WEB_BASE_URL=http://127.0.0.1:4173/ \
  node scripts/web/validate_web_runtime.mjs --all
```

GitHub Pages runs the static browser gate before deployment and an optional
WebGPU runtime gate on runners that expose a usable adapter.

## Publications

Compile and inspect both versioned papers:

```bash
(
  cd docs/papers/adaptive
  latexmk -pdf -interaction=nonstopmode -halt-on-error adaptive_npa.tex
)
(
  cd docs/papers/hypernpa
  latexmk -pdf -interaction=nonstopmode -halt-on-error hyper_npa.tex
)

pdfinfo docs/papers/adaptive/adaptive_npa.pdf
pdfinfo docs/papers/hypernpa/hyper_npa.pdf
```

Compilation must have no LaTeX errors, undefined references/citations, or
overfull boxes. Paper figures must be actual renderer captures with provenance,
not targets relabeled as rollouts.
