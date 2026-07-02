# burn_automata

[![CI](https://github.com/mosure/burn_automata/actions/workflows/test.yml/badge.svg)](https://github.com/mosure/burn_automata/actions/workflows/test.yml)
[![Bevy](https://img.shields.io/badge/bevy-0.19.0-232326)](https://bevy.org)
[![Burn](https://img.shields.io/badge/burn-0.21.0-8a4fff)](https://burn.dev)

Burn Neural Particle Automata kernels, rollout/training code, and a Bevy viewer.

This repository is structured after `burn_reconstruction`: a kernel crate, a Burn-facing core crate, a Bevy app crate, version correlation in the workspace manifest, CI workflows, scripts, and a `www/` publishing surface.

## Crates

| crate | purpose |
| --- | --- |
| `burn_automata_kernels` | optimized deterministic CPU kernels for hashgrid SPH perception, 2D splatting, 3D Gaussian decoding, and Euler integration |
| `burn_automata` | NPA model config, rollout, `.bpk` import/export, CLI, Burn tensor bridge, direct WGPU inference executor, and supervised forward/backward training baseline |
| `bevy_automata` | Bevy 0.19 viewer with BSN UI, live rollout/backward/training status, optional Gaussian splatting startup, and GPU interop extension points |
| `vendor/bevy_burn` | local bridge facade for binding Burn metadata and Bevy `ShaderBuffer` assets while upstream APIs settle |

## Version Matrix

| stack | version |
| --- | --- |
| Rust | `1.95` workspace minimum |
| Burn | `0.21.0` |
| Bevy | `0.19.0` |
| WGPU | `29` |
| bevy_gaussian_splatting | `8.0.1` |
| bevy_interleave | pulled through `bevy_gaussian_splatting` as `0.10` |

## Status

Implemented:

- 2D and 3D NPA presets based on the self-organising NPA reference project.
- Deterministic CPU SPH perception using normalized upstream poly6/spiky kernels, moment-matrix state-gradient correction, log-normalized feature scaling, hashgrid neighbor traversal, and fused operator passes.
- Optional `gpu_wgpu` executor with GPU linked-cell or fixed-bucket local neighbor traversal, density pass, perception/MLP update, stochastic update masking, Euler integration, persistent ping-pong rollout state, and direct gaussian planar-buffer writes on WGPU storage buffers. CLI export/bench convenience paths still read back for reporting.
- Seeded MLP update model, checksumed `.bpk` container, direct upstream `.pth` import, JSON fallback, and CLI import/infer/train/bench commands.
- Manual supervised NPA training baseline with finite gradient checks, clipping, repeatable convergence history, rollout-local teacher-seed/teacher-BPK distillation targets, explicit feature-row fallback, and trained `.bpk` export.
- Configurable inference equivariance modes: `None`, `ParticleDensity`, and default upstream-compatible `ParticleDensityAndScale`.
- Bevy 0.19 viewer crate using BSN for UI construction, model switching through `BURN_AUTOMATA_MODEL`, live forward status, backward-gradient probe, live training/convergence controls, `bevy_gaussian_splatting` compile coverage, and helpers for borrowing Bevy planar gaussian storage as Burn WGPU output buffers.
- Local `bevy_burn` buffer binding facade with Bevy `ShaderBuffer` asset helpers for future render-world and Burn backend handoff.
- Tests for kernel constants, rollout, training/backward gradients, `.bpk` manifest roundtrip/checksum rejection, upstream checkpoint import, Burn tensor bridge, WGPU/CPU one-step and persistent 3D parity, Bevy planar gaussian GPU-buffer linkage, headless offscreen gaussian rendering, and buffer bridge descriptors.
- Dependency-free imported-model parity script comparing CPU or WGPU BPK rollouts against a pure Python upstream-formula reference, including 2D raster PSNR.

Tracked next:

- Coalesced sorted/prefix-sum or tiled neighbor kernels for dense tens-of-thousands particle workloads.
- Native Burn training graph and autodiff rollout losses; the current 3D render-loss trainer defaults to a CPU direct-rollout backend that applies analytic render position/opacity/color adjoints through stored rollout MLP outputs, direct Euler position integration, fixed-neighborhood SPH state-perception adjoints, and a damped fixed-neighborhood SPH position-perception adjoint, while still treating changing neighbor membership and rollout-time render visibility as stop-gradient.
- Render-world extraction from native Burn buffers into the persistent WGPU automata state.
- Browser visual regression checks.

## Usage

Generate a seeded manifest:

```bash
cargo run -p burn_automata --bin burn_automata -- manifest --preset growing-2d --output models/seed.bpk
```

Run CPU inference:

```bash
cargo run --release -p burn_automata --bin burn_automata -- infer --model models/seed.bpk --steps 32 --particles 4096 --update-prob 1.0 --output artifacts/rollout.json
```

Run deterministic WGPU inference export:

```bash
cargo run --release -p burn_automata --features gpu_wgpu --bin burn_automata -- infer --gpu --model models/seed.bpk --steps 32 --particles 4096 --update-prob 1.0 --output artifacts/gpu_rollout.json
```

The Bevy growing/lizard viewer defaults to the upstream stochastic rollout setting `update_prob=0.5`; deterministic examples use `1.0` for exact CPU/WGPU parity checks.

Run rollout-local supervised training with a deterministic seeded teacher
target. If neither `--target-seed` nor `--target-model` is provided, the CLI
defaults to a seeded teacher target (`42`) and samples actual local rollout
states instead of random feature rows or a zero-update hold objective:

```bash
cargo run -p burn_automata --bin burn_automata -- train \
  --preset growing-2d \
  --rows 64 \
  --steps 64 \
  --report-interval 8 \
  --learning-rate 0.01 \
  --target-seed 99 \
  --batch-source rollout \
  --rollout-particles 1024 \
  --rollout-steps 16 \
  --output artifacts/training_report.json \
  --model-output artifacts/trained_student.bpk
```

For imported-model parity or distillation experiments, replace `--target-seed 99` with `--target-model models/catalog/growing/lizard.bpk`. The JSON report records initial/final/best loss and sampled convergence history; `--model-output` writes the trained student as a checksumed `.bpk`.
Use `--zero-update` only for deliberate stationary/hold artifacts; it is
mutually exclusive with teacher-seed and teacher-model targets.
Use `--batch-source features` only for low-level MLP regression checks; it is no
longer the default path for 2D or 3D training.

The retired built-in 3D mesh commands now default to writing legacy diagnostic
artifacts under `artifacts/`, not catalog models. Multi-view render-proxy
experiments should write to `target/` or `artifacts/` until they pass the
strict mesh/render gates. `train-render3d` refuses `assets/models/*` outputs
unless the run starts from a conditionless-local base model and uses the
matching local 3D growth seed; catalog-bound candidates are validated from a
temporary `target/` path and only promoted after strict app-scale multi-seed
growth validation passes at the 1024-particle catalog and viewer horizons.
Legacy, ablation, and retiming 3D commands refuse
catalog-bound output paths entirely. The preferred scaling path is the
many-object shared-base plus LoRA suite:

```bash
cargo run -p burn_automata --release --bin burn_automata -- train-render3d-adapters \
  --target-set many \
  --output-dir artifacts/render_3d_adapter_suite \
  --report-output artifacts/render_3d_adapter_suite_report.json
```

Use single-target `train-render3d` commands for focused diagnostics:

```bash
cargo run -p burn_automata --release --bin burn_automata -- train-render3d \
  --target torus \
  --base-model assets/models/uv_torus_growth_3d.bpk \
  --seed-mode torus-growth-3d \
  --extra-selection-seed 42 \
  --extra-selection-seed 99 \
  --model-output target/uv_torus_render_probe_3d.bpk \
  --report-output artifacts/uv_torus_render_probe_3d_training_report.json

cargo run -p burn_automata --release --bin burn_automata -- train-render3d \
  --target teapot \
  --base-model assets/models/teapot_growth_3d.bpk \
  --seed-mode teapot-growth-3d \
  --extra-selection-seed 42 \
  --extra-selection-seed 99 \
  --model-output target/teapot_render_probe_3d.bpk \
  --report-output artifacts/teapot_render_probe_3d_training_report.json
```

The legacy projection/seed-frame batch is still available as
`--training-mode projection-baseline` for mesh-target sanity checks, and
`--training-mode rollout-local` remains available for local teacher
distillation experiments. `--training-mode rollout-position-field` is available
for rollout-state mesh rows, but those commands are no longer catalog defaults.
`train-render3d` defaults to `--training-backend direct-rollout` and
`--weight-update-mode adapter`. Without `--base-model`, it starts from a
conditionless-local compact-growth prior with `position_features=false` and
target-agnostic local growth seed defaults; with `--base-model`, it treats the
provided local-growth BPK as a frozen shared base and trains a LoRA-style
low-rank object adapter (`--adapter-rank`, `--adapter-alpha`,
`--adapter-seed`). Reports serialize the adapter parameter count and base/full
parameter counts; the exported `.bpk` is a materialized compatibility model.
For many-object shared-base sweeps, use `train-render3d-adapters`. It defaults
to `--target-set many`, covering torus, teapot, sphere, ellipsoid, cube,
cylinder, cone, capsule, pyramid, bicone, dumbbell, and cross. `--target-set
core` is the smaller torus/teapot diagnostic set, and `--target-set primitives`
expands to the ten object-agnostic procedural mesh classes. Explicit
`--targets` runs focused subsets.
`--holdout-targets` removes targets from shared-base cycles while still fitting
adapter-only held-out objects. By default, no-arg many-object suites use an
effective `--auto-holdout-stride 4 --auto-holdout-offset 3`, holding out
ellipsoid, capsule, and cross while keeping torus and teapot in shared-base
training; override these flags for a different split. Without `--base-model`,
the suite initializes an object-agnostic conditionless-local 3D growth base,
alternates full-weight shared-base training for `--shared-base-cycles` cycles
(`many` defaults to two cycles), saves `shared_base.bpk`, evaluates that frozen
base on every target, then trains one compact `.adapter.json` LoRA artifact per
target. With `--base-model`, the suite freezes the supplied base by default;
pass `--shared-base-cycles` to continue shared-base training before adapter
fitting. Materialized validation/viewer BPKs are written beside the adapters,
and the suite report records shared-base generalization, adapted train/holdout
quality, explicit shared/holdout/adapter target counts, single-adapter
efficiency, and shared-base-plus-adapter-bank efficiency. The suite also writes
`adapter_bank.json` by default; this compact manifest includes the same split
counts and is the intended target format for future HyperNPA models that
predict object LoRA adapters from conditions. Both the suite report and
adapter-bank manifest include `strategy="shared_base_low_rank_object_adapters"`
and a `contract` block. The no-arg many-object contract fails if the run
collapses back to only torus/teapot, if too few non-core objects participate, or
if adapters are not produced for every target. Use `train-render3d` for
single-target diagnostics and `--target-set core` only for the small
torus/teapot regression suite; use `--weight-update-mode full` only for legacy
full-model ablations. The backend
backpropagates the deterministic CPU multi-view splat loss analytically to
final particle positions, opacity, and color, and applies those adjoints through
the stored rollout MLP outputs. It also
propagates RGB/opacity state adjoints through direct, blurred, and
state-gradient SPH perception channels over stored snapshots, carries position
adjoints through direct Euler integration, and applies a conservative
`--perception-position-gain` to fixed-neighborhood SPH position-perception
adjoints. Direct-rollout training now averages clipped SGD deltas over the
selection seed set by default; use `--no-direct-selection-seed-training` only
for the older single-seed ablation. `--trajectory-render-gain`/`--trajectory-render-samples`
can inject render adjoints at stored rollout snapshots, while
`--trajectory-mesh-gain` uses the same snapshot schedule for mesh
coverage/surface adjoints without enabling intermediate render gradients.
`--liveness-gain`
adds a bounded local-front state adjoint on the strict liveness channel
(`state[3]`), so render/coverage training can teach progressive activation
instead of only material opacity. This liveness snapshot path uses
`--trajectory-render-samples` but does not require nonzero
`--trajectory-render-gain`; far dormant particles receive no global activation
pressure. The direct backend now applies target-coverage pressure to
all active particles by default instead of only sampled render-gradient rows.
Coverage adjoints use `--coverage-gain` directly rather than being scaled by
the render `--motion-gain`; use `--no-full-coverage-adjoint` only for the older
sparse-row ablation. The older supervised row projection remains
available as `--training-backend proxy`.
`--coverage-mode soft-chamfer` adds an opt-in detached soft target-coverage
proxy, and `--coverage-repulsion-gain`/`--coverage-repulsion-radius` can add
mesh-tangent particle spread pressure for collapse ablations. These improve
some teapot/torus diagnostics but remain non-default because the strict
multi-seed render and torus tube-coverage gates still fail.
`--coverage-mode sliced-ot` is also available for balanced distributional
coverage in local 3D ablations; it is tested, but current torus probes regress
strict validation, so it is diagnostic only. `retime-growth3d --alpha` can
scale motion in a copied BPK, and `--skip-front-retime` isolates that from
front-controller retiming. Alpha retiming improves some torus diagnostics but
does not pass strict coverage/render gates and should not be promoted directly.
Mesh target sampling is now area-weighted and low-discrepancy instead of
face-prefix centroid sampling, so low sample counts exercise the whole target
surface during render/coverage training and validation. Random mesh surface
sampling uses the same area-weighted policy.
Gradient rows are sampled across the whole cloud instead of only the particle
array prefix, so under-covered regions participate in continuation probes.
`--gradient-mode finite-diff` remains available for regression checks. This is
an honest backend scaffold, not full WGPU/autodiff BPTT through particle
positions inside perception, changing neighbor membership, and render
visibility through time. `--surface-escape-gain` weights active particles that
escape past the strict surface-distance threshold more strongly in the
terminal, trajectory, and proxy surface projection objectives, so reported
surface tails have a matching training signal instead of being diagnostics
only. `train-render3d` reports serialize a
`growth_validation` section with the same strict gate and seed set used by
catalog promotion, and `--fail-on-validation` fails on that strict
runtime-dynamics gate rather than render PSNR alone. Candidate selection uses
that strict score, including active-surface max/tail and render-density
penalties. It also records target-normal-bin coverage, so missing broad surface
normal families such as torus tube collapse are strict blockers even when
pointwise target coverage looks acceptable. Selection only allows bounded
render/density slack when the strict score improves materially. Rejected rounds
are rolled back before the next rollout, so subsequent training continues from
the best strict-scored checkpoint instead of a regressed candidate.

For the stricter no-position experiment, use the explicit ablation command:

```bash
cargo run -p burn_automata --release --bin burn_automata -- ablate-local-3d --target torus
cargo run -p burn_automata --release --bin burn_automata -- ablate-local-3d --target teapot
```

This path uses `position_features=false`, compact random-ball seeds, refreshed
local rollout rows, full-cloud perception/target context before supervised row
sampling, and writes reports such as
`artifacts/conditionless_torus_3d_ablation_report.json`. Current 3D growth
artifacts also use compact growth seeds and no absolute position features, but
they are still experimental because they fail strict rendered density/mesh
validation. The Bevy app catalog currently exposes only the generic 3D preset
at 1024 particles; the latest teapot and torus artifacts are kept as hidden
regression targets because they still fail strict coverage/render gates at that
interactive scale. Catalog validation uses the app-eval seed from the render
sanity tests (`0x51a7_3d`) so viewer behavior matches the validated rollout
path; see `docs/local_3d_morphogenesis.md`.
Use `--base-model <local-growth.bpk>` to continue a previous conditionless-local
artifact with refreshed rollout rows; the command rejects position-field,
seed-frame, and render-proxy shortcut lineages.

Evaluate multi-view rendered density/color/depth loss for a saved 3D model:

```bash
cargo run -p burn_automata --release --bin burn_automata -- render-loss-3d \
  --model assets/models/uv_torus_growth_3d.bpk \
  --target torus \
  --seed-mode torus-growth-3d
```

The render harness uses deterministic orthographic Gaussian splats over
`xy`/`xz`/`yz`/isometric views, count-matches mesh target samples to rollout
particles by default, and reports relative density MSE/PSNR, gated color
MSE/PSNR, and depth-moment MSE/PSNR. It is currently a CPU correctness oracle
and validation objective. The `train-render3d` direct-rollout backend trains
against the analytic-gradient version of this objective over stored trajectory
snapshots, carries a fixed-neighborhood recurrent SPH state adjoint for
RGB/opacity channels, backpropagates through direct Euler position integration,
and can add bounded target-coverage residual rows. Native differentiable/GPU
training through position-dependent perception, changing neighborhoods, render
visibility, and the full rollout remains the next backend step. Checkpoint
selection uses the training seed, `--selection-seed`, and any
`--extra-selection-seed` values as a worst-case strict-score guard.

Run the viewer:

```bash
cargo run --release -p bevy_automata
cargo run --release -p bevy_automata --no-default-features --features "viewer splatting"
```

The default viewer command runs the resident render-world WGPU automata state
and writes directly into `bevy_gaussian_splatting` buffers. The
`--no-default-features` command keeps the CPU rollout-to-planar-gaussian
fallback for environments where the direct WGPU bridge is being isolated. The
viewer has a `train` toggle and `train lr` slider. Live training freezes the
current model as a local rollout teacher, samples a bounded probe batch from
the current rollout, applies clipped supervised SGD on the CPU-side model,
reports convergence in the BSN status panel, and pushes updated weights into
the resident render-world WGPU automata state without a host readback path for
gaussian rendering.

Run profiled target benchmarks:

```bash
scripts/bench_rollout.sh
PRESET=growing-3d-gs PARTICLES=16384 STEPS=1 scripts/bench_rollout.sh
cargo run --release -p burn_automata --features gpu_wgpu --bin burn_automata -- bench --preset growing-3d-gs --particles 4096 --steps 2 --gpu
cargo run --release -p burn_automata --features gpu_wgpu --bin burn_automata -- bench --preset texture-2d --particles 4096 --steps 16 --gpu --neighbor-mode auto
cargo run --release -p burn_automata --features gpu_wgpu --bin burn_automata -- bench --preset texture-2d --particles 4096 --steps 16 --gpu --neighbor-mode linked-list
cargo run --release -p burn_automata --features gpu_wgpu --bin burn_automata -- bench --preset growing-3d-gs --particles 16384 --steps 16 --gpu --gaussian
scripts/bench_gpu_matrix.py --matrix quick --output target/bench_gpu_matrix_final.json
scripts/bench_seed_scale_matrix.py --output target/bench_seed_scale_matrix.json
```

Run the focused checks:

```bash
scripts/ci_check.sh
scripts/check_inference_features.sh
scripts/validate_3d_catalog.py
REQUIRE_BPK=1 scripts/validate_gpu_e2e.sh
CATALOG_PARITY=1 SELFORG_WEB_ROOT=/tmp/selforg_npa_web scripts/validate_gpu_e2e.sh
```

## Upstream NPA Import

The upstream project at <https://selforg-npa.github.io/> publishes PyTorch checkpoints for 2D neural particle automata. This workspace can import the published PyTorch zip checkpoints directly:

```bash
cargo run -p burn_automata --bin burn_automata -- import \
  --input data/pretrained/lizard.pth \
  --output models/catalog/growing/lizard.bpk
```

The importer reads the checkpoint storages, infers NPA dimensions from the first two MLP layers, records source metadata, and writes a checksumed `.bpk` container. JSON interchange files from `scripts/export_npa_checkpoint.py` are still supported for debugging or unsupported checkpoint variants.

The web demo publishes many additional 2D growing/texture models as base64 JSON tensors. Import the curated Bevy catalog from a clone of `SelfOrg-NPA.github.io`:

```bash
git clone --depth 1 https://github.com/SelfOrg-NPA/SelfOrg-NPA.github.io /tmp/selforg_npa_web
python3 scripts/import_selforg_catalog.py --web-root /tmp/selforg_npa_web
```

Validate numerical parity for an imported checkpoint:

```bash
python3 scripts/validate_import_parity.py --model models/catalog/growing/lizard.bpk --particles 64 --preset growing-2d --seed-scale 0.2
python3 scripts/validate_import_parity.py --model models/catalog/texture/polka_dotted_0121.bpk --particles 64 --preset texture-2d --seed-scale 1.0
python3 scripts/validate_import_parity.py --model models/catalog/growing/lizard.bpk --particles 64 --preset growing-2d --seed-scale 0.2 --gpu --steps 4 --psnr-threshold 70 --hidden-psnr-threshold 70
python3 scripts/validate_import_parity.py --model models/catalog/texture/polka_dotted_0121.bpk --particles 64 --preset texture-2d --seed-scale 1.0 --gpu --steps 4 --psnr-threshold 70 --hidden-psnr-threshold 70
python3 scripts/validate_catalog_parity.py --web-root /tmp/selforg_npa_web --gpu --build-binary --require-all
```

On the current ARM/NVIDIA workstation, release-mode CPU inference measures roughly 14-29 ms for `growing-2d` at 4096 particles depending on benchmark harness, 6-25 ms for `texture-2d` at 4096 particles, and about 21 ms for `growing-3d-gs` at 16,384 particles in criterion. The resident WGPU benchmark keeps rollout state on GPU, times submit/wait separately from final readback, normalizes particle-grid `eps` to seed scale by default, and reports `grid_overflow_count`; any nonzero fixed-bucket overflow is not a valid exact result. `--fixed-eps` intentionally reproduces density-sensitive pathological seed-scale sweeps. `--neighbor-mode auto` keeps normal particle-hash grids on linked lists, but switches high initial occupancy particle-grid starts to fixed buckets with adaptive headroom. `--neighbor-mode tiled` enables the active-cell/shared-memory tiled fixed-bucket kernel for profiling; `--neighbor-mode sorted` enables the exact overflow-free prefix-sum cell layout with `O(cells + particles)` grid memory. Current measurements show scalar fixed buckets are faster for the present NPA update workload on this GPU when overflow stays zero. Local exact auto timings are about 9.1 ms/step for 4k 3D, 12.6 ms/step for 16k 3D, 31.9 ms/step for 32k 3D, 16.0 ms/step for 4k texture 2D, 60.6 ms/step for 16k texture 2D, 55.8 ms/step for dense 4k growing 2D, and 115.2 ms/step for dense 8k growing 2D.

## References

- Self-Organising Neural Particle Automata: <https://selforg-npa.github.io/>
- Burn: <https://burn.dev>
- Bevy: <https://bevy.org>
- bevy_gaussian_splatting: <https://github.com/mosure/bevy_gaussian_splatting>
