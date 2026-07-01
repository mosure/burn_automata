# Validation

## Unit

- `cargo test -p burn_automata_kernels`
- `cargo test -p burn_automata`
- `cargo test -p bevy_burn`

These cover shape checks, finite outputs, normalized SPH constants, rollout determinism, training/backward updates, `.bpk` roundtrips and checksum rejection, direct `.pth` checkpoint import, Burn tensor bridge shapes, and Bevy `ShaderBuffer` bridge descriptors.

The core training tests also finite-difference representative MLP weight and bias gradients against the exact supervised loss used by `supervised_backward`, and verify multi-step `run_supervised_training` convergence history.

## Compile Matrix

- `cargo check -p burn_automata --examples --benches`
- `cargo check -p burn_automata --no-default-features --features backend_wgpu`
- `cargo check -p burn_automata --no-default-features --features gpu_wgpu`
- `cargo test -p burn_automata --test gpu_wgpu --no-default-features --features gpu_wgpu`
- `cargo test -p bevy_automata --no-default-features --features "splatting gpu_wgpu" --test gaussian_gpu_link`
- `scripts/check_inference_features.sh`
- `scripts/build_wasm.sh`

The Burn/WGPU command verifies the Burn backend feature surface compiles. The `gpu_wgpu` commands compile and dispatch the direct WGPU inference executor, then compare one full step against the CPU oracle for growing 2D, periodic texture 2D, growing 3D Gaussian-splatting shape, and Point-MNIST shape. They validate linked-list, fixed-bucket, and auto neighbor traversal, a persistent 3D GPU rollout, scale-equivariant WGPU inference, and a GPU-only automata-step-to-gaussian-buffer write. The inference feature check asserts the direct WGPU CLI build does not pull Burn autodiff or Burn fusion. The Bevy gaussian test constructs the generated `PlanarStorageGaussian3d` storage buffers and verifies Burn WGPU compute writes the exact buffers consumed by `bevy_gaussian_splatting`/`bevy_interleave`; it also runs a headless offscreen gaussian render and asserts the captured image is nonblank. The wasm script builds the Bevy viewer target with the required `getrandom` backend configuration.

## Viewer

- `cargo check -p bevy_automata --no-default-features --features viewer`
- `cargo check -p bevy_automata`

The first command isolates BSN UI and rollout. The second includes Gaussian splatting dependencies.

Viewer unit tests cover the BSN UI toggle, catalog setting preservation, live training probe metric updates, and model-revision bumps used by the render-world WGPU weight update path.

## Training

Run a repeatable rollout-local teacher-seed distillation smoke test:

```bash
cargo run -p burn_automata --features cli -- train \
  --rows 64 \
  --steps 16 \
  --report-interval 4 \
  --learning-rate 0.01 \
  --target-seed 99 \
  --batch-source rollout \
  --rollout-particles 1024 \
  --rollout-steps 16 \
  --output target/training_report.json \
  --model-output target/trained_seeded_model.bpk
```

The generic `train` command defaults to a seeded teacher target (`42`) when no
target is supplied, and its default `--batch-source rollout` samples actual
local rollout states before building the supervised update rows. Accidental
zero-update/stationary model training and random feature-row regression are not
the default. Use `--target-model path/to/model.bpk` instead of `--target-seed`
to train a student against an imported upstream BPK target. Use
`--batch-source features` only for low-level MLP checks, and use
`--zero-update` only for deliberate hold/stationary artifacts; it is mutually
exclusive with teacher targets.

The latest local rollout-supervision smoke runs reduced 2D loss from
`2.1044495` to `1.2905985` and 3D loss from `1.6761311` to `0.44246006` over
16 steps with four history samples.

The legacy `train-torus3d`, `train-torus-morphogen3d`, and
`train-teapot-morphogen3d` commands now write diagnostic artifacts under
`artifacts/` and refuse catalog-bound output paths. Their position-field,
rollout-position-field, and projection-baseline modes remain useful for mesh
sanity checks, but they are not catalog-promotion paths.

The promotion-oriented path is `train-render3d`. Without `--base-model`, it now
starts from a conditionless-local compact-growth prior with
`position_features=false`, target-local growth seed defaults, and
`ablation-rust:*conditionless-local*` lineage. With `--base-model`, it can
continue an explicit local-growth BPK. Any `assets/models/*` output is written
through a temporary candidate and is refused unless strict multi-seed
`validate-growth3d` gates pass. Every `train-render3d` report also contains a
`growth_validation` section generated from the saved BPK with the same strict
gate, training seed, `--selection-seed`, and `--extra-selection-seed` set, so
diagnostic artifacts carry the runtime-dynamics blockers that prevent catalog
promotion.

The explicit conditionless-local ablation command is:

```bash
cargo run -p burn_automata --release --bin burn_automata -- ablate-local-3d --target torus
cargo run -p burn_automata --release --bin burn_automata -- ablate-local-3d --target teapot
```

It trains with `position_features=false`, random-ball seeds, no target
residual/color state, and refreshed local rollout rows. It writes JSON reports
without replacing catalog artifacts unless a caller explicitly uses the output
path. Passing `--base-model <path>` continues training from a previous
conditionless-local BPK and preserves `continued-from=...` lineage in the saved
manifest; the loader rejects position-feature, position-field, seed-frame, and
render-proxy shortcut bases before training starts.

The multi-view render-loss command evaluates saved 3D models against
orthographic Gaussian-splat density/color/depth targets:

```bash
cargo run -p burn_automata --release --bin burn_automata -- render-loss-3d \
  --model assets/models/uv_torus_growth_3d.bpk \
  --target torus \
  --seed-mode torus-growth-3d
```

The render target samples are count-matched to rollout particles by default.
The report stores relative density loss, density PSNR, density-gated color
loss, color PSNR, depth-moment loss, and depth PSNR for `xy`, `xz`, `yz`, and
isometric views. Particle splats are weighted by the model opacity state using
the same sigmoid opacity range as the Bevy/WGPU Gaussian path, so dormant
particles no longer count as fully visible in render validation. This is a CPU
correctness oracle and the objective used by `train-render3d`. The trainer now
defaults to `--training-backend direct-rollout`, which uses analytic CPU
gradients from render loss to final particle positions, opacity, and color,
then applies those adjoints through stored rollout MLP outputs and a
fixed-neighborhood SPH state-perception adjoint. The older supervised
projection path is still available as `--training-backend proxy`, with finite
differences left as an explicit regression fallback. The direct backend is
still not true BPTT: direct Euler position integration is differentiated, but
position-dependent perception, density-gradient position terms, future neighbor
geometry, and rendering through time are treated as stop-gradient. The
remaining backend gap is differentiable or WGPU training through the full
rollout dynamics.

The `validate-growth3d` command combines active 3D catalog promotion checks
into a JSON report:

```bash
cargo run -p burn_automata --release --bin burn_automata -- validate-growth3d \
  --model assets/models/uv_torus_growth_3d.bpk \
  --target torus \
  --seed-mode torus-growth-3d \
  --particles 256 \
  --steps 64 \
  --image-size 32 \
  --target-samples 512 \
  --gate strict
```

The report includes source-lineage checks, neutral non-opacity seed-state
checks, active-core growth metrics, raw and active-particle surface distances,
target-surface coverage, temporal geometry progress, local-front coherence, a
rollout motion profile, and the same multiview render loss. The motion profile
records first/peak/final/late mean motion plus active and sustained step
fractions; strict validation now rejects one-shot motion bursts as well as fully
static rollouts. The local-front report samples the rollout over time and
requires newly active particles to appear near the previous active front rather
than waking up directly at distant target positions. The temporal geometry
report records active-surface mean ratio, target-coverage mean ratio, and
coverage-fraction delta from seed to final sample, so activation that appears at
once without becoming more target-like is not accepted as morphogenesis. The
`strict_checks.failure_reasons` array names each failed criterion. The default
`--gate strict` is the research acceptance gate for future promoted fully local
3D morphogenesis models. Current catalog artifacts intentionally fail that
strict gate on target coverage/rendered density.

For strict low-particle regression checks of the current catalog artifacts, use
the catalog sanity gate with the held-out seed sweep:

```bash
cargo run -p burn_automata --release --bin burn_automata -- validate-growth3d \
  --model assets/models/uv_torus_growth_3d.bpk \
  --target torus \
  --seed-mode torus-growth-3d \
  --seed 5351229 \
  --particles 512 \
  --steps 64 \
  --seed-scale 0.54 \
  --image-size 48 \
  --target-samples 1024 \
  --world-scale 1.08 \
  --gate catalog-sanity \
  --extra-seed 42 \
  --extra-seed 99
```

The catalog gate still rejects absolute world-position features, target-bearing
non-opacity seed state, shortcut lineage, static rollouts, non-local front
activation, and weak active-core growth; it then applies the catalog render
floors. It is not a substitute for the strict gate. Run low-particle reports
without `--fail-on-validation` while inspecting the present
target-coverage/render-density gap. Add `--fail-on-validation` only when using
the command as a passing promotion gate.

`train-render3d` reports now distinguish `selection_loss` from
`selection_score`. The former is averaged multi-view render loss across
the effective selection seed list; the latter is the worst-case strict-score
distance to the promotion gate across that list. The list always starts with
the training seed, then includes `--selection-seed` and any deduped
`--extra-selection-seed` values. The report serializes the effective
`selection_seeds`. A candidate round also records
`selection_morphology_non_regressed=false` if any selection seed regresses
active-surface max or target coverage relative to the initial model. Those
rounds are not selected even when render loss improves. Each history row also
records `selection_worst_seed` and `selection_worst_failure_reasons`, which are
the held-out seed and strict-check blockers that currently dominate promotion
selection.

The current 3D growth artifacts are latest dynamic local-front artifacts with
`position_features=false`, but they are not selectable mesh models in the Bevy
catalog. The app now defaults 3D catalog/preset entries to `1024` particles for
interactive performance. At that scale, teapot still passes the older
`catalog-sanity` thresholds, but strict validation shows low target coverage
and low rendered density PSNR, matching the observed incoherent viewer output.
Both teapot and torus remain under `assets/models` as blocked regression
targets, while the visible catalog exposes only the generic 3D preset until a
trained mesh artifact passes the stricter coverage/render gate.

At the low-particle regression scale (`512` particles, three seeds), both
assets remain research artifacts and fail catalog/strict gates. Torus uses the
current app catalog seed scale (`0.54`) in this table:

| model | training objective | loop loss | saved render total | density PSNR | color PSNR | depth PSNR | final opacity max | catalog gate | strict gate | strict score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| `uv_torus_growth_3d.bpk` | bounded-opacity local-front dynamic growth + guarded render refinement | n/a | `0.935` | `0.514` | `19.789` | `14.373` | `0.257` | failed | failed | `1.382` |
| `teapot_growth_3d.bpk` | local-front dynamic growth + guarded render refinement | n/a | `0.983` | `0.147` | `22.680` | `19.665` | `0.193` | failed | failed | `1.322` |

At the current Bevy app scale (`1024` particles, primary app seed,
`target-samples=4096`), both trained mesh artifacts remain hidden. Teapot is
allowed to pass the old `catalog-sanity` metric as a regression artifact, but
the visible-catalog policy now uses strict blockers so this partial cloud is
not presented as a coherent teapot:

| model | particles | seed scale | render total | density PSNR | color PSNR | depth PSNR | target coverage | catalog gate | strict blockers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `uv_torus_growth_3d.bpk` | `1024` | `0.54` | `0.815` | `1.248` | `17.504` | `13.244` | `0.338` | failed | coverage fraction, torus angular coverage, render |
| `teapot_growth_3d.bpk` | `1024` | `0.72` | `0.591` | `2.368` | `20.063` | `27.605` | `0.180` | old sanity passed, hidden | coverage fraction, render |

Regenerate and assert the full app-facing 3D catalog state with:

```bash
scripts/validate_3d_catalog.py
```

Candidate 3D artifacts should be validated with the same `validate-growth3d`
settings, then compared against the active baseline before promotion:

```bash
scripts/compare_3d_candidate.py \
  --baseline-report target/uv_torus_growth_3d_catalog_active_validation.json \
  --candidate-report target/uv_torus_growth_3d_candidate_validation.json
```

The comparison rejects candidates that drop local-growth/no-shortcut checks or
regress primary and robust render loss, depth PSNR, target coverage, torus
angular coverage, seed-perturbation stability, or final opacity. Add
`--require-catalog-safe` when a candidate is intended to become visible in the
Bevy catalog.

The script runs the 64-step catalog horizon and 96-step viewer horizon for the
teapot and torus BPKs at the app seed plus held-out seeds `42` and `99`, then
checks lineage, `position_features=false`, neutral non-opacity seed state,
local motion/front progress, color-state emergence, particle-order permutation
consistency, seed-perturbation stability, worst-case render/color/depth
robustness, and Bevy catalog exposure policy. Visible trained 3D mesh entries
must pass strict checks; hidden regression artifacts may pass the older
`catalog-sanity` threshold, but they remain hidden until their strict coverage,
angular-support, and render-density failures are fixed.

At the teapot viewer cadence (`steps/frame=2`) the same artifact reaches the
validated 96-step horizon in 48 rendered frames. At 1024 particles, longer
rollout activates almost every particle but still does not cover the teapot
target densely enough:

| model | particles | steps | render total | density PSNR | target coverage | catalog gate | strict score | strict blockers |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `uv_torus_growth_3d.bpk` | `1024` | `96` | `0.876` | `0.924` | `0.368` | failed | `11.227` | surface mean, coverage fraction, torus angular coverage, render |
| `teapot_growth_3d.bpk` | `1024` | `96` | `0.577` | `2.487` | `0.273` | old sanity passed, hidden | `1.078` | coverage fraction, render |

Raw active-surface max distance is still serialized as a diagnostic, but strict
acceptance uses an active-particle tail gate (`p99 < 0.36`, unweighted and
opacity-weighted over-threshold fractions <= `0.005`). This keeps a few
low-opacity outliers from blocking otherwise local-growth-valid rollouts while
still rejecting broad surface drift. At the 96-step teapot horizon, raw max is
`0.691`, but p99 is `0.298` and only `0.002` of active particles exceed the
surface threshold, so the remaining strict failures are target coverage and
rendered density.

The same validation report now serializes `initial_color_state` and
`final_color_state` for the final three neural state channels. Strict/catalog
validation requires the active seed color state to be neutral and the final
active color state to become nonzero and non-uniform. Current 1024-particle
reports show this is true for the latest local-growth artifacts: teapot grows
active mean/stddev to `0.108/0.057` at 64 steps and `0.124/0.066` at 96 steps;
torus grows to `0.109/0.056` at 64 steps and `0.122/0.061` at 96 steps. This
catches future regressions
where particles are precolored at initialization or converge to a uniform tint
instead of deriving visible color from neural dynamics.

The report also serializes `permutation_consistency`, a cheap 256-particle,
8-step sub-rollout that shuffles the same seed cloud, evolves both orders, then
unshuffles and compares final positions/state. This catches particle-order or
index-assignment shortcuts in active catalog artifacts. Current reports pass
with tiny drift: teapot max position/state errors are `7.5e-7` / `2.4e-5`,
and torus max position/state errors are `3.5e-7` / `1.3e-5`.

The report now also serializes `seed_perturbation`, a deterministic 512-particle
sub-rollout that jitters neutral seed positions by 10% of the growth seed radius
and compares aggregate growth with the unperturbed rollout. The app-scale
validator requires every seed in the sweep to keep perturbed growth active and
within broad active-count/motion stability bounds. Current reports pass: teapot
minimum perturbed newly activated fraction is `0.876`, active-count ratio stays
in `0.978..1.057`, and peak-motion ratio stays in `1.009..1.023`; torus
minimum perturbed newly activated fraction is `0.830`, active-count ratio stays
in `0.972..1.061`, and peak-motion ratio stays in `1.001..1.020`.

Multi-seed robustness now aggregates these no-shortcut checks. With app seed
`0x51a7_3d` plus held-out seeds `42` and `99`, all current 3D artifacts show
color-state emergence, permutation consistency, seed-perturbation stability,
and sparse-core growth. Worst observed permutation position/state errors are
`1.7e-6` / `2.4e-5` for teapot and `2.7e-6` / `3.1e-5` for torus; minimum final active color-state
mean/stddev across those seeds are `0.102` / `0.058` for 64-step teapot,
`0.117` / `0.068` for 96-step teapot, `0.120` / `0.060` for 64-step torus,
and `0.137` / `0.070` for 96-step torus. Minimum active growth ratios across
the same seed sweep are above `35x`, and minimum newly activated fractions are
above `0.94`. This does not make the models strict-pass, but it prevents
single-seed, static, or precolored artifacts from being promoted silently.
Visible 3D mesh catalog entries must also clear the strict coverage/render
checks for the whole seed sweep. Current teapot worst-case target coverage is
only `0.180` at 64 steps and `0.201` at 96 steps, so it is hidden despite
passing the older catalog-sanity render thresholds. The torus artifact remains
hidden because its render total, target coverage, and torus angular support are
below the strict floor.

They should be treated as current pipeline artifacts, not as solved fully local
3D morphogenesis. Both trained mesh BPKs are hidden regression artifacts. The
blocked torus app-scale angular report catches collapsed/partial-tube support
that average mesh distance alone missed.

The strict score is a continuous distance-to-gate measure: lower is better and
zero means the strict gate passed. Current strict failure reasons with
active-surface validation:

| model | strict failure reasons |
| --- | --- |
| `uv_torus_growth_3d.bpk` | `target_coverage_fraction`, `torus_angular_coverage`, `render_loss_passed` |
| `teapot_growth_3d.bpk` | `temporal_activation_progressive`, `render_loss_passed` |

The active catalog artifacts now pass primary-seed temporal activation,
temporal geometry, and local-front coherence checks at 512 particles. In the
app-seed plus held-out seed sweep (`0x51a7_3d`, `42`, `99`), both artifacts are
local-front coherent for all seeds (`min_front_local_newly_activated_fraction=1.0`;
worst front distance `0.317` for torus and `0.314` for teapot, under the `0.36`
threshold). Multi-seed robustness remains incomplete: each model still has one
held-out seed that misses temporal activation, and both fail low-particle
coverage and rendered-density gates. Both are bounded across the same sweep
(`max_final_opacity=0.335` for torus and `0.201` for teapot, below the
validator limit of `24.0`).

`validate-growth3d --extra-seed ... --fail-on-validation` now fails on the
aggregate robustness gate, not only the primary seed. The command still prints
the primary metrics, but `robust_gate_passed=false` indicates an extra seed
failed the selected gate.

The latest local-front pipeline adds front-aware opacity supervision and a
seeded local controller based on blurred-neighbor opacity. It fixes the
temporal growth failure in probes, but those probes are not promoted because
they still fail target coverage and rendered density:

| candidate | active growth | progressive activation | saved render total | density PSNR | extent ratio | strict score | strict blockers |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| torus local-front render-refined probe | `6 -> 492` | passed | `1.052` | `-0.026` | `0.969` max-radius | `1.420` | coverage/render |
| torus local-front probe | `6 -> 501` | passed | `1.513` | `-1.692` | `0.490` max-radius | `1.638` | coverage/render |
| teapot local-front probe | `6 -> 501` | passed | `1.162` | `-0.593` | n/a | `1.390` | coverage/render |

Additional 2026-06-30 continuation probes are intentionally left under
`target/` and not promoted:

`train-render3d` now defaults diagnostic output to `artifacts/` and rejects
`assets/models/*` output paths unless the base model has conditionless-local
lineage and the seed mode is the matching local growth seed. This keeps
field-baseline or position-field render-proxy experiments from being written
directly into the catalog asset directory. Legacy mesh trainers,
`ablate-local-3d`, and `retime-growth3d` refuse catalog-bound output paths
entirely; their BPKs must be written to `target/` or `artifacts/` and then
promoted only through the app-scale validation harness.
`train-render3d` also has an opt-in render-alpha gradient path via
`--opacity-gain`; the analytic render gradient is with respect to rendered
opacity and is converted back to the NPA opacity-logit update. The
direct-rollout backend also applies color gradients to the visible RGB tail
state where the color clamp has nonzero derivative. Opacity gain remains
`0.0` by default so older position-dominant diagnostics are reproducible unless
a candidate explicitly enables the extra alpha channel.

| candidate | validation scale | render total | density PSNR | target coverage | strict score | promotion decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| torus opacity-aware render refinement | 8192 particles, app seed, scale `0.54` | `0.745` | `1.654` | `0.490` | `1.036` | rejected: no meaningful improvement over active BPK |
| torus spread-row opacity-aware render refinement | 8192 particles, app seed, scale `0.54` | `0.744` | `1.657` | `0.490` | `10.955` | rejected: numerically unchanged versus active BPK |
| torus latest local continuation, stronger coverage | 8192 particles, app seed, scale `0.54` | `0.755` | `1.659` | `0.587` | `11.161` | rejected: temporal geometry and surface gates regressed |
| torus full-context local continuation | 8192 particles, app seed, scale `0.54` | `0.704` | `1.971` | `0.482` | `21.250` | rejected: render improved but temporal geometry, surface, coverage, and torus angular support failed |
| torus full-context late-horizon local continuation | 8192 particles, app seed, scale `0.54` | `0.941` | `0.547` | `0.447` | `1.172` | rejected: render and coverage regressed |
| torus full-context surface-balanced local continuation | 8192 particles, app seed, scale `0.54` | `1.323` | `-1.021` | `0.381` | `21.488` | rejected: activation, temporal geometry, render, and coverage regressed |
| torus 1024-particle render-refinement probe | 8192 particles, app seed, scale `0.54` | `0.743` | `1.663` | `0.489` | `10.955` | rejected by `compare_3d_candidate.py`: strict score, depth, coverage, and angular regressions |
| torus render-opacity probe | 8192 particles, app seed, scale `0.54` | `0.743` | `1.664` | `0.489` | `10.955` | rejected by `compare_3d_candidate.py`: strict score, depth, coverage, and angular support regressed |
| torus local coverage/front continuation probe | 8192 particles, app seed, scale `0.54` | `1.348` | `-1.106` | `0.392` | `31.483` | rejected by `compare_3d_candidate.py`: render, geometry, opacity, coverage, and angular regressions |
| teapot spread-row opacity-aware render refinement | 8192 particles, app seed, scale `0.72` | `0.617` | `2.197` | `1.000` | `10.780` | rejected: identical to active BPK, no improvement to promote |
| teapot render-opacity probe | 8192 particles, app seed, scale `0.72` | `0.617` | `2.199` | `0.998` | `10.780` | rejected by `compare_3d_candidate.py`: depth PSNR and held-out target coverage regressed |
| torus `coverage-samples=2048` | 512 particles, seeds `0x51a7_3d`, `42`, `99` | `1.496` primary / `1.637` worst | `-1.623` primary / `-2.026` worst | `0.223` primary | `11.644` | rejected: activation/render/coverage regression |
| torus local `2048` rollout rows | 8192 particles, app seed | `1.470` | `-1.479` | `0.307` | `11.626` | rejected: activation and coverage regression |
| torus local scale `0.54` | 8192 particles, app seed | `0.731` | `1.807` | `0.499` | `11.160` | rejected: render improved but temporal geometry/coverage still regressed |
| teapot `coverage-samples=2048` | 512 particles, seeds `0x51a7_3d`, `42`, `99` | `1.105` primary / `1.195` worst | `-0.360` primary / `-0.701` worst | `0.334` primary | `11.302` | rejected: activation/render regression |
| teapot staged continuation, coverage `0.40`, extent `0.20` | `6 -> 488` | passed | `1.177` | `-0.640` | `0.804` max-radius | `1.484` | coverage/render |
| teapot coverage/extent probe | `6 -> 430` | failed | `0.882` | `0.794` | `1.512` max-radius | `11.280` | temporal/surface/render |
| teapot app-scale render-proxy continuation with target-aware selection | 8192 particles, app seed, scale `0.72` | `0.617` | `2.200` | `0.998` | `10.780` | rejected: internal seed improved, app seed did not materially improve strict score |
| teapot retime opacity-front gain `1.4` | 8192 particles, app seed, scale `0.72` | `0.623` | `2.174` | `0.954` | `10.786` | rejected: final active count/render regressed; full activation still failed |
| teapot retime opacity-front gain `2.0` | 8192 particles, app seed, scale `0.72` | `0.621` | `2.173` | `0.903` | `10.783` | rejected: surface max fixed but active count/render regressed; full activation still failed |
| teapot retime opacity-front gain `3.0` | 8192 particles, app seed, scale `0.72` | `3.966` | `-5.921` | `0.196` | `37.789` | rejected: opacity exploded and catalog sanity failed |
| torus catalog retimed to local-front opacity | `6 -> 466` | failed | `1.039` | `0.082` | n/a | `12.080` | temporal/coverage/render |
| torus direct-rollout MLP adjoint | 1024 particles, app seed, scale `0.72` | `0.746` | `1.525` | `0.276` | `11.195` | rejected: surface-tail, target coverage, tube angular support, and render gates failed |
| teapot direct-rollout MLP adjoint + opacity continuation | 1024 particles, app seed, scale `0.72` | `0.504` | `3.069` | `0.574` | `0.719` | rejected: target coverage and render gates still failed |
| torus recurrent-state direct adjoint | 1024 particles, app seed, scale `0.72` | `0.914` | `0.802` | low | `11.452` | rejected: temporal activation, target coverage, tube angular support, and render regressed |
| teapot recurrent-state direct adjoint from active BPK | 1024 particles, app seed, scale `0.72` | `0.578` | `2.466` | `0.183` | `1.171` | rejected: target coverage and render gates failed |
| teapot recurrent-state direct adjoint from best diagnostic | 1024 particles, app seed, scale `0.72` | `0.504` | `3.070` | `0.574` | `0.719` | no round selected; unchanged from previous best diagnostic |
| torus fixed-neighborhood SPH state adjoint | 1024 particles, app seed, scale `0.72` | `0.894` | `0.913` | low | `11.488` | rejected: temporal activation, target coverage, tube angular support, and render failed |
| teapot fixed-neighborhood SPH state adjoint, guarded selection | 1024 particles, app seed, scale `0.72` | `0.575` | `2.485` | low | `1.082` | rejected: temporal activation, target coverage, and render failed |
| torus truncated position+SPH state adjoint | 1024 particles, app seed, scale `0.72` | `0.865` | `1.075` | low | `11.454` | rejected: target coverage, tube angular support, and render failed |
| teapot truncated position+SPH state adjoint | 1024 particles, app seed, scale `0.72` | `0.565` | `2.572` | low | `1.147` | rejected: temporal activation, target coverage, and render failed |

The Bevy catalog exposes only the generic 3D preset. The teapot and torus mesh
artifacts remain hidden regression targets because none of the render-gradient,
direct-rollout, stronger-coverage, retiming, or full-context local-batch probes
pass the strict app catalog gate. Render-proxy selection now threads the target
kind into strict scoring and uses active-particle target coverage, so torus
candidates are penalized for missing analytic tube-angle support and dormant
particles cannot mask coverage failures during checkpoint selection. The
full-context fix is still retained because sampled local rows now use perception
features and coverage/front/extent targets computed from the complete rollout
cloud before row extraction, rather than recomputing an artificial neighborhood
on only the sampled subset. This prioritizes correct sparse-core growth dynamics
in the viewer over the older primary-seed render sanity probes. The artifacts
are still not accepted as solved local 3D morphogenesis because coverage and
rendered density remain below the strict gate.

A June 30, 2026 teapot render-proxy retry from the dynamic local-front teapot
artifact selected round `2` without morphology regression and produced the
current `assets/models/teapot_growth_3d.bpk`. Under the opacity-aware app-scale
oracle its app-seed render total is `0.591` with density PSNR `2.368` and
target coverage `0.180`, so it passes the older catalog-sanity gate while
remaining hidden. At the viewer cadence (`96` simulation steps via
`steps/frame=2`) it clears the strict hard local-growth gates but is rejected by
target coverage (`0.273`) and rendered density (`2.487` dB density PSNR).

The validation report now serializes `extent`, a target-bounds versus
final-active-bounds comparison. This catches the current local-front failure
directly: the gated torus probe grows in time but its active cloud spans only
about `0.32` of the torus X extent and `0.49` of the target max radius. The
local rollout trainer also has `front_motion_gate` enabled by default, so
inactive particles away from the active front no longer receive mesh
motion/color/coverage supervision before they become reachable by local
dynamics.

The same report serializes final opacity statistics and the strict gate now
includes `bounded_final_opacity`. This prevents high-opacity artifacts from
being promoted just because their render loss improves; such artifacts can
look like large blended blobs and create unnecessary fragment pressure in the
viewer.

The current conditionless-local ablations fail the mesh gates:

| target | final one-step loss | mesh gate | mean surface | max surface | mean color | max opacity error |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| torus, refreshed 8-step local rows | `7.185` | failed | `2.315` | `4.466` | `0.710` | `61746` |
| teapot, refreshed 8-step local rows | `5.055` | failed | `2.874` | `5.075` | `0.691` | `91907` |
| torus, 32-step local rows | `21737.740` | failed | `2.649` | `4.534` | `0.826` | `237019` |

These reports are kept as ablations, not catalog models. They show that the
current one-step mesh-projection proxy does not produce fully local 3D
morphogenesis from neutral random seeds.

Mesh rollout reports now include bidirectional target coverage in addition to
particle-to-surface distance. The coverage metric samples the target surface,
measures nearest-particle distance for each target sample, and records mean
coverage distance, max coverage distance, and the fraction of target samples
within a scale-normalized coverage radius. This catches partial-surface collapse
cases where particles sit near some mesh surface but do not cover the object.
The strict local-3D report also records growth-style activation metrics:
initial active seed count, final active count, newly activated inactive
particles, newly activated fraction, and the final active-front radius. The
current guard requires the active set to expand beyond the sparse seed core,
which catches static or assigned-seat models that move particles without
propagating 2D-NPA-like growth visibility.
The `ablate-local-3d` trainer exposes `--coverage-gain` for a clamped
target-surface coverage residual; the default is intentionally small (`0.05`)
because early sweeps improve some coverage/render metrics without solving the
strict gate. `--density-gain` is an initializer-only local density-gradient
prior for the conditionless student controller; it is serialized in the rollout
supervision report for reproducibility, but does not alter rollout labels.
Ablation JSON reports serialize the motion, density, coverage, color, opacity,
and update-clamp gains used for the rollout-row objective so candidate artifacts
can be reproduced exactly.

The latest catalog-promotion sweep replaced the older primary-seed render-safe
probes with dynamic local-front artifacts. With the opacity-aware app-facing
render oracle (`0x51a7_3d`, 512 particles, 64 steps, 48 px), the current torus
catalog artifact at seed scale `0.54` scores render loss `0.935` and density
PSNR `0.514`; the current teapot artifact scores render loss `0.983` and
density PSNR `0.147`. Both pass primary-seed temporal activation and temporal
geometry, both remain bounded in opacity, and both still fail the strict
rendered density gate at this low-particle regression scale.

The local 3D student also initializes a local opacity controller, using only the
current opacity state, to avoid the older failure where inactive growth-seed
particles stayed invisible through the rollout. This fixes visibility dynamics
in the local ablation reports, but by itself does not solve target coverage or
rendered density.

The June 30 trajectory-proxy check confirms that the new scaffold is not yet a
solution. At 1024 particles, 64 rollout steps, sparse growth seeds, and strict
multi-seed validation, `target/torus_traj_render_cov005.bpk` moved torus strict
score only from `11.331` to `11.283`, with active target coverage `0.234` and
density PSNR `0.822`; stronger `coverage_gain=0.20` regressed to score
`11.294`. The teapot trajectory proxy moved strict score from `1.184` to
`1.163` at `coverage_gain=0.05`, but active coverage remained `0.203` and
render loss still failed; `coverage_gain=0.20` regressed to score `1.207`.
These artifacts remain diagnostic-only and are not catalog candidates.

The June 30 direct-rollout check improves the backend but still does not close
the 3D morphogenesis gap. With `--training-backend direct-rollout`, final
render position/opacity/color adjoints produced nonzero MLP weight updates and
kept teapot temporal dynamics/local growth valid, but strict validation still
blocked promotion. The best teapot continuation reached render total `0.504`,
density PSNR `3.069`, active coverage `0.574`, and strict score `0.719`; the
best torus probe reached render total `0.746`, density PSNR `1.525`, coverage
`0.276`, and strict score `11.195`, failing tube angular support. No direct
candidate was written into `assets/models`.

The follow-up recurrent-state adjoint first propagated RGB/opacity final-state
gradients backward through the direct current-state feature channel over stored
snapshots. That verified that the direct backend can carry recurrent state
gradients into the MLP weights, but it still did not pass the strict geometry
gate. The teapot active-BPK probe reached strict score `1.171`; continuing from
the previous best diagnostic selected no new round and stayed at score `0.719`.
The torus probe regressed to score `11.452` and kept failing angular coverage.
This confirmed the remaining blocker was not just direct state recurrence.

The next fixed-neighborhood SPH state-perception adjoint extends that recurrence
through direct state, blurred-state, and moment-corrected state-gradient
features. The kernel-level adjoint is finite-difference checked against
`dot(perception_features, feature_adjoint)`. It gives the active teapot BPK a
better first non-regressing diagnostic round than the older direct-only backend,
but guarded strict validation still blocks promotion: teapot scores `1.082`
with render total `0.575` and density PSNR `2.485`; torus scores `11.488` and
still fails angular coverage. Checkpoint selection now refuses rounds with
`selection_morphology_non_regressed=false`, so lower render loss cannot become
the saved diagnostic candidate when it worsens growth morphology.

The truncated position adjoint then carries terminal render/coverage position
gradients backward through direct Euler integration. It improves local render
metrics in short probes, but strict validation remains blocked: teapot scores
`1.147` with render total `0.565` and density PSNR `2.572`; torus scores
`11.454` and still fails tube angular coverage. The large clipped gradient
norms in these probes show that direct position recurrence alone is not a
stable promotion path. The remaining gap is differentiating how positions
change perception, neighbor membership, occlusion/visibility, and rendered
coverage over the rollout.

The next direct-rollout backend pass differentiates fixed-neighborhood SPH
position-perception features as well as state features. Kernel tests now include
finite-difference checks for both non-hybrid and hybrid moment-corrected
position adjoints. Unit gain is too aggressive for the current 3D models, so
`train-render3d` defaults to a conservative `--perception-position-gain 0.05`.
At the same 1024-particle, 64-step, 4-round diagnostic settings, teapot from the
active hidden BPK reaches training render total `0.507` and strict score
`1.070`; torus reaches training render total `0.718` and strict score `11.287`.
Continuing the older best teapot diagnostic improves its training-seed render
total to `0.428`, but robust selection rejects every update for morphology
regression and the strict score remains `0.719`. Opt-in
`--direct-selection-seed-training` averages per-seed clipped direct-rollout SGD
deltas across the selection seed set; in the same short probe it does not improve
strict validation (`teapot=1.077`, `torus=11.311`) and remains an ablation mode,
not the default. No 3D mesh candidate is promoted.

Two rollout-level ablations were then added and measured. `--trajectory-render-gain`
with `--trajectory-render-samples` injects render/coverage adjoints at stored
rollout snapshots instead of only at the terminal trace. In short probes it did
not help strict validation (`teapot gain=0.1 -> strict score 1.105`; `torus
gain=0.05 -> strict score 11.314`). `--full-coverage-adjoint` applies target
coverage pressure to every active particle row rather than only sampled
render-gradient rows. It is now opt-in because it also regressed the same
diagnostic settings (`teapot=1.119`, `torus=11.326`). These failures suggest the
coverage proxy is still too nearest-surface/projection-like unless learned
through a stronger rollout objective and visibility-aware render loss.

The June 30 soft-coverage pass adds `--coverage-mode soft-chamfer`, with
optional `--coverage-repulsion-gain`/`--coverage-repulsion-radius` for
mesh-tangent spread pressure. The mode is intentionally opt-in. At 1024
particles, 64 rollout steps, and 16 rounds, the gentler repulsion probe improves
teapot strict score to `0.854` with active target coverage `0.477`, but still
fails target-coverage fraction and render-density gates. A focused continuation
with `coverage_gain=0.20` and `opacity_gain=2.0` improves the 1024-particle
strict score to `0.825` and active target coverage to `0.501`; at 2048 particles
it reaches active target coverage `0.769` and only fails render loss
(`density_psnr=2.924`, strict score `0.708`). Teapot is therefore partly
particle-budget limited but still lacks a strong render objective. Torus improves from
soft-only strict score `11.239` to `11.161` with gentle tangent repulsion, but
tube angular coverage remains `6/16` bins at both 1024 and 2048 particles.
Stronger spread pressure (`coverage_gain=0.20`, `coverage_repulsion_gain=0.20`)
regresses torus strict score to `11.299`. No soft-coverage artifact is promoted.

The next pass fixes a target-definition bug: `TriangleMeshTarget::surface_sample`
now uses area-weighted low-discrepancy triangle samples instead of face-prefix
centroids. This makes low sample counts cover the whole mesh and invalidates
older prefix-biased render/coverage numbers. Under the corrected sampler, the
best teapot continuation reaches target coverage `0.980`, render total `0.364`,
density PSNR `4.492`, and strict score `0.551` at 1024 particles; a
density-weighted continuation reaches density PSNR `4.727` and strict score
`0.527` at 2048 particles, but still fails the render gate. The old torus
continuation regresses under the corrected full-torus target (`coverage=0.097`,
strict score `21.747` at 1024 particles), and a corrected-sampler torus retrain
still fails (`strict score 21.721`). No corrected-sampler artifact is promoted.

The latest local 3D diagnostics add area-weighted random mesh surface sampling,
normal-aware soft coverage, Sliced OT coverage, and alpha-only retiming. None
produce a promotable 3D artifact:

| probe | strict score | render total | density PSNR | target coverage | tube bins | promoted |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| torus active hidden asset, 1024 particles | `11.502` | `1.123` | `-0.414` | `0.140` | `7/16` | no |
| torus full-coverage soft render adjoint | `11.497` | `1.106` | `-0.346` | `0.138` | `7/16` | no |
| torus normal-aware soft render adjoint | `11.497` | `1.107` | `-0.350` | `0.138` | `7/16` | no |
| torus Sliced OT terminal render adjoint | `11.506` | `1.130` | `-0.443` | low | n/a | no |
| torus Sliced OT local rollout continuation | `11.535` | `1.186` | `-0.658` | low | n/a | no |
| torus alpha-only `1.1` | `11.454` | `1.044` | `-0.082` | `0.167` | `9/16` | no |
| torus alpha/front continuation best probe | `11.351` | `0.853` | `0.819` | `0.167` | `9/16` | no |
| teapot alpha-only `1.05` | `0.548` | `0.363` | `4.518` | passes | n/a | no |

The particle-count sweep also argues against a simple particle-count-only
failure: the active torus asset improves to strict score `11.467` at 2048
particles and `11.411` at 4096 particles, but still covers only `8/16` tube
bins and fails target coverage/render. The current blocker is therefore the
learned local 3D growth objective and support allocation, not just sample count.

The same ablation reports now include multi-view render loss:

| target | render gate | render total | density PSNR | color PSNR | depth PSNR |
| --- | --- | ---: | ---: | ---: | ---: |
| torus local ablation | failed | `1.492` | `-0.107` | `7.351` | `5.476` |
| teapot local ablation | failed | `1.321` | `-0.046` | `8.399` | `7.813` |

The retired position-field artifacts also fail the stricter rendered density
gate (`uv_torus_morphogen_3d`: density PSNR `-6.773`; teapot: density PSNR
`0.588`). Those BPKs are no longer catalog assets. Truly general 3D
morphogenesis still needs a learned condition/initializer or minimal seed frame
plus rendered/geometry rollout loss rather than a hand-authored absolute
position field. See
`docs/local_3d_morphogenesis.md` for the current ablation plan and acceptance
gate.

## Benchmarks

The CLI benchmark command reports elapsed CPU rollout time and per-phase timings:

```bash
cargo run --release -p burn_automata --bin burn_automata -- bench --preset growing-2d --particles 4096 --steps 1 --profile
PRESET=growing-3d-gs PARTICLES=16384 STEPS=1 scripts/bench_rollout.sh
cargo run --release -p burn_automata --features gpu_wgpu --bin burn_automata -- bench --preset texture-2d --particles 4096 --steps 2 --gpu
cargo run --release -p burn_automata --features gpu_wgpu --bin burn_automata -- bench --preset texture-2d --particles 4096 --steps 16 --gpu --neighbor-mode auto
cargo run --release -p burn_automata --features gpu_wgpu --bin burn_automata -- bench --preset texture-2d --particles 4096 --steps 16 --gpu --neighbor-mode linked-list
cargo run --release -p burn_automata --features gpu_wgpu --bin burn_automata -- bench --preset growing-3d-gs --particles 8192 --steps 30 --gpu --geometry line --seed-mode torus-morphogen-dense-3d --seed-scale 0.04 --normalize-seed-scale --neighbor-mode sorted
cargo run --release -p burn_automata --features gpu_wgpu --bin burn_automata -- bench --preset growing-3d-gs --particles 8192 --steps 30 --gpu --geometry line --seed-mode torus-morphogen-dense-3d --seed-scale 0.04 --normalize-seed-scale --neighbor-mode tiled --bucket-capacity 256
cargo run --release -p burn_automata --features gpu_wgpu --bin burn_automata -- bench --preset growing-3d-gs --particles 16384 --steps 16 --gpu --gaussian
scripts/bench_gpu_matrix.py --matrix quick --output target/bench_gpu_matrix_final.json
```

Measured locally on the current ARM workstation:

| preset | particles | exact CPU time |
| --- | ---: | ---: |
| `growing-2d` | 4096 | ~14 ms/step |
| `texture-2d` | 4096 | ~6 ms/step |
| `growing-3d-gs` | 16384 | ~21 ms/step |
| `growing-2d` dense stress | 16384 | ~165 ms/step |

The direct WGPU path is validated against the same CPU oracle. The resident-state benchmark keeps rollout buffers on GPU, times submit/wait separately from final readback, and reports `grid_overflow_count`. Treat any nonzero fixed-bucket overflow as numerically invalid, even if it is faster. `--neighbor-mode auto` resolves normal particle grids to linked-list traversal, but switches high initial occupancy particle-grid starts to fixed buckets with adaptive headroom. `--neighbor-mode tiled` runs the active-cell/shared-memory tiled fixed-bucket kernel; `--neighbor-mode sorted` runs the exact overflow-free prefix-sum cell layout. Both remain opt-in because local measurements show scalar fixed buckets are currently faster when capacity is sufficient. Fixed buckets clear only cell counters and overflow, not the full slot storage. Local CLI timings for exact auto mode:

Seed-scale changes on scale-equivariant particle-hash grids now scale the
effective hashgrid `eps` by `seed_scale / reference_seed_scale` in the Bevy
runtime. The CLI benchmark exposes the same policy with
`--normalize-seed-scale`. This keeps normalized occupancy stable for the seed
radius slider while preserving the physical fixed-`eps` mode for stress tests.
The WGPU regression
`wgpu_normalized_seed_scale_preserves_3d_torus_morphogen_rollout` verifies the
scaled 3D torus rollout against the reference-scale rollout.

For dense 3D growth seeds, the previous 8192-particle viewer default was not
interactive enough on the current NVIDIA GB10 Vulkan adapter. The catalog now
defaults 3D entries to `1024` particles, and WGPU `auto` resolves dense 3D
particle-grid starts to exact sorted cells for `<=2048` particles. A focused
teapot-growth benchmark with gaussian buffer writes measured:

| particles | old/equivalent mode | avg step | current auto mode | avg step |
| ---: | --- | ---: | --- | ---: |
| `1024` | tiled fixed buckets | `77.4 ms` | sorted cells | `18.5 ms` |
| `2048` | tiled fixed buckets | `94.5 ms` | sorted cells | `65.4 ms` |
| `4096` | tiled fixed buckets | `92.0 ms` | tiled fixed buckets | `92.0 ms` |

The 1024-particle headless Bevy + gaussian-splatting pipeline benchmark reports
`median=18.1 ms`, `p95=26.7 ms`, `max=34.6 ms`, and `55.2 FPS` median with
`effective_neighbor=sorted-cells`. Sorted cells are not forced above 2048
particles because the same dense-seed workload regresses at 4096/8192.

| preset | particles | auto neighbor mode | WGPU time |
| --- | ---: | --- | ---: |
| `growing-3d-gs` | 4096 | linked-list | ~9.1 ms/step |
| `growing-3d-gs` | 16384 | linked-list | ~12.6 ms/step |
| `growing-3d-gs` | 32768 | linked-list | ~31.9 ms/step |
| `growing-3d-gs` | 65536 | fixed buckets, cap 64 | ~82.9 ms/step |
| `texture-2d` | 4096 | linked-list | ~16.0 ms/step |
| `texture-2d` | 16384 | linked-list | ~60.6 ms/step |
| `texture-2d` | 32768 | linked-list | ~230.1 ms/step |
| `point-mnist` | 4096 | fixed buckets, cap 512 | ~11.6 ms/step |
| `point-mnist` | 16384 | fixed buckets, cap 2048 | ~31.2 ms/step |
| `growing-2d` dense | 4096 | fixed buckets, cap 512 | ~55.8 ms/step |
| `growing-2d` dense | 8192 | fixed buckets, cap 1024 | ~115.2 ms/step |

High-particle stress results are saved in `target/bench_gpu_high_particles.json`: exact 3D remains usable at 65k particles on this machine, while exact dense 2D degrades to ~940 ms/step at 32k and ~3.6 s/step at 65k. The dense growing case has a very high local neighbor count. The exact GPU traversal preserves parity but is not the final throughput kernel for that workload. Persistent no-readback rollout state is covered by tests; the next throughput target is a sorted/prefix-sum or cooperative-neighbor traversal compared against the same fixed-seed CPU oracle.

## Import Parity

After importing upstream checkpoints, run:

```bash
python3 scripts/validate_import_parity.py --model /tmp/burn_automata_lizard.bpk --particles 64 --preset growing-2d --seed-scale 0.2
python3 scripts/validate_import_parity.py --model /tmp/burn_automata_polka.bpk --particles 64 --preset texture-2d --seed-scale 1.0
python3 scripts/validate_import_parity.py --model /tmp/burn_automata_lizard.bpk --particles 64 --preset growing-2d --seed-scale 0.2 --gpu --steps 4 --psnr-threshold 70 --hidden-psnr-threshold 70
python3 scripts/validate_import_parity.py --model /tmp/burn_automata_polka.bpk --particles 64 --preset texture-2d --seed-scale 1.0 --gpu --steps 4 --psnr-threshold 70 --hidden-psnr-threshold 70
```

The script runs deterministic Rust inference for zero or more steps, then compares the rollout with a dependency-free Python implementation of the same SPH, moment correction, MLP, and Euler update formulas. It reports max position/state errors, position PSNR, hidden-state PSNR, tail-RGB PSNR, and a deterministic 2D Gaussian-image PSNR. `scripts/validate_gpu_e2e.sh` combines the WGPU tests, Bevy planar gaussian linkage harness, and imported-model WGPU PSNR checks when the BPK files are available.

The project page web demo publishes many additional trained 2D models as packed JSON tensors. Import and validate the curated Bevy catalog with:

```bash
git clone --depth 1 https://github.com/SelfOrg-NPA/SelfOrg-NPA.github.io /tmp/selforg_npa_web
python3 scripts/import_selforg_catalog.py --web-root /tmp/selforg_npa_web
python3 scripts/validate_catalog_parity.py --web-root /tmp/selforg_npa_web --gpu --build-binary --require-all
```

The current catalog importer supports the scalar MLP growing/texture web models. The separate equivariant web-demo JSON contains extra `W_hidden`/`W_out` tensors and a different update layout; the importer rejects those explicitly until that architecture has a first-class BPK representation.

Deterministic parity commands force `--update-prob 1.0`; this is intentional. The upstream growing demo and viewer default use stochastic `update_prob = 0.5`, which changes the rollout trajectory by design. WGPU stochastic masking is covered by the GPU test suite, including an exact `update_prob = 0.0` no-op test.

Latest local WGPU imported-model parity over 4 steps at 64 particles:

| set | entries | max position error | max state error | min hidden PSNR | min rendered Gaussian PSNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| curated web/BPK catalog | `23` | `1.9e-5` | `1.1e-4` | `101.5 dB` | `107.2 dB` |
| `/tmp/burn_automata_lizard.bpk` | `1` | `2.3e-7` | `2.5e-6` | `132.7 dB` | `142.9 dB` |
| `/tmp/burn_automata_polka.bpk` | `1` | `4.6e-6` | `6.1e-5` | `105.8 dB` | `111.4 dB` |
