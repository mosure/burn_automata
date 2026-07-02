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
`validate-growth3d` gates pass at the app-facing 1024-particle 64-step catalog
horizon and 96-step viewer horizon. Every `train-render3d` report also contains
both a compact `strict_gate_summary` and the full `growth_validation` section
generated from the saved BPK with the same strict gate, training seed,
`--selection-seed`, and `--extra-selection-seed` set, so diagnostic artifacts
carry the runtime-dynamics blockers that prevent catalog promotion.

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
defaults to `--training-backend direct-rollout` and
`--weight-update-mode adapter`. Direct or proxy objectives first compute the
same full MLP gradient used by legacy full-weight training, then project that
gradient into a LoRA-style `NpaLowRankAdapter`; the shared base weights are
kept frozen during the update. Reports record adapter rank, alpha, seed,
adapter parameter count, base parameter count, and materialized parameter
count. The saved BPK remains a materialized adapted model for viewer/catalog
compatibility, so promotion gates validate the exact weights the app loads.
Use `--weight-update-mode full` for legacy full-model ablations only.

The direct backend uses analytic CPU gradients from render loss to final
particle positions, opacity, and color, then applies those adjoints through
stored rollout MLP outputs and a fixed-neighborhood SPH state-perception
adjoint. The older supervised projection path is still available as
`--training-backend proxy`, with finite differences left as an explicit
regression fallback. The direct backend is
still not true BPTT: direct Euler position integration is differentiated, but
position-dependent perception, density-gradient position terms, future neighbor
geometry, and rendering through time are treated as stop-gradient. The
terminal and trajectory surface adjoints now share `--surface-escape-gain` with
the proxy backend, boosting active particles that move beyond the strict
surface max-distance threshold so tail failures are optimized instead of only
reported. The same escape path also adds a bounded visibility-state barrier for
escaped active particles: it can suppress liveness/material opacity after the
local-front growth term, leaving near-surface and dormant particles untouched.
Direct-rollout training also includes a bounded local-front
`--liveness-gain` adjoint on strict liveness channel `state[3]`; this gives
temporal activation failures a training signal while preserving locality by
leaving far dormant particles untouched. The same liveness path now also has a
permissive temporal activation schedule: if a sampled rollout snapshot is
already too active for its rollout fraction, only the weakest active rows
receive a bounded positive liveness adjoint, slowing over-fast all-particle
activation without adding particle-index targets. If a sampled snapshot is
under the progressive lower-bound schedule, only dormant local-front candidates
receive bounded negative liveness adjoints, pushing growth outward without
globally activating far particles or assigning per-index targets. Snapshot
liveness uses the
`--trajectory-render-samples` schedule but is independent of
`--trajectory-render-gain`, so temporal activation can be trained even in
render-terminal-only ablations. Snapshot mesh coverage/surface adjoints are
likewise controlled by `--trajectory-mesh-gain`, so rollout-level geometry can
be trained without enabling intermediate image-render gradients. The CPU
multiview render adjoint now includes the density-gated color/depth
normalization terms, and `render_loss` unit tests compare the full
density/color/depth position, opacity, color-state, and learned-scale gradients
against finite differences. The `train-render3d --gradient-mode finite-diff`
fallback now also emits color-state gradients instead of silently optimizing
only position/opacity/scale channels. For `--gaussian-decode-mode learned-sh0`,
direct-rollout training also applies a trajectory-time learned-scale budget
objective to predicted scale updates, and `direct_objective_diagnostics`
serializes `scale_budget_rms` plus `scale_post_cap_rms` so learned-scale 3DGS
experiments can verify scale pressure is present before catalog promotion. The
remaining backend gap is differentiable or WGPU training through the full
rollout dynamics.

Mesh-motion and extent-front liveness are separate direct-rollout objective
channels. Dormant particles receive bounded liveness-output pressure only when
they are inside the local growth front and either the mesh objective assigns
them nonzero motion pressure or they expand the current active bounds toward
remaining target-side extent. Reports expose these as
`direct_objective_diagnostics.mesh_motion_liveness_rms`,
`mesh_motion_liveness_nonzero_fraction`,
`extent_front_liveness_rms`, and
`extent_front_liveness_nonzero_fraction`. It also emits
`extent_front_motion_rms` and `extent_front_motion_nonzero_fraction` for the
paired outward motion target on dormant local-front rows that can expand active
bounds. Material-visible coverage now has a matching local-front liveness
candidate path: active visible material represents current support, dormant
local-front rows compete as low-weight potential material support, and the
resulting coverage-update magnitudes feed liveness and materialization
objectives. Reports expose this as
`material_coverage_liveness_rms` and
`material_coverage_liveness_nonzero_fraction`. These channels couple
activation, motion, and material visibility to generic rollout mesh/extent
progress without assigning particles to fixed target seats, and they remain
subject to the same strict growth/render gates.

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
target-surface coverage, target-normal-bin coverage, temporal geometry
progress, local-front coherence, a rollout motion profile, and the same
multiview render loss. The motion profile
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
3D morphogenesis models. Strict validation now also rejects seed-frame
coordinate scaffolding via `no_seed_coordinate_scaffold`; coordinate-scaffold
artifacts remain hidden diagnostics until a truly local conditionless seed path
passes the same growth/render checks. Current catalog artifacts intentionally
fail that strict gate on scaffold, target coverage, and rendered density.

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
active-surface max, target coverage, material-visible coverage/liveness,
surface-profile/normal support, material-visible surface tails, active count,
newly activated fraction, local-front activation, or temporal
activation/geometry progress relative to the initial model. Those rounds are
not selected even when render loss improves. Strict-score selection now includes
the active-surface max-distance penalty and can spend only a small bounded
render/density slack when the worst-case strict score improves materially;
otherwise render loss and density PSNR must be non-regressing. Each history row
also records `before_selection_loss`, `before_selection_score`, and
`before_selection_density_psnr_db`, plus
`before_selection_min_active_extent_bbox_ratio` and
`before_selection_min_active_extent_min_axis_ratio`. Reports therefore expose
the strict rollout objective and active-extent progress before and after the
round instead of only the render-only `before_loss`/`after_loss` pair. Each
history row also records `selection_worst_seed` and
`selection_worst_failure_reasons`, which are the held-out seed and strict-check
blockers that currently dominate promotion selection.
Catalog-bound `train-render3d` runs stage a temporary validation candidate
instead of writing directly to `assets/models`. The JSON report is written even
when promotion validation fails, and it includes `catalog_promotion` with
`requested`, `validation_count`, `validation_passed`, and `rejection_reason`,
plus the full `catalog_promotion_validations` list. This keeps failed torus or
teapot promotion attempts auditable while preserving the no-overwrite guard for
strict-failing candidates.
It also records `selection_min_final_active_count`,
`selection_min_newly_activated_fraction`,
`selection_min_front_local_newly_activated_fraction`,
and a flat `strict_gate_summary` with the same coverage, surface-tail,
surface-normal, gaussian scale-budget, render, and temporal-growth metrics used
by strict validation. This makes failed torus/teapot promotion reports directly
sortable by the active blockers instead of requiring a nested report parser.
`selection_max_front_liveness_margin`,
`selection_min_front_liveness_candidate_count`,
`selection_max_extent_front_liveness_margin`,
`selection_min_extent_front_liveness_candidate_count`,
`selection_max_temporal_extent_front_liveness_margin`,
`selection_min_temporal_extent_front_liveness_candidate_count`,
`selection_material_visible_inactive_fraction`,
`selection_material_visible_max_inactive_opacity`,
`selection_all_temporal_activation_progressive`, and
`selection_all_temporal_geometry_progressive`. Treat these as per-round
morphogenesis diagnostics: a lower render loss is not sufficient if the selected
checkpoint loses active-front growth, temporal geometry progression, or
material-visible liveness on any selection seed.
`selection_max_front_liveness_margin` is a continuous pre-activation metric for
dormant particles inside the local active front. It lets guarded line search
retain sub-threshold local liveness improvements before
`newly_activated_fraction` flips, without relaxing the strict promotion gate.
`selection_max_extent_front_liveness_margin` is the same activation-margin
diagnostic restricted to dormant local-front rows that can expand active bounds
toward target mesh bounds. It lets line search retain bounded active-extent
precursor progress before `active_extent_growth` flips, while still rejecting
catalog promotion until strict active-extent gates pass.
The temporal extent-front fields apply the same target-bounds-aware margin to
intermediate rollout samples, so selection can retain staged extent-growth
pressure without accepting a final-snapshot extent collapse.

For direct-rollout training, history rows also serialize
`train_initial_loss`, `train_final_loss`, `train_best_loss`,
`train_step_count`, `train_loss_history`, `train_grad_norm_history`, and
`train_grad_scale_history`. The direct backend reruns the updated model on the
same rollout seed after each gradient step, so `train_final_loss` is a measured
post-update mesh/render objective rather than a copy of the pre-step loss.
`train_loss_history.len()` should match `train_step_count`, which should match
`supervised_steps_per_round` for a complete round. `supervised_loss` remains as
the backward-compatible alias for `train_final_loss`. Each round also records
`selected_checkpoint` and `rolled_back_to_best_checkpoint`; rejected rounds are
rolled back before the next rollout so later rounds continue from the best
strict-scored checkpoint rather than from a morphology-regressed update.
When direct selection-seed training is enabled, each inner step accumulates
per-seed deltas in the active update parameterization: low-rank adapter deltas
for adapter mode, full-weight deltas for legacy full mode. It then applies the
average and measures `train_final_loss` by rerunning the actual averaged model
over the training seed set. The reported loss is therefore the retained
multiseed update, not the mean of independent per-seed candidate models.
Direct-rollout training now defaults to strict-score line search over the
configured `direct_line_search_scales`; a history `train_step_scale` of `0`
means the no-op candidate was retained because no step improved the guarded
multi-seed mesh/render selection objective. Use
`--direct-line-search=false` only for single-step ablations or repro runs.
Selection scoring includes a small continuous local-front liveness-margin
penalty. Render/density slack still requires a larger strict-score improvement,
so this precursor can select margin-improving, render-non-regressing growth
updates but cannot by itself promote or spend meaningful render loss.
On a 32-particle, 4-step teapot probe, the previous guarded line search retained
no-op (`train_step_scale=0`) even though raw direct gradients moved the liveness
output row. With the margin metric, a one-step guarded run selected the scale
`32` candidate, reduced the worst local-front liveness margin from `5.686` to
`4.730`, and recorded `train_liveness_output_delta_norm=0.00158`, while strict
validation still failed with `newly_activated_fraction=0.0`. A four-inner-step
probe constrained to scales up to `8` retained a render-non-regressing
checkpoint with the same margin improvement; it also remained strict-failing.
This is useful optimizer progress toward local growth, not catalog evidence.
Material-visible suppression is also explicit: `train-render3d` reports
`material_suppression_update_multiplier` and accepts
`--material-suppression-update-multiplier` to let dormant/off-surface material
opacity use a larger negative cap than positive material-growth updates. This is
an ablation knob, not a promotion shortcut. On the short 256-particle teapot
probe (`24` rollout steps, seeds `42,99`, `direct_line_search_scales=0.25,0.5,1`)
the multiplier changed the guarded score but did not resolve the strict
morphogenesis blockers:

| material suppression multiplier | render total | density PSNR | selection strict score | inactive visible material | visible target coverage | promoted |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `1` | `0.04899` | `15.053` | `49.733` | `0.754` | `0.459` | no |
| `5` default | `0.04905` | `15.045` | `49.715` | `0.750` | `0.463` | no |
| `10` | `0.04905` | `15.044` | `49.712` | `0.750` | `0.459` | no |

The dominant failures remain newly activated fraction, temporal activation
progression, target/material-visible target coverage, surface normal support,
material-visible liveness, and surface-profile coverage.
After adding the temporal lower-bound adjoint, the same short teapot probe
reached render total `0.04901`, density PSNR `15.048`, and final strict score
`49.703`; a stronger `--liveness-gain 0.2` improved render total to `0.04880`
and density PSNR to `15.075`, but the worst selection seed still had
`selection_min_newly_activated_fraction=0.233` and
`selection_all_temporal_activation_progressive=false`. A matched short torus
probe remained geometry/render-limited (`selection_score=56.343`,
`density_psnr=0.358`) rather than material-liveness limited. These measurements
keep the lower-bound signal as part of rollout-level training, but they are not
promotion evidence.
The material-visible liveness path now has a paired signal: dormant
material-visible particles near the target surface receive liveness activation
pressure, while off-surface material-visible particles remain handled by
material-tail suppression. This is wired in both direct adjoints and proxy
target updates. On the same short teapot fixture, the default gain improved the
render total to `0.04882` and density PSNR to `15.073`, but
`selection_material_visible_inactive_fraction` remained `0.750`; a
`--material-liveness-gain 0.5` ablation was rejected by guarded line search
(`train_step_scale=0`) and retained the baseline strict score. The path is
therefore validated as a local consistency signal, not as sufficient promotion
evidence.
The serialized objective records `trajectory_mesh_gain`, `liveness_gain`,
`liveness_front_radius`, and `material_suppression_update_multiplier` alongside
render, coverage, surface, opacity, and scale weights.
`trajectory_render_samples=0` disables only intermediate render-loss adjoints;
mesh and liveness trajectory adjoints still sample the rollout with a bounded
fallback budget when their gains are non-zero, so non-render morphogenesis
supervision cannot be silently disabled by a render-only knob.

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
also below the old `catalog-sanity` render-density floor in the latest
app-scale report, and the visible-catalog policy uses the stricter blockers so
this partial cloud is not presented as a coherent teapot:

| model | particles | seed scale | render total | density PSNR | color PSNR | depth PSNR | target coverage | catalog gate | strict blockers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `uv_torus_growth_3d.bpk` | `1024` | `0.54` | `1.011` | `0.284` | `12.542` | `17.302` | `0.145` | failed | temporal activation, surface max, coverage max/fraction, surface-normal coverage, torus angular coverage, render |
| `teapot_growth_3d.bpk` | `1024` | `0.72` | `0.186` | `8.044` | `15.629` | `28.189` | `0.924` | failed | temporal activation, surface max, render |

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
angular coverage, surface-normal-bin coverage, active/material-visible surface
profile coverage, material-visible liveness, material-visible target/normal
support, material-visible surface-tail bounds, seed-perturbation stability, or
final opacity. Add `--require-catalog-safe` when a candidate is intended to
become visible in the Bevy catalog; that mode requires `strict_passed`,
`strict_checks.passed`, robust `all_strict_passed`, robust
`all_surface_normal_coverage`, robust `all_surface_coverage_profile`, robust
`all_material_visible_surface_coverage_profile`, robust
`all_material_visible_particles_live`, robust
`all_material_visible_surface_normal_coverage`, and robust
`all_material_visible_surface_tail_bounded`, not just the older
`catalog-sanity` threshold.

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
| `uv_torus_growth_3d.bpk` | `1024` | `96` | `1.023` | `0.266` | `0.154` | failed | `22.032` | temporal activation, surface mean/max/tail, coverage max/fraction, surface-normal coverage, torus angular coverage, render |
| `teapot_growth_3d.bpk` | `1024` | `96` | `0.177` | `8.436` | `0.910` | failed | `2.102` | surface max/tail, render |

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
zero means the strict gate passed. It includes active-surface max, tail, target
coverage, surface-normal coverage, material-visible liveness/tail penalties,
render-density, and scale-budget penalties so training selection is aligned
with the same blockers that prevent catalog promotion. Current strict
failure reasons with active-surface validation:

| model | strict failure reasons |
| --- | --- |
| `uv_torus_growth_3d.bpk` | `temporal_activation_progressive`, `surface_max_bounded`, `target_coverage_max_bounded`, `target_coverage_fraction`, `surface_normal_coverage`, `torus_angular_coverage`, `render_loss_passed` |
| `teapot_growth_3d.bpk` | `temporal_activation_progressive`, `surface_max_bounded`, `render_loss_passed` |

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

On 2026-07-01 the direct-rollout path added an early-biased liveness trajectory
sampler and a continuous temporal activation schedule/burst penalty. The
penalty is now serialized in `strict_score.temporal_activation_schedule_error`
and `strict_score.temporal_activation_schedule_penalty`, and
`scripts/validate_3d_catalog.py` includes both fields in summaries and finite
checks. Short 32-particle, 4-step line-search diagnostics show the scalar is
catching the remaining non-morphogenetic failure: candidates still defer
activation to the final sample instead of growing progressively, so they remain
diagnostics only.

| candidate | active timing | temporal error | temporal penalty | strict score | strict blockers |
| --- | --- | ---: | ---: | ---: | --- |
| teapot temporal strict-score probe | `1 -> 1 -> 1 -> 23` active at steps `0/1/2/4` | `0.199` | `4.987` | `81.473` | local front, temporal activation/geometry, coverage, normals, render |
| torus temporal strict-score probe | `1 -> 1 -> 1 -> 29` active at steps `0/1/2/4` | `0.280` | `6.996` | `93.553` | local front, temporal activation/geometry, coverage, normals, torus angular, render |

The next 2026-07-01 pass added three training-signal fixes without relaxing
promotion gates:

- direct-rollout MLP backward now caps the liveness output channel with the
  configured liveness update ceiling instead of always using the smaller render
  channel RMS cap;
- adjacent liveness trajectory samples add a burst-retiming adjoint, so rows
  that appear in an over-fast late activation burst are also trained at the
  previous local front;
- `local_front_weights` adapts the training front radius to the nearest dormant
  shell in sparse clouds, while keeping far dormant particles outside the front.

Regressions cover all three behaviors:
`output_gradient_liveness_cap_preserves_sparse_temporal_signal`,
`temporal_activation_jump_adjoint_retimes_late_burst_to_previous_front`, and
`local_front_weights_adapt_to_sparse_dormant_shell_without_global_front`.
However, compact end-to-end probes still fail strict morphogenesis gates:

| probe | active timing | temporal error | temporal penalty | strict score | result |
| --- | --- | ---: | ---: | ---: | --- |
| teapot explicit growth seed, liveness cap + burst adjoint | `1 -> 1 -> 1 -> 23` | `0.199` | `4.987` | `81.473` | unchanged late-burst failure |
| torus explicit growth seed, liveness cap + burst adjoint | `1 -> 1 -> 1 -> 29` | `0.280` | `6.996` | `93.551` | unchanged late-burst failure |
| teapot no-line-search ablation | `1 -> 1 -> 1 -> 32` | `0.341` | `8.536` | `85.062` | render loss improves, but activation shortcut worsens |
| torus no-line-search ablation | `1 -> 1 -> 1 -> 32` | `0.341` | `8.536` | `95.049` | render loss improves, but activation shortcut worsens |
| teapot substrate-seed default ablation | `1 -> 1 -> 1 -> 1` | `0.228` | `5.692` | `132.771` | stalls; no active-front expansion |
| torus substrate-seed default ablation | `1 -> 1 -> 1 -> 1` | `0.228` | `5.692` | `143.170` | stalls; no active-front expansion |

The substrate ablation was not kept as the no-base default because it regressed
the compact strict probes. It remains an explicit seed mode for larger-domain
experiments. The current blocker is therefore sharper: gradients reach the
liveness output and local-front candidate selection is tested, but the selected
short-horizon rollout objective still chooses either a late activation burst or
no activation. The next backend change should optimize a multi-step activation
state objective directly, rather than relying on terminal render selection to
discover progressive liveness timing.

A follow-up 2026-07-01 pass added direct temporal liveness supervision on the
MLP output and made render-proxy selection track the worst temporal activation
schedule error for each candidate. Activation-breakthrough and post-activation
refinement retention now require temporal activation schedule error to stay
within `TEMPORAL_ACTIVATION_SELECTION_REGRESSION_SLACK` unless the candidate is
already temporally progressive. This is covered by
`temporal_liveness_output_objective_boosts_underactive_local_front`,
`temporal_liveness_output_objective_suppresses_newly_predicted_burst_rows`,
`render_selection_rejects_bursty_activation_breakthrough_timing_regression`,
and `render_selection_rejects_post_activation_refinement_timing_regression`.
Training reports now serialize
`selection_max_temporal_activation_schedule_error` so line-search choices can
be audited.

The compact probes show the guard is useful but not sufficient:

| probe | active timing | temporal error | temporal penalty | strict score | result |
| --- | --- | ---: | ---: | ---: | --- |
| teapot direct liveness-output, narrow scale search | `1 -> 1 -> 1 -> 22` | `0.186` | `4.652` | `81.158` | slight timing gain, still late-burst |
| torus direct liveness-output, narrow scale search | `1 -> 1 -> 1 -> 29` | `0.280` | `6.996` | `93.550` | essentially unchanged late-burst failure |
| teapot guarded default-scale probe | `1 -> 1 -> 12 -> 32` | `0.102` | `2.558` | `69.082` | better timing and score, still fails temporal/geometry/render gates |
| torus guarded default-scale probe | `1 -> 1 -> 1 -> 19` | `0.147` | `3.683` | `90.250` | marginal timing gain, still late-burst |
| teapot quadratic temporal schedule | `1 -> 1 -> 10 -> 32` | `0.028` | `0.694` | `67.218` | local-growth-compatible timing improves, still fails geometry/material/render gates |
| torus quadratic temporal schedule | `1 -> 1 -> 9 -> 32` | `0.021` | `0.525` | `77.042` | large timing/score improvement, still fails coverage/normal/material/render gates |
| teapot material-cap split | `1 -> 1 -> 10 -> 32` | `0.028` | `0.694` | `67.218` | render loss improves to `0.918`, but visible-material coverage remains `0.0` |
| torus material-cap split | `1 -> 1 -> 9 -> 32` | `0.021` | `0.525` | `77.041` | render loss improves to `0.919`, but target coverage remains `0.0625` and visible-material coverage remains `0.0` |
| torus direct mesh-output objective | `1 -> 1 -> 9 -> 32` | `0.021` | `0.525` | `77.041` | rollout-level mesh output is active but small; one-round `train_motion_output_delta_norm=0.00069`, target coverage unchanged |

The default direct line-search scale range already covers the guarded
high-scale teapot checkpoint. Six-round compact probes plateau after the first
selected checkpoint, which confirms the remaining blocker is not just optimizer
step magnitude. The next backend change should add phase-aware recurrent state
or rollout-level liveness targets that can continue training after the first
activation breakthrough, while strict selection continues to reject
timing-regressive render shortcuts.
Tightening the burst-retiming slack to `0.05` was not retained: it moved the
compact teapot path to `1 -> 1 -> 16 -> 32`, but worsened strict score to
`70.332`, regressed torus to `90.928`, and stronger liveness-gain ablations
fell back to late-burst timing (`1 -> 1 -> 1 -> 21`, strict score `81.538`).
That historical pass used a quadratic temporal target to avoid non-local
first-step activation from a sparse seed. A later selector pass superseded it
with the current linear half-rollout-aligned target plus stricter burst
rejection; see the 2026-07-01 direct-front section below for the current
compact reports. Neither schedule produced catalog evidence.
Material opacity training also has a soft assignment radius at three times the
strict target-coverage threshold, plus a material-specific
`--material-max-opacity-update` cap that defaults to `0.75` instead of sharing
the smaller liveness/opacity cap. Strict validation still uses the original
coverage threshold; the soft radius and larger material cap only let
approaching active particles become render-visible soon enough to receive
material/render gradients. In the four-step compact probes this unit-tested
path doubles the material-output weight delta and slightly improves render
loss, but it still does not change strict score because geometry target
coverage remains too sparse for material-visible coverage to become nonzero.
Direct-rollout training now also applies the same target-generic coverage and
surface-projection updates directly to trajectory-snapshot motion outputs and
records `train_motion_output_delta_norm`. This is a correct rollout-level mesh
signal, but compact probes show the retained motion-output update is still much
smaller than the liveness update and does not yet solve surface coverage.

`train-render3d` now defaults diagnostic output to `artifacts/` and rejects
`assets/models/*` output paths unless the base model has conditionless-local
lineage and the seed mode is the matching local growth seed. Catalog-bound
render training is then promoted only if the temporary candidate passes the
same app-scale strict seed sweep used by `scripts/validate_3d_catalog.py`. This
keeps field-baseline, position-field, or small training-horizon render-proxy
experiments from being written directly into the catalog asset directory.
Legacy mesh trainers, `ablate-local-3d`, and `retime-growth3d` refuse
catalog-bound output paths entirely; their BPKs must be written to `target/` or
`artifacts/` and then promoted only through the app-scale validation harness.
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
final-active-bounds comparison. Strict validation includes
`active_extent_growth`, which requires both a minimum active bounding-box
diagonal ratio and a minimum per-axis extent ratio. This catches local-front
failures that activate or move particles without occupying target-scale 3D
space: the gated torus probe grows in time but its active cloud spans only about
`0.32` of the torus X extent and `0.49` of the target max radius. The local
rollout trainer also has `front_motion_gate` enabled by default, so inactive
particles away from the active front no longer receive mesh
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
regression and the strict score remains `0.719`. Direct-rollout training now
averages per-seed clipped SGD deltas across the selection seed set by default;
`--no-direct-selection-seed-training` keeps the older single-seed path available
as an ablation. Earlier short probes of this multi-seed update did not improve
strict validation (`teapot=1.077`, `torus=11.311`), so this remains a robustness
default rather than evidence for promotion. No 3D mesh candidate is promoted.

Two rollout-level ablations were then added and measured. `--trajectory-render-gain`
with `--trajectory-render-samples` injects render adjoints at stored rollout
snapshots instead of only at the terminal trace. `--trajectory-mesh-gain` uses
the same snapshot schedule for mesh coverage/surface adjoints without requiring
intermediate render gradients. Earlier short probes with trajectory render
signals did not help strict validation (`teapot gain=0.1 -> strict score
1.105`; `torus gain=0.05 -> strict score 11.314`), so the new mesh-only knob is
tracked separately for ablations. The direct backend now applies target
coverage pressure to every active particle row by default rather than only
sampled render-gradient rows, because full-cloud coverage updates are already
computed and the direct backward pass already visits every particle. Coverage
adjoints are now scaled by `--coverage-gain` alone instead of being multiplied
by render `--motion-gain`, matching the proxy backend's mesh-update scale. Use
`--no-full-coverage-adjoint` only to reproduce the older sparse-row coverage
ablation. The remaining failures suggest the coverage proxy is still too
nearest-surface/projection-like unless learned through a stronger rollout
objective and visibility-aware render loss.

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

## 3D Training Diagnostics

The direct rollout render-training path records per-round
`train_liveness_output_delta_norm` and `train_material_output_delta_norm` in the
serialized training history. These fields measure actual output-row movement
between the pre-training model and the candidate model before strict selection
may roll the candidate back. They diagnose whether liveness and material terms
reach the model, or whether global clipping and competing position/render terms
dominate the update.

Direct rollout training also exposes `--direct-output-gradient-rms-cap`, which
caps each output channel's RMS gradient across particles before MLP backward.
The robust default is `0.05`; set it to `0` for ablations. This cap does not
change the mesh/render adjoint construction, but prevents one high-magnitude
motion or render channel from forcing every output row into near-zero movement
through the global SGD clip.

The regression
`direct_rollout_training_activates_near_surface_material_visible_particles`
proves the full direct-rollout path from material-visible liveness state
adjoint through MLP backward and SGD: a dormant material-visible particle on the
teapot surface increases the liveness output bias and suppresses the material
opacity output while it is still dormant. The full bin test suite currently
passes `141` tests.

A short direct teapot probe kept under `target/` remains intentionally
unpromoted:

```bash
target/debug/burn_automata train-render3d --target teapot --rounds 1 \
  --supervised-steps-per-round 1 --particles 32 --rollout-steps 4 \
  --gradient-particles 16 --trajectory-render-samples 2 \
  --coverage-samples 64 --image-size 8 --target-samples 64 \
  --seed-scale 0.72 --model-output target/probe_teapot_direct_diag.bpk \
  --report-output target/probe_teapot_direct_diag_report.json \
  --training-backend direct-rollout --direct-line-search=false
```

With the cap disabled, that probe reported `strict_passed=false`,
`strict_score=107.034`, `train_grad_norm=398.3`, and
`train_grad_scale=6.3e-4`. With the default cap, it remained
`strict_passed=false` but reduced `train_grad_norm` to `115.7` and raised
`train_grad_scale` to `2.16e-3`. A matching torus probe reduced
`train_grad_norm` from `414.8` to `120.2` and raised `train_grad_scale` from
`6.0e-4` to `2.08e-3`. The result confirms channel balancing reduces gradient
domination, but these tiny probes still show `selection_min_newly_activated_fraction=0`
and fail growth-timing checks. The next training iteration should focus on
active-front/liveness target strength and rollout horizon, not catalog
promotion.

The latest direct-rollout selector pass keeps two additional diagnostic states
without relaxing catalog promotion. First, bounded local-front liveness
precursor progress can be retained even when the strict score improvement is
too small to spend general render slack. On the 32-particle teapot probe,
`target/probe_teapot_liveness_precursor005_r4_report.json` reduced the
selection front-liveness margin from `5.686` to `0.987`, but still produced
`selection_min_newly_activated_fraction=0` and remained strict-failing. Second,
actual local-front activation breakthroughs can be retained for continued
refinement when render loss and active-surface bounds stay controlled. The raw
diagnostic
`target/probe_teapot_raw_liveness_lr004_refine_r4_report.json` retained a
breakthrough from `1` to `32` active particles, reached
`newly_activated_fraction=1.0`, passed the local-front coherence report, and
reduced strict score from roughly `107` to `66.58`. It still fails strict
promotion because activation happens too abruptly (`first_growth_step=2`,
`activation_span_steps=0`), target/material-visible coverage remains tiny, and
surface-normal/render-density gates are far below threshold. This is optimizer
evidence that the local model can learn activation; it is not catalog evidence.
The matching torus diagnostic
`target/probe_torus_raw_liveness_lr004_refine_r4_report.json` behaved similarly:
it retained `32` active particles, passed local-front coherence, and reached
strict score `76.52`, but still failed progressive activation, target coverage,
surface-normal/material-visible support, torus angular coverage, and render
loss gates.

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

## 2026-07-01 Direct 3D Front Objectives

The direct-rollout backend now applies mesh motion and material visibility
targets through local-front output objectives, then boosts sparse nonzero
geometry/material output gradients before the global output-channel RMS cap.
This keeps far dormant particles unguided while giving newly activated local
front particles immediate motion/material targets.

Compact torus diagnostics remain strict-failing, so no 3D artifact is promoted.
The one-round probe
`target/probe_torus_front_material_r1_report.json` improved compact render loss
from `0.91883` to `0.91741` and increased
`train_material_output_delta_norm` from roughly `0.00118` to `0.00373`, but
material-visible target coverage stayed `0.0` and strict score stayed near
`77.04`. The four-round probe
`target/probe_torus_material_selection_r4_report.json` selected round `0` only;
raw continuation improved render loss but worsened temporal/geometry strict
score, so rollback remains correct.

The selector now records `selection_material_active_mean_opacity` and
`selection_material_visible_count`, and can retain bounded material precursor
progress when render, activation, and material-tail constraints do not regress.
The next pass added a bounded target-extent output objective to direct mesh
geometry supervision. It uses explicit active/local-front row weights, so the
extent term can expand collapsed support without waking far dormant substrate
particles. `render_proxy_target_extent_updates_expand_weighted_active_bounds`
guards the min/max-bound pressure and the zero-weight far-row behavior.

Compact two-round probes show this is useful plumbing but not the missing 3D
morphogenesis objective. `target/probe_torus_direct_extent_r2_report.json`
selected round `0`, kept render loss near `0.91739`, active target coverage at
`0.0625`, material-visible target coverage at `0.0`, and strict score near
`77.041`. `target/probe_teapot_direct_extent_r2_report.json` selected round
`0`, improved compact render loss from the prior `0.91774` probe to
`0.91724`, and moved strict score from `67.217` to `67.106`, but
material-visible target coverage still stayed `0.0` with only one
material-visible particle.

The material coverage pass then generalized direct material output targets from
active-only rows to explicit active/predicted-active/local-front row weights.
`weighted_material_target_coverage_updates_can_promote_local_front_rows` and
`weighted_material_surface_strata_updates_promote_local_front_bins` guard that
local-front rows can receive bounded material coverage pressure while
zero-weight far rows remain untouched. Compact probes
`target/probe_torus_weighted_material_r2_report.json` and
`target/probe_teapot_weighted_material_r2_report.json` increased the first-round
material output delta slightly for teapot (`0.00394`) and kept catalog
promotion blocked, but material-visible target coverage still stayed `0.0`
with one material-visible particle.

A paired material-visible liveness output helper was added and unit-tested by
`material_visible_liveness_output_objective_activates_local_front_material_rows`.
It is intentionally not active in the default direct trajectory objective yet:
compact ablations
`target/probe_torus_front_liveness_r2_report.json` and
`target/probe_torus_front_liveness025_r2_report.json` improved render loss
(`~0.91711`) but worsened torus strict score to `77.211` without increasing
material-visible coverage. The helper remains available for controlled
phase-aware liveness/material ablations, but the default path avoids that
strict-dynamics regression.

The retained follow-up is narrower and coupled to the material objective:
`add_material_visibility_output_objective` now gives dormant local-front rows a
bounded liveness-output target only when the same row is near the mesh surface,
is receiving positive material-opacity pressure, and the temporal activation
schedule still has capacity. This is covered by
`material_visibility_output_objective_promotes_local_front_rows`. The rejected
quadratic material phase schedule was removed from the default path after
`target/probe_torus_phase_quad_material_r2_report.json` and
`target/probe_teapot_phase_quad_material_r2_report.json` worsened compact
render loss without increasing material-visible coverage. The coupled pass
improved compact render loss relative to the weighted-material probes
(`torus: 0.91741 -> 0.91733`, `teapot: 0.91723 -> 0.91703`) and increased
first-round liveness/material output deltas, but material-visible target
coverage stayed `0.0` with one material-visible particle, so strict promotion
remains blocked.

The geometry follow-up added
`render_proxy_weighted_target_coverage_updates`, guarded by
`weighted_target_coverage_updates_include_local_front_rows`, so trajectory
mesh coverage can allocate uncovered target samples to active plus weighted
local-front rows while zero-weight dormant rows remain untouched. Compact
reports `target/probe_torus_weighted_front_coverage_r2_report.json` and
`target/probe_teapot_weighted_front_coverage_r2_report.json` show this is safe
but still not sufficient at four rollout steps: active target coverage remained
`0.0625` for torus and `0.013671875` for teapot, material-visible target
coverage stayed `0.0`, and both reports still selected only round `0`.

The compact probes show the next blocker clearly: after the first local
activation breakthrough, candidate updates still trade geometry/timing for
render loss instead of improving material-visible surface coverage. The final
strict gates and catalog dry-run continue to reject all current 3D models.

The latest selector pass tightened that conclusion instead of relaxing it.
Activation-breakthrough and post-activation refinement retention now require a
temporally progressive checkpoint, and bounded temporal-front precursors must
not increase final active count or newly activated fraction beyond a small
slack. The temporal target schedule is now linear in rollout fraction so the
training objective lines up with the strict half-activation gate rather than
asking for only a small mid-rollout active set. Compact probes still fail:
`target/probe_torus_offsurface_material_r2_report.json` selected round `0`,
kept active timing at `1 -> 1 -> 1 -> 20` for steps `0/1/2/4`, reported
`selection_all_temporal_activation_progressive=false`, material-visible target
coverage `0.0`, render loss `0.924598`, and strict score `90.546`.
`target/probe_teapot_offsurface_material_r2_report.json` selected round `0`,
kept timing at `1 -> 1 -> 1 -> 4`, material-visible target coverage `0.0`,
render loss `0.924266`, and strict score `102.139`.

`add_material_visibility_output_objective` also now allows off-surface
local-front rows to receive bounded material/liveness preactivation when the
same row is already a weighted local-front candidate; far dormant rows remain
zero-weighted. This is guarded by
`material_visibility_output_objective_preactivates_offsurface_local_front_rows`.
The compact reports above did not move, which narrows the blocker to rollout
optimization/selection rather than simple material-candidate eligibility. No
torus or teapot 3D artifact is catalog-promoted from these probes.

Direct-rollout history rows now include
`direct_objective_diagnostics`, guarded by
`direct_rollout_objective_diagnostics_reports_channel_pressure`. The diagnostic
summarizes temporal-liveness, local phase, mesh-motion,
material-visibility, pre-cap, and post-cap output-gradient RMS before MLP
backprop. The refreshed compact probes show that the local objectives are
present and survive the cap:

| probe | phase RMS | phase post-cap RMS | liveness post-cap RMS | material post-cap RMS | selected timing | strict score | material-visible target coverage |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `target/probe_torus_phase_state_r2_report.json` | `0.090` | `0.088` | `1.000` | `0.125` | `1 -> 1 -> 1 -> 20` | `90.546` | `0.0` |
| `target/probe_teapot_phase_state_r2_report.json` | `0.090` | `0.089` | `1.000` | `0.125` | `1 -> 1 -> 1 -> 4` | `102.139` | `0.0` |

This pass also wires `GROWTH_3D_PHASE_CHANNEL` into the local 3D growth
student and direct-rollout output objective. The phase signal is local
(neighbor/current liveness plus current phase, no position/index/target
identity), and tests cover active/front/far output-gradient signs plus seeded
student phase response. The compact probes show the phase objective is not
missing, but it is still insufficient by itself: active timing, render loss,
and material-visible coverage remain effectively unchanged, so no 3D catalog
promotion is allowed from these artifacts.

The follow-up phase-material ablation makes the seeded local student consume
that generic phase state in the material opacity output head. This is also
local and target-agnostic: mature phase increases material opacity, but it does
not inject target positions, normals, indices, or mesh-specific categories.
`local_growth_student_model_uses_phase_for_material_maturation` covers the
controller. The compact probes show a small render-loss improvement, but no
meaningful morphogenesis improvement:

| probe | render loss | selected timing | strict score | material-visible count | material-visible target coverage |
| --- | ---: | --- | ---: | ---: | ---: |
| `target/probe_torus_phase_state_r2_report.json` | `0.924593` | `1 -> 1 -> 1 -> 20` | `90.546` | `1` | `0.0` |
| `target/probe_torus_phase_material_r2_report.json` | `0.924173` | `1 -> 1 -> 1 -> 20` | `90.546` | `1` | `0.0` |
| `target/probe_teapot_phase_state_r2_report.json` | `0.924245` | `1 -> 1 -> 1 -> 4` | `102.139` | `1` | `0.0` |
| `target/probe_teapot_phase_material_r2_report.json` | `0.923622` | `1 -> 1 -> 1 -> 4` | `102.138` | `1` | `0.0` |

Training history also records `train_phase_output_delta_norm` now, alongside
motion/liveness/material output deltas, so future runs can distinguish
"objective is present" from "the phase output head actually changed." The
phase-material ablation narrows the next blocker further: the issue is not
just the absence of a phase-to-material path; the direct rollout optimizer
still fails to create progressive activation and visible material coverage on
the target surface.

The default no-base `train-render3d` seed mode now matches the strict
conditionless-local intent. It defaults to the target-agnostic
`LocalSubstrateGrowth3d`, which keeps the connected dormant substrate but
leaves state channels `0..2` neutral instead of writing normalized seed-frame
coordinates. The old `TorusLocalSubstrateGrowth3d` /
`TeapotLocalSubstrateGrowth3d` names remain legacy compatibility aliases for
historical diagnostics, not the preferred catalog-promotion path. Base-model
continuation still uses field seeds for
position-feature models. This is covered by
`render_training_defaults_match_model_family`,
`render_training_base_defaults_to_conditionless_local_growth`,
`local_growth_seed_modes_do_not_write_coordinate_scaffold`, and
`growth_3d_local_substrate_seed_keeps_topology_without_coordinate_state`. A
64-particle validation smoke with `--seed-mode local-substrate-growth-3d`
reports `seed_coordinate_scaffold=false`,
`strict_checks.no_seed_coordinate_scaffold=true`, and
`non_opacity_seed_abs_max=0.0`; it still fails strict growth/render checks
(`2 -> 6`, strict score `130.568`), so this is an eligibility fix, not a
promotion-quality artifact.

The first `strict_gate_summary` smoke reports from the no-scaffold
`train-render3d --training-backend direct-rollout` path are intentionally
negative but now directly comparable across targets. Torus at 64 particles,
4 rollout steps, and one supervised step reports no position features,
local-conditionless lineage, no seed coordinate scaffold, neutral non-opacity
seed state, `2 -> 5` active particles, target coverage `0.039`, material-visible
target coverage `0.0059`, surface-normal bin coverage `0.038`, render loss
`0.941574`, density PSNR `0.426 dB`, and strict score `132.647`. Teapot under
the same tiny settings reports `2 -> 6` active particles, target coverage
`0.0547`, material-visible target coverage `0.0`, surface-normal bin coverage
`0.0`, render loss `0.952361`, density PSNR `0.349 dB`, and strict score
`122.548`. Both summaries keep `strict_passed=false` and `gate_passed=false`;
no BPK from these probes is catalog-promoted.

A follow-up direct-objective pass couples material/liveness preactivation to
the same local-front mesh-motion candidate weights used by temporal liveness.
This is generic over torus and teapot and is covered by
`material_visibility_output_objective_couples_local_front_to_mesh_motion`.
The corresponding tiny smoke reports
`target/no_scaffold_torus_meshcoupled_material_smoke_report.json` and
`target/no_scaffold_teapot_meshcoupled_material_smoke_report.json` are neutral:
strict scores remain `132.647` and `122.548`, with no material-visible coverage
lift, so catalog promotion stays blocked.

The next generic mesh-motion pass adds a local-front expansion target for
dormant front particles. It pushes front rows away from nearby active support
before target-specific surface coverage is available, increasing direct
mesh-motion pressure while keeping the learned update local at inference time.
`local_front_expansion_updates_push_dormant_front_outward` guards active/front
/far-row behavior. Tiny no-scaffold smokes show the signal is present but still
insufficient: torus mesh-motion RMS rises `0.00204 -> 0.00239`, teapot rises
`0.00169 -> 0.00205`, but strict scores remain `132.647` and `122.548`, target
coverage stays `0.039`/`0.0547`, and material-visible coverage does not improve.
A current `validate-growth3d` rerun of those BPKs now fails the explicit
`active_extent_growth` gate: torus reaches only `0.0239` target-normalized
active bbox diagonal and `0.0191` minimum axis extent ratio, while teapot
reaches `0.0255` bbox diagonal and `0.00936` minimum axis extent. Those tiny
active extents keep collapsed compact rollouts from looking promotion-adjacent
just because the active-front radius or liveness counters moved.

The guarded `train-render3d` selector now carries the same signal forward into
training history. Each round serializes
`selection_min_active_extent_bbox_ratio` and
`selection_min_active_extent_min_axis_ratio` across the effective selection
seed list, and `render_selection_score_penalizes_active_extent_regression`
prevents line search from accepting a render-improving checkpoint that shrinks
target-normalized active support relative to the baseline. A one-round torus
smoke should also show `report.extent_gain` and
`report.objective.extent_gain` matching the requested `--extent-gain`; this
keeps the active-extent pressure as an explicit rollout objective rather than
an implicit fraction of coverage gain.
smoke, `target/selection_extent_smoke_report.json`, emits selection
bbox/min-axis ratios `0.0239`/`0.0157`, final validation bbox/min-axis ratios
`0.0248`/`0.0172`, `active_extent_growth=false`, and `strict_passed=false`.
This is still not a solved morphogenesis artifact; it is a selection guard that
keeps collapsed compact rollouts out of the catalog path while training signals
are improved.

The next local-student pass added a bounded active-liveness sustain controller.
It is still local and target-agnostic: it reads only the liveness channel, gives
already-active/front rows a small `liveness + 1` support term, and uses a second
ReLU branch to reduce that support once liveness is high. The
`local_growth_student_model_sustains_active_liveness_without_global_activation`
test verifies that dormant substrate rows are not globally activated and that
the controller pushes back for saturated active rows. A compact torus rerun,
`target/probe_torus_active_sustain_r1_report.json`, improved seed retention
(`selection_min_final_active_count=3` instead of `1`) and strict score
(`141.637` vs. `143.170`), but regressed render loss to `0.930443` and still
failed every geometry/material coverage gate. The matching teapot probe
`target/probe_teapot_active_sustain_r1_report.json` is worse than the earlier
phase/material compact run (`strict_score=131.167`, render loss `0.971618`,
`selection_min_final_active_count=3`). A 24-step inference rollout from each
candidate reached `11/128` active particles with tiny motion
(`mean_dx_last=0.000090` torus, `0.000067` teapot); at 64 steps both activated
all `128/128` particles while motion fell further (`0.000021` torus,
`0.000028` teapot). This is useful negative evidence: bounded liveness sustain
prevents core evaporation, but the current rollout objective can still turn
into liveness flooding without enough coupled geometry/material motion.

Direct-rollout diagnostics now also report `mesh_motion_post_cap_rms` and
`mesh_motion_post_cap_nonzero_fraction`. The stronger mesh-gain ablation showed
that mesh motion pressure is not being removed by the output-channel cap:
`target/probe_torus_motiondiag_meshgain020_r1_report.json` raised
`mesh_motion_rms` from `0.000745` to `0.002980`, and
`mesh_motion_post_cap_rms` matched the pre-cap value. Selection metrics and
rollout quality stayed effectively unchanged, so the blocker is not cap loss.

The direct temporal activation objective is now mesh-coupled in the default
direct-rollout path. It still suppresses overactive rows, but dormant
local-front activation candidates must also have nonzero mesh-motion pressure.
The older ungated helper remains as an explicit ablation helper, while
`temporal_liveness_output_objective_can_gate_activation_by_mesh_motion` and
`mesh_motion_candidate_weights_track_nonzero_motion_channels` guard the new
coupling. Compact probes confirm the coupling is active but not sufficient:
`target/probe_torus_meshcoupled_r1_report.json` reduces temporal liveness
nonzero fraction from `0.421875` to `0.078125`, lowers grad norm from `8.72`
to `7.88`, and slightly increases motion/material output deltas, but strict
score remains `142.215` and a 24-step inference rollout still reaches
`64/64` active particles with `mean_dx_last=0.000040`.
`target/probe_teapot_meshcoupled_r1_report.json` shows the same pattern:
temporal liveness nonzero fraction is `0.078125`, render loss improves versus
the earlier active-sustain compact run (`0.971618 -> 0.965044`), but strict
score is still `131.711` and 24-step inference still reaches `64/64` active
particles with `mean_dx_last=0.000044`. This keeps catalog promotion blocked
and narrows the next step to recurrent/local dynamics that convert mesh-motion
pressure into sustained geometry before activation floods the substrate.

The next pass added two generic local-growth fixes without relaxing any gates:

- The local 3D student now has velocity-memory state channels. Direct rollout
  mesh-motion pressure is mirrored into those velocity outputs, boosted with
  the same sparse-channel RMS normalization used for geometry/material heads,
  and the base model reads/damps velocity memory into later `dx`. Reports now
  serialize `motion_memory_rms`, `motion_memory_post_cap_rms`, and
  `train_motion_memory_output_delta_norm`. Coverage is guarded by
  `local_growth_student_model_wires_velocity_memory_to_motion_and_damping`,
  `motion_memory_output_objective_mirrors_mesh_motion_pressure`, and the
  direct-objective diagnostics test.
- `TorusSubstrateGrowth3d` / `TeapotSubstrateGrowth3d` now use a connected
  radial substrate: sparse active particles stay in the seed core, while
  dormant particles are distributed along fixed seed-frame rays from the active
  radius to the full domain radius. This preserves local dormant neighbors
  inside kernel support without using per-index target positions. The updated
  `growth_3d_substrate_seed_keeps_sparse_active_core_in_dormant_domain` test
  now checks both 64- and 512-particle seeds, requires each radial gap to fit
  inside `HashGridConfig::growing_3dgs().eps`, requires the first dormant shell
  to sit within kernel support of the active core, and still requires broad
  domain coverage.

The corrected substrate changes the compact failure mode, but it still does not
pass strict gates. Existing 64-particle, 64-step connected probes now activate
many dormant particles instead of stalling: torus reaches `2 -> 55` active
particles with strict score `96.051`, render loss `1.619`, density PSNR
`-1.926` dB, target coverage `0.047`, and material-visible target coverage
`0.023`; teapot reaches `2 -> 52` with strict score `103.014`, render loss
`0.957`, density PSNR `0.243` dB, target coverage `0.082`, and
material-visible target coverage `0.016`. The 256-particle probes show the same
pattern at larger scale: torus reaches `7 -> 202` but strict score remains
`102.039`, while teapot reaches `7 -> 192` but strict score remains `109.176`.
This proves the topology fix removed the disconnected-first-shell artifact, but
the learned dynamics now fail through timing/local-front coherence,
surface/normal support, material visibility, and rendered density. No catalog
model is promoted from this pass.

A follow-up seeding pass added `GROWTH_3D_MIN_ACTIVE_SEED_COUNT=8` so tiny
3D training and validation runs do not start from a degenerate two- or
three-particle active line segment. The sparse-seed strict gate now accepts the
exact one-eighth active boundary (`8/64`) while still rejecting dense seed
priors. This does not relax catalog promotion: the current 64-particle,
4-step render-training smokes improve render density and active extent, but
still fail strict dynamics and coverage gates. Torus validates at `8 -> 11`
active particles with density PSNR `1.173` dB, strict score `142.319`, and
remaining failures in active-count growth, temporal/local-front growth,
surface/normal/material coverage, torus angular coverage, and render loss.
Teapot validates at `8 -> 11` active particles with density PSNR `1.022` dB,
strict score `131.803`, and the same growth/coverage/material/render blockers.
These smokes prove the low-particle seed is no longer degenerate, but they are
not promotion evidence.

The next temporal-liveness pass removes a different shortcut: the direct
rollout objective no longer expands temporal activation candidates to the full
schedule deficit, because that taught far dormant substrate particles a global
late wake-up. Temporal candidate expansion is now bounded to a small nearest
local shell, and dormant rows outside that bounded front receive a suppression
gradient against positive liveness drift. Unit coverage includes
`temporal_liveness_output_objective_bounds_nearest_shell_expansion` and
`temporal_liveness_output_objective_suppresses_nonlocal_liveness_drift`.
Validation confirms the behavior change. The previous 16-step tiny smokes
activated all particles nonlocally (`8 -> 64`) with only about one third of new
activations front-local. With bounded temporal liveness, the same 16-step
validation stays local instead of bursting: torus remains `8 -> 11` with
density PSNR `1.727` dB, and teapot remains `8 -> 11` with density PSNR
`1.535` dB. A one-round 16-step torus training probe reaches `8 -> 13`; its
front report passes (`local_newly_activated_fraction=1.0`, max nearest
previous-active distance `0.052 < 0.36`) but strict score is still `129.987`.
This proves the nonlocal activation shortcut is blocked, while the remaining
blocker shifts to sufficiently fast local propagation plus surface/material
coverage, not catalog promotion.

The robust local-growth liveness cap was then raised from `5x` to `24x`
(`ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER=24.0`). With the bounded temporal-front
objective in place this is no longer a global wake-up path; it gives local
front rows enough supervised update budget to cross the dormant-to-active
threshold within a short rollout horizon. The guard test
`robust_liveness_cap_can_cross_activation_threshold_within_short_rollout`
locks the horizon math. A 16-step cap-24 torus probe remains local and still
validates at `8 -> 13` with strict score `129.988`, so it does not solve torus
coverage. The matching teapot probe improves versus cap-5 (`strict_score
129.897 -> 119.780`, density PSNR `1.535 -> 1.565`) while staying bounded at
`8 -> 12`. This is directional improvement, not promotion evidence.

A high liveness-cap ablation (`--liveness-update-multiplier 160`) confirms the
path is cap-sensitive but not solved by simply raising the cap. Teapot strict
score improved to `92.023`, but render loss regressed to `0.925256` and
material-visible target coverage remained `0.0`; torus worsened slightly
(`strict_score=90.893`, render loss `0.925601`). This keeps the promotion gate
closed and shifts the next implementation target to a stronger rollout-level
optimizer/representation for staged geometry plus material visibility, rather
than another local-front candidate or global liveness-gain tweak.

The next direct-objective pass made activation candidate selection more honest,
but did not solve strict morphogenesis. Mesh-motion candidate weights are now
normalized by relative motion magnitude instead of being binary, and the direct
rollout objective adds a bounded local-front floor so under-active snapshots can
still train local propagation before mesh gradients reach the next shell. The
floor is local only: far dormant substrate remains ineligible, and material
visibility is still trained separately. Focused tests cover relative mesh
weighting, local-only floor eligibility, and temporal liveness selection of the
stronger mesh-supported row.

The local-front shell budget now scales with particle count instead of using a
fixed tiny ceiling. Generic front objectives use a bounded `ceil(rows / 16)`
shell capped at 64 candidates, while temporal activation now uses
`ceil(rows / 4)` with a row-scaled cap of `ceil(rows / 8)`, clamped to
`16..512`. The mesh-coupled temporal liveness candidate weights now also use
the expanded temporal shell before calling the gated liveness objective; the
low-level objective still honors explicit candidate weights, but the default
direct trainer no longer masks expanded local-shell rows back to zero. This
keeps the objective local, but prevents 1024+ particle clouds from training the
temporal growth front through only 64 rows. The tests
`local_front_candidate_budget_scales_for_larger_clouds_without_global_default`,
`temporal_front_candidate_budget_scales_but_stays_bounded`, and
`temporal_liveness_candidate_floor_uses_expanded_local_shell` lock this
behavior alongside the existing nearest-shell liveness tests. Compact
64-particle, 16-step probes show the change is local but not sufficient: torus
selected more active local-front particles (`~12 -> 15/16`) and lower liveness
margins, yet strict validation remained false (`strict_score=130.121`,
material-visible target coverage `0.0`). Teapot stayed near the cap-24
baseline (`8 -> 12`, `strict_score=119.867`). No catalog model was promoted.

A 1024-particle one-round torus probe with the scaled temporal shell and
expanded-shell candidate weights selected a guarded checkpoint at line-search
scale `32`, with nonzero direct gradients
(`train_liveness_output_delta_norm=0.001448`,
`train_phase_output_delta_norm=0.000641`) and fully local new activation
(`selection_min_front_local_newly_activated_fraction=1.0`). It still failed the
strict gate: final active count was `216/1024`, newly activated fraction
`0.1888`, target coverage `0.1846`, material-visible target coverage `0.0313`,
density PSNR `0.612 dB`, and strict score `109.526`. The remaining blocker is
therefore not fixed candidate under-sampling alone; it is still a stronger
rollout-level optimizer/representation that can couple local activation,
surface support, material visibility, and rendered density.

The direct backend now also keeps a small terminal liveness anchor during
trajectory-supervised training (`25%` of `--liveness-gain`) and reports it
separately as `terminal_liveness_state_rms` /
`terminal_liveness_state_nonzero_fraction`. This avoids hiding endpoint
active-count pressure inside the output-gradient diagnostics while keeping the
trajectory schedule as the dominant timing signal. A 512-particle torus probe
reported `terminal_liveness_state_rms=0.0372` over `31.1%` of terminal rows and
selected a bounded checkpoint, but still failed strict validation
(`101/512` active, newly activated fraction `0.1747`, target coverage `0.125`,
material-visible target coverage `0.0605`, density PSNR `0.591 dB`, strict
score `109.321`). The anchor is therefore useful instrumentation and a small
active-growth signal, not a solution to rollout-level 3D morphogenesis.

The 2026-07-01 direct-objective pass added two generic rollout signals without
relaxing strict gates. First, mesh residuals now also supervise the 3D velocity
state outputs for active and local-front particles. This is not per-index
target assignment: it uses the nearest mesh residual, local-front gating, and
the same bounded output-gradient caps as the rest of direct-rollout training.
Reports expose this as `residual_velocity_rms`,
`residual_velocity_nonzero_fraction`, `residual_velocity_post_cap_rms`, and
`residual_velocity_post_cap_nonzero_fraction`. Second, escaped active particles
can receive a bounded per-snapshot liveness-output suppression term, reported
as `surface_escape_liveness_rms`; this mirrors the existing terminal
surface-escape state adjoint but can train rollout steps directly when escaped
active rows are present.

Short probes show the new velocity pressure is present but not sufficient. A
512-particle torus one-round probe
(`target/escape_liveness_torus_probe.json`) reported
`residual_velocity_rms=0.00516`, `residual_velocity_post_cap_rms=0.01435`,
`101/512` final active particles, newly activated fraction `0.1747`, target
coverage `0.125`, material-visible target coverage `0.0605`, density PSNR
`0.591 dB`, and strict score `109.321`. The matching teapot probe
(`target/escape_liveness_teapot_probe.json`) reported
`residual_velocity_rms=0.00681`, `residual_velocity_post_cap_rms=0.01466`,
`96/512` final active particles, newly activated fraction `0.1647`, target
coverage `0.4043`, material-visible target coverage `0.0020`, density PSNR
`0.507 dB`, and strict score `89.996`. The escape-liveness term was dormant in
these short primary-seed trajectories (`surface_escape_liveness_rms=0.0`)
because active rows did not cross the strict surface threshold during the
training snapshots.

Default multiseed direct training remains the right direction but is not yet
enough. With selection-seed training enabled, the torus one-round probe
(`target/multiseed_escape_liveness_torus_probe.json`) selected a larger
line-search step (`32`), improved render loss to `0.903445` and density PSNR to
`0.607 dB`, and slightly improved strict score to `109.315`; however it still
failed newly activated fraction, active extent, motion, temporal geometry,
target coverage, material-visible coverage, normal coverage, torus angular
coverage, and render gates. A four-round multiseed probe
(`target/multiseed_escape_liveness_torus_4round_probe.json`) showed later
rounds can increase heldout active count (`97 -> 137`) and improve temporal
schedule error (`0.1306 -> 0.1219`), but those checkpoints were rejected because
active-surface max exceeded the strict bound (`0.419..0.437`) while target and
material-visible coverage stayed low. The next backend target is therefore
coverage-coupled activation across all training/selection seeds, not higher
single-seed velocity gain or gate relaxation.

The guarded selector now records material-visible target mean/max distance in
each render-training history row and adds a small score penalty plus baseline
regression check for those distances. This exposes material-surface approach
before the strict material-visible coverage fraction flips from zero. Focused
tests cover both reward and regression behavior:
`render_selection_score_rewards_lower_material_visible_target_distance` and
`render_selection_score_penalizes_material_visible_target_distance_regression`.
The compact probes still fail strict validation, but now explain the material
blocker directly: torus material-visible target distance is `0.961` mean /
`1.230` max with coverage `0.0`; teapot is `0.383` / `0.711` with coverage
`0.0`. No catalog model was promoted.

The next material-surface pass added a generic material-visible surface
approach objective. It reuses mesh projection, not particle indices or
target-specific seats, and only applies to render-visible material rows that
are already active or inside the bounded local activation front. The same
signal now feeds proxy target updates, direct-rollout output gradients, and
terminal/trajectory position adjoints. Focused tests cover active visible
motion, local-front visible motion, no far dormant motion, output-gradient
sign, and position-adjoint sign:
`material_visible_surface_approach_updates_pull_visible_active_particles_toward_mesh`,
`material_visible_surface_approach_updates_do_not_move_far_dormant_material`,
`material_visible_surface_approach_output_objective_uses_training_gradient_sign`,
and `material_visible_surface_position_adjoint_tracks_visible_local_front_only`.
Compact one-round direct probes remain strict-failing, but the material rows
now cross the hard coverage gate slightly instead of staying at zero. Torus
reports material-visible mean/max distance `0.844`/`1.188`, coverage
`0.043`, surface-tail p99 `0.054`, and strict score `99.358`; teapot reports
`0.309`/`0.570`, coverage `0.023`, surface-tail p99 `0.234`, and strict score
`88.881`. Surface-profile and normal-bin coverage are still far below strict
thresholds, growth timing still fails, and guarded line search selected a
no-op update at this compact scale, so no catalog model was promoted.

The material-visible motion objective now also has a generic coverage
relocation path. A shared material-visible row-weight helper selects live
visible particles plus visible particles in the bounded local front, then
reuses the existing hard/soft/sliced-OT, surface-strata, and normal-bin
coverage helpers. The signal is wired into proxy targets, direct-rollout
output gradients, and terminal/trajectory position adjoints. Tests cover local
row eligibility, uncovered-bin relocation, output-gradient sign, and
position-adjoint sign:
`material_visible_surface_row_weights_include_local_front_but_not_far_dormant`,
`material_visible_surface_coverage_updates_move_visible_rows_to_uncovered_bins`,
`material_visible_surface_coverage_output_objective_uses_training_gradient_sign`,
and `material_visible_surface_coverage_position_adjoint_tracks_visible_rows`.
Compact torus/teapot probes still roll back to the baseline checkpoint
(`train_step_scale=0.0`), so this is objective coverage, not accepted model
improvement. The validation metrics remain strict-failing: torus
material-visible profile/normal coverage `0.0625`/`0.269`, score `99.358`;
teapot `0.1875`/`0.192`, score `88.881`. No catalog model was promoted.

The 2026-07-01 direct line-search pass separates strict checkpoint promotion
from bounded training continuation. Strict candidate selection is unchanged,
and catalog-bound outputs still run the same validation gate before replacing
catalog artifacts. The training loop can now continue from a non-promotable
candidate only when it improves render/coverage/extent/activation metrics
without a large temporal, surface-tail, or nonlocal-front regression. Line
search also evaluates every continuation candidate against the no-op baseline
for that step, so a larger render-improving but surface-escaping scale cannot
supersede a smaller bounded continuation.

The same pass adds a floor for the local phase objective
(`ROBUST_3D_PHASE_GAIN=0.10`) whenever liveness training is enabled. This
decouples the internal progression channel from raw activation pressure: phase
can be strengthened without increasing the liveness gain that previously drove
global wake-up shortcuts. Focused tests cover the phase floor, continuation
gate, bad-candidate rollback, and the full direct-rollout CLI suite.

Compact connected probes now move, but they remain diagnostic only. Torus
`target/direct_linesearch_continuation_torus_probe.json` selects a bounded
scale-16 update with nonzero motion/liveness/phase/material output deltas and
improves render loss to `0.82820076`, density PSNR to `1.0458411`, and strict
score to `99.308075`. It still fails newly activated fraction, temporal
activation/geometric progression, target/material-visible coverage,
surface/normal coverage, torus angular coverage, and render-density gates.
Teapot `target/direct_linesearch_continuation_teapot_probe3.json` chooses the
safe scale-16 continuation rather than the unsafe scale-32 projection
shortcut; render loss improves to `0.85735345`, density PSNR to `0.9054107`,
surface-bin coverage to `0.359375`, and normal-bin coverage to `0.46153846`,
but strict score remains `98.424095`. It still fails newly activated fraction,
temporal activation/geometric progression, surface mean improvement,
target/material-visible coverage, surface/normal coverage, and render gates.
No catalog model was promoted.

The coverage-coupled liveness pass makes target-coverage pressure visible to
the local activation controller. The helper converts generic mesh coverage
updates into dormant local-front candidate weights, then feeds those weights
into direct liveness, temporal liveness, and material-visibility activation
objectives. It does not use particle indices or target-specific seats. Focused
tests are `target_coverage_liveness_candidate_weights_prioritize_local_front_coverage_rows`,
`target_coverage_liveness_objective_activates_coverage_front_rows`, and the
expanded `direct_rollout_objective_diagnostics_reports_channel_pressure`.
Short one-round probes confirm the signal is active but still insufficient:
torus `target/coverage_liveness_torus_probe.json` reports
`target_coverage_liveness_rms=0.00147` over `0.129` of rows, final active
`102/512`, newly activated fraction `0.1767`, target coverage `0.1230`,
material-visible target coverage `0.0625`, density PSNR `0.608 dB`, and strict
score `109.308`. Teapot `target/coverage_liveness_teapot_probe.json` reports
`target_coverage_liveness_rms=0.00053` over `0.129` of rows, final active
`96/512`, newly activated fraction `0.1647`, target coverage `0.4043`,
material-visible target coverage `0.0020`, density PSNR `0.506 dB`, and strict
score `89.991`. Both still fail activation growth, temporal geometry, coverage,
normal/profile, render, and permutation gates, so no catalog model was
promoted.

The scheduled temporal-extent pass adds a mesh-generic curriculum pressure on
active and dormant local-front boundary rows. It uses target bounds and the
existing temporal activation target to ask the active support to expand over
time; it does not assign particles to target samples, and centered seed rows do
not receive one-sided drift. Focused tests are
`temporal_extent_motion_updates_expand_boundary_front_without_center_bias` and
`temporal_extent_motion_output_objective_trains_outward_boundary_motion`; the
direct diagnostics test also now asserts nonzero `temporal_extent_motion_rms`.
With the tuned gain fraction (`0.25`), compact one-round probes show the signal
is active but not sufficient. Torus
`target/temporal_extent_torus_probe.json` reports
`temporal_extent_motion_rms=0.00036`, final active `102/512`, active extent
bbox/min-axis `0.139`/`0.131`, density PSNR `0.608 dB`, and strict score
`109.315`; this is roughly neutral versus the previous coverage-liveness probe
and does not improve strict selection. Teapot
`target/temporal_extent_teapot_probe.json` reports
`temporal_extent_motion_rms=0.00038`, active extent `0.251`/`0.215`, density
PSNR `0.506 dB`, and strict score `89.991`. Both still fail progressive
activation, temporal geometry, target/material-visible coverage, normal/profile
coverage, and render gates. No catalog model was promoted.

The material-coverage liveness pass adds activation/material coupling for the
hard material-visible coverage blocker. It uses the same weighted mesh coverage
helpers as material-visible surface coverage, but allows dormant local-front
rows to act as low-weight potential visible support before they have crossed
the material-opacity threshold. The candidate weights feed direct liveness,
temporal liveness, and material-visibility objectives; active rows are never
reactivated and far dormant substrate remains ineligible. Focused tests are
`material_coverage_liveness_candidate_weights_prioritize_local_front_rows`,
`material_coverage_candidates_train_liveness_and_material_updates`, and the
expanded `direct_rollout_objective_diagnostics_reports_channel_pressure`.
Compact one-round probes confirm nonzero objective pressure without changing
promotion status. Torus
`target/material_coverage_torus_probe_training.json` reports
`material_coverage_liveness_rms=0.000115` over `0.174` of rows, render loss
`0.859797`, density PSNR `0.806 dB`, final active `23/128`, material-visible
coverage/profile/normal support still `0.0`, and strict score `130.879`.
Teapot `target/material_coverage_teapot_probe_training.json` reports
`material_coverage_liveness_rms=0.000135` over `0.120` of rows, render loss
`0.873122`, density PSNR `0.685 dB`, final active `15/128`, material-visible
coverage/profile/normal support still `0.0`, and strict score `121.074`. This
is objective wiring and diagnostic coverage, not a promotion-quality model.

The paired material-coverage front-motion pass now feeds the same local
potential-visible support into spatial output gradients, so candidate rows can
start redistributing toward uncovered target support before they cross the
material-visible threshold. This is still mesh-generic: active visible material
rows and dormant local-front candidates are weighted through the shared target
coverage helper, with no particle ids, per-sample seats, or far dormant
activation. Focused tests are
`material_coverage_front_motion_updates_move_potential_local_front_rows`,
`material_coverage_front_motion_output_objective_uses_training_gradient_sign`,
and the expanded `direct_rollout_objective_diagnostics_reports_channel_pressure`.
Compact one-round probes confirm the new motion term is present but not enough
to clear strict gates. Torus
`target/material_coverage_motion_torus_probe_training.json` reports
`material_coverage_motion_rms=0.000457` over `0.222` of rows, render loss
`0.859797`, density PSNR `0.806 dB`, final active `23/128`,
material-visible coverage/profile/normal support still `0.0`, and strict score
`130.879`. Teapot
`target/material_coverage_motion_teapot_probe_training.json` reports
`material_coverage_motion_rms=0.000388` over `0.190` of rows, render loss
`0.873122`, density PSNR `0.685 dB`, final active `15/128`,
material-visible coverage/profile/normal support still `0.0`, and strict score
`121.074`. No catalog model was promoted.

The material-coverage recurrent-memory pass mirrors that potential-support
motion into the 3D velocity-memory output channels. This closes a wiring gap
where material-coverage motion was a one-step spatial target, while the
recurrent velocity state only saw the older mesh-motion objective. The new
diagnostics fields are `material_coverage_motion_memory_rms` and
`material_coverage_motion_memory_nonzero_fraction`; focused coverage is
`material_coverage_motion_memory_mirrors_local_front_motion_pressure` plus the
direct diagnostics test. Compact probes confirm the recurrent signal is active
but still not a strict-quality training result. Torus
`target/material_coverage_memory_torus_probe_training.json` reports
`material_coverage_motion_memory_rms=0.00256` over `0.222` of rows, render loss
`0.859797`, density PSNR `0.806 dB`, `23/128` final active particles,
material-visible coverage/profile/normal support `0.0`, and strict score
`130.879`. Teapot
`target/material_coverage_memory_teapot_probe_training.json` reports
`material_coverage_motion_memory_rms=0.00218` over `0.190` of rows, render loss
`0.873122`, density PSNR `0.685 dB`, `15/128` final active particles,
material-visible coverage/profile/normal support `0.0`, and strict score
`121.074`. The MLP update remains very small
(`train_motion_memory_output_delta_norm` around `1.6e-5`), so the next blocker
is optimizer/rollout strength and sustained recurrent training, not missing
velocity-memory wiring. No catalog model was promoted.

The direct line-search continuation gate now rejects morphology-only
continuations. Previously `render_selection_training_progress_beats` returned
true for any `morphology_non_regressed` candidate, so a wide line search could
keep training from a candidate that did not improve render, coverage, extent,
activation, or strict score. The strict-selected checkpoint path is unchanged,
but non-promotable continuation now still needs bounded render progress plus a
coverage, extent, material-distance, or activation improvement. Focused test:
`render_selection_training_progress_rejects_morphology_only_continuation`.
The corrected compact probes show the intended behavior. Default teapot line
search still selects the useful scale-32 checkpoint
(`target/progress_gate_linesearch_teapot_probe_training.json`: render loss
`0.872612`, density PSNR `0.687 dB`, strict score `120.970`, final active
`16/128`), while the wide-scale teapot probe rolls back the scale-128
morphology-only attempt
(`target/progress_gate_widescale_teapot_probe_training.json`:
`rolled_back_to_best_checkpoint=true`, strict score `121.075`). A wide-scale
torus probe can still select scale `64` because its selection score/render loss
improve, but strict validation remains far from promotion
(`target/progress_gate_widescale_torus_probe_training.json`: strict score
`130.942`, material-visible coverage `0.0`). No catalog model was promoted.

The inner morphology-recovery fallback and round history now report the same
applied-checkpoint semantics. A candidate that only flips
`morphology_non_regressed` is not retained unless strict score also improves
and render/density do not regress, and a rolled-back round records
`train_step_scale=0.0` instead of the attempted candidate scale. Focused tests:
`render_selection_morphology_recovery_requires_strict_score_improvement` and
the expanded `render_proxy_training_rolls_back_rejected_round_before_next_round`.
The refreshed wide-scale teapot probe
`target/applied_scale_widescale_teapot_probe_training.json` reports
`rolled_back_to_best_checkpoint=true`, `train_step_scale=0.0`, render loss
`0.873138`, density PSNR `0.684 dB`, and strict score `121.075`. This improves
training diagnostics and prevents misleading scale reports, but it does not
change the strict failure state or promote a model.

The direct rollout objective now has an explicit temporal materialization term.
It uses the same bounded local-front candidate path as temporal liveness, then
trains material opacity toward a scheduled visible logit for under-materialized
front rows. This does not introduce particle seats or shape-specific logic; it
only couples local activation timing to material support. Focused tests:
`temporal_materialization_output_objective_grows_only_local_front_candidates`,
`temporal_materialization_target_follows_rollout_schedule`, and the expanded
`direct_rollout_objective_diagnostics_reports_channel_pressure`. Compact
one-round probes confirm the signal is present in real training. Torus
`target/temporal_materialization_torus_probe_training.json` reports
`temporal_materialization_rms=0.00780` over `21.3%` of rows,
`material_post_cap_rms=0.0500`, render loss `0.859991`, density PSNR
`0.799 dB`, `17/128` final active particles, material-visible coverage/profile
/normal support `0.0`, and strict score `131.493`. Teapot
`target/temporal_materialization_teapot_probe_training.json` reports
`temporal_materialization_rms=0.00770` over `21.6%` of rows,
`material_post_cap_rms=0.0500`, render loss `0.873010`, density PSNR
`0.685 dB`, `15/128` final active particles, material-visible coverage/profile
/normal support `0.0`, and strict score `121.072`. This proves material
schedule pressure is wired through the direct backend, but the material channel
is already cap-saturated in these compact probes; the remaining failure is
sustained local growth and surface redistribution, not missing material-output
pressure. No catalog model was promoted.

The extent-front and temporal-extent motion objectives now also feed recurrent
velocity-memory outputs. Before this pass, mesh/material coverage motion could
teach velocity memory, but scheduled extent expansion was a one-step spatial
target only. The new `add_extent_motion_memory_output_objective` mirrors both
extent-front and temporal-extent spatial gradients into the 3D velocity state
outputs, so local expansion pressure can persist across rollout steps. Focused
coverage is `extent_motion_memory_mirrors_front_and_temporal_extent_pressure`
plus the expanded direct diagnostics test. Compact probes confirm the wiring:
torus `target/extent_memory_torus_probe_training.json` reports
`extent_motion_memory_rms=0.00717` over `25.5%` of rows, render loss
`0.859991`, density PSNR `0.799 dB`, active extent bbox ratio `0.0459`,
`17/128` final active particles, and strict score `131.493`; teapot
`target/extent_memory_teapot_probe_training.json` reports
`extent_motion_memory_rms=0.00541` over `25.4%` of rows, render loss
`0.873010`, density PSNR `0.685 dB`, active extent bbox ratio `0.0646`,
`15/128` final active particles, and strict score `121.072`. Material-visible
coverage/profile/normal support remains `0.0` in both probes. This confirms
the recurrent extent path is no longer missing, but the compact runs still need
stronger or longer optimization to convert the available recurrent signal into
strict-quality morphology. No catalog model was promoted.

Direct rollout training now uses a sublinear trajectory-row normalizer instead
of full row averaging. The older supervised helper still performs true
mean-over-rows normalization for ordinary batches, but direct rollout objectives
are already per-channel RMS capped before MLP backprop; dividing the accumulated
trajectory gradient by every particle-step was damping sparse local-front
signals before line search could evaluate them. The new
`normalize_direct_rollout_gradients` scales by
`rows^-0.75`, preserving sublinear particle/trajectory scaling while making
compact local-front updates strong enough to hit the existing line-search and
gradient-clip guards. Tests:
`direct_rollout_gradient_normalization_averages_by_rows` preserves the generic
helper, and
`direct_rollout_gradient_normalization_keeps_sparse_rollout_signal_sublinear`
covers the direct helper.

Compact probes show the change is directionally useful but not sufficient for
promotion. Torus
`target/sublinear_norm_torus_probe_training.json` selects a smaller effective
scale (`train_step_scale=0.625`), has `train_grad_norm=1.151` and
`train_grad_scale=0.871`, and remains nearly neutral: render loss `0.859989`,
density PSNR `0.799 dB`, `17/128` final active particles, material-visible
support `0.0`, strict score `131.493`. Teapot
`target/sublinear_norm_teapot_probe_training.json` selects scale `4.0`, with
larger accepted output deltas (`train_liveness_output_delta_norm=0.00143`,
`train_motion_memory_output_delta_norm=0.000651`), improving compact render
loss to `0.872458`, density PSNR to `0.688 dB`, final active count to `16/128`,
active extent bbox ratio to `0.0772`, and strict score to `120.968`. Material
visible support remains `0.0`, so strict gates still block catalog promotion.

The direct rollout objective now also has an active-surface materialization
precursor plus a material-surface candidate motion path. Active or
predicted-active rows near the mesh surface receive bounded material-output
pressure before they cross the material-visible threshold, and the
material-visible approach/coverage helpers now include a low, scale-normalized
candidate floor for those rows. This remains mesh-generic: it uses target
projection distance, liveness/predicted-liveness, local-front weights, and
candidate weights, not particle ids, target seats, or torus/teapot branches.
Focused tests:
`active_surface_materialization_promotes_active_surface_rows_only`,
`active_surface_materialization_respects_local_front_candidate_weights`,
`material_surface_candidate_approach_moves_active_rows_before_visibility`,
`material_surface_candidate_coverage_uses_predicted_active_rows`, and the
expanded `direct_rollout_objective_diagnostics_reports_channel_pressure`.

Compact one-round probes show the new signals are present but still do not
solve strict 3D morphogenesis. Torus
`target/candidate_surface_torus_probe_training.json` reports
`active_surface_materialization_rms=0.02385` over `25.4%` of rows,
`material_visibility_rms=0.2248`, `15` material-visible particles, render loss
`0.859989`, density PSNR `0.799 dB`, `17/128` final active particles, target
coverage `0.0859`, material-visible coverage/profile/normal support `0.0`, and
strict score `131.493`. Teapot
`target/candidate_surface_teapot_probe_training.json` reports
`active_surface_materialization_rms=0.02379` over `25.2%` of rows,
`material_visibility_rms=0.2116`, `15` material-visible particles, render loss
`0.872461`, density PSNR `0.688 dB`, `16/128` final active particles, target
coverage `0.2109`, material-visible coverage/profile/normal support `0.0`, and
strict score `120.966`. This narrows the blocker again: material/surface
candidate wiring is active, but compact training still fails sustained active
growth, temporal geometry progression, and broad target/material-visible
redistribution. No catalog model was promoted.

The liveness objective now also has a recurrent phase-memory path. The new
`add_liveness_phase_memory_output_objective` mirrors liveness output pressure
into the existing growth phase channel, so temporal/front activation pressure
can become stateful across rollout steps. This is generic and inherits the
existing liveness locality and burst-suppression gates; it does not select
shape-specific rows independently. Focused coverage:
`liveness_phase_memory_mirrors_liveness_pressure_only` and the expanded
`direct_rollout_objective_diagnostics_reports_channel_pressure`.

Compact probes confirm the recurrent phase-memory signal is active but not
yet sufficient. Torus
`target/liveness_phase_memory_torus_probe_training.json` reports
`liveness_phase_memory_rms=0.0150` over `75.0%` of rows, `phase_post_cap_rms`
`0.0289`, render loss `0.859985`, density PSNR `0.799 dB`, `17/128` final
active particles, target coverage `0.0859`, material-visible support `0.0`,
and strict score `131.493`. Teapot
`target/liveness_phase_memory_teapot_probe_training.json` reports
`liveness_phase_memory_rms=0.0160` over `76.1%` of rows, `phase_post_cap_rms`
`0.0298`, render loss `0.872431`, density PSNR `0.688 dB`, `16/128` final
active particles, target coverage `0.2109`, material-visible support `0.0`,
and strict score `120.966`. This confirms phase-memory wiring is not the
missing piece by itself; the remaining blocker is converting recurrent phase,
liveness, and motion pressure into much larger accepted model updates over
multi-round rollouts while preserving render nonregression and temporal growth
constraints. No catalog model was promoted.

The local-growth seed model now also wires phase back into liveness through a
bounded local-front bridge. The hidden controller reads
`blurred_liveness - self_liveness + phase`, requires local-front contrast before
ReLU activation, and writes only to the liveness output channel. This gives
phase memory an inference-time activation path without absolute coordinates,
particle ids, target seats, or phase-only global activation. Focused coverage:
`local_growth_student_model_uses_phase_to_boost_local_front_liveness`.

Compact one-round probes show this bridge is active but still not sufficient
for strict 3D morphogenesis. Torus
`target/phase_liveness_bridge_torus_probe_training.json` improves compact
render loss from `0.859985` to `0.797211`, density PSNR from `0.799` to
`1.263`, final active count from `17/128` to `18/128`, and strict score from
`131.493` to `131.306`; target coverage remains `0.0859` and
material-visible target coverage remains `0.0`. Teapot
`target/phase_liveness_bridge_teapot_probe_training.json` improves active count
from `16/128` to `18/128` and strict score from `120.966` to `120.732`, but
compact render loss regresses from `0.872431` to `0.896240`, target coverage
stays near `0.207`, and material-visible target coverage remains `0.0`. These
results keep the blocker unchanged: local recurrent activation is stronger,
but the rollout objective still does not produce broad material-visible surface
redistribution under the strict gates. No catalog model was promoted.

The remaining backend step is not another scalar liveness/seed tweak. The
strict failures point to a stronger recurrent/local rollout objective that
turns bounded front activation into sustained geometry and material coverage
while preserving the temporal growth schedule.

## 2026-07-01 Material-Coverage Materialization Probe

The direct rollout objective now exposes material-coverage candidate
materialization separately from temporal and active-surface materialization.
`add_material_coverage_materialization_output_objective` writes only to the
render-material output channel and is driven by existing local
material-coverage candidate weights; it does not use particle ids, target
seats, or torus/teapot branches. Diagnostics now serialize
`material_coverage_materialization_rms` and
`material_coverage_materialization_nonzero_fraction`, and the focused tests
cover candidate-only locality plus the expanded direct-objective diagnostics.

The seeded conditionless local-growth controller also matures active rows more
quickly (`LOCAL_GROWTH_ACTIVE_MATERIAL_GAIN=0.50`). This is a generic active
liveness-to-material bridge with self-damping on high material opacity; it does
not wake dormant rows by itself.

Compact probes improved render density but still fail strict morphogenesis and
catalog gates:

| probe | render total | density PSNR | active growth | material-visible count | material-visible target coverage | strict score | decision |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `target/active_material_gain_torus_probe_training.json` | `0.6686` | `2.055` | `8 -> 19` | `15` | `0.0` | `131.202` | reject |
| `target/active_material_gain_teapot_probe_training.json` | `0.8117` | `1.078` | `8 -> 17` | `15` | `0.0` | `120.813` | reject |
| high-gain torus ablation (`lr=0.002`, liveness/mesh `0.2`, cap `0.12`) | `0.6676` | `2.063` | `8 -> 21` | `15` | `0.0` | `131.159` | reject |
| high-gain teapot ablation (`lr=0.002`, liveness/mesh `0.2`, cap `0.12`) | `0.8106` | `1.084` | `8 -> 21` | `15` | `0.0` | `120.610` | reject |

The new materialization signal is active but not sufficient: accepted updates
increase core render density and modestly improve activation/extent, while
material-visible support remains concentrated near the seed core and never
crosses the target surface coverage/profile/normal gates. No 3D artifact is
promoted from this pass.

## 2026-07-02 Material Surface-Motion Diagnostics

The direct rollout diagnostics now mirror the material visible-surface motion
term that the trainer already applies. Reports serialize
`material_surface_motion_rms` and
`material_surface_motion_nonzero_fraction`, and the combined-gradient
diagnostic includes that term before channel capping. This closes an
instrumentation gap: future torus/teapot probes can distinguish absent
material-surface pressure from pressure that is present but not yet converted
into accepted rollout motion.

The direct trainer also applies a generic post-composition spatial-motion RMS
floor before channel capping. This is not target-specific: it only boosts
existing nonzero spatial output gradients, uses the configured direct gradient
RMS budget, and remains bounded by the normal per-channel cap. The goal is to
keep local mesh/surface/coverage motion pressure from being underrepresented
relative to liveness/material channels.

Focused tests:
`direct_rollout_objective_diagnostics_reports_channel_pressure` now checks
material-surface motion diagnostics and post-cap spatial motion budget, and the
material-surface candidate tests continue to verify bounded-frontier locality.

Compact two-round probes show the new diagnostics are active, but strict
morphogenesis is still not solved:

| probe | strict score | active growth | mean / peak motion | target coverage | material-visible target coverage | material surface motion RMS / nonzero | decision |
| --- | ---: | --- | --- | ---: | ---: | --- | --- |
| `target/goal_probe_torus_motion_balance.json` | `109.288` | `8 -> 43` | `0.0288 / 0.00444` | `0.0781` | `0.0195` | `0.0316 / 0.399` | reject |
| `target/goal_probe_teapot_motion_balance.json` | `88.796` | `8 -> 48` | `0.0270 / 0.00411` | `0.2129` | `0.0` | `0.0320 / 0.410` | reject |

The post-cap spatial motion RMS rose from roughly `0.041..0.044` to
`0.050..0.051`, but the selected compact rollouts and strict scores were
unchanged over two rounds. This indicates the next blocker is not missing
surface-motion supervision; it is converting those bounded local output
gradients into larger accepted recurrent rollout dynamics under line-search,
SGD clipping, and strict render/nonregression selection. No catalog model was
promoted.
