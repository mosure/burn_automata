# Fully Local 3D Morphogenesis Roadmap

## Goal

The target capability is a conditionless 3D NPA that behaves like the 2D
growing morphogenesis pipeline: particles start from a simple neutral seed,
interact through local SPH perception, update position and hidden state through
a shared neural rule, and form a target object through rollout dynamics rather
than particle-index assignment, precolored particles, residual-state targets, or
absolute world-position fields.

## Alignment Contract

The local 3D path must keep these properties:

- `position_features=false` in the model config.
- Random or minimal neutral seed state, not target residual/color channels.
- Shared update rule with local density/state perception and directional
  gradients.
- Multi-step rollout supervision, not one-step projection only.
- Validation from a saved `.bpk` loaded back from disk.
- Geometry, color, opacity, and finite-state checks across particle counts,
  seed scales, and rollout horizons.

This mirrors the upstream NPA framing: SPH perception provides local particle
neighborhood features, morphogenesis is trained through rendered density/color
losses, and 3D objects use Gaussian-splat multi-view losses rather than fixed
per-particle targets.

## Current Baselines

| artifact or experiment | local? | target shortcut | validation |
| --- | --- | --- | --- |
| retired field render-proxy BPKs | no | absolute position features | diagnostic only; removed from `assets/models` |
| legacy morphogen BPKs | no | target residual/color seed state | diagnostic only; removed from `assets/models` |
| `uv_torus_growth_3d.bpk` | partly | compact seed-frame coordinate scaffold; no absolute position features | hidden from app catalog because it fails strict compact-growth coverage/render/topology gates |
| `teapot_growth_3d.bpk` | partly | compact seed-frame coordinate scaffold; no absolute position features | hidden from app catalog after seed-varied validation exposed held-out fragility |
| `--training-mode projection-baseline` | local update, biased seed | residual/color seed channels | diagnostic only |
| `--training-mode rollout-local` | local student | teacher generated from biased seed-frame baseline | plumbing diagnostic |
| `ablate-local-3d` | yes | none; random ball + local rollout rows | fails current gates |
| `render-loss-3d` | evaluator | none | CPU multi-view density/color/depth oracle |
| `train-render3d` | direct/proxy render trainer | saved growth starting model | direct MLP-output render adjoint scaffold; proxy fallback |

## Current Implementation Status

This pass adds explicit compact neutral growth seeds:

- `ParticleSeed::TorusGrowth3d`
- `ParticleSeed::TeapotGrowth3d`

These seeds match the 2D growing setup more closely than the legacy 3D
morphogen seeds: particles are sampled from a compact random ball, hidden state
is zero except for a sparse opacity/alive core, and no target residual, normal,
signed-distance, color, particle index, or target sample is written into state.
The older `TorusMorphogenDense3d` and `TeapotMorphogenDense3d` modes remain as
diagnostic seed-frame baselines only.

The conditionless local ablation now starts from a stable small-weight local
student with a weak opacity-gradient expansion prior plus a local opacity
controller instead of a fully random update network. The opacity controller uses
only the particle's current opacity state and drives compact sparse growth seeds
toward the visibility target over rollout. It exposes `--motion-gain`,
`--max-update-norm`, `--coverage-gain`, and `--opacity-gain` so geometry,
coverage, and visibility dynamics can be swept independently from the legacy
position-field constants.
Validation reports now include mean surface improvement and improvement ratio.
Torus robustness reports also include final radial and z coverage; this catches
the previous false-positive where particles collapsed onto the inner torus
surface and achieved a low mean surface distance without forming the full
toroid.

Active Bevy catalog inference now exposes:

- the generic `Growing3dGs` preset at `1024` particles for interactive 3D
  viewing

The torus and teapot regression artifacts are still validated with
`ParticleSeed::TorusGrowth3d` / `ParticleSeed::TeapotGrowth3d` and seed
`0x51a7_3d` plus held-out seeds. Both mesh artifacts are hidden from the app
catalog until robust seed-varied strict validation passes. Torus remains hidden
because every compact-growth candidate still fails target coverage, angular
support, and render-density gates. Teapot remains hidden because seed-varied
validation exposed near-threshold held-out failures even though the primary seed
can pass strict render/geometry checks.

The old target-bearing `*_morphogen_3d.bpk` files and standalone field
render-proxy BPKs are retired from `assets/models`. New diagnostic runs should
write into `target/` or `artifacts/` unless they pass the promotion checks
below.

The app-scale promotion/regression harness is:

```bash
scripts/validate_3d_catalog.py
```

It regenerates the teapot and torus validation reports at the app seed and
particle count plus held-out seeds `42` and `99`, checks the latest
local-growth lineage and no-shortcut properties, and keeps trained mesh
artifacts hidden until their strict coverage/render blockers are cleared. The robustness
report now requires color-state emergence and particle-order permutation
consistency across all three seeds before a 3D artifact is considered
app-catalog safe. It also records minimum active seed count, final active count,
newly activated fraction, active growth ratio, worst render loss, minimum
density/color/depth PSNR, minimum target coverage, and seed-perturbation
stability across the seed sweep so static, precolored, brittle, or single-seed
artifacts cannot look catalog-safe through primary-seed render metrics alone.

The active artifact validation reports can be regenerated with:

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
  --extra-seed 99 \
  --output target/uv_torus_growth_3d_catalog_sanity_report.json
```

```bash
cargo run -p burn_automata --release --bin burn_automata -- validate-growth3d \
  --model assets/models/teapot_growth_3d.bpk \
  --target teapot \
  --seed-mode teapot-growth-3d \
  --seed 5351229 \
  --particles 512 \
  --steps 64 \
  --image-size 48 \
  --target-samples 1024 \
  --world-scale 1.44 \
  --gate catalog-sanity \
  --extra-seed 42 \
  --extra-seed 99 \
  --output target/teapot_growth_3d_catalog_sanity_report.json
```

Use `--gate strict` for future promotion candidates. Render validation is now
opacity-aware and matches the Bevy/WGPU Gaussian opacity mapping. At the current
1024-particle Bevy catalog scale, teapot now passes strict validation after the
render-density retime (`density PSNR=11.144`, target coverage=`0.919`,
strict score=`0.000`) and is the only promoted 3D mesh artifact. Torus still
fails compact-growth target coverage, angular support, and render density. Add
`--fail-on-validation` only for future promotion candidates expected to pass the
selected gate.

## Ablations Run

The new ablation command trains a no-position model from random-ball seeds and
validates against mesh rollout gates:

```bash
cargo run -p burn_automata --release --bin burn_automata -- ablate-local-3d \
  --target torus \
  --rows 2048 \
  --steps 512 \
  --training-rounds 8 \
  --rollout-particles 1024 \
  --rollout-steps 8 \
  --rollouts 2 \
  --seed-scale 0.72 \
  --coverage-gain 0.05 \
  --coverage-samples 2048 \
  --density-gain 0.0 \
  --learning-rate 0.0005 \
  --grad-clip-norm 0.25 \
  --model-output /tmp/burn_automata_conditionless_torus_3d.bpk \
  --report-output artifacts/conditionless_torus_3d_ablation_report.json
```

```bash
cargo run -p burn_automata --release --bin burn_automata -- ablate-local-3d \
  --target teapot \
  --rows 2048 \
  --steps 512 \
  --training-rounds 8 \
  --rollout-particles 1024 \
  --rollout-steps 8 \
  --rollouts 2 \
  --seed-scale 0.72 \
  --coverage-samples 2048 \
  --learning-rate 0.0005 \
  --grad-clip-norm 0.25 \
  --model-output /tmp/burn_automata_conditionless_teapot_3d.bpk \
  --report-output artifacts/conditionless_teapot_3d_ablation_report.json
```

Staged local curricula can continue from a previous conditionless-local BPK:

```bash
cargo run -p burn_automata --release --bin burn_automata -- ablate-local-3d \
  --target teapot \
  --base-model target/teapot_front_growth_gain2_train.bpk \
  --coverage-gain 0.40 \
  --coverage-samples 4096 \
  --extent-gain 0.20 \
  --model-output target/teapot_front_continue.bpk
```

The continuation loader rejects position features and shortcut lineage before
training starts, then records `continued-from=...conditionless-local...` in the
output manifest.

`--coverage-samples` decouples target-surface coverage supervision from the
number of rollout rows. It is useful for larger app-scale particle counts, but
the 2026-06-30 probes show that denser coverage samples alone are not a
promotion path: torus candidates improved render loss only by regressing
temporal geometry or active coverage, and teapot denser-coverage continuation
regressed activation at the 512-particle multi-seed gate. After the
opacity-aware render fix, a bounded torus render-refinement at scale `0.54`
also failed to improve coverage or depth enough for promotion.

Results:

| target | final one-step loss | mesh gate | mean surface | max surface | mean color | max opacity error |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| torus, refreshed 8-step rows | `7.185` | failed | `2.315` | `4.466` | `0.710` | `61746` |
| teapot, refreshed 8-step rows | `5.055` | failed | `2.874` | `5.075` | `0.691` | `91907` |
| torus, 32-step rows | `21737.740` | failed | `2.649` | `4.534` | `0.826` | `237019` |

Additional compact-growth-seed ablations from this pass:

| target/config | final one-step loss | mesh gate | mean surface improvement | mean surface | max surface | motion/step | opacity range | render total |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |
| torus, quick stable init | `1.507` | failed | `0.1%` | `0.121` | `0.218` | `3.9e-5` | `2.82..2.85` | `5.344` |
| torus, opacity gain `0.10` | `0.969` | failed | `52.8%` | `0.057` | `0.203` | `2.8e-3` | `16.83..16.99` | `4.756` |
| torus, opacity gain `0.025` | `0.403` | failed | `-245.8%` | `0.419` | `0.517` | `1.0e-2` | `6.10..6.17` | `5.930` |
| torus, render-refined from gain `0.10` | n/a | failed | collapsed coverage | `~0.059` | `~0.171` | `2.8e-3` | high | `4.364` |
| torus, refreshed 16-step compact growth, this pass | `0.186` | failed | `-42.2%` | `0.172` | `1.533` | `6.8e-3..2.7e-2` | `2.76..15.76` | `0.996` |
| teapot, refreshed 16-step compact growth, this pass | `0.190` | failed | `-125.5%` | `0.417` | `1.608` | `6.9e-3..2.7e-2` | `2.59..16.13` | `1.021` |

The latest compact-growth ablation reduced one-step supervised loss, but mesh
surface error worsened across the seed-scale sweep and rendered density PSNR
stayed near `0 dB`. The active catalog now prefers the latest dynamic
local-front artifacts over the older primary-seed render-safe probes because
the viewer should show sparse-core growth rather than static/assigned behavior.
This is covered by core tests: active 3D seeds must be neutral and dynamic,
active BPKs must carry auditable conditionless-local lineage and avoid
position-field, seed-frame, and generic render-proxy shortcut lineage, while a
strict ignored acceptance gate captures the positive criteria required before
these assets can be called solved. The reports now include target-surface coverage: target
samples are checked against their nearest final particle, with mean/max coverage
distance and covered-surface fraction serialized for each rollout case. This
rejects partial-object collapse even when particle-to-surface distance looks
reasonable. The strict report also records final active count, newly activated
inactive particles, newly activated fraction, and active-front radius, so active
catalog artifacts must demonstrate visibility growth beyond the sparse core
instead of only moving pre-visible particles.
| teapot, opacity gain `0.025` | `0.0076` | failed | `40.8%` | `0.110` | `0.299` | `6.6e-3` | `6.12..6.16` | `2.743` |
| torus, sparse-core + clamped render-refined active asset | n/a | failed | n/a | `0.181` | `0.852` | `1.8e-2..3.3e-2` | `8.68..18.56` | `0.929` |
| teapot, older primary-seed render-safe asset | n/a | failed | n/a | not gated here | not gated here | `1.9e-2..3.4e-2` | bounded in catalog sanity | `0.798` |

The torus opacity-gain `0.10` run demonstrates why surface-distance-only
validation is insufficient: it reduces mean surface error, but the final radial
coverage remains near the inner tube instead of spanning the full torus. The
new coverage metrics reject that collapse. The local ablation trainer now has a
small default target-coverage pressure (`--coverage-gain 0.05`) that assigns
target samples to their nearest rollout particle and adds a clamped residual
update; this is an objective term, not target state in the seed. A quick torus
smoke sweep showed small render/coverage improvements but no strict-gate pass,
so it remains an ablation knob. Ablation reports serialize these training gains
next to the rollout-supervision settings for reproducibility. `--density-gain`
is an initializer-only local density-gradient prior; it is recorded in reports
because it changes the starting local controller, not the rollout labels. The
local opacity controller reduces opacity error from roughly `8-9` logits to
about `0.25` logits in the default torus/teapot ablations, but it exposes the
remaining coverage failure: more visible particles worsen rendered density
unless the geometry covers the target. The opacity-stable candidates were not
promoted over the dynamic local-front catalog artifacts.

Latest local-pipeline promotion sweep, app seed `0x51a7_3d`, 512 particles,
64 rollout steps, 48 px opacity-aware render oracle:

| candidate | density gain | coverage gain | app render total | density PSNR | final opacity max | gate | promoted |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| active torus catalog | n/a | n/a | `0.935` | `0.514` | `0.257` | failed | yes, bounded dynamic local-front + guarded refinement |
| torus local default | `0.00` | `0.05` | `3.146` | `-4.921` | n/a | failed | no |
| torus local density prior | `-0.02` | `0.05` | `1.129` | `-0.373` | n/a | failed | no |
| torus local density + coverage | `-0.02` | `0.30` | `1.145` | `-0.437` | n/a | failed | no |
| active teapot catalog | n/a | n/a | `0.983` | `0.147` | `0.193` | failed | yes, dynamic local-front |
| teapot local default | `0.00` | `0.05` | `1.754` | `-2.380` | n/a | failed | no |
| teapot local density prior | `-0.02` | `0.05` | `1.065` | `-0.189` | n/a | failed | no |
| teapot local density + coverage | `-0.02` | `0.30` | `1.067` | `-0.198` | n/a | failed | no |

The latest local-front training pass changes the opacity target from a global
target logit to a contact-local active-front target. This makes the rollout
look like a true growing automaton in time, but the geometry objective is still
too weak to promote:

| candidate | active growth | progressive activation | app render total | density PSNR | extent ratio | strict score | promoted |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| torus local-front, expansion `0.25`, front radius `0.40`, render-refined | `6 -> 492` | passed | `1.052` | `-0.026` | `0.969` max-radius | `1.420` | no, coverage/render fail |
| torus local-front, gain-2 seed + front supervision | `6 -> 501` | passed | `1.513` | `-1.692` | `0.490` max-radius | `1.638` | no, coverage/render fail |
| teapot local-front, gain-2 seed + front supervision | `6 -> 501` | passed | `1.162` | `-0.593` | n/a | `1.390` | no, coverage/render fail |
| teapot staged continuation from local-front, coverage `0.40`, extent `0.20` | `6 -> 488` | passed | `1.177` | `-0.640` | `0.804` max-radius | `1.484` | no, coverage/render fail |
| teapot coverage/extent sweep, coverage `0.80`, extent `0.60`, expansion `0.25`, front radius `0.40` | `6 -> 430` | failed | `0.882` | `0.794` | `1.512` max-radius | `11.280` | no, temporal/surface/render fail |

The rollout trainer now also gates mesh motion, color, and target-coverage
supervision by the same active-front predicate used for opacity. With
`front_motion_gate=true` (the default), far dormant particles receive no mesh
assignment until the local active front reaches them. The validator now also
serializes temporal geometry progress: active-surface mean ratio, target
coverage mean ratio, coverage-fraction delta, and a
`temporal_geometry_progressive` boolean. This keeps the local training target
and acceptance gate aligned with morphogenesis dynamics, but the latest gated
torus probe still collapses spatially: active bounds cover only about `0.32`
of target X extent and `0.49` of target max radius, with target coverage
fraction `0.268`.

A teapot coverage/extent sweep improved app render total from `1.162` to
`0.882` and final active target coverage from `0.270` to `0.949`, but it
activated only `430/512` particles by the 64-step app horizon and introduced
surface outliers (`max_surface=0.719`). It remains a diagnostic showing the
current tradeoff: stronger geometric support improves density coverage but
breaks the progressive/full-growth and bounded-surface requirements.

Continuing the progressive local-front teapot with gentler coverage/extent rows
preserved progressive activation (`6 -> 488`) and bounded surface max
(`0.255`), but render total regressed to `1.177` and target coverage stayed low
(`0.180`). Staged local training is now supported, but the current proxy rows do
not yet solve the support-vs-render tradeoff.

The `retime-growth3d` command can also load an existing local 3D BPK and
replace only the opacity-output row with a blurred-opacity front controller.
This was useful as a diagnostic but not a promotion path: retiming the active
torus catalog model with `--front-gain 2.0` produced local activation
(`6 -> 466`) but regressed app render total to `1.039`, density PSNR to
`0.082`, and still failed progressive activation because full activation did
not occur by the sampled horizon.

Latest render-proxy refinement probes from the active catalog models use
guarded worst-case checkpoint selection over the trainer seed and app seed
`0x51a7_3d`. The selection score combines render loss with active-surface and
target-coverage penalties, then rejects rounds that regress active-surface max
or coverage relative to the initial model on any selection seed. The current
trainer samples render-gradient rows across the full cloud rather than from the
particle-array prefix, which keeps under-covered support regions in the
supervised proxy batch. The mesh-local continuation path also computes
perception features and coverage/front/extent targets on the complete rollout
cloud before extracting spread-out supervised rows; this avoids training sampled
rows under an artificial subset-only neighborhood. Earlier
multi-seed-average probes improved some app-facing render metrics but regressed
another strict blocker, so the guarded trainer now keeps the active baseline
unless a round improves without morphology regression.

The latest trainer defaults to `--training-backend direct-rollout`, which
applies final multi-view render position, opacity, and color adjoints through
stored rollout MLP outputs. The older supervised row-projection path remains
available as `--training-backend proxy`. This is a real gradient path into the
shared update weights. It now propagates recurrent RGB/opacity state adjoints
through fixed-neighborhood SPH state perception, including direct state,
blurred state, and moment-corrected state-gradient feature channels. Particle
positions are also differentiated through direct Euler integration. Position
effects inside perception, changing neighborhood membership, density-gradient
position terms, and render visibility through time are still stop-gradient.
Render-opacity gradients are scaled by `--opacity-gain`; color adjoints are
applied to the visible RGB tail state where the output clamp has nonzero
derivative.

| candidate | app render total | density PSNR | color PSNR | depth PSNR | active max surface | coverage fraction | strict score | promoted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| older torus render-safe catalog | `0.830` | `1.074` | `16.895` | `15.350` | `0.579` | `0.443` | `21.433` | replaced by dynamic local-front asset |
| torus render-refined, multi-seed | `0.818` | `1.112` | `17.364` | `15.858` | `0.454` | `0.432` | n/a | no, coverage regressed |
| torus render-refined, gentle | `0.838` | `0.992` | `17.336` | `16.266` | `0.477` | `0.461` | n/a | no, render total/density regressed |
| torus render-refined, guarded | `0.830` | `1.074` | `16.895` | `15.350` | `0.579` | `0.443` | `21.433` | no new round selected |
| older teapot render-safe catalog | `0.798` | `1.126` | `18.652` | `18.998` | `0.771` | `1.000` | `11.299` | replaced by dynamic local-front asset |
| teapot render-refined, multi-seed | `0.776` | `1.253` | `18.391` | `19.318` | `1.050` | `1.000` | n/a | no, outlier surface regressed |
| teapot render-refined, gentle | `0.780` | `1.215` | `18.713` | `19.779` | `0.962` | `1.000` | n/a | no, outlier surface regressed |
| teapot render-refined, guarded | `0.798` | `1.126` | `18.652` | `18.998` | `0.771` | `1.000` | `11.299` | no new round selected |
| teapot render-refined, latest app-seed retry | `1.014` | `0.089` | `17.997` | `17.455` | n/a | n/a | n/a | no, catalog render floor regressed |
| torus spread-row opacity-aware retry | `0.744` | `1.657` | `16.916` | `13.860` | `0.453` | `0.490` | `10.955` | no, unchanged versus active |
| teapot spread-row opacity-aware retry | `0.617` | `2.197` | `19.041` | `27.047` | `0.411` | `1.000` | `10.780` | no, unchanged versus active |
| torus render-opacity retry | `0.743` | `1.664` | n/a | `13.854` | n/a | `0.489` | `10.955` | no, strict score/depth/coverage/angular support regressed |
| teapot render-opacity retry | `0.617` | `2.199` | n/a | `27.007` | n/a | `0.998` | `10.780` | no, depth PSNR and held-out coverage regressed |
| torus local continuation, stronger coverage | `0.755` | `1.659` | `15.808` | `13.337` | `0.414` | `0.587` | `11.161` | no, temporal geometry regressed |
| torus full-context local continuation | `0.704` | `1.971` | `16.086` | `13.512` | `0.502` | `0.551` | `11.250` | no, coverage/render improved but surface dynamics regressed |
| torus full-context late-horizon local continuation | `0.941` | `0.547` | `17.132` | `13.949` | `0.430` | `0.447` | `1.172` | no, render/coverage regressed |
| torus full-context surface-balanced local continuation | `1.323` | `-1.021` | `17.677` | `13.861` | `0.379` | `0.381` | `21.488` | no, activation/render/coverage regressed |
| torus direct-rollout MLP adjoint | `0.746` | `1.525` | n/a | n/a | n/a | `0.276` | `11.195` | no, tube angular support/coverage/render failed |
| teapot direct-rollout MLP adjoint + opacity continuation | `0.504` | `3.069` | n/a | n/a | n/a | `0.574` | `0.719` | no, coverage/render still below strict gate |
| torus recurrent-state direct adjoint | `0.914` | `0.802` | n/a | n/a | n/a | low | `11.452` | no, angular coverage/render regressed |
| teapot recurrent-state direct adjoint from best diagnostic | `0.504` | `3.070` | n/a | n/a | n/a | `0.574` | `0.719` | no new round selected |
| torus fixed-neighborhood SPH state adjoint | `0.894` | `0.913` | n/a | n/a | n/a | low | `11.488` | no, angular coverage/render failed |
| teapot fixed-neighborhood SPH state adjoint, guarded | `0.575` | `2.485` | n/a | n/a | n/a | low | `1.082` | no, temporal activation/coverage/render failed |
| torus truncated position+SPH state adjoint | `0.865` | `1.075` | n/a | n/a | n/a | low | `11.454` | no, angular coverage/render failed |
| teapot truncated position+SPH state adjoint | `0.565` | `2.572` | n/a | n/a | n/a | low | `1.147` | no, temporal activation/coverage/render failed |

After aligning `train-render3d` selection with the strict-score distance and
temporal activation gate, a short torus probe from the older render-safe catalog model
improved the training-seed render loss from `1.473` to `1.443`, but the held-out
selection score rose to `12.329` and
`selection_morphology_non_regressed=false`, so no checkpoint was selected. The
saved probe validates identically to that older baseline
(`strict_score=21.433`, app render total `0.830`), which is the desired
non-promotion behavior until a candidate improves strict geometry/render terms
and progressive activation without seed-specific regression.

A June 30, 2026 retry of `train-render3d` for teapot from the latest dynamic
local-front candidate with the app seed, 512 particles, 64 steps, 48 px render
oracle, and 1024 target samples selected round `2` and produced
`assets/models/teapot_growth_3d.bpk`. After revalidating at the current
1024-particle interactive catalog scale, it scores render total `0.591`,
density PSNR `2.368`, target coverage `0.180`, and strict score `1.184` at
64 steps. At the viewer cadence (`96` simulation steps via `steps/frame=2`),
coverage improves only to `0.273` and strict score to `1.078`. It is therefore
kept as a hidden regression artifact, not a promoted visible teapot model.

The app-scale validator now records neural color-state emergence explicitly.
The active seed color tail remains neutral (`active_mean_abs=0`,
`active_max_abs=0`, `active_channel_stddev_mean=0`) for both growth seeds, and
the final active tail becomes nonzero and non-uniform. The 96-step teapot report
measures `active_mean_abs=0.117`, `active_max_abs=0.411`, and
`active_channel_stddev_mean=0.072`; the 96-step torus regression report
measures `0.137`, `0.407`, and `0.071`. This keeps the catalog from accepting
precolored or uniform-tint artifacts that do not derive visible color from
local neural dynamics.

The same validator records a permutation-consistency sub-rollout. It shuffles a
256-particle seed cloud, evolves both particle orders for 8 steps, unshuffles
the result, and compares final positions/state. The current teapot artifact
passes with max position/state errors `7.5e-7` / `2.4e-5`; the hidden torus
regression artifact passes with `3.5e-7` / `1.3e-5`. This is an explicit guard
against index-order or assigned-seat shortcuts in the active 3D catalog path.

It also records a seed-perturbation sub-rollout. A 512-particle neutral seed
cloud is jittered by 10% of the growth seed radius, evolved for 32 steps, and
compared with the unperturbed aggregate growth. Current active reports pass this
guard across app seed `0x51a7_3d` and held-out seeds `42`/`99`: teapot keeps
perturbed newly activated fraction above `0.876`, active-count ratio inside
`0.978..1.057`, and peak-motion ratio inside `1.009..1.023`; torus keeps
perturbed newly activated fraction above `0.830`, active-count ratio inside
`0.972..1.061`, and peak-motion ratio inside `1.001..1.020`.

A later guarded refresh from both active catalog models used the trainer seed,
app seed `0x51a7_3d`, and extra selection seeds `42` and `99`. No round beat
the initial multi-seed strict score for either target. The torus BPK was
re-saved with `render-refined-rust:...conditionless-local...` lineage so the
catalog records the latest pipeline review; its weights and app-facing metrics
remain the active bounded dynamic local-front baseline. The teapot weights were
retained because the spread-row opacity-aware retry was numerically unchanged
at app scale and did not improve the strict gate. The render-opacity retries
were likewise retained only as `target/` probes because app-scale comparison
found depth/coverage regressions.

The failure mode is consistent: the local proxy learns nonzero motion but does
not form target geometry. The direct-rollout MLP-output adjoint improves teapot
render score and preserves local growth dynamics. A later state-only recurrence
propagated RGB/opacity gradients through the direct current-state feature, and
the current fixed-neighborhood adjoint additionally propagates through SPH
blurred-state and state-gradient perception features. This still does not solve
support allocation: torus angular coverage fails, and teapot remains below
coverage/render gates after guarded selection. Without gradients through
particle positions inside perception, changing neighborhoods, and rendered
visibility over time, the backend still cannot reliably allocate particles
across the whole target support. Refreshing rollout rows improves short-horizon
loss but does not solve the missing global target signal.

The render-loss harness adds a stricter view-space objective:

```bash
cargo run -p burn_automata --release --bin burn_automata -- render-loss-3d \
  --model assets/models/uv_torus_growth_3d.bpk \
  --target torus \
  --seed-mode torus-growth-3d
```

It renders `xy`, `xz`, `yz`, and isometric orthographic Gaussian splats for the
rollout and target mesh samples, then reports relative density loss, gated
color loss, and depth-moment loss. Current reports:

| model | render gate | total | density PSNR | color PSNR | depth PSNR | final opacity max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| torus active catalog, 512-particle app regression | failed | `0.935` | `0.514` | `19.789` | `14.373` | `0.257` |
| torus hidden regression artifact, 1024-particle app scale | failed | `0.815` | `1.248` | `17.504` | `13.244` | `0.323` |
| torus hidden regression artifact, 1024-particle 96-step cadence | failed | `0.876` | `0.924` | `17.150` | `13.026` | `0.532` |
| teapot hidden regression artifact, 512-particle app regression | failed | `0.983` | `0.147` | `22.680` | `19.665` | `0.193` |
| teapot hidden regression artifact, 1024-particle app scale | old sanity passed, strict failed | `0.591` | `2.368` | `20.063` | `27.605` | `0.348` |
| teapot hidden regression artifact, 1024-particle 96-step cadence | old sanity passed, strict failed | `0.577` | `2.487` | `19.587` | `26.197` | `0.776` |
| torus conditionless-local ablation | failed | `1.492` | `-0.107` | `7.351` | `5.476` | n/a |
| teapot conditionless-local ablation | failed | `1.321` | `-0.046` | `8.399` | `7.813` | n/a |

This confirms that mesh-distance validation and view-space density validation
are different objectives. The blocked teapot artifact defaults to two
simulation steps per rendered frame when loaded directly, so it reaches the
validated 96-step growth
horizon in 48 visible frames. The active and hidden regression models are useful
experiments, but still do not match the target projected alpha distribution.
The conditionless ablations fail both mesh geometry and rendered views.

## Interpretation

The current fixed-row supervised proxy is insufficient for conditionless 3D
mesh morphogenesis. It can label a local state with "move toward the closest
mesh point", but without absolute position features or target-bearing seed
state, symmetric local neighborhoods do not encode which target region a
particle should represent. Longer horizons make this clearer: the policy moves
and changes state, but the repeated dynamics leave the supervised distribution
and blow up.

This does not prove local 3D NPA is impossible. It shows the current training
objective is wrong for that goal. The upstream-style objective should optimize
the rendered result of the whole rollout, letting particles self-assign through
collective dynamics rather than through one-step mesh projection labels.

## Implementation Roadmap

1. Extend the rollout loss harness into training.
   - Done: CPU multi-view 3D density/color/depth render-loss oracle.
   - Done: `train-render3d` analytic render-position-gradient proxy backend for saved BPKs.
   - Done: `train-render3d --training-backend direct-rollout` applies final
     render position/opacity/color adjoints through stored rollout MLP outputs
     and updates the shared NPA weights directly.
   - Done: the direct backend now propagates RGB/opacity state adjoints
     backward through fixed-neighborhood SPH state perception over stored
     snapshots, including direct state, blurred state, and moment-corrected
     state-gradient features.
   - Done: terminal render/coverage position adjoints now flow backward through
     direct Euler integration over stored snapshots.
   - Done: `train-render3d` keeps the best checkpoint by multi-seed selection
     instead of blindly saving the final round; the seed list includes the
     training seed, `--selection-seed`, and deduped `--extra-selection-seed`
     values.
   - Done: selection is now guarded by per-seed active-surface/coverage
     non-regression checks, so lower render loss cannot silently worsen
     morphogenesis metrics.
   - Done: selection score is aligned with the strict growth-validation score,
     including sustained-motion, active-surface, target-coverage, and
     density/color/depth render gates.
   - Done: the proxy can optionally supervise all rollout snapshots instead of
     only the final state and can add a bounded nearest-surface target-coverage
     residual. The 1024-particle torus/teapot experiments improved a few scalar
     losses but still failed strict coverage/render gates, so these are
     diagnostic scaffolds, not solved models.
   - Done: catalog-bound `train-render3d` candidates are first written to a
     temporary `target/` path and must pass strict multi-seed growth validation
     before `assets/models/` is overwritten.
   - Next: 2D image/alpha wrapper around the same density/color objective.
   - Next: native differentiable or WGPU backward path through rollout and
     splat rendering.

2. Make training differentiable or add a proxy gradient path.
   - Preferred: backpropagate through SPH perception, MLP update, integration,
     and splat rendering for truncated rollout windows.
   - Done interim: analytic CPU render-loss gradients to final particle
     positions, projected into supervised one-step update rows.
   - Done interim: trajectory-supervised proxy rows reuse final render/coverage
     corrections at every stored rollout step.
   - Done interim: direct-rollout gradients backpropagate those final render
     adjoints through each stored-step MLP output into model weights.
   - Done interim: recurrent RGB/opacity state adjoints flow backward through
     fixed-neighborhood SPH state-perception features for truncated windows.
   - Done interim: position adjoints flow backward through direct Euler
     integration for truncated windows.
   - Done interim: fixed-neighborhood SPH position-perception adjoints now cover
     blurred-state, state-gradient, density-gradient, density-volume, direct
     position features, and hybrid moment-corrected state-gradient terms. Kernel
     finite-difference tests cover both hybrid and non-hybrid position adjoints.
   - Done interim: `--perception-position-gain` damps those sharp SPH position
     derivatives; unit gain regresses morphology, while the default `0.05`
     improves short render probes without passing strict gates.
   - Done ablation: `--direct-selection-seed-training` averages clipped
     direct-rollout SGD deltas over the selection seed set. It did not improve
     the 1024-particle torus/teapot strict probes, so it remains opt-in.
   - Done ablation: `--trajectory-render-gain`/`--trajectory-render-samples`
     inject render/coverage adjoints at sampled rollout snapshots. The first
     1024-particle probes regressed strict score (`teapot=1.105`,
     `torus=11.314`), so the default remains terminal-render BPTT.
   - Done ablation: `--full-coverage-adjoint` applies target-coverage pressure
     to all active particle rows rather than only sampled render-gradient rows.
     It regressed the same short strict probes (`teapot=1.119`,
     `torus=11.326`), so it remains opt-in.
   - Done ablation: `--coverage-mode soft-chamfer` adds detached soft
     target-sample assignment for target coverage, and
     `--coverage-repulsion-gain`/`--coverage-repulsion-radius` add optional
     mesh-tangent particle spread pressure. Gentle repulsion improves the
     16-round 1024-particle teapot strict score to `0.854` and the torus score
     to `11.161`, but neither passes strict validation. A teapot continuation
     with stronger coverage/opacity supervision reaches 1024-particle strict
     score `0.825` and target coverage `0.501`; at 2048 particles it reaches
     coverage `0.769` and only fails the render gate. Torus remains stuck at
     `6/16` tube bins even at 2048 particles. Stronger repulsion regresses
     torus, so this remains an ablation and no artifact is promoted.
   - Done bug fix: mesh target samples are now area-weighted low-discrepancy
     samples instead of ordered face centroids. With the corrected sampler,
     teapot target coverage passes at 1024 particles (`0.980`) and the best
     density-weighted 2048-particle continuation reaches strict score `0.527`
     but still fails render density. Torus retraining against the corrected full
     torus target still fails strict validation, so the remaining torus blocker
     is real full-surface/tube support rather than only a metric artifact.
   - Done bug fix: random mesh surface sampling now uses the same area-weighted
     target sampler as deterministic validation. This removes the last
     face-count-biased mesh seed/near-surface sampler in the generic target
     path.
   - Done ablation: `--coverage-mode sliced-ot` adds balanced sliced
     optimal-transport coverage pressure. It is permutation-invariant because
     particles are matched by spatial projection rank, not array index. Unit
     coverage tests verify that separated target modes pull particles apart
     rather than averaging them into the middle. In 1024-particle torus
     rollouts, however, Sliced OT regressed strict score (`11.535`) when used
     in local rollout continuation and did not improve terminal render-adjoint
     training (`11.506`), so it remains an opt-in diagnostic.
   - Done diagnostic: `retime-growth3d --alpha` can isolate motion scale from
     front retiming via `--skip-front-retime`. Pure torus `alpha=1.1` improves
     strict score from `11.502` to `11.454`; the best alpha/front continuation
     probe reached `11.351` and density PSNR `0.819`, but still fails target
     coverage, torus angular coverage, and render. Teapot alpha-only retiming
     regresses strict score, so no artifact is promoted.
   - Remaining: full truncated BPTT through changing neighbor membership,
     position-dependent render visibility/occlusion over time, and a stronger
     rollout/render objective that improves target coverage without damaging
     growth morphology.

3. Use a training pool.
   - Random rollout horizons.
   - Periodic pool reset to neutral seeds.
   - Damage/regeneration cases.
   - Stochastic update probability sweep.

4. Keep symmetry-breaking explicit and minimal.
   - A single active seed marker or learned initializer is acceptable.
   - Preloaded per-particle residual/color/target coordinates are not.
   - Arbitrary oriented shapes require either a fixed canonical frame or a
     learned condition/initializer that defines the frame.

5. Harden kernels for training throughput.
   - GPU-resident sorted-cell neighbor storage.
   - Reusable buffers across rollout steps and training batches.
   - Differentiable SPH backward kernels for state, density, and position
     gradients.
   - Render-loss buffers shared with the Bevy/GSplat bridge for validation.

6. Promote only validated artifacts.
   - Catalog replacement requires saved `.bpk` rollout gates to pass.
   - Local models must pass seed-scale and particle-count sweeps.
   - Reports must include image/render PSNR or SSIM where applicable.

## Latest 24-Channel Rollout Ablations

The 2026-06-30 h128 sweep moved the local 3D path closer to 2D-style
morphogenesis but did not produce a strict-pass artifact. No model from this
sweep was promoted to `assets/models`.

Code changes from the sweep:

- `NpaConfig::growing_3dgs()` now defaults to `24` state channels, matching the
  higher-capacity 3D NPA setting and leaving actual hidden state after
  coordinate, opacity, normal, signed-distance, and color slots.
- `ablate-local-3d` exposes `--color-gain` and `--aux-state-gain`, so nearest
  mesh-projection color/coordinate/normal/signed-distance scaffolding can be
  disabled for generic local morphogenesis experiments.
- `--coverage-mode gap-farthest` adds a target-surface coverage signal that
  preserves worst uncovered-bin residuals instead of averaging symmetric target
  residuals away.
- Sliced-OT coverage now uses 13 projection directions rather than only axes
  and body diagonals.

Best diagnostic results:

| candidate | validation horizon | render total | density PSNR | coverage fraction | support notes | decision |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `torus_24_sliced_no_projection_r8` | 128 | `0.732` | `1.537` | `0.184` | full major ring, `11/16` tube bins | reject: low coverage/render |
| `torus_24_sliced_no_projection_r8` | 256 | `0.648` | `2.207` | `0.313` | `15/16` tube bins, but overscaled extents and color below catalog sanity | reject: not stable attractor |
| `torus_24_sliced_h128_stable_r8` | 128 | `0.679` | `1.824` | `0.180` | bounded opacity, still under-covers tube/support | reject: strict failures remain |
| `torus_24_sliced_h128_stable_r8`, 2048 particles | 128 | `0.586` | `2.494` | `0.418` | full ring/tube bins and good extents, but particles remain off-surface | reject: surface/render gates fail |
| `torus_24_sliced_h128_stable_r8` | 256 | `0.642` | `2.146` | `0.314` | better support but surface mean/tail and render fail | reject: not strict-safe |
| `teapot_24_sliced_h128_surface_r8` | 128 | `0.445` | `3.697` | `0.985` | full distributed support, overscaled surface tails | reject: render/tail/temporal fail |
| `teapot_24_sliced_h128_contained_r8` | 128 | `0.517` | `2.972` | high enough for catalog sanity | better controlled, worse strict score | reject: strict fail |

Interpretation:

- Torus is no longer static or a per-index parking-lot model in the best
  diagnostic runs: it grows from a neutral sparse core, covers the full major
  ring, and reaches most tube bins at longer horizons. It still does not form a
  strict-quality torus because support remains sparse and the attractor keeps
  changing between 128 and 256 steps.
- Teapot demonstrates that the generic mesh path can learn broad distributed
  support from the canonical mesh (`0.985` active target coverage at 1024
  particles/128 steps), but the current proxy rows overscale the surface and do
  not stabilize opacity/render density over longer rollouts.
- Direct render refinement on the h128 torus candidate did not select any
  round; selection coverage stayed around `0.14-0.16`, so the current direct
  adjoint is not yet a replacement for native differentiable rollout/render
  training.

The next backend step is not another seed-radius or nearest-projection tweak.
It is a native rollout objective with truncated BPTT through local perception,
multi-view density/color/depth losses at multiple rollout times, and a target
coverage/support term that is optimized directly rather than distilled into
one-step projection rows.

## 2026-06-30 Static-Teacher And Uniformity Pass

Two concrete fixes landed after the h128 sweep:

- `train-*-morphogen3d --training-mode rollout-local` no longer distills from a
  residual-state teacher when using neutral sparse growth seeds. That teacher
  only moves when residual channels are preloaded, so it produced effectively
  static rollout-local targets. Rollout-local mode now uses the same sparse
  growth seed plus mesh rollout objective family as `ablate-local-3d`.
- Target coverage repulsion is now available outside soft-chamfer. Hard-nearest,
  gap-farthest, and sliced-OT coverage can add a normalized surface-tangent
  particle repulsion term, which directly targets the clumping seen in render
  density failures.

Validation added:

- `rollout_local_growth_seed_uses_mesh_objective_not_static_residual_teacher`
  proves the old residual teacher has zero position target from neutral growth
  seeds while the fixed mesh objective produces nonzero motion targets.
- `surface_tangent_repulsion_separates_close_surface_particles` proves the new
  repulsion separates close particles along the target tangent plane.

Latest diagnostic continuations:

| candidate | validation | render total | density PSNR | target coverage | assigned particle fraction | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `teapot_corrected_sampler_density_weighted_r8` | 1024p / 64 steps | `0.359` | `4.566` | `0.972` | `0.375` | baseline: strict render fail |
| `teapot_sliced_repel_r4` | 1024p / 64 steps | `0.353` | `4.650` | `0.974` | `0.385` | improved but still strict render fail |
| `teapot_sliced_repel_r8` | 1024p / 64 steps | `0.362` | `4.528` | `0.972` | `0.376` | reject: repulsion too strong/continued too far |
| `torus_24_sliced_h128_stable_r8` | 2048p / 128 steps | `0.586` | `2.494` | `0.418` | n/a in older report | baseline: strict support/render fail |
| `torus_sliced_repel_r4` | 2048p / 128 steps | `0.566` | `2.651` | still below strict gate | tracked in report | improved but still strict support/render fail |

The change is useful but not sufficient. Teapot already has credible local
growth dynamics and broad target coverage, but render density remains far below
the strict `10 dB` density PSNR gate. The blocker is now surface distribution:
coverage is high, yet too few active particles carry target samples, so
multi-view splats remain under-dense or uneven. Torus benefits modestly from
the same anti-clumping force but still fails support coverage and render.

No artifact from this pass was promoted to `assets/models`. The shipped 3D
entries remain hidden regression artifacts until strict render/support gates
pass.

## 2026-06-30 Guarded Geometry Refinement Pass

This pass added two training-safety mechanisms for local mesh rollout rows:

- Zero-assignment active particles can receive a bounded relocation update
  toward uncovered target surface gaps. The update is mesh-generic and depends
  on current particle position, not particle index.
- `ablate-local-3d --preserve-opacity-update` can preserve the model's current
  opacity output while optimizing geometry/color rows. The ablation path enables
  this automatically when both direct opacity gains are zero, preventing
  geometry-only refinement from silently teaching an opacity-to-zero target.

Validation added:

- `surface_gap_relocation_moves_redundant_particles_to_uncovered_regions`
  proves duplicate active particles can be redirected toward uncovered mesh
  support while respecting the update clamp.
- `mesh_local_rollout_can_preserve_opacity_update_targets` proves preserved
  opacity rows copy the model's current opacity update.

Guarded teapot continuation from the hidden catalog artifact improved render
metrics without breaking catalog sanity, but it is still not a strict artifact:

| candidate | validation | render total | density PSNR | target coverage | assigned particle fraction | opacity max | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `assets/models/teapot_growth_3d.bpk` | 1024p / 64 steps | `0.3435` | `4.764` | `0.930` | `0.328` | `0.348` | hidden baseline; strict render fail |
| `teapot_asset_geometry_refine_r20` | 1024p / 64 steps | `0.3360` | `4.863` | `0.936` | `0.343` | `0.260` | best diagnostic; strict render fail |
| `teapot_asset_geometry_preserve_r4` | 1024p / 64 steps | `0.3440` | `4.758` | `0.929` | `0.337` | `0.276` | safer opacity-preserve guard; did not improve render |

Torus remains substantially behind teapot:

| candidate | validation | render total | density PSNR | target coverage | assigned particle fraction | opacity max | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `assets/models/uv_torus_growth_3d.bpk` | 2048p / 128 steps | `1.507` | `-1.700` | `0.128` | `0.162` | `1.525` | hidden baseline; strict support/render fail |
| `torus_geometry_refine_r4_active_op005` | 2048p / 128 steps, scale `0.54` | `0.540` | `2.880` | `0.436` | `0.291` | `1.064` | best diagnostic; strict support/render fail |

The teapot diagnostic is a measurable improvement, but the strict render gate
requires density PSNR `>=10 dB`; the best teapot diagnostic is still `4.86 dB`.
The torus failure is more fundamental: coverage/support and temporal geometry
progress both fail, so opacity retiming alone is not a valid fix.

No artifact from this pass was promoted to `assets/models`.

## Next Acceptance Gate

A local 3D artifact can replace the current position-field catalog model only
when it satisfies all of the following:

- `position_features=false`.
- Seed mode is random ball or single/minimal active seed.
- No target residual/color initialized in state.
- Finite rollout over at least 64 steps.
- Mesh mean surface distance comparable to the current position-field baseline.
- Target-surface coverage mean/max distance and covered fraction comparable to
  the target renderer, not just particle-to-nearest-surface distance.
- Rendered multi-view SSIM/PSNR within an agreed tolerance of the target
  renderer.
- Stable behavior across 512, 2048, and 8192 particles and seed-scale sweeps.

## 2026-06-30 Substrate-Seed And Row-Sampling Pass

This pass tested a more 2D-NCA-like 3D substrate seed. Instead of placing every
particle in the compact growth ball, the new explicit
`TorusSubstrateGrowth3d`/`TeapotSubstrateGrowth3d` seed modes keep only a sparse
active core near the origin and distribute dormant neutral particles through a
normalized 3D domain. The compact `TorusGrowth3d`/`TeapotGrowth3d` seed modes
remain available and are still required for existing hidden catalog artifacts.

Code changes:

- `ablate-local-3d` now accepts `--seed-mode`, and new reports distinguish
  compact `random-ball` lineage from substrate lineage.
- Mesh rollout row selection now keeps deterministic spread rows and also
  prioritizes rows with large target-update norms. This prevents sparse front
  or material rows from being drowned by zero-update dormant substrate rows.
- Mesh-local opacity targets are surface-aware: particles near the target
  surface receive positive material opacity pressure, while off-surface active
  material is suppressed. This is target-generic and uses nearest-surface
  projection distance, not torus or teapot-specific labels.
- Substrate seeds use a less-negative dormant opacity logit (`-4`) than compact
  seeds (`-8`) so they behave more like a dormant canvas than dead particles.

Validation added:

- `growth_3d_substrate_seed_keeps_sparse_active_core_in_dormant_domain`
  verifies sparse active substrate seeds and neutral dormant domain particles.
- `mesh_rollout_row_indices_keep_sparse_high_signal_rows` verifies front/material
  rows are retained by rollout row sampling.
- `mesh_opacity_targets_surface_material_instead_of_whole_domain` verifies
  surface material grows while off-surface substrate opacity is suppressed.
- Existing front-gating tests continue to verify far dormant rows do not receive
  local-front target motion before the front reaches them.

Experimental results:

| candidate | seed | validation | render total | density PSNR | active growth | target coverage | strict score | decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `torus_substrate_surface_r8` | substrate | 1024p / 96 steps | `0.911` | `0.583` | `28 -> 48` | `0.288` all-particle coverage, low active coverage | `72.041` | reject: front does not propagate |
| `torus_substrate_front_r8` | substrate | 1024p / 128 steps | `0.902` | `0.637` | `28 -> 121` | low active coverage | `73.211` | reject: front still too weak |
| `torus_substrate_logit_front_r12` | substrate | 1024p / 160 steps | `0.939` | `0.443` | `28 -> 1` | `0.0` active coverage | `103.586` | reject: aggressive front destabilizes |
| `torus_compact_rows_surface_r12` | compact | 1024p / 96 steps | `1.362` | `-1.280` | `28 -> 1011` | `0.082` | `24.778` | reject: active cloud remains off-surface |
| `torus_compact_surface_opacity_r12` | compact | 1024p / 96 steps | `1.203` | `-0.731` | `28 -> 1012` | still low | `24.223` | reject: surface opacity does not fix support |
| `teapot_rows_surface_direct_r8_op003` | compact | 1024p / 96 steps | `0.361` | `4.579` | `28 -> 1020` | `0.995` | `0.542` | best diagnostic, still strict render fail |
| `teapot_rows_surface_op003_direct_r4` | compact | 1024p / 96 steps | `0.360` | `4.587` | `28 -> 1021` | high | `0.541` | best diagnostic, no promotion |

Interpretation:

- The substrate seed is the right structural direction for a particle analogue
  of a dormant 2D NCA canvas, but the current single opacity channel is doing
  two jobs: local liveness/front propagation and render material opacity. The
  substrate experiments show this coupling is the blocker. If opacity is used
  as liveness, the whole domain becomes visible; if it is trained as material,
  the front stalls.
- Compact teapot remains a credible local-growth diagnostic and improved
  slightly with high-signal row sampling plus guarded opacity bias. It still
  fails the strict density PSNR gate by a wide margin (`4.59 dB` vs. `10 dB`).
- Compact torus remains unresolved. It activates reliably but does not allocate
  material around the tube surface, and surface-aware opacity does not repair
  the topology/support failure.

No model from this pass was promoted to `assets/models`. The next architecture
step should separate hidden liveness/front state from render material opacity
for 3D substrate training, then train render opacity/color from multi-view
losses while validating local liveness propagation independently.

## 2026-06-30 Liveness/Material Opacity Split

This pass implemented the liveness/material split proposed above.

Code changes:

- State channel `3` remains the hidden liveness/front channel used by active
  masks, temporal activation metrics, front coherence, and growth propagation.
- State channel `8` is now the 3D render material-opacity channel when the
  state layout is large enough. Smaller legacy layouts fall back to channel `3`.
- CPU render loss, finite-difference opacity gradients, direct render-training
  adjoints, WGPU Gaussian buffer writes, and Bevy CPU fallback Gaussian writes
  now use the material-opacity channel.
- 3D seeds initialize active-core material opacity at `0` and dormant material
  opacity at `-4`, while compact liveness can still initialize dormant particles
  at `-8`. This keeps dormant material trainable without treating it as active.
- The WGPU and CPU reference Euler steps clamp both liveness and material
  opacity logits.
- The local growth student now includes a small liveness-driven material
  controller, so material visibility can move with a local growth front instead
  of relying only on supervised row fitting.
- Render minimum opacity was reduced from `0.05` to `0.001` so dormant particles
  do not create an unavoidable low-alpha density blob in render loss or Bevy
  Gaussian output.

Regression coverage:

- `growth_3d_seed_positions_are_stratified_inside_expected_radii` now verifies
  separate liveness/material seed logits.
- `growth_3d_substrate_seed_keeps_sparse_active_core_in_dormant_domain` now
  verifies substrate liveness can be less negative while dormant material stays
  low.
- `render_position_gradient_matches_density_finite_difference` now probes the
  material-opacity channel.
- `local_growth_student_opacity_controller_expands_sparse_growth_front` now
  verifies material opacity rises during local rollout, not just liveness.
- CPU and WGPU kernel tests pass with the two-channel clamp.

Experimental results after the split:

| candidate | seed | validation | render total | density PSNR | active growth | target coverage | strict score | decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `teapot_material_controller_compact_r8` | compact | 1024p / 64 steps | `0.602` | `2.309` | `28 -> 1022` | `0.850` | `0.769` | reject: strict density only |
| `teapot_material_controller_render_r8` | compact + direct render | 1024p / 64 steps | `0.599` | `2.325` | `28 -> 1021` | high | `0.768` | reject: render refinement barely moves density |
| `teapot_material_controller_gain016_r4` | compact | 1024p / 64 steps | `0.445` | `3.667` | `28 -> 1024` | `0.890` | `10.633` | reject: activation completes too early |
| `teapot_material_controller_gain016_slowfront_r8` | compact | 1024p / 64 steps | `0.433` | `3.795` | `28 -> 1022` | `0.890` | `0.621` | best diagnostic; strict density only |
| `teapot_material_controller_fullgrad_r2` | compact + full render gradients | 1024p / 64 steps | `0.423` | `3.904` | `28 -> 1021` | high | `0.610` | best render diagnostic; strict density only |
| `teapot_material_controller_gain024_slowfront_r6` | compact | 1024p / 64 steps | `0.454` | `3.594` | `28 -> 1023` | high | `0.641` | reject: worse than gain `0.16` |
| `torus_material_controller_compact_r8` | compact | 1024p / 64 steps | `1.094` | `-0.268` | `28 -> 1022` | `0.044` | `31.762` | reject: tube support/topology failure |

Interpretation:

- The split fixes a real architectural bug. Teapot now passes every non-render
  strict morphology gate under the best diagnostic candidate:
  `teapot_material_controller_gain016_slowfront_r8`.
- The remaining teapot failure is render density. Particle alpha support is
  still below the target render support (`~0.198` vs. `~0.241` nonzero alpha
  fraction at 1024 particles), and direct render refinement only improves
  density marginally. A full-gradient 1024-particle render refinement improved
  density only from `3.795 dB` to `3.904 dB`, so sparse gradient row selection is
  not the primary blocker. The current direct trainer is therefore not yet an
  effective density optimizer.
- Torus remains the stronger negative result. The model covers the major ring
  angle but not the minor tube angle (`tube_coverage_fraction ~= 0.25`), which
  is consistent with a local symmetry/topology ambiguity rather than a material
  opacity bug.
- No artifact from this pass passes strict promotion. The best teapot diagnostic
  is suitable for further experimentation but was intentionally not copied to
  `assets/models`.

Next steps:

- Replace the current render-proxy/direct trainer with a stronger differentiable
  density objective that can backpropagate dense image-space residuals to all
  material particles, not only a sparse gradient subset.
- Add a local 3D symmetry-breaking/morphogen channel objective for torus-like
  genus/topology so the model can distinguish tube angle without absolute
  position or per-index assignment.
- Revisit substrate seeds after the material split with an explicit front-speed
  schedule. The split removes the biggest substrate blocker, but the current
  front still propagates too slowly through a full dormant volume.

## 2026-06-30 Direct-Step Search and Material-Only Refinement

This pass added two guarded render-refinement tools and ran them against the
best teapot and torus diagnostics. Neither artifact was promoted.

Code changes:

- `train-render3d --direct-line-search` now evaluates multiple effective SGD
  step scales for direct-rollout render training and keeps the no-op candidate
  unless a candidate improves the existing strict multi-seed selection score
  without morphology regression.
- The selected scale is written to each render training history row as
  `train_step_scale`. A scale of `0` means the line search intentionally kept
  the current model for that round.
- Line search is opt-in because strict scoring evaluates additional rollouts and
  render losses for every candidate scale.
- `train-render3d --direct-material-output-only` masks direct-rollout gradients
  down to the final material-opacity output row. This keeps hidden features,
  motion rows, liveness rows, and color rows fixed during a render-density
  refinement stage.
- `retime-growth3d --material-opacity-bias` now offsets the separated material
  opacity update output instead of the liveness/front output.
- `material_output_only_gradients_freeze_hidden_and_motion_rows` guards the
  material-only mask.
- `material_opacity_bias_retime_only_offsets_material_output_bias` guards the
  material-opacity retime path.

New diagnostics:

| candidate | validation | render total | density PSNR | selected scales | strict score | decision |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| `teapot_line_search_r1` | 1024p / 64 steps | `0.421` | `3.928` | `4, 1` | `0.607` | reject: strict density only |
| `teapot_material_only_line_search_r1` | 1024p / 64 steps | `0.424` | `3.896` | `256, 1, 0` | `0.610` | reject: material-only refinement is stable but weaker |
| `teapot_line_search_r1_matbias_055` | 1024p / 64 steps | `0.186` | `7.812` | retime material bias `0.55` | `0.219` | reject: best strict-compatible density retime; still below 10 dB |
| `teapot_line_search_r1` | 2048p / 64 steps | `0.416` | `3.971` | eval only | n/a | density gap is not a 1024-particle artifact |
| `teapot_line_search_r1` | 4096p / 64 steps | `0.424` | `3.894` | eval only | n/a | more particles do not recover target density |
| `torus_coverage_topology_r1` | 1024p / 64 steps | `1.129` | `-0.407` | local rollout continuation | `31.760` | reject: same tube/topology failure |

Interpretation:

- Direct line search finds small strict-compatible teapot improvements, but the
  improvement is marginal. The strict failure remains only `render_loss_passed`.
- Material-output-only refinement is useful as a safety mechanism, but it does
  not by itself solve render density. The current local growth hidden state does
  not contain enough separable material-support signal for a final-layer-only
  adjustment to recover the target projection.
- Material opacity bias retiming shows the render-density gap is partly opacity
  amplitude. Bias `0.55` raises density PSNR to `7.812 dB` while preserving the
  non-render strict gates. Larger biases can raise opacity further but begin to
  trip temporal activation or color/catalog sanity, and still do not reach the
  strict 10 dB density render gate.
- Evaluating the same teapot candidate at 2048 and 4096 particles does not
  materially improve density PSNR, so the failed 1024-particle validation is not
  just a sampling floor.
- Stronger generic sliced-OT coverage did not fix torus tube support. The model
  still covers the major ring angle while missing most of the minor tube angle,
  which reinforces that the current conditionless local objective lacks a
  sufficient symmetry-breaking/morphogen mechanism for genus/topology.

Current conclusion:

The 3D path is correctly guarded and dynamic, but it is still not a successful
general conditionless 3D morphogenesis trainer. Teapot is a near-positive
diagnostic for growth dynamics with insufficient render density. Torus remains a
negative topology diagnostic. The next backend change should be a true
rollout-level differentiable render/density training objective with learned
local morphogen/state channels, not more projection-target tuning.

## 2026-06-30 Seed Scaffold, Render Calibration, and Promotion Status

This pass made the current 3D state explicit and moved only the artifact that
passes the strict gate into the visible catalog.

Code changes:

- Compact 3D growth seed modes initialize state channels `0..2` with a
  normalized seed-frame coordinate scaffold. Strict seed validation ignores only
  those scaffold channels; residual, normal, signed-distance, color, opacity,
  and other target-bearing seed state remain rejected.
- Surface-gap relocation is now independent from tangent repulsion, can run with
  soft Chamfer coverage, is normal-aware, and uses a larger uncovered-patch
  budget. A later regression fix allows a gap to fall back to an over-assigned
  active donor when every particle already owns target samples, instead of
  dropping the uncovered patch.
- The 3D CPU render oracle default splat footprint is `sigma=2.5`, matching the
  calibrated density support used for strict local-growth validation.
- `assets/models/teapot_growth_3d.bpk` was replaced by the strict-passing
  `teapot_line_search_r1_matbias_055` artifact and made visible in the Bevy
  catalog. Torus remains hidden/blocked.

Strict validation at the current 1024-particle promotion scale:

| candidate | validation | render density PSNR | target coverage | topology/support | strict score | decision |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| `assets/models/teapot_growth_3d.bpk` | 1024p / 64 steps / `sigma=2.5` | `11.144` | `0.919` | n/a | `0.000` | promoted and visible |
| `torus_seedcoord_expand_r1` | 1024p / 96 steps / `sigma=2.5` | `-2.926` | `0.184` | ring `1.000`, tube `0.813`, joint `0.367` | `23.727` | reject |
| `torus_seedcoord_expand_r1`, smaller scale | 1024p / 96 steps / `seed_scale=0.54` | `2.359` | `0.253` | ring `1.000`, tube `0.938`, joint `0.385` | `23.012` | reject |
| `torus_seedcoord_expand_direct_r1` | direct render continuation | `-0.802` | `0.184` | unchanged topology, no selected round | `23.514` | reject |
| `torus_gapfallback_surface_r1` | gap-fallback surface continuation | `-0.383` | still below gate | surface tail/render fail | `24.229` | reject |

The teapot result is a real positive for this code path: it is local
conditionless lineage, uses no absolute position features, grows from a sparse
compact seed, passes permutation and seed-perturbation checks, and passes the
strict 1024-particle growth/render gate after loading from disk.

The torus result is still a negative topology result. The best diagnostic grows
progressively, covers the full major ring, and reaches most tube bins at some
scales, but it overshoots the surface and leaves too much of the continuous
target uncovered. Direct render refinement improved image density only when it
regressed morphology, so guarded selection correctly kept the baseline. The gap
fallback is a valid generic coverage fix, but it did not solve torus.

Current conclusion:

- No 3D mesh artifact is currently promoted in the visible Bevy catalog under
  seed-varied robust strict validation.
- Torus remains the strongest regression target for conditionless 3D topology
  and genus formation.
- Teapot remains a useful positive diagnostic for local-front mesh growth, but
  it needs robust multi-seed retraining/selection before catalog promotion.
- The next meaningful training change should preserve the seed-frame scaffold
  as a morphogen coordinate and train a rollout-level surface/render objective
  that covers continuous mesh support without nearest-surface collapse or
  per-index target assignment.

## 2026-06-30 Compact Torus Topology Ablation Update

This pass tested whether the remaining torus blocker was a coverage residual
implementation bug, an expansion-prior tuning issue, or missing rollout-level
optimization. No torus artifact passed promotion and no torus artifact was
copied into `assets/models`.

Code changes:

- `gap-farthest` coverage now balances uncovered surface bins across available
  donors instead of letting multiple uncovered bins collapse into one donor row.
  The implementation keeps the rule mesh-generic: it uses target surface samples,
  current particle positions, assignment counts, and bounded relocation updates;
  it does not use torus angles or particle indices.
- A regression test verifies that separated uncovered target modes assign
  rightward gap updates to multiple donor particles rather than cancelling into a
  symmetric no-op.

Strict torus diagnostics at 1024 particles, compact `torus-growth-3d` seed,
96 rollout steps, `sigma=2.5`, and seeds `5904189,5351229,42`:

| candidate | density PSNR | target coverage | torus support | strict score | decision |
| --- | ---: | ---: | --- | ---: | --- |
| `torus_seedcoord_expand_r1` | `-2.926` | `0.184` | ring `1.000`, tube `0.813`, joint `0.367` | `23.727` | reject baseline |
| `torus_balanced_gap_farthest_r1` | `-0.418` | `0.159` | ring `1.000`, tube `0.563`, joint `0.313` | `23.687` | reject; tiny score gain, worse coverage |
| `torus_surface_anchor_r1` | `-0.474` | `0.155` | ring `1.000`, tube `0.563`, joint `0.323` | `24.137` | reject; projection term regressed coverage |
| `torus_opacity_coverage_r1` | `-0.592` | `0.188` | ring `1.000`, tube `0.875`, joint `0.326` | `24.992` | reject; opacity target did not fix density/coverage |
| `torus_project_coverage_late_r1` | `-0.261` | `0.154` | ring `1.000`, tube `0.688`, joint `0.310` | `24.246` | reject; late projection still overshoots |
| `torus_lowexp_compact_r1` | `2.239` | `0.067` | ring `1.000`, tube `0.250`, joint `0.208` | `31.423` | reject; near-surface but under-expanded |
| `torus_midexp_compact_r1` | `0.651` | below gate | under-expanded tube support | `33.850` | reject |
| `torus_coordstate_compact_r1` | `-1.944` | `0.101` | ring `1.000`, tube `0.313`, joint `0.273` | `25.230` | reject; stronger coordinate-state loss was insufficient |
| `torus_direct_tiny_lr_probe` | `-1.239` | below gate | strict score `24.388` | `24.388` | reject; direct step improved hard score slightly but worsened render |
| `torus_direct_tiny_lr_probe_nonregression` | `0.812` | below gate | selected round `none`; render loss unchanged at `0.877862` | `33.054` selection score | reject; render-nonregression correctly keeps no-op |

Interpretation:

- Balanced gap-farthest is a valid generic coverage fix, but it is not the
  missing topology mechanism.
- High expansion reaches more of the torus tube but overshoots the surface and
  fails temporal geometry. Low expansion stays near the surface and improves
  render density, but it does not expand to the continuous target support.
- The direct render backend is wired but not yet a reliable optimizer for this
  case. Raw direct steps worsen render loss; tiny line-search steps can improve
  the strict hard score while still worsening density PSNR. The line search now
  requires render-loss and density-PSNR nonregression, and the same tiny probe
  correctly rejects all candidate updates and keeps the no-op model. This makes
  the training selection safe, but it also confirms that the current direct
  gradients are not yet finding useful torus updates.
- The current one-step rollout-row surrogate remains underdetermined for compact
  conditionless torus growth. It can grow locally and cover the major ring angle,
  but it cannot reliably allocate particles across the minor tube angle without
  a stronger learned morphogen/state objective or a native differentiable
  rollout render loss.

Current blocker:

No 3D mesh artifact is promoted under seed-varied robust strict validation.
Torus remains the active negative diagnostic for fully local conditionless 3D
topology formation. The next required backend change is not more gain tuning; it
is a rollout-level optimizer with dense multi-view render/density gradients and
an explicit learned morphogen-coordinate loss that is generic over mesh targets.

## 2026-06-30 Seed-Varied Robustness And Normal-Coverage Update

This pass fixed two training/validation issues that made earlier 3D results too
optimistic:

- `TorusGrowth3d` / `TeapotGrowth3d` stratified seed positions now depend on the
  requested seed while preserving the same radial activation curriculum. Before
  this fix, held-out seed validation replayed the same compact point cloud.
- `run_supervised_training` now restores the best-loss checkpoint observed
  during training instead of always leaving the model at the final SGD step.
  This prevents local mesh ablations from returning a regressed final model
  after a better intermediate checkpoint.

The seed-varied teapot recheck changed the promotion status:

| candidate | robust result | density PSNR range | failure |
| --- | --- | ---: | --- |
| `assets/models/teapot_growth_3d.bpk` | reject | `10.29..10.87` dB | primary seed missed temporal activation; seed `99` had a small surface-tail failure |
| `teapot_seedvar_opbias003` | reject | `10.20..10.69` dB | held-out seed `99` surface-tail failure |
| `teapot_seedvar_local_refine_r4` | reject | `9.95..10.61` dB | held-out seed `42` density fell below the strict render gate |

The torus normal-coverage ablation added a generic surface-normal-bin coverage
term. It uses only target surface samples, target normals, projected active
particle normals, and donor relocation; no torus angles or particle indices are
used.

| candidate | density PSNR | target coverage | torus support | strict score | decision |
| --- | ---: | ---: | --- | ---: | --- |
| `uv_torus_asset_seedvar_strict` | `0.334` | `0.108` | ring `1.000`, tube `0.313`, joint `0.289` | `31.643` | reject baseline |
| `torus_normalbin_gap_r4` | `0.335` | `0.109` | ring `1.000`, tube `0.313`, joint `0.297` | `31.618` | reject; normal bins too weak as continuation |
| `torus_normalbin_scratch_r8` | `3.131` | `0.316` | ring `1.000`, tube `1.000`, joint `0.443` | `49.228` | reject; proves normal bins can recover tube support but opacity/surface blow up |
| `torus_normalbin_preserve_alpha035` | `0.231` | `0.155` | ring `1.000`, tube `0.750`, joint `0.318` | `25.971` | reject; less blow-up but low density and off-surface tail |
| `torus_normalbin_alpha035_surface_refine` | `-0.374` | `0.144` | ring `1.000`, tube `0.688`, joint `0.313` | `13.455` | reject; progressive activation/geometry recovered, but render density and coverage still fail |

The new normal-coverage term is directionally useful because it changes the
failure from "no tube support" to "surface/render control after tube support."
The remaining torus problem is balancing normal coverage, surface projection,
opacity/material density, and rollout timing under one robust multi-seed
selection objective.
