# Budgeted Adaptive NPA

This module adds conservative multiscale material particles to the hardened 2D
NPA path. It is additive: regular NPA artifacts, training, and inference retain
their existing behavior.

## Current Canonical Objective

The target experiment is a direct-active `3,070 / 4,096` comparison:

- the reference uses 4,096 visible, recurrent, and interacting particles;
- the adaptive model uses 3,070 visible, recurrent, and interacting particles;
- the adaptive seed contains 342 coarse four-unit particles and 2,728 fine
  one-unit particles, representing exactly 4,096 fine material units;
- there are no hidden templates, proxy rows, or recurrent-only fine rows;
- paired local-detail topology exchanges one coarse row for four nearby fine
  rows while coarsening four other fine rows into one coarse row, so active
  row count and represented material remain constant;
- the recurrent NPA rule is trained through the Target2D rollout objective;
- the WGPU path performs topology diagnostics and exchanges device-resident,
  without synchronizing particle state to the host.

The execution contract is implemented and tested. The canonical seed begins
with 342 coarse four-unit leaves and 2,728 fine one-unit leaves, so the
renderer exposes native and `2x` radii from frame zero. Device-resident
topology can expose reserve descendants and retire merged siblings without
reallocation or readback. Under matched stable material identities and update
masks, Burn and WGPU agree within `3.0e-8` position error and `1.2e-6` state
error.

Quality parity is not yet achieved. The latest completed normalized,
moment-corrected mixed-scale residual control measures:

| steps | adaptive mean / worst | static mixed mean / worst | matched fine/oracle mean / worst |
| ---: | ---: | ---: | ---: |
| 96 | 19.572 / 19.370 | 19.575 / 19.371 | 23.990 / 23.520 |
| 256 | 19.606 / 19.221 | 19.571 / 19.160 | 25.832 / 25.300 |
| 512 | 19.569 / 19.235 | 19.443 / 19.005 | 27.119 / 26.331 |

Across those rows, adaptive PSNR is `19.582 dB` mean and `19.221 dB` worst,
the mean matched-fine gap is `-6.065 dB`, and the mean gain over a static
mixed-scale cut is only `+0.053 dB`. The candidate performs real topology
events at `74.95%` of reference particle work, but normalized perception makes
its measured wall time `2.086x` the fine control. It does not demonstrate
useful adaptive allocation or lizard parity and is not promoted. The complete
positive execution and negative quality evidence is in
[`benchmarks/adaptive_canonical_mixed_scale_2026-07-25.json`](benchmarks/adaptive_canonical_mixed_scale_2026-07-25.json).

A matched expected-gate ablation treats each coarse row's stochastic update as
the mean of its independently masked represented material. Burn and WGPU agree
to `5.96e-7` state error for this path. It improves mean deployment PSNR to
`19.719 dB`, but lowers worst PSNR to `19.009 dB`, increases drift to
`0.950 dB`, and trails its static mixed cut by `0.038 dB` on average. Expected
gating is enabled explicitly by canonical expected-gate configs while the
generic training default remains backward-compatible, but it does not close
the quality gap.

Promotion still requires an uncentered deployment audit over seeds 42 through
49 at steps 96, 256, 512, 1,024, 2,048, and 4,096. From step 512 onward, the
aggregate adaptive result must reach at least `26 dB` mean and `24 dB`
worst-seed PSNR and may trail either the same-rule 4,096-particle control or
the released oracle by no more than `0.5 dB`. It must regress by no more than
`1 dB` after its peak, preserve material and bounded occupancy, show accepted
paired topology events, use no more than 80% of reference interaction work,
and take no more than 1.1 times the reference wall time.

The bounded structural regression is:

```bash
cargo run --release -p burn_automata --features gpu_wgpu \
  --bin burn_automata -- adaptive-npa --config \
  configs/verified/adaptive/task_resident_lizard_smoke_3070_2d_wgpu.toml
```

That smoke starts with 1,024 coarse leaves and activates a final cut containing
2,728 fine plus 342 coarse leaves. It reports a `2x` radius span and 682
bootstrap split events, so it is a direct guard against an all-equal-radius
viewer artifact. Its quality is intentionally not promoted: across two seeds
it measures `11.320 dB` versus `20.880 dB` for regular 4,096-particle rollout,
and dynamic topology trails the fixed mixed cut by `3.080 dB`. This isolates
bootstrap/state prolongation as another closure failure rather than a renderer
scale-decoding failure.

The checked-in
`configs/verified/adaptive/recurrent_target2d_active_material_smoke_2d_wgpu.toml`
is only a two-update 61-active/64-reference execution regression. It verifies
the canonical optimizer, paired topology, binary artifact, and report path. It
must not be cited as lizard-quality evidence.

### Superseded fixed-graded result

The earlier recurrent run below used a centered metric, fixed interaction
support, and only a `1.20x` largest-to-smallest radius span. It passed its
then-current numerical gates but does not satisfy the canonical `2x`
adaptive-resolution contract. These numbers are retained to make the
methodology correction auditable; they are not current deployment results.

| steps | adaptive mean / worst | same-rule fine mean / worst | released oracle mean / worst |
| ---: | ---: | ---: | ---: |
| 96 | 23.616 / 22.763 | 23.800 / 23.208 | 23.826 / 23.175 |
| 256 | 25.005 / 23.664 | 25.473 / 24.276 | 25.499 / 24.553 |
| 512 | 26.450 / 24.411 | 26.473 / 24.334 | 26.365 / 24.184 |
| 1,024 | 27.992 / 25.629 | 27.007 / 24.775 | 26.773 / 23.321 |
| 2,048 | 29.391 / 25.787 | 28.694 / 24.271 | 28.373 / 24.895 |
| 4,096 | 31.898 / 30.694 | 31.151 / 23.383 | 31.030 / 27.507 |

The historical path used `74.951%` of reference interaction work. All 3,070
rows were visible, recurrent, and interacting, but the graded material state
occupied a narrow `1.200x` scale range. Machine-readable evidence is explicitly
marked superseded in
[`benchmarks/adaptive_recurrent_target2d_lizard_2026-07-23.json`](benchmarks/adaptive_recurrent_target2d_lizard_2026-07-23.json).

## Historical Visible-Budget Result

The earlier LoD result uses 3,070 visible material leaves and 3,925 internal
recurrent rows. On 32 disjoint seeds (106 through 137) at 256 rollout steps, it
averages `22.211 dB`, compared with `22.081 dB` for regular NPA at 4,096
particles and `21.911 dB` for a represented-measure-matched regular
3,070-particle control. The corresponding mean gains are `+0.130 dB` and
`+0.300 dB`.

This historical control matched mean image quality at a 25% smaller visible
Gaussian budget. It does **not** establish dynamic adaptive-NPA parity or the
current strict objective because it retains 3,925 recurrent rows, relies on
host-mediated topology events, and does not provide the required 4,096-step
quality and work gates. Its worst gap to regular 4,096 is `-0.608 dB`. Every
seed occupies four material-scale bins with 11.1% of leaves at fractional
octaves. The checked-in historical evidence is in
[`benchmarks/adaptive_lizard_progressive_mixed_lod_2026-07-20.json`](benchmarks/adaptive_lizard_progressive_mixed_lod_2026-07-20.json).

## Isotropic Render Contract

One visible adaptive material leaf produces exactly one isotropic Gaussian:

```text
scale    = [s, s, s]
rotation = identity quaternion
```

The scalar `s` is a deterministic function of represented material measure,
the configured display calibration, and the bounded topology-transition
interpolation. It is not emitted by an NPA latent channel. A merged leaf starts
at no less than its material-conserving target radius because a smaller radius
would require opacity above one. A split leaf can safely start at its previous
larger radius and relax to its target while opacity compensates represented
measure.

Conservative covariance remains simulation metadata. It is used when material
leaves split, merge, and reconstruct persistent dynamics modes, but it cannot
be decoded into renderer scale axes or rotation. The old covariance-to-ellipsoid
decoder is crate-private and retained only as an evaluator counterfactual. It
is not exported by the library, Bevy viewer, headless exporter, or WGPU Gaussian
write path.

For the 2D lizard path, latent state controls position dynamics and DC color;
it cannot change Gaussian geometry. CPU/Bevy, Target2D, and WGPU tests inject
strongly anisotropic covariance and arbitrary latent values and require equal
XYZ scale and identity rotation.

The final Bevy/WGPU headless audit reports scalar Gaussian scales from `0.012`
to `0.024`, corresponding to material footprints from `0.003125` to `0.00625`.
Despite that intentional multiscale range, `anisotropic_particle_fraction = 0`,
`max_axis_ratio = 1`, and reconstructed-measure relative error is `2.97e-8`.
The checked renderer audit covers bootstrap steps 0 through 4 and every
restriction frame from 229 through 241, plus settled frames 252 and 256. Lit
support never falls below the pre-restriction frame and has a minimum ratio of
`1.00093`; the previous transition initialization fell to 78.6%.

## Material And Dynamics

Each material leaf stores independent quantities:

- represented measure controls physical material footprint;
- one scalar render footprint controls isotropic display size;
- covariance carries conservative second moments for simulation only;
- interaction bandwidth controls compact neighbor support;
- stable particle IDs control stochastic NPA update masks;
- hierarchy IDs define split siblings and merge parents.

The canonical direct-active contract has no nonmaterial or recurrent-only rows.
Every represented leaf is one visible Gaussian, one recurrent state row, and
one interaction row. The historical retained-mode LoD branch can retain
weighted nonmaterial sub-leaf modes, but it is not evidence for the canonical
objective and is reported separately.

The current candidate dynamics are:

```text
adaptively trained NPA rule + represented measure + conservative topology
```

The historical fixed-graded artifact disabled local/proxy residuals. Canonical
mixed-scale experiments now test a zero-initialized, material-conditioned
residual while freezing the released native-scale rule. Neither variant has
passed deployment parity, so neither is promoted.

### Recurrent closure diagnostic

A compact first-level closure experiment encoded each four-child group with
the material centroid/covariance, one oriented geometry phase, and one
affine-null coefficient per state channel. The implementation now reconstructs
child geometry and state causally; replay no longer refreshes hidden geometry
from the fine teacher. Exact round-trip, phase-causality, conserved-moment, and
teacher-template-independence tests gate this path.

That correction improved held-out recurrent closure NRMSE from `1.0013` to
`0.9484`, correlation from `0.1635` to `0.3501`, and halved mean teacher-update
error from `0.0671` to `0.0341`. It still failed the configured `0.5` NRMSE and
`0.8` correlation gates; the held-out state component remained at `0.9982`
NRMSE. The compact closure is therefore an experimental negative result, not a
deployable adaptive model. Resident WGPU closure execution is explicitly
rejected until it implements the causal phase/state feature contract. The
machine-readable comparison is
[`benchmarks/adaptive_recurrent_closure_causality_2026-07-22.json`](benchmarks/adaptive_recurrent_closure_causality_2026-07-22.json).

## Bootstrap And Reallocation

The LoD rollout starts from 1,024 visible parents backed by four exact
persistent modes each. It exposes 256 conservative parents per step, producing
1,024, 1,792, 2,560, 3,328, and 4,096 visible leaves over steps 0 through 4
without teleporting seed state or changing the hidden trajectory.

At step 230, a learned-cost-ranked restriction begins reducing the visible
budget by at most 96 leaves per step. It reaches 3,070 at step 240. Mixed
2/3/4-child aggregates produce native, `sqrt(2)`, `sqrt(3)`, and `2x` material
radii instead of a two-level dyadic cut. Coarse leaves retain at most three
persistent modes, yielding 3,925 recurrent rows. The progressive event budget
and 12-step scalar-radius interpolation prevent the one-frame support change
seen with the original cut.

The verified schedule recomputes the learned partition at each restriction
step. This permits coarse support to move toward the current detail field while
the visible count decreases. An identity-stable nested schedule was tested as
a control: at two retained modes it measured `-0.042 dB` mean and `-0.792 dB`
worst gap to regular 4,096, and it missed the verified tail gate. Retaining all
four hidden modes recovered only `+0.007 dB` mean with a `-0.885 dB` worst gap.
The rolling schedule is therefore intentional, not accidental partition churn.
Sorted material cells use deterministic particle ordering until the final
topology decision, preventing atomic scatter order from changing controller
features and partitions between runs. Once no future event exists, settled
dynamics return to the faster regular sorted-cell path.

The earlier one-shot dyadic selector and same-budget reallocation study remains
useful historical evidence. More frequent reallocations were worse:

| reallocation cadence | mean gap vs regular 4,096 | worst gap | mean gain vs material-matched 3,070 |
| --- | ---: | ---: | ---: |
| step 160 | -0.253 dB | -1.466 dB | -0.096 dB |
| step 192 | -0.107 dB | -1.313 dB | +0.062 dB |
| step 224 | -0.029 dB | -1.100 dB | +0.129 dB |
| **step 240** | **+0.045 dB** | **-1.091 dB** | **+0.206 dB** |
| step 256 | -0.016 dB | -1.026 dB | +0.146 dB |
| steps 224, 240, 256 | -0.022 dB | -1.243 dB | +0.148 dB |

Against fixed reduced-budget topology, the progressive mixed-arity schedule
gains `+2.584 dB` on average; all 32 paired seeds improve and the worst gain is
`+0.173 dB`. Topology changes outside the bounded step-230 through step-240
restriction are not enabled by the verified protocol.

## Objective Alignment

The restriction controller must be trained against the same render semantics
used at deployment. Earlier labels minimized a compact covariance-Gaussian
counterfactual, even though production inference now renders isotropic leaves.
That mismatch was corrected by adding an isotropic material primitive to the
shared Target2D merge-cost kernel and threading `AdaptiveRenderDecoder` through
CPU and WGPU label generation.

The corrected selector uses:

| metric | result |
| --- | ---: |
| training seeds / snapshots / rows | 64 / 704 / 720,896 |
| held-out seeds / snapshots / rows | 32 / 352 / 360,448 |
| controller width | 512 |
| optimizer | Burn/WGPU, 5,000 steps |
| loss | 4.0782 -> 0.02921 |
| training merge IoU | 0.6171 |
| held-out merge IoU | 0.5797 |
| optimizer throughput | 24.08M rows/s |
| label generation | 23.68 s train + 11.01 s held out |
| optimizer time | 149.66 s |
| total experiment time | 199.26 s |
| peak process RSS | 5.07 GiB |

The optimizer phase sustained 100% reported GPU utilization and approximately
404-446 W in sampled device telemetry. Label generation uses resident WGPU
rollouts plus the device merge-cost kernel.

## Gap Decomposition

A matched-seed evaluator separates recurrent dynamics from visible decoding.
Across eight seeds:

| component | mean PSNR change |
| --- | ---: |
| uncut adaptive backend vs regular fine | +0.0022 dB |
| full retained-mode dynamics vs uncut adaptive | +0.0171 dB |
| visible isotropic leaf vs internal full modes | -0.2071 dB |
| late target-independent dynamics cut | -0.9067 dB |
| late learned isotropic cut | +0.3684 dB |
| covariance decoder vs isotropic decoder | -0.2635 dB |

The recurrent persistent-mode dynamics are numerically aligned with the full
fine control. The remaining error is allocation/visible restriction, not an
anisotropic renderer deficiency. The covariance decoder is worse in this
control, which is additional evidence against exposing arbitrary Gaussian
shape as adaptive capacity.

## Inference Throughput

The superseded fixed-graded evaluator sustained `4.280M` adaptive
particle-steps/s versus `4.645M` for its same-rule 4,096-particle control. That
remains useful kernel-throughput evidence, but it is not a canonical
mixed-scale quality result.

The canonical resident path keeps active and reserve rows in one allocation,
changes an active mask and material metadata for split/merge exchanges, and
does not read particle state back to the host. Its interaction work is
`0.7495x` the 4,096-row reference. The latest canonical quality control has a
mean wall ratio of `1.208x`, so it also misses the `1.1x` promotion gate.
Normal inference avoids strict diagnostic readbacks and deterministic sorting
after topology settles.

The older retained-mode LoD viewer-cadence benchmark reports:

| visible / dynamics | adaptive ms/step | particle-steps/s | regular 4,096 | wall ratio |
| ---: | ---: | ---: | ---: | ---: |
| 3,070 / 3,925 | 0.639 | 6.14M | 0.659 ms | 0.969x |

Persistent restriction plus Gaussian export costs `0.103 ms/step` in that
historical measurement. The retained-mode path is not the canonical
direct-active contract.

The four bootstrap and 11 rolling-restriction events still cross a host
boundary for candidate selection, partition construction, and active-state
rebuilding. Across the serialized 32-seed evaluator, topology precomputation
consumed `87.1 s` of `97.3 s` total quality precomputation. The four-stage
schedule reduces topology time by 40.7% and end-to-end rollout time from
`17.327` to `10.671 ms/step` relative to the exact eight-stage schedule. Host
events can still hitch interactive frames; they do not turn steady rollout
into a CPU path. Moving selection and reallocation onto resident device state
is required before claiming end-to-end throughput parity for dynamic
allocation.

## Evaluation And Viewing

Run the bounded resident structural protocol:

```bash
cargo run --release -p burn_automata --features gpu_wgpu \
  --bin burn_automata -- adaptive-npa --config \
  configs/verified/adaptive/task_resident_lizard_smoke_3070_2d_wgpu.toml
```

Canonical quality candidates are intentionally sandboxed. After training one,
run its matched-seed deployment audit before opening it in the viewer:

```bash
cargo run --release -p burn_automata --features gpu_wgpu \
  --bin burn_automata -- eval-adaptive-target2d --config \
  configs/sandbox/adaptive/canonical_lizard_mixed3070_residual_eval_2d_wgpu.toml
```

The viewer accepts the resulting experimental artifact explicitly:

```bash
cargo run --release -p bevy_automata --features splatting,gpu_wgpu -- view \
  --adaptive-model \
  artifacts/adaptive/canonical_curriculum/lizard_mixed3070_residual.adaptive.bpk \
  --particles 3070 --seed-scale 0.2 --render-scale 0.5
```

This is for diagnosis only: the candidate does not pass quality promotion.
The historical 32-seed retained-mode protocol remains available through
`task_lod_lizard_eval_3070_2d_wgpu.toml`.

## Historical LoD Boundary

The broader continuous material-scale extension, its event algebra, training
order, execution-binning contract, and promotion gates are specified in
[`adaptive_continuous_scale.md`](adaptive_continuous_scale.md). The verified
historical retained-mode artifact exercises conservative mixed 2/3/4-child
material scales, but it does not claim arbitrary real-valued event measures or
a continuous learned resolution field.

Current machine-readable evidence is in
[`adaptive_continuous_topology_100k_2026-07-20.json`](benchmarks/adaptive_continuous_topology_100k_2026-07-20.json)
and
[`adaptive_continuous_scale_lizard_pilot_2026-07-20.json`](benchmarks/adaptive_continuous_scale_lizard_pilot_2026-07-20.json).
The exact CPU execution-bin work-count regression is
[`adaptive_support_bins_cpu_2026-07-20.json`](benchmarks/adaptive_support_bins_cpu_2026-07-20.json).

For the historical retained-mode LoD branch, the remaining work is:

1. Move its selector inference, split/merge selection, and persistent
   remapping to a device-resident WGPU/CUDA path so reallocation cannot stall
   the viewer. The strict paired local-detail path already performs its
   topology operation on resident device state.
2. Train the selector on future-horizon isotropic rollout loss, with explicit
   tail weighting for the remaining `-0.608 dB` worst seed.
3. Expand quality evaluation beyond the lizard seed distribution before
   claiming general adaptive-NPA superiority; the current result is a lizard
   LoD parity result.
4. Keep covariance-shaped and retained-mode renders diagnostic-only. They must
   never become hidden render capacity in a production adaptive artifact.
5. Train the shared normalized-adaptive NPA rule jointly across mixed material
   resolutions with Target2D and restriction-evolution losses. Do not continue
   capacity sweeps on the failed fixed-teacher compact closure head.
