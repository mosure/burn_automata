# Continuous-Scale Adaptive NPA

The publication, measured claim boundary, and current renderer evidence are in
[`adaptive_npa.pdf`](adaptive_npa.pdf) and
[`adaptive_npa.md`](adaptive_npa.md). This document is the deeper implementation
contract.

## Objective

Adaptive NPA should expose a continuous material-resolution field without
turning renderer scale, interaction reach, hierarchy level, or cache level into
the same variable. The target contract is:

```text
represented measure omega_i  -> material radius delta_i
material radius delta_i      -> one isotropic Gaussian radius
material radius + controller -> continuous interaction bandwidth h_i
continuous h_i               -> conservative execution bucket beta_i
```

Only accepted split and merge events change the integer number of material
degrees of freedom. `omega_i`, `delta_i`, the displayed scalar radius, and
`h_i` are real-valued. An execution bucket is not serialized as particle state
and cannot alter an accepted interaction.

This design follows the paper's separation of represented resolution,
interaction bandwidth, scale-context sampling, and nonmaterial proxies. It
does not treat a dyadic hierarchy row as a material species.

## Semantic Contracts

| Quantity | Meaning | Continuous | Rendered | May change leaf count |
| --- | --- | ---: | ---: | ---: |
| `represented_measure` | authoritative represented material | yes | through radius | no |
| material footprint `delta` | radius derived from measure | yes | yes | no |
| `render_footprint` | bounded transition toward `delta` | yes | yes | no |
| covariance | conservative simulation second moment | yes | no | no |
| bandwidth `h` | exact compact interaction support | yes | no | no |
| support bin `beta` | candidate-search acceleration | discrete | no | no |
| proxy row | nonmaterial context cache | discrete cache sample | no | no |
| split/merge event | conservative topology update | discrete event | indirectly | yes |

For intrinsic dimension `q`, material radius is always

```text
delta_i = (omega_i / c_q)^(1/q)
```

and the production decoder emits one isotropic Gaussian per active material
leaf. A larger coarse leaf therefore produces one larger Gaussian. Persistent
sub-leaf modes and communication proxies remain nonrendered.

"Coarse" does not mean "internal node that is rendered in addition to its
children." If a region is merged, the merged row becomes the active material
leaf and owns one larger Gaussian; its replaced descendants are inactive. If
the descendants are active, the hierarchy parent is only a communication
proxy and owns no Gaussian. Rendering both would double-count represented
measure, hide the actual leaf budget, and invalidate PSNR comparisons.

## Conservative Unequal Events

The compatibility event remains the existing equal `2q`-child canonical
split. Continuous topology is opt-in through
`max_unequal_split_measure_ratio > 1`.

For positive child fractions `alpha_c` with `sum(alpha_c) = 1`, the unequal
event sets

```text
omega_c = alpha_c * omega_parent
Sigma_c = alpha_c^(2/q) * Sigma_parent
```

so determinant-derived child footprint scales as `alpha_c^(1/q)`. A
deterministic weighted sigma-point frame supplies offsets satisfying

```text
sum(alpha_c * offset_c) = 0
sum(alpha_c * offset_c * offset_c^T)
    = Sigma_parent - sum(alpha_c * Sigma_c)
```

This preserves represented measure, centroid, uncentered second moment, and
every extensive channel. Intensive child state is prolonged affinely and
recentered with represented-measure weights. The merge path already supports
arbitrary positive measures and recovers those moments.

Equal fractions call the original canonical implementation exactly. Existing
artifacts therefore retain their positions, measures, event ranking, and
rollout behavior.

## Fraction Policy

The rollout does not add an untrained geometry head. It obtains unequal
fractions from the controller's existing globally normalized desired-footprint
field:

1. Fit a regularized local linear gradient of desired log footprint from the
   nearest material leaves.
2. Evaluate that field at the canonical child probe positions.
3. Convert desired radius to desired `q`-measure.
4. Clamp child log measures around their mean so the largest-to-smallest ratio
   cannot exceed `max_unequal_split_measure_ratio`.
5. Normalize fractions exactly and construct the conservative event.
6. Reject the event if any child leaves material bounds, crosses the domain, or
   violates `max_neighbor_footprint_ratio` on an interacting edge.

The field reconstruction is deterministic and target-independent at runtime.
The controller still owns desired resolution; the topology layer only realizes
a feasible local partition.

Relevant artifact fields are:

```toml
max_unequal_split_measure_ratio = 4.0
split_field_neighbors = 16
max_neighbor_footprint_ratio = 2.0
min_reallocation_relative_gain = 0.25
```

`min_reallocation_relative_gain` is a dimensionless hysteresis margin in
`[0, 1]`. Zero preserves the legacy selector, values below one require a
strictly better budget-neutral merge/split exchange, and one disables those
speculative exchanges while retaining bootstrap, scheduled restriction, and
other count-correcting events. A continuous artifact should use a measured
margin below one only after recurrent hard-event training passes the tail
gates; one is the safe deployment setting for an artifact that was not trained
for reallocation.

The recurrent unequal-split path remains experimental because the frozen base
rule was not trained through those events. The verified lizard LoD path instead
uses a bounded mixed-arity restriction: learned first-level costs rank local
fine groups, then deterministic 2/3/4-child merges partition every fine leaf
exactly once. This produces radii proportional to `sqrt(2)`, `sqrt(3)`, and
`2` in addition to the native radius while persistent quadrature carries the
underlying NPA trajectory. It is a conservative fractional-octave execution
of the continuous-scale contract, not evidence that arbitrary real-valued
event weights have passed recurrent training.

## Direct-Active 2D Candidate

The bounded checked-in regression is
`configs/verified/adaptive/recurrent_target2d_lizard_continuous_ratio4_smoke_3070_2d_wgpu.toml`.
The broad sandbox protocol evaluates 32 matched seeds at steps 96, 256, 512,
and 1,024. Both rebuild the same binary adaptive candidate from the trained
scale-conditioned rule. Its runtime contract is:

```text
visible/recurrent/interaction rows: 3,070
represented fine material units:    4,096
hidden fine rows:                    0
represented-measure range:          4.004x
isotropic material/render radius:   2.001x
occupied 1/64-octave audit bins:    33
local-detail exchanges:             8 every 64 steps
interaction bandwidth/support bins: fixed / 1
```

Across the broad 128-row protocol, uncentered deployment PSNR is `25.315 dB`
mean and `23.169 dB` worst. At the quality horizons it reaches `26.118 dB`
mean / `25.137 dB` worst at step 256 and `26.318 dB` mean / `25.223 dB`
worst at step 512. The overall mean gap to the released 4,096-row oracle is
`-0.162 dB`. Interaction work is `74.95%` of the 4,096-row control, and
measured wall time is `0.963x` the same-rule 4,096-row path.

Dynamic allocation improves scale/detail correlation over the static graded
control by `+0.0536` mean and `+0.0025` worst. Its mean PSNR gain over that
control is only `+0.014 dB`; the worst row is `-0.694 dB`. At step 1,024 the
mean falls to `24.969 dB`, and worst per-seed drift reaches `2.311 dB`.
Consequently this candidate does not pass the full promotion gate. A 64-event
variant lost `0.775 dB` mean versus static allocation, while a `9x`
represented-measure (`3x` radius) variant fell to `24.639 dB`. The selected
`4x`/8-event schedule is therefore the best measured broad operating point,
not evidence
that the rule is scale-free over arbitrary ranges.

The Bevy/3DGS audit exports nonblank captures at steps 0, 96, 256, and 512. It
measures Gaussian scales from `0.009423` through `0.018856`, zero anisotropic
particles, axis ratio exactly one, and represented-measure reconstruction error
of `1.23e-7`. The immutable numerical and renderer summary is
`docs/benchmarks/adaptive_continuous_scale_ratio4_2026-07-25.json`.

This path continuously reallocates represented measure among a fixed set of
3,070 resident rows. It demonstrates continuous material/render scale and
measurable spatial resolution allocation. It does not yet demonstrate a robust
quality advantage, continuous communication bandwidth, runtime row births or
deaths, or a generalized adaptive rule. Those boundaries prevent promotion.

### Scale limits and event-aware training

The runtime clamps are not the active limit on the measured lizard candidates.
The configured material and render footprint range is `0.0015625..0.1`
(`64x` in radius), while the interaction-bandwidth range is `0.025..0.4`
(`16x`). The scale feature supplied to a scale-conditioned rule represents
footprint ratios from `0.25x` through `4x`. The shared-rule experiments do not
consume the residual gate.

The narrower effective range comes from the fixed graded material spectrum.
`seed_measure_ratio` values of `4`, `6.25`, and `9` produce measured 2D radius
spans of `2.001x`, `2.499x`, and `2.999x`, respectively. Topology events
relocate these preallocated scale slots; they do not synthesize radii outside
the material layout. The `9x` experiment occupied only
`0.001892..0.005674` of the configured footprint range, and its raw scale
feature occupied `-0.395..0.816`, with zero lower or upper saturation.
Interaction bandwidth was intentionally held at `0.1` for these comparisons.
Allowing it to widen with particle radius degraded quality rather than exposing
a clamp.

Event-aware recurrent training now:

- samples persistent trajectories that cross scheduled topology events;
- splits TBPTT chunks at the exact event step;
- scores a bounded differentiable post-event recovery rollout;
- accumulates segment gradients and applies one Adam update per outer step; and
- optionally selects checkpoints by worst seed/horizon PSNR minus a drift
  penalty.

The exact-event `4x` candidate improved broad mean PSNR from `25.310` to
`25.351 dB` and improved the step-1,024 mean from `24.916` to `25.007 dB`.
However, worst per-seed drift increased from `2.222` to `2.475 dB`. A paired
pre/post-event degradation loss and a 16-seed drift-aware checkpoint selector
did not generalize to the 32-seed protocol. The `6.25x` and `9x` candidates
reached `24.934` and `24.650 dB` broad means. Therefore, the original `4x`
quality candidate remains the deployment leader. The remaining quality problem
is broad stochastic long-horizon recovery, not renderer scale decoding, hard
scale clipping, or another topology schedule sweep. Full measurements and
correctness gates are recorded in
`docs/benchmarks/adaptive_event_aware_long_horizon_2026-07-25.json`.

## Validated 2D LoD Path

The maintained WGPU configuration is
`configs/verified/adaptive/task_lod_lizard_eval_3070_2d_wgpu.toml`, with a
one-seed smoke companion in the same directory. It uses:

```text
visible leaves: 1,024 -> 1,792 -> 2,560 -> 3,328 -> 4,096 over steps 0..4
late restriction: 4,096 -> 3,070 over steps 230..240
restriction cadence: at most 96 visible leaves removed per step
event family: learned-cost-ranked 2/3/4-child conservative merges
partition schedule: rolling recomputation at each bounded restriction step
hidden dynamics: exact 4-mode bootstrap, then at most 3 modes per coarse leaf
rendering: one isotropic measure-derived Gaussian per visible material leaf
```

On seeds 106..137 at step 256, the mixed-arity artifact measured `22.211 dB`
mean PSNR versus `22.081 dB` for regular 4,096-particle inference. The mean
gains were `+3.411 dB` over the 3,070-count control and `+0.300 dB` over the
represented-material-matched control. The worst regular-4,096 gap was
`-0.608 dB`; this remains the tail limitation and prevents a stronger
per-seed noninferiority claim. Every seed occupied four material-scale audit
bins with `11.14%` off-dyadic leaves, and maximum represented-measure drift was
`4.95e-9`. The immutable summary is
`docs/benchmarks/adaptive_lizard_progressive_mixed_lod_2026-07-20.json`.

The progressive schedule is a renderer and topology requirement, not merely a
quality-report setting. Replacing all selected children in one frame caused a
visible centroid/support discontinuity. The verified rolling schedule retains
the same final budget while limiting the leaf-count change per frame. New
merged leaves begin at no less than their material-conserving target radius;
otherwise opacity above one would be required and the image would dim. Split
leaves retain bounded scalar-radius interpolation with opacity compensation.
In the Bevy/WGPU audit, minimum lit support through the restriction improved
from 78.6% to 100.09% of the pre-restriction frame.
Deterministic in-cell material ordering is active while future topology
decisions remain, so atomic scatter order cannot change the learned partition.
After the final event, the state returns to the faster steady sorted-cell path.

## Rendering

No new renderer degree of freedom is required. The existing CPU, Bevy, and
WGPU paths already consume arbitrary positive represented measures and
continuous scalar render footprints. They require:

```text
scale.x = scale.y = scale.z
rotation = identity
```

Opacity compensates only while the displayed scalar interpolates toward the
measure-derived target after an event. Covariance cannot alter production
Gaussian axes or rotation. Smooth visual interpolation is not evidence of
continuous material resolution; the represented measures themselves must be
non-dyadic.

Runtime traces now report:

- occupied material-scale bins at 1/64-octave audit resolution;
- fraction of leaves off integer octaves relative to the reference footprint;
- RMS distance to the nearest dyadic material scale.

These are diagnostics, not execution bins.

## Communication Binning

The exact pair support remains the power mean

```text
h_ij = ((h_i^p + h_j^p) / 2)^(1/p)
```

and a pair is accepted only when `distance(i,j) < h_ij`. The CPU kernel already
uses configurable geometric source-support bins (`support_bin_ratio = 2` is
the compatibility default). Each source is inserted into the smallest bin
whose upper bound covers its exact `h_j`; a target queries each bin with
`pair_mean(h_i, upper_bound_beta)`, then performs the exact test above. The
kernel test compares this path with all pairs across 2D/3D and all graph
policies.

Hard source bins are sufficient for exact scalar bandwidth because they only
add false-positive candidates. Cubic assignment to four neighboring scale
buckets is useful only if a future scale ensemble approximates a continuous
integral from cached samples. It is not needed, and would add work, for the
current exact scalar-support operator.

Configuration rejects support-bin layouts requiring more than 64 bins. This
prevents a ratio accidentally set arbitrarily close to one from creating an
unbounded host allocation or device scan surface. The cap constrains only the
broad phase; exact bandwidths and accepted interactions remain continuous.

Reports distinguish broad-phase `candidate_visits`, exact-support
`raw_messages`, and post-policy `accepted_messages`. On the deterministic
784-particle log-scale test, changing the execution ratio from `2` to `sqrt(2)`
reduced candidate visits from 47,337 to 46,810 while preserving all 5,298 exact
messages and every perception value exactly. This is a modest scene-specific
work reduction, not a device-throughput claim; the ratio remains an execution
tuning parameter, never material state.

The WGPU cooperative and subgroup sorted-cell kernels now evaluate exact
continuous pair bandwidth while sorting sources by

```text
(batch, source_support_bin, Morton_cell)
```

and query each source bin conservatively. The final exact support test remains
in every message traversal. Dynamic bandwidth rebuilds the bin key as part of
the existing per-step position-grid rebuild; changing `h_i` never moves the
learned state onto a discrete value.

### Device support-bin implementation

The device path should use one sorted index, not one duplicated particle set:

```text
cell_key = ((batch * support_bin_count + source_bin) * spatial_cells) + morton_cell
counts   = batch * support_bin_count * spatial_cells
offsets  = counts + 1
indices  = total material rows
```

Each source row appears exactly once. `source_bin` is recomputed from its exact
bandwidth during the existing per-step grid rebuild. For target `i`, the
perception kernel loops over source bins, obtains that bin's conservative upper
bandwidth, computes `ceil(M_p(h_i, upper_bin) / cell_width)`, visits those
cells, and then applies the exact `r_ij < M_p(h_i, h_j)` test. Directed and
mutual top-k ranking continues to use exact normalized distance and the stable
particle-ID tie break. There is therefore no interpolation across bins, no
duplicate source contribution, and no quantized state.

The implementation is split into independently testable layers:

1. Padded WGPU parameters carry bin count, support bounds, ratio, and spatial
   cell count; `bin_count = 1` executes the global-support traversal.
2. Count/scan/scatter use the composite key above. Storage grows by
   `O(batch * bins * cells)` for counts/offsets but indices remain `O(N)`.
3. Cooperative and subgroup adaptive-perception traversals loop over source
   bins. The global-radius kernel remains an explicit exact
   fallback when storage limits or unsupported layouts prevent binning.
4. State creation reserves only layouts accepted by the existing 65,536-cell
   scan bound and 128 MiB conservative storage-binding limit. Ordinary sorted,
   fixed-bucket, and BVH traversals continue to use one global bin.
5. A host-side policy estimates local cell occupancy and the reduction in
   bandwidth-histogram-weighted query-cell work. It activates bins only above
   1,024 particles, at an estimated 64 particles per occupied spatial cell,
   and when predicted binned candidate work is at most 60% of global work.
6. Host-synchronized topology events can switch `1 <-> capacity` in place.
   Capacity is allocated once; steady rollout performs no readback or device
   allocation for this decision.

The promotion gate is stronger than numerical closeness: candidate recall and
accepted graph identity must be exact against all pairs for deterministic CPU
tests and exact against the current global-radius WGPU path before comparing
perception values. Performance is selected by p50/p95 step time and the
candidate-work estimate; a finer ratio is rejected if extra scan/bin work
outweighs candidate savings. Timestamp-query phase breakdown and device
candidate counters remain useful profiling work, but are not required for
correctness or automatic selection.

On the 4,096-particle, 8x-range device sweep, bins were deliberately
scene-dependent. A concentrated distribution with 90% of rows at the finest
support improved from 2.4763 ms to 1.9623 ms p50 (`1.26x`) under the automatic
three-bin policy. Diffuse and balanced log-scale distributions remained on the
one-bin path. Forcing three bins on diffuse log-uniform data regressed p50 from
0.7922 ms to 1.1811 ms, which is why support bins are not unconditional. The
complete p50/p95 matrix is in
`docs/benchmarks/adaptive_support_bins_wgpu_2026-07-20.json`.

`PersistentFineQuadrature` is intentionally outside this material-continuum
claim. It restores and recuts a fixed native-particle template, so its active
leaves remain sums of equal native measures. Its retained children are useful
as a nonrendered diagnostic quality ceiling, but they are not an adaptive
compute result and cannot establish continuous material resolution. Production
continuous-scale training must use `RepresentedMeasure` dynamics (or a future
learned closure over those leaves).

## Training Sequence

Continuous topology should not be enabled inside the full recurrent objective
before the following ladder passes.

### Stage 0: event algebra

- Randomized SPD parents in intrinsic dimensions 2, 3, and 4.
- Equal and unequal child fractions spanning ratios 1, 2, 4, and 8.
- At least 100,000 events for a release report.
- Measure, centroid, second moment, determinant-footprint calibration, extensive
  channels, and child SPD checked in float64.

### Stage 1: unequal quadrature and decoder

- Fixed topology and oracle decompositions.
- Randomized density, noninteger global dilation, and smooth scale gradients.
- Constant and affine field reproduction across scale interfaces.
- Isotropic Target2D reconstruction with matched total represented measure.
- No learned hard event selection yet.

### Stage 2: resolution controller

- Train desired log footprint against measured counterfactual restriction loss.
- Retain global budget normalization.
- Add measure-weighted resolution loss and neighboring log-scale grading loss.
- Validate held-out desired-field calibration separately from event decisions.

### Stage 3: recurrent hard events

- Enable conflict-free conservative events with cooldown and event budgets.
- Stop gradients through event selection; backpropagate through pre-event and
  post-event recurrent segments with configured TBPTT chunks.
- Sample event times and rollout horizons rather than training one fixed cut.
- Include restriction-evolution commutator and future-horizon isotropic image
  loss.
- Preserve persistent pools, localized state erasure, and fresh-seed injection
  from the hardened 2D training path.

### Stage 4: bandwidth and proxy execution

- Train continuous `h_i` only after represented-measure dynamics pass.
- Compare direct exact support with binned candidate search at identical state.
- Distill proxy context from the direct operator after local graph accuracy is
  established.
- Proxies remain nonmaterial and nonrendered at every stage.

Representative objective terms are:

```text
L = L_target2d
  + lambda_future * L_future_rollout
  + lambda_comm * L_restrict_evolve
  + lambda_resolution * sum_i omega_i (log delta_i - log delta_i*)^2
  + lambda_grade * sum_(i,j) [abs(log delta_i - log delta_j) - g_max]_+^2
  + lambda_churn * event_churn
  + lambda_cost * measured_candidate_and_message_work
```

Count penalties alone are insufficient. Reports must separate active material
leaves, persistent recurrent rows, raw candidates, accepted messages, and
proxy visits.

## Validation Matrix

### Event distributions

- uniform fractions;
- smooth ramps;
- one large and several small children at the configured ratio bound;
- randomized log-uniform fractions;
- near-domain events;
- repeated split/merge cycles;
- mixed-measure spatial merge groups.

### Particle distributions

- uniform grids and jittered grids;
- sharp density interfaces;
- clustered particles;
- sparse interiors with fine boundaries;
- cavities and thin structures;
- moving coarse-fine interfaces;
- 1k, 3k, 4k, and 16k 2D particle budgets.

### Rollout quality

- lizard at matched seeds and rollout steps 96, 256, 512, and 4096;
- current equal-event adaptive artifact;
- continuous-event artifact;
- regular 3,070 and regular 4,096 controls;
- fixed-allocation and dynamic-reallocation ablations;
- mean, median, p10, p90, and worst-seed PSNR gap;
- occupancy, overflow, motion stability, scale chatter, and PSNR drift.

### Performance

- candidate build, perception, rule, integration, topology, and render times;
- p50/p95/p99 topology hitch;
- candidate visits and accepted messages per leaf;
- GPU utilization, board power, VRAM, host RSS, and host-device transfers;
- continuous bandwidth ranges of 1x, 2x, 4x, 8x, and 16x;
- direct global-radius search versus support-binned search.

## Promotion Gates

A continuous-scale configuration is not `verified` until all of these pass:

1. **Conservation:** maximum relative measure, second-moment, determinant-scale,
   and extensive error below `1e-10`; centroid L2 error below `1e-10`; zero SPD
   failures in 100,000 randomized unequal events.
2. **Legacy parity:** absent/new-default fields produce identical equal-event
   topology and pass the full adaptive unit suite.
3. **Real material continuum:** at least 25% of leaves are off integer octaves
   and at least eight 1/64-octave audit bins are occupied in a designated
   continuous-scale rollout. Render interpolation alone does not count.
4. **Grading:** no accepted interacting pair exceeds the configured footprint
   ratio; rejected event counts are reported.
5. **Render contract:** one visible Gaussian per active material leaf, zero
   rendered proxies/modes, axis ratio exactly one within `1e-6`, identity
   rotation within `1e-6`, and represented-measure reconstruction error below
   `1e-5`.
6. **Graph correctness:** binned CPU and device candidates have 100% recall
   against exact all pairs; accepted graph and perception values match the
   direct oracle within the established float tolerance for every graph policy.
7. **Quality:** the 32-seed lizard protocol must improve or preserve mean PSNR,
   and the worst gap to regular 4,096 must be no worse than `-0.25 dB`. The
   current mixed-arity artifact reaches mean parity but does not pass this
   stronger tail gate (`-0.608 dB`).
8. **Steady throughput:** continuous material values add no more than 5% to
   steady WGPU inference at matched visible/recurrent rows.
9. **Topology latency:** device-resident event selection/remapping must remove
   the current host hitch; p95 event latency is reported separately rather than
   amortized into steady steps.
10. **Training stability:** no NaN, OOM, swap growth, unbounded host cache, or
    pathological host-device transfer is accepted in the full training run.

## Implementation Status

Implemented:

- arbitrary continuous represented measure and isotropic Gaussian rendering;
- conservative unequal `2q`-child event algebra;
- deterministic fraction reconstruction from the desired-resolution field;
- material/domain/scale-grading event rejection;
- measure-weighted intensive-state recentering;
- continuous-scale rollout diagnostics;
- reusable dyadic support-bin contract in `burn_automata_kernels`;
- exact CPU support-bin parity with all pairs;
- bounded WGPU composite support-bin grids for cooperative and subgroup
  sorted-cell traversal;
- exact continuous support tests after every binned device lookup;
- density- and bandwidth-aware automatic activation with one-bin fallback;
- allocation-free support-policy refresh at host-synchronized topology events;
- a padded host/WGSL adaptive-parameter contract covering every device flag;
- randomized unequal-event and rollout integration tests;
- a bounded 3,070-row direct-active lizard candidate spanning 33 continuous
  material-scale bins and a `2.001x` isotropic Gaussian-radius range;
- 32-seed 96/256/512/1,024-step deployment, allocation, work, conservation,
  overflow, and renderer diagnostics.

Not yet promoted:

- a trained lizard artifact with learned unequal split/merge births and deaths;
- a direct-active lizard artifact that passes the 32-seed long-tail gate;
- continuous learned interaction bandwidth in the direct-active lizard;
- device-resident selector/remapping for topology events;
- future-horizon controller training with explicit PSNR-tail weighting;
- proxy-context training and validation;
- timestamp-query phase attribution and exact device candidate counters;
- the quality, tail, and event-latency gates above.

The compact recurrent closure controls are also not promoted. After removing a
teacher-geometry leak and making its geometry/state representation causal, it
still measured `0.9484` held-out on-policy NRMSE and `0.3501` correlation. Its
state component remained near the zero-predictor scale (`0.9982` NRMSE). This
rules out additional blind width/step sweeps on that fixed-teacher head; the
next dynamics experiment must train the shared adaptive rule under the actual
multiresolution objective. See
[`adaptive_recurrent_closure_causality_2026-07-22.json`](benchmarks/adaptive_recurrent_closure_causality_2026-07-22.json).
The subsequent quality-scale q24 compact-memory branch also failed: its
duration-matched 256-step CUDA run peaked at `18.886 dB`, drifted to
`17.533 dB`, and deployed at `19.369 dB` mean versus `25.315 dB` for the
direct-active continuous-scale candidate. It remains sandbox-only.

The direct-active ratio-4 artifact is the current continuous-scale candidate,
not a promoted baseline. The progressive mixed-arity artifact remains the
verified variable-leaf-count LoD baseline. Arbitrary learned unequal-event
topology and continuous interaction bandwidth remain opt-in research paths.

## 2026-07-20 Evidence

The topology-only release audit is reproducible with
`configs/verified/adaptive/continuous_topology_full.toml`. It exercised 100,000
canonical and 100,000 unequal events in intrinsic dimensions 2 through 4:

| Metric | Result |
| --- | ---: |
| Event throughput | 483,345 events/s |
| Maximum sampled child-measure ratio | 7.911 |
| Maximum measure relative error | `5.220e-16` |
| Maximum centroid L2 error | `6.844e-16` |
| Maximum second-moment relative error | `8.719e-16` |
| Maximum determinant-scale relative error | `3.263e-15` |
| SPD failures | 0 |

The machine-readable report is
`docs/benchmarks/adaptive_continuous_topology_100k_2026-07-20.json`.

Real-device tests additionally establish that 64 strictly increasing,
non-binned represented measures decode to 64 corresponding isotropic WGPU
Gaussian scales, and that a represented-measure WGPU step matches the CPU
oracle to `0.0` position error and `9e-7` state error while forcing four source
support bins. A separate transition test executes a resident state through
`1 -> 3 -> 1` active bins without reallocating as its bandwidth distribution
changes. The subgroup normalized-primary path also passes its unequal-measure
CPU parity gate with four forced bins. A direct global-radius versus forced
four-bin device comparison keeps base, normalized, update, and spacing inputs
within `1e-4` maximum absolute error and preserves the accepted graph degree
exactly. The corresponding CPU comparison has normalized-input RMSE below
`2e-4`; its `5e-3` maximum bound covers a cancellation-sensitive corrected
occupancy-gradient tail rather than missing neighbors.

A matched four-seed lizard pilot used direct `RepresentedMeasure` dynamics at
4,096 initial leaves, a learned cut to 3,070 leaves at step 128, and evaluation
at step 256. With budget-neutral reallocation explicitly disabled, enabling
the unequal-event substrate did not produce a material continuum: every seed
ended with two occupied dyadic levels, zero off-dyadic leaves, and zero steady
split events. Mean PSNR was 20.909 dB versus 21.322 dB for regular 4,096. The
matched equal-option control reached 20.924 dB versus 21.341 dB. The small
cross-run difference is not interpreted as an unequal-event effect because no
steady event occurred and independent WGPU runs are not bitwise deterministic.

The safety setting is necessary for this artifact. At a 0.1 relative-gain
margin, five identical seed-109 evaluations bifurcated: two runs accepted 24
to 26 paired reallocations and fell to 12.72--12.77 dB, while three no-event
runs remained at 20.48--20.58 dB. With the explicit `1.0` no-speculation
setting and the corrected WGPU parameter-uniform layout, five repeats accepted
zero steady events and remained within 20.518--20.751 dB. This is evidence that
the old selector is unsafe for recurrent hard events, not evidence that
continuous reallocation is already successful.
The machine-readable pilot is
`docs/benchmarks/adaptive_continuous_scale_lizard_pilot_2026-07-20.json`.

This pilot closes an important ambiguity: the existing controller was trained
to select a hierarchical cut, not to predict a smooth continuous resolution
field or to maintain quality through unequal events. The algebra, renderer,
and resident WGPU state now support continuous measures; the learned policy
does not yet exercise them.

## Ordered Next Work

1. **Continuous decomposition oracle.** Build target-independent training cuts
   whose leaf measures are sampled from smooth log-scale fields, including
   ramps, localized detail, sharp-but-graded interfaces, and randomized
   noninteger global dilations. Retain an equal-cut lane in every batch.
2. **Rule robustness curriculum.** Train the shared normalized adaptive rule on
   mixed equal/unequal cuts with `RepresentedMeasure` dynamics. Begin with fixed
   topology and short horizons, then extend TBPTT horizons while sampling event
   times. Gate constant/affine reproduction and equal-limit NPA parity every
   phase.
3. **Resolution supervision.** Train desired log footprint against measured
   counterfactual future image loss, not hierarchy level labels. Add explicit
   smoothness/grading, global budget, event churn, and restriction-evolution
   commutator terms. Report calibration by desired-scale quantile.
   Calibrate a positive reallocation gain margin during hard-event phases and
   require repeatability sweeps around every decision boundary. Keep the
   explicit `1.0` no-speculation setting until the learned controller passes.
4. **Hard-event fine-tuning.** Enable conservative unequal events only after
   stages 1-3 pass. Stop gradients through selection, preserve gradients within
   each TBPTT segment, randomize event cadence, and include 96/256/512-step
   image objectives plus bounded 4,096-step no-grad stability checks.
5. **Device topology.** Move conflict resolution, event selection, and
   conservative remapping onto the device so a topology event does not require
   a host synchronization. Keep the implemented support-bin policy refresh at
   bounded evaluation/checkpoint events until a device cost model is available.
6. **Promotion experiment.** Compare equal and continuous artifacts over the
   same 32 lizard seeds and at 1,024/3,070/4,096 visible-leaf budgets. Require
   the continuum, conservation, graph-recall, PSNR-tail, stability, and
   throughput gates above before moving a config from `sandbox` to `verified`.
