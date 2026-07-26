# Adaptive NPA Status

Adaptive NPA is a research extension of the fixed 2D NPA path. It keeps the
same local recurrent rule and device-resident execution model while allowing a
fixed number of resident rows to represent different amounts of material.
The current implementation proves conservative continuous scale, bounded
topology changes, and matching Gaussian decoding. It does not yet prove a
generalized scale-free automaton or a quality advantage over fixed NPA.

The publication and its exact claim boundary are maintained under
[`../papers/adaptive/`](../papers/adaptive/).

## Current Result

The selected lizard candidate uses 3,070 visible and recurrent rows to
represent 4,096 fine-material units. Its retained broad evaluation covers 32
seeds and rollout horizons 96, 256, 512, and 1,024.

| Measurement | Result | Interpretation |
| --- | ---: | --- |
| Occupied material-scale bins | 33 | Scale is not a binary coarse/fine flag |
| Material and Gaussian radius span | 2.001x | Continuous decoding is active |
| Mean PSNR, step 256 | 26.12 dB | Competitive at a bounded horizon |
| Mean PSNR, step 512 | 26.32 dB | Competitive at a bounded horizon |
| Mean PSNR, step 1,024 | 24.97 dB | Long-horizon drift remains |
| Aggregate gap to fixed oracle | -0.16 dB | Mean over all seed/horizon rows |
| Worst oracle gap | -3.57 dB | Tail parity is not established |
| Interaction-work ratio | 74.95% | Fewer effective pair interactions |
| Wall-time ratio | 96.27% | Work reduction is not yet a system speedup |
| Dynamic-over-static PSNR gain | +0.014 dB | Allocation quality benefit is weak |

This candidate is not promoted. Its long-horizon tail and negligible
dynamic-over-static quality gain fail the stronger standard that adaptive NPA
should improve quality, compute, or both.

## Representation Contract

Each adaptive row owns a positive represented measure. In two dimensions the
material footprint radius is:

```text
radius = sqrt(represented_measure / pi)
```

The same measure drives:

- conservation checks during split, merge, and exchange events;
- the recurrent material-scale feature;
- isotropic Gaussian footprint and opacity compensation;
- interaction support when bandwidth adaptation is explicitly enabled.

Material scale, render scale, and interaction bandwidth are separate values.
The retained quality candidate varies material and render scale while keeping
communication bandwidth fixed. This isolates dynamic allocation from a change
in the accepted neighbor graph.

The current publication evaluates isotropic material Gaussians. Arbitrary
anisotropic covariance remains available in the renderer, but is not generated
by the promoted adaptive decoder and is not part of the adaptive claim.

## Topology Contract

Topology operations must preserve:

- total represented measure;
- represented-measure-weighted centroid;
- finite positive material and render footprints;
- deterministic count and remapping behavior;
- bounded event latency;
- recurrent state continuity across the event boundary.

The runtime supports conservative split, merge, and budget-neutral exchange
operations. The selected candidate reallocates represented measure among a
fixed 3,070-row resident budget. It does not allocate or free GPU rows during
steady rollout.

The progressive LoD control is a separate topology baseline. It expands from
1,024 to 4,096 fine leaves, then applies bounded mixed-arity restriction to
3,070 visible leaves. Persistent fine quadrature retains the underlying
4,096-row dynamics. This demonstrates variable visible count and conservative
rendering, but not reduced recurrent work.

## Scale Limits

The configured clamps are wider than the scale range reached by the retained
lizard candidate:

| Quantity | Configured range | Selected measured range |
| --- | ---: | ---: |
| Material footprint | 0.0015625 to 0.1 | approximately 2x radius span |
| Interaction bandwidth | 0.025 to 0.4 | fixed at 0.1 |
| Scale feature ratio | 0.25x to 4x | unsaturated |

The effective limit comes from the trained material spectrum and event policy,
not a renderer clamp. Wider 2.5x and 3x radius experiments reduced broad
quality. Enlarging the permitted range without training through those scales
is therefore not a valid fix.

## Training Contract

The maintained three-stage curriculum separates:

1. scale-conditioned recurrent-rule training;
2. long-tail recovery with older trajectory ages;
3. full-rule recovery through scheduled topology events.

Event-aware TBPTT must split chunks at exact topology boundaries. The
post-event segment contributes differentiable image and dynamics loss before
one optimizer update is applied. Validation selects checkpoints using broad
seed/horizon quality and drift, not the latest training loss.

The remaining training problem is stochastic long-horizon recovery after
allocation events. Additional schedule tuning is not sufficient. The next
model must learn an event-risk or future-loss signal that predicts whether an
exchange improves downstream morphology, then train the recurrent rule on the
resulting state distribution.

## Device Execution

Adaptive inference remains device resident:

```text
material state
  -> support-bin/hashgrid construction
  -> adaptive perception
  -> recurrent rule
  -> conservative topology event when scheduled
  -> represented-measure Gaussian decode
```

Support bins are selected dynamically. Concentrated multiscale distributions
can reduce candidate work, while diffuse distributions remain on the
single-bin path when additional scans would cost more than they save. Exact
accepted-neighbor parity is required before a binned path can be selected.

Renderer and dynamics buffers are allocated to the maximum resident capacity.
Changing active support bins or applying a topology event does not require a
host readback or per-step device allocation.

## Verified Configurations

| Scope | Configuration |
| --- | --- |
| Continuous-scale bounded evaluation | `configs/verified/2d/adaptive/evaluation/recurrent_target2d_lizard_continuous_ratio4_smoke_3070_2d_wgpu.toml` |
| Progressive LoD smoke | `configs/verified/2d/adaptive/evaluation/task_lod_lizard_smoke_3070_2d_wgpu.toml` |
| Progressive LoD full evaluation | `configs/verified/2d/adaptive/evaluation/task_lod_lizard_eval_3070_2d_wgpu.toml` |
| Resident-topology smoke | `configs/verified/2d/adaptive/evaluation/task_resident_lizard_smoke_3070_2d_wgpu.toml` |
| Three-stage recurrent training | `configs/verified/2d/adaptive/training/recurrent_target2d_lizard_stage{1,2,3}_*.toml` |
| Multiscale command smoke | `configs/verified/2d/adaptive/training/task_multiscale_lizard_smoke_2d_wgpu.toml` |
| Topology algebra smoke/full audit | `configs/verified/2d/adaptive/audits/continuous_topology_{smoke,full}.toml` |

Local scale sweeps, recovery experiments, and resume variants belong under the
ignored `configs/sandbox/` tree.

## Retained Evidence

Machine-readable reports cited by the paper or current contracts live in
[`../evidence/adaptive/`](../evidence/adaptive/):

- `adaptive_continuous_scale_ratio4_2026-07-25.json`
- `adaptive_event_aware_long_horizon_2026-07-25.json`
- `adaptive_lizard_progressive_mixed_lod_2026-07-20.json`
- `adaptive_continuous_topology_100k_2026-07-20.json`
- `adaptive_support_bins_wgpu_2026-07-20.json`
- `adaptive_recurrent_closure_causality_2026-07-22.json`
- `adaptive_recurrent_target2d_lizard_2026-07-23.json`

Unselected sweeps and generated checkpoints remain under ignored experiment
output directories. They are not repository documentation.

## Promotion Gates

An adaptive model may replace the fixed 2D baseline only when all of these are
true on matched particles, seeds, update probability, and rollout horizons:

1. Mean and worst-seed PSNR are noninferior through step 1,024.
2. Dynamic allocation materially outperforms the same scale spectrum with a
   static allocation.
3. Measure, centroid, finite-state, occupancy, overflow, and topology-count
   gates pass.
4. Renderer footprint exactly matches represented material on CPU and GPU.
5. The accepted communication graph matches the reference for each selected
   support policy.
6. End-to-end wall time improves or quality rises enough to justify the cost.
7. The saved adaptive artifact reproduces the result after reload in
   `bevy_automata`.

## Next Work

The ordered research path is:

1. Train an event-risk predictor against counterfactual future image loss.
2. Sample event ages and long post-event horizons broadly during recurrent
   training.
3. Require worst-seed drift improvement before expanding the scale range.
4. Introduce continuously varying interaction bandwidth only after fixed-
   bandwidth event recovery passes.
5. Evaluate variable resident counts only after the fixed-capacity path proves
   a quality or wall-time advantage.
6. Generalize the rule and allocation policy across targets only after the
   lizard control passes every promotion gate.
