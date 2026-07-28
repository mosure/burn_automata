# Image-Conditioned HyperNPA Quality Status

Evidence refreshed: 2026-07-27.

Broad feed-forward image-to-NPA generalization above 26 dB is **not achieved**.
The repository has two artifact boundaries that must not be conflated:

1. The selected published quality checkpoint is a deterministic DINO
   spatial-token to dense-controller generator.
2. The maintained trainer is a conditioned row-flow model with noisy source
   rows, timestep-conditioned velocity prediction, and a four-step Heun solve
   with eight velocity evaluations.

The row-flow path has bounded controllability and throughput evidence, but no
completed identity-disjoint 1k or 10k run establishes oracle-quality
generalization. The numerical paper results therefore remain attached to the
deterministic checkpoint.

## Maintained Architecture

```text
RGBA image at 224 x 224
  -> frozen GPU DINOv2 ViT-S/14
  -> 257 spatial/CLS tokens
  -> aligned RGBA patch payload
  -> conditioned row-flow velocity transformer
       noisy controller rows + timestep + image tokens
  -> four-step deterministic Heun solve (eight velocity evaluations)
  -> 10,880 identifiable dense NPA controller residuals
  -> jointly trained shared growing-NPA trunk
  -> recurrent rollout and alpha-aware Target2D loss
```

The primary objective is teacher-free and end to end. Optional adapter-bank
pretraining is a warm-start/control path, not the generalization objective.
Detached TBPTT generates one controller endpoint per optimizer step,
accumulates rollout VJPs on device, and contracts the summed endpoint gradient
through the condition/flow graph once.

The maintained rollout objective optionally applies `log1p` to each trajectory
loss, computes a differentiable top-tail mean independently within each image,
and then computes a second tail over identity-level losses. This avoids letting
one identity consume the entire trajectory tail while retaining pressure on
the hardest identities. Background, foreground, density, and composited-RGB
weights remain explicit Target2D terms.

Validation holds the sampled identity subset fixed across rollout seeds.
Checkpoint selection pools every identity-by-seed trajectory from the selected
horizons and computes one p10. Per-horizon p10, worst-seed p10, density PSNR,
and soft IoU remain diagnostics rather than being collapsed into one favorable
seed.

## Strict Catalog Row-Flow Control

The maintained zero-output-bias path now passes a ten-condition controller
generation control against all released Growing checkpoints. The shared lizard
trunk is frozen. Phase one fits random-time source-to-teacher velocities;
phase two minimizes normalized endpoint-row error through the same eight-step
Heun solver serialized for inference. Raw LoRA factors and the deprecated
per-image output bias are not trained.

At 4,096 particles, a 256-pixel metric raster, seed 42, and autonomous rollout
through step 1,024:

| Horizon | Generated aggregate | Exact teacher aggregate | Generated p10 | Exact teacher p10 |
| ---: | ---: | ---: | ---: | ---: |
| 96 | 20.23 dB | 20.18 dB | 18.23 dB | 18.09 dB |
| 256 | 21.91 dB | 21.72 dB | 19.45 dB | 19.43 dB |
| 512 | 23.01 dB | 22.74 dB | 20.65 dB | 19.43 dB |
| 1,024 | **23.38 dB** | **23.62 dB** | 20.17 dB | 20.11 dB |

The generated checkpoint also records 21.46 dB density PSNR, 0.948 mean
density soft IoU, a 13.49 dB correct-versus-shuffled-condition gap, and a
15.15 dB generated-versus-base gain. Peak-to-final p10 drift is 0.48 dB.
Actual Bevy/WGPU exports preserve distinct lizard, rose, and sun topology from
steps 96 through 1,024 rather than converging to one shared attractor.
The exact released lizard and sun controls themselves score 19.92 and
20.13 dB under this strict composited metric, so an all-identities 22 dB gate
would reject the reference dynamics. Parity is therefore reported against the
matched aggregate and p10 teacher metrics rather than relabeling that gate.

The decisive correction was endpoint supervision. Random-time velocity
matching alone reached normalized flow MSE `2.50e-4` after 1,000 updates but
only 14.39 dB at the quality horizon. Backpropagating through the eight-step
Heun endpoint reduced normalized endpoint MSE to `1.19e-5` and reached the
quality result above in 250 further updates. Phase one sustained about 190
condition-time examples/s at 459 W. Endpoint refinement sustained 10.9
examples/s at 99% GPU utilization, about 432 W, and 50.5 GiB device memory.

This proves high-quality conditional control and solver fidelity on a
ten-condition train/control bank. It is not identity-disjoint generalization.
The reproducible configs are
`flow/production_contract_growing_catalog_row_flow_pretrain.toml` followed by
`flow/production_contract_growing_catalog_row_flow_endpoint_refine_cuda.toml`.

## Credit-Horizon And Throughput Control

A matched-memory OmniSVG-1k continuation compared two detached-TBPTT shapes
from the same step-7,525 row-flow checkpoint with the same optimizer rates,
4,096 particles, sampled 32--95-step rollouts, and 250 optimizer steps.

| Shape | Aggregate PSNR delta | P10 delta | Optimizer time | Particle-steps/s |
| --- | ---: | ---: | ---: | ---: |
| 8 identities x 8 trajectories, TBPTT 32 | -0.06 dB | +0.26 dB | 270 s | 15.19M |
| 8 identities x 4 trajectories, TBPTT 64 | +0.22 dB | +0.48 dB | 227 s | 9.02M |
| TBPTT 64, two-step training flow | +0.16 dB | +0.49 dB | 151 s | 13.61M |

The longer credit horizon improved both mean and tail quality while completing
the fixed step budget 16% sooner. Raw particle-step throughput is lower because
the selected shape evaluates half as many trajectories, but quality progress
per optimizer wall time is materially better. The verified 1k production
contract therefore uses four trajectories per identity and 64-step detached
credit, keeping the same 8,388,608-particle-step live graph cap as the old
eight-trajectory, 32-step shape. Using two Heun steps for the training endpoint
while retaining four for serialized inference reduced optimizer time by
another 34% and preserved the held-out p10 gain. The four-step validation path
is the explicit fidelity gate for this train/serve approximation.

The endpoint-table control remains much faster: an 8-by-8, 4,096-particle
100-step continuation sustained 65.2M particle-steps/s and 99--100% sampled SM
occupancy at roughly 374--410 W. Its stable `1e-4` update improved held-out p10
from 11.42 to 11.51 dB; `3e-4` regressed p10 to 10.80 dB. This proves the NPA
rollout kernels can saturate the device and identifies conditioned row-flow
forward/backward as the remaining throughput cost. It does not establish an
adequate adapter-substrate ceiling.

A subsequent 500-step continuation of the selected two-step-training-flow
checkpoint reduced sampled training loss by 0.76 but left held-out quality
flat: aggregate PSNR moved 13.23 to 13.21 dB and p10 moved 10.78 to 10.69 dB.
Freezing the shared trunk over the same continuation did not repair the gap:
aggregate moved 13.20 to 13.24 dB while p10 fell from 10.73 to 10.51 dB.
Ongoing trunk adaptation is therefore useful for tail quality, but neither
branch converts additional exposure into held-out PSNR at the required rate.
This is a conditional generalization/sample-efficiency plateau.

## Tail/Density Refinement Control

A four-identity, 4,096-particle refinement control tested the objective on the
upstream-compatible zero-output-bias row-flow checkpoint. It used eight
trajectories per identity, 16-step detached TBPTT, 25% fresh trajectories,
sampled 32--95-step loss windows, DINO spatial tokens, and four validation
seeds. The accepted 300-step stage sampled pre-rollout ages from 0--448.

| Metric | Initial | Selected step 11,550 | Delta |
| --- | ---: | ---: | ---: |
| Pooled p10, steps 512 and 1,024 | 15.07 dB | 15.82 dB | +0.75 dB |
| Step-512 aggregate PSNR | 18.21 dB | 18.77 dB | +0.56 dB |
| Step-512 p10 PSNR | 14.69 dB | 16.02 dB | +1.32 dB |
| Step-512 density PSNR | 18.25 dB | 19.15 dB | +0.90 dB |
| Step-512 density soft IoU | 0.906 | 0.910 | +0.004 |
| Step-1,024 aggregate PSNR | 19.65 dB | 19.35 dB | -0.30 dB |
| Step-1,024 p10 PSNR | 16.86 dB | 15.81 dB | -1.05 dB |
| Step-1,024 density soft IoU | 0.914 | 0.915 | +0.001 |

The accepted stage sustained a median 23.98 optimizer examples/s and 29.43M
particle-steps/s on the RTX PRO 6000 Blackwell. Generated adapters beat the
shared base by 11.16 dB and correct conditions beat shuffled conditions by
11.27 dB at the final horizon. A subsequent 224--448 age balance stage raised
direct endpoint quality to 20.87 dB but reduced pooled rollout p10 to 15.21 dB,
so checkpoint selection rejected it.

This is evidence that the hierarchical objective and multi-seed selection
improve a small refinement control. It is not held-out generalization evidence,
does not reach 26 dB, and does not replace the 1k/10k gate below.

## Measured Quality Baseline

The selected deterministic OmniSVG-1k checkpoint reports:

| Metric | Result | Boundary |
| --- | ---: | --- |
| Aggregate composited PSNR | 11.63 dB | 16 of 100 held-out identities |
| Worst-horizon p10 | 9.32 dB | 2,048 particles, through step 1,024 |
| Correct vs shuffled condition | +2.79 dB | Conditioning is active |
| Generated residual vs shared base | +3.26 dB | Controller path is active |
| Target-point-splat p10 | 28.08 dB | Rasterization capacity control |
| Rose HyperNPA / released oracle | 9.63 / 27.93 dB | 4,096 particles, step 1,024 |
| Fish HyperNPA / released oracle | 11.65 / 27.45 dB | 4,096 particles, step 1,024 |

These controls show that batching, condition routing, adapter materialization,
Target2D rendering, and long rollout execution function. They do not show
high-quality unseen-image generation.

## Adapter-Table Boundary

The per-identity adapter table is an auto-decoder capacity control, not a
generalized model. A historical 16-identity run reached 23.58 dB but trained a
per-image output bias absent from the released Growing NPA; it is excluded
from parity claims. The corrected zero-output-bias four-identity control
reached 11.63 dB aggregate and 10.52 dB p10 after 2,000 updates. Correct
identity routing beat a shuffle by 3.77 dB, proving control was active, but the
run used only 6.67% of the released single-target trajectory exposure and is
not a converged substrate ceiling.

## Current Bottleneck

The ten-model control resolves conditional-controller capacity, per-sample
routing, flow-solver fidelity, and released-model dynamics under the maintained
zero-output-bias contract. The unresolved problem is learning an equally
strong shared-trunk/controller substrate from broad image data, then preserving
that quality on identity-disjoint conditions. Full-rank canonical controller
residuals can represent arbitrary dense updates to both growing-NPA affine
layers; the current weak OmniSVG result is therefore not evidence that nominal
adapter rank is the limiter.

The next broad quality gate is:

1. Expand the parity-valid, upstream-compatible shared-trunk/controller
   substrate beyond the ten released controls at 4,096 particles and sampled
   32--95-step training rollouts.
2. Warm-start row flow with sampled-endpoint supervision, then train it against
   functional rollout loss without raw non-identifiable LoRA-factor regression.
3. Evaluate correct, shuffled, zero-controller, and base-only controls.
4. Report aggregate and p10 PSNR at 96, 256, 512, 1,024, and bounded 4,096
   steps against independently trained oracle NPAs.
5. Require identity-disjoint 1k evidence before scaling a quality claim to
   10k.

No README, paper, or verified config should claim generalized HyperNPA parity
until those gates pass.
