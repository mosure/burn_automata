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

The unresolved problem is quality-scale rollout optimization and robust
conditional controllability, not DINO staging or nominal adapter rank.
Full-rank canonical controller residuals can represent arbitrary dense updates
to both growing-NPA affine layers. More data alone is not justified until the
same online objective can fit a multi-identity direct-controller control close
to independent oracle quality under the released no-output-bias contract.

The maintained row-flow path adds the requested noisy-source and timestep
conditioned architecture, but architecture labels are not evidence. Its next
quality gate is:

1. Establish a parity-valid, upstream-compatible shared-trunk/controller
   ceiling at 4,096 particles and sampled 32--95-step training rollouts.
2. Train row flow against functional rollout loss, optionally warm-started
   from that ceiling, without raw non-identifiable LoRA-factor regression.
3. Evaluate correct, shuffled, zero-controller, and base-only controls.
4. Report aggregate and p10 PSNR at 96, 256, 512, 1,024, and bounded 4,096
   steps against independently trained oracle NPAs.
5. Require identity-disjoint 1k evidence before scaling a quality claim to
   10k.

No README, paper, or verified config should claim generalized HyperNPA parity
until those gates pass.
