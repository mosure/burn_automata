# Image-Conditioned HyperNPA Quality Status

Evidence refreshed: 2026-07-26.

Broad feed-forward image-to-NPA generalization above 26 dB is **not achieved**.
The repository has two artifact boundaries that must not be conflated:

1. The selected published quality checkpoint is a deterministic DINO
   spatial-token to dense-controller generator.
2. The maintained trainer is a conditioned row-flow model with noisy source
   rows, timestep-conditioned velocity prediction, and an eight-step Heun
   solve.

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
  -> eight-step deterministic Heun solve
  -> 10,898 identifiable dense NPA controller residuals
  -> jointly trained shared growing-NPA trunk
  -> recurrent rollout and alpha-aware Target2D loss
```

The primary objective is teacher-free and end to end. Optional adapter-bank
pretraining is a warm-start/control path, not the generalization objective.
Detached TBPTT generates one controller endpoint per optimizer step,
accumulates rollout VJPs on device, and contracts the summed endpoint gradient
through the condition/flow graph once.

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
