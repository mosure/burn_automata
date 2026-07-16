# Image-Conditioned HyperNPA Quality Status

Current status: broad feed-forward image-to-NPA generalization above 26 dB is
**not achieved**. The maintained implementation is a deterministic
DINO-conditioned LoRA generator, not a trained stochastic rectified flow.

## Architecture

```text
RGBA image at 224x224
  -> frozen GPU DINOv2 ViT-S/14
  -> 257 spatial tokens with alpha and optional RGB
  -> adapter-layout queries with true multi-head token cross-attention
  -> canonical full-rank generated LoRA / dense NPA delta
  -> shared growing NPA
  -> recurrent rollout and alpha-aware Target2D loss
```

The optional per-step field adds local condition residuals at particle
positions. Its state-aware variant starts from an exact zero projection, so an
older state-independent checkpoint can be migrated without changing its
initial output.

## What The Controls Show

1. Teacher-seen released adapters can be reconstructed above 26 dB. This is a
   runtime/capacity control, not held-out generalization.
2. On a strict 8-train/2-holdout catalog split, 4,096-particle/1,024-step
   aggregate composited PSNR is 10.52 dB. The condition-shuffle gap is only
   0.09 dB.
3. The nearest training oracle selected by full DINO-token distance reaches
   only 8.90 dB on the two catalog holdouts.
4. Canonical full-rank LoRA removes factor gauge ambiguity and predicts exact
   dense NPA deltas through fixed identity factors. A 16-image DINO control
   reaches 9.20 dB worst-horizon p10 and a 2.92 dB condition-shuffle gap.
5. On the identity-disjoint OmniSVG-1k split, the selected 2,048-particle
   multi-horizon checkpoint reaches 11.63 dB aggregate and 9.30 dB
   worst-horizon p10. This improves the prior factorized tail by about 1.2 dB.
6. The selected checkpoint remains stable through 1,024 steps, but a lower-LR
   continuation regresses to 8.95 dB p10. More identical refinement is not a
   reliable path to parity.
7. Target-point splat p10 is 28.08 dB on the same rows. The remaining gap is in
   learned formation and image-specific dynamics, not target rasterization.

## Role Of The Adapter Table

The Burn vectorized sample-ID table is an auto-decoder control, not the
generalized architecture. It is useful for testing per-row adapter batching,
estimating the shared-trunk-plus-adapter ceiling, and optionally pretraining a
trunk. It memorizes one free vector per training identity and has no operation
that maps a new image to an adapter.

The historical quality-scale table reached 23.58 dB aggregate, but that
artifact trained a per-image output bias. The released Growing NPA has no final
output bias, so those results are retained only as a legacy optimization
diagnostic and cannot establish an upstream-compatible adapter ceiling. A
bounded matched control now exists: an upstream-seeded shared rule with
zero-output-bias rank-82 per-ID controllers, 4,096 particles, eight trajectories
per identity, and sampled 32--95-step rollouts reached 11.63 dB aggregate and
10.52 dB p10 at 1,024 steps after 2,000 updates. The correct-vs-shuffled gap was
3.77 dB, so the controller path was active. This run represents only 16,000
trajectories per identity, or 6.67% of the released 240,000-trajectory exposure;
it is not a converged ceiling. Full-exposure oracle parity remains required
before assigning the remaining gap to DINO or rectified-flow amortization.

## Bottleneck

The first bottleneck remains rollout optimization and conditional
controllability at quality scale, not DINO throughput or raw LoRA rank.
Canonical rank-82 adapters can represent arbitrary dense deltas for both
growing-NPA matrices. Condition ablations prove that generated adapters affect
the rollout, but the legacy decoder used only one attention map despite a
multi-head configuration.

The maintained `module-token-decoder` is now a versioned v3 path: each learned
adapter-layout query independently attends to all 257 DINO tokens in multiple
channel heads. `module-token-decoder-v2` retains the old single-map semantics
only for checkpoint compatibility. This is a deterministic amortized adapter
generator trained end to end through rollout loss; it is not yet a stochastic
rectified flow.

The table control is not an intrinsic LoRA expressivity verdict. It remains far
below the released trainer's 4,096-particle, 32-96-step, 256px-loss,
512-pool-state, brush-damage, 30k-epoch regime. The immediate engineering
constraint is making that regime practical in Burn without dense all-pairs
autodiff or host readback.

The current selected curriculum fixes the earlier long-horizon regression, but
uses roughly 111 trajectories per train identity versus 240,000 in the
upstream single-target recipe. Uniform sampling also gives no extra updates to
the hard validation tail. The next training change should add train-side
per-identity EMA/hard-example sampling and substantially more bounded exposure
before increasing conditioner size or dataset scale.

## Required Gate Before Scaling

1. Use direct per-sample adapters only as a bounded substrate/trunk ceiling at
   particle counts and rollout horizons that overlap quality validation.
2. Compare every selected row against an independently overfit NPA oracle under
   identical targets, seeds, particles, and rollout steps.
3. Require both aggregate and p10 oracle-relative quality to improve; do not
   select checkpoints from short-rollout loss.
4. Freeze the best validated trunk and train the v3 full-token generator
   directly against functional rollout loss. Do not regress raw table vectors
   as the primary objective.
5. Pass 16- and 64-image correct/shuffled/zero/base controls before the 900/100
   identity-disjoint 1k run.
6. Re-run zero, swap, shuffle, base-only, and identity-disjoint holdout gates.
7. Introduce noisy-state/timestep velocity training only after the
   deterministic generated-adapter path is competitive.

No paper, README, or verified configuration should claim generalized HyperNPA
quality until this gate passes on identity-disjoint 1k and 10k splits.
