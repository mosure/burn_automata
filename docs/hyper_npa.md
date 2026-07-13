# HyperNPA 2D

## Maintained Path

The Burn-native `train-hyper2d-e2e-rollout` command owns training and quality
validation:

```text
RGBA image
  -> native 224x224 DINOv2 ViT-S/14 preprocessing on GPU
  -> CLS + 16x16 patch tokens
  -> structured multi-head adapter-layout decoder
  -> shared growing-2D NPA + generated rank-r LoRA
  -> recurrent NPA rollout
  -> alpha-aware Target2D image and density objective
```

The normal token contract is `257 x 385` (384 DINO channels plus alpha). The
optional high-fidelity contract is `257 x 388` and appends patch-aligned RGB
plus alpha. DINO runs without an autodiff graph in bounded batches; conditions
may remain device-resident. Every rollout batch row receives its own adapter.

## Generator Modes

- `token-attention-pool`: full adapter-space deterministic head.
- `module-token-decoder`: maintained v3 generalized path. Learned queries for
  the six NPA adapter parameter groups independently cross-attend to all DINO
  tokens in multiple channel heads, then emit a static adapter end to end from
  rollout loss.
- `module-token-decoder-v2`: legacy single-attention-map decoder retained only
  for loading and evaluating existing artifacts.
- `sample-id-table`: direct per-sample adapter substrate control. It memorizes
  identities, cannot infer an adapter for an unseen image, and is never
  evidence of HyperNPA generalization.

The maintained module-token decoder uses true per-head softmax token attention and the
`canonical-full-rank` adapter parameterization. For growing 2D, rank 82 is
full-rank for both NPA matrices. Canonical mode fixes one LoRA factor to an
identity embedding and predicts the other factor plus both bias deltas. This
is exactly equivalent to predicting dense NPA weight deltas, while reducing
the effective conditional output from 28,026 non-identifiable factor
coordinates to 12,946 identifiable delta coordinates. The serialized artifact
records the parameterization, and CPU/Bevy inference applies the same mask and
fixed factors as Burn training.

The historical artifact name contains `rectified_flow`, but the maintained
one-step deterministic generator is not a stochastic rectified-flow model. A
real adapter flow still requires noisy adapter states, timestep-conditioned
velocity training, and sampled-adapter rollout validation.

Optional per-step spatial control samples the DINO patch field at particle
positions. A recurrent-state projection can make that residual depend on the
current particle state. Both features are experimental and are serialized in
the artifact contract; the Bevy static image-to-NPA path rejects per-step-field
artifacts rather than silently dropping their control weights.

## Leakage Contract

Oracle adapters may be attached to training examples for bounded teacher
controls. Holdout examples are always teacher-free. Training aborts if a
holdout example carries an oracle adapter target.

This distinction invalidates an earlier interpretation of the released
rose/tropical-fish result. The reported 27.57 dB reconstruction came from a
generator trained on those exact adapters. It proves adapter reconstruction
and runtime correctness, not unseen-image generalization.

## Strict Held-Out Results

The latest identity-disjoint OmniSVG-1k split uses 900 train and 100 holdout
images. The fixed quality subset contains 16 holdout identities and is
evaluated with 2,048 particles at 96, 256, 512, and 1,024 rollout steps.
Checkpoint selection uses the minimum p10 over horizons of at least 256 steps.

| experiment | aggregate at 1,024 | worst-horizon p10 | condition shuffle gap | adapter gain | median throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| prior factorized module decoder | 10.88 dB | 8.07 dB | 2.12 dB | 2.51 dB | 3.5-5.2M particle-steps/s |
| canonical LoRA, 512p curriculum | 10.88 dB | 7.24 dB | 2.46 dB | 2.50 dB | 3.31M particle-steps/s |
| canonical LoRA, 1,024p composited refinement | **11.63 dB** | **9.30 dB** | **2.79 dB** | **3.26 dB** | 2.08M particle-steps/s |
| lower-LR continuation | 11.45 dB | 8.95 dB | retained | retained | 2.0M particle-steps/s |

The selected canonical checkpoint is temporally stable: horizon p10 is 9.13,
9.30, 9.35, and 9.32 dB at 96, 256, 512, and 1,024 steps. A deterministic
target-point splat of the same validation targets reaches 28.08 dB p10, so
particle count and target extraction do not explain the remaining 18.78 dB
tail gap.

A 16-image table control separates substrate capacity from held-out
generalization. At equal 2,000-step exposure, the canonical sample-ID table
reaches 11.09 dB aggregate, while the DINO module decoder reaches 10.99 dB
aggregate and 9.20 dB worst-horizon p10 with a 2.92 dB shuffle gap. The image
condition and generated adapter paths are therefore active; the broad gap is
not a batch broadcast bug or dead conditioner.

The selected 1k training sequence exposes each training identity to only about
111 rollout trajectories after the 512p pretrain and first 1,024p refinement.
The released upstream single-target recipe uses 240,000 trajectories per
target (10k epochs, three repetitions, batch size eight), 4,096 particles,
32-95 sampled steps from NumPy's `[32, 96)` range, a 256px splat loss, a 512-state pool, seed injection every
16 epochs, and brush damage 0.1. Current results therefore show a better
conditioned representation and curriculum, not oracle parity.

## Exposure-Matched Campaign

The maintained exposure campaign is bundled under
`configs/sandbox/hyper_e2e/omnisvg_*_upstream_exposure_*.toml` and
`omnisvg_1k_exposure_stage_*.toml`. It proceeds only when the prior gate passes:

1. A 16-identity `sample-id-table` control jointly trains the shared base and
   canonical full-rank dense deltas. The maintained 512-, 1,024-, and
   2,048-particle continuations together execute 104.17% of the upstream
   per-identity particle-update exposure. Each identity owns 512 persistent
   pool states, replaces one seed per 128 trajectories, and uses
   live-particle-centered brush damage. A bounded 4,096-particle cycle then
   tests whether matching particle density matters beyond total exposure. Its
   WGPU pool uses 256 states per identity so the monolithic recurrent-state
   buffer remains below WGPU's 2 GiB single-buffer limit.
2. The table-trained base is frozen while the deterministic full-token DINO
   module decoder is trained on the same 16 identities. This must prove that
   conditional adapter generation, rather than the shared base alone, can
   recover the table control.
3. Only then does the 900-train/100-holdout DINO campaign run its 256, 512, and
   1,024-particle stages. A stochastic adapter flow remains deferred until the
   deterministic generator passes.

Trajectory count and particle-update exposure are reported separately. The
16-target control first runs 240,000 trajectories at 512 particles (12.5% of
upstream particle exposure), another 240,000 at 1,024 particles (25%), and
320,000 at 2,048 particles (66.67%). The three planned 1k stages total 40,000
trajectories per identity but only about 2.29% of upstream per-identity
particle exposure; their justification is amortized learning across 900
identities, not compute parity with 900 separate oracles.

The completed table controls show that particle density is not
interchangeable with total particle-update exposure. All rows use the same 16
targets, canonical full-rank adapters, and p4096 validation at 96, 256, 512,
and 1,024 steps:

| training continuation | aggregate at 1,024 | worst selected-horizon p10 | shuffle gap | adapter gain | median training throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512p, 240k trajectories/identity | 13.06 dB | 10.82 dB | 4.85 dB | 4.35 dB | 13.26M particle-steps/s |
| +1,024p, 240k trajectories/identity | 15.87 dB | 13.24 dB | 7.79 dB | 7.18 dB | 10.21M particle-steps/s |
| +2,048p, 320k trajectories/identity | **20.71 dB** | **17.93 dB** | **12.60 dB** | **12.09 dB** | 6.90M particle-steps/s |
| +4,096p density-matched cycle | 20.71 dB | 17.93 dB | 12.60 dB | 12.09 dB | 4.11M particle-steps/s |
| frozen base, table-only LR refinement | **23.58 dB** | **18.98 dB** | active | active | bounded probe |

The p2048 run completed the full compute-matched exposure, but its selected
checkpoint is step 8,500. The p4096 run selected its initial checkpoint after
later updates regressed. Freezing the trunk and refining only table rows raised
aggregate quality, proving optimizer interference and trunk drift, but the
18.98 dB tail remained far from the 26 dB gate. These controls establish an
adapter-substrate ceiling and expose optimization faults; they do not form the
generalized model or a basis used at inference.

The final gate evaluates all 100 identity-disjoint holdouts with 4,096
particles at 96, 256, 512, and 1,024 steps. Both aggregate and p10 composited
PSNR must be at least 26 dB at every selected horizon of 256 steps or longer,
and generated adapters must outperform both the shared base and shuffled
conditions. Frequent validation uses one representative horizon to avoid
rerunning the same rollout prefixes.

## Training Throughput

The synchronized July 2026 CUDA profile uses an RTX PRO 6000 Blackwell,
release mode, 64 independent image conditions, 1,024 particles, and full BPTT.
Aggregate throughput divides exact optimizer particle-step exposure by time
between synchronized report boundaries, including kernel warmup but excluding
source staging, validation, and checkpoint writes.

| configuration | aggregate particle-steps/s | median steady interval | peak VRAM |
| --- | ---: | ---: | ---: |
| B64, random 32-96 steps, release/warm cache | 9.28M | 11.10M | 15.0 GiB |
| B64, fixed 96 steps, recomputed perception VJP | 14.24M | 14.91M | 15.0 GiB |
| B64, fixed 96 steps, retained perception VJP | **17.66M** | **18.36M** | 16.8 GiB |
| B128, fixed 96 steps, retained perception VJP | 17.30M | 17.83M | 29.6 GiB |

Fixed 96-step rollouts amortize condition generation, image loss, optimizer,
and pool persistence over more useful dynamics work. Retaining the forward
raw SPH state gradient and correction inverse removes a duplicate hash-grid
neighbor traversal in backward: Nsight measured the old precompute kernel at
0.835 ms per invocation and the replacement elementwise VJP at 0.018 ms.
The 1024-particle isolated forward+VJP benchmark improved from 0.621 ms for
grid/density recomputation to 0.537 ms with retained state over 50 synchronized
repeats. B128 raises power and memory without increasing throughput, so B64 is
the validated knee. `cubecl.toml` persists CubeCL compilation and autotune
caches under ignored `target/`; production training should use the release
CUDA binary and
`configs/verified/2d/hyper_e2e/throughput_omnisvg_64_b64_p1024_s96_cuda.toml`.

## Quality Contract

Quality reports include composited, raw-render, foreground, and density PSNR;
density soft IoU; generated-over-base and correct-over-shuffled gaps; adapter
norm and pairwise diagnostics; and optional nearest-training-oracle controls.
High-quality catalog validation uses 4,096 particles, 1,024 steps, update
probability 0.5, seed 42, and official transparent targets.

## Current Boundary

Proven:

- GPU DINO token extraction, including alpha and optional patch RGB;
- per-sample batched LoRA application;
- teacher-free identity splits and leakage rejection;
- condition-sensitive adapters and spatial control;
- bounded WGPU training with device Target2D/perception adjoints.

Not proven:

- greater than 26 dB unseen-image HyperNPA quality;
- a broadly useful shared NPA trunk over OmniSVG;
- parity between direct per-sample adapters and quality-scale NPA oracles;
- stochastic rectified-flow adapter generation.

The next gate is the v3 full-token, structured multi-head decoder trained
directly through rollout loss with a frozen, independently validated trunk.
It must first pass 16- and 64-image controls with a substantial correct-versus-
shuffled condition gap and improve unseen identities, then run the 900/100
identity-disjoint 1k split. Table controls remain diagnostic ceilings only; no
table row or sample identifier participates in the generalized inference
path. A true rectified-flow adapter generator remains a later option after the
deterministic conditioned path demonstrates generalization. Without adapter
endpoint/velocity supervision, calling the current end-to-end decoder
"rectified flow" would be inaccurate.
