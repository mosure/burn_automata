# HyperNPA Paper Companion

The publication source is [`hyper_npa.tex`](hyper_npa.tex), with references in
[`hyper_npa.bib`](hyper_npa.bib) and renderer evidence under
[`hyper_npa_figures/latest`](hyper_npa_figures/latest). The generated local PDF
is `docs/hyper_npa.pdf`.

This document records the claim boundary and the exact evidence used by the
paper. It is not a second manuscript.

## Evaluated Architecture

The selected quality checkpoint implements:

```text
RGBA image at 224 x 224
  -> frozen DINOv2 ViT-S/14
  -> 257 tokens x (384 DINO + 3 RGB + 1 alpha)
  -> module-token cross-attention decoder (artifact architecture v2)
  -> 10,898 identifiable full-rank controller deltas
  -> shared 66 -> 128 -> 18 growing-2D NPA
  -> recurrent particle rollout
  -> alpha-aware Target2D loss
```

The selected decoder is deterministic. Historical artifact names containing
`rectified_flow` do not make it a rectified-flow model: there is no noisy
adapter source, timestep-conditioned velocity objective, or multi-step adapter
sampling in the reported checkpoint.

The source tree now defaults to the v3 four-head module-token decoder. The
paper reports v3 only for the current CUDA throughput benchmark; it does not
attribute v3 to the older v2 quality checkpoint.

## Canonical Conditional Row-Flow Experiment

The maintained trainer implements the next architecture; it is not yet part of
the paper's quality claim:

```text
224 x 224 RGBA image
  -> frozen DINOv2 ViT-S/14, all 257 spatial/CLS tokens
  -> per-token-preserved DINO + aligned RGB/alpha condition projection
  -> deterministic seeded source rows for W1+b1 and W2+b2
  -> timestep-conditioned self/cross-attention velocity transformer
  -> eight-step deterministic Heun solve
  -> 10,898 identifiable dense controller residuals
  -> jointly trained shared NPA trunk + generated residual
  -> autonomous particle rollout
  -> alpha-aware Target2D image/density loss
```

The primary path is teacher-free and end to end. Image and recurrent-rollout
gradients update both the generated dense residual and shared NPA trunk. A
small self-rectification auxiliary detaches the current generated endpoint,
samples a continuous time on the source-to-endpoint line, and trains the
velocity field toward that endpoint; it reuses the endpoint already generated
for the rollout rather than paying for a second Heun solve. Optional endpoint
pretraining can still sample Gaussian controller rows and regress toward a
modest exact-oracle bank, but it is a warm start or control rather than the
generalization objective.

Teacher-free runs use a `1e-3` deterministic source-noise scale. This keeps the
initial residual near the pretrained shared trunk while row-normalized
velocity outputs remain free to grow to the scale demanded by rollout loss.

Row-wise RMS normalization is used without elementwise centering. Padding is
masked on ndarray, WGPU, and CUDA. The v5 architecture contract inside the v2
BPK container stores row widths, RMS scales, solver/source metadata, and
transformer tensors; it serializes neither JSON weights nor ambiguous raw LoRA
factors. Legacy v4 row-flow artifacts remain loadable.

The bounded endpoint-pretraining smoke is
`configs/verified/2d/hyper_e2e/smoke_conditional_row_flow.toml`. The canonical
teacher-free command smoke is `smoke_conditional_row_flow_e2e.toml`, and the
1k quality contract is
`production_omnisvg_1k_conditional_row_flow_e2e_cuda.toml`. The production
shape is an untrained contract, not evidence of broad 1k/10k generalization or
26 dB parity; those claims require completed held-out p10, condition-shuffle,
base-only, long-horizon, and matched-oracle gates.

A bounded four-identity CUDA control run establishes that the new conditional
path is active: at its selected checkpoint, correct conditions beat a cyclic
condition shuffle by 3.72 dB and generated residuals beat the shared trunk by
3.42 dB at 256 particles and 64 steps. The generated 10,898-value controller
rows have mean pairwise L2 distance 1.36. This is a controllability result, not
quality parity: p10 is 12.18 dB. A post-VJP-fix 2,000-step continuation trained
only at 64 particles still did not transfer to quality scale. Its selected
checkpoint reaches 10.96 dB p10 at 2,048 particles and 96 steps, where the
generated residual gain is 1.53 dB, then falls to 9.34 dB p10 and 0.02 dB gain
at 256 steps. The matched target-point-splat p10 is 28.42 dB. The corrected
production contract trains eight conditions with eight independent 4,096-
particle trajectories each. It samples rollout lengths from the upstream-style
exclusive range `32..96` (32 through 95) instead of optimizing a fixed 96-step
horizon. This preserves the previous aggregate 262,144 particles per optimizer
step while moving each trajectory to quality scale.

Long full-BPTT rollouts produce substantially larger startup gradients than the
short-horizon probes. Verified production configs linearly warm the optimizer
learning rate before applying the configured cosine or upstream schedule.
Gradient clipping alone is insufficient because Adam's first sign-normalized
update can still destroy a near-zero generated controller.

The E2E particle pool now enforces both configured seed policies. A global
cadence injects `seed_replacements_per_interval` fresh trajectories, while a
per-identity counter prevents frequently or infrequently sampled identities
from going indefinitely without a fresh seed. Both select explicit batch rows
and are checkpoint-contract inputs. Eight persistent slots are retained per
training identity, and localized state erasure remains active through the
configured brush perturbation.

Final quality validation reports held-out PSNR at 96, 256, and 512 steps. A
separate final-only stability pass measures the same continuous rollouts for 16
held-out conditions at 512 and 4,096 steps with detached parameters and
recurrent state chunks, generated adapters only, and no backward graph. Its
report includes aggregate and p10 PSNR drift, rendered occupancy drift,
particle/state overflow fractions, and mean particle motion over the final
256-step windows at both horizons.

The device perception VJP uses a stable small-radius series for
`log1p(r) / r`, includes the configured state-equivalence scale in dense,
sparse-channel, and sparse-plane forwards, and versions the corresponding
CubeCL/Fusion operation IDs. Component and aggregate VJP fixtures pass against
the analytic reference on both CUDA and WGPU with a 0.5% relative bound. The
shared dense Burn normalization uses the same series, avoiding the inaccurate
tiny-`log1p` lowering previously observed on WGPU.

## Claim-to-Evidence Matrix

| Claim | Evidence | Boundary |
| --- | --- | --- |
| Image-conditioned control is active | OmniSVG-1k correct-vs-shuffled gap: 2.79 dB | Fixed 16-image subset of 100 identity-disjoint holdouts |
| Generated deltas affect dynamics | OmniSVG-1k generated-vs-base gain: 3.26 dB | Same validation contract as above |
| Current broad quality is below parity | 11.63 dB aggregate and 9.32 dB p10 at step 1,024 | 2,048-particle numerical validation; all 16 samples miss 26 dB |
| Target extraction is not the dominant ceiling | 28.08 dB target-point-splat p10 | 4,096 extracted target points |
| Released oracles remain much stronger | Rose/fish HyperNPA: 9.63/11.65 dB; released oracles: 27.93/27.45 dB | Same target, seed, 4,096 particles, and 1,024 steps |
| Shared-base plus deltas has substantial capacity | 16-ID direct table reaches 23.58 dB aggregate and 19.30 dB p10 | Training identities only; this is memorization, not HyperNPA generalization |
| Current trainer has a high-throughput path | 17.66M measured and 18.36M median particle-steps/s | v3, CUDA, batch 64, 1,024 particles, 96 full-BPTT steps; quality disabled |

The paper intentionally does **not** claim:

- oracle-quality unseen-image generation;
- Burn-from-scratch oracle parity;
- rectified-flow adapter quality;
- a 1k result evaluated over all 100 holdouts or multiple rollout seeds;
- that the direct per-ID adapter table is an oracle or a generalized model.

## Primary Reports

The numerical tables are derived from these immutable local reports:

```text
artifacts/hyper2d_omnisvg_1k_dino_canonical_quality_refine/report.json
artifacts/hyper2d_growing_catalog_holdout2_quality_eval_p4096_s1024/report.json
artifacts/hyper2d_omnisvg_16_p4096_frozen_table_lr2e6/report.json
artifacts/throughput_v3_p1024_s96_b64_sources64_cuda_release_fixed96_retained_vjp/report.json
```

Each renderer directory in `docs/hyper_npa_figures/latest` also contains its
headless export report. Those reports record model paths, image conditions,
particle count, seed, update probability, requested capture steps, actual
capture steps, dimensions, and nonblank-pixel bounds.

## Rollout Contract

Every paper screenshot is an unmodified 512 x 512 capture from the same
Bevy/WGPU simulation and `bevy_gaussian_splatting` renderer used by the viewer:

```text
particles:          4,096
seed:               42
seed mode:          uniform circle
update probability: 0.5
rollout steps:      1,024
captures:           0, 32, 96, 256, 512, 1,024
render scale:       0.5
render opacity:     2.0
```

The paper includes eight identity-disjoint OmniSVG conditions, two paired
catalog holdouts with released target-specific NPA checkpoints, and the
released lizard checkpoint as a renderer sanity check. Numerical PSNR comes
from the differentiable Target2D renderer, not screenshot pixels. The 1k
metrics use 2,048 particles; the screenshots deliberately use 4,096 particles
and are labeled accordingly.

An equivalent HyperNPA export has this shape:

```bash
target/debug/bevy_automata export \
  --output-dir docs/hyper_npa_figures/latest/heldout/<sample> \
  --output-prefix <sample> \
  --particles 4096 \
  --steps 1024 \
  --capture-steps 0,32,96,256,512,1024 \
  --steps-per-frame 4 \
  --seed 42 \
  --update-prob 0.5 \
  --hyper-image data/omnisvg/mmsvg-illustration/train/<sample>.png \
  --hyper-base artifacts/hyper2d_omnisvg_1k_dino_canonical_quality_refine/shared_base.bpk \
  --hyper-model artifacts/hyper2d_omnisvg_1k_dino_canonical_quality_refine/hyper_2d.bpk \
  --dino-model models/dino/dino_vits.mpk
```

## Build and Validate

Build the manuscript with a TeX distribution containing `acmart`, TikZ, and
PGFPlots:

```bash
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error hyper_npa.tex
```

A publication check should confirm:

```bash
pdfinfo hyper_npa.pdf
pdftotext hyper_npa.pdf - | rg \
  'HyperNPA|Strict Released-Oracle Comparison|References'
rg 'undefined|LaTeX Error|Overfull \\hbox|Overfull \\vbox' hyper_npa.log
```

The checked source is self-contained for LaTeX compilation when
`hyper_npa.tex`, `hyper_npa.bib`, and `hyper_npa_figures/latest` are bundled
together. Model checkpoints and the OmniSVG cache are runtime artifacts and
are not part of the arXiv source archive.

## Interpretation

The result is a systems-and-method baseline with a useful negative finding.
Conditioning, per-sample controller materialization, long recurrent rollout,
GPU training, and renderer export all work. The learned attractors are stable
but usually coarse, generated controller deltas remain highly similar, and
the direct-table tail is itself below the 26 dB target. The evidence therefore
places the immediate research priority on a parity-validated recurrent
substrate and stronger conditional control, not on relabeling the current
deterministic decoder as a successful generalized flow model.
