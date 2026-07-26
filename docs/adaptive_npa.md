# Adaptive NPA Paper Companion

The current manuscript is
[`adaptive_npa.tex`](adaptive_npa.tex), with references in
[`adaptive_npa.bib`](adaptive_npa.bib), figures under
[`adaptive_npa_figures/`](adaptive_npa_figures/), and the compiled publication
at [`adaptive_npa.pdf`](adaptive_npa.pdf).

It supersedes the planning-only `Budgeted_Adaptive_NPA_Paper_v5.pdf` draft.
The theory has been narrowed to claims supported by the current repository:
a lizard-specific, fixed-budget, continuous-material-scale implementation with
device-resident topology, event-aware training, broad quality evaluation, and
matched Bevy/WGPU captures.

## Current Claim Boundary

| Claim | Evidence | Boundary |
| --- | --- | --- |
| Adaptive material is real | 33 occupied scale bins; 2.001x material and Gaussian radius span | One lizard-specific ratio-four candidate |
| Material is conservative | Maximum numerical relative error `8.31e-7`; renderer audit `1.22e-7` | Floating-point audit, not a nonlinear convergence proof |
| Quality is competitive at bounded horizons | `26.12 dB` at step 256 and `26.32 dB` at step 512 | 32 seeds; mean metric |
| Aggregate oracle gap is small | `-0.16 dB` over 128 seed-horizon rows | Worst row is `-3.57 dB` |
| Interaction work falls | `74.95%` of the 4,096-row control | Mean wall time is still `96.27%` |
| Allocation is active | 2,048 accepted exchanges and positive scale-detail correlation gain | Dynamic-over-static PSNR gain is only `+0.014 dB` |
| Long-horizon parity is unresolved | Step-1,024 mean `24.97 dB`; worst drift `2.31 dB` | Candidate is not promoted |

The paper does **not** claim generalized adaptive NPA, arbitrary runtime
birth/death, variable-bandwidth quality, anisotropic Gaussian decoding,
from-scratch official-trainer parity, or a production speedup.

## Primary Evidence

```text
docs/benchmarks/adaptive_continuous_scale_ratio4_2026-07-25.json
docs/benchmarks/adaptive_event_aware_long_horizon_2026-07-25.json
configs/verified/adaptive/recurrent_target2d_lizard_continuous_ratio4_smoke_3070_2d_wgpu.toml
docs/adaptive_npa_figures/manifest.json
```

The broad quality matrix covers seeds 42 through 73 and rollout horizons 96,
256, 512, and 1,024. Numerical PSNR comes from the Target2D renderer. Paper
figures are independent 768 x 768 captures from the release
`bevy_automata export` and `bevy_gaussian_splatting` path.

## Build

```bash
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error adaptive_npa.tex
```

Publication validation:

```bash
pdfinfo adaptive_npa.pdf
pdftotext adaptive_npa.pdf - | rg \
  'Budgeted Adaptive|Morphology Quality|Reproducibility'
rg 'undefined|LaTeX Error|Overfull \\hbox|Overfull \\vbox' adaptive_npa.log
```
