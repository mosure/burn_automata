# Documentation

This directory separates current publications, maintained contracts, and
immutable experiment evidence. Generated checkpoints and local datasets remain
under gitignored `artifacts/`, `models/`, and `data/`.

## Papers

| Publication | PDF | Source and evidence |
| --- | --- | --- |
| HyperNPA: Amortized Image-Conditioned Neural Particle Automata | [`hyper_npa.pdf`](hyper_npa.pdf) | [`hyper_npa.md`](hyper_npa.md), [`hyper_npa.tex`](hyper_npa.tex), [`hyper_npa_figures/latest/`](hyper_npa_figures/latest/) |
| Budgeted Adaptive Neural Particle Automata | [`adaptive_npa.pdf`](adaptive_npa.pdf) | [`adaptive_npa.md`](adaptive_npa.md), [`adaptive_npa.tex`](adaptive_npa.tex), [`adaptive_npa_figures/`](adaptive_npa_figures/) |

Paper PDFs are versioned deliverables. LaTeX intermediates remain ignored.
Each companion document states the exact claim boundary and primary reports.

## Maintained Contracts

- [`npa_reference.md`](npa_reference.md): official SelfOrg-NPA parity contract.
- [`hypernpa_dino_flow_quality_status.md`](hypernpa_dino_flow_quality_status.md):
  current deterministic-checkpoint and row-flow status.
- [`adaptive_continuous_scale.md`](adaptive_continuous_scale.md): continuous
  measure, topology, bandwidth, and rendering semantics.
- [`kernel_strategy.md`](kernel_strategy.md): optimized inference/training
  kernel ownership and parity gates.
- [`gpu_interop.md`](gpu_interop.md): Burn, WGPU, Bevy, and Gaussian-splatting
  buffer ownership.
- [`validation.md`](validation.md): validation inventory and commands.

## Evidence

Machine-readable experiment summaries live in [`benchmarks/`](benchmarks/).
They are retained when a current paper or maintained contract cites them.
Large raw run directories, model weights, and local report matrices are not
documentation and remain outside git.

The adaptive paper's central reports are:

- [`adaptive_continuous_scale_ratio4_2026-07-25.json`](benchmarks/adaptive_continuous_scale_ratio4_2026-07-25.json)
- [`adaptive_event_aware_long_horizon_2026-07-25.json`](benchmarks/adaptive_event_aware_long_horizon_2026-07-25.json)

The HyperNPA paper's curated renderer evidence is
[`hyper_npa_figures/latest/`](hyper_npa_figures/latest/). Earlier direct-basis
and duplicate renderer generations were removed because they are superseded
and are not cited by a maintained paper.
