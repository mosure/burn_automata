# Documentation

Documentation is organized by ownership rather than chronology. Generated
checkpoints, datasets, raw run directories, and local reports remain under
ignored `artifacts/`, `models/`, `data/`, and `target/` paths.

## Publications

| Paper | PDF | Sources and evidence |
| --- | --- | --- |
| HyperNPA: Amortized Image-Conditioned Neural Particle Automata | [`hyper_npa.pdf`](papers/hypernpa/hyper_npa.pdf) | [`papers/hypernpa/`](papers/hypernpa/) |
| Budgeted Adaptive Neural Particle Automata | [`adaptive_npa.pdf`](papers/adaptive/adaptive_npa.pdf) | [`papers/adaptive/`](papers/adaptive/) |

The PDFs are versioned deliverables. Each paper directory contains its LaTeX,
bibliography, renderer captures, and a companion README that states the exact
claim boundary.

## Architecture

- [`architecture/npa_reference.md`](architecture/npa_reference.md): official
  SelfOrg-NPA semantics and import/parity contract.
- [`architecture/kernel_strategy.md`](architecture/kernel_strategy.md):
  optimized CPU/WGPU/CUDA kernel ownership and dispatch evidence.
- [`architecture/gpu_interop.md`](architecture/gpu_interop.md): Burn, WGPU,
  Bevy, and Gaussian-splatting buffer ownership.

## Research Status

- [`research/hypernpa_status.md`](research/hypernpa_status.md): published
  deterministic checkpoint versus the maintained conditional row-flow path.
- [`research/adaptive_continuous_scale.md`](research/adaptive_continuous_scale.md):
  conservative scale, topology, bandwidth, and renderer contracts.
- [`research/3d_status.md`](research/3d_status.md): current 3D boundary and
  remaining promotion work.

## Development

- [`development/validation.md`](development/validation.md): canonical local,
  GPU, web, publication, and configuration checks.
- [`../configs/README.md`](../configs/README.md): verified versus sandbox
  experiment policy.
- [`../scripts/README.md`](../scripts/README.md): operational script ownership.

## Evidence

Machine-readable reports cited by maintained papers or contracts live under
[`evidence/`](evidence/). Reports without a current consumer are not repository
documentation and should remain in ignored experiment output directories.
