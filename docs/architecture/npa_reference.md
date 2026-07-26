# NPA Reference Notes

The reference implementation at <https://selforg-npa.github.io/> is PyTorch/CUDA based. The core loop is:

1. Seed particles in 2D.
2. Compute SPH perception: state, blurred state, state gradients, and density gradients.
3. Apply a small MLP to produce position and state updates.
4. Integrate positions and states with stochastic update masking.
5. Train against rendered target images or task-specific losses.

This repository mirrors those semantics in CPU Rust first. That gives the future GPU kernels a stable oracle and lets imported checkpoints fail early on shape/parameter mismatches.

## SPH Parity

The Rust CPU perception path matches the upstream inference formulas:

- Density and blur use the normalized poly6 kernel.
- State and density gradients use the normalized spiky gradient kernel.
- State gradients use the upstream hybrid moment-matrix correction in the forward pass.
- State gradients are scaled by `eps / eps0`.
- Density gradients are scaled by `(eps / eps0)^(1 + dim) / particle_count`.
- Optional log normalization is applied per vector before the MLP.
- Position deltas are scaled by `alpha * eps / (1 + ||dx||)`.

`NpaConfig::equivariance` controls these scale factors:

| mode | state/density scale | particle-count density normalization | motion epsilon |
| --- | --- | --- | --- |
| `None` | disabled | disabled | `eps0` |
| `ParticleDensity` | disabled | enabled | `eps0` |
| `ParticleDensityAndScale` | enabled | enabled | grid `eps` |

`ParticleDensityAndScale` is the default for imported and seeded NPA models because it matches the reference implementation. The supported modes preserve permutation/translation semantics from local neighbor aggregation and the reference particle-density/scale normalization. The upstream paper does not enforce rotational equivariance for the NPA update; this crate does not add an SO(2)/SO(3)-equivariant vector-neuron model on top of the imported checkpoints.

Reference growing-model rollout defaults are `dt = 1.0`, `update_prob = 0.5`, `eps = eps0 = 0.1`, 4096 particles, and circular seed radius `0.2`. The deterministic parity harness uses `update_prob = 1.0` intentionally so CPU, WGPU, and the dependency-free Python oracle can compare exact trajectories. The Bevy viewer uses the reference stochastic `0.5` update probability by default.

The unit test `perceive_matches_upstream_sph_kernel_constants` pins the
normalized kernel constants.
`scripts/reference/selforg/validate_import_parity.py` runs end-to-end CPU or
WGPU rollout comparisons between Rust BPK inference and a dependency-free
Python implementation of the same formulas, and reports deterministic 2D
raster PSNR for imported 2D models.

## Import Contract

The primary importer reads PyTorch zip checkpoints directly from `.pth` or `.pt` files:

1. Locate tensor storages under `data/0..4`.
2. Interpret the first two linear layers as row-major MLP weights and biases.
3. Infer perception, hidden, update, state, and dimensionality metadata.
4. Write a checksumed `.bpk` package with explicit config, hashgrid, weights, and source metadata.

Fetch and export the pinned reference fixture with:

```bash
scripts/reference/selforg/fetch_selforg_npa.sh
python3 scripts/reference/selforg/export_selforg_npa_fixture.py --help
```

Python remains only at this external-reference boundary. Runtime checkpoint
import, `.bpk` persistence, inference, and training are Rust/Burn paths.
