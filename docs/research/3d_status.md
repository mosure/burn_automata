# 3D NPA Status

The repository maintains two distinct 3D tracks. The mesh-conditioned
surface-support path now passes its checked-in Utah teapot quality contract.
The stricter neutral-seed morphogenesis path remains experimental.

## Passing Mesh-Conditioned Path

The interactive and CLI workflow compiles an imported Wavefront OBJ into a
recurrent particle-state controller:

```text
OBJ mesh
  -> center and isotropically normalize to scale 0.72
  -> deterministic surface sampling
  -> 3D position-conditioned NPA state rule
  -> oriented anisotropic Gaussian decoding
  -> WGPU rollout and Bevy PanOrbit rendering
```

The fourth position component marks imported surface support as an anchor.
Particle positions therefore preserve the input surface while recurrent
normal, opacity, color, signed-distance, and latent channels remain active.
Training includes pristine surface lanes, localized state-erasure lanes at
multiple recovery ages, and bounded off-surface field lanes.

The material channels use a structured 16-hidden-unit initialization. It
represents the best affine position-to-color fit and exact residual dynamics
for liveness and opacity. The remaining hidden units train normally. The
structured subspace is re-projected before evaluation and export, preventing
small one-step material errors from accumulating during long rollouts.

This is a mesh-conditioned NPA state-field compiler. It is useful for imported
mesh visualization, recurrent material-state repair, and testing the common
3D inference stack. It is not equivalent to object growth from a compact seed.

## Canonical Teapot Result

The canonical target is `assets/meshes/utah_teapot.obj`, the Utah teapot
triangle mesh in a z-up coordinate system. Import converts it to Bevy y-up,
centers it, and maps its longest extent to 1.44.

```bash
cargo run --release -p burn_automata --features gpu_wgpu -- \
  train-mesh3d \
  --config configs/verified/3d/mesh/teapot_quality.toml
```

The verified run uses a 3D, 24-state, 320-hidden-unit NPA; 4,096 particles by
eight training trajectories; 500 AdamW steps; and a 16,384-particle,
three-seed quality evaluation at steps 0, 32, 96, and 256.

| Metric | Measured result |
| --- | ---: |
| Dataset staging | 0.216 s |
| Training time | 9.006 s |
| Training rows | 16,384,000 |
| Training throughput | 1.819M rows/s |
| Density PSNR | 25.91-26.29 dB |
| Color PSNR | 43.04-43.41 dB |
| Depth PSNR | 41.17-41.82 dB |
| Worst damaged-region color PSNR at step 32 | 52.48 dB |
| Mean surface distance | at most 0.000469 |
| p95 surface distance | at most 0.002869 |
| Target coverage | 100% |
| Long-horizon position drift | 0 |

The actual 16,384-particle Bevy/WGPU recovery captures are in
`docs/papers/hypernpa/figures/mesh3d/teapot_recovery.png`.

The resident inference benchmark includes Gaussian-buffer generation and final
readback:

| Particles | Median step | p95 step | Steps/s |
| ---: | ---: | ---: | ---: |
| 4,096 | 1.089 ms | 1.431 ms | 918 |
| 16,384 | 9.295 ms | 10.340 ms | 108 |

These timings use a dense teapot-field seed geometry to exercise neighborhood
construction. The 256-step headless teapot recovery run with six 768x768
screenshots completed in 9.91 seconds, including rendering and PNG capture.

## Viewer Contract

The default native viewer exposes `open image` and `open mesh` as separate
target types. Loading a mesh:

1. parses and normalizes the OBJ off the render schedule;
2. installs a 16,384-particle surface preview;
3. selects `Growing3dGs`, oriented Gaussian decoding, depth sorting, and the
   3D PanOrbit camera;
4. routes the shared train button to the 3D trainer;
5. applies bounded live model snapshots without blocking camera interaction;
6. reloads the final model and pristine embedded initialization.

The headless exporter accepts the same BPK and can optionally apply a localized
state erasure with `--mesh-damage-radius`.

## Experimental Neutral-Seed Track

The research goal remains a conditionless local 3D NPA that starts from a
compact neutral seed:

```text
neutral particle seed
  -> local SPH/hashgrid perception
  -> shared recurrent NPA rule
  -> position, opacity, color, and hidden-state updates
  -> multi-view Gaussian-splat image loss
```

Object identity must come from learned controller weights or a compact
adapter. It must not be encoded in particle indices, target residuals,
precolored state, absolute world-position features, or object-specific seed
geometry. Existing `train-render3d` and `train-render3d-adapters` smoke configs
exercise this plumbing but do not pass quality-scale promotion gates.

The strict promotion contract requires:

1. `position_features = false`;
2. target-independent neutral seeding;
3. multi-step local rollout supervision;
4. finite position, state, opacity, color, and covariance values;
5. broad target surface coverage and normalized active extent;
6. multi-view density, color, and depth quality;
7. particle-order permutation consistency;
8. primary and held-out seed stability;
9. multiple particle-count and rollout-horizon checks;
10. release-mode GPU residency and throughput evidence.

## Next Work

1. Port the official differentiable 3D Gaussian raster loss and gradient
   contract into the Burn trainer.
2. Establish one neutral-seed single-object overfit with 16,384 particles and
   multi-view image supervision.
3. Add bounded TBPTT and persistent trajectory pools for 16-32 step sampled
   training horizons plus 128-512 step validation.
4. Repeat strict overfit on several geometrically distinct meshes.
5. Train a shared local 3D trunk and quantify full-model, adapter, and
   shared-only gaps before adding a conditioned 3D hypernetwork.

Generated checkpoints and reports remain under ignored `artifacts/` or
`target/` paths. Only smoke and passing quality contracts belong under
`configs/verified/`.
