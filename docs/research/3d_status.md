# 3D NPA Status

The 3D path is experimental. No locally trained 3D model currently passes the
repository's catalog-quality morphogenesis gates, and no 3D artifact is
promoted as a baseline.

## Goal

The target is a conditionless local 3D NPA that starts from a compact neutral
seed and forms a target object through recurrent particle interaction:

```text
neutral particle seed
  -> local SPH/hashgrid perception
  -> shared recurrent NPA rule
  -> position, opacity, color, and hidden-state updates
  -> multi-view Gaussian-splat image loss
```

Object identity must come from the training target or a compact adapter. It
must not be encoded in particle indices, target residuals, precolored state,
absolute world-position features, or object-specific seed geometry.

## Maintained Commands

The direct 3D command has bounded smoke and quality contracts:

```bash
cargo run --release -p burn_automata -- \
  train-render3d \
  --config configs/verified/3d/oracle/torus_oracle_smoke.toml
```

The shared-base and per-target adapter command is:

```bash
cargo run --release -p burn_automata -- \
  train-render3d-adapters \
  --config configs/verified/3d/adapters/torus_smoke.toml
```

The larger `torus_oracle_quality.toml` and `many_slice_quality.toml` files are
reproducible experiment contracts. They are not passing quality evidence.

## Architecture Boundary

The intended scalable architecture is:

```text
shared local 3D NPA trunk
  + compact target adapter
  + target-independent neutral seed
```

`NpaLowRankAdapter` can materialize low-rank deltas over shared `NpaWeights`,
and the adapter trainer can emit a shared BPK, per-target adapter records, and
materialized validation models. This plumbing does not prove that the shared
trunk is a valid 3D morphogenesis substrate.

Target-specific seed aliases remain only for loading historical regression
artifacts. New experiments must use the generic local growth seed variants.

## Current Blocker

The local rollout does not yet achieve broad, material-visible surface
coverage from a neutral seed. Earlier candidates could reduce mean
point-to-surface distance while collapsing onto a small or inner region of the
target. Strict validation therefore measures coverage, rendered density,
color, depth, active extent, new activation, and multi-seed stability in
addition to mean surface error.

Until a direct 3D overfit passes these gates, shared-base adapter and HyperNPA
experiments cannot establish generalization. Their infrastructure remains
useful for smoke testing only.

## Promotion Contract

A 3D model is catalog eligible only when a reloaded `.bpk` passes:

1. `position_features = false`;
2. target-independent neutral seeding;
3. multi-step local rollout supervision;
4. finite position, state, opacity, color, and covariance values;
5. target surface coverage and normalized active extent;
6. multi-view density, color, and depth quality;
7. particle-order permutation consistency;
8. primary and held-out seed stability;
9. multiple particle-count and rollout-horizon checks;
10. release-mode runtime and GPU-residency gates.

The catalog validator is:

```bash
python3 scripts/validation/3d/validate_3d_catalog.py
```

It may use existing artifacts when present, but missing local checkpoints are
an environment boundary rather than passing evidence.

## Next Work

1. Establish one strict direct 3D overfit from a generic neutral seed.
2. Match the CPU and device multi-view loss and gradient on a bounded fixture.
3. Scale the direct trainer while preserving local SPH/hashgrid semantics.
4. Repeat the overfit on several geometrically distinct targets.
5. Train one shared trunk across those targets and measure the frozen-base gap.
6. Fit compact per-target adapters and compare them with full overfits.
7. Add a conditioned 3D adapter generator only after the shared substrate
   passes held-out adapter tests.

Generated 3D checkpoints, reports, and sweeps belong under ignored
`artifacts/`, `target/`, or `configs/sandbox/` paths until they pass this
promotion contract.
