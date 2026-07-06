# Render3D Adapter-Suite Experiments

These TOML recipes drive Burn-native 3D shared-base plus per-target low-rank
adapter experiments through `train-render3d-adapters --config`.

```bash
cargo run --release -p burn_automata --bin burn_automata -- \
  train-render3d-adapters \
  --config configs/render3d_adapters/torus_smoke.toml
```

The smoke recipe is intentionally small and validates dispatch, shared-base
materialization, adapter-bank persistence, and report writing. It is not a
quality gate.

Use `many_slice_quality.toml` after a Burn-native 3D oracle has been validated;
it trains a shared base over a small multi-object slice, holds out two targets,
and writes the adapter bank needed for later conditioned 3D adapter generation.
