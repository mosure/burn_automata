# Render3D Experiments

These TOML recipes drive Burn-native 3D oracle overfit experiments through
`train-render3d --config`.

```bash
cargo run --release -p burn_automata --bin burn_automata -- \
  train-render3d \
  --config configs/render3d/torus_oracle_smoke.toml
```

The smoke recipe is intentionally small and is expected to exercise dispatch,
training, report writing, and validation without necessarily passing strict
quality gates. Use `torus_oracle_quality.toml` for a longer 3D parity/quality
run before using a model as a basis for 3D adapter or hypernet experiments.

3D adapter-suite experiments use the same TOML bundling style under
`configs/render3d_adapters/` via `train-render3d-adapters --config`.
