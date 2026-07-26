# 3D Adapter Experiments

Run the bounded shared-base and adapter-bank command path with:

```bash
cargo run --release -p burn_automata -- \
  train-render3d-adapters \
  --config configs/verified/3d/adapters/torus_smoke.toml
```

`many_slice_quality.toml` is the retained multi-object research contract. It is
not a HyperNPA quality result and depends on first establishing a valid 3D
oracle/shared trunk.
