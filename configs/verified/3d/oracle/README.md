# 3D Oracle Experiments

Run the bounded command path with:

```bash
cargo run --release -p burn_automata -- \
  train-render3d \
  --config configs/verified/3d/oracle/torus_oracle_smoke.toml
```

`torus_oracle_quality.toml` is the retained larger-scale contract. Neither file
claims strict multi-seed catalog parity.
