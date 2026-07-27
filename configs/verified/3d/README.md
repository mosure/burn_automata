# Verified 3D Configurations

## Mesh-Conditioned

`mesh/teapot_smoke.toml` is the bounded command and artifact-persistence gate.
`mesh/teapot_quality.toml` is the promoted Utah teapot surface-support
contract. It passes 16,384-particle multi-seed geometry, density, color,
depth, localized state-recovery, and long-horizon stability checks.

```bash
cargo run --release -p burn_automata --features gpu_wgpu -- \
  train-mesh3d \
  --config configs/verified/3d/mesh/teapot_quality.toml
```

This path is position-conditioned and embeds normalized mesh surface support in
the BPK. It must be described as mesh-conditioned state-field training, not
neutral-seed morphogenesis.

## Neutral-Seed Research

- `oracle/` contains direct torus smoke and research-quality recipes for
  `train-render3d`.
- `adapters/` contains shared-base and per-target adapter recipes for
  `train-render3d-adapters`.

The smoke recipes prove command dispatch and artifact/report persistence.
Their larger quality recipes remain research contracts and are not promoted
quality evidence. The strict boundary and next work are documented in
`docs/research/3d_status.md`.
