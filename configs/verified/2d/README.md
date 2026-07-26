# Verified 2D Configurations

The maintained 2D sequence is:

1. validate the released SelfOrg-NPA baseline with [`parity/`](parity/);
2. exercise image-conditioned training with [`hypernpa/`](hypernpa/);
3. evaluate conservative variable-resolution research with
   [`adaptive/`](adaptive/).

The fixed catalog and optimized WGPU inference kernels are the reference
runtime. Hidden `train-target2d`, direct-basis, and adapter-bank commands remain
diagnostics until the strict parity harness matches upstream target extraction,
initialization, perception, loss, gradients, optimizer updates, and a
4,096-particle rollout.

Canonical bounded commands:

```bash
cargo run -p burn_automata -- \
  validate-npa2d-parity \
  --config configs/verified/2d/parity/lizard_smoke.toml

cargo run --release -p burn_automata \
  --no-default-features \
  --features cli,backend_ndarray,backend_wgpu,dino -- \
  train-hypernpa2d \
  --config configs/verified/2d/hypernpa/flow/smoke_conditional_row_flow.toml
```

Each child directory documents which recipes are smoke tests, measured
controls, throughput contracts, or unproven quality-scale experiments.
