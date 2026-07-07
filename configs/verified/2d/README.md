# Verified 2D Configs

The maintained 2D path is:

1. Validate the official SelfOrg-NPA lizard baseline with
   `validate-npa2d-parity`.
2. Run canonical WGPU/CUDA inference and rollout diagnostics against imported
   `.bpk` models.
3. Train online HyperNPA with `train-hyper2d-e2e-rollout` using the configs in
   `hyper_e2e/`.

`train-target2d` is kept as an experimental Burn/Rust oracle diagnostic. It is
not a validated replacement for the upstream CUDA trainer until the parity gate
passes with matching target extraction, initialization, rollout, loss, and
optimizer semantics.
