# Verified 2D Configs

The maintained 2D path is:

1. Validate the official SelfOrg-NPA lizard baseline with
   `validate-npa2d-parity`.
2. Run canonical WGPU/CUDA inference and rollout diagnostics against imported
   `.bpk` models.
3. Train online HyperNPA with `train-hyper2d-e2e-rollout`
   (`train-hypernpa2d`) using the configs in `hyper_e2e/`.

`train-target2d`, direct-basis adapter banks, static LoRA reconstruction, and
direct-basis PSNR gates are hidden legacy diagnostics. They are not validated
replacements for the upstream CUDA trainer until the parity gate passes with
matching target extraction, initialization, rollout, loss, gradients, and
optimizer semantics.

Checked-in HyperNPA configs are either smoke/bench configs or explicitly marked
scale diagnostics. Do not label a config production-quality until its report
passes high-particle validation against the upstream/oracle baseline.

Canonical HyperNPA configs should set `[validation].interval` separately from
`[training].report_interval`, cap DINO condition cache residency with
`[gpu].condition_device_cache_max_bytes`, and spell out base/generator optimizer
settings even when they match the global defaults. They should also include
`[gates]`: smoke gates should prove the command path works, bench gates should
bound throughput and validation overhead, and quality configs should add a PSNR
gate only when the validation scale is sufficient for the claim. Scale configs
should train on an oracle-shaped rollout curriculum (`rollout.step_min` plus
`rollout.steps`). The current promoted 2D HyperNPA shape is stochastic 32..64
step training, endpoint-weighted TBPTT
(`training.tbptt_loss_mode = "endpoint-weighted"`), sample-keyed particle
pools (`training.use_particle_pool = true`), `training.target2d_loss_backend =
"tiled-adjoint"`, and `training.perception_backend = "auto"` so training-scale
128+ particle runs use the optimized device-side perception path. `dense`
remains the conservative smoke default, while `tiled-adjoint` is the
parity-gated device-side Target2D adjoint path for throughput benchmarking and
staged promotion.
`hyper_e2e/bench_omnisvg_8_b4_p128_tiled.toml` mirrors the dense bench shape
but exercises the real device-side tiled Target2D adjoint path.
