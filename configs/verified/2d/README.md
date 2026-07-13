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

Checked-in HyperNPA configs include smoke, throughput, and a three-stage
growing-catalog adapter-reconstruction control. Run
`scripts/fetch_selforg_npa_targets.sh`, then execute
`quality_growing_catalog_pretrain.toml`,
`quality_growing_catalog_refine.toml`, and
`quality_growing_catalog_nonbase_eval.toml` in order. The final teacher-seen
non-base reconstruction passes 26 dB at 4,096 particles and 1,024 rollout
steps. It is not an identity-disjoint generalization result; strict holdout
quality is documented in `docs/hyper_npa.md`.

Canonical HyperNPA configs should set `[validation].interval` separately from
`[training].report_interval`, cap DINO condition cache residency with
`[gpu].condition_device_cache_max_bytes`, and spell out base/generator optimizer
settings even when they match the global defaults. They also include
`[gates]`: smoke gates should prove the command path works, bench gates should
bound throughput and validation overhead. Quality configs may add a PSNR gate
only when the validation scale and oracle comparison support the claim.

The reconstruction control uses native 224px DINO ViT-S tokens, the full
adapter-space conditional head, exact released-oracle adapter warm starts,
fixed generated adapters, and alpha-aware image/density metrics. The compact
module-aware decoder now passes zero-adapter, condition-shuffle, and
sample-parallel separation controls with canonical full-rank LoRA, but remains
experimental because identity-disjoint OmniSVG-1k quality is below the 26 dB
gate. Detached TBPTT and persistent particle pools remain supported
diagnostics. `dense` remains the conservative smoke default;
`tiled-adjoint` is the parity-gated device-side Target2D adjoint used by the
throughput benchmark and staged quality experiments.
`hyper_e2e/bench_omnisvg_8_b4_p128_tiled.toml` mirrors the dense bench shape
but exercises the real device-side tiled Target2D adjoint path.
`hyper_e2e/throughput_omnisvg_64_b64_p1024_s96_cuda.toml` is the validated
quality-scale CUDA throughput shape: 64 independent image-conditioned
rollouts, 1,024 particles, and a fixed 96-step full-BPTT horizon. Run it with
the release CUDA build; its gate covers synchronized optimizer throughput, not
the older asynchronous per-step estimate.
