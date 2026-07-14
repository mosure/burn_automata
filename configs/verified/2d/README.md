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

`smoke_conditional_row_flow.toml` is the bounded regression for the new
structured controller-flow path. It trains a timestep-conditioned velocity
field over dense NPA parameter rows from full DINOv2 tokens, uses device-side
Gaussian sources, and samples a deterministic controller with Heun integration.
`production_contract_growing_catalog_row_flow_pretrain.toml` records the Flow-S
production shape (12 layers, width 768, 12 heads, FFN width 3072, eight Heun
steps). Its endpoint phase keeps the paired shared base frozen; rollout
refinement must set `flow_matching_weight = 0` before changing that base,
because endpoint deltas are defined relative to the frozen checkpoint. The
filename deliberately does not claim quality: only the bounded smoke has run.

`smoke_conditional_row_flow_e2e.toml` is the canonical teacher-free command
gate. It jointly optimizes the shared trunk and a DINO-conditioned dense NPA
row-residual flow from Target2D rollout loss, with a small self-rectification
auxiliary that reuses the generated endpoint. It has no oracle directory or
per-sample adapter targets. Teacher-free configs initialize the deterministic
source residual at `1e-3` RMS so it does not overwhelm the pretrained trunk;
velocity learning remains row-normalized. The corresponding quality-scale contract is
`production_omnisvg_1k_conditional_row_flow_e2e_cuda.toml`: 16 independent
images per optimizer step, 32 trajectories per image, online native-224 DINO
tokens, four Heun flow steps, fixed 96-step full-BPTT rollouts, deterministic
global and per-identity fresh-seed injection, a frozen-trunk generator warmup,
timed checkpoints, long-horizon p10 validation, condition-shuffle and base-only
controls, and a final 4,096-particle evaluation. The C16 x R32 shape is the
measured 96 GiB CUDA contract; use a smaller effective batch on lower-memory
devices rather than raising `gpu_memory_budget_gb`. Endpoint-bank
pretraining is optional warm-starting, not a required stage of this path.

The row flow uses Burn/CubeCL tiled attention and fused modulated layer
normalization behind explicit analytic autodiff adjoints. Static source, row,
and timestep tensors remain device-resident. A 10-step C16 x R32,
512-particle, 96-step profile measured 13.25M particle-steps/s, 430 W and 96.3%
mean steady-state GPU power/utilization, 479 W peak power, and 78.3 GiB peak
VRAM on an RTX PRO 6000 Blackwell. A steady-state Nsight comparison against the
unfused layer norm raised device-active duty from 84.6% to 89.6%, reduced gaps
of at least 1 ms from 74 to 49, and reduced the worst gap from 13.3 ms to
3.36 ms. Active-graph preflight accounts for the rollout graph, flow graph,
trainable state, caches, pool, and runtime reserve; oversized shapes fail
before GPU condition/model allocation.

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
