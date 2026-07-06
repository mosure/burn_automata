# Hyper2D Adapter-Bank Conditioning

These configs train the image-condition -> LoRA adapter stage from an existing
direct-basis shared base and adapter bank.

The Burn/WGPU trainer supports two adapter-bank objectives:

- `objective = "static-vector-mse"`: the original supervised static-LoRA vector
  regression baseline.
- `objective = "rectified-flow"`: a conditioned adapter-flow velocity field over
  `[condition features, timestep, noisy adapter state]`.

The rectified-flow path is now implemented, but it should still be treated as an
experiment until quality-scale rollout reports close the gap to direct stored
LoRAs and 2D overfit oracles.

The default backend is `burn-wgpu`. The command fails explicitly if the binary is
not built with `backend_wgpu`; use `backend = "cpu"` only for small correctness
checks.

```bash
scripts/safe_train_hyper2d_adapter_bank.sh \
  --config configs/hyper2d_adapter_bank/smoke_from_10k.toml
```

The output `hyper_2d.json` is a normal `HyperNpa2d` checkpoint consumable by
`infer-hyper2d`.

Summarize quality gates with the Rust reporting command. It writes JSON, Markdown, and LaTeX
reports:

```bash
cargo run -p burn_automata --features cli -- report-hyper2d \
  --report artifacts/hyper2d_adapter_bank/report.json \
  --psnr-report artifacts/hyper2d_adapter_bank/psnr_gate_report.json \
  --output-dir artifacts/hyper2d_adapter_bank/report_summary
```

Add `--require-quality-ready` to fail the command unless generated LoRAs pass the
adapter-vector and rollout gates. Paper-quality reports also require at least
2048 rollout particles and 2048 target samples; lower-count pilot reports are
blocked by the quality-scale gate. HyperNPA readiness also requires a PSNR gate
report from `validate-hyper2d-psnr-gate`; otherwise the reporter returns
`needs_psnr_oracle_validation` even when the adapter-vector and rollout summaries
are close.

The PSNR gate is TOML-configurable:

```bash
cargo run --release -p burn_automata --features cli --bin burn_automata -- \
  validate-hyper2d-psnr-gate \
  --config configs/hyper2d_adapter_bank/psnr_gate_1k_dino_token_grid_flow_h512_rms_noise.toml
```

Exact oracle-adapter banks are also TOML-configurable. Build them from a
persisted direct-basis oracle report before using them as clean condition-model
targets:

```bash
cargo run --release -p burn_automata --features cli --bin burn_automata -- \
  build-exact-adapter-bank \
  --config configs/hyper2d_adapter_bank/build_exact_oracle_bank_10k8x8_2048_rank132_bias_exact.toml
```

The summary distinguishes close rollout ratios from true adapter-vector
prediction quality, so weak condition-to-LoRA generalization is visible even when
short rollout losses are nearly unchanged.
Adapter-bank rollout reports also include a zero-adapter baseline. Treat
HyperNPA as unproven unless generated adapters beat zero/no-adapter rollouts and
direct stored LoRAs at quality scale.

## Recipes

| Config | Purpose |
| --- | --- |
| `smoke_from_10k.toml` | Fast correctness check against the current 10k direct-basis artifact. |
| `omnisvg_1k_from_direct_basis.toml` | Baseline 1k summary-token condition-to-LoRA run. |
| `omnisvg_1k_quality_2048_from_direct_basis.toml` | 1k quality-scale condition-to-LoRA run against a 2048-particle direct-basis bank. |
| `omnisvg_1k_aggressive_from_direct_basis.toml` | Higher learning-rate/output-scale ablation for the 1k baseline. |
| `omnisvg_1k_dino_from_direct_basis.toml` | DINO feature experiment; run `~/.venvs/torch/bin/python scripts/setup_dino_vits.py` first so `models/dino/dino_vits.mpk` exists. |
| `omnisvg_1k_dino_canonical_h1024_valselect.toml` | 1k DINO/canonical-adapter validation-selected run with chunked loss eval and a system-memory budget. |
| `omnisvg_1k_dino_patch_stats_smoke.toml` | Cached-feature two-step WGPU smoke for the patch-stat path. It reuses the 1k patch-stat cache and skips rollout evaluation. |
| `omnisvg_1k_dino_patch_stats_h1024_valselect.toml` | 1k DINO patch-stat condition run; uses CLS plus patch mean/std/min/max to test whether global feature compression is the blocker. DINO extraction uses batch 1 to avoid current WGPU DINO buffer allocation failures. |
| `omnisvg_20_dino_token_grid_flow_smoke.toml` | Small WGPU smoke for the DINO token-grid rectified-flow objective. |
| `omnisvg_1k_dino_token_grid_flow_h512.toml` | 1k DINO 8x8 token-grid rectified-flow experiment with 2048-particle rollout validation. |
| `omnisvg_1k_dino_token_grid_flow_h512_rms_noise.toml` | Same 1k flow experiment in a separate output directory after the flow source-noise default was corrected to adapter-RMS scale. |
| `omnisvg_1k_dino_token_grid_flow_h512_sampled_refine.toml` | 1k sampled-adapter refinement initialized from the RMS-noise flow checkpoint. Uses zero-source sampled-adapter loss to align training with the inference sampler. |
| `omnisvg_3_dino_full_tokens_flow_smoke.toml` | Tiny structural smoke that stores CLS plus all 37x37 ViT-S patch tokens for each image. |
| `build_exact_oracle_bank_10k8x8_2048_rank132_bias_exact_train_all.toml` | Build a train-only exact oracle bank for clean overfit checks before generalization claims. |
| `exact_oracle_10k8x8_dino_token_grid_linear_solve_overfit_train_all.toml` | DINO 8x8 token-grid linear-solve control proving the condition features can exactly memorize the clean train-only exact rank-132 bank. |
| `exact_oracle_10k8x8_dino_token_grid_flow_overfit_train_all.toml` | DINO 8x8 token-grid rectified-flow overfit check against the clean train-only exact rank-132 bank. |
| `exact_oracle_10k8x8_dino_token_grid_flow_zero_source_overfit_train_all.toml` | Stage-A WGPU zero-source rectified-flow overfit check. Uses leaky-ReLU hidden flow activations so random init does not collapse to dead ReLU gates. |
| `exact_oracle_10k8x8_dino_token_grid_flow_zero_source_h384_lr2e3_train_all.toml` | Higher-capacity random-init zero-source WGPU flow run. It fits adapter vectors much better than the first random run, but still misses the 2048-particle PSNR floor. |
| `exact_oracle_10k8x8_dino_token_grid_flow_zero_source_h384_lr2e4_refine_train_all.toml` | `flow_init = "from-hyper"` low-LR velocity-MSE refinement from the h384 checkpoint. Useful for proving vector-only refinement plateaus before quality parity. |
| `exact_oracle_10k8x8_dino_token_grid_flow_zero_source_h384_lr2e5_refine2_train_all.toml` | Second lower-LR velocity-MSE refinement. It only marginally improves vector error and does not close the PSNR gate. |
| `exact_oracle_10k8x8_dino_token_grid_flow_zero_source_h384_sampled_refine_train_all.toml` | Inference-aligned sampled-adapter refinement. Backpropagates through the 16-step flow sampler and is the current best WGPU random-init path. |
| `exact_oracle_10k8x8_dino_token_grid_flow_zero_source_h384_sampled_refine2_train_all.toml` | Lower-LR continuation of sampled-adapter refinement. It did not beat the first sampled-refine PSNR gate. |
| `exact_oracle_10k8x8_dino_token_grid_flow_zero_source_h384_sampled_weighted_refine_train_all.toml` | Hard-row sampled-adapter continuation. It uses the previous 2048-particle PSNR gate report to upweight generated-adapter rows below 26 dB. |
| `exact_oracle_10k8x8_dino_token_grid_flow_zero_source_h384_sampled_weighted_margin_refine_train_all.toml` | Guard-band hard-row sampled-adapter continuation. It upweights rows below 26.5 dB from the latest PSNR gate so near-threshold rows do not regress below the 26 dB floor. |
| `exact_oracle_10k8x8_dino_token_grid_flow_zero_source_h384_sampled_weighted_floor_refine_train_all.toml` | Final targeted floor refinement. It upweights only rows still below 26 dB from the guard-band gate and is the first random-init WGPU path to pass the 16-row exact-oracle PSNR gate. |
| `exact_oracle_10k8x8_dino_token_grid_flow_zero_source_warmstart_train_all.toml` | WGPU diagnostic run initialized from the host linear-solve condition interpolator, with no optimizer steps, to isolate tensor/layout/inference issues from optimizer failure. |
| `exact_oracle_10k8x8_dino_token_grid_flow_linear_solve_overfit_train_all.toml` | Deterministic DINO 8x8 token-grid rectified-flow overfit control using the host linear solver to prove the flow sampling path can materialize clean exact adapters. |
| `exact_oracle_10k8x8_dino_token_grid_flow_near_zero_source_overfit_train_all.toml` | DINO 8x8 token-grid rectified-flow overfit check with near-zero source noise to isolate denoising difficulty from condition-to-adapter fitting. |
| `exact_oracle_10k8x8_dino_token_grid_flow_split_smoke.toml` | DINO 8x8 token-grid rectified-flow split smoke against the clean 8-train/8-holdout exact rank-132 bank. |
| `omnisvg_10k_dino_memory_smoke.toml` | Bounded 10k cached-DINO WGPU smoke. It runs two steps, disables rollout, and should stay well below the 16 GiB RSS budget. |
| `omnisvg_10k_dino_canonical_h1024_valselect.toml` | 10k DINO/canonical-adapter validation-selected run with chunked loss eval and a system-memory budget. |
| `omnisvg_10k_dino_patch_stats_h1024_valselect.toml` | 10k DINO patch-stat condition run with the same memory-safe WGPU trainer. DINO extraction uses batch 1 to avoid current WGPU DINO buffer allocation failures. |
| `omnisvg_10k_from_direct_basis.toml` | 10k summary-token scale-up against the current direct-basis adapter bank. |
| `omnisvg_10k_quality_2048_from_direct_basis.toml` | 10k quality-scale condition-to-LoRA run against a 2048-particle direct-basis bank. |
| `build_exact_oracle_bank_10k8x8_2048_rank132_bias_exact.toml` | Build the clean 8-train/8-holdout rank-132 exact adapter bank from persisted oracle NPA models. |
| `psnr_gate_exact_oracle_10k8x8_2048_rank132_direct.toml` | Direct-only PSNR gate proving exact adapters materialize the persisted oracle NPA models at 2048-particle quality scale. |
| `psnr_gate_exact_oracle_10k8x8_2048_rank132_dino_flow_overfit.toml` | PSNR gate comparing DINO-flow generated adapters and direct exact adapters against persisted oracle NPA models. |
| `psnr_gate_exact_oracle_10k8x8_2048_rank132_dino_flow_linear_solve_overfit.toml` | PSNR gate for the deterministic DINO-flow linear-solve overfit control. |
| `psnr_gate_exact_oracle_10k8x8_2048_rank132_dino_flow_zero_source_overfit.toml` | PSNR gate for the WGPU random-init zero-source overfit checkpoint. |
| `psnr_gate_exact_oracle_10k8x8_2048_rank132_dino_flow_zero_source_h384_sampled_refine.toml` | PSNR gate for the current best sampled-adapter WGPU refinement checkpoint. |
| `psnr_gate_exact_oracle_10k8x8_2048_rank132_dino_flow_zero_source_h384_sampled_weighted_refine.toml` | PSNR gate for the hard-row weighted sampled-adapter refinement checkpoint. |
| `psnr_gate_exact_oracle_10k8x8_2048_rank132_dino_flow_zero_source_h384_sampled_weighted_margin_refine.toml` | PSNR gate for the guard-band weighted sampled-adapter refinement checkpoint. |
| `psnr_gate_exact_oracle_10k8x8_2048_rank132_dino_flow_zero_source_h384_sampled_weighted_floor_refine.toml` | PSNR gate for the final targeted floor refinement checkpoint. |
| `psnr_gate_exact_oracle_10k8x8_2048_rank132_dino_flow_zero_source_warmstart.toml` | PSNR gate for the WGPU zero-source warm-start diagnostic checkpoint. |
| `psnr_gate_1k_dino_token_grid_flow_h512_rms_noise.toml` | Quality-scale render-PSNR gate comparing direct stored LoRAs and generated HyperNPA LoRAs against oracle 2D rollouts. |
| `psnr_gate_1k_dino_token_grid_flow_h512_rms_noise_oracle8x8.toml` | Oracle-backed 16-row PSNR gate for the existing 1k RMS-noise flow checkpoint. Selection is constrained to rows with persisted oracle models. |
| `psnr_gate_1k_dino_token_grid_flow_h512_sampled_refine_oracle8x8.toml` | Oracle-backed 16-row PSNR gate for the 1k sampled-adapter refinement checkpoint. |
| `psnr_gate_10k_dino_canonical_h1024_valselect_oracle8x8.toml` | Oracle-backed 16-row PSNR gate for the existing 10k canonical-DINO checkpoint. |

Latest exact-oracle DINO-flow status:

- Direct exact adapters materialize the persisted oracle NPA models exactly in
  the 2048-particle PSNR gate.
- The deterministic zero-source DINO-flow linear-solve control passes the 26 dB
  2048-particle gate on the 16-entry train-only overfit slice
  (`mean=30.44 dB`, `min=27.31 dB`).
- The WGPU zero-source warm-start diagnostic reproduces that control at
  generated-vector precision (`nRMSE=1.18e-7`, cosine `1.0`) and passes the same
  2048-particle PSNR gate (`mean=30.44 dB`, `min=27.31 dB`).
- Random-init WGPU zero-source flow no longer collapses when configured with
  `flow_hidden_activation = "leaky-relu"`, but velocity-MSE alone still misses
  the PSNR floor. The h384 velocity-MSE run reached generated-vector
  `nRMSE=5.75e-4` and PSNR `mean=27.11 dB`, `min=23.78 dB`; two low-LR
  velocity refinements improved vector `nRMSE` to `1.27e-4`, but the PSNR floor
  remained below threshold.
- The sampled-adapter WGPU objective is the current best random-init path. It
  optimizes the final adapter produced by the same 16-step flow sampler used at
  inference, reached generated-vector `nRMSE=1.67e-5`, and improved the
  2048-particle PSNR gate to `mean=28.29 dB`, `min=24.81 dB`, with 2/16 rows
  below 26 dB. A lower-LR continuation plateaued and did not improve the gate
  (`mean=28.27 dB`, `min=24.88 dB`, 4/16 below threshold).
- Hard-row sampled refinement is configured to use that PSNR gate as supervision
  metadata. It upweights rows below 26 dB in the sampled-adapter loss while
  keeping the same 16-step inference sampler and zero-source flow path.
- Guard-band weighted refinement repeats that idea from the latest weighted
  checkpoint, but upweights all rows below 26.5 dB to reduce threshold
  whack-a-mole in the final PSNR floor.
- Targeted floor refinement from the guard-band checkpoint passes the 16-entry
  train-only exact-oracle 2048-particle PSNR gate (`mean=28.46 dB`,
  `min=26.04 dB`, 0/16 below 26 dB) with generated-vector
  `nRMSE=1.49e-5`. This is still a train-only exact-oracle overfit result, not
  broad 1k/10k HyperNPA generalization.
- Generalized 1k/10k PSNR validation must use oracle-backed selection. The
  persisted 10k `oracle8x8` report contains 16 quality oracle models spread
  across the 10k bank; only one of those rows is inside the old 1k train slice,
  so 1k validation is mostly out-of-slice generalization while 10k validation
  covers all 16 rows in-slice.
- Oracle-backed generalized validation is currently not close to parity. On the
  16 persisted `oracle8x8` rows at 2048 particles:
  - the existing 1k RMS-noise DINO token-grid flow checkpoint reaches
    `mean=19.58 dB`, `min=18.39 dB`, 16/16 below 26 dB;
  - the 1k sampled-adapter refinement reaches `mean=19.75 dB`,
    `min=18.84 dB`, 16/16 below 26 dB;
  - the existing 10k canonical-DINO checkpoint reaches `mean=19.41 dB`,
    `min=18.24 dB`, 16/16 below 26 dB.
- The broad 10k adapter-bank targets are themselves below oracle quality on the
  same rows: direct stored LoRAs reach only `mean=15.42 dB`, `min=10.69 dB`.
  This means broad HyperNPA quality cannot be proven by training to the current
  10k adapter bank; the next required dataset step is expanding the exact/oracle
  adapter bank with high-quality per-sample NPA oracles, then training the
  conditioned flow against that target distribution.
- Broad DINO/flow HyperNPA quality is therefore still not established. The
  architecture and sampling path are proven by linear-solve/warm-start controls,
  and WGPU optimization now moves the real PSNR gate, but the remaining gap is a
  rollout-sensitive quality objective problem, not DINO feature availability.

The DINO condition path is compiled behind the Rust `dino` feature. Run DINO
experiments with:

```sh
cargo run --release -p burn_automata --features dino --bin burn_automata -- \
  train-hyper2d-adapter-bank --config configs/hyper2d_adapter_bank/omnisvg_1k_dino_from_direct_basis.toml
```
The DINO recipes persist `condition.feature_cache` and set
`condition.dino_batch_size`; with WGPU enabled, cached feature extraction runs on
the GPU and interrupted runs can resume from the partially written cache. Large
runs should set `training.loss_eval_batch_size` and
`training.system_memory_budget_gb` so report-time validation is chunked and the
process fails before system RAM pressure reaches the OOM killer.
Use `condition.encoder = "dino-vits-patch-stats"` to cache the wider 1920-value
DINO descriptor; cache files are encoder-tagged and rejected if reused with the
wrong feature width.
Use `condition.encoder = "dino-vits-token-grid"` to cache CLS plus a spatial
patch-token grid. `token_grid_width = 8` and `token_grid_height = 8` pool the
DINO patch map into a bounded descriptor; `37x37` at `dino_image_size = 518`
stores the full ViT-S/14 patch-token grid. Full-token JSON caches are large, so
use them for tiny structural smokes until the cache format and condition model
are upgraded for broad-scale full-token training.
Rectified-flow source noise now defaults to adapter target RMS when
`training.flow_source_scale` is omitted. The earlier max-range default produced
oversized random adapters and invalid quality conclusions.
For bottleneck isolation, prefer explicit TOML values: `flow_source_scale = 0.0`
for deterministic Stage-A overfit, then small source-scale sweeps, then the
adapter-RMS value used by the noisy objective.

WGPU rectified-flow reports now include generated-adapter vector diagnostics at
report intervals. Check `training.vector_selection`, each history row's
`train_vector_metrics`/`validation_vector_metrics`, and `flow_optimizer`
activation/gradient diagnostics before trusting velocity MSE. The WGPU flow
trainer selects the best checkpoint by generated adapter-vector MSE when
`training.diagnostic_vector_examples > 0`.
Random-init flow diagnostics should use `flow_hidden_activation = "leaky-relu"`
so negative initial pre-activations still carry gradient. Keep the default ReLU
for linear-solve and warm-start controls, which rely on exact ReLU exemplar
gates.
Use `training.flow_init = "from-hyper"` with `input.initial_hyper` for explicit
checkpoint refinement. Use `training.flow_loss = "sampled-adapter-mse"` when
the velocity objective has a low vector error but still misses rollout/render
PSNR; this loss backpropagates through the configured flow sampler and directly
optimizes the generated adapter consumed by inference. It currently requires
`flow_source_scale = 0.0`.

Recommended bottleneck sequence:

1. Run `exact_oracle_10k8x8_dino_token_grid_flow_zero_source_warmstart_train_all.toml`.
   This should match the host linear-solve control and proves WGPU checkpoint
   serialization plus inference sampling are wired correctly.
2. Run `exact_oracle_10k8x8_dino_token_grid_flow_zero_source_overfit_train_all.toml`.
   This tests whether random-init WGPU optimization can learn the deterministic
   condition-to-adapter map.
3. Only after zero-source WGPU passes, run the near-zero source and RMS-noise
   configs, then the 1k/10k generalized configs.

## Next Model Step

Use the same adapter-bank reports to compare the static-regression baseline
against the conditioned rectified-flow generator. The gate remains the same:
generated LoRAs must close both adapter-vector metrics and rollout loss ratios
versus direct stored LoRAs before claiming HyperNPA generalization. Generated
LoRAs must also beat the zero-adapter rollout baseline; beating a malformed
stored-LoRA target is not sufficient. The final readiness check is the
`validate-hyper2d-psnr-gate` render comparison against 2D oracle trajectories at
quality particle counts.
