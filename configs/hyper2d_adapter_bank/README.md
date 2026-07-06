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
cargo run --release -p burn_automata --bin burn_automata -- \
  train-hyper2d-adapter-bank \
  --config configs/hyper2d_adapter_bank/smoke_from_10k.toml
```

The output `hyper_2d.json` is a normal `HyperNpa2d` checkpoint consumable by
`infer-hyper2d`.

Summarize quality gates with the Rust reporting command. It writes JSON, Markdown, and LaTeX
reports:

```bash
cargo run -p burn_automata --features cli -- report-hyper2d \
  --report artifacts/hyper2d_adapter_bank/report.json \
  --output-dir artifacts/hyper2d_adapter_bank/report_summary
```

Add `--require-quality-ready` to fail the command unless generated LoRAs pass the
adapter-vector and rollout gates. Paper-quality reports also require at least
2048 rollout particles and 2048 target samples; lower-count pilot reports are
blocked by the quality-scale gate.

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
| `omnisvg_3_dino_full_tokens_flow_smoke.toml` | Tiny structural smoke that stores CLS plus all 37x37 ViT-S patch tokens for each image. |
| `omnisvg_10k_dino_memory_smoke.toml` | Bounded 10k cached-DINO WGPU smoke. It runs two steps, disables rollout, and should stay well below the 16 GiB RSS budget. |
| `omnisvg_10k_dino_canonical_h1024_valselect.toml` | 10k DINO/canonical-adapter validation-selected run with chunked loss eval and a system-memory budget. |
| `omnisvg_10k_dino_patch_stats_h1024_valselect.toml` | 10k DINO patch-stat condition run with the same memory-safe WGPU trainer. DINO extraction uses batch 1 to avoid current WGPU DINO buffer allocation failures. |
| `omnisvg_10k_from_direct_basis.toml` | 10k summary-token scale-up against the current direct-basis adapter bank. |
| `omnisvg_10k_quality_2048_from_direct_basis.toml` | 10k quality-scale condition-to-LoRA run against a 2048-particle direct-basis bank. |

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

## Next Model Step

Use the same adapter-bank reports to compare the static-regression baseline
against the conditioned rectified-flow generator. The gate remains the same:
generated LoRAs must close both adapter-vector metrics and rollout loss ratios
versus direct stored LoRAs before claiming HyperNPA generalization. Generated
LoRAs must also beat the zero-adapter rollout baseline; beating a malformed
stored-LoRA target is not sufficient.
