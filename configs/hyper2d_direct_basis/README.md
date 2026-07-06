# Hyper2D Direct-Basis Experiments

Run a bundled experiment with:

```sh
scripts/safe_train_hyper2d_direct_basis.sh --config configs/hyper2d_direct_basis/omnisvg_1k.toml
```

The TOML file is the experiment recipe. Values supplied in TOML take precedence over the flat CLI flags for the same setting, while omitted values keep the existing CLI defaults.

The OmniSVG recipes default to `download = false` so they use the local cache. Set `[source.omnisvg].download = true` when the cache needs to be populated or refreshed.

These configs intentionally use `[gpu].backend = "burn-wgpu"`. The older
upstream Python/CUDA scripts remain useful for parity checks and historical
comparison, but new 2D direct-basis experiments should start from these
Burn/WGPU TOML recipes.

The Burn/WGPU direct-basis trainer is now guarded by a conservative memory
preflight. Dense all-pairs autodiff training is capped at
`max_dense_train_particles = 1024` until a fused/tiled backward backend exists.
Training recipes also set `gpu_memory_budget_gb`; the preflight estimates the
live WGPU autodiff graph separately from cached target tensors and rejects
unsafe phase batches before allocation. The observed 512-particle batch-4
dense path exhausted a 98 GB GPU, and a full 1k 512-particle batch-1 refine run
crossed the 64 GiB experiment budget, so the staged 384 recipe is the current
highest full-training curriculum target until the dense backward path is fused
or allocator lifetime is fixed.

Training batches use a shuffled epoch sampler rather than independent random
draws. Reports include per-phase adapter update coverage so 1k/10k quality
runs can distinguish optimizer/objective failure from undersampled adapters.
Adapter-only phases evaluate the phase-initial state for checkpoint selection;
if refine regresses, the incoming base/adapters are preserved instead of being
overwritten by the least-bad refine checkpoint.
Recipes with `2048` rollout particles are validation/evaluation recipes only:
their direct-basis training phases are set to zero, and nonzero 2048 training is
rejected before WGPU allocation.

Use `scripts/safe_train_hyper2d_direct_basis.sh` for local runs. It wraps the
trainer in `timeout` and a systemd memory scope (`BURN_AUTOMATA_MEMORY_MAX`,
default `32G`; `BURN_AUTOMATA_TIMEOUT`, default `6h`) so a bad experiment fails
boundedly instead of exhausting system RAM.

Summarize a direct-basis report with the Rust reporting command. It writes
`validation_summary.json`, `validation_report.md`, and `validation_report.tex`.

```sh
cargo run -p burn_automata --features cli -- report-hyper2d \
  --report artifacts/hyper2d_direct_basis/report.json \
  --oracle-report artifacts/hyper2d_direct_basis_oracles/report.json \
  --output-dir artifacts/hyper2d_direct_basis/report_summary
```

Add `--require-quality-ready` when the run should fail unless sampled oracle
ratios pass the direct-basis gate. Paper-quality reports also require at least
2048 rollout particles and 2048 target samples; lower-count pilot runs are
reported as not quality-ready even if their oracle ratio is close. Fresh oracle
validation also records a zero-adapter baseline; stored LoRAs must beat that
baseline on train and holdout before direct-basis readiness can pass.

Oracle validation is also TOML-configurable:

```sh
cargo run --release -p burn_automata --features cli --bin burn_automata -- \
  validate-hyper2d-direct-basis-oracles \
  --config configs/hyper2d_direct_basis/oracle_validate_10k_quality_2048.toml
```

For local safety, prefer the bounded wrapper:

```sh
scripts/safe_validate_hyper2d_direct_basis_oracles.sh \
  --config configs/hyper2d_direct_basis/oracle_validate_10k_quality_2048_smoke.toml
```

Latest diagnostic: `omnisvg_1k_p64_refine_lr1e4.toml` completed with full train
adapter coverage (`min_updates=4`, zero missing) and refine coverage
(`min_updates=6`, zero missing). The best checkpoint remained the base-training
step 300 loss (`5.8024`) because lower-LR adapter-only refine still regressed.
Low-particle oracle ratios on 2 train / 2 holdout samples were close
(`1.05x` train, `1.07x` holdout), but this is not quality-ready because it uses
64 rollout particles, 256 target points, and too few oracle samples.

See `docs/burn_first_cleanup.md` for the current Burn-first cleanup boundary and
the gates required before removing the legacy upstream-Python training backend.

## Recipes

| Config | Purpose |
| --- | --- |
| `smoke_particles64_tbptt.toml` | Tiny memory-capped Burn/WGPU TBPTT training smoke for launcher/preflight/regression checks. |
| `smoke_particles512_tbptt.toml` | Tiny 512-particle, batch-1 Burn/WGPU smoke for high-VRAM preflight/TBPTT coverage; requires a 96 GiB GPU budget. |
| `smoke_quality_2048.toml` | Bounded 2048-particle/2048-target-sample eval smoke with direct training disabled; not a quality result. |
| `smoke_rank132_particles512_quality_train.toml` | Bounded rank-132 staged training smoke at 512 particles, TBPTT-1, and 2048 target samples; validates the current dense path without crossing the VRAM guard. |
| `omnisvg_1k.toml` | 1k Burn/WGPU low-particle development run with sampled report-time eval and explicit memory caps; not paper-quality. |
| `omnisvg_1k_p64_refine_lr1e4.toml` | 1k low-particle diagnostic with epoch sampling and lower adapter-refine LR to test optimizer stability. |
| `omnisvg_1k_staged_particles384_tbptt.toml` | 1k staged mid-particle run with lightweight report-time eval before separate 2048 validation. |
| `omnisvg_1k_staged_particles512_tbptt.toml` | Guarded 512-particle staging recipe; expected to fail preflight under the default 64 GiB experiment budget. |
| `omnisvg_1k_quality_2048.toml` | 1k quality-scale validation recipe with 2048 rollout particles and 2048 target samples; direct training disabled on the dense path. |
| `omnisvg_10k_pilot.toml` | 10k rank-16, 64-particle pilot; not paper-quality. |
| `omnisvg_10k_quality_2048.toml` | 10k quality-scale validation recipe with 2048 rollout particles and 2048 target samples; direct training disabled on the dense path. |
| `omnisvg_10k_rank32_particles128.toml` | Higher-capacity rank-32, 128-particle/12-step pilot for expressivity and throughput comparison; not paper-quality. |
| `omnisvg_10k_rank64_particles128.toml` | Rank-64 version of the 128-particle expressivity pilot. |
| `omnisvg_10k_rank132_particles128.toml` | Rank-132 expressivity pilot matching exact-oracle adapter capacity; batch size is lower because adapter state is much larger. |
| `oracle_validate_10k_quality_2048.toml` | Quality-scale oracle plus zero-adapter validation for the current 10k rank-16 bank; run before training more conditioned HyperNPA models against that bank. |
| `oracle_validate_10k_quality_2048_smoke.toml` | 1-train/1-holdout quality-scale smoke for the current 10k rank-16 bank; useful for fast zero-baseline checks before the full 8/8 run. |
| `oracle_validate_smoke_rank132_particles512_quality_train_2048.toml` | 1-train/1-holdout 2048-particle oracle smoke for the staged rank-132/512-particle bank. |
| `oracle_validate_exact_oracle_10k8x8_2048_rank132_smoke.toml` | 1-train/1-holdout oracle smoke for the exact rank-132 oracle-delta adapter bank. |
