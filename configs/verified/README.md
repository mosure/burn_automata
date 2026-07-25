# Verified Configs

Only configs that describe a maintained, validated path belong under
`configs/verified/`. Exploratory sweeps, failed experiments, large local runs,
and one-off diagnostics belong under `configs/sandbox/`, which is intentionally
gitignored.

Current 2D status:

The regular NPA path remains the production path. Adaptive execution now has a
verified resident 3,070-active/4,096-material structural smoke, exact
Burn/WGPU rollout parity under matched material identities, and a bounded
direct-active continuous-scale lizard smoke. The corresponding 32-seed
candidate spans 33 material-scale bins and a measured `2.001x` isotropic
Gaussian-radius range, but it misses the 1,024-step quality/drift gate and is
not promoted. The former `31.898 dB` result used a centered metric and only a
`1.20x` fixed-graded radius span; it remains a historical ablation.

- `2d/hyper_e2e/` contains the canonical image-conditioned HyperNPA online
  training configs. The 10k config is a scale diagnostic until it passes the
  high-particle oracle/PSNR gates. These TOMLs include `[gates]` so throughput
  and validation-overhead regressions fail the run instead of becoming
  report-only warnings.
- `2d/parity/` contains upstream SelfOrg-NPA reference/parity gates.
- Dense Burn `train-target2d`, direct-basis adapter banks, static LoRA
  reconstruction, and direct-basis PSNR gates are hidden diagnostics until they
  pass the upstream parity harness.

`adaptive/` is the separate budgeted adaptive NPA path. Its maintained bundles gate
variable-support operator accuracy, canonical conservative events, hard graph
budgets, fixed-rule compatibility, exact hierarchical seed restoration, Burn
controller training, binary artifacts, isotropic measure-derived Gaussian
scale, and bounded topology without changing regular 2D behavior. Covariance
is simulation-only and cannot become a renderer scale/rotation channel.
`adaptive/recurrent_target2d_lizard_continuous_ratio4_smoke_3070_2d_wgpu.toml`
is the bounded direct-active regression. Its current candidate uses fixed
interaction bandwidth and a fixed resident row count, so it does not establish
continuous communication scale, runtime particle births/deaths, or broad
quality promotion.
`adaptive/task_resident_lizard_smoke_3070_2d_wgpu.toml` is the bounded
device-resident split/exchange regression. The fixed-graded recurrent bundle is
not a promotion gate: it reproduces an earlier narrow-scale control with
`require_adaptive_resolution = false`. Additional mixed-scale training and
quality experiments remain under `configs/sandbox/adaptive/` until they pass
uncentered deployment PSNR, long-horizon stability, dynamic-versus-static
allocation, conservation, overflow, work, and wall-time gates.

The legacy persistent reduced-budget evaluator applies one dyadic cut and is
`adaptive/task_budgeted_lizard_eval_3070_2d_wgpu.toml`. Its evidence remains in
`docs/benchmarks/adaptive_lizard_isotropic_reallocation_2026-07-20.json` for
comparison.

The maintained LoD evaluator is
`adaptive/task_lod_lizard_eval_3070_2d_wgpu.toml`, with a one-seed smoke
companion. It progressively exposes 1,024 to 4,096 visible leaves, then applies
a learned-cost-ranked mixed 2/3/4-child restriction to 3,070 visible leaves.
The 32-seed result uses 3,925 persistent dynamics rows, occupies four material
scale bins with 11.1% off-dyadic leaves, and measures `22.192 dB` mean PSNR
versus `22.080 dB` for regular 4,096-particle inference. The worst held-out gap
is `-0.568 dB`, so the config gates mean parity and a bounded tail rather than
claiming per-seed dominance. Evidence is in
`docs/benchmarks/adaptive_lizard_progressive_mixed_lod_2026-07-20.json`.

Do not promote a new 2D config into this tree unless it has a smoke config, a
production-quality config if applicable, a matching validation/report path, and
explicit gates for the metrics the run is meant to prove.
