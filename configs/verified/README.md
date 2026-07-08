# Verified Configs

Only configs that describe a maintained, validated path belong under
`configs/verified/`. Exploratory sweeps, failed experiments, large local runs,
and one-off diagnostics belong under `configs/sandbox/`, which is intentionally
gitignored.

Current 2D status:

- `2d/hyper_e2e/` contains the canonical image-conditioned HyperNPA online
  training configs. The 10k config is a scale diagnostic until it passes the
  high-particle oracle/PSNR gates. These TOMLs include `[gates]` so throughput
  and validation-overhead regressions fail the run instead of becoming
  report-only warnings.
- `2d/parity/` contains upstream SelfOrg-NPA reference/parity gates.
- Dense Burn `train-target2d`, direct-basis adapter banks, static LoRA
  reconstruction, and direct-basis PSNR gates are hidden diagnostics until they
  pass the upstream parity harness.

Do not promote a new 2D config into this tree unless it has a smoke config, a
production-quality config if applicable, a matching validation/report path, and
explicit gates for the metrics the run is meant to prove.
