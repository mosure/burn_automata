# Verified Configs

Only configs that describe a maintained, validated path belong under
`configs/verified/`. Exploratory sweeps, failed experiments, large local runs,
and one-off diagnostics belong under `configs/sandbox/`, which is intentionally
gitignored.

Current 2D status:

- `2d/hyper_e2e/` contains the canonical image-conditioned HyperNPA online
  training configs.
- `2d/parity/` contains upstream SelfOrg-NPA reference/parity gates.
- Dense Burn `train-target2d`, direct-basis adapter banks, static LoRA
  reconstruction, and direct-basis PSNR gates are experimental until they pass
  the upstream parity harness.

Do not promote a new 2D config into this tree unless it has a smoke config, a
production-quality config if applicable, and a matching validation/report path.
