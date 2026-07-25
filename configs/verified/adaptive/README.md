# Verified Budgeted Adaptive NPA Configs

These checked-in configs include bounded execution smokes and reproducible
historical controls. The canonical structural contract uses 3,070 visible,
recurrent, and interaction rows, no hidden fine state, and represents 4,096
fine material units. The direct-active candidate spans a continuous `2x`
isotropic radius range, but its broad quality audit is not promoted.
Unvalidated quality candidates remain in `configs/sandbox/adaptive/`.

The adaptive path is additive and does not change the hardened regular 2D
kernels or BPK format. Foundation configs audit the conservative operators.
Task configs preserve the regular NPA rule and train only the topology
controller unless a separate experimental bundle explicitly enables residuals.

- `foundation_compatibility_smoke_2d_wgpu.toml` is the bounded numerical,
  serialization, and fixed-rule compatibility regression.
- `recurrent_target2d_active_material_smoke_2d_wgpu.toml` executes two
  recurrent Burn/WGPU optimizer steps with a 61-active/64-reference material
  layout, paired topology, binary artifact export, and JSON reporting. It
  proves the canonical trainer path executes; its tiny shape and two updates
  are deliberately not a lizard-quality result.
- `recurrent_target2d_lizard_events1_eval_3070_2d_wgpu.toml` is a historical
  fixed-graded control. Its `1.20x` radius span and fixed interaction support
  deliberately set `require_adaptive_resolution = false`; it must not be used
  as adaptive-resolution promotion evidence.
- `recurrent_target2d_lizard_continuous_ratio4_smoke_3070_2d_wgpu.toml` is the
  bounded direct-active regression. It rebuilds the adaptive artifact from the
  trained scale-conditioned rule, uses 3,070 rows and eight local-detail
  exchanges per 64 steps, requires at least 32 occupied continuous scale bins
  and a `1.99x` radius span, and gates matched-seed quality, oracle gap,
  conservation, overflow, interaction work, wall time, and scale-detail
  allocation over steps 96 through 1,024. The 32-seed protocol remains a
  sandbox experiment because its long tail misses promotion gates.
- `task_resident_lizard_smoke_3070_2d_wgpu.toml` verifies the canonical
  resident 1-to-4 bootstrap and paired 4-to-1/1-to-4 exchanges at a constant
  active-row budget. It is a structural and throughput smoke, not a
  lizard-quality result.
- `recurrent_target2d_lizard_stage1_scale_age1024_2d_cuda.toml`,
  `recurrent_target2d_lizard_stage2_scale_tail_age4096_2d_cuda.toml`, and
  `recurrent_target2d_lizard_stage3_fullrule_tail_age4096_2d_cuda.toml`
  preserve the superseded fixed-graded CUDA curriculum. Stage 1 adapts the material
  scale input, stage 2 adds 4,096-step pool ages and a final-quarter trajectory
  loss, and stage 3 unfreezes the complete recurrent rule. Their in-training
  validation is deliberately diagnostic. No current promotion claim is attached
  to these configs.
- `continuous_topology_smoke.toml` and `continuous_topology_full.toml` isolate
  conservative equal/unequal event algebra from model loading and rollout. The
  full bundle covers 100,000 parents in dimensions 2 through 4, with one equal
  and one bounded-unequal event per parent.
- `foundation_compatibility_full_2d_wgpu.toml` runs the 100,000-event audit,
  larger manufactured/operator grid, 1,024/4,096-particle graph sweeps, and the
  analytic controller fit. It is not a task-trained adaptive lizard model.
- `task_multiscale_lizard_smoke_2d_wgpu.toml` is the bounded WGPU regression for
  exact hierarchical bootstrap, on-policy-only controller replay, learned
  event gates with refinement-defect allocation, matched task evaluation, and
  binary artifact export.
- `task_multiscale_lizard_full_2d_cuda.toml` reproduces the historical
  retained-mode 4,096-particle comparison. It selects the best regular checkpoint across
  random-horizon, fixed-96, and world-frame Target2D phases, restores a matched
  4,096-particle seed exactly from 1,024 coarse leaves, trains controller event
  gates and bandwidth modulation, uses deterministic refinement defect for
  stable spatial allocation, and gates mean/worst adaptive-versus-regular PSNR
  over seeds 42 through 49.
- `task_budgeted_lizard_eval_3070_2d_wgpu.toml` is the historical 32-seed
  held-out protocol for the one-shot isotropic two-mode persistent artifact. It reports 3,070 visible
  material leaves and 3,412 recurrent dynamics rows, applies one same-budget
  reallocation at step 240, and is invoked with an explicit artifact via
  `eval-adaptive-npa`. Its bounded eight-seed gap decomposition keeps the old
  covariance decoder strictly as a nondeployable counterfactual.
- `task_lod_lizard_eval_3070_2d_wgpu.toml` is the maintained 32-seed LoD
  protocol, with `task_lod_lizard_smoke_3070_2d_wgpu.toml` as its two-seed
  companion. It bootstraps 1,024 to 4,096 visible leaves over steps 1 through
  4, then applies bounded rolling mixed 2/3/4-child restriction over steps 230
  through 240. The final state has 3,070 visible leaves, 3,925 recurrent rows,
  four material-scale bins, and 11.1% off-dyadic leaves.

Run a bundle with:

```bash
cargo run --release -p burn_automata --bin burn_automata -- \
  adaptive-npa --config \
  configs/verified/adaptive/foundation_compatibility_smoke_2d_wgpu.toml
```

Run the selected recurrent curriculum in stage order:

```bash
cargo run --release -p burn_automata \
  --no-default-features --features cli,backend_cuda,gpu_wgpu \
  --bin burn_automata -- train-adaptive-target2d \
  --config configs/verified/adaptive/recurrent_target2d_lizard_stage1_scale_age1024_2d_cuda.toml

cargo run --release -p burn_automata \
  --no-default-features --features cli,backend_cuda,gpu_wgpu \
  --bin burn_automata -- train-adaptive-target2d \
  --config configs/verified/adaptive/recurrent_target2d_lizard_stage2_scale_tail_age4096_2d_cuda.toml

cargo run --release -p burn_automata \
  --no-default-features --features cli,backend_cuda,gpu_wgpu \
  --bin burn_automata -- train-adaptive-target2d \
  --config configs/verified/adaptive/recurrent_target2d_lizard_stage3_fullrule_tail_age4096_2d_cuda.toml
```

Run the topology-only release audit with:

```bash
cargo run --release -p burn_automata --bin burn_automata -- \
  audit-adaptive-topology --config \
  configs/verified/adaptive/continuous_topology_full.toml
```

Run the bounded direct-active lizard smoke with:

```bash
cargo run --release -p burn_automata \
  --no-default-features --features cli,backend_wgpu,gpu_wgpu \
  --bin burn_automata -- eval-adaptive-target2d \
  --config configs/verified/adaptive/recurrent_target2d_lizard_continuous_ratio4_smoke_3070_2d_wgpu.toml
```

The fixed-topology coarse-closure identifiability diagnostic has its own thin
TOML command. The checked-in sandbox bundle is capped at 256 particles because
it uses the all-pairs CPU oracle; it is a correctness/control experiment, not a
quality-scale performance result:

```bash
cargo run --release -p burn_automata --bin burn_automata -- \
  audit-adaptive-closure --config \
  configs/sandbox/adaptive/lizard_closure_identifiability_cpu.toml
```

The matched 4,096-to-3,070 quality-scale audit keeps teacher stepping resident
on WGPU and reads back only five bounded snapshots per rollout:

```bash
cargo run --release -p burn_automata --features gpu_wgpu \
  --bin burn_automata -- audit-adaptive-closure --config \
  configs/sandbox/adaptive/lizard_closure_identifiability_wgpu.toml
```

Recurrent compact closure training remains a CPU-reference diagnostic. Its
causal geometry/state correction improved the measured closure target but did
not pass held-out recurrent gates, and resident WGPU closure deployment is
rejected rather than silently using the older incomplete feature layout. It is
not represented by a verified config. Results are recorded in
`docs/benchmarks/adaptive_recurrent_closure_causality_2026-07-22.json`.

Generated `.adaptive.bpk` files are checksummed binary MessagePack containers.
Experiment reports are JSON because they are tabular analysis output, not model
weight storage.

The foundation bundles use `models/catalog/growing/lizard.bpk` as their fixed
compatibility rule. The superseded fixed-graded recurrent run also started
from that released rule and optimized the shared rule under the adaptive
Target2D objective. Its centered-metric result is preserved, with explicit
historical status, in
`docs/benchmarks/adaptive_recurrent_target2d_lizard_2026-07-23.json`. The older
controller-training result remains in
`docs/benchmarks/adaptive_lizard_parity_2026-07-18.json`.

The historical covariance-render budget sweep remains in
`docs/benchmarks/adaptive_lizard_budget_sweep_2026-07-20.json` and is diagnostic,
not a production render claim. The corrected one-shot isotropic objective,
selector, late-reallocation sweep, and historical 32-seed result are in
`docs/benchmarks/adaptive_lizard_isotropic_reallocation_2026-07-20.json`.
The maintained progressive mixed-arity evidence is in
`docs/benchmarks/adaptive_lizard_progressive_mixed_lod_2026-07-20.json`.
