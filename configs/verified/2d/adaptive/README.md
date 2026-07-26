# Adaptive 2D Configurations

Adaptive NPA remains a research path layered beside, not in place of, fixed 2D
NPA inference.

| Directory | Purpose |
| --- | --- |
| [`audits/`](audits/) | CPU topology conservation and fixed-rule WGPU compatibility |
| [`training/`](training/) | Bounded active-material smoke, multiscale smoke/full recipe, and the retained three-stage lizard curriculum |
| [`evaluation/`](evaluation/) | Current continuous-scale, resident-topology, and long-horizon LoD checks |

The current paper candidate uses 3,070 resident rows to represent 4,096
fine-material units and occupies 33 scale bins. It is not promoted: the
1,024-step drift gate and dynamic-over-static quality benefit remain
unresolved. Its bounded regression is:

```bash
cargo run --release -p burn_automata \
  --features cli,backend_ndarray,backend_wgpu -- \
  eval-adaptive-target2d \
  --config \
  configs/verified/2d/adaptive/evaluation/recurrent_target2d_lizard_continuous_ratio4_smoke_3070_2d_wgpu.toml
```

The three CUDA stage files preserve the selected scale-conditioning and
event-aware training sequence. The multiscale full recipe is a reproducible
research contract, not official-trainer parity. Historical fixed-graded and
single-cut evaluators were removed from `verified/`.

Evidence and exact limitations are maintained in:

- [`docs/papers/adaptive/`](../../../../docs/papers/adaptive/)
- [`docs/research/adaptive_continuous_scale.md`](../../../../docs/research/adaptive_continuous_scale.md)
- [`docs/evidence/adaptive/`](../../../../docs/evidence/adaptive/)
