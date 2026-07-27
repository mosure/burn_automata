# HyperNPA Configurations

These recipes all target the maintained image-conditioned pipeline:

```text
image -> DINOv2 spatial tokens -> conditioned row flow
      -> shared NPA plus generated residual -> rollout loss
```

| Directory | Purpose |
| --- | --- |
| [`e2e/`](e2e/) | Teacher-free smoke and OmniSVG-1k quality-scale contract |
| [`flow/`](flow/) | Structured flow smoke, amortized-stage smoke, and fixed-endpoint production shape |
| [`published/`](published/) | Three-stage growing-catalog control used by the current paper |
| [`benchmarks/`](benchmarks/) | Dense/tiled Target2D and quality-scale CUDA throughput gates |

The production-shaped files are resource and objective contracts, not evidence
of broad 26 dB generalization. The current measured claim boundary is in
[`docs/research/hypernpa_status.md`](../../../../docs/research/hypernpa_status.md)
and the paper under [`docs/papers/hypernpa/`](../../../../docs/papers/hypernpa/).

The maintained e2e objective uses log-compressed hierarchical tails: a hard
trajectory tail is computed independently inside each image identity, then a
second tail emphasizes hard identities. Validation keeps the identity subset
fixed across seeds and selects checkpoints using the p10 over all trajectories
from the configured long-horizon set. Per-horizon and worst-seed diagnostics
remain in `report.json`.

## Throughput Gates

Current local release measurements on the RTX PRO 6000 Blackwell:

| Config | Backend/path | Median particle-steps/s |
| --- | --- | ---: |
| `bench_omnisvg_8_b4_p128.toml` | WGPU dense Target2D | 96,936 |
| `bench_omnisvg_8_b4_p128_tiled.toml` | WGPU guarded-batched Target2D | 114,798 |
| `throughput_omnisvg_64_b64_p1024_s96_cuda.toml` | CUDA guarded-batched Target2D | 32,132,265 |

The guarded-batched Target2D custom op limits temporary workspace to 12 MiB,
reserves a duplicate tail row for Cube/Fusion correctness, and falls back to
sample-local launches when the image/particle shape is too large. At the CUDA
quality benchmark shape this reduces Target2D launches from 64 to 6 per step
and reduces the profiled Target2D stage from roughly 24.9 ms to 3.9 ms. The
checked-in minimum-throughput gates are intentionally below these medians but
above the superseded row-local implementations.

Do not add sample-ID tables, raw adapter-vector regression, legacy flat
condition pooling, or one-off resume recipes here. Those belong in the ignored
sandbox unless they establish a new canonical path.
