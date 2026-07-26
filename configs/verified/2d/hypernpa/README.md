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

Do not add sample-ID tables, raw adapter-vector regression, legacy flat
condition pooling, or one-off resume recipes here. Those belong in the ignored
sandbox unless they establish a new canonical path.
