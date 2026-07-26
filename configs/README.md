# Experiment Configurations

Experiment behavior belongs in TOML rather than long command lines or
environment-variable bundles.

- [`verified/`](verified/) contains maintained, reviewable recipes.
- [`sandbox/`](sandbox/) is an ignored workspace for local experiments.

A verified pipeline should provide a bounded smoke recipe and, when a quality
claim exists, a separate quality or production recipe with explicit gates.
Reports, checkpoints, caches, and downloaded data must be written under ignored
`artifacts/`, `models/`, `.cache/`, or `data/` paths. Do not write generated
outputs back into `configs/`.

The directory layout follows the modeled domain:

```text
verified/
  2d/
    parity/
    hypernpa/
    adaptive/
  3d/
    oracle/
    adapters/
```

Moving a sandbox recipe into `verified/` requires:

1. a maintained CLI command and parser test;
2. deterministic seeds and explicit input/output paths;
3. bounded resource settings or a documented hardware contract;
4. machine-enforced correctness, quality, or throughput gates;
5. a matching entry in the nearest README.
