# Verified Configurations

This tree contains the repository's maintained experiment contracts, not a
history of every run.

## Status

| Path | Contract |
| --- | --- |
| [`2d/parity/`](2d/parity/) | Official SelfOrg-NPA import and parity gates |
| [`2d/hypernpa/`](2d/hypernpa/) | DINO-conditioned HyperNPA smoke, benchmark, published-control, and end-to-end recipes |
| [`2d/adaptive/`](2d/adaptive/) | Conservative adaptive topology, training, and evaluation recipes |
| [`3d/oracle/`](3d/oracle/) | Experimental Burn-native 3D overfit smoke and quality recipes |
| [`3d/adapters/`](3d/adapters/) | Experimental shared-base plus per-target 3D adapter recipes |

The imported fixed 2D catalog is the production inference baseline.
From-scratch Burn oracle training, broad held-out HyperNPA quality, adaptive
long-horizon parity, and all 3D training remain research paths until their
quality gates pass.

## Policy

- Smoke configs prove dispatch, serialization, and bounded numerical behavior.
- Quality configs encode the largest retained reproducible claim.
- Benchmark configs must synchronize the measured work and state the measured
  device shape.
- A file labeled historical, failed, probe, or continuation does not belong in
  this tree.
- Checked-in configs must not depend on another file under `configs/sandbox/`.

See [`../README.md`](../README.md) for promotion requirements.
