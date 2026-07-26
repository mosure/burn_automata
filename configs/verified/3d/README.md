# Verified 3D Experiment Configurations

The 3D trainers are experimental and do not provide a catalog-quality
morphogenesis model yet.

- [`oracle/`](oracle/) contains torus overfit smoke and quality recipes for
  `train-render3d`.
- [`adapters/`](adapters/) contains shared-base plus per-target adapter recipes
  for `train-render3d-adapters`.

The smoke recipes prove command dispatch and artifact/report persistence. The
quality recipes are retained research contracts and must not be interpreted as
passing catalog promotion. Current blockers and validation commands are in
[`docs/research/3d_status.md`](../../../docs/research/3d_status.md).
