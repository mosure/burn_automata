# Hyper-NPA: Conditioned Generative Neural Particle Automata

This document proposes an experimental path from fixed, per-target Neural Particle Automata (NPA) checkpoints toward a generalizable family of conditioned, generative, and test-time-adaptable NPA systems. The target system takes a 2D or 3D condition, extracts geometry-aware latents, chunks the domain into local particle regions, predicts sparse initialization priors and NPA parameters, rolls particles forward with a hyper-rectified-flow objective, and optionally performs test-time training (TTT) against self-supervised reconstruction or consistency losses.

The concrete motivating chain is:

```text
3D condition
  e.g. VGGT-Omega / VGGT-style world latents + DINOv3 dense features
    -> chunking
    -> sparse volume mask and initialization prior
    -> hyper-rectified-flow Neural Particle Automata
    -> test-time training
```

This should be treated as a research roadmap, not a committed architecture. The first objective is to make the current `burn_automata` codebase able to host these experiments while preserving the existing CPU oracle, WGPU parity tests, imported 2D model validation, and Bevy gaussian rendering harness.

## Motivation

Current NPA checkpoints are specialized: a model is trained for a target image, texture, or object-like behavior, then rolled out from seeded particles. That is useful for controlled simulation and rendering, but it does not yet provide a general conditional model that can:

- infer a particle initialization from a new image, video, scene, or 3D latent;
- generate a local NPA model for each spatial chunk;
- compose chunk-local behavior into a large 2D/3D world;
- adapt at test time without retraining a global model;
- preserve the strengths of NPA: local updates, particle-density/scale equivariance, interactive rollout, and direct gaussian-splat rendering.

The proposed Hyper-NPA reframes an NPA checkpoint as the output of a hypernetwork. Instead of training one update MLP per target, train a condition encoder plus parameter generator that emits chunk-local NPA weights, initialization priors, and rollout schedules.

## Terms

| term | meaning |
| --- | --- |
| NPA | Neural Particle Automata update model with SPH perception, local particle state, and position/state updates. |
| Hyper-NPA | A conditional model that generates NPA weights and initialization priors. |
| Chunk | A 2D tile or 3D block with local particles, local condition tokens, and optional overlap with neighbors. |
| Sparse volume prior | Occupancy/probability mask, particle density prior, seed state prior, and optional gaussian color/opacity prior. |
| Hyper-rectified flow | A rectified-flow style training objective where the generated NPA defines the velocity/update field that transports particles/states from an initialization distribution to target particles, features, or rendered observations. |
| TTT | Test-time training/adaptation over selected generated parameters, chunk embeddings, or low-rank adapters. |

## Reference Inputs

The condition encoder should be modular. Candidate sources:

- 2D single image: DINOv3 dense patch tokens, segmentation/mask tokens, depth-normal estimates, or latent diffusion features.
- Multi-view image set: VGGT/VGGT-Omega-style outputs such as camera, depth, point maps, point tracks, dense geometric features, and dynamic-scene features.
- 3D world latent: sparse voxel features, triplane features, gaussian cloud features, point-map tokens, or world-model memory.
- User controls: text embeddings, masks, anchor points, trajectories, physics tags, or desired render style.

The current proposal should not assume a single foundation model. It should define a `ConditionProvider` trait and treat VGGT-Omega, DINOv3, and future world latents as interchangeable providers that produce typed chunk tokens.

## Architecture

### High-Level Dataflow

```text
Condition source(s)
  -> ConditionProvider
      dense 2D tokens, 3D point/voxel tokens, camera metadata, confidence
  -> ChunkPlanner
      chunk bounds, overlap, sparse occupancy, token routing
  -> PriorHead
      particle count, seed positions, seed states, density/mask priors
  -> HyperNpaGenerator
      base NPA config, low-rank or full MLP weights, chunk adapters, rollout schedule
  -> NpaRollout
      WGPU/CubeCL particle simulation per chunk, halo exchange, gaussian buffer writes
  -> Losses and TTT
      render, geometry, feature, flow, consistency, regularization
```

### ConditionProvider

The provider normalizes upstream encoders into a small number of internal types:

```rust
pub trait ConditionProvider {
    fn encode(&self, input: ConditionInput) -> AutomataResult<ConditionField>;
}

pub struct ConditionField {
    pub spatial_dim: usize,
    pub world_bounds: Bounds,
    pub tokens: Vec<ConditionToken>,
    pub cameras: Vec<Camera>,
    pub confidence: Vec<f32>,
}

pub struct ConditionToken {
    pub position: [f32; 4],
    pub scale: [f32; 3],
    pub feature: Vec<f32>,
    pub kind: ConditionTokenKind,
}
```

Providers can be implemented outside the core crate first. The core crate should only own schema, validation, chunk routing, and serialization.

### ChunkPlanner

Chunking is the main scaling mechanism. A chunk should be able to run independently for most steps but exchange halo particles or boundary summaries on a schedule.

Chunk metadata:

- `chunk_id`, `level`, `bounds`, `interior_bounds`, `halo_bounds`;
- routed condition token indices;
- target particle budget and maximum bucket capacity;
- sparse occupancy mask and confidence statistics;
- neighbors and overlap transform.

Chunk types:

- `Tile2d`: image/texture/growing NPA.
- `Block3d`: static or dynamic 3D gaussian/volume NPA.
- `SurfacePatch3d`: particles constrained near a surface, point map, depth shell, or mesh proxy.
- `Hybrid`: surface particles plus sparse interior particles.

Initial chunking strategies:

1. Uniform grid, fixed overlap.
2. Occupancy-adaptive split based on mask density.
3. Feature-adaptive split based on DINO/VGGT token variance.
4. Camera/frustum-aware split for multi-view training.

### Sparse Volume Mask and Initialization Prior

The prior predicts where particles should exist and what they initially represent.

Outputs:

- occupancy logits over sparse voxels or tiles;
- particle count distribution per chunk;
- seed position distribution, either sampled from occupancy or predicted as anchors;
- initial state vectors;
- optional gaussian attributes: color, opacity, scale, rotation;
- confidence and uncertainty for TTT weighting.

The first implementation should be conservative: sample seed positions from a sparse occupancy grid, then predict only state vectors and per-chunk adapters. Later versions can predict anchors and birth/death schedules.

### HyperNpaGenerator

The generator maps chunk condition summaries to NPA parameters.

Parameterization options:

| option | description | tradeoff |
| --- | --- | --- |
| Full weights | Emit `w1`, `b1`, `w2`, `b2` per chunk. | Simple, expensive, can overfit chunks. |
| Low-rank adapters | Shared base NPA plus generated LoRA-style deltas. | Better scaling, good first target. |
| FiLM modulation | Shared NPA with generated feature-wise scale/bias. | Stable and cheap, less expressive. |
| Weight basis mixture | Generated coefficients mix a learned weight dictionary. | Useful for large chunk counts. |
| Token-conditioned update | NPA reads chunk tokens through cross-attention or sampled fields. | Expressive, more expensive and less local. |

Recommended first version:

```text
shared base NPA
  + generated per-chunk LoRA adapters for MLP layers
  + generated initial state prior
  + generated rollout schedule
```

This is now the preferred direction for 3D object training in this codebase:
train a shared 3D NPA dynamics basis across mesh/object families, then train a
small adapter for each object or chunk. Full per-object weight sets are useful
diagnostics, but they should not be the default promotion path. Object-specific
`ParticleSeed` variants should likewise stay legacy-only; target identity
belongs in the target representation, condition tokens, or generated adapter.

The generated parameter package should become an extension of `.bpk`, for example:

```text
.bpk
  manifest.json
  base_model/
  chunks/<id>/adapter.bin
  chunks/<id>/prior.bin
  chunks/<id>/condition_summary.bin
  provenance.json
```

### Hyper-Rectified Flow

Rectified flow trains a velocity field that transports samples from a source distribution to a target distribution. Hyper-NPA adapts this idea by making the NPA update field the generated transport field.

Training tuple:

```text
(condition C, source particle state X0, target representation X1, time t)
```

The hypernetwork generates chunk-local NPA parameters:

```text
theta_c = H(C_c)
```

The NPA produces particle/state velocity:

```text
v_theta(x_t, s_t, C_c) = NPA_theta_c(perception(x_t, s_t))
```

Training loss:

```text
L_flow = || v_theta(x_t, s_t, C_c) - stopgrad(X1 - X0) ||_rho
```

where `rho` can be masked by occupancy/confidence and split into position, state, gaussian, and feature terms.

Practical schedule:

1. Pretrain prior head to match target sparse masks and particle/gaussian attributes.
2. Train generated adapters with one-step flow supervision.
3. Train multi-step rollouts with render/geometry consistency.
4. Add stochastic update masking, chunk dropout, and halo exchange.
5. Add TTT on held-out scenes.

### Rollout and Chunk Coupling

Chunk-local rollout should stay on GPU. Each chunk owns particle buffers and generated parameters. Neighboring chunks exchange either:

- real halo particles copied into ghost buffers;
- summary particles at chunk boundaries;
- sparse field samples from neighboring condition tokens;
- gaussian splat buffers for render-only validation.

Start with overlapping chunks and stitch rendered outputs. Add explicit halo exchange only when overlap artifacts become measurable.

## Losses

### Geometry Losses

- Chamfer or nearest-neighbor distance between predicted and target point maps.
- Depth reprojection loss under known or predicted cameras.
- Normal consistency from local neighborhoods.
- Occupancy BCE or focal loss for sparse mask.
- Eikonal-like smoothness if an implicit field head is introduced.

### Render Losses

- RGB L1/Charbonnier on differentiable gaussian renders.
- Alpha/mask loss.
- LPIPS or DINO feature loss for perceptual alignment.
- Multi-view consistency through camera reprojection.

### Particle and Automata Losses

- Flow matching on particle position/state updates.
- Density regularization to avoid collapse.
- State entropy or variance floor to avoid dead state channels.
- Equivariance consistency under scale, translation, particle thinning, and chunk shifts.
- Boundary consistency across overlapping chunks.

### Hypernetwork Regularization

- Adapter norm penalty.
- Neighboring chunk adapter smoothness.
- Weight-basis entropy control.
- Condition dropout robustness.
- TTT delta norm and early-stopping validation.

## Training Paradigm

### Stage 0: Current-Code Compatibility

Use current seeded/imported NPAs as pseudo-targets. Generate a condition from the known target image/volume, train a small hypernetwork to reconstruct the existing `.bpk` weights or rollout behavior, and verify exact rollout parity.

Validation:

- generated `.bpk` loads through existing runtime;
- CPU and WGPU rollout parity still pass;
- generated NPA matches source checkpoint PSNR after fixed rollout.

### Stage 1: 2D Conditional Hyper-NPA

Input: image latent or DINO dense features. Output: chunk-local 2D NPA adapters and particle priors.

Tasks:

- lizard/polka imported checkpoint reconstruction;
- synthetic shapes/textures with known masks;
- held-out image-conditioned texture growth.

Metrics:

- rollout PSNR/SSIM/LPIPS;
- particle count versus quality;
- chunk boundary error;
- generated adapter size;
- WGPU ms/step.

### Stage 2: 3D Sparse Volume Hyper-NPA

Input: multi-view features or a frozen 3D world latent. Output: sparse 3D particle/gaussian prior plus 3D NPA adapters.

Tasks:

- point-map to gaussian cloud;
- depth-conditioned sparse volume growth;
- static scene chunk reconstruction;
- simple dynamic scene rollout.

Metrics:

- Chamfer/F-score;
- rendered PSNR/SSIM/LPIPS across held-out views;
- depth error;
- gaussian opacity/scale stability;
- chunk memory and throughput.

### Stage 3: Hyper-Rectified Flow

Train the generated NPA as a transport field from sparse priors to target representations.

Experiments:

- one-step flow matching versus multi-step rollout;
- full weight generation versus LoRA adapters;
- fixed particle count versus predicted count;
- no halo versus overlap-only versus halo exchange.

### Stage 4: TTT

At test time, freeze most global parameters and adapt only:

- chunk embeddings;
- LoRA adapter coefficients;
- sparse prior logits;
- rollout schedule scalars;
- optionally the final state decoder.

TTT losses should be self-supervised:

- multi-view photometric/render consistency;
- DINO/VGGT feature reprojection consistency;
- depth/point-map consistency;
- mask sparsity and density regularization.

Stop criteria:

- validation view loss plateau;
- adapter norm limit;
- render stability;
- maximum iteration budget.

## Validation Matrix

| capability | first validation | stricter validation |
| --- | --- | --- |
| Generated 2D NPA | generated BPK rollout PSNR against imported checkpoint | held-out image-conditioned generation |
| Generated 3D NPA | gaussian cloud finite/nonblank headless render | held-out multi-view render PSNR |
| Chunking | overlap render has no visible seams | quantitative boundary consistency |
| Flow training | one-step target velocity MSE decreases | multi-step rollout improves target metrics |
| TTT | loss decreases without NaNs | improves held-out views without overfitting observed views |
| GPU residency | no per-step host transfer in persistent path | Bevy render-world schedule drives gaussian storage directly |
| Throughput | 4k/16k particle benchmarks | 50k+ particles with chunking and sparse active masks |

## Codebase Structure Proposal

The current crates are close to the right split, but Hyper-NPA needs clearer experiment boundaries.

### Suggested Crates and Modules

```text
crates/
  burn_automata_kernels/
    reference CPU oracle
    WGPU/CubeCL kernel contracts
  burn_automata/
    NPA config, BPK, rollout, import, training baseline
  burn_automata_hyper/
    condition schema
    chunk planning
    hypernetwork configs
    generated BPK-H manifest
    losses and experiment runners
  burn_automata_viewer/
    Bevy runtime app, render-world scheduling, UI
  bevy_burn/
    reusable buffer bridge
```

If a new crate is too early, start with modules under `burn_automata::hyper` and promote once the schema stabilizes.

### Core Trait Boundaries

- `ConditionProvider`: converts external encoders into internal condition fields.
- `ChunkPlanner`: maps condition fields to chunk graphs.
- `PriorInitializer`: generates seed particles/states.
- `HyperGenerator`: emits base config, adapters, and schedules.
- `GeneratedNpaPackage`: serializable generated model artifact.
- `ChunkRolloutExecutor`: CPU/WGPU execution for chunk graphs.
- `RenderLossHarness`: differentiable or proxy render losses.
  A CPU 3D multi-view density/color/depth oracle is now implemented through
  `render-loss-3d`, and `train-render3d` provides an analytic CPU
  render-position-gradient proxy scaffold. Native differentiable or WGPU
  training through the full rollout remains future work.

The current 3D mesh implementation should be treated as explicit baselines, not
as the final Hyper-NPA target:

- Preferred direction: train a shared 3D NPA basis across many mesh targets and
  specialize each target with a small LoRA-style adapter or other compact
  parameter subset. The shared weights should encode common local
  communication, activation, materialization, and splat-scale dynamics; the
  adapter should encode object-specific geometry/material bias. A future
  HyperNPA generator should predict that adapter from a condition rather than
  emitting a full independent weight set for every object.
- The core library now exposes a first adapter primitive:
  `NpaLowRankAdapter` can be materialized onto a shared `NpaModel`, full MLP
  gradients can be projected into adapter gradients, and supervised adapter
  training can update only the adapter. Render-rollout training now uses this
  path for both single-target training and `train-render3d-adapters` shared-base
  suites. The suite can initialize and train an object-agnostic shared local 3D
  growth base across the target list before freezing it for per-object adapter
  fitting. It now supports built-in target sets (`core`, `primitives`, `many`)
  where `many` is a 12-object bank spanning torus, teapot, sphere, ellipsoid,
  cube, cylinder, cone, capsule, pyramid, bicone, dumbbell, and cross. It also
  supports manual or automatic held-out adapter-only targets, so reports can
  separate shared-dynamics training quality from LoRA specialization and
  adapter-only generalization quality. The suite evaluates the frozen shared
  base across all targets before fitting adapters, records train/holdout
  aggregate summaries, and emits an `adapter_bank.json` manifest. That manifest
  is the desired bridge artifact for HyperNPA: a shared BPK plus many compact
  `.adapter.json` object adapters, with materialized BPKs treated only as
  validation/viewer compatibility outputs. The manifest and full suite report
  carry `strategy="shared_base_low_rank_object_adapters"` and a many-object
  coverage contract, so a default suite is rejected as a scaling run if it
  silently collapses back to torus/teapot-only training or misses adapter
  artifacts for any target. A conditional HyperNPA should first learn to predict
  this adapter bank distribution, then move to chunk-local adapters and finally
  to full end-to-end parameter generation.
- Promotion-facing 3D seeds should be object-agnostic (`Growth3d`,
  `LocalGrowth3d`, `SubstrateGrowth3d`, `LocalSubstrateGrowth3d`). Object-named
  seed modes are legacy diagnostics for existing BPK lineage and should not be
  used as the mechanism for new object identity.

- `uv_torus_growth_3d.bpk`: the current torus regression artifact. It uses
  `position_features=false` and `ParticleSeed::TorusGrowth3d`, so it starts from
  a compact neutral sparse-core seed rather than target residual/color state.
  Its manifest records `render-refined-rust:...conditionless-local...` lineage
  from a guarded render-refinement pass over the conditionless-local random-ball
  rollout base. The refinement selected no better checkpoint, so its behavior is
  the latest bounded dynamic local-front baseline and still fails app-facing
  coverage/depth plus torus tube-angle support. It is kept under `assets/models`
  for validation/regression, but hidden from the selectable Bevy catalog until a
  future artifact passes those gates.
- `teapot_growth_3d.bpk`: the current teapot regression artifact. It uses
  `position_features=false`, `ParticleSeed::TeapotGrowth3d`, and
  `render-refined-rust:...conditionless-local...` lineage. It is the latest
  dynamic local-front baseline, but it is hidden from the selectable Bevy
  catalog because at the current 1024-particle interactive scale it still fails
  strict target coverage and rendered density.
- Render-proxy probes should be written to `target/` or `artifacts/` until they
  pass validation; they are no longer catalog assets.
- Legacy `uv_torus_3d.bpk`, `uv_torus_morphogen_3d.bpk`, and
  `teapot_morphogen_3d.bpk` artifacts are retired from `assets/models` and
  should be regenerated only as diagnostics.
- The current catalog mesh artifacts no longer read absolute position features,
  but they are still not solved fully local 3D morphogenetic models.
- `--training-mode projection-baseline`: an explicit seed-frame baseline that
  stores residual/color, oriented normal, and signed-distance channels for
  mesh-projection sanity checks.
- `--training-mode rollout-local`: local teacher distillation for experiments,
  useful for testing rollout-row training plumbing but not a substitute for
  rendered multi-step target losses.
- `--training-mode rollout-position-field`: rollout-state mesh rows for the
  position-field baseline. This currently improves/passes torus and exposes the
  teapot failure instead of hiding it.
- `ablate-local-3d`: the strict conditionless-local experiment. It uses
  `position_features=false`, random-ball seeds, no target residual/color state,
  refreshed local rollout rows, and mesh rollout validation. Current torus and
  teapot reports fail the geometry/color/opacity gates, which is the expected
  pressure test showing that the one-step mesh-projection proxy is not enough
  for fully local 3D morphogenesis.
- `render-loss-3d`: CPU multi-view orthographic Gaussian splat validation for
  3D rollouts. It compares relative alpha density, density-gated color, and
  depth moments against mesh target samples. Current catalog and local-ablation
  artifacts fail the stricter rendered density gates, so this is a validation
  objective and training target rather than a passed catalog claim.

This is a validation baseline for the proposed `PriorInitializer` and
`RenderLossHarness`. Arbitrary unconditioned mesh growth still needs a learned
condition/initializer and differentiable or proxy rendered/geometry rollout
loss rather than baked projection-state channels or absolute position fields.
See `docs/local_3d_morphogenesis.md` for the concrete ablation results and
acceptance gate.

### Artifact Formats

Add a generated package format rather than overloading current `.bpk` immediately.

```text
.bpkh
  manifest_version
  base_npa_config
  condition_schema
  chunk_graph
  prior_schema
  adapter_schema
  generated_weights
  training_provenance
```

Keep `.bpk` as a single realized NPA model. Add export commands:

```bash
burn_automata hyper infer-condition --input scene.json --output scene.bpkh
burn_automata hyper realize-chunk --input scene.bpkh --chunk 12 --output chunk_12.bpk
burn_automata hyper rollout --input scene.bpkh --gpu --steps 64
```

### GPU Execution Direction

The direct WGPU executor already supports persistent state, direct gaussian buffer writes, and linked-list/fixed-bucket local neighbor traversal. Hyper-NPA should build on that by adding:

- per-chunk parameter buffer arrays;
- chunk graph dispatch;
- active chunk masks;
- sparse compaction for live particles;
- halo/overlap buffers;
- generated adapter binding layout;
- render-world scheduling in Bevy.

For dense cases, continue toward cooperative-neighbor storage. The current fixed bucket path is useful for exact validation and moderate 2D speedups, the active-cell tiled fixed-bucket path is available for profiling, and the sorted-cell prefix-sum path gives exact overflow-free `O(cells + particles)` storage, but tens-of-thousands dense particles still need more coalesced memory access than the current tiled/sorted implementations provide.

## Experiment Backlog

1. Add `hyper` module schemas: condition field, chunk graph, generated adapter manifest.
2. Add a tiny MLP hypernetwork in Rust/Burn that emits LoRA adapters for one chunk.
3. Train on existing imported 2D BPK rollouts and validate generated rollout PSNR.
4. Add chunked 2D texture experiment with four overlapping tiles.
5. Add sparse 3D initialization prior and gaussian render target.
6. Add WGPU chunk parameter binding and multi-chunk dispatch.
7. Add TTT loop over generated adapter coefficients.
8. Add Bevy UI panels for condition source, chunk graph, TTT iterations, and rollout/render metrics.
9. Add paper-quality render snapshots and benchmark tables.

## Open Questions

- Should the hypernetwork generate NPA weights directly, or generate a latent that conditions a shared update model?
- Are particles the only generated primitive, or should Hyper-NPA generate gaussian priors and let NPA update them?
- How much TTT can run inside Burn/WGPU without breaking Bevy interactivity?
- What is the right chunk overlap schedule for dynamic scenes?
- Should VGGT-style geometric features be sampled continuously as a field during rollout, or compressed into chunk embeddings?
- Can equivariant update layers improve 3D generalization without making imported 2D checkpoint support incompatible?

## Near-Term Definition of Done

An initial Hyper-NPA milestone is complete when:

- a condition image or synthetic 3D condition produces a `.bpkh`;
- each chunk can be realized as a `.bpk`;
- generated 2D chunks reproduce imported NPA rollouts above 35 dB PSNR;
- WGPU chunk rollout has no per-step host transfer;
- Bevy headless render captures nonblank gaussian output;
- docs include reproducible commands and benchmark numbers.
