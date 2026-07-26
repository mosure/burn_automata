# GPU Interop

`vendor/bevy_burn` is a small local facade, not a full replacement for upstream `bevy_burn`.

The current API records:

- Bevy `ShaderBuffer` asset handle.
- Byte offset and length.
- Transfer direction.
- Transfer kind: CPU staging or GPU storage intent.
- Binding metadata for storage buffers.

Implemented helpers:

- `shader_buffer_from_bytes` and `shader_buffer_with_size` produce Bevy 0.19 `ShaderBuffer` assets with the descriptor usage from `BurnBufferBinding`.
- `add_shader_buffer_bridge` and `add_empty_shader_buffer_bridge` insert those assets into `Assets<ShaderBuffer>` and return a `BurnBufferBridge` component plus a `BurnShaderBufferHandle` component.
- `burn_automata::gpu::WgpuAutomataExecutor` provides a native WGPU inference path. It builds local neighbor storage with GPU atomics, then runs density, perception/MLP, stochastic update masking, and Euler integration on storage buffers.
- `WgpuNeighborMode` selects `LinkedList`, `FixedCellBuckets { capacity }`, `TiledFixedCellBuckets { capacity }`, `SortedCells`, or `Auto`. Auto keeps normal particle-hash grids on linked lists, but switches high-occupancy particle-grid starts to fixed buckets with an adaptive capacity derived from the initial upload. Fixed buckets use contiguous per-cell slots and clear only counters/overflow each step. Tiled fixed buckets build a compact active-cell list during binning, use indirect dispatch over active cells and actual bucket-block occupancy, and stage neighbor chunks in workgroup memory for density/update. `SortedCells` counts, prefix-scans, and scatters particles into contiguous cell ranges every step, so it is exact and overflow-free with `O(cells + particles)` grid memory. Current benchmarks keep tiled and sorted opt-in because scalar fixed buckets are faster on the present GPU when capacity is sufficient.
- `WgpuAutomataExecutor::create_state` creates persistent ping-pong position/state buffers and cached ping-pong bind groups so multi-step rollout can remain GPU-resident after the initial seed upload.
- `WgpuAutomataExecutor::update_state_model` rewrites resident params/weights buffers for live-trained models when the model shape is unchanged, avoiding a particle-state rebuild for training-only updates.
- `WgpuAutomataExecutor::step_state_into_gaussians` runs one automata step and writes directly into the planar gaussian storage buffers expected by `bevy_interleave`/`bevy_gaussian_splatting`.
- `bevy_automata::automata_executor_from_render_device` builds the executor from Bevy's render device/queue, and `bevy_automata::gaussian_storage_buffer_refs` borrows `PlanarStorageGaussian3d` buffers as Burn gaussian output refs.

The intended path is:

1. Burn backend owns tensors during simulation or training.
2. `bevy_burn` exposes compatible buffers to Bevy render-world systems.
3. `bevy_interleave` describes splat layout and storage.
4. `bevy_gaussian_splatting` consumes Gaussian storage without CPU readback.

The current CLI WGPU benchmark and `infer --gpu` export paths read output buffers back for reporting and JSON artifacts. The benchmark reports `grid_overflow_count`; nonzero fixed-bucket overflow means the run dropped particles from neighbor traversal and should not be used as an exact result. The Bevy viewer path avoids per-step host transfers: main-world settings/model resources are extracted into the render app, persistent WGPU automata state is created from Bevy's `RenderDevice`/`RenderQueue`, training-only model revisions rewrite the resident WGPU weights buffer, and each render-frame step applies the configured update probability before writing directly into the prepared `PlanarStorageGaussian3d` storage buffers consumed by `bevy_gaussian_splatting`.

The remaining zero-copy work is deeper integration with backend-owned Burn tensor buffers for training-time tensors and a cooperative-neighbor storage path. The active-cell tiled fixed-bucket and sorted-cell kernels are available for profiling, but the current NPA update workload is barrier/scan/workgroup-underfill limited enough that scalar fixed buckets remain the fastest exact capped mode on the local GPU when overflow stays zero.

Headless validation covers the storage-buffer bridge, direct viewer bridge, compact automata gaussian captures, and offscreen gaussian screenshot smoke tests:

```bash
cargo test -p burn_automata --test gpu_wgpu --no-default-features --features "cli gpu_wgpu" -- --nocapture
cargo test -p bevy_automata --no-default-features --features "splatting gpu_wgpu" --test gaussian_gpu_link -- --nocapture
REQUIRE_BPK=1 scripts/validation/validate_gpu_e2e.sh
```
