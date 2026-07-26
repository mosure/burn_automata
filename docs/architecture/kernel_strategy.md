# Inference Kernel Strategy

## Current Inference Path

The direct WGPU path is inference-only. It does not use Burn tensors, Burn fusion, or Burn autodiff when built as:

```bash
cargo build -p burn_automata --no-default-features --features "cli gpu_wgpu"
scripts/ci/check_inference_features.sh
```

The current GPU rollout keeps positions, states, weights, density, and hashgrid storage in WGPU buffers:

1. `clear_grid_main`
2. `bin_particles_main`
3. `density_main`
4. `update_main`
5. optional `write_gaussian_main`

For `WgpuNeighborMode::TiledFixedCellBuckets`, the bin pass also appends cells
whose counter transitions from 0 to 1 into an active-cell list stored in the
grid buffer tail. It writes a separate three-u32 indirect dispatch buffer:

```text
[active_cell_count, max_bucket_blocks, 1]
```

The tiled density/update entry points dispatch from that indirect buffer, read
cell IDs from the compact active-cell list, and stage neighbor positions,
densities, and states into `var<workgroup>` chunks. Params are bound as a
uniform buffer so the portable eight-storage-buffer WGPU limit is still met.

For `WgpuNeighborMode::SortedCells`, the GPU path builds an exact overflow-free
cell range layout every step: clear per-cell counts, count particles, prefix-scan
counts in 256-cell blocks, scan block sums, add block offsets, and scatter
particle indices into a contiguous sorted-index array. Density and update then
traverse `[offset[cell], offset[cell + 1])` ranges. This reduces grid memory
from `O(cells * bucket_capacity)` to `O(cells + particles)` and removes
fixed-bucket overflow, but the extra scan/scatter passes are currently slower
than scalar fixed buckets on the tested GPU.

This is not an all-pairs search. Each particle only traverses the 3x3 2D or 3x3x3 3D local cell neighborhood. Dense 2D still becomes expensive because many particles occupy the same few cells, so the exact number of interacting pairs is genuinely high.

For scale-equivariant particle-hash grids, seed-size changes should not
accidentally change normalized simulation density. The runtime now derives an
effective hashgrid with:

```text
effective_eps = base_eps * seed_scale / reference_seed_scale
```

This applies only to `HashGridMode::Particle` and scale-equivariant NPA configs.
It keeps cell occupancy stable when a model is evaluated at a scale different
from the catalog/reference scale. Fixed-domain grids such as periodic texture
models keep their original `eps`. This does not remove the true dense-neighbor
cost: if a user intentionally keeps `eps` fixed and shrinks the physical seed
radius, the local particle density and exact neighbor count both increase.

## Fusion Review

Safe fusion boundaries:

| candidate | status | reason |
| --- | --- | --- |
| clear + bin | no | bin needs all grid heads/counters cleared before any particle writes |
| bin + density | no | density needs a complete grid for all particles; sorted mode also needs scan/scatter complete |
| density + update | no | update uses `density[j]` for every neighbor, so all particle densities must be globally complete |
| update + gaussian write | blocked for current layout | the step bind group still uses the portable storage-buffer budget for simulation data; adding four planar gaussian buffers exceeds the portable WGPU compute storage-buffer limit |
| intermediate viewer steps + gaussian write | done | only the final visible step writes gaussian buffers when `steps_per_frame > 1` |

Portable update+gaussian fusion would require one of:

- pack gaussian outputs into one storage buffer before handing them to `bevy_interleave`;
- reduce the step bind group further by packing positions/states/density or gaussian outputs;
- request higher storage-buffer limits for standalone WGPU and keep a fallback for Bevy devices created with default limits.

The current code keeps the portable two-pass gaussian write. Benchmarks show gaussian write overhead is small relative to neighbor traversal.

## Fixed-Bucket Correctness

Fixed buckets are exact only when `grid_max_overflow_count == 0`. A lower bucket capacity can look much faster because it drops particles from neighbor traversal, but that changes density, gradients, states, and final rendering. State creation now rejects explicit fixed/tiled bucket capacities that are already smaller than the initial max cell occupancy.

Benchmark output must therefore be filtered by:

```text
returncode == 0 && (grid_max_overflow_count if present, else grid_overflow_count) == 0
```

For `--step-timing` runs, prefer the stricter `grid_max_overflow_count == 0` and `grid_overflowed_steps == 0` fields. The benchmark matrix reports overflowed or rejected cases separately and never selects them as fastest exact configurations.

## Active-Cell Tiled Kernel

The implemented tiled fixed-bucket mode uses cell-local tiling with an
active-cell schedule. It avoids launching work for every empty grid cell and
uses the actual maximum bucket occupancy from the bin pass for the indirect
`y` dimension. The implemented sorted-cell mode uses a prefix-sum layout instead
of a capped bucket slab.

Data layout:

1. Fixed bucket or sorted cell storage:
   - `cell_counts[cell]`
   - `cell_slots[cell * cap + slot]`
   - `active_cells[k]` in the grid-buffer tail
   - `indirect_args = [active_cell_count, max_bucket_blocks, 1]`
2. During binning:
   - atomically append a cell to `active_cells` only when its counter transitions from 0 to 1;
   - atomically update the indirect `max_bucket_blocks` from observed occupancy;
   - store particle index into the cell slot.
3. Tiled density/update:
   - dispatch over active cells and target slot blocks;
   - load neighbor particle positions/states/density into `var<workgroup>` tiles;
   - every target particle in the workgroup accumulates against the shared tile;
   - preserve exact radius filtering and periodic/clamped boundary behavior.

Validation status:

- one-step CPU parity covers tiled texture 2D through `wgpu_neighbor_modes_match_cpu_oracle_for_2d`;
- shifted 3D particle-hash fixed buckets and tiled fixed buckets match the CPU oracle through `wgpu_particle_hashgrid_handles_shifted_3d_fixed_buckets`;
- explicit overflow counter remains mandatory: any nonzero overflow is invalid for both scalar and tiled fixed buckets;
- imported BPK hidden-state PSNR and rendered Gaussian PSNR remain validated through the existing catalog/render harnesses, which use the selected resident WGPU neighbor mode.

Benchmark matrix:

| axis | values |
| --- | --- |
| presets | `growing-2d`, `texture-2d`, `point-mnist`, `growing-3d-gs` |
| particles | 1k, 4k, 8k, 16k, 32k, 65k |
| modes | linked-list, fixed buckets, tiled fixed buckets, sorted cells, cooperative sorted cells, subgroup cooperative sorted cells |
| outputs | simulation-only, gaussian-write |
| metrics | ms/step, step p95/p99/max, jitter ratio, M particles/s, max overflow count, max occupancy, occupied cells |

Measured impact after the cooperative kernel pass:

- 2D `auto` now routes validated 1024-8192 particle workloads to cooperative sorted cells. This avoids fixed-bucket overflow/rejection on collapsed point and micro-cluster starts and removes the largest dense 2D tiled-kernel spikes in the local WGPU benchmark matrix.
- 3D `auto` now also routes validated 1024-8192 particle workloads to cooperative sorted cells. The cooperative update path reduces neighbor features across a 32-lane workgroup, then evaluates the MLP hidden/output layers across those lanes instead of serializing the model on lane 0.
- Larger 3D and 2D workloads remain deliberately bounded until their distributions are swept; the resolver keeps sparse sub-1024 particle 3D starts on linked lists and retains explicit overflow/rejection checks for fixed buckets.
- Auto now rejects concentrated distributions before dispatch only when they exceed the validated exact cooperative/tiled fallback range: cooperative sorted cells are capped at 8192 particles per cell, exact sorted/linked fallback is capped at 512 particles per cell, and full-cell tiled scans are capped at 2,048 particles per cell when the distribution occupies at most four cells. This prevents known larger point and micro-cluster stalls from entering the default GPU path.
- Sorted cells remain exact and memory-stable but slower in current measurements because count/scan/scatter overhead is not yet recovered by scalar contiguous range traversal. Cooperative sorted cells reuse the same compact layout but let one workgroup cooperatively reduce a target particle's neighbor range and MLP update, which is faster for validated 2D/3D 1k-8k cases.
- `SubgroupCooperativeSortedCells` is available as an explicit opt-in when the WGPU adapter exposes fixed 32-wide subgroups. Its shader is isolated in a separate module and is only compiled after `Features::SUBGROUP` is requested, so unsupported devices keep the portable cooperative path. The mode passed 2D and shifted 3D parity, but the local 4k/8k dense/point/micro sweep did not meet the promotion gate, so `auto` still resolves to `CooperativeSortedCells`.
- BVH now exists in five forms: a CPU structural oracle in `burn_automata_kernels::spatial`, an executable `WgpuNeighborMode::Bvh { leaf_size }` path, an executable `WgpuNeighborMode::GpuBvh { leaf_size }` fixed-order baseline, an executable `WgpuNeighborMode::GpuLbvh { leaf_size }` sorted-cell GPU baseline, and an executable `WgpuNeighborMode::GpuMortonLbvh { leaf_size }` Morton-key GPU baseline. `Bvh` rebuilds a median-split BVH from current GPU positions on the CPU, uploads packed nodes/leaf indices into the existing grid buffer, then runs density/update traversal in WGSL. `GpuBvh` initializes a complete fixed-order binary tree and reduces AABBs entirely on GPU, avoiding readback but not spatially ordering particles. `GpuLbvh` reuses the GPU sorted-cell count/scan/scatter order, builds BVH leaves over that spatially coherent order, and reduces the tree on GPU. `GpuMortonLbvh` generates Morton keys from clamped grid coordinates, bitonic-sorts `(key, particle)` pairs on GPU, and builds the same tree over Morton order. All executable BVH modes are exact for clamped grids and useful for ablation; the Morton path is intentionally simple and not yet a production radix LBVH.

## Latest Local Measurements

Representative post-change release WGPU `auto` timings on the current ARM/NVIDIA workstation:

| preset | particles | path | ms/step |
| --- | ---: | --- | ---: |
| `growing-2d` dense | 4,096 | cooperative sorted cells | 3.18 |
| `growing-2d` point | 4,096 | cooperative sorted cells | 3.38 |
| `growing-2d` micro-cluster | 4,096 | cooperative sorted cells | 5.61 |
| `texture-2d` dense | 4,096 | cooperative sorted cells | 1.31 |
| `texture-2d` point | 4,096 | cooperative sorted cells | 2.26 |
| `texture-2d` micro-cluster | 4,096 | cooperative sorted cells | 5.41 |
| `growing-3d-gs` dense | 4,096 | cooperative sorted cells | 1.63 |
| `growing-3d-gs` point | 4,096 | cooperative sorted cells | 3.82 |
| `growing-3d-gs` micro-cluster | 4,096 | cooperative sorted cells | 13.80 |
| `growing-2d` dense | 8,192 | cooperative sorted cells | 10.49 |
| `growing-2d` point | 8,192 | cooperative sorted cells | 7.43 |
| `growing-2d` micro-cluster | 8,192 | cooperative sorted cells | 23.78 |
| `texture-2d` dense | 8,192 | cooperative sorted cells | 3.15 |
| `texture-2d` point | 8,192 | cooperative sorted cells | 7.19 |
| `texture-2d` micro-cluster | 8,192 | cooperative sorted cells | 23.65 |
| `growing-3d-gs` dense | 8,192 | cooperative sorted cells | 3.38 |
| `growing-3d-gs` point | 8,192 | cooperative sorted cells | 11.90 |
| `growing-3d-gs` micro-cluster | 8,192 | cooperative sorted cells | 37.06 |

Subgroup cooperative probe from
`target/bench_gpu_subgroup_cooperative_8k.json`:

| preset/geometry | particles | cooperative ms/step | subgroup ms/step | subgroup delta |
| --- | ---: | ---: | ---: | ---: |
| `growing-2d` point | 4,096 | 2.59 | 2.37 | -8.5% |
| `growing-2d` point | 8,192 | 8.90 | 8.30 | -6.8% |
| `growing-2d` dense | 8,192 | 11.04 | 13.03 | +18.0% |
| `texture-2d` point | 4,096 | 2.07 | 2.67 | +29.0% |
| `texture-2d` micro-cluster | 8,192 | 25.24 | 28.46 | +12.8% |
| `growing-3d-gs` point | 8,192 | 13.16 | 14.44 | +9.7% |
| `growing-3d-gs` micro-cluster | 8,192 | 44.72 | 48.69 | +8.9% |

The subgroup path improves only the two `growing-2d` point rows in this sweep
and regresses dense or clustered rows, including p99 regressions on several
cases. It remains useful as an opt-in probe for future subgroup-specific
kernel work, but it is not promoted into `auto`.

Representative executable BVH ablation rows:

| preset/geometry | particles | best non-BVH | CPU-rebuilt BVH | GPU fixed-order BVH | GPU sorted LBVH | GPU Morton LBVH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `growing-3d-gs` dense | 4,096 | 8.89 ms | 12.41 ms | 22.46 ms | 21.25 ms | 17.38 ms |
| `growing-3d-gs` line | 4,096 | 14.71 ms | 35.03 ms | 110.63 ms | 87.44 ms | 89.17 ms |
| `growing-3d-gs` dense | 8,192 | 10.17 ms | 16.86 ms | 37.79 ms | 36.84 ms | 25.10 ms |
| `growing-3d-gs` line | 8,192 | 25.18 ms | 62.92 ms | 241.37 ms | 150.81 ms | 183.15 ms |
| `growing-2d` dense | 8,192 | 42.50 ms | 114.98 ms | 169.08 ms | 190.36 ms | not swept |
| `growing-2d` line | 8,192 | 65.76 ms | 163.30 ms | 151.85 ms | 214.58 ms | not swept |

BVH beats linked-list traversal on collapsed 3D line distributions, but the complete exact CPU-rebuilt path remains slower than the active-cell tiled mode. The no-readback fixed-order `GpuBvh` baseline is slower still on most rows because fixed particle order creates poor spatial tree quality. `GpuLbvh` improves the worst 3D line rows versus fixed-order GPU BVH by using a spatially coherent sorted-cell order, but scan/scatter plus stack traversal still leaves it slower than active-cell tiled grids at 4K-8K particles. `GpuMortonLbvh` improves broad dense/torus/uniform BVH coherence versus sorted-cell order, but its multi-dispatch bitonic sort is too expensive and is worse than sorted-cell order on the collapsed line case. The remaining BVH throughput question is therefore a true radix/persistent-order GPU LBVH builder/traverser with lower ordering overhead and stackless/short-stack traversal, measured with rebuild cost included.

The full ablation tables, BVH/tile/hash-grid candidate analysis, and paper-style
discussion are maintained from Rust `bench`/`bench-spatial` runs. Python paper
renderers have been retired; keep new sweep recipes as Rust commands or TOML
experiment bundles.

```bash
cargo run --release -p burn_automata --features gpu_wgpu --bin burn_automata -- bench --preset growing-3d-gs --particles 8192 --steps 8 --gpu --neighbor-mode auto --step-timing
cargo run --release -p burn_automata --features cli --bin burn_automata -- bench-spatial --preset growing-3d-gs --particles 8192 --strategy all
```

Generated benchmark output belongs under `target/` or `artifacts/`. Maintained
documentation records only selected, reproducible measurements.

Normalized 3D torus-morphogen seed-scale check at 8,192 particles and 12 WGPU
steps:

| seed scale | effective eps | max cell occupancy | ms/step |
| ---: | ---: | ---: | ---: |
| 0.04 | 0.005556 | 19 | 14.22 |
| 0.08 | 0.011111 | 19 | 14.11 |
| 0.16 | 0.022222 | 19 | 14.13 |
| 0.32 | 0.044444 | 19 | 14.29 |
| 0.72 | 0.100000 | 19 | 14.03 |
| 1.20 | 0.166667 | 19 | 13.89 |

Without normalized `eps`, the same 0.04 seed produced max occupancy 1,080 and
hundreds of milliseconds per step because it was a physically denser particle
cloud, not just a smaller view-scale version of the same rollout.

Adversarial low-dimensional 3D distributions remain expensive even with
normalized scale because the occupancy is genuinely high. At 8,192 particles,
line geometry used only 64 cells with max occupancy 164. Recent exact WGPU
measurements with normalized `eps`, 30 steps, and `bucket_capacity=256`:

| geometry | mode | max occupancy | overflow | ms/step |
| --- | --- | ---: | ---: | ---: |
| line, seed scale 0.04 | linked-list | 164 | 0 | 71.8 |
| line, seed scale 0.04 | fixed buckets | 164 | 0 | 41.6-62.2 |
| line, seed scale 0.04 | adaptive auto | 164 | 0 | 51.4-81.2 |
| line, seed scale 0.04 | sorted cells | 164 | 0 | 58.1 |
| line, seed scale 0.04 | tiled fixed buckets | 164 | 0 | 176.3 |
| torus seed, seed scale 0.72 | fixed buckets | 19 | 0 | 21.9 |
| torus seed, seed scale 0.72 | adaptive auto | 19 | 0 | 13.1 |
| torus seed, seed scale 0.72 | sorted cells | 19 | 0 | 45.4 |
| torus seed, seed scale 0.72 | tiled fixed buckets | 19 | 0 | 345.7 |

The tiled and sorted numbers are intentionally retained here: both paths are
implemented and exact, but this workload does not yet benefit enough from their
memory layouts on the tested GPU. The next meaningful improvement should
therefore be a warp/cooperative-neighbor strategy, not more seed-radius tuning
and not simply forcing tiled or sorted cells into auto mode.

The gaussian-write differences are within run-to-run noise. The throughput work should focus on neighbor memory reuse and active-cell scheduling rather than gaussian output writes.

The guarded clustered-density sweep is reproduced with Rust `bench` commands for
each preset/geometry/mode combination:

```bash
cargo run --release -p burn_automata --features gpu_wgpu --bin burn_automata -- bench --preset growing-2d --particles 8192 --steps 4 --gpu --geometry micro-cluster --neighbor-mode auto --step-timing
cargo run --release -p burn_automata --features gpu_wgpu --bin burn_automata -- bench --preset growing-2d --particles 8192 --steps 4 --gpu --geometry micro-cluster --neighbor-mode cooperative-sorted-cells --step-timing
```

The subgroup cooperative sweep used for the current opt-in decision is
reproduced the same way:

```bash
cargo run --release -p burn_automata --features gpu_wgpu --bin burn_automata -- bench --preset growing-2d --particles 8192 --steps 8 --gpu --geometry point --neighbor-mode subgroup-cooperative-sorted-cells --step-timing
```

The 8k auto/cooperative ceiling validation from the promotion pass is saved in
`target/bench_gpu_cooperative_8192_final.json` and
`target/bench_gpu_cooperative_8192_final.csv`. The repeated 8k auto-only sweep
used for the representative table is saved in
`target/bench_gpu_cooperative_8192_auto_repeats.json` and
`target/bench_gpu_cooperative_8192_auto_repeats.csv`.

Rejected follow-up probes:

- A sorted-cell tiled prototype that staged neighbor chunks in workgroup memory
  but assigned one lane per target particle was much slower than the 32-lane
  per-particle cooperative scan. At 8,192 particles and 8-step/3-repeat timing,
  `growing-2d` point regressed from 7.56 ms/step to 191.66 ms/step,
  `texture-2d` point regressed from 7.64 ms/step to 192.97 ms/step, and
  `growing-3d-gs` point regressed from 11.25 ms/step to 383.06 ms/step. The
  issue is lost target-level parallelism: reducing global rereads does not help
  if each target lane serializes the full neighbor range.
- A 64-lane cooperative scan compiled and passed parity, but it only improved
  one broad 2D row while regressing most point/micro and 3D rows. The validated
  default remains the 32-lane cooperative path. The next high-occupancy attempt
  should preserve at least the current per-target lane parallelism, likely via
  subgroup reductions or a true two-dimensional tile decomposition with partial
  accumulation buffers rather than one lane per target.

The reproducible gaussian-write sweep is reproduced with Rust `bench`:

```bash
cargo run --release -p burn_automata --features gpu_wgpu --bin burn_automata -- bench --preset growing-3d-gs --particles 16384 --steps 4 --gpu --gaussian --neighbor-mode auto
```

The seed-scale sensitivity sweep is reproduced with Rust `bench` runs using
`--normalize-seed-scale` and `--fixed-eps`:

```bash
cargo run --release -p burn_automata --features gpu_wgpu --bin burn_automata -- bench --preset growing-3d-gs --particles 8192 --steps 12 --gpu --seed-mode torus-morphogen-dense-3d --seed-scale 0.04 --normalize-seed-scale
cargo run --release -p burn_automata --features gpu_wgpu --bin burn_automata -- bench --preset growing-3d-gs --particles 8192 --steps 12 --gpu --seed-mode torus-morphogen-dense-3d --seed-scale 0.04 --fixed-eps
```

For kernel comparisons, prefer repeated medians:

```bash
cargo run --release -p burn_automata --features gpu_wgpu --bin burn_automata -- \
  bench --preset growing-3d-gs --particles 8192 --steps 30 --repeats 3 \
  --gpu --geometry line --seed-mode torus-morphogen-dense-3d \
  --seed-scale 0.04 --normalize-seed-scale --neighbor-mode auto
```

The CLI reports `min_avg_step_ms`, `median_avg_step_ms`, and
`max_avg_step_ms`; use the median field when comparing batched throughput. Add
`--step-timing` when measuring frame-time stability or spike regressions.
