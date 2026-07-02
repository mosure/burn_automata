use burn_automata_kernels::{Boundary, HashGridConfig, HashGridMode, build_hashgrid};

use crate::{AutomataError, AutomataResult};

use super::super::types::WgpuNeighborMode;
use super::{constants::*, util::u32_checked};

pub(in crate::gpu) fn neighbor_layout_code(mode: WgpuNeighborMode) -> u32 {
    match mode {
        WgpuNeighborMode::LinkedList => 0,
        WgpuNeighborMode::FixedCellBuckets { .. } => 1,
        WgpuNeighborMode::TiledFixedCellBuckets { .. } => 1,
        WgpuNeighborMode::SortedCells => 2,
        WgpuNeighborMode::Bvh { .. } => 3,
        WgpuNeighborMode::GpuBvh { .. } => 3,
        WgpuNeighborMode::GpuLbvh { .. } => 4,
        WgpuNeighborMode::GpuMortonLbvh { .. } => 5,
        WgpuNeighborMode::Auto => 0,
    }
}

pub(in crate::gpu) fn is_bvh_neighbor_mode(mode: WgpuNeighborMode) -> bool {
    matches!(
        mode,
        WgpuNeighborMode::Bvh { .. }
            | WgpuNeighborMode::GpuBvh { .. }
            | WgpuNeighborMode::GpuLbvh { .. }
            | WgpuNeighborMode::GpuMortonLbvh { .. }
    )
}

pub(in crate::gpu) fn resolve_bucket_capacity(
    grid: &HashGridConfig,
    particle_count: usize,
    mode: WgpuNeighborMode,
) -> AutomataResult<usize> {
    let capacity = match mode {
        WgpuNeighborMode::LinkedList => 0,
        WgpuNeighborMode::FixedCellBuckets { capacity } => capacity,
        WgpuNeighborMode::TiledFixedCellBuckets { capacity } => capacity,
        WgpuNeighborMode::Bvh { leaf_size } => {
            if leaf_size == 0 {
                return Err(AutomataError::InvalidArgument(
                    "BVH leaf_size must be greater than zero".to_owned(),
                ));
            }
            if grid.boundary == Boundary::Periodic {
                return Err(AutomataError::InvalidArgument(
                    "BVH WGPU mode currently supports clamped/non-periodic grids only".to_owned(),
                ));
            }
            leaf_size
        }
        WgpuNeighborMode::GpuBvh { leaf_size } => {
            if leaf_size == 0 {
                return Err(AutomataError::InvalidArgument(
                    "GPU BVH leaf_size must be greater than zero".to_owned(),
                ));
            }
            if grid.boundary == Boundary::Periodic {
                return Err(AutomataError::InvalidArgument(
                    "GPU BVH WGPU mode currently supports clamped/non-periodic grids only"
                        .to_owned(),
                ));
            }
            leaf_size
        }
        WgpuNeighborMode::GpuLbvh { leaf_size } => {
            if leaf_size == 0 {
                return Err(AutomataError::InvalidArgument(
                    "GPU LBVH leaf_size must be greater than zero".to_owned(),
                ));
            }
            if grid.boundary == Boundary::Periodic {
                return Err(AutomataError::InvalidArgument(
                    "GPU LBVH WGPU mode currently supports clamped/non-periodic grids only"
                        .to_owned(),
                ));
            }
            scan_block_count(grid.cell_count())?;
            leaf_size
        }
        WgpuNeighborMode::GpuMortonLbvh { leaf_size } => {
            if leaf_size == 0 {
                return Err(AutomataError::InvalidArgument(
                    "GPU Morton LBVH leaf_size must be greater than zero".to_owned(),
                ));
            }
            if grid.boundary == Boundary::Periodic {
                return Err(AutomataError::InvalidArgument(
                    "GPU Morton LBVH WGPU mode currently supports clamped/non-periodic grids only"
                        .to_owned(),
                ));
            }
            leaf_size
        }
        WgpuNeighborMode::SortedCells => 0,
        WgpuNeighborMode::Auto => {
            if grid.mode == HashGridMode::Particle {
                return Ok(0);
            }
            if grid.dim == 2 && grid.boundary == Boundary::Periodic {
                return Ok(0);
            }
            if grid.dim == 2 && particle_count <= grid.cell_count().saturating_mul(4) {
                return Ok(0);
            }
            if grid.dim == 2 {
                return Ok(particle_count);
            }
            if grid.dim == 3 && particle_count <= grid.cell_count() {
                return Ok(0);
            }
            let average = particle_count.div_ceil(grid.cell_count().max(1));
            let multiplier = match (grid.dim, grid.boundary) {
                (2, _) => 32,
                _ => 8,
            };
            average
                .saturating_mul(multiplier)
                .max(grid.max_particles_per_block)
                .min(particle_count)
        }
    };
    u32_checked(capacity, "bucket_capacity")?;
    let resolved = resolved_neighbor_mode(mode, capacity);
    let storage_len =
        grid_storage_len_for_mode(grid.cell_count(), particle_count, capacity, resolved)?;
    ensure_grid_storage_binding_limit(storage_len, resolved)?;
    Ok(capacity)
}

pub(in crate::gpu) fn resolve_neighbor_mode_for_state(
    grid: &HashGridConfig,
    particle_count: usize,
    positions: &[[f32; 4]],
    requested: WgpuNeighborMode,
) -> AutomataResult<(usize, WgpuNeighborMode)> {
    if requested != WgpuNeighborMode::Auto {
        let capacity = resolve_bucket_capacity(grid, particle_count, requested)?;
        return Ok((capacity, resolved_neighbor_mode(requested, capacity)));
    }

    let (nonempty_cells, max_occupancy) =
        initial_cell_occupancy_stats(grid, particle_count, positions)?;

    if grid.dim == 2 {
        if let Some(capacity) = adaptive_fixed_bucket_capacity(grid, max_occupancy, particle_count)?
        {
            return Ok((
                capacity,
                WgpuNeighborMode::TiledFixedCellBuckets { capacity },
            ));
        }
        return exact_neighbor_fallback_mode(grid, particle_count);
    }

    if grid.mode == HashGridMode::Particle {
        if grid.dim == 3 && particle_count <= 2048 && max_occupancy >= 96 {
            return exact_neighbor_fallback_mode(grid, particle_count);
        }
        if should_use_tiled_particle_grid(grid, particle_count, nonempty_cells, max_occupancy) {
            if let Some(capacity) =
                adaptive_fixed_bucket_capacity(grid, max_occupancy, particle_count)?
            {
                return Ok((
                    capacity,
                    WgpuNeighborMode::TiledFixedCellBuckets { capacity },
                ));
            }
            return exact_neighbor_fallback_mode(grid, particle_count);
        }
        if max_occupancy >= 96 {
            if let Some(capacity) =
                adaptive_fixed_bucket_capacity(grid, max_occupancy, particle_count)?
            {
                return Ok((capacity, WgpuNeighborMode::FixedCellBuckets { capacity }));
            }
            return exact_neighbor_fallback_mode(grid, particle_count);
        }
        return Ok((0, WgpuNeighborMode::LinkedList));
    }

    let capacity = resolve_bucket_capacity(grid, particle_count, requested)?;
    Ok((capacity, resolved_neighbor_mode(requested, capacity)))
}

pub(in crate::gpu) fn initial_cell_occupancy_stats(
    grid: &HashGridConfig,
    particle_count: usize,
    positions: &[[f32; 4]],
) -> AutomataResult<(usize, usize)> {
    let snapshot = build_hashgrid(positions, 1, particle_count, grid)?;
    let mut nonempty_cells = 0usize;
    let max_occupancy = snapshot
        .bin_offsets
        .windows(2)
        .map(|window| window[1] - window[0])
        .inspect(|occupancy| {
            if *occupancy > 0 {
                nonempty_cells += 1;
            }
        })
        .max()
        .unwrap_or(0);
    Ok((nonempty_cells, max_occupancy))
}

pub(in crate::gpu) fn should_use_tiled_particle_grid(
    grid: &HashGridConfig,
    particle_count: usize,
    nonempty_cells: usize,
    max_occupancy: usize,
) -> bool {
    if grid.dim != 3 || max_occupancy < 64 {
        return false;
    }
    nonempty_cells.saturating_mul(32) <= particle_count.max(1)
}

pub(in crate::gpu) fn adaptive_tiled_bucket_capacity(
    max_occupancy: usize,
    particle_count: usize,
) -> AutomataResult<usize> {
    let with_headroom = max_occupancy
        .saturating_mul(2)
        .saturating_add(64)
        .max(max_occupancy.saturating_add(16));
    let capacity = with_headroom.next_power_of_two().min(particle_count.max(1));
    u32_checked(capacity, "adaptive tiled bucket_capacity")?;
    Ok(capacity)
}

pub(in crate::gpu) fn adaptive_fixed_bucket_capacity(
    grid: &HashGridConfig,
    max_occupancy: usize,
    particle_count: usize,
) -> AutomataResult<Option<usize>> {
    let target = adaptive_tiled_bucket_capacity(max_occupancy, particle_count)?;
    let Some(max_safe_capacity) =
        max_fixed_bucket_capacity_for_binding_limit(grid.cell_count(), particle_count)?
    else {
        return Ok(None);
    };
    if target <= max_safe_capacity {
        return Ok(Some(target));
    }
    if max_safe_capacity < max_occupancy {
        return Ok(None);
    }

    let reduced_power_of_two = previous_power_of_two(max_safe_capacity);
    let reduced = if reduced_power_of_two >= max_occupancy {
        reduced_power_of_two
    } else {
        max_safe_capacity
    };
    u32_checked(reduced, "adaptive reduced bucket_capacity")?;
    Ok(Some(reduced))
}

pub(in crate::gpu) fn max_fixed_bucket_capacity_for_binding_limit(
    cell_count: usize,
    particle_count: usize,
) -> AutomataResult<Option<usize>> {
    if cell_count == 0 {
        return Ok(None);
    }
    let overhead = cell_count
        .checked_add(1)
        .and_then(|value| {
            value.checked_add(active_grid_storage_len(cell_count, particle_count).ok()?)
        })
        .ok_or_else(|| {
            AutomataError::InvalidArgument("fixed bucket storage overhead overflow".to_owned())
        })?;
    if overhead >= DEFAULT_MAX_STORAGE_BUFFER_BINDING_U32 {
        return Ok(None);
    }
    Ok(Some(
        (DEFAULT_MAX_STORAGE_BUFFER_BINDING_U32 - overhead) / cell_count,
    ))
}

pub(in crate::gpu) fn exact_neighbor_fallback_mode(
    grid: &HashGridConfig,
    particle_count: usize,
) -> AutomataResult<(usize, WgpuNeighborMode)> {
    for mode in [WgpuNeighborMode::SortedCells, WgpuNeighborMode::LinkedList] {
        let storage_len = grid_storage_len_for_mode(grid.cell_count(), particle_count, 0, mode)?;
        if grid_storage_binding_len_fits(storage_len) {
            return Ok((0, mode));
        }
    }
    Err(AutomataError::InvalidArgument(format!(
        "no exact WGPU neighbor layout fits storage buffer binding limit for {} cells and {} particles",
        grid.cell_count(),
        particle_count
    )))
}

pub(in crate::gpu) fn previous_power_of_two(value: usize) -> usize {
    if value == 0 {
        0
    } else {
        1usize << (usize::BITS - 1 - value.leading_zeros())
    }
}

pub(in crate::gpu) fn resolved_neighbor_mode(
    requested: WgpuNeighborMode,
    bucket_capacity: usize,
) -> WgpuNeighborMode {
    match requested {
        WgpuNeighborMode::Auto if bucket_capacity == 0 => WgpuNeighborMode::LinkedList,
        WgpuNeighborMode::Auto => WgpuNeighborMode::FixedCellBuckets {
            capacity: bucket_capacity,
        },
        WgpuNeighborMode::FixedCellBuckets { .. } => WgpuNeighborMode::FixedCellBuckets {
            capacity: bucket_capacity,
        },
        WgpuNeighborMode::TiledFixedCellBuckets { .. } => WgpuNeighborMode::TiledFixedCellBuckets {
            capacity: bucket_capacity,
        },
        WgpuNeighborMode::Bvh { .. } => WgpuNeighborMode::Bvh {
            leaf_size: bucket_capacity,
        },
        WgpuNeighborMode::GpuBvh { .. } => WgpuNeighborMode::GpuBvh {
            leaf_size: bucket_capacity,
        },
        WgpuNeighborMode::GpuLbvh { .. } => WgpuNeighborMode::GpuLbvh {
            leaf_size: bucket_capacity,
        },
        WgpuNeighborMode::GpuMortonLbvh { .. } => WgpuNeighborMode::GpuMortonLbvh {
            leaf_size: bucket_capacity,
        },
        WgpuNeighborMode::SortedCells => WgpuNeighborMode::SortedCells,
        WgpuNeighborMode::LinkedList => WgpuNeighborMode::LinkedList,
    }
}

pub(in crate::gpu) fn grid_storage_len_for_mode(
    cell_count: usize,
    particle_count: usize,
    bucket_capacity: usize,
    mode: WgpuNeighborMode,
) -> AutomataResult<usize> {
    if matches!(mode, WgpuNeighborMode::SortedCells) {
        return sorted_grid_storage_len(cell_count, particle_count);
    }
    if matches!(mode, WgpuNeighborMode::GpuLbvh { .. }) {
        return sorted_grid_storage_len(cell_count, particle_count)?
            .checked_add(bvh_grid_storage_len(particle_count, bucket_capacity)?)
            .ok_or_else(|| {
                AutomataError::InvalidArgument("GPU LBVH storage length overflow".to_owned())
            });
    }
    if matches!(mode, WgpuNeighborMode::GpuMortonLbvh { .. }) {
        return morton_bvh_grid_storage_len(particle_count, bucket_capacity);
    }
    if is_bvh_neighbor_mode(mode) {
        return bvh_grid_storage_len(particle_count, bucket_capacity);
    }
    grid_storage_payload_len(cell_count, particle_count, bucket_capacity)?
        .checked_add(active_grid_storage_len(cell_count, particle_count)?)
        .ok_or_else(|| AutomataError::InvalidArgument("grid storage length overflow".to_owned()))
}

pub(in crate::gpu) fn ensure_grid_storage_binding_limit(
    storage_len: usize,
    mode: WgpuNeighborMode,
) -> AutomataResult<()> {
    if grid_storage_binding_len_fits(storage_len) {
        return Ok(());
    }
    Err(AutomataError::InvalidArgument(format!(
        "WGPU neighbor storage for {mode:?} requires {} u32 values ({:.3} MiB), exceeding the conservative {} MiB storage buffer binding limit",
        storage_len,
        storage_len as f64 * std::mem::size_of::<u32>() as f64 / (1024.0 * 1024.0),
        DEFAULT_MAX_STORAGE_BUFFER_BINDING_BYTES / (1024 * 1024)
    )))
}

pub(in crate::gpu) fn grid_storage_binding_len_fits(storage_len: usize) -> bool {
    storage_len <= DEFAULT_MAX_STORAGE_BUFFER_BINDING_U32
}

pub(in crate::gpu) fn grid_storage_payload_len(
    cell_count: usize,
    particle_count: usize,
    bucket_capacity: usize,
) -> AutomataResult<usize> {
    if bucket_capacity == 0 {
        cell_count.checked_add(particle_count).ok_or_else(|| {
            AutomataError::InvalidArgument("grid storage length overflow".to_owned())
        })
    } else {
        cell_count
            .checked_mul(bucket_capacity)
            .and_then(|slots| slots.checked_add(cell_count))
            .and_then(|with_counts| with_counts.checked_add(1))
            .ok_or_else(|| {
                AutomataError::InvalidArgument("grid storage length overflow".to_owned())
            })
    }
}

pub(in crate::gpu) fn grid_clear_len(
    cell_count: usize,
    bucket_capacity: usize,
) -> AutomataResult<usize> {
    if bucket_capacity == 0 {
        Ok(cell_count)
    } else {
        cell_count
            .checked_add(1)
            .ok_or_else(|| AutomataError::InvalidArgument("grid clear length overflow".to_owned()))
    }
}

pub(in crate::gpu) fn grid_clear_len_for_mode(
    cell_count: usize,
    bucket_capacity: usize,
    mode: WgpuNeighborMode,
) -> AutomataResult<usize> {
    if is_bvh_neighbor_mode(mode) {
        if matches!(mode, WgpuNeighborMode::GpuLbvh { .. }) {
            return Ok(cell_count);
        }
        return Ok(0);
    }
    grid_clear_len(cell_count, bucket_capacity)
}

pub(in crate::gpu) fn active_grid_storage_len(
    cell_count: usize,
    particle_count: usize,
) -> AutomataResult<usize> {
    Ok(cell_count.min(particle_count))
}

pub(in crate::gpu) fn sorted_grid_storage_len(
    cell_count: usize,
    particle_count: usize,
) -> AutomataResult<usize> {
    let block_count = scan_block_count(cell_count)?;
    cell_count
        .checked_add(cell_count.checked_add(1).ok_or_else(|| {
            AutomataError::InvalidArgument("sorted offsets length overflow".to_owned())
        })?)
        .and_then(|with_offsets| with_offsets.checked_add(particle_count))
        .and_then(|with_particles| with_particles.checked_add(block_count))
        .ok_or_else(|| {
            AutomataError::InvalidArgument("sorted grid storage length overflow".to_owned())
        })
}

pub(in crate::gpu) fn bvh_grid_storage_len(
    particle_count: usize,
    leaf_size: usize,
) -> AutomataResult<usize> {
    if leaf_size == 0 {
        return Err(AutomataError::InvalidArgument(
            "BVH leaf_size must be greater than zero".to_owned(),
        ));
    }
    let leaf_count = bvh_leaf_count_pow2(particle_count, leaf_size)?;
    let gpu_node_count = leaf_count
        .checked_mul(2)
        .and_then(|nodes| nodes.checked_sub(1));
    let cpu_node_count = particle_count.saturating_mul(2).saturating_sub(1);
    let node_count = gpu_node_count
        .map(|count| count.max(cpu_node_count))
        .ok_or_else(|| {
            AutomataError::InvalidArgument("BVH node storage length overflow".to_owned())
        })?;
    BVH_HEADER_U32
        .checked_add(node_count.checked_mul(BVH_NODE_U32).ok_or_else(|| {
            AutomataError::InvalidArgument("BVH node storage length overflow".to_owned())
        })?)
        .and_then(|with_nodes| with_nodes.checked_add(particle_count))
        .ok_or_else(|| AutomataError::InvalidArgument("BVH storage length overflow".to_owned()))
}

pub(in crate::gpu) fn morton_bvh_grid_storage_len(
    particle_count: usize,
    leaf_size: usize,
) -> AutomataResult<usize> {
    bvh_sort_count_pow2(particle_count)?
        .checked_mul(2)
        .and_then(|sort_storage| {
            sort_storage.checked_add(bvh_grid_storage_len(particle_count, leaf_size).ok()?)
        })
        .ok_or_else(|| {
            AutomataError::InvalidArgument("GPU Morton LBVH storage length overflow".to_owned())
        })
}

pub(in crate::gpu) fn bvh_leaf_count_pow2(
    particle_count: usize,
    leaf_size: usize,
) -> AutomataResult<usize> {
    if leaf_size == 0 {
        return Err(AutomataError::InvalidArgument(
            "BVH leaf_size must be greater than zero".to_owned(),
        ));
    }
    Ok(particle_count
        .div_ceil(leaf_size)
        .max(1)
        .next_power_of_two())
}

pub(in crate::gpu) fn bvh_sort_count_pow2(particle_count: usize) -> AutomataResult<usize> {
    if particle_count == 0 {
        return Err(AutomataError::InvalidArgument(
            "BVH sort particle_count must be greater than zero".to_owned(),
        ));
    }
    Ok(particle_count.next_power_of_two())
}

pub(in crate::gpu) fn bvh_level_count(leaf_count_pow2: usize) -> usize {
    if leaf_count_pow2 <= 1 {
        return 0;
    }
    leaf_count_pow2.trailing_zeros() as usize
}

pub(in crate::gpu) fn scan_block_count(cell_count: usize) -> AutomataResult<usize> {
    let count = cell_count.div_ceil(256);
    if count > 256 {
        return Err(AutomataError::InvalidArgument(format!(
            "sorted WGPU scan supports at most 65536 cells, got {cell_count}"
        )));
    }
    Ok(count)
}
