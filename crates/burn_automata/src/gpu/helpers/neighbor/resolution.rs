use burn_automata_kernels::{Boundary, HashGridConfig, HashGridMode, build_hashgrid};

use crate::gpu::types::WgpuNeighborMode;
use crate::{AutomataError, AutomataResult};

use super::layout::resolved_neighbor_mode;
use super::storage::{
    ensure_grid_storage_binding_limit, exact_neighbor_fallback_mode, grid_storage_len_for_mode,
    max_fixed_bucket_capacity_for_binding_limit, scan_block_count,
};
use crate::gpu::helpers::util::u32_checked;

const MAX_AUTO_EXACT_CELL_OCCUPANCY: usize = 512;
const MAX_AUTO_TILED_CELL_OCCUPANCY: usize = 2048;
const MIN_AUTO_COOPERATIVE_CELL_OCCUPANCY: usize = 512;
const MAX_AUTO_COOPERATIVE_CELL_OCCUPANCY: usize = 8192;

pub(in crate::gpu) fn resolve_bucket_capacity(
    grid: &HashGridConfig,
    particle_count: usize,
    mode: WgpuNeighborMode,
) -> AutomataResult<usize> {
    let capacity = match mode {
        WgpuNeighborMode::LinkedList => 0,
        WgpuNeighborMode::FixedCellBuckets { capacity } => capacity,
        WgpuNeighborMode::TiledFixedCellBuckets { capacity } => capacity,
        WgpuNeighborMode::CooperativeSortedCells => 0,
        WgpuNeighborMode::Bvh { leaf_size } => {
            validate_bvh_mode(grid, leaf_size, "BVH")?;
            leaf_size
        }
        WgpuNeighborMode::GpuBvh { leaf_size } => {
            validate_bvh_mode(grid, leaf_size, "GPU BVH")?;
            leaf_size
        }
        WgpuNeighborMode::GpuLbvh { leaf_size } => {
            validate_bvh_mode(grid, leaf_size, "GPU LBVH")?;
            scan_block_count(grid.cell_count())?;
            leaf_size
        }
        WgpuNeighborMode::GpuMortonLbvh { leaf_size } => {
            validate_bvh_mode(grid, leaf_size, "GPU Morton LBVH")?;
            leaf_size
        }
        WgpuNeighborMode::SortedCells => 0,
        WgpuNeighborMode::Auto => auto_bucket_capacity(grid, particle_count),
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
        let resolved = resolved_neighbor_mode(requested, capacity);
        if matches!(
            resolved,
            WgpuNeighborMode::FixedCellBuckets { .. }
                | WgpuNeighborMode::TiledFixedCellBuckets { .. }
        ) {
            let (_nonempty_cells, max_occupancy) =
                initial_cell_occupancy_stats(grid, particle_count, positions)?;
            ensure_fixed_bucket_capacity_fits_initial_occupancy(capacity, max_occupancy)?;
        }
        return Ok((capacity, resolved));
    }

    let (nonempty_cells, max_occupancy) =
        initial_cell_occupancy_stats(grid, particle_count, positions)?;

    if should_use_cooperative_sorted_cells(grid, particle_count, nonempty_cells, max_occupancy) {
        ensure_auto_cooperative_scan_is_bounded(grid, particle_count, max_occupancy)?;
        return Ok((0, WgpuNeighborMode::CooperativeSortedCells));
    }

    if grid.dim == 2 {
        if let Some(capacity) = adaptive_fixed_bucket_capacity(grid, max_occupancy, particle_count)?
        {
            ensure_auto_tiled_scan_is_bounded(grid, particle_count, nonempty_cells, max_occupancy)?;
            return Ok((
                capacity,
                WgpuNeighborMode::TiledFixedCellBuckets { capacity },
            ));
        }
        ensure_auto_exact_fallback_is_bounded(grid, particle_count, max_occupancy)?;
        return exact_neighbor_fallback_mode(grid, particle_count);
    }

    if grid.mode == HashGridMode::Particle {
        if grid.dim == 3 && particle_count <= 2048 && max_occupancy >= 96 {
            ensure_auto_exact_fallback_is_bounded(grid, particle_count, max_occupancy)?;
            return exact_neighbor_fallback_mode(grid, particle_count);
        }
        if should_use_tiled_particle_grid(grid, particle_count, nonempty_cells, max_occupancy) {
            if let Some(capacity) =
                adaptive_fixed_bucket_capacity(grid, max_occupancy, particle_count)?
            {
                ensure_auto_tiled_scan_is_bounded(
                    grid,
                    particle_count,
                    nonempty_cells,
                    max_occupancy,
                )?;
                return Ok((
                    capacity,
                    WgpuNeighborMode::TiledFixedCellBuckets { capacity },
                ));
            }
            ensure_auto_exact_fallback_is_bounded(grid, particle_count, max_occupancy)?;
            return exact_neighbor_fallback_mode(grid, particle_count);
        }
        if max_occupancy >= 96 {
            if let Some(capacity) =
                adaptive_fixed_bucket_capacity(grid, max_occupancy, particle_count)?
            {
                return Ok((capacity, WgpuNeighborMode::FixedCellBuckets { capacity }));
            }
            ensure_auto_exact_fallback_is_bounded(grid, particle_count, max_occupancy)?;
            return exact_neighbor_fallback_mode(grid, particle_count);
        }
        return Ok((0, WgpuNeighborMode::LinkedList));
    }

    let capacity = resolve_bucket_capacity(grid, particle_count, requested)?;
    Ok((capacity, resolved_neighbor_mode(requested, capacity)))
}

fn ensure_auto_tiled_scan_is_bounded(
    grid: &HashGridConfig,
    particle_count: usize,
    nonempty_cells: usize,
    max_occupancy: usize,
) -> AutomataResult<()> {
    let concentrated = nonempty_cells <= 4;
    if max_occupancy <= MAX_AUTO_TILED_CELL_OCCUPANCY || !concentrated {
        return Ok(());
    }
    Err(AutomataError::InvalidArgument(format!(
        "WGPU auto would need a full-cell tiled scan for max cell occupancy {max_occupancy} across {nonempty_cells} occupied cells with {particle_count} particles on a {}D grid, which can cause frame-time spikes; reduce particle density, increase seed scale, or use fewer particles for this distribution",
        grid.dim
    )))
}

fn should_use_cooperative_sorted_cells(
    grid: &HashGridConfig,
    particle_count: usize,
    nonempty_cells: usize,
    max_occupancy: usize,
) -> bool {
    if grid.dim == 2 && (1024..=MAX_AUTO_COOPERATIVE_CELL_OCCUPANCY).contains(&particle_count) {
        return max_occupancy > 0;
    }
    if grid.dim == 3 && (1024..=MAX_AUTO_COOPERATIVE_CELL_OCCUPANCY).contains(&particle_count) {
        return max_occupancy > 0;
    }
    nonempty_cells <= 4 && max_occupancy >= MIN_AUTO_COOPERATIVE_CELL_OCCUPANCY
}

fn ensure_auto_cooperative_scan_is_bounded(
    grid: &HashGridConfig,
    particle_count: usize,
    max_occupancy: usize,
) -> AutomataResult<()> {
    if max_occupancy <= MAX_AUTO_COOPERATIVE_CELL_OCCUPANCY {
        return Ok(());
    }
    Err(AutomataError::InvalidArgument(format!(
        "WGPU cooperative sorted cells currently supports concentrated auto cases up to max cell occupancy {MAX_AUTO_COOPERATIVE_CELL_OCCUPANCY}, got {max_occupancy} with {particle_count} particles on a {}D grid; reduce particle density, increase seed scale, or use fewer particles for this distribution",
        grid.dim
    )))
}

fn ensure_fixed_bucket_capacity_fits_initial_occupancy(
    capacity: usize,
    max_occupancy: usize,
) -> AutomataResult<()> {
    if capacity >= max_occupancy {
        return Ok(());
    }
    Err(AutomataError::InvalidArgument(format!(
        "WGPU fixed bucket capacity {capacity} is smaller than initial max cell occupancy {max_occupancy}; use --neighbor-mode auto or a larger --bucket-capacity"
    )))
}

fn ensure_auto_exact_fallback_is_bounded(
    grid: &HashGridConfig,
    particle_count: usize,
    max_occupancy: usize,
) -> AutomataResult<()> {
    if max_occupancy <= MAX_AUTO_EXACT_CELL_OCCUPANCY {
        return Ok(());
    }
    Err(AutomataError::InvalidArgument(format!(
        "WGPU auto would need an exact neighbor scan for max cell occupancy {max_occupancy} with {particle_count} particles on a {}D grid, which can stall the GPU; reduce particle density, increase seed scale, or use fewer particles for this distribution",
        grid.dim
    )))
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

fn validate_bvh_mode(grid: &HashGridConfig, leaf_size: usize, label: &str) -> AutomataResult<()> {
    if leaf_size == 0 {
        return Err(AutomataError::InvalidArgument(format!(
            "{label} leaf_size must be greater than zero"
        )));
    }
    if grid.boundary == Boundary::Periodic {
        return Err(AutomataError::InvalidArgument(format!(
            "{label} WGPU mode currently supports clamped/non-periodic grids only"
        )));
    }
    Ok(())
}

fn auto_bucket_capacity(grid: &HashGridConfig, particle_count: usize) -> usize {
    if grid.mode == HashGridMode::Particle {
        return 0;
    }
    if grid.dim == 2 && grid.boundary == Boundary::Periodic {
        return 0;
    }
    if grid.dim == 2 && particle_count <= grid.cell_count().saturating_mul(4) {
        return 0;
    }
    if grid.dim == 2 {
        return particle_count;
    }
    if grid.dim == 3 && particle_count <= grid.cell_count() {
        return 0;
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

fn previous_power_of_two(value: usize) -> usize {
    if value == 0 {
        0
    } else {
        1usize << (usize::BITS - 1 - value.leading_zeros())
    }
}
