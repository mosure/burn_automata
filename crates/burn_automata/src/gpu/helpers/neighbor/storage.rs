use burn_automata_kernels::HashGridConfig;

use crate::gpu::types::WgpuNeighborMode;
use crate::{AutomataError, AutomataResult};

use super::layout::is_bvh_neighbor_mode;
use crate::gpu::helpers::constants::{
    BVH_HEADER_U32, BVH_NODE_U32, DEFAULT_MAX_STORAGE_BUFFER_BINDING_BYTES,
    DEFAULT_MAX_STORAGE_BUFFER_BINDING_U32,
};

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
