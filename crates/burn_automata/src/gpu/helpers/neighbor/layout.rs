use crate::gpu::types::WgpuNeighborMode;

pub(in crate::gpu) fn neighbor_layout_code(mode: WgpuNeighborMode) -> u32 {
    match mode {
        WgpuNeighborMode::LinkedList => 0,
        WgpuNeighborMode::FixedCellBuckets { .. } => 1,
        WgpuNeighborMode::TiledFixedCellBuckets { .. } => 1,
        WgpuNeighborMode::SortedCells => 2,
        WgpuNeighborMode::CooperativeSortedCells => 6,
        WgpuNeighborMode::SubgroupCooperativeSortedCells => 7,
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
        WgpuNeighborMode::CooperativeSortedCells => WgpuNeighborMode::CooperativeSortedCells,
        WgpuNeighborMode::SubgroupCooperativeSortedCells => {
            WgpuNeighborMode::SubgroupCooperativeSortedCells
        }
        WgpuNeighborMode::LinkedList => WgpuNeighborMode::LinkedList,
    }
}
