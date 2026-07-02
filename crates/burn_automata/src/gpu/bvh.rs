use crate::{AutomataError, AutomataResult};

use super::{constants::*, util::u32_checked};

#[derive(Clone, Debug)]
pub(in crate::gpu) struct CpuBvhNode {
    min: [f32; 3],
    max: [f32; 3],
    left_or_start: u32,
    right_or_count: u32,
    leaf: bool,
}

pub(in crate::gpu) fn build_bvh_storage_u32(
    positions: &[[f32; 4]],
    spatial_dims: usize,
    leaf_size: usize,
) -> AutomataResult<Vec<u32>> {
    if leaf_size == 0 {
        return Err(AutomataError::InvalidArgument(
            "BVH leaf_size must be greater than zero".to_owned(),
        ));
    }
    if !(spatial_dims == 2 || spatial_dims == 3) {
        return Err(AutomataError::InvalidArgument(format!(
            "BVH spatial_dims must be 2 or 3, got {spatial_dims}"
        )));
    }
    let mut indices = (0..positions.len()).collect::<Vec<_>>();
    let mut nodes = Vec::with_capacity(positions.len().saturating_mul(2).saturating_sub(1));
    let mut ordered = Vec::with_capacity(positions.len());
    if !positions.is_empty() {
        build_bvh_node(
            positions,
            spatial_dims,
            leaf_size,
            &mut indices,
            &mut nodes,
            &mut ordered,
        )?;
    }
    let index_base = BVH_HEADER_U32
        .checked_add(nodes.len().checked_mul(BVH_NODE_U32).ok_or_else(|| {
            AutomataError::InvalidArgument("BVH node storage length overflow".to_owned())
        })?)
        .ok_or_else(|| AutomataError::InvalidArgument("BVH index base overflow".to_owned()))?;
    let mut storage = Vec::with_capacity(index_base + ordered.len());
    storage.push(u32_checked(nodes.len(), "BVH node_count")?);
    storage.push(u32_checked(index_base, "BVH index_base")?);
    storage.push(0);
    storage.push(u32_checked(leaf_size, "BVH leaf_size")?);
    for node in &nodes {
        storage.push(node.min[0].to_bits());
        storage.push(node.min[1].to_bits());
        storage.push(node.min[2].to_bits());
        storage.push(node.max[0].to_bits());
        storage.push(node.max[1].to_bits());
        storage.push(node.max[2].to_bits());
        storage.push(node.left_or_start);
        storage.push(node.right_or_count);
        storage.push(u32::from(node.leaf));
    }
    for index in ordered {
        storage.push(u32_checked(index, "BVH particle index")?);
    }
    Ok(storage)
}

pub(in crate::gpu) fn build_bvh_node(
    positions: &[[f32; 4]],
    spatial_dims: usize,
    leaf_size: usize,
    indices: &mut [usize],
    nodes: &mut Vec<CpuBvhNode>,
    ordered: &mut Vec<usize>,
) -> AutomataResult<usize> {
    let (min, max) = bvh_bounds(positions, spatial_dims, indices);
    let node_index = nodes.len();
    nodes.push(CpuBvhNode {
        min,
        max,
        left_or_start: 0,
        right_or_count: 0,
        leaf: false,
    });
    if indices.len() <= leaf_size {
        let start = ordered.len();
        ordered.extend_from_slice(indices);
        nodes[node_index] = CpuBvhNode {
            min,
            max,
            left_or_start: u32_checked(start, "BVH leaf start")?,
            right_or_count: u32_checked(indices.len(), "BVH leaf count")?,
            leaf: true,
        };
        return Ok(node_index);
    }

    let extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    let axis = if spatial_dims == 3 && extent[2] > extent[0].max(extent[1]) {
        2
    } else if extent[1] > extent[0] {
        1
    } else {
        0
    };
    indices.sort_by(|lhs, rhs| {
        positions[*lhs][axis]
            .partial_cmp(&positions[*rhs][axis])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| lhs.cmp(rhs))
    });
    let mid = indices.len() / 2;
    let (left_indices, right_indices) = indices.split_at_mut(mid);
    let left = build_bvh_node(
        positions,
        spatial_dims,
        leaf_size,
        left_indices,
        nodes,
        ordered,
    )?;
    let right = build_bvh_node(
        positions,
        spatial_dims,
        leaf_size,
        right_indices,
        nodes,
        ordered,
    )?;
    nodes[node_index] = CpuBvhNode {
        min,
        max,
        left_or_start: u32_checked(left, "BVH left node")?,
        right_or_count: u32_checked(right, "BVH right node")?,
        leaf: false,
    };
    Ok(node_index)
}

pub(in crate::gpu) fn bvh_bounds(
    positions: &[[f32; 4]],
    spatial_dims: usize,
    indices: &[usize],
) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for &index in indices {
        for axis in 0..spatial_dims {
            let value = positions[index][axis];
            min[axis] = min[axis].min(value);
            max[axis] = max[axis].max(value);
        }
    }
    if spatial_dims == 2 {
        min[2] = 0.0;
        max[2] = 0.0;
    }
    (min, max)
}
