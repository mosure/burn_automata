#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    Boundary, HashGridConfig, HashGridMode, KernelError, KernelResult,
    hashgrid::{build_hashgrid, cell_coords_for_position, cell_index_from_coords, neighbor_delta},
    tile::{TileGridConfig, assign_tiles},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SpatialStrategyKind {
    HashGrid,
    TileBlocks { tile_size: [usize; 3] },
    Bvh { leaf_size: usize },
}

impl SpatialStrategyKind {
    pub fn label(self) -> &'static str {
        match self {
            SpatialStrategyKind::HashGrid => "hash-grid",
            SpatialStrategyKind::TileBlocks { .. } => "tile-blocks",
            SpatialStrategyKind::Bvh { .. } => "bvh",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpatialStrategyReport {
    pub strategy: SpatialStrategyKind,
    pub batch_size: usize,
    pub particle_count: usize,
    pub dim: usize,
    pub eps: f32,
    pub cell_count: usize,
    pub active_bins: usize,
    pub max_bin_occupancy: usize,
    pub candidate_entries_visited: usize,
    pub candidate_tests: usize,
    pub exact_neighbor_pairs: usize,
    pub node_visits: usize,
    pub leaf_visits: usize,
    pub node_count: usize,
    pub max_depth: usize,
}

impl SpatialStrategyReport {
    pub fn candidates_per_particle(&self) -> f64 {
        self.candidate_tests as f64 / self.total_particles().max(1) as f64
    }

    pub fn entries_per_particle(&self) -> f64 {
        self.candidate_entries_visited as f64 / self.total_particles().max(1) as f64
    }

    pub fn exact_neighbors_per_particle(&self) -> f64 {
        self.exact_neighbor_pairs as f64 / self.total_particles().max(1) as f64
    }

    pub fn node_visits_per_particle(&self) -> f64 {
        self.node_visits as f64 / self.total_particles().max(1) as f64
    }

    pub fn total_particles(&self) -> usize {
        self.batch_size * self.particle_count
    }
}

pub fn analyze_spatial_strategy(
    positions: &[[f32; 4]],
    batch_size: usize,
    particle_count: usize,
    cfg: &HashGridConfig,
    strategy: SpatialStrategyKind,
) -> KernelResult<SpatialStrategyReport> {
    cfg.validate()?;
    let expected = batch_size * particle_count;
    if positions.len() != expected {
        return Err(KernelError::PositionShape {
            positions: positions.len(),
            expected,
        });
    }

    match strategy {
        SpatialStrategyKind::HashGrid => {
            analyze_hashgrid_strategy(positions, batch_size, particle_count, cfg, strategy)
        }
        SpatialStrategyKind::TileBlocks { tile_size } => analyze_tile_strategy(
            positions,
            batch_size,
            particle_count,
            cfg,
            tile_size,
            strategy,
        ),
        SpatialStrategyKind::Bvh { leaf_size } => analyze_bvh_strategy(
            positions,
            batch_size,
            particle_count,
            cfg,
            leaf_size,
            strategy,
        ),
    }
}

fn analyze_hashgrid_strategy(
    positions: &[[f32; 4]],
    batch_size: usize,
    particle_count: usize,
    cfg: &HashGridConfig,
    strategy: SpatialStrategyKind,
) -> KernelResult<SpatialStrategyReport> {
    let snapshot = build_hashgrid(positions, batch_size, particle_count, cfg)?;
    let (active_bins, max_bin_occupancy) = occupancy_stats(&snapshot.bin_offsets);
    let mut report = base_report(strategy, batch_size, particle_count, cfg);
    report.cell_count = snapshot.cell_count;
    report.active_bins = active_bins;
    report.max_bin_occupancy = max_bin_occupancy;

    let eps2 = cfg.eps * cfg.eps;
    for idx in 0..positions.len() {
        let batch = idx / particle_count;
        let batch_base = batch * particle_count;
        let pi = positions[idx];
        let center = cell_coords_for_position(&pi, cfg);
        for_each_neighbor_cell(cfg.dim, |offset| {
            let coords = [
                center[0] + offset[0],
                center[1] + offset[1],
                center[2] + offset[2],
            ];
            let Some(cell) = cell_index_from_coords(coords, cfg) else {
                return;
            };
            let bin = batch * snapshot.cell_count + cell;
            for binned in snapshot.bin_offsets[bin]..snapshot.bin_offsets[bin + 1] {
                report.candidate_entries_visited += 1;
                let j = batch_base + snapshot.permutation[binned];
                if cfg.mode == HashGridMode::Particle
                    && cell_coords_for_position(&positions[j], cfg) != coords
                {
                    continue;
                }
                report.candidate_tests += 1;
                let delta = neighbor_delta(&pi, &positions[j], cfg);
                let r2 = delta[..cfg.dim].iter().map(|v| v * v).sum::<f32>();
                if r2 < eps2 {
                    report.exact_neighbor_pairs += 1;
                }
            }
        });
    }
    Ok(report)
}

fn analyze_tile_strategy(
    positions: &[[f32; 4]],
    batch_size: usize,
    particle_count: usize,
    cfg: &HashGridConfig,
    tile_size: [usize; 3],
    strategy: SpatialStrategyKind,
) -> KernelResult<SpatialStrategyReport> {
    if cfg.mode == HashGridMode::Particle {
        return Err(KernelError::UnsupportedSpatialStrategy {
            strategy: "tile-blocks",
            reason: "particle-hash grids do not have bounded geometric tile coordinates",
        });
    }
    let tiles = TileGridConfig::from_hashgrid(cfg, tile_size);
    let assignment = assign_tiles(positions, batch_size, particle_count, cfg, &tiles)?;
    let (active_bins, max_bin_occupancy) = occupancy_stats(&assignment.tile_offsets);
    let mut report = base_report(strategy, batch_size, particle_count, cfg);
    report.cell_count = tiles.tile_count();
    report.active_bins = active_bins;
    report.max_bin_occupancy = max_bin_occupancy;
    let eps2 = cfg.eps * cfg.eps;

    for idx in 0..positions.len() {
        let batch = idx / particle_count;
        let batch_base = batch * particle_count;
        let pi = positions[idx];
        let center = cell_coords_for_position(&pi, cfg);
        let mut neighbor_tiles = Vec::with_capacity(if cfg.dim == 2 { 9 } else { 27 });
        for_each_neighbor_cell(cfg.dim, |offset| {
            let coords = [
                center[0] + offset[0],
                center[1] + offset[1],
                center[2] + offset[2],
            ];
            if let Some(tile) = tiles.tile_index_for_cell(coords)
                && !neighbor_tiles.contains(&tile)
            {
                neighbor_tiles.push(tile);
            }
        });
        for tile in neighbor_tiles {
            let bin = batch * assignment.tile_count + tile;
            for binned in assignment.tile_offsets[bin]..assignment.tile_offsets[bin + 1] {
                report.candidate_entries_visited += 1;
                report.candidate_tests += 1;
                let j = batch_base + assignment.permutation[binned];
                let delta = neighbor_delta(&pi, &positions[j], cfg);
                let r2 = delta[..cfg.dim].iter().map(|v| v * v).sum::<f32>();
                if r2 < eps2 {
                    report.exact_neighbor_pairs += 1;
                }
            }
        }
    }
    Ok(report)
}

fn analyze_bvh_strategy(
    positions: &[[f32; 4]],
    batch_size: usize,
    particle_count: usize,
    cfg: &HashGridConfig,
    leaf_size: usize,
    strategy: SpatialStrategyKind,
) -> KernelResult<SpatialStrategyReport> {
    if leaf_size == 0 {
        return Err(KernelError::InvalidBvhLeafSize);
    }
    if cfg.boundary == Boundary::Periodic {
        return Err(KernelError::UnsupportedSpatialStrategy {
            strategy: "bvh",
            reason: "periodic wraparound requires replicated query images",
        });
    }
    let mut report = base_report(strategy, batch_size, particle_count, cfg);
    let eps2 = cfg.eps * cfg.eps;

    for batch in 0..batch_size {
        let batch_base = batch * particle_count;
        let batch_positions = &positions[batch_base..batch_base + particle_count];
        let bvh = Bvh::build(batch_positions, cfg.dim, leaf_size)?;
        report.node_count += bvh.nodes.len();
        report.max_depth = report.max_depth.max(bvh.max_depth);
        for local_idx in 0..particle_count {
            let pi = batch_positions[local_idx];
            bvh.query_radius(pi, cfg.eps, |candidate, node_visits, leaf_visits| {
                report.node_visits += node_visits;
                report.leaf_visits += leaf_visits;
                for j in candidate {
                    report.candidate_entries_visited += 1;
                    report.candidate_tests += 1;
                    let delta = neighbor_delta(&pi, &batch_positions[*j], cfg);
                    let r2 = delta[..cfg.dim].iter().map(|v| v * v).sum::<f32>();
                    if r2 < eps2 {
                        report.exact_neighbor_pairs += 1;
                    }
                }
            });
        }
    }
    report.active_bins = report.node_count;
    report.max_bin_occupancy = leaf_size;
    Ok(report)
}

fn base_report(
    strategy: SpatialStrategyKind,
    batch_size: usize,
    particle_count: usize,
    cfg: &HashGridConfig,
) -> SpatialStrategyReport {
    SpatialStrategyReport {
        strategy,
        batch_size,
        particle_count,
        dim: cfg.dim,
        eps: cfg.eps,
        cell_count: cfg.cell_count(),
        active_bins: 0,
        max_bin_occupancy: 0,
        candidate_entries_visited: 0,
        candidate_tests: 0,
        exact_neighbor_pairs: 0,
        node_visits: 0,
        leaf_visits: 0,
        node_count: 0,
        max_depth: 0,
    }
}

fn occupancy_stats(offsets: &[usize]) -> (usize, usize) {
    offsets
        .windows(2)
        .map(|window| window[1] - window[0])
        .fold((0usize, 0usize), |(active, max), count| {
            (active + usize::from(count > 0), max.max(count))
        })
}

fn for_each_neighbor_cell(mut f_dim: usize, mut f: impl FnMut([isize; 3])) {
    if f_dim != 3 {
        f_dim = 2;
    }
    let z_min = if f_dim == 3 { -1 } else { 0 };
    let z_max = if f_dim == 3 { 1 } else { 0 };
    for dz in z_min..=z_max {
        for dy in -1..=1 {
            for dx in -1..=1 {
                f([dx, dy, dz]);
            }
        }
    }
}

#[derive(Clone, Debug)]
struct Bvh {
    indices: Vec<usize>,
    nodes: Vec<BvhNode>,
    root: usize,
    max_depth: usize,
    dim: usize,
}

#[derive(Clone, Debug)]
struct BvhNode {
    min: [f32; 3],
    max: [f32; 3],
    left: Option<usize>,
    right: Option<usize>,
    start: usize,
    end: usize,
}

impl Bvh {
    fn build(positions: &[[f32; 4]], dim: usize, leaf_size: usize) -> KernelResult<Self> {
        if leaf_size == 0 {
            return Err(KernelError::InvalidBvhLeafSize);
        }
        let mut bvh = Self {
            indices: (0..positions.len()).collect(),
            nodes: Vec::with_capacity(positions.len().saturating_mul(2).max(1)),
            root: 0,
            max_depth: 0,
            dim,
        };
        bvh.root = bvh.build_range(positions, 0, positions.len(), leaf_size, 0);
        Ok(bvh)
    }

    fn build_range(
        &mut self,
        positions: &[[f32; 4]],
        start: usize,
        end: usize,
        leaf_size: usize,
        depth: usize,
    ) -> usize {
        self.max_depth = self.max_depth.max(depth);
        let (min, max) = bounds_for_indices(positions, &self.indices[start..end], self.dim);
        let node_idx = self.nodes.len();
        self.nodes.push(BvhNode {
            min,
            max,
            left: None,
            right: None,
            start,
            end,
        });
        if end.saturating_sub(start) <= leaf_size {
            return node_idx;
        }

        let axis = largest_extent_axis(min, max, self.dim);
        self.indices[start..end]
            .sort_by(|lhs, rhs| positions[*lhs][axis].total_cmp(&positions[*rhs][axis]));
        let mid = (start + end) / 2;
        let left = self.build_range(positions, start, mid, leaf_size, depth + 1);
        let right = self.build_range(positions, mid, end, leaf_size, depth + 1);
        self.nodes[node_idx].left = Some(left);
        self.nodes[node_idx].right = Some(right);
        node_idx
    }

    fn query_radius(
        &self,
        center: [f32; 4],
        radius: f32,
        mut emit_leaf: impl FnMut(&[usize], usize, usize),
    ) {
        let mut stack = vec![self.root];
        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx];
            if !sphere_intersects_aabb(center, radius, node.min, node.max, self.dim) {
                emit_leaf(&[], 1, 0);
                continue;
            }
            if let (Some(left), Some(right)) = (node.left, node.right) {
                emit_leaf(&[], 1, 0);
                stack.push(left);
                stack.push(right);
            } else {
                emit_leaf(&self.indices[node.start..node.end], 1, 1);
            }
        }
    }
}

fn bounds_for_indices(
    positions: &[[f32; 4]],
    indices: &[usize],
    dim: usize,
) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for idx in indices {
        for axis in 0..dim {
            min[axis] = min[axis].min(positions[*idx][axis]);
            max[axis] = max[axis].max(positions[*idx][axis]);
        }
    }
    for axis in dim..3 {
        min[axis] = 0.0;
        max[axis] = 0.0;
    }
    (min, max)
}

fn largest_extent_axis(min: [f32; 3], max: [f32; 3], dim: usize) -> usize {
    let mut axis = 0;
    let mut extent = max[0] - min[0];
    for candidate in 1..dim {
        let candidate_extent = max[candidate] - min[candidate];
        if candidate_extent > extent {
            extent = candidate_extent;
            axis = candidate;
        }
    }
    axis
}

fn sphere_intersects_aabb(
    center: [f32; 4],
    radius: f32,
    min: [f32; 3],
    max: [f32; 3],
    dim: usize,
) -> bool {
    let mut dist2 = 0.0;
    for axis in 0..dim {
        let value = center[axis];
        if value < min[axis] {
            let d = min[axis] - value;
            dist2 += d * d;
        } else if value > max[axis] {
            let d = value - max[axis];
            dist2 += d * d;
        }
    }
    dist2 <= radius * radius
}
