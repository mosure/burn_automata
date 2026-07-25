use std::{cmp::Ordering, collections::BinaryHeap};

use serde::{Deserialize, Serialize};

use super::{AdaptiveParticleSet, material_footprint_radius, state::fit_state_jacobian};
use crate::{AutomataError, AutomataResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptiveHierarchyMember {
    Leaf(usize),
    Proxy(usize),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveProxyNode {
    pub level: usize,
    pub children: Vec<AdaptiveHierarchyMember>,
    pub leaf_start: usize,
    pub leaf_end: usize,
    pub represented_measure: f32,
    pub position: [f32; 4],
    pub covariance: [f32; 9],
    pub state: Vec<f32>,
    pub mean_bandwidth: f32,
    pub max_bandwidth: f32,
    pub bounding_radius: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveProxyHierarchy {
    pub spatial_dims: usize,
    pub state_dims: usize,
    pub branch_factor: usize,
    pub leaf_order: Vec<usize>,
    leaf_rank: Vec<usize>,
    pub nodes: Vec<AdaptiveProxyNode>,
    pub levels: Vec<Vec<usize>>,
    pub roots: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveMaterialView {
    pub particles: AdaptiveParticleSet,
    pub members: Vec<AdaptiveHierarchyMember>,
    pub fine_to_material: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
struct SplitCandidate {
    score: f32,
    tie_break: usize,
    member: AdaptiveHierarchyMember,
}

impl PartialEq for SplitCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits() && self.tie_break == other.tie_break
    }
}

impl Eq for SplitCandidate {}

impl PartialOrd for SplitCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SplitCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.tie_break.cmp(&self.tie_break))
    }
}

impl AdaptiveProxyHierarchy {
    pub fn build(particles: &AdaptiveParticleSet, branch_factor: usize) -> AutomataResult<Self> {
        particles.validate()?;
        if branch_factor < 2 {
            return Err(AutomataError::InvalidArgument(
                "adaptive hierarchy branch factor must be at least two".to_string(),
            ));
        }
        let leaf_order = morton_order(particles);
        Self::build_with_leaf_order(particles, branch_factor, leaf_order)
    }

    pub(crate) fn build_with_leaf_order(
        particles: &AdaptiveParticleSet,
        branch_factor: usize,
        leaf_order: Vec<usize>,
    ) -> AutomataResult<Self> {
        particles.validate()?;
        if branch_factor < 2 {
            return Err(AutomataError::InvalidArgument(
                "adaptive hierarchy branch factor must be at least two".to_string(),
            ));
        }
        let mut permutation = leaf_order.clone();
        permutation.sort_unstable();
        if permutation != (0..particles.len()).collect::<Vec<_>>() {
            return Err(AutomataError::InvalidArgument(
                "adaptive hierarchy leaf order must be a complete permutation".to_owned(),
            ));
        }
        let mut leaf_rank = vec![0; leaf_order.len()];
        for (rank, leaf) in leaf_order.iter().copied().enumerate() {
            leaf_rank[leaf] = rank;
        }
        let mut hierarchy = Self {
            spatial_dims: particles.spatial_dims,
            state_dims: particles.state_dims,
            branch_factor,
            leaf_order,
            leaf_rank,
            nodes: Vec::new(),
            levels: Vec::new(),
            roots: Vec::new(),
        };

        let mut members = hierarchy
            .leaf_order
            .iter()
            .copied()
            .map(AdaptiveHierarchyMember::Leaf)
            .collect::<Vec<_>>();
        let mut level = 1;
        while members.len() > 1 {
            let mut next = Vec::with_capacity(members.len().div_ceil(branch_factor));
            let mut level_nodes = Vec::with_capacity(next.capacity());
            for children in members.chunks(branch_factor) {
                let node = hierarchy.aggregate_node(particles, level, children)?;
                let index = hierarchy.nodes.len();
                hierarchy.nodes.push(node);
                level_nodes.push(index);
                next.push(AdaptiveHierarchyMember::Proxy(index));
            }
            hierarchy.levels.push(level_nodes);
            members = next;
            level += 1;
        }
        hierarchy.roots = members
            .into_iter()
            .filter_map(|member| match member {
                AdaptiveHierarchyMember::Proxy(index) => Some(index),
                AdaptiveHierarchyMember::Leaf(_) => None,
            })
            .collect();
        hierarchy.validate(particles)?;
        Ok(hierarchy)
    }

    pub fn validate(&self, particles: &AdaptiveParticleSet) -> AutomataResult<()> {
        if self.spatial_dims != particles.spatial_dims
            || self.state_dims != particles.state_dims
            || self.leaf_order.len() != particles.len()
            || self.leaf_rank.len() != particles.len()
            || self.branch_factor < 2
        {
            return Err(AutomataError::InvalidModel(
                "adaptive proxy hierarchy shape mismatch".to_string(),
            ));
        }
        let mut order = self.leaf_order.clone();
        order.sort_unstable();
        if order != (0..particles.len()).collect::<Vec<_>>() {
            return Err(AutomataError::InvalidModel(
                "adaptive hierarchy leaf order is not a permutation".to_string(),
            ));
        }
        if self
            .leaf_order
            .iter()
            .copied()
            .enumerate()
            .any(|(rank, leaf)| self.leaf_rank[leaf] != rank)
        {
            return Err(AutomataError::InvalidModel(
                "adaptive hierarchy inverse leaf order is inconsistent".to_string(),
            ));
        }
        for (index, node) in self.nodes.iter().enumerate() {
            if node.children.is_empty()
                || node.leaf_start >= node.leaf_end
                || node.leaf_end > particles.len()
                || node.state.len() != self.state_dims
                || !node.represented_measure.is_finite()
                || node.represented_measure <= 0.0
                || !node.bounding_radius.is_finite()
                || node.bounding_radius < 0.0
            {
                return Err(AutomataError::InvalidModel(format!(
                    "adaptive proxy node {index} is malformed"
                )));
            }
        }
        Ok(())
    }

    /// Selects a conservative material cut. Proxy nodes selected by this cut
    /// become material leaves; all unselected hierarchy nodes remain cache-only.
    pub fn material_cut(
        &self,
        fine: &AdaptiveParticleSet,
        target_leaves: usize,
        detail_values: &[f32],
        detail_dims: usize,
    ) -> AutomataResult<AdaptiveMaterialView> {
        self.validate(fine)?;
        if target_leaves == 0 || target_leaves > fine.len() || detail_dims == 0 {
            return Err(AutomataError::InvalidArgument(
                "invalid adaptive hierarchy material-cut shape".to_string(),
            ));
        }
        if detail_values.len() != fine.len() * detail_dims {
            return Err(AutomataError::InvalidArgument(format!(
                "adaptive hierarchy detail len {} != {}",
                detail_values.len(),
                fine.len() * detail_dims
            )));
        }

        let mut selected = if self.roots.is_empty() {
            vec![AdaptiveHierarchyMember::Leaf(self.leaf_order[0])]
        } else {
            self.roots
                .iter()
                .copied()
                .map(AdaptiveHierarchyMember::Proxy)
                .collect::<Vec<_>>()
        };
        let mut candidates = BinaryHeap::new();
        for member in &selected {
            if let Some(candidate) = self.split_candidate(fine, *member, detail_values, detail_dims)
            {
                candidates.push(candidate);
            }
        }

        while selected.len() < target_leaves {
            let Some(candidate) = candidates.pop() else {
                break;
            };
            let Some(selected_index) = selected.iter().position(|item| *item == candidate.member)
            else {
                continue;
            };
            let children = self.children(candidate.member);
            if children.is_empty()
                || selected.len() + children.len().saturating_sub(1) > target_leaves
            {
                continue;
            }
            selected.swap_remove(selected_index);
            for child in children {
                selected.push(child);
                if let Some(next) = self.split_candidate(fine, child, detail_values, detail_dims) {
                    candidates.push(next);
                }
            }
        }
        selected.sort_unstable_by_key(|member| self.member_leaf_range(*member).0);
        self.material_view(fine, selected)
    }

    /// Returns every node from one uniform hierarchy level as material. Level
    /// zero is the first restriction above leaves, so a complete canonical
    /// split restores the original leaf count and equal-measure level.
    pub fn material_level_cut(
        &self,
        fine: &AdaptiveParticleSet,
        level: usize,
    ) -> AutomataResult<AdaptiveMaterialView> {
        self.validate(fine)?;
        let nodes = self.levels.get(level).ok_or_else(|| {
            AutomataError::InvalidArgument(format!(
                "adaptive hierarchy level {level} is outside 0..{}",
                self.levels.len()
            ))
        })?;
        self.material_view(
            fine,
            nodes
                .iter()
                .copied()
                .map(AdaptiveHierarchyMember::Proxy)
                .collect(),
        )
    }

    /// Restricts a fine population by replacing the lowest-cost first-level
    /// sibling groups with their conservative parent. This is the exact
    /// 4-to-1 budget family used by the 2D 1,024..=4,096 leaf experiments.
    pub(crate) fn material_cut_from_level_one_merge_costs(
        &self,
        fine: &AdaptiveParticleSet,
        target_leaves: usize,
        merge_costs: &[f32],
    ) -> AutomataResult<AdaptiveMaterialView> {
        self.validate(fine)?;
        let level = self.levels.first().ok_or_else(|| {
            AutomataError::InvalidArgument(
                "adaptive hierarchy has no first-level merge groups".to_string(),
            )
        })?;
        if merge_costs.len() != level.len()
            || merge_costs.iter().any(|value| !value.is_finite())
            || target_leaves < level.len()
            || target_leaves > fine.len()
        {
            return Err(AutomataError::InvalidArgument(format!(
                "level-one merge costs require {} finite scores and a target in {}..={}, got {} scores and target {target_leaves}",
                level.len(),
                level.len(),
                fine.len(),
                merge_costs.len(),
            )));
        }
        let merge_mask = self.level_one_merge_mask(fine, target_leaves, merge_costs)?;
        let mut members = Vec::with_capacity(target_leaves);
        for (group, node) in level.iter().copied().enumerate() {
            if merge_mask[group] {
                members.push(AdaptiveHierarchyMember::Proxy(node));
            } else {
                members.extend(self.nodes[node].children.iter().copied());
            }
        }
        members.sort_unstable_by_key(|member| self.member_leaf_range(*member).0);
        if members.len() != target_leaves {
            return Err(AutomataError::InvalidModel(format!(
                "level-one merge cut produced {} leaves instead of {target_leaves}",
                members.len(),
            )));
        }
        self.material_view(fine, members)
    }

    pub(crate) fn level_one_merge_mask(
        &self,
        fine: &AdaptiveParticleSet,
        target_leaves: usize,
        merge_costs: &[f32],
    ) -> AutomataResult<Vec<bool>> {
        self.validate(fine)?;
        let level = self.levels.first().ok_or_else(|| {
            AutomataError::InvalidArgument(
                "adaptive hierarchy has no first-level merge groups".to_string(),
            )
        })?;
        if merge_costs.len() != level.len() || merge_costs.iter().any(|value| !value.is_finite()) {
            return Err(AutomataError::InvalidArgument(format!(
                "level-one merge selection requires {} finite costs, got {}",
                level.len(),
                merge_costs.len(),
            )));
        }
        let count_reduction = fine.len().checked_sub(target_leaves).ok_or_else(|| {
            AutomataError::InvalidArgument(format!(
                "target {target_leaves} exceeds fine count {}",
                fine.len(),
            ))
        })?;
        let reduction_per_merge = self.branch_factor - 1;
        if !count_reduction.is_multiple_of(reduction_per_merge) {
            return Err(AutomataError::InvalidArgument(format!(
                "target {target_leaves} is not reachable from {} leaves by conservative {}-to-1 merges",
                fine.len(),
                self.branch_factor,
            )));
        }
        let merge_count = count_reduction / reduction_per_merge;
        if merge_count > level.len()
            || level.iter().any(|node| {
                self.nodes[*node].children.len() != self.branch_factor
                    || self.nodes[*node]
                        .children
                        .iter()
                        .any(|member| !matches!(member, AdaptiveHierarchyMember::Leaf(_)))
            })
        {
            return Err(AutomataError::InvalidArgument(
                "requested budget is not representable by complete first-level sibling merges"
                    .to_string(),
            ));
        }

        let mut ranked = level.iter().copied().enumerate().collect::<Vec<_>>();
        ranked.sort_unstable_by(|(lhs_score, lhs_node), (rhs_score, rhs_node)| {
            merge_costs[*lhs_score]
                .total_cmp(&merge_costs[*rhs_score])
                .then_with(|| {
                    self.nodes[*lhs_node]
                        .leaf_start
                        .cmp(&self.nodes[*rhs_node].leaf_start)
                })
        });
        let mut mask = vec![false; level.len()];
        for (group, _) in ranked.into_iter().take(merge_count) {
            mask[group] = true;
        }
        Ok(mask)
    }

    pub fn restrict_values(
        &self,
        fine: &AdaptiveParticleSet,
        members: &[AdaptiveHierarchyMember],
        values: &[f32],
        value_dims: usize,
    ) -> AutomataResult<Vec<f32>> {
        if value_dims == 0 || values.len() != fine.len() * value_dims {
            return Err(AutomataError::InvalidArgument(
                "adaptive hierarchy restriction shape mismatch".to_string(),
            ));
        }
        let mut restricted = Vec::with_capacity(members.len() * value_dims);
        for member in members {
            let leaves = self.member_leaf_indices(*member);
            let total = leaves
                .iter()
                .map(|index| fine.represented_measure[*index])
                .sum::<f32>()
                .max(f32::MIN_POSITIVE);
            for channel in 0..value_dims {
                let value = leaves
                    .iter()
                    .map(|index| {
                        fine.represented_measure[*index] * values[*index * value_dims + channel]
                    })
                    .sum::<f32>()
                    / total;
                restricted.push(value);
            }
        }
        Ok(restricted)
    }

    pub fn member_leaf_indices(&self, member: AdaptiveHierarchyMember) -> &[usize] {
        let (start, end) = self.member_leaf_range(member);
        &self.leaf_order[start..end]
    }

    fn material_view(
        &self,
        fine: &AdaptiveParticleSet,
        members: Vec<AdaptiveHierarchyMember>,
    ) -> AutomataResult<AdaptiveMaterialView> {
        let mut positions = Vec::with_capacity(members.len());
        let mut states = Vec::with_capacity(members.len() * fine.state_dims);
        let mut represented_measure = Vec::with_capacity(members.len());
        let mut state_jacobian =
            Vec::with_capacity(members.len() * fine.state_dims * fine.spatial_dims);
        let mut closure_mode = Vec::with_capacity(members.len() * fine.state_dims);
        let mut closure_basis = Vec::with_capacity(members.len() * 4);
        let mut closure_phase = Vec::with_capacity(members.len() * 2);
        let mut bandwidth = Vec::with_capacity(members.len());
        let mut covariance = Vec::with_capacity(members.len());
        let mut particle_id = Vec::with_capacity(members.len());
        let mut generation = Vec::with_capacity(members.len());
        let mut fine_to_material = vec![usize::MAX; fine.len()];
        for (material_index, member) in members.iter().copied().enumerate() {
            let (position, state, measure, mean_bandwidth, cov, level, id) = match member {
                AdaptiveHierarchyMember::Leaf(index) => (
                    fine.positions[index],
                    fine.states[index * fine.state_dims..(index + 1) * fine.state_dims].to_vec(),
                    fine.represented_measure[index],
                    fine.bandwidth[index],
                    fine.covariance[index],
                    0,
                    fine.particle_id[index],
                ),
                AdaptiveHierarchyMember::Proxy(index) => {
                    let node = &self.nodes[index];
                    (
                        node.position,
                        node.state.clone(),
                        node.represented_measure,
                        node.mean_bandwidth.max(material_footprint_radius(
                            node.represented_measure,
                            fine.spatial_dims,
                        )),
                        node.covariance,
                        node.level,
                        (1_u64 << 63) | index as u64,
                    )
                }
            };
            positions.push(position);
            states.extend(state);
            represented_measure.push(measure);
            bandwidth.push(mean_bandwidth);
            covariance.push(cov);
            let member_jacobian = match member {
                AdaptiveHierarchyMember::Leaf(index) => {
                    fine.state_jacobian[index * fine.state_dims * fine.spatial_dims
                        ..(index + 1) * fine.state_dims * fine.spatial_dims]
                        .to_vec()
                }
                AdaptiveHierarchyMember::Proxy(_) => fit_state_jacobian(
                    fine,
                    self.member_leaf_indices(member),
                    &states
                        [material_index * fine.state_dims..(material_index + 1) * fine.state_dims],
                    position,
                    cov,
                    measure,
                )?,
            };
            state_jacobian.extend(member_jacobian);
            match member {
                AdaptiveHierarchyMember::Leaf(index) if !fine.closure_mode.is_empty() => {
                    closure_mode.extend_from_slice(
                        &fine.closure_mode[index * fine.state_dims..(index + 1) * fine.state_dims],
                    );
                }
                _ => closure_mode.extend(std::iter::repeat_n(0.0, fine.state_dims)),
            }
            match member {
                AdaptiveHierarchyMember::Leaf(index) if !fine.closure_basis.is_empty() => {
                    closure_basis
                        .extend_from_slice(&fine.closure_basis[index * 4..(index + 1) * 4]);
                }
                _ => closure_basis.extend(std::iter::repeat_n(0.0, 4)),
            }
            match member {
                AdaptiveHierarchyMember::Leaf(index) if !fine.closure_phase.is_empty() => {
                    closure_phase
                        .extend_from_slice(&fine.closure_phase[index * 2..(index + 1) * 2]);
                }
                _ => closure_phase.extend(std::iter::repeat_n(0.0, 2)),
            }
            particle_id.push(id);
            generation.push(level.min(u16::MAX as usize) as u16);
            for leaf in self.member_leaf_indices(member) {
                fine_to_material[*leaf] = material_index;
            }
        }
        if fine_to_material.contains(&usize::MAX) {
            return Err(AutomataError::InvalidModel(
                "adaptive material cut does not cover every fine leaf".to_string(),
            ));
        }
        let render_footprint = represented_measure
            .iter()
            .map(|measure| material_footprint_radius(*measure, fine.spatial_dims))
            .collect();
        let particles = AdaptiveParticleSet {
            spatial_dims: fine.spatial_dims,
            state_dims: fine.state_dims,
            positions,
            states,
            state_jacobian,
            closure_mode,
            closure_basis,
            closure_phase,
            represented_measure,
            render_footprint,
            bandwidth,
            covariance,
            particle_id,
            sibling_group: vec![0; members.len()],
            generation,
            cooldown: vec![0; members.len()],
            next_id: fine.next_id.max((1_u64 << 63) | self.nodes.len() as u64),
            next_sibling_group: fine.next_sibling_group,
            bootstrap_templates: Vec::new(),
        };
        particles.validate()?;
        Ok(AdaptiveMaterialView {
            particles,
            members,
            fine_to_material,
        })
    }

    fn split_candidate(
        &self,
        fine: &AdaptiveParticleSet,
        member: AdaptiveHierarchyMember,
        detail_values: &[f32],
        detail_dims: usize,
    ) -> Option<SplitCandidate> {
        let AdaptiveHierarchyMember::Proxy(index) = member else {
            return None;
        };
        let leaves = self.member_leaf_indices(member);
        if leaves.len() <= 1 {
            return None;
        }
        let total = leaves
            .iter()
            .map(|leaf| fine.represented_measure[*leaf])
            .sum::<f32>()
            .max(f32::MIN_POSITIVE);
        let mut mean = vec![0.0; detail_dims];
        for leaf in leaves {
            let weight = fine.represented_measure[*leaf] / total;
            for channel in 0..detail_dims {
                mean[channel] += weight * detail_values[*leaf * detail_dims + channel];
            }
        }
        let mut variance = 0.0;
        for leaf in leaves {
            let weight = fine.represented_measure[*leaf] / total;
            for channel in 0..detail_dims {
                let delta = detail_values[*leaf * detail_dims + channel] - mean[channel];
                variance += weight * delta * delta;
            }
        }
        Some(SplitCandidate {
            score: variance * total + self.nodes[index].bounding_radius * 1.0e-9,
            tie_break: self.nodes[index].leaf_start,
            member,
        })
    }

    fn children(&self, member: AdaptiveHierarchyMember) -> Vec<AdaptiveHierarchyMember> {
        match member {
            AdaptiveHierarchyMember::Leaf(_) => Vec::new(),
            AdaptiveHierarchyMember::Proxy(index) => self.nodes[index].children.clone(),
        }
    }

    fn member_leaf_range(&self, member: AdaptiveHierarchyMember) -> (usize, usize) {
        match member {
            AdaptiveHierarchyMember::Leaf(index) => {
                let ordered = self.leaf_rank[index];
                (ordered, ordered + 1)
            }
            AdaptiveHierarchyMember::Proxy(index) => {
                let node = &self.nodes[index];
                (node.leaf_start, node.leaf_end)
            }
        }
    }

    fn aggregate_node(
        &self,
        particles: &AdaptiveParticleSet,
        level: usize,
        children: &[AdaptiveHierarchyMember],
    ) -> AutomataResult<AdaptiveProxyNode> {
        let leaf_start = children
            .first()
            .map(|member| self.member_leaf_range(*member).0)
            .ok_or_else(|| AutomataError::InvalidModel("empty proxy node".to_string()))?;
        let leaf_end = self
            .member_leaf_range(*children.last().expect("nonempty children"))
            .1;
        let leaves = &self.leaf_order[leaf_start..leaf_end];
        let represented_measure = leaves
            .iter()
            .map(|index| particles.represented_measure[*index])
            .sum::<f32>();
        let mut position = [0.0; 4];
        let mut state = vec![0.0; particles.state_dims];
        let mut mean_bandwidth = 0.0;
        let mut max_bandwidth = 0.0_f32;
        for index in leaves {
            let weight = particles.represented_measure[*index] / represented_measure;
            for (axis, value) in position.iter_mut().enumerate().take(particles.spatial_dims) {
                *value += weight * particles.positions[*index][axis];
            }
            for (channel, value) in state.iter_mut().enumerate() {
                *value += weight * particles.states[*index * particles.state_dims + channel];
            }
            mean_bandwidth += weight * particles.bandwidth[*index];
            max_bandwidth = max_bandwidth.max(particles.bandwidth[*index]);
        }
        let mut covariance = [0.0; 9];
        let mut bounding_radius = 0.0_f32;
        for index in leaves {
            let weight = particles.represented_measure[*index] / represented_measure;
            let mut distance2 = 0.0;
            for row in 0..particles.spatial_dims {
                let row_delta = particles.positions[*index][row] - position[row];
                distance2 += row_delta * row_delta;
                for col in 0..particles.spatial_dims {
                    let col_delta = particles.positions[*index][col] - position[col];
                    covariance[row * 3 + col] += weight
                        * (particles.covariance[*index][row * 3 + col] + row_delta * col_delta);
                }
            }
            bounding_radius = bounding_radius.max(
                distance2.sqrt()
                    + material_footprint_radius(
                        particles.represented_measure[*index],
                        particles.spatial_dims,
                    ),
            );
        }
        Ok(AdaptiveProxyNode {
            level,
            children: children.to_vec(),
            leaf_start,
            leaf_end,
            represented_measure,
            position,
            covariance,
            state,
            mean_bandwidth,
            max_bandwidth,
            bounding_radius,
        })
    }
}

fn morton_order(particles: &AdaptiveParticleSet) -> Vec<usize> {
    let mut minimum = [f32::INFINITY; 3];
    let mut maximum = [f32::NEG_INFINITY; 3];
    for position in &particles.positions {
        for axis in 0..particles.spatial_dims {
            minimum[axis] = minimum[axis].min(position[axis]);
            maximum[axis] = maximum[axis].max(position[axis]);
        }
    }
    let mut order = (0..particles.len()).collect::<Vec<_>>();
    order.sort_unstable_by_key(|index| {
        let mut coordinate = [0_u32; 3];
        for axis in 0..particles.spatial_dims {
            let extent = (maximum[axis] - minimum[axis]).max(f32::MIN_POSITIVE);
            coordinate[axis] = (((particles.positions[*index][axis] - minimum[axis]) / extent)
                .clamp(0.0, 1.0)
                * 1023.0)
                .round() as u32;
        }
        morton_code(coordinate, particles.spatial_dims)
    });
    order
}

fn morton_code(coordinate: [u32; 3], dim: usize) -> u64 {
    let mut code = 0_u64;
    for bit in 0..10 {
        for (axis, value) in coordinate.iter().enumerate().take(dim) {
            code |= u64::from((value >> bit) & 1) << (bit * dim + axis);
        }
    }
    code
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive::unit_ball_measure;

    fn particles() -> AdaptiveParticleSet {
        let positions = (0..64)
            .map(|index| {
                let x = index % 8;
                let y = index / 8;
                [x as f32 * 0.04, y as f32 * 0.04, 0.0, 0.0]
            })
            .collect::<Vec<_>>();
        let states = positions
            .iter()
            .flat_map(|position| [position[0], position[1]])
            .collect::<Vec<_>>();
        AdaptiveParticleSet::from_equal_measure(
            positions,
            states,
            2,
            2,
            unit_ball_measure(2) * 0.2_f32.powi(2),
            0.1,
        )
        .unwrap()
    }

    #[test]
    fn proxy_root_preserves_measure_centroid_state_and_second_moment() {
        let particles = particles();
        let hierarchy = AdaptiveProxyHierarchy::build(&particles, 4).unwrap();
        let root = &hierarchy.nodes[hierarchy.roots[0]];
        assert!((root.represented_measure as f64 - particles.total_measure()).abs() < 1.0e-7);
        for axis in 0..2 {
            let mean = particles
                .positions
                .iter()
                .zip(&particles.represented_measure)
                .map(|(position, measure)| position[axis] * measure)
                .sum::<f32>()
                / particles.total_measure() as f32;
            assert!((root.position[axis] - mean).abs() < 1.0e-6);
            assert!((root.state[axis] - mean).abs() < 1.0e-6);
        }
        for row in 0..2 {
            for col in 0..2 {
                let root_second = root.represented_measure
                    * (root.covariance[row * 3 + col] + root.position[row] * root.position[col]);
                let fine_second = (0..particles.len())
                    .map(|index| {
                        particles.represented_measure[index]
                            * (particles.covariance[index][row * 3 + col]
                                + particles.positions[index][row] * particles.positions[index][col])
                    })
                    .sum::<f32>();
                assert!((root_second - fine_second).abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn material_cut_is_conservative_and_allocates_detail_nonuniformly() {
        let particles = particles();
        let hierarchy = AdaptiveProxyHierarchy::build(&particles, 4).unwrap();
        let detail = particles
            .positions
            .iter()
            .map(|position| {
                if position[0] < 0.12 {
                    position[1] * 50.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let cut = hierarchy.material_cut(&particles, 16, &detail, 1).unwrap();
        assert_eq!(cut.particles.len(), 16);
        assert!(
            (cut.particles.total_measure() - particles.total_measure()).abs()
                < particles.total_measure() * 1.0e-6
        );
        assert!(
            cut.members
                .iter()
                .any(|member| matches!(member, AdaptiveHierarchyMember::Leaf(_)))
        );
        assert!(
            cut.members
                .iter()
                .any(|member| matches!(member, AdaptiveHierarchyMember::Proxy(_)))
        );
        assert!(
            cut.fine_to_material
                .iter()
                .all(|index| *index < cut.particles.len())
        );
    }

    #[test]
    fn level_one_merge_costs_select_exact_reachable_budget() {
        let particles = particles();
        let hierarchy = AdaptiveProxyHierarchy::build(&particles, 4).unwrap();
        let first_level = &hierarchy.levels[0];
        let mut costs = vec![10.0; first_level.len()];
        costs[3] = -1.0;
        costs[7] = 0.0;

        let cut = hierarchy
            .material_cut_from_level_one_merge_costs(&particles, 58, &costs)
            .unwrap();

        assert_eq!(cut.particles.len(), 58);
        assert_eq!(
            cut.members
                .iter()
                .filter(|member| matches!(member, AdaptiveHierarchyMember::Proxy(_)))
                .count(),
            2
        );
        assert!(
            cut.members
                .contains(&AdaptiveHierarchyMember::Proxy(first_level[3]))
        );
        assert!(
            cut.members
                .contains(&AdaptiveHierarchyMember::Proxy(first_level[7]))
        );
        assert!(
            (cut.particles.total_measure() - particles.total_measure()).abs()
                < particles.total_measure() * 1.0e-6
        );
    }

    #[test]
    fn level_one_merge_costs_reject_unreachable_budget() {
        let particles = particles();
        let hierarchy = AdaptiveProxyHierarchy::build(&particles, 4).unwrap();
        let costs = vec![0.0; hierarchy.levels[0].len()];

        let error = hierarchy
            .material_cut_from_level_one_merge_costs(&particles, 63, &costs)
            .unwrap_err();

        assert!(error.to_string().contains("not reachable"));
    }

    #[test]
    fn material_restriction_retains_an_affine_state_jacobian() {
        let mut particles = particles();
        for row in particles.state_jacobian.chunks_exact_mut(4) {
            row.copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);
        }
        let hierarchy = AdaptiveProxyHierarchy::build(&particles, 4).unwrap();
        let cut = hierarchy.material_level_cut(&particles, 0).unwrap();
        for row in cut.particles.state_jacobian.chunks_exact(4) {
            for (actual, expected) in row.iter().zip([1.0, 0.0, 0.0, 1.0]) {
                assert!((actual - expected).abs() < 2.0e-5, "{row:?}");
            }
        }
    }
}
