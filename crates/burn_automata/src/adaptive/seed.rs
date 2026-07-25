use std::collections::BTreeMap;

use super::{
    AdaptiveBootstrapChild, AdaptiveBootstrapTemplate, AdaptiveHierarchyMember,
    AdaptiveHierarchyRestrictionPolicy, AdaptiveMaterialView, AdaptiveNpaModel,
    AdaptiveParticleSet, AdaptiveProxyHierarchy, AdaptiveRestrictionArity,
    AdaptiveRestrictionSchedule, dynamics::primary_rule_features, features::material_detail_values,
    material_footprint_radius, perception::rule_perception_pair,
    restriction::learned_level_one_merge_costs,
};
use crate::{AutomataError, AutomataResult, ParticleSeed, rollout::seed_particles_scaled};

#[allow(clippy::too_many_arguments)]
pub fn seed_adaptive_particles_scaled(
    model: &AdaptiveNpaModel,
    particle_count: usize,
    seed: u64,
    seed_mode: ParticleSeed,
    scale: f32,
    total_measure: f32,
    bandwidth: f32,
) -> AutomataResult<AdaptiveParticleSet> {
    model.validate()?;
    if particle_count == 0 || !scale.is_finite() || scale <= 0.0 {
        return Err(AutomataError::InvalidArgument(
            "adaptive seed count and scale must be positive".to_string(),
        ));
    }
    if matches!(
        model.config.material_seed_layout,
        super::AdaptiveMaterialSeedLayout::UniformContinuous
            | super::AdaptiveMaterialSeedLayout::GradedContinuous
    ) && particle_count == model.config.target_leaves
    {
        let reference = model.config.bootstrap_fine_leaf_count().max(particle_count);
        let material_ratio = reference as f32 / particle_count as f32;
        let uniform_bandwidth =
            bandwidth * material_ratio.powf(model.config.material_seed_bandwidth_exponent);
        let mut particles = continuous_uniform_seed_from_reference(
            model,
            particle_count,
            reference,
            seed,
            seed_mode,
            scale,
            total_measure,
            uniform_bandwidth,
        )?;
        if model.config.material_seed_layout == super::AdaptiveMaterialSeedLayout::GradedContinuous
        {
            let units = continuous_material_units(
                particle_count,
                reference,
                model.config.material_seed_measure_ratio,
            )?;
            let fine_measure = total_measure / reference as f32;
            let represented_measure = units
                .iter()
                .map(|units| *units * fine_measure)
                .collect::<Vec<_>>();
            let bandwidths = units
                .iter()
                .map(|units| bandwidth * units.powf(model.config.material_seed_bandwidth_exponent))
                .collect::<Vec<_>>();
            apply_continuous_material_layout(&mut particles, &represented_measure, &bandwidths)?;
        }
        return Ok(particles);
    }
    let hierarchical_bootstrap = model.config.hierarchical_bootstrap_seed
        && model.config.bootstrap_end_step > 0
        && particle_count == model.config.initial_leaf_count()
        && particle_count < model.config.bootstrap_target_leaf_count();
    let fine_leaf_count = model.config.bootstrap_fine_leaf_count();
    let hierarchical_target_cut = model.config.hierarchical_bootstrap_seed
        && particle_count == model.config.target_leaves
        && fine_leaf_count > particle_count;
    if !hierarchical_bootstrap && !hierarchical_target_cut {
        return equal_measure_seed(
            model,
            particle_count,
            seed,
            seed_mode,
            scale,
            total_measure,
            bandwidth,
        );
    }

    let fine = equal_measure_seed(
        model,
        fine_leaf_count,
        seed,
        seed_mode,
        scale,
        total_measure,
        bandwidth,
    )?;
    if hierarchical_target_cut {
        return restrict_adaptive_particles_to_target(model, &fine);
    }
    let children_per_split = 2 * model.config.spatial_dims;
    let hierarchy = AdaptiveProxyHierarchy::build(&fine, children_per_split)?;
    let level = hierarchy
        .levels
        .iter()
        .position(|nodes| nodes.len() == particle_count)
        .ok_or_else(|| {
            AutomataError::InvalidArgument(format!(
                "hierarchical bootstrap has no uniform {particle_count}-leaf level below {fine_leaf_count} fine leaves",
            ))
        })?;
    let view = hierarchy.material_level_cut(&fine, level)?;
    if view.particles.len() != particle_count {
        return Err(AutomataError::InvalidArgument(format!(
            "hierarchical seed produced {} leaves instead of {particle_count}; choose a reachable conservative {}-ary cut",
            view.particles.len(),
            children_per_split,
        )));
    }
    hierarchical_seed_from_view(model, &fine, &hierarchy, view)
}

pub(crate) fn restrict_adaptive_particles_to_target(
    model: &AdaptiveNpaModel,
    fine: &AdaptiveParticleSet,
) -> AutomataResult<AdaptiveParticleSet> {
    restrict_adaptive_particles_to_leaf_budget(model, fine, model.config.target_leaves)
}

pub(crate) fn restrict_adaptive_particles_to_leaf_budget(
    model: &AdaptiveNpaModel,
    fine: &AdaptiveParticleSet,
    target: usize,
) -> AutomataResult<AdaptiveParticleSet> {
    model.validate()?;
    fine.validate()?;
    let fine_leaf_count = model.config.bootstrap_fine_leaf_count();
    if target < model.config.target_leaves
        || target >= fine.len()
        || fine.len() != fine_leaf_count
        || !fine.bootstrap_templates.is_empty()
    {
        return Err(AutomataError::InvalidArgument(format!(
            "adaptive restriction requires {fine_leaf_count} untemplated fine leaves above reachable budget {target}, got {}",
            fine.len(),
        )));
    }
    let children_per_split = 2 * model.config.spatial_dims;
    let event_leaf_delta = children_per_split - 1;
    if model.config.hierarchical_restriction_arity == AdaptiveRestrictionArity::Canonical
        && !(fine.len() - target).is_multiple_of(event_leaf_delta)
    {
        return Err(AutomataError::InvalidArgument(format!(
            "adaptive restriction budget {target} is not reachable from {} leaves by {children_per_split}-child events",
            fine.len(),
        )));
    }
    let hierarchy = AdaptiveProxyHierarchy::build(fine, children_per_split)?;
    if model.config.hierarchical_restriction_arity == AdaptiveRestrictionArity::Mixed {
        return mixed_arity_restriction(model, fine, &hierarchy, target);
    }
    let view = if let Some(level) = hierarchy
        .levels
        .iter()
        .position(|nodes| nodes.len() == target)
    {
        hierarchy.material_level_cut(fine, level)?
    } else {
        match model.config.hierarchical_restriction_policy {
            AdaptiveHierarchyRestrictionPolicy::SpatialCompactness => {
                let level = hierarchy.levels.first().ok_or_else(|| {
                    AutomataError::InvalidModel(
                        "adaptive hierarchy has no first-level spatial groups".to_owned(),
                    )
                })?;
                if target >= level.len() {
                    let costs = level_one_merge_costs(model, fine, &hierarchy, level)?;
                    hierarchy.material_cut_from_level_one_merge_costs(fine, target, &costs)?
                } else {
                    let position_detail = fine
                        .positions
                        .iter()
                        .flat_map(|position| {
                            position
                                .iter()
                                .copied()
                                .take(fine.spatial_dims)
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    hierarchy.material_cut(fine, target, &position_detail, fine.spatial_dims)?
                }
            }
            AdaptiveHierarchyRestrictionPolicy::DynamicsDetail => {
                let perception = rule_perception_pair(&model.config, &model.rule, fine)?;
                let features =
                    primary_rule_features(model, fine, &perception.npa_compatible.features)?;
                let raw_update = model.rule.forward_update_from_features(features.as_ref())?;
                let detail = material_detail_values(
                    fine,
                    &raw_update,
                    model.rule.config.update_dims(),
                    model.config.base_rule_footprint().recip(),
                );
                hierarchy.material_cut(
                    fine,
                    target,
                    &detail,
                    model.rule.config.update_dims() + fine.state_dims + fine.spatial_dims,
                )?
            }
            AdaptiveHierarchyRestrictionPolicy::LearnedController => {
                let costs = learned_level_one_merge_costs(model, fine, &hierarchy)?;
                hierarchy.material_cut_from_level_one_merge_costs(fine, target, &costs)?
            }
        }
    };
    if view.particles.len() != target {
        return Err(AutomataError::InvalidArgument(format!(
            "hierarchical restriction produced {} leaves instead of {target}; choose a reachable conservative {children_per_split}-ary cut",
            view.particles.len(),
        )));
    }
    hierarchical_seed_from_view(model, fine, &hierarchy, view)
}

/// Extends an existing mixed-arity hierarchy cut without replacing material
/// groups selected by earlier schedule intervals.
///
/// Re-ranking a complete cut at every interval creates avoidable topology
/// churn even when the requested leaf budget changes monotonically. This path
/// locks existing aggregate child sets, ranks only untouched canonical groups,
/// and adds another deterministic 4/3/2-child tranche.
pub(crate) fn progressively_restrict_adaptive_particles_to_leaf_budget(
    model: &AdaptiveNpaModel,
    current: &AdaptiveParticleSet,
    fine: &AdaptiveParticleSet,
    target: usize,
) -> AutomataResult<AdaptiveParticleSet> {
    if model.config.hierarchical_restriction_arity != AdaptiveRestrictionArity::Mixed
        || model.config.hierarchical_restriction_schedule != AdaptiveRestrictionSchedule::Nested
        || current.bootstrap_templates.is_empty()
    {
        return restrict_adaptive_particles_to_leaf_budget(model, fine, target);
    }
    model.validate()?;
    current.validate()?;
    fine.validate()?;
    if target >= current.len() {
        return Err(AutomataError::InvalidArgument(format!(
            "progressive restriction target {target} must be below the current {} leaves",
            current.len(),
        )));
    }

    let fine_index_by_id = fine
        .particle_id
        .iter()
        .copied()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect::<BTreeMap<_, _>>();
    let mut merged_groups = Vec::with_capacity(current.bootstrap_templates.len());
    let mut consumed = vec![false; fine.len()];
    for template in &current.bootstrap_templates {
        let mut children = template
            .children
            .iter()
            .map(|child| {
                fine_index_by_id
                    .get(&child.particle_id)
                    .copied()
                    .ok_or_else(|| {
                        AutomataError::InvalidModel(format!(
                            "progressive restriction child {} is absent from the restored fine state",
                            child.particle_id,
                        ))
                    })
            })
            .collect::<AutomataResult<Vec<_>>>()?;
        children.sort_unstable();
        if !(2..=4).contains(&children.len())
            || children
                .iter()
                .any(|child| std::mem::replace(&mut consumed[*child], true))
        {
            return Err(AutomataError::InvalidModel(
                "progressive restriction aggregates must contain two to four disjoint fine children"
                    .to_owned(),
            ));
        }
        merged_groups.push(children);
    }

    let existing_reduction = merged_groups
        .iter()
        .map(|children| children.len().saturating_sub(1))
        .sum::<usize>();
    let expected_reduction = fine.len().saturating_sub(current.len());
    if existing_reduction != expected_reduction {
        return Err(AutomataError::InvalidModel(format!(
            "progressive restriction templates encode reduction {existing_reduction}, expected {expected_reduction}",
        )));
    }
    let target_reduction = fine.len().checked_sub(target).ok_or_else(|| {
        AutomataError::InvalidArgument(format!(
            "progressive restriction target {target} exceeds {} fine leaves",
            fine.len(),
        ))
    })?;
    let mut remaining_reduction = target_reduction
        .checked_sub(existing_reduction)
        .ok_or_else(|| {
            AutomataError::InvalidArgument(
                "progressive restriction cannot refine an existing mixed-arity cut".to_owned(),
            )
        })?;

    let available_indices = consumed
        .iter()
        .enumerate()
        .filter_map(|(index, consumed)| (!consumed).then_some(index))
        .collect::<Vec<_>>();
    let available = adaptive_particle_subset(fine, &available_indices)?;
    let children_per_split = 2 * model.config.spatial_dims;
    let hierarchy = AdaptiveProxyHierarchy::build(&available, children_per_split)?;
    let level = hierarchy.levels.first().ok_or_else(|| {
        AutomataError::InvalidArgument(
            "progressive mixed-arity restriction requires first-level hierarchy groups".to_owned(),
        )
    })?;
    let canonical_groups = level_one_leaf_groups(&hierarchy, level)?;
    let costs = level_one_merge_costs(model, &available, &hierarchy, level)?;
    let mut ranked = level.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_unstable_by(|(lhs_group, lhs_node), (rhs_group, rhs_node)| {
        costs[*lhs_group]
            .total_cmp(&costs[*rhs_group])
            .then_with(|| {
                hierarchy.nodes[*lhs_node]
                    .leaf_start
                    .cmp(&hierarchy.nodes[*rhs_node].leaf_start)
            })
    });

    const REDUCTION_PATTERN: [usize; 3] = [3, 2, 1];
    let existing_groups = merged_groups.len();
    for (rank, (group, _)) in ranked.into_iter().enumerate() {
        if remaining_reduction == 0 {
            break;
        }
        let children = canonical_groups[group]
            .iter()
            .map(|child| available_indices[*child])
            .collect::<Vec<_>>();
        let reduction = REDUCTION_PATTERN[(existing_groups + rank) % REDUCTION_PATTERN.len()]
            .min(children.len().saturating_sub(1))
            .min(remaining_reduction);
        let merged = best_merge_subset(model, fine, &children, reduction + 1);
        for child in &merged {
            if std::mem::replace(&mut consumed[*child], true) {
                return Err(AutomataError::InvalidModel(
                    "progressive restriction selected an already aggregated child".to_owned(),
                ));
            }
        }
        merged_groups.push(merged);
        remaining_reduction -= reduction;
    }
    if remaining_reduction != 0 {
        return Err(AutomataError::InvalidArgument(format!(
            "progressive mixed-arity hierarchy cannot remove the remaining {remaining_reduction} leaves for target {target}",
        )));
    }

    let mut groups = merged_groups;
    groups.extend(
        consumed
            .iter()
            .enumerate()
            .filter_map(|(child, consumed)| (!consumed).then_some(vec![child])),
    );
    groups.sort_unstable_by_key(|members| members.iter().copied().min().unwrap_or(usize::MAX));
    if groups.len() != target {
        return Err(AutomataError::InvalidModel(format!(
            "progressive mixed-arity restriction produced {} groups instead of {target}",
            groups.len(),
        )));
    }
    restricted_seed_from_fine_groups(fine, groups)
}

pub(crate) fn adaptive_particle_subset(
    particles: &AdaptiveParticleSet,
    rows: &[usize],
) -> AutomataResult<AdaptiveParticleSet> {
    if rows.is_empty() || rows.iter().any(|row| *row >= particles.len()) {
        return Err(AutomataError::InvalidArgument(
            "adaptive particle subset requires valid non-empty rows".to_owned(),
        ));
    }
    let jacobian_dims = particles.state_dims * particles.spatial_dims;
    let subset = AdaptiveParticleSet {
        spatial_dims: particles.spatial_dims,
        state_dims: particles.state_dims,
        positions: rows.iter().map(|row| particles.positions[*row]).collect(),
        states: rows
            .iter()
            .flat_map(|row| {
                particles.states[*row * particles.state_dims..(*row + 1) * particles.state_dims]
                    .iter()
                    .copied()
            })
            .collect(),
        state_jacobian: rows
            .iter()
            .flat_map(|row| {
                particles.state_jacobian[*row * jacobian_dims..(*row + 1) * jacobian_dims]
                    .iter()
                    .copied()
            })
            .collect(),
        closure_mode: rows
            .iter()
            .flat_map(|row| {
                if particles.closure_mode.is_empty() {
                    vec![0.0; particles.state_dims]
                } else {
                    particles.closure_mode
                        [*row * particles.state_dims..(*row + 1) * particles.state_dims]
                        .to_vec()
                }
            })
            .collect(),
        closure_basis: rows
            .iter()
            .flat_map(|row| {
                if particles.closure_basis.is_empty() {
                    vec![0.0; 4]
                } else {
                    particles.closure_basis[*row * 4..(*row + 1) * 4].to_vec()
                }
            })
            .collect(),
        closure_phase: rows
            .iter()
            .flat_map(|row| {
                if particles.closure_phase.is_empty() {
                    vec![0.0; 2]
                } else {
                    particles.closure_phase[*row * 2..(*row + 1) * 2].to_vec()
                }
            })
            .collect(),
        represented_measure: rows
            .iter()
            .map(|row| particles.represented_measure[*row])
            .collect(),
        render_footprint: rows
            .iter()
            .map(|row| particles.render_footprint[*row])
            .collect(),
        bandwidth: rows.iter().map(|row| particles.bandwidth[*row]).collect(),
        covariance: rows.iter().map(|row| particles.covariance[*row]).collect(),
        particle_id: rows.iter().map(|row| particles.particle_id[*row]).collect(),
        sibling_group: rows
            .iter()
            .map(|row| particles.sibling_group[*row])
            .collect(),
        generation: rows.iter().map(|row| particles.generation[*row]).collect(),
        cooldown: rows.iter().map(|row| particles.cooldown[*row]).collect(),
        next_id: particles.next_id,
        next_sibling_group: particles.next_sibling_group,
        bootstrap_templates: Vec::new(),
    };
    subset.validate()?;
    Ok(subset)
}

fn mixed_arity_restriction(
    model: &AdaptiveNpaModel,
    fine: &AdaptiveParticleSet,
    hierarchy: &AdaptiveProxyHierarchy,
    target: usize,
) -> AutomataResult<AdaptiveParticleSet> {
    let level = hierarchy.levels.first().ok_or_else(|| {
        AutomataError::InvalidArgument(
            "mixed-arity restriction requires first-level hierarchy groups".to_owned(),
        )
    })?;
    let costs = level_one_merge_costs(model, fine, hierarchy, level)?;
    mixed_arity_restriction_from_merge_costs(model, fine, hierarchy, target, &costs)
}

fn mixed_arity_restriction_from_merge_costs(
    model: &AdaptiveNpaModel,
    fine: &AdaptiveParticleSet,
    hierarchy: &AdaptiveProxyHierarchy,
    target: usize,
    costs: &[f32],
) -> AutomataResult<AdaptiveParticleSet> {
    let level = hierarchy.levels.first().ok_or_else(|| {
        AutomataError::InvalidArgument(
            "mixed-arity restriction requires first-level hierarchy groups".to_owned(),
        )
    })?;
    if costs.len() != level.len() || costs.iter().any(|cost| !cost.is_finite()) {
        return Err(AutomataError::InvalidArgument(format!(
            "mixed-arity restriction requires {} finite merge costs, got {}",
            level.len(),
            costs.len(),
        )));
    }
    let mut ranked = level.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_unstable_by(|(lhs_group, lhs_node), (rhs_group, rhs_node)| {
        costs[*lhs_group]
            .total_cmp(&costs[*rhs_group])
            .then_with(|| {
                hierarchy.nodes[*lhs_node]
                    .leaf_start
                    .cmp(&hierarchy.nodes[*rhs_node].leaf_start)
            })
    });

    let mut remaining_reduction = fine.len() - target;
    let mut reductions = vec![0_usize; level.len()];
    // Repeating 4/3/2-child aggregates distributes a fixed budget across
    // integer event arities instead of collapsing every selected region to
    // the same dyadic scale. Lower-cost groups receive the larger event first.
    const REDUCTION_PATTERN: [usize; 3] = [3, 2, 1];
    for (rank, (group, node)) in ranked.into_iter().enumerate() {
        if remaining_reduction == 0 {
            break;
        }
        let capacity = hierarchy.nodes[node].children.len().saturating_sub(1);
        let reduction = REDUCTION_PATTERN[rank % REDUCTION_PATTERN.len()]
            .min(capacity)
            .min(remaining_reduction);
        reductions[group] = reduction;
        remaining_reduction -= reduction;
    }
    if remaining_reduction != 0 {
        return Err(AutomataError::InvalidArgument(format!(
            "mixed-arity hierarchy cannot remove the remaining {remaining_reduction} leaves for target {target}",
        )));
    }

    let canonical_groups = level_one_leaf_groups(hierarchy, level)?;
    let mut merged_by_group = vec![None::<Vec<usize>>; canonical_groups.len()];
    for (group, children) in canonical_groups.iter().enumerate() {
        let reduction = reductions[group];
        if reduction > 0 {
            merged_by_group[group] = Some(best_merge_subset(model, fine, children, reduction + 1));
        }
    }
    let groups = material_groups_from_mixed_cut(&canonical_groups, &merged_by_group);
    if groups.len() != target {
        return Err(AutomataError::InvalidModel(format!(
            "mixed-arity restriction produced {} groups instead of {target}",
            groups.len(),
        )));
    }
    restricted_seed_from_fine_groups(fine, groups)
}

fn level_one_merge_costs(
    model: &AdaptiveNpaModel,
    fine: &AdaptiveParticleSet,
    hierarchy: &AdaptiveProxyHierarchy,
    level: &[usize],
) -> AutomataResult<Vec<f32>> {
    match model.config.hierarchical_restriction_policy {
        AdaptiveHierarchyRestrictionPolicy::LearnedController => {
            learned_level_one_merge_costs(model, fine, hierarchy)
        }
        AdaptiveHierarchyRestrictionPolicy::SpatialCompactness => Ok(level
            .iter()
            .map(|node| {
                hierarchy.nodes[*node]
                    .covariance
                    .iter()
                    .enumerate()
                    .filter_map(|(index, value)| {
                        let row = index / 3;
                        let col = index % 3;
                        (row == col && row < fine.spatial_dims).then_some(*value)
                    })
                    .sum::<f32>()
            })
            .collect()),
        AdaptiveHierarchyRestrictionPolicy::DynamicsDetail => Ok(level
            .iter()
            .map(|node| {
                hierarchy.nodes[*node]
                    .state
                    .iter()
                    .map(|value| value * value)
                    .sum()
            })
            .collect()),
    }
}

fn level_one_leaf_groups(
    hierarchy: &AdaptiveProxyHierarchy,
    level: &[usize],
) -> AutomataResult<Vec<Vec<usize>>> {
    level
        .iter()
        .map(|node| {
            hierarchy.nodes[*node]
                .children
                .iter()
                .map(|member| match member {
                    AdaptiveHierarchyMember::Leaf(index) => Ok(*index),
                    AdaptiveHierarchyMember::Proxy(_) => Err(AutomataError::InvalidModel(
                        "mixed-arity first-level group contains a proxy".to_owned(),
                    )),
                })
                .collect()
        })
        .collect()
}

fn material_groups_from_mixed_cut(
    canonical_groups: &[Vec<usize>],
    merged_by_group: &[Option<Vec<usize>>],
) -> Vec<Vec<usize>> {
    let mut groups = Vec::new();
    for (children, merged) in canonical_groups.iter().zip(merged_by_group) {
        let Some(merged) = merged else {
            groups.extend(children.iter().copied().map(|child| vec![child]));
            continue;
        };
        let mut local_groups = vec![merged.clone()];
        local_groups.extend(
            children
                .iter()
                .copied()
                .filter(|child| !merged.contains(child))
                .map(|child| vec![child]),
        );
        local_groups.sort_unstable_by_key(|members| {
            children
                .iter()
                .position(|child| members.contains(child))
                .unwrap_or(usize::MAX)
        });
        groups.extend(local_groups);
    }
    groups
}

fn best_merge_subset(
    model: &AdaptiveNpaModel,
    fine: &AdaptiveParticleSet,
    children: &[usize],
    count: usize,
) -> Vec<usize> {
    let mut best = Vec::new();
    let mut best_score = f32::INFINITY;
    for mask in 1_usize..(1_usize << children.len()) {
        if mask.count_ones() as usize != count {
            continue;
        }
        let subset = children
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, child)| ((mask >> index) & 1 == 1).then_some(child))
            .collect::<Vec<_>>();
        let score = merge_subset_score(model, fine, &subset);
        if score < best_score {
            best_score = score;
            best = subset;
        }
    }
    best
}

pub(crate) fn merge_subset_score(
    model: &AdaptiveNpaModel,
    fine: &AdaptiveParticleSet,
    subset: &[usize],
) -> f32 {
    let total = subset
        .iter()
        .map(|index| fine.represented_measure[*index])
        .sum::<f32>()
        .max(f32::MIN_POSITIVE);
    let mut position = [0.0_f32; 3];
    let mut state = vec![0.0_f32; fine.state_dims];
    for index in subset {
        let weight = fine.represented_measure[*index] / total;
        for (axis, value) in position.iter_mut().enumerate().take(fine.spatial_dims) {
            *value += weight * fine.positions[*index][axis];
        }
        for (channel, value) in state.iter_mut().enumerate() {
            *value += weight * fine.states[*index * fine.state_dims + channel];
        }
    }
    let spatial_scale = model.config.base_rule_footprint().max(f32::MIN_POSITIVE);
    subset
        .iter()
        .map(|index| {
            let weight = fine.represented_measure[*index] / total;
            let spatial = (0..fine.spatial_dims)
                .map(|axis| {
                    ((fine.positions[*index][axis] - position[axis]) / spatial_scale).powi(2)
                })
                .sum::<f32>();
            let latent = (0..fine.state_dims)
                .map(|channel| {
                    (fine.states[*index * fine.state_dims + channel] - state[channel]).powi(2)
                })
                .sum::<f32>()
                / fine.state_dims.max(1) as f32;
            weight * (spatial + latent)
        })
        .sum()
}

pub(crate) fn restricted_seed_from_fine_groups(
    fine: &AdaptiveParticleSet,
    groups: Vec<Vec<usize>>,
) -> AutomataResult<AdaptiveParticleSet> {
    let mut covered = vec![false; fine.len()];
    for group in &groups {
        if group.is_empty() || group.len() > 4 {
            return Err(AutomataError::InvalidModel(
                "mixed-arity material group must contain one to four fine leaves".to_owned(),
            ));
        }
        for child in group {
            if *child >= fine.len() || std::mem::replace(&mut covered[*child], true) {
                return Err(AutomataError::InvalidModel(
                    "mixed-arity material groups are not a fine-leaf partition".to_owned(),
                ));
            }
        }
    }
    if covered.contains(&false) {
        return Err(AutomataError::InvalidModel(
            "mixed-arity material groups do not cover every fine leaf".to_owned(),
        ));
    }

    let count = groups.len();
    let mut particles = AdaptiveParticleSet {
        spatial_dims: fine.spatial_dims,
        state_dims: fine.state_dims,
        positions: Vec::with_capacity(count),
        states: Vec::with_capacity(count * fine.state_dims),
        state_jacobian: Vec::with_capacity(count * fine.state_dims * fine.spatial_dims),
        closure_mode: Vec::with_capacity(count * fine.state_dims),
        closure_basis: Vec::with_capacity(count * 4),
        closure_phase: Vec::with_capacity(count * 2),
        represented_measure: Vec::with_capacity(count),
        render_footprint: Vec::with_capacity(count),
        bandwidth: Vec::with_capacity(count),
        covariance: Vec::with_capacity(count),
        particle_id: Vec::with_capacity(count),
        sibling_group: vec![0; count],
        generation: Vec::with_capacity(count),
        cooldown: vec![0; count],
        next_id: fine
            .particle_id
            .iter()
            .copied()
            .max()
            .unwrap_or_default()
            .saturating_add(1),
        next_sibling_group: fine.next_sibling_group.max(1),
        bootstrap_templates: Vec::new(),
    };
    for group in groups {
        if group.len() == 1 {
            let child = group[0];
            particles.positions.push(fine.positions[child]);
            particles.states.extend_from_slice(
                &fine.states[child * fine.state_dims..(child + 1) * fine.state_dims],
            );
            particles.state_jacobian.extend_from_slice(
                &fine.state_jacobian[child * fine.state_dims * fine.spatial_dims
                    ..(child + 1) * fine.state_dims * fine.spatial_dims],
            );
            if fine.closure_mode.is_empty() {
                particles
                    .closure_mode
                    .extend(std::iter::repeat_n(0.0, fine.state_dims));
            } else {
                particles.closure_mode.extend_from_slice(
                    &fine.closure_mode[child * fine.state_dims..(child + 1) * fine.state_dims],
                );
            }
            if fine.closure_basis.is_empty() {
                particles.closure_basis.extend(std::iter::repeat_n(0.0, 4));
            } else {
                particles
                    .closure_basis
                    .extend_from_slice(&fine.closure_basis[child * 4..(child + 1) * 4]);
            }
            if fine.closure_phase.is_empty() {
                particles.closure_phase.extend(std::iter::repeat_n(0.0, 2));
            } else {
                particles
                    .closure_phase
                    .extend_from_slice(&fine.closure_phase[child * 2..(child + 1) * 2]);
            }
            particles
                .represented_measure
                .push(fine.represented_measure[child]);
            particles.render_footprint.push(material_footprint_radius(
                fine.represented_measure[child],
                fine.spatial_dims,
            ));
            particles.bandwidth.push(fine.bandwidth[child]);
            particles.covariance.push(fine.covariance[child]);
            particles.particle_id.push(fine.particle_id[child]);
            particles.generation.push(fine.generation[child]);
            continue;
        }

        let measure = group
            .iter()
            .map(|child| fine.represented_measure[*child])
            .sum::<f32>()
            .max(f32::MIN_POSITIVE);
        let mut position = [0.0_f32; 4];
        let mut state = vec![0.0_f32; fine.state_dims];
        let mut bandwidth = 0.0_f32;
        for child in &group {
            let weight = fine.represented_measure[*child] / measure;
            for (axis, value) in position.iter_mut().enumerate().take(fine.spatial_dims) {
                *value += weight * fine.positions[*child][axis];
            }
            for (channel, value) in state.iter_mut().enumerate() {
                *value += weight * fine.states[*child * fine.state_dims + channel];
            }
            bandwidth += weight * fine.bandwidth[*child];
        }
        let mut covariance = [0.0_f32; 9];
        for child in &group {
            let weight = fine.represented_measure[*child] / measure;
            for row in 0..fine.spatial_dims {
                let row_delta = fine.positions[*child][row] - position[row];
                for col in 0..fine.spatial_dims {
                    let col_delta = fine.positions[*child][col] - position[col];
                    covariance[row * 3 + col] +=
                        weight * (fine.covariance[*child][row * 3 + col] + row_delta * col_delta);
                }
            }
        }
        particles.positions.push(position);
        particles.states.extend_from_slice(&state);
        particles
            .state_jacobian
            .extend(super::state::fit_state_jacobian(
                fine, &group, &state, position, covariance, measure,
            )?);
        particles
            .closure_mode
            .extend(std::iter::repeat_n(0.0, fine.state_dims));
        particles.closure_basis.extend(std::iter::repeat_n(0.0, 4));
        particles.closure_phase.extend(std::iter::repeat_n(0.0, 2));
        particles.represented_measure.push(measure);
        particles
            .render_footprint
            .push(material_footprint_radius(measure, fine.spatial_dims));
        particles.bandwidth.push(bandwidth);
        particles.covariance.push(covariance);
        let parent_id = particles.next_id;
        particles.next_id += 1;
        particles.particle_id.push(parent_id);
        particles.generation.push(
            group
                .iter()
                .map(|child| fine.generation[*child])
                .max()
                .unwrap_or_default()
                .saturating_add(1),
        );
        particles
            .bootstrap_templates
            .push(AdaptiveBootstrapTemplate {
                parent_id,
                children: group
                    .iter()
                    .map(|child| {
                        let state_base = child * fine.state_dims;
                        AdaptiveBootstrapChild {
                            position: fine.positions[*child],
                            state: fine.states[state_base..state_base + fine.state_dims].to_vec(),
                            represented_measure: fine.represented_measure[*child],
                            bandwidth: fine.bandwidth[*child],
                            covariance: fine.covariance[*child],
                            particle_id: fine.particle_id[*child],
                            generation: fine.generation[*child],
                        }
                    })
                    .collect(),
            });
    }
    particles
        .bootstrap_templates
        .sort_unstable_by_key(|template| template.parent_id);
    particles.validate()?;
    Ok(particles)
}

pub(crate) fn restrict_adaptive_particles_to_target_by_merge_cost(
    model: &AdaptiveNpaModel,
    fine: &AdaptiveParticleSet,
    merge_costs: &[f32],
) -> AutomataResult<AdaptiveParticleSet> {
    model.validate()?;
    fine.validate()?;
    let fine_leaf_count = model.config.bootstrap_fine_leaf_count();
    let target = model.config.target_leaves;
    if fine.len() != fine_leaf_count || target >= fine.len() || !fine.bootstrap_templates.is_empty()
    {
        return Err(AutomataError::InvalidArgument(format!(
            "adaptive cost-ranked restriction requires {fine_leaf_count} untemplated fine leaves above target {target}, got {}",
            fine.len(),
        )));
    }
    let children_per_split = 2 * model.config.spatial_dims;
    let hierarchy = AdaptiveProxyHierarchy::build(fine, children_per_split)?;
    restrict_adaptive_particles_to_target_by_merge_cost_with_hierarchy(
        model,
        fine,
        &hierarchy,
        merge_costs,
    )
}

pub(crate) fn restrict_adaptive_particles_to_target_by_merge_cost_with_hierarchy(
    model: &AdaptiveNpaModel,
    fine: &AdaptiveParticleSet,
    hierarchy: &AdaptiveProxyHierarchy,
    merge_costs: &[f32],
) -> AutomataResult<AdaptiveParticleSet> {
    restrict_adaptive_particles_to_leaf_budget_by_merge_cost_with_hierarchy(
        model,
        fine,
        hierarchy,
        model.config.target_leaves,
        merge_costs,
    )
}

pub(crate) fn restrict_adaptive_particles_to_leaf_budget_by_merge_cost_with_hierarchy(
    model: &AdaptiveNpaModel,
    fine: &AdaptiveParticleSet,
    hierarchy: &AdaptiveProxyHierarchy,
    target: usize,
    merge_costs: &[f32],
) -> AutomataResult<AdaptiveParticleSet> {
    model.validate()?;
    fine.validate()?;
    let fine_leaf_count = model.config.bootstrap_fine_leaf_count();
    if fine.len() != fine_leaf_count || target >= fine.len() || !fine.bootstrap_templates.is_empty()
    {
        return Err(AutomataError::InvalidArgument(format!(
            "adaptive cost-ranked restriction requires {fine_leaf_count} untemplated fine leaves above target {target}, got {}",
            fine.len(),
        )));
    }
    if model.config.hierarchical_restriction_arity == AdaptiveRestrictionArity::Mixed {
        return mixed_arity_restriction_from_merge_costs(
            model,
            fine,
            hierarchy,
            target,
            merge_costs,
        );
    }
    let view = hierarchy.material_cut_from_level_one_merge_costs(fine, target, merge_costs)?;
    hierarchical_seed_from_view(model, fine, hierarchy, view)
}

/// Restores the fine material represented by a hierarchy cut. Persistent-mode
/// synchronization updates every template child before this function is used,
/// so a subsequent restriction can reassign the same material budget without
/// inventing state or exposing internal covariance to rendering.
pub(crate) fn restore_adaptive_particles_from_templates(
    restricted: &AdaptiveParticleSet,
) -> AutomataResult<AdaptiveParticleSet> {
    restricted.validate()?;
    if restricted.bootstrap_templates.is_empty() {
        return Ok(restricted.clone());
    }

    let templates = restricted
        .bootstrap_templates
        .iter()
        .map(|template| (template.parent_id, template))
        .collect::<std::collections::BTreeMap<_, _>>();
    let jacobian_dims = restricted.state_dims * restricted.spatial_dims;
    let restored_count = restricted.len()
        + restricted
            .bootstrap_templates
            .iter()
            .map(|template| template.children.len().saturating_sub(1))
            .sum::<usize>();
    let mut rows = Vec::with_capacity(restored_count);
    for row in 0..restricted.len() {
        let jacobian = &restricted.state_jacobian[row * jacobian_dims..(row + 1) * jacobian_dims];
        if let Some(template) = templates.get(&restricted.particle_id[row]) {
            for child in &template.children {
                rows.push(RestoredParticleRow {
                    position: child.position,
                    state: child.state.clone(),
                    state_jacobian: jacobian.to_vec(),
                    represented_measure: child.represented_measure,
                    bandwidth: child.bandwidth,
                    covariance: child.covariance,
                    particle_id: child.particle_id,
                    generation: child.generation,
                    cooldown: 0,
                });
            }
        } else {
            rows.push(RestoredParticleRow {
                position: restricted.positions[row],
                state: restricted.states
                    [row * restricted.state_dims..(row + 1) * restricted.state_dims]
                    .to_vec(),
                state_jacobian: jacobian.to_vec(),
                represented_measure: restricted.represented_measure[row],
                bandwidth: restricted.bandwidth[row],
                covariance: restricted.covariance[row],
                particle_id: restricted.particle_id[row],
                generation: restricted.generation[row],
                cooldown: restricted.cooldown[row],
            });
        }
    }
    rows.sort_unstable_by_key(|row| row.particle_id);
    if rows
        .windows(2)
        .any(|pair| pair[0].particle_id == pair[1].particle_id)
    {
        return Err(AutomataError::InvalidModel(
            "restored adaptive material contains duplicate particle IDs".to_owned(),
        ));
    }

    let particles = AdaptiveParticleSet {
        spatial_dims: restricted.spatial_dims,
        state_dims: restricted.state_dims,
        positions: rows.iter().map(|row| row.position).collect(),
        states: rows
            .iter()
            .flat_map(|row| row.state.iter().copied())
            .collect(),
        state_jacobian: rows
            .iter()
            .flat_map(|row| row.state_jacobian.iter().copied())
            .collect(),
        closure_mode: vec![0.0; rows.len() * restricted.state_dims],
        closure_basis: vec![0.0; rows.len() * 4],
        closure_phase: vec![0.0; rows.len() * 2],
        represented_measure: rows.iter().map(|row| row.represented_measure).collect(),
        render_footprint: rows
            .iter()
            .map(|row| material_footprint_radius(row.represented_measure, restricted.spatial_dims))
            .collect(),
        bandwidth: rows.iter().map(|row| row.bandwidth).collect(),
        covariance: rows.iter().map(|row| row.covariance).collect(),
        particle_id: rows.iter().map(|row| row.particle_id).collect(),
        sibling_group: vec![0; rows.len()],
        generation: rows.iter().map(|row| row.generation).collect(),
        cooldown: rows.iter().map(|row| row.cooldown).collect(),
        next_id: rows
            .iter()
            .map(|row| row.particle_id)
            .max()
            .unwrap_or_default()
            .saturating_add(1)
            .max(restricted.next_id),
        next_sibling_group: restricted.next_sibling_group,
        bootstrap_templates: Vec::new(),
    };
    particles.validate()?;
    Ok(particles)
}

struct RestoredParticleRow {
    position: [f32; 4],
    state: Vec<f32>,
    state_jacobian: Vec<f32>,
    represented_measure: f32,
    bandwidth: f32,
    covariance: [f32; 9],
    particle_id: u64,
    generation: u16,
    cooldown: u16,
}

pub(crate) fn adaptive_template_child_groups(
    particles: &AdaptiveParticleSet,
) -> std::collections::BTreeSet<Vec<u64>> {
    particles
        .bootstrap_templates
        .iter()
        .map(|template| {
            let mut children = template
                .children
                .iter()
                .map(|child| child.particle_id)
                .collect::<Vec<_>>();
            children.sort_unstable();
            children
        })
        .collect()
}

fn hierarchical_seed_from_view(
    model: &AdaptiveNpaModel,
    fine: &AdaptiveParticleSet,
    hierarchy: &AdaptiveProxyHierarchy,
    mut view: AdaptiveMaterialView,
) -> AutomataResult<AdaptiveParticleSet> {
    if model.config.closure_recurrent_mode || model.config.compact_recurrent_memory_dims > 0 {
        super::closure::attach_first_closure_mode(fine, hierarchy, &mut view)?;
    }
    let AdaptiveMaterialView {
        mut particles,
        members,
        ..
    } = view;
    // Native material retains its fine identity so stochastic update masks are
    // invariant under restriction. Only proxy material receives IDs outside
    // the fine range; its retained children can then be restored without
    // colliding with any active leaf.
    let mut next_proxy_id = fine
        .particle_id
        .iter()
        .copied()
        .max()
        .unwrap_or_default()
        .saturating_add(1);
    for (index, member) in members.iter().copied().enumerate() {
        particles.particle_id[index] = match member {
            AdaptiveHierarchyMember::Leaf(fine_index) => fine.particle_id[fine_index],
            AdaptiveHierarchyMember::Proxy(_) => {
                let id = next_proxy_id;
                next_proxy_id += 1;
                id
            }
        };
    }
    particles.sibling_group.fill(0);
    particles.generation.fill(0);
    particles.cooldown.fill(0);
    particles.next_id = next_proxy_id;
    particles.next_sibling_group = 1;
    particles.bootstrap_templates = if model.config.retain_bootstrap_templates {
        members
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(parent, member)| match member {
                AdaptiveHierarchyMember::Leaf(_) => None,
                AdaptiveHierarchyMember::Proxy(_) => Some(AdaptiveBootstrapTemplate {
                    parent_id: particles.particle_id[parent],
                    children: hierarchy
                        .member_leaf_indices(member)
                        .iter()
                        .copied()
                        .map(|child| {
                            let state_base = child * fine.state_dims;
                            AdaptiveBootstrapChild {
                                position: fine.positions[child],
                                state: fine.states[state_base..state_base + fine.state_dims]
                                    .to_vec(),
                                represented_measure: fine.represented_measure[child],
                                bandwidth: fine.bandwidth[child],
                                covariance: fine.covariance[child],
                                particle_id: fine.particle_id[child],
                                generation: fine.generation[child],
                            }
                        })
                        .collect(),
                }),
            })
            .collect()
    } else {
        Vec::new()
    };
    particles
        .bootstrap_templates
        .sort_unstable_by_key(|template| template.parent_id);
    let equal_fine_measure = fine
        .represented_measure
        .first()
        .copied()
        .filter(|reference| {
            fine.represented_measure
                .iter()
                .all(|measure| (*measure - *reference).abs() <= 2.0e-6 * reference.abs().max(1.0))
        });
    let equal_fine_bandwidth = fine.bandwidth.first().copied().filter(|reference| {
        fine.bandwidth
            .iter()
            .all(|bandwidth| (*bandwidth - *reference).abs() <= 2.0e-6 * reference.abs().max(1.0))
    });
    if let (Some(fine_measure), Some(fine_bandwidth)) = (equal_fine_measure, equal_fine_bandwidth) {
        for (measure, bandwidth) in particles
            .represented_measure
            .iter()
            .copied()
            .zip(&mut particles.bandwidth)
        {
            *bandwidth =
                fine_bandwidth * (measure / fine_measure).powf(1.0 / particles.spatial_dims as f32);
        }
    }
    for (index, measure) in particles.represented_measure.iter().copied().enumerate() {
        particles.render_footprint[index] =
            material_footprint_radius(measure, particles.spatial_dims);
    }
    initialize_compact_recurrent_memory(model, &mut particles)?;
    particles.validate()?;
    Ok(particles)
}

fn initialize_compact_recurrent_memory(
    model: &AdaptiveNpaModel,
    particles: &mut AdaptiveParticleSet,
) -> AutomataResult<()> {
    let memory_dims = model.config.compact_recurrent_memory_dims;
    if memory_dims == 0 {
        return Ok(());
    }
    let state_dims = particles.state_dims;
    let memory_range = model.compact_recurrent_memory_range().ok_or_else(|| {
        AutomataError::InvalidModel(
            "compact recurrent memory overlaps the three-channel RGB tail".to_owned(),
        )
    })?;
    if particles.closure_basis.len() != particles.len() * 4
        || particles.closure_phase.len() != particles.len() * 2
    {
        return Err(AutomataError::InvalidModel(
            "compact recurrent memory requires hierarchy closure geometry".to_owned(),
        ));
    }
    for row in 0..particles.len() {
        let memory = &mut particles.states
            [row * state_dims + memory_range.start..row * state_dims + memory_range.end];
        memory.fill(0.0);
        for (target, source) in memory
            .iter_mut()
            .take(4)
            .zip(&particles.closure_basis[row * 4..(row + 1) * 4])
        {
            *target = source.clamp(-4.0, 4.0);
        }
        for (target, source) in memory
            .iter_mut()
            .skip(4)
            .take(2)
            .zip(&particles.closure_phase[row * 2..(row + 1) * 2])
        {
            *target = source.clamp(-1.0, 1.0);
        }
        if memory_dims > 6 {
            let footprint = material_footprint_radius(
                particles.represented_measure[row],
                particles.spatial_dims,
            )
            .max(f32::MIN_POSITIVE);
            let scale = footprint * footprint;
            let covariance = particles.covariance[row];
            memory[6] = ((covariance[0] - covariance[4]) / scale).clamp(-4.0, 4.0);
            if memory_dims > 7 {
                memory[7] = (2.0 * covariance[1] / scale).clamp(-4.0, 4.0);
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn equal_measure_seed(
    model: &AdaptiveNpaModel,
    particle_count: usize,
    seed: u64,
    seed_mode: ParticleSeed,
    scale: f32,
    total_measure: f32,
    bandwidth: f32,
) -> AutomataResult<AdaptiveParticleSet> {
    let (positions, states) = seed_particles_scaled(
        1,
        particle_count,
        model.rule.config.state_dims,
        model.config.spatial_dims,
        seed,
        seed_mode,
        scale,
    );
    AdaptiveParticleSet::from_equal_measure(
        positions,
        states,
        model.config.spatial_dims,
        model.rule.config.state_dims,
        total_measure,
        bandwidth,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn continuous_uniform_seed_from_reference(
    model: &AdaptiveNpaModel,
    particle_count: usize,
    reference_particle_count: usize,
    seed: u64,
    seed_mode: ParticleSeed,
    scale: f32,
    total_measure: f32,
    bandwidth: f32,
) -> AutomataResult<AdaptiveParticleSet> {
    if particle_count == 0 || reference_particle_count < particle_count {
        return Err(AutomataError::InvalidArgument(
            "continuous uniform seed requires 0 < active <= reference particles".to_owned(),
        ));
    }
    let state_dims = model.rule.config.state_dims;
    let spatial_dims = model.config.spatial_dims;
    let (reference_positions, reference_states) = seed_particles_scaled(
        1,
        reference_particle_count,
        state_dims,
        spatial_dims,
        seed,
        seed_mode,
        scale,
    );
    let selected = (0..particle_count)
        .map(|row| row * reference_particle_count / particle_count)
        .collect::<Vec<_>>();
    let mut positions = selected
        .iter()
        .map(|index| reference_positions[*index])
        .collect::<Vec<_>>();
    let mut states = selected
        .iter()
        .flat_map(|index| {
            reference_states[index * state_dims..(index + 1) * state_dims]
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    for axis in 0..spatial_dims {
        let reference_mean = reference_positions
            .iter()
            .map(|position| position[axis] as f64)
            .sum::<f64>()
            / reference_particle_count as f64;
        let selected_mean = positions
            .iter()
            .map(|position| position[axis] as f64)
            .sum::<f64>()
            / particle_count as f64;
        let correction = (reference_mean - selected_mean) as f32;
        for position in &mut positions {
            position[axis] += correction;
        }
    }
    for channel in 0..state_dims {
        let reference_mean = reference_states
            .chunks_exact(state_dims)
            .map(|state| state[channel] as f64)
            .sum::<f64>()
            / reference_particle_count as f64;
        let selected_mean = states
            .chunks_exact(state_dims)
            .map(|state| state[channel] as f64)
            .sum::<f64>()
            / particle_count as f64;
        let correction = (reference_mean - selected_mean) as f32;
        for state in states.chunks_exact_mut(state_dims) {
            state[channel] += correction;
        }
    }
    AdaptiveParticleSet::from_equal_measure(
        positions,
        states,
        spatial_dims,
        state_dims,
        total_measure,
        bandwidth,
    )
}

pub(crate) fn continuous_material_units(
    active_particle_count: usize,
    reference_particle_count: usize,
    max_to_min_ratio: f32,
) -> AutomataResult<Vec<f32>> {
    if active_particle_count == 0
        || reference_particle_count < active_particle_count
        || !max_to_min_ratio.is_finite()
        || max_to_min_ratio < 1.0
    {
        return Err(AutomataError::InvalidArgument(
            "continuous material units require 0 < active <= reference and ratio >= 1".to_owned(),
        ));
    }
    if max_to_min_ratio == 1.0 {
        return Ok(vec![
            reference_particle_count as f32
                / active_particle_count as f32;
            active_particle_count
        ]);
    }

    let half_log_ratio = 0.5 * f64::from(max_to_min_ratio).ln();
    let mut quantiles = (0..active_particle_count)
        .map(|index| {
            let centered = 2.0 * (index as f64 + 0.5) / active_particle_count as f64 - 1.0;
            (centered * half_log_ratio).exp()
        })
        .collect::<Vec<_>>();
    let normalization = reference_particle_count as f64 / quantiles.iter().sum::<f64>();
    quantiles
        .iter_mut()
        .for_each(|value| *value *= normalization);

    // Avoid a monotone storage-order scale ramp while keeping the material
    // multiset deterministic and independent of rollout seed.
    let mut row_order = (0..active_particle_count)
        .map(|row| (splitmix64(row as u64), row))
        .collect::<Vec<_>>();
    row_order.sort_unstable();
    let mut units = vec![0.0_f32; active_particle_count];
    for ((_, row), value) in row_order.into_iter().zip(quantiles) {
        units[row] = value as f32;
    }
    let correction = reference_particle_count as f32 - units.iter().copied().sum::<f32>();
    let correction_row = units
        .iter()
        .enumerate()
        .max_by(|lhs, rhs| lhs.1.total_cmp(rhs.1))
        .map(|(row, _)| row)
        .unwrap_or(0);
    units[correction_row] += correction;
    if units
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(AutomataError::InvalidModel(
            "continuous material layout produced invalid units".to_owned(),
        ));
    }
    Ok(units)
}

pub(crate) fn apply_continuous_material_layout(
    particles: &mut AdaptiveParticleSet,
    represented_measure: &[f32],
    bandwidth: &[f32],
) -> AutomataResult<()> {
    particles.validate()?;
    if represented_measure.len() != particles.len()
        || bandwidth.len() != particles.len()
        || represented_measure
            .iter()
            .chain(bandwidth)
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(AutomataError::InvalidArgument(
            "continuous material layout shape or values are invalid".to_owned(),
        ));
    }
    let old_total = particles.total_measure();
    let new_total = represented_measure
        .iter()
        .map(|value| f64::from(*value))
        .sum::<f64>();
    if (old_total - new_total).abs() > 2.0e-6 * old_total.abs().max(1.0) {
        return Err(AutomataError::InvalidArgument(format!(
            "continuous material layout changes total measure from {old_total} to {new_total}"
        )));
    }

    for axis in 0..particles.spatial_dims {
        let old_moment = particles
            .positions
            .iter()
            .zip(&particles.represented_measure)
            .map(|(position, measure)| f64::from(position[axis]) * f64::from(*measure))
            .sum::<f64>();
        let new_moment = particles
            .positions
            .iter()
            .zip(represented_measure)
            .map(|(position, measure)| f64::from(position[axis]) * f64::from(*measure))
            .sum::<f64>();
        let correction = ((old_moment - new_moment) / new_total) as f32;
        particles
            .positions
            .iter_mut()
            .for_each(|position| position[axis] += correction);
    }
    for channel in 0..particles.state_dims {
        let old_extensive = particles
            .states
            .chunks_exact(particles.state_dims)
            .zip(&particles.represented_measure)
            .map(|(state, measure)| f64::from(state[channel]) * f64::from(*measure))
            .sum::<f64>();
        let new_extensive = particles
            .states
            .chunks_exact(particles.state_dims)
            .zip(represented_measure)
            .map(|(state, measure)| f64::from(state[channel]) * f64::from(*measure))
            .sum::<f64>();
        let correction = ((old_extensive - new_extensive) / new_total) as f32;
        particles
            .states
            .chunks_exact_mut(particles.state_dims)
            .for_each(|state| state[channel] += correction);
    }

    particles
        .represented_measure
        .copy_from_slice(represented_measure);
    particles.bandwidth.copy_from_slice(bandwidth);
    for ((measure, render_footprint), particle_covariance) in represented_measure
        .iter()
        .zip(&mut particles.render_footprint)
        .zip(&mut particles.covariance)
    {
        let footprint = material_footprint_radius(*measure, particles.spatial_dims);
        *render_footprint = footprint;
        let variance = (0.5 * footprint).powi(2);
        let mut covariance = [0.0_f32; 9];
        for axis in 0..particles.spatial_dims {
            covariance[axis * 3 + axis] = variance;
        }
        *particle_covariance = covariance;
    }
    particles.validate()
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NpaConfig, NpaModel};

    #[test]
    fn hierarchical_bootstrap_preserves_target_seed_material_moments() {
        let mut adaptive = super::super::AdaptiveNpaConfig::growing_2d();
        adaptive.min_leaves = 16;
        adaptive.initial_leaves = 16;
        adaptive.target_leaves = 64;
        adaptive.max_leaves = 64;
        adaptive.bootstrap_end_step = 1;
        adaptive.bootstrap_seed_spread = 0.0;
        let model =
            AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 7), adaptive, 11)
                .unwrap();
        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let coarse = seed_adaptive_particles_scaled(
            &model,
            16,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            0.1,
        )
        .unwrap();
        let fine = equal_measure_seed(
            &model,
            64,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            0.1,
        )
        .unwrap();

        assert_eq!(coarse.len(), 16);
        assert_eq!(coarse.bootstrap_templates.len(), 16);
        assert!((coarse.total_measure() - fine.total_measure()).abs() <= 1.0e-7);
        let expected_measure = total_measure / 16.0;
        assert!(
            coarse
                .represented_measure
                .iter()
                .all(|measure| (*measure - expected_measure).abs() <= 1.0e-7)
        );
        for axis in 0..2 {
            let centroid = |particles: &AdaptiveParticleSet| {
                particles
                    .positions
                    .iter()
                    .zip(&particles.represented_measure)
                    .map(|(position, measure)| position[axis] as f64 * *measure as f64)
                    .sum::<f64>()
                    / particles.total_measure()
            };
            assert!((centroid(&coarse) - centroid(&fine)).abs() <= 1.0e-7);
        }

        let mut restored = coarse;
        let update =
            super::super::apply_adaptive_topology_at_step(&model, &mut restored, 1, 0).unwrap();
        assert_eq!(update.split_events, 16);
        assert_eq!(restored.len(), fine.len());
        assert!(restored.bootstrap_templates.is_empty());
        for fine_index in 0..fine.len() {
            let id = fine.particle_id[fine_index];
            let restored_index = restored
                .particle_id
                .iter()
                .position(|candidate| *candidate == id)
                .unwrap();
            assert_eq!(
                restored.positions[restored_index],
                fine.positions[fine_index]
            );
            assert_eq!(
                &restored.states[restored_index * restored.state_dims
                    ..(restored_index + 1) * restored.state_dims],
                &fine.states[fine_index * fine.state_dims..(fine_index + 1) * fine.state_dims]
            );
            assert_eq!(
                restored.represented_measure[restored_index],
                fine.represented_measure[fine_index]
            );
            assert_eq!(
                restored.bandwidth[restored_index],
                fine.bandwidth[fine_index]
            );
            assert_eq!(
                restored.covariance[restored_index],
                fine.covariance[fine_index]
            );
        }
    }

    #[test]
    fn hierarchical_bootstrap_supports_a_partial_fine_seed_cut() {
        let mut adaptive = super::super::AdaptiveNpaConfig::growing_2d();
        adaptive.min_leaves = 16;
        adaptive.initial_leaves = 16;
        adaptive.target_leaves = 40;
        adaptive.max_leaves = 64;
        adaptive.bootstrap_fine_leaves = 64;
        adaptive.bootstrap_end_step = 1;
        adaptive.bootstrap_events_per_interval = 8;
        adaptive.max_events_per_interval = 8;
        adaptive.bootstrap_seed_spread = 0.0;
        let model =
            AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 7), adaptive, 11)
                .unwrap();
        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let mut partial = seed_adaptive_particles_scaled(
            &model,
            16,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            0.1,
        )
        .unwrap();
        let fine = equal_measure_seed(
            &model,
            64,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            0.1,
        )
        .unwrap();

        let update =
            super::super::apply_adaptive_topology_at_step(&model, &mut partial, 1, 0).unwrap();

        assert_eq!(update.split_events, 8);
        assert_eq!(partial.len(), 40);
        assert_eq!(partial.bootstrap_templates.len(), 8);
        assert!((partial.total_measure() - fine.total_measure()).abs() <= 1.0e-7);
        let fine_measure = total_measure / 64.0;
        let coarse_measure = total_measure / 16.0;
        assert_eq!(
            partial
                .represented_measure
                .iter()
                .filter(|measure| (**measure - fine_measure).abs() <= 1.0e-7)
                .count(),
            32
        );
        assert_eq!(
            partial
                .represented_measure
                .iter()
                .filter(|measure| (**measure - coarse_measure).abs() <= 1.0e-7)
                .count(),
            8
        );
        for axis in 0..2 {
            let centroid = |particles: &AdaptiveParticleSet| {
                particles
                    .positions
                    .iter()
                    .zip(&particles.represented_measure)
                    .map(|(position, measure)| position[axis] as f64 * *measure as f64)
                    .sum::<f64>()
                    / particles.total_measure()
            };
            assert!((centroid(&partial) - centroid(&fine)).abs() <= 1.0e-7);
        }
    }

    #[test]
    fn mixed_arity_restriction_preserves_material_and_populates_fractional_octaves() {
        let mut adaptive = super::super::AdaptiveNpaConfig::growing_2d();
        adaptive.min_leaves = 16;
        adaptive.initial_leaves = 64;
        adaptive.target_leaves = 40;
        adaptive.max_leaves = 64;
        adaptive.bootstrap_fine_leaves = 64;
        adaptive.hierarchical_restriction_step = 1;
        adaptive.hierarchical_restriction_arity = AdaptiveRestrictionArity::Mixed;
        let model =
            AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 7), adaptive, 11)
                .unwrap();
        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let fine = equal_measure_seed(
            &model,
            64,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            0.1,
        )
        .unwrap();

        let restricted = restrict_adaptive_particles_to_target(&model, &fine).unwrap();
        let hierarchy = AdaptiveProxyHierarchy::build(&fine, 4).unwrap();
        let level = hierarchy.levels.first().unwrap();
        let merge_costs = level_one_merge_costs(&model, &fine, &hierarchy, level).unwrap();
        let precomputed = restrict_adaptive_particles_to_leaf_budget_by_merge_cost_with_hierarchy(
            &model,
            &fine,
            &hierarchy,
            model.config.target_leaves,
            &merge_costs,
        )
        .unwrap();

        assert_eq!(restricted.len(), 40);
        assert_eq!(precomputed, restricted);
        assert_eq!(
            restricted
                .bootstrap_templates
                .iter()
                .filter(|template| template.children.len() == 2)
                .count(),
            4
        );
        assert_eq!(
            restricted
                .bootstrap_templates
                .iter()
                .filter(|template| template.children.len() == 3)
                .count(),
            4
        );
        assert_eq!(
            restricted
                .bootstrap_templates
                .iter()
                .filter(|template| template.children.len() == 4)
                .count(),
            4
        );
        assert!((restricted.total_measure() - fine.total_measure()).abs() <= 1.0e-7);
        for axis in 0..2 {
            let centroid = |particles: &AdaptiveParticleSet| {
                particles
                    .positions
                    .iter()
                    .zip(&particles.represented_measure)
                    .map(|(position, measure)| position[axis] as f64 * *measure as f64)
                    .sum::<f64>()
                    / particles.total_measure()
            };
            assert!((centroid(&restricted) - centroid(&fine)).abs() <= 1.0e-7);
        }
        let restored = restore_adaptive_particles_from_templates(&restricted).unwrap();
        assert_eq!(restored.particle_id, fine.particle_id);
        assert_eq!(restored.represented_measure, fine.represented_measure);
    }

    #[test]
    fn dynamics_detail_restriction_uses_scale_conditioned_rule_features() {
        let mut adaptive = super::super::AdaptiveNpaConfig::growing_2d();
        adaptive.min_leaves = 16;
        adaptive.initial_leaves = 64;
        adaptive.target_leaves = 40;
        adaptive.max_leaves = 64;
        adaptive.bootstrap_fine_leaves = 64;
        adaptive.hierarchical_restriction_step = 1;
        adaptive.hierarchical_restriction_policy =
            AdaptiveHierarchyRestrictionPolicy::DynamicsDetail;
        let mut model =
            AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 7), adaptive, 11)
                .unwrap();
        let native_rule = model.rule.clone();
        model.enable_material_scale_conditioning().unwrap();
        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let fine = equal_measure_seed(
            &model,
            64,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            model.rule.config.eps0,
        )
        .unwrap();

        let perception = rule_perception_pair(&model.config, &model.rule, &fine).unwrap();
        let conditioned =
            primary_rule_features(&model, &fine, &perception.npa_compatible.features).unwrap();
        assert_eq!(
            perception.npa_compatible.features.len(),
            fine.len() * native_rule.config.perception_dims()
        );
        assert_eq!(
            conditioned.len(),
            fine.len() * model.rule.config.perception_dims()
        );
        let native_update = native_rule
            .forward_update_from_features(&perception.npa_compatible.features)
            .unwrap();
        let conditioned_update = model
            .rule
            .forward_update_from_features(conditioned.as_ref())
            .unwrap();
        assert_eq!(conditioned_update, native_update);

        let restricted = restrict_adaptive_particles_to_target(&model, &fine).unwrap();
        assert_eq!(restricted.len(), model.config.target_leaves);
    }

    #[test]
    fn hierarchical_target_seed_matches_the_training_cut_distribution() {
        let mut adaptive = super::super::AdaptiveNpaConfig::growing_2d();
        adaptive.min_leaves = 40;
        adaptive.initial_leaves = 40;
        adaptive.target_leaves = 40;
        adaptive.max_leaves = 64;
        adaptive.bootstrap_fine_leaves = 64;
        adaptive.bootstrap_end_step = 0;
        adaptive.hierarchical_bootstrap_seed = true;
        let model =
            AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 7), adaptive, 11)
                .unwrap();
        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let cut = seed_adaptive_particles_scaled(
            &model,
            40,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            0.1,
        )
        .unwrap();
        let fine = equal_measure_seed(
            &model,
            64,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            0.1,
        )
        .unwrap();

        assert_eq!(cut.len(), 40);
        assert_eq!(cut.bootstrap_templates.len(), 8);
        assert!((cut.total_measure() - fine.total_measure()).abs() <= 1.0e-7);
        let fine_measure = total_measure / 64.0;
        assert_eq!(
            cut.represented_measure
                .iter()
                .filter(|measure| (**measure - fine_measure).abs() <= 1.0e-7)
                .count(),
            32
        );
        assert_eq!(
            cut.represented_measure
                .iter()
                .filter(|measure| (**measure - 4.0 * fine_measure).abs() <= 1.0e-7)
                .count(),
            8
        );
        assert_eq!(
            cut.particle_id
                .iter()
                .filter(|id| **id < fine.len() as u64)
                .count(),
            32
        );
        assert_eq!(
            cut.particle_id
                .iter()
                .filter(|id| **id >= fine.len() as u64)
                .count(),
            8
        );
        assert!(cut.bootstrap_templates.iter().all(|template| {
            template
                .children
                .iter()
                .all(|child| child.particle_id < fine.len() as u64)
        }));
        for row in 0..2 {
            for col in 0..2 {
                let moment = |particles: &AdaptiveParticleSet| {
                    particles
                        .positions
                        .iter()
                        .zip(&particles.covariance)
                        .zip(&particles.represented_measure)
                        .map(|((position, covariance), measure)| {
                            f64::from(*measure)
                                * f64::from(
                                    covariance[row * 3 + col] + position[row] * position[col],
                                )
                        })
                        .sum::<f64>()
                };
                assert!((moment(&cut) - moment(&fine)).abs() <= 1.0e-7);
            }
        }

        let restored = restore_adaptive_particles_from_templates(&cut).unwrap();
        assert_eq!(restored.len(), fine.len());
        assert!(restored.bootstrap_templates.is_empty());
        assert_eq!(restored.particle_id, fine.particle_id);
        assert_eq!(restored.positions, fine.positions);
        assert_eq!(restored.states, fine.states);
        assert_eq!(restored.represented_measure, fine.represented_measure);
    }

    #[test]
    fn hierarchical_target_seed_rejects_an_unreachable_cut() {
        let mut adaptive = super::super::AdaptiveNpaConfig::growing_2d();
        adaptive.min_leaves = 41;
        adaptive.initial_leaves = 41;
        adaptive.target_leaves = 41;
        adaptive.max_leaves = 64;
        adaptive.bootstrap_fine_leaves = 64;
        adaptive.bootstrap_end_step = 0;
        adaptive.hierarchical_bootstrap_seed = true;
        let model =
            AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 7), adaptive, 11)
                .unwrap();
        let error = seed_adaptive_particles_scaled(
            &model,
            41,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            std::f32::consts::PI * 0.2_f32.powi(2),
            0.1,
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("not reachable from 64 leaves by 4-child events")
        );
    }
}
