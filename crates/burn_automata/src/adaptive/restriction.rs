use super::{
    ADAPTIVE_CONTROLLER_INPUT_DIMS, AdaptiveHierarchyMember, AdaptiveNpaModel, AdaptiveParticleSet,
    AdaptiveProxyHierarchy, features::controller_features, perception::rule_perception_pair,
};
use crate::{AutomataError, AutomataResult};
use burn_automata_kernels::{AdaptiveGraphMetrics, AdaptivePerceptionOutput};

pub(crate) fn level_one_restriction_features(
    model: &AdaptiveNpaModel,
    fine: &AdaptiveParticleSet,
    hierarchy: &AdaptiveProxyHierarchy,
) -> AutomataResult<Vec<[f32; ADAPTIVE_CONTROLLER_INPUT_DIMS]>> {
    let perception = rule_perception_pair(&model.config, &model.rule, fine)?;
    let base_update = model
        .rule
        .forward_update_from_features(&perception.npa_compatible.features)?;
    level_one_restriction_features_from_perception(
        model,
        fine,
        hierarchy,
        &perception.normalized,
        &base_update,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn level_one_restriction_features_from_precomputed(
    model: &AdaptiveNpaModel,
    fine: &AdaptiveParticleSet,
    hierarchy: &AdaptiveProxyHierarchy,
    normalized_features: &[f32],
    base_update: &[f32],
    observed_spacing: &[f32],
    accepted_degree: &[usize],
    feature_dims: usize,
) -> AutomataResult<Vec<[f32; ADAPTIVE_CONTROLLER_INPUT_DIMS]>> {
    let rows = fine.len();
    let gradient_dims = fine.state_dims * fine.spatial_dims;
    let gradient_start = 2 * fine.state_dims;
    if feature_dims != model.config.perception.feature_dims(fine.state_dims)
        || normalized_features.len() != rows * feature_dims
        || base_update.len() != rows * model.rule.config.update_dims()
        || observed_spacing.len() != rows
        || accepted_degree.len() != rows
        || gradient_start + gradient_dims > feature_dims
    {
        return Err(AutomataError::InvalidArgument(
            "precomputed adaptive restriction feature shape mismatch".to_owned(),
        ));
    }
    let mut state_gradient = Vec::with_capacity(rows * gradient_dims);
    for row in normalized_features.chunks_exact(feature_dims) {
        state_gradient.extend_from_slice(&row[gradient_start..gradient_start + gradient_dims]);
    }
    let perception = AdaptivePerceptionOutput {
        features: normalized_features.to_vec(),
        normalized_state: Vec::new(),
        state_gradient,
        occupancy_gradient: Vec::new(),
        partition: Vec::new(),
        coarse_exposure: vec![0.0; rows],
        observed_spacing: observed_spacing.to_vec(),
        moment_condition: Vec::new(),
        moment_fallback: Vec::new(),
        accepted_degree: accepted_degree.to_vec(),
        graph: AdaptiveGraphMetrics::default(),
        feature_dims,
    };
    level_one_restriction_features_from_perception(model, fine, hierarchy, &perception, base_update)
}

fn level_one_restriction_features_from_perception(
    model: &AdaptiveNpaModel,
    fine: &AdaptiveParticleSet,
    hierarchy: &AdaptiveProxyHierarchy,
    perception: &AdaptivePerceptionOutput,
    base_update: &[f32],
) -> AutomataResult<Vec<[f32; ADAPTIVE_CONTROLLER_INPUT_DIMS]>> {
    let leaf_features = controller_features(&model.config, fine, perception, base_update);
    let level = hierarchy.levels.first().ok_or_else(|| {
        AutomataError::InvalidArgument(
            "learned hierarchy restriction requires first-level groups".to_string(),
        )
    })?;
    let perception_dims = perception
        .feature_dims
        .min(super::ADAPTIVE_CONTROLLER_CONTEXT_DIMS);
    let context_dims = perception_dims
        + model
            .rule
            .config
            .update_dims()
            .min(super::ADAPTIVE_CONTROLLER_CONTEXT_DIMS.saturating_sub(perception_dims));
    let extra_start = super::ADAPTIVE_CONTROLLER_SCALAR_DIMS + context_dims;
    let total_measure = fine
        .represented_measure
        .iter()
        .sum::<f32>()
        .max(f32::MIN_POSITIVE);
    let material_center = fine.positions.iter().zip(&fine.represented_measure).fold(
        [0.0_f32; 2],
        |mut center, (position, measure)| {
            center[0] += position[0] * measure / total_measure;
            center[1] += position[1] * measure / total_measure;
            center
        },
    );
    level
        .iter()
        .map(|node| {
            let node = &hierarchy.nodes[*node];
            let children = &node.children;
            if children.len() != hierarchy.branch_factor {
                return Err(AutomataError::InvalidArgument(
                    "learned hierarchy restriction requires complete sibling groups".to_string(),
                ));
            }
            let mut group = [0.0_f32; ADAPTIVE_CONTROLLER_INPUT_DIMS];
            for child in children {
                let AdaptiveHierarchyMember::Leaf(index) = child else {
                    return Err(AutomataError::InvalidArgument(
                        "learned hierarchy restriction expects first-level leaf children"
                            .to_string(),
                    ));
                };
                for (dst, value) in group.iter_mut().zip(&leaf_features[*index]) {
                    *dst += *value / children.len() as f32;
                }
            }
            let domain_extent = [
                (model.config.domain_max[0] - model.config.domain_min[0]).max(f32::MIN_POSITIVE),
                (model.config.domain_max[1] - model.config.domain_min[1]).max(f32::MIN_POSITIVE),
            ];
            let reference_variance = model.config.base_rule_footprint().powi(2);
            let relative_position = [
                2.0 * (node.position[0] - material_center[0]) / domain_extent[0],
                2.0 * (node.position[1] - material_center[1]) / domain_extent[1],
            ];
            let mut extras = Vec::with_capacity(22 + 2 * fine.state_dims);
            extras.extend(relative_position);
            for frequency in [1.0_f32, 2.0, 4.0, 8.0] {
                for coordinate in relative_position {
                    let (sin, cos) = (std::f32::consts::PI * frequency * coordinate).sin_cos();
                    extras.extend([sin, cos]);
                }
            }
            extras.extend([
                (node.covariance[0] / reference_variance).ln_1p(),
                (node.covariance[4] / reference_variance).ln_1p(),
                node.covariance[1] / reference_variance,
                (node.bounding_radius / model.config.base_rule_footprint()).ln_1p(),
            ]);
            extras.extend(node.state.iter().map(|value| value.tanh()));
            for channel in 0..fine.state_dims {
                let variance = children
                    .iter()
                    .map(|child| match child {
                        AdaptiveHierarchyMember::Leaf(index) => {
                            let value = fine.states[index * fine.state_dims + channel];
                            (value - node.state[channel]).powi(2)
                        }
                        AdaptiveHierarchyMember::Proxy(_) => unreachable!(),
                    })
                    .sum::<f32>()
                    / children.len() as f32;
                extras.push(variance.sqrt().ln_1p());
            }
            // Group means alone hide precisely the local disagreement that
            // determines whether replacing four children by one parent is
            // destructive. Fill the remaining controller context with
            // permutation-invariant leaf-feature deviations.
            for channel in 0..ADAPTIVE_CONTROLLER_INPUT_DIMS {
                if extra_start + extras.len() == ADAPTIVE_CONTROLLER_INPUT_DIMS {
                    break;
                }
                let variance = children
                    .iter()
                    .map(|child| match child {
                        AdaptiveHierarchyMember::Leaf(index) => {
                            (leaf_features[*index][channel] - group[channel]).powi(2)
                        }
                        AdaptiveHierarchyMember::Proxy(_) => unreachable!(),
                    })
                    .sum::<f32>()
                    / children.len() as f32;
                extras.push(variance.sqrt().ln_1p());
            }
            for (dst, value) in group[extra_start..].iter_mut().zip(extras) {
                *dst = value.clamp(-8.0, 8.0);
            }
            Ok(group)
        })
        .collect()
}

pub(crate) fn learned_level_one_merge_costs(
    model: &AdaptiveNpaModel,
    fine: &AdaptiveParticleSet,
    hierarchy: &AdaptiveProxyHierarchy,
) -> AutomataResult<Vec<f32>> {
    let controller = model.restriction_controller.as_ref().ok_or_else(|| {
        AutomataError::InvalidModel(
            "learned hierarchy restriction requires restriction_controller".to_string(),
        )
    })?;
    Ok(controller
        .forward(&level_one_restriction_features(model, fine, hierarchy)?)
        .into_iter()
        .map(|output| -output.merge_probability)
        .collect())
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "gpu_wgpu")]
pub(crate) fn learned_level_one_merge_costs_from_precomputed(
    model: &AdaptiveNpaModel,
    fine: &AdaptiveParticleSet,
    hierarchy: &AdaptiveProxyHierarchy,
    normalized_features: &[f32],
    base_update: &[f32],
    observed_spacing: &[f32],
    accepted_degree: &[usize],
    feature_dims: usize,
) -> AutomataResult<Vec<f32>> {
    let controller = model.restriction_controller.as_ref().ok_or_else(|| {
        AutomataError::InvalidModel(
            "learned hierarchy restriction requires restriction_controller".to_owned(),
        )
    })?;
    Ok(controller
        .forward(&level_one_restriction_features_from_precomputed(
            model,
            fine,
            hierarchy,
            normalized_features,
            base_update,
            observed_spacing,
            accepted_degree,
            feature_dims,
        )?)
        .into_iter()
        .map(|output| -output.merge_probability)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NpaConfig, NpaModel, ParticleSeed};

    #[test]
    fn restriction_features_are_translation_invariant() {
        let fine_count = 64;
        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let footprint =
            crate::adaptive::material_footprint_radius(total_measure / fine_count as f32, 2);
        let mut config = crate::adaptive::AdaptiveNpaConfig::growing_2d();
        config.reference_footprint = footprint;
        config.base_rule_footprint = footprint;
        config.min_footprint = 0.5 * footprint;
        config.max_footprint = 2.0 * footprint;
        config.min_leaves = 16;
        config.target_leaves = 58;
        config.max_leaves = fine_count;
        config.initial_leaves = fine_count;
        config.bootstrap_fine_leaves = fine_count;
        let model =
            AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 7), config, 11)
                .unwrap();
        let particles = crate::adaptive::seed_adaptive_particles_scaled(
            &model,
            fine_count,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            0.1,
        )
        .unwrap();
        let hierarchy = AdaptiveProxyHierarchy::build(&particles, 4).unwrap();
        let expected = level_one_restriction_features(&model, &particles, &hierarchy).unwrap();
        let mut translated = particles.clone();
        for position in &mut translated.positions {
            position[0] += 0.125;
            position[1] -= 0.075;
        }
        // A rigid translation must preserve the hierarchy membership as well
        // as the material state. Rebuilding the quantized Morton hierarchy can
        // move points across a 10-bit bin through floating-point roundoff,
        // which tests hierarchy construction rather than feature invariance.
        let offset = [0.125, -0.075];
        let mut translated_hierarchy = hierarchy.clone();
        for node in &mut translated_hierarchy.nodes {
            node.position[0] += offset[0];
            node.position[1] += offset[1];
        }
        let mut translated_model = model.clone();
        translated_model.config.domain_min[0] += offset[0];
        translated_model.config.domain_min[1] += offset[1];
        translated_model.config.domain_max[0] += offset[0];
        translated_model.config.domain_max[1] += offset[1];
        let actual =
            level_one_restriction_features(&translated_model, &translated, &translated_hierarchy)
                .unwrap();
        let max_error = expected
            .iter()
            .zip(actual)
            .flat_map(|(expected, actual)| {
                expected
                    .iter()
                    .zip(actual)
                    .map(|(expected, actual)| (expected - actual).abs())
            })
            .fold(0.0_f32, f32::max);
        assert!(max_error < 1.0e-5, "translation feature error {max_error}");
    }
}
