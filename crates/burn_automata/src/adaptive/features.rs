use std::borrow::Cow;

use super::{
    ADAPTIVE_CONTROLLER_CONTEXT_DIMS, ADAPTIVE_CONTROLLER_INPUT_DIMS,
    ADAPTIVE_CONTROLLER_SCALAR_DIMS, AdaptiveLocalRuleSemantics, AdaptiveNpaConfig,
    AdaptiveParticleSet, AdaptiveProxyHierarchy,
};
use crate::AutomataResult;
use burn_automata_kernels::{
    AdaptiveGraphPolicy, AdaptivePerceptionOutput, AdaptivePerceptionPair,
    adaptive_perceive_without_spacing, adaptive_proxy_perceive,
};

#[derive(Clone, Debug)]
pub(crate) struct AdaptiveProxyContext {
    pub perception: AdaptivePerceptionOutput,
    pub node_count: usize,
}

/// Selects the perception stream consumed by the deployed local rule.
///
/// Keeping training and runtime behind this helper is important: compatible
/// residuals preserve the regular NPA feature contract, while replacement and
/// normalized residual rules consume the adaptive normalized contract.
pub(crate) fn local_rule_perception<'a>(
    config: &AdaptiveNpaConfig,
    perception: &'a AdaptivePerceptionPair,
) -> &'a AdaptivePerceptionOutput {
    match config.local_rule_semantics {
        AdaptiveLocalRuleSemantics::CompatibleResidual => &perception.npa_compatible,
        AdaptiveLocalRuleSemantics::Residual
        | AdaptiveLocalRuleSemantics::NormalizedExposureResidual
        | AdaptiveLocalRuleSemantics::CoarseReplacement => &perception.normalized,
    }
}

/// Returns the exact scalar gate applied to a local residual at deployment.
pub(crate) fn local_residual_gate(
    config: &AdaptiveNpaConfig,
    particles: &AdaptiveParticleSet,
    perception: &AdaptivePerceptionOutput,
    row: usize,
) -> f32 {
    match config.local_rule_semantics {
        AdaptiveLocalRuleSemantics::CompatibleResidual => perception.coarse_exposure[row].max(0.0),
        AdaptiveLocalRuleSemantics::Residual => {
            config.residual_gate(particles.footprint(row)).max(0.0)
        }
        AdaptiveLocalRuleSemantics::NormalizedExposureResidual => {
            config.residual_gate(particles.footprint(row)).max(0.0)
        }
        AdaptiveLocalRuleSemantics::CoarseReplacement => {
            f32::from(config.is_coarse_rule_footprint(particles.footprint(row)))
        }
    }
}

pub(crate) fn controller_features(
    config: &AdaptiveNpaConfig,
    particles: &AdaptiveParticleSet,
    perception: &AdaptivePerceptionOutput,
    base_update: &[f32],
) -> Vec<[f32; ADAPTIVE_CONTROLLER_INPUT_DIMS]> {
    controller_features_from_rows(
        config,
        particles,
        perception,
        base_update,
        0..particles.len(),
    )
}

pub(crate) fn controller_features_for_rows(
    config: &AdaptiveNpaConfig,
    particles: &AdaptiveParticleSet,
    perception: &AdaptivePerceptionOutput,
    base_update: &[f32],
    rows: &[usize],
) -> Vec<[f32; ADAPTIVE_CONTROLLER_INPUT_DIMS]> {
    debug_assert!(rows.iter().all(|row| *row < particles.len()));
    controller_features_from_rows(
        config,
        particles,
        perception,
        base_update,
        rows.iter().copied(),
    )
}

fn controller_features_from_rows(
    config: &AdaptiveNpaConfig,
    particles: &AdaptiveParticleSet,
    perception: &AdaptivePerceptionOutput,
    base_update: &[f32],
    rows: impl IntoIterator<Item = usize>,
) -> Vec<[f32; ADAPTIVE_CONTROLLER_INPUT_DIMS]> {
    let variation = state_variation(particles, perception);
    let mean_variation = mean(&variation).max(1.0e-6);
    let perception_dims = perception.features.len() / particles.len();
    let update_dims = base_update.len() / particles.len();
    debug_assert_eq!(base_update.len(), particles.len() * update_dims);
    debug_assert!(perception_dims + update_dims <= ADAPTIVE_CONTROLLER_CONTEXT_DIMS);
    let copied_perception_dims = perception_dims.min(ADAPTIVE_CONTROLLER_CONTEXT_DIMS);
    let copied_update_dims =
        update_dims.min(ADAPTIVE_CONTROLLER_CONTEXT_DIMS.saturating_sub(copied_perception_dims));
    let context_dims = copied_perception_dims + copied_update_dims;
    let mut context_mean = vec![0.0; context_dims];
    let mut context_variance = vec![0.0; context_dims];
    for index in 0..particles.len() {
        let perception_row =
            &perception.features[index * perception_dims..(index + 1) * perception_dims];
        let update_row = &base_update[index * update_dims..(index + 1) * update_dims];
        for (channel, value) in perception_row
            .iter()
            .take(copied_perception_dims)
            .enumerate()
        {
            context_mean[channel] += value;
        }
        for (channel, value) in update_row.iter().take(copied_update_dims).enumerate() {
            context_mean[copied_perception_dims + channel] += value;
        }
    }
    for value in &mut context_mean {
        *value /= particles.len() as f32;
    }
    for index in 0..particles.len() {
        let perception_row =
            &perception.features[index * perception_dims..(index + 1) * perception_dims];
        let update_row = &base_update[index * update_dims..(index + 1) * update_dims];
        for (channel, value) in perception_row
            .iter()
            .take(copied_perception_dims)
            .enumerate()
        {
            context_variance[channel] += (*value - context_mean[channel]).powi(2);
        }
        for (channel, value) in update_row.iter().take(copied_update_dims).enumerate() {
            let context_channel = copied_perception_dims + channel;
            context_variance[context_channel] += (*value - context_mean[context_channel]).powi(2);
        }
    }
    for value in &mut context_variance {
        *value = (*value / particles.len() as f32).sqrt().max(1.0e-4);
    }
    rows.into_iter()
        .map(|index| {
            let boundary_distance = (0..particles.spatial_dims)
                .map(|axis| {
                    (particles.positions[index][axis] - config.domain_min[axis])
                        .min(config.domain_max[axis] - particles.positions[index][axis])
                })
                .fold(f32::INFINITY, f32::min)
                .max(0.0);
            let footprint = particles.footprint(index);
            let spacing = perception.observed_spacing[index].max(f32::MIN_POSITIVE);
            let degree = perception.accepted_degree[index] as f32;
            let mut features = [0.0; ADAPTIVE_CONTROLLER_INPUT_DIMS];
            features[..ADAPTIVE_CONTROLLER_SCALAR_DIMS].copy_from_slice(&[
                (variation[index] / mean_variation).ln_1p(),
                (boundary_distance / config.reference_footprint).ln_1p(),
                ((config.perception.spacing_target_neighbors + 1.0) / (degree + 1.0)).ln(),
                (spacing / footprint).ln(),
                (footprint / config.reference_footprint).ln(),
                (particles.bandwidth[index] / spacing).ln(),
                degree / config.perception.max_neighbors as f32,
                particles.cooldown[index] as f32 / config.cooldown_steps.max(1) as f32,
            ]);
            let perception_row =
                &perception.features[index * perception_dims..(index + 1) * perception_dims];
            let update_row = &base_update[index * update_dims..(index + 1) * update_dims];
            for (channel, value) in perception_row
                .iter()
                .take(copied_perception_dims)
                .enumerate()
            {
                features[ADAPTIVE_CONTROLLER_SCALAR_DIMS + channel] =
                    ((*value - context_mean[channel]) / context_variance[channel]).clamp(-8.0, 8.0);
            }
            for (channel, value) in update_row.iter().take(copied_update_dims).enumerate() {
                let context_channel = copied_perception_dims + channel;
                features[ADAPTIVE_CONTROLLER_SCALAR_DIMS + context_channel] =
                    ((*value - context_mean[context_channel]) / context_variance[context_channel])
                        .clamp(-8.0, 8.0);
            }
            features
        })
        .collect()
}

pub(crate) fn state_variation(
    particles: &AdaptiveParticleSet,
    perception: &AdaptivePerceptionOutput,
) -> Vec<f32> {
    let state_dims = particles.state_dims;
    (0..particles.len())
        .map(|index| {
            let gradient_base = index * state_dims * particles.spatial_dims;
            perception.state_gradient
                [gradient_base..gradient_base + state_dims * particles.spatial_dims]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt()
        })
        .collect()
}

/// Combines the instantaneous rule update and intensive state used to score
/// conservative hierarchy cuts. Training and seeded deployment must use the
/// same coordinates or the learned multiscale rule is evaluated on a different
/// material distribution than the one it saw during fitting.
pub(crate) fn material_detail_values(
    fine: &AdaptiveParticleSet,
    raw_update: &[f32],
    output_dims: usize,
    position_scale: f32,
) -> Vec<f32> {
    debug_assert_eq!(raw_update.len(), fine.len() * output_dims);
    debug_assert!(position_scale.is_finite() && position_scale > 0.0);
    let detail_dims = output_dims + fine.state_dims + fine.spatial_dims;
    let mut values = Vec::with_capacity(fine.len() * detail_dims);
    for row in 0..fine.len() {
        values.extend_from_slice(&raw_update[row * output_dims..(row + 1) * output_dims]);
        values.extend_from_slice(&fine.states[row * fine.state_dims..(row + 1) * fine.state_dims]);
        values.extend(
            fine.positions[row][..fine.spatial_dims]
                .iter()
                .map(|value| *value * position_scale),
        );
    }
    values
}

pub(crate) fn local_residual_features<'a>(
    config: &AdaptiveNpaConfig,
    particles: &AdaptiveParticleSet,
    perception: &'a AdaptivePerceptionOutput,
) -> AutomataResult<Cow<'a, [f32]>> {
    if !config.closure_moment_features && !config.compatible_residual_material_features {
        return Ok(Cow::Borrowed(&perception.features));
    }
    let input_dims =
        perception.feature_dims + local_residual_auxiliary_dims(config, particles.state_dims);
    let closure_neighbor_context = closure_neighbor_context(config, particles)?;
    let mut features = Vec::with_capacity(particles.len() * input_dims);
    for row in 0..particles.len() {
        append_local_residual_feature_row(
            config,
            particles,
            perception,
            row,
            closure_neighbor_context.as_deref(),
            &mut features,
        );
    }
    Ok(Cow::Owned(features))
}

pub(crate) fn local_residual_features_for_rows(
    config: &AdaptiveNpaConfig,
    particles: &AdaptiveParticleSet,
    perception: &AdaptivePerceptionOutput,
    rows: &[usize],
) -> AutomataResult<Vec<f32>> {
    let input_dims =
        perception.feature_dims + local_residual_auxiliary_dims(config, particles.state_dims);
    let closure_neighbor_context = closure_neighbor_context(config, particles)?;
    let mut features = Vec::with_capacity(rows.len() * input_dims);
    for &row in rows {
        debug_assert!(row < particles.len());
        append_local_residual_feature_row(
            config,
            particles,
            perception,
            row,
            closure_neighbor_context.as_deref(),
            &mut features,
        );
    }
    Ok(features)
}

fn append_local_residual_feature_row(
    config: &AdaptiveNpaConfig,
    particles: &AdaptiveParticleSet,
    perception: &AdaptivePerceptionOutput,
    row: usize,
    closure_neighbor_context: Option<&[f32]>,
    features: &mut Vec<f32>,
) {
    let dim = particles.spatial_dims;
    let jacobian_dims = particles.state_dims * dim;
    features.extend_from_slice(
        &perception.features[row * perception.feature_dims..(row + 1) * perception.feature_dims],
    );
    if config.compatible_residual_material_features {
        let native_footprint = config.base_rule_footprint().max(f32::MIN_POSITIVE);
        features.push((particles.footprint(row) / native_footprint - 1.0).clamp(-0.75, 3.0));
        features.push(perception.coarse_exposure[row].clamp(0.0, 1.0));
    }
    if !config.closure_moment_features {
        return;
    }
    let footprint = particles.footprint(row);
    features.push(
        ((footprint / config.reference_footprint).ln() / std::f32::consts::LN_2).clamp(-3.0, 3.0),
    );
    let footprint2 = footprint.powi(2).max(f32::MIN_POSITIVE);
    let covariance = particles.covariance[row];
    for lhs in 0..dim {
        for rhs in lhs..dim {
            features.push((covariance[lhs * 3 + rhs] / footprint2).clamp(-8.0, 8.0));
        }
    }
    let jacobian = &particles.state_jacobian[row * jacobian_dims..(row + 1) * jacobian_dims];
    features.extend(jacobian.iter().map(|value| {
        let scaled = *value * footprint;
        scaled.signum() * scaled.abs().ln_1p().min(8.0)
    }));
    if config.closure_recurrent_mode {
        if particles.closure_basis.is_empty() {
            features.extend(std::iter::repeat_n(0.0, 4));
        } else {
            features.extend_from_slice(&particles.closure_basis[row * 4..(row + 1) * 4]);
        }
        if particles.closure_phase.is_empty() {
            features.extend_from_slice(&[0.0; 2]);
        } else {
            features.extend_from_slice(&particles.closure_phase[row * 2..(row + 1) * 2]);
        }
        if particles.closure_mode.is_empty() {
            features.extend(std::iter::repeat_n(0.0, particles.state_dims));
        } else {
            features.extend_from_slice(
                &particles.closure_mode
                    [row * particles.state_dims..(row + 1) * particles.state_dims],
            );
        }
        let context_dims = closure_neighbor_context_dims(config, particles.state_dims);
        let context = closure_neighbor_context
            .expect("recurrent closure context is prepared with recurrent closure features");
        features.extend_from_slice(&context[row * context_dims..(row + 1) * context_dims]);
    }
}

pub(crate) fn closure_neighbor_context_dims(
    config: &AdaptiveNpaConfig,
    state_dims: usize,
) -> usize {
    usize::from(config.closure_recurrent_mode) * (state_dims + 6)
}

pub(crate) fn local_residual_auxiliary_dims(
    config: &AdaptiveNpaConfig,
    state_dims: usize,
) -> usize {
    usize::from(config.compatible_residual_material_features) * 2
        + usize::from(config.closure_moment_features)
            * (1 + config.spatial_dims * (config.spatial_dims + 1) / 2
                + state_dims * config.spatial_dims
                + usize::from(config.closure_recurrent_mode) * (state_dims + 6)
                + closure_neighbor_context_dims(config, state_dims))
}

pub(crate) fn closure_recurrent_auxiliary_dims(
    config: &AdaptiveNpaConfig,
    state_dims: usize,
) -> usize {
    if !config.closure_recurrent_mode {
        return 0;
    }
    local_residual_auxiliary_dims(config, state_dims)
}

pub(crate) fn closure_recurrent_features_for_rows(
    config: &AdaptiveNpaConfig,
    particles: &AdaptiveParticleSet,
    base_perception: &AdaptivePerceptionOutput,
    rows: &[usize],
) -> AutomataResult<Vec<f32>> {
    local_residual_features_for_rows(config, particles, base_perception, rows)
}

fn closure_neighbor_context(
    config: &AdaptiveNpaConfig,
    particles: &AdaptiveParticleSet,
) -> AutomataResult<Option<Vec<f32>>> {
    let context_dims = closure_neighbor_context_dims(config, particles.state_dims);
    if context_dims == 0 {
        return Ok(None);
    }
    if particles.is_empty() {
        return Ok(Some(Vec::new()));
    }

    let mut fields = Vec::with_capacity(particles.len() * context_dims);
    for row in 0..particles.len() {
        if particles.closure_basis.is_empty() {
            fields.extend_from_slice(&[0.0; 4]);
        } else {
            fields.extend_from_slice(&particles.closure_basis[row * 4..(row + 1) * 4]);
        }
        if particles.closure_phase.is_empty() {
            fields.extend_from_slice(&[0.0; 2]);
        } else {
            fields.extend_from_slice(&particles.closure_phase[row * 2..(row + 1) * 2]);
        }
        if particles.closure_mode.is_empty() {
            fields.extend(std::iter::repeat_n(0.0, particles.state_dims));
        } else {
            fields.extend_from_slice(
                &particles.closure_mode
                    [row * particles.state_dims..(row + 1) * particles.state_dims],
            );
        }
    }

    let transported = adaptive_perceive_without_spacing(
        &particles.positions,
        &fields,
        &particles.represented_measure,
        &particles.bandwidth,
        1,
        particles.len(),
        context_dims,
        config.perception,
    )?;
    Ok(Some(transported.normalized_state))
}

pub(crate) fn local_detail_risk(
    particles: &AdaptiveParticleSet,
    perception: &AdaptivePerceptionOutput,
) -> Vec<f32> {
    let state_detail = state_variation(particles, perception);
    let occupancy_detail = perception
        .occupancy_gradient
        .chunks_exact(particles.spatial_dims)
        .map(|gradient| {
            gradient
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt()
        })
        .collect::<Vec<_>>();
    let state_mean = mean(&state_detail).max(1.0e-6);
    let occupancy_mean = mean(&occupancy_detail).max(1.0e-6);
    state_detail
        .into_iter()
        .zip(occupancy_detail)
        .map(|(state, occupancy)| {
            (0.25 * state / state_mean + occupancy / occupancy_mean).max(1.0e-6)
        })
        .collect()
}

pub(crate) fn proxy_context(
    config: &AdaptiveNpaConfig,
    particles: &AdaptiveParticleSet,
) -> AutomataResult<Option<AdaptiveProxyContext>> {
    if !config.proxy.enabled {
        return Ok(None);
    }
    let hierarchy = AdaptiveProxyHierarchy::build(particles, config.proxy.branch_factor)?;
    let level_index = config
        .proxy
        .level
        .saturating_sub(1)
        .min(hierarchy.levels.len().saturating_sub(1));
    let Some(level) = hierarchy.levels.get(level_index) else {
        return Ok(None);
    };
    let mut positions = Vec::with_capacity(level.len());
    let mut states = Vec::with_capacity(level.len() * particles.state_dims);
    let mut measure = Vec::with_capacity(level.len());
    let mut bandwidth = Vec::with_capacity(level.len());
    for index in level {
        let node = &hierarchy.nodes[*index];
        positions.push(node.position);
        states.extend_from_slice(&node.state);
        measure.push(node.represented_measure);
        bandwidth.push(
            (node.bounding_radius * config.proxy.bandwidth_scale)
                .max(node.max_bandwidth)
                .clamp(
                    config.perception.min_bandwidth,
                    config.perception.max_bandwidth,
                ),
        );
    }
    let mut perception_config = config.perception;
    if perception_config.graph_policy == AdaptiveGraphPolicy::MutualTopK {
        perception_config.graph_policy = AdaptiveGraphPolicy::DirectedTopK;
    }
    let perception = adaptive_proxy_perceive(
        &particles.positions,
        &particles.states,
        &particles.bandwidth,
        &positions,
        &states,
        &measure,
        &bandwidth,
        particles.state_dims,
        perception_config,
    )?;
    Ok(Some(AdaptiveProxyContext {
        perception,
        node_count: level.len(),
    }))
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len().max(1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        NpaConfig, NpaModel, ParticleSeed,
        adaptive::{AdaptiveLocalRuleSemantics, perception::rule_perception_pair},
        rollout::seed_particles_scaled,
    };

    #[test]
    fn local_rule_training_contract_selects_the_deployed_stream_and_gate() {
        let rule = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 7);
        let mut config = AdaptiveNpaConfig::growing_2d();
        config.rule_graph_policy = AdaptiveGraphPolicy::RawSupport;
        config.local_rule_semantics = AdaptiveLocalRuleSemantics::CompatibleResidual;
        let (positions, states) = seed_particles_scaled(
            1,
            16,
            rule.config.state_dims,
            rule.config.spatial_dims,
            11,
            ParticleSeed::UniformCircle,
            0.2,
        );
        let mut particles = AdaptiveParticleSet::from_equal_measure(
            positions,
            states,
            rule.config.spatial_dims,
            rule.config.state_dims,
            1.0,
            0.1,
        )
        .unwrap();
        particles.represented_measure[0] *= 4.0;
        let pair = rule_perception_pair(&config, &rule, &particles).unwrap();

        let selected = local_rule_perception(&config, &pair);
        assert!(std::ptr::eq(selected, &pair.npa_compatible));
        assert_eq!(
            local_residual_gate(&config, &particles, selected, 0),
            pair.npa_compatible.coarse_exposure[0].max(0.0),
        );

        config.local_rule_semantics = AdaptiveLocalRuleSemantics::Residual;
        let selected = local_rule_perception(&config, &pair);
        assert!(std::ptr::eq(selected, &pair.normalized));
        assert_eq!(
            local_residual_gate(&config, &particles, selected, 0),
            config.residual_gate(particles.footprint(0)).max(0.0),
        );
    }
}
