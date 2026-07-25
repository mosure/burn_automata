use super::{AdaptiveNpaConfig, AdaptiveParticleSet};
use crate::{AutomataResult, NpaModel};
use burn_automata_kernels::{
    AdaptiveNpaPerceptionOptions, AdaptivePerceptionOutput, AdaptivePerceptionPair,
    adaptive_npa_perceive_without_spacing, adaptive_perceive_pair,
};

pub(crate) fn rule_perception_pair(
    config: &AdaptiveNpaConfig,
    rule: &NpaModel,
    particles: &AdaptiveParticleSet,
) -> AutomataResult<AdaptivePerceptionPair> {
    let mut pair = adaptive_perceive_pair(
        &particles.positions,
        &particles.states,
        &particles.represented_measure,
        &particles.bandwidth,
        1,
        particles.len(),
        particles.state_dims,
        config.perception,
        config.rule_graph_policy,
        npa_options(rule),
    )?;
    inject_retained_state_jacobian(config, particles, &mut pair.normalized);
    Ok(pair)
}

/// The normalized residual branch is responsible for unresolved within-leaf
/// dynamics. For genuinely coarse leaves, replace its neighbor-only gradient
/// estimate with the conservative Jacobian fitted during restriction/merge.
/// Native leaves retain byte-for-byte perception semantics.
fn inject_retained_state_jacobian(
    config: &AdaptiveNpaConfig,
    particles: &AdaptiveParticleSet,
    perception: &mut AdaptivePerceptionOutput,
) {
    let dim = particles.spatial_dims;
    let row_dims = particles.state_dims * dim;
    let feature_gradient_start = particles.state_dims * 2;
    let native_footprint = config.base_rule_footprint();
    for row in 0..particles.len() {
        if particles.footprint(row) <= native_footprint * 1.5 {
            continue;
        }
        let retained = &particles.state_jacobian[row * row_dims..(row + 1) * row_dims];
        if retained.iter().map(|value| value * value).sum::<f32>() <= 1.0e-12 {
            continue;
        }
        let mut encoded = retained.to_vec();
        for channel in 0..particles.state_dims {
            let gradient = &mut encoded[channel * dim..(channel + 1) * dim];
            gradient
                .iter_mut()
                .for_each(|value| *value *= particles.bandwidth[row]);
            if config.perception.log_normalize_gradients {
                log_normalize(gradient);
            }
        }
        perception.state_gradient[row * row_dims..(row + 1) * row_dims].copy_from_slice(&encoded);
        let feature_base = row * perception.feature_dims + feature_gradient_start;
        perception.features[feature_base..feature_base + row_dims].copy_from_slice(&encoded);
    }
}

fn log_normalize(values: &mut [f32]) {
    let norm = (values.iter().map(|value| value * value).sum::<f32>() + 1.0e-12).sqrt();
    let scale = norm.ln_1p() / norm.max(1.0e-6);
    values.iter_mut().for_each(|value| *value *= scale);
}

pub(crate) fn rule_perception_without_spacing(
    config: &AdaptiveNpaConfig,
    rule: &NpaModel,
    particles: &AdaptiveParticleSet,
) -> AutomataResult<AdaptivePerceptionOutput> {
    let mut perception = config.perception;
    perception.graph_policy = config.rule_graph_policy;
    Ok(adaptive_npa_perceive_without_spacing(
        &particles.positions,
        &particles.states,
        &particles.represented_measure,
        &particles.bandwidth,
        1,
        particles.len(),
        particles.state_dims,
        perception,
        npa_options(rule),
    )?)
}

pub(crate) fn decode_physical_state_gradient(
    encoded: &[f32],
    state_dims: usize,
    spatial_dims: usize,
    bandwidth: f32,
    log_normalized: bool,
) -> Vec<f32> {
    let mut decoded = encoded.to_vec();
    for channel in 0..state_dims {
        let row = &mut decoded[channel * spatial_dims..(channel + 1) * spatial_dims];
        let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
        let physical_norm =
            if log_normalized { norm.exp_m1() } else { norm } / bandwidth.max(f32::MIN_POSITIVE);
        let scale = physical_norm / norm.max(f32::MIN_POSITIVE);
        row.iter_mut().for_each(|value| *value *= scale);
    }
    decoded
}

pub(crate) fn npa_options(rule: &NpaModel) -> AdaptiveNpaPerceptionOptions {
    AdaptiveNpaPerceptionOptions {
        eps0: rule.config.eps0,
        scale_equivariance: rule.config.scale_equivariant(),
        particle_density_equivariance: rule.config.particle_density_equivariant(),
        log_norm_grad: rule.config.log_norm_grad,
        log_norm_density_grad: rule.config.log_norm_density_grad,
        position_features: rule.config.position_features,
    }
}
