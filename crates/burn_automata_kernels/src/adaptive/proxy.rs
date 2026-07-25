use rayon::prelude::*;

use super::{
    AdaptiveGraphMetrics, AdaptiveGraphPolicy, AdaptivePerceptionConfig, AdaptivePerceptionOutput,
    perception::{
        kernel_gradient, kernel_value, log_normalize, pair_bandwidth, percentile_usize,
        regularized_inverse,
    },
};
use crate::{KernelError, KernelResult};

#[derive(Clone, Copy, Debug)]
struct ProxyCandidate {
    index: usize,
    delta: [f32; 3],
    distance2: f32,
    pair_bandwidth: f32,
    normalized_distance2: f32,
}

#[derive(Debug)]
struct ProxyRow {
    features: Vec<f32>,
    normalized_state: Vec<f32>,
    state_gradient: Vec<f32>,
    occupancy_gradient: Vec<f32>,
    partition: f32,
    moment_condition: f32,
    moment_fallback: bool,
    raw_degree: usize,
    accepted_degree: usize,
}

/// Evaluates receiver-local context from a disjoint nonmaterial proxy row.
///
/// Proxy sources represent a conservative partition of material leaves, but
/// are not themselves updated or rendered. The result retains the normalized
/// adaptive feature layout so a separate context rule can consume it.
#[allow(clippy::too_many_arguments)]
pub fn adaptive_proxy_perceive(
    target_positions: &[[f32; 4]],
    target_states: &[f32],
    target_bandwidth: &[f32],
    proxy_positions: &[[f32; 4]],
    proxy_states: &[f32],
    proxy_measure: &[f32],
    proxy_bandwidth: &[f32],
    state_dims: usize,
    cfg: AdaptivePerceptionConfig,
) -> KernelResult<AdaptivePerceptionOutput> {
    validate(
        target_positions,
        target_states,
        target_bandwidth,
        proxy_positions,
        proxy_states,
        proxy_measure,
        proxy_bandwidth,
        state_dims,
        cfg,
    )?;
    let rows = (0..target_positions.len())
        .into_par_iter()
        .map(|index| {
            perceive_target(
                index,
                target_positions,
                target_states,
                target_bandwidth,
                proxy_positions,
                proxy_states,
                proxy_measure,
                proxy_bandwidth,
                state_dims,
                cfg,
            )
        })
        .collect::<Vec<_>>();
    let total = rows.len();
    let feature_dims = cfg.feature_dims(state_dims);
    let mut output = AdaptivePerceptionOutput {
        features: Vec::with_capacity(total * feature_dims),
        normalized_state: Vec::with_capacity(total * state_dims),
        state_gradient: Vec::with_capacity(total * state_dims * cfg.dim),
        occupancy_gradient: Vec::with_capacity(total * cfg.dim),
        partition: Vec::with_capacity(total),
        coarse_exposure: vec![0.0; total],
        observed_spacing: target_bandwidth.to_vec(),
        moment_condition: Vec::with_capacity(total),
        moment_fallback: Vec::with_capacity(total),
        accepted_degree: Vec::with_capacity(total),
        graph: AdaptiveGraphMetrics::default(),
        feature_dims,
    };
    let mut raw_messages = 0;
    for row in rows {
        output.features.extend(row.features);
        output.normalized_state.extend(row.normalized_state);
        output.state_gradient.extend(row.state_gradient);
        output.occupancy_gradient.extend(row.occupancy_gradient);
        output.partition.push(row.partition);
        output.moment_condition.push(row.moment_condition);
        output.moment_fallback.push(row.moment_fallback);
        output.accepted_degree.push(row.accepted_degree);
        raw_messages += row.raw_degree;
    }
    let accepted_messages = output.accepted_degree.iter().sum::<usize>();
    let mut degrees = output.accepted_degree.clone();
    degrees.sort_unstable();
    output.graph = AdaptiveGraphMetrics {
        candidate_visits: raw_messages,
        raw_messages,
        accepted_messages,
        degree_mean: accepted_messages as f32 / total.max(1) as f32,
        degree_p95: percentile_usize(&degrees, 0.95),
        degree_max: degrees.last().copied().unwrap_or_default(),
        isolated_particles: degrees.iter().filter(|degree| **degree == 0).count(),
        cross_scale_fraction: 1.0,
    };
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn perceive_target(
    target: usize,
    target_positions: &[[f32; 4]],
    target_states: &[f32],
    target_bandwidth: &[f32],
    proxy_positions: &[[f32; 4]],
    proxy_states: &[f32],
    proxy_measure: &[f32],
    proxy_bandwidth: &[f32],
    state_dims: usize,
    cfg: AdaptivePerceptionConfig,
) -> ProxyRow {
    let mut candidates = proxy_positions
        .iter()
        .enumerate()
        .filter_map(|(index, position)| {
            let mut delta = [0.0; 3];
            let mut distance2 = 0.0;
            for axis in 0..cfg.dim {
                delta[axis] = position[axis] - target_positions[target][axis];
                distance2 += delta[axis] * delta[axis];
            }
            let pair_bandwidth = pair_bandwidth(
                target_bandwidth[target],
                proxy_bandwidth[index],
                cfg.pair_scale_power,
            );
            (distance2 < pair_bandwidth * pair_bandwidth).then_some(ProxyCandidate {
                index,
                delta,
                distance2,
                pair_bandwidth,
                normalized_distance2: distance2 / (pair_bandwidth * pair_bandwidth),
            })
        })
        .collect::<Vec<_>>();
    let raw_degree = candidates.len();
    if cfg.graph_policy != AdaptiveGraphPolicy::RawSupport && candidates.len() > cfg.max_neighbors {
        candidates.select_nth_unstable_by(cfg.max_neighbors, candidate_order);
        candidates.truncate(cfg.max_neighbors);
    }
    candidates.sort_unstable_by(candidate_order);
    let accepted_degree = candidates.len();

    let state_base = target * state_dims;
    let state_i = &target_states[state_base..state_base + state_dims];
    let mut denominator = cfg.shepard_epsilon;
    let mut numerator = state_i
        .iter()
        .map(|value| cfg.shepard_epsilon * value)
        .collect::<Vec<_>>();
    let mut moment = [0.0_f32; 9];
    let mut gradient_rhs = vec![0.0; state_dims * cfg.dim];
    let mut occupancy_gradient = vec![0.0; cfg.dim];
    for candidate in candidates {
        let source = candidate.index;
        let weight = proxy_measure[source]
            * kernel_value(candidate.distance2, candidate.pair_bandwidth, cfg.dim);
        denominator += weight;
        for channel in 0..state_dims {
            numerator[channel] += weight * proxy_states[source * state_dims + channel];
        }
        let gradient = kernel_gradient(
            candidate.delta,
            candidate.distance2,
            candidate.pair_bandwidth,
            cfg.dim,
        );
        for axis in 0..cfg.dim {
            let weighted_gradient = proxy_measure[source] * gradient[axis];
            occupancy_gradient[axis] += weighted_gradient;
            for col in 0..cfg.dim {
                moment[axis * cfg.dim + col] += weighted_gradient * candidate.delta[col];
            }
            for channel in 0..state_dims {
                let difference = proxy_states[source * state_dims + channel] - state_i[channel];
                gradient_rhs[channel * cfg.dim + axis] += difference * weighted_gradient;
            }
        }
    }
    let inverse_denominator = denominator.recip();
    let normalized_state = numerator
        .into_iter()
        .map(|value| value * inverse_denominator)
        .collect::<Vec<_>>();
    occupancy_gradient
        .iter_mut()
        .for_each(|value| *value *= inverse_denominator);
    let (inverse, moment_condition, moment_fallback) = regularized_inverse(moment, cfg);
    let mut state_gradient = vec![0.0; state_dims * cfg.dim];
    for channel in 0..state_dims {
        for output_axis in 0..cfg.dim {
            for input_axis in 0..cfg.dim {
                state_gradient[channel * cfg.dim + output_axis] += inverse
                    [output_axis * cfg.dim + input_axis]
                    * gradient_rhs[channel * cfg.dim + input_axis];
            }
        }
    }
    for channel in 0..state_dims {
        let gradient = &mut state_gradient[channel * cfg.dim..(channel + 1) * cfg.dim];
        gradient
            .iter_mut()
            .for_each(|value| *value *= target_bandwidth[target]);
        if cfg.log_normalize_gradients {
            log_normalize(gradient);
        }
    }
    occupancy_gradient
        .iter_mut()
        .for_each(|value| *value *= target_bandwidth[target]);
    if cfg.log_normalize_gradients {
        log_normalize(&mut occupancy_gradient);
    }
    let mut features = Vec::with_capacity(cfg.feature_dims(state_dims));
    features.extend_from_slice(state_i);
    features.extend_from_slice(&normalized_state);
    features.extend_from_slice(&state_gradient);
    features.extend_from_slice(&occupancy_gradient);
    if cfg.include_position_features {
        features.extend_from_slice(&target_positions[target][..cfg.dim]);
    }
    ProxyRow {
        features,
        normalized_state,
        state_gradient,
        occupancy_gradient,
        partition: denominator,
        moment_condition,
        moment_fallback,
        raw_degree,
        accepted_degree,
    }
}

fn candidate_order(lhs: &ProxyCandidate, rhs: &ProxyCandidate) -> std::cmp::Ordering {
    lhs.normalized_distance2
        .total_cmp(&rhs.normalized_distance2)
        .then_with(|| lhs.index.cmp(&rhs.index))
}

#[allow(clippy::too_many_arguments)]
fn validate(
    target_positions: &[[f32; 4]],
    target_states: &[f32],
    target_bandwidth: &[f32],
    proxy_positions: &[[f32; 4]],
    proxy_states: &[f32],
    proxy_measure: &[f32],
    proxy_bandwidth: &[f32],
    state_dims: usize,
    cfg: AdaptivePerceptionConfig,
) -> KernelResult<()> {
    cfg.validate()?;
    if target_positions.is_empty()
        || proxy_positions.is_empty()
        || state_dims == 0
        || target_states.len() != target_positions.len() * state_dims
        || target_bandwidth.len() != target_positions.len()
        || proxy_states.len() != proxy_positions.len() * state_dims
        || proxy_measure.len() != proxy_positions.len()
        || proxy_bandwidth.len() != proxy_positions.len()
    {
        return Err(KernelError::InvalidArgument(
            "adaptive proxy perception shape mismatch".to_string(),
        ));
    }
    if cfg.graph_policy == AdaptiveGraphPolicy::MutualTopK {
        return Err(KernelError::InvalidArgument(
            "mutual top-k is undefined between distinct leaf and proxy rows".to_string(),
        ));
    }
    let finite_positions = target_positions
        .iter()
        .chain(proxy_positions)
        .flat_map(|position| position.iter().take(cfg.dim))
        .all(|value| value.is_finite());
    if !finite_positions
        || target_states
            .iter()
            .chain(proxy_states)
            .any(|value| !value.is_finite())
        || proxy_measure
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || target_bandwidth.iter().chain(proxy_bandwidth).any(|value| {
            !value.is_finite() || *value < cfg.min_bandwidth || *value > cfg.max_bandwidth
        })
    {
        return Err(KernelError::InvalidArgument(
            "adaptive proxy inputs must be finite and in range".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proxy_context_reproduces_constant_states() {
        let targets = vec![[0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]];
        let proxies = vec![[-0.05, 0.0, 0.0, 0.0], [0.15, 0.0, 0.0, 0.0]];
        let output = adaptive_proxy_perceive(
            &targets,
            &[2.0, -1.0, 2.0, -1.0],
            &[0.3; 2],
            &proxies,
            &[2.0, -1.0, 2.0, -1.0],
            &[0.5; 2],
            &[0.3; 2],
            2,
            AdaptivePerceptionConfig {
                min_bandwidth: 0.05,
                max_bandwidth: 0.5,
                graph_policy: AdaptiveGraphPolicy::RawSupport,
                ..AdaptivePerceptionConfig::growing_2d()
            },
        )
        .unwrap();
        for state in output.normalized_state.chunks_exact(2) {
            assert!((state[0] - 2.0).abs() < 1.0e-6);
            assert!((state[1] + 1.0).abs() < 1.0e-6);
        }
    }
}
