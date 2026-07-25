use super::{
    AdaptiveNpaPerceptionOptions, AdaptivePerceptionConfig,
    compatible::{poly6_kernel, safe_inverse_symmetric, spiky_gradient},
    perception::{
        ParticleNeighbors, build_neighborhoods, kernel_gradient, kernel_value, regularized_inverse,
        validate_inputs,
    },
};
use crate::{KernelError, KernelResult};

/// State-only adjoint of [`super::adaptive_npa_perceive`].
///
/// Positions, represented measure, bandwidth, and graph selection are treated as
/// detached material/topology data. This is the backward contract used by the
/// Growing-NPA trainer, where particle positions are stop-gradient.
#[allow(clippy::too_many_arguments)]
pub fn adaptive_npa_perceive_state_adjoint(
    positions: &[[f32; 4]],
    states: &[f32],
    represented_measure: &[f32],
    bandwidth: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    config: AdaptivePerceptionConfig,
    options: AdaptiveNpaPerceptionOptions,
    feature_adjoint: &[f32],
) -> KernelResult<Vec<f32>> {
    adaptive_npa_perceive_state_adjoint_impl(
        positions,
        states,
        represented_measure,
        bandwidth,
        batch_size,
        particle_count,
        state_dims,
        config,
        options,
        feature_adjoint,
        true,
    )
}

/// Deterministic all-pairs state-adjoint oracle for NPA-compatible perception.
#[allow(clippy::too_many_arguments)]
pub fn adaptive_npa_perceive_state_adjoint_all_pairs(
    positions: &[[f32; 4]],
    states: &[f32],
    represented_measure: &[f32],
    bandwidth: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    config: AdaptivePerceptionConfig,
    options: AdaptiveNpaPerceptionOptions,
    feature_adjoint: &[f32],
) -> KernelResult<Vec<f32>> {
    adaptive_npa_perceive_state_adjoint_impl(
        positions,
        states,
        represented_measure,
        bandwidth,
        batch_size,
        particle_count,
        state_dims,
        config,
        options,
        feature_adjoint,
        false,
    )
}

/// State-only adjoint of the normalized adaptive perception operator.
#[allow(clippy::too_many_arguments)]
pub fn adaptive_perceive_state_adjoint(
    positions: &[[f32; 4]],
    states: &[f32],
    represented_measure: &[f32],
    bandwidth: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    config: AdaptivePerceptionConfig,
    feature_adjoint: &[f32],
) -> KernelResult<Vec<f32>> {
    adaptive_perceive_state_adjoint_impl(
        positions,
        states,
        represented_measure,
        bandwidth,
        batch_size,
        particle_count,
        state_dims,
        config,
        feature_adjoint,
        true,
    )
}

/// Deterministic all-pairs state-adjoint oracle for normalized perception.
#[allow(clippy::too_many_arguments)]
pub fn adaptive_perceive_state_adjoint_all_pairs(
    positions: &[[f32; 4]],
    states: &[f32],
    represented_measure: &[f32],
    bandwidth: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    config: AdaptivePerceptionConfig,
    feature_adjoint: &[f32],
) -> KernelResult<Vec<f32>> {
    adaptive_perceive_state_adjoint_impl(
        positions,
        states,
        represented_measure,
        bandwidth,
        batch_size,
        particle_count,
        state_dims,
        config,
        feature_adjoint,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn adaptive_npa_perceive_state_adjoint_impl(
    positions: &[[f32; 4]],
    states: &[f32],
    represented_measure: &[f32],
    bandwidth: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    config: AdaptivePerceptionConfig,
    options: AdaptiveNpaPerceptionOptions,
    feature_adjoint: &[f32],
    spatial_hash: bool,
) -> KernelResult<Vec<f32>> {
    validate_adjoint_inputs(
        positions,
        states,
        represented_measure,
        bandwidth,
        batch_size,
        particle_count,
        state_dims,
        config,
        feature_adjoint,
    )?;
    options.validate()?;
    let neighborhoods =
        build_neighborhoods(positions, bandwidth, particle_count, config, spatial_hash);
    let density = npa_density(represented_measure, bandwidth, config, &neighborhoods);
    let mut state_adjoint = vec![0.0; states.len()];

    for (index, neighbors) in neighborhoods.iter().enumerate() {
        let feature_dims = config.feature_dims(state_dims);
        let feature_base = index * feature_dims;
        let state_base = index * state_dims;
        let mut cursor = feature_base;

        for channel in 0..state_dims {
            state_adjoint[state_base + channel] += feature_adjoint[cursor + channel];
        }
        cursor += state_dims;

        let blurred_adjoint = &feature_adjoint[cursor..cursor + state_dims];
        let self_weight = represented_measure[index] / density[index].max(f32::MIN_POSITIVE)
            * poly6_kernel(0.0, bandwidth[index], config.dim);
        add_scaled_state_adjoint(
            &mut state_adjoint,
            index,
            state_dims,
            blurred_adjoint,
            self_weight,
        );
        for candidate in &neighbors.candidates {
            let source = candidate.index;
            let weight = represented_measure[source] / density[source].max(f32::MIN_POSITIVE)
                * poly6_kernel(candidate.distance2, candidate.pair_bandwidth, config.dim);
            add_scaled_state_adjoint(
                &mut state_adjoint,
                source,
                state_dims,
                blurred_adjoint,
                weight,
            );
        }
        cursor += state_dims;

        let gradient_len = state_dims * config.dim;
        accumulate_npa_gradient_adjoint(
            index,
            states,
            represented_measure,
            bandwidth,
            &density,
            state_dims,
            config,
            options,
            neighbors,
            &feature_adjoint[cursor..cursor + gradient_len],
            &mut state_adjoint,
        );
    }
    Ok(state_adjoint)
}

#[allow(clippy::too_many_arguments)]
fn adaptive_perceive_state_adjoint_impl(
    positions: &[[f32; 4]],
    states: &[f32],
    represented_measure: &[f32],
    bandwidth: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    config: AdaptivePerceptionConfig,
    feature_adjoint: &[f32],
    spatial_hash: bool,
) -> KernelResult<Vec<f32>> {
    validate_adjoint_inputs(
        positions,
        states,
        represented_measure,
        bandwidth,
        batch_size,
        particle_count,
        state_dims,
        config,
        feature_adjoint,
    )?;
    let neighborhoods =
        build_neighborhoods(positions, bandwidth, particle_count, config, spatial_hash);
    let mut state_adjoint = vec![0.0; states.len()];

    for (index, neighbors) in neighborhoods.iter().enumerate() {
        let feature_dims = config.feature_dims(state_dims);
        let feature_base = index * feature_dims;
        let state_base = index * state_dims;
        let mut cursor = feature_base;

        for channel in 0..state_dims {
            state_adjoint[state_base + channel] += feature_adjoint[cursor + channel];
        }
        cursor += state_dims;

        let self_kernel = kernel_value(0.0, bandwidth[index], config.dim);
        let denominator = config.shepard_epsilon
            + represented_measure[index] * self_kernel
            + neighbors
                .candidates
                .iter()
                .map(|candidate| {
                    represented_measure[candidate.index]
                        * kernel_value(candidate.distance2, candidate.pair_bandwidth, config.dim)
                })
                .sum::<f32>();
        let normalized_adjoint = &feature_adjoint[cursor..cursor + state_dims];
        let self_weight =
            (config.shepard_epsilon + represented_measure[index] * self_kernel) / denominator;
        add_scaled_state_adjoint(
            &mut state_adjoint,
            index,
            state_dims,
            normalized_adjoint,
            self_weight,
        );
        for candidate in &neighbors.candidates {
            let weight = represented_measure[candidate.index]
                * kernel_value(candidate.distance2, candidate.pair_bandwidth, config.dim)
                / denominator;
            add_scaled_state_adjoint(
                &mut state_adjoint,
                candidate.index,
                state_dims,
                normalized_adjoint,
                weight,
            );
        }
        cursor += state_dims;

        let gradient_len = state_dims * config.dim;
        accumulate_normalized_gradient_adjoint(
            index,
            states,
            represented_measure,
            bandwidth,
            state_dims,
            config,
            neighbors,
            &feature_adjoint[cursor..cursor + gradient_len],
            &mut state_adjoint,
        );
    }
    Ok(state_adjoint)
}

#[allow(clippy::too_many_arguments)]
fn validate_adjoint_inputs(
    positions: &[[f32; 4]],
    states: &[f32],
    represented_measure: &[f32],
    bandwidth: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    config: AdaptivePerceptionConfig,
    feature_adjoint: &[f32],
) -> KernelResult<()> {
    validate_inputs(
        positions,
        states,
        represented_measure,
        bandwidth,
        batch_size,
        particle_count,
        state_dims,
        config,
    )?;
    let expected = positions.len() * config.feature_dims(state_dims);
    if feature_adjoint.len() != expected {
        return Err(KernelError::OutputShape {
            actual: feature_adjoint.len(),
            expected,
        });
    }
    Ok(())
}

fn npa_density(
    represented_measure: &[f32],
    bandwidth: &[f32],
    config: AdaptivePerceptionConfig,
    neighborhoods: &[ParticleNeighbors],
) -> Vec<f32> {
    neighborhoods
        .iter()
        .enumerate()
        .map(|(index, neighbors)| {
            represented_measure[index] * poly6_kernel(0.0, bandwidth[index], config.dim)
                + neighbors
                    .candidates
                    .iter()
                    .map(|candidate| {
                        represented_measure[candidate.index]
                            * poly6_kernel(
                                candidate.distance2,
                                candidate.pair_bandwidth,
                                config.dim,
                            )
                    })
                    .sum::<f32>()
        })
        .collect()
}

fn add_scaled_state_adjoint(
    state_adjoint: &mut [f32],
    particle: usize,
    state_dims: usize,
    source: &[f32],
    scale: f32,
) {
    let base = particle * state_dims;
    for channel in 0..state_dims {
        state_adjoint[base + channel] += scale * source[channel];
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_npa_gradient_adjoint(
    index: usize,
    states: &[f32],
    represented_measure: &[f32],
    bandwidth: &[f32],
    density: &[f32],
    state_dims: usize,
    config: AdaptivePerceptionConfig,
    options: AdaptiveNpaPerceptionOptions,
    neighbors: &ParticleNeighbors,
    gradient_adjoint: &[f32],
    state_adjoint: &mut [f32],
) {
    let state_base = index * state_dims;
    let mut raw_gradient = vec![0.0; state_dims * config.dim];
    let mut moment = [0.0; 9];
    let volume_gradients = neighbors
        .candidates
        .iter()
        .map(|candidate| {
            let volume = represented_measure[candidate.index]
                / density[candidate.index].max(f32::MIN_POSITIVE);
            let gradient = spiky_gradient(
                candidate.delta,
                candidate.distance2,
                candidate.pair_bandwidth,
                config.dim,
                volume,
            );
            for row in 0..config.dim {
                for col in 0..config.dim {
                    moment[row * config.dim + col] += candidate.delta[row] * gradient[col];
                }
            }
            for channel in 0..state_dims {
                let difference =
                    states[candidate.index * state_dims + channel] - states[state_base + channel];
                for axis in 0..config.dim {
                    raw_gradient[channel * config.dim + axis] += difference * gradient[axis];
                }
            }
            gradient
        })
        .collect::<Vec<_>>();
    let (inverse, _, _) = safe_inverse_symmetric(moment, config.dim);
    let scale = if options.scale_equivariance {
        bandwidth[index] / options.eps0
    } else {
        1.0
    };

    for channel in 0..state_dims {
        let base = channel * config.dim;
        let mut corrected = [0.0; 3];
        for out_axis in 0..config.dim {
            corrected[out_axis] = (0..config.dim)
                .map(|in_axis| {
                    raw_gradient[base + in_axis] * inverse[in_axis * config.dim + out_axis]
                })
                .sum::<f32>();
        }
        let mut normalized_input = [0.0; 3];
        for axis in 0..config.dim {
            normalized_input[axis] = corrected[axis] * scale;
        }
        let mut corrected_adjoint = [0.0; 3];
        if options.log_norm_grad {
            log_normalize_adjoint(
                &normalized_input[..config.dim],
                &gradient_adjoint[base..base + config.dim],
                &mut corrected_adjoint[..config.dim],
            );
        } else {
            corrected_adjoint[..config.dim]
                .copy_from_slice(&gradient_adjoint[base..base + config.dim]);
        }
        for value in corrected_adjoint.iter_mut().take(config.dim) {
            *value *= scale;
        }
        let mut raw_adjoint = [0.0; 3];
        for in_axis in 0..config.dim {
            for out_axis in 0..config.dim {
                raw_adjoint[in_axis] +=
                    corrected_adjoint[out_axis] * inverse[in_axis * config.dim + out_axis];
            }
        }
        for (candidate, volume_gradient) in neighbors.candidates.iter().zip(volume_gradients.iter())
        {
            let contribution = (0..config.dim)
                .map(|axis| raw_adjoint[axis] * volume_gradient[axis])
                .sum::<f32>();
            state_adjoint[candidate.index * state_dims + channel] += contribution;
            state_adjoint[state_base + channel] -= contribution;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_normalized_gradient_adjoint(
    index: usize,
    states: &[f32],
    represented_measure: &[f32],
    bandwidth: &[f32],
    state_dims: usize,
    config: AdaptivePerceptionConfig,
    neighbors: &ParticleNeighbors,
    gradient_adjoint: &[f32],
    state_adjoint: &mut [f32],
) {
    let state_base = index * state_dims;
    let mut raw_gradient = vec![0.0; state_dims * config.dim];
    let mut moment = [0.0; 9];
    let weighted_gradients = neighbors
        .candidates
        .iter()
        .map(|candidate| {
            let mut gradient = kernel_gradient(
                candidate.delta,
                candidate.distance2,
                candidate.pair_bandwidth,
                config.dim,
            );
            for axis in 0..config.dim {
                gradient[axis] *= represented_measure[candidate.index];
                for col in 0..config.dim {
                    moment[axis * config.dim + col] += gradient[axis] * candidate.delta[col];
                }
            }
            for channel in 0..state_dims {
                let difference =
                    states[candidate.index * state_dims + channel] - states[state_base + channel];
                for axis in 0..config.dim {
                    raw_gradient[channel * config.dim + axis] += difference * gradient[axis];
                }
            }
            gradient
        })
        .collect::<Vec<_>>();
    let (inverse, _, _) = regularized_inverse(moment, config);
    let scale = bandwidth[index];

    for channel in 0..state_dims {
        let base = channel * config.dim;
        let mut corrected = [0.0; 3];
        for out_axis in 0..config.dim {
            for in_axis in 0..config.dim {
                corrected[out_axis] +=
                    inverse[out_axis * config.dim + in_axis] * raw_gradient[base + in_axis];
            }
        }
        let mut normalized_input = [0.0; 3];
        for axis in 0..config.dim {
            normalized_input[axis] = corrected[axis] * scale;
        }
        let mut corrected_adjoint = [0.0; 3];
        if config.log_normalize_gradients {
            log_normalize_adjoint(
                &normalized_input[..config.dim],
                &gradient_adjoint[base..base + config.dim],
                &mut corrected_adjoint[..config.dim],
            );
        } else {
            corrected_adjoint[..config.dim]
                .copy_from_slice(&gradient_adjoint[base..base + config.dim]);
        }
        for value in corrected_adjoint.iter_mut().take(config.dim) {
            *value *= scale;
        }
        let mut raw_adjoint = [0.0; 3];
        for in_axis in 0..config.dim {
            for out_axis in 0..config.dim {
                raw_adjoint[in_axis] +=
                    corrected_adjoint[out_axis] * inverse[out_axis * config.dim + in_axis];
            }
        }
        for (candidate, weighted_gradient) in
            neighbors.candidates.iter().zip(weighted_gradients.iter())
        {
            let contribution = (0..config.dim)
                .map(|axis| raw_adjoint[axis] * weighted_gradient[axis])
                .sum::<f32>();
            state_adjoint[candidate.index * state_dims + channel] += contribution;
            state_adjoint[state_base + channel] -= contribution;
        }
    }
}

fn log_normalize_adjoint(input: &[f32], output_adjoint: &[f32], input_adjoint: &mut [f32]) {
    let norm = (input.iter().map(|value| value * value).sum::<f32>() + 1.0e-12)
        .sqrt()
        .max(1.0e-6);
    let scale = norm.ln_1p() / norm;
    let dscale_dnorm = (norm / (1.0 + norm) - norm.ln_1p()) / (norm * norm);
    let dot = input
        .iter()
        .zip(output_adjoint)
        .map(|(value, adjoint)| value * adjoint)
        .sum::<f32>();
    let radial = dscale_dnorm * dot / norm;
    for axis in 0..input.len() {
        input_adjoint[axis] += scale * output_adjoint[axis] + radial * input[axis];
    }
}
