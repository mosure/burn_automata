use rayon::prelude::*;

use super::{
    AdaptiveGraphMetrics, AdaptiveNpaPerceptionOptions, AdaptivePerceptionConfig,
    AdaptivePerceptionOutput,
    perception::{
        ParticleNeighbors, build_neighborhoods, build_neighborhoods_without_spacing, log_normalize,
        percentile_usize, validate_inputs,
    },
};
use crate::KernelResult;

/// Represented-measure, variable-support SPH perception for existing NPA
/// checkpoints. Equal represented measure and equal bandwidth reduce to the
/// hardened fixed NPA operator; unlike the normalized Shepard operator, this
/// preserves the feature semantics those checkpoints were trained against.
#[allow(clippy::too_many_arguments)]
pub fn adaptive_npa_perceive(
    positions: &[[f32; 4]],
    states: &[f32],
    represented_measure: &[f32],
    bandwidth: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    config: AdaptivePerceptionConfig,
    options: AdaptiveNpaPerceptionOptions,
) -> KernelResult<AdaptivePerceptionOutput> {
    adaptive_npa_perceive_impl(
        positions,
        states,
        represented_measure,
        bandwidth,
        batch_size,
        particle_count,
        state_dims,
        config,
        options,
        true,
    )
}

/// Exact NPA-compatible rule features without the controller-only spacing solve.
///
/// `features`, gradients, density, and graph metrics match [`adaptive_npa_perceive`].
/// `observed_spacing` is set to each target's bandwidth and must not be used for
/// bandwidth control or topology decisions.
#[allow(clippy::too_many_arguments)]
pub fn adaptive_npa_perceive_without_spacing(
    positions: &[[f32; 4]],
    states: &[f32],
    represented_measure: &[f32],
    bandwidth: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    config: AdaptivePerceptionConfig,
    options: AdaptiveNpaPerceptionOptions,
) -> KernelResult<AdaptivePerceptionOutput> {
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
    options.validate()?;
    let neighborhoods =
        build_neighborhoods_without_spacing(positions, bandwidth, particle_count, config);
    Ok(perceive_from_neighborhoods(
        positions,
        states,
        represented_measure,
        bandwidth,
        particle_count,
        state_dims,
        config,
        options,
        &neighborhoods,
    ))
}

/// Deterministic all-pairs compatibility oracle used by parity tests.
#[allow(clippy::too_many_arguments)]
pub fn adaptive_npa_perceive_all_pairs(
    positions: &[[f32; 4]],
    states: &[f32],
    represented_measure: &[f32],
    bandwidth: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    config: AdaptivePerceptionConfig,
    options: AdaptiveNpaPerceptionOptions,
) -> KernelResult<AdaptivePerceptionOutput> {
    adaptive_npa_perceive_impl(
        positions,
        states,
        represented_measure,
        bandwidth,
        batch_size,
        particle_count,
        state_dims,
        config,
        options,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn adaptive_npa_perceive_impl(
    positions: &[[f32; 4]],
    states: &[f32],
    represented_measure: &[f32],
    bandwidth: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    config: AdaptivePerceptionConfig,
    options: AdaptiveNpaPerceptionOptions,
    spatial_hash: bool,
) -> KernelResult<AdaptivePerceptionOutput> {
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
    options.validate()?;
    let neighborhoods =
        build_neighborhoods(positions, bandwidth, particle_count, config, spatial_hash);
    Ok(perceive_from_neighborhoods(
        positions,
        states,
        represented_measure,
        bandwidth,
        particle_count,
        state_dims,
        config,
        options,
        &neighborhoods,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn perceive_from_neighborhoods(
    positions: &[[f32; 4]],
    states: &[f32],
    represented_measure: &[f32],
    bandwidth: &[f32],
    particle_count: usize,
    state_dims: usize,
    config: AdaptivePerceptionConfig,
    options: AdaptiveNpaPerceptionOptions,
    neighborhoods: &[ParticleNeighbors],
) -> AdaptivePerceptionOutput {
    let source_is_coarse = |measure: f32, source_bandwidth: f32| {
        let tolerance = 1.0 + 32.0 * f32::EPSILON;
        if config.reference_measure > 0.0 {
            measure > config.reference_measure * tolerance
        } else {
            source_bandwidth > options.eps0 * tolerance
        }
    };
    let density_and_coarse = (0..positions.len())
        .into_par_iter()
        .map(|index| {
            let mut value =
                represented_measure[index] * poly6_kernel(0.0, bandwidth[index], config.dim);
            let mut coarse = if source_is_coarse(represented_measure[index], bandwidth[index]) {
                value
            } else {
                0.0
            };
            for candidate in &neighborhoods[index].candidates {
                let contribution = represented_measure[candidate.index]
                    * poly6_kernel(candidate.distance2, candidate.pair_bandwidth, config.dim);
                value += contribution;
                if source_is_coarse(
                    represented_measure[candidate.index],
                    bandwidth[candidate.index],
                ) {
                    coarse += contribution;
                }
            }
            (value, coarse)
        })
        .collect::<Vec<_>>();
    let density = density_and_coarse
        .iter()
        .map(|(density, _)| *density)
        .collect::<Vec<_>>();
    let coarse_exposure = density_and_coarse
        .iter()
        .map(|(density, coarse)| (coarse / density.max(f32::MIN_POSITIVE)).clamp(0.0, 1.0))
        .collect::<Vec<_>>();
    let batch_measure = represented_measure
        .chunks_exact(particle_count)
        .map(|batch| batch.iter().sum::<f32>())
        .collect::<Vec<_>>();
    let feature_dims = config.feature_dims(state_dims);
    let mut features = vec![0.0; positions.len() * feature_dims];
    let mut blurred_state = vec![0.0; states.len()];
    let mut state_gradient = vec![0.0; positions.len() * state_dims * config.dim];
    let mut density_gradient = vec![0.0; positions.len() * config.dim];
    let mut moment_condition = vec![0.0; positions.len()];
    let mut moment_fallback = vec![false; positions.len()];
    features
        .par_chunks_mut(feature_dims)
        .zip(blurred_state.par_chunks_mut(state_dims))
        .zip(state_gradient.par_chunks_mut(state_dims * config.dim))
        .zip(density_gradient.par_chunks_mut(config.dim))
        .zip(moment_condition.par_iter_mut())
        .zip(moment_fallback.par_iter_mut())
        .enumerate()
        .for_each(
            |(
                index,
                (((((features, blurred), gradient), density_gradient), condition), fallback),
            )| {
                perceive_particle_into(
                    index,
                    positions,
                    states,
                    represented_measure,
                    bandwidth,
                    &density,
                    batch_measure[index / particle_count],
                    particle_count,
                    state_dims,
                    &neighborhoods[index],
                    config,
                    options,
                    features,
                    blurred,
                    gradient,
                    density_gradient,
                    condition,
                    fallback,
                );
            },
        );
    let accepted_degree = neighborhoods
        .iter()
        .map(|neighbors| neighbors.candidates.len())
        .collect::<Vec<_>>();
    let graph = graph_metrics(neighborhoods, &accepted_degree, bandwidth);
    AdaptivePerceptionOutput {
        features,
        normalized_state: blurred_state,
        state_gradient,
        occupancy_gradient: density_gradient,
        partition: density,
        coarse_exposure,
        observed_spacing: neighborhoods
            .iter()
            .map(|neighbors| neighbors.observed_spacing)
            .collect(),
        moment_condition,
        moment_fallback,
        accepted_degree,
        graph,
        feature_dims,
    }
}

#[allow(clippy::too_many_arguments)]
fn perceive_particle_into(
    index: usize,
    positions: &[[f32; 4]],
    states: &[f32],
    represented_measure: &[f32],
    bandwidth: &[f32],
    density: &[f32],
    batch_measure: f32,
    particle_count: usize,
    state_dims: usize,
    neighbors: &ParticleNeighbors,
    config: AdaptivePerceptionConfig,
    options: AdaptiveNpaPerceptionOptions,
    features: &mut [f32],
    blurred_state: &mut [f32],
    state_gradient: &mut [f32],
    density_gradient: &mut [f32],
    moment_condition: &mut f32,
    moment_fallback: &mut bool,
) {
    let state_base = index * state_dims;
    let state_i = &states[state_base..state_base + state_dims];
    blurred_state.fill(0.0);
    state_gradient.fill(0.0);
    density_gradient.fill(0.0);
    let mut moment = [0.0; 9];
    accumulate_blur(
        state_i,
        represented_measure[index] / density[index].max(f32::MIN_POSITIVE),
        poly6_kernel(0.0, bandwidth[index], config.dim),
        blurred_state,
    );
    let mean_measure = batch_measure / particle_count as f32;
    for candidate in &neighbors.candidates {
        let source_base = candidate.index * state_dims;
        let state_j = &states[source_base..source_base + state_dims];
        let volume =
            represented_measure[candidate.index] / density[candidate.index].max(f32::MIN_POSITIVE);
        accumulate_blur(
            state_j,
            volume,
            poly6_kernel(candidate.distance2, candidate.pair_bandwidth, config.dim),
            blurred_state,
        );
        let volume_gradient = spiky_gradient(
            candidate.delta,
            candidate.distance2,
            candidate.pair_bandwidth,
            config.dim,
            volume,
        );
        for channel in 0..state_dims {
            let difference = state_j[channel] - state_i[channel];
            for axis in 0..config.dim {
                state_gradient[channel * config.dim + axis] += difference * volume_gradient[axis];
            }
        }
        for row in 0..config.dim {
            for col in 0..config.dim {
                moment[row * config.dim + col] += candidate.delta[row] * volume_gradient[col];
            }
        }
        let density_weight = if options.particle_density_equivariance {
            represented_measure[candidate.index] / batch_measure.max(f32::MIN_POSITIVE)
        } else {
            represented_measure[candidate.index] / mean_measure.max(f32::MIN_POSITIVE)
        };
        let gradient = spiky_gradient(
            candidate.delta,
            candidate.distance2,
            candidate.pair_bandwidth,
            config.dim,
            density_weight,
        );
        for axis in 0..config.dim {
            density_gradient[axis] += gradient[axis];
        }
    }
    let (inverse, condition, fallback) = safe_inverse_symmetric(moment, config.dim);
    *moment_condition = condition;
    *moment_fallback = fallback;
    for channel in 0..state_dims {
        let base = channel * config.dim;
        let mut raw = [0.0_f32; 3];
        raw[..config.dim].copy_from_slice(&state_gradient[base..base + config.dim]);
        for out_axis in 0..config.dim {
            state_gradient[base + out_axis] = (0..config.dim)
                .map(|in_axis| raw[in_axis] * inverse[in_axis * config.dim + out_axis])
                .sum();
        }
        if options.scale_equivariance {
            for value in &mut state_gradient[base..base + config.dim] {
                *value *= bandwidth[index] / options.eps0;
            }
        }
        if options.log_norm_grad {
            log_normalize(&mut state_gradient[base..base + config.dim]);
        }
    }
    if options.scale_equivariance {
        let scale = (bandwidth[index] / options.eps0).powi(config.dim as i32 + 1);
        density_gradient
            .iter_mut()
            .for_each(|value| *value *= scale);
    }
    if options.log_norm_density_grad {
        log_normalize(density_gradient);
    }
    let mut cursor = 0;
    features[cursor..cursor + state_dims].copy_from_slice(state_i);
    cursor += state_dims;
    features[cursor..cursor + state_dims].copy_from_slice(blurred_state);
    cursor += state_dims;
    features[cursor..cursor + state_gradient.len()].copy_from_slice(state_gradient);
    cursor += state_gradient.len();
    features[cursor..cursor + density_gradient.len()].copy_from_slice(density_gradient);
    cursor += density_gradient.len();
    if options.position_features {
        features[cursor..cursor + config.dim].copy_from_slice(&positions[index][..config.dim]);
    }
}

fn accumulate_blur(state: &[f32], volume: f32, kernel: f32, output: &mut [f32]) {
    for (output, state) in output.iter_mut().zip(state) {
        *output += state * volume * kernel;
    }
}

pub(super) fn poly6_kernel(distance2: f32, bandwidth: f32, dim: usize) -> f32 {
    let bandwidth2 = bandwidth * bandwidth;
    if distance2 >= bandwidth2 {
        return 0.0;
    }
    let normalization = if dim == 2 {
        4.0 / (std::f32::consts::PI * bandwidth.powi(8))
    } else {
        315.0 / (64.0 * std::f32::consts::PI * bandwidth.powi(9))
    };
    normalization * (bandwidth2 - distance2).powi(3)
}

pub(super) fn spiky_gradient(
    delta: [f32; 3],
    distance2: f32,
    bandwidth: f32,
    dim: usize,
    coefficient: f32,
) -> [f32; 3] {
    if distance2 <= 0.0 || distance2 >= bandwidth * bandwidth {
        return [0.0; 3];
    }
    let distance = distance2.sqrt();
    let normalization = if dim == 2 {
        10.0 / (std::f32::consts::PI * bandwidth.powi(5))
    } else {
        15.0 / (std::f32::consts::PI * bandwidth.powi(6))
    };
    let scale = coefficient * normalization * 3.0 * (bandwidth - distance).powi(2) / distance;
    let mut output = [0.0; 3];
    for axis in 0..dim {
        output[axis] = scale * delta[axis];
    }
    output
}

pub(super) fn safe_inverse_symmetric(matrix: [f32; 9], dim: usize) -> ([f32; 9], f32, bool) {
    const TOLERANCE: f32 = 1.0e-3;
    let mut output = [0.0; 9];
    if dim == 2 {
        let determinant = matrix[0] * matrix[3] - matrix[1] * matrix[1];
        if determinant.abs() < TOLERANCE {
            output[0] = 1.0;
            output[3] = 1.0;
            return (output, f32::INFINITY, true);
        }
        let inverse = determinant.recip();
        output[0] = matrix[3] * inverse;
        output[1] = -matrix[1] * inverse;
        output[2] = -matrix[1] * inverse;
        output[3] = matrix[0] * inverse;
        return (output, condition_number(matrix, output, dim), false);
    }
    let a = matrix[0];
    let b = matrix[1];
    let c = matrix[2];
    let d = matrix[4];
    let e = matrix[5];
    let f = matrix[8];
    let determinant = a * (d * f - e * e) + b * (c * e - b * f) + c * (b * e - c * d);
    if determinant.abs() < TOLERANCE {
        output[0] = 1.0;
        output[4] = 1.0;
        output[8] = 1.0;
        return (output, f32::INFINITY, true);
    }
    let inverse = determinant.recip();
    output[0] = (d * f - e * e) * inverse;
    output[1] = (c * e - b * f) * inverse;
    output[2] = (b * e - c * d) * inverse;
    output[3] = output[1];
    output[4] = (a * f - c * c) * inverse;
    output[5] = (b * c - a * e) * inverse;
    output[6] = output[2];
    output[7] = output[5];
    output[8] = (a * d - b * b) * inverse;
    (output, condition_number(matrix, output, dim), false)
}

fn condition_number(matrix: [f32; 9], inverse: [f32; 9], dim: usize) -> f32 {
    let norm = |values: [f32; 9]| {
        (0..dim)
            .flat_map(|row| (0..dim).map(move |col| values[row * dim + col].powi(2)))
            .sum::<f32>()
            .sqrt()
    };
    norm(matrix) * norm(inverse)
}

fn graph_metrics(
    neighborhoods: &[ParticleNeighbors],
    accepted_degree: &[usize],
    bandwidth: &[f32],
) -> AdaptiveGraphMetrics {
    let candidate_visits = neighborhoods.iter().map(|row| row.candidate_visits).sum();
    let raw_messages = neighborhoods.iter().map(|row| row.raw_count).sum();
    let accepted_messages = accepted_degree.iter().sum();
    let mut sorted_degree = accepted_degree.to_vec();
    sorted_degree.sort_unstable();
    let cross_scale_messages = neighborhoods
        .iter()
        .enumerate()
        .map(|(index, row)| {
            row.candidates
                .iter()
                .filter(|candidate| {
                    let ratio = bandwidth[index] / bandwidth[candidate.index];
                    !(0.5..=2.0).contains(&ratio)
                })
                .count()
        })
        .sum::<usize>();
    AdaptiveGraphMetrics {
        candidate_visits,
        raw_messages,
        accepted_messages,
        degree_mean: accepted_messages as f32 / neighborhoods.len().max(1) as f32,
        degree_p95: percentile_usize(&sorted_degree, 0.95),
        degree_max: sorted_degree.last().copied().unwrap_or_default(),
        isolated_particles: accepted_degree
            .iter()
            .filter(|degree| **degree == 0)
            .count(),
        cross_scale_fraction: cross_scale_messages as f32 / accepted_messages.max(1) as f32,
    }
}
