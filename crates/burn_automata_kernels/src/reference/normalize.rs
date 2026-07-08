use super::*;

const LOG_NORMALIZE_EPSILON: f32 = 1.0e-6;

pub(crate) fn normalize_state_gradient(
    gradient: &mut [f32],
    state_dims: usize,
    cfg: &HashGridConfig,
    options: PerceptionOptions,
) {
    let eps0 = options.eps0.max(f32::MIN_POSITIVE);
    let scale = if options.scale_equivariance {
        cfg.eps / eps0
    } else {
        1.0
    };
    gradient
        .par_chunks_mut(state_dims * cfg.dim)
        .for_each(|particle_gradient| {
            for channel in 0..state_dims {
                let base = channel * cfg.dim;
                for axis in 0..cfg.dim {
                    particle_gradient[base + axis] *= scale;
                }
                if options.log_norm_grad {
                    log_normalize_vector(&mut particle_gradient[base..base + cfg.dim]);
                }
            }
        });
}

pub(crate) fn normalize_density_gradient(
    gradient: &mut [f32],
    particle_count: usize,
    cfg: &HashGridConfig,
    options: PerceptionOptions,
) {
    let eps0 = options.eps0.max(f32::MIN_POSITIVE);
    let scale = if options.scale_equivariance {
        (cfg.eps / eps0).powi(1 + cfg.dim as i32)
    } else {
        1.0
    } * if options.particle_density_equivariance {
        1.0 / particle_count.max(1) as f32
    } else {
        1.0
    };
    gradient
        .par_chunks_mut(cfg.dim)
        .for_each(|particle_gradient| {
            for value in particle_gradient.iter_mut() {
                *value *= scale;
            }
            if options.log_norm_density_grad {
                log_normalize_vector(particle_gradient);
            }
        });
}

pub(crate) fn log_normalize_vector(values: &mut [f32]) {
    let norm = (values.iter().map(|v| v * v).sum::<f32>()
        + LOG_NORMALIZE_EPSILON * LOG_NORMALIZE_EPSILON)
        .sqrt()
        .max(LOG_NORMALIZE_EPSILON);
    let scale = norm.ln_1p() / norm;
    for value in values {
        *value *= scale;
    }
}

pub(crate) fn log_normalize_adjoint(
    input: &[f32],
    output_adjoint: &[f32],
    input_adjoint: &mut [f32],
) {
    let norm = (input.iter().map(|value| value * value).sum::<f32>()
        + LOG_NORMALIZE_EPSILON * LOG_NORMALIZE_EPSILON)
        .sqrt()
        .max(LOG_NORMALIZE_EPSILON);
    let scale = norm.ln_1p() / norm;
    let dscale_dnorm = (norm / (1.0 + norm) - norm.ln_1p()) / (norm * norm);
    let dot = input
        .iter()
        .zip(output_adjoint.iter())
        .map(|(value, adjoint)| value * adjoint)
        .sum::<f32>();
    let radial = dscale_dnorm * dot / norm;
    for axis in 0..input.len() {
        input_adjoint[axis] += scale * output_adjoint[axis] + radial * input[axis];
    }
}

pub(crate) fn apply_moment_correction(
    gradient: &mut [f32],
    moment: &[f32],
    state_dims: usize,
    cfg: &HashGridConfig,
) {
    let matrix_dims = cfg.dim * cfg.dim;
    gradient
        .par_chunks_mut(state_dims * cfg.dim)
        .zip(moment.par_chunks(matrix_dims))
        .for_each(|(particle_gradient, matrix)| {
            let inverse = safe_inverse_symmetric(matrix, cfg.dim);
            let mut corrected = [0.0; 4];
            for channel in 0..state_dims {
                let base = channel * cfg.dim;
                corrected[..cfg.dim].fill(0.0);
                for out_axis in 0..cfg.dim {
                    for in_axis in 0..cfg.dim {
                        corrected[out_axis] += particle_gradient[base + in_axis]
                            * inverse[in_axis * cfg.dim + out_axis];
                    }
                }
                particle_gradient[base..base + cfg.dim].copy_from_slice(&corrected[..cfg.dim]);
            }
        });
}
