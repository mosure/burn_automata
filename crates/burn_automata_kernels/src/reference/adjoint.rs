use super::*;

#[allow(clippy::too_many_arguments)]
pub fn perceive_state_adjoint_with_options(
    positions: &[[f32; 4]],
    states: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    cfg: &HashGridConfig,
    options: PerceptionOptions,
    feature_adjoint: &[f32],
) -> KernelResult<Vec<f32>> {
    Ok(perceive_adjoint_with_options(
        positions,
        states,
        batch_size,
        particle_count,
        state_dims,
        cfg,
        options,
        feature_adjoint,
    )?
    .state)
}

#[allow(clippy::too_many_arguments)]
pub fn perceive_adjoint_with_options(
    positions: &[[f32; 4]],
    states: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    cfg: &HashGridConfig,
    options: PerceptionOptions,
    feature_adjoint: &[f32],
) -> KernelResult<PerceptionAdjointOutput> {
    check_shapes(positions, states, batch_size, particle_count, state_dims)?;
    cfg.validate()?;
    let total = batch_size * particle_count;
    let feature_dims = state_dims * 2
        + usize::from(options.state_grad) * state_dims * cfg.dim
        + usize::from(options.density_grad) * cfg.dim
        + usize::from(options.position_features) * cfg.dim;
    let expected_feature_adjoint = total * feature_dims;
    if feature_adjoint.len() != expected_feature_adjoint {
        return Err(KernelError::OutputShape {
            actual: feature_adjoint.len(),
            expected: expected_feature_adjoint,
        });
    }

    let snapshot = build_hashgrid(positions, batch_size, particle_count, cfg)?;
    let density = density(positions, &snapshot, cfg);
    let mut state_adjoint = vec![0.0; states.len()];
    let mut position_adjoint = vec![[0.0; 4]; positions.len()];
    let mut density_adjoint = vec![0.0; positions.len()];

    for idx in 0..total {
        let feature_base = idx * feature_dims;
        let state_base = idx * state_dims;
        let mut cursor = feature_base;

        for channel in 0..state_dims {
            state_adjoint[state_base + channel] += feature_adjoint[cursor + channel];
        }
        cursor += state_dims;

        let blurred_adjoint = &feature_adjoint[cursor..cursor + state_dims];
        accumulate_blurred_state_adjoint(
            idx,
            positions,
            states,
            &density,
            state_dims,
            &snapshot,
            cfg,
            blurred_adjoint,
            &mut state_adjoint,
            &mut position_adjoint,
            &mut density_adjoint,
        );
        cursor += state_dims;

        if options.state_grad {
            let gradient_len = state_dims * cfg.dim;
            let gradient_adjoint = &feature_adjoint[cursor..cursor + gradient_len];
            accumulate_state_gradient_adjoint(
                idx,
                positions,
                states,
                &density,
                state_dims,
                &snapshot,
                cfg,
                options,
                gradient_adjoint,
                &mut state_adjoint,
                &mut position_adjoint,
                &mut density_adjoint,
            );
            cursor += gradient_len;
        }

        if options.density_grad {
            let density_gradient_adjoint = &feature_adjoint[cursor..cursor + cfg.dim];
            accumulate_density_gradient_position_adjoint(
                idx,
                positions,
                &snapshot,
                cfg,
                options,
                particle_count,
                density_gradient_adjoint,
                &mut position_adjoint,
            );
            cursor += cfg.dim;
        }
        if options.position_features {
            for axis in 0..cfg.dim {
                position_adjoint[idx][axis] += feature_adjoint[cursor + axis];
            }
        }
    }

    accumulate_density_kernel_position_adjoint(
        positions,
        &snapshot,
        cfg,
        &density_adjoint,
        &mut position_adjoint,
    );

    Ok(PerceptionAdjointOutput {
        state: state_adjoint,
        position: position_adjoint,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn accumulate_blurred_state_adjoint(
    idx: usize,
    positions: &[[f32; 4]],
    states: &[f32],
    density: &[f32],
    state_dims: usize,
    snapshot: &HashGridSnapshot,
    cfg: &HashGridConfig,
    blurred_adjoint: &[f32],
    state_adjoint: &mut [f32],
    position_adjoint: &mut [[f32; 4]],
    density_adjoint: &mut [f32],
) {
    for_each_neighbor(idx, positions, snapshot, cfg, |j, delta, r2| {
        let volume_j = density[j].recip_finite();
        let kernel = smoothing_poly6_kernel(r2, cfg);
        let weight = kernel * volume_j;
        let state_base = j * state_dims;
        let mut weight_adjoint = 0.0;
        for channel in 0..state_dims {
            state_adjoint[state_base + channel] += blurred_adjoint[channel] * weight;
            weight_adjoint += blurred_adjoint[channel] * states[state_base + channel];
        }
        let delta_adjoint =
            smoothing_poly6_delta_adjoint(delta, r2, cfg, weight_adjoint * volume_j);
        add_delta_position_adjoint(position_adjoint, idx, j, cfg.dim, &delta_adjoint);
        accumulate_volume_density_adjoint(density, density_adjoint, j, weight_adjoint * kernel);
    });
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn accumulate_state_gradient_adjoint(
    idx: usize,
    positions: &[[f32; 4]],
    states: &[f32],
    density: &[f32],
    state_dims: usize,
    snapshot: &HashGridSnapshot,
    cfg: &HashGridConfig,
    options: PerceptionOptions,
    gradient_adjoint: &[f32],
    state_adjoint: &mut [f32],
    position_adjoint: &mut [[f32; 4]],
    density_adjoint: &mut [f32],
) {
    let si = idx * state_dims;
    let gradient_dims = state_dims * cfg.dim;
    let mut raw_gradient = vec![0.0; gradient_dims];
    let mut moment_matrix = [0.0; 9];
    for_each_neighbor(idx, positions, snapshot, cfg, |j, delta, r2| {
        if idx == j {
            return;
        }
        let volume_j = density[j].recip_finite();
        let grad = spiky_gradient_kernel(delta, r2, cfg, volume_j);
        let sj = j * state_dims;
        for channel in 0..state_dims {
            let diff = states[sj + channel] - states[si + channel];
            for axis in 0..cfg.dim {
                raw_gradient[channel * cfg.dim + axis] += diff * grad[axis];
            }
        }
        if options.hybrid_state_gradient {
            for row in 0..cfg.dim {
                for col in 0..cfg.dim {
                    moment_matrix[row * cfg.dim + col] += delta[row] * grad[col];
                }
            }
        }
    });

    let moment = if options.hybrid_state_gradient {
        Some(moment_matrix)
    } else {
        None
    };
    let inverse = moment
        .as_ref()
        .map(|matrix| safe_inverse_symmetric(matrix, cfg.dim));
    let scale = if options.scale_equivariance {
        cfg.eps / options.eps0.max(f32::MIN_POSITIVE)
    } else {
        1.0
    };
    let mut raw_gradient_adjoint = vec![0.0; gradient_dims];
    let mut inverse_adjoint = [0.0; 9];

    for channel in 0..state_dims {
        let mut corrected = [0.0; 4];
        let gradient_base = channel * cfg.dim;
        corrected[..cfg.dim].copy_from_slice(&raw_gradient[gradient_base..gradient_base + cfg.dim]);

        if let Some(inverse) = inverse {
            let mut moment_corrected = [0.0; 4];
            for out_axis in 0..cfg.dim {
                for in_axis in 0..cfg.dim {
                    moment_corrected[out_axis] +=
                        corrected[in_axis] * inverse[in_axis * cfg.dim + out_axis];
                }
            }
            corrected = moment_corrected;
        }

        let mut normalized_input = [0.0; 4];
        for axis in 0..cfg.dim {
            normalized_input[axis] = corrected[axis] * scale;
        }

        let mut corrected_adjoint = [0.0; 4];
        if options.log_norm_grad {
            log_normalize_adjoint(
                &normalized_input[..cfg.dim],
                &gradient_adjoint[gradient_base..gradient_base + cfg.dim],
                &mut corrected_adjoint[..cfg.dim],
            );
        } else {
            corrected_adjoint[..cfg.dim]
                .copy_from_slice(&gradient_adjoint[gradient_base..gradient_base + cfg.dim]);
        }
        for value in corrected_adjoint.iter_mut().take(cfg.dim) {
            *value *= scale;
        }

        let mut raw_adjoint = [0.0; 4];
        if let Some(inverse) = inverse {
            for in_axis in 0..cfg.dim {
                for out_axis in 0..cfg.dim {
                    raw_adjoint[in_axis] +=
                        corrected_adjoint[out_axis] * inverse[in_axis * cfg.dim + out_axis];
                }
            }
        } else {
            raw_adjoint[..cfg.dim].copy_from_slice(&corrected_adjoint[..cfg.dim]);
        }
        raw_gradient_adjoint[gradient_base..gradient_base + cfg.dim]
            .copy_from_slice(&raw_adjoint[..cfg.dim]);

        if inverse.is_some() {
            for in_axis in 0..cfg.dim {
                for out_axis in 0..cfg.dim {
                    inverse_adjoint[in_axis * cfg.dim + out_axis] +=
                        raw_gradient[gradient_base + in_axis] * corrected_adjoint[out_axis];
                }
            }
        }
    }

    let mut moment_adjoint = [0.0; 9];
    if let Some(inverse) = inverse {
        inverse_matrix_adjoint(&inverse, &inverse_adjoint, cfg.dim, &mut moment_adjoint);
    }

    for_each_neighbor(idx, positions, snapshot, cfg, |j, delta, r2| {
        if idx == j {
            return;
        }
        let volume_j = density[j].recip_finite();
        let volume_grad = spiky_gradient_kernel(delta, r2, cfg, volume_j);
        let unit_grad = spiky_gradient_kernel(delta, r2, cfg, 1.0);
        let sj = j * state_dims;

        for channel in 0..state_dims {
            let gradient_base = channel * cfg.dim;
            let diff = states[sj + channel] - states[si + channel];
            let raw_adjoint = &raw_gradient_adjoint[gradient_base..gradient_base + cfg.dim];
            let state_contribution = raw_adjoint
                .iter()
                .zip(volume_grad[..cfg.dim].iter())
                .map(|(adjoint, gradient)| adjoint * gradient)
                .sum::<f32>();
            state_adjoint[j * state_dims + channel] += state_contribution;
            state_adjoint[si + channel] -= state_contribution;

            let mut grad_adjoint = [0.0; 4];
            for axis in 0..cfg.dim {
                grad_adjoint[axis] = raw_adjoint[axis] * diff;
            }
            let delta_adjoint =
                spiky_gradient_delta_adjoint(delta, r2, cfg, volume_j, &grad_adjoint);
            add_delta_position_adjoint(position_adjoint, idx, j, cfg.dim, &delta_adjoint);
            let volume_adjoint = grad_adjoint[..cfg.dim]
                .iter()
                .zip(unit_grad[..cfg.dim].iter())
                .map(|(adjoint, gradient)| adjoint * gradient)
                .sum::<f32>();
            accumulate_volume_density_adjoint(density, density_adjoint, j, volume_adjoint);
        }

        if options.hybrid_state_gradient {
            let mut direct_delta_adjoint = [0.0; 4];
            let mut grad_adjoint = [0.0; 4];
            for row in 0..cfg.dim {
                for col in 0..cfg.dim {
                    let adjoint = moment_adjoint[row * cfg.dim + col];
                    direct_delta_adjoint[row] += adjoint * volume_grad[col];
                    grad_adjoint[col] += adjoint * delta[row];
                }
            }
            let grad_delta_adjoint =
                spiky_gradient_delta_adjoint(delta, r2, cfg, volume_j, &grad_adjoint);
            for axis in 0..cfg.dim {
                direct_delta_adjoint[axis] += grad_delta_adjoint[axis];
            }
            add_delta_position_adjoint(position_adjoint, idx, j, cfg.dim, &direct_delta_adjoint);
            let volume_adjoint = grad_adjoint[..cfg.dim]
                .iter()
                .zip(unit_grad[..cfg.dim].iter())
                .map(|(adjoint, gradient)| adjoint * gradient)
                .sum::<f32>();
            accumulate_volume_density_adjoint(density, density_adjoint, j, volume_adjoint);
        }
    });
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn accumulate_density_gradient_position_adjoint(
    idx: usize,
    positions: &[[f32; 4]],
    snapshot: &HashGridSnapshot,
    cfg: &HashGridConfig,
    options: PerceptionOptions,
    particle_count: usize,
    gradient_adjoint: &[f32],
    position_adjoint: &mut [[f32; 4]],
) {
    let mut raw = [0.0; 4];
    for_each_neighbor(idx, positions, snapshot, cfg, |j, delta, r2| {
        if idx == j {
            return;
        }
        let grad = spiky_gradient_kernel(delta, r2, cfg, 1.0);
        for axis in 0..cfg.dim {
            raw[axis] += grad[axis];
        }
    });

    let scale = density_gradient_scale(cfg, options, particle_count);
    let mut normalized_input = [0.0; 4];
    for axis in 0..cfg.dim {
        normalized_input[axis] = raw[axis] * scale;
    }
    let mut raw_adjoint = [0.0; 4];
    if options.log_norm_density_grad {
        log_normalize_adjoint(
            &normalized_input[..cfg.dim],
            gradient_adjoint,
            &mut raw_adjoint[..cfg.dim],
        );
        for value in raw_adjoint.iter_mut().take(cfg.dim) {
            *value *= scale;
        }
    } else {
        for axis in 0..cfg.dim {
            raw_adjoint[axis] = gradient_adjoint[axis] * scale;
        }
    }

    for_each_neighbor(idx, positions, snapshot, cfg, |j, delta, r2| {
        if idx == j {
            return;
        }
        let delta_adjoint = spiky_gradient_delta_adjoint(delta, r2, cfg, 1.0, &raw_adjoint);
        add_delta_position_adjoint(position_adjoint, idx, j, cfg.dim, &delta_adjoint);
    });
}

pub(crate) fn accumulate_density_kernel_position_adjoint(
    positions: &[[f32; 4]],
    snapshot: &HashGridSnapshot,
    cfg: &HashGridConfig,
    density_adjoint: &[f32],
    position_adjoint: &mut [[f32; 4]],
) {
    for (idx, &adjoint) in density_adjoint.iter().enumerate().take(positions.len()) {
        if adjoint == 0.0 {
            continue;
        }
        for_each_neighbor(idx, positions, snapshot, cfg, |j, delta, r2| {
            let delta_adjoint = smoothing_poly6_delta_adjoint(delta, r2, cfg, adjoint);
            add_delta_position_adjoint(position_adjoint, idx, j, cfg.dim, &delta_adjoint);
        });
    }
}

pub(crate) fn accumulate_volume_density_adjoint(
    density: &[f32],
    density_adjoint: &mut [f32],
    idx: usize,
    volume_adjoint: f32,
) {
    let rho = density[idx];
    if rho.abs() <= 1.0e-20 || !rho.is_finite() {
        return;
    }
    density_adjoint[idx] -= volume_adjoint / (rho * rho);
}

pub(crate) fn add_delta_position_adjoint(
    position_adjoint: &mut [[f32; 4]],
    lhs: usize,
    rhs: usize,
    dim: usize,
    delta_adjoint: &[f32; 4],
) {
    for axis in 0..dim {
        position_adjoint[rhs][axis] += delta_adjoint[axis];
        position_adjoint[lhs][axis] -= delta_adjoint[axis];
    }
}

pub(crate) fn inverse_matrix_adjoint(
    inverse: &[f32; 9],
    inverse_adjoint: &[f32; 9],
    dim: usize,
    matrix_adjoint: &mut [f32; 9],
) {
    for row in 0..dim {
        for col in 0..dim {
            let mut value = 0.0;
            for inv_row in 0..dim {
                for inv_col in 0..dim {
                    value += inverse[inv_row * dim + row]
                        * inverse_adjoint[inv_row * dim + inv_col]
                        * inverse[col * dim + inv_col];
                }
            }
            matrix_adjoint[row * dim + col] -= value;
        }
    }
}

pub(crate) fn density_gradient_scale(
    cfg: &HashGridConfig,
    options: PerceptionOptions,
    particle_count: usize,
) -> f32 {
    let eps0 = options.eps0.max(f32::MIN_POSITIVE);
    let scale = if options.scale_equivariance {
        (cfg.eps / eps0).powi(1 + cfg.dim as i32)
    } else {
        1.0
    };
    scale
        * if options.particle_density_equivariance {
            1.0 / particle_count.max(1) as f32
        } else {
            1.0
        }
}
