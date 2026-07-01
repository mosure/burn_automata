use rayon::prelude::*;

use crate::{
    HashGridConfig, HashGridMode, KernelError, KernelResult,
    hashgrid::{
        HashGridSnapshot, build_hashgrid, cell_coords_for_position, cell_index_from_coords,
        neighbor_delta, wrap_position,
    },
};

const GROWTH_3D_MIN_OPACITY_LOGIT: f32 = -8.0;
const GROWTH_3D_MAX_OPACITY_LOGIT: f32 = 24.0;

#[derive(Clone, Copy, Debug)]
pub struct PerceptionOptions {
    pub state_grad: bool,
    pub density_grad: bool,
    pub eps0: f32,
    pub scale_equivariance: bool,
    pub particle_density_equivariance: bool,
    pub log_norm_grad: bool,
    pub log_norm_density_grad: bool,
    pub hybrid_state_gradient: bool,
    pub position_features: bool,
}

impl PerceptionOptions {
    pub fn new(state_grad: bool, density_grad: bool, eps0: f32) -> Self {
        Self {
            state_grad,
            density_grad,
            eps0,
            scale_equivariance: true,
            particle_density_equivariance: true,
            log_norm_grad: false,
            log_norm_density_grad: false,
            hybrid_state_gradient: true,
            position_features: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PerceptionOutput {
    pub features: Vec<f32>,
    pub density: Vec<f32>,
    pub blurred_state: Vec<f32>,
    pub state_gradient: Vec<f32>,
    pub density_gradient: Vec<f32>,
    pub feature_dims: usize,
}

#[derive(Clone, Debug)]
pub struct PerceptionAdjointOutput {
    pub state: Vec<f32>,
    pub position: Vec<[f32; 4]>,
}

#[allow(clippy::too_many_arguments)]
pub fn perceive(
    positions: &[[f32; 4]],
    states: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    cfg: &HashGridConfig,
    state_grad: bool,
    density_grad: bool,
) -> KernelResult<PerceptionOutput> {
    perceive_with_options(
        positions,
        states,
        batch_size,
        particle_count,
        state_dims,
        cfg,
        PerceptionOptions::new(state_grad, density_grad, cfg.eps),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn perceive_with_options(
    positions: &[[f32; 4]],
    states: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    cfg: &HashGridConfig,
    options: PerceptionOptions,
) -> KernelResult<PerceptionOutput> {
    check_shapes(positions, states, batch_size, particle_count, state_dims)?;
    cfg.validate()?;
    let snapshot = build_hashgrid(positions, batch_size, particle_count, cfg)?;
    let total = batch_size * particle_count;

    let density = density(positions, &snapshot, cfg);
    let mut blurred_state = vec![0.0; states.len()];
    let mut state_gradient = if options.state_grad {
        vec![0.0; total * state_dims * cfg.dim]
    } else {
        Vec::new()
    };
    let mut density_gradient = if options.density_grad {
        vec![0.0; total * cfg.dim]
    } else {
        Vec::new()
    };

    if options.state_grad && options.density_grad {
        let mut moment = if options.hybrid_state_gradient {
            vec![0.0; total * cfg.dim * cfg.dim]
        } else {
            Vec::new()
        };
        second_pass_state_and_density(
            positions,
            states,
            &density,
            state_dims,
            &snapshot,
            cfg,
            &mut blurred_state,
            &mut state_gradient,
            &mut density_gradient,
            if options.hybrid_state_gradient {
                Some(&mut moment)
            } else {
                None
            },
        );
        if options.hybrid_state_gradient {
            apply_moment_correction(&mut state_gradient, &moment, state_dims, cfg);
        }
        normalize_state_gradient(&mut state_gradient, state_dims, cfg, options);
        normalize_density_gradient(&mut density_gradient, particle_count, cfg, options);
    } else {
        blurred_state = blur_state(positions, states, &density, state_dims, &snapshot, cfg);
        if options.state_grad {
            state_gradient = compute_state_gradient(
                positions,
                states,
                &density,
                state_dims,
                &snapshot,
                cfg,
                options.hybrid_state_gradient,
            );
            normalize_state_gradient(&mut state_gradient, state_dims, cfg, options);
        }
        if options.density_grad {
            density_gradient = compute_density_gradient(positions, &snapshot, cfg);
            normalize_density_gradient(&mut density_gradient, particle_count, cfg, options);
        }
    }

    let feature_dims = state_dims * 2
        + usize::from(options.state_grad) * state_dims * cfg.dim
        + usize::from(options.density_grad) * cfg.dim
        + usize::from(options.position_features) * cfg.dim;
    let mut features = vec![0.0; total * feature_dims];
    features
        .par_chunks_mut(feature_dims)
        .enumerate()
        .for_each(|(idx, feature)| {
            let mut cursor = 0;
            let s_base = idx * state_dims;
            feature[cursor..cursor + state_dims]
                .copy_from_slice(&states[s_base..s_base + state_dims]);
            cursor += state_dims;
            feature[cursor..cursor + state_dims]
                .copy_from_slice(&blurred_state[s_base..s_base + state_dims]);
            cursor += state_dims;
            if options.state_grad {
                let g_base = idx * state_dims * cfg.dim;
                let len = state_dims * cfg.dim;
                feature[cursor..cursor + len]
                    .copy_from_slice(&state_gradient[g_base..g_base + len]);
                cursor += len;
            }
            if options.density_grad {
                let d_base = idx * cfg.dim;
                feature[cursor..cursor + cfg.dim]
                    .copy_from_slice(&density_gradient[d_base..d_base + cfg.dim]);
                cursor += cfg.dim;
            }
            if options.position_features {
                feature[cursor..cursor + cfg.dim].copy_from_slice(&positions[idx][..cfg.dim]);
            }
        });

    Ok(PerceptionOutput {
        features,
        density,
        blurred_state,
        state_gradient,
        density_gradient,
        feature_dims,
    })
}

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
pub fn euler_step(
    positions: &[[f32; 4]],
    states: &[f32],
    dx: &[[f32; 4]],
    ds: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    cfg: &HashGridConfig,
    dt: f32,
    update_mask: Option<&[f32]>,
) -> KernelResult<(Vec<[f32; 4]>, Vec<f32>)> {
    check_shapes(positions, states, batch_size, particle_count, state_dims)?;
    if dx.len() != positions.len() {
        return Err(KernelError::OutputShape {
            actual: dx.len(),
            expected: positions.len(),
        });
    }
    if ds.len() != states.len() {
        return Err(KernelError::OutputShape {
            actual: ds.len(),
            expected: states.len(),
        });
    }

    let total = batch_size * particle_count;
    let mut out_pos = vec![[0.0; 4]; total];
    let mut out_state = vec![0.0; total * state_dims];
    out_pos
        .par_iter_mut()
        .zip(out_state.par_chunks_mut(state_dims))
        .enumerate()
        .for_each(|(idx, (out_position, out_state_row))| {
            let mask = update_mask.map(|m| m[idx]).unwrap_or(1.0);
            let mut p = positions[idx];
            for axis in 0..cfg.spatial_mem_dims() {
                p[axis] += dt * dx[idx][axis] * mask;
            }
            *out_position = wrap_position(p, cfg);

            let state_base = idx * state_dims;
            for c in 0..state_dims {
                let mut next = states[state_base + c] + dt * ds[state_base + c] * mask;
                if cfg.dim == 3 && state_dims > 3 && (c == 3 || (state_dims > 8 && c == 8)) {
                    next = next.clamp(GROWTH_3D_MIN_OPACITY_LOGIT, GROWTH_3D_MAX_OPACITY_LOGIT);
                }
                out_state_row[c] = next;
            }
        });
    Ok((out_pos, out_state))
}

fn check_shapes(
    positions: &[[f32; 4]],
    states: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
) -> KernelResult<()> {
    let expected_positions = batch_size * particle_count;
    if positions.len() != expected_positions {
        return Err(KernelError::PositionShape {
            positions: positions.len(),
            expected: expected_positions,
        });
    }
    let expected_states = expected_positions * state_dims;
    if states.len() != expected_states {
        return Err(KernelError::StateShape {
            states: states.len(),
            expected: expected_states,
        });
    }
    Ok(())
}

fn density(positions: &[[f32; 4]], snapshot: &HashGridSnapshot, cfg: &HashGridConfig) -> Vec<f32> {
    (0..positions.len())
        .into_par_iter()
        .map(|idx| {
            let mut rho = 0.0;
            for_each_neighbor(idx, positions, snapshot, cfg, |_, _, r2| {
                rho += smoothing_poly6_kernel(r2, cfg);
            });
            rho
        })
        .collect()
}

fn blur_state(
    positions: &[[f32; 4]],
    states: &[f32],
    density: &[f32],
    state_dims: usize,
    snapshot: &HashGridSnapshot,
    cfg: &HashGridConfig,
) -> Vec<f32> {
    let mut out = vec![0.0; states.len()];
    out.par_chunks_mut(state_dims)
        .enumerate()
        .for_each(|(idx, dst)| {
            for_each_neighbor(idx, positions, snapshot, cfg, |j, _, r2| {
                let volume_j = density[j].recip_finite();
                let w = smoothing_poly6_kernel(r2, cfg) * volume_j;
                let src = j * state_dims;
                for c in 0..state_dims {
                    dst[c] += states[src + c] * w;
                }
            });
        });
    out
}

#[allow(clippy::too_many_arguments)]
fn second_pass_state_and_density(
    positions: &[[f32; 4]],
    states: &[f32],
    density: &[f32],
    state_dims: usize,
    snapshot: &HashGridSnapshot,
    cfg: &HashGridConfig,
    blurred_state: &mut [f32],
    state_gradient: &mut [f32],
    density_gradient: &mut [f32],
    moment: Option<&mut [f32]>,
) {
    let gradient_dims = state_dims * cfg.dim;
    let matrix_dims = cfg.dim * cfg.dim;
    match moment {
        Some(moment) => blurred_state
            .par_chunks_mut(state_dims)
            .zip(state_gradient.par_chunks_mut(gradient_dims))
            .zip(density_gradient.par_chunks_mut(cfg.dim))
            .zip(moment.par_chunks_mut(matrix_dims))
            .enumerate()
            .for_each(
                |(idx, (((blur_row, state_grad_row), density_grad_row), moment_row))| {
                    second_pass_particle(
                        idx,
                        positions,
                        states,
                        density,
                        state_dims,
                        snapshot,
                        cfg,
                        blur_row,
                        state_grad_row,
                        density_grad_row,
                        Some(moment_row),
                    );
                },
            ),
        None => blurred_state
            .par_chunks_mut(state_dims)
            .zip(state_gradient.par_chunks_mut(gradient_dims))
            .zip(density_gradient.par_chunks_mut(cfg.dim))
            .enumerate()
            .for_each(|(idx, ((blur_row, state_grad_row), density_grad_row))| {
                second_pass_particle(
                    idx,
                    positions,
                    states,
                    density,
                    state_dims,
                    snapshot,
                    cfg,
                    blur_row,
                    state_grad_row,
                    density_grad_row,
                    None,
                );
            }),
    }
}

#[allow(clippy::too_many_arguments)]
fn second_pass_particle(
    idx: usize,
    positions: &[[f32; 4]],
    states: &[f32],
    density: &[f32],
    state_dims: usize,
    snapshot: &HashGridSnapshot,
    cfg: &HashGridConfig,
    blur_row: &mut [f32],
    state_grad_row: &mut [f32],
    density_grad_row: &mut [f32],
    mut moment_row: Option<&mut [f32]>,
) {
    let si = idx * state_dims;
    for_each_neighbor(idx, positions, snapshot, cfg, |j, delta, r2| {
        let volume_j = density[j].recip_finite();
        let smooth = smoothing_poly6_kernel(r2, cfg);
        let src = j * state_dims;
        for c in 0..state_dims {
            blur_row[c] += states[src + c] * smooth * volume_j;
        }

        if idx == j {
            return;
        }
        let density_grad = spiky_gradient_kernel(delta, r2, cfg, 1.0);
        for axis in 0..cfg.dim {
            density_grad_row[axis] += density_grad[axis];
        }

        let volume_grad = spiky_gradient_kernel(delta, r2, cfg, volume_j);
        for c in 0..state_dims {
            let diff = states[src + c] - states[si + c];
            for axis in 0..cfg.dim {
                state_grad_row[c * cfg.dim + axis] += diff * volume_grad[axis];
            }
        }

        if let Some(moment_row) = moment_row.as_deref_mut() {
            for row in 0..cfg.dim {
                for col in 0..cfg.dim {
                    moment_row[row * cfg.dim + col] += delta[row] * volume_grad[col];
                }
            }
        }
    });
}

fn compute_state_gradient(
    positions: &[[f32; 4]],
    states: &[f32],
    density: &[f32],
    state_dims: usize,
    snapshot: &HashGridSnapshot,
    cfg: &HashGridConfig,
    hybrid: bool,
) -> Vec<f32> {
    let mut out = vec![0.0; positions.len() * state_dims * cfg.dim];
    out.par_chunks_mut(state_dims * cfg.dim)
        .enumerate()
        .for_each(|(idx, dst)| {
            let si = idx * state_dims;
            for_each_neighbor(idx, positions, snapshot, cfg, |j, delta, r2| {
                if idx == j {
                    return;
                }
                let volume_j = density[j].recip_finite();
                let grad = spiky_gradient_kernel(delta, r2, cfg, volume_j);
                let sj = j * state_dims;
                for c in 0..state_dims {
                    let diff = states[sj + c] - states[si + c];
                    for axis in 0..cfg.dim {
                        dst[c * cfg.dim + axis] += diff * grad[axis];
                    }
                }
            });
        });

    if hybrid {
        let moment = moment_matrix(positions, density, snapshot, cfg);
        apply_moment_correction(&mut out, &moment, state_dims, cfg);
    }

    out
}

#[allow(clippy::too_many_arguments)]
fn accumulate_blurred_state_adjoint(
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
fn accumulate_state_gradient_adjoint(
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
fn accumulate_density_gradient_position_adjoint(
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

fn accumulate_density_kernel_position_adjoint(
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

fn accumulate_volume_density_adjoint(
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

fn add_delta_position_adjoint(
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

fn inverse_matrix_adjoint(
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

fn density_gradient_scale(
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

fn compute_density_gradient(
    positions: &[[f32; 4]],
    snapshot: &HashGridSnapshot,
    cfg: &HashGridConfig,
) -> Vec<f32> {
    let mut out = vec![0.0; positions.len() * cfg.dim];
    out.par_chunks_mut(cfg.dim)
        .enumerate()
        .for_each(|(idx, dst)| {
            for_each_neighbor(idx, positions, snapshot, cfg, |j, delta, r2| {
                if idx == j {
                    return;
                }
                let grad = spiky_gradient_kernel(delta, r2, cfg, 1.0);
                for axis in 0..cfg.dim {
                    dst[axis] += grad[axis];
                }
            });
        });
    out
}

fn moment_matrix(
    positions: &[[f32; 4]],
    density: &[f32],
    snapshot: &HashGridSnapshot,
    cfg: &HashGridConfig,
) -> Vec<f32> {
    let matrix_dims = cfg.dim * cfg.dim;
    let mut out = vec![0.0; positions.len() * matrix_dims];
    out.par_chunks_mut(matrix_dims)
        .enumerate()
        .for_each(|(idx, dst)| {
            for_each_neighbor(idx, positions, snapshot, cfg, |j, delta, r2| {
                if idx == j {
                    return;
                }
                let volume_j = density[j].recip_finite();
                let grad = spiky_gradient_kernel(delta, r2, cfg, volume_j);
                for row in 0..cfg.dim {
                    for col in 0..cfg.dim {
                        dst[row * cfg.dim + col] += delta[row] * grad[col];
                    }
                }
            });
        });
    out
}

fn for_each_neighbor<F>(
    idx: usize,
    positions: &[[f32; 4]],
    snapshot: &HashGridSnapshot,
    cfg: &HashGridConfig,
    mut f: F,
) where
    F: FnMut(usize, [f32; 4], f32),
{
    let batch = idx / snapshot.particle_count;
    let batch_base = batch * snapshot.particle_count;
    let pi = positions[idx];
    let center = cell_coords_for_position(&pi, cfg);
    let z_min = if cfg.dim == 3 { -1 } else { 0 };
    let z_max = if cfg.dim == 3 { 1 } else { 0 };
    let eps2 = cfg.eps * cfg.eps;

    for dz in z_min..=z_max {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let coords = [center[0] + dx, center[1] + dy, center[2] + dz];
                let Some(cell) = cell_index_from_coords(coords, cfg) else {
                    continue;
                };
                let bin = batch * snapshot.cell_count + cell;
                for binned in snapshot.bin_offsets[bin]..snapshot.bin_offsets[bin + 1] {
                    let j = batch_base + snapshot.permutation[binned];
                    if cfg.mode == HashGridMode::Particle
                        && cell_coords_for_position(&positions[j], cfg) != coords
                    {
                        continue;
                    }
                    let delta = neighbor_delta(&pi, &positions[j], cfg);
                    let r2 = delta[..cfg.dim].iter().map(|v| v * v).sum::<f32>();
                    if r2 < eps2 {
                        f(j, delta, r2);
                    }
                }
            }
        }
    }
}

fn normalize_state_gradient(
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

fn normalize_density_gradient(
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

fn log_normalize_vector(values: &mut [f32]) {
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm <= 1e-12 {
        for value in values {
            *value = 0.0;
        }
        return;
    }
    let scale = norm.ln_1p() / norm;
    for value in values {
        *value *= scale;
    }
}

fn log_normalize_adjoint(input: &[f32], output_adjoint: &[f32], input_adjoint: &mut [f32]) {
    let norm = input.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= 1e-12 {
        return;
    }
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

fn apply_moment_correction(
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

fn safe_inverse_symmetric(matrix: &[f32], dim: usize) -> [f32; 9] {
    const TOL: f32 = 1e-3;
    let mut out = [0.0; 9];
    if dim == 2 {
        let a = matrix[0];
        let b = matrix[1];
        let d = matrix[3];
        let det = a * d - b * b;
        if det.abs() < TOL {
            out[0] = 1.0;
            out[3] = 1.0;
            return out;
        }
        let inv_det = det.recip();
        out[0] = d * inv_det;
        out[1] = -b * inv_det;
        out[2] = -b * inv_det;
        out[3] = a * inv_det;
        return out;
    }

    let a = matrix[0];
    let b = matrix[1];
    let c = matrix[2];
    let d = matrix[4];
    let e = matrix[5];
    let f = matrix[8];
    let t1 = d * f - e * e;
    let t2 = c * e - b * f;
    let t3 = b * e - c * d;
    let det = a * t1 + b * t2 + c * t3;
    if det.abs() < TOL {
        out[0] = 1.0;
        out[4] = 1.0;
        out[8] = 1.0;
        return out;
    }
    let inv_det = det.recip();
    out[0] = t1 * inv_det;
    out[1] = t2 * inv_det;
    out[2] = t3 * inv_det;
    out[3] = t2 * inv_det;
    out[4] = (a * f - c * c) * inv_det;
    out[5] = (b * c - a * e) * inv_det;
    out[6] = t3 * inv_det;
    out[7] = (b * c - a * e) * inv_det;
    out[8] = (a * d - b * b) * inv_det;
    out
}

fn smoothing_poly6_kernel(r2: f32, cfg: &HashGridConfig) -> f32 {
    let eps2 = cfg.eps * cfg.eps;
    if r2 >= eps2 {
        return 0.0;
    }
    let x = eps2 - r2;
    smoothing_poly6_normalization(cfg) * x * x * x
}

fn smoothing_poly6_delta_adjoint(
    delta: [f32; 4],
    r2: f32,
    cfg: &HashGridConfig,
    output_adjoint: f32,
) -> [f32; 4] {
    let mut out = [0.0; 4];
    let eps2 = cfg.eps * cfg.eps;
    if r2 >= eps2 {
        return out;
    }
    let x = eps2 - r2;
    let dkernel_dr2 = -3.0 * smoothing_poly6_normalization(cfg) * x * x;
    for axis in 0..cfg.dim {
        out[axis] = output_adjoint * dkernel_dr2 * 2.0 * delta[axis];
    }
    out
}

fn spiky_gradient_kernel(delta: [f32; 4], r2: f32, cfg: &HashGridConfig, coeff: f32) -> [f32; 4] {
    let mut out = [0.0; 4];
    let eps2 = cfg.eps * cfg.eps;
    if r2 <= 0.0 || r2 >= eps2 {
        return out;
    }
    let r = r2.sqrt();
    let mag = coeff * gradient_spiky_normalization(cfg) * 3.0 * (cfg.eps - r).powi(2) / r;
    for axis in 0..cfg.dim {
        out[axis] = mag * delta[axis];
    }
    out
}

fn spiky_gradient_delta_adjoint(
    delta: [f32; 4],
    r2: f32,
    cfg: &HashGridConfig,
    coeff: f32,
    output_adjoint: &[f32],
) -> [f32; 4] {
    let mut out = [0.0; 4];
    let eps2 = cfg.eps * cfg.eps;
    if r2 <= 0.0 || r2 >= eps2 {
        return out;
    }
    let r = r2.sqrt();
    let norm = coeff * gradient_spiky_normalization(cfg) * 3.0;
    let scale = norm * (cfg.eps - r).powi(2) / r;
    let dscale_dr = norm * (1.0 - (cfg.eps * cfg.eps) / r2);
    let dot = output_adjoint
        .iter()
        .take(cfg.dim)
        .zip(delta.iter())
        .map(|(adjoint, value)| adjoint * value)
        .sum::<f32>();
    for axis in 0..cfg.dim {
        out[axis] = scale * output_adjoint[axis] + dscale_dr * dot * delta[axis] / r;
    }
    out
}

fn smoothing_poly6_normalization(cfg: &HashGridConfig) -> f32 {
    if cfg.dim == 2 {
        4.0 / (std::f32::consts::PI * cfg.eps.powi(8))
    } else {
        315.0 / (64.0 * std::f32::consts::PI * cfg.eps.powi(9))
    }
}

fn gradient_spiky_normalization(cfg: &HashGridConfig) -> f32 {
    if cfg.dim == 2 {
        10.0 / (std::f32::consts::PI * cfg.eps.powi(5))
    } else {
        15.0 / (std::f32::consts::PI * cfg.eps.powi(6))
    }
}

trait RecipFinite {
    fn recip_finite(self) -> Self;
}

impl RecipFinite for f32 {
    fn recip_finite(self) -> Self {
        if self.abs() <= 1e-20 || !self.is_finite() {
            0.0
        } else {
            self.recip()
        }
    }
}
