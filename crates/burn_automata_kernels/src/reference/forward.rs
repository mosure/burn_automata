use super::*;

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

pub(crate) fn density(
    positions: &[[f32; 4]],
    snapshot: &HashGridSnapshot,
    cfg: &HashGridConfig,
) -> Vec<f32> {
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

pub(crate) fn blur_state(
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
pub(crate) fn second_pass_state_and_density(
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
pub(crate) fn second_pass_particle(
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

pub(crate) fn compute_state_gradient(
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

pub(crate) fn compute_density_gradient(
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

pub(crate) fn moment_matrix(
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
