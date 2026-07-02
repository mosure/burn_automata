use super::*;

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
