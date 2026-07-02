#![allow(
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use super::*;

#[allow(dead_code)]
pub(crate) fn terminal_render_position_adjoint(
    config: &NpaConfig,
    trace: &crate::RolloutTrace,
    gradient: &RenderProxyGradientRows,
    coverage_updates: &[[f32; 3]],
    motion_gain: f32,
    full_coverage_adjoint: bool,
    rows: usize,
) -> Vec<[f32; 4]> {
    terminal_render_position_adjoint_weighted(
        config,
        trace,
        gradient,
        coverage_updates,
        motion_gain,
        full_coverage_adjoint,
        rows,
        None,
    )
}

pub(crate) fn terminal_render_locality_weights(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    front_radius: f32,
) -> Vec<f32> {
    let rows = positions.len();
    let mut weights = vec![0.0_f32; rows];
    if rows == 0 {
        return weights;
    }
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || states.len() < rows.saturating_mul(config.state_dims)
    {
        weights.fill(1.0);
        return weights;
    }
    let front_weights = if front_radius > 0.0 && front_radius.is_finite() {
        Some(local_front_weights(config, positions, states, front_radius))
    } else {
        None
    };
    for row in 0..rows {
        let liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
        weights[row] = if liveness > -1.0 {
            1.0
        } else {
            front_weights
                .as_ref()
                .and_then(|front| front.get(row))
                .copied()
                .unwrap_or(0.0)
                .clamp(0.0, 1.0)
        };
    }
    weights
}

pub(crate) fn terminal_render_position_adjoint_weighted(
    config: &NpaConfig,
    trace: &crate::RolloutTrace,
    gradient: &RenderProxyGradientRows,
    coverage_updates: &[[f32; 3]],
    motion_gain: f32,
    full_coverage_adjoint: bool,
    rows: usize,
    row_weights: Option<&[f32]>,
) -> Vec<[f32; 4]> {
    let mut position_adjoint = vec![[0.0; 4]; trace.positions.len()];
    if full_coverage_adjoint {
        for particle_row in 0..position_adjoint.len() {
            let row_weight = row_weights
                .and_then(|weights| weights.get(particle_row))
                .copied()
                .unwrap_or(1.0)
                .clamp(0.0, 1.0);
            if row_weight <= 0.0 {
                continue;
            }
            for axis in 0..config.spatial_dims {
                let coverage = coverage_updates
                    .get(particle_row)
                    .map(|update| update[axis])
                    .unwrap_or(0.0);
                position_adjoint[particle_row][axis] -= row_weight * coverage;
            }
            clamp_position_adjoint_row(&mut position_adjoint[particle_row], config.spatial_dims);
        }
    }
    for (gradient_row, &particle_row) in gradient.row_indices.iter().enumerate().take(rows) {
        if particle_row >= position_adjoint.len() {
            continue;
        }
        let row_weight = row_weights
            .and_then(|weights| weights.get(particle_row))
            .copied()
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        if row_weight <= 0.0 {
            continue;
        }
        for axis in 0..config.spatial_dims {
            let coverage = if full_coverage_adjoint {
                0.0
            } else {
                coverage_updates
                    .get(particle_row)
                    .map(|update| update[axis])
                    .unwrap_or(0.0)
            };
            position_adjoint[particle_row][axis] +=
                row_weight * (motion_gain * gradient.gradients[gradient_row][axis] - coverage);
        }
        clamp_position_adjoint_row(&mut position_adjoint[particle_row], config.spatial_dims);
    }
    position_adjoint
}
