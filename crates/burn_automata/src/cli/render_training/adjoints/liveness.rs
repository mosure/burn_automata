#![allow(clippy::too_many_arguments)]

use super::*;

pub(crate) fn add_liveness_front_state_adjoint(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    liveness_gain: f32,
    front_radius: f32,
    step_fraction: f32,
    max_adjoint: f32,
    state_adjoint: &mut [f32],
) {
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || positions.is_empty()
        || states.len() < positions.len() * config.state_dims
        || state_adjoint.len() < states.len()
        || liveness_gain <= 0.0
        || !liveness_gain.is_finite()
        || front_radius <= 0.0
        || !front_radius.is_finite()
    {
        return;
    }
    let schedule = step_fraction.clamp(0.0, 1.0);
    if schedule <= 0.0 {
        return;
    }
    let max_adjoint = if max_adjoint.is_finite() && max_adjoint > 0.0 {
        max_adjoint
    } else {
        f32::INFINITY
    };
    let front_weights = local_front_weights(config, positions, states, front_radius);
    for (row, front_weight) in front_weights.iter().copied().enumerate() {
        if front_weight <= 0.0 {
            continue;
        }
        let state_base = row * config.state_dims;
        let liveness_index = state_base + GROWTH_3D_LIVENESS_CHANNEL;
        let current_liveness = states[liveness_index];
        let scheduled_target = GROWTH_3D_INACTIVE_OPACITY_LOGIT
            + schedule
                * front_weight
                * (DEFAULT_3D_FIELD_OPACITY_TARGET - GROWTH_3D_INACTIVE_OPACITY_LOGIT);
        let target_liveness = if current_liveness > -1.0 {
            current_liveness.max(scheduled_target)
        } else {
            scheduled_target
        };
        let adjoint =
            (liveness_gain * (current_liveness - target_liveness)).clamp(-max_adjoint, max_adjoint);
        state_adjoint[liveness_index] += adjoint;
    }
}

pub(crate) fn liveness_front_temporal_target_updates(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    liveness_gain: f32,
    front_radius: f32,
    step_fraction: f32,
    max_update: f32,
) -> Vec<f32> {
    let mut updates = vec![0.0_f32; positions.len()];
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || positions.is_empty()
        || states.len() < positions.len() * config.state_dims
        || liveness_gain <= 0.0
        || !liveness_gain.is_finite()
        || front_radius <= 0.0
        || !front_radius.is_finite()
    {
        return updates;
    }
    let max_update = if max_update.is_finite() && max_update > 0.0 {
        max_update
    } else {
        f32::INFINITY
    };
    let schedule = step_fraction.clamp(0.0, 1.0);
    let front_weights = local_front_weights(config, positions, states, front_radius);
    for (row, front_weight) in front_weights.iter().copied().enumerate() {
        if front_weight <= 0.0 {
            continue;
        }
        let state_base = row * config.state_dims;
        let current_liveness = states[state_base + GROWTH_3D_LIVENESS_CHANNEL];
        let scheduled_target = GROWTH_3D_INACTIVE_OPACITY_LOGIT
            + schedule
                * front_weight
                * (DEFAULT_3D_FIELD_OPACITY_TARGET - GROWTH_3D_INACTIVE_OPACITY_LOGIT);
        let target_liveness = if current_liveness > -1.0 {
            current_liveness.max(scheduled_target)
        } else {
            scheduled_target
        };
        updates[row] += liveness_gain * (target_liveness - current_liveness);
    }

    let allowed_fraction = temporal_activation_allowed_fraction(schedule);
    let allowed_active = ((positions.len() as f32) * allowed_fraction).ceil() as usize;
    let mut active_rows = positions
        .iter()
        .enumerate()
        .filter_map(|(row, _)| {
            let liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
            (liveness > -1.0).then_some((row, liveness))
        })
        .collect::<Vec<_>>();
    if active_rows.len() > allowed_active {
        active_rows.sort_by(|(_, lhs), (_, rhs)| {
            lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal)
        });
        let suppress_count = active_rows.len().saturating_sub(allowed_active);
        for (row, current_liveness) in active_rows.into_iter().take(suppress_count) {
            updates[row] += liveness_gain * (-1.0 - current_liveness);
        }
    }

    for update in &mut updates {
        *update = update.clamp(-max_update, max_update);
    }
    updates
}
