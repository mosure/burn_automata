use super::*;

pub(crate) fn add_temporal_activation_schedule_state_adjoint(
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
    let allowed_fraction = temporal_activation_allowed_fraction(step_fraction);
    let allowed_active = ((positions.len() as f32) * allowed_fraction).ceil() as usize;
    let mut active_rows = positions
        .iter()
        .enumerate()
        .filter_map(|(row, _)| {
            let liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
            (liveness > -1.0).then_some((row, liveness))
        })
        .collect::<Vec<_>>();
    let max_adjoint = if max_adjoint.is_finite() && max_adjoint > 0.0 {
        max_adjoint
    } else {
        f32::INFINITY
    };
    let active_count = active_rows.len();
    if active_count > allowed_active {
        active_rows.sort_by(|(_, lhs), (_, rhs)| {
            lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal)
        });
        let suppress_count = active_count.saturating_sub(allowed_active);
        for (row, current_liveness) in active_rows.iter().copied().take(suppress_count) {
            let target = -1.0_f32;
            let adjoint = (liveness_gain * (current_liveness - target)).clamp(0.0, max_adjoint);
            state_adjoint[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] += adjoint;
        }
    }

    if schedule <= 0.0 {
        return;
    }
    let target_fraction = temporal_activation_target_fraction(schedule);
    let target_active = ((positions.len() as f32) * target_fraction).ceil() as usize;
    if active_count >= target_active {
        return;
    }
    let deficit = target_active.saturating_sub(active_count);
    let temporal_front_candidates = temporal_front_candidate_count(positions.len(), deficit);
    let front_weights = local_front_weights_with_min_candidates(
        config,
        positions,
        states,
        front_radius,
        temporal_front_candidates,
    );
    let mut candidates = front_weights
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(row, front_weight)| {
            let liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
            (liveness <= -1.0 && front_weight > 0.0).then_some((row, front_weight, liveness))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(
        |(_, lhs_weight, lhs_liveness), (_, rhs_weight, rhs_liveness)| {
            rhs_weight
                .partial_cmp(lhs_weight)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    rhs_liveness
                        .partial_cmp(lhs_liveness)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        },
    );
    let target_liveness = temporal_activation_candidate_liveness_target(schedule);
    for (row, front_weight, current_liveness) in candidates.into_iter().take(deficit) {
        let adjoint = (liveness_gain * front_weight * (current_liveness - target_liveness))
            .clamp(-max_adjoint, 0.0);
        state_adjoint[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] += adjoint;
    }
}

pub(crate) fn active_liveness_count(states: &[f32], rows: usize, state_dims: usize) -> usize {
    if state_dims <= GROWTH_3D_LIVENESS_CHANNEL || states.len() < rows * state_dims {
        return 0;
    }
    (0..rows)
        .filter(|row| states[row * state_dims + GROWTH_3D_LIVENESS_CHANNEL] > -1.0)
        .count()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_temporal_activation_jump_state_adjoint(
    config: &NpaConfig,
    previous_positions: &[[f32; 4]],
    previous_states: &[f32],
    current_states: &[f32],
    liveness_gain: f32,
    front_radius: f32,
    previous_step_fraction: f32,
    current_step_fraction: f32,
    max_adjoint: f32,
    previous_state_adjoint: &mut [f32],
    current_state_adjoint: &mut [f32],
) {
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || previous_positions.is_empty()
        || previous_states.len() < previous_positions.len() * config.state_dims
        || current_states.len() < previous_positions.len() * config.state_dims
        || previous_state_adjoint.len() < previous_states.len()
        || current_state_adjoint.len() < current_states.len()
        || liveness_gain <= 0.0
        || !liveness_gain.is_finite()
        || front_radius <= 0.0
        || !front_radius.is_finite()
    {
        return;
    }
    let rows = previous_positions.len();
    let previous_active = active_liveness_count(previous_states, rows, config.state_dims);
    let current_active = active_liveness_count(current_states, rows, config.state_dims);
    if current_active <= previous_active {
        return;
    }
    let previous_target = temporal_activation_target_fraction(previous_step_fraction);
    let current_target = temporal_activation_target_fraction(current_step_fraction);
    let scheduled_delta = (current_target - previous_target).max(0.0);
    let actual_delta = (current_active - previous_active) as f32 / rows as f32;
    let excess_delta = (actual_delta - scheduled_delta - TEMPORAL_ACTIVATION_JUMP_SLACK).max(0.0);
    if excess_delta <= 0.0 {
        return;
    }
    let excess_count = ((excess_delta * rows as f32).ceil() as usize).max(1);
    let max_adjoint = if max_adjoint.is_finite() && max_adjoint > 0.0 {
        max_adjoint
    } else {
        f32::INFINITY
    };
    let front_weights = local_front_weights_with_min_candidates(
        config,
        previous_positions,
        previous_states,
        front_radius,
        excess_count,
    );
    let mut newly_active = (0..rows)
        .filter_map(|row| {
            let previous_liveness =
                previous_states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
            let current_liveness =
                current_states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
            (previous_liveness <= -1.0 && current_liveness > -1.0).then_some((
                row,
                front_weights.get(row).copied().unwrap_or(0.0),
                previous_liveness,
                current_liveness,
            ))
        })
        .collect::<Vec<_>>();
    newly_active.sort_by(
        |(_, lhs_front, lhs_previous, lhs_current), (_, rhs_front, rhs_previous, rhs_current)| {
            rhs_front
                .partial_cmp(lhs_front)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    rhs_previous
                        .partial_cmp(lhs_previous)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| {
                    lhs_current
                        .partial_cmp(rhs_current)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        },
    );
    let previous_target = temporal_activation_candidate_liveness_target(previous_step_fraction);
    for (row, front_weight, previous_liveness, current_liveness) in
        newly_active.into_iter().take(excess_count)
    {
        if front_weight > 0.0 {
            let previous_adjoint =
                (liveness_gain * front_weight * (previous_liveness - previous_target))
                    .clamp(-max_adjoint, 0.0);
            previous_state_adjoint[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] +=
                previous_adjoint;
        }
        let current_adjoint =
            (liveness_gain * (current_liveness - -1.0_f32)).clamp(0.0, max_adjoint);
        current_state_adjoint[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] +=
            current_adjoint;
    }
}

pub(crate) fn temporal_activation_allowed_fraction(step_fraction: f32) -> f32 {
    (temporal_activation_target_fraction(step_fraction) + 0.10).clamp(0.10, 0.75)
}

pub(crate) fn temporal_activation_schedule_error(
    temporal: &Growth3dTemporalReport,
    rollout_steps: usize,
) -> f32 {
    if temporal.samples.is_empty() || rollout_steps == 0 {
        return 1.0;
    }
    let rollout_steps = rollout_steps.max(1) as f32;
    let mut error = 0.0_f32;
    let mut count = 0usize;
    let mut previous_active_fraction: Option<f32> = None;
    let mut previous_target_fraction: Option<f32> = None;
    for sample in &temporal.samples {
        let step_fraction = (sample.steps as f32 / rollout_steps).clamp(0.0, 1.0);
        let active_fraction = sample.active_fraction.clamp(0.0, 1.0);
        let target = temporal_activation_target_fraction(step_fraction);
        let allowed = temporal_activation_allowed_fraction(step_fraction);
        let over_active = (active_fraction - allowed).max(0.0);
        let under_active = (target - active_fraction).max(0.0);
        error += 2.0 * over_active + under_active;
        count += 1;
        if let (Some(previous_active), Some(previous_target)) =
            (previous_active_fraction, previous_target_fraction)
        {
            let actual_delta = (active_fraction - previous_active).max(0.0);
            let scheduled_delta = (target - previous_target).max(0.0);
            let jump_error =
                (actual_delta - scheduled_delta - TEMPORAL_ACTIVATION_JUMP_SLACK).max(0.0);
            error += 4.0 * jump_error;
            count += 1;
        }
        previous_active_fraction = Some(active_fraction);
        previous_target_fraction = Some(target);
    }
    if count == 0 {
        1.0
    } else {
        error / count as f32
    }
}

pub(crate) fn temporal_activation_target_fraction(step_fraction: f32) -> f32 {
    let schedule = step_fraction.clamp(0.0, 1.0);
    if schedule <= 0.5 {
        (0.02 + 0.96 * schedule).clamp(0.02, 0.50)
    } else {
        let late = ((schedule - 0.5) * 2.0).clamp(0.0, 1.0);
        (0.50 + 0.15 * late).clamp(0.50, 0.65)
    }
}

pub(crate) fn temporal_activation_candidate_liveness_target(step_fraction: f32) -> f32 {
    let schedule = step_fraction.clamp(0.0, 1.0);
    (-0.25 + 1.25 * schedule * schedule).clamp(-0.25, 1.0)
}

pub(crate) fn temporal_materialization_target_logit(step_fraction: f32) -> f32 {
    let schedule = step_fraction.clamp(0.0, 1.0);
    let visible_threshold = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT + 1.0;
    let target = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    (visible_threshold + (target - visible_threshold) * schedule * schedule)
        .clamp(visible_threshold, target)
}

pub(crate) fn temporal_front_candidate_count(rows: usize, deficit: usize) -> usize {
    if rows == 0 || deficit == 0 {
        return 0;
    }
    let row_fraction = if rows < TEMPORAL_FRONT_CANDIDATE_WIDE_MIN_ROWS {
        TEMPORAL_FRONT_CANDIDATE_SMALL_ROW_FRACTION
    } else {
        TEMPORAL_FRONT_CANDIDATE_ROW_FRACTION
    };
    let local_budget = rows.div_ceil(row_fraction);
    let scaled_cap = rows.div_ceil(row_fraction).clamp(
        TEMPORAL_FRONT_CANDIDATE_MIN_CAP,
        TEMPORAL_FRONT_CANDIDATE_MAX_CAP,
    );
    deficit.min(local_budget.min(scaled_cap).max(1))
}
