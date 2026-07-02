use super::*;

pub(crate) fn growth_3d_dormant_drift_threshold(seed_scale: f32) -> f32 {
    growth_3d_seed_radius(seed_scale) * 1.25
}

pub(crate) fn growth_3d_dormant_drift_report(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    rollout_cfg: RolloutConfig,
    seed_mode: ParticleSeed,
    seed_positions: &[[f32; 4]],
    seed_states: &[f32],
    final_trace: &crate::RolloutTrace,
) -> Result<Growth3dDormantDriftReport, Box<dyn std::error::Error>> {
    let seed_snapshot =
        growth_3d_front_snapshot(seed_positions, seed_states, model.config.state_dims);
    let max_front_distance = growth_3d_front_distance_threshold(rollout_cfg.seed_scale);
    let max_allowed_displacement = growth_3d_dormant_drift_threshold(rollout_cfg.seed_scale);
    let mut sampled_steps = 0usize;
    let mut checked_rows = 0usize;
    let mut drifting_rows = 0usize;
    let mut displacement_sum = 0.0_f32;
    let mut max_dormant_displacement = 0.0_f32;
    let mut finite = true;

    for steps in growth_3d_temporal_sample_steps(rollout_cfg.steps) {
        let snapshot = if steps == 0 {
            seed_snapshot.clone()
        } else if steps == rollout_cfg.steps {
            growth_3d_front_snapshot(
                &final_trace.positions,
                &final_trace.states,
                final_trace.state_dims,
            )
        } else {
            let trace = run_rollout(
                model,
                grid,
                &RolloutConfig {
                    steps,
                    ..rollout_cfg.clone()
                },
                seed_mode,
            )?;
            growth_3d_front_snapshot(&trace.positions, &trace.states, trace.state_dims)
        };
        sampled_steps += 1;
        if snapshot.positions.len() != seed_snapshot.positions.len()
            || snapshot.active.len() != seed_snapshot.active.len()
        {
            finite = false;
            continue;
        }
        let active_positions = snapshot
            .positions
            .iter()
            .zip(snapshot.active.iter())
            .filter_map(|(position, active)| (*active).then_some(*position))
            .collect::<Vec<_>>();
        if active_positions.is_empty() {
            continue;
        }
        for row in 0..snapshot.positions.len() {
            if seed_snapshot.active[row] || snapshot.active[row] {
                continue;
            }
            let nearest_active =
                nearest_position_distance(snapshot.positions[row], &active_positions);
            finite &= nearest_active.is_finite();
            if nearest_active <= max_front_distance {
                continue;
            }
            let displacement =
                position_distance(seed_snapshot.positions[row], snapshot.positions[row]);
            finite &= displacement.is_finite();
            checked_rows += 1;
            displacement_sum += displacement;
            max_dormant_displacement = max_dormant_displacement.max(displacement);
            if displacement > max_allowed_displacement {
                drifting_rows += 1;
            }
        }
    }

    let drifting_fraction = drifting_rows as f32 / checked_rows.max(1) as f32;
    let mean_dormant_displacement = if checked_rows > 0 {
        displacement_sum / checked_rows as f32
    } else {
        0.0
    };
    let passed = finite
        && (checked_rows == 0
            || (drifting_rows == 0 && max_dormant_displacement <= max_allowed_displacement));

    Ok(Growth3dDormantDriftReport {
        sampled_steps,
        checked_rows,
        drifting_rows,
        drifting_fraction,
        mean_dormant_displacement,
        max_dormant_displacement,
        max_allowed_displacement,
        finite,
        passed,
    })
}

pub(crate) fn apply_dormant_drift_strict_check(
    checks: &mut Growth3dStrictChecksReport,
    dormant_drift: Growth3dDormantDriftReport,
) {
    checks.dormant_drift_bounded = dormant_drift.passed;
    if !dormant_drift.passed && !checks.failure_reasons.contains(&"dormant_drift_bounded") {
        checks.failure_reasons.push("dormant_drift_bounded");
    }
    checks.passed &= dormant_drift.passed;
}

fn position_distance(lhs: [f32; 4], rhs: [f32; 4]) -> f32 {
    ((lhs[0] - rhs[0]).powi(2) + (lhs[1] - rhs[1]).powi(2) + (lhs[2] - rhs[2]).powi(2)).sqrt()
}
