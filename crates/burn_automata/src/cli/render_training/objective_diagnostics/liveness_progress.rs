use super::*;

pub(crate) const LOCAL_FRONT_LIVENESS_SCORE_WEIGHT: f32 = 0.01;
pub(crate) const MATERIAL_VISIBLE_TARGET_MEAN_DISTANCE_SCORE_WEIGHT: f32 = 0.05;
pub(crate) const MATERIAL_VISIBLE_TARGET_MAX_DISTANCE_SCORE_WEIGHT: f32 = 0.02;
pub(crate) const MATERIAL_VISIBLE_TARGET_DISTANCE_REGRESSION_SLACK: f32 = 0.02;
pub(crate) const TEMPORAL_ACTIVATION_SCORE_WEIGHT: f32 = 25.0;
pub(crate) const TEMPORAL_ACTIVATION_SELECTION_REGRESSION_SLACK: f32 = 0.02;
pub(crate) const TEMPORAL_LIVENESS_TRAJECTORY_SAMPLE_CAP: usize = 8;
pub(crate) const DIRECT_TRAJECTORY_TERMINAL_LIVENESS_WEIGHT: f32 = 0.25;
pub(crate) const TEMPORAL_ACTIVATION_JUMP_SLACK: f32 = 0.10;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LocalFrontLivenessProgress {
    pub(crate) candidate_count: usize,
    pub(crate) weighted_activation_margin: f32,
}

pub(crate) fn liveness_progress_from_candidate_weights(
    config: &NpaConfig,
    states: &[f32],
    candidate_weights: &[f32],
) -> LocalFrontLivenessProgress {
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL || candidate_weights.is_empty() {
        return LocalFrontLivenessProgress::default();
    }
    let rows = candidate_weights
        .len()
        .min(states.len() / config.state_dims.max(1));
    let mut candidate_count = 0usize;
    let mut weighted_margin = 0.0_f32;
    let mut weight_sum = 0.0_f32;
    for (row, candidate_weight) in candidate_weights.iter().take(rows).copied().enumerate() {
        let liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
        if liveness > -1.0 || candidate_weight <= 0.0 || !candidate_weight.is_finite() {
            continue;
        }
        candidate_count += 1;
        weight_sum += candidate_weight;
        weighted_margin += candidate_weight * (-1.0 - liveness).max(0.0);
    }

    LocalFrontLivenessProgress {
        candidate_count,
        weighted_activation_margin: if weight_sum > 0.0 {
            weighted_margin / weight_sum
        } else {
            0.0
        },
    }
}

pub(crate) fn local_front_liveness_progress(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    front_radius: f32,
) -> LocalFrontLivenessProgress {
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || positions.is_empty()
        || states.len() < positions.len() * config.state_dims
        || front_radius <= 0.0
        || !front_radius.is_finite()
    {
        return LocalFrontLivenessProgress::default();
    }

    let front_weights = local_front_weights(config, positions, states, front_radius);
    liveness_progress_from_candidate_weights(config, states, &front_weights)
}

pub(crate) fn extent_front_liveness_progress(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    front_radius: f32,
) -> LocalFrontLivenessProgress {
    let weights =
        extent_front_liveness_candidate_weights(config, target, positions, states, front_radius);
    liveness_progress_from_candidate_weights(config, states, &weights)
}

pub(crate) fn direct_terminal_liveness_gain(cfg: &RenderProxyTrainingConfig) -> f32 {
    if cfg.trajectory_supervision
        && cfg.liveness_gain > 0.0
        && cfg.liveness_gain.is_finite()
        && DIRECT_TRAJECTORY_TERMINAL_LIVENESS_WEIGHT.is_finite()
    {
        cfg.liveness_gain * DIRECT_TRAJECTORY_TERMINAL_LIVENESS_WEIGHT
    } else {
        cfg.liveness_gain
    }
}

pub(crate) fn temporal_front_liveness_progress(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    cfg: &RenderProxyTrainingConfig,
    seed: u64,
    seed_positions: &[[f32; 4]],
    seed_states: &[f32],
) -> Result<LocalFrontLivenessProgress, Box<dyn std::error::Error>> {
    let rollout_steps = cfg.rollout_steps.max(1);
    let mut candidate_count = 0usize;
    let mut worst_margin = 0.0_f32;
    for steps in growth_3d_temporal_sample_steps(rollout_steps) {
        if steps == 0 || steps >= rollout_steps {
            continue;
        }
        let (positions, states) = if steps == 0 {
            (seed_positions.to_vec(), seed_states.to_vec())
        } else {
            let trace = run_rollout(
                model,
                grid,
                &RolloutConfig {
                    particle_count: cfg.particles,
                    steps,
                    update_prob: 1.0,
                    seed,
                    seed_scale: cfg.seed_scale,
                    ..RolloutConfig::default()
                },
                cfg.seed_mode,
            )?;
            (trace.positions, trace.states)
        };
        let rows = positions.len();
        let active_count = active_liveness_count(&states, rows, model.config.state_dims);
        let schedule = (steps as f32 / rollout_steps as f32).clamp(0.0, 1.0);
        let target_active =
            ((rows as f32) * temporal_activation_target_fraction(schedule)).ceil() as usize;
        if active_count >= target_active {
            continue;
        }
        let progress = local_front_liveness_progress(
            &model.config,
            &positions,
            &states,
            cfg.liveness_front_radius,
        );
        if progress.candidate_count == 0 {
            continue;
        }
        candidate_count += progress.candidate_count;
        worst_margin = worst_margin.max(progress.weighted_activation_margin);
    }
    Ok(LocalFrontLivenessProgress {
        candidate_count,
        weighted_activation_margin: worst_margin,
    })
}

pub(crate) fn temporal_extent_front_liveness_progress(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: &RenderProxyTrainingConfig,
    seed: u64,
    seed_positions: &[[f32; 4]],
    seed_states: &[f32],
) -> Result<LocalFrontLivenessProgress, Box<dyn std::error::Error>> {
    let rollout_steps = cfg.rollout_steps.max(1);
    let mut candidate_count = 0usize;
    let mut worst_margin = 0.0_f32;
    for steps in growth_3d_temporal_sample_steps(rollout_steps) {
        if steps == 0 || steps >= rollout_steps {
            continue;
        }
        let (positions, states) = if steps == 0 {
            (seed_positions.to_vec(), seed_states.to_vec())
        } else {
            let trace = run_rollout(
                model,
                grid,
                &RolloutConfig {
                    particle_count: cfg.particles,
                    steps,
                    update_prob: 1.0,
                    seed,
                    seed_scale: cfg.seed_scale,
                    ..RolloutConfig::default()
                },
                cfg.seed_mode,
            )?;
            (trace.positions, trace.states)
        };
        let progress = extent_front_liveness_progress(
            &model.config,
            target,
            &positions,
            &states,
            cfg.liveness_front_radius,
        );
        if progress.candidate_count == 0 {
            continue;
        }
        candidate_count += progress.candidate_count;
        worst_margin = worst_margin.max(progress.weighted_activation_margin);
    }
    Ok(LocalFrontLivenessProgress {
        candidate_count,
        weighted_activation_margin: worst_margin,
    })
}

pub(crate) fn gaussian_volume_stats_for_trace(
    trace: &crate::RolloutTrace,
    render: RenderLossConfig,
) -> GaussianVolumeStats {
    GaussianVolumeStats::from_render_config(trace, render)
}
