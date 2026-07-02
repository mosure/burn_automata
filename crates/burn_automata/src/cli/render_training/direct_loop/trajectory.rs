use super::*;

pub(crate) fn render_training_trace(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    cfg: &RenderProxyTrainingConfig,
    round: usize,
) -> Result<crate::RolloutTrace, Box<dyn std::error::Error>> {
    render_training_trace_for_seed(model, grid, cfg, render_training_round_seed(cfg, round))
}

pub(crate) fn render_training_round_seed(cfg: &RenderProxyTrainingConfig, round: usize) -> u64 {
    cfg.seed
        .wrapping_add((round as u64).wrapping_mul(0x9e37_79b9))
}

#[derive(Clone, Debug)]
pub(crate) struct RenderTrajectorySnapshot {
    pub(crate) positions: Vec<[f32; 4]>,
    pub(crate) states: Vec<f32>,
    pub(crate) features: Vec<f32>,
    pub(crate) step_fraction: f32,
}

pub(crate) fn render_training_trajectory(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    cfg: &RenderProxyTrainingConfig,
    round: usize,
) -> Result<(crate::RolloutTrace, Vec<RenderTrajectorySnapshot>), Box<dyn std::error::Error>> {
    render_training_trajectory_for_seed(model, grid, cfg, render_training_round_seed(cfg, round))
}

pub(crate) fn render_training_trajectory_for_seed(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    cfg: &RenderProxyTrainingConfig,
    seed: u64,
) -> Result<(crate::RolloutTrace, Vec<RenderTrajectorySnapshot>), Box<dyn std::error::Error>> {
    let rollout_cfg = RolloutConfig {
        particle_count: cfg.particles,
        steps: cfg.rollout_steps,
        update_prob: 1.0,
        seed,
        seed_scale: cfg.seed_scale,
        ..RolloutConfig::default()
    };
    let (mut positions, mut states) = seed_particles_scaled(
        rollout_cfg.batch_size,
        rollout_cfg.particle_count,
        model.config.state_dims,
        model.config.spatial_dims,
        rollout_cfg.seed,
        cfg.seed_mode,
        rollout_cfg.seed_scale,
    );
    let mut mean_dx = Vec::with_capacity(rollout_cfg.steps);
    let mut snapshots = Vec::with_capacity(rollout_cfg.steps);

    for step_idx in 0..rollout_cfg.steps {
        let step = model.step_cpu(
            &positions,
            &states,
            rollout_cfg.batch_size,
            rollout_cfg.particle_count,
            grid,
            rollout_cfg.dt,
            None,
        )?;
        let dx_norm = step
            .dx
            .iter()
            .map(|delta| (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt())
            .sum::<f32>()
            / step.dx.len().max(1) as f32;
        mean_dx.push(dx_norm);
        snapshots.push(RenderTrajectorySnapshot {
            positions: positions.clone(),
            states: states.clone(),
            features: step.perception.features.clone(),
            step_fraction: (step_idx + 1) as f32 / rollout_cfg.steps.max(1) as f32,
        });
        positions = step.next_positions;
        states = step.next_states;
    }

    Ok((
        crate::RolloutTrace {
            positions,
            states,
            batch_size: rollout_cfg.batch_size,
            particle_count: rollout_cfg.particle_count,
            state_dims: model.config.state_dims,
            steps: rollout_cfg.steps,
            mean_dx,
        },
        snapshots,
    ))
}

pub(crate) fn render_training_trace_for_seed(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    cfg: &RenderProxyTrainingConfig,
    seed: u64,
) -> Result<crate::RolloutTrace, Box<dyn std::error::Error>> {
    Ok(run_rollout(
        model,
        grid,
        &RolloutConfig {
            particle_count: cfg.particles,
            steps: cfg.rollout_steps,
            update_prob: 1.0,
            seed,
            seed_scale: cfg.seed_scale,
            ..RolloutConfig::default()
        },
        cfg.seed_mode,
    )?)
}
