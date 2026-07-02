#![allow(clippy::too_many_arguments)]

use super::super::gradients::{RenderProxyGradientRows, render_position_gradient};
use super::*;

pub(crate) fn render_direct_rollout_multiseed_training_step(
    model: &mut NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: &RenderProxyTrainingConfig,
    round: usize,
    trace: &crate::RolloutTrace,
    trajectory: &[RenderTrajectorySnapshot],
    gradient: &RenderProxyGradientRows,
) -> Result<TrainingRunReport, Box<dyn std::error::Error>> {
    let training_seeds = render_direct_rollout_training_seeds(cfg, round);

    let base_model = model.clone();
    let base_weights = model.weights.clone();
    let mut delta = NpaWeights::zeros(&model.config);
    let mut reports = Vec::with_capacity(training_seeds.len());
    let mut render_cfg = cfg.render;
    if render_cfg.target_samples == 0 {
        render_cfg.target_samples = cfg.particles;
    }

    for seed in training_seeds {
        let mut candidate = base_model.clone();
        let report = if seed == render_training_round_seed(cfg, round) {
            render_direct_rollout_training_step(
                &mut candidate,
                grid,
                target,
                trace,
                trajectory,
                gradient,
                cfg,
                seed,
            )?
        } else {
            let (seed_trace, seed_trajectory) =
                render_training_trajectory_for_seed(&candidate, grid, cfg, seed)?;
            let seed_gradient = render_position_gradient(&seed_trace, target, render_cfg, cfg)?;
            render_direct_rollout_training_step(
                &mut candidate,
                grid,
                target,
                &seed_trace,
                &seed_trajectory,
                &seed_gradient,
                cfg,
                seed,
            )?
        };
        accumulate_weight_delta(&mut delta, &base_weights, &candidate.weights);
        reports.push(report);
    }

    let count = reports.len().max(1) as f32;
    apply_average_weight_delta(&mut model.weights, &base_weights, &delta, count.recip());

    let rows = reports.iter().map(|report| report.rows).sum();
    let initial_loss = reports
        .iter()
        .map(|report| report.initial_loss)
        .sum::<f32>()
        / count;
    let final_loss = render_direct_rollout_average_loss_for_seeds(
        model,
        grid,
        target,
        cfg,
        render_cfg,
        &render_direct_rollout_training_seeds(cfg, round),
    )?;
    let best_loss = initial_loss.min(final_loss);
    let grad_norm = reports
        .iter()
        .filter_map(|report| report.history.last())
        .map(|entry| entry.grad_norm)
        .sum::<f32>()
        / count;
    let grad_scale = reports
        .iter()
        .filter_map(|report| report.history.last())
        .map(|entry| entry.grad_scale)
        .sum::<f32>()
        / count;

    Ok(TrainingRunReport {
        steps: 1,
        rows,
        initial_loss,
        final_loss,
        best_loss,
        history: vec![TrainingHistoryEntry {
            step: 1,
            loss: final_loss,
            grad_norm,
            grad_scale,
        }],
    })
}

pub(crate) fn render_direct_rollout_training_seeds(
    cfg: &RenderProxyTrainingConfig,
    round: usize,
) -> Vec<u64> {
    let round_seed = render_training_round_seed(cfg, round);
    let mut training_seeds = vec![round_seed];
    for seed in render_proxy_selection_seeds(cfg) {
        if !training_seeds.contains(&seed) {
            training_seeds.push(seed);
        }
    }
    training_seeds
}

pub(crate) fn render_direct_rollout_average_loss_for_seeds(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: &RenderProxyTrainingConfig,
    mut render_cfg: RenderLossConfig,
    seeds: &[u64],
) -> Result<f32, Box<dyn std::error::Error>> {
    if render_cfg.target_samples == 0 {
        render_cfg.target_samples = cfg.particles;
    }
    let mut total = 0.0_f32;
    let mut count = 0usize;
    for &seed in seeds {
        let trace = render_training_trace_for_seed(model, grid, cfg, seed)?;
        total += mesh_multiview_render_loss_from_trace(&trace, target, render_cfg)?.total_loss;
        count += 1;
    }
    if count == 0 {
        Ok(f32::INFINITY)
    } else {
        Ok(total / count as f32)
    }
}
