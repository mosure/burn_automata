#![allow(clippy::too_many_arguments)]

use super::super::gradients::{RenderProxyGradientRows, render_position_gradient};
use super::*;

const DIRECT_ROLLOUT_MULTI_SEED_MIN_LOSS_WEIGHT: f32 = 0.25;
const DIRECT_ROLLOUT_MULTI_SEED_MAX_LOSS_WEIGHT: f32 = 8.0;
const DIRECT_ROLLOUT_MULTI_SEED_LOSS_FLOOR: f32 = 1.0e-6;

struct DirectRolloutSeedUpdate {
    weights: NpaWeights,
    report: TrainingRunReport,
}

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
    let mut seed_updates = Vec::with_capacity(training_seeds.len());
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
        seed_updates.push(DirectRolloutSeedUpdate {
            weights: candidate.weights,
            report,
        });
    }

    let seed_weights = direct_rollout_multiseed_loss_weights(
        &seed_updates
            .iter()
            .map(|update| update.report.initial_loss)
            .collect::<Vec<_>>(),
    );
    let mut delta = NpaWeights::zeros(&model.config);
    for (update, weight) in seed_updates.iter().zip(seed_weights.iter()) {
        accumulate_scaled_weight_delta(&mut delta, &base_weights, &update.weights, *weight);
    }
    apply_average_weight_delta(&mut model.weights, &base_weights, &delta, 1.0);

    let reports = seed_updates
        .iter()
        .map(|update| &update.report)
        .collect::<Vec<_>>();
    let rows = reports.iter().map(|report| report.rows).sum();
    let initial_loss = weighted_report_mean(&reports, &seed_weights, |report| report.initial_loss);
    let final_loss = render_direct_rollout_weighted_loss_for_seeds(
        model,
        grid,
        target,
        cfg,
        render_cfg,
        &render_direct_rollout_training_seeds(cfg, round),
        &seed_weights,
    )?;
    let best_loss = initial_loss.min(final_loss);
    let grad_norm = weighted_report_mean(&reports, &seed_weights, |report| {
        report
            .history
            .last()
            .map(|entry| entry.grad_norm)
            .unwrap_or(0.0)
    });
    let grad_scale = weighted_report_mean(&reports, &seed_weights, |report| {
        report
            .history
            .last()
            .map(|entry| entry.grad_scale)
            .unwrap_or(1.0)
    });

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

pub(crate) fn direct_rollout_multiseed_loss_weights(losses: &[f32]) -> Vec<f32> {
    if losses.is_empty() {
        return Vec::new();
    }
    let minimum_loss = losses
        .iter()
        .copied()
        .filter(|loss| loss.is_finite() && *loss >= 0.0)
        .fold(f32::INFINITY, f32::min);
    if !minimum_loss.is_finite() {
        return vec![(losses.len() as f32).recip(); losses.len()];
    }

    let loss_floor = minimum_loss.max(DIRECT_ROLLOUT_MULTI_SEED_LOSS_FLOOR);
    let mut weights = losses
        .iter()
        .map(|loss| {
            if !loss.is_finite() || *loss < 0.0 {
                1.0
            } else {
                ((*loss + loss_floor) / (minimum_loss + loss_floor)).clamp(
                    DIRECT_ROLLOUT_MULTI_SEED_MIN_LOSS_WEIGHT,
                    DIRECT_ROLLOUT_MULTI_SEED_MAX_LOSS_WEIGHT,
                )
            }
        })
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f32>();
    if total.is_finite() && total > 0.0 {
        for weight in &mut weights {
            *weight /= total;
        }
    } else {
        let uniform = (weights.len().max(1) as f32).recip();
        weights.fill(uniform);
    }
    weights
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

pub(crate) fn render_direct_rollout_weighted_loss_for_seeds(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: &RenderProxyTrainingConfig,
    mut render_cfg: RenderLossConfig,
    seeds: &[u64],
    weights: &[f32],
) -> Result<f32, Box<dyn std::error::Error>> {
    if render_cfg.target_samples == 0 {
        render_cfg.target_samples = cfg.particles;
    }
    let mut total = 0.0_f32;
    let mut weight_sum = 0.0_f32;
    for (index, &seed) in seeds.iter().enumerate() {
        let weight = weights.get(index).copied().unwrap_or(0.0);
        if weight <= 0.0 || !weight.is_finite() {
            continue;
        }
        let trace = render_training_trace_for_seed(model, grid, cfg, seed)?;
        total +=
            weight * mesh_multiview_render_loss_from_trace(&trace, target, render_cfg)?.total_loss;
        weight_sum += weight;
    }
    if weight_sum > 0.0 {
        Ok(total / weight_sum)
    } else {
        Ok(f32::INFINITY)
    }
}

fn weighted_report_mean(
    reports: &[&TrainingRunReport],
    weights: &[f32],
    value: impl Fn(&TrainingRunReport) -> f32,
) -> f32 {
    let mut total = 0.0_f32;
    let mut weight_sum = 0.0_f32;
    for (index, report) in reports.iter().enumerate() {
        let weight = weights.get(index).copied().unwrap_or(0.0);
        if weight <= 0.0 || !weight.is_finite() {
            continue;
        }
        let value = value(report);
        if value.is_finite() {
            total += weight * value;
            weight_sum += weight;
        }
    }
    if weight_sum > 0.0 {
        total / weight_sum
    } else {
        f32::INFINITY
    }
}
