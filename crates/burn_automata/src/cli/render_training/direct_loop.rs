use super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn render_direct_rollout_training_steps(
    model: &mut NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: &RenderProxyTrainingConfig,
    round: usize,
    render_cfg: RenderLossConfig,
    selection_baseline: &[RenderSelectionBaselineCase],
) -> Result<(TrainingRunReport, f32), Box<dyn std::error::Error>> {
    let steps = cfg.supervised_steps_per_round.max(1);
    let mut reports = Vec::with_capacity(steps);
    let mut step_scale_sum = 0.0_f32;
    let mut best_inner_model = model.clone();
    let mut best_inner_selection = if cfg.direct_line_search {
        Some(render_selection_metrics(
            model,
            grid,
            target,
            cfg,
            render_cfg,
            Some(selection_baseline),
        )?)
    } else {
        None
    };
    let mut selected_inner_checkpoint = false;
    for _ in 0..steps {
        let (trace, trajectory) = render_training_trajectory(model, grid, cfg, round)?;
        let gradient = render_position_gradient(&trace, target, render_cfg, cfg)?;
        let (report, step_scale) = if cfg.direct_line_search {
            render_direct_rollout_training_step_with_line_search(
                model,
                grid,
                target,
                cfg,
                round,
                &trace,
                &trajectory,
                &gradient,
                render_cfg,
                selection_baseline,
            )?
        } else {
            let report = if cfg.direct_selection_seed_training {
                render_direct_rollout_multiseed_training_step(
                    model,
                    grid,
                    target,
                    cfg,
                    round,
                    &trace,
                    &trajectory,
                    &gradient,
                )?
            } else {
                render_direct_rollout_training_step(
                    model,
                    grid,
                    target,
                    &trace,
                    &trajectory,
                    &gradient,
                    cfg,
                    render_training_round_seed(cfg, round),
                )?
            };
            (report, 1.0)
        };
        reports.push(report);
        step_scale_sum += step_scale;
        if cfg.direct_line_search {
            let selection = render_selection_metrics(
                model,
                grid,
                target,
                cfg,
                render_cfg,
                Some(selection_baseline),
            )?;
            let should_retain = best_inner_selection
                .as_ref()
                .map(|best| render_selection_candidate_metrics_beats(&selection, best))
                .unwrap_or(true);
            if should_retain {
                best_inner_model = model.clone();
                best_inner_selection = Some(selection);
                selected_inner_checkpoint = true;
            } else if best_inner_selection
                .as_ref()
                .is_some_and(|best| render_selection_training_progress_beats(&selection, best))
            {
                // Keep a bounded non-promotable candidate for subsequent inner
                // optimization steps. Strict best checkpointing remains
                // unchanged and is restored below once a strict checkpoint
                // exists.
            } else {
                *model = best_inner_model.clone();
            }
        }
    }
    let mut report = combine_direct_rollout_training_reports(&reports);
    if cfg.direct_line_search && selected_inner_checkpoint {
        *model = best_inner_model;
        let final_trace = render_training_trace_for_seed(
            model,
            grid,
            cfg,
            render_training_round_seed(cfg, round),
        )?;
        report.final_loss =
            mesh_multiview_render_loss_from_trace(&final_trace, target, render_cfg)?.total_loss;
        report.best_loss = report.best_loss.min(report.final_loss);
    }
    Ok((report, step_scale_sum / steps as f32))
}

pub(crate) fn combine_direct_rollout_training_reports(
    reports: &[TrainingRunReport],
) -> TrainingRunReport {
    let mut history = Vec::new();
    let mut rows = 0usize;
    let mut best_loss = f32::INFINITY;
    let mut initial_loss = f32::INFINITY;
    let mut final_loss = f32::INFINITY;
    for report in reports {
        if history.is_empty() {
            initial_loss = report.initial_loss;
        }
        rows += report.rows;
        best_loss = best_loss.min(report.best_loss);
        final_loss = report.final_loss;
        for entry in &report.history {
            history.push(TrainingHistoryEntry {
                step: history.len() + 1,
                loss: entry.loss,
                grad_norm: entry.grad_norm,
                grad_scale: entry.grad_scale,
            });
        }
    }
    if reports.is_empty() {
        initial_loss = f32::INFINITY;
        final_loss = f32::INFINITY;
        best_loss = f32::INFINITY;
    }
    TrainingRunReport {
        steps: history.len(),
        rows,
        initial_loss,
        final_loss,
        best_loss,
        history,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn render_direct_rollout_training_step_with_line_search(
    model: &mut NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: &RenderProxyTrainingConfig,
    round: usize,
    trace: &crate::RolloutTrace,
    trajectory: &[RenderTrajectorySnapshot],
    gradient: &RenderProxyGradientRows,
    render_cfg: RenderLossConfig,
    selection_baseline: &[RenderSelectionBaselineCase],
) -> Result<(TrainingRunReport, f32), Box<dyn std::error::Error>> {
    let scales = sanitized_direct_line_search_scales(cfg);
    if scales.is_empty() {
        let report = if cfg.direct_selection_seed_training {
            render_direct_rollout_multiseed_training_step(
                model, grid, target, cfg, round, trace, trajectory, gradient,
            )?
        } else {
            render_direct_rollout_training_step(
                model,
                grid,
                target,
                trace,
                trajectory,
                gradient,
                cfg,
                render_training_round_seed(cfg, round),
            )?
        };
        return Ok((report, 1.0));
    }

    let base_model = model.clone();
    let initial_loss = mesh_multiview_render_loss_from_trace(trace, target, render_cfg)?.total_loss;
    let no_op_selection = render_selection_metrics(
        model,
        grid,
        target,
        cfg,
        render_cfg,
        Some(selection_baseline),
    )?;
    let mut best_model = base_model.clone();
    let mut best_report = render_direct_rollout_noop_report(initial_loss, gradient);
    let mut best_selection = no_op_selection.clone();
    let mut best_scale = 0.0_f32;
    let mut best_progress_model = None::<NpaModel>;
    let mut best_progress_report = None::<TrainingRunReport>;
    let mut best_progress_selection = no_op_selection.clone();
    let mut best_progress_scale = 0.0_f32;

    for scale in scales {
        let scaled_learning_rate = cfg.sgd.learning_rate * scale;
        if !scaled_learning_rate.is_finite() {
            continue;
        }
        let mut candidate_cfg = cfg.clone();
        candidate_cfg.direct_line_search = false;
        candidate_cfg.sgd.learning_rate = scaled_learning_rate;
        let mut candidate = base_model.clone();
        let report = if cfg.direct_selection_seed_training {
            render_direct_rollout_multiseed_training_step(
                &mut candidate,
                grid,
                target,
                &candidate_cfg,
                round,
                trace,
                trajectory,
                gradient,
            )?
        } else {
            render_direct_rollout_training_step(
                &mut candidate,
                grid,
                target,
                trace,
                trajectory,
                gradient,
                &candidate_cfg,
                render_training_round_seed(cfg, round),
            )?
        };
        let selection = render_selection_metrics(
            &candidate,
            grid,
            target,
            cfg,
            render_cfg,
            Some(selection_baseline),
        )?;
        let candidate_beats = render_selection_candidate_metrics_beats(&selection, &best_selection);
        if candidate_beats
            || (best_scale == 0.0
                && render_selection_morphology_recovery_beats(&selection, &best_selection))
        {
            best_model = candidate;
            best_report = report;
            best_scale = scale;
            best_selection = selection;
        } else if render_selection_training_progress_beats(&selection, &no_op_selection)
            && (best_progress_model.is_none()
                || selection.render_loss < best_progress_selection.render_loss
                || selection.score < best_progress_selection.score)
        {
            best_progress_model = Some(candidate);
            best_progress_report = Some(report);
            best_progress_scale = scale;
            best_progress_selection = selection;
        }
    }

    if best_scale == 0.0
        && let (Some(progress_model), Some(progress_report)) =
            (best_progress_model, best_progress_report)
    {
        *model = progress_model;
        return Ok((progress_report, best_progress_scale));
    }

    *model = best_model;
    Ok((best_report, best_scale))
}

pub(crate) fn sanitized_direct_line_search_scales(cfg: &RenderProxyTrainingConfig) -> Vec<f32> {
    if !cfg.direct_line_search {
        return Vec::new();
    }
    let mut scales = Vec::with_capacity(cfg.direct_line_search_scales.len());
    for &scale in &cfg.direct_line_search_scales {
        if scale.is_finite() && scale > 0.0 && !scales.contains(&scale) {
            scales.push(scale);
        }
    }
    if scales.is_empty() {
        scales.push(1.0);
    }
    scales
}

pub(crate) fn render_direct_rollout_noop_report(
    initial_loss: f32,
    gradient: &RenderProxyGradientRows,
) -> TrainingRunReport {
    let rows = gradient
        .gradients
        .len()
        .min(gradient.row_indices.len())
        .min(gradient.opacity_gradients.len())
        .min(gradient.scale_gradients.len())
        .min(gradient.color_gradients.len());
    TrainingRunReport {
        steps: 0,
        rows,
        initial_loss,
        final_loss: initial_loss,
        best_loss: initial_loss,
        history: vec![TrainingHistoryEntry {
            step: 0,
            loss: initial_loss,
            grad_norm: 0.0,
            grad_scale: 0.0,
        }],
    }
}

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
