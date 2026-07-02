use super::*;

mod adapter;
mod line_search;
mod trajectory;

pub(crate) use adapter::*;
pub(crate) use line_search::*;
pub(crate) use trajectory::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn render_direct_rollout_training_steps(
    model: &mut NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: &RenderProxyTrainingConfig,
    round: usize,
    render_cfg: RenderLossConfig,
    selection_baseline: &[RenderSelectionBaselineCase],
) -> Result<
    (TrainingRunReport, f32, Vec<DirectLineSearchCandidateReport>),
    Box<dyn std::error::Error>,
> {
    let steps = cfg.supervised_steps_per_round.max(1);
    let mut reports = Vec::with_capacity(steps);
    let mut line_search_candidates = Vec::new();
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
    for inner_step in 0..steps {
        let (trace, trajectory) = render_training_trajectory(model, grid, cfg, round)?;
        let gradient = render_position_gradient(&trace, target, render_cfg, cfg)?;
        let (report, step_scale, mut candidates) = if cfg.direct_line_search {
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
            (report, 1.0, Vec::new())
        };
        for candidate in &mut candidates {
            candidate.inner_step = inner_step;
        }
        line_search_candidates.extend(candidates);
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
    Ok((
        report,
        step_scale_sum / steps as f32,
        line_search_candidates,
    ))
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
