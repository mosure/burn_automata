use super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn render_direct_rollout_adapter_training_steps(
    base_model: &NpaModel,
    adapter: &mut NpaLowRankAdapter,
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
    let mut step_scale_sum = 0.0_f32;
    let mut best_inner_model = model.clone();
    let mut best_inner_adapter = adapter.clone();
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
    for _inner_step in 0..steps {
        let (trace, trajectory) = render_training_trajectory(model, grid, cfg, round)?;
        let gradient = render_position_gradient(&trace, target, render_cfg, cfg)?;
        let (report, step_scale) = if cfg.direct_line_search {
            render_direct_rollout_adapter_training_step_with_line_search(
                base_model,
                adapter,
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
                render_direct_rollout_adapter_multiseed_training_step(
                    base_model,
                    adapter,
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
                render_direct_rollout_adapter_training_step(
                    base_model,
                    adapter,
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
                best_inner_adapter = adapter.clone();
                best_inner_selection = Some(selection);
                selected_inner_checkpoint = true;
            } else if best_inner_selection
                .as_ref()
                .is_some_and(|best| render_selection_training_progress_beats(&selection, best))
            {
                // Keep bounded adapter progress for subsequent inner steps;
                // strict checkpointing is restored below when available.
            } else {
                *model = best_inner_model.clone();
                *adapter = best_inner_adapter.clone();
            }
        }
    }
    let mut report = combine_direct_rollout_training_reports(&reports);
    if cfg.direct_line_search && selected_inner_checkpoint {
        *model = best_inner_model;
        *adapter = best_inner_adapter;
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
    Ok((report, step_scale_sum / steps as f32, Vec::new()))
}

#[allow(clippy::too_many_arguments)]
fn render_direct_rollout_adapter_training_step_with_line_search(
    base_model: &NpaModel,
    adapter: &mut NpaLowRankAdapter,
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
            render_direct_rollout_adapter_multiseed_training_step(
                base_model, adapter, model, grid, target, cfg, round, trace, trajectory, gradient,
            )?
        } else {
            render_direct_rollout_adapter_training_step(
                base_model,
                adapter,
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

    let base_adapter = adapter.clone();
    let base_materialized = model.clone();
    let initial_loss = mesh_multiview_render_loss_from_trace(trace, target, render_cfg)?.total_loss;
    let no_op_selection = render_selection_metrics(
        model,
        grid,
        target,
        cfg,
        render_cfg,
        Some(selection_baseline),
    )?;
    let mut best_adapter = base_adapter.clone();
    let mut best_model = base_materialized.clone();
    let mut best_report = render_direct_rollout_noop_report(initial_loss, gradient);
    let mut best_selection = no_op_selection.clone();
    let mut best_scale = 0.0_f32;
    let mut selected = false;
    let mut best_progress_adapter = None::<NpaLowRankAdapter>;
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
        let mut candidate_adapter = base_adapter.clone();
        let mut candidate_model = candidate_adapter.apply_to_model(base_model)?;
        let report = if cfg.direct_selection_seed_training {
            render_direct_rollout_adapter_multiseed_training_step(
                base_model,
                &mut candidate_adapter,
                &mut candidate_model,
                grid,
                target,
                &candidate_cfg,
                round,
                trace,
                trajectory,
                gradient,
            )?
        } else {
            render_direct_rollout_adapter_training_step(
                base_model,
                &mut candidate_adapter,
                &mut candidate_model,
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
            &candidate_model,
            grid,
            target,
            cfg,
            render_cfg,
            Some(selection_baseline),
        )?;
        let checkpoint_candidate =
            render_selection_candidate_metrics_beats(&selection, &best_selection)
                || (!selected
                    && render_selection_morphology_recovery_beats(&selection, &best_selection));
        if checkpoint_candidate {
            best_adapter = candidate_adapter;
            best_model = candidate_model;
            best_report = report;
            best_selection = selection;
            best_scale = scale;
            selected = true;
        } else if render_selection_training_progress_beats(&selection, &no_op_selection)
            && (best_progress_model.is_none()
                || render_selection_progress_candidate_preferred(
                    &selection,
                    &best_progress_selection,
                    &no_op_selection,
                ))
        {
            best_progress_adapter = Some(candidate_adapter);
            best_progress_model = Some(candidate_model);
            best_progress_report = Some(report);
            best_progress_selection = selection;
            best_progress_scale = scale;
        }
    }

    if !selected
        && let (Some(progress_adapter), Some(progress_model), Some(progress_report)) = (
            best_progress_adapter,
            best_progress_model,
            best_progress_report,
        )
    {
        *adapter = progress_adapter;
        *model = progress_model;
        return Ok((progress_report, best_progress_scale));
    }

    *adapter = best_adapter;
    *model = best_model;
    Ok((best_report, best_scale))
}
