use super::*;

struct DirectLineSearchCandidate {
    model: NpaModel,
    report: TrainingRunReport,
    selection: RenderSelectionMetrics,
    candidate_report: DirectLineSearchCandidateReport,
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
) -> Result<
    (TrainingRunReport, f32, Vec<DirectLineSearchCandidateReport>),
    Box<dyn std::error::Error>,
> {
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
        return Ok((report, 1.0, Vec::new()));
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
    let mut candidate_reports = Vec::with_capacity(scales.len());

    for scale in scales {
        let Some(candidate) = evaluate_direct_line_search_candidate(
            &base_model,
            grid,
            target,
            cfg,
            round,
            trace,
            trajectory,
            gradient,
            render_cfg,
            selection_baseline,
            scale,
        )?
        else {
            continue;
        };
        update_direct_line_search_state(
            candidate,
            &mut best_model,
            &mut best_report,
            &mut best_selection,
            &mut best_scale,
            &mut best_progress_model,
            &mut best_progress_report,
            &mut best_progress_selection,
            &mut best_progress_scale,
            &mut candidate_reports,
            &no_op_selection,
        );
    }

    let refinement_scales =
        adaptive_direct_line_search_refinement_scales(&candidate_reports, cfg.particles);
    candidate_reports.reserve(refinement_scales.len());
    for scale in refinement_scales {
        let Some(candidate) = evaluate_direct_line_search_candidate(
            &base_model,
            grid,
            target,
            cfg,
            round,
            trace,
            trajectory,
            gradient,
            render_cfg,
            selection_baseline,
            scale,
        )?
        else {
            continue;
        };
        update_direct_line_search_state(
            candidate,
            &mut best_model,
            &mut best_report,
            &mut best_selection,
            &mut best_scale,
            &mut best_progress_model,
            &mut best_progress_report,
            &mut best_progress_selection,
            &mut best_progress_scale,
            &mut candidate_reports,
            &no_op_selection,
        );
    }

    if best_scale != 0.0
        && best_progress_model.is_some()
        && best_progress_report.is_some()
        && render_selection_training_progress_beats(&best_progress_selection, &best_selection)
        && (best_progress_selection.score < best_selection.score
            || best_progress_selection.render_loss < best_selection.render_loss)
    {
        let progress_model = best_progress_model.take().expect("checked above");
        let progress_report = best_progress_report.take().expect("checked above");
        *model = progress_model;
        mark_selected_line_search_candidate(&mut candidate_reports, best_progress_scale, false);
        return Ok((progress_report, best_progress_scale, candidate_reports));
    }

    if best_scale == 0.0
        && let (Some(progress_model), Some(progress_report)) =
            (best_progress_model, best_progress_report)
    {
        *model = progress_model;
        mark_selected_line_search_candidate(&mut candidate_reports, best_progress_scale, false);
        return Ok((progress_report, best_progress_scale, candidate_reports));
    }

    *model = best_model;
    mark_selected_line_search_candidate(&mut candidate_reports, best_scale, true);
    Ok((best_report, best_scale, candidate_reports))
}

#[allow(clippy::too_many_arguments)]
fn evaluate_direct_line_search_candidate(
    base_model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: &RenderProxyTrainingConfig,
    round: usize,
    trace: &crate::RolloutTrace,
    trajectory: &[RenderTrajectorySnapshot],
    gradient: &RenderProxyGradientRows,
    render_cfg: RenderLossConfig,
    selection_baseline: &[RenderSelectionBaselineCase],
    scale: f32,
) -> Result<Option<DirectLineSearchCandidate>, Box<dyn std::error::Error>> {
    let scaled_learning_rate = cfg.sgd.learning_rate * scale;
    if !scaled_learning_rate.is_finite() {
        return Ok(None);
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
    let candidate_report = direct_line_search_candidate_report(scale, &selection, &report);
    Ok(Some(DirectLineSearchCandidate {
        model: candidate,
        report,
        selection,
        candidate_report,
    }))
}

#[allow(clippy::too_many_arguments)]
fn update_direct_line_search_state(
    mut candidate: DirectLineSearchCandidate,
    best_model: &mut NpaModel,
    best_report: &mut TrainingRunReport,
    best_selection: &mut RenderSelectionMetrics,
    best_scale: &mut f32,
    best_progress_model: &mut Option<NpaModel>,
    best_progress_report: &mut Option<TrainingRunReport>,
    best_progress_selection: &mut RenderSelectionMetrics,
    best_progress_scale: &mut f32,
    candidate_reports: &mut Vec<DirectLineSearchCandidateReport>,
    no_op_selection: &RenderSelectionMetrics,
) {
    let candidate_beats =
        render_selection_candidate_metrics_beats(&candidate.selection, best_selection);
    let morphology_recovery = *best_scale == 0.0
        && render_selection_morphology_recovery_beats(&candidate.selection, best_selection);
    let progress_candidate =
        render_selection_training_progress_beats(&candidate.selection, no_op_selection);
    candidate.candidate_report.checkpoint_candidate = candidate_beats || morphology_recovery;
    candidate.candidate_report.progress_candidate = progress_candidate;
    if candidate_beats || morphology_recovery {
        *best_model = candidate.model;
        *best_report = candidate.report;
        *best_scale = candidate.candidate_report.scale;
        *best_selection = candidate.selection;
    } else if progress_candidate
        && (best_progress_model.is_none()
            || render_selection_progress_candidate_preferred(
                &candidate.selection,
                best_progress_selection,
                no_op_selection,
            ))
    {
        *best_progress_model = Some(candidate.model);
        *best_progress_report = Some(candidate.report);
        *best_progress_scale = candidate.candidate_report.scale;
        *best_progress_selection = candidate.selection;
    }
    candidate_reports.push(candidate.candidate_report);
}

fn direct_line_search_candidate_report(
    scale: f32,
    selection: &RenderSelectionMetrics,
    report: &TrainingRunReport,
) -> DirectLineSearchCandidateReport {
    let last_history = report.history.last();
    DirectLineSearchCandidateReport {
        inner_step: 0,
        scale,
        checkpoint_candidate: false,
        progress_candidate: false,
        selected_checkpoint: false,
        selected_progress: false,
        render_loss: selection.render_loss,
        score: selection.score,
        density_psnr_db: selection.density_psnr_db,
        morphology_non_regressed: selection.morphology_non_regressed,
        active_surface_max: selection.active_surface_max,
        target_coverage_fraction: selection.target_coverage_fraction,
        material_visible_target_mean_distance: selection.material_visible_target_mean_distance,
        material_visible_target_max_distance: selection.material_visible_target_max_distance,
        material_visible_target_coverage_fraction: selection
            .material_visible_target_coverage_fraction,
        strict_surface_active_count: selection.strict_surface_active_count,
        strict_surface_materialized_fraction: selection.strict_surface_materialized_fraction,
        strict_surface_material_mean_opacity: selection.strict_surface_material_mean_opacity,
        strict_surface_material_visible_margin: selection.strict_surface_material_visible_margin,
        strict_surface_material_max_visible_margin: selection
            .strict_surface_material_max_visible_margin,
        material_visible_inactive_fraction: selection.material_visible_inactive_fraction,
        material_visible_max_inactive_opacity: selection.material_visible_max_inactive_opacity,
        material_active_mean_opacity: selection.material_active_mean_opacity,
        material_visible_count: selection.material_visible_count,
        active_color_state_mean_abs: selection.active_color_state_mean_abs,
        active_color_state_max_abs: selection.active_color_state_max_abs,
        active_color_state_stddev_mean: selection.active_color_state_stddev_mean,
        surface_covered_bin_fraction: selection.surface_covered_bin_fraction,
        surface_mean_bin_covered_fraction: selection.surface_mean_bin_covered_fraction,
        material_visible_surface_covered_bin_fraction: selection
            .material_visible_surface_covered_bin_fraction,
        material_visible_surface_mean_bin_covered_fraction: selection
            .material_visible_surface_mean_bin_covered_fraction,
        surface_normal_covered_bin_fraction: selection.surface_normal_covered_bin_fraction,
        surface_normal_mean_bin_covered_fraction: selection
            .surface_normal_mean_bin_covered_fraction,
        material_visible_surface_normal_covered_bin_fraction: selection
            .material_visible_surface_normal_covered_bin_fraction,
        material_visible_surface_normal_mean_bin_covered_fraction: selection
            .material_visible_surface_normal_mean_bin_covered_fraction,
        material_visible_surface_tail_p99_distance: selection
            .material_visible_surface_tail_p99_distance,
        material_visible_surface_tail_over_threshold_fraction: selection
            .material_visible_surface_tail_over_threshold_fraction,
        min_active_extent_bbox_ratio: selection.min_active_extent_bbox_ratio,
        min_active_extent_min_axis_ratio: selection.min_active_extent_min_axis_ratio,
        min_final_active_count: selection.min_final_active_count,
        min_newly_activated_fraction: selection.min_newly_activated_fraction,
        min_front_local_newly_activated_fraction: selection
            .min_front_local_newly_activated_fraction,
        max_front_liveness_margin: selection.max_front_liveness_margin,
        min_front_liveness_candidate_count: selection.min_front_liveness_candidate_count,
        max_extent_front_liveness_margin: selection.max_extent_front_liveness_margin,
        min_extent_front_liveness_candidate_count: selection
            .min_extent_front_liveness_candidate_count,
        max_temporal_front_liveness_margin: selection.max_temporal_front_liveness_margin,
        min_temporal_front_liveness_candidate_count: selection
            .min_temporal_front_liveness_candidate_count,
        max_temporal_extent_front_liveness_margin: selection
            .max_temporal_extent_front_liveness_margin,
        min_temporal_extent_front_liveness_candidate_count: selection
            .min_temporal_extent_front_liveness_candidate_count,
        max_temporal_activation_schedule_error: selection.max_temporal_activation_schedule_error,
        all_temporal_activation_progressive: selection.all_temporal_activation_progressive,
        all_temporal_geometry_progressive: selection.all_temporal_geometry_progressive,
        train_final_loss: report.final_loss,
        train_grad_norm: last_history.map(|entry| entry.grad_norm).unwrap_or(0.0),
        train_grad_scale: last_history.map(|entry| entry.grad_scale).unwrap_or(0.0),
        failure_reasons: selection.worst_failure_reasons.clone(),
    }
}

pub(crate) fn adaptive_direct_line_search_refinement_scales(
    reports: &[DirectLineSearchCandidateReport],
    particle_count: usize,
) -> Vec<f32> {
    const REFINEMENT_FRACTIONS: [f32; 3] = [1.0 / 3.0, 0.5, 2.0 / 3.0];
    const STRICT_NEWLY_ACTIVATED_FRACTION: f32 = 0.50;
    let target_active =
        ((particle_count as f32) * temporal_activation_target_fraction(1.0)).ceil() as usize;
    let allowed_active =
        ((particle_count as f32) * temporal_activation_allowed_fraction(1.0)).ceil() as usize;
    if reports.len() < 2 || target_active == 0 || allowed_active == 0 {
        return Vec::new();
    }

    let mut sorted = reports
        .iter()
        .filter(|report| report.scale.is_finite() && report.scale > 0.0 && report.score.is_finite())
        .collect::<Vec<_>>();
    sorted.sort_by(|lhs, rhs| {
        lhs.scale
            .partial_cmp(&rhs.scale)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut scales = Vec::new();
    for pair in sorted.windows(2) {
        let lower = pair[0];
        let upper = pair[1];
        if lower.scale >= upper.scale {
            continue;
        }
        let lower_under_active = lower.min_final_active_count < target_active
            || lower.min_newly_activated_fraction < STRICT_NEWLY_ACTIVATED_FRACTION;
        let upper_over_active = upper.min_final_active_count > allowed_active
            || upper.max_temporal_activation_schedule_error
                > lower.max_temporal_activation_schedule_error
                    + TEMPORAL_ACTIVATION_SELECTION_REGRESSION_SLACK;
        if !lower_under_active || !upper_over_active {
            continue;
        }
        let log_lower = lower.scale.ln();
        let log_upper = upper.scale.ln();
        for fraction in REFINEMENT_FRACTIONS {
            let scale = (log_lower + (log_upper - log_lower) * fraction).exp();
            if !scales.iter().any(|existing: &f32| {
                ((*existing - scale).abs() / existing.abs().max(scale.abs()).max(1.0e-6)) < 1.0e-5
            }) && !sorted.iter().any(|existing| {
                ((existing.scale - scale).abs() / existing.scale.abs().max(scale.abs()).max(1.0e-6))
                    < 1.0e-5
            }) {
                scales.push(scale);
            }
        }
    }
    scales
}

fn mark_selected_line_search_candidate(
    reports: &mut [DirectLineSearchCandidateReport],
    selected_scale: f32,
    checkpoint: bool,
) {
    if selected_scale <= 0.0 || !selected_scale.is_finite() {
        return;
    }
    if let Some(report) = reports
        .iter_mut()
        .find(|report| (report.scale - selected_scale).abs() <= f32::EPSILON)
    {
        if checkpoint {
            report.selected_checkpoint = true;
        } else {
            report.selected_progress = true;
        }
    }
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
