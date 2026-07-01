use super::*;

#[derive(Clone)]
pub(crate) struct RenderSelectionMetrics {
    pub(crate) render_loss: f32,
    pub(crate) score: f32,
    pub(crate) density_psnr_db: f32,
    pub(crate) active_surface_max: f32,
    pub(crate) target_coverage_fraction: f32,
    pub(crate) material_visible_target_mean_distance: f32,
    pub(crate) material_visible_target_max_distance: f32,
    pub(crate) material_visible_target_coverage_fraction: f32,
    pub(crate) material_visible_inactive_fraction: f32,
    pub(crate) material_visible_max_inactive_opacity: f32,
    pub(crate) material_active_mean_opacity: f32,
    pub(crate) material_visible_count: usize,
    pub(crate) surface_covered_bin_fraction: f32,
    pub(crate) surface_mean_bin_covered_fraction: f32,
    pub(crate) material_visible_surface_covered_bin_fraction: f32,
    pub(crate) material_visible_surface_mean_bin_covered_fraction: f32,
    pub(crate) surface_normal_covered_bin_fraction: f32,
    pub(crate) surface_normal_mean_bin_covered_fraction: f32,
    pub(crate) material_visible_surface_normal_covered_bin_fraction: f32,
    pub(crate) material_visible_surface_normal_mean_bin_covered_fraction: f32,
    pub(crate) material_visible_surface_tail_p99_distance: f32,
    pub(crate) material_visible_surface_tail_over_threshold_fraction: f32,
    pub(crate) min_active_extent_bbox_ratio: f32,
    pub(crate) min_active_extent_min_axis_ratio: f32,
    pub(crate) min_final_active_count: usize,
    pub(crate) min_newly_activated_fraction: f32,
    pub(crate) min_front_local_newly_activated_fraction: f32,
    pub(crate) max_front_liveness_margin: f32,
    pub(crate) min_front_liveness_candidate_count: usize,
    pub(crate) max_extent_front_liveness_margin: f32,
    pub(crate) min_extent_front_liveness_candidate_count: usize,
    pub(crate) max_temporal_front_liveness_margin: f32,
    pub(crate) min_temporal_front_liveness_candidate_count: usize,
    pub(crate) max_temporal_extent_front_liveness_margin: f32,
    pub(crate) min_temporal_extent_front_liveness_candidate_count: usize,
    pub(crate) max_temporal_activation_schedule_error: f32,
    pub(crate) all_temporal_activation_progressive: bool,
    pub(crate) all_temporal_geometry_progressive: bool,
    pub(crate) morphology_non_regressed: bool,
    pub(crate) worst_seed: u64,
    pub(crate) worst_failure_reasons: Vec<&'static str>,
    pub(crate) base_report: MultiViewRenderLossReport,
}

#[derive(Clone, Copy)]
pub(crate) struct RenderSelectionBaselineCase {
    pub(crate) seed: u64,
    pub(crate) active_surface_max: f32,
    pub(crate) target_coverage_fraction: f32,
    pub(crate) material_visible_target_mean_distance: f32,
    pub(crate) material_visible_target_max_distance: f32,
    pub(crate) material_visible_target_coverage_fraction: f32,
    pub(crate) material_visible_inactive_fraction: f32,
    pub(crate) material_visible_max_inactive_opacity: f32,
    pub(crate) surface_covered_bin_fraction: f32,
    pub(crate) surface_mean_bin_covered_fraction: f32,
    pub(crate) material_visible_surface_covered_bin_fraction: f32,
    pub(crate) material_visible_surface_mean_bin_covered_fraction: f32,
    pub(crate) surface_normal_covered_bin_fraction: f32,
    pub(crate) surface_normal_mean_bin_covered_fraction: f32,
    pub(crate) material_visible_surface_normal_covered_bin_fraction: f32,
    pub(crate) material_visible_surface_normal_mean_bin_covered_fraction: f32,
    pub(crate) material_visible_surface_tail_p99_distance: f32,
    pub(crate) material_visible_surface_tail_over_threshold_fraction: f32,
    pub(crate) active_extent_bbox_ratio: f32,
    pub(crate) active_extent_min_axis_ratio: f32,
    pub(crate) final_active_count: usize,
    pub(crate) newly_activated_fraction: f32,
    pub(crate) front_local_newly_activated_fraction: f32,
    pub(crate) front_liveness: LocalFrontLivenessProgress,
    pub(crate) extent_front_liveness: LocalFrontLivenessProgress,
    pub(crate) temporal_front_liveness: LocalFrontLivenessProgress,
    pub(crate) temporal_extent_front_liveness: LocalFrontLivenessProgress,
    pub(crate) temporal_activation_schedule_error: f32,
    pub(crate) temporal_activation_progressive: bool,
    pub(crate) temporal_geometry_progressive: bool,
}

pub(crate) fn render_selection_metrics(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: &RenderProxyTrainingConfig,
    render_cfg: RenderLossConfig,
    baseline: Option<&[RenderSelectionBaselineCase]>,
) -> Result<RenderSelectionMetrics, Box<dyn std::error::Error>> {
    let selection_seeds = render_proxy_selection_seeds(cfg);
    let base_seed = selection_seeds[0];
    let base_case = render_selection_case_metrics(model, grid, target, cfg, render_cfg, base_seed)?;
    let mut render_loss = 0.0_f32;
    let mut morphology_non_regressed = true;
    let mut score = f32::NEG_INFINITY;
    let mut worst_seed = base_seed;
    let mut worst_failure_reasons = Vec::new();
    let mut density_psnr_db = 0.0_f32;
    let mut active_surface_max = f32::NEG_INFINITY;
    let mut target_coverage_fraction = f32::INFINITY;
    let mut material_visible_target_mean_distance = f32::NEG_INFINITY;
    let mut material_visible_target_max_distance = f32::NEG_INFINITY;
    let mut material_visible_target_coverage_fraction = f32::INFINITY;
    let mut material_visible_inactive_fraction = f32::NEG_INFINITY;
    let mut material_visible_max_inactive_opacity = f32::NEG_INFINITY;
    let mut material_active_mean_opacity = f32::INFINITY;
    let mut material_visible_count = usize::MAX;
    let mut surface_covered_bin_fraction = f32::INFINITY;
    let mut surface_mean_bin_covered_fraction = f32::INFINITY;
    let mut material_visible_surface_covered_bin_fraction = f32::INFINITY;
    let mut material_visible_surface_mean_bin_covered_fraction = f32::INFINITY;
    let mut surface_normal_covered_bin_fraction = f32::INFINITY;
    let mut surface_normal_mean_bin_covered_fraction = f32::INFINITY;
    let mut material_visible_surface_normal_covered_bin_fraction = f32::INFINITY;
    let mut material_visible_surface_normal_mean_bin_covered_fraction = f32::INFINITY;
    let mut material_visible_surface_tail_p99_distance = f32::NEG_INFINITY;
    let mut material_visible_surface_tail_over_threshold_fraction = f32::NEG_INFINITY;
    let mut min_active_extent_bbox_ratio = f32::INFINITY;
    let mut min_active_extent_min_axis_ratio = f32::INFINITY;
    let mut min_final_active_count = usize::MAX;
    let mut min_newly_activated_fraction = f32::INFINITY;
    let mut min_front_local_newly_activated_fraction = f32::INFINITY;
    let mut max_front_liveness_margin = f32::NEG_INFINITY;
    let mut min_front_liveness_candidate_count = usize::MAX;
    let mut max_extent_front_liveness_margin = f32::NEG_INFINITY;
    let mut min_extent_front_liveness_candidate_count = usize::MAX;
    let mut max_temporal_front_liveness_margin = f32::NEG_INFINITY;
    let mut min_temporal_front_liveness_candidate_count = usize::MAX;
    let mut max_temporal_extent_front_liveness_margin = f32::NEG_INFINITY;
    let mut min_temporal_extent_front_liveness_candidate_count = usize::MAX;
    let mut max_temporal_activation_schedule_error = f32::NEG_INFINITY;
    let mut all_temporal_activation_progressive = true;
    let mut all_temporal_geometry_progressive = true;
    for seed in &selection_seeds {
        let owned_case;
        let selection_case = if *seed == base_seed {
            &base_case
        } else {
            owned_case =
                render_selection_case_metrics(model, grid, target, cfg, render_cfg, *seed)?;
            &owned_case
        };
        render_loss += selection_case.render_loss.total_loss;
        let selection_score =
            render_selection_case_score_with_baseline(*seed, selection_case, baseline);
        if !selection_score.morphology_non_regressed {
            morphology_non_regressed = false;
        }
        if selection_score.score > score {
            worst_seed = *seed;
            worst_failure_reasons = selection_case.failure_reasons.clone();
            if !selection_score.morphology_non_regressed {
                worst_failure_reasons.push("selection_morphology_non_regressed");
            }
        }
        score = score.max(selection_score.score);
        density_psnr_db += selection_case.render_loss.density_psnr_db;
        active_surface_max = active_surface_max.max(selection_case.active_surface.max_distance);
        target_coverage_fraction =
            target_coverage_fraction.min(selection_case.target_coverage.covered_fraction);
        material_visible_target_mean_distance = material_visible_target_mean_distance.max(
            selection_case
                .material_visible_target_coverage
                .mean_distance,
        );
        material_visible_target_max_distance = material_visible_target_max_distance
            .max(selection_case.material_visible_target_coverage.max_distance);
        material_visible_target_coverage_fraction = material_visible_target_coverage_fraction.min(
            selection_case
                .material_visible_target_coverage
                .covered_fraction,
        );
        material_visible_inactive_fraction = material_visible_inactive_fraction.max(
            selection_case
                .material_liveness
                .inactive_material_visible_fraction,
        );
        material_visible_max_inactive_opacity = material_visible_max_inactive_opacity.max(
            selection_case
                .material_liveness
                .max_inactive_material_opacity,
        );
        material_active_mean_opacity =
            material_active_mean_opacity.min(selection_case.material_opacity.active_mean);
        material_visible_count =
            material_visible_count.min(selection_case.material_liveness.material_visible_count);
        surface_covered_bin_fraction = surface_covered_bin_fraction
            .min(selection_case.surface_coverage_profile.covered_bin_fraction);
        surface_mean_bin_covered_fraction = surface_mean_bin_covered_fraction.min(
            selection_case
                .surface_coverage_profile
                .mean_bin_covered_fraction,
        );
        material_visible_surface_covered_bin_fraction =
            material_visible_surface_covered_bin_fraction.min(
                selection_case
                    .material_visible_surface_coverage_profile
                    .covered_bin_fraction,
            );
        material_visible_surface_mean_bin_covered_fraction =
            material_visible_surface_mean_bin_covered_fraction.min(
                selection_case
                    .material_visible_surface_coverage_profile
                    .mean_bin_covered_fraction,
            );
        surface_normal_covered_bin_fraction = surface_normal_covered_bin_fraction.min(
            selection_case
                .surface_normal_coverage
                .covered_target_bin_fraction,
        );
        surface_normal_mean_bin_covered_fraction = surface_normal_mean_bin_covered_fraction.min(
            selection_case
                .surface_normal_coverage
                .mean_bin_covered_fraction,
        );
        material_visible_surface_normal_covered_bin_fraction =
            material_visible_surface_normal_covered_bin_fraction.min(
                selection_case
                    .material_visible_surface_normal_coverage
                    .covered_target_bin_fraction,
            );
        material_visible_surface_normal_mean_bin_covered_fraction =
            material_visible_surface_normal_mean_bin_covered_fraction.min(
                selection_case
                    .material_visible_surface_normal_coverage
                    .mean_bin_covered_fraction,
            );
        material_visible_surface_tail_p99_distance = material_visible_surface_tail_p99_distance
            .max(selection_case.material_visible_surface_tail.p99_distance);
        material_visible_surface_tail_over_threshold_fraction =
            material_visible_surface_tail_over_threshold_fraction.max(
                selection_case
                    .material_visible_surface_tail
                    .over_threshold_fraction,
            );
        min_active_extent_bbox_ratio =
            min_active_extent_bbox_ratio.min(selection_case.extent.bbox_diagonal_ratio);
        min_active_extent_min_axis_ratio =
            min_active_extent_min_axis_ratio.min(selection_case.extent.min_axis_extent_ratio);
        min_final_active_count = min_final_active_count.min(selection_case.final_active_count);
        min_newly_activated_fraction =
            min_newly_activated_fraction.min(selection_case.newly_activated_fraction);
        min_front_local_newly_activated_fraction = min_front_local_newly_activated_fraction
            .min(selection_case.front_local_newly_activated_fraction);
        max_front_liveness_margin =
            max_front_liveness_margin.max(selection_case.front_liveness.weighted_activation_margin);
        min_front_liveness_candidate_count =
            min_front_liveness_candidate_count.min(selection_case.front_liveness.candidate_count);
        max_extent_front_liveness_margin = max_extent_front_liveness_margin.max(
            selection_case
                .extent_front_liveness
                .weighted_activation_margin,
        );
        min_extent_front_liveness_candidate_count = min_extent_front_liveness_candidate_count
            .min(selection_case.extent_front_liveness.candidate_count);
        max_temporal_front_liveness_margin = max_temporal_front_liveness_margin.max(
            selection_case
                .temporal_front_liveness
                .weighted_activation_margin,
        );
        min_temporal_front_liveness_candidate_count = min_temporal_front_liveness_candidate_count
            .min(selection_case.temporal_front_liveness.candidate_count);
        max_temporal_extent_front_liveness_margin = max_temporal_extent_front_liveness_margin.max(
            selection_case
                .temporal_extent_front_liveness
                .weighted_activation_margin,
        );
        min_temporal_extent_front_liveness_candidate_count =
            min_temporal_extent_front_liveness_candidate_count.min(
                selection_case
                    .temporal_extent_front_liveness
                    .candidate_count,
            );
        max_temporal_activation_schedule_error = max_temporal_activation_schedule_error
            .max(selection_case.temporal_activation_schedule_error);
        all_temporal_activation_progressive &= selection_case.temporal_activation_progressive;
        all_temporal_geometry_progressive &= selection_case.temporal_geometry_progressive;
    }
    let count = selection_seeds.len().max(1) as f32;

    Ok(RenderSelectionMetrics {
        render_loss: finite_report_metric(render_loss / count, RENDER_SELECTION_BAD_SCORE),
        score: finite_report_metric(score, RENDER_SELECTION_BAD_SCORE),
        density_psnr_db: finite_report_metric(density_psnr_db / count, -RENDER_SELECTION_BAD_SCORE),
        active_surface_max: finite_report_metric(active_surface_max, RENDER_SELECTION_BAD_SCORE),
        target_coverage_fraction: finite_report_metric(target_coverage_fraction, 0.0),
        material_visible_target_mean_distance: finite_report_metric(
            material_visible_target_mean_distance,
            RENDER_SELECTION_BAD_SCORE,
        ),
        material_visible_target_max_distance: finite_report_metric(
            material_visible_target_max_distance,
            RENDER_SELECTION_BAD_SCORE,
        ),
        material_visible_target_coverage_fraction: finite_report_metric(
            material_visible_target_coverage_fraction,
            0.0,
        ),
        material_visible_inactive_fraction: finite_report_metric(
            material_visible_inactive_fraction,
            1.0,
        ),
        material_visible_max_inactive_opacity,
        material_active_mean_opacity: finite_report_metric(material_active_mean_opacity, 0.0),
        material_visible_count: if material_visible_count == usize::MAX {
            0
        } else {
            material_visible_count
        },
        surface_covered_bin_fraction: finite_report_metric(surface_covered_bin_fraction, 0.0),
        surface_mean_bin_covered_fraction: finite_report_metric(
            surface_mean_bin_covered_fraction,
            0.0,
        ),
        material_visible_surface_covered_bin_fraction: finite_report_metric(
            material_visible_surface_covered_bin_fraction,
            0.0,
        ),
        material_visible_surface_mean_bin_covered_fraction: finite_report_metric(
            material_visible_surface_mean_bin_covered_fraction,
            0.0,
        ),
        surface_normal_covered_bin_fraction: finite_report_metric(
            surface_normal_covered_bin_fraction,
            0.0,
        ),
        surface_normal_mean_bin_covered_fraction: finite_report_metric(
            surface_normal_mean_bin_covered_fraction,
            0.0,
        ),
        material_visible_surface_normal_covered_bin_fraction: finite_report_metric(
            material_visible_surface_normal_covered_bin_fraction,
            0.0,
        ),
        material_visible_surface_normal_mean_bin_covered_fraction: finite_report_metric(
            material_visible_surface_normal_mean_bin_covered_fraction,
            0.0,
        ),
        material_visible_surface_tail_p99_distance: finite_report_metric(
            material_visible_surface_tail_p99_distance,
            RENDER_SELECTION_BAD_SCORE,
        ),
        material_visible_surface_tail_over_threshold_fraction: finite_report_metric(
            material_visible_surface_tail_over_threshold_fraction,
            1.0,
        ),
        min_active_extent_bbox_ratio: finite_report_metric(min_active_extent_bbox_ratio, 0.0),
        min_active_extent_min_axis_ratio: finite_report_metric(
            min_active_extent_min_axis_ratio,
            0.0,
        ),
        min_final_active_count: if min_final_active_count == usize::MAX {
            0
        } else {
            min_final_active_count
        },
        min_newly_activated_fraction: finite_report_metric(min_newly_activated_fraction, 0.0),
        min_front_local_newly_activated_fraction: finite_report_metric(
            min_front_local_newly_activated_fraction,
            0.0,
        ),
        max_front_liveness_margin: finite_report_metric(max_front_liveness_margin, 0.0),
        min_front_liveness_candidate_count: if min_front_liveness_candidate_count == usize::MAX {
            0
        } else {
            min_front_liveness_candidate_count
        },
        max_extent_front_liveness_margin: finite_report_metric(
            max_extent_front_liveness_margin,
            0.0,
        ),
        min_extent_front_liveness_candidate_count: if min_extent_front_liveness_candidate_count
            == usize::MAX
        {
            0
        } else {
            min_extent_front_liveness_candidate_count
        },
        max_temporal_front_liveness_margin: finite_report_metric(
            max_temporal_front_liveness_margin,
            0.0,
        ),
        min_temporal_front_liveness_candidate_count: if min_temporal_front_liveness_candidate_count
            == usize::MAX
        {
            0
        } else {
            min_temporal_front_liveness_candidate_count
        },
        max_temporal_extent_front_liveness_margin: finite_report_metric(
            max_temporal_extent_front_liveness_margin,
            0.0,
        ),
        min_temporal_extent_front_liveness_candidate_count:
            if min_temporal_extent_front_liveness_candidate_count == usize::MAX {
                0
            } else {
                min_temporal_extent_front_liveness_candidate_count
            },
        max_temporal_activation_schedule_error: finite_report_metric(
            max_temporal_activation_schedule_error,
            RENDER_SELECTION_BAD_SCORE,
        ),
        all_temporal_activation_progressive,
        all_temporal_geometry_progressive,
        morphology_non_regressed,
        worst_seed,
        worst_failure_reasons,
        base_report: base_case.render_loss,
    })
}

pub(crate) fn render_selection_candidate_beats(
    selection_score: f32,
    best_selection_score: f32,
    morphology_non_regressed: bool,
    selection_render_loss: f32,
    best_render_loss: f32,
    selection_density_psnr_db: f32,
    best_density_psnr_db: f32,
) -> bool {
    let strict_score_improvement = best_selection_score - selection_score;
    morphology_non_regressed
        && selection_score < best_selection_score
        && (render_selection_render_non_regressed(
            selection_render_loss,
            best_render_loss,
            selection_density_psnr_db,
            best_density_psnr_db,
        ) || render_selection_render_within_strict_improvement_slack(
            strict_score_improvement,
            selection_render_loss,
            best_render_loss,
            selection_density_psnr_db,
            best_density_psnr_db,
        ))
}

pub(crate) fn render_selection_candidate_metrics_beats(
    selection: &RenderSelectionMetrics,
    best: &RenderSelectionMetrics,
) -> bool {
    render_selection_candidate_beats(
        selection.score,
        best.score,
        selection.morphology_non_regressed,
        selection.render_loss,
        best.render_loss,
        selection.density_psnr_db,
        best.density_psnr_db,
    ) || render_selection_liveness_precursor_beats(selection, best)
        || render_selection_activation_breakthrough_beats(selection, best)
        || render_selection_material_precursor_beats(selection, best)
        || render_selection_post_activation_refinement_beats(selection, best)
}

pub(crate) fn render_selection_morphology_recovery_beats(
    selection: &RenderSelectionMetrics,
    best: &RenderSelectionMetrics,
) -> bool {
    !best.morphology_non_regressed
        && selection.morphology_non_regressed
        && selection.score.is_finite()
        && best.score.is_finite()
        && selection.score < best.score
        && render_selection_render_non_regressed(
            selection.render_loss,
            best.render_loss,
            selection.density_psnr_db,
            best.density_psnr_db,
        )
}

pub(crate) fn render_selection_training_progress_beats(
    selection: &RenderSelectionMetrics,
    previous: &RenderSelectionMetrics,
) -> bool {
    const RENDER_LOSS_PROGRESS: f32 = 5.0e-3;
    const COVERAGE_PROGRESS: f32 = 0.05;
    const MATERIAL_DISTANCE_PROGRESS: f32 = 0.02;
    const EXTENT_PROGRESS: f32 = 0.05;
    const TEMPORAL_ERROR_SLACK: f32 = 0.035;
    const SURFACE_MAX_SLACK: f32 = 0.30;
    const MATERIAL_TAIL_SLACK: f32 = 0.02;
    const MIN_LOCAL_FRONT_FRACTION: f32 = 0.50;

    if !selection.render_loss.is_finite()
        || !previous.render_loss.is_finite()
        || !selection.score.is_finite()
    {
        return false;
    }
    let render_improved = selection.render_loss + RENDER_LOSS_PROGRESS <= previous.render_loss
        || selection.density_psnr_db >= previous.density_psnr_db + 0.05;
    if !render_improved {
        return false;
    }

    let coverage_improved = selection.surface_covered_bin_fraction
        >= previous.surface_covered_bin_fraction + COVERAGE_PROGRESS
        || selection.surface_normal_covered_bin_fraction
            >= previous.surface_normal_covered_bin_fraction + COVERAGE_PROGRESS
        || selection.material_visible_surface_covered_bin_fraction
            >= previous.material_visible_surface_covered_bin_fraction + COVERAGE_PROGRESS
        || selection.material_visible_surface_normal_covered_bin_fraction
            >= previous.material_visible_surface_normal_covered_bin_fraction + COVERAGE_PROGRESS
        || selection.target_coverage_fraction >= previous.target_coverage_fraction + 0.02
        || selection.material_visible_target_coverage_fraction
            >= previous.material_visible_target_coverage_fraction + 0.02;
    let material_distance_improved = previous.material_visible_target_mean_distance.is_finite()
        && selection.material_visible_target_mean_distance.is_finite()
        && selection.material_visible_target_mean_distance + MATERIAL_DISTANCE_PROGRESS
            <= previous.material_visible_target_mean_distance;
    let extent_improved = selection.min_active_extent_bbox_ratio
        >= previous.min_active_extent_bbox_ratio + EXTENT_PROGRESS
        || selection.min_active_extent_min_axis_ratio
            >= previous.min_active_extent_min_axis_ratio + EXTENT_PROGRESS;
    let activation_improved = selection.min_final_active_count > previous.min_final_active_count
        && selection.min_newly_activated_fraction >= previous.min_newly_activated_fraction;
    if !(coverage_improved || material_distance_improved || extent_improved || activation_improved)
    {
        return false;
    }

    let temporal_ok = selection.max_temporal_activation_schedule_error.is_finite()
        && previous.max_temporal_activation_schedule_error.is_finite()
        && selection.max_temporal_activation_schedule_error
            <= previous.max_temporal_activation_schedule_error + TEMPORAL_ERROR_SLACK;
    let surface_ok = selection.active_surface_max.is_finite()
        && previous.active_surface_max.is_finite()
        && selection.active_surface_max
            <= (GROWTH_3D_SURFACE_MAX_DISTANCE + SURFACE_MAX_SLACK)
                .max(previous.active_surface_max + SURFACE_MAX_SLACK);
    let material_tail_ok = selection.material_visible_surface_tail_over_threshold_fraction
        <= previous.material_visible_surface_tail_over_threshold_fraction + MATERIAL_TAIL_SLACK;
    let local_front_ok = selection.min_front_local_newly_activated_fraction
        >= MIN_LOCAL_FRONT_FRACTION
        || selection.min_front_local_newly_activated_fraction
            >= previous.min_front_local_newly_activated_fraction - 0.02;
    temporal_ok && surface_ok && material_tail_ok && local_front_ok
}

pub(crate) fn render_selection_liveness_precursor_beats(
    selection: &RenderSelectionMetrics,
    best: &RenderSelectionMetrics,
) -> bool {
    const MIN_FRONT_MARGIN_IMPROVEMENT: f32 = 0.05;

    if selection.score >= best.score {
        return false;
    }
    if !render_selection_render_within_liveness_precursor_slack(
        selection.render_loss,
        best.render_loss,
        selection.density_psnr_db,
        best.density_psnr_db,
    ) {
        return false;
    }

    let front_margin_improvement = if selection.max_front_liveness_margin.is_finite()
        && best.max_front_liveness_margin.is_finite()
    {
        best.max_front_liveness_margin - selection.max_front_liveness_margin
    } else {
        0.0
    };
    let temporal_front_margin_improvement =
        if selection.max_temporal_front_liveness_margin.is_finite()
            && best.max_temporal_front_liveness_margin.is_finite()
        {
            best.max_temporal_front_liveness_margin - selection.max_temporal_front_liveness_margin
        } else {
            0.0
        };
    let temporal_extent_front_margin_improvement = if selection
        .max_temporal_extent_front_liveness_margin
        .is_finite()
        && best.max_temporal_extent_front_liveness_margin.is_finite()
    {
        best.max_temporal_extent_front_liveness_margin
            - selection.max_temporal_extent_front_liveness_margin
    } else {
        0.0
    };
    let extent_front_margin_improvement = if selection.max_extent_front_liveness_margin.is_finite()
        && best.max_extent_front_liveness_margin.is_finite()
    {
        best.max_extent_front_liveness_margin - selection.max_extent_front_liveness_margin
    } else {
        0.0
    };
    let newly_activated_improvement = selection.min_newly_activated_fraction.is_finite()
        && best.min_newly_activated_fraction.is_finite()
        && selection.min_newly_activated_fraction > best.min_newly_activated_fraction + 1.0e-4;
    let local_front_improvement = selection
        .min_front_local_newly_activated_fraction
        .is_finite()
        && best.min_front_local_newly_activated_fraction.is_finite()
        && selection.min_front_local_newly_activated_fraction
            > best.min_front_local_newly_activated_fraction + 1.0e-4;
    let terminal_front_liveness_improvement = selection.min_front_liveness_candidate_count > 0
        && front_margin_improvement >= MIN_FRONT_MARGIN_IMPROVEMENT;
    let extent_front_liveness_improvement = selection.min_extent_front_liveness_candidate_count > 0
        && extent_front_margin_improvement >= MIN_FRONT_MARGIN_IMPROVEMENT;
    let temporal_front_liveness_improvement = selection.min_temporal_front_liveness_candidate_count
        > 0
        && temporal_front_margin_improvement >= MIN_FRONT_MARGIN_IMPROVEMENT;
    let temporal_extent_front_liveness_improvement =
        selection.min_temporal_extent_front_liveness_candidate_count > 0
            && temporal_extent_front_margin_improvement >= MIN_FRONT_MARGIN_IMPROVEMENT;
    let bounded_temporal_front_precursor = selection.morphology_non_regressed
        || render_selection_bounded_temporal_front_precursor(selection, best);

    (selection.morphology_non_regressed
        && (newly_activated_improvement
            || local_front_improvement
            || terminal_front_liveness_improvement
            || extent_front_liveness_improvement))
        || (bounded_temporal_front_precursor
            && (temporal_front_liveness_improvement || temporal_extent_front_liveness_improvement))
}

pub(crate) fn render_selection_bounded_temporal_front_precursor(
    selection: &RenderSelectionMetrics,
    best: &RenderSelectionMetrics,
) -> bool {
    render_selection_temporal_activation_not_regressed(selection, best)
        && selection.min_final_active_count <= best.min_final_active_count.saturating_add(1)
        && selection.min_newly_activated_fraction
            <= best.min_newly_activated_fraction.max(0.0) + 0.02
        && selection.active_surface_max.is_finite()
        && selection.active_surface_max <= GROWTH_3D_SURFACE_MAX_DISTANCE + 0.05
        && selection.material_visible_inactive_fraction
            <= best.material_visible_inactive_fraction + 0.01
        && selection.material_visible_surface_tail_over_threshold_fraction <= 0.01
}

pub(crate) fn render_selection_render_within_liveness_precursor_slack(
    selection_render_loss: f32,
    best_render_loss: f32,
    selection_density_psnr_db: f32,
    best_density_psnr_db: f32,
) -> bool {
    const RENDER_LOSS_SLACK_ABS: f32 = 5.0e-4;
    const RENDER_LOSS_SLACK_FRACTION: f32 = 5.0e-4;
    const DENSITY_PSNR_SLACK_DB: f32 = 0.01;

    if !selection_render_loss.is_finite()
        || !best_render_loss.is_finite()
        || !selection_density_psnr_db.is_finite()
        || !best_density_psnr_db.is_finite()
    {
        return false;
    }
    let render_loss_slack =
        RENDER_LOSS_SLACK_ABS.max(best_render_loss.abs() * RENDER_LOSS_SLACK_FRACTION);
    selection_render_loss <= best_render_loss + render_loss_slack
        && selection_density_psnr_db + DENSITY_PSNR_SLACK_DB >= best_density_psnr_db
}

pub(crate) fn render_selection_activation_breakthrough_beats(
    selection: &RenderSelectionMetrics,
    best: &RenderSelectionMetrics,
) -> bool {
    const MIN_ACTIVATION_FRACTION_IMPROVEMENT: f32 = 0.05;
    if selection.score >= best.score {
        return false;
    }
    if !selection.min_newly_activated_fraction.is_finite()
        || !best.min_newly_activated_fraction.is_finite()
        || !selection
            .min_front_local_newly_activated_fraction
            .is_finite()
        || !best.min_front_local_newly_activated_fraction.is_finite()
    {
        return false;
    }
    let activation_improved = selection.min_newly_activated_fraction
        >= best.min_newly_activated_fraction + MIN_ACTIVATION_FRACTION_IMPROVEMENT
        && selection.min_front_local_newly_activated_fraction
            >= best.min_front_local_newly_activated_fraction + MIN_ACTIVATION_FRACTION_IMPROVEMENT
        && selection.min_final_active_count > best.min_final_active_count;
    if !activation_improved {
        return false;
    }
    let strict_score_improvement = best.score - selection.score;
    if !(render_selection_render_non_regressed(
        selection.render_loss,
        best.render_loss,
        selection.density_psnr_db,
        best.density_psnr_db,
    ) || render_selection_render_within_strict_improvement_slack(
        strict_score_improvement,
        selection.render_loss,
        best.render_loss,
        selection.density_psnr_db,
        best.density_psnr_db,
    )) {
        return false;
    }
    if !render_selection_temporal_activation_not_regressed(selection, best) {
        return false;
    }
    selection.all_temporal_activation_progressive
        && selection.active_surface_max.is_finite()
        && selection.active_surface_max <= GROWTH_3D_SURFACE_MAX_DISTANCE + 0.05
        && selection.material_visible_inactive_fraction
            <= best.material_visible_inactive_fraction + 0.01
        && selection.material_visible_surface_tail_over_threshold_fraction <= 0.01
}

pub(crate) fn render_selection_material_precursor_beats(
    selection: &RenderSelectionMetrics,
    best: &RenderSelectionMetrics,
) -> bool {
    const MATERIAL_MEAN_IMPROVEMENT: f32 = 0.05;
    let visible_count_improved = selection.material_visible_count > best.material_visible_count;
    let active_mean_improved = selection.material_active_mean_opacity.is_finite()
        && best.material_active_mean_opacity.is_finite()
        && selection.material_active_mean_opacity
            > best.material_active_mean_opacity + MATERIAL_MEAN_IMPROVEMENT;
    if !visible_count_improved && !active_mean_improved {
        return false;
    }
    if selection.min_final_active_count < best.min_final_active_count {
        return false;
    }
    if !render_selection_temporal_activation_not_regressed(selection, best) {
        return false;
    }
    if !render_selection_render_within_liveness_precursor_slack(
        selection.render_loss,
        best.render_loss,
        selection.density_psnr_db,
        best.density_psnr_db,
    ) {
        return false;
    }
    selection.material_visible_inactive_fraction
        <= best.material_visible_inactive_fraction.max(0.0) + 0.01
        && selection.material_visible_surface_tail_over_threshold_fraction <= 0.01
        && selection.active_surface_max.is_finite()
        && selection.active_surface_max <= GROWTH_3D_SURFACE_MAX_DISTANCE + 0.05
}

pub(crate) fn render_selection_post_activation_refinement_beats(
    selection: &RenderSelectionMetrics,
    best: &RenderSelectionMetrics,
) -> bool {
    const ACTIVATION_TOLERANCE: f32 = 0.02;
    if selection.score >= best.score
        || best.min_newly_activated_fraction < 0.05
        || best.min_front_local_newly_activated_fraction < 0.05
        || !best.all_temporal_activation_progressive
    {
        return false;
    }
    if selection.min_final_active_count < best.min_final_active_count
        || selection.min_newly_activated_fraction + ACTIVATION_TOLERANCE
            < best.min_newly_activated_fraction
        || selection.min_front_local_newly_activated_fraction + ACTIVATION_TOLERANCE
            < best.min_front_local_newly_activated_fraction
    {
        return false;
    }
    let strict_score_improvement = best.score - selection.score;
    if !(render_selection_render_non_regressed(
        selection.render_loss,
        best.render_loss,
        selection.density_psnr_db,
        best.density_psnr_db,
    ) || render_selection_render_within_strict_improvement_slack(
        strict_score_improvement,
        selection.render_loss,
        best.render_loss,
        selection.density_psnr_db,
        best.density_psnr_db,
    )) {
        return false;
    }
    if !render_selection_temporal_activation_not_regressed(selection, best) {
        return false;
    }
    selection.active_surface_max.is_finite()
        && selection.active_surface_max <= GROWTH_3D_SURFACE_MAX_DISTANCE + 0.05
        && selection.material_visible_inactive_fraction
            <= best.material_visible_inactive_fraction + 0.01
        && selection.material_visible_surface_tail_over_threshold_fraction <= 0.01
}

pub(crate) fn render_selection_temporal_activation_not_regressed(
    selection: &RenderSelectionMetrics,
    best: &RenderSelectionMetrics,
) -> bool {
    if selection.all_temporal_activation_progressive {
        return true;
    }
    if !selection.max_temporal_activation_schedule_error.is_finite()
        || !best.max_temporal_activation_schedule_error.is_finite()
    {
        return false;
    }
    selection.max_temporal_activation_schedule_error
        <= best.max_temporal_activation_schedule_error
            + TEMPORAL_ACTIVATION_SELECTION_REGRESSION_SLACK
}

pub(crate) fn render_selection_render_non_regressed(
    selection_render_loss: f32,
    best_render_loss: f32,
    selection_density_psnr_db: f32,
    best_density_psnr_db: f32,
) -> bool {
    const LOSS_TOLERANCE: f32 = 1.0e-5;
    const DENSITY_PSNR_TOLERANCE_DB: f32 = 1.0e-4;
    selection_render_loss.is_finite()
        && best_render_loss.is_finite()
        && selection_density_psnr_db.is_finite()
        && best_density_psnr_db.is_finite()
        && selection_render_loss <= best_render_loss + LOSS_TOLERANCE
        && selection_density_psnr_db + DENSITY_PSNR_TOLERANCE_DB >= best_density_psnr_db
}

pub(crate) fn render_selection_render_within_strict_improvement_slack(
    strict_score_improvement: f32,
    selection_render_loss: f32,
    best_render_loss: f32,
    selection_density_psnr_db: f32,
    best_density_psnr_db: f32,
) -> bool {
    const MIN_STRICT_SCORE_IMPROVEMENT: f32 = 0.25;
    const RENDER_LOSS_SLACK_ABS: f32 = 0.005;
    const RENDER_LOSS_SLACK_FRACTION: f32 = 0.03;
    const DENSITY_PSNR_SLACK_DB: f32 = 0.25;

    if strict_score_improvement < MIN_STRICT_SCORE_IMPROVEMENT {
        return false;
    }
    if !selection_render_loss.is_finite()
        || !best_render_loss.is_finite()
        || !selection_density_psnr_db.is_finite()
        || !best_density_psnr_db.is_finite()
    {
        return false;
    }
    let render_loss_slack =
        RENDER_LOSS_SLACK_ABS.max(best_render_loss * RENDER_LOSS_SLACK_FRACTION);
    selection_render_loss <= best_render_loss + render_loss_slack
        && selection_density_psnr_db + DENSITY_PSNR_SLACK_DB >= best_density_psnr_db
}

pub(crate) fn finite_report_metric(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

pub(crate) fn material_training_soft_coverage_threshold(seed_scale: f32) -> f32 {
    let strict = target_coverage_threshold(seed_scale).max(1.0e-6);
    (strict * 3.0).min(seed_scale.max(strict)).max(strict)
}

pub(crate) fn direct_trajectory_geometry_weight(step_fraction: f32) -> f32 {
    let schedule = step_fraction.clamp(0.0, 1.0);
    0.5 + 0.5 * schedule
}

pub(crate) fn direct_growth_phase_gain(cfg: &RenderProxyTrainingConfig) -> f32 {
    if cfg.liveness_gain <= 0.0 || !cfg.liveness_gain.is_finite() {
        return 0.0;
    }
    (cfg.liveness_gain * DIRECT_GROWTH_PHASE_GAIN_FRACTION).max(ROBUST_3D_PHASE_GAIN)
}

pub(crate) fn soft_material_assignment_weight(
    distance: f32,
    strict_threshold: f32,
    soft_threshold: f32,
) -> f32 {
    if !distance.is_finite() {
        return 0.0;
    }
    let strict = strict_threshold.max(1.0e-6);
    let soft = soft_threshold.max(strict);
    if distance <= strict {
        1.0
    } else if distance >= soft {
        0.0
    } else {
        (1.0 - (distance - strict) / (soft - strict).max(1.0e-6)).clamp(0.0, 1.0)
    }
}

pub(crate) fn render_selection_baseline(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: &RenderProxyTrainingConfig,
    render_cfg: RenderLossConfig,
) -> Result<Vec<RenderSelectionBaselineCase>, Box<dyn std::error::Error>> {
    let selection_seeds = render_proxy_selection_seeds(cfg);
    let mut baselines = Vec::with_capacity(selection_seeds.len());
    for seed in selection_seeds {
        let selection_case =
            render_selection_case_metrics(model, grid, target, cfg, render_cfg, seed)?;
        baselines.push(RenderSelectionBaselineCase {
            seed,
            active_surface_max: selection_case.active_surface.max_distance,
            target_coverage_fraction: selection_case.target_coverage.covered_fraction,
            material_visible_target_mean_distance: selection_case
                .material_visible_target_coverage
                .mean_distance,
            material_visible_target_max_distance: selection_case
                .material_visible_target_coverage
                .max_distance,
            material_visible_target_coverage_fraction: selection_case
                .material_visible_target_coverage
                .covered_fraction,
            material_visible_inactive_fraction: selection_case
                .material_liveness
                .inactive_material_visible_fraction,
            material_visible_max_inactive_opacity: selection_case
                .material_liveness
                .max_inactive_material_opacity,
            surface_covered_bin_fraction: selection_case
                .surface_coverage_profile
                .covered_bin_fraction,
            surface_mean_bin_covered_fraction: selection_case
                .surface_coverage_profile
                .mean_bin_covered_fraction,
            material_visible_surface_covered_bin_fraction: selection_case
                .material_visible_surface_coverage_profile
                .covered_bin_fraction,
            material_visible_surface_mean_bin_covered_fraction: selection_case
                .material_visible_surface_coverage_profile
                .mean_bin_covered_fraction,
            surface_normal_covered_bin_fraction: selection_case
                .surface_normal_coverage
                .covered_target_bin_fraction,
            surface_normal_mean_bin_covered_fraction: selection_case
                .surface_normal_coverage
                .mean_bin_covered_fraction,
            material_visible_surface_normal_covered_bin_fraction: selection_case
                .material_visible_surface_normal_coverage
                .covered_target_bin_fraction,
            material_visible_surface_normal_mean_bin_covered_fraction: selection_case
                .material_visible_surface_normal_coverage
                .mean_bin_covered_fraction,
            material_visible_surface_tail_p99_distance: selection_case
                .material_visible_surface_tail
                .p99_distance,
            material_visible_surface_tail_over_threshold_fraction: selection_case
                .material_visible_surface_tail
                .over_threshold_fraction,
            active_extent_bbox_ratio: selection_case.extent.bbox_diagonal_ratio,
            active_extent_min_axis_ratio: selection_case.extent.min_axis_extent_ratio,
            final_active_count: selection_case.final_active_count,
            newly_activated_fraction: selection_case.newly_activated_fraction,
            front_local_newly_activated_fraction: selection_case
                .front_local_newly_activated_fraction,
            front_liveness: selection_case.front_liveness,
            extent_front_liveness: selection_case.extent_front_liveness,
            temporal_front_liveness: selection_case.temporal_front_liveness,
            temporal_extent_front_liveness: selection_case.temporal_extent_front_liveness,
            temporal_activation_schedule_error: selection_case.temporal_activation_schedule_error,
            temporal_activation_progressive: selection_case.temporal_activation_progressive,
            temporal_geometry_progressive: selection_case.temporal_geometry_progressive,
        });
    }
    Ok(baselines)
}

pub(crate) fn render_selection_case_score_with_baseline(
    seed: u64,
    case: &RenderSelectionCaseMetrics,
    baseline: Option<&[RenderSelectionBaselineCase]>,
) -> RenderSelectionCaseScore {
    let front_liveness_penalty = finite_report_metric(
        case.front_liveness.weighted_activation_margin,
        RENDER_SELECTION_BAD_SCORE,
    )
    .clamp(0.0, RENDER_SELECTION_BAD_SCORE)
        * LOCAL_FRONT_LIVENESS_SCORE_WEIGHT;
    let temporal_front_liveness_penalty = finite_report_metric(
        case.temporal_front_liveness.weighted_activation_margin,
        RENDER_SELECTION_BAD_SCORE,
    )
    .clamp(0.0, RENDER_SELECTION_BAD_SCORE)
        * LOCAL_FRONT_LIVENESS_SCORE_WEIGHT;
    let temporal_extent_front_liveness_penalty = finite_report_metric(
        case.temporal_extent_front_liveness
            .weighted_activation_margin,
        RENDER_SELECTION_BAD_SCORE,
    )
    .clamp(0.0, RENDER_SELECTION_BAD_SCORE)
        * LOCAL_FRONT_LIVENESS_SCORE_WEIGHT;
    let extent_front_liveness_penalty = finite_report_metric(
        case.extent_front_liveness.weighted_activation_margin,
        RENDER_SELECTION_BAD_SCORE,
    )
    .clamp(0.0, RENDER_SELECTION_BAD_SCORE)
        * LOCAL_FRONT_LIVENESS_SCORE_WEIGHT;
    let temporal_activation_penalty = finite_report_metric(
        case.temporal_activation_schedule_error,
        RENDER_SELECTION_BAD_SCORE,
    )
    .clamp(0.0, RENDER_SELECTION_BAD_SCORE)
        * TEMPORAL_ACTIVATION_SCORE_WEIGHT;
    let material_visible_target_mean_distance_penalty = finite_report_metric(
        case.material_visible_target_coverage.mean_distance,
        RENDER_SELECTION_BAD_SCORE,
    )
    .clamp(0.0, RENDER_SELECTION_BAD_SCORE)
        * MATERIAL_VISIBLE_TARGET_MEAN_DISTANCE_SCORE_WEIGHT;
    let material_visible_target_max_distance_penalty = finite_report_metric(
        case.material_visible_target_coverage.max_distance,
        RENDER_SELECTION_BAD_SCORE,
    )
    .clamp(0.0, RENDER_SELECTION_BAD_SCORE)
        * MATERIAL_VISIBLE_TARGET_MAX_DISTANCE_SCORE_WEIGHT;
    let mut score = finite_report_metric(case.score, RENDER_SELECTION_BAD_SCORE)
        + front_liveness_penalty
        + extent_front_liveness_penalty
        + temporal_front_liveness_penalty
        + temporal_extent_front_liveness_penalty
        + temporal_activation_penalty
        + material_visible_target_mean_distance_penalty
        + material_visible_target_max_distance_penalty;
    let mut morphology_non_regressed = true;
    if let Some(baseline_case) = baseline.and_then(|cases| {
        cases
            .iter()
            .find(|baseline_case| baseline_case.seed == seed)
    }) {
        let surface_regression = if case.active_surface.max_distance.is_finite() {
            (case.active_surface.max_distance - baseline_case.active_surface_max - 0.02).max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let coverage_regression = if case.target_coverage.covered_fraction.is_finite() {
            (baseline_case.target_coverage_fraction - case.target_coverage.covered_fraction - 0.02)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let material_visible_coverage_regression = if case
            .material_visible_target_coverage
            .covered_fraction
            .is_finite()
        {
            (baseline_case.material_visible_target_coverage_fraction
                - case.material_visible_target_coverage.covered_fraction
                - 0.02)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let material_visible_target_mean_regression = if case
            .material_visible_target_coverage
            .mean_distance
            .is_finite()
            && baseline_case
                .material_visible_target_mean_distance
                .is_finite()
        {
            (case.material_visible_target_coverage.mean_distance
                - baseline_case.material_visible_target_mean_distance
                - MATERIAL_VISIBLE_TARGET_DISTANCE_REGRESSION_SLACK)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let material_visible_target_max_regression = if case
            .material_visible_target_coverage
            .max_distance
            .is_finite()
            && baseline_case
                .material_visible_target_max_distance
                .is_finite()
        {
            (case.material_visible_target_coverage.max_distance
                - baseline_case.material_visible_target_max_distance
                - MATERIAL_VISIBLE_TARGET_DISTANCE_REGRESSION_SLACK)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let material_visible_inactive_fraction_regression = if case
            .material_liveness
            .inactive_material_visible_fraction
            .is_finite()
        {
            (case.material_liveness.inactive_material_visible_fraction
                - baseline_case.material_visible_inactive_fraction
                - 0.005)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let material_visible_max_inactive_opacity_regression = if case
            .material_liveness
            .max_inactive_material_opacity
            .is_finite()
            && baseline_case
                .material_visible_max_inactive_opacity
                .is_finite()
        {
            (case.material_liveness.max_inactive_material_opacity
                - baseline_case.material_visible_max_inactive_opacity
                - 0.25)
                .max(0.0)
        } else if case
            .material_liveness
            .max_inactive_material_opacity
            .is_finite()
            && !baseline_case
                .material_visible_max_inactive_opacity
                .is_finite()
        {
            (case.material_liveness.inactive_material_visible_fraction - 0.005).max(0.0)
        } else {
            0.0
        };
        let surface_bin_regression = if case
            .surface_coverage_profile
            .covered_bin_fraction
            .is_finite()
        {
            (baseline_case.surface_covered_bin_fraction
                - case.surface_coverage_profile.covered_bin_fraction
                - 0.05)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let surface_mean_regression = if case
            .surface_coverage_profile
            .mean_bin_covered_fraction
            .is_finite()
        {
            (baseline_case.surface_mean_bin_covered_fraction
                - case.surface_coverage_profile.mean_bin_covered_fraction
                - 0.05)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let material_visible_surface_bin_regression = if case
            .material_visible_surface_coverage_profile
            .covered_bin_fraction
            .is_finite()
        {
            (baseline_case.material_visible_surface_covered_bin_fraction
                - case
                    .material_visible_surface_coverage_profile
                    .covered_bin_fraction
                - 0.05)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let material_visible_surface_mean_regression = if case
            .material_visible_surface_coverage_profile
            .mean_bin_covered_fraction
            .is_finite()
        {
            (baseline_case.material_visible_surface_mean_bin_covered_fraction
                - case
                    .material_visible_surface_coverage_profile
                    .mean_bin_covered_fraction
                - 0.05)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let normal_bin_regression = if case
            .surface_normal_coverage
            .covered_target_bin_fraction
            .is_finite()
        {
            (baseline_case.surface_normal_covered_bin_fraction
                - case.surface_normal_coverage.covered_target_bin_fraction
                - 0.05)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let normal_mean_regression = if case
            .surface_normal_coverage
            .mean_bin_covered_fraction
            .is_finite()
        {
            (baseline_case.surface_normal_mean_bin_covered_fraction
                - case.surface_normal_coverage.mean_bin_covered_fraction
                - 0.05)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let material_visible_normal_bin_regression = if case
            .material_visible_surface_normal_coverage
            .covered_target_bin_fraction
            .is_finite()
        {
            (baseline_case.material_visible_surface_normal_covered_bin_fraction
                - case
                    .material_visible_surface_normal_coverage
                    .covered_target_bin_fraction
                - 0.05)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let material_visible_normal_mean_regression = if case
            .material_visible_surface_normal_coverage
            .mean_bin_covered_fraction
            .is_finite()
        {
            (baseline_case.material_visible_surface_normal_mean_bin_covered_fraction
                - case
                    .material_visible_surface_normal_coverage
                    .mean_bin_covered_fraction
                - 0.05)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let material_visible_tail_p99_regression =
            if case.material_visible_surface_tail.p99_distance.is_finite() {
                (case.material_visible_surface_tail.p99_distance
                    - baseline_case.material_visible_surface_tail_p99_distance
                    - 0.02)
                    .max(0.0)
            } else {
                RENDER_SELECTION_BAD_SCORE
            };
        let material_visible_tail_fraction_regression = if case
            .material_visible_surface_tail
            .over_threshold_fraction
            .is_finite()
        {
            (case.material_visible_surface_tail.over_threshold_fraction
                - baseline_case.material_visible_surface_tail_over_threshold_fraction
                - 0.005)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let active_extent_bbox_regression = if case.extent.bbox_diagonal_ratio.is_finite() {
            (baseline_case.active_extent_bbox_ratio - case.extent.bbox_diagonal_ratio - 0.02)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let active_extent_min_axis_regression = if case.extent.min_axis_extent_ratio.is_finite() {
            (baseline_case.active_extent_min_axis_ratio - case.extent.min_axis_extent_ratio - 0.02)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let active_count_regression = if baseline_case.final_active_count > 0 {
            let baseline_active_count = baseline_case.final_active_count as f32;
            ((baseline_active_count - case.final_active_count as f32) / baseline_active_count
                - 0.02)
                .max(0.0)
        } else {
            0.0
        };
        let newly_activated_regression = if case.newly_activated_fraction.is_finite() {
            (baseline_case.newly_activated_fraction - case.newly_activated_fraction - 0.02).max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let front_local_newly_activated_regression =
            if case.front_local_newly_activated_fraction.is_finite() {
                (baseline_case.front_local_newly_activated_fraction
                    - case.front_local_newly_activated_fraction
                    - 0.02)
                    .max(0.0)
            } else {
                RENDER_SELECTION_BAD_SCORE
            };
        let front_liveness_margin_regression =
            if case.front_liveness.weighted_activation_margin.is_finite()
                && baseline_case
                    .front_liveness
                    .weighted_activation_margin
                    .is_finite()
            {
                (case.front_liveness.weighted_activation_margin
                    - baseline_case.front_liveness.weighted_activation_margin
                    - 0.10)
                    .max(0.0)
            } else {
                RENDER_SELECTION_BAD_SCORE
            };
        let temporal_front_liveness_margin_regression = if case
            .temporal_front_liveness
            .weighted_activation_margin
            .is_finite()
            && baseline_case
                .temporal_front_liveness
                .weighted_activation_margin
                .is_finite()
        {
            (case.temporal_front_liveness.weighted_activation_margin
                - baseline_case
                    .temporal_front_liveness
                    .weighted_activation_margin
                - 0.10)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let temporal_extent_front_liveness_margin_regression = if case
            .temporal_extent_front_liveness
            .weighted_activation_margin
            .is_finite()
            && baseline_case
                .temporal_extent_front_liveness
                .weighted_activation_margin
                .is_finite()
        {
            (case
                .temporal_extent_front_liveness
                .weighted_activation_margin
                - baseline_case
                    .temporal_extent_front_liveness
                    .weighted_activation_margin
                - 0.10)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let extent_front_liveness_margin_regression = if case
            .extent_front_liveness
            .weighted_activation_margin
            .is_finite()
            && baseline_case
                .extent_front_liveness
                .weighted_activation_margin
                .is_finite()
        {
            (case.extent_front_liveness.weighted_activation_margin
                - baseline_case
                    .extent_front_liveness
                    .weighted_activation_margin
                - 0.10)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let temporal_activation_schedule_regression =
            if case.temporal_activation_schedule_error.is_finite()
                && baseline_case.temporal_activation_schedule_error.is_finite()
            {
                (case.temporal_activation_schedule_error
                    - baseline_case.temporal_activation_schedule_error
                    - TEMPORAL_ACTIVATION_SELECTION_REGRESSION_SLACK)
                    .max(0.0)
            } else {
                RENDER_SELECTION_BAD_SCORE
            };
        let temporal_activation_regression = if baseline_case.temporal_activation_progressive
            && !case.temporal_activation_progressive
        {
            1.0
        } else {
            0.0
        };
        let temporal_geometry_regression =
            if baseline_case.temporal_geometry_progressive && !case.temporal_geometry_progressive {
                1.0
            } else {
                0.0
            };
        if surface_regression > 0.0
            || coverage_regression > 0.0
            || material_visible_coverage_regression > 0.0
            || material_visible_target_mean_regression > 0.0
            || material_visible_target_max_regression > 0.0
            || material_visible_inactive_fraction_regression > 0.0
            || material_visible_max_inactive_opacity_regression > 0.0
            || surface_bin_regression > 0.0
            || surface_mean_regression > 0.0
            || material_visible_surface_bin_regression > 0.0
            || material_visible_surface_mean_regression > 0.0
            || normal_bin_regression > 0.0
            || normal_mean_regression > 0.0
            || material_visible_normal_bin_regression > 0.0
            || material_visible_normal_mean_regression > 0.0
            || material_visible_tail_p99_regression > 0.0
            || material_visible_tail_fraction_regression > 0.0
            || active_extent_bbox_regression > 0.0
            || active_extent_min_axis_regression > 0.0
            || active_count_regression > 0.0
            || newly_activated_regression > 0.0
            || front_local_newly_activated_regression > 0.0
            || front_liveness_margin_regression > 0.0
            || extent_front_liveness_margin_regression > 0.0
            || temporal_front_liveness_margin_regression > 0.0
            || temporal_extent_front_liveness_margin_regression > 0.0
            || temporal_activation_schedule_regression > 0.0
            || temporal_activation_regression > 0.0
            || temporal_geometry_regression > 0.0
        {
            morphology_non_regressed = false;
        }
        score += (surface_regression
            + coverage_regression
            + material_visible_coverage_regression
            + material_visible_target_mean_regression
            + material_visible_target_max_regression
            + material_visible_inactive_fraction_regression
            + material_visible_max_inactive_opacity_regression
            + surface_bin_regression
            + surface_mean_regression
            + material_visible_surface_bin_regression
            + material_visible_surface_mean_regression
            + normal_bin_regression
            + normal_mean_regression
            + material_visible_normal_bin_regression
            + material_visible_normal_mean_regression
            + material_visible_tail_p99_regression
            + material_visible_tail_fraction_regression
            + active_extent_bbox_regression
            + active_extent_min_axis_regression
            + active_count_regression
            + newly_activated_regression
            + front_local_newly_activated_regression
            + front_liveness_margin_regression
            + extent_front_liveness_margin_regression
            + temporal_front_liveness_margin_regression
            + temporal_extent_front_liveness_margin_regression
            + temporal_activation_schedule_regression
            + temporal_activation_regression
            + temporal_geometry_regression)
            * 10.0;
    }
    RenderSelectionCaseScore {
        score,
        morphology_non_regressed,
    }
}

pub(crate) struct RenderSelectionCaseScore {
    pub(crate) score: f32,
    pub(crate) morphology_non_regressed: bool,
}

pub(crate) struct RenderSelectionCaseMetrics {
    pub(crate) render_loss: MultiViewRenderLossReport,
    pub(crate) active_surface: Growth3dSurfaceStats,
    pub(crate) target_coverage: TargetCoverageStats,
    pub(crate) material_visible_target_coverage: TargetCoverageStats,
    pub(crate) material_opacity: Growth3dOpacityStats,
    pub(crate) material_liveness: Growth3dMaterialLivenessReport,
    pub(crate) surface_coverage_profile: SurfaceCoverageProfileReport,
    pub(crate) material_visible_surface_coverage_profile: SurfaceCoverageProfileReport,
    pub(crate) surface_normal_coverage: SurfaceNormalCoverageReport,
    pub(crate) material_visible_surface_normal_coverage: SurfaceNormalCoverageReport,
    pub(crate) material_visible_surface_tail: Growth3dSurfaceTailReport,
    pub(crate) extent: Growth3dExtentReport,
    pub(crate) final_active_count: usize,
    pub(crate) newly_activated_fraction: f32,
    pub(crate) front_local_newly_activated_fraction: f32,
    pub(crate) front_liveness: LocalFrontLivenessProgress,
    pub(crate) extent_front_liveness: LocalFrontLivenessProgress,
    pub(crate) temporal_front_liveness: LocalFrontLivenessProgress,
    pub(crate) temporal_extent_front_liveness: LocalFrontLivenessProgress,
    pub(crate) temporal_activation_schedule_error: f32,
    pub(crate) temporal_activation_progressive: bool,
    pub(crate) temporal_geometry_progressive: bool,
    pub(crate) score: f32,
    pub(crate) failure_reasons: Vec<&'static str>,
}

pub(crate) fn render_selection_case_metrics(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: &RenderProxyTrainingConfig,
    render_cfg: RenderLossConfig,
    seed: u64,
) -> Result<RenderSelectionCaseMetrics, Box<dyn std::error::Error>> {
    let trace = render_training_trace_for_seed(model, grid, cfg, seed)?;
    let render_loss = mesh_multiview_render_loss_from_trace(&trace, target, render_cfg)?;
    let final_gaussian_volume = gaussian_volume_stats_for_trace(&trace, render_cfg);
    let rollout_cfg = RolloutConfig {
        particle_count: cfg.particles,
        steps: cfg.rollout_steps,
        update_prob: 1.0,
        seed,
        seed_scale: cfg.seed_scale,
        ..RolloutConfig::default()
    };
    let (seed_positions, seed_states) = seed_particles_scaled(
        1,
        rollout_cfg.particle_count,
        model.config.state_dims,
        model.config.spatial_dims,
        rollout_cfg.seed,
        cfg.seed_mode,
        rollout_cfg.seed_scale,
    );
    let mut active_seed_count = 0usize;
    let mut seed_active = Vec::with_capacity(rollout_cfg.particle_count);
    for state in seed_states.chunks_exact(model.config.state_dims) {
        let active = state[3] > -1.0;
        seed_active.push(active);
        if active {
            active_seed_count += 1;
        }
    }
    let non_opacity_seed_abs_max =
        growth_3d_non_scaffold_seed_abs_max(model.config.state_dims, cfg.seed_mode, &seed_states);
    let activation = growth_3d_activation_report(&trace, &seed_active, active_seed_count);
    let initial_active_surface = growth_3d_active_surface_stats(
        &seed_positions,
        &seed_states,
        model.config.state_dims,
        target,
    );
    let active_surface =
        growth_3d_active_surface_stats(&trace.positions, &trace.states, trace.state_dims, target);
    let active_surface_tail = growth_3d_active_surface_tail_report(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        target,
        GROWTH_3D_SURFACE_MAX_DISTANCE,
    );
    let coverage_samples = cfg.particles.max(512);
    let coverage_threshold = target_coverage_threshold(cfg.seed_scale);
    let initial_target_coverage = active_target_coverage_stats(
        &seed_positions,
        &seed_states,
        model.config.state_dims,
        target,
        coverage_samples,
        coverage_threshold,
    );
    let target_coverage = active_target_coverage_stats(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        target,
        coverage_samples,
        coverage_threshold,
    );
    let material_visible_target_coverage = material_visible_target_coverage_stats(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        target,
        coverage_samples,
        coverage_threshold,
    );
    let surface_coverage_profile = active_surface_coverage_profile(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        target,
        coverage_samples,
        coverage_threshold,
        64,
    );
    let material_visible_surface_coverage_profile = material_visible_surface_coverage_profile(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        target,
        coverage_samples,
        coverage_threshold,
        64,
    );
    let surface_normal_coverage = active_surface_normal_coverage_report(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        target,
        coverage_samples,
        coverage_threshold,
    );
    let material_visible_surface_normal_coverage = material_visible_surface_normal_coverage_report(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        target,
        coverage_samples,
        coverage_threshold,
    );
    let material_visible_surface_tail = growth_3d_material_visible_surface_tail_report(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        target,
        GROWTH_3D_SURFACE_MAX_DISTANCE,
    );
    let torus_angular_coverage = (cfg.target == MeshTargetArg::Torus).then(|| {
        torus_angular_coverage_report(
            &trace.positions,
            &trace.states,
            trace.state_dims,
            cfg.seed_scale,
            coverage_threshold,
            TORUS_ANGULAR_COVERAGE_RINGS,
            TORUS_ANGULAR_COVERAGE_TUBES,
        )
    });
    let motion = growth_3d_motion_report(&trace.mean_dx);
    let extent = growth_3d_extent_report(&trace.positions, &trace.states, trace.state_dims, target);
    let final_opacity = growth_3d_opacity_stats(&trace.states, trace.state_dims);
    let material_opacity = growth_3d_material_opacity_stats(&trace.states, trace.state_dims);
    let material_liveness = growth_3d_material_liveness_report(&trace.states, trace.state_dims);
    let initial_color_state = growth_3d_color_state_report(&seed_states, model.config.state_dims);
    let final_color_state = growth_3d_color_state_report(&trace.states, trace.state_dims);
    let temporal = growth_3d_temporal_report(
        model,
        grid,
        target,
        rollout_cfg.clone(),
        cfg.seed_mode,
        &seed_positions,
        &seed_states,
        &seed_active,
        active_seed_count,
        &trace,
        coverage_samples,
        coverage_threshold,
    )?;
    let permutation_consistency =
        growth_3d_permutation_report(model, grid, &rollout_cfg, cfg.seed_mode)?;
    let front = growth_3d_front_report(
        model,
        grid,
        rollout_cfg,
        cfg.seed_mode,
        &seed_positions,
        &seed_states,
        &trace,
    )?;
    let front_liveness = local_front_liveness_progress(
        &model.config,
        &trace.positions,
        &trace.states,
        cfg.liveness_front_radius,
    );
    let extent_front_liveness = extent_front_liveness_progress(
        &model.config,
        target,
        &trace.positions,
        &trace.states,
        cfg.liveness_front_radius,
    );
    let temporal_front_liveness =
        temporal_front_liveness_progress(model, grid, cfg, seed, &seed_positions, &seed_states)?;
    let temporal_extent_front_liveness = temporal_extent_front_liveness_progress(
        model,
        grid,
        target,
        cfg,
        seed,
        &seed_positions,
        &seed_states,
    )?;
    let mean_final_displacement = growth_3d_mean_displacement(&seed_positions, &trace.positions);
    let mut strict_checks = growth_3d_strict_checks_report(
        model.config.position_features,
        true,
        growth_3d_seed_has_coordinate_scaffold(cfg.seed_mode),
        non_opacity_seed_abs_max,
        final_opacity,
        initial_color_state,
        final_color_state,
        &permutation_consistency,
        &activation,
        initial_active_surface,
        active_surface,
        active_surface_tail,
        initial_target_coverage,
        target_coverage,
        material_visible_target_coverage,
        &surface_normal_coverage,
        &material_visible_surface_normal_coverage,
        torus_angular_coverage.as_ref(),
        final_gaussian_volume,
        &motion,
        &front,
        &temporal,
        extent,
        mean_final_displacement,
        cfg.seed_scale,
        cfg.particles,
        render_loss.passed,
    );
    apply_material_liveness_strict_check(&mut strict_checks, material_liveness);
    apply_material_visible_surface_tail_strict_check(
        &mut strict_checks,
        material_visible_surface_tail,
    );
    apply_surface_profile_strict_check(
        &mut strict_checks,
        &surface_coverage_profile,
        &material_visible_surface_coverage_profile,
    );
    let mut strict_score = growth_3d_strict_score_report(
        &strict_checks,
        initial_active_surface,
        active_surface,
        active_surface_tail,
        initial_target_coverage,
        target_coverage,
        material_visible_target_coverage,
        &surface_normal_coverage,
        &material_visible_surface_normal_coverage,
        extent,
        cfg.seed_scale,
        &render_loss,
        final_gaussian_volume,
    );
    apply_temporal_activation_strict_score(&mut strict_score, &temporal, cfg.rollout_steps);
    apply_material_liveness_strict_score(&mut strict_score, material_liveness);
    apply_material_visible_surface_tail_strict_score(
        &mut strict_score,
        material_visible_surface_tail,
    );
    apply_surface_profile_strict_score(
        &mut strict_score,
        &surface_coverage_profile,
        &material_visible_surface_coverage_profile,
    );
    let score = strict_score.score;
    let failure_reasons = strict_checks.failure_reasons.clone();
    Ok(RenderSelectionCaseMetrics {
        render_loss,
        active_surface,
        target_coverage,
        material_visible_target_coverage,
        material_opacity,
        material_liveness,
        surface_coverage_profile,
        material_visible_surface_coverage_profile,
        surface_normal_coverage,
        material_visible_surface_normal_coverage,
        material_visible_surface_tail,
        extent,
        final_active_count: activation.final_active_count,
        newly_activated_fraction: activation.newly_activated_fraction,
        front_local_newly_activated_fraction: front.local_newly_activated_fraction,
        front_liveness,
        extent_front_liveness,
        temporal_front_liveness,
        temporal_extent_front_liveness,
        temporal_activation_schedule_error: temporal_activation_schedule_error(
            &temporal,
            cfg.rollout_steps,
        ),
        temporal_activation_progressive: temporal.progressive_activation,
        temporal_geometry_progressive: temporal.geometry_progressive,
        score,
        failure_reasons,
    })
}
