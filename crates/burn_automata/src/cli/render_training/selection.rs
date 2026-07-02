use super::selection_cases::{
    render_selection_case_metrics, render_selection_case_score_with_baseline,
};
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
    const PRECURSOR_RENDER_LOSS_PROGRESS: f32 = 5.0e-4;
    const PRECURSOR_DENSITY_PSNR_PROGRESS_DB: f32 = 0.01;
    const PRECURSOR_MATERIAL_MEAN_PROGRESS: f32 = 0.02;
    const PRECURSOR_MATERIAL_DISTANCE_PROGRESS: f32 = 1.0e-3;
    const PRECURSOR_FRONT_MARGIN_PROGRESS: f32 = 0.02;
    const PRECURSOR_COVERAGE_REGRESSION_SLACK: f32 = 0.005;
    const PRECURSOR_SURFACE_BIN_REGRESSION_SLACK: f32 = 0.001;

    if !selection.render_loss.is_finite()
        || !previous.render_loss.is_finite()
        || !selection.score.is_finite()
    {
        return false;
    }
    let render_improved = selection.render_loss + RENDER_LOSS_PROGRESS <= previous.render_loss
        || selection.density_psnr_db >= previous.density_psnr_db + 0.05;
    let precursor_render_improved = selection.render_loss + PRECURSOR_RENDER_LOSS_PROGRESS
        <= previous.render_loss
        || selection.density_psnr_db
            >= previous.density_psnr_db + PRECURSOR_DENSITY_PSNR_PROGRESS_DB;

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
    let material_precursor_improved = selection.material_visible_count
        > previous.material_visible_count
        || (selection.material_active_mean_opacity.is_finite()
            && previous.material_active_mean_opacity.is_finite()
            && selection.material_active_mean_opacity
                >= previous.material_active_mean_opacity + PRECURSOR_MATERIAL_MEAN_PROGRESS)
        || (previous.material_visible_target_mean_distance.is_finite()
            && selection.material_visible_target_mean_distance.is_finite()
            && selection.material_visible_target_mean_distance
                + PRECURSOR_MATERIAL_DISTANCE_PROGRESS
                <= previous.material_visible_target_mean_distance);
    let liveness_precursor_improved = precursor_front_liveness_margin_improved(
        selection.max_front_liveness_margin,
        previous.max_front_liveness_margin,
        selection.min_front_liveness_candidate_count,
        PRECURSOR_FRONT_MARGIN_PROGRESS,
    ) || precursor_front_liveness_margin_improved(
        selection.max_temporal_front_liveness_margin,
        previous.max_temporal_front_liveness_margin,
        selection.min_temporal_front_liveness_candidate_count,
        PRECURSOR_FRONT_MARGIN_PROGRESS,
    ) || precursor_front_liveness_margin_improved(
        selection.max_extent_front_liveness_margin,
        previous.max_extent_front_liveness_margin,
        selection.min_extent_front_liveness_candidate_count,
        PRECURSOR_FRONT_MARGIN_PROGRESS,
    ) || precursor_front_liveness_margin_improved(
        selection.max_temporal_extent_front_liveness_margin,
        previous.max_temporal_extent_front_liveness_margin,
        selection.min_temporal_extent_front_liveness_candidate_count,
        PRECURSOR_FRONT_MARGIN_PROGRESS,
    );
    let precursor_non_regressed = selection.target_coverage_fraction
        + PRECURSOR_COVERAGE_REGRESSION_SLACK
        >= previous.target_coverage_fraction
        && selection.material_visible_target_coverage_fraction
            + PRECURSOR_COVERAGE_REGRESSION_SLACK
            >= previous.material_visible_target_coverage_fraction
        && selection.surface_covered_bin_fraction + PRECURSOR_SURFACE_BIN_REGRESSION_SLACK
            >= previous.surface_covered_bin_fraction
        && selection.material_visible_surface_covered_bin_fraction
            + PRECURSOR_SURFACE_BIN_REGRESSION_SLACK
            >= previous.material_visible_surface_covered_bin_fraction;
    let precursor_improved = precursor_render_improved
        && precursor_non_regressed
        && (material_precursor_improved || liveness_precursor_improved);
    if !((render_improved
        && (coverage_improved
            || material_distance_improved
            || extent_improved
            || activation_improved))
        || precursor_improved)
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

fn precursor_front_liveness_margin_improved(
    selection_margin: f32,
    previous_margin: f32,
    candidate_count: usize,
    min_improvement: f32,
) -> bool {
    candidate_count > 0
        && selection_margin.is_finite()
        && previous_margin.is_finite()
        && selection_margin + min_improvement <= previous_margin
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

pub(crate) fn material_training_frontier_coverage_threshold(seed_scale: f32) -> f32 {
    let soft = material_training_soft_coverage_threshold(seed_scale);
    soft.max(seed_scale.max(soft) * 1.25).max(soft)
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

pub(crate) fn frontier_material_assignment_weight(
    distance: f32,
    strict_threshold: f32,
    soft_threshold: f32,
    frontier_threshold: f32,
) -> f32 {
    let soft_weight = soft_material_assignment_weight(distance, strict_threshold, soft_threshold);
    if soft_weight > 0.0 || !distance.is_finite() {
        return soft_weight;
    }
    let soft = soft_threshold.max(strict_threshold.max(1.0e-6));
    let frontier = frontier_threshold.max(soft);
    if distance >= frontier {
        return 0.0;
    }
    let falloff = 1.0 - (distance - soft) / (frontier - soft).max(1.0e-6);
    0.25 * falloff.clamp(0.0, 1.0).powi(2)
}
