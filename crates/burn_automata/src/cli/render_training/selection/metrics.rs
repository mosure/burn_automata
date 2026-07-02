use super::super::selection_cases::{
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
    pub(crate) strict_surface_active_count: usize,
    pub(crate) strict_surface_materialized_fraction: f32,
    pub(crate) strict_surface_material_mean_opacity: f32,
    pub(crate) strict_surface_material_visible_margin: f32,
    pub(crate) strict_surface_material_max_visible_margin: f32,
    pub(crate) material_visible_inactive_fraction: f32,
    pub(crate) material_visible_max_inactive_opacity: f32,
    pub(crate) material_active_mean_opacity: f32,
    pub(crate) material_visible_count: usize,
    pub(crate) active_color_state_mean_abs: f32,
    pub(crate) active_color_state_max_abs: f32,
    pub(crate) active_color_state_stddev_mean: f32,
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
    pub(crate) max_dormant_drift_fraction: f32,
    pub(crate) max_dormant_drift: f32,
    pub(crate) all_dormant_drift_bounded: bool,
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
    #[cfg(test)]
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
    pub(crate) dormant_drift_fraction: f32,
    pub(crate) max_dormant_drift: f32,
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
    let mut strict_surface_active_count = usize::MAX;
    let mut strict_surface_materialized_fraction = f32::INFINITY;
    let mut strict_surface_material_mean_opacity = f32::INFINITY;
    let mut strict_surface_material_visible_margin = f32::NEG_INFINITY;
    let mut strict_surface_material_max_visible_margin = f32::NEG_INFINITY;
    let mut material_visible_inactive_fraction = f32::NEG_INFINITY;
    let mut material_visible_max_inactive_opacity = f32::NEG_INFINITY;
    let mut material_active_mean_opacity = f32::INFINITY;
    let mut material_visible_count = usize::MAX;
    let mut active_color_state_mean_abs = f32::INFINITY;
    let mut active_color_state_max_abs = f32::INFINITY;
    let mut active_color_state_stddev_mean = f32::INFINITY;
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
    let mut max_dormant_drift_fraction = f32::NEG_INFINITY;
    let mut max_dormant_drift = f32::NEG_INFINITY;
    let mut all_dormant_drift_bounded = true;
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
        strict_surface_active_count = strict_surface_active_count.min(
            selection_case
                .strict_surface_materialization
                .active_strict_count,
        );
        strict_surface_materialized_fraction = strict_surface_materialized_fraction.min(
            selection_case
                .strict_surface_materialization
                .materialized_fraction,
        );
        strict_surface_material_mean_opacity = strict_surface_material_mean_opacity.min(
            selection_case
                .strict_surface_materialization
                .mean_material_opacity,
        );
        strict_surface_material_visible_margin = strict_surface_material_visible_margin.max(
            selection_case
                .strict_surface_materialization
                .mean_visible_margin,
        );
        strict_surface_material_max_visible_margin = strict_surface_material_max_visible_margin
            .max(
                selection_case
                    .strict_surface_materialization
                    .max_visible_margin,
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
        active_color_state_mean_abs =
            active_color_state_mean_abs.min(selection_case.final_color_state.active_mean_abs);
        active_color_state_max_abs =
            active_color_state_max_abs.min(selection_case.final_color_state.active_max_abs);
        active_color_state_stddev_mean = active_color_state_stddev_mean
            .min(selection_case.final_color_state.active_channel_stddev_mean);
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
        max_dormant_drift_fraction =
            max_dormant_drift_fraction.max(selection_case.dormant_drift.drifting_fraction);
        max_dormant_drift =
            max_dormant_drift.max(selection_case.dormant_drift.max_dormant_displacement);
        all_dormant_drift_bounded &= selection_case.dormant_drift.passed;
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
        strict_surface_active_count: if strict_surface_active_count == usize::MAX {
            0
        } else {
            strict_surface_active_count
        },
        strict_surface_materialized_fraction: finite_report_metric(
            strict_surface_materialized_fraction,
            0.0,
        ),
        strict_surface_material_mean_opacity: finite_report_metric(
            strict_surface_material_mean_opacity,
            f32::NEG_INFINITY,
        ),
        strict_surface_material_visible_margin: finite_report_metric(
            strict_surface_material_visible_margin,
            RENDER_SELECTION_BAD_SCORE,
        ),
        strict_surface_material_max_visible_margin: finite_report_metric(
            strict_surface_material_max_visible_margin,
            RENDER_SELECTION_BAD_SCORE,
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
        active_color_state_mean_abs: finite_report_metric(active_color_state_mean_abs, 0.0),
        active_color_state_max_abs: finite_report_metric(active_color_state_max_abs, 0.0),
        active_color_state_stddev_mean: finite_report_metric(active_color_state_stddev_mean, 0.0),
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
        max_dormant_drift_fraction: finite_report_metric(max_dormant_drift_fraction, 1.0),
        max_dormant_drift: finite_report_metric(max_dormant_drift, RENDER_SELECTION_BAD_SCORE),
        all_dormant_drift_bounded,
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
        #[cfg(test)]
        base_report: base_case.render_loss,
    })
}
