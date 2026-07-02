use super::super::selection::{RenderSelectionBaselineCase, finite_report_metric};
use super::super::*;
use super::metrics::RenderSelectionCaseMetrics;

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
