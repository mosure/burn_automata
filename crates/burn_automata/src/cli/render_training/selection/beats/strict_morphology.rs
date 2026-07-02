use crate::cli::prelude::GROWTH_3D_SURFACE_MAX_DISTANCE;
use crate::cli::render_training::selection::RenderSelectionMetrics;

use super::{
    render_selection_dormant_drift_not_regressed, render_selection_inactive_material_not_regressed,
    render_selection_render_within_strict_improvement_slack,
};

pub(super) fn render_selection_strict_morphology_progress_beats(
    selection: &RenderSelectionMetrics,
    previous: &RenderSelectionMetrics,
) -> bool {
    const MIN_STRICT_SCORE_IMPROVEMENT: f32 = 0.25;
    const TARGET_COVERAGE_PROGRESS: f32 = 0.02;
    const SURFACE_BIN_PROGRESS: f32 = 0.02;
    const NORMAL_BIN_PROGRESS: f32 = 0.04;
    const TEMPORAL_ERROR_PROGRESS: f32 = 0.03;
    const SURFACE_MAX_SLACK: f32 = 0.30;
    const MATERIAL_TAIL_SLACK: f32 = 0.01;
    const MIN_LOCAL_FRONT_FRACTION: f32 = 0.50;

    if !selection.score.is_finite() || !previous.score.is_finite() {
        return false;
    }
    let strict_score_improvement = previous.score - selection.score;
    if strict_score_improvement < MIN_STRICT_SCORE_IMPROVEMENT {
        return false;
    }
    if !render_selection_render_within_strict_improvement_slack(
        strict_score_improvement,
        selection.render_loss,
        previous.render_loss,
        selection.density_psnr_db,
        previous.density_psnr_db,
    ) {
        return false;
    }

    let coverage_progressed = finite_progress(
        selection.target_coverage_fraction,
        previous.target_coverage_fraction,
        TARGET_COVERAGE_PROGRESS,
    ) || finite_progress(
        selection.material_visible_target_coverage_fraction,
        previous.material_visible_target_coverage_fraction,
        TARGET_COVERAGE_PROGRESS,
    ) || finite_progress(
        selection.surface_covered_bin_fraction,
        previous.surface_covered_bin_fraction,
        SURFACE_BIN_PROGRESS,
    ) || finite_progress(
        selection.material_visible_surface_covered_bin_fraction,
        previous.material_visible_surface_covered_bin_fraction,
        SURFACE_BIN_PROGRESS,
    ) || finite_progress(
        selection.surface_normal_covered_bin_fraction,
        previous.surface_normal_covered_bin_fraction,
        NORMAL_BIN_PROGRESS,
    ) || finite_progress(
        selection.material_visible_surface_normal_covered_bin_fraction,
        previous.material_visible_surface_normal_covered_bin_fraction,
        NORMAL_BIN_PROGRESS,
    );
    let temporal_progressed =
        temporal_schedule_error_improved(selection, previous, TEMPORAL_ERROR_PROGRESS)
            || (!previous.all_temporal_activation_progressive
                && selection.all_temporal_activation_progressive)
            || (!previous.all_temporal_geometry_progressive
                && selection.all_temporal_geometry_progressive);
    if !coverage_progressed && !temporal_progressed {
        return false;
    }

    coverage_not_regressed(
        selection.target_coverage_fraction,
        previous.target_coverage_fraction,
    ) && coverage_not_regressed(
        selection.material_visible_target_coverage_fraction,
        previous.material_visible_target_coverage_fraction,
    ) && coverage_not_regressed(
        selection.surface_covered_bin_fraction,
        previous.surface_covered_bin_fraction,
    ) && coverage_not_regressed(
        selection.material_visible_surface_covered_bin_fraction,
        previous.material_visible_surface_covered_bin_fraction,
    ) && coverage_not_regressed(
        selection.surface_normal_covered_bin_fraction,
        previous.surface_normal_covered_bin_fraction,
    ) && coverage_not_regressed(
        selection.material_visible_surface_normal_covered_bin_fraction,
        previous.material_visible_surface_normal_covered_bin_fraction,
    ) && selection.active_surface_max.is_finite()
        && previous.active_surface_max.is_finite()
        && selection.active_surface_max
            <= (GROWTH_3D_SURFACE_MAX_DISTANCE + SURFACE_MAX_SLACK)
                .max(previous.active_surface_max + SURFACE_MAX_SLACK)
        && selection.material_visible_surface_tail_over_threshold_fraction
            <= previous.material_visible_surface_tail_over_threshold_fraction + MATERIAL_TAIL_SLACK
        && render_selection_dormant_drift_not_regressed(selection, previous)
        && render_selection_inactive_material_not_regressed(selection, previous)
        && (selection.min_front_local_newly_activated_fraction >= MIN_LOCAL_FRONT_FRACTION
            || selection.min_front_local_newly_activated_fraction
                >= previous.min_front_local_newly_activated_fraction - 0.02)
        && selection.min_newly_activated_fraction + 0.02 >= previous.min_newly_activated_fraction
}

fn finite_progress(selection: f32, previous: f32, min_progress: f32) -> bool {
    selection.is_finite() && previous.is_finite() && selection >= previous + min_progress
}

fn temporal_schedule_error_improved(
    selection: &RenderSelectionMetrics,
    previous: &RenderSelectionMetrics,
    min_progress: f32,
) -> bool {
    selection.max_temporal_activation_schedule_error.is_finite()
        && previous.max_temporal_activation_schedule_error.is_finite()
        && selection.max_temporal_activation_schedule_error + min_progress
            <= previous.max_temporal_activation_schedule_error
}

fn coverage_not_regressed(selection: f32, previous: f32) -> bool {
    const COVERAGE_REGRESSION_SLACK: f32 = 0.005;
    selection.is_finite()
        && previous.is_finite()
        && selection + COVERAGE_REGRESSION_SLACK >= previous
}
