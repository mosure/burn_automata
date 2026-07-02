use crate::cli::render_training::selection::RenderSelectionMetrics;

use super::material_surface_support_progressed;

pub(super) fn render_selection_checkpoint_morphogenesis_progressed(
    selection: &RenderSelectionMetrics,
    previous: &RenderSelectionMetrics,
) -> bool {
    const COVERAGE_PROGRESS: f32 = 5.0e-3;
    const MATERIAL_DISTANCE_PROGRESS: f32 = 5.0e-3;
    const EXTENT_PROGRESS: f32 = 1.0e-2;
    const ACTIVATION_PROGRESS: f32 = 1.0e-4;
    const TEMPORAL_ERROR_PROGRESS: f32 = 1.0e-2;

    let coverage_progressed = finite_progress(
        selection.target_coverage_fraction,
        previous.target_coverage_fraction,
        COVERAGE_PROGRESS,
    ) || finite_progress(
        selection.surface_covered_bin_fraction,
        previous.surface_covered_bin_fraction,
        COVERAGE_PROGRESS,
    ) || finite_progress(
        selection.surface_normal_covered_bin_fraction,
        previous.surface_normal_covered_bin_fraction,
        COVERAGE_PROGRESS,
    ) || material_surface_support_progressed(selection, previous);
    let material_distance_progressed = previous.material_visible_target_mean_distance.is_finite()
        && selection.material_visible_target_mean_distance.is_finite()
        && selection.material_visible_target_mean_distance + MATERIAL_DISTANCE_PROGRESS
            <= previous.material_visible_target_mean_distance;
    let extent_progressed = finite_progress(
        selection.min_active_extent_bbox_ratio,
        previous.min_active_extent_bbox_ratio,
        EXTENT_PROGRESS,
    ) || finite_progress(
        selection.min_active_extent_min_axis_ratio,
        previous.min_active_extent_min_axis_ratio,
        EXTENT_PROGRESS,
    );
    let activation_progressed = selection.min_final_active_count > previous.min_final_active_count
        && finite_progress(
            selection.min_newly_activated_fraction,
            previous.min_newly_activated_fraction,
            ACTIVATION_PROGRESS,
        )
        && finite_progress(
            selection.min_front_local_newly_activated_fraction,
            previous.min_front_local_newly_activated_fraction,
            ACTIVATION_PROGRESS,
        );
    let temporal_schedule_progressed = previous.max_temporal_activation_schedule_error.is_finite()
        && selection.max_temporal_activation_schedule_error.is_finite()
        && selection.max_temporal_activation_schedule_error + TEMPORAL_ERROR_PROGRESS
            <= previous.max_temporal_activation_schedule_error;

    coverage_progressed
        || material_distance_progressed
        || extent_progressed
        || activation_progressed
        || temporal_schedule_progressed
}

fn finite_progress(selection: f32, previous: f32, min_progress: f32) -> bool {
    selection.is_finite() && previous.is_finite() && selection >= previous + min_progress
}
