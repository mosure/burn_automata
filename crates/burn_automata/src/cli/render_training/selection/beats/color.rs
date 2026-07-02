use crate::cli::prelude::GROWTH_3D_SURFACE_MAX_DISTANCE;
use crate::cli::render_training::selection::RenderSelectionMetrics;

use super::render_selection_dormant_drift_not_regressed;

pub(super) fn render_selection_color_training_progress_beats(
    selection: &RenderSelectionMetrics,
    previous: &RenderSelectionMetrics,
) -> bool {
    const MEAN_PROGRESS: f32 = 0.002;
    const MAX_PROGRESS: f32 = 0.002;
    const STDDEV_PROGRESS: f32 = 0.001;

    render_selection_color_state_progressed(
        selection,
        previous,
        MEAN_PROGRESS,
        MAX_PROGRESS,
        STDDEV_PROGRESS,
    ) && render_selection_color_progress_safe(selection, previous)
}

fn render_selection_color_state_progressed(
    selection: &RenderSelectionMetrics,
    previous: &RenderSelectionMetrics,
    mean_progress: f32,
    max_progress: f32,
    stddev_progress: f32,
) -> bool {
    finite_progress(
        selection.active_color_state_mean_abs,
        previous.active_color_state_mean_abs,
        mean_progress,
    ) || finite_progress(
        selection.active_color_state_max_abs,
        previous.active_color_state_max_abs,
        max_progress,
    ) || finite_progress(
        selection.active_color_state_stddev_mean,
        previous.active_color_state_stddev_mean,
        stddev_progress,
    )
}

fn render_selection_color_progress_safe(
    selection: &RenderSelectionMetrics,
    previous: &RenderSelectionMetrics,
) -> bool {
    const RENDER_LOSS_SLACK_ABS: f32 = 0.002;
    const RENDER_LOSS_SLACK_FRACTION: f32 = 0.003;
    const DENSITY_PSNR_SLACK_DB: f32 = 0.03;
    const TARGET_COVERAGE_REGRESSION_SLACK: f32 = 0.005;
    const SURFACE_BIN_REGRESSION_SLACK: f32 = 0.005;
    const SURFACE_MAX_SLACK: f32 = 0.15;
    const INACTIVE_MATERIAL_SLACK: f32 = 0.01;
    const MATERIAL_TAIL_SLACK: f32 = 0.01;

    render_within_color_progress_slack(
        selection.render_loss,
        previous.render_loss,
        selection.density_psnr_db,
        previous.density_psnr_db,
        RENDER_LOSS_SLACK_ABS,
        RENDER_LOSS_SLACK_FRACTION,
        DENSITY_PSNR_SLACK_DB,
    ) && selection.target_coverage_fraction + TARGET_COVERAGE_REGRESSION_SLACK
        >= previous.target_coverage_fraction
        && selection.material_visible_target_coverage_fraction + TARGET_COVERAGE_REGRESSION_SLACK
            >= previous.material_visible_target_coverage_fraction
        && selection.surface_covered_bin_fraction + SURFACE_BIN_REGRESSION_SLACK
            >= previous.surface_covered_bin_fraction
        && selection.material_visible_surface_covered_bin_fraction + SURFACE_BIN_REGRESSION_SLACK
            >= previous.material_visible_surface_covered_bin_fraction
        && selection.active_surface_max.is_finite()
        && previous.active_surface_max.is_finite()
        && selection.active_surface_max
            <= (GROWTH_3D_SURFACE_MAX_DISTANCE + SURFACE_MAX_SLACK)
                .max(previous.active_surface_max + SURFACE_MAX_SLACK)
        && selection.material_visible_inactive_fraction
            <= previous.material_visible_inactive_fraction + INACTIVE_MATERIAL_SLACK
        && selection.material_visible_surface_tail_over_threshold_fraction
            <= previous.material_visible_surface_tail_over_threshold_fraction + MATERIAL_TAIL_SLACK
        && render_selection_dormant_drift_not_regressed(selection, previous)
        && selection.min_final_active_count >= previous.min_final_active_count.saturating_sub(1)
}

fn finite_progress(selection: f32, previous: f32, min_progress: f32) -> bool {
    selection.is_finite() && previous.is_finite() && selection >= previous + min_progress
}

fn render_within_color_progress_slack(
    selection_render_loss: f32,
    previous_render_loss: f32,
    selection_density_psnr_db: f32,
    previous_density_psnr_db: f32,
    loss_slack_abs: f32,
    loss_slack_fraction: f32,
    density_psnr_slack_db: f32,
) -> bool {
    if !selection_render_loss.is_finite()
        || !previous_render_loss.is_finite()
        || !selection_density_psnr_db.is_finite()
        || !previous_density_psnr_db.is_finite()
    {
        return false;
    }
    let render_loss_slack = loss_slack_abs.max(previous_render_loss.abs() * loss_slack_fraction);
    selection_render_loss <= previous_render_loss + render_loss_slack
        && selection_density_psnr_db + density_psnr_slack_db >= previous_density_psnr_db
}
