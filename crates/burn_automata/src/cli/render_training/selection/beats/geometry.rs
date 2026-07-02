use crate::cli::prelude::GROWTH_3D_SURFACE_MAX_DISTANCE;
use crate::cli::render_training::selection::RenderSelectionMetrics;

use super::{
    render_selection_dormant_drift_not_regressed, render_selection_render_non_regressed,
    render_selection_render_within_strict_improvement_slack,
};

pub(super) fn render_selection_geometry_growth_precursor_beats(
    selection: &RenderSelectionMetrics,
    best: &RenderSelectionMetrics,
) -> bool {
    const MIN_SCORE_IMPROVEMENT: f32 = 1.0;
    const TARGET_COVERAGE_REGRESSION_SLACK: f32 = 0.005;
    const EXTENT_PROGRESS: f32 = 0.05;
    const SURFACE_BIN_PROGRESS: f32 = 0.02;
    const NORMAL_BIN_PROGRESS: f32 = 0.05;
    const ACTIVATION_PROGRESS: f32 = 0.05;

    let strict_score_improvement = best.score - selection.score;
    strict_score_improvement >= MIN_SCORE_IMPROVEMENT
        && selection.score.is_finite()
        && best.score.is_finite()
        && (render_selection_render_non_regressed(
            selection.max_render_loss,
            best.max_render_loss,
            selection.min_density_psnr_db,
            best.min_density_psnr_db,
        ) || render_selection_render_within_strict_improvement_slack(
            strict_score_improvement,
            selection.max_render_loss,
            best.max_render_loss,
            selection.min_density_psnr_db,
            best.min_density_psnr_db,
        ))
        && selection.target_coverage_fraction + TARGET_COVERAGE_REGRESSION_SLACK
            >= best.target_coverage_fraction
        && render_selection_geometry_growth_progressed(
            selection,
            best,
            EXTENT_PROGRESS,
            SURFACE_BIN_PROGRESS,
            NORMAL_BIN_PROGRESS,
            ACTIVATION_PROGRESS,
        )
        && render_selection_geometry_growth_safe(selection, best)
}

pub(super) fn render_selection_geometry_training_progress_beats(
    selection: &RenderSelectionMetrics,
    previous: &RenderSelectionMetrics,
) -> bool {
    geometry_growth_improved(selection, previous)
        || geometry_expansion_continuation(selection, previous)
}

fn geometry_growth_improved(
    selection: &RenderSelectionMetrics,
    previous: &RenderSelectionMetrics,
) -> bool {
    const SCORE_IMPROVEMENT: f32 = 1.0;
    const TARGET_REGRESSION_SLACK: f32 = 0.005;
    const EXTENT_PROGRESS: f32 = 0.05;
    const SURFACE_BIN_PROGRESS: f32 = 0.02;
    const NORMAL_BIN_PROGRESS: f32 = 0.05;
    const ACTIVATION_PROGRESS: f32 = 0.05;

    previous.score - selection.score >= SCORE_IMPROVEMENT
        && selection.target_coverage_fraction + TARGET_REGRESSION_SLACK
            >= previous.target_coverage_fraction
        && render_selection_geometry_growth_progressed(
            selection,
            previous,
            EXTENT_PROGRESS,
            SURFACE_BIN_PROGRESS,
            NORMAL_BIN_PROGRESS,
            ACTIVATION_PROGRESS,
        )
        && render_selection_geometry_growth_safe(selection, previous)
}

fn geometry_expansion_continuation(
    selection: &RenderSelectionMetrics,
    previous: &RenderSelectionMetrics,
) -> bool {
    const SCORE_REGRESSION_SLACK: f32 = 10.0;
    const RENDER_LOSS_SLACK_ABS: f32 = 0.015;
    const RENDER_LOSS_SLACK_FRACTION: f32 = 0.02;
    const DENSITY_PSNR_SLACK_DB: f32 = 0.10;
    const TARGET_REGRESSION_SLACK: f32 = 0.005;
    const EXTENT_PROGRESS: f32 = 0.12;
    const SURFACE_BIN_PROGRESS: f32 = 0.015;
    const NORMAL_BIN_PROGRESS: f32 = 0.05;
    const ACTIVATION_PROGRESS: f32 = 0.10;
    const MIN_LOCAL_FRONT_FRACTION: f32 = 0.65;

    selection.score <= previous.score + SCORE_REGRESSION_SLACK
        && render_selection_render_within_geometry_expansion_slack(
            selection.max_render_loss,
            previous.max_render_loss,
            selection.min_density_psnr_db,
            previous.min_density_psnr_db,
            RENDER_LOSS_SLACK_ABS,
            RENDER_LOSS_SLACK_FRACTION,
            DENSITY_PSNR_SLACK_DB,
        )
        && selection.target_coverage_fraction + TARGET_REGRESSION_SLACK
            >= previous.target_coverage_fraction
        && selection.min_front_local_newly_activated_fraction >= MIN_LOCAL_FRONT_FRACTION
        && render_selection_geometry_expansion_progressed(
            selection,
            previous,
            EXTENT_PROGRESS,
            SURFACE_BIN_PROGRESS,
            NORMAL_BIN_PROGRESS,
            ACTIVATION_PROGRESS,
        )
        && render_selection_geometry_growth_safe(selection, previous)
}

fn render_selection_geometry_growth_progressed(
    selection: &RenderSelectionMetrics,
    previous: &RenderSelectionMetrics,
    extent_progress: f32,
    surface_bin_progress: f32,
    normal_bin_progress: f32,
    activation_progress: f32,
) -> bool {
    let extent_improved = finite_progress(
        selection.min_active_extent_bbox_ratio,
        previous.min_active_extent_bbox_ratio,
        extent_progress,
    ) || finite_progress(
        selection.min_active_extent_min_axis_ratio,
        previous.min_active_extent_min_axis_ratio,
        extent_progress,
    );
    let surface_improved = finite_progress(
        selection.surface_covered_bin_fraction,
        previous.surface_covered_bin_fraction,
        surface_bin_progress,
    ) || finite_progress(
        selection.surface_normal_covered_bin_fraction,
        previous.surface_normal_covered_bin_fraction,
        normal_bin_progress,
    );
    let activation_improved = selection.min_final_active_count > previous.min_final_active_count
        && finite_progress(
            selection.min_newly_activated_fraction,
            previous.min_newly_activated_fraction,
            activation_progress,
        );

    extent_improved && (surface_improved || activation_improved)
}

fn render_selection_geometry_expansion_progressed(
    selection: &RenderSelectionMetrics,
    previous: &RenderSelectionMetrics,
    extent_progress: f32,
    surface_bin_progress: f32,
    normal_bin_progress: f32,
    activation_progress: f32,
) -> bool {
    let extent_improved = finite_progress(
        selection.min_active_extent_bbox_ratio,
        previous.min_active_extent_bbox_ratio,
        extent_progress,
    ) || finite_progress(
        selection.min_active_extent_min_axis_ratio,
        previous.min_active_extent_min_axis_ratio,
        extent_progress,
    );
    let surface_support_improved = finite_progress(
        selection.surface_covered_bin_fraction,
        previous.surface_covered_bin_fraction,
        surface_bin_progress,
    ) || finite_progress(
        selection.surface_normal_covered_bin_fraction,
        previous.surface_normal_covered_bin_fraction,
        normal_bin_progress,
    ) || finite_progress(
        selection.target_coverage_fraction,
        previous.target_coverage_fraction,
        surface_bin_progress,
    );
    let activation_improved = selection.min_final_active_count > previous.min_final_active_count
        && finite_progress(
            selection.min_newly_activated_fraction,
            previous.min_newly_activated_fraction,
            activation_progress,
        );

    extent_improved && surface_support_improved && activation_improved
}

fn render_selection_geometry_growth_safe(
    selection: &RenderSelectionMetrics,
    previous: &RenderSelectionMetrics,
) -> bool {
    const INACTIVE_MATERIAL_SLACK: f32 = 0.01;
    const MATERIAL_TAIL_MAX: f32 = 0.01;
    const SURFACE_MAX_SLACK: f32 = 0.30;

    selection.active_surface_max.is_finite()
        && previous.active_surface_max.is_finite()
        && selection.active_surface_max
            <= (GROWTH_3D_SURFACE_MAX_DISTANCE + SURFACE_MAX_SLACK)
                .max(previous.active_surface_max + SURFACE_MAX_SLACK)
        && selection.material_visible_inactive_fraction
            <= previous.material_visible_inactive_fraction + INACTIVE_MATERIAL_SLACK
        && selection.material_visible_surface_tail_over_threshold_fraction <= MATERIAL_TAIL_MAX
        && render_selection_dormant_drift_not_regressed(selection, previous)
}

fn finite_progress(selection: f32, previous: f32, min_progress: f32) -> bool {
    selection.is_finite() && previous.is_finite() && selection >= previous + min_progress
}

fn render_selection_render_within_geometry_expansion_slack(
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
