use crate::cli::prelude::GROWTH_3D_SURFACE_MAX_DISTANCE;
use crate::cli::render_training::selection::RenderSelectionMetrics;

use super::render_selection_temporal_activation_not_regressed;

pub(super) fn precursor_front_liveness_margin_improved(
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

pub(super) fn render_selection_liveness_precursor_beats(
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

    let front_margin_improvement = finite_margin_improvement(
        selection.max_front_liveness_margin,
        best.max_front_liveness_margin,
    );
    let temporal_front_margin_improvement = finite_margin_improvement(
        selection.max_temporal_front_liveness_margin,
        best.max_temporal_front_liveness_margin,
    );
    let temporal_extent_front_margin_improvement = finite_margin_improvement(
        selection.max_temporal_extent_front_liveness_margin,
        best.max_temporal_extent_front_liveness_margin,
    );
    let extent_front_margin_improvement = finite_margin_improvement(
        selection.max_extent_front_liveness_margin,
        best.max_extent_front_liveness_margin,
    );
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

pub(super) fn render_selection_render_within_liveness_precursor_slack(
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

fn finite_margin_improvement(selection_margin: f32, best_margin: f32) -> f32 {
    if selection_margin.is_finite() && best_margin.is_finite() {
        best_margin - selection_margin
    } else {
        0.0
    }
}

fn render_selection_bounded_temporal_front_precursor(
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
