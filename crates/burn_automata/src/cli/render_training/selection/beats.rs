use super::*;

mod checkpoint;
mod color;
mod geometry;
mod liveness;
mod strict_morphology;

use checkpoint::render_selection_checkpoint_morphogenesis_progressed;
use color::render_selection_color_training_progress_beats;
use geometry::{
    render_selection_geometry_growth_precursor_beats,
    render_selection_geometry_training_progress_beats,
};
use liveness::{
    precursor_front_liveness_margin_improved, render_selection_liveness_precursor_beats,
    render_selection_render_within_liveness_precursor_slack,
};
use strict_morphology::render_selection_strict_morphology_progress_beats;

const PRECURSOR_STRICT_SURFACE_MATERIAL_MEAN_PROGRESS: f32 = 0.01;
const PRECURSOR_STRICT_SURFACE_MATERIAL_MARGIN_PROGRESS: f32 = 0.01;
const MATURE_VISIBLE_MATERIAL_COUNT: usize = 8;

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
    if !render_selection_mature_material_temporal_dynamics_ok(selection, best) {
        return false;
    }

    (render_selection_dormant_drift_not_regressed(selection, best)
        && render_selection_candidate_beats(
            selection.score,
            best.score,
            selection.morphology_non_regressed,
            selection.max_render_loss,
            best.max_render_loss,
            selection.min_density_psnr_db,
            best.min_density_psnr_db,
        )
        && render_selection_checkpoint_morphogenesis_progressed(selection, best))
        || render_selection_liveness_precursor_beats(selection, best)
        || render_selection_activation_breakthrough_beats(selection, best)
        || render_selection_material_precursor_beats(selection, best)
        || render_selection_geometry_growth_precursor_beats(selection, best)
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
            selection.max_render_loss,
            best.max_render_loss,
            selection.min_density_psnr_db,
            best.min_density_psnr_db,
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
    const GEOMETRY_TEMPORAL_ERROR_SLACK: f32 = 0.075;
    const GEOMETRY_TEMPORAL_ERROR_MAX: f32 = 0.15;
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

    if !selection.max_render_loss.is_finite()
        || !previous.max_render_loss.is_finite()
        || !selection.score.is_finite()
    {
        return false;
    }
    let render_improved = selection.max_render_loss + RENDER_LOSS_PROGRESS
        <= previous.max_render_loss
        || selection.min_density_psnr_db >= previous.min_density_psnr_db + 0.05;
    let precursor_render_improved = selection.max_render_loss + PRECURSOR_RENDER_LOSS_PROGRESS
        <= previous.max_render_loss
        || selection.min_density_psnr_db
            >= previous.min_density_psnr_db + PRECURSOR_DENSITY_PSNR_PROGRESS_DB;

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
    let strict_surface_material_progress = strict_surface_materialization_progress(
        selection,
        previous,
        PRECURSOR_STRICT_SURFACE_MATERIAL_MEAN_PROGRESS,
        PRECURSOR_STRICT_SURFACE_MATERIAL_MARGIN_PROGRESS,
    );
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
    let material_visible_surface_supported =
        material_visible_surface_support_progressed(selection, previous);
    let material_precursor_improved = precursor_render_improved
        && material_precursor_improved
        && (previous.material_visible_count < MATURE_VISIBLE_MATERIAL_COUNT
            || material_surface_support_progressed(selection, previous));
    let strict_surface_material_supported = material_visible_surface_supported
        || mature_strict_surface_material_support_progressed(selection, previous);
    let strict_surface_material_precursor_improved = strict_surface_material_progress > 0.0
        && strict_surface_material_supported
        && render_selection_render_within_strict_surface_materialization_slack(
            selection.max_render_loss,
            previous.max_render_loss,
            selection.min_density_psnr_db,
            previous.min_density_psnr_db,
            strict_surface_material_progress,
        );
    let precursor_improved = precursor_non_regressed
        && (material_precursor_improved
            || strict_surface_material_precursor_improved
            || (precursor_render_improved && liveness_precursor_improved));
    let geometry_progressed =
        render_selection_geometry_training_progress_beats(selection, previous);
    let color_progressed = render_selection_color_training_progress_beats(selection, previous);
    let strict_morphology_progressed =
        render_selection_strict_morphology_progress_beats(selection, previous);
    if !((render_improved
        && (coverage_improved
            || material_distance_improved
            || extent_improved
            || activation_improved))
        || precursor_improved
        || geometry_progressed
        || color_progressed
        || strict_morphology_progressed)
    {
        return false;
    }

    let temporal_ok = selection.max_temporal_activation_schedule_error.is_finite()
        && previous.max_temporal_activation_schedule_error.is_finite()
        && selection.max_temporal_activation_schedule_error
            <= previous.max_temporal_activation_schedule_error + TEMPORAL_ERROR_SLACK;
    let geometry_temporal_ok = geometry_progressed
        && selection.max_temporal_activation_schedule_error.is_finite()
        && previous.max_temporal_activation_schedule_error.is_finite()
        && selection.max_temporal_activation_schedule_error <= GEOMETRY_TEMPORAL_ERROR_MAX
        && selection.max_temporal_activation_schedule_error
            <= previous.max_temporal_activation_schedule_error + GEOMETRY_TEMPORAL_ERROR_SLACK;
    let surface_ok = selection.active_surface_max.is_finite()
        && previous.active_surface_max.is_finite()
        && selection.active_surface_max
            <= (GROWTH_3D_SURFACE_MAX_DISTANCE + SURFACE_MAX_SLACK)
                .max(previous.active_surface_max + SURFACE_MAX_SLACK);
    let material_tail_ok = selection.material_visible_surface_tail_over_threshold_fraction
        <= previous.material_visible_surface_tail_over_threshold_fraction + MATERIAL_TAIL_SLACK;
    let inactive_material_ok =
        render_selection_inactive_material_not_regressed(selection, previous);
    let local_front_ok = selection.min_front_local_newly_activated_fraction
        >= MIN_LOCAL_FRONT_FRACTION
        || selection.min_front_local_newly_activated_fraction
            >= previous.min_front_local_newly_activated_fraction - 0.02;
    (temporal_ok || geometry_temporal_ok)
        && surface_ok
        && material_tail_ok
        && inactive_material_ok
        && render_selection_mature_material_training_dynamics_ok(
            selection,
            previous,
            activation_improved,
            render_improved,
        )
        && render_selection_dormant_drift_not_regressed(selection, previous)
        && local_front_ok
}

pub(crate) fn render_selection_progress_candidate_preferred(
    candidate: &RenderSelectionMetrics,
    best_progress: &RenderSelectionMetrics,
    no_op: &RenderSelectionMetrics,
) -> bool {
    const STRICT_SURFACE_MATERIAL_PROGRESS_TIEBREAK: f32 = 0.005;
    const STRICT_SURFACE_MATERIAL_SCORE_SLACK: f32 = 0.25;
    const PROGRESS_SCORE_REGRESSION_SLACK: f32 = 2.0;
    const MATERIAL_VISIBLE_COVERAGE_TIEBREAK: f32 = 0.01;
    const MATERIAL_VISIBLE_SUPPORT_REGRESSION_SLACK: f32 = 0.005;

    let candidate_strict_progress = strict_surface_materialization_progress(
        candidate,
        no_op,
        PRECURSOR_STRICT_SURFACE_MATERIAL_MEAN_PROGRESS,
        PRECURSOR_STRICT_SURFACE_MATERIAL_MARGIN_PROGRESS,
    );
    let best_strict_progress = strict_surface_materialization_progress(
        best_progress,
        no_op,
        PRECURSOR_STRICT_SURFACE_MATERIAL_MEAN_PROGRESS,
        PRECURSOR_STRICT_SURFACE_MATERIAL_MARGIN_PROGRESS,
    );
    let material_visible_coverage_progress = render_selection_material_visible_coverage_progress(
        candidate,
        best_progress,
        MATERIAL_VISIBLE_COVERAGE_TIEBREAK,
    );
    if candidate_strict_progress >= best_strict_progress + STRICT_SURFACE_MATERIAL_PROGRESS_TIEBREAK
        && material_visible_coverage_progress
        && candidate.score.is_finite()
        && best_progress.score.is_finite()
        && candidate.score <= best_progress.score + STRICT_SURFACE_MATERIAL_SCORE_SLACK
        && render_selection_inactive_material_not_regressed(candidate, no_op)
        && render_selection_dormant_drift_not_regressed(candidate, no_op)
        && render_selection_render_within_strict_surface_materialization_slack(
            candidate.max_render_loss,
            no_op.max_render_loss,
            candidate.min_density_psnr_db,
            no_op.min_density_psnr_db,
            candidate_strict_progress,
        )
    {
        return true;
    }
    if best_strict_progress >= candidate_strict_progress + STRICT_SURFACE_MATERIAL_PROGRESS_TIEBREAK
    {
        return false;
    }
    if render_selection_material_visible_support_regressed(
        candidate,
        best_progress,
        MATERIAL_VISIBLE_SUPPORT_REGRESSION_SLACK,
    ) {
        return false;
    }

    if candidate.score.is_finite()
        && best_progress.score.is_finite()
        && candidate.score > best_progress.score + PROGRESS_SCORE_REGRESSION_SLACK
        && !material_visible_coverage_progress
    {
        return false;
    }

    candidate.max_render_loss < best_progress.max_render_loss
        || candidate.score < best_progress.score
}

fn render_selection_material_visible_support_regressed(
    candidate: &RenderSelectionMetrics,
    best_progress: &RenderSelectionMetrics,
    slack: f32,
) -> bool {
    let candidate_support = render_selection_material_visible_support(candidate);
    let best_support = render_selection_material_visible_support(best_progress);
    best_support.is_finite()
        && candidate_support.is_finite()
        && best_support > 0.0
        && candidate_support + slack < best_support
}

fn render_selection_material_visible_support(selection: &RenderSelectionMetrics) -> f32 {
    selection
        .material_visible_target_coverage_fraction
        .max(selection.material_visible_surface_covered_bin_fraction)
        .max(selection.material_visible_surface_normal_covered_bin_fraction)
}

fn render_selection_material_visible_coverage_progress(
    candidate: &RenderSelectionMetrics,
    best_progress: &RenderSelectionMetrics,
    min_progress: f32,
) -> bool {
    candidate.material_visible_target_coverage_fraction
        >= best_progress.material_visible_target_coverage_fraction + min_progress
        || candidate.material_visible_surface_covered_bin_fraction
            >= best_progress.material_visible_surface_covered_bin_fraction + min_progress
        || candidate.material_visible_surface_normal_covered_bin_fraction
            >= best_progress.material_visible_surface_normal_covered_bin_fraction + min_progress
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
    )) {
        return false;
    }
    if !render_selection_temporal_activation_not_regressed(selection, best) {
        return false;
    }
    selection.all_temporal_activation_progressive
        && selection.active_surface_max.is_finite()
        && selection.active_surface_max <= GROWTH_3D_SURFACE_MAX_DISTANCE + 0.05
        && render_selection_dormant_drift_not_regressed(selection, best)
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
    if best.material_visible_count >= MATURE_VISIBLE_MATERIAL_COUNT
        && !material_surface_support_progressed(selection, best)
    {
        return false;
    }
    if selection.min_final_active_count < best.min_final_active_count {
        return false;
    }
    if !render_selection_temporal_activation_not_regressed(selection, best) {
        return false;
    }
    if !render_selection_render_within_liveness_precursor_slack(
        selection.max_render_loss,
        best.max_render_loss,
        selection.min_density_psnr_db,
        best.min_density_psnr_db,
    ) {
        return false;
    }
    selection.material_visible_inactive_fraction
        <= best.material_visible_inactive_fraction.max(0.0) + 0.01
        && selection.material_visible_surface_tail_over_threshold_fraction <= 0.01
        && render_selection_dormant_drift_not_regressed(selection, best)
        && selection.active_surface_max.is_finite()
        && selection.active_surface_max <= GROWTH_3D_SURFACE_MAX_DISTANCE + 0.05
}

fn material_surface_support_progressed(
    selection: &RenderSelectionMetrics,
    previous: &RenderSelectionMetrics,
) -> bool {
    const STRICT_MATERIAL_MEAN_PROGRESS: f32 = 0.005;
    const STRICT_MATERIAL_MARGIN_PROGRESS: f32 = 0.005;

    material_visible_surface_support_progressed(selection, previous)
        || strict_surface_materialization_progress(
            selection,
            previous,
            STRICT_MATERIAL_MEAN_PROGRESS,
            STRICT_MATERIAL_MARGIN_PROGRESS,
        ) > 0.0
}

fn mature_strict_surface_material_support_progressed(
    selection: &RenderSelectionMetrics,
    previous: &RenderSelectionMetrics,
) -> bool {
    previous.material_visible_count >= MATURE_VISIBLE_MATERIAL_COUNT
        && selection.material_visible_count >= previous.material_visible_count
        && selection.all_temporal_geometry_progressive
}

fn material_visible_surface_support_progressed(
    selection: &RenderSelectionMetrics,
    previous: &RenderSelectionMetrics,
) -> bool {
    const MATERIAL_TARGET_DISTANCE_PROGRESS: f32 = 5.0e-3;
    const MATERIAL_TARGET_COVERAGE_PROGRESS: f32 = 5.0e-3;
    const MATERIAL_SURFACE_BIN_PROGRESS: f32 = 5.0e-3;

    (previous.material_visible_target_mean_distance.is_finite()
        && selection.material_visible_target_mean_distance.is_finite()
        && selection.material_visible_target_mean_distance + MATERIAL_TARGET_DISTANCE_PROGRESS
            <= previous.material_visible_target_mean_distance)
        || finite_progress(
            selection.material_visible_target_coverage_fraction,
            previous.material_visible_target_coverage_fraction,
            MATERIAL_TARGET_COVERAGE_PROGRESS,
        )
        || finite_progress(
            selection.material_visible_surface_covered_bin_fraction,
            previous.material_visible_surface_covered_bin_fraction,
            MATERIAL_SURFACE_BIN_PROGRESS,
        )
        || finite_progress(
            selection.material_visible_surface_normal_covered_bin_fraction,
            previous.material_visible_surface_normal_covered_bin_fraction,
            MATERIAL_SURFACE_BIN_PROGRESS,
        )
}

fn finite_progress(selection: f32, previous: f32, min_progress: f32) -> bool {
    selection.is_finite() && previous.is_finite() && selection >= previous + min_progress
}

fn strict_surface_materialization_progress(
    selection: &RenderSelectionMetrics,
    previous: &RenderSelectionMetrics,
    mean_opacity_progress: f32,
    margin_progress: f32,
) -> f32 {
    if selection.strict_surface_active_count == 0
        || selection.strict_surface_active_count < previous.strict_surface_active_count
    {
        return 0.0;
    }
    let fraction_progress = if selection.strict_surface_materialized_fraction.is_finite()
        && previous.strict_surface_materialized_fraction.is_finite()
    {
        (selection.strict_surface_materialized_fraction
            - previous.strict_surface_materialized_fraction)
            .max(0.0)
    } else {
        0.0
    };
    let mean_opacity_progress_value = if selection.strict_surface_material_mean_opacity.is_finite()
        && previous.strict_surface_material_mean_opacity.is_finite()
    {
        (selection.strict_surface_material_mean_opacity
            - previous.strict_surface_material_mean_opacity)
            .max(0.0)
    } else {
        0.0
    };
    let margin_progress_value = if selection.strict_surface_material_visible_margin.is_finite()
        && previous.strict_surface_material_visible_margin.is_finite()
    {
        (previous.strict_surface_material_visible_margin
            - selection.strict_surface_material_visible_margin)
            .max(0.0)
    } else {
        0.0
    };

    if fraction_progress <= 0.0
        && mean_opacity_progress_value < mean_opacity_progress
        && margin_progress_value < margin_progress
    {
        return 0.0;
    }
    fraction_progress
        .max(mean_opacity_progress_value)
        .max(margin_progress_value)
}

fn render_selection_inactive_material_not_regressed(
    selection: &RenderSelectionMetrics,
    previous: &RenderSelectionMetrics,
) -> bool {
    const INACTIVE_MATERIAL_FRACTION_SLACK: f32 = 0.01;
    selection.material_visible_inactive_fraction.is_finite()
        && previous.material_visible_inactive_fraction.is_finite()
        && selection.material_visible_inactive_fraction
            <= previous.material_visible_inactive_fraction.max(0.0)
                + INACTIVE_MATERIAL_FRACTION_SLACK
}

fn render_selection_render_within_strict_surface_materialization_slack(
    selection_render_loss: f32,
    previous_render_loss: f32,
    selection_density_psnr_db: f32,
    previous_density_psnr_db: f32,
    material_progress: f32,
) -> bool {
    const RENDER_LOSS_SLACK_ABS: f32 = 1.5e-3;
    const RENDER_LOSS_SLACK_FRACTION: f32 = 2.5e-3;
    const RENDER_LOSS_PROGRESS_SLACK_FRACTION: f32 = 0.04;
    const MAX_RENDER_LOSS_SLACK: f32 = 3.0e-3;
    const DENSITY_PSNR_SLACK_DB: f32 = 0.02;
    const DENSITY_PSNR_PROGRESS_SLACK_DB: f32 = 0.10;
    const MAX_DENSITY_PSNR_SLACK_DB: f32 = 0.03;

    if !selection_render_loss.is_finite()
        || !previous_render_loss.is_finite()
        || !selection_density_psnr_db.is_finite()
        || !previous_density_psnr_db.is_finite()
        || !material_progress.is_finite()
    {
        return false;
    }
    let render_loss_slack = (RENDER_LOSS_SLACK_ABS
        .max(previous_render_loss.abs() * RENDER_LOSS_SLACK_FRACTION)
        + material_progress.max(0.0) * RENDER_LOSS_PROGRESS_SLACK_FRACTION)
        .min(MAX_RENDER_LOSS_SLACK);
    let density_psnr_slack = (DENSITY_PSNR_SLACK_DB
        + material_progress.max(0.0) * DENSITY_PSNR_PROGRESS_SLACK_DB)
        .min(MAX_DENSITY_PSNR_SLACK_DB);
    selection_render_loss <= previous_render_loss + render_loss_slack
        && selection_density_psnr_db + density_psnr_slack >= previous_density_psnr_db
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
    )) {
        return false;
    }
    if !render_selection_temporal_activation_not_regressed(selection, best) {
        return false;
    }
    selection.active_surface_max.is_finite()
        && selection.active_surface_max <= GROWTH_3D_SURFACE_MAX_DISTANCE + 0.05
        && render_selection_dormant_drift_not_regressed(selection, best)
        && selection.material_visible_inactive_fraction
            <= best.material_visible_inactive_fraction + 0.01
        && selection.material_visible_surface_tail_over_threshold_fraction <= 0.01
}

fn render_selection_mature_material_temporal_dynamics_ok(
    selection: &RenderSelectionMetrics,
    previous: &RenderSelectionMetrics,
) -> bool {
    let mature_material = selection.material_visible_count >= MATURE_VISIBLE_MATERIAL_COUNT
        || previous.material_visible_count >= MATURE_VISIBLE_MATERIAL_COUNT;
    !mature_material
        || (selection.all_temporal_geometry_progressive
            && render_selection_temporal_activation_not_regressed(selection, previous))
}

fn render_selection_mature_material_training_dynamics_ok(
    selection: &RenderSelectionMetrics,
    previous: &RenderSelectionMetrics,
    activation_improved: bool,
    render_improved: bool,
) -> bool {
    if render_selection_mature_material_temporal_dynamics_ok(selection, previous) {
        return true;
    }
    let strict_score_improvement = previous.score - selection.score;
    let temporal_error_improved = selection.max_temporal_activation_schedule_error.is_finite()
        && previous.max_temporal_activation_schedule_error.is_finite()
        && selection.max_temporal_activation_schedule_error + 0.005
            <= previous.max_temporal_activation_schedule_error;

    activation_improved
        && render_improved
        && strict_score_improvement >= 0.25
        && temporal_error_improved
        && render_selection_temporal_activation_not_regressed(selection, previous)
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

pub(super) fn render_selection_dormant_drift_not_regressed(
    selection: &RenderSelectionMetrics,
    previous: &RenderSelectionMetrics,
) -> bool {
    const DRIFT_FRACTION_SLACK: f32 = 0.005;
    const DRIFT_DISTANCE_SLACK: f32 = 0.02;

    selection.all_dormant_drift_bounded
        && selection.max_dormant_drift_fraction.is_finite()
        && previous.max_dormant_drift_fraction.is_finite()
        && selection.max_dormant_drift.is_finite()
        && previous.max_dormant_drift.is_finite()
        && selection.max_dormant_drift_fraction
            <= previous.max_dormant_drift_fraction.max(0.0) + DRIFT_FRACTION_SLACK
        && selection.max_dormant_drift <= previous.max_dormant_drift.max(0.0) + DRIFT_DISTANCE_SLACK
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
