use super::*;

#[test]
fn render_selection_training_progress_can_continue_non_promotable_refinement() {
    let mut previous = render_selection_metrics_with_liveness(102.6, 0.838, 0.35, 1.5105798);
    previous.active_surface_max = 0.43;
    previous.min_active_extent_bbox_ratio = 0.35;
    previous.min_active_extent_min_axis_ratio = 0.22;
    previous.min_final_active_count = 54;
    previous.min_newly_activated_fraction = 0.18;
    previous.min_front_local_newly_activated_fraction = 0.92;
    previous.max_temporal_activation_schedule_error = 0.12;
    previous.surface_covered_bin_fraction = 0.06;
    previous.surface_normal_covered_bin_fraction = 0.23;
    previous.material_visible_surface_covered_bin_fraction = 0.06;
    previous.material_visible_surface_normal_covered_bin_fraction = 0.23;
    previous.material_visible_target_mean_distance = 0.86;

    let mut continued = previous.clone();
    continued.morphology_non_regressed = false;
    continued.score = 110.5;
    set_render_selection_metrics_render(&mut continued, 0.790, 0.39);
    continued.active_surface_max = 0.61;
    continued.min_active_extent_bbox_ratio = 0.95;
    continued.min_active_extent_min_axis_ratio = 0.88;
    continued.min_final_active_count = 205;
    continued.min_newly_activated_fraction = 0.79;
    continued.min_front_local_newly_activated_fraction = 0.62;
    continued.max_temporal_activation_schedule_error = 0.138;
    continued.surface_covered_bin_fraction = 0.31;
    continued.surface_normal_covered_bin_fraction = 0.54;
    continued.material_visible_surface_covered_bin_fraction = 0.25;
    continued.material_visible_surface_normal_covered_bin_fraction = 0.50;
    continued.material_visible_target_mean_distance = 0.89;

    assert!(
        !render_selection_candidate_metrics_beats(&continued, &previous),
        "continuation should not be promoted as a strict selected checkpoint"
    );
    assert!(
        render_selection_training_progress_beats(&continued, &previous),
        "bounded render, coverage, extent, and activation progress should continue training even before strict gates pass"
    );

    let mut dormant_drift = continued.clone();
    mark_selection_dormant_drift_unbounded(&mut dormant_drift);
    assert!(
        !render_selection_training_progress_beats(&dormant_drift, &previous),
        "training continuation must not use far dormant-particle drift as a progress mechanism"
    );

    let mut bursty = continued.clone();
    bursty.active_surface_max = 0.90;
    bursty.max_temporal_activation_schedule_error = 0.27;
    bursty.material_visible_surface_tail_over_threshold_fraction = 0.125;
    bursty.min_front_local_newly_activated_fraction = 0.48;
    assert!(
        !render_selection_training_progress_beats(&bursty, &previous),
        "training continuation must still reject global activation/projection shortcuts"
    );
}

#[test]
fn render_selection_training_progress_retains_mature_material_activation_breakthrough() {
    let mut previous = render_selection_metrics_with_liveness(212.04, 0.5817, 2.476, 0.0);
    previous.material_visible_count = 21;
    previous.min_final_active_count = 21;
    previous.min_newly_activated_fraction = 0.232;
    previous.min_front_local_newly_activated_fraction = 1.0;
    previous.max_temporal_activation_schedule_error = 0.071;
    previous.all_temporal_activation_progressive = false;
    previous.all_temporal_geometry_progressive = false;
    previous.active_surface_max = 0.15;
    previous.target_coverage_fraction = 0.009;
    previous.material_visible_target_coverage_fraction = 0.005;
    previous.surface_covered_bin_fraction = 0.046;
    previous.material_visible_surface_covered_bin_fraction = 0.046;
    previous.surface_normal_covered_bin_fraction = 0.076;
    previous.material_visible_surface_normal_covered_bin_fraction = 0.076;

    let mut continued = previous.clone();
    continued.score = 200.83;
    set_render_selection_metrics_render(&mut continued, 0.536, 2.840);
    continued.material_visible_count = 24;
    continued.min_final_active_count = 24;
    continued.min_newly_activated_fraction = 0.286;
    continued.max_temporal_activation_schedule_error = 0.062;
    continued.target_coverage_fraction = 0.011;
    continued.material_visible_target_coverage_fraction = 0.0058;
    continued.surface_normal_covered_bin_fraction = 0.115;

    assert!(
        !render_selection_candidate_metrics_beats(&continued, &previous),
        "non-progressive temporal geometry must not promote a mature-material checkpoint"
    );
    assert!(
        render_selection_training_progress_beats(&continued, &previous),
        "safe activation, render, strict-score, and temporal-error progress should not be rolled back before later geometry/render rounds can compound"
    );

    let mut bursty = continued.clone();
    bursty.max_temporal_activation_schedule_error = previous.max_temporal_activation_schedule_error;
    assert!(
        !render_selection_training_progress_beats(&bursty, &previous),
        "mature-material activation continuation still needs temporal schedule progress"
    );
}

#[test]
fn render_selection_training_progress_rejects_morphology_only_continuation() {
    let previous = render_selection_metrics_with_liveness(125.047, 0.87312, 0.6845, 0.0);
    let mut unchanged = previous.clone();
    unchanged.morphology_non_regressed = true;

    assert!(
        !render_selection_training_progress_beats(&unchanged, &previous),
        "line search should not continue from a morphology-only no-op candidate"
    );

    let mut render_only = previous.clone();
    render_only.morphology_non_regressed = true;
    set_render_selection_metrics_render(
        &mut render_only,
        previous.render_loss - 0.010,
        previous.density_psnr_db + 0.10,
    );
    assert!(
        !render_selection_training_progress_beats(&render_only, &previous),
        "render-only improvement should not count as rollout training progress without geometry/material/activation progress"
    );

    let mut coverage_progress = render_only;
    coverage_progress.surface_covered_bin_fraction = previous.surface_covered_bin_fraction + 0.06;
    assert!(
        render_selection_training_progress_beats(&coverage_progress, &previous),
        "bounded morphology-preserving render plus coverage progress should still continue training"
    );
}

#[test]
fn render_selection_training_progress_accepts_bounded_precursor_continuation() {
    let mut previous = render_selection_metrics_with_liveness(91.34, 0.75218, 1.39, 0.79);
    previous.material_active_mean_opacity = 1.64;
    previous.material_visible_count = 4;
    previous.material_visible_target_mean_distance = 0.3482;
    previous.surface_covered_bin_fraction = 0.046875;
    previous.material_visible_surface_covered_bin_fraction = 0.0;
    previous.target_coverage_fraction = 0.005859375;
    previous.material_visible_target_coverage_fraction = 0.0;

    let mut continued = previous.clone();
    continued.score = previous.score + 0.02;
    set_render_selection_metrics_render(
        &mut continued,
        previous.render_loss - 0.0010,
        previous.density_psnr_db + 0.015,
    );
    continued.material_active_mean_opacity = previous.material_active_mean_opacity + 0.03;
    continued.material_visible_target_mean_distance =
        previous.material_visible_target_mean_distance - 0.0015;
    continued.max_front_liveness_margin = previous.max_front_liveness_margin - 0.03;

    assert!(
        !render_selection_candidate_metrics_beats(&continued, &previous),
        "precursor continuation should not be promoted as a selected checkpoint"
    );
    assert!(
        render_selection_training_progress_beats(&continued, &previous),
        "bounded render, material, and local-front precursor progress should keep training moving"
    );
}

#[test]
fn render_selection_training_progress_rejects_mature_material_opacity_without_surface_progress() {
    let mut previous = render_selection_metrics_with_liveness(91.34, 0.75218, 1.39, 0.79);
    previous.material_active_mean_opacity = 1.64;
    previous.material_visible_count = 16;
    previous.material_visible_target_mean_distance = 0.3482;
    previous.surface_covered_bin_fraction = 0.046875;
    previous.material_visible_surface_covered_bin_fraction = 0.0;
    previous.target_coverage_fraction = 0.005859375;
    previous.material_visible_target_coverage_fraction = 0.0;

    let mut opacity_only = previous.clone();
    opacity_only.score = previous.score + 0.02;
    set_render_selection_metrics_render(
        &mut opacity_only,
        previous.render_loss - 0.0010,
        previous.density_psnr_db + 0.015,
    );
    opacity_only.material_active_mean_opacity = previous.material_active_mean_opacity + 0.03;
    opacity_only.max_front_liveness_margin = previous.max_front_liveness_margin;

    assert!(
        !render_selection_candidate_metrics_beats(&opacity_only, &previous),
        "mature material opacity-only continuation should not be promoted"
    );
    assert!(
        !render_selection_training_progress_beats(&opacity_only, &previous),
        "mature visible-material continuation should not keep training on core opacity alone"
    );

    let mut surface_approach = opacity_only;
    surface_approach.material_visible_target_mean_distance =
        previous.material_visible_target_mean_distance - 0.006;
    assert!(
        !render_selection_training_progress_beats(&surface_approach, &previous),
        "mature visible-material continuation should not retain static target-surface approach without temporal geometry progress"
    );
    surface_approach.all_temporal_geometry_progressive = true;
    assert!(
        render_selection_training_progress_beats(&surface_approach, &previous),
        "mature material continuation remains valid once visible material moves toward target support with progressive geometry"
    );
}

#[test]
fn render_selection_training_progress_accepts_strict_surface_material_margin() {
    let mut previous = render_selection_metrics_with_liveness(99.49, 0.6847, 1.94, 0.0);
    previous.material_visible_count = 46;
    previous.material_active_mean_opacity = 1.75;
    previous.material_visible_target_coverage_fraction = 0.0;
    previous.material_visible_surface_covered_bin_fraction = 0.0;
    previous.strict_surface_active_count = 24;
    previous.strict_surface_materialized_fraction = 0.0;
    previous.strict_surface_material_mean_opacity = -3.20;
    previous.strict_surface_material_visible_margin = 2.20;
    previous.strict_surface_material_max_visible_margin = 2.80;
    previous.surface_covered_bin_fraction = 0.140625;
    previous.target_coverage_fraction = 0.060546875;

    let mut continued = previous.clone();
    continued.score = previous.score + 0.01;
    set_render_selection_metrics_render(
        &mut continued,
        previous.render_loss + 0.001,
        previous.density_psnr_db - 0.007,
    );
    continued.strict_surface_material_mean_opacity = -3.05;
    continued.strict_surface_material_visible_margin = 2.05;

    assert!(
        !render_selection_candidate_metrics_beats(&continued, &previous),
        "strict-band material margin alone should not promote a checkpoint before material visibility gates flip"
    );
    assert!(
        !render_selection_training_progress_beats(&continued, &previous),
        "mature strict-band material margin progress should not continue if temporal geometry is static"
    );
    continued.all_temporal_geometry_progressive = true;
    assert!(
        render_selection_training_progress_beats(&continued, &previous),
        "bounded strict-band material margin progress should keep material-only training from rolling back"
    );

    let mut larger_bounded_step = continued.clone();
    set_render_selection_metrics_render(
        &mut larger_bounded_step,
        previous.render_loss + 0.0024,
        previous.density_psnr_db - 0.014,
    );
    larger_bounded_step.strict_surface_material_mean_opacity = -2.95;
    larger_bounded_step.strict_surface_material_visible_margin = 1.95;
    larger_bounded_step.all_temporal_geometry_progressive = true;
    assert!(
        render_selection_training_progress_beats(&larger_bounded_step, &previous),
        "larger strict-band material progress should earn a tightly capped render slack"
    );

    let mut render_degraded = larger_bounded_step.clone();
    set_render_selection_metrics_render(
        &mut render_degraded,
        previous.render_loss + 0.004,
        previous.density_psnr_db - 0.014,
    );
    assert!(
        !render_selection_training_progress_beats(&render_degraded, &previous),
        "strict-band material progress must still reject larger render degradation"
    );

    let mut tail_regressed = continued.clone();
    tail_regressed.material_visible_surface_tail_over_threshold_fraction = 0.05;
    assert!(
        !render_selection_training_progress_beats(&tail_regressed, &previous),
        "strict-band material progress must not hide material tail leaks"
    );

    let mut inactive_material_leaked = continued.clone();
    inactive_material_leaked.material_visible_inactive_fraction = 0.05;
    assert!(
        !render_selection_training_progress_beats(&inactive_material_leaked, &previous),
        "strict-band material progress must not make dormant particles render-visible"
    );

    let mut coverage_collapsed = continued;
    coverage_collapsed.surface_covered_bin_fraction = previous.surface_covered_bin_fraction - 0.01;
    assert!(
        !render_selection_training_progress_beats(&coverage_collapsed, &previous),
        "strict-band material progress must not carry active surface coverage collapse"
    );
}
