use super::*;

#[test]
fn render_selection_training_progress_accepts_bounded_strict_target_coverage() {
    let mut previous = render_selection_metrics_with_liveness(128.4, 0.682, 1.86, 0.0);
    previous.morphology_non_regressed = false;
    previous.target_coverage_fraction = 0.42;
    previous.material_visible_target_coverage_fraction = 0.28;
    previous.surface_covered_bin_fraction = 0.31;
    previous.material_visible_surface_covered_bin_fraction = 0.24;
    previous.surface_normal_covered_bin_fraction = 0.46;
    previous.material_visible_surface_normal_covered_bin_fraction = 0.37;
    previous.min_newly_activated_fraction = 0.62;
    previous.min_front_local_newly_activated_fraction = 0.83;
    previous.max_temporal_activation_schedule_error = 0.18;

    let mut coverage = previous.clone();
    coverage.score = previous.score - 0.50;
    set_render_selection_metrics_render(
        &mut coverage,
        previous.render_loss + 0.003,
        previous.density_psnr_db - 0.08,
    );
    coverage.target_coverage_fraction = previous.target_coverage_fraction + 0.03;
    coverage.surface_covered_bin_fraction = previous.surface_covered_bin_fraction + 0.025;

    assert!(
        !render_selection_candidate_metrics_beats(&coverage, &previous),
        "bounded strict target coverage progress should remain a continuation, not a promoted checkpoint"
    );
    assert!(
        render_selection_training_progress_beats(&coverage, &previous),
        "bounded target/surface coverage improvement should continue 3D training even with a small render tradeoff"
    );

    let mut leaked = coverage.clone();
    leaked.material_visible_inactive_fraction = 0.05;
    assert!(
        !render_selection_training_progress_beats(&leaked, &previous),
        "strict target coverage progress must not be retained by making dormant material visible"
    );

    let mut render_regressed = coverage;
    set_render_selection_metrics_render(
        &mut render_regressed,
        previous.render_loss + 0.030,
        previous.density_psnr_db - 0.08,
    );
    assert!(
        !render_selection_training_progress_beats(&render_regressed, &previous),
        "strict target coverage progress still needs bounded render slack"
    );
}

#[test]
fn render_selection_training_progress_accepts_bounded_temporal_schedule_progress() {
    let mut previous = render_selection_metrics_with_liveness(94.3, 0.241, 8.2, 0.0);
    previous.morphology_non_regressed = false;
    previous.min_newly_activated_fraction = 0.71;
    previous.min_front_local_newly_activated_fraction = 0.91;
    previous.max_temporal_activation_schedule_error = 0.24;
    previous.all_temporal_activation_progressive = false;
    previous.all_temporal_geometry_progressive = true;
    previous.target_coverage_fraction = 0.64;
    previous.material_visible_target_coverage_fraction = 0.61;
    previous.surface_covered_bin_fraction = 0.68;
    previous.material_visible_surface_covered_bin_fraction = 0.65;
    previous.surface_normal_covered_bin_fraction = 0.60;
    previous.material_visible_surface_normal_covered_bin_fraction = 0.58;

    let mut scheduled = previous.clone();
    scheduled.score = previous.score - 0.70;
    set_render_selection_metrics_render(
        &mut scheduled,
        previous.render_loss + 0.004,
        previous.density_psnr_db - 0.11,
    );
    scheduled.max_temporal_activation_schedule_error = 0.19;

    assert!(
        render_selection_training_progress_beats(&scheduled, &previous),
        "bounded temporal schedule improvement should keep seed-varied 3D training moving"
    );

    let mut local_front_regressed = scheduled.clone();
    local_front_regressed.min_front_local_newly_activated_fraction = 0.42;
    assert!(
        !render_selection_training_progress_beats(&local_front_regressed, &previous),
        "temporal schedule progress must preserve local-front morphogenesis behavior"
    );

    let mut activation_collapsed = scheduled;
    activation_collapsed.min_newly_activated_fraction = 0.60;
    assert!(
        !render_selection_training_progress_beats(&activation_collapsed, &previous),
        "temporal schedule progress must not hide reduced activation coverage"
    );
}
