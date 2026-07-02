use super::*;

#[test]
fn render_selection_progress_prefers_bounded_strict_material_margin_step() {
    let mut no_op = render_selection_metrics_with_liveness(99.49, 0.6847, 1.94, 0.0);
    no_op.strict_surface_active_count = 24;
    no_op.strict_surface_materialized_fraction = 0.0;
    no_op.strict_surface_material_mean_opacity = -3.20;
    no_op.strict_surface_material_visible_margin = 2.20;
    no_op.strict_surface_material_max_visible_margin = 2.80;
    no_op.material_visible_target_coverage_fraction = 0.0;
    no_op.material_visible_surface_covered_bin_fraction = 0.0;

    let mut render_preferred = no_op.clone();
    set_render_selection_metrics_render(
        &mut render_preferred,
        no_op.render_loss + 0.001,
        no_op.density_psnr_db - 0.007,
    );
    render_preferred.strict_surface_material_mean_opacity = -3.185;
    render_preferred.strict_surface_material_visible_margin = 2.185;

    let mut material_preferred = no_op.clone();
    set_render_selection_metrics_render(
        &mut material_preferred,
        no_op.render_loss + 0.0024,
        no_op.density_psnr_db - 0.014,
    );
    material_preferred.strict_surface_material_mean_opacity = -3.17;
    material_preferred.strict_surface_material_visible_margin = 2.17;

    assert!(
        render_selection_progress_candidate_preferred(
            &material_preferred,
            &render_preferred,
            &no_op,
        ),
        "bounded progress selection should prefer the candidate that closes more strict-band material margin"
    );

    let mut dormant_drift = material_preferred.clone();
    mark_selection_dormant_drift_unbounded(&mut dormant_drift);
    assert!(
        !render_selection_progress_candidate_preferred(&dormant_drift, &render_preferred, &no_op),
        "progress tie-breakers must not prefer strict-band material progress with nonlocal dormant drift"
    );

    let mut degraded = material_preferred.clone();
    set_render_selection_metrics_render(
        &mut degraded,
        no_op.render_loss + 0.004,
        no_op.density_psnr_db - 0.014,
    );
    assert!(
        !render_selection_progress_candidate_preferred(&degraded, &render_preferred, &no_op),
        "strict-band material progress should not win the tie-breaker outside render slack"
    );

    let mut inactive_material_leaked = material_preferred;
    inactive_material_leaked.material_visible_inactive_fraction = 0.05;
    assert!(
        !render_selection_progress_candidate_preferred(
            &inactive_material_leaked,
            &render_preferred,
            &no_op,
        ),
        "strict-band material progress should not win the tie-breaker by leaking dormant material"
    );
}

#[test]
fn render_selection_progress_rejects_strict_material_tiebreak_with_score_regression() {
    let mut no_op = render_selection_metrics_with_liveness(99.49, 0.6847, 1.94, 0.0);
    no_op.strict_surface_active_count = 24;
    no_op.strict_surface_materialized_fraction = 0.2;
    no_op.strict_surface_material_mean_opacity = -3.20;
    no_op.strict_surface_material_visible_margin = 2.20;
    no_op.material_visible_target_coverage_fraction = 0.0;

    let mut better_score = no_op.clone();
    better_score.score = 191.60;
    set_render_selection_metrics_render(
        &mut better_score,
        no_op.render_loss - 0.02,
        no_op.density_psnr_db + 0.15,
    );
    better_score.strict_surface_materialized_fraction = 0.2;
    better_score.strict_surface_material_mean_opacity = -3.19;
    better_score.strict_surface_material_visible_margin = 2.19;

    let mut material_only = better_score.clone();
    material_only.score = 202.0;
    set_render_selection_metrics_render(
        &mut material_only,
        better_score.render_loss - 0.01,
        better_score.density_psnr_db + 0.10,
    );
    material_only.strict_surface_materialized_fraction = 0.25;
    material_only.strict_surface_material_mean_opacity = -3.05;
    material_only.strict_surface_material_visible_margin = 2.05;

    assert!(
        !render_selection_progress_candidate_preferred(&material_only, &better_score, &no_op),
        "strict-band materialization is a tie-breaker and must not override a large strict-score regression"
    );
}

#[test]
fn render_selection_training_progress_rejects_precursor_coverage_collapse() {
    let mut previous = render_selection_metrics_with_liveness(91.34, 0.75218, 1.39, 0.79);
    previous.material_active_mean_opacity = 1.64;
    previous.surface_covered_bin_fraction = 0.046875;
    previous.target_coverage_fraction = 0.005859375;

    let mut collapsed = previous.clone();
    set_render_selection_metrics_render(
        &mut collapsed,
        previous.render_loss - 0.01,
        previous.density_psnr_db + 0.05,
    );
    collapsed.material_active_mean_opacity = previous.material_active_mean_opacity + 0.20;
    collapsed.max_front_liveness_margin = previous.max_front_liveness_margin - 0.20;
    collapsed.surface_covered_bin_fraction = 0.015625;
    collapsed.target_coverage_fraction = 0.0;

    assert!(
        !render_selection_training_progress_beats(&collapsed, &previous),
        "precursor continuation must not carry a render-improving coverage collapse"
    );
}

#[test]
fn render_selection_training_progress_accepts_bounded_color_emergence() {
    let mut previous = render_selection_metrics_with_liveness(99.4, 0.6756, 1.99, 0.0);
    previous.active_color_state_mean_abs = 0.020;
    previous.active_color_state_max_abs = 0.045;
    previous.active_color_state_stddev_mean = 0.011;
    previous.min_final_active_count = 128;

    let mut colored = previous.clone();
    colored.active_color_state_mean_abs = 0.023;
    colored.active_color_state_max_abs = 0.0475;
    colored.active_color_state_stddev_mean = 0.0125;

    assert!(
        !render_selection_candidate_metrics_beats(&colored, &previous),
        "color-only progress should not be promoted as a strict selected checkpoint"
    );
    assert!(
        render_selection_training_progress_beats(&colored, &previous),
        "bounded rollout color-state emergence should keep direct training moving"
    );
}

#[test]
fn render_selection_training_progress_rejects_color_emergence_with_geometry_regression() {
    let mut previous = render_selection_metrics_with_liveness(99.4, 0.6756, 1.99, 0.0);
    previous.active_color_state_stddev_mean = 0.011;
    previous.target_coverage_fraction = 0.30;
    previous.material_visible_target_coverage_fraction = 0.24;
    previous.surface_covered_bin_fraction = 0.32;
    previous.material_visible_surface_covered_bin_fraction = 0.26;

    let mut collapsed = previous.clone();
    collapsed.active_color_state_stddev_mean = 0.013;
    collapsed.target_coverage_fraction = 0.28;

    assert!(
        !render_selection_training_progress_beats(&collapsed, &previous),
        "color progress must not carry a target-coverage collapse"
    );
}

#[test]
fn render_selection_training_progress_rejects_color_emergence_with_material_tail_leak() {
    let mut previous = render_selection_metrics_with_liveness(99.4, 0.6756, 1.99, 0.0);
    previous.active_color_state_stddev_mean = 0.011;
    previous.material_visible_surface_tail_over_threshold_fraction = 0.0;

    let mut leaked = previous.clone();
    leaked.active_color_state_stddev_mean = 0.013;
    leaked.material_visible_surface_tail_over_threshold_fraction = 0.05;

    assert!(
        !render_selection_training_progress_beats(&leaked, &previous),
        "color progress must remain bounded by material-visible tail safety"
    );
}

#[test]
fn render_selection_can_retain_geometry_growth_before_material_visibility() {
    let mut previous = render_selection_metrics_with_liveness(112.13, 0.6346, 2.26, 0.0);
    previous.morphology_non_regressed = false;
    previous.active_surface_max = 0.43;
    previous.min_active_extent_bbox_ratio = 0.09;
    previous.min_active_extent_min_axis_ratio = 0.07;
    previous.min_final_active_count = 43;
    previous.min_newly_activated_fraction = 0.29;
    previous.min_front_local_newly_activated_fraction = 0.62;
    previous.target_coverage_fraction = 0.048;
    previous.surface_covered_bin_fraction = 0.078;
    previous.surface_normal_covered_bin_fraction = 0.269;
    previous.material_visible_target_coverage_fraction = 0.019;
    previous.material_visible_surface_covered_bin_fraction = 0.062;
    previous.material_visible_surface_normal_covered_bin_fraction = 0.153;
    previous.max_temporal_activation_schedule_error = 0.18;

    let mut geometry = previous.clone();
    geometry.score = 104.12;
    set_render_selection_metrics_render(&mut geometry, 0.646, previous.density_psnr_db - 0.08);
    geometry.active_surface_max = 0.61;
    geometry.min_active_extent_bbox_ratio = 0.284;
    geometry.min_active_extent_min_axis_ratio = 0.175;
    geometry.min_final_active_count = 54;
    geometry.min_newly_activated_fraction = 0.383;
    geometry.target_coverage_fraction = previous.target_coverage_fraction;
    geometry.surface_covered_bin_fraction = 0.109;
    geometry.surface_normal_covered_bin_fraction = 0.384;
    geometry.material_visible_target_coverage_fraction = 0.0;
    geometry.material_visible_surface_covered_bin_fraction = 0.0;
    geometry.material_visible_surface_normal_covered_bin_fraction = 0.0;
    geometry.max_temporal_activation_schedule_error =
        previous.max_temporal_activation_schedule_error;

    assert!(
        render_selection_candidate_metrics_beats(&geometry, &previous),
        "strict-score-backed active geometry growth should be retained even when surface-gated material is temporarily invisible"
    );
    assert!(
        render_selection_training_progress_beats(&geometry, &previous),
        "geometry-first progress should keep training moving while materialization catches up"
    );

    let mut target_collapse = geometry.clone();
    target_collapse.target_coverage_fraction = previous.target_coverage_fraction - 0.02;
    assert!(
        !render_selection_candidate_metrics_beats(&target_collapse, &previous),
        "geometry precursor selection must not accept target support collapse"
    );
    assert!(!render_selection_training_progress_beats(
        &target_collapse,
        &previous
    ));

    let mut material_leak = geometry;
    material_leak.material_visible_surface_tail_over_threshold_fraction = 0.02;
    assert!(
        !render_selection_candidate_metrics_beats(&material_leak, &previous),
        "geometry precursor selection must remain bounded by material-visible tail safety"
    );
    assert!(!render_selection_training_progress_beats(
        &material_leak,
        &previous
    ));
}

#[test]
fn render_selection_can_continue_bounded_surface_support_expansion() {
    let mut previous = render_selection_metrics_with_liveness(101.402_374, 0.676_347_6, 1.991, 0.0);
    previous.morphology_non_regressed = false;
    previous.active_surface_max = 0.348_897_04;
    previous.target_coverage_fraction = 0.048_828_125;
    previous.surface_covered_bin_fraction = 0.109_375;
    previous.surface_normal_covered_bin_fraction = 0.384;
    previous.material_visible_target_coverage_fraction = 0.0;
    previous.material_visible_surface_covered_bin_fraction = 0.0;
    previous.material_visible_surface_normal_covered_bin_fraction = 0.0;
    previous.min_active_extent_bbox_ratio = 0.287;
    previous.min_active_extent_min_axis_ratio = 0.184;
    previous.min_final_active_count = 55;
    previous.min_newly_activated_fraction = 0.392;
    previous.min_front_local_newly_activated_fraction = 0.957_446_8;
    previous.max_temporal_activation_schedule_error = 0.072_395_83;
    previous.material_visible_surface_tail_over_threshold_fraction = 0.0;

    let mut bounded = previous.clone();
    bounded.score = 108.680_3;
    set_render_selection_metrics_render(&mut bounded, 0.689_317_1, 1.914_362_3);
    bounded.active_surface_max = 0.402_585_36;
    bounded.target_coverage_fraction = 0.058_593_75;
    bounded.surface_covered_bin_fraction = 0.171_875;
    bounded.surface_normal_covered_bin_fraction = 0.461_538_46;
    bounded.min_active_extent_bbox_ratio = 0.597_709_06;
    bounded.min_active_extent_min_axis_ratio = 0.489_307_8;
    bounded.min_final_active_count = 90;
    bounded.min_newly_activated_fraction = 0.683_333_34;
    bounded.min_front_local_newly_activated_fraction = 0.674_698_77;
    bounded.max_temporal_activation_schedule_error = 0.142_534_73;

    assert!(
        !render_selection_candidate_metrics_beats(&bounded, &previous),
        "moderate score-regressing surface expansion should not become a selected checkpoint"
    );
    assert!(
        render_selection_training_progress_beats(&bounded, &previous),
        "bounded active support expansion should continue training before material coverage catches up"
    );

    let mut bursty = bounded.clone();
    bursty.score = 118.882_69;
    set_render_selection_metrics_render(&mut bursty, 0.694_972_16, 1.881_357);
    bursty.active_surface_max = 0.451_338_35;
    bursty.target_coverage_fraction = 0.070_312_5;
    bursty.surface_covered_bin_fraction = 0.218_75;
    bursty.surface_normal_covered_bin_fraction = 0.5;
    bursty.min_active_extent_bbox_ratio = 0.800_067_3;
    bursty.min_active_extent_min_axis_ratio = 0.688_672_8;
    bursty.min_final_active_count = 119;
    bursty.min_newly_activated_fraction = 0.925;
    bursty.min_front_local_newly_activated_fraction = 0.522_522_5;
    bursty.max_temporal_activation_schedule_error = 0.279_687_52;

    assert!(
        !render_selection_training_progress_beats(&bursty, &previous),
        "surface expansion continuation must still reject bursty, nonlocal activation"
    );
}

#[test]
fn render_selection_can_continue_bounded_geometry_expansion_without_checkpointing() {
    let mut previous = render_selection_metrics_with_liveness(101.40, 0.6763, 1.991, 0.0);
    previous.morphology_non_regressed = true;
    previous.active_surface_max = 0.348;
    previous.min_active_extent_bbox_ratio = 0.287;
    previous.min_active_extent_min_axis_ratio = 0.184;
    previous.min_final_active_count = 55;
    previous.min_newly_activated_fraction = 0.392;
    previous.min_front_local_newly_activated_fraction = 0.90;
    previous.target_coverage_fraction = 0.0488;
    previous.surface_covered_bin_fraction = 0.109;
    previous.surface_normal_covered_bin_fraction = 0.384;
    previous.material_visible_target_coverage_fraction = 0.0;
    previous.material_visible_surface_covered_bin_fraction = 0.0;
    previous.material_visible_surface_normal_covered_bin_fraction = 0.0;
    previous.max_temporal_activation_schedule_error = 0.068;

    let mut expanded = previous.clone();
    expanded.morphology_non_regressed = false;
    expanded.score = 105.29;
    set_render_selection_metrics_render(
        &mut expanded,
        previous.render_loss + 0.009,
        previous.density_psnr_db - 0.055,
    );
    expanded.active_surface_max = 0.402;
    expanded.min_active_extent_bbox_ratio = 0.490;
    expanded.min_active_extent_min_axis_ratio = 0.349;
    expanded.min_final_active_count = 74;
    expanded.min_newly_activated_fraction = 0.55;
    expanded.min_front_local_newly_activated_fraction = 0.88;
    expanded.surface_covered_bin_fraction = 0.125;
    expanded.max_temporal_activation_schedule_error = 0.096;

    assert!(
        !render_selection_candidate_metrics_beats(&expanded, &previous),
        "bounded render-regressing geometry expansion should not become a selected checkpoint"
    );
    assert!(
        render_selection_training_progress_beats(&expanded, &previous),
        "bounded local-front geometry expansion should continue training even before render/material metrics improve"
    );

    let mut bursty = expanded.clone();
    set_render_selection_metrics_render(
        &mut bursty,
        previous.render_loss + 0.019,
        previous.density_psnr_db - 0.12,
    );
    bursty.min_final_active_count = 119;
    bursty.min_newly_activated_fraction = 0.925;
    bursty.min_front_local_newly_activated_fraction = 0.52;
    bursty.max_temporal_activation_schedule_error = previous.max_temporal_activation_schedule_error
        + TEMPORAL_ACTIVATION_SELECTION_REGRESSION_SLACK
        + 0.10;
    assert!(
        !render_selection_training_progress_beats(&bursty, &previous),
        "geometry expansion continuation must still reject bursty timing and excessive render regression"
    );
}

#[test]
fn render_selection_morphology_recovery_requires_strict_score_improvement() {
    let mut regressed = render_selection_metrics_with_liveness(125.047, 0.87312, 0.6845, 0.0);
    regressed.morphology_non_regressed = false;

    let mut same_score_recovery = regressed.clone();
    same_score_recovery.morphology_non_regressed = true;
    set_render_selection_metrics_render(
        &mut same_score_recovery,
        regressed.render_loss - 0.001,
        regressed.density_psnr_db + 0.001,
    );
    assert!(
        !render_selection_morphology_recovery_beats(&same_score_recovery, &regressed),
        "line search should not accept morphology recovery without strict-score improvement"
    );

    let mut strict_recovery = same_score_recovery;
    strict_recovery.score = regressed.score - 0.01;
    assert!(
        render_selection_morphology_recovery_beats(&strict_recovery, &regressed),
        "bounded morphology recovery with strict-score and render non-regression should stay eligible"
    );

    let mut render_regressed = strict_recovery;
    set_render_selection_metrics_render(
        &mut render_regressed,
        regressed.render_loss + 0.02,
        regressed.density_psnr_db + 0.001,
    );
    assert!(
        !render_selection_morphology_recovery_beats(&render_regressed, &regressed),
        "strict-score recovery should not hide render regression"
    );
}

#[test]
fn render_selection_candidate_can_retain_bounded_material_precursor() {
    let mut best = render_selection_metrics_with_liveness(77.0, 0.9188, 0.398, 0.0);
    best.min_final_active_count = 32;
    best.material_active_mean_opacity = -3.60;
    best.material_visible_count = 1;

    let mut improved = best.clone();
    improved.material_active_mean_opacity = -3.40;
    set_render_selection_metrics_render(&mut improved, 0.91881, 0.39799);

    assert!(
        render_selection_candidate_metrics_beats(&improved, &best),
        "bounded material opacity precursor progress should be retained before visibility coverage gates flip"
    );

    let mut tail_regressed = improved.clone();
    tail_regressed.material_visible_surface_tail_over_threshold_fraction = 0.02;
    assert!(!render_selection_candidate_metrics_beats(
        &tail_regressed,
        &best
    ));

    let mut activation_regressed = improved;
    activation_regressed.min_final_active_count = 16;
    assert!(!render_selection_candidate_metrics_beats(
        &activation_regressed,
        &best
    ));
}

#[test]
fn render_selection_candidate_rejects_mature_material_opacity_without_surface_progress() {
    let mut best = render_selection_metrics_with_liveness(77.0, 0.9188, 0.398, 0.0);
    best.min_final_active_count = 32;
    best.material_active_mean_opacity = -3.60;
    best.material_visible_count = 16;
    best.material_visible_target_mean_distance = 0.72;
    best.material_visible_target_coverage_fraction = 0.0;
    best.material_visible_surface_covered_bin_fraction = 0.0;
    best.material_visible_surface_normal_covered_bin_fraction = 0.0;

    let mut opacity_only = best.clone();
    opacity_only.score = best.score + 0.01;
    opacity_only.material_active_mean_opacity = -3.40;
    set_render_selection_metrics_render(&mut opacity_only, 0.91881, 0.39799);

    assert!(
        !render_selection_candidate_metrics_beats(&opacity_only, &best),
        "mature visible-material candidates should not be selected by core opacity alone"
    );

    let mut surface_approach = opacity_only;
    surface_approach.material_visible_target_mean_distance =
        best.material_visible_target_mean_distance - 0.006;
    assert!(
        !render_selection_candidate_metrics_beats(&surface_approach, &best),
        "mature visible-material candidates should not be selected by static surface approach alone"
    );
    surface_approach.all_temporal_geometry_progressive = true;
    assert!(
        render_selection_candidate_metrics_beats(&surface_approach, &best),
        "mature visible-material precursors remain selectable when surface approach also has progressive temporal geometry"
    );
}

#[test]
fn render_selection_rejects_bursty_activation_breakthrough_timing_regression() {
    let mut best = render_selection_metrics_with_liveness(81.15844, 0.925548, 0.3570, 1.5105798);
    best.max_temporal_activation_schedule_error = 0.18607144;
    best.all_temporal_activation_progressive = false;

    let mut bursty = render_selection_metrics_with_liveness(75.13471, 0.91643715, 0.3985, 0.0);
    bursty.morphology_non_regressed = false;
    bursty.min_final_active_count = 32;
    bursty.min_newly_activated_fraction = 1.0;
    bursty.min_front_local_newly_activated_fraction = 1.0;
    bursty.active_surface_max = 0.25;
    bursty.max_temporal_activation_schedule_error = 0.34142855;

    assert!(
        !render_selection_candidate_metrics_beats(&bursty, &best),
        "render and activation breakthroughs must not hide a worse growth schedule"
    );

    bursty.max_temporal_activation_schedule_error = best.max_temporal_activation_schedule_error
        + TEMPORAL_ACTIVATION_SELECTION_REGRESSION_SLACK;
    assert!(
        !render_selection_candidate_metrics_beats(&bursty, &best),
        "timing-neutral all-active bursts still fail unless the temporal rollout is progressive"
    );

    bursty.all_temporal_activation_progressive = true;
    assert!(
        render_selection_candidate_metrics_beats(&bursty, &best),
        "temporally progressive activation breakthroughs remain valid"
    );
}

#[test]
fn render_selection_candidate_can_refine_after_activation_breakthrough() {
    let mut best = render_selection_metrics_with_liveness(76.6387, 0.9238937, 0.36455846, 0.0);
    best.morphology_non_regressed = false;
    best.min_final_active_count = 32;
    best.min_newly_activated_fraction = 1.0;
    best.min_front_local_newly_activated_fraction = 1.0;
    best.active_surface_max = 0.25;
    best.all_temporal_activation_progressive = true;

    let mut refined = best.clone();
    refined.score = 66.73771;
    set_render_selection_metrics_render(&mut refined, 0.91643715, 0.39857513);

    assert!(
        render_selection_candidate_metrics_beats(&refined, &best),
        "after a retained activation breakthrough, continued bounded render/strict-score refinement should not be blocked by the initial breakthrough morphology flag"
    );

    let mut lost_activation = refined;
    lost_activation.min_newly_activated_fraction = 0.50;
    assert!(!render_selection_candidate_metrics_beats(
        &lost_activation,
        &best
    ));
}

#[test]
fn render_selection_rejects_post_activation_refinement_timing_regression() {
    let mut best = render_selection_metrics_with_liveness(76.6387, 0.9238937, 0.36455846, 0.0);
    best.morphology_non_regressed = false;
    best.min_final_active_count = 32;
    best.min_newly_activated_fraction = 1.0;
    best.min_front_local_newly_activated_fraction = 1.0;
    best.active_surface_max = 0.25;
    best.max_temporal_activation_schedule_error = 0.20;
    best.all_temporal_activation_progressive = true;

    let mut refined = best.clone();
    refined.score = 66.73771;
    set_render_selection_metrics_render(&mut refined, 0.91643715, 0.39857513);
    refined.max_temporal_activation_schedule_error = 0.35;
    refined.all_temporal_activation_progressive = false;

    assert!(
        !render_selection_candidate_metrics_beats(&refined, &best),
        "post-activation refinement should also preserve temporal growth timing"
    );
}
