use super::*;

#[test]
fn render_selection_candidate_requires_morphology_and_bounded_render_regression() {
    assert!(render_selection_candidate_beats(
        0.5, 1.0, true, 0.8, 0.9, 2.0, 1.5,
    ));
    assert!(!render_selection_candidate_beats(
        0.5, 1.0, false, 0.8, 0.9, 2.0, 1.5,
    ));
    assert!(!render_selection_candidate_beats(
        1.5, 1.0, true, 0.8, 0.9, 2.0, 1.5,
    ));
    assert!(
        !render_selection_candidate_beats(0.98, 1.0, true, 0.91, 0.9, 2.0, 1.5),
        "weak strict score improvement should not spend render slack"
    );
    assert!(
        render_selection_candidate_beats(0.5, 1.0, true, 0.92, 0.9, 1.35, 1.5),
        "material strict score improvement can accept bounded render/density slack"
    );
    assert!(
        !render_selection_candidate_beats(0.5, 1.0, true, 0.95, 0.9, 2.0, 1.5),
        "strict score improvement should not accept large render loss regression"
    );
    assert!(
        !render_selection_candidate_beats(0.5, 1.0, true, 0.92, 0.9, 1.2, 1.5),
        "strict score improvement should not accept large density PSNR regression"
    );
}

#[test]
fn render_selection_candidate_can_retain_bounded_liveness_precursor_progress() {
    let best = render_selection_metrics_with_liveness(107.10512, 0.925306, 0.3583453, 5.686245);
    let improved =
        render_selection_metrics_with_liveness(107.10004, 0.92530984, 0.3582421, 3.6128867);

    assert!(
        !render_selection_candidate_beats(
            improved.score,
            best.score,
            improved.morphology_non_regressed,
            improved.render_loss,
            best.render_loss,
            improved.density_psnr_db,
            best.density_psnr_db,
        ),
        "the scalar strict selector should still reject weak score improvements with tiny density regression"
    );
    assert!(
        render_selection_candidate_metrics_beats(&improved, &best),
        "metric-aware selection should be able to accumulate bounded local-front liveness progress"
    );

    let mut dormant_drift = improved.clone();
    mark_selection_dormant_drift_unbounded(&mut dormant_drift);
    assert!(
        !render_selection_candidate_metrics_beats(&dormant_drift, &best),
        "selection must not retain local-front progress by moving far dormant particles"
    );

    let mut morphology_regressed = improved.clone();
    morphology_regressed.morphology_non_regressed = false;
    assert!(!render_selection_candidate_metrics_beats(
        &morphology_regressed,
        &best
    ));

    let mut render_regressed = improved.clone();
    render_regressed.render_loss = 0.930;
    assert!(!render_selection_candidate_metrics_beats(
        &render_regressed,
        &best
    ));

    let mut weak_front_progress = improved;
    weak_front_progress.max_front_liveness_margin = 5.66;
    assert!(!render_selection_candidate_metrics_beats(
        &weak_front_progress,
        &best
    ));
}

#[test]
fn render_selection_candidate_can_retain_bounded_temporal_front_liveness_progress() {
    let mut best = render_selection_metrics_with_liveness(107.10512, 0.925306, 0.3583453, 0.0);
    best.max_temporal_front_liveness_margin = 5.686245;
    best.min_temporal_front_liveness_candidate_count = 12;
    let mut improved =
        render_selection_metrics_with_liveness(107.10004, 0.92530984, 0.3582421, 0.0);
    improved.max_temporal_front_liveness_margin = 3.6128867;
    improved.min_temporal_front_liveness_candidate_count = 12;

    assert!(
        !render_selection_candidate_beats(
            improved.score,
            best.score,
            improved.morphology_non_regressed,
            improved.render_loss,
            best.render_loss,
            improved.density_psnr_db,
            best.density_psnr_db,
        ),
        "the scalar strict selector should still reject weak score improvements with tiny density regression"
    );
    assert!(
        render_selection_candidate_metrics_beats(&improved, &best),
        "metric-aware selection should be able to accumulate bounded temporal-front liveness progress"
    );

    let mut weak_temporal_progress = improved;
    weak_temporal_progress.max_temporal_front_liveness_margin = 5.66;
    assert!(!render_selection_candidate_metrics_beats(
        &weak_temporal_progress,
        &best
    ));
}

#[test]
fn render_selection_candidate_can_retain_bounded_extent_front_liveness_progress() {
    let mut best = render_selection_metrics_with_liveness(107.10512, 0.925306, 0.3583453, 0.0);
    best.max_extent_front_liveness_margin = 5.686245;
    best.min_extent_front_liveness_candidate_count = 12;
    let mut improved =
        render_selection_metrics_with_liveness(107.10004, 0.92530984, 0.3582421, 0.0);
    improved.max_extent_front_liveness_margin = 3.6128867;
    improved.min_extent_front_liveness_candidate_count = 12;

    assert!(
        !render_selection_candidate_beats(
            improved.score,
            best.score,
            improved.morphology_non_regressed,
            improved.render_loss,
            best.render_loss,
            improved.density_psnr_db,
            best.density_psnr_db,
        ),
        "the scalar strict selector should still reject weak score improvements with tiny density regression"
    );
    assert!(
        render_selection_candidate_metrics_beats(&improved, &best),
        "metric-aware selection should be able to accumulate bounded extent-front liveness progress"
    );

    let mut morphology_regressed = improved.clone();
    morphology_regressed.morphology_non_regressed = false;
    assert!(!render_selection_candidate_metrics_beats(
        &morphology_regressed,
        &best
    ));

    let mut weak_extent_progress = improved;
    weak_extent_progress.max_extent_front_liveness_margin = 5.66;
    assert!(!render_selection_candidate_metrics_beats(
        &weak_extent_progress,
        &best
    ));
}

#[test]
fn render_selection_candidate_can_retain_bounded_temporal_extent_front_liveness_progress() {
    let mut best = render_selection_metrics_with_liveness(107.10512, 0.925306, 0.3583453, 0.0);
    best.max_temporal_extent_front_liveness_margin = 5.686245;
    best.min_temporal_extent_front_liveness_candidate_count = 12;
    let mut improved =
        render_selection_metrics_with_liveness(107.10004, 0.92530984, 0.3582421, 0.0);
    improved.max_temporal_extent_front_liveness_margin = 3.6128867;
    improved.min_temporal_extent_front_liveness_candidate_count = 12;

    assert!(
        !render_selection_candidate_beats(
            improved.score,
            best.score,
            improved.morphology_non_regressed,
            improved.render_loss,
            best.render_loss,
            improved.density_psnr_db,
            best.density_psnr_db,
        ),
        "the scalar strict selector should still reject weak score improvements with tiny density regression"
    );
    assert!(
        render_selection_candidate_metrics_beats(&improved, &best),
        "metric-aware selection should retain bounded temporal extent-front liveness progress"
    );

    let mut timing_regressed = improved.clone();
    timing_regressed.max_temporal_activation_schedule_error = best
        .max_temporal_activation_schedule_error
        + TEMPORAL_ACTIVATION_SELECTION_REGRESSION_SLACK
        + 0.01;
    timing_regressed.morphology_non_regressed = false;
    assert!(
        !render_selection_candidate_metrics_beats(&timing_regressed, &best),
        "bounded temporal extent-front progress should not hide temporal activation regression"
    );

    let mut weak_temporal_extent_progress = improved;
    weak_temporal_extent_progress.max_temporal_extent_front_liveness_margin = 5.66;
    assert!(!render_selection_candidate_metrics_beats(
        &weak_temporal_extent_progress,
        &best
    ));
}

#[test]
fn render_selection_candidate_can_carry_bounded_temporal_front_precursor_through_morphology_failure()
 {
    let mut best = render_selection_metrics_with_liveness(107.10512, 0.925306, 0.3583453, 0.0);
    best.max_temporal_front_liveness_margin = 5.686245;
    best.min_temporal_front_liveness_candidate_count = 12;
    best.max_temporal_activation_schedule_error = 0.20;

    let mut improved =
        render_selection_metrics_with_liveness(107.10004, 0.92530984, 0.3582421, 0.0);
    improved.morphology_non_regressed = false;
    improved.max_temporal_front_liveness_margin = 3.6128867;
    improved.min_temporal_front_liveness_candidate_count = 12;
    improved.max_temporal_activation_schedule_error = 0.20;
    improved.active_surface_max = 0.25;

    assert!(
        render_selection_candidate_metrics_beats(&improved, &best),
        "bounded temporal-front liveness progress should survive temporary strict morphology failure"
    );

    let mut dormant_drift = improved.clone();
    mark_selection_dormant_drift_unbounded(&mut dormant_drift);
    assert!(
        !render_selection_candidate_metrics_beats(&dormant_drift, &best),
        "temporal-front precursor selection must remain bounded by dormant-drift safety"
    );

    let mut activated_burst = improved.clone();
    activated_burst.min_final_active_count = 32;
    activated_burst.min_newly_activated_fraction = 1.0;
    activated_burst.min_front_local_newly_activated_fraction = 1.0;
    assert!(
        !render_selection_candidate_metrics_beats(&activated_burst, &best),
        "temporal-front precursor selection must not carry an all-active burst through morphology failure"
    );

    let mut escaped = improved.clone();
    escaped.active_surface_max = GROWTH_3D_SURFACE_MAX_DISTANCE + 0.10;
    assert!(!render_selection_candidate_metrics_beats(&escaped, &best));

    let mut timing_regressed = improved.clone();
    timing_regressed.max_temporal_activation_schedule_error = best
        .max_temporal_activation_schedule_error
        + TEMPORAL_ACTIVATION_SELECTION_REGRESSION_SLACK
        + 0.01;
    assert!(!render_selection_candidate_metrics_beats(
        &timing_regressed,
        &best
    ));

    let mut render_regressed = improved;
    render_regressed.render_loss = 0.930;
    assert!(!render_selection_candidate_metrics_beats(
        &render_regressed,
        &best
    ));
}

#[test]
fn render_selection_candidate_can_retain_local_activation_breakthrough() {
    let best = render_selection_metrics_with_liveness(107.08897, 0.92548585, 0.35743254, 1.5105798);
    let mut activated = render_selection_metrics_with_liveness(76.6387, 0.9238937, 0.36455846, 0.0);
    activated.morphology_non_regressed = false;
    activated.min_final_active_count = 32;
    activated.min_newly_activated_fraction = 1.0;
    activated.min_front_local_newly_activated_fraction = 1.0;
    activated.active_surface_max = 0.25;
    activated.all_temporal_activation_progressive = true;

    assert!(
        render_selection_candidate_metrics_beats(&activated, &best),
        "a bounded local-front activation breakthrough with improved render metrics should be retained for continued training"
    );

    let mut escaped = activated.clone();
    escaped.active_surface_max = GROWTH_3D_SURFACE_MAX_DISTANCE + 0.10;
    assert!(!render_selection_candidate_metrics_beats(&escaped, &best));

    let mut nonlocal = activated;
    nonlocal.min_front_local_newly_activated_fraction = 0.0;
    assert!(!render_selection_candidate_metrics_beats(&nonlocal, &best));
}
