use super::*;

#[test]
fn local_front_liveness_progress_measures_dormant_activation_margin() {
    let config = NpaConfig::growing_3dgs();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.1_f32, 0.0, 0.0, 0.0],
        [0.5_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = -3.0;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;

    let progress = local_front_liveness_progress(&config, &positions, &states, 0.2);

    assert_eq!(progress.candidate_count, 1);
    assert!(
        (progress.weighted_activation_margin - 2.0).abs() < 1.0e-5,
        "near dormant front particle is two logits below the active threshold"
    );
}

#[test]
fn render_selection_score_rewards_lower_local_front_liveness_margin() {
    let weak = render_selection_case_with_front_liveness_margin(8.0);
    let better = render_selection_case_with_front_liveness_margin(2.0);

    let weak_score = render_selection_case_score_with_baseline(7, &weak, None);
    let better_score = render_selection_case_score_with_baseline(7, &better, None);

    assert!(weak_score.morphology_non_regressed);
    assert!(better_score.morphology_non_regressed);
    assert!(
        better_score.score < weak_score.score,
        "line search should see sub-threshold local-front liveness progress before strict activation gates flip"
    );
    assert!(
        (weak_score.score - better_score.score - 6.0 * LOCAL_FRONT_LIVENESS_SCORE_WEIGHT).abs()
            < 1.0e-6
    );
}

#[test]
fn render_selection_score_rewards_lower_temporal_front_liveness_margin() {
    let mut weak = render_selection_case_with_front_liveness_margin(0.0);
    weak.temporal_front_liveness = LocalFrontLivenessProgress {
        candidate_count: 4,
        weighted_activation_margin: 8.0,
    };
    let mut better = render_selection_case_with_front_liveness_margin(0.0);
    better.temporal_front_liveness = LocalFrontLivenessProgress {
        candidate_count: 4,
        weighted_activation_margin: 2.0,
    };

    let weak_score = render_selection_case_score_with_baseline(7, &weak, None);
    let better_score = render_selection_case_score_with_baseline(7, &better, None);

    assert!(weak_score.morphology_non_regressed);
    assert!(better_score.morphology_non_regressed);
    assert!(
        better_score.score < weak_score.score,
        "line search should see sub-threshold temporal-front liveness progress before strict activation gates flip"
    );
    assert!(
        (weak_score.score - better_score.score - 6.0 * LOCAL_FRONT_LIVENESS_SCORE_WEIGHT).abs()
            < 1.0e-6
    );
}

#[test]
fn render_selection_score_rewards_lower_extent_front_liveness_margin() {
    let mut weak = render_selection_case_with_front_liveness_margin(0.0);
    weak.extent_front_liveness = LocalFrontLivenessProgress {
        candidate_count: 4,
        weighted_activation_margin: 8.0,
    };
    let mut better = render_selection_case_with_front_liveness_margin(0.0);
    better.extent_front_liveness = LocalFrontLivenessProgress {
        candidate_count: 4,
        weighted_activation_margin: 2.0,
    };

    let weak_score = render_selection_case_score_with_baseline(7, &weak, None);
    let better_score = render_selection_case_score_with_baseline(7, &better, None);

    assert!(weak_score.morphology_non_regressed);
    assert!(better_score.morphology_non_regressed);
    assert!(
        better_score.score < weak_score.score,
        "line search should see sub-threshold extent-front liveness progress before strict active-extent gates flip"
    );
    assert!(
        (weak_score.score - better_score.score - 6.0 * LOCAL_FRONT_LIVENESS_SCORE_WEIGHT).abs()
            < 1.0e-6
    );
}

#[test]
fn render_selection_score_rewards_lower_temporal_extent_front_liveness_margin() {
    let mut weak = render_selection_case_with_front_liveness_margin(0.0);
    weak.temporal_extent_front_liveness = LocalFrontLivenessProgress {
        candidate_count: 4,
        weighted_activation_margin: 8.0,
    };
    let mut better = render_selection_case_with_front_liveness_margin(0.0);
    better.temporal_extent_front_liveness = LocalFrontLivenessProgress {
        candidate_count: 4,
        weighted_activation_margin: 2.0,
    };

    let weak_score = render_selection_case_score_with_baseline(7, &weak, None);
    let better_score = render_selection_case_score_with_baseline(7, &better, None);

    assert!(weak_score.morphology_non_regressed);
    assert!(better_score.morphology_non_regressed);
    assert!(
        better_score.score < weak_score.score,
        "line search should see temporal extent-front progress before strict active-extent gates flip"
    );
    assert!(
        (weak_score.score - better_score.score - 6.0 * LOCAL_FRONT_LIVENESS_SCORE_WEIGHT).abs()
            < 1.0e-6
    );
}

#[test]
fn render_selection_score_rewards_lower_temporal_activation_schedule_error() {
    let mut abrupt = render_selection_case_with_front_liveness_margin(0.0);
    abrupt.temporal_activation_schedule_error = 0.40;
    let mut staged = render_selection_case_with_front_liveness_margin(0.0);
    staged.temporal_activation_schedule_error = 0.05;

    let abrupt_score = render_selection_case_score_with_baseline(7, &abrupt, None);
    let staged_score = render_selection_case_score_with_baseline(7, &staged, None);

    assert!(abrupt_score.morphology_non_regressed);
    assert!(staged_score.morphology_non_regressed);
    assert!(
        staged_score.score < abrupt_score.score,
        "selection should prefer rollouts whose activation follows the schedule"
    );
    assert!(
        (abrupt_score.score - staged_score.score - 0.35 * TEMPORAL_ACTIVATION_SCORE_WEIGHT).abs()
            < 1.0e-5
    );
}

#[test]
fn render_selection_score_penalizes_active_extent_regression() {
    let baseline_case = render_selection_case_with_front_liveness_margin(0.0);
    let baseline = vec![render_selection_baseline_case_from_metrics(
        7,
        &baseline_case,
    )];
    let mut collapsed = render_selection_case_with_front_liveness_margin(0.0);
    collapsed.extent.bbox_diagonal_ratio = 0.12;
    collapsed.extent.min_axis_extent_ratio = 0.02;

    let scored = render_selection_case_score_with_baseline(7, &collapsed, Some(&baseline));

    assert!(!scored.morphology_non_regressed);
    assert!(
        scored.score > collapsed.score + 1.0,
        "active extent regression should contribute to selection penalty"
    );
}
