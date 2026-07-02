use super::*;

#[test]
fn render_proxy_gradient_rows_cover_full_cloud_instead_of_prefix_only() {
    assert_eq!(
        render_proxy_gradient_row_indices(1024, 8),
        vec![0, 128, 256, 384, 512, 640, 768, 896]
    );
    assert_eq!(render_proxy_gradient_row_indices(4, 8), vec![0, 1, 2, 3]);
    assert_eq!(render_proxy_gradient_row_indices(1024, 1), vec![0]);
}

#[test]
fn trajectory_render_sample_indices_cover_late_rollout_evenly() {
    assert_eq!(trajectory_render_sample_indices(0, 4), Vec::<usize>::new());
    assert_eq!(trajectory_render_sample_indices(8, 0), Vec::<usize>::new());
    assert_eq!(trajectory_render_sample_indices(8, 3), vec![1, 4, 7]);
    assert_eq!(trajectory_render_sample_indices(4, 16), vec![0, 1, 2, 3]);
}

#[test]
fn trajectory_liveness_sample_indices_cover_early_rollout() {
    assert_eq!(
        trajectory_liveness_sample_indices(0, 4),
        Vec::<usize>::new()
    );
    assert_eq!(
        trajectory_liveness_sample_indices(4, 2),
        vec![0, 1, 2, 3],
        "short rollouts should expose every temporal transition to liveness scheduling"
    );
    let long = trajectory_liveness_sample_indices(32, 4);
    assert_eq!(&long[..2], &[0, 1]);
    assert_eq!(long.last().copied(), Some(31));
    assert!(
        long.len() <= TEMPORAL_LIVENESS_TRAJECTORY_SAMPLE_CAP + 2,
        "long rollout liveness sampling should stay bounded"
    );
}

#[test]
fn temporal_activation_allowed_fraction_matches_progressive_growth_gate() {
    assert!(
        (0.20..0.30).contains(&temporal_activation_target_fraction(0.25)),
        "quarter-rollout activation should start expanding the local front without waking the whole cloud"
    );
    assert!(
        (0.48..0.52).contains(&temporal_activation_target_fraction(0.50)),
        "mid-rollout activation target should align with the strict half-activation gate"
    );
    assert!(
        temporal_activation_allowed_fraction(1.0) < 1.0,
        "the temporal schedule should not treat all-particle activation as a valid final shortcut"
    );
    assert!(
        (0.60..0.70).contains(&temporal_activation_target_fraction(1.0)),
        "final activation should ask for enough active particles to pass strict growth without pushing the whole cloud live"
    );
    assert_eq!(temporal_activation_allowed_fraction(1.0), 0.75);
}

#[test]
fn direct_trajectory_geometry_weight_ramps_without_disabling_late_support() {
    assert!((direct_trajectory_geometry_weight(0.0) - 0.5).abs() <= 1.0e-6);
    assert!((direct_trajectory_geometry_weight(0.5) - 0.75).abs() <= 1.0e-6);
    assert!((direct_trajectory_geometry_weight(1.0) - 1.0).abs() <= 1.0e-6);
}

#[test]
fn temporal_activation_schedule_error_penalizes_burst_growth() {
    let sample = |steps: usize, active_fraction: f32| Growth3dTemporalSampleReport {
        steps,
        active_count: (active_fraction * 32.0).round() as usize,
        active_fraction,
        newly_activated_count: 0,
        final_active_mean_radius: 0.0,
        final_active_max_radius: 0.0,
        mean_displacement: 0.0,
        active_surface: Growth3dSurfaceStats {
            mean_distance: 1.0,
            max_distance: 1.0,
        },
        target_coverage: TargetCoverageStats {
            mean_distance: 1.0,
            max_distance: 1.0,
            covered_fraction: 0.0,
        },
    };
    let report = |fractions: [f32; 4]| Growth3dTemporalReport {
        samples: vec![
            sample(0, fractions[0]),
            sample(1, fractions[1]),
            sample(2, fractions[2]),
            sample(4, fractions[3]),
        ],
        first_growth_step: None,
        half_activation_step: None,
        full_activation_step: None,
        activation_span_steps: 0,
        progressive_activation: false,
        surface_mean_ratio: 1.0,
        target_coverage_mean_ratio: 1.0,
        target_coverage_fraction_delta: 0.0,
        geometry_progressive: false,
    };
    let abrupt = report([0.03, 0.03, 0.53, 1.0]);
    let staged = report([0.03, 0.08, 0.25, 0.65]);

    assert!(
        temporal_activation_schedule_error(&abrupt, 4)
            > temporal_activation_schedule_error(&staged, 4),
        "selection should distinguish burst activation from staged rollout growth"
    );
}

#[test]
fn mesh_rollout_snapshot_steps_include_initial_and_final_when_temporal() {
    assert_eq!(mesh_rollout_snapshot_steps(8, 1), vec![8]);
    assert_eq!(mesh_rollout_snapshot_steps(8, 3), vec![0, 4, 8]);
    assert_eq!(mesh_rollout_snapshot_steps(8, 4), vec![0, 2, 5, 8]);
    assert_eq!(mesh_rollout_snapshot_steps(0, 4), vec![0]);
}

#[test]
fn mesh_rollout_row_indices_keep_sparse_high_signal_rows() {
    let output_dims = 6;
    let particle_count = 32;
    let row_budget = 6;
    let mut target_update = vec![0.0_f32; particle_count * output_dims];
    target_update[17 * output_dims + 3] = 2.0;
    target_update[23 * output_dims] = -1.5;

    let rows = mesh_rollout_row_indices(&target_update, output_dims, particle_count, row_budget);

    assert_eq!(rows.len(), row_budget);
    assert!(
        rows.contains(&17) && rows.contains(&23),
        "sparse front/material rows should not be lost to uniform spread sampling: {rows:?}"
    );
    assert_eq!(
        rows.iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        rows.len()
    );
}
