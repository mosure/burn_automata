use super::*;

#[test]
fn mesh_axis_expansion_gains_follow_target_bounds() {
    let gains = mesh_axis_expansion_gains(&uv_torus_mesh_target(0.72), 0.05);
    assert!(gains[0] > gains[2]);
    assert!(gains[1] > gains[2]);
    assert!(gains.iter().all(|gain| gain.is_finite() && *gain > 0.0));
}

#[test]
fn torus_angular_coverage_distinguishes_full_support_from_arc_collapse() {
    let config = NpaConfig::growing_3dgs();
    let scale = 0.72;
    let rings = 12;
    let tubes = 8;
    let mut full_positions = Vec::new();
    for ring in 0..rings {
        for tube in 0..tubes {
            full_positions.push(torus_angular_sample_position(
                scale, ring, rings, tube, tubes,
            ));
        }
    }
    let mut full_states = vec![0.0_f32; full_positions.len() * config.state_dims];
    for state in full_states.chunks_exact_mut(config.state_dims) {
        state[3] = 0.0;
    }
    let full = torus_angular_coverage_report(
        &full_positions,
        &full_states,
        config.state_dims,
        scale,
        1.0e-5,
        rings,
        tubes,
    );
    assert_eq!(full.covered_joint_bins, rings * tubes);
    assert_eq!(full.max_ring_gap_bins, 0);
    assert_eq!(full.max_tube_gap_bins, 0);

    let arc_positions = full_positions[..tubes].to_vec();
    let mut arc_states = vec![0.0_f32; arc_positions.len() * config.state_dims];
    for state in arc_states.chunks_exact_mut(config.state_dims) {
        state[3] = 0.0;
    }
    let arc = torus_angular_coverage_report(
        &arc_positions,
        &arc_states,
        config.state_dims,
        scale,
        0.05,
        rings,
        tubes,
    );
    assert!(arc.ring_coverage_fraction < 0.25, "{arc:?}");
    assert_eq!(arc.tube_coverage_fraction, 1.0);
    assert!(arc.max_ring_gap_bins >= rings - 2, "{arc:?}");
}

#[test]
fn active_surface_tail_report_ignores_inactive_and_tracks_opacity_weighted_tail() {
    let config = NpaConfig::growing_3dgs();
    let scale = 0.72;
    let target = uv_torus_mesh_target(scale);
    let on_surface = uv_torus_sample(0, 16, scale).position;
    let positions = vec![
        [on_surface[0], on_surface[1], on_surface[2], 1.0],
        [3.0, 0.0, 0.0, 1.0],
        [-3.0, 0.0, 0.0, 1.0],
    ];
    let mut states = vec![0.0_f32; positions.len() * config.state_dims];
    states[3] = 4.0;
    states[config.state_dims + 3] = -0.5;
    states[2 * config.state_dims + 3] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;

    let report = growth_3d_active_surface_tail_report(
        &positions,
        &states,
        config.state_dims,
        &target,
        GROWTH_3D_SURFACE_MAX_DISTANCE,
    );
    assert_eq!(report.count, 2);
    assert_eq!(report.over_threshold_count, 1);
    assert!((report.over_threshold_fraction - 0.5).abs() <= 1.0e-6);
    assert!(report.p95_distance >= GROWTH_3D_SURFACE_MAX_DISTANCE);
    assert!(report.p99_distance >= report.p95_distance);
    assert!(
        report.opacity_weighted_over_threshold_fraction < report.over_threshold_fraction,
        "{report:?}"
    );
}

#[test]
fn material_visible_surface_tail_report_tracks_render_material_not_liveness() {
    let config = NpaConfig::growing_3dgs();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let scale = 0.72;
    let target = uv_torus_mesh_target(scale);
    let on_surface = uv_torus_sample(0, 16, scale).position;
    let positions = vec![
        [on_surface[0], on_surface[1], on_surface[2], 1.0],
        [3.0, 0.0, 0.0, 1.0],
        [-3.0, 0.0, 0.0, 1.0],
    ];
    let mut states = vec![0.0_f32; positions.len() * config.state_dims];
    for row in states.chunks_exact_mut(config.state_dims) {
        row[3] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
        row[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    }
    states[material_channel] = 4.0;
    states[config.state_dims + material_channel] = 4.0;

    let active = growth_3d_active_surface_tail_report(
        &positions,
        &states,
        config.state_dims,
        &target,
        GROWTH_3D_SURFACE_MAX_DISTANCE,
    );
    let material_visible = growth_3d_material_visible_surface_tail_report(
        &positions,
        &states,
        config.state_dims,
        &target,
        GROWTH_3D_SURFACE_MAX_DISTANCE,
    );

    assert_eq!(active.count, 0);
    assert_eq!(material_visible.count, 2);
    assert_eq!(material_visible.over_threshold_count, 1);
    assert!(material_visible.p99_distance >= GROWTH_3D_SURFACE_MAX_DISTANCE);
}

#[test]
fn material_visible_surface_tail_strict_check_rejects_render_visible_escape() {
    let mut checks = passing_growth_3d_strict_checks();
    let escaped = Growth3dSurfaceTailReport {
        p99_distance: GROWTH_3D_SURFACE_MAX_DISTANCE + 0.10,
        over_threshold_count: 8,
        over_threshold_fraction: 0.25,
        opacity_weighted_over_threshold_fraction: 0.20,
        ..passing_growth_3d_surface_tail_report()
    };

    apply_material_visible_surface_tail_strict_check(&mut checks, escaped);

    assert!(!checks.passed);
    assert!(!checks.material_visible_surface_tail_bounded);
    assert!(
        checks
            .failure_reasons
            .contains(&"material_visible_surface_tail_bounded")
    );

    let mut score = growth_3d_strict_score_report(
        &checks,
        Growth3dSurfaceStats {
            mean_distance: 0.20,
            max_distance: 0.30,
        },
        Growth3dSurfaceStats {
            mean_distance: 0.10,
            max_distance: 0.20,
        },
        passing_growth_3d_surface_tail_report(),
        TargetCoverageStats {
            mean_distance: 0.20,
            max_distance: 0.30,
            covered_fraction: 0.80,
        },
        TargetCoverageStats {
            mean_distance: 0.10,
            max_distance: 0.20,
            covered_fraction: 0.80,
        },
        TargetCoverageStats {
            mean_distance: 0.10,
            max_distance: 0.20,
            covered_fraction: 0.80,
        },
        &passing_surface_normal_coverage_report(),
        &passing_surface_normal_coverage_report(),
        passing_growth_3d_extent_report(),
        0.72,
        &synthetic_render_loss(0.1, 20.0, 20.0, 20.0),
        GaussianVolumeStats::default(),
        None,
    );
    apply_material_visible_surface_tail_strict_score(&mut score, escaped);

    assert!(score.material_visible_surface_tail_p99_penalty > 0.0);
    assert!(score.material_visible_surface_tail_fraction_penalty > 0.0);
}
