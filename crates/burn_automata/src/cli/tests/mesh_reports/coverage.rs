use super::*;

#[test]
fn target_extent_updates_push_active_bounds_outward() {
    let config = NpaConfig::growing_3dgs();
    let positions = vec![
        [-0.10_f32, 0.0, 0.0, 0.0],
        [0.10_f32, 0.0, 0.0, 0.0],
        [0.0_f32, 0.0, 0.0, 0.0],
    ];
    let mut target_update = vec![0.0; positions.len() * config.update_dims()];
    let target = uv_torus_mesh_target(0.72);

    add_target_extent_updates_for_rows(
        &config,
        &target,
        &positions,
        None,
        &mut target_update,
        0.10,
        0.25,
        0.30,
    );

    let output_dims = config.update_dims();
    assert!(
        target_update[0] < -1.0e-4,
        "min-x active boundary should be pushed toward target min x"
    );
    assert!(
        target_update[output_dims] > 1.0e-4,
        "max-x active boundary should be pushed toward target max x"
    );
    assert!(
        target_update[2 * output_dims].abs() < target_update[output_dims].abs(),
        "center row should receive less x extent pressure than boundary row"
    );
}

#[test]
fn active_target_coverage_ignores_inactive_particles() {
    let config = NpaConfig::growing_3dgs();
    let target = uv_torus_mesh_target(0.72);
    let sample = target.surface_sample(0);
    let positions = vec![
        [
            sample.position[0],
            sample.position[1],
            sample.position[2],
            0.0,
        ],
        [0.0, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![0.0; 2 * config.state_dims];
    states[3] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[config.state_dims + 3] = 0.0;

    let all = target_coverage_stats(&positions, &target, 16, 0.20);
    let active =
        active_target_coverage_stats(&positions, &states, config.state_dims, &target, 16, 0.20);

    assert!(
        all.covered_fraction > active.covered_fraction,
        "inactive particle exactly on target surface should not count toward active coverage"
    );
}

#[test]
fn material_visible_target_coverage_requires_visible_material() {
    let config = NpaConfig::growing_3dgs();
    let target = uv_torus_mesh_target(0.72);
    let sample = target.surface_sample(0);
    let positions = vec![
        [
            sample.position[0],
            sample.position[1],
            sample.position[2],
            0.0,
        ],
        [0.0, 0.0, 0.0, 0.0],
    ];
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let mut states = vec![0.0; 2 * config.state_dims];
    states[material_channel] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[config.state_dims + material_channel] = 0.0;

    let active =
        active_target_coverage_stats(&positions, &states, config.state_dims, &target, 16, 0.20);
    let visible = material_visible_target_coverage_stats(
        &positions,
        &states,
        config.state_dims,
        &target,
        16,
        0.20,
    );

    assert!(
        active.covered_fraction > visible.covered_fraction,
        "live but material-transparent particles should not count toward material-visible coverage"
    );
}

#[test]
fn material_liveness_report_detects_dormant_render_visible_material() {
    let config = NpaConfig::growing_3dgs();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let mut states = vec![0.0; 3 * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    states[config.state_dims + material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;

    let report = growth_3d_material_liveness_report(&states, config.state_dims);

    assert_eq!(report.material_visible_count, 2);
    assert_eq!(report.inactive_material_visible_count, 1);
    assert_eq!(report.inactive_material_visible_fraction, 0.5);
    assert!(!report.passed);
}

#[test]
fn material_liveness_strict_score_tracks_inactive_visible_material() {
    let mut score = growth_3d_strict_score_report(
        &passing_growth_3d_strict_checks(),
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
    );
    let base_score = score.score;
    let material_liveness = Growth3dMaterialLivenessReport {
        material_visible_count: 4,
        inactive_material_visible_count: 1,
        inactive_material_visible_fraction: 0.25,
        inactive_material_logit_threshold: 1.0,
        max_inactive_material_opacity: 6.0,
        passed: false,
    };

    apply_material_liveness_strict_score(&mut score, material_liveness);

    assert_eq!(score.material_visible_inactive_fraction, 0.25);
    assert_eq!(score.material_visible_inactive_fraction_penalty, 2.5);
    assert_eq!(score.material_visible_max_inactive_opacity, 6.0);
    assert_eq!(score.material_visible_max_inactive_opacity_penalty, 0.5);
    assert!(score.score > base_score);
}

#[test]
fn surface_coverage_profile_reports_sparse_target_support() {
    let target = uv_torus_mesh_target(0.72);
    let sample = target.surface_sample(0);
    let positions = vec![[
        sample.position[0],
        sample.position[1],
        sample.position[2],
        0.0,
    ]];
    let sparse = surface_coverage_profile(&positions, &target, 128, 0.05, 16);
    let empty = surface_coverage_profile(&[], &target, 128, 0.05, 16);

    assert!(sparse.covered_fraction > 0.0);
    assert!(sparse.covered_bin_fraction < 1.0);
    assert!(sparse.empty_bins > 0);
    assert_eq!(empty.covered_fraction, 0.0);
    assert_eq!(empty.assigned_particle_fraction, 0.0);
}
