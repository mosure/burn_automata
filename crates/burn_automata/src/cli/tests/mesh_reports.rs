use super::*;

#[test]
fn local_growth_student_opacity_controller_expands_sparse_growth_front() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let model =
        local_growth_student_model(config.clone(), 13, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let (initial_positions, initial_states) = seed_particles_scaled(
        1,
        128,
        config.state_dims,
        config.spatial_dims,
        RolloutConfig::default().seed,
        ParticleSeed::TorusGrowth3d,
        UV_TORUS_FIELD_SCALE,
    );
    let initial_step = model
        .step_cpu(
            &initial_positions,
            &initial_states,
            1,
            128,
            &grid,
            1.0,
            None,
        )
        .unwrap();
    let mut max_inactive_opacity_ds = f32::MIN;
    for row in 0..128 {
        if initial_states[row * config.state_dims + 3] <= -1.0 {
            max_inactive_opacity_ds =
                max_inactive_opacity_ds.max(initial_step.ds[row * config.state_dims + 3]);
        }
    }
    assert!(
        max_inactive_opacity_ds > 0.1,
        "inactive particles on the active front should receive positive local opacity updates, max={max_inactive_opacity_ds}"
    );
    let trace = run_rollout(
        &model,
        &grid,
        &RolloutConfig {
            particle_count: 128,
            steps: 64,
            update_prob: 1.0,
            seed_scale: UV_TORUS_FIELD_SCALE,
            ..RolloutConfig::default()
        },
        ParticleSeed::TorusGrowth3d,
    )
    .unwrap();

    let active_threshold = -1.0_f32;
    let initial_active = initial_states
        .chunks_exact(config.state_dims)
        .filter(|state| state[3] > active_threshold)
        .count();
    let final_active = trace
        .states
        .chunks_exact(config.state_dims)
        .filter(|state| state[3] > active_threshold)
        .count();
    let max_opacity = trace
        .states
        .chunks_exact(config.state_dims)
        .map(|state| state[3])
        .fold(f32::MIN, f32::max);
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let initial_material_mean = initial_states
        .chunks_exact(config.state_dims)
        .map(|state| state[material_channel])
        .sum::<f32>()
        / 128.0;
    let final_material_mean = trace
        .states
        .chunks_exact(config.state_dims)
        .map(|state| state[material_channel])
        .sum::<f32>()
        / trace.particle_count as f32;

    assert!(
        final_active > initial_active,
        "front controller should activate more particles, initial={initial_active} final={final_active}"
    );
    assert!(
        final_active < trace.particle_count,
        "front controller should not activate the whole cloud in one global sweep, final={final_active}"
    );
    assert!(
        max_opacity < UV_TORUS_FIELD_OPACITY_TARGET + 0.5,
        "front opacity should remain bounded, max opacity={max_opacity}"
    );
    assert!(
        final_material_mean > initial_material_mean + 0.25,
        "material opacity should rise with the local growth front, initial={initial_material_mean} final={final_material_mean}"
    );
}

#[test]
fn active_opacity_retime_leaves_dormant_particles_untouched() {
    let config = NpaConfig::growing_3dgs();
    let mut model = NpaModel {
        config: config.clone(),
        weights: NpaWeights::zeros(&config),
    };
    let gain = 0.035;
    retime_growth_3d_active_opacity_model(&mut model, Some(32), gain).unwrap();

    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let opacity_out = config.spatial_dims + 3;
    let mut features = vec![0.0_f32; 3 * input_dims];
    features[3] = -3.0;
    features[input_dims + 3] = -0.5;
    features[2 * input_dims + 3] = 2.0;
    let update = model.forward_update_from_features(&features).unwrap();

    assert!(update[opacity_out].abs() < 1.0e-6);
    assert!((update[output_dims + opacity_out] - gain * 0.5).abs() < 1.0e-6);
    assert!((update[2 * output_dims + opacity_out] - gain).abs() < 1.0e-6);
}

#[test]
fn opacity_bias_retime_only_offsets_opacity_output_bias() {
    let mut model = NpaModel::seeded(NpaConfig::growing_3dgs(), 11);
    let before = model.weights.b2.clone();
    let opacity_out = model.config.spatial_dims + 3;
    add_growth_3d_opacity_update_bias(&mut model, 0.0125).unwrap();
    for (idx, (&current, &initial)) in model.weights.b2.iter().zip(before.iter()).enumerate() {
        if idx == opacity_out {
            assert!((current - initial - 0.0125).abs() <= 1.0e-7);
        } else {
            assert_eq!(current, initial);
        }
    }

    let mut position_field = NpaModel::seeded(NpaConfig::torus_field_3dgs(), 11);
    assert!(add_growth_3d_opacity_update_bias(&mut position_field, 0.01).is_err());
}

#[test]
fn material_opacity_bias_retime_only_offsets_material_output_bias() {
    let mut model = NpaModel::seeded(NpaConfig::growing_3dgs(), 11);
    let before = model.weights.b2.clone();
    let material_channel = growth_3d_material_opacity_channel(model.config.state_dims).unwrap();
    let material_opacity_out = model.config.spatial_dims + material_channel;
    let liveness_opacity_out = model.config.spatial_dims + 3;
    add_growth_3d_material_opacity_update_bias(&mut model, 0.0125).unwrap();
    for (idx, (&current, &initial)) in model.weights.b2.iter().zip(before.iter()).enumerate() {
        if idx == material_opacity_out {
            assert!((current - initial - 0.0125).abs() <= 1.0e-7);
        } else {
            assert_eq!(current, initial);
        }
    }
    assert_eq!(
        model.weights.b2[liveness_opacity_out],
        before[liveness_opacity_out]
    );

    let mut position_field = NpaModel::seeded(NpaConfig::torus_field_3dgs(), 11);
    assert!(add_growth_3d_material_opacity_update_bias(&mut position_field, 0.01).is_err());
}

#[test]
fn local_front_opacity_targets_activate_only_near_active_neighbors() {
    let config = NpaConfig::growing_3dgs();
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; 3 * config.state_dims];
    states[3] = 0.0;
    states[config.state_dims + 3] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + 3] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.8_f32, 0.0, 0.0, 0.0],
    ];

    let updates = local_front_opacity_targets(
        &config,
        &positions,
        &states,
        LOCAL_GROWTH_FRONT_OPACITY_GAIN,
        0.20,
        LOCAL_GROWTH_FRONT_MAX_OPACITY_UPDATE,
    );

    assert!(
        updates[1] > 0.0,
        "inactive particle near an active neighbor should receive positive opacity update"
    );
    assert!(
        updates[2].abs() < 1.0e-6,
        "far inactive particle should stay dormant until the front reaches it"
    );
}

#[test]
fn front_motion_gate_suppresses_far_dormant_mesh_targets() {
    let config = NpaConfig::growing_3dgs();
    let mut states = vec![0.0; 3 * config.state_dims];
    states[3] = 0.0;
    states[config.state_dims + 3] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + 3] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.8_f32, 0.0, 0.0, 0.0],
    ];
    let output_dims = config.update_dims();
    let target = uv_torus_mesh_target(0.72);

    let ungated = mesh_field_target_update_for_rows(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        f32::INFINITY,
        0.0,
        1.0,
        0.0,
        0.0,
        0.20,
        0.0,
        false,
    );
    let gated = mesh_field_target_update_for_rows(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        f32::INFINITY,
        0.0,
        1.0,
        0.0,
        LOCAL_GROWTH_FRONT_OPACITY_GAIN,
        0.20,
        LOCAL_GROWTH_FRONT_MAX_OPACITY_UPDATE,
        true,
    );
    let opacity_gated = mesh_field_target_update_for_rows(
        &config,
        &target,
        &positions,
        &states,
        0.0,
        f32::INFINITY,
        0.0,
        0.0,
        UV_TORUS_FIELD_OPACITY_GAIN,
        LOCAL_GROWTH_FRONT_OPACITY_GAIN,
        0.20,
        LOCAL_GROWTH_FRONT_MAX_OPACITY_UPDATE,
        true,
    );

    let far_base = 2 * output_dims;
    let far_ungated_motion =
        (ungated[far_base].powi(2) + ungated[far_base + 1].powi(2) + ungated[far_base + 2].powi(2))
            .sqrt();
    let far_gated_motion =
        (gated[far_base].powi(2) + gated[far_base + 1].powi(2) + gated[far_base + 2].powi(2))
            .sqrt();
    let near_base = output_dims;
    let near_gated_motion =
        (gated[near_base].powi(2) + gated[near_base + 1].powi(2) + gated[near_base + 2].powi(2))
            .sqrt();
    let opacity_out = config.spatial_dims + 3;
    let far_gated_opacity = opacity_gated[far_base + opacity_out];
    let near_gated_opacity = opacity_gated[near_base + opacity_out];

    assert!(
        far_ungated_motion > 1.0e-4,
        "fixture should have a nonzero target motion without front gating"
    );
    assert!(
        far_gated_motion < 1.0e-6,
        "far dormant particle should not receive target motion before the active front reaches it"
    );
    assert!(
        near_gated_motion > 1.0e-4,
        "near-front inactive particle should still receive gated target motion"
    );
    assert!(
        far_gated_opacity.abs() < 1.0e-6,
        "far dormant particle should not receive direct opacity target before the active front reaches it"
    );
    assert!(
        near_gated_opacity > 0.0,
        "near-front inactive particle should receive front-gated opacity growth"
    );
}

#[test]
fn mesh_opacity_targets_surface_material_instead_of_whole_domain() {
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
        [0.0_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![0.0; 2 * config.state_dims];
    let material_opacity_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    states[3] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[material_opacity_channel] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[config.state_dims + 3] = 0.0;
    states[config.state_dims + material_opacity_channel] = 0.0;

    let updates = mesh_field_target_update_for_rows(
        &config,
        &target,
        &positions,
        &states,
        0.0,
        f32::INFINITY,
        0.0,
        0.0,
        UV_TORUS_FIELD_OPACITY_GAIN,
        0.0,
        0.20,
        LOCAL_GROWTH_FRONT_MAX_OPACITY_UPDATE,
        false,
    );
    let opacity_out = config.spatial_dims + material_opacity_channel;

    assert!(
        updates[opacity_out] > 0.0,
        "near-surface dormant material should receive positive render opacity pressure"
    );
    assert!(
        updates[config.update_dims() + opacity_out] < 0.0,
        "off-surface active material should be suppressed instead of making the whole substrate visible"
    );
}

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

#[test]
fn mesh_local_rollout_rows_reject_position_field_models() {
    let config = NpaConfig::torus_field_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let model = NpaModel::seeded(config, 13);
    assert!(model.config.position_features);

    let err = mesh_local_rollout_supervised_batch(
        &model,
        &grid,
        &uv_torus_mesh_target(0.72),
        MeshFieldRolloutBatchConfig {
            max_rows: 16,
            particle_count: 32,
            rollout_steps: 2,
            rollouts: 1,
            temporal_samples: 1,
            seed: 17,
            seed_scale: 0.72,
            seed_mode: ParticleSeed::UniformCircle,
            motion_gain: UV_TORUS_FIELD_MOTION_GAIN,
            max_update_norm: f32::INFINITY,
            coverage_gain: 0.0,
            coverage_samples: 0,
            coverage_mode: CoverageUpdateModeArg::HardNearest,
            coverage_softness: 0.0,
            coverage_repulsion_gain: 0.0,
            coverage_gap_gain: 0.0,
            coverage_repulsion_radius: 0.0,
            coverage_normal_weight: 0.0,
            extent_gain: 0.0,
            color_gain: UV_TORUS_FIELD_COLOR_GAIN,
            aux_state_gain: 1.0,
            opacity_gain: UV_TORUS_FIELD_OPACITY_GAIN,
            front_opacity_gain: 0.0,
            front_radius: 0.0,
            front_max_opacity_update: 0.0,
            front_motion_gate: false,
            preserve_opacity_update: false,
        },
    )
    .unwrap_err();

    assert!(err.to_string().contains("position_features=false"));
}

#[test]
fn torus_robustness_report_rejects_static_opacity_only_prior() {
    let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing3dGs);
    let mut weights = NpaWeights::zeros(&config);
    weights.b2[config.spatial_dims + 3] = UV_TORUS_OPACITY_GROWTH_DELTA;
    let model = NpaModel {
        config: config.clone(),
        weights,
    };
    let report = torus_robustness_report_for_cases(
        &model,
        &grid,
        &[TorusRobustnessCaseConfig {
            particle_count: 64,
            steps: 4,
            seed: 11,
            seed_scale: 0.72,
            seed_mode: ParticleSeed::UvTorusDense3d,
        }],
    )
    .unwrap();

    assert!(!report.passed);
    assert!(report.max_motion_per_step <= 1.0e-6);
    assert!(report.max_target_position_error > 0.1);
    assert!(report.max_opacity_target_error <= 1.0e-5);
    assert_eq!(report.cases.len(), 1);
}

#[test]
fn torus_robustness_report_accepts_residual_motion_growth_prior() {
    let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing3dGs);
    let model = torus_growth_model(config).unwrap();
    let report = torus_robustness_report_for_cases(
        &model,
        &grid,
        &[TorusRobustnessCaseConfig {
            particle_count: 128,
            steps: 180,
            seed: 11,
            seed_scale: 0.72,
            seed_mode: ParticleSeed::UvTorusDense3d,
        }],
    )
    .unwrap();

    assert!(
        report.passed,
        "target_position={} surface={} color={} opacity={} first_motion={} max_motion={}",
        report.max_target_position_error,
        report.max_torus_surface_error,
        report.max_color_target_error,
        report.max_opacity_target_error,
        report.first_motion_per_step,
        report.max_motion_per_step
    );
    assert!(report.first_motion_per_step >= 1.0e-3);
    assert!(report.max_motion_per_step >= 1.0e-3);
    assert!(report.max_target_position_error <= 1.2e-1);
    assert_eq!(report.cases.len(), 1);
}

#[test]
fn torus_robustness_report_accepts_seed_frame_morphogen_prior() {
    let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing3dGs);
    let model = torus_morphogen_model(config).unwrap();
    assert!(!model.config.position_features);
    let report = torus_robustness_report_for_cases(
        &model,
        &grid,
        &[TorusRobustnessCaseConfig {
            particle_count: 128,
            steps: 180,
            seed: 11,
            seed_scale: 0.72,
            seed_mode: ParticleSeed::TorusMorphogenDense3d,
        }],
    )
    .unwrap();

    assert!(
        report.passed,
        "target_position={} surface={} color={} opacity={} first_motion={} max_motion={}",
        report.max_target_position_error,
        report.max_torus_surface_error,
        report.max_color_target_error,
        report.max_opacity_target_error,
        report.first_motion_per_step,
        report.max_motion_per_step
    );
    assert!(report.first_motion_per_step >= 1.0e-3);
    assert!(report.max_motion_per_step >= 1.0e-3);
    assert!(report.max_target_position_error <= 1.2e-1);
    assert!(report.max_color_target_error <= 2.0e-2);
    assert_eq!(report.cases.len(), 1);
}

#[test]
fn mesh_rollout_report_rejects_static_teapot_field_prior() {
    let config = NpaConfig::torus_field_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let model = NpaModel {
        config: config.clone(),
        weights: NpaWeights::zeros(&config),
    };
    let report = mesh_rollout_report_for_cases(
        &model,
        &grid,
        &utah_teapot_mesh_target(0.72),
        &[MeshRolloutCaseConfig {
            particle_count: 64,
            steps: 4,
            seed: 11,
            seed_scale: 0.72,
            seed_mode: ParticleSeed::TeapotFieldDense3d,
        }],
    )
    .unwrap();

    assert!(!report.passed);
    assert!(report.max_motion_per_step <= 1.0e-6);
    assert!(report.min_final_opacity <= UV_TORUS_INITIAL_OPACITY_LOGIT);
    assert_eq!(report.cases.len(), 1);
}

#[test]
fn mesh_rollout_report_rejects_static_conditionless_local_prior() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let model = NpaModel {
        config: config.clone(),
        weights: NpaWeights::zeros(&config),
    };
    let report = mesh_rollout_report_for_cases(
        &model,
        &grid,
        &uv_torus_mesh_target(0.72),
        &[MeshRolloutCaseConfig {
            particle_count: 64,
            steps: 4,
            seed: 11,
            seed_scale: 0.72,
            seed_mode: ParticleSeed::UniformCircle,
        }],
    )
    .unwrap();

    assert!(!report.passed);
    assert!(report.max_motion_per_step <= 1.0e-6);
    assert_eq!(report.cases.len(), 1);
}

#[test]
fn mesh_target_update_trains_oriented_state_from_neutral_seed() {
    let config = NpaConfig::growing_3dgs();
    let target = uv_torus_mesh_target(0.72);
    let positions = vec![[0.1_f32, 0.0, 0.0, 0.0]];
    let states = vec![0.0; config.state_dims];
    let update = mesh_field_target_update_for_rows(
        &config, &target, &positions, &states, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, false,
    );
    let base = config.spatial_dims;
    let coordinate_norm =
        (update[base].powi(2) + update[base + 1].powi(2) + update[base + 2].powi(2)).sqrt();
    let normal_update = [
        update[base + UV_TORUS_NORMAL_STATE_OFFSET],
        update[base + UV_TORUS_NORMAL_STATE_OFFSET + 1],
        update[base + UV_TORUS_NORMAL_STATE_OFFSET + 2],
    ];
    let normal_norm =
        (normal_update[0].powi(2) + normal_update[1].powi(2) + normal_update[2].powi(2)).sqrt();
    assert!(coordinate_norm > 1.0e-4);
    assert!(normal_norm > 1.0e-4);
    assert!(update[base + UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET].abs() > 1.0e-4);
}

#[test]
fn mesh_target_update_can_disable_projection_aux_state_targets() {
    let config = NpaConfig::growing_3dgs();
    let target = uv_torus_mesh_target(0.72);
    let positions = vec![[0.1_f32, 0.0, 0.0, 0.0]];
    let states = vec![0.0; config.state_dims];
    let update = mesh_field_target_update_for_rows(
        &config, &target, &positions, &states, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false,
    );
    let base = config.spatial_dims;

    for channel in [
        0,
        1,
        2,
        UV_TORUS_NORMAL_STATE_OFFSET,
        UV_TORUS_NORMAL_STATE_OFFSET + 1,
        UV_TORUS_NORMAL_STATE_OFFSET + 2,
        UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET,
    ] {
        assert_eq!(update[base + channel], 0.0);
    }
}

#[test]
fn torus_morphogen_supervision_writes_oriented_mesh_channels() {
    let config = NpaConfig::growing_3dgs();
    let rows = 32;
    let batch = torus_morphogen_supervised_batch(&config, rows);
    let input_dims = config.perception_dims();
    let blur_offset = config.state_dims;

    for row in 0..rows {
        let base = row * input_dims;
        let normal = [
            batch.features[base + UV_TORUS_NORMAL_STATE_OFFSET],
            batch.features[base + UV_TORUS_NORMAL_STATE_OFFSET + 1],
            batch.features[base + UV_TORUS_NORMAL_STATE_OFFSET + 2],
        ];
        let signed_distance = batch.features[base + UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET];
        let normal_len =
            (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        assert!((normal_len - 1.0).abs() < 1.0e-4);
        assert!(signed_distance.is_finite());
        assert!(signed_distance.abs() <= 1.5);
        for channel in 0..config.state_dims {
            assert_eq!(
                batch.features[base + channel],
                batch.features[base + blur_offset + channel]
            );
        }
    }
}
