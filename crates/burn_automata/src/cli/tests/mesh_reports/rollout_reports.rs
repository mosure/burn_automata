use super::*;

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
