use super::*;

#[test]
fn mesh_local_rollout_rows_do_not_require_position_features() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let model = NpaModel::seeded(config.clone(), 13);
    assert!(!model.config.position_features);

    let batch = mesh_local_rollout_supervised_batch(
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
    .unwrap();

    assert_eq!(batch.features.len(), 16 * config.perception_dims());
    assert_eq!(batch.target_update.len(), 16 * config.update_dims());
}

#[test]
fn rollout_local_growth_seed_uses_mesh_objective_not_static_residual_teacher() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let seed_scale = UV_TORUS_FIELD_SCALE;
    let particle_count = 64;
    let output_dims = config.update_dims();
    let residual_teacher = torus_morphogen_model(config.clone()).unwrap();
    let static_teacher_batch = rollout_supervised_batch_from_model(
        &residual_teacher,
        &residual_teacher,
        &grid,
        SupervisedTarget::Teacher(&residual_teacher),
        RolloutSupervisionConfig {
            max_rows: particle_count,
            particle_count,
            rollout_steps: 1,
            rollouts: 1,
            update_prob: 1.0,
            seed: 0x70_75,
            seed_scale,
            seed_mode: ParticleSeed::TorusGrowth3d,
            ..RolloutSupervisionConfig::default()
        },
    )
    .unwrap();
    let max_static_teacher_motion = static_teacher_batch
        .target_update
        .chunks_exact(output_dims)
        .map(|row| (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt())
        .fold(0.0_f32, f32::max);

    let target = uv_torus_mesh_target(seed_scale);
    let local_student = local_growth_student_model_with_axis_gains(
        config.clone(),
        0x70_75,
        0.0,
        mesh_axis_expansion_gains(&target, LOCAL_GROWTH_EXPANSION_GAIN),
    )
    .unwrap();
    let mesh_objective_batch = mesh_local_rollout_supervised_batch(
        &local_student,
        &grid,
        &target,
        MeshFieldRolloutBatchConfig {
            max_rows: particle_count,
            particle_count,
            rollout_steps: 1,
            rollouts: 1,
            temporal_samples: 1,
            seed: 0x70_75,
            seed_scale,
            seed_mode: ParticleSeed::TorusGrowth3d,
            motion_gain: LOCAL_TORUS_MOTION_GAIN,
            max_update_norm: 0.25,
            coverage_gain: 5.0e-2,
            coverage_samples: 0,
            coverage_mode: CoverageUpdateModeArg::HardNearest,
            coverage_softness: 0.0,
            coverage_repulsion_gain: 0.0,
            coverage_gap_gain: 0.0,
            coverage_repulsion_radius: 0.0,
            coverage_normal_weight: 0.0,
            extent_gain: LOCAL_GROWTH_EXTENT_GAIN,
            color_gain: UV_TORUS_FIELD_COLOR_GAIN,
            aux_state_gain: 1.0,
            opacity_gain: 0.0,
            front_opacity_gain: LOCAL_GROWTH_FRONT_OPACITY_GAIN,
            front_radius: LOCAL_GROWTH_FRONT_RADIUS,
            front_max_opacity_update: LOCAL_GROWTH_FRONT_MAX_OPACITY_UPDATE,
            front_motion_gate: true,
            preserve_opacity_update: false,
        },
    )
    .unwrap();
    let max_mesh_objective_motion = mesh_objective_batch
        .target_update
        .chunks_exact(output_dims)
        .map(|row| (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt())
        .fold(0.0_f32, f32::max);

    assert!(
        max_static_teacher_motion < max_mesh_objective_motion * 0.5,
        "residual teacher should stay weaker than rollout-local mesh supervision with seed-coordinate scaffolds, residual={max_static_teacher_motion} mesh={max_mesh_objective_motion}"
    );
    assert!(
        max_mesh_objective_motion > 1.0e-3,
        "rollout-local mesh objective should produce nonzero motion targets from neutral growth seeds, got {max_mesh_objective_motion}"
    );
}

#[test]
fn mesh_local_rollout_rows_keep_full_cloud_perception_context() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let model = NpaModel::seeded(config.clone(), 13);
    let rollout_cfg = RolloutConfig {
        particle_count: 32,
        steps: 2,
        update_prob: 1.0,
        seed: 17,
        seed_scale: 0.72,
        ..RolloutConfig::default()
    };
    let trace = run_rollout(&model, &grid, &rollout_cfg, ParticleSeed::UniformCircle).unwrap();
    let full_step = model
        .step_cpu(
            &trace.positions,
            &trace.states,
            1,
            trace.particle_count,
            &grid,
            1.0,
            None,
        )
        .unwrap();

    let batch = mesh_local_rollout_supervised_batch(
        &model,
        &grid,
        &uv_torus_mesh_target(0.72),
        MeshFieldRolloutBatchConfig {
            max_rows: 16,
            particle_count: rollout_cfg.particle_count,
            rollout_steps: rollout_cfg.steps,
            rollouts: 1,
            temporal_samples: 1,
            seed: rollout_cfg.seed,
            seed_scale: rollout_cfg.seed_scale,
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
    .unwrap();

    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let target = uv_torus_mesh_target(0.72);
    let target_update = mesh_field_target_update_for_rows(
        &config,
        &target,
        &trace.positions,
        &trace.states,
        UV_TORUS_FIELD_MOTION_GAIN,
        f32::INFINITY,
        UV_TORUS_FIELD_COLOR_GAIN,
        1.0,
        UV_TORUS_FIELD_OPACITY_GAIN,
        0.0,
        0.0,
        0.0,
        false,
    );
    let row_indices =
        mesh_rollout_row_indices(&target_update, output_dims, rollout_cfg.particle_count, 16);
    assert_eq!(row_indices.len(), 16);
    for (batch_row, full_row) in row_indices.iter().copied().enumerate() {
        let batch_base = batch_row * input_dims;
        let full_base = full_row * input_dims;
        assert_eq!(
            &batch.features[batch_base..batch_base + input_dims],
            &full_step.perception.features[full_base..full_base + input_dims]
        );
    }
}

#[test]
fn mesh_local_temporal_rollout_rows_include_initial_snapshot() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let model = NpaModel::seeded(config.clone(), 13);
    let seed = 17;
    let seed_scale = 0.72;
    let (positions, states) = seed_particles_scaled(
        1,
        16,
        config.state_dims,
        config.spatial_dims,
        seed,
        ParticleSeed::TorusGrowth3d,
        seed_scale,
    );
    let initial_step = model
        .step_cpu(&positions, &states, 1, 16, &grid, 1.0, None)
        .unwrap();
    let batch = mesh_local_rollout_supervised_batch(
        &model,
        &grid,
        &uv_torus_mesh_target(seed_scale),
        MeshFieldRolloutBatchConfig {
            max_rows: 12,
            particle_count: 16,
            rollout_steps: 4,
            rollouts: 1,
            temporal_samples: 3,
            seed,
            seed_scale,
            seed_mode: ParticleSeed::TorusGrowth3d,
            motion_gain: UV_TORUS_FIELD_MOTION_GAIN,
            max_update_norm: 0.25,
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
    .unwrap();

    let input_dims = config.perception_dims();
    assert_eq!(batch.features.len(), 12 * input_dims);
    assert_eq!(batch.target_update.len(), 12 * config.update_dims());
    assert_eq!(
        &batch.features[..input_dims],
        &initial_step.perception.features[..input_dims],
        "first temporal batch row should come from the initial rollout snapshot"
    );
}

#[test]
fn mesh_local_rollout_can_preserve_opacity_update_targets() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let model = NpaModel::seeded(config.clone(), 13);
    let seed = 17;
    let seed_scale = 0.72;
    let particle_count = 16;
    let (positions, states) = seed_particles_scaled(
        1,
        particle_count,
        config.state_dims,
        config.spatial_dims,
        seed,
        ParticleSeed::TorusGrowth3d,
        seed_scale,
    );
    let initial_step = model
        .step_cpu(&positions, &states, 1, particle_count, &grid, 1.0, None)
        .unwrap();
    let batch = mesh_local_rollout_supervised_batch(
        &model,
        &grid,
        &uv_torus_mesh_target(seed_scale),
        MeshFieldRolloutBatchConfig {
            max_rows: particle_count,
            particle_count,
            rollout_steps: 0,
            rollouts: 1,
            temporal_samples: 1,
            seed,
            seed_scale,
            seed_mode: ParticleSeed::TorusGrowth3d,
            motion_gain: 0.0,
            max_update_norm: 0.25,
            coverage_gain: 0.0,
            coverage_samples: 0,
            coverage_mode: CoverageUpdateModeArg::HardNearest,
            coverage_softness: 0.0,
            coverage_repulsion_gain: 0.0,
            coverage_gap_gain: 0.0,
            coverage_repulsion_radius: 0.0,
            coverage_normal_weight: 0.0,
            extent_gain: 0.0,
            color_gain: 0.0,
            aux_state_gain: 0.0,
            opacity_gain: 0.0,
            front_opacity_gain: 0.0,
            front_radius: 0.0,
            front_max_opacity_update: 0.0,
            front_motion_gate: false,
            preserve_opacity_update: true,
        },
    )
    .unwrap();

    let output_dims = config.update_dims();
    for row in 0..particle_count {
        let target_opacity = batch.target_update[row * output_dims + config.spatial_dims + 3];
        let current_opacity_update = initial_step.ds[row * config.state_dims + 3];
        assert!(
            (target_opacity - current_opacity_update).abs() <= 1.0e-6,
            "preserved opacity target should match current model update for row {row}: target={target_opacity} current={current_opacity_update}"
        );
    }
}

#[test]
fn mesh_local_rollout_rows_can_include_target_coverage_pressure() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let model = NpaModel::seeded(config.clone(), 13);
    let target = uv_torus_mesh_target(0.72);
    let base_cfg = MeshFieldRolloutBatchConfig {
        max_rows: 16,
        particle_count: 32,
        rollout_steps: 2,
        rollouts: 1,
        temporal_samples: 1,
        seed: 17,
        seed_scale: 0.72,
        seed_mode: ParticleSeed::TorusGrowth3d,
        motion_gain: UV_TORUS_FIELD_MOTION_GAIN,
        max_update_norm: 0.25,
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
    };
    let no_coverage =
        mesh_local_rollout_supervised_batch(&model, &grid, &target, base_cfg).unwrap();
    let with_coverage = mesh_local_rollout_supervised_batch(
        &model,
        &grid,
        &target,
        MeshFieldRolloutBatchConfig {
            coverage_gain: 0.15,
            ..base_cfg
        },
    )
    .unwrap();
    let with_dense_coverage = mesh_local_rollout_supervised_batch(
        &model,
        &grid,
        &target,
        MeshFieldRolloutBatchConfig {
            coverage_gain: 0.15,
            coverage_samples: 2048,
            ..base_cfg
        },
    )
    .unwrap();
    let with_soft_coverage = mesh_local_rollout_supervised_batch(
        &model,
        &grid,
        &target,
        MeshFieldRolloutBatchConfig {
            coverage_gain: 0.15,
            coverage_samples: 128,
            coverage_mode: CoverageUpdateModeArg::SoftChamfer,
            coverage_softness: 0.12,
            ..base_cfg
        },
    )
    .unwrap();
    let with_gap_coverage = mesh_local_rollout_supervised_batch(
        &model,
        &grid,
        &target,
        MeshFieldRolloutBatchConfig {
            coverage_gain: 0.15,
            coverage_samples: 2048,
            coverage_mode: CoverageUpdateModeArg::SlicedOt,
            coverage_gap_gain: 1.0,
            ..base_cfg
        },
    )
    .unwrap();
    let with_soft_gap_coverage = mesh_local_rollout_supervised_batch(
        &model,
        &grid,
        &target,
        MeshFieldRolloutBatchConfig {
            coverage_gain: 0.15,
            coverage_samples: 2048,
            coverage_mode: CoverageUpdateModeArg::SoftChamfer,
            coverage_softness: 0.12,
            coverage_gap_gain: 1.0,
            ..base_cfg
        },
    )
    .unwrap();

    let output_dims = config.update_dims();
    let position_delta_sum = |lhs: &SupervisedBatch, rhs: &SupervisedBatch| {
        lhs.target_update
            .chunks_exact(output_dims)
            .zip(rhs.target_update.chunks_exact(output_dims))
            .map(|(a, b)| {
                ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
            })
            .sum::<f32>()
    };
    let coverage_delta = position_delta_sum(&no_coverage, &with_coverage);
    let sample_budget_delta = position_delta_sum(&with_coverage, &with_dense_coverage);
    let soft_delta = position_delta_sum(&with_coverage, &with_soft_coverage);
    let gap_delta = position_delta_sum(&with_dense_coverage, &with_gap_coverage);
    let soft_gap_delta = position_delta_sum(&with_soft_coverage, &with_soft_gap_coverage);
    assert!(
        coverage_delta > 1.0e-5,
        "coverage pressure should alter at least one position update"
    );
    assert!(
        sample_budget_delta > 1.0e-6,
        "coverage sample budget should alter the coverage pressure signal"
    );
    assert!(
        soft_delta > 1.0e-6,
        "soft-chamfer coverage should not silently match hard-nearest updates"
    );
    assert!(
        gap_delta > 1.0e-6,
        "surface-gap gain should alter uncovered-surface pressure independently of tangent repulsion"
    );
    assert!(
        soft_gap_delta > 1.0e-6,
        "surface-gap gain should also alter soft/normal-aware coverage updates"
    );
}
