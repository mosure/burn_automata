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

#[test]
fn surface_tangent_repulsion_separates_close_surface_particles() {
    let target = uv_torus_mesh_target(0.72);
    let sample = target.surface_sample(0);
    let tangent = if sample.normal[0].abs() < 0.9 {
        [0.0, -sample.normal[2], sample.normal[1]]
    } else {
        [-sample.normal[1], sample.normal[0], 0.0]
    };
    let tangent_norm =
        (tangent[0] * tangent[0] + tangent[1] * tangent[1] + tangent[2] * tangent[2]).sqrt();
    let tangent = [
        tangent[0] / tangent_norm,
        tangent[1] / tangent_norm,
        tangent[2] / tangent_norm,
    ];
    let positions = vec![
        [
            sample.position[0] - 0.01 * tangent[0],
            sample.position[1] - 0.01 * tangent[1],
            sample.position[2] - 0.01 * tangent[2],
            1.0,
        ],
        [
            sample.position[0] + 0.01 * tangent[0],
            sample.position[1] + 0.01 * tangent[1],
            sample.position[2] + 0.01 * tangent[2],
            1.0,
        ],
    ];
    let mut updates = vec![[0.0; 3]; positions.len()];
    add_surface_tangent_repulsion_to_updates(
        &target,
        &positions,
        &[0, 1],
        1.0,
        1.0,
        0.08,
        0.72,
        1.0,
        &mut updates,
    );

    let lhs_dot =
        updates[0][0] * -tangent[0] + updates[0][1] * -tangent[1] + updates[0][2] * -tangent[2];
    let rhs_dot =
        updates[1][0] * tangent[0] + updates[1][1] * tangent[1] + updates[1][2] * tangent[2];
    assert!(
        lhs_dot > 0.0 && rhs_dot > 0.0,
        "repulsion should push close particles apart along the surface tangent, updates={updates:?}"
    );
    let projected_normal = target
        .project([positions[0][0], positions[0][1], positions[0][2]])
        .normal;
    assert!(
        (updates[0][0] * projected_normal[0]
            + updates[0][1] * projected_normal[1]
            + updates[0][2] * projected_normal[2])
            .abs()
            < 1.0e-4,
        "repulsion should remove the projected normal component"
    );
}

#[test]
fn surface_gap_relocation_moves_redundant_particles_to_uncovered_regions() {
    let target = uv_torus_mesh_target(0.72);
    let sample = target.surface_sample(0);
    let positions = vec![
        [
            sample.position[0],
            sample.position[1],
            sample.position[2],
            1.0,
        ],
        [
            sample.position[0],
            sample.position[1],
            sample.position[2],
            1.0,
        ],
    ];
    let mut updates = vec![[0.0; 3]; positions.len()];
    add_surface_gap_relocation_to_updates(
        &target,
        &positions,
        &[0, 1],
        1.0,
        1.0,
        512,
        0.0,
        0.72,
        1.0,
        &mut updates,
    );

    let update_norms = updates
        .iter()
        .map(|update| (update[0].powi(2) + update[1].powi(2) + update[2].powi(2)).sqrt())
        .collect::<Vec<_>>();
    let redundant_norm = update_norms.iter().copied().fold(0.0_f32, f32::max);
    assert!(
        redundant_norm > 0.05,
        "a redundant active particle should receive a relocation update toward an uncovered surface gap, updates={updates:?}"
    );
    assert!(
        update_norms.iter().all(|norm| *norm <= 1.0 + 1.0e-5),
        "gap relocation should respect max_update_norm, norms={update_norms:?}"
    );
}

#[test]
fn surface_normal_coverage_moves_redundant_particles_to_missing_normal_bins() {
    let target = uv_torus_mesh_target(0.72);
    let sample = target.surface_sample(0);
    let positions = vec![
        [
            sample.position[0],
            sample.position[1],
            sample.position[2],
            1.0,
        ],
        [
            sample.position[0],
            sample.position[1],
            sample.position[2],
            1.0,
        ],
        [
            sample.position[0],
            sample.position[1],
            sample.position[2],
            1.0,
        ],
    ];
    let active_rows = [0, 1, 2];
    let mut updates = vec![[0.0; 3]; positions.len()];

    add_surface_normal_coverage_to_updates(
        &target,
        &positions,
        &active_rows,
        1.0,
        1.0,
        512,
        1.0,
        &mut updates,
    );

    let max_update_norm = updates
        .iter()
        .map(|update| (update[0].powi(2) + update[1].powi(2) + update[2].powi(2)).sqrt())
        .fold(0.0_f32, f32::max);
    assert!(
        max_update_norm > 0.05,
        "normal-bin coverage should relocate a redundant particle toward an under-covered normal bin, updates={updates:?}"
    );
    assert!(updates.iter().flatten().all(|value| value.is_finite()));
}

#[test]
fn surface_normal_coverage_fills_normal_bin_deficits_with_multiple_donors() {
    let target = TriangleMeshTarget::new(
        vec![
            [0.0, 0.0, 0.0],
            [0.08, 0.0, 0.0],
            [0.0, 0.08, 0.0],
            [0.0, 0.0, 0.04],
            [0.08, 0.0, 0.04],
            [0.0, 0.08, 0.04],
        ],
        vec![[0, 1, 2], [5, 4, 3]],
    )
    .unwrap();
    let positions = vec![
        [0.010, 0.010, 0.0, 1.0],
        [0.020, 0.010, 0.0, 1.0],
        [0.010, 0.020, 0.0, 1.0],
        [0.030, 0.010, 0.0, 1.0],
        [0.010, 0.030, 0.0, 1.0],
        [0.020, 0.020, 0.0, 1.0],
        [0.035, 0.015, 0.0, 1.0],
        [0.015, 0.035, 0.0, 1.0],
    ];
    let active_rows = (0..positions.len()).collect::<Vec<_>>();
    let mut updates = vec![[0.0; 3]; positions.len()];

    add_surface_normal_coverage_to_updates(
        &target,
        &positions,
        &active_rows,
        1.0,
        1.0,
        512,
        1.0,
        &mut updates,
    );

    let upward_relocations = updates.iter().filter(|update| update[2] > 1.0e-3).count();
    assert!(
        upward_relocations >= 3,
        "normal coverage should fill an opposite-normal deficit with multiple donors, updates={updates:?}"
    );
    assert!(updates.iter().flatten().all(|value| value.is_finite()));
}

#[test]
fn surface_normal_coverage_report_detects_missing_opposite_normal_support() {
    let target = TriangleMeshTarget::new(
        vec![
            [0.0, 0.0, 0.0],
            [0.08, 0.0, 0.0],
            [0.0, 0.08, 0.0],
            [0.0, 0.0, 0.04],
            [0.08, 0.0, 0.04],
            [0.0, 0.08, 0.04],
        ],
        vec![[0, 1, 2], [5, 4, 3]],
    )
    .unwrap();
    let lower_only = vec![[0.02, 0.02, 0.0, 1.0], [0.04, 0.02, 0.0, 1.0]];
    let both_sides = vec![
        [0.02, 0.02, 0.0, 1.0],
        [0.04, 0.02, 0.0, 1.0],
        [0.02, 0.02, 0.04, 1.0],
        [0.04, 0.02, 0.04, 1.0],
    ];

    let missing = surface_normal_coverage_report(&lower_only, &target, 512, 0.012);
    let covered = surface_normal_coverage_report(&both_sides, &target, 512, 0.012);

    assert!(
        missing.covered_target_bin_fraction < covered.covered_target_bin_fraction,
        "normal coverage should detect that one of the target normal families is absent: missing={missing:?} covered={covered:?}"
    );
    assert!(covered.covered_target_bin_fraction >= 0.99);
    assert!(covered.mean_bin_covered_fraction > missing.mean_bin_covered_fraction);
}

#[test]
fn surface_gap_relocation_can_use_normal_mismatch_as_uncovered_support() {
    let target = TriangleMeshTarget::new(
        vec![
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.02],
            [0.01, 0.0, 0.02],
            [0.0, 0.01, 0.02],
        ],
        vec![[0, 1, 2], [5, 4, 3]],
    )
    .unwrap();
    let positions = vec![[0.003, 0.003, 0.0, 1.0], [0.006, 0.002, 0.0, 1.0]];
    let active_rows = [0, 1];
    let mut position_only = vec![[0.0; 3]; positions.len()];
    let mut normal_aware = vec![[0.0; 3]; positions.len()];

    add_surface_gap_relocation_to_updates(
        &target,
        &positions,
        &active_rows,
        1.0,
        1.0,
        512,
        0.0,
        0.72,
        1.0,
        &mut position_only,
    );
    add_surface_gap_relocation_to_updates(
        &target,
        &positions,
        &active_rows,
        1.0,
        1.0,
        512,
        10.0,
        0.72,
        1.0,
        &mut normal_aware,
    );

    let position_only_z = position_only
        .iter()
        .map(|update| update[2].abs())
        .fold(0.0_f32, f32::max);
    let normal_aware_z = normal_aware
        .iter()
        .map(|update| update[2])
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        normal_aware_z > position_only_z + 1.0e-3,
        "normal-aware gap relocation should expose nearby opposite-normal support: position_only={position_only:?} normal_aware={normal_aware:?}"
    );
}

#[test]
fn gap_relocation_donor_falls_back_to_overassigned_particles() {
    let active_rows = [0, 1];
    let positions = vec![[0.0, 0.0, 0.0, 1.0], [0.25, 0.0, 0.0, 1.0]];
    let assigned_counts = vec![16, 12];
    let used_donors = vec![false, false];
    let average_assignments = 8.0;
    let gap = [0.5, 0.0, 0.0];

    let under_assigned = gap_relocation_donor(
        gap,
        &active_rows,
        &positions,
        positions.len(),
        &assigned_counts,
        average_assignments,
        &used_donors,
        true,
    );
    let fallback = gap_relocation_donor(
        gap,
        &active_rows,
        &positions,
        positions.len(),
        &assigned_counts,
        average_assignments,
        &used_donors,
        false,
    );

    assert_eq!(under_assigned, None);
    assert_eq!(
        fallback,
        Some(1),
        "uncovered surface patches should still get a donor when every active particle is already assigned"
    );
}

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
    );
    apply_material_visible_surface_tail_strict_score(&mut score, escaped);

    assert!(score.material_visible_surface_tail_p99_penalty > 0.0);
    assert!(score.material_visible_surface_tail_fraction_penalty > 0.0);
}
