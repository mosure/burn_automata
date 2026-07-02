use crate::cli::prelude::*;

pub(crate) fn run_train_teapot_morphogen_3d(
    command: Command,
) -> Result<(), Box<dyn std::error::Error>> {
    let Command::TrainTeapotMorphogen3d {
        model_output,
        report_output,
        rows,
        steps,
        training_mode,
        rollout_particles,
        rollout_steps,
        rollouts,
    } = command
    else {
        unreachable!("run_train_teapot_morphogen_3d called with the wrong command variant");
    };

    validate_diagnostic_3d_output_not_catalog(&model_output, "train-teapot-morphogen3d")?;
    let hashgrid = crate::kernels::HashGridConfig::growing_3dgs();
    let (_config, mut model, batch, sgd, target_source, rollout_report) = match training_mode {
        MeshTrainingModeArg::PositionField => {
            let config = NpaConfig::torus_field_3dgs();
            (
                config.clone(),
                teapot_field_model(config.clone())?,
                teapot_field_supervised_batch(&config, rows),
                SgdConfig {
                    learning_rate: 2.0e-3,
                    grad_clip_norm: 1.0,
                    ..SgdConfig::default()
                },
                TEAPOT_POSITION_FIELD_TARGET_SOURCE,
                None,
            )
        }
        MeshTrainingModeArg::RolloutPositionField => {
            let config = NpaConfig::torus_field_3dgs();
            let model = teapot_field_model(config.clone())?;
            let feature_rows = rows / 2;
            let rollout_rows = rows.saturating_sub(feature_rows).max(1);
            let batch = merge_supervised_batches(
                teapot_field_supervised_batch(&config, feature_rows.max(1)),
                mesh_field_rollout_supervised_batch(
                    &model,
                    &hashgrid,
                    &utah_teapot_mesh_target(UV_TORUS_FIELD_SCALE),
                    MeshFieldRolloutBatchConfig {
                        max_rows: rollout_rows,
                        particle_count: rollout_particles,
                        rollout_steps,
                        rollouts,
                        temporal_samples: 1,
                        seed: 0x7ea9_07d0,
                        seed_scale: UV_TORUS_FIELD_SCALE,
                        seed_mode: ParticleSeed::TeapotFieldDense3d,
                        motion_gain: TEAPOT_FIELD_MOTION_GAIN,
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
                        color_gain: TEAPOT_FIELD_COLOR_GAIN,
                        aux_state_gain: 1.0,
                        opacity_gain: UV_TORUS_FIELD_OPACITY_GAIN,
                        front_opacity_gain: 0.0,
                        front_radius: 0.0,
                        front_max_opacity_update: 0.0,
                        front_motion_gate: false,
                        preserve_opacity_update: false,
                    },
                )?,
            );
            let rollout_report = CliRolloutSupervisionReport {
                particle_count: rollout_particles,
                rollout_steps,
                rollouts,
                temporal_samples: 1,
                update_prob: 1.0,
                seed_scale: UV_TORUS_FIELD_SCALE,
                seed_mode: ParticleSeed::TeapotFieldDense3d,
                motion_gain: Some(TEAPOT_FIELD_MOTION_GAIN),
                max_update_norm: Some(f32::INFINITY),
                density_gain: Some(0.0),
                expansion_gain: None,
                coverage_gain: Some(0.0),
                coverage_samples: None,
                coverage_mode: None,
                coverage_softness: None,
                coverage_repulsion_gain: None,
                coverage_gap_gain: None,
                coverage_repulsion_radius: None,
                coverage_normal_weight: None,
                extent_gain: None,
                color_gain: Some(TEAPOT_FIELD_COLOR_GAIN),
                aux_state_gain: Some(1.0),
                opacity_gain: Some(UV_TORUS_FIELD_OPACITY_GAIN),
                front_opacity_gain: None,
                front_radius: None,
                front_max_opacity_update: None,
                front_motion_gate: None,
                preserve_opacity_update: None,
            };
            (
                config,
                model,
                batch,
                SgdConfig {
                    learning_rate: 2.0e-3,
                    grad_clip_norm: 1.0,
                    ..SgdConfig::default()
                },
                TEAPOT_ROLLOUT_FIELD_TARGET_SOURCE,
                Some(rollout_report),
            )
        }
        MeshTrainingModeArg::RolloutLocal => {
            let config = NpaConfig::growing_3dgs();
            let target_mesh = utah_teapot_mesh_target(UV_TORUS_FIELD_SCALE);
            let student = local_growth_student_model_with_axis_gains(
                config.clone(),
                0x7ea9_07d0,
                0.0,
                mesh_axis_expansion_gains(&target_mesh, LOCAL_GROWTH_EXPANSION_GAIN),
            )?;
            let rollout_report = CliRolloutSupervisionReport {
                particle_count: rollout_particles,
                rollout_steps,
                rollouts,
                temporal_samples: 4,
                update_prob: 1.0,
                seed_scale: UV_TORUS_FIELD_SCALE,
                seed_mode: ParticleSeed::TeapotGrowth3d,
                motion_gain: Some(LOCAL_TEAPOT_MOTION_GAIN),
                max_update_norm: Some(0.06),
                density_gain: Some(0.0),
                expansion_gain: Some(LOCAL_GROWTH_EXPANSION_GAIN),
                coverage_gain: Some(0.35),
                coverage_samples: Some(4096),
                coverage_mode: Some(CoverageUpdateModeArg::SlicedOt),
                coverage_softness: Some(0.0),
                coverage_repulsion_gain: Some(0.2),
                coverage_gap_gain: Some(0.2),
                coverage_repulsion_radius: Some(0.0),
                coverage_normal_weight: Some(0.0),
                extent_gain: Some(0.14),
                color_gain: Some(LOCAL_TEAPOT_COLOR_GAIN),
                aux_state_gain: Some(0.3),
                opacity_gain: Some(0.12),
                front_opacity_gain: Some(0.05),
                front_radius: Some(0.24),
                front_max_opacity_update: Some(0.16),
                front_motion_gate: Some(true),
                preserve_opacity_update: Some(false),
            };
            let batch = mesh_local_rollout_supervised_batch(
                &student,
                &hashgrid,
                &target_mesh,
                MeshFieldRolloutBatchConfig {
                    max_rows: rows,
                    particle_count: rollout_particles,
                    rollout_steps,
                    rollouts,
                    temporal_samples: 4,
                    seed: 0x7ea9_07d0,
                    seed_scale: UV_TORUS_FIELD_SCALE,
                    seed_mode: ParticleSeed::TeapotGrowth3d,
                    motion_gain: LOCAL_TEAPOT_MOTION_GAIN,
                    max_update_norm: 0.06,
                    coverage_gain: 0.35,
                    coverage_samples: 4096,
                    coverage_mode: CoverageUpdateModeArg::SlicedOt,
                    coverage_softness: 0.0,
                    coverage_repulsion_gain: 0.2,
                    coverage_gap_gain: 0.2,
                    coverage_repulsion_radius: 0.0,
                    coverage_normal_weight: 0.0,
                    extent_gain: 0.14,
                    color_gain: LOCAL_TEAPOT_COLOR_GAIN,
                    aux_state_gain: 0.3,
                    opacity_gain: 0.12,
                    front_opacity_gain: 0.05,
                    front_radius: 0.24,
                    front_max_opacity_update: 0.16,
                    front_motion_gate: true,
                    preserve_opacity_update: false,
                },
            )?;
            (
                config,
                student,
                batch,
                SgdConfig {
                    learning_rate: 5.0e-5,
                    grad_clip_norm: 0.08,
                    ..SgdConfig::default()
                },
                TEAPOT_MORPHOGEN_ROLLOUT_TARGET_SOURCE,
                Some(rollout_report),
            )
        }
        MeshTrainingModeArg::ProjectionBaseline => {
            let config = NpaConfig::growing_3dgs();
            (
                config.clone(),
                seed_frame_morphogen_model(config.clone())?,
                teapot_morphogen_supervised_batch(&config, rows),
                SgdConfig {
                    learning_rate: 0.0,
                    grad_clip_norm: 1.0,
                    ..SgdConfig::default()
                },
                TEAPOT_MORPHOGEN_BASELINE_TARGET_SOURCE,
                None,
            )
        }
    };
    let report = run_supervised_training(
        &mut model,
        &batch,
        TrainingRunConfig {
            steps,
            report_interval: steps.max(1),
            sgd,
        },
    )?;
    if let Some(parent) = model_output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if let Some(parent) = report_output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let manifest = BpkModelManifest::from_model(
        &model,
        hashgrid.clone(),
        Some(format!("trained-rust:{target_source}")),
    );
    crate::import::save_manifest(&model_output, &manifest)?;
    let loaded = crate::import::load_manifest(&model_output)?;
    let loaded_hashgrid = loaded.hashgrid.clone();
    let loaded_model = loaded.into_model();
    let mesh_rollout = if matches!(
        training_mode,
        MeshTrainingModeArg::PositionField | MeshTrainingModeArg::RolloutPositionField
    ) {
        Some(mesh_rollout_report_for_cases(
            &loaded_model,
            &loaded_hashgrid,
            &utah_teapot_mesh_target(UV_TORUS_FIELD_SCALE),
            TEAPOT_FIELD_ROLLOUT_CASES,
        )?)
    } else {
        None
    };
    let output_report = CliTrainingReport {
        preset: AutomataPreset::Growing3dGs,
        target_source: target_source.to_string(),
        student_seed: 0,
        sgd,
        report,
        model_output: Some(model_output.display().to_string()),
        batch_source: if matches!(
            training_mode,
            MeshTrainingModeArg::RolloutLocal | MeshTrainingModeArg::RolloutPositionField
        ) {
            TrainingBatchArg::Rollout
        } else {
            TrainingBatchArg::Features
        },
        rollout_supervision: rollout_report,
        mesh_rollout,
        render_loss: None,
    };
    std::fs::write(
        &report_output,
        serde_json::to_string_pretty(&output_report)?,
    )?;
    println!(
        "wrote {} and {} final_loss={:.6} mesh_rollout={}",
        model_output.display(),
        report_output.display(),
        output_report.report.final_loss,
        output_report
            .mesh_rollout
            .as_ref()
            .map_or("skipped", |report| if report.passed {
                "passed"
            } else {
                "failed"
            })
    );
    if output_report
        .mesh_rollout
        .as_ref()
        .is_some_and(|report| !report.passed)
    {
        return Err(std::io::Error::other(format!(
            "teapot mesh rollout validation failed; see {}",
            report_output.display()
        ))
        .into());
    }

    Ok(())
}
