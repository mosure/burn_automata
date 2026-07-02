use crate::cli::prelude::*;

pub(crate) fn run_train_torus_3d(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::TrainTorus3d {
        model_output,
        report_output,
        rows,
        steps,
    } = command
    else {
        unreachable!("run_train_torus_3d called with the wrong command variant");
    };

    validate_diagnostic_3d_output_not_catalog(&model_output, "train-torus3d")?;
    let config = NpaConfig::torus_field_3dgs();
    let hashgrid = crate::kernels::HashGridConfig::growing_3dgs();
    let mut model = torus_field_model(config.clone())?;
    let batch = torus_field_supervised_batch(&config, rows);
    let sgd = SgdConfig {
        learning_rate: 0.002,
        grad_clip_norm: 1.0,
        ..SgdConfig::default()
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
        Some(format!("trained-rust:{UV_TORUS_TARGET_SOURCE}")),
    );
    crate::import::save_manifest(&model_output, &manifest)?;
    let loaded = crate::import::load_manifest(&model_output)?;
    let loaded_hashgrid = loaded.hashgrid.clone();
    let loaded_model = loaded.into_model();
    let robustness = torus_robustness_report(&loaded_model, &loaded_hashgrid)?;
    let output_report = CliTorusTrainingReport {
        preset: AutomataPreset::Growing3dGs,
        target_source: UV_TORUS_TARGET_SOURCE.to_string(),
        student_seed: 0,
        sgd,
        report,
        model_output: Some(model_output.display().to_string()),
        robustness,
        batch_source: TrainingBatchArg::Features,
        training_mode: MeshTrainingModeArg::ProjectionBaseline,
        rollout_supervision: None,
    };
    std::fs::write(
        &report_output,
        serde_json::to_string_pretty(&output_report)?,
    )?;
    println!(
        "wrote {} and {} final_loss={:.6} robust={}",
        model_output.display(),
        report_output.display(),
        output_report.report.final_loss,
        output_report.robustness.passed
    );
    if !output_report.robustness.passed {
        return Err(std::io::Error::other(format!(
            "torus robustness validation failed; see {}",
            report_output.display()
        ))
        .into());
    }

    Ok(())
}

pub(crate) fn run_train_torus_morphogen_3d(
    command: Command,
) -> Result<(), Box<dyn std::error::Error>> {
    let Command::TrainTorusMorphogen3d {
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
        unreachable!("run_train_torus_morphogen_3d called with the wrong command variant");
    };

    validate_diagnostic_3d_output_not_catalog(&model_output, "train-torus-morphogen3d")?;
    let hashgrid = crate::kernels::HashGridConfig::growing_3dgs();
    let (_config, mut model, batch, sgd, target_source, rollout_report) = match training_mode {
        MeshTrainingModeArg::PositionField => {
            let config = NpaConfig::torus_field_3dgs();
            (
                config.clone(),
                torus_field_model(config.clone())?,
                torus_field_supervised_batch(&config, rows),
                SgdConfig {
                    learning_rate: 2.0e-3,
                    grad_clip_norm: 1.0,
                    ..SgdConfig::default()
                },
                UV_TORUS_POSITION_FIELD_TARGET_SOURCE,
                None,
            )
        }
        MeshTrainingModeArg::RolloutPositionField => {
            let config = NpaConfig::torus_field_3dgs();
            let model = torus_field_model(config.clone())?;
            let feature_rows = rows / 2;
            let rollout_rows = rows.saturating_sub(feature_rows).max(1);
            let batch = merge_supervised_batches(
                torus_field_supervised_batch(&config, feature_rows.max(1)),
                mesh_field_rollout_supervised_batch(
                    &model,
                    &hashgrid,
                    &uv_torus_mesh_target(UV_TORUS_FIELD_SCALE),
                    MeshFieldRolloutBatchConfig {
                        max_rows: rollout_rows,
                        particle_count: rollout_particles,
                        rollout_steps,
                        rollouts,
                        temporal_samples: 1,
                        seed: 0x70_75,
                        seed_scale: UV_TORUS_FIELD_SCALE,
                        seed_mode: ParticleSeed::TorusFieldDense3d,
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
                )?,
            );
            let rollout_report = CliRolloutSupervisionReport {
                particle_count: rollout_particles,
                rollout_steps,
                rollouts,
                temporal_samples: 1,
                update_prob: 1.0,
                seed_scale: UV_TORUS_FIELD_SCALE,
                seed_mode: ParticleSeed::TorusFieldDense3d,
                motion_gain: Some(UV_TORUS_FIELD_MOTION_GAIN),
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
                color_gain: Some(UV_TORUS_FIELD_COLOR_GAIN),
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
                UV_TORUS_ROLLOUT_FIELD_TARGET_SOURCE,
                Some(rollout_report),
            )
        }
        MeshTrainingModeArg::RolloutLocal => {
            let config = NpaConfig::growing_3dgs();
            let target_mesh = uv_torus_mesh_target(UV_TORUS_FIELD_SCALE);
            let student = local_growth_student_model_with_axis_gains(
                config.clone(),
                0x70_75,
                0.0,
                mesh_axis_expansion_gains(&target_mesh, LOCAL_GROWTH_EXPANSION_GAIN),
            )?;
            let rollout_report = CliRolloutSupervisionReport {
                particle_count: rollout_particles,
                rollout_steps,
                rollouts,
                temporal_samples: 5,
                update_prob: 1.0,
                seed_scale: UV_TORUS_FIELD_SCALE,
                seed_mode: ParticleSeed::TorusGrowth3d,
                motion_gain: Some(LOCAL_TORUS_MOTION_GAIN),
                max_update_norm: Some(0.06),
                density_gain: Some(0.0),
                expansion_gain: Some(LOCAL_GROWTH_EXPANSION_GAIN),
                coverage_gain: Some(0.45),
                coverage_samples: Some(4096),
                coverage_mode: Some(CoverageUpdateModeArg::SlicedOt),
                coverage_softness: Some(0.0),
                coverage_repulsion_gain: Some(0.2),
                coverage_gap_gain: Some(0.2),
                coverage_repulsion_radius: Some(0.0),
                coverage_normal_weight: Some(0.0),
                extent_gain: Some(0.4),
                color_gain: Some(LOCAL_TORUS_COLOR_GAIN),
                aux_state_gain: Some(0.5),
                opacity_gain: Some(0.02),
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
                    temporal_samples: 5,
                    seed: 0x70_75,
                    seed_scale: UV_TORUS_FIELD_SCALE,
                    seed_mode: ParticleSeed::TorusGrowth3d,
                    motion_gain: LOCAL_TORUS_MOTION_GAIN,
                    max_update_norm: 0.06,
                    coverage_gain: 0.45,
                    coverage_samples: 4096,
                    coverage_mode: CoverageUpdateModeArg::SlicedOt,
                    coverage_softness: 0.0,
                    coverage_repulsion_gain: 0.2,
                    coverage_gap_gain: 0.2,
                    coverage_repulsion_radius: 0.0,
                    coverage_normal_weight: 0.0,
                    extent_gain: 0.4,
                    color_gain: LOCAL_TORUS_COLOR_GAIN,
                    aux_state_gain: 0.5,
                    opacity_gain: 0.02,
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
                    learning_rate: 4.0e-5,
                    grad_clip_norm: 0.06,
                    ..SgdConfig::default()
                },
                UV_TORUS_MORPHOGEN_ROLLOUT_TARGET_SOURCE,
                Some(rollout_report),
            )
        }
        MeshTrainingModeArg::ProjectionBaseline => {
            let config = NpaConfig::growing_3dgs();
            (
                config.clone(),
                torus_morphogen_model(config.clone())?,
                torus_morphogen_supervised_batch(&config, rows),
                SgdConfig {
                    learning_rate: 0.0,
                    grad_clip_norm: 1.0,
                    ..SgdConfig::default()
                },
                UV_TORUS_MORPHOGEN_BASELINE_TARGET_SOURCE,
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
    let robustness_cases = if loaded_model.config.position_features {
        TORUS_ROBUSTNESS_CASES
    } else {
        TORUS_MORPHOGEN_ROBUSTNESS_CASES
    };
    let robustness =
        torus_robustness_report_for_cases(&loaded_model, &loaded_hashgrid, robustness_cases)?;
    let output_report = CliTorusTrainingReport {
        preset: AutomataPreset::Growing3dGs,
        target_source: target_source.to_string(),
        student_seed: 0,
        sgd,
        report,
        model_output: Some(model_output.display().to_string()),
        robustness,
        batch_source: if matches!(
            training_mode,
            MeshTrainingModeArg::RolloutLocal | MeshTrainingModeArg::RolloutPositionField
        ) {
            TrainingBatchArg::Rollout
        } else {
            TrainingBatchArg::Features
        },
        training_mode,
        rollout_supervision: rollout_report,
    };
    std::fs::write(
        &report_output,
        serde_json::to_string_pretty(&output_report)?,
    )?;
    println!(
        "wrote {} and {} final_loss={:.6} robust={}",
        model_output.display(),
        report_output.display(),
        output_report.report.final_loss,
        output_report.robustness.passed
    );
    if !output_report.robustness.passed {
        return Err(std::io::Error::other(format!(
            "torus morphogen robustness validation failed; see {}",
            report_output.display()
        ))
        .into());
    }

    Ok(())
}

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

pub(crate) fn run_ablate_local_3d(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::AblateLocal3d {
        target,
        base_model,
        model_output,
        report_output,
        rows,
        steps,
        rollout_particles,
        rollout_steps,
        rollouts,
        temporal_samples,
        training_rounds,
        seed_scale,
        seed_mode,
        student_seed,
        learning_rate,
        grad_clip_norm,
        weight_decay,
        motion_gain,
        max_update_norm,
        density_gain,
        expansion_gain,
        coverage_gain,
        coverage_samples,
        coverage_mode,
        coverage_softness,
        coverage_repulsion_gain,
        coverage_gap_gain,
        coverage_repulsion_radius,
        coverage_normal_weight,
        extent_gain,
        color_gain,
        aux_state_gain,
        opacity_gain,
        front_opacity_gain,
        front_radius,
        front_max_opacity_update,
        front_motion_gate,
        preserve_opacity_update,
        fail_on_validation,
    } = command
    else {
        unreachable!("run_ablate_local_3d called with the wrong command variant");
    };

    validate_diagnostic_3d_output_not_catalog(&model_output, "ablate-local-3d")?;
    let target_mesh = mesh_target_for_arg(target, seed_scale);
    let seed_mode = seed_mode
        .map(ParticleSeed::from)
        .unwrap_or_else(|| conditionless_local_seed_mode(target));
    let target_source = mesh_conditionless_local_target_source_for_seed(target, seed_mode);
    let (mut model, hashgrid, output_source) = if let Some(path) = base_model.as_ref() {
        load_conditionless_local_base_model(path, target_source)?
    } else {
        let config = NpaConfig::growing_3dgs();
        let hashgrid = crate::kernels::HashGridConfig::growing_3dgs();
        let model = local_growth_student_model_with_axis_gains(
            config,
            student_seed,
            density_gain,
            mesh_axis_expansion_gains(&target_mesh, expansion_gain),
        )?;
        (model, hashgrid, format!("ablation-rust:{target_source}"))
    };
    let sgd = SgdConfig {
        learning_rate,
        grad_clip_norm,
        weight_decay,
    };
    let preserve_opacity_update =
        preserve_opacity_update || (opacity_gain == 0.0 && front_opacity_gain == 0.0);
    let coverage_gap_gain = coverage_gap_gain.unwrap_or(coverage_repulsion_gain);
    let report = run_refreshed_mesh_local_training(
        &mut model,
        &hashgrid,
        &target_mesh,
        MeshLocalTrainingConfig {
            max_rows: rows,
            particle_count: rollout_particles,
            rollout_steps,
            rollouts,
            temporal_samples,
            training_rounds,
            total_steps: steps,
            seed: student_seed ^ 0x005e_ed3d,
            seed_scale,
            seed_mode,
            motion_gain: motion_gain.unwrap_or_else(|| mesh_target_motion_gain(target)),
            max_update_norm,
            coverage_gain,
            coverage_samples,
            coverage_mode,
            coverage_softness,
            coverage_repulsion_gain,
            coverage_gap_gain,
            coverage_repulsion_radius,
            coverage_normal_weight,
            extent_gain,
            color_gain: color_gain.unwrap_or_else(|| mesh_target_color_gain(target)),
            aux_state_gain,
            opacity_gain,
            front_opacity_gain,
            front_radius,
            front_max_opacity_update,
            front_motion_gate,
            preserve_opacity_update,
            sgd,
        },
    )?;
    if let Some(parent) = model_output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if let Some(parent) = report_output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let manifest =
        BpkModelManifest::from_model(&model, hashgrid.clone(), Some(output_source.clone()));
    crate::import::save_manifest(&model_output, &manifest)?;
    let loaded = crate::import::load_manifest(&model_output)?;
    let loaded_hashgrid = loaded.hashgrid.clone();
    let loaded_model = loaded.into_model();
    let validation_cases = conditionless_local_rollout_cases(target, seed_scale, rollout_particles);
    let mesh_rollout = Some(mesh_rollout_report_for_cases(
        &loaded_model,
        &loaded_hashgrid,
        &target_mesh,
        &validation_cases,
    )?);
    let render_loss = Some(mesh_render_loss_for_model(
        &loaded_model,
        &loaded_hashgrid,
        &target_mesh,
        RenderLossEvalConfig {
            particle_count: rollout_particles,
            steps: 64,
            seed: 0x010c_a202,
            extra_seeds: Vec::new(),
            seed_scale,
            seed_mode,
            render: default_render_loss_config(seed_scale),
        },
    )?);
    let rollout_supervision = Some(CliRolloutSupervisionReport {
        particle_count: rollout_particles,
        rollout_steps,
        rollouts,
        temporal_samples,
        update_prob: 1.0,
        seed_scale,
        seed_mode,
        motion_gain: Some(motion_gain.unwrap_or_else(|| mesh_target_motion_gain(target))),
        max_update_norm: Some(max_update_norm),
        density_gain: Some(density_gain),
        expansion_gain: Some(expansion_gain),
        coverage_gain: Some(coverage_gain),
        coverage_samples: Some(coverage_samples),
        coverage_mode: Some(coverage_mode),
        coverage_softness: Some(coverage_softness),
        coverage_repulsion_gain: Some(coverage_repulsion_gain),
        coverage_gap_gain: Some(coverage_gap_gain),
        coverage_repulsion_radius: Some(coverage_repulsion_radius),
        coverage_normal_weight: Some(coverage_normal_weight),
        extent_gain: Some(extent_gain),
        color_gain: Some(color_gain.unwrap_or_else(|| mesh_target_color_gain(target))),
        aux_state_gain: Some(aux_state_gain),
        opacity_gain: Some(opacity_gain),
        front_opacity_gain: Some(front_opacity_gain),
        front_radius: Some(front_radius),
        front_max_opacity_update: Some(front_max_opacity_update),
        front_motion_gate: Some(front_motion_gate),
        preserve_opacity_update: Some(preserve_opacity_update),
    });
    let output_report = CliTrainingReport {
        preset: AutomataPreset::Growing3dGs,
        target_source: output_source,
        student_seed,
        sgd,
        report,
        model_output: Some(model_output.display().to_string()),
        batch_source: TrainingBatchArg::Rollout,
        rollout_supervision,
        mesh_rollout,
        render_loss,
    };
    std::fs::write(
        &report_output,
        serde_json::to_string_pretty(&output_report)?,
    )?;
    let mesh_status = output_report
        .mesh_rollout
        .as_ref()
        .map_or(
            "skipped",
            |report| if report.passed { "passed" } else { "failed" },
        );
    let render_status = output_report
        .render_loss
        .as_ref()
        .map_or(
            "skipped",
            |report| if report.passed { "passed" } else { "failed" },
        );
    println!(
        "wrote {} and {} final_loss={:.6} mesh_rollout={mesh_status} render_loss={render_status}",
        model_output.display(),
        report_output.display(),
        output_report.report.final_loss
    );
    if fail_on_validation
        && output_report
            .mesh_rollout
            .as_ref()
            .is_some_and(|report| !report.passed)
    {
        return Err(std::io::Error::other(format!(
            "conditionless local 3d ablation failed validation; see {}",
            report_output.display()
        ))
        .into());
    }
    if fail_on_validation
        && output_report
            .render_loss
            .as_ref()
            .is_some_and(|report| !report.passed)
    {
        return Err(std::io::Error::other(format!(
            "conditionless local 3d render validation failed; see {}",
            report_output.display()
        ))
        .into());
    }

    Ok(())
}
