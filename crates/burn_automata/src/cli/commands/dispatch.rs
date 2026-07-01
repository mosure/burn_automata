use super::{resolve_direct_selection_seed_training, resolve_full_coverage_adjoint};
use crate::cli::prelude::*;

pub(crate) fn run_command(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Command::Infer {
            preset,
            steps,
            particles,
            update_prob,
            model,
            gpu,
            neighbor_mode,
            bucket_capacity,
            seed,
            seed_scale,
            seed_mode,
            output,
        } => {
            #[cfg(not(feature = "gpu_wgpu"))]
            let _ = (neighbor_mode, bucket_capacity);
            let preset: AutomataPreset = preset.into();
            let seed_scale = seed_scale.unwrap_or_else(|| NpaConfig::seed_scale_for_preset(preset));
            let (config, grid) = NpaConfig::for_preset(preset);
            let (model, grid) = if let Some(path) = model {
                let manifest = crate::import::load_manifest(path)?;
                let grid = manifest.hashgrid.clone();
                (manifest.into_model(), grid)
            } else {
                (NpaModel::seeded(config, 42), grid)
            };
            let cfg = RolloutConfig {
                steps,
                particle_count: particles,
                update_prob,
                seed: seed.unwrap_or_else(|| RolloutConfig::default().seed),
                seed_scale,
                ..RolloutConfig::default()
            };
            let trace = if gpu {
                #[cfg(feature = "gpu_wgpu")]
                {
                    gpu_rollout_trace(
                        &model,
                        &grid,
                        &cfg,
                        seed_mode.into(),
                        wgpu_neighbor_mode(neighbor_mode, bucket_capacity),
                    )?
                }
                #[cfg(not(feature = "gpu_wgpu"))]
                {
                    return Err(std::io::Error::other(
                        "infer --gpu requires building burn_automata with --features gpu_wgpu",
                    )
                    .into());
                }
            } else {
                run_rollout(&model, &grid, &cfg, seed_mode.into())?
            };
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&output, serde_json::to_string_pretty(&trace)?)?;
            println!("wrote {}", output.display());
        }
        Command::Train {
            preset,
            output,
            model_output,
            rows,
            steps,
            learning_rate,
            grad_clip_norm,
            weight_decay,
            report_interval,
            target_model,
            target_seed,
            zero_update,
            student_seed,
            batch_source,
            rollout_particles,
            rollout_steps,
            rollouts,
            rollout_update_prob,
            seed_scale,
            seed_mode,
        } => {
            let preset: AutomataPreset = preset.into();
            let (preset_config, preset_grid) = NpaConfig::for_preset(preset);
            if target_model.is_some() && target_seed.is_some() {
                return Err(std::io::Error::other(
                    "--target-model and --target-seed are mutually exclusive",
                )
                .into());
            }
            if zero_update && (target_model.is_some() || target_seed.is_some()) {
                return Err(std::io::Error::other(
                    "--zero-update cannot be combined with --target-model or --target-seed",
                )
                .into());
            }
            let (config, hashgrid, target_source, teacher) = if let Some(path) = target_model {
                let manifest = crate::import::load_manifest(&path)?;
                (
                    manifest.config.clone(),
                    manifest.hashgrid.clone(),
                    format!("model:{}", path.display()),
                    Some(manifest.into_model()),
                )
            } else {
                let target_seed = default_train_target_seed(preset, target_seed, zero_update);
                let teacher = target_seed.map(|seed| NpaModel::seeded(preset_config.clone(), seed));
                let target_source = train_target_source(preset, target_seed, zero_update);
                (preset_config, preset_grid, target_source, teacher)
            };
            let mut model = NpaModel::seeded(config.clone(), student_seed);
            let target = if let Some(teacher) = teacher.as_ref() {
                SupervisedTarget::Teacher(teacher)
            } else {
                SupervisedTarget::ZeroUpdate
            };
            let seed_scale = seed_scale.unwrap_or_else(|| NpaConfig::seed_scale_for_preset(preset));
            let seed_mode: ParticleSeed = seed_mode.into();
            let rollout_report = match batch_source {
                TrainingBatchArg::Features => None,
                TrainingBatchArg::Rollout => Some(CliRolloutSupervisionReport {
                    particle_count: rollout_particles,
                    rollout_steps,
                    rollouts,
                    temporal_samples: 1,
                    update_prob: rollout_update_prob,
                    seed_scale,
                    seed_mode,
                    motion_gain: None,
                    max_update_norm: None,
                    density_gain: None,
                    expansion_gain: None,
                    coverage_gain: None,
                    coverage_samples: None,
                    coverage_mode: None,
                    coverage_softness: None,
                    coverage_repulsion_gain: None,
                    coverage_gap_gain: None,
                    coverage_repulsion_radius: None,
                    coverage_normal_weight: None,
                    extent_gain: None,
                    color_gain: None,
                    aux_state_gain: None,
                    opacity_gain: None,
                    front_opacity_gain: None,
                    front_radius: None,
                    front_max_opacity_update: None,
                    front_motion_gate: None,
                    preserve_opacity_update: None,
                }),
            };
            let batch = match batch_source {
                TrainingBatchArg::Features => feature_supervised_batch(
                    &model,
                    target,
                    FeatureBatchConfig {
                        rows,
                        seed: student_seed,
                        ..FeatureBatchConfig::default()
                    },
                )?,
                TrainingBatchArg::Rollout => {
                    let rollout_model = teacher.as_ref().unwrap_or(&model);
                    rollout_supervised_batch_from_model(
                        &model,
                        rollout_model,
                        &hashgrid,
                        target,
                        RolloutSupervisionConfig {
                            max_rows: rows,
                            particle_count: rollout_particles,
                            rollout_steps,
                            rollouts,
                            update_prob: rollout_update_prob,
                            seed: student_seed,
                            seed_scale,
                            seed_mode,
                            ..RolloutSupervisionConfig::default()
                        },
                    )?
                }
            };
            let cfg = SgdConfig {
                learning_rate,
                weight_decay,
                grad_clip_norm,
            };
            let report = run_supervised_training(
                &mut model,
                &batch,
                TrainingRunConfig {
                    steps,
                    report_interval,
                    sgd: cfg,
                },
            )?;
            if let Some(path) = &model_output {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let training_source = training_source_with_batch(batch_source, &target_source);
                let manifest = BpkModelManifest::from_model(
                    &model,
                    hashgrid,
                    Some(format!("trained-rust:{training_source}")),
                );
                crate::import::save_manifest(path, &manifest)?;
            }
            let training_source = training_source_with_batch(batch_source, &target_source);
            let output_report = CliTrainingReport {
                preset,
                target_source: training_source,
                student_seed,
                sgd: cfg,
                report,
                model_output: model_output.as_ref().map(|path| path.display().to_string()),
                batch_source,
                rollout_supervision: rollout_report,
                mesh_rollout: None,
                render_loss: None,
            };
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&output, serde_json::to_string_pretty(&output_report)?)?;
            println!(
                "wrote {} target={} final_loss={:.6} best_loss={:.6}",
                output.display(),
                output_report.target_source,
                output_report.report.final_loss,
                output_report.report.best_loss
            );
        }
        Command::TrainTorus3d {
            model_output,
            report_output,
            rows,
            steps,
        } => {
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
        }
        Command::TrainTorusMorphogen3d {
            model_output,
            report_output,
            rows,
            steps,
            training_mode,
            rollout_particles,
            rollout_steps,
            rollouts,
        } => {
            validate_diagnostic_3d_output_not_catalog(&model_output, "train-torus-morphogen3d")?;
            let hashgrid = crate::kernels::HashGridConfig::growing_3dgs();
            let (_config, mut model, batch, sgd, target_source, rollout_report) =
                match training_mode {
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
            let robustness = torus_robustness_report_for_cases(
                &loaded_model,
                &loaded_hashgrid,
                robustness_cases,
            )?;
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
        }
        Command::TrainTeapotMorphogen3d {
            model_output,
            report_output,
            rows,
            steps,
            training_mode,
            rollout_particles,
            rollout_steps,
            rollouts,
        } => {
            validate_diagnostic_3d_output_not_catalog(&model_output, "train-teapot-morphogen3d")?;
            let hashgrid = crate::kernels::HashGridConfig::growing_3dgs();
            let (_config, mut model, batch, sgd, target_source, rollout_report) =
                match training_mode {
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
        }
        Command::AblateLocal3d {
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
        } => {
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
            let validation_cases =
                conditionless_local_rollout_cases(target, seed_scale, rollout_particles);
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
        }
        Command::RenderLoss3d {
            model,
            target,
            output,
            particles,
            steps,
            seed,
            extra_seeds,
            seed_scale,
            seed_mode,
            image_size,
            target_samples,
            sigma,
            min_sigma,
            max_sigma,
            gaussian_decode_mode,
            world_scale,
            render_opacity_logit_bias,
            density_weight,
            color_weight,
            depth_weight,
            fail_on_validation,
        } => {
            let manifest = crate::import::load_manifest(&model)?;
            let hashgrid = manifest.hashgrid.clone();
            let loaded_model = manifest.into_model();
            let target_mesh = mesh_target_for_arg(target, seed_scale);
            let seed_mode: ParticleSeed = seed_mode.into();
            let render_loss = mesh_render_loss_for_model(
                &loaded_model,
                &hashgrid,
                &target_mesh,
                RenderLossEvalConfig {
                    particle_count: particles,
                    steps,
                    seed,
                    extra_seeds,
                    seed_scale,
                    seed_mode,
                    render: RenderLossConfig {
                        image_size,
                        sigma,
                        min_sigma,
                        max_sigma,
                        gaussian_decode_mode: gaussian_decode_mode.into(),
                        world_scale: world_scale.unwrap_or(seed_scale * 2.0),
                        target_samples,
                        opacity_logit_bias: render_opacity_logit_bias,
                        density_weight,
                        color_weight,
                        depth_weight,
                    },
                },
            )?;
            let output_report = CliRenderLossEvalReport {
                target,
                model: model.display().to_string(),
                particle_count: particles,
                steps,
                seed,
                seed_scale,
                seed_mode,
                render_loss,
            };
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&output, serde_json::to_string_pretty(&output_report)?)?;
            println!(
                "wrote {} render_loss={:.6} density_psnr={:.3} color_psnr={:.3} depth_psnr={:.3} passed={}",
                output.display(),
                output_report.render_loss.total_loss,
                output_report.render_loss.density_psnr_db,
                output_report.render_loss.color_psnr_db,
                output_report.render_loss.depth_psnr_db,
                output_report.render_loss.passed
            );
            if fail_on_validation && !output_report.render_loss.passed {
                return Err(std::io::Error::other(format!(
                    "render loss validation failed; see {}",
                    output.display()
                ))
                .into());
            }
        }
        Command::ValidateGrowth3d {
            model,
            target,
            output,
            particles,
            steps,
            seed,
            extra_seeds,
            seed_scale,
            seed_mode,
            image_size,
            target_samples,
            sigma,
            min_sigma,
            max_sigma,
            gaussian_decode_mode,
            world_scale,
            render_opacity_logit_bias,
            density_weight,
            color_weight,
            depth_weight,
            gate,
            fail_on_validation,
        } => {
            let seed_mode = seed_mode
                .map(ParticleSeed::from)
                .unwrap_or_else(|| conditionless_local_seed_mode(target));
            let report = growth_3d_validation_report(
                &model,
                target,
                Growth3dValidationConfig {
                    particle_count: particles,
                    steps,
                    seed,
                    extra_seeds,
                    seed_scale,
                    seed_mode,
                    gate,
                    render: RenderLossConfig {
                        image_size,
                        sigma,
                        min_sigma,
                        max_sigma,
                        gaussian_decode_mode: gaussian_decode_mode.into(),
                        world_scale: world_scale.unwrap_or(seed_scale * 2.0),
                        target_samples,
                        opacity_logit_bias: render_opacity_logit_bias,
                        density_weight,
                        color_weight,
                        depth_weight,
                    },
                },
            )?;
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&output, serde_json::to_string_pretty(&report)?)?;
            println!(
                "wrote {} gate={:?} gate_passed={} robust_gate_passed={} strict_passed={} strict_score={:.6} catalog_sanity={} render_loss={:.6} density_psnr={:.3} active={}->{} newly_activated_fraction={:.3} opacity_max={:.3}",
                output.display(),
                report.gate,
                report.gate_passed,
                report.robustness.all_gate_passed,
                report.strict_passed,
                report.strict_score.score,
                report.catalog_sanity.passed,
                report.render_loss.total_loss,
                report.render_loss.density_psnr_db,
                report.activation.active_seed_count,
                report.activation.final_active_count,
                report.activation.newly_activated_fraction,
                report.final_opacity.max,
            );
            if fail_on_validation && !growth_3d_fail_on_validation_passed(&report) {
                return Err(std::io::Error::other(format!(
                    "growth 3D validation failed; see {}",
                    output.display()
                ))
                .into());
            }
        }
        Command::RetimeGrowth3d {
            model,
            output,
            front_gain,
            hidden,
            skip_front_retime,
            active_opacity_gain,
            active_opacity_hidden,
            opacity_bias,
            material_opacity_bias,
            alpha,
        } => {
            validate_diagnostic_3d_output_not_catalog(&output, "retime-growth3d")?;
            let manifest = crate::import::load_manifest(&model)?;
            let source = manifest.source.clone();
            let hashgrid = manifest.hashgrid.clone();
            let mut model_value = manifest.into_model();
            let hidden = if skip_front_retime {
                hidden
            } else {
                Some(retime_growth_3d_front_model(
                    &mut model_value,
                    hidden,
                    front_gain,
                )?)
            };
            let active_opacity_hidden = if let Some(gain) = active_opacity_gain {
                Some(retime_growth_3d_active_opacity_model(
                    &mut model_value,
                    active_opacity_hidden,
                    gain,
                )?)
            } else {
                None
            };
            if let Some(alpha) = alpha {
                if !alpha.is_finite() || alpha <= 0.0 {
                    return Err(std::io::Error::other("alpha must be positive and finite").into());
                }
                model_value.config.alpha = alpha;
            }
            if let Some(opacity_bias) = opacity_bias {
                add_growth_3d_opacity_update_bias(&mut model_value, opacity_bias)?;
            }
            if let Some(material_opacity_bias) = material_opacity_bias {
                add_growth_3d_material_opacity_update_bias(
                    &mut model_value,
                    material_opacity_bias,
                )?;
            }
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let retimed_source = Some(format!(
                "retimed-local-front:hidden={}:gain={front_gain}:alpha={}:front_retime={}:active_opacity_hidden={}:active_opacity_gain={}:opacity_bias={}:material_opacity_bias={}:base={}",
                hidden.map_or_else(|| "skipped".to_string(), |hidden| hidden.to_string()),
                model_value.config.alpha,
                !skip_front_retime,
                active_opacity_hidden
                    .map_or_else(|| "skipped".to_string(), |hidden| hidden.to_string()),
                active_opacity_gain.map_or_else(|| "skipped".to_string(), |gain| gain.to_string()),
                opacity_bias.map_or_else(|| "skipped".to_string(), |bias| bias.to_string()),
                material_opacity_bias
                    .map_or_else(|| "skipped".to_string(), |bias| bias.to_string()),
                source.as_deref().unwrap_or("unknown")
            ));
            let retimed_manifest =
                BpkModelManifest::from_model(&model_value, hashgrid, retimed_source);
            crate::import::save_manifest(&output, &retimed_manifest)?;
            println!(
                "wrote {} retimed_hidden={} front_gain={} alpha={} front_retime={} active_opacity_hidden={} active_opacity_gain={} opacity_bias={} material_opacity_bias={}",
                output.display(),
                hidden.map_or_else(|| "skipped".to_string(), |hidden| hidden.to_string()),
                front_gain,
                model_value.config.alpha,
                !skip_front_retime,
                active_opacity_hidden
                    .map_or_else(|| "skipped".to_string(), |hidden| hidden.to_string()),
                active_opacity_gain.map_or_else(|| "skipped".to_string(), |gain| gain.to_string()),
                opacity_bias.map_or_else(|| "skipped".to_string(), |bias| bias.to_string()),
                material_opacity_bias
                    .map_or_else(|| "skipped".to_string(), |bias| bias.to_string())
            );
        }
        Command::TrainRender3d {
            target,
            base_model,
            model_output,
            report_output,
            rounds,
            supervised_steps_per_round,
            particles,
            rollout_steps,
            gradient_particles,
            gradient_mode,
            finite_diff_eps,
            motion_gain,
            perception_position_gain,
            max_update_norm,
            trajectory_supervision,
            trajectory_render_gain,
            trajectory_mesh_gain,
            trajectory_render_samples,
            liveness_gain,
            liveness_front_radius,
            liveness_update_multiplier,
            coverage_gain,
            coverage_samples,
            coverage_mode,
            coverage_softness,
            coverage_repulsion_gain,
            coverage_gap_gain,
            coverage_repulsion_radius,
            coverage_normal_weight,
            extent_gain,
            full_coverage_adjoint,
            no_full_coverage_adjoint,
            surface_gain,
            surface_escape_gain,
            opacity_gain,
            material_liveness_gain,
            material_tail_gain,
            material_suppression_update_multiplier,
            material_max_opacity_update,
            scale_gain,
            scale_budget_weight,
            max_opacity_update,
            learning_rate,
            grad_clip_norm,
            direct_output_gradient_rms_cap,
            direct_line_search,
            direct_line_search_scales,
            direct_material_output_only,
            training_backend,
            direct_selection_seed_training,
            no_direct_selection_seed_training,
            seed_scale,
            seed_mode,
            selection_seed,
            extra_selection_seeds,
            image_size,
            target_samples,
            sigma,
            min_sigma,
            max_sigma,
            gaussian_decode_mode,
            world_scale,
            render_opacity_logit_bias,
            density_weight,
            color_weight,
            depth_weight,
            fail_on_validation,
        } => {
            let full_coverage_adjoint =
                resolve_full_coverage_adjoint(full_coverage_adjoint, no_full_coverage_adjoint)?;
            let direct_selection_seed_training = resolve_direct_selection_seed_training(
                direct_selection_seed_training,
                no_direct_selection_seed_training,
            )?;
            let hashgrid = crate::kernels::HashGridConfig::growing_3dgs();
            let requested_seed_mode = seed_mode.map(ParticleSeed::from);
            let target_mesh = mesh_target_for_arg(target, seed_scale);
            let (mut model, base_source, default_seed_mode) = if let Some(path) =
                base_model.as_ref()
            {
                let manifest = crate::import::load_manifest(path)?;
                let base_source = manifest.source.clone();
                let model = manifest.into_model();
                let default_seed_mode = default_render_training_seed_mode(target, &model);
                (model, base_source, default_seed_mode)
            } else {
                let default_seed_mode = render_training_default_seed_mode(target);
                let seed_mode = requested_seed_mode.unwrap_or(default_seed_mode);
                if !target_local_growth_seed(target, seed_mode) {
                    return Err(std::io::Error::other(format!(
                        "train-render3d without --base-model defaults to conditionless-local growth and requires a target local growth seed; got seed_mode={seed_mode:?}"
                    ))
                    .into());
                }
                let (model, source) = render_training_base_model(target, &target_mesh, seed_mode)?;
                (model, Some(source), default_seed_mode)
            };
            let seed_mode = requested_seed_mode.unwrap_or(default_seed_mode);
            let catalog_bound_output = is_catalog_model_output_path(&model_output);
            validate_catalog_bound_render_training_output(
                &model_output,
                target,
                seed_mode,
                base_source.as_deref(),
            )?;
            let coverage_gap_gain = coverage_gap_gain.unwrap_or(coverage_repulsion_gain);
            let render = RenderLossConfig {
                image_size,
                sigma,
                min_sigma,
                max_sigma,
                gaussian_decode_mode: gaussian_decode_mode.into(),
                world_scale: world_scale.unwrap_or(seed_scale * 2.0),
                target_samples,
                opacity_logit_bias: render_opacity_logit_bias,
                density_weight,
                color_weight,
                depth_weight,
            };
            let sgd = SgdConfig {
                learning_rate,
                grad_clip_norm,
                weight_decay: 0.0,
            };
            let report = run_render_proxy_training(
                &mut model,
                &hashgrid,
                &target_mesh,
                RenderProxyTrainingConfig {
                    target,
                    rounds,
                    supervised_steps_per_round,
                    particles,
                    rollout_steps,
                    gradient_particles,
                    gradient_mode,
                    finite_diff_eps,
                    motion_gain,
                    perception_position_gain,
                    max_update_norm,
                    trajectory_supervision,
                    trajectory_render_gain,
                    trajectory_mesh_gain,
                    trajectory_render_samples,
                    liveness_gain,
                    liveness_front_radius,
                    liveness_update_multiplier,
                    coverage_gain,
                    coverage_samples,
                    coverage_mode,
                    coverage_softness,
                    coverage_repulsion_gain,
                    coverage_gap_gain,
                    coverage_repulsion_radius,
                    coverage_normal_weight,
                    extent_gain,
                    full_coverage_adjoint,
                    surface_gain,
                    surface_escape_gain,
                    opacity_gain,
                    material_liveness_gain,
                    material_tail_gain,
                    material_suppression_update_multiplier,
                    material_max_opacity_update,
                    scale_gain,
                    scale_budget_weight,
                    max_opacity_update,
                    direct_output_gradient_rms_cap,
                    direct_line_search,
                    direct_line_search_scales: direct_line_search_scales.clone(),
                    direct_material_output_only,
                    training_backend,
                    direct_selection_seed_training,
                    seed: 0x005a_173d,
                    selection_seed: Some(selection_seed),
                    selection_seeds: extra_selection_seeds.clone(),
                    seed_scale,
                    seed_mode,
                    render,
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
                Some(render_training_source(
                    target,
                    base_source.as_deref(),
                    seed_mode,
                )),
            );
            let validation_extra_seeds =
                render_training_validation_extra_seeds(selection_seed, &extra_selection_seeds);
            let mut catalog_promotion_validations = Vec::new();
            if catalog_bound_output {
                let candidate_path = catalog_bound_candidate_path(target, std::process::id());
                if let Some(parent) = candidate_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                crate::import::save_manifest(&candidate_path, &manifest)?;
                let promotion_result = (|| -> Result<(), Box<dyn std::error::Error>> {
                    for validation_cfg in catalog_promotion_validation_configs(
                        selection_seed,
                        &extra_selection_seeds,
                        seed_scale,
                        seed_mode,
                        render,
                    ) {
                        catalog_promotion_validations.push(growth_3d_validation_report(
                            &candidate_path,
                            target,
                            validation_cfg,
                        )?);
                    }
                    require_catalog_promotion_validations_pass(
                        &catalog_promotion_validations,
                        &model_output,
                    )
                })();
                if let Err(error) = promotion_result {
                    std::fs::remove_file(&candidate_path).ok();
                    return Err(error);
                }
                crate::import::save_manifest(&model_output, &manifest)?;
                std::fs::remove_file(&candidate_path).ok();
            } else {
                crate::import::save_manifest(&model_output, &manifest)?;
            }
            let loaded = crate::import::load_manifest(&model_output)?;
            let loaded_hashgrid = loaded.hashgrid.clone();
            let loaded_model = loaded.into_model();
            let growth_validation = growth_3d_validation_report(
                &model_output,
                target,
                Growth3dValidationConfig {
                    particle_count: particles,
                    steps: rollout_steps,
                    seed: 0x005a_173d,
                    extra_seeds: validation_extra_seeds,
                    seed_scale,
                    seed_mode,
                    gate: Growth3dValidationGateArg::Strict,
                    render,
                },
            )?;
            let final_render_loss = mesh_render_loss_for_model(
                &loaded_model,
                &loaded_hashgrid,
                &target_mesh,
                RenderLossEvalConfig {
                    particle_count: particles,
                    steps: rollout_steps,
                    seed: 0x005a_173d,
                    extra_seeds: Vec::new(),
                    seed_scale,
                    seed_mode,
                    render,
                },
            )?;
            let strict_gate_summary =
                CliRenderTrainingGateSummary::from_validation(&growth_validation);
            let output_report = CliRenderTrainingReport {
                target,
                base_model: base_model.as_ref().map(|path| path.display().to_string()),
                model_output: model_output.display().to_string(),
                particle_count: particles,
                rollout_steps,
                seed_scale,
                seed_mode,
                sgd,
                report,
                final_render_loss,
                strict_gate_summary,
                growth_validation,
                catalog_promotion_validations,
            };
            std::fs::write(
                &report_output,
                serde_json::to_string_pretty(&output_report)?,
            )?;
            println!(
                "wrote {} and {} render_loss={:.6} density_psnr={:.3} color_psnr={:.3} depth_psnr={:.3} passed={} strict_passed={} strict_score={:.3}",
                model_output.display(),
                report_output.display(),
                output_report.final_render_loss.total_loss,
                output_report.final_render_loss.density_psnr_db,
                output_report.final_render_loss.color_psnr_db,
                output_report.final_render_loss.depth_psnr_db,
                output_report.final_render_loss.passed,
                output_report.growth_validation.strict_passed,
                output_report.growth_validation.strict_score.score
            );
            if fail_on_validation
                && !growth_3d_fail_on_validation_passed(&output_report.growth_validation)
            {
                return Err(std::io::Error::other(format!(
                    "render-proxy training failed strict growth validation (score={:.6}, failures={:?}); see {}",
                    output_report.growth_validation.strict_score.score,
                    output_report.growth_validation.strict_checks.failure_reasons,
                    report_output.display(),
                ))
                .into());
            }
        }
        Command::Import { input, output } => {
            let report = import_model(input, output)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Bench {
            preset,
            particles,
            steps,
            repeats,
            update_prob,
            gpu,
            neighbor_mode,
            bucket_capacity,
            profile,
            seed_scale,
            normalize_seed_scale,
            fixed_eps,
            reference_seed_scale,
            seed_mode,
            geometry,
            gaussian,
        } => {
            #[cfg(not(feature = "gpu_wgpu"))]
            let _ = (neighbor_mode, bucket_capacity, gaussian, repeats);
            let preset: AutomataPreset = preset.into();
            let seed_scale = seed_scale.unwrap_or_else(|| NpaConfig::seed_scale_for_preset(preset));
            let seed_mode: ParticleSeed = seed_mode.into();
            let normalize_seed_scale = normalize_seed_scale || !fixed_eps;
            let reference_seed_scale = reference_seed_scale
                .unwrap_or_else(|| reference_seed_scale_for_seed_mode(preset, seed_mode));
            let (config, base_grid) = NpaConfig::for_preset(preset);
            let model = NpaModel::seeded(config.clone(), 42);
            let grid = if normalize_seed_scale {
                model
                    .config
                    .hashgrid_for_seed_scale(&base_grid, seed_scale, reference_seed_scale)
            } else {
                base_grid
            };
            let start = Instant::now();
            if gpu {
                #[cfg(feature = "gpu_wgpu")]
                {
                    let report = gpu_rollout_bench(
                        &model,
                        &grid,
                        GpuBenchConfig {
                            particles,
                            steps,
                            seed_scale,
                            update_prob,
                            seed_mode,
                            geometry,
                            neighbor_mode: wgpu_neighbor_mode(neighbor_mode, bucket_capacity),
                            gaussian_write: gaussian,
                        },
                    )?;
                    let reports = if repeats > 1 {
                        let mut reports = Vec::with_capacity(repeats);
                        reports.push(report);
                        for _ in 1..repeats {
                            reports.push(gpu_rollout_bench(
                                &model,
                                &grid,
                                GpuBenchConfig {
                                    particles,
                                    steps,
                                    seed_scale,
                                    update_prob,
                                    seed_mode,
                                    geometry,
                                    neighbor_mode: wgpu_neighbor_mode(
                                        neighbor_mode,
                                        bucket_capacity,
                                    ),
                                    gaussian_write: gaussian,
                                },
                            )?);
                        }
                        reports
                    } else {
                        vec![report]
                    };
                    let summary = summarize_gpu_reports(&reports, steps);
                    let report = summary.median_report;
                    let avg_step_ms = report.gpu_step_ms / steps.max(1) as f64;
                    println!(
                        "backend=wgpu particles={particles} steps={steps} repeats={} update_prob={update_prob:.3} geometry={geometry:?} elapsed_ms={:.6} gpu_step_ms={:.6} avg_step_ms={avg_step_ms:.6} min_avg_step_ms={:.6} median_avg_step_ms={:.6} max_avg_step_ms={:.6} final_mean_displacement_per_step={:.6} final_mean_density={:.6} initial_nonempty_cells={} initial_max_cell_occupancy={} hashgrid=gpu-local hashgrid_eps={:.6} normalized_seed_scale={} reference_seed_scale={:.6} resident_state=true timing=submit_wait readback=final gaussian_write={} neighbor_mode={:?} bucket_capacity={} grid_storage_u32={} grid_clear_u32={} grid_overflow_count={}",
                        summary.repeats,
                        start.elapsed().as_secs_f64() * 1000.0,
                        report.gpu_step_ms,
                        summary.min_avg_step_ms,
                        summary.median_avg_step_ms,
                        summary.max_avg_step_ms,
                        report.final_mean_dx,
                        report.final_mean_density,
                        report.initial_nonempty_cells,
                        report.initial_max_cell_occupancy,
                        grid.eps,
                        normalize_seed_scale,
                        reference_seed_scale,
                        report.gaussian_write,
                        report.neighbor_mode,
                        report.bucket_capacity,
                        report.grid_storage_len,
                        report.grid_clear_len,
                        report.grid_overflow_count
                    );
                }
                #[cfg(not(feature = "gpu_wgpu"))]
                {
                    return Err(std::io::Error::other(
                        "bench --gpu requires building burn_automata with --features gpu_wgpu",
                    )
                    .into());
                }
            } else if profile {
                let profile = profile_rollout(
                    &model,
                    &grid,
                    CpuProfileConfig {
                        particles,
                        steps,
                        seed_scale,
                        update_prob,
                        seed_mode,
                        geometry,
                    },
                )?;
                println!(
                    "particles={particles} steps={steps} update_prob={update_prob:.3} geometry={geometry:?} elapsed_ms={:.6} perceive_ms={:.6} forward_ms={:.6} integrate_ms={:.6} final_mean_dx={:.6}",
                    start.elapsed().as_secs_f64() * 1000.0,
                    profile.perceive_ms,
                    profile.forward_ms,
                    profile.integrate_ms,
                    profile.final_mean_dx
                );
            } else {
                let trace = run_rollout(
                    &model,
                    &grid,
                    &RolloutConfig {
                        steps,
                        particle_count: particles,
                        update_prob,
                        seed_scale,
                        ..RolloutConfig::default()
                    },
                    seed_mode,
                )?;
                println!(
                    "particles={particles} steps={steps} update_prob={update_prob:.3} elapsed_ms={} final_mean_dx={:.6}",
                    start.elapsed().as_secs_f64() * 1000.0,
                    trace.mean_dx.last().copied().unwrap_or_default()
                );
            }
        }
        Command::BenchSpatial {
            preset,
            particles,
            seed_scale,
            normalize_seed_scale,
            fixed_eps,
            reference_seed_scale,
            seed_mode,
            geometry,
            strategy,
            bvh_leaf_size,
            tile_size,
        } => {
            let preset: AutomataPreset = preset.into();
            let seed_scale = seed_scale.unwrap_or_else(|| NpaConfig::seed_scale_for_preset(preset));
            let seed_mode: ParticleSeed = seed_mode.into();
            let normalize_seed_scale = normalize_seed_scale || !fixed_eps;
            let reference_seed_scale = reference_seed_scale
                .unwrap_or_else(|| reference_seed_scale_for_seed_mode(preset, seed_mode));
            let (config, base_grid) = NpaConfig::for_preset(preset);
            let model = NpaModel::seeded(config.clone(), 42);
            let grid = if normalize_seed_scale {
                model
                    .config
                    .hashgrid_for_seed_scale(&base_grid, seed_scale, reference_seed_scale)
            } else {
                base_grid
            };
            let (positions, _states) = bench_particles(
                &model, &grid, particles, seed_scale, seed_mode, geometry, 42,
            );
            let strategies =
                spatial_strategies(strategy, &grid, parse_tile_size(&tile_size)?, bvh_leaf_size);
            for strategy in strategies {
                let started = Instant::now();
                match crate::kernels::analyze_spatial_strategy(
                    &positions, 1, particles, &grid, strategy,
                ) {
                    Ok(report) => {
                        println!(
                            "backend=cpu-spatial preset={preset:?} particles={particles} geometry={geometry:?} strategy={} dim={} eps={:.6} analyze_ms={:.6} active_bins={} max_bin_occupancy={} candidates_per_particle={:.6} entries_per_particle={:.6} exact_neighbors_per_particle={:.6} node_visits_per_particle={:.6} node_count={} max_depth={} exact_neighbor_pairs={} candidate_tests={} candidate_entries_visited={}",
                            strategy_label(report.strategy),
                            report.dim,
                            report.eps,
                            started.elapsed().as_secs_f64() * 1000.0,
                            report.active_bins,
                            report.max_bin_occupancy,
                            report.candidates_per_particle(),
                            report.entries_per_particle(),
                            report.exact_neighbors_per_particle(),
                            report.node_visits_per_particle(),
                            report.node_count,
                            report.max_depth,
                            report.exact_neighbor_pairs,
                            report.candidate_tests,
                            report.candidate_entries_visited,
                        );
                    }
                    Err(err) => {
                        println!(
                            "backend=cpu-spatial preset={preset:?} particles={particles} geometry={geometry:?} strategy={} error=\"{}\"",
                            strategy_label(strategy),
                            err
                        );
                    }
                }
            }
        }
        Command::Manifest { preset, output } => {
            let preset: AutomataPreset = preset.into();
            let (config, hashgrid) = NpaConfig::for_preset(preset);
            let model = NpaModel::seeded(config, 42);
            let manifest = BpkModelManifest::from_model(
                &model,
                hashgrid,
                Some(format!("seeded-rust:{preset:?}")),
            );
            crate::import::save_manifest(&output, &manifest)?;
            println!("wrote {}", output.display());
        }
    }
    Ok(())
}
