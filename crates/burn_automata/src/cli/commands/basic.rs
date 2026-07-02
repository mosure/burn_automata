use crate::cli::prelude::*;

pub(crate) fn run_infer(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::Infer {
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
    } = command
    else {
        unreachable!("run_infer called with the wrong command variant");
    };

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

    Ok(())
}

pub(crate) fn run_train(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::Train {
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
    } = command
    else {
        unreachable!("run_train called with the wrong command variant");
    };

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

    Ok(())
}

pub(crate) fn run_import(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::Import { input, output } = command else {
        unreachable!("run_import called with the wrong command variant");
    };

    let report = import_model(input, output)?;
    println!("{}", serde_json::to_string_pretty(&report)?);

    Ok(())
}

pub(crate) fn run_manifest(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::Manifest { preset, output } = command else {
        unreachable!("run_manifest called with the wrong command variant");
    };

    let preset: AutomataPreset = preset.into();
    let (config, hashgrid) = NpaConfig::for_preset(preset);
    let model = NpaModel::seeded(config, 42);
    let manifest =
        BpkModelManifest::from_model(&model, hashgrid, Some(format!("seeded-rust:{preset:?}")));
    crate::import::save_manifest(&output, &manifest)?;
    println!("wrote {}", output.display());

    Ok(())
}
