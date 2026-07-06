use crate::cli::prelude::*;

use super::hyper_support::*;

pub(crate) fn run_train_hyper_2d(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::TrainHyper2d {
        preset,
        condition,
        target_model,
        catalog,
        catalog_thumbnail_dir,
        catalog_group,
        catalog_targets,
        catalog_limit,
        holdout_stride,
        holdout_offset,
        base_model,
        hyper_input,
        hyper_output,
        report_output,
        adapter_output,
        materialized_output,
        generated_output_dir,
        steps,
        rows,
        learning_rate,
        grad_clip_norm,
        weight_decay,
        adapter_rank,
        adapter_alpha,
        hyper_hidden,
        hyper_output_scale,
        condition_token_grid_width,
        condition_token_grid_height,
        hyper_seed,
        adapter_bootstrap_steps,
        adapter_bootstrap_learning_rate,
        adapter_bootstrap_grad_clip_norm,
        rollout_particles,
        rollout_steps,
        rollouts,
        rollout_update_prob,
        seed_scale,
        seed_mode,
    } = command
    else {
        unreachable!("run_train_hyper_2d called with the wrong command variant");
    };

    let preset_arg = preset;
    let preset: AutomataPreset = preset.into();
    let descriptors = resolve_hyper2d_sources(ResolveHyper2dSourcesConfig {
        preset: preset_arg,
        condition: condition.as_ref(),
        target_model: target_model.as_ref(),
        catalog: catalog.as_ref(),
        catalog_thumbnail_dir: &catalog_thumbnail_dir,
        catalog_group,
        catalog_targets: &catalog_targets,
        catalog_limit,
    })?;
    if descriptors.is_empty() {
        return Err(std::io::Error::other("no train-hyper2d examples matched").into());
    }
    if (adapter_output.is_some() || materialized_output.is_some()) && descriptors.len() != 1 {
        return Err(std::io::Error::other(
            "--adapter-output and --materialized-output are single-example outputs; use --generated-output-dir for catalog runs",
        )
        .into());
    }

    let first_target_manifest = crate::import::load_manifest(&descriptors[0].target_path)?;
    let (base_manifest, effective_base_model_path) = if let Some(path) = &base_model {
        let manifest = crate::import::load_manifest(path)?;
        if manifest.hashgrid != first_target_manifest.hashgrid {
            return Err(std::io::Error::other(
                "base model hashgrid must match target model hashgrid",
            )
            .into());
        }
        (manifest, Some(path.clone()))
    } else {
        (
            first_target_manifest.clone(),
            Some(descriptors[0].target_path.clone()),
        )
    };
    let base = base_manifest.clone().into_model();
    if base.config.spatial_dims != 2 {
        return Err(std::io::Error::other("train-hyper2d requires a 2D NPA config").into());
    }
    let anchor_condition = if base_model.is_none() {
        Some(load_condition_image_2d(&descriptors[0].condition_path)?)
    } else {
        None
    };
    let seed_mode: ParticleSeed = seed_mode.into();
    let loaded_examples = load_hyper2d_examples(
        &base,
        &base_manifest,
        &descriptors,
        None,
        rows,
        rollout_particles,
        rollout_steps,
        rollouts,
        rollout_update_prob,
        seed_scale,
        preset,
        seed_mode,
        hyper_seed,
    )?;
    let (train_loaded, holdout_loaded) =
        split_hyper2d_examples(loaded_examples, holdout_stride, holdout_offset)?;
    let train_examples = flow_examples(&train_loaded);
    let holdout_examples = flow_examples(&holdout_loaded);

    let hyper_config = HyperNpa2dConfig {
        condition_encoder: ConditionEncoder2d::SummaryTokens,
        condition_feature_dims: condition_feature_dims_for_encoder(
            ConditionEncoder2d::SummaryTokens,
            condition_token_grid_width,
            condition_token_grid_height,
        )?,
        condition_token_grid_width,
        condition_token_grid_height,
        hidden_dims: hyper_hidden,
        adapter_rank,
        adapter_alpha,
        adapter_bias_correction: false,
        output_activation: HyperNpa2dOutputActivation::Tanh,
        output_scale: hyper_output_scale,
    };
    let mut hyper = if let Some(path) = &hyper_input {
        let loaded = load_hyper_2d(path)?;
        loaded.validate()?;
        loaded
    } else {
        HyperNpa2d::seeded(base.config.clone(), hyper_config, hyper_seed)?
    };
    if hyper.npa_config != base.config {
        return Err(std::io::Error::other(
            "hyper checkpoint NPA config must match base model config",
        )
        .into());
    }
    if hyper.anchor_input.is_none()
        && let Some(anchor_condition) = &anchor_condition
    {
        hyper.set_anchor_condition(anchor_condition)?;
    }
    let sgd = SgdConfig {
        learning_rate,
        weight_decay,
        grad_clip_norm,
    };
    let initial_loss = hyper_rectified_flow_loss(&base, &hyper, &train_examples)?;
    let holdout_initial_loss = if holdout_examples.is_empty() {
        None
    } else {
        Some(hyper_rectified_flow_loss(&base, &hyper, &holdout_examples)?)
    };
    let train_initial_example_losses = example_losses(&base, &hyper, &train_loaded)?;
    let holdout_initial_example_losses = example_losses(&base, &hyper, &holdout_loaded)?;
    let bootstrap_sgd = SgdConfig {
        learning_rate: adapter_bootstrap_learning_rate.unwrap_or(learning_rate),
        weight_decay,
        grad_clip_norm: adapter_bootstrap_grad_clip_norm.unwrap_or(grad_clip_norm),
    };
    let adapter_bootstrap = if adapter_bootstrap_steps > 0 {
        Some(bootstrap_hyper2d_adapters(
            &base,
            &train_loaded,
            hyper.config.adapter_rank,
            hyper.config.adapter_alpha,
            hyper_seed,
            TrainingRunConfig {
                steps: adapter_bootstrap_steps,
                report_interval: adapter_bootstrap_steps,
                sgd: bootstrap_sgd,
            },
        )?)
    } else {
        None
    };
    let mut best_loss = initial_loss;
    let mut final_loss = initial_loss;
    let mut best_step = 0;
    let mut best_hyper = hyper.clone();
    let adapter_bootstrap_reports = adapter_bootstrap
        .as_ref()
        .map(|bootstrap| bootstrap.reports.clone())
        .unwrap_or_default();
    if let Some(bootstrap) = &adapter_bootstrap {
        initialize_hyper_adapter_residual_fit(&mut hyper, &bootstrap.examples)?;
        for _ in 0..adapter_bootstrap_steps {
            hyper_adapter_regression_train_step(&mut hyper, &bootstrap.examples, bootstrap_sgd)?;
        }
        final_loss = hyper_rectified_flow_loss(&base, &hyper, &train_examples)?;
        if final_loss < best_loss {
            best_loss = final_loss;
            best_hyper = hyper.clone();
        }
    }
    let mut history = Vec::with_capacity(steps);
    for step in 1..=steps {
        let step_report = hyper_rectified_flow_train_step(&base, &mut hyper, &train_examples, sgd)?;
        final_loss = hyper_rectified_flow_loss(&base, &hyper, &train_examples)?;
        let holdout_loss = if holdout_examples.is_empty() {
            None
        } else {
            Some(hyper_rectified_flow_loss(&base, &hyper, &holdout_examples)?)
        };
        if final_loss < best_loss {
            best_loss = final_loss;
            best_step = step;
            best_hyper = hyper.clone();
        }
        history.push(CliHyper2dHistoryEntry {
            step,
            loss: final_loss,
            holdout_loss,
            grad_norm: step_report.grad_norm,
            grad_scale: step_report.grad_scale,
        });
    }
    if best_loss < final_loss {
        hyper = best_hyper;
        final_loss = best_loss;
    }
    let holdout_final_loss = if holdout_examples.is_empty() {
        None
    } else {
        Some(hyper_rectified_flow_loss(&base, &hyper, &holdout_examples)?)
    };

    save_hyper_2d(&hyper_output, &hyper)?;
    if let Some(example) = train_loaded.first().filter(|_| descriptors.len() == 1) {
        let conditioned = generate_conditioned_npa_2d(
            &base,
            &hyper,
            &example.condition,
            ParticlePriorConfig::default(),
        )?;
        save_conditioned_outputs(
            &base_manifest,
            effective_base_model_path.as_ref(),
            &example.descriptor.condition_path,
            &conditioned.adapter,
            &conditioned.model,
            adapter_output.as_ref(),
            materialized_output.as_ref(),
        )?;
    }
    if let Some(output_dir) = &generated_output_dir {
        save_generated_examples(
            &base,
            &base_manifest,
            effective_base_model_path.as_ref(),
            &hyper,
            &train_loaded,
            output_dir,
        )?;
        save_generated_examples(
            &base,
            &base_manifest,
            effective_base_model_path.as_ref(),
            &hyper,
            &holdout_loaded,
            output_dir,
        )?;
    }

    let train_example_reports = example_reports(
        &base,
        &base_manifest,
        &hyper,
        &train_loaded,
        &train_initial_example_losses,
        None,
        None,
    )?;
    let holdout_example_reports = example_reports(
        &base,
        &base_manifest,
        &hyper,
        &holdout_loaded,
        &holdout_initial_example_losses,
        None,
        None,
    )?;
    let representative = train_loaded
        .first()
        .expect("train split is validated to be non-empty");
    let report = CliHyper2dTrainingReport {
        preset,
        condition: condition.as_ref().map(|path| path.display().to_string()),
        catalog: catalog.as_ref().map(|path| path.display().to_string()),
        catalog_group,
        catalog_targets,
        base_model: effective_base_model_path
            .as_ref()
            .map(|path| path.display().to_string()),
        target_model: target_model.as_ref().map(|path| path.display().to_string()),
        hyper_input: hyper_input.as_ref().map(|path| path.display().to_string()),
        hyper_output: hyper_output.display().to_string(),
        adapter_output: adapter_output
            .as_ref()
            .map(|path| path.display().to_string()),
        materialized_output: materialized_output
            .as_ref()
            .map(|path| path.display().to_string()),
        generated_output_dir: generated_output_dir
            .as_ref()
            .map(|path| path.display().to_string()),
        npa_config: hyper.npa_config.clone(),
        hyper_config: hyper.config,
        sgd,
        rollout_supervision: CliRolloutSupervisionReport {
            particle_count: representative.particle_count,
            rollout_steps: representative.rollout_steps,
            rollouts: representative.rollouts,
            temporal_samples: 1,
            update_prob: representative.update_prob,
            seed_scale: representative.seed_scale,
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
        },
        initial_loss,
        holdout_initial_loss,
        final_loss,
        holdout_final_loss,
        best_loss,
        best_step,
        history,
        adapter_bootstrap: adapter_bootstrap_reports,
        train_examples: train_example_reports,
        holdout_examples: holdout_example_reports,
        adapter_parameter_count: hyper.adapter_parameter_count(),
        materialized_parameter_count: crate::import::parameter_count(&base_manifest),
    };
    write_pretty_json(&report_output, &report)?;
    println!(
        "wrote {} examples={} holdout={} final_loss={:.6} best_loss={:.6} hyper={}",
        report_output.display(),
        train_examples.len(),
        holdout_examples.len(),
        final_loss,
        best_loss,
        hyper_output.display()
    );

    Ok(())
}

pub(crate) fn run_infer_hyper_2d(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::InferHyper2d {
        preset,
        condition,
        hyper,
        base_model,
        report_output,
        adapter_output,
        materialized_output,
        rollout_output,
        steps,
        particles,
        update_prob,
        seed,
        seed_scale,
        seed_mode,
        gpu,
        neighbor_mode,
        bucket_capacity,
        min_particles,
        max_particles,
        min_seed_scale,
        max_seed_scale,
    } = command
    else {
        unreachable!("run_infer_hyper_2d called with the wrong command variant");
    };

    #[cfg(not(feature = "gpu_wgpu"))]
    let _ = (neighbor_mode, bucket_capacity);
    let preset: AutomataPreset = preset.into();
    let (preset_config, preset_grid) = NpaConfig::for_preset(preset);
    let base_manifest = if let Some(path) = &base_model {
        crate::import::load_manifest(path)?
    } else {
        let base = NpaModel::seeded(preset_config, 42);
        BpkModelManifest::from_model(
            &base,
            preset_grid,
            Some("seeded-hyper2d-infer-base:42".to_string()),
        )
    };
    let base = base_manifest.clone().into_model();
    if base.config.spatial_dims != 2 {
        return Err(std::io::Error::other("infer-hyper2d requires a 2D NPA config").into());
    }
    let hyper_model = load_hyper_2d(&hyper)?;
    if hyper_model.npa_config != base.config {
        return Err(std::io::Error::other(
            "hyper checkpoint NPA config must match base model config",
        )
        .into());
    }
    let condition_image = load_condition_image_2d(&condition)?;
    let prior_config = ParticlePriorConfig {
        min_particles,
        max_particles,
        min_seed_scale,
        max_seed_scale,
    };
    let conditioned =
        generate_conditioned_npa_2d(&base, &hyper_model, &condition_image, prior_config)?;
    save_conditioned_outputs(
        &base_manifest,
        base_model.as_ref(),
        &condition,
        &conditioned.adapter,
        &conditioned.model,
        adapter_output.as_ref(),
        materialized_output.as_ref(),
    )?;

    let actual_particles = particles.unwrap_or(conditioned.prior.particle_count);
    let actual_seed_scale = seed_scale.unwrap_or(conditioned.prior.seed_scale);
    let actual_seed = seed.unwrap_or_else(|| RolloutConfig::default().seed);
    if let Some(path) = &rollout_output {
        let rollout_cfg = RolloutConfig {
            steps,
            particle_count: actual_particles,
            update_prob,
            seed: actual_seed,
            seed_scale: actual_seed_scale,
            ..RolloutConfig::default()
        };
        let trace = if gpu {
            #[cfg(feature = "gpu_wgpu")]
            {
                gpu_rollout_trace(
                    &conditioned.model,
                    &base_manifest.hashgrid,
                    &rollout_cfg,
                    seed_mode.into(),
                    wgpu_neighbor_mode(neighbor_mode, bucket_capacity),
                )?
            }
            #[cfg(not(feature = "gpu_wgpu"))]
            {
                return Err(std::io::Error::other(
                    "infer-hyper2d --gpu requires building burn_automata with --features gpu_wgpu",
                )
                .into());
            }
        } else {
            run_rollout(
                &conditioned.model,
                &base_manifest.hashgrid,
                &rollout_cfg,
                seed_mode.into(),
            )?
        };
        write_pretty_json(path, &trace)?;
    }

    let materialized_manifest = BpkModelManifest::from_model(
        &conditioned.model,
        base_manifest.hashgrid.clone(),
        Some(format!("hyper2d-materialized:{}", condition.display())),
    );
    let report = CliHyper2dInferReport {
        preset,
        condition: condition.display().to_string(),
        base_model: base_model.as_ref().map(|path| path.display().to_string()),
        hyper: hyper.display().to_string(),
        adapter_output: adapter_output
            .as_ref()
            .map(|path| path.display().to_string()),
        materialized_output: materialized_output
            .as_ref()
            .map(|path| path.display().to_string()),
        rollout_output: rollout_output
            .as_ref()
            .map(|path| path.display().to_string()),
        npa_config: hyper_model.npa_config.clone(),
        hyper_config: hyper_model.config,
        condition_summary: conditioned.summary,
        prior: conditioned.prior,
        rollout_particles: rollout_output.as_ref().map(|_| actual_particles),
        rollout_steps: rollout_output.as_ref().map(|_| steps),
        seed: rollout_output.as_ref().map(|_| actual_seed),
        seed_scale: rollout_output.as_ref().map(|_| actual_seed_scale),
        seed_mode: seed_mode.into(),
        adapter_parameter_count: conditioned.adapter.parameter_count(),
        materialized_parameter_count: crate::import::parameter_count(&materialized_manifest),
    };
    write_pretty_json(&report_output, &report)?;
    println!("wrote {}", report_output.display());

    Ok(())
}

pub(crate) fn run_eval_hyper_2d(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::EvalHyper2d {
        preset,
        condition,
        target_model,
        catalog,
        catalog_thumbnail_dir,
        catalog_group,
        catalog_targets,
        catalog_limit,
        holdout_stride,
        holdout_offset,
        hyper,
        base_model,
        report_output,
        generated_output_dir,
        rows,
        rollout_particles,
        rollout_steps,
        rollouts,
        rollout_update_prob,
        seed_scale,
        seed_mode,
        seed,
        image_metrics,
        image_metric_size,
        image_metric_steps,
        image_metric_particles,
        image_metric_update_prob,
        image_metric_sigma,
        image_metric_threshold,
        dynamics_metrics,
        dynamics_metric_particles,
        dynamics_metric_steps,
        dynamics_metric_update_prob,
        dynamics_metric_image_size,
        dynamics_metric_sigma,
    } = command
    else {
        unreachable!("run_eval_hyper_2d called with the wrong command variant");
    };

    let preset_arg = preset;
    let preset: AutomataPreset = preset.into();
    let descriptors = resolve_hyper2d_sources(ResolveHyper2dSourcesConfig {
        preset: preset_arg,
        condition: condition.as_ref(),
        target_model: target_model.as_ref(),
        catalog: catalog.as_ref(),
        catalog_thumbnail_dir: &catalog_thumbnail_dir,
        catalog_group,
        catalog_targets: &catalog_targets,
        catalog_limit,
    })?;
    if descriptors.is_empty() {
        return Err(std::io::Error::other("no eval-hyper2d examples matched").into());
    }

    let hyper_model = load_hyper_2d(&hyper)?;
    let first_target_manifest = crate::import::load_manifest(&descriptors[0].target_path)?;
    let (base_manifest, effective_base_model_path) = if let Some(path) = &base_model {
        let manifest = crate::import::load_manifest(path)?;
        if manifest.hashgrid != first_target_manifest.hashgrid {
            return Err(std::io::Error::other(
                "base model hashgrid must match target model hashgrid",
            )
            .into());
        }
        (manifest, Some(path.clone()))
    } else {
        (
            first_target_manifest.clone(),
            Some(descriptors[0].target_path.clone()),
        )
    };
    let base = base_manifest.clone().into_model();
    if base.config.spatial_dims != 2 {
        return Err(std::io::Error::other("eval-hyper2d requires a 2D NPA config").into());
    }
    if hyper_model.npa_config != base.config {
        return Err(std::io::Error::other(
            "hyper checkpoint NPA config must match base model config",
        )
        .into());
    }

    let seed_mode: ParticleSeed = seed_mode.into();
    let loaded_examples = load_hyper2d_examples(
        &base,
        &base_manifest,
        &descriptors,
        None,
        rows,
        rollout_particles,
        rollout_steps,
        rollouts,
        rollout_update_prob,
        seed_scale,
        preset,
        seed_mode,
        seed,
    )?;
    let (train_loaded, holdout_loaded) =
        split_hyper2d_examples(loaded_examples, holdout_stride, holdout_offset)?;
    let train_examples = flow_examples(&train_loaded);
    let holdout_examples = flow_examples(&holdout_loaded);
    let train_loss = hyper_rectified_flow_loss(&base, &hyper_model, &train_examples)?;
    let holdout_loss = if holdout_examples.is_empty() {
        None
    } else {
        Some(hyper_rectified_flow_loss(
            &base,
            &hyper_model,
            &holdout_examples,
        )?)
    };
    let train_example_losses = example_losses(&base, &hyper_model, &train_loaded)?;
    let holdout_example_losses = example_losses(&base, &hyper_model, &holdout_loaded)?;
    let image_metric_config = image_metrics.then_some(Hyper2dImageMetricConfig {
        image_size: image_metric_size,
        rollout_steps: image_metric_steps,
        particle_count: image_metric_particles,
        update_prob: image_metric_update_prob,
        sigma: image_metric_sigma,
        threshold: image_metric_threshold,
    });
    let dynamics_metric_config = dynamics_metrics.then_some(Hyper2dDynamicsMetricConfig {
        particle_count: dynamics_metric_particles,
        rollout_steps: dynamics_metric_steps,
        update_prob: dynamics_metric_update_prob,
        image_size: dynamics_metric_image_size,
        sigma: dynamics_metric_sigma,
    });
    let train_example_reports = example_reports(
        &base,
        &base_manifest,
        &hyper_model,
        &train_loaded,
        &train_example_losses,
        image_metric_config,
        dynamics_metric_config,
    )?;
    let holdout_example_reports = example_reports(
        &base,
        &base_manifest,
        &hyper_model,
        &holdout_loaded,
        &holdout_example_losses,
        image_metric_config,
        dynamics_metric_config,
    )?;
    if let Some(output_dir) = &generated_output_dir {
        save_generated_examples(
            &base,
            &base_manifest,
            effective_base_model_path.as_ref(),
            &hyper_model,
            &train_loaded,
            output_dir,
        )?;
        save_generated_examples(
            &base,
            &base_manifest,
            effective_base_model_path.as_ref(),
            &hyper_model,
            &holdout_loaded,
            output_dir,
        )?;
    }

    let representative = train_loaded
        .first()
        .expect("train split is validated to be non-empty");
    let report = CliHyper2dEvalReport {
        preset,
        condition: condition.as_ref().map(|path| path.display().to_string()),
        catalog: catalog.as_ref().map(|path| path.display().to_string()),
        catalog_group,
        catalog_targets,
        base_model: effective_base_model_path
            .as_ref()
            .map(|path| path.display().to_string()),
        hyper: hyper.display().to_string(),
        report_output: report_output.display().to_string(),
        generated_output_dir: generated_output_dir
            .as_ref()
            .map(|path| path.display().to_string()),
        npa_config: hyper_model.npa_config.clone(),
        hyper_config: hyper_model.config,
        rollout_supervision: CliRolloutSupervisionReport {
            particle_count: representative.particle_count,
            rollout_steps: representative.rollout_steps,
            rollouts: representative.rollouts,
            temporal_samples: 1,
            update_prob: representative.update_prob,
            seed_scale: representative.seed_scale,
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
        },
        train_loss,
        holdout_loss,
        train_examples: train_example_reports,
        holdout_examples: holdout_example_reports,
        adapter_parameter_count: hyper_model.adapter_parameter_count(),
        materialized_parameter_count: crate::import::parameter_count(&base_manifest),
    };
    write_pretty_json(&report_output, &report)?;
    println!(
        "wrote {} examples={} holdout={} train_loss={:.6}{}",
        report_output.display(),
        train_examples.len(),
        holdout_examples.len(),
        train_loss,
        holdout_loss
            .map(|loss| format!(" holdout_loss={loss:.6}"))
            .unwrap_or_default()
    );

    Ok(())
}
