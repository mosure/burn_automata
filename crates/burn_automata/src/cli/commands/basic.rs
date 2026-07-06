use crate::cli::prelude::*;
use std::collections::HashMap;

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
        optimizer,
        training_device,
        adam_beta1,
        adam_beta2,
        adam_epsilon,
        rounds,
        report_interval,
        target_model,
        target_seed,
        zero_update,
        student_seed,
        batch_source,
        rollout_particles,
        rollout_steps,
        rollouts,
        temporal_samples,
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
    if rounds == 0 {
        return Err(std::io::Error::other("--rounds must be greater than zero").into());
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
            temporal_samples,
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
    let cfg = SgdConfig {
        learning_rate,
        weight_decay,
        grad_clip_norm,
    };
    let optimizer_config = match optimizer {
        TrainingOptimizerArg::Sgd => SupervisedOptimizerConfig::Sgd(cfg),
        TrainingOptimizerArg::AdamW => SupervisedOptimizerConfig::AdamW(AdamWConfig {
            learning_rate,
            weight_decay,
            grad_clip_norm,
            beta1: adam_beta1,
            beta2: adam_beta2,
            epsilon: adam_epsilon,
        }),
    };
    let actual_training_device = resolve_training_device(training_device)?;
    let mut completed_steps = 0;
    let mut total_rows_seen = 0;
    let mut initial_loss = None;
    let mut final_loss = 0.0;
    let mut best_loss = f32::INFINITY;
    let mut history = Vec::new();
    for round in 0..rounds {
        let round_steps = distributed_round_steps(steps, rounds, round);
        let round_seed = student_seed.wrapping_add((round as u64).wrapping_mul(0x517c_c1b7));
        let batch = train_batch_for_round(
            &model,
            teacher.as_ref(),
            &hashgrid,
            target,
            batch_source,
            rows,
            round_seed,
            rollout_particles,
            rollout_steps,
            rollouts,
            temporal_samples,
            rollout_update_prob,
            seed_scale,
            seed_mode,
        )?;
        let round_report = run_training_on_device(
            actual_training_device,
            &mut model,
            &batch,
            TrainingRunConfig {
                steps: round_steps,
                report_interval,
                sgd: cfg,
            },
            optimizer_config,
        )?;
        initial_loss.get_or_insert(round_report.initial_loss);
        final_loss = round_report.final_loss;
        best_loss = best_loss.min(round_report.best_loss);
        total_rows_seen += round_report.rows;
        history.extend(round_report.history.into_iter().map(|mut entry| {
            entry.step += completed_steps;
            entry
        }));
        completed_steps += round_steps;
    }
    let report = TrainingRunReport {
        steps,
        rows,
        initial_loss: initial_loss.unwrap_or_default(),
        final_loss,
        best_loss,
        history,
    };
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
        objective: "supervised_update_distillation",
        target_source: training_source,
        student_seed,
        sgd: cfg,
        optimizer: optimizer_config,
        training_device: actual_training_device,
        rounds,
        total_rows_seen,
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

pub(crate) fn resolve_training_device(
    requested: TrainingDeviceArg,
) -> Result<TrainingDeviceArg, Box<dyn std::error::Error>> {
    match requested {
        TrainingDeviceArg::Auto => {
            #[cfg(feature = "backend_wgpu")]
            {
                Ok(TrainingDeviceArg::Gpu)
            }
            #[cfg(not(feature = "backend_wgpu"))]
            {
                Ok(TrainingDeviceArg::Cpu)
            }
        }
        TrainingDeviceArg::Cpu => Ok(TrainingDeviceArg::Cpu),
        TrainingDeviceArg::Gpu => {
            #[cfg(feature = "backend_wgpu")]
            {
                Ok(TrainingDeviceArg::Gpu)
            }
            #[cfg(not(feature = "backend_wgpu"))]
            {
                Err(std::io::Error::other(
                    "--training-device gpu requires building burn_automata with --features backend_wgpu",
                )
                .into())
            }
        }
    }
}

pub(crate) fn run_training_on_device(
    training_device: TrainingDeviceArg,
    model: &mut NpaModel,
    batch: &SupervisedBatch,
    cfg: TrainingRunConfig,
    optimizer: SupervisedOptimizerConfig,
) -> Result<TrainingRunReport, Box<dyn std::error::Error>> {
    match training_device {
        TrainingDeviceArg::Auto => unreachable!("training device should be resolved before run"),
        TrainingDeviceArg::Cpu => Ok(run_supervised_training_with_optimizer(
            model, batch, cfg, optimizer,
        )?),
        TrainingDeviceArg::Gpu => {
            #[cfg(feature = "backend_wgpu")]
            {
                Ok(run_supervised_training_wgpu(model, batch, cfg, optimizer)?)
            }
            #[cfg(not(feature = "backend_wgpu"))]
            {
                let _ = (model, batch, cfg, optimizer);
                Err(std::io::Error::other(
                    "--training-device gpu requires building burn_automata with --features backend_wgpu",
                )
                .into())
            }
        }
    }
}

fn distributed_round_steps(total_steps: usize, rounds: usize, round: usize) -> usize {
    let base = total_steps / rounds;
    let remainder = total_steps % rounds;
    base + usize::from(round < remainder)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn train_batch_for_round(
    model: &NpaModel,
    teacher: Option<&NpaModel>,
    hashgrid: &burn_automata_kernels::HashGridConfig,
    target: SupervisedTarget<'_>,
    batch_source: TrainingBatchArg,
    rows: usize,
    seed: u64,
    rollout_particles: usize,
    rollout_steps: usize,
    rollouts: usize,
    temporal_samples: usize,
    rollout_update_prob: f32,
    seed_scale: f32,
    seed_mode: ParticleSeed,
) -> Result<SupervisedBatch, Box<dyn std::error::Error>> {
    Ok(match batch_source {
        TrainingBatchArg::Features => feature_supervised_batch(
            model,
            target,
            FeatureBatchConfig {
                rows,
                seed,
                ..FeatureBatchConfig::default()
            },
        )?,
        TrainingBatchArg::Rollout => {
            let rollout_model = teacher.unwrap_or(model);
            rollout_supervised_batch_from_model(
                model,
                rollout_model,
                hashgrid,
                target,
                RolloutSupervisionConfig {
                    max_rows: rows,
                    particle_count: rollout_particles,
                    rollout_steps,
                    rollouts,
                    temporal_samples,
                    update_prob: rollout_update_prob,
                    seed,
                    seed_scale,
                    seed_mode,
                    ..RolloutSupervisionConfig::default()
                },
            )?
        }
    })
}

pub(crate) fn run_import(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::Import { input, output } = command else {
        unreachable!("run_import called with the wrong command variant");
    };

    let report = import_model(input, output)?;
    println!("{}", serde_json::to_string_pretty(&report)?);

    Ok(())
}

pub(crate) fn run_materialize_adapter(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::MaterializeAdapter {
        base_model,
        adapter,
        output,
    } = command
    else {
        unreachable!("run_materialize_adapter called with the wrong command variant");
    };

    let base_manifest = crate::import::load_manifest(&base_model)?;
    let adapter_manifest = crate::import::load_adapter_manifest(&adapter)?;
    let materialized = adapter_manifest.materialize(&base_manifest)?;
    crate::import::save_manifest(&output, &materialized)?;
    println!(
        "wrote {} base={} adapter={} parameters={}",
        output.display(),
        base_model.display(),
        adapter.display(),
        crate::import::parameter_count(&materialized),
    );

    Ok(())
}

#[derive(Deserialize)]
struct ExactAdapterBankSource {
    entries: Vec<ExactAdapterBankEntry>,
}

#[derive(Clone, Deserialize, Serialize)]
struct ExactAdapterBankEntry {
    slug: String,
    split: String,
    title: Option<String>,
    group: Option<String>,
    condition: String,
    adapter_output: String,
    #[serde(default)]
    target_source_width: usize,
    #[serde(default)]
    target_source_height: usize,
    #[serde(default)]
    target_points: usize,
    #[serde(default)]
    last_train_loss: Option<f32>,
    #[serde(default)]
    adapter_parameter_count: usize,
}

#[derive(Serialize)]
struct ExactAdapterBankOutput {
    base_model: String,
    adapter_rank: usize,
    adapter_alpha: f32,
    entries: Vec<ExactAdapterBankEntry>,
}

pub(crate) fn run_build_exact_adapter_bank(
    command: Command,
) -> Result<(), Box<dyn std::error::Error>> {
    let Command::BuildExactAdapterBank {
        base_model,
        source_adapter_bank,
        oracle_report,
        output_dir,
        adapter_bank_output,
        rank,
        alpha,
        force_split,
    } = command
    else {
        unreachable!("run_build_exact_adapter_bank called with the wrong command variant");
    };

    let base_manifest = crate::import::load_manifest(&base_model)?;
    let base = base_manifest.clone().into_model();
    base.validate()?;
    let exact_rank = base.config.perception_dims().max(base.config.update_dims());
    let rank = rank.unwrap_or(exact_rank * 2);
    let alpha = alpha.unwrap_or(rank as f32);
    if rank < exact_rank {
        return Err(std::io::Error::other(format!(
            "exact adapter rank {rank} is too small for this NPA; use rank >= {exact_rank}"
        ))
        .into());
    }
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(std::io::Error::other("--alpha must be finite and greater than zero").into());
    }

    let source_text = std::fs::read_to_string(&source_adapter_bank)?;
    let source_bank: ExactAdapterBankSource = serde_json::from_str(&source_text)?;
    let oracle_models = exact_oracle_models_by_slug(&oracle_report)?;
    let adapter_dir = output_dir.join("adapters");
    let adapter_bank_output =
        adapter_bank_output.unwrap_or_else(|| output_dir.join("adapter_bank.json"));
    let mut entries = Vec::new();
    for mut entry in source_bank.entries {
        let Some(target_model_path) = oracle_models.get(&entry.slug) else {
            continue;
        };
        let target_manifest = crate::import::load_manifest(target_model_path)?;
        if target_manifest.hashgrid != base_manifest.hashgrid {
            return Err(std::io::Error::other(format!(
                "oracle model hashgrid differs for {}",
                entry.slug
            ))
            .into());
        }
        let target = target_manifest.into_model();
        let adapter = NpaLowRankAdapter::exact_model_delta(&base, &target, rank, alpha)?;
        entry.adapter_parameter_count = adapter.parameter_count();
        let adapter_output = adapter_dir.join(format!("{}.adapter.json", entry.slug));
        let adapter_manifest = BpkAdapterManifest::from_adapter(
            &base_manifest,
            Some(base_model.display().to_string()),
            adapter,
            Some(format!(
                "exact-oracle-delta:{}",
                target_model_path.display()
            )),
        )?;
        crate::import::save_adapter_manifest(&adapter_output, &adapter_manifest)?;
        entry.adapter_output = adapter_output.display().to_string();
        if let Some(force_split) = &force_split {
            entry.split = force_split.clone();
        }
        entries.push(entry);
    }
    if entries.is_empty() {
        return Err(std::io::Error::other(
            "oracle report did not match any source adapter-bank entries",
        )
        .into());
    }

    let output = ExactAdapterBankOutput {
        base_model: base_model.display().to_string(),
        adapter_rank: rank,
        adapter_alpha: alpha,
        entries,
    };
    if let Some(parent) = adapter_bank_output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&adapter_bank_output, serde_json::to_string_pretty(&output)?)?;
    println!(
        "wrote {} entries={} rank={} alpha={}",
        adapter_bank_output.display(),
        output.entries.len(),
        rank,
        alpha
    );

    Ok(())
}

fn exact_oracle_models_by_slug(
    oracle_report: &Path,
) -> Result<HashMap<String, PathBuf>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(oracle_report)?;
    let value: serde_json::Value = serde_json::from_str(&text)?;
    let entries = value
        .pointer("/oracle_validation/entries")
        .or_else(|| value.get("entries"))
        .and_then(|entries| entries.as_array())
        .ok_or_else(|| {
            std::io::Error::other("oracle report must contain oracle_validation.entries or entries")
        })?;
    let report_parent = oracle_report.parent().unwrap_or_else(|| Path::new(""));
    let mut models = HashMap::new();
    for entry in entries {
        let Some(slug) = entry.get("slug").and_then(|value| value.as_str()) else {
            continue;
        };
        let Some(path) = entry
            .get("oracle_model_output")
            .and_then(|value| value.as_str())
        else {
            continue;
        };
        let mut path = PathBuf::from(path);
        if path.is_relative() && !path.exists() {
            path = report_parent.join(path);
        }
        models.insert(slug.to_string(), path);
    }
    Ok(models)
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
