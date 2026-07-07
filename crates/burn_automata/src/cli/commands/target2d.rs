use crate::cli::prelude::*;
use crate::target2d::{Target2dRenderedSplat, render_rollout_2d_splat, render_target_2d_splat};

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Target2dTrainExperimentConfig {
    preset: Option<String>,
    source: Target2dTrainSourceConfig,
    output: Target2dTrainOutputConfig,
    training: Target2dTrainTrainingConfig,
    optimizer: Target2dTrainOptimizerConfig,
    target: Target2dTrainTargetConfig,
    loss: Target2dTrainLossConfig,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Target2dTrainSourceConfig {
    target_image: Option<PathBuf>,
    reference_model: Option<PathBuf>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Target2dTrainOutputConfig {
    report: Option<PathBuf>,
    model: Option<PathBuf>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Target2dTrainTrainingConfig {
    experimental: Option<bool>,
    device: Option<String>,
    gpu_backend: Option<String>,
    epochs: Option<usize>,
    repetitions: Option<usize>,
    report_interval: Option<usize>,
    batch_size: Option<usize>,
    pool_size: Option<usize>,
    particles: Option<usize>,
    step_min: Option<usize>,
    step_max: Option<usize>,
    inject_seed_interval: Option<usize>,
    update_prob: Option<f32>,
    seed: Option<u64>,
    student_seed: Option<u64>,
    seed_scale: Option<f32>,
    seed_mode: Option<String>,
    brush_size: Option<f32>,
    per_parameter_grad_normalization: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Target2dTrainOptimizerConfig {
    learning_rate: Option<f32>,
    weight_decay: Option<f32>,
    grad_clip_norm: Option<f32>,
    adam_beta1: Option<f32>,
    adam_beta2: Option<f32>,
    adam_epsilon: Option<f32>,
    scheduler_milestones: Option<Vec<usize>>,
    scheduler_gamma: Option<f32>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Target2dTrainTargetConfig {
    points: Option<usize>,
    image_size: Option<usize>,
    threshold: Option<f32>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Target2dTrainLossConfig {
    image_size: Option<usize>,
    splat_sigma: Option<f32>,
    center: Option<bool>,
    splat_loss_weight: Option<f32>,
    color_loss_weight: Option<f32>,
    density_loss_weight: Option<f32>,
    background_density_loss_weight: Option<f32>,
    foreground_density_loss_weight: Option<f32>,
    shape_chamfer_loss_weight: Option<f32>,
    displacement_regularizer_weight: Option<f32>,
    overflow_regularizer_weight: Option<f32>,
    bound_regularizer_weight: Option<f32>,
}

pub(crate) fn run_eval_target_2d(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::EvalTarget2d {
        preset,
        model,
        target_image,
        reference_model,
        particles,
        steps,
        update_prob,
        seed,
        seed_scale,
        seed_mode,
        target_points,
        target_image_size,
        target_threshold,
        image_size,
        splat_sigma,
        center_loss,
        splat_loss_weight,
        color_loss_weight,
        density_loss_weight,
        background_density_loss_weight,
        foreground_density_loss_weight,
        displacement_regularizer_weight,
        overflow_regularizer_weight,
        bound_regularizer_weight,
        render_output_dir,
        wgpu_render_diagnostic,
        output,
    } = command
    else {
        unreachable!("run_eval_target_2d called with the wrong command variant");
    };

    validate_target2d_rollout_args(particles, steps, update_prob)?;
    let preset: AutomataPreset = preset.into();
    let seed_scale = seed_scale.unwrap_or_else(|| NpaConfig::seed_scale_for_preset(preset));
    let seed_mode: ParticleSeed = seed_mode.into();
    let loss_config = target2d_loss_config(
        image_size,
        splat_sigma,
        center_loss,
        splat_loss_weight,
        color_loss_weight,
        density_loss_weight,
        background_density_loss_weight,
        foreground_density_loss_weight,
        displacement_regularizer_weight,
        overflow_regularizer_weight,
        bound_regularizer_weight,
    )?;
    let target = load_target_image_2d_adaptive(
        &target_image,
        target_threshold,
        target_points,
        target_image_size,
    )?;
    let loss = evaluate_target2d_model_loss(
        &model,
        &target,
        loss_config,
        particles,
        steps,
        update_prob,
        seed,
        seed_scale,
        seed_mode,
    )?;
    let reference_loss = reference_model
        .as_ref()
        .map(|path| {
            evaluate_target2d_model_loss(
                path,
                &target,
                loss_config,
                particles,
                steps,
                update_prob,
                seed,
                seed_scale,
                seed_mode,
            )
        })
        .transpose()?;
    let (gap, ratio) = loss_gap_and_ratio(loss.total_loss, reference_loss);
    let render_diagnostics = target2d_render_diagnostics(
        render_output_dir.as_deref(),
        wgpu_render_diagnostic,
        &model,
        reference_model.as_deref(),
        &target,
        loss_config,
        particles,
        steps,
        update_prob,
        seed,
        seed_scale,
        seed_mode,
    )?;
    let report = CliTarget2dEvalReport {
        preset,
        model: model.display().to_string(),
        target_image: target_image.display().to_string(),
        reference_model: reference_model
            .as_ref()
            .map(|path| path.display().to_string()),
        particle_count: particles,
        rollout_steps: steps,
        update_prob,
        seed,
        seed_scale,
        seed_mode,
        loss_config,
        target_source_width: target.source_width,
        target_source_height: target.source_height,
        target_points: target.point_count(),
        loss,
        reference_loss,
        total_loss_gap_to_reference: gap,
        total_loss_ratio_to_reference: ratio,
        render_diagnostics,
    };
    write_json_report(&output, &report)?;
    println!(
        "wrote {} target_points={} loss={:.6} reference_loss={} ratio={}",
        output.display(),
        report.target_points,
        report.loss.total_loss,
        report
            .reference_loss
            .map(|loss| format!("{:.6}", loss.total_loss))
            .unwrap_or_else(|| "none".to_string()),
        report
            .total_loss_ratio_to_reference
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "none".to_string()),
    );
    Ok(())
}

pub(crate) fn run_train_target_2d(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::TrainTarget2d {
        config,
        experimental,
        preset,
        target_image,
        output,
        training_device,
        gpu_backend,
        model_output,
        reference_model,
        epochs,
        repetitions,
        report_interval,
        batch_size,
        pool_size,
        particles,
        step_min,
        step_max,
        inject_seed_interval,
        update_prob,
        seed,
        student_seed,
        seed_scale,
        seed_mode,
        brush_size,
        learning_rate,
        weight_decay,
        grad_clip_norm,
        adam_beta1,
        adam_beta2,
        adam_epsilon,
        scheduler_milestones,
        scheduler_gamma,
        per_parameter_grad_normalization,
        target_points,
        target_image_size,
        target_threshold,
        image_size,
        splat_sigma,
        center_loss,
        splat_loss_weight,
        color_loss_weight,
        density_loss_weight,
        background_density_loss_weight,
        foreground_density_loss_weight,
        displacement_regularizer_weight,
        overflow_regularizer_weight,
        bound_regularizer_weight,
    } = command
    else {
        unreachable!("run_train_target_2d called with the wrong command variant");
    };

    let file_config = load_target2d_train_config(config.as_deref())?;
    let experimental = file_config.training.experimental.unwrap_or(experimental);
    if !experimental {
        return Err(std::io::Error::other(
            "train-target2d is experimental and does not yet pass official SelfOrg-NPA parity; pass --experimental or training.experimental=true for diagnostics",
        )
        .into());
    }
    let preset = target2d_config_value_enum("preset", file_config.preset, preset)?;
    let target_image = file_config
        .source
        .target_image
        .or(target_image)
        .ok_or_else(|| {
            std::io::Error::other("train-target2d requires --target-image or source.target_image")
        })?;
    let reference_model = file_config.source.reference_model.or(reference_model);
    let output = file_config.output.report.unwrap_or(output);
    let model_output = file_config.output.model.or(model_output);
    let training_device = target2d_config_value_enum(
        "training.device",
        file_config.training.device,
        training_device,
    )?;
    let gpu_backend = target2d_config_value_enum(
        "training.gpu_backend",
        file_config.training.gpu_backend,
        gpu_backend,
    )?;
    let epochs = file_config.training.epochs.unwrap_or(epochs);
    let repetitions = file_config.training.repetitions.unwrap_or(repetitions);
    let report_interval = file_config
        .training
        .report_interval
        .unwrap_or(report_interval);
    let batch_size = file_config.training.batch_size.unwrap_or(batch_size);
    let pool_size = file_config.training.pool_size.unwrap_or(pool_size);
    let particles = file_config.training.particles.unwrap_or(particles);
    let step_min = file_config.training.step_min.unwrap_or(step_min);
    let step_max = file_config.training.step_max.unwrap_or(step_max);
    let inject_seed_interval = file_config
        .training
        .inject_seed_interval
        .unwrap_or(inject_seed_interval);
    let update_prob = file_config.training.update_prob.unwrap_or(update_prob);
    let seed = file_config.training.seed.unwrap_or(seed);
    let student_seed = file_config.training.student_seed.unwrap_or(student_seed);
    let seed_scale = file_config.training.seed_scale.or(seed_scale);
    let seed_mode = target2d_config_value_enum(
        "training.seed_mode",
        file_config.training.seed_mode,
        seed_mode,
    )?;
    let brush_size = file_config.training.brush_size.unwrap_or(brush_size);
    let per_parameter_grad_normalization = file_config
        .training
        .per_parameter_grad_normalization
        .unwrap_or(per_parameter_grad_normalization);
    let learning_rate = file_config.optimizer.learning_rate.unwrap_or(learning_rate);
    let weight_decay = file_config.optimizer.weight_decay.unwrap_or(weight_decay);
    let grad_clip_norm = file_config
        .optimizer
        .grad_clip_norm
        .unwrap_or(grad_clip_norm);
    let adam_beta1 = file_config.optimizer.adam_beta1.unwrap_or(adam_beta1);
    let adam_beta2 = file_config.optimizer.adam_beta2.unwrap_or(adam_beta2);
    let adam_epsilon = file_config.optimizer.adam_epsilon.unwrap_or(adam_epsilon);
    let scheduler_milestones = file_config
        .optimizer
        .scheduler_milestones
        .unwrap_or(scheduler_milestones);
    let scheduler_gamma = file_config
        .optimizer
        .scheduler_gamma
        .unwrap_or(scheduler_gamma);
    let target_points = file_config.target.points.unwrap_or(target_points);
    let target_image_size = file_config.target.image_size.or(target_image_size);
    let target_threshold = file_config.target.threshold.unwrap_or(target_threshold);
    let image_size = file_config.loss.image_size.unwrap_or(image_size);
    let splat_sigma = file_config.loss.splat_sigma.unwrap_or(splat_sigma);
    let center_loss = file_config.loss.center.unwrap_or(center_loss);
    let splat_loss_weight = file_config
        .loss
        .splat_loss_weight
        .unwrap_or(splat_loss_weight);
    let color_loss_weight = file_config
        .loss
        .color_loss_weight
        .unwrap_or(color_loss_weight);
    let density_loss_weight = file_config
        .loss
        .density_loss_weight
        .unwrap_or(density_loss_weight);
    let background_density_loss_weight = file_config
        .loss
        .background_density_loss_weight
        .unwrap_or(background_density_loss_weight);
    let foreground_density_loss_weight = file_config
        .loss
        .foreground_density_loss_weight
        .unwrap_or(foreground_density_loss_weight);
    let shape_chamfer_loss_weight = file_config.loss.shape_chamfer_loss_weight.unwrap_or(0.0);
    let displacement_regularizer_weight = file_config
        .loss
        .displacement_regularizer_weight
        .unwrap_or(displacement_regularizer_weight);
    let overflow_regularizer_weight = file_config
        .loss
        .overflow_regularizer_weight
        .unwrap_or(overflow_regularizer_weight);
    let bound_regularizer_weight = file_config
        .loss
        .bound_regularizer_weight
        .unwrap_or(bound_regularizer_weight);

    let preset: AutomataPreset = preset.into();
    if preset != AutomataPreset::Growing2d {
        return Err(std::io::Error::other(
            "train-target2d experimental diagnostics currently support only growing-2d",
        )
        .into());
    }
    let seed_scale = seed_scale.unwrap_or_else(|| NpaConfig::seed_scale_for_preset(preset));
    let seed_mode: ParticleSeed = seed_mode.into();
    if !shape_chamfer_loss_weight.is_finite() || shape_chamfer_loss_weight < 0.0 {
        return Err(std::io::Error::other(
            "loss.shape_chamfer_loss_weight must be finite and non-negative",
        )
        .into());
    }
    let mut loss_config = target2d_loss_config(
        image_size,
        splat_sigma,
        center_loss,
        splat_loss_weight,
        color_loss_weight,
        density_loss_weight,
        background_density_loss_weight,
        foreground_density_loss_weight,
        displacement_regularizer_weight,
        overflow_regularizer_weight,
        bound_regularizer_weight,
    )?;
    loss_config.shape_chamfer_loss_weight = shape_chamfer_loss_weight;
    let target = load_target_image_2d_adaptive(
        &target_image,
        target_threshold,
        target_points,
        target_image_size,
    )?;
    let hashgrid = upstream_growing_2d_hashgrid();
    let training_config = Target2dTrainingConfig {
        epochs,
        repetitions,
        report_interval,
        batch_size,
        pool_size,
        particle_count: particles,
        step_min,
        step_max,
        inject_seed_interval,
        update_prob,
        seed,
        seed_scale,
        seed_mode,
        brush_size,
        per_parameter_grad_normalization,
        optimizer: AdamWConfig {
            learning_rate,
            weight_decay,
            grad_clip_norm,
            beta1: adam_beta1,
            beta2: adam_beta2,
            epsilon: adam_epsilon,
        },
        scheduler_milestones: if scheduler_milestones.is_empty() {
            Target2dTrainingConfig::default().scheduler_milestones
        } else {
            scheduler_milestones
        },
        scheduler_gamma,
    };
    let actual_training_device = match training_device {
        TrainingDeviceArg::Auto => TrainingDeviceArg::Gpu,
        TrainingDeviceArg::Cpu => TrainingDeviceArg::Cpu,
        TrainingDeviceArg::Gpu => TrainingDeviceArg::Gpu,
    };
    let (training, gpu_training, model_output, model_eval_loss) = match actual_training_device {
        TrainingDeviceArg::Cpu => {
            let mut model = NpaModel::upstream_seeded(NpaConfig::growing_2d(), student_seed);
            let training = train_target_2d(
                &mut model,
                &hashgrid,
                &target,
                training_config.clone(),
                loss_config,
            )?;
            if let Some(path) = &model_output {
                let manifest = BpkModelManifest::from_model(
                    &model,
                    hashgrid.clone(),
                    Some(format!(
                        "experimental-rust:target2d-diagnostic:{}",
                        target_image.display()
                    )),
                );
                crate::import::save_manifest(path, &manifest)?;
            }
            (
                Some(training.clone()),
                None,
                model_output,
                Some(training.final_loss),
            )
        }
        TrainingDeviceArg::Gpu => {
            let mut model = NpaModel::upstream_seeded(NpaConfig::growing_2d(), student_seed);
            let burn_output = super::hyper_e2e::train_target_2d_burn_oracle(
                gpu_backend,
                &mut model,
                &hashgrid,
                &target_image,
                target.clone(),
                training_config.clone(),
                loss_config,
            )?;
            if let Some(path) = &model_output {
                let manifest = BpkModelManifest::from_model(
                    &model,
                    hashgrid.clone(),
                    Some(format!(
                        "experimental-burn-gpu:{gpu_backend:?}:target2d-diagnostic:{}",
                        target_image.display()
                    )),
                );
                crate::import::save_manifest(path, &manifest)?;
            }
            let model_eval_loss = evaluate_target2d_loaded_model_loss(
                &model,
                &hashgrid,
                &target,
                loss_config,
                particles,
                step_max,
                update_prob,
                seed,
                seed_scale,
                seed_mode,
            )?;
            (
                None,
                Some(CliHyper2dDirectBasisGpuTrainingReport {
                    backend: burn_output.backend,
                    device: burn_output.device,
                    metrics: {
                        let mut metrics = burn_output.metrics;
                        metrics["best_train_loss"] = serde_json::json!(burn_output.best_train_loss);
                        metrics["best_train_step"] = serde_json::json!(burn_output.best_train_step);
                        metrics["history"] = serde_json::json!(burn_output.history);
                        metrics
                    },
                }),
                model_output,
                Some(model_eval_loss),
            )
        }
        TrainingDeviceArg::Auto => unreachable!("target2d training device should be resolved"),
    };
    let reference_loss = reference_model
        .as_ref()
        .map(|path| {
            evaluate_target2d_model_loss(
                path,
                &target,
                loss_config,
                particles,
                step_max,
                update_prob,
                seed,
                seed_scale,
                seed_mode,
            )
        })
        .transpose()?;
    let (gap, ratio) = model_eval_loss
        .map(|loss| loss_gap_and_ratio(loss.total_loss, reference_loss))
        .unwrap_or((None, None));
    let report = CliTarget2dTrainingReport {
        preset,
        requested_training_device: training_device,
        training_device: actual_training_device,
        gpu_backend: (actual_training_device == TrainingDeviceArg::Gpu).then_some(gpu_backend),
        gpu_training,
        target_image: target_image.display().to_string(),
        target_source_width: target.source_width,
        target_source_height: target.source_height,
        target_points: target.point_count(),
        model_output: model_output.as_ref().map(|path| path.display().to_string()),
        model_eval_loss,
        reference_model: reference_model
            .as_ref()
            .map(|path| path.display().to_string()),
        reference_loss,
        final_loss_gap_to_reference: gap,
        final_loss_ratio_to_reference: ratio,
        hashgrid,
        training,
    };
    write_json_report(&output, &report)?;
    println!(
        "wrote {} device={:?} target_points={} final_loss={} reference_loss={} ratio={}",
        output.display(),
        report.training_device,
        report.target_points,
        report
            .model_eval_loss
            .map(|loss| format!("{:.6}", loss.total_loss))
            .unwrap_or_else(|| "none".to_string()),
        report
            .reference_loss
            .map(|loss| format!("{:.6}", loss.total_loss))
            .unwrap_or_else(|| "none".to_string()),
        report
            .final_loss_ratio_to_reference
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "none".to_string()),
    );
    Ok(())
}

fn load_target2d_train_config(
    path: Option<&Path>,
) -> Result<Target2dTrainExperimentConfig, Box<dyn std::error::Error>> {
    let Some(path) = path else {
        return Ok(Target2dTrainExperimentConfig::default());
    };
    let text = std::fs::read_to_string(path)?;
    toml::from_str(&text).map_err(|err| {
        std::io::Error::other(format!("failed to parse {}: {err}", path.display())).into()
    })
}

fn target2d_config_value_enum<T: ValueEnum>(
    name: &str,
    configured: Option<String>,
    fallback: T,
) -> Result<T, Box<dyn std::error::Error>> {
    let Some(value) = configured else {
        return Ok(fallback);
    };
    T::from_str(&value, true).map_err(|err| {
        std::io::Error::other(format!("invalid {name} value {value:?}: {err}")).into()
    })
}

fn validate_target2d_rollout_args(
    particles: usize,
    steps: usize,
    update_prob: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    if particles == 0 {
        return Err(std::io::Error::other("--particles must be greater than zero").into());
    }
    if steps == 0 {
        return Err(std::io::Error::other("--steps must be greater than zero").into());
    }
    if !(0.0..=1.0).contains(&update_prob) || !update_prob.is_finite() {
        return Err(std::io::Error::other("--update-prob must be finite and in [0, 1]").into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn target2d_loss_config(
    image_size: usize,
    splat_sigma: f32,
    center_loss: bool,
    splat_loss_weight: f32,
    color_loss_weight: f32,
    density_loss_weight: f32,
    background_density_loss_weight: f32,
    foreground_density_loss_weight: f32,
    displacement_regularizer_weight: f32,
    overflow_regularizer_weight: f32,
    bound_regularizer_weight: f32,
) -> Result<Target2dLossConfig, Box<dyn std::error::Error>> {
    if image_size == 0 {
        return Err(std::io::Error::other("--image-size must be greater than zero").into());
    }
    for (name, value) in [
        ("--splat-sigma", splat_sigma),
        ("--splat-loss-weight", splat_loss_weight),
        ("--color-loss-weight", color_loss_weight),
        ("--density-loss-weight", density_loss_weight),
        (
            "--background-density-loss-weight",
            background_density_loss_weight,
        ),
        (
            "--foreground-density-loss-weight",
            foreground_density_loss_weight,
        ),
        (
            "--displacement-regularizer-weight",
            displacement_regularizer_weight,
        ),
        ("--overflow-regularizer-weight", overflow_regularizer_weight),
        ("--bound-regularizer-weight", bound_regularizer_weight),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(
                std::io::Error::other(format!("{name} must be finite and non-negative")).into(),
            );
        }
    }
    Ok(Target2dLossConfig {
        image_size,
        sigma: splat_sigma,
        center: center_loss,
        splat_loss_weight,
        color_loss_weight,
        density_loss_weight,
        background_density_loss_weight,
        foreground_density_loss_weight,
        displacement_regularizer_weight,
        overflow_regularizer_weight,
        bound_regularizer_weight,
        ..Target2dLossConfig::default()
    })
}

#[allow(clippy::too_many_arguments)]
fn evaluate_target2d_model_loss(
    model_path: &Path,
    target: &TargetImage2d,
    loss_config: Target2dLossConfig,
    particles: usize,
    steps: usize,
    update_prob: f32,
    seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
) -> Result<Target2dLossReport, Box<dyn std::error::Error>> {
    let manifest = crate::import::load_manifest(model_path)?;
    if manifest.config.spatial_dims != 2 || manifest.hashgrid.dim != 2 {
        return Err(std::io::Error::other("eval-target2d requires a 2D model").into());
    }
    let hashgrid = manifest.hashgrid.clone();
    let model = manifest.into_model();
    evaluate_target2d_loaded_model_loss(
        &model,
        &hashgrid,
        target,
        loss_config,
        particles,
        steps,
        update_prob,
        seed,
        seed_scale,
        seed_mode,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_target2d_loaded_model_loss(
    model: &NpaModel,
    hashgrid: &burn_automata_kernels::HashGridConfig,
    target: &TargetImage2d,
    loss_config: Target2dLossConfig,
    particles: usize,
    steps: usize,
    update_prob: f32,
    seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
) -> Result<Target2dLossReport, Box<dyn std::error::Error>> {
    let trace = run_rollout(
        model,
        hashgrid,
        &RolloutConfig {
            particle_count: particles,
            steps,
            update_prob,
            seed,
            seed_scale,
            ..RolloutConfig::default()
        },
        seed_mode,
    )?;
    Ok(target_2d_loss_with_adjoint(
        &trace.positions,
        &trace.states,
        trace.batch_size,
        trace.particle_count,
        trace.state_dims,
        target,
        loss_config,
        trace.mean_dx.iter().copied().sum(),
        trace.steps,
    )?
    .report)
}

fn loss_gap_and_ratio(
    loss: f32,
    reference_loss: Option<Target2dLossReport>,
) -> (Option<f32>, Option<f32>) {
    reference_loss.map_or((None, None), |reference| {
        let gap = loss - reference.total_loss;
        let ratio = if reference.total_loss.abs() > f32::MIN_POSITIVE {
            Some(loss / reference.total_loss)
        } else {
            None
        };
        (Some(gap), ratio)
    })
}

#[allow(clippy::too_many_arguments)]
fn target2d_render_diagnostics(
    output_dir: Option<&Path>,
    wgpu_render_diagnostic: bool,
    model_path: &Path,
    reference_model_path: Option<&Path>,
    target: &TargetImage2d,
    loss_config: Target2dLossConfig,
    particles: usize,
    steps: usize,
    update_prob: f32,
    seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
) -> Result<Option<CliTarget2dRenderDiagnosticsReport>, Box<dyn std::error::Error>> {
    let Some(output_dir) = output_dir else {
        if wgpu_render_diagnostic {
            return Err(std::io::Error::other(
                "--wgpu-render-diagnostic requires --render-output-dir",
            )
            .into());
        }
        return Ok(None);
    };
    #[cfg(not(feature = "gpu_wgpu"))]
    if wgpu_render_diagnostic {
        return Err(std::io::Error::other(
            "--wgpu-render-diagnostic requires building burn_automata with feature gpu_wgpu",
        )
        .into());
    }

    std::fs::create_dir_all(output_dir)?;
    let target_render = render_target_2d_splat(target, loss_config)?;
    let density_lit_threshold = render_density_lit_threshold(&target_render);
    let target_particle_stats_report = target_particle_stats(target);
    let target_report = write_target2d_render_report(
        output_dir,
        "target",
        &target_render,
        None,
        loss_config.image_size,
        density_lit_threshold,
        None,
        Some(target_particle_stats_report.clone()),
    )?;

    let (model_cpu_render, model_cpu_stats) = render_target2d_model_path_cpu(
        model_path,
        target,
        loss_config,
        particles,
        steps,
        update_prob,
        seed,
        seed_scale,
        seed_mode,
    )?;
    let model_cpu_report = write_target2d_render_report(
        output_dir,
        "model_cpu",
        &model_cpu_render,
        Some(&target_render),
        loss_config.image_size,
        density_lit_threshold,
        Some(&target_particle_stats_report),
        Some(model_cpu_stats),
    )?;

    let reference_cpu_report = reference_model_path
        .map(|path| {
            let (render, stats) = render_target2d_model_path_cpu(
                path,
                target,
                loss_config,
                particles,
                steps,
                update_prob,
                seed,
                seed_scale,
                seed_mode,
            )?;
            write_target2d_render_report(
                output_dir,
                "reference_cpu",
                &render,
                Some(&target_render),
                loss_config.image_size,
                density_lit_threshold,
                Some(&target_particle_stats_report),
                Some(stats),
            )
        })
        .transpose()?;

    #[cfg(feature = "gpu_wgpu")]
    let model_wgpu_report = if wgpu_render_diagnostic {
        let (render, stats) = render_target2d_model_path_wgpu(
            model_path,
            target,
            loss_config,
            particles,
            steps,
            update_prob,
            seed,
            seed_scale,
            seed_mode,
        )?;
        Some(write_target2d_render_report(
            output_dir,
            "model_wgpu",
            &render,
            Some(&target_render),
            loss_config.image_size,
            density_lit_threshold,
            Some(&target_particle_stats_report),
            Some(stats),
        )?)
    } else {
        None
    };
    #[cfg(not(feature = "gpu_wgpu"))]
    let model_wgpu_report = None;

    #[cfg(feature = "gpu_wgpu")]
    let reference_wgpu_report = if wgpu_render_diagnostic {
        reference_model_path
            .map(|path| {
                let (render, stats) = render_target2d_model_path_wgpu(
                    path,
                    target,
                    loss_config,
                    particles,
                    steps,
                    update_prob,
                    seed,
                    seed_scale,
                    seed_mode,
                )?;
                write_target2d_render_report(
                    output_dir,
                    "reference_wgpu",
                    &render,
                    Some(&target_render),
                    loss_config.image_size,
                    density_lit_threshold,
                    Some(&target_particle_stats_report),
                    Some(stats),
                )
            })
            .transpose()?
    } else {
        None
    };
    #[cfg(not(feature = "gpu_wgpu"))]
    let reference_wgpu_report = None;

    Ok(Some(CliTarget2dRenderDiagnosticsReport {
        output_dir: output_dir.display().to_string(),
        image_size: loss_config.image_size,
        density_lit_threshold,
        target: target_report,
        model_cpu: model_cpu_report,
        reference_cpu: reference_cpu_report,
        model_wgpu: model_wgpu_report,
        reference_wgpu: reference_wgpu_report,
    }))
}

#[allow(clippy::too_many_arguments)]
fn render_target2d_model_path_cpu(
    path: &Path,
    target: &TargetImage2d,
    loss_config: Target2dLossConfig,
    particles: usize,
    steps: usize,
    update_prob: f32,
    seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
) -> Result<(Target2dRenderedSplat, CliTarget2dParticleStatsReport), Box<dyn std::error::Error>> {
    let manifest = crate::import::load_manifest(path)?;
    if manifest.config.spatial_dims != 2 || manifest.hashgrid.dim != 2 {
        return Err(std::io::Error::other("target2d render diagnostics require a 2D model").into());
    }
    let hashgrid = manifest.hashgrid.clone();
    let model = manifest.into_model();
    let trace = run_rollout(
        &model,
        &hashgrid,
        &RolloutConfig {
            particle_count: particles,
            steps,
            update_prob,
            seed,
            seed_scale,
            ..RolloutConfig::default()
        },
        seed_mode,
    )?;
    render_target2d_trace(&trace, target, loss_config)
}

#[cfg(feature = "gpu_wgpu")]
#[allow(clippy::too_many_arguments)]
fn render_target2d_model_path_wgpu(
    path: &Path,
    target: &TargetImage2d,
    loss_config: Target2dLossConfig,
    particles: usize,
    steps: usize,
    update_prob: f32,
    seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
) -> Result<(Target2dRenderedSplat, CliTarget2dParticleStatsReport), Box<dyn std::error::Error>> {
    let manifest = crate::import::load_manifest(path)?;
    if manifest.config.spatial_dims != 2 || manifest.hashgrid.dim != 2 {
        return Err(
            std::io::Error::other("target2d WGPU render diagnostics require a 2D model").into(),
        );
    }
    let hashgrid = manifest.hashgrid.clone();
    let model = manifest.into_model();
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        model.config.state_dims,
        model.config.spatial_dims,
        seed,
        seed_mode,
        seed_scale,
    );
    let executor = pollster::block_on(crate::gpu::WgpuAutomataExecutor::new())?;
    let mut state = executor.create_state_with_neighbor_mode_and_update_prob(
        &model,
        &positions,
        &states,
        1,
        particles,
        &hashgrid,
        RolloutConfig::default().dt,
        crate::gpu::WgpuNeighborMode::Auto,
        update_prob,
        seed,
    )?;
    for _ in 0..steps {
        executor.step_state(&mut state)?;
    }
    let readback = executor.read_state(&state)?;
    let trace = crate::RolloutTrace {
        positions: readback.next_positions,
        states: readback.next_states,
        batch_size: 1,
        particle_count: particles,
        state_dims: model.config.state_dims,
        steps,
        mean_dx: Vec::new(),
    };
    render_target2d_trace(&trace, target, loss_config)
}

fn render_target2d_trace(
    trace: &crate::RolloutTrace,
    target: &TargetImage2d,
    loss_config: Target2dLossConfig,
) -> Result<(Target2dRenderedSplat, CliTarget2dParticleStatsReport), Box<dyn std::error::Error>> {
    let output_scale = target.point_count() as f32 / trace.particle_count.max(1) as f32;
    let render = render_rollout_2d_splat(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        target.pixel_size,
        loss_config,
        None,
        output_scale,
    )?;
    Ok((render, rollout_particle_stats(&trace.positions)))
}

#[allow(clippy::too_many_arguments)]
fn write_target2d_render_report(
    output_dir: &Path,
    label: &'static str,
    render: &Target2dRenderedSplat,
    target_render: Option<&Target2dRenderedSplat>,
    image_size: usize,
    density_lit_threshold: f32,
    target_particle_stats: Option<&CliTarget2dParticleStatsReport>,
    particle_stats: Option<CliTarget2dParticleStatsReport>,
) -> Result<CliTarget2dRenderImageReport, Box<dyn std::error::Error>> {
    let rgb_png = output_dir.join(format!("{label}_rgb.png"));
    let density_png = output_dir.join(format!("{label}_density.png"));
    write_rgb_splat_png(&rgb_png, render, image_size)?;
    write_density_splat_png(&density_png, render, image_size)?;
    let (rgb_mse_to_target, rgb_psnr_db_to_target) = target_render
        .map(|target| image_metric(&render.rgb, &target.rgb))
        .transpose()?
        .map_or((None, None), |(mse, psnr)| (Some(mse), Some(psnr)));
    let (density_mse_to_target, density_psnr_db_to_target) = target_render
        .map(|target| image_metric(&render.density, &target.density))
        .transpose()?
        .map_or((None, None), |(mse, psnr)| (Some(mse), Some(psnr)));
    let density_total = render.density.iter().copied().sum::<f32>();
    let density_max = render
        .density
        .iter()
        .copied()
        .fold(0.0_f32, |max_value, value| max_value.max(value));
    let (lit_pixels, lit_bbox_xyxy) =
        density_lit_stats(&render.density, image_size, density_lit_threshold)?;
    let geometry_to_target = target_render
        .map(|target| {
            let (target_lit_pixels, target_bbox) =
                density_lit_stats(&target.density, image_size, density_lit_threshold)?;
            let overlap =
                density_overlap_stats(&render.density, &target.density, density_lit_threshold)?;
            Ok::<_, Box<dyn std::error::Error>>(CliTarget2dRenderGeometryReport {
                lit_pixel_ratio: lit_pixels as f32 / target_lit_pixels.max(1) as f32,
                foreground_iou: overlap.iou,
                target_recall: overlap.target_recall,
                generated_precision: overlap.generated_precision,
                bbox_iou: bbox_iou(lit_bbox_xyxy, target_bbox),
                bbox_width_ratio: bbox_width_ratio(lit_bbox_xyxy, target_bbox),
                bbox_height_ratio: bbox_height_ratio(lit_bbox_xyxy, target_bbox),
                bbox_area_ratio: bbox_area_ratio(lit_bbox_xyxy, target_bbox),
                particle_rms_radius_ratio: particle_stats.as_ref().zip(target_particle_stats).map(
                    |(model, target)| model.rms_radius / target.rms_radius.max(f32::MIN_POSITIVE),
                ),
            })
        })
        .transpose()?;
    Ok(CliTarget2dRenderImageReport {
        label,
        rgb_png: rgb_png.display().to_string(),
        density_png: density_png.display().to_string(),
        rgb_mse_to_target,
        rgb_psnr_db_to_target,
        density_mse_to_target,
        density_psnr_db_to_target,
        density_total,
        density_max,
        lit_pixels,
        lit_bbox_xyxy,
        geometry_to_target,
        particle_stats,
    })
}

fn write_rgb_splat_png(
    path: &Path,
    render: &Target2dRenderedSplat,
    image_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if render.rgb.len() != image_size * image_size * 3 {
        return Err(std::io::Error::other("target2d RGB render buffer has the wrong size").into());
    }
    let mut image = image::RgbaImage::new(image_size as u32, image_size as u32);
    for (pixel_index, pixel) in image.pixels_mut().enumerate() {
        let base = pixel_index * 3;
        *pixel = image::Rgba([
            unit_to_u8(render.rgb[base]),
            unit_to_u8(render.rgb[base + 1]),
            unit_to_u8(render.rgb[base + 2]),
            255,
        ]);
    }
    image.save(path)?;
    Ok(())
}

fn write_density_splat_png(
    path: &Path,
    render: &Target2dRenderedSplat,
    image_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if render.density.len() != image_size * image_size {
        return Err(
            std::io::Error::other("target2d density render buffer has the wrong size").into(),
        );
    }
    let max_density = render
        .density
        .iter()
        .copied()
        .fold(0.0_f32, |max_value, value| max_value.max(value))
        .max(f32::MIN_POSITIVE);
    let mut image = image::RgbaImage::new(image_size as u32, image_size as u32);
    for (pixel_index, pixel) in image.pixels_mut().enumerate() {
        let value = unit_to_u8(render.density[pixel_index] / max_density);
        *pixel = image::Rgba([value, value, value, 255]);
    }
    image.save(path)?;
    Ok(())
}

fn image_metric(
    generated: &[f32],
    target: &[f32],
) -> Result<(f32, f32), Box<dyn std::error::Error>> {
    if generated.len() != target.len() {
        return Err(std::io::Error::other("target2d render metric buffer sizes differ").into());
    }
    if generated.is_empty() {
        return Err(
            std::io::Error::other("target2d render metric buffers must not be empty").into(),
        );
    }
    let mse = generated
        .iter()
        .zip(target)
        .map(|(&generated_value, &target_value)| {
            let diff = generated_value - target_value;
            diff * diff
        })
        .sum::<f32>()
        / generated.len() as f32;
    Ok((mse, psnr_db(mse)))
}

fn psnr_db(mse: f32) -> f32 {
    if mse <= f32::EPSILON {
        99.0
    } else {
        10.0 * (1.0 / mse.max(1.0e-12)).log10()
    }
}

fn unit_to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn render_density_lit_threshold(render: &Target2dRenderedSplat) -> f32 {
    let max_density = render
        .density
        .iter()
        .copied()
        .fold(0.0_f32, |max_value, value| max_value.max(value));
    (max_density * 0.05).max(1.0e-6)
}

type DensityLitStats = (usize, Option<[usize; 4]>);

fn density_lit_stats(
    density: &[f32],
    image_size: usize,
    threshold: f32,
) -> Result<DensityLitStats, Box<dyn std::error::Error>> {
    if density.len() != image_size * image_size {
        return Err(
            std::io::Error::other("target2d density stats buffer has the wrong size").into(),
        );
    }
    let mut lit_pixels = 0usize;
    let mut min_x = image_size;
    let mut min_y = image_size;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    for y in 0..image_size {
        for x in 0..image_size {
            let value = density[y * image_size + x];
            if value < threshold {
                continue;
            }
            lit_pixels += 1;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    let bbox = (lit_pixels > 0).then_some([min_x, min_y, max_x, max_y]);
    Ok((lit_pixels, bbox))
}

#[derive(Clone, Copy)]
struct DensityOverlapStats {
    iou: f32,
    target_recall: f32,
    generated_precision: f32,
}

fn density_overlap_stats(
    generated: &[f32],
    target: &[f32],
    threshold: f32,
) -> Result<DensityOverlapStats, Box<dyn std::error::Error>> {
    if generated.len() != target.len() {
        return Err(
            std::io::Error::other("target2d density overlap buffers differ in size").into(),
        );
    }
    let mut generated_count = 0usize;
    let mut target_count = 0usize;
    let mut intersection = 0usize;
    let mut union = 0usize;
    for (&generated_density, &target_density) in generated.iter().zip(target) {
        let generated_hit = generated_density >= threshold;
        let target_hit = target_density >= threshold;
        generated_count += usize::from(generated_hit);
        target_count += usize::from(target_hit);
        intersection += usize::from(generated_hit && target_hit);
        union += usize::from(generated_hit || target_hit);
    }
    Ok(DensityOverlapStats {
        iou: intersection as f32 / union.max(1) as f32,
        target_recall: intersection as f32 / target_count.max(1) as f32,
        generated_precision: intersection as f32 / generated_count.max(1) as f32,
    })
}

fn bbox_iou(left: Option<[usize; 4]>, right: Option<[usize; 4]>) -> Option<f32> {
    let left = left?;
    let right = right?;
    let x0 = left[0].max(right[0]);
    let y0 = left[1].max(right[1]);
    let x1 = left[2].min(right[2]);
    let y1 = left[3].min(right[3]);
    let intersection = if x1 >= x0 && y1 >= y0 {
        bbox_area([x0, y0, x1, y1])
    } else {
        0.0
    };
    let union = bbox_area(left) + bbox_area(right) - intersection;
    Some(intersection / union.max(f32::MIN_POSITIVE))
}

fn bbox_width_ratio(left: Option<[usize; 4]>, right: Option<[usize; 4]>) -> Option<f32> {
    Some(bbox_width(left?) / bbox_width(right?).max(f32::MIN_POSITIVE))
}

fn bbox_height_ratio(left: Option<[usize; 4]>, right: Option<[usize; 4]>) -> Option<f32> {
    Some(bbox_height(left?) / bbox_height(right?).max(f32::MIN_POSITIVE))
}

fn bbox_area_ratio(left: Option<[usize; 4]>, right: Option<[usize; 4]>) -> Option<f32> {
    Some(bbox_area(left?) / bbox_area(right?).max(f32::MIN_POSITIVE))
}

fn bbox_width(bbox: [usize; 4]) -> f32 {
    bbox[2].saturating_sub(bbox[0]).saturating_add(1) as f32
}

fn bbox_height(bbox: [usize; 4]) -> f32 {
    bbox[3].saturating_sub(bbox[1]).saturating_add(1) as f32
}

fn bbox_area(bbox: [usize; 4]) -> f32 {
    bbox_width(bbox) * bbox_height(bbox)
}

fn target_particle_stats(target: &TargetImage2d) -> CliTarget2dParticleStatsReport {
    particle_stats_2d(target.positions.iter().copied(), target.positions.len())
}

fn rollout_particle_stats(positions: &[[f32; 4]]) -> CliTarget2dParticleStatsReport {
    particle_stats_2d(
        positions.iter().map(|position| [position[0], position[1]]),
        positions.len(),
    )
}

fn particle_stats_2d(
    positions: impl Iterator<Item = [f32; 2]>,
    count: usize,
) -> CliTarget2dParticleStatsReport {
    let mut mean = [0.0_f32; 2];
    let mut bounds_min = [f32::INFINITY; 2];
    let mut bounds_max = [f32::NEG_INFINITY; 2];
    let mut copied = Vec::with_capacity(count);
    for position in positions {
        mean[0] += position[0];
        mean[1] += position[1];
        bounds_min[0] = bounds_min[0].min(position[0]);
        bounds_min[1] = bounds_min[1].min(position[1]);
        bounds_max[0] = bounds_max[0].max(position[0]);
        bounds_max[1] = bounds_max[1].max(position[1]);
        copied.push(position);
    }
    let denom = copied.len().max(1) as f32;
    mean[0] /= denom;
    mean[1] /= denom;
    let mut radius_sq_sum = 0.0_f32;
    let mut out_of_domain = 0usize;
    for position in &copied {
        let dx = position[0] - mean[0];
        let dy = position[1] - mean[1];
        radius_sq_sum += dx * dx + dy * dy;
        if position[0] < -1.0 || position[0] > 1.0 || position[1] < -1.0 || position[1] > 1.0 {
            out_of_domain += 1;
        }
    }
    if copied.is_empty() {
        bounds_min = [0.0; 2];
        bounds_max = [0.0; 2];
    }
    CliTarget2dParticleStatsReport {
        count: copied.len(),
        mean_xy: mean,
        rms_radius: (radius_sq_sum / denom).sqrt(),
        bounds_min_xy: bounds_min,
        bounds_max_xy: bounds_max,
        out_of_domain_fraction: out_of_domain as f32 / denom,
    }
}

pub(super) fn load_target_image_2d_adaptive(
    path: &Path,
    threshold: f32,
    target_points: usize,
    image_size: Option<usize>,
) -> Result<TargetImage2d, Box<dyn std::error::Error>> {
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(
            std::io::Error::other("--target-threshold must be finite and non-negative").into(),
        );
    }
    let max_size = if let Some(size) = image_size {
        if size == 0 {
            return Err(
                std::io::Error::other("--target-image-size must be greater than zero").into(),
            );
        }
        size
    } else {
        adaptive_target_image_size(path, threshold, target_points)?
    };
    let rgba = load_rgba_thumbnail(path, max_size)?;
    target_from_rgba(&rgba, threshold)
}

fn adaptive_target_image_size(
    path: &Path,
    threshold: f32,
    target_points: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    if target_points == 0 {
        return Err(std::io::Error::other("--target-points must be greater than zero").into());
    }
    let mut size = 128usize;
    for _ in 0..5 {
        let image = load_rgba_thumbnail(path, size)?;
        let count = foreground_alpha_count(&image, threshold).max(1);
        size = ((target_points as f32 / count as f32).sqrt() * size as f32)
            .round()
            .clamp(1.0, 2048.0) as usize;
    }
    for _ in 0..8 {
        let image = load_rgba_thumbnail(path, size)?;
        let count = foreground_alpha_count(&image, threshold);
        if count >= target_points || size >= 2048 {
            break;
        }
        let next_size = ((target_points as f32 / count.max(1) as f32).sqrt() * size as f32 * 1.02)
            .ceil()
            .clamp((size + 1) as f32, 2048.0) as usize;
        if next_size == size {
            break;
        }
        size = next_size;
    }
    Ok(size)
}

fn load_rgba_thumbnail(
    path: &Path,
    max_size: usize,
) -> Result<image::RgbaImage, Box<dyn std::error::Error>> {
    let image = image::ImageReader::open(path)?.decode()?;
    Ok(image.thumbnail(max_size as u32, max_size as u32).to_rgba8())
}

fn foreground_alpha_count(image: &image::RgbaImage, threshold: f32) -> usize {
    image
        .pixels()
        .filter(|pixel| {
            crate::target2d::target_2d_foreground_rgba_pixel(
                pixel[0] as f32 / 255.0,
                pixel[1] as f32 / 255.0,
                pixel[2] as f32 / 255.0,
                pixel[3] as f32 / 255.0,
                threshold,
            )
        })
        .count()
}

fn target_from_rgba(
    image: &image::RgbaImage,
    threshold: f32,
) -> Result<TargetImage2d, Box<dyn std::error::Error>> {
    let values = image
        .as_raw()
        .iter()
        .map(|value| *value as f32 / 255.0)
        .collect::<Vec<_>>();
    Ok(TargetImage2d::from_rgba_pixels(
        image.width() as usize,
        image.height() as usize,
        &values,
        TargetImage2dExtractConfig {
            threshold,
            ..TargetImage2dExtractConfig::default()
        },
    )?)
}

fn write_json_report<T: Serialize>(
    path: &Path,
    report: &T,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(report)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use image::{Rgba, RgbaImage};

    use super::*;

    #[test]
    fn adaptive_target_image_size_does_not_undershoot_requested_points() {
        let path = std::env::temp_dir().join(format!(
            "burn_automata_target2d_full_alpha_{}.png",
            std::process::id()
        ));
        let image = RgbaImage::from_pixel(128, 128, Rgba([255, 255, 255, 255]));
        image.save(&path).unwrap();

        let size = adaptive_target_image_size(&path, 0.05, 2048).unwrap();
        let resized = load_rgba_thumbnail(&path, size).unwrap();
        let count = foreground_alpha_count(&resized, 0.05);
        std::fs::remove_file(&path).ok();

        assert!(
            count >= 2048,
            "adaptive target count {count} should meet the requested floor at size {size}"
        );
    }

    #[test]
    fn target2d_train_config_parses_explicit_experimental_gate() {
        let config: Target2dTrainExperimentConfig = toml::from_str(
            r#"
            [source]
            target_image = "assets/catalog_thumbnails/lizard.png"

            [training]
            experimental = true
            device = "gpu"

            [loss]
            background_density_loss_weight = 0.0
            "#,
        )
        .unwrap();

        assert_eq!(
            config.source.target_image.as_deref(),
            Some(Path::new("assets/catalog_thumbnails/lizard.png"))
        );
        assert_eq!(config.training.experimental, Some(true));
        assert_eq!(config.training.device.as_deref(), Some("gpu"));
        assert_eq!(config.loss.background_density_loss_weight, Some(0.0));
    }
}
