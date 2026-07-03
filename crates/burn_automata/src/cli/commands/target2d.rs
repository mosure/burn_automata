use crate::cli::prelude::*;

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
        splat_loss_weight,
        color_loss_weight,
        density_loss_weight,
        displacement_regularizer_weight,
        overflow_regularizer_weight,
        bound_regularizer_weight,
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
        splat_loss_weight,
        color_loss_weight,
        density_loss_weight,
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
        preset,
        target_image,
        output,
        training_device,
        python,
        gpu_upstream_root,
        gpu_device,
        gpu_checkpoint_output,
        gpu_metrics_output,
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
        splat_loss_weight,
        color_loss_weight,
        density_loss_weight,
        displacement_regularizer_weight,
        overflow_regularizer_weight,
        bound_regularizer_weight,
    } = command
    else {
        unreachable!("run_train_target_2d called with the wrong command variant");
    };

    let preset: AutomataPreset = preset.into();
    if preset != AutomataPreset::Growing2d {
        return Err(std::io::Error::other(
            "train-target2d currently implements the upstream growing-2d splat objective",
        )
        .into());
    }
    let seed_scale = seed_scale.unwrap_or_else(|| NpaConfig::seed_scale_for_preset(preset));
    let seed_mode: ParticleSeed = seed_mode.into();
    let loss_config = target2d_loss_config(
        image_size,
        splat_sigma,
        splat_loss_weight,
        color_loss_weight,
        density_loss_weight,
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
    let actual_training_device = resolve_target2d_training_device(training_device);
    let (training, gpu_training, model_output, model_eval_loss) = match actual_training_device {
        TrainingDeviceArg::Auto => unreachable!("target2d training device should be resolved"),
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
                        "trained-rust:target2d-upstream-splat:{}",
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
            let gpu = run_target2d_upstream_gpu_training(Target2dGpuTrainingRequest {
                python,
                upstream_root: gpu_upstream_root,
                device: gpu_device,
                target_image: target_image.clone(),
                output_report: output.clone(),
                model_output,
                checkpoint_output: gpu_checkpoint_output,
                metrics_output: gpu_metrics_output,
                training_config: &training_config,
                loss_config,
                target_threshold,
                target_image_size,
                target_points,
            })?;
            let model_eval_loss = evaluate_target2d_model_loss(
                &gpu.model_output,
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
                Some(gpu.report),
                Some(gpu.model_output),
                Some(model_eval_loss),
            )
        }
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
        target_image: target_image.display().to_string(),
        target_source_width: target.source_width,
        target_source_height: target.source_height,
        target_points: target.point_count(),
        model_output: model_output.as_ref().map(|path| path.display().to_string()),
        model_eval_loss,
        gpu_training,
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

struct Target2dGpuTrainingRequest<'a> {
    python: PathBuf,
    upstream_root: Option<PathBuf>,
    device: String,
    target_image: PathBuf,
    output_report: PathBuf,
    model_output: Option<PathBuf>,
    checkpoint_output: Option<PathBuf>,
    metrics_output: Option<PathBuf>,
    training_config: &'a Target2dTrainingConfig,
    loss_config: Target2dLossConfig,
    target_threshold: f32,
    target_image_size: Option<usize>,
    target_points: usize,
}

struct Target2dGpuTrainingRun {
    model_output: PathBuf,
    report: CliTarget2dGpuTrainingReport,
}

fn resolve_target2d_training_device(requested: TrainingDeviceArg) -> TrainingDeviceArg {
    match requested {
        TrainingDeviceArg::Auto | TrainingDeviceArg::Gpu => TrainingDeviceArg::Gpu,
        TrainingDeviceArg::Cpu => TrainingDeviceArg::Cpu,
    }
}

fn run_target2d_upstream_gpu_training(
    request: Target2dGpuTrainingRequest<'_>,
) -> Result<Target2dGpuTrainingRun, Box<dyn std::error::Error>> {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/train_target2d_upstream_gpu.py");
    let model_output = request
        .model_output
        .unwrap_or_else(|| path_with_extra_extension(&request.output_report, "bpk"));
    let checkpoint_output = request
        .checkpoint_output
        .unwrap_or_else(|| path_with_extra_extension(&request.output_report, "pth"));
    let metrics_output = request
        .metrics_output
        .unwrap_or_else(|| path_with_extra_extension(&request.output_report, "gpu.json"));
    let mut command = std::process::Command::new(&request.python);
    command
        .arg(&script)
        .arg("--target-image")
        .arg(&request.target_image)
        .arg("--checkpoint-output")
        .arg(&checkpoint_output)
        .arg("--metrics-output")
        .arg(&metrics_output)
        .arg("--device")
        .arg(&request.device)
        .arg("--epochs")
        .arg(request.training_config.epochs.to_string())
        .arg("--repetitions")
        .arg(request.training_config.repetitions.to_string())
        .arg("--report-interval")
        .arg(request.training_config.report_interval.to_string())
        .arg("--batch-size")
        .arg(request.training_config.batch_size.to_string())
        .arg("--pool-size")
        .arg(request.training_config.pool_size.to_string())
        .arg("--particles")
        .arg(request.training_config.particle_count.to_string())
        .arg("--step-min")
        .arg(request.training_config.step_min.to_string())
        .arg("--step-max")
        .arg(request.training_config.step_max.to_string())
        .arg("--inject-seed-interval")
        .arg(request.training_config.inject_seed_interval.to_string())
        .arg("--update-prob")
        .arg(request.training_config.update_prob.to_string())
        .arg("--seed")
        .arg(request.training_config.seed.to_string())
        .arg("--seed-scale")
        .arg(request.training_config.seed_scale.to_string())
        .arg("--seed-mode")
        .arg(upstream_seed_mode(request.training_config.seed_mode)?)
        .arg("--brush-size")
        .arg(request.training_config.brush_size.to_string())
        .arg("--learning-rate")
        .arg(request.training_config.optimizer.learning_rate.to_string())
        .arg("--weight-decay")
        .arg(request.training_config.optimizer.weight_decay.to_string())
        .arg("--adam-beta1")
        .arg(request.training_config.optimizer.beta1.to_string())
        .arg("--adam-beta2")
        .arg(request.training_config.optimizer.beta2.to_string())
        .arg("--adam-epsilon")
        .arg(request.training_config.optimizer.epsilon.to_string())
        .arg("--scheduler-milestones")
        .arg(join_usize_csv(
            &request.training_config.scheduler_milestones,
        ))
        .arg("--scheduler-gamma")
        .arg(request.training_config.scheduler_gamma.to_string())
        .arg("--target-points")
        .arg(request.target_points.to_string())
        .arg("--target-threshold")
        .arg(request.target_threshold.to_string())
        .arg("--image-size")
        .arg(request.loss_config.image_size.to_string())
        .arg("--splat-sigma")
        .arg(request.loss_config.sigma.to_string())
        .arg("--splat-loss-weight")
        .arg(request.loss_config.splat_loss_weight.to_string())
        .arg("--color-loss-weight")
        .arg(request.loss_config.color_loss_weight.to_string())
        .arg("--density-loss-weight")
        .arg(request.loss_config.density_loss_weight.to_string())
        .arg("--displacement-regularizer-weight")
        .arg(
            request
                .loss_config
                .displacement_regularizer_weight
                .to_string(),
        )
        .arg("--overflow-regularizer-weight")
        .arg(request.loss_config.overflow_regularizer_weight.to_string())
        .arg("--bound-regularizer-weight")
        .arg(request.loss_config.bound_regularizer_weight.to_string());
    if let Some(upstream_root) = &request.upstream_root {
        command.arg("--upstream-root").arg(upstream_root);
    }
    if let Some(target_image_size) = request.target_image_size {
        command
            .arg("--target-image-size")
            .arg(target_image_size.to_string());
    }
    if request.training_config.per_parameter_grad_normalization {
        command.arg("--normalize-grads");
    } else {
        command.arg("--no-normalize-grads");
    }

    let output = command.output()?;
    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "GPU target2d training failed with status {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ))
        .into());
    }
    let metrics_text = std::fs::read_to_string(&metrics_output)?;
    let metrics: serde_json::Value = serde_json::from_str(&metrics_text)?;
    let import = import_model(&checkpoint_output, &model_output)?;
    Ok(Target2dGpuTrainingRun {
        model_output,
        report: CliTarget2dGpuTrainingReport {
            backend: "upstream_torch_cuda_sphops",
            python: request.python.display().to_string(),
            device: request.device,
            upstream_root: request
                .upstream_root
                .as_ref()
                .map(|path| path.display().to_string()),
            checkpoint_output: checkpoint_output.display().to_string(),
            metrics_output: metrics_output.display().to_string(),
            import,
            metrics,
        },
    })
}

fn upstream_seed_mode(seed_mode: ParticleSeed) -> Result<&'static str, Box<dyn std::error::Error>> {
    match seed_mode {
        ParticleSeed::Gaussian => Ok("gaussian"),
        ParticleSeed::Uniform => Ok("uniform"),
        ParticleSeed::UniformCircle => Ok("uniform_circle"),
        other => Err(std::io::Error::other(format!(
            "upstream target2d GPU training supports only 2D seed modes; got {other:?}",
        ))
        .into()),
    }
}

fn join_usize_csv(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn path_with_extra_extension(path: &Path, extension: &str) -> PathBuf {
    let mut output = path.to_path_buf();
    output.set_extension(extension);
    output
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
    splat_loss_weight: f32,
    color_loss_weight: f32,
    density_loss_weight: f32,
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
        splat_loss_weight,
        color_loss_weight,
        density_loss_weight,
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
        .filter(|pixel| pixel[3] as f32 / 255.0 >= threshold)
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
