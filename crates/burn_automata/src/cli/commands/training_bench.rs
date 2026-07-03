use crate::cli::prelude::*;

use super::basic::{resolve_training_device, run_training_on_device, train_batch_for_round};

pub(crate) fn run_bench_training(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::BenchTraining {
        preset,
        target_model,
        rows,
        steps,
        repeats,
        warmup_steps,
        report_interval,
        learning_rate,
        grad_clip_norm,
        weight_decay,
        optimizer,
        training_device,
        adam_beta1,
        adam_beta2,
        adam_epsilon,
        student_seed,
        batch_source,
        rollout_particles,
        rollout_steps,
        rollouts,
        temporal_samples,
        rollout_update_prob,
        seed_scale,
        seed_mode,
        output,
    } = command
    else {
        unreachable!("run_bench_training called with the wrong command variant");
    };

    validate_training_bench_shape(rows, steps, repeats)?;

    let requested_training_device = training_device;
    let actual_training_device = resolve_training_device(training_device)?;
    let preset: AutomataPreset = preset.into();
    let (preset_config, preset_grid) = NpaConfig::for_preset(preset);
    let target_model_report = target_model.as_ref().map(|path| path.display().to_string());
    let (config, hashgrid, teacher) = if let Some(path) = target_model.as_ref() {
        let manifest = crate::import::load_manifest(path)?;
        (
            manifest.config.clone(),
            manifest.hashgrid.clone(),
            Some(manifest.into_model()),
        )
    } else {
        (preset_config, preset_grid, None)
    };
    let base_model = NpaModel::seeded(config.clone(), student_seed);
    let target = if let Some(teacher) = teacher.as_ref() {
        SupervisedTarget::Teacher(teacher)
    } else {
        SupervisedTarget::ZeroUpdate
    };
    let seed_scale = seed_scale.unwrap_or_else(|| NpaConfig::seed_scale_for_preset(preset));
    let seed_mode: ParticleSeed = seed_mode.into();
    let batch = train_batch_for_round(
        &base_model,
        teacher.as_ref(),
        &hashgrid,
        target,
        batch_source,
        rows,
        student_seed,
        rollout_particles,
        rollout_steps,
        rollouts,
        temporal_samples,
        rollout_update_prob,
        seed_scale,
        seed_mode,
    )?;
    let sgd = SgdConfig {
        learning_rate,
        weight_decay,
        grad_clip_norm,
    };
    let optimizer_config = match optimizer {
        TrainingOptimizerArg::Sgd => SupervisedOptimizerConfig::Sgd(sgd),
        TrainingOptimizerArg::AdamW => SupervisedOptimizerConfig::AdamW(AdamWConfig {
            learning_rate,
            weight_decay,
            grad_clip_norm,
            beta1: adam_beta1,
            beta2: adam_beta2,
            epsilon: adam_epsilon,
        }),
    };
    let report_interval = if report_interval == 0 {
        steps
    } else {
        report_interval
    };
    if warmup_steps > 0 {
        let mut warmup_model = base_model.clone();
        run_training_on_device(
            actual_training_device,
            &mut warmup_model,
            &batch,
            TrainingRunConfig {
                steps: warmup_steps,
                report_interval: warmup_steps,
                sgd,
            },
            optimizer_config,
        )?;
    }

    let mut runs = Vec::with_capacity(repeats);
    for repeat in 0..repeats {
        let mut model = base_model.clone();
        let started = Instant::now();
        let report = run_training_on_device(
            actual_training_device,
            &mut model,
            &batch,
            TrainingRunConfig {
                steps,
                report_interval,
                sgd,
            },
            optimizer_config,
        )?;
        let elapsed = started.elapsed().as_secs_f64();
        runs.push(CliTrainingBenchRunReport {
            repeat,
            elapsed_ms: elapsed * 1000.0,
            row_steps_per_sec: report.rows as f64 * report.steps as f64 / elapsed.max(f64::EPSILON),
            initial_loss: report.initial_loss,
            final_loss: report.final_loss,
            best_loss: report.best_loss,
            history_points: report.history.len(),
        });
    }
    let (min_row_steps_per_sec, median_row_steps_per_sec, max_row_steps_per_sec) =
        training_bench_speed_summary(&runs);
    let report = CliTrainingBenchReport {
        preset,
        requested_training_device,
        training_device: actual_training_device,
        batch_source,
        optimizer: optimizer_config,
        rows: batch.features.len() / config.perception_dims(),
        steps,
        repeats,
        warmup_steps,
        report_interval,
        target_model: target_model_report,
        runs,
        min_row_steps_per_sec,
        median_row_steps_per_sec,
        max_row_steps_per_sec,
    };
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&output, serde_json::to_string_pretty(&report)?)?;
    println!(
        "backend={:?} requested={:?} rows={} steps={steps} repeats={repeats} median_row_steps_per_sec={:.3} min_row_steps_per_sec={:.3} max_row_steps_per_sec={:.3} wrote {}",
        report.training_device,
        report.requested_training_device,
        report.rows,
        report.median_row_steps_per_sec,
        report.min_row_steps_per_sec,
        report.max_row_steps_per_sec,
        output.display(),
    );

    Ok(())
}

fn validate_training_bench_shape(
    rows: usize,
    steps: usize,
    repeats: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if rows == 0 {
        return Err(std::io::Error::other("--rows must be greater than zero").into());
    }
    if steps == 0 {
        return Err(std::io::Error::other("--steps must be greater than zero").into());
    }
    if repeats == 0 {
        return Err(std::io::Error::other("--repeats must be greater than zero").into());
    }
    Ok(())
}

fn training_bench_speed_summary(runs: &[CliTrainingBenchRunReport]) -> (f64, f64, f64) {
    let mut speeds = runs
        .iter()
        .map(|run| run.row_steps_per_sec)
        .collect::<Vec<_>>();
    speeds.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let min = speeds.first().copied().unwrap_or_default();
    let median = speeds.get(speeds.len() / 2).copied().unwrap_or_default();
    let max = speeds.last().copied().unwrap_or_default();
    (min, median, max)
}
