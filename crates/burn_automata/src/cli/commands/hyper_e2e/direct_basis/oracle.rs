use crate::cli::prelude::*;

use super::{
    DirectBasisExample, DirectBasisOracleConfig, DirectBasisTrainConfig, EvalConfig, eval_indices,
    evaluate_direct_basis_example,
};

#[derive(Clone)]
struct DirectBasisOracleEvalContext<'a> {
    base: &'a NpaModel,
    hashgrid: &'a burn_automata_kernels::HashGridConfig,
    train_config: DirectBasisTrainConfig,
    oracle_config: DirectBasisOracleConfig,
    model_output_dir: Option<&'a Path>,
}

#[derive(Clone, Copy)]
struct DirectBasisOracleJob<'a> {
    example: &'a DirectBasisExample,
    idx: usize,
    seed: u64,
}

const MAX_TILED_BURN_ORACLE_PARTICLES: usize = 2048;
const QUALITY_TILED_PARTICLE_THRESHOLD: usize = 1024;
const DEFAULT_DENSE_CHUNK_FLOATS: usize = 16_000_000;
const DEFAULT_SPLAT_CHUNK_FLOATS: usize = 16_000_000;
const QUALITY_DENSE_CHUNK_FLOATS: usize = 512 * 1024;
const QUALITY_SPLAT_CHUNK_FLOATS: usize = 512 * 1024;

pub(super) fn evaluate_direct_basis_oracles(
    base: &NpaModel,
    train_examples: &[DirectBasisExample],
    holdout_examples: &[DirectBasisExample],
    hashgrid: &burn_automata_kernels::HashGridConfig,
    eval_config: DirectBasisTrainConfig,
    oracle_config: DirectBasisOracleConfig,
    oracle_model_dir: Option<&Path>,
) -> Result<Option<CliHyper2dDirectBasisOracleReport>, Box<dyn std::error::Error>> {
    if oracle_config.train_examples == 0 && oracle_config.holdout_examples == 0 {
        return Ok(None);
    }
    let started = Instant::now();
    let train_indices = oracle_indices(
        train_examples.len(),
        oracle_config.train_examples,
        oracle_config.seed,
    );
    let holdout_indices = oracle_indices(
        holdout_examples.len(),
        oracle_config.holdout_examples,
        oracle_config.seed ^ 0x90_1d_2d,
    );
    let context = DirectBasisOracleEvalContext {
        base,
        hashgrid,
        train_config: eval_config,
        oracle_config: oracle_config.clone(),
        model_output_dir: oracle_model_dir,
    };
    let mut jobs = Vec::with_capacity(train_indices.len() + holdout_indices.len());
    for &idx in &train_indices {
        let seed_index = train_examples[idx].bank_split_index.unwrap_or(idx);
        jobs.push(DirectBasisOracleJob {
            example: &train_examples[idx],
            idx: seed_index,
            seed: oracle_config.seed,
        });
    }
    for &idx in &holdout_indices {
        let seed_index = holdout_examples[idx].bank_split_index.unwrap_or(idx);
        jobs.push(DirectBasisOracleJob {
            example: &holdout_examples[idx],
            idx: seed_index,
            seed: oracle_config.seed ^ 0x90_1d_2d,
        });
    }
    let entries = if matches!(
        oracle_config.backend,
        DirectBasisOracleBackendArg::Wgpu | DirectBasisOracleBackendArg::Cuda
    ) && oracle_config.gpu_parallel_jobs > 1
    {
        evaluate_direct_basis_oracle_entries_burn_model_batch(
            context,
            &jobs,
            oracle_config.gpu_parallel_jobs,
        )?
    } else {
        let mut entries = Vec::with_capacity(jobs.len());
        for job in &jobs {
            entries.push(evaluate_direct_basis_oracle_entry(
                context.clone(),
                job.example,
                job.idx,
                job.seed,
            )?);
        }
        entries
    };
    let elapsed = started.elapsed();
    let effective_particle_steps_per_sec = oracle_effective_particle_steps_per_sec(
        &entries,
        oracle_config.batch_size,
        eval_config.rollout_steps,
        elapsed,
    );
    let mean_reported_particle_steps_per_sec =
        oracle_mean_reported_particle_steps_per_sec(&entries);
    let train_summary = oracle_summary(
        entries
            .iter()
            .filter(|entry| entry.split == "train")
            .collect(),
    );
    let holdout_summary = oracle_summary(
        entries
            .iter()
            .filter(|entry| entry.split == "holdout")
            .collect(),
    );
    Ok(Some(CliHyper2dDirectBasisOracleReport {
        backend: oracle_config.backend,
        gpu_device: matches!(oracle_config.backend, DirectBasisOracleBackendArg::Cuda)
            .then(|| oracle_config.gpu_device.clone()),
        resume_existing: oracle_config.resume_existing,
        gpu_parallel_jobs: oracle_config.gpu_parallel_jobs,
        train_examples_requested: oracle_config.train_examples,
        holdout_examples_requested: oracle_config.holdout_examples,
        train_examples: train_indices.len(),
        holdout_examples: holdout_indices.len(),
        epochs: oracle_config.epochs,
        repetitions: oracle_config.repetitions,
        batch_size: oracle_config.batch_size,
        pool_size: oracle_config.pool_size,
        learning_rate: oracle_config.learning_rate,
        weight_decay: oracle_config.weight_decay,
        grad_clip_norm: oracle_config.grad_clip_norm,
        seed: oracle_config.seed,
        effective_particle_steps_per_sec,
        mean_reported_particle_steps_per_sec,
        train_summary,
        holdout_summary,
        entries,
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
    }))
}

fn evaluate_direct_basis_oracle_entries_burn_model_batch(
    context: DirectBasisOracleEvalContext<'_>,
    jobs: &[DirectBasisOracleJob<'_>],
    model_batch_size: usize,
) -> Result<Vec<CliHyper2dDirectBasisOracleEntry>, Box<dyn std::error::Error>> {
    let model_batch_size = model_batch_size.max(1);
    let mut entries = Vec::with_capacity(jobs.len());
    for chunk in jobs.chunks(model_batch_size) {
        if context.oracle_config.resume_existing {
            for job in chunk {
                entries.push(evaluate_direct_basis_oracle_entry(
                    context.clone(),
                    job.example,
                    job.idx,
                    job.seed,
                )?);
            }
            continue;
        }
        let mut groups = Vec::<(usize, Vec<DirectBasisOracleJob<'_>>)>::new();
        for job in chunk {
            let particles = job
                .example
                .source
                .particles
                .unwrap_or(context.train_config.rollout_particles);
            if let Some((_, group)) = groups
                .iter_mut()
                .find(|(group_particles, _)| *group_particles == particles)
            {
                group.push(*job);
            } else {
                groups.push((particles, vec![*job]));
            }
        }
        for (particles, group) in groups {
            println!(
                "oracle target2d {:?} vectorized model batch jobs={} model_batch_size={} particles={}",
                context.oracle_config.backend,
                group.len(),
                model_batch_size,
                particles
            );
            entries.extend(train_burn_model_batch_direct_basis_oracles(
                context.clone(),
                &group,
            )?);
        }
    }
    Ok(entries)
}

fn train_burn_model_batch_direct_basis_oracles(
    context: DirectBasisOracleEvalContext<'_>,
    jobs: &[DirectBasisOracleJob<'_>],
) -> Result<Vec<CliHyper2dDirectBasisOracleEntry>, Box<dyn std::error::Error>> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }
    let DirectBasisOracleEvalContext {
        base,
        hashgrid,
        train_config: eval_config,
        oracle_config,
        model_output_dir,
    } = context;
    let backend = oracle_config.backend;
    let (backend_label, metrics_suffix) = match backend {
        DirectBasisOracleBackendArg::Wgpu => ("Burn/WGPU", "burn-wgpu"),
        DirectBasisOracleBackendArg::Cuda => ("Burn/CUDA", "burn-cuda"),
        _ => unreachable!("Burn model batch requires WGPU or CUDA backend"),
    };
    let Some(dir) = model_output_dir else {
        return Err(std::io::Error::other(format!(
            "{backend_label} vectorized oracle validation requires an oracle model output directory"
        ))
        .into());
    };

    let evals = jobs
        .iter()
        .map(|job| oracle_eval_config(eval_config, job.example, job.idx, job.seed))
        .collect::<Vec<_>>();
    let particle_count = evals[0].particle_count;
    if evals
        .iter()
        .any(|eval| eval.particle_count != particle_count)
    {
        return Err(std::io::Error::other(
            "Burn vectorized oracle batches require homogeneous particle counts; reduce oracle.gpu_parallel_jobs or group homogeneous slices",
        )
        .into());
    }
    if particle_count > MAX_TILED_BURN_ORACLE_PARTICLES {
        return Err(std::io::Error::other(format!(
            "{backend_label} vectorized tiled-autodiff oracle is capped at {MAX_TILED_BURN_ORACLE_PARTICLES} particles; requested {particle_count}",
        ))
        .into());
    }

    let mut shared_losses = Vec::with_capacity(jobs.len());
    let mut zero_adapter_losses = Vec::with_capacity(jobs.len());
    let mut initial_eval_losses = Vec::with_capacity(jobs.len());
    let mut models = Vec::with_capacity(jobs.len());
    let mut train_examples = Vec::with_capacity(jobs.len());
    for (local, job) in jobs.iter().enumerate() {
        let eval = evals[local];
        let shared_loss = evaluate_direct_basis_example(
            base,
            job.example,
            hashgrid,
            eval,
            eval_config.loss_config,
        )?;
        let zero_shared = zero_adapter_example(base, job.example);
        let zero_adapter_loss = evaluate_direct_basis_example(
            base,
            &zero_shared,
            hashgrid,
            eval,
            eval_config.loss_config,
        )?;
        let oracle_model = NpaModel::upstream_seeded(
            NpaConfig::growing_2d(),
            oracle_config.seed.wrapping_add(job.idx as u64),
        );
        let zero_oracle = zero_adapter_example(&oracle_model, job.example);
        let initial_eval_loss = evaluate_direct_basis_example(
            &oracle_model,
            &zero_oracle,
            hashgrid,
            eval,
            eval_config.loss_config,
        )?;
        shared_losses.push(shared_loss);
        zero_adapter_losses.push(zero_adapter_loss);
        initial_eval_losses.push(initial_eval_loss);
        train_examples.push(zero_oracle);
        models.push(oracle_model);
    }

    let quality_tiled = particle_count >= QUALITY_TILED_PARTICLE_THRESHOLD;
    let tbptt_chunk_steps = if quality_tiled {
        1
    } else {
        eval_config.rollout_steps.max(1)
    };
    let max_dense_chunk_floats = if quality_tiled {
        QUALITY_DENSE_CHUNK_FLOATS
    } else {
        DEFAULT_DENSE_CHUNK_FLOATS
    };
    let max_splat_chunk_floats = if quality_tiled {
        QUALITY_SPLAT_CHUNK_FLOATS
    } else {
        DEFAULT_SPLAT_CHUNK_FLOATS
    };
    let training_steps = oracle_config
        .epochs
        .saturating_add(1)
        .saturating_mul(oracle_config.repetitions);
    let burn_config = DirectBasisTrainConfig {
        steps: training_steps,
        report_interval: oracle_config.report_interval.max(1),
        example_batch_size: jobs.len(),
        tbptt_chunk_steps,
        loss_on_final_chunk_only: false,
        use_particle_pool: false,
        pool_size: 0,
        inject_seed_interval: 0,
        brush_size: 0.0,
        stopgrad_pos: models[0].config.stopgrad_pos,
        stopgrad_state: models[0].config.stopgrad_state,
        rollout_particles: particle_count,
        rollout_step_min: eval_config.rollout_steps,
        rollout_steps: eval_config.rollout_steps,
        update_prob: eval_config.update_prob,
        seed: evals[0].seed,
        seed_scale: eval_config.seed_scale,
        seed_mode: eval_config.seed_mode,
        grid_eps: hashgrid.eps,
        motion_scale: models[0].config.alpha * models[0].config.motion_eps(hashgrid.eps),
        loss_config: eval_config.loss_config,
        per_parameter_grad_normalization: eval_config.per_parameter_grad_normalization,
        base_sgd: SgdConfig {
            learning_rate: oracle_config.learning_rate,
            weight_decay: oracle_config.weight_decay,
            grad_clip_norm: oracle_config.grad_clip_norm,
        },
        adapter_sgd: SgdConfig {
            learning_rate: 0.0,
            weight_decay: 0.0,
            grad_clip_norm: 0.0,
        },
        adapter_l2_weight: 0.0,
        update_base: true,
        eval_examples: 0,
        eval_interval: 0,
        eval_batch_size: 1,
        eval_seed: evals[0].seed,
        system_memory_budget_gb: Some(24.0),
        gpu_memory_budget_gb: Some(24.0),
        max_dense_train_particles: MAX_TILED_BURN_ORACLE_PARTICLES,
        max_dense_chunk_floats,
        max_splat_chunk_floats,
    };
    let batch_report = match backend {
        DirectBasisOracleBackendArg::Wgpu => {
            super::dense::train_oracle_models_burn_wgpu(&mut models, &train_examples, burn_config)?
        }
        DirectBasisOracleBackendArg::Cuda => {
            super::dense::train_oracle_models_burn_cuda(&mut models, &train_examples, burn_config)?
        }
        _ => unreachable!("Burn model batch requires WGPU or CUDA backend"),
    };

    let mut entries = Vec::with_capacity(jobs.len());
    for (local, job) in jobs.iter().enumerate() {
        let eval = evals[local];
        let slug = super::super::sources::sanitize_slug(&job.example.source.slug);
        let split_dir = dir.join(job.example.split.label());
        let model_output = split_dir.join(format!("{slug}.bpk"));
        let metrics_output = split_dir.join(format!("{slug}.{metrics_suffix}.json"));
        let final_loss = evaluate_direct_basis_example(
            &models[local],
            &train_examples[local],
            hashgrid,
            eval,
            eval_config.loss_config,
        )?;
        let best_eval_loss = batch_report.best_train_loss[local]
            .map(|loss| Target2dLossReport {
                total_loss: loss,
                ..final_loss
            })
            .unwrap_or(final_loss);
        let manifest = BpkModelManifest::from_model(
            &models[local],
            hashgrid.clone(),
            Some(format!(
                "oracle-target2d-{metrics_suffix}-model-batch:{}:{}:steps={training_steps}",
                job.example.split.label(),
                job.example.source.slug
            )),
        );
        crate::import::save_manifest(&model_output, &manifest)?;
        let per_model_history = batch_report
            .per_model_history
            .get(local)
            .cloned()
            .unwrap_or_default();
        let median_particle_steps_per_sec =
            median_direct_basis_particle_steps_per_sec(&per_model_history);
        let metrics = serde_json::json!({
            "backend": batch_report.backend,
            "device": batch_report.device.clone(),
            "dense_autodiff_oracle": !quality_tiled,
            "tiled_autodiff_oracle": quality_tiled,
            "model_vectorized_oracle": true,
            "model_batch_size": jobs.len(),
            "model_batch_index": local,
            "optimizer_state": "separate_adamw_state_per_oracle_model",
            "parameter_sharing": false,
            "particle_cap": MAX_TILED_BURN_ORACLE_PARTICLES,
            "rollout_batch_size": 1,
            "tbptt_chunk_steps": tbptt_chunk_steps,
            "max_dense_chunk_floats": max_dense_chunk_floats,
            "max_splat_chunk_floats": max_splat_chunk_floats,
            "initial_eval_loss": initial_eval_losses[local],
            "final_eval_loss": final_loss,
            "best_eval_loss": best_eval_loss,
            "epochs_completed": training_steps,
            "median_particle_steps_per_sec": median_particle_steps_per_sec,
            "best_train_loss": batch_report.best_train_loss[local],
            "best_train_step": batch_report.best_train_step[local],
            "history": per_model_history,
            "batch_history": batch_report.history.clone(),
            "metrics": batch_report.metrics.clone(),
        });
        if let Some(parent) = metrics_output.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&metrics_output, serde_json::to_string_pretty(&metrics)?)?;
        let oracle_loss = final_loss.total_loss;
        let shared_loss = shared_losses[local];
        let zero_adapter_loss = zero_adapter_losses[local];
        let loss_gap_to_oracle = shared_loss.total_loss - oracle_loss;
        let loss_ratio_to_oracle = shared_loss.total_loss / oracle_loss.max(f32::MIN_POSITIVE);
        let loss_gap_to_zero = shared_loss.total_loss - zero_adapter_loss.total_loss;
        let loss_ratio_to_zero =
            shared_loss.total_loss / zero_adapter_loss.total_loss.max(f32::MIN_POSITIVE);
        let zero_ratio_to_oracle =
            zero_adapter_loss.total_loss / oracle_loss.max(f32::MIN_POSITIVE);
        entries.push(CliHyper2dDirectBasisOracleEntry {
            slug: job.example.source.slug.clone(),
            split: job.example.split.label(),
            condition: job.example.source.condition_path.display().to_string(),
            oracle_backend: backend,
            oracle_model_output: Some(model_output.display().to_string()),
            oracle_checkpoint_output: None,
            oracle_metrics_output: Some(metrics_output.display().to_string()),
            shared_loss,
            zero_adapter_loss,
            oracle_initial_eval_loss: initial_eval_losses[local],
            oracle_final_loss: final_loss,
            oracle_best_eval_loss: best_eval_loss,
            oracle_epochs_completed: training_steps,
            oracle_median_particle_steps_per_sec: median_particle_steps_per_sec,
            loss_gap_to_oracle,
            loss_ratio_to_oracle,
            loss_gap_to_zero,
            loss_ratio_to_zero,
            zero_ratio_to_oracle,
        });
    }
    Ok(entries)
}

fn oracle_eval_config(
    eval_config: DirectBasisTrainConfig,
    example: &DirectBasisExample,
    idx: usize,
    seed: u64,
) -> EvalConfig {
    let eval_seed = seed.wrapping_add(idx as u64);
    EvalConfig {
        particle_count: example
            .source
            .particles
            .unwrap_or(eval_config.rollout_particles),
        rollout_steps: eval_config.rollout_steps,
        update_prob: example
            .source
            .update_prob
            .unwrap_or(eval_config.update_prob),
        seed: eval_seed,
        seed_scale: example.source.seed_scale.unwrap_or(eval_config.seed_scale),
        seed_mode: eval_config.seed_mode,
    }
}

fn evaluate_direct_basis_oracle_entry(
    context: DirectBasisOracleEvalContext<'_>,
    example: &DirectBasisExample,
    idx: usize,
    seed: u64,
) -> Result<CliHyper2dDirectBasisOracleEntry, Box<dyn std::error::Error>> {
    let DirectBasisOracleEvalContext {
        base,
        hashgrid,
        train_config: eval_config,
        oracle_config,
        model_output_dir,
    } = context;
    let oracle_backend = oracle_config.backend;
    let eval = oracle_eval_config(eval_config, example, idx, seed);
    let shared_loss =
        evaluate_direct_basis_example(base, example, hashgrid, eval, eval_config.loss_config)?;
    let zero_example = zero_adapter_example(base, example);
    let zero_adapter_loss = evaluate_direct_basis_example(
        base,
        &zero_example,
        hashgrid,
        eval,
        eval_config.loss_config,
    )?;
    let training_config = Target2dTrainingConfig {
        epochs: oracle_config.epochs,
        repetitions: oracle_config.repetitions,
        report_interval: oracle_config.report_interval,
        batch_size: oracle_config.batch_size,
        pool_size: oracle_config.pool_size.max(oracle_config.batch_size),
        particle_count: example
            .source
            .particles
            .unwrap_or(eval_config.rollout_particles),
        step_min: eval_config.rollout_steps,
        step_max: eval_config.rollout_steps,
        inject_seed_interval: 16,
        update_prob: example
            .source
            .update_prob
            .unwrap_or(eval_config.update_prob),
        seed: eval.seed,
        seed_scale: example.source.seed_scale.unwrap_or(eval_config.seed_scale),
        seed_mode: eval_config.seed_mode,
        brush_size: 0.1,
        per_parameter_grad_normalization: eval_config.per_parameter_grad_normalization,
        optimizer: AdamWConfig {
            learning_rate: oracle_config.learning_rate,
            weight_decay: oracle_config.weight_decay,
            grad_clip_norm: oracle_config.grad_clip_norm,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1.0e-8,
        },
        scheduler_milestones: Vec::new(),
        scheduler_gamma: 0.3,
    };
    println!(
        "oracle target2d {:?} {} {} epochs={} particles={} steps={}",
        oracle_config.backend,
        example.split.label(),
        example.source.slug,
        training_config.epochs * training_config.repetitions,
        training_config.particle_count,
        training_config.step_max
    );
    let oracle_training = match oracle_config.backend {
        DirectBasisOracleBackendArg::Cpu => train_cpu_direct_basis_oracle(
            base,
            example,
            hashgrid,
            eval_config,
            &oracle_config,
            training_config,
            idx,
            model_output_dir,
        )?,
        DirectBasisOracleBackendArg::Wgpu => train_burn_wgpu_direct_basis_oracle(
            example,
            hashgrid,
            eval,
            eval_config.loss_config,
            &oracle_config,
            training_config,
            idx,
            model_output_dir,
        )?,
        DirectBasisOracleBackendArg::Cuda => train_burn_cuda_direct_basis_oracle(
            example,
            hashgrid,
            eval,
            eval_config.loss_config,
            &oracle_config,
            training_config,
            idx,
            model_output_dir,
        )?,
    };
    let oracle_loss = oracle_training.final_loss.total_loss;
    let loss_gap_to_oracle = shared_loss.total_loss - oracle_loss;
    let loss_ratio_to_oracle = shared_loss.total_loss / oracle_loss.max(f32::MIN_POSITIVE);
    let loss_gap_to_zero = shared_loss.total_loss - zero_adapter_loss.total_loss;
    let loss_ratio_to_zero =
        shared_loss.total_loss / zero_adapter_loss.total_loss.max(f32::MIN_POSITIVE);
    let zero_ratio_to_oracle = zero_adapter_loss.total_loss / oracle_loss.max(f32::MIN_POSITIVE);
    Ok(CliHyper2dDirectBasisOracleEntry {
        slug: example.source.slug.clone(),
        split: example.split.label(),
        condition: example.source.condition_path.display().to_string(),
        oracle_backend,
        oracle_model_output: oracle_training.model_output,
        oracle_checkpoint_output: oracle_training.checkpoint_output,
        oracle_metrics_output: oracle_training.metrics_output,
        shared_loss,
        zero_adapter_loss,
        oracle_initial_eval_loss: oracle_training.initial_eval_loss,
        oracle_final_loss: oracle_training.final_loss,
        oracle_best_eval_loss: oracle_training.best_eval_loss,
        oracle_epochs_completed: oracle_training.epochs_completed,
        oracle_median_particle_steps_per_sec: oracle_training.median_particle_steps_per_sec,
        loss_gap_to_oracle,
        loss_ratio_to_oracle,
        loss_gap_to_zero,
        loss_ratio_to_zero,
        zero_ratio_to_oracle,
    })
}

struct DirectBasisOracleTrainingResult {
    model_output: Option<String>,
    checkpoint_output: Option<String>,
    metrics_output: Option<String>,
    initial_eval_loss: Target2dLossReport,
    final_loss: Target2dLossReport,
    best_eval_loss: Target2dLossReport,
    epochs_completed: usize,
    median_particle_steps_per_sec: f64,
}

#[allow(clippy::too_many_arguments)]
fn train_cpu_direct_basis_oracle(
    base: &NpaModel,
    example: &DirectBasisExample,
    hashgrid: &burn_automata_kernels::HashGridConfig,
    eval_config: DirectBasisTrainConfig,
    oracle_config: &DirectBasisOracleConfig,
    training_config: Target2dTrainingConfig,
    idx: usize,
    model_output_dir: Option<&Path>,
) -> Result<DirectBasisOracleTrainingResult, Box<dyn std::error::Error>> {
    let mut oracle_model = NpaModel::upstream_seeded(
        NpaConfig::growing_2d(),
        oracle_config.seed.wrapping_add(idx as u64),
    );
    let oracle_training = train_target_2d(
        &mut oracle_model,
        hashgrid,
        &example.target,
        training_config,
        eval_config.loss_config,
    )?;
    let model_output = if let Some(dir) = model_output_dir {
        let slug = super::super::sources::sanitize_slug(&example.source.slug);
        let path = dir.join(example.split.label()).join(format!("{slug}.bpk"));
        let manifest = BpkModelManifest::from_model(
            &oracle_model,
            hashgrid.clone(),
            Some(format!(
                "oracle-target2d:{}:{}:epochs={}",
                example.split.label(),
                example.source.slug,
                oracle_training.epochs_completed
            )),
        );
        crate::import::save_manifest(&path, &manifest)?;
        Some(path.display().to_string())
    } else {
        None
    };
    let _ = base;
    Ok(DirectBasisOracleTrainingResult {
        model_output,
        checkpoint_output: None,
        metrics_output: None,
        initial_eval_loss: oracle_training.initial_eval_loss,
        final_loss: oracle_training.final_loss,
        best_eval_loss: oracle_training.best_eval_loss,
        epochs_completed: oracle_training.epochs_completed,
        median_particle_steps_per_sec: oracle_training.median_particle_steps_per_sec,
    })
}

#[allow(clippy::too_many_arguments)]
fn train_burn_wgpu_direct_basis_oracle(
    example: &DirectBasisExample,
    hashgrid: &burn_automata_kernels::HashGridConfig,
    eval: EvalConfig,
    loss_config: Target2dLossConfig,
    oracle_config: &DirectBasisOracleConfig,
    training_config: Target2dTrainingConfig,
    idx: usize,
    model_output_dir: Option<&Path>,
) -> Result<DirectBasisOracleTrainingResult, Box<dyn std::error::Error>> {
    train_burn_dense_direct_basis_oracle(
        DirectBasisOracleBackendArg::Wgpu,
        example,
        hashgrid,
        eval,
        loss_config,
        oracle_config,
        training_config,
        idx,
        model_output_dir,
    )
}

#[allow(clippy::too_many_arguments)]
fn train_burn_cuda_direct_basis_oracle(
    example: &DirectBasisExample,
    hashgrid: &burn_automata_kernels::HashGridConfig,
    eval: EvalConfig,
    loss_config: Target2dLossConfig,
    oracle_config: &DirectBasisOracleConfig,
    training_config: Target2dTrainingConfig,
    idx: usize,
    model_output_dir: Option<&Path>,
) -> Result<DirectBasisOracleTrainingResult, Box<dyn std::error::Error>> {
    train_burn_dense_direct_basis_oracle(
        DirectBasisOracleBackendArg::Cuda,
        example,
        hashgrid,
        eval,
        loss_config,
        oracle_config,
        training_config,
        idx,
        model_output_dir,
    )
}

#[allow(clippy::too_many_arguments)]
fn train_burn_dense_direct_basis_oracle(
    backend: DirectBasisOracleBackendArg,
    example: &DirectBasisExample,
    hashgrid: &burn_automata_kernels::HashGridConfig,
    eval: EvalConfig,
    loss_config: Target2dLossConfig,
    oracle_config: &DirectBasisOracleConfig,
    training_config: Target2dTrainingConfig,
    idx: usize,
    model_output_dir: Option<&Path>,
) -> Result<DirectBasisOracleTrainingResult, Box<dyn std::error::Error>> {
    let (backend_label, metrics_suffix) = match backend {
        DirectBasisOracleBackendArg::Wgpu => ("Burn/WGPU", "burn-wgpu"),
        DirectBasisOracleBackendArg::Cuda => ("Burn/CUDA", "burn-cuda"),
        _ => unreachable!("dense Burn oracle backend must be WGPU or CUDA"),
    };
    if training_config.particle_count > MAX_TILED_BURN_ORACLE_PARTICLES {
        return Err(std::io::Error::other(format!(
            "{backend_label} tiled-autodiff oracle is capped at {MAX_TILED_BURN_ORACLE_PARTICLES} particles; requested {}",
            training_config.particle_count
        ))
        .into());
    }
    let Some(dir) = model_output_dir else {
        return Err(std::io::Error::other(format!(
            "{backend_label} oracle validation requires an oracle model output directory"
        ))
        .into());
    };

    let slug = super::super::sources::sanitize_slug(&example.source.slug);
    let split_dir = dir.join(example.split.label());
    let model_output = split_dir.join(format!("{slug}.bpk"));
    let metrics_output = split_dir.join(format!("{slug}.{metrics_suffix}.json"));

    if oracle_config.resume_existing && model_output.is_file() && metrics_output.is_file() {
        println!(
            "oracle target2d {metrics_suffix} resume {} {}",
            example.split.label(),
            example.source.slug
        );
        let metrics = read_oracle_metrics(&metrics_output)?;
        let oracle_model = crate::import::load_manifest(&model_output)?.into_model();
        let zero_example = zero_adapter_example(&oracle_model, example);
        let final_loss = evaluate_direct_basis_example(
            &oracle_model,
            &zero_example,
            hashgrid,
            eval,
            loss_config,
        )?;
        let initial_eval_loss = target2d_loss_metric(&metrics, "initial_eval_loss")?;
        let best_eval_loss = target2d_loss_metric(&metrics, "best_eval_loss").unwrap_or(final_loss);
        return Ok(DirectBasisOracleTrainingResult {
            model_output: Some(model_output.display().to_string()),
            checkpoint_output: None,
            metrics_output: Some(metrics_output.display().to_string()),
            initial_eval_loss,
            final_loss,
            best_eval_loss,
            epochs_completed: metrics_usize(&metrics, "epochs_completed").unwrap_or_else(|| {
                training_config
                    .epochs
                    .saturating_add(1)
                    .saturating_mul(training_config.repetitions)
            }),
            median_particle_steps_per_sec: metrics_f64(&metrics, "median_particle_steps_per_sec")
                .unwrap_or_default(),
        });
    }

    let mut oracle_model = NpaModel::upstream_seeded(
        NpaConfig::growing_2d(),
        oracle_config.seed.wrapping_add(idx as u64),
    );
    let rollout_batch_size = training_config.batch_size.max(1);
    let mut train_examples = (0..rollout_batch_size)
        .map(|_| zero_adapter_example(&oracle_model, example))
        .collect::<Vec<_>>();
    let mut holdout_examples = Vec::new();
    let initial_eval_loss = evaluate_direct_basis_example(
        &oracle_model,
        &train_examples[0],
        hashgrid,
        eval,
        loss_config,
    )?;
    let training_steps = training_config
        .epochs
        .saturating_add(1)
        .saturating_mul(training_config.repetitions);
    let quality_tiled = training_config.particle_count >= QUALITY_TILED_PARTICLE_THRESHOLD;
    let tbptt_chunk_steps = if quality_tiled {
        1
    } else {
        eval.rollout_steps.max(1)
    };
    let max_dense_chunk_floats = if quality_tiled {
        QUALITY_DENSE_CHUNK_FLOATS
    } else {
        DEFAULT_DENSE_CHUNK_FLOATS
    };
    let max_splat_chunk_floats = if quality_tiled {
        QUALITY_SPLAT_CHUNK_FLOATS
    } else {
        DEFAULT_SPLAT_CHUNK_FLOATS
    };
    let burn_config = DirectBasisTrainConfig {
        steps: training_steps,
        report_interval: training_config.report_interval.max(1),
        example_batch_size: rollout_batch_size,
        tbptt_chunk_steps,
        loss_on_final_chunk_only: false,
        use_particle_pool: false,
        pool_size: 0,
        inject_seed_interval: 0,
        brush_size: 0.0,
        stopgrad_pos: oracle_model.config.stopgrad_pos,
        stopgrad_state: oracle_model.config.stopgrad_state,
        rollout_particles: training_config.particle_count,
        rollout_step_min: eval.rollout_steps,
        rollout_steps: eval.rollout_steps,
        update_prob: training_config.update_prob,
        seed: training_config.seed,
        seed_scale: training_config.seed_scale,
        seed_mode: training_config.seed_mode,
        grid_eps: hashgrid.eps,
        motion_scale: oracle_model.config.alpha * oracle_model.config.motion_eps(hashgrid.eps),
        loss_config,
        per_parameter_grad_normalization: training_config.per_parameter_grad_normalization,
        base_sgd: SgdConfig {
            learning_rate: oracle_config.learning_rate,
            weight_decay: oracle_config.weight_decay,
            grad_clip_norm: oracle_config.grad_clip_norm,
        },
        adapter_sgd: SgdConfig {
            learning_rate: 0.0,
            weight_decay: 0.0,
            grad_clip_norm: 0.0,
        },
        adapter_l2_weight: 0.0,
        update_base: true,
        eval_examples: 1,
        eval_interval: training_config.report_interval.max(1),
        eval_batch_size: 1,
        eval_seed: eval.seed,
        system_memory_budget_gb: Some(24.0),
        gpu_memory_budget_gb: Some(24.0),
        max_dense_train_particles: MAX_TILED_BURN_ORACLE_PARTICLES,
        max_dense_chunk_floats,
        max_splat_chunk_floats,
    };
    let no_phase_config = DirectBasisTrainConfig {
        steps: 0,
        update_base: false,
        ..burn_config
    };
    let burn_report = match backend {
        DirectBasisOracleBackendArg::Wgpu => super::dense::train_direct_basis_burn_wgpu(
            &mut oracle_model,
            &mut train_examples,
            &mut holdout_examples,
            burn_config,
            no_phase_config,
            no_phase_config,
        )?,
        DirectBasisOracleBackendArg::Cuda => super::dense::train_direct_basis_burn_cuda(
            &mut oracle_model,
            &mut train_examples,
            &mut holdout_examples,
            burn_config,
            no_phase_config,
            no_phase_config,
        )?,
        _ => unreachable!("dense Burn oracle backend must be WGPU or CUDA"),
    };
    let final_loss = evaluate_direct_basis_example(
        &oracle_model,
        &train_examples[0],
        hashgrid,
        eval,
        loss_config,
    )?;
    let best_eval_loss = burn_report
        .best_train_loss
        .map(|loss| Target2dLossReport {
            total_loss: loss,
            ..final_loss
        })
        .unwrap_or(final_loss);
    let median_particle_steps_per_sec =
        median_direct_basis_particle_steps_per_sec(&burn_report.history);
    let manifest = BpkModelManifest::from_model(
        &oracle_model,
        hashgrid.clone(),
        Some(format!(
            "oracle-target2d-{metrics_suffix}:{}:{}:steps={training_steps}",
            example.split.label(),
            example.source.slug
        )),
    );
    crate::import::save_manifest(&model_output, &manifest)?;
    let metrics = serde_json::json!({
        "backend": burn_report.backend,
        "device": burn_report.device,
        "dense_autodiff_oracle": !quality_tiled,
        "tiled_autodiff_oracle": quality_tiled,
        "particle_cap": MAX_TILED_BURN_ORACLE_PARTICLES,
        "rollout_batch_size": rollout_batch_size,
        "tbptt_chunk_steps": tbptt_chunk_steps,
        "max_dense_chunk_floats": max_dense_chunk_floats,
        "max_splat_chunk_floats": max_splat_chunk_floats,
        "initial_eval_loss": initial_eval_loss,
        "final_eval_loss": final_loss,
        "best_eval_loss": best_eval_loss,
        "epochs_completed": training_steps,
        "median_particle_steps_per_sec": median_particle_steps_per_sec,
        "best_train_loss": burn_report.best_train_loss,
        "best_train_step": burn_report.best_train_step,
        "history": burn_report.history,
        "train_refine_history": burn_report.train_refine_history,
        "holdout_history": burn_report.holdout_history,
        "metrics": burn_report.metrics,
    });
    if let Some(parent) = metrics_output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&metrics_output, serde_json::to_string_pretty(&metrics)?)?;
    Ok(DirectBasisOracleTrainingResult {
        model_output: Some(model_output.display().to_string()),
        checkpoint_output: None,
        metrics_output: Some(metrics_output.display().to_string()),
        initial_eval_loss,
        final_loss,
        best_eval_loss,
        epochs_completed: training_steps,
        median_particle_steps_per_sec,
    })
}

fn read_oracle_metrics(path: &Path) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    serde_json::from_str(&text).map_err(|err| {
        std::io::Error::other(format!(
            "failed to parse GPU oracle metrics {}: {err}",
            path.display()
        ))
        .into()
    })
}

fn median_direct_basis_particle_steps_per_sec(
    history: &[CliHyper2dDirectBasisHistoryEntry],
) -> f64 {
    let mut values = history
        .iter()
        .filter_map(|entry| {
            (entry.particle_steps_per_sec.is_finite() && entry.particle_steps_per_sec > 0.0)
                .then_some(entry.particle_steps_per_sec)
        })
        .collect::<Vec<_>>();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

fn target2d_loss_metric(
    metrics: &serde_json::Value,
    key: &str,
) -> Result<Target2dLossReport, Box<dyn std::error::Error>> {
    let value = metrics.get(key).cloned().ok_or_else(|| {
        std::io::Error::other(format!("GPU oracle metrics missing `{key}` loss block"))
    })?;
    serde_json::from_value(value).map_err(|err| {
        std::io::Error::other(format!("invalid GPU oracle `{key}` loss block: {err}")).into()
    })
}

fn metrics_usize(metrics: &serde_json::Value, key: &str) -> Option<usize> {
    metrics
        .get(key)?
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
}

fn metrics_f64(metrics: &serde_json::Value, key: &str) -> Option<f64> {
    metrics.get(key)?.as_f64()
}

fn oracle_effective_particle_steps_per_sec(
    entries: &[CliHyper2dDirectBasisOracleEntry],
    training_batch_size: usize,
    rollout_steps: usize,
    elapsed: std::time::Duration,
) -> f64 {
    let elapsed_secs = elapsed.as_secs_f64();
    if elapsed_secs <= f64::MIN_POSITIVE {
        return 0.0;
    }
    let particle_steps = entries
        .iter()
        .map(|entry| {
            entry.oracle_epochs_completed as f64
                * training_batch_size as f64
                * entry.oracle_final_loss.particle_count as f64
                * rollout_steps as f64
        })
        .sum::<f64>();
    particle_steps / elapsed_secs
}

fn oracle_mean_reported_particle_steps_per_sec(
    entries: &[CliHyper2dDirectBasisOracleEntry],
) -> f64 {
    let mut count = 0usize;
    let mut sum = 0.0;
    for entry in entries {
        if entry.oracle_median_particle_steps_per_sec > 0.0 {
            count += 1;
            sum += entry.oracle_median_particle_steps_per_sec;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

fn zero_adapter_example(base: &NpaModel, example: &DirectBasisExample) -> DirectBasisExample {
    let mut zero = example.clone();
    zero.adapter =
        NpaLowRankAdapter::zeros(&base.config, example.adapter.rank, example.adapter.alpha);
    zero
}

fn oracle_summary(
    entries: Vec<&CliHyper2dDirectBasisOracleEntry>,
) -> Option<CliHyper2dDirectBasisOracleSummary> {
    if entries.is_empty() {
        return None;
    }
    let mut shared = 0.0_f32;
    let mut zero = 0.0_f32;
    let mut oracle = 0.0_f32;
    let mut gap = 0.0_f32;
    let mut ratio = 0.0_f32;
    let mut max_ratio = 0.0_f32;
    let mut gap_to_zero = 0.0_f32;
    let mut ratio_to_zero = 0.0_f32;
    let mut max_ratio_to_zero = 0.0_f32;
    let mut zero_ratio_to_oracle = 0.0_f32;
    for entry in &entries {
        shared += entry.shared_loss.total_loss;
        zero += entry.zero_adapter_loss.total_loss;
        oracle += entry.oracle_final_loss.total_loss;
        gap += entry.loss_gap_to_oracle;
        ratio += entry.loss_ratio_to_oracle;
        max_ratio = max_ratio.max(entry.loss_ratio_to_oracle);
        gap_to_zero += entry.loss_gap_to_zero;
        ratio_to_zero += entry.loss_ratio_to_zero;
        max_ratio_to_zero = max_ratio_to_zero.max(entry.loss_ratio_to_zero);
        zero_ratio_to_oracle += entry.zero_ratio_to_oracle;
    }
    let scale = 1.0 / entries.len() as f32;
    Some(CliHyper2dDirectBasisOracleSummary {
        examples: entries.len(),
        mean_shared_loss: shared * scale,
        mean_zero_loss: zero * scale,
        mean_oracle_loss: oracle * scale,
        mean_gap_to_oracle: gap * scale,
        mean_ratio_to_oracle: ratio * scale,
        max_ratio_to_oracle: max_ratio,
        mean_gap_to_zero: gap_to_zero * scale,
        mean_ratio_to_zero: ratio_to_zero * scale,
        max_ratio_to_zero,
        mean_zero_ratio_to_oracle: zero_ratio_to_oracle * scale,
    })
}

fn oracle_indices(examples_len: usize, requested_examples: usize, seed: u64) -> Vec<usize> {
    if requested_examples == 0 || examples_len == 0 {
        return Vec::new();
    }
    eval_indices(examples_len, requested_examples, seed)
}
