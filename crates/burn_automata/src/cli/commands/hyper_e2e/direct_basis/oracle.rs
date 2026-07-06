use crate::cli::prelude::*;

use super::{
    DirectBasisExample, DirectBasisOracleConfig, DirectBasisTrainConfig, EvalConfig, eval_indices,
    evaluate_direct_basis_example,
};

#[derive(Clone, Copy)]
struct DirectBasisOracleEvalContext<'a> {
    base: &'a NpaModel,
    hashgrid: &'a burn_automata_kernels::HashGridConfig,
    train_config: DirectBasisTrainConfig,
    oracle_config: DirectBasisOracleConfig,
    model_output_dir: Option<&'a Path>,
}

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
        oracle_config,
        model_output_dir: oracle_model_dir,
    };
    let mut entries = Vec::with_capacity(train_indices.len() + holdout_indices.len());
    for &idx in &train_indices {
        let seed_index = train_examples[idx].bank_split_index.unwrap_or(idx);
        entries.push(evaluate_direct_basis_oracle_entry(
            context,
            &train_examples[idx],
            seed_index,
            oracle_config.seed,
        )?);
    }
    for &idx in &holdout_indices {
        let seed_index = holdout_examples[idx].bank_split_index.unwrap_or(idx);
        entries.push(evaluate_direct_basis_oracle_entry(
            context,
            &holdout_examples[idx],
            seed_index,
            oracle_config.seed ^ 0x90_1d_2d,
        )?);
    }
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
        train_summary,
        holdout_summary,
        entries,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    }))
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
    let eval_seed = seed.wrapping_add(idx as u64);
    let eval = EvalConfig {
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
    };
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
    let mut oracle_model = NpaModel::upstream_seeded(
        NpaConfig::growing_2d(),
        oracle_config.seed.wrapping_add(idx as u64),
    );
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
        seed: eval_seed,
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
        "oracle target2d {} {} epochs={} particles={} steps={}",
        example.split.label(),
        example.source.slug,
        training_config.epochs * training_config.repetitions,
        training_config.particle_count,
        training_config.step_max
    );
    let oracle_training = train_target_2d(
        &mut oracle_model,
        hashgrid,
        &example.target,
        training_config,
        eval_config.loss_config,
    )?;
    let oracle_model_output = if let Some(dir) = model_output_dir {
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
        oracle_model_output,
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
