//! Rollout-free distillation from training-only endpoint latents into row flow.

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn train_e2e_amortization_distillation_step(
    generator: &mut BurnE2eGeneratorParams,
    generator_optimizer: &mut BurnE2eGeneratorAdamWState,
    npa_config: &NpaConfig,
    conditions: &BurnE2eConditionCache,
    condition_indices: &[usize],
    prepared_dino: Option<&BurnE2ePreparedDinoBatch>,
    config: BurnE2eRolloutTrainConfig,
    step_seed: u64,
    collect_metrics: bool,
    initial_state: Option<(Tensor3, Tensor3)>,
) -> Result<BurnE2eStepOutput, Box<dyn std::error::Error>> {
    let started = Instant::now();

    let condition_started = Instant::now();
    let (condition, expansion) = select_rollout_conditions(
        conditions,
        condition_indices,
        prepared_dino,
        config.rollouts_per_example,
    )?;
    let device = condition.device();
    let (x, s) = initial_state.unwrap_or_else(|| {
        (
            Tensor::<BurnBackend, 3>::zeros([1, 1, 2], &device),
            Tensor::<BurnBackend, 3>::zeros([1, 1, npa_config.state_dims], &device),
        )
    });
    if collect_metrics {
        sync_training_device(&device)?;
    }
    let generator_indices = generator_condition_indices(
        condition_indices,
        expansion.as_deref(),
        config.rollouts_per_example,
    );
    let endpoint_rows = generator
        .amortization_residual_rows(&generator_indices)
        .ok_or_else(|| {
            AutomataError::InvalidArgument(
                "amortization distillation requires a restored endpoint table".to_string(),
            )
        })?;
    let flow = generator.row_flow.as_ref().ok_or_else(|| {
        AutomataError::InvalidArgument(
            "amortization distillation requires conditional-row-flow".to_string(),
        )
    })?;
    let condition_batches = condition.shape().dims::<3>()[0];
    let prepared_flow_condition = flow.prepare_condition(condition);
    if collect_metrics {
        sync_training_device(&device)?;
    }
    let condition_adapter_ms = if collect_metrics {
        condition_started.elapsed().as_secs_f64() * 1_000.0
    } else {
        0.0
    };

    let loss_started = Instant::now();
    let distillation_objective = (config.amortization_distillation_weight > 0.0).then(|| {
        let generated_rows = flow.sample_rows_prepared_steps(
            &prepared_flow_condition,
            condition_batches,
            &device,
            config.generator_train_sample_steps,
        );
        flow.amortization_distillation_loss(
            generated_rows,
            endpoint_rows.clone(),
            npa_config,
        )
    });
    let self_rectification_objective = (config.flow_self_rectification_weight > 0.0).then(|| {
        flow.self_rectification_loss_to_endpoint_prepared(
            &prepared_flow_condition,
            endpoint_rows.clone(),
            npa_config,
            step_seed ^ 0x616d_6f72_745f_666c,
        )
    });
    let distillation_loss = if collect_metrics {
        distillation_objective
            .as_ref()
            .map(|loss| loss.clone().inner().into_scalar())
            .unwrap_or_default()
    } else {
        0.0
    };
    let self_rectification_loss = if collect_metrics {
        self_rectification_objective
            .as_ref()
            .map(|loss| loss.clone().inner().into_scalar())
            .unwrap_or_default()
    } else {
        0.0
    };
    let amortization_residual_rms = if collect_metrics {
        flow.endpoint_rms(endpoint_rows.clone(), npa_config)
            .inner()
            .into_scalar()
    } else {
        0.0
    };
    if collect_metrics {
        sync_training_device(&device)?;
    }
    let rollout_loss_ms = if collect_metrics {
        loss_started.elapsed().as_secs_f64() * 1_000.0
    } else {
        0.0
    };

    let objective = [
        distillation_objective.map(|loss| {
            loss.mul_scalar(config.amortization_distillation_weight.max(0.0))
        }),
        self_rectification_objective.map(|loss| {
            loss.mul_scalar(config.flow_self_rectification_weight.max(0.0))
        }),
    ]
    .into_iter()
    .flatten()
    .reduce(|left, right| left + right)
    .expect("rollout-free flow training has at least one active objective");
    let backward_started = Instant::now();
    let mut grads = objective.backward();
    let (generator_grad_norm, generator_grad_scale) = generator.apply_row_flow_adamw(
        &mut grads,
        generator_optimizer,
        config.generator_optimizer,
        config.generator_per_parameter_grad_normalization,
        collect_metrics,
    )?;
    if collect_metrics {
        sync_training_device(&device)?;
    }
    let backward_update_ms = if collect_metrics {
        backward_started.elapsed().as_secs_f64() * 1_000.0
    } else {
        0.0
    };
    let elapsed = started.elapsed();
    let loss = distillation_loss * config.amortization_distillation_weight.max(0.0)
        + self_rectification_loss * config.flow_self_rectification_weight.max(0.0);
    Ok(BurnE2eStepOutput {
        history: BurnE2eRolloutHistoryEntry {
            step: 0,
            loss,
            task_loss: 0.0,
            adapter_teacher_loss: 0.0,
            flow_matching_loss: 0.0,
            flow_self_rectification_loss: self_rectification_loss,
            amortization_distillation_loss: distillation_loss,
            amortization_residual_scale: 1.0,
            amortization_residual_rms,
            amortization_grad_norm: 0.0,
            amortization_endpoint_psnr_db: None,
            amortization_endpoint_p10_psnr_db: None,
            learning_rate_scale: 1.0,
            base_learning_rate: 0.0,
            generator_learning_rate: config.generator_optimizer.learning_rate,
            holdout_mean_psnr_db: None,
            holdout_mean_loss: None,
            base_grad_norm: 0.0,
            base_grad_scale: 1.0,
            generator_grad_norm,
            generator_grad_scale,
            examples_seen: generator_indices.len(),
            optimizer_examples_per_sec: generator_indices.len() as f64
                / elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
            pool_seed_replacements: 0,
            particle_steps_per_sec: 0.0,
            dense_pair_interactions_per_sec: 0.0,
            elapsed_ms: elapsed.as_secs_f64() * 1_000.0,
            condition_adapter_ms,
            rollout_loss_ms,
            backward_update_ms,
        },
        particle_steps: 0,
        final_x: x,
        final_s: s,
        per_example_losses: None,
    })
}
