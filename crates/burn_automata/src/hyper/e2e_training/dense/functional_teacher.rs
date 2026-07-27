//! Behavior-space distillation for generated NPA controllers.

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn train_e2e_functional_teacher_step(
    params: &BurnBaseParams,
    generator: &mut BurnE2eGeneratorParams,
    generator_optimizer: &mut BurnE2eGeneratorAdamWState,
    npa_config: &NpaConfig,
    conditions: &BurnE2eConditionCache,
    condition_indices: &[usize],
    prepared_dino: Option<&BurnE2ePreparedDinoBatch>,
    targets: &[BurnTargetExample],
    target_indices: &[usize],
    particle_count: usize,
    config: BurnE2eRolloutTrainConfig,
    step_seed: u64,
    collect_metrics: bool,
    initial_state: Option<(Tensor3, Tensor3)>,
) -> Result<BurnE2eStepOutput, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let batch_len = condition_indices.len();
    let device = &targets[target_indices[0]].target_rgb.device();
    let direct_config = direct_config_view(config);
    let (x, s) = initial_state.unwrap_or_else(|| {
        seed_batch_tensors(
            targets,
            target_indices,
            particle_count,
            direct_config,
            step_seed,
            device,
        )
    });

    if collect_metrics {
        sync_training_device(device)?;
    }
    let condition_started = Instant::now();
    let condition = conditions.select_prepared(condition_indices, prepared_dino)?;
    let teacher_vector = conditions.select_teacher(condition_indices).ok_or_else(|| {
        AutomataError::InvalidArgument(
            "functional teacher supervision requires adapter endpoints".to_string(),
        )
    })?;
    let (generated_adapter, _, prepared_flow_condition) =
        generator.adapter_batch_with_dense_rows(condition, npa_config, config);
    let teacher_adapter = BurnAdapterBatch::from_parameter_vector(
        teacher_vector.clone(),
        npa_config,
        config.adapter_rank,
        config.adapter_alpha,
    );
    if collect_metrics {
        sync_training_device(device)?;
    }
    let condition_adapter_ms = if collect_metrics {
        condition_started.elapsed().as_secs_f64() * 1_000.0
    } else {
        0.0
    };

    let probe_started = Instant::now();
    let probe_steps = if config.adapter_teacher_probe_rollout_steps == 0 {
        0
    } else {
        1 + step_seed as usize % config.adapter_teacher_probe_rollout_steps
    };
    let (probe_x, probe_s) = if probe_steps == 0 {
        (detach3(x), detach3(s))
    } else {
        let mut teacher_rng = StdRng::seed_from_u64(step_seed ^ 0x7465_6163_6865_7270);
        let (probe_x, probe_s, _) = rollout_batch_chunk(
            &params.detached(),
            &teacher_adapter,
            targets,
            target_indices,
            detach3(x),
            detach3(s),
            direct_config,
            particle_count,
            &mut teacher_rng,
            probe_steps,
            Tensor::<BurnBackend, 1>::zeros([batch_len], device),
            None,
        );
        (detach3(probe_x), detach3(probe_s))
    };
    let probes = rollout_dense_perception_batch(&probe_x, &probe_s, direct_config);
    let generated_update = params.forward_adapter_batch(probes.clone(), &generated_adapter);
    let teacher_update = detach3(
        params
            .detached()
            .forward_adapter_batch(probes, &teacher_adapter),
    );
    let functional_delta = generated_update - teacher_update;
    let functional_mse = functional_delta.clone().mul(functional_delta).mean();
    let parameter_delta = generated_adapter.to_parameter_vector() - teacher_vector.clone();
    let parameter_mse = parameter_delta.clone().mul(parameter_delta).mean();
    let teacher_objective = if config.adapter_teacher_objective
        == E2eAdapterTeacherObjective::Hybrid
    {
        functional_mse + parameter_mse.mul_scalar(FUNCTIONAL_TEACHER_PARAMETER_AUX_WEIGHT)
    } else {
        functional_mse
    };
    let flow_objective = if config.flow_matching_weight > 0.0 {
        Some(
            generator
                .row_flow
                .as_ref()
                .ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "flow_matching_weight requires conditional-row-flow".to_string(),
                    )
                })?
                .flow_matching_loss_prepared(
                    prepared_flow_condition
                        .as_ref()
                        .expect("row flow prepared a condition"),
                    teacher_vector,
                    npa_config,
                    config.adapter_rank,
                    config.adapter_alpha,
                    config.flow_match_inference_source,
                ),
        )
    } else {
        None
    };
    let teacher_loss = if collect_metrics {
        teacher_objective.clone().inner().into_scalar()
    } else {
        0.0
    };
    let flow_loss = if collect_metrics {
        flow_objective
            .as_ref()
            .map(|loss| loss.clone().inner().into_scalar())
            .unwrap_or_default()
    } else {
        0.0
    };
    if collect_metrics {
        sync_training_device(device)?;
    }
    let rollout_loss_ms = if collect_metrics {
        probe_started.elapsed().as_secs_f64() * 1_000.0
    } else {
        0.0
    };

    let objective = flow_objective.map_or_else(
        || teacher_objective.clone().mul_scalar(config.adapter_teacher_weight.max(0.0)),
        |flow| {
            teacher_objective
                .clone()
                .mul_scalar(config.adapter_teacher_weight.max(0.0))
                + flow.mul_scalar(config.flow_matching_weight.max(0.0))
        },
    );
    let backward_started = Instant::now();
    let mut grads = objective.backward();
    let (generator_grad_norm, generator_grad_scale, amortization_grad_norm) = generator
        .apply_adamw(
            &mut grads,
            generator_optimizer,
            config.generator_optimizer,
            GeneratorAdamWOptions {
                normalize: config.generator_per_parameter_grad_normalization,
                collect_metrics,
                active_identities: condition_indices,
                upstream_growing_min_lr_scale: (config.lr_schedule
                    == E2eLrSchedule::UpstreamGrowing)
                    .then_some(config.min_lr_scale),
            },
        )?;
    if collect_metrics {
        sync_training_device(device)?;
    }
    let backward_update_ms = if collect_metrics {
        backward_started.elapsed().as_secs_f64() * 1_000.0
    } else {
        0.0
    };
    let particle_steps = batch_len
        .saturating_mul(particle_count)
        .saturating_mul(probe_steps);
    let elapsed = started.elapsed();
    Ok(BurnE2eStepOutput {
        history: BurnE2eRolloutHistoryEntry {
            step: 0,
            loss: teacher_loss * config.adapter_teacher_weight.max(0.0)
                + flow_loss * config.flow_matching_weight.max(0.0),
            task_loss: 0.0,
            adapter_teacher_loss: teacher_loss,
            flow_matching_loss: flow_loss,
            flow_self_rectification_loss: 0.0,
            amortization_distillation_loss: 0.0,
            amortization_residual_scale: 0.0,
            amortization_residual_rms: 0.0,
            amortization_grad_norm,
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
            examples_seen: batch_len,
            optimizer_examples_per_sec: batch_len as f64
                / elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
            pool_seed_replacements: 0,
            particle_steps_per_sec: particle_steps as f64
                / elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
            dense_pair_interactions_per_sec: particle_steps as f64 * particle_count as f64
                / elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
            elapsed_ms: elapsed.as_secs_f64() * 1_000.0,
            condition_adapter_ms,
            rollout_loss_ms,
            backward_update_ms,
        },
        particle_steps: particle_steps as u64,
        final_x: probe_x,
        final_s: probe_s,
        per_example_losses: None,
    })
}
