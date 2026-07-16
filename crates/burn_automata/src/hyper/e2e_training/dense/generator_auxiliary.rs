//! Condition-level objectives shared by truncated rollout training.

use super::*;

pub(super) struct BurnE2eGeneratorAuxiliary {
    pub(super) objective: Tensor1,
    pub(super) teacher_loss: f32,
    pub(super) flow_matching_loss: f32,
    pub(super) flow_self_rectification_loss: f32,
    pub(super) amortization_distillation_loss: f32,
    pub(super) particle_steps: usize,
}

pub(super) fn functional_adapter_distillation_loss(
    params: &BurnBaseParams,
    probes: Tensor3,
    generated: &BurnAdapterBatch,
    endpoint: &BurnAdapterBatch,
) -> Tensor1 {
    let detached_params = params.detached();
    let generated_update = detached_params.forward_adapter_batch(probes.clone(), generated);
    let endpoint_update = detach3(detached_params.forward_adapter_batch(probes, endpoint));
    let delta = generated_update - endpoint_update;
    delta.clone().mul(delta).mean()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn e2e_generator_auxiliary_objective(
    params: &BurnBaseParams,
    generator: &BurnE2eGeneratorParams,
    npa_config: &NpaConfig,
    conditions: &BurnE2eConditionCache,
    condition_indices: &[usize],
    prepared_dino: Option<&BurnE2ePreparedDinoBatch>,
    targets: &[BurnTargetExample],
    target_indices: &[usize],
    particle_count: usize,
    config: BurnE2eRolloutTrainConfig,
    step_seed: u64,
    x: Tensor3,
    s: Tensor3,
    collect_metrics: bool,
    reused_endpoint: Option<&BurnRowFlowEndpointBridge>,
) -> Result<Option<BurnE2eGeneratorAuxiliary>, Box<dyn std::error::Error>> {
    if config.adapter_teacher_weight <= 0.0
        && config.flow_matching_weight <= 0.0
        && config.flow_self_rectification_weight <= 0.0
        && config.amortization_distillation_weight <= 0.0
    {
        return Ok(None);
    }

    let batch_len = condition_indices.len();
    let device = &targets[target_indices[0]].target_rgb.device();
    let (condition, expansion) = select_rollout_conditions(
        conditions,
        condition_indices,
        prepared_dino,
        config.rollouts_per_example,
    )?;
    let (generated_adapter, generated_rows, prepared_flow_condition) =
        if let Some(endpoint) = reused_endpoint {
            let rows = endpoint.generated_rows();
            (
                BurnAdapterBatch::from_dense_residual_rows(rows.clone(), npa_config),
                Some(rows),
                endpoint.prepared_condition(),
            )
        } else {
            generator.adapter_batch_with_dense_rows(condition, npa_config, config)
        };
    let generated_adapter = generated_adapter.select_rows_or_identity(expansion.as_deref());
    let generator_indices = generator_condition_indices(
        condition_indices,
        expansion.as_deref(),
        config.rollouts_per_example,
    );

    let teacher_vector = (config.adapter_teacher_weight > 0.0)
        .then(|| conditions.select_teacher(condition_indices))
        .flatten();
    let teacher_objective = if config.adapter_teacher_weight > 0.0 {
        let teacher_vector = teacher_vector.clone().ok_or_else(|| {
            AutomataError::InvalidArgument(
                "adapter teacher supervision requires adapter endpoints".to_string(),
            )
        })?;
        let parameter_delta = generated_adapter.to_parameter_vector() - teacher_vector.clone();
        let parameter_mse = parameter_delta.clone().mul(parameter_delta).mean();
        if config.adapter_teacher_objective == E2eAdapterTeacherObjective::ParameterMse {
            Some(parameter_mse)
        } else {
            let teacher_adapter = BurnAdapterBatch::from_parameter_vector(
                teacher_vector,
                npa_config,
                config.adapter_rank,
                config.adapter_alpha,
            );
            let probe_steps = if config.adapter_teacher_probe_rollout_steps == 0 {
                0
            } else {
                1 + step_seed as usize % config.adapter_teacher_probe_rollout_steps
            };
            let (probe_x, probe_s) = if probe_steps == 0 {
                (detach3(x.clone()), detach3(s.clone()))
            } else {
                let mut teacher_rng =
                    StdRng::seed_from_u64(step_seed ^ 0x7465_6163_6865_7270);
                let (probe_x, probe_s, _) = rollout_batch_chunk(
                    &params.detached(),
                    &teacher_adapter,
                    targets,
                    target_indices,
                    detach3(x.clone()),
                    detach3(s.clone()),
                    direct_config_view(config),
                    particle_count,
                    &mut teacher_rng,
                    probe_steps,
                    Tensor::<BurnBackend, 1>::zeros([batch_len], device),
                    None,
                );
                (detach3(probe_x), detach3(probe_s))
            };
            let probes = rollout_dense_perception_batch(
                &probe_x,
                &probe_s,
                direct_config_view(config),
            );
            let generated_update =
                params.forward_adapter_batch(probes.clone(), &generated_adapter);
            let teacher_update = detach3(
                params
                    .detached()
                    .forward_adapter_batch(probes, &teacher_adapter),
            );
            let functional_delta = generated_update - teacher_update;
            let functional_mse = functional_delta.clone().mul(functional_delta).mean();
            Some(if config.adapter_teacher_objective == E2eAdapterTeacherObjective::Hybrid {
                functional_mse + parameter_mse.mul_scalar(FUNCTIONAL_TEACHER_PARAMETER_AUX_WEIGHT)
            } else {
                functional_mse
            })
        }
    } else {
        None
    };

    let flow = generator.row_flow.as_ref();
    let flow_matching_objective = if config.flow_matching_weight > 0.0 {
        let teacher = conditions.select_teacher(&generator_indices).ok_or_else(|| {
            AutomataError::InvalidArgument(
                "flow matching requires adapter endpoints".to_string(),
            )
        })?;
        Some(
            flow.ok_or_else(|| {
                AutomataError::InvalidArgument(
                    "flow matching requires conditional-row-flow".to_string(),
                )
            })?
            .flow_matching_loss_prepared(
                prepared_flow_condition
                    .as_ref()
                    .expect("row flow prepared a condition"),
                teacher,
                npa_config,
                config.adapter_rank,
                config.adapter_alpha,
                config.flow_match_inference_source,
            ),
        )
    } else {
        None
    };
    let amortization_endpoint = generator.amortization_residual_rows(&generator_indices);
    let self_rectification_objective = if config.flow_self_rectification_weight > 0.0 {
        Some(
            flow.ok_or_else(|| {
                AutomataError::InvalidArgument(
                    "flow self-rectification requires conditional-row-flow".to_string(),
                )
            })?
            .self_rectification_loss_to_endpoint_prepared(
                prepared_flow_condition
                    .as_ref()
                    .expect("row flow prepared a condition"),
                amortization_endpoint
                    .as_ref()
                    .or(generated_rows.as_ref())
                    .expect("row flow generated or amortized dense endpoint rows")
                    .clone(),
                npa_config,
                step_seed ^ 0x7365_6c66_7265_6374,
            ),
        )
    } else {
        None
    };
    let amortization_probe_steps = if config.amortization_distillation_weight > 0.0
        && config.amortization_distillation_objective
            != E2eAdapterTeacherObjective::ParameterMse
        && config.amortization_distillation_probe_rollout_steps > 0
    {
        1 + step_seed as usize % config.amortization_distillation_probe_rollout_steps
    } else {
        0
    };
    let amortization_distillation_objective = if config.amortization_distillation_weight > 0.0 {
        let parameter_mse = flow
            .ok_or_else(|| {
                AutomataError::InvalidArgument(
                    "amortization distillation requires conditional-row-flow".to_string(),
                )
            })?
            .amortization_distillation_loss(
                generated_rows
                    .as_ref()
                    .expect("amortization flow generated dense endpoint rows")
                    .clone(),
                amortization_endpoint
                    .as_ref()
                    .expect("amortization distillation has endpoint rows")
                    .clone(),
                npa_config,
            );
        if config.amortization_distillation_objective
            == E2eAdapterTeacherObjective::ParameterMse
        {
            Some(parameter_mse)
        } else {
            let endpoint_adapter = BurnAdapterBatch::from_dense_residual_rows(
                generator
                    .amortization_residual_rows(condition_indices)
                    .expect("functional amortization distillation has endpoint rows"),
                npa_config,
            );
            let (probe_x, probe_s) = if amortization_probe_steps == 0 {
                (detach3(x.clone()), detach3(s.clone()))
            } else {
                let mut probe_rng =
                    StdRng::seed_from_u64(step_seed ^ 0x616d_6f72_745f_6675);
                let (probe_x, probe_s, _) = rollout_batch_chunk(
                    &params.detached(),
                    &endpoint_adapter,
                    targets,
                    target_indices,
                    detach3(x.clone()),
                    detach3(s.clone()),
                    direct_config_view(config),
                    particle_count,
                    &mut probe_rng,
                    amortization_probe_steps,
                    Tensor::<BurnBackend, 1>::zeros([batch_len], device),
                    None,
                );
                (detach3(probe_x), detach3(probe_s))
            };
            let probes = rollout_dense_perception_batch(
                &probe_x,
                &probe_s,
                direct_config_view(config),
            );
            let functional_mse = functional_adapter_distillation_loss(
                params,
                probes,
                &generated_adapter,
                &endpoint_adapter,
            );
            Some(
                if config.amortization_distillation_objective
                    == E2eAdapterTeacherObjective::Hybrid
                {
                    functional_mse
                        + parameter_mse.mul_scalar(FUNCTIONAL_TEACHER_PARAMETER_AUX_WEIGHT)
                } else {
                    functional_mse
                },
            )
        }
    } else {
        None
    };

    let teacher_loss = if collect_metrics {
        teacher_objective
            .as_ref()
            .map(|loss| loss.clone().inner().into_scalar())
            .unwrap_or_default()
    } else {
        0.0
    };
    let flow_matching_loss = if collect_metrics {
        flow_matching_objective
            .as_ref()
            .map(|loss| loss.clone().inner().into_scalar())
            .unwrap_or_default()
    } else {
        0.0
    };
    let flow_self_rectification_loss = if collect_metrics {
        self_rectification_objective
            .as_ref()
            .map(|loss| loss.clone().inner().into_scalar())
            .unwrap_or_default()
    } else {
        0.0
    };
    let amortization_distillation_loss = if collect_metrics {
        amortization_distillation_objective
            .as_ref()
            .map(|loss| loss.clone().inner().into_scalar())
            .unwrap_or_default()
    } else {
        0.0
    };

    let objective = teacher_objective
        .map(|loss| loss.mul_scalar(config.adapter_teacher_weight.max(0.0)))
        .into_iter()
        .chain(
            flow_matching_objective
                .map(|loss| loss.mul_scalar(config.flow_matching_weight.max(0.0))),
        )
        .chain(
            self_rectification_objective
                .map(|loss| loss.mul_scalar(config.flow_self_rectification_weight.max(0.0))),
        )
        .chain(
            amortization_distillation_objective.map(|loss| {
                loss.mul_scalar(config.amortization_distillation_weight.max(0.0))
            }),
        )
        .reduce(|left, right| left + right)
        .expect("positive auxiliary weight creates an objective");

    Ok(Some(BurnE2eGeneratorAuxiliary {
        objective,
        teacher_loss,
        flow_matching_loss,
        flow_self_rectification_loss,
        amortization_distillation_loss,
        particle_steps: (if config.adapter_teacher_weight > 0.0
            && config.adapter_teacher_objective != E2eAdapterTeacherObjective::ParameterMse
        {
            batch_len
                * particle_count
                * if config.adapter_teacher_probe_rollout_steps == 0 {
                    0
                } else {
                    1 + step_seed as usize % config.adapter_teacher_probe_rollout_steps
                }
        } else {
            0
        }) + batch_len * particle_count * amortization_probe_steps,
    }))
}
