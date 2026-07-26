//! Recurrent Target2D training on conserved adaptive material.
//!
//! This path trains the shared NPA rule directly through adaptive perception
//! and variable-footprint rendering. It intentionally excludes teacher
//! closure, hidden fine rows, and render-only fine reconstruction.

use super::*;

struct BurnAdaptiveMaterial {
    represented_measure: Tensor2Inner,
    bandwidth: Tensor2Inner,
    scale_feature: Tensor2Inner,
    footprint_ratio: Tensor2Inner,
    residual_gate: Tensor2Inner,
    reference_particle_count: usize,
    active_particle_count: usize,
}

impl BurnAdaptiveMaterial {
    fn new(
        config: &AdaptiveTarget2dBurnConfig,
        device: &BurnDevice,
    ) -> AutomataResult<Self> {
        let active_particle_count = config.material.active_particle_count();
        let reference_particle_count = config.material.reference_particle_count;
        if active_particle_count == 0 || reference_particle_count < active_particle_count {
            return Err(AutomataError::InvalidArgument(
                "adaptive Target2D material budget is invalid".to_string(),
            ));
        }
        let inner_device = Device::<InnerBackend>::from(device.clone());
        Ok(Self {
            represented_measure: Tensor::<InnerBackend, 2>::from_data(
                TensorData::new(
                    config.material.represented_measure.clone(),
                    [1, active_particle_count],
                ),
                &inner_device,
            ),
            bandwidth: Tensor::<InnerBackend, 2>::from_data(
                TensorData::new(
                    config.material.bandwidth.clone(),
                    [1, active_particle_count],
                ),
                &inner_device,
            ),
            scale_feature: Tensor::<InnerBackend, 2>::from_data(
                TensorData::new(
                    config
                        .material
                        .footprint_ratio
                        .iter()
                        .map(|ratio| (ratio - 1.0).clamp(-0.75, 3.0))
                        .collect::<Vec<_>>(),
                    [1, active_particle_count],
                ),
                &inner_device,
            ),
            footprint_ratio: Tensor::<InnerBackend, 2>::from_data(
                TensorData::new(
                    config.material.footprint_ratio.clone(),
                    [1, active_particle_count],
                ),
                &inner_device,
            ),
            residual_gate: Tensor::<InnerBackend, 2>::from_data(
                TensorData::new(
                    if config.residual_perception_semantics
                        == Some(
                            burn_automata_kernels::AdaptivePerceptionSemantics::NormalizedAdaptive,
                        )
                    {
                        config
                            .material
                            .footprint_ratio
                            .iter()
                            .map(|ratio| {
                                (ratio.max(f32::MIN_POSITIVE).ln() / std::f32::consts::LN_2)
                                    .clamp(0.0, 3.0)
                            })
                            .collect()
                    } else {
                        vec![
                            f32::from(
                                config
                                    .material
                                    .footprint_ratio
                                    .iter()
                                    .any(|ratio| *ratio > 1.0 + 32.0 * f32::EPSILON)
                            );
                            active_particle_count
                        ]
                    },
                    [1, active_particle_count],
                ),
                &inner_device,
            ),
            reference_particle_count,
            active_particle_count,
        })
    }

    fn batch(
        &self,
        targets: &[BurnTargetExample],
        indices: &[usize],
    ) -> (Tensor2, Tensor2, Tensor2, Tensor2, Tensor2, Tensor2) {
        let batches = indices.len();
        let shape = [batches, self.active_particle_count];
        let represented_measure =
            Tensor::<BurnBackend, 2>::from_inner(self.represented_measure.clone()).expand(shape);
        let bandwidth =
            Tensor::<BurnBackend, 2>::from_inner(self.bandwidth.clone()).expand(shape);
        let scale_feature =
            Tensor::<BurnBackend, 2>::from_inner(self.scale_feature.clone()).expand(shape);
        let footprint_ratio =
            Tensor::<BurnBackend, 2>::from_inner(self.footprint_ratio.clone()).expand(shape);
        let residual_gate =
            Tensor::<BurnBackend, 2>::from_inner(self.residual_gate.clone()).expand(shape);
        let pixel_size = stack_pixel_sizes(targets, indices)
            .reshape([batches, 1])
            .expand(shape)
            .mul(footprint_ratio);
        let output_scale = stack_target_point_counts(targets, indices)
            .reshape([batches, 1])
            .div_scalar(self.reference_particle_count as f32)
            .expand(shape);
        (
            represented_measure,
            bandwidth,
            scale_feature,
            pixel_size,
            output_scale,
            residual_gate,
        )
    }
}

struct AdaptiveTrainStepStats {
    metrics_collected: bool,
    loss: f32,
    optimization_loss: f32,
    max_trajectory_loss: f32,
    grad_norm: f32,
    grad_scale: f32,
    backward_scale: f32,
    topology_events: usize,
    event_exposed_rows: usize,
    unique_event_exposed_rows: usize,
    event_quota_shortfall_rows: usize,
    max_event_delay_steps: usize,
    post_event_row_chunks: usize,
    post_event_mean_loss: Option<f32>,
    recovery_extension_steps: usize,
    sampled_age_min: usize,
    sampled_age_max: usize,
    sampled_age_mean: f64,
    particle_steps_per_sec: f64,
    particle_steps: f64,
    elapsed_ms: f64,
}

struct BurnAdaptiveEvalSeed {
    positions: Tensor3,
    states: Tensor3,
    update_masks: Vec<crate::adaptive::AdaptiveTarget2dUpdateMask>,
    seeds: Vec<u64>,
}

struct AdaptiveFreshSeedEval {
    mean_loss: f32,
    mean_render_rgb_mse: f32,
    selection_psnr_db: f32,
    worst_psnr_drift_db: f32,
    selection_score: f32,
    horizon_mean_render_rgb_mse: Vec<f32>,
    horizon_mean_psnr_db: Vec<f32>,
    topology_events: usize,
}

pub(super) fn adaptive_checkpoint_is_better(candidate_psnr: f32, best_psnr: Option<f32>) -> bool {
    candidate_psnr.is_finite() && best_psnr.is_none_or(|best| candidate_psnr > best)
}

pub(super) fn validate_adaptive_target2d_primal_parity(
    kernel_splat: &[f32],
    dense_splat: &[f32],
    horizon: usize,
) -> AutomataResult<()> {
    if kernel_splat.len() != dense_splat.len() {
        return Err(AutomataError::InvalidArgument(format!(
            "adaptive Target2D primal parity length mismatch at horizon {horizon}: kernel={} dense={}",
            kernel_splat.len(),
            dense_splat.len(),
        )));
    }
    for (row, (&kernel, &dense)) in kernel_splat.iter().zip(dense_splat).enumerate() {
        if !kernel.is_finite() || !dense.is_finite() {
            return Err(AutomataError::InvalidArgument(format!(
                "adaptive Target2D non-finite primal at horizon {horizon} row {row}: kernel={kernel} dense={dense}",
            )));
        }
        let tolerance = 5.0e-4 + 0.05 * dense.abs();
        if (kernel - dense).abs() > tolerance {
            return Err(AutomataError::InvalidArgument(format!(
                "adaptive Target2D primal parity failed at horizon {horizon} row {row}: kernel={kernel} dense={dense} tolerance={tolerance}",
            )));
        }
    }
    Ok(())
}

pub(super) fn adaptive_backward_scale(configured: f32, normalize_gradients: bool) -> f32 {
    if normalize_gradients { configured } else { 1.0 }
}

pub(super) fn adaptive_trajectory_tail_count(batch_size: usize, fraction: f32) -> usize {
    if batch_size == 0 || !fraction.is_finite() || fraction <= 0.0 {
        0
    } else {
        ((batch_size as f32 * fraction.clamp(0.0, 1.0)).ceil() as usize)
            .clamp(1, batch_size)
    }
}

fn validate_adaptive_target2d_resume(
    state: &Target2dTrainingCheckpoint,
    checkpoint: &Target2dBurnCheckpointConfig,
    params: &BurnBaseBatch,
    config: DirectBasisTrainConfig,
    total_steps: usize,
) -> AutomataResult<()> {
    let expected_model_sha256 = checkpoint.resume_model_sha256.as_deref().ok_or_else(|| {
        AutomataError::InvalidArgument(
            "Target2D resume requires the loaded model payload hash".to_owned(),
        )
    })?;
    if state.backend != BACKEND
        || state.model_sha256 != expected_model_sha256
        || state.model_count != params.model_count()
        || (!checkpoint.curriculum_resume
            && (state.rollout_particles != config.rollout_particles
                || state.completed_step > total_steps))
    {
        return Err(AutomataError::InvalidArgument(format!(
            "Target2D training state is incompatible: backend={}/{BACKEND} model_sha256={}/{} model_count={}/{} particles={}/{} completed_step={}/{} optimizer_step={}",
            state.backend,
            state.model_sha256,
            expected_model_sha256,
            state.model_count,
            params.model_count(),
            state.rollout_particles,
            config.rollout_particles,
            state.completed_step,
            total_steps,
            state.optimizer_step,
        )));
    }
    if !checkpoint.curriculum_resume
        && checkpoint.include_particle_pool
        && state.particle_pool.is_none()
    {
        return Err(AutomataError::InvalidArgument(
            "Target2D resume requested an exact particle pool, but the sidecar has none"
                .to_owned(),
        ));
    }
    Ok(())
}

fn write_adaptive_target2d_training_state(
    checkpoint: &Target2dBurnCheckpointConfig,
    checkpoint_state: &BurnDenseCheckpointState<'_>,
    optimizer: &BurnBaseBatchAdamWState,
    pool: &BurnDeviceParticlePool,
    params: &BurnBaseBatch,
    config: DirectBasisTrainConfig,
    completed_step: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let Some(output) = checkpoint.training_state_output.as_ref() else {
        return Ok(());
    };
    let model_sha256 = checkpoint_state.last_model_sha256().ok_or_else(|| {
        AutomataError::InvalidArgument(
            "Target2D state cannot be written before its paired model checkpoint".to_owned(),
        )
    })?;
    let state = Target2dTrainingCheckpoint {
        version: TARGET2D_TRAINING_CHECKPOINT_VERSION,
        backend: BACKEND.to_owned(),
        model_sha256: model_sha256.to_owned(),
        completed_step,
        optimizer_step: optimizer.step,
        model_count: params.model_count(),
        rollout_particles: config.rollout_particles,
        optimizer_tensors: optimizer.target2d_snapshots()?,
        particle_pool: checkpoint
            .include_particle_pool
            .then(|| pool.target2d_snapshot())
            .transpose()?,
    };
    state.write_atomic(output)?;
    println!(
        "{LOG_BACKEND} adaptive-target2d training-state step={completed_step} optimizer_step={} pool={} state={}",
        optimizer.step,
        checkpoint.include_particle_pool,
        output.display(),
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn train_adaptive_step_tbptt(
    params: &mut BurnBaseBatch,
    frozen_base: Option<&BurnBaseBatch>,
    optimizer: &mut BurnBaseBatchAdamWState,
    target: &BurnTargetExample,
    material: &BurnAdaptiveMaterial,
    topology: &BurnAdaptiveTopology,
    adaptive: &AdaptiveTarget2dBurnConfig,
    config: DirectBasisTrainConfig,
    step_seed: u64,
    pool: &mut BurnDeviceParticlePool,
    replace_pool_seed: bool,
    collect_metrics: bool,
    optimizer_config: AdamWConfig,
) -> Result<AdaptiveTrainStepStats, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let device = &target.target_rgb.device();
    let trajectories = config.example_batch_size.max(1);
    let indices = vec![0usize; trajectories];
    let mut rng = StdRng::seed_from_u64(step_seed ^ 0xada2_7a2d);
    let chunk_steps = tbptt_chunk_steps(config);
    let rollout_steps = sampled_training_rollout_steps(config, step_seed);
    let event_preference = adaptive.event_training.enabled.then_some(
        BurnPoolEventPreference {
            start_step: adaptive.topology.start_step,
            end_step: adaptive.topology.end_step,
            interval_steps: adaptive.topology.interval_steps,
            lookahead_steps: rollout_steps,
            min_rows: adaptive
                .event_training
                .min_event_trajectories_per_batch,
        },
    );
    let pool_batch = pool.sample_batch_with_fresh_rows(
        &mut rng,
        trajectories,
        BurnPoolSampling {
            fresh_seed_rows: usize::from(replace_pool_seed)
                * adaptive.fresh_seed_trajectories,
            max_age_steps: (adaptive.max_pool_age_steps > 0)
                .then_some(adaptive.max_pool_age_steps),
            age_strata: adaptive.pool_age_strata,
            event_preference,
        },
        config,
        device,
    )?;
    let actual_trajectories = pool_batch.pool_indices.len();
    let indices = &indices[..actual_trajectories];
    let mut rngs = (0..actual_trajectories)
        .map(|row| {
            StdRng::seed_from_u64(
                step_seed.wrapping_add((row as u64).wrapping_mul(0x9e37_79b9))
                    ^ 0xada2_5eed,
            )
        })
        .collect::<Vec<_>>();
    let (
        represented_measure,
        bandwidth,
        scale_feature,
        pixel_size,
        output_scale,
        residual_gate,
    ) =
        material.batch(std::slice::from_ref(target), indices);
    let mut x = pool_batch.x;
    let mut s = pool_batch.s;
    let mut trajectory_ages = pool_batch.ages;
    let sampled_age_min = trajectory_ages.iter().copied().min().unwrap_or(0);
    let sampled_age_max = trajectory_ages.iter().copied().max().unwrap_or(0);
    let sampled_age_mean = trajectory_ages.iter().copied().sum::<usize>() as f64
        / trajectory_ages.len().max(1) as f64;
    let mut displacement = Tensor::<BurnBackend, 1>::zeros([actual_trajectories], device);
    let mut remaining_steps = rollout_steps;
    let mut loss_sum = 0.0_f32;
    let mut optimization_loss_sum = 0.0_f32;
    let mut max_trajectory_loss = 0.0_f32;
    let mut loss_chunks = 0usize;
    let mut backward_scale_sum = 0.0_f32;
    let mut accumulated_gradients = None::<Vec<Tensor3Inner>>;
    let mut accumulated_gradient_chunks = 0usize;
    let mut particle_steps = 0.0_f64;
    let mut topology_events = 0usize;
    let mut post_event_remaining = vec![0usize; actual_trajectories];
    let mut pre_event_reference_loss =
        Tensor::<BurnBackend, 1>::zeros([actual_trajectories], device);
    let mut event_exposed = vec![false; actual_trajectories];
    let mut event_exposed_rows = 0usize;
    let mut max_event_delay_steps = 0usize;
    let mut post_event_row_chunks = 0usize;
    let mut post_event_loss_sum = 0.0_f64;
    let mut post_event_loss_count = 0usize;
    let mut recovery_extension_budget = adaptive
        .event_training
        .recovery_extension_budget();
    let mut recovery_extension_steps = 0usize;

    while remaining_steps > 0 {
        let post_event_rows = post_event_remaining
            .iter()
            .map(|remaining| *remaining > 0)
            .collect::<Vec<_>>();
        let active_post_event_rows = post_event_rows.iter().filter(|active| **active).count();
        let nominal_steps = tbptt_next_chunk_steps(
            remaining_steps,
            chunk_steps,
            config.loss_on_final_chunk_only,
        );
        let steps = topology.steps_until_next_event(&trajectory_ages, nominal_steps);
        let final_chunk = steps == remaining_steps;
        let optimize_chunk =
            active_post_event_rows > 0 || !config.loss_on_final_chunk_only || final_chunk;
        let (mut next_x, mut next_s, mut next_displacement, mut detail) = if optimize_chunk {
            rollout_adaptive_oracle_model_batch_chunk(
                params,
                frozen_base,
                std::slice::from_ref(target),
                indices,
                x,
                s,
                represented_measure.clone(),
                bandwidth.clone(),
                scale_feature.clone(),
                residual_gate.clone(),
                adaptive.perception,
                adaptive.perception_options,
                adaptive.perception_semantics,
                adaptive.residual_perception_semantics,
                adaptive.material_scale_conditioning,
                adaptive.compatible_residual_material_features,
                config,
                material.active_particle_count,
                &mut rngs,
                false,
                None,
                None,
                0,
                steps,
                displacement,
            )
        } else {
            let detached = params.detached();
            rollout_adaptive_oracle_model_batch_chunk(
                &detached,
                frozen_base,
                std::slice::from_ref(target),
                indices,
                x,
                s,
                represented_measure.clone(),
                bandwidth.clone(),
                scale_feature.clone(),
                residual_gate.clone(),
                adaptive.perception,
                adaptive.perception_options,
                adaptive.perception_semantics,
                adaptive.residual_perception_semantics,
                adaptive.material_scale_conditioning,
                adaptive.compatible_residual_material_features,
                config,
                material.active_particle_count,
                &mut rngs,
                false,
                None,
                None,
                0,
                steps,
                displacement,
            )
        };
        let restored = pool.restore_unhealthy_batch(
            &pool_batch.pool_indices,
            next_x,
            next_s,
        )?;
        next_x = restored.0;
        next_s = restored.1;
        let unhealthy_rows = restored.2;
        next_displacement = next_displacement
            .clone()
            .mask_fill(next_displacement.is_finite().bool_not(), 0.0)
            .clamp(0.0, 1.0e6)
            .mask_fill(unhealthy_rows.clone(), 0.0);
        let detail_dims = detail.shape().dims::<2>();
        detail = detail
            .clone()
            .mask_fill(detail.is_finite().bool_not(), 0.0)
            .clamp(0.0, 1.0e6)
            .mask_fill(
                unhealthy_rows
                    .reshape([actual_trajectories, 1])
                    .expand(detail_dims),
                0.0,
            );
        let next_trajectory_ages = trajectory_ages
            .iter()
            .map(|age| age.saturating_add(steps))
            .collect::<Vec<_>>();
        let scheduled_event_rows =
            topology.scheduled_event_rows(&trajectory_ages, &next_trajectory_ages);
        let event_reference_loss = (adaptive
            .event_training
            .post_event_degradation_weight
            > 0.0
            && scheduled_event_rows
                .iter()
                .any(|scheduled| *scheduled))
        .then(|| {
                adaptive_target_splat_loss_batch_vector_base_only_selected(
                    &detach3(next_x.clone()),
                    &detach3(next_s.clone()),
                    std::slice::from_ref(target),
                    indices,
                    config,
                    represented_measure.clone(),
                    pixel_size.clone(),
                    output_scale.clone(),
                    detach1(next_displacement.clone()),
                )
                .map(|loss| detach1(loss.total))
            })
            .transpose()?;
        let event_delays =
            topology.scheduled_event_delay_steps(&trajectory_ages, &next_trajectory_ages);
        let chunk_max_event_delay = event_delays.iter().flatten().copied().max().unwrap_or(0);
        debug_assert_eq!(
            chunk_max_event_delay, 0,
            "adaptive topology events must land on exact rollout boundaries"
        );
        max_event_delay_steps = max_event_delay_steps.max(chunk_max_event_delay);
        // A recovery chunk that lands on the next topology boundary must be
        // scored before that next detached exchange. The persistent state
        // still receives the exchange below, but its loss should measure the
        // differentiable dynamics between events rather than two stacked
        // discontinuities.
        let pre_topology_recovery_state =
            (active_post_event_rows > 0).then(|| (next_x.clone(), next_s.clone()));
        let applied = topology.apply_scheduled(
            next_x,
            next_s,
            detach2(detail),
            &trajectory_ages,
            &next_trajectory_ages,
        );
        next_x = applied.0;
        next_s = applied.1;
        topology_events += applied.2;
        for (row, scheduled) in scheduled_event_rows.iter().copied().enumerate() {
            if scheduled {
                event_exposed_rows += 1;
                event_exposed[row] = true;
            }
        }
        trajectory_ages = next_trajectory_ages;
        if optimize_chunk {
            let (loss_x, loss_s) = pre_topology_recovery_state
                .as_ref()
                .map_or((&next_x, &next_s), |(x, s)| (x, s));
            let loss = adaptive_target_splat_loss_batch_vector_base_only_selected(
                loss_x,
                loss_s,
                std::slice::from_ref(target),
                indices,
                config,
                represented_measure.clone(),
                pixel_size.clone(),
                output_scale.clone(),
                next_displacement.clone(),
            )?;
            let event_loss_weight = if adaptive.event_training.enabled {
                adaptive.event_training.post_event_loss_weight
            } else {
                0.0
            };
            let mut scalar = 0.0_f32;
            let mut optimization_scalar = 0.0_f32;
            if collect_metrics {
                let loss_values = tensor1_vec_async(loss.total.clone().inner()).await?;
                let finite_losses = loss_values
                    .iter()
                    .copied()
                    .filter(|value| value.is_finite())
                    .collect::<Vec<_>>();
                scalar = if finite_losses.is_empty() {
                    1.0e6
                } else {
                    finite_losses.iter().sum::<f32>() / finite_losses.len() as f32
                };
                let finite_optimization_losses = loss_values
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(_, value)| value.is_finite())
                    .map(|(row, value)| {
                        let transformed = if adaptive.log1p_trajectory_loss {
                            value.max(0.0).ln_1p()
                        } else {
                            value
                        };
                        (row, transformed)
                    })
                    .collect::<Vec<_>>();
                let optimization_weight_sum = finite_optimization_losses
                    .iter()
                    .map(|(row, _)| {
                        1.0 + event_loss_weight * f32::from(post_event_rows[*row])
                    })
                    .sum::<f32>();
                optimization_scalar = if finite_optimization_losses.is_empty() {
                    if adaptive.log1p_trajectory_loss {
                        1.0e6_f32.ln_1p()
                    } else {
                        1.0e6
                    }
                } else {
                    finite_optimization_losses
                        .iter()
                        .map(|(row, value)| {
                            value * (1.0 + event_loss_weight * f32::from(post_event_rows[*row]))
                        })
                        .sum::<f32>()
                        / optimization_weight_sum.max(f32::MIN_POSITIVE)
                };
                let tail_count = adaptive_trajectory_tail_count(
                    finite_optimization_losses.len(),
                    adaptive.trajectory_tail_fraction,
                );
                if tail_count > 0 && adaptive.trajectory_tail_weight > 0.0 {
                    let mut ordered = finite_optimization_losses
                        .iter()
                        .map(|(_, value)| *value)
                        .collect::<Vec<_>>();
                    ordered.sort_by(|lhs, rhs| rhs.total_cmp(lhs));
                    let tail_mean =
                        ordered[..tail_count].iter().sum::<f32>() / tail_count as f32;
                    optimization_scalar =
                        (optimization_scalar + adaptive.trajectory_tail_weight * tail_mean)
                            / (1.0 + adaptive.trajectory_tail_weight);
                }
                max_trajectory_loss = max_trajectory_loss.max(
                    finite_losses
                        .iter()
                        .copied()
                        .fold(0.0_f32, f32::max),
                );
                if active_post_event_rows > 0 {
                    for (row, value) in loss_values.iter().copied().enumerate() {
                        if post_event_rows[row] && value.is_finite() {
                            post_event_loss_sum += f64::from(value);
                            post_event_loss_count += 1;
                        }
                    }
                }
            }
            if active_post_event_rows > 0 {
                post_event_row_chunks += active_post_event_rows;
            }
            // Per-parameter normalization makes the optimizer invariant to a
            // common positive gradient scale. Normalize the linear-adjoint
            // objective before recurrent backpropagation so a long Jacobian
            // cannot overflow to Inf before the optimizer sees it.
            let backward_scale = adaptive_backward_scale(
                adaptive.backward_loss_scale,
                config.per_parameter_grad_normalization,
            );
            let safe_total = loss
                .total
                .clone()
                .mask_fill(loss.total.is_finite().bool_not(), 0.0);
            let optimization_total = if adaptive.log1p_trajectory_loss {
                safe_total.clamp_min(0.0).log1p()
            } else {
                safe_total
            };
            let tail_count = adaptive_trajectory_tail_count(
                actual_trajectories,
                adaptive.trajectory_tail_fraction,
            );
            let base_objective = if event_loss_weight > 0.0 && active_post_event_rows > 0 {
                let row_weights = post_event_rows
                    .iter()
                    .map(|active| 1.0 + event_loss_weight * f32::from(*active))
                    .collect::<Vec<_>>();
                let row_weight_sum = row_weights.iter().sum::<f32>();
                let row_weights = Tensor::<BurnBackend, 1>::from_data(
                    TensorData::new(row_weights, [actual_trajectories]),
                    device,
                );
                optimization_total
                    .clone()
                    .mul(row_weights)
                    .sum()
                    .div_scalar(row_weight_sum.max(f32::MIN_POSITIVE))
            } else {
                optimization_total.clone().mean()
            };
            let objective = if tail_count > 0 && adaptive.trajectory_tail_weight > 0.0 {
                let tail_indices = device_topk_indices(
                    optimization_total
                        .clone()
                        .reshape([1, actual_trajectories]),
                    tail_count,
                )
                .reshape([tail_count]);
                let tail_mean = optimization_total.clone().select(0, tail_indices).mean();
                (base_objective + tail_mean.mul_scalar(adaptive.trajectory_tail_weight))
                    .div_scalar(1.0 + adaptive.trajectory_tail_weight)
            } else {
                base_objective
            };
            let objective = if active_post_event_rows > 0
                && adaptive.event_training.post_event_degradation_weight > 0.0
            {
                let post_event_mask = Tensor::<BurnBackend, 1>::from_data(
                    TensorData::new(
                        post_event_rows.iter().copied().map(f32::from).collect(),
                        [actual_trajectories],
                    ),
                    device,
                );
                let safe_reference = pre_event_reference_loss
                    .clone()
                    .mask_fill(
                        pre_event_reference_loss
                            .clone()
                            .is_finite()
                            .bool_not(),
                        0.0,
                    );
                let optimization_reference = if adaptive.log1p_trajectory_loss {
                    safe_reference.clamp_min(0.0).log1p()
                } else {
                    safe_reference
                };
                let degradation = (optimization_total - optimization_reference)
                    .clamp_min(0.0)
                    .mul(post_event_mask)
                    .sum()
                    .div_scalar(active_post_event_rows as f32);
                objective
                    + degradation
                        .mul_scalar(adaptive.event_training.post_event_degradation_weight)
            } else {
                objective
            };
            let mut gradients = objective.mul_scalar(backward_scale).backward();
            let chunk_gradients = params.take_gradients(&mut gradients);
            if let Some(accumulated) = accumulated_gradients.as_mut() {
                for (total, chunk) in accumulated.iter_mut().zip(chunk_gradients) {
                    *total = total.clone() + chunk;
                }
            } else {
                accumulated_gradients = Some(chunk_gradients);
            }
            accumulated_gradient_chunks += 1;
            loss_sum += scalar;
            optimization_loss_sum += optimization_scalar;
            loss_chunks += 1;
            backward_scale_sum += backward_scale;
        }
        if let Some(event_reference_loss) = event_reference_loss {
            let event_mask = Tensor::<BurnBackend, 1>::from_data(
                TensorData::new(
                    scheduled_event_rows
                        .iter()
                        .copied()
                        .map(f32::from)
                        .collect(),
                    [actual_trajectories],
                ),
                device,
            );
            pre_event_reference_loss = detach1(
                pre_event_reference_loss
                    .mul(event_mask.clone().neg().add_scalar(1.0))
                    + event_reference_loss.mul(event_mask),
            );
        }
        x = detach3(next_x);
        s = detach3(next_s);
        displacement = detach1(next_displacement);
        particle_steps += (actual_trajectories * material.active_particle_count * steps) as f64;
        for remaining in &mut post_event_remaining {
            *remaining = remaining.saturating_sub(steps);
        }
        remaining_steps -= steps;
        if adaptive.event_training.enabled {
            for (row, scheduled) in scheduled_event_rows.iter().copied().enumerate() {
                if scheduled {
                    post_event_remaining[row] =
                        adaptive.event_training.post_event_recovery_steps;
                }
            }
            if scheduled_event_rows.iter().any(|scheduled| *scheduled) {
                let required = adaptive
                    .event_training
                    .post_event_recovery_steps
                    .saturating_sub(remaining_steps);
                let extension = required.min(recovery_extension_budget);
                remaining_steps = remaining_steps.saturating_add(extension);
                recovery_extension_budget =
                    recovery_extension_budget.saturating_sub(extension);
                recovery_extension_steps =
                    recovery_extension_steps.saturating_add(extension);
            }
        }
    }

    let mut accumulated_gradients = accumulated_gradients.ok_or_else(|| {
        AutomataError::InvalidArgument(
            "adaptive Target2D produced no differentiable TBPTT objective".to_owned(),
        )
    })?;
    let gradient_average = 1.0 / accumulated_gradient_chunks.max(1) as f32;
    for gradient in &mut accumulated_gradients {
        *gradient = gradient.clone().mul_scalar(gradient_average);
    }
    // Event boundaries may partition one sampled trajectory into many graph
    // segments. Apply one update to keep optimizer cadence independent of pool
    // ages, topology timing, and TBPTT partitioning.
    let (grad_norms, grad_scales) = if adaptive.optimize_material_scale_only {
        params
            .apply_adamw_last_input_column_gradients_async(
                accumulated_gradients,
                optimizer,
                optimizer_config,
                config.per_parameter_grad_normalization,
                collect_metrics,
            )
            .await?
    } else {
        params
            .apply_adamw_gradients_async(
                accumulated_gradients,
                optimizer,
                optimizer_config,
                config.per_parameter_grad_normalization,
                collect_metrics,
            )
            .await?
    };
    pool.update_batch_with_ages(&pool_batch.pool_indices, &trajectory_ages, x, s)?;
    let elapsed = started.elapsed();
    let loss_chunks = loss_chunks.max(1);
    let unique_event_exposed_rows = event_exposed.iter().filter(|exposed| **exposed).count();
    let event_quota_shortfall_rows = if adaptive.event_training.enabled {
        adaptive
            .event_training
            .min_event_trajectories_per_batch
            .min(actual_trajectories)
            .saturating_sub(unique_event_exposed_rows)
    } else {
        0
    };
    Ok(AdaptiveTrainStepStats {
        metrics_collected: collect_metrics,
        loss: loss_sum / loss_chunks as f32,
        optimization_loss: optimization_loss_sum / loss_chunks as f32,
        max_trajectory_loss,
        grad_norm: grad_norms.first().copied().unwrap_or(0.0),
        grad_scale: grad_scales.first().copied().unwrap_or(1.0),
        backward_scale: backward_scale_sum / loss_chunks as f32,
        topology_events,
        event_exposed_rows,
        unique_event_exposed_rows,
        event_quota_shortfall_rows,
        max_event_delay_steps,
        post_event_row_chunks,
        post_event_mean_loss: (post_event_loss_count > 0)
            .then_some((post_event_loss_sum / post_event_loss_count as f64) as f32),
        recovery_extension_steps,
        sampled_age_min,
        sampled_age_max,
        sampled_age_mean,
        particle_steps_per_sec: particle_steps / elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
        particle_steps,
        elapsed_ms: elapsed.as_secs_f64() * 1_000.0,
    })
}

struct AdaptiveFreshSeedEvalContext<'a> {
    params: &'a BurnBaseBatch,
    frozen_base: Option<&'a BurnBaseBatch>,
    target: &'a BurnTargetExample,
    material: &'a BurnAdaptiveMaterial,
    topology: &'a BurnAdaptiveTopology,
    adaptive: &'a AdaptiveTarget2dBurnConfig,
    config: DirectBasisTrainConfig,
}

async fn evaluate_adaptive_fresh_seed(
    context: AdaptiveFreshSeedEvalContext<'_>,
    seed: &BurnAdaptiveEvalSeed,
) -> Result<AdaptiveFreshSeedEval, Box<dyn std::error::Error>> {
    let AdaptiveFreshSeedEvalContext {
        params,
        frozen_base,
        target,
        material,
        topology,
        adaptive,
        config,
    } = context;
    let device = &target.target_rgb.device();
    let batch_size = seed.seeds.len();
    let indices = vec![0usize; batch_size];
    let mut x = seed.positions.clone();
    let mut s = seed.states.clone();
    let (
        represented_measure,
        bandwidth,
        scale_feature,
        pixel_size,
        output_scale,
        residual_gate,
    ) = material.batch(std::slice::from_ref(target), &indices);
    let mut rngs = seed
        .seeds
        .iter()
        .copied()
        .map(canonical_eval_mask_rng)
        .collect::<Vec<_>>();
    let mut displacement = Tensor::<BurnBackend, 1>::zeros([batch_size], device);
    let detached = params.detached();
    let chunk_steps = tbptt_chunk_steps(config);
    let mut completed_steps = 0usize;
    let mut topology_events = 0usize;
    let mut total_loss = 0.0_f32;
    let mut loss_count = 0usize;
    let mut total_render_rgb_mse = 0.0_f32;
    let mut render_rgb_mse_count = 0usize;
    let mut selection_psnr_db = f32::INFINITY;
    let mut horizon_seed_psnr_db = Vec::with_capacity(adaptive.checkpoint_horizons.len());
    let mut horizon_mean_render_rgb_mse =
        Vec::with_capacity(adaptive.checkpoint_horizons.len());
    let mut horizon_mean_psnr_db = Vec::with_capacity(adaptive.checkpoint_horizons.len());
    for horizon in adaptive.checkpoint_horizons.iter().copied() {
        while completed_steps < horizon {
            let steps = (horizon - completed_steps).min(chunk_steps);
            let rolled = rollout_adaptive_oracle_model_batch_chunk(
                &detached,
                frozen_base,
                std::slice::from_ref(target),
                &indices,
                x,
                s,
                represented_measure.clone(),
                bandwidth.clone(),
                scale_feature.clone(),
                residual_gate.clone(),
                adaptive.perception,
                adaptive.perception_options,
                adaptive.perception_semantics,
                adaptive.residual_perception_semantics,
                adaptive.material_scale_conditioning,
                adaptive.compatible_residual_material_features,
                config,
                material.active_particle_count,
                &mut rngs,
                true,
                Some(&seed.seeds),
                Some(&seed.update_masks),
                completed_steps,
                steps,
                displacement,
            );
            let before = vec![completed_steps; batch_size];
            completed_steps += steps;
            let after = vec![completed_steps; batch_size];
            let applied = topology.apply_scheduled(
                rolled.0,
                rolled.1,
                detach2(rolled.3),
                &before,
                &after,
            );
            let next_x = applied.0;
            let next_s = applied.1;
            topology_events += applied.2;
            displacement = detach1(rolled.2);
            x = detach3(next_x);
            s = detach3(next_s);
        }
        let loss = adaptive_target_splat_loss_batch_vector_base_only_selected(
            &x,
            &s,
            std::slice::from_ref(target),
            &indices,
            config,
            represented_measure.clone(),
            pixel_size.clone(),
            output_scale.clone(),
            displacement.clone(),
        )?;
        let splat_values = tensor1_vec_async(loss.splat.clone().inner()).await?;
        let loss_values = tensor1_vec_async(loss.total.inner()).await?;
        if adaptive.checkpoint_horizons.first().copied() == Some(horizon) {
            let dense_config = DirectBasisTrainConfig {
                example_batch_size: 1,
                target2d_loss_backend: Target2dLossBackend::Dense,
                ..config
            };
            let dense_loss = adaptive_target_splat_loss_batch_vector_base_only_selected(
                &x.clone().narrow(0, 0, 1),
                &s.clone().narrow(0, 0, 1),
                std::slice::from_ref(target),
                &indices[..1],
                dense_config,
                represented_measure.clone().narrow(0, 0, 1),
                pixel_size.clone().narrow(0, 0, 1),
                output_scale.clone().narrow(0, 0, 1),
                displacement.clone().narrow(0, 0, 1),
            )?;
            let dense_splat = tensor1_vec_async(dense_loss.splat.inner()).await?;
            validate_adaptive_target2d_primal_parity(
                &splat_values[..1],
                &dense_splat,
                horizon,
            )?;
            eprintln!(
                "{LOG_BACKEND} adaptive-target2d primal-parity horizon={horizon} tiled_splat={:.8} dense_splat={:.8}",
                splat_values[0], dense_splat[0],
            );
        }
        total_loss += loss_values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .sum::<f32>();
        loss_count += loss_values.iter().filter(|value| value.is_finite()).count();
        let mse_values = tensor1_vec_async(
            adaptive_uncentered_render_rgb_mse_batch(
                &x,
                &s,
                std::slice::from_ref(target),
                &indices,
                config,
                pixel_size.clone(),
                output_scale.clone(),
            )
            .inner(),
        )
        .await?;
        let finite_mse_count = mse_values.iter().filter(|value| value.is_finite()).count();
        let finite_mse_mean = mse_values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .sum::<f32>()
            / finite_mse_count.max(1) as f32;
        total_render_rgb_mse += mse_values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .sum::<f32>();
        render_rgb_mse_count += finite_mse_count;
        horizon_mean_render_rgb_mse.push(finite_mse_mean);
        let finite_loss_count = loss_values.iter().filter(|value| value.is_finite()).count();
        let finite_loss_mean = loss_values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .sum::<f32>()
            / finite_loss_count.max(1) as f32;
        let finite_splat_count = splat_values
            .iter()
            .filter(|value| value.is_finite())
            .count();
        let finite_splat_mean = splat_values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .sum::<f32>()
            / finite_splat_count.max(1) as f32;
        eprintln!(
            "{LOG_BACKEND} adaptive-target2d eval horizon={horizon} loss_mean={finite_loss_mean:.8} splat_mean={finite_splat_mean:.8} render_rgb_mse_mean={finite_mse_mean:.8} finite_loss={finite_loss_count}/{batch_size} finite_splat={finite_splat_count}/{batch_size} finite_mse={finite_mse_count}/{batch_size}"
        );
        let psnr_values = mse_values
            .into_iter()
            .map(|mse| {
                if mse.is_finite() {
                    -10.0 * mse.max(1.0e-12).log10()
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect::<Vec<_>>();
        horizon_seed_psnr_db.push(psnr_values.clone());
        selection_psnr_db = psnr_values
            .iter()
            .copied()
            .fold(selection_psnr_db, f32::min);
        horizon_mean_psnr_db.push(
            psnr_values.iter().copied().sum::<f32>()
                / psnr_values.len().max(1) as f32,
        );
    }
    let worst_psnr_drift_db = (0..batch_size)
        .map(|seed_index| {
            let values = horizon_seed_psnr_db
                .iter()
                .map(|horizon| horizon[seed_index])
                .collect::<Vec<_>>();
            if values.iter().any(|value| !value.is_finite()) {
                f32::INFINITY
            } else {
                values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                    - values.iter().copied().fold(f32::INFINITY, f32::min)
            }
        })
        .fold(0.0_f32, f32::max);
    let selection_score = if selection_psnr_db.is_finite() && worst_psnr_drift_db.is_finite() {
        selection_psnr_db
            - adaptive.event_training.checkpoint_drift_penalty_weight * worst_psnr_drift_db
    } else {
        f32::NEG_INFINITY
    };
    Ok(AdaptiveFreshSeedEval {
        mean_loss: if loss_count == 0 {
            1.0e6
        } else {
            total_loss / loss_count as f32
        },
        mean_render_rgb_mse: if render_rgb_mse_count == 0 {
            1.0e6
        } else {
            total_render_rgb_mse / render_rgb_mse_count as f32
        },
        selection_psnr_db,
        worst_psnr_drift_db,
        selection_score,
        horizon_mean_render_rgb_mse,
        horizon_mean_psnr_db,
        topology_events,
    })
}

#[allow(dead_code)]
pub(crate) fn train_adaptive_target2d_burn_dense(
    model: &mut NpaModel,
    example: &DirectBasisExample,
    plan: Target2dOracleTrainPlan,
    adaptive: AdaptiveTarget2dBurnConfig,
    checkpoint: Option<&Target2dBurnCheckpointConfig>,
    observer: Option<&mut dyn Target2dGpuTrainingObserver>,
) -> Result<BurnDenseOracleBatchOutput, Box<dyn std::error::Error>> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        pollster::block_on(train_adaptive_target2d_burn_dense_async(
            model, example, plan, adaptive, checkpoint, observer,
        ))
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (model, example, plan, adaptive, checkpoint, observer);
        Err(std::io::Error::other(
            "synchronous adaptive Burn/WGPU training is unavailable on wasm32; use the async adaptive Target2D API",
        )
        .into())
    }
}

pub(crate) async fn train_adaptive_target2d_burn_dense_async(
    model: &mut NpaModel,
    example: &DirectBasisExample,
    plan: Target2dOracleTrainPlan,
    mut adaptive: AdaptiveTarget2dBurnConfig,
    checkpoint: Option<&Target2dBurnCheckpointConfig>,
    mut observer: Option<&mut dyn Target2dGpuTrainingObserver>,
) -> Result<BurnDenseOracleBatchOutput, Box<dyn std::error::Error>> {
    let config = plan.train;
    let total_steps = plan.total_steps();
    if total_steps == 0
        || plan.steps_per_repetition == 0
        || plan.repetitions == 0
        || config.steps != total_steps
    {
        return Err(std::io::Error::other(
            "adaptive Target2D requires a non-empty consistent training plan",
        )
        .into());
    }
    if model.config.spatial_dims != 2
        || !model.config.stopgrad_pos
        || model.config.stopgrad_state
        || adaptive.perception.dim != 2
        || adaptive.perception.graph_policy
            != burn_automata_kernels::AdaptiveGraphPolicy::RawSupport
        || adaptive.material.active_particle_count() != config.rollout_particles
    {
        return Err(std::io::Error::other(
            "adaptive Target2D requires canonical 2D state-gradient training, raw-support perception, and matching active rows",
        )
        .into());
    }

    let _ = check_process_memory_budget("adaptive_target2d:start", config)?;
    let _ = check_gpu_memory_budget("adaptive_target2d:start", config)?;
    let device = BurnDevice::default();
    let frozen_base = adaptive
        .frozen_base
        .as_ref()
        .map(|base| BurnBaseBatch::from_models(std::slice::from_ref(base), &device))
        .transpose()?
        .map(|params| params.detached());
    if adaptive.frozen_base.as_ref().is_some_and(|base| {
        let mut normalized = model.config.clone();
        normalized.auxiliary_input_dims = base.config.auxiliary_input_dims;
        normalized != base.config
    }) {
        return Err(std::io::Error::other(
            "adaptive compatible-residual base and closure configurations differ",
        )
        .into());
    }
    let mut params = BurnBaseBatch::from_models(std::slice::from_ref(model), &device)?;
    let targets = burn_targets(std::slice::from_ref(example), config, &device)?;
    let target = &targets[0];
    let material = BurnAdaptiveMaterial::new(&adaptive, &device)?;
    let topology = BurnAdaptiveTopology::new(&adaptive, &device)?;
    let seed_bank = &mut adaptive.seed_bank;
    if seed_bank.pool_size != config.pool_size.max(config.example_batch_size).max(1)
        || seed_bank.particle_count != material.active_particle_count
        || seed_bank.state_dims != model.config.state_dims
        || seed_bank.update_masks.len() != seed_bank.pool_size * seed_bank.particle_count
        || seed_bank.eval_update_masks.len()
            != seed_bank.eval_seeds.len() * seed_bank.particle_count
    {
        return Err(std::io::Error::other(
            "adaptive Target2D seed-bank and training pool shapes differ",
        )
        .into());
    }
    let mut pool = BurnDeviceParticlePool::from_flat_values(
        std::mem::take(&mut seed_bank.positions),
        std::mem::take(&mut seed_bank.states),
        seed_bank.pool_size,
        seed_bank.particle_count,
        seed_bank.state_dims,
        &device,
    )?;
    let eval_count = seed_bank.eval_seeds.len();
    let eval_seed = BurnAdaptiveEvalSeed {
        positions: tensor3(
            std::mem::take(&mut seed_bank.eval_positions),
            [eval_count, seed_bank.particle_count, 2],
            &device,
        ),
        states: tensor3(
            std::mem::take(&mut seed_bank.eval_states),
            [eval_count, seed_bank.particle_count, seed_bank.state_dims],
            &device,
        ),
        update_masks: std::mem::take(&mut seed_bank.eval_update_masks),
        seeds: std::mem::take(&mut seed_bank.eval_seeds),
    };
    let resume_state = checkpoint
        .and_then(|checkpoint| checkpoint.resume_training_state.as_ref())
        .map(|path| Target2dTrainingCheckpoint::read(path))
        .transpose()?;
    if let (Some(state), Some(checkpoint)) = (resume_state.as_ref(), checkpoint) {
        validate_adaptive_target2d_resume(state, checkpoint, &params, config, total_steps)?;
        if !checkpoint.curriculum_resume
            && let Some(pool_snapshot) = state.particle_pool.as_ref()
        {
            pool.restore_target2d_snapshot(pool_snapshot, &device)?;
        }
    }
    let resume_completed_step = if checkpoint.is_some_and(|value| value.curriculum_resume) {
        0
    } else {
        resume_state
            .as_ref()
            .map_or(0, |state| state.completed_step)
    };
    let particle_pool_restored = resume_state
        .as_ref()
        .is_some_and(|state| {
            state.particle_pool.is_some()
                && !checkpoint.is_some_and(|value| value.curriculum_resume)
        });
    let mut optimizer = if let Some(state) = resume_state.as_ref() {
        BurnBaseBatchAdamWState::restore_target2d(state, &params, &device)?
    } else {
        BurnBaseBatchAdamWState::zeros_like(&params)
    };
    let mut checkpoint_state = checkpoint.map(BurnDenseCheckpointState::new);
    let mut history = Vec::new();
    let mut best_train_loss = None::<f32>;
    let mut best_train_step = 0usize;
    let initial_eval = evaluate_adaptive_fresh_seed(
        AdaptiveFreshSeedEvalContext {
            params: &params,
            frozen_base: frozen_base.as_ref(),
            target,
            material: &material,
            topology: &topology,
            adaptive: &adaptive,
            config,
        },
        &eval_seed,
    )
    .await?;
    let mut best_eval_loss = Some(initial_eval.mean_loss);
    let mut best_eval_render_rgb_mse = Some(initial_eval.mean_render_rgb_mse);
    let mut best_eval_psnr = Some(initial_eval.selection_psnr_db);
    let mut best_eval_worst_psnr_drift = initial_eval.worst_psnr_drift_db;
    let mut best_eval_selection_score = Some(initial_eval.selection_score);
    let mut best_eval_horizon_render_rgb_mse =
        initial_eval.horizon_mean_render_rgb_mse.clone();
    let mut best_eval_horizon_psnr = initial_eval.horizon_mean_psnr_db.clone();
    let mut best_eval_topology_events = initial_eval.topology_events;
    let mut best_eval_step = resume_completed_step;
    let mut best_params = params.detached();
    let mut measured_particle_steps = 0.0_f64;
    let mut measured_topology_events = 0usize;
    let mut measured_event_exposed_rows = 0usize;
    let mut measured_unique_event_exposed_rows = 0usize;
    let mut measured_event_quota_shortfall_rows = 0usize;
    let mut measured_max_event_delay_steps = 0usize;
    let mut event_quota_met_steps = 0usize;
    let mut measured_post_event_row_chunks = 0usize;
    let mut measured_post_event_loss_sum = 0.0_f64;
    let mut measured_recovery_extension_steps = 0usize;
    let mut min_backward_scale = 1.0_f32;
    let mut max_backward_scale = 0.0_f32;
    let mut sampled_age_min = usize::MAX;
    let mut sampled_age_max = 0usize;
    let mut sampled_age_mean_sum = 0.0_f64;
    let mut sampled_age_batches = 0usize;
    let training_started_at = Instant::now();
    let mut observer_last_snapshot_at = training_started_at;
    let mut observer_last_snapshot_step = resume_completed_step;
    let mut steps_completed = resume_completed_step;
    let mut stopped_early = false;
    let mut metric_collection_steps = 0usize;
    let mut evaluation_steps = 0usize;
    if resume_completed_step == 0
        && let Some(state) = checkpoint_state.as_mut()
    {
        state.write_best_batch(
            &params,
            "adaptive-target2d-initial",
            0,
            None,
            Some(initial_eval.mean_loss),
            Some(initial_eval.selection_score),
        )?;
    }

    for step in resume_completed_step.saturating_add(1)..=total_steps {
        if observer
            .as_deref()
            .is_some_and(Target2dGpuTrainingObserver::should_stop)
        {
            stopped_early = true;
            break;
        }
        let (repetition, phase_step, upstream_epoch) =
            oracle_repetition_position(step, plan.steps_per_repetition);
        if phase_step == 1 && repetition > 0 {
            optimizer = BurnBaseBatchAdamWState::zeros_like(&params);
            pool.reset();
        }
        let lr_scale =
            milestone_lr_scale(phase_step, &plan.scheduler_milestones, plan.scheduler_gamma);
        let optimizer_config = AdamWConfig {
            learning_rate: plan.optimizer.learning_rate * lr_scale,
            ..plan.optimizer
        };
        let should_report =
            step == total_steps || step.is_multiple_of(config.report_interval.max(1));
        let should_eval = config.eval_interval > 0
            && (step == total_steps || step.is_multiple_of(config.eval_interval.max(1)));
        let checkpoint_due = checkpoint_state
            .as_ref()
            .is_some_and(|state| state.should_write_current(step));
        let observer_due = observer.as_deref().is_some_and(|observer| {
            step == resume_completed_step.saturating_add(1)
                || step == total_steps
                || (step.saturating_sub(observer_last_snapshot_step)
                    >= observer.snapshot_interval_steps().max(1)
                    && observer_last_snapshot_at.elapsed()
                        >= observer.snapshot_interval_duration())
        });
        let collect_metrics = should_report || checkpoint_due || observer_due;
        metric_collection_steps += usize::from(collect_metrics);
        evaluation_steps += usize::from(should_eval);
        let stats = train_adaptive_step_tbptt(
            &mut params,
            frozen_base.as_ref(),
            &mut optimizer,
            target,
            &material,
            &topology,
            &adaptive,
            config,
            config
                .seed
                .wrapping_add((step as u64).wrapping_mul(0x9e37_79b9)),
            &mut pool,
            upstream_epoch.is_multiple_of(config.inject_seed_interval.max(1)),
            collect_metrics,
            optimizer_config,
        )
        .await?;
        debug_assert_eq!(stats.metrics_collected, collect_metrics);
        measured_particle_steps += stats.particle_steps;
        steps_completed = step;
        measured_topology_events += stats.topology_events;
        measured_event_exposed_rows += stats.event_exposed_rows;
        measured_unique_event_exposed_rows += stats.unique_event_exposed_rows;
        measured_event_quota_shortfall_rows += stats.event_quota_shortfall_rows;
        measured_max_event_delay_steps =
            measured_max_event_delay_steps.max(stats.max_event_delay_steps);
        event_quota_met_steps += usize::from(
            adaptive.event_training.enabled && stats.event_quota_shortfall_rows == 0,
        );
        measured_post_event_row_chunks += stats.post_event_row_chunks;
        if stats.metrics_collected {
            measured_post_event_loss_sum += stats
                .post_event_mean_loss
                .map_or(0.0, |loss| f64::from(loss) * stats.post_event_row_chunks as f64);
        }
        measured_recovery_extension_steps += stats.recovery_extension_steps;
        min_backward_scale = min_backward_scale.min(stats.backward_scale);
        max_backward_scale = max_backward_scale.max(stats.backward_scale);
        sampled_age_min = sampled_age_min.min(stats.sampled_age_min);
        sampled_age_max = sampled_age_max.max(stats.sampled_age_max);
        sampled_age_mean_sum += stats.sampled_age_mean;
        sampled_age_batches += 1;
        if stats.metrics_collected && best_train_loss.is_none_or(|best| stats.loss < best) {
            best_train_loss = Some(stats.loss);
            best_train_step = step;
        }

        let evaluation = if should_eval {
            Some(
                evaluate_adaptive_fresh_seed(
                    AdaptiveFreshSeedEvalContext {
                        params: &params,
                        frozen_base: frozen_base.as_ref(),
                        target,
                        material: &material,
                        topology: &topology,
                        adaptive: &adaptive,
                        config,
                    },
                    &eval_seed,
                )
                .await?,
            )
        } else {
            None
        };
        if let Some(evaluation) = evaluation.as_ref()
            && adaptive_checkpoint_is_better(
                evaluation.selection_score,
                best_eval_selection_score,
            )
        {
            best_eval_loss = Some(evaluation.mean_loss);
            best_eval_render_rgb_mse = Some(evaluation.mean_render_rgb_mse);
            best_eval_psnr = Some(evaluation.selection_psnr_db);
            best_eval_worst_psnr_drift = evaluation.worst_psnr_drift_db;
            best_eval_selection_score = Some(evaluation.selection_score);
            best_eval_horizon_render_rgb_mse
                .clone_from(&evaluation.horizon_mean_render_rgb_mse);
            best_eval_horizon_psnr.clone_from(&evaluation.horizon_mean_psnr_db);
            best_eval_topology_events = evaluation.topology_events;
            best_eval_step = step;
            best_params = params.detached();
            if let Some(state) = checkpoint_state.as_mut() {
                state.write_best_batch(
                    &params,
                    "adaptive-target2d",
                    step,
                    Some(stats.loss),
                    Some(evaluation.mean_loss),
                    Some(evaluation.selection_score),
                )?;
            }
        }
        if checkpoint_due {
            let state = checkpoint_state.as_mut().expect("checkpoint state");
            state.write_current_batch(
                &params,
                "adaptive-target2d",
                step,
                Some(stats.loss),
                evaluation.as_ref().map(|value| value.mean_loss),
                evaluation
                    .as_ref()
                    .map(|value| value.selection_score),
            )?;
            if let Some(checkpoint) = checkpoint {
                write_adaptive_target2d_training_state(
                    checkpoint,
                    state,
                    &optimizer,
                    &pool,
                    &params,
                    config,
                    step,
                )?;
            }
        }
        if should_report {
            println!(
                "{LOG_BACKEND} adaptive-target2d step {step}/{total_steps} repetition={}/{} phase_step={phase_step}/{} lr={:.3e} loss={:.6} optimization_loss={:.6} max_trajectory_loss={:.6} post_event_loss={} eval_loss={} psnr_db={} psnr_drift_db={} selection_score={} grad_norm={:.3e} grad_scale={:.3e} backward_scale={:.3e} sampled_age={}/{:.0}/{} active={} reference={} topology_events={} event_rows={} event_quota_shortfall={} max_event_delay_steps={} post_event_row_chunks={} recovery_extension_steps={} particle_steps_per_sec={:.0} elapsed_ms={:.1}",
                repetition + 1,
                plan.repetitions,
                plan.steps_per_repetition,
                optimizer_config.learning_rate,
                stats.loss,
                stats.optimization_loss,
                stats.max_trajectory_loss,
                stats
                    .post_event_mean_loss
                    .map(|value| format!("{value:.6}"))
                    .unwrap_or_else(|| "n/a".to_owned()),
                evaluation
                    .as_ref()
                    .map(|value| format!("{:.6}", value.mean_loss))
                    .unwrap_or_else(|| "n/a".to_string()),
                evaluation
                    .as_ref()
                    .map(|value| format!("{:.3}", value.selection_psnr_db))
                    .unwrap_or_else(|| "n/a".to_string()),
                evaluation
                    .as_ref()
                    .map(|value| format!("{:.3}", value.worst_psnr_drift_db))
                    .unwrap_or_else(|| "n/a".to_string()),
                evaluation
                    .as_ref()
                    .map(|value| format!("{:.3}", value.selection_score))
                    .unwrap_or_else(|| "n/a".to_string()),
                stats.grad_norm,
                stats.grad_scale,
                stats.backward_scale,
                stats.sampled_age_min,
                stats.sampled_age_mean,
                stats.sampled_age_max,
                material.active_particle_count,
                material.reference_particle_count,
                stats.topology_events,
                stats.event_exposed_rows,
                stats.event_quota_shortfall_rows,
                stats.max_event_delay_steps,
                stats.post_event_row_chunks,
                stats.recovery_extension_steps,
                stats.particle_steps_per_sec,
                stats.elapsed_ms,
            );
            history.push(CliHyper2dDirectBasisHistoryEntry {
                step,
                loss: stats.loss,
                eval_loss: evaluation.as_ref().map(|value| CliHyper2dDirectBasisLossSummary {
                    examples: eval_seed.seeds.len()
                        * adaptive.checkpoint_horizons.len(),
                    mean_total_loss: value.mean_loss,
                    max_total_loss: value.mean_loss,
                    mean_splat_loss: value.mean_loss,
                    mean_color_loss: 0.0,
                    mean_density_loss: 0.0,
                }),
                base_grad_norm: stats.grad_norm,
                base_grad_scale: stats.grad_scale,
                mean_adapter_grad_norm: 0.0,
                max_adapter_grad_norm: 0.0,
                examples_seen: config.example_batch_size.max(1),
                particle_steps_per_sec: stats.particle_steps_per_sec,
                elapsed_ms: stats.elapsed_ms,
            });
        }
        if observer_due {
            params
                .write_to_models_async(std::slice::from_mut(model))
                .await?;
            let eval_loss = evaluation.as_ref().map(|value| Target2dGpuLossSummary {
                examples: eval_seed.seeds.len() * adaptive.checkpoint_horizons.len(),
                mean_total_loss: value.mean_loss,
                max_total_loss: value.mean_loss,
                mean_splat_loss: value.mean_loss,
                mean_color_loss: 0.0,
                mean_density_loss: 0.0,
            });
            if let Some(observer) = observer.as_deref_mut() {
                observer.on_progress(Target2dGpuTrainingProgress {
                    step,
                    total_steps,
                    loss: stats.loss,
                    eval_loss,
                    render_rgb_psnr_db: evaluation
                        .as_ref()
                        .map(|value| value.selection_psnr_db),
                    base_grad_norm: stats.grad_norm,
                    base_grad_scale: stats.grad_scale,
                    particle_steps_per_sec: stats.particle_steps_per_sec,
                    elapsed_ms: training_started_at.elapsed().as_secs_f64() * 1_000.0,
                    model: model.clone(),
                });
            }
            observer_last_snapshot_step = if step == resume_completed_step.saturating_add(1) {
                resume_completed_step
            } else {
                step
            };
            observer_last_snapshot_at = Instant::now();
            if observer
                .as_deref()
                .is_some_and(Target2dGpuTrainingObserver::should_stop)
            {
                stopped_early = step < total_steps;
                break;
            }
        }
    }

    if let Some(state) = checkpoint_state.as_mut() {
        state.write_current_batch(
            &params,
            "adaptive-target2d-final",
            steps_completed,
            best_train_loss,
            best_eval_loss,
            best_eval_selection_score,
        )?;
        if let Some(checkpoint) = checkpoint {
            write_adaptive_target2d_training_state(
                checkpoint,
                state,
                &optimizer,
                &pool,
                &params,
                config,
                steps_completed,
            )?;
        }
    }
    // Checkpoint the final optimizer state above, but publish the parameters
    // selected by the fresh-seed PSNR gate. Returning the last update can
    // silently replace a strong initialization with a divergent trajectory.
    if stopped_early {
        params
            .write_to_models_async(std::slice::from_mut(model))
            .await?;
    } else {
        best_params
            .write_to_models_async(std::slice::from_mut(model))
            .await?;
    }
    let material_scale_column = material_scale_column_stats(model);
    let mean_throughput = measured_particle_steps
        / training_started_at
            .elapsed()
            .as_secs_f64()
            .max(f64::MIN_POSITIVE);
    let checkpoint_report = checkpoint_state
        .as_ref()
        .map(BurnDenseCheckpointState::report_json);
    let pool_age_sampling = json!({
        "strata": adaptive.pool_age_strata,
        "max_age_steps": adaptive.max_pool_age_steps,
        "sampled_min_steps": (sampled_age_min != usize::MAX).then_some(sampled_age_min),
        "sampled_max_steps": sampled_age_max,
        "sampled_mean_steps": (sampled_age_batches > 0)
            .then_some(sampled_age_mean_sum / sampled_age_batches as f64),
        "final_min_steps": pool.ages.iter().copied().min(),
        "final_max_steps": pool.ages.iter().copied().max(),
        "final_mean_steps": (!pool.ages.is_empty()).then_some(
            pool.ages.iter().copied().sum::<usize>() as f64 / pool.ages.len() as f64
        ),
    });
    let event_training_report = json!({
        "enabled": adaptive.event_training.enabled,
        "post_event_recovery_steps": adaptive.event_training.post_event_recovery_steps,
        "post_event_loss_weight": adaptive.event_training.post_event_loss_weight,
        "post_event_degradation_weight":
            adaptive.event_training.post_event_degradation_weight,
        "checkpoint_drift_penalty_weight":
            adaptive.event_training.checkpoint_drift_penalty_weight,
        "degradation_objective": (adaptive.event_training.post_event_degradation_weight > 0.0)
            .then_some("positive_part(post_event_loss-detached_pre_event_loss)"),
        "min_event_trajectories_per_batch":
            adaptive.event_training.min_event_trajectories_per_batch,
        "max_recovery_extension_steps":
            adaptive.event_training.recovery_extension_budget(),
        "scheduled_event_rows": measured_event_exposed_rows,
        "unique_event_rows_per_step_sum": measured_unique_event_exposed_rows,
        "quota_shortfall_rows": measured_event_quota_shortfall_rows,
        "quota_met_steps": event_quota_met_steps,
        "max_event_delay_steps": measured_max_event_delay_steps,
        "training_steps": steps_completed.saturating_sub(resume_completed_step),
        "post_event_row_chunks": measured_post_event_row_chunks,
        "mean_post_event_loss": (measured_post_event_row_chunks > 0).then_some(
            measured_post_event_loss_sum / measured_post_event_row_chunks as f64
        ),
        "recovery_extension_steps": measured_recovery_extension_steps,
    });
    let loss_objective_report = json!({
        "center": config.loss_config.center,
        "splat_loss_weight": config.loss_config.splat_loss_weight,
        "color_loss_weight": config.loss_config.color_loss_weight,
        "density_loss_weight": config.loss_config.density_loss_weight,
        "background_density_loss_weight":
            config.loss_config.background_density_loss_weight,
        "foreground_density_loss_weight":
            config.loss_config.foreground_density_loss_weight,
        "composited_rgb_loss_weight":
            config.loss_config.composited_rgb_loss_weight,
        "render_rgb_loss_weight": config.loss_config.render_rgb_loss_weight,
        "shape_chamfer_loss_weight":
            config.loss_config.shape_chamfer_loss_weight,
        "displacement_regularizer_weight":
            config.loss_config.displacement_regularizer_weight,
        "overflow_regularizer_weight":
            config.loss_config.overflow_regularizer_weight,
        "bound_regularizer_weight":
            config.loss_config.bound_regularizer_weight,
        "psnr_aligned_render_only":
            config.loss_config.splat_loss_weight > 0.0
                && config.loss_config.render_rgb_loss_weight > 0.0
                && config.loss_config.color_loss_weight == 0.0
                && config.loss_config.density_loss_weight == 0.0
                && config.loss_config.background_density_loss_weight == 0.0
                && config.loss_config.foreground_density_loss_weight == 0.0
                && config.loss_config.composited_rgb_loss_weight == 0.0
                && config.loss_config.shape_chamfer_loss_weight == 0.0
                && config.loss_config.displacement_regularizer_weight == 0.0
                && config.loss_config.overflow_regularizer_weight == 0.0
                && config.loss_config.bound_regularizer_weight == 0.0,
    });
    let mut metrics = json!({
        "training_path": if adaptive.residual_perception_semantics
            == Some(burn_automata_kernels::AdaptivePerceptionSemantics::NormalizedAdaptive)
        {
            "adaptive_recurrent_target2d_frozen_base_normalized_adaptive_residual"
        } else if adaptive.compatible_residual_material_features {
            "adaptive_recurrent_target2d_frozen_base_material_conditioned_residual"
        } else if frozen_base.is_some() {
            "adaptive_recurrent_target2d_frozen_base_compatible_residual"
        } else if adaptive.perception_semantics
            == burn_automata_kernels::AdaptivePerceptionSemantics::NormalizedAdaptive
        {
            "adaptive_recurrent_target2d_normalized_adaptive_material_conditioned_rule"
        } else if adaptive.material_scale_conditioning {
            "adaptive_recurrent_target2d_shared_scale_conditioned_rule"
        } else {
            "adaptive_recurrent_target2d_shared_rule"
        },
        "frozen_native_scale_base": frozen_base.is_some(),
        "optimize_material_scale_only": adaptive.optimize_material_scale_only,
        "log1p_trajectory_loss": adaptive.log1p_trajectory_loss,
        "trajectory_tail_fraction": adaptive.trajectory_tail_fraction,
        "trajectory_tail_weight": adaptive.trajectory_tail_weight,
        "perception": if adaptive.residual_perception_semantics
            == Some(burn_automata_kernels::AdaptivePerceptionSemantics::NormalizedAdaptive)
        {
            "represented_measure_base_plus_shepard_normalized_moment_corrected_residual_device_vjp"
        } else if adaptive.perception_semantics
            == burn_automata_kernels::AdaptivePerceptionSemantics::NormalizedAdaptive
        {
            "shepard_normalized_moment_corrected_raw_support_device_vjp"
        } else {
            "represented_measure_raw_support_device_vjp"
        },
        "render": "one_variable_isotropic_gaussian_per_active_row",
        "hidden_fine_rows": 0,
        "resume_completed_step": resume_completed_step,
        "resume_optimizer_step": resume_state.as_ref().map(|state| state.optimizer_step),
        "curriculum_resume": checkpoint.is_some_and(|value| value.curriculum_resume),
        "optimizer_step": optimizer.step,
        "particle_pool_restored": particle_pool_restored,
        "topology": "device_resident_local_detail_paired_four_to_one_split_merge",
        "topology_events": measured_topology_events,
        "topology_material_conservation": "exact_by_construction",
        "active_particle_count": material.active_particle_count,
        "reference_particle_count": material.reference_particle_count,
        "seed_restriction": match adaptive.material.seed_layout {
            crate::adaptive::AdaptiveMaterialSeedLayout::CanonicalGrouped => {
                "canonical_grouped_reference_to_active_no_templates"
            }
            crate::adaptive::AdaptiveMaterialSeedLayout::UniformContinuous => {
                "continuous_uniform_active_material_no_templates"
            }
            crate::adaptive::AdaptiveMaterialSeedLayout::GradedContinuous => {
                "continuous_graded_active_material_no_templates"
            }
        },
        "fresh_seed_trajectories_per_injection": adaptive.fresh_seed_trajectories,
        "backward_loss_scaling": if adaptive.backward_loss_scale != 1.0 {
            "configured_common_scale_before_per_parameter_normalization"
        } else {
            "disabled"
        },
        "min_backward_scale": min_backward_scale,
        "max_backward_scale": max_backward_scale,
        "seed_max_measure_relative_error": adaptive.seed_bank.max_measure_relative_error,
        "seed_max_centroid_l2_error": adaptive.seed_bank.max_centroid_l2_error,
        "seed_max_extensive_state_l2_error": adaptive.seed_bank.max_extensive_state_l2_error,
        "mean_particle_steps_per_sec": mean_throughput,
        "best_fresh_seed_eval_loss": best_eval_loss,
        "best_fresh_seed_render_rgb_psnr_db": best_eval_psnr.map(|value| vec![value]),
        "best_fresh_seed_horizon_mean_psnr_db": best_eval_horizon_psnr,
        "best_fresh_seed_topology_events": best_eval_topology_events,
        "checkpoint_seeds": eval_seed.seeds,
        "checkpoint_horizons": adaptive.checkpoint_horizons,
        "pool_age_sampling": pool_age_sampling,
        "best_fresh_seed_eval_step": best_eval_step,
        "returned_model_selection":
            "best_worst_checkpoint_seed_horizon_psnr_minus_weighted_worst_seed_drift",
        "material_scale_w1_l2_norm": material_scale_column.map(|stats| stats.0),
        "material_scale_w1_rms": material_scale_column.map(|stats| stats.1),
        "material_scale_w1_max_abs": material_scale_column.map(|stats| stats.2),
        "checkpoint": checkpoint_report,
    });
    if let Some(metrics) = metrics.as_object_mut() {
        metrics.insert("steps_completed".to_owned(), steps_completed.into());
        metrics.insert("stopped_early".to_owned(), stopped_early.into());
        metrics.insert(
            "metric_collection_steps".to_owned(),
            metric_collection_steps.into(),
        );
        metrics.insert(
            "optimizer_steps_without_host_metrics".to_owned(),
            steps_completed
                .saturating_sub(resume_completed_step)
                .saturating_sub(metric_collection_steps)
                .into(),
        );
        metrics.insert("evaluation_steps".to_owned(), evaluation_steps.into());
        metrics.insert(
            "best_fresh_seed_worst_psnr_drift_db".to_owned(),
            best_eval_worst_psnr_drift.into(),
        );
        metrics.insert(
            "best_fresh_seed_selection_score".to_owned(),
            best_eval_selection_score.into(),
        );
        metrics.insert(
            "optimizer_update_cadence".to_owned(),
            "once_per_outer_step_after_accumulating_tbptt_and_event_segments".into(),
        );
        metrics.insert("event_training".to_owned(), event_training_report);
        metrics.insert("loss_objective".to_owned(), loss_objective_report);
        metrics.insert(
            "best_fresh_seed_render_rgb_mse".to_owned(),
            best_eval_render_rgb_mse.into(),
        );
        metrics.insert(
            "best_fresh_seed_horizon_mean_render_rgb_mse".to_owned(),
            best_eval_horizon_render_rgb_mse.into(),
        );
        metrics.insert(
            "compact_recurrent_memory_dims".to_owned(),
            adaptive.compact_recurrent_memory_dims.into(),
        );
        metrics.insert(
            "compact_recurrent_memory_rows".to_owned(),
            (adaptive.compact_recurrent_memory_dims > 0)
                .then_some(material.active_particle_count)
                .into(),
        );
    }
    Ok(BurnDenseOracleBatchOutput {
        backend: BACKEND,
        device: DEVICE_LABEL.to_string(),
        metrics,
        per_model_history: vec![history.clone()],
        history,
        best_train_loss: vec![best_train_loss],
        best_train_step: vec![best_train_step],
    })
}

fn material_scale_column_stats(model: &NpaModel) -> Option<(f32, f32, f32)> {
    (model.config.auxiliary_input_dims == 1).then(|| {
        let input_dims = model.config.perception_dims();
        let column = model
            .weights
            .w1
            .chunks_exact(input_dims)
            .map(|row| row[input_dims - 1]);
        let (sum_squares, max_abs, count) = column.fold(
            (0.0_f64, 0.0_f32, 0usize),
            |(sum_squares, max_abs, count), value| {
                (
                    sum_squares + f64::from(value) * f64::from(value),
                    max_abs.max(value.abs()),
                    count + 1,
                )
            },
        );
        let l2 = sum_squares.sqrt() as f32;
        let rms = (sum_squares / count.max(1) as f64).sqrt() as f32;
        (l2, rms, max_abs)
    })
}
