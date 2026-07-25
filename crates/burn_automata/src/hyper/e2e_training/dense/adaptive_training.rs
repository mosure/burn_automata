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
    loss: f32,
    optimization_loss: f32,
    max_trajectory_loss: f32,
    grad_norm: f32,
    grad_scale: f32,
    backward_scale: f32,
    topology_events: usize,
    sampled_age_min: usize,
    sampled_age_max: usize,
    sampled_age_mean: f64,
    particle_steps_per_sec: f64,
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
    selection_psnr_db: f32,
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
fn train_adaptive_step_tbptt(
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
    optimizer_config: AdamWConfig,
) -> Result<AdaptiveTrainStepStats, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let device = &target.target_rgb.device();
    let trajectories = config.example_batch_size.max(1);
    let indices = vec![0usize; trajectories];
    let mut rng = StdRng::seed_from_u64(step_seed ^ 0xada2_7a2d);
    let pool_batch = pool.sample_batch_with_fresh_rows(
        &mut rng,
        trajectories,
        BurnPoolSampling {
            fresh_seed_rows: usize::from(replace_pool_seed)
                * adaptive.fresh_seed_trajectories,
            max_age_steps: (adaptive.max_pool_age_steps > 0)
                .then_some(adaptive.max_pool_age_steps),
            age_strata: adaptive.pool_age_strata,
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
    let chunk_steps = tbptt_chunk_steps(config);
    let rollout_steps = sampled_training_rollout_steps(config, step_seed);
    let mut remaining_steps = rollout_steps;
    let mut loss_sum = 0.0_f32;
    let mut optimization_loss_sum = 0.0_f32;
    let mut max_trajectory_loss = 0.0_f32;
    let mut loss_chunks = 0usize;
    let mut grad_norm_sum = 0.0_f32;
    let mut grad_scale_sum = 0.0_f32;
    let mut backward_scale_sum = 0.0_f32;
    let mut particle_steps = 0.0_f64;
    let mut topology_events = 0usize;

    while remaining_steps > 0 {
        let final_chunk = remaining_steps <= chunk_steps;
        let steps = tbptt_next_chunk_steps(
            remaining_steps,
            chunk_steps,
            config.loss_on_final_chunk_only,
        );
        let optimize_chunk = !config.loss_on_final_chunk_only || final_chunk;
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
        trajectory_ages = next_trajectory_ages;
        if optimize_chunk {
            let loss = adaptive_target_splat_loss_batch_vector_base_only_selected(
                &next_x,
                &next_s,
                std::slice::from_ref(target),
                indices,
                config,
                represented_measure.clone(),
                pixel_size.clone(),
                output_scale.clone(),
                next_displacement.clone(),
            )?;
            let loss_values = tensor1_vec(loss.total.clone().inner())?;
            let finite_losses = loss_values
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>();
            let scalar = if finite_losses.is_empty() {
                1.0e6
            } else {
                finite_losses.iter().sum::<f32>() / finite_losses.len() as f32
            };
            let optimization_scalar = if adaptive.log1p_trajectory_loss {
                if finite_losses.is_empty() {
                    1.0e6_f32.ln_1p()
                } else {
                    finite_losses
                        .iter()
                        .map(|value| value.max(0.0).ln_1p())
                        .sum::<f32>()
                        / finite_losses.len() as f32
                }
            } else {
                scalar
            };
            let tail_count = adaptive_trajectory_tail_count(
                finite_losses.len(),
                adaptive.trajectory_tail_fraction,
            );
            let optimization_scalar = if tail_count > 0 && adaptive.trajectory_tail_weight > 0.0 {
                let mut ordered = finite_losses
                    .iter()
                    .map(|value| {
                        if adaptive.log1p_trajectory_loss {
                            value.max(0.0).ln_1p()
                        } else {
                            *value
                        }
                    })
                    .collect::<Vec<_>>();
                ordered.sort_by(|lhs, rhs| rhs.total_cmp(lhs));
                let tail_mean =
                    ordered[..tail_count].iter().sum::<f32>() / tail_count as f32;
                (optimization_scalar + adaptive.trajectory_tail_weight * tail_mean)
                    / (1.0 + adaptive.trajectory_tail_weight)
            } else {
                optimization_scalar
            };
            max_trajectory_loss = max_trajectory_loss.max(
                finite_losses
                    .iter()
                    .copied()
                    .fold(0.0_f32, f32::max),
            );
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
            let objective = if tail_count > 0 && adaptive.trajectory_tail_weight > 0.0 {
                let tail_mean = optimization_total
                    .clone()
                    .topk_with_indices(tail_count, 0)
                    .0
                    .mean();
                (optimization_total.mean()
                    + tail_mean.mul_scalar(adaptive.trajectory_tail_weight))
                .div_scalar(1.0 + adaptive.trajectory_tail_weight)
            } else {
                optimization_total.mean()
            };
            let mut gradients = objective.mul_scalar(backward_scale).backward();
            let (grad_norms, grad_scales) = if adaptive.optimize_material_scale_only {
                params.apply_adamw_last_input_column(
                    &mut gradients,
                    optimizer,
                    optimizer_config,
                    config.per_parameter_grad_normalization,
                    true,
                )?
            } else {
                params.apply_adamw(
                    &mut gradients,
                    optimizer,
                    optimizer_config,
                    config.per_parameter_grad_normalization,
                    true,
                )?
            };
            loss_sum += scalar;
            optimization_loss_sum += optimization_scalar;
            loss_chunks += 1;
            grad_norm_sum += grad_norms.first().copied().unwrap_or(0.0);
            grad_scale_sum += grad_scales.first().copied().unwrap_or(1.0);
            backward_scale_sum += backward_scale;
        }
        x = detach3(next_x);
        s = detach3(next_s);
        displacement = detach1(next_displacement);
        particle_steps += (actual_trajectories * material.active_particle_count * steps) as f64;
        remaining_steps -= steps;
    }

    pool.update_batch_with_ages(&pool_batch.pool_indices, &trajectory_ages, x, s)?;
    let elapsed = started.elapsed();
    let loss_chunks = loss_chunks.max(1);
    Ok(AdaptiveTrainStepStats {
        loss: loss_sum / loss_chunks as f32,
        optimization_loss: optimization_loss_sum / loss_chunks as f32,
        max_trajectory_loss,
        grad_norm: grad_norm_sum / loss_chunks as f32,
        grad_scale: grad_scale_sum / loss_chunks as f32,
        backward_scale: backward_scale_sum / loss_chunks as f32,
        topology_events,
        sampled_age_min,
        sampled_age_max,
        sampled_age_mean,
        particle_steps_per_sec: particle_steps / elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
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

fn evaluate_adaptive_fresh_seed(
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
    let mut selection_psnr_db = f32::INFINITY;
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
        let splat_values = tensor1_vec(loss.splat.clone().inner())?;
        let loss_values = tensor1_vec(loss.total.inner())?;
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
            let dense_splat = tensor1_vec(dense_loss.splat.inner())?;
            validate_adaptive_target2d_primal_parity(
                &splat_values[..1],
                &dense_splat,
                horizon,
            )?;
        }
        total_loss += loss_values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .sum::<f32>();
        loss_count += loss_values.iter().filter(|value| value.is_finite()).count();
        let mse_values = tensor1_vec(
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
        )?;
        let finite_mse_count = mse_values.iter().filter(|value| value.is_finite()).count();
        let finite_mse_mean = mse_values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .sum::<f32>()
            / finite_mse_count.max(1) as f32;
        let finite_loss_count = loss_values.iter().filter(|value| value.is_finite()).count();
        let finite_loss_mean = loss_values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .sum::<f32>()
            / finite_loss_count.max(1) as f32;
        eprintln!(
            "{LOG_BACKEND} adaptive-target2d eval horizon={horizon} loss_mean={finite_loss_mean:.8} render_rgb_mse_mean={finite_mse_mean:.8} finite_loss={finite_loss_count}/{batch_size} finite_mse={finite_mse_count}/{batch_size}"
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
        selection_psnr_db = psnr_values
            .iter()
            .copied()
            .fold(selection_psnr_db, f32::min);
        horizon_mean_psnr_db.push(
            psnr_values.iter().copied().sum::<f32>()
                / psnr_values.len().max(1) as f32,
        );
    }
    Ok(AdaptiveFreshSeedEval {
        mean_loss: if loss_count == 0 {
            1.0e6
        } else {
            total_loss / loss_count as f32
        },
        selection_psnr_db,
        horizon_mean_psnr_db,
        topology_events,
    })
}

pub(crate) fn train_adaptive_target2d_burn_dense(
    model: &mut NpaModel,
    example: &DirectBasisExample,
    plan: Target2dOracleTrainPlan,
    mut adaptive: AdaptiveTarget2dBurnConfig,
    checkpoint: Option<&Target2dBurnCheckpointConfig>,
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
    )?;
    let mut best_eval_loss = Some(initial_eval.mean_loss);
    let mut best_eval_psnr = Some(initial_eval.selection_psnr_db);
    let mut best_eval_horizon_psnr = initial_eval.horizon_mean_psnr_db.clone();
    let mut best_eval_topology_events = initial_eval.topology_events;
    let mut best_eval_step = resume_completed_step;
    let mut best_params = params.detached();
    let mut measured_particle_steps = 0.0_f64;
    let mut measured_elapsed_ms = 0.0_f64;
    let mut measured_topology_events = 0usize;
    let mut min_backward_scale = 1.0_f32;
    let mut max_backward_scale = 0.0_f32;
    let mut sampled_age_min = usize::MAX;
    let mut sampled_age_max = 0usize;
    let mut sampled_age_mean_sum = 0.0_f64;
    let mut sampled_age_batches = 0usize;
    if resume_completed_step == 0
        && let Some(state) = checkpoint_state.as_mut()
    {
        state.write_best_batch(
            &params,
            "adaptive-target2d-initial",
            0,
            None,
            Some(initial_eval.mean_loss),
            Some(initial_eval.selection_psnr_db),
        )?;
    }

    for step in resume_completed_step.saturating_add(1)..=total_steps {
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
            optimizer_config,
        )?;
        measured_particle_steps += stats.particle_steps_per_sec * stats.elapsed_ms / 1_000.0;
        measured_elapsed_ms += stats.elapsed_ms;
        measured_topology_events += stats.topology_events;
        min_backward_scale = min_backward_scale.min(stats.backward_scale);
        max_backward_scale = max_backward_scale.max(stats.backward_scale);
        sampled_age_min = sampled_age_min.min(stats.sampled_age_min);
        sampled_age_max = sampled_age_max.max(stats.sampled_age_max);
        sampled_age_mean_sum += stats.sampled_age_mean;
        sampled_age_batches += 1;
        if best_train_loss.is_none_or(|best| stats.loss < best) {
            best_train_loss = Some(stats.loss);
            best_train_step = step;
        }

        let should_report =
            step == total_steps || step.is_multiple_of(config.report_interval.max(1));
        let should_eval = config.eval_interval > 0
            && (step == total_steps || step.is_multiple_of(config.eval_interval.max(1)));
        let checkpoint_due = checkpoint_state
            .as_ref()
            .is_some_and(|state| state.should_write_current(step));
        let evaluation = should_eval
            .then(|| {
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
            })
            .transpose()?;
        if let Some(evaluation) = evaluation.as_ref()
            && adaptive_checkpoint_is_better(
                evaluation.selection_psnr_db,
                best_eval_psnr,
            )
        {
            best_eval_loss = Some(evaluation.mean_loss);
            best_eval_psnr = Some(evaluation.selection_psnr_db);
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
                    Some(evaluation.selection_psnr_db),
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
                    .map(|value| value.selection_psnr_db),
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
                "{LOG_BACKEND} adaptive-target2d step {step}/{total_steps} repetition={}/{} phase_step={phase_step}/{} lr={:.3e} loss={:.6} optimization_loss={:.6} max_trajectory_loss={:.6} eval_loss={} psnr_db={} grad_norm={:.3e} grad_scale={:.3e} backward_scale={:.3e} sampled_age={}/{:.0}/{} active={} reference={} topology_events={} particle_steps_per_sec={:.0} elapsed_ms={:.1}",
                repetition + 1,
                plan.repetitions,
                plan.steps_per_repetition,
                optimizer_config.learning_rate,
                stats.loss,
                stats.optimization_loss,
                stats.max_trajectory_loss,
                evaluation
                    .as_ref()
                    .map(|value| format!("{:.6}", value.mean_loss))
                    .unwrap_or_else(|| "n/a".to_string()),
                evaluation
                    .as_ref()
                    .map(|value| format!("{:.3}", value.selection_psnr_db))
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
    }

    if let Some(state) = checkpoint_state.as_mut() {
        state.write_current_batch(
            &params,
            "adaptive-target2d-final",
            total_steps,
            best_train_loss,
            best_eval_loss,
            best_eval_psnr,
        )?;
        if let Some(checkpoint) = checkpoint {
            write_adaptive_target2d_training_state(
                checkpoint,
                state,
                &optimizer,
                &pool,
                &params,
                config,
                total_steps,
            )?;
        }
    }
    // Checkpoint the final optimizer state above, but publish the parameters
    // selected by the fresh-seed PSNR gate. Returning the last update can
    // silently replace a strong initialization with a divergent trajectory.
    best_params.write_to_models(std::slice::from_mut(model))?;
    let material_scale_column = material_scale_column_stats(model);
    let mean_throughput =
        measured_particle_steps / (measured_elapsed_ms / 1_000.0).max(f64::MIN_POSITIVE);
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
        "returned_model_selection": "best_worst_checkpoint_seed_horizon_psnr",
        "material_scale_w1_l2_norm": material_scale_column.map(|stats| stats.0),
        "material_scale_w1_rms": material_scale_column.map(|stats| stats.1),
        "material_scale_w1_max_abs": material_scale_column.map(|stats| stats.2),
        "checkpoint": checkpoint_report,
    });
    if let Some(metrics) = metrics.as_object_mut() {
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
