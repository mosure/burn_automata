//! Public backend entrypoints and checkpoint orchestration.

use super::*;

    pub(crate) fn predict_conditional_row_flow_adapter(
        hyper: &E2eHyperNpa2d,
        config: &NpaConfig,
        condition: &[f32],
    ) -> AutomataResult<NpaLowRankAdapter> {
        hyper.validate()?;
        if !hyper.is_conditional_row_flow() {
            return Err(AutomataError::InvalidArgument(
                "Burn row-flow inference requires a conditional-row-flow artifact".to_string(),
            ));
        }
        let flow = hyper
            .row_flow
            .as_ref()
            .expect("validated conditional row flow");
        let expected = flow.condition_tokens * flow.condition_dims;
        if condition.len() != expected || condition.iter().any(|value| !value.is_finite()) {
            return Err(AutomataError::InvalidArgument(format!(
                "conditional row flow expected {expected} finite condition values, got {}",
                condition.len()
            )));
        }
        let device = BurnDevice::default();
        let params = BurnRowFlowParams::from_artifact(hyper, config, &device)?;
        let adapter = params.sample_adapter_batch(
            tensor3(
                condition.to_vec(),
                [1, flow.condition_tokens, flow.condition_dims],
                &device,
            ),
            config,
        );
        let values = tensor_vec(adapter.to_parameter_vector().inner())?;
        let spec = hyper.adapter_spec(config)?;
        NpaLowRankAdapter::from_parameter_vector(config, spec.rank, spec.alpha, values)
    }

    pub(crate) fn train_direct_basis_burn_dense(
        base: &mut NpaModel,
        train_examples: &mut [DirectBasisExample],
        holdout_examples: &mut [DirectBasisExample],
        train_config: DirectBasisTrainConfig,
        train_refine_config: DirectBasisTrainConfig,
        holdout_config: DirectBasisTrainConfig,
        checkpoint: Option<&Target2dBurnCheckpointConfig>,
    ) -> Result<BurnWgpuDirectBasisOutput, Box<dyn std::error::Error>> {
        if base.config.spatial_dims != 2 {
            return Err(std::io::Error::other(
                "Burn dense direct-basis training currently supports 2D",
            )
            .into());
        }
        let mut memory_snapshots = Vec::new();
        let mut gpu_memory_snapshots = Vec::new();
        memory_snapshots.push(check_process_memory_budget("start", train_config)?);
        gpu_memory_snapshots.push(check_gpu_memory_budget("start", train_config)?);
        let device = BurnDevice::default();
        let mut params = BurnBaseParams::from_model(base, &device)?;
        let mut train_adapters = train_examples
            .iter()
            .map(|example| BurnAdapterParams::from_adapter(&example.adapter, base, &device))
            .collect::<AutomataResult<Vec<_>>>()?;
        let train_targets = burn_targets(train_examples, train_config, &device)?;
        memory_snapshots.push(check_process_memory_budget(
            "after_train_tensor_cache",
            train_config,
        )?);
        gpu_memory_snapshots.push(check_gpu_memory_budget(
            "after_train_tensor_cache",
            train_config,
        )?);
        let mut checkpoint_state = checkpoint.map(BurnDenseCheckpointState::new);
        let train_phase = run_phase(
            &mut params,
            &mut train_adapters,
            &train_targets,
            train_config,
            true,
            "train",
            checkpoint_state.as_mut(),
        )?;
        memory_snapshots.push(check_process_memory_budget(
            "after_train_phase",
            train_config,
        )?);
        gpu_memory_snapshots.push(check_gpu_memory_budget("after_train_phase", train_config)?);
        params.write_to_model(base)?;
        let train_refine_phase = run_phase(
            &mut params,
            &mut train_adapters,
            &train_targets,
            train_refine_config,
            false,
            "train-refine",
            checkpoint_state.as_mut(),
        )?;
        memory_snapshots.push(check_process_memory_budget(
            "after_train_refine_phase",
            train_refine_config,
        )?);
        gpu_memory_snapshots.push(check_gpu_memory_budget(
            "after_train_refine_phase",
            train_refine_config,
        )?);
        write_adapters(train_examples, &train_adapters)?;

        let mut holdout_adapters = holdout_examples
            .iter()
            .map(|example| BurnAdapterParams::from_adapter(&example.adapter, base, &device))
            .collect::<AutomataResult<Vec<_>>>()?;
        let holdout_targets = burn_targets(holdout_examples, holdout_config, &device)?;
        memory_snapshots.push(check_process_memory_budget(
            "after_holdout_tensor_cache",
            holdout_config,
        )?);
        gpu_memory_snapshots.push(check_gpu_memory_budget(
            "after_holdout_tensor_cache",
            holdout_config,
        )?);
        let holdout_phase = run_phase(
            &mut params,
            &mut holdout_adapters,
            &holdout_targets,
            holdout_config,
            false,
            "holdout",
            checkpoint_state.as_mut(),
        )?;
        memory_snapshots.push(check_process_memory_budget(
            "after_holdout_phase",
            holdout_config,
        )?);
        gpu_memory_snapshots.push(check_gpu_memory_budget(
            "after_holdout_phase",
            holdout_config,
        )?);
        write_adapters(holdout_examples, &holdout_adapters)?;

        let particle_pool_metrics = json!({
            "enabled": train_config.use_particle_pool,
            "size": train_config.pool_size,
            "inject_seed_interval": train_config.inject_seed_interval,
            "brush_size": train_config.brush_size,
            "representation": "resident_gpu_tensors",
            "per_step_host_state_transfer": false,
        });
        let checkpoint_selection_metrics = json!({
            "mode": if train_config.use_particle_pool {
                "restore_best_reported_geometry_score_including_phase_initial_state"
            } else {
                "restore_best_reported_eval_loss_including_phase_initial_state"
            },
            "train_best_geometry_score": train_phase.best_geometry_score,
        });
        let mut metrics = json!({
            "backend": format!("{BACKEND}_e2e_rollout"),
            "device": DEVICE_LABEL,
            "objective": "target2d_pixel_splat_loss_full_image",
            "perception": "dense_compact_sph_blur_state_grad_density_grad_hybrid_moment_log_norm",
            "adapter_cache": adapter_cache_metrics(
                base,
                &params,
                &train_adapters,
                &holdout_adapters,
                &train_targets,
                &holdout_targets,
            )?,
            "train_examples": train_examples.len(),
            "holdout_examples": holdout_examples.len(),
            "train_steps": train_config.steps,
            "train_adapter_refine_steps": train_refine_config.steps,
            "holdout_adapter_steps": holdout_config.steps,
            "train_final_dense_loss": train_phase.history.last().map(|entry| entry.loss),
            "train_refine_final_dense_loss": train_refine_phase.history.last().map(|entry| entry.loss),
            "holdout_final_dense_loss": holdout_phase.history.last().map(|entry| entry.loss),
            "checkpoint_selection": checkpoint_selection_metrics,
            "optimizer": "adamw",
            "optimizer_cli_fields": "base/adapter learning_rate, weight_decay, grad_clip_norm",
            "adamw_beta1": 0.9,
            "adamw_beta2": 0.999,
            "adamw_epsilon": 1.0e-8,
            "adapter_gradient_scale": "unaverage_batch_loss_for_per_sample_adapter_adamw",
            "batching": "homogeneous_particle_count_batched_rollout_perception_splat_loss",
            "training_graph": "tbptt_chunked_rollout_state_detach",
            "tbptt_chunk_steps": train_config.tbptt_chunk_steps,
            "loss_on_final_chunk_only": train_config.loss_on_final_chunk_only,
            "particle_pool": particle_pool_metrics,
            "eval_interval": train_config.eval_interval,
            "eval_batch_size": train_config.eval_batch_size,
            "max_dense_train_particles": train_config.max_dense_train_particles,
            "max_dense_chunk_floats": train_config.max_dense_chunk_floats,
            "max_splat_chunk_floats": train_config.max_splat_chunk_floats,
            "system_memory_budget_gb": train_config.system_memory_budget_gb,
            "gpu_memory_budget_gb": train_config.gpu_memory_budget_gb,
            "process_memory_snapshots": memory_snapshots,
            "gpu_memory_snapshots": gpu_memory_snapshots,
            "evaluation": "bounded_tbptt_chunked_loss_vectors_state_detach",
            "train_mean_adapter_updates_per_sample": mean_updates_per_sample(
                train_config.steps,
                train_config.example_batch_size,
                train_examples.len(),
            ),
            "train_adapter_update_coverage": train_phase.sample_updates,
            "train_refine_mean_adapter_updates_per_sample": mean_updates_per_sample(
                train_refine_config.steps,
                train_refine_config.example_batch_size,
                train_examples.len(),
            ),
            "train_refine_adapter_update_coverage": train_refine_phase.sample_updates,
            "holdout_mean_adapter_updates_per_sample": mean_updates_per_sample(
                holdout_config.steps,
                holdout_config.example_batch_size,
                holdout_examples.len(),
            ),
            "holdout_adapter_update_coverage": holdout_phase.sample_updates,
        });
        metrics["target2d_loss_backend"] = json!(train_config.target2d_loss_backend.as_str());
        metrics["target2d_loss_backend_effective"] =
            json!(target2d_loss_backend_effective(train_config).as_str());
        metrics["perception_backend"] = json!(train_config.perception_backend.as_str());
        metrics["perception_backend_effective"] =
            json!(perception_backend_effective(train_config).as_str());
        metrics["perception_sparse_grid_effective"] = json!(
            perception_backend_effective(train_config)
                == PerceptionRolloutBackend::TiledAdjoint
                && train_config.rollout_particles >= 512
                && train_config.stopgrad_pos
        );
        metrics["perception_cube_adjoint_device_hits"] =
            json!(PERCEPTION_CUBE_ADJOINT_DEVICE_HITS.load(Ordering::Relaxed));
        metrics["perception_cube_adjoint_fallback_hits"] =
            json!(PERCEPTION_CUBE_ADJOINT_FALLBACK_HITS.load(Ordering::Relaxed));
        metrics["perception_cube_forward_device_hits"] =
            json!(PERCEPTION_CUBE_FORWARD_DEVICE_HITS.load(Ordering::Relaxed));
        metrics["perception_cube_forward_fallback_hits"] =
            json!(PERCEPTION_CUBE_FORWARD_FALLBACK_HITS.load(Ordering::Relaxed));
        metrics["perception_cube_prepared_reuse_hits"] =
            json!(PERCEPTION_CUBE_PREPARED_REUSE_HITS.load(Ordering::Relaxed));
        metrics["target2d_cube_adjoint_device_hits"] =
            json!(TARGET2D_CUBE_ADJOINT_DEVICE_HITS.load(Ordering::Relaxed));
        metrics["target2d_cube_adjoint_fallback_hits"] =
            json!(TARGET2D_CUBE_ADJOINT_FALLBACK_HITS.load(Ordering::Relaxed));
        metrics["model_checkpoints"] = checkpoint_state
            .as_ref()
            .map(BurnDenseCheckpointState::report_json)
            .unwrap_or(serde_json::Value::Null);
        let (best_train_loss, best_train_step) =
            best_training_checkpoint(train_config.steps, &train_phase, &train_refine_phase);
        Ok(BurnWgpuDirectBasisOutput {
            backend: BACKEND,
            device: DEVICE_LABEL.to_string(),
            metrics,
            history: train_phase.history,
            train_refine_history: train_refine_phase.history,
            holdout_history: holdout_phase.history,
            best_train_loss,
            best_train_step,
        })
    }

    pub(crate) fn train_e2e_rollout_burn_dense(
        base: &mut NpaModel,
        train_examples: &mut [BurnE2eRolloutExample],
        holdout_examples: &mut [BurnE2eRolloutExample],
        config: BurnE2eRolloutTrainConfig,
        initial_generator: Option<&E2eHyperNpa2d>,
    ) -> Result<BurnE2eRolloutOutput, Box<dyn std::error::Error>> {
        PERCEPTION_CUBE_ADJOINT_DEVICE_HITS.store(0, Ordering::Relaxed);
        PERCEPTION_CUBE_ADJOINT_FALLBACK_HITS.store(0, Ordering::Relaxed);
        PERCEPTION_CUBE_FORWARD_DEVICE_HITS.store(0, Ordering::Relaxed);
        PERCEPTION_CUBE_FORWARD_FALLBACK_HITS.store(0, Ordering::Relaxed);
        PERCEPTION_CUBE_PREPARED_REUSE_HITS.store(0, Ordering::Relaxed);
        TARGET2D_CUBE_ADJOINT_DEVICE_HITS.store(0, Ordering::Relaxed);
        TARGET2D_CUBE_ADJOINT_FALLBACK_HITS.store(0, Ordering::Relaxed);
        STOCHASTIC_MASK_UPLOAD_HITS.store(0, Ordering::Relaxed);
        STOCHASTIC_MASK_DEVICE_HITS.store(0, Ordering::Relaxed);
        if base.config.spatial_dims != 2 {
            return Err(std::io::Error::other(
                "Burn dense HyperNPA e2e rollout training currently supports 2D",
            )
            .into());
        }
        if train_examples.is_empty() {
            return Err(std::io::Error::other(
                "Burn dense HyperNPA e2e rollout training requires train examples",
            )
            .into());
        }
        if config.rollout_particles > config.max_dense_train_particles {
            return Err(std::io::Error::other(format!(
                "rollout_particles={} exceeds max_dense_train_particles={}",
                config.rollout_particles, config.max_dense_train_particles
            ))
            .into());
        }
        let resume_checkpoint = config
            .resume_checkpoint
            .map(|path| load_e2e_training_checkpoint(path, train_examples, &base.config, config))
            .transpose()?;
        let mut resumed_generator = None;
        if let Some(resume_path) = config.resume_checkpoint {
            let requested = Path::new(resume_path);
            let checkpoint_dir = if requested.is_dir() {
                requested
            } else {
                requested.parent().ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "training resume checkpoint has no parent directory".to_string(),
                    )
                })?
            };
            let shared_base_path = checkpoint_dir.join("current_shared_base.bpk");
            let hyper_path = checkpoint_dir.join("current_hyper_2d.bpk");
            *base = crate::import::load_manifest(&shared_base_path)?.into_model();
            resumed_generator = Some(crate::load_e2e_hyper_npa_2d(&hyper_path)?);
        }
        let initial_generator = resumed_generator.as_ref().or(initial_generator);
        let started = Instant::now();
        let device = BurnDevice::default();
        <BurnBackend as Backend>::seed(&device, config.seed);
        let npa_config = base.config.clone();
        let mut params = BurnBaseParams::from_model(base, &device)?;
        let mut base_optimizer = if let Some(checkpoint) = &resume_checkpoint {
            BurnBaseAdamWState::restore(checkpoint, &device)?
        } else {
            BurnBaseAdamWState::zeros_like(&params)
        };
        let mut generator = BurnE2eGeneratorParams::from_seed_or_artifact(
            base,
            train_examples,
            config,
            initial_generator,
            &device,
        )?;
        let mut generator_optimizer = if let Some(checkpoint) = &resume_checkpoint {
            BurnE2eGeneratorAdamWState::restore(checkpoint, &generator, &device)?
        } else {
            BurnE2eGeneratorAdamWState::new(&generator)
        };
        let mut particle_pool = if config.use_particle_pool {
            Some(if let Some(snapshot) = resume_checkpoint
                .as_ref()
                .and_then(|checkpoint| checkpoint.particle_pool.as_ref())
            {
                BurnE2eDeviceParticlePool::restore(snapshot, config, &device)?
            } else {
                BurnE2eDeviceParticlePool::new(
                    config.pool_capacity,
                    config.rollout_particles,
                    16,
                    config.pool_slots_per_example,
                    &device,
                )
            })
        } else {
            None
        };
        let train_conditions = BurnE2eConditionCache::from_examples_drain(
            train_examples,
            &device,
            config.condition_device_cache_max_bytes,
            config,
        )?;
        let holdout_conditions =
            BurnE2eConditionCache::from_examples_drain(
                holdout_examples,
                &device,
                config.condition_device_cache_max_bytes,
                config,
            )?;
        let train_condition_cache_bytes = train_conditions.feature_bytes();
        let holdout_condition_cache_bytes = holdout_conditions.feature_bytes();
        let condition_cache_bytes =
            train_condition_cache_bytes.saturating_add(holdout_condition_cache_bytes);
        let train_condition_pairwise_l2 = train_conditions.mean_pairwise_l2()?;
        let train_teacher_pairwise_l2 = train_conditions.mean_teacher_pairwise_l2()?;
        eprintln!(
            "hyper2d condition diagnostics examples={} condition_pairwise_l2={:.6} teacher_pairwise_l2={:.6}",
            train_conditions.examples,
            train_condition_pairwise_l2.unwrap_or_default(),
            train_teacher_pairwise_l2.unwrap_or_default(),
        );
        check_process_memory_budget("e2e_rollout:start", direct_config_view(config))?;
        check_gpu_memory_budget("e2e_rollout:start", direct_config_view(config))?;
        let initial_quality_validation = evaluate_e2e_rollout_quality(
            &params.detached(),
            &generator.detached(),
            &npa_config,
            train_examples,
            holdout_examples,
            &train_conditions,
            &holdout_conditions,
            BurnE2eRolloutTrainConfig {
                validation_examples: config.initial_validation_examples,
                ..config
            },
            &device,
        )?;
        if let Some(quality) = &initial_quality_validation {
            eprintln!(
                "hyper2d e2e rollout initial {} quality composited_psnr={:.3}dB p10={:.3}dB density_psnr={:.3}dB soft_iou={:.3} mean_loss={:.6e}",
                quality.split,
                quality.aggregate_composited_rgb_psnr_db,
                quality.p10_composited_rgb_psnr_db,
                quality.aggregate_density_psnr_db,
                quality.mean_density_soft_iou,
                quality.mean_total_loss,
            );
        }
        let mut quality_validation_evaluations = initial_quality_validation
            .as_ref()
            .map_or(0usize, |_| 1usize);
        let mut quality_validation_elapsed_ms = initial_quality_validation
            .as_ref()
            .map_or(0.0_f64, |quality| quality.elapsed_ms);

        let mut sampler_init_rng = StdRng::seed_from_u64(config.seed);
        let condition_batch_size =
            normalized_batch_size(config.example_batch_size, train_examples.len());
        let rollout_replicas = config.rollouts_per_example.max(1);
        let batch_size = condition_batch_size.saturating_mul(rollout_replicas);
        let mut identity_sampler = resume_checkpoint.as_ref().map_or_else(
            || {
                E2eIdentitySampler::new(
                    train_examples.len(),
                    condition_batch_size,
                    config.sampling_uniform_fraction,
                    config.sampling_priority_ema_beta,
                    config.sampling_priority_min_weight,
                    config.sampling_priority_max_weight,
                    &mut sampler_init_rng,
                )
            },
            |checkpoint| checkpoint.sampler.clone(),
        );
        let mut seed_trajectory_counts = resume_checkpoint.as_ref().map_or_else(
            || vec![0usize; train_examples.len()],
            |checkpoint| checkpoint.seed_trajectory_counts.clone(),
        );
        let completed_step = resume_checkpoint
            .as_ref()
            .map_or(0, |checkpoint| checkpoint.completed_step);
        if completed_step > 0 {
            eprintln!(
                "hyper2d resumed exact training state at completed_step={completed_step} optimizer_steps={}/{} pending_batches={}",
                base_optimizer.step,
                generator_optimizer.step,
                resume_checkpoint
                    .as_ref()
                    .map_or(0, |checkpoint| checkpoint.pending_batches.len()),
            );
        }
        let train_pixel_xy = burn_e2e_pixel_xy(config, &device);
        let projected_target_cache_bytes = e2e_target_cache_bytes(train_examples, config);
        let train_target_cache = if projected_target_cache_bytes
            <= config.target_device_cache_max_bytes
        {
            eprintln!(
                "hyper2d target cache examples={} bytes={} storage=device-resident",
                train_examples.len(), projected_target_cache_bytes
            );
            Some(burn_e2e_target_cache(
                train_examples,
                config,
                &train_pixel_xy,
                &device,
            )?)
        } else {
            eprintln!(
                "hyper2d target cache examples={} bytes={} exceeds limit={}; using CPU prefetch",
                train_examples.len(),
                projected_target_cache_bytes,
                config.target_device_cache_max_bytes,
            );
            None
        };
        check_process_memory_budget("e2e_rollout:after_target_cache", direct_config_view(config))?;
        check_gpu_memory_budget("e2e_rollout:after_target_cache", direct_config_view(config))?;
        let prefetch_depth = e2e_cpu_prefetch_depth(batch_size, config.steps);
        let mut prefetch_queue = VecDeque::with_capacity(prefetch_depth);
        for indices in resume_checkpoint
            .as_ref()
            .map(|checkpoint| checkpoint.pending_batches.as_slice())
            .unwrap_or_default()
        {
            prefetch_queue.push_back(spawn_e2e_cpu_batch_prefetch(
                train_examples,
                &train_conditions,
                indices.clone(),
                config,
                train_target_cache.is_some(),
            )?);
        }
        let mut next_prefetch_step = completed_step
            .saturating_add(prefetch_queue.len())
            .saturating_add(1);
        while next_prefetch_step <= config.steps && prefetch_queue.len() < prefetch_depth {
            let mut sample_rng = e2e_sampling_rng(config.seed, next_prefetch_step);
            prefetch_queue.push_back(spawn_e2e_cpu_batch_prefetch(
                train_examples,
                &train_conditions,
                sample_rollout_indices(
                    &mut identity_sampler,
                    rollout_replicas,
                    &mut sample_rng,
                ),
                config,
                train_target_cache.is_some(),
            )?);
            next_prefetch_step += 1;
        }
        let mut history = Vec::new();
        let mut final_loss = None;
        let mut best_checkpoint = initial_quality_validation
            .as_ref()
            .filter(|_| initial_validation_is_checkpoint_comparable(config))
            .map(|quality| BurnE2eSelectedCheckpoint {
                step: completed_step,
                train_loss: quality.mean_total_loss,
                selection_score: quality.selection_psnr_db,
                validation_contract: Some(e2e_validation_contract(
                    BurnE2eRolloutTrainConfig {
                        validation_examples: config.initial_validation_examples,
                        ..config
                    },
                )),
                holdout_mean_psnr_db: Some(quality.aggregate_composited_rgb_psnr_db),
                holdout_mean_loss: Some(quality.mean_total_loss),
                quality_validation: Some(quality.clone()),
                params: params.detached(),
                generator: generator.detached(),
            });
        let mut final_checkpoint_candidate = None::<BurnE2eSelectedCheckpoint>;
        let mut early_stop_step = None::<usize>;
        let validation_interval = config.validation_interval.max(1);
        let mut last_checkpoint_at = Instant::now();
        let mut throughput_interval_started = Instant::now();
        let mut throughput_interval_particle_steps = 0u128;
        let mut total_optimizer_particle_steps = 0u128;
        let mut measured_optimizer_training_ms = 0.0_f64;
        for step in completed_step.saturating_add(1)..=config.steps {
            let prepared_batch =
                join_e2e_cpu_batch_prefetch(prefetch_queue.pop_front().ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "HyperNPA e2e CPU prefetch queue was empty".to_string(),
                    )
                })?)?;
            let BurnE2ePreparedCpuBatch {
                indices,
                targets: prepared_targets,
                prepared_dino,
            } = prepared_batch;
            while next_prefetch_step <= config.steps && prefetch_queue.len() < prefetch_depth {
                let mut sample_rng = e2e_sampling_rng(config.seed, next_prefetch_step);
                prefetch_queue.push_back(spawn_e2e_cpu_batch_prefetch(
                    train_examples,
                    &train_conditions,
                    sample_rollout_indices(
                        &mut identity_sampler,
                        rollout_replicas,
                        &mut sample_rng,
                    ),
                    config,
                    train_target_cache.is_some(),
                )?);
                next_prefetch_step += 1;
            }
            let step_seed = config
                .seed
                .wrapping_add((step as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let report_due =
                step == config.steps || step.is_multiple_of(config.report_interval.max(1));
            let validation_due = config.validation_examples > 0
                && (step == config.steps || step.is_multiple_of(validation_interval));
            let checkpoint_due = should_write_e2e_checkpoint(step, last_checkpoint_at, config);
            let priority_due = step
                .is_multiple_of(config.sampling_priority_update_interval.max(1));
            let collect_metrics = report_due || validation_due || checkpoint_due;
            let collect_per_example_losses = collect_metrics || priority_due;
            if config.lr_schedule == E2eLrSchedule::UpstreamGrowing
                && step > 1
                && step.saturating_sub(1).is_multiple_of(10_000)
            {
                base_optimizer = BurnBaseAdamWState::zeros_like(&params);
                generator_optimizer = BurnE2eGeneratorAdamWState::new(&generator);
                if config.use_particle_pool {
                    particle_pool = Some(BurnE2eDeviceParticlePool::new(
                        config.pool_capacity,
                        config.rollout_particles,
                        16,
                        config.pool_slots_per_example,
                        &device,
                    ));
                }
                seed_trajectory_counts.fill(0);
                eprintln!(
                    "hyper2d upstream-growing repetition reset at optimizer step {step}"
                );
            }
            let lr_scale = e2e_lr_scale(config, step);
            let mut step_config = e2e_config_with_lr_scale(config, lr_scale);
            step_config.shared_base_trainable =
                config.shared_base_trainable && step >= config.shared_base_train_start_step;
            let pool_batch = if let Some(pool) = particle_pool.as_mut() {
                let mut pool_rng = e2e_pool_rng(config.seed, step);
                let seed_replacement_rows = e2e_seed_replacement_rows(
                    &indices,
                    &mut seed_trajectory_counts,
                    config.seed_trajectory_interval,
                    step,
                    config.inject_seed_interval,
                    config.seed_replacements_per_interval,
                );
                Some(pool.sample_batch(
                    &indices,
                    &mut pool_rng,
                    &seed_replacement_rows,
                    step_config.seed_scale,
                    direct_config_view(step_config),
                    &device,
                )?)
            } else {
                None
            };
            let seed_replacements = pool_batch
                .as_ref()
                .map_or(0usize, |batch| batch.seed_replacements);
            let (initial_state, pool_slots) = pool_batch
                .map(|batch| (Some((batch.x, batch.s)), Some(batch.slots)))
                .unwrap_or((None, None));
            let uncached_targets;
            let (step_targets, target_indices) = if let Some(cache) = train_target_cache.as_ref() {
                (cache.as_slice(), indices.clone())
            } else {
                uncached_targets = burn_e2e_prepared_targets_to_burn(
                    prepared_targets,
                    &train_pixel_xy,
                    &device,
                )?;
                let indices = (0..uncached_targets.len()).collect::<Vec<_>>();
                (uncached_targets.as_slice(), indices)
            };
            let step_output = train_e2e_homogeneous_step_tbptt(
                &mut params,
                &mut generator,
                &mut base_optimizer,
                &mut generator_optimizer,
                &npa_config,
                &train_conditions,
                &indices,
                prepared_dino.as_ref(),
                step_targets,
                &target_indices,
                step_config,
                step_seed,
                collect_metrics,
                collect_per_example_losses,
                initial_state,
            )?;
            let step_particle_steps = step_output.particle_steps as u128;
            throughput_interval_particle_steps = throughput_interval_particle_steps
                .saturating_add(step_particle_steps);
            total_optimizer_particle_steps =
                total_optimizer_particle_steps.saturating_add(step_particle_steps);
            let condition_identities = indices
                .chunks(rollout_replicas)
                .filter_map(|replicas| replicas.first().copied())
                .collect::<Vec<_>>();
            identity_sampler.record_trajectories(&condition_identities, rollout_replicas);
            if let Some(per_example_losses) = step_output.per_example_losses.as_deref() {
                identity_sampler.update_losses(&indices, per_example_losses);
            }
            if let (Some(pool), Some(pool_slots)) = (particle_pool.as_mut(), pool_slots) {
                pool.update_batch(&pool_slots, step_output.final_x, step_output.final_s)?;
            }
            let mut stats = step_output.history;
            if collect_metrics {
                sync_training_device(&device)?;
                let interval_elapsed = throughput_interval_started.elapsed();
                let interval_elapsed_ms = interval_elapsed.as_secs_f64() * 1_000.0;
                measured_optimizer_training_ms += interval_elapsed_ms;
                stats.particle_steps_per_sec = throughput_interval_particle_steps as f64
                    / interval_elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
                stats.dense_pair_interactions_per_sec =
                    stats.particle_steps_per_sec * config.rollout_particles as f64;
                stats.elapsed_ms = interval_elapsed_ms;
            }
            stats.step = step;
            stats.learning_rate_scale = lr_scale;
            stats.base_learning_rate = step_config.base_optimizer.learning_rate;
            stats.generator_learning_rate = step_config.generator_optimizer.learning_rate;
            stats.pool_seed_replacements = seed_replacements;
            if collect_metrics {
                final_loss = Some(stats.loss);
                let validation_config = validation_due.then(|| {
                    if step == config.steps {
                        e2e_final_validation_config(config)
                    } else {
                        config
                    }
                });
                let checkpoint_quality = if let Some(validation_config) = validation_config {
                    evaluate_e2e_rollout_quality(
                        &params.detached(),
                        &generator.detached(),
                        &npa_config,
                        train_examples,
                        holdout_examples,
                        &train_conditions,
                        &holdout_conditions,
                        validation_config,
                        &device,
                    )?
                } else {
                    None
                };
                if let Some(quality) = &checkpoint_quality {
                    quality_validation_evaluations =
                        quality_validation_evaluations.saturating_add(1);
                    quality_validation_elapsed_ms += quality.elapsed_ms;
                }
                let (holdout_mean_psnr_db, holdout_mean_loss, selection_score) =
                    if let Some(quality) = &checkpoint_quality {
                        stats.holdout_mean_psnr_db =
                            Some(quality.aggregate_composited_rgb_psnr_db);
                        stats.holdout_mean_loss = Some(quality.mean_total_loss);
                        (
                            Some(quality.aggregate_composited_rgb_psnr_db),
                            Some(quality.mean_total_loss),
                            quality.selection_psnr_db,
                        )
                    } else {
                        (None, None, -stats.loss)
                    };
                if report_due || validation_due {
                    let exposure = identity_sampler.exposure_stats();
                    eprintln!(
                        "hyper2d e2e rollout step {step}/{} loss={:.6e} task={:.6e} teacher={:.6e} flow={:.6e} self_rect={:.6e} lr_scale={:.3e} exposure_min={} exposure_mean={:.1} exposure_p90={} final_horizon_psnr={} worst_horizon_p10={} validation_due={} base_grad={:.6e} generator_grad={:.6e} particle_steps/s={:.3e} condition_ms={:.2} rollout_loss_ms={:.2} backward_ms={:.2}",
                        config.steps,
                        stats.loss,
                        stats.task_loss,
                        stats.adapter_teacher_loss,
                        stats.flow_matching_loss,
                        stats.flow_self_rectification_loss,
                        stats.learning_rate_scale,
                        exposure.min,
                        exposure.mean,
                        exposure.p90,
                        format_optional_f32(holdout_mean_psnr_db),
                        checkpoint_quality
                            .as_ref()
                            .map(|quality| format!("{:.3}", quality.selection_psnr_db))
                            .unwrap_or_else(|| "n/a".to_string()),
                        validation_due,
                        stats.base_grad_norm,
                        stats.generator_grad_norm,
                        stats.particle_steps_per_sec,
                        stats.condition_adapter_ms,
                        stats.rollout_loss_ms,
                        stats.backward_update_ms,
                    );
                }
                let mut wrote_new_best_checkpoint = false;
                if selection_score.is_finite() {
                    let candidate = BurnE2eSelectedCheckpoint {
                        step,
                        train_loss: stats.loss,
                        selection_score,
                        validation_contract: checkpoint_quality
                            .as_ref()
                            .and(validation_config.map(e2e_validation_contract)),
                        holdout_mean_psnr_db,
                        holdout_mean_loss,
                        quality_validation: checkpoint_quality.clone(),
                        params: params.detached(),
                        generator: generator.detached(),
                    };
                    if step == config.steps && checkpoint_quality.is_some() {
                        final_checkpoint_candidate = Some(candidate);
                    } else if best_checkpoint.as_ref().is_none_or(|checkpoint| {
                        comparable_selection_score_is_better(
                            candidate.validation_contract.as_ref(),
                            candidate.selection_score,
                            checkpoint.validation_contract.as_ref(),
                            checkpoint.selection_score,
                        )
                    }) {
                        wrote_new_best_checkpoint = checkpoint_quality.is_some();
                        best_checkpoint = Some(candidate);
                    }
                }
                if checkpoint_due {
                    let artifact_hashes = write_e2e_rollout_checkpoint(
                        "current",
                        step,
                        &params.detached(),
                        &generator.detached(),
                        &npa_config,
                        &train_conditions,
                        config,
                    )?;
                    write_e2e_training_checkpoint(
                        step,
                        &base_optimizer,
                        &generator_optimizer,
                        &identity_sampler,
                        &seed_trajectory_counts,
                        particle_pool.as_ref(),
                        prefetch_queue
                            .iter()
                            .map(|batch| batch.indices.clone())
                            .collect(),
                        artifact_hashes.as_ref(),
                        train_examples,
                        config,
                    )?;
                    last_checkpoint_at = Instant::now();
                }
                if wrote_new_best_checkpoint
                    && let Some(checkpoint) = &best_checkpoint
                {
                    let _ = write_e2e_rollout_checkpoint(
                        "best",
                        checkpoint.step,
                        &checkpoint.params,
                        &checkpoint.generator,
                        &npa_config,
                        &train_conditions,
                        config,
                    )?;
                }
                history.push(stats);
                throughput_interval_particle_steps = 0;
                throughput_interval_started = Instant::now();
                if step == config.steps
                    && let Some(quality) = &checkpoint_quality
                    && quality.passed
                {
                    early_stop_step = Some(step);
                    eprintln!(
                        "hyper2d e2e rollout reached validation PSNR threshold at step {step}: composited_psnr={:.3}dB p10={:.3}dB threshold={:.3}dB",
                        quality.aggregate_composited_rgb_psnr_db,
                        quality.p10_composited_rgb_psnr_db,
                        config.validation_psnr_threshold_db,
                    );
                    break;
                }
            }
        }
        let final_validation_config = e2e_final_validation_config(config);
        let final_validation_contract = e2e_validation_contract(final_validation_config);
        if let Some(final_candidate) = final_checkpoint_candidate {
            let mut selected = final_candidate;
            if let Some(mut prior_best) = best_checkpoint.take() {
                if prior_best.validation_contract.as_ref() != Some(&final_validation_contract) {
                    let quality = evaluate_e2e_rollout_quality(
                        &prior_best.params.detached(),
                        &prior_best.generator.detached(),
                        &npa_config,
                        train_examples,
                        holdout_examples,
                        &train_conditions,
                        &holdout_conditions,
                        final_validation_config,
                        &device,
                    )?;
                    if let Some(quality) = quality {
                        quality_validation_evaluations =
                            quality_validation_evaluations.saturating_add(1);
                        quality_validation_elapsed_ms += quality.elapsed_ms;
                        prior_best.selection_score = quality.selection_psnr_db;
                        prior_best.validation_contract = Some(final_validation_contract.clone());
                        prior_best.holdout_mean_psnr_db =
                            Some(quality.aggregate_composited_rgb_psnr_db);
                        prior_best.holdout_mean_loss = Some(quality.mean_total_loss);
                        prior_best.quality_validation = Some(quality);
                    }
                }
                if comparable_selection_score_is_better(
                    prior_best.validation_contract.as_ref(),
                    prior_best.selection_score,
                    selected.validation_contract.as_ref(),
                    selected.selection_score,
                ) {
                    selected = prior_best;
                }
            }
            best_checkpoint = Some(selected);
            if let Some(checkpoint) = &best_checkpoint {
                let _ = write_e2e_rollout_checkpoint(
                    "best",
                    checkpoint.step,
                    &checkpoint.params,
                    &checkpoint.generator,
                    &npa_config,
                    &train_conditions,
                    config,
                )?;
            }
        }
        let selected_checkpoint_step = best_checkpoint.as_ref().map(|checkpoint| checkpoint.step);
        let selected_checkpoint_loss =
            best_checkpoint.as_ref().map(|checkpoint| checkpoint.train_loss);
        let selected_checkpoint_score = best_checkpoint
            .as_ref()
            .map(|checkpoint| checkpoint.selection_score);
        let selected_checkpoint_holdout_psnr_db = best_checkpoint
            .as_ref()
            .and_then(|checkpoint| checkpoint.holdout_mean_psnr_db);
        let selected_checkpoint_holdout_loss = best_checkpoint
            .as_ref()
            .and_then(|checkpoint| checkpoint.holdout_mean_loss);
        let selected_checkpoint_quality_validation = best_checkpoint
            .as_ref()
            .filter(|checkpoint| {
                checkpoint.validation_contract.as_ref() == Some(&final_validation_contract)
            })
            .and_then(|checkpoint| checkpoint.quality_validation.clone());
        let selected_checkpoint_source = if selected_checkpoint_step == Some(0) {
            "initial_p10_composited_rgb_psnr"
        } else if selected_checkpoint_step == Some(config.steps)
            && selected_checkpoint_quality_validation.is_some()
        {
            "final_common_contract_p10_composited_rgb_psnr"
        } else if selected_checkpoint_holdout_psnr_db.is_some() {
            "best_common_contract_p10_composited_rgb_psnr"
        } else if selected_checkpoint_step.is_some() {
            "best_reported_loss"
        } else {
            "final"
        };
        if let Some(best_checkpoint) = best_checkpoint {
            params = best_checkpoint.params;
            generator = best_checkpoint.generator;
        }
        params.write_to_model(base)?;
        let final_quality_validation_reused_from_selected_checkpoint =
            selected_checkpoint_quality_validation.is_some();
        let quality_validation = if let Some(quality_validation) =
            selected_checkpoint_quality_validation
        {
            Some(quality_validation)
        } else {
            let quality_validation = evaluate_e2e_rollout_quality(
                &params.detached(),
                &generator.detached(),
                &npa_config,
                train_examples,
                holdout_examples,
                &train_conditions,
                &holdout_conditions,
                final_validation_config,
                &device,
            )?;
            if let Some(quality) = &quality_validation {
                quality_validation_evaluations = quality_validation_evaluations.saturating_add(1);
                quality_validation_elapsed_ms += quality.elapsed_ms;
            }
            quality_validation
        };
        let stability_validation = evaluate_e2e_rollout_stability(
            &params,
            &generator,
            &npa_config,
            holdout_examples,
            &holdout_conditions,
            config,
            &device,
        )?;
        if let Some(stability) = &stability_validation {
            eprintln!(
                "hyper2d e2e detached stability examples={} particles={} reference={} final={} aggregate_psnr_drift={:.3}dB p10_psnr_drift={:.3}dB occupancy_drift={:.6} position_overflow={:.6} state_overflow={:.6} tail_motion_ratio={:.3}",
                stability.examples,
                stability.particle_count,
                stability.reference_steps,
                stability.rollout_steps,
                stability.aggregate_composited_rgb_psnr_drift_db,
                stability.p10_composited_rgb_psnr_drift_db,
                stability.mean_render_occupancy_drift,
                stability.mean_final_position_overflow_fraction,
                stability.mean_final_state_overflow_fraction,
                stability.mean_tail_motion_ratio,
            );
        }
        let generator_hyper = generator.to_hyper(config)?;
        let (min_reported_particle_steps_per_sec, median_reported_particle_steps_per_sec, max_reported_particle_steps_per_sec) =
            reported_particle_step_speed_summary(&history);
        let (first_reported_loss, best_reported_loss, best_reported_step, final_reported_loss) =
            reported_loss_summary(&history);
        let reported_loss_delta =
            first_reported_loss.zip(final_reported_loss).map(|(first, final_loss)| final_loss - first);
        let best_reported_loss_delta =
            first_reported_loss.zip(best_reported_loss).map(|(first, best)| best - first);
        let train_condition_cache_storage = train_conditions.storage_label();
        let holdout_condition_cache_storage = holdout_conditions.storage_label();
        let condition_features_drained_from_examples =
            train_conditions.drained_cpu_features_from_examples()
                || holdout_conditions.drained_cpu_features_from_examples();
        let condition_features_uploaded_as_resident_device_cache =
            train_conditions.is_device_resident() && holdout_conditions.is_device_resident();
        let exposure = identity_sampler.exposure_stats();
        let upstream_reference_trajectories = 240_000.0_f64;
        let upstream_reference_particles = 4_096.0_f64;
        let upstream_equivalent_mean_trajectories =
            exposure.mean * config.rollout_particles as f64 / upstream_reference_particles;
        let mut metrics = serde_json::Map::new();
        metrics.insert("backend".to_string(), json!(format!("{BACKEND}_e2e_rollout")));
        metrics.insert("device".to_string(), json!(DEVICE_LABEL));
        metrics.insert(
            "objective".to_string(),
            json!(if config.task_loss_weight == 0.0 && config.flow_matching_weight > 0.0 {
                "conditional_row_rectified_flow_matching"
            } else if config.flow_matching_weight > 0.0 {
                "conditional_row_flow_plus_target2d_rollout"
            } else if config.flow_self_rectification_weight > 0.0 {
                "self_rectified_conditional_row_flow_plus_target2d_rollout"
            } else {
                "target2d_rollout_image_loss_generated_npa_residual"
            }),
        );
        metrics.insert(
            "generator_weight_warm_start".to_string(),
            json!(initial_generator.is_some()),
        );
        metrics.insert(
            "optimizer_state_resumed".to_string(),
            json!(resume_checkpoint.is_some()),
        );
        metrics.insert("resumed_from_step".to_string(), json!(completed_step));
        metrics.insert(
            "conditioner".to_string(),
            json!(generator.kind.artifact_architecture()),
        );
        metrics.insert("adapter_rank".to_string(), json!(config.adapter_rank));
        metrics.insert("adapter_alpha".to_string(), json!(config.adapter_alpha));
        metrics.insert(
            "adapter_parameterization".to_string(),
            json!(if generator.kind == E2eHyperGeneratorKind::ConditionalRowFlow {
                E2E_HYPER_ADAPTER_DENSE_ROW_RESIDUAL
            } else if config.canonical_full_rank_lora {
                E2E_HYPER_ADAPTER_CANONICAL_FULL_RANK
            } else {
                E2E_HYPER_ADAPTER_FACTORIZED
            }),
        );
        metrics.insert(
            "adapter_effective_output_dims".to_string(),
            json!(if config.canonical_full_rank_lora {
                crate::hyper::adapter_layout::CanonicalFullRankLora2d::new(
                    &npa_config,
                    config.adapter_rank,
                    config.adapter_alpha,
                )?
                .trainable_parameters
            } else {
                generator.output_dims
            }),
        );
        metrics.insert(
            "adapter_chunk_size".to_string(),
            json!(generator.adapter_chunk_size),
        );
        metrics.insert(
            "generator_hidden_dims".to_string(),
            json!(config.generator_hidden_dims),
        );
        metrics.insert(
            "token_attention_heads".to_string(),
            json!(config.token_attention_heads),
        );
        metrics.insert(
            "generator_sample_steps".to_string(),
            json!(config.generator_sample_steps),
        );
        metrics.insert(
            "generator_layers".to_string(),
            json!(config.generator_layers),
        );
        metrics.insert(
            "generator_ffn_dims".to_string(),
            json!(config.generator_ffn_dims),
        );
        metrics.insert(
            "generator_source_seed".to_string(),
            json!(config.generator_source_seed),
        );
        metrics.insert(
            "flow_matching_weight".to_string(),
            json!(config.flow_matching_weight),
        );
        metrics.insert(
            "flow_self_rectification_weight".to_string(),
            json!(config.flow_self_rectification_weight),
        );
        metrics.insert("generator_output_dims".to_string(), json!(generator.output_dims));
        metrics.insert(
            "generator_output_scale".to_string(),
            json!(config.generator_output_scale),
        );
        metrics.insert(
            "generator_condition_init_scale".to_string(),
            json!(config.generator_condition_init_scale),
        );
        metrics.insert(
            "generator_output_init_scale".to_string(),
            json!(config.generator_output_init_scale),
        );
        metrics.insert(
            "condition_token_count".to_string(),
            json!(train_conditions.token_count),
        );
        metrics.insert(
            "condition_embed_dims".to_string(),
            json!(train_conditions.embed_dims),
        );
        metrics.insert(
            "train_condition_cache_bytes_f32".to_string(),
            json!(train_condition_cache_bytes),
        );
        metrics.insert(
            "holdout_condition_cache_bytes_f32".to_string(),
            json!(holdout_condition_cache_bytes),
        );
        metrics.insert(
            "condition_cache_bytes_f32".to_string(),
            json!(condition_cache_bytes),
        );
        metrics.insert(
            "condition_device_cache_max_bytes".to_string(),
            json!(config.condition_device_cache_max_bytes),
        );
        metrics.insert(
            "target_device_cache".to_string(),
            json!({
                "resident": train_target_cache.is_some(),
                "projected_bytes_f32": projected_target_cache_bytes,
                "max_bytes": config.target_device_cache_max_bytes,
                "step_target_upload": train_target_cache.is_none(),
            }),
        );
        metrics.insert(
            "condition_cache_gib_f32".to_string(),
            json!(bytes_to_gib(condition_cache_bytes as u64)),
        );
        metrics.insert(
            "train_condition_cache_storage".to_string(),
            json!(train_condition_cache_storage),
        );
        metrics.insert(
            "holdout_condition_cache_storage".to_string(),
            json!(holdout_condition_cache_storage),
        );
        metrics.insert(
            "cpu_condition_features_drained_from_examples".to_string(),
            json!(condition_features_drained_from_examples),
        );
        metrics.insert(
            "cpu_condition_features_uploaded_as_resident_device_cache".to_string(),
            json!(condition_features_uploaded_as_resident_device_cache),
        );
        metrics.insert("train_examples".to_string(), json!(train_examples.len()));
        metrics.insert("holdout_examples".to_string(), json!(holdout_examples.len()));
        metrics.insert("steps".to_string(), json!(config.steps));
        metrics.insert(
            "example_batch_size".to_string(),
            json!(config.example_batch_size),
        );
        metrics.insert(
            "example_batch_size_effective".to_string(),
            json!(batch_size),
        );
        metrics.insert(
            "example_batch_semantics".to_string(),
            json!("independent_image_conditioned_rollouts"),
        );
        metrics.insert(
            "example_batch_parallel_samples".to_string(),
            json!(batch_size.min(train_examples.len())),
        );
        metrics.insert(
            "identity_sampling".to_string(),
            json!({
                "uniform_fraction": config.sampling_uniform_fraction,
                "priority_fraction": 1.0 - config.sampling_uniform_fraction,
                "priority_ema_beta": config.sampling_priority_ema_beta,
                "priority_min_weight": config.sampling_priority_min_weight,
                "priority_max_weight": config.sampling_priority_max_weight,
                "priority_update_interval": config.sampling_priority_update_interval,
                "trajectory_exposure": exposure,
                "upstream_reference_trajectories_per_identity": upstream_reference_trajectories,
                "upstream_reference_particles_per_trajectory": upstream_reference_particles,
                "upstream_equivalent_mean_4096_particle_trajectories_per_identity": upstream_equivalent_mean_trajectories,
                "upstream_compute_exposure_fraction": upstream_equivalent_mean_trajectories / upstream_reference_trajectories,
            }),
        );
        metrics.insert("rollout_particles".to_string(), json!(config.rollout_particles));
        metrics.insert(
            "rollout_step_min".to_string(),
            json!(config.rollout_step_min),
        );
        metrics.insert("rollout_steps".to_string(), json!(config.rollout_steps));
        metrics.insert(
            "tbptt_chunk_steps".to_string(),
            json!(config.tbptt_chunk_steps),
        );
        metrics.insert(
            "validation_interval".to_string(),
            json!(config.validation_interval),
        );
        metrics.insert(
            "quality_validation_evaluations".to_string(),
            json!(quality_validation_evaluations),
        );
        metrics.insert(
            "quality_validation_elapsed_ms".to_string(),
            json!(quality_validation_elapsed_ms),
        );
        metrics.insert(
            "stability_validation_contract".to_string(),
            json!({
                "examples": config.stability_examples,
                "particles": config.stability_particles,
                "reference_steps": config.stability_reference_steps,
                "steps": config.stability_steps,
                "tail_steps": config.stability_tail_steps,
                "split": "holdout",
                "condition_mode": "generated-adapter-only",
                "autodiff_graph_retained": false,
            }),
        );
        metrics.insert(
            "loss_on_final_chunk_only".to_string(),
            json!(config.loss_on_final_chunk_only),
        );
        metrics.insert(
            "tbptt_loss_mode".to_string(),
            json!(config.tbptt_loss_mode.as_str()),
        );
        metrics.insert(
            "tbptt_intermediate_loss_weight".to_string(),
            json!(config.tbptt_intermediate_loss_weight),
        );
        metrics.insert(
            "tbptt_final_loss_weight".to_string(),
            json!(config.tbptt_final_loss_weight),
        );
        metrics.insert(
            "credit_assignment".to_string(),
            json!(config.credit_assignment.as_str()),
        );
        metrics.insert(
            "task_loss_weight".to_string(),
            json!(config.task_loss_weight),
        );
        metrics.insert(
            "adapter_teacher_weight".to_string(),
            json!(config.adapter_teacher_weight),
        );
        metrics.insert(
            "adapter_teacher_objective".to_string(),
            json!(config.adapter_teacher_objective.as_str()),
        );
        metrics.insert(
            "adapter_teacher_probe_rollout_steps".to_string(),
            json!(config.adapter_teacher_probe_rollout_steps),
        );
        metrics.insert(
            "base_per_parameter_grad_normalization".to_string(),
            json!(config.base_per_parameter_grad_normalization),
        );
        metrics.insert(
            "generator_per_parameter_grad_normalization".to_string(),
            json!(config.generator_per_parameter_grad_normalization),
        );
        metrics.insert(
            "sample_id_table_grad_normalization".to_string(),
            json!(if config.generator_kind == E2eHyperGeneratorKind::SampleIdTable
                && config.generator_per_parameter_grad_normalization
            {
                "per-adapter-component-per-identity"
            } else {
                "not-applicable"
            }),
        );
        metrics.insert(
            "max_full_bptt_particle_steps".to_string(),
            json!(config.max_full_bptt_particle_steps),
        );
        metrics.insert(
            "pre_rollout_steps".to_string(),
            json!(config.pre_rollout_steps),
        );
        metrics.insert(
            "particle_pool".to_string(),
            json!({
                "enabled": config.use_particle_pool,
                "capacity": config.pool_capacity,
                "storage_bytes_f32": config.pool_capacity
                    .saturating_mul(config.rollout_particles)
                    .saturating_mul(npa_config.state_dims.saturating_add(2))
                    .saturating_mul(std::mem::size_of::<f32>()),
                "stored_slots_per_example": config.pool_slots_per_example,
                "rollouts_sampled_per_example": config.rollouts_per_example,
                "inject_seed_interval": config.inject_seed_interval,
                "seed_replacements_per_interval": config.seed_replacements_per_interval,
                "seed_trajectory_interval_per_identity": config.seed_trajectory_interval,
                "mode": "bounded-sample-replica-keyed-device-state-pool",
                "step_readback": false,
                "position_persistence_clamp": [-1.0, 1.0],
                "state_finite_safety_clamp": [-32.0, 32.0],
            }),
        );
        metrics.insert(
            "target2d_loss_backend".to_string(),
            json!(config.target2d_loss_backend.as_str()),
        );
        metrics.insert(
            "target2d_loss_backend_effective".to_string(),
            json!(target2d_loss_backend_effective(direct_config_view(config)).as_str()),
        );
        metrics.insert(
            "perception_backend".to_string(),
            json!(config.perception_backend.as_str()),
        );
        metrics.insert(
            "perception_backend_effective".to_string(),
            json!(perception_backend_effective(direct_config_view(config)).as_str()),
        );
        metrics.insert(
            "perception_cube_adjoint_device_hits".to_string(),
            json!(PERCEPTION_CUBE_ADJOINT_DEVICE_HITS.load(Ordering::Relaxed)),
        );
        metrics.insert(
            "perception_cube_adjoint_fallback_hits".to_string(),
            json!(PERCEPTION_CUBE_ADJOINT_FALLBACK_HITS.load(Ordering::Relaxed)),
        );
        metrics.insert(
            "perception_cube_forward_device_hits".to_string(),
            json!(PERCEPTION_CUBE_FORWARD_DEVICE_HITS.load(Ordering::Relaxed)),
        );
        metrics.insert(
            "perception_cube_forward_fallback_hits".to_string(),
            json!(PERCEPTION_CUBE_FORWARD_FALLBACK_HITS.load(Ordering::Relaxed)),
        );
        metrics.insert(
            "perception_cube_prepared_reuse_hits".to_string(),
            json!(PERCEPTION_CUBE_PREPARED_REUSE_HITS.load(Ordering::Relaxed)),
        );
        let retained_perception_state_bytes_per_step = batch_size
            .saturating_mul(config.rollout_particles)
            .saturating_mul(npa_config.state_dims.saturating_mul(2).saturating_add(4))
            .saturating_mul(std::mem::size_of::<f32>());
        metrics.insert(
            "perception_cube_prepared_vjp".to_string(),
            json!({
                "mode": "retained_raw_state_gradient_and_correction_inverse",
                "additional_bytes_per_rollout_step_f32": retained_perception_state_bytes_per_step,
                "additional_bytes_at_max_full_bptt_horizon_f32": retained_perception_state_bytes_per_step
                    .saturating_mul(config.rollout_steps),
                "neighbor_recompute_in_backward": false,
            }),
        );
        metrics.insert(
            "target2d_cube_adjoint_device_hits".to_string(),
            json!(TARGET2D_CUBE_ADJOINT_DEVICE_HITS.load(Ordering::Relaxed)),
        );
        metrics.insert(
            "target2d_cube_adjoint_fallback_hits".to_string(),
            json!(TARGET2D_CUBE_ADJOINT_FALLBACK_HITS.load(Ordering::Relaxed)),
        );
        metrics.insert(
            "stochastic_mask_upload_hits".to_string(),
            json!(STOCHASTIC_MASK_UPLOAD_HITS.load(Ordering::Relaxed)),
        );
        metrics.insert(
            "stochastic_mask_device_hits".to_string(),
            json!(STOCHASTIC_MASK_DEVICE_HITS.load(Ordering::Relaxed)),
        );
        metrics.insert(
            "stochastic_mask_backend_effective".to_string(),
            json!("device-random-training-host-seeded-eval"),
        );
        metrics.insert(
            "max_dense_train_particles".to_string(),
            json!(config.max_dense_train_particles),
        );
        metrics.insert(
            "training_graph".to_string(),
            json!(if config.task_loss_weight == 0.0 && config.flow_matching_weight > 0.0 {
                "condition_tokens_to_row_velocity_no_particle_rollout"
            } else {
                match config.credit_assignment {
                    E2eCreditAssignment::FullBptt => {
                        "generated_adapter_fixed_full_rollout_single_loss_single_update"
                    }
                    E2eCreditAssignment::DetachedTbptt => {
                        "generated_adapter_tbptt_chunked_rollout_state_detach_with_optional_sample_keyed_pool"
                    }
                }
            }),
        );
        let measured_optimizer_seconds = measured_optimizer_training_ms / 1_000.0;
        let flow_examples = exposure.total;
        if config.flow_matching_weight > 0.0 && measured_optimizer_seconds > 0.0 {
            let flow_examples_per_sec = flow_examples as f64 / measured_optimizer_seconds;
            metrics.insert("flow_examples".to_string(), json!(flow_examples));
            metrics.insert(
                "flow_examples_per_sec".to_string(),
                json!(flow_examples_per_sec),
            );
            metrics.insert(
                "flow_valid_row_values_per_sec".to_string(),
                json!(flow_examples_per_sec
                    * NpaParameterRowLayout2d::new(&npa_config).parameter_count() as f64),
            );
        }
        metrics.insert(
            "shared_base_trainable".to_string(),
            json!(config.shared_base_trainable),
        );
        metrics.insert(
            "shared_base_train_start_step".to_string(),
            json!(config.shared_base_train_start_step),
        );
        metrics.insert("lr_schedule".to_string(), json!(config.lr_schedule.as_str()));
        metrics.insert(
            "lr_warmup_steps".to_string(),
            json!(config.lr_warmup_steps),
        );
        metrics.insert("min_lr_scale".to_string(), json!(config.min_lr_scale));
        metrics.insert(
            "selected_checkpoint_source".to_string(),
            json!(selected_checkpoint_source),
        );
        metrics.insert(
            "selected_checkpoint_step".to_string(),
            json!(selected_checkpoint_step),
        );
        metrics.insert(
            "selected_checkpoint_loss".to_string(),
            json!(selected_checkpoint_loss),
        );
        metrics.insert(
            "selected_checkpoint_score".to_string(),
            json!(selected_checkpoint_score),
        );
        metrics.insert(
            "selected_checkpoint_holdout_psnr_db".to_string(),
            json!(selected_checkpoint_holdout_psnr_db),
        );
        metrics.insert(
            "selected_checkpoint_holdout_loss".to_string(),
            json!(selected_checkpoint_holdout_loss),
        );
        metrics.insert(
            "final_quality_validation_reused_from_selected_checkpoint".to_string(),
            json!(final_quality_validation_reused_from_selected_checkpoint),
        );
        metrics.insert("early_stop_step".to_string(), json!(early_stop_step));
        metrics.insert(
            "early_stop_reason".to_string(),
            json!(early_stop_step.map(|_| "validation_psnr_threshold")),
        );
        metrics.insert(
            "min_reported_particle_steps_per_sec".to_string(),
            json!(min_reported_particle_steps_per_sec),
        );
        metrics.insert(
            "median_reported_particle_steps_per_sec".to_string(),
            json!(median_reported_particle_steps_per_sec),
        );
        metrics.insert(
            "max_reported_particle_steps_per_sec".to_string(),
            json!(max_reported_particle_steps_per_sec),
        );
        metrics.insert(
            "total_optimizer_particle_steps".to_string(),
            json!(total_optimizer_particle_steps),
        );
        metrics.insert(
            "measured_optimizer_training_ms".to_string(),
            json!(measured_optimizer_training_ms),
        );
        metrics.insert(
            "measured_optimizer_particle_steps_per_sec".to_string(),
            json!(total_optimizer_particle_steps as f64
                / (measured_optimizer_training_ms / 1_000.0).max(f64::MIN_POSITIVE)),
        );
        metrics.insert("first_reported_loss".to_string(), json!(first_reported_loss));
        metrics.insert("best_reported_loss".to_string(), json!(best_reported_loss));
        metrics.insert("best_reported_step".to_string(), json!(best_reported_step));
        metrics.insert("final_reported_loss".to_string(), json!(final_reported_loss));
        metrics.insert("reported_loss_delta".to_string(), json!(reported_loss_delta));
        metrics.insert(
            "best_reported_loss_delta".to_string(),
            json!(best_reported_loss_delta),
        );
        metrics.insert(
            "initial_quality_validation".to_string(),
            json!(initial_quality_validation.clone()),
        );
        metrics.insert(
            "quality_validation".to_string(),
            json!(quality_validation.clone()),
        );
        metrics.insert(
            "stability_validation".to_string(),
            json!(stability_validation.clone()),
        );
        metrics.insert(
            "elapsed_ms".to_string(),
            json!(started.elapsed().as_secs_f64() * 1000.0),
        );
        let metrics = serde_json::Value::Object(metrics);
        Ok(BurnE2eRolloutOutput {
            backend: format!("{BACKEND}_e2e_rollout"),
            device: DEVICE_LABEL.to_string(),
            metrics,
            history,
            final_loss,
            generator: generator_hyper,
            quality_validation,
            stability_validation,
        })
    }

    pub(super) fn should_write_e2e_checkpoint(
        step: usize,
        last_checkpoint_at: Instant,
        config: BurnE2eRolloutTrainConfig,
    ) -> bool {
        config.checkpoint_dir.is_some()
            && (step == config.steps
                || step.is_multiple_of(config.checkpoint_interval_steps.max(1))
                || last_checkpoint_at.elapsed().as_secs()
                    >= config.checkpoint_interval_seconds.max(1) as u64)
    }

    pub(super) fn initial_validation_is_checkpoint_comparable(config: BurnE2eRolloutTrainConfig) -> bool {
        config.initial_validation_examples == config.validation_examples
    }

    pub(super) fn e2e_validation_contract(
        config: BurnE2eRolloutTrainConfig,
    ) -> BurnE2eValidationContract {
        let mut horizons = config.validation_horizons
            [..config.validation_horizon_count.min(config.validation_horizons.len())]
            .iter()
            .copied()
            .filter(|steps| *steps > 0)
            .collect::<Vec<_>>();
        horizons.push(config.validation_steps.max(1));
        horizons.sort_unstable();
        horizons.dedup();
        BurnE2eValidationContract {
            examples: config.validation_examples,
            particles: config.validation_particles,
            horizons,
            selection_horizon_min_steps: config.validation_selection_horizon_min_steps,
        }
    }

    pub(super) fn comparable_selection_score_is_better(
        candidate_contract: Option<&BurnE2eValidationContract>,
        candidate_score: f32,
        incumbent_contract: Option<&BurnE2eValidationContract>,
        incumbent_score: f32,
    ) -> bool {
        if !candidate_score.is_finite() {
            return false;
        }
        match (candidate_contract, incumbent_contract) {
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (candidate, incumbent) if candidate == incumbent => candidate_score > incumbent_score,
            (Some(_), Some(_)) => false,
            (None, None) => unreachable!("equal optional contracts matched above"),
        }
    }

    pub(super) fn e2e_final_validation_config(
        mut config: BurnE2eRolloutTrainConfig,
    ) -> BurnE2eRolloutTrainConfig {
        config.validation_examples = config.final_validation_examples;
        config.validation_particles = config.final_validation_particles;
        config.validation_steps = config.final_validation_steps;
        config.validation_horizons = config.final_validation_horizons;
        config.validation_horizon_count = config.final_validation_horizon_count;
        config.validation_selection_horizon_min_steps =
            config.final_validation_selection_horizon_min_steps;
        config
    }

    pub(super) struct E2eRolloutCheckpointHashes {
        shared_base_sha256: String,
        hyper_sha256: String,
    }

    pub(super) fn write_e2e_rollout_checkpoint(
        label: &str,
        step: usize,
        params: &BurnBaseParams,
        generator: &BurnE2eGeneratorParams,
        npa_config: &NpaConfig,
        train_conditions: &BurnE2eConditionCache,
        config: BurnE2eRolloutTrainConfig,
    ) -> AutomataResult<Option<E2eRolloutCheckpointHashes>> {
        let Some(checkpoint_dir) = config.checkpoint_dir else {
            return Ok(None);
        };
        let checkpoint_dir = Path::new(checkpoint_dir);
        fs::create_dir_all(checkpoint_dir)?;
        let shared_base_output = checkpoint_dir.join(format!("{label}_shared_base.bpk"));
        let hyper_output = checkpoint_dir.join(format!("{label}_hyper_2d.bpk"));
        let metadata_output = checkpoint_dir.join(format!("{label}_metadata.json"));
        let source =
            format!("checkpoint:{BACKEND}:hyper2d-e2e-rollout:label={label}:step={step}");

        let mut model = NpaModel {
            config: npa_config.clone(),
            weights: NpaWeights::zeros(npa_config),
        };
        params.write_to_model(&mut model)?;
        let manifest = BpkModelManifest::from_model(
            &model,
            burn_automata_kernels::HashGridConfig::growing_2d(),
            Some(source.clone()),
        );
        let shared_base_sha256 = crate::import::save_manifest(&shared_base_output, &manifest)?
            .ok_or_else(|| {
                AutomataError::InvalidFormat(
                    "HyperNPA checkpoint base path did not use the BPK format".to_string(),
                )
            })?;

        let mut hyper = generator.to_hyper(config)?;
        hyper.condition_encoder = config.checkpoint_condition_encoder.map(str::to_string);
        hyper.condition_token_count = Some(train_conditions.token_count);
        hyper.condition_embed_dims = Some(train_conditions.embed_dims);
        hyper.condition_token_grid_width = Some(config.dino_token_grid_width);
        hyper.condition_token_grid_height = Some(config.dino_token_grid_height);
        hyper.shared_base_sha256 = Some(shared_base_sha256.clone());
        let hyper_sha256 = save_e2e_hyper_npa_2d(&hyper_output, &hyper)?;

        let metadata = json!({
            "label": label,
            "step": step,
            "backend": format!("{BACKEND}_e2e_rollout"),
            "device": DEVICE_LABEL,
            "source": source,
            "shared_base_output": shared_base_output,
            "shared_base_sha256": shared_base_sha256,
            "hyper_output": hyper_output,
            "hyper_sha256": hyper_sha256,
            "condition_token_count": train_conditions.token_count,
            "condition_embed_dims": train_conditions.embed_dims,
            "condition_token_grid_width": config.dino_token_grid_width,
            "condition_token_grid_height": config.dino_token_grid_height,
            "condition_image_size": config.dino_image_size,
            "condition_alpha_mode": "composite-white",
            "condition_rgb_channels": config.dino_rgb_channels,
            "condition_rgb_channel_scale": config.dino_rgb_channel_scale,
            "condition_alpha_channel": config.dino_alpha_channel,
            "condition_alpha_channel_scale": config.dino_alpha_channel_scale,
            "condition_l2_normalize_features": config.dino_l2_normalize_features,
            "condition_resize_mode": "stretch",
        });
        fs::write(&metadata_output, serde_json::to_vec_pretty(&metadata)?)?;
        eprintln!(
            "hyper2d e2e rollout checkpoint {label} step {step} wrote {} and {}",
            shared_base_output.display(),
            hyper_output.display(),
        );
        Ok(Some(E2eRolloutCheckpointHashes {
            shared_base_sha256,
            hyper_sha256,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn e2e_training_contract_sha256(
        train_examples: &[BurnE2eRolloutExample],
        config: BurnE2eRolloutTrainConfig,
    ) -> String {
        let mut hasher = Sha256::new();
        for example in train_examples {
            hasher.update(example.slug.as_bytes());
            hasher.update([0]);
        }
        hasher.update(
            format!(
                "backend={BACKEND};steps={};batch={};replicas={};particles={};step_min={};step_max={};update_prob={:.9};seed={};seed_mode={:?};pool={}:{}:{};seed_interval={};brush={:.9};loss={}:{}:{:.9}:{:.9}:{:.9}:{:.9}:{:.9}:{:.9}:{:.9};adapter={}:{:.9}:{}:{};generator={}:{}:{}:{:.9}:{:.9}:{:.9};optimizer={:?}:{:.9}:{:.9}:{:.9}:{:.9}:{:.9}:{:.9}:{:.9}:{:.9}:{:.9};condition={}:{}:{}:{}:{}:{}",
                config.steps,
                config.example_batch_size,
                config.rollouts_per_example,
                config.rollout_particles,
                config.rollout_step_min,
                config.rollout_steps,
                config.update_prob,
                config.seed,
                config.seed_mode,
                config.use_particle_pool,
                config.pool_capacity,
                config.pool_slots_per_example,
                config.seed_trajectory_interval,
                config.brush_size,
                config.loss_config.image_size,
                config.loss_config.center,
                config.loss_config.splat_loss_weight,
                config.loss_config.color_loss_weight,
                config.loss_config.density_loss_weight,
                config.loss_config.composited_rgb_loss_weight,
                config.loss_config.displacement_regularizer_weight,
                config.loss_config.overflow_regularizer_weight,
                config.loss_config.bound_regularizer_weight,
                config.adapter_rank,
                config.adapter_alpha,
                config.canonical_full_rank_lora,
                config.generator_kind.artifact_architecture(),
                config.generator_hidden_dims,
                config.adapter_chunk_size,
                config.generator_sample_steps,
                config.generator_output_scale,
                config.generator_condition_init_scale,
                config.generator_output_init_scale,
                config.lr_schedule,
                config.base_optimizer.learning_rate,
                config.generator_optimizer.learning_rate,
                config.base_optimizer.weight_decay,
                config.generator_optimizer.weight_decay,
                config.base_optimizer.grad_clip_norm,
                config.generator_optimizer.grad_clip_norm,
                config.base_optimizer.beta1,
                config.base_optimizer.beta2,
                config.base_optimizer.epsilon,
                config.dino_image_size,
                config.dino_token_grid_width,
                config.dino_token_grid_height,
                config.dino_rgb_channels,
                config.dino_alpha_channel,
                config.condition_device_cache_max_bytes,
            )
            .as_bytes(),
        );
        hasher.update(
            format!(
                "credit={:?}:{}:{:?}:{:.9}:{:.9}:{};warmup={};sampling={:.9}:{:.9}:{:.9}:{:.9}:{};pool={}:{}:{};seed_scale={:.9};pre_rollout={};dynamics={:.9}:{:.9};backends={:?}:{:?};grad_norm={}:{}:{};objectives={:.9}:{:.9}:{:?}:{}:{:.9}:{:.9};base={}:{};flow={}:{}:{}:{}:{}:{:.9};dino={}:{:.9}:{:.9}:{};spatial={}:{:.9}:{:.9}:{}",
                config.credit_assignment,
                config.tbptt_chunk_steps,
                config.tbptt_loss_mode,
                config.tbptt_intermediate_loss_weight,
                config.tbptt_final_loss_weight,
                config.max_full_bptt_particle_steps,
                config.lr_warmup_steps,
                config.sampling_uniform_fraction,
                config.sampling_priority_ema_beta,
                config.sampling_priority_min_weight,
                config.sampling_priority_max_weight,
                config.sampling_priority_update_interval,
                config.inject_seed_interval,
                config.seed_replacements_per_interval,
                config.seed_trajectory_interval,
                config.seed_scale,
                config.pre_rollout_steps,
                config.grid_eps,
                config.motion_scale,
                config.target2d_loss_backend,
                config.perception_backend,
                config.per_parameter_grad_normalization,
                config.base_per_parameter_grad_normalization,
                config.generator_per_parameter_grad_normalization,
                config.task_loss_weight,
                config.adapter_teacher_weight,
                config.adapter_teacher_objective,
                config.adapter_teacher_probe_rollout_steps,
                config.flow_matching_weight,
                config.flow_self_rectification_weight,
                config.shared_base_trainable,
                config.shared_base_train_start_step,
                config.generator_layers,
                config.generator_ffn_dims,
                config.token_attention_heads,
                config.softmax_token_attention,
                config.generator_source_seed,
                config.generator_init_scale,
                config.dino_l2_normalize_features,
                config.dino_rgb_channel_scale,
                config.dino_alpha_channel_scale,
                config.dino_batch_size,
                config.spatial_condition_control,
                config.spatial_condition_control_scale,
                config.spatial_condition_control_sigma,
                config.spatial_condition_state_control,
            )
            .as_bytes(),
        );
        format!("{:x}", hasher.finalize())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn write_e2e_training_checkpoint(
        step: usize,
        base_optimizer: &BurnBaseAdamWState,
        generator_optimizer: &BurnE2eGeneratorAdamWState,
        sampler: &E2eIdentitySampler,
        seed_trajectory_counts: &[usize],
        particle_pool: Option<&BurnE2eDeviceParticlePool>,
        pending_batches: Vec<Vec<usize>>,
        artifact_hashes: Option<&E2eRolloutCheckpointHashes>,
        train_examples: &[BurnE2eRolloutExample],
        config: BurnE2eRolloutTrainConfig,
    ) -> AutomataResult<()> {
        let Some(checkpoint_dir) = config.checkpoint_dir else {
            return Ok(());
        };
        let mut optimizer_tensors = base_optimizer.snapshots()?;
        optimizer_tensors.extend(generator_optimizer.snapshots()?);
        let checkpoint = E2eTrainingCheckpoint {
            version: E2E_TRAINING_CHECKPOINT_VERSION,
            backend: BACKEND.to_string(),
            contract_sha256: e2e_training_contract_sha256(train_examples, config),
            shared_base_sha256: artifact_hashes
                .map(|hashes| hashes.shared_base_sha256.clone())
                .unwrap_or_default(),
            hyper_sha256: artifact_hashes
                .map(|hashes| hashes.hyper_sha256.clone())
                .unwrap_or_default(),
            completed_step: step,
            train_examples: train_examples.len(),
            rollout_particles: config.rollout_particles,
            rollout_step_min: config.rollout_step_min,
            rollout_steps: config.rollout_steps,
            rollouts_per_example: config.rollouts_per_example,
            base_optimizer_step: base_optimizer.step,
            generator_optimizer_step: generator_optimizer.step,
            optimizer_tensors,
            sampler: sampler.clone(),
            seed_trajectory_counts: seed_trajectory_counts.to_vec(),
            pending_batches,
            particle_pool: particle_pool.map(BurnE2eDeviceParticlePool::snapshot).transpose()?,
        };
        let path = Path::new(checkpoint_dir).join("current_training_state.mpk");
        checkpoint.write_atomic(&path)?;
        eprintln!(
            "hyper2d e2e rollout checkpoint current step {step} wrote {}",
            path.display()
        );
        Ok(())
    }

    pub(super) fn load_e2e_training_checkpoint(
        path: &str,
        train_examples: &[BurnE2eRolloutExample],
        npa_config: &NpaConfig,
        config: BurnE2eRolloutTrainConfig,
    ) -> AutomataResult<E2eTrainingCheckpoint> {
        let path = Path::new(path);
        let path = if path.is_dir() {
            path.join("current_training_state.mpk")
        } else {
            path.to_path_buf()
        };
        let checkpoint = E2eTrainingCheckpoint::read(&path)?;
        let checkpoint_dir = path.parent().ok_or_else(|| {
            AutomataError::InvalidArgument(format!(
                "training checkpoint {} has no parent directory",
                path.display()
            ))
        })?;
        if !checkpoint.shared_base_sha256.is_empty() {
            let artifact = checkpoint_dir.join("current_shared_base.bpk");
            let actual = crate::import::bpk_payload_sha256(&fs::read(&artifact)?)?;
            if actual != checkpoint.shared_base_sha256 {
                return Err(AutomataError::InvalidFormat(format!(
                    "training checkpoint base BPK hash mismatch for {}; state={} artifact={actual}",
                    artifact.display(),
                    checkpoint.shared_base_sha256,
                )));
            }
        }
        if !checkpoint.hyper_sha256.is_empty() {
            let artifact = checkpoint_dir.join("current_hyper_2d.bpk");
            let actual = crate::hyper::e2e::e2e_hyper_bpk_payload_sha256(&fs::read(&artifact)?)?;
            if actual != checkpoint.hyper_sha256 {
                return Err(AutomataError::InvalidFormat(format!(
                    "training checkpoint hyper BPK hash mismatch for {}; state={} artifact={actual}",
                    artifact.display(),
                    checkpoint.hyper_sha256,
                )));
            }
        }
        if checkpoint.backend != BACKEND
            || (!checkpoint.contract_sha256.is_empty()
                && checkpoint.contract_sha256
                    != e2e_training_contract_sha256(train_examples, config))
            || checkpoint.train_examples != train_examples.len()
            || checkpoint.rollout_particles != config.rollout_particles
            || checkpoint.rollout_step_min != config.rollout_step_min
            || checkpoint.rollout_steps != config.rollout_steps
            || checkpoint.rollouts_per_example != config.rollouts_per_example
            || checkpoint.seed_trajectory_counts.len() != train_examples.len()
            || npa_config.spatial_dims != 2
        {
            return Err(AutomataError::InvalidArgument(format!(
                "training checkpoint {} is incompatible with backend/data/rollout config",
                path.display()
            )));
        }
        if checkpoint.completed_step >= config.steps {
            return Err(AutomataError::InvalidArgument(format!(
                "training checkpoint completed step {} is not below configured total steps {}",
                checkpoint.completed_step, config.steps
            )));
        }
        Ok(checkpoint)
    }

    pub(crate) fn train_oracle_models_burn_dense(
        models: &mut [NpaModel],
        examples: &[DirectBasisExample],
        config: DirectBasisTrainConfig,
    ) -> Result<BurnDenseOracleBatchOutput, Box<dyn std::error::Error>> {
        if models.is_empty() || examples.is_empty() {
            return Err(std::io::Error::other(
                "Burn dense oracle model batch requires at least one model/example",
            )
            .into());
        }
        if models.len() != examples.len() {
            return Err(std::io::Error::other(format!(
                "Burn dense oracle model batch length mismatch: models={} examples={}",
                models.len(),
                examples.len()
            ))
            .into());
        }
        if models
            .iter()
            .any(|model| model.config.spatial_dims != 2 || model.config != models[0].config)
        {
            return Err(std::io::Error::other(
                "Burn dense oracle model batch requires matching 2D NPA model configs",
            )
            .into());
        }

        let mut memory_snapshots = Vec::new();
        let mut gpu_memory_snapshots = Vec::new();
        memory_snapshots.push(check_process_memory_budget("oracle_batch:start", config)?);
        gpu_memory_snapshots.push(check_gpu_memory_budget("oracle_batch:start", config)?);
        let device = BurnDevice::default();
        let mut params = BurnBaseBatch::from_models(models, &device)?;
        let targets = burn_targets(examples, config, &device)?;
        let indices = (0..targets.len()).collect::<Vec<_>>();
        let Some(particle_count) = homogeneous_particle_count(&targets, &indices) else {
            return Err(std::io::Error::other(
                "Burn dense vectorized oracle batch requires homogeneous particle counts",
            )
            .into());
        };
        memory_snapshots.push(check_process_memory_budget(
            "oracle_batch:after_target_cache",
            config,
        )?);
        gpu_memory_snapshots.push(check_gpu_memory_budget(
            "oracle_batch:after_target_cache",
            config,
        )?);

        let mut optimizer = BurnBaseBatchAdamWState::zeros_like(&params);
        let mut particle_pools = config.use_particle_pool.then(|| {
            targets
                .iter()
                .enumerate()
                .map(|(model_index, target)| {
                    let mut pool_config = config;
                    pool_config.seed = config.seed.wrapping_add(
                        (model_index as u64).wrapping_mul(0x517c_c1b7_2722_0a95),
                    );
                    BurnDeviceParticlePool::new(
                        config.pool_size.max(config.example_batch_size).max(1),
                        particle_count,
                        models[0].config.state_dims,
                        target.seed_scale,
                        pool_config,
                        &device,
                    )
                })
                .collect::<Vec<_>>()
        });
        let mut history = Vec::new();
        let mut per_model_history = vec![Vec::new(); models.len()];
        let mut best_train_loss = vec![None::<f32>; models.len()];
        let mut best_train_step = vec![0usize; models.len()];
        let mut measured_particle_steps = 0.0_f64;
        let mut measured_elapsed_ms = 0.0_f64;
        let mut steady_particle_steps = 0.0_f64;
        let mut steady_elapsed_ms = 0.0_f64;

        for step in 1..=config.steps {
            let should_report =
                step == config.steps || step.is_multiple_of(config.report_interval.max(1));
            let step_seed = config
                .seed
                .wrapping_add((step as u64).wrapping_mul(0x9e37_79b9));
            let stats = train_oracle_model_batch_step_tbptt(
                &mut params,
                &mut optimizer,
                &targets,
                particle_count,
                config,
                step_seed,
                particle_pools.as_deref_mut(),
                config.use_particle_pool
                    && step.is_multiple_of(config.inject_seed_interval.max(1)),
                should_report,
            )?;
            let particle_steps = stats.particle_steps_per_sec * stats.elapsed_ms / 1_000.0;
            measured_particle_steps += particle_steps;
            measured_elapsed_ms += stats.elapsed_ms;
            if step > 1 {
                steady_particle_steps += particle_steps;
                steady_elapsed_ms += stats.elapsed_ms;
            }
            if should_report {
                let mean_loss = stats
                    .per_model_loss
                    .iter()
                    .copied()
                    .sum::<f32>()
                    / stats.per_model_loss.len().max(1) as f32;
                let mean_base_grad_norm = stats
                    .per_model_base_grad_norm
                    .iter()
                    .copied()
                    .sum::<f32>()
                    / stats.per_model_base_grad_norm.len().max(1) as f32;
                let mean_base_grad_scale = stats
                    .per_model_base_grad_scale
                    .iter()
                    .copied()
                    .sum::<f32>()
                    / stats.per_model_base_grad_scale.len().max(1) as f32;
                println!(
                    "{LOG_BACKEND} oracle-model-batch train step {step}/{} loss={mean_loss:.6} models={} particle_steps_per_sec={:.0} elapsed_ms={:.1}",
                    config.steps,
                    models.len(),
                    stats.particle_steps_per_sec,
                    stats.elapsed_ms
                );
                history.push(CliHyper2dDirectBasisHistoryEntry {
                    step,
                    loss: mean_loss,
                    eval_loss: None,
                    base_grad_norm: mean_base_grad_norm,
                    base_grad_scale: mean_base_grad_scale,
                    mean_adapter_grad_norm: 0.0,
                    max_adapter_grad_norm: 0.0,
                    examples_seen: models.len(),
                    particle_steps_per_sec: stats.particle_steps_per_sec,
                    elapsed_ms: stats.elapsed_ms,
                });
                for (idx, loss) in stats.per_model_loss.iter().copied().enumerate() {
                    if best_train_loss[idx].is_none_or(|best| loss < best) {
                        best_train_loss[idx] = Some(loss);
                        best_train_step[idx] = step;
                    }
                    per_model_history[idx].push(CliHyper2dDirectBasisHistoryEntry {
                        step,
                        loss,
                        eval_loss: None,
                        base_grad_norm: stats.per_model_base_grad_norm[idx],
                        base_grad_scale: stats.per_model_base_grad_scale[idx],
                        mean_adapter_grad_norm: 0.0,
                        max_adapter_grad_norm: 0.0,
                        examples_seen: 1,
                        particle_steps_per_sec: stats.particle_steps_per_sec
                            / models.len().max(1) as f64,
                        elapsed_ms: stats.elapsed_ms,
                    });
                }
                let _ = check_process_memory_budget(
                    &format!("oracle_batch:report_step:{step}"),
                    config,
                )?;
                let _ = check_gpu_memory_budget(
                    &format!("oracle_batch:report_step:{step}"),
                    config,
                )?;
            }
        }

        params.write_to_models(models)?;
        memory_snapshots.push(check_process_memory_budget("oracle_batch:end", config)?);
        gpu_memory_snapshots.push(check_gpu_memory_budget("oracle_batch:end", config)?);
        let metrics = json!({
            "backend": BACKEND,
            "device": DEVICE_LABEL,
            "objective": "target2d_pixel_splat_loss_full_image",
            "mode": "vectorized_independent_oracle_models",
            "model_batch_size": models.len(),
            "optimizer_state": "vectorized_independent_adamw_moments_per_oracle_model",
            "parameter_sharing": false,
            "rollout_batch_size_per_model": config.example_batch_size,
            "particle_pool": {
                "enabled": config.use_particle_pool,
                "slots_per_model": config.pool_size,
                "inject_seed_interval": config.inject_seed_interval,
                "brush_size": config.brush_size,
            },
            "particle_count": particle_count,
            "steps": config.steps,
            "rollout_steps": config.rollout_steps,
            "tbptt_chunk_steps": config.tbptt_chunk_steps,
            "loss_on_final_chunk_only": config.loss_on_final_chunk_only,
            "measured_particle_steps": measured_particle_steps,
            "measured_elapsed_ms": measured_elapsed_ms,
            "measured_particle_steps_per_sec": measured_particle_steps
                / (measured_elapsed_ms / 1_000.0).max(f64::MIN_POSITIVE),
            "steady_state_excludes_first_step": true,
            "steady_particle_steps": steady_particle_steps,
            "steady_elapsed_ms": steady_elapsed_ms,
            "steady_particle_steps_per_sec": steady_particle_steps
                / (steady_elapsed_ms / 1_000.0).max(f64::MIN_POSITIVE),
            "max_dense_chunk_floats": config.max_dense_chunk_floats,
            "max_splat_chunk_floats": config.max_splat_chunk_floats,
            "system_memory_budget_gb": config.system_memory_budget_gb,
            "gpu_memory_budget_gb": config.gpu_memory_budget_gb,
            "process_memory_snapshots": memory_snapshots,
            "gpu_memory_snapshots": gpu_memory_snapshots,
        });
        Ok(BurnDenseOracleBatchOutput {
            backend: BACKEND,
            device: DEVICE_LABEL.to_string(),
            metrics,
            history,
            per_model_history,
            best_train_loss,
            best_train_step,
        })
    }

    pub(super) struct BurnPhaseReport {
        pub(super) history: Vec<CliHyper2dDirectBasisHistoryEntry>,
        pub(super) best_loss: Option<f32>,
        pub(super) best_step: usize,
        pub(super) best_geometry_score: Option<f32>,
        pub(super) sample_updates: SampleUpdateStats,
    }

    #[derive(Clone, Serialize)]
    pub(super) struct BurnDenseCheckpointEvent {
        kind: &'static str,
        phase: String,
        step: usize,
        elapsed_seconds: f64,
        train_loss: Option<f32>,
        eval_loss: Option<f32>,
        geometry_score: Option<f32>,
        model_output: String,
        sha256: Option<String>,
    }

    pub(super) struct BurnDenseCheckpointWrite {
        kind: &'static str,
        output: std::path::PathBuf,
        phase: String,
        step: usize,
        train_loss: Option<f32>,
        eval_loss: Option<f32>,
        geometry_score: Option<f32>,
    }

    pub(super) struct BurnDenseCheckpointState<'a> {
        config: &'a Target2dBurnCheckpointConfig,
        started: Instant,
        last_current_write: Instant,
        current_writes: usize,
        best_writes: usize,
        events: Vec<BurnDenseCheckpointEvent>,
    }

    impl<'a> BurnDenseCheckpointState<'a> {
        pub(super) fn new(config: &'a Target2dBurnCheckpointConfig) -> Self {
            let now = Instant::now();
            Self {
                config,
                started: now,
                last_current_write: now,
                current_writes: 0,
                best_writes: 0,
                events: Vec::new(),
            }
        }

        pub(super) fn should_write_current(&self, step: usize) -> bool {
            let step_due =
                self.config.interval_steps > 0 && step.is_multiple_of(self.config.interval_steps);
            let time_due = self
                .config
                .interval_duration
                .is_some_and(|interval| self.last_current_write.elapsed() >= interval);
            step_due || time_due
        }

        pub(super) fn write_current(
            &mut self,
            params: &BurnBaseParams,
            phase: &str,
            step: usize,
            train_loss: Option<f32>,
            eval_loss: Option<f32>,
            geometry_score: Option<f32>,
        ) -> Result<(), Box<dyn std::error::Error>> {
            self.write_model(params, BurnDenseCheckpointWrite {
                kind: "current",
                output: self.config.current_model_output.clone(),
                phase: phase.to_string(),
                step,
                train_loss,
                eval_loss,
                geometry_score,
            })?;
            self.current_writes = self.current_writes.saturating_add(1);
            self.last_current_write = Instant::now();
            self.write_metadata()?;
            Ok(())
        }

        pub(super) fn write_best(
            &mut self,
            params: &BurnBaseParams,
            phase: &str,
            step: usize,
            train_loss: Option<f32>,
            eval_loss: Option<f32>,
            geometry_score: Option<f32>,
        ) -> Result<(), Box<dyn std::error::Error>> {
            self.write_model(params, BurnDenseCheckpointWrite {
                kind: "best",
                output: self.config.best_model_output.clone(),
                phase: phase.to_string(),
                step,
                train_loss,
                eval_loss,
                geometry_score,
            })?;
            self.best_writes = self.best_writes.saturating_add(1);
            self.write_metadata()?;
            Ok(())
        }

        pub(super) fn write_model(
            &mut self,
            params: &BurnBaseParams,
            request: BurnDenseCheckpointWrite,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let mut model = NpaModel {
                config: self.config.model_config.clone(),
                weights: NpaWeights::zeros(&self.config.model_config),
            };
            params.write_to_model(&mut model)?;
            let source = Some(format!(
                "{}:checkpoint:{}:phase={}:step={}",
                self.config.source, request.kind, request.phase, request.step
            ));
            let manifest =
                crate::import::BpkModelManifest::from_model(&model, self.config.hashgrid.clone(), source);
            let sha256 = atomic_save_manifest(&request.output, &manifest)?;
            let event = BurnDenseCheckpointEvent {
                kind: request.kind,
                phase: request.phase.clone(),
                step: request.step,
                elapsed_seconds: self.started.elapsed().as_secs_f64(),
                train_loss: request.train_loss,
                eval_loss: request.eval_loss,
                geometry_score: request.geometry_score,
                model_output: request.output.display().to_string(),
                sha256,
            };
            self.events.push(event.clone());
            println!(
                "{LOG_BACKEND} direct-basis checkpoint {} phase={} step={} model={}",
                request.kind,
                request.phase,
                request.step,
                request.output.display()
            );
            Ok(())
        }

        pub(super) fn write_metadata(&self) -> Result<(), Box<dyn std::error::Error>> {
            let report = self.report_json();
            atomic_write_json(&self.config.metadata_output, &report)
        }

        pub(super) fn report_json(&self) -> serde_json::Value {
            json!({
                "current_model_output": self.config.current_model_output.display().to_string(),
                "best_model_output": self.config.best_model_output.display().to_string(),
                "metadata_output": self.config.metadata_output.display().to_string(),
                "interval_steps": self.config.interval_steps,
                "interval_seconds": self.config.interval_duration.map(|duration| duration.as_secs()),
                "current_writes": self.current_writes,
                "best_writes": self.best_writes,
                "elapsed_seconds": self.started.elapsed().as_secs_f64(),
                "events": &self.events,
            })
        }
    }

    pub(super) fn atomic_save_manifest(
        path: &std::path::Path,
        manifest: &crate::import::BpkModelManifest,
    ) -> Result<Option<String>, Box<dyn std::error::Error>> {
        let tmp_path = atomic_temp_path(path);
        let sha256 = crate::import::save_manifest(&tmp_path, manifest)?;
        fs::rename(&tmp_path, path)?;
        Ok(sha256)
    }

    pub(super) fn atomic_write_json(
        path: &std::path::Path,
        value: &serde_json::Value,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp_path = atomic_temp_path(path);
        fs::write(&tmp_path, serde_json::to_string_pretty(value)?)?;
        fs::rename(&tmp_path, path)?;
        Ok(())
    }

    pub(super) fn atomic_temp_path(path: &std::path::Path) -> std::path::PathBuf {
        let extension = path.extension().and_then(|value| value.to_str()).unwrap_or("json");
        let file_name = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("checkpoint");
        path.with_file_name(format!(".{file_name}.tmp.{extension}"))
    }

    #[derive(Clone, Copy, Debug, Serialize)]
    pub(super) struct BurnGeometrySummary {
        pub(super) examples: usize,
        pub(super) mean_score: f32,
        pub(super) mean_foreground_iou: f32,
        pub(super) mean_target_recall: f32,
        pub(super) mean_generated_precision: f32,
        pub(super) mean_bbox_iou: f32,
        pub(super) mean_lit_pixel_ratio: f32,
        pub(super) mean_bbox_width_ratio: f32,
        pub(super) mean_bbox_area_ratio: f32,
    }
