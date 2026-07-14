//! Direct-basis, oracle, and end-to-end HyperNPA optimization steps.

use super::*;

    pub(super) fn run_phase(
        params: &mut BurnBaseParams,
        adapters: &mut [BurnAdapterParams],
        targets: &[BurnTargetExample],
        config: DirectBasisTrainConfig,
        update_base: bool,
        phase_label: &str,
        mut checkpoint_state: Option<&mut BurnDenseCheckpointState<'_>>,
    ) -> Result<BurnPhaseReport, Box<dyn std::error::Error>> {
        if targets.is_empty() || config.steps == 0 {
            return Ok(BurnPhaseReport {
                history: Vec::new(),
                best_loss: None,
                best_step: 0,
                best_geometry_score: None,
                sample_updates: sample_update_stats(&vec![0; targets.len()]),
            });
        }
        let mut rng = StdRng::seed_from_u64(config.seed);
        let mut sampler =
            PhaseBatchSampler::new(targets.len(), config.example_batch_size, &mut rng);
        let homogeneous_pool_particle_count = if config.use_particle_pool {
            let particle_count = targets[0].particle_count;
            if targets
                .iter()
                .any(|target| target.particle_count != particle_count)
            {
                return Err(std::io::Error::other(
                    "Burn target2d particle-pool training requires homogeneous particle counts",
                )
                .into());
            }
            Some(particle_count)
        } else {
            None
        };
        let mut particle_pool = homogeneous_pool_particle_count.map(|particle_count| {
            BurnDeviceParticlePool::new(
                config.pool_size.max(config.example_batch_size).max(1),
                particle_count,
                16,
                targets[0].seed_scale,
                config,
                &targets[0].target_rgb.device(),
            )
        });
        let mut sample_update_counts = vec![0usize; targets.len()];
        let mut history = Vec::new();
        let mut best_loss = None;
        let mut best_step = 0;
        let mut best_geometry_score = None;
        let mut best_params = None::<BurnBaseParams>;
        let mut best_adapters = None::<Vec<BurnAdapterParams>>;
        if config.eval_interval > 0
            && let Some(eval_loss) = evaluate_targets(
                params,
                adapters,
                targets,
                config,
                config.eval_examples,
                config.eval_seed,
            )?
        {
            best_loss = Some(eval_loss.mean_total_loss);
            best_geometry_score = if config.use_particle_pool {
                evaluate_target_geometry(
                    params,
                    adapters,
                    targets,
                    config,
                    config.eval_examples,
                    config.eval_seed,
                )?
                .map(|geometry| geometry.mean_score)
            } else {
                None
            };
            best_params = update_base.then(|| params.clone());
            best_adapters = Some(adapters.to_vec());
            if let Some(checkpoint_state) = checkpoint_state.as_deref_mut() {
                checkpoint_state.write_best(
                    params,
                    phase_label,
                    0,
                    None,
                    Some(eval_loss.mean_total_loss),
                    best_geometry_score,
                )?;
                checkpoint_state.write_current(
                    params,
                    phase_label,
                    0,
                    None,
                    Some(eval_loss.mean_total_loss),
                    best_geometry_score,
                )?;
            }
        }
        let mut base_optimizer = BurnBaseAdamWState::zeros_like(params);
        let mut adapter_optimizers = adapters
            .iter()
            .map(BurnAdapterAdamWState::zeros_like)
            .collect::<Vec<_>>();
        for step in 1usize..=config.steps {
            let should_report =
                step == config.steps || step.is_multiple_of(config.report_interval.max(1));
            let should_eval = config.eval_interval > 0
                && (step == config.steps || step.is_multiple_of(config.eval_interval.max(1)));
            let started = Instant::now();
            let step_seed = config
                .seed
                .wrapping_add((step as u64).wrapping_mul(0x9e37_79b9));
            let stats = if let (Some(pool), Some(particle_count)) =
                (particle_pool.as_mut(), homogeneous_pool_particle_count)
            {
                let replace_seed = step.is_multiple_of(config.inject_seed_interval.max(1));
                let device = &targets[0].target_rgb.device();
                let pool_batch = pool.sample_batch(
                    &mut rng,
                    config.example_batch_size.max(1),
                    replace_seed,
                    targets[0].seed_scale,
                    config,
                    device,
                )?;
                let indices = (0..pool_batch.pool_indices.len())
                    .map(|local| local % targets.len())
                    .collect::<Vec<_>>();
                if indices.is_empty() {
                    return Err(std::io::Error::other("Burn direct-basis pool batch was empty")
                        .into());
                }
                for &idx in &indices {
                    sample_update_counts[idx] = sample_update_counts[idx].saturating_add(1);
                }
                train_homogeneous_step_tbptt(
                    params,
                    adapters,
                    &mut base_optimizer,
                    &mut adapter_optimizers,
                    targets,
                    &indices,
                    particle_count,
                    config,
                    step_seed,
                    update_base,
                    should_report,
                    Some((pool_batch.x, pool_batch.s)),
                    Some((pool, pool_batch.pool_indices)),
                )?
            } else {
                let indices = sampler.next_batch(&mut rng);
                if indices.is_empty() {
                    return Err(
                        std::io::Error::other("Burn direct-basis batch was empty").into(),
                    );
                };
                for &idx in &indices {
                    sample_update_counts[idx] = sample_update_counts[idx].saturating_add(1);
                }
                train_step_tbptt(
                    params,
                    adapters,
                    &mut base_optimizer,
                    &mut adapter_optimizers,
                    targets,
                    &indices,
                    config,
                    step_seed,
                    update_base,
                    should_report,
                )?
            };
            let elapsed = started.elapsed();
            if should_report {
                let eval_loss = if should_eval {
                    evaluate_targets(
                        params,
                        adapters,
                        targets,
                        config,
                        config.eval_examples,
                        config.eval_seed + step as u64,
                    )?
                } else {
                    None
                };
                if let Some(eval_loss) = eval_loss {
                    let geometry = if config.use_particle_pool {
                        evaluate_target_geometry(
                            params,
                            adapters,
                            targets,
                            config,
                            config.eval_examples,
                            config.eval_seed + step as u64,
                        )?
                    } else {
                        None
                    };
                    let is_better = if let Some(geometry) = geometry {
                        best_geometry_score
                            .is_none_or(|best| geometry.mean_score > best)
                    } else {
                        best_loss.is_none_or(|best| eval_loss.mean_total_loss < best)
                    };
                    if is_better {
                        best_loss = Some(eval_loss.mean_total_loss);
                        best_step = step;
                        if let Some(geometry) = geometry {
                            best_geometry_score = Some(geometry.mean_score);
                        }
                        best_params = update_base.then(|| params.clone());
                        best_adapters = Some(adapters.to_vec());
                        if let Some(checkpoint_state) = checkpoint_state.as_deref_mut() {
                            checkpoint_state.write_best(
                                params,
                                phase_label,
                                step,
                                Some(stats.loss),
                                Some(eval_loss.mean_total_loss),
                                best_geometry_score,
                            )?;
                        }
                    }
                    println!(
                        "{LOG_BACKEND} direct-basis {phase_label} step {step}/{} loss={:.6} eval_mean={:.6} examples={} particle_steps_per_sec={:.0} elapsed_ms={:.1}",
                        config.steps,
                        stats.loss,
                        eval_loss.mean_total_loss,
                        stats.examples_seen,
                        stats.particle_steps_per_sec,
                        elapsed.as_secs_f64() * 1000.0
                    );
                    history.push(CliHyper2dDirectBasisHistoryEntry {
                        step,
                        loss: stats.loss,
                        eval_loss: Some(eval_loss),
                        base_grad_norm: stats.base_grad_norm,
                        base_grad_scale: stats.base_grad_scale,
                        mean_adapter_grad_norm: stats.mean_adapter_grad_norm,
                        max_adapter_grad_norm: stats.max_adapter_grad_norm,
                        examples_seen: stats.examples_seen,
                        particle_steps_per_sec: stats.particle_steps_per_sec,
                        elapsed_ms: stats.elapsed_ms,
                    });
                } else {
                    println!(
                        "{LOG_BACKEND} direct-basis {phase_label} step {step}/{} loss={:.6} examples={} particle_steps_per_sec={:.0} elapsed_ms={:.1}",
                        config.steps,
                        stats.loss,
                        stats.examples_seen,
                        stats.particle_steps_per_sec,
                        elapsed.as_secs_f64() * 1000.0
                    );
                    history.push(CliHyper2dDirectBasisHistoryEntry {
                        step,
                        loss: stats.loss,
                        eval_loss: None,
                        base_grad_norm: stats.base_grad_norm,
                        base_grad_scale: stats.base_grad_scale,
                        mean_adapter_grad_norm: stats.mean_adapter_grad_norm,
                        max_adapter_grad_norm: stats.max_adapter_grad_norm,
                        examples_seen: stats.examples_seen,
                        particle_steps_per_sec: stats.particle_steps_per_sec,
                        elapsed_ms: stats.elapsed_ms,
                    });
                }
                let _ = check_process_memory_budget(
                    &format!("{phase_label}:report_step:{step}"),
                    config,
                )?;
                let _ =
                    check_gpu_memory_budget(&format!("{phase_label}:report_step:{step}"), config)?;
            }
            if let Some(checkpoint_state) = checkpoint_state.as_deref_mut()
                && step != config.steps
                && checkpoint_state.should_write_current(step)
            {
                checkpoint_state.write_current(
                    params,
                    phase_label,
                    step,
                    Some(stats.loss),
                    None,
                    best_geometry_score,
                )?;
            }
        }
        if let Some(saved) = best_params {
            *params = saved;
        }
        if let Some(saved) = best_adapters {
            adapters.clone_from_slice(&saved);
        }
        Ok(BurnPhaseReport {
            history,
            best_loss,
            best_step,
            best_geometry_score,
            sample_updates: sample_update_stats(&sample_update_counts),
        })
    }

    pub(super) fn best_training_checkpoint(
        train_steps: usize,
        train_phase: &BurnPhaseReport,
        train_refine_phase: &BurnPhaseReport,
    ) -> (Option<f32>, usize) {
        let train_best = train_phase
            .best_loss
            .map(|loss| (loss, train_phase.best_step));
        let refine_best = train_refine_phase
            .best_loss
            .map(|loss| (loss, train_steps + train_refine_phase.best_step));
        match (train_best, refine_best) {
            (Some(train), Some(refine)) => {
                if refine.0 < train.0 {
                    (Some(refine.0), refine.1)
                } else {
                    (Some(train.0), train.1)
                }
            }
            (Some(train), None) => (Some(train.0), train.1),
            (None, Some(refine)) => (Some(refine.0), refine.1),
            (None, None) => (None, 0),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn train_step_tbptt(
        params: &mut BurnBaseParams,
        adapters: &mut [BurnAdapterParams],
        base_optimizer: &mut BurnBaseAdamWState,
        adapter_optimizers: &mut [BurnAdapterAdamWState],
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        step_seed: u64,
        update_base: bool,
        collect_metrics: bool,
    ) -> Result<DirectBasisStepStats, Box<dyn std::error::Error>> {
        if indices.is_empty() {
            return Err(std::io::Error::other("Burn direct-basis batch was empty").into());
        }
        if let Some(particle_count) = homogeneous_particle_count(targets, indices) {
            return train_homogeneous_step_tbptt(
                params,
                adapters,
                base_optimizer,
                adapter_optimizers,
                targets,
                indices,
                particle_count,
                config,
                step_seed,
                update_base,
                collect_metrics,
                None,
                None,
            );
        }
        train_mixed_step_tbptt(
            params,
            adapters,
            base_optimizer,
            adapter_optimizers,
            targets,
            indices,
            config,
            step_seed,
            update_base,
            collect_metrics,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn train_homogeneous_step_tbptt(
        params: &mut BurnBaseParams,
        adapters: &mut [BurnAdapterParams],
        base_optimizer: &mut BurnBaseAdamWState,
        adapter_optimizers: &mut [BurnAdapterAdamWState],
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        config: DirectBasisTrainConfig,
        step_seed: u64,
        update_base: bool,
        collect_metrics: bool,
        initial_state: Option<(Tensor3, Tensor3)>,
        pool_update: Option<(&mut BurnDeviceParticlePool, Vec<usize>)>,
    ) -> Result<DirectBasisStepStats, Box<dyn std::error::Error>> {
        let started = Instant::now();
        let device = &targets[indices[0]].target_rgb.device();
        let (mut x, mut s) = initial_state.unwrap_or_else(|| {
            seed_batch_tensors(targets, indices, particle_count, config, step_seed, device)
        });
        let mut rng = StdRng::seed_from_u64(step_seed ^ 0x005e_ed2d);
        let chunk_steps = tbptt_chunk_steps(config);
        let rollout_steps = sampled_training_rollout_steps(config, step_seed);
        let chunk_count = rollout_steps.div_ceil(chunk_steps).max(1);
        let mut loss_sum = collect_metrics.then_some(0.0_f32);
        let mut base_grad_norm_sum = 0.0_f32;
        let mut base_grad_scale_sum = 0.0_f32;
        let mut adapter_grad_sum = 0.0_f32;
        let mut adapter_grad_max = 0.0_f32;
        let mut grad_metric_chunks = 0usize;
        let mut particle_steps = 0.0_f64;
        let mut remaining_steps = rollout_steps;
        while remaining_steps > 0 {
            let steps = remaining_steps.min(chunk_steps);
            let final_chunk = remaining_steps <= chunk_steps;
            if config.loss_on_final_chunk_only && !final_chunk {
                let detached_params = params.detached();
                let adapter_batch =
                    BurnAdapterBatch::from_indices(adapters, indices).detached();
                let displacement = Tensor::<BurnBackend, 1>::zeros([1], device);
                let (next_x, next_s, _) = rollout_batch_chunk(
                    &detached_params,
                    &adapter_batch,
                    targets,
                    indices,
                    x,
                    s,
                    config,
                    particle_count,
                    &mut rng,
                    steps,
                    displacement,
                    None,
                );
                x = detach3(next_x);
                s = detach3(next_s);
                particle_steps += indices.len() as f64 * particle_count as f64 * steps as f64;
                remaining_steps -= steps;
                continue;
            }
            let adapter_batch = BurnAdapterBatch::from_indices(adapters, indices);
            let displacement = Tensor::<BurnBackend, 1>::zeros([1], device);
            let (next_x, next_s, displacement) = rollout_batch_chunk(
                params,
                &adapter_batch,
                targets,
                indices,
                x,
                s,
                config,
                particle_count,
                &mut rng,
                steps,
                displacement,
                None,
            );
            let loss = target_splat_loss_batch(
                &next_x,
                &next_s,
                targets,
                indices,
                config,
                &adapter_batch,
                displacement,
            )?;
            if let Some(loss_sum) = loss_sum.as_mut() {
                *loss_sum += loss_scalars(&loss)?.total * indices.len() as f32;
            }
            let grad_stats = apply_chunk_gradients(
                params,
                adapters,
                base_optimizer,
                adapter_optimizers,
                indices,
                loss.total,
                config,
                update_base,
                indices.len() as f32,
                collect_metrics,
            )?;
            if collect_metrics {
                base_grad_norm_sum += grad_stats.base_grad_norm;
                base_grad_scale_sum += grad_stats.base_grad_scale;
                adapter_grad_sum += grad_stats.adapter_grad_sum;
                adapter_grad_max = adapter_grad_max.max(grad_stats.adapter_grad_max);
                grad_metric_chunks += 1;
            }
            x = detach3(next_x);
            s = detach3(next_s);
            particle_steps += indices.len() as f64 * particle_count as f64 * steps as f64;
            remaining_steps -= steps;
        }
        if let Some((pool, pool_indices)) = pool_update {
            pool.update_batch(&pool_indices, x, s)?;
        }
        let elapsed = started.elapsed();
        let grad_metric_chunks = grad_metric_chunks.max(1);
        let loss_chunk_count = if config.loss_on_final_chunk_only {
            1
        } else {
            chunk_count
        };
        Ok(DirectBasisStepStats {
            loss: loss_sum.map_or(0.0, |value| {
                value / indices.len() as f32 / loss_chunk_count as f32
            }),
            base_grad_norm: base_grad_norm_sum / grad_metric_chunks as f32,
            base_grad_scale: if collect_metrics {
                base_grad_scale_sum / grad_metric_chunks as f32
            } else {
                1.0
            },
            mean_adapter_grad_norm: if collect_metrics {
                adapter_grad_sum / (indices.len() * grad_metric_chunks).max(1) as f32
            } else {
                0.0
            },
            max_adapter_grad_norm: adapter_grad_max,
            examples_seen: indices.len(),
            particle_steps_per_sec: particle_steps / elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn train_mixed_step_tbptt(
        params: &mut BurnBaseParams,
        adapters: &mut [BurnAdapterParams],
        base_optimizer: &mut BurnBaseAdamWState,
        adapter_optimizers: &mut [BurnAdapterAdamWState],
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        step_seed: u64,
        update_base: bool,
        collect_metrics: bool,
    ) -> Result<DirectBasisStepStats, Box<dyn std::error::Error>> {
        let started = Instant::now();
        let chunk_steps = tbptt_chunk_steps(config);
        let rollout_steps = sampled_training_rollout_steps(config, step_seed);
        let chunk_count = rollout_steps.div_ceil(chunk_steps).max(1);
        let mut loss_sum = collect_metrics.then_some(0.0_f32);
        let mut base_grad_norm_sum = 0.0_f32;
        let mut base_grad_scale_sum = 0.0_f32;
        let mut adapter_grad_sum = 0.0_f32;
        let mut adapter_grad_max = 0.0_f32;
        let mut grad_metric_chunks = 0usize;
        let mut particle_steps = 0.0_f64;
        for &idx in indices {
            let target = &targets[idx];
            let device = &target.target_rgb.device();
            let (mut x, mut s) = seed_tensors(
                target.particle_count,
                config,
                target.seed_scale,
                step_seed.wrapping_add(idx as u64),
                device,
            );
            let mut rng = StdRng::seed_from_u64(step_seed.wrapping_add(idx as u64) ^ 0x005e_ed2d);
            let mut remaining_steps = rollout_steps;
            while remaining_steps > 0 {
                let steps = remaining_steps.min(chunk_steps);
                let final_chunk = remaining_steps <= chunk_steps;
                if config.loss_on_final_chunk_only && !final_chunk {
                    let detached_params = params.detached();
                    let detached_adapter = adapters[idx].detached();
                    let displacement = Tensor::<BurnBackend, 1>::zeros([1], device);
                    let (next_x, next_s, _) = rollout_single_chunk(
                        &detached_params,
                        &detached_adapter,
                        target,
                        x,
                        s,
                        config,
                        &mut rng,
                        steps,
                        displacement,
                    );
                    x = detach2(next_x);
                    s = detach2(next_s);
                    particle_steps += target.particle_count as f64 * steps as f64;
                    remaining_steps -= steps;
                    continue;
                }
                let displacement = Tensor::<BurnBackend, 1>::zeros([1], device);
                let (next_x, next_s, displacement) = rollout_single_chunk(
                    params,
                    &adapters[idx],
                    target,
                    x,
                    s,
                    config,
                    &mut rng,
                    steps,
                    displacement,
                );
                let loss = target_splat_loss(
                    &next_x,
                    &next_s,
                    target,
                    config,
                    &adapters[idx],
                    displacement,
                );
                if let Some(loss_sum) = loss_sum.as_mut() {
                    *loss_sum += loss_scalars(&loss)?.total;
                }
                let scaled_total = loss.total.div_scalar(indices.len() as f32);
                let single_index = [idx];
                let grad_stats = apply_chunk_gradients(
                    params,
                    adapters,
                    base_optimizer,
                    adapter_optimizers,
                    &single_index,
                    scaled_total,
                    config,
                    update_base,
                    indices.len() as f32,
                    collect_metrics,
                )?;
                if collect_metrics {
                    base_grad_norm_sum += grad_stats.base_grad_norm;
                    base_grad_scale_sum += grad_stats.base_grad_scale;
                    adapter_grad_sum += grad_stats.adapter_grad_sum;
                    adapter_grad_max = adapter_grad_max.max(grad_stats.adapter_grad_max);
                    grad_metric_chunks += 1;
                }
                x = detach2(next_x);
                s = detach2(next_s);
                particle_steps += target.particle_count as f64 * steps as f64;
                remaining_steps -= steps;
            }
        }
        let elapsed = started.elapsed();
        let grad_metric_chunks = grad_metric_chunks.max(1);
        let loss_chunk_count = if config.loss_on_final_chunk_only {
            1
        } else {
            chunk_count
        };
        Ok(DirectBasisStepStats {
            loss: loss_sum.map_or(0.0, |value| {
                value / indices.len() as f32 / loss_chunk_count as f32
            }),
            base_grad_norm: base_grad_norm_sum / grad_metric_chunks as f32,
            base_grad_scale: if collect_metrics {
                base_grad_scale_sum / grad_metric_chunks as f32
            } else {
                1.0
            },
            mean_adapter_grad_norm: if collect_metrics {
                adapter_grad_sum / grad_metric_chunks as f32
            } else {
                0.0
            },
            max_adapter_grad_norm: adapter_grad_max,
            examples_seen: indices.len(),
            particle_steps_per_sec: particle_steps / elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn select_rollout_conditions(
        conditions: &BurnE2eConditionCache,
        condition_indices: &[usize],
        prepared_dino: Option<&BurnE2ePreparedDinoBatch>,
        rollouts_per_example: usize,
    ) -> AutomataResult<(Tensor3, Option<Vec<usize>>)> {
        let replicas = rollouts_per_example.max(1);
        if replicas == 1 || prepared_dino.is_some() || !condition_indices.len().is_multiple_of(replicas)
        {
            return conditions
                .select_prepared(condition_indices, prepared_dino)
                .map(|condition| (condition, None));
        }
        let chunks = condition_indices.chunks(replicas).collect::<Vec<_>>();
        if chunks
            .iter()
            .any(|chunk| chunk.iter().any(|identity| *identity != chunk[0]))
        {
            return conditions
                .select_prepared(condition_indices, prepared_dino)
                .map(|condition| (condition, None));
        }
        let unique = chunks.iter().map(|chunk| chunk[0]).collect::<Vec<_>>();
        let expansion = (0..unique.len())
            .flat_map(|row| std::iter::repeat_n(row, replicas))
            .collect::<Vec<_>>();
        conditions
            .select(&unique)
            .map(|condition| (condition, Some(expansion)))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn train_e2e_homogeneous_step_tbptt(
        params: &mut BurnBaseParams,
        generator: &mut BurnE2eGeneratorParams,
        base_optimizer: &mut BurnBaseAdamWState,
        generator_optimizer: &mut BurnE2eGeneratorAdamWState,
        npa_config: &NpaConfig,
        conditions: &BurnE2eConditionCache,
        condition_indices: &[usize],
        prepared_dino: Option<&BurnE2ePreparedDinoBatch>,
        targets: &[BurnTargetExample],
        target_indices: &[usize],
        config: BurnE2eRolloutTrainConfig,
        step_seed: u64,
        collect_metrics: bool,
        collect_per_example_losses: bool,
        initial_state: Option<(Tensor3, Tensor3)>,
    ) -> Result<BurnE2eStepOutput, Box<dyn std::error::Error>> {
        if config.credit_assignment == E2eCreditAssignment::FullBptt {
            return train_e2e_homogeneous_step_full_bptt(
                params,
                generator,
                base_optimizer,
                generator_optimizer,
                npa_config,
                conditions,
                condition_indices,
                prepared_dino,
                targets,
                target_indices,
                config,
                step_seed,
                collect_metrics,
                collect_per_example_losses,
                initial_state,
            );
        }
        if condition_indices.len() != target_indices.len() {
            return Err(std::io::Error::other(
                "Burn HyperNPA e2e rollout condition/target batch length mismatch",
            )
            .into());
        }
        let batch_len = condition_indices.len();
        let Some(particle_count) = homogeneous_particle_count(targets, target_indices) else {
            return Err(std::io::Error::other(
                "Burn HyperNPA e2e rollout batches require homogeneous particle counts",
            )
            .into());
        };
        let started = Instant::now();
        let direct_config = direct_config_view(config);
        let device = &targets[target_indices[0]].target_rgb.device();
        let (mut x, mut s) = initial_state.unwrap_or_else(|| {
            seed_batch_tensors(
                targets,
                target_indices,
                particle_count,
                direct_config,
                step_seed,
                device,
            )
        });
        let mut rng = StdRng::seed_from_u64(step_seed ^ 0x005e_ed2d);
        let mut particle_steps = 0.0_f64;
        if config.pre_rollout_steps > 0 {
            let detached_params = params.detached();
            let detached_generator = generator.detached();
            let (condition, expansion) = select_rollout_conditions(
                conditions,
                condition_indices,
                prepared_dino,
                config.rollouts_per_example,
            )?;
            let adapter_batch = detached_generator
                .adapter_batch(condition.clone(), npa_config, config)
                .select_rows_or_identity(expansion.as_deref());
            let condition_control = detached_generator
                .condition_control_batch(condition.clone(), config)
                .map(|control| control.select_rows_or_identity(expansion.as_deref()));
            let displacement = Tensor::<BurnBackend, 1>::zeros([batch_len], device);
            let (next_x, next_s, _) = rollout_batch_chunk(
                &detached_params,
                &adapter_batch,
                targets,
                target_indices,
                x,
                s,
                direct_config,
                particle_count,
                &mut rng,
                config.pre_rollout_steps,
                displacement,
                condition_control.as_ref(),
            );
            x = detach3(next_x);
            s = detach3(next_s);
            particle_steps +=
                batch_len as f64 * particle_count as f64 * config.pre_rollout_steps as f64;
        }
        let chunk_steps = tbptt_chunk_steps(direct_config);
        let rollout_steps = sampled_training_rollout_steps(direct_config, step_seed);
        let mut loss_sum = collect_metrics.then_some(0.0_f32);
        let mut loss_weight_sum =
            (collect_metrics || collect_per_example_losses).then_some(0.0_f32);
        let mut per_example_loss_sum =
            collect_per_example_losses.then(|| vec![0.0_f32; batch_len]);
        let mut base_grad_norm_sum = 0.0_f32;
        let mut base_grad_scale_sum = 0.0_f32;
        let mut generator_grad_norm_sum = 0.0_f32;
        let mut generator_grad_scale_sum = 0.0_f32;
        let mut grad_metric_chunks = 0usize;
        let (condition, expansion) = select_rollout_conditions(
            conditions,
            condition_indices,
            prepared_dino,
            config.rollouts_per_example,
        )?;
        let mut remaining_steps = rollout_steps;
        while remaining_steps > 0 {
            let steps = remaining_steps.min(chunk_steps);
            let final_chunk = remaining_steps <= chunk_steps;
            let loss_weight = e2e_chunk_loss_weight(config, final_chunk);
            if loss_weight <= 0.0 {
                let detached_params = params.detached();
                let detached_generator = generator.detached();
                let adapter_batch = detached_generator
                    .adapter_batch(condition.clone(), npa_config, config)
                    .select_rows_or_identity(expansion.as_deref());
                let condition_control = detached_generator
                    .condition_control_batch(condition.clone(), config)
                    .map(|control| control.select_rows_or_identity(expansion.as_deref()));
                let displacement = Tensor::<BurnBackend, 1>::zeros([batch_len], device);
                let (next_x, next_s, _) = rollout_batch_chunk(
                    &detached_params,
                    &adapter_batch,
                    targets,
                    target_indices,
                    x,
                    s,
                    direct_config,
                    particle_count,
                    &mut rng,
                    steps,
                    displacement,
                    condition_control.as_ref(),
                );
                x = detach3(next_x);
                s = detach3(next_s);
                particle_steps += batch_len as f64 * particle_count as f64 * steps as f64;
                remaining_steps -= steps;
                continue;
            }
            let adapter_batch = generator
                .adapter_batch(condition.clone(), npa_config, config)
                .select_rows_or_identity(expansion.as_deref());
            let condition_control = generator
                .condition_control_batch(condition.clone(), config)
                .map(|control| control.select_rows_or_identity(expansion.as_deref()));
            let displacement = Tensor::<BurnBackend, 1>::zeros([batch_len], device);
            let (next_x, next_s, displacement) = rollout_batch_chunk(
                params,
                &adapter_batch,
                targets,
                target_indices,
                x,
                s,
                direct_config,
                particle_count,
                &mut rng,
                steps,
                displacement,
                condition_control.as_ref(),
            );
            let loss = target_splat_loss_batch_vector_selected(
                &next_x,
                &next_s,
                targets,
                target_indices,
                direct_config,
                &adapter_batch,
                displacement,
            )?;
            if collect_metrics || collect_per_example_losses {
                let scalars = loss_vector_scalars(loss.clone())?;
                if let Some(loss_sum) = loss_sum.as_mut() {
                    for value in &scalars {
                        *loss_sum += value.total * loss_weight;
                    }
                }
                if let Some(per_example_loss_sum) = per_example_loss_sum.as_mut() {
                    for (sum, value) in per_example_loss_sum.iter_mut().zip(&scalars) {
                        *sum += value.total * loss_weight;
                    }
                }
            }
            if let Some(loss_weight_sum) = loss_weight_sum.as_mut() {
                *loss_weight_sum += loss_weight;
            }
            let mut grads = loss
                .total
                .sum()
                .mul_scalar(loss_weight)
                .div_scalar(batch_len as f32)
                .backward();
            let (base_grad_norm, base_grad_scale) = if config.shared_base_trainable {
                params.apply_adamw(
                    &mut grads,
                    base_optimizer,
                    config.base_optimizer,
                    config.base_per_parameter_grad_normalization,
                    collect_metrics,
                )?
            } else {
                (0.0, 1.0)
            };
            let (generator_grad_norm, generator_grad_scale) = generator.apply_adamw(
                &mut grads,
                generator_optimizer,
                config.generator_optimizer,
                config.generator_per_parameter_grad_normalization,
                collect_metrics,
            )?;
            if collect_metrics {
                base_grad_norm_sum += base_grad_norm;
                base_grad_scale_sum += base_grad_scale;
                generator_grad_norm_sum += generator_grad_norm;
                generator_grad_scale_sum += generator_grad_scale;
                grad_metric_chunks += 1;
            }
            x = detach3(next_x);
            s = detach3(next_s);
            particle_steps += batch_len as f64 * particle_count as f64 * steps as f64;
            remaining_steps -= steps;
        }
        let elapsed = started.elapsed();
        let grad_metric_chunks = grad_metric_chunks.max(1);
        let loss_weight_count = loss_weight_sum.unwrap_or(1.0).max(f32::MIN_POSITIVE);
        let per_example_losses = per_example_loss_sum.map(|losses| {
            losses
                .into_iter()
                .map(|loss| loss / loss_weight_count)
                .collect()
        });
        let particle_steps_per_sec =
            particle_steps / elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
        Ok(BurnE2eStepOutput {
            history: BurnE2eRolloutHistoryEntry {
                step: 0,
                loss: loss_sum.map_or(0.0, |value| {
                    value / batch_len as f32 / loss_weight_count
                }),
                task_loss: loss_sum.map_or(0.0, |value| {
                    value / batch_len as f32 / loss_weight_count
                }),
                adapter_teacher_loss: 0.0,
                flow_matching_loss: 0.0,
                flow_self_rectification_loss: 0.0,
                learning_rate_scale: 1.0,
                base_learning_rate: config.base_optimizer.learning_rate,
                generator_learning_rate: config.generator_optimizer.learning_rate,
                holdout_mean_psnr_db: None,
                holdout_mean_loss: None,
                base_grad_norm: base_grad_norm_sum / grad_metric_chunks as f32,
                base_grad_scale: base_grad_scale_sum / grad_metric_chunks as f32,
                generator_grad_norm: generator_grad_norm_sum / grad_metric_chunks as f32,
                generator_grad_scale: generator_grad_scale_sum / grad_metric_chunks as f32,
                examples_seen: batch_len,
                pool_seed_replacements: 0,
                particle_steps_per_sec,
                dense_pair_interactions_per_sec: particle_steps_per_sec * particle_count as f64,
                elapsed_ms: elapsed.as_secs_f64() * 1000.0,
                condition_adapter_ms: 0.0,
                rollout_loss_ms: 0.0,
                backward_update_ms: 0.0,
            },
            particle_steps: particle_steps.round().max(0.0) as u64,
            final_x: x,
            final_s: s,
            per_example_losses,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn train_e2e_flow_matching_only_step(
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
        let condition = conditions.select_prepared(condition_indices, prepared_dino)?;
        let teacher = conditions.select_teacher(condition_indices).ok_or_else(|| {
            AutomataError::InvalidArgument(
                "conditional flow matching requires teacher adapter endpoints".to_string(),
            )
        })?;
        let flow = generator.row_flow.as_ref().ok_or_else(|| {
            AutomataError::InvalidArgument(
                "flow_matching_weight requires conditional-row-flow".to_string(),
            )
        })?;
        let objective = flow.flow_matching_loss(
            condition,
            teacher,
            npa_config,
            config.adapter_rank,
            config.adapter_alpha,
            step_seed ^ 0x666c_6f77_6d61_7463,
        );
        let flow_loss = if collect_metrics {
            objective.clone().inner().into_scalar()
        } else {
            0.0
        };
        if collect_metrics {
            sync_training_device(device)?;
        }
        let forward_ms = if collect_metrics {
            started.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };
        let backward_started = Instant::now();
        let mut grads = objective
            .mul_scalar(config.flow_matching_weight.max(0.0))
            .backward();
        let (generator_grad_norm, generator_grad_scale) = generator.apply_adamw(
            &mut grads,
            generator_optimizer,
            config.generator_optimizer,
            config.generator_per_parameter_grad_normalization,
            collect_metrics,
        )?;
        if collect_metrics {
            sync_training_device(device)?;
        }
        let backward_update_ms = if collect_metrics {
            backward_started.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };
        let elapsed = started.elapsed();
        Ok(BurnE2eStepOutput {
            history: BurnE2eRolloutHistoryEntry {
                step: 0,
                loss: flow_loss * config.flow_matching_weight.max(0.0),
                task_loss: 0.0,
                adapter_teacher_loss: 0.0,
                flow_matching_loss: flow_loss,
                flow_self_rectification_loss: 0.0,
                learning_rate_scale: 1.0,
                base_learning_rate: 0.0,
                generator_learning_rate: config.generator_optimizer.learning_rate,
                holdout_mean_psnr_db: None,
                holdout_mean_loss: None,
                base_grad_norm: 0.0,
                base_grad_scale: 1.0,
                generator_grad_norm,
                generator_grad_scale,
                examples_seen: condition_indices.len(),
                pool_seed_replacements: 0,
                particle_steps_per_sec: 0.0,
                dense_pair_interactions_per_sec: 0.0,
                elapsed_ms: elapsed.as_secs_f64() * 1000.0,
                condition_adapter_ms: forward_ms,
                rollout_loss_ms: 0.0,
                backward_update_ms,
            },
            particle_steps: 0,
            final_x: detach3(x),
            final_s: detach3(s),
            per_example_losses: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn train_e2e_homogeneous_step_full_bptt(
        params: &mut BurnBaseParams,
        generator: &mut BurnE2eGeneratorParams,
        base_optimizer: &mut BurnBaseAdamWState,
        generator_optimizer: &mut BurnE2eGeneratorAdamWState,
        npa_config: &NpaConfig,
        conditions: &BurnE2eConditionCache,
        condition_indices: &[usize],
        prepared_dino: Option<&BurnE2ePreparedDinoBatch>,
        targets: &[BurnTargetExample],
        target_indices: &[usize],
        config: BurnE2eRolloutTrainConfig,
        step_seed: u64,
        collect_metrics: bool,
        collect_per_example_losses: bool,
        initial_state: Option<(Tensor3, Tensor3)>,
    ) -> Result<BurnE2eStepOutput, Box<dyn std::error::Error>> {
        if condition_indices.len() != target_indices.len() {
            return Err(std::io::Error::other(
                "Burn HyperNPA full-BPTT condition/target batch length mismatch",
            )
            .into());
        }
        let batch_len = condition_indices.len();
        let Some(particle_count) = homogeneous_particle_count(targets, target_indices) else {
            return Err(std::io::Error::other(
                "Burn HyperNPA full-BPTT requires homogeneous particle counts",
            )
            .into());
        };
        let started = Instant::now();
        if config.task_loss_weight == 0.0
            && config.adapter_teacher_weight == 0.0
            && config.flow_matching_weight > 0.0
            && config.flow_self_rectification_weight == 0.0
        {
            return train_e2e_flow_matching_only_step(
                generator,
                generator_optimizer,
                npa_config,
                conditions,
                condition_indices,
                prepared_dino,
                targets,
                target_indices,
                particle_count,
                config,
                step_seed,
                collect_metrics,
                initial_state,
            );
        }
        let direct_config = direct_config_view(config);
        let rollout_steps = sampled_training_rollout_steps(direct_config, step_seed);
        let particle_steps = batch_len
            .saturating_mul(particle_count)
            .saturating_mul(rollout_steps);
        if particle_steps > config.max_full_bptt_particle_steps {
            return Err(std::io::Error::other(format!(
                "HyperNPA full-BPTT runtime preflight rejected {particle_steps} particle-steps, above configured cap {}",
                config.max_full_bptt_particle_steps
            ))
            .into());
        }
        let device = &targets[target_indices[0]].target_rgb.device();
        let (mut x, mut s) = initial_state.unwrap_or_else(|| {
            seed_batch_tensors(
                targets,
                target_indices,
                particle_count,
                direct_config,
                step_seed,
                device,
            )
        });
        let mut rng = StdRng::seed_from_u64(step_seed ^ 0x005e_ed2d);
        if config.pre_rollout_steps > 0 {
            let detached_params = params.detached();
            let detached_generator = generator.detached();
            let (condition, expansion) = select_rollout_conditions(
                conditions,
                condition_indices,
                prepared_dino,
                config.rollouts_per_example,
            )?;
            let adapter = detached_generator
                .adapter_batch(condition.clone(), npa_config, config)
                .select_rows_or_identity(expansion.as_deref());
            let condition_control = detached_generator
                .condition_control_batch(condition, config)
                .map(|control| control.select_rows_or_identity(expansion.as_deref()));
            let displacement = Tensor::<BurnBackend, 1>::zeros([batch_len], device);
            let (next_x, next_s, _) = rollout_batch_chunk(
                &detached_params,
                &adapter,
                targets,
                target_indices,
                x,
                s,
                direct_config,
                particle_count,
                &mut rng,
                config.pre_rollout_steps,
                displacement,
                condition_control.as_ref(),
            );
            x = detach3(next_x);
            s = detach3(next_s);
        }

        if collect_metrics {
            sync_training_device(device)?;
        }
        let condition_started = Instant::now();
        let (condition, expansion) = select_rollout_conditions(
            conditions,
            condition_indices,
            prepared_dino,
            config.rollouts_per_example,
        )?;
        let (adapter, generated_dense_rows, prepared_flow_condition) = generator
            .adapter_batch_with_dense_rows(condition.clone(), npa_config, config);
        let adapter = adapter.select_rows_or_identity(expansion.as_deref());
        let condition_control = generator
            .condition_control_batch(condition, config)
            .map(|control| control.select_rows_or_identity(expansion.as_deref()));
        let teacher_vector = conditions.select_teacher(condition_indices);
        let flow_objective = if config.flow_matching_weight > 0.0 {
            let teacher = teacher_vector
                .clone()
                .ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "conditional flow matching requires teacher adapter endpoints".to_string(),
                    )
                })?;
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
                            .expect("flow condition was prepared")
                            ,
                        teacher,
                        npa_config,
                        config.adapter_rank,
                        config.adapter_alpha,
                    ),
            )
        } else {
            None
        };
        let flow_self_rectification_objective = if config.flow_self_rectification_weight > 0.0 {
            Some(
                generator
                    .row_flow
                    .as_ref()
                    .ok_or_else(|| {
                        AutomataError::InvalidArgument(
                            "flow_self_rectification_weight requires conditional-row-flow"
                                .to_string(),
                        )
                    })?
                    .self_rectification_loss_to_endpoint_prepared(
                        prepared_flow_condition
                            .as_ref()
                            .expect("flow condition was prepared")
                            ,
                        generated_dense_rows
                            .as_ref()
                            .expect("row flow generated dense endpoint rows")
                            .clone(),
                        npa_config,
                        step_seed ^ 0x7365_6c66_7265_6374,
                    ),
            )
        } else {
            None
        };
        let teacher_adapter = teacher_vector.clone().map(|teacher| {
            BurnAdapterBatch::from_parameter_vector(
                teacher,
                npa_config,
                config.adapter_rank,
                config.adapter_alpha,
            )
        });
        let teacher_probe_features = if config.adapter_teacher_weight > 0.0
            && config.adapter_teacher_objective != E2eAdapterTeacherObjective::ParameterMse
        {
            let teacher_adapter = teacher_adapter
                .as_ref()
                .expect("functional teacher objective requires teacher adapters");
            let max_probe_steps = config.adapter_teacher_probe_rollout_steps;
            let probe_steps = if max_probe_steps == 0 {
                0
            } else {
                1 + step_seed as usize % max_probe_steps
            };
            let (probe_x, probe_s) = if probe_steps == 0 {
                (detach3(x.clone()), detach3(s.clone()))
            } else {
                let mut teacher_rng = rng.clone();
                let (probe_x, probe_s, _) = rollout_batch_chunk(
                    &params.detached(),
                    teacher_adapter,
                    targets,
                    target_indices,
                    detach3(x.clone()),
                    detach3(s.clone()),
                    direct_config,
                    particle_count,
                    &mut teacher_rng,
                    probe_steps,
                    Tensor::<BurnBackend, 1>::zeros([batch_len], device),
                    None,
                );
                (detach3(probe_x), detach3(probe_s))
            };
            Some(rollout_dense_perception_batch(
                &probe_x,
                &probe_s,
                direct_config,
            ))
        } else {
            None
        };
        if collect_metrics {
            sync_training_device(device)?;
        }
        let condition_adapter_ms = if collect_metrics {
            condition_started.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };
        let rollout_started = Instant::now();
        let displacement = Tensor::<BurnBackend, 1>::zeros([batch_len], device);
        let (next_x, next_s, displacement) = rollout_batch_chunk(
            params,
            &adapter,
            targets,
            target_indices,
            x,
            s,
            direct_config,
            particle_count,
            &mut rng,
            rollout_steps,
            displacement,
            condition_control.as_ref(),
        );
        let loss = target_splat_loss_batch_vector_selected(
            &next_x,
            &next_s,
            targets,
            target_indices,
            direct_config,
            &adapter,
            displacement.clone(),
        )?;
        let task_objective = loss.total.clone().sum().div_scalar(batch_len.max(1) as f32);
        let teacher_objective = teacher_vector.map(|teacher| {
            let generated_vector = adapter.to_parameter_vector();
            let parameter_delta = generated_vector - teacher.clone();
            let parameter_mse = parameter_delta.clone().mul(parameter_delta).mean();
            if config.adapter_teacher_objective == E2eAdapterTeacherObjective::ParameterMse {
                return parameter_mse;
            }

            let teacher_adapter = teacher_adapter
                .as_ref()
                .expect("functional teacher objective requires teacher adapters");
            let probes = teacher_probe_features
                .as_ref()
                .expect("functional teacher objective prepared perception probes")
                .clone();
            let generated_update = params.forward_adapter_batch(probes.clone(), &adapter);
            let teacher_update = detach3(
                params
                    .detached()
                    .forward_adapter_batch(probes, teacher_adapter),
            );
            let functional_delta = generated_update - teacher_update;
            let functional_mse = functional_delta.clone().mul(functional_delta).mean();
            if config.adapter_teacher_objective == E2eAdapterTeacherObjective::Hybrid {
                functional_mse
                    + parameter_mse.mul_scalar(FUNCTIONAL_TEACHER_PARAMETER_AUX_WEIGHT)
            } else {
                functional_mse
            }
        });
        let teacher_loss_value = if collect_metrics {
            teacher_objective
                .as_ref()
                .map(|teacher| teacher.clone().inner().into_scalar())
                .unwrap_or_default()
        } else {
            0.0
        };
        let flow_loss_value = if collect_metrics {
            flow_objective
                .as_ref()
                .map(|flow| flow.clone().inner().into_scalar())
                .unwrap_or_default()
        } else {
            0.0
        };
        let flow_self_rectification_loss_value = if collect_metrics {
            flow_self_rectification_objective
                .as_ref()
                .map(|flow| flow.clone().inner().into_scalar())
                .unwrap_or_default()
        } else {
            0.0
        };
        let weighted_task = task_objective.mul_scalar(config.task_loss_weight.max(0.0));
        let objective = teacher_objective.map_or(weighted_task.clone(), |teacher| {
            weighted_task + teacher.mul_scalar(config.adapter_teacher_weight.max(0.0))
        });
        let objective = flow_objective.map_or(objective.clone(), |flow| {
            objective + flow.mul_scalar(config.flow_matching_weight.max(0.0))
        });
        let objective =
            flow_self_rectification_objective.map_or(objective.clone(), |self_rectification| {
                objective
                    + self_rectification
                        .mul_scalar(config.flow_self_rectification_weight.max(0.0))
            });
        if collect_metrics {
            sync_training_device(device)?;
        }
        let rollout_loss_ms = if collect_metrics {
            rollout_started.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };
        let loss_scalars = if collect_per_example_losses {
            Some(loss_vector_scalars(loss.clone())?)
        } else {
            None
        };
        let task_loss_value = loss_scalars.as_ref().map_or(0.0, |losses| {
            losses.iter().map(|value| value.total).sum::<f32>() / batch_len.max(1) as f32
        });
        let loss_value = task_loss_value * config.task_loss_weight.max(0.0)
            + teacher_loss_value * config.adapter_teacher_weight.max(0.0)
            + flow_loss_value * config.flow_matching_weight.max(0.0)
            + flow_self_rectification_loss_value
                * config.flow_self_rectification_weight.max(0.0);
        let backward_started = Instant::now();
        let mut grads = objective.backward();
        let (base_grad_norm, base_grad_scale) = if config.shared_base_trainable {
            params.apply_adamw(
                &mut grads,
                base_optimizer,
                config.base_optimizer,
                config.base_per_parameter_grad_normalization,
                collect_metrics,
            )?
        } else {
            (0.0, 1.0)
        };
        let (generator_grad_norm, generator_grad_scale) = generator.apply_adamw(
            &mut grads,
            generator_optimizer,
            config.generator_optimizer,
            config.generator_per_parameter_grad_normalization,
            collect_metrics,
        )?;
        if collect_metrics {
            sync_training_device(device)?;
        }
        let backward_update_ms = if collect_metrics {
            backward_started.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };
        let elapsed = started.elapsed();
        let measured_particle_steps = batch_len as f64
            * particle_count as f64
            * (rollout_steps + config.pre_rollout_steps) as f64;
        let particle_steps_per_sec =
            measured_particle_steps / elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
        Ok(BurnE2eStepOutput {
            history: BurnE2eRolloutHistoryEntry {
                step: 0,
                loss: loss_value,
                task_loss: task_loss_value,
                adapter_teacher_loss: teacher_loss_value,
                flow_matching_loss: flow_loss_value,
                flow_self_rectification_loss: flow_self_rectification_loss_value,
                learning_rate_scale: 1.0,
                base_learning_rate: config.base_optimizer.learning_rate,
                generator_learning_rate: config.generator_optimizer.learning_rate,
                holdout_mean_psnr_db: None,
                holdout_mean_loss: None,
                base_grad_norm,
                base_grad_scale,
                generator_grad_norm,
                generator_grad_scale,
                examples_seen: batch_len,
                pool_seed_replacements: 0,
                particle_steps_per_sec,
                dense_pair_interactions_per_sec: particle_steps_per_sec * particle_count as f64,
                elapsed_ms: elapsed.as_secs_f64() * 1000.0,
                condition_adapter_ms,
                rollout_loss_ms,
                backward_update_ms,
            },
            particle_steps: measured_particle_steps.round().max(0.0) as u64,
            final_x: detach3(next_x),
            final_s: detach3(next_s),
            per_example_losses: loss_scalars.map(|losses| {
                losses.into_iter().map(|value| value.total).collect()
            }),
        })
    }

    pub(super) struct OracleModelBatchStepStats {
        pub(super) per_model_loss: Vec<f32>,
        pub(super) per_model_base_grad_norm: Vec<f32>,
        pub(super) per_model_base_grad_scale: Vec<f32>,
        pub(super) particle_steps_per_sec: f64,
        pub(super) elapsed_ms: f64,
    }

    pub(super) fn sampled_training_rollout_steps(config: DirectBasisTrainConfig, seed: u64) -> usize {
        let max_steps = config.rollout_steps.max(1);
        let min_steps = config.rollout_step_min.max(1).min(max_steps);
        if min_steps == max_steps {
            return max_steps;
        }
        let mut rng = StdRng::seed_from_u64(seed ^ 0x6d2b_79f5);
        rng.random_range(min_steps..max_steps)
    }

    pub(super) fn e2e_chunk_loss_weight(config: BurnE2eRolloutTrainConfig, final_chunk: bool) -> f32 {
        let mode = if config.loss_on_final_chunk_only {
            E2eTbpttLossMode::FinalOnly
        } else {
            config.tbptt_loss_mode
        };
        match mode {
            E2eTbpttLossMode::AllChunks => 1.0,
            E2eTbpttLossMode::FinalOnly => {
                if final_chunk {
                    1.0
                } else {
                    0.0
                }
            }
            E2eTbpttLossMode::EndpointWeighted => {
                if final_chunk {
                    config.tbptt_final_loss_weight.max(0.0)
                } else {
                    config.tbptt_intermediate_loss_weight.max(0.0)
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn train_oracle_model_batch_step_tbptt(
        params: &mut BurnBaseBatch,
        optimizer: &mut BurnBaseBatchAdamWState,
        targets: &[BurnTargetExample],
        particle_count: usize,
        config: DirectBasisTrainConfig,
        step_seed: u64,
        particle_pools: Option<&mut [BurnDeviceParticlePool]>,
        replace_pool_seed: bool,
        collect_metrics: bool,
    ) -> Result<OracleModelBatchStepStats, Box<dyn std::error::Error>> {
        if params.model_count() == 0 || params.model_count() != targets.len() {
            return Err(
                std::io::Error::other("Burn oracle model batch length mismatch").into(),
            );
        }
        let started = Instant::now();
        let device = &targets[0].target_rgb.device();
        let model_count = params.model_count();
        let trajectories_per_model = config.example_batch_size.max(1);
        let row_count = model_count.saturating_mul(trajectories_per_model);
        let indices = (0..model_count)
            .flat_map(|model| std::iter::repeat_n(model, trajectories_per_model))
            .collect::<Vec<_>>();
        let (mut x, mut s, pool_indices) = oracle_model_batch_initial_state(
            targets,
            &indices,
            particle_count,
            config,
            step_seed,
            particle_pools.as_deref(),
            replace_pool_seed,
            device,
        )?;
        let mut rngs = (0..row_count)
            .map(|row| {
                StdRng::seed_from_u64(
                    step_seed
                        .wrapping_add((row as u64).wrapping_mul(0x9e37_79b9))
                        ^ 0x005e_ed2d,
                )
            })
            .collect::<Vec<_>>();
        let chunk_steps = tbptt_chunk_steps(config);
        let rollout_steps = sampled_training_rollout_steps(config, step_seed);
        let chunk_count = rollout_steps.div_ceil(chunk_steps).max(1);
        let loss_chunk_count = if config.loss_on_final_chunk_only {
            1
        } else {
            chunk_count
        };
        let mut loss_sums = collect_metrics.then(|| vec![0.0_f32; model_count]);
        let mut grad_norm_sums = vec![0.0_f32; model_count];
        let mut grad_scale_sums = vec![0.0_f32; model_count];
        let mut grad_metric_chunks = 0usize;
        let mut particle_steps = 0.0_f64;
        let mut remaining_steps = rollout_steps;
        let mut displacement = Tensor::<BurnBackend, 1>::zeros([row_count], device);
        while remaining_steps > 0 {
            let steps = remaining_steps.min(chunk_steps);
            let final_chunk = remaining_steps <= chunk_steps;
            if config.loss_on_final_chunk_only && !final_chunk {
                let detached_params = params.detached();
                let (next_x, next_s, next_displacement) = rollout_oracle_model_batch_chunk(
                    &detached_params,
                    targets,
                    &indices,
                    x,
                    s,
                    config,
                    particle_count,
                    &mut rngs,
                    steps,
                    displacement,
                );
                x = detach3(next_x);
                s = detach3(next_s);
                displacement = detach1(next_displacement);
                particle_steps += row_count as f64 * particle_count as f64 * steps as f64;
                remaining_steps -= steps;
                continue;
            }
            let (next_x, next_s, next_displacement) = rollout_oracle_model_batch_chunk(
                params,
                targets,
                &indices,
                x,
                s,
                config,
                particle_count,
                &mut rngs,
                steps,
                displacement,
            );
            let loss = target_splat_loss_batch_vector_base_only_selected(
                &next_x,
                &next_s,
                targets,
                &indices,
                config,
                next_displacement.clone(),
            )?;
            if let Some(loss_sums) = loss_sums.as_mut() {
                let row_losses = loss_vector_scalars(loss.clone())?;
                for (model, model_rows) in row_losses.chunks(trajectories_per_model).enumerate() {
                    loss_sums[model] += model_rows
                        .iter()
                        .map(|scalars| scalars.total)
                        .sum::<f32>()
                        / model_rows.len().max(1) as f32;
                }
            }
            let mut grads = loss
                .total
                .reshape([model_count, trajectories_per_model])
                .mean_dim(1)
                .sum()
                .backward();
            let (grad_norms, grad_scales) = params.apply_adamw(
                &mut grads,
                optimizer,
                adamw_from_sgd(config.base_sgd),
                config.per_parameter_grad_normalization,
                collect_metrics,
            )?;
            if collect_metrics {
                for (sum, value) in grad_norm_sums.iter_mut().zip(grad_norms) {
                    *sum += value;
                }
                for (sum, value) in grad_scale_sums.iter_mut().zip(grad_scales) {
                    *sum += value;
                }
            }
            if collect_metrics {
                grad_metric_chunks += 1;
            }
            x = detach3(next_x);
            s = detach3(next_s);
            displacement = detach1(next_displacement);
            particle_steps += row_count as f64 * particle_count as f64 * steps as f64;
            remaining_steps -= steps;
        }
        if let (Some(pools), Some(pool_indices)) = (particle_pools, pool_indices) {
            for (model, (pool, indices)) in pools.iter_mut().zip(pool_indices).enumerate() {
                let start = model * trajectories_per_model;
                pool.update_batch(
                    &indices,
                    x.clone().narrow(0, start, trajectories_per_model),
                    s.clone().narrow(0, start, trajectories_per_model),
                )?;
            }
        }
        let elapsed = started.elapsed();
        let grad_metric_chunks = grad_metric_chunks.max(1);
        let per_model_loss = loss_sums
            .unwrap_or_else(|| vec![0.0; params.model_count()])
            .into_iter()
            .map(|loss| loss / loss_chunk_count as f32)
            .collect::<Vec<_>>();
        Ok(OracleModelBatchStepStats {
            per_model_loss,
            per_model_base_grad_norm: grad_norm_sums
                .into_iter()
                .map(|value| value / grad_metric_chunks as f32)
                .collect(),
            per_model_base_grad_scale: grad_scale_sums
                .into_iter()
                .map(|value| {
                    if collect_metrics {
                        value / grad_metric_chunks as f32
                    } else {
                        1.0
                    }
                })
                .collect(),
            particle_steps_per_sec: particle_steps / elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        })
    }

    type OracleModelBatchInitialState = (Tensor3, Tensor3, Option<Vec<Vec<usize>>>);

    #[allow(clippy::too_many_arguments)]
    pub(super) fn oracle_model_batch_initial_state(
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        config: DirectBasisTrainConfig,
        step_seed: u64,
        particle_pools: Option<&[BurnDeviceParticlePool]>,
        replace_pool_seed: bool,
        device: &BurnDevice,
    ) -> Result<OracleModelBatchInitialState, Box<dyn std::error::Error>> {
        let Some(pools) = particle_pools else {
            let (x, s) =
                seed_batch_tensors(targets, indices, particle_count, config, step_seed, device);
            return Ok((x, s, None));
        };
        if pools.len() != targets.len() || !indices.len().is_multiple_of(pools.len().max(1)) {
            return Err(std::io::Error::other(
                "Burn oracle particle-pool/model batch shape mismatch",
            )
            .into());
        }
        let trajectories_per_model = indices.len() / pools.len();
        let mut xs = Vec::with_capacity(pools.len());
        let mut states = Vec::with_capacity(pools.len());
        let mut selected = Vec::with_capacity(pools.len());
        for (model, (pool, target)) in pools.iter().zip(targets).enumerate() {
            let mut rng = StdRng::seed_from_u64(
                step_seed ^ (model as u64).wrapping_mul(0x517c_c1b7_2722_0a95),
            );
            let batch = pool.sample_batch(
                &mut rng,
                trajectories_per_model,
                replace_pool_seed,
                target.seed_scale,
                config,
                device,
            )?;
            if batch.pool_indices.len() != trajectories_per_model {
                return Err(std::io::Error::other(format!(
                    "oracle pool returned {} trajectories for requested batch {}",
                    batch.pool_indices.len(),
                    trajectories_per_model,
                ))
                .into());
            }
            selected.push(batch.pool_indices);
            xs.push(batch.x);
            states.push(batch.s);
        }
        Ok((
            Tensor::cat(xs, 0),
            Tensor::cat(states, 0),
            Some(selected),
        ))
    }

    pub(super) struct ChunkGradStats {
        base_grad_norm: f32,
        base_grad_scale: f32,
        adapter_grad_sum: f32,
        adapter_grad_max: f32,
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply_chunk_gradients(
        params: &mut BurnBaseParams,
        adapters: &mut [BurnAdapterParams],
        base_optimizer: &mut BurnBaseAdamWState,
        adapter_optimizers: &mut [BurnAdapterAdamWState],
        indices: &[usize],
        loss_total: Tensor1,
        config: DirectBasisTrainConfig,
        update_base: bool,
        adapter_gradient_scale: f32,
        collect_metrics: bool,
    ) -> AutomataResult<ChunkGradStats> {
        let mut grads = loss_total.backward();
        let (base_grad_norm, base_grad_scale) = if update_base {
            params.apply_adamw(
                &mut grads,
                base_optimizer,
                adamw_from_sgd(config.base_sgd),
                config.per_parameter_grad_normalization,
                collect_metrics,
            )?
        } else {
            (0.0, 1.0)
        };
        let mut adapter_grad_sum = 0.0_f32;
        let mut adapter_grad_max = 0.0_f32;
        for &idx in indices {
            let (grad_norm, _) = adapters[idx].apply_adamw(
                &mut grads,
                &mut adapter_optimizers[idx],
                adamw_from_sgd(config.adapter_sgd),
                config.per_parameter_grad_normalization,
                adapter_gradient_scale,
                collect_metrics,
            )?;
            if collect_metrics {
                adapter_grad_sum += grad_norm;
                adapter_grad_max = adapter_grad_max.max(grad_norm);
            }
        }
        Ok(ChunkGradStats {
            base_grad_norm,
            base_grad_scale,
            adapter_grad_sum,
            adapter_grad_max,
        })
    }

    pub(super) fn tbptt_chunk_steps(config: DirectBasisTrainConfig) -> usize {
        config
            .tbptt_chunk_steps
            .max(1)
            .min(config.rollout_steps.max(1))
    }
