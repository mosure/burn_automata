#[cfg(feature = "backend_wgpu")]
mod imp {
    use std::{fs, time::Instant};

    use burn::{
        backend::{Autodiff, Wgpu},
        tensor::{Device, Tensor, TensorData, activation::relu},
    };
    use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
    use serde::Serialize;
    use serde_json::json;

    use super::super::{DirectBasisExample, DirectBasisStepStats, DirectBasisTrainConfig};
    use crate::cli::reports::{
        CliHyper2dDirectBasisHistoryEntry, CliHyper2dDirectBasisLossSummary,
    };
    use crate::{
        AdamWConfig, AutomataError, AutomataResult, NpaLowRankAdapter, NpaModel, NpaWeights,
        SgdConfig,
        rollout::{seed_particles_scaled, stochastic_mask},
        target2d::render_target_2d_splat,
    };

    type InnerBackend = Wgpu<f32>;
    type BurnBackend = Autodiff<InnerBackend>;
    type BurnDevice = Device<BurnBackend>;
    type Tensor1 = Tensor<BurnBackend, 1>;
    type Tensor2 = Tensor<BurnBackend, 2>;
    type Tensor3 = Tensor<BurnBackend, 3>;
    type Tensor4 = Tensor<BurnBackend, 4>;
    type Tensor1Inner = Tensor<InnerBackend, 1>;
    type Tensor2Inner = Tensor<InnerBackend, 2>;

    const BACKEND: &str = "burn_wgpu_autodiff_dense_direct_basis";
    const EPSILON: f32 = 1.0e-6;

    pub(crate) struct BurnWgpuDirectBasisOutput {
        pub(crate) backend: &'static str,
        pub(crate) device: String,
        pub(crate) metrics: serde_json::Value,
        pub(crate) history: Vec<CliHyper2dDirectBasisHistoryEntry>,
        pub(crate) train_refine_history: Vec<CliHyper2dDirectBasisHistoryEntry>,
        pub(crate) holdout_history: Vec<CliHyper2dDirectBasisHistoryEntry>,
        pub(crate) best_train_loss: Option<f32>,
        pub(crate) best_train_step: usize,
    }

    #[derive(Clone)]
    struct BurnBaseParams {
        w1: Tensor2,
        b1: Tensor2,
        w2: Tensor2,
        b2: Tensor2,
    }

    #[derive(Clone)]
    struct BurnAdapterParams {
        rank: usize,
        alpha: f32,
        w1_down: Tensor2,
        w1_up: Tensor2,
        w2_down: Tensor2,
        w2_up: Tensor2,
        b1_delta: Tensor2,
        b2_delta: Tensor2,
    }

    struct BurnAdapterBatch {
        rank: usize,
        alpha: f32,
        w1_down: Tensor3,
        w1_up: Tensor3,
        w2_down: Tensor3,
        w2_up: Tensor3,
        b1_delta: Tensor3,
        b2_delta: Tensor3,
    }

    struct BurnBaseAdamWState {
        step: usize,
        w1_m: Tensor2Inner,
        w1_v: Tensor2Inner,
        b1_m: Tensor2Inner,
        b1_v: Tensor2Inner,
        w2_m: Tensor2Inner,
        w2_v: Tensor2Inner,
        b2_m: Tensor2Inner,
        b2_v: Tensor2Inner,
    }

    struct BurnAdapterAdamWState {
        step: usize,
        w1_down_m: Tensor2Inner,
        w1_down_v: Tensor2Inner,
        w1_up_m: Tensor2Inner,
        w1_up_v: Tensor2Inner,
        w2_down_m: Tensor2Inner,
        w2_down_v: Tensor2Inner,
        w2_up_m: Tensor2Inner,
        w2_up_v: Tensor2Inner,
        b1_delta_m: Tensor2Inner,
        b1_delta_v: Tensor2Inner,
        b2_delta_m: Tensor2Inner,
        b2_delta_v: Tensor2Inner,
    }

    struct BurnTargetExample {
        target_rgb: Tensor2,
        target_density: Tensor2,
        target_mean: Tensor2,
        pixel_xy: Tensor2,
        pixel_size: f32,
        target_points: usize,
        particle_count: usize,
        update_prob: f32,
        seed_scale: f32,
    }

    struct BurnLossTensors {
        total: Tensor1,
        splat: Tensor1,
        color: Tensor1,
        density: Tensor1,
    }

    struct BurnLossBatchTensors {
        total: Tensor1,
        splat: Tensor1,
        color: Tensor1,
        density: Tensor1,
    }

    #[derive(Clone, Copy, Default)]
    struct BurnLossScalars {
        total: f32,
        splat: f32,
        color: f32,
        density: f32,
    }

    #[derive(Clone, Serialize)]
    struct ProcessMemorySnapshot {
        label: String,
        rss_bytes: Option<u64>,
        budget_bytes: Option<u64>,
    }

    pub(crate) fn train_direct_basis_burn_wgpu(
        base: &mut NpaModel,
        train_examples: &mut [DirectBasisExample],
        holdout_examples: &mut [DirectBasisExample],
        train_config: DirectBasisTrainConfig,
        train_refine_config: DirectBasisTrainConfig,
        holdout_config: DirectBasisTrainConfig,
    ) -> Result<BurnWgpuDirectBasisOutput, Box<dyn std::error::Error>> {
        if base.config.spatial_dims != 2 {
            return Err(std::io::Error::other(
                "Burn/WGPU direct-basis training currently supports 2D",
            )
            .into());
        }
        let mut memory_snapshots = Vec::new();
        memory_snapshots.push(check_process_memory_budget("start", train_config)?);
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
        let train_phase = run_phase(
            &mut params,
            &mut train_adapters,
            &train_targets,
            train_config,
            true,
            "train",
        )?;
        memory_snapshots.push(check_process_memory_budget(
            "after_train_phase",
            train_config,
        )?);
        params.write_to_model(base)?;
        let train_refine_phase = run_phase(
            &mut params,
            &mut train_adapters,
            &train_targets,
            train_refine_config,
            false,
            "train-refine",
        )?;
        memory_snapshots.push(check_process_memory_budget(
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
        let holdout_phase = run_phase(
            &mut params,
            &mut holdout_adapters,
            &holdout_targets,
            holdout_config,
            false,
            "holdout",
        )?;
        memory_snapshots.push(check_process_memory_budget(
            "after_holdout_phase",
            holdout_config,
        )?);
        write_adapters(holdout_examples, &holdout_adapters)?;

        let metrics = json!({
            "backend": BACKEND,
            "device": "wgpu-default",
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
            "checkpoint_selection": "restore_best_reported_eval_loss",
            "optimizer": "adamw",
            "optimizer_cli_fields": "base/adapter learning_rate, weight_decay, grad_clip_norm",
            "adamw_beta1": 0.9,
            "adamw_beta2": 0.999,
            "adamw_epsilon": 1.0e-8,
            "adapter_gradient_scale": "unaverage_batch_loss_for_per_sample_adapter_adamw",
            "batching": "homogeneous_particle_count_batched_rollout_perception_splat_loss",
            "training_graph": "tbptt_chunked_rollout_state_detach",
            "tbptt_chunk_steps": train_config.tbptt_chunk_steps,
            "eval_interval": train_config.eval_interval,
            "eval_batch_size": train_config.eval_batch_size,
            "max_dense_train_particles": train_config.max_dense_train_particles,
            "max_dense_chunk_floats": train_config.max_dense_chunk_floats,
            "max_splat_chunk_floats": train_config.max_splat_chunk_floats,
            "system_memory_budget_gb": train_config.system_memory_budget_gb,
            "gpu_memory_budget_gb": train_config.gpu_memory_budget_gb,
            "process_memory_snapshots": memory_snapshots,
            "evaluation": "bounded_tbptt_chunked_loss_vectors_state_detach",
            "train_mean_adapter_updates_per_sample": mean_updates_per_sample(
                train_config.steps,
                train_config.example_batch_size,
                train_examples.len(),
            ),
            "train_refine_mean_adapter_updates_per_sample": mean_updates_per_sample(
                train_refine_config.steps,
                train_refine_config.example_batch_size,
                train_examples.len(),
            ),
            "holdout_mean_adapter_updates_per_sample": mean_updates_per_sample(
                holdout_config.steps,
                holdout_config.example_batch_size,
                holdout_examples.len(),
            ),
        });
        let (best_train_loss, best_train_step) = match train_refine_phase.best_loss {
            Some(loss) => (
                Some(loss),
                train_config.steps + train_refine_phase.best_step,
            ),
            None => (train_phase.best_loss, train_phase.best_step),
        };
        Ok(BurnWgpuDirectBasisOutput {
            backend: BACKEND,
            device: "wgpu-default".to_string(),
            metrics,
            history: train_phase.history,
            train_refine_history: train_refine_phase.history,
            holdout_history: holdout_phase.history,
            best_train_loss,
            best_train_step,
        })
    }

    struct BurnPhaseReport {
        history: Vec<CliHyper2dDirectBasisHistoryEntry>,
        best_loss: Option<f32>,
        best_step: usize,
    }

    fn run_phase(
        params: &mut BurnBaseParams,
        adapters: &mut [BurnAdapterParams],
        targets: &[BurnTargetExample],
        config: DirectBasisTrainConfig,
        update_base: bool,
        phase_label: &str,
    ) -> Result<BurnPhaseReport, Box<dyn std::error::Error>> {
        if targets.is_empty() || config.steps == 0 {
            return Ok(BurnPhaseReport {
                history: Vec::new(),
                best_loss: None,
                best_step: 0,
            });
        }
        let mut rng = StdRng::seed_from_u64(config.seed);
        let mut history = Vec::new();
        let mut best_loss = None;
        let mut best_step = 0;
        let mut best_params = None::<BurnBaseParams>;
        let mut best_adapters = None::<Vec<BurnAdapterParams>>;
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
            let indices = sample_indices(targets.len(), config.example_batch_size, &mut rng);
            let step_seed = config
                .seed
                .wrapping_add((step as u64).wrapping_mul(0x9e37_79b9));
            if indices.is_empty() {
                return Err(std::io::Error::other("Burn direct-basis batch was empty").into());
            };
            let stats = train_step_tbptt(
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
            )?;
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
                    if best_loss.is_none_or(|best| eval_loss.mean_total_loss < best) {
                        best_loss = Some(eval_loss.mean_total_loss);
                        best_step = step;
                        best_params = update_base.then(|| params.clone());
                        best_adapters = Some(adapters.to_vec());
                    }
                    println!(
                        "burn-wgpu direct-basis {phase_label} step {step}/{} loss={:.6} eval_mean={:.6} examples={} particle_steps_per_sec={:.0} elapsed_ms={:.1}",
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
                        "burn-wgpu direct-basis {phase_label} step {step}/{} loss={:.6} examples={} particle_steps_per_sec={:.0} elapsed_ms={:.1}",
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
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn train_step_tbptt(
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
    fn train_homogeneous_step_tbptt(
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
    ) -> Result<DirectBasisStepStats, Box<dyn std::error::Error>> {
        let started = Instant::now();
        let device = &targets[indices[0]].target_rgb.device();
        let (mut x, mut s) =
            seed_batch_tensors(targets, indices, particle_count, config, step_seed, device);
        let mut rng = StdRng::seed_from_u64(step_seed ^ 0x005e_ed2d);
        let chunk_steps = tbptt_chunk_steps(config);
        let chunk_count = config.rollout_steps.div_ceil(chunk_steps).max(1);
        let mut loss_sum = collect_metrics.then_some(0.0_f32);
        let mut base_grad_norm_sum = 0.0_f32;
        let mut base_grad_scale_sum = 0.0_f32;
        let mut adapter_grad_sum = 0.0_f32;
        let mut adapter_grad_max = 0.0_f32;
        let mut grad_metric_chunks = 0usize;
        let mut particle_steps = 0.0_f64;
        let mut remaining_steps = config.rollout_steps;
        while remaining_steps > 0 {
            let steps = remaining_steps.min(chunk_steps);
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
            );
            let loss = target_splat_loss_batch(
                &next_x,
                &next_s,
                targets,
                indices,
                config,
                &adapter_batch,
                displacement,
            );
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
        let elapsed = started.elapsed();
        let grad_metric_chunks = grad_metric_chunks.max(1);
        Ok(DirectBasisStepStats {
            loss: loss_sum.map_or(0.0, |value| {
                value / indices.len() as f32 / chunk_count as f32
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
    fn train_mixed_step_tbptt(
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
        let chunk_count = config.rollout_steps.div_ceil(chunk_steps).max(1);
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
            let mut remaining_steps = config.rollout_steps;
            while remaining_steps > 0 {
                let steps = remaining_steps.min(chunk_steps);
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
        Ok(DirectBasisStepStats {
            loss: loss_sum.map_or(0.0, |value| {
                value / indices.len() as f32 / chunk_count as f32
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

    struct ChunkGradStats {
        base_grad_norm: f32,
        base_grad_scale: f32,
        adapter_grad_sum: f32,
        adapter_grad_max: f32,
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_chunk_gradients(
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

    fn tbptt_chunk_steps(config: DirectBasisTrainConfig) -> usize {
        config
            .tbptt_chunk_steps
            .max(1)
            .min(config.rollout_steps.max(1))
    }

    fn evaluate_targets(
        params: &BurnBaseParams,
        adapters: &[BurnAdapterParams],
        targets: &[BurnTargetExample],
        config: DirectBasisTrainConfig,
        requested_examples: usize,
        seed: u64,
    ) -> Result<Option<CliHyper2dDirectBasisLossSummary>, Box<dyn std::error::Error>> {
        if targets.is_empty() {
            return Ok(None);
        }
        let mut indices = (0..targets.len()).collect::<Vec<_>>();
        if requested_examples > 0 && requested_examples < indices.len() {
            let mut rng = StdRng::seed_from_u64(seed);
            indices.shuffle(&mut rng);
            indices.truncate(requested_examples);
            indices.sort_unstable();
        }
        let mut summary = CliHyper2dDirectBasisLossSummary {
            examples: indices.len(),
            mean_total_loss: 0.0,
            max_total_loss: 0.0,
            mean_splat_loss: 0.0,
            mean_color_loss: 0.0,
            mean_density_loss: 0.0,
        };
        let eval_batch_size = normalized_eval_batch_size(config.eval_batch_size, indices.len());
        for chunk in indices.chunks(eval_batch_size) {
            if homogeneous_particle_count(targets, chunk).is_some() {
                let loss = batch_example_eval_loss(params, adapters, targets, chunk, config, seed)?;
                for scalars in loss_vector_scalars(loss)? {
                    summary.mean_total_loss += scalars.total;
                    summary.max_total_loss = summary.max_total_loss.max(scalars.total);
                    summary.mean_splat_loss += scalars.splat;
                    summary.mean_color_loss += scalars.color;
                    summary.mean_density_loss += scalars.density;
                }
            } else {
                for &idx in chunk {
                    let loss = example_eval_loss_bounded(
                        params,
                        &adapters[idx],
                        &targets[idx],
                        config,
                        seed.wrapping_add(idx as u64),
                    );
                    let scalars = loss_scalars(&loss)?;
                    summary.mean_total_loss += scalars.total;
                    summary.max_total_loss = summary.max_total_loss.max(scalars.total);
                    summary.mean_splat_loss += scalars.splat;
                    summary.mean_color_loss += scalars.color;
                    summary.mean_density_loss += scalars.density;
                }
            }
        }
        let scale = 1.0 / indices.len() as f32;
        summary.mean_total_loss *= scale;
        summary.mean_splat_loss *= scale;
        summary.mean_color_loss *= scale;
        summary.mean_density_loss *= scale;
        Ok(Some(summary))
    }

    fn normalized_eval_batch_size(requested: usize, examples: usize) -> usize {
        if requested == 0 {
            examples.max(1)
        } else {
            requested.min(examples).max(1)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn rollout_single_chunk(
        params: &BurnBaseParams,
        adapter: &BurnAdapterParams,
        target: &BurnTargetExample,
        mut x: Tensor2,
        mut s: Tensor2,
        config: DirectBasisTrainConfig,
        rng: &mut StdRng,
        steps: usize,
        mut displacement: Tensor1,
    ) -> (Tensor2, Tensor2, Tensor1) {
        for _ in 0..steps {
            let features = dense_perception(&x, &s, config);
            let update = params.forward_adapter(features, adapter, config);
            let dx_raw = update.clone().narrow(1, 0, 2);
            let ds = update.narrow(1, 2, s.shape().dims::<2>()[1]);
            let norm = dx_raw
                .clone()
                .mul(dx_raw.clone())
                .sum_dim(1)
                .add_scalar(EPSILON * EPSILON)
                .sqrt()
                .add_scalar(1.0)
                .expand([target.particle_count, 2]);
            let dx = dx_raw.mul_scalar(config.motion_scale).div(norm);
            let dx_norm = dx
                .clone()
                .mul(dx.clone())
                .sum_dim(1)
                .add_scalar(EPSILON * EPSILON)
                .sqrt()
                .mean();
            displacement = displacement + dx_norm;
            let mask = tensor(
                stochastic_mask(target.particle_count, target.update_prob, rng),
                [target.particle_count, 1],
                &target.target_rgb.device(),
            );
            let state_dims = s.shape().dims::<2>()[1];
            x = x + dx.mul(mask.clone().expand([target.particle_count, 2]));
            s = s + ds.mul(mask.expand([target.particle_count, state_dims]));
        }
        (x, s, displacement)
    }

    #[allow(clippy::too_many_arguments)]
    fn rollout_batch_chunk(
        params: &BurnBaseParams,
        adapter_batch: &BurnAdapterBatch,
        targets: &[BurnTargetExample],
        indices: &[usize],
        mut x: Tensor3,
        mut s: Tensor3,
        config: DirectBasisTrainConfig,
        particle_count: usize,
        rng: &mut StdRng,
        steps: usize,
        mut displacement: Tensor1,
    ) -> (Tensor3, Tensor3, Tensor1) {
        for _ in 0..steps {
            let features = dense_perception_batch(&x, &s, config);
            let update = params.forward_adapter_batch(features, adapter_batch);
            let state_dims = s.shape().dims::<3>()[2];
            let dx_raw = update.clone().narrow(2, 0, 2);
            let ds = update.narrow(2, 2, state_dims);
            let norm = dx_raw
                .clone()
                .mul(dx_raw.clone())
                .sum_dim(2)
                .add_scalar(EPSILON * EPSILON)
                .sqrt()
                .add_scalar(1.0)
                .expand([indices.len(), particle_count, 2]);
            let dx = dx_raw.mul_scalar(config.motion_scale).div(norm);
            let dx_norm = dx
                .clone()
                .mul(dx.clone())
                .sum_dim(2)
                .add_scalar(EPSILON * EPSILON)
                .sqrt()
                .mean();
            displacement = displacement + dx_norm;
            let mask = tensor3(
                batch_masks(targets, indices, particle_count, rng),
                [indices.len(), particle_count, 1],
                &targets[indices[0]].target_rgb.device(),
            );
            x = x + dx.mul(mask.clone().expand([indices.len(), particle_count, 2]));
            s = s + ds.mul(mask.expand([indices.len(), particle_count, state_dims]));
        }
        (x, s, displacement)
    }

    #[allow(clippy::too_many_arguments)]
    fn rollout_batch_eval_chunk(
        params: &BurnBaseParams,
        adapter_batch: &BurnAdapterBatch,
        targets: &[BurnTargetExample],
        indices: &[usize],
        mut x: Tensor3,
        mut s: Tensor3,
        config: DirectBasisTrainConfig,
        particle_count: usize,
        rngs: &mut [StdRng],
        steps: usize,
        mut displacement: Tensor1,
    ) -> (Tensor3, Tensor3, Tensor1) {
        for _ in 0..steps {
            let features = dense_perception_batch(&x, &s, config);
            let update = params.forward_adapter_batch(features, adapter_batch);
            let state_dims = s.shape().dims::<3>()[2];
            let dx_raw = update.clone().narrow(2, 0, 2);
            let ds = update.narrow(2, 2, state_dims);
            let norm = dx_raw
                .clone()
                .mul(dx_raw.clone())
                .sum_dim(2)
                .add_scalar(EPSILON * EPSILON)
                .sqrt()
                .add_scalar(1.0)
                .expand([indices.len(), particle_count, 2]);
            let dx = dx_raw.mul_scalar(config.motion_scale).div(norm);
            let dx_norm = dx
                .clone()
                .mul(dx.clone())
                .sum_dim(2)
                .add_scalar(EPSILON * EPSILON)
                .sqrt()
                .reshape([indices.len(), particle_count])
                .mean_dim(1)
                .squeeze_dim::<1>(1);
            displacement = displacement + dx_norm;
            let mask = tensor3(
                batch_masks_with_rngs(targets, indices, particle_count, rngs),
                [indices.len(), particle_count, 1],
                &targets[indices[0]].target_rgb.device(),
            );
            x = x + dx.mul(mask.clone().expand([indices.len(), particle_count, 2]));
            s = s + ds.mul(mask.expand([indices.len(), particle_count, state_dims]));
        }
        (x, s, displacement)
    }

    fn example_eval_loss_bounded(
        params: &BurnBaseParams,
        adapter: &BurnAdapterParams,
        target: &BurnTargetExample,
        config: DirectBasisTrainConfig,
        seed: u64,
    ) -> BurnLossTensors {
        let device = &target.target_rgb.device();
        let (mut x, mut s) = seed_tensors(
            target.particle_count,
            config,
            target.seed_scale,
            seed,
            device,
        );
        let mut rng = StdRng::seed_from_u64(seed ^ 0x005e_ed2d);
        let mut displacement = Tensor::<BurnBackend, 1>::zeros([1], device);
        let chunk_steps = tbptt_chunk_steps(config);
        let mut remaining_steps = config.rollout_steps;
        while remaining_steps > 0 {
            let steps = remaining_steps.min(chunk_steps);
            (x, s, displacement) = rollout_single_chunk(
                params,
                adapter,
                target,
                x,
                s,
                config,
                &mut rng,
                steps,
                displacement,
            );
            remaining_steps -= steps;
            if remaining_steps > 0 {
                x = detach2(x);
                s = detach2(s);
                displacement = detach1(displacement);
            }
        }
        target_splat_loss(&x, &s, target, config, adapter, displacement)
    }

    fn batch_example_eval_loss(
        params: &BurnBaseParams,
        adapters: &[BurnAdapterParams],
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        seed: u64,
    ) -> Result<BurnLossBatchTensors, Box<dyn std::error::Error>> {
        let Some(particle_count) = homogeneous_particle_count(targets, indices) else {
            return Err(std::io::Error::other(
                "Burn eval batch path requires homogeneous particle counts",
            )
            .into());
        };
        let device = &targets[indices[0]].target_rgb.device();
        let adapter_batch = BurnAdapterBatch::from_indices(adapters, indices);
        let (mut x, mut s) =
            seed_batch_tensors(targets, indices, particle_count, config, seed, device);
        let mut rngs = indices
            .iter()
            .map(|idx| StdRng::seed_from_u64(seed.wrapping_add(*idx as u64) ^ 0x005e_ed2d))
            .collect::<Vec<_>>();
        let mut displacement = Tensor::<BurnBackend, 1>::zeros([indices.len()], device);
        let chunk_steps = tbptt_chunk_steps(config);
        let mut remaining_steps = config.rollout_steps;
        while remaining_steps > 0 {
            let steps = remaining_steps.min(chunk_steps);
            (x, s, displacement) = rollout_batch_eval_chunk(
                params,
                &adapter_batch,
                targets,
                indices,
                x,
                s,
                config,
                particle_count,
                &mut rngs,
                steps,
                displacement,
            );
            remaining_steps -= steps;
            if remaining_steps > 0 {
                x = detach3(x);
                s = detach3(s);
                displacement = detach1(displacement);
            }
        }
        Ok(target_splat_loss_batch_vector(
            &x,
            &s,
            targets,
            indices,
            config,
            &adapter_batch,
            displacement,
        ))
    }

    fn homogeneous_particle_count(
        targets: &[BurnTargetExample],
        indices: &[usize],
    ) -> Option<usize> {
        let mut iter = indices.iter().map(|idx| targets[*idx].particle_count);
        let first = iter.next()?;
        iter.all(|count| count == first).then_some(first)
    }

    fn dense_perception(x: &Tensor2, s: &Tensor2, config: DirectBasisTrainConfig) -> Tensor2 {
        let dims = s.shape().dims::<2>();
        let rows = dims[0];
        let state_dims = dims[1];
        let density = dense_particle_density(x, config);
        let chunk_size = dense_query_chunk_size(1, rows, state_dims, config.max_dense_chunk_floats);
        let mut chunks = Vec::new();
        for (start, len) in chunks_for(rows, chunk_size) {
            chunks.push(dense_perception_chunk(x, s, &density, config, start, len));
        }
        Tensor::cat(chunks, 0)
    }

    fn dense_perception_chunk(
        x: &Tensor2,
        s: &Tensor2,
        density: &Tensor2,
        config: DirectBasisTrainConfig,
        start: usize,
        len: usize,
    ) -> Tensor2 {
        let dims = s.shape().dims::<2>();
        let rows = dims[0];
        let state_dims = dims[1];
        let xi = x
            .clone()
            .narrow(0, start, len)
            .unsqueeze_dim::<3>(1)
            .expand([len, rows, 2]);
        let xj = x.clone().unsqueeze_dim::<3>(0).expand([len, rows, 2]);
        let diff = xj - xi;
        let dist2 = diff
            .clone()
            .mul(diff.clone())
            .sum_dim(2)
            .squeeze_dim::<2>(2);
        let eps = config.grid_eps.max(EPSILON);
        let compact = relu(dist2.clone().mul_scalar(-1.0).add_scalar(eps * eps));
        let compact2 = compact.clone().mul(compact.clone());
        let smooth = compact2
            .mul(compact)
            .mul_scalar(4.0 / (std::f32::consts::PI * eps.powi(8)));
        let volume_j = density.clone().transpose().recip().expand([len, rows]);
        let blur = smooth.clone().mul(volume_j.clone()).matmul(s.clone());

        let r = dist2.add_scalar(EPSILON * EPSILON).sqrt();
        let spiky = relu(r.clone().mul_scalar(-1.0).add_scalar(eps));
        let spiky_mag = spiky
            .clone()
            .mul(spiky)
            .div(r)
            .mul_scalar(30.0 / (std::f32::consts::PI * eps.powi(5)));
        let grad = diff
            .clone()
            .mul(spiky_mag.unsqueeze_dim::<3>(2).expand([len, rows, 2]));
        let density_grad = log_normalize_vectors(
            grad.clone()
                .sum_dim(1)
                .squeeze_dim::<2>(1)
                .mul_scalar((eps / 0.1).powi(3) / rows.max(1) as f32),
        );

        let sj = s
            .clone()
            .unsqueeze_dim::<3>(0)
            .expand([len, rows, state_dims]);
        let si = s
            .clone()
            .narrow(0, start, len)
            .unsqueeze_dim::<3>(1)
            .expand([len, rows, state_dims]);
        let state_diff = sj - si;
        let volume_grad = grad.mul(volume_j.unsqueeze_dim::<3>(2).expand([len, rows, 2]));
        let state_grad = state_diff
            .unsqueeze_dim::<4>(3)
            .expand([len, rows, state_dims, 2])
            .mul(
                volume_grad
                    .clone()
                    .unsqueeze_dim::<4>(2)
                    .expand([len, rows, state_dims, 2]),
            )
            .sum_dim(1)
            .squeeze_dim::<3>(1);
        let state_grad = apply_moment_correction_2d(state_grad, diff, volume_grad);
        let state_grad = log_normalize_state_gradient(state_grad);

        Tensor::cat(
            vec![
                s.clone().narrow(0, start, len),
                blur,
                state_grad,
                density_grad,
            ],
            1,
        )
    }

    fn dense_perception_batch(x: &Tensor3, s: &Tensor3, config: DirectBasisTrainConfig) -> Tensor3 {
        let dims = s.shape().dims::<3>();
        let batches = dims[0];
        let rows = dims[1];
        let state_dims = dims[2];
        let density = dense_particle_density_batch(x, config);
        let chunk_size =
            dense_query_chunk_size(batches, rows, state_dims, config.max_dense_chunk_floats);
        let mut chunks = Vec::new();
        for (start, len) in chunks_for(rows, chunk_size) {
            chunks.push(dense_perception_batch_chunk(
                x, s, &density, config, start, len,
            ));
        }
        Tensor::cat(chunks, 1)
    }

    fn dense_perception_batch_chunk(
        x: &Tensor3,
        s: &Tensor3,
        density: &Tensor3,
        config: DirectBasisTrainConfig,
        start: usize,
        len: usize,
    ) -> Tensor3 {
        let dims = s.shape().dims::<3>();
        let batches = dims[0];
        let rows = dims[1];
        let state_dims = dims[2];
        let xi = x
            .clone()
            .narrow(1, start, len)
            .unsqueeze_dim::<4>(2)
            .expand([batches, len, rows, 2]);
        let xj = x
            .clone()
            .unsqueeze_dim::<4>(1)
            .expand([batches, len, rows, 2]);
        let diff = xj - xi;
        let dist2 = diff
            .clone()
            .mul(diff.clone())
            .sum_dim(3)
            .squeeze_dim::<3>(3);
        let eps = config.grid_eps.max(EPSILON);
        let compact = relu(dist2.clone().mul_scalar(-1.0).add_scalar(eps * eps));
        let compact2 = compact.clone().mul(compact.clone());
        let smooth = compact2
            .mul(compact)
            .mul_scalar(4.0 / (std::f32::consts::PI * eps.powi(8)));
        let volume_j = density
            .clone()
            .swap_dims(1, 2)
            .recip()
            .expand([batches, len, rows]);
        let blur = smooth.clone().mul(volume_j.clone()).matmul(s.clone());

        let r = dist2.add_scalar(EPSILON * EPSILON).sqrt();
        let spiky = relu(r.clone().mul_scalar(-1.0).add_scalar(eps));
        let spiky_mag = spiky
            .clone()
            .mul(spiky)
            .div(r)
            .mul_scalar(30.0 / (std::f32::consts::PI * eps.powi(5)));
        let grad = diff.clone().mul(
            spiky_mag
                .unsqueeze_dim::<4>(3)
                .expand([batches, len, rows, 2]),
        );
        let density_grad = log_normalize_vectors_batch(
            grad.clone()
                .sum_dim(2)
                .squeeze_dim::<3>(2)
                .mul_scalar((eps / 0.1).powi(3) / rows.max(1) as f32),
        );

        let sj = s
            .clone()
            .unsqueeze_dim::<4>(1)
            .expand([batches, len, rows, state_dims]);
        let si = s
            .clone()
            .narrow(1, start, len)
            .unsqueeze_dim::<4>(2)
            .expand([batches, len, rows, state_dims]);
        let state_diff = sj - si;
        let volume_grad = grad.mul(
            volume_j
                .unsqueeze_dim::<4>(3)
                .expand([batches, len, rows, 2]),
        );
        let state_grad = state_diff
            .unsqueeze_dim::<5>(4)
            .expand([batches, len, rows, state_dims, 2])
            .mul(
                volume_grad
                    .clone()
                    .unsqueeze_dim::<5>(3)
                    .expand([batches, len, rows, state_dims, 2]),
            )
            .sum_dim(2)
            .squeeze_dim::<4>(2);
        let state_grad = apply_moment_correction_2d_batch(state_grad, diff, volume_grad);
        let state_grad = log_normalize_state_gradient_batch(state_grad);

        Tensor::cat(
            vec![
                s.clone().narrow(1, start, len),
                blur,
                state_grad,
                density_grad,
            ],
            2,
        )
    }

    fn target_splat_loss(
        x: &Tensor2,
        s: &Tensor2,
        target: &BurnTargetExample,
        config: DirectBasisTrainConfig,
        adapter: &BurnAdapterParams,
        displacement: Tensor1,
    ) -> BurnLossTensors {
        let particle_count = x.shape().dims::<2>()[0];
        let state_dims = s.shape().dims::<2>()[1];
        let centered = if config.loss_config.center {
            x.clone() - x.clone().mean_dim(0).expand([particle_count, 2])
                + target.target_mean.clone().expand([particle_count, 2])
        } else {
            x.clone()
        };
        let colors = s.clone().narrow(1, state_dims - 3, 3).add_scalar(0.5);
        let (rgb, density) = splat_render(&centered, &colors, target, config, particle_count);
        let density_diff = density - target.target_density.clone();
        let density_term = l1l2_tensor(density_diff);
        let density_loss = density_term.clone().mean();
        let color_gate = density_term.mul_scalar(-1.0).exp().expand([
            config.loss_config.image_size * config.loss_config.image_size,
            3,
        ]);
        let color_loss = l1l2_tensor(rgb - target.target_rgb.clone())
            .mul(color_gate)
            .mean();
        let splat = color_loss
            .clone()
            .mul_scalar(config.loss_config.color_loss_weight)
            + density_loss
                .clone()
                .mul_scalar(config.loss_config.density_loss_weight);
        let bound = relu(x.clone().abs().add_scalar(-1.0));
        let bound_loss = bound.mean();
        let overflow = relu(s.clone().abs().add_scalar(-1.0));
        let overflow_loss = overflow.mean();
        let mut total = splat
            .clone()
            .mul_scalar(config.loss_config.splat_loss_weight)
            + displacement.mul_scalar(config.loss_config.displacement_regularizer_weight)
            + bound_loss.mul_scalar(config.loss_config.bound_regularizer_weight)
            + overflow_loss.mul_scalar(config.loss_config.overflow_regularizer_weight);
        if config.adapter_l2_weight > 0.0 {
            total = total + adapter.l2_loss().mul_scalar(config.adapter_l2_weight);
        }
        BurnLossTensors {
            total,
            splat,
            color: color_loss,
            density: density_loss,
        }
    }

    fn target_splat_loss_batch(
        x: &Tensor3,
        s: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        adapter: &BurnAdapterBatch,
        displacement: Tensor1,
    ) -> BurnLossTensors {
        let dims = x.shape().dims::<3>();
        let batches = dims[0];
        let particle_count = dims[1];
        let state_dims = s.shape().dims::<3>()[2];
        let target_mean = stack_target_mean(targets, indices);
        let centered = if config.loss_config.center {
            x.clone() - x.clone().mean_dim(1).expand([batches, particle_count, 2])
                + target_mean.expand([batches, particle_count, 2])
        } else {
            x.clone()
        };
        let colors = s.clone().narrow(2, state_dims - 3, 3).add_scalar(0.5);
        let (rgb, density) =
            splat_render_batch(&centered, &colors, targets, indices, config, particle_count);
        let target_density = stack_target_density(targets, indices);
        let density_diff = density - target_density;
        let density_term = l1l2_tensor3(density_diff);
        let density_loss = density_term.clone().mean();
        let color_gate = density_term.mul_scalar(-1.0).exp().expand([
            batches,
            config.loss_config.image_size * config.loss_config.image_size,
            3,
        ]);
        let color_loss = l1l2_tensor3(rgb - stack_target_rgb(targets, indices))
            .mul(color_gate)
            .mean();
        let splat = color_loss
            .clone()
            .mul_scalar(config.loss_config.color_loss_weight)
            + density_loss
                .clone()
                .mul_scalar(config.loss_config.density_loss_weight);
        let bound_loss = relu(x.clone().abs().add_scalar(-1.0)).mean();
        let overflow_loss = relu(s.clone().abs().add_scalar(-1.0)).mean();
        let mut total = splat
            .clone()
            .mul_scalar(config.loss_config.splat_loss_weight)
            + displacement.mul_scalar(config.loss_config.displacement_regularizer_weight)
            + bound_loss.mul_scalar(config.loss_config.bound_regularizer_weight)
            + overflow_loss.mul_scalar(config.loss_config.overflow_regularizer_weight);
        if config.adapter_l2_weight > 0.0 {
            total = total + adapter.l2_loss().mul_scalar(config.adapter_l2_weight);
        }
        BurnLossTensors {
            total,
            splat,
            color: color_loss,
            density: density_loss,
        }
    }

    fn target_splat_loss_batch_vector(
        x: &Tensor3,
        s: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        adapter: &BurnAdapterBatch,
        displacement: Tensor1,
    ) -> BurnLossBatchTensors {
        let dims = x.shape().dims::<3>();
        let batches = dims[0];
        let particle_count = dims[1];
        let state_dims = s.shape().dims::<3>()[2];
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let target_mean = stack_target_mean(targets, indices);
        let centered = if config.loss_config.center {
            x.clone() - x.clone().mean_dim(1).expand([batches, particle_count, 2])
                + target_mean.expand([batches, particle_count, 2])
        } else {
            x.clone()
        };
        let colors = s.clone().narrow(2, state_dims - 3, 3).add_scalar(0.5);
        let (rgb, density) =
            splat_render_batch(&centered, &colors, targets, indices, config, particle_count);
        let density_diff = density - stack_target_density(targets, indices);
        let density_term = l1l2_tensor3(density_diff);
        let density_loss = density_term
            .clone()
            .reshape([batches, pixels])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let color_gate = density_term
            .mul_scalar(-1.0)
            .exp()
            .expand([batches, pixels, 3]);
        let color_loss = l1l2_tensor3(rgb - stack_target_rgb(targets, indices))
            .mul(color_gate)
            .reshape([batches, pixels * 3])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let splat = color_loss
            .clone()
            .mul_scalar(config.loss_config.color_loss_weight)
            + density_loss
                .clone()
                .mul_scalar(config.loss_config.density_loss_weight);
        let bound_loss = relu(x.clone().abs().add_scalar(-1.0))
            .reshape([batches, particle_count * 2])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let overflow_loss = relu(s.clone().abs().add_scalar(-1.0))
            .reshape([batches, particle_count * state_dims])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let mut total = splat
            .clone()
            .mul_scalar(config.loss_config.splat_loss_weight)
            + displacement.mul_scalar(config.loss_config.displacement_regularizer_weight)
            + bound_loss.mul_scalar(config.loss_config.bound_regularizer_weight)
            + overflow_loss.mul_scalar(config.loss_config.overflow_regularizer_weight);
        if config.adapter_l2_weight > 0.0 {
            total = total
                + adapter
                    .l2_loss_vector()
                    .mul_scalar(config.adapter_l2_weight);
        }
        BurnLossBatchTensors {
            total,
            splat,
            color: color_loss,
            density: density_loss,
        }
    }

    fn splat_render(
        x: &Tensor2,
        colors: &Tensor2,
        target: &BurnTargetExample,
        config: DirectBasisTrainConfig,
        particle_count: usize,
    ) -> (Tensor2, Tensor2) {
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let particle_pixels = particle_pixel_positions(x, config);
        let sigma =
            (config.loss_config.sigma * config.loss_config.image_size as f32 * target.pixel_size
                / (config.loss_config.hi - config.loss_config.lo))
                .max(EPSILON);
        let denom =
            splat_particle_denominator(&particle_pixels, target, particle_count, sigma, config);
        let norm_scale = (config.loss_config.image_size as f32 * target.pixel_size
            / (config.loss_config.hi - config.loss_config.lo))
            .powi(2);
        let output_scale = target.target_points as f32 / particle_count.max(1) as f32;
        let chunk_size =
            splat_pixel_chunk_size(1, particle_count, pixels, config.max_splat_chunk_floats);
        let mut rgbs = Vec::new();
        let mut densities = Vec::new();
        for (start, len) in chunks_for(pixels, chunk_size) {
            let g =
                splat_gaussian_chunk(&particle_pixels, target, particle_count, sigma, start, len);
            let weights = g
                .div(denom.clone().expand([len, particle_count]))
                .mul_scalar(output_scale * norm_scale);
            densities.push(weights.clone().sum_dim(1));
            rgbs.push(weights.matmul(colors.clone()));
        }
        (Tensor::cat(rgbs, 0), Tensor::cat(densities, 0))
    }

    fn splat_render_batch(
        x: &Tensor3,
        colors: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        particle_count: usize,
    ) -> (Tensor3, Tensor3) {
        let batches = indices.len();
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let particle_pixels = particle_pixel_positions_batch(x, config);
        let sigma = stack_pixel_sizes(targets, indices)
            .mul_scalar(config.loss_config.sigma * config.loss_config.image_size as f32)
            .div_scalar(config.loss_config.hi - config.loss_config.lo)
            .clamp_min(EPSILON);
        let denom = splat_particle_denominator_batch(
            &particle_pixels,
            targets,
            indices,
            particle_count,
            sigma.clone(),
            config,
        );
        let norm_scale = stack_pixel_sizes(targets, indices)
            .mul_scalar(config.loss_config.image_size as f32)
            .div_scalar(config.loss_config.hi - config.loss_config.lo);
        let norm_scale = norm_scale.clone().mul(norm_scale);
        let output_scale =
            stack_target_point_counts(targets, indices).div_scalar(particle_count.max(1) as f32);
        let chunk_size = splat_pixel_chunk_size(
            batches,
            particle_count,
            pixels,
            config.max_splat_chunk_floats,
        );
        let mut rgbs = Vec::new();
        let mut densities = Vec::new();
        for (start, len) in chunks_for(pixels, chunk_size) {
            let g = splat_gaussian_batch_chunk(
                &particle_pixels,
                targets,
                indices,
                particle_count,
                sigma.clone(),
                start,
                len,
            );
            let weights = g
                .div(denom.clone().expand([batches, len, particle_count]))
                .mul(norm_scale.clone().expand([batches, len, particle_count]))
                .mul(output_scale.clone().expand([batches, len, particle_count]));
            densities.push(weights.clone().sum_dim(2));
            rgbs.push(weights.matmul(colors.clone()));
        }
        (Tensor::cat(rgbs, 1), Tensor::cat(densities, 1))
    }

    fn dense_particle_density(x: &Tensor2, config: DirectBasisTrainConfig) -> Tensor2 {
        let rows = x.shape().dims::<2>()[0];
        let chunk_size = dense_query_chunk_size(1, rows, 1, config.max_dense_chunk_floats);
        let mut chunks = Vec::new();
        for (start, len) in chunks_for(rows, chunk_size) {
            let xi = x
                .clone()
                .narrow(0, start, len)
                .unsqueeze_dim::<3>(1)
                .expand([len, rows, 2]);
            let xj = x.clone().unsqueeze_dim::<3>(0).expand([len, rows, 2]);
            let diff = xj - xi;
            let dist2 = diff.clone().mul(diff).sum_dim(2).squeeze_dim::<2>(2);
            let eps = config.grid_eps.max(EPSILON);
            let compact = relu(dist2.mul_scalar(-1.0).add_scalar(eps * eps));
            let compact2 = compact.clone().mul(compact.clone());
            chunks.push(
                compact2
                    .mul(compact)
                    .mul_scalar(4.0 / (std::f32::consts::PI * eps.powi(8)))
                    .sum_dim(1)
                    .clamp_min(EPSILON),
            );
        }
        Tensor::cat(chunks, 0)
    }

    fn dense_particle_density_batch(x: &Tensor3, config: DirectBasisTrainConfig) -> Tensor3 {
        let dims = x.shape().dims::<3>();
        let batches = dims[0];
        let rows = dims[1];
        let chunk_size = dense_query_chunk_size(batches, rows, 1, config.max_dense_chunk_floats);
        let mut chunks = Vec::new();
        for (start, len) in chunks_for(rows, chunk_size) {
            let xi = x
                .clone()
                .narrow(1, start, len)
                .unsqueeze_dim::<4>(2)
                .expand([batches, len, rows, 2]);
            let xj = x
                .clone()
                .unsqueeze_dim::<4>(1)
                .expand([batches, len, rows, 2]);
            let diff = xj - xi;
            let dist2 = diff.clone().mul(diff).sum_dim(3).squeeze_dim::<3>(3);
            let eps = config.grid_eps.max(EPSILON);
            let compact = relu(dist2.mul_scalar(-1.0).add_scalar(eps * eps));
            let compact2 = compact.clone().mul(compact.clone());
            chunks.push(
                compact2
                    .mul(compact)
                    .mul_scalar(4.0 / (std::f32::consts::PI * eps.powi(8)))
                    .sum_dim(2)
                    .clamp_min(EPSILON),
            );
        }
        Tensor::cat(chunks, 1)
    }

    fn splat_particle_denominator(
        particle_pixels: &Tensor2,
        target: &BurnTargetExample,
        particle_count: usize,
        sigma: f32,
        config: DirectBasisTrainConfig,
    ) -> Tensor2 {
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let chunk_size =
            splat_pixel_chunk_size(1, particle_count, pixels, config.max_splat_chunk_floats);
        let mut denom = None::<Tensor2>;
        for (start, len) in chunks_for(pixels, chunk_size) {
            let g =
                splat_gaussian_chunk(particle_pixels, target, particle_count, sigma, start, len);
            let contribution = g.sum_dim(0);
            denom = Some(match denom {
                Some(value) => value + contribution,
                None => contribution,
            });
        }
        denom
            .unwrap_or_else(|| {
                Tensor::<BurnBackend, 2>::zeros([1, particle_count], &target.target_rgb.device())
            })
            .add_scalar(EPSILON)
    }

    fn splat_particle_denominator_batch(
        particle_pixels: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        sigma: Tensor3,
        config: DirectBasisTrainConfig,
    ) -> Tensor3 {
        let batches = indices.len();
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let chunk_size = splat_pixel_chunk_size(
            batches,
            particle_count,
            pixels,
            config.max_splat_chunk_floats,
        );
        let mut denom = None::<Tensor3>;
        for (start, len) in chunks_for(pixels, chunk_size) {
            let g = splat_gaussian_batch_chunk(
                particle_pixels,
                targets,
                indices,
                particle_count,
                sigma.clone(),
                start,
                len,
            );
            let contribution = g.sum_dim(1);
            denom = Some(match denom {
                Some(value) => value + contribution,
                None => contribution,
            });
        }
        denom
            .unwrap_or_else(|| {
                Tensor::<BurnBackend, 3>::zeros(
                    [batches, 1, particle_count],
                    &targets[indices[0]].target_rgb.device(),
                )
            })
            .add_scalar(EPSILON)
    }

    fn splat_gaussian_chunk(
        particle_pixels: &Tensor2,
        target: &BurnTargetExample,
        particle_count: usize,
        sigma: f32,
        start: usize,
        len: usize,
    ) -> Tensor2 {
        let pixel_i = target
            .pixel_xy
            .clone()
            .narrow(0, start, len)
            .unsqueeze_dim::<3>(1)
            .expand([len, particle_count, 2]);
        let particle_j =
            particle_pixels
                .clone()
                .unsqueeze_dim::<3>(0)
                .expand([len, particle_count, 2]);
        let diff = pixel_i - particle_j;
        let dist2 = diff.clone().mul(diff).sum_dim(2).squeeze_dim::<2>(2);
        dist2.mul_scalar(-0.5 / (sigma * sigma)).exp()
    }

    fn splat_gaussian_batch_chunk(
        particle_pixels: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        sigma: Tensor3,
        start: usize,
        len: usize,
    ) -> Tensor3 {
        let batches = indices.len();
        let pixel_i = targets[indices[0]]
            .pixel_xy
            .clone()
            .narrow(0, start, len)
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(2)
            .expand([batches, len, particle_count, 2]);
        let particle_j =
            particle_pixels
                .clone()
                .unsqueeze_dim::<4>(1)
                .expand([batches, len, particle_count, 2]);
        let diff = pixel_i - particle_j;
        let dist2 = diff.clone().mul(diff).sum_dim(3).squeeze_dim::<3>(3);
        let sigma2 = sigma
            .clone()
            .mul(sigma)
            .expand([batches, len, particle_count]);
        dist2.mul_scalar(-0.5).div(sigma2).exp()
    }

    fn chunks_for(total: usize, chunk_size: usize) -> impl Iterator<Item = (usize, usize)> {
        let chunk_size = chunk_size.max(1);
        (0..total)
            .step_by(chunk_size)
            .map(move |start| (start, (total - start).min(chunk_size)))
    }

    fn dense_query_chunk_size(
        batches: usize,
        rows: usize,
        state_dims: usize,
        max_floats: usize,
    ) -> usize {
        let denominator = batches
            .max(1)
            .saturating_mul(rows.max(1))
            .saturating_mul(state_dims.max(1))
            .saturating_mul(2)
            .max(1);
        (max_floats / denominator).max(1).min(rows.max(1))
    }

    fn splat_pixel_chunk_size(
        batches: usize,
        particle_count: usize,
        pixels: usize,
        max_floats: usize,
    ) -> usize {
        let denominator = batches
            .max(1)
            .saturating_mul(particle_count.max(1))
            .saturating_mul(2)
            .max(1);
        (max_floats / denominator).max(1).min(pixels.max(1))
    }

    fn particle_pixel_positions(x: &Tensor2, config: DirectBasisTrainConfig) -> Tensor2 {
        let size = config.loss_config.image_size as f32;
        let world_scale = (size - 1.0) / (config.loss_config.hi - config.loss_config.lo);
        let px = x
            .clone()
            .narrow(1, 0, 1)
            .add_scalar(-config.loss_config.lo)
            .mul_scalar(world_scale);
        let py = x
            .clone()
            .narrow(1, 1, 1)
            .add_scalar(-config.loss_config.lo)
            .mul_scalar(-world_scale)
            .add_scalar(size - 1.0);
        Tensor::cat(vec![px, py], 1)
    }

    fn particle_pixel_positions_batch(x: &Tensor3, config: DirectBasisTrainConfig) -> Tensor3 {
        let size = config.loss_config.image_size as f32;
        let world_scale = (size - 1.0) / (config.loss_config.hi - config.loss_config.lo);
        let px = x
            .clone()
            .narrow(2, 0, 1)
            .add_scalar(-config.loss_config.lo)
            .mul_scalar(world_scale);
        let py = x
            .clone()
            .narrow(2, 1, 1)
            .add_scalar(-config.loss_config.lo)
            .mul_scalar(-world_scale)
            .add_scalar(size - 1.0);
        Tensor::cat(vec![px, py], 2)
    }

    fn l1l2_tensor(value: Tensor2) -> Tensor2 {
        value.clone().abs() + value.clone().mul(value)
    }

    fn l1l2_tensor3(value: Tensor3) -> Tensor3 {
        value.clone().abs() + value.clone().mul(value)
    }

    fn log_normalize_vectors(values: Tensor2) -> Tensor2 {
        let dims = values.shape().dims::<2>();
        let norm = values
            .clone()
            .mul(values.clone())
            .sum_dim(1)
            .add_scalar(EPSILON * EPSILON)
            .sqrt()
            .clamp_min(EPSILON);
        values * norm.clone().log1p().div(norm).expand([dims[0], dims[1]])
    }

    fn log_normalize_vectors_batch(values: Tensor3) -> Tensor3 {
        let dims = values.shape().dims::<3>();
        let norm = values
            .clone()
            .mul(values.clone())
            .sum_dim(2)
            .add_scalar(EPSILON * EPSILON)
            .sqrt()
            .clamp_min(EPSILON);
        values
            * norm
                .clone()
                .log1p()
                .div(norm)
                .expand([dims[0], dims[1], dims[2]])
    }

    fn log_normalize_state_gradient(values: Tensor3) -> Tensor2 {
        let dims = values.shape().dims::<3>();
        let norm = values
            .clone()
            .mul(values.clone())
            .sum_dim(2)
            .add_scalar(EPSILON * EPSILON)
            .sqrt()
            .clamp_min(EPSILON);
        (values
            * norm
                .clone()
                .log1p()
                .div(norm)
                .expand([dims[0], dims[1], dims[2]]))
        .reshape([dims[0], dims[1] * dims[2]])
    }

    fn log_normalize_state_gradient_batch(values: Tensor4) -> Tensor3 {
        let dims = values.shape().dims::<4>();
        let norm = values
            .clone()
            .mul(values.clone())
            .sum_dim(3)
            .add_scalar(EPSILON * EPSILON)
            .sqrt()
            .clamp_min(EPSILON);
        (values
            * norm
                .clone()
                .log1p()
                .div(norm)
                .expand([dims[0], dims[1], dims[2], dims[3]]))
        .reshape([dims[0], dims[1], dims[2] * dims[3]])
    }

    fn apply_moment_correction_2d(
        state_gradient: Tensor3,
        diff: Tensor3,
        volume_grad: Tensor3,
    ) -> Tensor3 {
        let dims = state_gradient.shape().dims::<3>();
        let query_rows = dims[0];
        let state_dims = dims[1];
        let neighbor_rows = diff.shape().dims::<3>()[1];
        let moment = diff
            .unsqueeze_dim::<4>(3)
            .expand([query_rows, neighbor_rows, 2, 2])
            .mul(
                volume_grad
                    .unsqueeze_dim::<4>(2)
                    .expand([query_rows, neighbor_rows, 2, 2]),
            )
            .sum_dim(1)
            .squeeze_dim::<3>(1);
        let a = moment
            .clone()
            .narrow(1, 0, 1)
            .narrow(2, 0, 1)
            .reshape([query_rows, 1]);
        let b = moment
            .clone()
            .narrow(1, 0, 1)
            .narrow(2, 1, 1)
            .reshape([query_rows, 1]);
        let d = moment
            .narrow(1, 1, 1)
            .narrow(2, 1, 1)
            .reshape([query_rows, 1]);
        let det = a.clone().mul(d.clone()) - b.clone().mul(b.clone());
        let near_singular = det.clone().abs().lower_elem(1.0e-3);
        let ones = Tensor::<BurnBackend, 2>::ones([query_rows, 1], &state_gradient.device());
        let zeros = Tensor::<BurnBackend, 2>::zeros([query_rows, 1], &state_gradient.device());
        let inv_det = det.mask_where(near_singular.clone(), ones.clone()).recip();
        let inv00 = d
            .mul(inv_det.clone())
            .mask_where(near_singular.clone(), ones);
        let inv01 = b
            .mul_scalar(-1.0)
            .mul(inv_det.clone())
            .mask_where(near_singular.clone(), zeros.clone());
        let inv11 = a.mul(inv_det).mask_where(
            near_singular,
            Tensor::<BurnBackend, 2>::ones([query_rows, 1], &state_gradient.device()),
        );
        let gx = state_gradient.clone().narrow(2, 0, 1);
        let gy = state_gradient.narrow(2, 1, 1);
        let inv00 = inv00
            .unsqueeze_dim::<3>(1)
            .expand([query_rows, state_dims, 1]);
        let inv01 = inv01
            .unsqueeze_dim::<3>(1)
            .expand([query_rows, state_dims, 1]);
        let inv11 = inv11
            .unsqueeze_dim::<3>(1)
            .expand([query_rows, state_dims, 1]);
        let corrected_x = gx.clone().mul(inv00) + gy.clone().mul(inv01.clone());
        let corrected_y = gx.mul(inv01) + gy.mul(inv11);
        Tensor::cat(vec![corrected_x, corrected_y], 2)
    }

    fn apply_moment_correction_2d_batch(
        state_gradient: Tensor4,
        diff: Tensor4,
        volume_grad: Tensor4,
    ) -> Tensor4 {
        let dims = state_gradient.shape().dims::<4>();
        let batches = dims[0];
        let query_rows = dims[1];
        let state_dims = dims[2];
        let neighbor_rows = diff.shape().dims::<4>()[2];
        let moment = diff
            .unsqueeze_dim::<5>(4)
            .expand([batches, query_rows, neighbor_rows, 2, 2])
            .mul(volume_grad.unsqueeze_dim::<5>(3).expand([
                batches,
                query_rows,
                neighbor_rows,
                2,
                2,
            ]))
            .sum_dim(2)
            .squeeze_dim::<4>(2);
        let a = moment
            .clone()
            .narrow(2, 0, 1)
            .narrow(3, 0, 1)
            .reshape([batches, query_rows, 1]);
        let b = moment
            .clone()
            .narrow(2, 0, 1)
            .narrow(3, 1, 1)
            .reshape([batches, query_rows, 1]);
        let d = moment
            .narrow(2, 1, 1)
            .narrow(3, 1, 1)
            .reshape([batches, query_rows, 1]);
        let det = a.clone().mul(d.clone()) - b.clone().mul(b.clone());
        let near_singular = det.clone().abs().lower_elem(1.0e-3);
        let ones =
            Tensor::<BurnBackend, 3>::ones([batches, query_rows, 1], &state_gradient.device());
        let zeros =
            Tensor::<BurnBackend, 3>::zeros([batches, query_rows, 1], &state_gradient.device());
        let inv_det = det.mask_where(near_singular.clone(), ones.clone()).recip();
        let inv00 = d
            .mul(inv_det.clone())
            .mask_where(near_singular.clone(), ones);
        let inv01 = b
            .mul_scalar(-1.0)
            .mul(inv_det.clone())
            .mask_where(near_singular.clone(), zeros);
        let inv11 = a.mul(inv_det).mask_where(
            near_singular,
            Tensor::<BurnBackend, 3>::ones([batches, query_rows, 1], &state_gradient.device()),
        );
        let gx = state_gradient.clone().narrow(3, 0, 1);
        let gy = state_gradient.narrow(3, 1, 1);
        let inv00 = inv00
            .unsqueeze_dim::<4>(2)
            .expand([batches, query_rows, state_dims, 1]);
        let inv01 = inv01
            .unsqueeze_dim::<4>(2)
            .expand([batches, query_rows, state_dims, 1]);
        let inv11 = inv11
            .unsqueeze_dim::<4>(2)
            .expand([batches, query_rows, state_dims, 1]);
        let corrected_x = gx.clone().mul(inv00) + gy.clone().mul(inv01.clone());
        let corrected_y = gx.mul(inv01) + gy.mul(inv11);
        Tensor::cat(vec![corrected_x, corrected_y], 3)
    }

    #[derive(Clone, Copy)]
    struct AdamWBiasCorrection {
        beta1: f32,
        beta2: f32,
    }

    impl BurnBaseAdamWState {
        fn zeros_like(params: &BurnBaseParams) -> Self {
            Self {
                step: 0,
                w1_m: params.w1.clone().inner().zeros_like(),
                w1_v: params.w1.clone().inner().zeros_like(),
                b1_m: params.b1.clone().inner().zeros_like(),
                b1_v: params.b1.clone().inner().zeros_like(),
                w2_m: params.w2.clone().inner().zeros_like(),
                w2_v: params.w2.clone().inner().zeros_like(),
                b2_m: params.b2.clone().inner().zeros_like(),
                b2_v: params.b2.clone().inner().zeros_like(),
            }
        }

        fn next_bias_correction(&mut self, cfg: AdamWConfig) -> AdamWBiasCorrection {
            next_adamw_bias_correction(&mut self.step, cfg)
        }
    }

    impl BurnAdapterAdamWState {
        fn zeros_like(params: &BurnAdapterParams) -> Self {
            Self {
                step: 0,
                w1_down_m: params.w1_down.clone().inner().zeros_like(),
                w1_down_v: params.w1_down.clone().inner().zeros_like(),
                w1_up_m: params.w1_up.clone().inner().zeros_like(),
                w1_up_v: params.w1_up.clone().inner().zeros_like(),
                w2_down_m: params.w2_down.clone().inner().zeros_like(),
                w2_down_v: params.w2_down.clone().inner().zeros_like(),
                w2_up_m: params.w2_up.clone().inner().zeros_like(),
                w2_up_v: params.w2_up.clone().inner().zeros_like(),
                b1_delta_m: params.b1_delta.clone().inner().zeros_like(),
                b1_delta_v: params.b1_delta.clone().inner().zeros_like(),
                b2_delta_m: params.b2_delta.clone().inner().zeros_like(),
                b2_delta_v: params.b2_delta.clone().inner().zeros_like(),
            }
        }

        fn next_bias_correction(&mut self, cfg: AdamWConfig) -> AdamWBiasCorrection {
            next_adamw_bias_correction(&mut self.step, cfg)
        }
    }

    fn next_adamw_bias_correction(step: &mut usize, cfg: AdamWConfig) -> AdamWBiasCorrection {
        *step = step.saturating_add(1);
        let step_i32 = (*step).min(i32::MAX as usize) as i32;
        AdamWBiasCorrection {
            beta1: 1.0 - cfg.beta1.powi(step_i32),
            beta2: 1.0 - cfg.beta2.powi(step_i32),
        }
    }

    impl BurnBaseParams {
        fn from_model(model: &NpaModel, device: &BurnDevice) -> AutomataResult<Self> {
            let config = &model.config;
            Ok(Self {
                w1: tracked_tensor(
                    model.weights.w1.clone(),
                    [config.hidden_dims, config.perception_dims()],
                    device,
                ),
                b1: tracked_tensor(model.weights.b1.clone(), [1, config.hidden_dims], device),
                w2: tracked_tensor(
                    model.weights.w2.clone(),
                    [config.update_dims(), config.hidden_dims],
                    device,
                ),
                b2: tracked_tensor(model.weights.b2.clone(), [1, config.update_dims()], device),
            })
        }

        fn forward_adapter(
            &self,
            features: Tensor2,
            adapter: &BurnAdapterParams,
            _config: DirectBasisTrainConfig,
        ) -> Tensor2 {
            let rows = features.shape().dims::<2>()[0];
            let scale = adapter.alpha / adapter.rank.max(1) as f32;
            let w1 = self.w1.clone()
                + adapter
                    .w1_up
                    .clone()
                    .matmul(adapter.w1_down.clone())
                    .mul_scalar(scale);
            let w2 = self.w2.clone()
                + adapter
                    .w2_up
                    .clone()
                    .matmul(adapter.w2_down.clone())
                    .mul_scalar(scale);
            let b1 = self.b1.clone() + adapter.b1_delta.clone();
            let b2 = self.b2.clone() + adapter.b2_delta.clone();
            let hidden_dims = b1.shape().dims::<2>()[1];
            let output_dims = b2.shape().dims::<2>()[1];
            relu(features.matmul(w1.transpose()) + b1.expand([rows, hidden_dims]))
                .matmul(w2.transpose())
                + b2.expand([rows, output_dims])
        }

        fn forward_adapter_batch(&self, features: Tensor3, adapter: &BurnAdapterBatch) -> Tensor3 {
            let dims = features.shape().dims::<3>();
            let batches = dims[0];
            let rows = dims[1];
            let scale = adapter.alpha / adapter.rank.max(1) as f32;
            let w1 = self.w1.clone().unsqueeze_dim::<3>(0).expand([
                batches,
                self.w1.shape().dims::<2>()[0],
                self.w1.shape().dims::<2>()[1],
            ]) + adapter
                .w1_up
                .clone()
                .matmul(adapter.w1_down.clone())
                .mul_scalar(scale);
            let w2 = self.w2.clone().unsqueeze_dim::<3>(0).expand([
                batches,
                self.w2.shape().dims::<2>()[0],
                self.w2.shape().dims::<2>()[1],
            ]) + adapter
                .w2_up
                .clone()
                .matmul(adapter.w2_down.clone())
                .mul_scalar(scale);
            let hidden_dims = self.b1.shape().dims::<2>()[1];
            let output_dims = self.b2.shape().dims::<2>()[1];
            let b1 = self
                .b1
                .clone()
                .unsqueeze_dim::<3>(0)
                .expand([batches, rows, hidden_dims])
                + adapter
                    .b1_delta
                    .clone()
                    .expand([batches, rows, hidden_dims]);
            let b2 = self
                .b2
                .clone()
                .unsqueeze_dim::<3>(0)
                .expand([batches, rows, output_dims])
                + adapter
                    .b2_delta
                    .clone()
                    .expand([batches, rows, output_dims]);
            relu(features.matmul(w1.swap_dims(1, 2)) + b1).matmul(w2.swap_dims(1, 2)) + b2
        }

        fn apply_adamw(
            &mut self,
            grads: &mut <BurnBackend as burn::tensor::backend::AutodiffBackend>::Gradients,
            state: &mut BurnBaseAdamWState,
            cfg: AdamWConfig,
            normalize: bool,
            collect_metrics: bool,
        ) -> AutomataResult<(f32, f32)> {
            let mut tensors = vec![
                self.w1
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.w1.clone().inner().zeros_like()),
                self.b1
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.b1.clone().inner().zeros_like()),
                self.w2
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.w2.clone().inner().zeros_like()),
                self.b2
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.b2.clone().inner().zeros_like()),
            ];
            let (norm, scale, scale_tensor) =
                prepare_grad_group(&mut tensors, cfg.grad_clip_norm, normalize, collect_metrics)?;
            let bias = state.next_bias_correction(cfg);
            self.w1 = track(apply_adamw_tensor(
                self.w1.clone().inner(),
                tensors.remove(0),
                &mut state.w1_m,
                &mut state.w1_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.b1 = track(apply_adamw_tensor(
                self.b1.clone().inner(),
                tensors.remove(0),
                &mut state.b1_m,
                &mut state.b1_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.w2 = track(apply_adamw_tensor(
                self.w2.clone().inner(),
                tensors.remove(0),
                &mut state.w2_m,
                &mut state.w2_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.b2 = track(apply_adamw_tensor(
                self.b2.clone().inner(),
                tensors.remove(0),
                &mut state.b2_m,
                &mut state.b2_v,
                cfg,
                scale_tensor,
                bias,
            ));
            Ok((norm, scale))
        }

        fn write_to_model(&self, model: &mut NpaModel) -> AutomataResult<()> {
            model.weights = NpaWeights {
                w1: tensor_vec(self.w1.clone().inner())?,
                b1: tensor_vec(self.b1.clone().inner())?,
                w2: tensor_vec(self.w2.clone().inner())?,
                b2: tensor_vec(self.b2.clone().inner())?,
            };
            model.validate()
        }
    }

    impl BurnAdapterParams {
        fn from_adapter(
            adapter: &NpaLowRankAdapter,
            model: &NpaModel,
            device: &BurnDevice,
        ) -> AutomataResult<Self> {
            let config = &model.config;
            Ok(Self {
                rank: adapter.rank,
                alpha: adapter.alpha,
                w1_down: tracked_tensor(
                    adapter.w1_down.clone(),
                    [adapter.rank, config.perception_dims()],
                    device,
                ),
                w1_up: tracked_tensor(
                    adapter.w1_up.clone(),
                    [config.hidden_dims, adapter.rank],
                    device,
                ),
                w2_down: tracked_tensor(
                    adapter.w2_down.clone(),
                    [adapter.rank, config.hidden_dims],
                    device,
                ),
                w2_up: tracked_tensor(
                    adapter.w2_up.clone(),
                    [config.update_dims(), adapter.rank],
                    device,
                ),
                b1_delta: tracked_tensor(adapter.b1_delta.clone(), [1, config.hidden_dims], device),
                b2_delta: tracked_tensor(
                    adapter.b2_delta.clone(),
                    [1, config.update_dims()],
                    device,
                ),
            })
        }

        fn to_adapter(&self) -> AutomataResult<NpaLowRankAdapter> {
            Ok(NpaLowRankAdapter {
                rank: self.rank,
                alpha: self.alpha,
                w1_down: tensor_vec(self.w1_down.clone().inner())?,
                w1_up: tensor_vec(self.w1_up.clone().inner())?,
                w2_down: tensor_vec(self.w2_down.clone().inner())?,
                w2_up: tensor_vec(self.w2_up.clone().inner())?,
                b1_delta: tensor_vec(self.b1_delta.clone().inner())?,
                b2_delta: tensor_vec(self.b2_delta.clone().inner())?,
                b1_delta_correction: Vec::new(),
                b2_delta_correction: Vec::new(),
            })
        }

        fn l2_loss(&self) -> Tensor1 {
            let terms = vec![
                self.w1_down.clone(),
                self.w1_up.clone(),
                self.w2_down.clone(),
                self.w2_up.clone(),
                self.b1_delta.clone(),
                self.b2_delta.clone(),
            ];
            let mut total = None::<Tensor1>;
            for tensor in terms {
                let value = tensor.clone().mul(tensor).mean();
                total = Some(match total {
                    Some(total) => total + value,
                    None => value,
                });
            }
            total.expect("adapter has parameters").div_scalar(6.0)
        }

        fn apply_adamw(
            &mut self,
            grads: &mut <BurnBackend as burn::tensor::backend::AutodiffBackend>::Gradients,
            state: &mut BurnAdapterAdamWState,
            cfg: AdamWConfig,
            normalize: bool,
            gradient_scale: f32,
            collect_metrics: bool,
        ) -> AutomataResult<(f32, f32)> {
            let mut tensors = vec![
                self.w1_down
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.w1_down.clone().inner().zeros_like()),
                self.w1_up
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.w1_up.clone().inner().zeros_like()),
                self.w2_down
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.w2_down.clone().inner().zeros_like()),
                self.w2_up
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.w2_up.clone().inner().zeros_like()),
                self.b1_delta
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.b1_delta.clone().inner().zeros_like()),
                self.b2_delta
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.b2_delta.clone().inner().zeros_like()),
            ];
            if gradient_scale != 1.0 {
                for tensor in &mut tensors {
                    *tensor = tensor.clone().mul_scalar(gradient_scale);
                }
            }
            let (norm, scale, scale_tensor) =
                prepare_grad_group(&mut tensors, cfg.grad_clip_norm, normalize, collect_metrics)?;
            let bias = state.next_bias_correction(cfg);
            self.w1_down = track(apply_adamw_tensor(
                self.w1_down.clone().inner(),
                tensors.remove(0),
                &mut state.w1_down_m,
                &mut state.w1_down_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.w1_up = track(apply_adamw_tensor(
                self.w1_up.clone().inner(),
                tensors.remove(0),
                &mut state.w1_up_m,
                &mut state.w1_up_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.w2_down = track(apply_adamw_tensor(
                self.w2_down.clone().inner(),
                tensors.remove(0),
                &mut state.w2_down_m,
                &mut state.w2_down_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.w2_up = track(apply_adamw_tensor(
                self.w2_up.clone().inner(),
                tensors.remove(0),
                &mut state.w2_up_m,
                &mut state.w2_up_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.b1_delta = track(apply_adamw_tensor(
                self.b1_delta.clone().inner(),
                tensors.remove(0),
                &mut state.b1_delta_m,
                &mut state.b1_delta_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.b2_delta = track(apply_adamw_tensor(
                self.b2_delta.clone().inner(),
                tensors.remove(0),
                &mut state.b2_delta_m,
                &mut state.b2_delta_v,
                cfg,
                scale_tensor,
                bias,
            ));
            Ok((norm, scale))
        }
    }

    impl BurnAdapterBatch {
        fn from_indices(adapters: &[BurnAdapterParams], indices: &[usize]) -> Self {
            let first = &adapters[indices[0]];
            Self {
                rank: first.rank,
                alpha: first.alpha,
                w1_down: stack_adapter_tensor(adapters, indices, |adapter| &adapter.w1_down),
                w1_up: stack_adapter_tensor(adapters, indices, |adapter| &adapter.w1_up),
                w2_down: stack_adapter_tensor(adapters, indices, |adapter| &adapter.w2_down),
                w2_up: stack_adapter_tensor(adapters, indices, |adapter| &adapter.w2_up),
                b1_delta: stack_adapter_tensor(adapters, indices, |adapter| &adapter.b1_delta),
                b2_delta: stack_adapter_tensor(adapters, indices, |adapter| &adapter.b2_delta),
            }
        }

        fn l2_loss(&self) -> Tensor1 {
            let terms = vec![
                self.w1_down.clone(),
                self.w1_up.clone(),
                self.w2_down.clone(),
                self.w2_up.clone(),
                self.b1_delta.clone(),
                self.b2_delta.clone(),
            ];
            let mut total = None::<Tensor1>;
            for tensor in terms {
                let value = tensor.clone().mul(tensor).mean();
                total = Some(match total {
                    Some(total) => total + value,
                    None => value,
                });
            }
            total.expect("adapter batch has parameters").div_scalar(6.0)
        }

        fn l2_loss_vector(&self) -> Tensor1 {
            let terms = vec![
                self.w1_down.clone(),
                self.w1_up.clone(),
                self.w2_down.clone(),
                self.w2_up.clone(),
                self.b1_delta.clone(),
                self.b2_delta.clone(),
            ];
            let mut total = None::<Tensor1>;
            for tensor in terms {
                let dims = tensor.shape().dims::<3>();
                let value = tensor
                    .clone()
                    .mul(tensor)
                    .reshape([dims[0], dims[1] * dims[2]])
                    .mean_dim(1)
                    .squeeze_dim::<1>(1);
                total = Some(match total {
                    Some(total) => total + value,
                    None => value,
                });
            }
            total.expect("adapter batch has parameters").div_scalar(6.0)
        }
    }

    fn stack_adapter_tensor(
        adapters: &[BurnAdapterParams],
        indices: &[usize],
        select: impl Fn(&BurnAdapterParams) -> &Tensor2,
    ) -> Tensor3 {
        Tensor::cat(
            indices
                .iter()
                .map(|idx| select(&adapters[*idx]).clone().unsqueeze_dim::<3>(0))
                .collect::<Vec<_>>(),
            0,
        )
    }

    fn burn_targets(
        examples: &[DirectBasisExample],
        config: DirectBasisTrainConfig,
        device: &BurnDevice,
    ) -> AutomataResult<Vec<BurnTargetExample>> {
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let pixel_xy = tensor(
            pixel_xy_values(config.loss_config.image_size),
            [pixels, 2],
            device,
        );
        examples
            .iter()
            .map(|example| {
                let render = render_target_2d_splat(&example.target, config.loss_config)?;
                let target_mean = example.target.mean_position();
                Ok(BurnTargetExample {
                    target_rgb: tensor(render.rgb, [pixels, 3], device),
                    target_density: tensor(render.density, [pixels, 1], device),
                    target_mean: tensor([target_mean[0], target_mean[1]].to_vec(), [1, 2], device),
                    pixel_xy: pixel_xy.clone(),
                    pixel_size: example.target.pixel_size,
                    target_points: example.target.point_count(),
                    particle_count: example.source.particles.unwrap_or(config.rollout_particles),
                    update_prob: example.source.update_prob.unwrap_or(config.update_prob),
                    seed_scale: example.source.seed_scale.unwrap_or(config.seed_scale),
                })
            })
            .collect()
    }

    fn seed_batch_tensors(
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        config: DirectBasisTrainConfig,
        step_seed: u64,
        device: &BurnDevice,
    ) -> (Tensor3, Tensor3) {
        let mut positions = Vec::with_capacity(indices.len() * particle_count * 2);
        let mut states = Vec::with_capacity(indices.len() * particle_count * 16);
        for &idx in indices {
            let (example_positions, example_states) = seed_particles_scaled(
                1,
                particle_count,
                16,
                2,
                step_seed.wrapping_add(idx as u64),
                config.seed_mode,
                targets[idx].seed_scale,
            );
            positions.extend(
                example_positions
                    .iter()
                    .flat_map(|position| [position[0], position[1]]),
            );
            states.extend(example_states);
        }
        (
            tensor3(positions, [indices.len(), particle_count, 2], device),
            tensor3(states, [indices.len(), particle_count, 16], device),
        )
    }

    fn batch_masks(
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        rng: &mut StdRng,
    ) -> Vec<f32> {
        let mut values = Vec::with_capacity(indices.len() * particle_count);
        for &idx in indices {
            values.extend(stochastic_mask(
                particle_count,
                targets[idx].update_prob,
                rng,
            ));
        }
        values
    }

    fn batch_masks_with_rngs(
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        rngs: &mut [StdRng],
    ) -> Vec<f32> {
        let mut values = Vec::with_capacity(indices.len() * particle_count);
        for (local, &idx) in indices.iter().enumerate() {
            values.extend(stochastic_mask(
                particle_count,
                targets[idx].update_prob,
                &mut rngs[local],
            ));
        }
        values
    }

    fn stack_target_rgb(targets: &[BurnTargetExample], indices: &[usize]) -> Tensor3 {
        Tensor::cat(
            indices
                .iter()
                .map(|idx| targets[*idx].target_rgb.clone().unsqueeze_dim::<3>(0))
                .collect::<Vec<_>>(),
            0,
        )
    }

    fn stack_target_density(targets: &[BurnTargetExample], indices: &[usize]) -> Tensor3 {
        Tensor::cat(
            indices
                .iter()
                .map(|idx| targets[*idx].target_density.clone().unsqueeze_dim::<3>(0))
                .collect::<Vec<_>>(),
            0,
        )
    }

    fn stack_target_mean(targets: &[BurnTargetExample], indices: &[usize]) -> Tensor3 {
        Tensor::cat(
            indices
                .iter()
                .map(|idx| targets[*idx].target_mean.clone().unsqueeze_dim::<3>(0))
                .collect::<Vec<_>>(),
            0,
        )
    }

    fn stack_pixel_sizes(targets: &[BurnTargetExample], indices: &[usize]) -> Tensor3 {
        let values = indices
            .iter()
            .map(|idx| targets[*idx].pixel_size)
            .collect::<Vec<_>>();
        tensor3(
            values,
            [indices.len(), 1, 1],
            &targets[indices[0]].target_rgb.device(),
        )
    }

    fn stack_target_point_counts(targets: &[BurnTargetExample], indices: &[usize]) -> Tensor3 {
        let values = indices
            .iter()
            .map(|idx| targets[*idx].target_points as f32)
            .collect::<Vec<_>>();
        tensor3(
            values,
            [indices.len(), 1, 1],
            &targets[indices[0]].target_rgb.device(),
        )
    }

    fn pixel_xy_values(image_size: usize) -> Vec<f32> {
        let mut values = Vec::with_capacity(image_size * image_size * 2);
        for y in 0..image_size {
            for x in 0..image_size {
                values.push(x as f32);
                values.push(y as f32);
            }
        }
        values
    }

    fn adapter_cache_metrics(
        base: &NpaModel,
        params: &BurnBaseParams,
        train_adapters: &[BurnAdapterParams],
        holdout_adapters: &[BurnAdapterParams],
        train_targets: &[BurnTargetExample],
        holdout_targets: &[BurnTargetExample],
    ) -> AutomataResult<serde_json::Value> {
        let rank = train_adapters
            .first()
            .or_else(|| holdout_adapters.first())
            .map_or(0, |adapter| adapter.rank);
        let parameters_per_adapter = if rank == 0 {
            0
        } else {
            NpaLowRankAdapter::parameter_count_for_config(&base.config, rank)
        };
        let total_adapters = train_adapters.len() + holdout_adapters.len();
        let total_adapter_parameters = parameters_per_adapter * total_adapters;
        let train_target_points = train_targets
            .iter()
            .map(|target| target.target_points)
            .sum::<usize>();
        let holdout_target_points = holdout_targets
            .iter()
            .map(|target| target.target_points)
            .sum::<usize>();
        let train_render_pixels = train_targets
            .iter()
            .map(|target| target.target_density.shape().dims::<2>()[0])
            .sum::<usize>();
        let holdout_render_pixels = holdout_targets
            .iter()
            .map(|target| target.target_density.shape().dims::<2>()[0])
            .sum::<usize>();
        Ok(json!({
            "representation": "resident_gpu_tensor_set_per_sample",
            "readback_policy": "report_interval_scalars_and_end_of_phase_artifacts_only",
            "non_report_step_loss_readbacks": false,
            "adapter_tensors_per_sample": 6,
            "rank": rank,
            "parameters_per_adapter": parameters_per_adapter,
            "train_adapters": train_adapters.len(),
            "holdout_adapters": holdout_adapters.len(),
            "total_adapters": total_adapters,
            "total_adapter_parameters": total_adapter_parameters,
            "estimated_adapter_weight_bytes_f32": total_adapter_parameters * std::mem::size_of::<f32>(),
            "estimated_adapter_tensor_count": total_adapters * 6,
            "train_target_points": train_target_points,
            "holdout_target_points": holdout_target_points,
            "train_render_pixels": train_render_pixels,
            "holdout_render_pixels": holdout_render_pixels,
            "estimated_target_render_cache_bytes_f32": (train_render_pixels + holdout_render_pixels) * 4 * std::mem::size_of::<f32>(),
            "base_norms": base_norm_metrics(params)?,
            "train_adapter_norms": adapter_norm_metrics(train_adapters)?,
            "holdout_adapter_norms": adapter_norm_metrics(holdout_adapters)?,
        }))
    }

    fn base_norm_metrics(params: &BurnBaseParams) -> AutomataResult<serde_json::Value> {
        let w1 = tensor_l2_norm(&params.w1.clone().inner())?;
        let b1 = tensor_l2_norm(&params.b1.clone().inner())?;
        let w2 = tensor_l2_norm(&params.w2.clone().inner())?;
        let b2 = tensor_l2_norm(&params.b2.clone().inner())?;
        Ok(json!({
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
            "total": finite_scalar("Burn direct base norm", (w1 * w1 + b1 * b1 + w2 * w2 + b2 * b2).sqrt())?,
        }))
    }

    fn adapter_norm_metrics(adapters: &[BurnAdapterParams]) -> AutomataResult<serde_json::Value> {
        if adapters.is_empty() {
            return Ok(json!({
                "examples": 0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
            }));
        }
        let mut sum = 0.0_f32;
        let mut min = f32::INFINITY;
        let mut max = 0.0_f32;
        for adapter in adapters {
            let norm = adapter_l2_norm(adapter)?;
            sum += norm;
            min = min.min(norm);
            max = max.max(norm);
        }
        Ok(json!({
            "examples": adapters.len(),
            "mean": finite_scalar("Burn direct mean adapter norm", sum / adapters.len() as f32)?,
            "min": finite_scalar("Burn direct min adapter norm", min)?,
            "max": finite_scalar("Burn direct max adapter norm", max)?,
        }))
    }

    fn adapter_l2_norm(adapter: &BurnAdapterParams) -> AutomataResult<f32> {
        let tensors = [
            adapter.w1_down.clone().inner(),
            adapter.w1_up.clone().inner(),
            adapter.w2_down.clone().inner(),
            adapter.w2_up.clone().inner(),
            adapter.b1_delta.clone().inner(),
            adapter.b2_delta.clone().inner(),
        ];
        finite_scalar(
            "Burn direct adapter norm",
            group_norm_tensor(&tensors).into_scalar(),
        )
    }

    fn mean_updates_per_sample(steps: usize, batch_size: usize, examples: usize) -> f32 {
        if examples == 0 {
            return 0.0;
        }
        steps as f32 * batch_size.min(examples).max(1) as f32 / examples as f32
    }

    fn seed_tensors(
        particle_count: usize,
        config: DirectBasisTrainConfig,
        seed_scale: f32,
        seed: u64,
        device: &BurnDevice,
    ) -> (Tensor2, Tensor2) {
        let (positions, states) =
            seed_particles_scaled(1, particle_count, 16, 2, seed, config.seed_mode, seed_scale);
        let flat_positions = positions
            .iter()
            .flat_map(|position| [position[0], position[1]])
            .collect::<Vec<_>>();
        (
            tensor(flat_positions, [particle_count, 2], device),
            tensor(states, [particle_count, 16], device),
        )
    }

    fn write_adapters(
        examples: &mut [DirectBasisExample],
        adapters: &[BurnAdapterParams],
    ) -> AutomataResult<()> {
        for (example, adapter) in examples.iter_mut().zip(adapters) {
            example.adapter = adapter.to_adapter()?;
        }
        Ok(())
    }

    fn sample_indices(len: usize, requested: usize, rng: &mut StdRng) -> Vec<usize> {
        let count = if requested == 0 {
            len
        } else {
            requested.min(len)
        };
        if count.saturating_mul(4) < len {
            let mut indices = std::collections::BTreeSet::new();
            while indices.len() < count {
                indices.insert(rng.random_range(0..len));
            }
            return indices.into_iter().collect();
        }
        let mut indices = (0..len).collect::<Vec<_>>();
        indices.shuffle(rng);
        indices.truncate(count);
        indices
    }

    fn loss_scalars(loss: &BurnLossTensors) -> AutomataResult<BurnLossScalars> {
        Ok(BurnLossScalars {
            total: finite_scalar(
                "Burn direct total loss",
                loss.total.clone().inner().into_scalar(),
            )?,
            splat: finite_scalar(
                "Burn direct splat loss",
                loss.splat.clone().inner().into_scalar(),
            )?,
            color: finite_scalar(
                "Burn direct color loss",
                loss.color.clone().inner().into_scalar(),
            )?,
            density: finite_scalar(
                "Burn direct density loss",
                loss.density.clone().inner().into_scalar(),
            )?,
        })
    }

    fn loss_vector_scalars(loss: BurnLossBatchTensors) -> AutomataResult<Vec<BurnLossScalars>> {
        let total = tensor1_vec(loss.total.inner())?;
        let splat = tensor1_vec(loss.splat.inner())?;
        let color = tensor1_vec(loss.color.inner())?;
        let density = tensor1_vec(loss.density.inner())?;
        if total.len() != splat.len() || total.len() != color.len() || total.len() != density.len()
        {
            return Err(AutomataError::InvalidArgument(
                "Burn direct vector loss readback length mismatch".to_string(),
            ));
        }
        total
            .into_iter()
            .zip(splat)
            .zip(color)
            .zip(density)
            .enumerate()
            .map(|(idx, (((total, splat), color), density))| {
                Ok(BurnLossScalars {
                    total: finite_scalar(&format!("Burn direct total loss[{idx}]"), total)?,
                    splat: finite_scalar(&format!("Burn direct splat loss[{idx}]"), splat)?,
                    color: finite_scalar(&format!("Burn direct color loss[{idx}]"), color)?,
                    density: finite_scalar(&format!("Burn direct density loss[{idx}]"), density)?,
                })
            })
            .collect()
    }

    fn prepare_grad_group(
        tensors: &mut [Tensor2Inner],
        clip_norm: f32,
        normalize: bool,
        collect_metrics: bool,
    ) -> AutomataResult<(f32, f32, Tensor1Inner)> {
        let original_norm_tensor = group_norm_tensor(tensors);
        let original_norm = if collect_metrics {
            finite_scalar(
                "Burn direct grad norm",
                original_norm_tensor.clone().into_scalar(),
            )?
        } else {
            0.0
        };
        if normalize {
            for tensor in tensors.iter_mut() {
                let dims = tensor.shape().dims::<2>();
                let norm = tensor_l2_norm_tensor(tensor).add_scalar(1.0e-8);
                *tensor = tensor.clone().div(norm.expand(dims));
            }
        }
        let clip_norm_source = if normalize {
            group_norm_tensor(tensors)
        } else {
            original_norm_tensor
        };
        let scale_tensor = if clip_norm > 0.0 {
            clip_norm_source
                .clone()
                .clamp_min(clip_norm)
                .recip()
                .mul_scalar(clip_norm)
        } else {
            clip_norm_source.zeros_like().add_scalar(1.0)
        };
        let scale = if collect_metrics {
            finite_scalar("Burn direct grad scale", scale_tensor.clone().into_scalar())?
        } else {
            1.0
        };
        Ok((original_norm, scale, scale_tensor))
    }

    fn group_norm_tensor(tensors: &[Tensor2Inner]) -> Tensor1Inner {
        let mut total = None::<Tensor1Inner>;
        for tensor in tensors {
            let value = tensor.clone().mul(tensor.clone()).sum();
            total = Some(match total {
                Some(total) => total + value,
                None => value,
            });
        }
        total.expect("gradient group has tensors").sqrt()
    }

    fn tensor_l2_norm_tensor(tensor: &Tensor2Inner) -> Tensor1Inner {
        tensor.clone().mul(tensor.clone()).sum().sqrt()
    }

    fn tensor_l2_norm(tensor: &Tensor2Inner) -> AutomataResult<f32> {
        finite_scalar(
            "Burn direct tensor norm",
            tensor_l2_norm_tensor(tensor).into_scalar(),
        )
    }

    fn adamw_from_sgd(cfg: SgdConfig) -> AdamWConfig {
        AdamWConfig {
            learning_rate: cfg.learning_rate,
            weight_decay: cfg.weight_decay,
            grad_clip_norm: cfg.grad_clip_norm,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1.0e-8,
        }
    }

    fn apply_adamw_tensor(
        param: Tensor2Inner,
        grad: Tensor2Inner,
        moment: &mut Tensor2Inner,
        velocity: &mut Tensor2Inner,
        cfg: AdamWConfig,
        scale: Tensor1Inner,
        bias: AdamWBiasCorrection,
    ) -> Tensor2Inner {
        let dims = param.shape().dims::<2>();
        let grad = grad.mul(scale.expand(dims));
        let decayed = if cfg.weight_decay > 0.0 {
            param
                .clone()
                .mul_scalar(1.0 - cfg.learning_rate * cfg.weight_decay)
        } else {
            param.clone()
        };
        *moment = moment.clone().mul_scalar(cfg.beta1) + grad.clone().mul_scalar(1.0 - cfg.beta1);
        *velocity = velocity.clone().mul_scalar(cfg.beta2)
            + grad.clone().mul(grad).mul_scalar(1.0 - cfg.beta2);
        let normalized_step = moment
            .clone()
            .div_scalar(bias.beta1.max(f32::MIN_POSITIVE))
            .div(
                velocity
                    .clone()
                    .div_scalar(bias.beta2.max(f32::MIN_POSITIVE))
                    .sqrt()
                    .add_scalar(cfg.epsilon),
            );
        decayed - normalized_step.mul_scalar(cfg.learning_rate)
    }

    fn tracked_tensor(values: Vec<f32>, shape: [usize; 2], device: &BurnDevice) -> Tensor2 {
        tensor(values, shape, device).require_grad()
    }

    fn tensor(values: Vec<f32>, shape: [usize; 2], device: &BurnDevice) -> Tensor2 {
        Tensor::<BurnBackend, 2>::from_data(TensorData::new(values, shape), device)
    }

    fn tensor3(values: Vec<f32>, shape: [usize; 3], device: &BurnDevice) -> Tensor3 {
        Tensor::<BurnBackend, 3>::from_data(TensorData::new(values, shape), device)
    }

    fn detach1(tensor: Tensor1) -> Tensor1 {
        Tensor::<BurnBackend, 1>::from_inner(tensor.inner())
    }

    fn detach2(tensor: Tensor2) -> Tensor2 {
        Tensor::<BurnBackend, 2>::from_inner(tensor.inner())
    }

    fn detach3(tensor: Tensor3) -> Tensor3 {
        Tensor::<BurnBackend, 3>::from_inner(tensor.inner())
    }

    fn track(tensor: Tensor2Inner) -> Tensor2 {
        Tensor::<BurnBackend, 2>::from_inner(tensor).require_grad()
    }

    fn tensor_vec(tensor: Tensor2Inner) -> AutomataResult<Vec<f32>> {
        tensor.into_data().to_vec::<f32>().map_err(|err| {
            AutomataError::InvalidArgument(format!("Burn/WGPU tensor readback failed: {err}"))
        })
    }

    fn tensor1_vec(tensor: Tensor1Inner) -> AutomataResult<Vec<f32>> {
        tensor.into_data().to_vec::<f32>().map_err(|err| {
            AutomataError::InvalidArgument(format!("Burn/WGPU tensor readback failed: {err}"))
        })
    }

    fn finite_scalar(name: &str, value: f32) -> AutomataResult<f32> {
        if value.is_finite() {
            Ok(value)
        } else {
            Err(AutomataError::InvalidArgument(format!(
                "{name} is not finite"
            )))
        }
    }

    fn check_process_memory_budget(
        label: &str,
        config: DirectBasisTrainConfig,
    ) -> Result<ProcessMemorySnapshot, Box<dyn std::error::Error>> {
        let budget_bytes = config
            .system_memory_budget_gb
            .map(memory_budget_gb_to_bytes);
        let snapshot = ProcessMemorySnapshot {
            label: label.to_string(),
            rss_bytes: current_process_rss_bytes(),
            budget_bytes,
        };
        if let (Some(rss_bytes), Some(budget_bytes)) = (snapshot.rss_bytes, snapshot.budget_bytes)
            && rss_bytes > budget_bytes
        {
            return Err(std::io::Error::other(format!(
                "Burn/WGPU direct-basis memory budget exceeded at {label}: rss={:.2} GiB budget={:.2} GiB",
                bytes_to_gib(rss_bytes),
                bytes_to_gib(budget_bytes)
            ))
            .into());
        }
        Ok(snapshot)
    }

    fn current_process_rss_bytes() -> Option<u64> {
        let status = fs::read_to_string("/proc/self/status").ok()?;
        status.lines().find_map(|line| {
            let rest = line.strip_prefix("VmRSS:")?;
            let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            Some(kb.saturating_mul(1024))
        })
    }

    fn memory_budget_gb_to_bytes(gb: f32) -> u64 {
        (gb as f64 * 1024.0 * 1024.0 * 1024.0).round() as u64
    }

    fn bytes_to_gib(bytes: u64) -> f64 {
        bytes as f64 / 1024.0 / 1024.0 / 1024.0
    }
}

#[cfg(feature = "backend_wgpu")]
pub(super) use imp::*;

#[cfg(not(feature = "backend_wgpu"))]
pub(super) struct BurnWgpuDirectBasisOutput {
    pub(super) backend: &'static str,
    pub(super) device: String,
    pub(super) metrics: serde_json::Value,
    pub(super) history: Vec<super::CliHyper2dDirectBasisHistoryEntry>,
    pub(super) train_refine_history: Vec<super::CliHyper2dDirectBasisHistoryEntry>,
    pub(super) holdout_history: Vec<super::CliHyper2dDirectBasisHistoryEntry>,
    pub(super) best_train_loss: Option<f32>,
    pub(super) best_train_step: usize,
}

#[cfg(not(feature = "backend_wgpu"))]
pub(super) fn train_direct_basis_burn_wgpu(
    _base: &mut super::NpaModel,
    _train_examples: &mut [super::DirectBasisExample],
    _holdout_examples: &mut [super::DirectBasisExample],
    _train_config: super::DirectBasisTrainConfig,
    _train_refine_config: super::DirectBasisTrainConfig,
    _holdout_config: super::DirectBasisTrainConfig,
) -> Result<BurnWgpuDirectBasisOutput, Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "Burn/WGPU direct-basis training requires the backend_wgpu feature; rebuild with --features cli,backend_wgpu or use the legacy --gpu-backend legacy-upstream-python parity path",
    )
    .into())
}
