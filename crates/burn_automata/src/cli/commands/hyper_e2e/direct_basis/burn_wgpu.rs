#[cfg(feature = "backend_wgpu")]
mod imp {
    use std::time::Instant;

    use burn::{
        backend::{Autodiff, Wgpu},
        tensor::{
            Device, Tensor, TensorData,
            activation::{relu, sigmoid},
        },
    };
    use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
    use serde_json::json;

    use super::super::{DirectBasisExample, DirectBasisTrainConfig};
    use crate::cli::reports::{
        CliHyper2dDirectBasisHistoryEntry, CliHyper2dDirectBasisLossSummary,
    };
    use crate::{
        AutomataError, AutomataResult, NpaLowRankAdapter, NpaModel, NpaWeights, SgdConfig,
        rollout::seed_particles_scaled,
    };

    type InnerBackend = Wgpu<f32>;
    type BurnBackend = Autodiff<InnerBackend>;
    type BurnDevice = Device<BurnBackend>;
    type Tensor1 = Tensor<BurnBackend, 1>;
    type Tensor2 = Tensor<BurnBackend, 2>;
    type Tensor2Inner = Tensor<InnerBackend, 2>;

    const BACKEND: &str = "burn_wgpu_autodiff_dense_direct_basis";
    const EPSILON: f32 = 1.0e-6;

    pub(crate) struct BurnWgpuDirectBasisOutput {
        pub(crate) backend: &'static str,
        pub(crate) device: String,
        pub(crate) metrics: serde_json::Value,
        pub(crate) history: Vec<CliHyper2dDirectBasisHistoryEntry>,
        pub(crate) holdout_history: Vec<CliHyper2dDirectBasisHistoryEntry>,
        pub(crate) best_train_loss: Option<f32>,
        pub(crate) best_train_step: usize,
    }

    struct BurnBaseParams {
        w1: Tensor2,
        b1: Tensor2,
        w2: Tensor2,
        b2: Tensor2,
    }

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

    struct BurnTargetExample {
        positions: Tensor2,
        colors: Tensor2,
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

    #[derive(Clone, Copy, Default)]
    struct BurnLossScalars {
        total: f32,
        splat: f32,
        color: f32,
        density: f32,
    }

    pub(crate) fn train_direct_basis_burn_wgpu(
        base: &mut NpaModel,
        train_examples: &mut [DirectBasisExample],
        holdout_examples: &mut [DirectBasisExample],
        train_config: DirectBasisTrainConfig,
        holdout_config: DirectBasisTrainConfig,
    ) -> Result<BurnWgpuDirectBasisOutput, Box<dyn std::error::Error>> {
        if base.config.spatial_dims != 2 {
            return Err(std::io::Error::other(
                "Burn/WGPU direct-basis training currently supports 2D",
            )
            .into());
        }
        let device = BurnDevice::default();
        let mut params = BurnBaseParams::from_model(base, &device)?;
        let mut train_adapters = train_examples
            .iter()
            .map(|example| BurnAdapterParams::from_adapter(&example.adapter, base, &device))
            .collect::<AutomataResult<Vec<_>>>()?;
        let train_targets = burn_targets(train_examples, train_config, &device)?;
        let train_phase = run_phase(
            &mut params,
            &mut train_adapters,
            &train_targets,
            train_config,
            true,
            "train",
        )?;
        params.write_to_model(base)?;
        write_adapters(train_examples, &train_adapters)?;

        let mut holdout_adapters = holdout_examples
            .iter()
            .map(|example| BurnAdapterParams::from_adapter(&example.adapter, base, &device))
            .collect::<AutomataResult<Vec<_>>>()?;
        let holdout_targets = burn_targets(holdout_examples, holdout_config, &device)?;
        let holdout_phase = run_phase(
            &mut params,
            &mut holdout_adapters,
            &holdout_targets,
            holdout_config,
            false,
            "holdout",
        )?;
        write_adapters(holdout_examples, &holdout_adapters)?;

        let metrics = json!({
            "backend": BACKEND,
            "device": "wgpu-default",
            "objective": "dense_burn_target_point_splat_loss",
            "perception": "dense_all_pairs_blur_density_grad_zero_state_grad",
            "adapter_cache": adapter_cache_metrics(
                base,
                &train_adapters,
                &holdout_adapters,
                &train_targets,
                &holdout_targets,
            ),
            "train_examples": train_examples.len(),
            "holdout_examples": holdout_examples.len(),
            "train_steps": train_config.steps,
            "holdout_adapter_steps": holdout_config.steps,
            "train_final_dense_loss": train_phase.history.last().map(|entry| entry.loss),
            "holdout_final_dense_loss": holdout_phase.history.last().map(|entry| entry.loss),
        });
        Ok(BurnWgpuDirectBasisOutput {
            backend: BACKEND,
            device: "wgpu-default".to_string(),
            metrics,
            history: train_phase.history,
            holdout_history: holdout_phase.history,
            best_train_loss: train_phase.best_loss,
            best_train_step: train_phase.best_step,
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
        for step in 1usize..=config.steps {
            let started = Instant::now();
            let indices = sample_indices(targets.len(), config.example_batch_size, &mut rng);
            let mut total = None::<Tensor1>;
            let mut loss_sum = 0.0_f32;
            let mut particle_steps = 0.0_f64;
            for &idx in &indices {
                let loss = example_loss(
                    params,
                    &adapters[idx],
                    &targets[idx],
                    config,
                    config
                        .seed
                        .wrapping_add((step as u64).wrapping_mul(0x9e37_79b9))
                        .wrapping_add(idx as u64),
                );
                let scalars = loss_scalars(&loss)?;
                loss_sum += scalars.total;
                particle_steps += targets[idx].particle_count as f64 * config.rollout_steps as f64;
                let scaled = loss.total.div_scalar(indices.len() as f32);
                total = Some(match total {
                    Some(value) => value + scaled,
                    None => scaled,
                });
            }
            let Some(total_loss) = total else {
                return Err(std::io::Error::other("Burn direct-basis batch was empty").into());
            };
            let mut grads = total_loss.backward();
            let (base_grad_norm, base_grad_scale) = if update_base {
                params.apply_sgd(
                    &mut grads,
                    config.base_sgd,
                    config.per_parameter_grad_normalization,
                )?
            } else {
                (0.0, 1.0)
            };
            let mut adapter_grad_sum = 0.0_f32;
            let mut adapter_grad_max = 0.0_f32;
            for &idx in &indices {
                let (grad_norm, _) = adapters[idx].apply_sgd(
                    &mut grads,
                    config.adapter_sgd,
                    config.per_parameter_grad_normalization,
                )?;
                adapter_grad_sum += grad_norm;
                adapter_grad_max = adapter_grad_max.max(grad_norm);
            }
            let elapsed = started.elapsed();
            let should_report =
                step == config.steps || step.is_multiple_of(config.report_interval.max(1));
            if should_report {
                let eval_loss = evaluate_targets(
                    params,
                    adapters,
                    targets,
                    config,
                    config.eval_examples,
                    config.eval_seed + step as u64,
                )?;
                if let Some(eval_loss) = eval_loss {
                    if best_loss.is_none_or(|best| eval_loss.mean_total_loss < best) {
                        best_loss = Some(eval_loss.mean_total_loss);
                        best_step = step;
                    }
                    history.push(CliHyper2dDirectBasisHistoryEntry {
                        step,
                        loss: loss_sum / indices.len() as f32,
                        eval_loss: Some(eval_loss),
                        base_grad_norm,
                        base_grad_scale,
                        mean_adapter_grad_norm: adapter_grad_sum / indices.len() as f32,
                        max_adapter_grad_norm: adapter_grad_max,
                        examples_seen: indices.len(),
                        particle_steps_per_sec: particle_steps
                            / elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
                        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
                    });
                    println!(
                        "burn-wgpu direct-basis {phase_label} step {step}/{} loss={:.6} eval_mean={:.6} examples={} particle_steps_per_sec={:.0} elapsed_ms={:.1}",
                        config.steps,
                        loss_sum / indices.len() as f32,
                        eval_loss.mean_total_loss,
                        indices.len(),
                        particle_steps / elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
                        elapsed.as_secs_f64() * 1000.0
                    );
                }
            }
        }
        Ok(BurnPhaseReport {
            history,
            best_loss,
            best_step,
        })
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
        for &idx in &indices {
            let loss = example_loss(
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
        let scale = 1.0 / indices.len() as f32;
        summary.mean_total_loss *= scale;
        summary.mean_splat_loss *= scale;
        summary.mean_color_loss *= scale;
        summary.mean_density_loss *= scale;
        Ok(Some(summary))
    }

    fn example_loss(
        params: &BurnBaseParams,
        adapter: &BurnAdapterParams,
        target: &BurnTargetExample,
        config: DirectBasisTrainConfig,
        seed: u64,
    ) -> BurnLossTensors {
        let (mut x, mut s) = seed_tensors(
            target.particle_count,
            config,
            target.seed_scale,
            seed,
            &target.positions.device(),
        );
        for _ in 0..config.rollout_steps {
            let features = dense_perception(&x, &s, target.seed_scale);
            let update = params.forward_adapter(features, adapter, config);
            let dx_raw = update.clone().narrow(1, 0, 2);
            let ds = update.narrow(1, 2, s.shape().dims::<2>()[1]);
            let norm = dx_raw
                .clone()
                .mul(dx_raw.clone())
                .sum_dim(1)
                .sqrt()
                .add_scalar(1.0)
                .expand([target.particle_count, 2]);
            let dx = dx_raw.mul_scalar(0.1 * 0.5 * target.update_prob).div(norm);
            x = x + dx;
            s = s + ds.mul_scalar(target.update_prob);
        }
        target_point_loss(&x, &s, target, config, adapter)
    }

    fn dense_perception(x: &Tensor2, s: &Tensor2, seed_scale: f32) -> Tensor2 {
        let dims = s.shape().dims::<2>();
        let rows = dims[0];
        let state_dims = dims[1];
        let xi = x.clone().unsqueeze_dim::<3>(1).expand([rows, rows, 2]);
        let xj = x.clone().unsqueeze_dim::<3>(0).expand([rows, rows, 2]);
        let diff = xj - xi;
        let dist2 = diff
            .clone()
            .mul(diff.clone())
            .sum_dim(2)
            .squeeze_dim::<2>(2);
        let sigma = seed_scale.max(0.05);
        let weights = dist2.mul_scalar(-0.5 / (sigma * sigma)).exp();
        let density = weights
            .clone()
            .sum_dim(1)
            .clamp_min(EPSILON)
            .expand([rows, state_dims]);
        let blur = weights.clone().matmul(s.clone()).div(density);
        let state_grad = Tensor::<BurnBackend, 2>::zeros([rows, state_dims * 2], &s.device());
        let density_grad = weights
            .clone()
            .unsqueeze_dim::<3>(2)
            .mul(diff)
            .sum_dim(1)
            .squeeze_dim::<2>(1)
            .div(weights.sum_dim(1).clamp_min(EPSILON).expand([rows, 2]))
            .div_scalar(rows.max(1) as f32);
        Tensor::cat(vec![s.clone(), blur, state_grad, density_grad], 1)
    }

    fn target_point_loss(
        x: &Tensor2,
        s: &Tensor2,
        target: &BurnTargetExample,
        config: DirectBasisTrainConfig,
        adapter: &BurnAdapterParams,
    ) -> BurnLossTensors {
        let target_shape = target.positions.shape().dims::<2>();
        let target_points = target_shape[0];
        let particle_count = x.shape().dims::<2>()[0];
        let target_i = target.positions.clone().unsqueeze_dim::<3>(1).expand([
            target_points,
            particle_count,
            2,
        ]);
        let xj = x
            .clone()
            .unsqueeze_dim::<3>(0)
            .expand([target_points, particle_count, 2]);
        let diff = target_i - xj;
        let dist2 = diff.clone().mul(diff).sum_dim(2).squeeze_dim::<2>(2);
        let sigma = (config.loss_config.sigma * 0.01).max(0.01);
        let weights = dist2.mul_scalar(-0.5 / (sigma * sigma)).exp();
        let density = weights
            .clone()
            .sum_dim(1)
            .div_scalar(particle_count.max(1) as f32)
            .clamp_min(EPSILON);
        let colors = sigmoid(s.clone().narrow(1, s.shape().dims::<2>()[1] - 3, 3));
        let predicted_colors = weights.matmul(colors).div(
            density
                .clone()
                .expand([target_points, 3])
                .mul_scalar(particle_count.max(1) as f32),
        );
        let color_diff = predicted_colors - target.colors.clone();
        let color_loss = color_diff.clone().mul(color_diff).mean();
        let density_diff = density
            - Tensor::<BurnBackend, 2>::ones([target_points, 1], &x.device())
                .div_scalar(target_points.max(1) as f32);
        let density_loss = density_diff.clone().mul(density_diff).mean();
        let splat = color_loss
            .clone()
            .mul_scalar(config.loss_config.color_loss_weight)
            + density_loss
                .clone()
                .mul_scalar(config.loss_config.density_loss_weight);
        let bound = relu(x.clone().abs().add_scalar(-1.0));
        let bound_loss = bound.clone().mul(bound).mean();
        let overflow = relu(s.clone().abs().add_scalar(-4.0));
        let overflow_loss = overflow.clone().mul(overflow).mean();
        let mut total = splat
            .clone()
            .mul_scalar(config.loss_config.splat_loss_weight)
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

        fn apply_sgd(
            &mut self,
            grads: &mut <BurnBackend as burn::tensor::backend::AutodiffBackend>::Gradients,
            cfg: SgdConfig,
            normalize: bool,
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
            let (norm, scale) = prepare_grad_group(&mut tensors, cfg.grad_clip_norm, normalize)?;
            self.w1 = track(apply_sgd_tensor(
                self.w1.clone().inner(),
                tensors.remove(0),
                cfg,
                scale,
            ));
            self.b1 = track(apply_sgd_tensor(
                self.b1.clone().inner(),
                tensors.remove(0),
                cfg,
                scale,
            ));
            self.w2 = track(apply_sgd_tensor(
                self.w2.clone().inner(),
                tensors.remove(0),
                cfg,
                scale,
            ));
            self.b2 = track(apply_sgd_tensor(
                self.b2.clone().inner(),
                tensors.remove(0),
                cfg,
                scale,
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

        fn apply_sgd(
            &mut self,
            grads: &mut <BurnBackend as burn::tensor::backend::AutodiffBackend>::Gradients,
            cfg: SgdConfig,
            normalize: bool,
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
            let (norm, scale) = prepare_grad_group(&mut tensors, cfg.grad_clip_norm, normalize)?;
            self.w1_down = track(apply_sgd_tensor(
                self.w1_down.clone().inner(),
                tensors.remove(0),
                cfg,
                scale,
            ));
            self.w1_up = track(apply_sgd_tensor(
                self.w1_up.clone().inner(),
                tensors.remove(0),
                cfg,
                scale,
            ));
            self.w2_down = track(apply_sgd_tensor(
                self.w2_down.clone().inner(),
                tensors.remove(0),
                cfg,
                scale,
            ));
            self.w2_up = track(apply_sgd_tensor(
                self.w2_up.clone().inner(),
                tensors.remove(0),
                cfg,
                scale,
            ));
            self.b1_delta = track(apply_sgd_tensor(
                self.b1_delta.clone().inner(),
                tensors.remove(0),
                cfg,
                scale,
            ));
            self.b2_delta = track(apply_sgd_tensor(
                self.b2_delta.clone().inner(),
                tensors.remove(0),
                cfg,
                scale,
            ));
            Ok((norm, scale))
        }
    }

    fn burn_targets(
        examples: &[DirectBasisExample],
        config: DirectBasisTrainConfig,
        device: &BurnDevice,
    ) -> AutomataResult<Vec<BurnTargetExample>> {
        examples
            .iter()
            .map(|example| {
                let positions = example
                    .target
                    .positions
                    .iter()
                    .flat_map(|position| [position[0], position[1]])
                    .collect::<Vec<_>>();
                let colors = example
                    .target
                    .colors
                    .iter()
                    .flat_map(|color| [color[0], color[1], color[2]])
                    .collect::<Vec<_>>();
                Ok(BurnTargetExample {
                    positions: tensor(positions, [example.target.positions.len(), 2], device),
                    colors: tensor(colors, [example.target.colors.len(), 3], device),
                    particle_count: example.source.particles.unwrap_or(config.rollout_particles),
                    update_prob: example.source.update_prob.unwrap_or(config.update_prob),
                    seed_scale: example.source.seed_scale.unwrap_or(config.seed_scale),
                })
            })
            .collect()
    }

    fn adapter_cache_metrics(
        base: &NpaModel,
        train_adapters: &[BurnAdapterParams],
        holdout_adapters: &[BurnAdapterParams],
        train_targets: &[BurnTargetExample],
        holdout_targets: &[BurnTargetExample],
    ) -> serde_json::Value {
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
            .map(|target| target.positions.shape().dims::<2>()[0])
            .sum::<usize>();
        let holdout_target_points = holdout_targets
            .iter()
            .map(|target| target.positions.shape().dims::<2>()[0])
            .sum::<usize>();
        json!({
            "representation": "resident_gpu_tensor_set_per_sample",
            "readback_policy": "end_of_phase_only",
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
            "estimated_target_cache_bytes_f32": (train_target_points + holdout_target_points) * 5 * std::mem::size_of::<f32>(),
        })
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

    fn prepare_grad_group(
        tensors: &mut [Tensor2Inner],
        clip_norm: f32,
        normalize: bool,
    ) -> AutomataResult<(f32, f32)> {
        let original_norm = group_norm(tensors)?;
        if normalize {
            for tensor in tensors.iter_mut() {
                let norm = tensor_l2_norm(tensor)?;
                if norm > 0.0 {
                    *tensor = tensor.clone().div_scalar(norm + 1.0e-8);
                }
            }
        }
        let clip_norm_source = group_norm(tensors)?;
        let scale = if clip_norm > 0.0 && clip_norm_source > clip_norm {
            clip_norm / clip_norm_source.max(f32::MIN_POSITIVE)
        } else {
            1.0
        };
        Ok((original_norm, scale))
    }

    fn group_norm(tensors: &[Tensor2Inner]) -> AutomataResult<f32> {
        let mut total = 0.0_f32;
        for tensor in tensors {
            let norm = tensor_l2_norm(tensor)?;
            total += norm * norm;
        }
        finite_scalar("Burn direct grad norm", total.sqrt())
    }

    fn tensor_l2_norm(tensor: &Tensor2Inner) -> AutomataResult<f32> {
        finite_scalar(
            "Burn direct tensor norm",
            tensor
                .clone()
                .mul(tensor.clone())
                .sum()
                .sqrt()
                .into_scalar(),
        )
    }

    fn apply_sgd_tensor(
        param: Tensor2Inner,
        grad: Tensor2Inner,
        cfg: SgdConfig,
        scale: f32,
    ) -> Tensor2Inner {
        let update = grad.mul_scalar(scale) + param.clone().mul_scalar(cfg.weight_decay);
        param - update.mul_scalar(cfg.learning_rate)
    }

    fn tracked_tensor(values: Vec<f32>, shape: [usize; 2], device: &BurnDevice) -> Tensor2 {
        tensor(values, shape, device).require_grad()
    }

    fn tensor(values: Vec<f32>, shape: [usize; 2], device: &BurnDevice) -> Tensor2 {
        Tensor::<BurnBackend, 2>::from_data(TensorData::new(values, shape), device)
    }

    fn track(tensor: Tensor2Inner) -> Tensor2 {
        Tensor::<BurnBackend, 2>::from_inner(tensor).require_grad()
    }

    fn tensor_vec(tensor: Tensor2Inner) -> AutomataResult<Vec<f32>> {
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
}

#[cfg(feature = "backend_wgpu")]
pub(super) use imp::*;

#[cfg(not(feature = "backend_wgpu"))]
pub(super) struct BurnWgpuDirectBasisOutput {
    pub(super) backend: &'static str,
    pub(super) device: String,
    pub(super) metrics: serde_json::Value,
    pub(super) history: Vec<super::CliHyper2dDirectBasisHistoryEntry>,
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
    _holdout_config: super::DirectBasisTrainConfig,
) -> Result<BurnWgpuDirectBasisOutput, Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "Burn/WGPU direct-basis training requires the backend_wgpu feature; rebuild with --features cli,backend_wgpu or use --gpu-backend upstream-python",
    )
    .into())
}
