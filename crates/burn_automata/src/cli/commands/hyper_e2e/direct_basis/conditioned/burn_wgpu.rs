use burn::{
    backend::Wgpu,
    tensor::{Device, Tensor, TensorData, activation::relu},
};

use super::*;

type WgpuBackend = Wgpu<f32>;
type WgpuDevice = Device<WgpuBackend>;
type Tensor1 = Tensor<WgpuBackend, 1>;
type Tensor2 = Tensor<WgpuBackend, 2>;

const MAX_SAFE_WGPU_TENSOR_BYTES: usize = 120 * 1024 * 1024;

type FlowVectorMetricsSnapshot = Option<(
    AdapterBankVectorMetricsReport,
    Option<AdapterBankVectorMetricsReport>,
)>;

struct WgpuFlowTrainStepReport {
    grad_norm: f32,
    grad_scale: f32,
    diagnostics: Option<AdapterBankFlowOptimizerDiagnosticsReport>,
}

struct WgpuAdapterBankTrainer {
    features: Vec<f32>,
    target: Vec<f32>,
    validation_features: Option<Vec<f32>>,
    validation_target: Option<Vec<f32>>,
    device: WgpuDevice,
    w1: Tensor2,
    b1: Tensor2,
    w2: Tensor2,
    b2: Tensor2,
    w1_m: Tensor2,
    w1_v: Tensor2,
    b1_m: Tensor2,
    b1_v: Tensor2,
    w2_m: Tensor2,
    w2_v: Tensor2,
    b2_m: Tensor2,
    b2_v: Tensor2,
    rows: usize,
    validation_rows: usize,
    input_dims: usize,
    hidden_dims: usize,
    output_dims: usize,
    output_scale: f32,
    loss_eval_batch_size: usize,
    memory_budget_gb: Option<f32>,
    step: usize,
}

pub(super) fn train_adapter_bank_burn_wgpu(
    hyper: &mut HyperNpa2d,
    examples: &[AdapterBankConditionedExample],
    validation_examples: Option<&[AdapterBankConditionedExample]>,
    config: AdapterBankTrainConfig,
) -> Result<AdapterBankTrainingPhaseReport, Box<dyn std::error::Error>> {
    if config.objective == AdapterBankTrainingObjective::RectifiedFlow {
        return train_adapter_bank_flow_burn_wgpu(hyper, examples, validation_examples, config);
    }
    let started = Instant::now();
    let mut trainer = WgpuAdapterBankTrainer::new(hyper, examples, validation_examples, config)?;
    let mut memory = vec![check_process_memory_budget(
        "burn-wgpu adapter-bank tensors initialized",
        config.system_memory_budget_gb,
    )?];
    let initial_loss = trainer.loss(None)?;
    let initial_validation_loss = trainer.validation_loss()?;
    memory.push(check_process_memory_budget(
        "burn-wgpu initial loss evaluated",
        config.system_memory_budget_gb,
    )?);
    let mut final_loss = initial_loss;
    let mut final_validation_loss = initial_validation_loss;
    let mut best_loss = initial_validation_loss.unwrap_or(initial_loss);
    let mut best_validation_loss = initial_validation_loss;
    let mut best_step = 0usize;
    let mut best_weights = hyper.weights.clone();
    let mut history = Vec::new();
    let mut rng = StdRng::seed_from_u64(config.seed);
    let batch_size = normalized_batch_size(config.example_batch_size, examples.len());
    for step in 1..=config.steps {
        let indices = sample_indices(examples.len(), batch_size, &mut rng);
        let step_started = Instant::now();
        let (grad_norm, grad_scale) = trainer.train_step(&indices, config.optimizer)?;
        let step_elapsed = step_started.elapsed();
        if step == config.steps || step.is_multiple_of(config.report_interval.max(1)) {
            final_loss = trainer.loss(None)?;
            final_validation_loss = trainer.validation_loss()?;
            let selection_loss = final_validation_loss.unwrap_or(final_loss);
            if selection_loss < best_loss {
                best_loss = selection_loss;
                best_validation_loss = final_validation_loss;
                best_step = step;
                best_weights = trainer.to_hyper_weights()?;
            }
            let memory_snapshot = check_process_memory_budget(
                format!("burn-wgpu adapter-bank step {step}"),
                config.system_memory_budget_gb,
            )?;
            eprintln!(
                "adapter-bank step {step}/{} loss={:.6e} val={} grad_norm={:.6e} values/s={:.3e} rss={}",
                config.steps,
                final_loss,
                final_validation_loss
                    .map(|loss| format!("{loss:.6e}"))
                    .unwrap_or_else(|| "n/a".to_string()),
                grad_norm,
                (indices.len() * trainer.output_dims) as f64
                    / step_elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
                memory_snapshot
                    .rss_bytes
                    .map(|rss| format!("{:.2}GiB", rss as f64 / (1024.0 * 1024.0 * 1024.0)))
                    .unwrap_or_else(|| "n/a".to_string())
            );
            memory.push(memory_snapshot.clone());
            history.push(AdapterBankTrainingHistoryEntry {
                step,
                loss: final_loss,
                grad_norm,
                grad_scale,
                examples_seen: indices.len(),
                adapter_values_per_sec: (indices.len() * trainer.output_dims) as f64
                    / step_elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
                validation_loss: final_validation_loss,
                memory: memory_snapshot,
                elapsed_ms: step_elapsed.as_secs_f64() * 1000.0,
                train_vector_metrics: None,
                validation_vector_metrics: None,
                flow_optimizer: None,
            });
        }
    }
    let final_selection_loss = final_validation_loss.unwrap_or(final_loss);
    hyper.weights = if best_loss <= final_selection_loss {
        best_weights
    } else {
        trainer.to_hyper_weights()?
    };
    hyper.validate()?;
    let final_trainer = WgpuAdapterBankTrainer::new(hyper, examples, validation_examples, config)?;
    final_loss = final_trainer.loss(None)?;
    final_validation_loss = final_trainer.validation_loss()?;
    memory.push(check_process_memory_budget(
        "burn-wgpu final loss evaluated",
        config.system_memory_budget_gb,
    )?);
    Ok(AdapterBankTrainingPhaseReport {
        backend: "burn_wgpu_manual_mlp_adapter_regression".to_string(),
        device: "wgpu-default".to_string(),
        selection_metric: if initial_validation_loss.is_some() {
            "holdout_adapter_vector_mse".to_string()
        } else {
            "train_adapter_vector_mse".to_string()
        },
        initial_loss,
        initial_validation_loss,
        final_loss,
        final_validation_loss,
        best_loss,
        best_validation_loss,
        best_step,
        history,
        memory,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        vector_selection: None,
    })
}

impl WgpuAdapterBankTrainer {
    fn new(
        hyper: &HyperNpa2d,
        examples: &[AdapterBankConditionedExample],
        validation_examples: Option<&[AdapterBankConditionedExample]>,
        config: AdapterBankTrainConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        hyper.validate()?;
        if examples.is_empty() {
            return Err(
                std::io::Error::other("Burn adapter-bank training requires examples").into(),
            );
        }
        let rows = examples.len();
        let input_dims = hyper.config.condition_feature_dims;
        let output_dims = hyper.adapter_parameter_count();
        let hidden_dims = hyper.config.hidden_dims;
        let device = WgpuDevice::default();
        let (features, target) = example_buffers(hyper, examples, input_dims, output_dims)?;
        let validation_rows = validation_examples.map_or(0, |examples| examples.len());
        let (validation_features, validation_target) = if let Some(validation_examples) =
            validation_examples.filter(|examples| !examples.is_empty())
        {
            let (features, target) =
                example_buffers(hyper, validation_examples, input_dims, output_dims)?;
            (Some(features), Some(target))
        } else {
            (None, None)
        };
        validate_wgpu_tensor_allocation("adapter-bank w1", hidden_dims, input_dims)?;
        validate_wgpu_tensor_allocation("adapter-bank w2", output_dims, hidden_dims)?;
        let w1 = tensor2(hyper.weights.w1.clone(), [hidden_dims, input_dims], &device);
        let b1 = tensor2(hyper.weights.b1.clone(), [1, hidden_dims], &device);
        let w2 = tensor2(
            hyper.weights.w2.clone(),
            [output_dims, hidden_dims],
            &device,
        );
        let b2 = tensor2(hyper.weights.b2.clone(), [1, output_dims], &device);
        Ok(Self {
            w1_m: w1.zeros_like(),
            w1_v: w1.zeros_like(),
            b1_m: b1.zeros_like(),
            b1_v: b1.zeros_like(),
            w2_m: w2.zeros_like(),
            w2_v: w2.zeros_like(),
            b2_m: b2.zeros_like(),
            b2_v: b2.zeros_like(),
            features,
            target,
            validation_features,
            validation_target,
            device,
            w1,
            b1,
            w2,
            b2,
            rows,
            validation_rows,
            input_dims,
            hidden_dims,
            output_dims,
            output_scale: hyper.config.output_scale,
            loss_eval_batch_size: config.loss_eval_batch_size,
            memory_budget_gb: config.system_memory_budget_gb,
            step: 0,
        })
    }

    fn loss(&self, indices: Option<&[usize]>) -> Result<f32, Box<dyn std::error::Error>> {
        if let Some(indices) = indices {
            let (features, target, rows) = self.batch(Some(indices));
            return self.loss_for(features, target, rows);
        }
        self.dataset_loss(&self.features, &self.target, self.rows, "train")
    }

    fn validation_loss(&self) -> Result<Option<f32>, Box<dyn std::error::Error>> {
        let Some(features) = self.validation_features.as_ref() else {
            return Ok(None);
        };
        let target = self
            .validation_target
            .as_ref()
            .ok_or_else(|| std::io::Error::other("validation target missing"))?;
        Ok(Some(self.dataset_loss(
            features,
            target,
            self.validation_rows,
            "validation",
        )?))
    }

    fn dataset_loss(
        &self,
        features: &[f32],
        target: &[f32],
        rows: usize,
        label: &str,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        if rows == 0 {
            return Err(std::io::Error::other("Burn/WGPU loss requires non-empty rows").into());
        }
        let chunk = self.loss_eval_batch_size.min(rows).max(1);
        let mut sum = 0.0_f64;
        for start in (0..rows).step_by(chunk) {
            let end = (start + chunk).min(rows);
            let (feature_tensor, target_tensor, chunk_rows) =
                self.row_range_tensors(features, target, start, end);
            sum += f64::from(self.loss_sum_for(feature_tensor, target_tensor, chunk_rows)?);
            check_process_memory_budget(
                format!("burn-wgpu {label} loss chunk {end}/{rows}"),
                self.memory_budget_gb,
            )?;
        }
        finite_scalar(
            "Burn/WGPU adapter-bank chunked loss",
            (sum / (rows * self.output_dims) as f64) as f32,
        )
    }

    fn loss_for(
        &self,
        features: Tensor2,
        target: Tensor2,
        rows: usize,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let sum = self.loss_sum_for(features, target, rows)?;
        finite_scalar(
            "Burn/WGPU adapter-bank loss",
            sum / (rows * self.output_dims) as f32,
        )
    }

    fn loss_sum_for(
        &self,
        features: Tensor2,
        target: Tensor2,
        rows: usize,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let (_, _, _, output) = self.forward(features, rows);
        let diff = output - target;
        let loss = diff.clone().mul(diff).sum();
        finite_scalar("Burn/WGPU adapter-bank loss sum", loss.into_scalar())
    }

    fn train_step(
        &mut self,
        indices: &[usize],
        optimizer: AdamWConfig,
    ) -> Result<(f32, f32), Box<dyn std::error::Error>> {
        let (features, target, rows) = self.batch(Some(indices));
        let (pre_hidden, hidden, pre_output, output) = self.forward(features.clone(), rows);
        let diff = output - target;
        let d_output = diff
            .clone()
            .mul_scalar(2.0 / (rows * self.output_dims) as f32);
        let tanh_pre = pre_output.tanh();
        let d_pre_output = d_output
            .mul(
                tanh_pre
                    .clone()
                    .mul(tanh_pre)
                    .mul_scalar(-1.0)
                    .add_scalar(1.0),
            )
            .mul_scalar(self.output_scale);
        let gb2 = d_pre_output.clone().sum_dim(0);
        let gw2 = d_pre_output.clone().transpose().matmul(hidden.clone());
        let d_hidden = d_pre_output.matmul(self.w2.clone());
        let d_pre_hidden = d_hidden.mask_fill(pre_hidden.clone().lower_equal_elem(0.0), 0.0);
        let gb1 = d_pre_hidden.clone().sum_dim(0);
        let gw1 = d_pre_hidden.transpose().matmul(features);
        let grad_norm_tensor = tensor_l2_norm([gw1.clone(), gb1.clone(), gw2.clone(), gb2.clone()]);
        let grad_scale_tensor =
            gradient_scale_tensor(grad_norm_tensor.clone(), optimizer.grad_clip_norm);
        self.step = self.step.saturating_add(1);
        self.apply_adamw([gw1, gb1, gw2, gb2], optimizer, grad_scale_tensor.clone());
        self.detach_state();
        let grad_norm = finite_scalar(
            "Burn/WGPU adapter-bank grad norm",
            grad_norm_tensor.into_scalar(),
        )?;
        let grad_scale = finite_scalar(
            "Burn/WGPU adapter-bank grad scale",
            grad_scale_tensor.into_scalar(),
        )?;
        Ok((grad_norm, grad_scale))
    }

    fn batch(&self, indices: Option<&[usize]>) -> (Tensor2, Tensor2, usize) {
        let Some(indices) = indices else {
            return self.row_range_tensors(&self.features, &self.target, 0, self.rows);
        };
        if indices.len() >= self.rows {
            return self.row_range_tensors(&self.features, &self.target, 0, self.rows);
        }
        let mut feature_values = Vec::with_capacity(indices.len() * self.input_dims);
        let mut target_values = Vec::with_capacity(indices.len() * self.output_dims);
        for &idx in indices {
            let feature_base = idx * self.input_dims;
            let target_base = idx * self.output_dims;
            feature_values
                .extend_from_slice(&self.features[feature_base..feature_base + self.input_dims]);
            target_values
                .extend_from_slice(&self.target[target_base..target_base + self.output_dims]);
        }
        (
            tensor2(
                feature_values,
                [indices.len(), self.input_dims],
                &self.device,
            ),
            tensor2(
                target_values,
                [indices.len(), self.output_dims],
                &self.device,
            ),
            indices.len(),
        )
    }

    fn row_range_tensors(
        &self,
        features: &[f32],
        target: &[f32],
        start: usize,
        end: usize,
    ) -> (Tensor2, Tensor2, usize) {
        let rows = end - start;
        let feature_start = start * self.input_dims;
        let feature_end = end * self.input_dims;
        let target_start = start * self.output_dims;
        let target_end = end * self.output_dims;
        (
            tensor2(
                features[feature_start..feature_end].to_vec(),
                [rows, self.input_dims],
                &self.device,
            ),
            tensor2(
                target[target_start..target_end].to_vec(),
                [rows, self.output_dims],
                &self.device,
            ),
            rows,
        )
    }

    fn forward(&self, features: Tensor2, rows: usize) -> (Tensor2, Tensor2, Tensor2, Tensor2) {
        let pre_hidden = features.matmul(self.w1.clone().transpose())
            + self.b1.clone().expand([rows, self.hidden_dims]);
        let hidden = relu(pre_hidden.clone());
        let pre_output = hidden.clone().matmul(self.w2.clone().transpose())
            + self.b2.clone().expand([rows, self.output_dims]);
        let output = pre_output.clone().tanh().mul_scalar(self.output_scale);
        (pre_hidden, hidden, pre_output, output)
    }

    fn apply_adamw(&mut self, grads: [Tensor2; 4], cfg: AdamWConfig, grad_scale: Tensor2) {
        let [gw1, gb1, gw2, gb2] = grads;
        let bias_correction1 = 1.0 - cfg.beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - cfg.beta2.powi(self.step as i32);
        (self.w1, self.w1_m, self.w1_v) = apply_adamw_tensor(
            self.w1.clone(),
            gw1,
            self.w1_m.clone(),
            self.w1_v.clone(),
            cfg,
            grad_scale.clone(),
            bias_correction1,
            bias_correction2,
        );
        (self.b1, self.b1_m, self.b1_v) = apply_adamw_tensor(
            self.b1.clone(),
            gb1,
            self.b1_m.clone(),
            self.b1_v.clone(),
            cfg,
            grad_scale.clone(),
            bias_correction1,
            bias_correction2,
        );
        (self.w2, self.w2_m, self.w2_v) = apply_adamw_tensor(
            self.w2.clone(),
            gw2,
            self.w2_m.clone(),
            self.w2_v.clone(),
            cfg,
            grad_scale.clone(),
            bias_correction1,
            bias_correction2,
        );
        (self.b2, self.b2_m, self.b2_v) = apply_adamw_tensor(
            self.b2.clone(),
            gb2,
            self.b2_m.clone(),
            self.b2_v.clone(),
            cfg,
            grad_scale,
            bias_correction1,
            bias_correction2,
        );
    }

    fn detach_state(&mut self) {
        self.w1 = self.w1.clone().detach();
        self.b1 = self.b1.clone().detach();
        self.w2 = self.w2.clone().detach();
        self.b2 = self.b2.clone().detach();
        self.w1_m = self.w1_m.clone().detach();
        self.w1_v = self.w1_v.clone().detach();
        self.b1_m = self.b1_m.clone().detach();
        self.b1_v = self.b1_v.clone().detach();
        self.w2_m = self.w2_m.clone().detach();
        self.w2_v = self.w2_v.clone().detach();
        self.b2_m = self.b2_m.clone().detach();
        self.b2_v = self.b2_v.clone().detach();
    }

    fn to_hyper_weights(&self) -> Result<crate::HyperNpa2dWeights, Box<dyn std::error::Error>> {
        let weights = crate::HyperNpa2dWeights {
            w1: tensor_vec(self.w1.clone())?,
            b1: tensor_vec(self.b1.clone())?,
            w2: tensor_vec(self.w2.clone())?,
            b2: tensor_vec(self.b2.clone())?,
        };
        if weights.w1.len() != self.hidden_dims * self.input_dims
            || weights.b1.len() != self.hidden_dims
            || weights.w2.len() != self.output_dims * self.hidden_dims
            || weights.b2.len() != self.output_dims
        {
            return Err(
                std::io::Error::other("Burn/WGPU hyper weight readback shape mismatch").into(),
            );
        }
        Ok(weights)
    }
}

struct WgpuAdapterFlowTrainer {
    features: Vec<f32>,
    target: Vec<f32>,
    sample_weights: Vec<f32>,
    validation_features: Option<Vec<f32>>,
    validation_target: Option<Vec<f32>>,
    validation_sample_weights: Option<Vec<f32>>,
    device: WgpuDevice,
    w1: Tensor2,
    b1: Tensor2,
    w2: Tensor2,
    b2: Tensor2,
    w1_m: Tensor2,
    w1_v: Tensor2,
    b1_m: Tensor2,
    b1_v: Tensor2,
    w2_m: Tensor2,
    w2_v: Tensor2,
    b2_m: Tensor2,
    b2_v: Tensor2,
    rows: usize,
    validation_rows: usize,
    condition_dims: usize,
    input_dims: usize,
    hidden_dims: usize,
    output_dims: usize,
    sample_steps: usize,
    source_scale: f32,
    hidden_activation: HyperNpa2dFlowActivation,
    flow_loss: AdapterBankFlowLoss,
    loss_eval_batch_size: usize,
    memory_budget_gb: Option<f32>,
    sample_seed: u64,
    step: usize,
}

fn train_adapter_bank_flow_burn_wgpu(
    hyper: &mut HyperNpa2d,
    examples: &[AdapterBankConditionedExample],
    validation_examples: Option<&[AdapterBankConditionedExample]>,
    config: AdapterBankTrainConfig,
) -> Result<AdapterBankTrainingPhaseReport, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let mut trainer = WgpuAdapterFlowTrainer::new(hyper, examples, validation_examples, config)?;
    let mut memory = vec![check_process_memory_budget(
        "burn-wgpu adapter-flow tensors initialized",
        config.system_memory_budget_gb,
    )?];
    let initial_loss = trainer.loss(None, config.seed)?;
    let initial_validation_loss = trainer.validation_loss(config.seed ^ 0x5eed_5eed)?;
    let initial_vector_metrics =
        flow_vector_metrics_from_trainer(hyper, &trainer, examples, validation_examples, config)?;
    memory.push(check_process_memory_budget(
        "burn-wgpu adapter-flow initial loss evaluated",
        config.system_memory_budget_gb,
    )?);
    let mut final_loss = initial_loss;
    let mut final_validation_loss = initial_validation_loss;
    let mut final_vector_metrics = initial_vector_metrics;
    let mut best_loss = flow_selection_loss(
        final_vector_metrics,
        initial_validation_loss.unwrap_or(initial_loss),
    );
    let mut best_validation_loss = initial_validation_loss;
    let mut best_vector_metrics = final_vector_metrics;
    let mut best_step = 0usize;
    let mut best_weights = trainer.to_flow_weights()?;
    let mut history = Vec::new();
    let mut rng = StdRng::seed_from_u64(config.seed);
    let batch_size = normalized_batch_size(config.example_batch_size, examples.len());
    for step in 1..=config.steps {
        let indices = sample_indices(examples.len(), batch_size, &mut rng);
        let step_started = Instant::now();
        let is_report_step =
            step == config.steps || step.is_multiple_of(config.report_interval.max(1));
        let step_report =
            trainer.train_step(&indices, config.optimizer, config.seed, is_report_step)?;
        let step_elapsed = step_started.elapsed();
        if is_report_step {
            final_loss = trainer.loss(None, config.seed)?;
            final_validation_loss = trainer.validation_loss(config.seed ^ 0x5eed_5eed)?;
            final_vector_metrics = flow_vector_metrics_from_trainer(
                hyper,
                &trainer,
                examples,
                validation_examples,
                config,
            )?;
            let selection_loss = flow_selection_loss(
                final_vector_metrics,
                final_validation_loss.unwrap_or(final_loss),
            );
            if selection_loss < best_loss {
                best_loss = selection_loss;
                best_validation_loss = final_validation_loss;
                best_step = step;
                best_vector_metrics = final_vector_metrics;
                best_weights = trainer.to_flow_weights()?;
            }
            let memory_snapshot = check_process_memory_budget(
                format!("burn-wgpu adapter-flow step {step}"),
                config.system_memory_budget_gb,
            )?;
            let train_vector_suffix = final_vector_metrics
                .map(|(train, validation)| {
                    format!(
                        " train_vec_nrmse={} train_vec_cos={:.6} val_vec_nrmse={} val_vec_cos={}",
                        train
                            .normalized_rmse_to_target_rms
                            .map(|value| format!("{value:.6e}"))
                            .unwrap_or_else(|| "n/a".to_string()),
                        train.mean_cosine_similarity,
                        validation
                            .and_then(|metrics| metrics.normalized_rmse_to_target_rms)
                            .map(|value| format!("{value:.6e}"))
                            .unwrap_or_else(|| "n/a".to_string()),
                        validation
                            .map(|metrics| format!("{:.6}", metrics.mean_cosine_similarity))
                            .unwrap_or_else(|| "n/a".to_string())
                    )
                })
                .unwrap_or_default();
            eprintln!(
                "adapter-flow step {step}/{} loss={:.6e} val={} grad_norm={:.6e} values/s={:.3e} rss={}{}",
                config.steps,
                final_loss,
                final_validation_loss
                    .map(|loss| format!("{loss:.6e}"))
                    .unwrap_or_else(|| "n/a".to_string()),
                step_report.grad_norm,
                (indices.len() * trainer.output_dims) as f64
                    / step_elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
                memory_snapshot
                    .rss_bytes
                    .map(|rss| format!("{:.2}GiB", rss as f64 / (1024.0 * 1024.0 * 1024.0)))
                    .unwrap_or_else(|| "n/a".to_string()),
                train_vector_suffix
            );
            memory.push(memory_snapshot.clone());
            history.push(AdapterBankTrainingHistoryEntry {
                step,
                loss: final_loss,
                grad_norm: step_report.grad_norm,
                grad_scale: step_report.grad_scale,
                examples_seen: indices.len(),
                adapter_values_per_sec: (indices.len() * trainer.output_dims) as f64
                    / step_elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
                validation_loss: final_validation_loss,
                memory: memory_snapshot,
                elapsed_ms: step_elapsed.as_secs_f64() * 1000.0,
                train_vector_metrics: final_vector_metrics.map(|(train, _)| train),
                validation_vector_metrics: final_vector_metrics
                    .and_then(|(_, validation)| validation),
                flow_optimizer: step_report.diagnostics,
            });
        }
    }
    let final_selection_loss = flow_selection_loss(
        final_vector_metrics,
        final_validation_loss.unwrap_or(final_loss),
    );
    let final_weights = if best_loss <= final_selection_loss {
        best_weights
    } else {
        trainer.to_flow_weights()?
    };
    hyper.set_flow(crate::HyperNpa2dFlow {
        config: crate::HyperNpa2dFlowConfig {
            hidden_dims: config.flow.hidden_dims,
            sample_steps: config.flow.sample_steps,
            source_scale: config.flow.source_scale,
            sample_seed: config.flow.sample_seed,
            hidden_activation: config.flow.hidden_activation,
        },
        weights: final_weights,
    })?;
    let final_trainer = WgpuAdapterFlowTrainer::new(hyper, examples, validation_examples, config)?;
    final_loss = final_trainer.loss(None, config.seed)?;
    final_validation_loss = final_trainer.validation_loss(config.seed ^ 0x5eed_5eed)?;
    final_vector_metrics =
        flow_vector_metrics_from_hyper(hyper, examples, validation_examples, config)?;
    memory.push(check_process_memory_budget(
        "burn-wgpu adapter-flow final loss evaluated",
        config.system_memory_budget_gb,
    )?);
    Ok(AdapterBankTrainingPhaseReport {
        backend: "burn_wgpu_rectified_flow_lora_vector".to_string(),
        device: "wgpu-default".to_string(),
        selection_metric: if config.diagnostic_vector_examples > 0 {
            if validation_examples.is_some_and(|examples| !examples.is_empty()) {
                "generated_holdout_adapter_vector_mse".to_string()
            } else {
                "generated_train_adapter_vector_mse".to_string()
            }
        } else if initial_validation_loss.is_some() {
            "holdout_flow_velocity_mse".to_string()
        } else {
            "train_flow_velocity_mse".to_string()
        },
        initial_loss,
        initial_validation_loss,
        final_loss,
        final_validation_loss,
        best_loss,
        best_validation_loss,
        best_step,
        history,
        memory,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        vector_selection: final_vector_metrics.map(|(final_train, final_validation)| {
            let (initial_train, initial_validation) = initial_vector_metrics
                .expect("initial flow vector metrics must exist when final metrics exist");
            let (best_train, best_validation) = best_vector_metrics
                .expect("best flow vector metrics must exist when final metrics exist");
            AdapterBankTrainingVectorSelectionReport {
                requested_examples: config.diagnostic_vector_examples,
                initial_train,
                initial_validation,
                final_train,
                final_validation,
                best_train,
                best_validation,
            }
        }),
    })
}

fn flow_vector_metrics_from_trainer(
    hyper: &HyperNpa2d,
    trainer: &WgpuAdapterFlowTrainer,
    examples: &[AdapterBankConditionedExample],
    validation_examples: Option<&[AdapterBankConditionedExample]>,
    config: AdapterBankTrainConfig,
) -> Result<FlowVectorMetricsSnapshot, Box<dyn std::error::Error>> {
    if config.diagnostic_vector_examples == 0 {
        return Ok(None);
    }
    let mut current = hyper.clone();
    current.set_flow(crate::HyperNpa2dFlow {
        config: crate::HyperNpa2dFlowConfig {
            hidden_dims: config.flow.hidden_dims,
            sample_steps: config.flow.sample_steps,
            source_scale: config.flow.source_scale,
            sample_seed: config.flow.sample_seed,
            hidden_activation: config.flow.hidden_activation,
        },
        weights: trainer.to_flow_weights()?,
    })?;
    flow_vector_metrics_from_hyper(&current, examples, validation_examples, config)
}

fn flow_vector_metrics_from_hyper(
    hyper: &HyperNpa2d,
    examples: &[AdapterBankConditionedExample],
    validation_examples: Option<&[AdapterBankConditionedExample]>,
    config: AdapterBankTrainConfig,
) -> Result<FlowVectorMetricsSnapshot, Box<dyn std::error::Error>> {
    if config.diagnostic_vector_examples == 0 {
        return Ok(None);
    }
    let train = vector_metrics(
        hyper,
        examples,
        config.diagnostic_vector_examples,
        config.seed,
    )?;
    let validation = if let Some(validation_examples) =
        validation_examples.filter(|examples| !examples.is_empty())
    {
        Some(vector_metrics(
            hyper,
            validation_examples,
            config.diagnostic_vector_examples,
            config.seed ^ 0xa11c_e5e1,
        )?)
    } else {
        None
    };
    Ok(Some((train, validation)))
}

fn flow_selection_loss(vector_metrics: FlowVectorMetricsSnapshot, fallback_loss: f32) -> f32 {
    vector_metrics
        .map(|(train, validation)| validation.unwrap_or(train).mse)
        .unwrap_or(fallback_loss)
}

impl WgpuAdapterFlowTrainer {
    fn new(
        hyper: &HyperNpa2d,
        examples: &[AdapterBankConditionedExample],
        validation_examples: Option<&[AdapterBankConditionedExample]>,
        config: AdapterBankTrainConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        hyper.validate()?;
        if examples.is_empty() {
            return Err(
                std::io::Error::other("Burn adapter-flow training requires examples").into(),
            );
        }
        let rows = examples.len();
        let condition_dims = hyper.config.condition_feature_dims;
        let output_dims = hyper.adapter_parameter_count();
        let input_dims = condition_dims
            .checked_add(1)
            .and_then(|dims| dims.checked_add(output_dims))
            .ok_or_else(|| std::io::Error::other("adapter-flow input dimensions overflow"))?;
        let hidden_dims = config.flow.hidden_dims;
        let device = WgpuDevice::default();
        let (features, target) = example_buffers(hyper, examples, condition_dims, output_dims)?;
        let sample_weights = example_sample_weights(examples);
        let validation_rows = validation_examples.map_or(0, |examples| examples.len());
        let (validation_features, validation_target, validation_sample_weights) =
            if let Some(validation_examples) =
                validation_examples.filter(|examples| !examples.is_empty())
            {
                let (features, target) =
                    example_buffers(hyper, validation_examples, condition_dims, output_dims)?;
                (
                    Some(features),
                    Some(target),
                    Some(example_sample_weights(validation_examples)),
                )
            } else {
                (None, None, None)
            };
        let flow = hyper.flow.as_ref();
        let (w1_values, b1_values, w2_values, b2_values) = if let Some(flow) = flow {
            (
                flow.weights.w1.clone(),
                flow.weights.b1.clone(),
                flow.weights.w2.clone(),
                flow.weights.b2.clone(),
            )
        } else {
            match config.flow.init {
                AdapterBankFlowInit::Random => {
                    seeded_flow_weights(input_dims, hidden_dims, output_dims, config.seed)
                }
                AdapterBankFlowInit::LinearSolveConditionWarmstart => {
                    let weights = linear_solve_rectified_flow_condition_weights(
                        hyper,
                        examples,
                        hidden_dims,
                    )?;
                    (weights.w1, weights.b1, weights.w2, weights.b2)
                }
                AdapterBankFlowInit::FromHyper => {
                    return Err(std::io::Error::other(
                        "flow_init=from-hyper requires input.initial_hyper to preload flow weights",
                    )
                    .into());
                }
            }
        };
        validate_wgpu_tensor_allocation("adapter-flow w1", hidden_dims, input_dims)?;
        validate_wgpu_tensor_allocation("adapter-flow w2", output_dims, hidden_dims)?;
        let w1 = tensor2(w1_values, [hidden_dims, input_dims], &device);
        let b1 = tensor2(b1_values, [1, hidden_dims], &device);
        let w2 = tensor2(w2_values, [output_dims, hidden_dims], &device);
        let b2 = tensor2(b2_values, [1, output_dims], &device);
        Ok(Self {
            w1_m: w1.zeros_like(),
            w1_v: w1.zeros_like(),
            b1_m: b1.zeros_like(),
            b1_v: b1.zeros_like(),
            w2_m: w2.zeros_like(),
            w2_v: w2.zeros_like(),
            b2_m: b2.zeros_like(),
            b2_v: b2.zeros_like(),
            features,
            target,
            sample_weights,
            validation_features,
            validation_target,
            validation_sample_weights,
            device,
            w1,
            b1,
            w2,
            b2,
            rows,
            validation_rows,
            condition_dims,
            input_dims,
            hidden_dims,
            output_dims,
            sample_steps: config.flow.sample_steps,
            source_scale: config.flow.source_scale,
            hidden_activation: config.flow.hidden_activation,
            flow_loss: config.flow.loss,
            loss_eval_batch_size: config.loss_eval_batch_size,
            memory_budget_gb: config.system_memory_budget_gb,
            sample_seed: config.flow.sample_seed,
            step: 0,
        })
    }

    fn loss(
        &self,
        indices: Option<&[usize]>,
        sample_seed: u64,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        if self.flow_loss == AdapterBankFlowLoss::SampledAdapterMse {
            return self.sampled_adapter_loss(indices);
        }
        if let Some(indices) = indices {
            let (inputs, velocity, rows) = self.batch(Some(indices), sample_seed);
            return self.loss_for(inputs, velocity, rows);
        }
        self.dataset_loss(
            &self.features,
            &self.target,
            self.rows,
            "train",
            sample_seed,
        )
    }

    fn validation_loss(&self, sample_seed: u64) -> Result<Option<f32>, Box<dyn std::error::Error>> {
        let Some(features) = self.validation_features.as_ref() else {
            return Ok(None);
        };
        let target = self
            .validation_target
            .as_ref()
            .ok_or_else(|| std::io::Error::other("validation target missing"))?;
        if self.flow_loss == AdapterBankFlowLoss::SampledAdapterMse {
            return Ok(Some(
                self.sampled_adapter_dataset_loss(
                    features,
                    target,
                    self.validation_sample_weights.as_ref().ok_or_else(|| {
                        std::io::Error::other("validation sample weights missing")
                    })?,
                    self.validation_rows,
                    "validation",
                )?,
            ));
        }
        Ok(Some(self.dataset_loss(
            features,
            target,
            self.validation_rows,
            "validation",
            sample_seed,
        )?))
    }

    fn dataset_loss(
        &self,
        features: &[f32],
        target: &[f32],
        rows: usize,
        label: &str,
        sample_seed: u64,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        if rows == 0 {
            return Err(
                std::io::Error::other("Burn/WGPU flow loss requires non-empty rows").into(),
            );
        }
        let chunk = self.loss_eval_batch_size.min(rows).max(1);
        let mut sum = 0.0_f64;
        for start in (0..rows).step_by(chunk) {
            let end = (start + chunk).min(rows);
            let (input_tensor, velocity_tensor, chunk_rows) =
                self.row_range_tensors(features, target, start, end, sample_seed);
            sum += f64::from(self.loss_sum_for(input_tensor, velocity_tensor, chunk_rows)?);
            check_process_memory_budget(
                format!("burn-wgpu {label} flow loss chunk {end}/{rows}"),
                self.memory_budget_gb,
            )?;
        }
        finite_scalar(
            "Burn/WGPU adapter-flow chunked loss",
            (sum / (rows * self.output_dims) as f64) as f32,
        )
    }

    fn sampled_adapter_loss(
        &self,
        indices: Option<&[usize]>,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        if let Some(indices) = indices {
            let (features, target, weights, rows, weight_sum) =
                self.feature_target_batch(Some(indices));
            let sum = self.sampled_adapter_loss_sum(features, target, weights, rows)?;
            return finite_scalar(
                "Burn/WGPU sampled adapter-flow loss",
                sum / (weight_sum * self.output_dims as f32),
            );
        }
        self.sampled_adapter_dataset_loss(
            &self.features,
            &self.target,
            &self.sample_weights,
            self.rows,
            "train",
        )
    }

    fn sampled_adapter_dataset_loss(
        &self,
        features: &[f32],
        target: &[f32],
        sample_weights: &[f32],
        rows: usize,
        label: &str,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        if rows == 0 {
            return Err(std::io::Error::other(
                "Burn/WGPU sampled adapter-flow loss requires non-empty rows",
            )
            .into());
        }
        let chunk = self.loss_eval_batch_size.min(rows).max(1);
        let mut sum = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        for start in (0..rows).step_by(chunk) {
            let end = (start + chunk).min(rows);
            let (feature_tensor, target_tensor, weight_tensor, chunk_rows, chunk_weight_sum) =
                self.feature_target_range_tensors(features, target, sample_weights, start, end);
            sum += f64::from(self.sampled_adapter_loss_sum(
                feature_tensor,
                target_tensor,
                weight_tensor,
                chunk_rows,
            )?);
            weight_sum += f64::from(chunk_weight_sum);
            check_process_memory_budget(
                format!("burn-wgpu {label} sampled adapter-flow loss chunk {end}/{rows}"),
                self.memory_budget_gb,
            )?;
        }
        if weight_sum <= 0.0 {
            return Err(std::io::Error::other(
                "Burn/WGPU sampled adapter-flow loss has zero weight sum",
            )
            .into());
        }
        finite_scalar(
            "Burn/WGPU sampled adapter-flow chunked loss",
            (sum / (weight_sum * self.output_dims as f64)) as f32,
        )
    }

    fn loss_for(
        &self,
        inputs: Tensor2,
        velocity: Tensor2,
        rows: usize,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let sum = self.loss_sum_for(inputs, velocity, rows)?;
        finite_scalar(
            "Burn/WGPU adapter-flow loss",
            sum / (rows * self.output_dims) as f32,
        )
    }

    fn loss_sum_for(
        &self,
        inputs: Tensor2,
        velocity: Tensor2,
        rows: usize,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let (_, _, output) = self.forward(inputs, rows);
        let diff = output - velocity;
        let loss = diff.clone().mul(diff).sum();
        finite_scalar("Burn/WGPU adapter-flow loss sum", loss.into_scalar())
    }

    fn train_step(
        &mut self,
        indices: &[usize],
        optimizer: AdamWConfig,
        sample_seed: u64,
        collect_diagnostics: bool,
    ) -> Result<WgpuFlowTrainStepReport, Box<dyn std::error::Error>> {
        if self.flow_loss == AdapterBankFlowLoss::SampledAdapterMse {
            return self.train_sampled_adapter_step(indices, optimizer, collect_diagnostics);
        }
        let step_seed = sample_seed ^ ((self.step as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let (inputs, velocity, rows) = self.batch(Some(indices), step_seed);
        let (pre_hidden, hidden, output) = self.forward(inputs.clone(), rows);
        let diff = output.clone() - velocity.clone();
        let d_output = diff
            .clone()
            .mul_scalar(2.0 / (rows * self.output_dims) as f32);
        let gb2 = d_output.clone().sum_dim(0);
        let gw2 = d_output.clone().transpose().matmul(hidden.clone());
        let d_hidden = d_output.matmul(self.w2.clone());
        let d_pre_hidden =
            flow_hidden_backward(d_hidden, pre_hidden.clone(), self.hidden_activation);
        let gb1 = d_pre_hidden.clone().sum_dim(0);
        let gw1 = d_pre_hidden.transpose().matmul(inputs);
        let grad_norm_tensor = tensor_l2_norm([gw1.clone(), gb1.clone(), gw2.clone(), gb2.clone()]);
        let grad_scale_tensor =
            gradient_scale_tensor(grad_norm_tensor.clone(), optimizer.grad_clip_norm);
        let diagnostics = if collect_diagnostics {
            Some(AdapterBankFlowOptimizerDiagnosticsReport {
                prediction_rms: tensor_rms(output, rows * self.output_dims, "flow prediction rms")?,
                velocity_rms: tensor_rms(velocity, rows * self.output_dims, "flow velocity rms")?,
                residual_rms: tensor_rms(diff, rows * self.output_dims, "flow residual rms")?,
                pre_hidden_rms: tensor_rms(
                    pre_hidden.clone(),
                    rows * self.hidden_dims,
                    "flow pre-hidden rms",
                )?,
                hidden_rms: tensor_rms(hidden, rows * self.hidden_dims, "flow hidden rms")?,
                hidden_zero_fraction: hidden_zero_fraction(pre_hidden)?,
                grad_w1_norm: tensor_l2_norm_single(gw1.clone(), "flow grad w1 norm")?,
                grad_b1_norm: tensor_l2_norm_single(gb1.clone(), "flow grad b1 norm")?,
                grad_w2_norm: tensor_l2_norm_single(gw2.clone(), "flow grad w2 norm")?,
                grad_b2_norm: tensor_l2_norm_single(gb2.clone(), "flow grad b2 norm")?,
            })
        } else {
            None
        };
        self.step = self.step.saturating_add(1);
        self.apply_adamw([gw1, gb1, gw2, gb2], optimizer, grad_scale_tensor.clone());
        self.detach_state();
        let grad_norm = finite_scalar(
            "Burn/WGPU adapter-flow grad norm",
            grad_norm_tensor.into_scalar(),
        )?;
        let grad_scale = finite_scalar(
            "Burn/WGPU adapter-flow grad scale",
            grad_scale_tensor.into_scalar(),
        )?;
        Ok(WgpuFlowTrainStepReport {
            grad_norm,
            grad_scale,
            diagnostics,
        })
    }

    fn train_sampled_adapter_step(
        &mut self,
        indices: &[usize],
        optimizer: AdamWConfig,
        collect_diagnostics: bool,
    ) -> Result<WgpuFlowTrainStepReport, Box<dyn std::error::Error>> {
        let (features, target, weights, rows, weight_sum) =
            self.feature_target_batch(Some(indices));
        let (state, mut caches) = self.sampled_adapter_forward_with_cache(features, rows);
        let diff = state.clone() - target.clone();
        let mut d_state = diff
            .clone()
            .mul(weights.clone().expand([rows, self.output_dims]))
            .mul_scalar(2.0 / (weight_sum * self.output_dims as f32));
        let diagnostic_cache = if collect_diagnostics {
            caches
                .last()
                .map(|(_input, pre_hidden, hidden)| (pre_hidden.clone(), hidden.clone()))
        } else {
            None
        };
        let mut gw1 = self.w1.zeros_like();
        let mut gb1 = self.b1.zeros_like();
        let mut gw2 = self.w2.zeros_like();
        let mut gb2 = self.b2.zeros_like();
        let dt = 1.0 / self.sample_steps.max(1) as f32;
        while let Some((input, pre_hidden, hidden)) = caches.pop() {
            let d_output = d_state.clone().mul_scalar(dt);
            gb2 = gb2 + d_output.clone().sum_dim(0);
            gw2 = gw2 + d_output.clone().transpose().matmul(hidden.clone());
            let d_hidden = d_output.matmul(self.w2.clone());
            let d_pre_hidden =
                flow_hidden_backward(d_hidden, pre_hidden.clone(), self.hidden_activation);
            gb1 = gb1 + d_pre_hidden.clone().sum_dim(0);
            gw1 = gw1 + d_pre_hidden.clone().transpose().matmul(input.clone());
            let d_input = d_pre_hidden.matmul(self.w1.clone());
            d_state = d_state + d_input.narrow(1, self.condition_dims + 1, self.output_dims);
        }
        let grad_norm_tensor = tensor_l2_norm([gw1.clone(), gb1.clone(), gw2.clone(), gb2.clone()]);
        let grad_scale_tensor =
            gradient_scale_tensor(grad_norm_tensor.clone(), optimizer.grad_clip_norm);
        let diagnostics = if collect_diagnostics {
            let (pre_hidden, hidden) = diagnostic_cache.ok_or_else(|| {
                std::io::Error::other("sampled adapter diagnostics require a cached flow step")
            })?;
            Some(AdapterBankFlowOptimizerDiagnosticsReport {
                prediction_rms: tensor_rms(state, rows * self.output_dims, "flow prediction rms")?,
                velocity_rms: tensor_rms(target, rows * self.output_dims, "flow target rms")?,
                residual_rms: tensor_rms(diff, rows * self.output_dims, "flow residual rms")?,
                pre_hidden_rms: tensor_rms(
                    pre_hidden.clone(),
                    rows * self.hidden_dims,
                    "flow pre-hidden rms",
                )?,
                hidden_rms: tensor_rms(hidden, rows * self.hidden_dims, "flow hidden rms")?,
                hidden_zero_fraction: hidden_zero_fraction(pre_hidden)?,
                grad_w1_norm: tensor_l2_norm_single(gw1.clone(), "flow grad w1 norm")?,
                grad_b1_norm: tensor_l2_norm_single(gb1.clone(), "flow grad b1 norm")?,
                grad_w2_norm: tensor_l2_norm_single(gw2.clone(), "flow grad w2 norm")?,
                grad_b2_norm: tensor_l2_norm_single(gb2.clone(), "flow grad b2 norm")?,
            })
        } else {
            None
        };
        self.step = self.step.saturating_add(1);
        self.apply_adamw([gw1, gb1, gw2, gb2], optimizer, grad_scale_tensor.clone());
        self.detach_state();
        let grad_norm = finite_scalar(
            "Burn/WGPU sampled adapter-flow grad norm",
            grad_norm_tensor.into_scalar(),
        )?;
        let grad_scale = finite_scalar(
            "Burn/WGPU sampled adapter-flow grad scale",
            grad_scale_tensor.into_scalar(),
        )?;
        Ok(WgpuFlowTrainStepReport {
            grad_norm,
            grad_scale,
            diagnostics,
        })
    }

    fn sampled_adapter_loss_sum(
        &self,
        features: Tensor2,
        target: Tensor2,
        weights: Tensor2,
        rows: usize,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let state = self.sampled_adapter_forward(features, rows);
        let diff = state - target;
        let loss = diff
            .clone()
            .mul(diff)
            .mul(weights.expand([rows, self.output_dims]))
            .sum();
        finite_scalar(
            "Burn/WGPU sampled adapter-flow loss sum",
            loss.into_scalar(),
        )
    }

    fn sampled_adapter_forward(&self, features: Tensor2, rows: usize) -> Tensor2 {
        let mut state = Tensor::<WgpuBackend, 2>::zeros([rows, self.output_dims], &self.device);
        let dt = 1.0 / self.sample_steps.max(1) as f32;
        for step in 0..self.sample_steps.max(1) {
            let t = (step as f32 + 0.5) * dt;
            let input = self.sampled_flow_input(features.clone(), state.clone(), rows, t);
            let (_, _, velocity) = self.forward(input, rows);
            state = state + velocity.mul_scalar(dt);
        }
        state
    }

    fn sampled_adapter_forward_with_cache(
        &self,
        features: Tensor2,
        rows: usize,
    ) -> (Tensor2, Vec<(Tensor2, Tensor2, Tensor2)>) {
        let mut state = Tensor::<WgpuBackend, 2>::zeros([rows, self.output_dims], &self.device);
        let mut caches = Vec::with_capacity(self.sample_steps.max(1));
        let dt = 1.0 / self.sample_steps.max(1) as f32;
        for step in 0..self.sample_steps.max(1) {
            let t = (step as f32 + 0.5) * dt;
            let input = self.sampled_flow_input(features.clone(), state.clone(), rows, t);
            let (pre_hidden, hidden, velocity) = self.forward(input.clone(), rows);
            state = state + velocity.mul_scalar(dt);
            caches.push((input, pre_hidden, hidden));
        }
        (state, caches)
    }

    fn sampled_flow_input(
        &self,
        features: Tensor2,
        state: Tensor2,
        rows: usize,
        t: f32,
    ) -> Tensor2 {
        let time = Tensor::<WgpuBackend, 2>::full([rows, 1], t, &self.device);
        Tensor::cat(vec![features, time, state], 1)
    }

    fn feature_target_batch(
        &self,
        indices: Option<&[usize]>,
    ) -> (Tensor2, Tensor2, Tensor2, usize, f32) {
        let Some(indices) = indices else {
            return self.feature_target_range_tensors(
                &self.features,
                &self.target,
                &self.sample_weights,
                0,
                self.rows,
            );
        };
        if indices.len() >= self.rows {
            return self.feature_target_range_tensors(
                &self.features,
                &self.target,
                &self.sample_weights,
                0,
                self.rows,
            );
        }
        let mut feature_values = Vec::with_capacity(indices.len() * self.condition_dims);
        let mut target_values = Vec::with_capacity(indices.len() * self.output_dims);
        let mut weight_values = Vec::with_capacity(indices.len());
        let mut weight_sum = 0.0_f32;
        for &idx in indices {
            let feature_base = idx * self.condition_dims;
            let target_base = idx * self.output_dims;
            feature_values.extend_from_slice(
                &self.features[feature_base..feature_base + self.condition_dims],
            );
            target_values
                .extend_from_slice(&self.target[target_base..target_base + self.output_dims]);
            let weight = self.sample_weights[idx];
            weight_values.push(weight);
            weight_sum += weight;
        }
        (
            tensor2(
                feature_values,
                [indices.len(), self.condition_dims],
                &self.device,
            ),
            tensor2(
                target_values,
                [indices.len(), self.output_dims],
                &self.device,
            ),
            tensor2(weight_values, [indices.len(), 1], &self.device),
            indices.len(),
            weight_sum,
        )
    }

    fn feature_target_range_tensors(
        &self,
        features: &[f32],
        target: &[f32],
        sample_weights: &[f32],
        start: usize,
        end: usize,
    ) -> (Tensor2, Tensor2, Tensor2, usize, f32) {
        let rows = end - start;
        let feature_start = start * self.condition_dims;
        let feature_end = end * self.condition_dims;
        let target_start = start * self.output_dims;
        let target_end = end * self.output_dims;
        let weight_values = sample_weights[start..end].to_vec();
        let weight_sum = weight_values.iter().sum::<f32>();
        (
            tensor2(
                features[feature_start..feature_end].to_vec(),
                [rows, self.condition_dims],
                &self.device,
            ),
            tensor2(
                target[target_start..target_end].to_vec(),
                [rows, self.output_dims],
                &self.device,
            ),
            tensor2(weight_values, [rows, 1], &self.device),
            rows,
            weight_sum,
        )
    }

    fn batch(&self, indices: Option<&[usize]>, sample_seed: u64) -> (Tensor2, Tensor2, usize) {
        let Some(indices) = indices else {
            return self.row_range_tensors(&self.features, &self.target, 0, self.rows, sample_seed);
        };
        if indices.len() >= self.rows {
            return self.row_range_tensors(&self.features, &self.target, 0, self.rows, sample_seed);
        }
        let mut input_values = Vec::with_capacity(indices.len() * self.input_dims);
        let mut velocity_values = Vec::with_capacity(indices.len() * self.output_dims);
        for &idx in indices {
            self.push_flow_row(
                &self.features,
                &self.target,
                idx,
                sample_seed,
                &mut input_values,
                &mut velocity_values,
            );
        }
        (
            tensor2(input_values, [indices.len(), self.input_dims], &self.device),
            tensor2(
                velocity_values,
                [indices.len(), self.output_dims],
                &self.device,
            ),
            indices.len(),
        )
    }

    fn row_range_tensors(
        &self,
        features: &[f32],
        target: &[f32],
        start: usize,
        end: usize,
        sample_seed: u64,
    ) -> (Tensor2, Tensor2, usize) {
        let rows = end - start;
        let mut input_values = Vec::with_capacity(rows * self.input_dims);
        let mut velocity_values = Vec::with_capacity(rows * self.output_dims);
        for idx in start..end {
            self.push_flow_row(
                features,
                target,
                idx,
                sample_seed,
                &mut input_values,
                &mut velocity_values,
            );
        }
        (
            tensor2(input_values, [rows, self.input_dims], &self.device),
            tensor2(velocity_values, [rows, self.output_dims], &self.device),
            rows,
        )
    }

    fn push_flow_row(
        &self,
        features: &[f32],
        target: &[f32],
        idx: usize,
        sample_seed: u64,
        input_values: &mut Vec<f32>,
        velocity_values: &mut Vec<f32>,
    ) {
        append_rectified_flow_training_row(
            features,
            target,
            self.condition_dims,
            self.output_dims,
            idx,
            self.source_scale,
            self.sample_seed,
            sample_seed,
            input_values,
            velocity_values,
        );
    }

    fn forward(&self, inputs: Tensor2, rows: usize) -> (Tensor2, Tensor2, Tensor2) {
        let pre_hidden = inputs.matmul(self.w1.clone().transpose())
            + self.b1.clone().expand([rows, self.hidden_dims]);
        let hidden = flow_hidden_activation(pre_hidden.clone(), self.hidden_activation);
        let output = hidden.clone().matmul(self.w2.clone().transpose())
            + self.b2.clone().expand([rows, self.output_dims]);
        (pre_hidden, hidden, output)
    }

    fn apply_adamw(&mut self, grads: [Tensor2; 4], cfg: AdamWConfig, grad_scale: Tensor2) {
        let [gw1, gb1, gw2, gb2] = grads;
        let bias_correction1 = 1.0 - cfg.beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - cfg.beta2.powi(self.step as i32);
        (self.w1, self.w1_m, self.w1_v) = apply_adamw_tensor(
            self.w1.clone(),
            gw1,
            self.w1_m.clone(),
            self.w1_v.clone(),
            cfg,
            grad_scale.clone(),
            bias_correction1,
            bias_correction2,
        );
        (self.b1, self.b1_m, self.b1_v) = apply_adamw_tensor(
            self.b1.clone(),
            gb1,
            self.b1_m.clone(),
            self.b1_v.clone(),
            cfg,
            grad_scale.clone(),
            bias_correction1,
            bias_correction2,
        );
        (self.w2, self.w2_m, self.w2_v) = apply_adamw_tensor(
            self.w2.clone(),
            gw2,
            self.w2_m.clone(),
            self.w2_v.clone(),
            cfg,
            grad_scale.clone(),
            bias_correction1,
            bias_correction2,
        );
        (self.b2, self.b2_m, self.b2_v) = apply_adamw_tensor(
            self.b2.clone(),
            gb2,
            self.b2_m.clone(),
            self.b2_v.clone(),
            cfg,
            grad_scale,
            bias_correction1,
            bias_correction2,
        );
    }

    fn detach_state(&mut self) {
        self.w1 = self.w1.clone().detach();
        self.b1 = self.b1.clone().detach();
        self.w2 = self.w2.clone().detach();
        self.b2 = self.b2.clone().detach();
        self.w1_m = self.w1_m.clone().detach();
        self.w1_v = self.w1_v.clone().detach();
        self.b1_m = self.b1_m.clone().detach();
        self.b1_v = self.b1_v.clone().detach();
        self.w2_m = self.w2_m.clone().detach();
        self.w2_v = self.w2_v.clone().detach();
        self.b2_m = self.b2_m.clone().detach();
        self.b2_v = self.b2_v.clone().detach();
    }

    fn to_flow_weights(&self) -> Result<crate::HyperNpa2dFlowWeights, Box<dyn std::error::Error>> {
        let weights = crate::HyperNpa2dFlowWeights {
            w1: tensor_vec(self.w1.clone())?,
            b1: tensor_vec(self.b1.clone())?,
            w2: tensor_vec(self.w2.clone())?,
            b2: tensor_vec(self.b2.clone())?,
        };
        if weights.w1.len() != self.hidden_dims * self.input_dims
            || weights.b1.len() != self.hidden_dims
            || weights.w2.len() != self.output_dims * self.hidden_dims
            || weights.b2.len() != self.output_dims
        {
            return Err(
                std::io::Error::other("Burn/WGPU flow weight readback shape mismatch").into(),
            );
        }
        Ok(weights)
    }
}

fn seeded_flow_weights(
    input_dims: usize,
    hidden_dims: usize,
    output_dims: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(seed ^ 0x51ed_f10a_2d0d_f00d);
    let w1_scale = (2.0 / input_dims.max(1) as f32).sqrt();
    let w2_scale = (2.0 / hidden_dims.max(1) as f32).sqrt();
    let w1 = (0..hidden_dims * input_dims)
        .map(|_| rng.random_range(-w1_scale..=w1_scale))
        .collect::<Vec<_>>();
    let b1 = vec![0.0; hidden_dims];
    let w2 = (0..output_dims * hidden_dims)
        .map(|_| rng.random_range(-w2_scale..=w2_scale))
        .collect::<Vec<_>>();
    let b2 = vec![0.0; output_dims];
    (w1, b1, w2, b2)
}

fn validate_wgpu_tensor_allocation(
    label: &str,
    rows: usize,
    cols: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let values = rows.checked_mul(cols).ok_or_else(|| {
        std::io::Error::other(format!("{label} tensor shape {rows}x{cols} overflows"))
    })?;
    let bytes = values
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            std::io::Error::other(format!("{label} tensor byte size {rows}x{cols} overflows"))
        })?;
    if bytes > MAX_SAFE_WGPU_TENSOR_BYTES {
        return Err(std::io::Error::other(format!(
            "{label} tensor would allocate {:.2} MiB on WGPU; reduce hidden size/token grid or use a token-efficient condition encoder",
            bytes as f64 / (1024.0 * 1024.0)
        ))
        .into());
    }
    Ok(())
}

fn tensor2(values: Vec<f32>, shape: [usize; 2], device: &WgpuDevice) -> Tensor2 {
    Tensor::<WgpuBackend, 2>::from_data(TensorData::new(values, shape), device)
}

fn example_buffers(
    hyper: &HyperNpa2d,
    examples: &[AdapterBankConditionedExample],
    input_dims: usize,
    output_dims: usize,
) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    let mut features = Vec::with_capacity(examples.len() * input_dims);
    let mut target = Vec::with_capacity(examples.len() * output_dims);
    for example in examples {
        let input = hyper.condition_input_vector(&example.condition)?;
        if input.len() != input_dims || example.target_vector.len() != output_dims {
            return Err(std::io::Error::other("adapter-bank tensor shape mismatch").into());
        }
        features.extend_from_slice(&input);
        target.extend_from_slice(&example.target_vector);
    }
    Ok((features, target))
}

fn example_sample_weights(examples: &[AdapterBankConditionedExample]) -> Vec<f32> {
    examples
        .iter()
        .map(|example| example.sample_weight)
        .collect()
}

fn tensor_vec(tensor: Tensor2) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    tensor.into_data().to_vec::<f32>().map_err(|err| {
        std::io::Error::other(format!("Burn/WGPU tensor readback failed: {err}")).into()
    })
}

fn tensor_l2_norm(tensors: [Tensor2; 4]) -> Tensor1 {
    let [w1, b1, w2, b2] = tensors;
    let total = w1.clone().mul(w1).sum()
        + b1.clone().mul(b1).sum()
        + w2.clone().mul(w2).sum()
        + b2.clone().mul(b2).sum();
    total.sqrt()
}

fn flow_hidden_activation(pre_hidden: Tensor2, activation: HyperNpa2dFlowActivation) -> Tensor2 {
    match activation {
        HyperNpa2dFlowActivation::Relu => relu(pre_hidden),
        HyperNpa2dFlowActivation::LeakyRelu => {
            relu(pre_hidden.clone()) - relu(pre_hidden.mul_scalar(-1.0)).mul_scalar(0.01)
        }
    }
}

fn flow_hidden_backward(
    d_hidden: Tensor2,
    pre_hidden: Tensor2,
    activation: HyperNpa2dFlowActivation,
) -> Tensor2 {
    match activation {
        HyperNpa2dFlowActivation::Relu => d_hidden.mask_fill(pre_hidden.lower_equal_elem(0.0), 0.0),
        HyperNpa2dFlowActivation::LeakyRelu => {
            let positive = d_hidden
                .clone()
                .mask_fill(pre_hidden.clone().lower_equal_elem(0.0), 0.0);
            let negative = d_hidden
                .mul_scalar(0.01)
                .mask_fill(pre_hidden.greater_elem(0.0), 0.0);
            positive + negative
        }
    }
}

fn tensor_l2_norm_single(tensor: Tensor2, name: &str) -> Result<f32, Box<dyn std::error::Error>> {
    let norm = tensor.clone().mul(tensor).sum().sqrt();
    finite_scalar(name, norm.into_scalar())
}

fn tensor_rms(
    tensor: Tensor2,
    values: usize,
    name: &str,
) -> Result<f32, Box<dyn std::error::Error>> {
    if values == 0 {
        return Err(std::io::Error::other(format!("{name} requires non-empty tensor")).into());
    }
    let sum_sq = tensor.clone().mul(tensor).sum();
    finite_scalar(name, (sum_sq.into_scalar() / values as f32).sqrt())
}

fn hidden_zero_fraction(tensor: Tensor2) -> Result<f32, Box<dyn std::error::Error>> {
    let values = tensor_vec(tensor)?;
    if values.is_empty() {
        return Err(std::io::Error::other("flow hidden zero fraction requires values").into());
    }
    let zeros = values.iter().filter(|value| **value <= 0.0).count();
    Ok(zeros as f32 / values.len() as f32)
}

fn gradient_scale_tensor(grad_norm: Tensor1, grad_clip_norm: f32) -> Tensor2 {
    let scale = if grad_clip_norm > 0.0 {
        grad_norm.recip().mul_scalar(grad_clip_norm).clamp_max(1.0)
    } else {
        grad_norm.mul_scalar(0.0).add_scalar(1.0)
    };
    scale.reshape([1, 1])
}

#[allow(clippy::too_many_arguments)]
fn apply_adamw_tensor(
    param: Tensor2,
    grad: Tensor2,
    moment: Tensor2,
    velocity: Tensor2,
    cfg: AdamWConfig,
    grad_scale: Tensor2,
    bias_correction1: f32,
    bias_correction2: f32,
) -> (Tensor2, Tensor2, Tensor2) {
    let param = if cfg.weight_decay > 0.0 {
        param.mul_scalar(1.0 - cfg.learning_rate * cfg.weight_decay)
    } else {
        param
    };
    let grad = grad.mul(grad_scale);
    let moment = moment.mul_scalar(cfg.beta1) + grad.clone().mul_scalar(1.0 - cfg.beta1);
    let velocity =
        velocity.mul_scalar(cfg.beta2) + grad.clone().mul(grad).mul_scalar(1.0 - cfg.beta2);
    let moment_hat = moment
        .clone()
        .div_scalar(bias_correction1.max(f32::MIN_POSITIVE));
    let velocity_hat = velocity
        .clone()
        .div_scalar(bias_correction2.max(f32::MIN_POSITIVE));
    let param = param
        - moment_hat
            .div(velocity_hat.sqrt().add_scalar(cfg.epsilon))
            .mul_scalar(cfg.learning_rate);
    (param, moment, velocity)
}

fn finite_scalar(name: &str, value: f32) -> Result<f32, Box<dyn std::error::Error>> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(std::io::Error::other(format!("{name} is not finite")).into())
    }
}
