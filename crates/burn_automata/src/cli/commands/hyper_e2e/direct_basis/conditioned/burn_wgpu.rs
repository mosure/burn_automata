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
        let d_output = diff.mul_scalar(2.0 / (rows * self.output_dims) as f32);
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
        let d_pre_hidden = d_hidden.mask_fill(pre_hidden.lower_equal_elem(0.0), 0.0);
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
    condition_dims: usize,
    input_dims: usize,
    hidden_dims: usize,
    output_dims: usize,
    source_scale: f32,
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
    memory.push(check_process_memory_budget(
        "burn-wgpu adapter-flow initial loss evaluated",
        config.system_memory_budget_gb,
    )?);
    let mut final_loss = initial_loss;
    let mut final_validation_loss = initial_validation_loss;
    let mut best_loss = initial_validation_loss.unwrap_or(initial_loss);
    let mut best_validation_loss = initial_validation_loss;
    let mut best_step = 0usize;
    let mut best_weights = trainer.to_flow_weights()?;
    let mut history = Vec::new();
    let mut rng = StdRng::seed_from_u64(config.seed);
    let batch_size = normalized_batch_size(config.example_batch_size, examples.len());
    for step in 1..=config.steps {
        let indices = sample_indices(examples.len(), batch_size, &mut rng);
        let step_started = Instant::now();
        let (grad_norm, grad_scale) =
            trainer.train_step(&indices, config.optimizer, config.seed)?;
        let step_elapsed = step_started.elapsed();
        if step == config.steps || step.is_multiple_of(config.report_interval.max(1)) {
            final_loss = trainer.loss(None, config.seed)?;
            final_validation_loss = trainer.validation_loss(config.seed ^ 0x5eed_5eed)?;
            let selection_loss = final_validation_loss.unwrap_or(final_loss);
            if selection_loss < best_loss {
                best_loss = selection_loss;
                best_validation_loss = final_validation_loss;
                best_step = step;
                best_weights = trainer.to_flow_weights()?;
            }
            let memory_snapshot = check_process_memory_budget(
                format!("burn-wgpu adapter-flow step {step}"),
                config.system_memory_budget_gb,
            )?;
            eprintln!(
                "adapter-flow step {step}/{} loss={:.6e} val={} grad_norm={:.6e} values/s={:.3e} rss={}",
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
            });
        }
    }
    let final_selection_loss = final_validation_loss.unwrap_or(final_loss);
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
        },
        weights: final_weights,
    })?;
    let final_trainer = WgpuAdapterFlowTrainer::new(hyper, examples, validation_examples, config)?;
    final_loss = final_trainer.loss(None, config.seed)?;
    final_validation_loss = final_trainer.validation_loss(config.seed ^ 0x5eed_5eed)?;
    memory.push(check_process_memory_budget(
        "burn-wgpu adapter-flow final loss evaluated",
        config.system_memory_budget_gb,
    )?);
    Ok(AdapterBankTrainingPhaseReport {
        backend: "burn_wgpu_rectified_flow_lora_vector".to_string(),
        device: "wgpu-default".to_string(),
        selection_metric: if initial_validation_loss.is_some() {
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
    })
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
        let validation_rows = validation_examples.map_or(0, |examples| examples.len());
        let (validation_features, validation_target) = if let Some(validation_examples) =
            validation_examples.filter(|examples| !examples.is_empty())
        {
            let (features, target) =
                example_buffers(hyper, validation_examples, condition_dims, output_dims)?;
            (Some(features), Some(target))
        } else {
            (None, None)
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
            seeded_flow_weights(input_dims, hidden_dims, output_dims, config.seed)
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
            validation_features,
            validation_target,
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
            source_scale: config.flow.source_scale,
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
    ) -> Result<(f32, f32), Box<dyn std::error::Error>> {
        let step_seed = sample_seed ^ ((self.step as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let (inputs, velocity, rows) = self.batch(Some(indices), step_seed);
        let (pre_hidden, hidden, output) = self.forward(inputs.clone(), rows);
        let diff = output - velocity;
        let d_output = diff.mul_scalar(2.0 / (rows * self.output_dims) as f32);
        let gb2 = d_output.clone().sum_dim(0);
        let gw2 = d_output.clone().transpose().matmul(hidden.clone());
        let d_hidden = d_output.matmul(self.w2.clone());
        let d_pre_hidden = d_hidden.mask_fill(pre_hidden.lower_equal_elem(0.0), 0.0);
        let gb1 = d_pre_hidden.clone().sum_dim(0);
        let gw1 = d_pre_hidden.transpose().matmul(inputs);
        let grad_norm_tensor = tensor_l2_norm([gw1.clone(), gb1.clone(), gw2.clone(), gb2.clone()]);
        let grad_scale_tensor =
            gradient_scale_tensor(grad_norm_tensor.clone(), optimizer.grad_clip_norm);
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
        Ok((grad_norm, grad_scale))
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
        let feature_start = idx * self.condition_dims;
        let target_start = idx * self.output_dims;
        let condition = &features[feature_start..feature_start + self.condition_dims];
        let target = &target[target_start..target_start + self.output_dims];
        let mut rng = StdRng::seed_from_u64(
            self.sample_seed ^ sample_seed ^ ((idx as u64 + 1).wrapping_mul(0xd1b5_4a32_d192_ed03)),
        );
        let t = rng.random_range(0.0..=1.0);
        input_values.extend_from_slice(condition);
        input_values.push(t);
        for &target_value in target {
            let source = rng.random_range(-self.source_scale..=self.source_scale);
            let state = source.mul_add(1.0 - t, target_value * t);
            input_values.push(state);
            velocity_values.push(target_value - source);
        }
    }

    fn forward(&self, inputs: Tensor2, rows: usize) -> (Tensor2, Tensor2, Tensor2) {
        let pre_hidden = inputs.matmul(self.w1.clone().transpose())
            + self.b1.clone().expand([rows, self.hidden_dims]);
        let hidden = relu(pre_hidden.clone());
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
