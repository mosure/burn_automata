use burn::{
    backend::Wgpu,
    tensor::{Device, Tensor, TensorData, activation::relu},
};

use crate::{
    AdamWConfig, AutomataError, AutomataResult, NpaModel, NpaWeights, SgdConfig, SupervisedBatch,
    SupervisedOptimizerConfig, SupervisedStepReport, TrainingHistoryEntry, TrainingRunConfig,
    TrainingRunReport,
};

type WgpuBackend = Wgpu<f32>;
type WgpuDevice = Device<WgpuBackend>;
type Tensor2 = Tensor<WgpuBackend, 2>;

pub fn run_supervised_training_wgpu(
    model: &mut NpaModel,
    batch: &SupervisedBatch,
    cfg: TrainingRunConfig,
    optimizer: SupervisedOptimizerConfig,
) -> AutomataResult<TrainingRunReport> {
    run_weighted_supervised_training_wgpu(model, batch, cfg, optimizer, None)
}

pub fn run_weighted_supervised_training_wgpu(
    model: &mut NpaModel,
    batch: &SupervisedBatch,
    cfg: TrainingRunConfig,
    optimizer: SupervisedOptimizerConfig,
    output_weights: Option<&[f32]>,
) -> AutomataResult<TrainingRunReport> {
    run_weighted_supervised_training_wgpu_with_observer(
        model,
        batch,
        cfg,
        optimizer,
        output_weights,
        None,
    )
}

pub trait WgpuSupervisedTrainingObserver {
    fn should_stop(&self) -> bool {
        false
    }

    fn on_progress(
        &mut self,
        step: usize,
        total_steps: usize,
        entry: &TrainingHistoryEntry,
        model: &NpaModel,
    );
}

pub fn run_weighted_supervised_training_wgpu_with_observer(
    model: &mut NpaModel,
    batch: &SupervisedBatch,
    cfg: TrainingRunConfig,
    optimizer: SupervisedOptimizerConfig,
    output_weights: Option<&[f32]>,
    observer: Option<&mut dyn WgpuSupervisedTrainingObserver>,
) -> AutomataResult<TrainingRunReport> {
    validate_gpu_optimizer(optimizer)?;
    validate_gpu_sgd_config(cfg.sgd)?;
    let mut trainer = WgpuSupervisedTrainingSession::new(model, batch, output_weights)?;
    trainer.train_into_model(model, cfg, optimizer, true, observer)
}

pub(crate) struct WgpuSupervisedTrainingSession {
    features: Tensor2,
    target: Tensor2,
    ones_rows: Tensor2,
    output_weights: Tensor2,
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
    input_dims: usize,
    hidden_dims: usize,
    output_dims: usize,
    step: usize,
}

impl WgpuSupervisedTrainingSession {
    pub(crate) fn new(
        model: &NpaModel,
        batch: &SupervisedBatch,
        output_weights: Option<&[f32]>,
    ) -> AutomataResult<Self> {
        model.validate()?;
        let input_dims = model.config.perception_dims();
        let output_dims = model.config.update_dims();
        let rows = batch.features.len() / input_dims;
        if rows == 0 || batch.features.len() != rows * input_dims {
            return Err(AutomataError::InvalidArgument(
                "features do not form whole perception rows".to_string(),
            ));
        }
        if batch.target_update.len() != rows * output_dims {
            return Err(AutomataError::InvalidArgument(format!(
                "target_update len {} != {}",
                batch.target_update.len(),
                rows * output_dims
            )));
        }
        ensure_finite("features", &batch.features)?;
        ensure_finite("target_update", &batch.target_update)?;
        let output_weights =
            output_weights.map_or_else(|| vec![1.0; output_dims], |weights| weights.to_vec());
        if output_weights.len() != output_dims
            || output_weights
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || output_weights.iter().all(|value| *value == 0.0)
        {
            return Err(AutomataError::InvalidArgument(format!(
                "output weights must contain {output_dims} finite non-negative values with at least one positive entry"
            )));
        }

        let device = Default::default();
        let hidden_dims = model.config.hidden_dims;
        let features = tensor2(batch.features.clone(), [rows, input_dims], &device);
        let target = tensor2(batch.target_update.clone(), [rows, output_dims], &device);
        let output_weights = tensor2(output_weights, [1, output_dims], &device);
        let ones_rows = Tensor::<WgpuBackend, 2>::ones([rows, 1], &device);
        let w1 = tensor2(model.weights.w1.clone(), [hidden_dims, input_dims], &device);
        let b1 = tensor2(model.weights.b1.clone(), [1, hidden_dims], &device);
        let w2 = tensor2(
            model.weights.w2.clone(),
            [output_dims, hidden_dims],
            &device,
        );
        let b2 = tensor2(model.weights.b2.clone(), [1, output_dims], &device);
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
            ones_rows,
            output_weights,
            w1,
            b1,
            w2,
            b2,
            rows,
            input_dims,
            hidden_dims,
            output_dims,
            step: 0,
        })
    }

    pub(crate) fn replace_batch(&mut self, batch: &SupervisedBatch) -> AutomataResult<()> {
        let rows = validate_batch(batch, self.input_dims, self.output_dims)?;
        let device = Default::default();
        self.features = tensor2(batch.features.clone(), [rows, self.input_dims], &device);
        self.target = tensor2(
            batch.target_update.clone(),
            [rows, self.output_dims],
            &device,
        );
        self.ones_rows = Tensor::<WgpuBackend, 2>::ones([rows, 1], &device);
        self.rows = rows;
        Ok(())
    }

    pub(crate) fn train_into_model(
        &mut self,
        model: &mut NpaModel,
        cfg: TrainingRunConfig,
        optimizer: SupervisedOptimizerConfig,
        restore_best: bool,
        mut observer: Option<&mut dyn WgpuSupervisedTrainingObserver>,
    ) -> AutomataResult<TrainingRunReport> {
        validate_gpu_optimizer(optimizer)?;
        validate_gpu_sgd_config(cfg.sgd)?;
        let rows = self.rows;
        let initial_loss = self.loss()?;
        let mut final_loss = initial_loss;
        let mut best_loss = initial_loss;
        let mut best_weights = restore_best.then(|| model.weights.clone());
        let report_interval = cfg.report_interval.max(1);
        let mut history = Vec::new();
        let mut completed_steps = 0;

        for step in 1..=cfg.steps {
            if observer
                .as_deref()
                .is_some_and(WgpuSupervisedTrainingObserver::should_stop)
            {
                break;
            }
            let capture_metrics = step == cfg.steps || step.is_multiple_of(report_interval);
            let report = self.train_step(optimizer, capture_metrics)?;
            completed_steps = step;
            let Some(step_report) = report else {
                continue;
            };
            final_loss = step_report.loss;
            let current_weights = if restore_best || observer.is_some() {
                Some(self.to_weights()?)
            } else {
                None
            };
            if final_loss < best_loss {
                best_loss = final_loss;
                if restore_best {
                    best_weights = current_weights.clone();
                }
            }
            let entry = TrainingHistoryEntry {
                step,
                loss: final_loss,
                grad_norm: step_report.grad_norm,
                grad_scale: step_report.grad_scale,
            };
            history.push(entry);
            if let Some(observer) = observer.as_deref_mut() {
                model.weights = current_weights
                    .clone()
                    .expect("observer progress captures weights");
                observer.on_progress(step, cfg.steps, &entry, model);
            }
        }

        let current_weights = self.to_weights()?;
        model.weights = if restore_best && best_loss < final_loss {
            final_loss = best_loss;
            best_weights.expect("restored best weights")
        } else {
            current_weights
        };

        Ok(TrainingRunReport {
            steps: completed_steps,
            rows,
            initial_loss,
            final_loss,
            best_loss,
            history,
        })
    }

    fn loss(&self) -> AutomataResult<f32> {
        let output = self.forward().2;
        let diff = output - self.target.clone();
        let loss = diff
            .clone()
            .mul(diff)
            .mul(self.output_weights.clone())
            .sum()
            .div_scalar(self.rows as f32);
        finite_scalar("gpu supervised loss", loss.into_scalar())
    }

    fn train_step(
        &mut self,
        optimizer: SupervisedOptimizerConfig,
        capture_metrics: bool,
    ) -> AutomataResult<Option<SupervisedStepReport>> {
        let (pre_hidden, hidden, output) = self.forward();
        let diff = output - self.target.clone();
        let d_out = diff
            .mul(self.output_weights.clone())
            .mul_scalar(2.0 / self.rows as f32);
        let gb2 = d_out.clone().sum_dim(0);
        let gw2 = d_out.clone().transpose().matmul(hidden.clone());
        let d_hidden = d_out.matmul(self.w2.clone());
        let d_pre = d_hidden.mask_fill(pre_hidden.lower_equal_elem(0.0), 0.0);
        let gb1 = d_pre.clone().sum_dim(0);
        let gw1 = d_pre.transpose().matmul(self.features.clone());
        let grad_norm_tensor = tensor_l2_norm([gw1.clone(), gb1.clone(), gw2.clone(), gb2.clone()]);

        let grad_clip_norm = match optimizer {
            SupervisedOptimizerConfig::Sgd(cfg) => cfg.grad_clip_norm,
            SupervisedOptimizerConfig::AdamW(cfg) => cfg.grad_clip_norm,
        };
        let grad_scale_tensor = gradient_scale_tensor(grad_norm_tensor.clone(), grad_clip_norm);
        self.step = self.step.saturating_add(1);
        match optimizer {
            SupervisedOptimizerConfig::Sgd(cfg) => {
                self.apply_sgd([gw1, gb1, gw2, gb2], cfg, grad_scale_tensor.clone())
            }
            SupervisedOptimizerConfig::AdamW(cfg) => {
                self.apply_adamw([gw1, gb1, gw2, gb2], cfg, grad_scale_tensor.clone())
            }
        }

        if !capture_metrics {
            return Ok(None);
        }
        let loss = self.loss()?;
        let grad_norm = finite_scalar("gpu supervised grad norm", grad_norm_tensor.into_scalar())?;
        let grad_scale =
            finite_scalar("gpu supervised grad scale", grad_scale_tensor.into_scalar())?;
        Ok(Some(SupervisedStepReport {
            loss,
            rows: self.rows,
            grad_norm,
            grad_scale,
            clipped: grad_scale < 1.0,
        }))
    }

    fn forward(&self) -> (Tensor2, Tensor2, Tensor2) {
        let pre_hidden = self.features.clone().matmul(self.w1.clone().transpose())
            + self.ones_rows.clone().matmul(self.b1.clone());
        let hidden = relu(pre_hidden.clone());
        let output = hidden.clone().matmul(self.w2.clone().transpose())
            + self.ones_rows.clone().matmul(self.b2.clone());
        (pre_hidden, hidden, output)
    }

    fn apply_sgd(&mut self, grads: [Tensor2; 4], cfg: SgdConfig, grad_scale: Tensor2) {
        let [gw1, gb1, gw2, gb2] = grads;
        self.w1 = apply_sgd_tensor(self.w1.clone(), gw1, cfg, grad_scale.clone());
        self.b1 = apply_sgd_tensor(self.b1.clone(), gb1, cfg, grad_scale.clone());
        self.w2 = apply_sgd_tensor(self.w2.clone(), gw2, cfg, grad_scale.clone());
        self.b2 = apply_sgd_tensor(self.b2.clone(), gb2, cfg, grad_scale);
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

    fn to_weights(&self) -> AutomataResult<NpaWeights> {
        let weights = NpaWeights {
            w1: tensor_vec(self.w1.clone())?,
            b1: tensor_vec(self.b1.clone())?,
            w2: tensor_vec(self.w2.clone())?,
            b2: tensor_vec(self.b2.clone())?,
        };
        if weights.w1.len() != self.hidden_dims * self.input_dims {
            return Err(AutomataError::InvalidArgument(
                "gpu w1 readback size mismatch".to_string(),
            ));
        }
        if weights.b1.len() != self.hidden_dims
            || weights.w2.len() != self.output_dims * self.hidden_dims
            || weights.b2.len() != self.output_dims
        {
            return Err(AutomataError::InvalidArgument(
                "gpu weight readback size mismatch".to_string(),
            ));
        }
        Ok(weights)
    }
}

fn validate_batch(
    batch: &SupervisedBatch,
    input_dims: usize,
    output_dims: usize,
) -> AutomataResult<usize> {
    let rows = batch.features.len() / input_dims;
    if rows == 0 || batch.features.len() != rows * input_dims {
        return Err(AutomataError::InvalidArgument(
            "features do not form whole perception rows".to_string(),
        ));
    }
    if batch.target_update.len() != rows * output_dims {
        return Err(AutomataError::InvalidArgument(format!(
            "target_update len {} != {}",
            batch.target_update.len(),
            rows * output_dims
        )));
    }
    ensure_finite("features", &batch.features)?;
    ensure_finite("target_update", &batch.target_update)?;
    Ok(rows)
}

fn tensor2(values: Vec<f32>, shape: [usize; 2], device: &WgpuDevice) -> Tensor2 {
    Tensor::<WgpuBackend, 2>::from_data(TensorData::new(values, shape), device)
}

fn tensor_vec(tensor: Tensor2) -> AutomataResult<Vec<f32>> {
    tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| AutomataError::InvalidArgument(format!("gpu tensor readback failed: {err}")))
}

fn apply_sgd_tensor(param: Tensor2, grad: Tensor2, cfg: SgdConfig, grad_scale: Tensor2) -> Tensor2 {
    let update = grad.mul(grad_scale) + param.clone().mul_scalar(cfg.weight_decay);
    param - update.mul_scalar(cfg.learning_rate)
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

fn tensor_l2_norm(tensors: [Tensor2; 4]) -> Tensor<WgpuBackend, 1> {
    let [w1, b1, w2, b2] = tensors;
    let total = w1.clone().mul(w1).sum()
        + b1.clone().mul(b1).sum()
        + w2.clone().mul(w2).sum()
        + b2.clone().mul(b2).sum();
    total.sqrt()
}

fn gradient_scale_tensor(grad_norm: Tensor<WgpuBackend, 1>, grad_clip_norm: f32) -> Tensor2 {
    let scale = if grad_clip_norm > 0.0 {
        grad_norm.recip().mul_scalar(grad_clip_norm).clamp_max(1.0)
    } else {
        grad_norm.mul_scalar(0.0).add_scalar(1.0)
    };
    scale.reshape([1, 1])
}

fn validate_gpu_optimizer(cfg: SupervisedOptimizerConfig) -> AutomataResult<()> {
    match cfg {
        SupervisedOptimizerConfig::Sgd(cfg) => validate_gpu_sgd_config(cfg),
        SupervisedOptimizerConfig::AdamW(cfg) => validate_gpu_adamw_config(cfg),
    }
}

fn validate_gpu_sgd_config(cfg: SgdConfig) -> AutomataResult<()> {
    if !cfg.learning_rate.is_finite() || cfg.learning_rate <= 0.0 {
        return Err(AutomataError::InvalidArgument(
            "learning_rate must be finite and positive".to_string(),
        ));
    }
    if !cfg.weight_decay.is_finite() || cfg.weight_decay < 0.0 {
        return Err(AutomataError::InvalidArgument(
            "weight_decay must be finite and non-negative".to_string(),
        ));
    }
    if !cfg.grad_clip_norm.is_finite() || cfg.grad_clip_norm < 0.0 {
        return Err(AutomataError::InvalidArgument(
            "grad_clip_norm must be finite and non-negative".to_string(),
        ));
    }
    Ok(())
}

fn validate_gpu_adamw_config(cfg: AdamWConfig) -> AutomataResult<()> {
    validate_gpu_sgd_config(SgdConfig {
        learning_rate: cfg.learning_rate,
        weight_decay: cfg.weight_decay,
        grad_clip_norm: cfg.grad_clip_norm,
    })?;
    if !cfg.beta1.is_finite() || !(0.0..1.0).contains(&cfg.beta1) {
        return Err(AutomataError::InvalidArgument(
            "adam beta1 must be finite and in [0, 1)".to_string(),
        ));
    }
    if !cfg.beta2.is_finite() || !(0.0..1.0).contains(&cfg.beta2) {
        return Err(AutomataError::InvalidArgument(
            "adam beta2 must be finite and in [0, 1)".to_string(),
        ));
    }
    if !cfg.epsilon.is_finite() || cfg.epsilon <= 0.0 {
        return Err(AutomataError::InvalidArgument(
            "adam epsilon must be finite and positive".to_string(),
        ));
    }
    Ok(())
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

fn ensure_finite(name: &str, values: &[f32]) -> AutomataResult<()> {
    if values.iter().all(|value| value.is_finite()) {
        return Ok(());
    }
    Err(AutomataError::InvalidArgument(format!(
        "{name} contain non-finite values"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NpaConfig, NpaWeights, run_supervised_training_with_optimizer, supervised_loss};

    #[test]
    fn wgpu_supervised_step_reduces_simple_loss() {
        let config = NpaConfig {
            hidden_dims: 4,
            ..NpaConfig::growing_2d()
        };
        let mut model = NpaModel {
            config: config.clone(),
            weights: NpaWeights::seeded(&config, 7),
        };
        let rows = 16;
        let batch = SupervisedBatch {
            features: vec![0.2; rows * config.perception_dims()],
            target_update: vec![0.0; rows * config.update_dims()],
        };
        let before = supervised_loss(&model, &batch).unwrap();

        let report = run_supervised_training_wgpu(
            &mut model,
            &batch,
            TrainingRunConfig {
                steps: 8,
                report_interval: 8,
                sgd: SgdConfig::default(),
            },
            SupervisedOptimizerConfig::AdamW(AdamWConfig {
                learning_rate: 1.0e-2,
                grad_clip_norm: 10.0,
                ..AdamWConfig::default()
            }),
        )
        .unwrap();

        assert!(report.final_loss < before);
    }

    #[test]
    fn wgpu_supervised_training_matches_cpu_loss() {
        let config = NpaConfig {
            hidden_dims: 8,
            ..NpaConfig::growing_2d()
        };
        let initial = NpaModel {
            config: config.clone(),
            weights: NpaWeights::seeded(&config, 11),
        };
        let rows = 24;
        let batch = SupervisedBatch {
            features: (0..rows * config.perception_dims())
                .map(|idx| (idx as f32 * 0.013).sin() * 0.25)
                .collect(),
            target_update: (0..rows * config.update_dims())
                .map(|idx| (idx as f32 * 0.007).cos() * 0.05)
                .collect(),
        };
        let train_cfg = TrainingRunConfig {
            steps: 6,
            report_interval: 6,
            sgd: SgdConfig::default(),
        };
        let optimizer = SupervisedOptimizerConfig::AdamW(AdamWConfig {
            learning_rate: 1.0e-3,
            grad_clip_norm: 1.0,
            ..AdamWConfig::default()
        });
        let mut cpu_model = initial.clone();
        let mut gpu_model = initial;

        let cpu_report =
            run_supervised_training_with_optimizer(&mut cpu_model, &batch, train_cfg, optimizer)
                .unwrap();
        let gpu_report =
            run_supervised_training_wgpu(&mut gpu_model, &batch, train_cfg, optimizer).unwrap();

        assert!(
            (cpu_report.final_loss - gpu_report.final_loss).abs() < 1.0e-3,
            "cpu final loss {} != gpu final loss {}",
            cpu_report.final_loss,
            gpu_report.final_loss
        );
    }

    #[test]
    fn wgpu_supervised_session_reuses_optimizer_across_batch_replacement() {
        let config = NpaConfig {
            hidden_dims: 8,
            ..NpaConfig::growing_2d()
        };
        let mut model = NpaModel {
            config: config.clone(),
            weights: NpaWeights::seeded(&config, 19),
        };
        let batch = |offset: f32| SupervisedBatch {
            features: (0..32 * config.perception_dims())
                .map(|idx| (idx as f32 * 0.011).sin() * 0.2 + offset)
                .collect(),
            target_update: vec![0.0; 32 * config.update_dims()],
        };
        let first = batch(0.0);
        let second = batch(0.03);
        let mut session = WgpuSupervisedTrainingSession::new(&model, &first, None).unwrap();
        let optimizer = SupervisedOptimizerConfig::AdamW(AdamWConfig {
            learning_rate: 1.0e-3,
            grad_clip_norm: 1.0,
            ..AdamWConfig::default()
        });
        let first_report = session
            .train_into_model(
                &mut model,
                TrainingRunConfig {
                    steps: 4,
                    report_interval: 4,
                    ..TrainingRunConfig::default()
                },
                optimizer,
                false,
                None,
            )
            .unwrap();
        let optimizer_step = session.step;
        session.replace_batch(&second).unwrap();
        let second_report = session
            .train_into_model(
                &mut model,
                TrainingRunConfig {
                    steps: 4,
                    report_interval: 4,
                    ..TrainingRunConfig::default()
                },
                optimizer,
                false,
                None,
            )
            .unwrap();
        assert_eq!(optimizer_step, 4);
        assert_eq!(session.step, 8);
        assert_eq!(first_report.rows, 32);
        assert_eq!(second_report.rows, 32);
        assert!(second_report.final_loss.is_finite());
    }
}
