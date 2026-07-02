use serde::{Deserialize, Serialize};

use crate::{AutomataResult, NpaLowRankAdapter, NpaModel};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SgdConfig {
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub grad_clip_norm: f32,
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            weight_decay: 0.0,
            grad_clip_norm: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SupervisedBatch {
    pub features: Vec<f32>,
    pub target_update: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SupervisedStepReport {
    pub loss: f32,
    pub rows: usize,
    pub grad_norm: f32,
    pub grad_scale: f32,
    pub clipped: bool,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TrainingRunConfig {
    pub steps: usize,
    pub report_interval: usize,
    pub sgd: SgdConfig,
}

impl Default for TrainingRunConfig {
    fn default() -> Self {
        Self {
            steps: 64,
            report_interval: 1,
            sgd: SgdConfig::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TrainingHistoryEntry {
    pub step: usize,
    pub loss: f32,
    pub grad_norm: f32,
    pub grad_scale: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingRunReport {
    pub steps: usize,
    pub rows: usize,
    pub initial_loss: f32,
    pub final_loss: f32,
    pub best_loss: f32,
    pub history: Vec<TrainingHistoryEntry>,
}

#[derive(Clone, Debug)]
pub struct SupervisedGradients {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub features: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct LowRankAdapterGradients {
    pub w1_down: Vec<f32>,
    pub w1_up: Vec<f32>,
    pub w2_down: Vec<f32>,
    pub w2_up: Vec<f32>,
    pub b1_delta: Vec<f32>,
    pub b2_delta: Vec<f32>,
    pub rows: usize,
}

impl LowRankAdapterGradients {
    pub fn grad_norm(&self) -> f32 {
        grad_norm(&[
            &self.w1_down,
            &self.w1_up,
            &self.w2_down,
            &self.w2_up,
            &self.b1_delta,
            &self.b2_delta,
        ])
    }
}

pub fn supervised_train_step(
    model: &mut NpaModel,
    batch: &SupervisedBatch,
    cfg: SgdConfig,
) -> AutomataResult<SupervisedStepReport> {
    validate_sgd_config(cfg)?;
    let (grads, mut report) = supervised_backward(model, batch)?;
    let step = apply_sgd_gradients(model, &grads, cfg)?;
    report.grad_norm = step.grad_norm;
    report.grad_scale = step.grad_scale;
    report.clipped = step.clipped;
    Ok(report)
}

pub fn supervised_adapter_loss(
    base_model: &NpaModel,
    adapter: &NpaLowRankAdapter,
    batch: &SupervisedBatch,
) -> AutomataResult<f32> {
    let adapted = adapter.apply_to_model(base_model)?;
    supervised_loss(&adapted, batch)
}

pub fn supervised_adapter_train_step(
    base_model: &NpaModel,
    adapter: &mut NpaLowRankAdapter,
    batch: &SupervisedBatch,
    cfg: SgdConfig,
) -> AutomataResult<SupervisedStepReport> {
    validate_sgd_config(cfg)?;
    let adapted = adapter.apply_to_model(base_model)?;
    let (full_grads, mut report) = supervised_backward(&adapted, batch)?;
    let adapter_grads = project_low_rank_adapter_gradients(base_model, adapter, &full_grads)?;
    let step = apply_sgd_adapter_gradients(adapter, &adapter_grads, cfg)?;
    report.grad_norm = step.grad_norm;
    report.grad_scale = step.grad_scale;
    report.clipped = step.clipped;
    Ok(report)
}

pub fn run_supervised_training(
    model: &mut NpaModel,
    batch: &SupervisedBatch,
    cfg: TrainingRunConfig,
) -> AutomataResult<TrainingRunReport> {
    validate_sgd_config(cfg.sgd)?;
    let (rows, _) = validate_batch(model, batch)?;
    let initial_loss = supervised_loss(model, batch)?;
    let mut final_loss = initial_loss;
    let mut best_loss = initial_loss;
    let mut best_model = model.clone();
    let report_interval = cfg.report_interval.max(1);
    let mut history = Vec::new();

    for step in 1..=cfg.steps {
        let step_report = supervised_train_step(model, batch, cfg.sgd)?;
        if step == cfg.steps || step.is_multiple_of(report_interval) {
            final_loss = supervised_loss(model, batch)?;
            if final_loss < best_loss {
                best_loss = final_loss;
                best_model = model.clone();
            }
            history.push(TrainingHistoryEntry {
                step,
                loss: final_loss,
                grad_norm: step_report.grad_norm,
                grad_scale: step_report.grad_scale,
            });
        }
    }
    if best_loss < final_loss {
        *model = best_model;
        final_loss = best_loss;
    }

    Ok(TrainingRunReport {
        steps: cfg.steps,
        rows,
        initial_loss,
        final_loss,
        best_loss,
        history,
    })
}

pub fn run_supervised_adapter_training(
    base_model: &NpaModel,
    adapter: &mut NpaLowRankAdapter,
    batch: &SupervisedBatch,
    cfg: TrainingRunConfig,
) -> AutomataResult<TrainingRunReport> {
    validate_sgd_config(cfg.sgd)?;
    let (rows, _) = validate_batch(&adapter.apply_to_model(base_model)?, batch)?;
    let initial_loss = supervised_adapter_loss(base_model, adapter, batch)?;
    let mut final_loss = initial_loss;
    let mut best_loss = initial_loss;
    let mut best_adapter = adapter.clone();
    let report_interval = cfg.report_interval.max(1);
    let mut history = Vec::new();

    for step in 1..=cfg.steps {
        let step_report = supervised_adapter_train_step(base_model, adapter, batch, cfg.sgd)?;
        if step == cfg.steps || step.is_multiple_of(report_interval) {
            final_loss = supervised_adapter_loss(base_model, adapter, batch)?;
            if final_loss < best_loss {
                best_loss = final_loss;
                best_adapter = adapter.clone();
            }
            history.push(TrainingHistoryEntry {
                step,
                loss: final_loss,
                grad_norm: step_report.grad_norm,
                grad_scale: step_report.grad_scale,
            });
        }
    }
    if best_loss < final_loss {
        *adapter = best_adapter;
        final_loss = best_loss;
    }

    Ok(TrainingRunReport {
        steps: cfg.steps,
        rows,
        initial_loss,
        final_loss,
        best_loss,
        history,
    })
}

pub fn supervised_loss(model: &NpaModel, batch: &SupervisedBatch) -> AutomataResult<f32> {
    let (rows, output_dims) = validate_batch(model, batch)?;
    let output = model.forward_update_from_features(&batch.features)?;
    let loss = output
        .iter()
        .zip(batch.target_update.iter())
        .map(|(actual, expected)| {
            let diff = actual - expected;
            diff * diff
        })
        .sum::<f32>()
        / rows as f32;
    if !loss.is_finite() {
        return Err(crate::AutomataError::InvalidArgument(
            "supervised loss is not finite".to_string(),
        ));
    }
    if output.len() != rows * output_dims {
        return Err(crate::AutomataError::InvalidArgument(format!(
            "model output len {} != {}",
            output.len(),
            rows * output_dims
        )));
    }
    Ok(loss)
}

pub fn supervised_backward(
    model: &NpaModel,
    batch: &SupervisedBatch,
) -> AutomataResult<(SupervisedGradients, SupervisedStepReport)> {
    let (rows, output_dims) = validate_batch(model, batch)?;
    let input_dims = model.config.perception_dims();

    let mut gw1 = vec![0.0; model.weights.w1.len()];
    let mut gb1 = vec![0.0; model.weights.b1.len()];
    let mut gw2 = vec![0.0; model.weights.w2.len()];
    let mut gb2 = vec![0.0; model.weights.b2.len()];
    let mut gfeatures = vec![0.0; batch.features.len()];
    let mut hidden = vec![0.0; model.config.hidden_dims];
    let mut pre_hidden = vec![0.0; model.config.hidden_dims];
    let mut output = vec![0.0; output_dims];
    let mut d_hidden = vec![0.0; model.config.hidden_dims];
    let mut loss = 0.0;

    for row in 0..rows {
        output.fill(0.0);
        d_hidden.fill(0.0);

        let feature = &batch.features[row * input_dims..(row + 1) * input_dims];
        for (h, (pre_hidden_value, hidden_value)) in
            pre_hidden.iter_mut().zip(hidden.iter_mut()).enumerate()
        {
            let mut sum = model.weights.b1[h];
            let base = h * input_dims;
            for (i, value) in feature.iter().enumerate().take(input_dims) {
                sum += model.weights.w1[base + i] * *value;
            }
            *pre_hidden_value = sum;
            *hidden_value = sum.max(0.0);
        }

        for (o, out) in output.iter_mut().enumerate() {
            let mut sum = model.weights.b2[o];
            let base = o * model.config.hidden_dims;
            for (h, value) in hidden.iter().enumerate().take(model.config.hidden_dims) {
                sum += model.weights.w2[base + h] * *value;
            }
            *out = sum;
        }

        let target = &batch.target_update[row * output_dims..(row + 1) * output_dims];
        for o in 0..output_dims {
            let diff = output[o] - target[o];
            loss += diff * diff;
            let d_out = 2.0 * diff / rows as f32;
            gb2[o] += d_out;
            let w2_base = o * model.config.hidden_dims;
            for h in 0..model.config.hidden_dims {
                gw2[w2_base + h] += d_out * hidden[h];
                d_hidden[h] += d_out * model.weights.w2[w2_base + h];
            }
        }

        for h in 0..model.config.hidden_dims {
            let d_pre = if pre_hidden[h] > 0.0 {
                d_hidden[h]
            } else {
                0.0
            };
            gb1[h] += d_pre;
            let w1_base = h * input_dims;
            for i in 0..input_dims {
                gw1[w1_base + i] += d_pre * feature[i];
                gfeatures[row * input_dims + i] += d_pre * model.weights.w1[w1_base + i];
            }
        }
    }

    loss /= rows as f32;
    let grad_norm = grad_norm(&[&gw1, &gb1, &gw2, &gb2]);
    if !loss.is_finite() || !grad_norm.is_finite() {
        return Err(crate::AutomataError::InvalidArgument(
            "supervised backward produced non-finite values".to_string(),
        ));
    }
    Ok((
        SupervisedGradients {
            w1: gw1,
            b1: gb1,
            w2: gw2,
            b2: gb2,
            features: gfeatures,
        },
        SupervisedStepReport {
            loss,
            rows,
            grad_norm,
            grad_scale: 1.0,
            clipped: false,
        },
    ))
}

pub fn mlp_backward_from_output_gradients(
    model: &NpaModel,
    features: &[f32],
    output_gradients: &[f32],
) -> AutomataResult<SupervisedGradients> {
    model.validate()?;
    let input_dims = model.config.perception_dims();
    let output_dims = model.config.update_dims();
    let rows = features.len() / input_dims;
    if rows == 0 || features.len() != rows * input_dims {
        return Err(crate::AutomataError::InvalidArgument(
            "features do not form whole perception rows".to_string(),
        ));
    }
    if output_gradients.len() != rows * output_dims {
        return Err(crate::AutomataError::InvalidArgument(format!(
            "output_gradients len {} != {}",
            output_gradients.len(),
            rows * output_dims
        )));
    }
    ensure_finite("features", features)?;
    ensure_finite("output_gradients", output_gradients)?;

    let mut gw1 = vec![0.0; model.weights.w1.len()];
    let mut gb1 = vec![0.0; model.weights.b1.len()];
    let mut gw2 = vec![0.0; model.weights.w2.len()];
    let mut gb2 = vec![0.0; model.weights.b2.len()];
    let mut gfeatures = vec![0.0; features.len()];
    let mut hidden = vec![0.0; model.config.hidden_dims];
    let mut pre_hidden = vec![0.0; model.config.hidden_dims];
    let mut d_hidden = vec![0.0; model.config.hidden_dims];

    for row in 0..rows {
        d_hidden.fill(0.0);

        let feature = &features[row * input_dims..(row + 1) * input_dims];
        for (h, (pre_hidden_value, hidden_value)) in
            pre_hidden.iter_mut().zip(hidden.iter_mut()).enumerate()
        {
            let mut sum = model.weights.b1[h];
            let base = h * input_dims;
            for (i, value) in feature.iter().enumerate().take(input_dims) {
                sum += model.weights.w1[base + i] * *value;
            }
            *pre_hidden_value = sum;
            *hidden_value = sum.max(0.0);
        }

        let row_output_gradients = &output_gradients[row * output_dims..(row + 1) * output_dims];
        for (o, d_out) in row_output_gradients.iter().copied().enumerate() {
            gb2[o] += d_out;
            let w2_base = o * model.config.hidden_dims;
            for h in 0..model.config.hidden_dims {
                gw2[w2_base + h] += d_out * hidden[h];
                d_hidden[h] += d_out * model.weights.w2[w2_base + h];
            }
        }

        for h in 0..model.config.hidden_dims {
            let d_pre = if pre_hidden[h] > 0.0 {
                d_hidden[h]
            } else {
                0.0
            };
            gb1[h] += d_pre;
            let w1_base = h * input_dims;
            for i in 0..input_dims {
                gw1[w1_base + i] += d_pre * feature[i];
                gfeatures[row * input_dims + i] += d_pre * model.weights.w1[w1_base + i];
            }
        }
    }

    let grads = SupervisedGradients {
        w1: gw1,
        b1: gb1,
        w2: gw2,
        b2: gb2,
        features: gfeatures,
    };
    validate_gradients(&grads)?;
    Ok(grads)
}

pub fn apply_sgd_gradients(
    model: &mut NpaModel,
    grads: &SupervisedGradients,
    cfg: SgdConfig,
) -> AutomataResult<SupervisedStepReport> {
    validate_sgd_config(cfg)?;
    validate_gradients(grads)?;
    let input_dims = model.config.perception_dims();
    let rows = if grads.features.is_empty() {
        0
    } else {
        grads.features.len() / input_dims
    };
    let grad_norm = grad_norm(&[&grads.w1, &grads.b1, &grads.w2, &grads.b2]);
    if !grad_norm.is_finite() {
        return Err(crate::AutomataError::InvalidArgument(
            "gradient norm is not finite".to_string(),
        ));
    }
    let scale = if cfg.grad_clip_norm > 0.0 && grad_norm > cfg.grad_clip_norm {
        cfg.grad_clip_norm / grad_norm
    } else {
        1.0
    };

    apply_sgd(&mut model.weights.w1, &grads.w1, cfg, scale);
    apply_sgd(&mut model.weights.b1, &grads.b1, cfg, scale);
    apply_sgd(&mut model.weights.w2, &grads.w2, cfg, scale);
    apply_sgd(&mut model.weights.b2, &grads.b2, cfg, scale);

    Ok(SupervisedStepReport {
        loss: 0.0,
        rows,
        grad_norm,
        grad_scale: scale,
        clipped: scale < 1.0,
    })
}

pub fn project_low_rank_adapter_gradients(
    base_model: &NpaModel,
    adapter: &NpaLowRankAdapter,
    full_grads: &SupervisedGradients,
) -> AutomataResult<LowRankAdapterGradients> {
    base_model.validate()?;
    adapter.validate(&base_model.config)?;
    validate_gradients(full_grads)?;

    let input_dims = base_model.config.perception_dims();
    let hidden_dims = base_model.config.hidden_dims;
    let output_dims = base_model.config.update_dims();
    let rows = if full_grads.features.is_empty() {
        0
    } else {
        full_grads.features.len() / input_dims
    };
    let scale = adapter.alpha / adapter.rank as f32;

    let mut w1_down = vec![0.0; adapter.w1_down.len()];
    let mut w1_up = vec![0.0; adapter.w1_up.len()];
    let mut w2_down = vec![0.0; adapter.w2_down.len()];
    let mut w2_up = vec![0.0; adapter.w2_up.len()];

    project_low_rank_matrix_gradients(
        &full_grads.w1,
        hidden_dims,
        input_dims,
        adapter.rank,
        &adapter.w1_up,
        &adapter.w1_down,
        scale,
        &mut w1_up,
        &mut w1_down,
    );
    project_low_rank_matrix_gradients(
        &full_grads.w2,
        output_dims,
        hidden_dims,
        adapter.rank,
        &adapter.w2_up,
        &adapter.w2_down,
        scale,
        &mut w2_up,
        &mut w2_down,
    );

    let grads = LowRankAdapterGradients {
        w1_down,
        w1_up,
        w2_down,
        w2_up,
        b1_delta: full_grads.b1.clone(),
        b2_delta: full_grads.b2.clone(),
        rows,
    };
    validate_adapter_gradients(adapter, &grads)?;
    Ok(grads)
}

pub fn apply_sgd_adapter_gradients(
    adapter: &mut NpaLowRankAdapter,
    grads: &LowRankAdapterGradients,
    cfg: SgdConfig,
) -> AutomataResult<SupervisedStepReport> {
    validate_sgd_config(cfg)?;
    validate_adapter_gradients(adapter, grads)?;
    let grad_norm = grads.grad_norm();
    if !grad_norm.is_finite() {
        return Err(crate::AutomataError::InvalidArgument(
            "adapter gradient norm is not finite".to_string(),
        ));
    }
    let scale = if cfg.grad_clip_norm > 0.0 && grad_norm > cfg.grad_clip_norm {
        cfg.grad_clip_norm / grad_norm
    } else {
        1.0
    };

    apply_sgd(&mut adapter.w1_down, &grads.w1_down, cfg, scale);
    apply_sgd(&mut adapter.w1_up, &grads.w1_up, cfg, scale);
    apply_sgd(&mut adapter.w2_down, &grads.w2_down, cfg, scale);
    apply_sgd(&mut adapter.w2_up, &grads.w2_up, cfg, scale);
    apply_sgd(&mut adapter.b1_delta, &grads.b1_delta, cfg, scale);
    apply_sgd(&mut adapter.b2_delta, &grads.b2_delta, cfg, scale);

    Ok(SupervisedStepReport {
        loss: 0.0,
        rows: grads.rows,
        grad_norm,
        grad_scale: scale,
        clipped: scale < 1.0,
    })
}

#[allow(clippy::too_many_arguments)]
fn project_low_rank_matrix_gradients(
    matrix_grads: &[f32],
    rows: usize,
    cols: usize,
    rank: usize,
    up: &[f32],
    down: &[f32],
    scale: f32,
    up_grads: &mut [f32],
    down_grads: &mut [f32],
) {
    for row in 0..rows {
        for col in 0..cols {
            let grad = matrix_grads[row * cols + col] * scale;
            for r in 0..rank {
                up_grads[row * rank + r] += grad * down[r * cols + col];
                down_grads[r * cols + col] += grad * up[row * rank + r];
            }
        }
    }
}

fn grad_norm(groups: &[&[f32]]) -> f32 {
    groups
        .iter()
        .flat_map(|g| g.iter())
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt()
}

fn apply_sgd(values: &mut [f32], grads: &[f32], cfg: SgdConfig, scale: f32) {
    for (value, grad) in values.iter_mut().zip(grads.iter()) {
        let grad = grad * scale + cfg.weight_decay * *value;
        *value -= cfg.learning_rate * grad;
    }
}

fn validate_sgd_config(cfg: SgdConfig) -> AutomataResult<()> {
    if !cfg.learning_rate.is_finite() || cfg.learning_rate < 0.0 {
        return Err(crate::AutomataError::InvalidArgument(format!(
            "learning_rate must be finite and non-negative, got {}",
            cfg.learning_rate
        )));
    }
    if !cfg.weight_decay.is_finite() || cfg.weight_decay < 0.0 {
        return Err(crate::AutomataError::InvalidArgument(format!(
            "weight_decay must be finite and non-negative, got {}",
            cfg.weight_decay
        )));
    }
    if !cfg.grad_clip_norm.is_finite() || cfg.grad_clip_norm < 0.0 {
        return Err(crate::AutomataError::InvalidArgument(format!(
            "grad_clip_norm must be finite and non-negative, got {}",
            cfg.grad_clip_norm
        )));
    }
    Ok(())
}

fn validate_batch(model: &NpaModel, batch: &SupervisedBatch) -> AutomataResult<(usize, usize)> {
    model.validate()?;
    let input_dims = model.config.perception_dims();
    let output_dims = model.config.update_dims();
    let rows = batch.features.len() / input_dims;
    if rows == 0 || batch.features.len() != rows * input_dims {
        return Err(crate::AutomataError::InvalidArgument(
            "features do not form whole perception rows".to_string(),
        ));
    }
    if batch.target_update.len() != rows * output_dims {
        return Err(crate::AutomataError::InvalidArgument(format!(
            "target_update len {} != {}",
            batch.target_update.len(),
            rows * output_dims
        )));
    }
    ensure_finite("features", &batch.features)?;
    ensure_finite("target_update", &batch.target_update)?;
    Ok((rows, output_dims))
}

fn validate_gradients(grads: &SupervisedGradients) -> AutomataResult<()> {
    ensure_finite("w1 gradients", &grads.w1)?;
    ensure_finite("b1 gradients", &grads.b1)?;
    ensure_finite("w2 gradients", &grads.w2)?;
    ensure_finite("b2 gradients", &grads.b2)?;
    ensure_finite("feature gradients", &grads.features)
}

fn validate_adapter_gradients(
    adapter: &NpaLowRankAdapter,
    grads: &LowRankAdapterGradients,
) -> AutomataResult<()> {
    let expected = [
        ("w1_down", adapter.w1_down.len(), grads.w1_down.len()),
        ("w1_up", adapter.w1_up.len(), grads.w1_up.len()),
        ("w2_down", adapter.w2_down.len(), grads.w2_down.len()),
        ("w2_up", adapter.w2_up.len(), grads.w2_up.len()),
        ("b1_delta", adapter.b1_delta.len(), grads.b1_delta.len()),
        ("b2_delta", adapter.b2_delta.len(), grads.b2_delta.len()),
    ];
    for (name, expected_len, actual_len) in expected {
        if actual_len != expected_len {
            return Err(crate::AutomataError::InvalidArgument(format!(
                "adapter gradient {name} len {actual_len} != {expected_len}"
            )));
        }
    }
    ensure_finite("adapter gradient w1_down", &grads.w1_down)?;
    ensure_finite("adapter gradient w1_up", &grads.w1_up)?;
    ensure_finite("adapter gradient w2_down", &grads.w2_down)?;
    ensure_finite("adapter gradient w2_up", &grads.w2_up)?;
    ensure_finite("adapter gradient b1_delta", &grads.b1_delta)?;
    ensure_finite("adapter gradient b2_delta", &grads.b2_delta)?;
    Ok(())
}

fn ensure_finite(name: &str, values: &[f32]) -> AutomataResult<()> {
    if values.iter().all(|value| value.is_finite()) {
        return Ok(());
    }
    Err(crate::AutomataError::InvalidArgument(format!(
        "{name} contain non-finite values"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NpaConfig, NpaWeights};

    fn adapter_loss_for_param(
        base: &NpaModel,
        adapter: &mut NpaLowRankAdapter,
        batch: &SupervisedBatch,
        param_idx: usize,
        delta: f32,
    ) -> f32 {
        adapter.w2_up[param_idx] += delta;
        let loss = supervised_adapter_loss(base, adapter, batch).unwrap();
        adapter.w2_up[param_idx] -= delta;
        loss
    }

    #[test]
    fn supervised_training_restores_best_checkpoint_after_overshoot() {
        let config = NpaConfig::growing_2d();
        let mut weights = NpaWeights::zeros(&config);
        weights.b1[0] = 1.0;
        let mut model = NpaModel { config, weights };
        let initial = model.clone();

        let mut target_update = vec![0.0; model.config.update_dims()];
        target_update[0] = 1.0;
        let batch = SupervisedBatch {
            features: vec![0.0; model.config.perception_dims()],
            target_update,
        };
        let initial_loss = supervised_loss(&model, &batch).unwrap();
        let report = run_supervised_training(
            &mut model,
            &batch,
            TrainingRunConfig {
                steps: 1,
                report_interval: 1,
                sgd: SgdConfig {
                    learning_rate: 2.0,
                    weight_decay: 0.0,
                    grad_clip_norm: 0.0,
                },
            },
        )
        .unwrap();

        assert!(report.history[0].loss > initial_loss);
        assert_eq!(report.best_loss, initial_loss);
        assert_eq!(report.final_loss, initial_loss);
        assert_eq!(model.weights.w1, initial.weights.w1);
        assert_eq!(model.weights.b1, initial.weights.b1);
        assert_eq!(model.weights.w2, initial.weights.w2);
        assert_eq!(model.weights.b2, initial.weights.b2);
    }

    #[test]
    fn low_rank_adapter_gradient_projection_matches_finite_difference() {
        let mut config = NpaConfig::growing_3dgs();
        config.hidden_dims = 3;
        config.state_dims = 5;
        config.position_features = false;
        let input_dims = config.perception_dims();
        let output_dims = config.update_dims();
        let mut base = NpaModel {
            weights: NpaWeights::seeded(&config, 11),
            config,
        };
        for bias in &mut base.weights.b1 {
            *bias = 0.25;
        }
        let mut adapter = NpaLowRankAdapter::seeded(&base.config, 2, 2.0, 17);
        let batch = SupervisedBatch {
            features: vec![0.2; input_dims],
            target_update: vec![0.05; output_dims],
        };
        let adapted = adapter.apply_to_model(&base).unwrap();
        let (full_grads, _) = supervised_backward(&adapted, &batch).unwrap();
        let adapter_grads =
            project_low_rank_adapter_gradients(&base, &adapter, &full_grads).unwrap();

        let param_idx = 0;
        let eps = 1.0e-3;
        let plus = adapter_loss_for_param(&base, &mut adapter, &batch, param_idx, eps);
        let minus = adapter_loss_for_param(&base, &mut adapter, &batch, param_idx, -eps);
        let numeric = (plus - minus) / (2.0 * eps);
        let analytic = adapter_grads.w2_up[param_idx];

        assert!(
            (analytic - numeric).abs() < 2.0e-3,
            "adapter gradient mismatch analytic={analytic} numeric={numeric}"
        );
    }

    #[test]
    fn supervised_adapter_step_updates_adapter_without_mutating_base() {
        let config = NpaConfig::growing_3dgs();
        let base = NpaModel {
            weights: NpaWeights::zeros(&config),
            config,
        };
        let base_before = base.clone();
        let mut adapter = NpaLowRankAdapter::zeros(&base.config, 2, 2.0);
        let mut target_update = vec![0.0; base.config.update_dims()];
        target_update[0] = 1.0;
        let batch = SupervisedBatch {
            features: vec![0.0; base.config.perception_dims()],
            target_update,
        };
        let before = supervised_adapter_loss(&base, &adapter, &batch).unwrap();

        let report = supervised_adapter_train_step(
            &base,
            &mut adapter,
            &batch,
            SgdConfig {
                learning_rate: 0.1,
                weight_decay: 0.0,
                grad_clip_norm: 0.0,
            },
        )
        .unwrap();
        let after = supervised_adapter_loss(&base, &adapter, &batch).unwrap();

        assert_eq!(base.weights.w1, base_before.weights.w1);
        assert_eq!(base.weights.b1, base_before.weights.b1);
        assert_eq!(base.weights.w2, base_before.weights.w2);
        assert_eq!(base.weights.b2, base_before.weights.b2);
        assert!(adapter.b2_delta[0] > 0.0);
        assert!(after < before, "adapter step should reduce supervised loss");
        assert_eq!(report.rows, 1);
        assert!(report.grad_norm > 0.0);
    }

    #[test]
    fn supervised_adapter_training_run_tracks_history_without_mutating_base() {
        let mut config = NpaConfig::growing_3dgs();
        config.hidden_dims = 4;
        let base = NpaModel {
            weights: NpaWeights::zeros(&config),
            config,
        };
        let base_before = base.clone();
        let mut adapter = NpaLowRankAdapter::zeros(&base.config, 2, 2.0);
        let mut target_update = vec![0.0; base.config.update_dims()];
        target_update[1] = -0.75;
        let batch = SupervisedBatch {
            features: vec![0.0; base.config.perception_dims()],
            target_update,
        };
        let before = supervised_adapter_loss(&base, &adapter, &batch).unwrap();

        let report = run_supervised_adapter_training(
            &base,
            &mut adapter,
            &batch,
            TrainingRunConfig {
                steps: 3,
                report_interval: 1,
                sgd: SgdConfig {
                    learning_rate: 0.1,
                    weight_decay: 0.0,
                    grad_clip_norm: 0.0,
                },
            },
        )
        .unwrap();
        let after = supervised_adapter_loss(&base, &adapter, &batch).unwrap();

        assert_eq!(base.weights.w1, base_before.weights.w1);
        assert_eq!(base.weights.b1, base_before.weights.b1);
        assert_eq!(base.weights.w2, base_before.weights.w2);
        assert_eq!(base.weights.b2, base_before.weights.b2);
        assert_eq!(report.steps, 3);
        assert_eq!(report.history.len(), 3);
        assert!(report.best_loss <= before);
        assert!(after <= before);
        assert!(adapter.b2_delta[1] < 0.0);
    }
}
