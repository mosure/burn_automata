use serde::{Deserialize, Serialize};

use crate::{AutomataResult, NpaModel};

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
}
