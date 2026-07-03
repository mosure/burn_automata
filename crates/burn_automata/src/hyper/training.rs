use serde::{Deserialize, Serialize};

use crate::{
    AutomataError, AutomataResult, LowRankAdapterGradients, NpaLowRankAdapter, NpaModel, SgdConfig,
    SupervisedBatch, project_low_rank_adapter_gradients, supervised_backward, supervised_loss,
};

use super::{
    condition::ConditionImage2d,
    hypernet::{HyperForwardCache, HyperNpa2d, HyperNpa2dGradients},
};

#[derive(Clone, Debug)]
pub struct HyperAdapterExample2d {
    pub condition: ConditionImage2d,
    pub target_adapter: NpaLowRankAdapter,
}

#[derive(Clone, Debug)]
pub struct HyperFlowExample2d {
    pub condition: ConditionImage2d,
    pub batch: SupervisedBatch,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct HyperAdapterTrainingReport {
    pub loss: f32,
    pub examples: usize,
    pub rows: usize,
    pub grad_norm: f32,
    pub grad_scale: f32,
    pub clipped: bool,
}

pub fn hyper_adapter_regression_loss(
    hyper: &HyperNpa2d,
    examples: &[HyperAdapterExample2d],
) -> AutomataResult<f32> {
    validate_examples(examples)?;
    hyper.validate()?;
    let output_dims = hyper.adapter_parameter_count();
    let mut loss = 0.0_f32;
    for example in examples {
        example.target_adapter.validate(&hyper.npa_config)?;
        let predicted = hyper.predict_adapter_vector(&example.condition)?;
        let target = example.target_adapter.to_parameter_vector();
        for (actual, expected) in predicted.iter().zip(target.iter()) {
            let diff = actual - expected;
            loss += diff * diff;
        }
    }
    Ok(loss / (examples.len() * output_dims) as f32)
}

pub fn hyper_adapter_regression_train_step(
    hyper: &mut HyperNpa2d,
    examples: &[HyperAdapterExample2d],
    cfg: SgdConfig,
) -> AutomataResult<HyperAdapterTrainingReport> {
    validate_examples(examples)?;
    validate_sgd_config(cfg)?;
    hyper.validate()?;
    let output_dims = hyper.adapter_parameter_count();
    let mut loss = 0.0_f32;
    let mut grads = HyperNpa2dGradients::zeros_like(hyper);
    let normalizer = (examples.len() * output_dims) as f32;

    for example in examples {
        example.target_adapter.validate(&hyper.npa_config)?;
        let cache = hyper.forward_cache(&example.condition)?;
        let target = example.target_adapter.to_parameter_vector();
        let mut output_gradients = vec![0.0; output_dims];
        for (idx, (actual, expected)) in cache.output.iter().zip(target.iter()).enumerate() {
            let diff = actual - expected;
            loss += diff * diff;
            output_gradients[idx] = 2.0 * diff / normalizer;
        }
        hyper.accumulate_output_gradients(&cache, &output_gradients, 1.0, &mut grads)?;
    }

    loss /= normalizer;
    let (grad_norm, grad_scale, clipped) = apply_hyper_sgd(hyper, &grads, cfg)?;
    Ok(HyperAdapterTrainingReport {
        loss,
        examples: examples.len(),
        rows: examples.len(),
        grad_norm,
        grad_scale,
        clipped,
    })
}

pub fn hyper_rectified_flow_loss(
    base_model: &NpaModel,
    hyper: &HyperNpa2d,
    examples: &[HyperFlowExample2d],
) -> AutomataResult<f32> {
    validate_flow_examples(examples)?;
    validate_base(base_model, hyper)?;
    let mut loss = 0.0_f32;
    for example in examples {
        let adapter = hyper.predict_adapter(&example.condition)?;
        let adapted = adapter.apply_to_model(base_model)?;
        loss += supervised_loss(&adapted, &example.batch)?;
    }
    Ok(loss / examples.len() as f32)
}

pub fn hyper_rectified_flow_train_step(
    base_model: &NpaModel,
    hyper: &mut HyperNpa2d,
    examples: &[HyperFlowExample2d],
    cfg: SgdConfig,
) -> AutomataResult<HyperAdapterTrainingReport> {
    validate_flow_examples(examples)?;
    validate_sgd_config(cfg)?;
    validate_base(base_model, hyper)?;

    let mut loss = 0.0_f32;
    let mut rows = 0_usize;
    let mut grads = HyperNpa2dGradients::zeros_like(hyper);
    let example_scale = 1.0 / examples.len() as f32;

    for example in examples {
        let cache = hyper.forward_cache(&example.condition)?;
        let adapter = adapter_from_cache(hyper, &cache)?;
        let adapted = adapter.apply_to_model(base_model)?;
        let (full_grads, report) = supervised_backward(&adapted, &example.batch)?;
        let adapter_grads = project_low_rank_adapter_gradients(base_model, &adapter, &full_grads)?;
        let output_gradients = adapter_gradient_vector(&adapter_grads);
        loss += report.loss;
        rows += report.rows;
        hyper.accumulate_output_gradients(&cache, &output_gradients, example_scale, &mut grads)?;
    }

    loss /= examples.len() as f32;
    let (grad_norm, grad_scale, clipped) = apply_hyper_sgd(hyper, &grads, cfg)?;
    Ok(HyperAdapterTrainingReport {
        loss,
        examples: examples.len(),
        rows,
        grad_norm,
        grad_scale,
        clipped,
    })
}

fn adapter_from_cache(
    hyper: &HyperNpa2d,
    cache: &HyperForwardCache,
) -> AutomataResult<NpaLowRankAdapter> {
    NpaLowRankAdapter::from_parameter_vector(
        &hyper.npa_config,
        hyper.config.adapter_rank,
        hyper.config.adapter_alpha,
        cache.output.clone(),
    )
}

pub(crate) fn adapter_gradient_vector(grads: &LowRankAdapterGradients) -> Vec<f32> {
    let mut values = Vec::with_capacity(
        grads.w1_down.len()
            + grads.w1_up.len()
            + grads.w2_down.len()
            + grads.w2_up.len()
            + grads.b1_delta.len()
            + grads.b2_delta.len(),
    );
    values.extend_from_slice(&grads.w1_down);
    values.extend_from_slice(&grads.w1_up);
    values.extend_from_slice(&grads.w2_down);
    values.extend_from_slice(&grads.w2_up);
    values.extend_from_slice(&grads.b1_delta);
    values.extend_from_slice(&grads.b2_delta);
    values
}

fn validate_examples(examples: &[HyperAdapterExample2d]) -> AutomataResult<()> {
    if examples.is_empty() {
        return Err(AutomataError::InvalidArgument(
            "hyper adapter training requires at least one example".to_string(),
        ));
    }
    Ok(())
}

fn validate_flow_examples(examples: &[HyperFlowExample2d]) -> AutomataResult<()> {
    if examples.is_empty() {
        return Err(AutomataError::InvalidArgument(
            "hyper flow training requires at least one example".to_string(),
        ));
    }
    Ok(())
}

fn validate_base(base_model: &NpaModel, hyper: &HyperNpa2d) -> AutomataResult<()> {
    base_model.validate()?;
    hyper.validate()?;
    if base_model.config != hyper.npa_config {
        return Err(AutomataError::InvalidArgument(
            "hyper NPA config must match base model config".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn apply_hyper_sgd(
    hyper: &mut HyperNpa2d,
    grads: &HyperNpa2dGradients,
    cfg: SgdConfig,
) -> AutomataResult<(f32, f32, bool)> {
    grads.validate(hyper)?;
    let grad_norm = grads.grad_norm();
    if !grad_norm.is_finite() {
        return Err(AutomataError::InvalidArgument(
            "hyper gradient norm is not finite".to_string(),
        ));
    }
    let grad_scale = if cfg.grad_clip_norm > 0.0 && grad_norm > cfg.grad_clip_norm {
        cfg.grad_clip_norm / grad_norm
    } else {
        1.0
    };
    apply_sgd(&mut hyper.weights.w1, &grads.w1, cfg, grad_scale);
    apply_sgd(&mut hyper.weights.b1, &grads.b1, cfg, grad_scale);
    apply_sgd(&mut hyper.weights.w2, &grads.w2, cfg, grad_scale);
    apply_sgd(&mut hyper.weights.b2, &grads.b2, cfg, grad_scale);
    Ok((grad_norm, grad_scale, grad_scale < 1.0))
}

fn apply_sgd(values: &mut [f32], grads: &[f32], cfg: SgdConfig, grad_scale: f32) {
    for (value, grad) in values.iter_mut().zip(grads.iter()) {
        let grad = grad * grad_scale + cfg.weight_decay * *value;
        *value -= cfg.learning_rate * grad;
    }
}

fn validate_sgd_config(cfg: SgdConfig) -> AutomataResult<()> {
    if !cfg.learning_rate.is_finite() || cfg.learning_rate < 0.0 {
        return Err(AutomataError::InvalidArgument(format!(
            "learning_rate must be finite and non-negative, got {}",
            cfg.learning_rate
        )));
    }
    if !cfg.weight_decay.is_finite() || cfg.weight_decay < 0.0 {
        return Err(AutomataError::InvalidArgument(format!(
            "weight_decay must be finite and non-negative, got {}",
            cfg.weight_decay
        )));
    }
    if !cfg.grad_clip_norm.is_finite() || cfg.grad_clip_norm < 0.0 {
        return Err(AutomataError::InvalidArgument(format!(
            "grad_clip_norm must be finite and non-negative, got {}",
            cfg.grad_clip_norm
        )));
    }
    Ok(())
}
