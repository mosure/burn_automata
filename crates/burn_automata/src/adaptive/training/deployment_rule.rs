use burn::tensor::backend::Backend;

use super::{
    AdaptiveDeploymentRuleTrainingReport, AdaptiveDeploymentRuleValidationReport,
    AdaptiveDeploymentStrategy, AdaptiveDeploymentTarget, AdaptiveMultiscaleTrainingBatch,
    AdaptiveMultiscaleTrainingConfig, AdaptiveRuleTrainingHistory,
    mlp::{
        MlpShape, MlpTrainConfig, mlp_weights, npa_weights, train_weighted_mlp,
        train_weighted_scaled_mlp,
    },
};
use crate::{
    AdaptiveNpaModel, AutomataError, AutomataResult, NpaModel,
    adaptive::model::expanded_compatible_rule,
};

#[cfg(feature = "backend_wgpu")]
pub fn train_adaptive_deployment_rule_wgpu(
    model: &mut AdaptiveNpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
    config: &AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<AdaptiveDeploymentRuleTrainingReport> {
    train_rule::<burn::backend::Wgpu<f32>>(model, batch, config, &Default::default(), "burn-wgpu")
}

#[cfg(not(feature = "backend_wgpu"))]
pub fn train_adaptive_deployment_rule_wgpu(
    _model: &mut AdaptiveNpaModel,
    _batch: &AdaptiveMultiscaleTrainingBatch,
    _config: &AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<AdaptiveDeploymentRuleTrainingReport> {
    Err(AutomataError::InvalidArgument(
        "adaptive deployment WGPU training requires backend_wgpu".to_string(),
    ))
}

#[cfg(feature = "backend_cuda")]
pub fn train_adaptive_deployment_rule_cuda(
    model: &mut AdaptiveNpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
    config: &AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<AdaptiveDeploymentRuleTrainingReport> {
    train_rule::<burn::backend::Cuda<f32>>(model, batch, config, &Default::default(), "burn-cuda")
}

#[cfg(not(feature = "backend_cuda"))]
pub fn train_adaptive_deployment_rule_cuda(
    _model: &mut AdaptiveNpaModel,
    _batch: &AdaptiveMultiscaleTrainingBatch,
    _config: &AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<AdaptiveDeploymentRuleTrainingReport> {
    Err(AutomataError::InvalidArgument(
        "adaptive deployment CUDA training requires backend_cuda".to_string(),
    ))
}

#[cfg(feature = "backend_ndarray")]
pub fn train_adaptive_deployment_rule_ndarray(
    model: &mut AdaptiveNpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
    config: &AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<AdaptiveDeploymentRuleTrainingReport> {
    train_rule::<burn::backend::NdArray<f32>>(
        model,
        batch,
        config,
        &Default::default(),
        "burn-ndarray",
    )
}

#[cfg(not(feature = "backend_ndarray"))]
pub fn train_adaptive_deployment_rule_ndarray(
    _model: &mut AdaptiveNpaModel,
    _batch: &AdaptiveMultiscaleTrainingBatch,
    _config: &AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<AdaptiveDeploymentRuleTrainingReport> {
    Err(AutomataError::InvalidArgument(
        "adaptive deployment NdArray training requires backend_ndarray".to_string(),
    ))
}

fn train_rule<B: Backend>(
    model: &mut AdaptiveNpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
    config: &AdaptiveMultiscaleTrainingConfig,
    device: &B::Device,
    backend: &str,
) -> AutomataResult<AdaptiveDeploymentRuleTrainingReport> {
    model.validate()?;
    batch.validate(
        model.rule.config.perception_dims(),
        model.rule.config.update_dims(),
    )?;
    let hidden_dims = config.resolved_deployment_hidden_dims(model.rule.config.hidden_dims);
    let (features, row_weights, functional_target, output_scale, mut rule, label) = match config
        .deployment_strategy
    {
        AdaptiveDeploymentStrategy::Flat => {
            if hidden_dims < model.rule.config.hidden_dims || hidden_dims > 320 {
                return Err(AutomataError::InvalidArgument(format!(
                    "flat adaptive deployment hidden width must be in {}..=320, got {hidden_dims}",
                    model.rule.config.hidden_dims
                )));
            }
            let target = deployment_target(model, batch, config.deployment_target)?;
            let rule = model
                .deployment_rule
                .as_ref()
                .filter(|rule| rule.config.hidden_dims == hidden_dims)
                .cloned()
                .map_or_else(
                    || {
                        expanded_compatible_rule(
                            &model.rule,
                            hidden_dims,
                            config.seed ^ 0xde91_0a11,
                        )
                    },
                    Ok,
                )?;
            (
                batch.deployment_features.clone(),
                batch.deployment_row_weights.clone(),
                target,
                None,
                rule,
                "adaptive flat deployment rule",
            )
        }
        AdaptiveDeploymentStrategy::FusedLocal => {
            let max_local_hidden = 320usize
                .saturating_sub(model.rule.config.hidden_dims)
                .saturating_sub(1);
            if hidden_dims < model.rule.config.hidden_dims || hidden_dims > max_local_hidden {
                return Err(AutomataError::InvalidArgument(format!(
                    "fused-local adaptive deployment hidden width must be in {}..={max_local_hidden}, got {hidden_dims}",
                    model.rule.config.hidden_dims
                )));
            }
            let source = model.local_residual_rule.as_ref().ok_or_else(|| {
                AutomataError::InvalidModel(
                    "fused-local deployment requires the exact local residual rule".to_string(),
                )
            })?;
            let (target, output_scale) = match config.deployment_target {
                AdaptiveDeploymentTarget::Policy => {
                    (adaptive_policy_residual_target(model, batch)?, None)
                }
                AdaptiveDeploymentTarget::RestrictedFineTeacher => (
                    batch.target_update.clone(),
                    Some(local_runtime_output_scale(model)?),
                ),
            };
            let rule = model
                .deployment_local_rule
                .as_ref()
                .filter(|rule| rule.config.hidden_dims == hidden_dims)
                .cloned()
                .map_or_else(
                    || expanded_compatible_rule(source, hidden_dims, config.seed ^ 0x10ca_1de9),
                    Ok,
                )?;
            (
                batch.local_features.clone(),
                batch.row_weights.clone(),
                target,
                output_scale,
                rule,
                "adaptive fused-local deployment rule",
            )
        }
    };
    let initial_validation = validate_rule(
        &rule,
        &features,
        &row_weights,
        batch.rows,
        &functional_target,
        output_scale.as_deref(),
    )?;
    let steps = config.resolved_deployment_steps();
    let shape = MlpShape {
        input_dims: rule.config.perception_dims(),
        hidden_dims,
        output_dims: rule.config.update_dims(),
    };
    let train_config = MlpTrainConfig {
        steps,
        report_interval: config.report_interval.min(steps).max(1),
        optimizer: config.optimizer,
        gradient_reduction_chunk_rows: config.gradient_reduction_chunk_rows,
        optimizer_batch_rows: 0,
    };
    let output = if let Some(output_scale) = output_scale.clone() {
        train_weighted_scaled_mlp::<B>(
            mlp_weights(&rule.weights),
            features.clone(),
            functional_target.clone(),
            row_weights.clone(),
            output_scale,
            batch.rows,
            shape,
            train_config,
            device,
            label,
        )?
    } else {
        train_weighted_mlp::<B>(
            mlp_weights(&rule.weights),
            features.clone(),
            functional_target.clone(),
            row_weights.clone(),
            batch.rows,
            shape,
            train_config,
            device,
            label,
        )?
    };
    rule.weights = npa_weights(output.weights);
    rule.validate()?;
    let trained_validation = validate_rule(
        &rule,
        &features,
        &row_weights,
        batch.rows,
        &functional_target,
        output_scale.as_deref(),
    )?;
    match config.deployment_strategy {
        AdaptiveDeploymentStrategy::Flat => {
            model.deployment_rule = Some(rule);
            model.deployment_local_rule = None;
        }
        AdaptiveDeploymentStrategy::FusedLocal => {
            model.deployment_rule = None;
            model.deployment_local_rule = Some(rule);
        }
    }
    model.validate()?;
    Ok(AdaptiveDeploymentRuleTrainingReport {
        backend: backend.to_string(),
        rows: batch.rows,
        steps,
        hidden_dims,
        target: config.deployment_target,
        initial_validation,
        trained_validation,
        initial_mean_squared_error: output.initial_loss,
        final_mean_squared_error: output.final_loss,
        best_mean_squared_error: output.best_loss,
        training_elapsed_ms: output.elapsed_ms,
        rows_per_second: output.rows_per_second,
        history: output
            .history
            .into_iter()
            .map(|entry| AdaptiveRuleTrainingHistory {
                step: entry.step,
                mean_squared_error: entry.loss,
                gradient_norm: entry.gradient_norm,
                elapsed_ms: entry.elapsed_ms,
            })
            .collect(),
    })
}

pub fn adaptive_deployment_rule_validation(
    model: &AdaptiveNpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
    config: &AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<AdaptiveDeploymentRuleValidationReport> {
    if let Some(rule) = &model.deployment_local_rule {
        let (target, output_scale) = match config.deployment_target {
            AdaptiveDeploymentTarget::Policy => {
                (adaptive_policy_residual_target(model, batch)?, None)
            }
            AdaptiveDeploymentTarget::RestrictedFineTeacher => (
                batch.target_update.clone(),
                Some(local_runtime_output_scale(model)?),
            ),
        };
        return validate_rule(
            rule,
            &batch.local_features,
            &batch.row_weights,
            batch.rows,
            &target,
            output_scale.as_deref(),
        );
    }
    let rule = model.deployment_rule.as_ref().ok_or_else(|| {
        AutomataError::InvalidModel("adaptive deployment rule is not initialized".to_string())
    })?;
    let target = deployment_target(model, batch, config.deployment_target)?;
    validate_rule(
        rule,
        &batch.deployment_features,
        &batch.deployment_row_weights,
        batch.rows,
        &target,
        None,
    )
}

fn deployment_target(
    model: &AdaptiveNpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
    target: AdaptiveDeploymentTarget,
) -> AutomataResult<Vec<f32>> {
    match target {
        AdaptiveDeploymentTarget::Policy => adaptive_policy_target(model, batch),
        AdaptiveDeploymentTarget::RestrictedFineTeacher => {
            Ok(batch.deployment_target_update.clone())
        }
    }
}

fn validate_rule(
    rule: &NpaModel,
    features: &[f32],
    row_weights: &[f32],
    rows: usize,
    functional_target: &[f32],
    output_scale: Option<&[f32]>,
) -> AutomataResult<AdaptiveDeploymentRuleValidationReport> {
    rule.validate()?;
    if features.len() != rows * rule.config.perception_dims()
        || row_weights.len() != rows
        || functional_target.len() != rows * rule.config.update_dims()
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive deployment functional target shape mismatch".to_string(),
        ));
    }
    let mut prediction = rule.forward_update_from_features(features)?;
    let output_dims = rule.config.update_dims();
    if let Some(output_scale) = output_scale {
        for row in prediction.chunks_exact_mut(output_dims) {
            for (value, scale) in row.iter_mut().zip(output_scale) {
                *value *= *scale;
            }
        }
    }
    let denominator = row_weights.iter().sum::<f32>() * output_dims as f32;
    let mut squared_error = 0.0_f64;
    let mut squared_target = 0.0_f64;
    let mut weighted_prediction = Vec::with_capacity(prediction.len());
    let mut weighted_target = Vec::with_capacity(prediction.len());
    for ((prediction, target), weight) in prediction
        .chunks_exact(output_dims)
        .zip(functional_target.chunks_exact(output_dims))
        .zip(row_weights)
    {
        let correlation_weight = weight.sqrt();
        for (&prediction, &target) in prediction.iter().zip(target) {
            squared_error += (*weight * (prediction - target).powi(2)) as f64;
            squared_target += (*weight * target.powi(2)) as f64;
            weighted_prediction.push(correlation_weight * prediction);
            weighted_target.push(correlation_weight * target);
        }
    }
    let denominator = denominator.max(f32::MIN_POSITIVE) as f64;
    let mean_squared_error = (squared_error / denominator) as f32;
    let target_root_mean_square = (squared_target / denominator).sqrt() as f32;
    Ok(AdaptiveDeploymentRuleValidationReport {
        rows,
        mean_squared_error,
        normalized_mean_squared_error: mean_squared_error.sqrt()
            / target_root_mean_square.max(f32::MIN_POSITIVE),
        update_correlation: correlation(&weighted_prediction, &weighted_target),
        target_root_mean_square,
    })
}

fn adaptive_policy_target(
    model: &AdaptiveNpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
) -> AutomataResult<Vec<f32>> {
    let local_rule = model.local_residual_rule.as_ref().ok_or_else(|| {
        AutomataError::InvalidModel(
            "adaptive deployment training requires the trained local residual rule".to_string(),
        )
    })?;
    let base = model
        .rule
        .forward_update_from_features(&batch.deployment_features)?;
    let local = local_rule.forward_update_from_features(&batch.local_features)?;
    let proxy = if model.config.proxy.enabled && model.config.proxy.context_scale > 0.0 {
        model
            .proxy_rule
            .as_ref()
            .ok_or_else(|| {
                AutomataError::InvalidModel(
                    "adaptive deployment training requires the trained proxy rule".to_string(),
                )
            })?
            .forward_update_from_features(&batch.proxy_features)?
    } else {
        vec![0.0; batch.rows * model.rule.config.update_dims()]
    };
    let output_dims = model.rule.config.update_dims();
    Ok((0..batch.rows)
        .flat_map(|row| {
            let gate = batch.deployment_residual_gate[row];
            let start = row * output_dims;
            let end = start + output_dims;
            base[start..end]
                .iter()
                .zip(&local[start..end])
                .zip(&proxy[start..end])
                .enumerate()
                .map(move |(output, ((base, local), proxy))| {
                    *base
                        + gate
                            * (model.config.local_residual_scale
                                * model.config.local_residual_output_scale(output)
                                * *local
                                + model.config.proxy.context_scale * *proxy)
                })
        })
        .collect())
}

fn adaptive_policy_residual_target(
    model: &AdaptiveNpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
) -> AutomataResult<Vec<f32>> {
    if model.config.local_residual_scale <= f32::MIN_POSITIVE {
        return Err(AutomataError::InvalidModel(
            "fused-local deployment requires a positive local residual scale".to_string(),
        ));
    }
    let local_rule = model.local_residual_rule.as_ref().ok_or_else(|| {
        AutomataError::InvalidModel(
            "fused-local deployment requires the exact local residual rule".to_string(),
        )
    })?;
    let local = local_rule.forward_update_from_features(&batch.local_features)?;
    let proxy = if model.config.proxy.enabled && model.config.proxy.context_scale > 0.0 {
        model
            .proxy_rule
            .as_ref()
            .ok_or_else(|| {
                AutomataError::InvalidModel(
                    "fused-local deployment requires the exact proxy rule".to_string(),
                )
            })?
            .forward_update_from_features(&batch.proxy_features)?
    } else {
        vec![0.0; local.len()]
    };
    let proxy_scale = model.config.proxy.context_scale / model.config.local_residual_scale;
    let output_dims = model.rule.config.update_dims();
    Ok(local
        .into_iter()
        .zip(proxy)
        .enumerate()
        .map(|(index, (local, proxy))| {
            model
                .config
                .local_residual_output_scale(index % output_dims)
                * local
                + proxy_scale * proxy
        })
        .collect())
}

fn local_runtime_output_scale(model: &AdaptiveNpaModel) -> AutomataResult<Vec<f32>> {
    if model.config.local_residual_scale <= f32::MIN_POSITIVE {
        return Err(AutomataError::InvalidModel(
            "restricted-fine fused-local deployment requires a positive local residual scale"
                .to_string(),
        ));
    }
    let output_dims = model.rule.config.update_dims();
    Ok((0..output_dims)
        .map(|output| {
            model.config.local_residual_scale * model.config.local_residual_output_scale(output)
        })
        .collect())
}

#[cfg(all(test, feature = "backend_ndarray"))]
mod tests {
    use super::super::AdaptiveMultiscaleDatasetReport;
    use super::*;
    use crate::{AdaptiveNpaConfig, NpaConfig};

    #[test]
    fn restricted_fine_output_scale_matches_runtime_local_gains() {
        let base = NpaModel::seeded(NpaConfig::growing_2d(), 11);
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.proxy.enabled = false;
        adaptive.proxy.context_scale = 0.0;
        adaptive.local_residual_scale = 0.5;
        adaptive.local_residual_motion_scale = 0.0;
        adaptive.local_residual_state_scale = 2.0;
        let model = AdaptiveNpaModel::seeded(base.clone(), adaptive, 13).unwrap();
        let output_dims = base.config.update_dims();
        let scale = local_runtime_output_scale(&model).unwrap();
        assert_eq!(scale[0], 0.0);
        assert_eq!(scale[1], 0.0);
        for value in scale.iter().skip(2).take(output_dims - 2) {
            assert_eq!(*value, 1.0);
        }
    }

    #[test]
    fn fused_local_training_installs_a_distinct_deployment_residual() {
        let base = NpaModel::seeded(NpaConfig::growing_2d(), 31);
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.proxy.enabled = false;
        adaptive.proxy.context_scale = 0.0;
        let mut model = AdaptiveNpaModel::seeded(base.clone(), adaptive, 37).unwrap();
        model.local_residual_rule = Some(NpaModel::seeded(base.config.clone(), 41));
        model.validate().unwrap();
        let rows = 8;
        let input_dims = base.config.perception_dims();
        let output_dims = base.config.update_dims();
        let features = (0..rows * input_dims)
            .map(|index| (index as f32 * 0.013).sin())
            .collect::<Vec<_>>();
        let batch = AdaptiveMultiscaleTrainingBatch {
            local_features: features.clone(),
            closure_features: Vec::new(),
            proxy_features: vec![0.0; rows * input_dims],
            target_update: vec![0.0; rows * output_dims],
            closure_mode_target_update: Vec::new(),
            closure_basis_target_update: Vec::new(),
            closure_mode_row_weights: Vec::new(),
            deployment_features: features,
            deployment_target_update: vec![0.0; rows * output_dims],
            deployment_row_weights: vec![1.0; rows],
            deployment_residual_gate: vec![1.0; rows],
            controller_features: vec![0.0; rows * crate::adaptive::ADAPTIVE_CONTROLLER_INPUT_DIMS],
            controller_targets: vec![0.0; rows * crate::adaptive::ADAPTIVE_CONTROLLER_OUTPUT_DIMS],
            row_weights: vec![1.0; rows],
            rows,
            report: AdaptiveMultiscaleDatasetReport::default(),
        };
        let config = AdaptiveMultiscaleTrainingConfig {
            deployment_strategy: AdaptiveDeploymentStrategy::FusedLocal,
            deployment_hidden_dims: base.config.hidden_dims,
            deployment_steps: 2,
            report_interval: 1,
            ..AdaptiveMultiscaleTrainingConfig::default()
        };
        let report = train_adaptive_deployment_rule_ndarray(&mut model, &batch, &config).unwrap();
        assert_eq!(report.steps, 2);
        assert!(model.deployment_local_rule.is_some());
        assert!(model.deployment_rule.is_none());
        model.validate().unwrap();
    }
}

fn correlation(lhs: &[f32], rhs: &[f32]) -> f32 {
    if lhs.len() != rhs.len() || lhs.is_empty() {
        return 0.0;
    }
    let lhs_mean = lhs.iter().sum::<f32>() / lhs.len() as f32;
    let rhs_mean = rhs.iter().sum::<f32>() / rhs.len() as f32;
    let mut covariance = 0.0_f32;
    let mut lhs_variance = 0.0_f32;
    let mut rhs_variance = 0.0_f32;
    for (&lhs, &rhs) in lhs.iter().zip(rhs) {
        let lhs_delta = lhs - lhs_mean;
        let rhs_delta = rhs - rhs_mean;
        covariance += lhs_delta * rhs_delta;
        lhs_variance += lhs_delta * lhs_delta;
        rhs_variance += rhs_delta * rhs_delta;
    }
    covariance / (lhs_variance * rhs_variance).sqrt().max(f32::MIN_POSITIVE)
}
