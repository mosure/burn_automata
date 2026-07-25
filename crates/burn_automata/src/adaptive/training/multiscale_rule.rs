use burn::tensor::{Tensor, backend::Backend};

use super::{
    AdaptiveMultiscaleRuleStrategy, AdaptiveMultiscaleRuleTrainingReport,
    AdaptiveMultiscaleRuleValidationReport, AdaptiveMultiscaleTrainingBatch,
    AdaptiveMultiscaleTrainingConfig, AdaptiveRuleTrainingHistory,
    dual_mlp::train_dual_mlp,
    mlp::{
        MlpRegressionStats, MlpShape, MlpTrainConfig, MlpTrainingOutput, MlpWeights, mlp_weights,
        npa_weights, tensor_values, tensor2, train_weighted_row_scaled_mlp_with_stats,
        train_weighted_scaled_mlp_with_stats,
    },
};
use crate::{AdaptiveNpaModel, AutomataError, AutomataResult};

struct CompleteRuleTrainingOutput {
    output: MlpTrainingOutput,
    rows: usize,
}

struct CompleteRuleDataset {
    features: Vec<f32>,
    targets: Vec<f32>,
    row_weights: Vec<f32>,
    rows: usize,
    input_dims: usize,
    output_dims: usize,
}

#[cfg(feature = "backend_wgpu")]
pub fn train_adaptive_multiscale_rule_wgpu(
    model: &mut AdaptiveNpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
    config: &AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<AdaptiveMultiscaleRuleTrainingReport> {
    train_rule::<burn::backend::Wgpu<f32>>(model, batch, config, &Default::default(), "burn-wgpu")
}

#[cfg(not(feature = "backend_wgpu"))]
pub fn train_adaptive_multiscale_rule_wgpu(
    _model: &mut AdaptiveNpaModel,
    _batch: &AdaptiveMultiscaleTrainingBatch,
    _config: &AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<AdaptiveMultiscaleRuleTrainingReport> {
    Err(AutomataError::InvalidArgument(
        "adaptive multiscale WGPU training requires backend_wgpu".to_string(),
    ))
}

#[cfg(feature = "backend_cuda")]
pub fn train_adaptive_multiscale_rule_cuda(
    model: &mut AdaptiveNpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
    config: &AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<AdaptiveMultiscaleRuleTrainingReport> {
    train_rule::<burn::backend::Cuda<f32>>(model, batch, config, &Default::default(), "burn-cuda")
}

#[cfg(not(feature = "backend_cuda"))]
pub fn train_adaptive_multiscale_rule_cuda(
    _model: &mut AdaptiveNpaModel,
    _batch: &AdaptiveMultiscaleTrainingBatch,
    _config: &AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<AdaptiveMultiscaleRuleTrainingReport> {
    Err(AutomataError::InvalidArgument(
        "adaptive multiscale CUDA training requires backend_cuda".to_string(),
    ))
}

#[cfg(feature = "backend_ndarray")]
pub fn train_adaptive_multiscale_rule_ndarray(
    model: &mut AdaptiveNpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
    config: &AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<AdaptiveMultiscaleRuleTrainingReport> {
    train_rule::<burn::backend::NdArray<f32>>(
        model,
        batch,
        config,
        &Default::default(),
        "burn-ndarray",
    )
}

#[cfg(not(feature = "backend_ndarray"))]
pub fn train_adaptive_multiscale_rule_ndarray(
    _model: &mut AdaptiveNpaModel,
    _batch: &AdaptiveMultiscaleTrainingBatch,
    _config: &AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<AdaptiveMultiscaleRuleTrainingReport> {
    Err(AutomataError::InvalidArgument(
        "adaptive multiscale NdArray training requires backend_ndarray".to_string(),
    ))
}

fn train_rule<B: Backend>(
    model: &mut AdaptiveNpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
    config: &AdaptiveMultiscaleTrainingConfig,
    device: &B::Device,
    backend: &str,
) -> AutomataResult<AdaptiveMultiscaleRuleTrainingReport> {
    match config.rule_strategy {
        AdaptiveMultiscaleRuleStrategy::Residual => {
            train_residual_rule::<B>(model, batch, config, device, backend)
        }
        AdaptiveMultiscaleRuleStrategy::CoarseReplacement => {
            train_coarse_replacement_rule::<B>(model, batch, config, device, backend)
        }
        AdaptiveMultiscaleRuleStrategy::FullNormalized => {
            train_full_normalized_rule::<B>(model, batch, config, device, backend)
        }
    }
}

fn train_full_normalized_rule<B: Backend>(
    model: &mut AdaptiveNpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
    config: &AdaptiveMultiscaleTrainingConfig,
    device: &B::Device,
    backend: &str,
) -> AutomataResult<AdaptiveMultiscaleRuleTrainingReport> {
    model.validate()?;
    batch.validate(
        model.rule.config.perception_dims(),
        model.rule.config.update_dims(),
    )?;
    if model.config.rule_perception != crate::adaptive::AdaptiveRulePerception::NormalizedAdaptive
        || model.local_residual_rule.is_some()
        || model.proxy_rule.is_some()
        || model.config.closure_moment_features
    {
        return Err(AutomataError::InvalidArgument(
            "full-normalized multiscale training requires normalized-adaptive perception, no residual/proxy rules, and no residual-only closure features"
                .to_string(),
        ));
    }
    let input_dims = model.rule.config.perception_dims();
    if batch.local_features.len() != batch.rows * input_dims {
        return Err(AutomataError::InvalidArgument(format!(
            "full-normalized multiscale features have width {}, expected {input_dims}",
            batch.local_features.len() / batch.rows,
        )));
    }
    let output = train_complete_rule::<B>(
        mlp_weights(&model.rule.weights),
        model.rule.config.hidden_dims,
        batch,
        config,
        false,
        device,
        "adaptive full normalized rule",
    )?;
    model.rule.weights = npa_weights(output.output.weights.clone());
    model.validate()?;
    complete_rule_training_report(
        AdaptiveMultiscaleRuleStrategy::FullNormalized,
        backend,
        config,
        output.output,
        output.rows,
    )
}

fn train_coarse_replacement_rule<B: Backend>(
    model: &mut AdaptiveNpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
    config: &AdaptiveMultiscaleTrainingConfig,
    device: &B::Device,
    backend: &str,
) -> AutomataResult<AdaptiveMultiscaleRuleTrainingReport> {
    model.validate()?;
    batch.validate(
        model.rule.config.perception_dims(),
        model.rule.config.update_dims(),
    )?;
    if model.config.local_rule_semantics
        != crate::adaptive::AdaptiveLocalRuleSemantics::CoarseReplacement
        || model.config.rule_perception != crate::adaptive::AdaptiveRulePerception::NpaCompatible
        || model.proxy_rule.is_some()
    {
        return Err(AutomataError::InvalidArgument(
            "coarse-replacement training requires NPA-compatible base perception, coarse-replacement local semantics, and no proxy"
                .to_string(),
        ));
    }
    let local = model.local_residual_rule.as_ref().ok_or_else(|| {
        AutomataError::InvalidModel(
            "coarse-replacement training requires an initialized local rule".to_string(),
        )
    })?;
    let output = train_complete_rule::<B>(
        mlp_weights(&local.weights),
        local.config.hidden_dims,
        batch,
        config,
        true,
        device,
        "adaptive coarse replacement rule",
    )?;
    model
        .local_residual_rule
        .as_mut()
        .expect("coarse replacement rule checked before training")
        .weights = npa_weights(output.output.weights.clone());
    model.validate()?;
    complete_rule_training_report(
        AdaptiveMultiscaleRuleStrategy::CoarseReplacement,
        backend,
        config,
        output.output,
        output.rows,
    )
}

fn train_complete_rule<B: Backend>(
    weights: MlpWeights,
    hidden_dims: usize,
    batch: &AdaptiveMultiscaleTrainingBatch,
    config: &AdaptiveMultiscaleTrainingConfig,
    coarse_only: bool,
    device: &B::Device,
    name: &str,
) -> AutomataResult<CompleteRuleTrainingOutput> {
    let dataset = complete_rule_dataset(batch, coarse_only)?;
    let output = train_weighted_scaled_mlp_with_stats::<B>(
        weights,
        dataset.features,
        dataset.targets,
        dataset.row_weights,
        vec![1.0; dataset.output_dims],
        dataset.rows,
        MlpShape {
            input_dims: dataset.input_dims,
            hidden_dims,
            output_dims: dataset.output_dims,
        },
        MlpTrainConfig {
            steps: config.steps,
            report_interval: config.report_interval,
            optimizer: config.optimizer,
            gradient_reduction_chunk_rows: config.gradient_reduction_chunk_rows,
            optimizer_batch_rows: 0,
        },
        device,
        name,
    )?;
    Ok(CompleteRuleTrainingOutput {
        output,
        rows: dataset.rows,
    })
}

fn complete_rule_dataset(
    batch: &AdaptiveMultiscaleTrainingBatch,
    coarse_only: bool,
) -> AutomataResult<CompleteRuleDataset> {
    let input_dims = batch.local_features.len() / batch.rows;
    let output_dims = batch.deployment_target_update.len() / batch.rows;
    let selected = (0..batch.rows)
        .filter(|row| !coarse_only || batch.deployment_residual_gate[*row] > 1.0e-6)
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err(AutomataError::InvalidArgument(
            "coarse-replacement training batch contains no coarse rows".to_string(),
        ));
    }
    let mut features = Vec::with_capacity(selected.len() * input_dims);
    let mut targets = Vec::with_capacity(selected.len() * output_dims);
    let mut row_weights = Vec::with_capacity(selected.len());
    for row in selected {
        features.extend_from_slice(&batch.local_features[row * input_dims..(row + 1) * input_dims]);
        targets.extend_from_slice(
            &batch.deployment_target_update[row * output_dims..(row + 1) * output_dims],
        );
        row_weights.push(batch.deployment_row_weights[row]);
    }
    let rows = row_weights.len();
    let mean_weight = row_weights.iter().sum::<f32>() / rows as f32;
    if mean_weight <= f32::MIN_POSITIVE {
        return Err(AutomataError::InvalidArgument(
            "coarse-replacement training rows have zero aggregate weight".to_string(),
        ));
    }
    for weight in &mut row_weights {
        *weight /= mean_weight;
    }
    Ok(CompleteRuleDataset {
        features,
        targets,
        row_weights,
        rows,
        input_dims,
        output_dims,
    })
}

fn complete_rule_training_report(
    strategy: AdaptiveMultiscaleRuleStrategy,
    backend: &str,
    config: &AdaptiveMultiscaleTrainingConfig,
    output: MlpTrainingOutput,
    rows: usize,
) -> AutomataResult<AdaptiveMultiscaleRuleTrainingReport> {
    let initial_validation = local_only_validation_from_stats(
        output.initial_regression_stats.as_ref().ok_or_else(|| {
            AutomataError::InvalidModel(
                "complete-rule training omitted initial regression statistics".to_string(),
            )
        })?,
        rows,
    );
    let trained_validation = local_only_validation_from_stats(
        output.final_regression_stats.as_ref().ok_or_else(|| {
            AutomataError::InvalidModel(
                "complete-rule training omitted final regression statistics".to_string(),
            )
        })?,
        rows,
    );
    Ok(AdaptiveMultiscaleRuleTrainingReport {
        strategy,
        backend: backend.to_string(),
        rows,
        steps: config.steps,
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

fn train_residual_rule<B: Backend>(
    model: &mut AdaptiveNpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
    config: &AdaptiveMultiscaleTrainingConfig,
    device: &B::Device,
    backend: &str,
) -> AutomataResult<AdaptiveMultiscaleRuleTrainingReport> {
    model.validate()?;
    batch.validate(
        model.rule.config.perception_dims(),
        model.rule.config.update_dims(),
    )?;
    let local_residual = model.local_residual_rule.as_ref().ok_or_else(|| {
        AutomataError::InvalidModel(
            "adaptive multiscale training requires an initialized local residual rule".to_string(),
        )
    })?;
    let proxy = model.proxy_rule.as_ref();
    if (config.local_residual_training_scale - model.config.local_residual_scale).abs()
        > 1.0e-6
            * config
                .local_residual_training_scale
                .abs()
                .max(model.config.local_residual_scale.abs())
                .max(1.0)
    {
        return Err(AutomataError::InvalidArgument(format!(
            "adaptive local training gain {} differs from deployed gain {}",
            config.local_residual_training_scale, model.config.local_residual_scale
        )));
    }
    if config.proxy_residual_training_scale > 0.0 {
        let proxy = proxy.ok_or_else(|| {
            AutomataError::InvalidModel(
                "positive proxy training scale requires an initialized proxy rule".to_string(),
            )
        })?;
        if proxy.config.hidden_dims != local_residual.config.hidden_dims
            || proxy.config.perception_dims() != local_residual.config.perception_dims()
        {
            return Err(AutomataError::InvalidArgument(
                "joint local/proxy training requires equal hidden widths; use proxy_residual_training_scale = 0 for a wider exact local branch"
                    .to_string(),
            ));
        }
    }
    let initial_validation = (config.proxy_residual_training_scale != 0.0)
        .then(|| adaptive_multiscale_rule_validation_backend::<B>(model, batch, device))
        .transpose()?;
    let shape = MlpShape {
        input_dims: local_residual.config.perception_dims(),
        hidden_dims: local_residual.config.hidden_dims,
        output_dims: model.rule.config.update_dims(),
    };
    let train_config = MlpTrainConfig {
        steps: config.steps,
        report_interval: config.report_interval,
        optimizer: config.optimizer,
        gradient_reduction_chunk_rows: config.gradient_reduction_chunk_rows,
        optimizer_batch_rows: 0,
    };
    let (
        local_weights,
        proxy_weights,
        initial_loss,
        final_loss,
        best_loss,
        elapsed_ms,
        rows_per_second,
        history,
        initial_regression_stats,
        final_regression_stats,
    ) = if config.proxy_residual_training_scale == 0.0 {
        if config.local_residual_training_scale <= 0.0 {
            return Err(AutomataError::InvalidArgument(
                "adaptive multiscale training requires a positive local gain when the proxy gain is zero"
                    .to_string(),
            ));
        }
        let local_scale = config.local_residual_training_scale;
        let output_scale = (0..shape.output_dims)
            .map(|channel| local_scale * model.config.local_residual_output_scale(channel))
            .collect();
        let output = if model.config.local_rule_semantics
            == crate::adaptive::AdaptiveLocalRuleSemantics::CompatibleResidual
        {
            train_weighted_row_scaled_mlp_with_stats::<B>(
                mlp_weights(&local_residual.weights),
                batch.local_features.clone(),
                batch.target_update.clone(),
                batch.row_weights.clone(),
                output_scale,
                batch.deployment_residual_gate.clone(),
                batch.rows,
                shape,
                train_config,
                device,
                "adaptive compatible residual",
            )?
        } else {
            train_weighted_scaled_mlp_with_stats::<B>(
                mlp_weights(&local_residual.weights),
                batch.local_features.clone(),
                batch.target_update.clone(),
                batch.row_weights.clone(),
                output_scale,
                batch.rows,
                shape,
                train_config,
                device,
                "adaptive local residual",
            )?
        };
        (
            output.weights,
            None,
            output.initial_loss,
            output.final_loss,
            output.best_loss,
            output.elapsed_ms,
            output.rows_per_second,
            output.history,
            output.initial_regression_stats,
            output.final_regression_stats,
        )
    } else {
        let proxy = proxy.expect("positive proxy scale was validated");
        let output = train_dual_mlp::<B>(
            mlp_weights(&local_residual.weights),
            mlp_weights(&proxy.weights),
            batch.local_features.clone(),
            batch.proxy_features.clone(),
            batch.target_update.clone(),
            batch.row_weights.clone(),
            batch.rows,
            shape,
            config.local_residual_training_scale,
            config.proxy_residual_training_scale,
            train_config,
            device,
        )?;
        (
            output.local_weights,
            Some(output.proxy_weights),
            output.initial_loss,
            output.final_loss,
            output.best_loss,
            output.elapsed_ms,
            output.rows_per_second,
            output.history,
            None,
            None,
        )
    };
    model
        .local_residual_rule
        .as_mut()
        .expect("local residual checked before training")
        .weights = npa_weights(local_weights);
    if let Some(proxy_weights) = proxy_weights {
        model
            .proxy_rule
            .as_mut()
            .expect("trained proxy weights require a proxy rule")
            .weights = npa_weights(proxy_weights);
    }
    model.validate()?;
    let (initial_validation, trained_validation) = match (
        initial_regression_stats,
        final_regression_stats,
        initial_validation,
    ) {
        (Some(initial), Some(trained), None) => (
            local_only_validation_from_stats(&initial, batch.rows),
            local_only_validation_from_stats(&trained, batch.rows),
        ),
        (None, None, Some(initial)) => (
            initial,
            adaptive_multiscale_rule_validation_backend::<B>(model, batch, device)?,
        ),
        _ => {
            return Err(AutomataError::InvalidModel(
                "adaptive multiscale validation statistics are incomplete".to_string(),
            ));
        }
    };
    Ok(AdaptiveMultiscaleRuleTrainingReport {
        strategy: AdaptiveMultiscaleRuleStrategy::Residual,
        backend: backend.to_string(),
        rows: batch.rows,
        steps: config.steps,
        initial_validation,
        trained_validation,
        initial_mean_squared_error: initial_loss,
        final_mean_squared_error: final_loss,
        best_mean_squared_error: best_loss,
        training_elapsed_ms: elapsed_ms,
        rows_per_second,
        history: history
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

fn local_only_validation_from_stats(
    stats: &MlpRegressionStats,
    rows: usize,
) -> AdaptiveMultiscaleRuleValidationReport {
    let output_dims = stats.channel_weighted_squared_error.len();
    let weight_sum = stats.channel_weight_sum[0].max(f32::MIN_POSITIVE);
    let weighted_denominator = weight_sum * output_dims as f32;
    let mean_squared_error =
        stats.channel_weighted_squared_error.iter().sum::<f32>() / weighted_denominator;
    let target_root_mean_square =
        (stats.channel_weighted_target_square.iter().sum::<f32>() / weighted_denominator).sqrt();
    let channel_normalized_root_mean_squared_error = stats
        .channel_weighted_squared_error
        .iter()
        .zip(&stats.channel_weighted_target_square)
        .zip(&stats.channel_weight_sum)
        .map(|((error, target), weight)| {
            (error / weight.max(f32::MIN_POSITIVE)).sqrt()
                / (target / weight.max(f32::MIN_POSITIVE))
                    .sqrt()
                    .max(f32::MIN_POSITIVE)
        })
        .collect();
    let values = (rows * output_dims) as f32;
    let prediction_mean = stats.prediction_sum / values;
    let target_mean = stats.target_sum / values;
    let covariance = stats.prediction_target_sum - values * prediction_mean * target_mean;
    let prediction_variance =
        (stats.prediction_square_sum - values * prediction_mean.powi(2)).max(0.0);
    let target_variance = (stats.target_square_sum - values * target_mean.powi(2)).max(0.0);
    let update_correlation = covariance
        / (prediction_variance * target_variance)
            .sqrt()
            .max(f32::MIN_POSITIVE);
    AdaptiveMultiscaleRuleValidationReport {
        rows,
        local_only_mean_squared_error: mean_squared_error,
        combined_mean_squared_error: mean_squared_error,
        normalized_mean_squared_error: mean_squared_error.sqrt()
            / target_root_mean_square.max(f32::MIN_POSITIVE),
        update_correlation,
        proxy_update_root_mean_square: 0.0,
        proxy_relative_mse_gain: 0.0,
        functional_relative_mse_gain: 1.0
            - mean_squared_error / target_root_mean_square.powi(2).max(f32::MIN_POSITIVE),
        channel_normalized_root_mean_squared_error,
    }
}

fn adaptive_multiscale_rule_validation_backend<B: Backend>(
    model: &AdaptiveNpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
    device: &B::Device,
) -> AutomataResult<AdaptiveMultiscaleRuleValidationReport> {
    model.validate()?;
    batch.validate(
        model.rule.config.perception_dims(),
        model.rule.config.update_dims(),
    )?;
    if model.config.rule_perception == crate::adaptive::AdaptiveRulePerception::NormalizedAdaptive
        && model.local_residual_rule.is_none()
        && model.proxy_rule.is_none()
    {
        return complete_rule_validation_backend::<B>(&model.rule, batch, false, device);
    }
    if model.config.local_rule_semantics
        == crate::adaptive::AdaptiveLocalRuleSemantics::CoarseReplacement
    {
        let local = model.local_residual_rule.as_ref().ok_or_else(|| {
            AutomataError::InvalidModel(
                "coarse-replacement validation requires a local rule".to_string(),
            )
        })?;
        return complete_rule_validation_backend::<B>(local, batch, true, device);
    }
    let local_residual_rule = model.local_residual_rule.as_ref().ok_or_else(|| {
        AutomataError::InvalidModel(
            "adaptive multiscale validation requires a local residual rule".to_string(),
        )
    })?;
    let proxy_rule = model.proxy_rule.as_ref();
    let rows = batch.rows;
    let output_dims = model.rule.config.update_dims();
    let local_features = tensor2::<B>(
        batch.local_features.clone(),
        [rows, local_residual_rule.config.perception_dims()],
        device,
    );
    let local_scale = tensor2::<B>(
        (0..output_dims)
            .map(|channel| {
                model.config.local_residual_scale
                    * model.config.local_residual_output_scale(channel)
            })
            .collect(),
        [1, output_dims],
        device,
    );
    let local = local_residual_rule
        .forward_update_tensor(local_features, device)?
        .mul(local_scale.expand([rows, output_dims]));
    let local = if model.config.local_rule_semantics
        == crate::adaptive::AdaptiveLocalRuleSemantics::CompatibleResidual
    {
        local.mul(
            tensor2::<B>(batch.deployment_residual_gate.clone(), [rows, 1], device)
                .expand([rows, output_dims]),
        )
    } else {
        local
    };
    let proxy_scale = model.config.proxy.context_scale;
    let proxy = if proxy_scale > 0.0 {
        let proxy_rule = proxy_rule.ok_or_else(|| {
            AutomataError::InvalidModel(
                "positive proxy context scale requires a proxy rule".to_string(),
            )
        })?;
        let proxy_features = tensor2::<B>(
            batch.proxy_features.clone(),
            [rows, proxy_rule.config.perception_dims()],
            device,
        );
        proxy_rule
            .forward_update_tensor(proxy_features, device)?
            .mul_scalar(proxy_scale)
    } else {
        local.clone().zeros_like()
    };
    let combined = local.clone() + proxy.clone();
    let targets = tensor2::<B>(batch.target_update.clone(), [rows, output_dims], device);
    let row_weights = tensor2::<B>(batch.row_weights.clone(), [rows, 1], device);
    let expanded_weights = row_weights.clone().expand([rows, output_dims]);
    let local_difference = local - targets.clone();
    let combined_difference = combined.clone() - targets.clone();
    let local_error = local_difference
        .clone()
        .mul(local_difference)
        .mul(expanded_weights.clone())
        .sum_dim(0);
    let combined_error = combined_difference
        .clone()
        .mul(combined_difference)
        .mul(expanded_weights.clone())
        .sum_dim(0);
    let weighted_target_square = targets
        .clone()
        .mul(targets.clone())
        .mul(expanded_weights)
        .sum_dim(0);
    let scalar_stats = Tensor::cat(
        vec![
            row_weights.sum().reshape([1, 1]),
            combined.clone().sum().reshape([1, 1]),
            targets.clone().sum().reshape([1, 1]),
            combined.clone().mul(combined.clone()).sum().reshape([1, 1]),
            targets.clone().mul(targets.clone()).sum().reshape([1, 1]),
            combined.clone().mul(targets).sum().reshape([1, 1]),
            proxy.clone().mul(proxy).sum().reshape([1, 1]),
        ],
        1,
    );
    let stats = tensor_values(Tensor::cat(
        vec![
            local_error,
            combined_error,
            weighted_target_square,
            scalar_stats,
        ],
        1,
    ))?;
    let local_error = &stats[..output_dims];
    let combined_error = &stats[output_dims..2 * output_dims];
    let target_square = &stats[2 * output_dims..3 * output_dims];
    let scalars = &stats[3 * output_dims..];
    let weight_sum = scalars[0].max(f32::MIN_POSITIVE);
    let weighted_denominator = weight_sum * output_dims as f32;
    let local_mse = local_error.iter().sum::<f32>() / weighted_denominator;
    let combined_mse = combined_error.iter().sum::<f32>() / weighted_denominator;
    let target_rms = (target_square.iter().sum::<f32>() / weighted_denominator).sqrt();
    let channel_normalized_root_mean_squared_error = combined_error
        .iter()
        .zip(target_square)
        .map(|(error, target)| {
            (error / weight_sum).sqrt() / (target / weight_sum).sqrt().max(f32::MIN_POSITIVE)
        })
        .collect();
    let values = (rows * output_dims) as f32;
    let prediction_mean = scalars[1] / values;
    let target_mean = scalars[2] / values;
    let covariance = scalars[5] - values * prediction_mean * target_mean;
    let prediction_variance = (scalars[3] - values * prediction_mean.powi(2)).max(0.0);
    let target_variance = (scalars[4] - values * target_mean.powi(2)).max(0.0);
    let update_correlation = covariance
        / (prediction_variance * target_variance)
            .sqrt()
            .max(f32::MIN_POSITIVE);
    let proxy_rms = (scalars[6] / values).sqrt();
    Ok(AdaptiveMultiscaleRuleValidationReport {
        rows,
        local_only_mean_squared_error: local_mse,
        combined_mean_squared_error: combined_mse,
        normalized_mean_squared_error: combined_mse.sqrt() / target_rms.max(f32::MIN_POSITIVE),
        update_correlation,
        proxy_update_root_mean_square: proxy_rms,
        proxy_relative_mse_gain: (local_mse - combined_mse) / local_mse.max(f32::MIN_POSITIVE),
        functional_relative_mse_gain: 1.0
            - combined_mse / target_rms.powi(2).max(f32::MIN_POSITIVE),
        channel_normalized_root_mean_squared_error,
    })
}

fn complete_rule_validation_backend<B: Backend>(
    rule: &crate::NpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
    coarse_only: bool,
    device: &B::Device,
) -> AutomataResult<AdaptiveMultiscaleRuleValidationReport> {
    let dataset = complete_rule_dataset(batch, coarse_only)?;
    let rows = dataset.rows;
    let input_dims = rule.config.perception_dims();
    let output_dims = rule.config.update_dims();
    if dataset.input_dims != input_dims {
        return Err(AutomataError::InvalidArgument(
            "complete-rule validation feature width mismatch".to_string(),
        ));
    }
    let prediction = rule.forward_update_tensor(
        tensor2::<B>(dataset.features, [rows, input_dims], device),
        device,
    )?;
    let targets = tensor2::<B>(dataset.targets, [rows, output_dims], device);
    let row_weights = tensor2::<B>(dataset.row_weights, [rows, 1], device);
    let expanded_weights = row_weights.clone().expand([rows, output_dims]);
    let difference = prediction.clone() - targets.clone();
    let weighted_error = difference
        .clone()
        .mul(difference)
        .mul(expanded_weights.clone())
        .sum_dim(0);
    let weighted_target_square = targets
        .clone()
        .mul(targets.clone())
        .mul(expanded_weights)
        .sum_dim(0);
    let scalar_stats = Tensor::cat(
        vec![
            row_weights.sum().reshape([1, 1]),
            prediction.clone().sum().reshape([1, 1]),
            targets.clone().sum().reshape([1, 1]),
            prediction
                .clone()
                .mul(prediction.clone())
                .sum()
                .reshape([1, 1]),
            targets.clone().mul(targets.clone()).sum().reshape([1, 1]),
            prediction.mul(targets).sum().reshape([1, 1]),
        ],
        1,
    );
    let stats = tensor_values(Tensor::cat(
        vec![weighted_error, weighted_target_square, scalar_stats],
        1,
    ))?;
    let error = &stats[..output_dims];
    let target_square = &stats[output_dims..2 * output_dims];
    let scalars = &stats[2 * output_dims..];
    let weight_sum = scalars[0].max(f32::MIN_POSITIVE);
    let denominator = weight_sum * output_dims as f32;
    let mean_squared_error = error.iter().sum::<f32>() / denominator;
    let target_root_mean_square = (target_square.iter().sum::<f32>() / denominator).sqrt();
    let channel_normalized_root_mean_squared_error = error
        .iter()
        .zip(target_square)
        .map(|(error, target)| {
            (error / weight_sum).sqrt() / (target / weight_sum).sqrt().max(f32::MIN_POSITIVE)
        })
        .collect();
    let values = (rows * output_dims) as f32;
    let prediction_mean = scalars[1] / values;
    let target_mean = scalars[2] / values;
    let covariance = scalars[5] - values * prediction_mean * target_mean;
    let prediction_variance = (scalars[3] - values * prediction_mean.powi(2)).max(0.0);
    let target_variance = (scalars[4] - values * target_mean.powi(2)).max(0.0);
    let update_correlation = covariance
        / (prediction_variance * target_variance)
            .sqrt()
            .max(f32::MIN_POSITIVE);
    Ok(AdaptiveMultiscaleRuleValidationReport {
        rows,
        local_only_mean_squared_error: mean_squared_error,
        combined_mean_squared_error: mean_squared_error,
        normalized_mean_squared_error: mean_squared_error.sqrt()
            / target_root_mean_square.max(f32::MIN_POSITIVE),
        update_correlation,
        proxy_update_root_mean_square: 0.0,
        proxy_relative_mse_gain: 0.0,
        functional_relative_mse_gain: 1.0
            - mean_squared_error / target_root_mean_square.powi(2).max(f32::MIN_POSITIVE),
        channel_normalized_root_mean_squared_error,
    })
}

#[cfg(feature = "backend_cuda")]
pub fn adaptive_multiscale_rule_validation_cuda(
    model: &AdaptiveNpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
) -> AutomataResult<AdaptiveMultiscaleRuleValidationReport> {
    adaptive_multiscale_rule_validation_backend::<burn::backend::Cuda<f32>>(
        model,
        batch,
        &Default::default(),
    )
}

#[cfg(not(feature = "backend_cuda"))]
pub fn adaptive_multiscale_rule_validation_cuda(
    _model: &AdaptiveNpaModel,
    _batch: &AdaptiveMultiscaleTrainingBatch,
) -> AutomataResult<AdaptiveMultiscaleRuleValidationReport> {
    Err(AutomataError::InvalidArgument(
        "adaptive multiscale CUDA validation requires backend_cuda".to_string(),
    ))
}

#[cfg(feature = "backend_wgpu")]
pub fn adaptive_multiscale_rule_validation_wgpu(
    model: &AdaptiveNpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
) -> AutomataResult<AdaptiveMultiscaleRuleValidationReport> {
    adaptive_multiscale_rule_validation_backend::<burn::backend::Wgpu<f32>>(
        model,
        batch,
        &Default::default(),
    )
}

#[cfg(not(feature = "backend_wgpu"))]
pub fn adaptive_multiscale_rule_validation_wgpu(
    _model: &AdaptiveNpaModel,
    _batch: &AdaptiveMultiscaleTrainingBatch,
) -> AutomataResult<AdaptiveMultiscaleRuleValidationReport> {
    Err(AutomataError::InvalidArgument(
        "adaptive multiscale WGPU validation requires backend_wgpu".to_string(),
    ))
}

#[cfg(feature = "backend_ndarray")]
pub fn adaptive_multiscale_rule_validation_ndarray(
    model: &AdaptiveNpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
) -> AutomataResult<AdaptiveMultiscaleRuleValidationReport> {
    adaptive_multiscale_rule_validation_backend::<burn::backend::NdArray<f32>>(
        model,
        batch,
        &Default::default(),
    )
}

#[cfg(not(feature = "backend_ndarray"))]
pub fn adaptive_multiscale_rule_validation_ndarray(
    _model: &AdaptiveNpaModel,
    _batch: &AdaptiveMultiscaleTrainingBatch,
) -> AutomataResult<AdaptiveMultiscaleRuleValidationReport> {
    Err(AutomataError::InvalidArgument(
        "adaptive multiscale NdArray validation requires backend_ndarray".to_string(),
    ))
}

pub fn adaptive_multiscale_rule_validation(
    model: &AdaptiveNpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
) -> AutomataResult<AdaptiveMultiscaleRuleValidationReport> {
    model.validate()?;
    batch.validate(
        model.rule.config.perception_dims(),
        model.rule.config.update_dims(),
    )?;
    if model.config.rule_perception == crate::adaptive::AdaptiveRulePerception::NormalizedAdaptive
        && model.local_residual_rule.is_none()
        && model.proxy_rule.is_none()
    {
        return complete_rule_validation(&model.rule, batch, false);
    }
    if model.config.local_rule_semantics
        == crate::adaptive::AdaptiveLocalRuleSemantics::CoarseReplacement
    {
        let local = model.local_residual_rule.as_ref().ok_or_else(|| {
            AutomataError::InvalidModel(
                "coarse-replacement validation requires a local rule".to_string(),
            )
        })?;
        return complete_rule_validation(local, batch, true);
    }
    let local_residual_rule = model.local_residual_rule.as_ref().ok_or_else(|| {
        AutomataError::InvalidModel(
            "adaptive multiscale validation requires a local residual rule".to_string(),
        )
    })?;
    let proxy_rule = model.proxy_rule.as_ref();
    let output_dims = model.rule.config.update_dims();
    let mut local = local_residual_rule
        .forward_update_from_features(&batch.local_features)?
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            model.config.local_residual_scale
                * model
                    .config
                    .local_residual_output_scale(index % output_dims)
                * value
        })
        .collect::<Vec<_>>();
    if model.config.local_rule_semantics
        == crate::adaptive::AdaptiveLocalRuleSemantics::CompatibleResidual
    {
        for (row, output) in local.chunks_exact_mut(output_dims).enumerate() {
            let gate = batch.deployment_residual_gate[row];
            output.iter_mut().for_each(|value| *value *= gate);
        }
    }
    let scale = model.config.proxy.context_scale;
    let proxy = if scale > 0.0 {
        let proxy_rule = proxy_rule.ok_or_else(|| {
            AutomataError::InvalidModel(
                "positive proxy context scale requires a proxy rule".to_string(),
            )
        })?;
        proxy_rule.forward_update_from_features(&batch.proxy_features)?
    } else {
        vec![0.0; local.len()]
    };
    let combined = local
        .iter()
        .zip(&proxy)
        .map(|(local, proxy)| *local + scale * *proxy)
        .collect::<Vec<_>>();
    let weighted_denominator = batch.row_weights.iter().sum::<f32>() * output_dims as f32;
    let weighted_mse = |prediction: &[f32]| {
        prediction
            .chunks_exact(output_dims)
            .zip(batch.target_update.chunks_exact(output_dims))
            .zip(&batch.row_weights)
            .map(|((prediction, target), weight)| {
                *weight
                    * prediction
                        .iter()
                        .zip(target)
                        .map(|(prediction, target)| (*prediction - *target).powi(2))
                        .sum::<f32>()
            })
            .sum::<f32>()
            / weighted_denominator.max(f32::MIN_POSITIVE)
    };
    let local_mse = weighted_mse(&local);
    let combined_mse = weighted_mse(&combined);
    let target_rms = (batch
        .target_update
        .chunks_exact(output_dims)
        .zip(&batch.row_weights)
        .map(|(target, weight)| *weight * target.iter().map(|value| value * value).sum::<f32>())
        .sum::<f32>()
        / weighted_denominator.max(f32::MIN_POSITIVE))
    .sqrt();
    let channel_normalized_root_mean_squared_error = (0..output_dims)
        .map(|channel| {
            let (squared_error, target_square, weight_sum) = combined
                .chunks_exact(output_dims)
                .zip(batch.target_update.chunks_exact(output_dims))
                .zip(&batch.row_weights)
                .fold(
                    (0.0_f32, 0.0_f32, 0.0_f32),
                    |(error_sum, target_sum, weight_sum), ((prediction, target), weight)| {
                        (
                            error_sum + *weight * (prediction[channel] - target[channel]).powi(2),
                            target_sum + *weight * target[channel].powi(2),
                            weight_sum + *weight,
                        )
                    },
                );
            let rms = (squared_error / weight_sum.max(f32::MIN_POSITIVE)).sqrt();
            let target_rms = (target_square / weight_sum.max(f32::MIN_POSITIVE)).sqrt();
            rms / target_rms.max(f32::MIN_POSITIVE)
        })
        .collect();
    let proxy_rms = (proxy
        .iter()
        .map(|value| (scale * *value).powi(2))
        .sum::<f32>()
        / proxy.len().max(1) as f32)
        .sqrt();
    Ok(AdaptiveMultiscaleRuleValidationReport {
        rows: batch.rows,
        local_only_mean_squared_error: local_mse,
        combined_mean_squared_error: combined_mse,
        normalized_mean_squared_error: combined_mse.sqrt() / target_rms.max(f32::MIN_POSITIVE),
        update_correlation: correlation(&combined, &batch.target_update),
        proxy_update_root_mean_square: proxy_rms,
        proxy_relative_mse_gain: (local_mse - combined_mse) / local_mse.max(f32::MIN_POSITIVE),
        functional_relative_mse_gain: 1.0
            - combined_mse / target_rms.powi(2).max(f32::MIN_POSITIVE),
        channel_normalized_root_mean_squared_error,
    })
}

fn complete_rule_validation(
    rule: &crate::NpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
    coarse_only: bool,
) -> AutomataResult<AdaptiveMultiscaleRuleValidationReport> {
    let dataset = complete_rule_dataset(batch, coarse_only)?;
    let input_dims = rule.config.perception_dims();
    let output_dims = rule.config.update_dims();
    if dataset.input_dims != input_dims {
        return Err(AutomataError::InvalidArgument(
            "complete-rule validation feature width mismatch".to_string(),
        ));
    }
    let prediction = rule.forward_update_from_features(&dataset.features)?;
    let target = &dataset.targets;
    let weight_sum = dataset
        .row_weights
        .iter()
        .sum::<f32>()
        .max(f32::MIN_POSITIVE);
    let mut channel_error = vec![0.0_f32; output_dims];
    let mut channel_target = vec![0.0_f32; output_dims];
    for ((prediction, target), weight) in prediction
        .chunks_exact(output_dims)
        .zip(target.chunks_exact(output_dims))
        .zip(&dataset.row_weights)
    {
        for channel in 0..output_dims {
            channel_error[channel] += *weight * (prediction[channel] - target[channel]).powi(2);
            channel_target[channel] += *weight * target[channel].powi(2);
        }
    }
    let denominator = weight_sum * output_dims as f32;
    let mean_squared_error = channel_error.iter().sum::<f32>() / denominator;
    let target_root_mean_square = (channel_target.iter().sum::<f32>() / denominator).sqrt();
    Ok(AdaptiveMultiscaleRuleValidationReport {
        rows: dataset.rows,
        local_only_mean_squared_error: mean_squared_error,
        combined_mean_squared_error: mean_squared_error,
        normalized_mean_squared_error: mean_squared_error.sqrt()
            / target_root_mean_square.max(f32::MIN_POSITIVE),
        update_correlation: correlation(&prediction, target),
        proxy_update_root_mean_square: 0.0,
        proxy_relative_mse_gain: 0.0,
        functional_relative_mse_gain: 1.0
            - mean_squared_error / target_root_mean_square.powi(2).max(f32::MIN_POSITIVE),
        channel_normalized_root_mean_squared_error: channel_error
            .iter()
            .zip(channel_target)
            .map(|(error, target)| {
                (error / weight_sum).sqrt() / (target / weight_sum).sqrt().max(f32::MIN_POSITIVE)
            })
            .collect(),
    })
}

fn correlation(lhs: &[f32], rhs: &[f32]) -> f32 {
    if lhs.len() != rhs.len() || lhs.is_empty() {
        return 0.0;
    }
    let lhs_mean = lhs.iter().sum::<f32>() / lhs.len() as f32;
    let rhs_mean = rhs.iter().sum::<f32>() / rhs.len() as f32;
    let mut covariance = 0.0;
    let mut lhs_variance = 0.0;
    let mut rhs_variance = 0.0;
    for (lhs, rhs) in lhs.iter().zip(rhs) {
        let lhs_delta = *lhs - lhs_mean;
        let rhs_delta = *rhs - rhs_mean;
        covariance += lhs_delta * rhs_delta;
        lhs_variance += lhs_delta * lhs_delta;
        rhs_variance += rhs_delta * rhs_delta;
    }
    covariance / (lhs_variance * rhs_variance).sqrt().max(f32::MIN_POSITIVE)
}

#[cfg(all(test, feature = "backend_ndarray"))]
mod tests {
    use super::*;
    use crate::{
        NpaConfig, NpaModel,
        adaptive::{AdaptiveNpaConfig, AdaptiveRulePerception, adaptive_multiscale_training_batch},
        upstream_growing_2d_hashgrid,
    };

    fn assert_validation_close(
        actual: &AdaptiveMultiscaleRuleValidationReport,
        expected: &AdaptiveMultiscaleRuleValidationReport,
    ) {
        let close = |actual: f32, expected: f32| {
            (actual - expected).abs() <= 2.0e-5 * expected.abs().max(1.0)
        };
        assert_eq!(actual.rows, expected.rows);
        assert!(close(
            actual.local_only_mean_squared_error,
            expected.local_only_mean_squared_error
        ));
        assert!(close(
            actual.combined_mean_squared_error,
            expected.combined_mean_squared_error
        ));
        assert!(close(
            actual.normalized_mean_squared_error,
            expected.normalized_mean_squared_error
        ));
        assert!(close(
            actual.update_correlation,
            expected.update_correlation
        ));
        assert!(close(
            actual.proxy_update_root_mean_square,
            expected.proxy_update_root_mean_square
        ));
        assert!(close(
            actual.proxy_relative_mse_gain,
            expected.proxy_relative_mse_gain
        ));
        assert!(close(
            actual.functional_relative_mse_gain,
            expected.functional_relative_mse_gain
        ));
        assert_eq!(
            actual.channel_normalized_root_mean_squared_error.len(),
            expected.channel_normalized_root_mean_squared_error.len()
        );
        for (&actual, &expected) in actual
            .channel_normalized_root_mean_squared_error
            .iter()
            .zip(&expected.channel_normalized_root_mean_squared_error)
        {
            assert!(close(actual, expected));
        }
    }

    #[test]
    fn dual_local_proxy_training_reduces_measure_weighted_coarse_graining_loss() {
        let teacher = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 7);
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.rule_perception = AdaptiveRulePerception::NpaCompatible;
        adaptive.rule_graph_policy = burn_automata_kernels::AdaptiveGraphPolicy::RawSupport;
        adaptive.proxy.enabled = true;
        adaptive.min_footprint = 0.003;
        adaptive.max_footprint = 0.2;
        let config = AdaptiveMultiscaleTrainingConfig {
            fine_particle_count: 32,
            cut_leaf_counts: vec![8, 16, 32],
            rollout_steps: 1,
            rollouts: 1,
            temporal_samples: 2,
            rows_per_cut: 32,
            validation_rollouts: 1,
            steps: 100,
            report_interval: 20,
            controller_steps: 1,
            optimizer: crate::AdamWConfig {
                learning_rate: 1.0e-4,
                weight_decay: 0.0,
                grad_clip_norm: 5.0,
                ..crate::AdamWConfig::default()
            },
            ..AdaptiveMultiscaleTrainingConfig::default()
        };
        let batch = adaptive_multiscale_training_batch(
            &teacher,
            &upstream_growing_2d_hashgrid(),
            &adaptive,
            &config,
        )
        .unwrap();
        let mut model = AdaptiveNpaModel::seeded(teacher, adaptive, 9).unwrap();
        model.enable_zero_local_residual_rule().unwrap();
        let initial_validation = adaptive_multiscale_rule_validation(&model, &batch).unwrap();
        let report = train_adaptive_multiscale_rule_ndarray(&mut model, &batch, &config).unwrap();
        let trained_validation = adaptive_multiscale_rule_validation(&model, &batch).unwrap();
        assert_validation_close(&report.initial_validation, &initial_validation);
        assert_validation_close(&report.trained_validation, &trained_validation);
        assert!(
            report.final_mean_squared_error < report.initial_mean_squared_error,
            "initial={} final={}",
            report.initial_mean_squared_error,
            report.final_mean_squared_error
        );
        assert!(report.trained_validation.update_correlation.is_finite());
    }

    #[test]
    fn non_unit_local_gain_trains_without_materializing_a_disabled_proxy_branch() {
        let teacher = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 17);
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.rule_perception = AdaptiveRulePerception::NpaCompatible;
        adaptive.rule_graph_policy = burn_automata_kernels::AdaptiveGraphPolicy::RawSupport;
        adaptive.proxy.enabled = true;
        adaptive.proxy.context_scale = 0.0;
        adaptive.local_residual_scale = 0.25;
        adaptive.min_footprint = 0.003;
        adaptive.max_footprint = 0.2;
        let config = AdaptiveMultiscaleTrainingConfig {
            fine_particle_count: 32,
            cut_leaf_counts: vec![8, 16, 32],
            rollout_steps: 1,
            rollouts: 1,
            temporal_samples: 2,
            rows_per_cut: 32,
            validation_rollouts: 1,
            steps: 50,
            report_interval: 10,
            controller_steps: 1,
            local_residual_training_scale: 0.25,
            proxy_residual_training_scale: 0.0,
            optimizer: crate::AdamWConfig {
                learning_rate: 1.0e-3,
                weight_decay: 0.0,
                grad_clip_norm: 5.0,
                ..crate::AdamWConfig::default()
            },
            ..AdaptiveMultiscaleTrainingConfig::default()
        };
        let batch = adaptive_multiscale_training_batch(
            &teacher,
            &upstream_growing_2d_hashgrid(),
            &adaptive,
            &config,
        )
        .unwrap();
        assert!(batch.proxy_features.is_empty());
        let mut model = AdaptiveNpaModel::seeded(teacher, adaptive, 19).unwrap();
        model.enable_zero_local_residual_rule().unwrap();
        let report = train_adaptive_multiscale_rule_ndarray(&mut model, &batch, &config).unwrap();
        assert!(report.final_mean_squared_error < report.initial_mean_squared_error);
    }

    #[test]
    fn full_normalized_training_updates_shared_rule_on_unequal_measure_cuts() {
        let teacher = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 23);
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.rule_perception = AdaptiveRulePerception::NormalizedAdaptive;
        adaptive.rule_graph_policy = burn_automata_kernels::AdaptiveGraphPolicy::RawSupport;
        adaptive.proxy.enabled = false;
        adaptive.local_residual_scale = 0.0;
        adaptive.min_footprint = 0.003;
        adaptive.max_footprint = 0.2;
        let config = AdaptiveMultiscaleTrainingConfig {
            rule_strategy: AdaptiveMultiscaleRuleStrategy::FullNormalized,
            fine_particle_count: 32,
            cut_leaf_counts: vec![8, 16, 32],
            rollout_steps: 1,
            rollouts: 1,
            temporal_samples: 2,
            rows_per_cut: 32,
            validation_rollouts: 1,
            steps: 10,
            report_interval: 5,
            controller_steps: 1,
            optimizer: crate::AdamWConfig {
                learning_rate: 1.0e-3,
                weight_decay: 0.0,
                grad_clip_norm: 5.0,
                ..crate::AdamWConfig::default()
            },
            ..AdaptiveMultiscaleTrainingConfig::default()
        };
        let batch = adaptive_multiscale_training_batch(
            &teacher,
            &upstream_growing_2d_hashgrid(),
            &adaptive,
            &config,
        )
        .unwrap();
        let mut model = AdaptiveNpaModel::seeded(teacher, adaptive, 29).unwrap();
        let initial_validation = adaptive_multiscale_rule_validation(&model, &batch).unwrap();
        let report = train_adaptive_multiscale_rule_ndarray(&mut model, &batch, &config).unwrap();
        let trained_validation = adaptive_multiscale_rule_validation(&model, &batch).unwrap();
        assert_eq!(
            report.strategy,
            AdaptiveMultiscaleRuleStrategy::FullNormalized
        );
        assert_validation_close(&report.initial_validation, &initial_validation);
        assert_validation_close(&report.trained_validation, &trained_validation);
        assert!(
            report.final_mean_squared_error < report.initial_mean_squared_error,
            "initial={} final={}",
            report.initial_mean_squared_error,
            report.final_mean_squared_error
        );
        assert!(report.trained_validation.update_correlation.is_finite());
    }

    #[test]
    fn coarse_replacement_training_updates_only_the_coarse_rule() {
        let teacher = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 31);
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.rule_perception = AdaptiveRulePerception::NpaCompatible;
        adaptive.local_rule_semantics =
            crate::adaptive::AdaptiveLocalRuleSemantics::CoarseReplacement;
        adaptive.closure_moment_features = true;
        adaptive.rule_graph_policy = burn_automata_kernels::AdaptiveGraphPolicy::RawSupport;
        adaptive.proxy.enabled = false;
        adaptive.min_footprint = 0.003;
        adaptive.max_footprint = 0.2;
        adaptive.reference_footprint = 0.2 / 32.0_f32.sqrt();
        adaptive.base_rule_footprint = adaptive.reference_footprint;
        let config = AdaptiveMultiscaleTrainingConfig {
            rule_strategy: AdaptiveMultiscaleRuleStrategy::CoarseReplacement,
            fine_particle_count: 32,
            cut_leaf_counts: vec![8, 16, 32],
            rollout_steps: 1,
            rollouts: 1,
            temporal_samples: 2,
            rows_per_cut: 32,
            validation_rollouts: 1,
            steps: 100,
            report_interval: 20,
            controller_steps: 1,
            optimizer: crate::AdamWConfig {
                learning_rate: 1.0e-4,
                weight_decay: 0.0,
                grad_clip_norm: 5.0,
                ..crate::AdamWConfig::default()
            },
            ..AdaptiveMultiscaleTrainingConfig::default()
        };
        let batch = adaptive_multiscale_training_batch(
            &teacher,
            &upstream_growing_2d_hashgrid(),
            &adaptive,
            &config,
        )
        .unwrap();
        let mut model = AdaptiveNpaModel::seeded(teacher, adaptive, 37).unwrap();
        model.enable_base_initialized_local_rule().unwrap();
        let frozen_base = model.rule.weights.clone();
        let initial_validation = adaptive_multiscale_rule_validation(&model, &batch).unwrap();
        let report = train_adaptive_multiscale_rule_ndarray(&mut model, &batch, &config).unwrap();
        let trained_validation = adaptive_multiscale_rule_validation(&model, &batch).unwrap();

        assert_eq!(
            report.strategy,
            AdaptiveMultiscaleRuleStrategy::CoarseReplacement
        );
        assert!(report.rows < batch.rows);
        assert_eq!(model.rule.weights.w1, frozen_base.w1);
        assert_eq!(model.rule.weights.b1, frozen_base.b1);
        assert_eq!(model.rule.weights.w2, frozen_base.w2);
        assert_eq!(model.rule.weights.b2, frozen_base.b2);
        assert_validation_close(&report.initial_validation, &initial_validation);
        assert_validation_close(&report.trained_validation, &trained_validation);
        assert!(
            report.final_mean_squared_error < report.initial_mean_squared_error,
            "initial={} final={}",
            report.initial_mean_squared_error,
            report.final_mean_squared_error,
        );
    }
}
