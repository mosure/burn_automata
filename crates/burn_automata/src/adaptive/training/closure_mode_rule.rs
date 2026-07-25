use burn::tensor::backend::Backend;

use super::{
    AdaptiveClosureModeTrainingReport, AdaptiveClosureModeValidationReport,
    AdaptiveMultiscaleTrainingBatch, AdaptiveMultiscaleTrainingConfig, AdaptiveRuleTrainingHistory,
    mlp::{
        MlpShape, MlpTrainConfig, mlp_weights, npa_weights, train_weighted_scaled_mlp_with_stats,
    },
};
use crate::{AdaptiveNpaModel, AutomataError, AutomataResult};

#[cfg(feature = "backend_wgpu")]
pub fn train_adaptive_closure_mode_rule_wgpu(
    model: &mut AdaptiveNpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
    config: &AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<AdaptiveClosureModeTrainingReport> {
    train_rule::<burn::backend::Wgpu<f32>>(model, batch, config, &Default::default(), "burn-wgpu")
}

#[cfg(not(feature = "backend_wgpu"))]
pub fn train_adaptive_closure_mode_rule_wgpu(
    _model: &mut AdaptiveNpaModel,
    _batch: &AdaptiveMultiscaleTrainingBatch,
    _config: &AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<AdaptiveClosureModeTrainingReport> {
    Err(AutomataError::InvalidArgument(
        "adaptive closure WGPU training requires backend_wgpu".to_owned(),
    ))
}

#[cfg(feature = "backend_cuda")]
pub fn train_adaptive_closure_mode_rule_cuda(
    model: &mut AdaptiveNpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
    config: &AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<AdaptiveClosureModeTrainingReport> {
    train_rule::<burn::backend::Cuda<f32>>(model, batch, config, &Default::default(), "burn-cuda")
}

#[cfg(not(feature = "backend_cuda"))]
pub fn train_adaptive_closure_mode_rule_cuda(
    _model: &mut AdaptiveNpaModel,
    _batch: &AdaptiveMultiscaleTrainingBatch,
    _config: &AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<AdaptiveClosureModeTrainingReport> {
    Err(AutomataError::InvalidArgument(
        "adaptive closure CUDA training requires backend_cuda".to_owned(),
    ))
}

#[cfg(feature = "backend_ndarray")]
pub fn train_adaptive_closure_mode_rule_ndarray(
    model: &mut AdaptiveNpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
    config: &AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<AdaptiveClosureModeTrainingReport> {
    train_rule::<burn::backend::NdArray<f32>>(
        model,
        batch,
        config,
        &Default::default(),
        "burn-ndarray",
    )
}

#[cfg(not(feature = "backend_ndarray"))]
pub fn train_adaptive_closure_mode_rule_ndarray(
    _model: &mut AdaptiveNpaModel,
    _batch: &AdaptiveMultiscaleTrainingBatch,
    _config: &AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<AdaptiveClosureModeTrainingReport> {
    Err(AutomataError::InvalidArgument(
        "adaptive closure NdArray training requires backend_ndarray".to_owned(),
    ))
}

fn train_rule<B: Backend>(
    model: &mut AdaptiveNpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
    config: &AdaptiveMultiscaleTrainingConfig,
    device: &B::Device,
    backend: &str,
) -> AutomataResult<AdaptiveClosureModeTrainingReport> {
    model.validate()?;
    batch.validate(
        model.rule.config.perception_dims(),
        model.rule.config.update_dims(),
    )?;
    let rule = model.closure_mode_rule.as_ref().ok_or_else(|| {
        AutomataError::InvalidModel(
            "adaptive closure training requires an initialized closure rule".to_owned(),
        )
    })?;
    if batch.closure_mode_target_update.is_empty()
        || batch.closure_features.len() != batch.rows * rule.config.perception_dims()
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive closure training batch has no closure targets or neighbor features"
                .to_owned(),
        ));
    }
    let initial_validation = adaptive_closure_mode_validation(model, batch)?;
    let output_dims = model.rule.config.update_dims();
    let (target_channel_root_mean_square, target_channel_standardization) =
        closure_target_standardization(
            &batch.closure_mode_target_update,
            &batch.closure_mode_row_weights,
            output_dims,
        );
    let mut standardized_weights = mlp_weights(&rule.weights);
    scale_output_coordinates(
        &mut standardized_weights,
        &target_channel_standardization,
        rule.config.hidden_dims,
    );
    let standardized_targets = scale_output_rows(
        &batch.closure_mode_target_update,
        &target_channel_standardization,
        output_dims,
    );
    let output = train_weighted_scaled_mlp_with_stats::<B>(
        standardized_weights,
        batch.closure_features.clone(),
        standardized_targets,
        batch.closure_mode_row_weights.clone(),
        vec![1.0; output_dims],
        batch.rows,
        MlpShape {
            input_dims: rule.config.perception_dims(),
            hidden_dims: rule.config.hidden_dims,
            output_dims,
        },
        MlpTrainConfig {
            steps: config.steps,
            report_interval: config.report_interval,
            optimizer: config.optimizer,
            gradient_reduction_chunk_rows: config.gradient_reduction_chunk_rows,
            optimizer_batch_rows: 0,
        },
        device,
        "adaptive recurrent closure mode",
    )?;
    let mut physical_weights = output.weights;
    unscale_output_coordinates(
        &mut physical_weights,
        &target_channel_standardization,
        rule.config.hidden_dims,
    );
    model
        .closure_mode_rule
        .as_mut()
        .expect("closure rule checked before training")
        .weights = npa_weights(physical_weights);

    let basis_output = if let Some(rule) = model.closure_basis_rule.as_ref() {
        if batch.closure_basis_target_update.is_empty() {
            return Err(AutomataError::InvalidArgument(
                "adaptive closure-basis training batch has no basis targets".to_owned(),
            ));
        }
        let (basis_target_rms, basis_target_scale) = closure_target_standardization(
            &batch.closure_basis_target_update,
            &batch.closure_mode_row_weights,
            output_dims,
        );
        let mut standardized_weights = mlp_weights(&rule.weights);
        scale_output_coordinates(
            &mut standardized_weights,
            &basis_target_scale,
            rule.config.hidden_dims,
        );
        let standardized_targets = scale_output_rows(
            &batch.closure_basis_target_update,
            &basis_target_scale,
            output_dims,
        );
        let active_basis_dims = 4;
        let active_scale = (output_dims as f32 / active_basis_dims as f32).sqrt();
        let mut basis_loss_scale = vec![0.0; output_dims];
        basis_loss_scale[..active_basis_dims].fill(active_scale);
        let standardized_targets =
            scale_output_rows(&standardized_targets, &basis_loss_scale, output_dims);
        let trained = train_weighted_scaled_mlp_with_stats::<B>(
            standardized_weights,
            batch.closure_features.clone(),
            standardized_targets,
            batch.closure_mode_row_weights.clone(),
            basis_loss_scale,
            batch.rows,
            MlpShape {
                input_dims: rule.config.perception_dims(),
                hidden_dims: rule.config.hidden_dims,
                output_dims,
            },
            MlpTrainConfig {
                steps: config.steps,
                report_interval: config.report_interval,
                optimizer: config.optimizer,
                gradient_reduction_chunk_rows: config.gradient_reduction_chunk_rows,
                optimizer_batch_rows: 0,
            },
            device,
            "adaptive recurrent closure basis",
        )?;
        let mut physical_weights = trained.weights.clone();
        unscale_output_coordinates(
            &mut physical_weights,
            &basis_target_scale,
            rule.config.hidden_dims,
        );
        model
            .closure_basis_rule
            .as_mut()
            .expect("closure-basis rule checked before training")
            .weights = npa_weights(physical_weights);
        Some((trained, basis_target_rms, basis_target_scale))
    } else {
        None
    };
    model.validate()?;
    let trained_validation = adaptive_closure_mode_validation(model, batch)?;
    let (
        basis_initial_mean_squared_error,
        basis_final_mean_squared_error,
        basis_best_mean_squared_error,
        basis_training_elapsed_ms,
        basis_rows_per_second,
        basis_target_channel_root_mean_square,
        basis_target_channel_standardization,
        basis_history,
    ) = basis_output.map_or_else(
        || (0.0, 0.0, 0.0, 0.0, 0.0, Vec::new(), Vec::new(), Vec::new()),
        |(output, rms, scale)| {
            (
                output.initial_loss,
                output.final_loss,
                output.best_loss,
                output.elapsed_ms,
                output.rows_per_second,
                rms,
                scale,
                output
                    .history
                    .into_iter()
                    .map(|entry| AdaptiveRuleTrainingHistory {
                        step: entry.step,
                        mean_squared_error: entry.loss,
                        gradient_norm: entry.gradient_norm,
                        elapsed_ms: entry.elapsed_ms,
                    })
                    .collect(),
            )
        },
    );
    Ok(AdaptiveClosureModeTrainingReport {
        backend: backend.to_owned(),
        rows: batch.rows,
        active_rows: trained_validation.active_rows,
        steps: config.steps,
        initial_validation,
        trained_validation,
        initial_mean_squared_error: output.initial_loss,
        final_mean_squared_error: output.final_loss,
        best_mean_squared_error: output.best_loss,
        training_elapsed_ms: output.elapsed_ms,
        rows_per_second: output.rows_per_second,
        target_channel_root_mean_square,
        target_channel_standardization,
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
        basis_initial_mean_squared_error,
        basis_final_mean_squared_error,
        basis_best_mean_squared_error,
        basis_training_elapsed_ms,
        basis_rows_per_second,
        basis_target_channel_root_mean_square,
        basis_target_channel_standardization,
        basis_history,
    })
}

fn closure_target_standardization(
    targets: &[f32],
    row_weights: &[f32],
    output_dims: usize,
) -> (Vec<f32>, Vec<f32>) {
    const MIN_TARGET_RMS: f64 = 1.0e-4;
    const MAX_STANDARDIZATION: f64 = 256.0;

    let mut target_square = vec![0.0_f64; output_dims];
    let mut weight_sum = 0.0_f64;
    for (row, &weight) in row_weights.iter().enumerate() {
        if weight <= 0.0 {
            continue;
        }
        let weight = f64::from(weight);
        weight_sum += weight;
        for channel in 0..output_dims {
            target_square[channel] +=
                weight * f64::from(targets[row * output_dims + channel]).powi(2);
        }
    }
    let divisor = weight_sum.max(f64::MIN_POSITIVE);
    let rms = target_square
        .into_iter()
        .map(|square| (square / divisor).sqrt() as f32)
        .collect::<Vec<_>>();
    let scale = rms
        .iter()
        .map(|value| (1.0 / f64::from(*value).max(MIN_TARGET_RMS)).min(MAX_STANDARDIZATION) as f32)
        .collect();
    (rms, scale)
}

fn scale_output_rows(values: &[f32], scale: &[f32], output_dims: usize) -> Vec<f32> {
    values
        .chunks_exact(output_dims)
        .flat_map(|row| row.iter().zip(scale).map(|(value, scale)| value * scale))
        .collect()
}

fn scale_output_coordinates(
    weights: &mut super::mlp::MlpWeights,
    scale: &[f32],
    hidden_dims: usize,
) {
    for (output, scale) in scale.iter().copied().enumerate() {
        for weight in &mut weights.w2[output * hidden_dims..(output + 1) * hidden_dims] {
            *weight *= scale;
        }
        weights.b2[output] *= scale;
    }
}

fn unscale_output_coordinates(
    weights: &mut super::mlp::MlpWeights,
    scale: &[f32],
    hidden_dims: usize,
) {
    for (output, scale) in scale.iter().copied().enumerate() {
        for weight in &mut weights.w2[output * hidden_dims..(output + 1) * hidden_dims] {
            *weight /= scale;
        }
        weights.b2[output] /= scale;
    }
}

pub fn adaptive_closure_mode_validation(
    model: &AdaptiveNpaModel,
    batch: &AdaptiveMultiscaleTrainingBatch,
) -> AutomataResult<AdaptiveClosureModeValidationReport> {
    model.validate()?;
    batch.validate(
        model.rule.config.perception_dims(),
        model.rule.config.update_dims(),
    )?;
    let rule = model.closure_mode_rule.as_ref().ok_or_else(|| {
        AutomataError::InvalidModel(
            "adaptive closure validation requires an initialized closure rule".to_owned(),
        )
    })?;
    if batch.closure_mode_target_update.is_empty()
        || batch.closure_features.len() != batch.rows * rule.config.perception_dims()
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive closure validation batch has no closure targets or neighbor features"
                .to_owned(),
        ));
    }
    let prediction = rule.forward_update_from_features(&batch.closure_features)?;
    let mut validation = closure_validation(
        &prediction,
        &batch.closure_mode_target_update,
        &batch.closure_mode_row_weights,
        model.rule.config.spatial_dims,
        model.rule.config.state_dims,
    )?;
    if let Some(basis_rule) = &model.closure_basis_rule {
        if batch.closure_basis_target_update.is_empty() {
            return Err(AutomataError::InvalidArgument(
                "adaptive closure validation batch has no basis targets".to_owned(),
            ));
        }
        let basis_prediction = basis_rule.forward_update_from_features(&batch.closure_features)?;
        let basis = closure_channel_metrics(
            &basis_prediction,
            &batch.closure_basis_target_update,
            &batch.closure_mode_row_weights,
            model.rule.config.update_dims(),
            0..4,
        );
        validation.basis_normalized_root_mean_squared_error =
            basis.normalized_root_mean_squared_error;
        validation.basis_update_correlation = basis.update_correlation;
        validation.basis_maximum_absolute_error = basis.maximum_absolute_error;
    }
    Ok(validation)
}

fn closure_validation(
    prediction: &[f32],
    target: &[f32],
    row_weights: &[f32],
    spatial_dims: usize,
    state_dims: usize,
) -> AutomataResult<AdaptiveClosureModeValidationReport> {
    let output_dims = spatial_dims + state_dims;
    if prediction.len() != target.len()
        || target.len() != row_weights.len() * output_dims
        || row_weights.is_empty()
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive closure validation shape mismatch".to_owned(),
        ));
    }
    let overall =
        closure_channel_metrics(prediction, target, row_weights, output_dims, 0..output_dims);
    let phase = closure_channel_metrics(
        prediction,
        target,
        row_weights,
        output_dims,
        0..spatial_dims,
    );
    let state = closure_channel_metrics(
        prediction,
        target,
        row_weights,
        output_dims,
        spatial_dims..output_dims,
    );
    Ok(AdaptiveClosureModeValidationReport {
        active_rows: row_weights.iter().filter(|weight| **weight > 0.0).count(),
        weighted_mean_squared_error: overall.mean_squared_error,
        target_root_mean_square: overall.target_root_mean_square,
        normalized_root_mean_squared_error: overall.normalized_root_mean_squared_error,
        update_correlation: overall.update_correlation,
        maximum_absolute_error: overall.maximum_absolute_error,
        phase_normalized_root_mean_squared_error: phase.normalized_root_mean_squared_error,
        phase_update_correlation: phase.update_correlation,
        state_normalized_root_mean_squared_error: state.normalized_root_mean_squared_error,
        state_update_correlation: state.update_correlation,
        basis_normalized_root_mean_squared_error: 0.0,
        basis_update_correlation: 0.0,
        basis_maximum_absolute_error: 0.0,
    })
}

struct ClosureChannelMetrics {
    mean_squared_error: f32,
    target_root_mean_square: f32,
    normalized_root_mean_squared_error: f32,
    update_correlation: f32,
    maximum_absolute_error: f32,
}

fn closure_channel_metrics(
    prediction: &[f32],
    target: &[f32],
    row_weights: &[f32],
    output_dims: usize,
    channels: std::ops::Range<usize>,
) -> ClosureChannelMetrics {
    let channel_count = channels.len();
    let mut error_square = 0.0_f64;
    let mut target_square = 0.0_f64;
    let mut weight_sum = 0.0_f64;
    let mut maximum_error = 0.0_f32;
    let mut prediction_values = Vec::new();
    let mut target_values = Vec::new();
    for (row, &weight) in row_weights.iter().enumerate() {
        if weight <= 0.0 {
            continue;
        }
        weight_sum += f64::from(weight);
        for channel in channels.clone() {
            let index = row * output_dims + channel;
            let error = prediction[index] - target[index];
            error_square += f64::from(weight) * f64::from(error).powi(2);
            target_square += f64::from(weight) * f64::from(target[index]).powi(2);
            maximum_error = maximum_error.max(error.abs());
            prediction_values.push(prediction[index]);
            target_values.push(target[index]);
        }
    }
    let divisor = (weight_sum * channel_count as f64).max(f64::MIN_POSITIVE);
    let mean_squared_error = (error_square / divisor) as f32;
    let target_root_mean_square = (target_square / divisor).sqrt() as f32;
    ClosureChannelMetrics {
        mean_squared_error,
        target_root_mean_square,
        normalized_root_mean_squared_error: mean_squared_error.sqrt()
            / target_root_mean_square.max(f32::MIN_POSITIVE),
        update_correlation: correlation(&prediction_values, &target_values),
        maximum_absolute_error: maximum_error,
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
        let lhs = lhs - lhs_mean;
        let rhs = rhs - rhs_mean;
        covariance += lhs * rhs;
        lhs_variance += lhs * lhs;
        rhs_variance += rhs * rhs;
    }
    covariance / (lhs_variance * rhs_variance).sqrt().max(f32::MIN_POSITIVE)
}

#[cfg(all(test, feature = "backend_ndarray"))]
mod tests {
    use super::*;
    use crate::{
        NpaConfig, NpaModel,
        adaptive::{AdaptiveNpaConfig, AdaptiveRulePerception},
        upstream_growing_2d_hashgrid,
    };

    #[test]
    fn closure_target_standardization_balances_channels_and_preserves_weights() {
        let targets = vec![3.0, 0.03, 4.0, 0.04, 100.0, 100.0];
        let row_weights = vec![1.0, 1.0, 0.0];
        let (rms, scale) = closure_target_standardization(&targets, &row_weights, 2);
        let standardized = scale_output_rows(&targets, &scale, 2);
        let (standardized_rms, _) = closure_target_standardization(&standardized, &row_weights, 2);

        assert!((rms[0] - 12.5_f32.sqrt()).abs() < 1.0e-6);
        assert!((rms[1] - 0.00125_f32.sqrt()).abs() < 1.0e-6);
        assert!((standardized_rms[0] - 1.0).abs() < 1.0e-6);
        assert!((standardized_rms[1] - 1.0).abs() < 1.0e-6);

        let model = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 19);
        let mut weights = mlp_weights(&model.weights);
        let original = weights.clone();
        let output_scale = vec![2.0; model.config.update_dims()];
        scale_output_coordinates(&mut weights, &output_scale, model.config.hidden_dims);
        unscale_output_coordinates(&mut weights, &output_scale, model.config.hidden_dims);
        assert_eq!(weights.w2, original.w2);
        assert_eq!(weights.b2, original.b2);
    }

    #[test]
    fn closure_head_reduces_teacher_forced_transition_error() {
        let teacher = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 23);
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.rule_perception = AdaptiveRulePerception::NormalizedAdaptive;
        adaptive.proxy.enabled = true;
        adaptive.closure_moment_features = true;
        adaptive.closure_recurrent_mode = true;
        adaptive.min_footprint = 0.003;
        adaptive.max_footprint = 0.2;
        let mut model = AdaptiveNpaModel::seeded(teacher.clone(), adaptive.clone(), 29).unwrap();
        let config = AdaptiveMultiscaleTrainingConfig {
            fine_particle_count: 32,
            cut_leaf_counts: vec![8, 16, 32],
            rollout_steps: 4,
            rollouts: 2,
            temporal_samples: 5,
            rows_per_cut: 32,
            validation_rollouts: 1,
            update_prob: 1.0,
            steps: 50,
            report_interval: 25,
            controller_steps: 1,
            optimizer: crate::AdamWConfig {
                learning_rate: 2.0e-3,
                weight_decay: 0.0,
                grad_clip_norm: 5.0,
                ..crate::AdamWConfig::default()
            },
            ..AdaptiveMultiscaleTrainingConfig::default()
        };
        let batch = super::super::adaptive_multiscale_training_batch(
            &teacher,
            &upstream_growing_2d_hashgrid(),
            &adaptive,
            &config,
        )
        .unwrap();
        let report = train_adaptive_closure_mode_rule_ndarray(&mut model, &batch, &config).unwrap();
        assert!(report.active_rows > 0);
        assert!(
            report.trained_validation.weighted_mean_squared_error
                < report.initial_validation.weighted_mean_squared_error
        );
        assert!(report.final_mean_squared_error < report.initial_mean_squared_error);
        assert!(
            report
                .trained_validation
                .basis_normalized_root_mean_squared_error
                < report
                    .initial_validation
                    .basis_normalized_root_mean_squared_error
        );
        assert!(report.basis_final_mean_squared_error < report.basis_initial_mean_squared_error);
    }
}
