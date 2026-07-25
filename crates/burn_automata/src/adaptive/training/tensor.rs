use burn::tensor::{Tensor, activation::relu, activation::sigmoid, backend::Backend};

use super::{
    AdaptiveControllerTrainConfig, AdaptiveControllerTrainingBatch,
    AdaptiveControllerTrainingHistory, AdaptiveControllerTrainingReport,
    AdaptiveRestrictionTrainingBatch,
    mlp::{
        MlpObjective, MlpShape, MlpTrainConfig, MlpWeights, tensor_values, tensor2, train_mlp,
        train_weighted_scaled_topk_mlp,
    },
    restriction_dataset::validate_adaptive_restriction_selection_from_merge_scores,
};
use crate::adaptive::experiments::AdaptiveControllerValidationReport;
use crate::adaptive::{
    ADAPTIVE_CONTROLLER_INPUT_DIMS, ADAPTIVE_CONTROLLER_OUTPUT_DIMS, AdaptiveController,
    AdaptiveControllerWeights,
};
use crate::{AutomataError, AutomataResult};

#[cfg(feature = "backend_wgpu")]
pub fn validate_adaptive_controller_wgpu(
    controller: &AdaptiveController,
    batch: &AdaptiveControllerTrainingBatch,
    split_probability: f32,
    merge_probability: f32,
) -> AutomataResult<AdaptiveControllerValidationReport> {
    validate_controller::<burn::backend::Wgpu<f32>>(
        controller,
        batch,
        split_probability,
        merge_probability,
        &Default::default(),
    )
}

#[cfg(feature = "backend_cuda")]
pub fn validate_adaptive_controller_cuda(
    controller: &AdaptiveController,
    batch: &AdaptiveControllerTrainingBatch,
    split_probability: f32,
    merge_probability: f32,
) -> AutomataResult<AdaptiveControllerValidationReport> {
    validate_controller::<burn::backend::Cuda<f32>>(
        controller,
        batch,
        split_probability,
        merge_probability,
        &Default::default(),
    )
}

#[cfg(feature = "backend_ndarray")]
pub fn validate_adaptive_controller_ndarray(
    controller: &AdaptiveController,
    batch: &AdaptiveControllerTrainingBatch,
    split_probability: f32,
    merge_probability: f32,
) -> AutomataResult<AdaptiveControllerValidationReport> {
    validate_controller::<burn::backend::NdArray<f32>>(
        controller,
        batch,
        split_probability,
        merge_probability,
        &Default::default(),
    )
}

#[cfg(feature = "backend_wgpu")]
pub fn train_adaptive_controller_wgpu(
    controller: &mut AdaptiveController,
    batch: &AdaptiveControllerTrainingBatch,
    config: AdaptiveControllerTrainConfig,
) -> AutomataResult<AdaptiveControllerTrainingReport> {
    train_controller::<burn::backend::Wgpu<f32>>(
        controller,
        batch,
        config,
        &Default::default(),
        "burn-wgpu",
    )
}

#[cfg(feature = "backend_wgpu")]
pub fn train_adaptive_restriction_controller_wgpu(
    controller: &mut AdaptiveController,
    batch: &AdaptiveRestrictionTrainingBatch,
    validation_batch: &AdaptiveRestrictionTrainingBatch,
    config: AdaptiveControllerTrainConfig,
) -> AutomataResult<AdaptiveControllerTrainingReport> {
    train_restriction_controller::<burn::backend::Wgpu<f32>>(
        controller,
        batch,
        validation_batch,
        config,
        &Default::default(),
        "burn-wgpu",
    )
}

#[cfg(feature = "backend_cuda")]
pub fn train_adaptive_controller_cuda(
    controller: &mut AdaptiveController,
    batch: &AdaptiveControllerTrainingBatch,
    config: AdaptiveControllerTrainConfig,
) -> AutomataResult<AdaptiveControllerTrainingReport> {
    train_controller::<burn::backend::Cuda<f32>>(
        controller,
        batch,
        config,
        &Default::default(),
        "burn-cuda",
    )
}

#[cfg(feature = "backend_cuda")]
pub fn train_adaptive_restriction_controller_cuda(
    controller: &mut AdaptiveController,
    batch: &AdaptiveRestrictionTrainingBatch,
    validation_batch: &AdaptiveRestrictionTrainingBatch,
    config: AdaptiveControllerTrainConfig,
) -> AutomataResult<AdaptiveControllerTrainingReport> {
    train_restriction_controller::<burn::backend::Cuda<f32>>(
        controller,
        batch,
        validation_batch,
        config,
        &Default::default(),
        "burn-cuda",
    )
}

#[cfg(feature = "backend_ndarray")]
pub fn train_adaptive_controller_ndarray(
    controller: &mut AdaptiveController,
    batch: &AdaptiveControllerTrainingBatch,
    config: AdaptiveControllerTrainConfig,
) -> AutomataResult<AdaptiveControllerTrainingReport> {
    train_controller::<burn::backend::NdArray<f32>>(
        controller,
        batch,
        config,
        &Default::default(),
        "burn-ndarray",
    )
}

fn validate_controller<B: Backend>(
    controller: &AdaptiveController,
    batch: &AdaptiveControllerTrainingBatch,
    split_probability: f32,
    merge_probability: f32,
    device: &B::Device,
) -> AutomataResult<AdaptiveControllerValidationReport> {
    controller.validate()?;
    batch.validate()?;

    let rows = batch.rows;
    let features = tensor2::<B>(
        batch.features.clone(),
        [rows, ADAPTIVE_CONTROLLER_INPUT_DIMS],
        device,
    );
    let w1 = tensor2::<B>(
        controller.weights.input_weights.clone(),
        [controller.hidden_dims, ADAPTIVE_CONTROLLER_INPUT_DIMS],
        device,
    );
    let b1 = tensor2::<B>(
        controller.weights.input_bias.clone(),
        [1, controller.hidden_dims],
        device,
    );
    let w2 = tensor2::<B>(
        controller.weights.output_weights.clone(),
        [ADAPTIVE_CONTROLLER_OUTPUT_DIMS, controller.hidden_dims],
        device,
    );
    let b2 = tensor2::<B>(
        controller.weights.output_bias.clone(),
        [1, ADAPTIVE_CONTROLLER_OUTPUT_DIMS],
        device,
    );
    let hidden = relu(features.matmul(w1.transpose()) + b1.expand([rows, controller.hidden_dims]));
    let raw = hidden.matmul(w2.transpose()) + b2.expand([rows, ADAPTIVE_CONTROLLER_OUTPUT_DIMS]);
    let targets = tensor2::<B>(
        batch.targets.clone(),
        [rows, ADAPTIVE_CONTROLLER_OUTPUT_DIMS],
        device,
    );
    let regression_mask = tensor2::<B>(
        vec![1.0, 1.0, 0.0, 0.0],
        [1, ADAPTIVE_CONTROLLER_OUTPUT_DIMS],
        device,
    )
    .expand([rows, ADAPTIVE_CONTROLLER_OUTPUT_DIMS]);
    let event_mask = tensor2::<B>(
        vec![0.0, 0.0, 1.0, 1.0],
        [1, ADAPTIVE_CONTROLLER_OUTPUT_DIMS],
        device,
    )
    .expand([rows, ADAPTIVE_CONTROLLER_OUTPUT_DIMS]);
    let prediction =
        raw.clone().mul(regression_mask) + sigmoid(raw.clone()).mul(event_mask.clone());
    let difference = prediction.clone() - targets.clone();
    let channel_squared_error = difference.clone().mul(difference).sum_dim(0);

    let target_positive = targets
        .clone()
        .greater_equal_elem(0.5)
        .float()
        .mul(event_mask.clone());
    let thresholds = tensor2::<B>(
        vec![f32::MAX, f32::MAX, split_probability, merge_probability],
        [1, ADAPTIVE_CONTROLLER_OUTPUT_DIMS],
        device,
    )
    .expand([rows, ADAPTIVE_CONTROLLER_OUTPUT_DIMS]);
    let predicted_positive = prediction
        .clone()
        .greater_equal(thresholds)
        .float()
        .mul(event_mask.clone());
    let true_positive = target_positive.clone().mul(predicted_positive.clone());

    let scale_mask = tensor2::<B>(
        vec![1.0, 0.0, 0.0, 0.0],
        [1, ADAPTIVE_CONTROLLER_OUTPUT_DIMS],
        device,
    )
    .expand([rows, ADAPTIVE_CONTROLLER_OUTPUT_DIMS]);
    let predicted_scale = raw.mul(scale_mask.clone());
    let target_scale = targets.mul(scale_mask);
    let scalar_stats = Tensor::cat(
        vec![
            predicted_scale.clone().sum().reshape([1, 1]),
            target_scale.clone().sum().reshape([1, 1]),
            predicted_scale
                .clone()
                .mul(predicted_scale.clone())
                .sum()
                .reshape([1, 1]),
            target_scale
                .clone()
                .mul(target_scale.clone())
                .sum()
                .reshape([1, 1]),
            predicted_scale.mul(target_scale).sum().reshape([1, 1]),
        ],
        1,
    );
    let stats = tensor_values(Tensor::cat(
        vec![
            channel_squared_error,
            target_positive.sum_dim(0),
            predicted_positive.sum_dim(0),
            true_positive.sum_dim(0),
            scalar_stats,
        ],
        1,
    ))?;

    let channel_mean_squared_error = std::array::from_fn(|channel| stats[channel] / rows as f32);
    let mean_squared_error =
        channel_mean_squared_error.iter().sum::<f32>() / ADAPTIVE_CONTROLLER_OUTPUT_DIMS as f32;
    let target_positive =
        &stats[ADAPTIVE_CONTROLLER_OUTPUT_DIMS..2 * ADAPTIVE_CONTROLLER_OUTPUT_DIMS];
    let predicted_positive =
        &stats[2 * ADAPTIVE_CONTROLLER_OUTPUT_DIMS..3 * ADAPTIVE_CONTROLLER_OUTPUT_DIMS];
    let true_positive =
        &stats[3 * ADAPTIVE_CONTROLLER_OUTPUT_DIMS..4 * ADAPTIVE_CONTROLLER_OUTPUT_DIMS];
    let correlation = &stats[4 * ADAPTIVE_CONTROLLER_OUTPUT_DIMS..];
    let predicted_mean = correlation[0] / rows as f32;
    let target_mean = correlation[1] / rows as f32;
    let covariance = correlation[4] - rows as f32 * predicted_mean * target_mean;
    let predicted_variance = (correlation[2] - rows as f32 * predicted_mean.powi(2)).max(0.0);
    let target_variance = (correlation[3] - rows as f32 * target_mean.powi(2)).max(0.0);
    let desired_scale_correlation =
        covariance / (predicted_variance * target_variance).sqrt().max(1.0e-12);

    Ok(AdaptiveControllerValidationReport {
        rows,
        mean_squared_error,
        channel_mean_squared_error,
        desired_scale_correlation,
        event_positive_fraction: std::array::from_fn(|event| {
            target_positive[event + 2] / rows as f32
        }),
        event_precision: std::array::from_fn(|event| {
            true_positive[event + 2] / predicted_positive[event + 2].max(1.0)
        }),
        event_recall: std::array::from_fn(|event| {
            true_positive[event + 2] / target_positive[event + 2].max(1.0)
        }),
    })
}

#[cfg(feature = "backend_ndarray")]
pub fn train_adaptive_restriction_controller_ndarray(
    controller: &mut AdaptiveController,
    batch: &AdaptiveRestrictionTrainingBatch,
    validation_batch: &AdaptiveRestrictionTrainingBatch,
    config: AdaptiveControllerTrainConfig,
) -> AutomataResult<AdaptiveControllerTrainingReport> {
    train_restriction_controller::<burn::backend::NdArray<f32>>(
        controller,
        batch,
        validation_batch,
        config,
        &Default::default(),
        "burn-ndarray",
    )
}

fn train_restriction_controller<B: Backend>(
    controller: &mut AdaptiveController,
    batch: &AdaptiveRestrictionTrainingBatch,
    validation_batch: &AdaptiveRestrictionTrainingBatch,
    config: AdaptiveControllerTrainConfig,
    device: &B::Device,
    backend: &str,
) -> AutomataResult<AdaptiveControllerTrainingReport> {
    controller.validate()?;
    batch.validate()?;
    validation_batch.validate()?;
    if !config.restriction_cost_utility_weight.is_finite()
        || !(0.0..=1.0).contains(&config.restriction_cost_utility_weight)
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive restriction cost-utility weight must be finite and in [0, 1]".to_owned(),
        ));
    }
    let row_weights = restriction_rank_row_weights(batch, config)?;
    let mut targets = vec![0.0_f32; batch.controller.rows * ADAPTIVE_CONTROLLER_OUTPUT_DIMS];
    for (row, (rank, utility)) in batch
        .oracle_rank_targets
        .iter()
        .copied()
        .zip(batch.oracle_cost_utility_targets.iter().copied())
        .enumerate()
    {
        let target = rank + config.restriction_cost_utility_weight * (utility - rank);
        targets[row * ADAPTIVE_CONTROLLER_OUTPUT_DIMS + 3] = target;
    }
    let output = train_weighted_scaled_topk_mlp::<B>(
        MlpWeights {
            w1: controller.weights.input_weights.clone(),
            b1: controller.weights.input_bias.clone(),
            w2: controller.weights.output_weights.clone(),
            b2: controller.weights.output_bias.clone(),
        },
        batch.controller.features.clone(),
        targets,
        row_weights,
        vec![0.0, 0.0, 0.0, 1.0],
        batch.controller.rows,
        MlpShape {
            input_dims: ADAPTIVE_CONTROLLER_INPUT_DIMS,
            hidden_dims: controller.hidden_dims,
            output_dims: ADAPTIVE_CONTROLLER_OUTPUT_DIMS,
        },
        MlpTrainConfig {
            steps: config.steps,
            report_interval: config.report_interval,
            optimizer: config.optimizer,
            gradient_reduction_chunk_rows: config.gradient_reduction_chunk_rows,
            optimizer_batch_rows: config.optimizer_batch_rows,
        },
        restriction_rank_boundary_target(batch),
        config.restriction_topk_loss_weight,
        config.restriction_topk_temperature,
        device,
        "adaptive restriction ranker",
    )?;
    let validation_features = tensor2::<B>(
        validation_batch.controller.features.clone(),
        [
            validation_batch.controller.rows,
            ADAPTIVE_CONTROLLER_INPUT_DIMS,
        ],
        device,
    );
    let mut selected_step = config.steps;
    let mut selected_mean_regret = None::<f32>;
    let mut selected_worst_regret = None::<f32>;
    let mut selected_criterion = f32::INFINITY;
    let mut selected_weights = output.weights.clone();
    for checkpoint in &output.checkpoints {
        let scores = restriction_merge_scores::<B>(
            validation_features.clone(),
            validation_batch.controller.rows,
            controller.hidden_dims,
            &checkpoint.weights,
            device,
        )?;
        let selection =
            validate_adaptive_restriction_selection_from_merge_scores(&scores, validation_batch)?;
        let criterion =
            selection.mean_normalized_cost_regret + 0.25 * selection.worst_normalized_cost_regret;
        if criterion < selected_criterion {
            selected_criterion = criterion;
            selected_step = checkpoint.step;
            selected_mean_regret = Some(selection.mean_normalized_cost_regret);
            selected_worst_regret = Some(selection.worst_normalized_cost_regret);
            selected_weights = checkpoint.weights.clone();
        }
    }
    controller.weights = controller_weights_from_mlp(selected_weights);
    controller.validate()?;
    Ok(AdaptiveControllerTrainingReport {
        backend: backend.to_string(),
        rows: batch.controller.rows,
        steps: config.steps,
        initial_loss: output.initial_loss,
        final_loss: output.final_loss,
        best_loss: output.best_loss,
        best_step: output.best_step,
        selected_step,
        selected_heldout_normalized_cost_regret: selected_mean_regret,
        selected_heldout_worst_normalized_cost_regret: selected_worst_regret,
        elapsed_ms: output.elapsed_ms,
        rows_per_second: output.rows_per_second,
        optimizer_batch_rows: if config.optimizer_batch_rows == 0 {
            batch.controller.rows
        } else {
            config.optimizer_batch_rows
        },
        event_positive_weights: [1.0, 1.0],
        restriction_rank_boundary_emphasis: config.restriction_rank_boundary_emphasis,
        restriction_rank_boundary_width: config.restriction_rank_boundary_width,
        restriction_topk_loss_weight: config.restriction_topk_loss_weight,
        restriction_topk_temperature: config.restriction_topk_temperature,
        restriction_cost_utility_weight: config.restriction_cost_utility_weight,
        history: output
            .history
            .into_iter()
            .map(|entry| AdaptiveControllerTrainingHistory {
                step: entry.step,
                loss: entry.loss,
                gradient_norm: entry.gradient_norm,
                elapsed_ms: entry.elapsed_ms,
            })
            .collect(),
    })
}

fn controller_weights_from_mlp(weights: MlpWeights) -> AdaptiveControllerWeights {
    AdaptiveControllerWeights {
        input_weights: weights.w1,
        input_bias: weights.b1,
        output_weights: weights.w2,
        output_bias: weights.b2,
    }
}

fn restriction_merge_scores<B: Backend>(
    features: Tensor<B, 2>,
    rows: usize,
    hidden_dims: usize,
    weights: &MlpWeights,
    device: &B::Device,
) -> AutomataResult<Vec<f32>> {
    const SCORE_BATCH_ROWS: usize = 65_536;
    let w1 = tensor2::<B>(
        weights.w1.clone(),
        [hidden_dims, ADAPTIVE_CONTROLLER_INPUT_DIMS],
        device,
    );
    let b1 = tensor2::<B>(weights.b1.clone(), [1, hidden_dims], device);
    let w2 = tensor2::<B>(
        weights.w2.clone(),
        [ADAPTIVE_CONTROLLER_OUTPUT_DIMS, hidden_dims],
        device,
    );
    let b2 = tensor2::<B>(
        weights.b2.clone(),
        [1, ADAPTIVE_CONTROLLER_OUTPUT_DIMS],
        device,
    );
    let mut scores = Vec::with_capacity(rows);
    for start in (0..rows).step_by(SCORE_BATCH_ROWS) {
        let end = (start + SCORE_BATCH_ROWS).min(rows);
        let batch_rows = end - start;
        let input = features
            .clone()
            .slice([start..end, 0..ADAPTIVE_CONTROLLER_INPUT_DIMS]);
        let hidden = relu(
            input.matmul(w1.clone().transpose()) + b1.clone().expand([batch_rows, hidden_dims]),
        );
        let output = hidden.matmul(w2.clone().transpose())
            + b2.clone()
                .expand([batch_rows, ADAPTIVE_CONTROLLER_OUTPUT_DIMS]);
        scores.extend(tensor_values(
            output.slice([0..batch_rows, 3..ADAPTIVE_CONTROLLER_OUTPUT_DIMS]),
        )?);
    }
    Ok(scores)
}

fn restriction_rank_row_weights(
    batch: &AdaptiveRestrictionTrainingBatch,
    config: AdaptiveControllerTrainConfig,
) -> AutomataResult<Vec<f32>> {
    let emphasis = config.restriction_rank_boundary_emphasis;
    let width = config.restriction_rank_boundary_width;
    if !emphasis.is_finite()
        || emphasis < 0.0
        || !width.is_finite()
        || (emphasis > 0.0 && width <= 0.0)
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive restriction rank-boundary weighting must have finite non-negative emphasis and positive width"
                .to_owned(),
        ));
    }
    if emphasis == 0.0 {
        return Ok(vec![1.0; batch.controller.rows]);
    }

    let boundary_target = restriction_rank_boundary_target(batch);
    let mut weights = batch
        .oracle_rank_targets
        .iter()
        .map(|target| {
            let distance = (*target - boundary_target) / width;
            1.0 + emphasis * (-0.5 * distance * distance).exp()
        })
        .collect::<Vec<_>>();
    let mean = weights.iter().sum::<f32>() / weights.len() as f32;
    for weight in &mut weights {
        *weight /= mean;
    }
    Ok(weights)
}

fn restriction_rank_boundary_target(batch: &AdaptiveRestrictionTrainingBatch) -> f32 {
    let denominator = batch.groups_per_snapshot.saturating_sub(1).max(1) as f32;
    let boundary_rank = batch.merges_per_snapshot as f32 - 0.5;
    1.0 - 2.0 * boundary_rank / denominator
}

fn train_controller<B: Backend>(
    controller: &mut AdaptiveController,
    batch: &AdaptiveControllerTrainingBatch,
    config: AdaptiveControllerTrainConfig,
    device: &B::Device,
    backend: &str,
) -> AutomataResult<AdaptiveControllerTrainingReport> {
    controller.validate()?;
    batch.validate()?;
    let event_positive_weights = [
        positive_class_weight(batch, 2),
        positive_class_weight(batch, 3),
    ];
    let output = train_mlp::<B>(
        MlpWeights {
            w1: controller.weights.input_weights.clone(),
            b1: controller.weights.input_bias.clone(),
            w2: controller.weights.output_weights.clone(),
            b2: controller.weights.output_bias.clone(),
        },
        batch.features.clone(),
        batch.targets.clone(),
        batch.rows,
        MlpShape {
            input_dims: ADAPTIVE_CONTROLLER_INPUT_DIMS,
            hidden_dims: controller.hidden_dims,
            output_dims: ADAPTIVE_CONTROLLER_OUTPUT_DIMS,
        },
        MlpTrainConfig {
            steps: config.steps,
            report_interval: config.report_interval,
            optimizer: config.optimizer,
            gradient_reduction_chunk_rows: config.gradient_reduction_chunk_rows,
            optimizer_batch_rows: config.optimizer_batch_rows,
        },
        MlpObjective::Controller {
            event_positive_weights,
        },
        device,
        "adaptive controller",
    )?;
    controller.weights = AdaptiveControllerWeights {
        input_weights: output.weights.w1,
        input_bias: output.weights.b1,
        output_weights: output.weights.w2,
        output_bias: output.weights.b2,
    };
    controller.validate()?;
    Ok(AdaptiveControllerTrainingReport {
        backend: backend.to_string(),
        rows: batch.rows,
        steps: config.steps,
        initial_loss: output.initial_loss,
        final_loss: output.final_loss,
        best_loss: output.best_loss,
        best_step: output.best_step,
        selected_step: config.steps,
        selected_heldout_normalized_cost_regret: None,
        selected_heldout_worst_normalized_cost_regret: None,
        elapsed_ms: output.elapsed_ms,
        rows_per_second: output.rows_per_second,
        optimizer_batch_rows: if config.optimizer_batch_rows == 0 {
            batch.rows
        } else {
            config.optimizer_batch_rows
        },
        event_positive_weights,
        restriction_rank_boundary_emphasis: 0.0,
        restriction_rank_boundary_width: 0.0,
        restriction_topk_loss_weight: 0.0,
        restriction_topk_temperature: 0.0,
        restriction_cost_utility_weight: 0.0,
        history: output
            .history
            .into_iter()
            .map(|entry| AdaptiveControllerTrainingHistory {
                step: entry.step,
                loss: entry.loss,
                gradient_norm: entry.gradient_norm,
                elapsed_ms: entry.elapsed_ms,
            })
            .collect(),
    })
}

fn positive_class_weight(batch: &AdaptiveControllerTrainingBatch, channel: usize) -> f32 {
    let positives = batch
        .targets
        .chunks_exact(ADAPTIVE_CONTROLLER_OUTPUT_DIMS)
        .filter(|row| row[channel] >= 0.5)
        .count();
    if positives == 0 {
        return 1.0;
    }
    let negatives = batch.rows - positives;
    (negatives as f32 / positives as f32).clamp(1.0, 32.0)
}

#[cfg(test)]
mod tests {
    use super::restriction_rank_row_weights;
    use crate::adaptive::{
        ADAPTIVE_CONTROLLER_INPUT_DIMS, ADAPTIVE_CONTROLLER_OUTPUT_DIMS,
        AdaptiveControllerTrainConfig, AdaptiveControllerTrainingBatch,
        AdaptiveRestrictionTrainingBatch,
    };

    #[test]
    fn restriction_rank_weighting_focuses_the_deployed_cut_boundary() {
        let groups = 8;
        let ranks = (0..groups)
            .map(|rank| 1.0 - 2.0 * rank as f32 / (groups - 1) as f32)
            .collect::<Vec<_>>();
        let batch = AdaptiveRestrictionTrainingBatch {
            controller: AdaptiveControllerTrainingBatch {
                features: vec![0.0; groups * ADAPTIVE_CONTROLLER_INPUT_DIMS],
                targets: vec![0.0; groups * ADAPTIVE_CONTROLLER_OUTPUT_DIMS],
                rows: groups,
            },
            oracle_rank_targets: ranks,
            oracle_cost_utility_targets: vec![0.0; groups],
            snapshots: 1,
            groups_per_snapshot: groups,
            merges_per_snapshot: 2,
        };
        let config = AdaptiveControllerTrainConfig {
            restriction_rank_boundary_emphasis: 8.0,
            restriction_rank_boundary_width: 0.125,
            ..AdaptiveControllerTrainConfig::default()
        };

        let weights = restriction_rank_row_weights(&batch, config).unwrap();
        let mean = weights.iter().sum::<f32>() / weights.len() as f32;
        assert!((mean - 1.0).abs() < 1.0e-6);
        assert!(weights[1] > weights[0]);
        assert!(weights[2] > weights[7]);
    }
}
