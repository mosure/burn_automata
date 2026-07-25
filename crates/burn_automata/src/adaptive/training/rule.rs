use burn::tensor::backend::Backend;

use super::{
    AdaptiveRuleDistillationConfig, AdaptiveRuleTrainingBatch, AdaptiveRuleTrainingHistory,
    AdaptiveRuleTrainingReport,
    mlp::{MlpShape, MlpTrainConfig, MlpWeights, train_mlp},
};
use crate::{AutomataResult, NpaModel, NpaWeights};

#[cfg(feature = "backend_wgpu")]
pub fn train_adaptive_rule_wgpu(
    rule: &mut NpaModel,
    batch: &AdaptiveRuleTrainingBatch,
    config: AdaptiveRuleDistillationConfig,
) -> AutomataResult<AdaptiveRuleTrainingReport> {
    train_rule::<burn::backend::Wgpu<f32>>(rule, batch, config, &Default::default(), "burn-wgpu")
}

#[cfg(feature = "backend_cuda")]
pub fn train_adaptive_rule_cuda(
    rule: &mut NpaModel,
    batch: &AdaptiveRuleTrainingBatch,
    config: AdaptiveRuleDistillationConfig,
) -> AutomataResult<AdaptiveRuleTrainingReport> {
    train_rule::<burn::backend::Cuda<f32>>(rule, batch, config, &Default::default(), "burn-cuda")
}

#[cfg(feature = "backend_ndarray")]
pub fn train_adaptive_rule_ndarray(
    rule: &mut NpaModel,
    batch: &AdaptiveRuleTrainingBatch,
    config: AdaptiveRuleDistillationConfig,
) -> AutomataResult<AdaptiveRuleTrainingReport> {
    train_rule::<burn::backend::NdArray<f32>>(
        rule,
        batch,
        config,
        &Default::default(),
        "burn-ndarray",
    )
}

fn train_rule<B: Backend>(
    rule: &mut NpaModel,
    batch: &AdaptiveRuleTrainingBatch,
    config: AdaptiveRuleDistillationConfig,
    device: &B::Device,
    backend: &str,
) -> AutomataResult<AdaptiveRuleTrainingReport> {
    rule.validate()?;
    batch.validate(rule.config.perception_dims(), rule.config.update_dims())?;
    let output = train_mlp::<B>(
        MlpWeights {
            w1: rule.weights.w1.clone(),
            b1: rule.weights.b1.clone(),
            w2: rule.weights.w2.clone(),
            b2: rule.weights.b2.clone(),
        },
        batch.features.clone(),
        batch.target_update.clone(),
        batch.rows,
        MlpShape {
            input_dims: rule.config.perception_dims(),
            hidden_dims: rule.config.hidden_dims,
            output_dims: rule.config.update_dims(),
        },
        MlpTrainConfig {
            steps: config.steps,
            report_interval: config.report_interval,
            optimizer: config.optimizer,
            gradient_reduction_chunk_rows: super::mlp::DEFAULT_GRADIENT_REDUCTION_CHUNK_ROWS,
            optimizer_batch_rows: 0,
        },
        super::mlp::MlpObjective::MeanSquared,
        device,
        "adaptive rule distillation",
    )?;
    rule.weights = NpaWeights {
        w1: output.weights.w1,
        b1: output.weights.b1,
        w2: output.weights.w2,
        b2: output.weights.b2,
    };
    rule.validate()?;
    Ok(AdaptiveRuleTrainingReport {
        backend: backend.to_string(),
        rows: batch.rows,
        steps: config.steps,
        initial_mean_squared_error: output.initial_loss,
        final_mean_squared_error: output.final_loss,
        best_mean_squared_error: output.best_loss,
        dataset_generation_ms: batch.generation_elapsed_ms,
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
