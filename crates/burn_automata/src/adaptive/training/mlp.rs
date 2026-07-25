use std::time::Instant;

use burn::tensor::{
    ElementConversion, Tensor, TensorData,
    activation::{relu, sigmoid, softplus},
    backend::Backend,
};

use crate::{AdamWConfig, AutomataError, AutomataResult, NpaWeights};

#[derive(Clone, Debug)]
pub(super) struct MlpWeights {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

pub(super) fn mlp_weights(weights: &NpaWeights) -> MlpWeights {
    MlpWeights {
        w1: weights.w1.clone(),
        b1: weights.b1.clone(),
        w2: weights.w2.clone(),
        b2: weights.b2.clone(),
    }
}

pub(super) fn npa_weights(weights: MlpWeights) -> NpaWeights {
    NpaWeights {
        w1: weights.w1,
        b1: weights.b1,
        w2: weights.w2,
        b2: weights.b2,
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct MlpShape {
    pub input_dims: usize,
    pub hidden_dims: usize,
    pub output_dims: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct MlpTrainConfig {
    pub steps: usize,
    pub report_interval: usize,
    pub optimizer: AdamWConfig,
    /// Rows per partial weight-gradient reduction. Zero retains the direct
    /// reference GEMM for parity/benchmarking.
    pub gradient_reduction_chunk_rows: usize,
    /// Rows materialized for one forward/backward update. Zero uses all rows.
    pub optimizer_batch_rows: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum MlpObjective {
    MeanSquared,
    Controller {
        event_positive_weights: [f32; 2],
    },
    TopKRank {
        cutoff: f32,
        classification_weight: f32,
        temperature: f32,
    },
}

#[derive(Clone, Debug)]
pub(super) struct MlpHistoryEntry {
    pub step: usize,
    pub loss: f32,
    pub gradient_norm: f32,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug)]
pub(super) struct MlpTrainingOutput {
    pub weights: MlpWeights,
    pub initial_loss: f32,
    pub final_loss: f32,
    pub best_loss: f32,
    pub best_step: usize,
    pub elapsed_ms: f64,
    pub rows_per_second: f64,
    pub history: Vec<MlpHistoryEntry>,
    pub initial_regression_stats: Option<MlpRegressionStats>,
    pub final_regression_stats: Option<MlpRegressionStats>,
    pub checkpoints: Vec<MlpCheckpoint>,
}

#[derive(Clone, Debug)]
pub(super) struct MlpCheckpoint {
    pub step: usize,
    pub weights: MlpWeights,
}

#[derive(Clone, Debug)]
pub(super) struct MlpRegressionStats {
    pub channel_weighted_squared_error: Vec<f32>,
    pub channel_weighted_target_square: Vec<f32>,
    pub channel_weight_sum: Vec<f32>,
    pub prediction_sum: f32,
    pub target_sum: f32,
    pub prediction_square_sum: f32,
    pub target_square_sum: f32,
    pub prediction_target_sum: f32,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn train_mlp<B: Backend>(
    weights: MlpWeights,
    features: Vec<f32>,
    targets: Vec<f32>,
    rows: usize,
    shape: MlpShape,
    config: MlpTrainConfig,
    objective: MlpObjective,
    device: &B::Device,
    name: &str,
) -> AutomataResult<MlpTrainingOutput> {
    train_mlp_impl::<B>(
        weights, features, targets, None, None, None, rows, shape, config, objective, false,
        device, name,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn train_weighted_mlp<B: Backend>(
    weights: MlpWeights,
    features: Vec<f32>,
    targets: Vec<f32>,
    row_weights: Vec<f32>,
    rows: usize,
    shape: MlpShape,
    config: MlpTrainConfig,
    device: &B::Device,
    name: &str,
) -> AutomataResult<MlpTrainingOutput> {
    train_mlp_impl::<B>(
        weights,
        features,
        targets,
        Some(row_weights),
        None,
        None,
        rows,
        shape,
        config,
        MlpObjective::MeanSquared,
        false,
        device,
        name,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn train_weighted_scaled_mlp<B: Backend>(
    weights: MlpWeights,
    features: Vec<f32>,
    targets: Vec<f32>,
    row_weights: Vec<f32>,
    output_scale: Vec<f32>,
    rows: usize,
    shape: MlpShape,
    config: MlpTrainConfig,
    device: &B::Device,
    name: &str,
) -> AutomataResult<MlpTrainingOutput> {
    train_mlp_impl::<B>(
        weights,
        features,
        targets,
        Some(row_weights),
        Some(output_scale),
        None,
        rows,
        shape,
        config,
        MlpObjective::MeanSquared,
        false,
        device,
        name,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn train_weighted_scaled_topk_mlp<B: Backend>(
    weights: MlpWeights,
    features: Vec<f32>,
    targets: Vec<f32>,
    row_weights: Vec<f32>,
    output_scale: Vec<f32>,
    rows: usize,
    shape: MlpShape,
    config: MlpTrainConfig,
    cutoff: f32,
    classification_weight: f32,
    temperature: f32,
    device: &B::Device,
    name: &str,
) -> AutomataResult<MlpTrainingOutput> {
    if !cutoff.is_finite()
        || !classification_weight.is_finite()
        || classification_weight < 0.0
        || !temperature.is_finite()
        || temperature <= 0.0
    {
        return Err(AutomataError::InvalidArgument(format!(
            "{name} top-k objective requires finite cutoff, non-negative weight, and positive temperature"
        )));
    }
    train_mlp_impl::<B>(
        weights,
        features,
        targets,
        Some(row_weights),
        Some(output_scale),
        None,
        rows,
        shape,
        config,
        MlpObjective::TopKRank {
            cutoff,
            classification_weight,
            temperature,
        },
        false,
        device,
        name,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn train_weighted_scaled_mlp_with_stats<B: Backend>(
    weights: MlpWeights,
    features: Vec<f32>,
    targets: Vec<f32>,
    row_weights: Vec<f32>,
    output_scale: Vec<f32>,
    rows: usize,
    shape: MlpShape,
    config: MlpTrainConfig,
    device: &B::Device,
    name: &str,
) -> AutomataResult<MlpTrainingOutput> {
    train_mlp_impl::<B>(
        weights,
        features,
        targets,
        Some(row_weights),
        Some(output_scale),
        None,
        rows,
        shape,
        config,
        MlpObjective::MeanSquared,
        true,
        device,
        name,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn train_weighted_row_scaled_mlp_with_stats<B: Backend>(
    weights: MlpWeights,
    features: Vec<f32>,
    targets: Vec<f32>,
    row_weights: Vec<f32>,
    output_scale: Vec<f32>,
    row_output_scale: Vec<f32>,
    rows: usize,
    shape: MlpShape,
    config: MlpTrainConfig,
    device: &B::Device,
    name: &str,
) -> AutomataResult<MlpTrainingOutput> {
    train_mlp_impl::<B>(
        weights,
        features,
        targets,
        Some(row_weights),
        Some(output_scale),
        Some(row_output_scale),
        rows,
        shape,
        config,
        MlpObjective::MeanSquared,
        true,
        device,
        name,
    )
}

#[allow(clippy::too_many_arguments)]
fn train_mlp_impl<B: Backend>(
    weights: MlpWeights,
    features: Vec<f32>,
    targets: Vec<f32>,
    row_weights: Option<Vec<f32>>,
    output_scale: Option<Vec<f32>>,
    row_output_scale: Option<Vec<f32>>,
    rows: usize,
    shape: MlpShape,
    config: MlpTrainConfig,
    objective: MlpObjective,
    capture_regression_stats: bool,
    device: &B::Device,
    name: &str,
) -> AutomataResult<MlpTrainingOutput> {
    validate(
        &weights,
        &features,
        &targets,
        row_weights.as_deref(),
        output_scale.as_deref(),
        row_output_scale.as_deref(),
        rows,
        shape,
        config,
        name,
    )?;
    let optimizer_batch_rows = effective_optimizer_batch_rows(rows, config.optimizer_batch_rows);
    let mut trainer = MlpTensorTrainer::<B>::new(
        weights,
        features,
        targets,
        row_weights,
        output_scale,
        row_output_scale,
        rows,
        shape,
        objective,
        optimizer_batch_rows,
        device,
    );
    let initial_regression_stats = capture_regression_stats
        .then(|| trainer.regression_stats())
        .transpose()?;
    let initial_loss = trainer.loss(name)?;
    let mut final_loss = initial_loss;
    let mut best_loss = initial_loss;
    let mut best_step = 0;
    let mut history = Vec::new();
    let mut checkpoints = if matches!(objective, MlpObjective::TopKRank { .. }) {
        vec![MlpCheckpoint {
            step: 0,
            weights: trainer.weights()?,
        }]
    } else {
        Vec::new()
    };
    let started = Instant::now();
    for step in 1..=config.steps {
        let capture =
            step == 1 || step == config.steps || step.is_multiple_of(config.report_interval);
        let (loss, gradient_norm) = trainer.step(
            config.optimizer,
            config.gradient_reduction_chunk_rows,
            capture,
            name,
        )?;
        if let Some(loss) = loss {
            final_loss = loss;
            if loss < best_loss {
                best_loss = loss;
                best_step = step;
            }
            history.push(MlpHistoryEntry {
                step,
                loss,
                gradient_norm: gradient_norm.unwrap_or_default(),
                elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
            });
            if matches!(objective, MlpObjective::TopKRank { .. }) {
                checkpoints.push(MlpCheckpoint {
                    step,
                    weights: trainer.weights()?,
                });
            }
        }
    }
    let elapsed = started.elapsed();
    let elapsed_seconds = elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
    let final_regression_stats = capture_regression_stats
        .then(|| trainer.regression_stats())
        .transpose()?;
    Ok(MlpTrainingOutput {
        weights: trainer.into_weights()?,
        initial_loss,
        final_loss,
        best_loss,
        best_step,
        elapsed_ms: elapsed_seconds * 1_000.0,
        rows_per_second: optimizer_batch_rows as f64 * config.steps as f64 / elapsed_seconds,
        history,
        initial_regression_stats,
        final_regression_stats,
        checkpoints,
    })
}

fn effective_optimizer_batch_rows(rows: usize, requested: usize) -> usize {
    if requested == 0 { rows } else { requested }
}

fn tensor_bytes(rows: usize, dims: usize) -> usize {
    rows.checked_mul(dims)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .unwrap_or(usize::MAX)
}

pub(super) fn validate_mlp_buffer_plan(
    rows: usize,
    shape: MlpShape,
    requested_batch_rows: usize,
    name: &str,
) -> AutomataResult<usize> {
    if rows == 0
        || shape.input_dims == 0
        || shape.hidden_dims == 0
        || shape.output_dims == 0
        || requested_batch_rows > rows
        || (requested_batch_rows != 0 && !rows.is_multiple_of(requested_batch_rows))
    {
        return Err(AutomataError::InvalidArgument(format!(
            "{name} MLP optimizer_batch_rows must be zero or a divisor of {rows}",
        )));
    }
    let optimizer_batch_rows = effective_optimizer_batch_rows(rows, requested_batch_rows);
    let largest_buffer_bytes = [
        tensor_bytes(rows, shape.input_dims),
        tensor_bytes(rows, shape.output_dims),
        tensor_bytes(optimizer_batch_rows, shape.hidden_dims),
        tensor_bytes(optimizer_batch_rows, shape.output_dims),
    ]
    .into_iter()
    .max()
    .unwrap_or_default();
    if largest_buffer_bytes > MAX_MLP_SINGLE_BUFFER_BYTES {
        return Err(AutomataError::InvalidArgument(format!(
            "{name} MLP would materialize a {:.2} GiB buffer, above the {:.2} GiB backend-safe limit; set optimizer_batch_rows to a divisor of {rows}",
            largest_buffer_bytes as f64 / (1_u64 << 30) as f64,
            MAX_MLP_SINGLE_BUFFER_BYTES as f64 / (1_u64 << 30) as f64,
        )));
    }
    Ok(optimizer_batch_rows)
}

#[allow(clippy::too_many_arguments)]
fn validate(
    weights: &MlpWeights,
    features: &[f32],
    targets: &[f32],
    row_weights: Option<&[f32]>,
    output_scale: Option<&[f32]>,
    row_output_scale: Option<&[f32]>,
    rows: usize,
    shape: MlpShape,
    config: MlpTrainConfig,
    name: &str,
) -> AutomataResult<()> {
    if rows == 0
        || shape.input_dims == 0
        || shape.hidden_dims == 0
        || shape.output_dims == 0
        || config.steps == 0
        || config.report_interval == 0
        || config.optimizer_batch_rows > rows
        || (config.optimizer_batch_rows != 0 && !rows.is_multiple_of(config.optimizer_batch_rows))
        || features.len() != rows * shape.input_dims
        || targets.len() != rows * shape.output_dims
        || weights.w1.len() != shape.hidden_dims * shape.input_dims
        || weights.b1.len() != shape.hidden_dims
        || weights.w2.len() != shape.output_dims * shape.hidden_dims
        || weights.b2.len() != shape.output_dims
        || row_weights.is_some_and(|weights| weights.len() != rows)
        || output_scale.is_some_and(|scale| scale.len() != shape.output_dims)
        || row_output_scale.is_some_and(|scale| scale.len() != rows)
    {
        return Err(AutomataError::InvalidArgument(format!(
            "{name} MLP training shape/config mismatch"
        )));
    }
    validate_mlp_buffer_plan(rows, shape, config.optimizer_batch_rows, name)?;
    if features
        .iter()
        .chain(targets)
        .chain(&weights.w1)
        .chain(&weights.b1)
        .chain(&weights.w2)
        .chain(&weights.b2)
        .any(|value| !value.is_finite())
        || row_weights.is_some_and(|weights| {
            weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight < 0.0)
                || weights.iter().sum::<f32>() <= f32::MIN_POSITIVE
        })
        || output_scale.is_some_and(|scale| {
            scale.iter().any(|value| !value.is_finite() || *value < 0.0)
                || scale.iter().sum::<f32>() <= f32::MIN_POSITIVE
        })
        || row_output_scale.is_some_and(|scale| {
            scale.iter().any(|value| !value.is_finite() || *value < 0.0)
                || scale.iter().sum::<f32>() <= f32::MIN_POSITIVE
        })
    {
        return Err(AutomataError::InvalidArgument(format!(
            "{name} MLP training contains non-finite values"
        )));
    }
    Ok(())
}

struct MlpTensorTrainer<B: Backend> {
    features: Tensor<B, 2>,
    targets: Tensor<B, 2>,
    row_weights: Option<Tensor<B, 2>>,
    output_scale: Option<Tensor<B, 2>>,
    row_output_scale: Option<Tensor<B, 2>>,
    w1: Tensor<B, 2>,
    b1: Tensor<B, 2>,
    w2: Tensor<B, 2>,
    b2: Tensor<B, 2>,
    moments: [Tensor<B, 2>; 8],
    rows: usize,
    output_dims: usize,
    hidden_dims: usize,
    optimizer_batch_rows: usize,
    objective: MlpObjective,
    controller_objective: Option<ControllerObjectiveTensors<B>>,
    step: usize,
}

// The adaptive MLPs have a very small output width and tens of thousands of
// training rows. A direct [D, rows] x [rows, H] GEMM exposes only a handful of
// output tiles, leaving most of a large GPU idle while those tiles serially
// reduce the full row dimension. Split that reduction into independent batch
// matmuls, then reduce their small partial-gradient tensor.
pub(super) const DEFAULT_GRADIENT_REDUCTION_CHUNK_ROWS: usize = 1_024;
const MAX_MLP_SINGLE_BUFFER_BYTES: usize = 2_000_000_000;

struct ControllerObjectiveTensors<B: Backend> {
    regression_mask: Tensor<B, 2>,
    event_mask: Tensor<B, 2>,
    positive_weights: Tensor<B, 2>,
}

impl<B: Backend> MlpTensorTrainer<B> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        weights: MlpWeights,
        features: Vec<f32>,
        targets: Vec<f32>,
        row_weights: Option<Vec<f32>>,
        output_scale: Option<Vec<f32>>,
        row_output_scale: Option<Vec<f32>>,
        rows: usize,
        shape: MlpShape,
        objective: MlpObjective,
        optimizer_batch_rows: usize,
        device: &B::Device,
    ) -> Self {
        let w1 = tensor2(weights.w1, [shape.hidden_dims, shape.input_dims], device);
        let b1 = tensor2(weights.b1, [1, shape.hidden_dims], device);
        let w2 = tensor2(weights.w2, [shape.output_dims, shape.hidden_dims], device);
        let b2 = tensor2(weights.b2, [1, shape.output_dims], device);
        let moments = [
            w1.clone().zeros_like(),
            w1.clone().zeros_like(),
            b1.clone().zeros_like(),
            b1.clone().zeros_like(),
            w2.clone().zeros_like(),
            w2.clone().zeros_like(),
            b2.clone().zeros_like(),
            b2.clone().zeros_like(),
        ];
        let controller_objective = match objective {
            MlpObjective::MeanSquared | MlpObjective::TopKRank { .. } => None,
            MlpObjective::Controller {
                event_positive_weights,
            } => Some(ControllerObjectiveTensors {
                regression_mask: controller_regression_mask::<B>(optimizer_batch_rows, device),
                event_mask: controller_event_mask::<B>(optimizer_batch_rows, device),
                positive_weights: controller_positive_weights::<B>(
                    optimizer_batch_rows,
                    event_positive_weights,
                    device,
                ),
            }),
        };
        Self {
            features: tensor2(features, [rows, shape.input_dims], device),
            targets: tensor2(targets, [rows, shape.output_dims], device),
            row_weights: row_weights.map(|weights| tensor2(weights, [rows, 1], device)),
            output_scale: output_scale.map(|scale| tensor2(scale, [1, shape.output_dims], device)),
            row_output_scale: row_output_scale.map(|scale| tensor2(scale, [rows, 1], device)),
            w1,
            b1,
            w2,
            b2,
            moments,
            rows,
            output_dims: shape.output_dims,
            hidden_dims: shape.hidden_dims,
            optimizer_batch_rows,
            objective,
            controller_objective,
            step: 0,
        }
    }

    fn batch_count(&self) -> usize {
        self.rows / self.optimizer_batch_rows
    }

    fn batch_start(&self, batch: usize) -> usize {
        batch * self.optimizer_batch_rows
    }

    fn slice_rows(&self, tensor: Tensor<B, 2>, start: usize, dims: usize) -> Tensor<B, 2> {
        tensor.slice([start..start + self.optimizer_batch_rows, 0..dims])
    }

    fn forward_batch(
        &self,
        start: usize,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let rows = self.optimizer_batch_rows;
        let features = self.slice_rows(self.features.clone(), start, self.features.dims()[1]);
        let pre_hidden = features.clone().matmul(self.w1.clone().transpose())
            + self.b1.clone().expand([rows, self.hidden_dims]);
        let hidden = relu(pre_hidden.clone());
        let output = hidden.clone().matmul(self.w2.clone().transpose())
            + self.b2.clone().expand([rows, self.output_dims]);
        let output = if let Some(scale) = &self.output_scale {
            output.mul(scale.clone().expand([rows, self.output_dims]))
        } else {
            output
        };
        let output = if let Some(scale) = &self.row_output_scale {
            output.mul(
                self.slice_rows(scale.clone(), start, 1)
                    .expand([rows, self.output_dims]),
            )
        } else {
            output
        };
        (features, pre_hidden, hidden, output)
    }

    fn loss(&self, name: &str) -> AutomataResult<f32> {
        let mut loss = 0.0_f32;
        for batch in 0..self.batch_count() {
            loss += self.loss_batch(self.batch_start(batch), name)?;
        }
        Ok(loss / self.batch_count() as f32)
    }

    fn loss_batch(&self, start: usize, name: &str) -> AutomataResult<f32> {
        let rows = self.optimizer_batch_rows;
        let output = self.forward_batch(start).3;
        let targets = self.slice_rows(self.targets.clone(), start, self.output_dims);
        let loss = match self.objective {
            MlpObjective::MeanSquared => {
                let difference = output - targets;
                let squared = difference.clone().mul(difference);
                match &self.row_weights {
                    Some(weights) => squared
                        .mul(
                            self.slice_rows(weights.clone(), start, 1)
                                .expand([rows, self.output_dims]),
                        )
                        .mean(),
                    None => squared.mean(),
                }
            }
            MlpObjective::TopKRank {
                cutoff,
                classification_weight,
                temperature,
            } => {
                let expanded_weights = self.row_weights.as_ref().map_or_else(
                    || output.clone().zeros_like().add_scalar(1.0),
                    |weights| {
                        self.slice_rows(weights.clone(), start, 1)
                            .expand([rows, self.output_dims])
                    },
                );
                let difference = output.clone() - targets.clone();
                let rank_loss = difference
                    .clone()
                    .mul(difference)
                    .mul(expanded_weights.clone())
                    .mean();
                let mask = self
                    .output_scale
                    .as_ref()
                    .expect("top-k rank objective requires an output mask")
                    .clone()
                    .expand([rows, self.output_dims]);
                let labels = targets.greater_equal_elem(cutoff).float();
                let logits = (output - cutoff).div_scalar(temperature);
                let one = labels.clone().ones_like();
                let classification_loss = (labels.clone().mul(softplus(logits.clone().neg(), 1.0))
                    + (one - labels).mul(softplus(logits, 1.0)))
                .mul(mask)
                .mul(expanded_weights)
                .sum()
                .div_scalar(rows as f32);
                rank_loss + classification_loss.mul_scalar(classification_weight)
            }
            MlpObjective::Controller {
                event_positive_weights: _,
            } => {
                let objective = self
                    .controller_objective
                    .as_ref()
                    .expect("controller objective tensors initialized");
                let difference = output.clone() - targets.clone();
                let regression = difference
                    .clone()
                    .mul(difference)
                    .mul(objective.regression_mask.clone())
                    .sum()
                    .div_scalar((rows * 2) as f32);
                let one = targets.clone().ones_like();
                let positive = targets
                    .clone()
                    .mul(objective.positive_weights.clone())
                    .mul(softplus(output.clone().neg(), 1.0));
                let negative = (one - targets).mul(softplus(output, 1.0));
                regression
                    + (positive + negative)
                        .mul(objective.event_mask.clone())
                        .sum()
                        .div_scalar((rows * 2) as f32)
            }
        };
        scalar::<B>(loss, &format!("{name} loss"))
    }

    fn regression_stats(&self) -> AutomataResult<MlpRegressionStats> {
        let mut total = MlpRegressionStats {
            channel_weighted_squared_error: vec![0.0; self.output_dims],
            channel_weighted_target_square: vec![0.0; self.output_dims],
            channel_weight_sum: vec![0.0; self.output_dims],
            prediction_sum: 0.0,
            target_sum: 0.0,
            prediction_square_sum: 0.0,
            target_square_sum: 0.0,
            prediction_target_sum: 0.0,
        };
        for batch in 0..self.batch_count() {
            let stats = self.regression_stats_batch(self.batch_start(batch))?;
            for channel in 0..self.output_dims {
                total.channel_weighted_squared_error[channel] +=
                    stats.channel_weighted_squared_error[channel];
                total.channel_weighted_target_square[channel] +=
                    stats.channel_weighted_target_square[channel];
                total.channel_weight_sum[channel] += stats.channel_weight_sum[channel];
            }
            total.prediction_sum += stats.prediction_sum;
            total.target_sum += stats.target_sum;
            total.prediction_square_sum += stats.prediction_square_sum;
            total.target_square_sum += stats.target_square_sum;
            total.prediction_target_sum += stats.prediction_target_sum;
        }
        Ok(total)
    }

    fn regression_stats_batch(&self, start: usize) -> AutomataResult<MlpRegressionStats> {
        let rows = self.optimizer_batch_rows;
        let output = self.forward_batch(start).3;
        let targets = self.slice_rows(self.targets.clone(), start, self.output_dims);
        let expanded_weights = self.row_weights.as_ref().map_or_else(
            || output.clone().zeros_like().add_scalar(1.0),
            |weights| {
                self.slice_rows(weights.clone(), start, 1)
                    .expand([rows, self.output_dims])
            },
        );
        let difference = output.clone() - targets.clone();
        let channel_weighted_squared_error = difference
            .clone()
            .mul(difference)
            .mul(expanded_weights.clone())
            .sum_dim(0);
        let channel_weighted_target_square = targets
            .clone()
            .mul(targets.clone())
            .mul(expanded_weights.clone())
            .sum_dim(0);
        let scalar_stats = Tensor::cat(
            vec![
                output.clone().sum().reshape([1, 1]),
                targets.clone().sum().reshape([1, 1]),
                output.clone().mul(output.clone()).sum().reshape([1, 1]),
                targets.clone().mul(targets.clone()).sum().reshape([1, 1]),
                output.mul(targets).sum().reshape([1, 1]),
            ],
            1,
        );
        let stats = tensor_values(Tensor::cat(
            vec![
                channel_weighted_squared_error,
                channel_weighted_target_square,
                expanded_weights.sum_dim(0),
                scalar_stats,
            ],
            1,
        ))?;
        let channel_stats = self.output_dims;
        let scalars = &stats[3 * channel_stats..];
        Ok(MlpRegressionStats {
            channel_weighted_squared_error: stats[..channel_stats].to_vec(),
            channel_weighted_target_square: stats[channel_stats..2 * channel_stats].to_vec(),
            channel_weight_sum: stats[2 * channel_stats..3 * channel_stats].to_vec(),
            prediction_sum: scalars[0],
            target_sum: scalars[1],
            prediction_square_sum: scalars[2],
            target_square_sum: scalars[3],
            prediction_target_sum: scalars[4],
        })
    }

    fn step(
        &mut self,
        config: AdamWConfig,
        gradient_reduction_chunk_rows: usize,
        capture: bool,
        name: &str,
    ) -> AutomataResult<(Option<f32>, Option<f32>)> {
        let rows = self.optimizer_batch_rows;
        let start = self.batch_start(self.step % self.batch_count());
        let (features, pre_hidden, hidden, output) = self.forward_batch(start);
        let targets = self.slice_rows(self.targets.clone(), start, self.output_dims);
        let difference = output.clone() - targets.clone();
        let d_output = match self.objective {
            MlpObjective::MeanSquared => {
                let difference = match &self.row_weights {
                    Some(weights) => difference.mul(
                        self.slice_rows(weights.clone(), start, 1)
                            .expand([rows, self.output_dims]),
                    ),
                    None => difference,
                };
                difference.mul_scalar(2.0 / (rows * self.output_dims) as f32)
            }
            MlpObjective::TopKRank {
                cutoff,
                classification_weight,
                temperature,
            } => {
                let expanded_weights = self.row_weights.as_ref().map_or_else(
                    || output.clone().zeros_like().add_scalar(1.0),
                    |weights| {
                        self.slice_rows(weights.clone(), start, 1)
                            .expand([rows, self.output_dims])
                    },
                );
                let rank = difference
                    .mul(expanded_weights.clone())
                    .mul_scalar(2.0 / (rows * self.output_dims) as f32);
                let labels = targets.greater_equal_elem(cutoff).float();
                let classification = (sigmoid((output - cutoff).div_scalar(temperature)) - labels)
                    .mul(expanded_weights)
                    .mul_scalar(classification_weight / (rows as f32 * temperature));
                rank + classification
            }
            MlpObjective::Controller {
                event_positive_weights: _,
            } => {
                let objective = self
                    .controller_objective
                    .as_ref()
                    .expect("controller objective tensors initialized");
                let probability = sigmoid(output);
                let one = targets.clone().ones_like();
                let regression = difference
                    .mul(objective.regression_mask.clone())
                    .mul_scalar(1.0 / rows as f32);
                let positive = targets
                    .clone()
                    .mul(objective.positive_weights.clone())
                    .mul(probability.clone() - 1.0);
                let negative = (one - targets).mul(probability);
                regression
                    + (positive + negative)
                        .mul(objective.event_mask.clone())
                        .mul_scalar(1.0 / (rows * 2) as f32)
            }
        };
        let d_output = if let Some(scale) = &self.output_scale {
            d_output.mul(scale.clone().expand([rows, self.output_dims]))
        } else {
            d_output
        };
        let d_output = if let Some(scale) = &self.row_output_scale {
            d_output.mul(
                self.slice_rows(scale.clone(), start, 1)
                    .expand([rows, self.output_dims]),
            )
        } else {
            d_output
        };
        let gb2 = chunked_row_sum(
            d_output.clone(),
            rows,
            self.output_dims,
            gradient_reduction_chunk_rows,
        );
        let gw2 = chunked_outer_sum(
            d_output.clone(),
            hidden,
            rows,
            self.output_dims,
            self.hidden_dims,
            gradient_reduction_chunk_rows,
        );
        let d_hidden = d_output.matmul(self.w2.clone());
        let d_pre = d_hidden.mask_fill(pre_hidden.lower_equal_elem(0.0), 0.0);
        let gb1 = chunked_row_sum(
            d_pre.clone(),
            rows,
            self.hidden_dims,
            gradient_reduction_chunk_rows,
        );
        let gw1 = chunked_outer_sum(
            d_pre,
            features,
            rows,
            self.hidden_dims,
            self.features.dims()[1],
            gradient_reduction_chunk_rows,
        );
        let gradient_norm = (gw1.clone().mul(gw1.clone()).sum()
            + gb1.clone().mul(gb1.clone()).sum()
            + gw2.clone().mul(gw2.clone()).sum()
            + gb2.clone().mul(gb2.clone()).sum())
        .sqrt();
        let gradient_scale = if config.grad_clip_norm > 0.0 {
            gradient_norm
                .clone()
                .recip()
                .mul_scalar(config.grad_clip_norm)
                .clamp_max(1.0)
                .reshape([1, 1])
        } else {
            gradient_norm
                .clone()
                .mul_scalar(0.0)
                .add_scalar(1.0)
                .reshape([1, 1])
        };
        self.step += 1;
        let beta1_correction = 1.0 - config.beta1.powi(self.step as i32);
        let beta2_correction = 1.0 - config.beta2.powi(self.step as i32);
        let [w1_m, w1_v, b1_m, b1_v, w2_m, w2_v, b2_m, b2_v] = self.moments.clone();
        (self.w1, self.moments[0], self.moments[1]) = adamw_update(
            self.w1.clone(),
            gw1,
            w1_m,
            w1_v,
            gradient_scale.clone(),
            config,
            beta1_correction,
            beta2_correction,
        );
        (self.b1, self.moments[2], self.moments[3]) = adamw_update(
            self.b1.clone(),
            gb1,
            b1_m,
            b1_v,
            gradient_scale.clone(),
            config,
            beta1_correction,
            beta2_correction,
        );
        (self.w2, self.moments[4], self.moments[5]) = adamw_update(
            self.w2.clone(),
            gw2,
            w2_m,
            w2_v,
            gradient_scale.clone(),
            config,
            beta1_correction,
            beta2_correction,
        );
        (self.b2, self.moments[6], self.moments[7]) = adamw_update(
            self.b2.clone(),
            gb2,
            b2_m,
            b2_v,
            gradient_scale,
            config,
            beta1_correction,
            beta2_correction,
        );
        if !capture {
            return Ok((None, None));
        }
        Ok((
            Some(self.loss(name)?),
            Some(scalar::<B>(
                gradient_norm,
                &format!("{name} gradient norm"),
            )?),
        ))
    }

    fn into_weights(self) -> AutomataResult<MlpWeights> {
        Ok(MlpWeights {
            w1: tensor_values(self.w1)?,
            b1: tensor_values(self.b1)?,
            w2: tensor_values(self.w2)?,
            b2: tensor_values(self.b2)?,
        })
    }

    fn weights(&self) -> AutomataResult<MlpWeights> {
        Ok(MlpWeights {
            w1: tensor_values(self.w1.clone())?,
            b1: tensor_values(self.b1.clone())?,
            w2: tensor_values(self.w2.clone())?,
            b2: tensor_values(self.b2.clone())?,
        })
    }
}

fn gradient_reduction_shape(
    rows: usize,
    requested_chunk_rows: usize,
) -> Option<(usize, usize, usize)> {
    if requested_chunk_rows == 0 || rows < 256 {
        return None;
    }
    let mut chunk_rows = requested_chunk_rows.min(rows / 2);
    while chunk_rows >= 128 && !rows.is_multiple_of(chunk_rows) {
        chunk_rows /= 2;
    }
    if chunk_rows >= 128 {
        return Some((rows / chunk_rows, chunk_rows, 0));
    }

    // Prime and otherwise awkward row counts are common after selecting only
    // coarse leaves. Keep the large regular prefix on the parallel batched
    // reduction and handle only the short tail directly instead of silently
    // falling back to a low-parallelism [D, rows] x [rows, H] GEMM.
    let chunk_rows = requested_chunk_rows.min(rows / 2);
    let chunks = rows / chunk_rows;
    (chunk_rows >= 128 && chunks >= 2).then_some((chunks, chunk_rows, rows - chunks * chunk_rows))
}

fn chunked_row_sum<B: Backend>(
    values: Tensor<B, 2>,
    rows: usize,
    dims: usize,
    chunk_rows: usize,
) -> Tensor<B, 2> {
    let Some((chunks, chunk_rows, remainder_rows)) = gradient_reduction_shape(rows, chunk_rows)
    else {
        return values.sum_dim(0);
    };
    let regular_rows = chunks * chunk_rows;
    let reduced = values
        .clone()
        .slice([0..regular_rows, 0..dims])
        .reshape([chunks, chunk_rows, dims])
        .sum_dim(1)
        .sum_dim(0)
        .reshape([1, dims]);
    if remainder_rows == 0 {
        reduced
    } else {
        reduced + values.slice([regular_rows..rows, 0..dims]).sum_dim(0)
    }
}

fn chunked_outer_sum<B: Backend>(
    lhs: Tensor<B, 2>,
    rhs: Tensor<B, 2>,
    rows: usize,
    lhs_dims: usize,
    rhs_dims: usize,
    chunk_rows: usize,
) -> Tensor<B, 2> {
    let Some((chunks, chunk_rows, remainder_rows)) = gradient_reduction_shape(rows, chunk_rows)
    else {
        return lhs.transpose().matmul(rhs);
    };
    let regular_rows = chunks * chunk_rows;
    let reduced = lhs
        .clone()
        .slice([0..regular_rows, 0..lhs_dims])
        .reshape([chunks, chunk_rows, lhs_dims])
        .swap_dims(1, 2)
        .matmul(
            rhs.clone()
                .slice([0..regular_rows, 0..rhs_dims])
                .reshape([chunks, chunk_rows, rhs_dims]),
        )
        .sum_dim(0)
        .reshape([lhs_dims, rhs_dims]);
    if remainder_rows == 0 {
        reduced
    } else {
        reduced
            + lhs
                .slice([regular_rows..rows, 0..lhs_dims])
                .transpose()
                .matmul(rhs.slice([regular_rows..rows, 0..rhs_dims]))
    }
}

fn controller_regression_mask<B: Backend>(rows: usize, device: &B::Device) -> Tensor<B, 2> {
    tensor2(
        (0..rows).flat_map(|_| [1.0, 1.0, 0.0, 0.0]).collect(),
        [rows, 4],
        device,
    )
}

fn controller_event_mask<B: Backend>(rows: usize, device: &B::Device) -> Tensor<B, 2> {
    tensor2(
        (0..rows).flat_map(|_| [0.0, 0.0, 1.0, 1.0]).collect(),
        [rows, 4],
        device,
    )
}

fn controller_positive_weights<B: Backend>(
    rows: usize,
    event_positive_weights: [f32; 2],
    device: &B::Device,
) -> Tensor<B, 2> {
    tensor2(
        (0..rows)
            .flat_map(|_| {
                [
                    0.0,
                    0.0,
                    event_positive_weights[0],
                    event_positive_weights[1],
                ]
            })
            .collect(),
        [rows, 4],
        device,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn adamw_update<B: Backend>(
    parameter: Tensor<B, 2>,
    gradient: Tensor<B, 2>,
    moment: Tensor<B, 2>,
    velocity: Tensor<B, 2>,
    gradient_scale: Tensor<B, 2>,
    config: AdamWConfig,
    beta1_correction: f32,
    beta2_correction: f32,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    let parameter = parameter.mul_scalar(1.0 - config.learning_rate * config.weight_decay);
    let gradient = gradient.mul(gradient_scale);
    let moment = moment.mul_scalar(config.beta1) + gradient.clone().mul_scalar(1.0 - config.beta1);
    let velocity = velocity.mul_scalar(config.beta2)
        + gradient
            .clone()
            .mul(gradient)
            .mul_scalar(1.0 - config.beta2);
    let update = moment
        .clone()
        .div_scalar(beta1_correction.max(f32::MIN_POSITIVE))
        .div(
            velocity
                .clone()
                .div_scalar(beta2_correction.max(f32::MIN_POSITIVE))
                .sqrt()
                .add_scalar(config.epsilon),
        );
    (
        parameter - update.mul_scalar(config.learning_rate),
        moment,
        velocity,
    )
}

pub(super) fn tensor2<B: Backend>(
    values: Vec<f32>,
    shape: [usize; 2],
    device: &B::Device,
) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(values, shape), device)
}

pub(super) fn tensor_values<B: Backend>(tensor: Tensor<B, 2>) -> AutomataResult<Vec<f32>> {
    tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| AutomataError::InvalidArgument(error.to_string()))
}

pub(super) fn scalar<B: Backend>(tensor: Tensor<B, 1>, name: &str) -> AutomataResult<f32> {
    let value = tensor.into_scalar().elem::<f32>();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(AutomataError::InvalidArgument(format!(
            "{name} is non-finite: {value}"
        )))
    }
}

#[cfg(all(test, feature = "backend_ndarray"))]
mod tests {
    use super::*;

    type TestBackend = burn::backend::NdArray<f32>;

    #[test]
    fn chunked_gradient_reductions_match_direct_reference() {
        let device = Default::default();
        let rows = 259;
        let lhs_dims = 3;
        let rhs_dims = 5;
        let lhs_values = (0..rows * lhs_dims)
            .map(|index| ((index % 23) as f32 - 11.0) * 0.013)
            .collect::<Vec<_>>();
        let rhs_values = (0..rows * rhs_dims)
            .map(|index| ((index % 29) as f32 - 14.0) * 0.009)
            .collect::<Vec<_>>();
        let lhs = tensor2::<TestBackend>(lhs_values, [rows, lhs_dims], &device);
        let rhs = tensor2::<TestBackend>(rhs_values, [rows, rhs_dims], &device);

        let direct_outer = tensor_values(lhs.clone().transpose().matmul(rhs.clone())).unwrap();
        let chunked_outer = tensor_values(chunked_outer_sum(
            lhs.clone(),
            rhs,
            rows,
            lhs_dims,
            rhs_dims,
            128,
        ))
        .unwrap();
        let direct_sum = tensor_values(lhs.clone().sum_dim(0)).unwrap();
        let chunked_sum = tensor_values(chunked_row_sum(lhs, rows, lhs_dims, 128)).unwrap();

        for (direct, chunked) in direct_outer.iter().zip(&chunked_outer) {
            assert!((direct - chunked).abs() <= 2.0e-5, "{direct} != {chunked}");
        }
        for (direct, chunked) in direct_sum.iter().zip(&chunked_sum) {
            assert!((direct - chunked).abs() <= 2.0e-5, "{direct} != {chunked}");
        }
    }

    #[test]
    fn direct_gradient_reduction_remains_available_for_parity_ablation() {
        assert_eq!(gradient_reduction_shape(46_080, 0), None);
        assert_eq!(gradient_reduction_shape(1, 1_024), None);
        assert_eq!(gradient_reduction_shape(255, 128), None);
        assert_eq!(
            gradient_reduction_shape(46_080, 1_024),
            Some((45, 1_024, 0))
        );
        assert_eq!(gradient_reduction_shape(2_304, 1_024), Some((9, 256, 0)));
        assert_eq!(gradient_reduction_shape(8_283, 1_024), Some((8, 1_024, 91)));
    }

    #[test]
    fn buffer_preflight_rejects_the_previous_unbounded_restriction_shape() {
        let shape = MlpShape {
            input_dims: 200,
            hidden_dims: 512,
            output_dims: 4,
        };
        let error = validate_mlp_buffer_plan(1_441_792, shape, 0, "restriction").unwrap_err();
        assert!(error.to_string().contains("2.75 GiB"), "{error}");
        assert_eq!(
            validate_mlp_buffer_plan(1_441_792, shape, 180_224, "restriction").unwrap(),
            180_224,
        );
    }

    #[test]
    fn output_scale_detaches_disabled_channels_and_fits_active_channels() {
        let shape = MlpShape {
            input_dims: 1,
            hidden_dims: 4,
            output_dims: 2,
        };
        let weights = MlpWeights {
            w1: vec![0.1, -0.1, 0.2, -0.2],
            b1: vec![0.1; 4],
            w2: vec![0.3; 8],
            b2: vec![0.4, 0.0],
        };
        let disabled_w2 = weights.w2[..shape.hidden_dims].to_vec();
        let disabled_b2 = weights.b2[0];
        let output = train_weighted_scaled_mlp::<TestBackend>(
            weights,
            vec![0.0, 1.0, 2.0, 3.0],
            vec![100.0, 0.0, 100.0, 0.5, 100.0, 1.0, 100.0, 1.5],
            vec![1.0; 4],
            vec![0.0, 0.25],
            4,
            shape,
            MlpTrainConfig {
                steps: 200,
                report_interval: 50,
                optimizer: AdamWConfig {
                    learning_rate: 0.01,
                    weight_decay: 0.0,
                    grad_clip_norm: 5.0,
                    ..AdamWConfig::default()
                },
                gradient_reduction_chunk_rows: 0,
                optimizer_batch_rows: 2,
            },
            &Default::default(),
            "scaled output test",
        )
        .unwrap();
        assert!(output.final_loss < output.initial_loss);
        assert_eq!(&output.weights.w2[..shape.hidden_dims], &disabled_w2);
        assert_eq!(output.weights.b2[0], disabled_b2);
    }

    #[test]
    fn topk_rank_objective_fits_the_deployed_cut() {
        let shape = MlpShape {
            input_dims: 1,
            hidden_dims: 4,
            output_dims: 4,
        };
        let ranks = [1.0, 0.5, -0.5, -1.0];
        let targets = ranks
            .into_iter()
            .flat_map(|rank| [0.0, 0.0, 0.0, rank])
            .collect();
        let output = train_weighted_scaled_topk_mlp::<TestBackend>(
            MlpWeights {
                w1: vec![0.2, 0.1, -0.1, -0.2],
                b1: vec![0.1; 4],
                w2: vec![0.0; shape.output_dims * shape.hidden_dims],
                b2: vec![0.0; shape.output_dims],
            },
            vec![1.0, 0.5, -0.5, -1.0],
            targets,
            vec![1.0; 4],
            vec![0.0, 0.0, 0.0, 1.0],
            4,
            shape,
            MlpTrainConfig {
                steps: 400,
                report_interval: 100,
                optimizer: AdamWConfig {
                    learning_rate: 0.01,
                    weight_decay: 0.0,
                    grad_clip_norm: 5.0,
                    ..AdamWConfig::default()
                },
                gradient_reduction_chunk_rows: 0,
                optimizer_batch_rows: 2,
            },
            0.0,
            0.25,
            0.25,
            &Default::default(),
            "top-k rank test",
        )
        .unwrap();
        assert!(
            output.final_loss < output.initial_loss * 0.1,
            "{} !< {}",
            output.final_loss,
            output.initial_loss * 0.1
        );
    }
}

#[cfg(all(test, any(feature = "backend_cuda", feature = "backend_wgpu")))]
fn benchmark_coarse_replacement_shape<B>(backend: &str)
where
    B: Backend,
    B::Device: Default,
{
    let rows = 8_283;
    let shape = MlpShape {
        input_dims: 102,
        hidden_dims: 191,
        output_dims: 18,
    };
    let weights = MlpWeights {
        w1: (0..shape.hidden_dims * shape.input_dims)
            .map(|index| ((index % 31) as f32 - 15.0) * 1.0e-3)
            .collect(),
        b1: vec![0.01; shape.hidden_dims],
        w2: (0..shape.output_dims * shape.hidden_dims)
            .map(|index| ((index % 17) as f32 - 8.0) * 1.0e-3)
            .collect(),
        b2: vec![0.0; shape.output_dims],
    };
    let features = (0..rows * shape.input_dims)
        .map(|index| ((index % 37) as f32 - 18.0) * 0.02)
        .collect::<Vec<_>>();
    let targets = (0..rows * shape.output_dims)
        .map(|index| ((index % 23) as f32 - 11.0) * 0.01)
        .collect::<Vec<_>>();
    let base_config = MlpTrainConfig {
        steps: 250,
        report_interval: 250,
        optimizer: AdamWConfig {
            learning_rate: 1.0e-4,
            weight_decay: 1.0e-6,
            grad_clip_norm: 5.0,
            ..AdamWConfig::default()
        },
        gradient_reduction_chunk_rows: 0,
        optimizer_batch_rows: 0,
    };
    let device = Default::default();

    // Compile both graph shapes before measuring steady-state execution.
    for chunk_rows in [0, DEFAULT_GRADIENT_REDUCTION_CHUNK_ROWS] {
        train_mlp::<B>(
            weights.clone(),
            features.clone(),
            targets.clone(),
            rows,
            shape,
            MlpTrainConfig {
                steps: 2,
                report_interval: 2,
                gradient_reduction_chunk_rows: chunk_rows,
                ..base_config
            },
            MlpObjective::MeanSquared,
            &device,
            "coarse replacement warmup",
        )
        .unwrap();
    }

    let direct = train_mlp::<B>(
        weights.clone(),
        features.clone(),
        targets.clone(),
        rows,
        shape,
        base_config,
        MlpObjective::MeanSquared,
        &device,
        "coarse replacement direct benchmark",
    )
    .unwrap();
    let chunked = train_mlp::<B>(
        weights,
        features,
        targets,
        rows,
        shape,
        MlpTrainConfig {
            gradient_reduction_chunk_rows: DEFAULT_GRADIENT_REDUCTION_CHUNK_ROWS,
            ..base_config
        },
        MlpObjective::MeanSquared,
        &device,
        "coarse replacement ragged benchmark",
    )
    .unwrap();
    let speedup = chunked.rows_per_second / direct.rows_per_second;
    let loss_relative_difference = (chunked.final_loss - direct.final_loss).abs()
        / direct.final_loss.abs().max(f32::MIN_POSITIVE);
    eprintln!(
        "adaptive {backend} coarse MLP: direct={:.2}M rows/s ({:.3}ms/update), ragged={:.2}M rows/s ({:.3}ms/update), speedup={speedup:.2}x, final-loss relative difference={loss_relative_difference:.3e}",
        direct.rows_per_second / 1.0e6,
        direct.elapsed_ms / base_config.steps as f64,
        chunked.rows_per_second / 1.0e6,
        chunked.elapsed_ms / base_config.steps as f64,
    );
    assert!(loss_relative_difference <= 5.0e-3);
}

#[cfg(all(test, feature = "backend_wgpu"))]
mod wgpu_benchmarks {
    use super::*;

    #[test]
    #[ignore = "WGPU throughput benchmark; run explicitly with --ignored --nocapture"]
    fn ragged_gradient_reduction_matches_coarse_replacement_shape() {
        benchmark_coarse_replacement_shape::<burn::backend::Wgpu<f32>>("WGPU");
    }
}

#[cfg(all(test, feature = "backend_cuda"))]
mod cuda_benchmarks {
    use super::*;

    #[test]
    #[ignore = "CUDA throughput benchmark; run explicitly with --ignored --nocapture"]
    fn ragged_gradient_reduction_matches_coarse_replacement_shape() {
        benchmark_coarse_replacement_shape::<burn::backend::Cuda<f32>>("CUDA");
    }

    #[test]
    #[ignore = "CUDA throughput benchmark; run explicitly with --ignored --nocapture"]
    fn chunked_gradient_reduction_outpaces_direct_reference() {
        type CudaBackend = burn::backend::Cuda<f32>;

        let rows = 46_080;
        let shape = MlpShape {
            input_dims: 15,
            hidden_dims: 191,
            output_dims: 6,
        };
        let weights = MlpWeights {
            w1: (0..shape.hidden_dims * shape.input_dims)
                .map(|index| ((index % 31) as f32 - 15.0) * 1.0e-3)
                .collect(),
            b1: vec![0.01; shape.hidden_dims],
            w2: (0..shape.output_dims * shape.hidden_dims)
                .map(|index| ((index % 17) as f32 - 8.0) * 1.0e-3)
                .collect(),
            b2: vec![0.0; shape.output_dims],
        };
        let features = (0..rows * shape.input_dims)
            .map(|index| ((index % 37) as f32 - 18.0) * 0.02)
            .collect::<Vec<_>>();
        let targets = (0..rows * shape.output_dims)
            .map(|index| ((index % 23) as f32 - 11.0) * 0.01)
            .collect::<Vec<_>>();
        let base_config = MlpTrainConfig {
            steps: 100,
            report_interval: 100,
            optimizer: AdamWConfig {
                learning_rate: 1.0e-3,
                weight_decay: 1.0e-6,
                grad_clip_norm: 5.0,
                ..AdamWConfig::default()
            },
            gradient_reduction_chunk_rows: 0,
            optimizer_batch_rows: 0,
        };
        let device = Default::default();
        let direct = train_mlp::<CudaBackend>(
            weights.clone(),
            features.clone(),
            targets.clone(),
            rows,
            shape,
            base_config,
            MlpObjective::MeanSquared,
            &device,
            "direct gradient reduction benchmark",
        )
        .unwrap();
        let chunked = train_mlp::<CudaBackend>(
            weights,
            features,
            targets,
            rows,
            shape,
            MlpTrainConfig {
                gradient_reduction_chunk_rows: DEFAULT_GRADIENT_REDUCTION_CHUNK_ROWS,
                ..base_config
            },
            MlpObjective::MeanSquared,
            &device,
            "chunked gradient reduction benchmark",
        )
        .unwrap();
        let speedup = chunked.rows_per_second / direct.rows_per_second;
        let loss_relative_difference = (chunked.final_loss - direct.final_loss).abs()
            / direct.final_loss.abs().max(f32::MIN_POSITIVE);
        eprintln!(
            "adaptive CUDA MLP: direct={:.2}M rows/s ({:.3}ms), chunked={:.2}M rows/s ({:.3}ms), speedup={speedup:.2}x, final-loss relative difference={loss_relative_difference:.3e}",
            direct.rows_per_second / 1.0e6,
            direct.elapsed_ms,
            chunked.rows_per_second / 1.0e6,
            chunked.elapsed_ms,
        );
        assert!(speedup >= 2.0);
        assert!(loss_relative_difference <= 5.0e-3);
    }
}
