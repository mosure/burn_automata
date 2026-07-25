use std::time::Instant;

use burn::tensor::{Tensor, activation::relu, backend::Backend};

use super::mlp::{
    MlpHistoryEntry, MlpShape, MlpTrainConfig, MlpWeights, adamw_update, scalar, tensor_values,
    tensor2,
};
use crate::{AutomataError, AutomataResult};

#[derive(Clone, Debug)]
pub(super) struct DualMlpTrainingOutput {
    pub local_weights: MlpWeights,
    pub proxy_weights: MlpWeights,
    pub initial_loss: f32,
    pub final_loss: f32,
    pub best_loss: f32,
    pub elapsed_ms: f64,
    pub rows_per_second: f64,
    pub history: Vec<MlpHistoryEntry>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn train_dual_mlp<B: Backend>(
    local_weights: MlpWeights,
    proxy_weights: MlpWeights,
    local_features: Vec<f32>,
    proxy_features: Vec<f32>,
    targets: Vec<f32>,
    row_weights: Vec<f32>,
    rows: usize,
    shape: MlpShape,
    local_scale: f32,
    proxy_scale: f32,
    config: MlpTrainConfig,
    device: &B::Device,
) -> AutomataResult<DualMlpTrainingOutput> {
    validate(
        &local_weights,
        &proxy_weights,
        &local_features,
        &proxy_features,
        &targets,
        &row_weights,
        rows,
        shape,
        local_scale,
        proxy_scale,
        config,
    )?;
    let mut trainer = DualMlpTensorTrainer::<B>::new(
        local_weights,
        proxy_weights,
        local_features,
        proxy_features,
        targets,
        row_weights,
        rows,
        shape,
        local_scale,
        proxy_scale,
        device,
    );
    let initial_loss = trainer.loss()?;
    let mut final_loss = initial_loss;
    let mut best_loss = initial_loss;
    let mut history = Vec::new();
    let started = Instant::now();
    for step in 1..=config.steps {
        let capture =
            step == 1 || step == config.steps || step.is_multiple_of(config.report_interval);
        let (loss, gradient_norm) = trainer.step(config.optimizer, capture)?;
        if let Some(loss) = loss {
            final_loss = loss;
            best_loss = best_loss.min(loss);
            history.push(MlpHistoryEntry {
                step,
                loss,
                gradient_norm: gradient_norm.unwrap_or_default(),
                elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
            });
        }
    }
    let elapsed_seconds = started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE);
    let (local_weights, proxy_weights) = trainer.into_weights()?;
    Ok(DualMlpTrainingOutput {
        local_weights,
        proxy_weights,
        initial_loss,
        final_loss,
        best_loss,
        elapsed_ms: elapsed_seconds * 1_000.0,
        rows_per_second: rows as f64 * config.steps as f64 / elapsed_seconds,
        history,
    })
}

#[allow(clippy::too_many_arguments)]
fn validate(
    local: &MlpWeights,
    proxy: &MlpWeights,
    local_features: &[f32],
    proxy_features: &[f32],
    targets: &[f32],
    row_weights: &[f32],
    rows: usize,
    shape: MlpShape,
    local_scale: f32,
    proxy_scale: f32,
    config: MlpTrainConfig,
) -> AutomataResult<()> {
    let valid_weights = |weights: &MlpWeights| {
        weights.w1.len() == shape.hidden_dims * shape.input_dims
            && weights.b1.len() == shape.hidden_dims
            && weights.w2.len() == shape.output_dims * shape.hidden_dims
            && weights.b2.len() == shape.output_dims
    };
    if rows == 0
        || shape.input_dims == 0
        || shape.hidden_dims == 0
        || shape.output_dims == 0
        || config.steps == 0
        || config.report_interval == 0
        || local_features.len() != rows * shape.input_dims
        || proxy_features.len() != rows * shape.input_dims
        || targets.len() != rows * shape.output_dims
        || row_weights.len() != rows
        || !valid_weights(local)
        || !valid_weights(proxy)
        || !local_scale.is_finite()
        || local_scale < 0.0
        || !proxy_scale.is_finite()
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive dual MLP training shape/config mismatch".to_string(),
        ));
    }
    if local_features
        .iter()
        .chain(proxy_features)
        .chain(targets)
        .chain(row_weights)
        .chain(&local.w1)
        .chain(&local.b1)
        .chain(&local.w2)
        .chain(&local.b2)
        .chain(&proxy.w1)
        .chain(&proxy.b1)
        .chain(&proxy.w2)
        .chain(&proxy.b2)
        .any(|value| !value.is_finite())
        || row_weights.iter().any(|value| *value < 0.0)
        || row_weights.iter().sum::<f32>() <= f32::MIN_POSITIVE
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive dual MLP training contains invalid values".to_string(),
        ));
    }
    Ok(())
}

struct MlpBranch<B: Backend> {
    features: Tensor<B, 2>,
    w1: Tensor<B, 2>,
    b1: Tensor<B, 2>,
    w2: Tensor<B, 2>,
    b2: Tensor<B, 2>,
    moments: [Tensor<B, 2>; 8],
    rows: usize,
    shape: MlpShape,
}

impl<B: Backend> MlpBranch<B> {
    fn new(
        weights: MlpWeights,
        features: Vec<f32>,
        rows: usize,
        shape: MlpShape,
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
        Self {
            features: tensor2(features, [rows, shape.input_dims], device),
            w1,
            b1,
            w2,
            b2,
            moments,
            rows,
            shape,
        }
    }

    fn forward(&self) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let pre_hidden = self.features.clone().matmul(self.w1.clone().transpose())
            + self.b1.clone().expand([self.rows, self.shape.hidden_dims]);
        let hidden = relu(pre_hidden.clone());
        let output = hidden.clone().matmul(self.w2.clone().transpose())
            + self.b2.clone().expand([self.rows, self.shape.output_dims]);
        (pre_hidden, hidden, output)
    }

    fn gradients(
        &self,
        pre_hidden: Tensor<B, 2>,
        hidden: Tensor<B, 2>,
        d_output: Tensor<B, 2>,
    ) -> [Tensor<B, 2>; 4] {
        let gb2 = d_output.clone().sum_dim(0);
        let gw2 = d_output.clone().transpose().matmul(hidden);
        let d_hidden = d_output.matmul(self.w2.clone());
        let d_pre = d_hidden.mask_fill(pre_hidden.lower_equal_elem(0.0), 0.0);
        let gb1 = d_pre.clone().sum_dim(0);
        let gw1 = d_pre.transpose().matmul(self.features.clone());
        [gw1, gb1, gw2, gb2]
    }

    fn apply_gradients(
        &mut self,
        [gw1, gb1, gw2, gb2]: [Tensor<B, 2>; 4],
        gradient_scale: Tensor<B, 2>,
        config: crate::AdamWConfig,
        beta1_correction: f32,
        beta2_correction: f32,
    ) {
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
    }

    fn into_weights(self) -> AutomataResult<MlpWeights> {
        Ok(MlpWeights {
            w1: tensor_values(self.w1)?,
            b1: tensor_values(self.b1)?,
            w2: tensor_values(self.w2)?,
            b2: tensor_values(self.b2)?,
        })
    }
}

struct DualMlpTensorTrainer<B: Backend> {
    local: Option<MlpBranch<B>>,
    frozen_local_weights: Option<MlpWeights>,
    proxy: MlpBranch<B>,
    targets: Tensor<B, 2>,
    row_weights: Tensor<B, 2>,
    rows: usize,
    output_dims: usize,
    local_scale: f32,
    proxy_scale: f32,
    step: usize,
}

type MlpForward<B> = (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>);
type DualMlpForward<B> = (Option<MlpForward<B>>, MlpForward<B>, Tensor<B, 2>);

impl<B: Backend> DualMlpTensorTrainer<B> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        local_weights: MlpWeights,
        proxy_weights: MlpWeights,
        local_features: Vec<f32>,
        proxy_features: Vec<f32>,
        targets: Vec<f32>,
        row_weights: Vec<f32>,
        rows: usize,
        shape: MlpShape,
        local_scale: f32,
        proxy_scale: f32,
        device: &B::Device,
    ) -> Self {
        let (local, frozen_local_weights) = if local_scale > 0.0 {
            (
                Some(MlpBranch::new(
                    local_weights,
                    local_features,
                    rows,
                    shape,
                    device,
                )),
                None,
            )
        } else {
            (None, Some(local_weights))
        };
        Self {
            local,
            frozen_local_weights,
            proxy: MlpBranch::new(proxy_weights, proxy_features, rows, shape, device),
            targets: tensor2(targets, [rows, shape.output_dims], device),
            row_weights: tensor2(row_weights, [rows, 1], device),
            rows,
            output_dims: shape.output_dims,
            local_scale,
            proxy_scale,
            step: 0,
        }
    }

    fn forward(&self) -> DualMlpForward<B> {
        let local = self.local.as_ref().map(MlpBranch::forward);
        let proxy = self.proxy.forward();
        let mut output = proxy.2.clone().mul_scalar(self.proxy_scale);
        if let Some(local) = &local {
            output = output + local.2.clone().mul_scalar(self.local_scale);
        }
        (local, proxy, output)
    }

    fn loss(&self) -> AutomataResult<f32> {
        let difference = self.forward().2 - self.targets.clone();
        scalar::<B>(
            difference
                .clone()
                .mul(difference)
                .mul(self.row_weights.clone())
                .mean(),
            "adaptive multiscale rule loss",
        )
    }

    fn step(
        &mut self,
        config: crate::AdamWConfig,
        capture: bool,
    ) -> AutomataResult<(Option<f32>, Option<f32>)> {
        let (local_forward, proxy_forward, output) = self.forward();
        let difference = output - self.targets.clone();
        let d_output = difference
            .mul(self.row_weights.clone())
            .mul_scalar(2.0 / (self.rows * self.output_dims) as f32);
        let local_gradients = local_forward.map(|forward| {
            self.local
                .as_ref()
                .expect("local forward requires local branch")
                .gradients(
                    forward.0,
                    forward.1,
                    d_output.clone().mul_scalar(self.local_scale),
                )
        });
        let proxy_gradients = self.proxy.gradients(
            proxy_forward.0,
            proxy_forward.1,
            d_output.mul_scalar(self.proxy_scale),
        );
        let mut gradient_square_sum = proxy_gradients[0]
            .clone()
            .mul(proxy_gradients[0].clone())
            .sum();
        for gradient in proxy_gradients.iter().skip(1).chain(
            local_gradients
                .as_ref()
                .into_iter()
                .flat_map(|gradients| gradients.iter()),
        ) {
            gradient_square_sum =
                gradient_square_sum + gradient.clone().mul(gradient.clone()).sum();
        }
        let gradient_norm = gradient_square_sum.sqrt();
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
        if let (Some(local), Some(local_gradients)) = (&mut self.local, local_gradients) {
            local.apply_gradients(
                local_gradients,
                gradient_scale.clone(),
                config,
                beta1_correction,
                beta2_correction,
            );
        }
        self.proxy.apply_gradients(
            proxy_gradients,
            gradient_scale,
            config,
            beta1_correction,
            beta2_correction,
        );
        if !capture {
            return Ok((None, None));
        }
        Ok((
            Some(self.loss()?),
            Some(scalar::<B>(
                gradient_norm,
                "adaptive multiscale rule gradient norm",
            )?),
        ))
    }

    fn into_weights(self) -> AutomataResult<(MlpWeights, MlpWeights)> {
        let local = match (self.local, self.frozen_local_weights) {
            (Some(local), None) => local.into_weights()?,
            (None, Some(weights)) => weights,
            _ => {
                return Err(AutomataError::InvalidModel(
                    "adaptive local branch ownership is inconsistent".to_string(),
                ));
            }
        };
        Ok((local, self.proxy.into_weights()?))
    }
}

#[cfg(all(test, feature = "backend_ndarray"))]
mod tests {
    use super::*;

    #[test]
    fn zero_local_scale_trains_only_proxy_branch() {
        let shape = MlpShape {
            input_dims: 2,
            hidden_dims: 3,
            output_dims: 1,
        };
        let local = MlpWeights {
            w1: vec![0.1; 6],
            b1: vec![0.2; 3],
            w2: vec![0.3; 3],
            b2: vec![0.4],
        };
        let original_local = local.clone();
        let proxy = MlpWeights {
            w1: vec![0.05; 6],
            b1: vec![0.0; 3],
            w2: vec![0.05; 3],
            b2: vec![0.0],
        };
        let output = train_dual_mlp::<burn::backend::NdArray<f32>>(
            local,
            proxy,
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![0.0, 1.0, 1.0, 2.0],
            vec![1.0; 4],
            4,
            shape,
            0.0,
            1.0,
            MlpTrainConfig {
                steps: 100,
                report_interval: 25,
                optimizer: crate::AdamWConfig {
                    learning_rate: 0.01,
                    weight_decay: 0.0,
                    grad_clip_norm: 5.0,
                    ..crate::AdamWConfig::default()
                },
                gradient_reduction_chunk_rows:
                    super::super::mlp::DEFAULT_GRADIENT_REDUCTION_CHUNK_ROWS,
                optimizer_batch_rows: 0,
            },
            &Default::default(),
        )
        .unwrap();
        assert_eq!(output.local_weights.w1, original_local.w1);
        assert_eq!(output.local_weights.b1, original_local.b1);
        assert_eq!(output.local_weights.w2, original_local.w2);
        assert_eq!(output.local_weights.b2, original_local.b2);
        assert!(output.final_loss < output.initial_loss);
    }
}
