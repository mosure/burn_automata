use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{AutomataError, AutomataResult, NpaConfig};
use burn::tensor::{Tensor, TensorData, activation::relu, backend::Backend};
use burn_automata_kernels::{
    HashGridConfig, PerceptionOptions, PerceptionOutput, euler_step, perceive_with_options,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NpaWeights {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

impl NpaWeights {
    pub fn zeros(config: &NpaConfig) -> Self {
        Self {
            w1: vec![0.0; config.hidden_dims * config.perception_dims()],
            b1: vec![0.0; config.hidden_dims],
            w2: vec![0.0; config.update_dims() * config.hidden_dims],
            b2: vec![0.0; config.update_dims()],
        }
    }

    pub fn seeded(config: &NpaConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut weights = Self::zeros(config);
        for v in &mut weights.w1 {
            *v = rng.random_range(-0.1..0.1);
        }
        for v in &mut weights.w2 {
            *v = rng.random_range(-0.1..0.1);
        }
        weights
    }

    pub fn upstream_seeded(config: &NpaConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut weights = Self::zeros(config);
        fill_xavier_uniform(
            &mut weights.w1,
            config.perception_dims(),
            config.hidden_dims,
            0.1,
            &mut rng,
        );
        fill_xavier_uniform(
            &mut weights.w2,
            config.hidden_dims,
            config.update_dims(),
            0.1,
            &mut rng,
        );
        let b1_bound = (config.perception_dims() as f32).sqrt().recip();
        for value in &mut weights.b1 {
            *value = rng.random_range(-b1_bound..b1_bound);
        }
        weights
    }

    pub fn validate(&self, config: &NpaConfig) -> AutomataResult<()> {
        let expected_w1 = config.hidden_dims * config.perception_dims();
        let expected_w2 = config.update_dims() * config.hidden_dims;
        if self.w1.len() != expected_w1 {
            return Err(AutomataError::InvalidModel(format!(
                "w1 len {} != {expected_w1}",
                self.w1.len()
            )));
        }
        if self.b1.len() != config.hidden_dims {
            return Err(AutomataError::InvalidModel(format!(
                "b1 len {} != {}",
                self.b1.len(),
                config.hidden_dims
            )));
        }
        if self.w2.len() != expected_w2 {
            return Err(AutomataError::InvalidModel(format!(
                "w2 len {} != {expected_w2}",
                self.w2.len()
            )));
        }
        if self.b2.len() != config.update_dims() {
            return Err(AutomataError::InvalidModel(format!(
                "b2 len {} != {}",
                self.b2.len(),
                config.update_dims()
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NpaLowRankAdapter {
    pub rank: usize,
    pub alpha: f32,
    pub w1_down: Vec<f32>,
    pub w1_up: Vec<f32>,
    pub w2_down: Vec<f32>,
    pub w2_up: Vec<f32>,
    pub b1_delta: Vec<f32>,
    pub b2_delta: Vec<f32>,
}

impl NpaLowRankAdapter {
    pub fn zeros(config: &NpaConfig, rank: usize, alpha: f32) -> Self {
        let rank = rank.max(1);
        Self {
            rank,
            alpha,
            w1_down: vec![0.0; rank * config.perception_dims()],
            w1_up: vec![0.0; config.hidden_dims * rank],
            w2_down: vec![0.0; rank * config.hidden_dims],
            w2_up: vec![0.0; config.update_dims() * rank],
            b1_delta: vec![0.0; config.hidden_dims],
            b2_delta: vec![0.0; config.update_dims()],
        }
    }

    pub fn seeded(config: &NpaConfig, rank: usize, alpha: f32, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut adapter = Self::zeros(config, rank, alpha);
        for value in adapter
            .w1_down
            .iter_mut()
            .chain(adapter.w1_up.iter_mut())
            .chain(adapter.w2_down.iter_mut())
            .chain(adapter.w2_up.iter_mut())
        {
            *value = rng.random_range(-0.01..0.01);
        }
        adapter
    }

    pub fn parameter_count(&self) -> usize {
        self.w1_down.len()
            + self.w1_up.len()
            + self.w2_down.len()
            + self.w2_up.len()
            + self.b1_delta.len()
            + self.b2_delta.len()
    }

    pub fn parameter_count_for_config(config: &NpaConfig, rank: usize) -> usize {
        let rank = rank.max(1);
        rank * config.perception_dims()
            + config.hidden_dims * rank
            + rank * config.hidden_dims
            + config.update_dims() * rank
            + config.hidden_dims
            + config.update_dims()
    }

    pub fn to_parameter_vector(&self) -> Vec<f32> {
        let mut values = Vec::with_capacity(self.parameter_count());
        values.extend_from_slice(&self.w1_down);
        values.extend_from_slice(&self.w1_up);
        values.extend_from_slice(&self.w2_down);
        values.extend_from_slice(&self.w2_up);
        values.extend_from_slice(&self.b1_delta);
        values.extend_from_slice(&self.b2_delta);
        values
    }

    pub fn from_parameter_vector(
        config: &NpaConfig,
        rank: usize,
        alpha: f32,
        values: Vec<f32>,
    ) -> AutomataResult<Self> {
        let rank = rank.max(1);
        let expected = Self::parameter_count_for_config(config, rank);
        if values.len() != expected {
            return Err(AutomataError::InvalidModel(format!(
                "adapter parameter vector len {} != {expected}",
                values.len()
            )));
        }
        if !values.iter().all(|value| value.is_finite()) {
            return Err(AutomataError::InvalidModel(
                "adapter parameter vector contains non-finite values".to_string(),
            ));
        }
        if !alpha.is_finite() {
            return Err(AutomataError::InvalidModel(format!(
                "adapter alpha must be finite, got {alpha}"
            )));
        }

        let input_dims = config.perception_dims();
        let hidden_dims = config.hidden_dims;
        let output_dims = config.update_dims();
        let mut offset = 0;
        let mut take = |len: usize| {
            let end = offset + len;
            let out = values[offset..end].to_vec();
            offset = end;
            out
        };
        let adapter = Self {
            rank,
            alpha,
            w1_down: take(rank * input_dims),
            w1_up: take(hidden_dims * rank),
            w2_down: take(rank * hidden_dims),
            w2_up: take(output_dims * rank),
            b1_delta: take(hidden_dims),
            b2_delta: take(output_dims),
        };
        adapter.validate(config)?;
        Ok(adapter)
    }

    pub fn exact_model_delta(
        base: &NpaModel,
        target: &NpaModel,
        rank: usize,
        alpha: f32,
    ) -> AutomataResult<Self> {
        base.validate()?;
        target.validate()?;
        if base.config != target.config {
            return Err(AutomataError::InvalidArgument(
                "exact adapter delta requires matching NPA configs".to_string(),
            ));
        }
        let rank = rank.max(1);
        let input_dims = base.config.perception_dims();
        let output_dims = base.config.update_dims();
        let required_rank = input_dims.max(output_dims);
        if rank < required_rank {
            return Err(AutomataError::InvalidArgument(format!(
                "exact adapter delta rank {rank} is too small; need at least {required_rank}"
            )));
        }
        if !alpha.is_finite() {
            return Err(AutomataError::InvalidArgument(format!(
                "exact adapter alpha must be finite, got {alpha}"
            )));
        }
        let scale = alpha / rank as f32;
        if !scale.is_finite() || scale == 0.0 {
            return Err(AutomataError::InvalidArgument(format!(
                "exact adapter scale must be finite and non-zero, got {scale}"
            )));
        }

        let hidden_dims = base.config.hidden_dims;
        let mut adapter = Self::zeros(&base.config, rank, alpha);
        for idx in 0..input_dims {
            adapter.w1_down[idx * input_dims + idx] = 1.0;
        }
        for row in 0..hidden_dims {
            let matrix_base = row * input_dims;
            let adapter_base = row * rank;
            for col in 0..input_dims {
                let delta =
                    target.weights.w1[matrix_base + col] - base.weights.w1[matrix_base + col];
                adapter.w1_up[adapter_base + col] = delta / scale;
            }
        }
        for row in 0..output_dims {
            adapter.w2_up[row * rank + row] = 1.0;
            let matrix_base = row * hidden_dims;
            let adapter_base = row * hidden_dims;
            for col in 0..hidden_dims {
                let delta =
                    target.weights.w2[matrix_base + col] - base.weights.w2[matrix_base + col];
                adapter.w2_down[adapter_base + col] = delta / scale;
            }
        }
        for (delta, (target_value, base_value)) in adapter
            .b1_delta
            .iter_mut()
            .zip(target.weights.b1.iter().zip(base.weights.b1.iter()))
        {
            *delta = target_value - base_value;
        }
        for (delta, (target_value, base_value)) in adapter
            .b2_delta
            .iter_mut()
            .zip(target.weights.b2.iter().zip(base.weights.b2.iter()))
        {
            *delta = target_value - base_value;
        }
        adapter.validate(&base.config)?;
        Ok(adapter)
    }

    pub fn validate(&self, config: &NpaConfig) -> AutomataResult<()> {
        if self.rank == 0 {
            return Err(AutomataError::InvalidModel(
                "low-rank adapter rank must be > 0".to_string(),
            ));
        }
        let input_dims = config.perception_dims();
        let update_dims = config.update_dims();
        let expected = [
            ("w1_down", self.rank * input_dims, self.w1_down.len()),
            ("w1_up", config.hidden_dims * self.rank, self.w1_up.len()),
            (
                "w2_down",
                self.rank * config.hidden_dims,
                self.w2_down.len(),
            ),
            ("w2_up", update_dims * self.rank, self.w2_up.len()),
            ("b1_delta", config.hidden_dims, self.b1_delta.len()),
            ("b2_delta", update_dims, self.b2_delta.len()),
        ];
        for (name, expected_len, actual_len) in expected {
            if actual_len != expected_len {
                return Err(AutomataError::InvalidModel(format!(
                    "{name} len {actual_len} != {expected_len}"
                )));
            }
        }
        Ok(())
    }

    pub fn apply_to_weights(
        &self,
        config: &NpaConfig,
        base: &NpaWeights,
    ) -> AutomataResult<NpaWeights> {
        base.validate(config)?;
        self.validate(config)?;

        let mut adapted = base.clone();
        let scale = self.alpha / self.rank as f32;
        add_low_rank_delta(
            &mut adapted.w1,
            config.hidden_dims,
            config.perception_dims(),
            self.rank,
            &self.w1_up,
            &self.w1_down,
            scale,
        );
        add_low_rank_delta(
            &mut adapted.w2,
            config.update_dims(),
            config.hidden_dims,
            self.rank,
            &self.w2_up,
            &self.w2_down,
            scale,
        );
        for (value, delta) in adapted.b1.iter_mut().zip(&self.b1_delta) {
            *value += delta;
        }
        for (value, delta) in adapted.b2.iter_mut().zip(&self.b2_delta) {
            *value += delta;
        }
        Ok(adapted)
    }

    pub fn apply_to_model(&self, base: &NpaModel) -> AutomataResult<NpaModel> {
        Ok(NpaModel {
            config: base.config.clone(),
            weights: self.apply_to_weights(&base.config, &base.weights)?,
        })
    }
}

fn add_low_rank_delta(
    matrix: &mut [f32],
    rows: usize,
    cols: usize,
    rank: usize,
    up: &[f32],
    down: &[f32],
    scale: f32,
) {
    for row in 0..rows {
        for col in 0..cols {
            let mut delta = 0.0_f32;
            for r in 0..rank {
                delta += up[row * rank + r] * down[r * cols + col];
            }
            matrix[row * cols + col] += scale * delta;
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NpaModel {
    pub config: NpaConfig,
    pub weights: NpaWeights,
}

#[derive(Clone, Debug)]
pub struct StepOutput {
    pub next_positions: Vec<[f32; 4]>,
    pub next_states: Vec<f32>,
    pub dx: Vec<[f32; 4]>,
    pub ds: Vec<f32>,
    pub perception: PerceptionOutput,
}

impl NpaModel {
    pub fn seeded(config: NpaConfig, seed: u64) -> Self {
        let weights = NpaWeights::seeded(&config, seed);
        Self { config, weights }
    }

    pub fn upstream_seeded(config: NpaConfig, seed: u64) -> Self {
        let weights = NpaWeights::upstream_seeded(&config, seed);
        Self { config, weights }
    }

    pub fn validate(&self) -> AutomataResult<()> {
        if !(self.config.spatial_dims == 2 || self.config.spatial_dims == 3) {
            return Err(AutomataError::InvalidModel(format!(
                "spatial_dims must be 2 or 3, got {}",
                self.config.spatial_dims
            )));
        }
        self.weights.validate(&self.config)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_cpu(
        &self,
        positions: &[[f32; 4]],
        states: &[f32],
        batch_size: usize,
        particle_count: usize,
        grid: &HashGridConfig,
        dt: f32,
        update_mask: Option<&[f32]>,
    ) -> AutomataResult<StepOutput> {
        self.validate()?;
        if grid.dim != self.config.spatial_dims {
            return Err(AutomataError::InvalidArgument(format!(
                "grid dim {} does not match model spatial dims {}",
                grid.dim, self.config.spatial_dims
            )));
        }
        let perception = perceive_with_options(
            positions,
            states,
            batch_size,
            particle_count,
            self.config.state_dims,
            grid,
            PerceptionOptions {
                state_grad: self.config.state_grad,
                density_grad: self.config.density_grad,
                eps0: self.config.eps0,
                scale_equivariance: self.config.scale_equivariant(),
                particle_density_equivariance: self.config.particle_density_equivariant(),
                log_norm_grad: self.config.log_norm_grad,
                log_norm_density_grad: self.config.log_norm_density_grad,
                hybrid_state_gradient: true,
                position_features: self.config.position_features,
            },
        )?;
        let (dx, ds) = self.forward_from_features_with_eps(&perception.features, grid.eps)?;
        let (next_positions, next_states) = euler_step(
            positions,
            states,
            &dx,
            &ds,
            batch_size,
            particle_count,
            self.config.state_dims,
            grid,
            dt,
            update_mask,
        )?;

        Ok(StepOutput {
            next_positions,
            next_states,
            dx,
            ds,
            perception,
        })
    }

    pub fn forward_from_features(
        &self,
        features: &[f32],
    ) -> AutomataResult<(Vec<[f32; 4]>, Vec<f32>)> {
        self.forward_from_features_with_eps(features, self.config.eps0)
    }

    pub fn forward_from_features_with_eps(
        &self,
        features: &[f32],
        eps: f32,
    ) -> AutomataResult<(Vec<[f32; 4]>, Vec<f32>)> {
        let update = self.forward_update_from_features(features)?;
        let input_dims = self.config.perception_dims();
        let rows = features.len() / input_dims;
        let output_dims = self.config.update_dims();
        let mut dx = vec![[0.0; 4]; rows];
        let mut ds = vec![0.0; rows * self.config.state_dims];

        for row in 0..rows {
            let row_update = &update[row * output_dims..(row + 1) * output_dims];
            let mut norm = 0.0;
            for value in row_update.iter().take(self.config.spatial_dims) {
                norm += value * value;
            }
            norm = norm.sqrt();
            let motion_eps = self.config.motion_eps(eps);
            for (axis, value) in row_update.iter().enumerate().take(self.config.spatial_dims) {
                dx[row][axis] = self.config.alpha * *value * motion_eps / (1.0 + norm);
            }
            let state_base = row * self.config.state_dims;
            let update_state_base = self.config.spatial_dims;
            ds[state_base..state_base + self.config.state_dims].copy_from_slice(
                &row_update[update_state_base..update_state_base + self.config.state_dims],
            );
        }

        Ok((dx, ds))
    }

    pub fn forward_update_from_features(&self, features: &[f32]) -> AutomataResult<Vec<f32>> {
        let input_dims = self.config.perception_dims();
        if !features.len().is_multiple_of(input_dims) {
            return Err(AutomataError::InvalidArgument(format!(
                "feature len {} is not divisible by perception dims {input_dims}",
                features.len()
            )));
        }
        let rows = features.len() / input_dims;
        let output_dims = self.config.update_dims();
        let mut update_rows = vec![0.0; rows * output_dims];

        update_rows
            .par_chunks_mut(output_dims)
            .enumerate()
            .for_each_init(
                || vec![0.0; self.config.hidden_dims],
                |hidden, (row, update)| {
                    let feature = &features[row * input_dims..(row + 1) * input_dims];
                    for (h, hidden_value) in hidden.iter_mut().enumerate() {
                        let mut sum = self.weights.b1[h];
                        let w_base = h * input_dims;
                        for (i, value) in feature.iter().enumerate().take(input_dims) {
                            sum += self.weights.w1[w_base + i] * *value;
                        }
                        *hidden_value = sum.max(0.0);
                    }

                    for (o, update_value) in update.iter_mut().enumerate() {
                        let mut sum = self.weights.b2[o];
                        let w_base = o * self.config.hidden_dims;
                        for (h, value) in hidden.iter().enumerate().take(self.config.hidden_dims) {
                            sum += self.weights.w2[w_base + h] * *value;
                        }
                        *update_value = sum;
                    }
                },
            );

        Ok(update_rows)
    }

    pub fn forward_update_tensor<B: Backend>(
        &self,
        features: Tensor<B, 2>,
        device: &B::Device,
    ) -> AutomataResult<Tensor<B, 2>> {
        self.validate()?;
        let input_dims = self.config.perception_dims();
        let dims: [usize; 2] = features.shape().dims();
        if dims[1] != input_dims {
            return Err(AutomataError::InvalidArgument(format!(
                "feature tensor shape {:?} does not end with perception dims {input_dims}",
                dims
            )));
        }

        let rows = dims[0];
        let hidden_dims = self.config.hidden_dims;
        let output_dims = self.config.update_dims();
        let w1 = Tensor::<B, 2>::from_data(
            TensorData::new(self.weights.w1.clone(), [hidden_dims, input_dims]),
            device,
        )
        .transpose();
        let b1 = Tensor::<B, 2>::from_data(
            TensorData::new(repeated_rows(&self.weights.b1, rows), [rows, hidden_dims]),
            device,
        );
        let w2 = Tensor::<B, 2>::from_data(
            TensorData::new(self.weights.w2.clone(), [output_dims, hidden_dims]),
            device,
        )
        .transpose();
        let b2 = Tensor::<B, 2>::from_data(
            TensorData::new(repeated_rows(&self.weights.b2, rows), [rows, output_dims]),
            device,
        );

        Ok(relu(features.matmul(w1) + b1).matmul(w2) + b2)
    }
}

fn fill_xavier_uniform(
    values: &mut [f32],
    fan_in: usize,
    fan_out: usize,
    gain: f32,
    rng: &mut StdRng,
) {
    let denom = (fan_in + fan_out).max(1) as f32;
    let bound = gain * (6.0 / denom).sqrt();
    for value in values {
        *value = rng.random_range(-bound..bound);
    }
}

fn repeated_rows(values: &[f32], rows: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(values.len() * rows);
    for _ in 0..rows {
        out.extend_from_slice(values);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AutomataPreset;

    #[test]
    fn low_rank_adapter_parameter_count_is_smaller_than_full_weights() {
        let config = NpaConfig::for_preset(AutomataPreset::Growing3dGs).0;
        let adapter = NpaLowRankAdapter::zeros(&config, 2, 2.0);
        let full = NpaWeights::zeros(&config);
        let full_count = full.w1.len() + full.b1.len() + full.w2.len() + full.b2.len();

        adapter.validate(&config).unwrap();
        assert!(adapter.parameter_count() < full_count);
    }

    #[test]
    fn upstream_seeded_matches_npa_initializer_shape() {
        let config = NpaConfig::growing_2d();
        let weights = NpaWeights::upstream_seeded(&config, 42);
        let w1_bound = 0.1 * (6.0 / (config.perception_dims() + config.hidden_dims) as f32).sqrt();
        let w2_bound = 0.1 * (6.0 / (config.hidden_dims + config.update_dims()) as f32).sqrt();

        assert!(weights.w1.iter().all(|value| value.abs() <= w1_bound));
        assert!(weights.w2.iter().all(|value| value.abs() <= w2_bound));
        assert!(weights.b1.iter().any(|value| *value != 0.0));
        assert!(weights.b2.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn low_rank_adapter_materializes_expected_weight_delta() {
        let mut config = NpaConfig::for_preset(AutomataPreset::Growing3dGs).0;
        config.hidden_dims = 2;
        config.state_dims = 4;
        config.position_features = false;
        let input_dims = config.perception_dims();
        let update_dims = config.update_dims();
        let base = NpaWeights::zeros(&config);
        let mut adapter = NpaLowRankAdapter::zeros(&config, 1, 2.0);

        adapter.w1_up = vec![3.0, -5.0];
        adapter.w1_down = vec![0.0; input_dims];
        adapter.w1_down[0] = 7.0;
        adapter.w2_up = vec![0.0; update_dims];
        adapter.w2_up[0] = 11.0;
        adapter.w2_down = vec![0.0; config.hidden_dims];
        adapter.w2_down[1] = 13.0;
        adapter.b1_delta = vec![0.25, -0.5];
        adapter.b2_delta = vec![0.75; update_dims];

        let adapted = adapter.apply_to_weights(&config, &base).unwrap();

        assert_eq!(adapted.w1[0], 42.0);
        assert_eq!(adapted.w1[input_dims], -70.0);
        assert_eq!(adapted.w2[1], 286.0);
        assert_eq!(adapted.b1, vec![0.25, -0.5]);
        assert_eq!(adapted.b2, vec![0.75; update_dims]);
    }

    #[test]
    fn low_rank_adapter_rejects_mismatched_dimensions() {
        let config = NpaConfig::for_preset(AutomataPreset::Growing3dGs).0;
        let mut adapter = NpaLowRankAdapter::zeros(&config, 2, 1.0);
        adapter.w1_down.pop();

        let err = adapter.validate(&config).unwrap_err().to_string();
        assert!(err.contains("w1_down len"));
    }

    #[test]
    fn low_rank_adapter_parameter_vector_roundtrips() {
        let config = NpaConfig::for_preset(AutomataPreset::Growing2d).0;
        let adapter = NpaLowRankAdapter::seeded(&config, 3, 2.0, 19);
        let values = adapter.to_parameter_vector();

        assert_eq!(
            values.len(),
            NpaLowRankAdapter::parameter_count_for_config(&config, 3)
        );
        let restored =
            NpaLowRankAdapter::from_parameter_vector(&config, 3, 2.0, values.clone()).unwrap();

        assert_eq!(restored.to_parameter_vector(), values);
        assert_eq!(restored.rank, 3);
        assert_eq!(restored.alpha, 2.0);
    }

    #[test]
    fn exact_low_rank_adapter_reconstructs_target_weights() {
        let config = NpaConfig {
            state_dims: 2,
            hidden_dims: 4,
            ..NpaConfig::growing_2d()
        };
        let base = NpaModel {
            weights: NpaWeights::seeded(&config, 1),
            config: config.clone(),
        };
        let target = NpaModel {
            weights: NpaWeights::seeded(&config, 2),
            config: config.clone(),
        };
        let rank = config.perception_dims().max(config.update_dims());
        let adapter =
            NpaLowRankAdapter::exact_model_delta(&base, &target, rank, rank as f32).unwrap();
        let reconstructed = adapter.apply_to_model(&base).unwrap();

        assert_eq!(reconstructed.config, target.config);
        for (actual, expected) in reconstructed
            .weights
            .w1
            .iter()
            .chain(reconstructed.weights.b1.iter())
            .chain(reconstructed.weights.w2.iter())
            .chain(reconstructed.weights.b2.iter())
            .zip(
                target
                    .weights
                    .w1
                    .iter()
                    .chain(target.weights.b1.iter())
                    .chain(target.weights.w2.iter())
                    .chain(target.weights.b2.iter()),
            )
        {
            assert!((actual - expected).abs() <= 1.0e-6);
        }
    }
}
