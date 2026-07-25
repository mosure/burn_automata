use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{AutomataError, AutomataResult};

pub const ADAPTIVE_CONTROLLER_SCALAR_DIMS: usize = 8;
pub const ADAPTIVE_CONTROLLER_CONTEXT_DIMS: usize = 192;
pub const ADAPTIVE_CONTROLLER_INPUT_DIMS: usize =
    ADAPTIVE_CONTROLLER_SCALAR_DIMS + ADAPTIVE_CONTROLLER_CONTEXT_DIMS;
pub const ADAPTIVE_CONTROLLER_OUTPUT_DIMS: usize = 4;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveControllerWeights {
    pub input_weights: Vec<f32>,
    pub input_bias: Vec<f32>,
    pub output_weights: Vec<f32>,
    pub output_bias: Vec<f32>,
}

impl AdaptiveControllerWeights {
    pub fn seeded(hidden_dims: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut input_weights = vec![0.0; hidden_dims * ADAPTIVE_CONTROLLER_INPUT_DIMS];
        let mut output_weights = vec![0.0; ADAPTIVE_CONTROLLER_OUTPUT_DIMS * hidden_dims];
        let input_bound = (6.0 / (ADAPTIVE_CONTROLLER_INPUT_DIMS + hidden_dims) as f32).sqrt();
        let output_bound = (6.0 / (hidden_dims + ADAPTIVE_CONTROLLER_OUTPUT_DIMS) as f32).sqrt();
        input_weights
            .iter_mut()
            .for_each(|value| *value = rng.random_range(-input_bound..input_bound));
        output_weights
            .iter_mut()
            .for_each(|value| *value = rng.random_range(-output_bound..output_bound));
        // Start close to a fixed-resolution NPA. The controller must earn topology changes.
        output_weights.iter_mut().for_each(|value| *value *= 0.01);
        Self {
            input_weights,
            input_bias: vec![0.0; hidden_dims],
            output_weights,
            output_bias: vec![0.0, 0.0, -4.0, -4.0],
        }
    }

    pub fn validate(&self, hidden_dims: usize) -> AutomataResult<()> {
        let expected_input = hidden_dims * ADAPTIVE_CONTROLLER_INPUT_DIMS;
        let expected_output = ADAPTIVE_CONTROLLER_OUTPUT_DIMS * hidden_dims;
        if self.input_weights.len() != expected_input
            || self.input_bias.len() != hidden_dims
            || self.output_weights.len() != expected_output
            || self.output_bias.len() != ADAPTIVE_CONTROLLER_OUTPUT_DIMS
        {
            return Err(AutomataError::InvalidModel(format!(
                "adaptive controller shape mismatch: input={}/{} input_bias={}/{} output={}/{} output_bias={}/{}",
                self.input_weights.len(),
                expected_input,
                self.input_bias.len(),
                hidden_dims,
                self.output_weights.len(),
                expected_output,
                self.output_bias.len(),
                ADAPTIVE_CONTROLLER_OUTPUT_DIMS
            )));
        }
        if self
            .input_weights
            .iter()
            .chain(&self.input_bias)
            .chain(&self.output_weights)
            .chain(&self.output_bias)
            .any(|value| !value.is_finite())
        {
            return Err(AutomataError::InvalidModel(
                "adaptive controller contains non-finite weights".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AdaptiveControllerOutput {
    pub desired_log_footprint: f32,
    pub log_bandwidth_ratio: f32,
    pub split_probability: f32,
    pub merge_probability: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveController {
    pub hidden_dims: usize,
    pub weights: AdaptiveControllerWeights,
}

impl AdaptiveController {
    pub fn seeded(hidden_dims: usize, seed: u64) -> Self {
        Self {
            hidden_dims,
            weights: AdaptiveControllerWeights::seeded(hidden_dims, seed),
        }
    }

    pub fn validate(&self) -> AutomataResult<()> {
        if self.hidden_dims == 0 {
            return Err(AutomataError::InvalidModel(
                "adaptive controller hidden_dims must be non-zero".to_string(),
            ));
        }
        self.weights.validate(self.hidden_dims)
    }

    pub fn forward(
        &self,
        features: &[[f32; ADAPTIVE_CONTROLLER_INPUT_DIMS]],
    ) -> Vec<AdaptiveControllerOutput> {
        features
            .par_iter()
            .map_init(
                || vec![0.0; self.hidden_dims],
                |hidden, feature| self.forward_one(feature, hidden),
            )
            .collect()
    }

    pub fn forward_raw(&self, features: &[f32]) -> AutomataResult<Vec<f32>> {
        if !features
            .len()
            .is_multiple_of(ADAPTIVE_CONTROLLER_INPUT_DIMS)
        {
            return Err(AutomataError::InvalidArgument(format!(
                "adaptive controller feature len {} is not divisible by {}",
                features.len(),
                ADAPTIVE_CONTROLLER_INPUT_DIMS
            )));
        }
        let rows = features.len() / ADAPTIVE_CONTROLLER_INPUT_DIMS;
        let mut output = vec![0.0; rows * ADAPTIVE_CONTROLLER_OUTPUT_DIMS];
        output
            .par_chunks_mut(ADAPTIVE_CONTROLLER_OUTPUT_DIMS)
            .enumerate()
            .for_each_init(
                || vec![0.0; self.hidden_dims],
                |hidden, (row, dst)| {
                    let feature: &[f32; ADAPTIVE_CONTROLLER_INPUT_DIMS] = features[row
                        * ADAPTIVE_CONTROLLER_INPUT_DIMS
                        ..(row + 1) * ADAPTIVE_CONTROLLER_INPUT_DIMS]
                        .try_into()
                        .expect("controller row has static width");
                    let raw = self.forward_raw_one(feature, hidden);
                    dst.copy_from_slice(&raw);
                },
            );
        Ok(output)
    }

    fn forward_one(
        &self,
        feature: &[f32; ADAPTIVE_CONTROLLER_INPUT_DIMS],
        hidden: &mut [f32],
    ) -> AdaptiveControllerOutput {
        let raw = self.forward_raw_one(feature, hidden);
        AdaptiveControllerOutput {
            desired_log_footprint: raw[0],
            log_bandwidth_ratio: raw[1].clamp(-1.5, 1.5),
            split_probability: sigmoid(raw[2]),
            merge_probability: sigmoid(raw[3]),
        }
    }

    fn forward_raw_one(
        &self,
        feature: &[f32; ADAPTIVE_CONTROLLER_INPUT_DIMS],
        hidden: &mut [f32],
    ) -> [f32; 4] {
        for (row, value) in hidden.iter_mut().enumerate() {
            let base = row * ADAPTIVE_CONTROLLER_INPUT_DIMS;
            let sum = feature
                .iter()
                .enumerate()
                .fold(self.weights.input_bias[row], |sum, (col, input)| {
                    sum + self.weights.input_weights[base + col] * input
                });
            *value = sum.max(0.0);
        }
        let mut output = [0.0; ADAPTIVE_CONTROLLER_OUTPUT_DIMS];
        for (row, value) in output.iter_mut().enumerate() {
            let base = row * self.hidden_dims;
            *value = hidden
                .iter()
                .enumerate()
                .fold(self.weights.output_bias[row], |sum, (col, input)| {
                    sum + self.weights.output_weights[base + col] * input
                });
        }
        output
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        let exponential = (-value).exp();
        1.0 / (1.0 + exponential)
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}
