use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

use crate::{AutomataError, AutomataResult, NpaConfig, NpaLowRankAdapter};

use super::condition::{
    ConditionEncoder2d, ConditionImage2d, DEFAULT_CONDITION_TOKEN_GRID_HEIGHT,
    DEFAULT_CONDITION_TOKEN_GRID_WIDTH, condition_feature_dims_for_encoder,
};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct HyperNpa2dConfig {
    #[serde(default)]
    pub condition_encoder: ConditionEncoder2d,
    pub condition_feature_dims: usize,
    #[serde(default)]
    pub condition_token_grid_width: usize,
    #[serde(default)]
    pub condition_token_grid_height: usize,
    pub hidden_dims: usize,
    pub adapter_rank: usize,
    pub adapter_alpha: f32,
    #[serde(default)]
    pub adapter_bias_correction: bool,
    #[serde(default)]
    pub output_activation: HyperNpa2dOutputActivation,
    pub output_scale: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HyperNpa2dOutputActivation {
    #[default]
    Tanh,
    Linear,
}

impl Default for HyperNpa2dConfig {
    fn default() -> Self {
        let condition_encoder = ConditionEncoder2d::SummaryTokens;
        let condition_feature_dims = condition_feature_dims_for_encoder(
            condition_encoder,
            DEFAULT_CONDITION_TOKEN_GRID_WIDTH,
            DEFAULT_CONDITION_TOKEN_GRID_HEIGHT,
        )
        .expect("default condition token grid is valid");
        Self {
            condition_encoder,
            condition_feature_dims,
            condition_token_grid_width: DEFAULT_CONDITION_TOKEN_GRID_WIDTH,
            condition_token_grid_height: DEFAULT_CONDITION_TOKEN_GRID_HEIGHT,
            hidden_dims: 32,
            adapter_rank: 2,
            adapter_alpha: 2.0,
            adapter_bias_correction: false,
            output_activation: HyperNpa2dOutputActivation::Tanh,
            output_scale: 0.05,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyperNpa2dWeights {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyperNpa2dPreciseWeights {
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyperNpa2d {
    pub npa_config: NpaConfig,
    pub config: HyperNpa2dConfig,
    pub weights: HyperNpa2dWeights,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precise_weights: Option<HyperNpa2dPreciseWeights>,
    #[serde(default)]
    pub anchor_input: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flow: Option<HyperNpa2dFlow>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyperNpa2dFlow {
    pub config: HyperNpa2dFlowConfig,
    pub weights: HyperNpa2dFlowWeights,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct HyperNpa2dFlowConfig {
    pub hidden_dims: usize,
    pub sample_steps: usize,
    pub source_scale: f32,
    pub sample_seed: u64,
    #[serde(default)]
    pub hidden_activation: HyperNpa2dFlowActivation,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HyperNpa2dFlowActivation {
    #[default]
    Relu,
    LeakyRelu,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyperNpa2dFlowWeights {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

#[derive(Clone, Debug)]
pub(crate) struct HyperForwardCache {
    pub input: Vec<f32>,
    pub pre_hidden: Vec<f32>,
    pub hidden: Vec<f32>,
    pub pre_output: Vec<f32>,
    pub output: Vec<f32>,
    pub anchor: Option<HyperLayerCache>,
}

#[derive(Clone, Debug)]
pub(crate) struct HyperLayerCache {
    pub input: Vec<f32>,
    pub pre_hidden: Vec<f32>,
    pub hidden: Vec<f32>,
    pub pre_output: Vec<f32>,
    pub output: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
struct HyperLayerCacheRef<'a> {
    input: &'a [f32],
    pre_hidden: &'a [f32],
    hidden: &'a [f32],
    pre_output: &'a [f32],
}

#[derive(Clone, Debug)]
pub(crate) struct HyperNpa2dGradients {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

impl HyperNpa2d {
    pub fn zeros(npa_config: NpaConfig, config: HyperNpa2dConfig) -> AutomataResult<Self> {
        validate_hyper_config(&npa_config, config)?;
        let output_dims = NpaLowRankAdapter::parameter_count_for_config_with_bias_correction(
            &npa_config,
            config.adapter_rank,
            config.adapter_bias_correction,
        );
        Ok(Self {
            npa_config,
            config,
            weights: HyperNpa2dWeights::zeros(
                config.condition_feature_dims,
                config.hidden_dims,
                output_dims,
            ),
            precise_weights: None,
            anchor_input: None,
            flow: None,
        })
    }

    pub fn seeded(
        npa_config: NpaConfig,
        config: HyperNpa2dConfig,
        seed: u64,
    ) -> AutomataResult<Self> {
        validate_hyper_config(&npa_config, config)?;
        let output_dims = NpaLowRankAdapter::parameter_count_for_config_with_bias_correction(
            &npa_config,
            config.adapter_rank,
            config.adapter_bias_correction,
        );
        Ok(Self {
            npa_config,
            config,
            weights: HyperNpa2dWeights::seeded(
                config.condition_feature_dims,
                config.hidden_dims,
                output_dims,
                seed,
            ),
            precise_weights: None,
            anchor_input: None,
            flow: None,
        })
    }

    pub fn validate(&self) -> AutomataResult<()> {
        validate_hyper_config(&self.npa_config, self.config)?;
        let output_dims = self.adapter_parameter_count();
        let expected = [
            (
                "hyper w1",
                self.config.hidden_dims * self.config.condition_feature_dims,
                self.weights.w1.len(),
            ),
            ("hyper b1", self.config.hidden_dims, self.weights.b1.len()),
            (
                "hyper w2",
                output_dims * self.config.hidden_dims,
                self.weights.w2.len(),
            ),
            ("hyper b2", output_dims, self.weights.b2.len()),
        ];
        for (name, expected_len, actual_len) in expected {
            if actual_len != expected_len {
                return Err(AutomataError::InvalidModel(format!(
                    "{name} len {actual_len} != {expected_len}"
                )));
            }
        }
        ensure_finite("hyper w1", &self.weights.w1)?;
        ensure_finite("hyper b1", &self.weights.b1)?;
        ensure_finite("hyper w2", &self.weights.w2)?;
        ensure_finite("hyper b2", &self.weights.b2)?;
        if let Some(precise) = &self.precise_weights {
            let expected = [
                (
                    "hyper precise w1",
                    self.config.hidden_dims * self.config.condition_feature_dims,
                    precise.w1.len(),
                ),
                (
                    "hyper precise b1",
                    self.config.hidden_dims,
                    precise.b1.len(),
                ),
                (
                    "hyper precise w2",
                    output_dims * self.config.hidden_dims,
                    precise.w2.len(),
                ),
                ("hyper precise b2", output_dims, precise.b2.len()),
            ];
            for (name, expected_len, actual_len) in expected {
                if actual_len != expected_len {
                    return Err(AutomataError::InvalidModel(format!(
                        "{name} len {actual_len} != {expected_len}"
                    )));
                }
            }
            ensure_finite_f64("hyper precise w1", &precise.w1)?;
            ensure_finite_f64("hyper precise b1", &precise.b1)?;
            ensure_finite_f64("hyper precise w2", &precise.w2)?;
            ensure_finite_f64("hyper precise b2", &precise.b2)?;
        }
        if let Some(anchor_input) = &self.anchor_input {
            if anchor_input.len() != self.config.condition_feature_dims {
                return Err(AutomataError::InvalidModel(format!(
                    "hyper anchor input len {} != {}",
                    anchor_input.len(),
                    self.config.condition_feature_dims
                )));
            }
            ensure_finite("hyper anchor input", anchor_input)?;
        }
        if let Some(flow) = &self.flow {
            validate_flow_head(self, flow)?;
        }
        Ok(())
    }

    pub fn adapter_parameter_count(&self) -> usize {
        NpaLowRankAdapter::parameter_count_for_config_with_bias_correction(
            &self.npa_config,
            self.config.adapter_rank,
            self.config.adapter_bias_correction,
        )
    }

    pub fn predict_adapter(
        &self,
        condition: &ConditionImage2d,
    ) -> AutomataResult<NpaLowRankAdapter> {
        let values = self.predict_adapter_vector(condition)?;
        NpaLowRankAdapter::from_parameter_vector_with_bias_correction(
            &self.npa_config,
            self.config.adapter_rank,
            self.config.adapter_alpha,
            values,
            self.config.adapter_bias_correction,
        )
    }

    pub fn predict_adapter_vector(&self, condition: &ConditionImage2d) -> AutomataResult<Vec<f32>> {
        if self.flow.is_some() {
            return self.predict_adapter_vector_flow(condition);
        }
        Ok(self.forward_cache(condition)?.output)
    }

    pub fn set_flow(&mut self, flow: HyperNpa2dFlow) -> AutomataResult<()> {
        self.flow = Some(flow);
        self.validate()
    }

    pub fn set_anchor_condition(&mut self, condition: &ConditionImage2d) -> AutomataResult<()> {
        let input = self.condition_input(condition)?;
        self.anchor_input = Some(input);
        self.validate()
    }

    pub fn condition_input_vector(&self, condition: &ConditionImage2d) -> AutomataResult<Vec<f32>> {
        self.condition_input(condition)
    }

    pub(crate) fn forward_cache(
        &self,
        condition: &ConditionImage2d,
    ) -> AutomataResult<HyperForwardCache> {
        self.validate()?;
        let input = self.condition_input(condition)?;
        let mut layer = self.forward_layer(input);
        let anchor = self
            .anchor_input
            .as_ref()
            .map(|anchor_input| self.forward_layer(anchor_input.clone()));
        if let Some(anchor) = &anchor {
            for (output, anchor_output) in layer.output.iter_mut().zip(anchor.output.iter()) {
                *output -= *anchor_output;
            }
        }
        Ok(HyperForwardCache {
            input: layer.input,
            pre_hidden: layer.pre_hidden,
            hidden: layer.hidden,
            pre_output: layer.pre_output,
            output: layer.output,
            anchor,
        })
    }

    fn condition_input(&self, condition: &ConditionImage2d) -> AutomataResult<Vec<f32>> {
        let input = condition.feature_vector_for_encoder(
            self.config.condition_encoder,
            self.config.condition_token_grid_width,
            self.config.condition_token_grid_height,
        )?;
        if input.len() != self.config.condition_feature_dims {
            return Err(AutomataError::InvalidArgument(format!(
                "condition feature len {} != {}",
                input.len(),
                self.config.condition_feature_dims
            )));
        }
        Ok(input)
    }

    fn predict_adapter_vector_flow(
        &self,
        condition: &ConditionImage2d,
    ) -> AutomataResult<Vec<f32>> {
        self.validate()?;
        let flow = self
            .flow
            .as_ref()
            .ok_or_else(|| AutomataError::InvalidModel("missing HyperNPA flow head".to_string()))?;
        let condition_input = self.condition_input(condition)?;
        let output_dims = self.adapter_parameter_count();
        let steps = flow.config.sample_steps.max(1);
        let mut state = seeded_flow_source(
            &condition_input,
            output_dims,
            flow.config.source_scale,
            flow.config.sample_seed,
        );
        let dt = 1.0 / steps as f32;
        for step in 0..steps {
            let t = (step as f32 + 0.5) * dt;
            let velocity = self.flow_velocity(flow, &condition_input, t, &state)?;
            for (value, delta) in state.iter_mut().zip(velocity) {
                *value += delta * dt;
            }
        }
        ensure_finite("hyper flow sampled adapter", &state)?;
        Ok(state)
    }

    fn flow_velocity(
        &self,
        flow: &HyperNpa2dFlow,
        condition_input: &[f32],
        t: f32,
        state: &[f32],
    ) -> AutomataResult<Vec<f32>> {
        let input = flow_input(condition_input, t, state);
        let output_dims = self.adapter_parameter_count();
        let mut hidden = vec![0.0; flow.config.hidden_dims];
        for (h, hidden_value) in hidden.iter_mut().enumerate() {
            let mut sum = flow.weights.b1[h] as f64;
            let base = h * input.len();
            for (i, value) in input.iter().enumerate() {
                sum += flow.weights.w1[base + i] as f64 * *value as f64;
            }
            *hidden_value = activate_flow_hidden(flow.config.hidden_activation, sum as f32);
        }
        let mut output = vec![0.0; output_dims];
        for (o, output_value) in output.iter_mut().enumerate() {
            let mut sum = flow.weights.b2[o] as f64;
            let base = o * flow.config.hidden_dims;
            for (h, value) in hidden.iter().enumerate() {
                sum += flow.weights.w2[base + h] as f64 * *value as f64;
            }
            *output_value = sum as f32;
        }
        ensure_finite("hyper flow velocity", &output)?;
        Ok(output)
    }

    fn forward_layer(&self, input: Vec<f32>) -> HyperLayerCache {
        if let Some(precise) = &self.precise_weights {
            return self.forward_precise_layer(input, precise);
        }
        let mut pre_hidden = vec![0.0; self.config.hidden_dims];
        let mut hidden = vec![0.0; self.config.hidden_dims];
        for h in 0..self.config.hidden_dims {
            let mut sum = self.weights.b1[h] as f64;
            let base = h * self.config.condition_feature_dims;
            for (i, value) in input.iter().enumerate() {
                sum += self.weights.w1[base + i] as f64 * *value as f64;
            }
            let sum = sum as f32;
            pre_hidden[h] = sum;
            hidden[h] = sum.max(0.0);
        }

        let output_dims = self.adapter_parameter_count();
        let mut pre_output = vec![0.0; output_dims];
        let mut output = vec![0.0; output_dims];
        for o in 0..output_dims {
            let mut sum = self.weights.b2[o] as f64;
            let base = o * self.config.hidden_dims;
            for (h, value) in hidden.iter().enumerate() {
                sum += self.weights.w2[base + h] as f64 * *value as f64;
            }
            let pre = sum as f32;
            pre_output[o] = pre;
            output[o] = self.activate_output(sum);
        }

        HyperLayerCache {
            input,
            pre_hidden,
            hidden,
            pre_output,
            output,
        }
    }

    fn forward_precise_layer(
        &self,
        input: Vec<f32>,
        weights: &HyperNpa2dPreciseWeights,
    ) -> HyperLayerCache {
        let mut pre_hidden = vec![0.0; self.config.hidden_dims];
        let mut hidden = vec![0.0; self.config.hidden_dims];
        for h in 0..self.config.hidden_dims {
            let mut sum = weights.b1[h];
            let base = h * self.config.condition_feature_dims;
            for (i, value) in input.iter().enumerate() {
                sum += weights.w1[base + i] * *value as f64;
            }
            let sum = sum as f32;
            pre_hidden[h] = sum;
            hidden[h] = sum.max(0.0);
        }

        let output_dims = self.adapter_parameter_count();
        let mut pre_output = vec![0.0; output_dims];
        let mut output = vec![0.0; output_dims];
        for o in 0..output_dims {
            let mut sum = weights.b2[o];
            let base = o * self.config.hidden_dims;
            for (h, value) in hidden.iter().enumerate() {
                sum += weights.w2[base + h] * *value as f64;
            }
            pre_output[o] = sum as f32;
            output[o] = self.activate_output(sum);
        }

        HyperLayerCache {
            input,
            pre_hidden,
            hidden,
            pre_output,
            output,
        }
    }

    fn activate_output(&self, value: f64) -> f32 {
        match self.config.output_activation {
            HyperNpa2dOutputActivation::Tanh => {
                (value.tanh() * self.config.output_scale as f64) as f32
            }
            HyperNpa2dOutputActivation::Linear => value as f32,
        }
    }

    pub(crate) fn accumulate_output_gradients(
        &self,
        cache: &HyperForwardCache,
        output_gradients: &[f32],
        scale: f32,
        grads: &mut HyperNpa2dGradients,
    ) -> AutomataResult<()> {
        if output_gradients.len() != self.adapter_parameter_count() {
            return Err(AutomataError::InvalidArgument(format!(
                "hyper output gradient len {} != {}",
                output_gradients.len(),
                self.adapter_parameter_count()
            )));
        }
        if !scale.is_finite() {
            return Err(AutomataError::InvalidArgument(format!(
                "hyper gradient scale must be finite, got {scale}"
            )));
        }
        self.accumulate_layer_output_gradients(cache.primary_ref(), output_gradients, scale, grads);
        if let Some(anchor) = &cache.anchor {
            self.accumulate_layer_output_gradients(
                anchor.cache_ref(),
                output_gradients,
                -scale,
                grads,
            );
        }
        Ok(())
    }

    fn accumulate_layer_output_gradients(
        &self,
        cache: HyperLayerCacheRef<'_>,
        output_gradients: &[f32],
        scale: f32,
        grads: &mut HyperNpa2dGradients,
    ) {
        let mut hidden_grads = vec![0.0; self.config.hidden_dims];
        for (o, output_grad) in output_gradients.iter().copied().enumerate() {
            let d_pre_output = match self.config.output_activation {
                HyperNpa2dOutputActivation::Tanh => {
                    let tanh_pre = cache.pre_output[o].tanh();
                    output_grad * self.config.output_scale * (1.0 - tanh_pre * tanh_pre) * scale
                }
                HyperNpa2dOutputActivation::Linear => output_grad * scale,
            };
            grads.b2[o] += d_pre_output;
            let w2_base = o * self.config.hidden_dims;
            for (h, hidden_grad) in hidden_grads.iter_mut().enumerate() {
                grads.w2[w2_base + h] += d_pre_output * cache.hidden[h];
                *hidden_grad += d_pre_output * self.weights.w2[w2_base + h];
            }
        }
        for (h, hidden_grad) in hidden_grads.into_iter().enumerate() {
            let d_pre_hidden = if cache.pre_hidden[h] > 0.0 {
                hidden_grad
            } else {
                0.0
            };
            grads.b1[h] += d_pre_hidden;
            let w1_base = h * self.config.condition_feature_dims;
            for (i, input_value) in cache.input.iter().enumerate() {
                grads.w1[w1_base + i] += d_pre_hidden * *input_value;
            }
        }
    }
}

impl HyperForwardCache {
    fn primary_ref(&self) -> HyperLayerCacheRef<'_> {
        HyperLayerCacheRef {
            input: &self.input,
            pre_hidden: &self.pre_hidden,
            hidden: &self.hidden,
            pre_output: &self.pre_output,
        }
    }
}

impl<'a> From<&'a HyperLayerCache> for HyperLayerCacheRef<'a> {
    fn from(cache: &'a HyperLayerCache) -> Self {
        Self {
            input: &cache.input,
            pre_hidden: &cache.pre_hidden,
            hidden: &cache.hidden,
            pre_output: &cache.pre_output,
        }
    }
}

impl HyperLayerCache {
    fn cache_ref(&self) -> HyperLayerCacheRef<'_> {
        self.into()
    }
}

impl HyperNpa2dWeights {
    pub fn zeros(input_dims: usize, hidden_dims: usize, output_dims: usize) -> Self {
        Self {
            w1: vec![0.0; hidden_dims * input_dims],
            b1: vec![0.0; hidden_dims],
            w2: vec![0.0; output_dims * hidden_dims],
            b2: vec![0.0; output_dims],
        }
    }

    pub fn seeded(input_dims: usize, hidden_dims: usize, output_dims: usize, seed: u64) -> Self {
        let mut weights = Self::zeros(input_dims, hidden_dims, output_dims);
        let mut rng = StdRng::seed_from_u64(seed);
        for value in weights.w1.iter_mut().chain(weights.w2.iter_mut()) {
            *value = rng.random_range(-0.02..0.02);
        }
        weights
    }
}

impl HyperNpa2dPreciseWeights {
    pub fn zeros(input_dims: usize, hidden_dims: usize, output_dims: usize) -> Self {
        Self {
            w1: vec![0.0; hidden_dims * input_dims],
            b1: vec![0.0; hidden_dims],
            w2: vec![0.0; output_dims * hidden_dims],
            b2: vec![0.0; output_dims],
        }
    }
}

impl HyperNpa2dGradients {
    pub(crate) fn zeros_like(model: &HyperNpa2d) -> Self {
        Self {
            w1: vec![0.0; model.weights.w1.len()],
            b1: vec![0.0; model.weights.b1.len()],
            w2: vec![0.0; model.weights.w2.len()],
            b2: vec![0.0; model.weights.b2.len()],
        }
    }

    pub(crate) fn grad_norm(&self) -> f32 {
        self.w1
            .iter()
            .chain(self.b1.iter())
            .chain(self.w2.iter())
            .chain(self.b2.iter())
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt()
    }

    pub(crate) fn validate(&self, model: &HyperNpa2d) -> AutomataResult<()> {
        let expected = [
            ("hyper grad w1", model.weights.w1.len(), self.w1.len()),
            ("hyper grad b1", model.weights.b1.len(), self.b1.len()),
            ("hyper grad w2", model.weights.w2.len(), self.w2.len()),
            ("hyper grad b2", model.weights.b2.len(), self.b2.len()),
        ];
        for (name, expected_len, actual_len) in expected {
            if actual_len != expected_len {
                return Err(AutomataError::InvalidArgument(format!(
                    "{name} len {actual_len} != {expected_len}"
                )));
            }
        }
        ensure_finite("hyper grad w1", &self.w1)?;
        ensure_finite("hyper grad b1", &self.b1)?;
        ensure_finite("hyper grad w2", &self.w2)?;
        ensure_finite("hyper grad b2", &self.b2)?;
        Ok(())
    }
}

fn validate_hyper_config(npa_config: &NpaConfig, config: HyperNpa2dConfig) -> AutomataResult<()> {
    if npa_config.spatial_dims != 2 {
        return Err(AutomataError::InvalidArgument(format!(
            "HyperNpa2d requires a 2D NPA config, got spatial_dims={}",
            npa_config.spatial_dims
        )));
    }
    if config.condition_feature_dims == 0 || config.hidden_dims == 0 || config.adapter_rank == 0 {
        return Err(AutomataError::InvalidArgument(format!(
            "invalid hyper config dims input={} hidden={} rank={}",
            config.condition_feature_dims, config.hidden_dims, config.adapter_rank
        )));
    }
    let expected_condition_feature_dims = condition_feature_dims_for_encoder(
        config.condition_encoder,
        config.condition_token_grid_width,
        config.condition_token_grid_height,
    )?;
    if config.condition_feature_dims != expected_condition_feature_dims {
        return Err(AutomataError::InvalidArgument(format!(
            "hyper condition_feature_dims {} does not match encoder {:?} token grid {}x{} expected {}",
            config.condition_feature_dims,
            config.condition_encoder,
            config.condition_token_grid_width,
            config.condition_token_grid_height,
            expected_condition_feature_dims
        )));
    }
    if !config.adapter_alpha.is_finite() || !config.output_scale.is_finite() {
        return Err(AutomataError::InvalidArgument(
            "hyper adapter alpha and output scale must be finite".to_string(),
        ));
    }
    if config.output_scale <= 0.0 {
        return Err(AutomataError::InvalidArgument(format!(
            "hyper output_scale must be positive, got {}",
            config.output_scale
        )));
    }
    Ok(())
}

pub(crate) fn flow_input(condition_input: &[f32], t: f32, state: &[f32]) -> Vec<f32> {
    let mut input = Vec::with_capacity(condition_input.len() + 1 + state.len());
    input.extend_from_slice(condition_input);
    input.push(t);
    input.extend_from_slice(state);
    input
}

fn seeded_flow_source(
    condition_input: &[f32],
    output_dims: usize,
    source_scale: f32,
    sample_seed: u64,
) -> Vec<f32> {
    let mut hasher = DefaultHasher::new();
    sample_seed.hash(&mut hasher);
    for value in condition_input {
        value.to_bits().hash(&mut hasher);
    }
    if source_scale == 0.0 {
        return vec![0.0; output_dims];
    }
    let mut rng = StdRng::seed_from_u64(hasher.finish());
    let scale = source_scale.abs();
    (0..output_dims)
        .map(|_| rng.random_range(-scale..=scale))
        .collect()
}

fn validate_flow_head(model: &HyperNpa2d, flow: &HyperNpa2dFlow) -> AutomataResult<()> {
    let output_dims = model.adapter_parameter_count();
    let input_dims = model
        .config
        .condition_feature_dims
        .checked_add(1)
        .and_then(|dims| dims.checked_add(output_dims))
        .ok_or_else(|| {
            AutomataError::InvalidModel("hyper flow input dimensions overflow".to_string())
        })?;
    if flow.config.hidden_dims == 0 || flow.config.sample_steps == 0 {
        return Err(AutomataError::InvalidModel(format!(
            "hyper flow requires hidden_dims and sample_steps > 0, got {} and {}",
            flow.config.hidden_dims, flow.config.sample_steps
        )));
    }
    if !flow.config.source_scale.is_finite() || flow.config.source_scale < 0.0 {
        return Err(AutomataError::InvalidModel(format!(
            "hyper flow source_scale must be finite and non-negative, got {}",
            flow.config.source_scale
        )));
    }
    let expected = [
        (
            "hyper flow w1",
            flow.config.hidden_dims * input_dims,
            flow.weights.w1.len(),
        ),
        (
            "hyper flow b1",
            flow.config.hidden_dims,
            flow.weights.b1.len(),
        ),
        (
            "hyper flow w2",
            output_dims * flow.config.hidden_dims,
            flow.weights.w2.len(),
        ),
        ("hyper flow b2", output_dims, flow.weights.b2.len()),
    ];
    for (name, expected_len, actual_len) in expected {
        if actual_len != expected_len {
            return Err(AutomataError::InvalidModel(format!(
                "{name} len {actual_len} != {expected_len}"
            )));
        }
    }
    ensure_finite("hyper flow w1", &flow.weights.w1)?;
    ensure_finite("hyper flow b1", &flow.weights.b1)?;
    ensure_finite("hyper flow w2", &flow.weights.w2)?;
    ensure_finite("hyper flow b2", &flow.weights.b2)?;
    Ok(())
}

fn activate_flow_hidden(activation: HyperNpa2dFlowActivation, value: f32) -> f32 {
    match activation {
        HyperNpa2dFlowActivation::Relu => value.max(0.0),
        HyperNpa2dFlowActivation::LeakyRelu => {
            if value >= 0.0 {
                value
            } else {
                value * 0.01
            }
        }
    }
}

fn ensure_finite(name: &str, values: &[f32]) -> AutomataResult<()> {
    if values.iter().all(|value| value.is_finite()) {
        return Ok(());
    }
    Err(AutomataError::InvalidArgument(format!(
        "{name} contains non-finite values"
    )))
}

fn ensure_finite_f64(name: &str, values: &[f64]) -> AutomataResult<()> {
    if values.iter().all(|value| value.is_finite()) {
        return Ok(());
    }
    Err(AutomataError::InvalidArgument(format!(
        "{name} contains non-finite values"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precise_linear_output_predicts_adapter_vector_exactly() {
        let npa_config = NpaConfig {
            state_dims: 2,
            hidden_dims: 3,
            ..NpaConfig::growing_2d()
        };
        let condition_feature_dims =
            condition_feature_dims_for_encoder(ConditionEncoder2d::DinoVitsClsPatchMean, 0, 0)
                .unwrap();
        let hyper_config = HyperNpa2dConfig {
            condition_encoder: ConditionEncoder2d::DinoVitsClsPatchMean,
            condition_feature_dims,
            condition_token_grid_width: 0,
            condition_token_grid_height: 0,
            hidden_dims: 2,
            adapter_rank: 2,
            adapter_alpha: 2.0,
            adapter_bias_correction: true,
            output_activation: HyperNpa2dOutputActivation::Linear,
            output_scale: 1.0,
        };
        let output_dims = NpaLowRankAdapter::parameter_count_for_config_with_bias_correction(
            &npa_config,
            2,
            true,
        );
        let mut hyper = HyperNpa2d::zeros(npa_config, hyper_config).unwrap();
        hyper.weights.b2.fill(123.0);

        let target = (0..output_dims)
            .map(|idx| (idx as f32 + 1.0) * 0.0001)
            .collect::<Vec<_>>();
        let mut precise = HyperNpa2dPreciseWeights::zeros(
            condition_feature_dims,
            hyper_config.hidden_dims,
            output_dims,
        );
        for (bias, target) in precise.b2.iter_mut().zip(&target) {
            *bias = f64::from(*target);
        }
        hyper.precise_weights = Some(precise);

        let condition = ConditionImage2d::from_luma(1, 1, vec![0.0])
            .unwrap()
            .with_dino_vits_features(vec![0.25; condition_feature_dims])
            .unwrap();
        let predicted = hyper.predict_adapter_vector(&condition).unwrap();
        assert_eq!(predicted, target);

        let adapter = hyper.predict_adapter(&condition).unwrap();
        assert!(adapter.has_bias_correction());
        assert_eq!(adapter.to_parameter_vector(), target);
    }

    #[test]
    fn flow_head_predicts_adapter_vector_from_dino_token_grid() {
        let npa_config = NpaConfig {
            state_dims: 2,
            hidden_dims: 3,
            ..NpaConfig::growing_2d()
        };
        let condition_feature_dims =
            condition_feature_dims_for_encoder(ConditionEncoder2d::DinoVitsTokenGrid, 2, 2)
                .unwrap();
        let hyper_config = HyperNpa2dConfig {
            condition_encoder: ConditionEncoder2d::DinoVitsTokenGrid,
            condition_feature_dims,
            condition_token_grid_width: 2,
            condition_token_grid_height: 2,
            hidden_dims: 2,
            adapter_rank: 2,
            adapter_alpha: 2.0,
            adapter_bias_correction: false,
            output_activation: HyperNpa2dOutputActivation::Linear,
            output_scale: 1.0,
        };
        let mut hyper = HyperNpa2d::zeros(npa_config, hyper_config).unwrap();
        let output_dims = hyper.adapter_parameter_count();
        hyper
            .set_flow(HyperNpa2dFlow {
                config: HyperNpa2dFlowConfig {
                    hidden_dims: 2,
                    sample_steps: 2,
                    source_scale: 0.01,
                    sample_seed: 7,
                    hidden_activation: HyperNpa2dFlowActivation::Relu,
                },
                weights: HyperNpa2dFlowWeights {
                    w1: vec![0.0; 2 * (condition_feature_dims + 1 + output_dims)],
                    b1: vec![0.0; 2],
                    w2: vec![0.0; output_dims * 2],
                    b2: vec![0.0; output_dims],
                },
            })
            .unwrap();
        let condition = ConditionImage2d::from_luma(1, 1, vec![0.0])
            .unwrap()
            .with_dino_vits_features(vec![0.25; condition_feature_dims])
            .unwrap();
        let predicted = hyper.predict_adapter_vector(&condition).unwrap();
        assert_eq!(predicted.len(), output_dims);
        assert!(predicted.iter().all(|value| value.is_finite()));
    }
}
