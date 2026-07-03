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
    pub output_scale: f32,
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
pub struct HyperNpa2d {
    pub npa_config: NpaConfig,
    pub config: HyperNpa2dConfig,
    pub weights: HyperNpa2dWeights,
    #[serde(default)]
    pub anchor_input: Option<Vec<f32>>,
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
        let output_dims =
            NpaLowRankAdapter::parameter_count_for_config(&npa_config, config.adapter_rank);
        Ok(Self {
            npa_config,
            config,
            weights: HyperNpa2dWeights::zeros(
                config.condition_feature_dims,
                config.hidden_dims,
                output_dims,
            ),
            anchor_input: None,
        })
    }

    pub fn seeded(
        npa_config: NpaConfig,
        config: HyperNpa2dConfig,
        seed: u64,
    ) -> AutomataResult<Self> {
        validate_hyper_config(&npa_config, config)?;
        let output_dims =
            NpaLowRankAdapter::parameter_count_for_config(&npa_config, config.adapter_rank);
        Ok(Self {
            npa_config,
            config,
            weights: HyperNpa2dWeights::seeded(
                config.condition_feature_dims,
                config.hidden_dims,
                output_dims,
                seed,
            ),
            anchor_input: None,
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
        Ok(())
    }

    pub fn adapter_parameter_count(&self) -> usize {
        NpaLowRankAdapter::parameter_count_for_config(&self.npa_config, self.config.adapter_rank)
    }

    pub fn predict_adapter(
        &self,
        condition: &ConditionImage2d,
    ) -> AutomataResult<NpaLowRankAdapter> {
        let values = self.predict_adapter_vector(condition)?;
        NpaLowRankAdapter::from_parameter_vector(
            &self.npa_config,
            self.config.adapter_rank,
            self.config.adapter_alpha,
            values,
        )
    }

    pub fn predict_adapter_vector(&self, condition: &ConditionImage2d) -> AutomataResult<Vec<f32>> {
        Ok(self.forward_cache(condition)?.output)
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

    fn forward_layer(&self, input: Vec<f32>) -> HyperLayerCache {
        let mut pre_hidden = vec![0.0; self.config.hidden_dims];
        let mut hidden = vec![0.0; self.config.hidden_dims];
        for h in 0..self.config.hidden_dims {
            let mut sum = self.weights.b1[h];
            let base = h * self.config.condition_feature_dims;
            for (i, value) in input.iter().enumerate() {
                sum += self.weights.w1[base + i] * *value;
            }
            pre_hidden[h] = sum;
            hidden[h] = sum.max(0.0);
        }

        let output_dims = self.adapter_parameter_count();
        let mut pre_output = vec![0.0; output_dims];
        let mut output = vec![0.0; output_dims];
        for o in 0..output_dims {
            let mut sum = self.weights.b2[o];
            let base = o * self.config.hidden_dims;
            for (h, value) in hidden.iter().enumerate() {
                sum += self.weights.w2[base + h] * *value;
            }
            pre_output[o] = sum;
            output[o] = sum.tanh() * self.config.output_scale;
        }

        HyperLayerCache {
            input,
            pre_hidden,
            hidden,
            pre_output,
            output,
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
            let tanh_pre = cache.pre_output[o].tanh();
            let d_pre_output =
                output_grad * self.config.output_scale * (1.0 - tanh_pre * tanh_pre) * scale;
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

fn ensure_finite(name: &str, values: &[f32]) -> AutomataResult<()> {
    if values.iter().all(|value| value.is_finite()) {
        return Ok(());
    }
    Err(AutomataError::InvalidArgument(format!(
        "{name} contains non-finite values"
    )))
}
