use std::{fs, path::Path};

use serde::{Deserialize, Serialize};

use crate::{AutomataError, AutomataResult, NpaConfig, NpaLowRankAdapter, NpaModel};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct E2eHyperNpa2d {
    #[serde(default)]
    pub version: usize,
    pub architecture: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    pub hidden_dims: usize,
    pub token_attention_heads: usize,
    pub output_dims: usize,
    pub sample_steps: usize,
    pub output_scale: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_rank: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_alpha: Option<f32>,
    pub weights: E2eHyperNpa2dWeights,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct E2eHyperNpa2dWeights {
    pub token_w: Vec<f32>,
    pub token_b: Vec<f32>,
    pub token_gate_w: Vec<f32>,
    pub token_gate_b: Vec<f32>,
    pub state_w: Vec<f32>,
    pub time_w: Vec<f32>,
    pub output_w: Vec<f32>,
    pub output_b: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct E2eHyperNpa2dAdapterSpec {
    pub rank: usize,
    pub alpha: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct E2eConditionedNpa2d {
    pub adapter: NpaLowRankAdapter,
    pub model: NpaModel,
}

pub fn load_e2e_hyper_npa_2d(path: impl AsRef<Path>) -> AutomataResult<E2eHyperNpa2d> {
    let contents = fs::read_to_string(path)?;
    let hyper = serde_json::from_str::<E2eHyperNpa2d>(&contents)?;
    hyper.validate()?;
    Ok(hyper)
}

pub fn generate_e2e_conditioned_npa_2d(
    base_model: &NpaModel,
    hyper: &E2eHyperNpa2d,
    condition_tokens: &[f32],
) -> AutomataResult<E2eConditionedNpa2d> {
    base_model.validate()?;
    hyper.validate()?;
    let adapter = hyper.predict_adapter(&base_model.config, condition_tokens)?;
    let model = adapter.apply_to_model(base_model)?;
    Ok(E2eConditionedNpa2d { adapter, model })
}

impl E2eHyperNpa2d {
    pub fn validate(&self) -> AutomataResult<()> {
        if self.version > 1 {
            return Err(AutomataError::InvalidFormat(format!(
                "unsupported E2E HyperNPA version {}",
                self.version
            )));
        }
        if self.architecture != "token_attention_pool_rectified_flow_generated_lora" {
            return Err(AutomataError::InvalidFormat(format!(
                "unsupported E2E HyperNPA architecture {:?}",
                self.architecture
            )));
        }
        if self.hidden_dims == 0 || self.output_dims == 0 {
            return Err(AutomataError::InvalidModel(format!(
                "E2E HyperNPA hidden_dims={} output_dims={} must be positive",
                self.hidden_dims, self.output_dims
            )));
        }
        if self.token_attention_heads == 0 || self.sample_steps == 0 {
            return Err(AutomataError::InvalidModel(format!(
                "E2E HyperNPA token_attention_heads={} sample_steps={} must be positive",
                self.token_attention_heads, self.sample_steps
            )));
        }
        if !self.output_scale.is_finite() {
            return Err(AutomataError::InvalidModel(
                "E2E HyperNPA output_scale must be finite".to_string(),
            ));
        }
        if let Some(rank) = self.adapter_rank
            && rank == 0
        {
            return Err(AutomataError::InvalidModel(
                "E2E HyperNPA adapter_rank must be positive".to_string(),
            ));
        }
        if let Some(alpha) = self.adapter_alpha
            && (!alpha.is_finite() || alpha <= 0.0)
        {
            return Err(AutomataError::InvalidModel(format!(
                "E2E HyperNPA adapter_alpha must be positive and finite, got {alpha}"
            )));
        }
        let embed_dims = self.embed_dims()?;
        let expected = [
            ("token_b", self.hidden_dims, self.weights.token_b.len()),
            (
                "token_gate_w",
                self.token_attention_heads * self.hidden_dims,
                self.weights.token_gate_w.len(),
            ),
            (
                "token_gate_b",
                self.token_attention_heads,
                self.weights.token_gate_b.len(),
            ),
            (
                "state_w",
                self.hidden_dims * self.output_dims,
                self.weights.state_w.len(),
            ),
            ("time_w", self.hidden_dims, self.weights.time_w.len()),
            (
                "output_w",
                self.output_dims * self.hidden_dims,
                self.weights.output_w.len(),
            ),
            ("output_b", self.output_dims, self.weights.output_b.len()),
        ];
        for (name, expected_len, actual_len) in expected {
            if actual_len != expected_len {
                return Err(AutomataError::InvalidModel(format!(
                    "E2E HyperNPA {name} len {actual_len} != {expected_len}"
                )));
            }
        }
        let _ = embed_dims;
        self.weights.ensure_finite()
    }

    pub fn embed_dims(&self) -> AutomataResult<usize> {
        if self.hidden_dims == 0 || !self.weights.token_w.len().is_multiple_of(self.hidden_dims) {
            return Err(AutomataError::InvalidModel(format!(
                "E2E HyperNPA token_w len {} is not divisible by hidden_dims {}",
                self.weights.token_w.len(),
                self.hidden_dims
            )));
        }
        let embed_dims = self.weights.token_w.len() / self.hidden_dims;
        if embed_dims == 0 {
            return Err(AutomataError::InvalidModel(
                "E2E HyperNPA embed_dims must be positive".to_string(),
            ));
        }
        Ok(embed_dims)
    }

    pub fn adapter_spec(&self, config: &NpaConfig) -> AutomataResult<E2eHyperNpa2dAdapterSpec> {
        let inferred_rank = self.infer_adapter_rank(config)?;
        if let Some(rank) = self.adapter_rank
            && rank != inferred_rank
        {
            return Err(AutomataError::InvalidModel(format!(
                "E2E HyperNPA adapter_rank {rank} does not match output_dims {}; inferred rank {inferred_rank}",
                self.output_dims
            )));
        }
        Ok(E2eHyperNpa2dAdapterSpec {
            rank: inferred_rank,
            alpha: self.adapter_alpha.unwrap_or(inferred_rank as f32),
        })
    }

    pub fn infer_adapter_rank(&self, config: &NpaConfig) -> AutomataResult<usize> {
        let fixed = config.hidden_dims + config.update_dims();
        let per_rank = config.perception_dims()
            + config.hidden_dims
            + config.hidden_dims
            + config.update_dims();
        if self.output_dims < fixed || per_rank == 0 {
            return Err(AutomataError::InvalidModel(format!(
                "E2E HyperNPA output_dims {} is too small for NPA config",
                self.output_dims
            )));
        }
        let low_rank_values = self.output_dims - fixed;
        if !low_rank_values.is_multiple_of(per_rank) {
            return Err(AutomataError::InvalidModel(format!(
                "E2E HyperNPA output_dims {} cannot be represented as a non-bias-corrected LoRA vector for this NPA config",
                self.output_dims
            )));
        }
        let rank = low_rank_values / per_rank;
        if rank == 0 {
            return Err(AutomataError::InvalidModel(
                "E2E HyperNPA inferred adapter rank is zero".to_string(),
            ));
        }
        let expected = NpaLowRankAdapter::parameter_count_for_config(config, rank);
        if expected != self.output_dims {
            return Err(AutomataError::InvalidModel(format!(
                "E2E HyperNPA inferred rank {rank} has parameter count {expected}, not {}",
                self.output_dims
            )));
        }
        Ok(rank)
    }

    pub fn predict_adapter(
        &self,
        config: &NpaConfig,
        condition_tokens: &[f32],
    ) -> AutomataResult<NpaLowRankAdapter> {
        self.validate()?;
        let embed_dims = self.embed_dims()?;
        if condition_tokens.is_empty() || !condition_tokens.len().is_multiple_of(embed_dims) {
            return Err(AutomataError::InvalidArgument(format!(
                "E2E HyperNPA condition token len {} must be a positive multiple of embed_dims {embed_dims}",
                condition_tokens.len()
            )));
        }
        if !condition_tokens.iter().all(|value| value.is_finite()) {
            return Err(AutomataError::InvalidArgument(
                "E2E HyperNPA condition tokens contain non-finite values".to_string(),
            ));
        }

        let token_count = condition_tokens.len() / embed_dims;
        let hidden_dims = self.hidden_dims;
        let heads = self.token_attention_heads;

        let mut token_hidden = vec![0.0_f32; token_count * hidden_dims];
        for token in 0..token_count {
            let condition = &condition_tokens[token * embed_dims..(token + 1) * embed_dims];
            for hidden in 0..hidden_dims {
                let mut value = self.weights.token_b[hidden];
                let weight_base = hidden * embed_dims;
                for (dim, condition_value) in condition.iter().enumerate() {
                    value += *condition_value * self.weights.token_w[weight_base + dim];
                }
                token_hidden[token * hidden_dims + hidden] = value.max(0.0);
            }
        }

        let inv_tokens = 1.0 / token_count.max(1) as f32;
        let mut mean_pooled = vec![0.0_f32; hidden_dims];
        for token in 0..token_count {
            let base = token * hidden_dims;
            for hidden in 0..hidden_dims {
                mean_pooled[hidden] += token_hidden[base + hidden] * inv_tokens;
            }
        }

        let mut attention_weights = vec![0.0_f32; token_count * heads];
        let mut attention_denominator = vec![0.0_f32; heads];
        for token in 0..token_count {
            let token_base = token * hidden_dims;
            for head in 0..heads {
                let mut logit = self.weights.token_gate_b[head];
                let weight_base = head * hidden_dims;
                for hidden in 0..hidden_dims {
                    logit += token_hidden[token_base + hidden]
                        * self.weights.token_gate_w[weight_base + hidden];
                }
                let value = logit.tanh().exp();
                attention_weights[token * heads + head] = value;
                attention_denominator[head] += value;
            }
        }
        for denominator in &mut attention_denominator {
            *denominator = (*denominator).max(f32::MIN_POSITIVE);
        }

        let mut attended = vec![0.0_f32; hidden_dims];
        for token in 0..token_count {
            let token_base = token * hidden_dims;
            for head in 0..heads {
                let weight = attention_weights[token * heads + head] / attention_denominator[head];
                for hidden in 0..hidden_dims {
                    attended[hidden] += token_hidden[token_base + hidden] * weight;
                }
            }
        }
        let inv_heads = 1.0 / heads as f32;
        let mut pooled = vec![0.0_f32; hidden_dims];
        for hidden in 0..hidden_dims {
            attended[hidden] *= inv_heads;
            pooled[hidden] = 0.5 * (mean_pooled[hidden] + attended[hidden]);
        }

        let mut vector = vec![0.0_f32; self.output_dims];
        let inv_steps = 1.0 / self.sample_steps as f32;
        for step in 0..self.sample_steps {
            let t = if self.sample_steps <= 1 {
                0.0
            } else {
                step as f32 / (self.sample_steps - 1) as f32
            };
            let mut hidden = vec![0.0_f32; hidden_dims];
            for (hidden_idx, hidden_value) in hidden.iter_mut().enumerate() {
                let mut value = pooled[hidden_idx]
                    + self.weights.token_b[hidden_idx]
                    + self.weights.time_w[hidden_idx] * t;
                let state_base = hidden_idx * self.output_dims;
                for (out, vector_value) in vector.iter().enumerate() {
                    value += *vector_value * self.weights.state_w[state_base + out];
                }
                *hidden_value = value.max(0.0);
            }
            for (out, vector_value) in vector.iter_mut().enumerate() {
                let mut velocity = self.weights.output_b[out];
                let weight_base = out * hidden_dims;
                for (hidden_idx, hidden_value) in hidden.iter().enumerate() {
                    velocity += *hidden_value * self.weights.output_w[weight_base + hidden_idx];
                }
                *vector_value += velocity * inv_steps;
            }
        }
        for value in &mut vector {
            *value = value.tanh() * self.output_scale;
        }

        let spec = self.adapter_spec(config)?;
        NpaLowRankAdapter::from_parameter_vector(config, spec.rank, spec.alpha, vector)
    }
}

impl E2eHyperNpa2dWeights {
    fn ensure_finite(&self) -> AutomataResult<()> {
        let checks = [
            ("token_w", self.token_w.as_slice()),
            ("token_b", self.token_b.as_slice()),
            ("token_gate_w", self.token_gate_w.as_slice()),
            ("token_gate_b", self.token_gate_b.as_slice()),
            ("state_w", self.state_w.as_slice()),
            ("time_w", self.time_w.as_slice()),
            ("output_w", self.output_w.as_slice()),
            ("output_b", self.output_b.as_slice()),
        ];
        for (name, values) in checks {
            if !values.iter().all(|value| value.is_finite()) {
                return Err(AutomataError::InvalidModel(format!(
                    "E2E HyperNPA {name} contains non-finite values"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::{AutomataPreset, NpaConfig};

    fn tiny_hyper(config: &NpaConfig, rank: usize) -> E2eHyperNpa2d {
        let hidden_dims = 2;
        let embed_dims = 3;
        let output_dims = NpaLowRankAdapter::parameter_count_for_config(config, rank);
        E2eHyperNpa2d {
            version: 1,
            architecture: "token_attention_pool_rectified_flow_generated_lora".to_string(),
            backend: None,
            hidden_dims,
            token_attention_heads: 1,
            output_dims,
            sample_steps: 1,
            output_scale: 1.0,
            adapter_rank: None,
            adapter_alpha: None,
            weights: E2eHyperNpa2dWeights {
                token_w: vec![0.01; hidden_dims * embed_dims],
                token_b: vec![0.0; hidden_dims],
                token_gate_w: vec![0.01; hidden_dims],
                token_gate_b: vec![0.0],
                state_w: vec![0.0; hidden_dims * output_dims],
                time_w: vec![0.0; hidden_dims],
                output_w: vec![0.001; output_dims * hidden_dims],
                output_b: vec![0.0; output_dims],
            },
        }
    }

    #[test]
    fn e2e_hyper_infers_rank_from_output_dims() {
        let (config, _) = NpaConfig::for_preset(AutomataPreset::Growing2d);
        let hyper = tiny_hyper(&config, 4);
        let spec = hyper.adapter_spec(&config).unwrap();
        assert_eq!(spec.rank, 4);
        assert_eq!(spec.alpha, 4.0);
    }

    #[test]
    fn e2e_hyper_predicts_adapter_from_token_condition() {
        let (config, _) = NpaConfig::for_preset(AutomataPreset::Growing2d);
        let hyper = tiny_hyper(&config, 2);
        let adapter = hyper
            .predict_adapter(&config, &[0.2, -0.1, 0.4, 0.5, 0.0, -0.2])
            .unwrap();
        adapter.validate(&config).unwrap();
        assert_eq!(adapter.rank, 2);
        assert_eq!(adapter.parameter_count(), hyper.output_dims);
    }

    #[test]
    fn local_default_e2e_artifacts_load_when_present() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let artifact_dir = workspace.join(
            "artifacts/hyper2d_e2e_rollout_train_omnisvg_10k_steps3000_b16_p128s4_rank16_cosine_cuda",
        );
        let base_path = artifact_dir.join("shared_base.bpk");
        let hyper_path = artifact_dir.join("hyper_2d.json");
        if !base_path.exists() || !hyper_path.exists() {
            return;
        }

        let base = crate::import::load_manifest(base_path)
            .unwrap()
            .into_model();
        let hyper = load_e2e_hyper_npa_2d(hyper_path).unwrap();
        let spec = hyper.adapter_spec(&base.config).unwrap();
        assert_eq!(hyper.embed_dims().unwrap(), crate::DINO_VITS_EMBED_DIMS);
        assert_eq!(spec.rank, 16);
        assert_eq!(spec.alpha, 16.0);
    }
}
