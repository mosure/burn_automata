use std::{fs, path::Path};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{AutomataError, AutomataResult, NpaConfig, NpaLowRankAdapter, NpaModel};

const E2E_HYPER_BPK_MAGIC: [u8; 8] = *b"BAUTHYP1";
const E2E_HYPER_BPK_HEADER_LEN: usize = 8 + 4 + 8 + 8 + 32;
const E2E_HYPER_BPK_CONTAINER_VERSION: u32 = 1;
const E2E_HYPER_MODEL_KIND: &str = "e2e-hypernpa-2d";
pub const E2E_HYPER_ARCH_POOLED_FLOW: &str = "token_attention_pool_rectified_flow_generated_lora";
pub const E2E_HYPER_ARCH_SPATIAL_TOKEN_FLOW: &str = "spatial_token_chunked_rectified_flow_lora_v1";
pub const DEFAULT_E2E_HYPER_ADAPTER_CHUNK_SIZE: usize = 64;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Target2dLossBackend {
    #[default]
    Dense,
    TiledAdjoint,
    Auto,
}

impl Target2dLossBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::TiledAdjoint => "tiled-adjoint",
            Self::Auto => "auto",
        }
    }

    pub fn parse(value: &str) -> AutomataResult<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "dense" | "burn-dense" | "autodiff-dense" => Ok(Self::Dense),
            "tiled-adjoint" | "tiled_adjoint" | "cpu-adjoint" | "cpu_adjoint" => {
                Ok(Self::TiledAdjoint)
            }
            "auto" => Ok(Self::Auto),
            other => Err(AutomataError::InvalidArgument(format!(
                "unknown target2d loss backend `{other}`; expected dense, tiled-adjoint, or auto"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PerceptionRolloutBackend {
    #[default]
    Dense,
    TiledAdjoint,
    Auto,
}

impl PerceptionRolloutBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::TiledAdjoint => "tiled-adjoint",
            Self::Auto => "auto",
        }
    }

    pub fn parse(value: &str) -> AutomataResult<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "dense" | "burn-dense" | "autodiff-dense" => Ok(Self::Dense),
            "tiled-adjoint" | "tiled_adjoint" | "reference-adjoint" | "reference_adjoint" => {
                Ok(Self::TiledAdjoint)
            }
            "auto" => Ok(Self::Auto),
            other => Err(AutomataError::InvalidArgument(format!(
                "unknown perception backend `{other}`; expected dense, tiled-adjoint, or auto"
            ))),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct E2eHyperNpa2d {
    #[serde(default)]
    pub version: usize,
    pub architecture: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition_encoder: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition_token_count: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition_embed_dims: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition_token_grid_width: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition_token_grid_height: Option<usize>,
    pub hidden_dims: usize,
    pub token_attention_heads: usize,
    pub output_dims: usize,
    pub sample_steps: usize,
    pub output_scale: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_rank: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_alpha: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_chunk_size: Option<usize>,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
struct E2eHyperNpa2dBinaryMetadata {
    format_version: u32,
    model_kind: String,
    version: usize,
    architecture: String,
    backend: Option<String>,
    condition_encoder: Option<String>,
    condition_token_count: Option<usize>,
    condition_embed_dims: Option<usize>,
    condition_token_grid_width: Option<usize>,
    condition_token_grid_height: Option<usize>,
    hidden_dims: usize,
    token_attention_heads: usize,
    output_dims: usize,
    sample_steps: usize,
    output_scale: f32,
    adapter_rank: Option<usize>,
    adapter_alpha: Option<f32>,
    #[serde(default)]
    adapter_chunk_size: Option<usize>,
    weight_lens: E2eHyperNpa2dWeightLens,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct E2eHyperNpa2dWeightLens {
    token_w: usize,
    token_b: usize,
    token_gate_w: usize,
    token_gate_b: usize,
    state_w: usize,
    time_w: usize,
    output_w: usize,
    output_b: usize,
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
    let path = path.as_ref();
    let bytes = fs::read(path)?;
    let hyper = if bytes.starts_with(&E2E_HYPER_BPK_MAGIC) {
        decode_e2e_hyper_npa_2d(&bytes)?
    } else {
        eprintln!(
            "warning: loading legacy JSON E2E HyperNPA artifact {}; use .bpk for trained hypernet artifacts",
            path.display()
        );
        serde_json::from_slice::<E2eHyperNpa2d>(&bytes)?
    };
    hyper.validate()?;
    Ok(hyper)
}

pub fn save_e2e_hyper_npa_2d(
    path: impl AsRef<Path>,
    hyper: &E2eHyperNpa2d,
) -> AutomataResult<String> {
    let path = path.as_ref();
    if is_json_path(path) {
        return Err(AutomataError::InvalidArgument(format!(
            "refusing to write trained E2E HyperNPA weights as JSON at {}; use .bpk",
            path.display()
        )));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let encoded = encode_e2e_hyper_npa_2d(hyper)?;
    let sha256 = e2e_hyper_bpk_payload_sha256(&encoded)?;
    fs::write(path, encoded)?;
    Ok(sha256)
}

pub fn encode_e2e_hyper_npa_2d(hyper: &E2eHyperNpa2d) -> AutomataResult<Vec<u8>> {
    hyper.validate()?;
    let metadata = E2eHyperNpa2dBinaryMetadata::from_hyper(hyper);
    let metadata_bytes = serde_json::to_vec(&metadata)?;
    let weight_bytes = hyper.weights.to_le_bytes();
    let metadata_len = u64::try_from(metadata_bytes.len()).map_err(|_| {
        AutomataError::InvalidFormat("E2E HyperNPA metadata is too large".to_string())
    })?;
    let weights_len = u64::try_from(weight_bytes.len()).map_err(|_| {
        AutomataError::InvalidFormat("E2E HyperNPA weight payload is too large".to_string())
    })?;
    let mut hasher = Sha256::new();
    hasher.update(&metadata_bytes);
    hasher.update(&weight_bytes);
    let digest = hasher.finalize();
    let mut out =
        Vec::with_capacity(E2E_HYPER_BPK_HEADER_LEN + metadata_bytes.len() + weight_bytes.len());
    out.extend_from_slice(&E2E_HYPER_BPK_MAGIC);
    out.extend_from_slice(&E2E_HYPER_BPK_CONTAINER_VERSION.to_le_bytes());
    out.extend_from_slice(&metadata_len.to_le_bytes());
    out.extend_from_slice(&weights_len.to_le_bytes());
    out.extend_from_slice(&digest);
    out.extend_from_slice(&metadata_bytes);
    out.extend_from_slice(&weight_bytes);
    Ok(out)
}

pub fn decode_e2e_hyper_npa_2d(bytes: &[u8]) -> AutomataResult<E2eHyperNpa2d> {
    if bytes.len() < E2E_HYPER_BPK_HEADER_LEN {
        return Err(AutomataError::InvalidFormat(format!(
            "E2E HyperNPA bpk is shorter than header: {} < {E2E_HYPER_BPK_HEADER_LEN}",
            bytes.len()
        )));
    }
    if !bytes.starts_with(&E2E_HYPER_BPK_MAGIC) {
        return Err(AutomataError::InvalidFormat(
            "missing E2E HyperNPA bpk magic".to_string(),
        ));
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().expect("header version slice"));
    if version != E2E_HYPER_BPK_CONTAINER_VERSION {
        return Err(AutomataError::InvalidFormat(format!(
            "unsupported E2E HyperNPA bpk container version {version}"
        )));
    }
    let metadata_len =
        u64::from_le_bytes(bytes[12..20].try_into().expect("metadata len slice")) as usize;
    let weights_len =
        u64::from_le_bytes(bytes[20..28].try_into().expect("weights len slice")) as usize;
    let expected_len = E2E_HYPER_BPK_HEADER_LEN
        .saturating_add(metadata_len)
        .saturating_add(weights_len);
    if bytes.len() != expected_len {
        return Err(AutomataError::InvalidFormat(format!(
            "E2E HyperNPA bpk length mismatch: file {} != expected {expected_len}",
            bytes.len()
        )));
    }
    let expected_digest = &bytes[28..60];
    let metadata_start = E2E_HYPER_BPK_HEADER_LEN;
    let weights_start = metadata_start + metadata_len;
    let metadata_bytes = &bytes[metadata_start..weights_start];
    let weight_bytes = &bytes[weights_start..];
    let mut hasher = Sha256::new();
    hasher.update(metadata_bytes);
    hasher.update(weight_bytes);
    let actual_digest = hasher.finalize();
    if expected_digest != actual_digest.as_slice() {
        return Err(AutomataError::InvalidFormat(
            "E2E HyperNPA bpk sha256 checksum mismatch".to_string(),
        ));
    }
    let metadata: E2eHyperNpa2dBinaryMetadata = serde_json::from_slice(metadata_bytes)?;
    metadata.into_hyper(weight_bytes)
}

pub fn e2e_hyper_bpk_payload_sha256(bytes: &[u8]) -> AutomataResult<String> {
    if bytes.len() < E2E_HYPER_BPK_HEADER_LEN || !bytes.starts_with(&E2E_HYPER_BPK_MAGIC) {
        return Err(AutomataError::InvalidFormat(
            "not an E2E HyperNPA bpk container".to_string(),
        ));
    }
    Ok(hex_digest(&bytes[28..60]))
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
        if !matches!(
            self.architecture.as_str(),
            E2E_HYPER_ARCH_POOLED_FLOW | E2E_HYPER_ARCH_SPATIAL_TOKEN_FLOW
        ) {
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
        if let Some(chunk_size) = self.adapter_chunk_size
            && chunk_size == 0
        {
            return Err(AutomataError::InvalidModel(
                "E2E HyperNPA adapter_chunk_size must be positive".to_string(),
            ));
        }
        if let Some(token_count) = self.condition_token_count
            && token_count == 0
        {
            return Err(AutomataError::InvalidModel(
                "E2E HyperNPA condition_token_count must be positive".to_string(),
            ));
        }
        if let Some(embed_dims) = self.condition_embed_dims
            && embed_dims != self.embed_dims()?
        {
            return Err(AutomataError::InvalidModel(format!(
                "E2E HyperNPA condition_embed_dims {embed_dims} does not match token_w embed dims {}",
                self.embed_dims()?
            )));
        }
        let embed_dims = self.embed_dims()?;
        let expected = if self.is_spatial_token_flow() {
            let chunk_size = self.adapter_chunk_size_value();
            let chunks = self.spatial_chunk_count();
            [
                ("token_b", self.hidden_dims, self.weights.token_b.len()),
                (
                    "token_gate_w",
                    chunks * self.hidden_dims,
                    self.weights.token_gate_w.len(),
                ),
                (
                    "token_gate_b",
                    chunks * self.hidden_dims,
                    self.weights.token_gate_b.len(),
                ),
                (
                    "state_w",
                    self.hidden_dims * chunk_size,
                    self.weights.state_w.len(),
                ),
                ("time_w", self.hidden_dims, self.weights.time_w.len()),
                (
                    "output_w",
                    chunk_size * self.hidden_dims,
                    self.weights.output_w.len(),
                ),
                ("output_b", chunks * chunk_size, self.weights.output_b.len()),
            ]
        } else {
            [
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
            ]
        };
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

    pub fn is_spatial_token_flow(&self) -> bool {
        self.architecture == E2E_HYPER_ARCH_SPATIAL_TOKEN_FLOW
    }

    pub fn adapter_chunk_size_value(&self) -> usize {
        self.adapter_chunk_size
            .unwrap_or(DEFAULT_E2E_HYPER_ADAPTER_CHUNK_SIZE)
            .max(1)
    }

    pub fn spatial_chunk_count(&self) -> usize {
        self.output_dims.div_ceil(self.adapter_chunk_size_value())
    }

    pub fn predict_adapter(
        &self,
        config: &NpaConfig,
        condition_tokens: &[f32],
    ) -> AutomataResult<NpaLowRankAdapter> {
        self.validate()?;
        if self.is_spatial_token_flow() {
            return self.predict_spatial_token_adapter(config, condition_tokens);
        }
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

    fn predict_spatial_token_adapter(
        &self,
        config: &NpaConfig,
        condition_tokens: &[f32],
    ) -> AutomataResult<NpaLowRankAdapter> {
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
        let chunk_size = self.adapter_chunk_size_value();
        let chunks = self.spatial_chunk_count();
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

        let mut chunk_state = vec![0.0_f32; chunks * chunk_size];
        let inv_steps = 1.0 / self.sample_steps as f32;
        let attention_scale = 1.0 / (hidden_dims as f32).sqrt().max(1.0);
        for step in 0..self.sample_steps {
            let t = if self.sample_steps <= 1 {
                0.0
            } else {
                step as f32 / (self.sample_steps - 1) as f32
            };
            let mut next_state = chunk_state.clone();
            for chunk in 0..chunks {
                let mut query_hidden = vec![0.0_f32; hidden_dims];
                for (hidden, hidden_value) in query_hidden.iter_mut().enumerate() {
                    let mut value = self.weights.token_b[hidden]
                        + self.weights.token_gate_w[chunk * hidden_dims + hidden]
                        + self.weights.token_gate_b[chunk * hidden_dims + hidden]
                        + self.weights.time_w[hidden] * t;
                    for coord in 0..chunk_size {
                        value += chunk_state[chunk * chunk_size + coord]
                            * self.weights.state_w[hidden * chunk_size + coord];
                    }
                    *hidden_value = value.max(0.0);
                }

                let mut attention_denominator = f32::MIN_POSITIVE;
                let mut attention_weights = vec![0.0_f32; token_count];
                for (token, attention_weight) in attention_weights.iter_mut().enumerate() {
                    let token_base = token * hidden_dims;
                    let mut logit = 0.0_f32;
                    for hidden in 0..hidden_dims {
                        logit += query_hidden[hidden] * token_hidden[token_base + hidden];
                    }
                    let weight = (logit * attention_scale).tanh().exp();
                    *attention_weight = weight;
                    attention_denominator += weight;
                }

                let mut attended = vec![0.0_f32; hidden_dims];
                for (token, attention_weight) in attention_weights.iter().enumerate() {
                    let weight = *attention_weight / attention_denominator;
                    let token_base = token * hidden_dims;
                    for hidden in 0..hidden_dims {
                        attended[hidden] += token_hidden[token_base + hidden] * weight;
                    }
                }
                for hidden in 0..hidden_dims {
                    attended[hidden] = (attended[hidden] + query_hidden[hidden]).max(0.0);
                }

                for coord in 0..chunk_size {
                    let padded_idx = chunk * chunk_size + coord;
                    let mut velocity = self.weights.output_b[padded_idx];
                    let weight_base = coord * hidden_dims;
                    for (hidden, hidden_value) in attended.iter().enumerate() {
                        velocity += *hidden_value * self.weights.output_w[weight_base + hidden];
                    }
                    next_state[padded_idx] += velocity * inv_steps * self.output_scale;
                }
            }
            chunk_state = next_state;
        }
        chunk_state.truncate(self.output_dims);

        let spec = self.adapter_spec(config)?;
        NpaLowRankAdapter::from_parameter_vector(config, spec.rank, spec.alpha, chunk_state)
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

    fn lens(&self) -> E2eHyperNpa2dWeightLens {
        E2eHyperNpa2dWeightLens {
            token_w: self.token_w.len(),
            token_b: self.token_b.len(),
            token_gate_w: self.token_gate_w.len(),
            token_gate_b: self.token_gate_b.len(),
            state_w: self.state_w.len(),
            time_w: self.time_w.len(),
            output_w: self.output_w.len(),
            output_b: self.output_b.len(),
        }
    }

    fn to_le_bytes(&self) -> Vec<u8> {
        let value_count = self.token_w.len()
            + self.token_b.len()
            + self.token_gate_w.len()
            + self.token_gate_b.len()
            + self.state_w.len()
            + self.time_w.len()
            + self.output_w.len()
            + self.output_b.len();
        let mut bytes = Vec::with_capacity(value_count * 4);
        for values in [
            self.token_w.as_slice(),
            self.token_b.as_slice(),
            self.token_gate_w.as_slice(),
            self.token_gate_b.as_slice(),
            self.state_w.as_slice(),
            self.time_w.as_slice(),
            self.output_w.as_slice(),
            self.output_b.as_slice(),
        ] {
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }
}

impl E2eHyperNpa2dBinaryMetadata {
    fn from_hyper(hyper: &E2eHyperNpa2d) -> Self {
        Self {
            format_version: 1,
            model_kind: E2E_HYPER_MODEL_KIND.to_string(),
            version: hyper.version,
            architecture: hyper.architecture.clone(),
            backend: hyper.backend.clone(),
            condition_encoder: hyper.condition_encoder.clone(),
            condition_token_count: hyper.condition_token_count,
            condition_embed_dims: hyper.condition_embed_dims,
            condition_token_grid_width: hyper.condition_token_grid_width,
            condition_token_grid_height: hyper.condition_token_grid_height,
            hidden_dims: hyper.hidden_dims,
            token_attention_heads: hyper.token_attention_heads,
            output_dims: hyper.output_dims,
            sample_steps: hyper.sample_steps,
            output_scale: hyper.output_scale,
            adapter_rank: hyper.adapter_rank,
            adapter_alpha: hyper.adapter_alpha,
            adapter_chunk_size: hyper.adapter_chunk_size,
            weight_lens: hyper.weights.lens(),
        }
    }

    fn into_hyper(self, weight_bytes: &[u8]) -> AutomataResult<E2eHyperNpa2d> {
        if self.format_version != 1 {
            return Err(AutomataError::InvalidFormat(format!(
                "unsupported E2E HyperNPA metadata version {}",
                self.format_version
            )));
        }
        if self.model_kind != E2E_HYPER_MODEL_KIND {
            return Err(AutomataError::InvalidFormat(format!(
                "unexpected E2E HyperNPA model_kind {:?}",
                self.model_kind
            )));
        }
        let mut cursor = F32Cursor::new(weight_bytes);
        let weights = E2eHyperNpa2dWeights {
            token_w: cursor.take("token_w", self.weight_lens.token_w)?,
            token_b: cursor.take("token_b", self.weight_lens.token_b)?,
            token_gate_w: cursor.take("token_gate_w", self.weight_lens.token_gate_w)?,
            token_gate_b: cursor.take("token_gate_b", self.weight_lens.token_gate_b)?,
            state_w: cursor.take("state_w", self.weight_lens.state_w)?,
            time_w: cursor.take("time_w", self.weight_lens.time_w)?,
            output_w: cursor.take("output_w", self.weight_lens.output_w)?,
            output_b: cursor.take("output_b", self.weight_lens.output_b)?,
        };
        cursor.finish()?;
        let hyper = E2eHyperNpa2d {
            version: self.version,
            architecture: self.architecture,
            backend: self.backend,
            condition_encoder: self.condition_encoder,
            condition_token_count: self.condition_token_count,
            condition_embed_dims: self.condition_embed_dims,
            condition_token_grid_width: self.condition_token_grid_width,
            condition_token_grid_height: self.condition_token_grid_height,
            hidden_dims: self.hidden_dims,
            token_attention_heads: self.token_attention_heads,
            output_dims: self.output_dims,
            sample_steps: self.sample_steps,
            output_scale: self.output_scale,
            adapter_rank: self.adapter_rank,
            adapter_alpha: self.adapter_alpha,
            adapter_chunk_size: self.adapter_chunk_size,
            weights,
        };
        hyper.validate()?;
        Ok(hyper)
    }
}

struct F32Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> F32Cursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, name: &str, len: usize) -> AutomataResult<Vec<f32>> {
        let byte_len = len.checked_mul(4).ok_or_else(|| {
            AutomataError::InvalidFormat(format!("E2E HyperNPA {name} byte length overflow"))
        })?;
        let end = self.offset.checked_add(byte_len).ok_or_else(|| {
            AutomataError::InvalidFormat(format!("E2E HyperNPA {name} offset overflow"))
        })?;
        if end > self.bytes.len() {
            return Err(AutomataError::InvalidFormat(format!(
                "E2E HyperNPA {name} requires {byte_len} bytes at offset {}, but payload has {} bytes",
                self.offset,
                self.bytes.len()
            )));
        }
        let mut values = Vec::with_capacity(len);
        for chunk in self.bytes[self.offset..end].chunks_exact(4) {
            values.push(f32::from_le_bytes(chunk.try_into().expect("f32 chunk")));
        }
        self.offset = end;
        Ok(values)
    }

    fn finish(self) -> AutomataResult<()> {
        if self.offset != self.bytes.len() {
            return Err(AutomataError::InvalidFormat(format!(
                "E2E HyperNPA weight payload has {} trailing bytes",
                self.bytes.len() - self.offset
            )));
        }
        Ok(())
    }
}

fn is_json_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
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
            condition_encoder: None,
            condition_token_count: None,
            condition_embed_dims: None,
            condition_token_grid_width: None,
            condition_token_grid_height: None,
            hidden_dims,
            token_attention_heads: 1,
            output_dims,
            sample_steps: 1,
            output_scale: 1.0,
            adapter_rank: None,
            adapter_alpha: None,
            adapter_chunk_size: None,
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

    fn tiny_spatial_hyper(config: &NpaConfig, rank: usize, chunk_size: usize) -> E2eHyperNpa2d {
        let hidden_dims = 2;
        let embed_dims = 3;
        let output_dims = NpaLowRankAdapter::parameter_count_for_config(config, rank);
        let chunks = output_dims.div_ceil(chunk_size);
        let mut output_b = NpaLowRankAdapter::seeded_zero_delta(config, rank, rank as f32, 9)
            .to_parameter_vector();
        output_b.resize(chunks * chunk_size, 0.0);
        E2eHyperNpa2d {
            version: 1,
            architecture: E2E_HYPER_ARCH_SPATIAL_TOKEN_FLOW.to_string(),
            backend: None,
            condition_encoder: Some("dino-vits-full-tokens".to_string()),
            condition_token_count: Some(2),
            condition_embed_dims: Some(embed_dims),
            condition_token_grid_width: Some(1),
            condition_token_grid_height: Some(1),
            hidden_dims,
            token_attention_heads: 1,
            output_dims,
            sample_steps: 1,
            output_scale: 1.0,
            adapter_rank: Some(rank),
            adapter_alpha: Some(rank as f32),
            adapter_chunk_size: Some(chunk_size),
            weights: E2eHyperNpa2dWeights {
                token_w: vec![0.0; hidden_dims * embed_dims],
                token_b: vec![0.0; hidden_dims],
                token_gate_w: vec![0.0; chunks * hidden_dims],
                token_gate_b: vec![0.0; chunks * hidden_dims],
                state_w: vec![0.0; hidden_dims * chunk_size],
                time_w: vec![0.0; hidden_dims],
                output_w: vec![0.0; chunk_size * hidden_dims],
                output_b,
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
    fn e2e_hyper_binary_artifact_round_trips() {
        let (config, _) = NpaConfig::for_preset(AutomataPreset::Growing2d);
        let mut hyper = tiny_hyper(&config, 4);
        hyper.backend = Some("test".to_string());
        hyper.condition_encoder = Some("dino-vits-token-grid".to_string());
        hyper.condition_token_count = Some(2);
        hyper.condition_embed_dims = Some(hyper.embed_dims().unwrap());
        hyper.condition_token_grid_width = Some(1);
        hyper.condition_token_grid_height = Some(1);
        hyper.adapter_rank = Some(4);
        hyper.adapter_alpha = Some(4.0);

        let encoded = encode_e2e_hyper_npa_2d(&hyper).unwrap();
        let json = serde_json::to_vec(&hyper).unwrap();
        assert!(encoded.len() < json.len());
        let decoded = decode_e2e_hyper_npa_2d(&encoded).unwrap();
        assert_eq!(decoded.architecture, hyper.architecture);
        assert_eq!(decoded.backend, hyper.backend);
        assert_eq!(decoded.condition_encoder, hyper.condition_encoder);
        assert_eq!(decoded.weights.output_w, hyper.weights.output_w);
        assert_eq!(decoded.weights.output_b, hyper.weights.output_b);
        assert_eq!(e2e_hyper_bpk_payload_sha256(&encoded).unwrap().len(), 64);
    }

    #[test]
    fn e2e_hyper_spatial_token_artifact_round_trips_and_predicts() {
        let (config, _) = NpaConfig::for_preset(AutomataPreset::Growing2d);
        let hyper = tiny_spatial_hyper(&config, 4, 7);
        hyper.validate().unwrap();
        assert!(hyper.is_spatial_token_flow());
        assert_eq!(hyper.spatial_chunk_count(), hyper.output_dims.div_ceil(7));

        let condition = vec![0.0_f32; 2 * hyper.embed_dims().unwrap()];
        let adapter = hyper.predict_adapter(&config, &condition).unwrap();
        assert_eq!(adapter.parameter_count(), hyper.output_dims);

        let encoded = encode_e2e_hyper_npa_2d(&hyper).unwrap();
        let decoded = decode_e2e_hyper_npa_2d(&encoded).unwrap();
        assert_eq!(decoded.architecture, E2E_HYPER_ARCH_SPATIAL_TOKEN_FLOW);
        assert_eq!(decoded.adapter_chunk_size, Some(7));
        let decoded_adapter = decoded.predict_adapter(&config, &condition).unwrap();
        assert_eq!(decoded_adapter.parameter_count(), hyper.output_dims);
    }

    #[test]
    fn e2e_hyper_save_refuses_json_weight_artifact() {
        let (config, _) = NpaConfig::for_preset(AutomataPreset::Growing2d);
        let hyper = tiny_hyper(&config, 2);
        let path = std::env::temp_dir().join(format!(
            "burn_automata_hyper_refuse_{}.json",
            std::process::id()
        ));
        let err = save_e2e_hyper_npa_2d(&path, &hyper).unwrap_err();
        assert!(err.to_string().contains("refusing to write"));
        assert!(!path.exists());
    }

    #[test]
    fn local_default_e2e_artifacts_load_when_present() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let artifact_dir = workspace.join(
            "artifacts/hyper2d_e2e_rollout_train_omnisvg_10k_steps3000_b16_p128s4_rank16_cosine_cuda",
        );
        let base_path = artifact_dir.join("shared_base.bpk");
        let hyper_bpk_path = artifact_dir.join("hyper_2d.bpk");
        let hyper_json_path = artifact_dir.join("hyper_2d.json");
        let hyper_path = if hyper_bpk_path.exists() {
            hyper_bpk_path
        } else {
            hyper_json_path
        };
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
