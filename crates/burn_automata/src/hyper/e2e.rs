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
pub const E2E_HYPER_ARCH_MODULE_TOKEN_DECODER_V2: &str = "module_token_cross_attention_lora_v2";
pub const E2E_HYPER_ARCH_MODULE_TOKEN_DECODER: &str =
    "module_token_multihead_cross_attention_lora_v3";
pub const E2E_HYPER_ARCH_SAMPLE_ID_TABLE: &str = "sample_id_adapter_table_v1";
pub const E2E_HYPER_ATTENTION_TANH_EXP: &str = "tanh-exp";
pub const E2E_HYPER_ATTENTION_SOFTMAX: &str = "softmax";
pub const E2E_HYPER_ADAPTER_FACTORIZED: &str = "factorized";
pub const E2E_HYPER_ADAPTER_CANONICAL_FULL_RANK: &str = "canonical-full-rank";
pub const DEFAULT_E2E_HYPER_ADAPTER_CHUNK_SIZE: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum E2eHyperGeneratorKind {
    PooledFlow,
    SpatialTokenFlow,
    ModuleTokenDecoderV2,
    ModuleTokenDecoder,
    SampleIdTable,
}

impl E2eHyperGeneratorKind {
    pub const fn artifact_architecture(self) -> &'static str {
        match self {
            Self::PooledFlow => E2E_HYPER_ARCH_POOLED_FLOW,
            Self::SpatialTokenFlow => E2E_HYPER_ARCH_SPATIAL_TOKEN_FLOW,
            Self::ModuleTokenDecoderV2 => E2E_HYPER_ARCH_MODULE_TOKEN_DECODER_V2,
            Self::ModuleTokenDecoder => E2E_HYPER_ARCH_MODULE_TOKEN_DECODER,
            Self::SampleIdTable => E2E_HYPER_ARCH_SAMPLE_ID_TABLE,
        }
    }

    pub const fn is_module_token_decoder(self) -> bool {
        matches!(self, Self::ModuleTokenDecoderV2 | Self::ModuleTokenDecoder)
    }

    pub const fn is_chunked(self) -> bool {
        matches!(
            self,
            Self::SpatialTokenFlow | Self::ModuleTokenDecoderV2 | Self::ModuleTokenDecoder
        )
    }

    pub fn parse(value: Option<&str>) -> AutomataResult<Self> {
        let normalized = value.unwrap_or("module-token-decoder").trim();
        match normalized {
            "token-aware-rectified-flow"
            | "token-attention-pool"
            | "pooled-token-flow"
            | E2E_HYPER_ARCH_POOLED_FLOW => Ok(Self::PooledFlow),
            "spatial-token-flow"
            | "spatial-token-rectified-flow"
            | "spatial-token-chunked-flow"
            | E2E_HYPER_ARCH_SPATIAL_TOKEN_FLOW => Ok(Self::SpatialTokenFlow),
            "module-token-decoder-v2" | E2E_HYPER_ARCH_MODULE_TOKEN_DECODER_V2 => {
                Ok(Self::ModuleTokenDecoderV2)
            }
            "module-token-decoder"
            | "module-token-cross-attention"
            | "structured-token-decoder"
            | "module-token-decoder-v3"
            | E2E_HYPER_ARCH_MODULE_TOKEN_DECODER => Ok(Self::ModuleTokenDecoder),
            "sample-id-table" | "adapter-table" | E2E_HYPER_ARCH_SAMPLE_ID_TABLE => {
                Ok(Self::SampleIdTable)
            }
            other => Err(AutomataError::InvalidArgument(format!(
                "unknown HyperNPA adapter generator {other:?}; expected module-token-decoder, module-token-decoder-v2, sample-id-table, spatial-token-flow, or pooled-token-flow"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Target2dLossBackend {
    Dense,
    TiledAdjoint,
    #[default]
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
    Dense,
    TiledAdjoint,
    #[default]
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition_image_size: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition_alpha_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition_rgb_channels: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition_rgb_channel_scale: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition_alpha_channel: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition_alpha_channel_scale: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition_l2_normalize_features: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition_resize_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition_application: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_base_sha256: Option<String>,
    pub hidden_dims: usize,
    pub token_attention_heads: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_normalization: Option<String>,
    pub output_dims: usize,
    pub sample_steps: usize,
    pub output_scale: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_rank: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_alpha: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_parameterization: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_chunk_size: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spatial_condition_control: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spatial_condition_control_scale: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spatial_condition_control_sigma: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spatial_condition_state_control: Option<bool>,
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub condition_control_w: Vec<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub condition_control_b: Vec<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub condition_control_state_w: Vec<f32>,
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
    #[serde(default)]
    condition_image_size: Option<usize>,
    #[serde(default)]
    condition_alpha_mode: Option<String>,
    #[serde(default)]
    condition_rgb_channels: Option<bool>,
    #[serde(default)]
    condition_rgb_channel_scale: Option<f32>,
    #[serde(default)]
    condition_alpha_channel: Option<bool>,
    condition_alpha_channel_scale: Option<f32>,
    #[serde(default)]
    condition_l2_normalize_features: Option<bool>,
    #[serde(default)]
    condition_resize_mode: Option<String>,
    #[serde(default)]
    condition_application: Option<String>,
    #[serde(default)]
    shared_base_sha256: Option<String>,
    hidden_dims: usize,
    token_attention_heads: usize,
    #[serde(default)]
    attention_normalization: Option<String>,
    output_dims: usize,
    sample_steps: usize,
    output_scale: f32,
    adapter_rank: Option<usize>,
    adapter_alpha: Option<f32>,
    #[serde(default)]
    adapter_parameterization: Option<String>,
    #[serde(default)]
    adapter_chunk_size: Option<usize>,
    #[serde(default)]
    spatial_condition_control: Option<bool>,
    #[serde(default)]
    spatial_condition_control_scale: Option<f32>,
    #[serde(default)]
    spatial_condition_control_sigma: Option<f32>,
    #[serde(default)]
    spatial_condition_state_control: Option<bool>,
    weight_lens: E2eHyperNpa2dWeightLens,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
struct E2eHyperNpa2dWeightLens {
    token_w: usize,
    token_b: usize,
    token_gate_w: usize,
    token_gate_b: usize,
    state_w: usize,
    time_w: usize,
    output_w: usize,
    output_b: usize,
    #[serde(default)]
    condition_control_w: usize,
    #[serde(default)]
    condition_control_b: usize,
    #[serde(default)]
    condition_control_state_w: usize,
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
            E2E_HYPER_ARCH_POOLED_FLOW
                | E2E_HYPER_ARCH_SPATIAL_TOKEN_FLOW
                | E2E_HYPER_ARCH_MODULE_TOKEN_DECODER_V2
                | E2E_HYPER_ARCH_MODULE_TOKEN_DECODER
                | E2E_HYPER_ARCH_SAMPLE_ID_TABLE
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
        if self.is_multihead_module_token_decoder()
            && !self.hidden_dims.is_multiple_of(self.token_attention_heads)
        {
            return Err(AutomataError::InvalidModel(format!(
                "multi-head module-token HyperNPA hidden_dims={} must be divisible by token_attention_heads={}",
                self.hidden_dims, self.token_attention_heads
            )));
        }
        if let Some(normalization) = self.attention_normalization.as_deref()
            && !matches!(
                normalization,
                E2E_HYPER_ATTENTION_TANH_EXP | E2E_HYPER_ATTENTION_SOFTMAX
            )
        {
            return Err(AutomataError::InvalidModel(format!(
                "unsupported E2E HyperNPA attention_normalization {normalization:?}"
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
        if let Some(parameterization) = self.adapter_parameterization.as_deref()
            && !matches!(
                parameterization,
                E2E_HYPER_ADAPTER_FACTORIZED | E2E_HYPER_ADAPTER_CANONICAL_FULL_RANK
            )
        {
            return Err(AutomataError::InvalidModel(format!(
                "unsupported E2E HyperNPA adapter_parameterization {parameterization:?}"
            )));
        }
        if self.uses_canonical_full_rank_lora()
            && (self.adapter_rank.is_none() || self.adapter_alpha.is_none())
        {
            return Err(AutomataError::InvalidModel(
                "canonical-full-rank HyperNPA artifacts require adapter_rank and adapter_alpha"
                    .to_string(),
            ));
        }
        if let Some(chunk_size) = self.adapter_chunk_size
            && chunk_size == 0
        {
            return Err(AutomataError::InvalidModel(
                "E2E HyperNPA adapter_chunk_size must be positive".to_string(),
            ));
        }
        let has_condition_control = self.has_spatial_condition_control();
        if has_condition_control {
            if !self
                .weights
                .condition_control_w
                .len()
                .is_multiple_of(self.hidden_dims)
            {
                return Err(AutomataError::InvalidModel(format!(
                    "E2E HyperNPA condition_control_w len {} is not divisible by hidden_dims {}",
                    self.weights.condition_control_w.len(),
                    self.hidden_dims
                )));
            }
            let update_dims = self.weights.condition_control_w.len() / self.hidden_dims;
            if update_dims == 0 || self.weights.condition_control_b.len() != update_dims {
                return Err(AutomataError::InvalidModel(format!(
                    "E2E HyperNPA condition control update dims {update_dims} do not match condition_control_b len {}",
                    self.weights.condition_control_b.len()
                )));
            }
            if let Some(scale) = self.spatial_condition_control_scale
                && !scale.is_finite()
            {
                return Err(AutomataError::InvalidModel(
                    "E2E HyperNPA spatial_condition_control_scale must be finite".to_string(),
                ));
            }
            if let Some(sigma) = self.spatial_condition_control_sigma
                && (!sigma.is_finite() || sigma <= 0.0)
            {
                return Err(AutomataError::InvalidModel(
                    "E2E HyperNPA spatial_condition_control_sigma must be positive and finite"
                        .to_string(),
                ));
            }
            if self.spatial_condition_state_control.unwrap_or(false) {
                if self.weights.condition_control_state_w.is_empty()
                    || !self
                        .weights
                        .condition_control_state_w
                        .len()
                        .is_multiple_of(self.hidden_dims)
                {
                    return Err(AutomataError::InvalidModel(format!(
                        "E2E HyperNPA condition_control_state_w len {} must be a non-zero multiple of hidden_dims {}",
                        self.weights.condition_control_state_w.len(),
                        self.hidden_dims,
                    )));
                }
            } else if !self.weights.condition_control_state_w.is_empty() {
                return Err(AutomataError::InvalidModel(
                    "E2E HyperNPA condition_control_state_w requires spatial_condition_state_control=true"
                        .to_string(),
                ));
            }
        } else if !self.weights.condition_control_w.is_empty()
            || !self.weights.condition_control_b.is_empty()
            || !self.weights.condition_control_state_w.is_empty()
        {
            return Err(AutomataError::InvalidModel(
                "E2E HyperNPA condition control weights require spatial_condition_control=true"
                    .to_string(),
            ));
        }
        if let Some(token_count) = self.condition_token_count
            && token_count == 0
        {
            return Err(AutomataError::InvalidModel(
                "E2E HyperNPA condition_token_count must be positive".to_string(),
            ));
        }
        if self.is_sample_id_table()
            && (self.condition_token_count != Some(1)
                || self.condition_encoder.as_deref() != Some("sample-id-onehot"))
        {
            return Err(AutomataError::InvalidModel(
                "sample-ID adapter table requires condition_encoder=sample-id-onehot and one condition token"
                    .to_string(),
            ));
        }
        if let Some(image_size) = self.condition_image_size
            && image_size == 0
        {
            return Err(AutomataError::InvalidModel(
                "E2E HyperNPA condition_image_size must be positive".to_string(),
            ));
        }
        if let Some(alpha_mode) = &self.condition_alpha_mode
            && alpha_mode != "composite-white"
        {
            return Err(AutomataError::InvalidModel(format!(
                "unsupported E2E HyperNPA condition_alpha_mode {alpha_mode:?}"
            )));
        }
        if let Some(scale) = self.condition_alpha_channel_scale
            && (!scale.is_finite() || scale <= 0.0)
        {
            return Err(AutomataError::InvalidModel(format!(
                "E2E HyperNPA condition_alpha_channel_scale must be positive and finite, got {scale}"
            )));
        }
        if let Some(scale) = self.condition_rgb_channel_scale
            && (!scale.is_finite() || scale <= 0.0)
        {
            return Err(AutomataError::InvalidModel(format!(
                "E2E HyperNPA condition_rgb_channel_scale must be positive and finite, got {scale}"
            )));
        }
        if let Some(resize_mode) = &self.condition_resize_mode
            && resize_mode != "stretch"
        {
            return Err(AutomataError::InvalidModel(format!(
                "unsupported E2E HyperNPA condition_resize_mode {resize_mode:?}"
            )));
        }
        if let Some(application) = &self.condition_application
            && !matches!(application.as_str(), "static-adapter" | "per-step-field")
        {
            return Err(AutomataError::InvalidModel(format!(
                "unsupported E2E HyperNPA condition_application {application:?}"
            )));
        }
        if self.condition_application.as_deref() == Some("static-adapter")
            && self.has_spatial_condition_control()
        {
            return Err(AutomataError::InvalidModel(
                "static-adapter HyperNPA cannot contain per-step condition-field weights"
                    .to_string(),
            ));
        }
        if let Some(digest) = &self.shared_base_sha256
            && (digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()))
        {
            return Err(AutomataError::InvalidModel(
                "E2E HyperNPA shared_base_sha256 must be a 64-character hexadecimal digest"
                    .to_string(),
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
        let expected = if self.is_sample_id_table() {
            [
                ("token_b", 1, self.weights.token_b.len()),
                ("token_gate_w", 1, self.weights.token_gate_w.len()),
                ("token_gate_b", 1, self.weights.token_gate_b.len()),
                ("state_w", 1, self.weights.state_w.len()),
                ("time_w", 1, self.weights.time_w.len()),
                ("output_w", 1, self.weights.output_w.len()),
                ("output_b", 1, self.weights.output_b.len()),
            ]
        } else if self.is_chunked_token_generator() {
            let chunk_size = self.adapter_chunk_size_value();
            if !self.weights.output_b.len().is_multiple_of(chunk_size) {
                return Err(AutomataError::InvalidModel(format!(
                    "E2E HyperNPA output_b len {} is not divisible by chunk size {chunk_size}",
                    self.weights.output_b.len()
                )));
            }
            let chunks = self.weights.output_b.len() / chunk_size;
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
        let projection_rows = if self.is_sample_id_table() {
            self.output_dims
        } else {
            self.hidden_dims
        };
        if projection_rows == 0 || !self.weights.token_w.len().is_multiple_of(projection_rows) {
            return Err(AutomataError::InvalidModel(format!(
                "E2E HyperNPA token_w len {} is not divisible by projection rows {}",
                self.weights.token_w.len(),
                projection_rows
            )));
        }
        let embed_dims = self.weights.token_w.len() / projection_rows;
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

    pub fn is_module_token_decoder(&self) -> bool {
        matches!(
            self.architecture.as_str(),
            E2E_HYPER_ARCH_MODULE_TOKEN_DECODER_V2 | E2E_HYPER_ARCH_MODULE_TOKEN_DECODER
        )
    }

    pub fn is_multihead_module_token_decoder(&self) -> bool {
        self.architecture == E2E_HYPER_ARCH_MODULE_TOKEN_DECODER
    }

    pub fn is_sample_id_table(&self) -> bool {
        self.architecture == E2E_HYPER_ARCH_SAMPLE_ID_TABLE
    }

    pub fn is_chunked_token_generator(&self) -> bool {
        self.is_spatial_token_flow() || self.is_module_token_decoder()
    }

    pub fn uses_softmax_attention(&self) -> bool {
        self.attention_normalization.as_deref() == Some(E2E_HYPER_ATTENTION_SOFTMAX)
    }

    pub fn uses_canonical_full_rank_lora(&self) -> bool {
        self.adapter_parameterization.as_deref() == Some(E2E_HYPER_ADAPTER_CANONICAL_FULL_RANK)
    }

    fn normalized_attention_weights(&self, logits: &[f32]) -> AutomataResult<Vec<f32>> {
        if logits.is_empty() || !logits.iter().all(|value| value.is_finite()) {
            return Err(AutomataError::InvalidModel(
                "E2E HyperNPA attention logits must be non-empty and finite".to_string(),
            ));
        }
        let mut weights = Vec::with_capacity(logits.len());
        if self.uses_softmax_attention() {
            let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            weights.extend(logits.iter().map(|logit| (*logit - max_logit).exp()));
        } else {
            weights.extend(logits.iter().map(|logit| logit.tanh().exp()));
        }
        let denominator = weights.iter().sum::<f32>();
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(AutomataError::InvalidModel(
                "E2E HyperNPA attention normalization is not finite".to_string(),
            ));
        }
        for weight in &mut weights {
            *weight /= denominator;
        }
        Ok(weights)
    }

    pub fn has_spatial_condition_control(&self) -> bool {
        self.spatial_condition_control.unwrap_or(false)
            || !self.weights.condition_control_w.is_empty()
            || !self.weights.condition_control_b.is_empty()
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
        if self.has_spatial_condition_control() {
            return Err(AutomataError::InvalidModel(
                "E2E HyperNPA artifact uses spatial condition control and cannot be collapsed to a static LoRA adapter".to_string(),
            ));
        }
        if self.is_sample_id_table() {
            return self.predict_sample_id_table_adapter(config, condition_tokens);
        }
        if self.is_chunked_token_generator() {
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

        let mut attended = vec![0.0_f32; hidden_dims];
        for head in 0..heads {
            let mut logits = Vec::with_capacity(token_count);
            for token in 0..token_count {
                let token_base = token * hidden_dims;
                let mut logit = self.weights.token_gate_b[head];
                let weight_base = head * hidden_dims;
                for hidden in 0..hidden_dims {
                    logit += token_hidden[token_base + hidden]
                        * self.weights.token_gate_w[weight_base + hidden];
                }
                logits.push(logit);
            }
            let weights = self.normalized_attention_weights(&logits)?;
            for (token, weight) in weights.into_iter().enumerate() {
                let token_base = token * hidden_dims;
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

        self.adapter_from_generated_vector(config, vector)
    }

    fn predict_sample_id_table_adapter(
        &self,
        config: &NpaConfig,
        condition_tokens: &[f32],
    ) -> AutomataResult<NpaLowRankAdapter> {
        let embed_dims = self.embed_dims()?;
        if condition_tokens.len() != embed_dims {
            return Err(AutomataError::InvalidArgument(format!(
                "sample-ID adapter table requires one condition token with {embed_dims} values, got {}",
                condition_tokens.len()
            )));
        }
        if !condition_tokens.iter().all(|value| value.is_finite()) {
            return Err(AutomataError::InvalidArgument(
                "sample-ID adapter table condition contains non-finite values".to_string(),
            ));
        }
        let mut vector = vec![0.0; self.output_dims];
        for (output, value) in vector.iter_mut().enumerate() {
            let row = &self.weights.token_w[output * embed_dims..(output + 1) * embed_dims];
            *value = row
                .iter()
                .zip(condition_tokens.iter())
                .map(|(weight, condition)| weight * condition)
                .sum();
        }
        self.adapter_from_generated_vector(config, vector)
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
        let module_layout = if self.is_module_token_decoder() {
            let spec = self.adapter_spec(config)?;
            Some(crate::hyper::adapter_layout::AdapterParameterLayout2d::new(
                config, spec.rank, chunk_size,
            )?)
        } else {
            None
        };
        let chunks = module_layout
            .as_ref()
            .map_or_else(|| self.spatial_chunk_count(), |layout| layout.chunk_count);
        if self.weights.output_b.len() != chunks * chunk_size {
            return Err(AutomataError::InvalidModel(format!(
                "E2E HyperNPA chunk output count {} does not match expected {}",
                self.weights.output_b.len(),
                chunks * chunk_size
            )));
        }
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
        let attention_heads = if self.is_multihead_module_token_decoder() {
            self.token_attention_heads
        } else {
            1
        };
        let head_dims = hidden_dims / attention_heads;
        let attention_scale = 1.0 / (head_dims as f32).sqrt().max(1.0);
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

                let mut attended = vec![0.0_f32; hidden_dims];
                for head in 0..attention_heads {
                    let hidden_start = head * head_dims;
                    let hidden_end = hidden_start + head_dims;
                    let mut attention_logits = vec![0.0_f32; token_count];
                    for (token, attention_logit) in attention_logits.iter_mut().enumerate() {
                        let token_base = token * hidden_dims;
                        let mut logit = 0.0_f32;
                        for hidden in hidden_start..hidden_end {
                            logit += query_hidden[hidden] * token_hidden[token_base + hidden];
                        }
                        *attention_logit = logit * attention_scale;
                    }
                    let attention_weights = self.normalized_attention_weights(&attention_logits)?;
                    for (token, attention_weight) in attention_weights.iter().enumerate() {
                        let token_base = token * hidden_dims;
                        for hidden in hidden_start..hidden_end {
                            attended[hidden] +=
                                token_hidden[token_base + hidden] * *attention_weight;
                        }
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
        let chunk_state = if let Some(layout) = module_layout {
            layout.unpack(&chunk_state)?
        } else {
            chunk_state.truncate(self.output_dims);
            chunk_state
        };

        self.adapter_from_generated_vector(config, chunk_state)
    }

    fn adapter_from_generated_vector(
        &self,
        config: &NpaConfig,
        mut vector: Vec<f32>,
    ) -> AutomataResult<NpaLowRankAdapter> {
        let spec = self.adapter_spec(config)?;
        if self.uses_canonical_full_rank_lora() {
            vector = crate::hyper::adapter_layout::CanonicalFullRankLora2d::new(
                config, spec.rank, spec.alpha,
            )?
            .apply(&vector)?;
        }
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
            ("condition_control_w", self.condition_control_w.as_slice()),
            ("condition_control_b", self.condition_control_b.as_slice()),
            (
                "condition_control_state_w",
                self.condition_control_state_w.as_slice(),
            ),
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
            condition_control_w: self.condition_control_w.len(),
            condition_control_b: self.condition_control_b.len(),
            condition_control_state_w: self.condition_control_state_w.len(),
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
            + self.output_b.len()
            + self.condition_control_w.len()
            + self.condition_control_b.len()
            + self.condition_control_state_w.len();
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
            self.condition_control_w.as_slice(),
            self.condition_control_b.as_slice(),
            self.condition_control_state_w.as_slice(),
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
            condition_image_size: hyper.condition_image_size,
            condition_alpha_mode: hyper.condition_alpha_mode.clone(),
            condition_rgb_channels: hyper.condition_rgb_channels,
            condition_rgb_channel_scale: hyper.condition_rgb_channel_scale,
            condition_alpha_channel: hyper.condition_alpha_channel,
            condition_alpha_channel_scale: hyper.condition_alpha_channel_scale,
            condition_l2_normalize_features: hyper.condition_l2_normalize_features,
            condition_resize_mode: hyper.condition_resize_mode.clone(),
            condition_application: hyper.condition_application.clone(),
            shared_base_sha256: hyper.shared_base_sha256.clone(),
            hidden_dims: hyper.hidden_dims,
            token_attention_heads: hyper.token_attention_heads,
            attention_normalization: hyper.attention_normalization.clone(),
            output_dims: hyper.output_dims,
            sample_steps: hyper.sample_steps,
            output_scale: hyper.output_scale,
            adapter_rank: hyper.adapter_rank,
            adapter_alpha: hyper.adapter_alpha,
            adapter_parameterization: hyper.adapter_parameterization.clone(),
            adapter_chunk_size: hyper.adapter_chunk_size,
            spatial_condition_control: hyper.spatial_condition_control,
            spatial_condition_control_scale: hyper.spatial_condition_control_scale,
            spatial_condition_control_sigma: hyper.spatial_condition_control_sigma,
            spatial_condition_state_control: hyper.spatial_condition_state_control,
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
            condition_control_w: cursor
                .take("condition_control_w", self.weight_lens.condition_control_w)?,
            condition_control_b: cursor
                .take("condition_control_b", self.weight_lens.condition_control_b)?,
            condition_control_state_w: cursor.take(
                "condition_control_state_w",
                self.weight_lens.condition_control_state_w,
            )?,
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
            condition_image_size: self.condition_image_size,
            condition_alpha_mode: self.condition_alpha_mode,
            condition_rgb_channels: self.condition_rgb_channels,
            condition_rgb_channel_scale: self.condition_rgb_channel_scale,
            condition_alpha_channel: self.condition_alpha_channel,
            condition_alpha_channel_scale: self.condition_alpha_channel_scale,
            condition_l2_normalize_features: self.condition_l2_normalize_features,
            condition_resize_mode: self.condition_resize_mode,
            condition_application: self.condition_application,
            shared_base_sha256: self.shared_base_sha256,
            hidden_dims: self.hidden_dims,
            token_attention_heads: self.token_attention_heads,
            attention_normalization: self.attention_normalization,
            output_dims: self.output_dims,
            sample_steps: self.sample_steps,
            output_scale: self.output_scale,
            adapter_rank: self.adapter_rank,
            adapter_alpha: self.adapter_alpha,
            adapter_parameterization: self.adapter_parameterization,
            adapter_chunk_size: self.adapter_chunk_size,
            spatial_condition_control: self.spatial_condition_control,
            spatial_condition_control_scale: self.spatial_condition_control_scale,
            spatial_condition_control_sigma: self.spatial_condition_control_sigma,
            spatial_condition_state_control: self.spatial_condition_state_control,
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
            condition_image_size: None,
            condition_alpha_mode: None,
            condition_rgb_channels: None,
            condition_rgb_channel_scale: None,
            condition_alpha_channel: None,
            condition_alpha_channel_scale: None,
            condition_l2_normalize_features: None,
            condition_resize_mode: None,
            condition_application: None,
            shared_base_sha256: None,
            hidden_dims,
            token_attention_heads: 1,
            attention_normalization: None,
            output_dims,
            sample_steps: 1,
            output_scale: 1.0,
            adapter_rank: None,
            adapter_alpha: None,
            adapter_parameterization: None,
            adapter_chunk_size: None,
            spatial_condition_control: None,
            spatial_condition_control_scale: None,
            spatial_condition_control_sigma: None,
            spatial_condition_state_control: None,
            weights: E2eHyperNpa2dWeights {
                token_w: vec![0.01; hidden_dims * embed_dims],
                token_b: vec![0.0; hidden_dims],
                token_gate_w: vec![0.01; hidden_dims],
                token_gate_b: vec![0.0],
                state_w: vec![0.0; hidden_dims * output_dims],
                time_w: vec![0.0; hidden_dims],
                output_w: vec![0.001; output_dims * hidden_dims],
                output_b: vec![0.0; output_dims],
                condition_control_w: Vec::new(),
                condition_control_b: Vec::new(),
                condition_control_state_w: Vec::new(),
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
            condition_image_size: Some(224),
            condition_alpha_mode: Some("composite-white".to_string()),
            condition_rgb_channels: Some(false),
            condition_rgb_channel_scale: Some(1.0),
            condition_alpha_channel: Some(false),
            condition_alpha_channel_scale: Some(1.0),
            condition_l2_normalize_features: Some(false),
            condition_resize_mode: Some("stretch".to_string()),
            condition_application: Some("static-adapter".to_string()),
            shared_base_sha256: None,
            hidden_dims,
            token_attention_heads: 1,
            attention_normalization: None,
            output_dims,
            sample_steps: 1,
            output_scale: 1.0,
            adapter_rank: Some(rank),
            adapter_alpha: Some(rank as f32),
            adapter_parameterization: None,
            adapter_chunk_size: Some(chunk_size),
            spatial_condition_control: None,
            spatial_condition_control_scale: None,
            spatial_condition_control_sigma: None,
            spatial_condition_state_control: None,
            weights: E2eHyperNpa2dWeights {
                token_w: vec![0.0; hidden_dims * embed_dims],
                token_b: vec![0.0; hidden_dims],
                token_gate_w: vec![0.0; chunks * hidden_dims],
                token_gate_b: vec![0.0; chunks * hidden_dims],
                state_w: vec![0.0; hidden_dims * chunk_size],
                time_w: vec![0.0; hidden_dims],
                output_w: vec![0.0; chunk_size * hidden_dims],
                output_b,
                condition_control_w: Vec::new(),
                condition_control_b: Vec::new(),
                condition_control_state_w: Vec::new(),
            },
        }
    }

    fn tiny_module_hyper(config: &NpaConfig, rank: usize, chunk_size: usize) -> E2eHyperNpa2d {
        let hidden_dims = 2;
        let embed_dims = 3;
        let layout =
            crate::hyper::adapter_layout::AdapterParameterLayout2d::new(config, rank, chunk_size)
                .unwrap();
        let output_dims = layout.parameter_count;
        let output_b = layout
            .pack(
                &NpaLowRankAdapter::seeded_zero_delta(config, rank, rank as f32, 19)
                    .to_parameter_vector(),
            )
            .unwrap();
        E2eHyperNpa2d {
            version: 1,
            architecture: E2E_HYPER_ARCH_MODULE_TOKEN_DECODER.to_string(),
            backend: None,
            condition_encoder: Some("dino-vits-full-tokens".to_string()),
            condition_token_count: Some(2),
            condition_embed_dims: Some(embed_dims),
            condition_token_grid_width: Some(1),
            condition_token_grid_height: Some(1),
            condition_image_size: Some(224),
            condition_alpha_mode: Some("composite-white".to_string()),
            condition_rgb_channels: Some(false),
            condition_rgb_channel_scale: Some(1.0),
            condition_alpha_channel: Some(false),
            condition_alpha_channel_scale: Some(1.0),
            condition_l2_normalize_features: Some(false),
            condition_resize_mode: Some("stretch".to_string()),
            condition_application: Some("static-adapter".to_string()),
            shared_base_sha256: None,
            hidden_dims,
            token_attention_heads: 1,
            attention_normalization: None,
            output_dims,
            sample_steps: 1,
            output_scale: 1.0,
            adapter_rank: Some(rank),
            adapter_alpha: Some(rank as f32),
            adapter_parameterization: None,
            adapter_chunk_size: Some(chunk_size),
            spatial_condition_control: None,
            spatial_condition_control_scale: None,
            spatial_condition_control_sigma: None,
            spatial_condition_state_control: None,
            weights: E2eHyperNpa2dWeights {
                token_w: vec![0.0; hidden_dims * embed_dims],
                token_b: vec![0.0; hidden_dims],
                token_gate_w: vec![0.0; layout.chunk_count * hidden_dims],
                token_gate_b: layout.structured_query_initialization(hidden_dims, 0.01),
                state_w: vec![0.0; hidden_dims * chunk_size],
                time_w: vec![0.0; hidden_dims],
                output_w: vec![0.0; chunk_size * hidden_dims],
                output_b,
                condition_control_w: Vec::new(),
                condition_control_b: Vec::new(),
                condition_control_state_w: Vec::new(),
            },
        }
    }

    fn tiny_sample_id_table(config: &NpaConfig, rank: usize) -> E2eHyperNpa2d {
        let embed_dims = 2;
        let output_dims = NpaLowRankAdapter::parameter_count_for_config(config, rank);
        let first = NpaLowRankAdapter::seeded_zero_delta(config, rank, rank as f32, 31)
            .to_parameter_vector();
        let second = NpaLowRankAdapter::seeded_zero_delta(config, rank, rank as f32, 47)
            .to_parameter_vector();
        let mut token_w = vec![0.0; output_dims * embed_dims];
        for output in 0..output_dims {
            token_w[output * embed_dims] = first[output];
            token_w[output * embed_dims + 1] = second[output];
        }
        E2eHyperNpa2d {
            version: 1,
            architecture: E2E_HYPER_ARCH_SAMPLE_ID_TABLE.to_string(),
            backend: None,
            condition_encoder: Some("sample-id-onehot".to_string()),
            condition_token_count: Some(1),
            condition_embed_dims: Some(embed_dims),
            condition_token_grid_width: Some(1),
            condition_token_grid_height: Some(1),
            condition_image_size: None,
            condition_alpha_mode: None,
            condition_rgb_channels: Some(false),
            condition_rgb_channel_scale: Some(1.0),
            condition_alpha_channel: Some(false),
            condition_alpha_channel_scale: Some(1.0),
            condition_l2_normalize_features: Some(false),
            condition_resize_mode: None,
            condition_application: Some("static-adapter".to_string()),
            shared_base_sha256: None,
            hidden_dims: 1,
            token_attention_heads: 1,
            attention_normalization: None,
            output_dims,
            sample_steps: 1,
            output_scale: 1.0,
            adapter_rank: Some(rank),
            adapter_alpha: Some(rank as f32),
            adapter_parameterization: None,
            adapter_chunk_size: None,
            spatial_condition_control: None,
            spatial_condition_control_scale: None,
            spatial_condition_control_sigma: None,
            spatial_condition_state_control: None,
            weights: E2eHyperNpa2dWeights {
                token_w,
                token_b: vec![0.0],
                token_gate_w: vec![0.0],
                token_gate_b: vec![0.0],
                state_w: vec![0.0],
                time_w: vec![0.0],
                output_w: vec![0.0],
                output_b: vec![0.0],
                condition_control_w: Vec::new(),
                condition_control_b: Vec::new(),
                condition_control_state_w: Vec::new(),
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
    fn e2e_hyper_spatial_condition_control_round_trips_and_rejects_static_adapter() {
        let (config, _) = NpaConfig::for_preset(AutomataPreset::Growing2d);
        let mut hyper = tiny_spatial_hyper(&config, 4, 7);
        hyper.condition_application = Some("per-step-field".to_string());
        hyper.spatial_condition_control = Some(true);
        hyper.spatial_condition_control_scale = Some(0.1);
        hyper.spatial_condition_control_sigma = Some(0.25);
        hyper.spatial_condition_state_control = Some(true);
        hyper.weights.condition_control_w = vec![0.01; config.update_dims() * hyper.hidden_dims];
        hyper.weights.condition_control_b = vec![0.0; config.update_dims()];
        hyper.weights.condition_control_state_w = vec![0.0; config.state_dims * hyper.hidden_dims];
        hyper.validate().unwrap();

        let encoded = encode_e2e_hyper_npa_2d(&hyper).unwrap();
        let decoded = decode_e2e_hyper_npa_2d(&encoded).unwrap();
        assert!(decoded.has_spatial_condition_control());
        assert_eq!(decoded.spatial_condition_control_scale, Some(0.1));
        assert_eq!(decoded.spatial_condition_state_control, Some(true));
        assert_eq!(
            decoded.weights.condition_control_w,
            hyper.weights.condition_control_w
        );
        assert_eq!(
            decoded.weights.condition_control_state_w,
            hyper.weights.condition_control_state_w
        );

        let condition = vec![0.0_f32; 2 * hyper.embed_dims().unwrap()];
        let err = decoded.predict_adapter(&config, &condition).unwrap_err();
        assert!(
            err.to_string()
                .contains("cannot be collapsed to a static LoRA adapter")
        );
    }

    #[test]
    fn e2e_module_token_artifact_round_trips_and_preserves_parameter_layout() {
        let config = NpaConfig::growing_2d();
        let mut hyper = tiny_module_hyper(&config, 16, 64);
        hyper.attention_normalization = Some(E2E_HYPER_ATTENTION_SOFTMAX.to_string());
        hyper.validate().unwrap();
        let condition = vec![0.0; 2 * hyper.embed_dims().unwrap()];
        let adapter = hyper.predict_adapter(&config, &condition).unwrap();
        assert_eq!(adapter.parameter_count(), hyper.output_dims);

        let encoded = encode_e2e_hyper_npa_2d(&hyper).unwrap();
        let decoded = decode_e2e_hyper_npa_2d(&encoded).unwrap();
        assert!(decoded.is_module_token_decoder());
        assert!(decoded.uses_softmax_attention());
        assert_eq!(
            decoded
                .predict_adapter(&config, &condition)
                .unwrap()
                .to_parameter_vector(),
            adapter.to_parameter_vector()
        );
    }

    #[test]
    fn module_token_decoder_defaults_to_generalized_v3_but_keeps_v2_explicit() {
        assert_eq!(
            E2eHyperGeneratorKind::parse(None).unwrap(),
            E2eHyperGeneratorKind::ModuleTokenDecoder
        );
        assert_eq!(
            E2eHyperGeneratorKind::parse(Some("module-token-decoder")).unwrap(),
            E2eHyperGeneratorKind::ModuleTokenDecoder
        );
        assert_eq!(
            E2eHyperGeneratorKind::parse(Some("module-token-decoder-v2")).unwrap(),
            E2eHyperGeneratorKind::ModuleTokenDecoderV2
        );
    }

    #[test]
    fn module_token_v3_requires_even_attention_head_partitions() {
        let config = NpaConfig::growing_2d();
        let mut hyper = tiny_module_hyper(&config, 4, 16);
        hyper.hidden_dims = 2;
        hyper.token_attention_heads = 3;
        let error = hyper.validate().unwrap_err();
        assert!(error.to_string().contains("must be divisible"));

        hyper.architecture = E2E_HYPER_ARCH_MODULE_TOKEN_DECODER_V2.to_string();
        hyper.validate().unwrap();
    }

    #[test]
    fn e2e_canonical_full_rank_lora_round_trips_with_fixed_identity_factors() {
        let config = NpaConfig::growing_2d();
        let rank = config.perception_dims().max(config.update_dims());
        let mut hyper = tiny_module_hyper(&config, rank, 64);
        hyper.adapter_parameterization = Some(E2E_HYPER_ADAPTER_CANONICAL_FULL_RANK.to_string());
        hyper.weights.output_b.fill(0.0);
        let condition = vec![0.0; 2 * hyper.embed_dims().unwrap()];
        let adapter = hyper.predict_adapter(&config, &condition).unwrap();

        assert_eq!(adapter.w1_down[0], 1.0);
        assert_eq!(
            adapter.w1_down[(config.perception_dims() - 1) * config.perception_dims()
                + config.perception_dims()
                - 1],
            1.0
        );
        assert_eq!(adapter.w2_up[0], 1.0);
        assert!(adapter.w1_up.iter().all(|value| *value == 0.0));
        assert!(adapter.w2_down.iter().all(|value| *value == 0.0));

        let encoded = encode_e2e_hyper_npa_2d(&hyper).unwrap();
        let decoded = decode_e2e_hyper_npa_2d(&encoded).unwrap();
        assert!(decoded.uses_canonical_full_rank_lora());
        assert_eq!(
            decoded
                .predict_adapter(&config, &condition)
                .unwrap()
                .to_parameter_vector(),
            adapter.to_parameter_vector()
        );
    }

    #[test]
    fn e2e_softmax_attention_can_select_a_spatial_token_sharply() {
        let config = NpaConfig::growing_2d();
        let mut hyper = tiny_module_hyper(&config, 4, 16);
        let logits = [8.0, 0.0, 0.0, 0.0];
        let legacy = hyper.normalized_attention_weights(&logits).unwrap();
        hyper.attention_normalization = Some(E2E_HYPER_ATTENTION_SOFTMAX.to_string());
        let softmax = hyper.normalized_attention_weights(&logits).unwrap();

        assert!(legacy[0] < 0.72);
        assert!(softmax[0] > 0.99);
        assert!((softmax.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn e2e_module_token_adapter_changes_with_spatial_tokens() {
        let config = NpaConfig::growing_2d();
        let mut hyper = tiny_module_hyper(&config, 4, 16);
        let embed_dims = hyper.embed_dims().unwrap();
        hyper.weights.token_w = vec![0.0; hyper.hidden_dims * embed_dims];
        hyper.weights.token_w[0] = 1.0;
        hyper.weights.token_w[embed_dims + 1] = 1.0;
        hyper.weights.output_w.fill(0.1);

        let zero_condition = vec![0.0; 2 * embed_dims];
        let mut spatial_condition = zero_condition.clone();
        spatial_condition[embed_dims] = 1.0;
        let zero_adapter = hyper
            .predict_adapter(&config, &zero_condition)
            .unwrap()
            .to_parameter_vector();
        let spatial_adapter = hyper
            .predict_adapter(&config, &spatial_condition)
            .unwrap()
            .to_parameter_vector();
        let max_delta = zero_adapter
            .iter()
            .zip(spatial_adapter.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);

        assert!(
            max_delta > 1.0e-4,
            "module-token adapter ignored token content"
        );
    }

    #[test]
    fn e2e_sample_id_table_selects_independent_adapters() {
        let config = NpaConfig::growing_2d();
        let hyper = tiny_sample_id_table(&config, 4);
        hyper.validate().unwrap();
        let first = hyper
            .predict_adapter(&config, &[1.0, 0.0])
            .unwrap()
            .to_parameter_vector();
        let second = hyper
            .predict_adapter(&config, &[0.0, 1.0])
            .unwrap()
            .to_parameter_vector();
        assert!(
            first
                .iter()
                .zip(second.iter())
                .any(|(left, right)| (left - right).abs() > 1.0e-4)
        );

        let encoded = encode_e2e_hyper_npa_2d(&hyper).unwrap();
        let decoded = decode_e2e_hyper_npa_2d(&encoded).unwrap();
        assert!(decoded.is_sample_id_table());
        assert_eq!(
            decoded
                .predict_adapter(&config, &[1.0, 0.0])
                .unwrap()
                .to_parameter_vector(),
            first
        );
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
}
