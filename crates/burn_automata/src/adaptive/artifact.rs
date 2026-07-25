use std::{fs, path::Path};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{AdaptiveNpaConfig, AdaptiveNpaModel};
use crate::{AutomataError, AutomataResult};

const ADAPTIVE_MAGIC: [u8; 8] = *b"BANPABP1";
const ADAPTIVE_CONTAINER_VERSION: u32 = 1;
const HEADER_LEN: usize = 8 + 4 + 8 + 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdaptiveTrainingStage {
    /// Numerical/operator validation or fixed-rule compatibility only.
    FoundationCompatibility,
    /// Rule and controller were trained together across material resolutions.
    TaskTrainedMultiscale,
    /// Material rule, proxy, and controller were all trained from deterministic
    /// random initialization without imported task dynamics.
    FreshTaskTrainedMultiscale,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveModelArtifact {
    pub format_version: u32,
    pub model_kind: String,
    #[serde(default = "foundation_compatibility_stage")]
    pub training_stage: AdaptiveTrainingStage,
    pub source: Option<String>,
    pub model: AdaptiveNpaModel,
}

const fn foundation_compatibility_stage() -> AdaptiveTrainingStage {
    AdaptiveTrainingStage::FoundationCompatibility
}

impl AdaptiveModelArtifact {
    pub fn new(model: AdaptiveNpaModel, source: Option<String>) -> AutomataResult<Self> {
        model.validate()?;
        Ok(Self {
            format_version: 1,
            model_kind: "budgeted-adaptive-npa".to_string(),
            training_stage: AdaptiveTrainingStage::FoundationCompatibility,
            source,
            model,
        })
    }

    pub fn task_trained(model: AdaptiveNpaModel, source: Option<String>) -> AutomataResult<Self> {
        let mut artifact = Self::new(model, source)?;
        artifact.training_stage = AdaptiveTrainingStage::TaskTrainedMultiscale;
        Ok(artifact)
    }

    pub fn fresh_task_trained(
        model: AdaptiveNpaModel,
        source: Option<String>,
    ) -> AutomataResult<Self> {
        let mut artifact = Self::new(model, source)?;
        artifact.training_stage = AdaptiveTrainingStage::FreshTaskTrainedMultiscale;
        Ok(artifact)
    }

    pub fn validate(&self) -> AutomataResult<()> {
        if self.format_version != 1 || self.model_kind != "budgeted-adaptive-npa" {
            return Err(AutomataError::InvalidFormat(format!(
                "unsupported adaptive model artifact {} {:?}",
                self.format_version, self.model_kind
            )));
        }
        self.model.validate()
    }

    /// Rebinds trained weights to a validated deployment schedule without
    /// changing their training-stage provenance.
    pub fn with_runtime_config(
        mut self,
        config: AdaptiveNpaConfig,
        source: Option<String>,
    ) -> AutomataResult<Self> {
        self.model.config = config;
        if source.is_some() {
            self.source = source;
        }
        self.validate()?;
        Ok(self)
    }
}

pub fn save_adaptive_model(
    path: impl AsRef<Path>,
    artifact: &AdaptiveModelArtifact,
) -> AutomataResult<String> {
    artifact.validate()?;
    let payload = rmp_serde::to_vec_named(artifact)
        .map_err(|error| AutomataError::InvalidFormat(error.to_string()))?;
    let digest = Sha256::digest(&payload);
    let payload_len = u64::try_from(payload.len())
        .map_err(|_| AutomataError::InvalidFormat("adaptive payload too large".to_string()))?;
    let mut bytes = Vec::with_capacity(HEADER_LEN + payload.len());
    bytes.extend_from_slice(&ADAPTIVE_MAGIC);
    bytes.extend_from_slice(&ADAPTIVE_CONTAINER_VERSION.to_le_bytes());
    bytes.extend_from_slice(&payload_len.to_le_bytes());
    bytes.extend_from_slice(&digest);
    bytes.extend_from_slice(&payload);
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, bytes)?;
    Ok(hex_digest(&digest))
}

pub fn load_adaptive_model(path: impl AsRef<Path>) -> AutomataResult<AdaptiveModelArtifact> {
    let bytes = fs::read(path)?;
    if bytes.len() < HEADER_LEN || !bytes.starts_with(&ADAPTIVE_MAGIC) {
        return Err(AutomataError::InvalidFormat(
            "missing budgeted adaptive NPA binary header".to_string(),
        ));
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().expect("version header"));
    if version != ADAPTIVE_CONTAINER_VERSION {
        return Err(AutomataError::InvalidFormat(format!(
            "unsupported adaptive container version {version}"
        )));
    }
    let payload_len = u64::from_le_bytes(bytes[12..20].try_into().expect("length header")) as usize;
    if bytes.len() != HEADER_LEN + payload_len {
        return Err(AutomataError::InvalidFormat(
            "adaptive payload length mismatch".to_string(),
        ));
    }
    let payload = &bytes[HEADER_LEN..];
    let digest = Sha256::digest(payload);
    if digest[..] != bytes[20..52] {
        return Err(AutomataError::InvalidFormat(
            "adaptive payload sha256 mismatch".to_string(),
        ));
    }
    let mut artifact: AdaptiveModelArtifact = rmp_serde::from_slice(payload)
        .map_err(|error| AutomataError::InvalidFormat(error.to_string()))?;
    if artifact.model.config.closure_recurrent_mode {
        // Recurrent feature schemas only grow through zero-column input
        // expansion. Migrate before validation so older binary checkpoints
        // remain loadable without weakening the current model contract.
        artifact.model.enable_zero_closure_mode_rule()?;
    }
    artifact.validate()?;
    Ok(artifact)
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
