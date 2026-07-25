use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    AutomataError, AutomataResult, EquivarianceMode, NpaConfig, NpaLowRankAdapter, NpaModel,
    NpaWeights,
};
use burn_automata_kernels::{Boundary, HashGridConfig, HashGridMode};

pub const BPK_MAGIC: [u8; 8] = *b"BAUTBPK1";
const BPK_HEADER_LEN: usize = 8 + 4 + 8 + 32;
const BPK_CONTAINER_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BpkModelManifest {
    pub format_version: u32,
    pub model_kind: String,
    pub source: Option<String>,
    pub config: NpaConfig,
    pub hashgrid: HashGridConfig,
    pub weights: NpaWeights,
}

impl BpkModelManifest {
    pub fn from_model(model: &NpaModel, hashgrid: HashGridConfig, source: Option<String>) -> Self {
        Self {
            format_version: 1,
            model_kind: "npa".to_string(),
            source,
            config: model.config.clone(),
            hashgrid,
            weights: model.weights.clone(),
        }
    }

    pub fn into_model(self) -> NpaModel {
        NpaModel {
            config: self.config,
            weights: self.weights,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BpkAdapterManifest {
    pub format_version: u32,
    pub model_kind: String,
    pub source: Option<String>,
    pub base_model: Option<String>,
    pub base_source: Option<String>,
    pub config: NpaConfig,
    pub hashgrid: HashGridConfig,
    pub adapter: NpaLowRankAdapter,
}

impl BpkAdapterManifest {
    pub fn from_adapter(
        base_manifest: &BpkModelManifest,
        base_model: Option<String>,
        adapter: NpaLowRankAdapter,
        source: Option<String>,
    ) -> AutomataResult<Self> {
        adapter.validate(&base_manifest.config)?;
        Ok(Self {
            format_version: 1,
            model_kind: "npa-lora-adapter".to_string(),
            source,
            base_model,
            base_source: base_manifest.source.clone(),
            config: base_manifest.config.clone(),
            hashgrid: base_manifest.hashgrid.clone(),
            adapter,
        })
    }

    pub fn validate(&self, base_manifest: &BpkModelManifest) -> AutomataResult<()> {
        if self.format_version != 1 {
            return Err(AutomataError::InvalidFormat(format!(
                "unsupported adapter manifest version {}",
                self.format_version
            )));
        }
        if self.model_kind != "npa-lora-adapter" {
            return Err(AutomataError::InvalidFormat(format!(
                "adapter manifest has unexpected model_kind {:?}",
                self.model_kind
            )));
        }
        if self.config != base_manifest.config {
            return Err(AutomataError::InvalidModel(
                "adapter config does not match base model config".to_string(),
            ));
        }
        if self.hashgrid != base_manifest.hashgrid {
            return Err(AutomataError::InvalidModel(
                "adapter hashgrid does not match base model hashgrid".to_string(),
            ));
        }
        self.adapter.validate(&self.config)
    }

    pub fn materialize(
        &self,
        base_manifest: &BpkModelManifest,
    ) -> AutomataResult<BpkModelManifest> {
        self.validate(base_manifest)?;
        let base_model = NpaModel {
            config: base_manifest.config.clone(),
            weights: base_manifest.weights.clone(),
        };
        let materialized = self.adapter.apply_to_model(&base_model)?;
        Ok(BpkModelManifest::from_model(
            &materialized,
            base_manifest.hashgrid.clone(),
            self.source.clone().or_else(|| {
                base_manifest
                    .source
                    .as_ref()
                    .map(|source| format!("materialized-adapter:{source}"))
            }),
        ))
    }

    pub fn adapter_parameter_count(&self) -> usize {
        self.adapter.parameter_count()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExportedCheckpoint {
    pub config: NpaConfig,
    pub hashgrid: HashGridConfig,
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub source: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImportReport {
    pub output: String,
    pub format_version: u32,
    pub container: String,
    pub source: Option<String>,
    pub parameter_count: usize,
    pub sha256: Option<String>,
}

pub fn import_model(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
) -> AutomataResult<ImportReport> {
    let input = input.as_ref();
    if is_pytorch_checkpoint_path(input) {
        import_pytorch_npa_checkpoint(input, output)
    } else {
        import_exported_checkpoint(input, output)
    }
}

pub fn import_exported_checkpoint(
    input_json: impl AsRef<Path>,
    output_json: impl AsRef<Path>,
) -> AutomataResult<ImportReport> {
    let input_json = input_json.as_ref();
    let output_json = output_json.as_ref();
    let text = fs::read_to_string(input_json)?;
    let exported: ExportedCheckpoint = serde_json::from_str(&text)?;
    let model = NpaModel {
        config: exported.config,
        weights: NpaWeights {
            w1: exported.w1,
            b1: exported.b1,
            w2: exported.w2,
            b2: exported.b2,
        },
    };
    model.validate()?;
    let parameter_count = model.weights.w1.len()
        + model.weights.b1.len()
        + model.weights.w2.len()
        + model.weights.b2.len();
    let manifest = BpkModelManifest::from_model(&model, exported.hashgrid, exported.source.clone());
    if let Some(parent) = output_json.parent() {
        fs::create_dir_all(parent)?;
    }
    let sha256 = save_manifest(output_json, &manifest)?;
    Ok(ImportReport {
        output: output_json.display().to_string(),
        format_version: manifest.format_version,
        container: container_name(output_json).to_string(),
        source: manifest.source,
        parameter_count,
        sha256,
    })
}

pub fn import_pytorch_npa_checkpoint(
    input_pth: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
) -> AutomataResult<ImportReport> {
    let input_pth = input_pth.as_ref();
    let output_path = output_path.as_ref();
    let exported = load_pytorch_npa_checkpoint(input_pth)?;
    let model = NpaModel {
        config: exported.config,
        weights: NpaWeights {
            w1: exported.w1,
            b1: exported.b1,
            w2: exported.w2,
            b2: exported.b2,
        },
    };
    model.validate()?;
    let parameter_count = model.weights.w1.len()
        + model.weights.b1.len()
        + model.weights.w2.len()
        + model.weights.b2.len();
    let manifest = BpkModelManifest::from_model(&model, exported.hashgrid, exported.source.clone());
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let sha256 = save_manifest(output_path, &manifest)?;
    Ok(ImportReport {
        output: output_path.display().to_string(),
        format_version: manifest.format_version,
        container: container_name(output_path).to_string(),
        source: manifest.source,
        parameter_count,
        sha256,
    })
}

pub fn load_pytorch_npa_checkpoint(path: impl AsRef<Path>) -> AutomataResult<ExportedCheckpoint> {
    let path = path.as_ref();
    let file = fs::File::open(path)?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|err| AutomataError::InvalidFormat(err.to_string()))?;
    let byteorder = read_zip_text(&mut archive, "byteorder")?;
    if byteorder.trim() != "little" {
        return Err(AutomataError::InvalidFormat(format!(
            "unsupported PyTorch checkpoint byteorder {byteorder:?}"
        )));
    }

    let eps0 = read_zip_f32_vec(&mut archive, "data/0")?
        .first()
        .copied()
        .ok_or_else(|| AutomataError::InvalidFormat("missing eps0 tensor".to_string()))?;
    let alpha = read_zip_f32_vec(&mut archive, "data/1")?
        .first()
        .copied()
        .ok_or_else(|| AutomataError::InvalidFormat("missing alpha tensor".to_string()))?;
    let w1 = read_zip_f32_vec(&mut archive, "data/2")?;
    let b1 = read_zip_f32_vec(&mut archive, "data/3")?;
    let w2 = read_zip_f32_vec(&mut archive, "data/4")?;

    let hidden_dims = b1.len();
    if hidden_dims == 0 {
        return Err(AutomataError::InvalidFormat(
            "model.0.bias storage is empty".to_string(),
        ));
    }
    if !w1.len().is_multiple_of(hidden_dims) || !w2.len().is_multiple_of(hidden_dims) {
        return Err(AutomataError::InvalidFormat(format!(
            "weight sizes are not divisible by hidden dims {hidden_dims}"
        )));
    }
    let perception_dims = w1.len() / hidden_dims;
    let update_dims = w2.len() / hidden_dims;
    let spatial_dims = 2usize;
    if update_dims <= spatial_dims {
        return Err(AutomataError::InvalidFormat(format!(
            "update dims {update_dims} cannot fit 2D dx plus state"
        )));
    }
    let state_dims = update_dims - spatial_dims;
    let expected_perception_dims = state_dims * 2 + state_dims * spatial_dims + spatial_dims;
    if perception_dims != expected_perception_dims {
        return Err(AutomataError::InvalidFormat(format!(
            "perception dims {perception_dims} != expected NPA dims {expected_perception_dims}"
        )));
    }

    let hashgrid = inferred_hashgrid(eps0);
    let config = NpaConfig {
        spatial_dims,
        state_dims,
        hidden_dims,
        eps0,
        alpha,
        density_grad: true,
        state_grad: true,
        log_norm_grad: true,
        log_norm_density_grad: true,
        stopgrad_pos: true,
        stopgrad_state: false,
        equivariance: EquivarianceMode::ParticleDensityAndScale,
        position_features: false,
        auxiliary_input_dims: 0,
        decoder_dims: None,
        output_dims: None,
    };
    let b2 = vec![0.0; update_dims];

    Ok(ExportedCheckpoint {
        config,
        hashgrid,
        w1,
        b1,
        w2,
        b2,
        source: Some(path.display().to_string()),
    })
}

pub fn load_manifest(path: impl AsRef<Path>) -> AutomataResult<BpkModelManifest> {
    let bytes = fs::read(path)?;
    if bytes.starts_with(&BPK_MAGIC) {
        return decode_bpk_manifest(&bytes);
    }
    Ok(serde_json::from_slice(&bytes)?)
}

pub fn save_manifest(
    path: impl AsRef<Path>,
    manifest: &BpkModelManifest,
) -> AutomataResult<Option<String>> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    if is_bpk_path(path) {
        let encoded = encode_bpk_manifest(manifest)?;
        let sha256 = hex_digest(&payload_hash(manifest)?);
        fs::write(path, encoded)?;
        Ok(Some(sha256))
    } else {
        fs::write(path, serde_json::to_string_pretty(manifest)?)?;
        Ok(None)
    }
}

pub fn load_adapter_manifest(path: impl AsRef<Path>) -> AutomataResult<BpkAdapterManifest> {
    let bytes = fs::read(path)?;
    let manifest: BpkAdapterManifest = serde_json::from_slice(&bytes)?;
    manifest.adapter.validate(&manifest.config)?;
    Ok(manifest)
}

pub fn save_adapter_manifest(
    path: impl AsRef<Path>,
    manifest: &BpkAdapterManifest,
) -> AutomataResult<()> {
    manifest.adapter.validate(&manifest.config)?;
    let path = path.as_ref();
    if is_bpk_path(path) {
        return Err(AutomataError::InvalidArgument(
            "adapter manifests are JSON artifacts; use .json or .adapter.json".to_string(),
        ));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(manifest)?)?;
    Ok(())
}

pub fn parameter_count(manifest: &BpkModelManifest) -> usize {
    manifest.weights.w1.len()
        + manifest.weights.b1.len()
        + manifest.weights.w2.len()
        + manifest.weights.b2.len()
}

pub fn encode_bpk_manifest(manifest: &BpkModelManifest) -> AutomataResult<Vec<u8>> {
    let payload = serde_json::to_vec(manifest)?;
    let digest = Sha256::digest(&payload);
    let payload_len = u64::try_from(payload.len()).map_err(|_| {
        AutomataError::InvalidFormat("payload is too large for bpk header".to_string())
    })?;
    let mut out = Vec::with_capacity(BPK_HEADER_LEN + payload.len());
    out.extend_from_slice(&BPK_MAGIC);
    out.extend_from_slice(&BPK_CONTAINER_VERSION.to_le_bytes());
    out.extend_from_slice(&payload_len.to_le_bytes());
    out.extend_from_slice(&digest);
    out.extend_from_slice(&payload);
    Ok(out)
}

pub fn decode_bpk_manifest(bytes: &[u8]) -> AutomataResult<BpkModelManifest> {
    if bytes.len() < BPK_HEADER_LEN {
        return Err(AutomataError::InvalidFormat(format!(
            "bpk file is shorter than header: {} < {BPK_HEADER_LEN}",
            bytes.len()
        )));
    }
    if !bytes.starts_with(&BPK_MAGIC) {
        return Err(AutomataError::InvalidFormat(
            "missing bpk magic".to_string(),
        ));
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().expect("header version slice"));
    if version != BPK_CONTAINER_VERSION {
        return Err(AutomataError::InvalidFormat(format!(
            "unsupported bpk container version {version}"
        )));
    }
    let payload_len =
        u64::from_le_bytes(bytes[12..20].try_into().expect("payload len slice")) as usize;
    let expected_len = BPK_HEADER_LEN + payload_len;
    if bytes.len() != expected_len {
        return Err(AutomataError::InvalidFormat(format!(
            "bpk payload length mismatch: file {} != expected {expected_len}",
            bytes.len()
        )));
    }
    let expected_digest = &bytes[20..52];
    let payload = &bytes[BPK_HEADER_LEN..];
    let actual_digest = Sha256::digest(payload);
    if expected_digest != actual_digest.as_slice() {
        return Err(AutomataError::InvalidFormat(
            "bpk sha256 checksum mismatch".to_string(),
        ));
    }
    let manifest: BpkModelManifest = serde_json::from_slice(payload)?;
    manifest.clone().into_model().validate()?;
    Ok(manifest)
}

pub fn bpk_payload_sha256(bytes: &[u8]) -> AutomataResult<String> {
    if bytes.len() < BPK_HEADER_LEN || !bytes.starts_with(&BPK_MAGIC) {
        return Err(AutomataError::InvalidFormat(
            "not a bpk container".to_string(),
        ));
    }
    Ok(hex_digest(&bytes[20..52]))
}

fn payload_hash(manifest: &BpkModelManifest) -> AutomataResult<[u8; 32]> {
    let payload = serde_json::to_vec(manifest)?;
    Ok(Sha256::digest(payload).into())
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

fn is_bpk_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("bpk"))
}

fn container_name(path: &Path) -> &'static str {
    if is_bpk_path(path) { "bpk" } else { "json" }
}

fn is_pytorch_checkpoint_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("pth") || ext.eq_ignore_ascii_case("pt"))
}

fn inferred_hashgrid(eps0: f32) -> HashGridConfig {
    if (eps0 - 0.2).abs() < 1e-5 {
        HashGridConfig {
            dim: 2,
            boundary: Boundary::Periodic,
            mode: HashGridMode::Grid,
            grid_size: [10, 10, 1],
            eps: eps0,
            max_particles_per_block: 32,
        }
    } else {
        HashGridConfig {
            dim: 2,
            boundary: Boundary::Clamped,
            mode: HashGridMode::Particle,
            grid_size: [64, 64, 1],
            eps: eps0,
            max_particles_per_block: 32,
        }
    }
}

fn read_zip_text<R: Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
    suffix: &str,
) -> AutomataResult<String> {
    let mut file = zip_file_by_suffix(archive, suffix)?;
    let mut text = String::new();
    file.read_to_string(&mut text)?;
    Ok(text)
}

fn read_zip_f32_vec<R: Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
    suffix: &str,
) -> AutomataResult<Vec<f32>> {
    let mut file = zip_file_by_suffix(archive, suffix)?;
    let mut bytes = Vec::with_capacity(file.size() as usize);
    file.read_to_end(&mut bytes)?;
    if !bytes.len().is_multiple_of(4) {
        return Err(AutomataError::InvalidFormat(format!(
            "storage {suffix} byte length {} is not divisible by 4",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk")))
        .collect())
}

fn zip_file_by_suffix<'a, R: Read + std::io::Seek>(
    archive: &'a mut zip::ZipArchive<R>,
    suffix: &str,
) -> AutomataResult<zip::read::ZipFile<'a, R>> {
    let mut matches = Vec::<PathBuf>::new();
    for idx in 0..archive.len() {
        let file = archive
            .by_index(idx)
            .map_err(|err| AutomataError::InvalidFormat(err.to_string()))?;
        let name = file.name().to_string();
        if name.ends_with(suffix) {
            matches.push(PathBuf::from(name));
        }
    }
    let name = matches.into_iter().next().ok_or_else(|| {
        AutomataError::InvalidFormat(format!("missing zip entry ending with {suffix}"))
    })?;
    archive
        .by_name(name.to_string_lossy().as_ref())
        .map_err(|err| AutomataError::InvalidFormat(err.to_string()))
}
