use std::path::{Path, PathBuf};

use bevy::{tasks::AsyncComputeTaskPool, window::FileDragAndDrop};
use burn_automata::{
    ConditionEncoder2d, ConditionImage2d, DinoVitsConditionEncoder, NpaConfig, NpaModel,
    generate_e2e_conditioned_npa_2d, import::load_manifest, load_e2e_hyper_npa_2d,
};
use crossbeam_channel::{Receiver, Sender, unbounded};
use image::GenericImageView;

use super::*;

#[derive(Message, Clone, Debug, Default)]
pub(in crate::viewer) struct OpenHyperNpaImage;

#[derive(Resource)]
pub(in crate::viewer) struct HyperNpaImageDialogChannel {
    sender: Sender<Result<HyperNpaImageSource, String>>,
    receiver: Receiver<Result<HyperNpaImageSource, String>>,
}

impl Default for HyperNpaImageDialogChannel {
    fn default() -> Self {
        let (sender, receiver) = unbounded();
        Self { sender, receiver }
    }
}

#[derive(Resource)]
pub(in crate::viewer) struct HyperNpaInferenceChannel {
    sender: Sender<HyperNpaInferenceResult>,
    receiver: Receiver<HyperNpaInferenceResult>,
}

impl Default for HyperNpaInferenceChannel {
    fn default() -> Self {
        let (sender, receiver) = unbounded();
        Self { sender, receiver }
    }
}

#[derive(Resource, Clone, Debug, Default)]
pub(in crate::viewer) struct HyperNpaInferenceState {
    pub(in crate::viewer) pending: usize,
}

#[derive(Clone, Debug)]
struct HyperNpaImageSource {
    file_name: String,
    path: Option<PathBuf>,
    bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
struct HyperNpaInferenceRequest {
    source: HyperNpaImageSource,
    base_model_path: PathBuf,
    hyper_model_path: PathBuf,
    dino_model_path: PathBuf,
    dino_image_size: usize,
    dino_patch_size: usize,
}

#[derive(Clone, Debug)]
struct HyperNpaInferenceResult {
    file_name: String,
    result: Result<GeneratedHyperNpaModel, String>,
}

#[derive(Clone, Debug)]
pub(in crate::viewer) struct GeneratedHyperNpaModel {
    pub(in crate::viewer) model: NpaModel,
    pub(in crate::viewer) hashgrid: HashGridConfig,
    pub(in crate::viewer) image_width: u32,
    pub(in crate::viewer) image_height: u32,
    pub(in crate::viewer) adapter_rank: usize,
    pub(in crate::viewer) adapter_alpha: f32,
    pub(in crate::viewer) token_count: usize,
    pub(in crate::viewer) embed_dims: usize,
}

pub(in crate::viewer) fn handle_open_hyper_npa_image_dialog(
    mut requests: MessageReader<OpenHyperNpaImage>,
    channel: Res<HyperNpaImageDialogChannel>,
    mut runtime: ResMut<AutomataRuntime>,
) {
    for _request in requests.read() {
        let sender = channel.sender.clone();
        runtime.status = "opening image for HyperNPA inference".to_string();
        AsyncComputeTaskPool::get()
            .spawn(async move {
                let picked = rfd::AsyncFileDialog::new()
                    .add_filter("image", &["png", "jpg", "jpeg"])
                    .pick_file()
                    .await;
                let result = match picked {
                    Some(file) => {
                        let file_name = file.file_name();
                        #[cfg(not(target_arch = "wasm32"))]
                        let path = Some(file.path().to_path_buf());
                        #[cfg(target_arch = "wasm32")]
                        let path = None;
                        let bytes = file.read().await;
                        Ok(HyperNpaImageSource {
                            file_name,
                            path,
                            bytes,
                        })
                    }
                    None => Err("image selection cancelled".to_string()),
                };
                let _ = sender.send(result);
            })
            .detach();
    }
}

pub(in crate::viewer) fn handle_hyper_npa_image_drop(
    mut drops: MessageReader<FileDragAndDrop>,
    channel: Res<HyperNpaImageDialogChannel>,
    mut runtime: ResMut<AutomataRuntime>,
) {
    for event in drops.read() {
        let FileDragAndDrop::DroppedFile { path_buf, .. } = event else {
            continue;
        };
        let sender = channel.sender.clone();
        let path = path_buf.clone();
        runtime.status = format!(
            "loading dropped image {}",
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("image")
        );
        AsyncComputeTaskPool::get()
            .spawn(async move {
                let result = read_image_source_from_path(&path);
                let _ = sender.send(result);
            })
            .detach();
    }
}

pub(in crate::viewer) fn poll_hyper_npa_image_sources(
    source_channel: Res<HyperNpaImageDialogChannel>,
    inference_channel: Res<HyperNpaInferenceChannel>,
    mut inference_state: ResMut<HyperNpaInferenceState>,
    settings: Res<AutomataSettings>,
    mut runtime: ResMut<AutomataRuntime>,
) {
    for source in source_channel.receiver.try_iter() {
        match source {
            Ok(source) => {
                let origin = source
                    .path
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| source.file_name.clone());
                let request = match build_inference_request(source, &settings) {
                    Ok(request) => request,
                    Err(err) => {
                        runtime.status = err;
                        continue;
                    }
                };
                let file_name = request.source.file_name.clone();
                let sender = inference_channel.sender.clone();
                inference_state.pending = inference_state.pending.saturating_add(1);
                runtime.status =
                    format!("running DINO -> HyperNPA inference for {file_name} ({origin})");
                AsyncComputeTaskPool::get()
                    .spawn(async move {
                        let result = run_hyper_npa_inference(request);
                        let _ = sender.send(result);
                    })
                    .detach();
            }
            Err(err) => {
                runtime.status = format!("HyperNPA image load skipped: {err}");
            }
        }
    }
}

pub(in crate::viewer) fn poll_hyper_npa_inference_results(
    inference_channel: Res<HyperNpaInferenceChannel>,
    mut inference_state: ResMut<HyperNpaInferenceState>,
    mut settings: ResMut<AutomataSettings>,
    mut runtime: ResMut<AutomataRuntime>,
) {
    for result in inference_channel.receiver.try_iter() {
        inference_state.pending = inference_state.pending.saturating_sub(1);
        match result.result {
            Ok(generated) => {
                let label = format!("hyper {}", result.file_name);
                settings.model_path = None;
                settings.generated_model_label = Some(label);
                settings.preset = AutomataPreset::Growing2d;
                settings.seed = RolloutConfig::default().seed;
                settings.seed_mode = ParticleSeed::UniformCircle;
                settings.seed_scale = NpaConfig::seed_scale_for_preset(AutomataPreset::Growing2d);
                settings.reference_seed_scale = settings.seed_scale;
                settings.particle_count = settings.particle_count.max(2048);
                settings.mark_changed();

                runtime.model = generated.model;
                runtime.hashgrid = generated.hashgrid;
                runtime.loaded_model_path = None;
                runtime.loaded_preset = None;
                runtime.trace = None;
                runtime.frame = 0;
                runtime.backward_loss = None;
                runtime.backward_grad_norm = None;
                reset_training_stats(&mut runtime);
                runtime.model_revision = runtime.model_revision.wrapping_add(1);
                runtime.status = format!(
                    "generated HyperNPA for {} | image {}x{} | LoRA r{} a{:.1} | {} tokens x {} dims",
                    result.file_name,
                    generated.image_width,
                    generated.image_height,
                    generated.adapter_rank,
                    generated.adapter_alpha,
                    generated.token_count,
                    generated.embed_dims
                );
            }
            Err(err) => {
                runtime.status =
                    format!("HyperNPA inference failed for {}: {err}", result.file_name);
            }
        }
    }
}

fn build_inference_request(
    source: HyperNpaImageSource,
    settings: &AutomataSettings,
) -> Result<HyperNpaInferenceRequest, String> {
    let base_model_path = required_existing_path(
        settings.hyper_base_model_path.as_deref(),
        "BURN_AUTOMATA_HYPER_E2E_BASE",
        "shared base model",
    )?;
    let hyper_model_path = required_existing_path(
        settings.hyper_model_path.as_deref(),
        "BURN_AUTOMATA_HYPER_E2E_MODEL",
        "E2E hypernet model",
    )?;
    let dino_model_path = required_existing_path(
        settings.hyper_dino_model_path.as_deref(),
        "BURN_AUTOMATA_DINO_MODEL",
        "DINO model",
    )?;
    Ok(HyperNpaInferenceRequest {
        source,
        base_model_path,
        hyper_model_path,
        dino_model_path,
        dino_image_size: settings.hyper_dino_image_size.max(1),
        dino_patch_size: settings.hyper_dino_patch_size.max(1),
    })
}

fn required_existing_path(
    value: Option<&str>,
    env_key: &str,
    label: &str,
) -> Result<PathBuf, String> {
    let Some(value) = value else {
        return Err(format!(
            "missing {label}; set {env_key} or place the default artifact in the workspace"
        ));
    };
    resolve_workspace_path(value).ok_or_else(|| format!("missing {label} at {value}"))
}

fn run_hyper_npa_inference(request: HyperNpaInferenceRequest) -> HyperNpaInferenceResult {
    let file_name = request.source.file_name.clone();
    let result = generate_model_for_source(request).map_err(|err| err.to_string());
    HyperNpaInferenceResult { file_name, result }
}

pub(in crate::viewer) fn generate_hyper_npa_model_from_image_path(
    path: &Path,
    settings: &AutomataSettings,
) -> Result<GeneratedHyperNpaModel, Box<dyn std::error::Error>> {
    let source = read_image_source_from_path(path).map_err(std::io::Error::other)?;
    let request = build_inference_request(source, settings).map_err(std::io::Error::other)?;
    generate_model_for_source(request)
}

fn generate_model_for_source(
    request: HyperNpaInferenceRequest,
) -> Result<GeneratedHyperNpaModel, Box<dyn std::error::Error>> {
    if !request
        .dino_image_size
        .is_multiple_of(request.dino_patch_size)
    {
        return Err(std::io::Error::other(format!(
            "DINO image size {} must be divisible by patch size {}",
            request.dino_image_size, request.dino_patch_size
        ))
        .into());
    }
    let condition = decode_condition_image(&request.source.bytes)?;
    let image_width = condition.width as u32;
    let image_height = condition.height as u32;
    let dino = DinoVitsConditionEncoder::load(&request.dino_model_path, request.dino_image_size)?;
    let token_grid = request.dino_image_size / request.dino_patch_size;
    let mut encoded = dino.encode_batch(
        &[condition],
        ConditionEncoder2d::DinoVitsTokenGrid,
        token_grid,
        token_grid,
    )?;
    let condition_tokens = encoded
        .pop()
        .ok_or_else(|| std::io::Error::other("DINO did not return condition tokens"))?;
    let manifest = load_manifest(&request.base_model_path)?;
    let hashgrid = manifest.hashgrid.clone();
    let base_model = manifest.into_model();
    let hyper = load_e2e_hyper_npa_2d(&request.hyper_model_path)?;
    let embed_dims = hyper.embed_dims()?;
    let token_count = condition_tokens.len() / embed_dims;
    let spec = hyper.adapter_spec(&base_model.config)?;
    let conditioned = generate_e2e_conditioned_npa_2d(&base_model, &hyper, &condition_tokens)?;
    Ok(GeneratedHyperNpaModel {
        model: conditioned.model,
        hashgrid,
        image_width,
        image_height,
        adapter_rank: spec.rank,
        adapter_alpha: spec.alpha,
        token_count,
        embed_dims,
    })
}

fn decode_condition_image(bytes: &[u8]) -> Result<ConditionImage2d, Box<dyn std::error::Error>> {
    let image = image::load_from_memory(bytes)?;
    let (width, height) = image.dimensions();
    let rgb = image.to_rgb8();
    let values = rgb
        .pixels()
        .flat_map(|pixel| pixel.0.map(|value| value as f32 / 255.0))
        .collect::<Vec<_>>();
    Ok(ConditionImage2d::from_rgb(
        width as usize,
        height as usize,
        values,
    )?)
}

fn read_image_source_from_path(path: &Path) -> Result<HyperNpaImageSource, String> {
    let bytes = std::fs::read(path)
        .map_err(|err| format!("failed to read dropped image {}: {err}", path.display()))?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("image")
        .to_string();
    Ok(HyperNpaImageSource {
        file_name,
        path: Some(path.to_path_buf()),
        bytes,
    })
}
