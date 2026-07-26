//! Isolated browser worker for canonical Burn/WGPU Target2D training.

#![cfg(target_arch = "wasm32")]

use std::time::Duration;

use burn_automata::{
    AdaptiveModelArtifact, AdaptiveNpaConfig, AdaptiveNpaModel,
    AdaptiveTarget2dGpuTrainingObserver, AdaptiveTarget2dGpuTrainingProgress,
    AdaptiveTarget2dTrainingConfig, BpkModelManifest, Target2dGpuBackend,
    Target2dGpuTrainingObserver, Target2dGpuTrainingProgress, Target2dLossConfig,
    Target2dTrainingConfig, decode_target_image_2d_upstream, encode_adaptive_model,
    import::encode_bpk_manifest, load_adaptive_model_bytes,
    train_adaptive_target_2d_gpu_with_observer_async, train_target_2d_gpu_with_observer_async,
    upstream_growing_2d_hashgrid,
};
use js_sys::{Object, Reflect, Uint8Array};
use wasm_bindgen::{JsCast, JsValue, prelude::wasm_bindgen};

const TARGET_ALPHA_THRESHOLD: f32 = 0.05;
const MAX_LIVE_TRAINING_LOSS: f32 = 100.0;
const MAX_LIVE_TRAINING_GRAD_NORM: f32 = 10_000.0;

#[wasm_bindgen]
pub async fn initialize_worker_webgpu() {
    console_error_panic_hook::set_once();
    burn_automata::initialize_webgpu_backend().await;
}

#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub async fn train_target_image(
    job_id: u32,
    target_id: u32,
    mode: String,
    target_bytes: Uint8Array,
    model_bytes: Uint8Array,
    config_json: String,
    snapshot_interval_steps: u32,
    snapshot_interval_ms: u32,
) -> Result<(), JsValue> {
    let result = match mode.as_str() {
        "fixed" => {
            train_fixed(
                job_id,
                target_id,
                target_bytes.to_vec(),
                model_bytes.to_vec(),
                &config_json,
                snapshot_interval_steps as usize,
                snapshot_interval_ms,
            )
            .await
        }
        "adaptive" => {
            train_adaptive(
                job_id,
                target_id,
                target_bytes.to_vec(),
                model_bytes.to_vec(),
                &config_json,
                snapshot_interval_steps as usize,
                snapshot_interval_ms,
            )
            .await
        }
        _ => Err(format!("unsupported browser training mode {mode:?}")),
    };

    if let Err(error) = result {
        post_failure(job_id, target_id, &error)?;
    }
    Ok(())
}

async fn train_fixed(
    job_id: u32,
    target_id: u32,
    target_bytes: Vec<u8>,
    model_bytes: Vec<u8>,
    config_json: &str,
    snapshot_interval_steps: usize,
    snapshot_interval_ms: u32,
) -> Result<(), String> {
    let config: Target2dTrainingConfig =
        serde_json::from_str(config_json).map_err(|error| error.to_string())?;
    let manifest = burn_automata::import::load_manifest_bytes(&model_bytes)
        .map_err(|error| error.to_string())?;
    let hashgrid = manifest.hashgrid.clone();
    let mut model = manifest.into_model();
    let target = decode_target_image_2d_upstream(
        &target_bytes,
        TARGET_ALPHA_THRESHOLD,
        config.particle_count,
        None,
    )
    .map_err(|error| error.to_string())?;
    let mut observer = FixedObserver {
        common: WorkerObserver::new(
            job_id,
            target_id,
            snapshot_interval_steps,
            snapshot_interval_ms,
        ),
        hashgrid: hashgrid.clone(),
    };
    let report = train_target_2d_gpu_with_observer_async(
        Target2dGpuBackend::Wgpu,
        &mut model,
        &hashgrid,
        target,
        config,
        Target2dLossConfig::default(),
        None,
        &mut observer,
    )
    .await
    .map_err(|error| error.to_string())?;
    let model_bytes = encode_fixed_model(&model, hashgrid)?;
    post_finished(
        job_id,
        target_id,
        "fixed",
        model_bytes,
        serde_json::to_string(&report).map_err(|error| error.to_string())?,
    )
}

async fn train_adaptive(
    job_id: u32,
    target_id: u32,
    target_bytes: Vec<u8>,
    model_bytes: Vec<u8>,
    config_json: &str,
    snapshot_interval_steps: usize,
    snapshot_interval_ms: u32,
) -> Result<(), String> {
    let config: AdaptiveTarget2dTrainingConfig =
        serde_json::from_str(config_json).map_err(|error| error.to_string())?;
    let mut model = match load_adaptive_model_bytes(&model_bytes) {
        Ok(artifact) => artifact.model,
        Err(adaptive_error) => {
            let rule = burn_automata::import::load_manifest_bytes(&model_bytes)
                .map_err(|fixed_error| {
                    format!(
                        "browser adaptive training input is neither an adaptive artifact ({adaptive_error}) nor a fixed NPA artifact ({fixed_error})"
                    )
                })?
                .into_model();
            let mut adaptive = AdaptiveNpaConfig::growing_2d();
            adaptive.min_leaves = config.target2d.particle_count;
            adaptive.max_leaves = config
                .material
                .reference_particle_count
                .max(config.target2d.particle_count);
            adaptive.target_leaves = config.target2d.particle_count;
            adaptive.bootstrap_fine_leaves = config.material.reference_particle_count;
            adaptive.retain_bootstrap_templates = false;
            AdaptiveNpaModel::seeded(rule, adaptive, config.target2d.seed ^ 0xada2_7a2d)
                .map_err(|error| error.to_string())?
        }
    };
    let hashgrid = upstream_growing_2d_hashgrid();
    let target = decode_target_image_2d_upstream(
        &target_bytes,
        TARGET_ALPHA_THRESHOLD,
        config.material.reference_particle_count,
        None,
    )
    .map_err(|error| error.to_string())?;
    let mut observer = AdaptiveObserver {
        common: WorkerObserver::new(
            job_id,
            target_id,
            snapshot_interval_steps,
            snapshot_interval_ms,
        ),
    };
    let report = train_adaptive_target_2d_gpu_with_observer_async(
        Target2dGpuBackend::Wgpu,
        &mut model,
        &hashgrid,
        target,
        config,
        Target2dLossConfig::default(),
        None,
        &mut observer,
    )
    .await
    .map_err(|error| error.to_string())?;
    let model_bytes = encode_adaptive_snapshot(&model)?;
    post_finished(
        job_id,
        target_id,
        "adaptive",
        model_bytes,
        serde_json::to_string(&report).map_err(|error| error.to_string())?,
    )
}

struct WorkerObserver {
    job_id: u32,
    target_id: u32,
    snapshot_interval_steps: usize,
    snapshot_interval_duration: Duration,
    failure: Option<String>,
}

impl WorkerObserver {
    fn new(
        job_id: u32,
        target_id: u32,
        snapshot_interval_steps: usize,
        snapshot_interval_ms: u32,
    ) -> Self {
        Self {
            job_id,
            target_id,
            snapshot_interval_steps,
            snapshot_interval_duration: Duration::from_millis(snapshot_interval_ms.into()),
            failure: None,
        }
    }

    fn should_stop(&self) -> bool {
        self.failure.is_some()
    }

    #[allow(clippy::too_many_arguments)]
    fn publish(
        &mut self,
        mode: &str,
        step: usize,
        total_steps: usize,
        loss: f32,
        render_rgb_psnr_db: Option<f32>,
        base_grad_norm: f32,
        base_grad_scale: f32,
        particle_steps_per_sec: f64,
        model_bytes: Result<Vec<u8>, String>,
    ) {
        if let Some(reason) = unsafe_progress_reason(step, loss, base_grad_norm, base_grad_scale) {
            self.failure = Some(reason);
            return;
        }
        let result = model_bytes.and_then(|model_bytes| {
            post_progress(
                self.job_id,
                self.target_id,
                mode,
                step,
                total_steps,
                loss,
                render_rgb_psnr_db,
                base_grad_norm,
                base_grad_scale,
                particle_steps_per_sec,
                model_bytes,
            )
        });
        if let Err(error) = result {
            self.failure = Some(error);
        }
    }
}

struct FixedObserver {
    common: WorkerObserver,
    hashgrid: burn_automata::kernels::HashGridConfig,
}

impl Target2dGpuTrainingObserver for FixedObserver {
    fn should_stop(&self) -> bool {
        self.common.should_stop()
    }

    fn snapshot_interval_steps(&self) -> usize {
        self.common.snapshot_interval_steps
    }

    fn snapshot_interval_duration(&self) -> Duration {
        self.common.snapshot_interval_duration
    }

    fn on_progress(&mut self, progress: Target2dGpuTrainingProgress) {
        let model_bytes = encode_fixed_model(&progress.model, self.hashgrid.clone());
        self.common.publish(
            "fixed",
            progress.step,
            progress.total_steps,
            progress.loss,
            progress.render_rgb_psnr_db,
            progress.base_grad_norm,
            progress.base_grad_scale,
            progress.particle_steps_per_sec,
            model_bytes,
        );
    }
}

struct AdaptiveObserver {
    common: WorkerObserver,
}

impl AdaptiveTarget2dGpuTrainingObserver for AdaptiveObserver {
    fn should_stop(&self) -> bool {
        self.common.should_stop()
    }

    fn snapshot_interval_steps(&self) -> usize {
        self.common.snapshot_interval_steps
    }

    fn snapshot_interval_duration(&self) -> Duration {
        self.common.snapshot_interval_duration
    }

    fn on_progress(&mut self, progress: AdaptiveTarget2dGpuTrainingProgress) {
        let model_bytes = encode_adaptive_snapshot(&progress.model);
        self.common.publish(
            "adaptive",
            progress.step,
            progress.total_steps,
            progress.loss,
            progress.render_rgb_psnr_db,
            progress.base_grad_norm,
            progress.base_grad_scale,
            progress.particle_steps_per_sec,
            model_bytes,
        );
    }
}

fn encode_fixed_model(
    model: &burn_automata::NpaModel,
    hashgrid: burn_automata::kernels::HashGridConfig,
) -> Result<Vec<u8>, String> {
    encode_bpk_manifest(&BpkModelManifest::from_model(
        model,
        hashgrid,
        Some("browser-target2d-training".to_string()),
    ))
    .map_err(|error| error.to_string())
}

fn encode_adaptive_snapshot(model: &AdaptiveNpaModel) -> Result<Vec<u8>, String> {
    let artifact = AdaptiveModelArtifact::fresh_task_trained(
        model.clone(),
        Some("browser-adaptive-target2d-training".to_string()),
    )
    .map_err(|error| error.to_string())?;
    encode_adaptive_model(&artifact)
        .map(|(bytes, _digest)| bytes)
        .map_err(|error| error.to_string())
}

fn unsafe_progress_reason(
    step: usize,
    loss: f32,
    base_grad_norm: f32,
    base_grad_scale: f32,
) -> Option<String> {
    if !loss.is_finite() || !base_grad_norm.is_finite() || !base_grad_scale.is_finite() {
        return Some(format!(
            "training stopped before publishing a non-finite update at step {step}: loss={loss:?}, grad={base_grad_norm:?}, scale={base_grad_scale:?}"
        ));
    }
    if loss > MAX_LIVE_TRAINING_LOSS {
        return Some(format!(
            "training stopped at step {step} because loss {loss:.3} exceeded the live safety limit {MAX_LIVE_TRAINING_LOSS:.1}"
        ));
    }
    if base_grad_norm > MAX_LIVE_TRAINING_GRAD_NORM {
        return Some(format!(
            "training stopped at step {step} because gradient norm {base_grad_norm:.3} exceeded the live safety limit {MAX_LIVE_TRAINING_GRAD_NORM:.1}"
        ));
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn post_progress(
    job_id: u32,
    target_id: u32,
    mode: &str,
    step: usize,
    total_steps: usize,
    loss: f32,
    render_rgb_psnr_db: Option<f32>,
    base_grad_norm: f32,
    base_grad_scale: f32,
    particle_steps_per_sec: f64,
    model_bytes: Vec<u8>,
) -> Result<(), String> {
    let message = base_message("progress", job_id, target_id)?;
    set(&message, "mode", JsValue::from_str(mode))?;
    set(&message, "step", JsValue::from_f64(step as f64))?;
    set(
        &message,
        "totalSteps",
        JsValue::from_f64(total_steps as f64),
    )?;
    set(&message, "loss", JsValue::from_f64(loss.into()))?;
    set(
        &message,
        "psnr",
        render_rgb_psnr_db.map_or(JsValue::NULL, |value| JsValue::from_f64(value.into())),
    )?;
    set(
        &message,
        "gradNorm",
        JsValue::from_f64(base_grad_norm.into()),
    )?;
    set(
        &message,
        "gradScale",
        JsValue::from_f64(base_grad_scale.into()),
    )?;
    set(
        &message,
        "particleStepsPerSec",
        JsValue::from_f64(particle_steps_per_sec),
    )?;
    set_model_bytes(&message, &model_bytes)?;
    post(message)
}

fn post_finished(
    job_id: u32,
    target_id: u32,
    mode: &str,
    model_bytes: Vec<u8>,
    report_json: String,
) -> Result<(), String> {
    let message = base_message("finished", job_id, target_id)?;
    set(&message, "mode", JsValue::from_str(mode))?;
    set(&message, "reportJson", JsValue::from_str(&report_json))?;
    set_model_bytes(&message, &model_bytes)?;
    post(message)
}

fn post_failure(job_id: u32, target_id: u32, error: &str) -> Result<(), JsValue> {
    let message =
        base_message("failed", job_id, target_id).map_err(|error| JsValue::from_str(&error))?;
    set(&message, "error", JsValue::from_str(error)).map_err(|error| JsValue::from_str(&error))?;
    post(message).map_err(|error| JsValue::from_str(&error))
}

fn base_message(kind: &str, job_id: u32, target_id: u32) -> Result<Object, String> {
    let message = Object::new();
    set(&message, "type", JsValue::from_str(kind))?;
    set(&message, "jobId", JsValue::from_f64(job_id.into()))?;
    set(&message, "targetId", JsValue::from_f64(target_id.into()))?;
    Ok(message)
}

fn set_model_bytes(message: &Object, bytes: &[u8]) -> Result<(), String> {
    set(
        message,
        "modelBytes",
        Uint8Array::from(bytes).unchecked_into(),
    )
}

fn set(message: &Object, key: &str, value: JsValue) -> Result<(), String> {
    Reflect::set(message, &JsValue::from_str(key), &value)
        .map(|_| ())
        .map_err(|error| format!("failed to build worker message {key}: {error:?}"))
}

fn post(message: Object) -> Result<(), String> {
    js_sys::global()
        .unchecked_into::<web_sys::DedicatedWorkerGlobalScope>()
        .post_message(&message)
        .map_err(|error| format!("failed to publish browser training update: {error:?}"))
}
