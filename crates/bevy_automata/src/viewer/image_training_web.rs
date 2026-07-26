use burn_automata::{
    AdaptiveModelArtifact, BpkModelManifest, encode_adaptive_model,
    import::{encode_bpk_manifest, load_manifest_bytes},
    load_adaptive_model_bytes,
};
use js_sys::{Object, Reflect, Uint8Array};
use wasm_bindgen::{JsCast, JsValue, closure::Closure};
use web_sys::{ErrorEvent, MessageEvent, Worker, WorkerOptions, WorkerType};

use super::*;

const TRAINING_WORKER_URL: &str = "./training_worker.js";

#[derive(Default)]
pub(in crate::viewer) struct BrowserTrainingWorker {
    worker: Option<Worker>,
    active_job_id: Option<u64>,
    active_target_id: Option<u64>,
    on_message: Option<Closure<dyn FnMut(MessageEvent)>>,
    on_error: Option<Closure<dyn FnMut(ErrorEvent)>>,
}

impl BrowserTrainingWorker {
    pub(super) fn stop(&mut self) {
        if let Some(worker) = self.worker.take() {
            worker.terminate();
            worker.set_onmessage(None);
            worker.set_onerror(None);
        }
        self.active_job_id = None;
        self.active_target_id = None;
        self.on_message = None;
        self.on_error = None;
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn start(
        &mut self,
        channel: &ImageTargetTrainingChannel,
        job_id: u64,
        target_id: u64,
        target_bytes: &[u8],
        model: ImageTargetTrainingModel,
        hashgrid: burn_automata::kernels::HashGridConfig,
        fixed_config: &Target2dTrainingConfig,
        adaptive_config: &AdaptiveTarget2dTrainingConfig,
        snapshot_interval_steps: usize,
        snapshot_interval_duration: Duration,
    ) -> Result<(), String> {
        self.stop();

        let job_id = u32::try_from(job_id)
            .map_err(|_| "browser training job identifier overflowed u32".to_string())?;
        let target_id = u32::try_from(target_id)
            .map_err(|_| "browser target identifier overflowed u32".to_string())?;
        let (mode, model_bytes, config_json) = match model {
            ImageTargetTrainingModel::Fixed(model) => (
                "fixed",
                encode_bpk_manifest(&BpkModelManifest::from_model(
                    &model,
                    hashgrid,
                    Some("browser-target2d-training-input".to_string()),
                ))
                .map_err(|error| error.to_string())?,
                serde_json::to_string(fixed_config).map_err(|error| error.to_string())?,
            ),
            ImageTargetTrainingModel::Adaptive(model) => {
                let artifact = AdaptiveModelArtifact::fresh_task_trained(
                    *model,
                    Some("browser-adaptive-target2d-training-input".to_string()),
                )
                .map_err(|error| error.to_string())?;
                (
                    "adaptive",
                    encode_adaptive_model(&artifact)
                        .map_err(|error| error.to_string())?
                        .0,
                    serde_json::to_string(adaptive_config).map_err(|error| error.to_string())?,
                )
            }
        };

        let options = WorkerOptions::new();
        options.set_type(WorkerType::Module);
        let worker = Worker::new_with_options(TRAINING_WORKER_URL, &options)
            .map_err(|error| format!("failed to create browser training worker: {error:?}"))?;

        let sender = channel.sender.clone();
        let on_message = Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
            let result = parse_worker_event(event.data());
            let training_event = match result {
                Ok(event) => event,
                Err(error) => ImageTargetTrainingEvent::Finished {
                    job_id: u64::from(job_id),
                    target_id: u64::from(target_id),
                    result: Err(error),
                },
            };
            let _ = sender.send(training_event);
        });
        worker.set_onmessage(Some(on_message.as_ref().unchecked_ref()));

        let sender = channel.sender.clone();
        let on_error = Closure::<dyn FnMut(ErrorEvent)>::new(move |event: ErrorEvent| {
            let error = format!(
                "browser training worker failed at {}:{}: {}",
                event.filename(),
                event.lineno(),
                event.message()
            );
            let _ = sender.send(ImageTargetTrainingEvent::Finished {
                job_id: u64::from(job_id),
                target_id: u64::from(target_id),
                result: Err(error),
            });
        });
        worker.set_onerror(Some(on_error.as_ref().unchecked_ref()));

        let request = Object::new();
        set(&request, "type", JsValue::from_str("train"))?;
        set(&request, "jobId", JsValue::from_f64(job_id.into()))?;
        set(&request, "targetId", JsValue::from_f64(target_id.into()))?;
        set(&request, "mode", JsValue::from_str(mode))?;
        set(
            &request,
            "targetBytes",
            Uint8Array::from(target_bytes).unchecked_into(),
        )?;
        set(
            &request,
            "modelBytes",
            Uint8Array::from(model_bytes.as_slice()).unchecked_into(),
        )?;
        set(&request, "configJson", JsValue::from_str(&config_json))?;
        set(
            &request,
            "snapshotIntervalSteps",
            JsValue::from_f64(snapshot_interval_steps as f64),
        )?;
        set(
            &request,
            "snapshotIntervalMs",
            JsValue::from_f64(snapshot_interval_duration.as_millis() as f64),
        )?;
        worker
            .post_message(&request)
            .map_err(|error| format!("failed to start browser training: {error:?}"))?;

        self.worker = Some(worker);
        self.active_job_id = Some(u64::from(job_id));
        self.active_target_id = Some(u64::from(target_id));
        self.on_message = Some(on_message);
        self.on_error = Some(on_error);
        Ok(())
    }
}

pub(in crate::viewer) fn stop_stale_browser_training(
    state: Res<ImageTargetTrainingState>,
    mut worker: NonSendMut<BrowserTrainingWorker>,
) {
    let target_id = state.target.as_ref().map(|target| target.id);
    if worker.active_job_id.is_some()
        && (worker.active_job_id != state.active_job_id || worker.active_target_id != target_id)
    {
        worker.stop();
    }
}

fn parse_worker_event(value: JsValue) -> Result<ImageTargetTrainingEvent, String> {
    let kind = string_field(&value, "type")?;
    let job_id = integer_field(&value, "jobId")?;
    let target_id = integer_field(&value, "targetId")?;
    match kind.as_str() {
        "progress" => Ok(ImageTargetTrainingEvent::Progress {
            job_id,
            target_id,
            progress: ImageTargetTrainingProgress {
                step: integer_field(&value, "step")? as usize,
                total_steps: integer_field(&value, "totalSteps")? as usize,
                loss: number_field(&value, "loss")? as f32,
                render_rgb_psnr_db: optional_number_field(&value, "psnr")?
                    .map(|value| value as f32),
                base_grad_norm: number_field(&value, "gradNorm")? as f32,
                base_grad_scale: number_field(&value, "gradScale")? as f32,
                particle_steps_per_sec: number_field(&value, "particleStepsPerSec")?,
                model: decode_model(&value)?,
            },
        }),
        "finished" => {
            let mode = string_field(&value, "mode")?;
            let report_json = string_field(&value, "reportJson")?;
            let report = match mode.as_str() {
                "fixed" => ImageTargetTrainingReport::Fixed(
                    serde_json::from_str::<Target2dGpuTrainingReport>(&report_json)
                        .map_err(|error| error.to_string())?,
                ),
                "adaptive" => ImageTargetTrainingReport::Adaptive(
                    serde_json::from_str::<AdaptiveTarget2dGpuTrainingReport>(&report_json)
                        .map_err(|error| error.to_string())?,
                ),
                _ => return Err(format!("browser worker returned unsupported mode {mode:?}")),
            };
            Ok(ImageTargetTrainingEvent::Finished {
                job_id,
                target_id,
                result: Ok(ImageTargetTrainingCompletion {
                    model: decode_model(&value)?,
                    report,
                }),
            })
        }
        "failed" => Ok(ImageTargetTrainingEvent::Finished {
            job_id,
            target_id,
            result: Err(string_field(&value, "error")?),
        }),
        _ => Err(format!("browser training worker returned event {kind:?}")),
    }
}

fn decode_model(value: &JsValue) -> Result<ImageTargetTrainingModel, String> {
    let mode = string_field(value, "mode")?;
    let bytes = Uint8Array::new(&field(value, "modelBytes")?).to_vec();
    match mode.as_str() {
        "fixed" => Ok(ImageTargetTrainingModel::Fixed(
            load_manifest_bytes(&bytes)
                .map_err(|error| error.to_string())?
                .into_model(),
        )),
        "adaptive" => Ok(ImageTargetTrainingModel::Adaptive(Box::new(
            load_adaptive_model_bytes(&bytes)
                .map_err(|error| error.to_string())?
                .model,
        ))),
        _ => Err(format!("browser worker returned unsupported mode {mode:?}")),
    }
}

fn field(value: &JsValue, name: &str) -> Result<JsValue, String> {
    Reflect::get(value, &JsValue::from_str(name))
        .map_err(|error| format!("browser worker event field {name} failed: {error:?}"))
}

fn string_field(value: &JsValue, name: &str) -> Result<String, String> {
    field(value, name)?
        .as_string()
        .ok_or_else(|| format!("browser worker event field {name} is not a string"))
}

fn number_field(value: &JsValue, name: &str) -> Result<f64, String> {
    field(value, name)?
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("browser worker event field {name} is not finite"))
}

fn optional_number_field(value: &JsValue, name: &str) -> Result<Option<f64>, String> {
    let value = field(value, name)?;
    if value.is_null() || value.is_undefined() {
        Ok(None)
    } else {
        value
            .as_f64()
            .filter(|value| value.is_finite())
            .map(Some)
            .ok_or_else(|| format!("browser worker event field {name} is not finite"))
    }
}

fn integer_field(value: &JsValue, name: &str) -> Result<u64, String> {
    let value = number_field(value, name)?;
    if value.fract() != 0.0 || value < 0.0 || value > u32::MAX as f64 {
        return Err(format!(
            "browser worker event field {name} is outside the supported integer range"
        ));
    }
    Ok(value as u64)
}

fn set(target: &Object, name: &str, value: JsValue) -> Result<(), String> {
    Reflect::set(target, &JsValue::from_str(name), &value)
        .map(|_| ())
        .map_err(|error| format!("failed to set browser training request {name}: {error:?}"))
}
