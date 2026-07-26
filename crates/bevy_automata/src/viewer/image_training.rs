#[cfg(not(target_arch = "wasm32"))]
use std::sync::Mutex;
#[cfg(not(target_arch = "wasm32"))]
use std::thread;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use bevy::{
    asset::RenderAssetUsages,
    ecs::system::SystemParam,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use burn_automata::{
    AdaptiveNpaConfig, AdaptiveNpaModel, AdaptiveTarget2dGpuTrainingReport,
    AdaptiveTarget2dRuleTraining, AdaptiveTarget2dTrainingConfig, Target2dGpuBackend,
    Target2dGpuTrainingReport, Target2dTrainingConfig, upstream_growing_2d_hashgrid,
    upstream_growing_2d_model,
};
#[cfg(not(target_arch = "wasm32"))]
use burn_automata::{
    AdaptiveTarget2dGpuTrainingObserver, AdaptiveTarget2dGpuTrainingProgress,
    Target2dGpuTrainingObserver, Target2dGpuTrainingProgress, Target2dLossConfig,
    decode_target_image_2d_upstream, train_adaptive_target_2d_gpu_with_observer,
    train_target_2d_gpu_with_observer,
};
use crossbeam_channel::{Receiver, Sender, unbounded};

use super::*;

#[cfg(target_arch = "wasm32")]
#[path = "image_training_web.rs"]
mod web;
#[cfg(target_arch = "wasm32")]
pub(in crate::viewer) use web::{BrowserTrainingWorker, stop_stale_browser_training};

const LIVE_MODEL_SNAPSHOT_INTERVAL: Duration = Duration::from_secs(4);
const LIVE_MODEL_SNAPSHOT_FALLBACK_STEPS: usize = 100;
const LIVE_TRAINING_REPORT_INTERVAL: usize = 1_000;
#[cfg(any(target_arch = "wasm32", test))]
const BROWSER_TRAINING_SESSION_STEPS: usize = 1_000;
#[cfg(any(target_arch = "wasm32", test))]
const BROWSER_TRAINING_PARTICLES: usize = 256;
#[cfg(any(target_arch = "wasm32", test))]
const BROWSER_ADAPTIVE_ACTIVE_PARTICLES: usize = 192;
#[cfg(not(target_arch = "wasm32"))]
const TARGET_ALPHA_THRESHOLD: f32 = 0.05;
#[cfg(not(target_arch = "wasm32"))]
const MAX_LIVE_TRAINING_LOSS: f32 = 100.0;
#[cfg(not(target_arch = "wasm32"))]
const MAX_LIVE_TRAINING_GRAD_NORM: f32 = 10_000.0;

#[derive(Message, Clone, Copy, Debug, Default)]
pub(in crate::viewer) struct ToggleImageTargetTraining;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::viewer) enum ImageTargetTrainingPhase {
    #[default]
    Empty,
    Ready,
    Running,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    Stopping,
    Complete,
    Failed,
}

#[derive(Clone, Debug)]
struct ImageTarget {
    id: u64,
    source: HyperNpaImageSource,
}

#[derive(Resource, Debug, Default)]
pub(in crate::viewer) struct ImageTargetTrainingState {
    target: Option<ImageTarget>,
    pub(in crate::viewer) phase: ImageTargetTrainingPhase,
    pub(in crate::viewer) step: usize,
    pub(in crate::viewer) total_steps: usize,
    pub(in crate::viewer) loss: Option<f32>,
    pub(in crate::viewer) best_loss: Option<f32>,
    pub(in crate::viewer) grad_norm: Option<f32>,
    pub(in crate::viewer) render_rgb_psnr_db: Option<f32>,
    pub(in crate::viewer) particle_steps_per_sec: Option<f64>,
    pub(in crate::viewer) error: Option<String>,
    target_revision: u64,
    next_target_id: u64,
    next_job_id: u64,
    active_job_id: Option<u64>,
    cancel: Option<Arc<AtomicBool>>,
    training_initialized: bool,
    initialized_mode: Option<ImageTargetTrainingMode>,
    last_rollout_reset_step: usize,
}

impl ImageTargetTrainingState {
    pub(in crate::viewer) fn has_target(&self) -> bool {
        self.target.is_some()
    }

    pub(in crate::viewer) fn is_training(&self) -> bool {
        matches!(
            self.phase,
            ImageTargetTrainingPhase::Running | ImageTargetTrainingPhase::Stopping
        )
    }

    pub(in crate::viewer) fn train_action_label(&self) -> &'static str {
        match self.phase {
            ImageTargetTrainingPhase::Running => "stop",
            ImageTargetTrainingPhase::Stopping => "stopping",
            ImageTargetTrainingPhase::Empty
            | ImageTargetTrainingPhase::Ready
            | ImageTargetTrainingPhase::Failed => {
                if self.training_initialized {
                    "continue"
                } else {
                    "train fresh"
                }
            }
            ImageTargetTrainingPhase::Complete => "continue",
        }
    }

    pub(in crate::viewer) fn train_action_available(&self) -> bool {
        matches!(
            self.phase,
            ImageTargetTrainingPhase::Ready
                | ImageTargetTrainingPhase::Running
                | ImageTargetTrainingPhase::Complete
                | ImageTargetTrainingPhase::Failed
        )
    }

    pub(in crate::viewer) fn set_source(&mut self, source: &HyperNpaImageSource) -> u64 {
        self.stop_active_job();
        self.next_target_id = self.next_target_id.wrapping_add(1).max(1);
        let id = self.next_target_id;
        self.target = Some(ImageTarget {
            id,
            source: source.clone(),
        });
        self.phase = ImageTargetTrainingPhase::Ready;
        self.step = 0;
        self.total_steps = 0;
        self.loss = None;
        self.best_loss = None;
        self.grad_norm = None;
        self.render_rgb_psnr_db = None;
        self.particle_steps_per_sec = None;
        self.error = None;
        self.training_initialized = false;
        self.initialized_mode = None;
        self.last_rollout_reset_step = 0;
        self.target_revision = self.target_revision.wrapping_add(1);
        id
    }

    pub(in crate::viewer) fn clear_target(&mut self) {
        self.stop_active_job();
        self.target = None;
        self.phase = ImageTargetTrainingPhase::Empty;
        self.training_initialized = false;
        self.initialized_mode = None;
        self.last_rollout_reset_step = 0;
        self.target_revision = self.target_revision.wrapping_add(1);
    }

    pub(in crate::viewer) fn inference_source(&self) -> Option<(u64, HyperNpaImageSource)> {
        self.target
            .as_ref()
            .map(|target| (target.id, target.source.clone()))
    }

    pub(in crate::viewer) fn mark_inference_applied(&mut self, target_id: u64) {
        if self.target.as_ref().map(|target| target.id) == Some(target_id) {
            self.phase = ImageTargetTrainingPhase::Ready;
            self.step = 0;
            self.total_steps = 0;
            self.loss = None;
            self.best_loss = None;
            self.grad_norm = None;
            self.render_rgb_psnr_db = None;
            self.particle_steps_per_sec = None;
            self.error = None;
            self.training_initialized = false;
            self.initialized_mode = None;
            self.last_rollout_reset_step = 0;
        }
    }

    pub(in crate::viewer) fn mark_inference_failed(&mut self, target_id: u64, error: String) {
        if self.target.as_ref().map(|target| target.id) == Some(target_id) {
            self.error = Some(error);
        }
    }

    fn stop_active_job(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            cancel.store(true, Ordering::Release);
        }
        self.active_job_id = None;
    }
}

#[derive(Resource, Default)]
pub(in crate::viewer) struct ImageTargetPreviewState {
    handle: Option<Handle<Image>>,
    revision: u64,
}

#[derive(SystemParam)]
pub(in crate::viewer) struct ImageTargetInteraction<'w> {
    target_training: ResMut<'w, ImageTargetTrainingState>,
    inference: ResMut<'w, HyperNpaInferenceState>,
}

impl ImageTargetInteraction<'_> {
    pub(in crate::viewer) fn clear_current_target(&mut self) {
        self.target_training.clear_target();
        self.inference.cancel_current();
    }
}

#[derive(Resource)]
pub(in crate::viewer) struct ImageTargetTrainingChannel {
    sender: Sender<ImageTargetTrainingEvent>,
    receiver: Receiver<ImageTargetTrainingEvent>,
}

impl Default for ImageTargetTrainingChannel {
    fn default() -> Self {
        let (sender, receiver) = unbounded();
        Self { sender, receiver }
    }
}

enum ImageTargetTrainingEvent {
    Progress {
        job_id: u64,
        target_id: u64,
        progress: ImageTargetTrainingProgress,
    },
    Finished {
        job_id: u64,
        target_id: u64,
        result: Result<ImageTargetTrainingCompletion, String>,
    },
}

struct ImageTargetTrainingCompletion {
    model: ImageTargetTrainingModel,
    report: ImageTargetTrainingReport,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ImageTargetTrainingMode {
    Fixed,
    Adaptive,
}

impl ImageTargetTrainingMode {
    fn from_settings(settings: &AutomataSettings) -> Self {
        if settings.adaptive_training_enabled {
            Self::Adaptive
        } else {
            Self::Fixed
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Adaptive => "adaptive",
        }
    }
}

enum ImageTargetTrainingModel {
    Fixed(NpaModel),
    Adaptive(Box<AdaptiveNpaModel>),
}

enum ImageTargetTrainingReport {
    Fixed(Target2dGpuTrainingReport),
    Adaptive(AdaptiveTarget2dGpuTrainingReport),
}

impl ImageTargetTrainingReport {
    fn training(&self) -> &Target2dGpuTrainingReport {
        match self {
            Self::Fixed(report) => report,
            Self::Adaptive(report) => &report.training,
        }
    }
}

struct ImageTargetTrainingProgress {
    step: usize,
    total_steps: usize,
    loss: f32,
    render_rgb_psnr_db: Option<f32>,
    base_grad_norm: f32,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    base_grad_scale: f32,
    particle_steps_per_sec: f64,
    model: ImageTargetTrainingModel,
}

#[cfg(not(target_arch = "wasm32"))]
struct ViewerTrainingObserver {
    job_id: u64,
    target_id: u64,
    cancel: Arc<AtomicBool>,
    safety_failure: Arc<Mutex<Option<String>>>,
    sender: Sender<ImageTargetTrainingEvent>,
    snapshot_interval_steps: usize,
    snapshot_interval_duration: Duration,
}

#[cfg(not(target_arch = "wasm32"))]
impl Target2dGpuTrainingObserver for ViewerTrainingObserver {
    fn should_stop(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }

    fn snapshot_interval_steps(&self) -> usize {
        self.snapshot_interval_steps
    }

    fn snapshot_interval_duration(&self) -> Duration {
        self.snapshot_interval_duration
    }

    fn on_progress(&mut self, progress: Target2dGpuTrainingProgress) {
        self.publish_progress(ImageTargetTrainingProgress {
            step: progress.step,
            total_steps: progress.total_steps,
            loss: progress.loss,
            render_rgb_psnr_db: progress.render_rgb_psnr_db,
            base_grad_norm: progress.base_grad_norm,
            base_grad_scale: progress.base_grad_scale,
            particle_steps_per_sec: progress.particle_steps_per_sec,
            model: ImageTargetTrainingModel::Fixed(progress.model),
        });
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl AdaptiveTarget2dGpuTrainingObserver for ViewerTrainingObserver {
    fn should_stop(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }

    fn snapshot_interval_steps(&self) -> usize {
        self.snapshot_interval_steps
    }

    fn snapshot_interval_duration(&self) -> Duration {
        self.snapshot_interval_duration
    }

    fn on_progress(&mut self, progress: AdaptiveTarget2dGpuTrainingProgress) {
        self.publish_progress(ImageTargetTrainingProgress {
            step: progress.step,
            total_steps: progress.total_steps,
            loss: progress.loss,
            render_rgb_psnr_db: progress.render_rgb_psnr_db,
            base_grad_norm: progress.base_grad_norm,
            base_grad_scale: progress.base_grad_scale,
            particle_steps_per_sec: progress.particle_steps_per_sec,
            model: ImageTargetTrainingModel::Adaptive(Box::new(progress.model)),
        });
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl ViewerTrainingObserver {
    fn publish_progress(&mut self, progress: ImageTargetTrainingProgress) {
        if let Some(reason) = unsafe_training_progress_reason(&progress) {
            *self
                .safety_failure
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(reason);
            self.cancel.store(true, Ordering::Release);
            return;
        }
        if self
            .sender
            .send(ImageTargetTrainingEvent::Progress {
                job_id: self.job_id,
                target_id: self.target_id,
                progress,
            })
            .is_err()
        {
            self.cancel.store(true, Ordering::Release);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn unsafe_training_progress_reason(progress: &ImageTargetTrainingProgress) -> Option<String> {
    if !progress.loss.is_finite()
        || !progress.base_grad_norm.is_finite()
        || !progress.base_grad_scale.is_finite()
    {
        return Some(format!(
            "training stopped before publishing a non-finite update at step {}: loss={:?}, grad={:?}, scale={:?}",
            progress.step, progress.loss, progress.base_grad_norm, progress.base_grad_scale
        ));
    }
    if progress.loss > MAX_LIVE_TRAINING_LOSS {
        return Some(format!(
            "training stopped at step {} because loss {:.3} exceeded the live safety limit {:.1}",
            progress.step, progress.loss, MAX_LIVE_TRAINING_LOSS
        ));
    }
    if progress.base_grad_norm > MAX_LIVE_TRAINING_GRAD_NORM {
        return Some(format!(
            "training stopped at step {} because gradient norm {:.3} exceeded the live safety limit {:.1}",
            progress.step, progress.base_grad_norm, MAX_LIVE_TRAINING_GRAD_NORM
        ));
    }
    None
}

pub(in crate::viewer) fn handle_toggle_image_target_training(
    mut requests: MessageReader<ToggleImageTargetTraining>,
    channel: Res<ImageTargetTrainingChannel>,
    mut state: ResMut<ImageTargetTrainingState>,
    mut settings: ResMut<AutomataSettings>,
    mut runtime: ResMut<AutomataRuntime>,
    #[cfg(target_arch = "wasm32")] mut browser_worker: NonSendMut<BrowserTrainingWorker>,
) {
    for _request in requests.read() {
        if matches!(state.phase, ImageTargetTrainingPhase::Running) {
            #[cfg(target_arch = "wasm32")]
            {
                browser_worker.stop();
                state.active_job_id = None;
                state.cancel = None;
                state.phase = ImageTargetTrainingPhase::Complete;
                runtime.status = format!(
                    "image-target training stopped | {}",
                    image_training_status(&state)
                );
            }
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(cancel) = state.cancel.as_ref() {
                cancel.store(true, Ordering::Release);
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                state.phase = ImageTargetTrainingPhase::Stopping;
                runtime.status =
                    "stopping image-target training after the current optimizer step".to_string();
            }
            continue;
        }
        if matches!(state.phase, ImageTargetTrainingPhase::Stopping) {
            continue;
        }
        let Some(target) = state.target.clone() else {
            runtime.status = "open a target image before training".to_string();
            continue;
        };
        let Ok(backend) = image_training_backend() else {
            runtime.status =
                "image-target training requires hyper_dino_wgpu or hyper_dino_cuda".to_string();
            state.phase = ImageTargetTrainingPhase::Failed;
            continue;
        };

        state.next_job_id = state.next_job_id.wrapping_add(1).max(1);
        let job_id = state.next_job_id;
        #[cfg(not(target_arch = "wasm32"))]
        let cancel = Arc::new(AtomicBool::new(false));
        #[cfg(not(target_arch = "wasm32"))]
        let worker_cancel = cancel.clone();
        #[cfg(not(target_arch = "wasm32"))]
        let safety_failure = Arc::new(Mutex::new(None));
        #[cfg(not(target_arch = "wasm32"))]
        let observer_safety_failure = safety_failure.clone();
        #[cfg(not(target_arch = "wasm32"))]
        let sender = channel.sender.clone();
        let mode = ImageTargetTrainingMode::from_settings(&settings);
        let fixed_config = image_training_config(&settings);
        let adaptive_config = image_adaptive_training_config(&settings);
        let (
            training_particle_count,
            target_point_count,
            training_update_prob,
            training_seed,
            training_seed_scale,
            training_seed_mode,
            total_steps,
        ) = match mode {
            ImageTargetTrainingMode::Fixed => (
                fixed_config.particle_count,
                fixed_config.particle_count,
                fixed_config.update_prob,
                fixed_config.seed,
                fixed_config.seed_scale,
                fixed_config.seed_mode,
                fixed_config
                    .epochs
                    .saturating_add(1)
                    .saturating_mul(fixed_config.repetitions),
            ),
            ImageTargetTrainingMode::Adaptive => (
                adaptive_config.target2d.particle_count,
                adaptive_config.material.reference_particle_count,
                adaptive_config.target2d.update_prob,
                adaptive_config.target2d.seed,
                adaptive_config.target2d.seed_scale,
                adaptive_config.target2d.seed_mode,
                adaptive_config
                    .target2d
                    .epochs
                    .saturating_add(1)
                    .saturating_mul(adaptive_config.target2d.repetitions),
            ),
        };
        #[cfg(target_arch = "wasm32")]
        let _ = target_point_count;
        let starting_fresh = !state.training_initialized || state.initialized_mode != Some(mode);
        runtime.hashgrid = upstream_growing_2d_hashgrid();
        let training_model = match mode {
            ImageTargetTrainingMode::Fixed => {
                if starting_fresh {
                    runtime.model = upstream_growing_2d_model(training_seed);
                    runtime.adaptive = None;
                    runtime.trace = None;
                    runtime.frame = 0;
                    runtime.model_revision = runtime.model_revision.wrapping_add(1);
                }
                ImageTargetTrainingModel::Fixed(runtime.model.clone())
            }
            ImageTargetTrainingMode::Adaptive => {
                let adaptive = if starting_fresh {
                    match fresh_adaptive_image_model(&adaptive_config) {
                        Ok(model) => model,
                        Err(error) => {
                            state.phase = ImageTargetTrainingPhase::Failed;
                            state.error = Some(error.to_string());
                            runtime.status =
                                format!("failed to initialize adaptive training: {error}");
                            continue;
                        }
                    }
                } else if let Some(adaptive) = runtime.adaptive.as_ref() {
                    adaptive.model.clone()
                } else {
                    match fresh_adaptive_image_model(&adaptive_config) {
                        Ok(model) => model,
                        Err(error) => {
                            state.phase = ImageTargetTrainingPhase::Failed;
                            state.error = Some(error.to_string());
                            runtime.status =
                                format!("failed to initialize adaptive training: {error}");
                            continue;
                        }
                    }
                };
                if starting_fresh
                    && let Err(error) = apply_adaptive_model_snapshot(
                        &mut runtime,
                        &settings,
                        adaptive.clone(),
                        true,
                    )
                {
                    state.phase = ImageTargetTrainingPhase::Failed;
                    state.error = Some(error.to_string());
                    runtime.status = format!("failed to seed adaptive training view: {error}");
                    continue;
                }
                ImageTargetTrainingModel::Adaptive(Box::new(adaptive))
            }
        };
        settings.preset = AutomataPreset::Growing2d;
        settings.particle_count = training_particle_count;
        settings.update_prob = training_update_prob;
        settings.seed = training_seed;
        settings.seed_scale = training_seed_scale;
        settings.reference_seed_scale = training_seed_scale;
        settings.seed_mode = training_seed_mode;
        settings.mark_changed();
        let hashgrid = runtime.hashgrid.clone();
        let target_id = target.id;
        let reset_interval = settings.training_rollout_reset_interval;
        let snapshot_interval_steps = if reset_interval == 0 {
            LIVE_MODEL_SNAPSHOT_FALLBACK_STEPS
        } else {
            reset_interval
        };
        let snapshot_interval_duration = if reset_interval == 0 {
            LIVE_MODEL_SNAPSHOT_INTERVAL
        } else {
            Duration::ZERO
        };
        #[cfg(not(target_arch = "wasm32"))]
        let spawn = thread::Builder::new()
            .name("bevy-automata-target2d".to_string())
            .spawn(move || {
                let result = (|| {
                    let target_image = decode_target_image_2d_upstream(
                        target.source.bytes.as_slice(),
                        TARGET_ALPHA_THRESHOLD,
                        target_point_count,
                        None,
                    )?;
                    let mut observer = ViewerTrainingObserver {
                        job_id,
                        target_id,
                        cancel: worker_cancel,
                        safety_failure: observer_safety_failure,
                        sender: sender.clone(),
                        snapshot_interval_steps,
                        snapshot_interval_duration,
                    };
                    let completion = match training_model {
                        ImageTargetTrainingModel::Fixed(mut model) => {
                            let report = train_target_2d_gpu_with_observer(
                                backend,
                                &mut model,
                                &hashgrid,
                                target_image,
                                fixed_config,
                                Target2dLossConfig::default(),
                                None,
                                &mut observer,
                            )?;
                            ImageTargetTrainingCompletion {
                                model: ImageTargetTrainingModel::Fixed(model),
                                report: ImageTargetTrainingReport::Fixed(report),
                            }
                        }
                        ImageTargetTrainingModel::Adaptive(mut model) => {
                            let report = train_adaptive_target_2d_gpu_with_observer(
                                backend,
                                &mut model,
                                &hashgrid,
                                target_image,
                                adaptive_config,
                                Target2dLossConfig::default(),
                                None,
                                &mut observer,
                            )?;
                            ImageTargetTrainingCompletion {
                                model: ImageTargetTrainingModel::Adaptive(model),
                                report: ImageTargetTrainingReport::Adaptive(report),
                            }
                        }
                    };
                    if let Some(reason) = safety_failure
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .take()
                    {
                        return Err(std::io::Error::other(reason).into());
                    }
                    Ok::<_, Box<dyn std::error::Error>>(completion)
                })()
                .map_err(|error| error.to_string());
                let _ = sender.send(ImageTargetTrainingEvent::Finished {
                    job_id,
                    target_id,
                    result,
                });
            });

        #[cfg(target_arch = "wasm32")]
        let spawn = browser_worker.start(
            &channel,
            job_id,
            target_id,
            target.source.bytes.as_slice(),
            training_model,
            hashgrid,
            &fixed_config,
            &adaptive_config,
            snapshot_interval_steps,
            snapshot_interval_duration,
        );

        match spawn {
            Ok(_handle) => {
                state.phase = ImageTargetTrainingPhase::Running;
                state.active_job_id = Some(job_id);
                #[cfg(not(target_arch = "wasm32"))]
                {
                    state.cancel = Some(cancel);
                }
                #[cfg(target_arch = "wasm32")]
                {
                    state.cancel = None;
                }
                state.step = 0;
                state.total_steps = total_steps;
                state.loss = None;
                state.best_loss = None;
                state.grad_norm = None;
                state.render_rgb_psnr_db = None;
                state.particle_steps_per_sec = None;
                state.error = None;
                state.training_initialized = true;
                state.initialized_mode = Some(mode);
                state.last_rollout_reset_step = 0;
                settings.train_live = false;
                settings.model_path = None;
                settings.adaptive_model_path = None;
                settings.generated_model_label =
                    Some(format!("training {}", target.source.file_name));
                runtime.loaded_model_path = None;
                runtime.loaded_adaptive_model_path = None;
                runtime.loaded_preset = None;
                if mode == ImageTargetTrainingMode::Fixed {
                    runtime.adaptive = None;
                }
                reset_training_stats(&mut runtime);
                runtime.status = format!(
                    "{} {} training {} on {} | {} particles | {} optimizer steps",
                    if starting_fresh { "fresh" } else { "resumed" },
                    mode.label(),
                    target.source.file_name,
                    image_training_backend_label(backend),
                    training_particle_count,
                    total_steps
                );
            }
            Err(error) => {
                state.phase = ImageTargetTrainingPhase::Failed;
                state.error = Some(error.to_string());
                runtime.status = format!("failed to start image-target training: {error}");
            }
        }
    }
}

pub(in crate::viewer) fn poll_image_target_training(
    channel: Res<ImageTargetTrainingChannel>,
    settings: Res<AutomataSettings>,
    mut state: ResMut<ImageTargetTrainingState>,
    mut runtime: ResMut<AutomataRuntime>,
) {
    let mut latest_progress = None;
    let mut finished = None;
    for event in channel.receiver.try_iter() {
        match event {
            ImageTargetTrainingEvent::Progress {
                job_id,
                target_id,
                progress,
            } if state.active_job_id == Some(job_id)
                && state.target.as_ref().map(|target| target.id) == Some(target_id) =>
            {
                latest_progress = Some(progress);
            }
            ImageTargetTrainingEvent::Finished {
                job_id,
                target_id,
                result,
            } if state.active_job_id == Some(job_id)
                && state.target.as_ref().map(|target| target.id) == Some(target_id) =>
            {
                finished = Some(result);
            }
            _ => {}
        }
    }

    if let Some(progress) = latest_progress {
        let reset_rollout = rollout_reset_due(
            state.last_rollout_reset_step,
            progress.step,
            settings.training_rollout_reset_interval,
        );
        if let Err(error) =
            apply_live_model_snapshot(&mut runtime, &settings, progress.model, reset_rollout)
        {
            if let Some(cancel) = state.cancel.as_ref() {
                cancel.store(true, Ordering::Release);
            }
            state.phase = ImageTargetTrainingPhase::Failed;
            state.error = Some(error.clone());
            runtime.status = format!("failed to apply live training snapshot: {error}");
            return;
        }
        if reset_rollout {
            state.last_rollout_reset_step =
                progress.step - progress.step % settings.training_rollout_reset_interval.max(1);
        }
        state.phase = ImageTargetTrainingPhase::Running;
        state.step = progress.step;
        state.total_steps = progress.total_steps;
        state.loss = Some(progress.loss);
        state.grad_norm = Some(progress.base_grad_norm);
        state.best_loss = Some(
            state
                .best_loss
                .map_or(progress.loss, |best| best.min(progress.loss)),
        );
        state.render_rgb_psnr_db = progress.render_rgb_psnr_db.or(state.render_rgb_psnr_db);
        state.particle_steps_per_sec = Some(progress.particle_steps_per_sec);
        runtime.training_step = progress.step;
        runtime.training_loss = Some(progress.loss);
        runtime.training_best_loss = state.best_loss;
        runtime.training_grad_norm = Some(progress.base_grad_norm);
        runtime.status = image_training_status(&state);
    }

    if let Some(result) = finished {
        state.active_job_id = None;
        state.cancel = None;
        match result {
            Ok(completion) => {
                if let Err(error) =
                    apply_live_model_snapshot(&mut runtime, &settings, completion.model, false)
                {
                    state.phase = ImageTargetTrainingPhase::Failed;
                    state.error = Some(error.clone());
                    runtime.status = format!("failed to apply trained model: {error}");
                    return;
                }
                let report = completion.report.training();
                let stopped_early = report
                    .metrics
                    .get("stopped_early")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false);
                let steps_completed = report
                    .metrics
                    .get("steps_completed")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(state.step as u64) as usize;
                state.step = steps_completed;
                state.loss = report.history.last().map(|entry| entry.loss).or(state.loss);
                state.best_loss = report.best_train_loss.or(state.best_loss);
                state.render_rgb_psnr_db = report
                    .best_fresh_seed_render_rgb_psnr_db
                    .or(state.render_rgb_psnr_db);
                state.phase = ImageTargetTrainingPhase::Complete;
                runtime.training_step = state.step;
                runtime.training_loss = state.loss;
                runtime.training_best_loss = state.best_loss;
                runtime.status = if stopped_early {
                    format!(
                        "image-target training stopped | {}",
                        image_training_status(&state)
                    )
                } else {
                    format!(
                        "image-target training complete | {}",
                        image_training_status(&state)
                    )
                };
            }
            Err(error) => {
                state.phase = ImageTargetTrainingPhase::Failed;
                state.error = Some(error.clone());
                runtime.status = format!("image-target training failed: {error}");
            }
        }
    }
}

fn rollout_reset_due(last_reset_step: usize, step: usize, interval: usize) -> bool {
    interval > 0 && step >= last_reset_step.saturating_add(interval)
}

fn apply_live_model_snapshot(
    runtime: &mut AutomataRuntime,
    settings: &AutomataSettings,
    model: ImageTargetTrainingModel,
    reset_rollout: bool,
) -> Result<(), String> {
    match model {
        ImageTargetTrainingModel::Fixed(model) => {
            runtime.model = model;
            runtime.adaptive = None;
            if reset_rollout {
                runtime.trace = None;
                runtime.frame = 0;
            }
            runtime.model_revision = runtime.model_revision.wrapping_add(1);
            Ok(())
        }
        ImageTargetTrainingModel::Adaptive(model) => {
            apply_adaptive_model_snapshot(runtime, settings, *model, reset_rollout)
                .map_err(|error| error.to_string())
        }
    }
}

fn image_training_config(settings: &AutomataSettings) -> Target2dTrainingConfig {
    let mut config = Target2dTrainingConfig::default();
    config.optimizer.learning_rate = settings.training_learning_rate;
    config.report_interval = LIVE_TRAINING_REPORT_INTERVAL;
    #[cfg(target_arch = "wasm32")]
    apply_browser_training_budget(&mut config);
    config
}

#[cfg(any(target_arch = "wasm32", test))]
fn apply_browser_training_budget(config: &mut Target2dTrainingConfig) {
    config.epochs = BROWSER_TRAINING_SESSION_STEPS - 1;
    config.repetitions = 1;
    config.report_interval = 100;
    config.batch_size = 1;
    config.pool_size = 16;
    config.particle_count = BROWSER_TRAINING_PARTICLES;
    config.step_min = 8;
    config.step_max = 16;
    config.tbptt_chunk_steps = 8;
    config.inject_seed_interval = 4;
}

#[cfg(any(target_arch = "wasm32", test))]
fn apply_browser_adaptive_training_budget(config: &mut AdaptiveTarget2dTrainingConfig) {
    apply_browser_training_budget(&mut config.target2d);
    config.target2d.particle_count = BROWSER_ADAPTIVE_ACTIVE_PARTICLES;
    config.material.reference_particle_count = BROWSER_TRAINING_PARTICLES;
    config.event_training.post_event_recovery_steps = 8;
    config.event_training.max_recovery_extension_steps = 8;
    config.checkpoint_horizons = vec![config.target2d.step_max];
}

fn image_adaptive_training_config(settings: &AutomataSettings) -> AdaptiveTarget2dTrainingConfig {
    let mut config = AdaptiveTarget2dTrainingConfig::default();
    config.rule_training = AdaptiveTarget2dRuleTraining::SharedScaleConditionedRule;
    config.log1p_trajectory_loss = true;
    config.trajectory_tail_fraction = 0.25;
    config.trajectory_tail_weight = 1.0;
    config.max_pool_age_steps = 1_024;
    config.pool_age_strata = 8;
    config.backward_loss_scale = 1.0e-6;
    config.expected_coarse_update_mask = true;
    config.checkpoint_seeds = vec![config.target2d.seed];
    config.checkpoint_horizons = vec![config.target2d.step_max];
    config.event_training.enabled = true;
    config.event_training.post_event_recovery_steps = 32;
    config.event_training.max_recovery_extension_steps = 32;
    config.target2d.report_interval = LIVE_TRAINING_REPORT_INTERVAL;
    config.target2d.optimizer.learning_rate = settings.training_learning_rate;
    #[cfg(target_arch = "wasm32")]
    apply_browser_adaptive_training_budget(&mut config);
    config
}

fn fresh_adaptive_image_model(
    config: &AdaptiveTarget2dTrainingConfig,
) -> burn_automata::AutomataResult<AdaptiveNpaModel> {
    let mut adaptive = AdaptiveNpaConfig::growing_2d();
    adaptive.target_leaves = config.target2d.particle_count;
    adaptive.bootstrap_fine_leaves = config.material.reference_particle_count;
    adaptive.retain_bootstrap_templates = false;
    AdaptiveNpaModel::seeded(
        upstream_growing_2d_model(config.target2d.seed),
        adaptive,
        config.target2d.seed ^ 0xada2_7a2d,
    )
}

fn image_training_backend() -> Result<Target2dGpuBackend, ()> {
    #[cfg(feature = "hyper_dino_cuda")]
    {
        return Ok(Target2dGpuBackend::Cuda);
    }
    #[cfg(all(not(feature = "hyper_dino_cuda"), feature = "hyper_dino_wgpu"))]
    {
        return Ok(Target2dGpuBackend::Wgpu);
    }
    #[allow(unreachable_code)]
    Err(())
}

fn image_training_backend_label(backend: Target2dGpuBackend) -> &'static str {
    match backend {
        Target2dGpuBackend::Wgpu => "Burn/WGPU",
        Target2dGpuBackend::Cuda => "Burn/CUDA",
    }
}

pub(in crate::viewer) fn image_training_status(state: &ImageTargetTrainingState) -> String {
    let Some(target) = state.target.as_ref() else {
        return "no target".to_string();
    };
    let mut fields = vec![format!(
        "{} | {}x{}",
        compact_file_name(&target.source.file_name, 26),
        target.source.width,
        target.source.height
    )];
    if let Some(mode) = state.initialized_mode {
        fields.push(mode.label().to_string());
    }
    match state.phase {
        ImageTargetTrainingPhase::Ready => fields.push(if state.training_initialized {
            "ready to resume".to_string()
        } else {
            "ready | fresh training init".to_string()
        }),
        ImageTargetTrainingPhase::Running | ImageTargetTrainingPhase::Stopping => {
            fields.push(format!("step {}/{}", state.step, state.total_steps));
        }
        ImageTargetTrainingPhase::Complete => fields.push(format!("step {}", state.step)),
        ImageTargetTrainingPhase::Failed => fields.push("failed".to_string()),
        ImageTargetTrainingPhase::Empty => {}
    }
    if let Some(loss) = state.loss {
        fields.push(format!("loss {loss:.4}"));
    }
    if let Some(grad_norm) = state.grad_norm {
        fields.push(format!("grad {grad_norm:.2}"));
    }
    if let Some(psnr) = state.render_rgb_psnr_db {
        fields.push(format!("{psnr:.2} dB"));
    }
    if let Some(rate) = state.particle_steps_per_sec {
        fields.push(format!("{:.2}M psteps/s", rate / 1.0e6));
    }
    fields.join(" | ")
}

fn compact_file_name(file_name: &str, max_chars: usize) -> String {
    if file_name.chars().count() <= max_chars {
        return file_name.to_string();
    }
    let suffix = file_name
        .chars()
        .rev()
        .take(max_chars.saturating_sub(3))
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    format!("...{suffix}")
}

pub(in crate::viewer) fn sync_image_target_summary(
    state: Res<ImageTargetTrainingState>,
    mut summaries: Query<&mut Visibility, With<ImageTargetSummary>>,
    mut names: Query<&mut Text, With<ImageTargetName>>,
    mut progress_labels: Query<&mut Text, (With<ImageTargetProgress>, Without<ImageTargetName>)>,
) {
    if !state.is_changed() {
        return;
    }
    let visibility = if state.has_target() {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut summary in &mut summaries {
        *summary = visibility;
    }
    let Some(target) = state.target.as_ref() else {
        return;
    };
    for mut name in &mut names {
        name.0 = compact_file_name(&target.source.file_name, 30);
    }
    for mut progress in &mut progress_labels {
        progress.0 = image_training_status(&state);
    }
}

pub(in crate::viewer) fn sync_image_training_button_label(
    state: Res<ImageTargetTrainingState>,
    inference: Res<HyperNpaInferenceState>,
    mut labels: Query<(&RunControlButtonLabel, &mut Text, &mut TextColor)>,
) {
    if !state.is_changed() && !inference.is_changed() {
        return;
    }
    for (label, mut text, mut color) in &mut labels {
        if label.0 == RunControlKind::Train {
            text.0 = state.train_action_label().to_string();
            color.0 = if state.train_action_available() && inference.pending == 0 {
                Color::srgb(0.86, 0.91, 0.94)
            } else {
                Color::srgb(0.42, 0.47, 0.50)
            };
        }
    }
}

pub(in crate::viewer) fn sync_image_target_preview(
    state: Res<ImageTargetTrainingState>,
    mut preview_state: ResMut<ImageTargetPreviewState>,
    mut images: ResMut<Assets<Image>>,
    mut image_nodes: Query<&mut ImageNode, With<ImageTargetPreview>>,
) {
    let Some(target) = state.target.as_ref() else {
        return;
    };
    if preview_state.revision == state.target_revision
        && preview_state
            .handle
            .as_ref()
            .is_some_and(|handle| images.contains(handle))
    {
        return;
    }
    let mut image = Image::new(
        Extent3d {
            width: target.source.preview_width,
            height: target.source.preview_height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        target.source.preview_rgba.as_ref().clone(),
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.sampler = ImageSampler::linear();
    let handle = images.add(image);
    for mut image_node in &mut image_nodes {
        image_node.image = handle.clone();
    }
    preview_state.handle = Some(handle);
    preview_state.revision = state.target_revision;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_file_name_preserves_a_bounded_suffix() {
        assert_eq!(compact_file_name("lizard.png", 12), "lizard.png");
        let compact = compact_file_name("a-very-long-target-image.png", 16);
        assert!(compact.starts_with("..."));
        assert_eq!(compact.chars().count(), 16);
        assert!(compact.ends_with("image.png"));
    }

    #[test]
    fn image_training_config_isolated_from_rollout_view_controls() {
        let settings = AutomataSettings {
            particle_count: 2048,
            seed: 19,
            seed_scale: 0.3,
            update_prob: 0.65,
            training_learning_rate: 2.5e-4,
            ..Default::default()
        };
        let config = image_training_config(&settings);
        assert_eq!(
            config.particle_count,
            Target2dTrainingConfig::default().particle_count
        );
        assert_eq!(config.seed, Target2dTrainingConfig::default().seed);
        assert_eq!(
            config.seed_scale,
            Target2dTrainingConfig::default().seed_scale
        );
        assert_eq!(
            config.update_prob,
            Target2dTrainingConfig::default().update_prob
        );
        assert_eq!(config.optimizer.learning_rate, 2.5e-4);
        assert_eq!(config.report_interval, LIVE_TRAINING_REPORT_INTERVAL);
    }

    #[test]
    fn browser_training_budget_is_bounded_but_keeps_temporal_credit() {
        let mut config = Target2dTrainingConfig::default();
        apply_browser_training_budget(&mut config);

        assert_eq!(config.epochs + 1, BROWSER_TRAINING_SESSION_STEPS);
        assert_eq!(config.repetitions, 1);
        assert_eq!(config.batch_size, 1);
        assert_eq!(config.particle_count, BROWSER_TRAINING_PARTICLES);
        assert_eq!(config.step_min, 8);
        assert_eq!(config.step_max, 16);
        assert_eq!(config.tbptt_chunk_steps, 8);
        assert!(config.pool_size >= config.batch_size);

        let mut adaptive = AdaptiveTarget2dTrainingConfig::default();
        apply_browser_adaptive_training_budget(&mut adaptive);
        assert_eq!(
            adaptive.target2d.particle_count,
            BROWSER_ADAPTIVE_ACTIVE_PARTICLES
        );
        assert_eq!(
            adaptive.material.reference_particle_count,
            BROWSER_TRAINING_PARTICLES
        );
        assert_eq!(adaptive.checkpoint_horizons, vec![16]);
        assert_eq!(adaptive.event_training.post_event_recovery_steps, 8);
    }

    #[test]
    fn adaptive_image_training_uses_device_resident_sparse_telemetry_contract() {
        let settings = AutomataSettings {
            training_learning_rate: 2.5e-4,
            ..Default::default()
        };
        let config = image_adaptive_training_config(&settings);
        assert_eq!(
            config.rule_training,
            AdaptiveTarget2dRuleTraining::SharedScaleConditionedRule
        );
        assert_eq!(config.backward_loss_scale, 1.0e-6);
        assert!(config.event_training.enabled);
        assert_eq!(
            config.target2d.report_interval,
            LIVE_TRAINING_REPORT_INTERVAL
        );
        assert_eq!(config.target2d.optimizer.learning_rate, 2.5e-4);
        assert_eq!(
            config.material.reference_particle_count,
            Target2dTrainingConfig::default().particle_count
        );
    }

    #[test]
    fn rollout_reset_cadence_can_be_disabled_and_crossed() {
        assert!(!rollout_reset_due(0, 99, 100));
        assert!(rollout_reset_due(0, 100, 100));
        assert!(!rollout_reset_due(100, 199, 100));
        assert!(rollout_reset_due(100, 200, 100));
        assert!(!rollout_reset_due(0, usize::MAX, 0));
    }

    #[test]
    fn train_action_is_distinct_from_open_image() {
        let mut state = ImageTargetTrainingState::default();
        assert_eq!(state.train_action_label(), "train fresh");
        assert!(!state.train_action_available());

        state.phase = ImageTargetTrainingPhase::Ready;
        assert_eq!(state.train_action_label(), "train fresh");
        assert!(state.train_action_available());

        state.phase = ImageTargetTrainingPhase::Running;
        assert_eq!(state.train_action_label(), "stop");
        assert!(state.train_action_available());

        state.training_initialized = true;
        state.phase = ImageTargetTrainingPhase::Complete;
        assert_eq!(state.train_action_label(), "continue");
    }

    #[test]
    fn live_training_safety_rejects_exploding_updates() {
        let mut progress = ImageTargetTrainingProgress {
            step: 1,
            total_steps: 10,
            loss: 5.6,
            render_rgb_psnr_db: None,
            base_grad_norm: 23.0,
            base_grad_scale: 1.0,
            particle_steps_per_sec: 1.0,
            model: ImageTargetTrainingModel::Fixed(upstream_growing_2d_model(42)),
        };
        assert!(unsafe_training_progress_reason(&progress).is_none());

        progress.loss = 35_243.68;
        assert!(
            unsafe_training_progress_reason(&progress)
                .unwrap()
                .contains("loss")
        );

        progress.loss = 5.0;
        progress.base_grad_norm = f32::INFINITY;
        assert!(
            unsafe_training_progress_reason(&progress)
                .unwrap()
                .contains("non-finite")
        );
    }

    #[test]
    fn training_channel_retains_terminal_event_behind_progress() {
        let channel = ImageTargetTrainingChannel::default();
        for step in 1..=3 {
            channel
                .sender
                .send(ImageTargetTrainingEvent::Progress {
                    job_id: 7,
                    target_id: 9,
                    progress: ImageTargetTrainingProgress {
                        step,
                        total_steps: 3,
                        loss: 1.0 / step as f32,
                        render_rgb_psnr_db: None,
                        base_grad_norm: 1.0,
                        base_grad_scale: 1.0,
                        particle_steps_per_sec: 64.0,
                        model: ImageTargetTrainingModel::Fixed(upstream_growing_2d_model(
                            step as u64,
                        )),
                    },
                })
                .unwrap();
        }
        channel
            .sender
            .send(ImageTargetTrainingEvent::Finished {
                job_id: 7,
                target_id: 9,
                result: Err("terminal".to_string()),
            })
            .unwrap();

        let events = channel.receiver.try_iter().collect::<Vec<_>>();
        assert_eq!(events.len(), 4);
        assert!(matches!(
            events.last(),
            Some(ImageTargetTrainingEvent::Finished {
                job_id: 7,
                target_id: 9,
                result: Err(error),
            }) if error == "terminal"
        ));
    }
}
