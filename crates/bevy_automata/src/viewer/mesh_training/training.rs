#[cfg(not(target_arch = "wasm32"))]
use std::thread;

#[cfg(not(target_arch = "wasm32"))]
use burn_automata::{
    Mesh3dTrainingConfig, Mesh3dTrainingObserver, train_mesh3d_wgpu_with_observer,
};
use burn_automata::{
    mesh3d_damaged_initialization, mesh3d_model_config, mesh3d_surface_initialization,
};

use super::*;

const VIEWER_MESH_PARTICLES: usize = 16_384;
#[cfg(not(target_arch = "wasm32"))]
const VIEWER_MESH_TRAINING_PARTICLES: usize = 4_096;
#[cfg(not(target_arch = "wasm32"))]
const VIEWER_MESH_TRAINING_TRAJECTORIES: usize = 8;
#[cfg(not(target_arch = "wasm32"))]
const VIEWER_MESH_TRAINING_STEPS: usize = 500;
#[cfg(not(target_arch = "wasm32"))]
const VIEWER_MESH_ROLLOUT_HORIZON: usize = 32;
#[cfg(not(target_arch = "wasm32"))]
const MAX_VIEWER_MESH_LOSS: f32 = 100_000.0;
#[cfg(not(target_arch = "wasm32"))]
const MAX_VIEWER_MESH_GRAD_NORM: f32 = 100_000.0;

#[cfg(not(target_arch = "wasm32"))]
struct ViewerMeshTrainingObserver {
    job_id: u64,
    target_id: u64,
    cancel: Arc<AtomicBool>,
    sender: Sender<MeshTargetTrainingEvent>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Mesh3dTrainingObserver for ViewerMeshTrainingObserver {
    fn should_stop(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }

    fn on_progress(&mut self, progress: Mesh3dTrainingProgress) {
        if !progress.loss.is_finite()
            || !progress.grad_norm.is_finite()
            || progress.loss > MAX_VIEWER_MESH_LOSS
            || progress.grad_norm > MAX_VIEWER_MESH_GRAD_NORM
        {
            self.cancel.store(true, Ordering::Release);
            return;
        }
        if self
            .sender
            .send(MeshTargetTrainingEvent::Progress {
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
pub(in crate::viewer) fn handle_toggle_mesh_target_training(
    mut requests: MessageReader<ToggleMeshTargetTraining>,
    channel: Res<MeshTargetTrainingChannel>,
    mut state: ResMut<MeshTargetTrainingState>,
    mut settings: ResMut<AutomataSettings>,
    mut runtime: ResMut<AutomataRuntime>,
) {
    for _request in requests.read() {
        if state.phase == MeshTargetTrainingPhase::Running {
            if let Some(cancel) = state.cancel.as_ref() {
                cancel.store(true, Ordering::Release);
            }
            state.phase = MeshTargetTrainingPhase::Stopping;
            runtime.status =
                "stopping 3D mesh training after the current optimizer step".to_string();
            continue;
        }
        if state.phase == MeshTargetTrainingPhase::Stopping {
            continue;
        }
        let Some(target) = state.target.clone() else {
            runtime.status = "open a mesh before training a 3D NPA".to_string();
            continue;
        };

        let config = viewer_mesh_training_config(&settings);
        let model = NpaModel::upstream_seeded(mesh3d_model_config(config.hidden_dims), config.seed);
        let hashgrid = HashGridConfig::growing_3dgs();
        if let Err(error) = apply_mesh_model_snapshot(
            &target.source.target,
            &mut settings,
            &mut runtime,
            model,
            hashgrid.clone(),
            MeshSnapshotInitialization::Damaged,
        ) {
            state.phase = MeshTargetTrainingPhase::Failed;
            state.error = Some(error.clone());
            runtime.status = format!("failed to initialize 3D training view: {error}");
            continue;
        }

        state.next_job_id = state.next_job_id.wrapping_add(1).max(1);
        let job_id = state.next_job_id;
        let target_id = target.id;
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let sender = channel.sender.clone();
        let target_geometry = target.source.target.clone();

        let spawn = thread::Builder::new()
            .name("bevy-automata-mesh3d".to_string())
            .spawn(move || {
                let mut observer = ViewerMeshTrainingObserver {
                    job_id,
                    target_id,
                    cancel: worker_cancel,
                    sender: sender.clone(),
                };
                let result =
                    train_mesh3d_wgpu_with_observer(&target_geometry, config, Some(&mut observer))
                        .map(|(model, hashgrid, report)| MeshTargetTrainingCompletion {
                            model,
                            hashgrid,
                            report,
                        })
                        .map_err(|error| error.to_string());
                let _ = sender.send(MeshTargetTrainingEvent::Finished {
                    job_id,
                    target_id,
                    result,
                });
            });

        match spawn {
            Ok(_handle) => {
                state.phase = MeshTargetTrainingPhase::Running;
                state.active_job_id = Some(job_id);
                state.cancel = Some(cancel);
                state.step = 0;
                state.total_steps = VIEWER_MESH_TRAINING_STEPS;
                state.loss = None;
                state.best_loss = None;
                state.grad_norm = None;
                state.error = None;
                state.last_rollout_reset_step = 0;
                settings.train_live = false;
                settings.model_path = None;
                settings.adaptive_model_path = None;
                settings.generated_model_label =
                    Some(format!("training 3D {}", target.source.file_name));
                runtime.status = format!(
                    "training 3D NPA for {} | {}x{} replay rows | {} optimizer steps",
                    target.source.file_name,
                    VIEWER_MESH_TRAINING_PARTICLES,
                    VIEWER_MESH_TRAINING_TRAJECTORIES,
                    VIEWER_MESH_TRAINING_STEPS,
                );
            }
            Err(error) => {
                let error = error.to_string();
                state.phase = MeshTargetTrainingPhase::Failed;
                state.error = Some(error.clone());
                runtime.status = format!("failed to start 3D mesh training: {error}");
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub(in crate::viewer) fn handle_toggle_mesh_target_training(
    mut requests: MessageReader<ToggleMeshTargetTraining>,
    mut state: ResMut<MeshTargetTrainingState>,
    mut runtime: ResMut<AutomataRuntime>,
) {
    for _request in requests.read() {
        let error =
            "browser mesh training is not available yet; use native training for the loaded OBJ"
                .to_string();
        state.phase = MeshTargetTrainingPhase::Failed;
        state.error = Some(error.clone());
        runtime.status = error;
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(in crate::viewer) fn poll_mesh_target_training(
    channel: Res<MeshTargetTrainingChannel>,
    mut state: ResMut<MeshTargetTrainingState>,
    mut settings: ResMut<AutomataSettings>,
    mut runtime: ResMut<AutomataRuntime>,
) {
    let mut latest_progress = None;
    let mut finished = None;
    for event in channel.receiver.try_iter() {
        match event {
            MeshTargetTrainingEvent::Progress {
                job_id,
                target_id,
                progress,
            } if state.active_job_id == Some(job_id)
                && state.target.as_ref().map(|target| target.id) == Some(target_id) =>
            {
                latest_progress = Some(progress);
            }
            MeshTargetTrainingEvent::Finished {
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
        let Some(target) = state.target.as_ref() else {
            return;
        };
        let initialization = if reset_rollout {
            MeshSnapshotInitialization::Damaged
        } else {
            MeshSnapshotInitialization::Keep
        };
        if let Err(error) = apply_mesh_model_snapshot(
            &target.source.target,
            &mut settings,
            &mut runtime,
            progress.model,
            HashGridConfig::growing_3dgs(),
            initialization,
        ) {
            if let Some(cancel) = state.cancel.as_ref() {
                cancel.store(true, Ordering::Release);
            }
            state.phase = MeshTargetTrainingPhase::Failed;
            state.error = Some(error.clone());
            runtime.status = format!("failed to apply live 3D training snapshot: {error}");
            return;
        }
        if reset_rollout {
            state.last_rollout_reset_step =
                progress.step - progress.step % settings.training_rollout_reset_interval.max(1);
        }
        state.phase = MeshTargetTrainingPhase::Running;
        state.step = progress.step;
        state.total_steps = progress.total_steps;
        state.loss = Some(progress.loss);
        state.best_loss = Some(
            state
                .best_loss
                .map_or(progress.loss, |best| best.min(progress.loss)),
        );
        state.grad_norm = Some(progress.grad_norm);
        state.refresh = progress.refresh;
        state.refreshes = progress.refreshes;
        state.policy_horizon = progress.policy_horizon;
        runtime.training_step = progress.step;
        runtime.training_loss = state.loss;
        runtime.training_best_loss = state.best_loss;
        runtime.training_grad_norm = state.grad_norm;
        runtime.status = mesh_training_status(&state);
    }

    if let Some(result) = finished {
        state.active_job_id = None;
        state.cancel = None;
        match result {
            Ok(completion) => {
                let Some(target) = state.target.as_ref() else {
                    return;
                };
                if let Err(error) = apply_mesh_model_snapshot(
                    &target.source.target,
                    &mut settings,
                    &mut runtime,
                    completion.model,
                    completion.hashgrid,
                    MeshSnapshotInitialization::Pristine,
                ) {
                    state.phase = MeshTargetTrainingPhase::Failed;
                    state.error = Some(error.clone());
                    runtime.status = format!("failed to apply trained 3D NPA: {error}");
                    return;
                }
                state.step = completion.report.training.steps;
                state.loss = Some(completion.report.training.final_loss);
                state.best_loss = Some(completion.report.training.best_loss);
                state.phase = MeshTargetTrainingPhase::Complete;
                runtime.training_step = state.step;
                runtime.training_loss = state.loss;
                runtime.training_best_loss = state.best_loss;
                runtime.status = format!(
                    "3D mesh training complete | quality {} | {:.2}M rows/s | {}",
                    if completion.report.quality.passed {
                        "passed"
                    } else {
                        "diagnostic"
                    },
                    completion.report.rows_per_second / 1.0e6,
                    mesh_training_status(&state),
                );
            }
            Err(error) => {
                state.phase = MeshTargetTrainingPhase::Failed;
                state.error = Some(error.clone());
                runtime.status = format!("3D mesh training failed: {error}");
            }
        }
    }
}

pub(super) fn install_mesh_preview(
    target: &TriangleMeshTarget,
    file_name: &str,
    settings: &mut AutomataSettings,
    runtime: &mut AutomataRuntime,
) -> Result<(), String> {
    let model = NpaModel::upstream_seeded(mesh3d_model_config(256), settings.seed);
    apply_mesh_model_snapshot(
        target,
        settings,
        runtime,
        model,
        HashGridConfig::growing_3dgs(),
        MeshSnapshotInitialization::Pristine,
    )?;
    settings.generated_model_label = Some(format!("normalized 3D {file_name}"));
    Ok(())
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
#[derive(Clone, Copy)]
enum MeshSnapshotInitialization {
    Keep,
    Pristine,
    Damaged,
}

fn apply_mesh_model_snapshot(
    target: &TriangleMeshTarget,
    settings: &mut AutomataSettings,
    runtime: &mut AutomataRuntime,
    model: NpaModel,
    hashgrid: HashGridConfig,
    initialization: MeshSnapshotInitialization,
) -> Result<(), String> {
    let reset = !matches!(initialization, MeshSnapshotInitialization::Keep);
    if reset {
        let particles = match initialization {
            MeshSnapshotInitialization::Pristine => mesh3d_surface_initialization(
                target,
                &model.config,
                VIEWER_MESH_PARTICLES,
                settings.seed,
            ),
            MeshSnapshotInitialization::Damaged => mesh3d_damaged_initialization(
                target,
                &model.config,
                VIEWER_MESH_PARTICLES,
                settings.seed,
                0.22,
                0.0,
            ),
            MeshSnapshotInitialization::Keep => unreachable!(),
        }
        .map_err(|error| error.to_string())?;
        set_particle_initialization(runtime, Some(Arc::new(particles)));
        runtime.trace = None;
        runtime.frame = 0;
    }
    runtime.model = model;
    runtime.hashgrid = hashgrid;
    runtime.loaded_model_path = None;
    runtime.loaded_adaptive_model_path = None;
    runtime.loaded_preset = None;
    runtime.adaptive = None;
    runtime.model_revision = runtime.model_revision.wrapping_add(1);
    settings.preset = AutomataPreset::Growing3dGs;
    settings.particle_count = VIEWER_MESH_PARTICLES;
    settings.seed_mode = ParticleSeed::UniformCircle;
    settings.seed_scale = 0.72;
    settings.reference_seed_scale = 0.72;
    settings.render_scale = 1.0;
    settings.render_opacity = 1.0;
    settings.model_path = None;
    settings.adaptive_model_path = None;
    settings.mark_changed();
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn viewer_mesh_training_config(settings: &AutomataSettings) -> Mesh3dTrainingConfig {
    let mut config = Mesh3dTrainingConfig {
        dataset_particles: VIEWER_MESH_TRAINING_PARTICLES,
        dataset_trajectories: VIEWER_MESH_TRAINING_TRAJECTORIES,
        teacher_rollout_max_steps: VIEWER_MESH_ROLLOUT_HORIZON,
        steps: VIEWER_MESH_TRAINING_STEPS,
        report_interval: settings.training_rollout_reset_interval.clamp(50, 250),
        seed: settings.seed,
        ..Mesh3dTrainingConfig::default()
    };
    config.optimizer.learning_rate = settings.training_learning_rate.clamp(1.0e-5, 2.0e-3);
    config.evaluation.particle_count = VIEWER_MESH_TRAINING_PARTICLES;
    config.evaluation.rollout_steps = vec![0, 32, 96];
    config.evaluation.seeds = vec![settings.seed];
    config.evaluation.target_samples = 2_048;
    config.evaluation.render_image_size = 64;
    config.evaluation.render_target_samples = VIEWER_MESH_TRAINING_PARTICLES;
    config
}

#[cfg(not(target_arch = "wasm32"))]
fn rollout_reset_due(last_reset_step: usize, step: usize, interval: usize) -> bool {
    interval > 0 && step >= last_reset_step.saturating_add(interval)
}

pub(in crate::viewer) fn mesh_training_status(state: &MeshTargetTrainingState) -> String {
    let name = state
        .target
        .as_ref()
        .map(|target| target.source.file_name.as_str())
        .unwrap_or("mesh");
    let loss = state
        .loss
        .map_or_else(|| "loss --".to_string(), |loss| format!("loss {loss:.4}"));
    format!(
        "{name} | step {}/{} | refresh {}/{} @ {} | {loss}",
        state.step,
        state.total_steps,
        state.refresh.saturating_add(1).min(state.refreshes.max(1)),
        state.refreshes.max(1),
        state.policy_horizon,
    )
}
