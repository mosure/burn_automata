use super::*;

#[cfg(target_arch = "wasm32")]
#[derive(Resource)]
pub(super) struct BrowserModelLoadChannel {
    sender: crossbeam_channel::Sender<BrowserModelLoadResult>,
    receiver: crossbeam_channel::Receiver<BrowserModelLoadResult>,
}

#[cfg(target_arch = "wasm32")]
impl Default for BrowserModelLoadChannel {
    fn default() -> Self {
        let (sender, receiver) = crossbeam_channel::unbounded();
        Self { sender, receiver }
    }
}

#[cfg(target_arch = "wasm32")]
#[derive(Resource, Default)]
pub(super) struct BrowserModelLoadState {
    requested_path: Option<String>,
}

#[cfg(target_arch = "wasm32")]
struct BrowserModelLoadResult {
    path: String,
    result: Result<burn_automata::import::BpkModelManifest, String>,
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn load_selected_model(
    mut runtime: ResMut<AutomataRuntime>,
    settings: Res<AutomataSettings>,
) {
    if settings.adaptive_model_path.is_some() {
        return;
    }
    if let Some(model_path) = &settings.model_path {
        if runtime.loaded_model_path.as_ref() == Some(model_path) {
            return;
        }
        match burn_automata::import::load_manifest(model_path) {
            Ok(manifest) => apply_loaded_manifest(&mut runtime, model_path, manifest),
            Err(err) => {
                runtime.status = format!("model load failed: {err}");
            }
        }
        return;
    }

    if settings.generated_model_label.is_some() {
        return;
    }

    if runtime.loaded_model_path.is_none() && runtime.loaded_preset == Some(settings.preset) {
        return;
    }
    apply_seeded_preset(&mut runtime, settings.preset);
}

#[cfg(target_arch = "wasm32")]
pub(super) fn load_selected_model(
    mut runtime: ResMut<AutomataRuntime>,
    settings: Res<AutomataSettings>,
    channel: Res<BrowserModelLoadChannel>,
    mut state: ResMut<BrowserModelLoadState>,
) {
    for loaded in channel.receiver.try_iter() {
        if settings.model_path.as_deref() != Some(loaded.path.as_str()) {
            continue;
        }
        match loaded.result {
            Ok(manifest) => apply_loaded_manifest(&mut runtime, &loaded.path, manifest),
            Err(error) => runtime.status = format!("model load failed: {error}"),
        }
    }

    if settings.adaptive_model_path.is_some() {
        return;
    }
    if let Some(model_path) = &settings.model_path {
        if runtime.loaded_model_path.as_ref() == Some(model_path)
            || state.requested_path.as_ref() == Some(model_path)
        {
            return;
        }
        state.requested_path = Some(model_path.clone());
        runtime.status = format!("downloading model {model_path}");
        let path = model_path.clone();
        let sender = channel.sender.clone();
        bevy::tasks::IoTaskPool::get()
            .spawn(async move {
                let result = super::web::fetch_bytes(&path).await.and_then(|bytes| {
                    burn_automata::import::load_manifest_bytes(&bytes)
                        .map_err(|error| error.to_string())
                });
                let _ = sender.send(BrowserModelLoadResult { path, result });
            })
            .detach();
        return;
    }

    state.requested_path = None;
    if settings.generated_model_label.is_some() {
        return;
    }
    if runtime.loaded_model_path.is_none() && runtime.loaded_preset == Some(settings.preset) {
        return;
    }
    apply_seeded_preset(&mut runtime, settings.preset);
}

fn apply_loaded_manifest(
    runtime: &mut AutomataRuntime,
    model_path: &str,
    manifest: burn_automata::import::BpkModelManifest,
) {
    runtime.hashgrid = manifest.hashgrid.clone();
    runtime.model = manifest.into_model();
    runtime.loaded_model_path = Some(model_path.to_string());
    runtime.loaded_preset = None;
    runtime.trace = None;
    runtime.frame = 0;
    runtime.status = format!("loaded model {model_path}");
    runtime.backward_loss = None;
    runtime.backward_grad_norm = None;
    reset_training_stats(runtime);
    runtime.model_revision = runtime.model_revision.wrapping_add(1);
}

fn apply_seeded_preset(runtime: &mut AutomataRuntime, preset: AutomataPreset) {
    let (config, hashgrid) = NpaConfig::for_preset(preset);
    runtime.model = NpaModel::seeded(config, 42);
    runtime.hashgrid = hashgrid;
    runtime.loaded_model_path = None;
    runtime.loaded_preset = Some(preset);
    runtime.trace = None;
    runtime.frame = 0;
    runtime.status = format!("seeded preset {preset:?}");
    runtime.backward_loss = None;
    runtime.backward_grad_norm = None;
    reset_training_stats(runtime);
    runtime.model_revision = runtime.model_revision.wrapping_add(1);
}

#[cfg(not(all(feature = "splatting", feature = "gpu_wgpu")))]
pub(super) fn advance_rollout(
    mut runtime: ResMut<AutomataRuntime>,
    settings: Res<AutomataSettings>,
) {
    if settings.paused || runtime.adaptive.is_some() {
        return;
    }
    if runtime.trace.is_none() || runtime.frame.is_multiple_of(60) {
        initialize_cpu_rollout(&mut runtime, &settings);
        if settings.train_live {
            let trace = runtime.trace.clone();
            if let Some(trace) = trace.as_ref() {
                let hashgrid = effective_hashgrid(&runtime, &settings);
                update_training_probe(
                    &mut runtime,
                    trace,
                    &hashgrid,
                    settings.training_learning_rate,
                );
            }
        }
        return;
    }
    let previous_frame = runtime.frame;
    runtime.frame = runtime.frame.wrapping_add(1);
    if settings.train_live
        && crossed_interval(previous_frame, runtime.frame, TRAINING_INTERVAL_FRAMES)
    {
        let trace = runtime.trace.clone();
        if let Some(trace) = trace.as_ref() {
            let hashgrid = effective_hashgrid(&runtime, &settings);
            update_training_probe(
                &mut runtime,
                trace,
                &hashgrid,
                settings.training_learning_rate,
            );
        }
    }
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
pub(super) fn advance_rollout(
    mut runtime: ResMut<AutomataRuntime>,
    settings: Res<AutomataSettings>,
) {
    if settings.paused || runtime.adaptive.is_some() {
        return;
    }
    let previous_frame = runtime.frame;
    runtime.frame = runtime.frame.wrapping_add(settings.steps_per_frame.max(1));
    if runtime.status == "ready" {
        runtime.status = "gpu automata -> planar gaussian buffers".to_string();
    }
    let crossed_training_interval =
        crossed_interval(previous_frame, runtime.frame, TRAINING_INTERVAL_FRAMES);
    let should_probe =
        settings.visualize_backward || (settings.train_live && crossed_training_interval);
    if should_probe {
        let cfg = RolloutConfig {
            particle_count: settings
                .particle_count
                .min(BACKWARD_PROBE_PARTICLES.max(TRAINING_PROBE_PARTICLES)),
            steps: 1,
            update_prob: settings.update_prob,
            dt: settings.dt,
            seed: settings.seed,
            seed_scale: settings.seed_scale,
            ..RolloutConfig::default()
        };
        let hashgrid = effective_hashgrid(&runtime, &settings);
        if let Ok(trace) = run_rollout(&runtime.model, &hashgrid, &cfg, settings.seed_mode) {
            if settings.visualize_backward {
                update_backward_probe(&mut runtime, &trace, &hashgrid);
            }
            if settings.train_live {
                update_training_probe(
                    &mut runtime,
                    &trace,
                    &hashgrid,
                    settings.training_learning_rate,
                );
            }
        }
    }
}

pub(super) fn crossed_interval(previous: usize, current: usize, interval: usize) -> bool {
    interval > 0 && current / interval != previous / interval
}

#[cfg(not(all(feature = "splatting", feature = "gpu_wgpu")))]
pub(super) fn initialize_cpu_rollout(runtime: &mut AutomataRuntime, settings: &AutomataSettings) {
    let cfg = RolloutConfig {
        particle_count: settings.particle_count,
        steps: settings.steps_per_frame.max(1),
        update_prob: settings.update_prob,
        dt: settings.dt,
        seed: settings.seed,
        seed_scale: settings.seed_scale,
        ..RolloutConfig::default()
    };
    let hashgrid = effective_hashgrid(runtime, settings);
    match run_rollout(&runtime.model, &hashgrid, &cfg, settings.seed_mode) {
        Ok(trace) => {
            update_backward_probe(runtime, &trace, &hashgrid);
            runtime.trace = Some(trace);
            runtime.status = "initialized CPU rollout".to_string();
        }
        Err(err) => {
            runtime.status = format!("rollout failed: {err}");
        }
    }
}

#[cfg(all(feature = "splatting", not(feature = "gpu_wgpu")))]
pub(super) fn sync_cpu_trace_to_gaussian_asset(
    runtime: Res<AutomataRuntime>,
    cloud_state: Res<AutomataCloudState>,
    mut assets: ResMut<Assets<PlanarGaussian3d>>,
) {
    let Some(trace) = runtime.trace.as_ref() else {
        return;
    };
    let Some(handle) = cloud_state.handle.as_ref() else {
        return;
    };
    let Some(mut cloud) = assets.get_mut(handle) else {
        return;
    };
    let count = trace.positions.len().min(cloud_state.particle_count);
    let gaussians = (0..count)
        .map(|idx| trace_gaussian(&runtime, trace, idx))
        .collect::<Vec<_>>();
    *cloud = gaussians.into();
}

#[cfg(any(not(feature = "splatting"), feature = "gpu_wgpu"))]
pub(super) fn sync_cpu_trace_to_gaussian_asset() {}

#[cfg(all(feature = "splatting", not(feature = "gpu_wgpu")))]
pub(super) fn trace_gaussian(
    runtime: &AutomataRuntime,
    trace: &RolloutTrace,
    idx: usize,
) -> Gaussian3d {
    let position = trace.positions[idx];
    let state_base = idx * trace.state_dims;
    let state = &trace.states[state_base..state_base + trace.state_dims];
    let scale = (runtime.hashgrid.eps * 0.12).max(0.00008);
    let opacity = if runtime.model.config.spatial_dims == 3 {
        growth_3d_material_opacity_channel(trace.state_dims)
            .map(|channel| (1.0 / (1.0 + (-state[channel]).exp())).clamp(0.001, 0.95))
            .unwrap_or(1.0)
    } else {
        1.0
    };

    particle_gaussian(
        position,
        state,
        runtime.model.config.spatial_dims,
        scale,
        opacity,
    )
}

pub(super) fn update_status_label(
    runtime: Res<AutomataRuntime>,
    settings: Res<AutomataSettings>,
    mut labels: Query<&mut Text, With<StatusLabel>>,
) {
    for mut text in &mut labels {
        let mut metrics = Vec::new();
        if settings.visualize_backward {
            metrics.push("backward probe on".to_string());
        }
        if let (Some(loss), Some(grad_norm)) = (runtime.backward_loss, runtime.backward_grad_norm) {
            metrics.push(format!("backward loss {:.5}", loss));
            metrics.push(format!("backward grad {:.5}", grad_norm));
        }
        if settings.train_live {
            metrics.push(format!(
                "train {} {}r/{}f",
                LIVE_TRAINING_TARGET, TRAINING_PROBE_PARTICLES, TRAINING_INTERVAL_FRAMES
            ));
            metrics.push(format!("model rev {}", runtime.model_revision));
        }
        if let (Some(loss), Some(grad_norm)) = (runtime.training_loss, runtime.training_grad_norm) {
            metrics.push(format!("train step {}", runtime.training_step));
            metrics.push(format!("train loss {:.5}", loss));
            if let Some(best) = runtime.training_best_loss {
                metrics.push(format!("best {:.5}", best));
            }
            metrics.push(format!("train grad {:.5}", grad_norm));
        }
        let metric_text = if metrics.is_empty() {
            String::new()
        } else {
            format!(" | {}", metrics.join(" | "))
        };
        text.0 = format!("{}{}", runtime.status, metric_text);
    }
}

const PERFORMANCE_UI_REFRESH_SECONDS: f64 = 0.25;
const PERFORMANCE_UI_EMA_ALPHA: f64 = 0.25;

type PerformanceFrameQuery<'w, 's> = Query<
    'w,
    's,
    &'static mut Text,
    (
        With<PerformanceFrameLabel>,
        Without<PerformanceFpsLabel>,
        Without<PerformanceStepRateLabel>,
    ),
>;
type PerformanceFpsQuery<'w, 's> = Query<
    'w,
    's,
    &'static mut Text,
    (
        With<PerformanceFpsLabel>,
        Without<PerformanceFrameLabel>,
        Without<PerformanceStepRateLabel>,
    ),
>;
type PerformanceStepRateQuery<'w, 's> = Query<
    'w,
    's,
    &'static mut Text,
    (
        With<PerformanceStepRateLabel>,
        Without<PerformanceFrameLabel>,
        Without<PerformanceFpsLabel>,
    ),
>;
type AdaptiveDiagnosticsQuery<'w, 's> = Query<
    'w,
    's,
    &'static mut Text,
    (
        With<AdaptiveDiagnosticsLabel>,
        Without<PerformanceFrameLabel>,
        Without<PerformanceFpsLabel>,
        Without<PerformanceStepRateLabel>,
    ),
>;

#[allow(clippy::too_many_arguments)]
pub(super) fn update_performance_labels(
    time: Res<Time<Real>>,
    runtime: Res<AutomataRuntime>,
    diagnostics: Option<Res<DiagnosticsStore>>,
    telemetry: Res<AutomataPerformanceTelemetry>,
    mut state: ResMut<PerformanceUiState>,
    mut frame_labels: PerformanceFrameQuery,
    mut fps_labels: PerformanceFpsQuery,
    mut step_rate_labels: PerformanceStepRateQuery,
    mut adaptive_labels: AdaptiveDiagnosticsQuery,
) {
    let now = time.elapsed_secs_f64();
    let snapshot = telemetry.snapshot();
    let completed_steps = if snapshot.render_thread_active {
        snapshot.completed_steps
    } else {
        runtime.frame
    };
    if !state.initialized {
        state.initialized = true;
        state.last_sample_seconds = now;
        state.last_completed_steps = completed_steps;
        write_performance_labels(
            completed_steps,
            state.smoothed_fps,
            state.smoothed_step_rate,
            &mut frame_labels,
            &mut fps_labels,
            &mut step_rate_labels,
        );
        write_adaptive_diagnostics(&runtime, &snapshot, &mut adaptive_labels);
        return;
    }

    let elapsed = now - state.last_sample_seconds;
    if elapsed < PERFORMANCE_UI_REFRESH_SECONDS {
        return;
    }
    let fps = diagnostics
        .as_deref()
        .and_then(|store| store.get(&FrameTimeDiagnosticsPlugin::FPS))
        .and_then(|diagnostic| diagnostic.smoothed().or_else(|| diagnostic.average()));
    state.smoothed_fps = update_ema(state.smoothed_fps, fps);
    let completed_delta = completed_steps.saturating_sub(state.last_completed_steps);
    let step_rate = (elapsed > 0.0).then_some(completed_delta as f64 / elapsed);
    state.smoothed_step_rate = update_ema(state.smoothed_step_rate, step_rate);
    state.last_sample_seconds = now;
    state.last_completed_steps = completed_steps;

    write_performance_labels(
        completed_steps,
        state.smoothed_fps,
        state.smoothed_step_rate,
        &mut frame_labels,
        &mut fps_labels,
        &mut step_rate_labels,
    );
    write_adaptive_diagnostics(&runtime, &snapshot, &mut adaptive_labels);
}

fn update_ema(previous: Option<f64>, sample: Option<f64>) -> Option<f64> {
    sample
        .filter(|value| value.is_finite() && *value >= 0.0)
        .map(|sample| {
            previous.map_or(sample, |previous| {
                previous + PERFORMANCE_UI_EMA_ALPHA * (sample - previous)
            })
        })
        .or(previous)
}

fn write_performance_labels(
    completed_steps: usize,
    fps: Option<f64>,
    step_rate: Option<f64>,
    frame_labels: &mut PerformanceFrameQuery,
    fps_labels: &mut PerformanceFpsQuery,
    step_rate_labels: &mut PerformanceStepRateQuery,
) {
    for mut text in frame_labels {
        text.0 = format_counter(completed_steps);
    }
    for mut text in fps_labels {
        text.0 = format_rate(fps);
    }
    for mut text in step_rate_labels {
        text.0 = format_rate(step_rate);
    }
}

fn write_adaptive_diagnostics(
    runtime: &AutomataRuntime,
    snapshot: &AutomataPerformanceSnapshot,
    labels: &mut AdaptiveDiagnosticsQuery,
) {
    let message = if snapshot.render_thread_active && snapshot.adaptive {
        let min_radius = snapshot.min_material_radius.max(f32::MIN_POSITIVE);
        let median_ratio = snapshot.median_material_radius / min_radius;
        let max_ratio = snapshot.max_material_radius / min_radius;
        format!(
            "adaptive {}/{} rows | radius 1/{:.2}/{:.2}x | support {}/{} | events +{}/-{}",
            snapshot.resident_particle_count,
            snapshot.dynamics_particle_count,
            median_ratio,
            max_ratio,
            snapshot.support_bin_count,
            snapshot.requested_support_bin_count,
            snapshot.split_events,
            snapshot.merge_events,
        )
    } else if let Some(adaptive) = runtime.adaptive.as_ref() {
        let min_radius = adaptive
            .particles
            .represented_measure
            .iter()
            .map(|measure| {
                burn_automata::material_footprint_radius(*measure, adaptive.particles.spatial_dims)
            })
            .fold(f32::INFINITY, f32::min);
        let max_radius = adaptive
            .particles
            .represented_measure
            .iter()
            .map(|measure| {
                burn_automata::material_footprint_radius(*measure, adaptive.particles.spatial_dims)
            })
            .fold(0.0_f32, f32::max);
        format!(
            "adaptive {} leaves | radius {:.2}x | GPU diagnostics pending",
            adaptive.particles.len(),
            max_radius / min_radius.max(f32::MIN_POSITIVE),
        )
    } else {
        String::new()
    };
    for mut text in labels {
        text.0.clone_from(&message);
    }
}

pub(super) fn format_counter(value: usize) -> String {
    format!("{value:>8}")
}

pub(super) fn format_rate(value: Option<f64>) -> String {
    let Some(value) = value.filter(|value| value.is_finite() && *value >= 0.0) else {
        return "   --.-".to_owned();
    };
    if value < 999.95 {
        format!("{value:>7.1}")
    } else if value < 999_950.0 {
        format!("{:>6.2}k", value / 1_000.0)
    } else if value < 999_950_000.0 {
        format!("{:>6.2}M", value / 1_000_000.0)
    } else {
        "  >999M".to_owned()
    }
}

pub(super) fn update_settings_label(
    settings: Res<AutomataSettings>,
    #[cfg(feature = "hyper_dino")] target_training: Res<ImageTargetTrainingState>,
    mut labels: Query<&mut Text, With<SettingsLabel>>,
) {
    #[cfg(feature = "hyper_dino")]
    let target_changed = target_training.is_changed();
    #[cfg(not(feature = "hyper_dino"))]
    let target_changed = false;
    if !settings.is_changed() && !target_changed {
        return;
    }
    #[cfg(feature = "hyper_dino")]
    let image_training_active = target_training.is_training();
    #[cfg(not(feature = "hyper_dino"))]
    let image_training_active = false;
    for mut text in &mut labels {
        let train_state = if image_training_active {
            #[cfg(feature = "hyper_dino")]
            {
                format!(
                    "target2d {}/{}",
                    target_training.step, target_training.total_steps
                )
            }
            #[cfg(not(feature = "hyper_dino"))]
            {
                unreachable!()
            }
        } else if settings.train_live {
            format!(
                "{} {}r/{}f",
                LIVE_TRAINING_TARGET, TRAINING_PROBE_PARTICLES, TRAINING_INTERVAL_FRAMES
            )
        } else {
            "off".to_string()
        };
        text.0 = format!(
            "preset: {:?} | model: {}\nparticles: {} | steps: {} | p: {:.2} | dt: {:.3}\nmodel scale: {:.3} | splat: {:.2}x | opacity: {:.2}x | color: {}\nbackward: {} | train: {} | lr: {:.4}",
            settings.preset,
            model_display_name(&settings),
            settings.particle_count,
            settings.steps_per_frame,
            settings.update_prob,
            settings.dt,
            settings.seed_scale,
            settings.render_scale,
            settings.render_opacity,
            if settings.pca_visualization {
                "state PCA"
            } else {
                "decoded"
            },
            settings.visualize_backward,
            train_state,
            settings.training_learning_rate,
        );
    }
}

pub(super) fn model_display_name(settings: &AutomataSettings) -> String {
    settings
        .generated_model_label
        .as_deref()
        .map(ToString::to_string)
        .or_else(|| {
            settings
                .adaptive_model_path
                .as_deref()
                .and_then(|path| std::path::Path::new(path).file_name())
                .and_then(|name| name.to_str())
                .map(|name| format!("adaptive {name}"))
        })
        .or_else(|| {
            settings
                .model_path
                .as_deref()
                .and_then(|path| std::path::Path::new(path).file_name())
                .and_then(|name| name.to_str())
                .map(ToString::to_string)
        })
        .unwrap_or_else(|| "seeded".to_string())
}

pub(super) fn update_backward_probe(
    runtime: &mut AutomataRuntime,
    trace: &RolloutTrace,
    hashgrid: &HashGridConfig,
) {
    match zero_update_batch_from_trace(runtime, hashgrid, trace, BACKWARD_PROBE_PARTICLES) {
        Ok(batch) => match supervised_backward(&runtime.model, &batch) {
            Ok((_grads, report)) => {
                runtime.backward_loss = Some(report.loss);
                runtime.backward_grad_norm = Some(report.grad_norm);
            }
            Err(err) => {
                runtime.backward_loss = None;
                runtime.backward_grad_norm = None;
                runtime.status = format!("backward failed: {err}");
            }
        },
        Err(err) => {
            runtime.backward_loss = None;
            runtime.backward_grad_norm = None;
            runtime.status = format!("probe failed: {err}");
        }
    }
}

#[cfg_attr(feature = "hyper_dino", allow(dead_code))]
pub(super) fn probe_trace_for_controls(
    runtime: &AutomataRuntime,
    settings: &AutomataSettings,
    max_particles: usize,
) -> burn_automata::AutomataResult<RolloutTrace> {
    let cfg = RolloutConfig {
        particle_count: settings.particle_count.min(max_particles).max(1),
        steps: 1,
        update_prob: settings.update_prob,
        dt: settings.dt,
        seed: settings.seed,
        seed_scale: settings.seed_scale,
        ..RolloutConfig::default()
    };
    let hashgrid = effective_hashgrid(runtime, settings);
    run_rollout(&runtime.model, &hashgrid, &cfg, settings.seed_mode)
}

pub(super) fn update_training_probe(
    runtime: &mut AutomataRuntime,
    trace: &RolloutTrace,
    hashgrid: &HashGridConfig,
    learning_rate: f32,
) {
    match training_batch_from_trace(runtime, hashgrid, trace, TRAINING_PROBE_PARTICLES) {
        Ok(batch) => {
            let rows = batch.features.len() / runtime.model.config.perception_dims();
            match supervised_train_step(
                &mut runtime.model,
                &batch,
                SgdConfig {
                    learning_rate,
                    grad_clip_norm: 1.0,
                    ..SgdConfig::default()
                },
            )
            .and_then(|report| {
                let loss = supervised_loss(&runtime.model, &batch)?;
                Ok((report, loss))
            }) {
                Ok((report, loss)) => {
                    runtime.training_step = runtime.training_step.wrapping_add(1);
                    runtime.training_loss = Some(loss);
                    runtime.training_best_loss = Some(
                        runtime
                            .training_best_loss
                            .map_or(loss, |best| best.min(loss)),
                    );
                    runtime.training_grad_norm = Some(report.grad_norm);
                    runtime.model_revision = runtime.model_revision.wrapping_add(1);
                    runtime.status = format!(
                        "live train rollout teacher | step {} | rows {} | lr {:.4} | grad scale {:.3}",
                        runtime.training_step, rows, learning_rate, report.grad_scale
                    );
                }
                Err(err) => {
                    runtime.training_loss = None;
                    runtime.training_grad_norm = None;
                    runtime.status = format!("training failed: {err}");
                }
            }
        }
        Err(err) => {
            runtime.training_loss = None;
            runtime.training_grad_norm = None;
            runtime.status = format!("training probe failed: {err}");
        }
    }
}

pub(super) fn training_batch_from_trace(
    runtime: &AutomataRuntime,
    hashgrid: &HashGridConfig,
    trace: &RolloutTrace,
    max_rows: usize,
) -> burn_automata::AutomataResult<burn_automata::SupervisedBatch> {
    let target = runtime
        .training_teacher
        .as_ref()
        .map(SupervisedTarget::Teacher)
        .unwrap_or(SupervisedTarget::ZeroUpdate);
    rollout_supervised_batch(
        &runtime.model,
        hashgrid,
        trace,
        target,
        RolloutBatchConfig { max_rows, dt: 1.0 },
    )
}

pub(super) fn zero_update_batch_from_trace(
    runtime: &AutomataRuntime,
    hashgrid: &HashGridConfig,
    trace: &RolloutTrace,
    max_rows: usize,
) -> burn_automata::AutomataResult<burn_automata::SupervisedBatch> {
    rollout_supervised_batch(
        &runtime.model,
        hashgrid,
        trace,
        SupervisedTarget::ZeroUpdate,
        RolloutBatchConfig { max_rows, dt: 1.0 },
    )
}

pub(super) fn apply_preset(runtime: &mut AutomataRuntime, preset: AutomataPreset) {
    let (config, hashgrid) = NpaConfig::for_preset(preset);
    runtime.model = NpaModel::seeded(config, 42);
    runtime.hashgrid = hashgrid;
    runtime.loaded_model_path = None;
    runtime.loaded_adaptive_model_path = None;
    runtime.adaptive = None;
    runtime.loaded_preset = Some(preset);
    runtime.trace = None;
    runtime.frame = 0;
    runtime.status = format!("preset changed to {preset:?}");
    runtime.backward_loss = None;
    runtime.backward_grad_norm = None;
    reset_training_stats(runtime);
    runtime.model_revision = runtime.model_revision.wrapping_add(1);
}

pub(super) fn reset_training_stats(runtime: &mut AutomataRuntime) {
    runtime.training_step = 0;
    runtime.training_loss = None;
    runtime.training_best_loss = None;
    runtime.training_grad_norm = None;
    runtime.training_teacher = None;
}
