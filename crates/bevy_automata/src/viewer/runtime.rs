use super::*;

pub(super) fn load_selected_model(
    mut runtime: ResMut<AutomataRuntime>,
    settings: Res<AutomataSettings>,
) {
    if let Some(model_path) = &settings.model_path {
        if runtime.loaded_model_path.as_ref() == Some(model_path) {
            return;
        }
        match burn_automata::import::load_manifest(model_path) {
            Ok(manifest) => {
                runtime.hashgrid = manifest.hashgrid.clone();
                runtime.model = manifest.into_model();
                runtime.loaded_model_path = Some(model_path.clone());
                runtime.loaded_preset = None;
                runtime.trace = None;
                runtime.frame = 0;
                runtime.status = format!("loaded model {model_path}");
                runtime.backward_loss = None;
                runtime.backward_grad_norm = None;
                reset_training_stats(&mut runtime);
                runtime.model_revision = runtime.model_revision.wrapping_add(1);
            }
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
    let (config, hashgrid) = NpaConfig::for_preset(settings.preset);
    runtime.model = NpaModel::seeded(config, 42);
    runtime.hashgrid = hashgrid;
    runtime.loaded_model_path = None;
    runtime.loaded_preset = Some(settings.preset);
    runtime.trace = None;
    runtime.frame = 0;
    runtime.status = format!("seeded preset {:?}", settings.preset);
    runtime.backward_loss = None;
    runtime.backward_grad_norm = None;
    reset_training_stats(&mut runtime);
    runtime.model_revision = runtime.model_revision.wrapping_add(1);
}

#[cfg(not(all(feature = "splatting", feature = "gpu_wgpu")))]
pub(super) fn advance_rollout(
    mut runtime: ResMut<AutomataRuntime>,
    settings: Res<AutomataSettings>,
) {
    if settings.paused {
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
    if settings.paused {
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
    let mut spherical_harmonic = SphericalHarmonicCoefficients::default();
    let tail = trace.state_dims.saturating_sub(3);
    let color = if trace.state_dims >= 3 {
        [
            (state[tail] + 0.5).clamp(0.0, 1.0),
            (state[tail + 1] + 0.5).clamp(0.0, 1.0),
            (state[tail + 2] + 0.5).clamp(0.0, 1.0),
        ]
    } else {
        [0.82, 0.88, 0.92]
    };
    spherical_harmonic.coefficients[0] = (color[0] - 0.5) / GAUSSIAN_SH_C0;
    spherical_harmonic.coefficients[1] = (color[1] - 0.5) / GAUSSIAN_SH_C0;
    spherical_harmonic.coefficients[2] = (color[2] - 0.5) / GAUSSIAN_SH_C0;

    let scale = (runtime.hashgrid.eps * 0.12).max(0.00008);
    let opacity = if runtime.model.config.spatial_dims == 3 {
        growth_3d_material_opacity_channel(trace.state_dims)
            .map(|channel| (1.0 / (1.0 + (-state[channel]).exp())).clamp(0.001, 0.95))
            .unwrap_or(1.0)
    } else {
        1.0
    };

    Gaussian3d {
        position_visibility: [
            position[0],
            position[1],
            if runtime.model.config.spatial_dims == 3 {
                position[2]
            } else {
                0.0
            },
            1.0,
        ]
        .into(),
        spherical_harmonic,
        rotation: [1.0, 0.0, 0.0, 0.0].into(),
        scale_opacity: [scale, scale, scale, opacity].into(),
    }
}

pub(super) fn update_status_label(
    runtime: Res<AutomataRuntime>,
    settings: Res<AutomataSettings>,
    diagnostics: Option<Res<DiagnosticsStore>>,
    mut labels: Query<&mut Text, With<StatusLabel>>,
) {
    let fps = diagnostics
        .as_deref()
        .and_then(|store| store.get(&FrameTimeDiagnosticsPlugin::FPS))
        .and_then(|diagnostic| diagnostic.smoothed().or_else(|| diagnostic.average()));
    let frame_text = if let Some(fps) = fps {
        format!("frame {} | fps {:.1}", runtime.frame, fps)
    } else {
        format!("frame {}", runtime.frame)
    };
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
        text.0 = format!("{}\n{}{}", runtime.status, frame_text, metric_text);
    }
}

pub(super) fn update_settings_label(
    settings: Res<AutomataSettings>,
    mut labels: Query<&mut Text, With<SettingsLabel>>,
) {
    if !settings.is_changed() {
        return;
    }
    for mut text in &mut labels {
        let train_state = if settings.train_live {
            format!(
                "{} {}r/{}f",
                LIVE_TRAINING_TARGET, TRAINING_PROBE_PARTICLES, TRAINING_INTERVAL_FRAMES
            )
        } else {
            "off".to_string()
        };
        text.0 = format!(
            "preset: {:?} | model: {}\nparticles: {} | steps: {} | p: {:.2} | dt: {:.3}\nmodel scale: {:.3} | splat: {:.2}x | opacity: {:.2}x\nbackward: {} | train: {} | lr: {:.4}",
            settings.preset,
            model_display_name(&settings),
            settings.particle_count,
            settings.steps_per_frame,
            settings.update_prob,
            settings.dt,
            settings.seed_scale,
            settings.render_scale,
            settings.render_opacity,
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
