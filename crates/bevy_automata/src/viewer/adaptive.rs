use super::*;

use burn_automata::{
    AdaptiveNpaModel, AdaptiveParticleSet, AdaptiveStepMetrics, AdaptiveTrainingStage,
    adaptive_isotropic_gaussian_geometry, load_adaptive_model, unit_ball_measure,
};
#[cfg(not(all(feature = "splatting", feature = "gpu_wgpu")))]
use burn_automata::{AdaptiveRolloutConfig, advance_adaptive_rollout};

#[derive(Clone, Debug)]
pub struct AdaptiveViewerState {
    pub model: AdaptiveNpaModel,
    pub training_stage: AdaptiveTrainingStage,
    pub particles: AdaptiveParticleSet,
    pub last_metrics: Option<AdaptiveStepMetrics>,
}

pub(super) fn load_selected_adaptive_model(
    mut runtime: ResMut<AutomataRuntime>,
    mut settings: ResMut<AutomataSettings>,
) {
    let Some(path) = settings.adaptive_model_path.clone() else {
        if runtime.adaptive.is_some() || runtime.loaded_adaptive_model_path.is_some() {
            runtime.adaptive = None;
            runtime.loaded_adaptive_model_path = None;
            runtime.loaded_preset = None;
            runtime.frame = 0;
        }
        return;
    };
    if runtime.loaded_adaptive_model_path.as_deref() == Some(path.as_str())
        && runtime.adaptive.is_some()
    {
        return;
    }

    match load_adaptive_model(&path).and_then(|artifact| {
        let training_stage = artifact.training_stage;
        let model = artifact.model;
        let count = model.config.initial_leaf_count();
        if count < model.config.min_leaves || count > model.config.max_leaves {
            return Err(burn_automata::AutomataError::InvalidArgument(format!(
                "adaptive viewer particle count {count} is outside {}..={}",
                model.config.min_leaves, model.config.max_leaves
            )));
        }
        let total_measure = unit_ball_measure(model.config.spatial_dims)
            * settings.seed_scale.powi(model.config.spatial_dims as i32);
        let bandwidth = model.rule.config.eps0.clamp(
            model.config.perception.min_bandwidth,
            model.config.perception.max_bandwidth,
        );
        let particles = burn_automata::seed_adaptive_particles_scaled(
            &model,
            count,
            settings.seed,
            settings.seed_mode,
            settings.seed_scale,
            total_measure,
            bandwidth,
        )?;
        Ok((model, training_stage, particles))
    }) {
        Ok((model, training_stage, particles)) => {
            settings.particle_count = settings.particle_count.max(model.config.target_leaves);
            runtime.model = model.rule.clone();
            runtime.adaptive = Some(AdaptiveViewerState {
                model,
                training_stage,
                particles,
                last_metrics: None,
            });
            runtime.loaded_adaptive_model_path = Some(path.clone());
            runtime.loaded_model_path = None;
            runtime.loaded_preset = None;
            runtime.trace = None;
            runtime.frame = 0;
            runtime.status = format!(
                "loaded adaptive NPA {path} ({})",
                match training_stage {
                    AdaptiveTrainingStage::FoundationCompatibility => "foundation compatibility",
                    AdaptiveTrainingStage::TaskTrainedMultiscale => "task-trained multiscale",
                    AdaptiveTrainingStage::FreshTaskTrainedMultiscale => {
                        "fresh task-trained multiscale"
                    }
                }
            );
            runtime.backward_loss = None;
            runtime.backward_grad_norm = None;
            reset_training_stats(&mut runtime);
            runtime.model_revision = runtime.model_revision.wrapping_add(1);
        }
        Err(error) => {
            runtime.status = format!("adaptive model load failed: {error}");
        }
    }
}

#[cfg(not(all(feature = "splatting", feature = "gpu_wgpu")))]
pub(super) fn advance_adaptive_viewer(
    mut runtime: ResMut<AutomataRuntime>,
    settings: Res<AutomataSettings>,
) {
    if settings.paused || runtime.adaptive.is_none() {
        return;
    }
    let completed_steps = runtime.frame;
    let steps = settings.steps_per_frame.max(1);
    let mut adaptive = runtime.adaptive.take().expect("adaptive state checked");
    let result = advance_adaptive_rollout(
        &adaptive.model,
        adaptive.particles,
        AdaptiveRolloutConfig {
            steps,
            dt: settings.dt,
            update_prob: settings.update_prob,
            seed: settings.seed,
            bandwidth_adaptation_enabled: settings.adaptive_bandwidth_enabled,
            topology_enabled: settings.adaptive_topology_enabled,
            snapshot_interval: steps,
        },
        completed_steps,
    );
    match result {
        Ok(trace) => {
            adaptive.particles = trace.particles;
            adaptive.last_metrics = trace.metrics.last().cloned();
            runtime.frame = completed_steps.saturating_add(trace.steps);
            if let Some(metrics) = adaptive.last_metrics.as_ref() {
                let bandwidth_mode = if settings.adaptive_bandwidth_enabled
                    && adaptive.model.config.supports_bandwidth_adaptation()
                {
                    "adaptive"
                } else {
                    "fixed"
                };
                runtime.status = format!(
                    "adaptive leaves {} | footprint {:.4}..{:.4} | h {:.4} ({bandwidth_mode}) | edges {} | events +{}/-{}",
                    metrics.leaf_count,
                    metrics.min_footprint,
                    metrics.max_footprint,
                    metrics.mean_bandwidth,
                    metrics.accepted_messages,
                    metrics.split_events,
                    metrics.merge_events,
                );
            }
            runtime.adaptive = Some(adaptive);
        }
        Err(error) => {
            runtime.status = format!("adaptive rollout failed: {error}");
        }
    }
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
pub(super) fn advance_adaptive_viewer(
    mut runtime: ResMut<AutomataRuntime>,
    settings: Res<AutomataSettings>,
) {
    if settings.paused || runtime.adaptive.is_none() {
        return;
    }
    runtime.frame = runtime.frame.wrapping_add(settings.steps_per_frame.max(1));
    if runtime.status.starts_with("loaded adaptive NPA") {
        runtime.status = "gpu adaptive NPA -> planar gaussian buffers".to_string();
    }
}

#[cfg(all(feature = "splatting", not(feature = "gpu_wgpu")))]
pub(super) fn sync_adaptive_particles_to_gaussian_asset(
    runtime: Res<AutomataRuntime>,
    cloud_state: Res<AutomataCloudState>,
    mut assets: ResMut<Assets<PlanarGaussian3d>>,
) {
    let Some(adaptive) = runtime.adaptive.as_ref() else {
        return;
    };
    let Some(handle) = cloud_state.handle.as_ref() else {
        return;
    };
    let Some(mut cloud) = assets.get_mut(handle) else {
        return;
    };
    let particles = &adaptive.particles;
    let display_scale = adaptive_display_scale_per_footprint(&adaptive.model);
    let capacity = cloud_state
        .particle_count
        .min(cloud.position_visibility.len());
    let count = particles.len().min(capacity);
    for index in 0..count {
        let state_base = index * particles.state_dims;
        let state = &particles.states[state_base..state_base + particles.state_dims];
        let gaussian = adaptive_particle_gaussian(
            particles.positions[index],
            state,
            particles.spatial_dims,
            AdaptiveGaussianMaterial {
                represented_measure: particles.represented_measure[index],
                render_footprint: particles.render_footprint[index],
                display_scale_per_footprint: display_scale,
            },
            1.0,
        );
        write_planar_gaussian(&mut cloud, index, gaussian);
    }
    for index in count..capacity {
        cloud.position_visibility[index].visibility = 0.0;
        cloud.scale_opacity[index].opacity = 0.0;
    }
}

#[cfg(any(not(feature = "splatting"), feature = "gpu_wgpu"))]
pub(super) fn sync_adaptive_particles_to_gaussian_asset() {}

#[cfg(feature = "splatting")]
pub(super) fn particle_gaussian(
    position: [f32; 4],
    state: &[f32],
    spatial_dims: usize,
    material_scale: f32,
    opacity: f32,
) -> Gaussian3d {
    let mut spherical_harmonic = SphericalHarmonicCoefficients::default();
    let color = if state.len() >= 3 {
        let tail = state.len() - 3;
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
    let scale = material_scale.max(0.00008);
    Gaussian3d {
        position_visibility: [
            position[0],
            position[1],
            if spatial_dims == 3 { position[2] } else { 0.0 },
            1.0,
        ]
        .into(),
        spherical_harmonic,
        rotation: [1.0, 0.0, 0.0, 0.0].into(),
        scale_opacity: [scale, scale, scale, opacity.clamp(0.001, 1.0)].into(),
    }
}

/// Converts one conservative material leaf into an isotropic Gaussian.
///
/// Represented measure controls the physical radius. The display footprint is
/// allowed to interpolate across a topology event, with opacity preserving
/// measure during that transition. Covariance is intentionally not decoded:
/// one visible adaptive particle remains one isotropic render primitive.
#[cfg(feature = "splatting")]
#[derive(Clone, Copy, Debug)]
pub(super) struct AdaptiveGaussianMaterial {
    pub represented_measure: f32,
    pub render_footprint: f32,
    pub display_scale_per_footprint: f32,
}

#[cfg(feature = "splatting")]
pub(super) fn adaptive_particle_gaussian(
    position: [f32; 4],
    state: &[f32],
    spatial_dims: usize,
    material: AdaptiveGaussianMaterial,
    opacity: f32,
) -> Gaussian3d {
    let geometry = adaptive_isotropic_gaussian_geometry(
        material.represented_measure,
        material.render_footprint,
        spatial_dims,
    )
    .expect("validated adaptive material produces isotropic Gaussian geometry");
    let mut gaussian = particle_gaussian(position, state, spatial_dims, 1.0, opacity);
    gaussian.scale_opacity = [
        geometry.scale[0] * material.display_scale_per_footprint,
        geometry.scale[1] * material.display_scale_per_footprint,
        geometry.scale[2] * material.display_scale_per_footprint,
        (opacity * geometry.opacity).clamp(0.001, 1.0),
    ]
    .into();
    gaussian
}

#[cfg(feature = "splatting")]
pub(super) fn adaptive_display_scale_per_footprint(model: &AdaptiveNpaModel) -> f32 {
    burn_automata::adaptive_display_scale_per_footprint(model)
}

#[cfg(all(feature = "splatting", not(feature = "gpu_wgpu")))]
fn write_planar_gaussian(cloud: &mut PlanarGaussian3d, index: usize, gaussian: Gaussian3d) {
    cloud.position_visibility[index] = gaussian.position_visibility;
    cloud.spherical_harmonic[index] = gaussian.spherical_harmonic;
    cloud.rotation[index] = gaussian.rotation;
    cloud.scale_opacity[index] = gaussian.scale_opacity;
}
