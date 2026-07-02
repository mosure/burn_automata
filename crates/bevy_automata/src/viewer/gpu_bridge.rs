use super::*;

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
pub(super) fn automata_render_reinit_key(
    model: &NpaModel,
    hashgrid: &HashGridConfig,
    settings: &AutomataSettings,
    neighbor_mode: WgpuNeighborMode,
) -> AutomataRenderReinitKey {
    AutomataRenderReinitKey {
        settings_revision: settings.revision,
        particle_count: settings.particle_count,
        seed: settings.seed,
        seed_scale_bits: settings.seed_scale.to_bits(),
        reference_seed_scale_bits: settings.reference_seed_scale.to_bits(),
        seed_mode: settings.seed_mode,
        neighbor_mode,
        model_shape: AutomataRenderModelShapeKey {
            state_dims: model.config.state_dims,
            hidden_dims: model.config.hidden_dims,
            spatial_dims: model.config.spatial_dims,
            perception_dims: model.config.perception_dims(),
            update_dims: model.config.update_dims(),
        },
        hashgrid_shape: AutomataRenderHashGridShapeKey {
            dim: hashgrid.dim,
            grid_size: hashgrid.grid_size,
        },
    }
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
pub(super) fn effective_gpu_neighbor_mode(
    _runtime: &AutomataRuntime,
    settings: &AutomataSettings,
) -> WgpuNeighborMode {
    if settings.gpu_neighbor_mode != WgpuNeighborMode::Auto {
        return settings.gpu_neighbor_mode;
    }
    WgpuNeighborMode::Auto
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
#[derive(Resource, Clone)]
pub(super) struct AutomataRenderConfig {
    model: NpaModel,
    hashgrid: HashGridConfig,
    reinit_key: AutomataRenderReinitKey,
    param_key: AutomataRenderParamKey,
    particle_count: usize,
    steps_per_frame: usize,
    update_prob: f32,
    dt: f32,
    seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
    neighbor_mode: WgpuNeighborMode,
    paused: bool,
    model_revision: u64,
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct AutomataRenderParamKey {
    model_revision: u64,
    dt_bits: u32,
    update_prob_bits: u32,
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct AutomataRenderReinitKey {
    settings_revision: u64,
    particle_count: usize,
    seed: u64,
    seed_scale_bits: u32,
    reference_seed_scale_bits: u32,
    seed_mode: ParticleSeed,
    neighbor_mode: WgpuNeighborMode,
    model_shape: AutomataRenderModelShapeKey,
    hashgrid_shape: AutomataRenderHashGridShapeKey,
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct AutomataRenderModelShapeKey {
    state_dims: usize,
    hidden_dims: usize,
    spatial_dims: usize,
    perception_dims: usize,
    update_dims: usize,
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct AutomataRenderHashGridShapeKey {
    dim: usize,
    grid_size: [usize; 3],
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
#[derive(Resource, Default)]
pub(super) struct AutomataRenderState {
    executor: Option<burn_automata::gpu::WgpuAutomataExecutor>,
    state: Option<burn_automata::gpu::WgpuAutomataState>,
    gaussian_bind_group: Option<burn_automata::gpu::WgpuGaussianBindGroup>,
    reinit_key: AutomataRenderReinitKey,
    param_key: AutomataRenderParamKey,
    model_revision: u64,
    asset_id: Option<AssetId<PlanarGaussian3d>>,
    frame: usize,
    last_error: Option<String>,
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
#[derive(Resource, Clone, Debug, Default)]
pub struct AutomataRenderDiagnostics {
    pub requested_particle_count: usize,
    pub gaussian_storage_count: usize,
    pub resident_particle_count: usize,
    pub resolved_neighbor_mode: String,
    pub bucket_capacity: usize,
    pub grid_storage_len: usize,
    pub grid_clear_len: usize,
    pub frame: usize,
    pub last_error: Option<String>,
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
#[derive(Resource, Default)]
pub(super) struct AutomataRenderBridgeInstalled;

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
pub(super) fn install_automata_render_bridge(app: &mut App) {
    if app
        .world()
        .contains_resource::<AutomataRenderBridgeInstalled>()
    {
        return;
    }
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .init_resource::<AutomataRenderState>()
        .init_resource::<AutomataRenderDiagnostics>()
        .add_systems(ExtractSchedule, extract_automata_render_config)
        .add_systems(
            Render,
            step_automata_into_gaussians.in_set(RenderSystems::Prepare),
        );
    app.world_mut()
        .insert_resource(AutomataRenderBridgeInstalled);
}

#[cfg(all(feature = "gpu_wgpu", feature = "splatting"))]
pub fn automata_executor_from_render_device(
    render_device: &bevy::render::renderer::RenderDevice,
    render_queue: &bevy::render::renderer::RenderQueue,
) -> burn_automata::AutomataResult<burn_automata::gpu::WgpuAutomataExecutor> {
    use std::ops::Deref;

    burn_automata::gpu::WgpuAutomataExecutor::from_device_queue(
        render_device.wgpu_device().clone(),
        render_queue.0.deref().deref().clone(),
    )
}

#[cfg(all(feature = "gpu_wgpu", feature = "splatting"))]
pub fn gaussian_storage_buffer_refs(
    storage: &PlanarStorageGaussian3d,
) -> burn_automata::gpu::WgpuGaussianBufferRefs<'_> {
    burn_automata::gpu::WgpuGaussianBufferRefs {
        position_visibility: &storage.position_visibility,
        spherical_harmonic: &storage.spherical_harmonic,
        rotation: &storage.rotation,
        scale_opacity: &storage.scale_opacity,
    }
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
pub(super) fn write_gaussian_draw_indirect_count(
    render_queue: &RenderQueue,
    storage: &PlanarStorageGaussian3d,
    count: usize,
) {
    let instance_count = count.min(storage.count) as u32;
    let mut bytes = [0u8; 16];
    for (index, value) in [4u32, instance_count, 0u32, 0u32].iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_ne_bytes());
    }
    render_queue.write_buffer(&storage.draw_indirect_buffer, 0, &bytes);
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
pub(super) fn extract_automata_render_config(
    mut commands: Commands,
    main_world: ResMut<bevy::render::MainWorld>,
) {
    let Some(settings) = main_world.get_resource::<AutomataSettings>().cloned() else {
        return;
    };
    let Some(runtime) = main_world.get_resource::<AutomataRuntime>().cloned() else {
        return;
    };
    let hashgrid = effective_hashgrid(&runtime, &settings);
    let neighbor_mode = effective_gpu_neighbor_mode(&runtime, &settings);
    let reinit_key =
        automata_render_reinit_key(&runtime.model, &hashgrid, &settings, neighbor_mode);
    let param_key = AutomataRenderParamKey {
        model_revision: runtime.model_revision,
        dt_bits: settings.dt.to_bits(),
        update_prob_bits: settings.update_prob.to_bits(),
    };
    commands.insert_resource(AutomataRenderConfig {
        model: runtime.model,
        hashgrid,
        reinit_key,
        param_key,
        particle_count: settings.particle_count,
        steps_per_frame: settings.steps_per_frame,
        update_prob: settings.update_prob,
        dt: settings.dt,
        seed: settings.seed,
        seed_scale: settings.seed_scale,
        seed_mode: settings.seed_mode,
        neighbor_mode,
        paused: settings.paused,
        model_revision: runtime.model_revision,
    });
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn step_automata_into_gaussians(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    config: Option<Res<AutomataRenderConfig>>,
    mut render_state: ResMut<AutomataRenderState>,
    mut diagnostics: ResMut<AutomataRenderDiagnostics>,
    gpu_planars: Res<RenderAssets<PlanarStorageGaussian3d>>,
    cloud_handles: Query<(Entity, &PlanarGaussian3dHandle)>,
) {
    let Some(config) = config else {
        return;
    };
    diagnostics.requested_particle_count = config.particle_count;
    if config.paused {
        return;
    }
    let Some((cloud_entity, cloud_handle)) = cloud_handles.iter().next() else {
        diagnostics.last_error = Some("waiting for gaussian cloud entity".to_string());
        return;
    };
    let Some(storage) = gpu_planars.get(&cloud_handle.0) else {
        diagnostics.last_error = Some("waiting for gaussian render asset".to_string());
        return;
    };
    diagnostics.gaussian_storage_count = storage.count;
    if storage.count < config.particle_count {
        let message = format!(
            "waiting for gaussian storage resize: storage={} particles={}",
            storage.count, config.particle_count
        );
        render_state.last_error = Some(message.clone());
        diagnostics.last_error = Some(message);
        return;
    }

    let asset_id = cloud_handle.0.id();
    let asset_changed = render_state.asset_id != Some(asset_id);
    if asset_changed {
        render_state.gaussian_bind_group = None;
        commands
            .entity(cloud_entity)
            .remove::<PlanarStorageBindGroup<Gaussian3d>>()
            .remove::<SortBindGroup>();
    }
    let needs_reinit = render_state.state.is_none()
        || render_state.reinit_key != config.reinit_key
        || asset_changed;
    if needs_reinit {
        if render_state.executor.is_none() {
            match automata_executor_from_render_device(&render_device, &render_queue) {
                Ok(executor) => render_state.executor = Some(executor),
                Err(err) => {
                    let message = err.to_string();
                    render_state.last_error = Some(message.clone());
                    diagnostics.last_error = Some(message);
                    return;
                }
            }
        }
        let (positions, states) = burn_automata::rollout::seed_particles_scaled(
            1,
            config.particle_count,
            config.model.config.state_dims,
            config.model.config.spatial_dims,
            config.seed,
            config.seed_mode,
            config.seed_scale,
        );
        let Some(executor) = render_state.executor.as_ref() else {
            return;
        };
        match executor.create_state_with_neighbor_mode_and_update_prob(
            &config.model,
            &positions,
            &states,
            1,
            config.particle_count,
            &config.hashgrid,
            config.dt,
            config.neighbor_mode,
            config.update_prob,
            config.seed,
        ) {
            Ok(state) => {
                let neighbor = executor.neighbor_report(&state);
                render_state.state = Some(state);
                render_state.gaussian_bind_group = None;
                render_state.reinit_key = config.reinit_key;
                render_state.param_key = config.param_key;
                render_state.model_revision = config.model_revision;
                render_state.asset_id = Some(asset_id);
                render_state.frame = 0;
                render_state.last_error = None;
                diagnostics.resident_particle_count = config.particle_count;
                diagnostics.resolved_neighbor_mode = format!("{:?}", neighbor.mode);
                diagnostics.bucket_capacity = neighbor.bucket_capacity;
                diagnostics.grid_storage_len = neighbor.grid_storage_len;
                diagnostics.grid_clear_len = neighbor.grid_clear_len;
                diagnostics.last_error = None;
            }
            Err(err) => {
                let message = err.to_string();
                render_state.last_error = Some(message.clone());
                diagnostics.last_error = Some(message);
                return;
            }
        }
    } else if render_state.param_key != config.param_key {
        let update_result = {
            let AutomataRenderState {
                executor, state, ..
            } = &mut *render_state;
            match (executor.as_ref(), state.as_mut()) {
                (Some(executor), Some(state)) => executor.update_state_model(
                    state,
                    &config.model,
                    &config.hashgrid,
                    config.dt,
                    config.update_prob,
                    config.seed,
                ),
                _ => Ok(()),
            }
        };
        match update_result {
            Ok(()) => {
                render_state.param_key = config.param_key;
                render_state.model_revision = config.model_revision;
                render_state.last_error = None;
                diagnostics.last_error = None;
            }
            Err(err) => {
                let message = err.to_string();
                render_state.last_error = Some(message.clone());
                diagnostics.last_error = Some(message);
                return;
            }
        }
    }

    let gaussian_refs = gaussian_storage_buffer_refs(storage);
    if render_state.gaussian_bind_group.is_none() {
        let Some(executor) = render_state.executor.as_ref() else {
            return;
        };
        match executor.create_gaussian_bind_group(&gaussian_refs, storage.count) {
            Ok(bind_group) => render_state.gaussian_bind_group = Some(bind_group),
            Err(err) => {
                let message = err.to_string();
                render_state.last_error = Some(message.clone());
                diagnostics.last_error = Some(message);
                return;
            }
        }
    }
    if let (Some(executor), Some(state)) =
        (render_state.executor.as_ref(), render_state.state.as_ref())
    {
        let neighbor = executor.neighbor_report(state);
        diagnostics.resolved_neighbor_mode = format!("{:?}", neighbor.mode);
        diagnostics.bucket_capacity = neighbor.bucket_capacity;
        diagnostics.grid_storage_len = neighbor.grid_storage_len;
        diagnostics.grid_clear_len = neighbor.grid_clear_len;
    }
    let steps = config.steps_per_frame.max(1);
    let step_result = {
        let AutomataRenderState {
            executor,
            state,
            gaussian_bind_group,
            ..
        } = &mut *render_state;
        let Some(executor) = executor.as_ref() else {
            return;
        };
        let Some(state) = state.as_mut() else {
            return;
        };
        let Some(gaussian_bind_group) = gaussian_bind_group.as_ref() else {
            return;
        };
        executor
            .step_state_many_into_gaussian_bind_group(state, gaussian_bind_group, steps)
            .map_err(|err| err.to_string())
    };
    match step_result {
        Ok(completed) => {
            write_gaussian_draw_indirect_count(&render_queue, storage, config.particle_count);
            render_state.frame = render_state.frame.wrapping_add(completed);
            render_state.last_error = None;
            diagnostics.frame = render_state.frame;
            diagnostics.last_error = None;
        }
        Err(err) => {
            render_state.last_error = Some(err.clone());
            diagnostics.last_error = Some(err);
        }
    };
}
