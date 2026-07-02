#[cfg(feature = "splatting")]
use super::*;

#[cfg(feature = "splatting")]
pub(super) fn setup_gaussian_cloud(
    mut commands: Commands,
    mut assets: ResMut<Assets<PlanarGaussian3d>>,
    mut sorted_entries: ResMut<Assets<SortedEntries>>,
    settings: Res<AutomataSettings>,
    mut cloud_state: ResMut<AutomataCloudState>,
) {
    let cloud_asset = automata_gaussian_cloud(settings.particle_count);
    let sorted_len = sorted_entry_capacity(cloud_asset.len());
    let cloud = assets.add(cloud_asset);
    let sorted = sorted_entries.add(SortedEntries::new(1, sorted_len));
    cloud_state.handle = Some(cloud.clone());
    cloud_state.particle_count = settings.particle_count;
    commands.spawn((
        PlanarGaussian3dHandle(cloud),
        SortedEntriesHandle(sorted),
        automata_cloud_settings(&settings, 2),
        automata_cloud_aabb(&settings),
        Transform::default(),
        Visibility::default(),
        AutomataGaussianCloud,
        Name::new("automata_gaussian_cloud"),
    ));
    commands.spawn((
        Camera3d::default(),
        Camera {
            order: 0,
            clear_color: ClearColorConfig::Default,
            ..default()
        },
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::FixedVertical {
                viewport_height: 2.6,
            },
            near: -1000.0,
            far: 1000.0,
            ..OrthographicProjection::default_2d()
        }),
        Transform::from_xyz(0.0, 0.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
        GaussianCamera::default(),
        AutomataCamera2d,
        Name::new("pancam_locked_gaussian_camera_2d"),
    ));
    commands.spawn((
        Camera3d::default(),
        Camera {
            is_active: false,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
        GaussianCamera::default(),
        PanOrbitCamera {
            enabled: false,
            allow_upside_down: true,
            target_focus: Vec3::ZERO,
            target_radius: 3.0,
            zoom_lower_limit: 0.05,
            zoom_upper_limit: Some(128.0),
            orbit_smoothness: 0.1,
            pan_smoothness: 0.1,
            zoom_smoothness: 0.1,
            ..default()
        },
        AutomataCamera3d,
        Name::new("panorbit_gaussian_camera"),
    ));
}

#[cfg(not(feature = "splatting"))]
pub(super) fn setup_gaussian_cloud() {}

#[cfg(feature = "splatting")]
pub(super) fn automata_cloud_settings(
    settings: &AutomataSettings,
    spatial_dims: usize,
) -> CloudSettings {
    CloudSettings {
        global_opacity: settings.render_opacity,
        global_scale: settings.render_scale,
        opacity_adaptive_radius: true,
        sort_mode: if spatial_dims == 2 {
            SortMode::None
        } else {
            settings.render_sort_mode_3d.clone()
        },
        radix_sort_depth_bits: RadixSortDepthBits::Bits32,
        gaussian_mode: if spatial_dims == 2 {
            GaussianMode::Gaussian2d
        } else {
            GaussianMode::Gaussian3d
        },
        color_space: GaussianColorSpace::SrgbRec709Display,
        ..default()
    }
}

#[cfg(feature = "splatting")]
pub(super) fn automata_cloud_aabb(settings: &AutomataSettings) -> Aabb {
    let extent = settings
        .seed_scale
        .max(settings.reference_seed_scale)
        .max(1.6)
        * 2.25;
    Aabb::from_min_max(Vec3::splat(-extent), Vec3::splat(extent))
}

#[cfg(feature = "splatting")]
pub(super) fn sync_gaussian_cloud_settings(
    settings: Res<AutomataSettings>,
    runtime: Res<AutomataRuntime>,
    mut clouds: Query<(&mut CloudSettings, &mut Aabb), With<AutomataGaussianCloud>>,
) {
    if !settings.is_changed() && !runtime.is_changed() {
        return;
    }
    let next = automata_cloud_settings(&settings, runtime.model.config.spatial_dims);
    let next_aabb = automata_cloud_aabb(&settings);
    for (mut cloud, mut aabb) in &mut clouds {
        *cloud = next.clone();
        *aabb = next_aabb;
    }
}

#[cfg(not(feature = "splatting"))]
pub(super) fn sync_gaussian_cloud_settings() {}

#[cfg(feature = "splatting")]
pub(super) fn sync_gaussian_cloud_asset(
    mut commands: Commands,
    mut assets: ResMut<Assets<PlanarGaussian3d>>,
    mut sorted_entries: ResMut<Assets<SortedEntries>>,
    settings: Res<AutomataSettings>,
    mut cloud_state: ResMut<AutomataCloudState>,
    mut clouds: Query<
        (
            Entity,
            &mut PlanarGaussian3dHandle,
            &mut SortedEntriesHandle,
            &mut Visibility,
        ),
        With<AutomataGaussianCloud>,
    >,
    gaussian_cameras: Query<&Camera, With<GaussianCamera>>,
) {
    if cloud_state.handle.is_some() && cloud_state.particle_count == settings.particle_count {
        return;
    }
    let cloud_asset = automata_gaussian_cloud(settings.particle_count);
    let sorted_len = sorted_entry_capacity(cloud_asset.len());
    let cloud = assets.add(cloud_asset);
    let camera_count = active_gaussian_camera_count(&gaussian_cameras);
    let sorted = sorted_entries.add(SortedEntries::new(camera_count, sorted_len));
    cloud_state.handle = Some(cloud.clone());
    cloud_state.particle_count = settings.particle_count;
    for (entity, mut handle, mut sorted_handle, mut visibility) in &mut clouds {
        *handle = PlanarGaussian3dHandle(cloud.clone());
        *sorted_handle = SortedEntriesHandle(sorted.clone());
        *visibility = Visibility::Hidden;
        let mut entity_commands = commands.entity(entity);
        entity_commands.insert(AutomataCloudResizeCooldown(2));
        #[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
        entity_commands
            .remove::<PlanarStorageBindGroup<Gaussian3d>>()
            .remove::<SortBindGroup>();
    }
}

#[cfg(feature = "splatting")]
pub(super) fn sorted_entry_capacity(cloud_len: usize) -> usize {
    cloud_len.max(SORTED_ENTRY_MIN_CAPACITY)
}

#[cfg(feature = "splatting")]
pub(super) fn restore_resized_gaussian_cloud_visibility(
    mut commands: Commands,
    mut clouds: Query<(Entity, &mut Visibility, &mut AutomataCloudResizeCooldown)>,
) {
    for (entity, mut visibility, mut cooldown) in &mut clouds {
        cooldown.0 = cooldown.0.saturating_sub(1);
        if cooldown.0 == 0 {
            *visibility = Visibility::Inherited;
            commands
                .entity(entity)
                .remove::<AutomataCloudResizeCooldown>();
        }
    }
}

#[cfg(not(feature = "splatting"))]
pub(super) fn restore_resized_gaussian_cloud_visibility() {}

#[cfg(feature = "splatting")]
pub(super) fn automata_gaussian_cloud(count: usize) -> PlanarGaussian3d {
    let gaussian = Gaussian3d {
        position_visibility: [0.0, 0.0, 0.0, 0.0].into(),
        rotation: [1.0, 0.0, 0.0, 0.0].into(),
        scale_opacity: [0.00008, 0.00008, 0.00008, 0.0].into(),
        ..Default::default()
    };
    vec![gaussian; count].into()
}

#[cfg(feature = "splatting")]
pub(super) fn active_gaussian_camera_count(
    cameras: &Query<&Camera, With<GaussianCamera>>,
) -> usize {
    cameras
        .iter()
        .filter(|camera| camera.is_active)
        .count()
        .max(1)
}

#[cfg(not(feature = "splatting"))]
pub(super) fn sync_gaussian_cloud_asset() {}
