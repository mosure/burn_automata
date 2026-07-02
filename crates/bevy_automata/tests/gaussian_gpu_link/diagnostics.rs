use super::prelude::*;

pub(crate) fn assert_viewer_cloud_capacity(apps: &mut SubApps, expected_particles: usize) {
    let world = apps.main.world_mut();
    let pairs = {
        let mut query = world.query::<(&PlanarGaussian3dHandle, &SortedEntriesHandle)>();
        query
            .iter(world)
            .map(|(cloud, sorted)| (cloud.0.clone(), sorted.0.clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(pairs.len(), 1, "expected one automata gaussian cloud");
    let (cloud_handle, sorted_handle) = &pairs[0];
    let cloud_len = world
        .resource::<Assets<PlanarGaussian3d>>()
        .get(cloud_handle)
        .expect("cloud asset should be present")
        .len();
    let sorted_count = world
        .resource::<Assets<SortedEntries>>()
        .get(sorted_handle)
        .expect("sorted entries asset should be present")
        .entry_count;
    assert_eq!(cloud_len, expected_particles);
    assert!(
        sorted_count >= cloud_len,
        "sorted entry count {sorted_count} is smaller than cloud len {cloud_len}"
    );
}

pub(crate) fn assert_runtime_spatial_dims(apps: &mut SubApps, expected_spatial_dims: usize) {
    let runtime = apps.main.world().resource::<AutomataRuntime>();
    assert_eq!(runtime.model.config.spatial_dims, expected_spatial_dims);
    assert_eq!(runtime.hashgrid.dim, expected_spatial_dims);
}

pub(crate) fn assert_cloud_mode(apps: &mut SubApps, expected_spatial_dims: usize) {
    let world = apps.main.world_mut();
    let modes = {
        let mut query = world.query::<&CloudSettings>();
        query
            .iter(world)
            .map(|settings| settings.gaussian_mode)
            .collect::<Vec<_>>()
    };
    assert_eq!(modes.len(), 1, "expected one cloud settings component");
    let expected = if expected_spatial_dims == 2 {
        GaussianMode::Gaussian2d
    } else {
        GaussianMode::Gaussian3d
    };
    assert_eq!(modes[0], expected);
}

pub(crate) fn assert_render_resize_caught_up(apps: &SubApps, expected_particles: usize) {
    let diagnostics = render_diagnostics(apps);
    assert_eq!(
        diagnostics.requested_particle_count, expected_particles,
        "render bridge did not receive the latest particle count"
    );
    assert_eq!(
        diagnostics.resident_particle_count, expected_particles,
        "resident automata GPU state did not resize"
    );
    assert!(
        diagnostics.gaussian_storage_count >= expected_particles,
        "gaussian storage count {} is smaller than requested particles {}",
        diagnostics.gaussian_storage_count,
        expected_particles
    );
    assert!(
        diagnostics.last_error.is_none(),
        "render bridge still reports an error after resize: {:?}",
        diagnostics.last_error
    );
}

pub(crate) fn render_diagnostics(apps: &SubApps) -> AutomataRenderDiagnostics {
    apps.sub_apps
        .values()
        .find_map(|sub_app| {
            sub_app
                .world()
                .get_resource::<AutomataRenderDiagnostics>()
                .cloned()
        })
        .expect("render diagnostics resource should exist")
}

pub(crate) fn gaussian_camera_snapshot(apps: &mut SubApps) -> Vec<String> {
    let world = apps.main.world_mut();
    let mut query = world.query::<(
        &Camera,
        Option<&Name>,
        Option<&Camera3d>,
        Option<&Projection>,
        Option<&GaussianCamera>,
        Option<&SortTrigger>,
        Option<&Transform>,
        Option<&RenderTarget>,
    )>();
    query
        .iter(world)
        .map(|(camera, name, camera_3d, projection, gaussian, sort_trigger, transform, target)| {
            format!(
                "{} active={} camera3d={} projection={:?} viewport={:?} gaussian_warmup={:?} sort_trigger={:?} transform={:?} target={:?}",
                name.map(|name| name.as_str()).unwrap_or("<unnamed>"),
                camera.is_active,
                camera_3d.is_some(),
                projection.map(|projection| match projection {
                    Projection::Perspective(_) => "perspective",
                    Projection::Orthographic(_) => "orthographic",
                    Projection::Custom(_) => "custom",
                }),
                camera.viewport,
                gaussian.map(|camera| camera.warmup),
                sort_trigger.map(|trigger| (trigger.camera_index, trigger.needs_sort)),
                transform.map(|transform| (transform.translation, transform.rotation)),
                target
            )
        })
        .collect()
}

pub(crate) fn gaussian_cloud_snapshot(apps: &mut SubApps) -> Vec<String> {
    let world = apps.main.world_mut();
    let cloud_info = {
        let mut query = world.query::<(
            &PlanarGaussian3dHandle,
            &SortedEntriesHandle,
            &CloudSettings,
            &Visibility,
            Option<&ViewVisibility>,
            Option<&Aabb>,
            Option<&Name>,
        )>();
        query
            .iter(world)
            .map(
                |(cloud, sorted, settings, visibility, view_visibility, aabb, name)| {
                    (
                        cloud.0.clone(),
                        sorted.0.clone(),
                        settings.gaussian_mode,
                        settings.sort_mode.clone(),
                        *visibility,
                        view_visibility.map(|visibility| visibility.get()),
                        aabb.map(|aabb| (aabb.min(), aabb.max())),
                        name.map(|name| name.as_str().to_string()),
                    )
                },
            )
            .collect::<Vec<_>>()
    };
    let clouds = world.resource::<Assets<PlanarGaussian3d>>();
    let sorted_entries = world.resource::<Assets<SortedEntries>>();
    cloud_info
        .into_iter()
        .map(
            |(cloud, sorted, mode, sort_mode, visibility, view_visible, aabb, name)| {
            let cloud_len = clouds.get(&cloud).map(PlanarGaussian3d::len);
            let sorted_len = sorted_entries.get(&sorted).map(|sorted| sorted.entry_count);
            format!(
                "{} mode={mode:?} sort={sort_mode:?} visibility={visibility:?} view_visible={view_visible:?} aabb={aabb:?} cloud_len={cloud_len:?} sorted_len={sorted_len:?}",
                name.as_deref().unwrap_or("<unnamed>")
            )
        })
        .collect()
}

pub(crate) fn gaussian_render_world_snapshot(apps: &mut SubApps) -> Vec<String> {
    apps.sub_apps
        .values_mut()
        .flat_map(|sub_app| {
            let world = sub_app.world_mut();
            let mut query = world.query::<(
                Option<&PlanarGaussian3dHandle>,
                Option<&PlanarStorageBindGroup<Gaussian3d>>,
                Option<&SortBindGroup>,
                Option<&CloudSettings>,
                Option<&Name>,
            )>();
            query
                .iter(world)
                .filter(|(handle, storage, sort, settings, _name)| {
                    handle.is_some() || storage.is_some() || sort.is_some() || settings.is_some()
                })
                .map(|(handle, storage, sort, settings, name)| {
                    format!(
                        "{} handle={} storage_bind={} sort_bind={} settings={:?}",
                        name.map(|name| name.as_str()).unwrap_or("<unnamed>"),
                        handle.is_some(),
                        storage.is_some(),
                        sort.is_some(),
                        settings
                            .map(|settings| (settings.gaussian_mode, settings.sort_mode.clone()))
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

pub(crate) fn gaussian_render_visibility_snapshot(apps: &mut SubApps) -> Vec<String> {
    apps.sub_apps
        .values_mut()
        .flat_map(|sub_app| {
            let world = sub_app.world_mut();
            let phase_counts = world
                .get_resource::<ViewSortedRenderPhases<Transparent3d>>()
                .map(|phases| {
                    phases
                        .0
                        .iter()
                        .map(|(view, phase)| (*view, phase.items.len(), phase.transient_items.len()))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let mut query = world.query::<(
                Option<&ExtractedView>,
                Option<&GaussianCamera>,
                Option<&RenderVisibleEntities>,
                Option<&Name>,
            )>();
            query
                .iter(world)
                .filter(|(view, gaussian, visible, _name)| {
                    view.is_some() || gaussian.is_some() || visible.is_some()
                })
                .map(|(view, gaussian, visible, name)| {
                    let visible_count = visible
                        .and_then(|visible| visible.get::<CloudVisibilityClass>())
                        .map(|class| class.entities_cpu_culling.len());
                    let phase_count = view.and_then(|view| {
                        phase_counts
                            .iter()
                            .find(|(retained, _, _)| *retained == view.retained_view_entity)
                            .map(|(_, items, transient)| (*items, *transient))
                    });
                    format!(
                        "{} extracted_view={} gaussian={} visible_clouds={visible_count:?} phase={phase_count:?}",
                        name.map(|name| name.as_str()).unwrap_or("<unnamed>"),
                        view.is_some(),
                        gaussian.is_some(),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}
