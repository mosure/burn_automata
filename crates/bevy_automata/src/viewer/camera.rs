use super::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct AutomataUiLayoutMetrics {
    pub mobile: bool,
    pub panel_width: f32,
    pub panel_height: f32,
    pub render_size: Vec2,
}

pub(super) fn automata_ui_layout_metrics(logical_size: Vec2) -> AutomataUiLayoutMetrics {
    let width = logical_size.x.max(1.0);
    let height = logical_size.y.max(1.0);
    if width >= AUTOMATA_UI_DESKTOP_MIN_WIDTH
        && width - AUTOMATA_UI_PANEL_WIDTH >= AUTOMATA_MIN_VIEWPORT_WIDTH as f32
    {
        return AutomataUiLayoutMetrics {
            mobile: false,
            panel_width: AUTOMATA_UI_PANEL_WIDTH,
            panel_height: height,
            render_size: Vec2::new(width - AUTOMATA_UI_PANEL_WIDTH, height),
        };
    }

    let maximum_panel_height = (height - AUTOMATA_UI_MOBILE_MIN_VIEW_HEIGHT).max(height * 0.36);
    let panel_height = (height * AUTOMATA_UI_MOBILE_PANEL_HEIGHT_RATIO)
        .clamp(
            AUTOMATA_UI_MOBILE_PANEL_MIN_HEIGHT.min(maximum_panel_height),
            AUTOMATA_UI_MOBILE_PANEL_MAX_HEIGHT.min(maximum_panel_height),
        )
        .min(height - 1.0);
    AutomataUiLayoutMetrics {
        mobile: true,
        panel_width: width,
        panel_height,
        render_size: Vec2::new(width, (height - panel_height).max(1.0)),
    }
}

#[cfg(feature = "splatting")]
#[allow(clippy::type_complexity)]
pub(super) fn sync_view_cameras(
    runtime: Res<AutomataRuntime>,
    mut camera_2d: Query<&mut Camera, (With<AutomataCamera2d>, Without<AutomataCamera3d>)>,
    mut camera_3d: Query<
        (&mut Camera, &mut PanOrbitCamera),
        (With<AutomataCamera3d>, Without<AutomataCamera2d>),
    >,
) {
    let use_2d = runtime.model.config.spatial_dims == 2;

    for mut camera in &mut camera_2d {
        camera.is_active = use_2d;
    }

    for (mut camera, mut pan_orbit) in &mut camera_3d {
        camera.is_active = !use_2d;
        pan_orbit.enabled = !use_2d;
    }
}

#[cfg(feature = "splatting")]
#[allow(clippy::type_complexity)]
pub(super) fn sync_automata_camera_viewports(
    ui_state: Res<AutomataUiState>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut cameras: Query<&mut Camera, Or<(With<AutomataCamera2d>, With<AutomataCamera3d>)>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let viewport = automata_camera_viewport(
        window.physical_size(),
        window.scale_factor(),
        ui_state.visible,
    );
    for mut camera in &mut cameras {
        camera.viewport = viewport.clone();
    }
}

#[cfg(not(feature = "splatting"))]
pub(super) fn sync_automata_camera_viewports() {}

#[cfg(feature = "splatting")]
pub(super) fn automata_camera_viewport(
    physical_size: UVec2,
    scale_factor: f32,
    ui_visible: bool,
) -> Option<Viewport> {
    if !ui_visible || physical_size.x <= AUTOMATA_MIN_VIEWPORT_WIDTH {
        return None;
    }

    let scale_factor = scale_factor.max(1.0e-4);
    let logical_size = physical_size.as_vec2() / scale_factor;
    let layout = automata_ui_layout_metrics(logical_size);
    if layout.mobile {
        let render_height = (layout.render_size.y * scale_factor)
            .round()
            .clamp(1.0, physical_size.y as f32) as u32;
        return Some(Viewport {
            physical_position: UVec2::ZERO,
            physical_size: UVec2::new(physical_size.x.max(1), render_height),
            depth: 0.0..1.0,
        });
    }

    let panel_physical_width = (layout.panel_width * scale_factor).round() as u32;
    let right_width = physical_size.x.saturating_sub(panel_physical_width);
    if right_width < AUTOMATA_MIN_VIEWPORT_WIDTH {
        return None;
    }

    Some(Viewport {
        physical_position: UVec2::new(panel_physical_width, 0),
        physical_size: UVec2::new(right_width, physical_size.y.max(1)),
        depth: 0.0..1.0,
    })
}

#[cfg(feature = "splatting")]
#[allow(clippy::too_many_arguments)]
pub(super) fn gate_camera_controls_while_using_ui(
    ui_state: Res<AutomataUiState>,
    preview: Res<CatalogPreviewState>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut capture: ResMut<AutomataUiInputCapture>,
    panels: Query<(&ComputedNode, &UiGlobalTransform), With<AutomataUiPanel>>,
    sliders: Query<&SliderDragState, With<AutomataSlider>>,
    mut cameras: Query<(&mut PanOrbitCamera, &Camera)>,
) {
    let dragging_slider = ui_state.visible && sliders.iter().any(|state| state.dragging);
    let cursor_over_panel = ui_state.visible
        && windows
            .single()
            .ok()
            .and_then(Window::cursor_position)
            .is_some_and(|cursor| {
                panels
                    .iter()
                    .any(|(node, transform)| node.contains_point(*transform, cursor))
            });
    let mouse_just_pressed = mouse_buttons.just_pressed(MouseButton::Left)
        || mouse_buttons.just_pressed(MouseButton::Middle)
        || mouse_buttons.just_pressed(MouseButton::Right);
    let mouse_pressed = mouse_buttons.pressed(MouseButton::Left)
        || mouse_buttons.pressed(MouseButton::Middle)
        || mouse_buttons.pressed(MouseButton::Right);

    if mouse_just_pressed && cursor_over_panel {
        capture.active = true;
    } else if !mouse_pressed {
        capture.active = false;
    }

    let ui_owns_pointer = capture.active
        || dragging_slider
        || cursor_over_panel
        || (ui_state.visible && preview.open);

    for (mut pan_orbit, camera) in &mut cameras {
        pan_orbit.enabled = camera.is_active && !ui_owns_pointer;
    }
}

#[cfg(feature = "splatting")]
#[allow(clippy::too_many_arguments)]
pub(super) fn pan_zoom_2d_camera(
    ui_state: Res<AutomataUiState>,
    preview: Res<CatalogPreviewState>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut mouse_wheel: MessageReader<MouseWheel>,
    capture: Res<AutomataUiInputCapture>,
    panels: Query<(&ComputedNode, &UiGlobalTransform), With<AutomataUiPanel>>,
    sliders: Query<&SliderDragState, With<AutomataSlider>>,
    mut last_cursor: Local<Option<Vec2>>,
    mut cameras: Query<(&Camera, &mut Projection, &mut Transform), With<AutomataCamera2d>>,
) {
    let Ok(window) = windows.single() else {
        *last_cursor = None;
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        *last_cursor = None;
        return;
    };

    let dragging_slider = ui_state.visible && sliders.iter().any(|state| state.dragging);
    let cursor_over_panel = ui_state.visible
        && panels
            .iter()
            .any(|(node, transform)| node.contains_point(*transform, cursor));
    let ui_owns_pointer = capture.active
        || dragging_slider
        || cursor_over_panel
        || (ui_state.visible && preview.open);

    let current_cursor = Vec2::new(cursor.x, -cursor.y);
    let mut wheel_delta = 0.0;
    for event in mouse_wheel.read() {
        let unit_scale = match event.unit {
            MouseScrollUnit::Line => 100.0,
            MouseScrollUnit::Pixel => 1.0,
        };
        wheel_delta += event.y * unit_scale * 0.001;
    }

    let mut active_camera = false;
    for (camera, mut projection, mut transform) in &mut cameras {
        if !camera.is_active {
            continue;
        }
        active_camera = true;
        if ui_owns_pointer {
            continue;
        }

        let Projection::Orthographic(projection) = &mut *projection else {
            continue;
        };
        let view_size = camera.logical_viewport_size().unwrap_or(window.size());
        if view_size.x <= 0.0 || view_size.y <= 0.0 {
            continue;
        }
        projection.update(view_size.x, view_size.y);

        if wheel_delta != 0.0 {
            let previous_scale = projection.scale;
            let previous_area = projection.area;
            projection.scale = (projection.scale * (1.0 - wheel_delta)).clamp(0.05, 16.0);
            projection.update(view_size.x, view_size.y);

            let view_origin = camera
                .logical_viewport_rect()
                .map(|viewport| viewport.min)
                .unwrap_or(Vec2::ZERO);
            let cursor_ndc = ((cursor - view_origin) / view_size) * 2.0 - Vec2::ONE;
            let cursor_view = Vec2::new(cursor_ndc.x, -cursor_ndc.y);
            let previous_size = previous_area.size() / previous_scale;
            let cursor_world =
                transform.translation.truncate() + cursor_view * previous_size * previous_scale;
            let proposed_position = cursor_world - cursor_view * previous_size * projection.scale;
            transform.translation.x = proposed_position.x;
            transform.translation.y = proposed_position.y;
        }

        let dragging = (mouse_buttons.pressed(MouseButton::Left)
            || mouse_buttons.pressed(MouseButton::Middle)
            || mouse_buttons.pressed(MouseButton::Right))
            && !(mouse_buttons.just_pressed(MouseButton::Left)
                || mouse_buttons.just_pressed(MouseButton::Middle)
                || mouse_buttons.just_pressed(MouseButton::Right));
        if dragging {
            let delta_device_pixels = current_cursor - last_cursor.unwrap_or(current_cursor);
            let world_units_per_pixel = projection.area.size() / view_size;
            let proposed_position =
                transform.translation.truncate() - delta_device_pixels * world_units_per_pixel;
            transform.translation.x = proposed_position.x;
            transform.translation.y = proposed_position.y;
        }
    }

    *last_cursor = active_camera.then_some(current_cursor);
}

#[cfg(not(feature = "splatting"))]
pub(super) fn sync_view_cameras() {}
