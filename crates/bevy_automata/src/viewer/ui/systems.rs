use super::*;

pub(in crate::viewer) fn toggle_ui_visibility(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut ui_state: ResMut<AutomataUiState>,
    mut roots: Query<&mut Visibility, With<AutomataUiRoot>>,
) {
    if !keyboard.just_pressed(KeyCode::KeyM) {
        return;
    }

    ui_state.visible = !ui_state.visible;
    let visibility = if ui_state.visible {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut root in &mut roots {
        *root = visibility;
    }
}

pub(in crate::viewer) fn handle_model_catalog_press(
    mut event: On<Pointer<Press>>,
    time: Res<Time>,
    cards: Query<&ModelCatalogCard>,
    mut settings: ResMut<AutomataSettings>,
    mut runtime: ResMut<AutomataRuntime>,
    mut preview: ResMut<CatalogPreviewState>,
    #[cfg(feature = "hyper_dino")] mut image_target: ImageTargetInteraction,
) {
    event.trigger_mut().propagate = false;
    let Ok(card) = cards.get(event.entity) else {
        return;
    };
    let now = time.elapsed_secs_f64();
    let double_click = preview.last_pressed_key == Some(card.0)
        && now - preview.last_press_time <= CATALOG_DOUBLE_CLICK_SECONDS;
    preview.last_pressed_key = Some(card.0);
    preview.last_press_time = now;
    #[cfg(feature = "hyper_dino")]
    image_target.clear_current_target();
    select_catalog_entry(card.0, &mut settings, &mut runtime);
    if double_click {
        preview.open = true;
        preview.key = Some(card.0);
        runtime.status = format!("previewing {} target", catalog_entry(card.0).title);
    }
}

pub(in crate::viewer) fn handle_slider_value_change(
    value_change: On<ValueChange<f32>>,
    sliders: Query<&AutomataSlider>,
    mut settings: ResMut<AutomataSettings>,
    mut runtime: ResMut<AutomataRuntime>,
) {
    let Ok(slider) = sliders.get(value_change.source) else {
        return;
    };
    match slider.0 {
        AutomataSliderKind::ParticleLog2 => {
            if !value_change.is_final {
                return;
            }
            let next = particles_from_slider_value(value_change.value);
            if settings.particle_count != next {
                settings.particle_count = next;
                settings.mark_changed();
                runtime.trace = None;
                runtime.adaptive = None;
                runtime.loaded_adaptive_model_path = None;
                runtime.frame = 0;
            }
        }
        AutomataSliderKind::StepsPerFrame => {
            let next = value_change.value.round().clamp(1.0, 8.0) as usize;
            if settings.steps_per_frame != next {
                settings.steps_per_frame = next;
            }
        }
        AutomataSliderKind::UpdateProb => {
            let next = value_change.value.clamp(0.0, 1.0);
            if (settings.update_prob - next).abs() > 1.0e-5 {
                settings.update_prob = next;
                settings.mark_changed();
                runtime.frame = 0;
            }
        }
        AutomataSliderKind::DtLog2 => {
            let next = exp2_slider_value(value_change.value).clamp(0.03125, 4.0);
            if (settings.dt - next).abs() > 1.0e-5 {
                settings.dt = next;
                settings.mark_changed();
                runtime.frame = 0;
            }
        }
        AutomataSliderKind::RenderScaleLog2 => {
            let next = exp2_slider_value(value_change.value).clamp(0.125, 4.0);
            if (settings.render_scale - next).abs() > 1.0e-5 {
                settings.render_scale = next;
            }
        }
        AutomataSliderKind::RenderOpacityLog2 => {
            let next = exp2_slider_value(value_change.value).clamp(0.0625, 2.0);
            if (settings.render_opacity - next).abs() > 1.0e-5 {
                settings.render_opacity = next;
            }
        }
        AutomataSliderKind::TrainingLearningRateLog2 => {
            let next = exp2_slider_value(value_change.value).clamp(1.0e-5, 0.1);
            if (settings.training_learning_rate - next).abs() > 1.0e-7 {
                settings.training_learning_rate = next;
            }
        }
        AutomataSliderKind::TrainingRolloutResetInterval => {
            let next = value_change.value.round().clamp(0.0, 1_000.0) as usize;
            if settings.training_rollout_reset_interval != next {
                settings.training_rollout_reset_interval = next;
            }
        }
    }
}

pub(in crate::viewer) fn sync_slider_values(
    settings: Res<AutomataSettings>,
    mut commands: Commands,
    sliders: Query<(Entity, &AutomataSlider, Option<&SliderValue>)>,
) {
    if !settings.is_changed() {
        return;
    }
    for (entity, slider, current) in &sliders {
        let next = slider_value_for_settings(slider.0, &settings);
        if current.is_none_or(|value| (value.0 - next).abs() > 1.0e-4) {
            commands.entity(entity).insert(SliderValue(next));
        }
    }
}

#[allow(clippy::type_complexity)]
pub(in crate::viewer) fn update_slider_visuals(
    sliders: Query<
        (
            Entity,
            &SliderValue,
            &SliderRange,
            &Hovered,
            &SliderDragState,
        ),
        (
            Or<(
                Changed<SliderValue>,
                Changed<Hovered>,
                Changed<SliderDragState>,
            )>,
            With<AutomataSlider>,
        ),
    >,
    children: Query<&Children>,
    mut thumbs: Query<(&mut Node, &mut BackgroundColor), With<AutomataSliderThumb>>,
    mut fills: Query<&mut Node, (With<AutomataSliderFill>, Without<AutomataSliderThumb>)>,
) {
    for (slider_entity, value, range, hovered, drag_state) in &sliders {
        let position = range.thumb_position(value.0).clamp(0.0, 1.0) * 100.0;
        let active = hovered.0 || drag_state.dragging;
        for child in children.iter_descendants(slider_entity) {
            if let Ok((mut node, mut background)) = thumbs.get_mut(child) {
                node.left = percent(position);
                background.0 = if active {
                    Color::srgb(0.92, 0.98, 0.90)
                } else {
                    Color::srgb(0.78, 0.88, 0.82)
                };
            }
            if let Ok(mut node) = fills.get_mut(child) {
                node.width = percent(position);
            }
        }
    }
}

pub(in crate::viewer) fn update_slider_value_labels(
    settings: Res<AutomataSettings>,
    mut labels: Query<(&AutomataSliderValueLabel, &mut Text)>,
) {
    if !settings.is_changed() {
        return;
    }
    for (label, mut text) in &mut labels {
        text.0 = slider_label(label.0, &settings);
    }
}

pub(in crate::viewer) fn update_run_control_button_styles(
    settings: Res<AutomataSettings>,
    #[cfg(feature = "hyper_dino")] target_training: Res<ImageTargetTrainingState>,
    #[cfg(feature = "hyper_dino")] inference: Res<HyperNpaInferenceState>,
    mut buttons: Query<(
        &RunControlButton,
        &Hovered,
        &mut BackgroundColor,
        &mut BorderColor,
    )>,
) {
    for (button, hovered, mut background, mut border) in &mut buttons {
        #[cfg(feature = "hyper_dino")]
        let available = button.0 != RunControlKind::Train
            || (target_training.train_action_available() && inference.pending == 0);
        #[cfg(not(feature = "hyper_dino"))]
        let available = true;
        if !available {
            background.0 = Color::srgb(0.065, 0.075, 0.085);
            *border = BorderColor::from(Color::srgb(0.16, 0.19, 0.21));
            continue;
        }
        let active = if button.0 == RunControlKind::Train {
            #[cfg(feature = "hyper_dino")]
            {
                target_training.is_training()
            }
            #[cfg(not(feature = "hyper_dino"))]
            {
                settings.train_live
            }
        } else {
            run_control_is_active(button.0, &settings)
        };
        background.0 = match (active, hovered.0) {
            (true, true) => Color::srgb(0.19, 0.36, 0.37),
            (true, false) => Color::srgb(0.14, 0.28, 0.29),
            (false, true) => Color::srgb(0.13, 0.15, 0.17),
            (false, false) => Color::srgb(0.10, 0.12, 0.14),
        };
        *border = BorderColor::from(match (active, hovered.0) {
            (true, true) => Color::srgb(0.48, 0.86, 0.78),
            (true, false) => Color::srgb(0.36, 0.70, 0.66),
            (false, true) => Color::srgb(0.36, 0.42, 0.46),
            (false, false) => Color::srgb(0.25, 0.30, 0.34),
        });
    }
}

#[cfg(feature = "hyper_dino")]
pub(in crate::viewer) fn update_hyper_image_button_styles(
    mut buttons: Query<(&Hovered, &mut BackgroundColor, &mut BorderColor), With<HyperImageButton>>,
) {
    for (hovered, mut background, mut border) in &mut buttons {
        background.0 = match hovered.0 {
            true => Color::srgb(0.12, 0.15, 0.15),
            false => Color::srgb(0.09, 0.12, 0.13),
        };
        *border = BorderColor::from(match hovered.0 {
            true => Color::srgb(0.39, 0.47, 0.48),
            false => Color::srgb(0.28, 0.35, 0.37),
        });
    }
}

#[cfg(feature = "hyper_dino")]
pub(in crate::viewer) fn update_hyper_inference_button_styles(
    inference: Res<HyperNpaInferenceState>,
    target_training: Res<ImageTargetTrainingState>,
    mut buttons: Query<
        (&Hovered, &mut BackgroundColor, &mut BorderColor),
        With<HyperInferenceButton>,
    >,
) {
    let running = inference.pending > 0;
    let available = target_training.has_target() && !target_training.is_training() && !running;
    for (hovered, mut background, mut border) in &mut buttons {
        if !available && !running {
            background.0 = Color::srgb(0.065, 0.075, 0.085);
            *border = BorderColor::from(Color::srgb(0.16, 0.19, 0.21));
        } else {
            background.0 = match (running, hovered.0) {
                (true, true) => Color::srgb(0.19, 0.34, 0.31),
                (true, false) => Color::srgb(0.13, 0.25, 0.23),
                (false, true) => Color::srgb(0.13, 0.15, 0.17),
                (false, false) => Color::srgb(0.10, 0.12, 0.14),
            };
            *border = BorderColor::from(match (running, hovered.0) {
                (true, true) => Color::srgb(0.48, 0.82, 0.70),
                (true, false) => Color::srgb(0.34, 0.64, 0.56),
                (false, true) => Color::srgb(0.36, 0.42, 0.46),
                (false, false) => Color::srgb(0.25, 0.30, 0.34),
            });
        }
    }
}

#[cfg(feature = "hyper_dino")]
pub(in crate::viewer) fn sync_hyper_inference_button_label(
    inference: Res<HyperNpaInferenceState>,
    target_training: Res<ImageTargetTrainingState>,
    mut labels: Query<(&mut Text, &mut TextColor), With<HyperInferenceButtonLabel>>,
) {
    if !inference.is_changed() && !target_training.is_changed() {
        return;
    }
    let running = inference.pending > 0;
    let available = target_training.has_target() && !target_training.is_training() && !running;
    for (mut text, mut color) in &mut labels {
        text.0 = if running { "inferring" } else { "infer" }.to_string();
        color.0 = if available || running {
            Color::srgb(0.86, 0.91, 0.94)
        } else {
            Color::srgb(0.42, 0.47, 0.50)
        };
    }
}

pub(in crate::viewer) fn run_control_is_active(
    kind: RunControlKind,
    settings: &AutomataSettings,
) -> bool {
    match kind {
        RunControlKind::Pause => settings.paused,
        RunControlKind::Reset => false,
        RunControlKind::Train => settings.train_live,
    }
}

#[cfg(feature = "hyper_dino")]
pub(in crate::viewer) fn handle_adaptive_training_toggle(
    value_change: On<ValueChange<bool>>,
    state: Res<ImageTargetTrainingState>,
    mut settings: ResMut<AutomataSettings>,
) {
    if !state.is_training() {
        settings.adaptive_training_enabled = value_change.value;
    }
}

#[cfg(feature = "hyper_dino")]
pub(in crate::viewer) fn sync_adaptive_training_checkbox(
    settings: Res<AutomataSettings>,
    mut commands: Commands,
    checkboxes: Query<(Entity, Has<Checked>), With<AdaptiveTrainingCheckbox>>,
) {
    if !settings.is_changed() {
        return;
    }
    for (entity, checked) in &checkboxes {
        if settings.adaptive_training_enabled != checked {
            let mut entity = commands.entity(entity);
            if settings.adaptive_training_enabled {
                entity.insert(Checked);
            } else {
                entity.remove::<Checked>();
            }
        }
    }
}

#[cfg(feature = "hyper_dino")]
pub(in crate::viewer) fn update_adaptive_training_checkbox_style(
    settings: Res<AutomataSettings>,
    state: Res<ImageTargetTrainingState>,
    mut checkboxes: Query<
        (&Hovered, &mut BackgroundColor, &mut BorderColor),
        With<AdaptiveTrainingCheckbox>,
    >,
    mut marks: Query<&mut BackgroundColor, With<AdaptiveTrainingCheckboxMark>>,
) {
    let available = !state.is_training();
    for (hovered, mut background, mut border) in &mut checkboxes {
        background.0 = match (available, settings.adaptive_training_enabled, hovered.0) {
            (false, _, _) => Color::srgb(0.055, 0.065, 0.072),
            (true, true, true) => Color::srgb(0.16, 0.31, 0.30),
            (true, true, false) => Color::srgb(0.12, 0.24, 0.24),
            (true, false, true) => Color::srgb(0.11, 0.13, 0.14),
            (true, false, false) => Color::srgb(0.075, 0.09, 0.10),
        };
        *border = BorderColor::from(if available {
            Color::srgb(0.32, 0.43, 0.43)
        } else {
            Color::srgb(0.16, 0.19, 0.21)
        });
    }
    for mut mark in &mut marks {
        mark.0 = if settings.adaptive_training_enabled {
            Color::srgb(0.48, 0.86, 0.76)
        } else {
            Color::NONE
        };
    }
}

pub(in crate::viewer) fn assign_catalog_thumbnails(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    thumbnails: Query<(Entity, &ModelCatalogThumbnail), Added<ModelCatalogThumbnail>>,
) {
    for (entity, thumbnail) in &thumbnails {
        let mut image = catalog_thumbnail_image(thumbnail.0);
        image.sampler = ImageSampler::linear();
        let handle = images.add(image);
        commands.entity(entity).insert(ImageNode::new(handle));
    }
}

pub(in crate::viewer) fn assign_catalog_text_fonts(
    mut commands: Commands,
    text_sizes: Query<(Entity, &ModelCatalogTextSize), Added<ModelCatalogTextSize>>,
) {
    for (entity, text_size) in &text_sizes {
        commands
            .entity(entity)
            .insert(TextFont::from_font_size(text_size.0));
    }
}

#[allow(clippy::too_many_arguments)]
pub(in crate::viewer) fn update_catalog_preview_modal(
    time: Res<Time>,
    preview: Res<CatalogPreviewState>,
    ui_state: Res<AutomataUiState>,
    mut preview_image_state: ResMut<CatalogPreviewImageState>,
    mut images: ResMut<Assets<Image>>,
    mut roots: Query<&mut Visibility, With<CatalogPreviewRoot>>,
    mut titles: Query<&mut Text, With<CatalogPreviewTitle>>,
    mut details: Query<&mut Text, (With<CatalogPreviewDetail>, Without<CatalogPreviewTitle>)>,
    mut image_nodes: Query<&mut ImageNode, With<CatalogPreviewImage>>,
) {
    let visible = ui_state.visible && preview.open && preview.key.is_some();
    let visibility = if visible {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut root in &mut roots {
        *root = visibility;
    }
    if !visible {
        return;
    }

    let Some(key) = preview.key else {
        return;
    };
    let entry = catalog_entry(key);
    for mut title in &mut titles {
        title.0 = format!("{} target", entry.title);
    }
    for mut detail in &mut details {
        detail.0 = format!(
            "{} | {} | {}",
            entry.kind, entry.detail, entry.particle_count
        );
    }

    let needs_new_handle = preview_image_state.key != Some(key)
        || preview_image_state
            .handle
            .as_ref()
            .is_none_or(|handle| !images.contains(handle));

    if needs_new_handle {
        let mut image = catalog_preview_image(key, time.elapsed_secs());
        image.sampler = ImageSampler::linear();
        let handle = images.add(image);
        preview_image_state.handle = Some(handle.clone());
        preview_image_state.key = Some(key);
        for mut image_node in &mut image_nodes {
            image_node.image = handle.clone();
        }
        return;
    }

    if matches!(
        key,
        ModelCatalogKey::UvTorusMorphogen3d | ModelCatalogKey::TeapotMorphogen3d
    ) && let Some(handle) = preview_image_state.handle.as_ref()
        && let Some(mut image) = images.get_mut(handle)
    {
        *image = catalog_preview_image(key, time.elapsed_secs());
        image.sampler = ImageSampler::linear();
    }
}

pub(in crate::viewer) fn update_catalog_card_styles(
    settings: Res<AutomataSettings>,
    mut cards: Query<(
        &ModelCatalogCard,
        &Hovered,
        &mut BackgroundColor,
        &mut BorderColor,
    )>,
) {
    for (card, hovered, mut background, mut border) in &mut cards {
        let entry = catalog_entry(card.0);
        let selected = catalog_entry_matches_settings(entry, &settings);
        let available = catalog_entry_is_available(entry);
        background.0 = if selected {
            Color::srgb(0.105, 0.15, 0.15)
        } else if !available && hovered.0 {
            Color::srgb(0.13, 0.075, 0.072)
        } else if !available {
            Color::srgb(0.075, 0.066, 0.064)
        } else if hovered.0 {
            Color::srgb(0.095, 0.112, 0.122)
        } else {
            Color::srgb(0.072, 0.084, 0.094)
        };
        *border = BorderColor::from(if selected {
            Color::srgb(0.34, 0.70, 0.66)
        } else if !available && hovered.0 {
            Color::srgb(0.58, 0.24, 0.22)
        } else if !available {
            Color::srgb(0.34, 0.20, 0.18)
        } else if hovered.0 {
            Color::srgb(0.32, 0.39, 0.42)
        } else {
            Color::srgb(0.24, 0.29, 0.32)
        });
    }
}

pub(in crate::viewer) fn scroll_ui_panel(
    ui_state: Res<AutomataUiState>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut mouse_wheel: MessageReader<MouseWheel>,
    panels: Query<(&ComputedNode, &UiGlobalTransform), With<AutomataUiPanel>>,
    mut scroll_areas: Query<&mut ScrollPosition, With<AutomataUiScrollArea>>,
) {
    if !ui_state.visible {
        return;
    }
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    if !panels
        .iter()
        .any(|(node, transform)| node.contains_point(*transform, cursor))
    {
        return;
    }

    let mut scroll_delta = 0.0;
    for event in mouse_wheel.read() {
        let unit_scale = match event.unit {
            MouseScrollUnit::Line => 48.0,
            MouseScrollUnit::Pixel => 1.0,
        };
        scroll_delta += event.y * unit_scale;
    }
    if scroll_delta == 0.0 {
        return;
    }

    for mut scroll_position in &mut scroll_areas {
        scroll_position.0.y = (scroll_position.0.y - scroll_delta).max(0.0);
    }
}
