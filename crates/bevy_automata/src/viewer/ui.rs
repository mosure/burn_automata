use super::*;

#[derive(Component, Clone, Debug, Default)]
pub(super) struct StatusLabel;

#[derive(Component, Clone, Debug, Default)]
pub(super) struct SettingsLabel;

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum AutomataSliderKind {
    #[default]
    ParticleLog2,
    StepsPerFrame,
    UpdateProb,
    DtLog2,
    RenderScaleLog2,
    RenderOpacityLog2,
    TrainingLearningRateLog2,
}

#[derive(Component, Clone, Copy, Debug, Default)]
pub(super) struct AutomataSlider(AutomataSliderKind);

#[derive(Component, Clone, Copy, Debug, Default)]
pub(super) struct AutomataSliderValueLabel(AutomataSliderKind);

#[derive(Component, Clone, Debug, Default)]
pub(super) struct AutomataSliderThumb;

#[derive(Component, Clone, Debug, Default)]
pub(super) struct AutomataSliderFill;

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum RunControlKind {
    #[default]
    Pause,
    Reset,
    Backward,
    Train,
}

#[derive(Component, Clone, Copy, Debug, Default)]
pub(super) struct RunControlButton(RunControlKind);

#[derive(Component, Clone, Debug, Default)]
pub(super) struct AutomataUiPanel;

#[derive(Component, Clone, Debug, Default)]
pub(super) struct AutomataUiRoot;

#[derive(Component, Clone, Debug, Default)]
pub(super) struct AutomataUiScrollArea;

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ModelCatalogCard(ModelCatalogKey);

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ModelCatalogThumbnail(ModelCatalogKey);

#[derive(Component, Clone, Copy, Debug, Default)]
pub(super) struct ModelCatalogTextSize(f32);

#[derive(Component, Clone, Debug, Default)]
pub(super) struct CatalogPreviewRoot;

#[derive(Component, Clone, Debug, Default)]
pub(super) struct CatalogPreviewTitle;

#[derive(Component, Clone, Debug, Default)]
pub(super) struct CatalogPreviewDetail;

#[derive(Component, Clone, Debug, Default)]
pub(super) struct CatalogPreviewImage;

#[derive(Resource, Clone, Debug, Default)]
pub(super) struct CatalogPreviewState {
    pub(super) open: bool,
    key: Option<ModelCatalogKey>,
    last_pressed_key: Option<ModelCatalogKey>,
    last_press_time: f64,
}

#[derive(Resource, Clone, Debug, Default)]
pub(super) struct CatalogPreviewImageState {
    handle: Option<Handle<Image>>,
    key: Option<ModelCatalogKey>,
}

#[cfg(feature = "splatting")]
#[derive(Resource, Clone, Debug, Default)]
pub(super) struct AutomataUiInputCapture {
    pub(super) active: bool,
}

#[derive(Resource, Clone, Debug)]
pub(super) struct AutomataUiState {
    pub(super) visible: bool,
}

impl Default for AutomataUiState {
    fn default() -> Self {
        Self { visible: true }
    }
}

pub(super) fn scene() -> impl SceneList {
    bsn_list![(
        Camera2d
        Camera {
            order: 1,
            clear_color: ClearColorConfig::None
        }
    ), controls_panel(), catalog_preview_modal()]
}

pub(super) fn controls_panel() -> impl Scene {
    bsn! {
        Node {
            width: px(AUTOMATA_UI_PANEL_WIDTH),
            height: percent(100),
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Stretch,
            justify_content: JustifyContent::Start,
            overflow: Overflow::scroll_y(),
            scrollbar_width: 8.0,
            padding: px(14),
            row_gap: px(8),
        }
        BackgroundColor(Color::srgba(0.035, 0.04, 0.045, 0.88))
        ScrollPosition(Vec2::ZERO)
        AutomataUiRoot
        AutomataUiPanel
        AutomataUiScrollArea
        Children [
            (
                Text("burn_automata")
                TextColor(Color::srgb(0.92, 0.95, 0.98))
            ),
            (
                Text("status loading")
                template_value(ModelCatalogTextSize(13.0))
                TextColor(Color::srgb(0.84, 0.88, 0.76))
                StatusLabel
            ),
            controls_section("run", run_controls_row()),
            controls_section("training", training_controls_row()),
            controls_section("simulation", simulation_controls_row()),
            controls_section("view", view_controls_row()),
            controls_section("model", model_controls_row()),
            (
                Text("settings loading")
                template_value(ModelCatalogTextSize(13.0))
                TextColor(Color::srgb(0.72, 0.77, 0.82))
                SettingsLabel
            ),
            (
                Node {
                    height: px(1),
                    width: percent(100),
                    margin: UiRect::vertical(px(4)),
                }
                BackgroundColor(Color::srgb(0.20, 0.23, 0.26))
            ),
        ]
    }
}

pub(super) fn catalog_preview_modal() -> impl Scene {
    bsn! {
        Node {
            position_type: PositionType::Absolute,
            left: px(0),
            right: px(0),
            top: px(0),
            bottom: px(0),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            padding: px(18),
        }
        Visibility::Hidden
        AutomataUiRoot
        CatalogPreviewRoot
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.48))
        Children [(
            Node {
                width: px(430),
                max_width: percent(92),
                height: px(330),
                max_height: percent(84),
                border: px(1),
                border_radius: BorderRadius::all(px(8)),
                padding: px(14),
                flex_direction: FlexDirection::Column,
                row_gap: px(10),
                align_items: AlignItems::Stretch,
            }
            BorderColor::from(Color::srgb(0.26, 0.33, 0.36))
            BackgroundColor(Color::srgb(0.035, 0.042, 0.048))
            Children [
                (
                    Node {
                        width: percent(100),
                        flex_direction: FlexDirection::Row,
                        column_gap: px(8),
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::SpaceBetween,
                    }
                    Children [
                        (
                            Text("target")
                            template_value(ModelCatalogTextSize(14.0))
                            TextColor(Color::srgb(0.91, 0.95, 0.96))
                            CatalogPreviewTitle
                        ),
                        catalog_preview_close_button(),
                    ]
                ),
                (
                    Node {
                        width: percent(100),
                        height: px(232),
                        border: px(1),
                        border_radius: BorderRadius::all(px(6)),
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        overflow: Overflow::clip(),
                    }
                    BorderColor::from(Color::srgb(0.16, 0.20, 0.22))
                    BackgroundColor(Color::srgb(0.015, 0.019, 0.023))
                    ImageNode::default()
                    CatalogPreviewImage
                ),
                (
                    Text("model target")
                    template_value(ModelCatalogTextSize(12.0))
                    TextColor(Color::srgb(0.63, 0.70, 0.74))
                    CatalogPreviewDetail
                ),
            ]
        )]
    }
}

pub(super) fn catalog_preview_close_button() -> impl Scene {
    bsn! {
        Button
        Node {
            width: px(64),
            height: px(28),
            border: px(1),
            border_radius: BorderRadius::all(px(6)),
            padding: UiRect::horizontal(px(8)),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
        }
        BorderColor::from(Color::srgb(0.28, 0.34, 0.37))
        BackgroundColor(Color::srgb(0.09, 0.11, 0.125))
        on(|mut event: On<Pointer<Press>>, mut preview: ResMut<CatalogPreviewState>| {
            event.trigger_mut().propagate = false;
            preview.open = false;
        })
        Children [(
            Text("close")
            template_value(ModelCatalogTextSize(12.0))
            TextColor(Color::srgb(0.86, 0.91, 0.93))
        )]
    }
}

pub(super) fn controls_section(label: &'static str, row: impl Scene) -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            flex_direction: FlexDirection::Column,
            row_gap: px(5),
        }
        Children [
            (
                Text(label)
                TextColor(Color::srgb(0.48, 0.56, 0.62))
            ),
            row,
        ]
    }
}

pub(super) fn run_controls_row() -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            flex_direction: FlexDirection::Row,
            flex_wrap: FlexWrap::Wrap,
            column_gap: px(6),
            row_gap: px(6),
            align_items: AlignItems::Center,
        }
        Children [
            pause_button(),
            reset_button(),
            backward_button(),
            train_button(),
        ]
    }
}

pub(super) fn training_controls_row() -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            flex_direction: FlexDirection::Column,
            row_gap: px(6),
        }
        Children [
            (
                Text("target rollout teacher | 256 rows | 60f")
                template_value(ModelCatalogTextSize(12.0))
                TextColor(Color::srgb(0.56, 0.64, 0.68))
            ),
            slider_row("train lr", "0.0010", AutomataSliderKind::TrainingLearningRateLog2, log2_slider_value(1.0e-3), -16.0, -4.0, 0.125),
        ]
    }
}

pub(super) fn simulation_controls_row() -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            flex_direction: FlexDirection::Column,
            row_gap: px(8),
        }
        Children [
            slider_row("particles", "4096", AutomataSliderKind::ParticleLog2, particle_slider_value(4096), 6.0, 14.0, 1.0),
            slider_row("steps/frame", "1", AutomataSliderKind::StepsPerFrame, 1.0, 1.0, 8.0, 1.0),
            slider_row("update prob", "0.50", AutomataSliderKind::UpdateProb, 0.5, 0.0, 1.0, 0.05),
            slider_row("dt", "1.000", AutomataSliderKind::DtLog2, 0.0, -5.0, 2.0, 0.125),
        ]
    }
}

pub(super) fn view_controls_row() -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            flex_direction: FlexDirection::Column,
            row_gap: px(8),
        }
        Children [
            slider_row("splat scale", "0.50x", AutomataSliderKind::RenderScaleLog2, -1.0, -5.0, 2.0, 0.0625),
            slider_row("splat opacity", "2.00x", AutomataSliderKind::RenderOpacityLog2, 1.0, -4.0, 1.0, 0.0625),
        ]
    }
}

pub(super) fn model_controls_row() -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            flex_direction: FlexDirection::Column,
            row_gap: px(8),
        }
        Children [
            (
                Node {
                    width: percent(100),
                    flex_direction: FlexDirection::Row,
                    column_gap: px(6),
                    align_items: AlignItems::Center,
                }
                Children [
                    (
                        Text("catalog")
                        template_value(ModelCatalogTextSize(12.0))
                        TextColor(Color::srgb(0.66, 0.72, 0.76))
                    ),
                    (
                        Text("select a model; view settings persist")
                        template_value(ModelCatalogTextSize(12.0))
                        TextColor(Color::srgb(0.42, 0.49, 0.53))
                    ),
                ]
            ),
            (
                Node {
                    width: percent(100),
                    flex_direction: FlexDirection::Row,
                    flex_wrap: FlexWrap::Wrap,
                    column_gap: px(8),
                    row_gap: px(8),
                    align_items: AlignItems::Stretch,
                }
                Children [
                    model_catalog_card(ModelCatalogKey::Lizard),
                    model_catalog_card(ModelCatalogKey::Butterfly),
                    model_catalog_card(ModelCatalogKey::Rose),
                    model_catalog_card(ModelCatalogKey::Turtle),
                    model_catalog_card(ModelCatalogKey::Mushroom),
                    model_catalog_card(ModelCatalogKey::TropicalFish),
                    model_catalog_card(ModelCatalogKey::Sun),
                    model_catalog_card(ModelCatalogKey::Ghost),
                    model_catalog_card(ModelCatalogKey::Frog),
                    model_catalog_card(ModelCatalogKey::Apple),
                    model_catalog_card(ModelCatalogKey::Polka),
                    model_catalog_card(ModelCatalogKey::Bubbly),
                    model_catalog_card(ModelCatalogKey::Clouds),
                    model_catalog_card(ModelCatalogKey::Galaxy),
                    model_catalog_card(ModelCatalogKey::Hearts),
                    model_catalog_card(ModelCatalogKey::Rings),
                    model_catalog_card(ModelCatalogKey::Stars),
                    model_catalog_card(ModelCatalogKey::Grid),
                    model_catalog_card(ModelCatalogKey::Banded),
                    model_catalog_card(ModelCatalogKey::Tree),
                    model_catalog_card(ModelCatalogKey::Snow),
                    model_catalog_card(ModelCatalogKey::Digit0),
                    model_catalog_card(ModelCatalogKey::LetterA),
                    model_catalog_card(ModelCatalogKey::Growing2d),
                    model_catalog_card(ModelCatalogKey::Texture2d),
                    model_catalog_card(ModelCatalogKey::Growing3dGs),
                    model_catalog_card(ModelCatalogKey::PointMnist),
                ]
            ),
        ]
    }
}

pub(super) fn model_catalog_card(key: ModelCatalogKey) -> impl Scene {
    let entry = catalog_entry(key);
    let title = entry.title;
    bsn! {
        Button
        Node {
            width: px(72),
            height: px(72),
            border: px(1),
            border_radius: BorderRadius::all(px(6)),
            padding: px(6),
            flex_direction: FlexDirection::Column,
            row_gap: px(4),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
        }
        template_value(ModelCatalogCard(key))
        template_value(Hovered::default())
        BorderColor::from(Color::srgb(0.24, 0.29, 0.32))
        BackgroundColor(Color::srgb(0.072, 0.084, 0.094))
        on(handle_model_catalog_press)
        Children [
            (
                Node {
                    width: px(44),
                    height: px(36),
                    border_radius: BorderRadius::all(px(6)),
                    border: px(1),
                    flex_shrink: 0.0,
                }
                BorderColor::from(Color::srgb(0.15, 0.18, 0.20))
                BackgroundColor(Color::srgb(0.018, 0.022, 0.026))
                ImageNode::default()
                template_value(ModelCatalogThumbnail(key))
            ),
            (
                Node {
                    width: percent(100),
                    align_items: AlignItems::Center,
                    justify_content: JustifyContent::Center,
                }
                Children [(
                    Text(title)
                    template_value(ModelCatalogTextSize(8.0))
                    TextColor(Color::srgb(0.86, 0.91, 0.93))
                )]
            ),
        ]
    }
}

pub(super) fn slider_row(
    label: &'static str,
    value_text: &'static str,
    kind: AutomataSliderKind,
    value: f32,
    min: f32,
    max: f32,
    step: f32,
) -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            height: px(42),
            flex_direction: FlexDirection::Row,
            column_gap: px(10),
            align_items: AlignItems::Center,
        }
        Children [
            (
                Node {
                    width: px(104),
                    align_items: AlignItems::Center,
                }
                Children [(
                    Text(label)
                    TextColor(Color::srgb(0.70, 0.77, 0.82))
                )]
            ),
            slider_widget(kind, value, min, max, step),
            (
                Node {
                    width: px(78),
                    align_items: AlignItems::Center,
                    justify_content: JustifyContent::End,
                }
                Children [(
                    Text(value_text)
                    TextColor(Color::srgb(0.88, 0.92, 0.89))
                    template_value(AutomataSliderValueLabel(kind))
                )]
            ),
        ]
    }
}

pub(super) fn slider_widget(
    kind: AutomataSliderKind,
    value: f32,
    min: f32,
    max: f32,
    step: f32,
) -> impl Scene {
    bsn! {
        Node {
            height: px(22),
            flex_grow: 1.0,
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Stretch,
        }
        template_value(Hovered::default())
        Slider {
            track_click: TrackClick::Drag,
            orientation: SliderOrientation::Horizontal,
        }
        SliderValue(value)
        SliderRange::new(min, max)
        SliderStep(step)
        template_value(AutomataSlider(kind))
        on(slider_self_update)
        on(handle_slider_value_change)
        Children [
            (
                Node {
                    height: px(6),
                    width: percent(100),
                    border_radius: BorderRadius::all(px(3)),
                    align_self: AlignSelf::Center,
                }
                BackgroundColor(Color::srgb(0.07, 0.085, 0.095))
                Children [(
                    Node {
                        width: percent(0),
                        height: percent(100),
                        border_radius: BorderRadius::all(px(3)),
                    }
                    BackgroundColor(Color::srgb(0.28, 0.56, 0.62))
                    AutomataSliderFill
                )]
            ),
            (
                Node {
                    position_type: PositionType::Absolute,
                    left: px(0),
                    right: px(12),
                    top: px(0),
                    bottom: px(0),
                }
                Children [(
                    SliderThumb
                    AutomataSliderThumb
                    Node {
                        width: px(12),
                        height: px(12),
                        position_type: PositionType::Absolute,
                        left: percent(0),
                        align_self: AlignSelf::Center,
                        border_radius: BorderRadius::MAX,
                    }
                    BackgroundColor(Color::srgb(0.78, 0.88, 0.82))
                )]
            ),
        ]
    }
}

pub(super) fn control_button(label: &'static str, kind: RunControlKind) -> impl Scene {
    bsn! {
        Button
        Node {
            width: px(104),
            max_width: percent(48),
            flex_grow: 1.0,
            flex_shrink: 1.0,
            height: px(30),
            border: px(1),
            padding: UiRect::horizontal(px(8)),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
        }
        template_value(RunControlButton(kind))
        template_value(Hovered::default())
        BorderColor::from(Color::srgb(0.25, 0.30, 0.34))
        BackgroundColor(Color::srgb(0.10, 0.12, 0.14))
        Children [(
            Text(label)
            template_value(ModelCatalogTextSize(12.0))
            TextColor(Color::srgb(0.86, 0.91, 0.94))
        )]
    }
}

pub(super) fn pause_button() -> impl Scene {
    bsn! {
        control_button("pause", RunControlKind::Pause)
        on(|_event: On<Pointer<Press>>, mut settings: ResMut<AutomataSettings>| {
            settings.paused = !settings.paused;
        })
    }
}

pub(super) fn reset_button() -> impl Scene {
    bsn! {
        control_button("reset", RunControlKind::Reset)
        on(|_event: On<Pointer<Press>>, mut settings: ResMut<AutomataSettings>, mut runtime: ResMut<AutomataRuntime>| {
            settings.mark_changed();
            runtime.trace = None;
            runtime.frame = 0;
            runtime.status = "reset requested".to_string();
        })
    }
}

pub(super) fn backward_button() -> impl Scene {
    bsn! {
        control_button("backward", RunControlKind::Backward)
        on(|_event: On<Pointer<Press>>, mut settings: ResMut<AutomataSettings>, mut runtime: ResMut<AutomataRuntime>| {
            settings.visualize_backward = !settings.visualize_backward;
            if settings.visualize_backward {
                match probe_trace_for_controls(&runtime, &settings, BACKWARD_PROBE_PARTICLES) {
                    Ok(trace) => {
                        let hashgrid = effective_hashgrid(&runtime, &settings);
                        update_backward_probe(&mut runtime, &trace, &hashgrid);
                        if let (Some(loss), Some(grad_norm)) = (runtime.backward_loss, runtime.backward_grad_norm) {
                            runtime.status = format!("backward probe on | loss {loss:.5} | grad {grad_norm:.5}");
                        }
                    }
                    Err(err) => {
                        settings.visualize_backward = false;
                        runtime.backward_loss = None;
                        runtime.backward_grad_norm = None;
                        runtime.status = format!("backward probe failed: {err}");
                    }
                }
            } else {
                runtime.backward_loss = None;
                runtime.backward_grad_norm = None;
                runtime.status = "backward probe off".to_string();
            }
        })
    }
}

pub(super) fn train_button() -> impl Scene {
    bsn! {
        control_button("train", RunControlKind::Train)
        on(|_event: On<Pointer<Press>>, mut settings: ResMut<AutomataSettings>, mut runtime: ResMut<AutomataRuntime>| {
            settings.train_live = !settings.train_live;
            if settings.train_live {
                runtime.training_teacher = Some(runtime.model.clone());
                match probe_trace_for_controls(&runtime, &settings, TRAINING_PROBE_PARTICLES) {
                    Ok(trace) => {
                        let hashgrid = effective_hashgrid(&runtime, &settings);
                        update_training_probe(
                            &mut runtime,
                            &trace,
                            &hashgrid,
                            settings.training_learning_rate,
                        );
                        if runtime.training_loss.is_none() {
                            settings.train_live = false;
                            runtime.training_teacher = None;
                        }
                    }
                    Err(err) => {
                        settings.train_live = false;
                        runtime.training_teacher = None;
                        runtime.training_loss = None;
                        runtime.training_grad_norm = None;
                        runtime.status = format!("training probe failed: {err}");
                    }
                }
            } else {
                runtime.training_teacher = None;
                runtime.status = format!("live training paused at step {}", runtime.training_step);
            }
        })
    }
}

pub(super) fn toggle_ui_visibility(
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

pub(super) fn handle_model_catalog_press(
    mut event: On<Pointer<Press>>,
    time: Res<Time>,
    cards: Query<&ModelCatalogCard>,
    mut settings: ResMut<AutomataSettings>,
    mut runtime: ResMut<AutomataRuntime>,
    mut preview: ResMut<CatalogPreviewState>,
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
    select_catalog_entry(card.0, &mut settings, &mut runtime);
    if double_click {
        preview.open = true;
        preview.key = Some(card.0);
        runtime.status = format!("previewing {} target", catalog_entry(card.0).title);
    }
}

pub(super) fn handle_slider_value_change(
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
    }
}

pub(super) fn sync_slider_values(
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
pub(super) fn update_slider_visuals(
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

pub(super) fn update_slider_value_labels(
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

pub(super) fn update_run_control_button_styles(
    settings: Res<AutomataSettings>,
    mut buttons: Query<(
        &RunControlButton,
        &Hovered,
        &mut BackgroundColor,
        &mut BorderColor,
    )>,
) {
    for (button, hovered, mut background, mut border) in &mut buttons {
        let active = run_control_is_active(button.0, &settings);
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

pub(super) fn run_control_is_active(kind: RunControlKind, settings: &AutomataSettings) -> bool {
    match kind {
        RunControlKind::Pause => settings.paused,
        RunControlKind::Reset => false,
        RunControlKind::Backward => settings.visualize_backward,
        RunControlKind::Train => settings.train_live,
    }
}

pub(super) fn assign_catalog_thumbnails(
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

pub(super) fn assign_catalog_text_fonts(
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
pub(super) fn update_catalog_preview_modal(
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

pub(super) fn update_catalog_card_styles(
    settings: Res<AutomataSettings>,
    mut cards: Query<(
        &ModelCatalogCard,
        &Hovered,
        &mut BackgroundColor,
        &mut BorderColor,
    )>,
) {
    for (card, hovered, mut background, mut border) in &mut cards {
        let selected = catalog_entry_matches_settings(catalog_entry(card.0), &settings);
        background.0 = if selected {
            Color::srgb(0.105, 0.15, 0.15)
        } else if hovered.0 {
            Color::srgb(0.095, 0.112, 0.122)
        } else {
            Color::srgb(0.072, 0.084, 0.094)
        };
        *border = BorderColor::from(if selected {
            Color::srgb(0.34, 0.70, 0.66)
        } else if hovered.0 {
            Color::srgb(0.32, 0.39, 0.42)
        } else {
            Color::srgb(0.24, 0.29, 0.32)
        });
    }
}

pub(super) fn scroll_ui_panel(
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

pub(super) fn slider_value_for_settings(
    kind: AutomataSliderKind,
    settings: &AutomataSettings,
) -> f32 {
    match kind {
        AutomataSliderKind::ParticleLog2 => particle_slider_value(settings.particle_count),
        AutomataSliderKind::StepsPerFrame => settings.steps_per_frame as f32,
        AutomataSliderKind::UpdateProb => settings.update_prob,
        AutomataSliderKind::DtLog2 => log2_slider_value(settings.dt),
        AutomataSliderKind::RenderScaleLog2 => log2_slider_value(settings.render_scale),
        AutomataSliderKind::RenderOpacityLog2 => log2_slider_value(settings.render_opacity),
        AutomataSliderKind::TrainingLearningRateLog2 => {
            log2_slider_value(settings.training_learning_rate)
        }
    }
}

pub(super) fn slider_label(kind: AutomataSliderKind, settings: &AutomataSettings) -> String {
    match kind {
        AutomataSliderKind::ParticleLog2 => settings.particle_count.to_string(),
        AutomataSliderKind::StepsPerFrame => settings.steps_per_frame.to_string(),
        AutomataSliderKind::UpdateProb => format!("{:.2}", settings.update_prob),
        AutomataSliderKind::DtLog2 => format!("{:.3}", settings.dt),
        AutomataSliderKind::RenderScaleLog2 => format!("{:.2}x", settings.render_scale),
        AutomataSliderKind::RenderOpacityLog2 => format!("{:.2}x", settings.render_opacity),
        AutomataSliderKind::TrainingLearningRateLog2 => {
            format!("{:.4}", settings.training_learning_rate)
        }
    }
}

pub(super) fn log2_slider_value(value: f32) -> f32 {
    value.max(f32::MIN_POSITIVE).log2()
}

pub(super) fn exp2_slider_value(value: f32) -> f32 {
    2.0_f32.powf(value)
}

pub(super) fn particle_slider_value(particles: usize) -> f32 {
    (particles.max(64) as f32).log2().clamp(6.0, 16.0)
}

pub(super) fn particles_from_slider_value(value: f32) -> usize {
    let log2 = value.round().clamp(6.0, 16.0) as u32;
    1usize << log2
}
