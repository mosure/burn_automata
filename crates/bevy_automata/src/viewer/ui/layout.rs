use super::*;

pub(in crate::viewer) fn scene() -> impl SceneList {
    bsn_list![(
        Camera2d
        Camera {
            order: 1,
            clear_color: ClearColorConfig::None
        }
    ), controls_panel(), catalog_preview_modal()]
}

pub(in crate::viewer) fn controls_panel() -> impl Scene {
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

pub(in crate::viewer) fn catalog_preview_modal() -> impl Scene {
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

pub(in crate::viewer) fn catalog_preview_close_button() -> impl Scene {
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

pub(in crate::viewer) fn controls_section(label: &'static str, row: impl Scene) -> impl Scene {
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

pub(in crate::viewer) fn run_controls_row() -> impl Scene {
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

pub(in crate::viewer) fn training_controls_row() -> impl Scene {
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

pub(in crate::viewer) fn simulation_controls_row() -> impl Scene {
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

pub(in crate::viewer) fn view_controls_row() -> impl Scene {
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

pub(in crate::viewer) fn model_controls_row() -> impl Scene {
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

pub(in crate::viewer) fn model_catalog_card(key: ModelCatalogKey) -> impl Scene {
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

pub(in crate::viewer) fn slider_row(
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

pub(in crate::viewer) fn slider_widget(
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
