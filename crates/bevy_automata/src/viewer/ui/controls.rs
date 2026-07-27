use super::*;

pub(in crate::viewer) fn control_button(label: &'static str, kind: RunControlKind) -> impl Scene {
    bsn! {
        Button
        Node {
            width: px(86),
            max_width: percent(32),
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
            template_value(RunControlButtonLabel(kind))
            TextColor(Color::srgb(0.86, 0.91, 0.94))
        )]
    }
}

pub(in crate::viewer) fn pause_button() -> impl Scene {
    bsn! {
        control_button("pause", RunControlKind::Pause)
        on(|_event: On<Pointer<Press>>, mut settings: ResMut<AutomataSettings>| {
            settings.paused = !settings.paused;
        })
    }
}

pub(in crate::viewer) fn reset_button() -> impl Scene {
    bsn! {
        control_button("reset", RunControlKind::Reset)
        on(|_event: On<Pointer<Press>>,
            mut settings: ResMut<AutomataSettings>,
            mut runtime: ResMut<AutomataRuntime>| {
            settings.mark_changed();
            runtime.trace = None;
            runtime.adaptive = None;
            runtime.loaded_adaptive_model_path = None;
            runtime.frame = 0;
            runtime.status = "displayed rollout reset; training continues independently".to_string();
        })
    }
}

#[cfg(all(feature = "hyper_dino", feature = "mesh_training"))]
pub(in crate::viewer) fn train_button() -> impl Scene {
    bsn! {
        control_button("train", RunControlKind::Train)
        on(|_event: On<Pointer<Press>>,
            image_state: Res<ImageTargetTrainingState>,
            mesh_state: Res<MeshTargetTrainingState>,
            inference: Res<HyperNpaInferenceState>,
            mut image_training: MessageWriter<ToggleImageTargetTraining>,
            mut mesh_training: MessageWriter<ToggleMeshTargetTraining>| {
            if mesh_state.has_target() {
                if mesh_state.train_action_available() {
                    mesh_training.write(ToggleMeshTargetTraining);
                }
            } else if image_state.train_action_available() && inference.pending == 0 {
                image_training.write(ToggleImageTargetTraining);
            }
        })
    }
}

#[cfg(all(feature = "hyper_dino", not(feature = "mesh_training")))]
pub(in crate::viewer) fn train_button() -> impl Scene {
    bsn! {
        control_button("train", RunControlKind::Train)
        on(|_event: On<Pointer<Press>>,
            state: Res<ImageTargetTrainingState>,
            inference: Res<HyperNpaInferenceState>,
            mut training: MessageWriter<ToggleImageTargetTraining>| {
            if state.train_action_available() && inference.pending == 0 {
                training.write(ToggleImageTargetTraining);
            }
        })
    }
}

#[cfg(all(not(feature = "hyper_dino"), feature = "mesh_training"))]
pub(in crate::viewer) fn train_button() -> impl Scene {
    bsn! {
        control_button("train 3d", RunControlKind::Train)
        on(|_event: On<Pointer<Press>>,
            state: Res<MeshTargetTrainingState>,
            mut training: MessageWriter<ToggleMeshTargetTraining>| {
            if state.train_action_available() {
                training.write(ToggleMeshTargetTraining);
            }
        })
    }
}

#[cfg(not(any(feature = "hyper_dino", feature = "mesh_training")))]
pub(in crate::viewer) fn train_button() -> impl Scene {
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

#[cfg(any(feature = "hyper_dino", feature = "mesh_training"))]
pub(in crate::viewer) fn run_train_button() -> impl Scene {
    bsn! {
        Node {
            display: Display::None,
        }
    }
}

#[cfg(not(any(feature = "hyper_dino", feature = "mesh_training")))]
pub(in crate::viewer) fn run_train_button() -> impl Scene {
    train_button()
}

#[cfg(feature = "hyper_dino")]
pub(in crate::viewer) fn hyper_image_button() -> impl Scene {
    bsn! {
        Button
        Node {
            width: px(82),
            flex_grow: 1.0,
            max_width: percent(31),
            height: px(30),
            border: px(1),
            padding: UiRect::horizontal(px(10)),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
        }
        template_value(HyperImageButton)
        template_value(Hovered::default())
        BorderColor::from(Color::srgb(0.28, 0.35, 0.37))
        BackgroundColor(Color::srgb(0.09, 0.12, 0.13))
        on(|mut event: On<Pointer<Press>>, mut requests: MessageWriter<OpenHyperNpaImage>| {
            event.trigger_mut().propagate = false;
            requests.write(OpenHyperNpaImage);
        })
        Children [(
            Text("open image")
            template_value(ModelCatalogTextSize(12.0))
            TextColor(Color::srgb(0.86, 0.92, 0.90))
        )]
    }
}

#[cfg(feature = "mesh_training")]
pub(in crate::viewer) fn mesh_open_button() -> impl Scene {
    bsn! {
        Button
        Node {
            width: px(112),
            flex_grow: 1.0,
            max_width: percent(49),
            height: px(30),
            border: px(1),
            padding: UiRect::horizontal(px(10)),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
        }
        template_value(MeshOpenButton)
        template_value(Hovered::default())
        BorderColor::from(Color::srgb(0.25, 0.30, 0.34))
        BackgroundColor(Color::srgb(0.10, 0.12, 0.14))
        on(|mut event: On<Pointer<Press>>, mut requests: MessageWriter<OpenNpaMesh>| {
            event.trigger_mut().propagate = false;
            requests.write(OpenNpaMesh);
        })
        Children [(
            Text("open mesh")
            template_value(ModelCatalogTextSize(12.0))
            TextColor(Color::srgb(0.86, 0.91, 0.94))
        )]
    }
}

#[cfg(feature = "hyper_dino")]
pub(in crate::viewer) fn hyper_inference_button() -> impl Scene {
    bsn! {
        Button
        Node {
            width: px(82),
            flex_grow: 1.0,
            max_width: percent(31),
            height: px(30),
            border: px(1),
            padding: UiRect::horizontal(px(8)),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
        }
        template_value(HyperInferenceButton)
        template_value(Hovered::default())
        BorderColor::from(Color::srgb(0.25, 0.30, 0.34))
        BackgroundColor(Color::srgb(0.10, 0.12, 0.14))
        on(|mut event: On<Pointer<Press>>,
            state: Res<ImageTargetTrainingState>,
            inference: Res<HyperNpaInferenceState>,
            mut requests: MessageWriter<RunHyperNpaInference>| {
            event.trigger_mut().propagate = false;
            if state.has_target() && !state.is_training() && inference.pending == 0 {
                requests.write(RunHyperNpaInference);
            }
        })
        Children [(
            Text("infer")
            template_value(ModelCatalogTextSize(12.0))
            template_value(HyperInferenceButtonLabel)
            TextColor(Color::srgb(0.42, 0.47, 0.50))
        )]
    }
}

#[cfg(all(feature = "hyper_dino", feature = "mesh_training"))]
pub(in crate::viewer) fn image_training_actions_row() -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            flex_direction: FlexDirection::Column,
            row_gap: px(6),
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
                    hyper_image_button(),
                    mesh_open_button(),
                ]
            ),
            (
                Node {
                    width: percent(100),
                    flex_direction: FlexDirection::Row,
                    column_gap: px(6),
                    align_items: AlignItems::Center,
                }
                Children [
                    hyper_inference_button(),
                    train_button(),
                ]
            ),
        ]
    }
}

#[cfg(all(feature = "hyper_dino", not(feature = "mesh_training")))]
pub(in crate::viewer) fn image_training_actions_row() -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            flex_direction: FlexDirection::Row,
            column_gap: px(6),
            align_items: AlignItems::Center,
        }
        Children [
            hyper_image_button(),
            hyper_inference_button(),
            train_button(),
        ]
    }
}

#[cfg(feature = "hyper_dino")]
pub(in crate::viewer) fn adaptive_training_toggle() -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            height: px(28),
            flex_direction: FlexDirection::Row,
            column_gap: px(8),
            align_items: AlignItems::Center,
        }
        Children [
            (
                Checkbox
                Node {
                    width: px(18),
                    height: px(18),
                    border: px(1),
                    align_items: AlignItems::Center,
                    justify_content: JustifyContent::Center,
                }
                AdaptiveTrainingCheckbox
                Hovered::default()
                BorderColor::from(Color::srgb(0.32, 0.39, 0.42))
                BackgroundColor(Color::srgb(0.075, 0.09, 0.10))
                on(handle_adaptive_training_toggle)
                Children [(
                    Node {
                        width: px(10),
                        height: px(10),
                    }
                    AdaptiveTrainingCheckboxMark
                    BackgroundColor(Color::NONE)
                )]
            ),
            (
                Text("adaptive training")
                template_value(ModelCatalogTextSize(12.0))
                TextColor(Color::srgb(0.72, 0.78, 0.81))
            ),
        ]
    }
}

pub(in crate::viewer) fn pca_visualization_toggle() -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            height: px(28),
            flex_direction: FlexDirection::Row,
            column_gap: px(8),
            align_items: AlignItems::Center,
        }
        Children [
            (
                Checkbox
                Node {
                    width: px(18),
                    height: px(18),
                    border: px(1),
                    align_items: AlignItems::Center,
                    justify_content: JustifyContent::Center,
                }
                PcaVisualizationCheckbox
                Hovered::default()
                BorderColor::from(Color::srgb(0.32, 0.39, 0.42))
                BackgroundColor(Color::srgb(0.075, 0.09, 0.10))
                on(handle_pca_visualization_toggle)
                Children [(
                    Node {
                        width: px(10),
                        height: px(10),
                    }
                    PcaVisualizationCheckboxMark
                    BackgroundColor(Color::NONE)
                )]
            ),
            (
                Text("particle state PCA")
                template_value(ModelCatalogTextSize(12.0))
                TextColor(Color::srgb(0.72, 0.78, 0.81))
            ),
        ]
    }
}

#[cfg(not(feature = "hyper_dino"))]
pub(in crate::viewer) fn adaptive_training_toggle() -> impl Scene {
    bsn! {
        Node {
            display: Display::None,
        }
    }
}

#[cfg(all(not(feature = "hyper_dino"), feature = "mesh_training"))]
pub(in crate::viewer) fn image_training_actions_row() -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            flex_direction: FlexDirection::Row,
            column_gap: px(6),
            align_items: AlignItems::Center,
        }
        Children [
            mesh_open_button(),
            train_button(),
        ]
    }
}

#[cfg(not(any(feature = "hyper_dino", feature = "mesh_training")))]
pub(in crate::viewer) fn image_training_actions_row() -> impl Scene {
    bsn! {
        Node {
            display: Display::None,
        }
    }
}

#[cfg(feature = "hyper_dino")]
pub(in crate::viewer) fn image_target_summary() -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            min_height: px(58),
            flex_direction: FlexDirection::Row,
            column_gap: px(10),
            align_items: AlignItems::Center,
        }
        Visibility::Hidden
        ImageTargetSummary
        Children [
            (
                Node {
                    width: px(56),
                    height: px(56),
                    flex_shrink: 0.0,
                    border: px(1),
                    overflow: Overflow::clip(),
                }
                BorderColor::from(Color::srgb(0.20, 0.26, 0.27))
                BackgroundColor(Color::srgb(0.025, 0.032, 0.034))
                ImageNode::default()
                ImageTargetPreview
            ),
            (
                Node {
                    min_width: px(0),
                    flex_grow: 1.0,
                    flex_direction: FlexDirection::Column,
                    row_gap: px(4),
                    justify_content: JustifyContent::Center,
                }
                Children [
                    (
                        Text("")
                        template_value(ModelCatalogTextSize(12.0))
                        TextColor(Color::srgb(0.86, 0.92, 0.90))
                        ImageTargetName
                    ),
                    (
                        Text("")
                        template_value(ModelCatalogTextSize(10.0))
                        TextColor(Color::srgb(0.53, 0.66, 0.64))
                        ImageTargetProgress
                    ),
                ]
            ),
        ]
    }
}

#[cfg(feature = "mesh_training")]
pub(in crate::viewer) fn mesh_target_summary() -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            min_height: px(44),
            flex_direction: FlexDirection::Column,
            row_gap: px(4),
            justify_content: JustifyContent::Center,
            padding: UiRect::horizontal(px(4)),
        }
        Visibility::Hidden
        MeshTargetSummary
        Children [
            (
                Text("")
                template_value(ModelCatalogTextSize(12.0))
                TextColor(Color::srgb(0.86, 0.92, 0.90))
                MeshTargetName
            ),
            (
                Text("")
                template_value(ModelCatalogTextSize(10.0))
                TextColor(Color::srgb(0.53, 0.66, 0.64))
                MeshTargetProgress
            ),
        ]
    }
}

#[cfg(not(feature = "mesh_training"))]
pub(in crate::viewer) fn mesh_target_summary() -> impl Scene {
    bsn! {
        Node {
            display: Display::None,
        }
    }
}

#[cfg(not(feature = "hyper_dino"))]
pub(in crate::viewer) fn image_target_summary() -> impl Scene {
    bsn! {
        Node {
            display: Display::None,
        }
    }
}
