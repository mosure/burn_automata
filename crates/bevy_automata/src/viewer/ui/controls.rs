use super::*;

pub(in crate::viewer) fn control_button(label: &'static str, kind: RunControlKind) -> impl Scene {
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
        on(|_event: On<Pointer<Press>>, mut settings: ResMut<AutomataSettings>, mut runtime: ResMut<AutomataRuntime>| {
            settings.mark_changed();
            runtime.trace = None;
            runtime.frame = 0;
            runtime.status = "reset requested".to_string();
        })
    }
}

pub(in crate::viewer) fn backward_button() -> impl Scene {
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

#[cfg(feature = "hyper_dino")]
pub(in crate::viewer) fn hyper_image_button() -> impl Scene {
    bsn! {
        Button
        Node {
            width: percent(100),
            height: px(32),
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
            Text("open image -> HyperNPA")
            template_value(ModelCatalogTextSize(12.0))
            TextColor(Color::srgb(0.86, 0.92, 0.90))
        )]
    }
}

#[cfg(not(feature = "hyper_dino"))]
pub(in crate::viewer) fn hyper_image_button() -> impl Scene {
    bsn! {
        Node {
            display: Display::None,
        }
    }
}
