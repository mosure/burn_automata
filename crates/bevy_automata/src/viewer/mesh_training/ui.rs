use super::*;

pub(in crate::viewer) fn sync_mesh_target_summary(
    state: Res<MeshTargetTrainingState>,
    mut summaries: Query<&mut Visibility, With<MeshTargetSummary>>,
    mut names: Query<&mut Text, (With<MeshTargetName>, Without<MeshTargetProgress>)>,
    mut progress_labels: Query<&mut Text, (With<MeshTargetProgress>, Without<MeshTargetName>)>,
) {
    if !state.is_changed() {
        return;
    }
    let visibility = if state.has_target() {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    for mut summary in &mut summaries {
        *summary = visibility;
    }
    let Some(target) = state.target.as_ref() else {
        return;
    };
    for mut name in &mut names {
        name.0 = compact_mesh_name(&target.source.file_name, 30);
    }
    for mut progress in &mut progress_labels {
        progress.0 = super::training::mesh_training_status(&state);
    }
}

pub(in crate::viewer) fn sync_mesh_training_button_label(
    state: Res<MeshTargetTrainingState>,
    mut labels: Query<(&RunControlButtonLabel, &mut Text, &mut TextColor)>,
) {
    if !state.is_changed() || !state.has_target() {
        return;
    }
    for (label, mut text, mut color) in &mut labels {
        if label.0 == RunControlKind::Train {
            text.0 = state.train_action_label().to_string();
            color.0 = if state.train_action_available() {
                Color::srgb(0.86, 0.91, 0.94)
            } else {
                Color::srgb(0.42, 0.47, 0.50)
            };
        }
    }
}

pub(in crate::viewer) fn update_mesh_button_styles(
    mut buttons: Query<(&Hovered, &mut BackgroundColor, &mut BorderColor), With<MeshOpenButton>>,
) {
    for (hovered, mut background, mut border) in &mut buttons {
        background.0 = if hovered.0 {
            Color::srgb(0.13, 0.15, 0.17)
        } else {
            Color::srgb(0.10, 0.12, 0.14)
        };
        *border = BorderColor::from(if hovered.0 {
            Color::srgb(0.36, 0.42, 0.46)
        } else {
            Color::srgb(0.25, 0.30, 0.34)
        });
    }
}

fn compact_mesh_name(value: &str, max_chars: usize) -> String {
    let mut chars = value.chars();
    let prefix = chars.by_ref().take(max_chars).collect::<String>();
    if chars.next().is_some() {
        format!("{prefix}...")
    } else {
        prefix
    }
}
