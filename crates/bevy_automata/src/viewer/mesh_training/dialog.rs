use bevy::{tasks::AsyncComputeTaskPool, window::FileDragAndDrop};

use super::*;

const MESH_NORMALIZATION_SCALE: f32 = 0.72;

pub(in crate::viewer) fn handle_open_npa_mesh_dialog(
    mut requests: MessageReader<OpenNpaMesh>,
    channel: Res<MeshTargetDialogChannel>,
    mut runtime: ResMut<AutomataRuntime>,
) {
    for _request in requests.read() {
        let sender = channel.sender.clone();
        runtime.status = "opening 3D mesh".to_string();
        AsyncComputeTaskPool::get()
            .spawn(async move {
                let picked = rfd::AsyncFileDialog::new()
                    .add_filter("Wavefront mesh", &["obj"])
                    .pick_file()
                    .await;
                let result = match picked {
                    Some(file) => {
                        let file_name = file.file_name();
                        let bytes = file.read().await;
                        build_mesh_source(file_name, bytes)
                    }
                    None => Err("mesh selection cancelled".to_string()),
                };
                let _ = sender.send(result);
            })
            .detach();
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(in crate::viewer) fn handle_npa_mesh_drop(
    mut drops: MessageReader<FileDragAndDrop>,
    channel: Res<MeshTargetDialogChannel>,
    mut runtime: ResMut<AutomataRuntime>,
) {
    for event in drops.read() {
        let FileDragAndDrop::DroppedFile { path_buf, .. } = event else {
            continue;
        };
        if path_buf
            .extension()
            .and_then(|extension| extension.to_str())
            .is_none_or(|extension| !extension.eq_ignore_ascii_case("obj"))
        {
            continue;
        }
        let sender = channel.sender.clone();
        let path = path_buf.clone();
        runtime.status = format!(
            "loading dropped mesh {}",
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("mesh.obj")
        );
        AsyncComputeTaskPool::get()
            .spawn(async move {
                let result = std::fs::read(&path)
                    .map_err(|error| error.to_string())
                    .and_then(|bytes| {
                        build_mesh_source(
                            path.file_name()
                                .and_then(|name| name.to_str())
                                .unwrap_or("mesh.obj")
                                .to_string(),
                            bytes,
                        )
                    });
                let _ = sender.send(result);
            })
            .detach();
    }
}

#[cfg(target_arch = "wasm32")]
pub(in crate::viewer) fn handle_npa_mesh_drop(
    mut drops: MessageReader<FileDragAndDrop>,
    mut runtime: ResMut<AutomataRuntime>,
) {
    if drops.read().next().is_some() {
        runtime.status = "use open mesh to select a browser-local OBJ".to_string();
    }
}

pub(in crate::viewer) fn poll_mesh_target_sources(
    channel: Res<MeshTargetDialogChannel>,
    mut state: ResMut<MeshTargetTrainingState>,
    #[cfg(feature = "hyper_dino")] mut image_target: ResMut<ImageTargetTrainingState>,
    #[cfg(feature = "hyper_dino")] mut inference: ResMut<HyperNpaInferenceState>,
    mut settings: ResMut<AutomataSettings>,
    mut runtime: ResMut<AutomataRuntime>,
) {
    for source in channel.receiver.try_iter() {
        match source {
            Ok(source) => {
                #[cfg(feature = "hyper_dino")]
                {
                    image_target.clear_target();
                    inference.cancel_current();
                }
                state.set_source(source);
                let Some(target) = state.target.as_ref() else {
                    continue;
                };
                if let Err(error) = super::training::install_mesh_preview(
                    &target.source.target,
                    &target.source.file_name,
                    &mut settings,
                    &mut runtime,
                ) {
                    state.phase = MeshTargetTrainingPhase::Failed;
                    state.error = Some(error.clone());
                    runtime.status = format!("failed to preview normalized mesh: {error}");
                    continue;
                }
                let (minimum, maximum) = target.source.target.bounds();
                runtime.status = format!(
                    "3D target ready: {} | {} vertices / {} faces / {:.1} MB | normalized [{:.2},{:.2},{:.2}] to [{:.2},{:.2},{:.2}]",
                    target.source.file_name,
                    target.source.target.vertices.len(),
                    target.source.target.faces.len(),
                    target.source.bytes.len() as f32 / (1024.0 * 1024.0),
                    minimum[0],
                    minimum[1],
                    minimum[2],
                    maximum[0],
                    maximum[1],
                    maximum[2],
                );
            }
            Err(error) => {
                runtime.status = format!("mesh load skipped: {error}");
            }
        }
    }
}

fn build_mesh_source(file_name: String, bytes: Vec<u8>) -> Result<MeshTargetSource, String> {
    let text = std::str::from_utf8(&bytes)
        .map_err(|error| format!("{file_name} is not a UTF-8 Wavefront OBJ: {error}"))?;
    let target = TriangleMeshTarget::from_obj_str(text, MESH_NORMALIZATION_SCALE)
        .map_err(|error| error.to_string())?;
    Ok(MeshTargetSource {
        file_name,
        bytes: Arc::new(bytes),
        target: Arc::new(target),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_teapot_source_is_normalized_to_viewer_domain() {
        let source = build_mesh_source(
            "utah_teapot.obj".to_string(),
            include_bytes!("../../../../../assets/meshes/utah_teapot.obj").to_vec(),
        )
        .unwrap();
        let (minimum, maximum) = source.target.bounds();
        let extent = [
            maximum[0] - minimum[0],
            maximum[1] - minimum[1],
            maximum[2] - minimum[2],
        ];
        assert!((extent.into_iter().fold(0.0_f32, f32::max) - 1.44).abs() < 1.0e-4);
    }
}
