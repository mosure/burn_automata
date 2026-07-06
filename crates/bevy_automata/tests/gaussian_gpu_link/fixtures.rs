use super::prelude::*;

use std::sync::Mutex;

pub(crate) const LIZARD_MODEL_PATH: &str = "models/catalog/growing/lizard.bpk";
pub(crate) const POLKA_MODEL_PATH: &str = "models/catalog/texture/polka_dotted_0121.bpk";
pub(crate) const TORUS_GROWTH_MODEL_PATH: &str = "assets/models/uv_torus_growth_3d.bpk";
pub(crate) const SH_C0: f32 = 0.282_094_8;
static BEVY_TEST_LOCK: Mutex<()> = Mutex::new(());

pub(crate) fn bevy_test_guard() -> MutexGuard<'static, ()> {
    BEVY_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub(crate) fn workspace_root_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

pub(crate) fn workspace_path(path: &str) -> std::path::PathBuf {
    workspace_root_dir().join(path)
}

pub(crate) fn existing_workspace_path(path: &str) -> Option<std::path::PathBuf> {
    let path = workspace_path(path);
    path.exists().then_some(path)
}

pub(crate) fn visible_test_cloud_3d(count: usize) -> PlanarGaussian3d {
    let side = (count as f32).cbrt().ceil().max(1.0) as usize;
    let mut gaussians = Vec::with_capacity(count);
    for idx in 0..count {
        let x = idx % side;
        let y = (idx / side) % side;
        let z = idx / (side * side);
        let denom = (side.saturating_sub(1)).max(1) as f32;
        let position = [
            (x as f32 / denom - 0.5) * 0.9,
            (y as f32 / denom - 0.5) * 0.9,
            (z as f32 / denom - 0.5) * 0.9,
            1.0,
        ];
        let color = [
            0.25 + 0.55 * x as f32 / denom,
            0.30 + 0.50 * y as f32 / denom,
            0.45 + 0.40 * z as f32 / denom,
        ];
        let mut coefficients = [0.0; GAUSSIAN_SH_COEFF_COUNT];
        coefficients[0] = (color[0] - 0.5) / SH_C0;
        coefficients[1] = (color[1] - 0.5) / SH_C0;
        coefficients[2] = (color[2] - 0.5) / SH_C0;
        gaussians.push(Gaussian3d {
            position_visibility: position.into(),
            spherical_harmonic: SphericalHarmonicCoefficients { coefficients },
            rotation: [1.0, 0.0, 0.0, 0.0].into(),
            scale_opacity: [0.06, 0.06, 0.06, 0.75].into(),
        });
    }
    gaussians.into()
}

pub(crate) fn lizard_or_seeded_model()
-> Result<(NpaModel, HashGridConfig, f32), Box<dyn std::error::Error>> {
    if let Some(path) = existing_workspace_path(LIZARD_MODEL_PATH) {
        let manifest = load_manifest(path)?;
        let hashgrid = manifest.hashgrid.clone();
        return Ok((manifest.into_model(), hashgrid, 0.2));
    }
    let preset = AutomataPreset::Growing2d;
    let (config, hashgrid) = NpaConfig::for_preset(preset);
    Ok((
        NpaModel::seeded(config, 42),
        hashgrid,
        NpaConfig::seed_scale_for_preset(preset),
    ))
}

pub(crate) fn is_missing_wgpu(message: &str) -> bool {
    message.contains("no WGPU adapter") || message.contains("failed to create WGPU device")
}
