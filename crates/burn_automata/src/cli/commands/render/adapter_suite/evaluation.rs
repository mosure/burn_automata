use crate::cli::prelude::*;

use super::{config::AdapterSuiteRenderSettings, splits::adapter_suite_split};

#[allow(clippy::too_many_arguments)]
pub(super) fn adapter_suite_shared_base_evaluations(
    shared_base_output: &PathBuf,
    base_manifest: &BpkModelManifest,
    targets: &[MeshTargetArg],
    holdout_targets: &[MeshTargetArg],
    seed_scale: Option<f32>,
    seed_mode: Option<SeedModeArg>,
    particles: usize,
    rollout_steps: usize,
    selection_seed: u64,
    training_selection_seeds: &[u64],
    render_settings: AdapterSuiteRenderSettings,
) -> Result<Vec<CliRenderAdapterSuiteBaseEvalEntry>, Box<dyn std::error::Error>> {
    let base_model = base_manifest.clone().into_model();
    let validation_extra_seeds =
        render_training_validation_extra_seeds(selection_seed, training_selection_seeds);
    let mut evaluations = Vec::with_capacity(targets.len());
    for &target in targets {
        let target_seed_scale =
            seed_scale.unwrap_or_else(|| mesh_target_render_training_seed_scale(target));
        let target_seed_mode = seed_mode
            .map(ParticleSeed::from)
            .unwrap_or_else(|| default_render_training_seed_mode(target, &base_model));
        let render = render_settings.loss_config(target_seed_scale);
        let growth_validation = growth_3d_validation_report(
            shared_base_output,
            target,
            Growth3dValidationConfig {
                particle_count: particles,
                steps: rollout_steps,
                seed: 0x005a_173d,
                extra_seeds: validation_extra_seeds.clone(),
                seed_scale: target_seed_scale,
                seed_mode: target_seed_mode,
                gate: Growth3dValidationGateArg::Strict,
                render,
            },
        )?;
        let strict_gate_summary = CliRenderTrainingGateSummary::from_validation(&growth_validation);
        evaluations.push(CliRenderAdapterSuiteBaseEvalEntry {
            target,
            split: adapter_suite_split(target, holdout_targets),
            seed_scale: target_seed_scale,
            seed_mode: target_seed_mode,
            strict_gate_summary,
            growth_validation,
        });
    }
    Ok(evaluations)
}
