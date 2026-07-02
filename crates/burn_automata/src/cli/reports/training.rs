use crate::cli::prelude::*;

use super::mesh_rollout::MeshRolloutReport;

#[derive(Serialize)]
pub(crate) struct CliTrainingReport {
    pub(crate) preset: AutomataPreset,
    pub(crate) target_source: String,
    pub(crate) student_seed: u64,
    pub(crate) sgd: SgdConfig,
    pub(crate) report: TrainingRunReport,
    pub(crate) model_output: Option<String>,
    pub(crate) batch_source: TrainingBatchArg,
    pub(crate) rollout_supervision: Option<CliRolloutSupervisionReport>,
    pub(crate) mesh_rollout: Option<MeshRolloutReport>,
    pub(crate) render_loss: Option<MultiViewRenderLossReport>,
}

#[derive(Serialize)]
pub(crate) struct CliRenderLossEvalReport {
    pub(crate) target: MeshTargetArg,
    pub(crate) model: String,
    pub(crate) particle_count: usize,
    pub(crate) steps: usize,
    pub(crate) seed: u64,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) render_loss: MultiViewRenderLossReport,
}

#[derive(Serialize)]
pub(crate) struct CliTorusTrainingReport {
    pub(crate) preset: AutomataPreset,
    pub(crate) target_source: String,
    pub(crate) student_seed: u64,
    pub(crate) sgd: SgdConfig,
    pub(crate) report: TrainingRunReport,
    pub(crate) model_output: Option<String>,
    pub(crate) robustness: TorusRobustnessReport,
    pub(crate) batch_source: TrainingBatchArg,
    pub(crate) training_mode: MeshTrainingModeArg,
    pub(crate) rollout_supervision: Option<CliRolloutSupervisionReport>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct CliRolloutSupervisionReport {
    pub(crate) particle_count: usize,
    pub(crate) rollout_steps: usize,
    pub(crate) rollouts: usize,
    pub(crate) temporal_samples: usize,
    pub(crate) update_prob: f32,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) motion_gain: Option<f32>,
    pub(crate) max_update_norm: Option<f32>,
    pub(crate) density_gain: Option<f32>,
    pub(crate) expansion_gain: Option<f32>,
    pub(crate) coverage_gain: Option<f32>,
    pub(crate) coverage_samples: Option<usize>,
    pub(crate) coverage_mode: Option<CoverageUpdateModeArg>,
    pub(crate) coverage_softness: Option<f32>,
    pub(crate) coverage_repulsion_gain: Option<f32>,
    pub(crate) coverage_gap_gain: Option<f32>,
    pub(crate) coverage_repulsion_radius: Option<f32>,
    pub(crate) coverage_normal_weight: Option<f32>,
    pub(crate) extent_gain: Option<f32>,
    pub(crate) color_gain: Option<f32>,
    pub(crate) aux_state_gain: Option<f32>,
    pub(crate) opacity_gain: Option<f32>,
    pub(crate) front_opacity_gain: Option<f32>,
    pub(crate) front_radius: Option<f32>,
    pub(crate) front_max_opacity_update: Option<f32>,
    pub(crate) front_motion_gate: Option<bool>,
    pub(crate) preserve_opacity_update: Option<bool>,
}
