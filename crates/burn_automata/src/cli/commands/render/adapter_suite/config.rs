use crate::cli::prelude::*;

#[derive(Clone, Copy)]
pub(super) struct AdapterSuiteRenderSettings {
    pub(super) image_size: usize,
    pub(super) target_samples: usize,
    pub(super) sigma: f32,
    pub(super) min_sigma: f32,
    pub(super) max_sigma: f32,
    pub(super) gaussian_decode_mode: RenderGaussianDecodeModeArg,
    pub(super) world_scale: Option<f32>,
    pub(super) render_opacity_logit_bias: f32,
    pub(super) density_weight: f32,
    pub(super) color_weight: f32,
    pub(super) depth_weight: f32,
}

impl AdapterSuiteRenderSettings {
    pub(super) fn loss_config(self, seed_scale: f32) -> RenderLossConfig {
        RenderLossConfig {
            image_size: self.image_size,
            sigma: self.sigma,
            min_sigma: self.min_sigma,
            max_sigma: self.max_sigma,
            gaussian_decode_mode: self.gaussian_decode_mode.into(),
            world_scale: self.world_scale.unwrap_or(seed_scale * 2.0),
            target_samples: self.target_samples,
            opacity_logit_bias: self.render_opacity_logit_bias,
            density_weight: self.density_weight,
            color_weight: self.color_weight,
            depth_weight: self.depth_weight,
        }
    }
}

#[derive(Clone)]
pub(super) struct AdapterSuiteTrainingSettings {
    pub(super) supervised_steps_per_round: usize,
    pub(super) particles: usize,
    pub(super) rollout_steps: usize,
    pub(super) gradient_particles: usize,
    pub(super) gradient_mode: RenderGradientModeArg,
    pub(super) finite_diff_eps: f32,
    pub(super) motion_gain: f32,
    pub(super) perception_position_gain: f32,
    pub(super) max_update_norm: f32,
    pub(super) trajectory_supervision: bool,
    pub(super) training_backend: RenderTrainingBackendArg,
    pub(super) direct_output_gradient_rms_cap: f32,
    pub(super) direct_line_search: bool,
    pub(super) direct_line_search_scales: Vec<f32>,
    pub(super) direct_material_output_only: bool,
    pub(super) direct_selection_seed_training: bool,
    pub(super) selection_seed: u64,
    pub(super) selection_seeds: Vec<u64>,
    pub(super) sgd: SgdConfig,
    pub(super) adapter_rank: usize,
    pub(super) adapter_alpha: f32,
}

impl AdapterSuiteTrainingSettings {
    pub(super) fn render_proxy_config(
        &self,
        phase: AdapterSuiteTrainingPhaseConfig,
        render: RenderLossConfig,
    ) -> RenderProxyTrainingConfig {
        RenderProxyTrainingConfig {
            target: phase.target,
            rounds: phase.rounds,
            supervised_steps_per_round: self.supervised_steps_per_round,
            particles: self.particles,
            rollout_steps: self.rollout_steps,
            gradient_particles: self.gradient_particles,
            gradient_mode: self.gradient_mode,
            finite_diff_eps: self.finite_diff_eps,
            motion_gain: self.motion_gain,
            perception_position_gain: self.perception_position_gain,
            max_update_norm: self.max_update_norm,
            trajectory_supervision: self.trajectory_supervision,
            trajectory_render_gain: ROBUST_3D_TRAJECTORY_RENDER_GAIN,
            trajectory_mesh_gain: ROBUST_3D_TRAJECTORY_MESH_GAIN,
            trajectory_render_samples: ROBUST_3D_TRAJECTORY_RENDER_SAMPLES,
            liveness_gain: ROBUST_3D_LIVENESS_GAIN,
            liveness_front_radius: ROBUST_3D_LIVENESS_FRONT_RADIUS,
            liveness_update_multiplier: ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER,
            coverage_gain: ROBUST_3D_COVERAGE_GAIN,
            coverage_samples: ROBUST_3D_COVERAGE_SAMPLES,
            coverage_mode: CoverageUpdateModeArg::SlicedOt,
            coverage_softness: 0.0,
            coverage_repulsion_gain: ROBUST_3D_COVERAGE_REPULSION_GAIN,
            coverage_gap_gain: ROBUST_3D_COVERAGE_REPULSION_GAIN,
            coverage_repulsion_radius: 0.0,
            coverage_normal_weight: ROBUST_3D_COVERAGE_NORMAL_WEIGHT,
            extent_gain: ROBUST_3D_EXTENT_GAIN,
            full_coverage_adjoint: true,
            surface_gain: ROBUST_3D_SURFACE_GAIN,
            surface_escape_gain: ROBUST_3D_SURFACE_ESCAPE_GAIN,
            opacity_gain: ROBUST_3D_OPACITY_GAIN,
            material_liveness_gain: ROBUST_3D_MATERIAL_LIVENESS_GAIN,
            material_tail_gain: ROBUST_3D_MATERIAL_TAIL_GAIN,
            material_suppression_update_multiplier:
                ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
            material_max_opacity_update: ROBUST_3D_MATERIAL_MAX_OPACITY_UPDATE,
            scale_gain: ROBUST_3D_SCALE_GAIN,
            scale_budget_weight: ROBUST_3D_SCALE_BUDGET_WEIGHT,
            max_opacity_update: 0.05,
            direct_output_gradient_rms_cap: self.direct_output_gradient_rms_cap,
            direct_line_search: self.direct_line_search,
            direct_line_search_scales: self.direct_line_search_scales.clone(),
            direct_material_output_only: self.direct_material_output_only,
            training_backend: self.training_backend,
            weight_update_mode: phase.weight_update_mode,
            adapter_rank: self.adapter_rank,
            adapter_alpha: self.adapter_alpha,
            adapter_seed: phase.adapter_seed,
            direct_selection_seed_training: self.direct_selection_seed_training,
            seed: phase.seed,
            selection_seed: Some(self.selection_seed),
            selection_seeds: self.selection_seeds.clone(),
            seed_scale: phase.seed_scale,
            seed_mode: phase.seed_mode,
            render,
            sgd: self.sgd,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct AdapterSuiteTrainingPhaseConfig {
    pub(super) target: MeshTargetArg,
    pub(super) rounds: usize,
    pub(super) weight_update_mode: RenderWeightUpdateModeArg,
    pub(super) adapter_seed: u64,
    pub(super) seed: u64,
    pub(super) seed_scale: f32,
    pub(super) seed_mode: ParticleSeed,
}
