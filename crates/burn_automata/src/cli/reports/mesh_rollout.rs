use crate::cli::prelude::*;

#[derive(Clone, Copy)]
pub(crate) struct MeshRolloutCaseConfig {
    pub(crate) particle_count: usize,
    pub(crate) steps: usize,
    pub(crate) seed: u64,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
}

#[derive(Serialize)]
pub(crate) struct MeshRolloutReport {
    pub(crate) passed: bool,
    pub(crate) max_initial_surface_distance: f32,
    pub(crate) mean_initial_surface_distance: f32,
    pub(crate) max_surface_distance: f32,
    pub(crate) mean_surface_distance: f32,
    pub(crate) mean_surface_improvement: f32,
    pub(crate) mean_surface_improvement_ratio: f32,
    pub(crate) max_target_coverage_distance: f32,
    pub(crate) mean_target_coverage_distance: f32,
    pub(crate) min_target_coverage_fraction: f32,
    pub(crate) max_color_target_error: f32,
    pub(crate) mean_color_target_error: f32,
    pub(crate) first_motion_per_step: f32,
    pub(crate) max_motion_per_step: f32,
    pub(crate) max_opacity_target_error: f32,
    pub(crate) min_final_opacity: f32,
    pub(crate) max_final_opacity: f32,
    pub(crate) cases: Vec<MeshRolloutCaseReport>,
}

#[derive(Serialize)]
pub(crate) struct MeshRolloutCaseReport {
    pub(crate) particle_count: usize,
    pub(crate) steps: usize,
    pub(crate) seed: u64,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) max_initial_surface_distance: f32,
    pub(crate) mean_initial_surface_distance: f32,
    pub(crate) max_surface_distance: f32,
    pub(crate) mean_surface_distance: f32,
    pub(crate) mean_surface_improvement: f32,
    pub(crate) mean_surface_improvement_ratio: f32,
    pub(crate) target_coverage_threshold: f32,
    pub(crate) max_target_coverage_distance: f32,
    pub(crate) mean_target_coverage_distance: f32,
    pub(crate) target_coverage_fraction: f32,
    pub(crate) max_color_target_error: f32,
    pub(crate) mean_color_target_error: f32,
    pub(crate) first_motion_per_step: f32,
    pub(crate) max_motion_per_step: f32,
    pub(crate) expected_final_opacity_logit: f32,
    pub(crate) min_final_opacity_logit: f32,
    pub(crate) max_final_opacity_logit: f32,
    pub(crate) max_opacity_target_error: f32,
    pub(crate) finite: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct TorusRobustnessCaseConfig {
    pub(crate) particle_count: usize,
    pub(crate) steps: usize,
    pub(crate) seed: u64,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
}

#[derive(Serialize)]
pub(crate) struct TorusRobustnessReport {
    pub(crate) passed: bool,
    pub(crate) target_opacity_delta: f32,
    pub(crate) trained_opacity_delta: f32,
    pub(crate) target_motion_gain: f32,
    pub(crate) target_residual_decay: f32,
    pub(crate) max_target_position_error: f32,
    pub(crate) mean_target_position_error: f32,
    pub(crate) max_torus_surface_error: f32,
    pub(crate) max_color_target_error: f32,
    pub(crate) first_motion_per_step: f32,
    pub(crate) max_motion_per_step: f32,
    pub(crate) max_opacity_target_error: f32,
    pub(crate) min_final_opacity: f32,
    pub(crate) max_final_opacity: f32,
    pub(crate) cases: Vec<TorusRobustnessCaseReport>,
}

#[derive(Serialize)]
pub(crate) struct TorusRobustnessCaseReport {
    pub(crate) particle_count: usize,
    pub(crate) steps: usize,
    pub(crate) seed: u64,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) torus_inner_radius: f32,
    pub(crate) torus_outer_radius: f32,
    pub(crate) max_initial_target_position_error: f32,
    pub(crate) mean_initial_target_position_error: f32,
    pub(crate) max_target_position_error: f32,
    pub(crate) mean_target_position_error: f32,
    pub(crate) max_torus_surface_error: f32,
    pub(crate) mean_torus_surface_error: f32,
    pub(crate) min_final_radial: f32,
    pub(crate) max_final_radial: f32,
    pub(crate) max_final_abs_z: f32,
    pub(crate) max_color_target_error: f32,
    pub(crate) mean_color_target_error: f32,
    pub(crate) first_motion_per_step: f32,
    pub(crate) max_motion_per_step: f32,
    pub(crate) expected_final_opacity_logit: f32,
    pub(crate) min_final_opacity_logit: f32,
    pub(crate) max_final_opacity_logit: f32,
    pub(crate) max_opacity_target_error: f32,
    pub(crate) finite: bool,
}
