use super::accumulator::OutputGradientAccumulator;
use super::*;

#[derive(Default)]
pub(super) struct DirectObjectiveAccumulators {
    pub(super) temporal_liveness: OutputGradientAccumulator,
    pub(super) mesh_motion_liveness: OutputGradientAccumulator,
    pub(super) surface_escape_liveness: OutputGradientAccumulator,
    pub(super) target_coverage_liveness: OutputGradientAccumulator,
    pub(super) material_coverage_liveness: OutputGradientAccumulator,
    pub(super) material_visible_liveness: OutputGradientAccumulator,
    pub(super) extent_front_liveness: OutputGradientAccumulator,
    pub(super) phase_progress: OutputGradientAccumulator,
    pub(super) liveness_phase_memory: OutputGradientAccumulator,
    pub(super) mesh_motion: OutputGradientAccumulator,
    pub(super) extent_front_motion: OutputGradientAccumulator,
    pub(super) temporal_extent_motion: OutputGradientAccumulator,
    pub(super) extent_motion_memory: OutputGradientAccumulator,
    pub(super) material_coverage_motion: OutputGradientAccumulator,
    pub(super) material_surface_motion: OutputGradientAccumulator,
    pub(super) residual_velocity: OutputGradientAccumulator,
    pub(super) motion_memory: OutputGradientAccumulator,
    pub(super) material_coverage_motion_memory: OutputGradientAccumulator,
    pub(super) material_coverage_materialization: OutputGradientAccumulator,
    pub(super) temporal_materialization: OutputGradientAccumulator,
    pub(super) active_surface_materialization: OutputGradientAccumulator,
    pub(super) strict_surface_materialization: OutputGradientAccumulator,
    pub(super) material_visibility: OutputGradientAccumulator,
    pub(super) surface_color: OutputGradientAccumulator,
    pub(super) scale_budget: OutputGradientAccumulator,
    pub(super) combined_pre_cap: OutputGradientAccumulator,
    pub(super) combined_post_cap: OutputGradientAccumulator,
    pub(super) mesh_motion_post_cap: OutputGradientAccumulator,
    pub(super) residual_velocity_post_cap: OutputGradientAccumulator,
    pub(super) motion_memory_post_cap: OutputGradientAccumulator,
    pub(super) liveness_post_cap: OutputGradientAccumulator,
    pub(super) phase_post_cap: OutputGradientAccumulator,
    pub(super) material_post_cap: OutputGradientAccumulator,
    pub(super) scale_post_cap: OutputGradientAccumulator,
    pub(super) color_post_cap: OutputGradientAccumulator,
    rows: usize,
}

impl DirectObjectiveAccumulators {
    pub(super) fn add_rows(&mut self, rows: usize) {
        self.rows += rows;
    }

    pub(super) fn into_diagnostics(
        self,
        snapshots: usize,
        terminal_liveness_state: OutputGradientAccumulator,
    ) -> DirectRolloutObjectiveDiagnostics {
        DirectRolloutObjectiveDiagnostics {
            snapshots,
            rows: self.rows,
            temporal_liveness_rms: self.temporal_liveness.rms(),
            temporal_liveness_nonzero_fraction: self.temporal_liveness.nonzero_fraction(),
            terminal_liveness_state_rms: terminal_liveness_state.rms(),
            terminal_liveness_state_nonzero_fraction: terminal_liveness_state.nonzero_fraction(),
            mesh_motion_liveness_rms: self.mesh_motion_liveness.rms(),
            mesh_motion_liveness_nonzero_fraction: self.mesh_motion_liveness.nonzero_fraction(),
            surface_escape_liveness_rms: self.surface_escape_liveness.rms(),
            surface_escape_liveness_nonzero_fraction: self
                .surface_escape_liveness
                .nonzero_fraction(),
            target_coverage_liveness_rms: self.target_coverage_liveness.rms(),
            target_coverage_liveness_nonzero_fraction: self
                .target_coverage_liveness
                .nonzero_fraction(),
            material_coverage_liveness_rms: self.material_coverage_liveness.rms(),
            material_coverage_liveness_nonzero_fraction: self
                .material_coverage_liveness
                .nonzero_fraction(),
            material_visible_liveness_rms: self.material_visible_liveness.rms(),
            material_visible_liveness_nonzero_fraction: self
                .material_visible_liveness
                .nonzero_fraction(),
            extent_front_liveness_rms: self.extent_front_liveness.rms(),
            extent_front_liveness_nonzero_fraction: self.extent_front_liveness.nonzero_fraction(),
            phase_rms: self.phase_progress.rms(),
            phase_nonzero_fraction: self.phase_progress.nonzero_fraction(),
            liveness_phase_memory_rms: self.liveness_phase_memory.rms(),
            liveness_phase_memory_nonzero_fraction: self.liveness_phase_memory.nonzero_fraction(),
            mesh_motion_rms: self.mesh_motion.rms(),
            mesh_motion_nonzero_fraction: self.mesh_motion.nonzero_fraction(),
            extent_front_motion_rms: self.extent_front_motion.rms(),
            extent_front_motion_nonzero_fraction: self.extent_front_motion.nonzero_fraction(),
            temporal_extent_motion_rms: self.temporal_extent_motion.rms(),
            temporal_extent_motion_nonzero_fraction: self.temporal_extent_motion.nonzero_fraction(),
            extent_motion_memory_rms: self.extent_motion_memory.rms(),
            extent_motion_memory_nonzero_fraction: self.extent_motion_memory.nonzero_fraction(),
            material_coverage_motion_rms: self.material_coverage_motion.rms(),
            material_coverage_motion_nonzero_fraction: self
                .material_coverage_motion
                .nonzero_fraction(),
            material_surface_motion_rms: self.material_surface_motion.rms(),
            material_surface_motion_nonzero_fraction: self
                .material_surface_motion
                .nonzero_fraction(),
            residual_velocity_rms: self.residual_velocity.rms(),
            residual_velocity_nonzero_fraction: self.residual_velocity.nonzero_fraction(),
            motion_memory_rms: self.motion_memory.rms(),
            motion_memory_nonzero_fraction: self.motion_memory.nonzero_fraction(),
            material_coverage_motion_memory_rms: self.material_coverage_motion_memory.rms(),
            material_coverage_motion_memory_nonzero_fraction: self
                .material_coverage_motion_memory
                .nonzero_fraction(),
            material_coverage_materialization_rms: self.material_coverage_materialization.rms(),
            material_coverage_materialization_nonzero_fraction: self
                .material_coverage_materialization
                .nonzero_fraction(),
            temporal_materialization_rms: self.temporal_materialization.rms(),
            temporal_materialization_nonzero_fraction: self
                .temporal_materialization
                .nonzero_fraction(),
            active_surface_materialization_rms: self.active_surface_materialization.rms(),
            active_surface_materialization_nonzero_fraction: self
                .active_surface_materialization
                .nonzero_fraction(),
            strict_surface_materialization_rms: self.strict_surface_materialization.rms(),
            strict_surface_materialization_nonzero_fraction: self
                .strict_surface_materialization
                .nonzero_fraction(),
            material_visibility_rms: self.material_visibility.rms(),
            material_visibility_nonzero_fraction: self.material_visibility.nonzero_fraction(),
            surface_color_rms: self.surface_color.rms(),
            surface_color_nonzero_fraction: self.surface_color.nonzero_fraction(),
            scale_budget_rms: self.scale_budget.rms(),
            scale_budget_nonzero_fraction: self.scale_budget.nonzero_fraction(),
            combined_pre_cap_rms: self.combined_pre_cap.rms(),
            combined_post_cap_rms: self.combined_post_cap.rms(),
            mesh_motion_post_cap_rms: self.mesh_motion_post_cap.rms(),
            mesh_motion_post_cap_nonzero_fraction: self.mesh_motion_post_cap.nonzero_fraction(),
            residual_velocity_post_cap_rms: self.residual_velocity_post_cap.rms(),
            residual_velocity_post_cap_nonzero_fraction: self
                .residual_velocity_post_cap
                .nonzero_fraction(),
            motion_memory_post_cap_rms: self.motion_memory_post_cap.rms(),
            motion_memory_post_cap_nonzero_fraction: self.motion_memory_post_cap.nonzero_fraction(),
            liveness_post_cap_rms: self.liveness_post_cap.rms(),
            phase_post_cap_rms: self.phase_post_cap.rms(),
            material_post_cap_rms: self.material_post_cap.rms(),
            scale_post_cap_rms: self.scale_post_cap.rms(),
            color_post_cap_rms: self.color_post_cap.rms(),
        }
    }
}
