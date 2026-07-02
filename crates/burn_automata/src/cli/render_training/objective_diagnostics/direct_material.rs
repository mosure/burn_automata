use super::accumulator::accumulate_output_channels;
use super::direct_accumulators::DirectObjectiveAccumulators;
use super::direct_channels::DirectObjectiveChannels;
use super::direct_combined::{DirectCombinedGradientInputs, accumulate_combined_direct_gradients};
use super::direct_liveness::DirectLivenessObjectiveOutputs;
use super::direct_motion::DirectMotionObjectiveOutputs;
use super::*;

pub(super) struct DirectMaterialObjectiveInputs<'a> {
    pub(super) motion: &'a DirectMotionObjectiveOutputs,
    pub(super) liveness: &'a DirectLivenessObjectiveOutputs,
}

pub(super) fn accumulate_direct_material_objectives(
    model: &NpaModel,
    target: &TriangleMeshTarget,
    snapshot: &RenderTrajectorySnapshot,
    cfg: &RenderProxyTrainingConfig,
    channels: &DirectObjectiveChannels,
    inputs: DirectMaterialObjectiveInputs<'_>,
    accumulators: &mut DirectObjectiveAccumulators,
) {
    let particle_count = snapshot.positions.len();
    let output_dims = channels.output_dims;
    let motion = inputs.motion;
    let liveness = inputs.liveness;

    let mut material_coverage_materialization_output_gradients =
        vec![0.0; particle_count * output_dims];
    add_material_coverage_materialization_output_objective(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        &motion.updates,
        snapshot.step_fraction,
        cfg.opacity_gain * DIRECT_GROWTH_MATERIAL_COVERAGE_MATERIALIZATION_GAIN_FRACTION,
        &motion.material_coverage_candidate_weights,
        cfg.seed_scale,
        cfg.material_max_opacity_update,
        &mut material_coverage_materialization_output_gradients,
    );
    if let Some(material_output) = channels.material_output {
        accumulate_output_channels(
            &mut accumulators.material_coverage_materialization,
            &material_coverage_materialization_output_gradients,
            particle_count,
            output_dims,
            [material_output],
        );
    }

    let mut temporal_materialization_output_gradients = vec![0.0; particle_count * output_dims];
    add_temporal_materialization_output_objective_with_candidate_weights(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        &motion.updates,
        snapshot.step_fraction,
        cfg.opacity_gain * DIRECT_GROWTH_TEMPORAL_MATERIALIZATION_GAIN_FRACTION,
        cfg.liveness_front_radius,
        &liveness.liveness_candidate_weights,
        cfg.seed_scale,
        cfg.material_max_opacity_update,
        &mut temporal_materialization_output_gradients,
    );
    if let Some(material_output) = channels.material_output {
        accumulate_output_channels(
            &mut accumulators.temporal_materialization,
            &temporal_materialization_output_gradients,
            particle_count,
            output_dims,
            [material_output],
        );
    }

    let mut active_surface_materialization_output_gradients =
        vec![0.0; particle_count * output_dims];
    add_active_surface_materialization_output_objective(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        &motion.updates,
        cfg.opacity_gain * DIRECT_GROWTH_ACTIVE_SURFACE_MATERIALIZATION_GAIN_FRACTION,
        cfg.seed_scale,
        cfg.material_max_opacity_update,
        cfg.liveness_front_radius,
        Some(&liveness.liveness_candidate_weights),
        &mut active_surface_materialization_output_gradients,
    );
    if let Some(material_output) = channels.material_output {
        accumulate_output_channels(
            &mut accumulators.active_surface_materialization,
            &active_surface_materialization_output_gradients,
            particle_count,
            output_dims,
            [material_output],
        );
    }

    let mut strict_surface_materialization_output_gradients =
        vec![0.0; particle_count * output_dims];
    add_strict_surface_materialization_output_objective(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        &motion.updates,
        cfg.opacity_gain * DIRECT_GROWTH_STRICT_SURFACE_MATERIALIZATION_GAIN_FRACTION,
        cfg.seed_scale,
        cfg.material_max_opacity_update,
        &mut strict_surface_materialization_output_gradients,
    );
    if let Some(material_output) = channels.material_output {
        accumulate_output_channels(
            &mut accumulators.strict_surface_materialization,
            &strict_surface_materialization_output_gradients,
            particle_count,
            output_dims,
            [material_output],
        );
    }

    let mut material_surface_motion_output_gradients = vec![0.0; particle_count * output_dims];
    add_material_visible_surface_approach_output_objective(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        &motion.updates,
        cfg.surface_gain,
        cfg.surface_escape_gain,
        cfg.max_update_norm,
        cfg.seed_scale,
        cfg.liveness_front_radius,
        Some(&liveness.liveness_candidate_weights),
        direct_material_surface_motion_weight(
            cfg.trajectory_mesh_gain,
            cfg.coverage_gain,
            snapshot.step_fraction,
        ),
        &mut material_surface_motion_output_gradients,
    );
    add_material_visible_surface_coverage_output_objective(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        &motion.updates,
        cfg.coverage_gain,
        cfg.coverage_samples,
        cfg.max_update_norm,
        cfg.coverage_mode,
        cfg.coverage_softness,
        cfg.coverage_repulsion_gain,
        cfg.coverage_gap_gain,
        cfg.coverage_repulsion_radius,
        cfg.coverage_normal_weight,
        cfg.seed_scale,
        cfg.liveness_front_radius,
        Some(&liveness.liveness_candidate_weights),
        direct_material_surface_motion_weight(
            cfg.trajectory_mesh_gain,
            cfg.coverage_gain,
            snapshot.step_fraction,
        ),
        &mut material_surface_motion_output_gradients,
    );
    boost_sparse_output_channel_rms(
        &mut material_surface_motion_output_gradients,
        output_dims,
        0..model.config.spatial_dims,
        cfg.direct_output_gradient_rms_cap
            * DIRECT_GROWTH_MATERIAL_SURFACE_MOTION_RMS_TARGET_FRACTION,
        16.0,
    );
    accumulate_output_channels(
        &mut accumulators.material_surface_motion,
        &material_surface_motion_output_gradients,
        particle_count,
        output_dims,
        0..model.config.spatial_dims,
    );

    let mut material_output_gradients = vec![0.0; particle_count * output_dims];
    add_material_visibility_output_objective(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        &motion.updates,
        cfg.opacity_gain,
        cfg.material_liveness_gain,
        cfg.material_tail_gain,
        cfg.coverage_samples,
        cfg.seed_scale,
        cfg.material_max_opacity_update,
        cfg.material_suppression_update_multiplier,
        cfg.liveness_front_radius,
        Some(&liveness.liveness_candidate_weights),
        snapshot.step_fraction,
        channels.liveness_update_cap,
        1.0,
        &mut material_output_gradients,
    );
    if let Some(material_output) = channels.material_output {
        accumulate_output_channels(
            &mut accumulators.material_visibility,
            &material_output_gradients,
            particle_count,
            output_dims,
            [material_output],
        );
    }

    let mut surface_color_output_gradients = vec![0.0; particle_count * output_dims];
    if let Some(color_outputs) = add_boosted_surface_color_output_objective(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        &motion.updates,
        cfg.opacity_gain * DIRECT_GROWTH_SURFACE_COLOR_GAIN_FRACTION,
        cfg.seed_scale,
        cfg.material_max_opacity_update,
        cfg.liveness_front_radius,
        Some(&liveness.liveness_candidate_weights),
        cfg.direct_output_gradient_rms_cap * 0.5,
        &mut surface_color_output_gradients,
    ) {
        accumulate_output_channels(
            &mut accumulators.surface_color,
            &surface_color_output_gradients,
            particle_count,
            output_dims,
            color_outputs,
        );
    }

    let mut scale_budget_output_gradients = vec![0.0; particle_count * output_dims];
    if let Some(scale_output) = add_gaussian_scale_budget_output_objective(
        &model.config,
        &snapshot.states,
        &motion.updates,
        cfg.render,
        cfg.scale_budget_weight,
        cfg.max_opacity_update,
        &mut scale_budget_output_gradients,
    ) {
        accumulate_output_channels(
            &mut accumulators.scale_budget,
            &scale_budget_output_gradients,
            particle_count,
            output_dims,
            [scale_output],
        );
    }

    let combined_additions: [&[f32]; 24] = [
        &liveness.mesh_liveness_output_gradients,
        &liveness.surface_escape_liveness_output_gradients,
        &liveness.target_coverage_liveness_output_gradients,
        &liveness.material_coverage_liveness_output_gradients,
        &liveness.material_visible_liveness_output_gradients,
        &liveness.extent_front_output_gradients,
        &liveness.phase_output_gradients,
        &liveness.liveness_phase_memory_output_gradients,
        &motion.mesh_output_gradients,
        &motion.extent_front_motion_output_gradients,
        &motion.temporal_extent_motion_output_gradients,
        &motion.extent_motion_memory_output_gradients,
        &motion.material_coverage_motion_output_gradients,
        &material_surface_motion_output_gradients,
        &motion.residual_velocity_output_gradients,
        &motion.motion_memory_output_gradients,
        &motion.material_coverage_motion_memory_output_gradients,
        &material_coverage_materialization_output_gradients,
        &temporal_materialization_output_gradients,
        &active_surface_materialization_output_gradients,
        &strict_surface_materialization_output_gradients,
        &material_output_gradients,
        &surface_color_output_gradients,
        &scale_budget_output_gradients,
    ];
    accumulate_combined_direct_gradients(
        accumulators,
        DirectCombinedGradientInputs {
            config: &model.config,
            cfg,
            output_dims,
            particle_count,
            liveness_update_cap: channels.liveness_update_cap,
            liveness_output: channels.liveness_output,
            phase_output: channels.phase_output,
            material_output: channels.material_output,
            scale_output: channels.scale_output,
            color_outputs: channels.color_outputs,
        },
        liveness.temporal_output_gradients.clone(),
        &combined_additions,
    );
}
