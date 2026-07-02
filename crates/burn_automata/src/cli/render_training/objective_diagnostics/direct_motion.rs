use super::accumulator::accumulate_output_channels;
use super::direct_accumulators::DirectObjectiveAccumulators;
use super::direct_channels::DirectObjectiveChannels;
use super::*;

pub(super) struct DirectMotionObjectiveOutputs {
    pub(super) updates: Vec<f32>,
    pub(super) mesh_output_gradients: Vec<f32>,
    pub(super) motion_memory_output_gradients: Vec<f32>,
    pub(super) mesh_motion_candidate_weights: Vec<f32>,
    pub(super) extent_front_candidate_weights: Vec<f32>,
    pub(super) target_coverage_candidate_weights: Vec<f32>,
    pub(super) material_coverage_candidate_weights: Vec<f32>,
    pub(super) extent_front_motion_output_gradients: Vec<f32>,
    pub(super) temporal_extent_motion_output_gradients: Vec<f32>,
    pub(super) extent_motion_memory_output_gradients: Vec<f32>,
    pub(super) material_coverage_motion_output_gradients: Vec<f32>,
    pub(super) material_coverage_motion_memory_output_gradients: Vec<f32>,
    pub(super) residual_velocity_output_gradients: Vec<f32>,
}

pub(super) fn collect_direct_motion_objectives(
    model: &NpaModel,
    target: &TriangleMeshTarget,
    snapshot: &RenderTrajectorySnapshot,
    cfg: &RenderProxyTrainingConfig,
    channels: &DirectObjectiveChannels,
    accumulators: &mut DirectObjectiveAccumulators,
) -> Result<DirectMotionObjectiveOutputs, Box<dyn std::error::Error>> {
    let particle_count = snapshot.positions.len();
    let output_dims = channels.output_dims;
    let updates = model.forward_update_from_features(&snapshot.features)?;

    let mut mesh_output_gradients = vec![0.0; particle_count * output_dims];
    add_mesh_geometry_output_objective(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        &updates,
        cfg.coverage_gain,
        cfg.coverage_samples,
        cfg.coverage_mode,
        cfg.coverage_softness,
        cfg.coverage_repulsion_gain,
        cfg.coverage_gap_gain,
        cfg.coverage_repulsion_radius,
        cfg.coverage_normal_weight,
        cfg.extent_gain,
        cfg.surface_gain,
        cfg.surface_escape_gain,
        cfg.seed_scale,
        cfg.max_update_norm,
        cfg.liveness_front_radius,
        cfg.trajectory_mesh_gain * direct_trajectory_geometry_weight(snapshot.step_fraction),
        &mut mesh_output_gradients,
    );
    accumulate_output_channels(
        &mut accumulators.mesh_motion,
        &mesh_output_gradients,
        particle_count,
        output_dims,
        0..model.config.spatial_dims,
    );

    let mut motion_memory_output_gradients = vec![0.0; particle_count * output_dims];
    add_motion_memory_output_objective(
        &model.config,
        &mesh_output_gradients,
        DIRECT_GROWTH_MOTION_MEMORY_GAIN_FRACTION,
        &mut motion_memory_output_gradients,
    );
    let velocity_outputs = growth_3d_velocity_output_channels(&model.config);
    if !velocity_outputs.is_empty() {
        boost_sparse_output_channel_rms(
            &mut motion_memory_output_gradients,
            output_dims,
            velocity_outputs.clone(),
            cfg.direct_output_gradient_rms_cap * 0.5,
            16.0,
        );
        accumulate_output_channels(
            &mut accumulators.motion_memory,
            &motion_memory_output_gradients,
            particle_count,
            output_dims,
            velocity_outputs,
        );
    }

    let mesh_motion_candidate_weights =
        mesh_motion_candidate_weights(&model.config, output_dims, &mesh_output_gradients);
    let extent_front_candidate_weights = extent_front_liveness_candidate_weights(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        cfg.liveness_front_radius,
    );
    let target_coverage_candidate_weights = target_coverage_liveness_candidate_weights(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        cfg.liveness_front_radius,
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
    );
    let material_coverage_candidate_weights = material_coverage_liveness_candidate_weights(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        &updates,
        cfg.liveness_front_radius,
        &target_coverage_candidate_weights,
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
    );

    let mut extent_front_motion_output_gradients = vec![0.0; particle_count * output_dims];
    add_extent_front_motion_output_objective(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        &updates,
        cfg.liveness_front_radius,
        cfg.extent_gain,
        cfg.max_update_norm,
        cfg.trajectory_mesh_gain
            * direct_trajectory_geometry_weight(snapshot.step_fraction)
            * DIRECT_GROWTH_EXTENT_FRONT_MOTION_GAIN_FRACTION,
        &mut extent_front_motion_output_gradients,
    );
    accumulate_output_channels(
        &mut accumulators.extent_front_motion,
        &extent_front_motion_output_gradients,
        particle_count,
        output_dims,
        0..model.config.spatial_dims,
    );

    let mut temporal_extent_motion_output_gradients = vec![0.0; particle_count * output_dims];
    add_temporal_extent_motion_output_objective(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        &updates,
        snapshot.step_fraction,
        cfg.liveness_front_radius,
        cfg.extent_gain,
        cfg.max_update_norm,
        cfg.trajectory_mesh_gain
            * direct_trajectory_geometry_weight(snapshot.step_fraction)
            * DIRECT_GROWTH_TEMPORAL_EXTENT_MOTION_GAIN_FRACTION,
        &mut temporal_extent_motion_output_gradients,
    );
    accumulate_output_channels(
        &mut accumulators.temporal_extent_motion,
        &temporal_extent_motion_output_gradients,
        particle_count,
        output_dims,
        0..model.config.spatial_dims,
    );

    let mut extent_motion_memory_output_gradients = vec![0.0; particle_count * output_dims];
    add_extent_motion_memory_output_objective(
        &model.config,
        &extent_front_motion_output_gradients,
        &temporal_extent_motion_output_gradients,
        DIRECT_GROWTH_EXTENT_MOTION_MEMORY_GAIN_FRACTION,
        &mut extent_motion_memory_output_gradients,
    );
    let velocity_outputs = growth_3d_velocity_output_channels(&model.config);
    if !velocity_outputs.is_empty() {
        boost_sparse_output_channel_rms(
            &mut extent_motion_memory_output_gradients,
            output_dims,
            velocity_outputs.clone(),
            cfg.direct_output_gradient_rms_cap * 0.5,
            16.0,
        );
        accumulate_output_channels(
            &mut accumulators.extent_motion_memory,
            &extent_motion_memory_output_gradients,
            particle_count,
            output_dims,
            velocity_outputs,
        );
    }

    let mut material_coverage_motion_output_gradients = vec![0.0; particle_count * output_dims];
    add_material_coverage_front_motion_output_objective(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        &updates,
        cfg.liveness_front_radius,
        &material_coverage_candidate_weights,
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
        cfg.trajectory_mesh_gain
            * direct_trajectory_geometry_weight(snapshot.step_fraction)
            * DIRECT_GROWTH_MATERIAL_COVERAGE_MOTION_GAIN_FRACTION,
        &mut material_coverage_motion_output_gradients,
    );
    accumulate_output_channels(
        &mut accumulators.material_coverage_motion,
        &material_coverage_motion_output_gradients,
        particle_count,
        output_dims,
        0..model.config.spatial_dims,
    );

    let mut material_coverage_motion_memory_output_gradients =
        vec![0.0; particle_count * output_dims];
    add_motion_memory_output_objective(
        &model.config,
        &material_coverage_motion_output_gradients,
        DIRECT_GROWTH_MATERIAL_COVERAGE_MOTION_MEMORY_GAIN_FRACTION,
        &mut material_coverage_motion_memory_output_gradients,
    );
    let velocity_outputs = growth_3d_velocity_output_channels(&model.config);
    if !velocity_outputs.is_empty() {
        boost_sparse_output_channel_rms(
            &mut material_coverage_motion_memory_output_gradients,
            output_dims,
            velocity_outputs.clone(),
            cfg.direct_output_gradient_rms_cap * 0.5,
            16.0,
        );
        accumulate_output_channels(
            &mut accumulators.material_coverage_motion_memory,
            &material_coverage_motion_memory_output_gradients,
            particle_count,
            output_dims,
            velocity_outputs,
        );
    }

    let mut residual_velocity_output_gradients = vec![0.0; particle_count * output_dims];
    add_mesh_residual_velocity_output_objective(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        &updates,
        cfg.coverage_gain,
        cfg.surface_gain,
        cfg.surface_escape_gain,
        cfg.max_update_norm,
        cfg.liveness_front_radius,
        cfg.trajectory_mesh_gain
            * direct_trajectory_geometry_weight(snapshot.step_fraction)
            * DIRECT_GROWTH_RESIDUAL_VELOCITY_GAIN_FRACTION,
        &mut residual_velocity_output_gradients,
    );
    let velocity_outputs = growth_3d_velocity_output_channels(&model.config);
    if !velocity_outputs.is_empty() {
        boost_sparse_output_channel_rms(
            &mut residual_velocity_output_gradients,
            output_dims,
            velocity_outputs.clone(),
            cfg.direct_output_gradient_rms_cap * 0.5,
            16.0,
        );
        accumulate_output_channels(
            &mut accumulators.residual_velocity,
            &residual_velocity_output_gradients,
            particle_count,
            output_dims,
            velocity_outputs,
        );
    }

    Ok(DirectMotionObjectiveOutputs {
        updates,
        mesh_output_gradients,
        motion_memory_output_gradients,
        mesh_motion_candidate_weights,
        extent_front_candidate_weights,
        target_coverage_candidate_weights,
        material_coverage_candidate_weights,
        extent_front_motion_output_gradients,
        temporal_extent_motion_output_gradients,
        extent_motion_memory_output_gradients,
        material_coverage_motion_output_gradients,
        material_coverage_motion_memory_output_gradients,
        residual_velocity_output_gradients,
    })
}
