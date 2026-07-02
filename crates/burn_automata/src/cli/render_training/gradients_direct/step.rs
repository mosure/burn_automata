#![allow(clippy::needless_range_loop)]

use super::*;

pub(super) fn add_direct_step_output_objectives(
    model: &NpaModel,
    target: &TriangleMeshTarget,
    snapshot: &RenderTrajectorySnapshot,
    updates: &[f32],
    cfg: &RenderProxyTrainingConfig,
    liveness_update_cap: f32,
    step_output_gradients: &mut [f32],
) {
    let output_dims = model.config.update_dims();
    let particle_count = snapshot.positions.len();
    let mut mesh_output_gradients = vec![0.0; particle_count * output_dims];
    add_mesh_geometry_output_objective(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        updates,
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
    boost_sparse_output_channel_rms(
        &mut mesh_output_gradients,
        output_dims,
        0..model.config.spatial_dims,
        cfg.direct_output_gradient_rms_cap * 0.5,
        16.0,
    );
    let mut motion_memory_output_gradients = vec![0.0; particle_count * output_dims];
    add_motion_memory_output_objective(
        &model.config,
        &mesh_output_gradients,
        DIRECT_GROWTH_MOTION_MEMORY_GAIN_FRACTION,
        &mut motion_memory_output_gradients,
    );
    let velocity_output_channels = growth_3d_velocity_output_channels(&model.config);
    if !velocity_output_channels.is_empty() {
        boost_sparse_output_channel_rms(
            &mut motion_memory_output_gradients,
            output_dims,
            velocity_output_channels,
            cfg.direct_output_gradient_rms_cap * 0.5,
            16.0,
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
        updates,
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
        updates,
        cfg.liveness_front_radius,
        cfg.extent_gain,
        cfg.max_update_norm,
        cfg.trajectory_mesh_gain
            * direct_trajectory_geometry_weight(snapshot.step_fraction)
            * DIRECT_GROWTH_EXTENT_FRONT_MOTION_GAIN_FRACTION,
        &mut extent_front_motion_output_gradients,
    );
    boost_sparse_output_channel_rms(
        &mut extent_front_motion_output_gradients,
        output_dims,
        0..model.config.spatial_dims,
        cfg.direct_output_gradient_rms_cap * 0.5,
        16.0,
    );
    let mut temporal_extent_motion_output_gradients = vec![0.0; particle_count * output_dims];
    add_temporal_extent_motion_output_objective(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        updates,
        snapshot.step_fraction,
        cfg.liveness_front_radius,
        cfg.extent_gain,
        cfg.max_update_norm,
        cfg.trajectory_mesh_gain
            * direct_trajectory_geometry_weight(snapshot.step_fraction)
            * DIRECT_GROWTH_TEMPORAL_EXTENT_MOTION_GAIN_FRACTION,
        &mut temporal_extent_motion_output_gradients,
    );
    boost_sparse_output_channel_rms(
        &mut temporal_extent_motion_output_gradients,
        output_dims,
        0..model.config.spatial_dims,
        cfg.direct_output_gradient_rms_cap * 0.5,
        16.0,
    );
    let mut extent_motion_memory_output_gradients = vec![0.0; particle_count * output_dims];
    add_extent_motion_memory_output_objective(
        &model.config,
        &extent_front_motion_output_gradients,
        &temporal_extent_motion_output_gradients,
        DIRECT_GROWTH_EXTENT_MOTION_MEMORY_GAIN_FRACTION,
        &mut extent_motion_memory_output_gradients,
    );
    let extent_motion_velocity_outputs = growth_3d_velocity_output_channels(&model.config);
    if !extent_motion_velocity_outputs.is_empty() {
        boost_sparse_output_channel_rms(
            &mut extent_motion_memory_output_gradients,
            output_dims,
            extent_motion_velocity_outputs,
            cfg.direct_output_gradient_rms_cap * 0.5,
            16.0,
        );
    }
    let mut material_coverage_motion_output_gradients = vec![0.0; particle_count * output_dims];
    add_material_coverage_front_motion_output_objective(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        updates,
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
    boost_sparse_output_channel_rms(
        &mut material_coverage_motion_output_gradients,
        output_dims,
        0..model.config.spatial_dims,
        cfg.direct_output_gradient_rms_cap * 0.5,
        16.0,
    );
    let mut material_coverage_motion_memory_output_gradients =
        vec![0.0; particle_count * output_dims];
    add_motion_memory_output_objective(
        &model.config,
        &material_coverage_motion_output_gradients,
        DIRECT_GROWTH_MATERIAL_COVERAGE_MOTION_MEMORY_GAIN_FRACTION,
        &mut material_coverage_motion_memory_output_gradients,
    );
    let material_coverage_velocity_outputs = growth_3d_velocity_output_channels(&model.config);
    if !material_coverage_velocity_outputs.is_empty() {
        boost_sparse_output_channel_rms(
            &mut material_coverage_motion_memory_output_gradients,
            output_dims,
            material_coverage_velocity_outputs,
            cfg.direct_output_gradient_rms_cap * 0.5,
            16.0,
        );
    }
    let mut residual_velocity_output_gradients = vec![0.0; particle_count * output_dims];
    add_mesh_residual_velocity_output_objective(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        updates,
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
    let residual_velocity_outputs = growth_3d_velocity_output_channels(&model.config);
    if !residual_velocity_outputs.is_empty() {
        boost_sparse_output_channel_rms(
            &mut residual_velocity_output_gradients,
            output_dims,
            residual_velocity_outputs,
            cfg.direct_output_gradient_rms_cap * 0.5,
            16.0,
        );
    }
    add_mesh_motion_liveness_output_objective(
        &model.config,
        &snapshot.positions,
        &snapshot.states,
        updates,
        &mesh_motion_candidate_weights,
        cfg.liveness_gain * DIRECT_GROWTH_MESH_MOTION_LIVENESS_GAIN_FRACTION,
        liveness_update_cap,
        step_output_gradients,
    );
    add_candidate_liveness_output_objective(
        &model.config,
        &snapshot.positions,
        &snapshot.states,
        updates,
        &target_coverage_candidate_weights,
        cfg.liveness_gain * DIRECT_GROWTH_TARGET_COVERAGE_LIVENESS_GAIN_FRACTION,
        liveness_update_cap,
        step_output_gradients,
    );
    add_candidate_liveness_output_objective(
        &model.config,
        &snapshot.positions,
        &snapshot.states,
        updates,
        &material_coverage_candidate_weights,
        cfg.liveness_gain * DIRECT_GROWTH_MATERIAL_COVERAGE_LIVENESS_GAIN_FRACTION,
        liveness_update_cap,
        step_output_gradients,
    );
    add_surface_escape_liveness_output_objective(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        updates,
        cfg.surface_escape_gain,
        cfg.liveness_gain,
        liveness_update_cap,
        1.0,
        step_output_gradients,
    );
    add_candidate_liveness_output_objective(
        &model.config,
        &snapshot.positions,
        &snapshot.states,
        updates,
        &extent_front_candidate_weights,
        cfg.liveness_gain * DIRECT_GROWTH_EXTENT_FRONT_LIVENESS_GAIN_FRACTION,
        liveness_update_cap,
        step_output_gradients,
    );
    let liveness_candidate_weights = mesh_motion_candidate_weights_with_local_front_floor(
        &model.config,
        &snapshot.positions,
        &snapshot.states,
        cfg.liveness_front_radius,
        DIRECT_GROWTH_LOCAL_FRONT_LIVENESS_FLOOR,
        &mesh_motion_candidate_weights,
    );
    let liveness_candidate_weights =
        max_candidate_weights(&liveness_candidate_weights, &extent_front_candidate_weights);
    let liveness_candidate_weights = max_candidate_weights(
        &liveness_candidate_weights,
        &target_coverage_candidate_weights,
    );
    let liveness_candidate_weights = max_candidate_weights(
        &liveness_candidate_weights,
        &material_coverage_candidate_weights,
    );
    let liveness_candidate_weights = temporal_liveness_candidate_weights_with_local_front_floor(
        &model.config,
        &snapshot.positions,
        &snapshot.states,
        updates,
        snapshot.step_fraction,
        cfg.liveness_front_radius,
        DIRECT_GROWTH_LOCAL_FRONT_LIVENESS_FLOOR,
        &liveness_candidate_weights,
    );
    add_temporal_liveness_output_objective_with_candidate_weights(
        &model.config,
        &snapshot.positions,
        &snapshot.states,
        updates,
        snapshot.step_fraction,
        cfg.liveness_gain,
        cfg.liveness_front_radius,
        Some(&liveness_candidate_weights),
        step_output_gradients,
    );
    let mut liveness_phase_memory_output_gradients = vec![0.0; particle_count * output_dims];
    add_liveness_phase_memory_output_objective(
        &model.config,
        step_output_gradients,
        DIRECT_GROWTH_LIVENESS_PHASE_MEMORY_GAIN_FRACTION,
        &mut liveness_phase_memory_output_gradients,
    );
    if let Some(phase_channel) = growth_3d_phase_channel(model.config.state_dims) {
        boost_sparse_output_channel_rms(
            &mut liveness_phase_memory_output_gradients,
            output_dims,
            [model.config.spatial_dims + phase_channel],
            cfg.direct_output_gradient_rms_cap * 0.5,
            16.0,
        );
    }
    add_output_gradients(
        step_output_gradients,
        &liveness_phase_memory_output_gradients,
    );
    add_growth_phase_output_objective(
        &model.config,
        &snapshot.positions,
        &snapshot.states,
        updates,
        snapshot.step_fraction,
        direct_growth_phase_gain(cfg),
        cfg.liveness_front_radius,
        step_output_gradients,
    );
    let mut material_coverage_materialization_output_gradients =
        vec![0.0; particle_count * output_dims];
    add_material_coverage_materialization_output_objective(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        updates,
        snapshot.step_fraction,
        cfg.opacity_gain * DIRECT_GROWTH_MATERIAL_COVERAGE_MATERIALIZATION_GAIN_FRACTION,
        &material_coverage_candidate_weights,
        cfg.seed_scale,
        cfg.material_max_opacity_update,
        &mut material_coverage_materialization_output_gradients,
    );
    if let Some(material_channel) = growth_3d_material_opacity_channel(model.config.state_dims) {
        boost_sparse_output_channel_rms(
            &mut material_coverage_materialization_output_gradients,
            output_dims,
            [model.config.spatial_dims + material_channel],
            cfg.direct_output_gradient_rms_cap,
            32.0,
        );
    }
    add_output_gradients(
        step_output_gradients,
        &material_coverage_materialization_output_gradients,
    );
    let mut temporal_materialization_output_gradients = vec![0.0; particle_count * output_dims];
    add_temporal_materialization_output_objective_with_candidate_weights(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        updates,
        snapshot.step_fraction,
        cfg.opacity_gain * DIRECT_GROWTH_TEMPORAL_MATERIALIZATION_GAIN_FRACTION,
        cfg.liveness_front_radius,
        &liveness_candidate_weights,
        cfg.seed_scale,
        cfg.material_max_opacity_update,
        &mut temporal_materialization_output_gradients,
    );
    if let Some(material_channel) = growth_3d_material_opacity_channel(model.config.state_dims) {
        boost_sparse_output_channel_rms(
            &mut temporal_materialization_output_gradients,
            output_dims,
            [model.config.spatial_dims + material_channel],
            cfg.direct_output_gradient_rms_cap * 0.5,
            16.0,
        );
    }
    let mut active_surface_materialization_output_gradients =
        vec![0.0; particle_count * output_dims];
    add_active_surface_materialization_output_objective(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        updates,
        cfg.opacity_gain * DIRECT_GROWTH_ACTIVE_SURFACE_MATERIALIZATION_GAIN_FRACTION,
        cfg.seed_scale,
        cfg.material_max_opacity_update,
        cfg.liveness_front_radius,
        Some(&liveness_candidate_weights),
        &mut active_surface_materialization_output_gradients,
    );
    if let Some(material_channel) = growth_3d_material_opacity_channel(model.config.state_dims) {
        boost_sparse_output_channel_rms(
            &mut active_surface_materialization_output_gradients,
            output_dims,
            [model.config.spatial_dims + material_channel],
            cfg.direct_output_gradient_rms_cap * 0.5,
            16.0,
        );
    }
    let mut strict_surface_materialization_output_gradients =
        vec![0.0; particle_count * output_dims];
    add_strict_surface_materialization_output_objective(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        updates,
        cfg.opacity_gain * DIRECT_GROWTH_STRICT_SURFACE_MATERIALIZATION_GAIN_FRACTION,
        cfg.seed_scale,
        cfg.material_max_opacity_update,
        &mut strict_surface_materialization_output_gradients,
    );
    if let Some(material_channel) = growth_3d_material_opacity_channel(model.config.state_dims) {
        boost_sparse_output_channel_rms(
            &mut strict_surface_materialization_output_gradients,
            output_dims,
            [model.config.spatial_dims + material_channel],
            cfg.direct_output_gradient_rms_cap * DIRECT_GROWTH_MATERIAL_OUTPUT_RMS_CAP_MULTIPLIER,
            16.0,
        );
    }
    let mut material_surface_motion_output_gradients = vec![0.0; particle_count * output_dims];
    add_material_visible_surface_approach_output_objective(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        updates,
        cfg.surface_gain,
        cfg.surface_escape_gain,
        cfg.max_update_norm,
        cfg.seed_scale,
        cfg.liveness_front_radius,
        Some(&liveness_candidate_weights),
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
        updates,
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
        Some(&liveness_candidate_weights),
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
    add_output_gradients(step_output_gradients, &mesh_output_gradients);
    add_output_gradients(step_output_gradients, &extent_front_motion_output_gradients);
    add_output_gradients(
        step_output_gradients,
        &temporal_extent_motion_output_gradients,
    );
    add_output_gradients(
        step_output_gradients,
        &extent_motion_memory_output_gradients,
    );
    add_output_gradients(
        step_output_gradients,
        &material_coverage_motion_output_gradients,
    );
    add_output_gradients(
        step_output_gradients,
        &material_surface_motion_output_gradients,
    );
    add_output_gradients(step_output_gradients, &residual_velocity_output_gradients);
    add_output_gradients(step_output_gradients, &motion_memory_output_gradients);
    add_output_gradients(
        step_output_gradients,
        &material_coverage_motion_memory_output_gradients,
    );
    add_output_gradients(
        step_output_gradients,
        &temporal_materialization_output_gradients,
    );
    add_output_gradients(
        step_output_gradients,
        &active_surface_materialization_output_gradients,
    );
    add_output_gradients(
        step_output_gradients,
        &strict_surface_materialization_output_gradients,
    );
    let mut material_output_gradients = vec![0.0; particle_count * output_dims];
    add_material_visibility_output_objective(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        updates,
        cfg.opacity_gain,
        cfg.material_liveness_gain,
        cfg.material_tail_gain,
        cfg.coverage_samples,
        cfg.seed_scale,
        cfg.material_max_opacity_update,
        cfg.material_suppression_update_multiplier,
        cfg.liveness_front_radius,
        Some(&liveness_candidate_weights),
        snapshot.step_fraction,
        liveness_update_cap,
        1.0,
        &mut material_output_gradients,
    );
    if let Some(material_channel) = growth_3d_material_opacity_channel(model.config.state_dims) {
        boost_sparse_output_channel_rms(
            &mut material_output_gradients,
            output_dims,
            [model.config.spatial_dims + material_channel],
            cfg.direct_output_gradient_rms_cap * 0.5,
            16.0,
        );
    }
    add_output_gradients(step_output_gradients, &material_output_gradients);
    let mut surface_color_output_gradients = vec![0.0; particle_count * output_dims];
    add_boosted_surface_color_output_objective(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        updates,
        cfg.opacity_gain * DIRECT_GROWTH_SURFACE_COLOR_GAIN_FRACTION,
        cfg.seed_scale,
        cfg.material_max_opacity_update,
        cfg.liveness_front_radius,
        Some(&liveness_candidate_weights),
        cfg.direct_output_gradient_rms_cap * 0.5,
        &mut surface_color_output_gradients,
    );
    add_output_gradients(step_output_gradients, &surface_color_output_gradients);
    boost_sparse_output_channel_rms(
        step_output_gradients,
        output_dims,
        0..model.config.spatial_dims,
        cfg.direct_output_gradient_rms_cap * DIRECT_GROWTH_SPATIAL_MOTION_RMS_TARGET_FRACTION,
        8.0,
    );
    cap_output_gradient_channel_rms_with_state_caps(
        &model.config,
        step_output_gradients,
        output_dims,
        cfg.direct_output_gradient_rms_cap,
        liveness_update_cap,
        cfg.direct_output_gradient_rms_cap * DIRECT_GROWTH_MATERIAL_OUTPUT_RMS_CAP_MULTIPLIER,
    );
}
