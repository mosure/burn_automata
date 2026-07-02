#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

mod multiseed;
mod weights;

use super::gradients::RenderProxyGradientRows;
use super::*;

pub(crate) use multiseed::*;
pub(crate) use weights::*;

pub(crate) fn render_direct_rollout_training_step(
    model: &mut NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    trace: &crate::RolloutTrace,
    trajectory: &[RenderTrajectorySnapshot],
    gradient: &RenderProxyGradientRows,
    cfg: &RenderProxyTrainingConfig,
    rollout_seed: u64,
) -> Result<TrainingRunReport, Box<dyn std::error::Error>> {
    if trajectory.is_empty() {
        return Err(std::io::Error::other(
            "direct rollout render training requires trajectory snapshots",
        )
        .into());
    }
    let rows = gradient
        .gradients
        .len()
        .min(gradient.row_indices.len())
        .min(gradient.opacity_gradients.len())
        .min(gradient.scale_gradients.len())
        .min(gradient.color_gradients.len());
    if rows == 0 {
        return Err(std::io::Error::other("direct rollout gradient produced no rows").into());
    }

    let input_dims = model.config.perception_dims();
    let output_dims = model.config.update_dims();
    let particle_count = trace.positions.len();
    let liveness_update_cap =
        liveness_max_update(cfg.max_opacity_update, cfg.liveness_update_multiplier);
    let terminal_liveness_gain = direct_terminal_liveness_gain(cfg);
    let mut accumulated_gradients = zero_supervised_gradients(model);
    accumulated_gradients
        .features
        .reserve(trajectory.len() * particle_count * input_dims);
    let mut state_adjoint = terminal_render_state_adjoint(
        &model.config,
        trace,
        gradient,
        cfg.opacity_gain,
        cfg.scale_gain,
        cfg.scale_budget_weight,
        terminal_liveness_gain,
        cfg.liveness_front_radius,
        1.0,
        liveness_update_cap,
        cfg.render,
        rows,
    );
    add_surface_material_opacity_state_adjoint(
        &model.config,
        target,
        &trace.positions,
        &trace.states,
        cfg.opacity_gain,
        cfg.seed_scale,
        GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET,
        cfg.material_max_opacity_update,
        &mut state_adjoint,
    );
    add_material_target_coverage_state_adjoint(
        &model.config,
        target,
        &trace.positions,
        &trace.states,
        cfg.opacity_gain,
        cfg.coverage_samples,
        cfg.seed_scale,
        GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET,
        cfg.material_max_opacity_update,
        &mut state_adjoint,
    );
    add_material_surface_strata_state_adjoint(
        &model.config,
        target,
        &trace.positions,
        &trace.states,
        cfg.opacity_gain,
        cfg.coverage_samples,
        cfg.seed_scale,
        GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET,
        cfg.material_max_opacity_update,
        &mut state_adjoint,
    );
    add_material_liveness_state_adjoint(
        &model.config,
        &trace.states,
        cfg.material_liveness_gain,
        material_suppression_max_update(
            cfg.material_max_opacity_update,
            cfg.material_suppression_update_multiplier,
        ),
        &mut state_adjoint,
    );
    add_material_visible_liveness_state_adjoint(
        &model.config,
        target,
        &trace.positions,
        &trace.states,
        cfg.material_liveness_gain,
        target_coverage_threshold(cfg.seed_scale),
        liveness_update_cap,
        &mut state_adjoint,
    );
    add_material_visible_surface_tail_state_adjoint(
        &model.config,
        target,
        &trace.positions,
        &trace.states,
        cfg.material_tail_gain,
        GROWTH_3D_SURFACE_MAX_DISTANCE,
        material_suppression_max_update(
            cfg.material_max_opacity_update,
            cfg.material_suppression_update_multiplier,
        ),
        &mut state_adjoint,
    );
    add_surface_escape_state_adjoint(
        &model.config,
        target,
        &trace.positions,
        &trace.states,
        cfg.surface_escape_gain,
        cfg.opacity_gain,
        cfg.liveness_gain,
        cfg.material_max_opacity_update,
        &mut state_adjoint,
    );
    let final_coverage_updates = render_proxy_target_coverage_updates(
        &model.config,
        target,
        &trace.positions,
        &trace.states,
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
    let mut position_adjoint = terminal_render_position_adjoint(
        &model.config,
        trace,
        gradient,
        &final_coverage_updates,
        cfg.motion_gain,
        cfg.full_coverage_adjoint,
        rows,
    );
    add_surface_position_adjoint(
        &model.config,
        target,
        &trace.positions,
        &trace.states,
        cfg.surface_gain,
        cfg.surface_escape_gain,
        &mut position_adjoint,
    );
    add_material_visible_surface_position_adjoint(
        &model.config,
        target,
        &trace.positions,
        &trace.states,
        cfg.surface_gain,
        cfg.surface_escape_gain,
        cfg.seed_scale,
        cfg.liveness_front_radius,
        &mut position_adjoint,
    );
    add_material_visible_surface_coverage_position_adjoint(
        &model.config,
        target,
        &trace.positions,
        &trace.states,
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
        &mut position_adjoint,
    );
    let trajectory_adjoint =
        trajectory_render_adjoints(&model.config, target, trajectory, trace, cfg)?;
    let dt = 1.0_f32;
    let perception_options = PerceptionOptions {
        state_grad: model.config.state_grad,
        density_grad: model.config.density_grad,
        eps0: model.config.eps0,
        scale_equivariance: model.config.scale_equivariant(),
        particle_density_equivariance: model.config.particle_density_equivariant(),
        log_norm_grad: model.config.log_norm_grad,
        log_norm_density_grad: model.config.log_norm_density_grad,
        hybrid_state_gradient: true,
        position_features: model.config.position_features,
    };

    for snapshot_idx in (0..trajectory.len()).rev() {
        let snapshot = &trajectory[snapshot_idx];
        if let Some(snapshot_adjoint) = trajectory_adjoint[snapshot_idx].as_ref() {
            for particle_row in 0..particle_count {
                if particle_row >= snapshot_adjoint.position.len()
                    || particle_row * model.config.state_dims + model.config.state_dims
                        > snapshot_adjoint.state.len()
                {
                    continue;
                }
                for axis in 0..model.config.spatial_dims {
                    position_adjoint[particle_row][axis] +=
                        snapshot_adjoint.weight * snapshot_adjoint.position[particle_row][axis];
                }
                clamp_position_adjoint_row(
                    &mut position_adjoint[particle_row],
                    model.config.spatial_dims,
                );
                let state_base = particle_row * model.config.state_dims;
                for channel in 0..model.config.state_dims {
                    state_adjoint[state_base + channel] +=
                        snapshot_adjoint.weight * snapshot_adjoint.state[state_base + channel];
                }
                clamp_state_adjoint_row(
                    &mut state_adjoint[state_base..state_base + model.config.state_dims],
                );
            }
        }
        let updates = model.forward_update_from_features(&snapshot.features)?;
        let step_features = snapshot.features.clone();
        let mut step_output_gradients = vec![0.0; particle_count * output_dims];
        for particle_row in 0..particle_count {
            if particle_row >= snapshot.positions.len() {
                return Err(std::io::Error::other(format!(
                    "direct rollout gradient row {particle_row} out of range for {} particles",
                    snapshot.positions.len()
                ))
                .into());
            }
            let raw_base = particle_row * output_dims;
            let output_base = particle_row * output_dims;
            accumulate_motion_output_gradient(
                &model.config,
                grid.eps,
                &updates[raw_base..raw_base + output_dims],
                [
                    position_adjoint[particle_row][0] * dt,
                    position_adjoint[particle_row][1] * dt,
                    position_adjoint[particle_row][2] * dt,
                ],
                &mut step_output_gradients[output_base..output_base + output_dims],
            );

            let state_base = particle_row * model.config.state_dims;
            let update_state_base = model.config.spatial_dims;
            for channel in 0..model.config.state_dims {
                step_output_gradients[output_base + update_state_base + channel] +=
                    state_adjoint[state_base + channel] * dt;
            }
        }
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
            &updates,
            &mesh_motion_candidate_weights,
            cfg.liveness_gain * DIRECT_GROWTH_MESH_MOTION_LIVENESS_GAIN_FRACTION,
            liveness_update_cap,
            &mut step_output_gradients,
        );
        add_candidate_liveness_output_objective(
            &model.config,
            &snapshot.positions,
            &snapshot.states,
            &updates,
            &target_coverage_candidate_weights,
            cfg.liveness_gain * DIRECT_GROWTH_TARGET_COVERAGE_LIVENESS_GAIN_FRACTION,
            liveness_update_cap,
            &mut step_output_gradients,
        );
        add_candidate_liveness_output_objective(
            &model.config,
            &snapshot.positions,
            &snapshot.states,
            &updates,
            &material_coverage_candidate_weights,
            cfg.liveness_gain * DIRECT_GROWTH_MATERIAL_COVERAGE_LIVENESS_GAIN_FRACTION,
            liveness_update_cap,
            &mut step_output_gradients,
        );
        add_surface_escape_liveness_output_objective(
            &model.config,
            target,
            &snapshot.positions,
            &snapshot.states,
            &updates,
            cfg.surface_escape_gain,
            cfg.liveness_gain,
            liveness_update_cap,
            1.0,
            &mut step_output_gradients,
        );
        add_candidate_liveness_output_objective(
            &model.config,
            &snapshot.positions,
            &snapshot.states,
            &updates,
            &extent_front_candidate_weights,
            cfg.liveness_gain * DIRECT_GROWTH_EXTENT_FRONT_LIVENESS_GAIN_FRACTION,
            liveness_update_cap,
            &mut step_output_gradients,
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
            &updates,
            snapshot.step_fraction,
            cfg.liveness_front_radius,
            DIRECT_GROWTH_LOCAL_FRONT_LIVENESS_FLOOR,
            &liveness_candidate_weights,
        );
        add_temporal_liveness_output_objective_with_candidate_weights(
            &model.config,
            &snapshot.positions,
            &snapshot.states,
            &updates,
            snapshot.step_fraction,
            cfg.liveness_gain,
            cfg.liveness_front_radius,
            Some(&liveness_candidate_weights),
            &mut step_output_gradients,
        );
        let mut liveness_phase_memory_output_gradients = vec![0.0; particle_count * output_dims];
        add_liveness_phase_memory_output_objective(
            &model.config,
            &step_output_gradients,
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
            &mut step_output_gradients,
            &liveness_phase_memory_output_gradients,
        );
        add_growth_phase_output_objective(
            &model.config,
            &snapshot.positions,
            &snapshot.states,
            &updates,
            snapshot.step_fraction,
            direct_growth_phase_gain(cfg),
            cfg.liveness_front_radius,
            &mut step_output_gradients,
        );
        let mut material_coverage_materialization_output_gradients =
            vec![0.0; particle_count * output_dims];
        add_material_coverage_materialization_output_objective(
            &model.config,
            target,
            &snapshot.positions,
            &snapshot.states,
            &updates,
            snapshot.step_fraction,
            cfg.opacity_gain * DIRECT_GROWTH_MATERIAL_COVERAGE_MATERIALIZATION_GAIN_FRACTION,
            &material_coverage_candidate_weights,
            cfg.seed_scale,
            cfg.material_max_opacity_update,
            &mut material_coverage_materialization_output_gradients,
        );
        if let Some(material_channel) = growth_3d_material_opacity_channel(model.config.state_dims)
        {
            boost_sparse_output_channel_rms(
                &mut material_coverage_materialization_output_gradients,
                output_dims,
                [model.config.spatial_dims + material_channel],
                cfg.direct_output_gradient_rms_cap,
                32.0,
            );
        }
        add_output_gradients(
            &mut step_output_gradients,
            &material_coverage_materialization_output_gradients,
        );
        let mut temporal_materialization_output_gradients = vec![0.0; particle_count * output_dims];
        add_temporal_materialization_output_objective_with_candidate_weights(
            &model.config,
            target,
            &snapshot.positions,
            &snapshot.states,
            &updates,
            snapshot.step_fraction,
            cfg.opacity_gain * DIRECT_GROWTH_TEMPORAL_MATERIALIZATION_GAIN_FRACTION,
            cfg.liveness_front_radius,
            &liveness_candidate_weights,
            cfg.seed_scale,
            cfg.material_max_opacity_update,
            &mut temporal_materialization_output_gradients,
        );
        if let Some(material_channel) = growth_3d_material_opacity_channel(model.config.state_dims)
        {
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
            &updates,
            cfg.opacity_gain * DIRECT_GROWTH_ACTIVE_SURFACE_MATERIALIZATION_GAIN_FRACTION,
            cfg.seed_scale,
            cfg.material_max_opacity_update,
            cfg.liveness_front_radius,
            Some(&liveness_candidate_weights),
            &mut active_surface_materialization_output_gradients,
        );
        if let Some(material_channel) = growth_3d_material_opacity_channel(model.config.state_dims)
        {
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
            &updates,
            cfg.opacity_gain * DIRECT_GROWTH_STRICT_SURFACE_MATERIALIZATION_GAIN_FRACTION,
            cfg.seed_scale,
            cfg.material_max_opacity_update,
            &mut strict_surface_materialization_output_gradients,
        );
        if let Some(material_channel) = growth_3d_material_opacity_channel(model.config.state_dims)
        {
            boost_sparse_output_channel_rms(
                &mut strict_surface_materialization_output_gradients,
                output_dims,
                [model.config.spatial_dims + material_channel],
                cfg.direct_output_gradient_rms_cap
                    * DIRECT_GROWTH_MATERIAL_OUTPUT_RMS_CAP_MULTIPLIER,
                16.0,
            );
        }
        let mut material_surface_motion_output_gradients = vec![0.0; particle_count * output_dims];
        add_material_visible_surface_approach_output_objective(
            &model.config,
            target,
            &snapshot.positions,
            &snapshot.states,
            &updates,
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
            &updates,
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
        add_output_gradients(&mut step_output_gradients, &mesh_output_gradients);
        add_output_gradients(
            &mut step_output_gradients,
            &extent_front_motion_output_gradients,
        );
        add_output_gradients(
            &mut step_output_gradients,
            &temporal_extent_motion_output_gradients,
        );
        add_output_gradients(
            &mut step_output_gradients,
            &extent_motion_memory_output_gradients,
        );
        add_output_gradients(
            &mut step_output_gradients,
            &material_coverage_motion_output_gradients,
        );
        add_output_gradients(
            &mut step_output_gradients,
            &material_surface_motion_output_gradients,
        );
        add_output_gradients(
            &mut step_output_gradients,
            &residual_velocity_output_gradients,
        );
        add_output_gradients(&mut step_output_gradients, &motion_memory_output_gradients);
        add_output_gradients(
            &mut step_output_gradients,
            &material_coverage_motion_memory_output_gradients,
        );
        add_output_gradients(
            &mut step_output_gradients,
            &temporal_materialization_output_gradients,
        );
        add_output_gradients(
            &mut step_output_gradients,
            &active_surface_materialization_output_gradients,
        );
        add_output_gradients(
            &mut step_output_gradients,
            &strict_surface_materialization_output_gradients,
        );
        let mut material_output_gradients = vec![0.0; particle_count * output_dims];
        add_material_visibility_output_objective(
            &model.config,
            target,
            &snapshot.positions,
            &snapshot.states,
            &updates,
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
        if let Some(material_channel) = growth_3d_material_opacity_channel(model.config.state_dims)
        {
            boost_sparse_output_channel_rms(
                &mut material_output_gradients,
                output_dims,
                [model.config.spatial_dims + material_channel],
                cfg.direct_output_gradient_rms_cap * 0.5,
                16.0,
            );
        }
        add_output_gradients(&mut step_output_gradients, &material_output_gradients);
        let mut surface_color_output_gradients = vec![0.0; particle_count * output_dims];
        add_boosted_surface_color_output_objective(
            &model.config,
            target,
            &snapshot.positions,
            &snapshot.states,
            &updates,
            cfg.opacity_gain * DIRECT_GROWTH_SURFACE_COLOR_GAIN_FRACTION,
            cfg.seed_scale,
            cfg.material_max_opacity_update,
            cfg.liveness_front_radius,
            Some(&liveness_candidate_weights),
            cfg.direct_output_gradient_rms_cap * 0.5,
            &mut surface_color_output_gradients,
        );
        add_output_gradients(&mut step_output_gradients, &surface_color_output_gradients);
        boost_sparse_output_channel_rms(
            &mut step_output_gradients,
            output_dims,
            0..model.config.spatial_dims,
            cfg.direct_output_gradient_rms_cap * DIRECT_GROWTH_SPATIAL_MOTION_RMS_TARGET_FRACTION,
            8.0,
        );
        cap_output_gradient_channel_rms_with_state_caps(
            &model.config,
            &mut step_output_gradients,
            output_dims,
            cfg.direct_output_gradient_rms_cap,
            liveness_update_cap,
            cfg.direct_output_gradient_rms_cap * DIRECT_GROWTH_MATERIAL_OUTPUT_RMS_CAP_MULTIPLIER,
        );

        let step_gradients =
            mlp_backward_from_output_gradients(model, &step_features, &step_output_gradients)?;
        let perception_adjoint = perceive_adjoint_with_options(
            &snapshot.positions,
            &snapshot.states,
            trace.batch_size,
            trace.particle_count,
            model.config.state_dims,
            grid,
            perception_options,
            &step_gradients.features,
        )?;
        for particle_row in 0..particle_count {
            for axis in 0..model.config.spatial_dims {
                position_adjoint[particle_row][axis] +=
                    cfg.perception_position_gain * perception_adjoint.position[particle_row][axis];
            }
            clamp_position_adjoint_row(
                &mut position_adjoint[particle_row],
                model.config.spatial_dims,
            );
            let state_base = particle_row * model.config.state_dims;
            for channel in 0..model.config.state_dims {
                state_adjoint[state_base + channel] +=
                    perception_adjoint.state[state_base + channel];
            }
            clamp_state_adjoint_row(
                &mut state_adjoint[state_base..state_base + model.config.state_dims],
            );
        }
        accumulate_supervised_gradients(&mut accumulated_gradients, &step_gradients);
    }

    let mut render_cfg = cfg.render;
    if render_cfg.target_samples == 0 {
        render_cfg.target_samples = cfg.particles;
    }
    let initial_loss = mesh_multiview_render_loss_from_trace(trace, target, render_cfg)?.total_loss;
    normalize_direct_rollout_gradients(&mut accumulated_gradients, input_dims);
    if cfg.direct_material_output_only {
        retain_material_output_gradients(model, &mut accumulated_gradients)?;
    }
    let step = apply_sgd_gradients(model, &accumulated_gradients, cfg.sgd)?;
    let final_trace = render_training_trace_for_seed(model, grid, cfg, rollout_seed)?;
    let final_loss =
        mesh_multiview_render_loss_from_trace(&final_trace, target, render_cfg)?.total_loss;
    let best_loss = initial_loss.min(final_loss);
    Ok(TrainingRunReport {
        steps: 1,
        rows: step.rows,
        initial_loss,
        final_loss,
        best_loss,
        history: vec![TrainingHistoryEntry {
            step: 1,
            loss: final_loss,
            grad_norm: step.grad_norm,
            grad_scale: step.grad_scale,
        }],
    })
}
