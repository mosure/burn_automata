use super::accumulator::{OutputGradientAccumulator, accumulate_output_channels};
use super::terminal::terminal_liveness_state_diagnostics;
use super::*;

pub(crate) fn direct_rollout_objective_diagnostics(
    model: &NpaModel,
    target: &TriangleMeshTarget,
    trajectory: &[RenderTrajectorySnapshot],
    cfg: &RenderProxyTrainingConfig,
) -> Result<DirectRolloutObjectiveDiagnostics, Box<dyn std::error::Error>> {
    if trajectory.is_empty() {
        return Ok(DirectRolloutObjectiveDiagnostics::default());
    }

    let output_dims = model.config.update_dims();
    let liveness_output = model.config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let material_output = growth_3d_material_opacity_channel(model.config.state_dims)
        .map(|channel| model.config.spatial_dims + channel)
        .filter(|channel| *channel < output_dims);
    let scale_output = growth_3d_scale_output_channel(&model.config, cfg.render);
    let color_outputs = growth_3d_color_output_channels(&model.config);
    let phase_output = growth_3d_phase_channel(model.config.state_dims)
        .map(|channel| model.config.spatial_dims + channel)
        .filter(|channel| *channel < output_dims);
    let liveness_update_cap =
        liveness_max_update(cfg.max_opacity_update, cfg.liveness_update_multiplier);

    let mut temporal_liveness = OutputGradientAccumulator::default();
    let mut mesh_motion_liveness = OutputGradientAccumulator::default();
    let mut surface_escape_liveness = OutputGradientAccumulator::default();
    let mut target_coverage_liveness = OutputGradientAccumulator::default();
    let mut material_coverage_liveness = OutputGradientAccumulator::default();
    let mut extent_front_liveness = OutputGradientAccumulator::default();
    let mut phase_progress = OutputGradientAccumulator::default();
    let mut liveness_phase_memory = OutputGradientAccumulator::default();
    let mut mesh_motion = OutputGradientAccumulator::default();
    let mut extent_front_motion = OutputGradientAccumulator::default();
    let mut temporal_extent_motion = OutputGradientAccumulator::default();
    let mut extent_motion_memory = OutputGradientAccumulator::default();
    let mut material_coverage_motion = OutputGradientAccumulator::default();
    let mut material_surface_motion = OutputGradientAccumulator::default();
    let mut residual_velocity = OutputGradientAccumulator::default();
    let mut motion_memory = OutputGradientAccumulator::default();
    let mut material_coverage_motion_memory = OutputGradientAccumulator::default();
    let mut material_coverage_materialization = OutputGradientAccumulator::default();
    let mut temporal_materialization = OutputGradientAccumulator::default();
    let mut active_surface_materialization = OutputGradientAccumulator::default();
    let mut strict_surface_materialization = OutputGradientAccumulator::default();
    let mut material_visibility = OutputGradientAccumulator::default();
    let mut surface_color = OutputGradientAccumulator::default();
    let mut scale_budget = OutputGradientAccumulator::default();
    let mut combined_pre_cap = OutputGradientAccumulator::default();
    let mut combined_post_cap = OutputGradientAccumulator::default();
    let mut mesh_motion_post_cap = OutputGradientAccumulator::default();
    let mut residual_velocity_post_cap = OutputGradientAccumulator::default();
    let mut motion_memory_post_cap = OutputGradientAccumulator::default();
    let mut liveness_post_cap = OutputGradientAccumulator::default();
    let mut phase_post_cap = OutputGradientAccumulator::default();
    let mut material_post_cap = OutputGradientAccumulator::default();
    let mut scale_post_cap = OutputGradientAccumulator::default();
    let mut color_post_cap = OutputGradientAccumulator::default();
    let mut rows = 0usize;

    for snapshot in trajectory {
        let particle_count = snapshot.positions.len();
        if particle_count == 0 {
            continue;
        }
        rows += particle_count;
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
            &mut mesh_motion,
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
                &mut motion_memory,
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
            &mut extent_front_motion,
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
            &mut temporal_extent_motion,
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
                &mut extent_motion_memory,
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
            &mut material_coverage_motion,
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
                &mut material_coverage_motion_memory,
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
                &mut residual_velocity,
                &residual_velocity_output_gradients,
                particle_count,
                output_dims,
                velocity_outputs,
            );
        }
        let mut mesh_liveness_output_gradients = vec![0.0; particle_count * output_dims];
        add_mesh_motion_liveness_output_objective(
            &model.config,
            &snapshot.positions,
            &snapshot.states,
            &updates,
            &mesh_motion_candidate_weights,
            cfg.liveness_gain * DIRECT_GROWTH_MESH_MOTION_LIVENESS_GAIN_FRACTION,
            liveness_update_cap,
            &mut mesh_liveness_output_gradients,
        );
        if liveness_output < output_dims {
            accumulate_output_channels(
                &mut mesh_motion_liveness,
                &mesh_liveness_output_gradients,
                particle_count,
                output_dims,
                [liveness_output],
            );
        }
        let mut target_coverage_liveness_output_gradients = vec![0.0; particle_count * output_dims];
        add_candidate_liveness_output_objective(
            &model.config,
            &snapshot.positions,
            &snapshot.states,
            &updates,
            &target_coverage_candidate_weights,
            cfg.liveness_gain * DIRECT_GROWTH_TARGET_COVERAGE_LIVENESS_GAIN_FRACTION,
            liveness_update_cap,
            &mut target_coverage_liveness_output_gradients,
        );
        if liveness_output < output_dims {
            accumulate_output_channels(
                &mut target_coverage_liveness,
                &target_coverage_liveness_output_gradients,
                particle_count,
                output_dims,
                [liveness_output],
            );
        }
        let mut material_coverage_liveness_output_gradients =
            vec![0.0; particle_count * output_dims];
        add_candidate_liveness_output_objective(
            &model.config,
            &snapshot.positions,
            &snapshot.states,
            &updates,
            &material_coverage_candidate_weights,
            cfg.liveness_gain * DIRECT_GROWTH_MATERIAL_COVERAGE_LIVENESS_GAIN_FRACTION,
            liveness_update_cap,
            &mut material_coverage_liveness_output_gradients,
        );
        if liveness_output < output_dims {
            accumulate_output_channels(
                &mut material_coverage_liveness,
                &material_coverage_liveness_output_gradients,
                particle_count,
                output_dims,
                [liveness_output],
            );
        }
        let mut surface_escape_liveness_output_gradients = vec![0.0; particle_count * output_dims];
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
            &mut surface_escape_liveness_output_gradients,
        );
        if liveness_output < output_dims {
            accumulate_output_channels(
                &mut surface_escape_liveness,
                &surface_escape_liveness_output_gradients,
                particle_count,
                output_dims,
                [liveness_output],
            );
        }
        let mut extent_front_output_gradients = vec![0.0; particle_count * output_dims];
        add_candidate_liveness_output_objective(
            &model.config,
            &snapshot.positions,
            &snapshot.states,
            &updates,
            &extent_front_candidate_weights,
            cfg.liveness_gain * DIRECT_GROWTH_EXTENT_FRONT_LIVENESS_GAIN_FRACTION,
            liveness_update_cap,
            &mut extent_front_output_gradients,
        );
        if liveness_output < output_dims {
            accumulate_output_channels(
                &mut extent_front_liveness,
                &extent_front_output_gradients,
                particle_count,
                output_dims,
                [liveness_output],
            );
        }
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
        let mut temporal_output_gradients = vec![0.0; particle_count * output_dims];
        add_temporal_liveness_output_objective_with_candidate_weights(
            &model.config,
            &snapshot.positions,
            &snapshot.states,
            &updates,
            snapshot.step_fraction,
            cfg.liveness_gain,
            cfg.liveness_front_radius,
            Some(&liveness_candidate_weights),
            &mut temporal_output_gradients,
        );
        if liveness_output < output_dims {
            accumulate_output_channels(
                &mut temporal_liveness,
                &temporal_output_gradients,
                particle_count,
                output_dims,
                [liveness_output],
            );
        }

        let mut liveness_phase_driver_gradients = temporal_output_gradients.clone();
        add_output_gradients(
            &mut liveness_phase_driver_gradients,
            &mesh_liveness_output_gradients,
        );
        add_output_gradients(
            &mut liveness_phase_driver_gradients,
            &surface_escape_liveness_output_gradients,
        );
        add_output_gradients(
            &mut liveness_phase_driver_gradients,
            &target_coverage_liveness_output_gradients,
        );
        add_output_gradients(
            &mut liveness_phase_driver_gradients,
            &material_coverage_liveness_output_gradients,
        );
        add_output_gradients(
            &mut liveness_phase_driver_gradients,
            &extent_front_output_gradients,
        );
        let mut liveness_phase_memory_output_gradients = vec![0.0; particle_count * output_dims];
        add_liveness_phase_memory_output_objective(
            &model.config,
            &liveness_phase_driver_gradients,
            DIRECT_GROWTH_LIVENESS_PHASE_MEMORY_GAIN_FRACTION,
            &mut liveness_phase_memory_output_gradients,
        );
        if let Some(phase_output) = phase_output {
            accumulate_output_channels(
                &mut liveness_phase_memory,
                &liveness_phase_memory_output_gradients,
                particle_count,
                output_dims,
                [phase_output],
            );
        }

        let mut phase_output_gradients = vec![0.0; particle_count * output_dims];
        add_growth_phase_output_objective(
            &model.config,
            &snapshot.positions,
            &snapshot.states,
            &updates,
            snapshot.step_fraction,
            direct_growth_phase_gain(cfg),
            cfg.liveness_front_radius,
            &mut phase_output_gradients,
        );
        if let Some(phase_output) = phase_output {
            accumulate_output_channels(
                &mut phase_progress,
                &phase_output_gradients,
                particle_count,
                output_dims,
                [phase_output],
            );
        }

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
        if let Some(material_output) = material_output {
            accumulate_output_channels(
                &mut material_coverage_materialization,
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
            &updates,
            snapshot.step_fraction,
            cfg.opacity_gain * DIRECT_GROWTH_TEMPORAL_MATERIALIZATION_GAIN_FRACTION,
            cfg.liveness_front_radius,
            &liveness_candidate_weights,
            cfg.seed_scale,
            cfg.material_max_opacity_update,
            &mut temporal_materialization_output_gradients,
        );
        if let Some(material_output) = material_output {
            accumulate_output_channels(
                &mut temporal_materialization,
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
            &updates,
            cfg.opacity_gain * DIRECT_GROWTH_ACTIVE_SURFACE_MATERIALIZATION_GAIN_FRACTION,
            cfg.seed_scale,
            cfg.material_max_opacity_update,
            cfg.liveness_front_radius,
            Some(&liveness_candidate_weights),
            &mut active_surface_materialization_output_gradients,
        );
        if let Some(material_output) = material_output {
            accumulate_output_channels(
                &mut active_surface_materialization,
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
            &updates,
            cfg.opacity_gain * DIRECT_GROWTH_STRICT_SURFACE_MATERIALIZATION_GAIN_FRACTION,
            cfg.seed_scale,
            cfg.material_max_opacity_update,
            &mut strict_surface_materialization_output_gradients,
        );
        if let Some(material_output) = material_output {
            accumulate_output_channels(
                &mut strict_surface_materialization,
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
        accumulate_output_channels(
            &mut material_surface_motion,
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
        if let Some(material_output) = material_output {
            accumulate_output_channels(
                &mut material_visibility,
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
            &updates,
            cfg.opacity_gain * DIRECT_GROWTH_SURFACE_COLOR_GAIN_FRACTION,
            cfg.seed_scale,
            cfg.material_max_opacity_update,
            cfg.liveness_front_radius,
            Some(&liveness_candidate_weights),
            cfg.direct_output_gradient_rms_cap * 0.5,
            &mut surface_color_output_gradients,
        ) {
            accumulate_output_channels(
                &mut surface_color,
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
            &updates,
            cfg.render,
            cfg.scale_budget_weight,
            cfg.max_opacity_update,
            &mut scale_budget_output_gradients,
        ) {
            accumulate_output_channels(
                &mut scale_budget,
                &scale_budget_output_gradients,
                particle_count,
                output_dims,
                [scale_output],
            );
        }

        let mut combined = temporal_output_gradients;
        add_output_gradients(&mut combined, &mesh_liveness_output_gradients);
        add_output_gradients(&mut combined, &surface_escape_liveness_output_gradients);
        add_output_gradients(&mut combined, &target_coverage_liveness_output_gradients);
        add_output_gradients(&mut combined, &material_coverage_liveness_output_gradients);
        add_output_gradients(&mut combined, &extent_front_output_gradients);
        add_output_gradients(&mut combined, &phase_output_gradients);
        add_output_gradients(&mut combined, &liveness_phase_memory_output_gradients);
        add_output_gradients(&mut combined, &mesh_output_gradients);
        add_output_gradients(&mut combined, &extent_front_motion_output_gradients);
        add_output_gradients(&mut combined, &temporal_extent_motion_output_gradients);
        add_output_gradients(&mut combined, &extent_motion_memory_output_gradients);
        add_output_gradients(&mut combined, &material_coverage_motion_output_gradients);
        add_output_gradients(&mut combined, &material_surface_motion_output_gradients);
        add_output_gradients(&mut combined, &residual_velocity_output_gradients);
        add_output_gradients(&mut combined, &motion_memory_output_gradients);
        add_output_gradients(
            &mut combined,
            &material_coverage_motion_memory_output_gradients,
        );
        add_output_gradients(
            &mut combined,
            &material_coverage_materialization_output_gradients,
        );
        add_output_gradients(&mut combined, &temporal_materialization_output_gradients);
        add_output_gradients(
            &mut combined,
            &active_surface_materialization_output_gradients,
        );
        add_output_gradients(
            &mut combined,
            &strict_surface_materialization_output_gradients,
        );
        add_output_gradients(&mut combined, &material_output_gradients);
        add_output_gradients(&mut combined, &surface_color_output_gradients);
        add_output_gradients(&mut combined, &scale_budget_output_gradients);
        boost_sparse_output_channel_rms(
            &mut combined,
            output_dims,
            0..model.config.spatial_dims,
            cfg.direct_output_gradient_rms_cap * DIRECT_GROWTH_SPATIAL_MOTION_RMS_TARGET_FRACTION,
            8.0,
        );
        accumulate_output_channels(
            &mut combined_pre_cap,
            &combined,
            particle_count,
            output_dims,
            0..output_dims,
        );
        cap_output_gradient_channel_rms_with_state_caps(
            &model.config,
            &mut combined,
            output_dims,
            cfg.direct_output_gradient_rms_cap,
            liveness_update_cap,
            cfg.direct_output_gradient_rms_cap * DIRECT_GROWTH_MATERIAL_OUTPUT_RMS_CAP_MULTIPLIER,
        );
        accumulate_output_channels(
            &mut combined_post_cap,
            &combined,
            particle_count,
            output_dims,
            0..output_dims,
        );
        accumulate_output_channels(
            &mut mesh_motion_post_cap,
            &combined,
            particle_count,
            output_dims,
            0..model.config.spatial_dims,
        );
        let velocity_outputs = growth_3d_velocity_output_channels(&model.config);
        if !velocity_outputs.is_empty() {
            accumulate_output_channels(
                &mut residual_velocity_post_cap,
                &combined,
                particle_count,
                output_dims,
                velocity_outputs.clone(),
            );
            accumulate_output_channels(
                &mut motion_memory_post_cap,
                &combined,
                particle_count,
                output_dims,
                velocity_outputs,
            );
        }
        if liveness_output < output_dims {
            accumulate_output_channels(
                &mut liveness_post_cap,
                &combined,
                particle_count,
                output_dims,
                [liveness_output],
            );
        }
        if let Some(phase_output) = phase_output {
            accumulate_output_channels(
                &mut phase_post_cap,
                &combined,
                particle_count,
                output_dims,
                [phase_output],
            );
        }
        if let Some(material_output) = material_output {
            accumulate_output_channels(
                &mut material_post_cap,
                &combined,
                particle_count,
                output_dims,
                [material_output],
            );
        }
        if let Some(scale_output) = scale_output {
            accumulate_output_channels(
                &mut scale_post_cap,
                &combined,
                particle_count,
                output_dims,
                [scale_output],
            );
        }
        if let Some(color_outputs) = color_outputs {
            accumulate_output_channels(
                &mut color_post_cap,
                &combined,
                particle_count,
                output_dims,
                color_outputs,
            );
        }
    }
    let terminal_liveness_state =
        terminal_liveness_state_diagnostics(model, trajectory, cfg, liveness_update_cap);

    Ok(DirectRolloutObjectiveDiagnostics {
        snapshots: trajectory.len(),
        rows,
        temporal_liveness_rms: temporal_liveness.rms(),
        temporal_liveness_nonzero_fraction: temporal_liveness.nonzero_fraction(),
        terminal_liveness_state_rms: terminal_liveness_state.rms(),
        terminal_liveness_state_nonzero_fraction: terminal_liveness_state.nonzero_fraction(),
        mesh_motion_liveness_rms: mesh_motion_liveness.rms(),
        mesh_motion_liveness_nonzero_fraction: mesh_motion_liveness.nonzero_fraction(),
        surface_escape_liveness_rms: surface_escape_liveness.rms(),
        surface_escape_liveness_nonzero_fraction: surface_escape_liveness.nonzero_fraction(),
        target_coverage_liveness_rms: target_coverage_liveness.rms(),
        target_coverage_liveness_nonzero_fraction: target_coverage_liveness.nonzero_fraction(),
        material_coverage_liveness_rms: material_coverage_liveness.rms(),
        material_coverage_liveness_nonzero_fraction: material_coverage_liveness.nonzero_fraction(),
        extent_front_liveness_rms: extent_front_liveness.rms(),
        extent_front_liveness_nonzero_fraction: extent_front_liveness.nonzero_fraction(),
        phase_rms: phase_progress.rms(),
        phase_nonzero_fraction: phase_progress.nonzero_fraction(),
        liveness_phase_memory_rms: liveness_phase_memory.rms(),
        liveness_phase_memory_nonzero_fraction: liveness_phase_memory.nonzero_fraction(),
        mesh_motion_rms: mesh_motion.rms(),
        mesh_motion_nonzero_fraction: mesh_motion.nonzero_fraction(),
        extent_front_motion_rms: extent_front_motion.rms(),
        extent_front_motion_nonzero_fraction: extent_front_motion.nonzero_fraction(),
        temporal_extent_motion_rms: temporal_extent_motion.rms(),
        temporal_extent_motion_nonzero_fraction: temporal_extent_motion.nonzero_fraction(),
        extent_motion_memory_rms: extent_motion_memory.rms(),
        extent_motion_memory_nonzero_fraction: extent_motion_memory.nonzero_fraction(),
        material_coverage_motion_rms: material_coverage_motion.rms(),
        material_coverage_motion_nonzero_fraction: material_coverage_motion.nonzero_fraction(),
        material_surface_motion_rms: material_surface_motion.rms(),
        material_surface_motion_nonzero_fraction: material_surface_motion.nonzero_fraction(),
        residual_velocity_rms: residual_velocity.rms(),
        residual_velocity_nonzero_fraction: residual_velocity.nonzero_fraction(),
        motion_memory_rms: motion_memory.rms(),
        motion_memory_nonzero_fraction: motion_memory.nonzero_fraction(),
        material_coverage_motion_memory_rms: material_coverage_motion_memory.rms(),
        material_coverage_motion_memory_nonzero_fraction: material_coverage_motion_memory
            .nonzero_fraction(),
        material_coverage_materialization_rms: material_coverage_materialization.rms(),
        material_coverage_materialization_nonzero_fraction: material_coverage_materialization
            .nonzero_fraction(),
        temporal_materialization_rms: temporal_materialization.rms(),
        temporal_materialization_nonzero_fraction: temporal_materialization.nonzero_fraction(),
        active_surface_materialization_rms: active_surface_materialization.rms(),
        active_surface_materialization_nonzero_fraction: active_surface_materialization
            .nonzero_fraction(),
        strict_surface_materialization_rms: strict_surface_materialization.rms(),
        strict_surface_materialization_nonzero_fraction: strict_surface_materialization
            .nonzero_fraction(),
        material_visibility_rms: material_visibility.rms(),
        material_visibility_nonzero_fraction: material_visibility.nonzero_fraction(),
        surface_color_rms: surface_color.rms(),
        surface_color_nonzero_fraction: surface_color.nonzero_fraction(),
        scale_budget_rms: scale_budget.rms(),
        scale_budget_nonzero_fraction: scale_budget.nonzero_fraction(),
        combined_pre_cap_rms: combined_pre_cap.rms(),
        combined_post_cap_rms: combined_post_cap.rms(),
        mesh_motion_post_cap_rms: mesh_motion_post_cap.rms(),
        mesh_motion_post_cap_nonzero_fraction: mesh_motion_post_cap.nonzero_fraction(),
        residual_velocity_post_cap_rms: residual_velocity_post_cap.rms(),
        residual_velocity_post_cap_nonzero_fraction: residual_velocity_post_cap.nonzero_fraction(),
        motion_memory_post_cap_rms: motion_memory_post_cap.rms(),
        motion_memory_post_cap_nonzero_fraction: motion_memory_post_cap.nonzero_fraction(),
        liveness_post_cap_rms: liveness_post_cap.rms(),
        phase_post_cap_rms: phase_post_cap.rms(),
        material_post_cap_rms: material_post_cap.rms(),
        scale_post_cap_rms: scale_post_cap.rms(),
        color_post_cap_rms: color_post_cap.rms(),
    })
}
