use super::*;

pub(crate) fn render_training_objective_config(
    cfg: &RenderProxyTrainingConfig,
    render: RenderLossConfig,
) -> MeshRolloutObjectiveConfig {
    MeshRolloutObjectiveConfig {
        gaussian_decode_mode: render.gaussian_decode_mode,
        render_sigma: render.sigma,
        render_min_sigma: render.min_sigma,
        render_max_sigma: render.max_sigma,
        render_density_weight: render.density_weight,
        render_color_weight: render.color_weight,
        render_depth_weight: render.depth_weight,
        surface_gain: cfg.surface_gain,
        surface_escape_gain: cfg.surface_escape_gain,
        coverage_gain: cfg.coverage_gain,
        coverage_samples: cfg.coverage_samples,
        coverage_repulsion_gain: cfg.coverage_repulsion_gain,
        coverage_normal_weight: cfg.coverage_normal_weight,
        extent_gain: cfg.extent_gain,
        trajectory_render_gain: cfg.trajectory_render_gain,
        trajectory_mesh_gain: cfg.trajectory_mesh_gain,
        trajectory_render_samples: cfg.trajectory_render_samples,
        liveness_gain: cfg.liveness_gain,
        phase_gain: direct_growth_phase_gain(cfg),
        liveness_front_radius: cfg.liveness_front_radius,
        liveness_update_multiplier: cfg.liveness_update_multiplier,
        opacity_gain: cfg.opacity_gain,
        material_liveness_gain: cfg.material_liveness_gain,
        material_tail_gain: cfg.material_tail_gain,
        material_suppression_update_multiplier: cfg.material_suppression_update_multiplier,
        material_max_opacity_update: cfg.material_max_opacity_update,
        gaussian_scale_gain: cfg.scale_gain,
        gaussian_scale_budget_weight: cfg.scale_budget_weight,
    }
}

pub(crate) fn material_suppression_max_update(
    max_opacity_update: f32,
    material_suppression_update_multiplier: f32,
) -> f32 {
    if max_opacity_update.is_finite()
        && max_opacity_update > 0.0
        && material_suppression_update_multiplier.is_finite()
        && material_suppression_update_multiplier > 0.0
    {
        max_opacity_update * material_suppression_update_multiplier
    } else {
        max_opacity_update
    }
}

pub(crate) fn liveness_max_update(max_opacity_update: f32, liveness_update_multiplier: f32) -> f32 {
    if max_opacity_update.is_finite()
        && max_opacity_update > 0.0
        && liveness_update_multiplier.is_finite()
        && liveness_update_multiplier > 0.0
    {
        max_opacity_update * liveness_update_multiplier
    } else {
        max_opacity_update
    }
}

#[derive(Default)]
struct OutputGradientAccumulator {
    sum_sq: f32,
    samples: usize,
    nonzero: usize,
}

impl OutputGradientAccumulator {
    fn add_value(&mut self, value: f32) {
        if !value.is_finite() {
            return;
        }
        self.sum_sq += value * value;
        self.samples += 1;
        if value.abs() > 1.0e-8 {
            self.nonzero += 1;
        }
    }

    fn rms(&self) -> f32 {
        if self.samples == 0 {
            0.0
        } else {
            (self.sum_sq / self.samples as f32).sqrt()
        }
    }

    fn nonzero_fraction(&self) -> f32 {
        if self.samples == 0 {
            0.0
        } else {
            self.nonzero as f32 / self.samples as f32
        }
    }
}

fn accumulate_output_channels<I>(
    accumulator: &mut OutputGradientAccumulator,
    gradients: &[f32],
    rows: usize,
    output_dims: usize,
    channels: I,
) where
    I: IntoIterator<Item = usize> + Clone,
{
    if output_dims == 0 || gradients.len() < rows.saturating_mul(output_dims) {
        return;
    }
    for row in 0..rows {
        let base = row * output_dims;
        for channel in channels.clone() {
            if channel < output_dims {
                accumulator.add_value(gradients[base + channel]);
            }
        }
    }
}

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
    let mut residual_velocity = OutputGradientAccumulator::default();
    let mut motion_memory = OutputGradientAccumulator::default();
    let mut material_coverage_motion_memory = OutputGradientAccumulator::default();
    let mut material_coverage_materialization = OutputGradientAccumulator::default();
    let mut temporal_materialization = OutputGradientAccumulator::default();
    let mut active_surface_materialization = OutputGradientAccumulator::default();
    let mut material_visibility = OutputGradientAccumulator::default();
    let mut combined_pre_cap = OutputGradientAccumulator::default();
    let mut combined_post_cap = OutputGradientAccumulator::default();
    let mut mesh_motion_post_cap = OutputGradientAccumulator::default();
    let mut residual_velocity_post_cap = OutputGradientAccumulator::default();
    let mut motion_memory_post_cap = OutputGradientAccumulator::default();
    let mut liveness_post_cap = OutputGradientAccumulator::default();
    let mut phase_post_cap = OutputGradientAccumulator::default();
    let mut material_post_cap = OutputGradientAccumulator::default();
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
            &snapshot.states,
            &updates,
            snapshot.step_fraction,
            cfg.opacity_gain * DIRECT_GROWTH_MATERIAL_COVERAGE_MATERIALIZATION_GAIN_FRACTION,
            &material_coverage_candidate_weights,
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
            &snapshot.positions,
            &snapshot.states,
            &updates,
            snapshot.step_fraction,
            cfg.opacity_gain * DIRECT_GROWTH_TEMPORAL_MATERIALIZATION_GAIN_FRACTION,
            cfg.liveness_front_radius,
            &liveness_candidate_weights,
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
        add_output_gradients(&mut combined, &material_output_gradients);
        accumulate_output_channels(
            &mut combined_pre_cap,
            &combined,
            particle_count,
            output_dims,
            0..output_dims,
        );
        cap_output_gradient_channel_rms_with_liveness_cap(
            &model.config,
            &mut combined,
            output_dims,
            cfg.direct_output_gradient_rms_cap,
            liveness_update_cap,
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
        material_visibility_rms: material_visibility.rms(),
        material_visibility_nonzero_fraction: material_visibility.nonzero_fraction(),
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
    })
}

fn terminal_liveness_state_diagnostics(
    model: &NpaModel,
    trajectory: &[RenderTrajectorySnapshot],
    cfg: &RenderProxyTrainingConfig,
    max_adjoint: f32,
) -> OutputGradientAccumulator {
    let mut accumulator = OutputGradientAccumulator::default();
    let terminal_gain = direct_terminal_liveness_gain(cfg);
    if terminal_gain <= 0.0 || !terminal_gain.is_finite() {
        return accumulator;
    }
    let Some(snapshot) = trajectory.last() else {
        return accumulator;
    };
    if snapshot.positions.is_empty()
        || snapshot.states.len()
            < snapshot
                .positions
                .len()
                .saturating_mul(model.config.state_dims)
        || model.config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
    {
        return accumulator;
    }
    let mut state_adjoint = vec![0.0_f32; snapshot.states.len()];
    add_liveness_front_state_adjoint(
        &model.config,
        &snapshot.positions,
        &snapshot.states,
        terminal_gain,
        cfg.liveness_front_radius,
        1.0,
        max_adjoint,
        &mut state_adjoint,
    );
    add_temporal_activation_schedule_state_adjoint(
        &model.config,
        &snapshot.positions,
        &snapshot.states,
        terminal_gain,
        cfg.liveness_front_radius,
        1.0,
        max_adjoint,
        &mut state_adjoint,
    );
    for row in 0..snapshot.positions.len() {
        accumulator
            .add_value(state_adjoint[row * model.config.state_dims + GROWTH_3D_LIVENESS_CHANNEL]);
    }
    accumulator
}

pub(crate) const LOCAL_FRONT_LIVENESS_SCORE_WEIGHT: f32 = 0.01;
pub(crate) const MATERIAL_VISIBLE_TARGET_MEAN_DISTANCE_SCORE_WEIGHT: f32 = 0.05;
pub(crate) const MATERIAL_VISIBLE_TARGET_MAX_DISTANCE_SCORE_WEIGHT: f32 = 0.02;
pub(crate) const MATERIAL_VISIBLE_TARGET_DISTANCE_REGRESSION_SLACK: f32 = 0.02;
pub(crate) const TEMPORAL_ACTIVATION_SCORE_WEIGHT: f32 = 25.0;
pub(crate) const TEMPORAL_ACTIVATION_SELECTION_REGRESSION_SLACK: f32 = 0.02;
pub(crate) const TEMPORAL_LIVENESS_TRAJECTORY_SAMPLE_CAP: usize = 8;
pub(crate) const DIRECT_TRAJECTORY_TERMINAL_LIVENESS_WEIGHT: f32 = 0.25;
pub(crate) const TEMPORAL_ACTIVATION_JUMP_SLACK: f32 = 0.10;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LocalFrontLivenessProgress {
    pub(crate) candidate_count: usize,
    pub(crate) weighted_activation_margin: f32,
}

pub(crate) fn liveness_progress_from_candidate_weights(
    config: &NpaConfig,
    states: &[f32],
    candidate_weights: &[f32],
) -> LocalFrontLivenessProgress {
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL || candidate_weights.is_empty() {
        return LocalFrontLivenessProgress::default();
    }
    let rows = candidate_weights
        .len()
        .min(states.len() / config.state_dims.max(1));
    let mut candidate_count = 0usize;
    let mut weighted_margin = 0.0_f32;
    let mut weight_sum = 0.0_f32;
    for (row, candidate_weight) in candidate_weights.iter().take(rows).copied().enumerate() {
        let liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
        if liveness > -1.0 || candidate_weight <= 0.0 || !candidate_weight.is_finite() {
            continue;
        }
        candidate_count += 1;
        weight_sum += candidate_weight;
        weighted_margin += candidate_weight * (-1.0 - liveness).max(0.0);
    }

    LocalFrontLivenessProgress {
        candidate_count,
        weighted_activation_margin: if weight_sum > 0.0 {
            weighted_margin / weight_sum
        } else {
            0.0
        },
    }
}

pub(crate) fn local_front_liveness_progress(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    front_radius: f32,
) -> LocalFrontLivenessProgress {
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || positions.is_empty()
        || states.len() < positions.len() * config.state_dims
        || front_radius <= 0.0
        || !front_radius.is_finite()
    {
        return LocalFrontLivenessProgress::default();
    }

    let front_weights = local_front_weights(config, positions, states, front_radius);
    liveness_progress_from_candidate_weights(config, states, &front_weights)
}

pub(crate) fn extent_front_liveness_progress(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    front_radius: f32,
) -> LocalFrontLivenessProgress {
    let weights =
        extent_front_liveness_candidate_weights(config, target, positions, states, front_radius);
    liveness_progress_from_candidate_weights(config, states, &weights)
}

pub(crate) fn direct_terminal_liveness_gain(cfg: &RenderProxyTrainingConfig) -> f32 {
    if cfg.trajectory_supervision
        && cfg.liveness_gain > 0.0
        && cfg.liveness_gain.is_finite()
        && DIRECT_TRAJECTORY_TERMINAL_LIVENESS_WEIGHT.is_finite()
    {
        cfg.liveness_gain * DIRECT_TRAJECTORY_TERMINAL_LIVENESS_WEIGHT
    } else {
        cfg.liveness_gain
    }
}

pub(crate) fn temporal_front_liveness_progress(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    cfg: &RenderProxyTrainingConfig,
    seed: u64,
    seed_positions: &[[f32; 4]],
    seed_states: &[f32],
) -> Result<LocalFrontLivenessProgress, Box<dyn std::error::Error>> {
    let rollout_steps = cfg.rollout_steps.max(1);
    let mut candidate_count = 0usize;
    let mut worst_margin = 0.0_f32;
    for steps in growth_3d_temporal_sample_steps(rollout_steps) {
        if steps == 0 || steps >= rollout_steps {
            continue;
        }
        let (positions, states) = if steps == 0 {
            (seed_positions.to_vec(), seed_states.to_vec())
        } else {
            let trace = run_rollout(
                model,
                grid,
                &RolloutConfig {
                    particle_count: cfg.particles,
                    steps,
                    update_prob: 1.0,
                    seed,
                    seed_scale: cfg.seed_scale,
                    ..RolloutConfig::default()
                },
                cfg.seed_mode,
            )?;
            (trace.positions, trace.states)
        };
        let rows = positions.len();
        let active_count = active_liveness_count(&states, rows, model.config.state_dims);
        let schedule = (steps as f32 / rollout_steps as f32).clamp(0.0, 1.0);
        let target_active =
            ((rows as f32) * temporal_activation_target_fraction(schedule)).ceil() as usize;
        if active_count >= target_active {
            continue;
        }
        let progress = local_front_liveness_progress(
            &model.config,
            &positions,
            &states,
            cfg.liveness_front_radius,
        );
        if progress.candidate_count == 0 {
            continue;
        }
        candidate_count += progress.candidate_count;
        worst_margin = worst_margin.max(progress.weighted_activation_margin);
    }
    Ok(LocalFrontLivenessProgress {
        candidate_count,
        weighted_activation_margin: worst_margin,
    })
}

pub(crate) fn temporal_extent_front_liveness_progress(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: &RenderProxyTrainingConfig,
    seed: u64,
    seed_positions: &[[f32; 4]],
    seed_states: &[f32],
) -> Result<LocalFrontLivenessProgress, Box<dyn std::error::Error>> {
    let rollout_steps = cfg.rollout_steps.max(1);
    let mut candidate_count = 0usize;
    let mut worst_margin = 0.0_f32;
    for steps in growth_3d_temporal_sample_steps(rollout_steps) {
        if steps == 0 || steps >= rollout_steps {
            continue;
        }
        let (positions, states) = if steps == 0 {
            (seed_positions.to_vec(), seed_states.to_vec())
        } else {
            let trace = run_rollout(
                model,
                grid,
                &RolloutConfig {
                    particle_count: cfg.particles,
                    steps,
                    update_prob: 1.0,
                    seed,
                    seed_scale: cfg.seed_scale,
                    ..RolloutConfig::default()
                },
                cfg.seed_mode,
            )?;
            (trace.positions, trace.states)
        };
        let progress = extent_front_liveness_progress(
            &model.config,
            target,
            &positions,
            &states,
            cfg.liveness_front_radius,
        );
        if progress.candidate_count == 0 {
            continue;
        }
        candidate_count += progress.candidate_count;
        worst_margin = worst_margin.max(progress.weighted_activation_margin);
    }
    Ok(LocalFrontLivenessProgress {
        candidate_count,
        weighted_activation_margin: worst_margin,
    })
}

pub(crate) fn gaussian_volume_stats_for_trace(
    trace: &crate::RolloutTrace,
    render: RenderLossConfig,
) -> GaussianVolumeStats {
    GaussianVolumeStats::from_render_config(trace, render)
}
