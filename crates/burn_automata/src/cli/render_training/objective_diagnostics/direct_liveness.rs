use super::accumulator::accumulate_output_channels;
use super::direct_accumulators::DirectObjectiveAccumulators;
use super::direct_channels::DirectObjectiveChannels;
use super::direct_motion::DirectMotionObjectiveOutputs;
use super::*;

pub(super) struct DirectLivenessObjectiveOutputs {
    pub(super) mesh_liveness_output_gradients: Vec<f32>,
    pub(super) target_coverage_liveness_output_gradients: Vec<f32>,
    pub(super) material_coverage_liveness_output_gradients: Vec<f32>,
    pub(super) material_visible_liveness_output_gradients: Vec<f32>,
    pub(super) surface_escape_liveness_output_gradients: Vec<f32>,
    pub(super) extent_front_output_gradients: Vec<f32>,
    pub(super) liveness_candidate_weights: Vec<f32>,
    pub(super) temporal_output_gradients: Vec<f32>,
    pub(super) phase_output_gradients: Vec<f32>,
    pub(super) liveness_phase_memory_output_gradients: Vec<f32>,
}

pub(super) fn collect_direct_liveness_objectives(
    model: &NpaModel,
    target: &TriangleMeshTarget,
    snapshot: &RenderTrajectorySnapshot,
    cfg: &RenderProxyTrainingConfig,
    channels: &DirectObjectiveChannels,
    motion: &DirectMotionObjectiveOutputs,
    accumulators: &mut DirectObjectiveAccumulators,
) -> DirectLivenessObjectiveOutputs {
    let particle_count = snapshot.positions.len();
    let output_dims = channels.output_dims;

    let mut mesh_liveness_output_gradients = vec![0.0; particle_count * output_dims];
    add_mesh_motion_liveness_output_objective(
        &model.config,
        &snapshot.positions,
        &snapshot.states,
        &motion.updates,
        &motion.mesh_motion_candidate_weights,
        cfg.liveness_gain * DIRECT_GROWTH_MESH_MOTION_LIVENESS_GAIN_FRACTION,
        channels.liveness_update_cap,
        &mut mesh_liveness_output_gradients,
    );
    if channels.liveness_output < output_dims {
        accumulate_output_channels(
            &mut accumulators.mesh_motion_liveness,
            &mesh_liveness_output_gradients,
            particle_count,
            output_dims,
            [channels.liveness_output],
        );
    }

    let mut target_coverage_liveness_output_gradients = vec![0.0; particle_count * output_dims];
    add_candidate_liveness_output_objective(
        &model.config,
        &snapshot.positions,
        &snapshot.states,
        &motion.updates,
        &motion.target_coverage_candidate_weights,
        cfg.liveness_gain * DIRECT_GROWTH_TARGET_COVERAGE_LIVENESS_GAIN_FRACTION,
        channels.liveness_update_cap,
        &mut target_coverage_liveness_output_gradients,
    );
    if channels.liveness_output < output_dims {
        accumulate_output_channels(
            &mut accumulators.target_coverage_liveness,
            &target_coverage_liveness_output_gradients,
            particle_count,
            output_dims,
            [channels.liveness_output],
        );
    }

    let mut material_coverage_liveness_output_gradients = vec![0.0; particle_count * output_dims];
    add_candidate_liveness_output_objective(
        &model.config,
        &snapshot.positions,
        &snapshot.states,
        &motion.updates,
        &motion.material_coverage_candidate_weights,
        cfg.liveness_gain * DIRECT_GROWTH_MATERIAL_COVERAGE_LIVENESS_GAIN_FRACTION,
        channels.liveness_update_cap,
        &mut material_coverage_liveness_output_gradients,
    );
    if channels.liveness_output < output_dims {
        accumulate_output_channels(
            &mut accumulators.material_coverage_liveness,
            &material_coverage_liveness_output_gradients,
            particle_count,
            output_dims,
            [channels.liveness_output],
        );
    }

    let mut material_visible_liveness_output_gradients = vec![0.0; particle_count * output_dims];
    add_material_visible_liveness_output_objective(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        &motion.updates,
        cfg.material_liveness_gain,
        target_coverage_threshold(cfg.seed_scale),
        channels.liveness_update_cap,
        cfg.liveness_front_radius,
        1.0,
        &mut material_visible_liveness_output_gradients,
    );
    if channels.liveness_output < output_dims {
        accumulate_output_channels(
            &mut accumulators.material_visible_liveness,
            &material_visible_liveness_output_gradients,
            particle_count,
            output_dims,
            [channels.liveness_output],
        );
    }

    let mut surface_escape_liveness_output_gradients = vec![0.0; particle_count * output_dims];
    add_surface_escape_liveness_output_objective(
        &model.config,
        target,
        &snapshot.positions,
        &snapshot.states,
        &motion.updates,
        cfg.surface_escape_gain,
        cfg.liveness_gain,
        channels.liveness_update_cap,
        1.0,
        &mut surface_escape_liveness_output_gradients,
    );
    if channels.liveness_output < output_dims {
        accumulate_output_channels(
            &mut accumulators.surface_escape_liveness,
            &surface_escape_liveness_output_gradients,
            particle_count,
            output_dims,
            [channels.liveness_output],
        );
    }

    let mut extent_front_output_gradients = vec![0.0; particle_count * output_dims];
    add_candidate_liveness_output_objective(
        &model.config,
        &snapshot.positions,
        &snapshot.states,
        &motion.updates,
        &motion.extent_front_candidate_weights,
        cfg.liveness_gain * DIRECT_GROWTH_EXTENT_FRONT_LIVENESS_GAIN_FRACTION,
        channels.liveness_update_cap,
        &mut extent_front_output_gradients,
    );
    if channels.liveness_output < output_dims {
        accumulate_output_channels(
            &mut accumulators.extent_front_liveness,
            &extent_front_output_gradients,
            particle_count,
            output_dims,
            [channels.liveness_output],
        );
    }

    let liveness_candidate_weights = mesh_motion_candidate_weights_with_local_front_floor(
        &model.config,
        &snapshot.positions,
        &snapshot.states,
        cfg.liveness_front_radius,
        DIRECT_GROWTH_LOCAL_FRONT_LIVENESS_FLOOR,
        &motion.mesh_motion_candidate_weights,
    );
    let liveness_candidate_weights = max_candidate_weights(
        &liveness_candidate_weights,
        &motion.extent_front_candidate_weights,
    );
    let liveness_candidate_weights = max_candidate_weights(
        &liveness_candidate_weights,
        &motion.target_coverage_candidate_weights,
    );
    let liveness_candidate_weights = max_candidate_weights(
        &liveness_candidate_weights,
        &motion.material_coverage_candidate_weights,
    );
    let material_visible_liveness_candidate_weights = liveness_candidate_weights_from_gradients(
        output_dims,
        channels.liveness_output,
        &material_visible_liveness_output_gradients,
    );
    let liveness_candidate_weights = max_candidate_weights(
        &liveness_candidate_weights,
        &material_visible_liveness_candidate_weights,
    );
    let liveness_candidate_weights = temporal_liveness_candidate_weights_with_local_front_floor(
        &model.config,
        &snapshot.positions,
        &snapshot.states,
        &motion.updates,
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
        &motion.updates,
        snapshot.step_fraction,
        cfg.liveness_gain,
        cfg.liveness_front_radius,
        Some(&liveness_candidate_weights),
        &mut temporal_output_gradients,
    );
    if channels.liveness_output < output_dims {
        accumulate_output_channels(
            &mut accumulators.temporal_liveness,
            &temporal_output_gradients,
            particle_count,
            output_dims,
            [channels.liveness_output],
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
        &material_visible_liveness_output_gradients,
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
    if let Some(phase_output) = channels.phase_output {
        accumulate_output_channels(
            &mut accumulators.liveness_phase_memory,
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
        &motion.updates,
        snapshot.step_fraction,
        direct_growth_phase_gain(cfg),
        cfg.liveness_front_radius,
        &mut phase_output_gradients,
    );
    if let Some(phase_output) = channels.phase_output {
        accumulate_output_channels(
            &mut accumulators.phase_progress,
            &phase_output_gradients,
            particle_count,
            output_dims,
            [phase_output],
        );
    }

    DirectLivenessObjectiveOutputs {
        mesh_liveness_output_gradients,
        target_coverage_liveness_output_gradients,
        material_coverage_liveness_output_gradients,
        material_visible_liveness_output_gradients,
        surface_escape_liveness_output_gradients,
        extent_front_output_gradients,
        liveness_candidate_weights,
        temporal_output_gradients,
        phase_output_gradients,
        liveness_phase_memory_output_gradients,
    }
}
