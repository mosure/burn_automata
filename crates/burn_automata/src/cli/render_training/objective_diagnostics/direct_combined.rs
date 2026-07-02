use super::accumulator::accumulate_output_channels;
use super::direct_accumulators::DirectObjectiveAccumulators;
use super::*;

pub(super) struct DirectCombinedGradientInputs<'a> {
    pub(super) config: &'a NpaConfig,
    pub(super) cfg: &'a RenderProxyTrainingConfig,
    pub(super) output_dims: usize,
    pub(super) particle_count: usize,
    pub(super) liveness_update_cap: f32,
    pub(super) liveness_output: usize,
    pub(super) phase_output: Option<usize>,
    pub(super) material_output: Option<usize>,
    pub(super) scale_output: Option<usize>,
    pub(super) color_outputs: Option<[usize; 3]>,
}

pub(super) fn accumulate_combined_direct_gradients(
    accumulators: &mut DirectObjectiveAccumulators,
    inputs: DirectCombinedGradientInputs<'_>,
    mut combined: Vec<f32>,
    additional_gradients: &[&[f32]],
) {
    for gradients in additional_gradients {
        add_output_gradients(&mut combined, gradients);
    }
    boost_sparse_output_channel_rms(
        &mut combined,
        inputs.output_dims,
        0..inputs.config.spatial_dims,
        inputs.cfg.direct_output_gradient_rms_cap
            * DIRECT_GROWTH_SPATIAL_MOTION_RMS_TARGET_FRACTION,
        8.0,
    );
    accumulate_output_channels(
        &mut accumulators.combined_pre_cap,
        &combined,
        inputs.particle_count,
        inputs.output_dims,
        0..inputs.output_dims,
    );
    cap_output_gradient_channel_rms_with_state_caps(
        inputs.config,
        &mut combined,
        inputs.output_dims,
        inputs.cfg.direct_output_gradient_rms_cap,
        inputs.liveness_update_cap,
        inputs.cfg.direct_output_gradient_rms_cap
            * DIRECT_GROWTH_MATERIAL_OUTPUT_RMS_CAP_MULTIPLIER,
    );
    accumulate_output_channels(
        &mut accumulators.combined_post_cap,
        &combined,
        inputs.particle_count,
        inputs.output_dims,
        0..inputs.output_dims,
    );
    accumulate_output_channels(
        &mut accumulators.mesh_motion_post_cap,
        &combined,
        inputs.particle_count,
        inputs.output_dims,
        0..inputs.config.spatial_dims,
    );
    let velocity_outputs = growth_3d_velocity_output_channels(inputs.config);
    if !velocity_outputs.is_empty() {
        accumulate_output_channels(
            &mut accumulators.residual_velocity_post_cap,
            &combined,
            inputs.particle_count,
            inputs.output_dims,
            velocity_outputs.clone(),
        );
        accumulate_output_channels(
            &mut accumulators.motion_memory_post_cap,
            &combined,
            inputs.particle_count,
            inputs.output_dims,
            velocity_outputs,
        );
    }
    if inputs.liveness_output < inputs.output_dims {
        accumulate_output_channels(
            &mut accumulators.liveness_post_cap,
            &combined,
            inputs.particle_count,
            inputs.output_dims,
            [inputs.liveness_output],
        );
    }
    if let Some(phase_output) = inputs.phase_output {
        accumulate_output_channels(
            &mut accumulators.phase_post_cap,
            &combined,
            inputs.particle_count,
            inputs.output_dims,
            [phase_output],
        );
    }
    if let Some(material_output) = inputs.material_output {
        accumulate_output_channels(
            &mut accumulators.material_post_cap,
            &combined,
            inputs.particle_count,
            inputs.output_dims,
            [material_output],
        );
    }
    if let Some(scale_output) = inputs.scale_output {
        accumulate_output_channels(
            &mut accumulators.scale_post_cap,
            &combined,
            inputs.particle_count,
            inputs.output_dims,
            [scale_output],
        );
    }
    if let Some(color_outputs) = inputs.color_outputs {
        accumulate_output_channels(
            &mut accumulators.color_post_cap,
            &combined,
            inputs.particle_count,
            inputs.output_dims,
            color_outputs,
        );
    }
}
