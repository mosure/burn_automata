use super::*;

pub(super) struct DirectObjectiveChannels {
    pub(super) output_dims: usize,
    pub(super) liveness_output: usize,
    pub(super) material_output: Option<usize>,
    pub(super) scale_output: Option<usize>,
    pub(super) color_outputs: Option<[usize; 3]>,
    pub(super) phase_output: Option<usize>,
    pub(super) liveness_update_cap: f32,
}

impl DirectObjectiveChannels {
    pub(super) fn new(model: &NpaModel, cfg: &RenderProxyTrainingConfig) -> Self {
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

        Self {
            output_dims,
            liveness_output,
            material_output,
            scale_output,
            color_outputs,
            phase_output,
            liveness_update_cap,
        }
    }
}
