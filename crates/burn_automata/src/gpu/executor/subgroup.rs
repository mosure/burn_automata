use std::borrow::Cow;

pub(super) fn supports_subgroup_cooperative_sorted_cells(
    features: wgpu::Features,
    adapter_info: &wgpu::AdapterInfo,
) -> bool {
    features.contains(wgpu::Features::SUBGROUP)
        && adapter_info.subgroup_min_size == 32
        && adapter_info.subgroup_max_size == 32
}

pub(super) fn subgroup_cooperative_required_features(
    features: wgpu::Features,
    adapter_info: &wgpu::AdapterInfo,
) -> wgpu::Features {
    if supports_subgroup_cooperative_sorted_cells(features, adapter_info) {
        wgpu::Features::SUBGROUP
    } else {
        wgpu::Features::empty()
    }
}

pub(super) fn create_subgroup_cooperative_pipelines(
    device: &wgpu::Device,
    pipeline_layout: &wgpu::PipelineLayout,
    supported: bool,
) -> (
    Option<wgpu::ComputePipeline>,
    Option<wgpu::ComputePipeline>,
    Option<wgpu::ComputePipeline>,
) {
    if !supported {
        return (None, None, None);
    }

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("burn_automata_gpu_step_subgroup"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(format!(
            "{}\n{}",
            include_str!("../../gpu_step.wgsl"),
            include_str!("../../gpu_step_subgroup.wgsl")
        ))),
    });
    let density = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("burn_automata_subgroup_cooperative_density"),
        layout: Some(pipeline_layout),
        module: &shader,
        entry_point: Some("subgroup_cooperative_density_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let update = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("burn_automata_subgroup_cooperative_update"),
        layout: Some(pipeline_layout),
        module: &shader,
        entry_point: Some("subgroup_cooperative_update_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let adaptive_local = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("burn_automata_subgroup_adaptive_local_residual"),
        layout: Some(pipeline_layout),
        module: &shader,
        entry_point: Some("subgroup_adaptive_local_residual_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    (Some(density), Some(update), Some(adaptive_local))
}
