use std::sync::mpsc;

use wgpu::util::DeviceExt;

use crate::{AutomataError, AutomataResult, NpaModel};

use super::super::types::WgpuGaussianBufferRefs;
use super::{constants::GAUSSIAN_SH_COEFF_COUNT, util::byte_len};

pub(in crate::gpu) fn flatten_positions(positions: &[[f32; 4]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(positions.len() * 4);
    for position in positions {
        out.extend_from_slice(position);
    }
    out
}

pub(in crate::gpu) fn packed_weights(model: &NpaModel) -> Vec<f32> {
    let mut out = Vec::with_capacity(
        model.weights.w1.len()
            + model.weights.b1.len()
            + model.weights.w2.len()
            + model.weights.b2.len(),
    );
    out.extend_from_slice(&model.weights.w1);
    out.extend_from_slice(&model.weights.b1);
    out.extend_from_slice(&model.weights.w2);
    out.extend_from_slice(&model.weights.b2);
    out
}

pub(in crate::gpu) fn unflatten_positions(values: &[f32]) -> AutomataResult<Vec<[f32; 4]>> {
    if !values.len().is_multiple_of(4) {
        return Err(AutomataError::InvalidArgument(format!(
            "position readback length {} is not divisible by 4",
            values.len()
        )));
    }
    Ok(values
        .chunks_exact(4)
        .map(|chunk| [chunk[0], chunk[1], chunk[2], chunk[3]])
        .collect())
}
pub(in crate::gpu) fn storage_buffer_f32(
    device: &wgpu::Device,
    label: &'static str,
    values: &[f32],
) -> wgpu::Buffer {
    storage_buffer_f32_with_usage(device, label, values, wgpu::BufferUsages::STORAGE)
}

pub(in crate::gpu) fn storage_buffer_f32_with_usage(
    device: &wgpu::Device,
    label: &'static str,
    values: &[f32],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(values),
        usage,
    })
}

pub(in crate::gpu) fn uniform_buffer_u32(
    device: &wgpu::Device,
    label: &'static str,
    values: &[u32],
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

pub(in crate::gpu) fn staging_read_buffer(
    device: &wgpu::Device,
    label: &'static str,
    f32_len: usize,
) -> AutomataResult<wgpu::Buffer> {
    Ok(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: byte_len::<f32>(f32_len)?,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }))
}

pub(in crate::gpu) fn staging_read_buffer_u32(
    device: &wgpu::Device,
    label: &'static str,
    u32_len: usize,
) -> AutomataResult<wgpu::Buffer> {
    Ok(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: byte_len::<u32>(u32_len)?,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }))
}

pub(in crate::gpu) fn bind_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

pub(in crate::gpu) fn storage_layout_entry(
    binding: u32,
    read_only: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(in crate::gpu) fn uniform_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(in crate::gpu) fn gaussian_count_too_small(
    gaussian: &WgpuGaussianBufferRefs<'_>,
    count: usize,
) -> bool {
    let required_vec4 = byte_len::<f32>(count * 4).unwrap_or(wgpu::BufferAddress::MAX);
    let required_sh =
        byte_len::<f32>(count * GAUSSIAN_SH_COEFF_COUNT).unwrap_or(wgpu::BufferAddress::MAX);
    gaussian.position_visibility.size() < required_vec4
        || gaussian.spherical_harmonic.size() < required_sh
        || gaussian.rotation.size() < required_vec4
        || gaussian.scale_opacity.size() < required_vec4
}

pub(in crate::gpu) fn read_f32_buffer(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    f32_len: usize,
) -> AutomataResult<Vec<f32>> {
    read_mapped_buffer(device, buffer, f32_len)
}

pub(in crate::gpu) fn read_u32_buffer(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    u32_len: usize,
) -> AutomataResult<Vec<u32>> {
    read_mapped_buffer(device, buffer, u32_len)
}

pub(in crate::gpu) fn read_mapped_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    len: usize,
) -> AutomataResult<Vec<T>> {
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|err| AutomataError::InvalidArgument(format!("WGPU poll failed: {err}")))?;
    receiver
        .recv()
        .map_err(|err| AutomataError::InvalidArgument(format!("WGPU map callback failed: {err}")))?
        .map_err(|err| AutomataError::InvalidArgument(format!("WGPU buffer map failed: {err}")))?;

    let mapped = slice.get_mapped_range();
    let values = bytemuck::cast_slice::<u8, T>(&mapped)
        .iter()
        .take(len)
        .copied()
        .collect();
    drop(mapped);
    buffer.unmap();
    Ok(values)
}
