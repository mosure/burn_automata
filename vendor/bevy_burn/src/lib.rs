//! Local Bevy/Burn bridge extension points used by `bevy_automata`.
//!
//! The published `bevy_burn` crate currently focuses on texture transfer. This
//! workspace crate adds the buffer-level API shape needed for gaussian planar
//! buffers without forcing the first CPU/reference viewer path to fake
//! zero-copy behavior.

use std::marker::PhantomData;

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        render_resource::{BufferAddress, BufferUsages},
        storage::ShaderBuffer,
    },
};
use burn::prelude::Backend;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingDirection {
    BurnToBevy,
    BevyToBurn,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferKind {
    Cpu,
    Gpu,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BurnBufferBinding {
    pub label: &'static str,
    pub binding: u32,
    pub offset: BufferAddress,
    pub size: Option<BufferAddress>,
    pub usage: BufferUsages,
}

impl BurnBufferBinding {
    pub fn storage(label: &'static str, binding: u32, size: Option<BufferAddress>) -> Self {
        Self {
            label,
            binding,
            offset: 0,
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        }
    }
}

#[derive(Component, Clone, Debug)]
pub struct BurnShaderBufferHandle(pub Handle<ShaderBuffer>);

#[derive(Component, Clone, Debug)]
pub struct BurnBufferBridge<B: Backend> {
    pub direction: BindingDirection,
    pub transfer: TransferKind,
    pub bindings: Vec<BurnBufferBinding>,
    pub dirty: bool,
    marker: PhantomData<fn() -> B>,
}

impl<B: Backend> Default for BurnBufferBridge<B> {
    fn default() -> Self {
        Self {
            direction: BindingDirection::BurnToBevy,
            transfer: TransferKind::Gpu,
            bindings: Vec::new(),
            dirty: true,
            marker: PhantomData,
        }
    }
}

impl<B: Backend> BurnBufferBridge<B> {
    pub fn new(
        direction: BindingDirection,
        transfer: TransferKind,
        bindings: Vec<BurnBufferBinding>,
    ) -> Self {
        Self {
            direction,
            transfer,
            bindings,
            dirty: true,
            marker: PhantomData,
        }
    }

    pub fn from_binding(
        direction: BindingDirection,
        transfer: TransferKind,
        binding: BurnBufferBinding,
    ) -> Self {
        Self::new(direction, transfer, vec![binding])
    }
}

pub fn add_shader_buffer_bridge<B: Backend>(
    buffers: &mut Assets<ShaderBuffer>,
    bytes: &[u8],
    binding: BurnBufferBinding,
    direction: BindingDirection,
    transfer: TransferKind,
) -> (BurnBufferBridge<B>, BurnShaderBufferHandle) {
    let handle = buffers.add(shader_buffer_from_bytes(bytes, &binding));
    (
        BurnBufferBridge::from_binding(direction, transfer, binding),
        BurnShaderBufferHandle(handle),
    )
}

pub fn add_empty_shader_buffer_bridge<B: Backend>(
    buffers: &mut Assets<ShaderBuffer>,
    byte_len: usize,
    binding: BurnBufferBinding,
    direction: BindingDirection,
    transfer: TransferKind,
) -> (BurnBufferBridge<B>, BurnShaderBufferHandle) {
    let handle = buffers.add(shader_buffer_with_size(byte_len, &binding));
    (
        BurnBufferBridge::from_binding(direction, transfer, binding),
        BurnShaderBufferHandle(handle),
    )
}

pub fn shader_buffer_from_bytes(bytes: &[u8], binding: &BurnBufferBinding) -> ShaderBuffer {
    let mut buffer = ShaderBuffer::new(bytes, RenderAssetUsages::default());
    buffer.buffer_description.label = Some(binding.label);
    buffer.buffer_description.size = bytes.len() as BufferAddress;
    buffer.buffer_description.usage = binding.usage;
    buffer
}

pub fn shader_buffer_with_size(byte_len: usize, binding: &BurnBufferBinding) -> ShaderBuffer {
    let mut buffer = ShaderBuffer::with_size(byte_len, RenderAssetUsages::default());
    buffer.buffer_description.label = Some(binding.label);
    buffer.buffer_description.usage = binding.usage;
    buffer
}

pub struct BevyBurnBufferBridgePlugin<B: Backend> {
    marker: PhantomData<fn() -> B>,
}

impl<B: Backend> Default for BevyBurnBufferBridgePlugin<B> {
    fn default() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<B: Backend + 'static> Plugin for BevyBurnBufferBridgePlugin<B> {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, mark_clean::<B>);
    }
}

fn mark_clean<B: Backend>(
    mut bridges: Query<&mut BurnBufferBridge<B>, Changed<BurnBufferBridge<B>>>,
) {
    for mut bridge in &mut bridges {
        if bridge.transfer == TransferKind::Cpu {
            bridge.dirty = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestBackend = burn::backend::NdArray;

    #[test]
    fn shader_buffer_keeps_binding_descriptor() {
        let binding = BurnBufferBinding::storage("particles", 2, Some(16));
        let buffer = shader_buffer_from_bytes(&[1, 2, 3, 4], &binding);

        assert_eq!(buffer.data.as_deref(), Some(&[1, 2, 3, 4][..]));
        assert_eq!(buffer.buffer_description.label, Some("particles"));
        assert_eq!(buffer.buffer_description.size, 4);
        assert_eq!(buffer.buffer_description.usage, binding.usage);
    }

    #[test]
    fn adding_bridge_registers_shader_buffer_asset() {
        let mut buffers = Assets::<ShaderBuffer>::default();
        let binding = BurnBufferBinding::storage("gaussians", 0, Some(64));

        let (bridge, handle) = add_empty_shader_buffer_bridge::<TestBackend>(
            &mut buffers,
            64,
            binding,
            BindingDirection::BurnToBevy,
            TransferKind::Gpu,
        );

        assert_eq!(bridge.bindings.len(), 1);
        assert_eq!(bridge.direction, BindingDirection::BurnToBevy);
        assert_eq!(bridge.transfer, TransferKind::Gpu);
        assert!(buffers.get(&handle.0).is_some());
        assert_eq!(buffers.get(&handle.0).unwrap().buffer_description.size, 64);
    }
}
