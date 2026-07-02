use crate::{AutomataError, AutomataResult};

use super::constants::WORKGROUP_SIZE;

pub(in crate::gpu) fn u32_checked(value: usize, label: &str) -> AutomataResult<u32> {
    u32::try_from(value).map_err(|_| {
        AutomataError::InvalidArgument(format!("{label} value {value} exceeds u32::MAX"))
    })
}

pub(in crate::gpu) fn dispatch_groups(total: usize) -> AutomataResult<u32> {
    let total = u32_checked(total, "dispatch total")?;
    Ok(total.div_ceil(WORKGROUP_SIZE))
}

pub(in crate::gpu) fn byte_len<T>(len: usize) -> AutomataResult<wgpu::BufferAddress> {
    let bytes = len
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| AutomataError::InvalidArgument("buffer byte length overflow".to_owned()))?;
    Ok(bytes as wgpu::BufferAddress)
}
