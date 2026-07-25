use burn_automata_kernels::{Boundary, HashGridConfig, HashGridMode};

use crate::{AutomataError, AutomataResult, NpaModel};

use super::super::types::WgpuNeighborMode;
use super::{constants::*, neighbor::neighbor_layout_code, util::u32_checked};

#[allow(clippy::too_many_arguments)]
pub(in crate::gpu) fn validate_gpu_step(
    model: &NpaModel,
    positions: &[[f32; 4]],
    states: &[f32],
    batch_size: usize,
    particle_count: usize,
    grid: &HashGridConfig,
    dt: f32,
    update_prob: f32,
) -> AutomataResult<()> {
    validate_gpu_step_impl(
        model,
        positions,
        states,
        batch_size,
        particle_count,
        grid,
        dt,
        update_prob,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_gpu_step_impl(
    model: &NpaModel,
    positions: &[[f32; 4]],
    states: &[f32],
    batch_size: usize,
    particle_count: usize,
    grid: &HashGridConfig,
    dt: f32,
    update_prob: f32,
) -> AutomataResult<()> {
    validate_gpu_model_config(model, grid)?;
    if batch_size == 0 {
        return Err(AutomataError::InvalidArgument(
            "WGPU step requires batch_size greater than zero".to_owned(),
        ));
    }
    if particle_count == 0 {
        return Err(AutomataError::InvalidArgument(
            "particle_count must be greater than zero".to_owned(),
        ));
    }
    if positions.len() != batch_size * particle_count {
        return Err(AutomataError::InvalidArgument(format!(
            "positions len {} does not match batch_size * particle_count {}",
            positions.len(),
            batch_size * particle_count
        )));
    }
    if states.len() != positions.len() * model.config.state_dims {
        return Err(AutomataError::InvalidArgument(format!(
            "states len {} does not match positions * state_dims {}",
            states.len(),
            positions.len() * model.config.state_dims
        )));
    }
    if !dt.is_finite() {
        return Err(AutomataError::InvalidArgument(format!(
            "dt must be finite, got {dt}"
        )));
    }
    if !(0.0..=1.0).contains(&update_prob) || !update_prob.is_finite() {
        return Err(AutomataError::InvalidArgument(format!(
            "update_prob must be finite and in [0, 1], got {update_prob}"
        )));
    }
    u32_checked(positions.len(), "positions len")?;
    u32_checked(particle_count, "particle_count")?;
    model.weights.validate(&model.config)
}

pub(in crate::gpu) fn validate_gpu_model_config(
    model: &NpaModel,
    grid: &HashGridConfig,
) -> AutomataResult<()> {
    model.validate()?;
    grid.validate()?;
    if grid.dim != model.config.spatial_dims {
        return Err(AutomataError::InvalidArgument(format!(
            "grid dim {} does not match model spatial dims {}",
            grid.dim, model.config.spatial_dims
        )));
    }
    if !model.config.state_grad || !model.config.density_grad {
        return Err(AutomataError::InvalidArgument(
            "WGPU step currently expects state_grad=true and density_grad=true".to_owned(),
        ));
    }
    if model.config.state_dims > MAX_STATE_DIMS {
        return Err(AutomataError::InvalidArgument(format!(
            "state_dims {} exceeds WGPU shader max {MAX_STATE_DIMS}",
            model.config.state_dims
        )));
    }
    if model.config.hidden_dims > MAX_HIDDEN_DIMS {
        return Err(AutomataError::InvalidArgument(format!(
            "hidden_dims {} exceeds WGPU shader max {MAX_HIDDEN_DIMS}",
            model.config.hidden_dims
        )));
    }
    if model.config.perception_dims() > MAX_FEATURE_DIMS {
        return Err(AutomataError::InvalidArgument(format!(
            "perception_dims {} exceeds WGPU shader max {MAX_FEATURE_DIMS}",
            model.config.perception_dims()
        )));
    }
    if model.config.update_dims() > MAX_OUTPUT_DIMS {
        return Err(AutomataError::InvalidArgument(format!(
            "update_dims {} exceeds WGPU shader max {MAX_OUTPUT_DIMS}",
            model.config.update_dims()
        )));
    }
    u32_checked(grid.cell_count(), "cell_count")?;
    model.weights.validate(&model.config)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::gpu) fn gpu_params(
    model: &NpaModel,
    total: usize,
    batch_size: usize,
    particle_count: usize,
    grid: &HashGridConfig,
    dt: f32,
    bucket_capacity: usize,
    neighbor_mode: WgpuNeighborMode,
    update_prob: f32,
    seed: u64,
) -> AutomataResult<[u32; PARAM_COUNT]> {
    let mut params = [0; PARAM_COUNT];
    params[PARAM_TOTAL] = u32_checked(total, "total")?;
    params[PARAM_PARTICLE_COUNT] = u32_checked(particle_count, "particle_count")?;
    params[PARAM_STATE_DIMS] = u32_checked(model.config.state_dims, "state_dims")?;
    params[PARAM_HIDDEN_DIMS] = u32_checked(model.config.hidden_dims, "hidden_dims")?;
    params[PARAM_SPATIAL_DIMS] = u32_checked(model.config.spatial_dims, "spatial_dims")?;
    params[PARAM_FEATURE_DIMS] = u32_checked(model.config.perception_dims(), "feature_dims")?;
    params[PARAM_OUTPUT_DIMS] = u32_checked(model.config.update_dims(), "output_dims")?;
    params[PARAM_GRID_X] = u32_checked(grid.grid_size[0], "grid_size[0]")?;
    params[PARAM_GRID_Y] = u32_checked(grid.grid_size[1], "grid_size[1]")?;
    params[PARAM_GRID_Z] = u32_checked(grid.grid_size[2], "grid_size[2]")?;
    let cell_count = grid
        .cell_count()
        .checked_mul(batch_size)
        .ok_or_else(|| AutomataError::InvalidArgument("batched cell count overflow".to_owned()))?;
    params[PARAM_CELL_COUNT] = u32_checked(cell_count, "batched cell_count")?;
    params[PARAM_PERIODIC] = u32::from(grid.boundary == Boundary::Periodic);
    params[PARAM_LOG_GRAD] = u32::from(model.config.log_norm_grad);
    params[PARAM_LOG_DENSITY_GRAD] = u32::from(model.config.log_norm_density_grad);
    params[PARAM_POSITION_FEATURES] = u32::from(model.config.position_features);
    params[PARAM_EPS] = grid.eps.to_bits();
    params[PARAM_ALPHA] = model.config.alpha.to_bits();
    params[PARAM_DT] = dt.to_bits();
    params[PARAM_SMOOTH_COEF] = smoothing_poly6_normalization(grid).to_bits();
    params[PARAM_SPIKY_COEF] = gradient_spiky_normalization(grid).to_bits();
    params[PARAM_DENSITY_SCALE] = density_gradient_scale(model, grid, particle_count).to_bits();
    params[PARAM_GRAD_SCALE] = state_gradient_scale(model, grid).to_bits();
    params[PARAM_BUCKET_CAPACITY] = u32_checked(bucket_capacity, "bucket_capacity")?;
    params[PARAM_MOTION_EPS] = model.config.motion_eps(grid.eps).to_bits();
    params[PARAM_UPDATE_PROB] = update_prob.to_bits();
    params[PARAM_STEP_INDEX] = 0;
    params[PARAM_RANDOM_SEED] = gpu_random_seed(seed);
    params[PARAM_LANE_SEEDS_START] = gpu_random_seed(seed);
    params[PARAM_PARTICLE_GRID] = u32::from(grid.mode == HashGridMode::Particle);
    params[PARAM_NEIGHBOR_LAYOUT] = neighbor_layout_code(neighbor_mode);
    params[PARAM_BATCH_SIZE] = u32_checked(batch_size, "batch_size")?;
    params[PARAM_SCALE_EQUIVARIANT] = u32::from(model.config.scale_equivariant());
    params[PARAM_SUPPORT_BIN_COUNT] = 1;
    params[PARAM_SPATIAL_CELL_COUNT] = u32_checked(grid.cell_count(), "spatial cell count")?;
    params[PARAM_SUPPORT_BIN_MIN] = grid.eps.to_bits();
    params[PARAM_SUPPORT_BIN_MAX] = grid.eps.to_bits();
    params[PARAM_SUPPORT_BIN_RATIO] = 2.0_f32.to_bits();
    params[PARAM_RESIDENT_CAPACITY] = u32_checked(total, "resident capacity")?;
    Ok(params)
}

pub(in crate::gpu) const fn gpu_random_seed(seed: u64) -> u32 {
    (seed as u32) ^ ((seed >> 32) as u32)
}

pub(in crate::gpu) fn smoothing_poly6_normalization(grid: &HashGridConfig) -> f32 {
    if grid.dim == 2 {
        4.0 / (std::f32::consts::PI * grid.eps.powi(8))
    } else {
        315.0 / (64.0 * std::f32::consts::PI * grid.eps.powi(9))
    }
}

pub(in crate::gpu) fn gradient_spiky_normalization(grid: &HashGridConfig) -> f32 {
    if grid.dim == 2 {
        10.0 / (std::f32::consts::PI * grid.eps.powi(5))
    } else {
        15.0 / (std::f32::consts::PI * grid.eps.powi(6))
    }
}

pub(in crate::gpu) fn state_gradient_scale(model: &NpaModel, grid: &HashGridConfig) -> f32 {
    if model.config.scale_equivariant() {
        grid.eps / model.config.eps0.max(f32::MIN_POSITIVE)
    } else {
        1.0
    }
}

pub(in crate::gpu) fn density_gradient_scale(
    model: &NpaModel,
    grid: &HashGridConfig,
    particle_count: usize,
) -> f32 {
    let scale = if model.config.scale_equivariant() {
        (grid.eps / model.config.eps0.max(f32::MIN_POSITIVE)).powi(1 + grid.dim as i32)
    } else {
        1.0
    };
    if model.config.particle_density_equivariant() {
        scale / particle_count.max(1) as f32
    } else {
        scale
    }
}
