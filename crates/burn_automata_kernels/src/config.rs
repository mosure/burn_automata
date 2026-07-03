#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Boundary {
    Periodic,
    #[default]
    Clamped,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum HashGridMode {
    #[default]
    Grid,
    Particle,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HashGridConfig {
    pub dim: usize,
    pub boundary: Boundary,
    pub mode: HashGridMode,
    pub grid_size: [usize; 3],
    pub eps: f32,
    pub max_particles_per_block: usize,
}

impl HashGridConfig {
    pub fn growing_2d() -> Self {
        Self {
            dim: 2,
            boundary: Boundary::Clamped,
            mode: HashGridMode::Grid,
            grid_size: [16, 16, 1],
            eps: 0.1,
            max_particles_per_block: 32,
        }
    }

    pub fn texture_2d() -> Self {
        Self {
            dim: 2,
            boundary: Boundary::Periodic,
            mode: HashGridMode::Grid,
            grid_size: [10, 10, 1],
            eps: 0.2,
            max_particles_per_block: 32,
        }
    }

    pub fn growing_3dgs() -> Self {
        Self {
            dim: 3,
            boundary: Boundary::Clamped,
            mode: HashGridMode::Particle,
            grid_size: [32, 32, 32],
            eps: 0.1,
            max_particles_per_block: 64,
        }
    }

    pub fn validate(&self) -> KernelResult<()> {
        if !(self.dim == 2 || self.dim == 3) {
            return Err(KernelError::InvalidDim(self.dim));
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return Err(KernelError::InvalidEps(self.eps));
        }
        for axis in 0..self.dim {
            if self.grid_size[axis] == 0 {
                return Err(KernelError::InvalidGridSize);
            }
            if self.mode == HashGridMode::Particle && !self.grid_size[axis].is_power_of_two() {
                return Err(KernelError::ParticleModeRequiresPowerOfTwo {
                    axis,
                    value: self.grid_size[axis],
                });
            }
        }
        Ok(())
    }

    pub fn cell_count(&self) -> usize {
        self.grid_size[..self.dim].iter().product()
    }

    pub fn spatial_mem_dims(&self) -> usize {
        if self.dim == 3 { 4 } else { 2 }
    }
}

impl Default for HashGridConfig {
    fn default() -> Self {
        Self::growing_2d()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    #[error("expected spatial dim 2 or 3, got {0}")]
    InvalidDim(usize),
    #[error("eps must be finite and positive, got {0}")]
    InvalidEps(f32),
    #[error("grid sizes must be non-zero for active dimensions")]
    InvalidGridSize,
    #[error("particle hashgrid mode requires power-of-two grid size at axis {axis}, got {value}")]
    ParticleModeRequiresPowerOfTwo { axis: usize, value: usize },
    #[error("tile size at axis {axis} must be non-zero")]
    InvalidTileSize { axis: usize },
    #[error("tile dim {tile_dim} does not match hashgrid dim {hashgrid_dim}")]
    TileDimMismatch {
        hashgrid_dim: usize,
        tile_dim: usize,
    },
    #[error("tile index was outside the configured tile grid")]
    TileIndexOutOfBounds,
    #[error("BVH leaf size must be non-zero")]
    InvalidBvhLeafSize,
    #[error("{strategy} spatial strategy does not support {reason}")]
    UnsupportedSpatialStrategy {
        strategy: &'static str,
        reason: &'static str,
    },
    #[error("positions length {positions} does not match batch_size * particle_count {expected}")]
    PositionShape { positions: usize, expected: usize },
    #[error(
        "state length {states} does not match batch_size * particle_count * state_dims {expected}"
    )]
    StateShape { states: usize, expected: usize },
    #[error("output buffer length {actual} does not match expected {expected}")]
    OutputShape { actual: usize, expected: usize },
}

pub type KernelResult<T> = Result<T, KernelError>;
