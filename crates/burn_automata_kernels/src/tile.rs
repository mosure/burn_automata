use crate::{
    Boundary, HashGridConfig, KernelError, KernelResult, hashgrid::cell_coords_for_position,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileGridConfig {
    pub dim: usize,
    pub boundary: Boundary,
    pub grid_size: [usize; 3],
    pub tile_size: [usize; 3],
}

impl TileGridConfig {
    pub fn from_hashgrid(hashgrid: &HashGridConfig, tile_size: [usize; 3]) -> Self {
        Self {
            dim: hashgrid.dim,
            boundary: hashgrid.boundary,
            grid_size: hashgrid.grid_size,
            tile_size,
        }
    }

    pub fn validate(&self) -> KernelResult<()> {
        if !(self.dim == 2 || self.dim == 3) {
            return Err(KernelError::InvalidDim(self.dim));
        }
        for axis in 0..self.dim {
            if self.grid_size[axis] == 0 {
                return Err(KernelError::InvalidGridSize);
            }
            if self.tile_size[axis] == 0 {
                return Err(KernelError::InvalidTileSize { axis });
            }
        }
        Ok(())
    }

    pub fn tile_grid_size(&self) -> [usize; 3] {
        let mut size = [1usize; 3];
        for (axis, slot) in size.iter_mut().enumerate().take(self.dim) {
            *slot = self.grid_size[axis].div_ceil(self.tile_size[axis]);
        }
        size
    }

    pub fn tile_count(&self) -> usize {
        self.tile_grid_size()[..self.dim].iter().product()
    }

    pub fn tile_index_for_cell(&self, cell_coords: [isize; 3]) -> Option<usize> {
        let tile_grid = self.tile_grid_size();
        let mut stride = 1usize;
        let mut index = 0usize;
        for axis in 0..self.dim {
            let grid_axis = self.grid_size[axis] as isize;
            let cell = match self.boundary {
                Boundary::Periodic => cell_coords[axis].rem_euclid(grid_axis),
                Boundary::Clamped => {
                    if cell_coords[axis] < 0 || cell_coords[axis] >= grid_axis {
                        return None;
                    }
                    cell_coords[axis]
                }
            };
            let tile = (cell as usize / self.tile_size[axis]).min(tile_grid[axis] - 1);
            index += tile * stride;
            stride *= tile_grid[axis];
        }
        Some(index)
    }

    pub fn neighbor_offsets(&self) -> Vec<[isize; 3]> {
        let mut offsets = Vec::with_capacity(if self.dim == 2 { 9 } else { 27 });
        let z_min = if self.dim == 3 { -1 } else { 0 };
        let z_max = if self.dim == 3 { 1 } else { 0 };
        for dz in z_min..=z_max {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    offsets.push([dx, dy, dz]);
                }
            }
        }
        offsets
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileAssignment {
    pub batch_size: usize,
    pub particle_count: usize,
    pub tile_count: usize,
    pub tile_offsets: Vec<usize>,
    pub permutation: Vec<usize>,
    pub inverse_permutation: Vec<usize>,
}

pub fn assign_tiles(
    positions: &[[f32; 4]],
    batch_size: usize,
    particle_count: usize,
    hashgrid: &HashGridConfig,
    tiles: &TileGridConfig,
) -> KernelResult<TileAssignment> {
    hashgrid.validate()?;
    tiles.validate()?;
    if hashgrid.dim != tiles.dim {
        return Err(KernelError::TileDimMismatch {
            hashgrid_dim: hashgrid.dim,
            tile_dim: tiles.dim,
        });
    }
    let expected = batch_size * particle_count;
    if positions.len() != expected {
        return Err(KernelError::PositionShape {
            positions: positions.len(),
            expected,
        });
    }

    let tile_count = tiles.tile_count();
    let mut counts = vec![0usize; batch_size * tile_count];
    for batch in 0..batch_size {
        for particle in 0..particle_count {
            let idx = batch * particle_count + particle;
            let tile = tile_for_position(&positions[idx], hashgrid, tiles)?;
            counts[batch * tile_count + tile] += 1;
        }
    }

    let mut tile_offsets = vec![0usize; counts.len() + 1];
    for (idx, count) in counts.iter().enumerate() {
        tile_offsets[idx + 1] = tile_offsets[idx] + count;
    }

    let mut write_heads = tile_offsets.clone();
    let mut permutation = vec![0usize; expected];
    let mut inverse_permutation = vec![0usize; expected];
    for batch in 0..batch_size {
        for particle in 0..particle_count {
            let idx = batch * particle_count + particle;
            let tile = tile_for_position(&positions[idx], hashgrid, tiles)?;
            let bin = batch * tile_count + tile;
            let dst = write_heads[bin];
            write_heads[bin] += 1;
            permutation[dst] = particle;
            inverse_permutation[idx] = dst - batch * particle_count;
        }
    }

    Ok(TileAssignment {
        batch_size,
        particle_count,
        tile_count,
        tile_offsets,
        permutation,
        inverse_permutation,
    })
}

pub fn tile_for_position(
    position: &[f32; 4],
    hashgrid: &HashGridConfig,
    tiles: &TileGridConfig,
) -> KernelResult<usize> {
    if hashgrid.dim != tiles.dim {
        return Err(KernelError::TileDimMismatch {
            hashgrid_dim: hashgrid.dim,
            tile_dim: tiles.dim,
        });
    }
    let coords = cell_coords_for_position(position, hashgrid);
    tiles
        .tile_index_for_cell(coords)
        .ok_or(KernelError::TileIndexOutOfBounds)
}
