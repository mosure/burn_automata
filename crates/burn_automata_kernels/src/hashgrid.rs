use crate::{Boundary, HashGridConfig, HashGridMode, KernelError, KernelResult};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HashGridSnapshot {
    pub batch_size: usize,
    pub particle_count: usize,
    pub cell_count: usize,
    /// Prefix-sum offsets into `permutation`, length `batch_size * cell_count + 1`.
    pub bin_offsets: Vec<usize>,
    /// Binned index -> original local particle index.
    pub permutation: Vec<usize>,
    /// Original local particle index -> binned index.
    pub inverse_permutation: Vec<usize>,
}

pub fn build_hashgrid(
    positions: &[[f32; 4]],
    batch_size: usize,
    particle_count: usize,
    cfg: &HashGridConfig,
) -> KernelResult<HashGridSnapshot> {
    cfg.validate()?;
    let expected = batch_size * particle_count;
    if positions.len() != expected {
        return Err(KernelError::PositionShape {
            positions: positions.len(),
            expected,
        });
    }

    let cell_count = cfg.cell_count();
    let mut counts = vec![0usize; batch_size * cell_count];
    for batch in 0..batch_size {
        for particle in 0..particle_count {
            let idx = batch * particle_count + particle;
            let cell = cell_for_position(&positions[idx], cfg);
            counts[batch * cell_count + cell] += 1;
        }
    }

    let mut bin_offsets = vec![0usize; counts.len() + 1];
    for (idx, count) in counts.iter().enumerate() {
        bin_offsets[idx + 1] = bin_offsets[idx] + count;
    }

    let mut write_heads = bin_offsets.clone();
    let mut permutation = vec![0usize; expected];
    let mut inverse_permutation = vec![0usize; expected];
    for batch in 0..batch_size {
        for particle in 0..particle_count {
            let idx = batch * particle_count + particle;
            let cell = cell_for_position(&positions[idx], cfg);
            let bin = batch * cell_count + cell;
            let dst = write_heads[bin];
            write_heads[bin] += 1;
            permutation[dst] = particle;
            inverse_permutation[batch * particle_count + particle] = dst - batch * particle_count;
        }
    }

    Ok(HashGridSnapshot {
        batch_size,
        particle_count,
        cell_count,
        bin_offsets,
        permutation,
        inverse_permutation,
    })
}

pub(crate) fn cell_for_position(position: &[f32; 4], cfg: &HashGridConfig) -> usize {
    if cfg.mode == HashGridMode::Particle {
        return particle_cell_hash(cell_coords_for_position(position, cfg), cfg);
    }

    let mut stride = 1usize;
    let mut hash = 0usize;
    for (axis, coordinate) in position.iter().enumerate().take(cfg.dim) {
        let size = cfg.grid_size[axis];
        let extent = cfg.eps * size as f32;
        let half = extent * 0.5;
        let mut cell = ((*coordinate + half) / cfg.eps).floor() as isize;
        match cfg.boundary {
            Boundary::Periodic => {
                let size_i = size as isize;
                cell = cell.rem_euclid(size_i);
            }
            Boundary::Clamped => {
                cell = cell.clamp(0, size as isize - 1);
            }
        }
        hash += cell as usize * stride;
        stride *= size;
    }
    hash
}

pub(crate) fn cell_coords_for_position(position: &[f32; 4], cfg: &HashGridConfig) -> [isize; 3] {
    let mut coords = [0isize; 3];
    for (axis, coordinate) in position.iter().enumerate().take(cfg.dim) {
        if cfg.mode == HashGridMode::Particle {
            coords[axis] = (coordinate / cfg.eps).floor() as isize;
            continue;
        }

        let size = cfg.grid_size[axis];
        let extent = cfg.eps * size as f32;
        let half = extent * 0.5;
        let mut cell = ((*coordinate + half) / cfg.eps).floor() as isize;
        match cfg.boundary {
            Boundary::Periodic => {
                cell = cell.rem_euclid(size as isize);
            }
            Boundary::Clamped => {
                cell = cell.clamp(0, size as isize - 1);
            }
        }
        coords[axis] = cell;
    }
    coords
}

pub(crate) fn cell_index_from_coords(coords: [isize; 3], cfg: &HashGridConfig) -> Option<usize> {
    if cfg.mode == HashGridMode::Particle {
        return Some(particle_cell_hash(coords, cfg));
    }

    let mut stride = 1usize;
    let mut hash = 0usize;
    for (axis, coord) in coords.iter().enumerate().take(cfg.dim) {
        let size = cfg.grid_size[axis] as isize;
        let cell = match cfg.boundary {
            Boundary::Periodic => coord.rem_euclid(size),
            Boundary::Clamped => {
                if *coord < 0 || *coord >= size {
                    return None;
                }
                *coord
            }
        };
        hash += cell as usize * stride;
        stride *= size as usize;
    }
    Some(hash)
}

fn particle_cell_hash(coords: [isize; 3], cfg: &HashGridConfig) -> usize {
    let mut hash = mix_coord(coords[0], 0x9e37_79b9) ^ mix_coord(coords[1], 0x85eb_ca6b);
    if cfg.dim == 3 {
        hash ^= mix_coord(coords[2], 0xc2b2_ae35);
    }
    hash as usize % cfg.cell_count()
}

fn mix_coord(value: isize, salt: u32) -> u32 {
    let mut x = (value as i32 as u32) ^ salt;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

pub(crate) fn neighbor_delta(lhs: &[f32; 4], rhs: &[f32; 4], cfg: &HashGridConfig) -> [f32; 4] {
    let mut delta = [0.0; 4];
    for axis in 0..cfg.dim {
        let mut d = rhs[axis] - lhs[axis];
        if cfg.boundary == Boundary::Periodic {
            let extent = cfg.grid_size[axis] as f32 * cfg.eps;
            let half = extent * 0.5;
            if d > half {
                d -= extent;
            } else if d < -half {
                d += extent;
            }
        }
        delta[axis] = d;
    }
    delta
}

pub(crate) fn wrap_position(mut position: [f32; 4], cfg: &HashGridConfig) -> [f32; 4] {
    if cfg.boundary != Boundary::Periodic {
        return position;
    }
    for (axis, coordinate) in position.iter_mut().enumerate().take(cfg.dim) {
        let extent = cfg.grid_size[axis] as f32 * cfg.eps;
        let half = extent * 0.5;
        *coordinate = (*coordinate + half).rem_euclid(extent) - half;
    }
    position
}
