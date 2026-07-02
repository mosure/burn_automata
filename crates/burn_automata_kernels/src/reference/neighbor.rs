use super::*;

pub(crate) fn for_each_neighbor<F>(
    idx: usize,
    positions: &[[f32; 4]],
    snapshot: &HashGridSnapshot,
    cfg: &HashGridConfig,
    mut f: F,
) where
    F: FnMut(usize, [f32; 4], f32),
{
    let batch = idx / snapshot.particle_count;
    let batch_base = batch * snapshot.particle_count;
    let pi = positions[idx];
    let center = cell_coords_for_position(&pi, cfg);
    let z_min = if cfg.dim == 3 { -1 } else { 0 };
    let z_max = if cfg.dim == 3 { 1 } else { 0 };
    let eps2 = cfg.eps * cfg.eps;

    for dz in z_min..=z_max {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let coords = [center[0] + dx, center[1] + dy, center[2] + dz];
                let Some(cell) = cell_index_from_coords(coords, cfg) else {
                    continue;
                };
                let bin = batch * snapshot.cell_count + cell;
                for binned in snapshot.bin_offsets[bin]..snapshot.bin_offsets[bin + 1] {
                    let j = batch_base + snapshot.permutation[binned];
                    if cfg.mode == HashGridMode::Particle
                        && cell_coords_for_position(&positions[j], cfg) != coords
                    {
                        continue;
                    }
                    let delta = neighbor_delta(&pi, &positions[j], cfg);
                    let r2 = delta[..cfg.dim].iter().map(|v| v * v).sum::<f32>();
                    if r2 < eps2 {
                        f(j, delta, r2);
                    }
                }
            }
        }
    }
}
