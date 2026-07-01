use burn_automata::{
    AutomataPreset, NpaConfig, ParticleSeed,
    kernels::{HashGridMode, TileGridConfig, assign_tiles},
    rollout::seed_particles_scaled,
};
use criterion::{Criterion, criterion_group, criterion_main};

fn tile_assignment_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("tile_assignment");
    for (preset, particles, tile_size) in [
        (AutomataPreset::Growing2d, 16_384usize, [2, 2, 1]),
        (AutomataPreset::Growing3dGs, 32_768usize, [4, 4, 4]),
    ] {
        let (config, mut grid) = NpaConfig::for_preset(preset);
        // Tile assignment is a bounded-grid strategy. Particle-hash grids are
        // intentionally rejected by the strategy analysis path because their
        // coordinates are hash-space, not finite geometric tile coordinates.
        grid.mode = HashGridMode::Grid;
        let (positions, _states) = seed_particles_scaled(
            1,
            particles,
            config.state_dims,
            config.spatial_dims,
            42,
            ParticleSeed::UniformCircle,
            NpaConfig::seed_scale_for_preset(preset),
        );
        let tiles = TileGridConfig::from_hashgrid(&grid, tile_size);

        group.bench_function(format!("{preset:?}_particles_{particles}"), |b| {
            b.iter(|| assign_tiles(&positions, 1, particles, &grid, &tiles).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, tile_assignment_bench);
criterion_main!(benches);
