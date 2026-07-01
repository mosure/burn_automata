use burn_automata::{
    AutomataPreset, NpaConfig, NpaModel, ParticleSeed, RolloutConfig, run_rollout,
};
use criterion::{Criterion, criterion_group, criterion_main};

fn rollout_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("rollout_cpu");
    for (preset, particles, steps) in [
        (AutomataPreset::Growing2d, 4096usize, 1usize),
        (AutomataPreset::Texture2d, 4096usize, 1usize),
        (AutomataPreset::Growing3dGs, 16_384usize, 1usize),
    ] {
        let (config, grid) = NpaConfig::for_preset(preset);
        let model = NpaModel::seeded(config, 42);
        group.bench_function(
            format!("{preset:?}_particles_{particles}_steps_{steps}"),
            |b| {
                b.iter(|| {
                    run_rollout(
                        &model,
                        &grid,
                        &RolloutConfig {
                            particle_count: particles,
                            steps,
                            update_prob: 1.0,
                            seed_scale: NpaConfig::seed_scale_for_preset(preset),
                            ..RolloutConfig::default()
                        },
                        ParticleSeed::UniformCircle,
                    )
                    .unwrap()
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, rollout_bench);
criterion_main!(benches);
