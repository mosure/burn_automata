use burn_automata::{
    AutomataPreset, FeatureBatchConfig, NpaConfig, NpaModel, SgdConfig, SupervisedTarget,
    feature_supervised_batch, supervised_train_step,
};
use criterion::{Criterion, criterion_group, criterion_main};

fn training_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("supervised_training");
    for (preset, rows) in [
        (AutomataPreset::Growing2d, 4096usize),
        (AutomataPreset::Growing3dGs, 2048usize),
    ] {
        let (config, _grid) = NpaConfig::for_preset(preset);
        let base_model = NpaModel::seeded(config, 42);
        let batch = feature_supervised_batch(
            &base_model,
            SupervisedTarget::ZeroUpdate,
            FeatureBatchConfig {
                rows,
                seed: 5,
                ..FeatureBatchConfig::default()
            },
        )
        .unwrap();

        group.bench_function(format!("{preset:?}_rows_{rows}"), |b| {
            b.iter_batched(
                || base_model.clone(),
                |mut model| {
                    supervised_train_step(
                        &mut model,
                        &batch,
                        SgdConfig {
                            learning_rate: 1.0e-3,
                            grad_clip_norm: 1.0,
                            ..SgdConfig::default()
                        },
                    )
                    .unwrap()
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, training_bench);
criterion_main!(benches);
