use burn_automata::{
    NpaConfig, NpaModel, NpaWeights, SgdConfig, SupervisedBatch, supervised_train_step,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = NpaConfig {
        hidden_dims: 32,
        ..NpaConfig::growing_2d()
    };
    let mut model = NpaModel {
        config: config.clone(),
        weights: NpaWeights::seeded(&config, 3),
    };
    let rows = 128;
    let features = (0..rows * config.perception_dims())
        .map(|idx| (idx as f32 * 0.017).sin())
        .collect::<Vec<_>>();
    let target_update = vec![0.0; rows * config.update_dims()];
    for step in 0..8 {
        let report = supervised_train_step(
            &mut model,
            &SupervisedBatch {
                features: features.clone(),
                target_update: target_update.clone(),
            },
            SgdConfig {
                learning_rate: 5e-3,
                grad_clip_norm: 10.0,
                ..SgdConfig::default()
            },
        )?;
        println!(
            "step={step} loss={:.6} grad={:.6}",
            report.loss, report.grad_norm
        );
    }
    Ok(())
}
