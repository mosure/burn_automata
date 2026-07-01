use burn_automata::{
    AutomataPreset, NpaConfig, NpaModel, ParticleSeed, RolloutConfig, kernels::Splat2dConfig,
    kernels::splat_particles_2d, run_rollout,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing2d);
    let model = NpaModel::seeded(config, 42);
    let trace = run_rollout(
        &model,
        &grid,
        &RolloutConfig {
            particle_count: 512,
            steps: 8,
            update_prob: 1.0,
            ..RolloutConfig::default()
        },
        ParticleSeed::UniformCircle,
    )?;
    let colors = trace
        .states
        .chunks_exact(trace.state_dims)
        .map(|state| {
            let base = trace.state_dims.saturating_sub(3);
            [
                (state[base] + 0.5).clamp(0.0, 1.0),
                (state[base + 1] + 0.5).clamp(0.0, 1.0),
                (state[base + 2] + 0.5).clamp(0.0, 1.0),
            ]
        })
        .collect::<Vec<_>>();
    let image = splat_particles_2d(
        &trace.positions,
        &colors,
        Splat2dConfig {
            image_size: 64,
            ..Splat2dConfig::default()
        },
    );
    println!(
        "2d inference particles={} pixels={} final_mean_dx={:.6}",
        trace.particle_count,
        image.len(),
        trace.mean_dx.last().copied().unwrap_or_default()
    );
    Ok(())
}
