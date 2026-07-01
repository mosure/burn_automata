use burn_automata::{
    AutomataPreset, NpaConfig, NpaModel, ParticleSeed, RolloutConfig,
    kernels::GaussianDecodeConfig, kernels::decode_gaussians_3d, run_rollout,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing3dGs);
    let model = NpaModel::seeded(config, 7);
    let trace = run_rollout(
        &model,
        &grid,
        &RolloutConfig {
            particle_count: 512,
            steps: 4,
            update_prob: 1.0,
            ..RolloutConfig::default()
        },
        ParticleSeed::UniformCircle,
    )?;
    let gaussian_dims = model.config.output_dims.unwrap_or(20);
    let mut decoded = vec![0.0; trace.particle_count * gaussian_dims];
    seed_gaussian_tail(&mut decoded, gaussian_dims);
    let gaussians = decode_gaussians_3d(
        &trace.positions,
        &decoded,
        gaussian_dims,
        GaussianDecodeConfig::default(),
    );
    println!(
        "3d inference particles={} gaussians={} final_mean_dx={:.6}",
        trace.particle_count,
        gaussians.len(),
        trace.mean_dx.last().copied().unwrap_or_default()
    );
    Ok(())
}

fn seed_gaussian_tail(states: &mut [f32], state_dims: usize) {
    for state in states.chunks_exact_mut(state_dims) {
        if state_dims >= 20 {
            state[state_dims - 16] = 1.0;
            state[state_dims - 12] = 1.0;
            state[state_dims - 8] = 1.0;
        }
    }
}
