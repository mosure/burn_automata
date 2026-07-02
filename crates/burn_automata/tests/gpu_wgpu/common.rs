pub(crate) use burn_automata::{
    AutomataError, AutomataPreset, NpaConfig, NpaModel, NpaWeights, ParticleSeed,
    rollout::{
        UV_TORUS_INITIAL_OPACITY_LOGIT, UV_TORUS_INITIAL_SCALE, UV_TORUS_MINOR_RATIO,
        UV_TORUS_MOTION_GAIN, UV_TORUS_OPACITY_GROWTH_DELTA, UV_TORUS_RESIDUAL_DECAY,
        seed_particles_scaled, uv_torus_position_color, uv_torus_sample,
    },
};

pub(crate) fn uv_torus_growth_model(config: NpaConfig) -> NpaModel {
    let mut weights = NpaWeights::zeros(&config);
    let input_dims = config.perception_dims();
    for axis in 0..3 {
        let pos_hidden = axis * 2;
        let neg_hidden = pos_hidden + 1;
        weights.w1[pos_hidden * input_dims + axis] = 1.0;
        weights.w1[neg_hidden * input_dims + axis] = -1.0;
        weights.w2[axis * config.hidden_dims + pos_hidden] = UV_TORUS_MOTION_GAIN;
        weights.w2[axis * config.hidden_dims + neg_hidden] = -UV_TORUS_MOTION_GAIN;
        let residual_out = config.spatial_dims + axis;
        weights.w2[residual_out * config.hidden_dims + pos_hidden] = -UV_TORUS_RESIDUAL_DECAY;
        weights.w2[residual_out * config.hidden_dims + neg_hidden] = UV_TORUS_RESIDUAL_DECAY;
    }
    weights.b2[config.spatial_dims + 3] = UV_TORUS_OPACITY_GROWTH_DELTA;
    NpaModel { config, weights }
}

pub(crate) fn assert_preset_parity(
    preset: AutomataPreset,
    particles: usize,
    seed: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let seed_scale = NpaConfig::seed_scale_for_preset(preset);
    let (config, grid) = NpaConfig::for_preset(preset);
    let model = NpaModel::seeded(config, 42);
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        model.config.state_dims,
        model.config.spatial_dims,
        seed,
        ParticleSeed::UniformCircle,
        seed_scale,
    );

    let cpu = model.step_cpu(&positions, &states, 1, particles, &grid, 1.0, None)?;
    let gpu = match burn_automata::gpu::step_wgpu_blocking(
        &model, &positions, &states, 1, particles, &grid, 1.0,
    ) {
        Ok(output) => output,
        Err(AutomataError::InvalidArgument(message)) if is_missing_wgpu(&message) => {
            eprintln!("skipping WGPU parity test: {message}");
            return Ok(());
        }
        Err(err) => return Err(err.into()),
    };

    let max_pos = max_position_abs_error(&cpu.next_positions, &gpu.next_positions);
    let max_state = max_abs_error(&cpu.next_states, &gpu.next_states);
    let max_density = max_abs_error(&cpu.perception.density, &gpu.density);
    let density_index = max_abs_error_index(&cpu.perception.density, &gpu.density);
    eprintln!(
        "{preset:?}: max_pos={max_pos:.8} max_state={max_state:.8} max_density={max_density:.8} density_index={density_index:?}"
    );

    assert!(
        max_pos <= 2.5e-3,
        "max position abs error {max_pos} exceeded tolerance"
    );
    assert!(
        max_state <= 2.5e-3,
        "max state abs error {max_state} exceeded tolerance"
    );
    assert!(
        max_density <= 2.5e-3,
        "max density abs error {max_density} exceeded tolerance"
    );
    Ok(())
}

pub(crate) fn new_executor_or_skip()
-> Result<Option<burn_automata::gpu::WgpuAutomataExecutor>, Box<dyn std::error::Error>> {
    match burn_automata::gpu::WgpuAutomataExecutor::new_blocking() {
        Ok(executor) => Ok(Some(executor)),
        Err(AutomataError::InvalidArgument(message)) if is_missing_wgpu(&message) => {
            eprintln!("skipping WGPU test: {message}");
            Ok(None)
        }
        Err(err) => Err(err.into()),
    }
}

pub(crate) fn is_missing_wgpu(message: &str) -> bool {
    message.contains("no WGPU adapter") || message.contains("failed to create WGPU device")
}

pub(crate) fn max_position_abs_error(lhs: &[[f32; 4]], rhs: &[[f32; 4]]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .flat_map(|(lhs, rhs)| lhs.iter().zip(rhs.iter()).map(|(a, b)| (a - b).abs()))
        .fold(0.0, f32::max)
}

pub(crate) fn gpu_update_mask(count: usize, update_prob: f32, step: u32, seed: u64) -> Vec<f32> {
    let seed = (seed as u32) ^ ((seed >> 32) as u32);
    (0..count)
        .map(|idx| {
            let random = gpu_random01(idx as u32, step, seed);
            f32::from(random < update_prob)
        })
        .collect()
}

pub(crate) fn gpu_random01(particle: u32, step: u32, seed: u32) -> f32 {
    let mixed = hash_u32(particle ^ hash_u32(step.wrapping_add(0x9e37_79b9)) ^ seed);
    ((mixed >> 8) as f32) * (1.0 / 16_777_216.0)
}

pub(crate) fn hash_u32(value: u32) -> u32 {
    let mut x = value;
    x = (x ^ 61) ^ (x >> 16);
    x = x.wrapping_add(x << 3);
    x ^= x >> 4;
    x = x.wrapping_mul(0x27d4_eb2d);
    x ^ (x >> 15)
}

pub(crate) fn max_abs_error(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

pub(crate) fn max_abs_error_index(lhs: &[f32], rhs: &[f32]) -> Option<(usize, f32, f32)> {
    lhs.iter()
        .zip(rhs.iter())
        .enumerate()
        .max_by(|(_, (lhs_a, rhs_a)), (_, (lhs_b, rhs_b))| {
            ((*lhs_a - *rhs_a).abs())
                .partial_cmp(&(*lhs_b - *rhs_b).abs())
                .unwrap()
        })
        .map(|(idx, (lhs, rhs))| (idx, *lhs, *rhs))
}
