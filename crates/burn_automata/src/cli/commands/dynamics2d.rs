use crate::cli::prelude::*;

pub(crate) fn run_eval_dynamics_2d(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::EvalDynamics2d {
        preset,
        model,
        target_model,
        particles,
        steps,
        update_prob,
        seed,
        seed_scale,
        seed_mode,
        image_size,
        render_sigma_px,
        output,
    } = command
    else {
        unreachable!("run_eval_dynamics_2d called with the wrong command variant");
    };

    if particles == 0 {
        return Err(std::io::Error::other("--particles must be greater than zero").into());
    }
    if steps == 0 {
        return Err(std::io::Error::other("--steps must be greater than zero").into());
    }
    if !(0.0..=1.0).contains(&update_prob) || !update_prob.is_finite() {
        return Err(std::io::Error::other("--update-prob must be finite and in [0, 1]").into());
    }
    if image_size == 0 {
        return Err(std::io::Error::other("--image-size must be greater than zero").into());
    }
    if !render_sigma_px.is_finite() || render_sigma_px <= 0.0 {
        return Err(std::io::Error::other(
            "--render-sigma-px must be finite and greater than zero",
        )
        .into());
    }

    let preset: AutomataPreset = preset.into();
    let seed_scale = seed_scale.unwrap_or_else(|| NpaConfig::seed_scale_for_preset(preset));
    let seed_mode: ParticleSeed = seed_mode.into();
    let model_manifest = crate::import::load_manifest(&model)?;
    let target_manifest = crate::import::load_manifest(&target_model)?;
    if model_manifest.config != target_manifest.config {
        return Err(std::io::Error::other("model and target-model configs differ").into());
    }
    if model_manifest.config.spatial_dims != 2 {
        return Err(std::io::Error::other("eval-dynamics2d requires a 2D model").into());
    }
    if model_manifest.hashgrid != target_manifest.hashgrid {
        return Err(std::io::Error::other("model and target-model hashgrids differ").into());
    }
    let hashgrid = model_manifest.hashgrid.clone();
    let generated_model = model_manifest.into_model();
    let target = target_manifest.into_model();
    let rollout_cfg = RolloutConfig {
        steps,
        particle_count: particles,
        update_prob,
        seed,
        seed_scale,
        ..RolloutConfig::default()
    };
    let target_trace = run_rollout(&target, &hashgrid, &rollout_cfg, seed_mode)?;
    let generated_trace = run_rollout(&generated_model, &hashgrid, &rollout_cfg, seed_mode)?;
    let metrics = compare_2d_dynamics(
        &generated_trace,
        &target_trace,
        &hashgrid,
        image_size,
        render_sigma_px,
        update_prob,
        seed,
        seed_scale,
        seed_mode,
    )?;
    let report = CliDynamics2dEvalReport {
        preset,
        model: model.display().to_string(),
        target_model: target_model.display().to_string(),
        particle_count: particles,
        rollout_steps: steps,
        update_prob,
        seed,
        seed_scale,
        seed_mode,
        image_size,
        render_sigma_px,
        metrics,
    };
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&output, serde_json::to_string_pretty(&report)?)?;
    println!(
        "wrote {} position_psnr={:.3} state_psnr={:.3} render_rgb_psnr={:.3} mean_dx_mae={:.6}",
        output.display(),
        report.metrics.position_psnr_db,
        report.metrics.state_psnr_db,
        report.metrics.render_rgb_psnr_db,
        report.metrics.mean_dx_mae,
    );

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn compare_2d_dynamics(
    generated: &crate::RolloutTrace,
    target: &crate::RolloutTrace,
    grid: &burn_automata_kernels::HashGridConfig,
    image_size: usize,
    sigma: f32,
    update_prob: f32,
    seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
) -> Result<CliHyper2dDynamicsMetricsReport, Box<dyn std::error::Error>> {
    if generated.particle_count != target.particle_count {
        return Err(std::io::Error::other("generated and target particle counts differ").into());
    }
    if generated.steps != target.steps {
        return Err(std::io::Error::other("generated and target rollout steps differ").into());
    }
    if generated.state_dims != target.state_dims {
        return Err(std::io::Error::other("generated and target state dims differ").into());
    }
    let target_positions = flatten_positions(&target.positions);
    let generated_positions = flatten_positions(&generated.positions);
    let position_stats = compare_dynamic_signal(&generated_positions, &target_positions)?;
    let state_stats = compare_dynamic_signal(&generated.states, &target.states)?;
    let target_tail = tail_rgb_values(&target.states, target.state_dims)?;
    let generated_tail = tail_rgb_values(&generated.states, generated.state_dims)?;
    let tail_stats = compare_unit_signal(&generated_tail, &target_tail)?;
    let target_render = rasterize_tail_rgb_gaussian(
        &target.positions,
        &target.states,
        target.state_dims,
        grid,
        image_size,
        sigma,
    )?;
    let generated_render = rasterize_tail_rgb_gaussian(
        &generated.positions,
        &generated.states,
        generated.state_dims,
        grid,
        image_size,
        sigma,
    )?;
    let render_stats = compare_rgb_images(&generated_render.rgb, &target_render.rgb)?;
    let (mean_dx_mse, mean_dx_mae) = compare_mean_dx(&generated.mean_dx, &target.mean_dx)?;
    Ok(CliHyper2dDynamicsMetricsReport {
        particle_count: generated.particle_count,
        rollout_steps: generated.steps,
        update_prob,
        seed,
        seed_scale,
        seed_mode,
        image_size,
        render_sigma_px: sigma,
        position_mse: position_stats.mse,
        position_psnr_db: position_stats.psnr_db,
        state_mse: state_stats.mse,
        state_psnr_db: state_stats.psnr_db,
        tail_rgb_mse: tail_stats.mse,
        tail_rgb_psnr_db: tail_stats.psnr_db,
        render_rgb_mse: render_stats.mse,
        render_rgb_psnr_db: render_stats.psnr_db,
        mean_dx_mse,
        mean_dx_mae,
        target_final_mean_dx: target.mean_dx.last().copied().unwrap_or_default(),
        generated_final_mean_dx: generated.mean_dx.last().copied().unwrap_or_default(),
    })
}

#[derive(Debug)]
struct RenderedImage2d {
    rgb: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
struct MetricStats2d {
    mse: f32,
    psnr_db: f32,
}

fn flatten_positions(positions: &[[f32; 4]]) -> Vec<f32> {
    positions
        .iter()
        .flat_map(|position| [position[0], position[1], position[2], position[3]])
        .collect()
}

fn tail_rgb_values(
    states: &[f32],
    state_dims: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if state_dims < 3 {
        return Err(
            std::io::Error::other("tail RGB metrics require at least three channels").into(),
        );
    }
    Ok(states
        .chunks_exact(state_dims)
        .flat_map(|state| {
            let tail = state_dims - 3;
            [
                (state[tail] + 0.5).clamp(0.0, 1.0),
                (state[tail + 1] + 0.5).clamp(0.0, 1.0),
                (state[tail + 2] + 0.5).clamp(0.0, 1.0),
            ]
        })
        .collect())
}

fn rasterize_tail_rgb_gaussian(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    grid: &burn_automata_kernels::HashGridConfig,
    image_size: usize,
    sigma: f32,
) -> Result<RenderedImage2d, Box<dyn std::error::Error>> {
    if state_dims < 3 {
        return Err(
            std::io::Error::other("tail RGB metrics require at least three channels").into(),
        );
    }
    if states.len() < positions.len().saturating_mul(state_dims) {
        return Err(std::io::Error::other("state buffer is shorter than positions").into());
    }
    let (extent_x, extent_y) = grid_extents(grid)?;
    let half_x = extent_x * 0.5;
    let half_y = extent_y * 0.5;
    let radius = (sigma * 3.0).ceil().max(1.0) as isize;
    let inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma);
    let max_pixel = image_size.saturating_sub(1) as f32;
    let mut rgb = vec![0.0_f32; image_size * image_size * 3];
    let mut weights = vec![0.0_f32; image_size * image_size];

    for (row, position) in positions.iter().enumerate() {
        let px = (position[0] + half_x) / extent_x * max_pixel;
        let py = (position[1] + half_y) / extent_y * max_pixel;
        if !px.is_finite() || !py.is_finite() {
            continue;
        }
        let state_base = row * state_dims;
        let color = [
            (states[state_base + state_dims - 3] + 0.5).clamp(0.0, 1.0),
            (states[state_base + state_dims - 2] + 0.5).clamp(0.0, 1.0),
            (states[state_base + state_dims - 1] + 0.5).clamp(0.0, 1.0),
        ];
        let raw_min_x = px.floor() as isize - radius;
        let raw_max_x = px.ceil() as isize + radius;
        let raw_min_y = py.floor() as isize - radius;
        let raw_max_y = py.ceil() as isize + radius;
        let image_limit = image_size as isize - 1;
        if raw_max_x < 0 || raw_max_y < 0 || raw_min_x > image_limit || raw_min_y > image_limit {
            continue;
        }
        let min_x = raw_min_x.max(0) as usize;
        let max_x = raw_max_x.min(image_limit) as usize;
        let min_y = raw_min_y.max(0) as usize;
        let max_y = raw_max_y.min(image_limit) as usize;
        for y in min_y..=max_y {
            let dy = y as f32 - py;
            for x in min_x..=max_x {
                let dx = x as f32 - px;
                let weight = (-(dx * dx + dy * dy) * inv_two_sigma2).exp();
                let pixel = y * image_size + x;
                weights[pixel] += weight;
                let rgb_base = pixel * 3;
                rgb[rgb_base] += color[0] * weight;
                rgb[rgb_base + 1] += color[1] * weight;
                rgb[rgb_base + 2] += color[2] * weight;
            }
        }
    }
    for (pixel, weight) in weights.iter().enumerate() {
        if *weight <= 0.0 {
            continue;
        }
        let rgb_base = pixel * 3;
        rgb[rgb_base] /= *weight;
        rgb[rgb_base + 1] /= *weight;
        rgb[rgb_base + 2] /= *weight;
    }
    Ok(RenderedImage2d { rgb })
}

fn grid_extents(
    grid: &burn_automata_kernels::HashGridConfig,
) -> Result<(f32, f32), Box<dyn std::error::Error>> {
    let extent_x = grid.grid_size[0] as f32 * grid.eps;
    let extent_y = grid.grid_size[1] as f32 * grid.eps;
    if !extent_x.is_finite() || !extent_y.is_finite() || extent_x <= 0.0 || extent_y <= 0.0 {
        return Err(std::io::Error::other("hashgrid extent must be finite and positive").into());
    }
    Ok((extent_x, extent_y))
}

fn compare_rgb_images(
    generated: &[f32],
    target: &[f32],
) -> Result<MetricStats2d, Box<dyn std::error::Error>> {
    compare_signal_with_peak(generated, target, 1.0)
}

fn compare_unit_signal(
    generated: &[f32],
    target: &[f32],
) -> Result<MetricStats2d, Box<dyn std::error::Error>> {
    compare_signal_with_peak(generated, target, 1.0)
}

fn compare_dynamic_signal(
    generated: &[f32],
    target: &[f32],
) -> Result<MetricStats2d, Box<dyn std::error::Error>> {
    if generated.len() != target.len() {
        return Err(std::io::Error::other("generated and target signal sizes differ").into());
    }
    let peak = generated
        .iter()
        .chain(target)
        .fold(1.0_f32, |peak, value| peak.max(value.abs()));
    compare_signal_with_peak(generated, target, peak)
}

fn compare_signal_with_peak(
    generated: &[f32],
    target: &[f32],
    peak: f32,
) -> Result<MetricStats2d, Box<dyn std::error::Error>> {
    if generated.len() != target.len() {
        return Err(std::io::Error::other("generated and target metric sizes differ").into());
    }
    if generated.is_empty() {
        return Err(std::io::Error::other("metric buffers must not be empty").into());
    }
    let mse = generated
        .iter()
        .zip(target)
        .map(|(&generated_value, &target_value)| {
            let diff = generated_value - target_value;
            diff * diff
        })
        .sum::<f32>()
        / generated.len() as f32;
    Ok(MetricStats2d {
        mse,
        psnr_db: psnr(mse, peak),
    })
}

fn compare_mean_dx(
    generated: &[f32],
    target: &[f32],
) -> Result<(f32, f32), Box<dyn std::error::Error>> {
    if generated.len() != target.len() {
        return Err(std::io::Error::other("generated and target mean_dx sizes differ").into());
    }
    if generated.is_empty() {
        return Err(std::io::Error::other("mean_dx buffers must not be empty").into());
    }
    let mut mse = 0.0_f32;
    let mut mae = 0.0_f32;
    for (&generated_value, &target_value) in generated.iter().zip(target) {
        let diff = generated_value - target_value;
        mse += diff * diff;
        mae += diff.abs();
    }
    let len = generated.len() as f32;
    Ok((mse / len, mae / len))
}

fn psnr(mse: f32, peak: f32) -> f32 {
    if mse <= f32::EPSILON {
        99.0
    } else {
        20.0 * (peak.max(f32::MIN_POSITIVE) / mse.sqrt()).log10()
    }
}
