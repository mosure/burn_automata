#![allow(clippy::too_many_arguments)]

use super::*;

pub(crate) fn growth_3d_channel_opacity_stats(
    states: &[f32],
    state_dims: usize,
    channel: usize,
) -> Growth3dOpacityStats {
    if state_dims <= channel || states.is_empty() {
        return Growth3dOpacityStats {
            finite: false,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            mean: f32::NAN,
            active_min: f32::INFINITY,
            active_max: f32::NEG_INFINITY,
            active_mean: f32::NAN,
            active_count: 0,
            max_allowed: GROWTH_3D_MAX_FINAL_OPACITY_LOGIT,
        };
    }

    let mut finite = true;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0_f32;
    let mut count = 0usize;
    let mut active_min = f32::INFINITY;
    let mut active_max = f32::NEG_INFINITY;
    let mut active_sum = 0.0_f32;
    let mut active_count = 0usize;
    for state in states.chunks_exact(state_dims) {
        let opacity = state[channel];
        finite &= opacity.is_finite();
        min = min.min(opacity);
        max = max.max(opacity);
        sum += opacity;
        count += 1;
        if opacity > -1.0 {
            active_min = active_min.min(opacity);
            active_max = active_max.max(opacity);
            active_sum += opacity;
            active_count += 1;
        }
    }

    Growth3dOpacityStats {
        finite,
        min,
        max,
        mean: sum / count.max(1) as f32,
        active_min: if active_count == 0 {
            f32::INFINITY
        } else {
            active_min
        },
        active_max: if active_count == 0 {
            f32::NEG_INFINITY
        } else {
            active_max
        },
        active_mean: if active_count == 0 {
            f32::NAN
        } else {
            active_sum / active_count as f32
        },
        active_count,
        max_allowed: GROWTH_3D_MAX_FINAL_OPACITY_LOGIT,
    }
}

pub(crate) fn growth_3d_color_state_report(
    states: &[f32],
    state_dims: usize,
) -> Growth3dColorStateReport {
    if state_dims < 6 || states.is_empty() {
        return Growth3dColorStateReport {
            available: false,
            finite: false,
            count: 0,
            active_count: 0,
            mean_abs: f32::NAN,
            max_abs: f32::NAN,
            active_mean_abs: f32::NAN,
            active_max_abs: f32::NAN,
            active_channel_stddev: [f32::NAN; 3],
            active_channel_stddev_mean: f32::NAN,
        };
    }

    let tail = state_dims - 3;
    let mut finite = true;
    let mut count = 0usize;
    let mut active_count = 0usize;
    let mut sum_abs = 0.0_f32;
    let mut max_abs = 0.0_f32;
    let mut active_sum_abs = 0.0_f32;
    let mut active_max_abs = 0.0_f32;
    let mut active_sum = [0.0_f32; 3];
    let mut active_sum_sq = [0.0_f32; 3];

    for state in states.chunks_exact(state_dims) {
        count += 1;
        let mut particle_max_abs = 0.0_f32;
        for channel in 0..3 {
            let value = state[tail + channel];
            finite &= value.is_finite();
            particle_max_abs = particle_max_abs.max(value.abs());
        }
        sum_abs += particle_max_abs;
        max_abs = max_abs.max(particle_max_abs);

        if state[3] > -1.0 {
            active_count += 1;
            active_sum_abs += particle_max_abs;
            active_max_abs = active_max_abs.max(particle_max_abs);
            for channel in 0..3 {
                let value = state[tail + channel];
                active_sum[channel] += value;
                active_sum_sq[channel] += value * value;
            }
        }
    }

    let mut active_channel_stddev = [f32::NAN; 3];
    if active_count > 0 {
        for channel in 0..3 {
            let mean = active_sum[channel] / active_count as f32;
            let variance = (active_sum_sq[channel] / active_count as f32 - mean * mean).max(0.0);
            active_channel_stddev[channel] = variance.sqrt();
        }
    }
    let active_channel_stddev_mean = if active_count > 0 {
        active_channel_stddev.iter().sum::<f32>() / 3.0
    } else {
        f32::NAN
    };

    Growth3dColorStateReport {
        available: true,
        finite,
        count,
        active_count,
        mean_abs: sum_abs / count.max(1) as f32,
        max_abs,
        active_mean_abs: if active_count > 0 {
            active_sum_abs / active_count as f32
        } else {
            f32::NAN
        },
        active_max_abs: if active_count > 0 {
            active_max_abs
        } else {
            f32::NAN
        },
        active_channel_stddev,
        active_channel_stddev_mean,
    }
}

pub(crate) fn growth_3d_permutation_report(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    cfg: &RolloutConfig,
    seed_mode: ParticleSeed,
) -> Result<Growth3dPermutationReport, Box<dyn std::error::Error>> {
    let particle_count = cfg.particle_count.clamp(2, 256);
    let steps = cfg.steps.min(8);
    let (positions, states) = seed_particles_scaled(
        1,
        particle_count,
        model.config.state_dims,
        model.config.spatial_dims,
        cfg.seed,
        seed_mode,
        cfg.seed_scale,
    );
    let base = run_rollout_from_state(
        model,
        grid,
        positions.clone(),
        states.clone(),
        1,
        particle_count,
        steps,
        cfg.dt,
    )?;

    let mut order = (0..particle_count).collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(cfg.seed ^ 0x9a55_19e3_7ac3);
    order.shuffle(&mut rng);

    let mut shuffled_positions = vec![[0.0; 4]; particle_count];
    let mut shuffled_states = vec![0.0; states.len()];
    for (shuffled_idx, &source_idx) in order.iter().enumerate() {
        shuffled_positions[shuffled_idx] = positions[source_idx];
        let src = source_idx * model.config.state_dims;
        let dst = shuffled_idx * model.config.state_dims;
        shuffled_states[dst..dst + model.config.state_dims]
            .copy_from_slice(&states[src..src + model.config.state_dims]);
    }

    let shuffled = run_rollout_from_state(
        model,
        grid,
        shuffled_positions,
        shuffled_states,
        1,
        particle_count,
        steps,
        cfg.dt,
    )?;

    let mut inverse_order = vec![0usize; particle_count];
    for (shuffled_idx, &source_idx) in order.iter().enumerate() {
        inverse_order[source_idx] = shuffled_idx;
    }

    let mut max_position_error = 0.0_f32;
    let mut sum_position_error = 0.0_f32;
    let mut max_state_error = 0.0_f32;
    let mut sum_state_error = 0.0_f32;
    let mut state_count = 0usize;

    for (source_idx, &shuffled_idx) in inverse_order.iter().enumerate() {
        let base_position = base.positions[source_idx];
        let shuffled_position = shuffled.positions[shuffled_idx];
        let position_error = ((base_position[0] - shuffled_position[0]).powi(2)
            + (base_position[1] - shuffled_position[1]).powi(2)
            + (base_position[2] - shuffled_position[2]).powi(2))
        .sqrt();
        max_position_error = max_position_error.max(position_error);
        sum_position_error += position_error;

        let base_state = source_idx * model.config.state_dims;
        let shuffled_state = shuffled_idx * model.config.state_dims;
        for channel in 0..model.config.state_dims {
            let state_error = (base.states[base_state + channel]
                - shuffled.states[shuffled_state + channel])
                .abs();
            max_state_error = max_state_error.max(state_error);
            sum_state_error += state_error;
            state_count += 1;
        }
    }

    let mean_position_error = sum_position_error / particle_count.max(1) as f32;
    let mean_state_error = sum_state_error / state_count.max(1) as f32;
    let passed = max_position_error <= 1.0e-3 && max_state_error <= 1.0e-3;

    Ok(Growth3dPermutationReport {
        particle_count,
        steps,
        max_position_error,
        mean_position_error,
        max_state_error,
        mean_state_error,
        passed,
    })
}

pub(crate) fn growth_3d_seed_perturbation_report(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    cfg: &RolloutConfig,
    seed_mode: ParticleSeed,
) -> Result<Growth3dSeedPerturbationReport, Box<dyn std::error::Error>> {
    let particle_count = cfg.particle_count.clamp(32, 512);
    let steps = cfg.steps.clamp(1, 32);
    let jitter_radius = (growth_3d_seed_radius(cfg.seed_scale) * 0.10).max(cfg.seed_scale * 0.002);
    let (positions, states) = seed_particles_scaled(
        1,
        particle_count,
        model.config.state_dims,
        model.config.spatial_dims,
        cfg.seed,
        seed_mode,
        cfg.seed_scale,
    );
    let mut seed_active = Vec::with_capacity(particle_count);
    let mut active_seed_count = 0usize;
    for state in states.chunks_exact(model.config.state_dims) {
        let active = state[3] > -1.0;
        seed_active.push(active);
        if active {
            active_seed_count += 1;
        }
    }

    let base = run_rollout_from_state(
        model,
        grid,
        positions.clone(),
        states.clone(),
        1,
        particle_count,
        steps,
        cfg.dt,
    )?;

    let mut perturbed_positions = positions;
    let mut rng = StdRng::seed_from_u64(cfg.seed ^ 0x005e_ed93_7d3d);
    for position in &mut perturbed_positions {
        for value in position.iter_mut().take(3) {
            *value += rng.random_range(-jitter_radius..=jitter_radius);
        }
    }
    let perturbed = run_rollout_from_state(
        model,
        grid,
        perturbed_positions,
        states,
        1,
        particle_count,
        steps,
        cfg.dt,
    )?;

    let base_activation = growth_3d_activation_report(&base, &seed_active, active_seed_count);
    let perturbed_activation =
        growth_3d_activation_report(&perturbed, &seed_active, active_seed_count);
    let base_motion = growth_3d_motion_report(&base.mean_dx);
    let perturbed_motion = growth_3d_motion_report(&perturbed.mean_dx);
    let base_color = growth_3d_color_state_report(&base.states, base.state_dims);
    let perturbed_color = growth_3d_color_state_report(&perturbed.states, perturbed.state_dims);

    let active_count_ratio = finite_ratio(
        perturbed_activation.final_active_count as f32,
        base_activation.final_active_count.max(1) as f32,
    );
    let final_active_max_radius_ratio = finite_ratio(
        perturbed_activation.final_active_max_radius,
        base_activation.final_active_max_radius,
    );
    let peak_motion_ratio = finite_ratio(perturbed_motion.peak_mean_dx, base_motion.peak_mean_dx);
    let color_state_mean_abs_ratio =
        finite_ratio(perturbed_color.active_mean_abs, base_color.active_mean_abs);

    let base_growth = base_activation.final_active_count > active_seed_count.max(1) * 2
        && base_activation.newly_activated_fraction >= 0.25
        && base_motion.peak_mean_dx > 1.0e-3;
    let perturbed_growth = perturbed_activation.final_active_count > active_seed_count.max(1) * 2
        && perturbed_activation.newly_activated_fraction >= 0.25
        && perturbed_motion.peak_mean_dx > 1.0e-3;
    let comparable_growth = (0.50..=2.00).contains(&active_count_ratio)
        && (0.50..=2.00).contains(&final_active_max_radius_ratio)
        && (0.25..=4.00).contains(&peak_motion_ratio);
    let passed = base_growth && perturbed_growth && comparable_growth;

    Ok(Growth3dSeedPerturbationReport {
        particle_count,
        steps,
        jitter_radius,
        seed: cfg.seed,
        active_seed_count,
        base_final_active_count: base_activation.final_active_count,
        perturbed_final_active_count: perturbed_activation.final_active_count,
        active_count_ratio,
        base_newly_activated_fraction: base_activation.newly_activated_fraction,
        perturbed_newly_activated_fraction: perturbed_activation.newly_activated_fraction,
        base_final_active_max_radius: base_activation.final_active_max_radius,
        perturbed_final_active_max_radius: perturbed_activation.final_active_max_radius,
        final_active_max_radius_ratio,
        base_peak_mean_dx: base_motion.peak_mean_dx,
        perturbed_peak_mean_dx: perturbed_motion.peak_mean_dx,
        peak_motion_ratio,
        base_color_state_mean_abs: base_color.active_mean_abs,
        perturbed_color_state_mean_abs: perturbed_color.active_mean_abs,
        color_state_mean_abs_ratio,
        passed,
    })
}

pub(crate) fn finite_ratio(numerator: f32, denominator: f32) -> f32 {
    if !numerator.is_finite() || !denominator.is_finite() {
        return f32::NAN;
    }
    if denominator.abs() <= 1.0e-8 {
        if numerator.abs() <= 1.0e-8 {
            1.0
        } else if numerator.is_sign_positive() {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        }
    } else {
        numerator / denominator
    }
}
