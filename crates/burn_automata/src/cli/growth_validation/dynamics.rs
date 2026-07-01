#![allow(clippy::too_many_arguments)]

use super::*;

pub(crate) fn growth_3d_empty_robustness_report(seed: u64) -> Growth3dRobustnessReport {
    growth_3d_robustness_report(vec![Growth3dRobustnessSeedReport {
        seed,
        gate_passed: false,
        strict_passed: false,
        catalog_sanity_passed: false,
        strict_score: f32::INFINITY,
        no_seed_coordinate_scaffold: false,
        render_loss: f32::INFINITY,
        density_psnr_db: f32::NEG_INFINITY,
        color_psnr_db: f32::NEG_INFINITY,
        depth_psnr_db: f32::NEG_INFINITY,
        active_seed_count: 0,
        final_active_count: 0,
        newly_activated_fraction: 0.0,
        active_extent_growth: false,
        active_extent_bbox_ratio: 0.0,
        active_extent_min_axis_ratio: 0.0,
        final_opacity_max: f32::INFINITY,
        material_visible_particles_live: false,
        inactive_material_visible_fraction: 1.0,
        max_inactive_material_opacity: f32::INFINITY,
        color_state_emerged: false,
        final_active_color_state_mean_abs: f32::NAN,
        final_active_color_state_stddev_mean: f32::NAN,
        permutation_consistent: false,
        permutation_max_position_error: f32::INFINITY,
        permutation_max_state_error: f32::INFINITY,
        gaussian_scale_budget: false,
        gaussian_scale_budget_loss: f32::INFINITY,
        gaussian_oversize_fraction: f32::INFINITY,
        seed_perturbation_stable: false,
        perturbed_newly_activated_fraction: 0.0,
        perturbed_active_count_ratio: 0.0,
        perturbed_peak_motion_ratio: 0.0,
        local_front_coherent: false,
        front_local_newly_activated_fraction: 0.0,
        front_max_nearest_previous_active_distance: f32::INFINITY,
        temporal_activation_progressive: false,
        temporal_geometry_progressive: false,
        final_active_target_coverage_fraction: 0.0,
        final_material_visible_target_coverage_fraction: 0.0,
        surface_coverage_profile: false,
        final_active_surface_covered_bin_fraction: 0.0,
        final_active_surface_mean_bin_covered_fraction: 0.0,
        material_visible_surface_coverage_profile: false,
        final_material_visible_surface_covered_bin_fraction: 0.0,
        final_material_visible_surface_mean_bin_covered_fraction: 0.0,
        surface_normal_coverage: false,
        final_active_surface_normal_covered_bin_fraction: 0.0,
        final_active_surface_normal_mean_bin_covered_fraction: 0.0,
        material_visible_surface_normal_coverage: false,
        final_material_visible_surface_normal_covered_bin_fraction: 0.0,
        final_material_visible_surface_normal_mean_bin_covered_fraction: 0.0,
        final_active_surface_max: f32::INFINITY,
        material_visible_surface_tail_bounded: false,
        final_material_visible_surface_tail_p99_distance: f32::INFINITY,
        final_material_visible_surface_tail_over_threshold_fraction: 1.0,
        final_material_visible_surface_tail_opacity_weighted_over_threshold_fraction: 1.0,
        failure_reasons: Vec::new(),
    }])
}

pub(crate) fn growth_3d_seed_has_coordinate_scaffold(seed_mode: ParticleSeed) -> bool {
    growth_3d_seed_writes_coordinate_scaffold(seed_mode)
}

pub(crate) fn growth_3d_non_scaffold_seed_abs_max(
    state_dims: usize,
    seed_mode: ParticleSeed,
    seed_states: &[f32],
) -> f32 {
    let material_opacity_channel = growth_3d_material_opacity_channel(state_dims);
    let allow_coordinate_scaffold = growth_3d_seed_has_coordinate_scaffold(seed_mode);
    let mut abs_max = 0.0_f32;
    for state in seed_states.chunks_exact(state_dims) {
        for (channel, value) in state.iter().enumerate() {
            if channel == GROWTH_3D_LIVENESS_CHANNEL
                || Some(channel) == material_opacity_channel
                || (allow_coordinate_scaffold && channel < 3)
            {
                continue;
            }
            abs_max = abs_max.max(value.abs());
        }
    }
    abs_max
}

pub(crate) fn growth_3d_motion_report(mean_dx: &[f32]) -> Growth3dMotionReport {
    if mean_dx.is_empty() {
        return Growth3dMotionReport {
            first_step_mean_dx: 0.0,
            peak_mean_dx: 0.0,
            peak_step: 0,
            final_step_mean_dx: 0.0,
            mean_dx: 0.0,
            late_mean_dx: 0.0,
            late_to_peak_ratio: 0.0,
            active_step_fraction: 0.0,
            sustained_step_fraction: 0.0,
        };
    }

    let first_step_mean_dx = mean_dx[0];
    let final_step_mean_dx = mean_dx[mean_dx.len() - 1];
    let mut peak_mean_dx = 0.0_f32;
    let mut peak_step = 0usize;
    let mut sum = 0.0_f32;
    for (step, value) in mean_dx.iter().copied().enumerate() {
        sum += value;
        if value > peak_mean_dx {
            peak_mean_dx = value;
            peak_step = step;
        }
    }
    let mean = sum / mean_dx.len() as f32;
    let late_start = mean_dx.len() * 3 / 4;
    let late_slice = &mean_dx[late_start..];
    let late_mean_dx = late_slice.iter().copied().sum::<f32>() / late_slice.len().max(1) as f32;
    let active_threshold = 1.0e-3;
    let sustained_threshold = (peak_mean_dx * 0.05).max(active_threshold);
    let active_steps = mean_dx
        .iter()
        .filter(|value| value.is_finite() && **value > active_threshold)
        .count();
    let sustained_steps = mean_dx
        .iter()
        .filter(|value| value.is_finite() && **value > sustained_threshold)
        .count();

    Growth3dMotionReport {
        first_step_mean_dx,
        peak_mean_dx,
        peak_step,
        final_step_mean_dx,
        mean_dx: mean,
        late_mean_dx,
        late_to_peak_ratio: if peak_mean_dx > 1.0e-8 {
            late_mean_dx / peak_mean_dx
        } else {
            0.0
        },
        active_step_fraction: active_steps as f32 / mean_dx.len() as f32,
        sustained_step_fraction: sustained_steps as f32 / mean_dx.len() as f32,
    }
}

#[derive(Clone)]
pub(crate) struct Growth3dFrontSnapshot {
    pub(crate) positions: Vec<[f32; 4]>,
    pub(crate) active: Vec<bool>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn growth_3d_front_report(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    rollout_cfg: RolloutConfig,
    seed_mode: ParticleSeed,
    seed_positions: &[[f32; 4]],
    seed_states: &[f32],
    final_trace: &crate::RolloutTrace,
) -> Result<Growth3dFrontReport, Box<dyn std::error::Error>> {
    let max_allowed_distance = growth_3d_front_distance_threshold(rollout_cfg.seed_scale);
    let mut snapshots = Vec::new();
    for steps in growth_3d_temporal_sample_steps(rollout_cfg.steps) {
        let snapshot = if steps == 0 {
            growth_3d_front_snapshot(seed_positions, seed_states, model.config.state_dims)
        } else if steps == rollout_cfg.steps {
            growth_3d_front_snapshot(
                &final_trace.positions,
                &final_trace.states,
                final_trace.state_dims,
            )
        } else {
            let trace = run_rollout(
                model,
                grid,
                &RolloutConfig {
                    steps,
                    ..rollout_cfg.clone()
                },
                seed_mode,
            )?;
            growth_3d_front_snapshot(&trace.positions, &trace.states, trace.state_dims)
        };
        snapshots.push(snapshot);
    }

    let mut transition_count = 0usize;
    let mut newly_activated_count = 0usize;
    let mut local_newly_activated_count = 0usize;
    let mut finite = true;
    let mut sum_nearest = 0.0_f32;
    let mut max_nearest = 0.0_f32;

    for pair in snapshots.windows(2) {
        let previous = &pair[0];
        let current = &pair[1];
        if previous.positions.len() != current.positions.len()
            || previous.active.len() != current.active.len()
        {
            finite = false;
            continue;
        }
        let previous_active_positions = previous
            .positions
            .iter()
            .zip(previous.active.iter())
            .filter_map(|(position, active)| (*active).then_some(*position))
            .collect::<Vec<_>>();
        if previous_active_positions.is_empty() {
            continue;
        }
        let mut transition_newly_activated = 0usize;
        for idx in 0..current.active.len() {
            if !current.active[idx] || previous.active[idx] {
                continue;
            }
            transition_newly_activated += 1;
            newly_activated_count += 1;
            let distance =
                nearest_position_distance(current.positions[idx], &previous_active_positions);
            finite &= distance.is_finite();
            sum_nearest += distance;
            max_nearest = max_nearest.max(distance);
            if distance <= max_allowed_distance {
                local_newly_activated_count += 1;
            }
        }
        if transition_newly_activated > 0 {
            transition_count += 1;
        }
    }

    let local_newly_activated_fraction = if newly_activated_count > 0 {
        local_newly_activated_count as f32 / newly_activated_count as f32
    } else {
        0.0
    };
    let mean_nearest_previous_active_distance = if newly_activated_count > 0 {
        sum_nearest / newly_activated_count as f32
    } else {
        f32::INFINITY
    };
    let passed = finite
        && newly_activated_count > 0
        && transition_count >= 2
        && local_newly_activated_fraction >= 0.90
        && mean_nearest_previous_active_distance <= max_allowed_distance * 0.75;

    Ok(Growth3dFrontReport {
        transition_count,
        newly_activated_count,
        local_newly_activated_count,
        local_newly_activated_fraction,
        mean_nearest_previous_active_distance,
        max_nearest_previous_active_distance: if newly_activated_count > 0 {
            max_nearest
        } else {
            f32::INFINITY
        },
        max_allowed_distance,
        finite,
        passed,
    })
}

pub(crate) fn growth_3d_front_snapshot(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
) -> Growth3dFrontSnapshot {
    let active = positions
        .iter()
        .enumerate()
        .map(|(idx, _)| state_dims > 3 && states[idx * state_dims + 3] > -1.0)
        .collect::<Vec<_>>();
    Growth3dFrontSnapshot {
        positions: positions.to_vec(),
        active,
    }
}

pub(crate) fn nearest_position_distance(position: [f32; 4], candidates: &[[f32; 4]]) -> f32 {
    candidates
        .iter()
        .map(|candidate| {
            ((position[0] - candidate[0]).powi(2)
                + (position[1] - candidate[1]).powi(2)
                + (position[2] - candidate[2]).powi(2))
            .sqrt()
        })
        .fold(f32::INFINITY, f32::min)
}

pub(crate) fn growth_3d_front_distance_threshold(seed_scale: f32) -> f32 {
    growth_3d_seed_radius(seed_scale) * 2.5
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn growth_3d_temporal_report(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    rollout_cfg: RolloutConfig,
    seed_mode: ParticleSeed,
    seed_positions: &[[f32; 4]],
    seed_states: &[f32],
    seed_active: &[bool],
    active_seed_count: usize,
    final_trace: &crate::RolloutTrace,
    coverage_samples: usize,
    coverage_threshold: f32,
) -> Result<Growth3dTemporalReport, Box<dyn std::error::Error>> {
    let mut samples = Vec::new();
    for steps in growth_3d_temporal_sample_steps(rollout_cfg.steps) {
        if steps == 0 {
            samples.push(growth_3d_temporal_sample_report(
                steps,
                seed_positions,
                seed_states,
                model.config.state_dims,
                seed_positions,
                seed_active,
                target,
                coverage_samples,
                coverage_threshold,
            ));
        } else if steps == rollout_cfg.steps {
            samples.push(growth_3d_temporal_sample_report(
                steps,
                &final_trace.positions,
                &final_trace.states,
                final_trace.state_dims,
                seed_positions,
                seed_active,
                target,
                coverage_samples,
                coverage_threshold,
            ));
        } else {
            let trace = run_rollout(
                model,
                grid,
                &RolloutConfig {
                    steps,
                    ..rollout_cfg.clone()
                },
                seed_mode,
            )?;
            samples.push(growth_3d_temporal_sample_report(
                steps,
                &trace.positions,
                &trace.states,
                trace.state_dims,
                seed_positions,
                seed_active,
                target,
                coverage_samples,
                coverage_threshold,
            ));
        }
    }

    let first_growth_step = samples
        .iter()
        .find(|sample| {
            sample.active_count > active_seed_count
                && sample.active_count >= active_seed_count.saturating_mul(2).max(1)
        })
        .map(|sample| sample.steps);
    let half_activation_step = samples
        .iter()
        .find(|sample| sample.active_fraction >= 0.50)
        .map(|sample| sample.steps);
    let full_activation_step = samples
        .iter()
        .find(|sample| sample.active_fraction >= 0.95)
        .map(|sample| sample.steps);
    let activation_span_steps =
        if let (Some(first), Some(full)) = (first_growth_step, full_activation_step) {
            full.saturating_sub(first)
        } else {
            0
        };
    let progressive_activation = match (
        first_growth_step,
        half_activation_step,
        full_activation_step,
    ) {
        (Some(first), Some(half), Some(full)) => {
            first < half && half < full && activation_span_steps >= rollout_cfg.steps / 4
        }
        _ => false,
    };
    let (surface_mean_ratio, target_coverage_mean_ratio, target_coverage_fraction_delta) =
        match (samples.first(), samples.last()) {
            (Some(initial), Some(final_sample)) => {
                let surface_mean_ratio = if initial.active_surface.mean_distance.is_finite()
                    && initial.active_surface.mean_distance > 1.0e-6
                {
                    final_sample.active_surface.mean_distance / initial.active_surface.mean_distance
                } else {
                    f32::INFINITY
                };
                let target_coverage_mean_ratio =
                    if initial.target_coverage.mean_distance.is_finite()
                        && initial.target_coverage.mean_distance > 1.0e-6
                    {
                        final_sample.target_coverage.mean_distance
                            / initial.target_coverage.mean_distance
                    } else {
                        f32::INFINITY
                    };
                let target_coverage_fraction_delta = final_sample.target_coverage.covered_fraction
                    - initial.target_coverage.covered_fraction;
                (
                    surface_mean_ratio,
                    target_coverage_mean_ratio,
                    target_coverage_fraction_delta,
                )
            }
            _ => (f32::INFINITY, f32::INFINITY, 0.0),
        };
    let geometry_progressive = target_coverage_mean_ratio < 0.85
        && target_coverage_fraction_delta >= 0.10
        && surface_mean_ratio < 0.95;

    Ok(Growth3dTemporalReport {
        samples,
        first_growth_step,
        half_activation_step,
        full_activation_step,
        activation_span_steps,
        progressive_activation,
        surface_mean_ratio,
        target_coverage_mean_ratio,
        target_coverage_fraction_delta,
        geometry_progressive,
    })
}

pub(crate) fn growth_3d_temporal_sample_steps(steps: usize) -> Vec<usize> {
    let mut samples = vec![0, steps];
    let mut step = 1usize;
    while step < steps {
        samples.push(step);
        step = step.saturating_mul(2);
        if step == 0 {
            break;
        }
    }
    samples.sort_unstable();
    samples.dedup();
    samples
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn growth_3d_temporal_sample_report(
    steps: usize,
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    seed_positions: &[[f32; 4]],
    seed_active: &[bool],
    target: &TriangleMeshTarget,
    coverage_samples: usize,
    coverage_threshold: f32,
) -> Growth3dTemporalSampleReport {
    let mut active_count = 0usize;
    let mut newly_activated_count = 0usize;
    let mut active_radius_sum = 0.0_f32;
    let mut active_max_radius = 0.0_f32;

    for (idx, position) in positions.iter().enumerate() {
        let opacity = states[idx * state_dims + 3];
        if opacity > -1.0 {
            active_count += 1;
            if !seed_active[idx] {
                newly_activated_count += 1;
            }
            let radius =
                (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                    .sqrt();
            active_radius_sum += radius;
            active_max_radius = active_max_radius.max(radius);
        }
    }

    Growth3dTemporalSampleReport {
        steps,
        active_count,
        active_fraction: active_count as f32 / positions.len().max(1) as f32,
        newly_activated_count,
        final_active_mean_radius: if active_count > 0 {
            active_radius_sum / active_count as f32
        } else {
            0.0
        },
        final_active_max_radius: active_max_radius,
        mean_displacement: growth_3d_mean_displacement(seed_positions, positions),
        active_surface: growth_3d_active_surface_stats(positions, states, state_dims, target),
        target_coverage: target_coverage_stats(
            positions,
            target,
            coverage_samples,
            coverage_threshold,
        ),
    }
}

pub(crate) fn growth_3d_activation_report(
    trace: &crate::RolloutTrace,
    seed_active: &[bool],
    active_seed_count: usize,
) -> Growth3dActivationReport {
    let mut final_active_count = 0usize;
    let mut newly_activated_count = 0usize;
    let mut final_active_radius_sum = 0.0_f32;
    let mut final_active_max_radius = 0.0_f32;
    for (idx, position) in trace.positions.iter().enumerate() {
        let opacity = trace.states[idx * trace.state_dims + 3];
        if opacity > -1.0 {
            final_active_count += 1;
            if !seed_active[idx] {
                newly_activated_count += 1;
            }
            let radius =
                (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                    .sqrt();
            final_active_radius_sum += radius;
            final_active_max_radius = final_active_max_radius.max(radius);
        }
    }
    let inactive_seed_count = trace.particle_count.saturating_sub(active_seed_count);
    Growth3dActivationReport {
        active_seed_count,
        inactive_seed_count,
        final_active_count,
        newly_activated_count,
        newly_activated_fraction: newly_activated_count as f32 / inactive_seed_count.max(1) as f32,
        final_active_mean_radius: final_active_radius_sum / final_active_count.max(1) as f32,
        final_active_max_radius,
    }
}

pub(crate) fn growth_3d_opacity_stats(states: &[f32], state_dims: usize) -> Growth3dOpacityStats {
    growth_3d_channel_opacity_stats(states, state_dims, GROWTH_3D_LIVENESS_CHANNEL)
}

pub(crate) fn growth_3d_material_opacity_stats(
    states: &[f32],
    state_dims: usize,
) -> Growth3dOpacityStats {
    let Some(channel) = growth_3d_material_opacity_channel(state_dims) else {
        return growth_3d_channel_opacity_stats(states, state_dims, GROWTH_3D_LIVENESS_CHANNEL);
    };
    growth_3d_channel_opacity_stats(states, state_dims, channel)
}

pub(crate) fn growth_3d_material_liveness_report(
    states: &[f32],
    state_dims: usize,
) -> Growth3dMaterialLivenessReport {
    let threshold = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT + 1.0;
    let Some(material_channel) = growth_3d_material_opacity_channel(state_dims) else {
        return Growth3dMaterialLivenessReport {
            material_visible_count: 0,
            inactive_material_visible_count: 0,
            inactive_material_visible_fraction: 0.0,
            inactive_material_logit_threshold: threshold,
            max_inactive_material_opacity: f32::NEG_INFINITY,
            passed: true,
        };
    };
    if state_dims <= GROWTH_3D_LIVENESS_CHANNEL || states.is_empty() {
        return Growth3dMaterialLivenessReport {
            material_visible_count: 0,
            inactive_material_visible_count: 0,
            inactive_material_visible_fraction: 0.0,
            inactive_material_logit_threshold: threshold,
            max_inactive_material_opacity: f32::NEG_INFINITY,
            passed: true,
        };
    }

    let mut material_visible_count = 0usize;
    let mut inactive_material_visible_count = 0usize;
    let mut max_inactive_material_opacity = f32::NEG_INFINITY;
    for state in states.chunks_exact(state_dims) {
        let material_opacity = state[material_channel];
        let liveness = state[GROWTH_3D_LIVENESS_CHANNEL];
        if material_opacity > threshold {
            material_visible_count += 1;
            if liveness <= -1.0 {
                inactive_material_visible_count += 1;
                max_inactive_material_opacity = max_inactive_material_opacity.max(material_opacity);
            }
        }
    }
    let inactive_material_visible_fraction =
        inactive_material_visible_count as f32 / material_visible_count.max(1) as f32;
    Growth3dMaterialLivenessReport {
        material_visible_count,
        inactive_material_visible_count,
        inactive_material_visible_fraction,
        inactive_material_logit_threshold: threshold,
        max_inactive_material_opacity,
        passed: inactive_material_visible_count == 0,
    }
}

pub(crate) fn apply_material_liveness_strict_check(
    checks: &mut Growth3dStrictChecksReport,
    material_liveness: Growth3dMaterialLivenessReport,
) {
    checks.material_visible_particles_live = material_liveness.passed;
    if !material_liveness.passed
        && !checks
            .failure_reasons
            .contains(&"material_visible_particles_live")
    {
        checks
            .failure_reasons
            .push("material_visible_particles_live");
    }
    checks.passed = checks.failure_reasons.is_empty();
}

pub(crate) fn apply_material_liveness_strict_score(
    score: &mut Growth3dStrictScoreReport,
    material_liveness: Growth3dMaterialLivenessReport,
) {
    let inactive_fraction_penalty = material_liveness.inactive_material_visible_fraction * 10.0;
    let max_inactive_opacity = material_liveness.max_inactive_material_opacity;
    let max_inactive_opacity_penalty = if max_inactive_opacity.is_finite() {
        ((max_inactive_opacity - material_liveness.inactive_material_logit_threshold).max(0.0))
            / 10.0
    } else {
        0.0
    };
    score.material_visible_inactive_fraction = material_liveness.inactive_material_visible_fraction;
    score.material_visible_inactive_fraction_penalty = inactive_fraction_penalty;
    score.material_visible_max_inactive_opacity = max_inactive_opacity;
    score.material_visible_max_inactive_opacity_penalty = max_inactive_opacity_penalty;
    score.score += inactive_fraction_penalty + max_inactive_opacity_penalty;
}

pub(crate) fn apply_temporal_activation_strict_score(
    score: &mut Growth3dStrictScoreReport,
    temporal: &Growth3dTemporalReport,
    rollout_steps: usize,
) {
    let schedule_error = temporal_activation_schedule_error(temporal, rollout_steps);
    let penalty = schedule_error * TEMPORAL_ACTIVATION_SCORE_WEIGHT;
    score.temporal_activation_schedule_error = schedule_error;
    score.temporal_activation_schedule_penalty = penalty;
    score.score += penalty;
}

pub(crate) fn apply_material_visible_surface_tail_strict_check(
    checks: &mut Growth3dStrictChecksReport,
    material_visible_surface_tail: Growth3dSurfaceTailReport,
) {
    let passed = material_visible_surface_tail.p99_distance < GROWTH_3D_SURFACE_MAX_DISTANCE
        && material_visible_surface_tail.over_threshold_fraction <= 0.005
        && material_visible_surface_tail.opacity_weighted_over_threshold_fraction <= 0.005;
    checks.material_visible_surface_tail_bounded = passed;
    if !passed
        && !checks
            .failure_reasons
            .contains(&"material_visible_surface_tail_bounded")
    {
        checks
            .failure_reasons
            .push("material_visible_surface_tail_bounded");
    }
    checks.passed = checks.failure_reasons.is_empty();
}

pub(crate) fn apply_material_visible_surface_tail_strict_score(
    score: &mut Growth3dStrictScoreReport,
    material_visible_surface_tail: Growth3dSurfaceTailReport,
) {
    let p99_penalty =
        (material_visible_surface_tail.p99_distance - GROWTH_3D_SURFACE_MAX_DISTANCE).max(0.0);
    let fraction_penalty = ((material_visible_surface_tail.over_threshold_fraction - 0.005)
        .max(0.0)
        + (material_visible_surface_tail.opacity_weighted_over_threshold_fraction - 0.005)
            .max(0.0))
        * 10.0;
    score.material_visible_surface_tail_p99_distance = material_visible_surface_tail.p99_distance;
    score.material_visible_surface_tail_p99_penalty = p99_penalty;
    score.material_visible_surface_tail_over_threshold_fraction =
        material_visible_surface_tail.over_threshold_fraction;
    score.material_visible_surface_tail_fraction_penalty = fraction_penalty;
    score.score += p99_penalty + fraction_penalty;
}

pub(crate) fn apply_surface_profile_strict_check(
    checks: &mut Growth3dStrictChecksReport,
    active_profile: &SurfaceCoverageProfileReport,
    material_visible_profile: &SurfaceCoverageProfileReport,
) {
    let active_passed = surface_profile_passes_strict_coverage(active_profile);
    let material_visible_passed = surface_profile_passes_strict_coverage(material_visible_profile);
    checks.surface_coverage_profile = active_passed;
    checks.material_visible_surface_coverage_profile = material_visible_passed;
    if !active_passed && !checks.failure_reasons.contains(&"surface_coverage_profile") {
        checks.failure_reasons.push("surface_coverage_profile");
    }
    if !material_visible_passed
        && !checks
            .failure_reasons
            .contains(&"material_visible_surface_coverage_profile")
    {
        checks
            .failure_reasons
            .push("material_visible_surface_coverage_profile");
    }
    checks.passed = checks.failure_reasons.is_empty();
}

pub(crate) fn surface_profile_passes_strict_coverage(
    profile: &SurfaceCoverageProfileReport,
) -> bool {
    profile.covered_bin_fraction >= GROWTH_3D_MIN_SURFACE_PROFILE_BIN_FRACTION
        && profile.mean_bin_covered_fraction >= GROWTH_3D_MIN_SURFACE_PROFILE_MEAN_BIN_COVERAGE
}

pub(crate) fn apply_surface_profile_strict_score(
    score: &mut Growth3dStrictScoreReport,
    active_profile: &SurfaceCoverageProfileReport,
    material_visible_profile: &SurfaceCoverageProfileReport,
) {
    let active_bin_penalty =
        (GROWTH_3D_MIN_SURFACE_PROFILE_BIN_FRACTION - active_profile.covered_bin_fraction).max(0.0);
    let active_mean_penalty = (GROWTH_3D_MIN_SURFACE_PROFILE_MEAN_BIN_COVERAGE
        - active_profile.mean_bin_covered_fraction)
        .max(0.0);
    let material_bin_penalty = (GROWTH_3D_MIN_SURFACE_PROFILE_BIN_FRACTION
        - material_visible_profile.covered_bin_fraction)
        .max(0.0);
    let material_mean_penalty = (GROWTH_3D_MIN_SURFACE_PROFILE_MEAN_BIN_COVERAGE
        - material_visible_profile.mean_bin_covered_fraction)
        .max(0.0);
    score.surface_covered_bin_fraction = active_profile.covered_bin_fraction;
    score.surface_bin_penalty = active_bin_penalty;
    score.surface_mean_bin_covered_fraction = active_profile.mean_bin_covered_fraction;
    score.surface_coverage_mean_penalty = active_mean_penalty;
    score.material_visible_surface_covered_bin_fraction =
        material_visible_profile.covered_bin_fraction;
    score.material_visible_surface_bin_penalty = material_bin_penalty;
    score.material_visible_surface_mean_bin_covered_fraction =
        material_visible_profile.mean_bin_covered_fraction;
    score.material_visible_surface_mean_penalty = material_mean_penalty;
    score.score +=
        active_bin_penalty + active_mean_penalty + material_bin_penalty + material_mean_penalty;
}
