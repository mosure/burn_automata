#![allow(clippy::too_many_arguments)]

use super::*;

pub(crate) fn run_rollout_from_state(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    mut positions: Vec<[f32; 4]>,
    mut states: Vec<f32>,
    batch_size: usize,
    particle_count: usize,
    steps: usize,
    dt: f32,
) -> Result<crate::RolloutTrace, Box<dyn std::error::Error>> {
    let mut mean_dx = Vec::with_capacity(steps);
    for _ in 0..steps {
        let step = model.step_cpu(
            &positions,
            &states,
            batch_size,
            particle_count,
            grid,
            dt,
            None,
        )?;
        let dx_norm = step
            .dx
            .iter()
            .map(|delta| (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt())
            .sum::<f32>()
            / step.dx.len().max(1) as f32;
        mean_dx.push(dx_norm);
        positions = step.next_positions;
        states = step.next_states;
    }

    Ok(crate::RolloutTrace {
        positions,
        states,
        batch_size,
        particle_count,
        state_dims: model.config.state_dims,
        steps,
        mean_dx,
    })
}

pub(crate) fn growth_3d_surface_stats(
    positions: &[[f32; 4]],
    target: &TriangleMeshTarget,
) -> Growth3dSurfaceStats {
    let mut max_distance = 0.0_f32;
    let mut sum_distance = 0.0_f32;
    for position in positions {
        let projection = target.project([position[0], position[1], position[2]]);
        max_distance = max_distance.max(projection.distance);
        sum_distance += projection.distance;
    }
    Growth3dSurfaceStats {
        mean_distance: sum_distance / positions.len().max(1) as f32,
        max_distance,
    }
}

pub(crate) fn growth_3d_active_surface_stats(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
) -> Growth3dSurfaceStats {
    let mut max_distance = 0.0_f32;
    let mut sum_distance = 0.0_f32;
    let mut count = 0usize;
    for (idx, position) in positions.iter().enumerate() {
        if state_dims <= 3 || states[idx * state_dims + 3] <= -1.0 {
            continue;
        }
        let projection = target.project([position[0], position[1], position[2]]);
        max_distance = max_distance.max(projection.distance);
        sum_distance += projection.distance;
        count += 1;
    }
    Growth3dSurfaceStats {
        mean_distance: if count > 0 {
            sum_distance / count as f32
        } else {
            f32::INFINITY
        },
        max_distance: if count > 0 {
            max_distance
        } else {
            f32::INFINITY
        },
    }
}

pub(crate) fn growth_3d_active_surface_tail_report(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
    threshold: f32,
) -> Growth3dSurfaceTailReport {
    let mut distances = Vec::new();
    let mut max_distance = 0.0_f32;
    let mut over_threshold_count = 0usize;
    let mut weighted_sum = 0.0_f32;
    let mut weight_sum = 0.0_f32;
    let mut weighted_over_threshold_sum = 0.0_f32;

    if state_dims > 3 {
        for (idx, position) in positions.iter().enumerate() {
            let opacity_logit = states[idx * state_dims + 3];
            if opacity_logit <= -1.0 {
                continue;
            }
            let projection = target.project([position[0], position[1], position[2]]);
            let distance = projection.distance;
            let weight = sigmoid_unit(opacity_logit);
            max_distance = max_distance.max(distance);
            if distance >= threshold {
                over_threshold_count += 1;
                weighted_over_threshold_sum += weight;
            }
            weighted_sum += distance * weight;
            weight_sum += weight;
            distances.push(distance);
        }
    }

    if distances.is_empty() {
        return empty_growth_3d_surface_tail_report(threshold);
    }

    distances.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let count = distances.len();
    Growth3dSurfaceTailReport {
        count,
        threshold,
        p95_distance: percentile_from_sorted(&distances, 0.95),
        p99_distance: percentile_from_sorted(&distances, 0.99),
        max_distance,
        over_threshold_count,
        over_threshold_fraction: over_threshold_count as f32 / count as f32,
        opacity_weighted_mean_distance: weighted_sum / weight_sum.max(1.0e-8),
        opacity_weighted_over_threshold_fraction: weighted_over_threshold_sum
            / weight_sum.max(1.0e-8),
    }
}

pub(crate) fn growth_3d_material_visible_surface_tail_report(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
    threshold: f32,
) -> Growth3dSurfaceTailReport {
    let Some(material_channel) = growth_3d_material_opacity_channel(state_dims) else {
        return empty_growth_3d_surface_tail_report(threshold);
    };
    let mut distances = Vec::new();
    let mut max_distance = 0.0_f32;
    let mut over_threshold_count = 0usize;
    let mut weighted_sum = 0.0_f32;
    let mut weight_sum = 0.0_f32;
    let mut weighted_over_threshold_sum = 0.0_f32;

    for (idx, position) in positions.iter().enumerate() {
        let material_logit = states[idx * state_dims + material_channel];
        if material_logit <= -1.0 {
            continue;
        }
        let projection = target.project([position[0], position[1], position[2]]);
        let distance = projection.distance;
        let weight = sigmoid_unit(material_logit);
        max_distance = max_distance.max(distance);
        if distance >= threshold {
            over_threshold_count += 1;
            weighted_over_threshold_sum += weight;
        }
        weighted_sum += distance * weight;
        weight_sum += weight;
        distances.push(distance);
    }

    if distances.is_empty() {
        return empty_growth_3d_surface_tail_report(threshold);
    }

    distances.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let count = distances.len();
    Growth3dSurfaceTailReport {
        count,
        threshold,
        p95_distance: percentile_from_sorted(&distances, 0.95),
        p99_distance: percentile_from_sorted(&distances, 0.99),
        max_distance,
        over_threshold_count,
        over_threshold_fraction: over_threshold_count as f32 / count as f32,
        opacity_weighted_mean_distance: weighted_sum / weight_sum.max(1.0e-8),
        opacity_weighted_over_threshold_fraction: weighted_over_threshold_sum
            / weight_sum.max(1.0e-8),
    }
}

pub(crate) fn empty_growth_3d_surface_tail_report(threshold: f32) -> Growth3dSurfaceTailReport {
    Growth3dSurfaceTailReport {
        count: 0,
        threshold,
        p95_distance: f32::INFINITY,
        p99_distance: f32::INFINITY,
        max_distance: f32::INFINITY,
        over_threshold_count: 0,
        over_threshold_fraction: 0.0,
        opacity_weighted_mean_distance: f32::INFINITY,
        opacity_weighted_over_threshold_fraction: 0.0,
    }
}

pub(crate) fn percentile_from_sorted(values: &[f32], percentile: f32) -> f32 {
    if values.is_empty() {
        return f32::INFINITY;
    }
    let clamped = percentile.clamp(0.0, 1.0);
    let idx = ((values.len() as f32 * clamped).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1);
    values[idx]
}

pub(crate) fn sigmoid_unit(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

pub(crate) fn sigmoid_unit_derivative(value: f32) -> f32 {
    let sigmoid = sigmoid_unit(value);
    sigmoid * (1.0 - sigmoid)
}

pub(crate) fn growth_3d_mean_displacement(
    initial: &[[f32; 4]],
    final_positions: &[[f32; 4]],
) -> f32 {
    initial
        .iter()
        .zip(final_positions.iter())
        .map(|(a, b)| {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
        })
        .sum::<f32>()
        / initial.len().max(1) as f32
}

pub(crate) fn mesh_rollout_report_for_cases(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cases: &[MeshRolloutCaseConfig],
) -> Result<MeshRolloutReport, Box<dyn std::error::Error>> {
    let mut case_reports = Vec::with_capacity(cases.len());
    let mut max_initial_surface_distance = 0.0_f32;
    let mut sum_mean_initial_surface_distance = 0.0_f32;
    let mut max_surface_distance = 0.0_f32;
    let mut sum_mean_surface_distance = 0.0_f32;
    let mut max_target_coverage_distance = 0.0_f32;
    let mut sum_mean_target_coverage_distance = 0.0_f32;
    let mut min_target_coverage_fraction = 1.0_f32;
    let mut max_color_target_error = 0.0_f32;
    let mut sum_mean_color_target_error = 0.0_f32;
    let mut first_motion_per_step = f32::MAX;
    let mut max_motion_per_step = 0.0_f32;
    let mut max_opacity_target_error = 0.0_f32;
    let mut min_final_opacity = f32::MAX;
    let mut max_final_opacity = f32::MIN;
    let mut passed = true;

    for case in cases {
        let cfg = RolloutConfig {
            particle_count: case.particle_count,
            steps: case.steps,
            update_prob: 1.0,
            seed: case.seed,
            seed_scale: case.seed_scale,
            ..RolloutConfig::default()
        };
        let trace = run_rollout(model, grid, &cfg, case.seed_mode)?;
        let report = mesh_rollout_case_report(&trace, target, *case);
        max_initial_surface_distance =
            max_initial_surface_distance.max(report.max_initial_surface_distance);
        sum_mean_initial_surface_distance += report.mean_initial_surface_distance;
        max_surface_distance = max_surface_distance.max(report.max_surface_distance);
        sum_mean_surface_distance += report.mean_surface_distance;
        max_target_coverage_distance =
            max_target_coverage_distance.max(report.max_target_coverage_distance);
        sum_mean_target_coverage_distance += report.mean_target_coverage_distance;
        min_target_coverage_fraction =
            min_target_coverage_fraction.min(report.target_coverage_fraction);
        max_color_target_error = max_color_target_error.max(report.max_color_target_error);
        sum_mean_color_target_error += report.mean_color_target_error;
        first_motion_per_step = first_motion_per_step.min(report.first_motion_per_step);
        max_motion_per_step = max_motion_per_step.max(report.max_motion_per_step);
        max_opacity_target_error = max_opacity_target_error.max(report.max_opacity_target_error);
        min_final_opacity = min_final_opacity.min(report.min_final_opacity_logit);
        max_final_opacity = max_final_opacity.max(report.max_final_opacity_logit);

        let case_passed = report.finite
            && report.max_initial_surface_distance >= 0.08
            && report.first_motion_per_step >= 1.0e-3
            && report.max_motion_per_step >= 1.0e-3
            && report.mean_surface_improvement_ratio >= 0.15
            && report.max_surface_distance <= 0.36
            && report.mean_surface_distance <= 0.16
            && report.mean_target_coverage_distance <= 0.20
            && report.max_target_coverage_distance <= 0.72
            && report.target_coverage_fraction >= 0.60
            && report.max_color_target_error <= 0.42
            && report.mean_color_target_error <= 0.16
            && report.max_opacity_target_error <= 2.0e-2
            && report.min_final_opacity_logit > UV_TORUS_INITIAL_OPACITY_LOGIT;
        passed &= case_passed;
        case_reports.push(report);
    }

    if first_motion_per_step == f32::MAX {
        first_motion_per_step = 0.0;
    }
    Ok(MeshRolloutReport {
        passed,
        max_initial_surface_distance,
        mean_initial_surface_distance: sum_mean_initial_surface_distance
            / cases.len().max(1) as f32,
        max_surface_distance,
        mean_surface_distance: sum_mean_surface_distance / cases.len().max(1) as f32,
        mean_surface_improvement: sum_mean_initial_surface_distance / cases.len().max(1) as f32
            - sum_mean_surface_distance / cases.len().max(1) as f32,
        mean_surface_improvement_ratio: if sum_mean_initial_surface_distance > 0.0 {
            1.0 - sum_mean_surface_distance / sum_mean_initial_surface_distance
        } else {
            0.0
        },
        max_target_coverage_distance,
        mean_target_coverage_distance: sum_mean_target_coverage_distance
            / cases.len().max(1) as f32,
        min_target_coverage_fraction,
        max_color_target_error,
        mean_color_target_error: sum_mean_color_target_error / cases.len().max(1) as f32,
        first_motion_per_step,
        max_motion_per_step,
        max_opacity_target_error,
        min_final_opacity,
        max_final_opacity,
        cases: case_reports,
    })
}

pub(crate) fn mesh_rollout_case_report(
    trace: &crate::RolloutTrace,
    target: &TriangleMeshTarget,
    case: MeshRolloutCaseConfig,
) -> MeshRolloutCaseReport {
    let (initial_positions, _) = seed_particles_scaled(
        trace.batch_size,
        case.particle_count,
        trace.state_dims,
        3,
        case.seed,
        case.seed_mode,
        case.seed_scale,
    );
    let expected_final_opacity_logit = UV_TORUS_FIELD_OPACITY_TARGET;
    let mut max_initial_surface_distance = 0.0_f32;
    let mut sum_initial_surface_distance = 0.0_f32;
    let mut max_surface_distance = 0.0_f32;
    let mut sum_surface_distance = 0.0_f32;
    let mut max_color_target_error = 0.0_f32;
    let mut sum_color_target_error = 0.0_f32;
    let mut min_final_opacity_logit = f32::MAX;
    let mut max_final_opacity_logit = f32::MIN;
    let mut max_opacity_target_error = 0.0_f32;
    let mut finite = true;

    for (idx, position) in trace.positions.iter().enumerate() {
        finite &= position.iter().all(|value| value.is_finite());
        let initial_position = initial_positions[idx];
        let initial_projection = target.project([
            initial_position[0],
            initial_position[1],
            initial_position[2],
        ]);
        max_initial_surface_distance =
            max_initial_surface_distance.max(initial_projection.distance);
        sum_initial_surface_distance += initial_projection.distance;

        let projection = target.project([position[0], position[1], position[2]]);
        max_surface_distance = max_surface_distance.max(projection.distance);
        sum_surface_distance += projection.distance;

        let state_base = idx * trace.state_dims;
        if trace.state_dims >= 6 {
            let tail = trace.state_dims - 3;
            let rgb = uv_torus_tail_state_to_rgb([
                trace.states[state_base + tail],
                trace.states[state_base + tail + 1],
                trace.states[state_base + tail + 2],
            ]);
            let expected_rgb = projection.color;
            let color_target_error = ((rgb[0] - expected_rgb[0]).powi(2)
                + (rgb[1] - expected_rgb[1]).powi(2)
                + (rgb[2] - expected_rgb[2]).powi(2))
            .sqrt();
            max_color_target_error = max_color_target_error.max(color_target_error);
            sum_color_target_error += color_target_error;
        }

        let opacity = trace.states[state_base + 3];
        finite &= opacity.is_finite();
        min_final_opacity_logit = min_final_opacity_logit.min(opacity);
        max_final_opacity_logit = max_final_opacity_logit.max(opacity);
        max_opacity_target_error =
            max_opacity_target_error.max((opacity - expected_final_opacity_logit).abs());
    }
    finite &= trace.states.iter().all(|value| value.is_finite());
    finite &= trace.mean_dx.iter().all(|value| value.is_finite());
    let mean_initial_surface_distance =
        sum_initial_surface_distance / trace.positions.len().max(1) as f32;
    let mean_surface_distance = sum_surface_distance / trace.positions.len().max(1) as f32;
    let coverage_threshold = target_coverage_threshold(case.seed_scale);
    let coverage = target_coverage_stats(
        &trace.positions,
        target,
        trace.particle_count.max(512),
        coverage_threshold,
    );

    MeshRolloutCaseReport {
        particle_count: case.particle_count,
        steps: case.steps,
        seed: case.seed,
        seed_scale: case.seed_scale,
        seed_mode: case.seed_mode,
        max_initial_surface_distance,
        mean_initial_surface_distance,
        max_surface_distance,
        mean_surface_distance,
        mean_surface_improvement: mean_initial_surface_distance - mean_surface_distance,
        mean_surface_improvement_ratio: if mean_initial_surface_distance > 0.0 {
            1.0 - mean_surface_distance / mean_initial_surface_distance
        } else {
            0.0
        },
        target_coverage_threshold: coverage_threshold,
        max_target_coverage_distance: coverage.max_distance,
        mean_target_coverage_distance: coverage.mean_distance,
        target_coverage_fraction: coverage.covered_fraction,
        max_color_target_error,
        mean_color_target_error: sum_color_target_error / trace.positions.len().max(1) as f32,
        first_motion_per_step: trace.mean_dx.first().copied().unwrap_or_default(),
        max_motion_per_step: trace.mean_dx.iter().copied().fold(0.0, f32::max),
        expected_final_opacity_logit,
        min_final_opacity_logit,
        max_final_opacity_logit,
        max_opacity_target_error,
        finite,
    }
}

pub(crate) fn torus_robustness_report(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
) -> Result<TorusRobustnessReport, Box<dyn std::error::Error>> {
    torus_robustness_report_for_cases(model, grid, TORUS_ROBUSTNESS_CASES)
}

pub(crate) fn torus_robustness_report_for_cases(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    cases: &[TorusRobustnessCaseConfig],
) -> Result<TorusRobustnessReport, Box<dyn std::error::Error>> {
    let opacity_update_index = model.config.spatial_dims + 3;
    let trained_opacity_delta = model.weights.b2[opacity_update_index];
    let field_mode = model.config.position_features;
    let mut case_reports = Vec::with_capacity(cases.len());
    let mut max_target_position_error = 0.0_f32;
    let mut sum_mean_target_position_error = 0.0_f32;
    let mut max_torus_surface_error = 0.0_f32;
    let mut max_color_target_error = 0.0_f32;
    let mut first_motion_per_step = f32::MAX;
    let mut max_motion_per_step = 0.0_f32;
    let mut max_opacity_target_error = 0.0_f32;
    let mut min_final_opacity = f32::MAX;
    let mut max_final_opacity = f32::MIN;
    let mut passed = true;

    for case in cases {
        let cfg = RolloutConfig {
            particle_count: case.particle_count,
            steps: case.steps,
            update_prob: 1.0,
            seed: case.seed,
            seed_scale: case.seed_scale,
            ..RolloutConfig::default()
        };
        let trace = run_rollout(model, grid, &cfg, case.seed_mode)?;
        let report = torus_robustness_case_report(&trace, *case);
        max_target_position_error = max_target_position_error.max(report.max_target_position_error);
        sum_mean_target_position_error += report.mean_target_position_error;
        max_torus_surface_error = max_torus_surface_error.max(report.max_torus_surface_error);
        max_color_target_error = max_color_target_error.max(report.max_color_target_error);
        first_motion_per_step = first_motion_per_step.min(report.first_motion_per_step);
        max_motion_per_step = max_motion_per_step.max(report.max_motion_per_step);
        max_opacity_target_error = max_opacity_target_error.max(report.max_opacity_target_error);
        min_final_opacity = min_final_opacity.min(report.min_final_opacity_logit);
        max_final_opacity = max_final_opacity.max(report.max_final_opacity_logit);
        let case_passed = if field_mode {
            report.finite
                && report.max_initial_target_position_error >= 0.12
                && report.first_motion_per_step >= 1.0e-3
                && report.max_motion_per_step >= 1.0e-3
                && report.max_torus_surface_error <= 1.2e-1
                && report.max_final_radial >= report.torus_outer_radius * 0.80
                && report.max_final_abs_z
                    >= (report.torus_outer_radius - report.torus_inner_radius) * 0.20
                && report.max_color_target_error <= 2.5e-1
                && report.max_opacity_target_error <= 2.0
                && report.min_final_opacity_logit > UV_TORUS_INITIAL_OPACITY_LOGIT
        } else {
            report.finite
                && report.max_initial_target_position_error >= 0.12
                && report.first_motion_per_step >= 1.0e-3
                && report.max_motion_per_step >= 1.0e-3
                && report.max_target_position_error <= 8.0e-2
                && report.max_torus_surface_error <= 8.0e-2
                && report.max_final_radial >= report.torus_outer_radius * 0.80
                && report.max_final_abs_z
                    >= (report.torus_outer_radius - report.torus_inner_radius) * 0.20
                && report.max_color_target_error <= 3.0e-2
                && report.max_opacity_target_error <= 1.0e-2
                && report.min_final_opacity_logit > UV_TORUS_INITIAL_OPACITY_LOGIT
        };
        passed &= case_passed;
        case_reports.push(report);
    }

    if !field_mode {
        passed &= (trained_opacity_delta - UV_TORUS_OPACITY_GROWTH_DELTA).abs() <= 1.0e-3;
    }
    if first_motion_per_step == f32::MAX {
        first_motion_per_step = 0.0;
    }

    Ok(TorusRobustnessReport {
        passed,
        target_opacity_delta: if field_mode {
            UV_TORUS_FIELD_OPACITY_GAIN
        } else {
            UV_TORUS_OPACITY_GROWTH_DELTA
        },
        trained_opacity_delta,
        target_motion_gain: UV_TORUS_MOTION_GAIN,
        target_residual_decay: UV_TORUS_RESIDUAL_DECAY,
        max_target_position_error,
        mean_target_position_error: sum_mean_target_position_error / cases.len().max(1) as f32,
        max_torus_surface_error,
        max_color_target_error,
        first_motion_per_step,
        max_motion_per_step,
        max_opacity_target_error,
        min_final_opacity,
        max_final_opacity,
        cases: case_reports,
    })
}

pub(crate) fn torus_robustness_case_report(
    trace: &crate::RolloutTrace,
    case: TorusRobustnessCaseConfig,
) -> TorusRobustnessCaseReport {
    let major = case.seed_scale.max(1.0e-4);
    let minor = major * UV_TORUS_MINOR_RATIO;
    let field_mode = case.seed_mode == ParticleSeed::TorusFieldDense3d;
    let morphogen_mode = case.seed_mode == ParticleSeed::TorusMorphogenDense3d;
    let target_mesh = if field_mode || morphogen_mode {
        Some(uv_torus_mesh_target(major))
    } else {
        None
    };
    let expected_final_opacity_logit = if field_mode {
        UV_TORUS_FIELD_OPACITY_TARGET
    } else {
        UV_TORUS_INITIAL_OPACITY_LOGIT + UV_TORUS_OPACITY_GROWTH_DELTA * case.steps as f32
    };
    let (initial_positions, _) = seed_particles_scaled(
        trace.batch_size,
        case.particle_count,
        trace.state_dims,
        3,
        case.seed,
        case.seed_mode,
        major,
    );
    let mut max_initial_target_position_error = 0.0_f32;
    let mut sum_initial_target_position_error = 0.0_f32;
    let mut max_target_position_error = 0.0_f32;
    let mut sum_target_position_error = 0.0_f32;
    let mut max_torus_surface_error = 0.0_f32;
    let mut sum_torus_surface_error = 0.0_f32;
    let mut min_final_radial = f32::MAX;
    let mut max_final_radial = f32::MIN;
    let mut max_final_abs_z = 0.0_f32;
    let mut max_color_target_error = 0.0_f32;
    let mut sum_color_target_error = 0.0_f32;
    let mut min_final_opacity_logit = f32::MAX;
    let mut max_final_opacity_logit = f32::MIN;
    let mut max_opacity_target_error = 0.0_f32;
    let mut finite = true;

    for (idx, position) in trace.positions.iter().enumerate() {
        finite &= position.iter().all(|value| value.is_finite());
        let initial_position = initial_positions[idx];
        let indexed_target =
            uv_torus_sample(idx % case.particle_count.max(1), case.particle_count, major).position;
        let initial_target = if field_mode || morphogen_mode {
            target_mesh
                .as_ref()
                .unwrap()
                .project([
                    initial_position[0],
                    initial_position[1],
                    initial_position[2],
                ])
                .closest
        } else {
            indexed_target
        };
        let target = if field_mode {
            target_mesh
                .as_ref()
                .unwrap()
                .project([position[0], position[1], position[2]])
                .closest
        } else if morphogen_mode {
            initial_target
        } else {
            indexed_target
        };
        let initial_target_position_error = ((initial_position[0] - target[0]).powi(2)
            + (initial_position[1] - target[1]).powi(2)
            + (initial_position[2] - target[2]).powi(2))
        .sqrt();
        let initial_target_position_error = if field_mode || morphogen_mode {
            ((initial_position[0] - initial_target[0]).powi(2)
                + (initial_position[1] - initial_target[1]).powi(2)
                + (initial_position[2] - initial_target[2]).powi(2))
            .sqrt()
        } else {
            initial_target_position_error
        };
        max_initial_target_position_error =
            max_initial_target_position_error.max(initial_target_position_error);
        sum_initial_target_position_error += initial_target_position_error;

        let target_position_error = ((position[0] - target[0]).powi(2)
            + (position[1] - target[1]).powi(2)
            + (position[2] - target[2]).powi(2))
        .sqrt();
        max_target_position_error = max_target_position_error.max(target_position_error);
        sum_target_position_error += target_position_error;

        let torus_surface_error =
            uv_torus_surface_error([position[0], position[1], position[2]], major);
        max_torus_surface_error = max_torus_surface_error.max(torus_surface_error);
        sum_torus_surface_error += torus_surface_error;
        let radial = (position[0] * position[0] + position[1] * position[1]).sqrt();
        min_final_radial = min_final_radial.min(radial);
        max_final_radial = max_final_radial.max(radial);
        max_final_abs_z = max_final_abs_z.max(position[2].abs());

        let state_base = idx * trace.state_dims;
        if trace.state_dims >= 6 {
            let tail = trace.state_dims - 3;
            let rgb = uv_torus_tail_state_to_rgb([
                trace.states[state_base + tail],
                trace.states[state_base + tail + 1],
                trace.states[state_base + tail + 2],
            ]);
            let expected_rgb = uv_torus_position_color(target, major);
            let color_target_error = ((rgb[0] - expected_rgb[0]).powi(2)
                + (rgb[1] - expected_rgb[1]).powi(2)
                + (rgb[2] - expected_rgb[2]).powi(2))
            .sqrt();
            max_color_target_error = max_color_target_error.max(color_target_error);
            sum_color_target_error += color_target_error;
        }

        let opacity = trace.states[state_base + 3];
        finite &= opacity.is_finite();
        min_final_opacity_logit = min_final_opacity_logit.min(opacity);
        max_final_opacity_logit = max_final_opacity_logit.max(opacity);
        max_opacity_target_error =
            max_opacity_target_error.max((opacity - expected_final_opacity_logit).abs());
    }
    finite &= trace.states.iter().all(|value| value.is_finite());
    finite &= trace.mean_dx.iter().all(|value| value.is_finite());

    TorusRobustnessCaseReport {
        particle_count: case.particle_count,
        steps: case.steps,
        seed: case.seed,
        seed_scale: case.seed_scale,
        seed_mode: case.seed_mode,
        torus_inner_radius: major - minor,
        torus_outer_radius: major + minor,
        max_initial_target_position_error,
        mean_initial_target_position_error: sum_initial_target_position_error
            / trace.positions.len().max(1) as f32,
        max_target_position_error,
        mean_target_position_error: sum_target_position_error / trace.positions.len().max(1) as f32,
        max_torus_surface_error,
        mean_torus_surface_error: sum_torus_surface_error / trace.positions.len().max(1) as f32,
        min_final_radial,
        max_final_radial,
        max_final_abs_z,
        max_color_target_error,
        mean_color_target_error: sum_color_target_error / trace.positions.len().max(1) as f32,
        first_motion_per_step: trace.mean_dx.first().copied().unwrap_or_default(),
        max_motion_per_step: trace.mean_dx.iter().copied().fold(0.0, f32::max),
        expected_final_opacity_logit,
        min_final_opacity_logit,
        max_final_opacity_logit,
        max_opacity_target_error,
        finite,
    }
}
