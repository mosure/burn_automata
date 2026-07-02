#![allow(clippy::too_many_arguments)]

use super::*;

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
