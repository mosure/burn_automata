#![allow(clippy::too_many_arguments)]

use super::*;

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
