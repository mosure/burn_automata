use super::*;

pub(crate) fn growth_3d_robustness_seed_report(
    report: &CliGrowth3dValidationReport,
) -> Growth3dRobustnessSeedReport {
    let (
        torus_angular_joint_coverage_fraction,
        torus_angular_tube_coverage_fraction,
        torus_angular_tube_gap_fraction,
    ) = report
        .torus_angular_coverage
        .as_ref()
        .map_or((1.0, 1.0, 0.0), torus_angular_seed_metrics);

    Growth3dRobustnessSeedReport {
        seed: report.seed,
        gate_passed: report.gate_passed,
        strict_passed: report.strict_passed,
        catalog_sanity_passed: report.catalog_sanity.passed,
        strict_score: report.strict_score.score,
        target_conditionless_lineage: report.strict_checks.target_conditionless_lineage,
        target_growth_seed_mode: report.strict_checks.target_growth_seed_mode,
        no_seed_coordinate_scaffold: report.strict_checks.no_seed_coordinate_scaffold,
        render_loss: report.render_loss.total_loss,
        density_psnr_db: report.render_loss.density_psnr_db,
        color_psnr_db: report.render_loss.color_psnr_db,
        depth_psnr_db: report.render_loss.depth_psnr_db,
        active_seed_count: report.activation.active_seed_count,
        final_active_count: report.activation.final_active_count,
        newly_activated_fraction: report.activation.newly_activated_fraction,
        active_extent_growth: report.strict_checks.active_extent_growth,
        active_extent_bbox_ratio: report.extent.bbox_diagonal_ratio,
        active_extent_min_axis_ratio: report.extent.min_axis_extent_ratio,
        final_opacity_max: report.final_opacity.max,
        material_visible_particles_live: report.final_material_liveness.passed,
        inactive_material_visible_fraction: report
            .final_material_liveness
            .inactive_material_visible_fraction,
        max_inactive_material_opacity: report.final_material_liveness.max_inactive_material_opacity,
        color_state_emerged: report.strict_checks.color_state_emerged,
        final_active_color_state_mean_abs: report.final_color_state.active_mean_abs,
        final_active_color_state_stddev_mean: report.final_color_state.active_channel_stddev_mean,
        permutation_consistent: report.permutation_consistency.passed,
        permutation_max_position_error: report.permutation_consistency.max_position_error,
        permutation_max_state_error: report.permutation_consistency.max_state_error,
        gaussian_scale_budget: report.strict_checks.gaussian_scale_budget,
        gaussian_scale_budget_loss: report.final_gaussian_volume.scale_budget_loss,
        gaussian_oversize_fraction: report.final_gaussian_volume.oversize_fraction,
        seed_perturbation_stable: report.seed_perturbation.passed,
        perturbed_newly_activated_fraction: report
            .seed_perturbation
            .perturbed_newly_activated_fraction,
        perturbed_active_count_ratio: report.seed_perturbation.active_count_ratio,
        perturbed_peak_motion_ratio: report.seed_perturbation.peak_motion_ratio,
        local_front_coherent: report.front.passed,
        front_local_newly_activated_fraction: report.front.local_newly_activated_fraction,
        front_max_nearest_previous_active_distance: report
            .front
            .max_nearest_previous_active_distance,
        temporal_activation_progressive: report.temporal.progressive_activation,
        temporal_geometry_progressive: report.temporal.geometry_progressive,
        final_active_target_coverage_fraction: report.final_active_target_coverage.covered_fraction,
        final_material_visible_target_coverage_fraction: report
            .final_material_visible_target_coverage
            .covered_fraction,
        surface_coverage_profile: report.strict_checks.surface_coverage_profile,
        final_active_surface_covered_bin_fraction: report
            .final_active_surface_coverage_profile
            .covered_bin_fraction,
        final_active_surface_mean_bin_covered_fraction: report
            .final_active_surface_coverage_profile
            .mean_bin_covered_fraction,
        material_visible_surface_coverage_profile: report
            .strict_checks
            .material_visible_surface_coverage_profile,
        final_material_visible_surface_covered_bin_fraction: report
            .final_material_visible_surface_coverage_profile
            .covered_bin_fraction,
        final_material_visible_surface_mean_bin_covered_fraction: report
            .final_material_visible_surface_coverage_profile
            .mean_bin_covered_fraction,
        surface_normal_coverage: report.strict_checks.surface_normal_coverage,
        final_active_surface_normal_covered_bin_fraction: report
            .final_active_surface_normal_coverage
            .covered_target_bin_fraction,
        final_active_surface_normal_mean_bin_covered_fraction: report
            .final_active_surface_normal_coverage
            .mean_bin_covered_fraction,
        material_visible_surface_normal_coverage: report
            .strict_checks
            .material_visible_surface_normal_coverage,
        final_material_visible_surface_normal_covered_bin_fraction: report
            .final_material_visible_surface_normal_coverage
            .covered_target_bin_fraction,
        final_material_visible_surface_normal_mean_bin_covered_fraction: report
            .final_material_visible_surface_normal_coverage
            .mean_bin_covered_fraction,
        torus_angular_joint_coverage_fraction,
        torus_angular_tube_coverage_fraction,
        torus_angular_tube_gap_fraction,
        final_active_surface_max: report.final_active_surface.max_distance,
        material_visible_surface_tail_bounded: report
            .strict_checks
            .material_visible_surface_tail_bounded,
        final_material_visible_surface_tail_p99_distance: report
            .final_material_visible_surface_tail
            .p99_distance,
        final_material_visible_surface_tail_over_threshold_fraction: report
            .final_material_visible_surface_tail
            .over_threshold_fraction,
        final_material_visible_surface_tail_opacity_weighted_over_threshold_fraction: report
            .final_material_visible_surface_tail
            .opacity_weighted_over_threshold_fraction,
        failure_reasons: report.strict_checks.failure_reasons.clone(),
    }
}

pub(crate) fn growth_3d_robustness_report(
    seeds: Vec<Growth3dRobustnessSeedReport>,
) -> Growth3dRobustnessReport {
    let seed_count = seeds.len();
    let all_gate_passed = seed_count > 0 && seeds.iter().all(|seed| seed.gate_passed);
    let all_catalog_sanity_passed =
        seed_count > 0 && seeds.iter().all(|seed| seed.catalog_sanity_passed);
    let all_strict_passed = seed_count > 0 && seeds.iter().all(|seed| seed.strict_passed);
    let all_temporal_activation_progressive = seed_count > 0
        && seeds
            .iter()
            .all(|seed| seed.temporal_activation_progressive);
    let all_temporal_geometry_progressive =
        seed_count > 0 && seeds.iter().all(|seed| seed.temporal_geometry_progressive);
    let all_local_front_coherent =
        seed_count > 0 && seeds.iter().all(|seed| seed.local_front_coherent);
    let all_target_conditionless_lineage =
        seed_count > 0 && seeds.iter().all(|seed| seed.target_conditionless_lineage);
    let all_target_growth_seed_mode =
        seed_count > 0 && seeds.iter().all(|seed| seed.target_growth_seed_mode);
    let all_no_seed_coordinate_scaffold =
        seed_count > 0 && seeds.iter().all(|seed| seed.no_seed_coordinate_scaffold);
    let all_active_extent_growth =
        seed_count > 0 && seeds.iter().all(|seed| seed.active_extent_growth);
    let all_bounded_final_opacity = seed_count > 0
        && seeds
            .iter()
            .all(|seed| seed.final_opacity_max <= GROWTH_3D_MAX_FINAL_OPACITY_LOGIT);
    let all_material_visible_particles_live = seed_count > 0
        && seeds
            .iter()
            .all(|seed| seed.material_visible_particles_live);
    let all_color_state_emerged =
        seed_count > 0 && seeds.iter().all(|seed| seed.color_state_emerged);
    let all_permutation_consistent =
        seed_count > 0 && seeds.iter().all(|seed| seed.permutation_consistent);
    let all_seed_perturbation_stable =
        seed_count > 0 && seeds.iter().all(|seed| seed.seed_perturbation_stable);
    let all_surface_coverage_profile =
        seed_count > 0 && seeds.iter().all(|seed| seed.surface_coverage_profile);
    let all_material_visible_surface_coverage_profile = seed_count > 0
        && seeds
            .iter()
            .all(|seed| seed.material_visible_surface_coverage_profile);
    let all_surface_normal_coverage =
        seed_count > 0 && seeds.iter().all(|seed| seed.surface_normal_coverage);
    let all_material_visible_surface_normal_coverage = seed_count > 0
        && seeds
            .iter()
            .all(|seed| seed.material_visible_surface_normal_coverage);
    let all_material_visible_surface_tail_bounded = seed_count > 0
        && seeds
            .iter()
            .all(|seed| seed.material_visible_surface_tail_bounded);
    let worst_strict_score = seeds
        .iter()
        .map(|seed| seed.strict_score)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_render_loss = seeds
        .iter()
        .map(|seed| seed.render_loss)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_density_psnr_db = seeds
        .iter()
        .map(|seed| seed.density_psnr_db)
        .fold(f32::INFINITY, f32::min);
    let min_color_psnr_db = seeds
        .iter()
        .map(|seed| seed.color_psnr_db)
        .fold(f32::INFINITY, f32::min);
    let min_depth_psnr_db = seeds
        .iter()
        .map(|seed| seed.depth_psnr_db)
        .fold(f32::INFINITY, f32::min);
    let min_active_seed_count = seeds
        .iter()
        .map(|seed| seed.active_seed_count)
        .min()
        .unwrap_or(0);
    let max_active_seed_count = seeds
        .iter()
        .map(|seed| seed.active_seed_count)
        .max()
        .unwrap_or(0);
    let min_final_active_count = seeds
        .iter()
        .map(|seed| seed.final_active_count)
        .min()
        .unwrap_or(0);
    let max_final_active_count = seeds
        .iter()
        .map(|seed| seed.final_active_count)
        .max()
        .unwrap_or(0);
    let min_newly_activated_fraction = seeds
        .iter()
        .map(|seed| seed.newly_activated_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_active_growth_ratio = seeds
        .iter()
        .map(|seed| seed.final_active_count as f32 / seed.active_seed_count.max(1) as f32)
        .fold(f32::INFINITY, f32::min);
    let min_active_extent_bbox_ratio = seeds
        .iter()
        .map(|seed| seed.active_extent_bbox_ratio)
        .fold(f32::INFINITY, f32::min);
    let min_active_extent_min_axis_ratio = seeds
        .iter()
        .map(|seed| seed.active_extent_min_axis_ratio)
        .fold(f32::INFINITY, f32::min);
    let max_final_opacity = seeds
        .iter()
        .map(|seed| seed.final_opacity_max)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_inactive_material_visible_fraction = seeds
        .iter()
        .map(|seed| seed.inactive_material_visible_fraction)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_inactive_material_opacity = seeds
        .iter()
        .map(|seed| seed.max_inactive_material_opacity)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_final_active_color_state_mean_abs = seeds
        .iter()
        .map(|seed| seed.final_active_color_state_mean_abs)
        .fold(f32::INFINITY, f32::min);
    let min_final_active_color_state_stddev_mean = seeds
        .iter()
        .map(|seed| seed.final_active_color_state_stddev_mean)
        .fold(f32::INFINITY, f32::min);
    let max_permutation_position_error = seeds
        .iter()
        .map(|seed| seed.permutation_max_position_error)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_permutation_state_error = seeds
        .iter()
        .map(|seed| seed.permutation_max_state_error)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_gaussian_scale_budget_loss = seeds
        .iter()
        .map(|seed| seed.gaussian_scale_budget_loss)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_gaussian_oversize_fraction = seeds
        .iter()
        .map(|seed| seed.gaussian_oversize_fraction)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_perturbed_newly_activated_fraction = seeds
        .iter()
        .map(|seed| seed.perturbed_newly_activated_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_perturbed_active_count_ratio = seeds
        .iter()
        .map(|seed| seed.perturbed_active_count_ratio)
        .fold(f32::INFINITY, f32::min);
    let max_perturbed_active_count_ratio = seeds
        .iter()
        .map(|seed| seed.perturbed_active_count_ratio)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_perturbed_peak_motion_ratio = seeds
        .iter()
        .map(|seed| seed.perturbed_peak_motion_ratio)
        .fold(f32::INFINITY, f32::min);
    let max_perturbed_peak_motion_ratio = seeds
        .iter()
        .map(|seed| seed.perturbed_peak_motion_ratio)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_front_nearest_previous_active_distance = seeds
        .iter()
        .map(|seed| seed.front_max_nearest_previous_active_distance)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_front_local_newly_activated_fraction = seeds
        .iter()
        .map(|seed| seed.front_local_newly_activated_fraction)
        .fold(f32::INFINITY, f32::min);
    let max_final_material_visible_surface_tail_p99_distance = seeds
        .iter()
        .map(|seed| seed.final_material_visible_surface_tail_p99_distance)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_final_material_visible_surface_tail_over_threshold_fraction = seeds
        .iter()
        .map(|seed| seed.final_material_visible_surface_tail_over_threshold_fraction)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_final_material_visible_surface_tail_opacity_weighted_over_threshold_fraction = seeds
        .iter()
        .map(|seed| {
            seed.final_material_visible_surface_tail_opacity_weighted_over_threshold_fraction
        })
        .fold(f32::NEG_INFINITY, f32::max);
    let min_final_active_target_coverage_fraction = seeds
        .iter()
        .map(|seed| seed.final_active_target_coverage_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_final_material_visible_target_coverage_fraction = seeds
        .iter()
        .map(|seed| seed.final_material_visible_target_coverage_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_final_active_surface_covered_bin_fraction = seeds
        .iter()
        .map(|seed| seed.final_active_surface_covered_bin_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_final_active_surface_mean_bin_covered_fraction = seeds
        .iter()
        .map(|seed| seed.final_active_surface_mean_bin_covered_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_final_material_visible_surface_covered_bin_fraction = seeds
        .iter()
        .map(|seed| seed.final_material_visible_surface_covered_bin_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_final_material_visible_surface_mean_bin_covered_fraction = seeds
        .iter()
        .map(|seed| seed.final_material_visible_surface_mean_bin_covered_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_final_active_surface_normal_covered_bin_fraction = seeds
        .iter()
        .map(|seed| seed.final_active_surface_normal_covered_bin_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_final_active_surface_normal_mean_bin_covered_fraction = seeds
        .iter()
        .map(|seed| seed.final_active_surface_normal_mean_bin_covered_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_final_material_visible_surface_normal_covered_bin_fraction = seeds
        .iter()
        .map(|seed| seed.final_material_visible_surface_normal_covered_bin_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_final_material_visible_surface_normal_mean_bin_covered_fraction = seeds
        .iter()
        .map(|seed| seed.final_material_visible_surface_normal_mean_bin_covered_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_torus_angular_joint_coverage_fraction = seeds
        .iter()
        .map(|seed| seed.torus_angular_joint_coverage_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_torus_angular_tube_coverage_fraction = seeds
        .iter()
        .map(|seed| seed.torus_angular_tube_coverage_fraction)
        .fold(f32::INFINITY, f32::min);
    let max_torus_angular_tube_gap_fraction = seeds
        .iter()
        .map(|seed| seed.torus_angular_tube_gap_fraction)
        .fold(f32::NEG_INFINITY, f32::max);
    Growth3dRobustnessReport {
        seed_count,
        all_gate_passed,
        all_catalog_sanity_passed,
        all_strict_passed,
        all_temporal_activation_progressive,
        all_temporal_geometry_progressive,
        all_local_front_coherent,
        all_target_conditionless_lineage,
        all_target_growth_seed_mode,
        all_no_seed_coordinate_scaffold,
        all_active_extent_growth,
        all_bounded_final_opacity,
        all_material_visible_particles_live,
        all_color_state_emerged,
        all_permutation_consistent,
        all_seed_perturbation_stable,
        all_surface_coverage_profile,
        all_material_visible_surface_coverage_profile,
        all_surface_normal_coverage,
        all_material_visible_surface_normal_coverage,
        all_material_visible_surface_tail_bounded,
        worst_strict_score: if seed_count == 0 {
            f32::INFINITY
        } else {
            worst_strict_score
        },
        max_render_loss: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_render_loss
        },
        min_density_psnr_db: if seed_count == 0 {
            f32::NEG_INFINITY
        } else {
            min_density_psnr_db
        },
        min_color_psnr_db: if seed_count == 0 {
            f32::NEG_INFINITY
        } else {
            min_color_psnr_db
        },
        min_depth_psnr_db: if seed_count == 0 {
            f32::NEG_INFINITY
        } else {
            min_depth_psnr_db
        },
        min_active_seed_count,
        max_active_seed_count,
        min_final_active_count,
        max_final_active_count,
        min_newly_activated_fraction: if seed_count == 0 {
            0.0
        } else {
            min_newly_activated_fraction
        },
        min_active_growth_ratio: if seed_count == 0 {
            0.0
        } else {
            min_active_growth_ratio
        },
        min_active_extent_bbox_ratio: if seed_count == 0 {
            0.0
        } else {
            min_active_extent_bbox_ratio
        },
        min_active_extent_min_axis_ratio: if seed_count == 0 {
            0.0
        } else {
            min_active_extent_min_axis_ratio
        },
        max_final_opacity: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_final_opacity
        },
        max_inactive_material_visible_fraction: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_inactive_material_visible_fraction
        },
        max_inactive_material_opacity: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_inactive_material_opacity
        },
        min_final_active_color_state_mean_abs: if seed_count == 0 {
            f32::NAN
        } else {
            min_final_active_color_state_mean_abs
        },
        min_final_active_color_state_stddev_mean: if seed_count == 0 {
            f32::NAN
        } else {
            min_final_active_color_state_stddev_mean
        },
        max_permutation_position_error: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_permutation_position_error
        },
        max_permutation_state_error: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_permutation_state_error
        },
        max_gaussian_scale_budget_loss: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_gaussian_scale_budget_loss
        },
        max_gaussian_oversize_fraction: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_gaussian_oversize_fraction
        },
        min_perturbed_newly_activated_fraction: if seed_count == 0 {
            0.0
        } else {
            min_perturbed_newly_activated_fraction
        },
        min_perturbed_active_count_ratio: if seed_count == 0 {
            0.0
        } else {
            min_perturbed_active_count_ratio
        },
        max_perturbed_active_count_ratio: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_perturbed_active_count_ratio
        },
        min_perturbed_peak_motion_ratio: if seed_count == 0 {
            0.0
        } else {
            min_perturbed_peak_motion_ratio
        },
        max_perturbed_peak_motion_ratio: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_perturbed_peak_motion_ratio
        },
        max_front_nearest_previous_active_distance: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_front_nearest_previous_active_distance
        },
        min_front_local_newly_activated_fraction: if seed_count == 0 {
            0.0
        } else {
            min_front_local_newly_activated_fraction
        },
        max_final_material_visible_surface_tail_p99_distance: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_final_material_visible_surface_tail_p99_distance
        },
        max_final_material_visible_surface_tail_over_threshold_fraction: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_final_material_visible_surface_tail_over_threshold_fraction
        },
        max_final_material_visible_surface_tail_opacity_weighted_over_threshold_fraction:
            if seed_count == 0 {
                f32::INFINITY
            } else {
                max_final_material_visible_surface_tail_opacity_weighted_over_threshold_fraction
            },
        min_final_active_target_coverage_fraction: if seed_count == 0 {
            0.0
        } else {
            min_final_active_target_coverage_fraction
        },
        min_final_material_visible_target_coverage_fraction: if seed_count == 0 {
            0.0
        } else {
            min_final_material_visible_target_coverage_fraction
        },
        min_final_active_surface_covered_bin_fraction: if seed_count == 0 {
            0.0
        } else {
            min_final_active_surface_covered_bin_fraction
        },
        min_final_active_surface_mean_bin_covered_fraction: if seed_count == 0 {
            0.0
        } else {
            min_final_active_surface_mean_bin_covered_fraction
        },
        min_final_material_visible_surface_covered_bin_fraction: if seed_count == 0 {
            0.0
        } else {
            min_final_material_visible_surface_covered_bin_fraction
        },
        min_final_material_visible_surface_mean_bin_covered_fraction: if seed_count == 0 {
            0.0
        } else {
            min_final_material_visible_surface_mean_bin_covered_fraction
        },
        min_final_active_surface_normal_covered_bin_fraction: if seed_count == 0 {
            0.0
        } else {
            min_final_active_surface_normal_covered_bin_fraction
        },
        min_final_active_surface_normal_mean_bin_covered_fraction: if seed_count == 0 {
            0.0
        } else {
            min_final_active_surface_normal_mean_bin_covered_fraction
        },
        min_final_material_visible_surface_normal_covered_bin_fraction: if seed_count == 0 {
            0.0
        } else {
            min_final_material_visible_surface_normal_covered_bin_fraction
        },
        min_final_material_visible_surface_normal_mean_bin_covered_fraction: if seed_count == 0 {
            0.0
        } else {
            min_final_material_visible_surface_normal_mean_bin_covered_fraction
        },
        min_torus_angular_joint_coverage_fraction: if seed_count == 0 {
            1.0
        } else {
            min_torus_angular_joint_coverage_fraction
        },
        min_torus_angular_tube_coverage_fraction: if seed_count == 0 {
            1.0
        } else {
            min_torus_angular_tube_coverage_fraction
        },
        max_torus_angular_tube_gap_fraction: if seed_count == 0 {
            0.0
        } else {
            max_torus_angular_tube_gap_fraction
        },
        seeds,
    }
}

fn torus_angular_seed_metrics(coverage: &TorusAngularCoverageReport) -> (f32, f32, f32) {
    let tube_gap_fraction = if coverage.tube_bins > 0 {
        coverage.max_tube_gap_bins as f32 / coverage.tube_bins as f32
    } else {
        1.0
    };
    (
        coverage.joint_coverage_fraction,
        coverage.tube_coverage_fraction,
        tube_gap_fraction,
    )
}
