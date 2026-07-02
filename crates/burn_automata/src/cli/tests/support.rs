use super::*;

pub(super) fn growth_validation_test_config(seed_mode: ParticleSeed) -> Growth3dValidationConfig {
    Growth3dValidationConfig {
        particle_count: 256,
        steps: 4,
        seed: 42,
        extra_seeds: Vec::new(),
        seed_scale: UV_TORUS_FIELD_SCALE,
        seed_mode,
        gate: Growth3dValidationGateArg::Strict,
        render: RenderLossConfig {
            image_size: 8,
            target_samples: 64,
            world_scale: UV_TORUS_FIELD_SCALE * 2.0,
            ..RenderLossConfig::default()
        },
    }
}

pub(super) fn synthetic_render_loss(
    total_loss: f32,
    density_psnr_db: f32,
    color_psnr_db: f32,
    depth_psnr_db: f32,
) -> MultiViewRenderLossReport {
    MultiViewRenderLossReport {
        passed: false,
        image_size: 48,
        target_samples: 1024,
        total_loss,
        density_mse: 0.0,
        color_mse: 0.0,
        depth_mse: 0.0,
        density_psnr_db,
        color_psnr_db,
        depth_psnr_db,
        nonzero_target_alpha_fraction: 1.0,
        nonzero_particle_alpha_fraction: 1.0,
        views: Vec::new(),
    }
}

pub(super) fn direct_rollout_test_config() -> RenderProxyTrainingConfig {
    RenderProxyTrainingConfig {
        target: MeshTargetArg::Torus,
        rounds: 1,
        supervised_steps_per_round: 1,
        particles: 16,
        rollout_steps: 1,
        gradient_particles: 4,
        gradient_mode: RenderGradientModeArg::Analytic,
        finite_diff_eps: 1.0e-3,
        motion_gain: 1.0,
        perception_position_gain: 0.05,
        max_update_norm: 0.05,
        trajectory_supervision: true,
        trajectory_render_gain: 0.0,
        trajectory_mesh_gain: 0.0,
        trajectory_render_samples: 0,
        liveness_gain: 0.0,
        liveness_front_radius: ROBUST_3D_LIVENESS_FRONT_RADIUS,
        liveness_update_multiplier: ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER,
        coverage_gain: 0.0,
        coverage_samples: 16,
        coverage_mode: CoverageUpdateModeArg::HardNearest,
        coverage_softness: 0.0,
        coverage_repulsion_gain: 0.0,
        coverage_gap_gain: 0.0,
        coverage_repulsion_radius: 0.0,
        coverage_normal_weight: 0.0,
        extent_gain: 0.0,
        full_coverage_adjoint: false,
        surface_gain: 0.0,
        surface_escape_gain: 0.0,
        opacity_gain: 0.0,
        material_liveness_gain: 0.0,
        material_tail_gain: 0.0,
        material_suppression_update_multiplier: ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
        material_max_opacity_update: ROBUST_3D_MATERIAL_MAX_OPACITY_UPDATE,
        scale_gain: 0.0,
        scale_budget_weight: 0.0,
        max_opacity_update: 0.05,
        direct_output_gradient_rms_cap: ROBUST_3D_DIRECT_OUTPUT_GRADIENT_RMS_CAP,
        direct_line_search: false,
        direct_line_search_scales: vec![1.0],
        direct_material_output_only: false,
        training_backend: RenderTrainingBackendArg::DirectRollout,
        direct_selection_seed_training: false,
        seed: 13,
        selection_seed: None,
        selection_seeds: Vec::new(),
        seed_scale: UV_TORUS_RENDER_TRAINING_SCALE,
        seed_mode: ParticleSeed::TorusGrowth3d,
        render: RenderLossConfig {
            image_size: 8,
            target_samples: 32,
            world_scale: UV_TORUS_RENDER_TRAINING_SCALE * 2.0,
            ..RenderLossConfig::default()
        },
        sgd: SgdConfig {
            learning_rate: 5.0e-4,
            grad_clip_norm: 1.0,
            weight_decay: 0.0,
        },
    }
}

pub(super) fn render_selection_metrics_with_liveness(
    score: f32,
    render_loss: f32,
    density_psnr_db: f32,
    front_liveness_margin: f32,
) -> RenderSelectionMetrics {
    RenderSelectionMetrics {
        render_loss,
        max_render_loss: render_loss,
        score,
        density_psnr_db,
        min_density_psnr_db: density_psnr_db,
        active_surface_max: 0.12,
        target_coverage_fraction: 0.82,
        material_visible_target_mean_distance: 0.05,
        material_visible_target_max_distance: 0.12,
        material_visible_target_coverage_fraction: 0.82,
        strict_surface_active_count: 8,
        strict_surface_materialized_fraction: 1.0,
        strict_surface_material_mean_opacity: 1.0,
        strict_surface_material_visible_margin: 0.0,
        strict_surface_material_max_visible_margin: 0.0,
        material_visible_inactive_fraction: 0.0,
        material_visible_max_inactive_opacity: f32::NEG_INFINITY,
        material_active_mean_opacity: 0.0,
        material_visible_count: 1,
        active_color_state_mean_abs: 0.05,
        active_color_state_max_abs: 0.08,
        active_color_state_stddev_mean: 0.05,
        surface_covered_bin_fraction: 0.75,
        surface_mean_bin_covered_fraction: 0.70,
        material_visible_surface_covered_bin_fraction: 0.75,
        material_visible_surface_mean_bin_covered_fraction: 0.70,
        surface_normal_covered_bin_fraction: 0.75,
        surface_normal_mean_bin_covered_fraction: 0.70,
        material_visible_surface_normal_covered_bin_fraction: 0.75,
        material_visible_surface_normal_mean_bin_covered_fraction: 0.70,
        material_visible_surface_tail_p99_distance: 0.12,
        material_visible_surface_tail_over_threshold_fraction: 0.0,
        max_dormant_drift_fraction: 0.0,
        max_dormant_drift: 0.02,
        all_dormant_drift_bounded: true,
        min_active_extent_bbox_ratio: 0.35,
        min_active_extent_min_axis_ratio: 0.15,
        min_final_active_count: 1,
        min_newly_activated_fraction: 0.0,
        min_front_local_newly_activated_fraction: 0.0,
        max_front_liveness_margin: front_liveness_margin,
        min_front_liveness_candidate_count: 31,
        max_extent_front_liveness_margin: 0.0,
        min_extent_front_liveness_candidate_count: 0,
        max_temporal_front_liveness_margin: 0.0,
        min_temporal_front_liveness_candidate_count: 0,
        max_temporal_extent_front_liveness_margin: 0.0,
        min_temporal_extent_front_liveness_candidate_count: 0,
        max_temporal_activation_schedule_error: 0.0,
        all_temporal_activation_progressive: false,
        all_temporal_geometry_progressive: false,
        morphology_non_regressed: true,
        worst_seed: 0,
        worst_failure_reasons: Vec::new(),
        #[cfg(test)]
        base_report: synthetic_render_loss(render_loss, density_psnr_db, 20.0, 20.0),
    }
}

pub(super) fn set_render_selection_metrics_render(
    metrics: &mut RenderSelectionMetrics,
    render_loss: f32,
    density_psnr_db: f32,
) {
    metrics.render_loss = render_loss;
    metrics.max_render_loss = render_loss;
    metrics.density_psnr_db = density_psnr_db;
    metrics.min_density_psnr_db = density_psnr_db;
    metrics.base_report = synthetic_render_loss(render_loss, density_psnr_db, 20.0, 20.0);
}

pub(super) fn render_selection_case_with_front_liveness_margin(
    margin: f32,
) -> RenderSelectionCaseMetrics {
    RenderSelectionCaseMetrics {
        render_loss: synthetic_render_loss(1.0, 20.0, 20.0, 20.0),
        active_surface: Growth3dSurfaceStats {
            mean_distance: 0.05,
            max_distance: 0.12,
        },
        target_coverage: TargetCoverageStats {
            mean_distance: 0.05,
            max_distance: 0.12,
            covered_fraction: 0.82,
        },
        material_visible_target_coverage: TargetCoverageStats {
            mean_distance: 0.05,
            max_distance: 0.12,
            covered_fraction: 0.82,
        },
        strict_surface_materialization: Growth3dStrictSurfaceMaterializationReport {
            active_strict_count: 8,
            materialized_count: 8,
            materialized_fraction: 1.0,
            mean_material_opacity: 1.0,
            mean_visible_margin: 0.0,
            max_visible_margin: 0.0,
        },
        material_opacity: passing_growth_3d_opacity_stats(),
        material_liveness: passing_material_liveness_report(),
        final_color_state: emerged_growth_3d_color_state_report(),
        surface_coverage_profile: passing_surface_coverage_profile_report(),
        material_visible_surface_coverage_profile: passing_surface_coverage_profile_report(),
        surface_normal_coverage: passing_surface_normal_coverage_report(),
        material_visible_surface_normal_coverage: passing_surface_normal_coverage_report(),
        material_visible_surface_tail: passing_growth_3d_surface_tail_report(),
        dormant_drift: passing_growth_3d_dormant_drift_report(),
        extent: passing_growth_3d_extent_report(),
        final_active_count: 64,
        newly_activated_fraction: 0.75,
        front_local_newly_activated_fraction: 0.70,
        front_liveness: LocalFrontLivenessProgress {
            candidate_count: 4,
            weighted_activation_margin: margin,
        },
        extent_front_liveness: LocalFrontLivenessProgress::default(),
        temporal_front_liveness: LocalFrontLivenessProgress::default(),
        temporal_extent_front_liveness: LocalFrontLivenessProgress::default(),
        temporal_activation_schedule_error: 0.0,
        temporal_activation_progressive: true,
        temporal_geometry_progressive: true,
        score: 1.0,
        failure_reasons: Vec::new(),
    }
}

pub(super) fn render_selection_baseline_case_from_metrics(
    seed: u64,
    case: &RenderSelectionCaseMetrics,
) -> RenderSelectionBaselineCase {
    RenderSelectionBaselineCase {
        seed,
        active_surface_max: case.active_surface.max_distance,
        target_coverage_fraction: case.target_coverage.covered_fraction,
        material_visible_target_mean_distance: case.material_visible_target_coverage.mean_distance,
        material_visible_target_max_distance: case.material_visible_target_coverage.max_distance,
        material_visible_target_coverage_fraction: case
            .material_visible_target_coverage
            .covered_fraction,
        material_visible_inactive_fraction: case
            .material_liveness
            .inactive_material_visible_fraction,
        material_visible_max_inactive_opacity: case.material_liveness.max_inactive_material_opacity,
        surface_covered_bin_fraction: case.surface_coverage_profile.covered_bin_fraction,
        surface_mean_bin_covered_fraction: case.surface_coverage_profile.mean_bin_covered_fraction,
        material_visible_surface_covered_bin_fraction: case
            .material_visible_surface_coverage_profile
            .covered_bin_fraction,
        material_visible_surface_mean_bin_covered_fraction: case
            .material_visible_surface_coverage_profile
            .mean_bin_covered_fraction,
        surface_normal_covered_bin_fraction: case
            .surface_normal_coverage
            .covered_target_bin_fraction,
        surface_normal_mean_bin_covered_fraction: case
            .surface_normal_coverage
            .mean_bin_covered_fraction,
        material_visible_surface_normal_covered_bin_fraction: case
            .material_visible_surface_normal_coverage
            .covered_target_bin_fraction,
        material_visible_surface_normal_mean_bin_covered_fraction: case
            .material_visible_surface_normal_coverage
            .mean_bin_covered_fraction,
        material_visible_surface_tail_p99_distance: case.material_visible_surface_tail.p99_distance,
        material_visible_surface_tail_over_threshold_fraction: case
            .material_visible_surface_tail
            .over_threshold_fraction,
        dormant_drift_fraction: case.dormant_drift.drifting_fraction,
        max_dormant_drift: case.dormant_drift.max_dormant_displacement,
        active_extent_bbox_ratio: case.extent.bbox_diagonal_ratio,
        active_extent_min_axis_ratio: case.extent.min_axis_extent_ratio,
        final_active_count: case.final_active_count,
        newly_activated_fraction: case.newly_activated_fraction,
        front_local_newly_activated_fraction: case.front_local_newly_activated_fraction,
        front_liveness: case.front_liveness,
        extent_front_liveness: case.extent_front_liveness,
        temporal_front_liveness: case.temporal_front_liveness,
        temporal_extent_front_liveness: case.temporal_extent_front_liveness,
        temporal_activation_schedule_error: case.temporal_activation_schedule_error,
        temporal_activation_progressive: case.temporal_activation_progressive,
        temporal_geometry_progressive: case.temporal_geometry_progressive,
    }
}

pub(super) fn mark_selection_dormant_drift_unbounded(selection: &mut RenderSelectionMetrics) {
    selection.all_dormant_drift_bounded = false;
    selection.max_dormant_drift_fraction = 0.125;
    selection.max_dormant_drift = 0.24;
    selection.morphology_non_regressed = false;
    selection
        .worst_failure_reasons
        .push("dormant_drift_bounded");
}

pub(super) fn robustness_seed_report_with_surface_normal_coverage(
    seed: u64,
    surface_normal_coverage: bool,
    normal_bin_fraction: f32,
    normal_mean_fraction: f32,
) -> Growth3dRobustnessSeedReport {
    Growth3dRobustnessSeedReport {
        seed,
        gate_passed: false,
        strict_passed: false,
        catalog_sanity_passed: false,
        strict_score: 1.0,
        no_seed_coordinate_scaffold: true,
        render_loss: 1.0,
        density_psnr_db: 1.0,
        color_psnr_db: 1.0,
        depth_psnr_db: 1.0,
        active_seed_count: 4,
        final_active_count: 64,
        newly_activated_fraction: 0.9,
        active_extent_growth: true,
        active_extent_bbox_ratio: 0.35,
        active_extent_min_axis_ratio: 0.15,
        final_opacity_max: 1.0,
        material_visible_particles_live: true,
        inactive_material_visible_fraction: 0.0,
        max_inactive_material_opacity: f32::NEG_INFINITY,
        color_state_emerged: true,
        final_active_color_state_mean_abs: 0.1,
        final_active_color_state_stddev_mean: 0.1,
        permutation_consistent: true,
        permutation_max_position_error: 0.0,
        permutation_max_state_error: 0.0,
        gaussian_scale_budget: true,
        gaussian_scale_budget_loss: 0.0,
        gaussian_oversize_fraction: 0.0,
        seed_perturbation_stable: true,
        perturbed_newly_activated_fraction: 0.9,
        perturbed_active_count_ratio: 1.0,
        perturbed_peak_motion_ratio: 1.0,
        local_front_coherent: true,
        front_local_newly_activated_fraction: 0.9,
        front_max_nearest_previous_active_distance: 0.1,
        temporal_activation_progressive: true,
        temporal_geometry_progressive: true,
        final_active_target_coverage_fraction: 0.75,
        final_material_visible_target_coverage_fraction: 0.75,
        surface_coverage_profile: surface_normal_coverage,
        final_active_surface_covered_bin_fraction: normal_bin_fraction,
        final_active_surface_mean_bin_covered_fraction: normal_mean_fraction,
        material_visible_surface_coverage_profile: surface_normal_coverage,
        final_material_visible_surface_covered_bin_fraction: normal_bin_fraction,
        final_material_visible_surface_mean_bin_covered_fraction: normal_mean_fraction,
        surface_normal_coverage,
        final_active_surface_normal_covered_bin_fraction: normal_bin_fraction,
        final_active_surface_normal_mean_bin_covered_fraction: normal_mean_fraction,
        material_visible_surface_normal_coverage: surface_normal_coverage,
        final_material_visible_surface_normal_covered_bin_fraction: normal_bin_fraction,
        final_material_visible_surface_normal_mean_bin_covered_fraction: normal_mean_fraction,
        final_active_surface_max: 0.1,
        material_visible_surface_tail_bounded: true,
        final_material_visible_surface_tail_p99_distance: 0.2,
        final_material_visible_surface_tail_over_threshold_fraction: 0.0,
        final_material_visible_surface_tail_opacity_weighted_over_threshold_fraction: 0.0,
        failure_reasons: Vec::new(),
    }
}

pub(super) fn passing_growth_3d_strict_checks() -> Growth3dStrictChecksReport {
    Growth3dStrictChecksReport {
        passed: true,
        no_position_features: true,
        local_conditionless_lineage: true,
        no_seed_coordinate_scaffold: true,
        neutral_non_opacity_seed_state: true,
        sparse_active_seed: true,
        active_count_growth: true,
        newly_activated_fraction: true,
        active_front_expanded: true,
        active_extent_growth: true,
        nonzero_motion: true,
        sustained_motion: true,
        local_front_coherent: true,
        dormant_drift_bounded: true,
        temporal_activation_progressive: true,
        temporal_geometry_progressive: true,
        mean_displacement_growth: true,
        bounded_final_opacity: true,
        material_visible_particles_live: true,
        color_state_emerged: true,
        permutation_consistent: true,
        surface_mean_improved: true,
        surface_max_bounded: true,
        surface_tail_bounded: true,
        material_visible_surface_tail_bounded: true,
        target_coverage_mean_improved: true,
        target_coverage_max_bounded: true,
        target_coverage_fraction: true,
        material_visible_target_coverage_fraction: true,
        surface_coverage_profile: true,
        material_visible_surface_coverage_profile: true,
        surface_normal_coverage: true,
        material_visible_surface_normal_coverage: true,
        torus_angular_coverage: true,
        gaussian_scale_budget: true,
        render_loss_passed: true,
        failure_reasons: Vec::new(),
    }
}

pub(super) fn passing_growth_3d_dormant_drift_report() -> Growth3dDormantDriftReport {
    Growth3dDormantDriftReport {
        sampled_steps: 4,
        checked_rows: 32,
        drifting_rows: 0,
        drifting_fraction: 0.0,
        mean_dormant_displacement: 0.01,
        max_dormant_displacement: 0.02,
        max_allowed_displacement: 0.10,
        finite: true,
        passed: true,
    }
}

pub(super) fn passing_growth_3d_extent_report() -> Growth3dExtentReport {
    Growth3dExtentReport {
        target_bounds_min: [-1.0, -1.0, -1.0],
        target_bounds_max: [1.0, 1.0, 1.0],
        final_active_bounds_min: [-0.45, -0.35, -0.15],
        final_active_bounds_max: [0.45, 0.35, 0.15],
        target_extent: [2.0, 2.0, 2.0],
        final_active_extent: [0.90, 0.70, 0.30],
        axis_extent_ratio: [0.45, 0.35, 0.15],
        min_axis_extent_ratio: 0.15,
        bbox_diagonal_ratio: 0.34,
        target_max_radius: 1.0,
        final_active_max_radius: 0.50,
        max_radius_ratio: 0.50,
    }
}

pub(super) fn passing_surface_normal_coverage_report() -> SurfaceNormalCoverageReport {
    SurfaceNormalCoverageReport {
        samples: 512,
        normal_bins: 26,
        threshold: 0.12,
        target_bins: 20,
        covered_target_bins: 16,
        covered_target_bin_fraction: 0.80,
        covered_sample_fraction: 0.80,
        min_bin_covered_fraction: 0.25,
        mean_bin_covered_fraction: 0.60,
        max_bin_covered_fraction: 1.0,
        target_bin_sample_fractions: Vec::new(),
        bin_covered_fractions: Vec::new(),
    }
}

pub(super) fn passing_material_liveness_report() -> Growth3dMaterialLivenessReport {
    Growth3dMaterialLivenessReport {
        material_visible_count: 16,
        inactive_material_visible_count: 0,
        inactive_material_visible_fraction: 0.0,
        inactive_material_logit_threshold: GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT + 1.0,
        max_inactive_material_opacity: f32::NEG_INFINITY,
        passed: true,
    }
}

pub(super) fn passing_strict_surface_materialization_report()
-> Growth3dStrictSurfaceMaterializationReport {
    Growth3dStrictSurfaceMaterializationReport {
        active_strict_count: 8,
        materialized_count: 8,
        materialized_fraction: 1.0,
        mean_material_opacity: 1.0,
        mean_visible_margin: 0.0,
        max_visible_margin: 0.0,
    }
}

pub(super) fn passing_surface_coverage_profile_report() -> SurfaceCoverageProfileReport {
    SurfaceCoverageProfileReport {
        samples: 512,
        bins: 64,
        threshold: 0.12,
        covered_fraction: 0.90,
        covered_bin_fraction: 0.90,
        empty_bins: 6,
        min_bin_covered_fraction: 0.20,
        mean_bin_covered_fraction: 0.80,
        max_bin_covered_fraction: 1.0,
        assigned_particle_fraction: 0.75,
        covered_assigned_particle_fraction: 0.70,
        max_assigned_sample_fraction: 0.04,
        max_covered_assigned_sample_fraction: 0.04,
        bin_covered_fractions: Vec::new(),
    }
}

pub(super) fn passing_growth_3d_front_report() -> Growth3dFrontReport {
    Growth3dFrontReport {
        transition_count: 4,
        newly_activated_count: 96,
        local_newly_activated_count: 94,
        local_newly_activated_fraction: 94.0 / 96.0,
        mean_nearest_previous_active_distance: 0.08,
        max_nearest_previous_active_distance: 0.18,
        max_allowed_distance: 0.36,
        finite: true,
        passed: true,
    }
}

pub(super) fn passing_growth_3d_opacity_stats() -> Growth3dOpacityStats {
    Growth3dOpacityStats {
        finite: true,
        min: -1.5,
        max: 1.0,
        mean: 0.0,
        active_min: -0.5,
        active_max: 1.0,
        active_mean: 0.25,
        active_count: 64,
        max_allowed: GROWTH_3D_MAX_FINAL_OPACITY_LOGIT,
    }
}

pub(super) fn neutral_growth_3d_color_state_report() -> Growth3dColorStateReport {
    Growth3dColorStateReport {
        available: true,
        finite: true,
        count: 64,
        active_count: 4,
        mean_abs: 0.0,
        max_abs: 0.0,
        active_mean_abs: 0.0,
        active_max_abs: 0.0,
        active_channel_stddev: [0.0; 3],
        active_channel_stddev_mean: 0.0,
    }
}

pub(super) fn emerged_growth_3d_color_state_report() -> Growth3dColorStateReport {
    Growth3dColorStateReport {
        available: true,
        finite: true,
        count: 64,
        active_count: 64,
        mean_abs: 0.12,
        max_abs: 0.31,
        active_mean_abs: 0.12,
        active_max_abs: 0.31,
        active_channel_stddev: [0.05, 0.04, 0.06],
        active_channel_stddev_mean: 0.05,
    }
}

pub(super) fn passing_growth_3d_permutation_report() -> Growth3dPermutationReport {
    Growth3dPermutationReport {
        particle_count: 128,
        steps: 8,
        max_position_error: 1.0e-6,
        mean_position_error: 1.0e-7,
        max_state_error: 1.0e-6,
        mean_state_error: 1.0e-7,
        passed: true,
    }
}

pub(super) fn passing_growth_3d_surface_tail_report() -> Growth3dSurfaceTailReport {
    Growth3dSurfaceTailReport {
        count: 64,
        threshold: GROWTH_3D_SURFACE_MAX_DISTANCE,
        p95_distance: 0.20,
        p99_distance: 0.30,
        max_distance: 0.30,
        over_threshold_count: 0,
        over_threshold_fraction: 0.0,
        opacity_weighted_mean_distance: 0.12,
        opacity_weighted_over_threshold_fraction: 0.0,
    }
}

pub(super) fn bin_temp_path(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "burn_automata_bin_{}_{}_{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test"),
        name
    ));
    path
}

pub(super) fn torus_angular_sample_position(
    scale: f32,
    ring: usize,
    ring_bins: usize,
    tube: usize,
    tube_bins: usize,
) -> [f32; 4] {
    let major = scale.max(1.0e-4);
    let minor = major * UV_TORUS_MINOR_RATIO;
    let theta = std::f32::consts::TAU * (ring as f32 + 0.5) / ring_bins as f32;
    let phi = std::f32::consts::TAU * (tube as f32 + 0.5) / tube_bins as f32;
    let radial = major + minor * phi.cos();
    [
        radial * theta.cos(),
        radial * theta.sin(),
        minor * phi.sin(),
        1.0,
    ]
}
