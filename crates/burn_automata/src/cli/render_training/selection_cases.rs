use super::selection::{RenderSelectionBaselineCase, finite_report_metric};
use super::*;

pub(crate) fn render_selection_baseline(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: &RenderProxyTrainingConfig,
    render_cfg: RenderLossConfig,
) -> Result<Vec<RenderSelectionBaselineCase>, Box<dyn std::error::Error>> {
    let selection_seeds = render_proxy_selection_seeds(cfg);
    let mut baselines = Vec::with_capacity(selection_seeds.len());
    for seed in selection_seeds {
        let selection_case =
            render_selection_case_metrics(model, grid, target, cfg, render_cfg, seed)?;
        baselines.push(RenderSelectionBaselineCase {
            seed,
            active_surface_max: selection_case.active_surface.max_distance,
            target_coverage_fraction: selection_case.target_coverage.covered_fraction,
            material_visible_target_mean_distance: selection_case
                .material_visible_target_coverage
                .mean_distance,
            material_visible_target_max_distance: selection_case
                .material_visible_target_coverage
                .max_distance,
            material_visible_target_coverage_fraction: selection_case
                .material_visible_target_coverage
                .covered_fraction,
            material_visible_inactive_fraction: selection_case
                .material_liveness
                .inactive_material_visible_fraction,
            material_visible_max_inactive_opacity: selection_case
                .material_liveness
                .max_inactive_material_opacity,
            surface_covered_bin_fraction: selection_case
                .surface_coverage_profile
                .covered_bin_fraction,
            surface_mean_bin_covered_fraction: selection_case
                .surface_coverage_profile
                .mean_bin_covered_fraction,
            material_visible_surface_covered_bin_fraction: selection_case
                .material_visible_surface_coverage_profile
                .covered_bin_fraction,
            material_visible_surface_mean_bin_covered_fraction: selection_case
                .material_visible_surface_coverage_profile
                .mean_bin_covered_fraction,
            surface_normal_covered_bin_fraction: selection_case
                .surface_normal_coverage
                .covered_target_bin_fraction,
            surface_normal_mean_bin_covered_fraction: selection_case
                .surface_normal_coverage
                .mean_bin_covered_fraction,
            material_visible_surface_normal_covered_bin_fraction: selection_case
                .material_visible_surface_normal_coverage
                .covered_target_bin_fraction,
            material_visible_surface_normal_mean_bin_covered_fraction: selection_case
                .material_visible_surface_normal_coverage
                .mean_bin_covered_fraction,
            material_visible_surface_tail_p99_distance: selection_case
                .material_visible_surface_tail
                .p99_distance,
            material_visible_surface_tail_over_threshold_fraction: selection_case
                .material_visible_surface_tail
                .over_threshold_fraction,
            active_extent_bbox_ratio: selection_case.extent.bbox_diagonal_ratio,
            active_extent_min_axis_ratio: selection_case.extent.min_axis_extent_ratio,
            final_active_count: selection_case.final_active_count,
            newly_activated_fraction: selection_case.newly_activated_fraction,
            front_local_newly_activated_fraction: selection_case
                .front_local_newly_activated_fraction,
            front_liveness: selection_case.front_liveness,
            extent_front_liveness: selection_case.extent_front_liveness,
            temporal_front_liveness: selection_case.temporal_front_liveness,
            temporal_extent_front_liveness: selection_case.temporal_extent_front_liveness,
            temporal_activation_schedule_error: selection_case.temporal_activation_schedule_error,
            temporal_activation_progressive: selection_case.temporal_activation_progressive,
            temporal_geometry_progressive: selection_case.temporal_geometry_progressive,
        });
    }
    Ok(baselines)
}

pub(crate) fn render_selection_case_score_with_baseline(
    seed: u64,
    case: &RenderSelectionCaseMetrics,
    baseline: Option<&[RenderSelectionBaselineCase]>,
) -> RenderSelectionCaseScore {
    let front_liveness_penalty = finite_report_metric(
        case.front_liveness.weighted_activation_margin,
        RENDER_SELECTION_BAD_SCORE,
    )
    .clamp(0.0, RENDER_SELECTION_BAD_SCORE)
        * LOCAL_FRONT_LIVENESS_SCORE_WEIGHT;
    let temporal_front_liveness_penalty = finite_report_metric(
        case.temporal_front_liveness.weighted_activation_margin,
        RENDER_SELECTION_BAD_SCORE,
    )
    .clamp(0.0, RENDER_SELECTION_BAD_SCORE)
        * LOCAL_FRONT_LIVENESS_SCORE_WEIGHT;
    let temporal_extent_front_liveness_penalty = finite_report_metric(
        case.temporal_extent_front_liveness
            .weighted_activation_margin,
        RENDER_SELECTION_BAD_SCORE,
    )
    .clamp(0.0, RENDER_SELECTION_BAD_SCORE)
        * LOCAL_FRONT_LIVENESS_SCORE_WEIGHT;
    let extent_front_liveness_penalty = finite_report_metric(
        case.extent_front_liveness.weighted_activation_margin,
        RENDER_SELECTION_BAD_SCORE,
    )
    .clamp(0.0, RENDER_SELECTION_BAD_SCORE)
        * LOCAL_FRONT_LIVENESS_SCORE_WEIGHT;
    let temporal_activation_penalty = finite_report_metric(
        case.temporal_activation_schedule_error,
        RENDER_SELECTION_BAD_SCORE,
    )
    .clamp(0.0, RENDER_SELECTION_BAD_SCORE)
        * TEMPORAL_ACTIVATION_SCORE_WEIGHT;
    let material_visible_target_mean_distance_penalty = finite_report_metric(
        case.material_visible_target_coverage.mean_distance,
        RENDER_SELECTION_BAD_SCORE,
    )
    .clamp(0.0, RENDER_SELECTION_BAD_SCORE)
        * MATERIAL_VISIBLE_TARGET_MEAN_DISTANCE_SCORE_WEIGHT;
    let material_visible_target_max_distance_penalty = finite_report_metric(
        case.material_visible_target_coverage.max_distance,
        RENDER_SELECTION_BAD_SCORE,
    )
    .clamp(0.0, RENDER_SELECTION_BAD_SCORE)
        * MATERIAL_VISIBLE_TARGET_MAX_DISTANCE_SCORE_WEIGHT;
    let mut score = finite_report_metric(case.score, RENDER_SELECTION_BAD_SCORE)
        + front_liveness_penalty
        + extent_front_liveness_penalty
        + temporal_front_liveness_penalty
        + temporal_extent_front_liveness_penalty
        + temporal_activation_penalty
        + material_visible_target_mean_distance_penalty
        + material_visible_target_max_distance_penalty;
    let mut morphology_non_regressed = true;
    if let Some(baseline_case) = baseline.and_then(|cases| {
        cases
            .iter()
            .find(|baseline_case| baseline_case.seed == seed)
    }) {
        let surface_regression = if case.active_surface.max_distance.is_finite() {
            (case.active_surface.max_distance - baseline_case.active_surface_max - 0.02).max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let coverage_regression = if case.target_coverage.covered_fraction.is_finite() {
            (baseline_case.target_coverage_fraction - case.target_coverage.covered_fraction - 0.02)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let material_visible_coverage_regression = if case
            .material_visible_target_coverage
            .covered_fraction
            .is_finite()
        {
            (baseline_case.material_visible_target_coverage_fraction
                - case.material_visible_target_coverage.covered_fraction
                - 0.02)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let material_visible_target_mean_regression = if case
            .material_visible_target_coverage
            .mean_distance
            .is_finite()
            && baseline_case
                .material_visible_target_mean_distance
                .is_finite()
        {
            (case.material_visible_target_coverage.mean_distance
                - baseline_case.material_visible_target_mean_distance
                - MATERIAL_VISIBLE_TARGET_DISTANCE_REGRESSION_SLACK)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let material_visible_target_max_regression = if case
            .material_visible_target_coverage
            .max_distance
            .is_finite()
            && baseline_case
                .material_visible_target_max_distance
                .is_finite()
        {
            (case.material_visible_target_coverage.max_distance
                - baseline_case.material_visible_target_max_distance
                - MATERIAL_VISIBLE_TARGET_DISTANCE_REGRESSION_SLACK)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let material_visible_inactive_fraction_regression = if case
            .material_liveness
            .inactive_material_visible_fraction
            .is_finite()
        {
            (case.material_liveness.inactive_material_visible_fraction
                - baseline_case.material_visible_inactive_fraction
                - 0.005)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let material_visible_max_inactive_opacity_regression = if case
            .material_liveness
            .max_inactive_material_opacity
            .is_finite()
            && baseline_case
                .material_visible_max_inactive_opacity
                .is_finite()
        {
            (case.material_liveness.max_inactive_material_opacity
                - baseline_case.material_visible_max_inactive_opacity
                - 0.25)
                .max(0.0)
        } else if case
            .material_liveness
            .max_inactive_material_opacity
            .is_finite()
            && !baseline_case
                .material_visible_max_inactive_opacity
                .is_finite()
        {
            (case.material_liveness.inactive_material_visible_fraction - 0.005).max(0.0)
        } else {
            0.0
        };
        let surface_bin_regression = if case
            .surface_coverage_profile
            .covered_bin_fraction
            .is_finite()
        {
            (baseline_case.surface_covered_bin_fraction
                - case.surface_coverage_profile.covered_bin_fraction
                - 0.05)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let surface_mean_regression = if case
            .surface_coverage_profile
            .mean_bin_covered_fraction
            .is_finite()
        {
            (baseline_case.surface_mean_bin_covered_fraction
                - case.surface_coverage_profile.mean_bin_covered_fraction
                - 0.05)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let material_visible_surface_bin_regression = if case
            .material_visible_surface_coverage_profile
            .covered_bin_fraction
            .is_finite()
        {
            (baseline_case.material_visible_surface_covered_bin_fraction
                - case
                    .material_visible_surface_coverage_profile
                    .covered_bin_fraction
                - 0.05)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let material_visible_surface_mean_regression = if case
            .material_visible_surface_coverage_profile
            .mean_bin_covered_fraction
            .is_finite()
        {
            (baseline_case.material_visible_surface_mean_bin_covered_fraction
                - case
                    .material_visible_surface_coverage_profile
                    .mean_bin_covered_fraction
                - 0.05)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let normal_bin_regression = if case
            .surface_normal_coverage
            .covered_target_bin_fraction
            .is_finite()
        {
            (baseline_case.surface_normal_covered_bin_fraction
                - case.surface_normal_coverage.covered_target_bin_fraction
                - 0.05)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let normal_mean_regression = if case
            .surface_normal_coverage
            .mean_bin_covered_fraction
            .is_finite()
        {
            (baseline_case.surface_normal_mean_bin_covered_fraction
                - case.surface_normal_coverage.mean_bin_covered_fraction
                - 0.05)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let material_visible_normal_bin_regression = if case
            .material_visible_surface_normal_coverage
            .covered_target_bin_fraction
            .is_finite()
        {
            (baseline_case.material_visible_surface_normal_covered_bin_fraction
                - case
                    .material_visible_surface_normal_coverage
                    .covered_target_bin_fraction
                - 0.05)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let material_visible_normal_mean_regression = if case
            .material_visible_surface_normal_coverage
            .mean_bin_covered_fraction
            .is_finite()
        {
            (baseline_case.material_visible_surface_normal_mean_bin_covered_fraction
                - case
                    .material_visible_surface_normal_coverage
                    .mean_bin_covered_fraction
                - 0.05)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let material_visible_tail_p99_regression =
            if case.material_visible_surface_tail.p99_distance.is_finite() {
                (case.material_visible_surface_tail.p99_distance
                    - baseline_case.material_visible_surface_tail_p99_distance
                    - 0.02)
                    .max(0.0)
            } else {
                RENDER_SELECTION_BAD_SCORE
            };
        let material_visible_tail_fraction_regression = if case
            .material_visible_surface_tail
            .over_threshold_fraction
            .is_finite()
        {
            (case.material_visible_surface_tail.over_threshold_fraction
                - baseline_case.material_visible_surface_tail_over_threshold_fraction
                - 0.005)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let active_extent_bbox_regression = if case.extent.bbox_diagonal_ratio.is_finite() {
            (baseline_case.active_extent_bbox_ratio - case.extent.bbox_diagonal_ratio - 0.02)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let active_extent_min_axis_regression = if case.extent.min_axis_extent_ratio.is_finite() {
            (baseline_case.active_extent_min_axis_ratio - case.extent.min_axis_extent_ratio - 0.02)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let active_count_regression = if baseline_case.final_active_count > 0 {
            let baseline_active_count = baseline_case.final_active_count as f32;
            ((baseline_active_count - case.final_active_count as f32) / baseline_active_count
                - 0.02)
                .max(0.0)
        } else {
            0.0
        };
        let newly_activated_regression = if case.newly_activated_fraction.is_finite() {
            (baseline_case.newly_activated_fraction - case.newly_activated_fraction - 0.02).max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let front_local_newly_activated_regression =
            if case.front_local_newly_activated_fraction.is_finite() {
                (baseline_case.front_local_newly_activated_fraction
                    - case.front_local_newly_activated_fraction
                    - 0.02)
                    .max(0.0)
            } else {
                RENDER_SELECTION_BAD_SCORE
            };
        let front_liveness_margin_regression =
            if case.front_liveness.weighted_activation_margin.is_finite()
                && baseline_case
                    .front_liveness
                    .weighted_activation_margin
                    .is_finite()
            {
                (case.front_liveness.weighted_activation_margin
                    - baseline_case.front_liveness.weighted_activation_margin
                    - 0.10)
                    .max(0.0)
            } else {
                RENDER_SELECTION_BAD_SCORE
            };
        let temporal_front_liveness_margin_regression = if case
            .temporal_front_liveness
            .weighted_activation_margin
            .is_finite()
            && baseline_case
                .temporal_front_liveness
                .weighted_activation_margin
                .is_finite()
        {
            (case.temporal_front_liveness.weighted_activation_margin
                - baseline_case
                    .temporal_front_liveness
                    .weighted_activation_margin
                - 0.10)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let temporal_extent_front_liveness_margin_regression = if case
            .temporal_extent_front_liveness
            .weighted_activation_margin
            .is_finite()
            && baseline_case
                .temporal_extent_front_liveness
                .weighted_activation_margin
                .is_finite()
        {
            (case
                .temporal_extent_front_liveness
                .weighted_activation_margin
                - baseline_case
                    .temporal_extent_front_liveness
                    .weighted_activation_margin
                - 0.10)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let extent_front_liveness_margin_regression = if case
            .extent_front_liveness
            .weighted_activation_margin
            .is_finite()
            && baseline_case
                .extent_front_liveness
                .weighted_activation_margin
                .is_finite()
        {
            (case.extent_front_liveness.weighted_activation_margin
                - baseline_case
                    .extent_front_liveness
                    .weighted_activation_margin
                - 0.10)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let temporal_activation_schedule_regression =
            if case.temporal_activation_schedule_error.is_finite()
                && baseline_case.temporal_activation_schedule_error.is_finite()
            {
                (case.temporal_activation_schedule_error
                    - baseline_case.temporal_activation_schedule_error
                    - TEMPORAL_ACTIVATION_SELECTION_REGRESSION_SLACK)
                    .max(0.0)
            } else {
                RENDER_SELECTION_BAD_SCORE
            };
        let temporal_activation_regression = if baseline_case.temporal_activation_progressive
            && !case.temporal_activation_progressive
        {
            1.0
        } else {
            0.0
        };
        let temporal_geometry_regression =
            if baseline_case.temporal_geometry_progressive && !case.temporal_geometry_progressive {
                1.0
            } else {
                0.0
            };
        if surface_regression > 0.0
            || coverage_regression > 0.0
            || material_visible_coverage_regression > 0.0
            || material_visible_target_mean_regression > 0.0
            || material_visible_target_max_regression > 0.0
            || material_visible_inactive_fraction_regression > 0.0
            || material_visible_max_inactive_opacity_regression > 0.0
            || surface_bin_regression > 0.0
            || surface_mean_regression > 0.0
            || material_visible_surface_bin_regression > 0.0
            || material_visible_surface_mean_regression > 0.0
            || normal_bin_regression > 0.0
            || normal_mean_regression > 0.0
            || material_visible_normal_bin_regression > 0.0
            || material_visible_normal_mean_regression > 0.0
            || material_visible_tail_p99_regression > 0.0
            || material_visible_tail_fraction_regression > 0.0
            || active_extent_bbox_regression > 0.0
            || active_extent_min_axis_regression > 0.0
            || active_count_regression > 0.0
            || newly_activated_regression > 0.0
            || front_local_newly_activated_regression > 0.0
            || front_liveness_margin_regression > 0.0
            || extent_front_liveness_margin_regression > 0.0
            || temporal_front_liveness_margin_regression > 0.0
            || temporal_extent_front_liveness_margin_regression > 0.0
            || temporal_activation_schedule_regression > 0.0
            || temporal_activation_regression > 0.0
            || temporal_geometry_regression > 0.0
        {
            morphology_non_regressed = false;
        }
        score += (surface_regression
            + coverage_regression
            + material_visible_coverage_regression
            + material_visible_target_mean_regression
            + material_visible_target_max_regression
            + material_visible_inactive_fraction_regression
            + material_visible_max_inactive_opacity_regression
            + surface_bin_regression
            + surface_mean_regression
            + material_visible_surface_bin_regression
            + material_visible_surface_mean_regression
            + normal_bin_regression
            + normal_mean_regression
            + material_visible_normal_bin_regression
            + material_visible_normal_mean_regression
            + material_visible_tail_p99_regression
            + material_visible_tail_fraction_regression
            + active_extent_bbox_regression
            + active_extent_min_axis_regression
            + active_count_regression
            + newly_activated_regression
            + front_local_newly_activated_regression
            + front_liveness_margin_regression
            + extent_front_liveness_margin_regression
            + temporal_front_liveness_margin_regression
            + temporal_extent_front_liveness_margin_regression
            + temporal_activation_schedule_regression
            + temporal_activation_regression
            + temporal_geometry_regression)
            * 10.0;
    }
    RenderSelectionCaseScore {
        score,
        morphology_non_regressed,
    }
}

pub(crate) struct RenderSelectionCaseScore {
    pub(crate) score: f32,
    pub(crate) morphology_non_regressed: bool,
}

pub(crate) struct RenderSelectionCaseMetrics {
    pub(crate) render_loss: MultiViewRenderLossReport,
    pub(crate) active_surface: Growth3dSurfaceStats,
    pub(crate) target_coverage: TargetCoverageStats,
    pub(crate) material_visible_target_coverage: TargetCoverageStats,
    pub(crate) material_opacity: Growth3dOpacityStats,
    pub(crate) material_liveness: Growth3dMaterialLivenessReport,
    pub(crate) surface_coverage_profile: SurfaceCoverageProfileReport,
    pub(crate) material_visible_surface_coverage_profile: SurfaceCoverageProfileReport,
    pub(crate) surface_normal_coverage: SurfaceNormalCoverageReport,
    pub(crate) material_visible_surface_normal_coverage: SurfaceNormalCoverageReport,
    pub(crate) material_visible_surface_tail: Growth3dSurfaceTailReport,
    pub(crate) extent: Growth3dExtentReport,
    pub(crate) final_active_count: usize,
    pub(crate) newly_activated_fraction: f32,
    pub(crate) front_local_newly_activated_fraction: f32,
    pub(crate) front_liveness: LocalFrontLivenessProgress,
    pub(crate) extent_front_liveness: LocalFrontLivenessProgress,
    pub(crate) temporal_front_liveness: LocalFrontLivenessProgress,
    pub(crate) temporal_extent_front_liveness: LocalFrontLivenessProgress,
    pub(crate) temporal_activation_schedule_error: f32,
    pub(crate) temporal_activation_progressive: bool,
    pub(crate) temporal_geometry_progressive: bool,
    pub(crate) score: f32,
    pub(crate) failure_reasons: Vec<&'static str>,
}

pub(crate) fn render_selection_case_metrics(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: &RenderProxyTrainingConfig,
    render_cfg: RenderLossConfig,
    seed: u64,
) -> Result<RenderSelectionCaseMetrics, Box<dyn std::error::Error>> {
    let trace = render_training_trace_for_seed(model, grid, cfg, seed)?;
    let render_loss = mesh_multiview_render_loss_from_trace(&trace, target, render_cfg)?;
    let final_gaussian_volume = gaussian_volume_stats_for_trace(&trace, render_cfg);
    let rollout_cfg = RolloutConfig {
        particle_count: cfg.particles,
        steps: cfg.rollout_steps,
        update_prob: 1.0,
        seed,
        seed_scale: cfg.seed_scale,
        ..RolloutConfig::default()
    };
    let (seed_positions, seed_states) = seed_particles_scaled(
        1,
        rollout_cfg.particle_count,
        model.config.state_dims,
        model.config.spatial_dims,
        rollout_cfg.seed,
        cfg.seed_mode,
        rollout_cfg.seed_scale,
    );
    let mut active_seed_count = 0usize;
    let mut seed_active = Vec::with_capacity(rollout_cfg.particle_count);
    for state in seed_states.chunks_exact(model.config.state_dims) {
        let active = state[3] > -1.0;
        seed_active.push(active);
        if active {
            active_seed_count += 1;
        }
    }
    let non_opacity_seed_abs_max =
        growth_3d_non_scaffold_seed_abs_max(model.config.state_dims, cfg.seed_mode, &seed_states);
    let activation = growth_3d_activation_report(&trace, &seed_active, active_seed_count);
    let initial_active_surface = growth_3d_active_surface_stats(
        &seed_positions,
        &seed_states,
        model.config.state_dims,
        target,
    );
    let active_surface =
        growth_3d_active_surface_stats(&trace.positions, &trace.states, trace.state_dims, target);
    let active_surface_tail = growth_3d_active_surface_tail_report(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        target,
        GROWTH_3D_SURFACE_MAX_DISTANCE,
    );
    let coverage_samples = cfg.particles.max(512);
    let coverage_threshold = target_coverage_threshold(cfg.seed_scale);
    let initial_target_coverage = active_target_coverage_stats(
        &seed_positions,
        &seed_states,
        model.config.state_dims,
        target,
        coverage_samples,
        coverage_threshold,
    );
    let target_coverage = active_target_coverage_stats(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        target,
        coverage_samples,
        coverage_threshold,
    );
    let material_visible_target_coverage = material_visible_target_coverage_stats(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        target,
        coverage_samples,
        coverage_threshold,
    );
    let surface_coverage_profile = active_surface_coverage_profile(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        target,
        coverage_samples,
        coverage_threshold,
        64,
    );
    let material_visible_surface_coverage_profile = material_visible_surface_coverage_profile(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        target,
        coverage_samples,
        coverage_threshold,
        64,
    );
    let surface_normal_coverage = active_surface_normal_coverage_report(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        target,
        coverage_samples,
        coverage_threshold,
    );
    let material_visible_surface_normal_coverage = material_visible_surface_normal_coverage_report(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        target,
        coverage_samples,
        coverage_threshold,
    );
    let material_visible_surface_tail = growth_3d_material_visible_surface_tail_report(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        target,
        GROWTH_3D_SURFACE_MAX_DISTANCE,
    );
    let torus_angular_coverage = (cfg.target == MeshTargetArg::Torus).then(|| {
        torus_angular_coverage_report(
            &trace.positions,
            &trace.states,
            trace.state_dims,
            cfg.seed_scale,
            coverage_threshold,
            TORUS_ANGULAR_COVERAGE_RINGS,
            TORUS_ANGULAR_COVERAGE_TUBES,
        )
    });
    let motion = growth_3d_motion_report(&trace.mean_dx);
    let extent = growth_3d_extent_report(&trace.positions, &trace.states, trace.state_dims, target);
    let final_opacity = growth_3d_opacity_stats(&trace.states, trace.state_dims);
    let material_opacity = growth_3d_material_opacity_stats(&trace.states, trace.state_dims);
    let material_liveness = growth_3d_material_liveness_report(&trace.states, trace.state_dims);
    let initial_color_state = growth_3d_color_state_report(&seed_states, model.config.state_dims);
    let final_color_state = growth_3d_color_state_report(&trace.states, trace.state_dims);
    let temporal = growth_3d_temporal_report(
        model,
        grid,
        target,
        rollout_cfg.clone(),
        cfg.seed_mode,
        &seed_positions,
        &seed_states,
        &seed_active,
        active_seed_count,
        &trace,
        coverage_samples,
        coverage_threshold,
    )?;
    let permutation_consistency =
        growth_3d_permutation_report(model, grid, &rollout_cfg, cfg.seed_mode)?;
    let front = growth_3d_front_report(
        model,
        grid,
        rollout_cfg,
        cfg.seed_mode,
        &seed_positions,
        &seed_states,
        &trace,
    )?;
    let front_liveness = local_front_liveness_progress(
        &model.config,
        &trace.positions,
        &trace.states,
        cfg.liveness_front_radius,
    );
    let extent_front_liveness = extent_front_liveness_progress(
        &model.config,
        target,
        &trace.positions,
        &trace.states,
        cfg.liveness_front_radius,
    );
    let temporal_front_liveness =
        temporal_front_liveness_progress(model, grid, cfg, seed, &seed_positions, &seed_states)?;
    let temporal_extent_front_liveness = temporal_extent_front_liveness_progress(
        model,
        grid,
        target,
        cfg,
        seed,
        &seed_positions,
        &seed_states,
    )?;
    let mean_final_displacement = growth_3d_mean_displacement(&seed_positions, &trace.positions);
    let mut strict_checks = growth_3d_strict_checks_report(
        model.config.position_features,
        true,
        growth_3d_seed_has_coordinate_scaffold(cfg.seed_mode),
        non_opacity_seed_abs_max,
        final_opacity,
        initial_color_state,
        final_color_state,
        &permutation_consistency,
        &activation,
        initial_active_surface,
        active_surface,
        active_surface_tail,
        initial_target_coverage,
        target_coverage,
        material_visible_target_coverage,
        &surface_normal_coverage,
        &material_visible_surface_normal_coverage,
        torus_angular_coverage.as_ref(),
        final_gaussian_volume,
        &motion,
        &front,
        &temporal,
        extent,
        mean_final_displacement,
        cfg.seed_scale,
        cfg.particles,
        render_loss.passed,
    );
    apply_material_liveness_strict_check(&mut strict_checks, material_liveness);
    apply_material_visible_surface_tail_strict_check(
        &mut strict_checks,
        material_visible_surface_tail,
    );
    apply_surface_profile_strict_check(
        &mut strict_checks,
        &surface_coverage_profile,
        &material_visible_surface_coverage_profile,
    );
    let mut strict_score = growth_3d_strict_score_report(
        &strict_checks,
        initial_active_surface,
        active_surface,
        active_surface_tail,
        initial_target_coverage,
        target_coverage,
        material_visible_target_coverage,
        &surface_normal_coverage,
        &material_visible_surface_normal_coverage,
        extent,
        cfg.seed_scale,
        &render_loss,
        final_gaussian_volume,
    );
    apply_temporal_activation_strict_score(&mut strict_score, &temporal, cfg.rollout_steps);
    apply_morphogenesis_dynamics_strict_score(
        &mut strict_score,
        &motion,
        mean_final_displacement,
        cfg.seed_scale,
    );
    apply_material_liveness_strict_score(&mut strict_score, material_liveness);
    apply_material_visible_surface_tail_strict_score(
        &mut strict_score,
        material_visible_surface_tail,
    );
    apply_surface_profile_strict_score(
        &mut strict_score,
        &surface_coverage_profile,
        &material_visible_surface_coverage_profile,
    );
    let score = strict_score.score;
    let failure_reasons = strict_checks.failure_reasons.clone();
    Ok(RenderSelectionCaseMetrics {
        render_loss,
        active_surface,
        target_coverage,
        material_visible_target_coverage,
        material_opacity,
        material_liveness,
        surface_coverage_profile,
        material_visible_surface_coverage_profile,
        surface_normal_coverage,
        material_visible_surface_normal_coverage,
        material_visible_surface_tail,
        extent,
        final_active_count: activation.final_active_count,
        newly_activated_fraction: activation.newly_activated_fraction,
        front_local_newly_activated_fraction: front.local_newly_activated_fraction,
        front_liveness,
        extent_front_liveness,
        temporal_front_liveness,
        temporal_extent_front_liveness,
        temporal_activation_schedule_error: temporal_activation_schedule_error(
            &temporal,
            cfg.rollout_steps,
        ),
        temporal_activation_progressive: temporal.progressive_activation,
        temporal_geometry_progressive: temporal.geometry_progressive,
        score,
        failure_reasons,
    })
}
