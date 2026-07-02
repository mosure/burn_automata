use super::super::*;

pub(crate) struct RenderSelectionCaseMetrics {
    pub(crate) render_loss: MultiViewRenderLossReport,
    pub(crate) active_surface: Growth3dSurfaceStats,
    pub(crate) target_coverage: TargetCoverageStats,
    pub(crate) material_visible_target_coverage: TargetCoverageStats,
    pub(crate) strict_surface_materialization: Growth3dStrictSurfaceMaterializationReport,
    pub(crate) material_opacity: Growth3dOpacityStats,
    pub(crate) material_liveness: Growth3dMaterialLivenessReport,
    pub(crate) final_color_state: Growth3dColorStateReport,
    pub(crate) surface_coverage_profile: SurfaceCoverageProfileReport,
    pub(crate) material_visible_surface_coverage_profile: SurfaceCoverageProfileReport,
    pub(crate) surface_normal_coverage: SurfaceNormalCoverageReport,
    pub(crate) material_visible_surface_normal_coverage: SurfaceNormalCoverageReport,
    pub(crate) material_visible_surface_tail: Growth3dSurfaceTailReport,
    pub(crate) dormant_drift: Growth3dDormantDriftReport,
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
    let strict_surface_materialization = active_strict_surface_materialization_report(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        target,
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
    let dormant_drift = growth_3d_dormant_drift_report(
        model,
        grid,
        rollout_cfg.clone(),
        cfg.seed_mode,
        &seed_positions,
        &seed_states,
        &trace,
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
    apply_dormant_drift_strict_check(&mut strict_checks, dormant_drift);
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
        strict_surface_materialization,
        material_opacity,
        material_liveness,
        final_color_state,
        surface_coverage_profile,
        material_visible_surface_coverage_profile,
        surface_normal_coverage,
        material_visible_surface_normal_coverage,
        material_visible_surface_tail,
        dormant_drift,
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
