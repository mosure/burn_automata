#![allow(clippy::too_many_arguments)]

use super::*;

pub(crate) fn growth_3d_validation_report(
    model_path: &PathBuf,
    target_arg: MeshTargetArg,
    cfg: Growth3dValidationConfig,
) -> Result<CliGrowth3dValidationReport, Box<dyn std::error::Error>> {
    let seeds = eval_seed_list(cfg.seed, &cfg.extra_seeds);
    let mut primary_cfg = cfg.clone();
    primary_cfg.extra_seeds.clear();
    let mut primary = growth_3d_validation_report_single(model_path, target_arg, primary_cfg)?;
    let mut seed_reports = Vec::with_capacity(seeds.len());
    seed_reports.push(growth_3d_robustness_seed_report(&primary));
    for seed in seeds.iter().skip(1) {
        let mut seed_cfg = cfg.clone();
        seed_cfg.seed = *seed;
        seed_cfg.extra_seeds.clear();
        let report = growth_3d_validation_report_single(model_path, target_arg, seed_cfg)?;
        seed_reports.push(growth_3d_robustness_seed_report(&report));
    }
    primary.robustness = growth_3d_robustness_report(seed_reports);
    Ok(primary)
}

pub(crate) fn growth_3d_fail_on_validation_passed(report: &CliGrowth3dValidationReport) -> bool {
    if report.robustness.seed_count > 1 {
        report.robustness.all_gate_passed
    } else {
        report.gate_passed
    }
}

pub(crate) fn growth_3d_validation_report_single(
    model_path: &PathBuf,
    target_arg: MeshTargetArg,
    cfg: Growth3dValidationConfig,
) -> Result<CliGrowth3dValidationReport, Box<dyn std::error::Error>> {
    let manifest = crate::import::load_manifest(model_path)?;
    if manifest.config.spatial_dims != 3 || manifest.config.state_dims <= 3 {
        return Err(std::io::Error::other(format!(
            "growth 3D validation requires spatial_dims=3 and state_dims>3; got spatial_dims={} state_dims={}",
            manifest.config.spatial_dims, manifest.config.state_dims
        ))
        .into());
    }
    let source = manifest.source.clone();
    let source_text = source.as_deref().unwrap_or_default();
    let local_conditionless_lineage = local_conditionless_lineage(source_text);
    let position_features = manifest.config.position_features;
    let seed_coordinate_scaffold = growth_3d_seed_has_coordinate_scaffold(cfg.seed_mode);
    let grid = manifest.hashgrid.clone();
    let model = manifest.into_model();
    let target = mesh_target_for_arg(target_arg, cfg.seed_scale);
    let rollout_cfg = RolloutConfig {
        particle_count: cfg.particle_count,
        steps: cfg.steps,
        update_prob: 1.0,
        seed: cfg.seed,
        seed_scale: cfg.seed_scale,
        ..RolloutConfig::default()
    };
    let rollout_steps = rollout_cfg.steps;
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

    let trace = run_rollout(&model, &grid, &rollout_cfg, cfg.seed_mode)?;
    let activation = growth_3d_activation_report(&trace, &seed_active, active_seed_count);
    let final_opacity = growth_3d_opacity_stats(&trace.states, trace.state_dims);
    let final_material_opacity = growth_3d_material_opacity_stats(&trace.states, trace.state_dims);
    let final_material_liveness =
        growth_3d_material_liveness_report(&trace.states, trace.state_dims);
    let initial_color_state = growth_3d_color_state_report(&seed_states, model.config.state_dims);
    let final_color_state = growth_3d_color_state_report(&trace.states, trace.state_dims);
    let permutation_consistency =
        growth_3d_permutation_report(&model, &grid, &rollout_cfg, cfg.seed_mode)?;
    let seed_perturbation =
        growth_3d_seed_perturbation_report(&model, &grid, &rollout_cfg, cfg.seed_mode)?;
    let initial_surface = growth_3d_surface_stats(&seed_positions, &target);
    let final_surface = growth_3d_surface_stats(&trace.positions, &target);
    let initial_active_surface = growth_3d_active_surface_stats(
        &seed_positions,
        &seed_states,
        model.config.state_dims,
        &target,
    );
    let final_active_surface =
        growth_3d_active_surface_stats(&trace.positions, &trace.states, trace.state_dims, &target);
    let initial_active_surface_tail = growth_3d_active_surface_tail_report(
        &seed_positions,
        &seed_states,
        model.config.state_dims,
        &target,
        GROWTH_3D_SURFACE_MAX_DISTANCE,
    );
    let final_active_surface_tail = growth_3d_active_surface_tail_report(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        &target,
        GROWTH_3D_SURFACE_MAX_DISTANCE,
    );
    let initial_material_visible_surface_tail = growth_3d_material_visible_surface_tail_report(
        &seed_positions,
        &seed_states,
        model.config.state_dims,
        &target,
        GROWTH_3D_SURFACE_MAX_DISTANCE,
    );
    let final_material_visible_surface_tail = growth_3d_material_visible_surface_tail_report(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        &target,
        GROWTH_3D_SURFACE_MAX_DISTANCE,
    );
    let target_coverage_threshold = target_coverage_threshold(cfg.seed_scale);
    let coverage_samples = cfg.particle_count.max(512);
    let initial_target_coverage = target_coverage_stats(
        &seed_positions,
        &target,
        coverage_samples,
        target_coverage_threshold,
    );
    let final_target_coverage = target_coverage_stats(
        &trace.positions,
        &target,
        coverage_samples,
        target_coverage_threshold,
    );
    let initial_active_target_coverage = active_target_coverage_stats(
        &seed_positions,
        &seed_states,
        model.config.state_dims,
        &target,
        coverage_samples,
        target_coverage_threshold,
    );
    let final_active_target_coverage = active_target_coverage_stats(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        &target,
        coverage_samples,
        target_coverage_threshold,
    );
    let initial_material_visible_target_coverage = material_visible_target_coverage_stats(
        &seed_positions,
        &seed_states,
        model.config.state_dims,
        &target,
        coverage_samples,
        target_coverage_threshold,
    );
    let final_material_visible_target_coverage = material_visible_target_coverage_stats(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        &target,
        coverage_samples,
        target_coverage_threshold,
    );
    let initial_strict_surface_materialization = active_strict_surface_materialization_report(
        &seed_positions,
        &seed_states,
        model.config.state_dims,
        &target,
        target_coverage_threshold,
    );
    let final_strict_surface_materialization = active_strict_surface_materialization_report(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        &target,
        target_coverage_threshold,
    );
    let final_active_surface_coverage_profile = active_surface_coverage_profile(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        &target,
        coverage_samples,
        target_coverage_threshold,
        64,
    );
    let final_material_visible_surface_coverage_profile = material_visible_surface_coverage_profile(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        &target,
        coverage_samples,
        target_coverage_threshold,
        64,
    );
    let final_active_surface_normal_coverage = active_surface_normal_coverage_report(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        &target,
        coverage_samples,
        target_coverage_threshold,
    );
    let final_material_visible_surface_normal_coverage =
        material_visible_surface_normal_coverage_report(
            &trace.positions,
            &trace.states,
            trace.state_dims,
            &target,
            coverage_samples,
            target_coverage_threshold,
        );
    let torus_angular_coverage = (target_arg == MeshTargetArg::Torus).then(|| {
        torus_angular_coverage_report(
            &trace.positions,
            &trace.states,
            trace.state_dims,
            cfg.seed_scale,
            target_coverage_threshold,
            TORUS_ANGULAR_COVERAGE_RINGS,
            TORUS_ANGULAR_COVERAGE_TUBES,
        )
    });
    let extent =
        growth_3d_extent_report(&trace.positions, &trace.states, trace.state_dims, &target);
    let mut render_cfg = cfg.render;
    if render_cfg.target_samples == 0 {
        render_cfg.target_samples = coverage_samples;
    }
    let render_loss = mesh_multiview_render_loss_from_trace(&trace, &target, render_cfg)?;
    let initial_trace = crate::RolloutTrace {
        positions: seed_positions.clone(),
        states: seed_states.clone(),
        batch_size: rollout_cfg.batch_size,
        particle_count: rollout_cfg.particle_count,
        state_dims: model.config.state_dims,
        steps: 0,
        mean_dx: Vec::new(),
    };
    let initial_gaussian_volume = gaussian_volume_stats_for_trace(&initial_trace, render_cfg);
    let final_gaussian_volume = gaussian_volume_stats_for_trace(&trace, render_cfg);
    let catalog_sanity = growth_3d_catalog_sanity_report(target_arg, &render_loss);
    let mean_final_displacement = growth_3d_mean_displacement(&seed_positions, &trace.positions);
    let motion = growth_3d_motion_report(&trace.mean_dx);
    let temporal = growth_3d_temporal_report(
        &model,
        &grid,
        &target,
        rollout_cfg.clone(),
        cfg.seed_mode,
        &seed_positions,
        &seed_states,
        &seed_active,
        active_seed_count,
        &trace,
        coverage_samples,
        target_coverage_threshold,
    )?;
    let front = growth_3d_front_report(
        &model,
        &grid,
        rollout_cfg,
        cfg.seed_mode,
        &seed_positions,
        &seed_states,
        &trace,
    )?;
    let mut strict_checks = growth_3d_strict_checks_report(
        position_features,
        local_conditionless_lineage,
        seed_coordinate_scaffold,
        non_opacity_seed_abs_max,
        final_opacity,
        initial_color_state,
        final_color_state,
        &permutation_consistency,
        &activation,
        initial_active_surface,
        final_active_surface,
        final_active_surface_tail,
        initial_active_target_coverage,
        final_active_target_coverage,
        final_material_visible_target_coverage,
        &final_active_surface_normal_coverage,
        &final_material_visible_surface_normal_coverage,
        torus_angular_coverage.as_ref(),
        final_gaussian_volume,
        &motion,
        &front,
        &temporal,
        extent,
        mean_final_displacement,
        cfg.seed_scale,
        cfg.particle_count,
        render_loss.passed,
    );
    apply_material_liveness_strict_check(&mut strict_checks, final_material_liveness);
    apply_material_visible_surface_tail_strict_check(
        &mut strict_checks,
        final_material_visible_surface_tail,
    );
    apply_surface_profile_strict_check(
        &mut strict_checks,
        &final_active_surface_coverage_profile,
        &final_material_visible_surface_coverage_profile,
    );
    let strict_passed = strict_checks.passed;
    let mut strict_score = growth_3d_strict_score_report(
        &strict_checks,
        initial_active_surface,
        final_active_surface,
        final_active_surface_tail,
        initial_active_target_coverage,
        final_active_target_coverage,
        final_material_visible_target_coverage,
        &final_active_surface_normal_coverage,
        &final_material_visible_surface_normal_coverage,
        extent,
        cfg.seed_scale,
        &render_loss,
        final_gaussian_volume,
    );
    apply_temporal_activation_strict_score(&mut strict_score, &temporal, rollout_steps);
    apply_morphogenesis_dynamics_strict_score(
        &mut strict_score,
        &motion,
        mean_final_displacement,
        cfg.seed_scale,
    );
    apply_material_liveness_strict_score(&mut strict_score, final_material_liveness);
    apply_material_visible_surface_tail_strict_score(
        &mut strict_score,
        final_material_visible_surface_tail,
    );
    apply_surface_profile_strict_score(
        &mut strict_score,
        &final_active_surface_coverage_profile,
        &final_material_visible_surface_coverage_profile,
    );
    let catalog_gate_passed = !position_features
        && local_conditionless_lineage
        && !seed_coordinate_scaffold
        && non_opacity_seed_abs_max <= 1.0e-6
        && final_opacity.finite
        && final_opacity.max <= GROWTH_3D_MAX_FINAL_OPACITY_LOGIT
        && final_material_liveness.passed
        && strict_checks.color_state_emerged
        && strict_checks.permutation_consistent
        && strict_checks.gaussian_scale_budget
        && strict_checks.material_visible_surface_tail_bounded
        && strict_checks.material_visible_target_coverage_fraction
        && strict_checks.surface_coverage_profile
        && strict_checks.material_visible_surface_coverage_profile
        && strict_checks.material_visible_surface_normal_coverage
        && activation.active_seed_count > 0
        && activation.active_seed_count <= cfg.particle_count / 8
        && activation.final_active_count > activation.active_seed_count * 4
        && activation.newly_activated_fraction >= 0.50
        && activation.final_active_max_radius > growth_3d_seed_radius(cfg.seed_scale)
        && motion.peak_mean_dx > 0.01
        && motion.active_step_fraction >= 0.50
        && motion.sustained_step_fraction >= 0.25
        && front.passed
        && mean_final_displacement > growth_3d_seed_radius(cfg.seed_scale)
        && catalog_sanity.passed;
    let gate_passed = match cfg.gate {
        Growth3dValidationGateArg::Strict => strict_passed,
        Growth3dValidationGateArg::CatalogSanity => catalog_gate_passed,
    };

    Ok(CliGrowth3dValidationReport {
        target: target_arg,
        model: model_path.display().to_string(),
        source,
        position_features,
        local_conditionless_lineage,
        seed_coordinate_scaffold,
        particle_count: cfg.particle_count,
        steps: cfg.steps,
        seed: cfg.seed,
        seed_scale: cfg.seed_scale,
        seed_mode: cfg.seed_mode,
        non_opacity_seed_abs_max,
        initial_color_state,
        final_color_state,
        permutation_consistency,
        seed_perturbation,
        mean_final_displacement,
        final_opacity,
        final_material_opacity,
        final_material_liveness,
        activation,
        initial_surface,
        final_surface,
        initial_active_surface,
        final_active_surface,
        initial_active_surface_tail,
        final_active_surface_tail,
        initial_material_visible_surface_tail,
        final_material_visible_surface_tail,
        target_coverage_threshold,
        initial_target_coverage,
        final_target_coverage,
        initial_active_target_coverage,
        final_active_target_coverage,
        initial_material_visible_target_coverage,
        final_material_visible_target_coverage,
        initial_strict_surface_materialization,
        final_strict_surface_materialization,
        final_active_surface_coverage_profile,
        final_material_visible_surface_coverage_profile,
        final_active_surface_normal_coverage,
        final_material_visible_surface_normal_coverage,
        torus_angular_coverage,
        extent,
        motion,
        temporal,
        front,
        max_motion_per_step: motion.peak_mean_dx,
        render_loss,
        initial_gaussian_volume,
        final_gaussian_volume,
        strict_checks,
        strict_score,
        catalog_sanity,
        robustness: growth_3d_empty_robustness_report(cfg.seed),
        gate: cfg.gate,
        gate_passed,
        strict_passed,
    })
}
