use super::*;

#[test]
fn shipped_3d_growth_assets_are_local_dynamic_and_not_target_seeded() {
    for (relative_path, seed_mode, expected_source) in [
        (
            "assets/models/uv_torus_growth_3d.bpk",
            ParticleSeed::TorusGrowth3d,
            "render-refined-rust:ablation-rust:uv-torus-3d:conditionless-local-random-ball-rollout-ablation",
        ),
        (
            "assets/models/teapot_growth_3d.bpk",
            ParticleSeed::TeapotGrowth3d,
            "retimed-local-front:hidden=skipped:gain=2:alpha=1:front_retime=false:active_opacity_hidden=skipped:active_opacity_gain=skipped:opacity_bias=skipped:material_opacity_bias=0.55:base=render-refined-rust:ablation-rust:utah-teapot-2026:conditionless-local-random-ball-rollout-ablation",
        ),
    ] {
        let path = workspace_path(relative_path);
        let manifest = burn_automata::import::load_manifest(path).unwrap();
        assert_eq!(manifest.model_kind, "npa", "{relative_path}");
        assert_eq!(manifest.config.spatial_dims, 3, "{relative_path}");
        assert!(
            !manifest.config.position_features,
            "{relative_path} must not use absolute world-position features"
        );
        let source = manifest.source.as_deref().unwrap_or_default();
        assert_eq!(
            source, expected_source,
            "{relative_path} should stay on the reviewed latest dynamic 3D growth artifact"
        );
        assert!(
            (source.starts_with("render-refined-rust:")
                || source.starts_with("retimed-local-front:"))
                && source.contains("conditionless-local")
                && !source.contains("position-field")
                && !source.contains("seed-frame")
                && !source.contains("render-proxy-rust"),
            "{relative_path} must use latest local render-refinement lineage without target-assigned shortcuts, source={source}"
        );
        let grid = manifest.hashgrid.clone();
        let model = manifest.into_model();
        let cfg = RolloutConfig {
            particle_count: 512,
            steps: 64,
            update_prob: 1.0,
            seed: CATALOG_3D_GROWTH_SEED,
            seed_scale: 0.72,
            ..RolloutConfig::default()
        };
        let (_seed_positions, seed_states) = seed_particles_scaled(
            1,
            cfg.particle_count,
            model.config.state_dims,
            model.config.spatial_dims,
            cfg.seed,
            seed_mode,
            cfg.seed_scale,
        );
        let active_seed_count = seed_states
            .chunks_exact(model.config.state_dims)
            .filter(|state| state[3] > -1.0)
            .count();
        assert!(
            active_seed_count > 0 && active_seed_count < cfg.particle_count / 8,
            "{relative_path} should start from a sparse active core, active={active_seed_count}"
        );

        let trace = run_rollout(&model, &grid, &cfg, seed_mode).unwrap();
        let initial_color_state = color_state_stats(&seed_states, model.config.state_dims);
        let final_color_state = color_state_stats(&trace.states, model.config.state_dims);
        assert!(
            initial_color_state.active_max_abs <= 1.0e-6,
            "{relative_path} should not precolor the sparse seed core: {initial_color_state:?}"
        );
        assert!(
            final_color_state.active_mean_abs >= initial_color_state.active_mean_abs + 0.02
                && final_color_state.active_max_abs >= 0.05,
            "{relative_path} should grow visible color state from neutral seed: initial={initial_color_state:?} final={final_color_state:?}"
        );
        assert!(
            final_color_state.active_channel_stddev_mean >= 0.02,
            "{relative_path} final color state should vary across active particles instead of becoming a uniform tint: {final_color_state:?}"
        );
        let max_motion = trace.mean_dx.iter().copied().fold(0.0_f32, f32::max);
        let mut max_radius = 0.0_f32;
        let mut max_abs_z = 0.0_f32;
        let mut min_opacity = f32::MAX;
        let mut max_opacity = f32::MIN;
        for (idx, position) in trace.positions.iter().enumerate() {
            assert!(position.iter().all(|value| value.is_finite()));
            max_radius = max_radius.max(
                (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                    .sqrt(),
            );
            max_abs_z = max_abs_z.max(position[2].abs());
            let opacity = trace.states[idx * model.config.state_dims + 3];
            min_opacity = min_opacity.min(opacity);
            max_opacity = max_opacity.max(opacity);
        }
        assert!(trace.mean_dx.iter().all(|value| value.is_finite()));
        assert!(
            max_motion > 0.01,
            "{relative_path} should move from the sparse-core seed, max mean dx={max_motion}"
        );
        assert!(
            max_radius > growth_3d_seed_radius(cfg.seed_scale),
            "{relative_path} should expand beyond the compact seed, max radius={max_radius}"
        );
        assert!(
            max_abs_z > cfg.seed_scale * 0.25,
            "{relative_path} should use 3D volume, max |z|={max_abs_z}"
        );
        assert!(
            min_opacity.is_finite() && max_opacity.is_finite() && max_opacity < 24.0,
            "{relative_path} opacity state should stay finite and bounded, min={min_opacity} max={max_opacity}"
        );
    }
}
#[test]
fn shipped_3d_growth_assets_remain_bounded_across_seed_sweep() {
    for (relative_path, seed_mode) in [
        (
            "assets/models/uv_torus_growth_3d.bpk",
            ParticleSeed::TorusGrowth3d,
        ),
        (
            "assets/models/teapot_growth_3d.bpk",
            ParticleSeed::TeapotGrowth3d,
        ),
    ] {
        let manifest = burn_automata::import::load_manifest(workspace_path(relative_path)).unwrap();
        let grid = manifest.hashgrid.clone();
        let model = manifest.into_model();
        for seed in [CATALOG_3D_GROWTH_SEED, 42, 99, 1234] {
            let cfg = RolloutConfig {
                particle_count: 512,
                steps: 64,
                update_prob: 1.0,
                seed,
                seed_scale: 0.72,
                ..RolloutConfig::default()
            };
            let (_seed_positions, seed_states) = seed_particles_scaled(
                1,
                cfg.particle_count,
                model.config.state_dims,
                model.config.spatial_dims,
                cfg.seed,
                seed_mode,
                cfg.seed_scale,
            );
            let active_seed_count = seed_states
                .chunks_exact(model.config.state_dims)
                .filter(|state| state[3] > -1.0)
                .count();
            let trace = run_rollout(&model, &grid, &cfg, seed_mode).unwrap();
            let max_motion = trace.mean_dx.iter().copied().fold(0.0_f32, f32::max);
            let mut final_active_count = 0usize;
            let mut max_opacity = f32::NEG_INFINITY;
            for state in trace.states.chunks_exact(model.config.state_dims) {
                let opacity = state[3];
                assert!(
                    opacity.is_finite(),
                    "{relative_path} seed {seed} produced non-finite opacity {opacity}"
                );
                max_opacity = max_opacity.max(opacity);
                if opacity > -1.0 {
                    final_active_count += 1;
                }
            }
            assert!(
                active_seed_count > 0 && active_seed_count < cfg.particle_count / 8,
                "{relative_path} seed {seed} should start from sparse growth core, active={active_seed_count}"
            );
            assert!(
                final_active_count > active_seed_count * 4,
                "{relative_path} seed {seed} should grow visible particles, active={active_seed_count}->{final_active_count}"
            );
            assert!(
                max_motion > 0.01,
                "{relative_path} seed {seed} should move dynamically, max mean dx={max_motion}"
            );
            assert!(
                max_opacity < 24.0,
                "{relative_path} seed {seed} opacity should stay bounded, max={max_opacity}"
            );
        }
    }
}
#[test]
fn shipped_3d_growth_assets_are_strictly_measured_before_promotion() {
    for case in [
        GrowthValidationCase::torus("assets/models/uv_torus_growth_3d.bpk"),
        GrowthValidationCase::teapot("assets/models/teapot_growth_3d.bpk"),
    ] {
        let report = strict_growth_validation_report(case);

        assert!(
            !report.position_features,
            "{} must stay local and must not use absolute world-position features",
            case.relative_path
        );
        assert!(
            report.non_opacity_seed_abs_max <= 1.0e-6,
            "{} seeds target or identity state outside opacity: max abs {}",
            case.relative_path,
            report.non_opacity_seed_abs_max
        );
        assert!(
            report.active_seed_count > 0 && report.active_seed_count < case.particle_count / 8,
            "{} should initialize like 2D growth with a sparse active core, active={}",
            case.relative_path,
            report.active_seed_count
        );
        assert!(
            report.final_active_count > report.active_seed_count * 4,
            "{} should activate substantially more particles than the seed core: seed_active={} final_active={}",
            case.relative_path,
            report.active_seed_count,
            report.final_active_count
        );
        assert!(
            report.newly_activated_fraction >= 0.50,
            "{} should activate at least half of initially inactive particles, activated={} fraction={}",
            case.relative_path,
            report.newly_activated_count,
            report.newly_activated_fraction
        );
        assert!(
            report.final_active_max_radius > growth_3d_seed_radius(case.seed_scale),
            "{} active front should expand beyond the initial seed ball, active_mean_radius={} active_max_radius={}",
            case.relative_path,
            report.final_active_mean_radius,
            report.final_active_max_radius
        );
        assert!(
            report.max_motion_per_step > 0.01,
            "{} appears static from the compact seed: max mean dx={}",
            case.relative_path,
            report.max_motion_per_step
        );
        assert!(
            report.mean_final_displacement > growth_3d_seed_radius(case.seed_scale),
            "{} should actually grow out of the compact seed, mean displacement={}",
            case.relative_path,
            report.mean_final_displacement
        );
        assert!(
            !report.strict_passed,
            "{} unexpectedly passed the strict local-3D morphogenesis gate; promote it by replacing this guard with the positive gate",
            case.relative_path
        );
        if matches!(case.target, GrowthTarget::Teapot) {
            assert!(
                !report.temporal_progressive_activation,
                "{} should remain blocked on seed-varied temporal activation until robust retraining replaces the shipped artifact: {report:?}",
                case.relative_path
            );
        }
        assert!(
            report.temporal_geometry_progressive.is_finite(),
            "{} temporal geometry progress should be measured: {report:?}",
            case.relative_path
        );
        if matches!(case.target, GrowthTarget::Teapot) {
            assert!(
                report.temporal_geometry_progressive.passed,
                "{} teapot geometry should remain progressive under corrected target sampling: {report:?}",
                case.relative_path
            );
        } else {
            assert!(
                !report.temporal_geometry_progressive.passed,
                "{} torus should expose the corrected full-surface coverage blocker until retrained: {report:?}",
                case.relative_path
            );
        }
        assert!(
            report.front_coherence.passed,
            "{} should activate through a local front instead of waking distant target particles directly: {report:?}",
            case.relative_path
        );
        assert!(
            report.front_coherence.transition_count >= 2
                && report.front_coherence.newly_activated_count > 0,
            "{} should grow through multiple measured local-front transitions: {report:?}",
            case.relative_path
        );
        assert!(
            report.front_coherence.local_newly_activated_fraction >= 0.90
                && report.front_coherence.max_nearest_previous_active_distance
                    <= report.front_coherence.max_allowed_distance * 1.05,
            "{} front coherence distances should remain bounded: {report:?}",
            case.relative_path
        );
        assert!(
            report.front_coherence.mean_nearest_previous_active_distance
                <= report.front_coherence.max_allowed_distance * 0.75,
            "{} mean local-front activation distance should stay comfortably below the threshold: {report:?}",
            case.relative_path
        );
        if matches!(case.target, GrowthTarget::Torus) {
            assert!(
                report.final_target_coverage.covered_fraction < 0.60,
                "{} unexpectedly passed full-torus target coverage; replace this guard with a positive assertion after the next artifact promotion",
                case.relative_path
            );
        } else {
            assert!(
                report.final_target_coverage.covered_fraction >= 0.60,
                "{} teapot should now pass target coverage under corrected surface sampling: {report:?}",
                case.relative_path
            );
        }
        if matches!(case.target, GrowthTarget::Teapot) {
            assert!(
                report.render_density_psnr_db >= 10.0,
                "{} teapot diagnostic should retain render-density PSNR even while robust temporal activation is blocked, got {}",
                case.relative_path,
                report.render_density_psnr_db
            );
        } else {
            assert!(
                report.render_density_psnr_db < 10.0,
                "{} strict gate should fail specifically on shape density today, got density PSNR {}",
                case.relative_path,
                report.render_density_psnr_db
            );
        }
        assert!(
            report.final_surface.mean.is_finite()
                && report.final_surface.max.is_finite()
                && report.initial_surface.mean.is_finite()
                && report.initial_surface.max.is_finite()
                && report.final_active_surface.mean.is_finite()
                && report.final_active_surface.max.is_finite()
                && report.initial_active_surface.mean.is_finite()
                && report.initial_active_surface.max.is_finite(),
            "{} surface metrics should be finite: {report:?}",
            case.relative_path
        );
        assert!(
            report.initial_target_coverage.mean.is_finite()
                && report.initial_target_coverage.max.is_finite()
                && report.initial_target_coverage.covered_fraction.is_finite()
                && report.final_target_coverage.mean.is_finite()
                && report.final_target_coverage.max.is_finite()
                && report.final_target_coverage.covered_fraction.is_finite(),
            "{} target coverage metrics should be finite: {report:?}",
            case.relative_path
        );
        assert!(
            report.render_color_psnr_db.is_finite() && report.render_depth_psnr_db.is_finite(),
            "{} render color/depth metrics should be finite: {report:?}",
            case.relative_path
        );
    }
}
#[test]
fn shipped_3d_growth_assets_are_dynamic_but_render_gap_is_measured() {
    for case in [
        CatalogRenderSanityCase {
            validation: GrowthValidationCase::torus("assets/models/uv_torus_growth_3d.bpk"),
            max_total_loss: 1.60,
            min_density_psnr_db: -1.85,
            min_color_psnr_db: 12.0,
            min_depth_psnr_db: 18.5,
        },
        CatalogRenderSanityCase {
            validation: GrowthValidationCase::teapot("assets/models/teapot_growth_3d.bpk"),
            max_total_loss: 0.25,
            min_density_psnr_db: 7.5,
            min_color_psnr_db: 15.0,
            min_depth_psnr_db: 28.0,
        },
    ] {
        let report = catalog_render_sanity_report(case.validation);
        assert!(
            report.total_loss <= case.max_total_loss
                && report.density_psnr_db >= case.min_density_psnr_db
                && report.color_psnr_db >= case.min_color_psnr_db
                && report.depth_psnr_db >= case.min_depth_psnr_db,
            "{} regressed below the latest dynamic-artifact render floor: {report:?}",
            case.validation.relative_path
        );
        assert!(
            report.density_psnr_db < 10.0,
            "{} 512-particle catalog sanity should still record the current low-count render-density gap: {report:?}",
            case.validation.relative_path
        );
    }
}
#[test]
#[ignore = "acceptance gate for the next promoted local 3D artifacts"]
fn promoted_3d_growth_assets_pass_strict_morphogenesis_gate() {
    for case in [
        GrowthValidationCase::torus("assets/models/uv_torus_growth_3d.bpk"),
        GrowthValidationCase::teapot("assets/models/teapot_growth_3d.bpk"),
    ] {
        let report = strict_growth_validation_report(case);
        assert!(
            report.strict_passed,
            "{} did not pass strict morphogenesis gate: {report:?}",
            case.relative_path
        );
    }
}
