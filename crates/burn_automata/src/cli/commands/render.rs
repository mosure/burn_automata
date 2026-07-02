use crate::cli::prelude::*;

use super::{resolve_direct_selection_seed_training, resolve_full_coverage_adjoint};

pub(crate) fn run_render_loss_3d(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::RenderLoss3d {
        model,
        target,
        output,
        particles,
        steps,
        seed,
        extra_seeds,
        seed_scale,
        seed_mode,
        image_size,
        target_samples,
        sigma,
        min_sigma,
        max_sigma,
        gaussian_decode_mode,
        world_scale,
        render_opacity_logit_bias,
        density_weight,
        color_weight,
        depth_weight,
        fail_on_validation,
    } = command
    else {
        unreachable!("run_render_loss_3d called with the wrong command variant");
    };

    let manifest = crate::import::load_manifest(&model)?;
    let hashgrid = manifest.hashgrid.clone();
    let loaded_model = manifest.into_model();
    let target_mesh = mesh_target_for_arg(target, seed_scale);
    let seed_mode: ParticleSeed = seed_mode.into();
    let render_loss = mesh_render_loss_for_model(
        &loaded_model,
        &hashgrid,
        &target_mesh,
        RenderLossEvalConfig {
            particle_count: particles,
            steps,
            seed,
            extra_seeds,
            seed_scale,
            seed_mode,
            render: RenderLossConfig {
                image_size,
                sigma,
                min_sigma,
                max_sigma,
                gaussian_decode_mode: gaussian_decode_mode.into(),
                world_scale: world_scale.unwrap_or(seed_scale * 2.0),
                target_samples,
                opacity_logit_bias: render_opacity_logit_bias,
                density_weight,
                color_weight,
                depth_weight,
            },
        },
    )?;
    let output_report = CliRenderLossEvalReport {
        target,
        model: model.display().to_string(),
        particle_count: particles,
        steps,
        seed,
        seed_scale,
        seed_mode,
        render_loss,
    };
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&output, serde_json::to_string_pretty(&output_report)?)?;
    println!(
        "wrote {} render_loss={:.6} density_psnr={:.3} color_psnr={:.3} depth_psnr={:.3} passed={}",
        output.display(),
        output_report.render_loss.total_loss,
        output_report.render_loss.density_psnr_db,
        output_report.render_loss.color_psnr_db,
        output_report.render_loss.depth_psnr_db,
        output_report.render_loss.passed
    );
    if fail_on_validation && !output_report.render_loss.passed {
        return Err(std::io::Error::other(format!(
            "render loss validation failed; see {}",
            output.display()
        ))
        .into());
    }

    Ok(())
}

pub(crate) fn run_validate_growth_3d(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::ValidateGrowth3d {
        model,
        target,
        output,
        particles,
        steps,
        seed,
        extra_seeds,
        seed_scale,
        seed_mode,
        image_size,
        target_samples,
        sigma,
        min_sigma,
        max_sigma,
        gaussian_decode_mode,
        world_scale,
        render_opacity_logit_bias,
        density_weight,
        color_weight,
        depth_weight,
        gate,
        fail_on_validation,
    } = command
    else {
        unreachable!("run_validate_growth_3d called with the wrong command variant");
    };

    let seed_mode = seed_mode
        .map(ParticleSeed::from)
        .unwrap_or_else(|| conditionless_local_seed_mode(target));
    let report = growth_3d_validation_report(
        &model,
        target,
        Growth3dValidationConfig {
            particle_count: particles,
            steps,
            seed,
            extra_seeds,
            seed_scale,
            seed_mode,
            gate,
            render: RenderLossConfig {
                image_size,
                sigma,
                min_sigma,
                max_sigma,
                gaussian_decode_mode: gaussian_decode_mode.into(),
                world_scale: world_scale.unwrap_or(seed_scale * 2.0),
                target_samples,
                opacity_logit_bias: render_opacity_logit_bias,
                density_weight,
                color_weight,
                depth_weight,
            },
        },
    )?;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&output, serde_json::to_string_pretty(&report)?)?;
    println!(
        "wrote {} gate={:?} gate_passed={} robust_gate_passed={} strict_passed={} strict_score={:.6} catalog_sanity={} render_loss={:.6} density_psnr={:.3} active={}->{} newly_activated_fraction={:.3} opacity_max={:.3}",
        output.display(),
        report.gate,
        report.gate_passed,
        report.robustness.all_gate_passed,
        report.strict_passed,
        report.strict_score.score,
        report.catalog_sanity.passed,
        report.render_loss.total_loss,
        report.render_loss.density_psnr_db,
        report.activation.active_seed_count,
        report.activation.final_active_count,
        report.activation.newly_activated_fraction,
        report.final_opacity.max,
    );
    if fail_on_validation && !growth_3d_fail_on_validation_passed(&report) {
        return Err(std::io::Error::other(format!(
            "growth 3D validation failed; see {}",
            output.display()
        ))
        .into());
    }

    Ok(())
}

pub(crate) fn run_retime_growth_3d(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::RetimeGrowth3d {
        model,
        output,
        front_gain,
        hidden,
        skip_front_retime,
        active_opacity_gain,
        active_opacity_hidden,
        opacity_bias,
        material_opacity_bias,
        alpha,
    } = command
    else {
        unreachable!("run_retime_growth_3d called with the wrong command variant");
    };

    validate_diagnostic_3d_output_not_catalog(&output, "retime-growth3d")?;
    let manifest = crate::import::load_manifest(&model)?;
    let source = manifest.source.clone();
    let hashgrid = manifest.hashgrid.clone();
    let mut model_value = manifest.into_model();
    let hidden = if skip_front_retime {
        hidden
    } else {
        Some(retime_growth_3d_front_model(
            &mut model_value,
            hidden,
            front_gain,
        )?)
    };
    let active_opacity_hidden = if let Some(gain) = active_opacity_gain {
        Some(retime_growth_3d_active_opacity_model(
            &mut model_value,
            active_opacity_hidden,
            gain,
        )?)
    } else {
        None
    };
    if let Some(alpha) = alpha {
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(std::io::Error::other("alpha must be positive and finite").into());
        }
        model_value.config.alpha = alpha;
    }
    if let Some(opacity_bias) = opacity_bias {
        add_growth_3d_opacity_update_bias(&mut model_value, opacity_bias)?;
    }
    if let Some(material_opacity_bias) = material_opacity_bias {
        add_growth_3d_material_opacity_update_bias(&mut model_value, material_opacity_bias)?;
    }
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let retimed_source = Some(format!(
        "retimed-local-front:hidden={}:gain={front_gain}:alpha={}:front_retime={}:active_opacity_hidden={}:active_opacity_gain={}:opacity_bias={}:material_opacity_bias={}:base={}",
        hidden.map_or_else(|| "skipped".to_string(), |hidden| hidden.to_string()),
        model_value.config.alpha,
        !skip_front_retime,
        active_opacity_hidden.map_or_else(|| "skipped".to_string(), |hidden| hidden.to_string()),
        active_opacity_gain.map_or_else(|| "skipped".to_string(), |gain| gain.to_string()),
        opacity_bias.map_or_else(|| "skipped".to_string(), |bias| bias.to_string()),
        material_opacity_bias.map_or_else(|| "skipped".to_string(), |bias| bias.to_string()),
        source.as_deref().unwrap_or("unknown")
    ));
    let retimed_manifest = BpkModelManifest::from_model(&model_value, hashgrid, retimed_source);
    crate::import::save_manifest(&output, &retimed_manifest)?;
    println!(
        "wrote {} retimed_hidden={} front_gain={} alpha={} front_retime={} active_opacity_hidden={} active_opacity_gain={} opacity_bias={} material_opacity_bias={}",
        output.display(),
        hidden.map_or_else(|| "skipped".to_string(), |hidden| hidden.to_string()),
        front_gain,
        model_value.config.alpha,
        !skip_front_retime,
        active_opacity_hidden.map_or_else(|| "skipped".to_string(), |hidden| hidden.to_string()),
        active_opacity_gain.map_or_else(|| "skipped".to_string(), |gain| gain.to_string()),
        opacity_bias.map_or_else(|| "skipped".to_string(), |bias| bias.to_string()),
        material_opacity_bias.map_or_else(|| "skipped".to_string(), |bias| bias.to_string())
    );

    Ok(())
}

pub(crate) fn run_train_render_3d(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::TrainRender3d {
        target,
        base_model,
        model_output,
        report_output,
        rounds,
        supervised_steps_per_round,
        particles,
        rollout_steps,
        gradient_particles,
        gradient_mode,
        finite_diff_eps,
        motion_gain,
        perception_position_gain,
        max_update_norm,
        trajectory_supervision,
        trajectory_render_gain,
        trajectory_mesh_gain,
        trajectory_render_samples,
        liveness_gain,
        liveness_front_radius,
        liveness_update_multiplier,
        coverage_gain,
        coverage_samples,
        coverage_mode,
        coverage_softness,
        coverage_repulsion_gain,
        coverage_gap_gain,
        coverage_repulsion_radius,
        coverage_normal_weight,
        extent_gain,
        full_coverage_adjoint,
        no_full_coverage_adjoint,
        surface_gain,
        surface_escape_gain,
        opacity_gain,
        material_liveness_gain,
        material_tail_gain,
        material_suppression_update_multiplier,
        material_max_opacity_update,
        scale_gain,
        scale_budget_weight,
        max_opacity_update,
        learning_rate,
        grad_clip_norm,
        direct_output_gradient_rms_cap,
        direct_line_search,
        direct_line_search_scales,
        direct_material_output_only,
        training_backend,
        direct_selection_seed_training,
        no_direct_selection_seed_training,
        seed_scale,
        seed_mode,
        selection_seed,
        extra_selection_seeds,
        image_size,
        target_samples,
        sigma,
        min_sigma,
        max_sigma,
        gaussian_decode_mode,
        world_scale,
        render_opacity_logit_bias,
        density_weight,
        color_weight,
        depth_weight,
        fail_on_validation,
    } = command
    else {
        unreachable!("run_train_render_3d called with the wrong command variant");
    };

    let full_coverage_adjoint =
        resolve_full_coverage_adjoint(full_coverage_adjoint, no_full_coverage_adjoint)?;
    let direct_selection_seed_training = resolve_direct_selection_seed_training(
        direct_selection_seed_training,
        no_direct_selection_seed_training,
    )?;
    let hashgrid = crate::kernels::HashGridConfig::growing_3dgs();
    let requested_seed_mode = seed_mode.map(ParticleSeed::from);
    let target_mesh = mesh_target_for_arg(target, seed_scale);
    let (mut model, base_source, default_seed_mode) = if let Some(path) = base_model.as_ref() {
        let manifest = crate::import::load_manifest(path)?;
        let base_source = manifest.source.clone();
        let model = manifest.into_model();
        let default_seed_mode = default_render_training_seed_mode(target, &model);
        (model, base_source, default_seed_mode)
    } else {
        let default_seed_mode = render_training_default_seed_mode(target);
        let seed_mode = requested_seed_mode.unwrap_or(default_seed_mode);
        if !target_local_growth_seed(target, seed_mode) {
            return Err(std::io::Error::other(format!(
                        "train-render3d without --base-model defaults to conditionless-local growth and requires a target local growth seed; got seed_mode={seed_mode:?}"
                    ))
                    .into());
        }
        let (model, source) = render_training_base_model(target, &target_mesh, seed_mode)?;
        (model, Some(source), default_seed_mode)
    };
    let seed_mode = requested_seed_mode.unwrap_or(default_seed_mode);
    let catalog_bound_output = is_catalog_model_output_path(&model_output);
    validate_catalog_bound_render_training_output(
        &model_output,
        target,
        seed_mode,
        base_source.as_deref(),
    )?;
    let training_particles = render_training_particle_count_for_output(&model_output, particles);
    let training_rollout_steps =
        render_training_rollout_steps_for_output(&model_output, rollout_steps);
    let coverage_gap_gain = coverage_gap_gain.unwrap_or(coverage_repulsion_gain);
    let render = RenderLossConfig {
        image_size,
        sigma,
        min_sigma,
        max_sigma,
        gaussian_decode_mode: gaussian_decode_mode.into(),
        world_scale: world_scale.unwrap_or(seed_scale * 2.0),
        target_samples,
        opacity_logit_bias: render_opacity_logit_bias,
        density_weight,
        color_weight,
        depth_weight,
    };
    let training_selection_seeds =
        render_training_default_extra_selection_seeds(selection_seed, &extra_selection_seeds);
    let sgd = SgdConfig {
        learning_rate,
        grad_clip_norm,
        weight_decay: 0.0,
    };
    let report = run_render_proxy_training(
        &mut model,
        &hashgrid,
        &target_mesh,
        RenderProxyTrainingConfig {
            target,
            rounds,
            supervised_steps_per_round,
            particles: training_particles,
            rollout_steps: training_rollout_steps,
            gradient_particles,
            gradient_mode,
            finite_diff_eps,
            motion_gain,
            perception_position_gain,
            max_update_norm,
            trajectory_supervision,
            trajectory_render_gain,
            trajectory_mesh_gain,
            trajectory_render_samples,
            liveness_gain,
            liveness_front_radius,
            liveness_update_multiplier,
            coverage_gain,
            coverage_samples,
            coverage_mode,
            coverage_softness,
            coverage_repulsion_gain,
            coverage_gap_gain,
            coverage_repulsion_radius,
            coverage_normal_weight,
            extent_gain,
            full_coverage_adjoint,
            surface_gain,
            surface_escape_gain,
            opacity_gain,
            material_liveness_gain,
            material_tail_gain,
            material_suppression_update_multiplier,
            material_max_opacity_update,
            scale_gain,
            scale_budget_weight,
            max_opacity_update,
            direct_output_gradient_rms_cap,
            direct_line_search,
            direct_line_search_scales: direct_line_search_scales.clone(),
            direct_material_output_only,
            training_backend,
            direct_selection_seed_training,
            seed: 0x005a_173d,
            selection_seed: Some(selection_seed),
            selection_seeds: training_selection_seeds.clone(),
            seed_scale,
            seed_mode,
            render,
            sgd,
        },
    )?;
    if let Some(parent) = model_output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if let Some(parent) = report_output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let manifest = BpkModelManifest::from_model(
        &model,
        hashgrid.clone(),
        Some(render_training_source(
            target,
            base_source.as_deref(),
            seed_mode,
        )),
    );
    let validation_extra_seeds =
        render_training_validation_extra_seeds(selection_seed, &training_selection_seeds);
    let mut catalog_promotion_validations = Vec::new();
    if catalog_bound_output {
        let candidate_path = catalog_bound_candidate_path(target, std::process::id());
        if let Some(parent) = candidate_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        crate::import::save_manifest(&candidate_path, &manifest)?;
        let promotion_result = (|| -> Result<(), Box<dyn std::error::Error>> {
            for validation_cfg in catalog_promotion_validation_configs(
                selection_seed,
                &training_selection_seeds,
                seed_scale,
                seed_mode,
                render,
            ) {
                catalog_promotion_validations.push(growth_3d_validation_report(
                    &candidate_path,
                    target,
                    validation_cfg,
                )?);
            }
            require_catalog_promotion_validations_pass(
                &catalog_promotion_validations,
                &model_output,
            )
        })();
        if let Err(error) = promotion_result {
            std::fs::remove_file(&candidate_path).ok();
            return Err(error);
        }
        crate::import::save_manifest(&model_output, &manifest)?;
        std::fs::remove_file(&candidate_path).ok();
    } else {
        crate::import::save_manifest(&model_output, &manifest)?;
    }
    let loaded = crate::import::load_manifest(&model_output)?;
    let loaded_hashgrid = loaded.hashgrid.clone();
    let loaded_model = loaded.into_model();
    let growth_validation = growth_3d_validation_report(
        &model_output,
        target,
        Growth3dValidationConfig {
            particle_count: training_particles,
            steps: training_rollout_steps,
            seed: 0x005a_173d,
            extra_seeds: validation_extra_seeds,
            seed_scale,
            seed_mode,
            gate: Growth3dValidationGateArg::Strict,
            render,
        },
    )?;
    let final_render_loss = mesh_render_loss_for_model(
        &loaded_model,
        &loaded_hashgrid,
        &target_mesh,
        RenderLossEvalConfig {
            particle_count: training_particles,
            steps: training_rollout_steps,
            seed: 0x005a_173d,
            extra_seeds: Vec::new(),
            seed_scale,
            seed_mode,
            render,
        },
    )?;
    let strict_gate_summary = CliRenderTrainingGateSummary::from_validation(&growth_validation);
    let output_report = CliRenderTrainingReport {
        target,
        base_model: base_model.as_ref().map(|path| path.display().to_string()),
        model_output: model_output.display().to_string(),
        particle_count: training_particles,
        rollout_steps: training_rollout_steps,
        seed_scale,
        seed_mode,
        sgd,
        report,
        final_render_loss,
        strict_gate_summary,
        growth_validation,
        catalog_promotion_validations,
    };
    std::fs::write(
        &report_output,
        serde_json::to_string_pretty(&output_report)?,
    )?;
    println!(
        "wrote {} and {} render_loss={:.6} density_psnr={:.3} color_psnr={:.3} depth_psnr={:.3} passed={} strict_passed={} strict_score={:.3}",
        model_output.display(),
        report_output.display(),
        output_report.final_render_loss.total_loss,
        output_report.final_render_loss.density_psnr_db,
        output_report.final_render_loss.color_psnr_db,
        output_report.final_render_loss.depth_psnr_db,
        output_report.final_render_loss.passed,
        output_report.growth_validation.strict_passed,
        output_report.growth_validation.strict_score.score
    );
    if fail_on_validation && !growth_3d_fail_on_validation_passed(&output_report.growth_validation)
    {
        return Err(std::io::Error::other(format!(
                    "render-proxy training failed strict growth validation (score={:.6}, failures={:?}); see {}",
                    output_report.growth_validation.strict_score.score,
                    output_report.growth_validation.strict_checks.failure_reasons,
                    report_output.display(),
                ))
                .into());
    }

    Ok(())
}
