use crate::cli::prelude::*;

use super::super::{resolve_direct_selection_seed_training, resolve_full_coverage_adjoint};

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
    let seed_scale = seed_scale.unwrap_or_else(|| mesh_target_render_training_seed_scale(target));
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
        if !target_strict_conditionless_local_growth_seed(target, seed_mode) {
            return Err(std::io::Error::other(format!(
                "train-render3d without --base-model defaults to conditionless-local growth and requires the target strict conditionless-local growth seed; got seed_mode={seed_mode:?}"
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
    let candidate_path =
        save_render_training_manifest_for_validation(&model_output, &manifest, target)?;
    let validation_model_path = candidate_path.as_ref().unwrap_or(&model_output);
    let mut catalog_promotion_validations = Vec::new();
    if catalog_bound_output {
        for validation_cfg in catalog_promotion_validation_configs(
            selection_seed,
            &training_selection_seeds,
            seed_scale,
            seed_mode,
            render,
        ) {
            catalog_promotion_validations.push(growth_3d_validation_report(
                validation_model_path,
                target,
                validation_cfg,
            )?);
        }
    }
    let loaded = crate::import::load_manifest(validation_model_path)?;
    let loaded_hashgrid = loaded.hashgrid.clone();
    let loaded_model = loaded.into_model();
    let growth_validation = growth_3d_validation_report(
        validation_model_path,
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
    let promotion_error = if catalog_bound_output {
        require_catalog_promotion_validations_pass(&catalog_promotion_validations, &model_output)
            .err()
    } else {
        None
    };
    let promotion_rejection_reason = promotion_error.as_ref().map(|error| error.to_string());
    let catalog_promotion = CliCatalogPromotionSummary::from_validation_result(
        catalog_bound_output,
        catalog_promotion_validations.len(),
        promotion_rejection_reason,
    );
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
        catalog_promotion,
        catalog_promotion_validations,
    };
    std::fs::write(
        &report_output,
        serde_json::to_string_pretty(&output_report)?,
    )?;
    if let Some(error) = promotion_error {
        println!(
            "wrote {} for rejected catalog candidate targeting {} render_loss={:.6} density_psnr={:.3} color_psnr={:.3} depth_psnr={:.3} passed={} strict_passed={} strict_score={:.3}",
            report_output.display(),
            model_output.display(),
            output_report.final_render_loss.total_loss,
            output_report.final_render_loss.density_psnr_db,
            output_report.final_render_loss.color_psnr_db,
            output_report.final_render_loss.depth_psnr_db,
            output_report.final_render_loss.passed,
            output_report.growth_validation.strict_passed,
            output_report.growth_validation.strict_score.score
        );
        return finalize_render_training_manifest_promotion(
            &model_output,
            &manifest,
            candidate_path.as_deref(),
            Some(error),
        );
    }
    finalize_render_training_manifest_promotion(
        &model_output,
        &manifest,
        candidate_path.as_deref(),
        None,
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
