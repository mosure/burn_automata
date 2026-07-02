use crate::cli::prelude::*;

mod train;

pub(crate) use train::run_train_render_3d;

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
