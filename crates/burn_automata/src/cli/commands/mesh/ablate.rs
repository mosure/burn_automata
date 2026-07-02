use crate::cli::prelude::*;

pub(crate) fn run_ablate_local_3d(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::AblateLocal3d {
        target,
        base_model,
        model_output,
        report_output,
        rows,
        steps,
        rollout_particles,
        rollout_steps,
        rollouts,
        temporal_samples,
        training_rounds,
        seed_scale,
        seed_mode,
        student_seed,
        learning_rate,
        grad_clip_norm,
        weight_decay,
        motion_gain,
        max_update_norm,
        density_gain,
        expansion_gain,
        coverage_gain,
        coverage_samples,
        coverage_mode,
        coverage_softness,
        coverage_repulsion_gain,
        coverage_gap_gain,
        coverage_repulsion_radius,
        coverage_normal_weight,
        extent_gain,
        color_gain,
        aux_state_gain,
        opacity_gain,
        front_opacity_gain,
        front_radius,
        front_max_opacity_update,
        front_motion_gate,
        preserve_opacity_update,
        fail_on_validation,
    } = command
    else {
        unreachable!("run_ablate_local_3d called with the wrong command variant");
    };

    validate_diagnostic_3d_output_not_catalog(&model_output, "ablate-local-3d")?;
    let target_mesh = mesh_target_for_arg(target, seed_scale);
    let seed_mode = seed_mode
        .map(ParticleSeed::from)
        .unwrap_or_else(|| conditionless_local_seed_mode(target));
    let target_source = mesh_conditionless_local_target_source_for_seed(target, seed_mode);
    let (mut model, hashgrid, output_source) = if let Some(path) = base_model.as_ref() {
        load_conditionless_local_base_model(path, target, target_source)?
    } else {
        let config = NpaConfig::growing_3dgs();
        let hashgrid = crate::kernels::HashGridConfig::growing_3dgs();
        let model = local_growth_student_model_with_axis_gains(
            config,
            student_seed,
            density_gain,
            mesh_axis_expansion_gains(&target_mesh, expansion_gain),
        )?;
        (model, hashgrid, format!("ablation-rust:{target_source}"))
    };
    let sgd = SgdConfig {
        learning_rate,
        grad_clip_norm,
        weight_decay,
    };
    let preserve_opacity_update =
        preserve_opacity_update || (opacity_gain == 0.0 && front_opacity_gain == 0.0);
    let coverage_gap_gain = coverage_gap_gain.unwrap_or(coverage_repulsion_gain);
    let report = run_refreshed_mesh_local_training(
        &mut model,
        &hashgrid,
        &target_mesh,
        MeshLocalTrainingConfig {
            max_rows: rows,
            particle_count: rollout_particles,
            rollout_steps,
            rollouts,
            temporal_samples,
            training_rounds,
            total_steps: steps,
            seed: student_seed ^ 0x005e_ed3d,
            seed_scale,
            seed_mode,
            motion_gain: motion_gain.unwrap_or_else(|| mesh_target_motion_gain(target)),
            max_update_norm,
            coverage_gain,
            coverage_samples,
            coverage_mode,
            coverage_softness,
            coverage_repulsion_gain,
            coverage_gap_gain,
            coverage_repulsion_radius,
            coverage_normal_weight,
            extent_gain,
            color_gain: color_gain.unwrap_or_else(|| mesh_target_color_gain(target)),
            aux_state_gain,
            opacity_gain,
            front_opacity_gain,
            front_radius,
            front_max_opacity_update,
            front_motion_gate,
            preserve_opacity_update,
            sgd,
        },
    )?;
    if let Some(parent) = model_output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if let Some(parent) = report_output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let manifest =
        BpkModelManifest::from_model(&model, hashgrid.clone(), Some(output_source.clone()));
    crate::import::save_manifest(&model_output, &manifest)?;
    let loaded = crate::import::load_manifest(&model_output)?;
    let loaded_hashgrid = loaded.hashgrid.clone();
    let loaded_model = loaded.into_model();
    let validation_cases = conditionless_local_rollout_cases(target, seed_scale, rollout_particles);
    let mesh_rollout = Some(mesh_rollout_report_for_cases(
        &loaded_model,
        &loaded_hashgrid,
        &target_mesh,
        &validation_cases,
    )?);
    let render_loss = Some(mesh_render_loss_for_model(
        &loaded_model,
        &loaded_hashgrid,
        &target_mesh,
        RenderLossEvalConfig {
            particle_count: rollout_particles,
            steps: 64,
            seed: 0x010c_a202,
            extra_seeds: Vec::new(),
            seed_scale,
            seed_mode,
            render: default_render_loss_config(seed_scale),
        },
    )?);
    let rollout_supervision = Some(CliRolloutSupervisionReport {
        particle_count: rollout_particles,
        rollout_steps,
        rollouts,
        temporal_samples,
        update_prob: 1.0,
        seed_scale,
        seed_mode,
        motion_gain: Some(motion_gain.unwrap_or_else(|| mesh_target_motion_gain(target))),
        max_update_norm: Some(max_update_norm),
        density_gain: Some(density_gain),
        expansion_gain: Some(expansion_gain),
        coverage_gain: Some(coverage_gain),
        coverage_samples: Some(coverage_samples),
        coverage_mode: Some(coverage_mode),
        coverage_softness: Some(coverage_softness),
        coverage_repulsion_gain: Some(coverage_repulsion_gain),
        coverage_gap_gain: Some(coverage_gap_gain),
        coverage_repulsion_radius: Some(coverage_repulsion_radius),
        coverage_normal_weight: Some(coverage_normal_weight),
        extent_gain: Some(extent_gain),
        color_gain: Some(color_gain.unwrap_or_else(|| mesh_target_color_gain(target))),
        aux_state_gain: Some(aux_state_gain),
        opacity_gain: Some(opacity_gain),
        front_opacity_gain: Some(front_opacity_gain),
        front_radius: Some(front_radius),
        front_max_opacity_update: Some(front_max_opacity_update),
        front_motion_gate: Some(front_motion_gate),
        preserve_opacity_update: Some(preserve_opacity_update),
    });
    let output_report = CliTrainingReport {
        preset: AutomataPreset::Growing3dGs,
        target_source: output_source,
        student_seed,
        sgd,
        report,
        model_output: Some(model_output.display().to_string()),
        batch_source: TrainingBatchArg::Rollout,
        rollout_supervision,
        mesh_rollout,
        render_loss,
    };
    std::fs::write(
        &report_output,
        serde_json::to_string_pretty(&output_report)?,
    )?;
    let mesh_status = output_report
        .mesh_rollout
        .as_ref()
        .map_or(
            "skipped",
            |report| if report.passed { "passed" } else { "failed" },
        );
    let render_status = output_report
        .render_loss
        .as_ref()
        .map_or(
            "skipped",
            |report| if report.passed { "passed" } else { "failed" },
        );
    println!(
        "wrote {} and {} final_loss={:.6} mesh_rollout={mesh_status} render_loss={render_status}",
        model_output.display(),
        report_output.display(),
        output_report.report.final_loss
    );
    if fail_on_validation
        && output_report
            .mesh_rollout
            .as_ref()
            .is_some_and(|report| !report.passed)
    {
        return Err(std::io::Error::other(format!(
            "conditionless local 3d ablation failed validation; see {}",
            report_output.display()
        ))
        .into());
    }
    if fail_on_validation
        && output_report
            .render_loss
            .as_ref()
            .is_some_and(|report| !report.passed)
    {
        return Err(std::io::Error::other(format!(
            "conditionless local 3d render validation failed; see {}",
            report_output.display()
        ))
        .into());
    }

    Ok(())
}
