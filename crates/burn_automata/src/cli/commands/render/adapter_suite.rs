use crate::cli::prelude::*;

use super::super::resolve_direct_selection_seed_training;

pub(crate) fn run_train_render_3d_adapters(
    command: Command,
) -> Result<(), Box<dyn std::error::Error>> {
    let Command::TrainRender3dAdapters {
        base_model,
        shared_base_output,
        shared_base_cycles,
        shared_base_seed,
        targets,
        output_dir,
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
        training_backend,
        adapter_rank,
        adapter_alpha,
        adapter_seed,
        learning_rate,
        grad_clip_norm,
        direct_output_gradient_rms_cap,
        direct_line_search,
        direct_line_search_scales,
        direct_material_output_only,
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
        unreachable!("run_train_render_3d_adapters called with the wrong command variant");
    };

    if is_catalog_model_output_path(&output_dir) {
        return Err(std::io::Error::other(format!(
            "adapter suites write candidate artifacts only; output_dir {} must not be under assets/models",
            output_dir.display()
        ))
        .into());
    }
    let targets = unique_targets(targets)?;
    let shared_base_output =
        shared_base_output.unwrap_or_else(|| output_dir.join("shared_base.bpk"));
    if is_catalog_model_output_path(&shared_base_output) {
        return Err(std::io::Error::other(format!(
            "adapter suite shared base output {} must not be under assets/models",
            shared_base_output.display()
        ))
        .into());
    }
    let shared_base_cycles = shared_base_cycles.unwrap_or(usize::from(base_model.is_none()));
    let direct_selection_seed_training = resolve_direct_selection_seed_training(
        direct_selection_seed_training,
        no_direct_selection_seed_training,
    )?;
    let sgd = SgdConfig {
        learning_rate,
        grad_clip_norm,
        weight_decay: 0.0,
    };

    std::fs::create_dir_all(&output_dir)?;
    if let Some(parent) = shared_base_output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if let Some(parent) = report_output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let base_model_input = base_model.as_ref().map(|path| path.display().to_string());
    let (mut shared_model, hashgrid, loaded_base_source, shared_base_initialized) =
        if let Some(path) = base_model.as_ref() {
            let manifest = crate::import::load_manifest(path)?;
            let hashgrid = manifest.hashgrid.clone();
            let loaded_base_source = manifest.source.clone();
            (manifest.into_model(), hashgrid, loaded_base_source, false)
        } else {
            let hashgrid = crate::kernels::HashGridConfig::growing_3dgs();
            let model = local_growth_student_model(
                NpaConfig::growing_3dgs(),
                shared_base_seed,
                0.0,
                LOCAL_GROWTH_EXPANSION_GAIN,
            )?;
            (model, hashgrid, None, true)
        };
    let training_selection_seeds =
        render_training_default_extra_selection_seeds(selection_seed, &extra_selection_seeds);
    let mut shared_base_training = Vec::new();
    for cycle in 0..shared_base_cycles {
        for (target_index, target) in targets.iter().copied().enumerate() {
            let target_seed_scale =
                seed_scale.unwrap_or_else(|| mesh_target_render_training_seed_scale(target));
            let target_mesh = mesh_target_for_arg(target, target_seed_scale);
            let target_seed_mode = seed_mode
                .map(ParticleSeed::from)
                .unwrap_or_else(|| default_render_training_seed_mode(target, &shared_model));
            let render = RenderLossConfig {
                image_size,
                sigma,
                min_sigma,
                max_sigma,
                gaussian_decode_mode: gaussian_decode_mode.into(),
                world_scale: world_scale.unwrap_or(target_seed_scale * 2.0),
                target_samples,
                opacity_logit_bias: render_opacity_logit_bias,
                density_weight,
                color_weight,
                depth_weight,
            };
            let report = run_render_proxy_training(
                &mut shared_model,
                &hashgrid,
                &target_mesh,
                RenderProxyTrainingConfig {
                    target,
                    rounds: 1,
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
                    trajectory_render_gain: ROBUST_3D_TRAJECTORY_RENDER_GAIN,
                    trajectory_mesh_gain: ROBUST_3D_TRAJECTORY_MESH_GAIN,
                    trajectory_render_samples: ROBUST_3D_TRAJECTORY_RENDER_SAMPLES,
                    liveness_gain: ROBUST_3D_LIVENESS_GAIN,
                    liveness_front_radius: ROBUST_3D_LIVENESS_FRONT_RADIUS,
                    liveness_update_multiplier: ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER,
                    coverage_gain: ROBUST_3D_COVERAGE_GAIN,
                    coverage_samples: ROBUST_3D_COVERAGE_SAMPLES,
                    coverage_mode: CoverageUpdateModeArg::SlicedOt,
                    coverage_softness: 0.0,
                    coverage_repulsion_gain: ROBUST_3D_COVERAGE_REPULSION_GAIN,
                    coverage_gap_gain: ROBUST_3D_COVERAGE_REPULSION_GAIN,
                    coverage_repulsion_radius: 0.0,
                    coverage_normal_weight: ROBUST_3D_COVERAGE_NORMAL_WEIGHT,
                    extent_gain: ROBUST_3D_EXTENT_GAIN,
                    full_coverage_adjoint: true,
                    surface_gain: ROBUST_3D_SURFACE_GAIN,
                    surface_escape_gain: ROBUST_3D_SURFACE_ESCAPE_GAIN,
                    opacity_gain: ROBUST_3D_OPACITY_GAIN,
                    material_liveness_gain: ROBUST_3D_MATERIAL_LIVENESS_GAIN,
                    material_tail_gain: ROBUST_3D_MATERIAL_TAIL_GAIN,
                    material_suppression_update_multiplier:
                        ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
                    material_max_opacity_update: ROBUST_3D_MATERIAL_MAX_OPACITY_UPDATE,
                    scale_gain: ROBUST_3D_SCALE_GAIN,
                    scale_budget_weight: ROBUST_3D_SCALE_BUDGET_WEIGHT,
                    max_opacity_update: 0.05,
                    direct_output_gradient_rms_cap,
                    direct_line_search,
                    direct_line_search_scales: direct_line_search_scales.clone(),
                    direct_material_output_only,
                    training_backend,
                    weight_update_mode: RenderWeightUpdateModeArg::Full,
                    adapter_rank,
                    adapter_alpha,
                    adapter_seed: adapter_seed.wrapping_add(target_index as u64),
                    direct_selection_seed_training,
                    seed: shared_base_seed
                        .wrapping_add((cycle * targets.len() + target_index) as u64),
                    selection_seed: Some(selection_seed),
                    selection_seeds: training_selection_seeds.clone(),
                    seed_scale: target_seed_scale,
                    seed_mode: target_seed_mode,
                    render,
                    sgd,
                },
            )?;
            shared_base_training.push(CliRenderAdapterSuiteBaseEntry {
                cycle,
                target,
                seed_scale: target_seed_scale,
                seed_mode: target_seed_mode,
                report,
            });
        }
    }
    let base_source = if shared_base_initialized || shared_base_cycles > 0 {
        Some(shared_render_adapter_base_source(
            &targets,
            shared_base_cycles,
        ))
    } else {
        loaded_base_source
    };
    let base_manifest =
        BpkModelManifest::from_model(&shared_model, hashgrid.clone(), base_source.clone());
    crate::import::save_manifest(&shared_base_output, &base_manifest)?;
    let base_parameter_count = crate::import::parameter_count(&base_manifest);

    let mut entries = Vec::with_capacity(targets.len());
    for (target_index, target) in targets.iter().copied().enumerate() {
        let target_seed_scale =
            seed_scale.unwrap_or_else(|| mesh_target_render_training_seed_scale(target));
        let target_mesh = mesh_target_for_arg(target, target_seed_scale);
        let base_npa = base_manifest.clone().into_model();
        let target_seed_mode = seed_mode
            .map(ParticleSeed::from)
            .unwrap_or_else(|| default_render_training_seed_mode(target, &base_npa));
        let training_selection_seeds =
            render_training_default_extra_selection_seeds(selection_seed, &extra_selection_seeds);
        let render = RenderLossConfig {
            image_size,
            sigma,
            min_sigma,
            max_sigma,
            gaussian_decode_mode: gaussian_decode_mode.into(),
            world_scale: world_scale.unwrap_or(target_seed_scale * 2.0),
            target_samples,
            opacity_logit_bias: render_opacity_logit_bias,
            density_weight,
            color_weight,
            depth_weight,
        };
        let mut model = base_npa;
        let target_adapter_seed = adapter_seed.wrapping_add(target_index as u64);
        let report = run_render_proxy_training(
            &mut model,
            &hashgrid,
            &target_mesh,
            RenderProxyTrainingConfig {
                target,
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
                trajectory_render_gain: ROBUST_3D_TRAJECTORY_RENDER_GAIN,
                trajectory_mesh_gain: ROBUST_3D_TRAJECTORY_MESH_GAIN,
                trajectory_render_samples: ROBUST_3D_TRAJECTORY_RENDER_SAMPLES,
                liveness_gain: ROBUST_3D_LIVENESS_GAIN,
                liveness_front_radius: ROBUST_3D_LIVENESS_FRONT_RADIUS,
                liveness_update_multiplier: ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER,
                coverage_gain: ROBUST_3D_COVERAGE_GAIN,
                coverage_samples: ROBUST_3D_COVERAGE_SAMPLES,
                coverage_mode: CoverageUpdateModeArg::SlicedOt,
                coverage_softness: 0.0,
                coverage_repulsion_gain: ROBUST_3D_COVERAGE_REPULSION_GAIN,
                coverage_gap_gain: ROBUST_3D_COVERAGE_REPULSION_GAIN,
                coverage_repulsion_radius: 0.0,
                coverage_normal_weight: ROBUST_3D_COVERAGE_NORMAL_WEIGHT,
                extent_gain: ROBUST_3D_EXTENT_GAIN,
                full_coverage_adjoint: true,
                surface_gain: ROBUST_3D_SURFACE_GAIN,
                surface_escape_gain: ROBUST_3D_SURFACE_ESCAPE_GAIN,
                opacity_gain: ROBUST_3D_OPACITY_GAIN,
                material_liveness_gain: ROBUST_3D_MATERIAL_LIVENESS_GAIN,
                material_tail_gain: ROBUST_3D_MATERIAL_TAIL_GAIN,
                material_suppression_update_multiplier:
                    ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
                material_max_opacity_update: ROBUST_3D_MATERIAL_MAX_OPACITY_UPDATE,
                scale_gain: ROBUST_3D_SCALE_GAIN,
                scale_budget_weight: ROBUST_3D_SCALE_BUDGET_WEIGHT,
                max_opacity_update: 0.05,
                direct_output_gradient_rms_cap,
                direct_line_search,
                direct_line_search_scales: direct_line_search_scales.clone(),
                direct_material_output_only,
                training_backend,
                weight_update_mode: RenderWeightUpdateModeArg::Adapter,
                adapter_rank,
                adapter_alpha,
                adapter_seed: target_adapter_seed,
                direct_selection_seed_training,
                seed: 0x005a_173d,
                selection_seed: Some(selection_seed),
                selection_seeds: training_selection_seeds.clone(),
                seed_scale: target_seed_scale,
                seed_mode: target_seed_mode,
                render,
                sgd,
            },
        )?;

        let adapter = report.trained_adapter.clone().ok_or_else(|| {
            std::io::Error::other("adapter suite training did not produce an adapter")
        })?;
        let slug = mesh_target_slug(target);
        let adapter_output = output_dir.join(format!("{slug}.adapter.json"));
        let materialized_model_output = output_dir.join(format!("{slug}_materialized.bpk"));
        let adapter_manifest = crate::import::BpkAdapterManifest::from_adapter(
            &base_manifest,
            Some(shared_base_output.display().to_string()),
            adapter,
            Some(render_adapter_training_source(
                target,
                base_source.as_deref(),
                target_seed_mode,
            )),
        )?;
        crate::import::save_adapter_manifest(&adapter_output, &adapter_manifest)?;
        let materialized_manifest = adapter_manifest.materialize(&base_manifest)?;
        crate::import::save_manifest(&materialized_model_output, &materialized_manifest)?;

        let loaded = crate::import::load_manifest(&materialized_model_output)?;
        let loaded_hashgrid = loaded.hashgrid.clone();
        let loaded_model = loaded.into_model();
        let validation_extra_seeds =
            render_training_validation_extra_seeds(selection_seed, &training_selection_seeds);
        let growth_validation = growth_3d_validation_report(
            &materialized_model_output,
            target,
            Growth3dValidationConfig {
                particle_count: particles,
                steps: rollout_steps,
                seed: 0x005a_173d,
                extra_seeds: validation_extra_seeds,
                seed_scale: target_seed_scale,
                seed_mode: target_seed_mode,
                gate: Growth3dValidationGateArg::Strict,
                render,
            },
        )?;
        let final_render_loss = mesh_render_loss_for_model(
            &loaded_model,
            &loaded_hashgrid,
            &target_mesh,
            RenderLossEvalConfig {
                particle_count: particles,
                steps: rollout_steps,
                seed: 0x005a_173d,
                extra_seeds: Vec::new(),
                seed_scale: target_seed_scale,
                seed_mode: target_seed_mode,
                render,
            },
        )?;
        let strict_gate_summary = CliRenderTrainingGateSummary::from_validation(&growth_validation);
        entries.push(CliRenderAdapterSuiteEntry {
            target,
            adapter_output: adapter_output.display().to_string(),
            materialized_model_output: materialized_model_output.display().to_string(),
            seed_scale: target_seed_scale,
            seed_mode: target_seed_mode,
            report,
            final_render_loss,
            strict_gate_summary,
            growth_validation,
        });
    }

    let adapter_parameter_count = entries
        .first()
        .map(|entry| entry.report.weight_update.adapter_parameter_count)
        .unwrap_or(0);
    let materialized_parameter_count = entries
        .first()
        .map(|entry| entry.report.weight_update.materialized_parameter_count)
        .unwrap_or(base_parameter_count);
    let adapter_to_full_ratio = if materialized_parameter_count == 0 {
        0.0
    } else {
        adapter_parameter_count as f32 / materialized_parameter_count as f32
    };
    let suite_report = CliRenderAdapterSuiteReport {
        base_model_input,
        base_model: shared_base_output.display().to_string(),
        base_source,
        shared_base_initialized,
        shared_base_cycles,
        shared_base_training,
        output_dir: output_dir.display().to_string(),
        targets,
        particle_count: particles,
        rollout_steps,
        sgd,
        adapter_rank,
        adapter_alpha,
        base_parameter_count,
        materialized_parameter_count,
        adapter_parameter_count,
        adapter_to_full_ratio,
        entries,
    };
    std::fs::write(&report_output, serde_json::to_string_pretty(&suite_report)?)?;

    let failed_targets = suite_report
        .entries
        .iter()
        .filter(|entry| !growth_3d_fail_on_validation_passed(&entry.growth_validation))
        .map(|entry| mesh_target_slug(entry.target))
        .collect::<Vec<_>>();
    println!(
        "wrote {} with {} adapters adapter/full={:.4} failed_targets={:?}",
        report_output.display(),
        suite_report.entries.len(),
        suite_report.adapter_to_full_ratio,
        failed_targets
    );
    if fail_on_validation && !failed_targets.is_empty() {
        return Err(std::io::Error::other(format!(
            "adapter suite failed strict growth validation for {failed_targets:?}; see {}",
            report_output.display()
        ))
        .into());
    }

    Ok(())
}

fn unique_targets(
    targets: Vec<MeshTargetArg>,
) -> Result<Vec<MeshTargetArg>, Box<dyn std::error::Error>> {
    if targets.is_empty() {
        return Err(std::io::Error::other("adapter suite requires at least one target").into());
    }
    let mut unique = Vec::with_capacity(targets.len());
    for target in targets {
        if !unique.contains(&target) {
            unique.push(target);
        }
    }
    Ok(unique)
}
