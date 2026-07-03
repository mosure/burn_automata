use crate::cli::prelude::*;

use super::super::resolve_direct_selection_seed_training;

mod bank;
mod config;
mod contract;
mod evaluation;
mod signal;
mod splits;
mod summary;

use bank::adapter_suite_bank_entries;
use config::{
    AdapterSuiteRenderSettings, AdapterSuiteTrainingPhaseConfig, AdapterSuiteTrainingSettings,
};
use contract::adapter_suite_contract;
use evaluation::adapter_suite_shared_base_evaluations;
use signal::{adapter_suite_missing_signal_labels, adapter_suite_missing_train_signal};
use splits::{
    adapter_suite_auto_holdout_targets, adapter_suite_holdout_targets, adapter_suite_split,
    default_adapter_suite_shared_base_cycles, effective_adapter_suite_auto_holdout_stride,
    resolve_adapter_suite_targets, suite_report_holdout_target_count,
    suite_report_shared_base_target_count, validate_holdout_targets,
};
use summary::{
    adapter_suite_adapter_summary, adapter_suite_shared_base_summary, adapter_suite_split_summaries,
};

pub(crate) fn run_train_render_3d_adapters(
    command: Command,
) -> Result<(), Box<dyn std::error::Error>> {
    let Command::TrainRender3dAdapters {
        base_model,
        shared_base_output,
        shared_base_cycles,
        shared_base_seed,
        target_set,
        targets,
        holdout_targets,
        auto_holdout_stride,
        auto_holdout_offset,
        output_dir,
        report_output,
        adapter_bank_output,
        skip_shared_base_eval,
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
    let explicit_targets_requested = !targets.is_empty();
    let manual_holdout_targets = holdout_targets;
    let targets = resolve_adapter_suite_targets(targets, target_set)?;
    let auto_holdout_stride = effective_adapter_suite_auto_holdout_stride(
        auto_holdout_stride,
        explicit_targets_requested,
        target_set,
        &manual_holdout_targets,
        targets.len(),
    );
    let auto_holdout_targets =
        adapter_suite_auto_holdout_targets(&targets, auto_holdout_stride, auto_holdout_offset)?;
    let holdout_targets =
        adapter_suite_holdout_targets(manual_holdout_targets, auto_holdout_targets.clone());
    validate_holdout_targets(&targets, &holdout_targets)?;
    let shared_base_targets = targets
        .iter()
        .copied()
        .filter(|target| !holdout_targets.contains(target))
        .collect::<Vec<_>>();
    let shared_base_output =
        shared_base_output.unwrap_or_else(|| output_dir.join("shared_base.bpk"));
    if is_catalog_model_output_path(&shared_base_output) {
        return Err(std::io::Error::other(format!(
            "adapter suite shared base output {} must not be under assets/models",
            shared_base_output.display()
        ))
        .into());
    }
    let shared_base_cycles = shared_base_cycles.unwrap_or_else(|| {
        default_adapter_suite_shared_base_cycles(base_model.is_some(), target_set, targets.len())
    });
    if shared_base_cycles > 0 && shared_base_targets.is_empty() {
        return Err(std::io::Error::other(
            "adapter suite shared-base training requires at least one non-holdout target",
        )
        .into());
    }
    let direct_selection_seed_training = resolve_direct_selection_seed_training(
        direct_selection_seed_training,
        no_direct_selection_seed_training,
    )?;
    let sgd = SgdConfig {
        learning_rate,
        grad_clip_norm,
        weight_decay: 0.0,
    };
    let render_settings = AdapterSuiteRenderSettings {
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
    };
    let training_settings = AdapterSuiteTrainingSettings {
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
        direct_output_gradient_rms_cap,
        direct_line_search,
        direct_line_search_scales,
        direct_material_output_only,
        direct_selection_seed_training,
        selection_seed,
        selection_seeds: render_training_default_extra_selection_seeds(
            selection_seed,
            &extra_selection_seeds,
        ),
        sgd,
        adapter_rank,
        adapter_alpha,
    };

    std::fs::create_dir_all(&output_dir)?;
    if let Some(parent) = shared_base_output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if let Some(parent) = report_output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let adapter_bank_output =
        adapter_bank_output.unwrap_or_else(|| output_dir.join("adapter_bank.json"));
    if let Some(parent) = adapter_bank_output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    println!(
        "adapter suite target_set={:?} target_count={} shared_base_target_count={} holdout_target_count={} shared_base_cycles={} adapter_target_count={} targets={:?} shared_base_targets={:?} holdout_targets={:?} adapter_rank={} adapter_bank={}",
        target_set,
        targets.len(),
        shared_base_targets.len(),
        holdout_targets.len(),
        shared_base_cycles,
        targets.len(),
        targets
            .iter()
            .copied()
            .map(mesh_target_slug)
            .collect::<Vec<_>>(),
        shared_base_targets
            .iter()
            .copied()
            .map(mesh_target_slug)
            .collect::<Vec<_>>(),
        holdout_targets
            .iter()
            .copied()
            .map(mesh_target_slug)
            .collect::<Vec<_>>(),
        adapter_rank,
        adapter_bank_output.display()
    );

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
    let training_selection_seeds = training_settings.selection_seeds.clone();
    let mut shared_base_training = Vec::new();
    for cycle in 0..shared_base_cycles {
        for (target_index, target) in shared_base_targets.iter().copied().enumerate() {
            let target_seed_scale =
                seed_scale.unwrap_or_else(|| mesh_target_render_training_seed_scale(target));
            let target_mesh = mesh_target_for_arg(target, target_seed_scale);
            let target_seed_mode = seed_mode
                .map(ParticleSeed::from)
                .unwrap_or_else(|| default_render_training_seed_mode(target, &shared_model));
            let render = render_settings.loss_config(target_seed_scale);
            let report = run_render_proxy_training(
                &mut shared_model,
                &hashgrid,
                &target_mesh,
                training_settings.render_proxy_config(
                    AdapterSuiteTrainingPhaseConfig {
                        target,
                        rounds: 1,
                        weight_update_mode: RenderWeightUpdateModeArg::Full,
                        adapter_seed: adapter_seed.wrapping_add(target_index as u64),
                        seed: shared_base_seed.wrapping_add(
                            (cycle * shared_base_targets.len() + target_index) as u64,
                        ),
                        seed_scale: target_seed_scale,
                        seed_mode: target_seed_mode,
                    },
                    render,
                ),
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
            &shared_base_targets,
            shared_base_cycles,
        ))
    } else {
        loaded_base_source
    };
    let base_manifest =
        BpkModelManifest::from_model(&shared_model, hashgrid.clone(), base_source.clone());
    crate::import::save_manifest(&shared_base_output, &base_manifest)?;
    let base_parameter_count = crate::import::parameter_count(&base_manifest);
    let shared_base_evaluations = if skip_shared_base_eval {
        Vec::new()
    } else {
        adapter_suite_shared_base_evaluations(
            &shared_base_output,
            &base_manifest,
            &targets,
            &holdout_targets,
            seed_scale,
            seed_mode,
            particles,
            rollout_steps,
            selection_seed,
            &training_selection_seeds,
            render_settings,
        )?
    };

    let mut entries = Vec::with_capacity(targets.len());
    for (target_index, target) in targets.iter().copied().enumerate() {
        let target_seed_scale =
            seed_scale.unwrap_or_else(|| mesh_target_render_training_seed_scale(target));
        let target_mesh = mesh_target_for_arg(target, target_seed_scale);
        let base_npa = base_manifest.clone().into_model();
        let target_seed_mode = seed_mode
            .map(ParticleSeed::from)
            .unwrap_or_else(|| default_render_training_seed_mode(target, &base_npa));
        let render = render_settings.loss_config(target_seed_scale);
        let mut model = base_npa;
        let target_adapter_seed = adapter_seed.wrapping_add(target_index as u64);
        let report = run_render_proxy_training(
            &mut model,
            &hashgrid,
            &target_mesh,
            training_settings.render_proxy_config(
                AdapterSuiteTrainingPhaseConfig {
                    target,
                    rounds,
                    weight_update_mode: RenderWeightUpdateModeArg::Adapter,
                    adapter_seed: target_adapter_seed,
                    seed: 0x005a_173d,
                    seed_scale: target_seed_scale,
                    seed_mode: target_seed_mode,
                },
                render,
            ),
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
            split: adapter_suite_split(target, &holdout_targets),
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
    let target_count = entries.len();
    let adapter_total_parameter_count = adapter_parameter_count * target_count;
    let full_bank_parameter_count = materialized_parameter_count * target_count;
    let shared_plus_adapter_parameter_count = base_parameter_count + adapter_total_parameter_count;
    let shared_plus_adapter_to_full_bank_ratio = if full_bank_parameter_count == 0 {
        0.0
    } else {
        shared_plus_adapter_parameter_count as f32 / full_bank_parameter_count as f32
    };
    let shared_plus_adapter_savings_ratio = 1.0 - shared_plus_adapter_to_full_bank_ratio;
    let missing_train_signal = adapter_suite_missing_train_signal(&shared_base_training, &entries);
    let training_signal_passed = missing_train_signal.is_empty();
    let shared_base_summary = adapter_suite_shared_base_summary(&shared_base_evaluations);
    let adapter_summary = adapter_suite_adapter_summary(&entries);
    let split_summaries = adapter_suite_split_summaries(&shared_base_evaluations, &entries);
    let adapter_training_targets = entries.iter().map(|entry| entry.target).collect::<Vec<_>>();
    let strategy = CliRenderAdapterSuiteStrategy::SharedBaseLowRankObjectAdapters;
    let contract = adapter_suite_contract(
        target_set,
        explicit_targets_requested,
        &targets,
        &shared_base_targets,
        &holdout_targets,
        &adapter_training_targets,
    );
    let shared_base_training_visit_count = shared_base_training.len();
    let adapter_bank_manifest = CliRenderAdapterBankManifest {
        schema_version: CLI_RENDER_ADAPTER_BANK_SCHEMA_VERSION,
        strategy,
        contract: contract.clone(),
        base_model: shared_base_output.display().to_string(),
        base_source: base_source.clone(),
        target_set,
        targets: targets.clone(),
        shared_base_targets: shared_base_targets.clone(),
        holdout_targets: holdout_targets.clone(),
        target_count,
        shared_base_target_count: shared_base_targets.len(),
        holdout_target_count: holdout_targets.len(),
        adapter_target_count: target_count,
        adapter_rank,
        adapter_alpha,
        base_parameter_count,
        materialized_parameter_count,
        adapter_parameter_count,
        adapter_to_full_ratio,
        shared_plus_adapter_to_full_bank_ratio,
        entries: adapter_suite_bank_entries(&entries),
    };
    std::fs::write(
        &adapter_bank_output,
        serde_json::to_string_pretty(&adapter_bank_manifest)?,
    )?;
    let suite_report = CliRenderAdapterSuiteReport {
        schema_version: CLI_RENDER_ADAPTER_SUITE_REPORT_SCHEMA_VERSION,
        strategy,
        contract,
        base_model_input,
        base_model: shared_base_output.display().to_string(),
        base_source,
        shared_base_initialized,
        shared_base_cycles,
        shared_base_training,
        shared_base_eval_enabled: !skip_shared_base_eval,
        shared_base_evaluations,
        output_dir: output_dir.display().to_string(),
        adapter_bank_manifest: adapter_bank_output.display().to_string(),
        target_set,
        targets,
        shared_base_targets: shared_base_targets.clone(),
        adapter_training_targets,
        auto_holdout_stride,
        auto_holdout_offset,
        auto_holdout_targets,
        holdout_targets,
        particle_count: particles,
        rollout_steps,
        sgd,
        adapter_rank,
        adapter_alpha,
        base_parameter_count,
        materialized_parameter_count,
        adapter_parameter_count,
        adapter_to_full_ratio,
        target_count,
        shared_base_target_count: suite_report_shared_base_target_count(&entries),
        holdout_target_count: suite_report_holdout_target_count(&entries),
        shared_base_training_visit_count,
        adapter_training_target_count: entries.len(),
        adapter_total_parameter_count,
        full_bank_parameter_count,
        shared_plus_adapter_parameter_count,
        shared_plus_adapter_to_full_bank_ratio,
        shared_plus_adapter_savings_ratio,
        shared_base_summary,
        adapter_summary,
        split_summaries,
        training_signal_passed,
        missing_train_signal,
        entries,
    };
    std::fs::write(&report_output, serde_json::to_string_pretty(&suite_report)?)?;

    let failed_targets = suite_report
        .entries
        .iter()
        .filter(|entry| !growth_3d_fail_on_validation_passed(&entry.growth_validation))
        .map(|entry| mesh_target_slug(entry.target))
        .collect::<Vec<_>>();
    let missing_signal_labels =
        adapter_suite_missing_signal_labels(&suite_report.missing_train_signal);
    println!(
        "wrote {} with {} adapters adapter/full={:.4} contract_passed={} failed_targets={:?} missing_train_signal={:?}",
        report_output.display(),
        suite_report.entries.len(),
        suite_report.shared_plus_adapter_to_full_bank_ratio,
        suite_report.contract.contract_passed,
        failed_targets,
        missing_signal_labels
    );
    if fail_on_validation && (!failed_targets.is_empty() || !missing_signal_labels.is_empty()) {
        return Err(std::io::Error::other(format!(
            "adapter suite failed validation failed_targets={failed_targets:?} missing_train_signal={missing_signal_labels:?}; see {}",
            report_output.display()
        ))
        .into());
    }

    Ok(())
}
