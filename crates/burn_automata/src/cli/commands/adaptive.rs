use std::fs;

use crate::{
    AdaptiveClosureAuditConfig, AdaptiveExperimentConfig, AdaptiveTopologyAuditConfig,
    evaluate_adaptive_task_quality, evaluate_adaptive_task_quality_validation, load_adaptive_model,
    run_adaptive_closure_audit, run_adaptive_experiment_suite, run_adaptive_topology_audit,
    save_adaptive_model, validate_adaptive_task_quality_validation_gates,
};

use super::super::args::Command;

pub(super) fn run_audit_adaptive_topology(
    command: Command,
) -> Result<(), Box<dyn std::error::Error>> {
    let Command::AuditAdaptiveTopology { config } = command else {
        unreachable!("adaptive topology dispatcher passed a different command")
    };
    let source = fs::read_to_string(&config)?;
    let audit: AdaptiveTopologyAuditConfig = toml::from_str(&source)?;
    let report = run_adaptive_topology_audit(&audit)?;
    if let Some(parent) = audit.report_output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&audit.report_output, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "adaptive topology audit: {} canonical + {} unequal events at {:.0}/s | max ratio {:.3} | measure {:.3e}, centroid {:.3e}, second moment {:.3e}, determinant scale {:.3e} | SPD failures {}",
        report.topology.canonical_events,
        report.topology.unequal_events,
        report.topology.events_per_second,
        report.topology.maximum_sampled_child_measure_ratio,
        report.topology.max_measure_relative_error,
        report.topology.max_centroid_l2_error,
        report.topology.max_second_moment_relative_error,
        report.topology.max_determinant_scale_relative_error,
        report.topology.spd_failures,
    );
    println!("report {}", audit.report_output.display());
    Ok(())
}

pub(super) fn run_audit_adaptive_closure(
    command: Command,
) -> Result<(), Box<dyn std::error::Error>> {
    let Command::AuditAdaptiveClosure { config } = command else {
        unreachable!("adaptive closure dispatcher passed a different command")
    };
    let source = fs::read_to_string(&config)?;
    let audit: AdaptiveClosureAuditConfig = toml::from_str(&source)?;
    let report = run_adaptive_closure_audit(&audit)?;
    if let Some(parent) = audit.report_output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&audit.report_output, serde_json::to_vec_pretty(&report)?)?;
    let feature_check = report
        .local_features_verified
        .then(|| format!("{:.2e} max error", report.maximum_local_feature_difference));
    println!(
        "adaptive closure audit: {} paired rows across {} snapshots | unresolved modes {} (max {}/row), augmented/fine state {:.3}x | observables {:.2e}, features {} | mode delta {:.4}, reconstruction affine/augmented {:.4}/{:.2e} (max {:.2e}) | target RMS {:.4}, pair delta {:.4} | memoryless NRMSE floor global/p95/max {:.4}/{:.4}/{:.4} | {:.3}s",
        report.paired_coarse_rows,
        report.snapshots,
        report.unresolved_state_modes,
        report.maximum_unresolved_state_modes_per_coarse_row,
        report.augmented_to_fine_state_value_ratio,
        report.maximum_restricted_observable_difference,
        feature_check.as_deref().unwrap_or("skipped"),
        report.paired_closure_mode_difference_root_mean_square,
        report.affine_state_reconstruction_root_mean_square_error,
        report.augmented_state_reconstruction_root_mean_square_error,
        report.maximum_augmented_state_reconstruction_error,
        report.target_root_mean_square,
        report.paired_target_difference_root_mean_square,
        report.memoryless_normalized_rmse_lower_bound,
        report.p95_row_normalized_rmse_lower_bound,
        report.maximum_row_normalized_rmse_lower_bound,
        report.elapsed_ms / 1_000.0,
    );
    println!("report {}", audit.report_output.display());
    Ok(())
}

pub(super) fn run_adaptive_npa(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::AdaptiveNpa { config } = command else {
        unreachable!("adaptive command dispatcher passed a different command")
    };
    let source = fs::read_to_string(&config)?;
    let experiment: AdaptiveExperimentConfig = toml::from_str(&source)?;
    let task_quality_execution = if !experiment.task_quality.enabled {
        "disabled"
    } else if experiment.backend == crate::AdaptiveTrainingBackend::NdArray {
        "CPU reference"
    } else {
        "resident WGPU rollout + CPU final metrics"
    };
    eprintln!(
        "adaptive execution: optimizer={:?}, exact replay={:?}, deployment replay={:?}, CPU reference rollout={}, task-quality execution={task_quality_execution}",
        experiment.backend,
        experiment.multiscale_training.on_policy_replay_backend,
        experiment.multiscale_training.deployment_replay_backend,
        experiment.rollout.enabled,
    );
    let report = run_adaptive_experiment_suite(&experiment)?;
    println!(
        "adaptive NPA ({:?}): operator 2D constant={:.3e} affine={:.3e}, 3D constant={:.3e} affine={:.3e} | events={} at {:.0}/s | controller {:.4}->{:.4} ({:.0} rows/s)",
        report.rule_perception,
        report.operator.adaptive_constant_max_error,
        report.operator.adaptive_affine_gradient_mean_error,
        report.operator_3d.adaptive_constant_max_error,
        report.operator_3d.adaptive_affine_gradient_mean_error,
        report.topology.samples,
        report.topology.events_per_second,
        report.training.initial_loss,
        report.training.final_loss,
        report.training.rows_per_second,
    );
    if let Some(base) = &report.base_training {
        for phase in &base.phases {
            let throughput = phase
                .training
                .metrics
                .get("steady_particle_steps_per_sec")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            println!(
                "fresh base phase {}: {} particles | {} steps | best loss {} | fresh eval {} @ {} | best PSNR {} dB @ {} | {:.0} particle-steps/s",
                phase.name,
                phase.particle_count,
                phase
                    .training
                    .metrics
                    .get("steps")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
                phase
                    .training
                    .best_train_loss
                    .map_or_else(|| "none".to_string(), |loss| format!("{loss:.6}")),
                phase
                    .training
                    .best_fresh_seed_eval_loss
                    .map_or_else(|| "none".to_string(), |loss| format!("{loss:.6}")),
                phase.training.best_fresh_seed_eval_step,
                phase
                    .training
                    .best_fresh_seed_render_rgb_psnr_db
                    .map_or_else(|| "none".to_string(), |psnr| format!("{psnr:.3}")),
                phase.training.best_fresh_seed_render_rgb_psnr_step,
                throughput,
            );
        }
    }
    let largest_cap = report
        .scaling
        .quality_rows
        .iter()
        .map(|row| row.spacing_cap_ratio)
        .reduce(f32::max);
    for solid in ["sphere-with-cavity", "solid-torus"] {
        let rows = report
            .scaling
            .quality_rows
            .iter()
            .filter(|row| {
                row.solid == solid
                    && row.allocation == "clearance-oracle"
                    && largest_cap
                        .is_some_and(|cap| (row.spacing_cap_ratio - cap).abs() <= f32::EPSILON)
            })
            .collect::<Vec<_>>();
        if !rows.is_empty() {
            let count = rows.len() as f64;
            let mean_reduction = rows.iter().map(|row| row.count_reduction).sum::<f64>() / count;
            let mean_field_nrmse = rows.iter().map(|row| row.field_nrmse).sum::<f64>() / count;
            let mean_protected_nrmse =
                rows.iter().map(|row| row.protected_band_nrmse).sum::<f64>() / count;
            let mean_hd95 = rows.iter().map(|row| row.boundary_hd95_voxels).sum::<f64>() / count;
            println!(
                "sparse oracle {solid}: {:.2}x fewer leaves | field NRMSE {:.4} | protected NRMSE {:.2e} | boundary HD95 {:.3} voxels",
                mean_reduction, mean_field_nrmse, mean_protected_nrmse, mean_hd95,
            );
        }
    }
    if let Some(rollout) = &report.rollout {
        println!(
            "adaptive rollout: leaves {}->{} (target {}, gap {:.1}%) | scale {:.4}..{:.4} (CV {:.3}, {} audit bins, {:.1}% off-dyadic, {:.3} octave RMSE) | bandwidth adaptive={} | measure drift {:.3e} | {:.0} particle-steps/s",
            rollout.initial_leaves,
            rollout.final_leaves,
            rollout.target_leaves,
            rollout.target_leaf_relative_error * 100.0,
            rollout.final_min_footprint,
            rollout.final_max_footprint,
            rollout.final_footprint_coefficient_of_variation,
            rollout.final_occupied_material_scale_bins,
            100.0 * rollout.final_fractional_material_scale_fraction,
            rollout.final_dyadic_scale_quantization_rmse_octaves,
            rollout.bandwidth_adaptation_active,
            rollout.measure_relative_drift,
            rollout.particle_steps_per_second,
        );
    }
    if let Some(multiscale) = &report.multiscale_training {
        println!(
            "multiscale lizard: rows {} | footprint {:.4}..{:.4} (CV {:.3}) | rule {:.4}->{:.4} | held-out NRMSE {:.4}, corr {:.4}, proxy gain {:.1}%",
            multiscale.training_dataset.rows,
            multiscale.training_dataset.minimum_footprint,
            multiscale.training_dataset.maximum_footprint,
            multiscale
                .training_dataset
                .footprint_coefficient_of_variation,
            multiscale.training.initial_mean_squared_error,
            multiscale.training.final_mean_squared_error,
            multiscale.heldout_validation.normalized_mean_squared_error,
            multiscale.heldout_validation.update_correlation,
            100.0 * multiscale.heldout_validation.proxy_relative_mse_gain,
        );
        if let Some(heldout) = &multiscale.heldout_on_policy_validation {
            let latest = multiscale.on_policy_training.last();
            let replay_seconds = multiscale
                .on_policy_datasets
                .iter()
                .map(|dataset| dataset.generation_elapsed_ms)
                .sum::<f64>()
                / 1_000.0;
            println!(
                "coupled recurrent rule: rounds {} | latest train NRMSE {} corr {} | held-out NRMSE {:.4}, corr {:.4} | replay {:.3}s",
                multiscale.on_policy_training.len(),
                latest.map_or_else(
                    || "none".to_owned(),
                    |report| format!(
                        "{:.4}",
                        report.trained_validation.normalized_mean_squared_error
                    ),
                ),
                latest.map_or_else(
                    || "none".to_owned(),
                    |report| format!("{:.4}", report.trained_validation.update_correlation),
                ),
                heldout.normalized_mean_squared_error,
                heldout.update_correlation,
                replay_seconds,
            );
        }
        if multiscale.closure_training.steps > 0 {
            println!(
                "recurrent closure: active rows {} | train {:.6}->{:.6} (NRMSE {:.4}, corr {:.4}, {:.1}M rows/s) | held-out NRMSE {:.4}, corr {:.4}, max error {:.4} | replay {:.3}s, DAgger fits {}",
                multiscale.closure_training.active_rows,
                multiscale.closure_training.initial_mean_squared_error,
                multiscale.closure_training.final_mean_squared_error,
                multiscale
                    .closure_training
                    .trained_validation
                    .normalized_root_mean_squared_error,
                multiscale
                    .closure_training
                    .trained_validation
                    .update_correlation,
                multiscale.closure_training.rows_per_second / 1.0e6,
                multiscale
                    .heldout_closure_validation
                    .normalized_root_mean_squared_error,
                multiscale.heldout_closure_validation.update_correlation,
                multiscale.heldout_closure_validation.maximum_absolute_error,
                multiscale
                    .on_policy_datasets
                    .iter()
                    .map(|dataset| dataset.generation_elapsed_ms)
                    .sum::<f64>()
                    / 1_000.0,
                multiscale.on_policy_closure_training.len(),
            );
            for (round, training) in multiscale.on_policy_closure_training.iter().enumerate() {
                println!(
                    "recurrent closure DAgger {}: {:.6}->{:.6} | NRMSE {:.4}, corr {:.4} | {:.1}M rows/s in {:.3}s",
                    round + 1,
                    training.initial_mean_squared_error,
                    training.final_mean_squared_error,
                    training
                        .trained_validation
                        .normalized_root_mean_squared_error,
                    training.trained_validation.update_correlation,
                    training.rows_per_second / 1.0e6,
                    training.training_elapsed_ms / 1_000.0,
                );
            }
            if let Some(heldout) = &multiscale.heldout_on_policy_closure_validation {
                println!(
                    "recurrent closure held-out replay: active rows {} | target RMS {:.4} | NRMSE {:.4}, corr {:.4}, max error {:.4}",
                    heldout.active_rows,
                    heldout.target_root_mean_square,
                    heldout.normalized_root_mean_squared_error,
                    heldout.update_correlation,
                    heldout.maximum_absolute_error,
                );
            }
        }
        if multiscale.deployment_training.steps > 0 {
            println!(
                "deployment optimizer: {:.1}M rows/s in {:.3}s | replay collection {:.3}s | DAgger fits {}",
                multiscale.deployment_training.rows_per_second / 1.0e6,
                multiscale.deployment_training.training_elapsed_ms / 1_000.0,
                multiscale
                    .deployment_on_policy_datasets
                    .iter()
                    .map(|dataset| dataset.generation_elapsed_ms)
                    .sum::<f64>()
                    / 1_000.0,
                multiscale.deployment_on_policy_training.len(),
            );
            for (round, training) in multiscale.deployment_on_policy_training.iter().enumerate() {
                println!(
                    "deployment DAgger {}: {:.1}M rows/s in {:.3}s",
                    round + 1,
                    training.rows_per_second / 1.0e6,
                    training.training_elapsed_ms / 1_000.0,
                );
            }
        }
    }
    if let Some(closure) = &report.closure_identifiability {
        let feature_check = closure
            .local_features_verified
            .then(|| format!("{:.2e} max error", closure.maximum_local_feature_difference));
        println!(
            "coarse closure identifiability: {} paired rows across {} snapshots | unresolved modes {} (max {}/row), augmented/fine state {:.3}x | observables {:.2e}, features {} | mode delta {:.4}, reconstruction affine/augmented {:.4}/{:.2e} (max {:.2e}) | target RMS {:.4}, pair delta {:.4} | memoryless NRMSE floor global/p95/max {:.4}/{:.4}/{:.4}",
            closure.paired_coarse_rows,
            closure.snapshots,
            closure.unresolved_state_modes,
            closure.maximum_unresolved_state_modes_per_coarse_row,
            closure.augmented_to_fine_state_value_ratio,
            closure.maximum_restricted_observable_difference,
            feature_check.as_deref().unwrap_or("skipped"),
            closure.paired_closure_mode_difference_root_mean_square,
            closure.affine_state_reconstruction_root_mean_square_error,
            closure.augmented_state_reconstruction_root_mean_square_error,
            closure.maximum_augmented_state_reconstruction_error,
            closure.target_root_mean_square,
            closure.paired_target_difference_root_mean_square,
            closure.memoryless_normalized_rmse_lower_bound,
            closure.p95_row_normalized_rmse_lower_bound,
            closure.maximum_row_normalized_rmse_lower_bound,
        );
    }
    if let Some(quality) = &report.task_quality {
        println!(
            "lizard seed {} @{}: teacher {:.2} dB | regular-4096 {:.2} dB | regular-count {:.2} dB | regular-material {:.2} dB | budget fixed {:.2} dB | adaptive {:.2} dB | adaptive/4096 {:+.2} dB | adaptive/material {:+.2} dB | leaves {}->{} visible/{} dynamics (fine/reference/coarse {}/{}/{}) | footprint {:.4}..{:.4} (CV {:.3}, {} audit bins, {:.1}% off-dyadic) | steady events {}/{} (+{} restriction, +{} bootstrap) | CPU eval {:.3}s",
            quality.seed,
            quality.rollout_steps,
            quality.teacher_target_composited_psnr_db,
            quality.regular_base_target_composited_psnr_db,
            quality.regular_matched_budget_target_composited_psnr_db,
            quality.regular_material_matched_budget_target_composited_psnr_db,
            quality.adaptive_budget_fixed_target_composited_psnr_db,
            quality.adaptive_target_composited_psnr_db,
            quality.adaptive_over_regular_base_psnr_gain_db,
            quality.adaptive_over_regular_material_matched_budget_psnr_gain_db,
            quality.adaptive_initial_particles,
            quality.adaptive_final_particles,
            quality.adaptive_dynamics_particles,
            quality.fine_leaf_count,
            quality.reference_leaf_count,
            quality.coarse_leaf_count,
            quality.final_min_footprint,
            quality.final_max_footprint,
            quality.final_footprint_coefficient_of_variation,
            quality.final_occupied_material_scale_bins,
            100.0 * quality.final_fractional_material_scale_fraction,
            quality.steady_split_events,
            quality.steady_merge_events,
            quality.restriction_merge_events,
            quality.bootstrap_split_events,
            quality.elapsed_ms / 1_000.0,
        );
    }
    if let Some(validation) = &report.task_quality_validation {
        println!(
            "lizard parity ({} seeds, {} structural audits): adaptive {:.3} dB vs upstream teacher {:.3} dB, artifact base {:.3} dB, and regular-material {:.3} dB | mean/worst teacher gap {:+.3}/{:+.3} dB | mean/worst artifact-base gap {:+.3}/{:+.3} dB | mean/worst material gain {:+.3}/{:+.3} dB | mean/worst topology gain {:+.3}/{:+.3} dB | min continuum {} bins/{:.1}% off-dyadic | min controller/oracle {:.3} | max measure drift {:.3e}",
            validation.rows.len(),
            validation.structural_audit_seeds,
            validation.mean_adaptive_target_composited_psnr_db,
            validation.mean_teacher_target_composited_psnr_db,
            validation.mean_regular_base_target_composited_psnr_db,
            validation.mean_regular_material_matched_budget_target_composited_psnr_db,
            validation.mean_adaptive_over_teacher_psnr_gain_db,
            validation.worst_adaptive_over_teacher_psnr_gain_db,
            validation.mean_adaptive_over_regular_base_psnr_gain_db,
            validation.worst_adaptive_over_regular_base_psnr_gain_db,
            validation.mean_adaptive_over_regular_material_matched_budget_psnr_gain_db,
            validation.worst_adaptive_over_regular_material_matched_budget_psnr_gain_db,
            validation.mean_adaptive_over_budget_fixed_psnr_gain_db,
            validation.worst_adaptive_over_budget_fixed_psnr_gain_db,
            validation.minimum_final_occupied_material_scale_bins,
            100.0 * validation.minimum_final_fractional_material_scale_fraction,
            validation.minimum_controller_oracle_refinement_scale_correlation,
            validation.maximum_measure_relative_drift,
        );
        print_gap_decomposition(validation);
    }
    println!("model {}", report.model_output);
    println!("report {}", experiment.report_output.display());
    Ok(())
}

pub(super) fn run_eval_adaptive_npa(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::EvalAdaptiveNpa {
        config,
        model,
        seeds,
        output,
        promote_model_output,
    } = command
    else {
        unreachable!("adaptive evaluator dispatcher passed a different command")
    };
    let source = fs::read_to_string(&config)?;
    let experiment: AdaptiveExperimentConfig = toml::from_str(&source)?;
    // Evaluation configs own the deployment schedule. Rebind before running any
    // metric so an optionally promoted artifact is exactly the one validated.
    let artifact = load_adaptive_model(&model)?.with_runtime_config(
        experiment.adaptive.clone(),
        Some(format!(
            "{} evaluated by {}",
            model.display(),
            config.display(),
        )),
    )?;
    if seeds.is_empty() && promote_model_output.is_some() {
        return Err(std::io::Error::other(
            "adaptive artifact promotion requires aggregate --seeds parity validation",
        )
        .into());
    }
    if !seeds.is_empty() {
        let report =
            evaluate_adaptive_task_quality_validation(&artifact.model, &experiment, &seeds)?;
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
        let gate_failures =
            validate_adaptive_task_quality_validation_gates(experiment.gates, Some(&report));
        if !gate_failures.is_empty() {
            return Err(std::io::Error::other(format!(
                "adaptive parity gates failed:\n- {}",
                gate_failures.join("\n- "),
            ))
            .into());
        }
        if let Some(promoted_path) = promote_model_output {
            let digest = save_adaptive_model(&promoted_path, &artifact)?;
            println!(
                "promoted adaptive model {} sha256 {}",
                promoted_path.display(),
                digest,
            );
        }
        println!(
            "adaptive parity ({} seeds, {} structural audits): adaptive {:.3} dB vs upstream teacher {:.3} dB, artifact base {:.3} dB, and regular-material {:.3} dB | mean/worst teacher gap {:+.3}/{:+.3} dB | mean/worst artifact-base gap {:+.3}/{:+.3} dB | mean/worst material gain {:+.3}/{:+.3} dB | mean/worst topology gain {:+.3}/{:+.3} dB | rollout/topology {:.2}/{:.2} ms mean, {:.2} ms max event | min continuum {} bins/{:.1}% off-dyadic | min controller/oracle {:.3} | max measure drift {:.3e}",
            report.rows.len(),
            report.structural_audit_seeds,
            report.mean_adaptive_target_composited_psnr_db,
            report.mean_teacher_target_composited_psnr_db,
            report.mean_regular_base_target_composited_psnr_db,
            report.mean_regular_material_matched_budget_target_composited_psnr_db,
            report.mean_adaptive_over_teacher_psnr_gain_db,
            report.worst_adaptive_over_teacher_psnr_gain_db,
            report.mean_adaptive_over_regular_base_psnr_gain_db,
            report.worst_adaptive_over_regular_base_psnr_gain_db,
            report.mean_adaptive_over_regular_material_matched_budget_psnr_gain_db,
            report.worst_adaptive_over_regular_material_matched_budget_psnr_gain_db,
            report.mean_adaptive_over_budget_fixed_psnr_gain_db,
            report.worst_adaptive_over_budget_fixed_psnr_gain_db,
            report.mean_adaptive_rollout_elapsed_ms,
            report.mean_adaptive_topology_elapsed_ms,
            report.maximum_topology_update_elapsed_ms,
            report.minimum_final_occupied_material_scale_bins,
            100.0 * report.minimum_final_fractional_material_scale_fraction,
            report.minimum_controller_oracle_refinement_scale_correlation,
            report.maximum_measure_relative_drift,
        );
        print_gap_decomposition(&report);
        println!("report {}", output.display());
        return Ok(());
    }
    let report = evaluate_adaptive_task_quality(&artifact.model, &experiment)?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "adaptive lizard @{}: task base {:.3} dB | hub fixed {:.3} dB | topology {:.3} dB ({:+.3} dB) | visible/dynamics/interactions {}/{}/{} ({:?}) | steady events {}/{} (+{} restriction, +{} bootstrap) | fine/reference/coarse {}/{}/{} | scale {:.4}..{:.4} ({} audit bins, {:.1}% off-dyadic) | rollout/topology {:.2}/{:.2} ms, {:.2} ms max event | controller/oracle {:.3}",
        report.rollout_steps,
        report.adaptive_budget_frozen_base_target_composited_psnr_db,
        report.adaptive_budget_fixed_target_composited_psnr_db,
        report.adaptive_target_composited_psnr_db,
        report.adaptive_target_composited_psnr_db
            - report.adaptive_budget_fixed_target_composited_psnr_db,
        report.adaptive_final_particles,
        report.adaptive_dynamics_particles,
        report.adaptive_interaction_particles,
        report.dynamics_semantics,
        report.steady_split_events,
        report.steady_merge_events,
        report.restriction_merge_events,
        report.bootstrap_split_events,
        report.fine_leaf_count,
        report.reference_leaf_count,
        report.coarse_leaf_count,
        report.final_min_footprint,
        report.final_max_footprint,
        report.final_occupied_material_scale_bins,
        100.0 * report.final_fractional_material_scale_fraction,
        report.adaptive_rollout_elapsed_ms,
        report.adaptive_topology_elapsed_ms,
        report.maximum_topology_update_elapsed_ms,
        report.controller_oracle_refinement_scale_correlation,
    );
    println!("report {}", output.display());
    Ok(())
}

fn print_gap_decomposition(report: &crate::AdaptiveTaskQualityValidationReport) {
    let Some(gap) = &report.gap_decomposition else {
        return;
    };
    let value =
        |value: Option<f32>| value.map_or_else(|| "n/a".to_owned(), |value| format!("{value:+.3}"));
    println!(
        "adaptive gap decomposition: cut-only/full {} dB | final/full {} dB | final/mode{} {} dB | recurrent change {} dB | mode compression {} dB | uncut/backend {} dB | internal dynamics {} dB | visible decode {} dB | late dynamics cut {} dB | late learned cut {} dB | late target-render cut {} dB | controller regret {} dB | covariance decoder advantage {} dB | {} rows",
        value(gap.mean_cut_only_full_mode_gap_db),
        value(gap.mean_final_full_mode_gap_db),
        gap.selected_mode_count,
        value(gap.mean_final_selected_mode_gap_db),
        value(gap.mean_post_cut_recurrent_gap_change_db),
        value(gap.mean_selected_mode_compression_penalty_db),
        value(gap.mean_final_uncut_fine_control_gap_db),
        value(gap.mean_final_full_mode_internal_gap_db),
        value(gap.mean_final_full_mode_visible_decode_penalty_db),
        value(gap.mean_final_late_dynamics_cut_gap_db),
        value(gap.mean_final_late_learned_cut_gap_db),
        value(gap.mean_final_late_target_render_cut_gap_db),
        value(gap.mean_final_controller_target_render_regret_db),
        value(gap.mean_covariance_decoder_advantage_db),
        gap.rows.len(),
    );
}
