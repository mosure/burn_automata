use burn_automata_kernels::HashGridConfig;
use std::time::Instant;

use super::{
    AdaptiveExperimentConfig, AdaptiveGapDecompositionReport, AdaptiveGapDecompositionRow,
    task_wgpu::{run_fixed_rollout_wgpu, run_task_quality_rollout_wgpu},
};
use crate::{
    AutomataError, AutomataResult, NpaModel,
    adaptive::{
        AdaptiveHierarchyRestrictionPolicy, AdaptiveNpaModel, AdaptiveRolloutConfig,
        dynamics::persistent_quadrature_particle_set,
        seed::{
            restrict_adaptive_particles_to_target,
            restrict_adaptive_particles_to_target_by_merge_cost,
        },
        task_merge_oracle::target_render_merge_costs,
    },
    gpu::WgpuAutomataExecutor,
    rollout::RolloutConfig,
    target2d::{
        Target2dLossConfig, Target2dRenderedSplat, load_target_image_2d_upstream,
        render_adaptive_rollout_2d_compact_splat, render_adaptive_rollout_2d_isotropic_splat,
        render_rollout_2d_splat, render_target_2d_splat,
    },
};

pub(super) fn run_gap_decomposition_wgpu(
    executor: &WgpuAutomataExecutor,
    regular_base: &NpaModel,
    grid: &HashGridConfig,
    model: &AdaptiveNpaModel,
    config: &AdaptiveExperimentConfig,
    seeds: &[u64],
) -> AutomataResult<AdaptiveGapDecompositionReport> {
    let audit = &config.task_quality.gap_decomposition;
    let cut_step = model.config.hierarchical_restriction_step;
    if !audit.enabled {
        return Err(AutomataError::InvalidArgument(
            "adaptive gap decomposition is disabled".to_owned(),
        ));
    }
    if cut_step == 0 || model.config.bootstrap_fine_leaf_count() <= model.config.target_leaves {
        return Err(AutomataError::InvalidArgument(
            "adaptive gap decomposition requires a scheduled fine-to-coarse hierarchy cut"
                .to_owned(),
        ));
    }
    if config.task_quality.restriction_policy
        == super::AdaptiveTaskRestrictionPolicy::TargetRenderOracle
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive gap decomposition requires a deployable target-independent restriction policy"
                .to_owned(),
        ));
    }

    let fine_particles = model.config.bootstrap_fine_leaf_count();
    let fine_measure = config.multiscale_training.total_measure / fine_particles as f32;
    let target = load_target_image_2d_upstream(
        &config.task_quality.target_image,
        0.05,
        fine_particles,
        None,
    )?;
    let render_config = Target2dLossConfig {
        image_size: config.task_quality.image_size,
        ..Target2dLossConfig::default()
    };
    let target_render = render_target_2d_splat(&target, render_config)?;
    let target_center = target.mean_position();
    let horizons = audit_horizons(config, cut_step);
    let final_horizon = *horizons.last().ok_or_else(|| {
        AutomataError::InvalidArgument(
            "adaptive gap decomposition has no valid horizons".to_owned(),
        )
    })?;
    let full_mode_count = 2 * model.config.spatial_dims;
    let selected_mode_count =
        normalized_mode_count(model.config.coarse_quadrature_points, full_mode_count);
    let mode_counts = audit_mode_counts(config, selected_mode_count, full_mode_count)?;
    let seed_limit = if audit.max_seeds == 0 {
        2
    } else {
        audit.max_seeds
    };
    let seeds = unique_bounded_seeds(seeds, seed_limit);
    if seeds.is_empty() {
        return Err(AutomataError::InvalidArgument(
            "adaptive gap decomposition requires at least one seed".to_owned(),
        ));
    }

    let mut rows = Vec::with_capacity(seeds.len() * horizons.len() * mode_counts.len());
    for seed in seeds {
        for &horizon in &horizons {
            let regular = run_fixed_rollout_wgpu(
                executor,
                regular_base,
                grid,
                &RolloutConfig {
                    batch_size: 1,
                    particle_count: fine_particles,
                    steps: horizon,
                    dt: 1.0,
                    update_prob: config.task_quality.update_prob,
                    seed,
                    seed_scale: config.multiscale_training.seed_scale,
                },
            )?;
            let regular_render = render_rollout_2d_splat(
                &regular.positions,
                &regular.states,
                regular_base.config.state_dims,
                target.pixel_size,
                render_config,
                Some(target_center),
                1.0,
            )?;
            let regular_psnr = psnr(composited_mse(&regular_render, &target_render));

            // Use the same adaptive resident path while moving the cut beyond
            // this horizon. This catches backend/order drift without mixing it
            // into the coarse-representation diagnosis.
            let mut fine_control_model = without_steady_reallocation(model, horizon);
            fine_control_model.config.hierarchical_restriction_step = horizon.saturating_add(1);
            fine_control_model.validate()?;
            let fine_control = run_task_quality_rollout_wgpu(
                executor,
                &fine_control_model,
                grid,
                fine_particles,
                AdaptiveRolloutConfig {
                    steps: horizon,
                    dt: 1.0,
                    update_prob: config.task_quality.update_prob,
                    seed,
                    bandwidth_adaptation_enabled: false,
                    topology_enabled: true,
                    snapshot_interval: horizon.max(1),
                },
                config.task_quality.topology_control,
                config.task_quality.restriction_policy,
                config.multiscale_training.seed_scale,
                config.multiscale_training.total_measure,
                config.multiscale_training.bandwidth,
            )?;
            let fine_control_render = render_adaptive_rollout_2d_isotropic_splat(
                &fine_control.particles,
                fine_measure,
                target.pixel_size,
                render_config,
                Some(target_center),
            )?;
            let fine_control_psnr = psnr(composited_mse(&fine_control_render, &target_render));
            let late_dynamics_cut_psnr = (horizon == final_horizon)
                .then(|| {
                    late_cut_psnr(
                        model,
                        &fine_control.particles,
                        AdaptiveHierarchyRestrictionPolicy::DynamicsDetail,
                        fine_measure,
                        target.pixel_size,
                        render_config,
                        target_center,
                        &target_render,
                    )
                })
                .transpose()?;
            let late_learned_cut_psnr = (horizon == final_horizon)
                .then(|| {
                    late_cut_psnr(
                        model,
                        &fine_control.particles,
                        AdaptiveHierarchyRestrictionPolicy::LearnedController,
                        fine_measure,
                        target.pixel_size,
                        render_config,
                        target_center,
                        &target_render,
                    )
                })
                .transpose()?;
            let late_target_render_cut_psnr = (horizon == final_horizon)
                .then(|| {
                    late_target_render_cut_psnr(
                        model,
                        &fine_control.particles,
                        &target,
                        fine_measure,
                        target.pixel_size,
                        render_config,
                        target_center,
                        &target_render,
                        config.task_quality.render_decoder,
                        config.task_quality.render_compactness,
                    )
                })
                .transpose()?;

            for &mode_count in mode_counts
                .iter()
                .filter(|mode| horizon > cut_step || **mode == full_mode_count)
            {
                let mut evaluation_model = without_steady_reallocation(model, horizon);
                evaluation_model.config.coarse_quadrature_points = mode_count;
                evaluation_model.validate()?;
                let rollout_started = Instant::now();
                let trace = run_task_quality_rollout_wgpu(
                    executor,
                    &evaluation_model,
                    grid,
                    fine_particles,
                    AdaptiveRolloutConfig {
                        steps: horizon,
                        dt: 1.0,
                        update_prob: config.task_quality.update_prob,
                        seed,
                        bandwidth_adaptation_enabled: false,
                        topology_enabled: true,
                        snapshot_interval: horizon.max(1),
                    },
                    config.task_quality.topology_control,
                    config.task_quality.restriction_policy,
                    config.multiscale_training.seed_scale,
                    config.multiscale_training.total_measure,
                    config.multiscale_training.bandwidth,
                )?;
                let isotropic = render_adaptive_rollout_2d_isotropic_splat(
                    &trace.particles,
                    fine_measure,
                    target.pixel_size,
                    render_config,
                    Some(target_center),
                )?;
                let isotropic_psnr = psnr(composited_mse(&isotropic, &target_render));
                let covariance_control_psnr = audit
                    .covariance_decoder_control
                    .then(|| {
                        render_adaptive_rollout_2d_compact_splat(
                            &trace.particles,
                            fine_measure,
                            target.pixel_size,
                            render_config,
                            Some(target_center),
                            config.task_quality.render_compactness,
                        )
                    })
                    .transpose()?
                    .map(|render| psnr(composited_mse(&render, &target_render)));
                let internal_modes =
                    persistent_quadrature_particle_set(&evaluation_model, &trace.particles)?;
                let internal_render = render_adaptive_rollout_2d_isotropic_splat(
                    &internal_modes,
                    fine_measure,
                    target.pixel_size,
                    render_config,
                    Some(target_center),
                )?;
                let internal_mode_psnr = psnr(composited_mse(&internal_render, &target_render));
                let dynamics_particles = internal_modes.len();
                let final_measure = trace.particles.total_measure();
                rows.push(AdaptiveGapDecompositionRow {
                    seed,
                    horizon,
                    post_cut_steps: horizon.saturating_sub(cut_step),
                    mode_count,
                    visible_particles: trace.particles.len(),
                    dynamics_particles,
                    regular_fine_psnr_db: regular_psnr,
                    adaptive_fine_control_psnr_db: fine_control_psnr,
                    adaptive_fine_control_gap_db: fine_control_psnr - regular_psnr,
                    internal_mode_psnr_db: internal_mode_psnr,
                    internal_mode_gap_vs_fine_control_db: internal_mode_psnr - fine_control_psnr,
                    adaptive_isotropic_psnr_db: isotropic_psnr,
                    adaptive_isotropic_gap_db: isotropic_psnr - regular_psnr,
                    visible_decode_penalty_db: isotropic_psnr - internal_mode_psnr,
                    late_dynamics_cut_psnr_db: late_dynamics_cut_psnr,
                    late_dynamics_cut_gap_db: late_dynamics_cut_psnr
                        .map(|value| value - fine_control_psnr),
                    late_learned_cut_psnr_db: late_learned_cut_psnr,
                    late_learned_cut_gap_db: late_learned_cut_psnr
                        .map(|value| value - fine_control_psnr),
                    late_target_render_cut_psnr_db: late_target_render_cut_psnr,
                    late_target_render_cut_gap_db: late_target_render_cut_psnr
                        .map(|value| value - fine_control_psnr),
                    covariance_control_psnr_db: covariance_control_psnr,
                    covariance_control_advantage_db: covariance_control_psnr
                        .map(|value| value - isotropic_psnr),
                    maximum_covariance_axis_ratio: maximum_covariance_axis_ratio(
                        &trace.particles.covariance,
                    )?,
                    represented_measure_relative_drift: (final_measure
                        - config.multiscale_training.total_measure as f64)
                        .abs()
                        / (config.multiscale_training.total_measure as f64)
                            .abs()
                            .max(f64::MIN_POSITIVE),
                    rollout_elapsed_ms: rollout_started.elapsed().as_secs_f64() * 1_000.0,
                });
            }
        }
    }

    let mean_cut_only_full_mode_gap_db = mean_rows(&rows, |row| {
        (row.horizon == cut_step && row.mode_count == full_mode_count)
            .then_some(row.adaptive_isotropic_gap_db)
    });
    let mean_final_full_mode_gap_db = mean_rows(&rows, |row| {
        (row.horizon == final_horizon && row.mode_count == full_mode_count)
            .then_some(row.adaptive_isotropic_gap_db)
    });
    let mean_final_selected_mode_gap_db = mean_rows(&rows, |row| {
        (row.horizon == final_horizon && row.mode_count == selected_mode_count)
            .then_some(row.adaptive_isotropic_gap_db)
    });
    let mean_post_cut_recurrent_gap_change_db = paired_seed_mean(
        &rows,
        cut_step,
        final_horizon,
        full_mode_count,
        |cut, final_row| final_row.adaptive_isotropic_gap_db - cut.adaptive_isotropic_gap_db,
    );
    let mean_final_selected_mode_gap_vs_fine_control_db = mean_rows(&rows, |row| {
        (row.horizon == final_horizon && row.mode_count == selected_mode_count)
            .then_some(row.adaptive_isotropic_psnr_db - row.adaptive_fine_control_psnr_db)
    });
    let mean_post_cut_recurrent_gap_change_vs_fine_control_db = paired_seed_mean(
        &rows,
        cut_step,
        final_horizon,
        full_mode_count,
        |cut, final_row| {
            (final_row.adaptive_isotropic_psnr_db - final_row.adaptive_fine_control_psnr_db)
                - (cut.adaptive_isotropic_psnr_db - cut.adaptive_fine_control_psnr_db)
        },
    );
    let mean_selected_mode_compression_penalty_db = paired_seed_mean(
        &rows,
        final_horizon,
        final_horizon,
        full_mode_count,
        |full, _| {
            rows.iter()
                .find(|row| {
                    row.seed == full.seed
                        && row.horizon == final_horizon
                        && row.mode_count == selected_mode_count
                })
                .map_or(0.0, |selected| {
                    full.adaptive_isotropic_psnr_db - selected.adaptive_isotropic_psnr_db
                })
        },
    );
    let mean_covariance_decoder_advantage_db = mean_rows(&rows, |row| {
        (row.horizon == final_horizon && row.mode_count == selected_mode_count)
            .then_some(row.covariance_control_advantage_db)
            .flatten()
    });
    let mean_final_uncut_fine_control_gap_db = mean_rows(&rows, |row| {
        (row.horizon == final_horizon && row.mode_count == full_mode_count)
            .then_some(row.adaptive_fine_control_gap_db)
    });
    let mean_final_full_mode_internal_gap_db = mean_rows(&rows, |row| {
        (row.horizon == final_horizon && row.mode_count == full_mode_count)
            .then_some(row.internal_mode_gap_vs_fine_control_db)
    });
    let mean_final_full_mode_visible_decode_penalty_db = mean_rows(&rows, |row| {
        (row.horizon == final_horizon && row.mode_count == full_mode_count)
            .then_some(row.visible_decode_penalty_db)
    });
    let mean_final_late_dynamics_cut_gap_db = mean_rows(&rows, |row| {
        (row.horizon == final_horizon && row.mode_count == full_mode_count)
            .then_some(row.late_dynamics_cut_gap_db)
            .flatten()
    });
    let mean_final_late_learned_cut_gap_db = mean_rows(&rows, |row| {
        (row.horizon == final_horizon && row.mode_count == full_mode_count)
            .then_some(row.late_learned_cut_gap_db)
            .flatten()
    });
    let mean_final_late_target_render_cut_gap_db = mean_rows(&rows, |row| {
        (row.horizon == final_horizon && row.mode_count == full_mode_count)
            .then_some(row.late_target_render_cut_gap_db)
            .flatten()
    });
    let mean_final_controller_target_render_regret_db = mean_rows(&rows, |row| {
        (row.horizon == final_horizon && row.mode_count == full_mode_count)
            .then(|| {
                row.late_target_render_cut_psnr_db
                    .zip(row.late_learned_cut_psnr_db)
                    .map(|(oracle, learned)| oracle - learned)
            })
            .flatten()
    });

    Ok(AdaptiveGapDecompositionReport {
        cut_step,
        final_horizon,
        regular_fine_particles: fine_particles,
        selected_mode_count,
        full_mode_count,
        render_contract: "one-isotropic-gaussian-per-visible-material-leaf".to_owned(),
        rows,
        mean_cut_only_full_mode_gap_db,
        mean_final_full_mode_gap_db,
        mean_final_selected_mode_gap_db,
        mean_post_cut_recurrent_gap_change_db,
        mean_final_selected_mode_gap_vs_fine_control_db,
        mean_post_cut_recurrent_gap_change_vs_fine_control_db,
        mean_selected_mode_compression_penalty_db,
        mean_covariance_decoder_advantage_db,
        mean_final_uncut_fine_control_gap_db,
        mean_final_full_mode_internal_gap_db,
        mean_final_full_mode_visible_decode_penalty_db,
        mean_final_late_dynamics_cut_gap_db,
        mean_final_late_learned_cut_gap_db,
        mean_final_late_target_render_cut_gap_db,
        mean_final_controller_target_render_regret_db,
    })
}

/// Keeps decomposition controls orthogonal to the deployment reallocation
/// schedule. Each diagnostic rollout owns exactly one hierarchy cut (or none
/// for the fine control); composing a later same-budget reallocation would
/// change the leaf templates before `late_cut_psnr` and double-count topology.
fn without_steady_reallocation(model: &AdaptiveNpaModel, horizon: usize) -> AdaptiveNpaModel {
    let mut model = model.clone();
    let disabled_step = horizon.saturating_add(1);
    model.config.topology_start_step = disabled_step;
    model.config.steady_topology_start_step = disabled_step;
    model.config.topology_end_step = 0;
    model
}

#[allow(clippy::too_many_arguments)]
fn late_cut_psnr(
    model: &AdaptiveNpaModel,
    fine: &crate::adaptive::AdaptiveParticleSet,
    policy: AdaptiveHierarchyRestrictionPolicy,
    fine_measure: f32,
    pixel_size: f32,
    render_config: Target2dLossConfig,
    target_center: [f32; 2],
    target_render: &Target2dRenderedSplat,
) -> AutomataResult<f32> {
    let mut cut_model = model.clone();
    cut_model.config.hierarchical_restriction_policy = policy;
    cut_model.validate()?;
    let restricted = restrict_adaptive_particles_to_target(&cut_model, fine)?;
    let rendered = render_adaptive_rollout_2d_isotropic_splat(
        &restricted,
        fine_measure,
        pixel_size,
        render_config,
        Some(target_center),
    )?;
    Ok(psnr(composited_mse(&rendered, target_render)))
}

#[allow(clippy::too_many_arguments)]
fn late_target_render_cut_psnr(
    model: &AdaptiveNpaModel,
    fine: &crate::adaptive::AdaptiveParticleSet,
    target: &crate::target2d::TargetImage2d,
    fine_measure: f32,
    pixel_size: f32,
    render_config: Target2dLossConfig,
    target_center: [f32; 2],
    target_render: &Target2dRenderedSplat,
    render_decoder: crate::adaptive::AdaptiveRenderDecoder,
    render_compactness: f32,
) -> AutomataResult<f32> {
    let costs = target_render_merge_costs(
        fine,
        model.config.target_leaves,
        target,
        render_config,
        fine_measure,
        render_decoder,
        render_compactness,
        crate::adaptive::AdaptiveRestrictionLabelTarget::TargetImage,
    )?;
    let restricted = restrict_adaptive_particles_to_target_by_merge_cost(model, fine, &costs)?;
    let rendered = render_adaptive_rollout_2d_isotropic_splat(
        &restricted,
        fine_measure,
        pixel_size,
        render_config,
        Some(target_center),
    )?;
    Ok(psnr(composited_mse(&rendered, target_render)))
}

fn audit_horizons(config: &AdaptiveExperimentConfig, cut_step: usize) -> Vec<usize> {
    let mut horizons = if config.task_quality.gap_decomposition.horizons.is_empty() {
        vec![
            cut_step.saturating_sub(1).max(1),
            cut_step,
            cut_step.saturating_add(1),
            cut_step.saturating_add(32),
            cut_step.saturating_add(64),
            config.task_quality.rollout_steps,
        ]
    } else {
        config.task_quality.gap_decomposition.horizons.clone()
    };
    horizons.retain(|horizon| *horizon > 0);
    horizons.sort_unstable();
    horizons.dedup();
    horizons
}

fn audit_mode_counts(
    config: &AdaptiveExperimentConfig,
    selected: usize,
    full: usize,
) -> AutomataResult<Vec<usize>> {
    let mut modes = config
        .task_quality
        .gap_decomposition
        .mode_counts
        .iter()
        .map(|mode| normalized_mode_count(*mode, full))
        .collect::<Vec<_>>();
    modes.extend([selected, full]);
    modes.sort_unstable();
    modes.dedup();
    if modes.iter().any(|mode| *mode == 0 || *mode > full) {
        return Err(AutomataError::InvalidArgument(format!(
            "adaptive gap mode counts must be in 1..={full}, got {modes:?}",
        )));
    }
    Ok(modes)
}

fn normalized_mode_count(mode: usize, full: usize) -> usize {
    if mode == 0 { full } else { mode }
}

fn unique_bounded_seeds(seeds: &[u64], limit: usize) -> Vec<u64> {
    let mut out = Vec::with_capacity(seeds.len().min(limit));
    for seed in seeds.iter().copied() {
        if !out.contains(&seed) {
            out.push(seed);
        }
        if out.len() == limit {
            break;
        }
    }
    out
}

fn maximum_covariance_axis_ratio(covariance: &[[f32; 9]]) -> AutomataResult<f32> {
    covariance.iter().try_fold(1.0_f32, |maximum, covariance| {
        let geometry =
            crate::adaptive::diagnostic_covariance_gaussian_geometry(1.0, *covariance, 2)?;
        let minor = geometry.scale[1].max(f32::MIN_POSITIVE);
        Ok(maximum.max(geometry.scale[0] / minor))
    })
}

fn mean_rows(
    rows: &[AdaptiveGapDecompositionRow],
    select: impl Fn(&AdaptiveGapDecompositionRow) -> Option<f32>,
) -> Option<f32> {
    let values = rows.iter().filter_map(select).collect::<Vec<_>>();
    (!values.is_empty()).then(|| values.iter().sum::<f32>() / values.len() as f32)
}

fn paired_seed_mean(
    rows: &[AdaptiveGapDecompositionRow],
    first_horizon: usize,
    second_horizon: usize,
    mode_count: usize,
    difference: impl Fn(&AdaptiveGapDecompositionRow, &AdaptiveGapDecompositionRow) -> f32,
) -> Option<f32> {
    let values = rows
        .iter()
        .filter(|row| row.horizon == first_horizon && row.mode_count == mode_count)
        .filter_map(|first| {
            rows.iter()
                .find(|second| {
                    second.seed == first.seed
                        && second.horizon == second_horizon
                        && second.mode_count == mode_count
                })
                .map(|second| difference(first, second))
        })
        .collect::<Vec<_>>();
    (!values.is_empty()).then(|| values.iter().sum::<f32>() / values.len() as f32)
}

fn composited_mse(prediction: &Target2dRenderedSplat, target: &Target2dRenderedSplat) -> f32 {
    let pixels = prediction.density.len().min(target.density.len());
    let mut squared_error = 0.0;
    for pixel in 0..pixels {
        let prediction_alpha = prediction.density[pixel].clamp(0.0, 1.0);
        let target_alpha = target.density[pixel].clamp(0.0, 1.0);
        for channel in 0..3 {
            let index = pixel * 3 + channel;
            let prediction_value = (prediction.rgb[index] + 1.0 - prediction_alpha).clamp(0.0, 1.0);
            let target_value = (target.rgb[index] + 1.0 - target_alpha).clamp(0.0, 1.0);
            squared_error += (prediction_value - target_value).powi(2);
        }
    }
    squared_error / (pixels * 3).max(1) as f32
}

fn psnr(mse: f32) -> f32 {
    -10.0 * mse.max(1.0e-12).log10()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive::AdaptiveNpaConfig;

    #[test]
    fn gap_controls_disable_deployment_reallocation() {
        let rule = NpaModel::upstream_seeded(crate::NpaConfig::growing_2d(), 7);
        let mut config = AdaptiveNpaConfig::growing_2d();
        config.topology_start_step = 192;
        config.steady_topology_start_step = 192;
        config.topology_end_step = 256;
        let model = AdaptiveNpaModel::seeded(rule, config, 11).unwrap();

        let isolated = without_steady_reallocation(&model, 256);

        assert_eq!(isolated.config.topology_start_step, 257);
        assert_eq!(isolated.config.steady_topology_start_step, 257);
        assert_eq!(isolated.config.topology_end_step, 0);
        isolated.validate().unwrap();
    }
}
