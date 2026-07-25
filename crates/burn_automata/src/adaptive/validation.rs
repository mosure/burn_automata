use serde::{Deserialize, Serialize};

use crate::ParticleSeed;

use super::AdaptiveHierarchyRestrictionPolicy;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AdaptiveTarget2dValidationConfig {
    /// Active material-row budget used only by matched validation. Zero keeps
    /// the artifact's trained target budget.
    pub active_particle_count: usize,
    pub seeds: Vec<u64>,
    pub horizons: Vec<usize>,
    pub update_prob: f32,
    pub seed_scale: f32,
    pub seed_mode: ParticleSeed,
    pub dt: f32,
    /// Exercise the model's resident topology policy during validation.
    ///
    /// Production parity gates keep this enabled. Disabling it is a bounded
    /// diagnostic control for separating recurrent material work from
    /// topology-selection overhead.
    pub topology_enabled: bool,
    /// Optional absolute step at which the matched 4,096-row fine trajectory
    /// is conservatively restricted to the active budget. Zero starts at the
    /// active budget, preserving the direct active-material validation path.
    pub delayed_restriction_step: usize,
    /// Target-independent selector used by `delayed_restriction_step`.
    pub restriction_policy: AdaptiveHierarchyRestrictionPolicy,
    /// Optional cadence for fused budget-neutral coarse/fine reallocation
    /// after the delayed cut. Zero keeps the selected material partition
    /// fixed. A positive value uses local-detail topology with the model's
    /// configured conservative exchange budget per interval.
    pub reallocation_interval_steps: usize,
    /// First absolute paired-reallocation step. Zero selects one interval
    /// after the delayed cut (or one interval after initialization).
    pub reallocation_start_step: usize,
    /// Last absolute paired-reallocation step. Zero keeps reallocation active
    /// through the longest validation horizon.
    pub reallocation_end_step: usize,
    /// Required relative detail gain for a budget-neutral merge/split pair.
    /// Zero accepts any strictly positive gain; one disables all pairs.
    pub min_reallocation_relative_gain: f32,
    pub min_adaptive_psnr_db: f32,
    pub max_oracle_psnr_gap_db: f32,
    pub max_same_rule_fine_psnr_gap_db: f32,
    pub max_psnr_drift_db: f32,
    /// First horizon included in aggregate quality gates. Zero disables the
    /// aggregate gates while retaining per-row diagnostics and limits.
    pub quality_horizon_min_steps: usize,
    pub min_quality_mean_adaptive_psnr_db: f32,
    pub min_quality_worst_adaptive_psnr_db: f32,
    pub max_quality_mean_oracle_gap_db: f32,
    pub max_quality_worst_oracle_gap_db: f32,
    pub max_quality_mean_same_rule_fine_gap_db: f32,
    pub max_quality_worst_same_rule_fine_gap_db: f32,
    pub max_material_relative_error: f32,
    pub max_out_of_bounds_fraction: f32,
    pub max_grid_overflow: u32,
    pub max_interaction_work_ratio: f32,
    pub max_wall_time_ratio: f32,
    /// Require the evaluated model to demonstrate a material-scale range,
    /// scale-dependent communication, and useful spatial reallocation in
    /// addition to ordinary Target2D quality.
    pub require_adaptive_resolution: bool,
    /// First horizon included in adaptive-resolution gates. Zero uses
    /// `quality_horizon_min_steps`, or the longest horizon when aggregate
    /// quality gates are disabled.
    pub adaptive_resolution_horizon_min_steps: usize,
    pub min_material_scale_ratio: f32,
    pub min_adaptive_support_bin_count: usize,
    pub min_occupied_material_scale_bins: usize,
    pub min_accepted_local_detail_exchanges: usize,
    pub min_mean_scale_detail_correlation_gain_vs_static: f32,
    pub min_worst_scale_detail_correlation_gain_vs_static: f32,
    pub min_mean_static_graded_psnr_gain_db: f32,
    pub min_worst_static_graded_psnr_gain_db: f32,
}

impl Default for AdaptiveTarget2dValidationConfig {
    fn default() -> Self {
        Self {
            active_particle_count: 0,
            seeds: (42..50).collect(),
            horizons: vec![96, 256, 512, 4_096],
            update_prob: 0.5,
            seed_scale: 0.2,
            seed_mode: ParticleSeed::UniformCircle,
            dt: 1.0,
            topology_enabled: true,
            delayed_restriction_step: 0,
            restriction_policy: AdaptiveHierarchyRestrictionPolicy::DynamicsDetail,
            reallocation_interval_steps: 0,
            reallocation_start_step: 0,
            reallocation_end_step: 0,
            min_reallocation_relative_gain: 0.0,
            min_adaptive_psnr_db: 26.0,
            max_oracle_psnr_gap_db: 0.5,
            max_same_rule_fine_psnr_gap_db: 0.5,
            max_psnr_drift_db: 1.0,
            quality_horizon_min_steps: 0,
            min_quality_mean_adaptive_psnr_db: 0.0,
            min_quality_worst_adaptive_psnr_db: 0.0,
            max_quality_mean_oracle_gap_db: 0.5,
            max_quality_worst_oracle_gap_db: 0.5,
            max_quality_mean_same_rule_fine_gap_db: 0.5,
            max_quality_worst_same_rule_fine_gap_db: 0.5,
            max_material_relative_error: 2.0e-5,
            max_out_of_bounds_fraction: 0.01,
            max_grid_overflow: 0,
            max_interaction_work_ratio: 0.8,
            max_wall_time_ratio: 1.1,
            require_adaptive_resolution: true,
            adaptive_resolution_horizon_min_steps: 0,
            min_material_scale_ratio: 2.0,
            min_adaptive_support_bin_count: 2,
            min_occupied_material_scale_bins: 2,
            min_accepted_local_detail_exchanges: 1,
            min_mean_scale_detail_correlation_gain_vs_static: 0.1,
            min_worst_scale_detail_correlation_gain_vs_static: -0.1,
            min_mean_static_graded_psnr_gain_db: 0.25,
            min_worst_static_graded_psnr_gain_db: -0.25,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveTarget2dValidationRow {
    pub seed: u64,
    pub horizon: usize,
    pub adaptive_psnr_db: f32,
    pub adaptive_composited_psnr_db: f32,
    #[serde(default)]
    pub static_graded_psnr_db: f32,
    pub same_rule_fine_psnr_db: f32,
    pub oracle_psnr_db: f32,
    pub adaptive_oracle_gap_db: f32,
    #[serde(default)]
    pub adaptive_static_graded_gap_db: f32,
    pub adaptive_same_rule_fine_gap_db: f32,
    pub adaptive_visible_rows: usize,
    pub adaptive_dynamics_rows: usize,
    pub adaptive_interaction_rows: usize,
    pub hidden_rows: usize,
    pub adaptive_neighbor_mode: String,
    pub adaptive_support_bin_count: usize,
    pub adaptive_requested_support_bin_count: usize,
    pub oracle_neighbor_mode: String,
    pub topology_passes: usize,
    pub topology_events: usize,
    #[serde(default)]
    pub accepted_local_detail_exchanges: usize,
    #[serde(default)]
    pub topology_acceptance_fraction: f32,
    #[serde(default)]
    pub initial_scale_detail_correlation: f32,
    #[serde(default)]
    pub scale_detail_correlation: f32,
    #[serde(default)]
    pub scale_detail_correlation_gain: f32,
    #[serde(default)]
    pub static_scale_detail_correlation: f32,
    #[serde(default)]
    pub scale_detail_correlation_gain_vs_static: f32,
    #[serde(default)]
    pub fine_to_coarse_detail_ratio: f32,
    #[serde(default)]
    pub static_fine_to_coarse_detail_ratio: f32,
    #[serde(default)]
    pub material_scale_ratio: f32,
    #[serde(default)]
    pub occupied_material_scale_bins: usize,
    #[serde(default)]
    pub fractional_material_scale_fraction: f32,
    pub material_relative_error: f32,
    pub occupied_pixel_fraction: f32,
    pub out_of_bounds_fraction: f32,
    pub retained_identity_fraction: f32,
    pub retained_identity_motion_per_step: f32,
    pub grid_overflow: u32,
    pub adaptive_particle_steps: usize,
    pub oracle_particle_steps: usize,
    pub interaction_work_ratio: f32,
    pub theoretical_pair_work_ratio: f32,
    pub adaptive_elapsed_ms: f64,
    pub same_rule_elapsed_ms: f64,
    pub oracle_elapsed_ms: f64,
    /// Adaptive elapsed time divided by the matched 4,096-row rollout using
    /// the same shared rule.
    pub wall_time_ratio: f32,
    /// Adaptive elapsed time divided by the external catalog-oracle rollout.
    pub oracle_wall_time_ratio: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveTarget2dValidationReport {
    pub objective: String,
    pub active_particle_count: usize,
    pub initial_particle_count: usize,
    pub delayed_restriction_step: usize,
    pub reallocation_interval_steps: usize,
    pub reallocation_start_step: usize,
    pub reallocation_end_step: usize,
    pub min_reallocation_relative_gain: f32,
    pub reference_particle_count: usize,
    pub visible_gaussian_count: usize,
    pub hidden_fine_rows: usize,
    pub topology_enabled: bool,
    pub mean_adaptive_psnr_db: f32,
    pub worst_adaptive_psnr_db: f32,
    pub mean_adaptive_oracle_gap_db: f32,
    pub worst_adaptive_oracle_gap_db: f32,
    #[serde(default)]
    pub mean_static_graded_gap_db: f32,
    #[serde(default)]
    pub worst_static_graded_gap_db: f32,
    pub mean_same_rule_fine_gap_db: f32,
    pub worst_same_rule_fine_gap_db: f32,
    pub worst_psnr_drift_db: f32,
    #[serde(default)]
    pub total_accepted_local_detail_exchanges: usize,
    #[serde(default)]
    pub mean_topology_acceptance_fraction: f32,
    #[serde(default)]
    pub mean_scale_detail_correlation_gain: f32,
    #[serde(default)]
    pub worst_scale_detail_correlation_gain: f32,
    #[serde(default)]
    pub mean_scale_detail_correlation_gain_vs_static: f32,
    #[serde(default)]
    pub worst_scale_detail_correlation_gain_vs_static: f32,
    pub mean_interaction_work_ratio: f32,
    pub mean_wall_time_ratio: f32,
    pub mean_oracle_wall_time_ratio: f32,
    #[serde(default)]
    pub adaptive_resolution: AdaptiveResolutionValidationSummary,
    #[serde(default)]
    pub horizon_summaries: Vec<AdaptiveTarget2dHorizonSummary>,
    pub rows: Vec<AdaptiveTarget2dValidationRow>,
    pub failures: Vec<String>,
    pub passed: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AdaptiveResolutionValidationSummary {
    pub required: bool,
    pub horizon_min_steps: usize,
    pub rows: usize,
    pub minimum_material_scale_ratio: f32,
    pub minimum_support_bin_count: usize,
    pub minimum_occupied_material_scale_bins: usize,
    pub accepted_local_detail_exchanges: usize,
    pub mean_scale_detail_correlation_gain_vs_static: f32,
    pub worst_scale_detail_correlation_gain_vs_static: f32,
    pub mean_static_graded_psnr_gain_db: f32,
    pub worst_static_graded_psnr_gain_db: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveTarget2dHorizonSummary {
    pub horizon: usize,
    pub seeds: usize,
    pub mean_adaptive_psnr_db: f32,
    pub worst_adaptive_psnr_db: f32,
    pub mean_static_graded_psnr_db: f32,
    pub worst_static_graded_psnr_db: f32,
    pub mean_same_rule_fine_psnr_db: f32,
    pub worst_same_rule_fine_psnr_db: f32,
    pub mean_oracle_psnr_db: f32,
    pub worst_oracle_psnr_db: f32,
    pub adaptive_mean_oracle_gap_db: f32,
    pub adaptive_worst_oracle_gap_db: f32,
    pub adaptive_mean_same_rule_fine_gap_db: f32,
    pub adaptive_worst_same_rule_fine_gap_db: f32,
}

impl AdaptiveTarget2dValidationReport {
    pub fn require_pass(&self) -> Result<(), crate::AutomataError> {
        if self.passed {
            Ok(())
        } else {
            Err(crate::AutomataError::InvalidModel(format!(
                "adaptive Target2D validation failed: {}",
                self.failures.join("; ")
            )))
        }
    }
}

#[cfg(feature = "gpu_wgpu")]
#[allow(clippy::too_many_arguments)]
pub fn validate_adaptive_target2d_wgpu(
    model: &super::AdaptiveNpaModel,
    oracle: &crate::NpaModel,
    grid: &burn_automata_kernels::HashGridConfig,
    target: &crate::TargetImage2d,
    material: super::AdaptiveTarget2dMaterialConfig,
    loss: crate::Target2dLossConfig,
    config: &AdaptiveTarget2dValidationConfig,
) -> crate::AutomataResult<AdaptiveTarget2dValidationReport> {
    use std::{collections::BTreeMap, time::Instant};

    use crate::{
        gpu::{WgpuAutomataExecutor, WgpuNeighborMode},
        rollout::seed_particles_scaled,
        target2d::{
            render_active_material_rollout_2d_splat, render_rollout_2d_splat,
            render_target_2d_splat,
        },
    };

    model.validate()?;
    oracle.validate()?;
    grid.validate()?;
    let active = if config.active_particle_count == 0 {
        model.config.target_leaves
    } else {
        config.active_particle_count
    };
    let canonical_rule_path = model.local_residual_rule.is_none()
        || model.uses_canonical_compatible_residual()
        || model.uses_canonical_normalized_residual();
    let mut canonical_rule_config = model.rule.config.clone();
    if model.config.material_scale_conditioning {
        canonical_rule_config.auxiliary_input_dims = canonical_rule_config
            .auxiliary_input_dims
            .checked_sub(1)
            .ok_or_else(|| {
                crate::AutomataError::InvalidModel(
                    "material-scale-conditioned rule is missing its auxiliary input".to_owned(),
                )
            })?;
    }
    canonical_rule_config.state_dims = canonical_rule_config
        .state_dims
        .checked_sub(model.config.compact_recurrent_memory_dims)
        .ok_or_else(|| {
            crate::AutomataError::InvalidModel(
                "compact recurrent memory exceeds the adaptive rule state width".to_owned(),
            )
        })?;
    if canonical_rule_config != oracle.config
        || active == 0
        || active > model.config.max_leaves
        || material.reference_particle_count <= active
        || model.config.bootstrap_fine_leaf_count() != material.reference_particle_count
        || model.config.retain_bootstrap_templates
        || model.config.coarse_dynamics != super::AdaptiveCoarseDynamics::RepresentedMeasure
        || !canonical_rule_path
        || model.proxy_rule.is_some()
        || model.deployment_rule.is_some()
        || model.deployment_local_rule.is_some()
        || model.closure_mode_rule.is_some()
    {
        return Err(crate::AutomataError::InvalidArgument(
            "adaptive Target2D validation requires a direct shared rule or canonical same-row adaptive residual, no hidden fine templates/other rules, and a matching 4096-row oracle"
                .to_owned(),
        ));
    }
    let mut horizons = config.horizons.clone();
    horizons.sort_unstable();
    horizons.dedup();
    let mut seeds = config.seeds.clone();
    seeds.sort_unstable();
    seeds.dedup();
    let max_horizon = horizons.last().copied().unwrap_or(0);
    let delayed_restriction = config.delayed_restriction_step > 0;
    let paired_reallocation = config.reallocation_interval_steps > 0;
    if horizons.is_empty()
        || horizons[0] == 0
        || seeds.is_empty()
        || !config.update_prob.is_finite()
        || !(0.0..=1.0).contains(&config.update_prob)
        || config.update_prob == 0.0
        || !config.seed_scale.is_finite()
        || config.seed_scale <= 0.0
        || !config.dt.is_finite()
        || config.dt <= 0.0
        || !config.min_adaptive_psnr_db.is_finite()
        || !config.max_oracle_psnr_gap_db.is_finite()
        || config.max_oracle_psnr_gap_db < 0.0
        || !config.max_same_rule_fine_psnr_gap_db.is_finite()
        || config.max_same_rule_fine_psnr_gap_db < 0.0
        || !config.max_psnr_drift_db.is_finite()
        || config.max_psnr_drift_db < 0.0
        || !config.min_quality_mean_adaptive_psnr_db.is_finite()
        || !config.min_quality_worst_adaptive_psnr_db.is_finite()
        || !config.max_quality_mean_oracle_gap_db.is_finite()
        || config.max_quality_mean_oracle_gap_db < 0.0
        || !config.max_quality_worst_oracle_gap_db.is_finite()
        || config.max_quality_worst_oracle_gap_db < 0.0
        || !config.max_quality_mean_same_rule_fine_gap_db.is_finite()
        || config.max_quality_mean_same_rule_fine_gap_db < 0.0
        || !config.max_quality_worst_same_rule_fine_gap_db.is_finite()
        || config.max_quality_worst_same_rule_fine_gap_db < 0.0
        || !config.max_material_relative_error.is_finite()
        || config.max_material_relative_error < 0.0
        || !config.max_out_of_bounds_fraction.is_finite()
        || !(0.0..=1.0).contains(&config.max_out_of_bounds_fraction)
        || !config.max_interaction_work_ratio.is_finite()
        || config.max_interaction_work_ratio <= 0.0
        || !config.max_wall_time_ratio.is_finite()
        || config.max_wall_time_ratio <= 0.0
        || !config.min_material_scale_ratio.is_finite()
        || config.min_material_scale_ratio < 1.0
        || config.min_adaptive_support_bin_count == 0
        || config.min_occupied_material_scale_bins == 0
        || !config
            .min_mean_scale_detail_correlation_gain_vs_static
            .is_finite()
        || !config
            .min_worst_scale_detail_correlation_gain_vs_static
            .is_finite()
        || !config.min_mean_static_graded_psnr_gain_db.is_finite()
        || !config.min_worst_static_graded_psnr_gain_db.is_finite()
        || (config.adaptive_resolution_horizon_min_steps > 0
            && config.adaptive_resolution_horizon_min_steps > max_horizon)
        || !config.min_reallocation_relative_gain.is_finite()
        || !(0.0..=1.0).contains(&config.min_reallocation_relative_gain)
        || (delayed_restriction
            && (!config.topology_enabled
                || config.delayed_restriction_step > max_horizon
                || !(material.reference_particle_count - active)
                    .is_multiple_of(2 * model.config.spatial_dims - 1)))
        || (paired_reallocation && (!config.topology_enabled || !delayed_restriction))
    {
        return Err(crate::AutomataError::InvalidArgument(
            "adaptive Target2D validation seeds, horizons, or rollout values are invalid"
                .to_owned(),
        ));
    }
    let reference = material.reference_particle_count;
    let initial = if delayed_restriction {
        reference
    } else {
        active
    };
    let default_reallocation_start = config
        .delayed_restriction_step
        .saturating_add(config.reallocation_interval_steps)
        .max(config.reallocation_interval_steps);
    let reallocation_start = if paired_reallocation {
        if config.reallocation_start_step == 0 {
            default_reallocation_start
        } else {
            config.reallocation_start_step
        }
    } else {
        0
    };
    let reallocation_end = if paired_reallocation {
        if config.reallocation_end_step == 0 {
            max_horizon
        } else {
            config.reallocation_end_step
        }
    } else {
        0
    };
    if paired_reallocation
        && (reallocation_start <= config.delayed_restriction_step
            || reallocation_start > reallocation_end
            || reallocation_end > max_horizon)
    {
        return Err(crate::AutomataError::InvalidArgument(
            "adaptive Target2D paired reallocation must start after the delayed cut and within the validation horizon"
                .to_owned(),
        ));
    }
    let mut evaluation_model = model.clone();
    if delayed_restriction || paired_reallocation {
        // Isolate the one-shot restriction from periodic split/merge control.
        // The active state starts as the exact matched fine seed, then owns
        // only the represented-measure rows after the configured cut.
        evaluation_model.config.initial_leaves = reference;
        evaluation_model.config.min_leaves = evaluation_model.config.min_leaves.min(active);
        evaluation_model.config.target_leaves = active;
        evaluation_model.config.bootstrap_target_leaves = reference;
        evaluation_model.config.hierarchical_bootstrap_seed = true;
        evaluation_model.config.hierarchical_restriction_step = config.delayed_restriction_step;
        evaluation_model
            .config
            .hierarchical_restriction_leaf_delta_per_interval = 0;
        evaluation_model.config.hierarchical_restriction_arity =
            super::AdaptiveRestrictionArity::Canonical;
        evaluation_model.config.hierarchical_restriction_policy = config.restriction_policy;
        evaluation_model.config.retain_bootstrap_templates = false;
        if paired_reallocation {
            evaluation_model.config.topology_interval = config.reallocation_interval_steps;
            evaluation_model.config.steady_topology_interval = config.reallocation_interval_steps;
            evaluation_model.config.topology_start_step = reallocation_start;
            evaluation_model.config.steady_topology_start_step = reallocation_start;
            evaluation_model.config.topology_end_step = reallocation_end;
            evaluation_model.config.runtime_topology_control =
                if material.seed_layout == super::AdaptiveMaterialSeedLayout::GradedContinuous {
                    super::AdaptiveTopologyControl::ContinuousLocalDetail
                } else {
                    evaluation_model.config.max_events_per_interval = 1;
                    super::AdaptiveTopologyControl::PairedLocalDetail
                };
            evaluation_model.config.min_reallocation_relative_gain =
                config.min_reallocation_relative_gain;
        } else {
            evaluation_model.config.topology_start_step = max_horizon.saturating_add(1);
            evaluation_model.config.steady_topology_start_step = max_horizon.saturating_add(1);
        }
        if !paired_reallocation {
            evaluation_model.config.topology_end_step = 0;
        }
        evaluation_model.validate()?;
    }
    let local_detail_reallocation = config.topology_enabled
        && matches!(
            evaluation_model.config.runtime_topology_control,
            super::AdaptiveTopologyControl::PairedLocalDetail
                | super::AdaptiveTopologyControl::ContinuousLocalDetail
        );
    let material_layout = material.layout(
        active,
        model.config.perception.min_bandwidth,
        model.config.perception.max_bandwidth,
    )?;
    let fine_measure = material.total_measure / reference as f32;
    let output_scale = target.point_count() as f32 / reference as f32;
    let target_render = render_target_2d_splat(target, loss)?;
    let executor = WgpuAutomataExecutor::new_blocking()?;
    let mut rows = Vec::with_capacity(seeds.len() * horizons.len());

    for seed in seeds.iter().copied() {
        let adaptive_seed = if delayed_restriction {
            super::seed_adaptive_particles_scaled(
                &evaluation_model,
                reference,
                seed,
                config.seed_mode,
                config.seed_scale,
                material.total_measure,
                material.fine_bandwidth,
            )?
        } else {
            super::training::adaptive_target2d_seed_particles(
                &evaluation_model,
                &material_layout,
                seed,
                config.seed_mode,
                config.seed_scale,
                material.total_measure,
                material.fine_bandwidth,
            )?
        };
        if !adaptive_seed.bootstrap_templates.is_empty() {
            return Err(crate::AutomataError::InvalidModel(
                "adaptive validation seed retained hidden fine templates".to_owned(),
            ));
        }
        let initial_adaptation =
            adaptation_diagnostics(&evaluation_model, &adaptive_seed, fine_measure)?;
        let static_adaptive_seed = adaptive_seed.clone();
        let (fine_positions, fine_states) = seed_particles_scaled(
            1,
            reference,
            model.rule.config.state_dims,
            2,
            seed,
            config.seed_mode,
            config.seed_scale,
        );
        let oracle_fine_states = canonical_oracle_states(model, &fine_states)?;
        let mut adaptive_state = executor.create_adaptive_state(
            &evaluation_model,
            adaptive_seed,
            grid,
            config.dt,
            WgpuNeighborMode::Auto,
            config.update_prob,
            seed,
        )?;
        let mut static_adaptive_state = executor.create_adaptive_state(
            &evaluation_model,
            static_adaptive_seed,
            grid,
            config.dt,
            WgpuNeighborMode::Auto,
            config.update_prob,
            seed,
        )?;
        let mut same_rule_state = executor.create_state_with_neighbor_mode_and_update_prob(
            &model.rule,
            &fine_positions,
            &fine_states,
            1,
            reference,
            grid,
            config.dt,
            WgpuNeighborMode::Auto,
            config.update_prob,
            seed,
        )?;
        let mut oracle_state = executor.create_state_with_neighbor_mode_and_update_prob(
            oracle,
            &fine_positions,
            &oracle_fine_states,
            1,
            reference,
            grid,
            config.dt,
            WgpuNeighborMode::Auto,
            config.update_prob,
            seed,
        )?;
        // The parity baseline must use the same deterministic within-cell
        // ordering as adaptive dynamics. Otherwise atomic scatter order can
        // dominate the measured long-horizon compression gap.
        executor.set_adaptive_stable_sorted_cells_enabled(&mut adaptive_state, true);
        executor.set_adaptive_stable_sorted_cells_enabled(&mut static_adaptive_state, true);
        executor.set_stable_sorted_cells_enabled(&mut same_rule_state, true);
        executor.set_stable_sorted_cells_enabled(&mut oracle_state, true);
        let mut completed = 0usize;
        let mut adaptive_particle_steps = 0usize;
        let mut oracle_particle_steps = 0usize;
        let mut topology_passes = 0usize;
        let mut local_detail_topology_passes = 0usize;
        let mut non_paired_topology_events = 0usize;
        let mut adaptive_elapsed_ms_total = 0.0_f64;
        let mut same_rule_elapsed_ms_total = 0.0_f64;
        let mut oracle_elapsed_ms_total = 0.0_f64;
        let mut prior_identity_positions = None::<BTreeMap<u64, [f32; 2]>>;

        for horizon in horizons.iter().copied() {
            let steps = horizon - completed;
            executor.wait_idle()?;
            let adaptive_started = Instant::now();
            let adaptive_step = executor.step_adaptive_state_many(
                &mut adaptive_state,
                steps,
                config.topology_enabled,
            )?;
            executor.wait_idle()?;
            let adaptive_elapsed_ms = adaptive_started.elapsed().as_secs_f64() * 1_000.0;
            adaptive_elapsed_ms_total += adaptive_elapsed_ms;
            adaptive_particle_steps =
                adaptive_particle_steps.saturating_add(adaptive_step.interaction_particle_steps);
            topology_passes = topology_passes.saturating_add(adaptive_step.topology_updates.len());
            local_detail_topology_passes = local_detail_topology_passes.saturating_add(
                adaptive_step
                    .topology_updates
                    .iter()
                    .filter(|update| {
                        local_detail_reallocation
                            && (!paired_reallocation
                                || (update.step >= reallocation_start
                                    && update.step <= reallocation_end))
                    })
                    .count(),
            );
            non_paired_topology_events = non_paired_topology_events.saturating_add(
                adaptive_step
                    .topology_updates
                    .iter()
                    .filter(|update| {
                        !local_detail_reallocation
                            || (paired_reallocation
                                && (update.step < reallocation_start
                                    || update.step > reallocation_end))
                    })
                    .map(|update| update.split_events + update.merge_events)
                    .sum::<usize>(),
            );
            let accepted_local_detail_exchanges = if local_detail_reallocation {
                executor.read_adaptive_local_detail_topology_accept_count(&adaptive_state)?
            } else {
                0
            };
            let topology_events =
                non_paired_topology_events.saturating_add(2 * accepted_local_detail_exchanges);
            executor.step_adaptive_state_many(&mut static_adaptive_state, steps, false)?;
            executor.wait_idle()?;
            let same_rule_started = Instant::now();
            executor.step_state_many(&mut same_rule_state, steps)?;
            executor.wait_idle()?;
            let same_rule_elapsed_ms = same_rule_started.elapsed().as_secs_f64() * 1_000.0;
            same_rule_elapsed_ms_total += same_rule_elapsed_ms;
            executor.wait_idle()?;
            let oracle_started = Instant::now();
            executor.step_state_many(&mut oracle_state, steps)?;
            executor.wait_idle()?;
            let oracle_elapsed_ms = oracle_started.elapsed().as_secs_f64() * 1_000.0;
            oracle_elapsed_ms_total += oracle_elapsed_ms;
            oracle_particle_steps = oracle_particle_steps.saturating_add(reference * steps);

            executor.synchronize_adaptive_particles(&mut adaptive_state)?;
            executor.synchronize_adaptive_particles(&mut static_adaptive_state)?;
            let (same_rule_positions, same_rule_states) =
                executor.read_positions_states(&same_rule_state)?;
            let (oracle_positions, oracle_states) =
                executor.read_positions_states(&oracle_state)?;
            // Deployment and viewer rollouts are not translated after dynamics.
            // Keep validation PSNR on that contract even when the differentiable
            // training objective uses translation-invariant centering.
            let adaptive_render = render_active_material_rollout_2d_splat(
                &adaptive_state.particles,
                fine_measure,
                target.pixel_size,
                target.point_count(),
                loss,
                None,
            )?;
            let static_graded_render = render_active_material_rollout_2d_splat(
                &static_adaptive_state.particles,
                fine_measure,
                target.pixel_size,
                target.point_count(),
                loss,
                None,
            )?;
            let same_rule_render = render_rollout_2d_splat(
                &same_rule_positions,
                &same_rule_states,
                model.rule.config.state_dims,
                target.pixel_size,
                loss,
                None,
                output_scale,
            )?;
            let oracle_render = render_rollout_2d_splat(
                &oracle_positions,
                &oracle_states,
                oracle.config.state_dims,
                target.pixel_size,
                loss,
                None,
                output_scale,
            )?;
            let adaptive_psnr = raw_psnr(&adaptive_render, &target_render);
            let static_graded_psnr = raw_psnr(&static_graded_render, &target_render);
            let same_rule_psnr = raw_psnr(&same_rule_render, &target_render);
            let oracle_psnr = raw_psnr(&oracle_render, &target_render);
            let current_identity_positions = adaptive_state
                .particles
                .particle_id
                .iter()
                .copied()
                .zip(&adaptive_state.particles.positions)
                .map(|(id, position)| (id, [position[0], position[1]]))
                .collect::<BTreeMap<_, _>>();
            let (retained_identity_fraction, retained_identity_motion_per_step) = identity_motion(
                prior_identity_positions.as_ref(),
                &current_identity_positions,
                steps,
            );
            prior_identity_positions = Some(current_identity_positions);
            let material_relative_error =
                (adaptive_state.particles.total_measure() as f32 - material.total_measure).abs()
                    / material.total_measure.max(f32::MIN_POSITIVE);
            let out_of_bounds = adaptive_state
                .particles
                .positions
                .iter()
                .filter(|position| {
                    position[0] < target.aabb[0]
                        || position[0] > target.aabb[1]
                        || position[1] < target.aabb[2]
                        || position[1] > target.aabb[3]
                })
                .count() as f32
                / adaptive_state.particles.len().max(1) as f32;
            let occupied_pixel_fraction = adaptive_render
                .density
                .iter()
                .filter(|density| **density > target.threshold)
                .count() as f32
                / adaptive_render.density.len().max(1) as f32;
            let grid_overflow = executor.read_adaptive_grid_overflow(&adaptive_state)?;
            let adaptation =
                adaptation_diagnostics(&evaluation_model, &adaptive_state.particles, fine_measure)?;
            let static_adaptation = adaptation_diagnostics(
                &evaluation_model,
                &static_adaptive_state.particles,
                fine_measure,
            )?;
            let adaptive_neighbor = executor.neighbor_report(&adaptive_state.resident);
            let oracle_neighbor = executor.neighbor_report(&oracle_state);
            let hidden_rows = adaptive_step
                .dynamics_particle_count
                .saturating_sub(adaptive_step.resident_particle_count);
            let interaction_work_ratio =
                adaptive_particle_steps as f32 / oracle_particle_steps.max(1) as f32;
            rows.push(AdaptiveTarget2dValidationRow {
                seed,
                horizon,
                adaptive_psnr_db: adaptive_psnr,
                adaptive_composited_psnr_db: composited_psnr(&adaptive_render, &target_render),
                static_graded_psnr_db: static_graded_psnr,
                same_rule_fine_psnr_db: same_rule_psnr,
                oracle_psnr_db: oracle_psnr,
                adaptive_oracle_gap_db: adaptive_psnr - oracle_psnr,
                adaptive_static_graded_gap_db: adaptive_psnr - static_graded_psnr,
                adaptive_same_rule_fine_gap_db: adaptive_psnr - same_rule_psnr,
                adaptive_visible_rows: adaptive_step.resident_particle_count,
                adaptive_dynamics_rows: adaptive_step.dynamics_particle_count,
                adaptive_interaction_rows: adaptive_step.interaction_particle_count,
                hidden_rows,
                adaptive_neighbor_mode: format!("{:?}", adaptive_neighbor.mode),
                adaptive_support_bin_count: adaptive_neighbor.support_bin_count,
                adaptive_requested_support_bin_count: adaptive_neighbor.requested_support_bin_count,
                oracle_neighbor_mode: format!("{:?}", oracle_neighbor.mode),
                topology_passes,
                topology_events,
                accepted_local_detail_exchanges,
                topology_acceptance_fraction: accepted_local_detail_exchanges as f32
                    / local_detail_topology_passes
                        .saturating_mul(evaluation_model.config.max_events_per_interval)
                        .max(1) as f32,
                initial_scale_detail_correlation: initial_adaptation.scale_detail_correlation,
                scale_detail_correlation: adaptation.scale_detail_correlation,
                scale_detail_correlation_gain: adaptation.scale_detail_correlation
                    - initial_adaptation.scale_detail_correlation,
                static_scale_detail_correlation: static_adaptation.scale_detail_correlation,
                scale_detail_correlation_gain_vs_static: adaptation.scale_detail_correlation
                    - static_adaptation.scale_detail_correlation,
                fine_to_coarse_detail_ratio: adaptation.fine_to_coarse_detail_ratio,
                static_fine_to_coarse_detail_ratio: static_adaptation.fine_to_coarse_detail_ratio,
                material_scale_ratio: adaptation.material_scale_ratio,
                occupied_material_scale_bins: adaptation.occupied_material_scale_bins,
                fractional_material_scale_fraction: adaptation.fractional_material_scale_fraction,
                material_relative_error,
                occupied_pixel_fraction,
                out_of_bounds_fraction: out_of_bounds,
                retained_identity_fraction,
                retained_identity_motion_per_step,
                grid_overflow,
                adaptive_particle_steps,
                oracle_particle_steps,
                interaction_work_ratio,
                theoretical_pair_work_ratio: theoretical_pair_work_ratio(
                    active,
                    reference,
                    horizon,
                    config.delayed_restriction_step,
                    topology_passes,
                ),
                adaptive_elapsed_ms: adaptive_elapsed_ms_total,
                same_rule_elapsed_ms: same_rule_elapsed_ms_total,
                oracle_elapsed_ms: oracle_elapsed_ms_total,
                wall_time_ratio: (adaptive_elapsed_ms_total
                    / same_rule_elapsed_ms_total.max(f64::MIN_POSITIVE))
                    as f32,
                oracle_wall_time_ratio: (adaptive_elapsed_ms_total
                    / oracle_elapsed_ms_total.max(f64::MIN_POSITIVE))
                    as f32,
            });
            completed = horizon;
        }
    }
    summarize_validation(
        rows,
        AdaptiveValidationSummaryContext {
            active,
            reference,
            initial,
            delayed_restriction_step: config.delayed_restriction_step,
            reallocation_interval_steps: config.reallocation_interval_steps,
            reallocation_start_step: reallocation_start,
            reallocation_end_step: reallocation_end,
            topology_start_step: evaluation_model.config.topology_start_step,
            topology_interval_steps: evaluation_model.config.topology_interval,
            topology_end_step: evaluation_model.config.topology_end_step,
            min_reallocation_relative_gain: evaluation_model.config.min_reallocation_relative_gain,
            max_events_per_interval: evaluation_model.config.max_events_per_interval,
        },
        config,
    )
}

#[cfg(feature = "gpu_wgpu")]
fn theoretical_pair_work_ratio(
    active: usize,
    reference: usize,
    horizon: usize,
    delayed_restriction_step: usize,
    topology_passes: usize,
) -> f32 {
    let fine_steps = if delayed_restriction_step == 0 {
        0
    } else {
        horizon.min(delayed_restriction_step)
    };
    let active_steps = horizon - fine_steps;
    let work = reference
        .pow(2)
        .saturating_mul(fine_steps)
        .saturating_add(active.pow(2).saturating_mul(active_steps))
        .saturating_add(active.saturating_mul(topology_passes));
    work as f32 / reference.pow(2).saturating_mul(horizon).max(1) as f32
}

#[cfg(feature = "gpu_wgpu")]
fn periodic_passes(horizon: usize, start: usize, interval: usize, end: usize) -> usize {
    if interval == 0 {
        0
    } else {
        let first = start.max(1).div_ceil(interval) * interval;
        let last = horizon.min(if end == 0 { horizon } else { end });
        if last < first {
            0
        } else {
            1 + (last - first) / interval
        }
    }
}

#[cfg(feature = "gpu_wgpu")]
#[derive(Clone, Copy)]
struct AdaptiveValidationSummaryContext {
    active: usize,
    reference: usize,
    initial: usize,
    delayed_restriction_step: usize,
    reallocation_interval_steps: usize,
    reallocation_start_step: usize,
    reallocation_end_step: usize,
    topology_start_step: usize,
    topology_interval_steps: usize,
    topology_end_step: usize,
    min_reallocation_relative_gain: f32,
    max_events_per_interval: usize,
}

#[cfg(feature = "gpu_wgpu")]
fn summarize_validation(
    rows: Vec<AdaptiveTarget2dValidationRow>,
    context: AdaptiveValidationSummaryContext,
    config: &AdaptiveTarget2dValidationConfig,
) -> crate::AutomataResult<AdaptiveTarget2dValidationReport> {
    use std::collections::BTreeMap;

    let AdaptiveValidationSummaryContext {
        active,
        reference,
        initial,
        delayed_restriction_step,
        reallocation_interval_steps,
        reallocation_start_step,
        reallocation_end_step,
        topology_start_step,
        topology_interval_steps,
        topology_end_step,
        min_reallocation_relative_gain,
        max_events_per_interval,
    } = context;
    if rows.is_empty() {
        return Err(crate::AutomataError::InvalidArgument(
            "adaptive Target2D validation produced no rows".to_owned(),
        ));
    }
    let count = rows.len() as f32;
    let mean_adaptive_psnr_db = rows.iter().map(|row| row.adaptive_psnr_db).sum::<f32>() / count;
    let worst_adaptive_psnr_db = rows
        .iter()
        .map(|row| row.adaptive_psnr_db)
        .fold(f32::INFINITY, f32::min);
    let mean_adaptive_oracle_gap_db = rows
        .iter()
        .map(|row| row.adaptive_oracle_gap_db)
        .sum::<f32>()
        / count;
    let worst_adaptive_oracle_gap_db = rows
        .iter()
        .map(|row| row.adaptive_oracle_gap_db)
        .fold(f32::INFINITY, f32::min);
    let mean_static_graded_gap_db = rows
        .iter()
        .map(|row| row.adaptive_static_graded_gap_db)
        .sum::<f32>()
        / count;
    let worst_static_graded_gap_db = rows
        .iter()
        .map(|row| row.adaptive_static_graded_gap_db)
        .fold(f32::INFINITY, f32::min);
    let mean_same_rule_fine_gap_db = rows
        .iter()
        .map(|row| row.adaptive_same_rule_fine_gap_db)
        .sum::<f32>()
        / count;
    let worst_same_rule_fine_gap_db = rows
        .iter()
        .map(|row| row.adaptive_same_rule_fine_gap_db)
        .fold(f32::INFINITY, f32::min);
    let mean_interaction_work_ratio = rows
        .iter()
        .map(|row| row.interaction_work_ratio)
        .sum::<f32>()
        / count;
    let mean_wall_time_ratio = rows.iter().map(|row| row.wall_time_ratio).sum::<f32>() / count;
    let mean_oracle_wall_time_ratio = rows
        .iter()
        .map(|row| row.oracle_wall_time_ratio)
        .sum::<f32>()
        / count;
    let mut rows_by_horizon = BTreeMap::<usize, Vec<&AdaptiveTarget2dValidationRow>>::new();
    for row in &rows {
        rows_by_horizon.entry(row.horizon).or_default().push(row);
    }
    let horizon_summaries = rows_by_horizon
        .into_iter()
        .map(|(horizon, horizon_rows)| {
            let seeds = horizon_rows.len();
            let count = seeds.max(1) as f32;
            let mean_adaptive_psnr_db = horizon_rows
                .iter()
                .map(|row| row.adaptive_psnr_db)
                .sum::<f32>()
                / count;
            let worst_adaptive_psnr_db = horizon_rows
                .iter()
                .map(|row| row.adaptive_psnr_db)
                .fold(f32::INFINITY, f32::min);
            let mean_static_graded_psnr_db = horizon_rows
                .iter()
                .map(|row| row.static_graded_psnr_db)
                .sum::<f32>()
                / count;
            let worst_static_graded_psnr_db = horizon_rows
                .iter()
                .map(|row| row.static_graded_psnr_db)
                .fold(f32::INFINITY, f32::min);
            let mean_same_rule_fine_psnr_db = horizon_rows
                .iter()
                .map(|row| row.same_rule_fine_psnr_db)
                .sum::<f32>()
                / count;
            let worst_same_rule_fine_psnr_db = horizon_rows
                .iter()
                .map(|row| row.same_rule_fine_psnr_db)
                .fold(f32::INFINITY, f32::min);
            let mean_oracle_psnr_db = horizon_rows
                .iter()
                .map(|row| row.oracle_psnr_db)
                .sum::<f32>()
                / count;
            let worst_oracle_psnr_db = horizon_rows
                .iter()
                .map(|row| row.oracle_psnr_db)
                .fold(f32::INFINITY, f32::min);
            AdaptiveTarget2dHorizonSummary {
                horizon,
                seeds,
                mean_adaptive_psnr_db,
                worst_adaptive_psnr_db,
                mean_static_graded_psnr_db,
                worst_static_graded_psnr_db,
                mean_same_rule_fine_psnr_db,
                worst_same_rule_fine_psnr_db,
                mean_oracle_psnr_db,
                worst_oracle_psnr_db,
                adaptive_mean_oracle_gap_db: mean_adaptive_psnr_db - mean_oracle_psnr_db,
                adaptive_worst_oracle_gap_db: worst_adaptive_psnr_db - worst_oracle_psnr_db,
                adaptive_mean_same_rule_fine_gap_db: mean_adaptive_psnr_db
                    - mean_same_rule_fine_psnr_db,
                adaptive_worst_same_rule_fine_gap_db: worst_adaptive_psnr_db
                    - worst_same_rule_fine_psnr_db,
            }
        })
        .collect::<Vec<_>>();
    let mut psnr_by_seed = BTreeMap::<u64, Vec<(usize, f32)>>::new();
    for row in &rows {
        psnr_by_seed
            .entry(row.seed)
            .or_default()
            .push((row.horizon, row.adaptive_psnr_db));
    }
    let worst_psnr_drift_db = psnr_by_seed
        .values_mut()
        .map(|values| {
            values.sort_by_key(|(horizon, _)| *horizon);
            let mut peak = f32::NEG_INFINITY;
            let mut worst_drop = 0.0_f32;
            for (_, value) in values {
                peak = peak.max(*value);
                worst_drop = worst_drop.max(peak - *value);
            }
            worst_drop
        })
        .fold(0.0_f32, f32::max);
    let max_horizon = rows.iter().map(|row| row.horizon).max().unwrap_or(0);
    let total_accepted_local_detail_exchanges = rows
        .iter()
        .filter(|row| row.horizon == max_horizon)
        .map(|row| row.accepted_local_detail_exchanges)
        .sum();
    let topology_rows = rows
        .iter()
        .filter(|row| row.topology_passes > 0)
        .collect::<Vec<_>>();
    let mean_topology_acceptance_fraction = if topology_rows.is_empty() {
        0.0
    } else {
        topology_rows
            .iter()
            .map(|row| row.topology_acceptance_fraction)
            .sum::<f32>()
            / topology_rows.len() as f32
    };
    let mean_scale_detail_correlation_gain = rows
        .iter()
        .map(|row| row.scale_detail_correlation_gain)
        .sum::<f32>()
        / count;
    let worst_scale_detail_correlation_gain = rows
        .iter()
        .map(|row| row.scale_detail_correlation_gain)
        .fold(f32::INFINITY, f32::min);
    let mean_scale_detail_correlation_gain_vs_static = rows
        .iter()
        .map(|row| row.scale_detail_correlation_gain_vs_static)
        .sum::<f32>()
        / count;
    let worst_scale_detail_correlation_gain_vs_static = rows
        .iter()
        .map(|row| row.scale_detail_correlation_gain_vs_static)
        .fold(f32::INFINITY, f32::min);
    let adaptive_resolution_horizon_min_steps = if config.adaptive_resolution_horizon_min_steps > 0
    {
        config.adaptive_resolution_horizon_min_steps
    } else if config.quality_horizon_min_steps > 0 {
        config.quality_horizon_min_steps
    } else {
        max_horizon
    };
    let adaptive_resolution_rows = rows
        .iter()
        .filter(|row| row.horizon >= adaptive_resolution_horizon_min_steps)
        .collect::<Vec<_>>();
    let adaptive_resolution_count = adaptive_resolution_rows.len();
    let adaptive_resolution = AdaptiveResolutionValidationSummary {
        required: config.require_adaptive_resolution,
        horizon_min_steps: adaptive_resolution_horizon_min_steps,
        rows: adaptive_resolution_count,
        minimum_material_scale_ratio: adaptive_resolution_rows
            .iter()
            .map(|row| row.material_scale_ratio)
            .fold(f32::INFINITY, f32::min),
        minimum_support_bin_count: adaptive_resolution_rows
            .iter()
            .map(|row| row.adaptive_support_bin_count)
            .min()
            .unwrap_or_default(),
        minimum_occupied_material_scale_bins: adaptive_resolution_rows
            .iter()
            .map(|row| row.occupied_material_scale_bins)
            .min()
            .unwrap_or_default(),
        accepted_local_detail_exchanges: total_accepted_local_detail_exchanges,
        mean_scale_detail_correlation_gain_vs_static: adaptive_resolution_rows
            .iter()
            .map(|row| row.scale_detail_correlation_gain_vs_static)
            .sum::<f32>()
            / adaptive_resolution_count.max(1) as f32,
        worst_scale_detail_correlation_gain_vs_static: adaptive_resolution_rows
            .iter()
            .map(|row| row.scale_detail_correlation_gain_vs_static)
            .fold(f32::INFINITY, f32::min),
        mean_static_graded_psnr_gain_db: adaptive_resolution_rows
            .iter()
            .map(|row| row.adaptive_static_graded_gap_db)
            .sum::<f32>()
            / adaptive_resolution_count.max(1) as f32,
        worst_static_graded_psnr_gain_db: adaptive_resolution_rows
            .iter()
            .map(|row| row.adaptive_static_graded_gap_db)
            .fold(f32::INFINITY, f32::min),
    };
    let mut failures = Vec::new();
    for row in &rows {
        let finite = [
            row.adaptive_psnr_db,
            row.adaptive_composited_psnr_db,
            row.static_graded_psnr_db,
            row.same_rule_fine_psnr_db,
            row.oracle_psnr_db,
            row.adaptive_oracle_gap_db,
            row.adaptive_static_graded_gap_db,
            row.adaptive_same_rule_fine_gap_db,
            row.topology_acceptance_fraction,
            row.initial_scale_detail_correlation,
            row.scale_detail_correlation,
            row.scale_detail_correlation_gain,
            row.static_scale_detail_correlation,
            row.scale_detail_correlation_gain_vs_static,
            row.fine_to_coarse_detail_ratio,
            row.static_fine_to_coarse_detail_ratio,
            row.material_scale_ratio,
            row.fractional_material_scale_fraction,
            row.material_relative_error,
            row.occupied_pixel_fraction,
            row.out_of_bounds_fraction,
            row.retained_identity_fraction,
            row.retained_identity_motion_per_step,
            row.interaction_work_ratio,
            row.theoretical_pair_work_ratio,
            row.wall_time_ratio,
            row.oracle_wall_time_ratio,
        ]
        .into_iter()
        .all(f32::is_finite)
            && row.adaptive_elapsed_ms.is_finite()
            && row.same_rule_elapsed_ms.is_finite()
            && row.oracle_elapsed_ms.is_finite();
        if !finite {
            failures.push(format!(
                "seed {} horizon {} produced a non-finite validation metric",
                row.seed, row.horizon
            ));
        }
        if row.adaptive_psnr_db < config.min_adaptive_psnr_db {
            failures.push(format!(
                "seed {} horizon {} adaptive PSNR {:.3} < {:.3}",
                row.seed, row.horizon, row.adaptive_psnr_db, config.min_adaptive_psnr_db
            ));
        }
        if row.adaptive_oracle_gap_db < -config.max_oracle_psnr_gap_db {
            failures.push(format!(
                "seed {} horizon {} oracle gap {:.3} < -{:.3}",
                row.seed, row.horizon, row.adaptive_oracle_gap_db, config.max_oracle_psnr_gap_db
            ));
        }
        if row.adaptive_same_rule_fine_gap_db < -config.max_same_rule_fine_psnr_gap_db {
            failures.push(format!(
                "seed {} horizon {} compression gap {:.3} < -{:.3}",
                row.seed,
                row.horizon,
                row.adaptive_same_rule_fine_gap_db,
                config.max_same_rule_fine_psnr_gap_db
            ));
        }
        let expected_rows =
            if delayed_restriction_step > 0 && row.horizon < delayed_restriction_step {
                reference
            } else {
                active
            };
        if row.hidden_rows != 0
            || row.adaptive_visible_rows != expected_rows
            || row.adaptive_dynamics_rows != expected_rows
            || row.adaptive_interaction_rows != expected_rows
        {
            failures.push(format!(
                "seed {} horizon {} row contract expected={} visible/dynamics/interaction/hidden={}/{}/{}/{}",
                row.seed,
                row.horizon,
                expected_rows,
                row.adaptive_visible_rows,
                row.adaptive_dynamics_rows,
                row.adaptive_interaction_rows,
                row.hidden_rows,
            ));
        }
        if delayed_restriction_step > 0 {
            let cut_passes = usize::from(row.horizon >= delayed_restriction_step);
            let reallocation_passes = periodic_passes(
                row.horizon,
                reallocation_start_step,
                reallocation_interval_steps,
                reallocation_end_step,
            );
            let expected_passes = cut_passes + reallocation_passes;
            let cut_events = cut_passes * (reference - active) / 3;
            let max_events = cut_events + 2 * max_events_per_interval * reallocation_passes;
            let paired_events = row.topology_events.saturating_sub(cut_events);
            if row.topology_passes != expected_passes
                || row.topology_events < cut_events
                || row.topology_events > max_events
                || !paired_events.is_multiple_of(2)
            {
                failures.push(format!(
                    "seed {} horizon {} adaptive topology passes/events={}/{} expected_passes={} event_range={}..={}",
                    row.seed,
                    row.horizon,
                    row.topology_passes,
                    row.topology_events,
                    expected_passes,
                    cut_events,
                    max_events,
                ));
            }
        } else if config.topology_enabled {
            let expected_passes = periodic_passes(
                row.horizon,
                topology_start_step,
                topology_interval_steps,
                topology_end_step,
            );
            if row.topology_passes != expected_passes
                || row.topology_events > 2 * max_events_per_interval * expected_passes
                || !row.topology_events.is_multiple_of(2)
            {
                failures.push(format!(
                    "seed {} horizon {} paired topology passes/events={}/{} expected_passes={} max_events={}",
                    row.seed,
                    row.horizon,
                    row.topology_passes,
                    row.topology_events,
                    expected_passes,
                    2 * max_events_per_interval * expected_passes,
                ));
            }
        }
        if row.material_relative_error > config.max_material_relative_error {
            failures.push(format!(
                "seed {} horizon {} material error {:.3e} > {:.3e}",
                row.seed,
                row.horizon,
                row.material_relative_error,
                config.max_material_relative_error
            ));
        }
        if row.out_of_bounds_fraction > config.max_out_of_bounds_fraction {
            failures.push(format!(
                "seed {} horizon {} out-of-bounds {:.3} > {:.3}",
                row.seed,
                row.horizon,
                row.out_of_bounds_fraction,
                config.max_out_of_bounds_fraction
            ));
        }
        if row.grid_overflow > config.max_grid_overflow {
            failures.push(format!(
                "seed {} horizon {} grid overflow {} > {}",
                row.seed, row.horizon, row.grid_overflow, config.max_grid_overflow
            ));
        }
        if row.interaction_work_ratio > config.max_interaction_work_ratio {
            failures.push(format!(
                "seed {} horizon {} work ratio {:.3} > {:.3}",
                row.seed,
                row.horizon,
                row.interaction_work_ratio,
                config.max_interaction_work_ratio
            ));
        }
        if row.wall_time_ratio > config.max_wall_time_ratio {
            failures.push(format!(
                "seed {} horizon {} wall ratio {:.3} > {:.3}",
                row.seed, row.horizon, row.wall_time_ratio, config.max_wall_time_ratio
            ));
        }
    }
    if config.quality_horizon_min_steps > 0 {
        let quality_summaries = horizon_summaries
            .iter()
            .filter(|summary| summary.horizon >= config.quality_horizon_min_steps)
            .collect::<Vec<_>>();
        if quality_summaries.is_empty() {
            failures.push(format!(
                "no validation horizon reaches aggregate quality gate start {}",
                config.quality_horizon_min_steps,
            ));
        }
        for summary in quality_summaries {
            if summary.mean_adaptive_psnr_db < config.min_quality_mean_adaptive_psnr_db {
                failures.push(format!(
                    "horizon {} adaptive mean PSNR {:.3} < {:.3}",
                    summary.horizon,
                    summary.mean_adaptive_psnr_db,
                    config.min_quality_mean_adaptive_psnr_db,
                ));
            }
            if summary.worst_adaptive_psnr_db < config.min_quality_worst_adaptive_psnr_db {
                failures.push(format!(
                    "horizon {} adaptive worst PSNR {:.3} < {:.3}",
                    summary.horizon,
                    summary.worst_adaptive_psnr_db,
                    config.min_quality_worst_adaptive_psnr_db,
                ));
            }
            if summary.adaptive_mean_oracle_gap_db < -config.max_quality_mean_oracle_gap_db {
                failures.push(format!(
                    "horizon {} adaptive/oracle mean gap {:.3} < -{:.3}",
                    summary.horizon,
                    summary.adaptive_mean_oracle_gap_db,
                    config.max_quality_mean_oracle_gap_db,
                ));
            }
            if summary.adaptive_worst_oracle_gap_db < -config.max_quality_worst_oracle_gap_db {
                failures.push(format!(
                    "horizon {} adaptive/oracle worst gap {:.3} < -{:.3}",
                    summary.horizon,
                    summary.adaptive_worst_oracle_gap_db,
                    config.max_quality_worst_oracle_gap_db,
                ));
            }
            if summary.adaptive_mean_same_rule_fine_gap_db
                < -config.max_quality_mean_same_rule_fine_gap_db
            {
                failures.push(format!(
                    "horizon {} adaptive/same-rule mean gap {:.3} < -{:.3}",
                    summary.horizon,
                    summary.adaptive_mean_same_rule_fine_gap_db,
                    config.max_quality_mean_same_rule_fine_gap_db,
                ));
            }
            if summary.adaptive_worst_same_rule_fine_gap_db
                < -config.max_quality_worst_same_rule_fine_gap_db
            {
                failures.push(format!(
                    "horizon {} adaptive/same-rule worst gap {:.3} < -{:.3}",
                    summary.horizon,
                    summary.adaptive_worst_same_rule_fine_gap_db,
                    config.max_quality_worst_same_rule_fine_gap_db,
                ));
            }
        }
    }
    if worst_psnr_drift_db > config.max_psnr_drift_db {
        failures.push(format!(
            "worst PSNR drift {worst_psnr_drift_db:.3} > {:.3}",
            config.max_psnr_drift_db
        ));
    }
    if config.require_adaptive_resolution {
        if adaptive_resolution.rows == 0 {
            failures.push(format!(
                "no validation horizon reaches adaptive-resolution gate start {}",
                adaptive_resolution.horizon_min_steps,
            ));
        }
        if adaptive_resolution.minimum_material_scale_ratio < config.min_material_scale_ratio {
            failures.push(format!(
                "adaptive material radius ratio {:.3} < {:.3}",
                adaptive_resolution.minimum_material_scale_ratio, config.min_material_scale_ratio,
            ));
        }
        if adaptive_resolution.minimum_support_bin_count < config.min_adaptive_support_bin_count {
            failures.push(format!(
                "adaptive support-bin count {} < {}",
                adaptive_resolution.minimum_support_bin_count,
                config.min_adaptive_support_bin_count,
            ));
        }
        if adaptive_resolution.minimum_occupied_material_scale_bins
            < config.min_occupied_material_scale_bins
        {
            failures.push(format!(
                "occupied material-scale bins {} < {}",
                adaptive_resolution.minimum_occupied_material_scale_bins,
                config.min_occupied_material_scale_bins,
            ));
        }
        if adaptive_resolution.accepted_local_detail_exchanges
            < config.min_accepted_local_detail_exchanges
        {
            failures.push(format!(
                "accepted local-detail exchanges {} < {}",
                adaptive_resolution.accepted_local_detail_exchanges,
                config.min_accepted_local_detail_exchanges,
            ));
        }
        if adaptive_resolution.mean_scale_detail_correlation_gain_vs_static
            < config.min_mean_scale_detail_correlation_gain_vs_static
        {
            failures.push(format!(
                "mean scale/detail correlation gain {:.3} < {:.3}",
                adaptive_resolution.mean_scale_detail_correlation_gain_vs_static,
                config.min_mean_scale_detail_correlation_gain_vs_static,
            ));
        }
        if adaptive_resolution.worst_scale_detail_correlation_gain_vs_static
            < config.min_worst_scale_detail_correlation_gain_vs_static
        {
            failures.push(format!(
                "worst scale/detail correlation gain {:.3} < {:.3}",
                adaptive_resolution.worst_scale_detail_correlation_gain_vs_static,
                config.min_worst_scale_detail_correlation_gain_vs_static,
            ));
        }
        if adaptive_resolution.mean_static_graded_psnr_gain_db
            < config.min_mean_static_graded_psnr_gain_db
        {
            failures.push(format!(
                "mean dynamic/static graded PSNR gain {:.3} < {:.3}",
                adaptive_resolution.mean_static_graded_psnr_gain_db,
                config.min_mean_static_graded_psnr_gain_db,
            ));
        }
        if adaptive_resolution.worst_static_graded_psnr_gain_db
            < config.min_worst_static_graded_psnr_gain_db
        {
            failures.push(format!(
                "worst dynamic/static graded PSNR gain {:.3} < {:.3}",
                adaptive_resolution.worst_static_graded_psnr_gain_db,
                config.min_worst_static_graded_psnr_gain_db,
            ));
        }
    }
    failures.sort();
    failures.dedup();
    Ok(AdaptiveTarget2dValidationReport {
        objective:
            "uncentered_deployment_psnr_matched_seed_active_material_vs_same_rule_fine_vs_external_oracle"
                .to_owned(),
        active_particle_count: active,
        initial_particle_count: initial,
        delayed_restriction_step,
        reallocation_interval_steps,
        reallocation_start_step,
        reallocation_end_step,
        min_reallocation_relative_gain,
        reference_particle_count: reference,
        visible_gaussian_count: active,
        hidden_fine_rows: 0,
        topology_enabled: config.topology_enabled,
        mean_adaptive_psnr_db,
        worst_adaptive_psnr_db,
        mean_adaptive_oracle_gap_db,
        worst_adaptive_oracle_gap_db,
        mean_static_graded_gap_db,
        worst_static_graded_gap_db,
        mean_same_rule_fine_gap_db,
        worst_same_rule_fine_gap_db,
        worst_psnr_drift_db,
        total_accepted_local_detail_exchanges,
        mean_topology_acceptance_fraction,
        mean_scale_detail_correlation_gain,
        worst_scale_detail_correlation_gain,
        mean_scale_detail_correlation_gain_vs_static,
        worst_scale_detail_correlation_gain_vs_static,
        mean_interaction_work_ratio,
        mean_wall_time_ratio,
        mean_oracle_wall_time_ratio,
        adaptive_resolution,
        horizon_summaries,
        passed: failures.is_empty(),
        failures,
        rows,
    })
}

#[cfg(feature = "gpu_wgpu")]
#[derive(Clone, Copy, Debug)]
struct AdaptationDiagnostics {
    scale_detail_correlation: f32,
    fine_to_coarse_detail_ratio: f32,
    material_scale_ratio: f32,
    occupied_material_scale_bins: usize,
    fractional_material_scale_fraction: f32,
}

#[cfg(feature = "gpu_wgpu")]
fn adaptation_diagnostics(
    model: &super::AdaptiveNpaModel,
    particles: &super::AdaptiveParticleSet,
    fine_measure: f32,
) -> crate::AutomataResult<AdaptationDiagnostics> {
    use std::collections::BTreeSet;

    let perception =
        super::perception::rule_perception_pair(&model.config, &model.rule, particles)?;
    let detail = super::features::local_detail_risk(particles, &perception.normalized);
    let mean_measure =
        particles.total_measure() as f32 / particles.represented_measure.len().max(1) as f32;
    let log_fineness = particles
        .represented_measure
        .iter()
        .map(|measure| -(measure / mean_measure.max(f32::MIN_POSITIVE)).ln())
        .collect::<Vec<_>>();
    let log_detail = detail
        .iter()
        .map(|value| value.max(f32::MIN_POSITIVE).ln())
        .collect::<Vec<_>>();
    let scale_detail_correlation = pearson_correlation(&log_fineness, &log_detail);

    let tolerance = 2.0e-4 * mean_measure;
    let mut fine_detail = 0.0_f64;
    let mut fine_count = 0usize;
    let mut coarse_detail = 0.0_f64;
    let mut coarse_count = 0usize;
    let mut min_footprint = f32::INFINITY;
    let mut max_footprint = 0.0_f32;
    let fine_footprint = super::material_footprint_radius(fine_measure.max(f32::MIN_POSITIVE), 2);
    let mut scale_bins = BTreeSet::new();
    let mut fractional_scales = 0usize;
    for (row, measure) in particles.represented_measure.iter().copied().enumerate() {
        if measure + tolerance < mean_measure {
            fine_detail += f64::from(detail[row]);
            fine_count += 1;
        } else if measure > mean_measure + tolerance {
            coarse_detail += f64::from(detail[row]);
            coarse_count += 1;
        }
        let footprint = super::material_footprint_radius(measure, 2);
        min_footprint = min_footprint.min(footprint);
        max_footprint = max_footprint.max(footprint);
        let octave = (footprint / fine_footprint.max(f32::MIN_POSITIVE)).log2();
        scale_bins.insert((octave * 32.0).round() as i32);
        if (octave - octave.round()).abs() > 1.0e-3 {
            fractional_scales += 1;
        }
    }
    let fine_mean = fine_detail / fine_count.max(1) as f64;
    let coarse_mean = coarse_detail / coarse_count.max(1) as f64;
    Ok(AdaptationDiagnostics {
        scale_detail_correlation,
        fine_to_coarse_detail_ratio: if fine_count == 0 || coarse_count == 0 {
            1.0
        } else {
            (fine_mean / coarse_mean.max(f64::MIN_POSITIVE)) as f32
        },
        material_scale_ratio: max_footprint / min_footprint.max(f32::MIN_POSITIVE),
        occupied_material_scale_bins: scale_bins.len(),
        fractional_material_scale_fraction: fractional_scales as f32
            / particles.len().max(1) as f32,
    })
}

#[cfg(feature = "gpu_wgpu")]
fn pearson_correlation(lhs: &[f32], rhs: &[f32]) -> f32 {
    if lhs.len() != rhs.len() || lhs.is_empty() {
        return 0.0;
    }
    let count = lhs.len() as f64;
    let lhs_mean = lhs.iter().map(|value| f64::from(*value)).sum::<f64>() / count;
    let rhs_mean = rhs.iter().map(|value| f64::from(*value)).sum::<f64>() / count;
    let mut covariance = 0.0_f64;
    let mut lhs_variance = 0.0_f64;
    let mut rhs_variance = 0.0_f64;
    for (lhs, rhs) in lhs.iter().zip(rhs) {
        let lhs = f64::from(*lhs) - lhs_mean;
        let rhs = f64::from(*rhs) - rhs_mean;
        covariance += lhs * rhs;
        lhs_variance += lhs * lhs;
        rhs_variance += rhs * rhs;
    }
    let denominator = (lhs_variance * rhs_variance).sqrt();
    if denominator <= f64::MIN_POSITIVE {
        0.0
    } else {
        (covariance / denominator).clamp(-1.0, 1.0) as f32
    }
}

#[cfg(feature = "gpu_wgpu")]
fn raw_psnr(
    prediction: &crate::target2d::Target2dRenderedSplat,
    target: &crate::target2d::Target2dRenderedSplat,
) -> f32 {
    let mse = prediction
        .rgb
        .iter()
        .zip(&target.rgb)
        .map(|(prediction, target)| (prediction - target).powi(2))
        .sum::<f32>()
        / prediction.rgb.len().max(1) as f32;
    -10.0 * mse.max(1.0e-12).log10()
}

#[cfg(feature = "gpu_wgpu")]
fn composited_psnr(
    prediction: &crate::target2d::Target2dRenderedSplat,
    target: &crate::target2d::Target2dRenderedSplat,
) -> f32 {
    let pixels = prediction.density.len().min(target.density.len());
    let mse = (0..pixels)
        .flat_map(|pixel| {
            let prediction_alpha = prediction.density[pixel].clamp(0.0, 1.0);
            let target_alpha = target.density[pixel].clamp(0.0, 1.0);
            (0..3).map(move |channel| {
                let index = pixel * 3 + channel;
                let prediction = (prediction.rgb[index] + 1.0 - prediction_alpha).clamp(0.0, 1.0);
                let target = (target.rgb[index] + 1.0 - target_alpha).clamp(0.0, 1.0);
                (prediction - target).powi(2)
            })
        })
        .sum::<f32>()
        / (pixels * 3).max(1) as f32;
    -10.0 * mse.max(1.0e-12).log10()
}

#[cfg(feature = "gpu_wgpu")]
fn canonical_oracle_states(
    model: &super::AdaptiveNpaModel,
    expanded_states: &[f32],
) -> crate::AutomataResult<Vec<f32>> {
    let state_dims = model.rule.config.state_dims;
    if !expanded_states.len().is_multiple_of(state_dims) {
        return Err(crate::AutomataError::InvalidArgument(format!(
            "expanded state length {} is not divisible by state_dims {state_dims}",
            expanded_states.len(),
        )));
    }
    let Some(memory) = model.compact_recurrent_memory_range() else {
        return Ok(expanded_states.to_vec());
    };
    let canonical_state_dims = state_dims - memory.len();
    let mut canonical =
        Vec::with_capacity(expanded_states.len() / state_dims * canonical_state_dims);
    for state in expanded_states.chunks_exact(state_dims) {
        canonical.extend_from_slice(&state[..memory.start]);
        canonical.extend_from_slice(&state[memory.end..]);
    }
    Ok(canonical)
}

#[cfg(feature = "gpu_wgpu")]
fn identity_motion(
    prior: Option<&std::collections::BTreeMap<u64, [f32; 2]>>,
    current: &std::collections::BTreeMap<u64, [f32; 2]>,
    steps: usize,
) -> (f32, f32) {
    let Some(prior) = prior else {
        return (1.0, 0.0);
    };
    let mut retained = 0usize;
    let mut displacement = 0.0_f32;
    for (id, position) in current {
        if let Some(previous) = prior.get(id) {
            retained += 1;
            displacement +=
                ((position[0] - previous[0]).powi(2) + (position[1] - previous[1]).powi(2)).sqrt();
        }
    }
    (
        retained as f32 / current.len().max(1) as f32,
        displacement / retained.max(1) as f32 / steps.max(1) as f32,
    )
}

#[cfg(all(test, feature = "gpu_wgpu"))]
mod tests {
    use super::*;

    #[test]
    fn compact_recurrent_memory_is_removed_from_oracle_seed_state_only() {
        let base = crate::NpaModel::upstream_seeded(crate::NpaConfig::growing_2d(), 7);
        let mut model = super::super::AdaptiveNpaModel::seeded(
            base,
            super::super::AdaptiveNpaConfig::growing_2d(),
            11,
        )
        .unwrap();
        model.enable_compact_recurrent_memory(8).unwrap();
        let expanded = (0..2 * 24).map(|value| value as f32).collect::<Vec<_>>();
        let canonical = canonical_oracle_states(&model, &expanded).unwrap();
        assert_eq!(canonical.len(), 2 * 16);
        assert_eq!(&canonical[..13], &expanded[..13]);
        assert_eq!(&canonical[13..16], &expanded[21..24]);
        assert_eq!(&canonical[16..29], &expanded[24..37]);
        assert_eq!(&canonical[29..32], &expanded[45..48]);
    }

    fn row(horizon: usize, psnr: f32, topology_passes: usize) -> AdaptiveTarget2dValidationRow {
        AdaptiveTarget2dValidationRow {
            seed: 42,
            horizon,
            adaptive_psnr_db: psnr,
            adaptive_composited_psnr_db: psnr,
            static_graded_psnr_db: psnr,
            same_rule_fine_psnr_db: psnr,
            oracle_psnr_db: psnr,
            adaptive_oracle_gap_db: 0.0,
            adaptive_static_graded_gap_db: 0.0,
            adaptive_same_rule_fine_gap_db: 0.0,
            adaptive_visible_rows: 3_070,
            adaptive_dynamics_rows: 3_070,
            adaptive_interaction_rows: 3_070,
            hidden_rows: 0,
            adaptive_neighbor_mode: "CooperativeSortedCells".to_owned(),
            adaptive_support_bin_count: 2,
            adaptive_requested_support_bin_count: 2,
            oracle_neighbor_mode: "CooperativeSortedCells".to_owned(),
            topology_passes,
            topology_events: 2 * topology_passes,
            accepted_local_detail_exchanges: topology_passes,
            topology_acceptance_fraction: if topology_passes > 0 { 1.0 } else { 0.0 },
            initial_scale_detail_correlation: 0.0,
            scale_detail_correlation: 0.1,
            scale_detail_correlation_gain: 0.1,
            static_scale_detail_correlation: 0.0,
            scale_detail_correlation_gain_vs_static: 0.1,
            fine_to_coarse_detail_ratio: 1.1,
            static_fine_to_coarse_detail_ratio: 1.0,
            material_scale_ratio: 1.2,
            occupied_material_scale_bins: 8,
            fractional_material_scale_fraction: 1.0,
            material_relative_error: 0.0,
            occupied_pixel_fraction: 0.2,
            out_of_bounds_fraction: 0.0,
            retained_identity_fraction: 1.0,
            retained_identity_motion_per_step: 0.0,
            grid_overflow: 0,
            adaptive_particle_steps: 3_070 * horizon,
            oracle_particle_steps: 4_096 * horizon,
            interaction_work_ratio: 0.76,
            theoretical_pair_work_ratio: 0.57,
            adaptive_elapsed_ms: 1.0,
            same_rule_elapsed_ms: 1.0,
            oracle_elapsed_ms: 1.0,
            wall_time_ratio: 1.0,
            oracle_wall_time_ratio: 1.0,
        }
    }

    fn permissive_config() -> AdaptiveTarget2dValidationConfig {
        AdaptiveTarget2dValidationConfig {
            min_adaptive_psnr_db: 0.0,
            max_oracle_psnr_gap_db: 100.0,
            max_same_rule_fine_psnr_gap_db: 100.0,
            max_psnr_drift_db: 0.1,
            max_material_relative_error: 1.0,
            max_out_of_bounds_fraction: 1.0,
            max_grid_overflow: u32::MAX,
            max_interaction_work_ratio: 1.0,
            max_wall_time_ratio: 2.0,
            require_adaptive_resolution: false,
            ..AdaptiveTarget2dValidationConfig::default()
        }
    }

    #[test]
    fn default_validation_rejects_narrow_single_support_adaptivity() {
        let config = AdaptiveTarget2dValidationConfig::default();
        assert!(config.require_adaptive_resolution);
        assert!(config.min_material_scale_ratio >= 2.0);
        assert!(config.min_adaptive_support_bin_count >= 2);
        assert!(config.min_accepted_local_detail_exchanges > 0);
        assert!(config.min_mean_scale_detail_correlation_gain_vs_static > 0.0);
    }

    fn summary_context() -> AdaptiveValidationSummaryContext {
        AdaptiveValidationSummaryContext {
            active: 3_070,
            reference: 4_096,
            initial: 3_070,
            delayed_restriction_step: 0,
            reallocation_interval_steps: 0,
            reallocation_start_step: 0,
            reallocation_end_step: 0,
            topology_start_step: 32,
            topology_interval_steps: 32,
            topology_end_step: 0,
            min_reallocation_relative_gain: 0.0,
            max_events_per_interval: 1,
        }
    }

    #[test]
    fn validation_drift_counts_regression_not_improvement() {
        let improving = summarize_validation(
            vec![row(96, 26.0, 3), row(256, 27.0, 8)],
            summary_context(),
            &permissive_config(),
        )
        .unwrap();
        assert_eq!(improving.worst_psnr_drift_db, 0.0);
        assert!(improving.passed, "{:?}", improving.failures);

        let regressing = summarize_validation(
            vec![row(96, 27.0, 3), row(256, 26.0, 8)],
            summary_context(),
            &permissive_config(),
        )
        .unwrap();
        assert_eq!(regressing.worst_psnr_drift_db, 1.0);
        assert!(!regressing.passed);
    }

    #[test]
    fn validation_rejects_missing_or_non_finite_topology_evidence() {
        let mut invalid = row(96, f32::NAN, 0);
        invalid.topology_events = 0;
        let report =
            summarize_validation(vec![invalid], summary_context(), &permissive_config()).unwrap();
        assert!(!report.passed);
        assert!(
            report
                .failures
                .iter()
                .any(|failure| failure.contains("non-finite"))
        );
        assert!(
            report
                .failures
                .iter()
                .any(|failure| failure.contains("topology passes/events"))
        );
    }

    #[test]
    fn delayed_restriction_accepts_fine_then_active_row_contract() {
        let mut before = row(96, 26.0, 0);
        before.adaptive_visible_rows = 4_096;
        before.adaptive_dynamics_rows = 4_096;
        before.adaptive_interaction_rows = 4_096;
        before.adaptive_particle_steps = 4_096 * 96;
        before.interaction_work_ratio = 1.0;
        before.theoretical_pair_work_ratio = 1.0;

        let mut after = row(256, 26.0, 1);
        after.topology_events = (4_096 - 3_070) / 3;
        after.adaptive_particle_steps = 4_096 * 128 + 3_070 * 128;
        after.interaction_work_ratio = after.adaptive_particle_steps as f32 / (4_096 * 256) as f32;
        after.theoretical_pair_work_ratio = theoretical_pair_work_ratio(3_070, 4_096, 256, 128, 1);

        let mut config = permissive_config();
        config.delayed_restriction_step = 128;
        let context = AdaptiveValidationSummaryContext {
            initial: 4_096,
            delayed_restriction_step: 128,
            topology_start_step: 257,
            ..summary_context()
        };
        let report = summarize_validation(vec![before, after], context, &config).unwrap();
        assert!(report.passed, "{:?}", report.failures);
        assert_eq!(report.initial_particle_count, 4_096);
        assert_eq!(report.delayed_restriction_step, 128);
    }

    #[test]
    fn delayed_restriction_accounts_for_fused_paired_reallocation() {
        let mut row = row(256, 26.0, 5);
        row.topology_events = (4_096 - 3_070) / 3 + 2 * 4;
        let mut config = permissive_config();
        config.delayed_restriction_step = 128;
        config.reallocation_interval_steps = 32;
        config.reallocation_start_step = 160;
        let context = AdaptiveValidationSummaryContext {
            initial: 4_096,
            delayed_restriction_step: 128,
            reallocation_interval_steps: 32,
            reallocation_start_step: 160,
            reallocation_end_step: 256,
            topology_start_step: 160,
            ..summary_context()
        };
        let report = summarize_validation(vec![row], context, &config).unwrap();
        assert!(report.passed, "{:?}", report.failures);
        assert_eq!(periodic_passes(256, 160, 32, 256), 4);
        assert_eq!(periodic_passes(4_096, 160, 32, 256), 4);
    }

    #[test]
    fn direct_topology_gate_respects_cadence_alignment_after_start() {
        let before_first_aligned_pass = row(96, 26.0, 0);
        let mut after_first_aligned_pass = row(256, 26.0, 1);
        after_first_aligned_pass.topology_events = 8;
        after_first_aligned_pass.accepted_local_detail_exchanges = 4;
        after_first_aligned_pass.topology_acceptance_fraction = 1.0;

        let context = AdaptiveValidationSummaryContext {
            topology_start_step: 64,
            topology_interval_steps: 256,
            max_events_per_interval: 4,
            ..summary_context()
        };
        let report = summarize_validation(
            vec![before_first_aligned_pass, after_first_aligned_pass],
            context,
            &permissive_config(),
        )
        .unwrap();
        assert!(report.passed, "{:?}", report.failures);
        assert_eq!(periodic_passes(96, 64, 256, 0), 0);
        assert_eq!(periodic_passes(256, 64, 256, 0), 1);
    }

    #[test]
    fn aggregate_quality_gates_compare_horizon_mean_and_worst() {
        let mut strong = row(512, 28.0, 0);
        strong.oracle_psnr_db = 26.5;
        strong.same_rule_fine_psnr_db = 27.0;
        strong.adaptive_oracle_gap_db = 1.5;
        strong.adaptive_same_rule_fine_gap_db = 1.0;
        let mut tail = row(512, 24.5, 0);
        tail.seed = 43;
        tail.oracle_psnr_db = 24.0;
        tail.same_rule_fine_psnr_db = 24.2;
        tail.adaptive_oracle_gap_db = 0.5;
        tail.adaptive_same_rule_fine_gap_db = 0.3;
        let mut config = permissive_config();
        config.quality_horizon_min_steps = 512;
        config.min_quality_mean_adaptive_psnr_db = 26.0;
        config.min_quality_worst_adaptive_psnr_db = 24.0;
        config.max_quality_mean_oracle_gap_db = 0.5;
        config.max_quality_worst_oracle_gap_db = 0.5;
        config.max_quality_mean_same_rule_fine_gap_db = 0.5;
        config.max_quality_worst_same_rule_fine_gap_db = 0.5;

        let context = AdaptiveValidationSummaryContext {
            topology_start_step: 513,
            ..summary_context()
        };
        let passing =
            summarize_validation(vec![strong.clone(), tail.clone()], context, &config).unwrap();
        assert!(passing.passed, "{:?}", passing.failures);
        assert_eq!(passing.horizon_summaries.len(), 1);
        assert_eq!(passing.horizon_summaries[0].horizon, 512);
        assert!((passing.horizon_summaries[0].mean_adaptive_psnr_db - 26.25).abs() < 1.0e-6);

        tail.adaptive_psnr_db = 23.0;
        tail.adaptive_composited_psnr_db = 23.0;
        tail.adaptive_oracle_gap_db = -1.0;
        tail.adaptive_same_rule_fine_gap_db = -1.2;
        let failing = summarize_validation(vec![strong, tail], context, &config).unwrap();
        assert!(!failing.passed);
        assert!(
            failing
                .failures
                .iter()
                .any(|failure| failure.contains("adaptive worst PSNR"))
        );
    }
}
