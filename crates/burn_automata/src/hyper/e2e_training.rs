#[cfg(any(test, feature = "backend_cuda", feature = "backend_wgpu"))]
mod checkpoint;
mod config;
pub(crate) mod dense;
mod direct_basis;
mod report;
#[cfg(any(test, feature = "backend_cuda", feature = "backend_wgpu"))]
mod sampling;

#[cfg(any(test, feature = "backend_cuda", feature = "backend_wgpu"))]
pub(crate) use checkpoint::*;
pub(crate) use config::*;
pub(crate) use direct_basis::*;
pub(crate) use report::*;
#[cfg(any(test, feature = "backend_cuda", feature = "backend_wgpu"))]
pub(crate) use sampling::*;

#[derive(Clone, Copy)]
struct AdaptiveTarget2dRuleMode {
    frozen_base_residual: bool,
    normalized_adaptive_residual: bool,
    residual_material_features: bool,
    normalized_adaptive_rule: bool,
    material_scale_conditioning: bool,
    topology: crate::adaptive::AdaptiveTarget2dTopologyConfig,
}

struct AdaptiveTarget2dObserverBridge<'a> {
    template: crate::adaptive::AdaptiveNpaModel,
    frozen_base_residual: bool,
    observer: &'a mut dyn crate::adaptive::AdaptiveTarget2dGpuTrainingObserver,
}

impl crate::Target2dGpuTrainingObserver for AdaptiveTarget2dObserverBridge<'_> {
    fn should_stop(&self) -> bool {
        self.observer.should_stop()
    }

    fn snapshot_interval_steps(&self) -> usize {
        self.observer.snapshot_interval_steps()
    }

    fn snapshot_interval_duration(&self) -> std::time::Duration {
        self.observer.snapshot_interval_duration()
    }

    fn on_progress(&mut self, progress: crate::Target2dGpuTrainingProgress) {
        let mut model = self.template.clone();
        if self.frozen_base_residual {
            model.local_residual_rule = Some(progress.model);
        } else {
            model.rule = progress.model;
        }
        self.observer
            .on_progress(crate::adaptive::AdaptiveTarget2dGpuTrainingProgress {
                step: progress.step,
                total_steps: progress.total_steps,
                loss: progress.loss,
                eval_loss: progress.eval_loss,
                render_rgb_psnr_db: progress.render_rgb_psnr_db,
                base_grad_norm: progress.base_grad_norm,
                base_grad_scale: progress.base_grad_scale,
                particle_steps_per_sec: progress.particle_steps_per_sec,
                elapsed_ms: progress.elapsed_ms,
                model,
            });
    }
}

fn configure_adaptive_target2d_model(
    model: &mut crate::adaptive::AdaptiveNpaModel,
    config: &crate::adaptive::AdaptiveTarget2dTrainingConfig,
) -> Result<AdaptiveTarget2dRuleMode, Box<dyn std::error::Error>> {
    use crate::AutomataError;
    use crate::adaptive::{
        AdaptiveCoarseDynamics, AdaptiveLocalRuleSemantics, AdaptiveMaterialSeedLayout,
        AdaptiveResidualGateReference, AdaptiveRestrictionArity, AdaptiveRulePerception,
        AdaptiveTarget2dRuleTraining, AdaptiveTopologyControl,
    };

    let training = &config.target2d;
    let reference_particle_count = config.material.reference_particle_count;
    let expected_bandwidth_exponent = 1.0 / model.config.spatial_dims as f32;
    if config.material.seed_layout == AdaptiveMaterialSeedLayout::CanonicalGrouped
        && (config.material.bandwidth_exponent - expected_bandwidth_exponent).abs() > 1.0e-6
    {
        return Err(AutomataError::InvalidArgument(format!(
            "adaptive Target2D canonical material requires bandwidth_exponent={expected_bandwidth_exponent}, got {}",
            config.material.bandwidth_exponent,
        ))
        .into());
    }
    if config.material.seed_layout == AdaptiveMaterialSeedLayout::UniformContinuous
        && config.topology.enabled
    {
        return Err(AutomataError::InvalidArgument(
            "uniform-continuous adaptive Target2D material has no distinct scale slots to reallocate"
                .to_owned(),
        )
        .into());
    }
    model.config.material_seed_layout = config.material.seed_layout;
    model.config.material_seed_bandwidth_exponent = config.material.bandwidth_exponent;
    model.config.material_seed_measure_ratio = config.material.seed_measure_ratio;
    let fine_measure = config.material.total_measure / reference_particle_count as f32;
    let fine_footprint = crate::adaptive::material_footprint_radius(fine_measure, 2);
    model.config.perception.reference_measure = fine_measure;
    model.config.reference_footprint = fine_footprint;
    model.config.base_rule_footprint = fine_footprint;
    model.config.min_footprint = model.config.min_footprint.min(fine_footprint);
    model.config.max_footprint = model.config.max_footprint.max(2.0 * fine_footprint);
    model.config.bootstrap_target_leaves = 0;
    model.config.initial_leaves = 0;
    model.config.bootstrap_end_step = 0;
    model.config.hierarchical_bootstrap_seed = true;
    model.config.bootstrap_fine_leaves = reference_particle_count;
    model.config.hierarchical_restriction_step = 0;
    model
        .config
        .hierarchical_restriction_leaf_delta_per_interval = 0;
    model.config.hierarchical_restriction_arity = AdaptiveRestrictionArity::Canonical;
    model.config.hierarchical_restriction_policy = config.restriction_policy;
    model.config.retain_bootstrap_templates = false;
    model.config.coarse_dynamics = AdaptiveCoarseDynamics::RepresentedMeasure;
    model.config.coarse_quadrature_points = 0;
    model.config.bootstrap_quadrature_points = 0;

    let automatic_tbptt = if training.tbptt_chunk_steps == 0 {
        training.step_max.clamp(1, 32)
    } else {
        training.tbptt_chunk_steps.clamp(1, training.step_max)
    };
    let mut topology = config.topology;
    if topology.enabled {
        let topology_interval = if topology.interval_steps == 0 {
            automatic_tbptt
        } else {
            topology.interval_steps
        };
        let continuous_topology =
            config.material.seed_layout == AdaptiveMaterialSeedLayout::GradedContinuous;
        if topology.events_per_interval == 0
            || (!continuous_topology && topology.events_per_interval != 1)
            || (continuous_topology
                && topology.events_per_interval
                    > crate::adaptive::CONTINUOUS_LOCAL_DETAIL_MAX_EXCHANGES)
        {
            return Err(AutomataError::InvalidArgument(format!(
                "adaptive Target2D requires one canonical pair or 1..={} graded-continuous exchanges per interval",
                crate::adaptive::CONTINUOUS_LOCAL_DETAIL_MAX_EXCHANGES,
            ))
            .into());
        }
        if !topology.min_relative_gain.is_finite()
            || !(0.0..=1.0).contains(&topology.min_relative_gain)
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive Target2D topology min_relative_gain must be finite and in [0, 1]"
                    .to_owned(),
            )
            .into());
        }
        if topology.end_step > 0 && topology.end_step < topology.start_step {
            return Err(AutomataError::InvalidArgument(
                "adaptive Target2D topology end_step must not precede start_step".to_owned(),
            )
            .into());
        }
        if !topology_interval.is_multiple_of(automatic_tbptt) {
            return Err(AutomataError::InvalidArgument(format!(
                "adaptive Target2D topology interval {topology_interval} must be a multiple of TBPTT chunk depth {automatic_tbptt}",
            ))
            .into());
        }
        topology.interval_steps = topology_interval;
        model.config.runtime_topology_control = match config.material.seed_layout {
            AdaptiveMaterialSeedLayout::CanonicalGrouped => {
                AdaptiveTopologyControl::PairedLocalDetail
            }
            AdaptiveMaterialSeedLayout::GradedContinuous => {
                AdaptiveTopologyControl::ContinuousLocalDetail
            }
            AdaptiveMaterialSeedLayout::UniformContinuous => unreachable!(
                "uniform-continuous topology was rejected before trainer configuration"
            ),
        };
        model.config.topology_start_step = topology.start_step;
        model.config.steady_topology_start_step = topology.start_step;
        model.config.topology_end_step = topology.end_step;
        model.config.topology_interval = topology_interval;
        model.config.steady_topology_interval = topology_interval;
        model.config.max_events_per_interval = topology.events_per_interval;
        model.config.paired_topology_split_radius_scale = topology.split_radius_scale;
        model.config.paired_topology_merge_detail_scale = topology.merge_detail_scale;
        model.config.min_reallocation_relative_gain = topology.min_relative_gain;
    }

    let mode = AdaptiveTarget2dRuleMode {
        frozen_base_residual: matches!(
            config.rule_training,
            AdaptiveTarget2dRuleTraining::FrozenBaseCompatibleResidual
                | AdaptiveTarget2dRuleTraining::FrozenBaseMaterialConditionedResidual
                | AdaptiveTarget2dRuleTraining::FrozenBaseNormalizedAdaptiveResidual
        ),
        normalized_adaptive_residual: config.rule_training
            == AdaptiveTarget2dRuleTraining::FrozenBaseNormalizedAdaptiveResidual,
        residual_material_features: matches!(
            config.rule_training,
            AdaptiveTarget2dRuleTraining::FrozenBaseMaterialConditionedResidual
                | AdaptiveTarget2dRuleTraining::FrozenBaseNormalizedAdaptiveResidual
        ),
        normalized_adaptive_rule: config.rule_training
            == AdaptiveTarget2dRuleTraining::NormalizedAdaptiveRule,
        material_scale_conditioning: matches!(
            config.rule_training,
            AdaptiveTarget2dRuleTraining::SharedScaleConditionedRule
                | AdaptiveTarget2dRuleTraining::NormalizedAdaptiveRule
        ),
        topology,
    };
    if mode.normalized_adaptive_rule {
        model.config.rule_perception = AdaptiveRulePerception::NormalizedAdaptive;
    }
    model.config.expected_coarse_update_mask = config.expected_coarse_update_mask
        && config.material.seed_layout == AdaptiveMaterialSeedLayout::CanonicalGrouped;
    if mode.material_scale_conditioning {
        model.enable_material_scale_conditioning()?;
    } else if model.config.material_scale_conditioning {
        return Err(AutomataError::InvalidArgument(
            "a material-scale-conditioned adaptive model must use rule_training=shared-scale-conditioned-rule or normalized-adaptive-rule"
                .to_owned(),
        )
        .into());
    }
    let compact_recurrent_memory_dims = config.compact_recurrent_memory_dims;
    if compact_recurrent_memory_dims > 0 {
        if config.rule_training
            != AdaptiveTarget2dRuleTraining::FrozenBaseNormalizedAdaptiveResidual
            || config.material.seed_layout != AdaptiveMaterialSeedLayout::CanonicalGrouped
            || config.optimize_material_scale_only
        {
            return Err(AutomataError::InvalidArgument(
                "compact recurrent memory currently requires canonical-grouped frozen-base-normalized-adaptive-residual training"
                    .to_owned(),
            )
            .into());
        }
        model.enable_compact_recurrent_memory(compact_recurrent_memory_dims)?;
    } else if model.config.compact_recurrent_memory_dims != 0 {
        return Err(AutomataError::InvalidArgument(format!(
            "training config requests zero compact recurrent memory channels but the source artifact contains {}",
            model.config.compact_recurrent_memory_dims,
        ))
        .into());
    }
    if config.optimize_material_scale_only
        && (config.rule_training != AdaptiveTarget2dRuleTraining::SharedScaleConditionedRule
            || training.optimizer.weight_decay != 0.0)
    {
        return Err(AutomataError::InvalidArgument(
            "optimize_material_scale_only requires shared-scale-conditioned-rule and zero optimizer weight decay"
                .to_owned(),
        )
        .into());
    }
    if mode.frozen_base_residual {
        model.config.local_rule_semantics = if mode.normalized_adaptive_residual {
            AdaptiveLocalRuleSemantics::NormalizedExposureResidual
        } else {
            AdaptiveLocalRuleSemantics::CompatibleResidual
        };
        model.config.residual_gate_reference = AdaptiveResidualGateReference::BaseRule;
        model.config.local_residual_scale = 1.0;
        model.config.local_residual_motion_scale = 1.0;
        model.config.local_residual_state_scale = 1.0;
        model.config.closure_moment_features = false;
        model.config.closure_recurrent_mode = false;
        if mode.normalized_adaptive_residual {
            if model.local_residual_rule.is_none() {
                model.enable_material_conditioned_normalized_residual_rule()?;
            }
        } else if mode.residual_material_features {
            if model.local_residual_rule.is_none() {
                model.enable_material_conditioned_compatible_residual_rule()?;
            }
        } else if model.local_residual_rule.is_none() {
            model.enable_zero_local_residual_rule()?;
        }
    }
    model.validate()?;
    Ok(mode)
}

fn adaptive_target2d_scale_limits_report(
    model: &crate::adaptive::AdaptiveNpaModel,
    config: &crate::adaptive::AdaptiveTarget2dTrainingConfig,
    material: &crate::adaptive::AdaptiveTarget2dMaterialLayout,
) -> serde_json::Value {
    let finite_range = |values: &[f32]| {
        values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), value| {
                (min.min(value), max.max(value))
            })
    };
    let ratio =
        |min: f32, max: f32| (min.is_finite() && max.is_finite() && min > 0.0).then_some(max / min);
    let (min_units, max_units) = finite_range(&material.represented_fine_units);
    let (min_footprint_ratio, max_footprint_ratio) = finite_range(&material.footprint_ratio);
    let (min_bandwidth, max_bandwidth) = finite_range(&material.bandwidth);
    let fine_footprint = crate::adaptive::material_footprint_radius(material.fine_measure, 2);
    let material_footprints = material
        .footprint_ratio
        .iter()
        .map(|value| fine_footprint * value)
        .collect::<Vec<_>>();
    let render_footprints = material_footprints
        .iter()
        .map(|value| model.config.render_footprint(*value))
        .collect::<Vec<_>>();
    let (min_material_footprint, max_material_footprint) = finite_range(&material_footprints);
    let (min_render_footprint, max_render_footprint) = finite_range(&render_footprints);
    let raw_scale_features = material
        .footprint_ratio
        .iter()
        .map(|ratio| ratio - 1.0)
        .collect::<Vec<_>>();
    let (min_raw_scale_feature, max_raw_scale_feature) = finite_range(&raw_scale_features);
    let support_bin_count = burn_automata_kernels::AdaptiveSupportBins::new(
        min_bandwidth,
        max_bandwidth,
        model.config.perception.support_bin_ratio,
    )
    .ok()
    .map(|bins| bins.len());
    let fraction = |count: usize| count as f32 / material.active_particle_count().max(1) as f32;
    let tolerance = 64.0 * f32::EPSILON;
    let residual_gate = match config.rule_training {
        crate::adaptive::AdaptiveTarget2dRuleTraining::FrozenBaseNormalizedAdaptiveResidual => {
            serde_json::json!({
                "active": true,
                "encoding": "clamp(log2(footprint/reference), 0, 3)",
                "representable_footprint_ratio_min": 1.0,
                "representable_footprint_ratio_max": 8.0,
                "purpose": "coarse-exposure gate; material scale remains a separate feature",
            })
        }
        crate::adaptive::AdaptiveTarget2dRuleTraining::FrozenBaseCompatibleResidual
        | crate::adaptive::AdaptiveTarget2dRuleTraining::FrozenBaseMaterialConditionedResidual => {
            serde_json::json!({
                "active": true,
                "encoding": "batch-global binary coarse-exposure gate",
                "representable_footprint_ratio_min": null,
                "representable_footprint_ratio_max": null,
                "purpose": "residual enablement, not continuous scale encoding",
            })
        }
        _ => serde_json::json!({
            "active": false,
            "encoding": null,
            "representable_footprint_ratio_min": null,
            "representable_footprint_ratio_max": null,
            "purpose": "the selected shared-rule path does not consume a residual gate",
        }),
    };
    serde_json::json!({
        "seed_layout": format!("{:?}", material.seed_layout),
        "requested_seed_measure_ratio": config.material.seed_measure_ratio,
        "observed_represented_fine_units_min": min_units,
        "observed_represented_fine_units_max": max_units,
        "observed_represented_measure_ratio": ratio(min_units, max_units),
        "observed_material_footprint_ratio_min": min_footprint_ratio,
        "observed_material_footprint_ratio_max": max_footprint_ratio,
        "observed_material_footprint_span": ratio(min_footprint_ratio, max_footprint_ratio),
        "observed_material_footprint_min": min_material_footprint,
        "observed_material_footprint_max": max_material_footprint,
        "configured_material_footprint_min": model.config.min_footprint,
        "configured_material_footprint_max": model.config.max_footprint,
        "material_rows_at_configured_min_fraction": fraction(material_footprints.iter()
            .filter(|value| (**value - model.config.min_footprint).abs()
                <= tolerance * model.config.min_footprint.max(1.0))
            .count()),
        "material_rows_at_configured_max_fraction": fraction(material_footprints.iter()
            .filter(|value| (**value - model.config.max_footprint).abs()
                <= tolerance * model.config.max_footprint.max(1.0))
            .count()),
        "observed_render_footprint_min": min_render_footprint,
        "observed_render_footprint_max": max_render_footprint,
        "observed_render_footprint_span": ratio(min_render_footprint, max_render_footprint),
        "configured_render_footprint_min": model.config.min_render_footprint(),
        "configured_render_footprint_max": model.config.max_render_footprint(),
        "render_footprint_exponent": model.config.render_footprint_exponent,
        "bandwidth_exponent": config.material.bandwidth_exponent,
        "observed_interaction_bandwidth_min": min_bandwidth,
        "observed_interaction_bandwidth_max": max_bandwidth,
        "observed_interaction_bandwidth_span": ratio(min_bandwidth, max_bandwidth),
        "configured_interaction_bandwidth_min": model.config.perception.min_bandwidth,
        "configured_interaction_bandwidth_max": model.config.perception.max_bandwidth,
        "support_bin_ratio": model.config.perception.support_bin_ratio,
        "support_bin_count": support_bin_count,
        "material_scale_feature": {
            "encoding": "clamp(footprint/reference - 1, -0.75, 3.0)",
            "raw_min": min_raw_scale_feature,
            "raw_max": max_raw_scale_feature,
            "lower_saturation_fraction": fraction(raw_scale_features.iter()
                .filter(|value| **value <= -0.75).count()),
            "upper_saturation_fraction": fraction(raw_scale_features.iter()
                .filter(|value| **value >= 3.0).count()),
            "representable_footprint_ratio_min": 0.25,
            "representable_footprint_ratio_max": 4.0,
        },
        "residual_gate": residual_gate,
        "topology": {
            "fixed_material_scale_slots": material.seed_layout
                == crate::adaptive::AdaptiveMaterialSeedLayout::GradedContinuous,
            "one_pass_desired_current_ratio_min":
                model.config.min_topology_footprint_ratio,
            "one_pass_desired_current_ratio_max":
                model.config.max_topology_footprint_ratio,
            "max_neighbor_footprint_ratio": model.config.max_neighbor_footprint_ratio,
        },
    })
}

pub(crate) fn prepare_adaptive_target2d_model(
    model: &mut crate::adaptive::AdaptiveNpaModel,
    config: &crate::adaptive::AdaptiveTarget2dTrainingConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    configure_adaptive_target2d_model(model, config).map(|_| ())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn train_adaptive_target_2d_gpu_impl(
    backend: crate::Target2dGpuBackend,
    model: &mut crate::adaptive::AdaptiveNpaModel,
    hashgrid: &burn_automata_kernels::HashGridConfig,
    target: crate::TargetImage2d,
    config: crate::adaptive::AdaptiveTarget2dTrainingConfig,
    loss_config: crate::Target2dLossConfig,
    checkpoint_config: Option<&crate::Target2dGpuCheckpointConfig>,
    observer: Option<&mut dyn crate::adaptive::AdaptiveTarget2dGpuTrainingObserver>,
) -> Result<crate::adaptive::AdaptiveTarget2dGpuTrainingReport, Box<dyn std::error::Error>> {
    use crate::adaptive::{
        AdaptiveLocalRuleSemantics, AdaptiveRulePerception, AdaptiveTarget2dGpuTrainingReport,
        build_adaptive_target2d_seed_bank,
    };
    use crate::hyper::e2e::{PerceptionRolloutBackend, Target2dLossBackend};
    use crate::{
        AdamWConfig, AutomataError, NpaLowRankAdapter, SgdConfig, Target2dGpuTrainingHistoryEntry,
        Target2dGpuTrainingReport,
    };
    use burn_automata_kernels::{
        AdaptiveGraphPolicy, AdaptiveNpaPerceptionOptions, AdaptivePerceptionConfig,
    };

    let training = &config.target2d;
    let mut checkpoint_seeds = if config.checkpoint_seeds.is_empty() {
        vec![training.seed]
    } else {
        config.checkpoint_seeds.clone()
    };
    checkpoint_seeds.sort_unstable();
    checkpoint_seeds.dedup();
    let mut checkpoint_horizons = if config.checkpoint_horizons.is_empty() {
        vec![training.step_max]
    } else {
        config.checkpoint_horizons.clone()
    };
    checkpoint_horizons.sort_unstable();
    checkpoint_horizons.dedup();
    let reference_particle_count = config.material.reference_particle_count;
    let fine_measure = config.material.total_measure / reference_particle_count as f32;
    let rule_mode = configure_adaptive_target2d_model(model, &config)?;
    let frozen_base_residual = rule_mode.frozen_base_residual;
    let normalized_adaptive_residual = rule_mode.normalized_adaptive_residual;
    let residual_material_features = rule_mode.residual_material_features;
    let normalized_adaptive_rule = rule_mode.normalized_adaptive_rule;
    let material_scale_conditioning = rule_mode.material_scale_conditioning;
    let topology_config = rule_mode.topology;
    let first_topology_event_step = topology_config
        .start_step
        .max(1)
        .div_ceil(topology_config.interval_steps.max(1))
        * topology_config.interval_steps.max(1);
    let training_rule_is_valid = if frozen_base_residual {
        model.local_residual_rule.is_some()
            && model.config.local_rule_semantics
                == if normalized_adaptive_residual {
                    AdaptiveLocalRuleSemantics::NormalizedExposureResidual
                } else {
                    AdaptiveLocalRuleSemantics::CompatibleResidual
                }
            && !model.config.closure_moment_features
            && !model.config.closure_recurrent_mode
            && model.config.compatible_residual_material_features == residual_material_features
    } else {
        model.local_residual_rule.is_none()
            && model.config.material_scale_conditioning == material_scale_conditioning
            && !model.config.compatible_residual_material_features
            && model.config.rule_perception
                == if normalized_adaptive_rule {
                    AdaptiveRulePerception::NormalizedAdaptive
                } else {
                    AdaptiveRulePerception::NpaCompatible
                }
    };
    if model.rule.config.spatial_dims != 2
        || hashgrid.dim != 2
        || model.config.spatial_dims != 2
        || !model.rule.config.stopgrad_pos
        || model.rule.config.stopgrad_state
        || model.config.rule_graph_policy != AdaptiveGraphPolicy::RawSupport
        || !training_rule_is_valid
        || model.proxy_rule.is_some()
        || model.deployment_rule.is_some()
        || model.deployment_local_rule.is_some()
        || model.closure_mode_rule.is_some()
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive Target2D requires canonical raw-support represented-measure perception and either one shared rule or one explicit frozen-base compatible residual, with no hidden fine-state or other auxiliary rule pathways"
                .to_string(),
        )
        .into());
    }
    if training.epochs == 0
        || training.repetitions == 0
        || training.batch_size == 0
        || training.particle_count == 0
        || training.particle_count > 4_096
        || training.step_min == 0
        || training.step_max < training.step_min
        || training.pool_size < training.batch_size
        || config.fresh_seed_trajectories == 0
        || config.fresh_seed_trajectories > training.batch_size
        || checkpoint_seeds.is_empty()
        || checkpoint_horizons.is_empty()
        || checkpoint_horizons
            .iter()
            .any(|horizon| *horizon == 0 || *horizon > 4_096)
        || (config.max_pool_age_steps > 0 && config.max_pool_age_steps < training.step_max)
        || config.pool_age_strata == 1
        || config.pool_age_strata > training.batch_size
        || (config.pool_age_strata > 1 && config.max_pool_age_steps == 0)
        || !config.backward_loss_scale.is_finite()
        || config.backward_loss_scale <= 0.0
        || config.backward_loss_scale > 1.0
        || (config.backward_loss_scale != 1.0 && !training.per_parameter_grad_normalization)
        || !config.trajectory_tail_fraction.is_finite()
        || !(0.0..=1.0).contains(&config.trajectory_tail_fraction)
        || !config.trajectory_tail_weight.is_finite()
        || config.trajectory_tail_weight < 0.0
        || !config.event_training.post_event_loss_weight.is_finite()
        || config.event_training.post_event_loss_weight < 0.0
        || !config
            .event_training
            .post_event_degradation_weight
            .is_finite()
        || config.event_training.post_event_degradation_weight < 0.0
        || !config
            .event_training
            .checkpoint_drift_penalty_weight
            .is_finite()
        || config.event_training.checkpoint_drift_penalty_weight < 0.0
        || (config.event_training.enabled
            && (!topology_config.enabled
                || config.event_training.post_event_recovery_steps == 0
                || config.event_training.min_event_trajectories_per_batch == 0
                || config.event_training.min_event_trajectories_per_batch > training.batch_size
                || training.step_min < first_topology_event_step
                || config.event_training.recovery_extension_budget() > 4_096))
        || !(0.0..=1.0).contains(&training.update_prob)
        || training.update_prob == 0.0
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive Target2D training dimensions, rollout range, pool, or update probability are invalid"
                .to_string(),
        )
        .into());
    }
    if model.config.target_leaves != training.particle_count {
        return Err(AutomataError::InvalidArgument(format!(
            "adaptive Target2D active rows ({}) must match model target_leaves ({})",
            training.particle_count, model.config.target_leaves,
        ))
        .into());
    }

    let material = config.material.layout(
        training.particle_count,
        model.config.perception.min_bandwidth,
        model.config.perception.max_bandwidth,
    )?;
    let coarse_particle_count = material.coarse_particle_count();
    let measure_error =
        (material.represented_measure.iter().sum::<f32>() - config.material.total_measure).abs();
    let scale_limits_report = adaptive_target2d_scale_limits_report(model, &config, &material);
    let seed_bank = build_adaptive_target2d_seed_bank(
        model,
        &material,
        training.pool_size.max(training.batch_size).max(1),
        training.seed,
        &checkpoint_seeds,
        training.seed_mode,
        training.seed_scale,
        config.material.total_measure,
        config.material.fine_bandwidth,
    )?;
    let mut perception: AdaptivePerceptionConfig = model.config.perception;
    perception.graph_policy = model.config.rule_graph_policy;
    perception.reference_measure = fine_measure;
    perception.min_bandwidth = material
        .bandwidth
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    perception.max_bandwidth = material.bandwidth.iter().copied().fold(0.0_f32, f32::max);
    let perception_options = AdaptiveNpaPerceptionOptions {
        eps0: model.rule.config.eps0,
        scale_equivariance: model.rule.config.scale_equivariant(),
        particle_density_equivariance: model.rule.config.particle_density_equivariant(),
        log_norm_grad: model.rule.config.log_norm_grad,
        log_norm_density_grad: model.rule.config.log_norm_density_grad,
        position_features: model.rule.config.position_features,
    };

    let automatic_tbptt = if training.tbptt_chunk_steps == 0 {
        training.step_max.clamp(1, 32)
    } else {
        training.tbptt_chunk_steps.clamp(1, training.step_max)
    };
    let trained_rule_config = if frozen_base_residual {
        model
            .local_residual_rule
            .as_ref()
            .expect("compatible residual was initialized")
            .config
            .clone()
    } else {
        model.rule.config.clone()
    };
    let training_example = DirectBasisTrainingExample {
        target,
        adapter: NpaLowRankAdapter::zeros(&trained_rule_config, 1, 1.0),
        last_train_loss: None,
        particle_count: Some(training.particle_count),
        update_prob: Some(training.update_prob),
        seed_scale: Some(training.seed_scale),
    };
    let total_steps = training
        .epochs
        .saturating_add(1)
        .saturating_mul(training.repetitions);
    let direct_config = DirectBasisTrainConfig {
        steps: total_steps,
        report_interval: training.report_interval.max(1),
        example_batch_size: training.batch_size,
        tbptt_chunk_steps: automatic_tbptt,
        loss_on_final_chunk_only: true,
        use_particle_pool: true,
        pool_size: training.pool_size,
        inject_seed_interval: training.inject_seed_interval.max(1),
        brush_size: training.brush_size,
        stopgrad_pos: trained_rule_config.stopgrad_pos,
        stopgrad_state: trained_rule_config.stopgrad_state,
        rollout_particles: training.particle_count,
        rollout_step_min: training.step_min,
        rollout_steps: training.step_max,
        update_prob: training.update_prob,
        seed: training.seed,
        seed_scale: training.seed_scale,
        seed_mode: training.seed_mode,
        grid_eps: hashgrid.eps,
        motion_scale: model.rule.config.alpha * model.rule.config.motion_eps(hashgrid.eps),
        loss_config,
        target2d_loss_backend: Target2dLossBackend::Auto,
        perception_backend: PerceptionRolloutBackend::Auto,
        per_parameter_grad_normalization: training.per_parameter_grad_normalization,
        base_sgd: SgdConfig {
            learning_rate: training.optimizer.learning_rate,
            weight_decay: training.optimizer.weight_decay,
            grad_clip_norm: training.optimizer.grad_clip_norm,
        },
        adapter_sgd: SgdConfig {
            learning_rate: 0.0,
            weight_decay: 0.0,
            grad_clip_norm: 0.0,
        },
        adapter_l2_weight: 0.0,
        update_base: true,
        eval_examples: 1,
        eval_interval: training.report_interval.max(1),
        eval_batch_size: 1,
        eval_seed: training.seed,
        system_memory_budget_gb: Some(24.0),
        gpu_memory_budget_gb: Some(24.0),
        max_dense_train_particles: 4_096,
        max_dense_chunk_floats: 512 * 1024,
        max_splat_chunk_floats: 512 * 1024,
    };
    let plan = Target2dOracleTrainPlan {
        train: direct_config,
        steps_per_repetition: training.epochs.saturating_add(1),
        repetitions: training.repetitions,
        optimizer: AdamWConfig {
            ..training.optimizer
        },
        scheduler_milestones: training.scheduler_milestones.clone(),
        scheduler_gamma: training.scheduler_gamma,
    };
    let adaptive = AdaptiveTarget2dBurnConfig {
        material,
        topology: topology_config,
        perception,
        perception_options,
        perception_semantics: if normalized_adaptive_rule {
            burn_automata_kernels::AdaptivePerceptionSemantics::NormalizedAdaptive
        } else {
            burn_automata_kernels::AdaptivePerceptionSemantics::NpaCompatible
        },
        residual_perception_semantics: frozen_base_residual.then_some(
            if normalized_adaptive_residual {
                burn_automata_kernels::AdaptivePerceptionSemantics::NormalizedAdaptive
            } else {
                burn_automata_kernels::AdaptivePerceptionSemantics::NpaCompatible
            },
        ),
        seed_bank,
        frozen_base: frozen_base_residual.then(|| model.rule.clone()),
        material_scale_conditioning,
        optimize_material_scale_only: config.optimize_material_scale_only,
        log1p_trajectory_loss: config.log1p_trajectory_loss,
        trajectory_tail_fraction: config.trajectory_tail_fraction,
        trajectory_tail_weight: config.trajectory_tail_weight,
        compatible_residual_material_features: residual_material_features,
        compact_recurrent_memory_dims: config.compact_recurrent_memory_dims,
        fresh_seed_trajectories: config.fresh_seed_trajectories,
        checkpoint_horizons,
        max_pool_age_steps: config.max_pool_age_steps,
        pool_age_strata: config.pool_age_strata,
        backward_loss_scale: config.backward_loss_scale,
        event_training: config.event_training,
    };
    let checkpoint = checkpoint_config.map(|checkpoint| Target2dBurnCheckpointConfig {
        current_model_output: checkpoint.current_model_output.clone(),
        best_model_output: checkpoint.best_model_output.clone(),
        metadata_output: checkpoint.metadata_output.clone(),
        training_state_output: checkpoint.training_state_output.clone(),
        resume_training_state: checkpoint.resume_training_state.clone(),
        resume_model_sha256: checkpoint.resume_model_sha256.clone(),
        curriculum_resume: checkpoint.curriculum_resume,
        include_particle_pool: checkpoint.include_particle_pool,
        model_config: trained_rule_config.clone(),
        hashgrid: hashgrid.clone(),
        source: checkpoint.source.clone(),
        interval_steps: checkpoint.interval_steps,
        interval_duration: checkpoint.interval_duration,
    });
    let mut observer_bridge = observer.map(|observer| AdaptiveTarget2dObserverBridge {
        template: model.clone(),
        frozen_base_residual,
        observer,
    });
    let mut output = match (backend, frozen_base_residual) {
        (crate::Target2dGpuBackend::Wgpu, false) => dense::train_adaptive_target2d_burn_wgpu(
            &mut model.rule,
            &training_example,
            plan,
            adaptive,
            checkpoint.as_ref(),
            observer_bridge
                .as_mut()
                .map(|observer| observer as &mut dyn crate::Target2dGpuTrainingObserver),
        )?,
        (crate::Target2dGpuBackend::Cuda, false) => dense::train_adaptive_target2d_burn_cuda(
            &mut model.rule,
            &training_example,
            plan,
            adaptive,
            checkpoint.as_ref(),
            observer_bridge
                .as_mut()
                .map(|observer| observer as &mut dyn crate::Target2dGpuTrainingObserver),
        )?,
        (crate::Target2dGpuBackend::Wgpu, true) => dense::train_adaptive_target2d_burn_wgpu(
            model
                .local_residual_rule
                .as_mut()
                .expect("compatible residual was initialized"),
            &training_example,
            plan,
            adaptive,
            checkpoint.as_ref(),
            observer_bridge
                .as_mut()
                .map(|observer| observer as &mut dyn crate::Target2dGpuTrainingObserver),
        )?,
        (crate::Target2dGpuBackend::Cuda, true) => dense::train_adaptive_target2d_burn_cuda(
            model
                .local_residual_rule
                .as_mut()
                .expect("compatible residual was initialized"),
            &training_example,
            plan,
            adaptive,
            checkpoint.as_ref(),
            observer_bridge
                .as_mut()
                .map(|observer| observer as &mut dyn crate::Target2dGpuTrainingObserver),
        )?,
    };
    if let Some(metrics) = output.metrics.as_object_mut() {
        metrics.insert("scale_limits".to_owned(), scale_limits_report);
    }
    if let Some(checkpoint) = checkpoint_config
        && checkpoint.best_model_output.is_file()
    {
        let manifest = crate::import::load_manifest(&checkpoint.best_model_output)?;
        if manifest.config != trained_rule_config || manifest.hashgrid != *hashgrid {
            return Err(AutomataError::InvalidModel(
                "adaptive Target2D best checkpoint does not match the trained rule/hashgrid"
                    .to_string(),
            )
            .into());
        }
        let best = manifest.into_model();
        if frozen_base_residual {
            model.local_residual_rule = Some(best);
        } else {
            model.rule = best;
        }
    }
    model.validate()?;

    let best_eval_loss = output
        .metrics
        .get("best_fresh_seed_eval_loss")
        .and_then(serde_json::Value::as_f64)
        .map(|value| value as f32);
    let best_psnr = output
        .metrics
        .get("best_fresh_seed_render_rgb_psnr_db")
        .and_then(serde_json::Value::as_array)
        .and_then(|values| values.first())
        .and_then(serde_json::Value::as_f64)
        .map(|value| value as f32);
    let best_psnr_step = output
        .metrics
        .get("best_fresh_seed_eval_step")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as usize;
    let active_ratio = training.particle_count as f32 / reference_particle_count as f32;
    Ok(AdaptiveTarget2dGpuTrainingReport {
        active_particle_count: training.particle_count,
        reference_particle_count,
        coarse_particle_count,
        visible_gaussian_count: training.particle_count,
        recurrent_row_reduction_fraction: 1.0 - active_ratio,
        pair_work_reduction_fraction: 1.0 - active_ratio * active_ratio,
        material_measure_error: measure_error,
        training: Target2dGpuTrainingReport {
            backend: output.backend.to_string(),
            device: output.device,
            metrics: output.metrics,
            history: output
                .history
                .into_iter()
                .map(|entry| Target2dGpuTrainingHistoryEntry {
                    step: entry.step,
                    loss: entry.loss,
                    eval_loss: entry.eval_loss.map(|loss| crate::Target2dGpuLossSummary {
                        examples: loss.examples,
                        mean_total_loss: loss.mean_total_loss,
                        max_total_loss: loss.max_total_loss,
                        mean_splat_loss: loss.mean_splat_loss,
                        mean_color_loss: loss.mean_color_loss,
                        mean_density_loss: loss.mean_density_loss,
                    }),
                    base_grad_norm: entry.base_grad_norm,
                    base_grad_scale: entry.base_grad_scale,
                    examples_seen: entry.examples_seen,
                    particle_steps_per_sec: entry.particle_steps_per_sec,
                    elapsed_ms: entry.elapsed_ms,
                })
                .collect(),
            best_train_loss: output.best_train_loss.first().copied().flatten(),
            best_train_step: output.best_train_step.first().copied().unwrap_or(0),
            best_fresh_seed_eval_loss: best_eval_loss,
            best_fresh_seed_eval_step: best_psnr_step,
            best_fresh_seed_render_rgb_psnr_db: best_psnr,
            best_fresh_seed_render_rgb_psnr_step: best_psnr_step,
        },
    })
}
