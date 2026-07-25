use std::{collections::BTreeMap, time::Instant};

use serde::{Deserialize, Serialize};

use super::{
    AdaptiveBootstrapChild, AdaptiveCoarseDynamics, AdaptiveControllerOutput, AdaptiveNpaModel,
    AdaptiveParticleSet, AdaptiveProxyHierarchy, AdaptiveRolloutConfig, AdaptiveTopologyControl,
    CanonicalMaterial, canonical_merge, canonical_split, constrained_unequal_split,
    dynamics::{add_proxy_raw_update, local_raw_update, local_raw_update_without_spacing},
    features::{controller_features, proxy_context},
    integration::{
        integrate_closure_basis_update, integrate_closure_mode_update,
        integrate_represented_measure_update, integration_masks,
    },
    material_footprint_radius, normalize_footprint_budget_bounded,
    perception::{rule_perception_pair, rule_perception_without_spacing},
    scale::{
        continuous_split_plan, material_scale_metrics, merge_respects_scale_grading,
        split_respects_scale_grading,
    },
    seed::{
        adaptive_template_child_groups, progressively_restrict_adaptive_particles_to_leaf_budget,
        restore_adaptive_particles_from_templates, restrict_adaptive_particles_to_target,
    },
};
use crate::{AutomataError, AutomataResult, rollout::stable_particle_uniform};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveStepMetrics {
    pub step: usize,
    pub leaf_count: usize,
    pub total_measure: f64,
    pub mean_footprint: f32,
    pub min_footprint: f32,
    pub max_footprint: f32,
    pub footprint_coefficient_of_variation: f32,
    /// Number of occupied material log-scale bins at 1/64 octave resolution.
    /// This is an audit metric, not an execution-level count.
    #[serde(default)]
    pub occupied_material_scale_bins: usize,
    /// Fraction of material leaves whose footprint is not on an integer
    /// octave relative to the configured reference footprint.
    #[serde(default)]
    pub fractional_material_scale_fraction: f32,
    /// RMS distance to the nearest dyadic material scale, in octaves.
    #[serde(default)]
    pub dyadic_scale_quantization_rmse_octaves: f32,
    pub mean_bandwidth: f32,
    /// Broad-phase source rows whose exact support was evaluated.
    #[serde(default)]
    pub candidate_visits: usize,
    pub raw_messages: usize,
    pub accepted_messages: usize,
    pub proxy_nodes: usize,
    pub proxy_messages: usize,
    pub moment_fallback_fraction: f32,
    pub split_events: usize,
    pub merge_events: usize,
    pub max_split_probability: f32,
    pub max_merge_probability: f32,
    pub max_compatible_merge_probability: f32,
    pub eligible_split_candidates: usize,
    pub eligible_merge_clusters: usize,
    #[serde(default)]
    pub min_desired_footprint_ratio: f32,
    #[serde(default)]
    pub max_desired_footprint_ratio: f32,
    #[serde(default)]
    pub mean_event_state_transfer_rms: f32,
    #[serde(default)]
    pub max_event_state_transfer_rms: f32,
    pub mean_displacement: f32,
    #[serde(default)]
    pub perception_ms: f64,
    #[serde(default)]
    pub controller_ms: f64,
    #[serde(default)]
    pub local_rule_ms: f64,
    #[serde(default)]
    pub proxy_rule_ms: f64,
    #[serde(default)]
    pub integration_ms: f64,
    #[serde(default)]
    pub topology_ms: f64,
    #[serde(default)]
    pub total_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct TopologyDecision {
    split_events: usize,
    merge_events: usize,
    max_compatible_merge_probability: f32,
    eligible_split_candidates: usize,
    eligible_merge_clusters: usize,
    min_desired_footprint_ratio: f32,
    max_desired_footprint_ratio: f32,
    state_transfer_rms_sum: f32,
    state_transfer_events: usize,
    max_event_state_transfer_rms: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveTopologyUpdate {
    pub step: usize,
    pub initial_leaf_count: usize,
    pub final_leaf_count: usize,
    pub split_events: usize,
    pub merge_events: usize,
    pub elapsed_ms: f64,
}

#[derive(Debug)]
struct MergeCandidate {
    resolution_cost: f32,
    state_rms: f32,
    confidence: f32,
    group: u64,
    indices: Vec<usize>,
}

#[derive(Clone, Debug)]
struct SplitCandidate {
    resolution_benefit: f32,
    confidence: f32,
    particle_id: u64,
    index: usize,
    child_fractions: Vec<f64>,
}

#[derive(Clone, Debug)]
struct SplitSelection {
    index: usize,
    child_fractions: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveSnapshot {
    pub step: usize,
    pub particles: AdaptiveParticleSet,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveRolloutTrace {
    pub particles: AdaptiveParticleSet,
    pub steps: usize,
    pub metrics: Vec<AdaptiveStepMetrics>,
    pub snapshots: Vec<AdaptiveSnapshot>,
}

pub fn run_adaptive_rollout(
    model: &AdaptiveNpaModel,
    initial_particles: AdaptiveParticleSet,
    rollout: AdaptiveRolloutConfig,
) -> AutomataResult<AdaptiveRolloutTrace> {
    advance_adaptive_rollout_internal(
        model,
        initial_particles,
        rollout,
        0,
        model.config.runtime_topology_control,
    )
}

pub(crate) fn run_adaptive_rollout_with_topology_control(
    model: &AdaptiveNpaModel,
    initial_particles: AdaptiveParticleSet,
    rollout: AdaptiveRolloutConfig,
    topology_control: AdaptiveTopologyControl,
) -> AutomataResult<AdaptiveRolloutTrace> {
    advance_adaptive_rollout_internal(model, initial_particles, rollout, 0, topology_control)
}

/// Advances an existing adaptive particle set from an absolute rollout step.
/// Chunked calls are numerically equivalent to one uninterrupted rollout.
pub fn advance_adaptive_rollout(
    model: &AdaptiveNpaModel,
    initial_particles: AdaptiveParticleSet,
    rollout: AdaptiveRolloutConfig,
    completed_steps: usize,
) -> AutomataResult<AdaptiveRolloutTrace> {
    advance_adaptive_rollout_internal(
        model,
        initial_particles,
        rollout,
        completed_steps,
        model.config.runtime_topology_control,
    )
}

pub(crate) fn advance_adaptive_rollout_with_topology_control(
    model: &AdaptiveNpaModel,
    initial_particles: AdaptiveParticleSet,
    rollout: AdaptiveRolloutConfig,
    completed_steps: usize,
    topology_control: AdaptiveTopologyControl,
) -> AutomataResult<AdaptiveRolloutTrace> {
    advance_adaptive_rollout_internal(
        model,
        initial_particles,
        rollout,
        completed_steps,
        topology_control,
    )
}

/// Applies only the adaptive metadata/topology phase to a particle state whose
/// ordinary dynamics were advanced by a resident GPU executor.
pub fn apply_adaptive_topology_at_step(
    model: &AdaptiveNpaModel,
    particles: &mut AdaptiveParticleSet,
    step: usize,
    elapsed_dynamics_steps: usize,
) -> AutomataResult<AdaptiveTopologyUpdate> {
    apply_adaptive_topology_at_step_with_control(
        model,
        particles,
        step,
        elapsed_dynamics_steps,
        model.config.runtime_topology_control,
    )
}

pub(crate) fn apply_adaptive_topology_at_step_with_control(
    model: &AdaptiveNpaModel,
    particles: &mut AdaptiveParticleSet,
    step: usize,
    elapsed_dynamics_steps: usize,
    topology_control: AdaptiveTopologyControl,
) -> AutomataResult<AdaptiveTopologyUpdate> {
    let started = Instant::now();
    model.validate()?;
    particles.validate()?;
    particles.decrement_cooldown_by(elapsed_dynamics_steps);
    let initial_leaf_count = particles.len();
    if let Some(target) = model
        .config
        .scheduled_restriction_target(step, particles.len())
    {
        let old_groups = adaptive_template_child_groups(particles);
        let fine = restore_adaptive_particles_from_templates(particles)?;
        *particles = progressively_restrict_adaptive_particles_to_leaf_budget(
            model, particles, &fine, target,
        )?;
        let new_groups = adaptive_template_child_groups(particles);
        let (split_events, merge_events) = if model.config.hierarchical_restriction_arity
            == super::AdaptiveRestrictionArity::Canonical
        {
            let event_delta = 2 * particles.spatial_dims - 1;
            (
                particles.len().saturating_sub(initial_leaf_count) / event_delta,
                initial_leaf_count.saturating_sub(particles.len()) / event_delta,
            )
        } else {
            (
                old_groups.difference(&new_groups).count(),
                new_groups.difference(&old_groups).count(),
            )
        };
        particles.validate()?;
        return Ok(AdaptiveTopologyUpdate {
            step,
            initial_leaf_count,
            final_leaf_count: particles.len(),
            split_events,
            merge_events,
            elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
        });
    }
    let perception = rule_perception_pair(&model.config, &model.rule, particles)?;
    if matches!(
        topology_control,
        AdaptiveTopologyControl::PairedLocalDetail | AdaptiveTopologyControl::ContinuousLocalDetail
    ) {
        let detail = super::features::local_detail_risk(particles, &perception.normalized);
        let decision =
            apply_local_detail_topology(model, particles, &detail, step, topology_control)?;
        particles.validate()?;
        return Ok(AdaptiveTopologyUpdate {
            step,
            initial_leaf_count,
            final_leaf_count: particles.len(),
            split_events: decision.split_events,
            merge_events: decision.merge_events,
            elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
        });
    }
    let base_update = model
        .rule
        .forward_update_from_features(&perception.npa_compatible.features)?;
    let features = controller_features(
        &model.config,
        particles,
        &perception.normalized,
        &base_update,
    );
    let controller = topology_controller_output(
        model,
        particles,
        &perception.normalized,
        &features,
        topology_control,
    )?;
    let decision = apply_topology(
        model,
        particles,
        &controller,
        Some(&perception.normalized.state_gradient),
        step,
    )?;
    particles.validate()?;
    Ok(AdaptiveTopologyUpdate {
        step,
        initial_leaf_count,
        final_leaf_count: particles.len(),
        split_events: decision.split_events,
        merge_events: decision.merge_events,
        elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
    })
}

/// Exposes a bounded tranche of an already materialized hierarchical seed.
///
/// Persistent-fine GPU rollout advances every fine mode from step zero. During
/// the coarse-to-fine bootstrap, changing the visible partition therefore does
/// not require perception, controller inference, or a resident-state readback.
/// This helper mutates only the host-side partition metadata; the caller remaps
/// the unchanged device-resident fine modes onto the resulting visible rows.
#[cfg(any(feature = "gpu_wgpu", test))]
pub(crate) fn apply_hierarchical_bootstrap_refinement(
    model: &AdaptiveNpaModel,
    particles: &mut AdaptiveParticleSet,
    step: usize,
    elapsed_dynamics_steps: usize,
) -> AutomataResult<AdaptiveTopologyUpdate> {
    let started = Instant::now();
    model.validate()?;
    particles.validate()?;
    let initial_leaf_count = particles.len();
    if !model
        .config
        .coarse_to_fine_bootstrap_active(step, initial_leaf_count)
    {
        return Err(AutomataError::InvalidArgument(format!(
            "hierarchical bootstrap refinement is inactive at step {step} with {initial_leaf_count} leaves",
        )));
    }
    particles.decrement_cooldown_by(elapsed_dynamics_steps);
    let child_count = 2 * particles.spatial_dims;
    let leaf_delta = child_count.saturating_sub(1);
    let target = model.config.bootstrap_target_leaf_count();
    let remaining = target.saturating_sub(initial_leaf_count);
    let split_count = model
        .config
        .topology_event_budget(step, initial_leaf_count)
        .min(remaining / leaf_delta);
    if split_count == 0 {
        return Ok(AdaptiveTopologyUpdate {
            step,
            initial_leaf_count,
            final_leaf_count: initial_leaf_count,
            split_events: 0,
            merge_events: 0,
            elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
        });
    }

    let templates = particles
        .bootstrap_templates
        .iter()
        .map(|template| (template.parent_id, template.children.len()))
        .collect::<BTreeMap<_, _>>();
    let mut candidates = particles
        .particle_id
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, particle_id)| {
            templates
                .get(&particle_id)
                .copied()
                .filter(|count| *count == child_count)
                .map(|count| (particle_id, index, count))
        })
        .collect::<Vec<_>>();
    candidates.sort_unstable_by_key(|(particle_id, _, _)| *particle_id);
    if candidates.len() < split_count {
        return Err(AutomataError::InvalidModel(format!(
            "hierarchical bootstrap has {} refinable parents but needs {split_count}",
            candidates.len(),
        )));
    }
    let splits = candidates
        .into_iter()
        .take(split_count)
        .map(|(_, index, count)| SplitSelection {
            index,
            child_fractions: vec![1.0 / count as f64; count],
        })
        .collect::<Vec<_>>();
    let transfer = rebuild_particles(
        particles,
        &[],
        &splits,
        RebuildOptions {
            state_gradient: None,
            log_normalized_gradient: model.config.perception.log_normalize_gradients,
            state_prolongation_scale: 0.0,
            max_state_transfer_rms: model.config.split_state_transfer_rms_limit(),
            restore_bootstrap_children: true,
            bootstrap_templates_are_current: true,
            bootstrap_seed_spread: 0.0,
            cooldown: model.config.cooldown_steps,
            domain_min: &model.config.domain_min,
            domain_max: &model.config.domain_max,
        },
    )?;
    particles.validate()?;
    debug_assert_eq!(transfer.events, split_count);
    Ok(AdaptiveTopologyUpdate {
        step,
        initial_leaf_count,
        final_leaf_count: particles.len(),
        split_events: split_count,
        merge_events: 0,
        elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
    })
}

/// Canonically splits current coarse rows without retained fine templates.
///
/// Candidate ordering matches the resident WGPU kernel exactly: largest
/// represented measure first, with the highest row as the deterministic
/// tiebreak. This keeps host material metadata aligned with the active device
/// prefix while positions and recurrent state remain device-resident.
#[cfg(any(feature = "gpu_wgpu", test))]
pub(crate) fn apply_resident_canonical_bootstrap_refinement(
    model: &AdaptiveNpaModel,
    particles: &mut AdaptiveParticleSet,
    step: usize,
    elapsed_dynamics_steps: usize,
) -> AutomataResult<AdaptiveTopologyUpdate> {
    const MAX_RESIDENT_SPLITS: usize = 256;
    let started = Instant::now();
    model.validate()?;
    particles.validate()?;
    let initial_leaf_count = particles.len();
    if particles.spatial_dims != 2
        || !particles.bootstrap_templates.is_empty()
        || model.config.bootstrap_seed_spread != 0.0
        || model.config.closure_recurrent_mode
        || !model
            .config
            .coarse_to_fine_bootstrap_active(step, initial_leaf_count)
    {
        return Err(AutomataError::InvalidArgument(
            "resident canonical bootstrap requires an active untemplated 2D zero-spread seed without recurrent closure"
                .to_owned(),
        ));
    }
    particles.decrement_cooldown_by(elapsed_dynamics_steps);
    let target = model.config.bootstrap_target_leaf_count();
    let split_count = model
        .config
        .topology_event_budget(step, initial_leaf_count)
        .min(MAX_RESIDENT_SPLITS)
        .min(target.saturating_sub(initial_leaf_count) / 3);
    if split_count == 0 {
        return Ok(AdaptiveTopologyUpdate {
            step,
            initial_leaf_count,
            final_leaf_count: initial_leaf_count,
            split_events: 0,
            merge_events: 0,
            elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
        });
    }

    let mut candidates = (0..particles.len()).collect::<Vec<_>>();
    candidates.sort_unstable_by(|lhs, rhs| {
        particles.represented_measure[*rhs]
            .total_cmp(&particles.represented_measure[*lhs])
            .then_with(|| rhs.cmp(lhs))
    });
    candidates.truncate(split_count);
    let first_sibling_group = particles.next_sibling_group;
    let parent_bandwidth = candidates
        .iter()
        .map(|index| particles.bandwidth[*index])
        .collect::<Vec<_>>();
    let splits = candidates
        .into_iter()
        .map(|index| SplitSelection {
            index,
            child_fractions: vec![0.25; 4],
        })
        .collect::<Vec<_>>();
    let transfer = rebuild_particles(
        particles,
        &[],
        &splits,
        RebuildOptions {
            state_gradient: None,
            log_normalized_gradient: model.config.perception.log_normalize_gradients,
            state_prolongation_scale: 0.0,
            max_state_transfer_rms: model.config.split_state_transfer_rms_limit(),
            restore_bootstrap_children: false,
            bootstrap_templates_are_current: false,
            bootstrap_seed_spread: 0.0,
            cooldown: model.config.cooldown_steps,
            domain_min: &model.config.domain_min,
            domain_max: &model.config.domain_max,
        },
    )?;
    let child_bandwidth_scale = 0.25_f32.powf(model.config.material_seed_bandwidth_exponent);
    for (event, bandwidth) in parent_bandwidth.into_iter().enumerate() {
        let group = first_sibling_group + event as u64;
        for (row, sibling_group) in particles.sibling_group.iter().copied().enumerate() {
            if sibling_group == group {
                particles.bandwidth[row] = bandwidth * child_bandwidth_scale;
            }
        }
    }
    particles.validate()?;
    debug_assert_eq!(transfer.events, split_count);
    Ok(AdaptiveTopologyUpdate {
        step,
        initial_leaf_count,
        final_leaf_count: particles.len(),
        split_events: split_count,
        merge_events: 0,
        elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
    })
}

fn advance_adaptive_rollout_internal(
    model: &AdaptiveNpaModel,
    initial_particles: AdaptiveParticleSet,
    rollout: AdaptiveRolloutConfig,
    completed_steps: usize,
    topology_control: AdaptiveTopologyControl,
) -> AutomataResult<AdaptiveRolloutTrace> {
    model.validate()?;
    initial_particles.validate()?;
    rollout.validate()?;
    if initial_particles.spatial_dims != model.config.spatial_dims
        || initial_particles.state_dims != model.rule.config.state_dims
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive rollout particle/model dimensions do not match".to_string(),
        ));
    }
    if initial_particles.len() < model.config.min_leaves
        || initial_particles.len() > model.config.max_leaves
    {
        return Err(AutomataError::InvalidArgument(format!(
            "adaptive initial leaf count {} is outside {}..={} ",
            initial_particles.len(),
            model.config.min_leaves,
            model.config.max_leaves
        )));
    }

    let mut particles = initial_particles;
    for footprint in &mut particles.render_footprint {
        *footprint = model.config.render_footprint(*footprint);
    }
    let mut metrics = Vec::with_capacity(rollout.steps);
    let adapt_bandwidth =
        rollout.bandwidth_adaptation_enabled && model.config.supports_bandwidth_adaptation();
    let mut snapshots = vec![AdaptiveSnapshot {
        step: completed_steps,
        particles: particles.clone(),
    }];
    for local_step in 1..=rollout.steps {
        let step_started = Instant::now();
        let step = completed_steps.saturating_add(local_step);
        // Bootstrap refines an unformed seed. Apply it before recurrent
        // dynamics so the fixed-resolution rule never receives one damaging
        // update on the under-resolved population. Steady topology remains a
        // post-update operation below.
        let bootstrap_started = Instant::now();
        let mut bootstrap_controller = None;
        let bootstrap_topology = if rollout.topology_enabled
            && model
                .config
                .coarse_to_fine_bootstrap_active(step, particles.len())
            && model.config.is_topology_step(step, particles.len())
        {
            let perception = rule_perception_pair(&model.config, &model.rule, &particles)?;
            let base_update = model
                .rule
                .forward_update_from_features(&perception.npa_compatible.features)?;
            let features = controller_features(
                &model.config,
                &particles,
                &perception.normalized,
                &base_update,
            );
            let controller = topology_controller_output(
                model,
                &particles,
                &perception.normalized,
                &features,
                topology_control,
            )?;
            let decision = if matches!(
                topology_control,
                AdaptiveTopologyControl::PairedLocalDetail
                    | AdaptiveTopologyControl::ContinuousLocalDetail
            ) {
                let detail = super::features::local_detail_risk(&particles, &perception.normalized);
                apply_local_detail_topology(model, &mut particles, &detail, step, topology_control)?
            } else {
                apply_topology(
                    model,
                    &mut particles,
                    &controller,
                    Some(&perception.normalized.state_gradient),
                    step,
                )?
            };
            bootstrap_controller = Some(controller);
            decision
        } else {
            TopologyDecision::default()
        };
        let bootstrap_elapsed = bootstrap_started.elapsed();
        let count = particles.len();
        let bootstrap_applied =
            bootstrap_topology.split_events + bootstrap_topology.merge_events > 0;
        let topology_step = rollout.topology_enabled
            && !bootstrap_applied
            && model.config.is_topology_step(step, particles.len());
        let scheduled_restriction = rollout
            .topology_enabled
            .then(|| {
                model
                    .config
                    .scheduled_restriction_target(step, particles.len())
            })
            .flatten();
        let persistent_reallocation = topology_step
            && model.config.coarse_dynamics
                == super::AdaptiveCoarseDynamics::PersistentFineQuadrature;
        let needs_full_perception = (topology_step && !persistent_reallocation)
            || adapt_bandwidth
            || model.config.rule_perception != super::AdaptiveRulePerception::NpaCompatible
            || model.config.local_residual_scale > 0.0
            || model.config.closure_recurrent_mode;
        let perception_started = Instant::now();
        let perception_pair = needs_full_perception
            .then(|| rule_perception_pair(&model.config, &model.rule, &particles))
            .transpose()?;
        let rule_only_perception = (!needs_full_perception)
            .then(|| rule_perception_without_spacing(&model.config, &model.rule, &particles))
            .transpose()?;
        let perception_elapsed = perception_started.elapsed();
        let local_rule_started = Instant::now();
        let mut quadrature_step = None;
        let local = if model.config.coarse_dynamics == super::AdaptiveCoarseDynamics::FineQuadrature
        {
            super::dynamics::fine_quadrature_raw_update(model, &particles)?
        } else if model.config.coarse_dynamics
            == super::AdaptiveCoarseDynamics::PersistentFineQuadrature
        {
            if adapt_bandwidth {
                return Err(AutomataError::InvalidArgument(
                    "persistent fine-quadrature control requires fixed bandwidth after bootstrap"
                        .to_string(),
                ));
            }
            let step = super::dynamics::persistent_fine_quadrature_step(model, &particles)?;
            let local = step.local.clone();
            quadrature_step = Some(step);
            local
        } else if let Some(perception) = &perception_pair {
            local_raw_update(model, &particles, perception)?
        } else {
            local_raw_update_without_spacing(
                model,
                &particles,
                rule_only_perception
                    .as_ref()
                    .expect("rule-only perception is available on the compatible fast path"),
            )?
        };
        let local_rule_elapsed = local_rule_started.elapsed();
        let closure_mode_update = if model.config.closure_recurrent_mode {
            super::dynamics::closure_mode_raw_update(
                model,
                &particles,
                perception_pair
                    .as_ref()
                    .expect("recurrent closure mode requires full perception"),
            )?
        } else {
            None
        };
        let closure_basis_update = if model.config.closure_recurrent_mode {
            super::dynamics::closure_basis_raw_update(
                model,
                &particles,
                perception_pair
                    .as_ref()
                    .expect("recurrent closure mode requires full perception"),
            )?
        } else {
            None
        };
        let controller_started = Instant::now();
        let controller_output = if (topology_step && !persistent_reallocation) || adapt_bandwidth {
            let perception = &perception_pair
                .as_ref()
                .expect("full perception is available for adaptive control")
                .normalized;
            let features = controller_features(&model.config, &particles, perception, &local.base);
            let active_topology_control = if topology_step {
                topology_control
            } else {
                AdaptiveTopologyControl::Learned
            };
            topology_controller_output(
                model,
                &particles,
                perception,
                &features,
                active_topology_control,
            )?
        } else {
            vec![AdaptiveControllerOutput::default(); count]
        };
        let controller_elapsed = controller_started.elapsed();
        let perception = perception_pair.as_ref().map_or_else(
            || {
                rule_only_perception
                    .as_ref()
                    .expect("rule-only perception is available")
            },
            |pair| &pair.normalized,
        );
        refresh_state_jacobian(
            &mut particles,
            &perception.state_gradient,
            model.config.perception.log_normalize_gradients,
            model.config.base_rule_footprint(),
        )?;
        let mut update = local.combined;
        let proxy_rule_started = Instant::now();
        let proxy_context = (!model.uses_deployment_rule()
            && model.config.proxy.context_scale > 0.0)
            .then(|| proxy_context(&model.config, &particles))
            .transpose()?
            .flatten();
        add_proxy_raw_update(model, &particles, proxy_context.as_ref(), &mut update)?;
        let proxy_rule_elapsed = proxy_rule_started.elapsed();
        let integration_started = Instant::now();
        let displacement_sum = if let Some(quadrature_step) = quadrature_step {
            super::dynamics::integrate_fine_quadrature(
                model,
                &mut particles,
                quadrature_step,
                rollout.seed,
                step,
                rollout.update_prob,
                rollout.dt,
            )?
        } else {
            let mask =
                integration_masks(model, &particles, rollout.seed, step, rollout.update_prob);
            let displacement_sum = integrate_represented_measure_update(
                model,
                &mut particles,
                &update,
                &mask,
                rollout.dt,
            )?;
            if let Some(closure_mode_update) = &closure_mode_update {
                integrate_closure_mode_update(
                    model,
                    &mut particles,
                    closure_mode_update,
                    &mask,
                    rollout.dt,
                )?;
            }
            if let Some(closure_basis_update) = &closure_basis_update {
                integrate_closure_basis_update(
                    model,
                    &mut particles,
                    closure_basis_update,
                    &mask,
                    rollout.dt,
                )?;
            }
            if adapt_bandwidth {
                for (index, controller) in controller_output.iter().enumerate().take(count) {
                    let desired_bandwidth = (perception.observed_spacing[index]
                        * controller.log_bandwidth_ratio.exp())
                    .clamp(
                        model.config.perception.min_bandwidth,
                        model.config.perception.max_bandwidth,
                    );
                    particles.bandwidth[index] = lerp(
                        particles.bandwidth[index],
                        desired_bandwidth,
                        model.config.bandwidth_relaxation,
                    );
                }
            }
            displacement_sum
        };
        let integration_elapsed = integration_started.elapsed();
        particles.decrement_cooldown();

        let steady_topology_started = Instant::now();
        let topology = if bootstrap_applied {
            bootstrap_topology
        } else if let Some(target) = scheduled_restriction {
            let initial_leaf_count = particles.len();
            let old_groups = adaptive_template_child_groups(&particles);
            let fine = restore_adaptive_particles_from_templates(&particles)?;
            particles = progressively_restrict_adaptive_particles_to_leaf_budget(
                model, &particles, &fine, target,
            )?;
            let new_groups = adaptive_template_child_groups(&particles);
            let (split_events, merge_events) = if model.config.hierarchical_restriction_arity
                == super::AdaptiveRestrictionArity::Canonical
            {
                let event_delta = 2 * particles.spatial_dims - 1;
                (
                    particles.len().saturating_sub(initial_leaf_count) / event_delta,
                    initial_leaf_count.saturating_sub(particles.len()) / event_delta,
                )
            } else {
                (
                    old_groups.difference(&new_groups).count(),
                    new_groups.difference(&old_groups).count(),
                )
            };
            TopologyDecision {
                split_events,
                merge_events,
                ..TopologyDecision::default()
            }
        } else if persistent_reallocation {
            let old_groups = adaptive_template_child_groups(&particles);
            let fine = restore_adaptive_particles_from_templates(&particles)?;
            particles = restrict_adaptive_particles_to_target(model, &fine)?;
            let new_groups = adaptive_template_child_groups(&particles);
            let changed_groups = old_groups.symmetric_difference(&new_groups).count() / 2;
            TopologyDecision {
                split_events: changed_groups,
                merge_events: changed_groups,
                ..TopologyDecision::default()
            }
        } else if topology_step {
            let perception = &perception_pair
                .as_ref()
                .expect("full perception is available for topology")
                .normalized;
            if matches!(
                topology_control,
                AdaptiveTopologyControl::PairedLocalDetail
                    | AdaptiveTopologyControl::ContinuousLocalDetail
            ) {
                let detail = super::features::local_detail_risk(&particles, perception);
                apply_local_detail_topology(model, &mut particles, &detail, step, topology_control)?
            } else {
                apply_topology(
                    model,
                    &mut particles,
                    &controller_output,
                    Some(&perception.state_gradient),
                    step,
                )?
            }
        } else {
            TopologyDecision::default()
        };
        relax_render_footprints(
            &mut particles,
            model.config.render_footprint_relaxation,
            &model.config,
        );
        let topology_elapsed = bootstrap_elapsed + steady_topology_started.elapsed();
        particles.validate()?;
        let footprints = particles
            .represented_measure
            .iter()
            .map(|measure| material_footprint_radius(*measure, particles.spatial_dims))
            .collect::<Vec<_>>();
        let min_footprint = footprints.iter().copied().fold(f32::INFINITY, f32::min);
        let max_footprint = footprints.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean_footprint = mean(&footprints);
        let footprint_variance = footprints
            .iter()
            .map(|footprint| (*footprint - mean_footprint).powi(2))
            .sum::<f32>()
            / footprints.len().max(1) as f32;
        let scale_metrics = material_scale_metrics(&particles, model.config.reference_footprint);
        metrics.push(AdaptiveStepMetrics {
            step,
            leaf_count: particles.len(),
            total_measure: particles.total_measure(),
            mean_footprint,
            min_footprint,
            max_footprint,
            footprint_coefficient_of_variation: footprint_variance.sqrt()
                / mean_footprint.max(f32::MIN_POSITIVE),
            occupied_material_scale_bins: scale_metrics.occupied_sixty_fourth_octave_bins,
            fractional_material_scale_fraction: scale_metrics.fractional_octave_fraction,
            dyadic_scale_quantization_rmse_octaves: scale_metrics.dyadic_quantization_rmse_octaves,
            mean_bandwidth: mean(&particles.bandwidth),
            candidate_visits: perception.graph.candidate_visits,
            raw_messages: perception.graph.raw_messages,
            accepted_messages: perception.graph.accepted_messages,
            proxy_nodes: proxy_context
                .as_ref()
                .map_or(0, |context| context.node_count),
            proxy_messages: proxy_context
                .as_ref()
                .map_or(0, |context| context.perception.graph.accepted_messages),
            moment_fallback_fraction: perception
                .moment_fallback
                .iter()
                .filter(|fallback| **fallback)
                .count() as f32
                / perception.moment_fallback.len().max(1) as f32,
            split_events: topology.split_events,
            merge_events: topology.merge_events,
            max_split_probability: controller_output
                .iter()
                .chain(bootstrap_controller.iter().flatten())
                .map(|output| output.split_probability)
                .fold(0.0_f32, f32::max),
            max_merge_probability: controller_output
                .iter()
                .chain(bootstrap_controller.iter().flatten())
                .map(|output| output.merge_probability)
                .fold(0.0_f32, f32::max),
            max_compatible_merge_probability: topology.max_compatible_merge_probability,
            eligible_split_candidates: topology.eligible_split_candidates,
            eligible_merge_clusters: topology.eligible_merge_clusters,
            min_desired_footprint_ratio: topology.min_desired_footprint_ratio,
            max_desired_footprint_ratio: topology.max_desired_footprint_ratio,
            mean_event_state_transfer_rms: topology.state_transfer_rms_sum
                / topology.state_transfer_events.max(1) as f32,
            max_event_state_transfer_rms: topology.max_event_state_transfer_rms,
            mean_displacement: displacement_sum / count.max(1) as f32,
            perception_ms: perception_elapsed.as_secs_f64() * 1_000.0,
            controller_ms: controller_elapsed.as_secs_f64() * 1_000.0,
            local_rule_ms: local_rule_elapsed.as_secs_f64() * 1_000.0,
            proxy_rule_ms: proxy_rule_elapsed.as_secs_f64() * 1_000.0,
            integration_ms: integration_elapsed.as_secs_f64() * 1_000.0,
            topology_ms: topology_elapsed.as_secs_f64() * 1_000.0,
            total_ms: step_started.elapsed().as_secs_f64() * 1_000.0,
        });
        if step.is_multiple_of(rollout.snapshot_interval) || local_step == rollout.steps {
            snapshots.push(AdaptiveSnapshot {
                step,
                particles: particles.clone(),
            });
        }
    }

    Ok(AdaptiveRolloutTrace {
        particles,
        steps: rollout.steps,
        metrics,
        snapshots,
    })
}

fn topology_controller_output(
    model: &AdaptiveNpaModel,
    particles: &AdaptiveParticleSet,
    perception: &burn_automata_kernels::AdaptivePerceptionOutput,
    controller_features: &[[f32; super::ADAPTIVE_CONTROLLER_INPUT_DIMS]],
    control: AdaptiveTopologyControl,
) -> AutomataResult<Vec<AdaptiveControllerOutput>> {
    let learned = matches!(
        control,
        AdaptiveTopologyControl::Learned | AdaptiveTopologyControl::LearnedRefinementDefect
    )
    .then(|| model.controller.forward(controller_features));
    if control == AdaptiveTopologyControl::Learned {
        return Ok(learned.expect("learned controller output was requested"));
    }
    let risk = match control {
        AdaptiveTopologyControl::Learned => unreachable!(),
        AdaptiveTopologyControl::LearnedRefinementDefect
        | AdaptiveTopologyControl::RefinementDefectOracle => {
            super::refinement::adaptive_refinement_defect(model, particles)?
        }
        AdaptiveTopologyControl::LocalDetailOracle
        | AdaptiveTopologyControl::PairedLocalDetail
        | AdaptiveTopologyControl::ContinuousLocalDetail => {
            super::features::local_detail_risk(particles, perception)
        }
    };
    let allocation = super::allocate_resolution_budget(
        &risk,
        &particles.represented_measure,
        particles.spatial_dims,
        2.0,
        model.config.reference_footprint,
        model.config.min_footprint,
        model.config.max_footprint,
        model.config.target_leaves,
    )?;
    let learned = learned.as_deref();
    // Spatial confidence is deliberately not used as a secondary candidate
    // ranking. Tiny optimizer differences otherwise select different balanced
    // exchanges despite equivalent controller quality. The learned controller
    // decides whether each event class is active; the refinement defect owns
    // the deterministic location ranking.
    let learned_split_gate = learned.map(|outputs| {
        outputs
            .iter()
            .map(|output| output.split_probability)
            .fold(0.0_f32, f32::max)
    });
    let learned_merge_gate = learned.map(|outputs| {
        outputs
            .iter()
            .map(|output| output.merge_probability)
            .fold(0.0_f32, f32::max)
    });
    Ok(allocation
        .desired_footprint
        .into_iter()
        .enumerate()
        .map(|(index, desired)| {
            let confidence = learned.map(|outputs| outputs[index]);
            AdaptiveControllerOutput {
                desired_log_footprint: (desired / model.config.reference_footprint).ln(),
                log_bandwidth_ratio: confidence.map_or(0.0, |output| output.log_bandwidth_ratio),
                split_probability: learned_split_gate.unwrap_or(1.0),
                merge_probability: learned_merge_gate.unwrap_or(1.0),
            }
        })
        .collect())
}

fn apply_local_detail_topology(
    model: &AdaptiveNpaModel,
    particles: &mut AdaptiveParticleSet,
    detail: &[f32],
    step: usize,
    control: AdaptiveTopologyControl,
) -> AutomataResult<TopologyDecision> {
    match control {
        AdaptiveTopologyControl::PairedLocalDetail => {
            apply_paired_local_detail_topology(model, particles, detail, step)
        }
        AdaptiveTopologyControl::ContinuousLocalDetail => {
            apply_continuous_local_detail_topology(model, particles, detail, step)
        }
        _ => Err(AutomataError::InvalidArgument(
            "local-detail topology dispatcher received a non-local policy".to_owned(),
        )),
    }
}

/// Relocates one fine and one coarse slot from a fixed graded material
/// continuum. The hard ranking is detached. Material metadata remains attached
/// to its slot while position and dynamic intensive fields exchange places.
/// A common measure-weighted correction then restores the original centroid and
/// extensive recurrent state without creating rows or hidden state.
fn apply_continuous_local_detail_topology(
    model: &AdaptiveNpaModel,
    particles: &mut AdaptiveParticleSet,
    detail: &[f32],
    step: usize,
) -> AutomataResult<TopologyDecision> {
    if model.config.material_seed_layout != super::AdaptiveMaterialSeedLayout::GradedContinuous {
        return Err(AutomataError::InvalidArgument(
            "continuous local-detail topology requires a graded-continuous material seed"
                .to_owned(),
        ));
    }
    if particles.spatial_dims != 2 || detail.len() != particles.len() {
        return Err(AutomataError::InvalidArgument(
            "continuous local-detail topology requires a 2D detail value for every active row"
                .to_owned(),
        ));
    }
    if !particles.bootstrap_templates.is_empty() {
        return Err(AutomataError::InvalidArgument(
            "continuous local-detail topology does not permit hidden bootstrap templates"
                .to_owned(),
        ));
    }
    if detail.iter().any(|value| !value.is_finite()) {
        return Err(AutomataError::InvalidArgument(
            "continuous local-detail topology received non-finite detail".to_owned(),
        ));
    }

    let total_measure = particles.total_measure() as f32;
    let mean_measure = total_measure / particles.len() as f32;
    let tolerance = 2.0e-4 * mean_measure;
    let coarse_rows = particles
        .represented_measure
        .iter()
        .enumerate()
        .filter_map(|(row, measure)| (*measure > mean_measure + tolerance).then_some(row))
        .collect::<Vec<_>>();
    let fine_rows = particles
        .represented_measure
        .iter()
        .enumerate()
        .filter_map(|(row, measure)| (*measure + tolerance < mean_measure).then_some(row))
        .collect::<Vec<_>>();
    if coarse_rows.is_empty() || fine_rows.is_empty() || !total_measure.is_finite() {
        return Err(AutomataError::InvalidArgument(
            "continuous local-detail topology requires finite graded material on both sides of the mean"
                .to_owned(),
        ));
    }
    let event_budget = model
        .config
        .topology_event_budget(step, particles.len())
        .min(coarse_rows.len())
        .min(fine_rows.len());
    if event_budget == 0 {
        return Ok(TopologyDecision::default());
    }

    let mut ranked_coarse = coarse_rows.clone();
    ranked_coarse.sort_by(|lhs, rhs| {
        stable_local_detail_rank(detail[*rhs])
            .total_cmp(&stable_local_detail_rank(detail[*lhs]))
            .then_with(|| lhs.cmp(rhs))
    });
    let mut ranked_fine = fine_rows.clone();
    ranked_fine.sort_by(|lhs, rhs| {
        stable_local_detail_rank(detail[*lhs])
            .total_cmp(&stable_local_detail_rank(detail[*rhs]))
            .then_with(|| lhs.cmp(rhs))
    });
    let exchanges = ranked_coarse
        .into_iter()
        .zip(ranked_fine)
        .take(event_budget)
        .filter(|(coarse_row, fine_row)| {
            reallocation_gain_is_sufficient(
                stable_local_detail_rank(detail[*coarse_row]),
                stable_local_detail_rank(detail[*fine_row]),
                model.config.min_reallocation_relative_gain,
            )
        })
        .collect::<Vec<_>>();
    if exchanges.is_empty() {
        return Ok(TopologyDecision {
            eligible_split_candidates: coarse_rows.len(),
            eligible_merge_clusters: fine_rows.len(),
            min_desired_footprint_ratio: 1.0,
            max_desired_footprint_ratio: 1.0,
            ..TopologyDecision::default()
        });
    }

    if !swap_particle_positions_conserving_moments(
        &mut particles.positions,
        &particles.represented_measure,
        &exchanges,
        total_measure,
    ) {
        return Ok(TopologyDecision {
            eligible_split_candidates: coarse_rows.len(),
            eligible_merge_clusters: fine_rows.len(),
            min_desired_footprint_ratio: 1.0,
            max_desired_footprint_ratio: 1.0,
            ..TopologyDecision::default()
        });
    }
    swap_intensive_row_pairs_conserving_extensive(
        &mut particles.states,
        particles.state_dims,
        &particles.represented_measure,
        &exchanges,
        total_measure,
    );
    swap_intensive_row_pairs_conserving_extensive(
        &mut particles.state_jacobian,
        particles.state_dims * particles.spatial_dims,
        &particles.represented_measure,
        &exchanges,
        total_measure,
    );
    if !particles.closure_mode.is_empty() {
        swap_intensive_row_pairs_conserving_extensive(
            &mut particles.closure_mode,
            particles.state_dims,
            &particles.represented_measure,
            &exchanges,
            total_measure,
        );
    }
    let mut min_ratio = 1.0_f32;
    let mut max_ratio = 1.0_f32;
    for (coarse_row, fine_row) in exchanges.iter().copied() {
        swap_optional_particle_rows(&mut particles.closure_basis, coarse_row, fine_row, 4);
        swap_optional_particle_rows(&mut particles.closure_phase, coarse_row, fine_row, 2);
        particles.render_footprint.swap(coarse_row, fine_row);
        let coarse_measure = particles.represented_measure[coarse_row];
        let fine_measure = particles.represented_measure[fine_row];
        min_ratio = min_ratio.min((fine_measure / coarse_measure).sqrt());
        max_ratio = max_ratio.max((coarse_measure / fine_measure).sqrt());
    }

    Ok(TopologyDecision {
        split_events: exchanges.len(),
        merge_events: exchanges.len(),
        eligible_split_candidates: coarse_rows.len(),
        eligible_merge_clusters: fine_rows.len(),
        min_desired_footprint_ratio: min_ratio,
        max_desired_footprint_ratio: max_ratio,
        ..TopologyDecision::default()
    })
}

fn stable_local_detail_rank(detail: f32) -> f32 {
    (detail * 256.0).round() / 256.0
}

/// Exchanges two material slots and projects the complete position cloud back
/// to its original weighted first and second moments. The projection is a
/// near-identity lower-triangular affine map obtained from the old and
/// post-swap covariance Cholesky factors. It is global because no two-point
/// correction can generally preserve a full 2D second moment.
fn swap_particle_positions_conserving_moments(
    positions: &mut [[f32; 4]],
    represented_measure: &[f32],
    exchanges: &[(usize, usize)],
    total_measure: f32,
) -> bool {
    let total = f64::from(total_measure);
    if positions.len() != represented_measure.len()
        || exchanges.is_empty()
        || exchanges.iter().any(|(coarse, fine)| coarse == fine)
        || !total.is_finite()
        || total <= 0.0
    {
        return false;
    }

    let mut source_rows = (0..positions.len()).collect::<Vec<_>>();
    for (coarse, fine) in exchanges.iter().copied() {
        source_rows.swap(coarse, fine);
    }
    let mut old_first = [0.0_f64; 2];
    let mut old_second = [0.0_f64; 3];
    let mut swapped_first = [0.0_f64; 2];
    let mut swapped_second = [0.0_f64; 3];
    for (row, measure) in represented_measure.iter().copied().enumerate() {
        let weight = f64::from(measure);
        let old = positions[row];
        let swapped = positions[source_rows[row]];
        let old_x = f64::from(old[0]);
        let old_y = f64::from(old[1]);
        let swapped_x = f64::from(swapped[0]);
        let swapped_y = f64::from(swapped[1]);
        old_first[0] += weight * old_x;
        old_first[1] += weight * old_y;
        old_second[0] += weight * old_x * old_x;
        old_second[1] += weight * old_x * old_y;
        old_second[2] += weight * old_y * old_y;
        swapped_first[0] += weight * swapped_x;
        swapped_first[1] += weight * swapped_y;
        swapped_second[0] += weight * swapped_x * swapped_x;
        swapped_second[1] += weight * swapped_x * swapped_y;
        swapped_second[2] += weight * swapped_y * swapped_y;
    }
    let old_mean = [old_first[0] / total, old_first[1] / total];
    let swapped_mean = [swapped_first[0] / total, swapped_first[1] / total];
    let old_covariance = [
        old_second[0] / total - old_mean[0] * old_mean[0],
        old_second[1] / total - old_mean[0] * old_mean[1],
        old_second[2] / total - old_mean[1] * old_mean[1],
    ];
    let swapped_covariance = [
        swapped_second[0] / total - swapped_mean[0] * swapped_mean[0],
        swapped_second[1] / total - swapped_mean[0] * swapped_mean[1],
        swapped_second[2] / total - swapped_mean[1] * swapped_mean[1],
    ];
    let Some(affine) = covariance_transport_2d(swapped_covariance, old_covariance) else {
        return false;
    };

    let mut translation = [
        old_mean[0] - swapped_mean[0],
        old_mean[1] - swapped_mean[1],
        0.0,
        0.0,
    ];
    for axis in 2..4 {
        let mut old_moment = 0.0_f64;
        let mut swapped_moment = 0.0_f64;
        for (row, measure) in represented_measure.iter().copied().enumerate() {
            old_moment += f64::from(measure) * f64::from(positions[row][axis]);
            swapped_moment += f64::from(measure) * f64::from(positions[source_rows[row]][axis]);
        }
        translation[axis] = (old_moment - swapped_moment) / total;
    }
    let source_positions = positions.to_vec();
    for (row, position) in positions.iter_mut().enumerate() {
        let source = source_positions[source_rows[row]];
        let centered_x = f64::from(source[0]) + translation[0] - old_mean[0];
        let centered_y = f64::from(source[1]) + translation[1] - old_mean[1];
        position[0] = (old_mean[0] + affine[0] * centered_x + affine[1] * centered_y) as f32;
        position[1] = (old_mean[1] + affine[2] * centered_x + affine[3] * centered_y) as f32;
        position[2] = (f64::from(source[2]) + translation[2]) as f32;
        position[3] = (f64::from(source[3]) + translation[3]) as f32;
    }
    true
}

/// Returns `A` such that `A source A^T = target` for symmetric 2x2
/// covariances, using `A = chol(target) * inv(chol(source))`.
fn covariance_transport_2d(source: [f64; 3], target: [f64; 3]) -> Option<[f64; 4]> {
    fn cholesky(covariance: [f64; 3]) -> Option<[f64; 3]> {
        if covariance.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let scale = (covariance[0].abs() + covariance[2].abs()).max(f64::MIN_POSITIVE);
        let floor = 1.0e-12 * scale;
        if covariance[0] <= floor {
            return None;
        }
        let l00 = covariance[0].sqrt();
        let l10 = covariance[1] / l00;
        let residual = covariance[2] - l10 * l10;
        if residual <= floor {
            return None;
        }
        Some([l00, l10, residual.sqrt()])
    }

    let source = cholesky(source)?;
    let target = cholesky(target)?;
    Some([
        target[0] / source[0],
        0.0,
        target[1] / source[0] - target[2] * source[1] / (source[0] * source[2]),
        target[2] / source[2],
    ])
}

fn swap_intensive_row_pairs_conserving_extensive(
    values: &mut [f32],
    width: usize,
    represented_measure: &[f32],
    exchanges: &[(usize, usize)],
    total_measure: f32,
) {
    if values.is_empty() || width == 0 {
        return;
    }
    let source = values.to_vec();
    let mut source_rows = (0..represented_measure.len()).collect::<Vec<_>>();
    for (coarse, fine) in exchanges.iter().copied() {
        source_rows.swap(coarse, fine);
    }
    let mut correction = vec![0.0_f64; width];
    for row in 0..represented_measure.len() {
        let weight = f64::from(represented_measure[row]);
        let source_row = source_rows[row];
        for channel in 0..width {
            correction[channel] += weight
                * (f64::from(source[row * width + channel])
                    - f64::from(source[source_row * width + channel]));
        }
    }
    let reciprocal_total = 1.0 / f64::from(total_measure);
    for row in 0..represented_measure.len() {
        let source_row = source_rows[row];
        for channel in 0..width {
            values[row * width + channel] = (f64::from(source[source_row * width + channel])
                + correction[channel] * reciprocal_total)
                as f32;
        }
    }
}

fn swap_optional_particle_rows(values: &mut [f32], lhs_row: usize, rhs_row: usize, width: usize) {
    if values.is_empty() {
        return;
    }
    for channel in 0..width {
        values.swap(lhs_row * width + channel, rhs_row * width + channel);
    }
}

/// Mirrors the detached, fixed-budget topology operation used by adaptive
/// Target2D training. Material metadata remains attached to static coarse and
/// fine slots; only position and recurrent intensive state are exchanged.
/// Consequently the operation conserves represented measure, centroid, and
/// every extensive recurrent-state channel exactly up to floating-point
/// reduction order.
fn apply_paired_local_detail_topology(
    model: &AdaptiveNpaModel,
    particles: &mut AdaptiveParticleSet,
    detail: &[f32],
    step: usize,
) -> AutomataResult<TopologyDecision> {
    if particles.spatial_dims != 2 || detail.len() != particles.len() {
        return Err(AutomataError::InvalidArgument(
            "paired local-detail topology requires a 2D detail value for every active row"
                .to_string(),
        ));
    }
    if !particles.bootstrap_templates.is_empty() {
        return Err(AutomataError::InvalidArgument(
            "paired local-detail topology does not permit hidden bootstrap templates".to_string(),
        ));
    }
    if detail.iter().any(|value| !value.is_finite()) {
        return Err(AutomataError::InvalidArgument(
            "paired local-detail topology received non-finite detail".to_string(),
        ));
    }

    let fine_measure = particles
        .represented_measure
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let classify =
        |measure: f32, units: f32| (measure / fine_measure - units).abs() <= 2.0e-4 * units;
    let coarse_rows = particles
        .represented_measure
        .iter()
        .enumerate()
        .filter_map(|(row, measure)| classify(*measure, 4.0).then_some(row))
        .collect::<Vec<_>>();
    let fine_rows = particles
        .represented_measure
        .iter()
        .enumerate()
        .filter_map(|(row, measure)| classify(*measure, 1.0).then_some(row))
        .collect::<Vec<_>>();
    if coarse_rows.is_empty()
        || fine_rows.len() < 4
        || coarse_rows.len() + fine_rows.len() != particles.len()
    {
        return Err(AutomataError::InvalidArgument(
            "paired local-detail topology requires only one-unit fine rows and four-unit coarse rows"
                .to_string(),
        ));
    }

    let pair_budget = model
        .config
        .topology_event_budget(step, particles.len())
        .min(coarse_rows.len())
        .min(fine_rows.len() / 4);
    if pair_budget == 0 {
        return Ok(TopologyDecision::default());
    }
    let fine_footprint_squared = material_footprint_radius(fine_measure, 2)
        .powi(2)
        .max(f32::MIN_POSITIVE);
    let split_radius = (1.5_f32).sqrt()
        * fine_footprint_squared.sqrt()
        * model.config.paired_topology_split_radius_scale;
    let mut consumed = vec![false; particles.len()];
    let mut split_events = 0;
    let mut merge_events = 0;

    for _ in 0..pair_budget {
        let Some(coarse_row) = coarse_rows
            .iter()
            .copied()
            .filter(|row| !consumed[*row])
            .max_by(|lhs, rhs| {
                detail[*lhs]
                    .total_cmp(&detail[*rhs])
                    .then_with(|| rhs.cmp(lhs))
            })
        else {
            break;
        };
        let Some(anchor_row) = fine_rows
            .iter()
            .copied()
            .filter(|row| !consumed[*row])
            .min_by(|lhs, rhs| {
                detail[*lhs]
                    .total_cmp(&detail[*rhs])
                    .then_with(|| lhs.cmp(rhs))
            })
        else {
            break;
        };
        let anchor = particles.positions[anchor_row];
        let mut scored_fine = fine_rows
            .iter()
            .copied()
            .filter(|row| !consumed[*row])
            .map(|row| {
                let dx = particles.positions[row][0] - anchor[0];
                let dy = particles.positions[row][1] - anchor[1];
                let score = (dx * dx + dy * dy) / fine_footprint_squared
                    + detail[row] * model.config.paired_topology_merge_detail_scale;
                (score, row)
            })
            .collect::<Vec<_>>();
        scored_fine.sort_by(|lhs, rhs| lhs.0.total_cmp(&rhs.0).then_with(|| lhs.1.cmp(&rhs.1)));
        if scored_fine.len() < 4 {
            break;
        }
        let merge_rows = scored_fine
            .iter()
            .take(4)
            .map(|(_, row)| *row)
            .collect::<Vec<_>>();
        let merge_detail =
            merge_rows.iter().map(|row| detail[*row]).sum::<f32>() / merge_rows.len() as f32;
        if !reallocation_gain_is_sufficient(
            detail[coarse_row],
            merge_detail,
            model.config.min_reallocation_relative_gain,
        ) {
            break;
        }

        let coarse_position = particles.positions[coarse_row];
        let coarse_state = particle_row(&particles.states, coarse_row, particles.state_dims);
        let coarse_jacobian = particle_row(
            &particles.state_jacobian,
            coarse_row,
            particles.state_dims * particles.spatial_dims,
        );
        let coarse_closure =
            optional_particle_row(&particles.closure_mode, coarse_row, particles.state_dims);
        let coarse_basis = optional_particle_row(&particles.closure_basis, coarse_row, 4);
        let coarse_phase = optional_particle_row(&particles.closure_phase, coarse_row, 2);
        let merged_state = mean_particle_rows(&particles.states, &merge_rows, particles.state_dims);
        let merged_jacobian = mean_particle_rows(
            &particles.state_jacobian,
            &merge_rows,
            particles.state_dims * particles.spatial_dims,
        );

        particles.positions[coarse_row] = mean_particle_positions(particles, &merge_rows);
        assign_particle_row(&mut particles.states, coarse_row, &merged_state);
        assign_particle_row(&mut particles.state_jacobian, coarse_row, &merged_jacobian);
        assign_optional_mean_row(
            &mut particles.closure_mode,
            coarse_row,
            &merge_rows,
            particles.state_dims,
        );
        assign_optional_mean_row(&mut particles.closure_basis, coarse_row, &merge_rows, 4);
        assign_optional_mean_row(&mut particles.closure_phase, coarse_row, &merge_rows, 2);

        let offsets = [
            [-split_radius, 0.0],
            [split_radius, 0.0],
            [0.0, -split_radius],
            [0.0, split_radius],
        ];
        for (row, offset) in merge_rows.iter().copied().zip(offsets) {
            particles.positions[row][0] = coarse_position[0] + offset[0];
            particles.positions[row][1] = coarse_position[1] + offset[1];
            particles.positions[row][2] = coarse_position[2];
            particles.positions[row][3] = coarse_position[3];
            assign_particle_row(&mut particles.states, row, &coarse_state);
            assign_particle_row(&mut particles.state_jacobian, row, &coarse_jacobian);
            assign_optional_particle_row(&mut particles.closure_mode, row, &coarse_closure);
            assign_optional_particle_row(&mut particles.closure_basis, row, &coarse_basis);
            assign_optional_particle_row(&mut particles.closure_phase, row, &coarse_phase);
            consumed[row] = true;
        }
        consumed[coarse_row] = true;
        split_events += 1;
        merge_events += 1;
    }

    Ok(TopologyDecision {
        split_events,
        merge_events,
        eligible_split_candidates: coarse_rows.len(),
        eligible_merge_clusters: fine_rows.len() / 4,
        min_desired_footprint_ratio: 1.0,
        max_desired_footprint_ratio: 1.0,
        ..TopologyDecision::default()
    })
}

fn particle_row(values: &[f32], row: usize, width: usize) -> Vec<f32> {
    values[row * width..(row + 1) * width].to_vec()
}

fn optional_particle_row(values: &[f32], row: usize, width: usize) -> Vec<f32> {
    if values.is_empty() {
        Vec::new()
    } else {
        particle_row(values, row, width)
    }
}

fn assign_particle_row(values: &mut [f32], row: usize, replacement: &[f32]) {
    let width = replacement.len();
    values[row * width..(row + 1) * width].copy_from_slice(replacement);
}

fn assign_optional_particle_row(values: &mut [f32], row: usize, replacement: &[f32]) {
    if !values.is_empty() {
        assign_particle_row(values, row, replacement);
    }
}

fn mean_particle_rows(values: &[f32], rows: &[usize], width: usize) -> Vec<f32> {
    let mut mean = vec![0.0; width];
    for row in rows {
        for (channel, value) in mean.iter_mut().enumerate() {
            *value += values[row * width + channel];
        }
    }
    let reciprocal = 1.0 / rows.len().max(1) as f32;
    mean.iter_mut().for_each(|value| *value *= reciprocal);
    mean
}

fn assign_optional_mean_row(values: &mut [f32], row: usize, source_rows: &[usize], width: usize) {
    if !values.is_empty() {
        let mean = mean_particle_rows(values, source_rows, width);
        assign_particle_row(values, row, &mean);
    }
}

fn mean_particle_positions(particles: &AdaptiveParticleSet, rows: &[usize]) -> [f32; 4] {
    let mut mean = [0.0; 4];
    for row in rows {
        for (axis, value) in mean.iter_mut().enumerate() {
            *value += particles.positions[*row][axis];
        }
    }
    let reciprocal = 1.0 / rows.len().max(1) as f32;
    mean.iter_mut().for_each(|value| *value *= reciprocal);
    mean
}

fn apply_topology(
    model: &AdaptiveNpaModel,
    particles: &mut AdaptiveParticleSet,
    controller: &[AdaptiveControllerOutput],
    state_gradient: Option<&[f32]>,
    step: usize,
) -> AutomataResult<TopologyDecision> {
    if controller.len() != particles.len() {
        return Err(AutomataError::InvalidArgument(
            "adaptive topology controller output shape mismatch".to_string(),
        ));
    }
    let proposed = controller
        .iter()
        .map(|output| {
            (model.config.reference_footprint * output.desired_log_footprint.exp())
                .clamp(model.config.min_footprint, model.config.max_footprint)
        })
        .collect::<Vec<_>>();
    let current = (0..particles.len())
        .map(|index| particles.footprint(index))
        .collect::<Vec<_>>();
    let initial_count = particles.len();
    let bootstrap_active = model
        .config
        .coarse_to_fine_bootstrap_active(step, initial_count);
    let active_target = if bootstrap_active {
        model.config.bootstrap_target_leaf_count()
    } else {
        model.config.target_leaves
    };
    let desired = normalize_footprint_budget_bounded(
        &proposed,
        &current,
        &particles.represented_measure,
        particles.spatial_dims,
        model.config.min_footprint,
        model.config.max_footprint,
        active_target,
        model.config.min_topology_footprint_ratio,
        model.config.max_topology_footprint_ratio,
    )?
    .desired_footprint;
    let (min_desired_footprint_ratio, max_desired_footprint_ratio) = desired
        .iter()
        .enumerate()
        .map(|(index, desired)| desired / particles.footprint(index))
        .fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(minimum, maximum), ratio| (minimum.min(ratio), maximum.max(ratio)),
        );
    let event_budget = model.config.topology_event_budget(step, initial_count);
    let mut consumed = vec![false; particles.len()];
    let mut merge_groups = BTreeMap::<u64, Vec<usize>>::new();
    for (index, group) in particles.sibling_group.iter().copied().enumerate() {
        if group != 0 {
            merge_groups.entry(group).or_default().push(index);
        }
    }
    let expected_siblings = 2 * particles.spatial_dims;
    if model.config.spatial_merge_groups_enabled {
        let hierarchy = AdaptiveProxyHierarchy::build(particles, expected_siblings)?;
        if let Some(level) = hierarchy.levels.first() {
            for (group_index, node_index) in level.iter().copied().enumerate() {
                let node = &hierarchy.nodes[node_index];
                let indices = node
                    .children
                    .iter()
                    .filter_map(|member| match member {
                        super::AdaptiveHierarchyMember::Leaf(index) => Some(*index),
                        super::AdaptiveHierarchyMember::Proxy(_) => None,
                    })
                    .collect::<Vec<_>>();
                if indices.len() == expected_siblings {
                    merge_groups
                        .entry((1_u64 << 63) | group_index as u64)
                        .or_insert(indices);
                }
            }
        }
    }
    let event_scale = (expected_siblings as f32).powf(1.0 / particles.spatial_dims as f32);
    let mut max_compatible_merge_probability = 0.0_f32;
    let mut merge_candidates = merge_groups
        .into_iter()
        .filter_map(|(group, indices)| {
            if indices.len() != expected_siblings
                || indices.iter().any(|index| particles.cooldown[*index] != 0)
                || merge_desired_ratio(particles, &desired, &indices) <= model.config.merge_ratio
                || controller_merge_probability(controller, &indices)
                    < model.config.merge_probability
            {
                return None;
            }
            let materials = indices
                .iter()
                .map(|index| material_at(particles, *index))
                .collect::<Vec<_>>();
            let merged = canonical_merge(&materials).ok()?;
            let merged_footprint = material_footprint_radius(
                merged.represented_measure as f32,
                particles.spatial_dims,
            );
            if merged_footprint > model.config.max_footprint {
                return None;
            }
            let total_measure = merged.represented_measure as f32;
            let merged_bandwidth = indices
                .iter()
                .map(|index| {
                    particles.represented_measure[*index] / total_measure
                        * particles.bandwidth[*index]
                })
                .sum::<f32>();
            if !merge_respects_scale_grading(
                &merged,
                merged_bandwidth,
                &indices,
                particles,
                model.config.max_neighbor_footprint_ratio,
                model.config.perception.pair_scale_power,
            ) {
                return None;
            }
            let state_rms = compatible_merge_cluster(model, particles, &indices)?;
            let confidence = controller_merge_probability(controller, &indices);
            max_compatible_merge_probability = max_compatible_merge_probability.max(confidence);
            Some(MergeCandidate {
                resolution_cost: merge_resolution_cost(particles, &desired, &indices),
                state_rms,
                confidence,
                group,
                indices,
            })
        })
        .collect::<Vec<_>>();
    let eligible_merge_clusters = merge_candidates.len();
    merge_candidates.sort_by(|lhs, rhs| {
        lhs.state_rms
            .total_cmp(&rhs.state_rms)
            .then_with(|| lhs.resolution_cost.total_cmp(&rhs.resolution_cost))
            .then_with(|| rhs.confidence.total_cmp(&lhs.confidence))
            .then_with(|| lhs.group.cmp(&rhs.group))
    });

    let mut split_candidates = Vec::new();
    for index in 0..particles.len() {
        if (!bootstrap_active
            && (particles.cooldown[index] != 0
                || desired[index] >= particles.footprint(index) * model.config.split_ratio
                || controller[index].split_probability < model.config.split_probability))
            || particles.footprint(index) / event_scale < model.config.min_footprint
        {
            continue;
        }
        let has_bootstrap_template = particles
            .bootstrap_templates
            .iter()
            .any(|template| template.parent_id == particles.particle_id[index]);
        let restores_bootstrap_template = bootstrap_active && has_bootstrap_template;
        let parent = material_at(particles, index);
        let plan = continuous_split_plan(
            index,
            &parent,
            particles,
            &desired,
            if restores_bootstrap_template {
                1.0
            } else {
                model.config.max_unequal_split_measure_ratio
            },
            model.config.split_field_neighbors,
        )?;
        let equal_fraction = 1.0 / plan.fractions.len() as f64;
        let effectively_equal = plan
            .fractions
            .iter()
            .all(|fraction| (*fraction - equal_fraction).abs() <= 1.0e-12);
        let children_fit = if restores_bootstrap_template {
            particle_split_fits_domain(
                particles,
                index,
                &model.config.domain_min,
                &model.config.domain_max,
            )
        } else {
            plan.children.iter().all(|child| {
                let footprint = material_footprint_radius(
                    child.represented_measure as f32,
                    particles.spatial_dims,
                );
                (model.config.min_footprint..=model.config.max_footprint).contains(&footprint)
                    && position_within_domain(
                        &position_from_material(child),
                        particles.spatial_dims,
                        &model.config.domain_min,
                        &model.config.domain_max,
                    )
            }) && split_respects_scale_grading(
                index,
                &plan.children,
                particles.bandwidth[index],
                particles,
                model.config.max_neighbor_footprint_ratio,
                model.config.perception.pair_scale_power,
            )
        };
        if !children_fit {
            continue;
        }
        let resolution_benefit = if restores_bootstrap_template || effectively_equal {
            split_resolution_benefit(
                particles.represented_measure[index],
                particles.footprint(index),
                desired[index],
                event_scale,
                particles.spatial_dims,
            )
        } else {
            continuous_split_resolution_benefit(
                particles.represented_measure[index],
                particles.footprint(index),
                desired[index],
                &plan.children,
                &plan.desired_footprints,
                particles.spatial_dims,
            )
        };
        split_candidates.push(SplitCandidate {
            resolution_benefit,
            confidence: controller[index].split_probability,
            particle_id: particles.particle_id[index],
            index,
            child_fractions: plan.fractions,
        });
    }
    let eligible_split_candidates = split_candidates.len();
    split_candidates.sort_by(|lhs, rhs| {
        rhs.resolution_benefit
            .total_cmp(&lhs.resolution_benefit)
            .then_with(|| rhs.confidence.total_cmp(&lhs.confidence))
            .then_with(|| lhs.particle_id.cmp(&rhs.particle_id))
    });

    let mut selected_merges = Vec::new();
    let mut selected_splits = Vec::new();
    let mut projected_count = initial_count;
    if initial_count == active_target {
        for candidate in merge_candidates {
            if selected_merges.len() + selected_splits.len() + 2 > event_budget {
                break;
            }
            if candidate.indices.iter().any(|index| consumed[*index]) {
                continue;
            }
            let Some(split) = split_candidates.iter().find(|split| {
                !consumed[split.index]
                    && !candidate.indices.contains(&split.index)
                    && reallocation_gain_is_sufficient(
                        split.resolution_benefit,
                        candidate.resolution_cost,
                        model.config.min_reallocation_relative_gain,
                    )
            }) else {
                continue;
            };
            candidate
                .indices
                .iter()
                .for_each(|index| consumed[*index] = true);
            consumed[split.index] = true;
            selected_merges.push(candidate.indices);
            selected_splits.push(SplitSelection {
                index: split.index,
                child_fractions: split.child_fractions.clone(),
            });
        }
    } else if initial_count > active_target {
        for candidate in merge_candidates {
            let merged_count = projected_count.saturating_sub(candidate.indices.len() - 1);
            if selected_merges.len() >= event_budget || merged_count < model.config.min_leaves {
                break;
            }
            if candidate.indices.iter().any(|index| consumed[*index])
                || merged_count.abs_diff(active_target) >= projected_count.abs_diff(active_target)
            {
                continue;
            }
            candidate
                .indices
                .iter()
                .for_each(|index| consumed[*index] = true);
            projected_count = merged_count;
            selected_merges.push(candidate.indices);
        }
    } else {
        for split in split_candidates {
            if selected_splits.len() >= event_budget
                || projected_count + expected_siblings - 1 > model.config.max_leaves
            {
                break;
            }
            let split_count = projected_count + expected_siblings - 1;
            if split_count.abs_diff(active_target) >= projected_count.abs_diff(active_target) {
                break;
            }
            projected_count = split_count;
            selected_splits.push(SplitSelection {
                index: split.index,
                child_fractions: split.child_fractions,
            });
        }
    }

    let transfer = rebuild_particles(
        particles,
        &selected_merges,
        &selected_splits,
        RebuildOptions {
            state_gradient,
            log_normalized_gradient: model.config.perception.log_normalize_gradients,
            state_prolongation_scale: model.config.split_state_prolongation_scale,
            max_state_transfer_rms: model.config.split_state_transfer_rms_limit(),
            restore_bootstrap_children: bootstrap_active,
            bootstrap_templates_are_current: model.config.coarse_dynamics
                == AdaptiveCoarseDynamics::PersistentFineQuadrature,
            bootstrap_seed_spread: if bootstrap_active {
                model.config.bootstrap_seed_spread
            } else {
                0.0
            },
            cooldown: model.config.cooldown_steps,
            domain_min: &model.config.domain_min,
            domain_max: &model.config.domain_max,
        },
    )?;
    Ok(TopologyDecision {
        split_events: selected_splits.len(),
        merge_events: selected_merges.len(),
        max_compatible_merge_probability,
        eligible_split_candidates,
        eligible_merge_clusters,
        min_desired_footprint_ratio,
        max_desired_footprint_ratio,
        state_transfer_rms_sum: transfer.rms_sum,
        state_transfer_events: transfer.events,
        max_event_state_transfer_rms: transfer.max_rms,
    })
}

fn reallocation_gain_is_sufficient(
    split_benefit: f32,
    merge_cost: f32,
    relative_margin: f32,
) -> bool {
    if relative_margin >= 1.0 {
        return false;
    }
    let comparison_scale = split_benefit
        .abs()
        .max(merge_cost.abs())
        .max(f32::MIN_POSITIVE);
    split_benefit > merge_cost + relative_margin * comparison_scale
}

fn controller_merge_probability(controller: &[AdaptiveControllerOutput], indices: &[usize]) -> f32 {
    indices
        .iter()
        .map(|index| controller[*index].merge_probability)
        .sum::<f32>()
        / indices.len().max(1) as f32
}

fn merge_desired_ratio(particles: &AdaptiveParticleSet, desired: &[f32], indices: &[usize]) -> f32 {
    let total_measure = indices
        .iter()
        .map(|index| particles.represented_measure[*index])
        .sum::<f32>()
        .max(f32::MIN_POSITIVE);
    indices
        .iter()
        .map(|index| {
            particles.represented_measure[*index] / total_measure
                * (desired[*index] / particles.footprint(*index)).ln()
        })
        .sum::<f32>()
        .exp()
}

fn split_resolution_benefit(
    measure: f32,
    current: f32,
    desired: f32,
    event_scale: f32,
    dim: usize,
) -> f32 {
    discrete_resolution_cost(measure, current, desired, dim)
        - discrete_resolution_cost(measure, current / event_scale, desired, dim)
}

fn continuous_split_resolution_benefit(
    parent_measure: f32,
    current_footprint: f32,
    parent_desired_footprint: f32,
    children: &[CanonicalMaterial],
    child_desired_footprint: &[f32],
    dim: usize,
) -> f32 {
    let before = discrete_resolution_cost(
        parent_measure,
        current_footprint,
        parent_desired_footprint,
        dim,
    );
    let after = children
        .iter()
        .zip(child_desired_footprint)
        .map(|(child, desired)| {
            let measure = child.represented_measure as f32;
            discrete_resolution_cost(
                measure,
                material_footprint_radius(measure, dim),
                *desired,
                dim,
            )
        })
        .sum::<f32>();
    before - after
}

fn merge_resolution_cost(
    particles: &AdaptiveParticleSet,
    desired: &[f32],
    indices: &[usize],
) -> f32 {
    let total_measure = indices
        .iter()
        .map(|index| particles.represented_measure[*index])
        .sum::<f32>()
        .max(f32::MIN_POSITIVE);
    let before = indices
        .iter()
        .map(|index| {
            discrete_resolution_cost(
                particles.represented_measure[*index],
                particles.footprint(*index),
                desired[*index],
                particles.spatial_dims,
            )
        })
        .sum::<f32>();
    let merged_footprint = material_footprint_radius(total_measure, particles.spatial_dims);
    let after = indices
        .iter()
        .map(|index| {
            discrete_resolution_cost(
                particles.represented_measure[*index],
                merged_footprint,
                desired[*index],
                particles.spatial_dims,
            )
        })
        .sum::<f32>();
    after - before
}

fn discrete_resolution_cost(measure: f32, footprint: f32, desired: f32, dim: usize) -> f32 {
    const ERROR_EXPONENT: i32 = 2;
    measure * footprint.powi(ERROR_EXPONENT)
        / desired
            .max(f32::MIN_POSITIVE)
            .powi(ERROR_EXPONENT + dim as i32)
}

fn compatible_merge_cluster(
    model: &AdaptiveNpaModel,
    particles: &AdaptiveParticleSet,
    indices: &[usize],
) -> Option<f32> {
    let materials = indices
        .iter()
        .map(|index| material_at(particles, *index))
        .collect::<Vec<_>>();
    let Ok(merged) = canonical_merge(&materials) else {
        return None;
    };
    let merged_position = position_from_material(&merged);
    let footprint =
        material_footprint_radius(merged.represented_measure as f32, particles.spatial_dims);
    let maximum_extent = indices
        .iter()
        .map(|index| {
            let distance = (0..particles.spatial_dims)
                .map(|axis| (particles.positions[*index][axis] - merged_position[axis]).powi(2))
                .sum::<f32>()
                .sqrt();
            distance + particles.footprint(*index)
        })
        .fold(0.0_f32, f32::max);
    if maximum_extent > model.config.merge_extent_ratio * footprint {
        return None;
    }
    let total_measure = indices
        .iter()
        .map(|index| particles.represented_measure[*index])
        .sum::<f32>()
        .max(f32::MIN_POSITIVE);
    let mut mean = vec![0.0; particles.state_dims];
    for index in indices {
        let weight = particles.represented_measure[*index] / total_measure;
        let base = *index * particles.state_dims;
        for (channel, value) in mean.iter_mut().enumerate() {
            *value += weight * particles.states[base + channel];
        }
    }
    let mut variance = 0.0;
    for index in indices {
        let weight = particles.represented_measure[*index] / total_measure;
        let base = *index * particles.state_dims;
        for (channel, mean_value) in mean.iter().enumerate() {
            variance += weight * (particles.states[base + channel] - mean_value).powi(2);
        }
    }
    let state_rms = (variance / particles.state_dims as f32).sqrt();
    (state_rms <= model.config.merge_state_rms_limit).then_some(state_rms)
}

fn refresh_state_jacobian(
    particles: &mut AdaptiveParticleSet,
    encoded: &[f32],
    log_normalized: bool,
    native_footprint: f32,
) -> AutomataResult<()> {
    let row_dims = particles.state_dims * particles.spatial_dims;
    if encoded.len() != particles.len() * row_dims {
        return Err(AutomataError::InvalidArgument(
            "adaptive perception state Jacobian shape mismatch".to_string(),
        ));
    }
    for (row, encoded_row) in encoded.chunks_exact(row_dims).enumerate() {
        if particles.footprint(row) <= native_footprint * 1.5 {
            let decoded = super::perception::decode_physical_state_gradient(
                encoded_row,
                particles.state_dims,
                particles.spatial_dims,
                particles.bandwidth[row],
                log_normalized,
            );
            particles.state_jacobian[row * row_dims..(row + 1) * row_dims]
                .copy_from_slice(&decoded);
        }
    }
    Ok(())
}

fn fit_merged_state_jacobian(
    particles: &AdaptiveParticleSet,
    indices: &[usize],
    mean_state: &[f32],
    merged: &CanonicalMaterial,
) -> AutomataResult<Vec<f32>> {
    super::state::fit_state_jacobian(
        particles,
        indices,
        mean_state,
        position_from_material(merged),
        covariance_from_material(merged),
        merged.represented_measure as f32,
    )
}

#[derive(Clone)]
struct ParticleRecord {
    position: [f32; 4],
    state: Vec<f32>,
    state_jacobian: Vec<f32>,
    closure_mode: Vec<f32>,
    closure_basis: Vec<f32>,
    closure_phase: Vec<f32>,
    measure: f32,
    render_footprint: f32,
    bandwidth: f32,
    covariance: [f32; 9],
    id: u64,
    sibling_group: u64,
    generation: u16,
    cooldown: u16,
}

#[derive(Clone, Copy)]
struct RebuildOptions<'a> {
    state_gradient: Option<&'a [f32]>,
    log_normalized_gradient: bool,
    state_prolongation_scale: f32,
    max_state_transfer_rms: f32,
    restore_bootstrap_children: bool,
    bootstrap_templates_are_current: bool,
    bootstrap_seed_spread: f32,
    cooldown: u16,
    domain_min: &'a [f32; 3],
    domain_max: &'a [f32; 3],
}

fn rebuild_particles(
    particles: &mut AdaptiveParticleSet,
    merges: &[Vec<usize>],
    splits: &[SplitSelection],
    options: RebuildOptions<'_>,
) -> AutomataResult<EventTransferMetrics> {
    if merges.is_empty() && splits.is_empty() {
        return Ok(EventTransferMetrics::default());
    }
    let split_set = splits
        .iter()
        .map(|split| split.index)
        .collect::<std::collections::BTreeSet<_>>();
    let mut merge_owner = BTreeMap::new();
    for (merge_index, indices) in merges.iter().enumerate() {
        for index in indices {
            merge_owner.insert(*index, merge_index);
        }
    }
    let consumed_template_ids = split_set
        .iter()
        .chain(merge_owner.keys())
        .map(|index| particles.particle_id[*index])
        .collect::<std::collections::BTreeSet<_>>();
    let mut records = Vec::new();
    let mut transfer = EventTransferMetrics::default();
    let mut recycled_ids = Vec::new();
    let bootstrap_templates = particles
        .bootstrap_templates
        .iter()
        .map(|template| (template.parent_id, template.children.clone()))
        .collect::<BTreeMap<_, _>>();
    for index in 0..particles.len() {
        if merge_owner.contains_key(&index) || split_set.contains(&index) {
            continue;
        }
        records.push(record_at(particles, index));
    }
    for indices in merges {
        let materials = indices
            .iter()
            .map(|index| material_at(particles, *index))
            .collect::<Vec<_>>();
        let merged = canonical_merge(&materials)?;
        let total_measure = merged.represented_measure as f32;
        let mut state = vec![0.0; particles.state_dims];
        let mut bandwidth = 0.0;
        let mut log_render_footprint = 0.0;
        let mut generation = 0;
        for index in indices {
            let weight = particles.represented_measure[*index] / total_measure;
            let base = index * particles.state_dims;
            for (channel, value) in state.iter_mut().enumerate() {
                *value += weight * particles.states[base + channel];
            }
            bandwidth += weight * particles.bandwidth[*index];
            log_render_footprint += weight
                * particles.render_footprint[*index]
                    .max(f32::MIN_POSITIVE)
                    .ln();
            generation = generation.max(particles.generation[*index]);
        }
        let state_rms = merge_state_rms(particles, indices, &state, total_measure);
        let state_jacobian = fit_merged_state_jacobian(particles, indices, &state, &merged)?;
        transfer.record(state_rms);
        let mut lineage_ids = indices
            .iter()
            .map(|index| particles.particle_id[*index])
            .collect::<Vec<_>>();
        lineage_ids.sort_unstable();
        let id = lineage_ids[0];
        recycled_ids.extend_from_slice(&lineage_ids[1..]);
        records.push(ParticleRecord {
            position: position_from_material(&merged),
            state,
            state_jacobian,
            closure_mode: vec![0.0; particles.state_dims],
            closure_basis: vec![0.0; 4],
            closure_phase: vec![0.0; 2],
            measure: total_measure,
            render_footprint: log_render_footprint.exp(),
            bandwidth,
            covariance: covariance_from_material(&merged),
            id,
            sibling_group: 0,
            generation: generation.saturating_sub(1),
            cooldown: options.cooldown,
        });
    }
    recycled_ids.sort_unstable_by(|lhs, rhs| rhs.cmp(lhs));
    for split in splits {
        let index = split.index;
        if options.restore_bootstrap_children
            && let Some(children) = bootstrap_templates.get(&particles.particle_id[index])
        {
            append_bootstrap_children(
                particles,
                index,
                children,
                options,
                &mut records,
                &mut transfer,
            )?;
            continue;
        }
        let parent = material_at(particles, index);
        let mut children = constrained_unequal_split(&parent, &split.child_fractions)?;
        spread_bootstrap_children(
            &parent,
            &mut children,
            options.bootstrap_seed_spread,
            particles.particle_id[index],
        );
        let group = particles.next_sibling_group;
        particles.next_sibling_group += 1;
        let state_base = index * particles.state_dims;
        let parent_state = particles.states[state_base..state_base + particles.state_dims].to_vec();
        let jacobian_dims = particles.state_dims * particles.spatial_dims;
        let parent_state_jacobian =
            particles.state_jacobian[index * jacobian_dims..(index + 1) * jacobian_dims].to_vec();
        let child_states = prolonged_child_states(
            particles,
            index,
            &children,
            options.state_gradient,
            options.log_normalized_gradient,
            options.state_prolongation_scale,
            options.max_state_transfer_rms,
        );
        let split_rms = child_states
            .iter()
            .map(|state| state_rms_difference(state, &parent_state))
            .fold(0.0_f32, f32::max);
        transfer.record(split_rms);
        for (child_index, (child, state)) in children.into_iter().zip(child_states).enumerate() {
            let position = position_from_material(&child);
            if !position_within_domain(
                &position,
                particles.spatial_dims,
                options.domain_min,
                options.domain_max,
            ) {
                return Err(AutomataError::InvalidArgument(
                    "adaptive split crossed the configured domain after event selection"
                        .to_string(),
                ));
            }
            let id = if child_index == 0 {
                particles.particle_id[index]
            } else if let Some(id) = recycled_ids.pop() {
                id
            } else {
                let id = particles.next_id;
                particles.next_id += 1;
                id
            };
            records.push(ParticleRecord {
                position,
                state,
                state_jacobian: parent_state_jacobian.clone(),
                closure_mode: vec![0.0; particles.state_dims],
                closure_basis: vec![0.0; 4],
                closure_phase: vec![0.0; 2],
                measure: child.represented_measure as f32,
                render_footprint: particles.render_footprint[index],
                bandwidth: particles.bandwidth[index],
                covariance: covariance_from_material(&child),
                id,
                sibling_group: group,
                generation: particles.generation[index].saturating_add(1),
                cooldown: options.cooldown,
            });
        }
    }
    records.sort_unstable_by_key(|record| record.id);
    assign_records(particles, records);
    let active_ids = particles
        .particle_id
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    particles.bootstrap_templates.retain(|template| {
        active_ids.contains(&template.parent_id)
            && !consumed_template_ids.contains(&template.parent_id)
    });
    Ok(transfer)
}

fn append_bootstrap_children(
    particles: &mut AdaptiveParticleSet,
    parent_index: usize,
    children: &[AdaptiveBootstrapChild],
    options: RebuildOptions<'_>,
    records: &mut Vec<ParticleRecord>,
    transfer: &mut EventTransferMetrics,
) -> AutomataResult<()> {
    let group = particles.next_sibling_group;
    particles.next_sibling_group += 1;
    let state_base = parent_index * particles.state_dims;
    let parent_state = &particles.states[state_base..state_base + particles.state_dims];
    let total_child_measure = children
        .iter()
        .map(|child| child.represented_measure)
        .sum::<f32>()
        .max(f32::MIN_POSITIVE);
    let mut template_centroid = [0.0_f32; 4];
    for child in children {
        let weight = child.represented_measure / total_child_measure;
        for (axis, value) in template_centroid
            .iter_mut()
            .enumerate()
            .take(particles.spatial_dims)
        {
            *value += weight * child.position[axis];
        }
    }
    let child_positions = children
        .iter()
        .map(|child| {
            let mut position = child.position;
            for axis in 0..particles.spatial_dims {
                position[axis] += particles.positions[parent_index][axis] - template_centroid[axis];
            }
            position
        })
        .collect::<Vec<_>>();
    let child_material = children
        .iter()
        .zip(&child_positions)
        .map(|(child, position)| CanonicalMaterial {
            represented_measure: child.represented_measure as f64,
            position: position[..particles.spatial_dims]
                .iter()
                .map(|value| *value as f64)
                .collect(),
            covariance: (0..particles.spatial_dims)
                .flat_map(|row| {
                    (0..particles.spatial_dims)
                        .map(move |col| child.covariance[row * 3 + col] as f64)
                })
                .collect(),
            extensive: Vec::new(),
        })
        .collect::<Vec<_>>();
    // Template offsets identify the original fine material, but absolute
    // positions and latent state belong to seed time. A delayed LoD split must
    // follow the current coarse centroid and prolong its current trajectory or
    // it teleports cold child rows back into the seed cloud.
    let child_states = if options.bootstrap_templates_are_current {
        let mut states = children
            .iter()
            .map(|child| child.state.clone())
            .collect::<Vec<_>>();
        // Persistent quadrature advances template states in place. Remove any
        // accumulated float error so exposing those modes still restricts to
        // the current visible parent exactly.
        for channel in 0..particles.state_dims {
            let mean = states
                .iter()
                .zip(children)
                .map(|(state, child)| {
                    child.represented_measure / total_child_measure * state[channel]
                })
                .sum::<f32>();
            let correction = parent_state[channel] - mean;
            for state in &mut states {
                state[channel] += correction;
            }
        }
        states
    } else {
        prolonged_child_states(
            particles,
            parent_index,
            &child_material,
            options.state_gradient,
            options.log_normalized_gradient,
            options.state_prolongation_scale,
            options.max_state_transfer_rms,
        )
    };
    let split_rms = child_states
        .iter()
        .map(|state| state_rms_difference(state, parent_state))
        .fold(0.0_f32, f32::max);
    transfer.record(split_rms);
    for ((child, position), state) in children.iter().zip(child_positions).zip(child_states) {
        if !position_within_domain(
            &position,
            particles.spatial_dims,
            options.domain_min,
            options.domain_max,
        ) {
            return Err(AutomataError::InvalidArgument(
                "adaptive bootstrap child lies outside the configured domain".to_string(),
            ));
        }
        records.push(ParticleRecord {
            position,
            state,
            state_jacobian: particles.state_jacobian[parent_index
                * particles.state_dims
                * particles.spatial_dims
                ..(parent_index + 1) * particles.state_dims * particles.spatial_dims]
                .to_vec(),
            closure_mode: vec![0.0; particles.state_dims],
            closure_basis: vec![0.0; 4],
            closure_phase: vec![0.0; 2],
            measure: child.represented_measure,
            render_footprint: particles.render_footprint[parent_index],
            bandwidth: child.bandwidth,
            covariance: child.covariance,
            id: child.particle_id,
            sibling_group: group,
            generation: child.generation,
            cooldown: options.cooldown,
        });
    }
    Ok(())
}

fn spread_bootstrap_children(
    parent: &CanonicalMaterial,
    children: &mut [CanonicalMaterial],
    spread: f32,
    particle_id: u64,
) {
    if spread <= 0.0 || children.len() != 2 * parent.position.len() {
        return;
    }
    let angle =
        stable_particle_uniform(0x5eed_5eed_d15c_a11c, 0, particle_id) * std::f32::consts::TAU;
    let (sin, cos) = angle.sin_cos();
    for (child_index, child) in children.iter_mut().enumerate() {
        let axis = child_index / 2;
        let sign = if child_index.is_multiple_of(2) {
            -1.0_f64
        } else {
            1.0_f64
        };
        child.position.copy_from_slice(&parent.position);
        match axis {
            0 => {
                child.position[0] += sign * f64::from(spread * cos);
                child.position[1] += sign * f64::from(spread * sin);
            }
            1 => {
                child.position[0] -= sign * f64::from(spread * sin);
                child.position[1] += sign * f64::from(spread * cos);
            }
            2 => child.position[2] += sign * f64::from(spread),
            _ => unreachable!("adaptive material dimensions are validated"),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct EventTransferMetrics {
    rms_sum: f32,
    events: usize,
    max_rms: f32,
}

impl EventTransferMetrics {
    fn record(&mut self, rms: f32) {
        self.rms_sum += rms;
        self.events += 1;
        self.max_rms = self.max_rms.max(rms);
    }
}

fn merge_state_rms(
    particles: &AdaptiveParticleSet,
    indices: &[usize],
    merged_state: &[f32],
    total_measure: f32,
) -> f32 {
    let variance = indices
        .iter()
        .map(|index| {
            let weight =
                particles.represented_measure[*index] / total_measure.max(f32::MIN_POSITIVE);
            let base = *index * particles.state_dims;
            weight
                * particles.states[base..base + particles.state_dims]
                    .iter()
                    .zip(merged_state)
                    .map(|(value, mean)| (*value - *mean).powi(2))
                    .sum::<f32>()
        })
        .sum::<f32>();
    (variance / particles.state_dims.max(1) as f32).sqrt()
}

fn prolonged_child_states(
    particles: &AdaptiveParticleSet,
    parent_index: usize,
    children: &[CanonicalMaterial],
    state_gradient: Option<&[f32]>,
    log_normalized_gradient: bool,
    state_prolongation_scale: f32,
    max_state_transfer_rms: f32,
) -> Vec<Vec<f32>> {
    let state_base = parent_index * particles.state_dims;
    let parent_state = &particles.states[state_base..state_base + particles.state_dims];
    let Some(state_gradient) = state_gradient else {
        return vec![parent_state.to_vec(); children.len()];
    };
    let gradient_dims = particles.state_dims * particles.spatial_dims;
    if state_gradient.len() != particles.len() * gradient_dims {
        return vec![parent_state.to_vec(); children.len()];
    }
    let gradient =
        &state_gradient[parent_index * gradient_dims..(parent_index + 1) * gradient_dims];
    let physical_gradient = super::perception::decode_physical_state_gradient(
        gradient,
        particles.state_dims,
        particles.spatial_dims,
        particles.bandwidth[parent_index],
        log_normalized_gradient,
    );
    let mut child_states = children
        .iter()
        .map(|child| {
            (0..particles.state_dims)
                .map(|channel| {
                    let delta = (0..particles.spatial_dims)
                        .map(|axis| {
                            physical_gradient[channel * particles.spatial_dims + axis]
                                * (child.position[axis] as f32
                                    - particles.positions[parent_index][axis])
                        })
                        .sum::<f32>();
                    parent_state[channel] + state_prolongation_scale * delta
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let maximum_rms = child_states
        .iter()
        .map(|state| state_rms_difference(state, parent_state))
        .fold(0.0_f32, f32::max);
    if maximum_rms > max_state_transfer_rms {
        let scale = max_state_transfer_rms / maximum_rms.max(f32::MIN_POSITIVE);
        for state in &mut child_states {
            for (value, parent) in state.iter_mut().zip(parent_state) {
                *value = *parent + (*value - *parent) * scale;
            }
        }
    }
    // Float32 event construction can leave a tiny nonzero affine mean. Remove
    // its measure-weighted value so restriction recovers every intensive
    // parent channel exactly for equal and unequal child measures.
    let total_measure = children
        .iter()
        .map(|child| child.represented_measure as f32)
        .sum::<f32>()
        .max(f32::MIN_POSITIVE);
    for channel in 0..particles.state_dims {
        let mean = child_states
            .iter()
            .zip(children)
            .map(|(state, child)| child.represented_measure as f32 / total_measure * state[channel])
            .sum::<f32>();
        let correction = parent_state[channel] - mean;
        for state in &mut child_states {
            state[channel] += correction;
        }
    }
    child_states
}

fn state_rms_difference(lhs: &[f32], rhs: &[f32]) -> f32 {
    (lhs.iter()
        .zip(rhs)
        .map(|(lhs, rhs)| (*lhs - *rhs).powi(2))
        .sum::<f32>()
        / lhs.len().max(1) as f32)
        .sqrt()
}

fn split_fits_domain(
    parent: &CanonicalMaterial,
    domain_min: &[f32; 3],
    domain_max: &[f32; 3],
) -> bool {
    canonical_split(parent).is_ok_and(|children| {
        children.iter().all(|child| {
            position_within_domain(
                &position_from_material(child),
                parent.position.len(),
                domain_min,
                domain_max,
            )
        })
    })
}

fn particle_split_fits_domain(
    particles: &AdaptiveParticleSet,
    index: usize,
    domain_min: &[f32; 3],
    domain_max: &[f32; 3],
) -> bool {
    particles
        .bootstrap_templates
        .iter()
        .find(|template| template.parent_id == particles.particle_id[index])
        .map_or_else(
            || split_fits_domain(&material_at(particles, index), domain_min, domain_max),
            |template| {
                template.children.iter().all(|child| {
                    position_within_domain(
                        &child.position,
                        particles.spatial_dims,
                        domain_min,
                        domain_max,
                    )
                })
            },
        )
}

fn position_within_domain(
    position: &[f32; 4],
    spatial_dims: usize,
    domain_min: &[f32; 3],
    domain_max: &[f32; 3],
) -> bool {
    (0..spatial_dims).all(|axis| (domain_min[axis]..=domain_max[axis]).contains(&position[axis]))
}

fn record_at(particles: &AdaptiveParticleSet, index: usize) -> ParticleRecord {
    let state_base = index * particles.state_dims;
    ParticleRecord {
        position: particles.positions[index],
        state: particles.states[state_base..state_base + particles.state_dims].to_vec(),
        state_jacobian: particles.state_jacobian[index
            * particles.state_dims
            * particles.spatial_dims
            ..(index + 1) * particles.state_dims * particles.spatial_dims]
            .to_vec(),
        closure_mode: if particles.closure_mode.is_empty() {
            vec![0.0; particles.state_dims]
        } else {
            particles.closure_mode[index * particles.state_dims..(index + 1) * particles.state_dims]
                .to_vec()
        },
        closure_basis: if particles.closure_basis.is_empty() {
            vec![0.0; 4]
        } else {
            particles.closure_basis[index * 4..(index + 1) * 4].to_vec()
        },
        closure_phase: if particles.closure_phase.is_empty() {
            vec![0.0; 2]
        } else {
            particles.closure_phase[index * 2..(index + 1) * 2].to_vec()
        },
        measure: particles.represented_measure[index],
        render_footprint: particles.render_footprint[index],
        bandwidth: particles.bandwidth[index],
        covariance: particles.covariance[index],
        id: particles.particle_id[index],
        sibling_group: particles.sibling_group[index],
        generation: particles.generation[index],
        cooldown: particles.cooldown[index],
    }
}

fn material_at(particles: &AdaptiveParticleSet, index: usize) -> CanonicalMaterial {
    let dim = particles.spatial_dims;
    CanonicalMaterial {
        represented_measure: particles.represented_measure[index] as f64,
        position: particles.positions[index][..dim]
            .iter()
            .map(|value| *value as f64)
            .collect(),
        covariance: (0..dim)
            .flat_map(|row| {
                (0..dim).map(move |col| particles.covariance[index][row * 3 + col] as f64)
            })
            .collect(),
        extensive: Vec::new(),
    }
}

fn position_from_material(material: &CanonicalMaterial) -> [f32; 4] {
    let mut position = [0.0; 4];
    for (axis, value) in material.position.iter().enumerate() {
        position[axis] = *value as f32;
    }
    position
}

fn covariance_from_material(material: &CanonicalMaterial) -> [f32; 9] {
    let dim = material.position.len();
    let mut covariance = [0.0; 9];
    for row in 0..dim {
        for col in 0..dim {
            covariance[row * 3 + col] = material.covariance[row * dim + col] as f32;
        }
    }
    covariance
}

fn assign_records(particles: &mut AdaptiveParticleSet, records: Vec<ParticleRecord>) {
    particles.positions.clear();
    particles.states.clear();
    particles.state_jacobian.clear();
    particles.closure_mode.clear();
    particles.closure_basis.clear();
    particles.closure_phase.clear();
    particles.represented_measure.clear();
    particles.render_footprint.clear();
    particles.bandwidth.clear();
    particles.covariance.clear();
    particles.particle_id.clear();
    particles.sibling_group.clear();
    particles.generation.clear();
    particles.cooldown.clear();
    for record in records {
        particles.positions.push(record.position);
        particles.states.extend(record.state);
        particles.state_jacobian.extend(record.state_jacobian);
        particles.closure_mode.extend(record.closure_mode);
        particles.closure_basis.extend(record.closure_basis);
        particles.closure_phase.extend(record.closure_phase);
        particles.represented_measure.push(record.measure);
        particles.render_footprint.push(record.render_footprint);
        particles.bandwidth.push(record.bandwidth);
        particles.covariance.push(record.covariance);
        particles.particle_id.push(record.id);
        particles.sibling_group.push(record.sibling_group);
        particles.generation.push(record.generation);
        particles.cooldown.push(record.cooldown);
    }
}

fn lerp(current: f32, target: f32, amount: f32) -> f32 {
    current + amount * (target - current)
}

fn relax_render_footprints(
    particles: &mut AdaptiveParticleSet,
    amount: f32,
    config: &super::AdaptiveNpaConfig,
) {
    let dim = particles.spatial_dims;
    for (displayed, measure) in particles
        .render_footprint
        .iter_mut()
        .zip(&particles.represented_measure)
    {
        let current = displayed.max(f32::MIN_POSITIVE).ln();
        let target = config
            .render_footprint(material_footprint_radius(*measure, dim))
            .max(f32::MIN_POSITIVE)
            .ln();
        *displayed = (current + amount * (target - current)).exp();
    }
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len().max(1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NpaConfig, NpaModel, ParticleSeed};

    #[test]
    fn reallocation_margin_rejects_near_equal_cost_bifurcations() {
        assert!(reallocation_gain_is_sufficient(1.01, 1.0, 0.0));
        assert!(!reallocation_gain_is_sufficient(1.01, 1.0, 0.05));
        assert!(reallocation_gain_is_sufficient(1.10, 1.0, 0.05));
        assert!(!reallocation_gain_is_sufficient(1.0e6, -1.0, 1.0));
    }

    #[test]
    fn hierarchical_bootstrap_refinement_is_nested_and_conservative() {
        let mut config = super::super::AdaptiveNpaConfig::growing_2d();
        config.min_leaves = 16;
        config.target_leaves = 64;
        config.bootstrap_target_leaves = 64;
        config.max_leaves = 64;
        config.initial_leaves = 16;
        config.bootstrap_fine_leaves = 64;
        config.topology_interval = 1;
        config.topology_start_step = 1;
        config.bootstrap_end_step = 4;
        config.bootstrap_events_per_interval = 4;
        let model =
            AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 7), config, 11)
                .unwrap();
        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let mut particles = super::super::seed_adaptive_particles_scaled(
            &model,
            16,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            0.1,
        )
        .unwrap();
        let initial_measure = particles.total_measure();

        for (step, expected) in [(1, 28), (2, 40), (3, 52), (4, 64)] {
            let previous_templates = particles
                .bootstrap_templates
                .iter()
                .map(|template| template.parent_id)
                .collect::<std::collections::BTreeSet<_>>();
            let update =
                apply_hierarchical_bootstrap_refinement(&model, &mut particles, step, 1).unwrap();
            assert_eq!(update.initial_leaf_count, expected - 12);
            assert_eq!(update.final_leaf_count, expected);
            assert_eq!(update.split_events, 4);
            assert_eq!(particles.len(), expected);
            assert!((particles.total_measure() - initial_measure).abs() <= 1.0e-8);
            assert!(
                particles
                    .bootstrap_templates
                    .iter()
                    .all(|template| { previous_templates.contains(&template.parent_id) })
            );
        }
        assert!(particles.bootstrap_templates.is_empty());
    }

    #[test]
    fn resident_canonical_bootstrap_builds_a_true_mixed_scale_cut() {
        let mut config = super::super::AdaptiveNpaConfig::growing_2d();
        config.min_leaves = 16;
        config.target_leaves = 40;
        config.bootstrap_target_leaves = 40;
        config.max_leaves = 64;
        config.initial_leaves = 16;
        config.bootstrap_fine_leaves = 64;
        config.topology_interval = 1;
        config.topology_start_step = 1;
        config.bootstrap_end_step = 2;
        config.bootstrap_events_per_interval = 4;
        config.retain_bootstrap_templates = false;
        config.runtime_topology_control = AdaptiveTopologyControl::PairedLocalDetail;
        config.material_seed_bandwidth_exponent = 0.5;
        let model =
            AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 7), config, 11)
                .unwrap();
        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let mut particles = super::super::seed_adaptive_particles_scaled(
            &model,
            16,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            0.1,
        )
        .unwrap();
        assert!(particles.bootstrap_templates.is_empty());
        let initial_measure = particles.total_measure();

        let first =
            apply_resident_canonical_bootstrap_refinement(&model, &mut particles, 1, 1).unwrap();
        assert_eq!(first.final_leaf_count, 28);
        let second =
            apply_resident_canonical_bootstrap_refinement(&model, &mut particles, 2, 1).unwrap();
        assert_eq!(second.final_leaf_count, 40);
        assert!((particles.total_measure() - initial_measure).abs() <= 1.0e-8);

        let min_measure = particles
            .represented_measure
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let max_measure = particles
            .represented_measure
            .iter()
            .copied()
            .fold(0.0_f32, f32::max);
        assert!((max_measure / min_measure - 4.0).abs() <= 1.0e-5);
        let min_bandwidth = particles
            .bandwidth
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let max_bandwidth = particles.bandwidth.iter().copied().fold(0.0_f32, f32::max);
        assert!((max_bandwidth / min_bandwidth - 2.0).abs() <= 1.0e-5);
    }

    #[test]
    fn delayed_bootstrap_prolongs_current_parent_state() {
        let mut config = crate::adaptive::AdaptiveNpaConfig::growing_2d();
        config.min_leaves = 16;
        config.initial_leaves = 16;
        config.target_leaves = 19;
        config.max_leaves = 64;
        config.bootstrap_fine_leaves = 64;
        config.bootstrap_end_step = 16;
        config.bootstrap_events_per_interval = 1;
        config.max_events_per_interval = 2;
        let model =
            AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 7), config, 11)
                .unwrap();
        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let mut particles = super::super::seed_adaptive_particles_scaled(
            &model,
            16,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            0.1,
        )
        .unwrap();
        let parent = 0;
        let parent_id = particles.particle_id[parent];
        let template = particles
            .bootstrap_templates
            .iter()
            .find(|template| template.parent_id == parent_id)
            .unwrap()
            .clone();
        let expected = (0..particles.state_dims)
            .map(|channel| 0.25 + channel as f32 * 0.01)
            .collect::<Vec<_>>();
        particles.states[parent * particles.state_dims..(parent + 1) * particles.state_dims]
            .copy_from_slice(&expected);
        particles.positions[parent][0] += 0.3;
        particles.positions[parent][1] -= 0.2;
        let expected_position = particles.positions[parent];
        let mut gradient = vec![0.0; particles.len() * particles.state_dims * 2];
        gradient[parent * particles.state_dims * 2] = 2.0;
        gradient[parent * particles.state_dims * 2 + 1] = -1.0;

        let transfer = rebuild_particles(
            &mut particles,
            &[],
            &[SplitSelection {
                index: parent,
                child_fractions: vec![0.25; 4],
            }],
            RebuildOptions {
                state_gradient: Some(&gradient),
                log_normalized_gradient: false,
                state_prolongation_scale: 1.0,
                max_state_transfer_rms: 10.0,
                restore_bootstrap_children: true,
                bootstrap_templates_are_current: false,
                bootstrap_seed_spread: 0.0,
                cooldown: 0,
                domain_min: &model.config.domain_min,
                domain_max: &model.config.domain_max,
            },
        )
        .unwrap();

        let child_rows = template
            .children
            .iter()
            .map(|child| {
                particles
                    .particle_id
                    .iter()
                    .position(|id| *id == child.particle_id)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let total_child_measure = child_rows
            .iter()
            .map(|row| particles.represented_measure[*row])
            .sum::<f32>();
        for (axis, expected_position) in expected_position
            .iter()
            .enumerate()
            .take(particles.spatial_dims)
        {
            let centroid = child_rows
                .iter()
                .map(|row| {
                    particles.represented_measure[*row] / total_child_measure
                        * particles.positions[*row][axis]
                })
                .sum::<f32>();
            assert!((centroid - expected_position).abs() <= 1.0e-6);
        }
        for (channel, expected) in expected.iter().enumerate().take(particles.state_dims) {
            let mean = child_rows
                .iter()
                .map(|row| {
                    particles.represented_measure[*row] / total_child_measure
                        * particles.states[*row * particles.state_dims + channel]
                })
                .sum::<f32>();
            assert!((mean - expected).abs() <= 1.0e-6);
        }
        assert!(child_rows.iter().all(|row| {
            particles.states[*row * particles.state_dims..(*row + 1) * particles.state_dims]
                .iter()
                .any(|value| value.abs() > 0.1)
        }));
        assert!(child_rows.iter().any(|row| {
            (particles.states[*row * particles.state_dims] - expected[0]).abs() > 1.0e-4
        }));
        assert!(transfer.max_rms > 0.0);
    }

    #[test]
    fn persistent_bootstrap_exposes_live_quadrature_state() {
        let measure = std::f32::consts::PI * 0.08_f32.powi(2);
        let mut particles = AdaptiveParticleSet::from_equal_measure(
            vec![[0.25, -0.1, 0.0, 0.0]],
            vec![0.5],
            2,
            1,
            measure,
            0.1,
        )
        .unwrap();
        let offsets = [[-0.02, 0.0], [0.02, 0.0], [0.0, -0.02], [0.0, 0.02]];
        let states = [0.2, 0.4, 0.6, 0.8];
        let mut child_covariance = [0.0; 9];
        child_covariance[0] = 1.0e-4;
        child_covariance[4] = 1.0e-4;
        particles.bootstrap_templates = vec![super::super::AdaptiveBootstrapTemplate {
            parent_id: particles.particle_id[0],
            children: offsets
                .into_iter()
                .zip(states)
                .enumerate()
                .map(|(index, (offset, state))| AdaptiveBootstrapChild {
                    position: [0.25 + offset[0], -0.1 + offset[1], 0.0, 0.0],
                    state: vec![state],
                    represented_measure: measure / 4.0,
                    bandwidth: 0.1,
                    covariance: child_covariance,
                    particle_id: 10 + index as u64,
                    generation: 1,
                })
                .collect(),
        }];

        rebuild_particles(
            &mut particles,
            &[],
            &[SplitSelection {
                index: 0,
                child_fractions: vec![0.25; 4],
            }],
            RebuildOptions {
                state_gradient: None,
                log_normalized_gradient: false,
                state_prolongation_scale: 0.0,
                max_state_transfer_rms: 1.0,
                restore_bootstrap_children: true,
                bootstrap_templates_are_current: true,
                bootstrap_seed_spread: 0.0,
                cooldown: 0,
                domain_min: &[-1.0, -1.0, 0.0],
                domain_max: &[1.0, 1.0, 0.0],
            },
        )
        .unwrap();

        let exposed = particles
            .particle_id
            .iter()
            .map(|id| {
                let row = (*id - 10) as usize;
                (row, particles.states[row])
            })
            .collect::<Vec<_>>();
        for (row, state) in exposed {
            assert!((state - states[row]).abs() <= 1.0e-6);
        }
        assert!((particles.states.iter().sum::<f32>() / 4.0 - 0.5).abs() <= 1.0e-6);
    }

    #[test]
    fn scheduled_restriction_builds_a_detail_aware_target_cut() {
        let base = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 7);
        let mut config = crate::adaptive::AdaptiveNpaConfig::growing_2d();
        config.min_leaves = 40;
        config.initial_leaves = 64;
        config.target_leaves = 40;
        config.max_leaves = 64;
        config.bootstrap_fine_leaves = 64;
        config.hierarchical_bootstrap_seed = true;
        config.hierarchical_restriction_step = 1;
        config.retain_bootstrap_templates = false;
        config.local_residual_scale = 0.0;
        config.proxy.context_scale = 0.0;
        let model = AdaptiveNpaModel::seeded(base, config, 9).unwrap();
        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let initial = super::super::seed_adaptive_particles_scaled(
            &model,
            64,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            0.1,
        )
        .unwrap();

        let trace = run_adaptive_rollout(
            &model,
            initial,
            AdaptiveRolloutConfig {
                steps: 1,
                update_prob: 0.5,
                topology_enabled: true,
                snapshot_interval: 1,
                ..AdaptiveRolloutConfig::default()
            },
        )
        .unwrap();

        assert_eq!(trace.particles.len(), 40);
        assert!(trace.particles.bootstrap_templates.is_empty());
        assert_eq!(trace.metrics[0].merge_events, 8);
        assert!((trace.particles.total_measure() - f64::from(total_measure)).abs() <= 1.0e-7);
    }

    #[test]
    fn scheduled_restriction_honors_per_interval_event_budget() {
        let base = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 7);
        let mut config = crate::adaptive::AdaptiveNpaConfig::growing_2d();
        config.min_leaves = 40;
        config.initial_leaves = 64;
        config.target_leaves = 40;
        config.max_leaves = 64;
        config.bootstrap_fine_leaves = 64;
        config.hierarchical_bootstrap_seed = true;
        config.hierarchical_restriction_step = 1;
        config.hierarchical_restriction_leaf_delta_per_interval = 6;
        config.hierarchical_restriction_arity = crate::adaptive::AdaptiveRestrictionArity::Mixed;
        config.hierarchical_restriction_schedule =
            crate::adaptive::AdaptiveRestrictionSchedule::Nested;
        config.topology_interval = 1;
        config.topology_end_step = 0;
        config.steady_topology_start_step = 32;
        config.local_residual_scale = 0.0;
        config.proxy.context_scale = 0.0;
        let model = AdaptiveNpaModel::seeded(base, config, 9).unwrap();
        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let initial = super::super::seed_adaptive_particles_scaled(
            &model,
            64,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            0.1,
        )
        .unwrap();

        let trace = run_adaptive_rollout(
            &model,
            initial,
            AdaptiveRolloutConfig {
                steps: 4,
                update_prob: 0.5,
                topology_enabled: true,
                snapshot_interval: 1,
                ..AdaptiveRolloutConfig::default()
            },
        )
        .unwrap();

        assert_eq!(
            trace
                .metrics
                .iter()
                .map(|metrics| metrics.leaf_count)
                .collect::<Vec<_>>(),
            [58, 52, 46, 40]
        );
        assert!(
            trace
                .metrics
                .iter()
                .all(|metrics| { metrics.merge_events == 3 && metrics.split_events == 0 })
        );
        for snapshots in trace.snapshots.windows(2) {
            let before = adaptive_template_child_groups(&snapshots[0].particles);
            let after = adaptive_template_child_groups(&snapshots[1].particles);
            assert!(before.is_subset(&after));
        }
        for arity in 2..=4 {
            assert_eq!(
                trace
                    .particles
                    .bootstrap_templates
                    .iter()
                    .filter(|template| template.children.len() == arity)
                    .count(),
                4
            );
        }
        assert!((trace.particles.total_measure() - f64::from(total_measure)).abs() <= 1.0e-7);
    }

    #[test]
    fn steady_split_never_restores_stale_bootstrap_state() {
        let footprint = 0.1_f32;
        let measure = crate::adaptive::unit_ball_measure(2) * footprint.powi(2);
        let mut particles =
            AdaptiveParticleSet::from_equal_measure(vec![[0.0; 4]], vec![1.0], 2, 1, measure, 0.1)
                .unwrap();
        particles.bootstrap_templates = vec![super::super::AdaptiveBootstrapTemplate {
            parent_id: particles.particle_id[0],
            children: (0..4)
                .map(|particle_id| AdaptiveBootstrapChild {
                    position: [particle_id as f32 * 0.01, 0.0, 0.0, 0.0],
                    state: vec![0.0],
                    represented_measure: measure / 4.0,
                    bandwidth: 0.1,
                    covariance: [0.0; 9],
                    particle_id: particle_id as u64 + 10,
                    generation: 1,
                })
                .collect(),
        }];
        let options = RebuildOptions {
            state_gradient: None,
            log_normalized_gradient: false,
            state_prolongation_scale: 0.0,
            max_state_transfer_rms: 1.0,
            restore_bootstrap_children: false,
            bootstrap_templates_are_current: false,
            bootstrap_seed_spread: 0.0,
            cooldown: 0,
            domain_min: &[-1.0, -1.0, 0.0],
            domain_max: &[1.0, 1.0, 0.0],
        };

        rebuild_particles(
            &mut particles,
            &[],
            &[SplitSelection {
                index: 0,
                child_fractions: vec![0.25; 4],
            }],
            options,
        )
        .unwrap();

        assert_eq!(particles.len(), 4);
        assert!(particles.states.iter().all(|state| *state == 1.0));
        assert!(particles.bootstrap_templates.is_empty());
    }

    #[test]
    fn rebuild_accepts_unequal_children_and_preserves_weighted_intensive_state() {
        let footprint = 0.08_f32;
        let measure = std::f32::consts::PI * footprint.powi(2);
        let mut particles =
            AdaptiveParticleSet::from_equal_measure(vec![[0.0; 4]], vec![0.35], 2, 1, measure, 0.1)
                .unwrap();
        let initial_measure = particles.total_measure();
        let options = RebuildOptions {
            state_gradient: Some(&[0.15, -0.08]),
            log_normalized_gradient: false,
            state_prolongation_scale: 1.0,
            max_state_transfer_rms: 1.0,
            restore_bootstrap_children: false,
            bootstrap_templates_are_current: false,
            bootstrap_seed_spread: 0.0,
            cooldown: 0,
            domain_min: &[-1.0, -1.0, 0.0],
            domain_max: &[1.0, 1.0, 0.0],
        };
        rebuild_particles(
            &mut particles,
            &[],
            &[SplitSelection {
                index: 0,
                child_fractions: vec![0.1, 0.2, 0.3, 0.4],
            }],
            options,
        )
        .unwrap();

        assert_eq!(particles.len(), 4);
        assert!((particles.total_measure() - initial_measure).abs() < 1.0e-8);
        let weighted_state = particles
            .represented_measure
            .iter()
            .zip(&particles.states)
            .map(|(measure, state)| *measure as f64 / initial_measure * f64::from(*state))
            .sum::<f64>();
        assert!((weighted_state - 0.35).abs() < 1.0e-6);
        let footprint_bits = (0..particles.len())
            .map(|index| particles.footprint(index).to_bits())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(footprint_bits.len(), 4);
    }

    #[test]
    fn learned_refinement_defect_uses_stable_ranking_and_learned_gates() {
        let base = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 7);
        let mut config = crate::adaptive::AdaptiveNpaConfig::growing_2d();
        config.min_leaves = 1;
        config.target_leaves = 8;
        config.max_leaves = 64;
        let mut model = AdaptiveNpaModel::seeded(base, config, 9).unwrap();
        model.controller.weights.output_weights.fill(0.0);
        model.controller.weights.output_bias = vec![0.25, 0.4, 2.0, -2.0];
        let particles = AdaptiveParticleSet::from_equal_measure(
            (0..8)
                .map(|index| [index as f32 * 0.02 - 0.07, 0.0, 0.0, 0.0])
                .collect(),
            vec![0.1; 8 * 16],
            2,
            16,
            0.2,
            0.1,
        )
        .unwrap();
        let perception = rule_perception_pair(&model.config, &model.rule, &particles).unwrap();
        let base_update = model
            .rule
            .forward_update_from_features(&perception.npa_compatible.features)
            .unwrap();
        let features = controller_features(
            &model.config,
            &particles,
            &perception.normalized,
            &base_update,
        );
        let learned = topology_controller_output(
            &model,
            &particles,
            &perception.normalized,
            &features,
            AdaptiveTopologyControl::Learned,
        )
        .unwrap();
        let oracle = topology_controller_output(
            &model,
            &particles,
            &perception.normalized,
            &features,
            AdaptiveTopologyControl::RefinementDefectOracle,
        )
        .unwrap();
        let hybrid = topology_controller_output(
            &model,
            &particles,
            &perception.normalized,
            &features,
            AdaptiveTopologyControl::LearnedRefinementDefect,
        )
        .unwrap();

        for ((hybrid, oracle), learned) in hybrid.iter().zip(&oracle).zip(&learned) {
            assert_eq!(hybrid.desired_log_footprint, oracle.desired_log_footprint);
            assert_eq!(hybrid.log_bandwidth_ratio, learned.log_bandwidth_ratio);
            assert_eq!(oracle.split_probability, 1.0);
            assert_eq!(oracle.merge_probability, 1.0);
        }
        let learned_split_gate = learned
            .iter()
            .map(|output| output.split_probability)
            .fold(0.0_f32, f32::max);
        let learned_merge_gate = learned
            .iter()
            .map(|output| output.merge_probability)
            .fold(0.0_f32, f32::max);
        assert!(hybrid.iter().all(|output| {
            output.split_probability == learned_split_gate
                && output.merge_probability == learned_merge_gate
        }));
    }

    #[test]
    fn bootstrap_seed_spread_is_deterministic_and_centroid_preserving() {
        let parent = CanonicalMaterial {
            represented_measure: 1.0,
            position: vec![0.1, -0.2],
            covariance: vec![0.01, 0.0, 0.0, 0.01],
            extensive: Vec::new(),
        };
        let mut children = canonical_split(&parent).unwrap();
        let expected_covariance = children[0].covariance.clone();
        spread_bootstrap_children(&parent, &mut children, 0.05, 17);
        let centroid = (0..2)
            .map(|axis| {
                children
                    .iter()
                    .map(|child| child.position[axis])
                    .sum::<f64>()
                    / children.len() as f64
            })
            .collect::<Vec<_>>();
        assert!((centroid[0] - parent.position[0]).abs() < 1.0e-12);
        assert!((centroid[1] - parent.position[1]).abs() < 1.0e-12);
        for child in &children {
            let radius = child
                .position
                .iter()
                .zip(&parent.position)
                .map(|(child, parent)| (child - parent).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!((radius - 0.05).abs() < 1.0e-7);
            assert_eq!(child.covariance, expected_covariance);
        }
        let mut repeated = canonical_split(&parent).unwrap();
        spread_bootstrap_children(&parent, &mut repeated, 0.05, 17);
        assert_eq!(children, repeated);
    }

    #[test]
    fn split_events_crossing_domain_are_rejected_without_clamping() {
        let material = |x| CanonicalMaterial {
            represented_measure: 1.0,
            position: vec![x, 0.0],
            covariance: vec![0.01, 0.0, 0.0, 0.01],
            extensive: Vec::new(),
        };
        let domain_min = [-1.0, -1.0, 0.0];
        let domain_max = [1.0, 1.0, 0.0];
        assert!(split_fits_domain(&material(0.0), &domain_min, &domain_max));
        assert!(!split_fits_domain(
            &material(0.95),
            &domain_min,
            &domain_max
        ));
    }

    #[test]
    fn compatible_hierarchy_cluster_can_merge_without_prior_split_siblings() {
        let positions = vec![
            [-0.12, -0.02, 0.0, 0.0],
            [-0.10, 0.02, 0.0, 0.0],
            [-0.08, -0.01, 0.0, 0.0],
            [-0.06, 0.01, 0.0, 0.0],
            [0.06, -0.02, 0.0, 0.0],
            [0.08, 0.02, 0.0, 0.0],
            [0.10, -0.01, 0.0, 0.0],
            [0.12, 0.01, 0.0, 0.0],
        ];
        let total_measure = std::f32::consts::PI * 0.2 * 0.2;
        let states = vec![0.0; positions.len() * 16];
        let mut particles =
            AdaptiveParticleSet::from_equal_measure(positions, states, 2, 16, total_measure, 0.1)
                .unwrap();
        assert!(particles.sibling_group.iter().all(|group| *group == 0));
        let fine_footprint = particles.footprint(0);
        let mut config = crate::adaptive::AdaptiveNpaConfig::growing_2d();
        config.reference_footprint = fine_footprint;
        config.min_footprint = 0.01;
        config.max_footprint = 0.2;
        config.min_leaves = 1;
        config.max_leaves = 16;
        config.target_leaves = 5;
        config.max_events_per_interval = 2;
        config.merge_ratio = 1.1;
        let model =
            AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 7), config, 9)
                .unwrap();
        let desired = (0.1 / fine_footprint).ln();
        let controller = vec![
            AdaptiveControllerOutput {
                desired_log_footprint: desired,
                log_bandwidth_ratio: 0.0,
                split_probability: 0.0,
                merge_probability: 1.0,
            };
            particles.len()
        ];
        let decision = apply_topology(&model, &mut particles, &controller, None, 0).unwrap();
        assert_eq!(decision.split_events, 0);
        assert_eq!(decision.merge_events, 1);
        assert_eq!(particles.len(), 5);
        let footprints = (0..particles.len())
            .map(|index| particles.footprint(index))
            .collect::<Vec<_>>();
        assert!(
            footprints.iter().copied().fold(0.0_f32, f32::max)
                > footprints.iter().copied().fold(f32::INFINITY, f32::min)
        );
    }

    #[test]
    fn merge_selection_prefers_the_lowest_state_transfer_error() {
        let positions = vec![
            [-0.12, -0.02, 0.0, 0.0],
            [-0.10, 0.02, 0.0, 0.0],
            [-0.08, -0.01, 0.0, 0.0],
            [-0.06, 0.01, 0.0, 0.0],
            [0.06, -0.02, 0.0, 0.0],
            [0.08, 0.02, 0.0, 0.0],
            [0.10, -0.01, 0.0, 0.0],
            [0.12, 0.01, 0.0, 0.0],
        ];
        let mut states = vec![0.0; positions.len() * 16];
        for index in 0..4 {
            states[index * 16] = index as f32 * 0.2;
        }
        let total_measure = std::f32::consts::PI * 0.2 * 0.2;
        let mut particles =
            AdaptiveParticleSet::from_equal_measure(positions, states, 2, 16, total_measure, 0.1)
                .unwrap();
        particles.sibling_group = vec![1, 1, 1, 1, 2, 2, 2, 2];
        let fine_footprint = particles.footprint(0);

        let mut config = crate::adaptive::AdaptiveNpaConfig::growing_2d();
        config.reference_footprint = fine_footprint;
        config.min_footprint = 0.01;
        config.max_footprint = 0.2;
        config.min_leaves = 1;
        config.max_leaves = 16;
        config.target_leaves = 5;
        config.max_events_per_interval = 1;
        config.merge_ratio = 1.1;
        config.merge_state_rms_limit = 1.0;
        config.spatial_merge_groups_enabled = false;
        let model =
            AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 7), config, 9)
                .unwrap();
        let desired = (0.2 / fine_footprint).ln();
        let controller = vec![
            AdaptiveControllerOutput {
                desired_log_footprint: desired,
                log_bandwidth_ratio: 0.0,
                split_probability: 0.0,
                merge_probability: 1.0,
            };
            particles.len()
        ];

        let decision = apply_topology(&model, &mut particles, &controller, None, 0).unwrap();
        assert_eq!((decision.split_events, decision.merge_events), (0, 1));
        assert_eq!(particles.len(), 5);
        assert!(particles.particle_id.starts_with(&[0, 1, 2, 3]));
    }

    #[test]
    fn topology_reallocates_resolution_without_drifting_the_leaf_budget() {
        let positions = vec![
            [-0.12, -0.02, 0.0, 0.0],
            [-0.10, 0.02, 0.0, 0.0],
            [-0.08, -0.01, 0.0, 0.0],
            [-0.06, 0.01, 0.0, 0.0],
            [0.06, -0.02, 0.0, 0.0],
            [0.08, 0.02, 0.0, 0.0],
            [0.10, -0.01, 0.0, 0.0],
            [0.12, 0.01, 0.0, 0.0],
        ];
        let total_measure = std::f32::consts::PI * 0.2 * 0.2;
        let states = vec![0.0; positions.len() * 16];
        let mut particles =
            AdaptiveParticleSet::from_equal_measure(positions, states, 2, 16, total_measure, 0.1)
                .unwrap();
        let initial_measure = particles.total_measure();
        let initial_ids = particles.particle_id.clone();
        let fine_footprint = particles.footprint(0);
        let hierarchy = AdaptiveProxyHierarchy::build(&particles, 4).unwrap();
        let merge_indices = hierarchy.nodes[hierarchy.levels[0][0]]
            .children
            .iter()
            .filter_map(|member| match member {
                crate::AdaptiveHierarchyMember::Leaf(index) => Some(*index),
                crate::AdaptiveHierarchyMember::Proxy(_) => None,
            })
            .collect::<Vec<_>>();
        let split_index = (0..particles.len())
            .find(|index| !merge_indices.contains(index))
            .unwrap();

        let mut config = crate::adaptive::AdaptiveNpaConfig::growing_2d();
        config.reference_footprint = fine_footprint;
        config.min_footprint = 0.01;
        config.max_footprint = 0.2;
        config.min_leaves = 1;
        config.max_leaves = 16;
        config.target_leaves = particles.len();
        config.max_events_per_interval = 2;
        config.merge_ratio = 1.1;
        let model =
            AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 7), config, 9)
                .unwrap();
        let mut controller = vec![
            AdaptiveControllerOutput {
                desired_log_footprint: 0.0,
                log_bandwidth_ratio: 0.0,
                split_probability: 1.0,
                merge_probability: 1.0,
            };
            particles.len()
        ];
        for index in merge_indices {
            controller[index].desired_log_footprint = (0.2 / fine_footprint).ln();
        }
        let mut unpaired = particles.clone();
        let unpaired_decision =
            apply_topology(&model, &mut unpaired, &controller, None, 0).unwrap();
        assert_eq!(unpaired_decision.merge_events, 0);
        assert_eq!(unpaired_decision.split_events, 0);
        assert_eq!(unpaired.len(), model.config.target_leaves);

        controller[split_index].desired_log_footprint = (0.01 / fine_footprint).ln();

        let decision = apply_topology(&model, &mut particles, &controller, None, 0).unwrap();
        assert_eq!((decision.split_events, decision.merge_events), (1, 1));
        assert_eq!(particles.len(), model.config.target_leaves);
        assert!((particles.total_measure() - initial_measure).abs() < 1.0e-9);
        assert_eq!(particles.particle_id, initial_ids);
        let min_footprint = (0..particles.len())
            .map(|index| particles.footprint(index))
            .fold(f32::INFINITY, f32::min);
        let max_footprint = (0..particles.len())
            .map(|index| particles.footprint(index))
            .fold(0.0_f32, f32::max);
        assert!(max_footprint >= 3.9 * min_footprint);
    }

    #[test]
    fn paired_local_detail_matches_static_material_training_semantics() {
        let positions = vec![
            [0.6, 0.4, 0.0, 0.0],
            [-0.42, -0.40, 0.0, 0.0],
            [-0.40, -0.42, 0.0, 0.0],
            [-0.38, -0.40, 0.0, 0.0],
            [-0.40, -0.38, 0.0, 0.0],
            [0.7, -0.7, 0.0, 0.0],
            [-0.7, 0.7, 0.0, 0.0],
            [0.8, 0.8, 0.0, 0.0],
        ];
        let state_dims = 3;
        let states = (0..positions.len())
            .flat_map(|row| [row as f32 * 0.1, 1.0 + row as f32 * 0.2, -0.3 * row as f32])
            .collect::<Vec<_>>();
        let fine_measure = 0.0025;
        let fine_footprint = material_footprint_radius(fine_measure, 2);
        let mut particles = AdaptiveParticleSet::from_equal_measure(
            positions,
            states,
            2,
            state_dims,
            fine_measure * 8.0,
            fine_footprint,
        )
        .unwrap();
        particles.represented_measure[0] = 4.0 * fine_measure;
        for row in 0..particles.len() {
            let footprint = particles.footprint(row);
            particles.render_footprint[row] = footprint;
            particles.bandwidth[row] = footprint;
            let variance = (0.5 * footprint).powi(2);
            particles.covariance[row] = [variance, 0.0, 0.0, 0.0, variance, 0.0, 0.0, 0.0, 0.0];
        }
        particles.validate().unwrap();

        let initial_measure = particles.represented_measure.clone();
        let initial_ids = particles.particle_id.clone();
        let weighted = |values: &[f32], width: usize, particles: &AdaptiveParticleSet| {
            let mut sum = vec![0.0_f32; width];
            for row in 0..particles.len() {
                for channel in 0..width {
                    sum[channel] +=
                        values[row * width + channel] * particles.represented_measure[row];
                }
            }
            sum
        };
        let flattened_positions = particles
            .positions
            .iter()
            .flat_map(|position| [position[0], position[1]])
            .collect::<Vec<_>>();
        let position_before = weighted(&flattened_positions, 2, &particles);
        let state_before = weighted(&particles.states, state_dims, &particles);
        let coarse_before = particles.positions[0];
        let expected_merged = [
            particles.positions[1..5]
                .iter()
                .map(|position| position[0])
                .sum::<f32>()
                / 4.0,
            particles.positions[1..5]
                .iter()
                .map(|position| position[1])
                .sum::<f32>()
                / 4.0,
        ];

        let mut config = crate::adaptive::AdaptiveNpaConfig::growing_2d();
        config.reference_footprint = fine_footprint;
        config.base_rule_footprint = fine_footprint;
        config.min_leaves = 1;
        config.max_leaves = 16;
        config.target_leaves = particles.len();
        config.max_events_per_interval = 1;
        config.paired_topology_split_radius_scale = 1.0;
        config.paired_topology_merge_detail_scale = 0.01;
        let model =
            AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 7), config, 9)
                .unwrap();
        let detail = [10.0, 0.01, 0.02, 0.03, 0.04, 5.0, 6.0, 7.0];
        let mut rejected_model = model.clone();
        rejected_model.config.min_reallocation_relative_gain = 1.0;
        rejected_model.validate().unwrap();
        let mut rejected_particles = particles.clone();
        let decision =
            apply_paired_local_detail_topology(&model, &mut particles, &detail, 32).unwrap();

        assert_eq!((decision.split_events, decision.merge_events), (1, 1));
        assert_eq!(particles.represented_measure, initial_measure);
        assert_eq!(particles.particle_id, initial_ids);
        assert_eq!(particles.len(), 8);
        assert!((particles.positions[0][0] - expected_merged[0]).abs() < 1.0e-6);
        assert!((particles.positions[0][1] - expected_merged[1]).abs() < 1.0e-6);
        let split_radius = (1.5_f32).sqrt() * fine_footprint;
        let expected_offsets = [
            [-split_radius, 0.0],
            [split_radius, 0.0],
            [0.0, -split_radius],
            [0.0, split_radius],
        ];
        for offset in expected_offsets {
            assert!(particles.positions[1..5].iter().any(|position| {
                (position[0] - coarse_before[0] - offset[0]).abs() < 1.0e-6
                    && (position[1] - coarse_before[1] - offset[1]).abs() < 1.0e-6
            }));
        }
        let flattened_positions = particles
            .positions
            .iter()
            .flat_map(|position| [position[0], position[1]])
            .collect::<Vec<_>>();
        let position_after = weighted(&flattened_positions, 2, &particles);
        let state_after = weighted(&particles.states, state_dims, &particles);
        assert!(
            position_before
                .iter()
                .zip(position_after)
                .all(|(before, after)| (*before - after).abs() < 2.0e-6)
        );
        assert!(
            state_before
                .iter()
                .zip(state_after)
                .all(|(before, after)| (*before - after).abs() < 2.0e-6)
        );

        let rejected_before = rejected_particles.clone();
        let rejected = apply_paired_local_detail_topology(
            &rejected_model,
            &mut rejected_particles,
            &detail,
            32,
        )
        .unwrap();
        assert_eq!((rejected.split_events, rejected.merge_events), (0, 0));
        assert_eq!(rejected_particles, rejected_before);
    }

    #[test]
    fn continuous_local_detail_relocates_scale_and_conserves_extensive_fields() {
        let positions = vec![
            [-0.6, -0.2, 0.0, 0.0],
            [-0.2, 0.1, 0.0, 0.0],
            [0.0, -0.1, 0.0, 0.0],
            [0.2, 0.2, 0.0, 0.0],
            [0.38, -0.04, 0.0, 0.0],
            [0.42, 0.02, 0.0, 0.0],
        ];
        let state_dims = 3;
        let states = (0..positions.len())
            .flat_map(|row| [row as f32 * 0.2, 1.0 - row as f32 * 0.1, row as f32 * -0.3])
            .collect::<Vec<_>>();
        let mut particles =
            AdaptiveParticleSet::from_equal_measure(positions, states, 2, state_dims, 6.0, 0.1)
                .unwrap();
        particles.represented_measure = vec![0.70, 0.82, 0.94, 1.06, 1.18, 1.30];
        for row in 0..particles.len() {
            let footprint = particles.footprint(row);
            particles.render_footprint[row] = footprint;
            particles.bandwidth[row] = footprint;
        }
        particles.validate().unwrap();

        let weighted = |values: &[f32], width: usize, particles: &AdaptiveParticleSet| {
            let mut sum = vec![0.0_f64; width];
            for row in 0..particles.len() {
                for channel in 0..width {
                    sum[channel] += values[row * width + channel] as f64
                        * particles.represented_measure[row] as f64;
                }
            }
            sum
        };
        let flatten_positions = |particles: &AdaptiveParticleSet| {
            particles
                .positions
                .iter()
                .flat_map(|position| [position[0], position[1]])
                .collect::<Vec<_>>()
        };
        let spatial_second_moment = |particles: &AdaptiveParticleSet| {
            let mut moment = [0.0_f64; 3];
            for row in 0..particles.len() {
                let weight = f64::from(particles.represented_measure[row]);
                let x = f64::from(particles.positions[row][0]);
                let y = f64::from(particles.positions[row][1]);
                moment[0] += weight * (f64::from(particles.covariance[row][0]) + x * x);
                moment[1] += weight * (f64::from(particles.covariance[row][1]) + x * y);
                moment[2] += weight * (f64::from(particles.covariance[row][4]) + y * y);
            }
            moment
        };
        let measure_before = particles.represented_measure.clone();
        let ids_before = particles.particle_id.clone();
        let position_before = weighted(&flatten_positions(&particles), 2, &particles);
        let second_moment_before = spatial_second_moment(&particles);
        let state_before = weighted(&particles.states, state_dims, &particles);
        let coarse_before = particles.positions[5];
        let fine_before = particles.positions[0];
        let coarse_render_before = particles.render_footprint[5];
        let fine_render_before = particles.render_footprint[0];

        let mut config = crate::adaptive::AdaptiveNpaConfig::growing_2d();
        config.material_seed_layout = crate::adaptive::AdaptiveMaterialSeedLayout::GradedContinuous;
        config.min_leaves = particles.len();
        config.target_leaves = particles.len();
        config.max_leaves = particles.len();
        config.max_events_per_interval = 2;
        let model =
            AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 7), config, 9)
                .unwrap();
        let detail = [0.01, 0.2, 0.3, 2.0, 5.0, 10.0];
        let decision =
            apply_continuous_local_detail_topology(&model, &mut particles, &detail, 32).unwrap();

        assert_eq!((decision.split_events, decision.merge_events), (2, 2));
        assert_eq!(particles.represented_measure, measure_before);
        assert_eq!(particles.particle_id, ids_before);
        assert_eq!(particles.render_footprint[5], fine_render_before);
        assert_eq!(particles.render_footprint[0], coarse_render_before);
        let distance_squared =
            |lhs: [f32; 4], rhs: [f32; 4]| (lhs[0] - rhs[0]).powi(2) + (lhs[1] - rhs[1]).powi(2);
        assert!(
            distance_squared(particles.positions[5], fine_before)
                < distance_squared(particles.positions[5], coarse_before)
        );
        assert!(
            distance_squared(particles.positions[0], coarse_before)
                < distance_squared(particles.positions[0], fine_before)
        );

        let position_after = weighted(&flatten_positions(&particles), 2, &particles);
        let second_moment_after = spatial_second_moment(&particles);
        let state_after = weighted(&particles.states, state_dims, &particles);
        assert!(
            position_before
                .iter()
                .zip(position_after)
                .all(|(before, after)| (*before - after).abs() < 2.0e-6)
        );
        assert!(
            state_before
                .iter()
                .zip(state_after)
                .all(|(before, after)| (*before - after).abs() < 2.0e-6)
        );
        assert!(
            second_moment_before
                .iter()
                .zip(second_moment_after)
                .all(|(before, after)| (*before - after).abs() < 2.0e-6)
        );

        let mut rejected_model = model.clone();
        rejected_model.config.min_reallocation_relative_gain = 1.0;
        let mut rejected_particles = particles.clone();
        let rejected_before = rejected_particles.clone();
        let rejected = apply_continuous_local_detail_topology(
            &rejected_model,
            &mut rejected_particles,
            &detail,
            64,
        )
        .unwrap();
        assert_eq!((rejected.split_events, rejected.merge_events), (0, 0));
        assert_eq!(rejected_particles, rejected_before);
    }

    #[test]
    fn topology_realizes_a_continuous_desired_field_with_unequal_children() {
        let positions = vec![
            [0.0, 0.0, 0.0, 0.0],
            [-0.10, 0.0, 0.0, 0.0],
            [0.10, 0.0, 0.0, 0.0],
            [0.0, -0.10, 0.0, 0.0],
            [0.0, 0.10, 0.0, 0.0],
        ];
        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let mut particles = AdaptiveParticleSet::from_equal_measure(
            positions,
            vec![0.0; 5 * 16],
            2,
            16,
            total_measure,
            0.1,
        )
        .unwrap();
        let initial_measure = particles.total_measure();
        let footprint = particles.footprint(0);
        let mut config = crate::adaptive::AdaptiveNpaConfig::growing_2d();
        config.reference_footprint = footprint;
        config.min_footprint = footprint * 0.2;
        config.max_footprint = footprint * 4.0;
        config.min_leaves = 1;
        config.target_leaves = 8;
        config.max_leaves = 8;
        config.max_events_per_interval = 1;
        config.split_ratio = 0.95;
        config.split_probability = 0.5;
        config.max_unequal_split_measure_ratio = 4.0;
        config.split_field_neighbors = 4;
        config.spatial_merge_groups_enabled = false;
        let model =
            AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 7), config, 9)
                .unwrap();
        let proposed = [0.35_f32, 0.45, 1.35, 0.65, 1.15];
        let controller = proposed
            .into_iter()
            .enumerate()
            .map(|(index, ratio)| AdaptiveControllerOutput {
                desired_log_footprint: ratio.ln(),
                log_bandwidth_ratio: 0.0,
                split_probability: if index == 0 { 1.0 } else { 0.0 },
                merge_probability: 0.0,
            })
            .collect::<Vec<_>>();

        let decision = apply_topology(&model, &mut particles, &controller, None, 0).unwrap();
        assert_eq!((decision.split_events, decision.merge_events), (1, 0));
        assert_eq!(particles.len(), 8);
        assert!((particles.total_measure() - initial_measure).abs() < 1.0e-8);
        let child_group = particles
            .sibling_group
            .iter()
            .copied()
            .find(|group| *group != 0)
            .unwrap();
        let child_footprints = (0..particles.len())
            .filter(|index| particles.sibling_group[*index] == child_group)
            .map(|index| particles.footprint(index).to_bits())
            .collect::<std::collections::BTreeSet<_>>();
        assert!(child_footprints.len() > 1);
        let scale_metrics = material_scale_metrics(&particles, footprint);
        assert!(scale_metrics.fractional_octave_fraction > 0.0);
        assert!(scale_metrics.dyadic_quantization_rmse_octaves > 0.0);
        assert!(scale_metrics.occupied_sixty_fourth_octave_bins >= 3);
    }

    #[test]
    fn event_probabilities_gate_discrete_topology() {
        let positions = vec![
            [-0.12, -0.02, 0.0, 0.0],
            [-0.10, 0.02, 0.0, 0.0],
            [-0.08, -0.01, 0.0, 0.0],
            [-0.06, 0.01, 0.0, 0.0],
        ];
        let total_measure = std::f32::consts::PI * 0.2 * 0.2;
        let states = vec![0.0; positions.len() * 16];
        let mut particles =
            AdaptiveParticleSet::from_equal_measure(positions, states, 2, 16, total_measure, 0.1)
                .unwrap();
        let fine_footprint = particles.footprint(0);
        let mut config = crate::adaptive::AdaptiveNpaConfig::growing_2d();
        config.reference_footprint = fine_footprint;
        config.min_footprint = 0.01;
        config.max_footprint = 0.2;
        config.min_leaves = 1;
        config.max_leaves = 8;
        config.target_leaves = 1;
        config.max_events_per_interval = 1;
        config.merge_ratio = 1.1;
        config.merge_probability = 0.5;
        let model =
            AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 7), config, 9)
                .unwrap();
        let desired = (0.2 / fine_footprint).ln();
        let controller = vec![
            AdaptiveControllerOutput {
                desired_log_footprint: desired,
                log_bandwidth_ratio: 0.0,
                split_probability: 0.0,
                merge_probability: 0.49,
            };
            particles.len()
        ];
        let decision = apply_topology(&model, &mut particles, &controller, None, 0).unwrap();
        assert_eq!(decision.merge_events, 0);
        assert_eq!(particles.len(), 4);
    }

    #[test]
    fn affine_split_prolongation_preserves_parent_mean_and_adds_detail() {
        let footprint = 0.1_f32;
        let measure = std::f32::consts::PI * footprint.powi(2);
        let particles =
            AdaptiveParticleSet::from_equal_measure(vec![[0.0; 4]], vec![0.25], 2, 1, measure, 0.1)
                .unwrap();
        let children = canonical_split(&material_at(&particles, 0)).unwrap();
        // Encoded non-log gradient is physical gradient times bandwidth.
        let child_states = prolonged_child_states(
            &particles,
            0,
            &children,
            Some(&[0.2, -0.1]),
            false,
            1.0,
            1.0,
        );
        let mean =
            child_states.iter().map(|state| state[0]).sum::<f32>() / child_states.len() as f32;
        assert!((mean - 0.25).abs() < 1.0e-6);
        assert!(
            child_states
                .iter()
                .any(|state| (state[0] - mean).abs() > 1.0e-3)
        );
    }

    #[test]
    fn affine_merge_recovers_exact_state_jacobian() {
        let positions = vec![
            [-0.1, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0],
            [0.0, -0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0, 0.0],
        ];
        let states = positions
            .iter()
            .flat_map(|position| {
                [
                    1.0 + 2.0 * position[0] + 3.0 * position[1],
                    -0.5 - position[0] + 0.5 * position[1],
                ]
            })
            .collect::<Vec<_>>();
        let mut particles = AdaptiveParticleSet::from_equal_measure(
            positions,
            states,
            2,
            2,
            std::f32::consts::PI * 0.2_f32.powi(2),
            0.1,
        )
        .unwrap();
        for row in particles.state_jacobian.chunks_exact_mut(4) {
            row.copy_from_slice(&[2.0, 3.0, -1.0, 0.5]);
        }
        let indices = [0, 1, 2, 3];
        let materials = indices
            .iter()
            .map(|index| material_at(&particles, *index))
            .collect::<Vec<_>>();
        let merged = canonical_merge(&materials).unwrap();
        let mean_state = [1.0, -0.5];
        let fitted = fit_merged_state_jacobian(&particles, &indices, &mean_state, &merged).unwrap();

        for (actual, expected) in fitted.iter().zip([2.0, 3.0, -1.0, 0.5]) {
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "{actual} != {expected}"
            );
        }
    }

    #[test]
    fn coarse_refresh_preserves_restricted_internal_jacobian() {
        let mut particles = AdaptiveParticleSet::from_equal_measure(
            vec![
                [-0.1, -0.1, 0.0, 0.0],
                [0.1, -0.1, 0.0, 0.0],
                [-0.1, 0.1, 0.0, 0.0],
                [0.1, 0.1, 0.0, 0.0],
            ],
            vec![0.0; 8],
            2,
            2,
            std::f32::consts::PI * 0.2_f32.powi(2),
            0.1,
        )
        .unwrap();
        particles.state_jacobian.fill(0.75);
        let encoded = vec![0.0; particles.state_jacobian.len()];
        let coarse_threshold = particles.footprint(0) / 2.0;
        refresh_state_jacobian(&mut particles, &encoded, true, coarse_threshold).unwrap();
        assert!(
            particles
                .state_jacobian
                .iter()
                .all(|value| (*value - 0.75).abs() <= f32::EPSILON)
        );

        let native_threshold = particles.footprint(0);
        refresh_state_jacobian(&mut particles, &encoded, true, native_threshold).unwrap();
        assert!(particles.state_jacobian.iter().all(|value| *value == 0.0));
    }
}
