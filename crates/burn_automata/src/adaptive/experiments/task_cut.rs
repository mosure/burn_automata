use super::AdaptiveTaskRestrictionPolicy;
use crate::{
    AutomataError, AutomataResult,
    adaptive::{
        AdaptiveHierarchyRestrictionPolicy, AdaptiveNpaModel, AdaptiveParticleSet,
        AdaptiveRenderDecoder, AdaptiveRolloutConfig, AdaptiveRolloutTrace,
        AdaptiveTopologyControl, material_footprint_radius,
        rollout::{
            advance_adaptive_rollout_with_topology_control,
            run_adaptive_rollout_with_topology_control,
        },
        seed::restrict_adaptive_particles_to_target_by_merge_cost,
        task_merge_oracle::target_render_merge_costs,
    },
    target2d::{Target2dLossConfig, TargetImage2d},
};

#[allow(clippy::too_many_arguments)]
pub(super) fn run_task_quality_rollout(
    model: &AdaptiveNpaModel,
    initial: AdaptiveParticleSet,
    rollout: AdaptiveRolloutConfig,
    topology_control: AdaptiveTopologyControl,
    restriction_policy: AdaptiveTaskRestrictionPolicy,
    render_decoder: AdaptiveRenderDecoder,
    render_compactness: f32,
    target: &TargetImage2d,
    render_config: Target2dLossConfig,
    fine_measure: f32,
) -> AutomataResult<AdaptiveRolloutTrace> {
    if !rollout.topology_enabled || model.config.hierarchical_restriction_step == 0 {
        return run_adaptive_rollout_with_topology_control(
            model,
            initial,
            rollout,
            topology_control,
        );
    }
    if restriction_policy != AdaptiveTaskRestrictionPolicy::TargetRenderOracle {
        let mut deployment_model = model.clone();
        deployment_model.config.hierarchical_restriction_policy = match restriction_policy {
            AdaptiveTaskRestrictionPolicy::DynamicsDetail => {
                AdaptiveHierarchyRestrictionPolicy::DynamicsDetail
            }
            AdaptiveTaskRestrictionPolicy::LearnedController => {
                AdaptiveHierarchyRestrictionPolicy::LearnedController
            }
            AdaptiveTaskRestrictionPolicy::TargetRenderOracle => unreachable!(),
        };
        deployment_model.validate()?;
        return run_adaptive_rollout_with_topology_control(
            &deployment_model,
            initial,
            rollout,
            topology_control,
        );
    }
    if !render_decoder.supports_restriction_labels() {
        return Err(AutomataError::InvalidArgument(
            "target-render-oracle restriction requires isotropic-material-gaussian or the diagnostic compact-moment-gaussian control".to_string(),
        ));
    }
    let cut_step = model.config.hierarchical_restriction_step;
    if cut_step > rollout.steps
        || initial.len() != model.config.bootstrap_fine_leaf_count()
        || initial.len() <= model.config.target_leaves
    {
        return Err(AutomataError::InvalidArgument(format!(
            "target-render-oracle restriction requires a fine {}-leaf rollout and a cut step in 1..={}, got {} leaves and step {cut_step}",
            model.config.bootstrap_fine_leaf_count(),
            rollout.steps,
            initial.len(),
        )));
    }

    // The generic rollout owns the deployable target-independent cut. Disable
    // that one for this bounded oracle control, advance through the same update
    // at the cut step, and then replace its post-update topology phase.
    let mut oracle_model = model.clone();
    oracle_model.config.hierarchical_restriction_step = 0;
    let mut pre = run_adaptive_rollout_with_topology_control(
        &oracle_model,
        initial,
        AdaptiveRolloutConfig {
            steps: cut_step,
            topology_enabled: false,
            snapshot_interval: rollout.snapshot_interval.min(cut_step).max(1),
            ..rollout
        },
        topology_control,
    )?;
    let costs = target_render_merge_costs(
        &pre.particles,
        oracle_model.config.target_leaves,
        target,
        render_config,
        fine_measure,
        render_decoder,
        render_compactness,
        crate::adaptive::AdaptiveRestrictionLabelTarget::TargetImage,
    )?;
    let restricted =
        restrict_adaptive_particles_to_target_by_merge_cost(&oracle_model, &pre.particles, &costs)?;
    record_manual_restriction(&mut pre, restricted.clone());

    let remaining_steps = rollout.steps - cut_step;
    if remaining_steps == 0 {
        return Ok(pre);
    }
    let post = advance_adaptive_rollout_with_topology_control(
        &oracle_model,
        restricted,
        AdaptiveRolloutConfig {
            steps: remaining_steps,
            snapshot_interval: rollout.snapshot_interval.min(remaining_steps).max(1),
            ..rollout
        },
        cut_step,
        topology_control,
    )?;
    pre.metrics.extend(post.metrics);
    pre.snapshots.extend(post.snapshots.into_iter().skip(1));
    pre.particles = post.particles;
    pre.steps = rollout.steps;
    Ok(pre)
}

fn record_manual_restriction(trace: &mut AdaptiveRolloutTrace, particles: AdaptiveParticleSet) {
    if let Some(metrics) = trace.metrics.last_mut() {
        let previous = metrics.leaf_count;
        let footprints = particles
            .represented_measure
            .iter()
            .map(|measure| material_footprint_radius(*measure, particles.spatial_dims))
            .collect::<Vec<_>>();
        let mean = footprints.iter().sum::<f32>() / footprints.len().max(1) as f32;
        let variance = footprints
            .iter()
            .map(|value| (*value - mean).powi(2))
            .sum::<f32>()
            / footprints.len().max(1) as f32;
        metrics.leaf_count = particles.len();
        metrics.total_measure = particles.total_measure();
        metrics.mean_footprint = mean;
        metrics.min_footprint = footprints.iter().copied().fold(f32::INFINITY, f32::min);
        metrics.max_footprint = footprints.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        metrics.footprint_coefficient_of_variation = variance.sqrt() / mean.max(f32::MIN_POSITIVE);
        metrics.merge_events += (previous - particles.len()).div_ceil(3);
    }
    if let Some(snapshot) = trace.snapshots.last_mut() {
        snapshot.particles = particles.clone();
    }
    trace.particles = particles;
}
