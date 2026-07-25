use std::{borrow::Cow, collections::BTreeMap};

use burn_automata_kernels::{AdaptivePerceptionOutput, AdaptivePerceptionPair};

use super::{
    AdaptiveCoarseDynamics, AdaptiveLocalRuleSemantics, AdaptiveNpaModel, AdaptiveParticleSet,
    features::{
        AdaptiveProxyContext, local_residual_features, local_residual_gate, local_rule_perception,
        proxy_context,
    },
    perception::{rule_perception_pair, rule_perception_without_spacing},
};
use crate::AutomataResult;

#[derive(Clone)]
pub(crate) struct LocalRawUpdate {
    pub base: Vec<f32>,
    pub combined: Vec<f32>,
}

pub(crate) struct PersistentFineQuadratureStep {
    pub local: LocalRawUpdate,
    quadrature: FineQuadratureParticles,
    raw: Vec<f32>,
    persistent_detail: bool,
}

pub(crate) fn primary_rule_features<'a>(
    model: &AdaptiveNpaModel,
    particles: &AdaptiveParticleSet,
    canonical_features: &'a [f32],
) -> AutomataResult<Cow<'a, [f32]>> {
    if !model.config.material_scale_conditioning {
        return Ok(Cow::Borrowed(canonical_features));
    }
    let input_dims = model.rule.config.perception_dims();
    let canonical_dims = input_dims.checked_sub(1).ok_or_else(|| {
        crate::AutomataError::InvalidModel(
            "material-scale-conditioned rule has no canonical perception inputs".to_owned(),
        )
    })?;
    if canonical_features.len() != particles.len() * canonical_dims {
        return Err(crate::AutomataError::InvalidModel(format!(
            "material-scale-conditioned perception has {} values, expected {} rows x {canonical_dims} inputs",
            canonical_features.len(),
            particles.len(),
        )));
    }
    let reference_footprint = model.config.reference_footprint.max(f32::MIN_POSITIVE);
    let mut features = Vec::with_capacity(particles.len() * input_dims);
    for (row, canonical) in canonical_features.chunks_exact(canonical_dims).enumerate() {
        features.extend_from_slice(canonical);
        features.push((particles.footprint(row) / reference_footprint - 1.0).clamp(-0.75, 3.0));
    }
    Ok(Cow::Owned(features))
}

#[cfg(feature = "gpu_wgpu")]
pub(crate) struct PersistentQuadratureLayout {
    pub particles: AdaptiveParticleSet,
    pub active_row: Vec<usize>,
    pub update_mask_members: Vec<Vec<(u64, f32)>>,
}

#[derive(Clone)]
struct FineQuadratureLineage {
    template_child_ids: Vec<u64>,
    mask_members: Vec<(u64, f32)>,
    previous_position: [f32; 4],
    previous_state: Vec<f32>,
}

pub(crate) fn local_raw_update(
    model: &AdaptiveNpaModel,
    particles: &AdaptiveParticleSet,
    perception: &AdaptivePerceptionPair,
) -> AutomataResult<LocalRawUpdate> {
    let rule_features = if model.uses_flat_deployment_rule()
        || model.config.rule_perception == super::AdaptiveRulePerception::NpaCompatible
    {
        perception.npa_compatible.features.as_slice()
    } else {
        perception.normalized.features.as_slice()
    };
    let primary_features = primary_rule_features(model, particles, rule_features)?;
    let base = model
        .rule
        .forward_update_from_features(primary_features.as_ref())?;
    if model.uses_flat_deployment_rule()
        && let Some(deployment_rule) = &model.deployment_rule
    {
        return Ok(LocalRawUpdate {
            base,
            combined: deployment_rule.forward_update_from_features(rule_features)?,
        });
    }
    let mut combined = base.clone();
    let output_dims = model.rule.config.update_dims();
    let local_residual_rule = model
        .deployment_local_rule
        .as_ref()
        .or(model.local_residual_rule.as_ref());
    if model.config.local_residual_scale > 0.0
        && let Some(local_residual_rule) = local_residual_rule
    {
        let residual_perception = local_rule_perception(&model.config, perception);
        let residual_features =
            local_residual_features(&model.config, particles, residual_perception)?;
        let residual = local_residual_rule.forward_update_from_features(&residual_features)?;
        for index in 0..particles.len() {
            let footprint = particles.footprint(index);
            let row_base = index * output_dims;
            for channel in 0..output_dims {
                let local = model.config.local_residual_output_scale(channel)
                    * residual[row_base + channel];
                match model.config.local_rule_semantics {
                    AdaptiveLocalRuleSemantics::Residual
                    | AdaptiveLocalRuleSemantics::NormalizedExposureResidual
                    | AdaptiveLocalRuleSemantics::CompatibleResidual => {
                        let gate = if model.config.local_rule_semantics
                            == AdaptiveLocalRuleSemantics::NormalizedExposureResidual
                        {
                            model
                                .config
                                .residual_gate(particles.footprint(index))
                                .max(perception.npa_compatible.coarse_exposure[index])
                                .max(0.0)
                        } else {
                            local_residual_gate(
                                &model.config,
                                particles,
                                residual_perception,
                                index,
                            )
                        };
                        combined[row_base + channel] +=
                            model.config.local_residual_scale * gate.max(0.0) * local;
                    }
                    AdaptiveLocalRuleSemantics::CoarseReplacement => {
                        if model.config.is_coarse_rule_footprint(footprint) {
                            let blend = model.config.local_residual_scale;
                            combined[row_base + channel] =
                                (1.0 - blend) * base[row_base + channel] + blend * local;
                        }
                    }
                }
            }
        }
    }
    Ok(LocalRawUpdate { base, combined })
}

pub(crate) fn local_raw_update_without_spacing(
    model: &AdaptiveNpaModel,
    particles: &AdaptiveParticleSet,
    perception: &AdaptivePerceptionOutput,
) -> AutomataResult<LocalRawUpdate> {
    debug_assert_eq!(
        model.config.rule_perception,
        super::AdaptiveRulePerception::NpaCompatible
    );
    debug_assert!(model.config.local_residual_scale <= 0.0);
    let primary_features = primary_rule_features(model, particles, &perception.features)?;
    let base = model
        .rule
        .forward_update_from_features(primary_features.as_ref())?;
    if model.uses_flat_deployment_rule()
        && let Some(deployment_rule) = &model.deployment_rule
    {
        return Ok(LocalRawUpdate {
            base,
            combined: deployment_rule.forward_update_from_features(primary_features.as_ref())?,
        });
    }
    Ok(LocalRawUpdate {
        combined: base.clone(),
        base,
    })
}

pub(crate) fn add_proxy_raw_update(
    model: &AdaptiveNpaModel,
    particles: &AdaptiveParticleSet,
    context: Option<&AdaptiveProxyContext>,
    update: &mut [f32],
) -> AutomataResult<()> {
    if model.uses_deployment_rule() || model.config.proxy.context_scale == 0.0 {
        return Ok(());
    }
    if let (Some(proxy_rule), Some(context)) = (&model.proxy_rule, context) {
        let proxy_update = proxy_rule.forward_update_from_features(&context.perception.features)?;
        let output_dims = model.rule.config.update_dims();
        for index in 0..particles.len() {
            let gate = model.config.residual_gate(particles.footprint(index));
            let base = index * output_dims;
            for channel in 0..output_dims {
                update[base + channel] +=
                    gate * model.config.proxy.context_scale * proxy_update[base + channel];
            }
        }
    }
    Ok(())
}

pub(crate) fn adaptive_raw_update(
    model: &AdaptiveNpaModel,
    particles: &AdaptiveParticleSet,
) -> AutomataResult<Vec<f32>> {
    if model.config.coarse_dynamics == AdaptiveCoarseDynamics::FineQuadrature {
        return fine_quadrature_raw_update(model, particles).map(|update| update.combined);
    }
    if model.config.coarse_dynamics == AdaptiveCoarseDynamics::PersistentFineQuadrature {
        return persistent_fine_quadrature_step(model, particles).map(|step| step.local.combined);
    }
    let perception = rule_perception_pair(&model.config, &model.rule, particles)?;
    let mut update = local_raw_update(model, particles, &perception)?.combined;
    let proxy = (model.config.proxy.context_scale > 0.0)
        .then(|| proxy_context(&model.config, particles))
        .transpose()?
        .flatten();
    add_proxy_raw_update(model, particles, proxy.as_ref(), &mut update)?;
    Ok(update)
}

pub(crate) fn closure_mode_raw_update(
    model: &AdaptiveNpaModel,
    particles: &AdaptiveParticleSet,
    perception: &AdaptivePerceptionPair,
) -> AutomataResult<Option<Vec<f32>>> {
    if !model.config.closure_recurrent_mode {
        return Ok(None);
    }
    let rule = model.closure_mode_rule.as_ref().ok_or_else(|| {
        crate::AutomataError::InvalidModel("recurrent closure mode has no closure rule".to_owned())
    })?;
    let rows = (0..particles.len()).collect::<Vec<_>>();
    let features = super::features::closure_recurrent_features_for_rows(
        &model.config,
        particles,
        &perception.normalized,
        &rows,
    )?;
    let output = rule.forward_update_from_features(&features)?;
    Ok(Some(output))
}

pub(crate) fn closure_basis_raw_update(
    model: &AdaptiveNpaModel,
    particles: &AdaptiveParticleSet,
    perception: &AdaptivePerceptionPair,
) -> AutomataResult<Option<Vec<f32>>> {
    if !model.config.closure_recurrent_mode {
        return Ok(None);
    }
    let Some(rule) = model.closure_basis_rule.as_ref() else {
        return Ok(None);
    };
    let rows = (0..particles.len()).collect::<Vec<_>>();
    let features = super::features::closure_recurrent_features_for_rows(
        &model.config,
        particles,
        &perception.normalized,
        &rows,
    )?;
    Ok(Some(rule.forward_update_from_features(&features)?))
}

/// Evaluates the frozen rule on the conservative fine stencil retained by
/// each coarse bootstrap leaf, then restricts physical motion and state change
/// back to the active material states.
pub(crate) fn fine_quadrature_raw_update(
    model: &AdaptiveNpaModel,
    particles: &AdaptiveParticleSet,
) -> AutomataResult<LocalRawUpdate> {
    fine_quadrature_step(model, particles, false).map(|step| step.local)
}

pub(crate) fn persistent_fine_quadrature_step(
    model: &AdaptiveNpaModel,
    particles: &AdaptiveParticleSet,
) -> AutomataResult<PersistentFineQuadratureStep> {
    fine_quadrature_step(model, particles, true)
}

pub(crate) fn persistent_quadrature_particle_set(
    model: &AdaptiveNpaModel,
    particles: &AdaptiveParticleSet,
) -> AutomataResult<AdaptiveParticleSet> {
    fine_quadrature_particles(model, particles, true).map(|quadrature| quadrature.set)
}

pub(crate) fn quadrature_particle_count(
    model: &AdaptiveNpaModel,
    particles: &AdaptiveParticleSet,
    persistent_detail: bool,
) -> AutomataResult<usize> {
    fine_quadrature_particles(model, particles, persistent_detail)
        .map(|quadrature| quadrature.set.len())
}

#[cfg(feature = "gpu_wgpu")]
pub(crate) fn quadrature_layout_with_points(
    model: &AdaptiveNpaModel,
    particles: &AdaptiveParticleSet,
    persistent_detail: bool,
    coarse_quadrature_points: usize,
) -> AutomataResult<PersistentQuadratureLayout> {
    let quadrature = fine_quadrature_particles_with_points(
        model,
        particles,
        persistent_detail,
        coarse_quadrature_points,
    )?;
    Ok(PersistentQuadratureLayout {
        particles: quadrature.set,
        active_row: quadrature.active_row,
        update_mask_members: quadrature
            .lineage
            .into_iter()
            .map(|lineage| lineage.mask_members)
            .collect(),
    })
}

pub(crate) fn fine_quadrature_step(
    model: &AdaptiveNpaModel,
    particles: &AdaptiveParticleSet,
    persistent_detail: bool,
) -> AutomataResult<PersistentFineQuadratureStep> {
    if model.config.local_residual_scale > 0.0
        || model.config.proxy.context_scale > 0.0
        || model.uses_deployment_rule()
    {
        return Err(crate::AutomataError::InvalidArgument(
            "fine-quadrature controls require the unmodified base rule".to_string(),
        ));
    }
    let quadrature = fine_quadrature_particles(model, particles, persistent_detail)?;
    let perception = rule_perception_without_spacing(&model.config, &model.rule, &quadrature.set)?;
    let raw = model
        .rule
        .forward_update_from_features(&perception.features)?;
    let spatial_dims = particles.spatial_dims;
    let state_dims = particles.state_dims;
    let output_dims = model.rule.config.update_dims();
    let mut restricted_dx = vec![0.0_f32; particles.len() * spatial_dims];
    let mut restricted_ds = vec![0.0_f32; particles.len() * state_dims];
    let mut restricted_measure = vec![0.0_f32; particles.len()];
    for virtual_row in 0..quadrature.set.len() {
        let active = quadrature.active_row[virtual_row];
        let measure = quadrature.set.represented_measure[virtual_row];
        restricted_measure[active] += measure;
        let output = &raw[virtual_row * output_dims..(virtual_row + 1) * output_dims];
        let motion_norm = output[..spatial_dims]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let motion_scale = model.rule.config.alpha
            * model
                .rule
                .config
                .motion_eps(quadrature.set.bandwidth[virtual_row])
            / (1.0 + motion_norm);
        for axis in 0..spatial_dims {
            restricted_dx[active * spatial_dims + axis] += measure * motion_scale * output[axis];
        }
        for channel in 0..state_dims {
            restricted_ds[active * state_dims + channel] +=
                measure * output[spatial_dims + channel];
        }
    }
    for active in 0..particles.len() {
        let inverse = restricted_measure[active].max(f32::MIN_POSITIVE).recip();
        for axis in 0..spatial_dims {
            restricted_dx[active * spatial_dims + axis] *= inverse;
        }
        for channel in 0..state_dims {
            restricted_ds[active * state_dims + channel] *= inverse;
        }
    }
    let restricted = raw_update_from_physical_step(
        &restricted_dx,
        &restricted_ds,
        &particles.bandwidth,
        &model.rule,
    );
    Ok(PersistentFineQuadratureStep {
        local: LocalRawUpdate {
            base: restricted.clone(),
            combined: restricted,
        },
        quadrature,
        raw,
        persistent_detail,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn integrate_fine_quadrature(
    model: &AdaptiveNpaModel,
    particles: &mut AdaptiveParticleSet,
    mut step: PersistentFineQuadratureStep,
    rollout_seed: u64,
    absolute_step: usize,
    update_prob: f32,
    dt: f32,
) -> AutomataResult<f32> {
    let spatial_dims = particles.spatial_dims;
    let state_dims = particles.state_dims;
    let output_dims = model.rule.config.update_dims();
    let old_positions = particles.positions.clone();
    for row in 0..step.quadrature.set.len() {
        let mask = step.quadrature.lineage[row]
            .mask_members
            .iter()
            .map(|(id, weight)| {
                weight
                    * f32::from(
                        crate::rollout::stable_material_uniform(rollout_seed, absolute_step, *id)
                            < update_prob,
                    )
            })
            .sum::<f32>();
        let output = &step.raw[row * output_dims..(row + 1) * output_dims];
        let motion_norm = output[..spatial_dims]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let motion_scale = model.rule.config.alpha
            * model
                .rule
                .config
                .motion_eps(step.quadrature.set.bandwidth[row])
            / (1.0 + motion_norm);
        for (axis, &delta) in output.iter().enumerate().take(spatial_dims) {
            step.quadrature.set.positions[row][axis] = (step.quadrature.set.positions[row][axis]
                + mask * dt * motion_scale * delta)
                .clamp(model.config.domain_min[axis], model.config.domain_max[axis]);
        }
        for channel in 0..state_dims {
            step.quadrature.set.states[row * state_dims + channel] +=
                mask * dt * output[spatial_dims + channel];
        }
    }

    let mut measure = vec![0.0_f32; particles.len()];
    let mut position = vec![[0.0_f32; 4]; particles.len()];
    let mut state = vec![0.0_f32; particles.len() * state_dims];
    let mut bandwidth = vec![0.0_f32; particles.len()];
    for row in 0..step.quadrature.set.len() {
        let active = step.quadrature.active_row[row];
        let weight = step.quadrature.set.represented_measure[row];
        measure[active] += weight;
        for (axis, value) in position[active].iter_mut().enumerate().take(spatial_dims) {
            *value += weight * step.quadrature.set.positions[row][axis];
        }
        for channel in 0..state_dims {
            state[active * state_dims + channel] +=
                weight * step.quadrature.set.states[row * state_dims + channel];
        }
        bandwidth[active] += weight * step.quadrature.set.bandwidth[row];
    }
    for active in 0..particles.len() {
        let inverse = measure[active].max(f32::MIN_POSITIVE).recip();
        for value in position[active].iter_mut().take(spatial_dims) {
            *value *= inverse;
        }
        for channel in 0..state_dims {
            state[active * state_dims + channel] *= inverse;
        }
        bandwidth[active] *= inverse;
    }
    let mut covariance = vec![[0.0_f32; 9]; particles.len()];
    for row in 0..step.quadrature.set.len() {
        let active = step.quadrature.active_row[row];
        let normalized_weight =
            step.quadrature.set.represented_measure[row] / measure[active].max(f32::MIN_POSITIVE);
        for out_axis in 0..spatial_dims {
            let out_delta =
                step.quadrature.set.positions[row][out_axis] - position[active][out_axis];
            for in_axis in 0..spatial_dims {
                let in_delta =
                    step.quadrature.set.positions[row][in_axis] - position[active][in_axis];
                covariance[active][out_axis * 3 + in_axis] += normalized_weight
                    * (step.quadrature.set.covariance[row][out_axis * 3 + in_axis]
                        + out_delta * in_delta);
            }
        }
    }

    if step.persistent_detail {
        let mut child_updates = BTreeMap::new();
        for (row, lineage) in step.quadrature.lineage.iter().enumerate() {
            if lineage.template_child_ids.is_empty() {
                continue;
            }
            let mut position_delta = [0.0_f32; 4];
            for (axis, value) in position_delta.iter_mut().enumerate().take(spatial_dims) {
                *value = step.quadrature.set.positions[row][axis] - lineage.previous_position[axis];
            }
            let state_delta = (0..state_dims)
                .map(|channel| {
                    step.quadrature.set.states[row * state_dims + channel]
                        - lineage.previous_state[channel]
                })
                .collect::<Vec<_>>();
            for id in &lineage.template_child_ids {
                child_updates.insert(*id, (position_delta, state_delta.clone()));
            }
        }
        for template in &mut particles.bootstrap_templates {
            for child in &mut template.children {
                let (position_delta, state_delta) =
                    child_updates.get(&child.particle_id).ok_or_else(|| {
                        crate::AutomataError::InvalidModel(format!(
                            "persistent quadrature child {} disappeared during integration",
                            child.particle_id,
                        ))
                    })?;
                for (axis, &delta) in position_delta.iter().enumerate().take(spatial_dims) {
                    child.position[axis] += delta;
                }
                for (channel, &delta) in state_delta.iter().enumerate().take(state_dims) {
                    child.state[channel] += delta;
                }
            }
        }
    }

    particles.positions = position;
    particles.states = state;
    particles.bandwidth = bandwidth;
    particles.covariance = covariance;
    let displacement_sum = particles
        .positions
        .iter()
        .zip(old_positions)
        .map(|(current, previous)| {
            (0..spatial_dims)
                .map(|axis| (current[axis] - previous[axis]).powi(2))
                .sum::<f32>()
                .sqrt()
        })
        .sum();
    particles.validate()?;
    Ok(displacement_sum)
}

struct FineQuadratureParticles {
    set: AdaptiveParticleSet,
    active_row: Vec<usize>,
    lineage: Vec<FineQuadratureLineage>,
}

fn fine_quadrature_particles(
    model: &AdaptiveNpaModel,
    particles: &AdaptiveParticleSet,
    persistent_detail: bool,
) -> AutomataResult<FineQuadratureParticles> {
    fine_quadrature_particles_with_points(
        model,
        particles,
        persistent_detail,
        model.config.coarse_quadrature_points,
    )
}

fn fine_quadrature_particles_with_points(
    model: &AdaptiveNpaModel,
    particles: &AdaptiveParticleSet,
    persistent_detail: bool,
    coarse_quadrature_points: usize,
) -> AutomataResult<FineQuadratureParticles> {
    let templates = particles
        .bootstrap_templates
        .iter()
        .map(|template| (template.parent_id, template))
        .collect::<BTreeMap<_, _>>();
    let native_measure =
        particles.total_measure() as f32 / model.config.bootstrap_fine_leaf_count().max(1) as f32;
    let capacity = model.config.bootstrap_fine_leaf_count();
    let mut set = AdaptiveParticleSet {
        spatial_dims: particles.spatial_dims,
        state_dims: particles.state_dims,
        positions: Vec::with_capacity(capacity),
        states: Vec::with_capacity(capacity * particles.state_dims),
        state_jacobian: Vec::with_capacity(
            capacity * particles.state_dims * particles.spatial_dims,
        ),
        closure_mode: Vec::with_capacity(capacity * particles.state_dims),
        closure_basis: Vec::with_capacity(capacity * 4),
        closure_phase: Vec::with_capacity(capacity * 2),
        represented_measure: Vec::with_capacity(capacity),
        render_footprint: Vec::with_capacity(capacity),
        bandwidth: Vec::with_capacity(capacity),
        covariance: Vec::with_capacity(capacity),
        particle_id: Vec::with_capacity(capacity),
        sibling_group: Vec::with_capacity(capacity),
        generation: Vec::with_capacity(capacity),
        cooldown: Vec::with_capacity(capacity),
        next_id: particles.next_id,
        next_sibling_group: particles.next_sibling_group,
        bootstrap_templates: Vec::new(),
    };
    let mut active_row = Vec::with_capacity(capacity);
    let mut lineage = Vec::with_capacity(capacity);
    for active in 0..particles.len() {
        let state =
            &particles.states[active * particles.state_dims..(active + 1) * particles.state_dims];
        if let Some(template) = templates.get(&particles.particle_id[active]) {
            let reference_position =
                weighted_child_position(&template.children, particles.spatial_dims);
            let jacobian_dims = particles.state_dims * particles.spatial_dims;
            let state_jacobian =
                &particles.state_jacobian[active * jacobian_dims..(active + 1) * jacobian_dims];
            let mode_count = if coarse_quadrature_points == 0 {
                template.children.len()
            } else {
                coarse_quadrature_points.min(template.children.len())
            };
            for mode in 0..mode_count {
                let start = mode * template.children.len() / mode_count;
                let end = (mode + 1) * template.children.len() / mode_count;
                push_template_quadrature_mode(
                    &mut set,
                    &mut active_row,
                    &mut lineage,
                    active,
                    &template.children[start..end],
                    particles.positions[active],
                    reference_position,
                    state,
                    state_jacobian,
                    persistent_detail,
                    particles.spatial_dims,
                );
            }
        } else {
            if particles.represented_measure[active] > native_measure * 1.5 {
                return Err(crate::AutomataError::InvalidModel(format!(
                    "coarse adaptive leaf {} has no fine quadrature template",
                    particles.particle_id[active],
                )));
            }
            push_quadrature_row(
                &mut set,
                &mut active_row,
                active,
                particles.positions[active],
                state,
                particles.represented_measure[active],
                particles.bandwidth[active],
                particles.covariance[active],
                particles.particle_id[active],
                particles.generation[active],
            );
            lineage.push(FineQuadratureLineage {
                template_child_ids: Vec::new(),
                mask_members: vec![(particles.particle_id[active], 1.0)],
                previous_position: particles.positions[active],
                previous_state: state.to_vec(),
            });
        }
    }
    set.next_id = set
        .particle_id
        .iter()
        .copied()
        .max()
        .unwrap_or_default()
        .saturating_add(1);
    set.validate()?;
    debug_assert_eq!(set.len(), lineage.len());
    Ok(FineQuadratureParticles {
        set,
        active_row,
        lineage,
    })
}

#[allow(clippy::too_many_arguments)]
fn push_template_quadrature_mode(
    set: &mut AdaptiveParticleSet,
    active_row: &mut Vec<usize>,
    lineage: &mut Vec<FineQuadratureLineage>,
    active: usize,
    children: &[super::AdaptiveBootstrapChild],
    active_position: [f32; 4],
    reference_position: [f32; 4],
    active_state: &[f32],
    active_state_jacobian: &[f32],
    persistent_detail: bool,
    spatial_dims: usize,
) {
    let measure = children
        .iter()
        .map(|child| child.represented_measure)
        .sum::<f32>()
        .max(f32::MIN_POSITIVE);
    let mut position = [0.0_f32; 4];
    let mut state = vec![0.0_f32; set.state_dims];
    let mut bandwidth = 0.0_f32;
    for child in children {
        let weight = child.represented_measure / measure;
        for (axis, value) in position.iter_mut().enumerate().take(spatial_dims) {
            *value += weight * child.position[axis];
        }
        if persistent_detail {
            for (channel, value) in state.iter_mut().enumerate() {
                *value += weight * child.state[channel];
            }
        }
        bandwidth += weight * child.bandwidth;
    }
    let mode_reference_position = position;
    if !persistent_detail {
        for axis in 0..spatial_dims {
            position[axis] = active_position[axis] + position[axis] - reference_position[axis];
        }
        for channel in 0..set.state_dims {
            state[channel] = active_state[channel]
                + (0..spatial_dims)
                    .map(|axis| {
                        active_state_jacobian[channel * spatial_dims + axis]
                            * (position[axis] - active_position[axis])
                    })
                    .sum::<f32>();
        }
    }
    let mut covariance = [0.0_f32; 9];
    for child in children {
        let weight = child.represented_measure / measure;
        for row in 0..spatial_dims {
            let row_delta = child.position[row] - mode_reference_position[row];
            for col in 0..spatial_dims {
                let col_delta = child.position[col] - mode_reference_position[col];
                covariance[row * 3 + col] +=
                    weight * (child.covariance[row * 3 + col] + row_delta * col_delta);
            }
        }
    }
    let particle_id = children
        .iter()
        .map(|child| child.particle_id)
        .min()
        .expect("quadrature mode has at least one child");
    let generation = children
        .iter()
        .map(|child| child.generation)
        .max()
        .unwrap_or_default();
    push_quadrature_row(
        set,
        active_row,
        active,
        position,
        &state,
        measure,
        bandwidth,
        covariance,
        particle_id,
        generation,
    );
    lineage.push(FineQuadratureLineage {
        template_child_ids: children.iter().map(|child| child.particle_id).collect(),
        mask_members: children
            .iter()
            .map(|child| (child.particle_id, child.represented_measure / measure))
            .collect(),
        previous_position: position,
        previous_state: state,
    });
}

fn weighted_child_position(
    children: &[super::AdaptiveBootstrapChild],
    spatial_dims: usize,
) -> [f32; 4] {
    let measure = children
        .iter()
        .map(|child| child.represented_measure)
        .sum::<f32>()
        .max(f32::MIN_POSITIVE);
    let mut position = [0.0_f32; 4];
    for child in children {
        let weight = child.represented_measure / measure;
        for (axis, value) in position.iter_mut().enumerate().take(spatial_dims) {
            *value += weight * child.position[axis];
        }
    }
    position
}

#[allow(clippy::too_many_arguments)]
fn push_quadrature_row(
    set: &mut AdaptiveParticleSet,
    active_row: &mut Vec<usize>,
    active: usize,
    position: [f32; 4],
    state: &[f32],
    measure: f32,
    bandwidth: f32,
    covariance: [f32; 9],
    particle_id: u64,
    generation: u16,
) {
    set.positions.push(position);
    set.states.extend_from_slice(state);
    set.state_jacobian
        .extend(std::iter::repeat_n(0.0, set.state_dims * set.spatial_dims));
    set.closure_mode
        .extend(std::iter::repeat_n(0.0, set.state_dims));
    set.closure_basis.extend(std::iter::repeat_n(0.0, 4));
    set.closure_phase.extend(std::iter::repeat_n(0.0, 2));
    set.represented_measure.push(measure);
    set.render_footprint
        .push(super::material_footprint_radius(measure, set.spatial_dims));
    set.bandwidth.push(bandwidth);
    set.covariance.push(covariance);
    set.particle_id.push(particle_id);
    set.sibling_group.push(0);
    set.generation.push(generation);
    set.cooldown.push(0);
    active_row.push(active);
}

fn raw_update_from_physical_step(
    dx: &[f32],
    ds: &[f32],
    bandwidth: &[f32],
    rule: &crate::NpaModel,
) -> Vec<f32> {
    let rows = bandwidth.len();
    let spatial_dims = rule.config.spatial_dims;
    let output_dims = rule.config.update_dims();
    let mut output = vec![0.0; rows * output_dims];
    for row in 0..rows {
        let spatial = &dx[row * spatial_dims..(row + 1) * spatial_dims];
        let norm = spatial
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let scale =
            (rule.config.alpha * rule.config.motion_eps(bandwidth[row])).max(f32::MIN_POSITIVE);
        let normalized = (norm / scale).clamp(0.0, 0.999);
        let raw_norm = normalized / (1.0 - normalized).max(1.0e-4);
        for axis in 0..spatial_dims {
            output[row * output_dims + axis] = if norm > 1.0e-12 {
                spatial[axis] * raw_norm / norm
            } else {
                0.0
            };
        }
        output[row * output_dims + spatial_dims..(row + 1) * output_dims]
            .copy_from_slice(&ds[row * rule.config.state_dims..(row + 1) * rule.config.state_dims]);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AdaptiveNpaConfig, NpaConfig, NpaModel, ParticleSeed,
        adaptive::{apply_adaptive_topology_at_step, seed_adaptive_particles_scaled},
    };

    fn model(mut adaptive: AdaptiveNpaConfig) -> AdaptiveNpaModel {
        adaptive.local_residual_scale = 0.0;
        adaptive.proxy.context_scale = 0.0;
        AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 7), adaptive, 11)
            .unwrap()
    }

    #[test]
    fn coarse_replacement_preserves_native_rows_and_blends_only_coarse_rows() {
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.local_rule_semantics = AdaptiveLocalRuleSemantics::CoarseReplacement;
        adaptive.local_residual_scale = 0.25;
        adaptive.proxy.enabled = false;
        adaptive.min_leaves = 40;
        adaptive.initial_leaves = 40;
        adaptive.target_leaves = 40;
        adaptive.max_leaves = 64;
        adaptive.bootstrap_fine_leaves = 64;
        adaptive.hierarchical_bootstrap_seed = true;
        let base = NpaModel::seeded(NpaConfig::growing_2d(), 47);
        let mut model = AdaptiveNpaModel::seeded(base, adaptive, 53).unwrap();
        model.enable_base_initialized_local_rule().unwrap();
        model.local_residual_rule.as_mut().unwrap().weights.b2[0] += 0.5;
        let particles = seed_adaptive_particles_scaled(
            &model,
            40,
            59,
            ParticleSeed::UniformCircle,
            0.2,
            std::f32::consts::PI * 0.2_f32.powi(2),
            0.1,
        )
        .unwrap();
        let perception = rule_perception_pair(&model.config, &model.rule, &particles).unwrap();
        let local_prediction = model
            .local_residual_rule
            .as_ref()
            .unwrap()
            .forward_update_from_features(
                &local_residual_features(&model.config, &particles, &perception.normalized)
                    .unwrap(),
            )
            .unwrap();
        let update = local_raw_update(&model, &particles, &perception).unwrap();
        let output_dims = model.rule.config.update_dims();
        let mut fine_rows = 0;
        let mut coarse_rows = 0;
        for row in 0..particles.len() {
            let range = row * output_dims..(row + 1) * output_dims;
            if model
                .config
                .is_coarse_rule_footprint(particles.footprint(row))
            {
                coarse_rows += 1;
                for channel in range.clone() {
                    let expected = 0.75 * update.base[channel] + 0.25 * local_prediction[channel];
                    assert!((update.combined[channel] - expected).abs() <= 1.0e-7);
                }
            } else {
                fine_rows += 1;
                assert_eq!(&update.combined[range.clone()], &update.base[range]);
            }
        }
        assert!(fine_rows > 0 && coarse_rows > 0);
    }

    #[test]
    fn compatible_residual_tracks_local_coarse_source_exposure() {
        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let fine_footprint = super::super::material_footprint_radius(total_measure / 64.0, 2);
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.local_rule_semantics = AdaptiveLocalRuleSemantics::CompatibleResidual;
        adaptive.reference_footprint = fine_footprint;
        adaptive.base_rule_footprint = fine_footprint;
        adaptive.proxy.enabled = false;
        adaptive.min_leaves = 40;
        adaptive.initial_leaves = 40;
        adaptive.target_leaves = 40;
        adaptive.max_leaves = 64;
        adaptive.bootstrap_fine_leaves = 64;
        adaptive.hierarchical_bootstrap_seed = true;
        let base = NpaModel::seeded(NpaConfig::growing_2d(), 61);
        let mut model = AdaptiveNpaModel::seeded(base, adaptive, 67).unwrap();
        model.enable_zero_local_residual_rule().unwrap();
        model.local_residual_rule.as_mut().unwrap().weights.b2[0] = 0.5;
        let mixed = seed_adaptive_particles_scaled(
            &model,
            40,
            71,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            0.1,
        )
        .unwrap();
        let mixed_perception = rule_perception_pair(&model.config, &model.rule, &mixed).unwrap();
        let mixed_update = local_raw_update(&model, &mixed, &mixed_perception).unwrap();
        let output_dims = model.rule.config.update_dims();
        let mut exposed_rows = 0;
        for row in 0..mixed.len() {
            let index = row * output_dims;
            let exposure = mixed_perception.npa_compatible.coarse_exposure[row];
            assert!(
                (mixed_update.combined[index] - mixed_update.base[index] - 0.5 * exposure).abs()
                    < 1.0e-6
            );
            exposed_rows += usize::from(exposure > 1.0e-6);
        }
        assert!(exposed_rows > 0);

        let mut uniform = mixed;
        let fine_measure = total_measure / 64.0;
        uniform.represented_measure.fill(fine_measure);
        uniform.bandwidth.fill(0.1);
        let uniform_perception =
            rule_perception_pair(&model.config, &model.rule, &uniform).unwrap();
        assert!(
            uniform_perception
                .npa_compatible
                .coarse_exposure
                .iter()
                .all(|value| value.abs() <= f32::EPSILON)
        );
        let uniform_update = local_raw_update(&model, &uniform, &uniform_perception).unwrap();
        assert_eq!(uniform_update.combined, uniform_update.base);
    }

    #[test]
    fn material_conditioned_residual_features_are_row_local_and_zero_output_safe() {
        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let fine_footprint = super::super::material_footprint_radius(total_measure / 64.0, 2);
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.reference_footprint = fine_footprint;
        adaptive.base_rule_footprint = fine_footprint;
        adaptive.proxy.enabled = false;
        adaptive.min_leaves = 40;
        adaptive.initial_leaves = 40;
        adaptive.target_leaves = 40;
        adaptive.max_leaves = 64;
        adaptive.bootstrap_fine_leaves = 64;
        adaptive.hierarchical_bootstrap_seed = true;
        let base = NpaModel::seeded(NpaConfig::growing_2d(), 73);
        let mut model = AdaptiveNpaModel::seeded(base, adaptive, 79).unwrap();
        model
            .enable_material_conditioned_compatible_residual_rule()
            .unwrap();
        let particles = seed_adaptive_particles_scaled(
            &model,
            40,
            83,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            0.1,
        )
        .unwrap();
        let perception = rule_perception_pair(&model.config, &model.rule, &particles).unwrap();
        let features =
            local_residual_features(&model.config, &particles, &perception.npa_compatible).unwrap();
        let canonical_dims = perception.npa_compatible.feature_dims;
        let input_dims = canonical_dims + 2;
        assert_eq!(features.len(), particles.len() * input_dims);
        for row in 0..particles.len() {
            let base = row * input_dims + canonical_dims;
            let expected_scale =
                (particles.footprint(row) / fine_footprint - 1.0).clamp(-0.75, 3.0);
            assert!((features[base] - expected_scale).abs() <= 1.0e-6);
            assert_eq!(
                features[base + 1],
                perception.npa_compatible.coarse_exposure[row],
            );
        }
        let update = local_raw_update(&model, &particles, &perception).unwrap();
        assert_eq!(update.combined, update.base);
    }

    #[test]
    fn fine_quadrature_is_exactly_native_when_no_leaf_is_coarse() {
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.min_leaves = 16;
        adaptive.max_leaves = 16;
        adaptive.target_leaves = 16;
        adaptive.initial_leaves = 16;
        adaptive.bootstrap_fine_leaves = 16;
        adaptive.bootstrap_end_step = 0;
        let model = model(adaptive);
        let particles = seed_adaptive_particles_scaled(
            &model,
            16,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            std::f32::consts::PI * 0.2_f32.powi(2),
            0.1,
        )
        .unwrap();
        let perception = rule_perception_pair(&model.config, &model.rule, &particles).unwrap();
        let expected = local_raw_update(&model, &particles, &perception).unwrap();
        let actual = fine_quadrature_raw_update(&model, &particles).unwrap();
        let max_error = actual
            .base
            .iter()
            .zip(&expected.base)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_error <= 1.0e-6, "native quadrature error {max_error}");
        assert_eq!(actual.base, actual.combined);
    }

    #[test]
    fn partial_hierarchy_quadrature_preserves_fine_rows_and_material() {
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.min_leaves = 16;
        adaptive.initial_leaves = 16;
        adaptive.target_leaves = 40;
        adaptive.max_leaves = 64;
        adaptive.bootstrap_fine_leaves = 64;
        adaptive.bootstrap_end_step = 1;
        adaptive.bootstrap_events_per_interval = 8;
        adaptive.max_events_per_interval = 8;
        let model = model(adaptive);
        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let mut particles = seed_adaptive_particles_scaled(
            &model,
            16,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            0.1,
        )
        .unwrap();
        apply_adaptive_topology_at_step(&model, &mut particles, 1, 0).unwrap();
        let quadrature = fine_quadrature_particles(&model, &particles, false).unwrap();

        assert_eq!(particles.len(), 40);
        assert_eq!(quadrature.set.len(), 64);
        assert!((quadrature.set.total_measure() - particles.total_measure()).abs() <= 1.0e-7);
        let mut restricted = vec![0.0_f64; particles.len()];
        for (row, active) in quadrature.active_row.iter().copied().enumerate() {
            restricted[active] += quadrature.set.represented_measure[row] as f64;
        }
        for (actual, expected) in restricted.iter().zip(&particles.represented_measure) {
            assert!((*actual - *expected as f64).abs() <= 1.0e-8);
        }
    }

    #[test]
    fn active_leaf_quadrature_recenters_positions_and_affine_state() {
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.min_leaves = 40;
        adaptive.initial_leaves = 40;
        adaptive.target_leaves = 40;
        adaptive.max_leaves = 64;
        adaptive.bootstrap_fine_leaves = 64;
        adaptive.hierarchical_bootstrap_seed = true;
        let model = model(adaptive);
        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let mut particles = seed_adaptive_particles_scaled(
            &model,
            40,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            0.1,
        )
        .unwrap();
        let parent_id = particles.bootstrap_templates[0].parent_id;
        let active = particles
            .particle_id
            .iter()
            .position(|id| *id == parent_id)
            .unwrap();
        particles.positions[active][0] += 0.37;
        particles.positions[active][1] -= 0.21;
        for channel in 0..particles.state_dims {
            particles.states[active * particles.state_dims + channel] = channel as f32 * 0.1;
        }
        let jacobian_base = active * particles.state_dims * particles.spatial_dims;
        particles.state_jacobian[jacobian_base] = 2.0;
        particles.state_jacobian[jacobian_base + 1] = -1.0;

        let quadrature = fine_quadrature_particles(&model, &particles, false).unwrap();
        let rows = quadrature
            .active_row
            .iter()
            .enumerate()
            .filter_map(|(row, owner)| (*owner == active).then_some(row))
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 4);
        let parent_measure = particles.represented_measure[active];
        let mut mean_position = [0.0_f32; 2];
        let mut mean_state = vec![0.0_f32; particles.state_dims];
        for row in &rows {
            let weight = quadrature.set.represented_measure[*row] / parent_measure;
            for (axis, value) in mean_position.iter_mut().enumerate() {
                *value += weight * quadrature.set.positions[*row][axis];
            }
            for (channel, value) in mean_state.iter_mut().enumerate() {
                *value += weight * quadrature.set.states[*row * particles.state_dims + channel];
            }
        }
        for (mean, expected) in mean_position.iter().zip(&particles.positions[active]) {
            assert!((mean - expected).abs() <= 1.0e-6);
        }
        for (mean, expected) in mean_state.iter().zip(
            &particles.states[active * particles.state_dims..(active + 1) * particles.state_dims],
        ) {
            assert!((mean - expected).abs() <= 1.0e-6);
        }
        assert!(rows.iter().any(|row| {
            (quadrature.set.states[*row * particles.state_dims]
                - particles.states[active * particles.state_dims])
                .abs()
                > 1.0e-5
        }));
    }

    #[test]
    fn compressed_quadrature_uses_two_modes_per_coarse_leaf_conservatively() {
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.min_leaves = 40;
        adaptive.initial_leaves = 40;
        adaptive.target_leaves = 40;
        adaptive.max_leaves = 64;
        adaptive.bootstrap_fine_leaves = 64;
        adaptive.hierarchical_bootstrap_seed = true;
        adaptive.coarse_quadrature_points = 2;
        let model = model(adaptive);
        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let particles = seed_adaptive_particles_scaled(
            &model,
            40,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            0.1,
        )
        .unwrap();
        let quadrature = fine_quadrature_particles(&model, &particles, true).unwrap();

        assert_eq!(particles.bootstrap_templates.len(), 8);
        assert_eq!(quadrature.set.len(), 32 + 8 * 2);
        assert!((quadrature.set.total_measure() - particles.total_measure()).abs() <= 1.0e-7);
        let mut restricted = vec![0.0_f64; particles.len()];
        for (row, active) in quadrature.active_row.iter().copied().enumerate() {
            restricted[active] += quadrature.set.represented_measure[row] as f64;
        }
        for (actual, expected) in restricted.iter().zip(&particles.represented_measure) {
            assert!((*actual - f64::from(*expected)).abs() <= 1.0e-8);
        }
    }

    #[test]
    fn persistent_quadrature_matches_a_full_fine_step_and_exact_restriction() {
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.min_leaves = 16;
        adaptive.initial_leaves = 16;
        adaptive.target_leaves = 40;
        adaptive.max_leaves = 64;
        adaptive.bootstrap_fine_leaves = 64;
        adaptive.bootstrap_end_step = 1;
        adaptive.bootstrap_events_per_interval = 8;
        adaptive.max_events_per_interval = 8;
        adaptive.coarse_dynamics = AdaptiveCoarseDynamics::PersistentFineQuadrature;
        let model = model(adaptive);
        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let mut mixed = seed_adaptive_particles_scaled(
            &model,
            16,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            0.1,
        )
        .unwrap();
        apply_adaptive_topology_at_step(&model, &mut mixed, 1, 0).unwrap();
        let mut fine = seed_adaptive_particles_scaled(
            &model,
            64,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            0.1,
        )
        .unwrap();

        let rollout_seed = 29;
        let absolute_step = 7;
        let update_prob = 0.73;
        let dt = 0.8;
        let mixed_step = persistent_fine_quadrature_step(&model, &mixed).unwrap();
        integrate_fine_quadrature(
            &model,
            &mut mixed,
            mixed_step,
            rollout_seed,
            absolute_step,
            update_prob,
            dt,
        )
        .unwrap();
        let fine_step = persistent_fine_quadrature_step(&model, &fine).unwrap();
        integrate_fine_quadrature(
            &model,
            &mut fine,
            fine_step,
            rollout_seed,
            absolute_step,
            update_prob,
            dt,
        )
        .unwrap();

        let mixed_fine = fine_quadrature_particles(&model, &mixed, true).unwrap();
        assert_eq!(mixed.len(), 40);
        assert_eq!(mixed_fine.set.len(), 64);
        let fine_rows = fine
            .particle_id
            .iter()
            .copied()
            .enumerate()
            .map(|(row, id)| (id, row))
            .collect::<BTreeMap<_, _>>();
        let mut max_position_error = 0.0_f32;
        let mut max_state_error = 0.0_f32;
        for row in 0..mixed_fine.set.len() {
            let fine_row = fine_rows[&mixed_fine.set.particle_id[row]];
            for axis in 0..mixed.spatial_dims {
                max_position_error = max_position_error.max(
                    (mixed_fine.set.positions[row][axis] - fine.positions[fine_row][axis]).abs(),
                );
            }
            for channel in 0..mixed.state_dims {
                max_state_error = max_state_error.max(
                    (mixed_fine.set.states[row * mixed.state_dims + channel]
                        - fine.states[fine_row * fine.state_dims + channel])
                        .abs(),
                );
            }
        }
        assert!(
            max_position_error <= 1.0e-6,
            "persistent fine position error {max_position_error}"
        );
        assert!(
            max_state_error <= 1.0e-6,
            "persistent fine state error {max_state_error}"
        );
        assert!((mixed.total_measure() - fine.total_measure()).abs() <= 1.0e-7);

        for active in 0..mixed.len() {
            let rows = mixed_fine
                .active_row
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(row, owner)| (owner == active).then_some(row))
                .collect::<Vec<_>>();
            let measure = rows
                .iter()
                .map(|row| mixed_fine.set.represented_measure[*row])
                .sum::<f32>();
            assert!((measure - mixed.represented_measure[active]).abs() <= 1.0e-7);
            for axis in 0..mixed.spatial_dims {
                let expected = rows
                    .iter()
                    .map(|row| {
                        mixed_fine.set.represented_measure[*row]
                            * mixed_fine.set.positions[*row][axis]
                    })
                    .sum::<f32>()
                    / measure;
                assert!((expected - mixed.positions[active][axis]).abs() <= 1.0e-6);
            }
            for channel in 0..mixed.state_dims {
                let expected = rows
                    .iter()
                    .map(|row| {
                        mixed_fine.set.represented_measure[*row]
                            * mixed_fine.set.states[*row * mixed.state_dims + channel]
                    })
                    .sum::<f32>()
                    / measure;
                assert!(
                    (expected - mixed.states[active * mixed.state_dims + channel]).abs() <= 1.0e-6
                );
            }
        }
    }
}
