use std::collections::BTreeMap;

use burn_automata_kernels::{AdaptiveGraphPolicy, HashGridConfig};

use super::{
    AdaptiveCoarseDynamics, AdaptiveHierarchyMember, AdaptiveHierarchyRestrictionPolicy,
    AdaptiveNpaModel, AdaptiveParticleSet, AdaptiveProxyHierarchy, AdaptiveRestrictionArity,
    AdaptiveRestrictionSchedule, AdaptiveTopologyControl, AdaptiveTopologyUpdate,
    adaptive_display_scale_per_footprint,
    dynamics::{PersistentQuadratureLayout, quadrature_layout_with_points},
    restriction::learned_level_one_merge_costs_from_precomputed,
    rollout::{
        apply_adaptive_topology_at_step_with_control, apply_hierarchical_bootstrap_refinement,
        apply_resident_canonical_bootstrap_refinement,
    },
    seed::{
        adaptive_particle_subset, adaptive_template_child_groups,
        progressively_restrict_adaptive_particles_to_leaf_budget,
        restore_adaptive_particles_from_templates,
        restrict_adaptive_particles_to_target_by_merge_cost_with_hierarchy,
        restricted_seed_from_fine_groups,
    },
};
use crate::{
    AutomataError, AutomataResult,
    gpu::{
        WGPU_MATERIAL_UPDATE_MASK_MEMBERS, WgpuAdaptiveDiagnostics, WgpuAutomataExecutor,
        WgpuAutomataState, WgpuGaussianBindGroup, WgpuMaterialStateInit, WgpuMaterialUpdateMask,
        WgpuNeighborMode, WgpuPersistentModeRestriction, WgpuStatePca, WgpuStatePcaConfig,
        WgpuSupportBinConfig,
    },
};

pub struct WgpuAdaptiveNpaState {
    pub particles: AdaptiveParticleSet,
    pub resident: WgpuAutomataState,
    pub completed_steps: usize,
    pub last_topology: Option<AdaptiveTopologyUpdate>,
    pub uses_deployment_rule: bool,
    pub uses_fused_local_rule: bool,
    model: AdaptiveNpaModel,
    grid: HashGridConfig,
    dt: f32,
    update_prob: f32,
    seed: u64,
    neighbor_mode: WgpuNeighborMode,
    display_scale_per_footprint: f32,
    render_from_scale: Vec<f32>,
    render_transition_start_step: usize,
    dynamics_since_topology: usize,
    local_detail_topology_captured: bool,
    force_stable_sorted_cells: bool,
    persistent_modes: Option<WgpuPersistentAdaptiveState>,
}

fn wgpu_support_bins(model: &AdaptiveNpaModel, bandwidth: &[f32]) -> WgpuSupportBinConfig {
    let ratio = model.config.perception.support_bin_ratio;
    let resident_bootstrap = model.config.bootstrap_end_step > 0
        && model.config.initial_leaf_count() < model.config.bootstrap_target_leaf_count()
        && !model.config.retain_bootstrap_templates;
    let (min_bandwidth, max_bandwidth) = if resident_bootstrap {
        (
            model.config.perception.min_bandwidth,
            model.config.perception.max_bandwidth,
        )
    } else if local_detail_topology_control(model.config.runtime_topology_control)
        && !bandwidth.is_empty()
    {
        let active_min = bandwidth.iter().copied().fold(f32::INFINITY, f32::min);
        let active_max = bandwidth.iter().copied().fold(0.0_f32, f32::max);
        let lower = if active_max > active_min {
            (active_min / ratio).max(model.config.perception.min_bandwidth)
        } else {
            active_min
        };
        (lower, active_max)
    } else {
        (
            model.config.perception.min_bandwidth,
            model.config.perception.max_bandwidth,
        )
    };
    WgpuSupportBinConfig {
        min_bandwidth,
        max_bandwidth,
        ratio,
        force: resident_bootstrap,
    }
}

fn resident_bootstrap_capacity(model: &AdaptiveNpaModel, particles: &AdaptiveParticleSet) -> usize {
    let capacity = model.config.bootstrap_target_leaf_count();
    if particles.spatial_dims == 2
        && model.config.coarse_dynamics == AdaptiveCoarseDynamics::RepresentedMeasure
        && model.config.bootstrap_end_step > 0
        && particles.len() < capacity
        && particles.bootstrap_templates.is_empty()
        && !model.config.retain_bootstrap_templates
        && model.config.bootstrap_seed_spread == 0.0
        && !model.config.closure_recurrent_mode
        && local_detail_topology_control(model.config.runtime_topology_control)
    {
        capacity
    } else {
        particles.len()
    }
}

struct WgpuPersistentAdaptiveState {
    active_resident: WgpuAutomataState,
    restriction: WgpuPersistentModeRestriction,
    prolongation: Option<crate::gpu::WgpuActiveQuadratureProlongation>,
    persistent_detail: bool,
    mode_offsets: Vec<u32>,
    mode_rows: Vec<u32>,
    mode_weights: Vec<f32>,
    mode_measure: Vec<f32>,
    initial_mode_positions: Vec<[f32; 4]>,
    initial_mode_states: Vec<f32>,
    mode_covariance: Vec<[f32; 9]>,
    mode_bandwidth: Vec<f32>,
    mode_mask_members: Vec<Vec<(u64, f32)>>,
    bootstrap_templates: Vec<super::AdaptiveBootstrapTemplate>,
}

struct PersistentFineDiagnosticSnapshot {
    particles: AdaptiveParticleSet,
    normalized_features: Vec<f32>,
    base_update: Vec<f32>,
    observed_spacing: Vec<f32>,
    accepted_degree: Vec<usize>,
    feature_dims: usize,
}

struct PersistentActiveMaterialValues {
    represented_measure: Vec<f32>,
    particle_ids: Vec<u64>,
    update_masks: Vec<WgpuMaterialUpdateMask>,
    bandwidth: Vec<f32>,
    covariance: Vec<[f32; 9]>,
    state_jacobian: Vec<f32>,
    render_from_scale: Vec<f32>,
    render_target_footprint: Vec<f32>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct WgpuAdaptiveStepReport {
    pub completed_steps: usize,
    /// Visible material rows exported to the renderer.
    pub resident_particle_count: usize,
    /// Internal persistent rows advanced by the NPA rule.
    pub dynamics_particle_count: usize,
    /// Rows evaluated by the NPA rule. Active stateless quadrature may exceed
    /// `dynamics_particle_count` without retaining those rows across steps.
    pub interaction_particle_count: usize,
    /// Ordinary recurrent dynamics work.
    pub particle_steps: usize,
    /// Additional full resident perception passes used to make topology
    /// decisions. This is zero for topology-free rollout.
    pub topology_particle_steps: usize,
    /// Total recurrent/perception row work, including topology diagnostics.
    pub interaction_particle_steps: usize,
    pub topology_updates: Vec<AdaptiveTopologyUpdate>,
}

pub(crate) fn material_update_masks(
    particles: &AdaptiveParticleSet,
) -> AutomataResult<Vec<WgpuMaterialUpdateMask>> {
    let templates = particles
        .bootstrap_templates
        .iter()
        .map(|template| (template.parent_id, template))
        .collect::<BTreeMap<_, _>>();
    particles
        .particle_id
        .iter()
        .copied()
        .map(|particle_id| {
            let Some(template) = templates.get(&particle_id) else {
                return Ok(WgpuMaterialUpdateMask::single(particle_id));
            };
            if template.children.len() > WGPU_MATERIAL_UPDATE_MASK_MEMBERS {
                return Err(AutomataError::InvalidArgument(format!(
                    "adaptive material parent {particle_id} has {} mask members; WGPU supports at most {WGPU_MATERIAL_UPDATE_MASK_MEMBERS}",
                    template.children.len(),
                )));
            }
            let total = template
                .children
                .iter()
                .map(|child| child.represented_measure)
                .sum::<f32>()
                .max(f32::MIN_POSITIVE);
            let mut mask = WgpuMaterialUpdateMask {
                particle_ids: [0; WGPU_MATERIAL_UPDATE_MASK_MEMBERS],
                weights: [0.0; WGPU_MATERIAL_UPDATE_MASK_MEMBERS],
            };
            for (member, child) in template.children.iter().enumerate() {
                mask.particle_ids[member] = child.particle_id;
                mask.weights[member] = child.represented_measure / total;
            }
            Ok(mask)
        })
        .collect()
}

pub(crate) fn material_update_masks_from_hierarchy(
    fine: &AdaptiveParticleSet,
    hierarchy: &AdaptiveProxyHierarchy,
    members: &[AdaptiveHierarchyMember],
) -> AutomataResult<Vec<WgpuMaterialUpdateMask>> {
    members
        .iter()
        .copied()
        .map(|member| {
            let leaves = hierarchy.member_leaf_indices(member);
            if leaves.len() > WGPU_MATERIAL_UPDATE_MASK_MEMBERS {
                return Err(AutomataError::InvalidArgument(format!(
                    "adaptive hierarchy member has {} mask members; WGPU supports at most {WGPU_MATERIAL_UPDATE_MASK_MEMBERS}",
                    leaves.len(),
                )));
            }
            let total = leaves
                .iter()
                .map(|leaf| fine.represented_measure[*leaf])
                .sum::<f32>()
                .max(f32::MIN_POSITIVE);
            let mut mask = WgpuMaterialUpdateMask {
                particle_ids: [0; WGPU_MATERIAL_UPDATE_MASK_MEMBERS],
                weights: [0.0; WGPU_MATERIAL_UPDATE_MASK_MEMBERS],
            };
            for (slot, leaf) in leaves.iter().copied().enumerate() {
                mask.particle_ids[slot] = fine.particle_id[leaf];
                mask.weights[slot] = fine.represented_measure[leaf] / total;
            }
            Ok(mask)
        })
        .collect()
}

fn persistent_update_masks(
    layout: &PersistentQuadratureLayout,
) -> AutomataResult<Vec<WgpuMaterialUpdateMask>> {
    layout
        .update_mask_members
        .iter()
        .map(|members| {
            if members.is_empty() || members.len() > WGPU_MATERIAL_UPDATE_MASK_MEMBERS {
                return Err(AutomataError::InvalidArgument(format!(
                    "persistent mode has {} update-mask members; expected 1..={WGPU_MATERIAL_UPDATE_MASK_MEMBERS}",
                    members.len(),
                )));
            }
            let mut mask = WgpuMaterialUpdateMask {
                particle_ids: [0; WGPU_MATERIAL_UPDATE_MASK_MEMBERS],
                weights: [0.0; WGPU_MATERIAL_UPDATE_MASK_MEMBERS],
            };
            for (slot, (particle_id, weight)) in members.iter().copied().enumerate() {
                mask.particle_ids[slot] = particle_id;
                mask.weights[slot] = weight;
            }
            Ok(mask)
        })
        .collect()
}

fn persistent_restriction_mapping(
    active: &AdaptiveParticleSet,
    layout: &PersistentQuadratureLayout,
) -> AutomataResult<(Vec<u32>, Vec<u32>, Vec<f32>)> {
    if layout.active_row.len() != layout.particles.len()
        || layout
            .active_row
            .iter()
            .any(|active_row| *active_row >= active.len())
    {
        return Err(AutomataError::InvalidModel(
            "persistent quadrature has an invalid active-row mapping".to_owned(),
        ));
    }
    persistent_restriction_mapping_from_active_rows(
        active.len(),
        &layout.active_row,
        &layout.particles.represented_measure,
    )
}

fn persistent_restriction_mapping_from_active_rows(
    active_count: usize,
    active_rows: &[usize],
    mode_measure: &[f32],
) -> AutomataResult<(Vec<u32>, Vec<u32>, Vec<f32>)> {
    if active_rows.len() != mode_measure.len()
        || active_rows
            .iter()
            .any(|active_row| *active_row >= active_count)
        || mode_measure
            .iter()
            .any(|measure| !measure.is_finite() || *measure <= 0.0)
    {
        return Err(AutomataError::InvalidModel(
            "persistent restriction has invalid active rows or mode measures".to_owned(),
        ));
    }
    let mut modes = vec![Vec::<usize>::new(); active_count];
    for (mode, active_row) in active_rows.iter().copied().enumerate() {
        modes[active_row].push(mode);
    }
    let mut offsets = Vec::with_capacity(active_count + 1);
    let mut rows = Vec::with_capacity(active_rows.len());
    let mut weights = Vec::with_capacity(active_rows.len());
    offsets.push(0);
    for (active_row, active_modes) in modes.into_iter().enumerate() {
        if active_modes.is_empty() {
            return Err(AutomataError::InvalidModel(format!(
                "persistent active row {active_row} has no dynamics mode",
            )));
        }
        let total = active_modes
            .iter()
            .map(|mode| mode_measure[*mode])
            .sum::<f32>()
            .max(f32::MIN_POSITIVE);
        for mode in active_modes {
            rows.push(u32::try_from(mode).map_err(|_| {
                AutomataError::InvalidArgument(
                    "persistent quadrature mode index exceeds u32".to_owned(),
                )
            })?);
            weights.push(mode_measure[mode] / total);
        }
        offsets.push(u32::try_from(rows.len()).map_err(|_| {
            AutomataError::InvalidArgument("persistent quadrature mapping exceeds u32".to_owned())
        })?);
    }
    Ok((offsets, rows, weights))
}

fn persistent_restriction_mapping_from_material_partition(
    active: &AdaptiveParticleSet,
    mode_mask_members: &[Vec<(u64, f32)>],
    mode_measure: &[f32],
) -> AutomataResult<(Vec<u32>, Vec<u32>, Vec<f32>)> {
    if mode_mask_members.len() != mode_measure.len() {
        return Err(AutomataError::InvalidModel(
            "persistent mode lineage and measure counts differ".to_owned(),
        ));
    }
    let template_by_parent = active
        .bootstrap_templates
        .iter()
        .map(|template| (template.parent_id, template))
        .collect::<BTreeMap<_, _>>();
    let mut active_row_by_material_id = BTreeMap::new();
    for (row, particle_id) in active.particle_id.iter().copied().enumerate() {
        if let Some(template) = template_by_parent.get(&particle_id) {
            for child in &template.children {
                if active_row_by_material_id
                    .insert(child.particle_id, row)
                    .is_some_and(|previous| previous != row)
                {
                    return Err(AutomataError::InvalidModel(format!(
                        "persistent material child {} belongs to multiple visible rows",
                        child.particle_id,
                    )));
                }
            }
        } else {
            active_row_by_material_id.insert(particle_id, row);
        }
    }

    let active_rows = mode_mask_members
        .iter()
        .enumerate()
        .map(|(mode, members)| {
            let mut owner = None;
            for (particle_id, _) in members {
                let row = active_row_by_material_id
                    .get(particle_id)
                    .copied()
                    .ok_or_else(|| {
                        AutomataError::InvalidModel(format!(
                            "persistent mode {mode} material child {particle_id} has no visible owner",
                        ))
                    })?;
                if owner.replace(row).is_some_and(|previous| previous != row) {
                    return Err(AutomataError::InvalidModel(format!(
                        "persistent mode {mode} spans multiple visible rows",
                    )));
                }
            }
            owner.ok_or_else(|| {
                AutomataError::InvalidModel(format!(
                    "persistent mode {mode} has no material lineage",
                ))
            })
        })
        .collect::<AutomataResult<Vec<_>>>()?;
    persistent_restriction_mapping_from_active_rows(active.len(), &active_rows, mode_measure)
}

fn pad_persistent_restriction_mapping(
    mut offsets: Vec<u32>,
    active_capacity: usize,
) -> AutomataResult<Vec<u32>> {
    if offsets.is_empty() || offsets.len() > active_capacity + 1 {
        return Err(AutomataError::InvalidModel(format!(
            "persistent restriction has {} offsets for active capacity {active_capacity}",
            offsets.len(),
        )));
    }
    let end = *offsets.last().expect("non-empty offsets were checked");
    offsets.resize(active_capacity + 1, end);
    Ok(offsets)
}

fn persistent_active_material_values(
    state: &WgpuAdaptiveNpaState,
    render_target_footprint: &[f32],
    active_capacity: usize,
) -> AutomataResult<PersistentActiveMaterialValues> {
    let visible = state.particles.len();
    if active_capacity < visible
        || render_target_footprint.len() != visible
        || state.render_from_scale.len() != visible
    {
        return Err(AutomataError::InvalidModel(format!(
            "persistent visible projection has {visible} rows, capacity {active_capacity}, {} render targets, and {} transition scales",
            render_target_footprint.len(),
            state.render_from_scale.len(),
        )));
    }
    let mut represented_measure = state.particles.represented_measure.clone();
    let mut particle_ids = state.particles.particle_id.clone();
    let mut update_masks = material_update_masks(&state.particles)?;
    let mut bandwidth = state.particles.bandwidth.clone();
    let mut covariance = state.particles.covariance.clone();
    let mut state_jacobian = state.particles.state_jacobian.clone();
    let mut render_from_scale = state.render_from_scale.clone();
    let mut render_target_footprint = render_target_footprint.to_vec();
    let padding = active_capacity - visible;
    if padding > 0 {
        let dummy_footprint = state
            .model
            .config
            .min_render_footprint
            .max(f32::MIN_POSITIVE);
        let dummy_scale =
            dummy_footprint * state.display_scale_per_footprint.max(f32::MIN_POSITIVE);
        let dummy_bandwidth = state
            .model
            .config
            .perception
            .min_bandwidth
            .max(f32::MIN_POSITIVE);
        let jacobian_dims = state.particles.state_dims * state.particles.spatial_dims;
        for slot in 0..padding {
            represented_measure.push(f32::MIN_POSITIVE);
            let particle_id = u64::MAX - slot as u64;
            particle_ids.push(particle_id);
            update_masks.push(WgpuMaterialUpdateMask::single(particle_id));
            bandwidth.push(dummy_bandwidth);
            covariance.push([0.0; 9]);
            state_jacobian.extend(std::iter::repeat_n(0.0, jacobian_dims));
            render_from_scale.push(dummy_scale);
            render_target_footprint.push(dummy_footprint);
        }
    }
    Ok(PersistentActiveMaterialValues {
        represented_measure,
        particle_ids,
        update_masks,
        bandwidth,
        covariance,
        state_jacobian,
        render_from_scale,
        render_target_footprint,
    })
}

fn persistent_active_position_state_values(
    state: &WgpuAdaptiveNpaState,
    active_capacity: usize,
) -> AutomataResult<(Vec<[f32; 4]>, Vec<f32>)> {
    let visible = state.particles.len();
    if active_capacity < visible {
        return Err(AutomataError::InvalidModel(format!(
            "persistent visible projection has {visible} rows above capacity {active_capacity}",
        )));
    }
    let mut positions = state.particles.positions.clone();
    let mut states = state.particles.states.clone();
    let extent = (state.model.config.domain_max[0] - state.model.config.domain_min[0])
        .abs()
        .max(1.0);
    let off_domain = state.model.config.domain_max[0] + 1_024.0 * extent;
    for slot in 0..active_capacity - visible {
        positions.push([off_domain + slot as f32 * extent, off_domain, 0.0, 0.0]);
        states.extend(std::iter::repeat_n(0.0, state.particles.state_dims));
    }
    Ok((positions, states))
}

impl WgpuAutomataExecutor {
    /// Creates a PCA projector large enough for the full adaptive resident
    /// allocation, including rows activated by later topology updates.
    pub fn create_adaptive_state_pca(
        &self,
        state: &WgpuAdaptiveNpaState,
        config: WgpuStatePcaConfig,
    ) -> AutomataResult<WgpuStatePca> {
        let projection_state = state
            .persistent_modes
            .as_ref()
            .map(|persistent| &persistent.active_resident)
            .filter(|active| {
                active.particle_capacity * active.batch_size
                    > state.resident.particle_capacity * state.resident.batch_size
            })
            .unwrap_or(&state.resident);
        self.create_state_pca(projection_state, config)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_adaptive_state(
        &self,
        model: &AdaptiveNpaModel,
        mut particles: AdaptiveParticleSet,
        grid: &HashGridConfig,
        dt: f32,
        mut neighbor_mode: WgpuNeighborMode,
        update_prob: f32,
        seed: u64,
    ) -> AutomataResult<WgpuAdaptiveNpaState> {
        model.validate()?;
        if matches!(
            model.config.coarse_dynamics,
            AdaptiveCoarseDynamics::FineQuadrature
                | AdaptiveCoarseDynamics::PersistentFineQuadrature
        ) && (model.config.local_residual_scale > 0.0
            || model.config.proxy.context_scale > 0.0
            || model.uses_deployment_rule())
        {
            return Err(AutomataError::InvalidArgument(
                "quadrature WGPU inference requires the unmodified base rule".to_owned(),
            ));
        }
        particles.validate()?;
        if (model.uses_canonical_compatible_residual() || model.config.closure_recurrent_mode)
            && !matches!(
                neighbor_mode,
                WgpuNeighborMode::CooperativeSortedCells
                    | WgpuNeighborMode::SubgroupCooperativeSortedCells
            )
        {
            neighbor_mode = WgpuNeighborMode::CooperativeSortedCells;
        }
        for footprint in &mut particles.render_footprint {
            *footprint = model.config.render_footprint(*footprint);
        }
        let display_scale_per_footprint = adaptive_display_scale_per_footprint(model);
        let render_from_scale =
            target_render_scales(model, &particles, display_scale_per_footprint)?;
        let render_target_footprint = target_render_footprints(model, &particles);
        let update_masks = material_update_masks(&particles)?;
        let gpu_rule = model.gpu_inference_rule()?;
        let material = WgpuMaterialStateInit {
            represented_measure: &particles.represented_measure,
            particle_ids: Some(&particles.particle_id),
            update_masks: Some(&update_masks),
            bandwidth: &particles.bandwidth,
            support_bins: Some(wgpu_support_bins(model, &particles.bandwidth)),
            covariance: &particles.covariance,
            state_jacobian: &particles.state_jacobian,
            closure_mode: Some(&particles.closure_mode),
            closure_basis: Some(&particles.closure_basis),
            closure_phase: Some(&particles.closure_phase),
            render_from_scale: &render_from_scale,
            render_target_footprint: &render_target_footprint,
            display_scale_per_footprint,
            render_transition_steps: model.config.render_transition_steps,
        };
        let resident_capacity = resident_bootstrap_capacity(model, &particles);
        let mut resident = if resident_capacity > particles.len() {
            self.create_material_state_with_capacity(
                &gpu_rule.rule,
                &particles.positions,
                &particles.states,
                particles.len(),
                resident_capacity,
                grid,
                dt,
                neighbor_mode,
                update_prob,
                seed,
                material,
            )?
        } else {
            self.create_material_state_with_neighbor_mode_and_update_prob(
                &gpu_rule.rule,
                &particles.positions,
                &particles.states,
                1,
                particles.len(),
                grid,
                dt,
                neighbor_mode,
                update_prob,
                seed,
                material,
            )?
        };
        self.configure_state_adaptive_integration(
            &mut resident,
            model.config.base_rule_footprint(),
            model.config.expected_coarse_update_mask,
        )?;
        self.configure_state_adaptive_reference_footprint(
            &mut resident,
            model.config.reference_footprint,
        )?;
        if let Some(local_hidden_start) = gpu_rule.local_hidden_start {
            self.configure_state_adaptive_local_rule(
                &mut resident,
                gpu_rule.local_rule_mode,
                local_hidden_start,
                model.config.local_residual_scale,
                model.config.base_rule_footprint(),
                model.config.reference_footprint,
                model.config.perception.shepard_epsilon,
                model.config.perception.moment_regularization,
                model.config.perception.moment_condition_limit,
                gpu_local_max_neighbors(model)?,
                model.config.perception.pair_scale_power,
            )?;
        }
        if let Some(closure) = &model.closure_mode_rule {
            self.configure_state_adaptive_closure_rule(&mut resident, closure)?;
        }
        if let Some(closure) = &model.closure_basis_rule {
            self.configure_state_adaptive_closure_basis_rule(&mut resident, closure)?;
        }
        if local_detail_topology_control(model.config.runtime_topology_control) {
            // Local-detail topology makes its one diagnostic pass deterministic
            // explicitly. Keeping stable cell sorting enabled for every
            // intervening dynamics step adds substantial work without
            // affecting the detached topology decision.
            self.set_stable_sorted_cells_enabled(&mut resident, false);
        }
        let mut state = WgpuAdaptiveNpaState {
            particles,
            resident,
            completed_steps: 0,
            last_topology: None,
            uses_deployment_rule: model.uses_deployment_rule(),
            uses_fused_local_rule: gpu_rule.local_hidden_start.is_some(),
            model: model.clone(),
            grid: grid.clone(),
            dt,
            update_prob,
            seed,
            neighbor_mode,
            display_scale_per_footprint,
            render_from_scale,
            render_transition_start_step: 0,
            dynamics_since_topology: 0,
            local_detail_topology_captured: false,
            force_stable_sorted_cells: false,
            persistent_modes: None,
        };
        if matches!(
            state.model.config.coarse_dynamics,
            AdaptiveCoarseDynamics::FineQuadrature
                | AdaptiveCoarseDynamics::PersistentFineQuadrature
        ) && !state.particles.bootstrap_templates.is_empty()
        {
            let quadrature_points = state.model.config.bootstrap_quadrature_point_count();
            self.install_persistent_modes(
                &mut state,
                &render_target_footprint,
                0,
                quadrature_points,
            )?;
        }
        Ok(state)
    }

    pub fn step_adaptive_state_many_into_gaussian_bind_group(
        &self,
        state: &mut WgpuAdaptiveNpaState,
        gaussian: &WgpuGaussianBindGroup,
        steps: usize,
        topology_enabled: bool,
    ) -> AutomataResult<WgpuAdaptiveStepReport> {
        let topology_control = state.model.config.runtime_topology_control;
        self.step_adaptive_state_many_inner(
            state,
            Some(gaussian),
            steps,
            topology_enabled,
            topology_control,
        )
    }

    pub fn step_adaptive_state_many_pca_into_gaussian_bind_group(
        &self,
        state: &mut WgpuAdaptiveNpaState,
        gaussian: &WgpuGaussianBindGroup,
        steps: usize,
        topology_enabled: bool,
        pca: &mut WgpuStatePca,
    ) -> AutomataResult<WgpuAdaptiveStepReport> {
        let report = self.step_adaptive_state_many(state, steps, topology_enabled)?;
        self.write_adaptive_state_pca_into_gaussian_bind_group(state, gaussian, pca)?;
        Ok(report)
    }

    /// Exports the current visible adaptive material without advancing it.
    ///
    /// Persistent-fine rollout keeps a larger internal quadrature state. The
    /// visible partition must be restricted before rendering rather than
    /// drawing an arbitrary prefix of those internal rows.
    pub fn write_adaptive_state_into_gaussian_bind_group(
        &self,
        state: &mut WgpuAdaptiveNpaState,
        gaussian: &WgpuGaussianBindGroup,
    ) -> AutomataResult<()> {
        if let Some(persistent) = state.persistent_modes.as_mut() {
            if persistent.persistent_detail {
                self.restrict_persistent_modes_into_gaussians(
                    &persistent.restriction,
                    &state.resident,
                    &mut persistent.active_resident,
                    gaussian,
                )
            } else {
                self.write_state_into_gaussian_bind_group(&persistent.active_resident, gaussian)
            }
        } else {
            self.write_state_into_gaussian_bind_group(&state.resident, gaussian)
        }
    }

    /// Exports the visible adaptive state with rolling PCA color. Persistent
    /// quadrature is restricted to its visible partition before projection.
    pub fn write_adaptive_state_pca_into_gaussian_bind_group(
        &self,
        state: &mut WgpuAdaptiveNpaState,
        gaussian: &WgpuGaussianBindGroup,
        pca: &mut WgpuStatePca,
    ) -> AutomataResult<()> {
        if let Some(persistent) = state.persistent_modes.as_mut() {
            if persistent.persistent_detail {
                self.restrict_persistent_modes(
                    &persistent.restriction,
                    &state.resident,
                    &persistent.active_resident,
                )?;
            }
            self.write_state_pca_into_gaussian_bind_group(
                &persistent.active_resident,
                gaussian,
                pca,
            )
        } else {
            self.write_state_pca_into_gaussian_bind_group(&state.resident, gaussian, pca)
        }
    }

    /// Advances an adaptive state entirely on the resident WGPU path without
    /// paying the Gaussian export cost used by the interactive viewer.
    pub fn step_adaptive_state_many(
        &self,
        state: &mut WgpuAdaptiveNpaState,
        steps: usize,
        topology_enabled: bool,
    ) -> AutomataResult<WgpuAdaptiveStepReport> {
        let topology_control = state.model.config.runtime_topology_control;
        self.step_adaptive_state_many_inner(state, None, steps, topology_enabled, topology_control)
    }

    /// Forces deterministic within-cell traversal for every subsequent
    /// adaptive dynamics step. Quality/parity validation uses this across the
    /// settled tail as well as pending topology decisions; interactive
    /// inference keeps the faster default unless explicitly requested.
    pub fn set_adaptive_stable_sorted_cells_enabled(
        &self,
        state: &mut WgpuAdaptiveNpaState,
        enabled: bool,
    ) {
        state.force_stable_sorted_cells = enabled;
        self.set_stable_sorted_cells_enabled(&mut state.resident, enabled);
        if let Some(persistent) = state.persistent_modes.as_mut() {
            self.set_stable_sorted_cells_enabled(&mut persistent.active_resident, enabled);
        }
    }

    pub(crate) fn step_adaptive_state_many_with_topology_control(
        &self,
        state: &mut WgpuAdaptiveNpaState,
        steps: usize,
        topology_enabled: bool,
        topology_control: AdaptiveTopologyControl,
    ) -> AutomataResult<WgpuAdaptiveStepReport> {
        self.step_adaptive_state_many_inner(state, None, steps, topology_enabled, topology_control)
    }

    fn step_adaptive_state_many_inner(
        &self,
        state: &mut WgpuAdaptiveNpaState,
        gaussian: Option<&WgpuGaussianBindGroup>,
        steps: usize,
        topology_enabled: bool,
        topology_control: AdaptiveTopologyControl,
    ) -> AutomataResult<WgpuAdaptiveStepReport> {
        let requested_steps = steps.max(1);
        if let Some(report) = self.step_batched_paired_local_detail(
            state,
            gaussian,
            requested_steps,
            topology_enabled,
            topology_control,
        )? {
            return Ok(report);
        }
        let mut remaining = requested_steps;
        let mut particle_steps = 0_usize;
        let mut topology_particle_steps = 0_usize;
        let mut topology_updates = Vec::new();
        let mut gaussian_exported = false;
        let mut topology_search_after = state.completed_steps;
        if topology_enabled {
            let bootstrap_step = state.completed_steps.saturating_add(1);
            if state
                .model
                .config
                .coarse_to_fine_bootstrap_active(bootstrap_step, state.particles.len())
                && state
                    .model
                    .config
                    .is_topology_step(bootstrap_step, state.particles.len())
            {
                let update = self.apply_adaptive_resident_topology(
                    state,
                    bootstrap_step,
                    state.completed_steps,
                    topology_control,
                )?;
                if local_detail_topology_control(topology_control)
                    && state.persistent_modes.is_none()
                {
                    topology_particle_steps =
                        topology_particle_steps.saturating_add(state.resident.particle_count);
                }
                topology_updates.push(update);
                // Bootstrap is applied before its dynamics step. Advance the
                // decision cursor even though `completed_steps` has not moved,
                // otherwise the segmented loop can select the same absolute
                // topology step a second time.
                topology_search_after = bootstrap_step;
            }
        }
        while remaining > 0 {
            let topology_step = topology_enabled
                .then(|| {
                    next_topology_step(&state.model, topology_search_after, state.particles.len())
                })
                .flatten();
            // A learned cut is a function of the recurrent trajectory. Atomic
            // scatter order can perturb that trajectory enough to select a
            // different cut hundreds of steps later, so persistent adaptive
            // dynamics stay deterministic while another topology decision is
            // pending. This includes the bootstrap prefix before persistent
            // fine modes are installed. Once the final decision has passed,
            // settled inference returns to the faster ordinary sorted-cell
            // path.
            self.set_stable_sorted_cells_enabled(
                &mut state.resident,
                adaptive_dynamics_require_stable_cells(
                    topology_step,
                    topology_control,
                    state.force_stable_sorted_cells,
                ),
            );
            let segment = topology_step
                .map(|step| step.saturating_sub(state.completed_steps).min(remaining))
                .unwrap_or(remaining)
                .max(1);
            let topology_at_segment_end =
                topology_step == Some(state.completed_steps.saturating_add(segment));
            let capture_local_detail = topology_at_segment_end
                && local_detail_capture_required(
                    &state.model,
                    state.completed_steps.saturating_add(segment),
                    state.particles.len(),
                    topology_control,
                    state.persistent_modes.is_none(),
                );
            let persistent_gaussian = (remaining == segment
                && !topology_at_segment_end
                && state
                    .persistent_modes
                    .as_ref()
                    .is_some_and(|modes| modes.persistent_detail))
            .then_some(gaussian)
            .flatten();
            if let Some(gaussian) = persistent_gaussian {
                if segment > 1 {
                    self.step_state_many(&mut state.resident, segment - 1)?;
                }
                let persistent = state
                    .persistent_modes
                    .as_mut()
                    .expect("persistent export was checked above");
                self.step_persistent_modes_into_gaussians(
                    &persistent.restriction,
                    &mut state.resident,
                    &mut persistent.active_resident,
                    gaussian,
                )?;
                gaussian_exported = true;
            } else if state
                .persistent_modes
                .as_ref()
                .is_some_and(|modes| !modes.persistent_detail)
            {
                let quadrature = state
                    .persistent_modes
                    .as_mut()
                    .expect("active quadrature was checked above");
                self.step_active_quadrature_many(
                    quadrature
                        .prolongation
                        .as_ref()
                        .expect("active quadrature has a prolongation"),
                    &quadrature.restriction,
                    &mut state.resident,
                    &mut quadrature.active_resident,
                    segment,
                )?;
                if let Some(gaussian) =
                    gaussian.filter(|_| remaining == segment && !topology_at_segment_end)
                {
                    self.write_state_into_gaussian_bind_group(
                        &quadrature.active_resident,
                        gaussian,
                    )?;
                    gaussian_exported = true;
                }
            } else {
                if capture_local_detail {
                    if segment > 1 {
                        self.step_state_many(&mut state.resident, segment - 1)?;
                    }
                    self.step_state_capturing_local_detail(&mut state.resident)?;
                    state.local_detail_topology_captured = true;
                } else {
                    self.step_state_many(&mut state.resident, segment)?;
                }
            }
            particle_steps = particle_steps.saturating_add(state.resident.particle_count * segment);
            state.completed_steps = state.completed_steps.saturating_add(segment);
            state.dynamics_since_topology = state.dynamics_since_topology.saturating_add(segment);
            topology_search_after = state.completed_steps;
            remaining -= segment;

            if topology_step == Some(state.completed_steps) {
                let used_captured_detail = state.local_detail_topology_captured;
                let update = self.apply_adaptive_resident_topology(
                    state,
                    state.completed_steps,
                    state.completed_steps,
                    topology_control,
                )?;
                if local_detail_topology_control(topology_control)
                    && state.persistent_modes.is_none()
                    && !used_captured_detail
                {
                    topology_particle_steps =
                        topology_particle_steps.saturating_add(state.resident.particle_count);
                }
                topology_updates.push(update);
            }
        }
        if let Some(gaussian) = gaussian.filter(|_| !gaussian_exported) {
            if let Some(persistent) = state.persistent_modes.as_mut() {
                if persistent.persistent_detail {
                    self.restrict_persistent_modes_into_gaussians(
                        &persistent.restriction,
                        &state.resident,
                        &mut persistent.active_resident,
                        gaussian,
                    )?;
                } else {
                    self.write_state_into_gaussian_bind_group(
                        &persistent.active_resident,
                        gaussian,
                    )?;
                }
            } else {
                self.write_state_into_gaussian_bind_group(&state.resident, gaussian)?;
            }
        }
        Ok(WgpuAdaptiveStepReport {
            completed_steps: requested_steps,
            resident_particle_count: state.particles.len(),
            dynamics_particle_count: state.persistent_modes.as_ref().map_or(
                state.resident.particle_count,
                |modes| {
                    if modes.persistent_detail {
                        state.resident.particle_count
                    } else {
                        state.particles.len()
                    }
                },
            ),
            interaction_particle_count: state.resident.particle_count,
            particle_steps,
            topology_particle_steps,
            interaction_particle_steps: particle_steps.saturating_add(topology_particle_steps),
            topology_updates,
        })
    }

    fn step_batched_paired_local_detail(
        &self,
        state: &mut WgpuAdaptiveNpaState,
        gaussian: Option<&WgpuGaussianBindGroup>,
        requested_steps: usize,
        topology_enabled: bool,
        topology_control: AdaptiveTopologyControl,
    ) -> AutomataResult<Option<WgpuAdaptiveStepReport>> {
        if !topology_enabled
            || topology_control != AdaptiveTopologyControl::PairedLocalDetail
            || state.persistent_modes.is_some()
            || !state.particles.bootstrap_templates.is_empty()
            || state.resident.particle_count != state.particles.len()
            || state.particles.len() != state.model.config.target_leaves
            || state.model.config.coarse_to_fine_bootstrap_active(
                state.completed_steps.saturating_add(1),
                state.particles.len(),
            )
        {
            return Ok(None);
        }
        validate_resident_paired_local_detail_material(state)?;

        let end_step = state.completed_steps.saturating_add(requested_steps);
        let mut topology_steps = Vec::new();
        let mut search_after = state.completed_steps;
        while let Some(step) = next_topology_step(&state.model, search_after, state.particles.len())
        {
            if step > end_step {
                break;
            }
            topology_steps.push(step);
            search_after = step;
        }
        if topology_steps.is_empty() {
            return Ok(None);
        }
        let topology_step_offsets = topology_steps
            .iter()
            .map(|step| step - state.completed_steps)
            .collect::<Vec<_>>();
        let paired_steps = *topology_step_offsets
            .last()
            .expect("non-empty topology offsets were checked above");
        self.step_state_many_with_paired_local_detail(
            &mut state.resident,
            paired_steps,
            &topology_step_offsets,
            state.model.config.paired_topology_split_radius_scale,
            state.model.config.paired_topology_merge_detail_scale,
            state.model.config.min_reallocation_relative_gain,
        )?;
        let settled_steps = requested_steps - paired_steps;
        if settled_steps > 0 {
            self.step_state_many(&mut state.resident, settled_steps)?;
        }

        let resident_particle_count = state.resident.particle_count;
        let topology_updates = topology_steps
            .into_iter()
            .map(|step| AdaptiveTopologyUpdate {
                step,
                initial_leaf_count: state.particles.len(),
                final_leaf_count: state.particles.len(),
                split_events: 1,
                merge_events: 1,
                elapsed_ms: 0.0,
            })
            .collect::<Vec<_>>();
        let last_topology_step = topology_updates
            .last()
            .map(|update| update.step)
            .expect("non-empty topology schedule was checked above");
        state.completed_steps = end_step;
        state.dynamics_since_topology = end_step.saturating_sub(last_topology_step);
        state.last_topology = topology_updates.last().copied();
        state.local_detail_topology_captured = false;
        if let Some(gaussian) = gaussian {
            self.write_state_into_gaussian_bind_group(&state.resident, gaussian)?;
        }

        let particle_steps = resident_particle_count.saturating_mul(requested_steps);
        Ok(Some(WgpuAdaptiveStepReport {
            completed_steps: requested_steps,
            resident_particle_count: state.particles.len(),
            dynamics_particle_count: resident_particle_count,
            interaction_particle_count: resident_particle_count,
            particle_steps,
            topology_particle_steps: 0,
            interaction_particle_steps: particle_steps,
            topology_updates,
        }))
    }

    /// Synchronizes the resident adaptive state into its CPU particle mirror.
    ///
    /// This performs device-to-host readback and should be reserved for bounded
    /// evaluation, checkpointing, or diagnostics rather than per-frame inference.
    pub fn synchronize_adaptive_particles(
        &self,
        state: &mut WgpuAdaptiveNpaState,
    ) -> AutomataResult<()> {
        let synchronized = if let Some(persistent) = state.persistent_modes.as_ref() {
            if persistent.persistent_detail {
                self.restrict_persistent_modes(
                    &persistent.restriction,
                    &state.resident,
                    &persistent.active_resident,
                )?;
            }
            let (positions, states) = self.read_positions_states(&persistent.active_resident)?;
            let mode_values = Some(self.read_positions_states(&state.resident)?);
            (positions, states, mode_values)
        } else {
            let (positions, states) = self.read_positions_states(&state.resident)?;
            (positions, states, None)
        };
        let (mut positions, mut states, mode_values) = synchronized;
        if state.persistent_modes.is_some() {
            positions.truncate(state.particles.len());
            states.truncate(state.particles.len() * state.particles.state_dims);
        }
        if positions.len() != state.particles.len()
            || states.len() != state.particles.len() * state.particles.state_dims
        {
            return Err(AutomataError::InvalidModel(
                "resident adaptive state does not match its visible material rows".to_owned(),
            ));
        }
        state.particles.positions = positions;
        state.particles.states = states;
        if state.model.config.closure_recurrent_mode {
            if state.persistent_modes.is_some() {
                return Err(AutomataError::InvalidModel(
                    "recurrent closure mode cannot synchronize persistent quadrature".to_owned(),
                ));
            }
            let (closure_mode, closure_basis, closure_phase) =
                self.read_material_closure_state(&state.resident)?;
            state.particles.closure_mode = closure_mode;
            state.particles.closure_basis = closure_basis;
            state.particles.closure_phase = closure_phase;
        }
        if let (Some(persistent), Some((mode_positions, mode_states))) =
            (state.persistent_modes.as_ref(), mode_values)
        {
            synchronize_quadrature_material(
                &mut state.particles,
                persistent,
                &mode_positions,
                &mode_states,
            )?;
        }
        let displayed = displayed_render_scales(state)?;
        let display_scale = state.display_scale_per_footprint.max(f32::MIN_POSITIVE);
        state.particles.render_footprint = displayed
            .into_iter()
            .map(|scale| scale / display_scale)
            .collect();
        state.particles.validate()
    }

    fn install_persistent_modes(
        &self,
        state: &mut WgpuAdaptiveNpaState,
        render_target_footprint: &[f32],
        resident_step: u32,
        coarse_quadrature_points: usize,
    ) -> AutomataResult<()> {
        let persistent_detail =
            state.model.config.coarse_dynamics == AdaptiveCoarseDynamics::PersistentFineQuadrature;
        let layout = quadrature_layout_with_points(
            &state.model,
            &state.particles,
            persistent_detail,
            coarse_quadrature_points,
        )?;
        let update_masks = persistent_update_masks(&layout)?;
        let (mode_offsets, mode_rows, mode_weights) =
            persistent_restriction_mapping(&state.particles, &layout)?;
        let internal_render = layout.particles.render_footprint.clone();
        let gpu_rule = state.model.gpu_inference_rule()?;
        let mut internal = self.create_material_state_with_neighbor_mode_and_update_prob(
            &gpu_rule.rule,
            &layout.particles.positions,
            &layout.particles.states,
            1,
            layout.particles.len(),
            &state.grid,
            state.dt,
            state.neighbor_mode,
            if persistent_detail {
                state.update_prob
            } else {
                1.0
            },
            state.seed,
            WgpuMaterialStateInit {
                represented_measure: &layout.particles.represented_measure,
                particle_ids: Some(&layout.particles.particle_id),
                update_masks: Some(&update_masks),
                bandwidth: &layout.particles.bandwidth,
                support_bins: Some(wgpu_support_bins(&state.model, &layout.particles.bandwidth)),
                covariance: &layout.particles.covariance,
                state_jacobian: &layout.particles.state_jacobian,
                closure_mode: None,
                closure_basis: None,
                closure_phase: None,
                render_from_scale: &internal_render,
                render_target_footprint: &internal_render,
                display_scale_per_footprint: 1.0,
                render_transition_steps: 0,
            },
        )?;
        self.configure_state_adaptive_integration(
            &mut internal,
            state.model.config.base_rule_footprint(),
            false,
        )?;
        self.set_state_step_index(&mut internal, resident_step)?;
        if state.force_stable_sorted_cells {
            self.set_stable_sorted_cells_enabled(&mut internal, true);
        }

        let active_resident =
            self.create_persistent_active_resident(state, render_target_footprint, resident_step)?;
        let mode_offsets = pad_persistent_restriction_mapping(mode_offsets, active_resident.total)?;
        let restriction = self.create_persistent_mode_restriction(
            &internal,
            &active_resident,
            &mode_offsets,
            &mode_rows,
            &mode_weights,
        )?;
        let prolongation = if persistent_detail {
            None
        } else {
            let mode_offsets = layout
                .active_row
                .iter()
                .enumerate()
                .map(|(mode, active)| {
                    let mut offset = [0.0_f32; 4];
                    for (axis, value) in offset
                        .iter_mut()
                        .enumerate()
                        .take(state.particles.spatial_dims)
                    {
                        *value = layout.particles.positions[mode][axis]
                            - state.particles.positions[*active][axis];
                    }
                    offset
                })
                .collect::<Vec<_>>();
            let active_rows = layout
                .active_row
                .iter()
                .map(|row| {
                    u32::try_from(*row).map_err(|_| {
                        AutomataError::InvalidArgument(
                            "active quadrature row exceeds u32".to_owned(),
                        )
                    })
                })
                .collect::<AutomataResult<Vec<_>>>()?;
            Some(self.create_active_quadrature_prolongation(
                &internal,
                &active_resident,
                &active_rows,
                &mode_offsets,
            )?)
        };
        state.resident = internal;
        state.persistent_modes = Some(WgpuPersistentAdaptiveState {
            active_resident,
            restriction,
            prolongation,
            persistent_detail,
            mode_offsets,
            mode_rows,
            mode_weights,
            mode_measure: layout.particles.represented_measure,
            initial_mode_positions: layout.particles.positions,
            initial_mode_states: layout.particles.states,
            mode_covariance: layout.particles.covariance,
            mode_bandwidth: layout.particles.bandwidth,
            mode_mask_members: layout.update_mask_members,
            bootstrap_templates: state.particles.bootstrap_templates.clone(),
        });
        Ok(())
    }

    /// Returns the cumulative number of budget-neutral local-detail topology
    /// exchanges accepted by the resident device controller.
    ///
    /// Scheduled topology passes that fail the configured gain margin do not
    /// increment this counter. Reading it synchronizes with the device and is
    /// intended for bounded validation rather than per-frame inference.
    pub fn read_adaptive_local_detail_topology_accept_count(
        &self,
        state: &WgpuAdaptiveNpaState,
    ) -> AutomataResult<usize> {
        usize::try_from(self.read_local_detail_topology_accept_count(&state.resident)?).map_err(
            |_| {
                AutomataError::InvalidArgument(
                    "local-detail topology accept counter exceeds usize".to_owned(),
                )
            },
        )
    }

    /// Reads the interaction-grid overflow counter for bounded adaptive
    /// validation. The canonical direct active-material path has no hidden
    /// resident state, so this is the exact dynamics grid used by the rollout.
    pub fn read_adaptive_grid_overflow(&self, state: &WgpuAdaptiveNpaState) -> AutomataResult<u32> {
        self.read_grid_overflow(&state.resident)
    }

    fn create_persistent_active_resident(
        &self,
        state: &WgpuAdaptiveNpaState,
        render_target_footprint: &[f32],
        resident_step: u32,
    ) -> AutomataResult<WgpuAutomataState> {
        let gpu_rule = state.model.gpu_inference_rule()?;
        let active_capacity = state
            .model
            .config
            .bootstrap_fine_leaf_count()
            .max(state.particles.len());
        let material =
            persistent_active_material_values(state, render_target_footprint, active_capacity)?;
        let (positions, states) = persistent_active_position_state_values(state, active_capacity)?;
        let mut active = self.create_material_state_with_neighbor_mode_and_update_prob(
            &gpu_rule.rule,
            &positions,
            &states,
            1,
            active_capacity,
            &state.grid,
            state.dt,
            state.neighbor_mode,
            state.update_prob,
            state.seed,
            WgpuMaterialStateInit {
                represented_measure: &material.represented_measure,
                particle_ids: Some(&material.particle_ids),
                update_masks: Some(&material.update_masks),
                bandwidth: &material.bandwidth,
                support_bins: Some(wgpu_support_bins(&state.model, &material.bandwidth)),
                covariance: &material.covariance,
                state_jacobian: &material.state_jacobian,
                closure_mode: None,
                closure_basis: None,
                closure_phase: None,
                render_from_scale: &material.render_from_scale,
                render_target_footprint: &material.render_target_footprint,
                display_scale_per_footprint: state.display_scale_per_footprint,
                render_transition_steps: state.model.config.render_transition_steps,
            },
        )?;
        self.set_state_step_index(&mut active, resident_step)?;
        self.begin_state_render_transition(&mut active, resident_step)?;
        Ok(active)
    }

    fn remap_persistent_active_partition(
        &self,
        state: &mut WgpuAdaptiveNpaState,
        render_target_footprint: &[f32],
        resident_step: u32,
    ) -> AutomataResult<()> {
        let mut persistent = state.persistent_modes.take().ok_or_else(|| {
            AutomataError::InvalidModel(
                "persistent partition remap lost its resident mode state".to_owned(),
            )
        })?;
        let active_capacity = persistent.active_resident.total;
        let material =
            persistent_active_material_values(state, render_target_footprint, active_capacity)?;
        self.update_state_material(
            &mut persistent.active_resident,
            WgpuMaterialStateInit {
                represented_measure: &material.represented_measure,
                particle_ids: Some(&material.particle_ids),
                update_masks: Some(&material.update_masks),
                bandwidth: &material.bandwidth,
                support_bins: Some(wgpu_support_bins(&state.model, &material.bandwidth)),
                covariance: &material.covariance,
                state_jacobian: &material.state_jacobian,
                closure_mode: None,
                closure_basis: None,
                closure_phase: None,
                render_from_scale: &material.render_from_scale,
                render_target_footprint: &material.render_target_footprint,
                display_scale_per_footprint: state.display_scale_per_footprint,
                render_transition_steps: state.model.config.render_transition_steps,
            },
        )?;
        self.set_state_step_index(&mut persistent.active_resident, resident_step)?;
        self.begin_state_render_transition(&mut persistent.active_resident, resident_step)?;
        let (mode_offsets, mode_rows, mode_weights) =
            persistent_restriction_mapping_from_material_partition(
                &state.particles,
                &persistent.mode_mask_members,
                &persistent.mode_measure,
            )?;
        let mode_offsets = pad_persistent_restriction_mapping(mode_offsets, active_capacity)?;
        let restriction = self.create_persistent_mode_restriction(
            &state.resident,
            &persistent.active_resident,
            &mode_offsets,
            &mode_rows,
            &mode_weights,
        )?;
        persistent.restriction = restriction;
        persistent.mode_offsets = mode_offsets;
        persistent.mode_rows = mode_rows;
        persistent.mode_weights = mode_weights;
        state.persistent_modes = Some(persistent);
        Ok(())
    }

    fn install_direct_resident(
        &self,
        state: &mut WgpuAdaptiveNpaState,
        render_target_footprint: &[f32],
        resident_step: u32,
    ) -> AutomataResult<()> {
        let gpu_rule = state.model.gpu_inference_rule()?;
        let update_masks = material_update_masks(&state.particles)?;
        state.resident = self.create_material_state_with_neighbor_mode_and_update_prob(
            &gpu_rule.rule,
            &state.particles.positions,
            &state.particles.states,
            1,
            state.particles.len(),
            &state.grid,
            state.dt,
            state.neighbor_mode,
            state.update_prob,
            state.seed,
            WgpuMaterialStateInit {
                represented_measure: &state.particles.represented_measure,
                particle_ids: Some(&state.particles.particle_id),
                update_masks: Some(&update_masks),
                bandwidth: &state.particles.bandwidth,
                support_bins: Some(wgpu_support_bins(&state.model, &state.particles.bandwidth)),
                covariance: &state.particles.covariance,
                state_jacobian: &state.particles.state_jacobian,
                closure_mode: Some(&state.particles.closure_mode),
                closure_basis: Some(&state.particles.closure_basis),
                closure_phase: Some(&state.particles.closure_phase),
                render_from_scale: &state.render_from_scale,
                render_target_footprint,
                display_scale_per_footprint: state.display_scale_per_footprint,
                render_transition_steps: state.model.config.render_transition_steps,
            },
        )?;
        self.configure_state_adaptive_integration(
            &mut state.resident,
            state.model.config.base_rule_footprint(),
            state.model.config.expected_coarse_update_mask,
        )?;
        if let Some(local_hidden_start) = gpu_rule.local_hidden_start {
            self.configure_state_adaptive_local_rule(
                &mut state.resident,
                gpu_rule.local_rule_mode,
                local_hidden_start,
                state.model.config.local_residual_scale,
                state.model.config.base_rule_footprint(),
                state.model.config.reference_footprint,
                state.model.config.perception.shepard_epsilon,
                state.model.config.perception.moment_regularization,
                state.model.config.perception.moment_condition_limit,
                gpu_local_max_neighbors(&state.model)?,
                state.model.config.perception.pair_scale_power,
            )?;
        }
        if let Some(closure) = &state.model.closure_mode_rule {
            self.configure_state_adaptive_closure_rule(&mut state.resident, closure)?;
        }
        if let Some(closure) = &state.model.closure_basis_rule {
            self.configure_state_adaptive_closure_basis_rule(&mut state.resident, closure)?;
        }
        self.set_state_step_index(&mut state.resident, resident_step)?;
        if state.force_stable_sorted_cells {
            self.set_stable_sorted_cells_enabled(&mut state.resident, true);
        }
        state.persistent_modes = None;
        Ok(())
    }

    fn refine_persistent_bootstrap(
        &self,
        state: &mut WgpuAdaptiveNpaState,
        decision_step: usize,
        resident_step: usize,
        topology_control: AdaptiveTopologyControl,
    ) -> AutomataResult<AdaptiveTopologyUpdate> {
        let _ = topology_control;
        let child_display_scale = displayed_scale_by_material_child_id(state)?;
        let update = apply_hierarchical_bootstrap_refinement(
            &state.model,
            &mut state.particles,
            decision_step,
            state.dynamics_since_topology,
        )?;
        state.dynamics_since_topology = 0;
        let render_target_footprint = target_render_footprints(&state.model, &state.particles);
        state.render_from_scale = reallocated_render_from_scale(
            &state.particles,
            &child_display_scale,
            state.display_scale_per_footprint,
            &state.model,
        )?;
        let resident_step = u32::try_from(resident_step).map_err(|_| {
            AutomataError::InvalidArgument("adaptive GPU rollout step exceeds u32".to_owned())
        })?;
        self.remap_persistent_active_partition(state, &render_target_footprint, resident_step)?;
        state.render_transition_start_step = decision_step;
        state.last_topology = Some(update);
        Ok(update)
    }

    fn refine_resident_canonical_bootstrap(
        &self,
        state: &mut WgpuAdaptiveNpaState,
        decision_step: usize,
        resident_step: usize,
    ) -> AutomataResult<AdaptiveTopologyUpdate> {
        let mut next_particles = state.particles.clone();
        let update = apply_resident_canonical_bootstrap_refinement(
            &state.model,
            &mut next_particles,
            decision_step,
            state.dynamics_since_topology,
        )?;
        if update.split_events == 0 {
            state.dynamics_since_topology = 0;
            state.last_topology = Some(update);
            return Ok(update);
        }
        self.apply_resident_bootstrap_splits(
            &mut state.resident,
            update.split_events,
            state.model.config.material_seed_bandwidth_exponent,
            state.model.config.render_footprint_exponent,
        )?;
        let mean_measure = (next_particles.total_measure() / next_particles.len() as f64) as f32;
        let max_bandwidth = next_particles
            .bandwidth
            .iter()
            .copied()
            .fold(0.0_f32, f32::max);
        let gpu_rule = state.model.gpu_inference_rule()?;
        self.activate_reserved_material_rows(
            &mut state.resident,
            &gpu_rule.rule,
            &state.grid,
            next_particles.len(),
            mean_measure,
            max_bandwidth,
        )?;
        let resident_step = u32::try_from(resident_step).map_err(|_| {
            AutomataError::InvalidArgument("adaptive GPU rollout step exceeds u32".to_owned())
        })?;
        self.begin_state_render_transition(&mut state.resident, resident_step)?;
        state.render_from_scale = target_render_scales(
            &state.model,
            &next_particles,
            state.display_scale_per_footprint,
        )?;
        state.particles = next_particles;
        state.render_transition_start_step = decision_step;
        state.dynamics_since_topology = 0;
        state.last_topology = Some(update);
        Ok(update)
    }

    fn apply_adaptive_resident_topology(
        &self,
        state: &mut WgpuAdaptiveNpaState,
        decision_step: usize,
        resident_step: usize,
        topology_control: AdaptiveTopologyControl,
    ) -> AutomataResult<AdaptiveTopologyUpdate> {
        if state.resident.particle_capacity > state.resident.particle_count
            && state
                .model
                .config
                .coarse_to_fine_bootstrap_active(decision_step, state.particles.len())
        {
            return self.refine_resident_canonical_bootstrap(state, decision_step, resident_step);
        }
        if state
            .persistent_modes
            .as_ref()
            .is_some_and(|modes| !modes.persistent_detail)
        {
            return self.reallocate_active_quadrature(
                state,
                decision_step,
                resident_step,
                topology_control,
            );
        }
        if let Some(target) = state
            .model
            .config
            .scheduled_restriction_target(decision_step, state.particles.len())
            && state.persistent_modes.is_some()
        {
            return self.reallocate_persistent_modes(state, decision_step, resident_step, target);
        }
        if state.persistent_modes.is_some()
            && state
                .model
                .config
                .coarse_to_fine_bootstrap_active(decision_step, state.particles.len())
        {
            return self.refine_persistent_bootstrap(
                state,
                decision_step,
                resident_step,
                topology_control,
            );
        }
        if state.persistent_modes.is_some() {
            return self.reallocate_persistent_modes(
                state,
                decision_step,
                resident_step,
                state.model.config.target_leaves,
            );
        }
        if local_detail_topology_required(
            &state.model,
            decision_step,
            state.particles.len(),
            topology_control,
        ) {
            let detail_already_captured = state.local_detail_topology_captured;
            let topology_result = match topology_control {
                AdaptiveTopologyControl::PairedLocalDetail => {
                    validate_resident_paired_local_detail_material(state)?;
                    self.apply_paired_local_detail_topology(
                        &mut state.resident,
                        state.model.config.base_rule_footprint(),
                        state.model.config.perception,
                        state.model.config.paired_topology_split_radius_scale,
                        state.model.config.paired_topology_merge_detail_scale,
                        state.model.config.min_reallocation_relative_gain,
                        detail_already_captured,
                    )
                }
                AdaptiveTopologyControl::ContinuousLocalDetail => {
                    validate_resident_continuous_local_detail_material(state)?;
                    self.apply_continuous_local_detail_topology(
                        &mut state.resident,
                        state.model.config.base_rule_footprint(),
                        state.model.config.perception,
                        state.model.config.min_reallocation_relative_gain,
                        state.model.config.max_events_per_interval,
                        detail_already_captured,
                    )
                }
                _ => unreachable!("local-detail topology was checked above"),
            };
            state.local_detail_topology_captured = false;
            topology_result?;
            if topology_control == AdaptiveTopologyControl::ContinuousLocalDetail {
                let resident_step = u32::try_from(resident_step).map_err(|_| {
                    AutomataError::InvalidArgument(
                        "adaptive GPU rollout step exceeds u32".to_owned(),
                    )
                })?;
                self.begin_state_render_transition(&mut state.resident, resident_step)?;
                state.render_transition_start_step = decision_step;
            }
            state.dynamics_since_topology = 0;
            let update = AdaptiveTopologyUpdate {
                step: decision_step,
                initial_leaf_count: state.particles.len(),
                final_leaf_count: state.particles.len(),
                split_events: if topology_control == AdaptiveTopologyControl::ContinuousLocalDetail
                {
                    state.model.config.max_events_per_interval
                } else {
                    1
                },
                merge_events: if topology_control == AdaptiveTopologyControl::ContinuousLocalDetail
                {
                    state.model.config.max_events_per_interval
                } else {
                    1
                },
                elapsed_ms: 0.0,
            };
            state.last_topology = Some(update);
            return Ok(update);
        }
        self.synchronize_adaptive_particles(state)?;
        let old_displayed = displayed_scale_by_id(state)?;
        for (index, id) in state.particles.particle_id.iter().enumerate() {
            state.particles.render_footprint[index] =
                old_displayed[id] / state.display_scale_per_footprint.max(f32::MIN_POSITIVE);
        }
        let update = apply_adaptive_topology_at_step_with_control(
            &state.model,
            &mut state.particles,
            decision_step,
            state.dynamics_since_topology,
            topology_control,
        )?;
        state.dynamics_since_topology = 0;
        let render_target_footprint = target_render_footprints(&state.model, &state.particles);
        state.render_from_scale = state
            .particles
            .particle_id
            .iter()
            .enumerate()
            .map(|(index, id)| {
                old_displayed.get(id).copied().unwrap_or(
                    state.particles.render_footprint[index] * state.display_scale_per_footprint,
                )
            })
            .collect();
        let resident_step = u32::try_from(resident_step).map_err(|_| {
            AutomataError::InvalidArgument("adaptive GPU rollout step exceeds u32".to_string())
        })?;
        if update.initial_leaf_count != update.final_leaf_count {
            if install_persistent_modes_after_topology(
                &state.model,
                decision_step,
                update.initial_leaf_count,
            ) {
                let quadrature_points = state.model.config.coarse_quadrature_points;
                self.install_persistent_modes(
                    state,
                    &render_target_footprint,
                    resident_step,
                    quadrature_points,
                )?;
            } else {
                self.install_direct_resident(state, &render_target_footprint, resident_step)?;
            }
        } else {
            let update_masks = material_update_masks(&state.particles)?;
            self.update_state_material_with_support_policy(
                &mut state.resident,
                &state.particles.positions,
                &state.grid,
                WgpuMaterialStateInit {
                    represented_measure: &state.particles.represented_measure,
                    particle_ids: Some(&state.particles.particle_id),
                    update_masks: Some(&update_masks),
                    bandwidth: &state.particles.bandwidth,
                    support_bins: Some(wgpu_support_bins(&state.model, &state.particles.bandwidth)),
                    covariance: &state.particles.covariance,
                    state_jacobian: &state.particles.state_jacobian,
                    closure_mode: Some(&state.particles.closure_mode),
                    closure_basis: Some(&state.particles.closure_basis),
                    closure_phase: Some(&state.particles.closure_phase),
                    render_from_scale: &state.render_from_scale,
                    render_target_footprint: &render_target_footprint,
                    display_scale_per_footprint: state.display_scale_per_footprint,
                    render_transition_steps: state.model.config.render_transition_steps,
                },
            )?;
        }
        if state.persistent_modes.is_none() {
            self.begin_state_render_transition(&mut state.resident, resident_step)?;
        }
        state.render_transition_start_step = resident_step as usize;
        state.last_topology = Some(update);
        Ok(update)
    }

    fn reallocate_active_quadrature(
        &self,
        state: &mut WgpuAdaptiveNpaState,
        decision_step: usize,
        resident_step: usize,
        topology_control: AdaptiveTopologyControl,
    ) -> AutomataResult<AdaptiveTopologyUpdate> {
        self.synchronize_adaptive_particles(state)?;
        let old_displayed = displayed_scale_by_id(state)?;
        for (index, id) in state.particles.particle_id.iter().enumerate() {
            state.particles.render_footprint[index] =
                old_displayed[id] / state.display_scale_per_footprint.max(f32::MIN_POSITIVE);
        }
        let update = apply_adaptive_topology_at_step_with_control(
            &state.model,
            &mut state.particles,
            decision_step,
            state.dynamics_since_topology,
            topology_control,
        )?;
        state.dynamics_since_topology = 0;
        let render_target_footprint = target_render_footprints(&state.model, &state.particles);
        state.render_from_scale = state
            .particles
            .particle_id
            .iter()
            .enumerate()
            .map(|(index, id)| {
                old_displayed.get(id).copied().unwrap_or(
                    state.particles.render_footprint[index] * state.display_scale_per_footprint,
                )
            })
            .collect();
        let resident_step = u32::try_from(resident_step).map_err(|_| {
            AutomataError::InvalidArgument("adaptive GPU rollout step exceeds u32".to_owned())
        })?;
        state.persistent_modes = None;
        if state.particles.bootstrap_templates.is_empty() {
            self.install_direct_resident(state, &render_target_footprint, resident_step)?;
        } else {
            self.install_persistent_modes(
                state,
                &render_target_footprint,
                resident_step,
                state.model.config.coarse_quadrature_points,
            )?;
        }
        state.render_transition_start_step = decision_step;
        state.last_topology = Some(update);
        Ok(update)
    }

    fn reallocate_persistent_modes(
        &self,
        state: &mut WgpuAdaptiveNpaState,
        decision_step: usize,
        resident_step: usize,
        target_leaves: usize,
    ) -> AutomataResult<AdaptiveTopologyUpdate> {
        if resident_fine_selector_supported(state, target_leaves) {
            return self.reallocate_persistent_modes_from_resident_diagnostics(
                state,
                decision_step,
                resident_step,
            );
        }

        let started = std::time::Instant::now();
        let initial_leaf_count = state.particles.len();
        let old_groups = adaptive_template_child_groups(&state.particles);
        let child_display_scale = displayed_scale_by_material_child_id(state)?;
        self.synchronize_adaptive_particles(state)?;
        let fine = restore_adaptive_particles_from_templates(&state.particles)?;
        if fine.len() != state.model.config.bootstrap_fine_leaf_count() {
            return Err(AutomataError::InvalidModel(format!(
                "persistent reallocation restored {} fine rows instead of {}",
                fine.len(),
                state.model.config.bootstrap_fine_leaf_count(),
            )));
        }
        let restricted = progressively_restrict_adaptive_particles_to_leaf_budget(
            &state.model,
            &state.particles,
            &fine,
            target_leaves,
        )?;
        let new_groups = adaptive_template_child_groups(&restricted);
        let split_events = old_groups.difference(&new_groups).count();
        let merge_events = new_groups.difference(&old_groups).count();
        if split_events + merge_events == 0 {
            state.dynamics_since_topology = 0;
            return Ok(AdaptiveTopologyUpdate {
                step: decision_step,
                initial_leaf_count,
                final_leaf_count: initial_leaf_count,
                split_events: 0,
                merge_events: 0,
                elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
            });
        }

        state.particles = restricted;
        state.render_from_scale = reallocated_render_from_scale(
            &state.particles,
            &child_display_scale,
            state.display_scale_per_footprint,
            &state.model,
        )?;
        let target_render_footprint = target_render_footprints(&state.model, &state.particles);
        state.persistent_modes = None;
        let resident_step = u32::try_from(resident_step).map_err(|_| {
            AutomataError::InvalidArgument("adaptive GPU rollout step exceeds u32".to_owned())
        })?;
        let quadrature_points = state.model.config.coarse_quadrature_points;
        self.install_persistent_modes(
            state,
            &target_render_footprint,
            resident_step,
            quadrature_points,
        )?;
        state.render_transition_start_step = decision_step;
        state.dynamics_since_topology = 0;
        let update = AdaptiveTopologyUpdate {
            step: decision_step,
            initial_leaf_count,
            final_leaf_count: state.particles.len(),
            split_events,
            merge_events,
            elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
        };
        state.last_topology = Some(update);
        Ok(update)
    }

    fn reallocate_persistent_modes_from_resident_diagnostics(
        &self,
        state: &mut WgpuAdaptiveNpaState,
        decision_step: usize,
        resident_step: usize,
    ) -> AutomataResult<AdaptiveTopologyUpdate> {
        let started = std::time::Instant::now();
        let initial_leaf_count = state.particles.len();
        let old_groups = adaptive_template_child_groups(&state.particles);
        let child_display_scale = displayed_scale_by_material_child_id(state)?;
        self.set_stable_sorted_cells_enabled(&mut state.resident, true);
        let captured = self.capture_adaptive_diagnostics(
            &mut state.resident,
            state.model.config.base_rule_footprint(),
            state.model.config.perception,
        );
        self.set_stable_sorted_cells_enabled(&mut state.resident, state.force_stable_sorted_cells);
        let (mode_positions, mode_states, diagnostics) = captured?;
        let snapshot = persistent_fine_snapshot_from_diagnostics(
            &state.particles,
            state.persistent_modes.as_ref().ok_or_else(|| {
                AutomataError::InvalidModel(
                    "resident diagnostic reallocation lost its persistent state".to_owned(),
                )
            })?,
            &mode_positions,
            &mode_states,
            &diagnostics,
        )?;
        let restricted = if state.model.config.hierarchical_restriction_schedule
            == AdaptiveRestrictionSchedule::BoundedRollingRecompute
            && !state.particles.bootstrap_templates.is_empty()
        {
            bounded_persistent_restriction(
                &state.model,
                &state.particles,
                &snapshot,
                decision_step,
            )?
        } else {
            let hierarchy = AdaptiveProxyHierarchy::build(
                &snapshot.particles,
                2 * snapshot.particles.spatial_dims,
            )?;
            let merge_costs = learned_level_one_merge_costs_from_precomputed(
                &state.model,
                &snapshot.particles,
                &hierarchy,
                &snapshot.normalized_features,
                &snapshot.base_update,
                &snapshot.observed_spacing,
                &snapshot.accepted_degree,
                snapshot.feature_dims,
            )?;
            restrict_adaptive_particles_to_target_by_merge_cost_with_hierarchy(
                &state.model,
                &snapshot.particles,
                &hierarchy,
                &merge_costs,
            )?
        };
        let new_groups = adaptive_template_child_groups(&restricted);
        let split_events = old_groups.difference(&new_groups).count();
        let merge_events = new_groups.difference(&old_groups).count();

        state.particles = restricted;
        state.dynamics_since_topology = 0;
        if split_events + merge_events > 0 {
            state.render_from_scale = reallocated_render_from_scale(
                &state.particles,
                &child_display_scale,
                state.display_scale_per_footprint,
                &state.model,
            )?;
            let target_render_footprint = target_render_footprints(&state.model, &state.particles);
            let resident_step = u32::try_from(resident_step).map_err(|_| {
                AutomataError::InvalidArgument("adaptive GPU rollout step exceeds u32".to_owned())
            })?;
            self.remap_persistent_active_partition(state, &target_render_footprint, resident_step)?;
            state.render_transition_start_step = decision_step;
        }
        let update = AdaptiveTopologyUpdate {
            step: decision_step,
            initial_leaf_count,
            final_leaf_count: state.particles.len(),
            split_events,
            merge_events,
            elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
        };
        state.last_topology = Some(update);
        Ok(update)
    }
}

fn bounded_persistent_restriction(
    model: &AdaptiveNpaModel,
    current: &AdaptiveParticleSet,
    snapshot: &PersistentFineDiagnosticSnapshot,
    decision_step: usize,
) -> AutomataResult<AdaptiveParticleSet> {
    let fine = &snapshot.particles;
    let branch_factor = 2 * fine.spatial_dims;
    let reduction_per_group = branch_factor.saturating_sub(1);
    let required_groups = fine
        .len()
        .checked_sub(model.config.target_leaves)
        .filter(|reduction| reduction.is_multiple_of(reduction_per_group))
        .map(|reduction| reduction / reduction_per_group)
        .ok_or_else(|| {
            AutomataError::InvalidModel(
                "bounded rolling restriction requires a reachable canonical budget".to_owned(),
            )
        })?;
    if current.bootstrap_templates.len() != required_groups {
        return Err(AutomataError::InvalidModel(format!(
            "bounded rolling restriction expected {required_groups} current groups, got {}",
            current.bootstrap_templates.len(),
        )));
    }
    let replacement_budget = model
        .config
        .topology_event_budget(decision_step, current.len())
        .min(required_groups);
    let retained_count = required_groups - replacement_budget;
    if retained_count == 0 {
        let hierarchy = AdaptiveProxyHierarchy::build(fine, branch_factor)?;
        let merge_costs = learned_level_one_merge_costs_from_precomputed(
            model,
            fine,
            &hierarchy,
            &snapshot.normalized_features,
            &snapshot.base_update,
            &snapshot.observed_spacing,
            &snapshot.accepted_degree,
            snapshot.feature_dims,
        )?;
        return restrict_adaptive_particles_to_target_by_merge_cost_with_hierarchy(
            model,
            fine,
            &hierarchy,
            &merge_costs,
        );
    }

    let fine_index_by_id = fine
        .particle_id
        .iter()
        .copied()
        .enumerate()
        .map(|(row, id)| (id, row))
        .collect::<BTreeMap<_, _>>();
    let mut ranked_existing = current
        .bootstrap_templates
        .iter()
        .map(|template| {
            let mut child_ids = template
                .children
                .iter()
                .map(|child| child.particle_id)
                .collect::<Vec<_>>();
            child_ids.sort_unstable();
            if child_ids.len() != branch_factor {
                return Err(AutomataError::InvalidModel(format!(
                    "bounded rolling restriction expected {branch_factor}-child canonical groups",
                )));
            }
            let child_rows = child_ids
                .iter()
                .map(|id| {
                    fine_index_by_id.get(id).copied().ok_or_else(|| {
                        AutomataError::InvalidModel(format!(
                            "bounded rolling child {id} is absent from the fine snapshot"
                        ))
                    })
                })
                .collect::<AutomataResult<Vec<_>>>()?;
            Ok((0.0_f32, child_ids, child_rows, template.parent_id))
        })
        .collect::<AutomataResult<Vec<_>>>()?;
    let mut old_partition_rows = ranked_existing
        .iter()
        .flat_map(|(_, _, rows, _)| rows.iter().copied())
        .collect::<Vec<_>>();
    let mut old_partition_membership = vec![false; fine.len()];
    for row in &old_partition_rows {
        if std::mem::replace(&mut old_partition_membership[*row], true) {
            return Err(AutomataError::InvalidModel(
                "bounded rolling current groups overlap".to_owned(),
            ));
        }
    }
    old_partition_rows.extend(
        old_partition_membership
            .iter()
            .enumerate()
            .filter_map(|(row, grouped)| (!grouped).then_some(row)),
    );
    let old_hierarchy =
        AdaptiveProxyHierarchy::build_with_leaf_order(fine, branch_factor, old_partition_rows)?;
    let old_merge_costs = learned_level_one_merge_costs_from_precomputed(
        model,
        fine,
        &old_hierarchy,
        &snapshot.normalized_features,
        &snapshot.base_update,
        &snapshot.observed_spacing,
        &snapshot.accepted_degree,
        snapshot.feature_dims,
    )?;
    for (group, existing) in ranked_existing.iter_mut().enumerate() {
        existing.0 = old_merge_costs[group];
    }
    ranked_existing
        .sort_unstable_by(|lhs, rhs| lhs.0.total_cmp(&rhs.0).then_with(|| lhs.1.cmp(&rhs.1)));

    let retained = ranked_existing
        .into_iter()
        .take(retained_count)
        .collect::<Vec<_>>();
    let mut consumed = vec![false; fine.len()];
    for (_, _, rows, _) in &retained {
        for row in rows {
            if std::mem::replace(&mut consumed[*row], true) {
                return Err(AutomataError::InvalidModel(
                    "bounded rolling restriction retained overlapping groups".to_owned(),
                ));
            }
        }
    }
    let available_rows = consumed
        .iter()
        .enumerate()
        .filter_map(|(row, consumed)| (!consumed).then_some(row))
        .collect::<Vec<_>>();
    let available = adaptive_particle_subset(fine, &available_rows)?;
    let hierarchy = AdaptiveProxyHierarchy::build(&available, branch_factor)?;
    let feature_dims = snapshot.feature_dims;
    let update_dims = model.rule.config.update_dims();
    let normalized_features = gather_rows(
        &snapshot.normalized_features,
        &available_rows,
        feature_dims,
        "normalized features",
    )?;
    let base_update = gather_rows(
        &snapshot.base_update,
        &available_rows,
        update_dims,
        "base update",
    )?;
    let observed_spacing = gather_rows(
        &snapshot.observed_spacing,
        &available_rows,
        1,
        "observed spacing",
    )?;
    let accepted_degree = available_rows
        .iter()
        .map(|row| snapshot.accepted_degree[*row])
        .collect::<Vec<_>>();
    let merge_costs = learned_level_one_merge_costs_from_precomputed(
        model,
        &available,
        &hierarchy,
        &normalized_features,
        &base_update,
        &observed_spacing,
        &accepted_degree,
        feature_dims,
    )?;
    let remaining_target = model
        .config
        .target_leaves
        .checked_sub(retained_count)
        .ok_or_else(|| {
            AutomataError::InvalidModel(
                "bounded rolling retained-group count exceeds target budget".to_owned(),
            )
        })?;
    let view = hierarchy.material_cut_from_level_one_merge_costs(
        &available,
        remaining_target,
        &merge_costs,
    )?;
    let mut groups = retained
        .iter()
        .map(|(_, _, rows, _)| rows.clone())
        .collect::<Vec<_>>();
    groups.extend(view.members.iter().map(|member| {
        hierarchy
            .member_leaf_indices(*member)
            .iter()
            .map(|row| available_rows[*row])
            .collect::<Vec<_>>()
    }));
    groups.sort_unstable_by_key(|members| members.iter().copied().min().unwrap_or(usize::MAX));
    let mut restricted = restricted_seed_from_fine_groups(fine, groups)?;

    let retained_parent = retained
        .iter()
        .map(|(_, ids, _, parent)| (ids.clone(), *parent))
        .collect::<BTreeMap<_, _>>();
    let mut next_parent = current
        .particle_id
        .iter()
        .chain(&fine.particle_id)
        .copied()
        .max()
        .unwrap_or_default()
        .saturating_add(1);
    for template in &mut restricted.bootstrap_templates {
        let generated_parent = template.parent_id;
        let mut child_ids = template
            .children
            .iter()
            .map(|child| child.particle_id)
            .collect::<Vec<_>>();
        child_ids.sort_unstable();
        let parent = retained_parent.get(&child_ids).copied().unwrap_or_else(|| {
            let parent = next_parent;
            next_parent = next_parent.saturating_add(1);
            parent
        });
        let row = restricted
            .particle_id
            .iter()
            .position(|id| *id == generated_parent)
            .ok_or_else(|| {
                AutomataError::InvalidModel(
                    "bounded rolling aggregate is absent from its material cut".to_owned(),
                )
            })?;
        restricted.particle_id[row] = parent;
        template.parent_id = parent;
    }
    restricted.next_id = next_parent;
    restricted
        .bootstrap_templates
        .sort_unstable_by_key(|template| template.parent_id);
    restricted.validate()?;
    Ok(restricted)
}

fn gather_rows<T: Copy>(
    values: &[T],
    rows: &[usize],
    width: usize,
    label: &str,
) -> AutomataResult<Vec<T>> {
    if width == 0
        || rows
            .iter()
            .any(|row| row.saturating_add(1).saturating_mul(width) > values.len())
    {
        return Err(AutomataError::InvalidModel(format!(
            "bounded rolling {label} row shape is invalid"
        )));
    }
    Ok(rows
        .iter()
        .flat_map(|row| values[*row * width..(*row + 1) * width].iter().copied())
        .collect())
}

fn resident_fine_selector_supported(state: &WgpuAdaptiveNpaState, target_leaves: usize) -> bool {
    let Some(persistent) = state.persistent_modes.as_ref() else {
        return false;
    };
    state.model.config.coarse_dynamics == AdaptiveCoarseDynamics::PersistentFineQuadrature
        && state.model.config.hierarchical_restriction_policy
            == AdaptiveHierarchyRestrictionPolicy::LearnedController
        && state.model.config.hierarchical_restriction_arity == AdaptiveRestrictionArity::Canonical
        && target_leaves == state.model.config.target_leaves
        && persistent.mode_mask_members.len() == state.model.config.bootstrap_fine_leaf_count()
        && persistent
            .mode_mask_members
            .iter()
            .all(|members| members.len() == 1 && (members[0].1 - 1.0).abs() <= 1.0e-6)
}

fn persistent_fine_snapshot_from_diagnostics(
    active: &AdaptiveParticleSet,
    persistent: &WgpuPersistentAdaptiveState,
    mode_positions: &[[f32; 4]],
    mode_states: &[f32],
    diagnostics: &WgpuAdaptiveDiagnostics,
) -> AutomataResult<PersistentFineDiagnosticSnapshot> {
    let mode_count = persistent.mode_mask_members.len();
    if mode_positions.len() != mode_count
        || mode_states.len() != mode_count * active.state_dims
        || persistent.mode_covariance.len() != mode_count
        || persistent.mode_bandwidth.len() != mode_count
        || diagnostics.normalized_features.len() != mode_count * diagnostics.feature_dims
        || diagnostics.base_update.len() != mode_count * diagnostics.output_dims
        || diagnostics.observed_spacing.len() != mode_count
        || diagnostics.accepted_degree.len() != mode_count
    {
        return Err(AutomataError::InvalidModel(
            "resident fine diagnostic snapshot has incompatible shapes".to_owned(),
        ));
    }
    let mut particles = restore_adaptive_particles_from_templates(active)?;
    if particles.len() != mode_count || !particles.bootstrap_templates.is_empty() {
        return Err(AutomataError::InvalidModel(format!(
            "resident fine diagnostic snapshot restored {} material rows for {mode_count} modes",
            particles.len(),
        )));
    }
    let source_modes = fine_source_modes(&particles.particle_id, &persistent.mode_mask_members)?;
    let mut normalized_features = vec![0.0; mode_count * diagnostics.feature_dims];
    let mut base_update = vec![0.0; mode_count * diagnostics.output_dims];
    let mut observed_spacing = vec![0.0; mode_count];
    let mut accepted_degree = vec![0; mode_count];
    for (row, mode) in source_modes.into_iter().enumerate() {
        particles.positions[row] = mode_positions[mode];
        particles.states[row * active.state_dims..(row + 1) * active.state_dims].copy_from_slice(
            &mode_states[mode * active.state_dims..(mode + 1) * active.state_dims],
        );
        particles.covariance[row] = persistent.mode_covariance[mode];
        particles.bandwidth[row] = persistent.mode_bandwidth[mode];
        normalized_features[row * diagnostics.feature_dims..(row + 1) * diagnostics.feature_dims]
            .copy_from_slice(
                &diagnostics.normalized_features
                    [mode * diagnostics.feature_dims..(mode + 1) * diagnostics.feature_dims],
            );
        base_update[row * diagnostics.output_dims..(row + 1) * diagnostics.output_dims]
            .copy_from_slice(
                &diagnostics.base_update
                    [mode * diagnostics.output_dims..(mode + 1) * diagnostics.output_dims],
            );
        observed_spacing[row] = diagnostics.observed_spacing[mode];
        accepted_degree[row] = diagnostics.accepted_degree[mode];
    }
    particles.validate()?;
    Ok(PersistentFineDiagnosticSnapshot {
        particles,
        normalized_features,
        base_update,
        observed_spacing,
        accepted_degree,
        feature_dims: diagnostics.feature_dims,
    })
}

fn fine_source_modes(
    fine_particle_ids: &[u64],
    mode_mask_members: &[Vec<(u64, f32)>],
) -> AutomataResult<Vec<usize>> {
    let mut mode_by_particle_id = BTreeMap::new();
    for (mode, members) in mode_mask_members.iter().enumerate() {
        let [(particle_id, weight)] = members.as_slice() else {
            return Err(AutomataError::InvalidModel(
                "resident fine diagnostic snapshot requires one material child per mode".to_owned(),
            ));
        };
        if (*weight - 1.0).abs() > 1.0e-6
            || mode_by_particle_id.insert(*particle_id, mode).is_some()
        {
            return Err(AutomataError::InvalidModel(
                "resident fine diagnostic mode lineage is not one-to-one".to_owned(),
            ));
        }
    }
    if fine_particle_ids.len() != mode_by_particle_id.len() {
        return Err(AutomataError::InvalidModel(
            "resident fine material and diagnostic mode counts differ".to_owned(),
        ));
    }
    fine_particle_ids
        .iter()
        .map(|particle_id| {
            mode_by_particle_id
                .get(particle_id)
                .copied()
                .ok_or_else(|| {
                    AutomataError::InvalidModel(format!(
                        "resident fine material child {particle_id} has no diagnostic mode",
                    ))
                })
        })
        .collect()
}

fn install_persistent_modes_after_topology(
    model: &AdaptiveNpaModel,
    decision_step: usize,
    initial_leaf_count: usize,
) -> bool {
    model.config.coarse_dynamics == AdaptiveCoarseDynamics::PersistentFineQuadrature
        && !model
            .config
            .coarse_to_fine_bootstrap_active(decision_step, initial_leaf_count)
}

fn synchronize_quadrature_material(
    particles: &mut AdaptiveParticleSet,
    persistent: &WgpuPersistentAdaptiveState,
    mode_positions: &[[f32; 4]],
    mode_states: &[f32],
) -> AutomataResult<()> {
    let mode_count = persistent.mode_rows.len();
    if mode_positions.len() != mode_count
        || mode_states.len() != mode_count * particles.state_dims
        || persistent.initial_mode_positions.len() != mode_count
        || persistent.initial_mode_states.len() != mode_states.len()
        || persistent.mode_covariance.len() != mode_count
        || persistent.mode_bandwidth.len() != mode_count
        || persistent.mode_mask_members.len() != mode_count
        || persistent.mode_offsets.len() < particles.len() + 1
    {
        return Err(AutomataError::InvalidModel(
            "persistent host synchronization has incompatible mode metadata".to_owned(),
        ));
    }

    for active_row in 0..particles.len() {
        let start = persistent.mode_offsets[active_row] as usize;
        let end = persistent.mode_offsets[active_row + 1] as usize;
        let mut covariance = [0.0_f32; 9];
        let mut bandwidth = 0.0_f32;
        for cursor in start..end {
            let mode = persistent.mode_rows[cursor] as usize;
            let weight = persistent.mode_weights[cursor];
            bandwidth += weight * persistent.mode_bandwidth[mode];
            for row_axis in 0..particles.spatial_dims {
                let row_delta =
                    mode_positions[mode][row_axis] - particles.positions[active_row][row_axis];
                for col_axis in 0..particles.spatial_dims {
                    let col_delta =
                        mode_positions[mode][col_axis] - particles.positions[active_row][col_axis];
                    covariance[row_axis * 3 + col_axis] += weight
                        * (persistent.mode_covariance[mode][row_axis * 3 + col_axis]
                            + row_delta * col_delta);
                }
            }
        }
        particles.covariance[active_row] = covariance;
        particles.bandwidth[active_row] = bandwidth;
    }

    if !persistent.persistent_detail {
        return particles.validate();
    }

    let child_modes = persistent
        .mode_mask_members
        .iter()
        .enumerate()
        .flat_map(|(mode, members)| members.iter().map(move |(id, _)| (*id, mode)))
        .collect::<BTreeMap<_, _>>();
    let initial_child_by_id = persistent
        .bootstrap_templates
        .iter()
        .flat_map(|template| {
            template
                .children
                .iter()
                .map(|child| (child.particle_id, child))
        })
        .collect::<BTreeMap<_, _>>();
    for template in &mut particles.bootstrap_templates {
        for child in &mut template.children {
            let mode = child_modes
                .get(&child.particle_id)
                .copied()
                .ok_or_else(|| {
                    AutomataError::InvalidModel(format!(
                        "persistent child {} has no dynamics mode",
                        child.particle_id,
                    ))
                })?;
            let initial_child = initial_child_by_id.get(&child.particle_id).copied();
            if initial_child.is_none()
                && (persistent.mode_mask_members[mode].len() != 1
                    || persistent.mode_mask_members[mode][0].0 != child.particle_id)
            {
                return Err(AutomataError::InvalidModel(format!(
                    "persistent child {} has no unambiguous initial material state",
                    child.particle_id,
                )));
            }
            for (axis, mode_position) in mode_positions[mode]
                .iter()
                .take(particles.spatial_dims)
                .enumerate()
            {
                let initial = initial_child
                    .map(|initial| initial.position[axis])
                    .unwrap_or(persistent.initial_mode_positions[mode][axis]);
                child.position[axis] =
                    initial + mode_position - persistent.initial_mode_positions[mode][axis];
            }
            for channel in 0..particles.state_dims {
                let initial = initial_child
                    .map(|initial| initial.state[channel])
                    .unwrap_or(
                        persistent.initial_mode_states[mode * particles.state_dims + channel],
                    );
                child.state[channel] = initial + mode_states[mode * particles.state_dims + channel]
                    - persistent.initial_mode_states[mode * particles.state_dims + channel];
            }
        }
    }
    Ok(())
}

fn next_topology_step(
    model: &AdaptiveNpaModel,
    completed_steps: usize,
    leaf_count: usize,
) -> Option<usize> {
    let next_step = completed_steps.checked_add(1)?;
    let scheduled_restriction = if leaf_count <= model.config.target_leaves
        || model.config.hierarchical_restriction_step == 0
    {
        None
    } else if model
        .config
        .hierarchical_restriction_leaf_delta_per_interval
        == 0
    {
        (model.config.hierarchical_restriction_step >= next_step)
            .then_some(model.config.hierarchical_restriction_step)
    } else {
        let lower = next_step.max(model.config.hierarchical_restriction_step);
        aligned_step_from(
            model.config.hierarchical_restriction_step,
            lower,
            model.config.topology_interval,
        )
    };

    let topology_end = if model.config.topology_end_step > 0 {
        model.config.topology_end_step
    } else {
        usize::MAX
    };
    let bootstrap = (model.config.bootstrap_end_step > 0
        && leaf_count < model.config.bootstrap_target_leaf_count())
    .then(|| {
        let lower = next_step.max(model.config.topology_start_step);
        aligned_step(lower, model.config.topology_interval)
    })
    .flatten()
    .filter(|step| *step <= model.config.bootstrap_end_step && *step <= topology_end);
    let steady_lower = next_step
        .max(model.config.steady_topology_start_step())
        .max(
            if model.config.bootstrap_end_step > 0
                && leaf_count < model.config.bootstrap_target_leaf_count()
            {
                model.config.bootstrap_end_step.saturating_add(1)
            } else {
                0
            },
        );
    let steady = aligned_step(steady_lower, model.config.steady_topology_interval())
        .filter(|step| *step <= topology_end);
    let periodic = bootstrap.or(steady);
    match (scheduled_restriction, periodic) {
        (Some(restriction), Some(periodic)) => Some(restriction.min(periodic)),
        (Some(restriction), None) => Some(restriction),
        (None, periodic) => periodic,
    }
}

fn adaptive_dynamics_require_stable_cells(
    next_topology_step: Option<usize>,
    _topology_control: AdaptiveTopologyControl,
    force_stable_sorted_cells: bool,
) -> bool {
    force_stable_sorted_cells || next_topology_step.is_some()
}

fn local_detail_capture_required(
    model: &AdaptiveNpaModel,
    decision_step: usize,
    leaf_count: usize,
    topology_control: AdaptiveTopologyControl,
    direct_active_material: bool,
) -> bool {
    direct_active_material
        && local_detail_topology_required(model, decision_step, leaf_count, topology_control)
}

fn local_detail_topology_required(
    model: &AdaptiveNpaModel,
    decision_step: usize,
    leaf_count: usize,
    topology_control: AdaptiveTopologyControl,
) -> bool {
    local_detail_topology_control(topology_control)
        && model
            .config
            .scheduled_restriction_target(decision_step, leaf_count)
            .is_none()
}

const fn local_detail_topology_control(control: AdaptiveTopologyControl) -> bool {
    matches!(
        control,
        AdaptiveTopologyControl::PairedLocalDetail | AdaptiveTopologyControl::ContinuousLocalDetail
    )
}

fn aligned_step(step: usize, interval: usize) -> Option<usize> {
    step.checked_add(interval.checked_sub(1)?)
        .map(|value| value / interval * interval)
}

fn aligned_step_from(origin: usize, step: usize, interval: usize) -> Option<usize> {
    let elapsed = step.checked_sub(origin)?;
    aligned_step(elapsed, interval)?.checked_add(origin)
}

fn gpu_local_max_neighbors(model: &AdaptiveNpaModel) -> AutomataResult<usize> {
    match model.config.perception.graph_policy {
        AdaptiveGraphPolicy::RawSupport => Ok(0),
        AdaptiveGraphPolicy::DirectedTopK => Ok(model.config.perception.max_neighbors),
        AdaptiveGraphPolicy::MutualTopK => Err(AutomataError::InvalidArgument(
            "adaptive WGPU inference does not support mutual-top-k local perception".to_string(),
        )),
    }
}

fn validate_resident_paired_local_detail_material(
    state: &WgpuAdaptiveNpaState,
) -> AutomataResult<()> {
    state.particles.validate()?;
    if state.model.config.max_events_per_interval != 1 {
        return Err(AutomataError::InvalidModel(
            "resident paired local-detail topology currently requires exactly one pair per interval"
                .to_owned(),
        ));
    }
    if !state.particles.bootstrap_templates.is_empty()
        || state.resident.particle_count != state.particles.len()
        || state.resident.batch_size != 1
    {
        return Err(AutomataError::InvalidModel(
            "resident paired local-detail topology requires one direct active-material trajectory without hidden templates"
                .to_owned(),
        ));
    }
    let fine_measure = state
        .particles
        .represented_measure
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let classify =
        |measure: f32, units: f32| (measure / fine_measure - units).abs() <= 2.0e-4 * units;
    let coarse = state
        .particles
        .represented_measure
        .iter()
        .filter(|measure| classify(**measure, 4.0))
        .count();
    let fine = state
        .particles
        .represented_measure
        .iter()
        .filter(|measure| classify(**measure, 1.0))
        .count();
    if coarse == 0 || fine < 4 || coarse + fine != state.particles.len() {
        let mut unit_histogram = BTreeMap::<i32, usize>::new();
        let mut maximum_unit_residual = 0.0_f32;
        for measure in &state.particles.represented_measure {
            let units = *measure / fine_measure;
            let rounded = units.round();
            maximum_unit_residual = maximum_unit_residual.max((units - rounded).abs());
            *unit_histogram.entry(rounded as i32).or_default() += 1;
        }
        return Err(AutomataError::InvalidModel(format!(
            "resident paired local-detail topology requires one/four-unit rows, got histogram={unit_histogram:?} fine_measure={fine_measure:.9e} max_unit_residual={maximum_unit_residual:.3e}"
        )));
    }
    Ok(())
}

fn validate_resident_continuous_local_detail_material(
    state: &WgpuAdaptiveNpaState,
) -> AutomataResult<()> {
    state.particles.validate()?;
    if state.model.config.material_seed_layout
        != super::AdaptiveMaterialSeedLayout::GradedContinuous
    {
        return Err(AutomataError::InvalidModel(
            "resident continuous local-detail topology requires graded-continuous material"
                .to_owned(),
        ));
    }
    if !(1..=super::CONTINUOUS_LOCAL_DETAIL_MAX_EXCHANGES)
        .contains(&state.model.config.max_events_per_interval)
    {
        return Err(AutomataError::InvalidModel(format!(
            "resident continuous local-detail topology requires 1..={} exchanges per interval",
            super::CONTINUOUS_LOCAL_DETAIL_MAX_EXCHANGES,
        )));
    }
    if !state.particles.bootstrap_templates.is_empty()
        || state.resident.particle_count != state.particles.len()
        || state.resident.batch_size != 1
    {
        return Err(AutomataError::InvalidModel(
            "resident continuous local-detail topology requires one direct active-material trajectory without hidden templates"
                .to_owned(),
        ));
    }
    let total = state.particles.total_measure() as f32;
    let mean = total / state.particles.len() as f32;
    let tolerance = 2.0e-4 * mean;
    let coarse = state
        .particles
        .represented_measure
        .iter()
        .filter(|measure| **measure > mean + tolerance)
        .count();
    let fine = state
        .particles
        .represented_measure
        .iter()
        .filter(|measure| **measure + tolerance < mean)
        .count();
    if !total.is_finite()
        || total <= 0.0
        || coarse < state.model.config.max_events_per_interval
        || fine < state.model.config.max_events_per_interval
    {
        return Err(AutomataError::InvalidModel(format!(
            "resident continuous local-detail topology requires finite material and enough rows on both sides of the mean for its event budget, got total={total:.9e} coarse={coarse} fine={fine} budget={}",
            state.model.config.max_events_per_interval,
        )));
    }
    Ok(())
}

fn target_render_scales(
    model: &AdaptiveNpaModel,
    particles: &AdaptiveParticleSet,
    display_scale_per_footprint: f32,
) -> AutomataResult<Vec<f32>> {
    Ok(target_render_footprints(model, particles)
        .into_iter()
        .map(|footprint| footprint * display_scale_per_footprint)
        .collect())
}

fn target_render_footprints(model: &AdaptiveNpaModel, particles: &AdaptiveParticleSet) -> Vec<f32> {
    particles
        .represented_measure
        .iter()
        .map(|measure| {
            model
                .config
                .render_footprint(super::material_footprint_radius(
                    *measure,
                    particles.spatial_dims,
                ))
        })
        .collect()
}

fn displayed_scale_by_id(state: &WgpuAdaptiveNpaState) -> AutomataResult<BTreeMap<u64, f32>> {
    let displayed = displayed_render_scales(state)?;
    Ok(state
        .particles
        .particle_id
        .iter()
        .copied()
        .zip(displayed)
        .collect())
}

fn displayed_render_scales(state: &WgpuAdaptiveNpaState) -> AutomataResult<Vec<f32>> {
    let targets = target_render_scales(
        &state.model,
        &state.particles,
        state.display_scale_per_footprint,
    )?;
    if state.render_from_scale.len() != targets.len() {
        return Err(AutomataError::InvalidModel(format!(
            "adaptive render transition has {} source scales for {} targets",
            state.render_from_scale.len(),
            targets.len(),
        )));
    }
    let duration = state.model.config.render_transition_steps as usize;
    let age = state
        .completed_steps
        .saturating_sub(state.render_transition_start_step);
    let mut progress = if duration == 0 {
        1.0
    } else {
        (age as f32 / duration as f32).clamp(0.0, 1.0)
    };
    progress = progress * progress * (3.0 - 2.0 * progress);
    Ok(state
        .render_from_scale
        .iter()
        .zip(targets)
        .map(|(initial, target)| {
            let initial = initial.max(f32::MIN_POSITIVE);
            let target = target.max(f32::MIN_POSITIVE);
            (initial.log2() + progress * (target.log2() - initial.log2())).exp2()
        })
        .collect())
}

fn displayed_scale_by_material_child_id(
    state: &WgpuAdaptiveNpaState,
) -> AutomataResult<BTreeMap<u64, f32>> {
    let displayed = displayed_scale_by_id(state)?;
    let templates = state
        .particles
        .bootstrap_templates
        .iter()
        .map(|template| (template.parent_id, template))
        .collect::<BTreeMap<_, _>>();
    let mut child_scale = BTreeMap::new();
    for particle_id in &state.particles.particle_id {
        let scale = displayed[particle_id];
        if let Some(template) = templates.get(particle_id) {
            for child in &template.children {
                child_scale.insert(child.particle_id, scale);
            }
        } else {
            child_scale.insert(*particle_id, scale);
        }
    }
    Ok(child_scale)
}

fn reallocated_render_from_scale(
    particles: &AdaptiveParticleSet,
    child_display_scale: &BTreeMap<u64, f32>,
    display_scale_per_footprint: f32,
    model: &AdaptiveNpaModel,
) -> AutomataResult<Vec<f32>> {
    let templates = particles
        .bootstrap_templates
        .iter()
        .map(|template| (template.parent_id, template))
        .collect::<BTreeMap<_, _>>();
    particles
        .particle_id
        .iter()
        .enumerate()
        .map(|(row, particle_id)| {
            let target = model.config.render_footprint(particles.footprint(row))
                * display_scale_per_footprint;
            if let Some(template) = templates.get(particle_id) {
                let total = template
                    .children
                    .iter()
                    .map(|child| child.represented_measure)
                    .sum::<f32>()
                    .max(f32::MIN_POSITIVE);
                let log_scale = template.children.iter().try_fold(0.0_f32, |sum, child| {
                    child_display_scale
                        .get(&child.particle_id)
                        .copied()
                        .map(|scale| sum + child.represented_measure / total * scale.ln())
                        .ok_or_else(|| {
                            AutomataError::InvalidModel(format!(
                                "reallocated child {} has no displayed scale",
                                child.particle_id,
                            ))
                        })
                })?;
                Ok(merged_render_from_scale(log_scale.exp(), target))
            } else if let Some(scale) = child_display_scale.get(particle_id) {
                Ok(*scale)
            } else {
                Ok(target)
            }
        })
        .collect()
}

/// A merged leaf cannot begin below its material-conserving target radius.
/// Doing so would require opacity above one to preserve represented measure,
/// which the renderer cannot express and would transiently dim the image.
fn merged_render_from_scale(child_geometric_scale: f32, target_scale: f32) -> f32 {
    child_geometric_scale.max(target_scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NpaConfig, NpaModel, ParticleSeed};

    #[test]
    fn merged_render_transition_never_requires_opacity_above_one() {
        assert_eq!(merged_render_from_scale(0.5, 1.0), 1.0);
        assert_eq!(merged_render_from_scale(2.0, 1.0), 2.0);
    }

    #[test]
    fn resident_schedule_fills_bootstrap_budget_before_steady_restriction() {
        let mut config = super::super::AdaptiveNpaConfig::growing_2d();
        config.min_leaves = 1_024;
        config.target_leaves = 3_070;
        config.bootstrap_target_leaves = 4_096;
        config.max_leaves = 4_096;
        config.initial_leaves = 1_024;
        config.bootstrap_fine_leaves = 4_096;
        config.topology_interval = 1;
        config.topology_start_step = 1;
        config.topology_end_step = 8;
        config.bootstrap_end_step = 8;
        config.bootstrap_events_per_interval = 128;
        config.hierarchical_restriction_step = 128;
        config.hierarchical_restriction_leaf_delta_per_interval = 96;
        config.steady_topology_start_step = 257;
        config.coarse_dynamics = AdaptiveCoarseDynamics::PersistentFineQuadrature;
        let model =
            AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 7), config, 9)
                .unwrap();

        for completed in 0..8 {
            let leaves = 1_024 + completed * 128 * 3;
            assert_eq!(
                next_topology_step(&model, completed, leaves),
                Some(completed + 1)
            );
        }
        assert_eq!(next_topology_step(&model, 8, 4_096), Some(128));
        assert_eq!(next_topology_step(&model, 128, 4_000), Some(129));
        assert_eq!(next_topology_step(&model, 137, 3_136), Some(138));
        assert_eq!(next_topology_step(&model, 128, 3_070), None);
        assert!(!install_persistent_modes_after_topology(&model, 1, 1_024));
        assert!(!install_persistent_modes_after_topology(&model, 8, 3_712));
        assert!(install_persistent_modes_after_topology(&model, 128, 4_096));
    }

    #[test]
    fn pending_topology_stabilizes_the_trajectory_for_every_controller() {
        assert!(adaptive_dynamics_require_stable_cells(
            Some(256),
            AdaptiveTopologyControl::Learned,
            false,
        ));
        assert!(adaptive_dynamics_require_stable_cells(
            Some(256),
            AdaptiveTopologyControl::PairedLocalDetail,
            false,
        ));
        assert!(!adaptive_dynamics_require_stable_cells(
            None,
            AdaptiveTopologyControl::Learned,
            false,
        ));
        assert!(adaptive_dynamics_require_stable_cells(
            None,
            AdaptiveTopologyControl::Learned,
            true,
        ));
    }

    #[test]
    fn scheduled_restriction_does_not_capture_local_detail_on_fine_rows() {
        let mut config = super::super::AdaptiveNpaConfig::growing_2d();
        config.min_leaves = 40;
        config.initial_leaves = 64;
        config.target_leaves = 40;
        config.max_leaves = 64;
        config.bootstrap_fine_leaves = 64;
        config.hierarchical_bootstrap_seed = true;
        config.hierarchical_restriction_step = 8;
        config.topology_interval = 8;
        config.topology_start_step = 16;
        config.steady_topology_start_step = 16;
        config.max_events_per_interval = 1;
        let model =
            AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 7), config, 9)
                .unwrap();

        assert!(!local_detail_capture_required(
            &model,
            8,
            64,
            AdaptiveTopologyControl::PairedLocalDetail,
            true,
        ));
        assert!(!local_detail_topology_required(
            &model,
            8,
            64,
            AdaptiveTopologyControl::PairedLocalDetail,
        ));
        assert!(local_detail_capture_required(
            &model,
            16,
            40,
            AdaptiveTopologyControl::PairedLocalDetail,
            true,
        ));
        assert!(local_detail_topology_required(
            &model,
            16,
            40,
            AdaptiveTopologyControl::PairedLocalDetail,
        ));
    }

    #[test]
    fn persistent_mode_mapping_tracks_nested_bootstrap_without_rebuilding_modes() {
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
        config.coarse_dynamics = AdaptiveCoarseDynamics::PersistentFineQuadrature;
        config.bootstrap_quadrature_points = 4;
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
        let layout = quadrature_layout_with_points(&model, &particles, true, 4).unwrap();
        assert_eq!(layout.particles.len(), 64);

        for (step, expected) in [(1, 28), (2, 40), (3, 52), (4, 64)] {
            apply_hierarchical_bootstrap_refinement(&model, &mut particles, step, 1).unwrap();
            let (offsets, rows, weights) = persistent_restriction_mapping_from_material_partition(
                &particles,
                &layout.update_mask_members,
                &layout.particles.represented_measure,
            )
            .unwrap();
            assert_eq!(particles.len(), expected);
            assert_eq!(offsets.len(), expected + 1);
            assert_eq!(rows.len(), 64);
            assert_eq!(weights.len(), 64);
            assert!(offsets.windows(2).all(|range| range[0] < range[1]));
            for range in offsets.windows(2) {
                let sum = weights[range[0] as usize..range[1] as usize]
                    .iter()
                    .sum::<f32>();
                assert!((sum - 1.0).abs() <= 1.0e-6);
            }
        }
        assert!(offsets_are_identity_for_fine_partition(
            &persistent_restriction_mapping_from_material_partition(
                &particles,
                &layout.update_mask_members,
                &layout.particles.represented_measure,
            )
            .unwrap()
            .0,
        ));
    }

    fn offsets_are_identity_for_fine_partition(offsets: &[u32]) -> bool {
        offsets
            .iter()
            .enumerate()
            .all(|(index, value)| *value as usize == index)
    }

    #[test]
    fn fine_diagnostic_rows_follow_material_identity_instead_of_mode_order() {
        let modes = vec![vec![(31, 1.0)], vec![(11, 1.0)], vec![(21, 1.0)]];
        assert_eq!(
            fine_source_modes(&[11, 21, 31], &modes).unwrap(),
            vec![1, 2, 0],
        );
        assert!(fine_source_modes(&[11, 21], &modes).is_err());
        assert!(fine_source_modes(&[11, 21, 31], &[vec![(11, 0.5), (31, 0.5)]]).is_err());
    }

    #[test]
    fn persistent_restriction_mapping_reserves_an_inert_capacity_tail() {
        assert_eq!(
            pad_persistent_restriction_mapping(vec![0, 2, 4], 4).unwrap(),
            vec![0, 2, 4, 4, 4],
        );
        assert!(pad_persistent_restriction_mapping(vec![0, 2, 4], 1).is_err());
    }

    #[test]
    #[ignore = "real-device WGPU active-quadrature parity test"]
    fn active_quadrature_wgpu_matches_cpu_reference() {
        let mut config = super::super::AdaptiveNpaConfig::growing_2d();
        config.min_leaves = 16;
        config.initial_leaves = 16;
        config.target_leaves = 16;
        config.max_leaves = 64;
        config.bootstrap_fine_leaves = 64;
        config.bootstrap_end_step = 1;
        config.coarse_dynamics = AdaptiveCoarseDynamics::FineQuadrature;
        config.coarse_quadrature_points = 0;
        config.local_residual_scale = 0.0;
        config.proxy.context_scale = 0.0;
        let model = AdaptiveNpaModel::seeded(
            NpaModel::upstream_seeded(NpaConfig::growing_2d(), 7),
            config,
            11,
        )
        .unwrap();
        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let particles = super::super::seed_adaptive_particles_scaled(
            &model,
            16,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            0.1,
        )
        .unwrap();
        let rollout = super::super::AdaptiveRolloutConfig {
            steps: 3,
            dt: 1.0,
            update_prob: 0.5,
            seed: 13,
            bandwidth_adaptation_enabled: false,
            topology_enabled: false,
            snapshot_interval: 3,
        };
        let cpu = super::super::run_adaptive_rollout(&model, particles.clone(), rollout).unwrap();

        let executor = WgpuAutomataExecutor::new_blocking().unwrap();
        let grid = HashGridConfig::growing_2d();
        let mut gpu = executor
            .create_adaptive_state(
                &model,
                particles,
                &grid,
                1.0,
                WgpuNeighborMode::CooperativeSortedCells,
                0.5,
                13,
            )
            .unwrap();
        let report = executor
            .step_adaptive_state_many(&mut gpu, 3, false)
            .unwrap();
        executor.synchronize_adaptive_particles(&mut gpu).unwrap();

        assert_eq!(report.dynamics_particle_count, 16);
        assert_eq!(report.interaction_particle_count, 64);
        assert_eq!(gpu.particles.len(), cpu.particles.len());
        let max_position_error = gpu
            .particles
            .positions
            .iter()
            .zip(&cpu.particles.positions)
            .flat_map(|(gpu, cpu)| (0..2).map(move |axis| (gpu[axis] - cpu[axis]).abs()))
            .fold(0.0_f32, f32::max);
        let max_state_error = gpu
            .particles
            .states
            .iter()
            .zip(&cpu.particles.states)
            .map(|(gpu, cpu)| (gpu - cpu).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_position_error <= 5.0e-4,
            "active quadrature position error {max_position_error}"
        );
        assert!(
            max_state_error <= 1.0e-3,
            "active quadrature state error {max_state_error}"
        );
    }

    #[test]
    #[ignore = "real-device WGPU restriction test"]
    fn persistent_mode_restriction_is_weighted_on_device() {
        let model = NpaModel::seeded(NpaConfig::growing_2d(), 3);
        let grid = HashGridConfig::growing_2d();
        let state_dims = model.config.state_dims;
        let internal_positions = [
            [0.0, 2.0, 0.0, 0.0],
            [4.0, 6.0, 0.0, 0.0],
            [8.0, 10.0, 0.0, 0.0],
        ];
        let mut internal_states = vec![0.0; 3 * state_dims];
        for channel in 0..state_dims {
            internal_states[channel] = 1.0;
            internal_states[state_dims + channel] = 5.0;
            internal_states[2 * state_dims + channel] = 9.0;
        }
        let active_positions = [[0.0; 4]; 2];
        let active_states = vec![0.0; 2 * state_dims];
        let executor = WgpuAutomataExecutor::new_blocking().unwrap();
        let internal = executor
            .create_state(
                &model,
                &internal_positions,
                &internal_states,
                1,
                3,
                &grid,
                1.0,
            )
            .unwrap();
        let active = executor
            .create_state(&model, &active_positions, &active_states, 1, 2, &grid, 1.0)
            .unwrap();
        let restriction = executor
            .create_persistent_mode_restriction(
                &internal,
                &active,
                &[0, 2, 3],
                &[0, 1, 2],
                &[0.25, 0.75, 1.0],
            )
            .unwrap();
        executor
            .restrict_persistent_modes(&restriction, &internal, &active)
            .unwrap();
        let (positions, states) = executor.read_positions_states(&active).unwrap();
        assert!((positions[0][0] - 3.0).abs() <= 1.0e-6);
        assert!((positions[0][1] - 5.0).abs() <= 1.0e-6);
        assert!((positions[1][0] - 8.0).abs() <= 1.0e-6);
        assert!((states[0] - 4.0).abs() <= 1.0e-6);
        assert!((states[state_dims] - 9.0).abs() <= 1.0e-6);
    }

    #[test]
    #[ignore = "real-device WGPU persistent rollout parity test"]
    fn persistent_mode_rollout_matches_cpu_reference() {
        let base = NpaModel::seeded(NpaConfig::growing_2d(), 7);
        let grid = HashGridConfig::growing_2d();
        let mut config = super::super::AdaptiveNpaConfig::growing_2d();
        config.min_leaves = 40;
        config.target_leaves = 40;
        config.max_leaves = 64;
        config.initial_leaves = 64;
        config.bootstrap_fine_leaves = 64;
        config.hierarchical_bootstrap_seed = true;
        config.hierarchical_restriction_step = 2;
        config.hierarchical_restriction_policy =
            super::super::AdaptiveHierarchyRestrictionPolicy::DynamicsDetail;
        config.topology_interval = 4;
        config.steady_topology_interval = 4;
        config.topology_start_step = 4;
        config.steady_topology_start_step = 4;
        config.topology_end_step = 4;
        config.local_residual_scale = 0.0;
        config.proxy.context_scale = 0.0;
        config.coarse_dynamics = AdaptiveCoarseDynamics::PersistentFineQuadrature;
        config.coarse_quadrature_points = 2;
        let model = AdaptiveNpaModel::seeded(base, config, 11).unwrap();
        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let particles = super::super::seed_adaptive_particles_scaled(
            &model,
            64,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            grid.eps,
        )
        .unwrap();
        let fused_particles = particles.clone();
        let cpu = super::super::run_adaptive_rollout(
            &model,
            particles.clone(),
            super::super::AdaptiveRolloutConfig {
                steps: 4,
                dt: 1.0,
                update_prob: 1.0,
                seed: 13,
                bandwidth_adaptation_enabled: false,
                topology_enabled: true,
                snapshot_interval: 4,
            },
        )
        .unwrap();
        let executor = WgpuAutomataExecutor::new_blocking().unwrap();
        let mut gpu = executor
            .create_adaptive_state(
                &model,
                particles,
                &grid,
                1.0,
                WgpuNeighborMode::SubgroupCooperativeSortedCells,
                1.0,
                13,
            )
            .unwrap();
        let report = executor
            .step_adaptive_state_many(&mut gpu, 4, true)
            .unwrap();
        executor.synchronize_adaptive_particles(&mut gpu).unwrap();
        let mut fused = executor
            .create_adaptive_state(
                &model,
                fused_particles,
                &grid,
                1.0,
                WgpuNeighborMode::SubgroupCooperativeSortedCells,
                1.0,
                13,
            )
            .unwrap();
        let gaussian_buffers = executor.create_gaussian_buffers(64).unwrap();
        let gaussian = executor
            .create_gaussian_bind_group(&gaussian_buffers.refs(), 64)
            .unwrap();
        let fused_report = executor
            .step_adaptive_state_many_into_gaussian_bind_group(&mut fused, &gaussian, 4, true)
            .unwrap();
        executor.synchronize_adaptive_particles(&mut fused).unwrap();
        assert_eq!(report.resident_particle_count, 40);
        assert_eq!(report.dynamics_particle_count, 48);
        assert_eq!(fused_report.completed_steps, report.completed_steps);
        assert_eq!(
            fused_report.resident_particle_count,
            report.resident_particle_count
        );
        assert_eq!(
            fused_report.dynamics_particle_count,
            report.dynamics_particle_count
        );
        assert_eq!(fused_report.particle_steps, report.particle_steps);
        assert_eq!(
            fused_report.topology_updates.len(),
            report.topology_updates.len()
        );
        for (fused, reference) in fused_report
            .topology_updates
            .iter()
            .zip(&report.topology_updates)
        {
            assert_eq!(fused.step, reference.step);
            assert_eq!(fused.initial_leaf_count, reference.initial_leaf_count);
            assert_eq!(fused.final_leaf_count, reference.final_leaf_count);
            assert_eq!(fused.split_events, reference.split_events);
            assert_eq!(fused.merge_events, reference.merge_events);
        }
        let max_fused_position_error = gpu
            .particles
            .positions
            .iter()
            .zip(&fused.particles.positions)
            .flat_map(|(reference, fused)| (0..2).map(|axis| (reference[axis] - fused[axis]).abs()))
            .fold(0.0_f32, f32::max);
        let max_fused_state_error = gpu
            .particles
            .states
            .iter()
            .zip(&fused.particles.states)
            .map(|(reference, fused)| (reference - fused).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_fused_position_error <= 2.0e-3 && max_fused_state_error <= 2.0e-3,
            "persistent fused/unfused mismatch: position={max_fused_position_error} state={max_fused_state_error}",
        );
        let max_position_error = cpu
            .particles
            .positions
            .iter()
            .zip(&gpu.particles.positions)
            .flat_map(|(cpu, gpu)| (0..2).map(|axis| (cpu[axis] - gpu[axis]).abs()))
            .fold(0.0_f32, f32::max);
        let max_state_error = cpu
            .particles
            .states
            .iter()
            .zip(&gpu.particles.states)
            .map(|(cpu, gpu)| (cpu - gpu).abs())
            .fold(0.0_f32, f32::max);
        let max_covariance_error = cpu
            .particles
            .covariance
            .iter()
            .zip(&gpu.particles.covariance)
            .flat_map(|(cpu, gpu)| (0..9).map(|element| (cpu[element] - gpu[element]).abs()))
            .fold(0.0_f32, f32::max);
        let max_child_state_error = cpu
            .particles
            .bootstrap_templates
            .iter()
            .zip(&gpu.particles.bootstrap_templates)
            .flat_map(|(cpu, gpu)| cpu.children.iter().zip(&gpu.children))
            .flat_map(|(cpu, gpu)| cpu.state.iter().zip(&gpu.state))
            .map(|(cpu, gpu)| (cpu - gpu).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_position_error <= 2.0e-3
                && max_state_error <= 2.0e-3
                && max_covariance_error <= 2.0e-3
                && max_child_state_error <= 2.0e-3,
            "persistent CPU/WGPU mismatch: position={max_position_error} state={max_state_error} covariance={max_covariance_error} child_state={max_child_state_error}",
        );
    }

    #[test]
    #[ignore = "real-device WGPU recurrent topology determinism test"]
    fn persistent_mode_recurrent_topology_is_reproducible() {
        let base = NpaModel::seeded(NpaConfig::growing_2d(), 7);
        let grid = HashGridConfig::growing_2d();
        let mut config = super::super::AdaptiveNpaConfig::growing_2d();
        config.min_leaves = 193;
        config.target_leaves = 193;
        config.max_leaves = 256;
        config.initial_leaves = 256;
        config.bootstrap_fine_leaves = 256;
        config.hierarchical_bootstrap_seed = true;
        config.hierarchical_restriction_step = 4;
        config.hierarchical_restriction_policy =
            super::super::AdaptiveHierarchyRestrictionPolicy::DynamicsDetail;
        config.topology_interval = 16;
        config.steady_topology_interval = 16;
        config.topology_start_step = 16;
        config.steady_topology_start_step = 16;
        config.topology_end_step = 64;
        config.local_residual_scale = 0.0;
        config.proxy.context_scale = 0.0;
        config.coarse_dynamics = AdaptiveCoarseDynamics::PersistentFineQuadrature;
        config.coarse_quadrature_points = 4;
        let model = AdaptiveNpaModel::seeded(base, config, 11).unwrap();
        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let particles = super::super::seed_adaptive_particles_scaled(
            &model,
            256,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            grid.eps,
        )
        .unwrap();
        let executor = WgpuAutomataExecutor::new_blocking().unwrap();
        let create_state = || {
            executor
                .create_adaptive_state(
                    &model,
                    particles.clone(),
                    &grid,
                    1.0,
                    WgpuNeighborMode::SubgroupCooperativeSortedCells,
                    1.0,
                    13,
                )
                .unwrap()
        };
        let mut lhs = create_state();
        let mut rhs = create_state();

        let lhs_report = executor
            .step_adaptive_state_many(&mut lhs, 64, true)
            .unwrap();
        let rhs_report = executor
            .step_adaptive_state_many(&mut rhs, 64, true)
            .unwrap();
        executor.synchronize_adaptive_particles(&mut lhs).unwrap();
        executor.synchronize_adaptive_particles(&mut rhs).unwrap();

        assert_eq!(lhs_report.completed_steps, rhs_report.completed_steps);
        assert_eq!(lhs_report.particle_steps, rhs_report.particle_steps);
        assert_eq!(
            lhs_report
                .topology_updates
                .iter()
                .map(|update| (
                    update.step,
                    update.initial_leaf_count,
                    update.final_leaf_count,
                    update.split_events,
                    update.merge_events,
                ))
                .collect::<Vec<_>>(),
            rhs_report
                .topology_updates
                .iter()
                .map(|update| (
                    update.step,
                    update.initial_leaf_count,
                    update.final_leaf_count,
                    update.split_events,
                    update.merge_events,
                ))
                .collect::<Vec<_>>(),
        );
        assert_eq!(lhs.particles.positions, rhs.particles.positions);
        assert_eq!(lhs.particles.states, rhs.particles.states);
        assert_eq!(
            lhs.particles.represented_measure,
            rhs.particles.represented_measure
        );
        assert_eq!(
            lhs.particles.render_footprint,
            rhs.particles.render_footprint
        );
        for (footprint, from_scale) in lhs
            .particles
            .render_footprint
            .iter()
            .zip(&lhs.render_from_scale)
        {
            let expected = from_scale / lhs.display_scale_per_footprint;
            assert!((footprint - expected).abs() <= 1.0e-6);
        }
        assert_eq!(lhs.particles.particle_id, rhs.particles.particle_id);
    }

    #[test]
    fn topology_schedule_respects_delayed_start() {
        let mut adaptive = super::super::AdaptiveNpaConfig::growing_2d();
        adaptive.topology_interval = 8;
        adaptive.topology_start_step = 21;
        let model =
            AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 3), adaptive, 4)
                .unwrap();
        assert_eq!(next_topology_step(&model, 0, 4_096), Some(24));
        assert_eq!(next_topology_step(&model, 24, 4_096), Some(32));
    }

    #[test]
    fn topology_schedule_switches_to_steady_cadence_after_bootstrap() {
        let mut adaptive = super::super::AdaptiveNpaConfig::growing_2d();
        adaptive.initial_leaves = 1_024;
        adaptive.topology_interval = 8;
        adaptive.steady_topology_interval = 32;
        adaptive.bootstrap_end_step = 96;
        let model =
            AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 3), adaptive, 4)
                .unwrap();

        assert_eq!(next_topology_step(&model, 64, 1_024), Some(72));
        assert_eq!(next_topology_step(&model, 96, 4_096), Some(128));
    }

    #[test]
    fn topology_schedule_does_not_skip_delayed_hierarchical_restriction() {
        let mut adaptive = super::super::AdaptiveNpaConfig::growing_2d();
        adaptive.min_leaves = 1_024;
        adaptive.target_leaves = 3_070;
        adaptive.max_leaves = 4_096;
        adaptive.initial_leaves = 4_096;
        adaptive.bootstrap_fine_leaves = 4_096;
        adaptive.hierarchical_restriction_step = 256;
        adaptive.topology_interval = 128;
        adaptive.topology_start_step = 512;
        let model =
            AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 3), adaptive, 4)
                .unwrap();

        assert_eq!(next_topology_step(&model, 0, 4_096), Some(256));
        assert_eq!(next_topology_step(&model, 256, 3_070), Some(512));
    }

    #[test]
    fn render_targets_are_continuous_bounded_and_independent_from_topology_limits() {
        let mut adaptive = super::super::AdaptiveNpaConfig::growing_2d();
        adaptive.reference_footprint = 0.01;
        adaptive.base_rule_footprint = 0.01;
        adaptive.min_footprint = 0.001;
        adaptive.max_footprint = 0.1;
        adaptive.min_render_footprint = 0.004;
        adaptive.max_render_footprint = 0.04;
        adaptive.render_footprint_exponent = 1.2;
        let model =
            AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 3), adaptive, 4)
                .unwrap();
        let count = 32;
        let positions = vec![[0.0; 4]; count];
        let states = vec![0.0; count * model.rule.config.state_dims];
        let mut particles = AdaptiveParticleSet::from_equal_measure(
            positions,
            states,
            2,
            model.rule.config.state_dims,
            std::f32::consts::PI * 0.01_f32.powi(2) * count as f32,
            0.1,
        )
        .unwrap();
        for (index, measure) in particles.represented_measure.iter_mut().enumerate() {
            let radius = 0.002 + index as f32 * 0.0015;
            *measure = std::f32::consts::PI * radius.powi(2);
        }

        let targets = target_render_footprints(&model, &particles);
        assert_eq!(targets.first().copied(), Some(0.004));
        assert_eq!(targets.last().copied(), Some(0.04));
        assert!(targets.windows(2).all(|pair| pair[1] >= pair[0]));
        assert!(
            targets.windows(2).filter(|pair| pair[1] > pair[0]).count() > 8,
            "continuous physical footprints collapsed into discrete render levels"
        );
        assert_eq!(model.config.min_footprint, 0.001);
        assert_eq!(model.config.max_footprint, 0.1);
    }
}
