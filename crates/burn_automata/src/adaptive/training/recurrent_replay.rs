use std::time::Instant;

use super::{
    AdaptiveMultiscaleTrainingBatch, AdaptiveMultiscaleTrainingConfig, AdaptiveReplayBackend,
    multiscale_dataset::raw_update_from_restricted_step,
    multiscale_on_policy::{OnPolicySnapshot, build_batch},
};
use crate::{
    AutomataError, AutomataResult, NpaModel, ParticleSeed,
    adaptive::{
        AdaptiveHierarchyMember, AdaptiveNpaModel, AdaptiveParticleSet, AdaptiveProxyHierarchy,
        dynamics::adaptive_raw_update, features::material_detail_values,
    },
    rollout::{seed_particles_scaled, stable_material_uniform},
};
use burn_automata_kernels::HashGridConfig;

#[cfg(feature = "gpu_wgpu")]
mod gpu;

/// Builds recurrent DAgger labels from a fine teacher coupled to a coarse
/// student by a fixed conservative material cut. Fine child offsets and state
/// residuals persist across oracle steps; before each query only their material
/// mean is recentered on the visited student leaf. This keeps the teacher
/// conditioned on unresolved detail instead of repeatedly collapsing it to the
/// coarse mean.
pub(super) fn adaptive_coupled_fine_replay_batch(
    teacher: &NpaModel,
    grid: &HashGridConfig,
    model: &AdaptiveNpaModel,
    config: &AdaptiveMultiscaleTrainingConfig,
    round: usize,
) -> AutomataResult<AdaptiveMultiscaleTrainingBatch> {
    if config.on_policy_replay_backend == AdaptiveReplayBackend::WgpuResident {
        #[cfg(feature = "gpu_wgpu")]
        {
            return gpu::adaptive_coupled_fine_replay_batch_wgpu(
                teacher, grid, model, config, round,
            );
        }
        #[cfg(not(feature = "gpu_wgpu"))]
        {
            return Err(AutomataError::InvalidArgument(
                "resident coupled-fine replay requires gpu_wgpu".to_string(),
            ));
        }
    }
    let cuts = config
        .cut_leaf_counts
        .iter()
        .copied()
        .filter(|count| *count < config.fine_particle_count)
        .collect::<Vec<_>>();
    if cuts.is_empty() {
        return Err(AutomataError::InvalidArgument(
            "coupled-fine replay requires at least one cut below fine_particle_count".to_string(),
        ));
    }
    let cut_steps = unique_cut_steps(&config.on_policy_cut_steps);

    let started = Instant::now();
    let snapshots_per_trajectory =
        config.on_policy_rollout_steps / config.on_policy_snapshot_interval + 1;
    let mut snapshots = Vec::with_capacity(
        config.on_policy_rollouts * cuts.len() * cut_steps.len() * snapshots_per_trajectory,
    );
    for rollout_index in 0..config.on_policy_rollouts {
        for (cut_index, requested_leaves) in cuts.iter().copied().enumerate() {
            for (cut_step_index, cut_step) in cut_steps.iter().copied().enumerate() {
                let replay_index = cut_index * cut_steps.len() + cut_step_index;
                let seed = replay_seed(config.seed, round, rollout_index, replay_index);
                append_coupled_trajectory(
                    teacher,
                    grid,
                    model,
                    config,
                    seed,
                    rollout_index,
                    requested_leaves,
                    cut_step,
                    &mut snapshots,
                )?;
            }
        }
    }
    build_batch(teacher, model, config, round, snapshots, started)
}

#[cfg(feature = "gpu_wgpu")]
pub(super) fn adaptive_coupled_fine_replay_batch_wgpu_with_executor(
    executor: &crate::gpu::WgpuAutomataExecutor,
    teacher: &NpaModel,
    grid: &HashGridConfig,
    model: &AdaptiveNpaModel,
    config: &AdaptiveMultiscaleTrainingConfig,
    round: usize,
) -> AutomataResult<AdaptiveMultiscaleTrainingBatch> {
    gpu::adaptive_coupled_fine_replay_batch_wgpu_with_executor(
        executor, teacher, grid, model, config, round,
    )
}

#[allow(clippy::too_many_arguments)]
fn append_coupled_trajectory(
    teacher: &NpaModel,
    grid: &HashGridConfig,
    model: &AdaptiveNpaModel,
    config: &AdaptiveMultiscaleTrainingConfig,
    seed: u64,
    rollout_index: usize,
    requested_leaves: usize,
    cut_step: usize,
    snapshots: &mut Vec<OnPolicySnapshot>,
) -> AutomataResult<()> {
    let (positions, states) = seed_particles_scaled(
        1,
        config.fine_particle_count,
        teacher.config.state_dims,
        teacher.config.spatial_dims,
        seed,
        ParticleSeed::UniformCircle,
        config.seed_scale,
    );
    let mut fine = AdaptiveParticleSet::from_equal_measure(
        positions,
        states,
        teacher.config.spatial_dims,
        teacher.config.state_dims,
        config.total_measure,
        config.bandwidth,
    )?;
    for step in 0..cut_step {
        let mask = fine
            .particle_id
            .iter()
            .map(|id| f32::from(stable_material_uniform(seed, step + 1, *id) < config.update_prob))
            .collect::<Vec<_>>();
        let teacher_step = teacher.step_cpu(
            &fine.positions,
            &fine.states,
            1,
            fine.len(),
            grid,
            config.dt,
            Some(&mask),
        )?;
        fine.positions = teacher_step.next_positions;
        fine.states = teacher_step.next_states;
    }
    let initial_step = teacher.step_cpu(
        &fine.positions,
        &fine.states,
        1,
        fine.len(),
        grid,
        config.dt,
        None,
    )?;
    let initial_raw = teacher.forward_update_from_features(&initial_step.perception.features)?;
    let hierarchy = AdaptiveProxyHierarchy::build(&fine, 2 * fine.spatial_dims)?;
    let detail = material_detail_values(
        &fine,
        &initial_raw,
        teacher.config.update_dims(),
        model.config.base_rule_footprint().recip(),
    );
    let mut view = hierarchy.material_cut(
        &fine,
        requested_leaves,
        &detail,
        teacher.config.update_dims() + teacher.config.state_dims + teacher.config.spatial_dims,
    )?;
    super::apply_multiscale_material_bandwidth(&model.config, config, &mut view.particles)?;
    if model.config.closure_recurrent_mode {
        super::super::closure::attach_first_closure_mode(&fine, &hierarchy, &mut view)?;
    }
    let members = view.members;
    let mut student = view.particles;
    let mut fine_detail = fine;

    for step in 0..=config.on_policy_rollout_steps {
        let counterfactual = reconstruct_fine_counterfactual(
            &fine_detail,
            &student,
            &hierarchy,
            &members,
            config.bandwidth,
            model.config.closure_recurrent_mode,
        )?;
        let fine_mask = counterfactual
            .particle_id
            .iter()
            .map(|id| {
                f32::from(
                    stable_material_uniform(seed, cut_step + step + 1, *id) < config.update_prob,
                )
            })
            .collect::<Vec<_>>();
        let teacher_step = teacher.step_cpu(
            &counterfactual.positions,
            &counterfactual.states,
            1,
            counterfactual.len(),
            grid,
            config.dt,
            Some(&fine_mask),
        )?;
        let teacher_dx = teacher_step
            .dx
            .iter()
            .flat_map(|value| value[..counterfactual.spatial_dims].iter().copied())
            .collect::<Vec<_>>();
        let restricted_dx = hierarchy.restrict_values(
            &counterfactual,
            &members,
            &teacher_dx,
            counterfactual.spatial_dims,
        )?;
        let restricted_ds = hierarchy.restrict_values(
            &counterfactual,
            &members,
            &teacher_step.ds,
            counterfactual.state_dims,
        )?;
        let teacher_update = raw_update_from_restricted_step(
            &restricted_dx,
            &restricted_ds,
            &student.bandwidth,
            teacher,
        );
        let closure_target = model
            .config
            .closure_recurrent_mode
            .then(|| {
                closure_mode_teacher_update(
                    model,
                    &counterfactual,
                    &student,
                    &hierarchy,
                    &members,
                    &teacher_dx,
                    &teacher_step.ds,
                    config.dt,
                )
            })
            .transpose()?;
        if step.is_multiple_of(config.on_policy_snapshot_interval) {
            snapshots.push(OnPolicySnapshot {
                particles: student.clone(),
                teacher_update: Some(teacher_update),
                closure_mode_target_update: closure_target
                    .as_ref()
                    .map(|target| target.mode.clone()),
                closure_basis_target_update: closure_target
                    .as_ref()
                    .map(|target| target.basis.clone()),
                captured_perception: None,
                rollout_index,
                step,
            });
        }
        if step == config.on_policy_rollout_steps {
            break;
        }

        fine_detail = counterfactual;
        fine_detail.positions = teacher_step.next_positions;
        fine_detail.states = teacher_step.next_states;

        let coarse_mask = super::super::integration::integration_masks(
            model,
            &student,
            seed,
            cut_step + step + 1,
            config.update_prob,
        );
        let student_update = adaptive_raw_update(model, &student)?;
        let closure_mode_update = if model.config.closure_recurrent_mode {
            let perception = super::super::perception::rule_perception_pair(
                &model.config,
                &model.rule,
                &student,
            )?;
            super::super::dynamics::closure_mode_raw_update(model, &student, &perception)?
        } else {
            None
        };
        let closure_basis_update = if model.config.closure_recurrent_mode {
            let perception = super::super::perception::rule_perception_pair(
                &model.config,
                &model.rule,
                &student,
            )?;
            super::super::dynamics::closure_basis_raw_update(model, &student, &perception)?
        } else {
            None
        };
        integrate_student(
            model,
            &mut student,
            &student_update,
            closure_mode_update.as_deref(),
            closure_basis_update.as_deref(),
            &coarse_mask,
            config.dt,
        )?;
    }
    Ok(())
}

fn reconstruct_fine_counterfactual(
    template: &AdaptiveParticleSet,
    student: &AdaptiveParticleSet,
    hierarchy: &AdaptiveProxyHierarchy,
    members: &[AdaptiveHierarchyMember],
    bandwidth: f32,
    reconstruct_closure_mode: bool,
) -> AutomataResult<AdaptiveParticleSet> {
    if student.len() != members.len()
        || template.spatial_dims != student.spatial_dims
        || template.state_dims != student.state_dims
    {
        return Err(AutomataError::InvalidArgument(
            "coupled fine/student material shape mismatch".to_string(),
        ));
    }
    let mut positions = template.positions.clone();
    let mut states = template.states.clone();
    for (material, member) in members.iter().copied().enumerate() {
        let leaves = hierarchy.member_leaf_indices(member);
        let total = leaves
            .iter()
            .map(|leaf| template.represented_measure[*leaf])
            .sum::<f32>()
            .max(f32::MIN_POSITIVE);
        let mut template_centroid = [0.0_f32; 4];
        let mut template_state_mean = vec![0.0_f32; template.state_dims];
        for leaf in leaves {
            let weight = template.represented_measure[*leaf] / total;
            for (axis, centroid) in template_centroid
                .iter_mut()
                .enumerate()
                .take(template.spatial_dims)
            {
                *centroid += weight * template.positions[*leaf][axis];
            }
            for (channel, mean) in template_state_mean.iter_mut().enumerate() {
                *mean += weight * template.states[*leaf * template.state_dims + channel];
            }
        }
        for leaf in leaves {
            for (axis, centroid) in template_centroid
                .iter()
                .enumerate()
                .take(template.spatial_dims)
            {
                positions[*leaf][axis] =
                    student.positions[material][axis] + template.positions[*leaf][axis] - centroid;
            }
            for channel in 0..template.state_dims {
                states[*leaf * template.state_dims + channel] = student.states
                    [material * template.state_dims + channel]
                    + template.states[*leaf * template.state_dims + channel]
                    - template_state_mean[channel];
            }
        }
    }
    let mut reconstructed = AdaptiveParticleSet::from_equal_measure(
        positions,
        states,
        template.spatial_dims,
        template.state_dims,
        template.total_measure() as f32,
        bandwidth,
    )?;
    if reconstruct_closure_mode {
        super::super::closure::reconstruct_first_closure_state_for_members(
            &mut reconstructed,
            hierarchy,
            members,
            student,
        )?;
    }
    Ok(reconstructed)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn closure_mode_teacher_update(
    model: &AdaptiveNpaModel,
    fine: &AdaptiveParticleSet,
    student: &AdaptiveParticleSet,
    hierarchy: &AdaptiveProxyHierarchy,
    members: &[AdaptiveHierarchyMember],
    teacher_dx: &[f32],
    teacher_ds: &[f32],
    dt: f32,
) -> AutomataResult<ClosureTeacherUpdate> {
    if !dt.is_finite()
        || dt <= 0.0
        || teacher_dx.len() != fine.len() * fine.spatial_dims
        || teacher_ds.len() != fine.len() * fine.state_dims
    {
        return Err(AutomataError::InvalidArgument(
            "coupled-fine closure teacher shape mismatch".to_owned(),
        ));
    }
    let mut next = fine.clone();
    for row in 0..next.len() {
        for axis in 0..next.spatial_dims {
            let index = row * next.spatial_dims + axis;
            next.positions[row][axis] = (next.positions[row][axis] + dt * teacher_dx[index])
                .clamp(model.config.domain_min[axis], model.config.domain_max[axis]);
        }
        for channel in 0..next.state_dims {
            let index = row * next.state_dims + channel;
            next.states[index] += dt * teacher_ds[index];
        }
    }
    let (next_mode, _) = super::super::closure::restrict_first_closure_mode_for_members_oriented(
        &next,
        hierarchy,
        members,
        Some(&student.closure_basis),
    )?;
    let output_dims = model.rule.config.update_dims();
    let mut target = vec![0.0; student.len() * output_dims];
    let mut basis_target = vec![0.0; student.len() * output_dims];
    for row in 0..student.len() {
        if !next_mode.active[row] {
            continue;
        }
        for axis in 0..student.spatial_dims {
            target[row * output_dims + axis] =
                (next_mode.phase[row * 2 + axis] - student.closure_phase[row * 2 + axis]) / dt;
        }
        for channel in 0..student.state_dims {
            let mode_index = row * student.state_dims + channel;
            target[row * output_dims + student.spatial_dims + channel] =
                (next_mode.values[mode_index] - student.closure_mode[mode_index]) / dt;
        }
        for component in 0..4 {
            basis_target[row * output_dims + component] = (next_mode.basis[row * 4 + component]
                - student.closure_basis[row * 4 + component])
                / dt;
        }
    }
    Ok(ClosureTeacherUpdate {
        mode: target,
        basis: basis_target,
    })
}

pub(super) struct ClosureTeacherUpdate {
    pub(super) mode: Vec<f32>,
    pub(super) basis: Vec<f32>,
}

fn integrate_student(
    model: &AdaptiveNpaModel,
    particles: &mut AdaptiveParticleSet,
    update: &[f32],
    closure_mode_update: Option<&[f32]>,
    closure_basis_update: Option<&[f32]>,
    mask: &[f32],
    dt: f32,
) -> AutomataResult<()> {
    super::super::integration::integrate_represented_measure_update(
        model, particles, update, mask, dt,
    )?;
    if let Some(update) = closure_mode_update {
        super::super::integration::integrate_closure_mode_update(
            model, particles, update, mask, dt,
        )?;
    }
    if let Some(update) = closure_basis_update {
        super::super::integration::integrate_closure_basis_update(
            model, particles, update, mask, dt,
        )?;
    }
    particles.decrement_cooldown();
    particles.validate()
}

fn replay_seed(seed: u64, round: usize, rollout: usize, cut: usize) -> u64 {
    seed.wrapping_add((round as u64 + 1).wrapping_mul(0xd1b5_4a32_d192_ed03))
        .wrapping_add((rollout as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .wrapping_add((cut as u64).wrapping_mul(0xa24b_aed4_963e_e407))
}

fn unique_cut_steps(cut_steps: &[usize]) -> Vec<usize> {
    let mut steps = cut_steps.to_vec();
    steps.sort_unstable();
    steps.dedup();
    steps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NpaConfig, adaptive::AdaptiveNpaConfig, upstream_growing_2d_hashgrid};

    #[test]
    fn counterfactual_reconstruction_preserves_material_centroids_and_student_state() {
        let base = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 7);
        let model = AdaptiveNpaModel::seeded(base, AdaptiveNpaConfig::growing_2d(), 9).unwrap();
        let (positions, states) = seed_particles_scaled(
            1,
            16,
            model.rule.config.state_dims,
            2,
            11,
            ParticleSeed::UniformCircle,
            0.2,
        );
        let mut fine = AdaptiveParticleSet::from_equal_measure(
            positions,
            states,
            2,
            model.rule.config.state_dims,
            1.0,
            0.1,
        )
        .unwrap();
        for row in 0..fine.len() {
            fine.states[row * fine.state_dims] = 2.0 * fine.positions[row][0];
        }
        let hierarchy = AdaptiveProxyHierarchy::build(&fine, 4).unwrap();
        let view = hierarchy.material_level_cut(&fine, 0).unwrap();
        let first_leaves = hierarchy.member_leaf_indices(view.members[0]);
        let before_delta = fine.positions[first_leaves[0]][0] - fine.positions[first_leaves[1]][0];
        let before_state_delta = fine.states[first_leaves[0] * fine.state_dims]
            - fine.states[first_leaves[1] * fine.state_dims];
        let mut student = view.particles;
        student.positions[0][0] += 0.25;
        student.states[0] += 0.5;
        let reconstructed =
            reconstruct_fine_counterfactual(&fine, &student, &hierarchy, &view.members, 0.1, false)
                .unwrap();
        let leaves = hierarchy.member_leaf_indices(view.members[0]);
        let position_mean = leaves
            .iter()
            .map(|leaf| reconstructed.positions[*leaf][0])
            .sum::<f32>()
            / leaves.len() as f32;
        let state_mean = leaves
            .iter()
            .map(|leaf| reconstructed.states[*leaf * reconstructed.state_dims])
            .sum::<f32>()
            / leaves.len() as f32;
        let after_delta =
            reconstructed.positions[leaves[0]][0] - reconstructed.positions[leaves[1]][0];
        let after_state_delta = reconstructed.states[leaves[0] * reconstructed.state_dims]
            - reconstructed.states[leaves[1] * reconstructed.state_dims];
        assert!((position_mean - student.positions[0][0]).abs() < 1.0e-6);
        assert!((state_mean - student.states[0]).abs() < 1.0e-6);
        assert!((before_delta - after_delta).abs() < 1.0e-6);
        assert!((before_state_delta - after_state_delta).abs() < 1.0e-6);
        assert!((reconstructed.total_measure() - student.total_measure()).abs() < 1.0e-7);
        upstream_growing_2d_hashgrid().validate().unwrap();
    }

    #[test]
    fn recurrent_counterfactual_geometry_is_student_causal() {
        let base = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 13);
        let (positions, states) = seed_particles_scaled(
            1,
            16,
            base.config.state_dims,
            2,
            17,
            ParticleSeed::UniformCircle,
            0.2,
        );
        let fine = AdaptiveParticleSet::from_equal_measure(
            positions,
            states,
            2,
            base.config.state_dims,
            1.0,
            0.1,
        )
        .unwrap();
        let hierarchy = AdaptiveProxyHierarchy::build(&fine, 4).unwrap();
        let mut view = hierarchy.material_level_cut(&fine, 0).unwrap();
        super::super::super::closure::attach_first_closure_mode(&fine, &hierarchy, &mut view)
            .unwrap();
        let student = view.particles;
        let baseline =
            reconstruct_fine_counterfactual(&fine, &student, &hierarchy, &view.members, 0.1, true)
                .unwrap();

        let mut changed_template = fine.clone();
        for (row, position) in changed_template.positions.iter_mut().enumerate() {
            position[0] += 0.05 * (row as f32 * 0.7).sin();
            position[1] += 0.05 * (row as f32 * 1.1).cos();
        }
        let template_independent = reconstruct_fine_counterfactual(
            &changed_template,
            &student,
            &hierarchy,
            &view.members,
            0.1,
            true,
        )
        .unwrap();
        let template_error = baseline
            .positions
            .iter()
            .zip(&template_independent.positions)
            .flat_map(|(baseline, changed)| {
                baseline[..2]
                    .iter()
                    .zip(&changed[..2])
                    .map(|(baseline, changed)| (baseline - changed).abs())
            })
            .fold(0.0_f32, f32::max);
        assert!(template_error < 1.0e-6, "template leakage {template_error}");

        let mut changed_student = student.clone();
        let phase = &mut changed_student.closure_phase[..2];
        let [cosine, sine] = [0.5_f32.cos(), 0.5_f32.sin()];
        [phase[0], phase[1]] = [
            cosine * phase[0] - sine * phase[1],
            sine * phase[0] + cosine * phase[1],
        ];
        let phase_controlled = reconstruct_fine_counterfactual(
            &fine,
            &changed_student,
            &hierarchy,
            &view.members,
            0.1,
            true,
        )
        .unwrap();
        let first_leaves = hierarchy.member_leaf_indices(view.members[0]);
        let phase_change = first_leaves
            .iter()
            .flat_map(|leaf| {
                baseline.positions[*leaf][..2]
                    .iter()
                    .zip(&phase_controlled.positions[*leaf][..2])
                    .map(|(baseline, changed)| (baseline - changed).abs())
            })
            .fold(0.0_f32, f32::max);
        assert!(phase_change > 1.0e-3, "phase control {phase_change}");
    }

    #[test]
    fn recurrent_closure_uses_the_resident_local_feature_contract() {
        let base = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 53);
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.closure_moment_features = true;
        adaptive.closure_recurrent_mode = true;
        adaptive.local_rule_semantics =
            crate::adaptive::AdaptiveLocalRuleSemantics::CompatibleResidual;
        adaptive.compatible_residual_material_features = true;
        adaptive.proxy.enabled = false;
        let model = AdaptiveNpaModel::seeded(base.clone(), adaptive, 59).unwrap();
        assert_eq!(
            model
                .closure_mode_rule
                .as_ref()
                .unwrap()
                .config
                .perception_dims(),
            model
                .local_residual_rule
                .as_ref()
                .unwrap()
                .config
                .perception_dims(),
        );
        assert!(model.uses_canonical_compatible_residual());
    }

    #[test]
    fn temporal_cut_coupled_replay_labels_recurrent_states_deterministically() {
        let base = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 17);
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.proxy.enabled = true;
        adaptive.min_footprint = 0.01;
        adaptive.max_footprint = 0.2;
        adaptive.reference_footprint = 0.025;
        adaptive.base_rule_footprint = 0.025;
        adaptive.closure_moment_features = true;
        adaptive.closure_recurrent_mode = true;
        let mut model = AdaptiveNpaModel::seeded(base.clone(), adaptive, 19).unwrap();
        model.enable_zero_local_residual_rule().unwrap();
        let config = AdaptiveMultiscaleTrainingConfig {
            fine_particle_count: 16,
            cut_leaf_counts: vec![4, 8, 16],
            on_policy_cut_steps: vec![2],
            on_policy_rollout_steps: 2,
            on_policy_rollouts: 1,
            on_policy_snapshot_interval: 1,
            on_policy_rows_per_snapshot: 8,
            on_policy_replay_backend: AdaptiveReplayBackend::CpuReference,
            ..AdaptiveMultiscaleTrainingConfig::default()
        };
        let first = adaptive_coupled_fine_replay_batch(
            &base,
            &upstream_growing_2d_hashgrid(),
            &model,
            &config,
            0,
        )
        .unwrap();
        let second = adaptive_coupled_fine_replay_batch(
            &base,
            &upstream_growing_2d_hashgrid(),
            &model,
            &config,
            0,
        )
        .unwrap();
        assert!(first.rows > 0);
        assert_eq!(first.local_features, second.local_features);
        assert_eq!(first.target_update, second.target_update);
        assert_eq!(
            first.closure_mode_target_update,
            second.closure_mode_target_update
        );
        assert_eq!(
            first.closure_basis_target_update,
            second.closure_basis_target_update
        );
        assert_eq!(
            first.closure_mode_row_weights,
            second.closure_mode_row_weights
        );
        assert_eq!(
            first.closure_mode_target_update.len(),
            first.rows * base.config.update_dims()
        );
        assert_eq!(
            first.closure_basis_target_update.len(),
            first.rows * base.config.update_dims()
        );
        assert!(
            first
                .closure_mode_row_weights
                .iter()
                .any(|weight| *weight > 0.0)
        );
        assert_eq!(first.report.snapshots, 6);
        assert_eq!(first.report.minimum_material_leaves, 4);
        assert!(first.report.maximum_material_leaves <= 8);
        assert!(first.report.mean_teacher_update_error > 0.0);
    }
}
