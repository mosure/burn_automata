use std::time::Instant;

use burn_automata_kernels::{AdaptiveGraphPolicy, AdaptivePerceptionPair, HashGridConfig};

use super::super::{
    AdaptiveMultiscaleTrainingBatch, AdaptiveMultiscaleTrainingConfig,
    multiscale_dataset::captured_perception_output,
    multiscale_on_policy::OnPolicyCapturedPerception,
};
use super::{raw_update_from_restricted_step, replay_seed, unique_cut_steps};
use crate::{
    AutomataError, AutomataResult, NpaModel, ParticleSeed,
    adaptive::{
        AdaptiveHierarchyMember, AdaptiveNpaModel, AdaptiveParticleSet, AdaptiveProxyHierarchy,
        features::material_detail_values,
    },
    gpu::{WgpuAutomataExecutor, WgpuMaterialStateInit, WgpuNeighborMode},
    rollout::seed_particles_scaled,
};

struct CoupledLane {
    rollout_index: usize,
    fine: AdaptiveParticleSet,
    hierarchy: AdaptiveProxyHierarchy,
    members: Vec<AdaptiveHierarchyMember>,
    student: AdaptiveParticleSet,
}

struct KeyedSnapshot {
    rollout_index: usize,
    cut_index: usize,
    step: usize,
    snapshot: super::super::multiscale_on_policy::OnPolicySnapshot,
}

pub(super) fn adaptive_coupled_fine_replay_batch_wgpu(
    teacher: &NpaModel,
    grid: &HashGridConfig,
    model: &AdaptiveNpaModel,
    config: &AdaptiveMultiscaleTrainingConfig,
    round: usize,
) -> AutomataResult<AdaptiveMultiscaleTrainingBatch> {
    let executor = WgpuAutomataExecutor::new_restriction_blocking()?;
    adaptive_coupled_fine_replay_batch_wgpu_with_executor(
        &executor, teacher, grid, model, config, round,
    )
}

pub(super) fn adaptive_coupled_fine_replay_batch_wgpu_with_executor(
    executor: &WgpuAutomataExecutor,
    teacher: &NpaModel,
    grid: &HashGridConfig,
    model: &AdaptiveNpaModel,
    config: &AdaptiveMultiscaleTrainingConfig,
    round: usize,
) -> AutomataResult<AdaptiveMultiscaleTrainingBatch> {
    if model.config.transport_coarse_moments {
        return Err(AutomataError::InvalidArgument(
            "resident coupled-fine replay does not yet transport coarse moments".to_string(),
        ));
    }
    if model.config.proxy.enabled && model.config.proxy.context_scale > 0.0 {
        return Err(AutomataError::InvalidArgument(
            "resident coupled-fine replay requires proxy.context_scale = 0".to_string(),
        ));
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
    let lanes_per_chunk = executor
        .max_independent_trajectory_lanes()
        .min(config.on_policy_rollouts.max(1));
    let mut snapshots = Vec::with_capacity(
        config.on_policy_rollouts
            * cuts.len()
            * cut_steps.len()
            * (config.on_policy_rollout_steps / config.on_policy_snapshot_interval + 1),
    );
    for (cut_index, requested_leaves) in cuts.into_iter().enumerate() {
        for (cut_step_index, cut_step) in cut_steps.iter().copied().enumerate() {
            let replay_index = cut_index * cut_steps.len() + cut_step_index;
            for rollout_start in (0..config.on_policy_rollouts).step_by(lanes_per_chunk) {
                let rollout_end = (rollout_start + lanes_per_chunk).min(config.on_policy_rollouts);
                append_coupled_chunk(
                    executor,
                    teacher,
                    grid,
                    model,
                    config,
                    round,
                    replay_index,
                    requested_leaves,
                    cut_step,
                    rollout_start,
                    rollout_end,
                    &mut snapshots,
                )?;
            }
        }
    }
    snapshots.sort_by_key(|snapshot| (snapshot.rollout_index, snapshot.cut_index, snapshot.step));
    super::super::multiscale_on_policy::build_batch(
        teacher,
        model,
        config,
        round,
        snapshots
            .into_iter()
            .map(|snapshot| snapshot.snapshot)
            .collect(),
        started,
    )
}

#[allow(clippy::too_many_arguments)]
fn append_coupled_chunk(
    executor: &WgpuAutomataExecutor,
    teacher: &NpaModel,
    grid: &HashGridConfig,
    model: &AdaptiveNpaModel,
    config: &AdaptiveMultiscaleTrainingConfig,
    round: usize,
    cut_index: usize,
    requested_leaves: usize,
    cut_step: usize,
    rollout_start: usize,
    rollout_end: usize,
    snapshots: &mut Vec<KeyedSnapshot>,
) -> AutomataResult<()> {
    let profile = std::env::var_os("BURN_AUTOMATA_PROFILE_ADAPTIVE_STAGING").is_some();
    let started = Instant::now();
    let seeds = (rollout_start..rollout_end)
        .map(|rollout_index| replay_seed(config.seed, round, rollout_index, cut_index))
        .collect::<Vec<_>>();
    let mut fine_sets = seeds
        .iter()
        .copied()
        .map(|seed| seed_fine(teacher, config, seed))
        .collect::<AutomataResult<Vec<_>>>()?;
    let seeded = started.elapsed();
    let mut fine_resident =
        create_fine_resident(executor, teacher, grid, config, &seeds, &fine_sets)?;
    let fine_resident_created = started.elapsed();
    if cut_step > 0 {
        executor.step_state_many(&mut fine_resident, cut_step)?;
        let (positions, states) = executor.read_positions_states(&fine_resident)?;
        synchronize_fine_sets(
            &mut fine_sets,
            &positions,
            &states,
            teacher.config.state_dims,
            config.fine_particle_count,
        )?;
    }
    let fine_prerolled = started.elapsed();
    let mut perception = model.config.perception;
    perception.graph_policy = AdaptiveGraphPolicy::RawSupport;
    let (_, _, diagnostics) = executor.capture_adaptive_diagnostics(
        &mut fine_resident,
        model.config.base_rule_footprint(),
        perception,
    )?;
    let initial_raw = diagnostics.base_update;
    let fine_diagnostics_captured = started.elapsed();
    let mut lanes = Vec::with_capacity(seeds.len());
    for (lane, fine) in fine_sets.into_iter().enumerate() {
        let output_dims = teacher.config.update_dims();
        let raw_start = lane * config.fine_particle_count * output_dims;
        let raw_end = raw_start + config.fine_particle_count * output_dims;
        let hierarchy = AdaptiveProxyHierarchy::build(&fine, 2 * fine.spatial_dims)?;
        let detail = material_detail_values(
            &fine,
            &initial_raw[raw_start..raw_end],
            output_dims,
            model.config.base_rule_footprint().recip(),
        );
        let mut view = hierarchy.material_cut(
            &fine,
            requested_leaves,
            &detail,
            output_dims + teacher.config.state_dims + teacher.config.spatial_dims,
        )?;
        super::super::apply_multiscale_material_bandwidth(
            &model.config,
            config,
            &mut view.particles,
        )?;
        if model.config.closure_recurrent_mode {
            super::super::super::closure::attach_first_closure_mode(&fine, &hierarchy, &mut view)?;
        }
        lanes.push(CoupledLane {
            rollout_index: rollout_start + lane,
            fine,
            hierarchy,
            members: view.members,
            student: view.particles,
        });
    }
    let hierarchy_built = started.elapsed();
    let student_count = lanes
        .first()
        .map(|lane| lane.student.len())
        .ok_or_else(|| {
            AutomataError::InvalidArgument("empty coupled-fine WGPU chunk".to_string())
        })?;
    if lanes.iter().any(|lane| lane.student.len() != student_count) {
        return Err(AutomataError::InvalidModel(
            "coupled-fine material cuts produced different per-lane particle counts".to_string(),
        ));
    }
    let mut student_resident =
        create_student_resident(executor, model, grid, config, &seeds, &lanes, student_count)?;
    executor.set_state_step_index(
        &mut student_resident,
        u32::try_from(cut_step).map_err(|_| {
            AutomataError::InvalidArgument("coupled-fine temporal cut step exceeds u32".to_owned())
        })?,
    )?;
    let student_resident_created = started.elapsed();
    let (member_offsets, member_leaves) = pack_members(&lanes, config.fine_particle_count)?;
    let coupling = executor.create_coupled_fine_recenter(
        &fine_resident,
        &student_resident,
        &member_offsets,
        &member_leaves,
        model.config.closure_recurrent_mode,
    )?;
    let coupling_created = started.elapsed();

    let snapshot_steps = (0..=config.on_policy_rollout_steps)
        .step_by(config.on_policy_snapshot_interval)
        .collect::<Vec<_>>();
    let mut pending = Vec::with_capacity(snapshot_steps.len());
    for (snapshot_index, step) in snapshot_steps.iter().copied().enumerate() {
        let next_step = snapshot_steps.get(snapshot_index + 1).copied();
        let advance = next_step.is_some();
        let capture = executor.enqueue_coupled_fine_snapshot(
            &coupling,
            &mut fine_resident,
            &mut student_resident,
            model.config.base_rule_footprint(),
            model.config.perception,
            advance,
        )?;
        pending.push((step, capture));
        if let Some(next_step) = next_step {
            executor.step_coupled_fine_states_many(
                &coupling,
                &mut fine_resident,
                &mut student_resident,
                next_step - step - usize::from(advance),
            )?;
        }
    }
    let snapshots_enqueued = started.elapsed();
    for (step, pending) in pending {
        let captured = executor.read_coupled_fine_snapshot(pending, &student_resident)?;
        append_snapshots(
            teacher,
            grid,
            model,
            config.dt,
            cut_index,
            step,
            captured,
            student_resident.particle_count,
            &lanes,
            snapshots,
        )?;
    }
    if profile {
        let finished = started.elapsed();
        eprintln!(
            "adaptive coupled replay lanes={} cut={} cut_step={} seed_ms={:.3} fine_resident_ms={:.3} fine_preroll_ms={:.3} fine_diagnostics_ms={:.3} hierarchy_ms={:.3} student_resident_ms={:.3} coupling_ms={:.3} enqueue_ms={:.3} readback_append_ms={:.3} total_ms={:.3}",
            seeds.len(),
            requested_leaves,
            cut_step,
            seeded.as_secs_f64() * 1_000.0,
            (fine_resident_created - seeded).as_secs_f64() * 1_000.0,
            (fine_prerolled - fine_resident_created).as_secs_f64() * 1_000.0,
            (fine_diagnostics_captured - fine_prerolled).as_secs_f64() * 1_000.0,
            (hierarchy_built - fine_diagnostics_captured).as_secs_f64() * 1_000.0,
            (student_resident_created - hierarchy_built).as_secs_f64() * 1_000.0,
            (coupling_created - student_resident_created).as_secs_f64() * 1_000.0,
            (snapshots_enqueued - coupling_created).as_secs_f64() * 1_000.0,
            (finished - snapshots_enqueued).as_secs_f64() * 1_000.0,
            finished.as_secs_f64() * 1_000.0,
        );
    }
    Ok(())
}

fn seed_fine(
    teacher: &NpaModel,
    config: &AdaptiveMultiscaleTrainingConfig,
    seed: u64,
) -> AutomataResult<AdaptiveParticleSet> {
    let (positions, states) = seed_particles_scaled(
        1,
        config.fine_particle_count,
        teacher.config.state_dims,
        teacher.config.spatial_dims,
        seed,
        ParticleSeed::UniformCircle,
        config.seed_scale,
    );
    AdaptiveParticleSet::from_equal_measure(
        positions,
        states,
        teacher.config.spatial_dims,
        teacher.config.state_dims,
        config.total_measure,
        config.bandwidth,
    )
}

#[allow(clippy::too_many_arguments)]
fn create_fine_resident(
    executor: &WgpuAutomataExecutor,
    teacher: &NpaModel,
    grid: &HashGridConfig,
    config: &AdaptiveMultiscaleTrainingConfig,
    seeds: &[u64],
    fine_sets: &[AdaptiveParticleSet],
) -> AutomataResult<crate::gpu::WgpuAutomataState> {
    let (
        positions,
        states,
        measure,
        particle_ids,
        bandwidth,
        covariance,
        jacobian,
        _closure_mode,
        _closure_basis,
        _closure_phase,
        render,
    ) = pack_particle_sets(
        fine_sets,
        teacher.config.state_dims,
        teacher.config.spatial_dims,
    );
    let update_masks = fine_sets
        .iter()
        .map(crate::adaptive::gpu::material_update_masks)
        .collect::<AutomataResult<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    executor.create_batched_material_state(
        teacher,
        &positions,
        &states,
        seeds.len(),
        config.fine_particle_count,
        grid,
        config.dt,
        WgpuNeighborMode::SubgroupCooperativeSortedCells,
        config.update_prob,
        seeds,
        WgpuMaterialStateInit {
            represented_measure: &measure,
            particle_ids: Some(&particle_ids),
            update_masks: Some(&update_masks),
            bandwidth: &bandwidth,
            support_bins: None,
            covariance: &covariance,
            state_jacobian: &jacobian,
            closure_mode: None,
            closure_basis: None,
            closure_phase: None,
            render_from_scale: &render,
            render_target_footprint: &render,
            display_scale_per_footprint: 1.0,
            render_transition_steps: 0,
        },
    )
}

fn synchronize_fine_sets(
    fine_sets: &mut [AdaptiveParticleSet],
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    fine_particle_count: usize,
) -> AutomataResult<()> {
    if positions.len() != fine_sets.len() * fine_particle_count
        || states.len() != positions.len() * state_dims
    {
        return Err(AutomataError::InvalidModel(
            "temporal coupled-fine readback shape mismatch".to_owned(),
        ));
    }
    for (lane, fine) in fine_sets.iter_mut().enumerate() {
        let row_start = lane * fine_particle_count;
        let row_end = row_start + fine_particle_count;
        fine.positions = positions[row_start..row_end].to_vec();
        fine.states = states[row_start * state_dims..row_end * state_dims].to_vec();
        fine.validate()?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn create_student_resident(
    executor: &WgpuAutomataExecutor,
    model: &AdaptiveNpaModel,
    grid: &HashGridConfig,
    config: &AdaptiveMultiscaleTrainingConfig,
    seeds: &[u64],
    lanes: &[CoupledLane],
    student_count: usize,
) -> AutomataResult<crate::gpu::WgpuAutomataState> {
    let students = lanes
        .iter()
        .map(|lane| lane.student.clone())
        .collect::<Vec<_>>();
    let (
        positions,
        states,
        measure,
        particle_ids,
        bandwidth,
        covariance,
        jacobian,
        closure_mode,
        closure_basis,
        closure_phase,
        render,
    ) = pack_particle_sets(
        &students,
        model.rule.config.state_dims,
        model.rule.config.spatial_dims,
    );
    let update_masks = lanes
        .iter()
        .map(|lane| {
            crate::adaptive::gpu::material_update_masks_from_hierarchy(
                &lane.fine,
                &lane.hierarchy,
                &lane.members,
            )
        })
        .collect::<AutomataResult<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let gpu_rule = model.gpu_inference_rule()?;
    let mut resident = executor.create_batched_material_state(
        &gpu_rule.rule,
        &positions,
        &states,
        seeds.len(),
        student_count,
        grid,
        config.dt,
        WgpuNeighborMode::SubgroupCooperativeSortedCells,
        config.update_prob,
        seeds,
        WgpuMaterialStateInit {
            represented_measure: &measure,
            particle_ids: Some(&particle_ids),
            update_masks: Some(&update_masks),
            bandwidth: &bandwidth,
            support_bins: None,
            covariance: &covariance,
            state_jacobian: &jacobian,
            closure_mode: Some(&closure_mode),
            closure_basis: Some(&closure_basis),
            closure_phase: Some(&closure_phase),
            render_from_scale: &render,
            render_target_footprint: &render,
            display_scale_per_footprint: 1.0,
            render_transition_steps: 0,
        },
    )?;
    if let Some(local_hidden_start) = gpu_rule.local_hidden_start {
        let max_neighbors = match model.config.perception.graph_policy {
            AdaptiveGraphPolicy::RawSupport => 0,
            AdaptiveGraphPolicy::DirectedTopK => model.config.perception.max_neighbors,
            AdaptiveGraphPolicy::MutualTopK => {
                return Err(AutomataError::InvalidArgument(
                    "resident coupled-fine replay does not support mutual-top-k".to_string(),
                ));
            }
        };
        executor.configure_state_adaptive_local_rule(
            &mut resident,
            gpu_rule.local_rule_mode,
            local_hidden_start,
            model.config.local_residual_scale,
            model.config.base_rule_footprint(),
            model.config.reference_footprint,
            model.config.perception.shepard_epsilon,
            model.config.perception.moment_regularization,
            model.config.perception.moment_condition_limit,
            max_neighbors,
            model.config.perception.pair_scale_power,
        )?;
    }
    if let Some(closure) = &model.closure_mode_rule {
        executor.configure_state_adaptive_closure_rule(&mut resident, closure)?;
    }
    if let Some(closure) = &model.closure_basis_rule {
        executor.configure_state_adaptive_closure_basis_rule(&mut resident, closure)?;
    }
    executor.configure_state_adaptive_integration(
        &mut resident,
        model.config.base_rule_footprint(),
        model.config.expected_coarse_update_mask,
    )?;
    Ok(resident)
}

#[allow(clippy::type_complexity)]
fn pack_particle_sets(
    sets: &[AdaptiveParticleSet],
    state_dims: usize,
    spatial_dims: usize,
) -> (
    Vec<[f32; 4]>,
    Vec<f32>,
    Vec<f32>,
    Vec<u64>,
    Vec<f32>,
    Vec<[f32; 9]>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
) {
    let total = sets.iter().map(AdaptiveParticleSet::len).sum::<usize>();
    let mut positions = Vec::with_capacity(total);
    let mut states = Vec::with_capacity(total * state_dims);
    let mut measure = Vec::with_capacity(total);
    let mut particle_ids = Vec::with_capacity(total);
    let mut bandwidth = Vec::with_capacity(total);
    let mut covariance = Vec::with_capacity(total);
    let mut jacobian = Vec::with_capacity(total * state_dims * spatial_dims);
    let mut closure_mode = Vec::with_capacity(total * state_dims);
    let mut closure_basis = Vec::with_capacity(total * 4);
    let mut closure_phase = Vec::with_capacity(total * 2);
    let mut render = Vec::with_capacity(total);
    for particles in sets {
        positions.extend_from_slice(&particles.positions);
        states.extend_from_slice(&particles.states);
        measure.extend_from_slice(&particles.represented_measure);
        particle_ids.extend_from_slice(&particles.particle_id);
        bandwidth.extend_from_slice(&particles.bandwidth);
        covariance.extend_from_slice(&particles.covariance);
        jacobian.extend_from_slice(&particles.state_jacobian);
        closure_mode.extend_from_slice(&particles.closure_mode);
        closure_basis.extend_from_slice(&particles.closure_basis);
        closure_phase.extend_from_slice(&particles.closure_phase);
        render.extend_from_slice(&particles.render_footprint);
    }
    (
        positions,
        states,
        measure,
        particle_ids,
        bandwidth,
        covariance,
        jacobian,
        closure_mode,
        closure_basis,
        closure_phase,
        render,
    )
}

fn pack_members(lanes: &[CoupledLane], fine_count: usize) -> AutomataResult<(Vec<u32>, Vec<u32>)> {
    let total_students = lanes.iter().map(|lane| lane.members.len()).sum::<usize>();
    let mut offsets = Vec::with_capacity(total_students + 1);
    let mut leaves = Vec::with_capacity(lanes.len() * fine_count);
    offsets.push(0);
    for (lane, data) in lanes.iter().enumerate() {
        for member in &data.members {
            for leaf in data.hierarchy.member_leaf_indices(*member) {
                let global = lane
                    .checked_mul(fine_count)
                    .and_then(|base| base.checked_add(*leaf))
                    .ok_or_else(|| {
                        AutomataError::InvalidArgument(
                            "coupled-fine member index overflow".to_string(),
                        )
                    })?;
                leaves.push(u32::try_from(global).map_err(|_| {
                    AutomataError::InvalidArgument(
                        "coupled-fine member index exceeds u32".to_string(),
                    )
                })?);
            }
            offsets.push(u32::try_from(leaves.len()).map_err(|_| {
                AutomataError::InvalidArgument("coupled-fine mapping exceeds u32".to_string())
            })?);
        }
    }
    Ok((offsets, leaves))
}

#[allow(clippy::too_many_arguments)]
fn append_snapshots(
    teacher: &NpaModel,
    grid: &HashGridConfig,
    model: &AdaptiveNpaModel,
    dt: f32,
    cut_index: usize,
    step: usize,
    captured: crate::gpu::WgpuCoupledFineSnapshot,
    student_count: usize,
    lanes: &[CoupledLane],
    snapshots: &mut Vec<KeyedSnapshot>,
) -> AutomataResult<()> {
    let output_dims = teacher.config.update_dims();
    let fine_count = captured.fine_base_update.len() / (lanes.len() * output_dims);
    for (lane_index, lane) in lanes.iter().enumerate() {
        let fine_start = lane_index * fine_count;
        let fine_end = fine_start + fine_count;
        let update_start = fine_start * output_dims;
        let update_end = fine_end * output_dims;
        let (teacher_dx, teacher_ds) = physical_update_from_raw(
            teacher,
            grid.eps,
            &captured.fine_base_update[update_start..update_end],
        );
        let restricted_dx = lane.hierarchy.restrict_values(
            &lane.fine,
            &lane.members,
            &teacher_dx,
            teacher.config.spatial_dims,
        )?;
        let restricted_ds = lane.hierarchy.restrict_values(
            &lane.fine,
            &lane.members,
            &teacher_ds,
            teacher.config.state_dims,
        )?;
        let student_start = lane_index * student_count;
        let student_end = student_start + student_count;
        let student_state_start = student_start * teacher.config.state_dims;
        let student_state_end = student_end * teacher.config.state_dims;
        let mut fine = lane.fine.clone();
        fine.positions = captured.fine_positions[fine_start..fine_end].to_vec();
        fine.states = captured.fine_states
            [fine_start * teacher.config.state_dims..fine_end * teacher.config.state_dims]
            .to_vec();
        let mut student = lane.student.clone();
        student.positions = captured.student_positions[student_start..student_end].to_vec();
        student.states = captured.student_states[student_state_start..student_state_end].to_vec();
        if model.config.closure_recurrent_mode {
            let (closure_mode, closure_basis, closure_phase) = captured_closure_state(
                &captured.student_diagnostics,
                lane_index,
                student_count,
                teacher.config.state_dims,
            )?;
            student.closure_mode = closure_mode;
            student.closure_basis = closure_basis;
            student.closure_phase = closure_phase;
        }
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
                super::closure_mode_teacher_update(
                    model,
                    &fine,
                    &student,
                    &lane.hierarchy,
                    &lane.members,
                    &teacher_dx,
                    &teacher_ds,
                    dt,
                )
            })
            .transpose()?;
        snapshots.push(KeyedSnapshot {
            rollout_index: lane.rollout_index,
            cut_index,
            step,
            snapshot: super::super::multiscale_on_policy::OnPolicySnapshot {
                particles: student,
                teacher_update: Some(teacher_update),
                closure_mode_target_update: closure_target
                    .as_ref()
                    .map(|target| target.mode.clone()),
                closure_basis_target_update: closure_target
                    .as_ref()
                    .map(|target| target.basis.clone()),
                captured_perception: Some(captured_student_perception(
                    &captured.student_diagnostics,
                    lane_index,
                    student_count,
                    teacher.config.state_dims,
                    teacher.config.spatial_dims,
                    teacher.config.perception_dims(),
                )),
                rollout_index: lane.rollout_index,
                step,
            },
        });
    }
    Ok(())
}

fn captured_closure_state(
    diagnostics: &crate::gpu::WgpuAdaptiveDiagnostics,
    lane: usize,
    particle_count: usize,
    state_dims: usize,
) -> AutomataResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let context_dims = state_dims + 6;
    if diagnostics.feature_dims < 2 * context_dims {
        return Err(AutomataError::InvalidModel(
            "resident closure diagnostics do not contain local and transported recurrent closure fields"
                .to_owned(),
        ));
    }
    let mode_offset = diagnostics.feature_dims - context_dims - state_dims;
    let phase_offset = mode_offset - 2;
    let basis_offset = phase_offset - 4;
    let row_start = lane * particle_count;
    let mut mode = Vec::with_capacity(particle_count * state_dims);
    let mut basis = Vec::with_capacity(particle_count * 4);
    let mut phase = Vec::with_capacity(particle_count * 2);
    for row in row_start..row_start + particle_count {
        let row_start = row * diagnostics.feature_dims;
        mode.extend_from_slice(
            &diagnostics.normalized_features
                [row_start + mode_offset..row_start + mode_offset + state_dims],
        );
        phase.extend_from_slice(
            &diagnostics.normalized_features
                [row_start + phase_offset..row_start + phase_offset + 2],
        );
        basis.extend_from_slice(
            &diagnostics.normalized_features
                [row_start + basis_offset..row_start + basis_offset + 4],
        );
    }
    Ok((mode, basis, phase))
}

fn captured_student_perception(
    diagnostics: &crate::gpu::WgpuAdaptiveDiagnostics,
    lane: usize,
    particle_count: usize,
    state_dims: usize,
    spatial_dims: usize,
    perception_dims: usize,
) -> OnPolicyCapturedPerception {
    let row_start = lane * particle_count;
    let row_end = row_start + particle_count;
    let update_start = row_start * diagnostics.output_dims;
    let update_end = row_end * diagnostics.output_dims;
    let observed_spacing = diagnostics.observed_spacing[row_start..row_end].to_vec();
    let accepted_degree = diagnostics.accepted_degree[row_start..row_end].to_vec();
    let coarse_exposure = diagnostics.coarse_exposure[row_start..row_end].to_vec();
    OnPolicyCapturedPerception {
        perception: AdaptivePerceptionPair {
            normalized: captured_perception_output(
                row_prefixes(
                    &diagnostics.normalized_features,
                    row_start,
                    particle_count,
                    diagnostics.feature_dims,
                    perception_dims,
                ),
                coarse_exposure.clone(),
                observed_spacing.clone(),
                accepted_degree.clone(),
                state_dims,
                spatial_dims,
                perception_dims,
            ),
            npa_compatible: captured_perception_output(
                row_prefixes(
                    &diagnostics.base_features,
                    row_start,
                    particle_count,
                    diagnostics.feature_dims,
                    perception_dims,
                ),
                coarse_exposure,
                observed_spacing,
                accepted_degree,
                state_dims,
                spatial_dims,
                perception_dims,
            ),
        },
        base_update: diagnostics.base_update[update_start..update_end].to_vec(),
        model_update: diagnostics.model_update[update_start..update_end].to_vec(),
    }
}

fn row_prefixes(
    values: &[f32],
    row_start: usize,
    rows: usize,
    source_dims: usize,
    prefix_dims: usize,
) -> Vec<f32> {
    debug_assert!(prefix_dims <= source_dims);
    let mut output = Vec::with_capacity(rows * prefix_dims);
    for row in row_start..row_start + rows {
        output.extend_from_slice(&values[row * source_dims..row * source_dims + prefix_dims]);
    }
    output
}

fn physical_update_from_raw(
    teacher: &NpaModel,
    grid_eps: f32,
    raw_update: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let rows = raw_update.len() / teacher.config.update_dims();
    let mut dx = vec![0.0; rows * teacher.config.spatial_dims];
    let mut ds = vec![0.0; rows * teacher.config.state_dims];
    for row in 0..rows {
        let update = &raw_update
            [row * teacher.config.update_dims()..(row + 1) * teacher.config.update_dims()];
        let norm = update[..teacher.config.spatial_dims]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let scale = teacher.config.alpha * teacher.config.motion_eps(grid_eps) / (1.0 + norm);
        for axis in 0..teacher.config.spatial_dims {
            dx[row * teacher.config.spatial_dims + axis] = update[axis] * scale;
        }
        ds[row * teacher.config.state_dims..(row + 1) * teacher.config.state_dims].copy_from_slice(
            &update[teacher.config.spatial_dims
                ..teacher.config.spatial_dims + teacher.config.state_dims],
        );
    }
    (dx, ds)
}
