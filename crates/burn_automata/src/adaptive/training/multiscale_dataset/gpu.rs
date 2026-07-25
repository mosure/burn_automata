use std::{collections::BTreeMap, time::Instant};

use burn_automata_kernels::HashGridConfig;
use rayon::prelude::*;

use super::{
    FineTeacherPerception, FineTeacherSnapshot, MultiscaleDatasetBuilder, PreparedMultiscaleCut,
    captured_fine_perception_pair, captured_perception_output, snapshot_steps, validate,
};
use crate::{
    AutomataError, AutomataResult, NpaModel, ParticleSeed,
    adaptive::{
        AdaptiveNpaConfig, AdaptiveParticleSet, material_footprint_radius,
        perception::decode_physical_state_gradient,
    },
    gpu::{
        WgpuAdaptiveDiagnostics, WgpuAutomataExecutor, WgpuAutomataState, WgpuMaterialStateInit,
        WgpuNeighborMode, WgpuPendingAdaptiveDiagnostics,
    },
    rollout::seed_particles_scaled,
};

struct PendingCutPerception {
    state_key: (usize, usize),
    batch_size: usize,
    particle_count: usize,
    readback: WgpuPendingAdaptiveDiagnostics,
}

pub fn adaptive_multiscale_training_batch_wgpu_with_executor(
    executor: &WgpuAutomataExecutor,
    teacher: &NpaModel,
    teacher_grid: &HashGridConfig,
    adaptive: &AdaptiveNpaConfig,
    config: &super::super::AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<super::super::AdaptiveMultiscaleTrainingBatch> {
    validate(teacher, teacher_grid, adaptive, config)?;
    if teacher.config.spatial_dims != 2 {
        return Err(AutomataError::InvalidArgument(
            "resident multiscale teacher collection currently supports 2D NPA training".to_string(),
        ));
    }
    let started = Instant::now();
    let mut builder = MultiscaleDatasetBuilder::new(teacher, adaptive, config)?;
    let mut cut_states = BTreeMap::new();
    collect_teacher_snapshot_groups(
        executor,
        teacher,
        teacher_grid,
        adaptive,
        config,
        |snapshots| {
            builder.append_wgpu_group_cached(executor, teacher_grid, snapshots, &mut cut_states)
        },
    )?;
    builder.finish(started)
}

impl MultiscaleDatasetBuilder<'_> {
    #[cfg(test)]
    pub(super) fn append_wgpu_group(
        &mut self,
        executor: &WgpuAutomataExecutor,
        teacher_grid: &HashGridConfig,
        snapshots: Vec<FineTeacherSnapshot>,
    ) -> AutomataResult<()> {
        self.append_wgpu_group_cached(executor, teacher_grid, snapshots, &mut BTreeMap::new())
    }

    fn append_wgpu_group_cached(
        &mut self,
        executor: &WgpuAutomataExecutor,
        teacher_grid: &HashGridConfig,
        snapshots: Vec<FineTeacherSnapshot>,
        cut_states: &mut BTreeMap<(usize, usize), WgpuAutomataState>,
    ) -> AutomataResult<()> {
        let profile = std::env::var_os("BURN_AUTOMATA_PROFILE_ADAPTIVE_STAGING").is_some();
        let group_started = Instant::now();
        let snapshot_count = snapshots.len();
        let prepared = snapshots
            .into_par_iter()
            .map(|snapshot| self.prepare_snapshot(snapshot))
            .collect::<AutomataResult<Vec<_>>>()?;
        self.generated_snapshots += prepared.len();
        let prepared_elapsed = group_started.elapsed();
        let requested_counts = self.config.cut_leaf_counts.clone();
        let cuts = prepared
            .par_iter()
            .map(|snapshot| {
                requested_counts
                    .iter()
                    .map(|&requested| self.prepare_cut(snapshot, requested))
                    .collect::<AutomataResult<Vec<_>>>()
            })
            .collect::<AutomataResult<Vec<_>>>()?;
        let cuts_elapsed = group_started.elapsed() - prepared_elapsed;
        let mut outputs = cuts
            .iter()
            .map(|row| (0..row.len()).map(|_| None).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let mut device_enqueue_elapsed = std::time::Duration::ZERO;
        let mut device_finish_elapsed = std::time::Duration::ZERO;
        let mut pending_captures = Vec::new();
        for cut_index in 0..requested_counts.len() {
            let mut device_groups = BTreeMap::<usize, Vec<usize>>::new();
            for (snapshot_index, snapshot) in prepared.iter().enumerate() {
                let cut = &cuts[snapshot_index][cut_index];
                if cut.view.particles.len() == snapshot.fine.len()
                    && let Some(captured) = &snapshot.captured_perception
                {
                    outputs[snapshot_index][cut_index] = Some(captured_fine_perception_pair(
                        captured,
                        &cut.view.members,
                        &snapshot.raw_update,
                        self.teacher.config.state_dims,
                        self.teacher.config.spatial_dims,
                        self.input_dims,
                        self.output_dims,
                    )?);
                } else {
                    device_groups
                        .entry(cut.view.particles.len())
                        .or_default()
                        .push(snapshot_index);
                }
            }
            for snapshot_indices in device_groups.values() {
                let cut_group = snapshot_indices
                    .iter()
                    .map(|&snapshot_index| &cuts[snapshot_index][cut_index])
                    .collect::<Vec<_>>();
                let device_started = Instant::now();
                let pending = self.enqueue_cut_perceptions_cached(
                    executor,
                    teacher_grid,
                    &cut_group,
                    cut_states,
                )?;
                device_enqueue_elapsed += device_started.elapsed();
                pending_captures.push((cut_index, snapshot_indices.clone(), pending));
            }
        }
        for (cut_index, snapshot_indices, pending) in pending_captures {
            let device_started = Instant::now();
            let captured = self.finish_cut_perceptions_cached(executor, cut_states, pending)?;
            device_finish_elapsed += device_started.elapsed();
            for (&snapshot_index, output) in snapshot_indices.iter().zip(captured) {
                outputs[snapshot_index][cut_index] = Some(output);
            }
        }
        let mut pending_rows = Vec::with_capacity(cuts.len() * requested_counts.len());
        for (cut_row, output_row) in cuts.into_iter().zip(outputs) {
            for (cut, output) in cut_row.into_iter().zip(output_row) {
                let (perception, base_update) = output.ok_or_else(|| {
                    AutomataError::InvalidModel(
                        "resident multiscale cut perception was not captured".to_string(),
                    )
                })?;
                let selected = self.sample_cut_rows(cut.rollout_index, cut.view.particles.len());
                pending_rows.push((cut, perception, base_update, selected));
            }
        }
        let prepared_rows = pending_rows
            .into_par_iter()
            .map(|(cut, perception, base_update, selected)| {
                self.prepare_cut_training_rows(cut, perception, base_update, &selected)
            })
            .collect::<AutomataResult<Vec<_>>>()?;
        for rows in prepared_rows {
            self.append_prepared_cut(rows);
        }
        if profile {
            let elapsed = group_started.elapsed();
            let device_elapsed = device_enqueue_elapsed + device_finish_elapsed;
            let append_elapsed =
                elapsed.saturating_sub(prepared_elapsed + cuts_elapsed + device_elapsed);
            eprintln!(
                "adaptive staging group snapshots={snapshot_count} prepare_ms={:.3} cuts_ms={:.3} device_enqueue_ms={:.3} device_finish_ms={:.3} append_ms={:.3} total_ms={:.3}",
                prepared_elapsed.as_secs_f64() * 1_000.0,
                cuts_elapsed.as_secs_f64() * 1_000.0,
                device_enqueue_elapsed.as_secs_f64() * 1_000.0,
                device_finish_elapsed.as_secs_f64() * 1_000.0,
                append_elapsed.as_secs_f64() * 1_000.0,
                elapsed.as_secs_f64() * 1_000.0,
            );
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn capture_cut_perceptions(
        &self,
        executor: &WgpuAutomataExecutor,
        teacher_grid: &HashGridConfig,
        cuts: &[&PreparedMultiscaleCut],
    ) -> AutomataResult<Vec<(burn_automata_kernels::AdaptivePerceptionPair, Vec<f32>)>> {
        let mut cut_states = BTreeMap::new();
        let pending =
            self.enqueue_cut_perceptions_cached(executor, teacher_grid, cuts, &mut cut_states)?;
        self.finish_cut_perceptions_cached(executor, &mut cut_states, pending)
    }

    fn enqueue_cut_perceptions_cached(
        &self,
        executor: &WgpuAutomataExecutor,
        teacher_grid: &HashGridConfig,
        cuts: &[&PreparedMultiscaleCut],
        cut_states: &mut BTreeMap<(usize, usize), WgpuAutomataState>,
    ) -> AutomataResult<PendingCutPerception> {
        let particle_count = cuts[0].view.particles.len();
        let total = particle_count * cuts.len();
        let mut positions = Vec::with_capacity(total);
        let mut states = Vec::with_capacity(total * self.teacher.config.state_dims);
        let mut represented_measure = Vec::with_capacity(total);
        let mut bandwidth = Vec::with_capacity(total);
        let mut covariance = Vec::with_capacity(total);
        let mut state_jacobian = Vec::with_capacity(
            total * self.teacher.config.state_dims * self.teacher.config.spatial_dims,
        );
        let mut render_scale = Vec::with_capacity(total);
        let mut seeds = Vec::with_capacity(cuts.len());
        for cut in cuts {
            let particles = &cut.view.particles;
            positions.extend_from_slice(&particles.positions);
            states.extend_from_slice(&particles.states);
            represented_measure.extend_from_slice(&particles.represented_measure);
            bandwidth.extend_from_slice(&particles.bandwidth);
            covariance.extend_from_slice(&particles.covariance);
            state_jacobian.extend_from_slice(&particles.state_jacobian);
            render_scale.extend(
                particles
                    .represented_measure
                    .iter()
                    .map(|measure| material_footprint_radius(*measure, 2)),
            );
            seeds.push(rollout_seed(self.config.seed, cut.rollout_index));
        }
        let neighbor_mode = if executor.subgroup_cooperative_supported() {
            WgpuNeighborMode::SubgroupCooperativeSortedCells
        } else {
            WgpuNeighborMode::CooperativeSortedCells
        };
        let material = WgpuMaterialStateInit {
            represented_measure: &represented_measure,
            particle_ids: None,
            update_masks: None,
            bandwidth: &bandwidth,
            support_bins: None,
            covariance: &covariance,
            state_jacobian: &state_jacobian,
            closure_mode: None,
            closure_basis: None,
            closure_phase: None,
            render_from_scale: &render_scale,
            render_target_footprint: &render_scale,
            display_scale_per_footprint: 1.0,
            render_transition_steps: 0,
        };
        let key = (cuts.len(), particle_count);
        if let Some(resident) = cut_states.get_mut(&key) {
            executor.update_state_particles(resident, &positions, &states)?;
            executor.update_state_material(resident, material)?;
        } else {
            let resident = executor.create_batched_material_state(
                self.teacher,
                &positions,
                &states,
                cuts.len(),
                particle_count,
                teacher_grid,
                self.config.dt,
                neighbor_mode,
                1.0,
                &seeds,
                material,
            )?;
            cut_states.insert(key, resident);
        }
        let resident = cut_states.get_mut(&key).ok_or_else(|| {
            AutomataError::InvalidModel("resident cut state cache insertion failed".to_string())
        })?;
        let readback = executor.begin_capture_adaptive_diagnostics_only(
            resident,
            self.adaptive.base_rule_footprint(),
            self.adaptive.perception,
        )?;
        Ok(PendingCutPerception {
            state_key: key,
            batch_size: cuts.len(),
            particle_count,
            readback,
        })
    }

    fn finish_cut_perceptions_cached(
        &self,
        executor: &WgpuAutomataExecutor,
        cut_states: &mut BTreeMap<(usize, usize), WgpuAutomataState>,
        pending: PendingCutPerception,
    ) -> AutomataResult<Vec<(burn_automata_kernels::AdaptivePerceptionPair, Vec<f32>)>> {
        let resident = cut_states.get_mut(&pending.state_key).ok_or_else(|| {
            AutomataError::InvalidModel("resident cut state cache lookup failed".to_string())
        })?;
        let diagnostics =
            executor.finish_capture_adaptive_diagnostics_only(resident, pending.readback)?;
        split_cut_diagnostics(
            diagnostics,
            pending.batch_size,
            pending.particle_count,
            self.teacher.config.state_dims,
            self.teacher.config.spatial_dims,
        )
    }
}

fn split_cut_diagnostics(
    diagnostics: WgpuAdaptiveDiagnostics,
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    spatial_dims: usize,
) -> AutomataResult<Vec<(burn_automata_kernels::AdaptivePerceptionPair, Vec<f32>)>> {
    let rows = batch_size * particle_count;
    if diagnostics.base_features.len() != rows * diagnostics.feature_dims
        || diagnostics.normalized_features.len() != rows * diagnostics.feature_dims
        || diagnostics.base_update.len() != rows * diagnostics.output_dims
        || diagnostics.observed_spacing.len() != rows
        || diagnostics.accepted_degree.len() != rows
        || diagnostics.coarse_exposure.len() != rows
    {
        return Err(AutomataError::InvalidModel(
            "resident cut diagnostics have an incompatible shape".to_string(),
        ));
    }
    let mut outputs = Vec::with_capacity(batch_size);
    for lane in 0..batch_size {
        let row_start = lane * particle_count;
        let row_end = row_start + particle_count;
        let feature_start = row_start * diagnostics.feature_dims;
        let feature_end = row_end * diagnostics.feature_dims;
        let update_start = row_start * diagnostics.output_dims;
        let update_end = row_end * diagnostics.output_dims;
        let observed_spacing = diagnostics.observed_spacing[row_start..row_end].to_vec();
        let accepted_degree = diagnostics.accepted_degree[row_start..row_end].to_vec();
        let coarse_exposure = diagnostics.coarse_exposure[row_start..row_end].to_vec();
        let base = captured_perception_output(
            diagnostics.base_features[feature_start..feature_end].to_vec(),
            coarse_exposure.clone(),
            observed_spacing.clone(),
            accepted_degree.clone(),
            state_dims,
            spatial_dims,
            diagnostics.feature_dims,
        );
        let normalized = captured_perception_output(
            diagnostics.normalized_features[feature_start..feature_end].to_vec(),
            coarse_exposure,
            observed_spacing,
            accepted_degree,
            state_dims,
            spatial_dims,
            diagnostics.feature_dims,
        );
        outputs.push((
            burn_automata_kernels::AdaptivePerceptionPair {
                normalized,
                npa_compatible: base,
            },
            diagnostics.base_update[update_start..update_end].to_vec(),
        ));
    }
    Ok(outputs)
}

#[cfg(test)]
pub(super) fn collect_teacher_snapshots(
    executor: &WgpuAutomataExecutor,
    teacher: &NpaModel,
    teacher_grid: &HashGridConfig,
    adaptive: &AdaptiveNpaConfig,
    config: &super::super::AdaptiveMultiscaleTrainingConfig,
    mut append: impl FnMut(FineTeacherSnapshot) -> AutomataResult<()>,
) -> AutomataResult<()> {
    collect_teacher_snapshot_groups(
        executor,
        teacher,
        teacher_grid,
        adaptive,
        config,
        |snapshots| {
            for snapshot in snapshots {
                append(snapshot)?;
            }
            Ok(())
        },
    )
}

fn collect_teacher_snapshot_groups(
    executor: &WgpuAutomataExecutor,
    teacher: &NpaModel,
    teacher_grid: &HashGridConfig,
    adaptive: &AdaptiveNpaConfig,
    config: &super::super::AdaptiveMultiscaleTrainingConfig,
    mut append: impl FnMut(Vec<FineTeacherSnapshot>) -> AutomataResult<()>,
) -> AutomataResult<()> {
    let steps = snapshot_steps(config.rollout_steps, config.temporal_samples);
    let lanes = executor.max_independent_trajectory_lanes();
    for rollout_start in (0..config.rollouts).step_by(lanes) {
        let rollout_end = (rollout_start + lanes).min(config.rollouts);
        let seeds = (rollout_start..rollout_end)
            .map(|rollout_index| rollout_seed(config.seed, rollout_index))
            .collect::<Vec<_>>();
        collect_teacher_snapshot_chunk(
            executor,
            teacher,
            teacher_grid,
            adaptive,
            config,
            rollout_start,
            &seeds,
            &steps,
            &mut append,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn collect_teacher_snapshot_chunk(
    executor: &WgpuAutomataExecutor,
    teacher: &NpaModel,
    teacher_grid: &HashGridConfig,
    adaptive: &AdaptiveNpaConfig,
    config: &super::super::AdaptiveMultiscaleTrainingConfig,
    rollout_start: usize,
    seeds: &[u64],
    steps: &[usize],
    append: &mut impl FnMut(Vec<FineTeacherSnapshot>) -> AutomataResult<()>,
) -> AutomataResult<()> {
    let profile = std::env::var_os("BURN_AUTOMATA_PROFILE_ADAPTIVE_STAGING").is_some();
    let chunk_started = Instant::now();
    let particle_sets = seeds
        .iter()
        .copied()
        .map(|seed| {
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
        })
        .collect::<AutomataResult<Vec<_>>>()?;
    let total = seeds.len() * config.fine_particle_count;
    let mut positions = Vec::with_capacity(total);
    let mut states = Vec::with_capacity(total * teacher.config.state_dims);
    let mut represented_measure = Vec::with_capacity(total);
    let mut bandwidth = Vec::with_capacity(total);
    let mut covariance = Vec::with_capacity(total);
    let mut state_jacobian =
        Vec::with_capacity(total * teacher.config.state_dims * teacher.config.spatial_dims);
    let mut render_scale = Vec::with_capacity(total);
    for particles in &particle_sets {
        positions.extend_from_slice(&particles.positions);
        states.extend_from_slice(&particles.states);
        represented_measure.extend_from_slice(&particles.represented_measure);
        bandwidth.extend_from_slice(&particles.bandwidth);
        covariance.extend_from_slice(&particles.covariance);
        state_jacobian.extend_from_slice(&particles.state_jacobian);
        render_scale.extend(
            particles
                .represented_measure
                .iter()
                .map(|measure| material_footprint_radius(*measure, 2)),
        );
    }
    let neighbor_mode = if executor.subgroup_cooperative_supported() {
        WgpuNeighborMode::SubgroupCooperativeSortedCells
    } else {
        WgpuNeighborMode::CooperativeSortedCells
    };
    let mut resident = executor.create_batched_material_state(
        teacher,
        &positions,
        &states,
        seeds.len(),
        config.fine_particle_count,
        teacher_grid,
        config.dt,
        neighbor_mode,
        config.update_prob,
        seeds,
        WgpuMaterialStateInit {
            represented_measure: &represented_measure,
            particle_ids: None,
            update_masks: None,
            bandwidth: &bandwidth,
            support_bins: None,
            covariance: &covariance,
            state_jacobian: &state_jacobian,
            closure_mode: None,
            closure_basis: None,
            closure_phase: None,
            render_from_scale: &render_scale,
            render_target_footprint: &render_scale,
            display_scale_per_footprint: 1.0,
            render_transition_steps: 0,
        },
    )?;
    let perception = adaptive.perception;
    let mut completed = 0usize;
    let mut pending_snapshots = Vec::with_capacity(executor.max_independent_trajectory_lanes());
    let mut capture_elapsed = std::time::Duration::ZERO;
    let mut append_elapsed = std::time::Duration::ZERO;
    for (snapshot_index, &step) in steps.iter().enumerate() {
        if step > completed {
            executor.step_state_many(&mut resident, step - completed)?;
            completed = step;
        }
        let capture_started = Instant::now();
        let snapshot = executor.capture_teacher_snapshot(
            &mut resident,
            adaptive.base_rule_footprint(),
            perception,
        )?;
        capture_elapsed += capture_started.elapsed();
        validate_snapshot(teacher, seeds.len(), config.fine_particle_count, &snapshot)?;

        // Keep the device busy while the current compact snapshot is decoded
        // and assembled into multiscale rows on the host.
        if let Some(&next_step) = steps.get(snapshot_index + 1)
            && next_step > completed
        {
            executor.step_state_many(&mut resident, next_step - completed)?;
            completed = next_step;
        }
        for lane in 0..seeds.len() {
            let particle_start = lane * config.fine_particle_count;
            let particle_end = particle_start + config.fine_particle_count;
            let state_start = particle_start * teacher.config.state_dims;
            let state_end = particle_end * teacher.config.state_dims;
            let update_start = particle_start * teacher.config.update_dims();
            let update_end = particle_end * teacher.config.update_dims();
            let feature_start = particle_start * teacher.config.perception_dims();
            let feature_end = particle_end * teacher.config.perception_dims();
            let raw_update = snapshot.base_update[update_start..update_end].to_vec();
            let (teacher_dx, teacher_ds) =
                physical_update_from_raw(teacher, teacher_grid.eps, &raw_update);
            pending_snapshots.push(FineTeacherSnapshot {
                rollout_index: rollout_start + lane,
                #[cfg(test)]
                step_index: step,
                positions: snapshot.positions[particle_start..particle_end].to_vec(),
                states: snapshot.states[state_start..state_end].to_vec(),
                raw_update,
                teacher_dx,
                teacher_ds,
                state_jacobian: decode_state_jacobian(
                    teacher,
                    teacher_grid,
                    &snapshot.base_features[feature_start..feature_end],
                )?,
                captured_perception: Some(FineTeacherPerception {
                    base_features: snapshot.base_features[feature_start..feature_end].to_vec(),
                    normalized_features: snapshot.normalized_features[feature_start..feature_end]
                        .to_vec(),
                    observed_spacing: snapshot.observed_spacing[particle_start..particle_end]
                        .to_vec(),
                    accepted_degree: snapshot.accepted_degree[particle_start..particle_end]
                        .to_vec(),
                }),
            });
        }
        if pending_snapshots.len() >= executor.max_independent_trajectory_lanes() {
            let append_started = Instant::now();
            append(std::mem::take(&mut pending_snapshots))?;
            append_elapsed += append_started.elapsed();
            pending_snapshots.reserve(executor.max_independent_trajectory_lanes());
        }
    }
    if !pending_snapshots.is_empty() {
        let append_started = Instant::now();
        append(pending_snapshots)?;
        append_elapsed += append_started.elapsed();
    }
    if profile {
        let elapsed = chunk_started.elapsed();
        let rollout_elapsed = elapsed.saturating_sub(capture_elapsed + append_elapsed);
        eprintln!(
            "adaptive staging chunk lanes={} rollout_ms={:.3} capture_ms={:.3} append_ms={:.3} total_ms={:.3}",
            seeds.len(),
            rollout_elapsed.as_secs_f64() * 1_000.0,
            capture_elapsed.as_secs_f64() * 1_000.0,
            append_elapsed.as_secs_f64() * 1_000.0,
            elapsed.as_secs_f64() * 1_000.0,
        );
    }
    Ok(())
}

fn validate_snapshot(
    teacher: &NpaModel,
    batch_size: usize,
    particle_count: usize,
    snapshot: &crate::gpu::WgpuTeacherSnapshot,
) -> AutomataResult<()> {
    let rows = batch_size * particle_count;
    if snapshot.positions.len() != rows
        || snapshot.states.len() != rows * teacher.config.state_dims
        || snapshot.base_features.len() != rows * teacher.config.perception_dims()
        || snapshot.normalized_features.len() != rows * teacher.config.perception_dims()
        || snapshot.base_update.len() != rows * teacher.config.update_dims()
        || snapshot.observed_spacing.len() != rows
        || snapshot.accepted_degree.len() != rows
    {
        return Err(AutomataError::InvalidModel(
            "resident multiscale teacher snapshot has an incompatible shape".to_string(),
        ));
    }
    Ok(())
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
        let motion_scale =
            teacher.config.alpha * teacher.config.motion_eps(grid_eps) / (1.0 + norm);
        for axis in 0..teacher.config.spatial_dims {
            dx[row * teacher.config.spatial_dims + axis] = update[axis] * motion_scale;
        }
        ds[row * teacher.config.state_dims..(row + 1) * teacher.config.state_dims].copy_from_slice(
            &update[teacher.config.spatial_dims
                ..teacher.config.spatial_dims + teacher.config.state_dims],
        );
    }
    (dx, ds)
}

fn decode_state_jacobian(
    teacher: &NpaModel,
    teacher_grid: &HashGridConfig,
    features: &[f32],
) -> AutomataResult<Vec<f32>> {
    let feature_dims = teacher.config.perception_dims();
    if !features.len().is_multiple_of(feature_dims) {
        return Err(AutomataError::InvalidModel(
            "resident multiscale teacher feature shape is invalid".to_string(),
        ));
    }
    let row_dims = teacher.config.state_dims * teacher.config.spatial_dims;
    if !teacher.config.state_grad {
        return Ok(vec![0.0; features.len() / feature_dims * row_dims]);
    }
    let gradient_start = 2 * teacher.config.state_dims;
    let gradient_scale = if teacher.config.scale_equivariant() {
        teacher_grid.eps / teacher.config.eps0.max(f32::MIN_POSITIVE)
    } else {
        1.0
    };
    let mut decoded = Vec::with_capacity(features.len() / feature_dims * row_dims);
    for row in features.chunks_exact(feature_dims) {
        decoded.extend(decode_physical_state_gradient(
            &row[gradient_start..gradient_start + row_dims],
            teacher.config.state_dims,
            teacher.config.spatial_dims,
            gradient_scale,
            teacher.config.log_norm_grad,
        ));
    }
    Ok(decoded)
}

fn rollout_seed(seed: u64, rollout_index: usize) -> u64 {
    seed.wrapping_add((rollout_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
}
