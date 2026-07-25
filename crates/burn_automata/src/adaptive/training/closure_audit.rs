use std::time::Instant;

use serde::{Deserialize, Serialize};

use super::{
    super::{
        AdaptiveMaterialView, AdaptiveNpaConfig, AdaptiveParticleSet, AdaptiveProxyHierarchy,
        closure::restrict_first_closure_mode,
        features::{local_residual_features, material_detail_values},
        perception::rule_perception_pair,
    },
    multiscale_dataset::raw_update_from_restricted_step,
};
use crate::{
    AutomataError, AutomataResult, NpaModel, ParticleSeed,
    rollout::{seed_particles_scaled, stable_material_uniform},
};
use burn_automata_kernels::HashGridConfig;

#[cfg(feature = "gpu_wgpu")]
use crate::gpu::{WgpuAutomataExecutor, WgpuNeighborMode};

const MAX_CPU_REFERENCE_PARTICLES: usize = 512;
#[cfg(feature = "gpu_wgpu")]
const MAX_WGPU_RESIDENT_PARTICLES: usize = 4_096;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdaptiveClosureAuditBackend {
    #[default]
    CpuReference,
    WgpuResident,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AdaptiveClosureIdentifiabilityConfig {
    pub enabled: bool,
    pub fine_particle_count: usize,
    pub cut_leaf_count: usize,
    pub rollout_steps: usize,
    pub temporal_samples: usize,
    pub rollouts: usize,
    pub seed: u64,
    pub seed_scale: f32,
    pub total_measure: f32,
    pub bandwidth: f32,
    pub update_prob: f32,
    /// Re-evaluate the complete coarse local feature vector for both paired
    /// states. Quality-scale WGPU audits may disable this O(N^2) CPU control;
    /// exact restricted-observable equality remains mandatory.
    pub verify_local_features: bool,
    /// Weighted RMS magnitude of unresolved state perturbations relative to
    /// each coarse group's existing per-channel variation.
    pub perturbation_scale: f32,
}

impl Default for AdaptiveClosureIdentifiabilityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            fine_particle_count: 256,
            cut_leaf_count: 192,
            rollout_steps: 32,
            temporal_samples: 5,
            rollouts: 2,
            seed: 42,
            seed_scale: 0.2,
            total_measure: std::f32::consts::PI * 0.2 * 0.2,
            bandwidth: 0.1,
            update_prob: 0.5,
            verify_local_features: true,
            perturbation_scale: 0.25,
        }
    }
}

impl AdaptiveClosureIdentifiabilityConfig {
    fn validate(&self, teacher: &NpaModel, maximum_particle_count: usize) -> AutomataResult<()> {
        if !self.enabled {
            return Ok(());
        }
        if self.fine_particle_count < teacher.config.spatial_dims + 2
            || self.fine_particle_count > maximum_particle_count
            || self.cut_leaf_count == 0
            || self.cut_leaf_count >= self.fine_particle_count
            || self.temporal_samples == 0
            || self.rollouts == 0
            || !self.seed_scale.is_finite()
            || self.seed_scale <= 0.0
            || !self.total_measure.is_finite()
            || self.total_measure <= 0.0
            || !self.bandwidth.is_finite()
            || self.bandwidth <= 0.0
            || !self.update_prob.is_finite()
            || self.update_prob <= 0.0
            || self.update_prob > 1.0
            || !self.perturbation_scale.is_finite()
            || self.perturbation_scale <= 0.0
        {
            return Err(AutomataError::InvalidArgument(format!(
                "adaptive closure-identifiability configuration is invalid; this backend supports at most {maximum_particle_count} fine particles",
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveClosureIdentifiabilityReport {
    pub backend: String,
    pub rollouts: usize,
    pub snapshots: usize,
    pub fine_particle_count: usize,
    pub cut_leaf_count: usize,
    pub paired_coarse_rows: usize,
    pub skipped_coarse_rows: usize,
    /// Scalar spatial modes per state channel that remain after preserving the
    /// weighted mean and affine state-position moments.
    pub unresolved_state_modes: usize,
    pub maximum_unresolved_state_modes_per_coarse_row: usize,
    /// State-float ratio of the affine coarse representation plus all missing
    /// modes to the original fine state. Values below one establish that an
    /// exact first-level recurrent state can still compress storage.
    pub augmented_to_fine_state_value_ratio: f32,
    pub target_dims: usize,
    pub perturbation_root_mean_square: f32,
    pub maximum_restricted_observable_difference: f32,
    pub local_features_verified: bool,
    pub maximum_local_feature_difference: f32,
    pub paired_closure_mode_difference_root_mean_square: f32,
    pub affine_state_reconstruction_root_mean_square_error: f32,
    pub augmented_state_reconstruction_root_mean_square_error: f32,
    pub maximum_augmented_state_reconstruction_error: f32,
    pub target_root_mean_square: f32,
    pub paired_target_difference_root_mean_square: f32,
    /// For an equally weighted pair with identical observed coarse state, the
    /// optimal deterministic predictor is their mean. Its unavoidable RMSE is
    /// half the pairwise target difference.
    pub memoryless_normalized_rmse_lower_bound: f32,
    pub median_row_normalized_rmse_lower_bound: f32,
    pub p95_row_normalized_rmse_lower_bound: f32,
    pub maximum_row_normalized_rmse_lower_bound: f32,
    pub elapsed_ms: f64,
}

#[derive(Default)]
struct AuditAccumulator {
    snapshots: usize,
    paired_rows: usize,
    skipped_rows: usize,
    unresolved_modes: usize,
    maximum_unresolved_modes: usize,
    perturbation_square_sum: f64,
    perturbation_values: usize,
    observable_max: f32,
    feature_max: f32,
    closure_mode_difference_square_sum: f64,
    closure_mode_difference_values: usize,
    affine_reconstruction_square_sum: f64,
    augmented_reconstruction_square_sum: f64,
    reconstruction_values: usize,
    maximum_augmented_reconstruction_error: f32,
    target_square_sum: f64,
    target_difference_square_sum: f64,
    target_values: usize,
    row_lower_bounds: Vec<f32>,
}

pub fn audit_adaptive_closure_identifiability(
    teacher: &NpaModel,
    grid: &HashGridConfig,
    adaptive: &AdaptiveNpaConfig,
    config: &AdaptiveClosureIdentifiabilityConfig,
) -> AutomataResult<AdaptiveClosureIdentifiabilityReport> {
    config.validate(teacher, MAX_CPU_REFERENCE_PARTICLES)?;
    if !config.enabled {
        return Err(AutomataError::InvalidArgument(
            "adaptive closure-identifiability audit is disabled".to_owned(),
        ));
    }
    adaptive.validate()?;
    grid.validate()?;
    if teacher.config.spatial_dims != adaptive.spatial_dims {
        return Err(AutomataError::InvalidArgument(
            "closure audit teacher/adaptive dimensions do not match".to_owned(),
        ));
    }

    let started = Instant::now();
    let snapshot_steps = snapshot_steps(config.rollout_steps, config.temporal_samples);
    let mut accumulator = AuditAccumulator::default();
    for rollout in 0..config.rollouts {
        let seed = config
            .seed
            .wrapping_add((rollout as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
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

        for step in 0..=config.rollout_steps {
            if snapshot_steps.binary_search(&step).is_ok() {
                audit_snapshot(teacher, grid, adaptive, config, &fine, &mut accumulator)?;
            }
            if step == config.rollout_steps {
                break;
            }
            let mask = fine
                .particle_id
                .iter()
                .map(|id| {
                    f32::from(stable_material_uniform(seed, step + 1, *id) < config.update_prob)
                })
                .collect::<Vec<_>>();
            let next = teacher.step_cpu(
                &fine.positions,
                &fine.states,
                1,
                fine.len(),
                grid,
                1.0,
                Some(&mask),
            )?;
            fine.positions = next.next_positions;
            fine.states = next.next_states;
        }
    }

    finish_report("cpu-reference", teacher, config, accumulator, started)
}

/// Quality-scale closure audit using resident WGPU teacher trajectories and a
/// two-lane paired target step. Only selected snapshots cross the host
/// boundary for deterministic hierarchy restriction and report reduction.
pub fn audit_adaptive_closure_identifiability_wgpu(
    teacher: &NpaModel,
    grid: &HashGridConfig,
    adaptive: &AdaptiveNpaConfig,
    config: &AdaptiveClosureIdentifiabilityConfig,
) -> AutomataResult<AdaptiveClosureIdentifiabilityReport> {
    #[cfg(feature = "gpu_wgpu")]
    {
        audit_adaptive_closure_identifiability_wgpu_impl(teacher, grid, adaptive, config)
    }
    #[cfg(not(feature = "gpu_wgpu"))]
    {
        let _ = (teacher, grid, adaptive, config);
        Err(AutomataError::InvalidArgument(
            "resident WGPU closure audit requires the gpu_wgpu feature".to_owned(),
        ))
    }
}

#[cfg(feature = "gpu_wgpu")]
fn audit_adaptive_closure_identifiability_wgpu_impl(
    teacher: &NpaModel,
    grid: &HashGridConfig,
    adaptive: &AdaptiveNpaConfig,
    config: &AdaptiveClosureIdentifiabilityConfig,
) -> AutomataResult<AdaptiveClosureIdentifiabilityReport> {
    config.validate(teacher, MAX_WGPU_RESIDENT_PARTICLES)?;
    if !config.enabled {
        return Err(AutomataError::InvalidArgument(
            "adaptive closure-identifiability audit is disabled".to_owned(),
        ));
    }
    adaptive.validate()?;
    grid.validate()?;
    if teacher.config.spatial_dims != adaptive.spatial_dims {
        return Err(AutomataError::InvalidArgument(
            "closure audit teacher/adaptive dimensions do not match".to_owned(),
        ));
    }

    let started = Instant::now();
    let executor = WgpuAutomataExecutor::new_blocking()?;
    let snapshot_steps = snapshot_steps(config.rollout_steps, config.temporal_samples);
    let mut accumulator = AuditAccumulator::default();
    for rollout in 0..config.rollouts {
        let seed = config
            .seed
            .wrapping_add((rollout as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let (positions, states) = seed_particles_scaled(
            1,
            config.fine_particle_count,
            teacher.config.state_dims,
            teacher.config.spatial_dims,
            seed,
            ParticleSeed::UniformCircle,
            config.seed_scale,
        );
        let mut resident = executor.create_state_with_neighbor_mode_and_update_prob(
            teacher,
            &positions,
            &states,
            1,
            config.fine_particle_count,
            grid,
            1.0,
            WgpuNeighborMode::SubgroupCooperativeSortedCells,
            config.update_prob,
            seed,
        )?;
        let mut resident_step = 0;
        for &snapshot_step in &snapshot_steps {
            if snapshot_step > resident_step {
                executor.step_state_many(&mut resident, snapshot_step - resident_step)?;
                resident_step = snapshot_step;
            }
            let (positions, states) = executor.read_positions_states(&resident)?;
            let fine = AdaptiveParticleSet::from_equal_measure(
                positions,
                states,
                teacher.config.spatial_dims,
                teacher.config.state_dims,
                config.total_measure,
                config.bandwidth,
            )?;
            let (reference_positions, reference_states) =
                wgpu_step_single(&executor, teacher, grid, &fine)?;
            let raw_update =
                raw_update_from_next(teacher, &fine, &reference_positions, &reference_states)?;
            audit_snapshot_with_targets(
                teacher,
                adaptive,
                config,
                &fine,
                &raw_update,
                &mut accumulator,
                |plus, minus, hierarchy, view| {
                    let (next_positions, next_states) =
                        wgpu_step_pair(&executor, teacher, grid, plus, minus)?;
                    let count = plus.len();
                    let state_values = count * plus.state_dims;
                    Ok((
                        restricted_teacher_target_from_next(
                            teacher,
                            plus,
                            hierarchy,
                            view,
                            &next_positions[..count],
                            &next_states[..state_values],
                        )?,
                        restricted_teacher_target_from_next(
                            teacher,
                            minus,
                            hierarchy,
                            view,
                            &next_positions[count..],
                            &next_states[state_values..],
                        )?,
                    ))
                },
            )?;
        }
    }
    finish_report("wgpu-resident", teacher, config, accumulator, started)
}

#[cfg(feature = "gpu_wgpu")]
fn wgpu_step_single(
    executor: &WgpuAutomataExecutor,
    teacher: &NpaModel,
    grid: &HashGridConfig,
    particles: &AdaptiveParticleSet,
) -> AutomataResult<(Vec<[f32; 4]>, Vec<f32>)> {
    let mut state = executor.create_state_with_neighbor_mode_and_update_prob(
        teacher,
        &particles.positions,
        &particles.states,
        1,
        particles.len(),
        grid,
        1.0,
        WgpuNeighborMode::SubgroupCooperativeSortedCells,
        1.0,
        0,
    )?;
    executor.step_state(&mut state)?;
    executor.read_positions_states(&state)
}

#[cfg(feature = "gpu_wgpu")]
fn wgpu_step_pair(
    executor: &WgpuAutomataExecutor,
    teacher: &NpaModel,
    grid: &HashGridConfig,
    plus: &AdaptiveParticleSet,
    minus: &AdaptiveParticleSet,
) -> AutomataResult<(Vec<[f32; 4]>, Vec<f32>)> {
    debug_assert_eq!(plus.len(), minus.len());
    let mut positions = Vec::with_capacity(2 * plus.len());
    positions.extend_from_slice(&plus.positions);
    positions.extend_from_slice(&minus.positions);
    let mut states = Vec::with_capacity(plus.states.len() + minus.states.len());
    states.extend_from_slice(&plus.states);
    states.extend_from_slice(&minus.states);
    let mut state = executor.create_state_with_neighbor_mode_and_update_prob(
        teacher,
        &positions,
        &states,
        2,
        plus.len(),
        grid,
        1.0,
        WgpuNeighborMode::SubgroupCooperativeSortedCells,
        1.0,
        0,
    )?;
    executor.step_state(&mut state)?;
    executor.read_positions_states(&state)
}

fn finish_report(
    backend: &str,
    teacher: &NpaModel,
    config: &AdaptiveClosureIdentifiabilityConfig,
    mut accumulator: AuditAccumulator,
    started: Instant,
) -> AutomataResult<AdaptiveClosureIdentifiabilityReport> {
    if accumulator.paired_rows == 0 || accumulator.target_values == 0 {
        return Err(AutomataError::InvalidModel(
            "closure audit found no coarse groups with unresolved affine-nullspace state"
                .to_owned(),
        ));
    }
    accumulator.row_lower_bounds.sort_by(f32::total_cmp);
    let target_rms =
        (accumulator.target_square_sum / accumulator.target_values as f64).sqrt() as f32;
    let target_difference_rms =
        (accumulator.target_difference_square_sum / accumulator.target_values as f64).sqrt() as f32;
    Ok(AdaptiveClosureIdentifiabilityReport {
        backend: backend.to_owned(),
        rollouts: config.rollouts,
        snapshots: accumulator.snapshots,
        fine_particle_count: config.fine_particle_count,
        cut_leaf_count: config.cut_leaf_count,
        paired_coarse_rows: accumulator.paired_rows,
        skipped_coarse_rows: accumulator.skipped_rows,
        unresolved_state_modes: accumulator.unresolved_modes,
        maximum_unresolved_state_modes_per_coarse_row: accumulator.maximum_unresolved_modes,
        augmented_to_fine_state_value_ratio: (config.cut_leaf_count
            * teacher.config.state_dims
            * accumulator.snapshots
            + accumulator.unresolved_modes * teacher.config.state_dims)
            as f32
            / (config.fine_particle_count * teacher.config.state_dims * accumulator.snapshots)
                as f32,
        target_dims: teacher.config.update_dims(),
        perturbation_root_mean_square: (accumulator.perturbation_square_sum
            / accumulator.perturbation_values.max(1) as f64)
            .sqrt() as f32,
        maximum_restricted_observable_difference: accumulator.observable_max,
        local_features_verified: config.verify_local_features,
        maximum_local_feature_difference: accumulator.feature_max,
        paired_closure_mode_difference_root_mean_square: (accumulator
            .closure_mode_difference_square_sum
            / accumulator.closure_mode_difference_values.max(1) as f64)
            .sqrt() as f32,
        affine_state_reconstruction_root_mean_square_error: (accumulator
            .affine_reconstruction_square_sum
            / accumulator.reconstruction_values.max(1) as f64)
            .sqrt() as f32,
        augmented_state_reconstruction_root_mean_square_error: (accumulator
            .augmented_reconstruction_square_sum
            / accumulator.reconstruction_values.max(1) as f64)
            .sqrt() as f32,
        maximum_augmented_state_reconstruction_error: accumulator
            .maximum_augmented_reconstruction_error,
        target_root_mean_square: target_rms,
        paired_target_difference_root_mean_square: target_difference_rms,
        memoryless_normalized_rmse_lower_bound: 0.5 * target_difference_rms
            / target_rms.max(f32::MIN_POSITIVE),
        median_row_normalized_rmse_lower_bound: percentile(&accumulator.row_lower_bounds, 0.50),
        p95_row_normalized_rmse_lower_bound: percentile(&accumulator.row_lower_bounds, 0.95),
        maximum_row_normalized_rmse_lower_bound: accumulator
            .row_lower_bounds
            .last()
            .copied()
            .unwrap_or_default(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
    })
}

fn audit_snapshot(
    teacher: &NpaModel,
    grid: &HashGridConfig,
    adaptive: &AdaptiveNpaConfig,
    config: &AdaptiveClosureIdentifiabilityConfig,
    fine: &AdaptiveParticleSet,
    accumulator: &mut AuditAccumulator,
) -> AutomataResult<()> {
    let reference_step = teacher.step_cpu(
        &fine.positions,
        &fine.states,
        1,
        fine.len(),
        grid,
        1.0,
        None,
    )?;
    let raw_update = teacher.forward_update_from_features(&reference_step.perception.features)?;
    audit_snapshot_with_targets(
        teacher,
        adaptive,
        config,
        fine,
        &raw_update,
        accumulator,
        |plus, minus, hierarchy, view| {
            Ok((
                restricted_teacher_target(teacher, grid, plus, hierarchy, view)?,
                restricted_teacher_target(teacher, grid, minus, hierarchy, view)?,
            ))
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn audit_snapshot_with_targets(
    teacher: &NpaModel,
    adaptive: &AdaptiveNpaConfig,
    config: &AdaptiveClosureIdentifiabilityConfig,
    fine: &AdaptiveParticleSet,
    raw_update: &[f32],
    accumulator: &mut AuditAccumulator,
    target_pair: impl FnOnce(
        &AdaptiveParticleSet,
        &AdaptiveParticleSet,
        &AdaptiveProxyHierarchy,
        &AdaptiveMaterialView,
    ) -> AutomataResult<(Vec<f32>, Vec<f32>)>,
) -> AutomataResult<()> {
    let hierarchy = AdaptiveProxyHierarchy::build(fine, 2 * fine.spatial_dims)?;
    let detail = material_detail_values(
        fine,
        raw_update,
        teacher.config.update_dims(),
        adaptive.base_rule_footprint().recip(),
    );
    let view = hierarchy.material_cut(
        fine,
        config.cut_leaf_count,
        &detail,
        teacher.config.update_dims() + fine.state_dims + fine.spatial_dims,
    )?;
    let mut plus = fine.clone();
    let mut minus = fine.clone();
    let mut perturbed_material = vec![false; view.members.len()];
    for (material, member) in view.members.iter().copied().enumerate() {
        let leaves = hierarchy.member_leaf_indices(member);
        let Some((direction, unresolved_modes)) = affine_null_direction(fine, leaves) else {
            accumulator.skipped_rows += 1;
            continue;
        };
        accumulator.unresolved_modes += unresolved_modes;
        accumulator.maximum_unresolved_modes =
            accumulator.maximum_unresolved_modes.max(unresolved_modes);
        apply_state_perturbation(
            fine,
            &mut plus,
            &mut minus,
            leaves,
            &direction,
            config.perturbation_scale,
            accumulator,
        );
        perturbed_material[material] = true;
    }

    let plus_view = hierarchy.material_cut(
        &plus,
        config.cut_leaf_count,
        &detail,
        teacher.config.update_dims() + fine.state_dims + fine.spatial_dims,
    )?;
    let minus_view = hierarchy.material_cut(
        &minus,
        config.cut_leaf_count,
        &detail,
        teacher.config.update_dims() + fine.state_dims + fine.spatial_dims,
    )?;
    if plus_view.members != view.members || minus_view.members != view.members {
        return Err(AutomataError::InvalidModel(
            "closure perturbation changed the fixed material cut".to_owned(),
        ));
    }
    accumulator.observable_max = accumulator.observable_max.max(max_observable_difference(
        &plus_view.particles,
        &minus_view.particles,
    ));
    if config.verify_local_features {
        let plus_features = closure_features(adaptive, teacher, &plus_view)?;
        let minus_features = closure_features(adaptive, teacher, &minus_view)?;
        accumulator.feature_max = accumulator
            .feature_max
            .max(max_abs_difference(&plus_features, &minus_features));
    }

    let (plus_modes, plus_reconstruction) =
        restrict_first_closure_mode(&plus, &hierarchy, &plus_view)?;
    let (minus_modes, minus_reconstruction) =
        restrict_first_closure_mode(&minus, &hierarchy, &minus_view)?;
    for metrics in [plus_reconstruction, minus_reconstruction] {
        let values = metrics.reconstructed_state_values;
        accumulator.affine_reconstruction_square_sum +=
            (metrics.affine_root_mean_square_error as f64).powi(2) * values as f64;
        accumulator.augmented_reconstruction_square_sum +=
            (metrics.augmented_root_mean_square_error as f64).powi(2) * values as f64;
        accumulator.reconstruction_values += values;
        accumulator.maximum_augmented_reconstruction_error = accumulator
            .maximum_augmented_reconstruction_error
            .max(metrics.maximum_augmented_absolute_error);
    }
    for (material, perturbed) in perturbed_material.iter().copied().enumerate() {
        if !perturbed {
            continue;
        }
        let range = material * fine.state_dims..(material + 1) * fine.state_dims;
        for (plus, minus) in plus_modes.values[range.clone()]
            .iter()
            .zip(&minus_modes.values[range])
        {
            accumulator.closure_mode_difference_square_sum +=
                (*plus as f64 - *minus as f64).powi(2);
            accumulator.closure_mode_difference_values += 1;
        }
    }

    let (plus_target, minus_target) = target_pair(&plus, &minus, &hierarchy, &view)?;
    let output_dims = teacher.config.update_dims();
    for (material, perturbed) in perturbed_material.into_iter().enumerate() {
        if !perturbed {
            continue;
        }
        let range = material * output_dims..(material + 1) * output_dims;
        let plus_row = &plus_target[range.clone()];
        let minus_row = &minus_target[range];
        let target_square = plus_row
            .iter()
            .chain(minus_row)
            .map(|value| (*value as f64).powi(2))
            .sum::<f64>()
            * 0.5;
        let difference_square = plus_row
            .iter()
            .zip(minus_row)
            .map(|(lhs, rhs)| (*lhs as f64 - *rhs as f64).powi(2))
            .sum::<f64>();
        accumulator.target_square_sum += target_square;
        accumulator.target_difference_square_sum += difference_square;
        accumulator.target_values += output_dims;
        accumulator.paired_rows += 1;
        let target_rms = (target_square / output_dims as f64).sqrt() as f32;
        let difference_rms = (difference_square / output_dims as f64).sqrt() as f32;
        accumulator
            .row_lower_bounds
            .push(0.5 * difference_rms / target_rms.max(1.0e-6));
    }
    accumulator.snapshots += 1;
    Ok(())
}

fn closure_features(
    adaptive: &AdaptiveNpaConfig,
    teacher: &NpaModel,
    view: &AdaptiveMaterialView,
) -> AutomataResult<Vec<f32>> {
    let perception = rule_perception_pair(adaptive, teacher, &view.particles)?;
    Ok(local_residual_features(adaptive, &view.particles, &perception.normalized)?.into_owned())
}

fn restricted_teacher_target(
    teacher: &NpaModel,
    grid: &HashGridConfig,
    fine: &AdaptiveParticleSet,
    hierarchy: &AdaptiveProxyHierarchy,
    view: &AdaptiveMaterialView,
) -> AutomataResult<Vec<f32>> {
    let step = teacher.step_cpu(
        &fine.positions,
        &fine.states,
        1,
        fine.len(),
        grid,
        1.0,
        None,
    )?;
    restricted_teacher_target_from_next(
        teacher,
        fine,
        hierarchy,
        view,
        &step.next_positions,
        &step.next_states,
    )
}

fn restricted_teacher_target_from_next(
    teacher: &NpaModel,
    fine: &AdaptiveParticleSet,
    hierarchy: &AdaptiveProxyHierarchy,
    view: &AdaptiveMaterialView,
    next_positions: &[[f32; 4]],
    next_states: &[f32],
) -> AutomataResult<Vec<f32>> {
    let (dx, ds) = physical_step_delta(fine, next_positions, next_states)?;
    let restricted_dx = hierarchy.restrict_values(fine, &view.members, &dx, fine.spatial_dims)?;
    let restricted_ds = hierarchy.restrict_values(fine, &view.members, &ds, fine.state_dims)?;
    Ok(raw_update_from_restricted_step(
        &restricted_dx,
        &restricted_ds,
        &view.particles.bandwidth,
        teacher,
    ))
}

#[cfg(feature = "gpu_wgpu")]
fn raw_update_from_next(
    teacher: &NpaModel,
    fine: &AdaptiveParticleSet,
    next_positions: &[[f32; 4]],
    next_states: &[f32],
) -> AutomataResult<Vec<f32>> {
    let (dx, ds) = physical_step_delta(fine, next_positions, next_states)?;
    Ok(raw_update_from_restricted_step(
        &dx,
        &ds,
        &fine.bandwidth,
        teacher,
    ))
}

fn physical_step_delta(
    fine: &AdaptiveParticleSet,
    next_positions: &[[f32; 4]],
    next_states: &[f32],
) -> AutomataResult<(Vec<f32>, Vec<f32>)> {
    if next_positions.len() != fine.len() || next_states.len() != fine.states.len() {
        return Err(AutomataError::InvalidArgument(
            "closure audit next-state shape mismatch".to_owned(),
        ));
    }
    let dx = fine
        .positions
        .iter()
        .zip(next_positions)
        .flat_map(|(before, after)| {
            (0..fine.spatial_dims).map(move |axis| after[axis] - before[axis])
        })
        .collect::<Vec<_>>();
    let ds = fine
        .states
        .iter()
        .zip(next_states)
        .map(|(before, after)| after - before)
        .collect::<Vec<_>>();
    Ok((dx, ds))
}

fn affine_null_direction(
    fine: &AdaptiveParticleSet,
    leaves: &[usize],
) -> Option<(Vec<f32>, usize)> {
    let constraint_count = fine.spatial_dims + 1;
    if leaves.len() <= constraint_count {
        return None;
    }
    let total = leaves
        .iter()
        .map(|leaf| fine.represented_measure[*leaf] as f64)
        .sum::<f64>()
        .max(f64::MIN_POSITIVE);
    let mut center = vec![0.0_f64; fine.spatial_dims];
    for leaf in leaves {
        let weight = fine.represented_measure[*leaf] as f64 / total;
        for (axis, value) in center.iter_mut().enumerate() {
            *value += weight * fine.positions[*leaf][axis] as f64;
        }
    }
    let mut basis = Vec::<Vec<f64>>::new();
    for constraint in 0..constraint_count {
        let mut vector = leaves
            .iter()
            .map(|leaf| {
                let weight = fine.represented_measure[*leaf] as f64 / total;
                if constraint == 0 {
                    weight
                } else {
                    weight * (fine.positions[*leaf][constraint - 1] as f64 - center[constraint - 1])
                }
            })
            .collect::<Vec<_>>();
        orthogonalize(&mut vector, &basis);
        let norm = l2_norm(&vector);
        if norm > 1.0e-12 {
            vector.iter_mut().for_each(|value| *value /= norm);
            basis.push(vector);
        }
    }
    for attempt in 1..=8 {
        let mut candidate = leaves
            .iter()
            .map(|leaf| {
                let key = (*leaf as f64 + 1.0) * (attempt as f64 * 1.618_033_988_75 + 0.5);
                key.sin() + 0.5 * (key * 0.754_877_666).cos()
            })
            .collect::<Vec<_>>();
        orthogonalize(&mut candidate, &basis);
        let weighted_rms = leaves
            .iter()
            .zip(&candidate)
            .map(|(leaf, value)| fine.represented_measure[*leaf] as f64 / total * value * value)
            .sum::<f64>()
            .sqrt();
        if weighted_rms > 1.0e-10 {
            candidate
                .iter_mut()
                .for_each(|value| *value /= weighted_rms);
            let unresolved_modes = leaves.len().saturating_sub(basis.len());
            return Some((
                candidate.into_iter().map(|value| value as f32).collect(),
                unresolved_modes,
            ));
        }
    }
    None
}

fn orthogonalize(vector: &mut [f64], basis: &[Vec<f64>]) {
    for direction in basis {
        let projection = vector
            .iter()
            .zip(direction)
            .map(|(lhs, rhs)| lhs * rhs)
            .sum::<f64>();
        for (value, direction) in vector.iter_mut().zip(direction) {
            *value -= projection * direction;
        }
    }
}

fn l2_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

#[allow(clippy::too_many_arguments)]
fn apply_state_perturbation(
    source: &AdaptiveParticleSet,
    plus: &mut AdaptiveParticleSet,
    minus: &mut AdaptiveParticleSet,
    leaves: &[usize],
    direction: &[f32],
    scale: f32,
    accumulator: &mut AuditAccumulator,
) {
    let total = leaves
        .iter()
        .map(|leaf| source.represented_measure[*leaf])
        .sum::<f32>()
        .max(f32::MIN_POSITIVE);
    for channel in 0..source.state_dims {
        let mean = leaves
            .iter()
            .map(|leaf| {
                source.represented_measure[*leaf] / total
                    * source.states[*leaf * source.state_dims + channel]
            })
            .sum::<f32>();
        let variation = leaves
            .iter()
            .map(|leaf| {
                let delta = source.states[*leaf * source.state_dims + channel] - mean;
                source.represented_measure[*leaf] / total * delta * delta
            })
            .sum::<f32>()
            .sqrt();
        let amplitude = scale * variation.max(0.05);
        for (leaf, direction) in leaves.iter().zip(direction) {
            let delta = amplitude * direction;
            plus.states[*leaf * source.state_dims + channel] += delta;
            minus.states[*leaf * source.state_dims + channel] -= delta;
            accumulator.perturbation_square_sum += (delta as f64).powi(2);
            accumulator.perturbation_values += 1;
        }
    }
}

fn max_observable_difference(lhs: &AdaptiveParticleSet, rhs: &AdaptiveParticleSet) -> f32 {
    debug_assert_eq!(lhs.len(), rhs.len());
    let mut maximum = max_abs_difference(&lhs.states, &rhs.states)
        .max(max_abs_difference(&lhs.state_jacobian, &rhs.state_jacobian))
        .max(max_abs_difference(
            &lhs.represented_measure,
            &rhs.represented_measure,
        ))
        .max(max_abs_difference(&lhs.bandwidth, &rhs.bandwidth));
    for (lhs, rhs) in lhs.positions.iter().zip(&rhs.positions) {
        maximum = maximum.max(max_abs_difference(lhs, rhs));
    }
    for (lhs, rhs) in lhs.covariance.iter().zip(&rhs.covariance) {
        maximum = maximum.max(max_abs_difference(lhs, rhs));
    }
    maximum
}

fn max_abs_difference(lhs: &[f32], rhs: &[f32]) -> f32 {
    debug_assert_eq!(lhs.len(), rhs.len());
    lhs.iter()
        .zip(rhs)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f32::max)
}

fn snapshot_steps(rollout_steps: usize, samples: usize) -> Vec<usize> {
    if samples == 1 {
        return vec![rollout_steps];
    }
    let mut steps = (0..samples)
        .map(|sample| sample * rollout_steps / (samples - 1))
        .collect::<Vec<_>>();
    steps.sort_unstable();
    steps.dedup();
    steps
}

fn percentile(sorted: &[f32], fraction: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f32 * fraction).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NpaConfig, adaptive::AdaptiveNpaConfig, upstream_growing_2d_hashgrid};

    #[test]
    fn affine_nullspace_perturbation_preserves_restricted_observables() {
        let teacher = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 7);
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.closure_moment_features = true;
        adaptive.proxy.branch_factor = 4;
        let config = AdaptiveClosureIdentifiabilityConfig {
            enabled: true,
            fine_particle_count: 16,
            cut_leaf_count: 4,
            rollout_steps: 0,
            temporal_samples: 1,
            rollouts: 1,
            perturbation_scale: 0.2,
            ..AdaptiveClosureIdentifiabilityConfig::default()
        };
        let report = audit_adaptive_closure_identifiability(
            &teacher,
            &upstream_growing_2d_hashgrid(),
            &adaptive,
            &config,
        )
        .unwrap();
        assert!(report.paired_coarse_rows > 0);
        assert_eq!(report.maximum_unresolved_state_modes_per_coarse_row, 1);
        assert!(report.augmented_to_fine_state_value_ratio <= 1.0);
        assert!(report.perturbation_root_mean_square > 0.0);
        assert!(report.maximum_restricted_observable_difference < 2.0e-5);
        assert!(report.maximum_local_feature_difference < 2.0e-4);
        assert!(report.paired_closure_mode_difference_root_mean_square > 0.0);
        assert!(report.augmented_state_reconstruction_root_mean_square_error < 2.0e-5);
        assert!(report.maximum_augmented_state_reconstruction_error < 1.0e-4);
        assert!(report.paired_target_difference_root_mean_square > 0.0);
        assert!(report.memoryless_normalized_rmse_lower_bound.is_finite());
    }

    #[test]
    fn closure_identifiability_audit_is_deterministic() {
        let teacher = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 11);
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.closure_moment_features = true;
        let config = AdaptiveClosureIdentifiabilityConfig {
            enabled: true,
            fine_particle_count: 16,
            cut_leaf_count: 7,
            rollout_steps: 2,
            temporal_samples: 2,
            rollouts: 1,
            perturbation_scale: 0.1,
            ..AdaptiveClosureIdentifiabilityConfig::default()
        };
        let first = audit_adaptive_closure_identifiability(
            &teacher,
            &upstream_growing_2d_hashgrid(),
            &adaptive,
            &config,
        )
        .unwrap();
        let second = audit_adaptive_closure_identifiability(
            &teacher,
            &upstream_growing_2d_hashgrid(),
            &adaptive,
            &config,
        )
        .unwrap();
        assert_eq!(first.paired_coarse_rows, second.paired_coarse_rows);
        assert_eq!(
            first.paired_target_difference_root_mean_square,
            second.paired_target_difference_root_mean_square
        );
        assert_eq!(
            first.memoryless_normalized_rmse_lower_bound,
            second.memoryless_normalized_rmse_lower_bound
        );
    }

    #[cfg(feature = "gpu_wgpu")]
    #[test]
    #[ignore = "real-device WGPU closure-audit parity test"]
    fn resident_wgpu_targets_match_cpu_reference() {
        let teacher = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 13);
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.closure_moment_features = true;
        let config = AdaptiveClosureIdentifiabilityConfig {
            enabled: true,
            fine_particle_count: 64,
            cut_leaf_count: 48,
            rollout_steps: 0,
            temporal_samples: 1,
            rollouts: 1,
            perturbation_scale: 0.2,
            ..AdaptiveClosureIdentifiabilityConfig::default()
        };
        let grid = upstream_growing_2d_hashgrid();
        let cpu =
            audit_adaptive_closure_identifiability(&teacher, &grid, &adaptive, &config).unwrap();
        let wgpu = audit_adaptive_closure_identifiability_wgpu(&teacher, &grid, &adaptive, &config)
            .unwrap();

        assert_eq!(cpu.paired_coarse_rows, wgpu.paired_coarse_rows);
        assert_eq!(cpu.unresolved_state_modes, wgpu.unresolved_state_modes);
        assert_eq!(
            cpu.maximum_unresolved_state_modes_per_coarse_row,
            wgpu.maximum_unresolved_state_modes_per_coarse_row
        );
        assert_relative_close(
            cpu.target_root_mean_square,
            wgpu.target_root_mean_square,
            2.0e-3,
        );
        assert_relative_close(
            cpu.paired_target_difference_root_mean_square,
            wgpu.paired_target_difference_root_mean_square,
            2.0e-2,
        );
        assert_relative_close(
            cpu.memoryless_normalized_rmse_lower_bound,
            wgpu.memoryless_normalized_rmse_lower_bound,
            2.0e-2,
        );
    }

    #[cfg(feature = "gpu_wgpu")]
    fn assert_relative_close(lhs: f32, rhs: f32, tolerance: f32) {
        let relative = (lhs - rhs).abs() / lhs.abs().max(rhs.abs()).max(1.0e-6);
        assert!(
            relative <= tolerance,
            "relative difference {relative} exceeds {tolerance}: {lhs} vs {rhs}"
        );
    }
}
