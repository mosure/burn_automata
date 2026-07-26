use std::{collections::BTreeMap, time::Duration};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    AutomataError, AutomataResult, ParticleSeed, Target2dGpuBackend, Target2dGpuCheckpointConfig,
    Target2dGpuTrainingReport, Target2dLossConfig, Target2dTrainingConfig,
    rollout::seed_particles_scaled,
};

use super::super::{
    AdaptiveMaterialSeedLayout, AdaptiveNpaModel, AdaptiveParticleSet,
    seed::{
        adaptive_particle_subset, apply_continuous_material_layout, continuous_material_units,
        continuous_uniform_seed_from_reference, seed_adaptive_particles_scaled,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AdaptiveTarget2dMaterialConfig {
    /// Fine-particle budget represented by the active material rows.
    pub reference_particle_count: usize,
    /// Conserved material measure across every active rollout.
    pub total_measure: f32,
    /// Interaction bandwidth of one fine material unit.
    pub fine_bandwidth: f32,
    /// Exponent mapping represented-measure ratio to interaction bandwidth.
    pub bandwidth_exponent: f32,
    /// Largest number of fine units represented by one initial active row.
    pub max_initial_fine_units: usize,
    /// Initial measure distribution used consistently by training, validation,
    /// serialized artifacts, and inference.
    pub seed_layout: AdaptiveMaterialSeedLayout,
    /// Largest-to-smallest represented-measure ratio for a graded continuous
    /// layout. One is the uniform control.
    pub seed_measure_ratio: f32,
}

impl Default for AdaptiveTarget2dMaterialConfig {
    fn default() -> Self {
        Self {
            reference_particle_count: 4_096,
            total_measure: std::f32::consts::PI * 0.2 * 0.2,
            fine_bandwidth: 0.1,
            bandwidth_exponent: 0.5,
            max_initial_fine_units: 4,
            seed_layout: AdaptiveMaterialSeedLayout::CanonicalGrouped,
            seed_measure_ratio: 1.0,
        }
    }
}

impl AdaptiveTarget2dMaterialConfig {
    pub fn layout(
        self,
        active_particle_count: usize,
        min_bandwidth: f32,
        max_bandwidth: f32,
    ) -> AutomataResult<AdaptiveTarget2dMaterialLayout> {
        if active_particle_count == 0
            || self.reference_particle_count < active_particle_count
            || self.max_initial_fine_units == 0
            || self.reference_particle_count
                > active_particle_count.saturating_mul(self.max_initial_fine_units)
        {
            return Err(AutomataError::InvalidArgument(format!(
                "adaptive Target2D requires active <= reference <= active * max_initial_fine_units, got active={active_particle_count} reference={} max_units={}",
                self.reference_particle_count, self.max_initial_fine_units,
            )));
        }
        if !self.total_measure.is_finite()
            || self.total_measure <= 0.0
            || !self.fine_bandwidth.is_finite()
            || self.fine_bandwidth <= 0.0
            || !self.bandwidth_exponent.is_finite()
            || self.bandwidth_exponent < 0.0
            || !self.seed_measure_ratio.is_finite()
            || self.seed_measure_ratio < 1.0
            || !min_bandwidth.is_finite()
            || !max_bandwidth.is_finite()
            || min_bandwidth <= 0.0
            || max_bandwidth < min_bandwidth
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive Target2D material scales must be finite and positive".to_string(),
            ));
        }

        let (fine_units, represented_fine_units) = match self.seed_layout {
            AdaptiveMaterialSeedLayout::CanonicalGrouped => {
                let mut fine_units = vec![1usize; active_particle_count];
                let mut excess = self.reference_particle_count - active_particle_count;
                let capacity_per_coarse = self.max_initial_fine_units - 1;
                if excess > 0 {
                    let coarse_rows = excess.div_ceil(capacity_per_coarse);
                    for coarse in 0..coarse_rows {
                        let row = coarse * active_particle_count / coarse_rows;
                        let add = excess.min(capacity_per_coarse);
                        fine_units[row] += add;
                        excess -= add;
                    }
                }
                debug_assert_eq!(
                    fine_units.iter().sum::<usize>(),
                    self.reference_particle_count
                );
                let represented = fine_units.iter().map(|units| *units as f32).collect();
                (Some(fine_units), represented)
            }
            AdaptiveMaterialSeedLayout::UniformContinuous => {
                let units = self.reference_particle_count as f32 / active_particle_count as f32;
                (None, vec![units; active_particle_count])
            }
            AdaptiveMaterialSeedLayout::GradedContinuous => (
                None,
                continuous_material_units(
                    active_particle_count,
                    self.reference_particle_count,
                    self.seed_measure_ratio,
                )?,
            ),
        };

        let fine_measure = self.total_measure / self.reference_particle_count as f32;
        let represented_measure = represented_fine_units
            .iter()
            .map(|units| *units * fine_measure)
            .collect::<Vec<_>>();
        let bandwidth = represented_fine_units
            .iter()
            .map(|units| {
                (self.fine_bandwidth * units.powf(self.bandwidth_exponent))
                    .clamp(min_bandwidth, max_bandwidth)
            })
            .collect::<Vec<_>>();
        let footprint_ratio = represented_fine_units
            .iter()
            .map(|units| units.sqrt())
            .collect::<Vec<_>>();
        AdaptiveTarget2dMaterialLayout {
            fine_units,
            represented_fine_units,
            represented_measure,
            bandwidth,
            footprint_ratio,
            fine_measure,
            reference_particle_count: self.reference_particle_count,
            seed_layout: self.seed_layout,
        }
        .validated(self)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveTarget2dMaterialLayout {
    /// Canonical integral material groups used by four-to-one hard topology.
    /// Continuous layouts deliberately leave this absent.
    pub fine_units: Option<Vec<usize>>,
    pub represented_fine_units: Vec<f32>,
    pub represented_measure: Vec<f32>,
    pub bandwidth: Vec<f32>,
    pub footprint_ratio: Vec<f32>,
    pub fine_measure: f32,
    pub reference_particle_count: usize,
    pub seed_layout: AdaptiveMaterialSeedLayout,
}

impl AdaptiveTarget2dMaterialLayout {
    fn validated(self, config: AdaptiveTarget2dMaterialConfig) -> AutomataResult<Self> {
        let rows = self.represented_fine_units.len();
        let measure_sum = self.represented_measure.iter().sum::<f32>();
        if rows == 0
            || self.represented_measure.len() != rows
            || self.bandwidth.len() != rows
            || self.footprint_ratio.len() != rows
            || self.reference_particle_count != config.reference_particle_count
            || self.seed_layout != config.seed_layout
            || self.fine_units.as_ref().is_some_and(|units| {
                units.len() != rows
                    || units.iter().sum::<usize>() != config.reference_particle_count
            })
            || (self.represented_fine_units.iter().sum::<f32>()
                - config.reference_particle_count as f32)
                .abs()
                > 2.0e-5 * config.reference_particle_count as f32
            || (measure_sum - config.total_measure).abs() > 2.0e-6 * config.total_measure.max(1.0)
            || self
                .represented_fine_units
                .iter()
                .chain(&self.represented_measure)
                .chain(&self.bandwidth)
                .chain(&self.footprint_ratio)
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive Target2D material layout violates conservation or shape".to_string(),
            ));
        }
        Ok(self)
    }

    pub fn active_particle_count(&self) -> usize {
        self.represented_fine_units.len()
    }

    pub fn coarse_particle_count(&self) -> usize {
        self.represented_fine_units
            .iter()
            .filter(|units| **units > 1.0 + 32.0 * f32::EPSILON)
            .count()
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(crate) struct AdaptiveTarget2dSeedBank {
    pub(crate) positions: Vec<f32>,
    pub(crate) states: Vec<f32>,
    pub(crate) update_masks: Vec<AdaptiveTarget2dUpdateMask>,
    pub(crate) eval_positions: Vec<f32>,
    pub(crate) eval_states: Vec<f32>,
    pub(crate) eval_update_masks: Vec<AdaptiveTarget2dUpdateMask>,
    pub(crate) eval_seeds: Vec<u64>,
    pub(crate) pool_size: usize,
    pub(crate) particle_count: usize,
    pub(crate) state_dims: usize,
    pub(crate) max_measure_relative_error: f32,
    pub(crate) max_centroid_l2_error: f32,
    pub(crate) max_extensive_state_l2_error: f32,
}

pub(crate) const ADAPTIVE_TARGET2D_UPDATE_MASK_MEMBERS: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AdaptiveTarget2dUpdateMask {
    pub(crate) keys: [u32; ADAPTIVE_TARGET2D_UPDATE_MASK_MEMBERS],
    pub(crate) weights: [f32; ADAPTIVE_TARGET2D_UPDATE_MASK_MEMBERS],
    pub(crate) expected: bool,
}

struct AdaptiveTarget2dSeedRow {
    positions: Vec<f32>,
    states: Vec<f32>,
    update_masks: Vec<AdaptiveTarget2dUpdateMask>,
    measure_relative_error: f32,
    centroid_l2_error: f32,
    extensive_state_l2_error: f32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_adaptive_target2d_seed_bank(
    model: &AdaptiveNpaModel,
    layout: &AdaptiveTarget2dMaterialLayout,
    pool_size: usize,
    seed: u64,
    eval_seeds: &[u64],
    seed_mode: ParticleSeed,
    seed_scale: f32,
    total_measure: f32,
    fine_bandwidth: f32,
) -> AutomataResult<AdaptiveTarget2dSeedBank> {
    model.validate()?;
    let pool_size = pool_size.max(1);
    if model.config.target_leaves != layout.active_particle_count()
        || model.config.bootstrap_fine_leaf_count() != layout.reference_particle_count
        || model.config.retain_bootstrap_templates
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive Target2D seed bank requires matching active/reference budgets and retain_bootstrap_templates=false"
                .to_owned(),
        ));
    }
    if eval_seeds.is_empty() {
        return Err(AutomataError::InvalidArgument(
            "adaptive Target2D seed bank requires at least one evaluation seed".to_owned(),
        ));
    }

    let rows = (0..pool_size)
        .into_par_iter()
        .map(|row| {
            adaptive_target2d_seed_row(
                model,
                layout,
                seed.wrapping_add((row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)),
                seed_mode,
                seed_scale,
                total_measure,
                fine_bandwidth,
            )
        })
        .collect::<AutomataResult<Vec<_>>>()?;
    let eval_rows = eval_seeds
        .par_iter()
        .copied()
        .map(|eval_seed| {
            adaptive_target2d_seed_row(
                model,
                layout,
                eval_seed,
                seed_mode,
                seed_scale,
                total_measure,
                fine_bandwidth,
            )
        })
        .collect::<AutomataResult<Vec<_>>>()?;
    let particle_count = layout.active_particle_count();
    let state_dims = model.rule.config.state_dims;
    let mut positions = Vec::with_capacity(pool_size * particle_count * 2);
    let mut states = Vec::with_capacity(pool_size * particle_count * state_dims);
    let mut update_masks = Vec::with_capacity(pool_size * particle_count);
    let mut eval_positions = Vec::with_capacity(eval_seeds.len() * particle_count * 2);
    let mut eval_states = Vec::with_capacity(eval_seeds.len() * particle_count * state_dims);
    let mut eval_update_masks = Vec::with_capacity(eval_seeds.len() * particle_count);
    let mut max_measure_relative_error = 0.0_f32;
    let mut max_centroid_l2_error = 0.0_f32;
    let mut max_extensive_state_l2_error = 0.0_f32;
    for row in rows {
        positions.extend(row.positions);
        states.extend(row.states);
        update_masks.extend(row.update_masks);
        max_measure_relative_error = max_measure_relative_error.max(row.measure_relative_error);
        max_centroid_l2_error = max_centroid_l2_error.max(row.centroid_l2_error);
        max_extensive_state_l2_error =
            max_extensive_state_l2_error.max(row.extensive_state_l2_error);
    }
    for row in eval_rows {
        eval_positions.extend(row.positions);
        eval_states.extend(row.states);
        eval_update_masks.extend(row.update_masks);
        max_measure_relative_error = max_measure_relative_error.max(row.measure_relative_error);
        max_centroid_l2_error = max_centroid_l2_error.max(row.centroid_l2_error);
        max_extensive_state_l2_error =
            max_extensive_state_l2_error.max(row.extensive_state_l2_error);
    }
    Ok(AdaptiveTarget2dSeedBank {
        positions,
        states,
        update_masks,
        eval_positions,
        eval_states,
        eval_update_masks,
        eval_seeds: eval_seeds.to_vec(),
        pool_size,
        particle_count,
        state_dims,
        max_measure_relative_error,
        max_centroid_l2_error,
        max_extensive_state_l2_error,
    })
}

#[allow(clippy::too_many_arguments)]
fn adaptive_target2d_seed_row(
    model: &AdaptiveNpaModel,
    layout: &AdaptiveTarget2dMaterialLayout,
    seed: u64,
    seed_mode: ParticleSeed,
    seed_scale: f32,
    total_measure: f32,
    fine_bandwidth: f32,
) -> AutomataResult<AdaptiveTarget2dSeedRow> {
    let reference_count = layout.reference_particle_count;
    let restricted = adaptive_target2d_seed_particles(
        model,
        layout,
        seed,
        seed_mode,
        seed_scale,
        total_measure,
        fine_bandwidth,
    )?;
    let state_dims = restricted.state_dims;
    let positions = restricted
        .positions
        .iter()
        .flat_map(|position| [position[0], position[1]])
        .collect::<Vec<_>>();
    let states = restricted.states.clone();
    let update_masks = adaptive_target2d_update_masks(
        &restricted,
        model.config.expected_coarse_update_mask,
        layout.fine_measure,
    )?;

    let (fine_positions, fine_states) = seed_particles_scaled(
        1,
        reference_count,
        state_dims,
        2,
        seed,
        seed_mode,
        seed_scale,
    );
    let fine_measure = total_measure / reference_count as f32;
    let restricted_measure = restricted.total_measure() as f32;
    let measure_relative_error =
        (restricted_measure - total_measure).abs() / total_measure.max(f32::MIN_POSITIVE);
    let mut fine_centroid = [0.0_f32; 2];
    let mut restricted_centroid = [0.0_f32; 2];
    for position in &fine_positions {
        fine_centroid[0] += fine_measure * position[0];
        fine_centroid[1] += fine_measure * position[1];
    }
    for row in 0..restricted.len() {
        restricted_centroid[0] +=
            restricted.represented_measure[row] * restricted.positions[row][0];
        restricted_centroid[1] +=
            restricted.represented_measure[row] * restricted.positions[row][1];
    }
    let centroid_l2_error = ((fine_centroid[0] - restricted_centroid[0]).powi(2)
        + (fine_centroid[1] - restricted_centroid[1]).powi(2))
    .sqrt()
        / total_measure.max(f32::MIN_POSITIVE);
    let mut fine_state = vec![0.0_f32; state_dims];
    let mut restricted_state = vec![0.0_f32; state_dims];
    for row in 0..reference_count {
        for (channel, value) in fine_state.iter_mut().enumerate() {
            *value += fine_measure * fine_states[row * state_dims + channel];
        }
    }
    for row in 0..restricted.len() {
        for (channel, value) in restricted_state.iter_mut().enumerate() {
            *value +=
                restricted.represented_measure[row] * restricted.states[row * state_dims + channel];
        }
    }
    let compact_memory = model.compact_recurrent_memory_range();
    let extensive_state_l2_error = fine_state
        .iter()
        .zip(&restricted_state)
        .enumerate()
        .filter(|(channel, _)| {
            !compact_memory
                .as_ref()
                .is_some_and(|memory| memory.contains(channel))
        })
        .map(|(_, (fine, restricted))| (fine - restricted).powi(2))
        .sum::<f32>()
        .sqrt()
        / total_measure.max(f32::MIN_POSITIVE);
    if measure_relative_error > 2.0e-5
        || centroid_l2_error > 2.0e-5
        || extensive_state_l2_error > 2.0e-5
    {
        return Err(AutomataError::InvalidModel(format!(
            "adaptive Target2D seed restriction violates conservation: measure={measure_relative_error:.3e} centroid={centroid_l2_error:.3e} physical_state={extensive_state_l2_error:.3e}"
        )));
    }
    Ok(AdaptiveTarget2dSeedRow {
        positions,
        states,
        update_masks,
        measure_relative_error,
        centroid_l2_error,
        extensive_state_l2_error,
    })
}

fn adaptive_target2d_update_masks(
    particles: &AdaptiveParticleSet,
    expected_coarse_update_mask: bool,
    fine_measure: f32,
) -> AutomataResult<Vec<AdaptiveTarget2dUpdateMask>> {
    let templates = particles
        .bootstrap_templates
        .iter()
        .map(|template| (template.parent_id, template))
        .collect::<BTreeMap<_, _>>();
    particles
        .particle_id
        .iter()
        .copied()
        .enumerate()
        .map(|(row, particle_id)| {
            let expected = expected_coarse_update_mask
                && particles.represented_measure[row] > fine_measure * (1.0 + 32.0 * f32::EPSILON);
            let Some(template) = templates.get(&particle_id) else {
                let mut keys = [0; ADAPTIVE_TARGET2D_UPDATE_MASK_MEMBERS];
                let mut weights = [0.0; ADAPTIVE_TARGET2D_UPDATE_MASK_MEMBERS];
                keys[0] = material_update_key(particle_id);
                weights[0] = 1.0;
                return Ok(AdaptiveTarget2dUpdateMask {
                    keys,
                    weights,
                    expected,
                });
            };
            if template.children.len() > ADAPTIVE_TARGET2D_UPDATE_MASK_MEMBERS {
                return Err(AutomataError::InvalidArgument(format!(
                    "adaptive Target2D material parent {particle_id} has {} update-mask members; at most {ADAPTIVE_TARGET2D_UPDATE_MASK_MEMBERS} are supported",
                    template.children.len(),
                )));
            }
            let total = template
                .children
                .iter()
                .map(|child| child.represented_measure)
                .sum::<f32>()
                .max(f32::MIN_POSITIVE);
            let mut keys = [0; ADAPTIVE_TARGET2D_UPDATE_MASK_MEMBERS];
            let mut weights = [0.0; ADAPTIVE_TARGET2D_UPDATE_MASK_MEMBERS];
            for (member, child) in template.children.iter().enumerate() {
                keys[member] = material_update_key(child.particle_id);
                weights[member] = child.represented_measure / total;
            }
            Ok(AdaptiveTarget2dUpdateMask {
                keys,
                weights,
                expected,
            })
        })
        .collect()
}

const fn material_update_key(particle_id: u64) -> u32 {
    (particle_id as u32) ^ ((particle_id >> 32) as u32)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn adaptive_target2d_seed_particles(
    model: &AdaptiveNpaModel,
    layout: &AdaptiveTarget2dMaterialLayout,
    seed: u64,
    seed_mode: ParticleSeed,
    seed_scale: f32,
    total_measure: f32,
    fine_bandwidth: f32,
) -> AutomataResult<AdaptiveParticleSet> {
    let active_count = layout.active_particle_count();
    let restricted = match layout.seed_layout {
        AdaptiveMaterialSeedLayout::CanonicalGrouped => seed_adaptive_particles_scaled(
            model,
            active_count,
            seed,
            seed_mode,
            seed_scale,
            total_measure,
            fine_bandwidth,
        )?,
        AdaptiveMaterialSeedLayout::UniformContinuous => continuous_uniform_seed_from_reference(
            model,
            active_count,
            layout.reference_particle_count,
            seed,
            seed_mode,
            seed_scale,
            total_measure,
            layout.bandwidth[0],
        )?,
        AdaptiveMaterialSeedLayout::GradedContinuous => {
            let mut particles = continuous_uniform_seed_from_reference(
                model,
                active_count,
                layout.reference_particle_count,
                seed,
                seed_mode,
                seed_scale,
                total_measure,
                layout.bandwidth.iter().sum::<f32>() / active_count as f32,
            )?;
            apply_continuous_material_layout(
                &mut particles,
                &layout.represented_measure,
                &layout.bandwidth,
            )?;
            particles
        }
    };
    if restricted.len() != active_count || !restricted.bootstrap_templates.is_empty() {
        return Err(AutomataError::InvalidModel(
            "adaptive Target2D seed must contain only active material rows".to_owned(),
        ));
    }

    if matches!(
        layout.seed_layout,
        AdaptiveMaterialSeedLayout::UniformContinuous
            | AdaptiveMaterialSeedLayout::GradedContinuous
    ) {
        return Ok(restricted);
    }
    let fine_units = layout.fine_units.as_ref().ok_or_else(|| {
        AutomataError::InvalidModel(
            "canonical adaptive seed ordering requires integral material groups".to_owned(),
        )
    })?;
    let max_units = fine_units.iter().copied().max().unwrap_or(0);
    let mut rows_by_units = vec![Vec::<usize>::new(); max_units + 1];
    for (row, measure) in restricted.represented_measure.iter().copied().enumerate() {
        let units_float = measure / layout.fine_measure;
        let units = units_float.round() as usize;
        if units == 0 || units > max_units || (units_float - units as f32).abs() > 2.0e-4 {
            return Err(AutomataError::InvalidModel(format!(
                "adaptive Target2D restricted row {row} represents non-layout measure {measure}"
            )));
        }
        rows_by_units[units].push(row);
    }
    let mut unit_cursors = vec![0usize; rows_by_units.len()];
    let mut ordered_rows = Vec::with_capacity(active_count);
    for units in fine_units.iter().copied() {
        let cursor = unit_cursors[units];
        let row = rows_by_units[units].get(cursor).copied().ok_or_else(|| {
            AutomataError::InvalidModel(format!(
                "adaptive Target2D seed has too few {units}-unit rows for the static material layout"
            ))
        })?;
        unit_cursors[units] += 1;
        ordered_rows.push(row);
    }
    if rows_by_units
        .iter()
        .zip(&unit_cursors)
        .any(|(rows, cursor)| rows.len() != *cursor)
    {
        return Err(AutomataError::InvalidModel(
            "adaptive Target2D seed/layout material multiplicities differ".to_owned(),
        ));
    }
    adaptive_particle_subset(&restricted, &ordered_rows)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AdaptiveTarget2dTrainingConfig {
    pub rule_training: AdaptiveTarget2dRuleTraining,
    /// Compact recurrent channels inserted before the RGB tail. The current
    /// implementation is restricted to frozen-base normalized residual
    /// training, where zero memory preserves native NPA behavior exactly.
    pub compact_recurrent_memory_dims: usize,
    /// Target-independent policy used to choose conservative coarse groups
    /// when constructing each active-material seed.
    pub restriction_policy: super::super::AdaptiveHierarchyRestrictionPolicy,
    /// Freeze every pre-existing NPA parameter and optimize only the final
    /// input column added by `SharedScaleConditionedRule`. This is the
    /// function-preserving first phase for adapting a validated fine-scale
    /// rule to mixed material scales.
    pub optimize_material_scale_only: bool,
    /// Optimize `log1p(loss)` per trajectory while retaining the untransformed
    /// objective for reporting and checkpoint evaluation. This bounds the
    /// influence of rare divergent rollouts without changing the ordering of
    /// finite nonnegative trajectory losses.
    pub log1p_trajectory_loss: bool,
    /// Fraction of the highest-loss rollout rows used by the additional
    /// trajectory-tail objective. Zero disables tail weighting.
    pub trajectory_tail_fraction: f32,
    /// Relative weight of the trajectory-tail mean. The final objective is
    /// normalized by `1 + weight`, preserving its approximate scale.
    pub trajectory_tail_weight: f32,
    /// Number of rollout rows replaced from the immutable seed bank whenever
    /// `target2d.inject_seed_interval` fires. One preserves the upstream-style
    /// persistent pool; setting this to the batch size creates a fresh-seed
    /// closure stage without host-side trajectory construction.
    pub fresh_seed_trajectories: usize,
    /// Deterministic fresh seeds evaluated together when selecting a training
    /// checkpoint. Empty uses `target2d.seed`.
    pub checkpoint_seeds: Vec<u64>,
    /// Rollout horizons used by checkpoint selection. The selected score is
    /// the worst PSNR over every configured seed/horizon pair. Empty uses
    /// `target2d.step_max`.
    pub checkpoint_horizons: Vec<usize>,
    /// Recycle persistent pool rows at or above this trajectory age before
    /// sampling them. Zero leaves age unbounded.
    pub max_pool_age_steps: usize,
    /// Number of fixed-width trajectory-age bands represented in each
    /// persistent-pool batch. Zero preserves uniform pool sampling. Values of
    /// two or greater require `max_pool_age_steps` and prevent short-lived
    /// trajectories from crowding long-horizon states out of a batch.
    pub pool_age_strata: usize,
    /// Positive common scale applied before recurrent backward. Values below
    /// one require scale-invariant per-parameter gradient normalization.
    pub backward_loss_scale: f32,
    /// A coarse row represents several independently gated fine particles.
    /// Use the update probability as its mean gate instead of one correlated
    /// Bernoulli draw for the complete represented material.
    pub expected_coarse_update_mask: bool,
    /// Event-relative sampling and loss controls for learning recovery after
    /// detached topology changes.
    pub event_training: AdaptiveTarget2dEventTrainingConfig,
    pub target2d: Target2dTrainingConfig,
    pub material: AdaptiveTarget2dMaterialConfig,
    pub topology: AdaptiveTarget2dTopologyConfig,
}

impl Default for AdaptiveTarget2dTrainingConfig {
    fn default() -> Self {
        Self {
            rule_training: AdaptiveTarget2dRuleTraining::default(),
            compact_recurrent_memory_dims: 0,
            restriction_policy:
                super::super::AdaptiveHierarchyRestrictionPolicy::SpatialCompactness,
            optimize_material_scale_only: false,
            log1p_trajectory_loss: false,
            trajectory_tail_fraction: 0.0,
            trajectory_tail_weight: 0.0,
            fresh_seed_trajectories: 1,
            checkpoint_seeds: Vec::new(),
            checkpoint_horizons: Vec::new(),
            max_pool_age_steps: 0,
            pool_age_strata: 0,
            backward_loss_scale: 1.0,
            expected_coarse_update_mask: false,
            event_training: AdaptiveTarget2dEventTrainingConfig::default(),
            target2d: Target2dTrainingConfig {
                particle_count: 3_070,
                ..Target2dTrainingConfig::default()
            },
            material: AdaptiveTarget2dMaterialConfig::default(),
            topology: AdaptiveTarget2dTopologyConfig::default(),
        }
    }
}

/// Event-relative objective for adaptive recurrent Target2D training.
///
/// Topology selection remains detached at TBPTT boundaries. This objective
/// ensures that selected events are followed by a bounded differentiable
/// recovery rollout instead of being scored only at the instant of exchange.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AdaptiveTarget2dEventTrainingConfig {
    /// Enable event-aware pool sampling and post-event loss weighting.
    pub enabled: bool,
    /// Number of rollout steps for which a trajectory remains in the
    /// post-event recovery objective.
    pub post_event_recovery_steps: usize,
    /// Additional relative weight applied to post-event trajectory losses.
    /// The weighted mean is normalized by the sum of row weights.
    pub post_event_loss_weight: f32,
    /// Weight of the positive post-event degradation relative to the same
    /// row's detached loss immediately before its topology event.
    pub post_event_degradation_weight: f32,
    /// Penalty applied to the worst per-seed PSNR drift across checkpoint
    /// horizons when selecting the returned model.
    pub checkpoint_drift_penalty_weight: f32,
    /// Minimum number of batch rows whose sampled age should cross a scheduled
    /// topology event during the sampled rollout. If the persistent pool lacks
    /// eligible ages and a fresh trajectory can reach an event, additional
    /// rows are restored from the immutable seed bank.
    pub min_event_trajectories_per_batch: usize,
    /// Maximum suffix appended when an event occurs too near the end of a
    /// sampled rollout to expose the requested recovery horizon. Zero reuses
    /// `post_event_recovery_steps`.
    pub max_recovery_extension_steps: usize,
}

impl AdaptiveTarget2dEventTrainingConfig {
    pub fn recovery_extension_budget(self) -> usize {
        if self.max_recovery_extension_steps == 0 {
            self.post_event_recovery_steps
        } else {
            self.max_recovery_extension_steps
        }
    }
}

impl Default for AdaptiveTarget2dEventTrainingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            post_event_recovery_steps: 64,
            post_event_loss_weight: 1.0,
            post_event_degradation_weight: 0.0,
            checkpoint_drift_penalty_weight: 0.0,
            min_event_trajectories_per_batch: 1,
            max_recovery_extension_steps: 0,
        }
    }
}

/// Which recurrent rule receives the adaptive Target2D gradient.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdaptiveTarget2dRuleTraining {
    /// Optimize the primary NPA rule. Retained only as an explicit research
    /// control because it can destroy the validated native-scale dynamics.
    #[default]
    SharedRule,
    /// Optimize one primary NPA rule with an explicit continuous material-scale
    /// input. The added input column starts at zero, preserving the supplied
    /// native-scale rule exactly before adaptive training.
    SharedScaleConditionedRule,
    /// Optimize the shared rule end-to-end through Shepard-normalized,
    /// moment-corrected variable-scale perception with a continuous
    /// material-scale input. Its added input column is initialized to zero, so
    /// widening a supplied fine-scale rule preserves its initial function.
    NormalizedAdaptiveRule,
    /// Preserve the validated native-scale NPA attractor and train a
    /// zero-initialized correction from Shepard-normalized, moment-corrected
    /// variable-scale perception on the same active material rows.
    FrozenBaseNormalizedAdaptiveResidual,
    /// Freeze the validated native-scale rule and train only a compatible
    /// residual that sees continuous row scale and coarse-source exposure.
    FrozenBaseMaterialConditionedResidual,
    /// Freeze the validated native-scale rule and optimize a zero-initialized
    /// mixed-resolution residual over the same NPA-compatible perception
    /// features. The residual is disabled in the uniform native-scale limit.
    FrozenBaseCompatibleResidual,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AdaptiveTarget2dTopologyConfig {
    /// Apply one conservative paired split/merge between TBPTT chunks.
    pub enabled: bool,
    /// Number of completed rollout steps before paired events begin.
    pub start_step: usize,
    /// Last completed rollout step on which topology may change. Zero leaves
    /// adaptation enabled for the full rollout.
    pub end_step: usize,
    /// Multiplier around the covariance-preserving canonical child offset.
    /// One gives the exact isotropic 2D four-child split implied by the
    /// represented material footprint.
    pub split_radius_scale: f32,
    /// Local-detail contribution to the dimensionless compact-neighborhood
    /// merge cost. Spatial distance is normalized by fine footprint squared.
    pub merge_detail_scale: f32,
    /// Required relative local-detail gain for a budget-neutral reallocation.
    /// Zero accepts every strict improvement; one disables reallocation.
    pub min_relative_gain: f32,
    /// Deployment topology interval. Zero matches the TBPTT chunk depth.
    pub interval_steps: usize,
    /// Conservative split/merge pairs attempted per interval.
    pub events_per_interval: usize,
}

impl Default for AdaptiveTarget2dTopologyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            start_step: 32,
            end_step: 0,
            split_radius_scale: 1.0,
            merge_detail_scale: 0.01,
            min_relative_gain: 0.0,
            interval_steps: 0,
            events_per_interval: 1,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveTarget2dGpuTrainingReport {
    pub active_particle_count: usize,
    pub reference_particle_count: usize,
    pub coarse_particle_count: usize,
    pub visible_gaussian_count: usize,
    pub recurrent_row_reduction_fraction: f32,
    pub pair_work_reduction_fraction: f32,
    pub material_measure_error: f32,
    pub training: Target2dGpuTrainingReport,
}

/// Bounded live update from canonical adaptive Target2D training.
#[derive(Clone, Debug)]
pub struct AdaptiveTarget2dGpuTrainingProgress {
    pub step: usize,
    pub total_steps: usize,
    pub loss: f32,
    pub eval_loss: Option<crate::Target2dGpuLossSummary>,
    pub render_rgb_psnr_db: Option<f32>,
    pub base_grad_norm: f32,
    pub base_grad_scale: f32,
    pub particle_steps_per_sec: f64,
    pub elapsed_ms: f64,
    pub model: AdaptiveNpaModel,
}

/// Receives throttled adaptive-model snapshots between optimizer steps.
pub trait AdaptiveTarget2dGpuTrainingObserver: Send {
    fn should_stop(&self) -> bool {
        false
    }

    fn snapshot_interval_steps(&self) -> usize {
        1
    }

    fn snapshot_interval_duration(&self) -> Duration {
        Duration::from_secs(5)
    }

    fn on_progress(&mut self, progress: AdaptiveTarget2dGpuTrainingProgress);
}

pub fn train_adaptive_target_2d_gpu(
    backend: Target2dGpuBackend,
    model: &mut AdaptiveNpaModel,
    hashgrid: &burn_automata_kernels::HashGridConfig,
    target: crate::TargetImage2d,
    config: AdaptiveTarget2dTrainingConfig,
    loss_config: Target2dLossConfig,
    checkpoint: Option<&Target2dGpuCheckpointConfig>,
) -> Result<AdaptiveTarget2dGpuTrainingReport, Box<dyn std::error::Error>> {
    crate::hyper::e2e_training::train_adaptive_target_2d_gpu_impl(
        backend,
        model,
        hashgrid,
        target,
        config,
        loss_config,
        checkpoint,
        None,
    )
}

/// Canonical adaptive Target2D training with cancellable live model updates.
#[allow(clippy::too_many_arguments)]
pub fn train_adaptive_target_2d_gpu_with_observer(
    backend: Target2dGpuBackend,
    model: &mut AdaptiveNpaModel,
    hashgrid: &burn_automata_kernels::HashGridConfig,
    target: crate::TargetImage2d,
    config: AdaptiveTarget2dTrainingConfig,
    loss_config: Target2dLossConfig,
    checkpoint: Option<&Target2dGpuCheckpointConfig>,
    observer: &mut dyn AdaptiveTarget2dGpuTrainingObserver,
) -> Result<AdaptiveTarget2dGpuTrainingReport, Box<dyn std::error::Error>> {
    crate::hyper::e2e_training::train_adaptive_target_2d_gpu_impl(
        backend,
        model,
        hashgrid,
        target,
        config,
        loss_config,
        checkpoint,
        Some(observer),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deployment_seed_bank_is_conservative_and_contains_no_hidden_fine_rows() {
        let material = AdaptiveTarget2dMaterialConfig::default();
        let layout = material.layout(3_070, 0.01, 0.4).unwrap();
        let mut adaptive = crate::adaptive::AdaptiveNpaConfig::growing_2d();
        adaptive.target_leaves = 3_070;
        adaptive.bootstrap_fine_leaves = 4_096;
        adaptive.hierarchical_restriction_policy =
            crate::adaptive::AdaptiveHierarchyRestrictionPolicy::SpatialCompactness;
        adaptive.retain_bootstrap_templates = false;
        adaptive.expected_coarse_update_mask = true;
        let model = AdaptiveNpaModel::seeded(
            crate::NpaModel::seeded(crate::NpaConfig::growing_2d(), 17),
            adaptive,
            19,
        )
        .unwrap();
        let seed = seed_adaptive_particles_scaled(
            &model,
            3_070,
            23,
            ParticleSeed::UniformCircle,
            0.2,
            material.total_measure,
            material.fine_bandwidth,
        )
        .unwrap();
        assert_eq!(seed.len(), 3_070);
        assert!(seed.bootstrap_templates.is_empty());
        assert_eq!(
            seed.represented_measure
                .iter()
                .filter(|measure| { (**measure / layout.fine_measure - 4.0).abs() < 2.0e-4 })
                .count(),
            342
        );
        for (measure, bandwidth) in seed.represented_measure.iter().zip(&seed.bandwidth) {
            let units = *measure / layout.fine_measure;
            assert!((*bandwidth - material.fine_bandwidth * units.sqrt()).abs() < 2.0e-6);
        }
        let ordered = adaptive_target2d_seed_particles(
            &model,
            &layout,
            23,
            ParticleSeed::UniformCircle,
            0.2,
            material.total_measure,
            material.fine_bandwidth,
        )
        .unwrap();
        for (row, units) in layout
            .fine_units
            .as_ref()
            .unwrap()
            .iter()
            .copied()
            .enumerate()
        {
            assert!(
                (ordered.represented_measure[row] / layout.fine_measure - units as f32).abs()
                    < 2.0e-4
            );
        }

        let bank = build_adaptive_target2d_seed_bank(
            &model,
            &layout,
            2,
            29,
            &[31, 33],
            ParticleSeed::UniformCircle,
            0.2,
            material.total_measure,
            material.fine_bandwidth,
        )
        .unwrap();
        assert_eq!(bank.positions.len(), 2 * 3_070 * 2);
        assert_eq!(bank.states.len(), 2 * 3_070 * model.rule.config.state_dims);
        assert_eq!(bank.update_masks.len(), 2 * 3_070);
        assert_eq!(
            bank.update_masks
                .iter()
                .filter(|mask| mask.expected)
                .count(),
            2 * 342
        );
        assert_eq!(bank.eval_positions.len(), 2 * 3_070 * 2);
        assert_eq!(
            bank.eval_states.len(),
            2 * 3_070 * model.rule.config.state_dims
        );
        assert_eq!(bank.eval_update_masks.len(), 2 * 3_070);
        assert_eq!(
            bank.eval_update_masks
                .iter()
                .filter(|mask| mask.expected)
                .count(),
            2 * 342
        );
        assert_eq!(bank.eval_seeds, vec![31, 33]);
        assert!(bank.max_measure_relative_error < 2.0e-5);
        assert!(bank.max_centroid_l2_error < 2.0e-5);
        assert!(bank.max_extensive_state_l2_error < 2.0e-5);
    }

    #[test]
    fn compact_recurrent_memory_seeds_only_coarse_geometry_and_preserves_rgb_tail() {
        let material = AdaptiveTarget2dMaterialConfig {
            reference_particle_count: 64,
            ..AdaptiveTarget2dMaterialConfig::default()
        };
        let layout = material.layout(61, 0.01, 0.4).unwrap();
        let mut adaptive = crate::adaptive::AdaptiveNpaConfig::growing_2d();
        adaptive.min_leaves = 61;
        adaptive.target_leaves = 61;
        adaptive.max_leaves = 64;
        adaptive.bootstrap_fine_leaves = 64;
        adaptive.retain_bootstrap_templates = false;
        let mut model = crate::AdaptiveNpaModel::seeded(
            crate::NpaModel::upstream_seeded(crate::NpaConfig::growing_2d(), 7),
            adaptive,
            11,
        )
        .unwrap();
        model.enable_compact_recurrent_memory(8).unwrap();
        let particles = adaptive_target2d_seed_particles(
            &model,
            &layout,
            42,
            ParticleSeed::UniformCircle,
            0.2,
            material.total_measure,
            material.fine_bandwidth,
        )
        .unwrap();
        let state_dims = particles.state_dims;
        let memory_start = state_dims - 3 - 8;
        let fine_units = layout.fine_units.as_ref().unwrap();
        let mut coarse_memory_nonzero = false;
        for (row, units) in fine_units.iter().copied().enumerate() {
            let state = &particles.states[row * state_dims..(row + 1) * state_dims];
            let memory = &state[memory_start..memory_start + 8];
            if units == 1 {
                assert!(memory.iter().all(|value| value.abs() < 1.0e-7));
            } else {
                coarse_memory_nonzero |= memory.iter().any(|value| value.abs() > 1.0e-5);
            }
            assert!(state[state_dims - 3..].iter().all(|value| *value == 0.0));
        }
        assert!(coarse_memory_nonzero);
        let bank = build_adaptive_target2d_seed_bank(
            &model,
            &layout,
            2,
            42,
            &[43, 44],
            ParticleSeed::UniformCircle,
            0.2,
            material.total_measure,
            material.fine_bandwidth,
        )
        .unwrap();
        assert!(bank.max_measure_relative_error < 2.0e-5);
        assert!(bank.max_centroid_l2_error < 2.0e-5);
        assert!(bank.max_extensive_state_l2_error < 2.0e-5);
    }

    #[test]
    fn compact_recurrent_memory_requires_the_canonical_normalized_residual_path() {
        let make_model = || {
            let mut adaptive = crate::adaptive::AdaptiveNpaConfig::growing_2d();
            adaptive.min_leaves = 61;
            adaptive.target_leaves = 61;
            adaptive.max_leaves = 64;
            adaptive.bootstrap_fine_leaves = 64;
            adaptive.retain_bootstrap_templates = false;
            crate::AdaptiveNpaModel::seeded(
                crate::NpaModel::upstream_seeded(crate::NpaConfig::growing_2d(), 7),
                adaptive,
                11,
            )
            .unwrap()
        };
        let mut config = AdaptiveTarget2dTrainingConfig {
            rule_training: AdaptiveTarget2dRuleTraining::FrozenBaseNormalizedAdaptiveResidual,
            compact_recurrent_memory_dims: 8,
            target2d: crate::Target2dTrainingConfig {
                particle_count: 61,
                ..crate::Target2dTrainingConfig::default()
            },
            material: AdaptiveTarget2dMaterialConfig {
                reference_particle_count: 64,
                ..AdaptiveTarget2dMaterialConfig::default()
            },
            ..AdaptiveTarget2dTrainingConfig::default()
        };
        let mut model = make_model();
        crate::hyper::e2e_training::prepare_adaptive_target2d_model(&mut model, &config).unwrap();
        assert_eq!(model.config.compact_recurrent_memory_dims, 8);
        assert_eq!(model.rule.config.state_dims, 24);
        assert_eq!(
            model
                .local_residual_rule
                .as_ref()
                .unwrap()
                .config
                .state_dims,
            24
        );

        config.rule_training = AdaptiveTarget2dRuleTraining::SharedRule;
        let error =
            crate::hyper::e2e_training::prepare_adaptive_target2d_model(&mut make_model(), &config)
                .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("canonical-grouped frozen-base-normalized-adaptive-residual")
        );
    }

    #[test]
    fn budgeted_layout_conserves_4096_fine_units_at_3070_rows() {
        let layout = AdaptiveTarget2dMaterialConfig::default()
            .layout(3_070, 0.01, 0.4)
            .unwrap();
        assert_eq!(layout.active_particle_count(), 3_070);
        assert_eq!(layout.coarse_particle_count(), 342);
        let fine_units = layout.fine_units.as_ref().unwrap();
        assert_eq!(fine_units.iter().sum::<usize>(), 4_096);
        assert_eq!(fine_units.iter().filter(|units| **units == 4).count(), 342);
        assert!(
            (layout.represented_measure.iter().sum::<f32>()
                - AdaptiveTarget2dMaterialConfig::default().total_measure)
                .abs()
                < 3.0e-6
        );
    }

    #[test]
    fn equal_budget_layout_is_the_fixed_npa_limit() {
        let config = AdaptiveTarget2dMaterialConfig::default();
        let layout = config.layout(4_096, 0.01, 0.4).unwrap();
        assert!(
            layout
                .fine_units
                .as_ref()
                .unwrap()
                .iter()
                .all(|units| *units == 1)
        );
        assert!(
            layout
                .bandwidth
                .iter()
                .all(|bandwidth| (*bandwidth - config.fine_bandwidth).abs() < f32::EPSILON)
        );
        assert!(
            layout
                .footprint_ratio
                .iter()
                .all(|ratio| (*ratio - 1.0).abs() < f32::EPSILON)
        );
    }

    #[test]
    fn continuous_uniform_layout_uses_noninteger_conserved_measure() {
        let config = AdaptiveTarget2dMaterialConfig {
            seed_layout: AdaptiveMaterialSeedLayout::UniformContinuous,
            bandwidth_exponent: 0.0,
            ..AdaptiveTarget2dMaterialConfig::default()
        };
        let layout = config.layout(3_070, 0.01, 0.4).unwrap();
        let expected_units = 4_096.0 / 3_070.0;
        assert!(layout.fine_units.is_none());
        assert!(
            layout
                .represented_fine_units
                .iter()
                .all(|units| (*units - expected_units).abs() < 1.0e-6)
        );
        assert!(
            layout
                .bandwidth
                .iter()
                .all(|bandwidth| (*bandwidth - config.fine_bandwidth).abs() < f32::EPSILON)
        );
        assert!(
            (layout.represented_measure.iter().sum::<f32>() - config.total_measure).abs() < 3.0e-6
        );

        let mut adaptive = crate::adaptive::AdaptiveNpaConfig::growing_2d();
        adaptive.target_leaves = 3_070;
        adaptive.bootstrap_fine_leaves = 4_096;
        adaptive.material_seed_layout = AdaptiveMaterialSeedLayout::UniformContinuous;
        adaptive.material_seed_bandwidth_exponent = 0.0;
        adaptive.retain_bootstrap_templates = false;
        let model = AdaptiveNpaModel::seeded(
            crate::NpaModel::seeded(crate::NpaConfig::growing_2d(), 17),
            adaptive,
            19,
        )
        .unwrap();
        let bank = build_adaptive_target2d_seed_bank(
            &model,
            &layout,
            2,
            29,
            &[31],
            ParticleSeed::UniformCircle,
            0.2,
            config.total_measure,
            config.fine_bandwidth,
        )
        .unwrap();
        assert!(bank.max_measure_relative_error < 2.0e-5);
        assert!(bank.max_centroid_l2_error < 2.0e-5);
        assert!(bank.max_extensive_state_l2_error < 2.0e-5);
    }

    #[test]
    fn graded_continuous_layout_is_deterministic_and_conservative() {
        let config = AdaptiveTarget2dMaterialConfig {
            seed_layout: AdaptiveMaterialSeedLayout::GradedContinuous,
            seed_measure_ratio: 1.44,
            bandwidth_exponent: 0.0,
            ..AdaptiveTarget2dMaterialConfig::default()
        };
        let layout = config.layout(3_070, 0.01, 0.4).unwrap();
        assert!(layout.fine_units.is_none());
        let min_units = layout
            .represented_fine_units
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let max_units = layout
            .represented_fine_units
            .iter()
            .copied()
            .fold(0.0_f32, f32::max);
        assert!((max_units / min_units - 1.44).abs() < 2.0e-3);
        assert!((layout.represented_fine_units.iter().sum::<f32>() - 4_096.0).abs() < 2.0e-3);
        assert!(
            layout
                .represented_fine_units
                .windows(2)
                .any(|pair| pair[0] > pair[1])
        );

        let mut adaptive = crate::adaptive::AdaptiveNpaConfig::growing_2d();
        adaptive.target_leaves = 3_070;
        adaptive.bootstrap_fine_leaves = 4_096;
        adaptive.material_seed_layout = AdaptiveMaterialSeedLayout::GradedContinuous;
        adaptive.material_seed_bandwidth_exponent = 0.0;
        adaptive.material_seed_measure_ratio = config.seed_measure_ratio;
        adaptive.retain_bootstrap_templates = false;
        let model = AdaptiveNpaModel::seeded(
            crate::NpaModel::seeded(crate::NpaConfig::growing_2d(), 17),
            adaptive,
            19,
        )
        .unwrap();
        let particles = adaptive_target2d_seed_particles(
            &model,
            &layout,
            31,
            ParticleSeed::UniformCircle,
            0.2,
            config.total_measure,
            config.fine_bandwidth,
        )
        .unwrap();
        assert_eq!(particles.represented_measure, layout.represented_measure);
        assert_eq!(particles.bandwidth, layout.bandwidth);
        assert!((particles.total_measure() - f64::from(config.total_measure)).abs() < 2.0e-6);

        let bank = build_adaptive_target2d_seed_bank(
            &model,
            &layout,
            2,
            29,
            &[31],
            ParticleSeed::UniformCircle,
            0.2,
            config.total_measure,
            config.fine_bandwidth,
        )
        .unwrap();
        assert!(bank.max_measure_relative_error < 2.0e-5);
        assert!(bank.max_centroid_l2_error < 2.0e-5);
        assert!(bank.max_extensive_state_l2_error < 2.0e-5);
    }
}
