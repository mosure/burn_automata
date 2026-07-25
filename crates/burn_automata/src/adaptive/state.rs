use serde::{Deserialize, Serialize};

use crate::{AutomataError, AutomataResult};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveBootstrapChild {
    pub position: [f32; 4],
    pub state: Vec<f32>,
    pub represented_measure: f32,
    pub bandwidth: f32,
    pub covariance: [f32; 9],
    pub particle_id: u64,
    pub generation: u16,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveBootstrapTemplate {
    pub parent_id: u64,
    pub children: Vec<AdaptiveBootstrapChild>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveParticleSet {
    pub spatial_dims: usize,
    pub state_dims: usize,
    pub positions: Vec<[f32; 4]>,
    pub states: Vec<f32>,
    /// Physical intensive-state Jacobian, row-major as
    /// `[particle, state_channel, spatial_axis]`. It is refreshed from
    /// normalized perception and conservatively fitted during merges.
    #[serde(default)]
    pub state_jacobian: Vec<f32>,
    /// One compact affine-null residual value per state channel. Native leaves
    /// and legacy artifacts carry zeros; first-level coarse leaves may use it
    /// as recurrent closure state when explicitly enabled by the model.
    #[serde(default)]
    pub closure_mode: Vec<f32>,
    /// Persistent orientation anchor for the four-child affine-null basis.
    /// This removes the otherwise unobservable per-aggregate sign gauge from
    /// recurrent closure targets. Values are laid out `[particle, 4]`.
    #[serde(default)]
    pub closure_basis: Vec<f32>,
    /// Two-component continuous phase of the unresolved four-child geometry.
    /// Native leaves carry zero; first-level coarse leaves use this as compact
    /// recurrent closure state alongside `closure_mode`.
    #[serde(default)]
    pub closure_phase: Vec<f32>,
    pub represented_measure: Vec<f32>,
    /// Continuously displayed Gaussian footprint. Topology changes represented
    /// measure discretely, while this value inherits the pre-event scale and
    /// relaxes toward the new physical footprint without a visual pop.
    pub render_footprint: Vec<f32>,
    pub bandwidth: Vec<f32>,
    pub covariance: Vec<[f32; 9]>,
    pub particle_id: Vec<u64>,
    pub sibling_group: Vec<u64>,
    pub generation: Vec<u16>,
    pub cooldown: Vec<u16>,
    pub next_id: u64,
    pub next_sibling_group: u64,
    /// Non-material refinement stencils for a hierarchy-restricted seed. They
    /// are consumed when their coarse parent splits and never enter dynamics.
    #[serde(default)]
    pub bootstrap_templates: Vec<AdaptiveBootstrapTemplate>,
}

impl AdaptiveParticleSet {
    pub fn from_equal_measure(
        positions: Vec<[f32; 4]>,
        states: Vec<f32>,
        spatial_dims: usize,
        state_dims: usize,
        total_measure: f32,
        bandwidth: f32,
    ) -> AutomataResult<Self> {
        let count = positions.len();
        if count == 0 {
            return Err(AutomataError::InvalidArgument(
                "adaptive particle set cannot be empty".to_string(),
            ));
        }
        if states.len() != count * state_dims {
            return Err(AutomataError::InvalidArgument(format!(
                "adaptive state len {} != {}",
                states.len(),
                count * state_dims
            )));
        }
        let measure = total_measure / count as f32;
        let footprint = material_footprint_radius(measure, spatial_dims);
        let variance = (0.5 * footprint).powi(2);
        let mut covariance = [0.0; 9];
        for axis in 0..spatial_dims {
            covariance[axis * 3 + axis] = variance;
        }
        let particles = Self {
            spatial_dims,
            state_dims,
            positions,
            states,
            state_jacobian: vec![0.0; count * state_dims * spatial_dims],
            closure_mode: vec![0.0; count * state_dims],
            closure_basis: vec![0.0; count * 4],
            closure_phase: vec![0.0; count * 2],
            represented_measure: vec![measure; count],
            render_footprint: vec![footprint; count],
            bandwidth: vec![bandwidth; count],
            covariance: vec![covariance; count],
            particle_id: (0..count as u64).collect(),
            sibling_group: vec![0; count],
            generation: vec![0; count],
            cooldown: vec![0; count],
            next_id: count as u64,
            next_sibling_group: 1,
            bootstrap_templates: Vec::new(),
        };
        particles.validate()?;
        Ok(particles)
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub fn total_measure(&self) -> f64 {
        self.represented_measure
            .iter()
            .map(|value| *value as f64)
            .sum()
    }

    pub fn footprint(&self, index: usize) -> f32 {
        material_footprint_radius(self.represented_measure[index], self.spatial_dims)
    }

    pub fn decrement_cooldown(&mut self) {
        self.decrement_cooldown_by(1);
    }

    pub fn decrement_cooldown_by(&mut self, steps: usize) {
        let steps = u16::try_from(steps).unwrap_or(u16::MAX);
        self.cooldown
            .iter_mut()
            .for_each(|value| *value = value.saturating_sub(steps));
    }

    pub fn validate(&self) -> AutomataResult<()> {
        if !(self.spatial_dims == 2 || self.spatial_dims == 3) || self.state_dims == 0 {
            return Err(AutomataError::InvalidArgument(
                "adaptive particle dimensions must be 2D/3D with non-zero state".to_string(),
            ));
        }
        let count = self.positions.len();
        if count == 0 || self.states.len() != count * self.state_dims {
            return Err(AutomataError::InvalidArgument(
                "adaptive particle/state shape mismatch".to_string(),
            ));
        }
        let jacobian_len = count * self.state_dims * self.spatial_dims;
        if self.state_jacobian.len() != jacobian_len
            || self.state_jacobian.iter().any(|value| !value.is_finite())
        {
            return Err(AutomataError::InvalidArgument(format!(
                "adaptive state Jacobian len {} != {jacobian_len} or contains non-finite values",
                self.state_jacobian.len(),
            )));
        }
        if !self.closure_mode.is_empty()
            && (self.closure_mode.len() != count * self.state_dims
                || self.closure_mode.iter().any(|value| !value.is_finite()))
        {
            return Err(AutomataError::InvalidArgument(format!(
                "adaptive closure-mode len {} != {} or contains non-finite values",
                self.closure_mode.len(),
                count * self.state_dims,
            )));
        }
        if !self.closure_basis.is_empty()
            && (self.closure_basis.len() != count * 4
                || self.closure_basis.iter().any(|value| !value.is_finite()))
        {
            return Err(AutomataError::InvalidArgument(format!(
                "adaptive closure-basis len {} != {} or contains non-finite values",
                self.closure_basis.len(),
                count * 4,
            )));
        }
        if !self.closure_phase.is_empty()
            && (self.closure_phase.len() != count * 2
                || self.closure_phase.iter().any(|value| !value.is_finite()))
        {
            return Err(AutomataError::InvalidArgument(format!(
                "adaptive closure-phase len {} != {} or contains non-finite values",
                self.closure_phase.len(),
                count * 2,
            )));
        }
        for (name, len) in [
            ("represented_measure", self.represented_measure.len()),
            ("render_footprint", self.render_footprint.len()),
            ("bandwidth", self.bandwidth.len()),
            ("covariance", self.covariance.len()),
            ("particle_id", self.particle_id.len()),
            ("sibling_group", self.sibling_group.len()),
            ("generation", self.generation.len()),
            ("cooldown", self.cooldown.len()),
        ] {
            if len != count {
                return Err(AutomataError::InvalidArgument(format!(
                    "adaptive {name} len {len} != particle count {count}"
                )));
            }
        }
        if self.positions.iter().any(|position| {
            position
                .iter()
                .take(self.spatial_dims)
                .any(|value| !value.is_finite())
        }) || self.states.iter().any(|value| !value.is_finite())
            || self
                .represented_measure
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            || self
                .render_footprint
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            || self
                .bandwidth
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            || self
                .covariance
                .iter()
                .any(|matrix| !covariance_is_spd(matrix, self.spatial_dims))
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive particle set contains invalid numeric values".to_string(),
            ));
        }
        let mut template_parents = std::collections::BTreeSet::new();
        let active_ids = self
            .particle_id
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        let mut child_ids = std::collections::BTreeSet::new();
        for template in &self.bootstrap_templates {
            let Some(parent_index) = self
                .particle_id
                .iter()
                .position(|id| *id == template.parent_id)
            else {
                return Err(AutomataError::InvalidArgument(format!(
                    "adaptive bootstrap parent {} is not material",
                    template.parent_id
                )));
            };
            if !template_parents.insert(template.parent_id) || template.children.is_empty() {
                return Err(AutomataError::InvalidArgument(
                    "adaptive bootstrap templates must have unique parents and children"
                        .to_string(),
                ));
            }
            let child_measure = template
                .children
                .iter()
                .map(|child| child.represented_measure)
                .sum::<f32>();
            let parent_measure = self.represented_measure[parent_index];
            if (child_measure - parent_measure).abs()
                > 1.0e-5 * parent_measure.abs().max(f32::MIN_POSITIVE)
            {
                return Err(AutomataError::InvalidArgument(format!(
                    "adaptive bootstrap child measure {child_measure} does not match parent measure {parent_measure}"
                )));
            }
            for child in &template.children {
                if child.state.len() != self.state_dims
                    || child
                        .position
                        .iter()
                        .take(self.spatial_dims)
                        .any(|value| !value.is_finite())
                    || child.state.iter().any(|value| !value.is_finite())
                    || !child.represented_measure.is_finite()
                    || child.represented_measure <= 0.0
                    || !child.bandwidth.is_finite()
                    || child.bandwidth <= 0.0
                    || !covariance_is_spd(&child.covariance, self.spatial_dims)
                    || !child_ids.insert(child.particle_id)
                {
                    return Err(AutomataError::InvalidArgument(
                        "adaptive bootstrap template contains invalid child material".to_string(),
                    ));
                }
            }
        }
        if !self.bootstrap_templates.is_empty()
            && child_ids.iter().any(|id| active_ids.contains(id))
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive bootstrap child IDs overlap active material IDs".to_string(),
            ));
        }
        Ok(())
    }
}

/// Fits the physical intensive-state Jacobian carried by a conservative
/// material aggregate. The fit includes each child's retained within-leaf
/// Jacobian, so repeated restriction does not discard affine detail.
pub(crate) fn fit_state_jacobian(
    particles: &AdaptiveParticleSet,
    indices: &[usize],
    mean_state: &[f32],
    merged_position: [f32; 4],
    merged_covariance: [f32; 9],
    total_measure: f32,
) -> AutomataResult<Vec<f32>> {
    let dim = particles.spatial_dims;
    if indices.is_empty()
        || mean_state.len() != particles.state_dims
        || !total_measure.is_finite()
        || total_measure <= 0.0
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive state-Jacobian fit received an invalid aggregate".to_string(),
        ));
    }
    let inverse_covariance = invert_small_covariance(merged_covariance, dim).ok_or_else(|| {
        AutomataError::InvalidArgument(
            "adaptive aggregate has a non-invertible reconstruction footprint".to_string(),
        )
    })?;
    let jacobian_dims = particles.state_dims * dim;
    let mut cross = vec![0.0_f32; jacobian_dims];
    for index in indices {
        let weight = particles.represented_measure[*index] / total_measure;
        let state_base = *index * particles.state_dims;
        let jacobian_base = *index * jacobian_dims;
        for channel in 0..particles.state_dims {
            let state_delta = particles.states[state_base + channel] - mean_state[channel];
            for axis in 0..dim {
                let position_delta = particles.positions[*index][axis] - merged_position[axis];
                let within = (0..dim)
                    .map(|inner| {
                        particles.state_jacobian[jacobian_base + channel * dim + inner]
                            * particles.covariance[*index][inner * 3 + axis]
                    })
                    .sum::<f32>();
                cross[channel * dim + axis] += weight * (state_delta * position_delta + within);
            }
        }
    }
    let mut fitted = vec![0.0_f32; jacobian_dims];
    for channel in 0..particles.state_dims {
        for out_axis in 0..dim {
            fitted[channel * dim + out_axis] = (0..dim)
                .map(|inner| {
                    cross[channel * dim + inner] * inverse_covariance[inner * dim + out_axis]
                })
                .sum();
        }
    }
    Ok(fitted)
}

fn invert_small_covariance(matrix: [f32; 9], dim: usize) -> Option<Vec<f32>> {
    if !(1..=3).contains(&dim) {
        return None;
    }
    let stride = dim * 2;
    let mut augmented = vec![0.0_f32; dim * stride];
    for row in 0..dim {
        for col in 0..dim {
            augmented[row * stride + col] = matrix[row * 3 + col];
        }
        augmented[row * stride + dim + row] = 1.0;
    }
    for pivot in 0..dim {
        let selected = (pivot..dim).max_by(|lhs, rhs| {
            augmented[*lhs * stride + pivot]
                .abs()
                .total_cmp(&augmented[*rhs * stride + pivot].abs())
        })?;
        if augmented[selected * stride + pivot].abs() <= 1.0e-14 {
            return None;
        }
        if selected != pivot {
            for col in 0..stride {
                augmented.swap(pivot * stride + col, selected * stride + col);
            }
        }
        let scale = augmented[pivot * stride + pivot].recip();
        for col in 0..stride {
            augmented[pivot * stride + col] *= scale;
        }
        for row in 0..dim {
            if row == pivot {
                continue;
            }
            let factor = augmented[row * stride + pivot];
            for col in 0..stride {
                augmented[row * stride + col] -= factor * augmented[pivot * stride + col];
            }
        }
    }
    Some(
        (0..dim)
            .flat_map(|row| {
                let augmented = &augmented;
                (0..dim).map(move |col| augmented[row * stride + dim + col])
            })
            .collect(),
    )
}

fn covariance_is_spd(matrix: &[f32; 9], dim: usize) -> bool {
    let mut factor = [0.0_f32; 9];
    for row in 0..dim {
        for col in 0..=row {
            let lhs = matrix[row * 3 + col];
            let rhs = matrix[col * 3 + row];
            if !lhs.is_finite()
                || !rhs.is_finite()
                || (lhs - rhs).abs() > 1.0e-5 * lhs.abs().max(rhs.abs()).max(1.0)
            {
                return false;
            }
            let mut value = lhs;
            for k in 0..col {
                value -= factor[row * 3 + k] * factor[col * 3 + k];
            }
            if row == col {
                if !value.is_finite() || value <= 1.0e-12 {
                    return false;
                }
                factor[row * 3 + col] = value.sqrt();
            } else {
                factor[row * 3 + col] = value / factor[col * 3 + col];
            }
        }
    }
    true
}

pub fn unit_ball_measure(dim: usize) -> f32 {
    match dim {
        1 => 2.0,
        2 => std::f32::consts::PI,
        3 => 4.0 * std::f32::consts::PI / 3.0,
        4 => std::f32::consts::PI.powi(2) / 2.0,
        _ => f32::NAN,
    }
}

pub fn material_footprint_radius(represented_measure: f32, dim: usize) -> f32 {
    (represented_measure / unit_ball_measure(dim)).powf(1.0 / dim as f32)
}
