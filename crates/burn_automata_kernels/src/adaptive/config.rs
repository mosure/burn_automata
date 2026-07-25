#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{KernelError, KernelResult};

use super::scale_bins::AdaptiveSupportBins;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum AdaptiveGraphPolicy {
    /// Accept every pair whose compact support overlaps. Intended for oracles.
    RawSupport,
    /// Keep the nearest normalized-distance neighbors for each target.
    #[default]
    DirectedTopK,
    /// Keep an edge only when both endpoint top-k lists contain the other.
    MutualTopK,
}

/// Feature semantics executed by the device adaptive-perception operator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum AdaptivePerceptionSemantics {
    /// Represented-volume SPH features that reduce to the fixed NPA operator.
    #[default]
    NpaCompatible,
    /// Shepard-normalized, moment-corrected variable-scale features from the
    /// budgeted adaptive NPA formulation.
    NormalizedAdaptive,
}

/// Normalization controls for the represented-measure SPH operator used by
/// existing NPA rules. With equal measure and bandwidth these match the fixed
/// NPA perception contract exactly.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AdaptiveNpaPerceptionOptions {
    pub eps0: f32,
    pub scale_equivariance: bool,
    pub particle_density_equivariance: bool,
    pub log_norm_grad: bool,
    pub log_norm_density_grad: bool,
    pub position_features: bool,
}

impl AdaptiveNpaPerceptionOptions {
    pub fn validate(self) -> KernelResult<()> {
        if !self.eps0.is_finite() || self.eps0 <= 0.0 {
            return Err(KernelError::InvalidArgument(format!(
                "adaptive NPA eps0 must be finite and positive, got {}",
                self.eps0
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AdaptivePerceptionConfig {
    pub dim: usize,
    pub graph_policy: AdaptiveGraphPolicy,
    pub max_neighbors: usize,
    /// Power mean used for the symmetric pair support. The paper recommends p >= 4.
    pub pair_scale_power: f32,
    /// Represented measure of one native-resolution particle. Positive values
    /// classify coarse density by represented material instead of communication
    /// support. Zero preserves the legacy bandwidth-based classification.
    #[cfg_attr(feature = "serde", serde(default))]
    pub reference_measure: f32,
    pub min_bandwidth: f32,
    pub max_bandwidth: f32,
    /// Ratio between conservative source-support search bins. This affects
    /// broad-phase candidate work only; exact pair support remains continuous.
    #[cfg_attr(feature = "serde", serde(default = "default_support_bin_ratio"))]
    pub support_bin_ratio: f32,
    pub spacing_target_neighbors: f32,
    pub spacing_root_iterations: usize,
    pub shepard_epsilon: f32,
    pub moment_regularization: f32,
    pub moment_condition_limit: f32,
    pub log_normalize_gradients: bool,
    pub include_position_features: bool,
}

impl AdaptivePerceptionConfig {
    pub fn growing_2d() -> Self {
        Self {
            dim: 2,
            graph_policy: AdaptiveGraphPolicy::DirectedTopK,
            max_neighbors: 64,
            pair_scale_power: 8.0,
            reference_measure: 0.0,
            min_bandwidth: 0.025,
            max_bandwidth: 0.4,
            support_bin_ratio: default_support_bin_ratio(),
            spacing_target_neighbors: 16.0,
            spacing_root_iterations: 16,
            shepard_epsilon: 1.0e-8,
            moment_regularization: 1.0e-4,
            moment_condition_limit: 1.0e5,
            log_normalize_gradients: true,
            include_position_features: false,
        }
    }

    pub fn sparse_3d() -> Self {
        Self {
            dim: 3,
            min_bandwidth: 0.02,
            max_bandwidth: 0.35,
            spacing_target_neighbors: 32.0,
            ..Self::growing_2d()
        }
    }

    pub fn feature_dims(&self, state_dims: usize) -> usize {
        state_dims * 2
            + state_dims * self.dim
            + self.dim
            + usize::from(self.include_position_features) * self.dim
    }

    pub fn validate(&self) -> KernelResult<()> {
        if !(self.dim == 2 || self.dim == 3) {
            return Err(KernelError::InvalidDim(self.dim));
        }
        if self.max_neighbors == 0 {
            return Err(KernelError::InvalidArgument(
                "adaptive max_neighbors must be non-zero".to_string(),
            ));
        }
        if !self.pair_scale_power.is_finite() || self.pair_scale_power < 1.0 {
            return Err(KernelError::InvalidArgument(format!(
                "adaptive pair_scale_power must be finite and >= 1, got {}",
                self.pair_scale_power
            )));
        }
        if !self.reference_measure.is_finite() || self.reference_measure < 0.0 {
            return Err(KernelError::InvalidArgument(format!(
                "adaptive reference_measure must be finite and non-negative, got {}",
                self.reference_measure
            )));
        }
        if !self.min_bandwidth.is_finite()
            || !self.max_bandwidth.is_finite()
            || self.min_bandwidth <= 0.0
            || self.max_bandwidth < self.min_bandwidth
        {
            return Err(KernelError::InvalidArgument(format!(
                "adaptive bandwidth bounds must satisfy 0 < min <= max, got {}..{}",
                self.min_bandwidth, self.max_bandwidth
            )));
        }
        if !self.support_bin_ratio.is_finite() || self.support_bin_ratio <= 1.0 {
            return Err(KernelError::InvalidArgument(format!(
                "adaptive support_bin_ratio must be finite and > 1, got {}",
                self.support_bin_ratio
            )));
        }
        AdaptiveSupportBins::new(
            self.min_bandwidth,
            self.max_bandwidth,
            self.support_bin_ratio,
        )?;
        if !self.spacing_target_neighbors.is_finite() || self.spacing_target_neighbors <= 0.0 {
            return Err(KernelError::InvalidArgument(format!(
                "adaptive spacing target must be finite and positive, got {}",
                self.spacing_target_neighbors
            )));
        }
        if self.spacing_root_iterations == 0 {
            return Err(KernelError::InvalidArgument(
                "adaptive spacing_root_iterations must be non-zero".to_string(),
            ));
        }
        if !self.shepard_epsilon.is_finite() || self.shepard_epsilon <= 0.0 {
            return Err(KernelError::InvalidArgument(
                "adaptive shepard_epsilon must be finite and positive".to_string(),
            ));
        }
        if !self.moment_regularization.is_finite() || self.moment_regularization < 0.0 {
            return Err(KernelError::InvalidArgument(
                "adaptive moment_regularization must be finite and non-negative".to_string(),
            ));
        }
        if !self.moment_condition_limit.is_finite() || self.moment_condition_limit < 1.0 {
            return Err(KernelError::InvalidArgument(
                "adaptive moment_condition_limit must be finite and >= 1".to_string(),
            ));
        }
        Ok(())
    }
}

const fn default_support_bin_ratio() -> f32 {
    2.0
}

impl Default for AdaptivePerceptionConfig {
    fn default() -> Self {
        Self::growing_2d()
    }
}
