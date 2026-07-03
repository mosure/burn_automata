use serde::{Deserialize, Serialize};

use crate::{AutomataError, AutomataResult, NpaConfig};

use super::condition::{ConditionImage2d, ConditionSummary2d};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ParticlePriorConfig {
    pub min_particles: usize,
    pub max_particles: usize,
    pub min_seed_scale: f32,
    pub max_seed_scale: f32,
}

impl Default for ParticlePriorConfig {
    fn default() -> Self {
        Self {
            min_particles: 256,
            max_particles: 4096,
            min_seed_scale: 0.05,
            max_seed_scale: 1.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParticlePrior2d {
    pub particle_count: usize,
    pub seed_scale: f32,
    pub center: [f32; 2],
    pub occupancy: f32,
    pub initial_state: Vec<f32>,
}

impl ParticlePrior2d {
    pub fn from_condition(
        npa_config: &NpaConfig,
        condition: &ConditionImage2d,
        prior_config: ParticlePriorConfig,
    ) -> AutomataResult<Self> {
        validate_prior_config(prior_config)?;
        validate_2d_config(npa_config)?;
        let summary = condition.summary()?;
        Self::from_summary(npa_config, &summary, prior_config)
    }

    pub fn from_summary(
        npa_config: &NpaConfig,
        summary: &ConditionSummary2d,
        prior_config: ParticlePriorConfig,
    ) -> AutomataResult<Self> {
        validate_prior_config(prior_config)?;
        validate_2d_config(npa_config)?;
        let occupancy = summary.occupancy.clamp(0.0, 1.0);
        let span = prior_config
            .max_particles
            .saturating_sub(prior_config.min_particles);
        let particle_count =
            prior_config.min_particles + (span as f32 * occupancy).round() as usize;
        let seed_scale = lerp(
            prior_config.min_seed_scale,
            prior_config.max_seed_scale,
            occupancy.max(summary.edge_energy).clamp(0.0, 1.0),
        );
        let mut initial_state = vec![0.0; npa_config.state_dims];
        let scalars = [
            summary.mean_luma,
            summary.variance_luma,
            summary.occupancy,
            summary.edge_energy,
            summary.center_of_mass[0],
            summary.center_of_mass[1],
            summary.mean_rgb[0],
            summary.mean_rgb[1],
            summary.mean_rgb[2],
        ];
        for (dst, src) in initial_state.iter_mut().zip(scalars) {
            *dst = src;
        }

        Ok(Self {
            particle_count,
            seed_scale,
            center: summary.center_of_mass,
            occupancy,
            initial_state,
        })
    }
}

fn validate_2d_config(config: &NpaConfig) -> AutomataResult<()> {
    if config.spatial_dims != 2 {
        return Err(AutomataError::InvalidArgument(format!(
            "2D hyper prior requires spatial_dims=2, got {}",
            config.spatial_dims
        )));
    }
    if config.state_dims == 0 {
        return Err(AutomataError::InvalidArgument(
            "2D hyper prior requires at least one state dimension".to_string(),
        ));
    }
    Ok(())
}

fn validate_prior_config(config: ParticlePriorConfig) -> AutomataResult<()> {
    if config.min_particles == 0 || config.max_particles < config.min_particles {
        return Err(AutomataError::InvalidArgument(format!(
            "invalid particle prior range {}..{}",
            config.min_particles, config.max_particles
        )));
    }
    if !config.min_seed_scale.is_finite()
        || !config.max_seed_scale.is_finite()
        || config.min_seed_scale <= 0.0
        || config.max_seed_scale < config.min_seed_scale
    {
        return Err(AutomataError::InvalidArgument(format!(
            "invalid seed scale range {}..{}",
            config.min_seed_scale, config.max_seed_scale
        )));
    }
    Ok(())
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
