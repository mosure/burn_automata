use serde::{Deserialize, Serialize};

use burn_automata_kernels::{HashGridConfig, HashGridMode};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    #[default]
    Bpk,
    Safetensors,
    Json,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutomataPreset {
    #[default]
    Growing2d,
    Texture2d,
    Growing3dGs,
    PointMnist,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum EquivarianceMode {
    None,
    ParticleDensity,
    #[default]
    ParticleDensityAndScale,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NpaConfig {
    pub spatial_dims: usize,
    pub state_dims: usize,
    pub hidden_dims: usize,
    pub eps0: f32,
    pub alpha: f32,
    pub density_grad: bool,
    pub state_grad: bool,
    pub log_norm_grad: bool,
    pub log_norm_density_grad: bool,
    pub stopgrad_pos: bool,
    pub stopgrad_state: bool,
    #[serde(default)]
    pub equivariance: EquivarianceMode,
    #[serde(default)]
    pub position_features: bool,
    /// Model-specific features supplied by a caller after the canonical NPA
    /// perception row. Core NPA presets keep this at zero.
    #[serde(default)]
    pub auxiliary_input_dims: usize,
    pub decoder_dims: Option<usize>,
    pub output_dims: Option<usize>,
}

impl NpaConfig {
    pub fn growing_2d() -> Self {
        Self {
            spatial_dims: 2,
            state_dims: 16,
            hidden_dims: 128,
            eps0: 0.1,
            alpha: 0.5,
            density_grad: true,
            state_grad: true,
            log_norm_grad: true,
            log_norm_density_grad: true,
            stopgrad_pos: true,
            stopgrad_state: false,
            equivariance: EquivarianceMode::ParticleDensityAndScale,
            position_features: false,
            auxiliary_input_dims: 0,
            decoder_dims: None,
            output_dims: None,
        }
    }

    pub fn texture_2d() -> Self {
        Self {
            eps0: 0.2,
            ..Self::growing_2d()
        }
    }

    pub fn growing_3dgs() -> Self {
        Self {
            spatial_dims: 3,
            state_dims: 24,
            hidden_dims: 256,
            eps0: 0.1,
            alpha: 1.0,
            density_grad: true,
            state_grad: true,
            log_norm_grad: true,
            log_norm_density_grad: true,
            stopgrad_pos: true,
            stopgrad_state: false,
            equivariance: EquivarianceMode::ParticleDensityAndScale,
            position_features: false,
            auxiliary_input_dims: 0,
            decoder_dims: Some(256),
            output_dims: Some(20),
        }
    }

    pub fn point_mnist() -> Self {
        Self {
            spatial_dims: 2,
            state_dims: 16,
            hidden_dims: 256,
            eps0: 0.1,
            alpha: 1.0,
            density_grad: true,
            state_grad: true,
            log_norm_grad: true,
            log_norm_density_grad: true,
            stopgrad_pos: true,
            stopgrad_state: false,
            equivariance: EquivarianceMode::ParticleDensityAndScale,
            position_features: false,
            auxiliary_input_dims: 0,
            decoder_dims: Some(256),
            output_dims: Some(10),
        }
    }

    pub fn torus_field_3dgs() -> Self {
        Self {
            position_features: true,
            ..Self::growing_3dgs()
        }
    }

    pub fn perception_dims(&self) -> usize {
        self.state_dims * 2
            + usize::from(self.state_grad) * self.state_dims * self.spatial_dims
            + usize::from(self.density_grad) * self.spatial_dims
            + usize::from(self.position_features) * self.spatial_dims
            + self.auxiliary_input_dims
    }

    pub fn update_dims(&self) -> usize {
        self.spatial_dims + self.state_dims
    }

    pub fn scale_equivariant(&self) -> bool {
        self.equivariance == EquivarianceMode::ParticleDensityAndScale
    }

    pub fn particle_density_equivariant(&self) -> bool {
        matches!(
            self.equivariance,
            EquivarianceMode::ParticleDensity | EquivarianceMode::ParticleDensityAndScale
        )
    }

    pub fn motion_eps(&self, grid_eps: f32) -> f32 {
        if self.scale_equivariant() {
            grid_eps
        } else {
            self.eps0
        }
    }

    pub fn hashgrid_for_seed_scale(
        &self,
        hashgrid: &HashGridConfig,
        seed_scale: f32,
        reference_seed_scale: f32,
    ) -> HashGridConfig {
        let mut scaled = hashgrid.clone();
        if self.scale_equivariant()
            && hashgrid.mode == HashGridMode::Particle
            && seed_scale.is_finite()
            && seed_scale > 0.0
            && reference_seed_scale.is_finite()
            && reference_seed_scale > 0.0
        {
            scaled.eps = (hashgrid.eps * seed_scale / reference_seed_scale).max(f32::MIN_POSITIVE);
        }
        scaled
    }

    pub fn for_preset(preset: AutomataPreset) -> (Self, HashGridConfig) {
        match preset {
            AutomataPreset::Growing2d => (Self::growing_2d(), HashGridConfig::growing_2d()),
            AutomataPreset::Texture2d => (Self::texture_2d(), HashGridConfig::texture_2d()),
            AutomataPreset::Growing3dGs => (Self::growing_3dgs(), HashGridConfig::growing_3dgs()),
            AutomataPreset::PointMnist => (Self::point_mnist(), HashGridConfig::growing_2d()),
        }
    }

    pub fn seed_scale_for_preset(preset: AutomataPreset) -> f32 {
        match preset {
            AutomataPreset::Growing2d => 0.2,
            AutomataPreset::Texture2d => 1.0,
            AutomataPreset::Growing3dGs => 1.0,
            AutomataPreset::PointMnist => 1.0,
        }
    }
}

impl Default for NpaConfig {
    fn default() -> Self {
        Self::growing_2d()
    }
}
