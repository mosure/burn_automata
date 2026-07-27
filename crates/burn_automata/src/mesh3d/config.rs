use serde::{Deserialize, Serialize};

use crate::{AdamWConfig, EquivarianceMode, NpaConfig, TrainingRunReport};

pub fn mesh3d_model_config(hidden_dims: usize) -> NpaConfig {
    NpaConfig {
        hidden_dims: hidden_dims.max(64),
        position_features: true,
        equivariance: EquivarianceMode::ParticleDensityAndScale,
        ..NpaConfig::growing_3dgs()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Mesh3dTrainingConfig {
    pub scale: f32,
    pub hidden_dims: usize,
    pub dataset_particles: usize,
    pub dataset_trajectories: usize,
    pub teacher_rollout_max_steps: usize,
    pub dataset_refreshes: usize,
    pub replay_accumulate: bool,
    pub steps: usize,
    pub report_interval: usize,
    pub seed: u64,
    pub near_surface_fraction: f32,
    pub surface_fraction: f32,
    pub surface_erasure_fraction: f32,
    pub deployment_surface_fraction: f32,
    pub deployment_damage_fraction: f32,
    pub max_motion_per_step: f32,
    pub normal_gain: f32,
    pub signed_distance_gain: f32,
    pub opacity_gain: f32,
    pub color_gain: f32,
    pub optimizer: AdamWConfig,
    pub evaluation: Mesh3dEvaluationConfig,
}

impl Default for Mesh3dTrainingConfig {
    fn default() -> Self {
        Self {
            scale: 0.72,
            hidden_dims: 256,
            dataset_particles: 4096,
            dataset_trajectories: 16,
            teacher_rollout_max_steps: 32,
            dataset_refreshes: 1,
            replay_accumulate: false,
            steps: 500,
            report_interval: 100,
            seed: 42,
            near_surface_fraction: 0.35,
            surface_fraction: 0.15,
            surface_erasure_fraction: 0.5,
            deployment_surface_fraction: 0.75,
            deployment_damage_fraction: 0.8,
            max_motion_per_step: 0.065,
            normal_gain: 0.35,
            signed_distance_gain: 0.35,
            opacity_gain: 0.25,
            color_gain: 0.35,
            optimizer: AdamWConfig {
                learning_rate: 1.0e-3,
                weight_decay: 1.0e-6,
                grad_clip_norm: 1.0,
                ..AdamWConfig::default()
            },
            evaluation: Mesh3dEvaluationConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Mesh3dEvaluationConfig {
    pub particle_count: usize,
    pub seed_scale: f32,
    pub rollout_steps: Vec<usize>,
    pub seeds: Vec<u64>,
    pub target_samples: usize,
    pub render_image_size: usize,
    pub render_target_samples: usize,
    pub damage_radius: f32,
    pub damage_displacement: f32,
    pub recovery_min_steps: usize,
    pub max_mean_surface_distance: f32,
    pub max_p95_surface_distance: f32,
    pub max_mean_coverage_distance: f32,
    pub min_coverage_fraction: f32,
    pub min_density_psnr_db: f32,
    pub min_color_psnr_db: f32,
    pub min_damage_region_color_psnr_db: f32,
    pub max_long_horizon_drift: f32,
}

impl Default for Mesh3dEvaluationConfig {
    fn default() -> Self {
        Self {
            particle_count: 16_384,
            seed_scale: 0.72,
            rollout_steps: vec![32, 96, 256],
            seeds: vec![42, 97, 131],
            target_samples: 8_192,
            render_image_size: 128,
            render_target_samples: 16_384,
            damage_radius: 0.22,
            damage_displacement: 0.0,
            recovery_min_steps: 32,
            max_mean_surface_distance: 0.018,
            max_p95_surface_distance: 0.045,
            max_mean_coverage_distance: 0.035,
            min_coverage_fraction: 0.985,
            min_density_psnr_db: 24.0,
            min_color_psnr_db: 24.0,
            min_damage_region_color_psnr_db: 24.0,
            max_long_horizon_drift: 0.012,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mesh3dTrainingReport {
    pub training: TrainingRunReport,
    pub stages: Vec<Mesh3dTrainingStageReport>,
    pub selected_refresh: usize,
    pub dataset_rows: usize,
    pub training_rows_processed: u64,
    pub dataset_generation_seconds: f64,
    pub training_seconds: f64,
    pub rows_per_second: f64,
    pub quality: Mesh3dQualityReport,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mesh3dTrainingStageReport {
    pub refresh: usize,
    pub policy_horizon: usize,
    pub dataset_rows: usize,
    pub selection_steps: usize,
    pub selection_density_psnr_db: f32,
    pub selection_color_psnr_db: f32,
    pub selection_damage_region_color_psnr_db: f32,
    pub training: TrainingRunReport,
}

#[derive(Clone, Debug)]
pub struct Mesh3dTrainingProgress {
    pub step: usize,
    pub total_steps: usize,
    pub refresh: usize,
    pub refreshes: usize,
    pub policy_horizon: usize,
    pub dataset_rows: usize,
    pub loss: f32,
    pub grad_norm: f32,
    pub grad_scale: f32,
    pub model: crate::NpaModel,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mesh3dQualityReport {
    pub passed: bool,
    pub particle_count: usize,
    pub target_samples: usize,
    pub rollouts: Vec<Mesh3dRolloutReport>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mesh3dRolloutReport {
    pub initialization: Mesh3dInitializationMode,
    pub required_for_quality: bool,
    pub seed: u64,
    pub steps: usize,
    pub finite: bool,
    pub mean_surface_distance: f32,
    pub p95_surface_distance: f32,
    pub max_surface_distance: f32,
    pub mean_coverage_distance: f32,
    pub coverage_fraction: f32,
    pub density_psnr_db: f32,
    pub color_psnr_db: f32,
    pub particle_color_psnr_db: f32,
    pub damage_region_color_psnr_db: f32,
    pub damage_region_mean_opacity: f32,
    pub damage_region_particle_fraction: f32,
    pub depth_psnr_db: f32,
    pub mean_normal_alignment: f32,
    pub mean_opacity: f32,
    pub drift_from_previous_horizon: Option<f32>,
    pub passed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mesh3dInitializationMode {
    UniformVolume,
    MeshSurface,
    MeshSurfaceDamaged,
}
