use serde::{Deserialize, Serialize};

use crate::{
    AutomataError, AutomataPreset, AutomataResult, BpkModelManifest, NpaConfig, NpaModel,
    ParticleSeed, RolloutConfig, RolloutTrace, SupervisedBatch,
    kernels::HashGridConfig,
    rollout::{run_rollout, seed_particles_scaled},
};

#[derive(Clone, Debug)]
pub struct AutomataPipeline {
    pub model: NpaModel,
    pub hashgrid: HashGridConfig,
    pub seed_scale: f32,
    pub seed_mode: ParticleSeed,
}

impl AutomataPipeline {
    pub fn for_preset(preset: AutomataPreset, model_seed: u64) -> Self {
        let (config, hashgrid) = NpaConfig::for_preset(preset);
        Self {
            model: NpaModel::seeded(config, model_seed),
            hashgrid,
            seed_scale: NpaConfig::seed_scale_for_preset(preset),
            seed_mode: ParticleSeed::UniformCircle,
        }
    }

    pub fn from_manifest(
        manifest: BpkModelManifest,
        seed_scale: f32,
        seed_mode: ParticleSeed,
    ) -> Self {
        let hashgrid = manifest.hashgrid.clone();
        Self {
            model: manifest.into_model(),
            hashgrid,
            seed_scale,
            seed_mode,
        }
    }

    pub fn validate(&self) -> AutomataResult<()> {
        self.model.validate()?;
        self.hashgrid.validate().map_err(AutomataError::from)?;
        if self.hashgrid.dim != self.model.config.spatial_dims {
            return Err(AutomataError::InvalidArgument(format!(
                "pipeline grid dim {} does not match model spatial dims {}",
                self.hashgrid.dim, self.model.config.spatial_dims
            )));
        }
        if !self.seed_scale.is_finite() || self.seed_scale <= 0.0 {
            return Err(AutomataError::InvalidArgument(format!(
                "seed_scale must be finite and positive, got {}",
                self.seed_scale
            )));
        }
        Ok(())
    }

    pub fn rollout_config(&self, particle_count: usize, steps: usize) -> RolloutConfig {
        RolloutConfig {
            particle_count,
            steps,
            seed_scale: self.seed_scale,
            ..RolloutConfig::default()
        }
    }

    pub fn seed_particles(
        &self,
        batch_size: usize,
        particle_count: usize,
        seed: u64,
    ) -> (Vec<[f32; 4]>, Vec<f32>) {
        seed_particles_scaled(
            batch_size,
            particle_count,
            self.model.config.state_dims,
            self.model.config.spatial_dims,
            seed,
            self.seed_mode,
            self.seed_scale,
        )
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FeatureBatchConfig {
    pub rows: usize,
    pub seed: u64,
    pub amplitude: f32,
}

impl Default for FeatureBatchConfig {
    fn default() -> Self {
        Self {
            rows: 4096,
            seed: 0,
            amplitude: 0.5,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct RolloutBatchConfig {
    pub max_rows: usize,
    pub dt: f32,
}

impl Default for RolloutBatchConfig {
    fn default() -> Self {
        Self {
            max_rows: 512,
            dt: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct RolloutSupervisionConfig {
    pub max_rows: usize,
    pub particle_count: usize,
    pub rollout_steps: usize,
    pub rollouts: usize,
    pub dt: f32,
    pub update_prob: f32,
    pub seed: u64,
    pub seed_scale: f32,
    pub seed_mode: ParticleSeed,
}

impl Default for RolloutSupervisionConfig {
    fn default() -> Self {
        Self {
            max_rows: 512,
            particle_count: 1024,
            rollout_steps: 16,
            rollouts: 1,
            dt: 1.0,
            update_prob: 1.0,
            seed: 42,
            seed_scale: 0.2,
            seed_mode: ParticleSeed::UniformCircle,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum SupervisedTarget<'a> {
    ZeroUpdate,
    Teacher(&'a NpaModel),
}

pub fn feature_supervised_batch(
    model: &NpaModel,
    target: SupervisedTarget<'_>,
    cfg: FeatureBatchConfig,
) -> AutomataResult<SupervisedBatch> {
    validate_feature_batch_config(cfg)?;
    validate_target(model, target)?;
    let feature_dims = model.config.perception_dims();
    let features = (0..cfg.rows * feature_dims)
        .map(|idx| {
            let phase = (idx as f32 + cfg.seed as f32 * 0.618_034) * 0.013;
            phase.sin() * cfg.amplitude
        })
        .collect::<Vec<_>>();
    target_batch_from_features(model, target, features)
}

pub fn rollout_supervised_batch(
    model: &NpaModel,
    grid: &HashGridConfig,
    trace: &RolloutTrace,
    target: SupervisedTarget<'_>,
    cfg: RolloutBatchConfig,
) -> AutomataResult<SupervisedBatch> {
    validate_target(model, target)?;
    validate_rollout_batch(model, grid, trace, cfg)?;
    let rows = trace.particle_count.min(cfg.max_rows);
    let positions = &trace.positions[..rows];
    let states = &trace.states[..rows * model.config.state_dims];
    let step = model.step_cpu(positions, states, 1, rows, grid, cfg.dt, None)?;
    target_batch_from_features(model, target, step.perception.features)
}

pub fn rollout_supervised_batch_from_model(
    model: &NpaModel,
    rollout_model: &NpaModel,
    grid: &HashGridConfig,
    target: SupervisedTarget<'_>,
    cfg: RolloutSupervisionConfig,
) -> AutomataResult<SupervisedBatch> {
    validate_rollout_supervision_config(cfg)?;
    validate_target(model, target)?;
    rollout_model.validate()?;
    if rollout_model.config.spatial_dims != model.config.spatial_dims
        || rollout_model.config.state_dims != model.config.state_dims
    {
        return Err(AutomataError::InvalidArgument(format!(
            "rollout model dims spatial/state {}:{} do not match student {}:{}",
            rollout_model.config.spatial_dims,
            rollout_model.config.state_dims,
            model.config.spatial_dims,
            model.config.state_dims
        )));
    }
    if grid.dim != rollout_model.config.spatial_dims {
        return Err(AutomataError::InvalidArgument(format!(
            "grid dim {} does not match rollout model spatial dims {}",
            grid.dim, rollout_model.config.spatial_dims
        )));
    }

    let mut features = Vec::new();
    let mut target_update = Vec::new();
    let mut remaining_rows = cfg.max_rows;
    for rollout_idx in 0..cfg.rollouts {
        if remaining_rows == 0 {
            break;
        }
        let trace = run_rollout(
            rollout_model,
            grid,
            &RolloutConfig {
                particle_count: cfg.particle_count,
                steps: cfg.rollout_steps,
                dt: cfg.dt,
                update_prob: cfg.update_prob,
                seed: cfg
                    .seed
                    .wrapping_add((rollout_idx as u64).wrapping_mul(0x9e37_79b9)),
                seed_scale: cfg.seed_scale,
                ..RolloutConfig::default()
            },
            cfg.seed_mode,
        )?;
        let batch = rollout_supervised_batch(
            model,
            grid,
            &trace,
            target,
            RolloutBatchConfig {
                max_rows: remaining_rows,
                dt: cfg.dt,
            },
        )?;
        remaining_rows = remaining_rows.saturating_sub(
            batch
                .features
                .len()
                .checked_div(model.config.perception_dims())
                .unwrap_or_default(),
        );
        features.extend(batch.features);
        target_update.extend(batch.target_update);
    }

    if features.is_empty() {
        return Err(AutomataError::InvalidArgument(
            "rollout supervision produced no rows".to_string(),
        ));
    }
    Ok(SupervisedBatch {
        features,
        target_update,
    })
}

fn target_batch_from_features(
    model: &NpaModel,
    target: SupervisedTarget<'_>,
    features: Vec<f32>,
) -> AutomataResult<SupervisedBatch> {
    let rows = features.len() / model.config.perception_dims();
    let target_update = match target {
        SupervisedTarget::ZeroUpdate => vec![0.0; rows * model.config.update_dims()],
        SupervisedTarget::Teacher(teacher) => teacher.forward_update_from_features(&features)?,
    };
    Ok(SupervisedBatch {
        features,
        target_update,
    })
}

fn validate_rollout_supervision_config(cfg: RolloutSupervisionConfig) -> AutomataResult<()> {
    if cfg.max_rows == 0 {
        return Err(AutomataError::InvalidArgument(
            "rollout supervision max_rows must be non-zero".to_string(),
        ));
    }
    if cfg.particle_count == 0 {
        return Err(AutomataError::InvalidArgument(
            "rollout supervision particle_count must be non-zero".to_string(),
        ));
    }
    if cfg.rollouts == 0 {
        return Err(AutomataError::InvalidArgument(
            "rollout supervision rollouts must be non-zero".to_string(),
        ));
    }
    if !cfg.dt.is_finite() {
        return Err(AutomataError::InvalidArgument(format!(
            "rollout supervision dt must be finite, got {}",
            cfg.dt
        )));
    }
    if !cfg.update_prob.is_finite() || !(0.0..=1.0).contains(&cfg.update_prob) {
        return Err(AutomataError::InvalidArgument(format!(
            "rollout supervision update_prob must be in [0, 1], got {}",
            cfg.update_prob
        )));
    }
    if !cfg.seed_scale.is_finite() || cfg.seed_scale <= 0.0 {
        return Err(AutomataError::InvalidArgument(format!(
            "rollout supervision seed_scale must be finite and positive, got {}",
            cfg.seed_scale
        )));
    }
    Ok(())
}

fn validate_feature_batch_config(cfg: FeatureBatchConfig) -> AutomataResult<()> {
    if cfg.rows == 0 {
        return Err(AutomataError::InvalidArgument(
            "feature batch rows must be non-zero".to_string(),
        ));
    }
    if !cfg.amplitude.is_finite() {
        return Err(AutomataError::InvalidArgument(format!(
            "feature batch amplitude must be finite, got {}",
            cfg.amplitude
        )));
    }
    Ok(())
}

fn validate_target(model: &NpaModel, target: SupervisedTarget<'_>) -> AutomataResult<()> {
    model.validate()?;
    if let SupervisedTarget::Teacher(teacher) = target {
        teacher.validate()?;
        if teacher.config.perception_dims() != model.config.perception_dims()
            || teacher.config.update_dims() != model.config.update_dims()
        {
            return Err(AutomataError::InvalidArgument(format!(
                "teacher dims perception/update {}:{} do not match student {}:{}",
                teacher.config.perception_dims(),
                teacher.config.update_dims(),
                model.config.perception_dims(),
                model.config.update_dims()
            )));
        }
    }
    Ok(())
}

fn validate_rollout_batch(
    model: &NpaModel,
    grid: &HashGridConfig,
    trace: &RolloutTrace,
    cfg: RolloutBatchConfig,
) -> AutomataResult<()> {
    if cfg.max_rows == 0 {
        return Err(AutomataError::InvalidArgument(
            "rollout batch max_rows must be non-zero".to_string(),
        ));
    }
    if !cfg.dt.is_finite() {
        return Err(AutomataError::InvalidArgument(format!(
            "rollout batch dt must be finite, got {}",
            cfg.dt
        )));
    }
    if grid.dim != model.config.spatial_dims {
        return Err(AutomataError::InvalidArgument(format!(
            "grid dim {} does not match model spatial dims {}",
            grid.dim, model.config.spatial_dims
        )));
    }
    if trace.state_dims != model.config.state_dims {
        return Err(AutomataError::InvalidArgument(format!(
            "trace state dims {} do not match model state dims {}",
            trace.state_dims, model.config.state_dims
        )));
    }
    if trace.positions.len() < trace.particle_count
        || trace.states.len() < trace.particle_count * trace.state_dims
    {
        return Err(AutomataError::InvalidArgument(
            "trace does not contain enough positions/states for one rollout batch".to_string(),
        ));
    }
    Ok(())
}
