use std::{
    fs,
    io::{BufReader, BufWriter, Write},
    path::Path,
};

use serde::{Deserialize, Serialize};

use super::E2eIdentitySampler;
use crate::{AutomataError, AutomataResult};

pub(crate) const E2E_TRAINING_CHECKPOINT_VERSION: u32 = 1;
pub(crate) const TARGET2D_TRAINING_CHECKPOINT_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct E2eTensorSnapshot {
    pub(crate) name: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) values: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct E2eParticlePoolSnapshot {
    pub(crate) positions: E2eTensorSnapshot,
    pub(crate) states: E2eTensorSnapshot,
    pub(crate) slot_examples: Vec<Option<(usize, usize)>>,
    pub(crate) next_evict: usize,
    pub(crate) slots_per_example: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Target2dParticlePoolSnapshot {
    pub(crate) positions: E2eTensorSnapshot,
    pub(crate) states: E2eTensorSnapshot,
    #[serde(default)]
    pub(crate) ages: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Target2dTrainingCheckpoint {
    pub(crate) version: u32,
    pub(crate) backend: String,
    pub(crate) model_sha256: String,
    pub(crate) completed_step: usize,
    pub(crate) optimizer_step: usize,
    pub(crate) model_count: usize,
    pub(crate) rollout_particles: usize,
    pub(crate) optimizer_tensors: Vec<E2eTensorSnapshot>,
    pub(crate) particle_pool: Option<Target2dParticlePoolSnapshot>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct E2eTrainingCheckpoint {
    pub(crate) version: u32,
    pub(crate) backend: String,
    #[serde(default)]
    pub(crate) contract_sha256: String,
    #[serde(default)]
    pub(crate) shared_base_sha256: String,
    #[serde(default)]
    pub(crate) hyper_sha256: String,
    pub(crate) completed_step: usize,
    pub(crate) train_examples: usize,
    pub(crate) rollout_particles: usize,
    pub(crate) rollout_step_min: usize,
    pub(crate) rollout_steps: usize,
    pub(crate) rollouts_per_example: usize,
    pub(crate) base_optimizer_step: usize,
    pub(crate) generator_optimizer_step: usize,
    #[serde(default)]
    pub(crate) row_flow_optimizer_step: Option<usize>,
    #[serde(default)]
    pub(crate) amortization_optimizer_step: Option<usize>,
    #[serde(default)]
    pub(crate) sample_id_optimizer_steps: Vec<usize>,
    #[serde(default)]
    pub(crate) amortization_identity_optimizer_steps: Vec<usize>,
    pub(crate) optimizer_tensors: Vec<E2eTensorSnapshot>,
    pub(crate) sampler: E2eIdentitySampler,
    pub(crate) seed_trajectory_counts: Vec<usize>,
    pub(crate) pending_batches: Vec<Vec<usize>>,
    pub(crate) particle_pool: Option<E2eParticlePoolSnapshot>,
}

impl E2eTrainingCheckpoint {
    pub(crate) fn write_atomic(&self, path: &Path) -> AutomataResult<()> {
        let parent = path.parent().ok_or_else(|| {
            AutomataError::InvalidArgument(format!(
                "training checkpoint path {} has no parent",
                path.display()
            ))
        })?;
        fs::create_dir_all(parent)?;
        let temp = path.with_extension("mpk.tmp");
        let file = fs::File::create(&temp)?;
        let mut writer = BufWriter::new(file);
        rmp_serde::encode::write_named(&mut writer, self).map_err(|error| {
            AutomataError::InvalidArgument(format!("training checkpoint encoding failed: {error}"))
        })?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        fs::rename(&temp, path)?;
        Ok(())
    }

    pub(crate) fn read(path: &Path) -> AutomataResult<Self> {
        let file = fs::File::open(path)?;
        let checkpoint: Self = rmp_serde::from_read(BufReader::new(file)).map_err(|error| {
            AutomataError::InvalidArgument(format!(
                "training checkpoint decoding failed for {}: {error}",
                path.display()
            ))
        })?;
        if checkpoint.version != E2E_TRAINING_CHECKPOINT_VERSION {
            return Err(AutomataError::InvalidArgument(format!(
                "unsupported training checkpoint version {}; expected {}",
                checkpoint.version, E2E_TRAINING_CHECKPOINT_VERSION
            )));
        }
        Ok(checkpoint)
    }

    pub(crate) fn tensor(&self, name: &str) -> AutomataResult<&E2eTensorSnapshot> {
        self.optimizer_tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| {
                AutomataError::InvalidArgument(format!(
                    "training checkpoint is missing tensor {name}"
                ))
            })
    }

    pub(crate) fn tensor_optional(&self, name: &str) -> Option<&E2eTensorSnapshot> {
        self.optimizer_tensors
            .iter()
            .find(|tensor| tensor.name == name)
    }
}

impl Target2dTrainingCheckpoint {
    pub(crate) fn write_atomic(&self, path: &Path) -> AutomataResult<()> {
        let parent = path.parent().ok_or_else(|| {
            AutomataError::InvalidArgument(format!(
                "Target2D training checkpoint path {} has no parent",
                path.display()
            ))
        })?;
        fs::create_dir_all(parent)?;
        let temp = path.with_extension("mpk.tmp");
        let file = fs::File::create(&temp)?;
        let mut writer = BufWriter::new(file);
        rmp_serde::encode::write_named(&mut writer, self).map_err(|error| {
            AutomataError::InvalidArgument(format!(
                "Target2D training checkpoint encoding failed: {error}"
            ))
        })?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        fs::rename(&temp, path)?;
        Ok(())
    }

    pub(crate) fn read(path: &Path) -> AutomataResult<Self> {
        let file = fs::File::open(path)?;
        let checkpoint: Self = rmp_serde::from_read(BufReader::new(file)).map_err(|error| {
            AutomataError::InvalidArgument(format!(
                "Target2D training checkpoint decoding failed for {}: {error}",
                path.display()
            ))
        })?;
        if checkpoint.version != TARGET2D_TRAINING_CHECKPOINT_VERSION {
            return Err(AutomataError::InvalidArgument(format!(
                "unsupported Target2D training checkpoint version {}; expected {}",
                checkpoint.version, TARGET2D_TRAINING_CHECKPOINT_VERSION
            )));
        }
        Ok(checkpoint)
    }

    pub(crate) fn tensor(&self, name: &str) -> AutomataResult<&E2eTensorSnapshot> {
        self.optimizer_tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| {
                AutomataError::InvalidArgument(format!(
                    "Target2D training checkpoint is missing tensor {name}"
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;

    #[test]
    fn binary_training_checkpoint_round_trips() {
        let mut rng = StdRng::seed_from_u64(5);
        let sampler = E2eIdentitySampler::new(4, 2, 0.75, 0.95, 0.5, 4.0, &mut rng);
        let checkpoint = E2eTrainingCheckpoint {
            version: E2E_TRAINING_CHECKPOINT_VERSION,
            backend: "test".to_string(),
            contract_sha256: "test-contract".to_string(),
            shared_base_sha256: "base-hash".to_string(),
            hyper_sha256: "hyper-hash".to_string(),
            completed_step: 7,
            train_examples: 4,
            rollout_particles: 8,
            rollout_step_min: 2,
            rollout_steps: 4,
            rollouts_per_example: 2,
            base_optimizer_step: 7,
            generator_optimizer_step: 7,
            row_flow_optimizer_step: None,
            amortization_optimizer_step: None,
            sample_id_optimizer_steps: vec![3, 4, 5, 6],
            amortization_identity_optimizer_steps: vec![1, 2, 3, 4],
            optimizer_tensors: vec![E2eTensorSnapshot {
                name: "base.w1.m".to_string(),
                shape: vec![2, 2],
                values: vec![1.0, 2.0, 3.0, 4.0],
            }],
            sampler,
            seed_trajectory_counts: vec![1, 2, 3, 4],
            pending_batches: Vec::new(),
            particle_pool: None,
        };
        let path = std::env::temp_dir().join(format!(
            "burn-automata-e2e-checkpoint-{}-{}.mpk",
            std::process::id(),
            5
        ));
        checkpoint.write_atomic(&path).unwrap();
        let restored = E2eTrainingCheckpoint::read(&path).unwrap();
        fs::remove_file(path).ok();
        assert_eq!(restored.completed_step, 7);
        assert_eq!(restored.shared_base_sha256, "base-hash");
        assert_eq!(restored.hyper_sha256, "hyper-hash");
        assert_eq!(restored.sample_id_optimizer_steps, [3, 4, 5, 6]);
        assert_eq!(restored.amortization_identity_optimizer_steps, [1, 2, 3, 4]);
        assert_eq!(
            restored.tensor("base.w1.m").unwrap().values,
            [1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn binary_target2d_training_checkpoint_round_trips() {
        let checkpoint = Target2dTrainingCheckpoint {
            version: TARGET2D_TRAINING_CHECKPOINT_VERSION,
            backend: "test".to_string(),
            model_sha256: "model-hash".to_string(),
            completed_step: 17,
            optimizer_step: 17,
            model_count: 1,
            rollout_particles: 4096,
            optimizer_tensors: vec![E2eTensorSnapshot {
                name: "target2d.w1.m".to_string(),
                shape: vec![1, 2, 2],
                values: vec![1.0, 2.0, 3.0, 4.0],
            }],
            particle_pool: Some(Target2dParticlePoolSnapshot {
                positions: E2eTensorSnapshot {
                    name: "target2d.pool.positions".to_string(),
                    shape: vec![1, 1, 2],
                    values: vec![0.25, -0.25],
                },
                states: E2eTensorSnapshot {
                    name: "target2d.pool.states".to_string(),
                    shape: vec![1, 1, 1],
                    values: vec![0.5],
                },
                ages: vec![17],
            }),
        };
        let root = std::env::temp_dir().join(format!(
            "burn_automata_target2d_checkpoint_{}_{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        let path = root.join("state.mpk");
        checkpoint.write_atomic(&path).unwrap();
        let restored = Target2dTrainingCheckpoint::read(&path).unwrap();
        assert_eq!(restored.completed_step, 17);
        assert_eq!(restored.optimizer_step, 17);
        assert_eq!(
            restored.tensor("target2d.w1.m").unwrap().values,
            vec![1.0, 2.0, 3.0, 4.0]
        );
        let restored_pool = restored.particle_pool.unwrap();
        assert_eq!(restored_pool.positions.values, vec![0.25, -0.25]);
        assert_eq!(restored_pool.ages, vec![17]);
        std::fs::remove_dir_all(root).unwrap();
    }
}
