use std::time::Instant;

use rand::{SeedableRng, rngs::StdRng, seq::index};

use super::{AdaptiveRuleDistillationConfig, AdaptiveRuleTrainingBatch};
use crate::{
    AutomataError, AutomataResult, NpaModel, ParticleSeed,
    adaptive::AdaptiveNpaConfig,
    rollout::{seed_particles_scaled, stochastic_mask},
};
use burn_automata_kernels::{HashGridConfig, adaptive_perceive, euler_step};

pub fn adaptive_rule_distillation_batch(
    teacher: &NpaModel,
    teacher_grid: &HashGridConfig,
    adaptive: &AdaptiveNpaConfig,
    config: AdaptiveRuleDistillationConfig,
) -> AutomataResult<AdaptiveRuleTrainingBatch> {
    adaptive_rule_training_batch(teacher, None, teacher_grid, adaptive, config)
}

pub fn adaptive_rule_on_policy_batch(
    teacher: &NpaModel,
    student: &NpaModel,
    teacher_grid: &HashGridConfig,
    adaptive: &AdaptiveNpaConfig,
    config: AdaptiveRuleDistillationConfig,
) -> AutomataResult<AdaptiveRuleTrainingBatch> {
    adaptive_rule_training_batch(teacher, Some(student), teacher_grid, adaptive, config)
}

fn adaptive_rule_training_batch(
    teacher: &NpaModel,
    rollout_student: Option<&NpaModel>,
    teacher_grid: &HashGridConfig,
    adaptive: &AdaptiveNpaConfig,
    config: AdaptiveRuleDistillationConfig,
) -> AutomataResult<AdaptiveRuleTrainingBatch> {
    validate(teacher, teacher_grid, adaptive, config)?;
    if let Some(student) = rollout_student {
        student.validate()?;
        if student.config != teacher.config {
            return Err(AutomataError::InvalidArgument(
                "adaptive on-policy student config differs from teacher".to_string(),
            ));
        }
    }
    let started = Instant::now();
    let snapshot_steps = snapshot_steps(config.rollout_steps, config.temporal_samples);
    let rows_per_snapshot = config.rows_per_snapshot.min(config.particle_count);
    let rows = config.rollouts * snapshot_steps.len() * rows_per_snapshot;
    let input_dims = teacher.config.perception_dims();
    let output_dims = teacher.config.update_dims();
    let mut features = Vec::with_capacity(rows * input_dims);
    let mut target_update = Vec::with_capacity(rows * output_dims);
    let represented_measure =
        vec![config.total_measure / config.particle_count as f32; config.particle_count];
    let bandwidth = vec![config.bandwidth; config.particle_count];

    for rollout_index in 0..config.rollouts {
        let rollout_seed = config
            .seed
            .wrapping_add((rollout_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let (mut positions, mut states) = seed_particles_scaled(
            1,
            config.particle_count,
            teacher.config.state_dims,
            teacher.config.spatial_dims,
            rollout_seed,
            ParticleSeed::UniformCircle,
            config.seed_scale,
        );
        let mut mask_rng = StdRng::seed_from_u64(rollout_seed ^ 0x5eed);
        let mut sample_rng = StdRng::seed_from_u64(rollout_seed ^ 0xa4a4_5eed);
        let mut snapshot_cursor = 0;
        for step_index in 0..=config.rollout_steps {
            let teacher_step = teacher.step_cpu(
                &positions,
                &states,
                1,
                config.particle_count,
                teacher_grid,
                config.dt,
                None,
            )?;
            let is_snapshot = snapshot_steps.get(snapshot_cursor).copied() == Some(step_index);
            let adaptive_perception = if rollout_student.is_some() || is_snapshot {
                Some(adaptive_perceive(
                    &positions,
                    &states,
                    &represented_measure,
                    &bandwidth,
                    1,
                    config.particle_count,
                    teacher.config.state_dims,
                    adaptive.perception,
                )?)
            } else {
                None
            };
            if is_snapshot {
                let perception = adaptive_perception
                    .as_ref()
                    .expect("adaptive perception is built for snapshots");
                let teacher_update =
                    teacher.forward_update_from_features(&teacher_step.perception.features)?;
                let selected =
                    index::sample(&mut sample_rng, config.particle_count, rows_per_snapshot);
                for row in selected.iter() {
                    features.extend_from_slice(
                        &perception.features[row * input_dims..(row + 1) * input_dims],
                    );
                    target_update.extend_from_slice(
                        &teacher_update[row * output_dims..(row + 1) * output_dims],
                    );
                }
                snapshot_cursor += 1;
            }
            if step_index == config.rollout_steps {
                break;
            }
            let update_mask =
                stochastic_mask(config.particle_count, config.update_prob, &mut mask_rng);
            let (dx, ds) = if let Some(student) = rollout_student {
                student.forward_from_features_with_eps(
                    &adaptive_perception
                        .as_ref()
                        .expect("adaptive perception is built for student rollout")
                        .features,
                    config.bandwidth,
                )?
            } else {
                (teacher_step.dx, teacher_step.ds)
            };
            (positions, states) = euler_step(
                &positions,
                &states,
                &dx,
                &ds,
                1,
                config.particle_count,
                teacher.config.state_dims,
                teacher_grid,
                config.dt,
                Some(&update_mask),
            )?;
        }
    }
    let batch = AdaptiveRuleTrainingBatch {
        rows: features.len() / input_dims,
        features,
        target_update,
        generation_elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
    };
    batch.validate(input_dims, output_dims)?;
    Ok(batch)
}

fn validate(
    teacher: &NpaModel,
    teacher_grid: &HashGridConfig,
    adaptive: &AdaptiveNpaConfig,
    config: AdaptiveRuleDistillationConfig,
) -> AutomataResult<()> {
    teacher.validate()?;
    teacher_grid.validate().map_err(AutomataError::from)?;
    adaptive.validate()?;
    if teacher.config.spatial_dims != adaptive.spatial_dims
        || teacher_grid.dim != teacher.config.spatial_dims
        || adaptive.perception.feature_dims(teacher.config.state_dims)
            != teacher.config.perception_dims()
        || config.particle_count == 0
        || config.rollouts == 0
        || config.temporal_samples == 0
        || config.rows_per_snapshot == 0
        || config.validation_rollouts == 0
        || config.steps == 0
        || config.report_interval == 0
        || !config.dt.is_finite()
        || config.dt <= 0.0
        || !config.update_prob.is_finite()
        || !(0.0..=1.0).contains(&config.update_prob)
        || !config.seed_scale.is_finite()
        || config.seed_scale <= 0.0
        || !config.total_measure.is_finite()
        || config.total_measure <= 0.0
        || !config.bandwidth.is_finite()
        || config.bandwidth < adaptive.perception.min_bandwidth
        || config.bandwidth > adaptive.perception.max_bandwidth
    {
        return Err(AutomataError::InvalidArgument(
            "invalid adaptive rule distillation config".to_string(),
        ));
    }
    Ok(())
}

fn snapshot_steps(rollout_steps: usize, temporal_samples: usize) -> Vec<usize> {
    if temporal_samples == 1 || rollout_steps == 0 {
        return vec![rollout_steps];
    }
    let mut steps = Vec::with_capacity(temporal_samples);
    for sample_index in 0..temporal_samples {
        let step = sample_index * rollout_steps / (temporal_samples - 1);
        if steps.last().copied() != Some(step) {
            steps.push(step);
        }
    }
    steps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NpaConfig, adaptive::AdaptiveNpaConfig, upstream_growing_2d_hashgrid};

    #[test]
    fn rule_distillation_batch_samples_whole_teacher_trajectories() {
        let teacher = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 7);
        let config = AdaptiveRuleDistillationConfig {
            particle_count: 32,
            rollout_steps: 2,
            rollouts: 2,
            temporal_samples: 3,
            rows_per_snapshot: 8,
            validation_rollouts: 1,
            steps: 1,
            report_interval: 1,
            ..AdaptiveRuleDistillationConfig::default()
        };
        let batch = adaptive_rule_distillation_batch(
            &teacher,
            &upstream_growing_2d_hashgrid(),
            &AdaptiveNpaConfig::growing_2d(),
            config,
        )
        .unwrap();
        assert_eq!(batch.rows, 2 * 3 * 8);
        assert_eq!(
            batch.features.len(),
            batch.rows * teacher.config.perception_dims()
        );
        assert_eq!(
            batch.target_update.len(),
            batch.rows * teacher.config.update_dims()
        );
    }
}
