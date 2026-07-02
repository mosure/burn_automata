#![allow(clippy::too_many_arguments)]

use super::*;

#[derive(Clone, Copy)]
pub(crate) struct MeshFieldRolloutBatchConfig {
    pub(crate) max_rows: usize,
    pub(crate) particle_count: usize,
    pub(crate) rollout_steps: usize,
    pub(crate) rollouts: usize,
    pub(crate) temporal_samples: usize,
    pub(crate) seed: u64,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) motion_gain: f32,
    pub(crate) max_update_norm: f32,
    pub(crate) coverage_gain: f32,
    pub(crate) coverage_samples: usize,
    pub(crate) coverage_mode: CoverageUpdateModeArg,
    pub(crate) coverage_softness: f32,
    pub(crate) coverage_repulsion_gain: f32,
    pub(crate) coverage_gap_gain: f32,
    pub(crate) coverage_repulsion_radius: f32,
    pub(crate) coverage_normal_weight: f32,
    pub(crate) extent_gain: f32,
    pub(crate) color_gain: f32,
    pub(crate) aux_state_gain: f32,
    pub(crate) opacity_gain: f32,
    pub(crate) front_opacity_gain: f32,
    pub(crate) front_radius: f32,
    pub(crate) front_max_opacity_update: f32,
    pub(crate) front_motion_gate: bool,
    pub(crate) preserve_opacity_update: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct MeshLocalTrainingConfig {
    pub(crate) max_rows: usize,
    pub(crate) particle_count: usize,
    pub(crate) rollout_steps: usize,
    pub(crate) rollouts: usize,
    pub(crate) temporal_samples: usize,
    pub(crate) training_rounds: usize,
    pub(crate) total_steps: usize,
    pub(crate) seed: u64,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) motion_gain: f32,
    pub(crate) max_update_norm: f32,
    pub(crate) coverage_gain: f32,
    pub(crate) coverage_samples: usize,
    pub(crate) coverage_mode: CoverageUpdateModeArg,
    pub(crate) coverage_softness: f32,
    pub(crate) coverage_repulsion_gain: f32,
    pub(crate) coverage_gap_gain: f32,
    pub(crate) coverage_repulsion_radius: f32,
    pub(crate) coverage_normal_weight: f32,
    pub(crate) extent_gain: f32,
    pub(crate) color_gain: f32,
    pub(crate) aux_state_gain: f32,
    pub(crate) opacity_gain: f32,
    pub(crate) front_opacity_gain: f32,
    pub(crate) front_radius: f32,
    pub(crate) front_max_opacity_update: f32,
    pub(crate) front_motion_gate: bool,
    pub(crate) preserve_opacity_update: bool,
    pub(crate) sgd: SgdConfig,
}

pub(crate) fn merge_supervised_batches(
    mut lhs: SupervisedBatch,
    rhs: SupervisedBatch,
) -> SupervisedBatch {
    lhs.features.extend(rhs.features);
    lhs.target_update.extend(rhs.target_update);
    lhs
}

pub(crate) fn run_refreshed_mesh_local_training(
    model: &mut NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: MeshLocalTrainingConfig,
) -> Result<TrainingRunReport, Box<dyn std::error::Error>> {
    if cfg.total_steps == 0 {
        return Err(std::io::Error::other("local mesh training requires at least one step").into());
    }
    let rounds = cfg.training_rounds.max(1);
    let mut history = Vec::new();
    let mut initial_loss = None;
    let mut final_loss = 0.0_f32;
    let mut best_loss = f32::MAX;
    let mut rows = cfg.max_rows;
    let mut steps_done = 0usize;

    for round in 0..rounds {
        if steps_done >= cfg.total_steps {
            break;
        }
        let remaining_steps = cfg.total_steps - steps_done;
        let rounds_left = rounds - round;
        let round_steps = remaining_steps.div_ceil(rounds_left).max(1);
        let batch = mesh_local_rollout_supervised_batch(
            model,
            grid,
            target,
            MeshFieldRolloutBatchConfig {
                max_rows: cfg.max_rows,
                particle_count: cfg.particle_count,
                rollout_steps: cfg.rollout_steps,
                rollouts: cfg.rollouts,
                temporal_samples: cfg.temporal_samples,
                seed: cfg
                    .seed
                    .wrapping_add((round as u64).wrapping_mul(0x51ed_f00d)),
                seed_scale: cfg.seed_scale,
                seed_mode: cfg.seed_mode,
                motion_gain: cfg.motion_gain,
                max_update_norm: cfg.max_update_norm,
                coverage_gain: cfg.coverage_gain,
                coverage_samples: cfg.coverage_samples,
                coverage_mode: cfg.coverage_mode,
                coverage_softness: cfg.coverage_softness,
                coverage_repulsion_gain: cfg.coverage_repulsion_gain,
                coverage_gap_gain: cfg.coverage_gap_gain,
                coverage_repulsion_radius: cfg.coverage_repulsion_radius,
                coverage_normal_weight: cfg.coverage_normal_weight,
                extent_gain: cfg.extent_gain,
                color_gain: cfg.color_gain,
                aux_state_gain: cfg.aux_state_gain,
                opacity_gain: cfg.opacity_gain,
                front_opacity_gain: cfg.front_opacity_gain,
                front_radius: cfg.front_radius,
                front_max_opacity_update: cfg.front_max_opacity_update,
                front_motion_gate: cfg.front_motion_gate,
                preserve_opacity_update: cfg.preserve_opacity_update,
            },
        )?;
        let report = run_supervised_training(
            model,
            &batch,
            TrainingRunConfig {
                steps: round_steps,
                report_interval: round_steps.max(1),
                sgd: cfg.sgd,
            },
        )?;
        initial_loss.get_or_insert(report.initial_loss);
        rows = report.rows;
        final_loss = report.final_loss;
        best_loss = best_loss.min(report.best_loss);
        for entry in report.history {
            history.push(TrainingHistoryEntry {
                step: steps_done + entry.step,
                loss: entry.loss,
                grad_norm: entry.grad_norm,
                grad_scale: entry.grad_scale,
            });
        }
        steps_done += round_steps;
    }

    Ok(TrainingRunReport {
        steps: steps_done,
        rows,
        initial_loss: initial_loss.unwrap_or(final_loss),
        final_loss,
        best_loss,
        history,
    })
}

pub(crate) fn mesh_field_rollout_supervised_batch(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: MeshFieldRolloutBatchConfig,
) -> Result<SupervisedBatch, Box<dyn std::error::Error>> {
    mesh_rollout_supervised_batch(model, grid, target, cfg, true)
}

pub(crate) fn mesh_local_rollout_supervised_batch(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: MeshFieldRolloutBatchConfig,
) -> Result<SupervisedBatch, Box<dyn std::error::Error>> {
    mesh_rollout_supervised_batch(model, grid, target, cfg, false)
}

pub(crate) fn mesh_rollout_supervised_batch(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: MeshFieldRolloutBatchConfig,
    require_position_features: bool,
) -> Result<SupervisedBatch, Box<dyn std::error::Error>> {
    if !model.config.position_features {
        if require_position_features {
            return Err(std::io::Error::other(
                "mesh field rollout rows require position_features=true",
            )
            .into());
        }
    } else if !require_position_features {
        return Err(std::io::Error::other(
            "mesh local rollout rows require position_features=false",
        )
        .into());
    }
    if cfg.max_rows == 0 || cfg.particle_count == 0 || cfg.rollouts == 0 {
        return Err(std::io::Error::other("mesh rollout rows require non-zero sizes").into());
    }

    let mut features = Vec::new();
    let mut target_update = Vec::new();
    let mut remaining_rows = cfg.max_rows;
    let snapshot_steps = mesh_rollout_snapshot_steps(cfg.rollout_steps, cfg.temporal_samples);
    let total_snapshots = cfg.rollouts.saturating_mul(snapshot_steps.len()).max(1);
    let distributed_row_limit = cfg.max_rows.div_ceil(total_snapshots).max(1);
    for rollout_idx in 0..cfg.rollouts {
        if remaining_rows == 0 {
            break;
        }
        let (mut positions, mut states) = seed_particles_scaled(
            1,
            cfg.particle_count,
            model.config.state_dims,
            model.config.spatial_dims,
            cfg.seed
                .wrapping_add((rollout_idx as u64).wrapping_mul(0x9e37_79b9)),
            cfg.seed_mode,
            cfg.seed_scale,
        );
        let mut current_step = 0usize;
        for &snapshot_step in &snapshot_steps {
            while current_step < snapshot_step {
                let step =
                    model.step_cpu(&positions, &states, 1, cfg.particle_count, grid, 1.0, None)?;
                positions = step.next_positions;
                states = step.next_states;
                current_step += 1;
            }
            let row_limit = if snapshot_steps.len() == 1 {
                remaining_rows
            } else {
                remaining_rows.min(distributed_row_limit)
            };
            let rows = append_mesh_rollout_snapshot_rows(
                model,
                grid,
                target,
                &cfg,
                &positions,
                &states,
                row_limit,
                &mut features,
                &mut target_update,
            )?;
            remaining_rows = remaining_rows.saturating_sub(rows);
            if remaining_rows == 0 {
                break;
            }
        }
    }

    if features.is_empty() {
        return Err(std::io::Error::other("mesh rollout rows produced no data").into());
    }
    Ok(SupervisedBatch {
        features,
        target_update,
    })
}
