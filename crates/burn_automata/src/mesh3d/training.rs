use std::time::Instant;

use burn_automata_kernels::HashGridConfig;

#[cfg(feature = "gpu_wgpu")]
use crate::gpu::{WgpuAutomataExecutor, WgpuNeighborMode};
use crate::{
    AutomataResult, NpaModel, ParticleSeed, SupervisedBatch, SupervisedOptimizerConfig,
    TrainingHistoryEntry, TrainingRunConfig, TrainingRunReport, TriangleMeshTarget,
    WgpuSupervisedTrainingObserver,
    rollout::seed_particles_scaled,
    rollout::{
        GROWTH_3D_LIVENESS_CHANNEL, GROWTH_3D_RENDER_OPACITY_CHANNEL, UV_TORUS_NORMAL_STATE_OFFSET,
        UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET,
    },
    training_wgpu::WgpuSupervisedTrainingSession,
};

use super::attractor::seed_mesh3d_state_attractor;
use super::dataset::{
    Mesh3dParticleBatch, mesh3d_damaged_initialization, mesh3d_supervised_batch_from_particles,
    write_mesh_signed_distance_state, write_mesh_surface_state,
};
use super::evaluation::evaluate_mesh3d_recovery_candidate;
use super::{
    Mesh3dTrainingConfig, Mesh3dTrainingProgress, Mesh3dTrainingReport, Mesh3dTrainingStageReport,
    evaluate_mesh3d_model, mesh3d_model_config, mesh3d_supervised_batch,
};

pub trait Mesh3dTrainingObserver {
    fn should_stop(&self) -> bool {
        false
    }

    fn on_progress(&mut self, progress: Mesh3dTrainingProgress);
}

pub fn train_mesh3d_wgpu(
    target: &TriangleMeshTarget,
    config: Mesh3dTrainingConfig,
) -> AutomataResult<(NpaModel, HashGridConfig, Mesh3dTrainingReport)> {
    train_mesh3d_wgpu_with_observer(target, config, None)
}

pub fn train_mesh3d_wgpu_with_observer(
    target: &TriangleMeshTarget,
    config: Mesh3dTrainingConfig,
    mut observer: Option<&mut dyn Mesh3dTrainingObserver>,
) -> AutomataResult<(NpaModel, HashGridConfig, Mesh3dTrainingReport)> {
    let model_config = mesh3d_model_config(config.hidden_dims);
    let hashgrid = HashGridConfig::growing_3dgs();
    let mut model = NpaModel::upstream_seeded(model_config, config.seed);
    seed_mesh3d_state_attractor(&mut model, target, config.color_gain, config.opacity_gain)?;

    let output_weights = mesh3d_output_weights(&model);
    let refreshes = config.dataset_refreshes.max(1).min(config.steps.max(1));
    let mut dataset_generation_seconds = 0.0;
    let mut training_seconds = 0.0;
    let mut dataset_rows = 0;
    let mut training_rows_processed = 0_u64;
    let mut training: Option<TrainingRunReport> = None;
    let mut trainer: Option<WgpuSupervisedTrainingSession> = None;
    let mut stages = Vec::with_capacity(refreshes);
    let mut replay: Option<SupervisedBatch> = None;
    let mut selected: Option<(NpaModel, usize, f32, f32, f32)> = None;
    let policy_rollout = (refreshes > 1).then(Mesh3dPolicyRollout::new).transpose()?;

    for refresh in 0..refreshes {
        let dataset_start = Instant::now();
        let new_batch = if refresh == 0 {
            mesh3d_supervised_batch(target, &model.config, &hashgrid, &config)?
        } else {
            mesh3d_policy_batch(
                policy_rollout
                    .as_ref()
                    .expect("policy rollout exists for refresh"),
                target,
                &model,
                &hashgrid,
                &config,
                refresh,
                refreshes,
            )?
        };
        dataset_generation_seconds += dataset_start.elapsed().as_secs_f64();
        let policy_horizon = policy_horizon(&config, refresh, refreshes);
        let batch = if config.replay_accumulate {
            append_supervised_batch(&mut replay, new_batch);
            replay.as_ref().expect("replay initialized")
        } else {
            replay = Some(new_batch);
            replay.as_ref().expect("stage batch initialized")
        };
        dataset_rows = batch.features.len() / model.config.perception_dims();
        let stage_steps = partition_steps(config.steps, refresh, refreshes);
        if let Some(trainer) = trainer.as_mut() {
            trainer.replace_batch(batch)?;
        } else {
            trainer = Some(WgpuSupervisedTrainingSession::new(
                &model,
                batch,
                Some(&output_weights),
            )?);
        }
        let training_start = Instant::now();
        let completed_before = training.as_ref().map_or(0, |report| report.steps);
        let mut stage_observer = observer.as_deref_mut().map(|observer| MeshStageObserver {
            observer,
            completed_before,
            total_steps: config.steps,
            refresh,
            refreshes,
            policy_horizon,
            dataset_rows,
        });
        let stage = trainer
            .as_mut()
            .expect("mesh3d trainer initialized")
            .train_into_model(
                &mut model,
                TrainingRunConfig {
                    steps: stage_steps,
                    report_interval: config.report_interval.min(stage_steps).max(1),
                    ..TrainingRunConfig::default()
                },
                SupervisedOptimizerConfig::AdamW(config.optimizer),
                false,
                stage_observer
                    .as_mut()
                    .map(|observer| observer as &mut dyn WgpuSupervisedTrainingObserver),
            )?;
        // The affine material attractor is an architectural prior rather than
        // a fitted target. Re-project it after unconstrained optimizer updates
        // so long rollouts cannot turn small approximation error into material
        // drift while geometry and normal channels remain learned.
        seed_mesh3d_state_attractor(&mut model, target, config.color_gain, config.opacity_gain)?;
        training_rows_processed = training_rows_processed
            .saturating_add((dataset_rows as u64).saturating_mul(stage.steps as u64));
        training_seconds += training_start.elapsed().as_secs_f64();
        let selection =
            evaluate_mesh3d_recovery_candidate(&model, &hashgrid, target, &config.evaluation)?;
        let selection_score = selection
            .density_psnr_db
            .min(selection.damage_region_color_psnr_db);
        let replace_selected =
            selected
                .as_ref()
                .is_none_or(|(_, _, best_score, best_density, _)| {
                    selection_score > *best_score
                        || (selection_score == *best_score
                            && selection.density_psnr_db > *best_density)
                });
        if replace_selected {
            selected = Some((
                model.clone(),
                refresh,
                selection_score,
                selection.density_psnr_db,
                selection.color_psnr_db,
            ));
        }
        stages.push(Mesh3dTrainingStageReport {
            refresh,
            policy_horizon,
            dataset_rows,
            selection_steps: selection.steps,
            selection_density_psnr_db: selection.density_psnr_db,
            selection_color_psnr_db: selection.color_psnr_db,
            selection_damage_region_color_psnr_db: selection.damage_region_color_psnr_db,
            training: stage.clone(),
        });
        merge_training_report(&mut training, stage);
        if observer
            .as_deref()
            .is_some_and(Mesh3dTrainingObserver::should_stop)
        {
            break;
        }
    }
    let training = training.expect("at least one mesh3d training refresh");
    let (selected_model, selected_refresh, _, _, _) =
        selected.expect("at least one mesh3d selection candidate");
    model = selected_model;
    drop(trainer);
    let quality = evaluate_mesh3d_model(&model, &hashgrid, target, &config.evaluation)?;
    let rows_per_second = training_rows_processed as f64 / training_seconds.max(f64::MIN_POSITIVE);
    Ok((
        model,
        hashgrid,
        Mesh3dTrainingReport {
            training,
            stages,
            selected_refresh,
            dataset_rows,
            training_rows_processed,
            dataset_generation_seconds,
            training_seconds,
            rows_per_second,
            quality,
        },
    ))
}

struct MeshStageObserver<'a> {
    observer: &'a mut dyn Mesh3dTrainingObserver,
    completed_before: usize,
    total_steps: usize,
    refresh: usize,
    refreshes: usize,
    policy_horizon: usize,
    dataset_rows: usize,
}

impl WgpuSupervisedTrainingObserver for MeshStageObserver<'_> {
    fn should_stop(&self) -> bool {
        self.observer.should_stop()
    }

    fn on_progress(
        &mut self,
        step: usize,
        _total_steps: usize,
        entry: &TrainingHistoryEntry,
        model: &NpaModel,
    ) {
        self.observer.on_progress(Mesh3dTrainingProgress {
            step: self.completed_before + step,
            total_steps: self.total_steps,
            refresh: self.refresh,
            refreshes: self.refreshes,
            policy_horizon: self.policy_horizon,
            dataset_rows: self.dataset_rows,
            loss: entry.loss,
            grad_norm: entry.grad_norm,
            grad_scale: entry.grad_scale,
            model: model.clone(),
        });
    }
}

fn mesh3d_policy_batch(
    policy_rollout: &Mesh3dPolicyRollout,
    target: &TriangleMeshTarget,
    model: &NpaModel,
    hashgrid: &HashGridConfig,
    config: &Mesh3dTrainingConfig,
    refresh: usize,
    refreshes: usize,
) -> AutomataResult<SupervisedBatch> {
    let particle_count = config.dataset_particles;
    let trajectories = config.dataset_trajectories;
    let (mut positions, mut states) = seed_particles_scaled(
        trajectories,
        particle_count,
        model.config.state_dims,
        model.config.spatial_dims,
        config.seed.wrapping_add(refresh as u64 * 0x9e37_79b9),
        ParticleSeed::UniformCircle,
        config.scale,
    );
    write_mesh_signed_distance_state(target, &model.config, &positions, &mut states);

    // Half the lanes cover the deployable surface and localized recovery
    // distributions. The remaining lanes retain the harder volume-to-surface
    // growth diagnostic.
    initialize_surface_lane(
        target,
        model,
        &mut positions,
        &mut states,
        particle_count,
        0,
        refresh,
    );
    let surface_lanes = trajectories.div_ceil(2);
    for lane in 1..surface_lanes {
        initialize_damaged_surface_lane(
            target,
            model,
            &mut positions,
            &mut states,
            particle_count,
            lane,
            refresh,
            config,
        )?;
    }

    let horizon = policy_horizon(config, refresh, refreshes);
    if horizon > 0 {
        (positions, states) = policy_rollout.rollout(
            model,
            hashgrid,
            positions,
            states,
            trajectories,
            particle_count,
            horizon,
            config.seed.wrapping_add(refresh as u64),
        )?;
    }
    stabilize_policy_rows(target, model, &mut positions, &mut states, config, refresh);

    // Every refresh retains a pristine, correctly populated surface anchor.
    // Without this lane, on-policy volume rows dominate the static-state
    // contract and a deployed mesh initialization drifts immediately.
    initialize_surface_lane(
        target,
        model,
        &mut positions,
        &mut states,
        particle_count,
        0,
        refresh,
    );
    mesh3d_supervised_batch_from_particles(
        target,
        &model.config,
        hashgrid,
        config,
        Mesh3dParticleBatch {
            positions,
            states,
            trajectories,
            particle_count,
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn initialize_damaged_surface_lane(
    target: &TriangleMeshTarget,
    model: &NpaModel,
    positions: &mut [[f32; 4]],
    states: &mut [f32],
    particle_count: usize,
    lane: usize,
    refresh: usize,
    config: &Mesh3dTrainingConfig,
) -> AutomataResult<()> {
    let row_start = lane * particle_count;
    if row_start >= positions.len() {
        return Ok(());
    }
    let seed = config
        .seed
        .wrapping_add((refresh as u64).wrapping_mul(0x9e37_79b9))
        .wrapping_add((lane as u64).wrapping_mul(0x85eb_ca6b));
    let damage = mesh3d_damaged_initialization(
        target,
        &model.config,
        particle_count,
        seed,
        config.evaluation.damage_radius,
        config.evaluation.damage_displacement,
    )?;
    positions[row_start..row_start + particle_count].copy_from_slice(&damage.positions);
    let state_start = row_start * model.config.state_dims;
    states[state_start..state_start + damage.states.len()].copy_from_slice(&damage.states);
    Ok(())
}

fn policy_horizon(config: &Mesh3dTrainingConfig, refresh: usize, refreshes: usize) -> usize {
    config
        .teacher_rollout_max_steps
        .saturating_mul(refresh)
        .div_ceil(refreshes.saturating_sub(1).max(1))
}

fn append_supervised_batch(replay: &mut Option<SupervisedBatch>, mut batch: SupervisedBatch) {
    if let Some(replay) = replay {
        replay.features.append(&mut batch.features);
        replay.target_update.append(&mut batch.target_update);
    } else {
        *replay = Some(batch);
    }
}

fn stabilize_policy_rows(
    target: &TriangleMeshTarget,
    model: &NpaModel,
    positions: &mut [[f32; 4]],
    states: &mut [f32],
    config: &Mesh3dTrainingConfig,
    refresh: usize,
) {
    let state_dims = model.config.state_dims;
    let domain = config.scale * 1.35;
    let row_count = positions.len();
    for (row, position) in positions.iter_mut().enumerate() {
        let state = &mut states[row * state_dims..(row + 1) * state_dims];
        let finite = position.iter().take(3).all(|value| value.is_finite())
            && state.iter().all(|value| value.is_finite());
        if !finite {
            let sample = target.surface_sample(row.wrapping_add(refresh * row_count));
            *position = [
                sample.position[0] + sample.normal[0] * config.scale * 0.03,
                sample.position[1] + sample.normal[1] * config.scale * 0.03,
                sample.position[2] + sample.normal[2] * config.scale * 0.03,
                0.0,
            ];
            write_mesh_surface_state(state, sample.normal, sample.color);
            for value in state.iter_mut() {
                *value *= 0.5;
            }
            continue;
        }
        for value in position.iter_mut().take(3) {
            *value = value.clamp(-domain, domain);
        }
        for value in state.iter_mut() {
            *value = value.clamp(-8.0, 8.0);
        }
        let projection = target.project([position[0], position[1], position[2]]);
        if projection.distance > config.scale * 0.007 {
            state[GROWTH_3D_LIVENESS_CHANNEL] = state[GROWTH_3D_LIVENESS_CHANNEL].min(2.5);
        }
    }
}

fn initialize_surface_lane(
    target: &TriangleMeshTarget,
    model: &NpaModel,
    positions: &mut [[f32; 4]],
    states: &mut [f32],
    particle_count: usize,
    lane: usize,
    refresh: usize,
) {
    let state_dims = model.config.state_dims;
    let row_start = lane * particle_count;
    if row_start >= positions.len() {
        return;
    }
    for local in 0..particle_count {
        let row = row_start + local;
        let sample = target.surface_sample(
            local
                .wrapping_add(refresh * particle_count)
                .wrapping_add(lane * 0x9e37),
        );
        positions[row] = [
            sample.position[0],
            sample.position[1],
            sample.position[2],
            1.0,
        ];
        let state = &mut states[row * state_dims..(row + 1) * state_dims];
        write_mesh_surface_state(state, sample.normal, sample.color);
    }
}

struct Mesh3dPolicyRollout {
    #[cfg(feature = "gpu_wgpu")]
    executor: WgpuAutomataExecutor,
}

impl Mesh3dPolicyRollout {
    fn new() -> AutomataResult<Self> {
        Ok(Self {
            #[cfg(feature = "gpu_wgpu")]
            executor: WgpuAutomataExecutor::new_blocking()?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn rollout(
        &self,
        model: &NpaModel,
        hashgrid: &HashGridConfig,
        positions: Vec<[f32; 4]>,
        states: Vec<f32>,
        trajectories: usize,
        particle_count: usize,
        horizon: usize,
        seed: u64,
    ) -> AutomataResult<(Vec<[f32; 4]>, Vec<f32>)> {
        #[cfg(feature = "gpu_wgpu")]
        {
            let mut state = self
                .executor
                .create_state_with_neighbor_mode_and_update_prob(
                    model,
                    &positions,
                    &states,
                    trajectories,
                    particle_count,
                    hashgrid,
                    1.0,
                    // Sorted-cell scans currently cap the aggregate batch grid at
                    // 65,536 cells. Eight independent 32^3 3D grids exceed that
                    // bound even though per-cell occupancy is sparse.
                    WgpuNeighborMode::LinkedList,
                    0.5,
                    seed,
                )?;
            self.executor.step_state_many(&mut state, horizon)?;
            self.executor.read_positions_states(&state)
        }
        #[cfg(not(feature = "gpu_wgpu"))]
        {
            use rand::{Rng, SeedableRng};

            let mut positions = positions;
            let mut states = states;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0x706f_6c69_6379);
            for _ in 0..horizon {
                let mask = (0..positions.len())
                    .map(|_| f32::from(rng.random::<f32>() < 0.5))
                    .collect::<Vec<_>>();
                let step = model.step_cpu(
                    &positions,
                    &states,
                    trajectories,
                    particle_count,
                    hashgrid,
                    1.0,
                    Some(&mask),
                )?;
                positions = step.next_positions;
                states = step.next_states;
            }
            Ok((positions, states))
        }
    }
}

fn partition_steps(total: usize, index: usize, parts: usize) -> usize {
    total / parts + usize::from(index < total % parts)
}

fn merge_training_report(accumulator: &mut Option<TrainingRunReport>, mut next: TrainingRunReport) {
    let Some(current) = accumulator.as_mut() else {
        *accumulator = Some(next);
        return;
    };
    let offset = current.steps;
    current.steps += next.steps;
    current.rows = next.rows;
    current.final_loss = next.final_loss;
    current.best_loss = current.best_loss.min(next.best_loss);
    current
        .history
        .extend(next.history.drain(..).map(|entry| TrainingHistoryEntry {
            step: entry.step + offset,
            ..entry
        }));
}

fn mesh3d_output_weights(model: &NpaModel) -> Vec<f32> {
    let spatial_dims = model.config.spatial_dims;
    let state_dims = model.config.state_dims;
    let mut weights = vec![0.02_f32; model.config.update_dims()];
    weights[..spatial_dims].fill(12.0);
    let state = &mut weights[spatial_dims..];
    state[GROWTH_3D_LIVENESS_CHANNEL] = 2.0;
    state[GROWTH_3D_RENDER_OPACITY_CHANNEL] = 2.0;
    state[UV_TORUS_NORMAL_STATE_OFFSET..UV_TORUS_NORMAL_STATE_OFFSET + 3].fill(4.0);
    state[UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET] = 2.0;
    state[state_dims - 3..].fill(16.0);
    weights
}
