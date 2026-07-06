use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

use crate::{
    AdamWConfig, AdamWState, AutomataError, AutomataResult, NpaConfig, NpaModel, ParticleSeed,
    SupervisedGradients, apply_adamw_gradients,
    kernels::{HashGridConfig, PerceptionOptions, euler_step, perceive_adjoint_with_options},
    mlp_backward_from_output_gradients,
    rollout::{RolloutConfig, run_rollout, seed_particles_scaled, stochastic_mask},
};

const DEFAULT_AABB: [f32; 4] = [-1.0, 1.0, -1.0, 1.0];
const SPATIAL_DIMS_2D: usize = 2;
const IMAGE_EPSILON: f32 = 1.0e-8;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TargetImage2dExtractConfig {
    pub threshold: f32,
    pub aabb: [f32; 4],
}

impl Default for TargetImage2dExtractConfig {
    fn default() -> Self {
        Self {
            threshold: 0.05,
            aabb: DEFAULT_AABB,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TargetImage2d {
    pub source_width: usize,
    pub source_height: usize,
    pub positions: Vec<[f32; 2]>,
    pub colors: Vec<[f32; 3]>,
    pub pixel_size: f32,
    pub threshold: f32,
    pub aabb: [f32; 4],
}

impl TargetImage2d {
    pub fn from_rgba_pixels(
        width: usize,
        height: usize,
        rgba: &[f32],
        cfg: TargetImage2dExtractConfig,
    ) -> AutomataResult<Self> {
        if width == 0 || height == 0 {
            return Err(AutomataError::InvalidArgument(
                "target image dimensions must be positive".to_string(),
            ));
        }
        if rgba.len() != width * height * 4 {
            return Err(AutomataError::InvalidArgument(format!(
                "target RGBA len {} != {}",
                rgba.len(),
                width * height * 4
            )));
        }
        if !cfg.threshold.is_finite() || cfg.threshold < 0.0 {
            return Err(AutomataError::InvalidArgument(format!(
                "target threshold must be finite and non-negative, got {}",
                cfg.threshold
            )));
        }
        let [min_x, max_x, min_y, max_y] = cfg.aabb;
        let size_x = max_x - min_x;
        let size_y = max_y - min_y;
        if !size_x.is_finite() || !size_y.is_finite() || size_x <= 0.0 || size_y <= 0.0 {
            return Err(AutomataError::InvalidArgument(
                "target aabb extents must be finite and positive".to_string(),
            ));
        }

        let mut positions = Vec::new();
        let mut colors = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let base = (y * width + x) * 4;
                let alpha = rgba[base + 3];
                if alpha <= cfg.threshold {
                    continue;
                }
                let world_x = min_x + size_x * (x as f32 + 0.5) / width as f32;
                let world_y = max_y - size_y * (y as f32 + 0.5) / height as f32;
                positions.push([world_x, world_y]);
                colors.push([rgba[base], rgba[base + 1], rgba[base + 2]]);
            }
        }
        if positions.is_empty() {
            return Err(AutomataError::InvalidArgument(
                "target image produced zero foreground points".to_string(),
            ));
        }

        Ok(Self {
            source_width: width,
            source_height: height,
            positions,
            colors,
            pixel_size: size_x / width as f32,
            threshold: cfg.threshold,
            aabb: cfg.aabb,
        })
    }

    pub fn point_count(&self) -> usize {
        self.positions.len()
    }

    pub fn mean_position(&self) -> [f32; 2] {
        let mut mean = [0.0_f32; 2];
        for position in &self.positions {
            mean[0] += position[0];
            mean[1] += position[1];
        }
        let denom = self.positions.len().max(1) as f32;
        mean[0] /= denom;
        mean[1] /= denom;
        mean
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Target2dLossConfig {
    pub image_size: usize,
    pub sigma: f32,
    pub lo: f32,
    pub hi: f32,
    pub center: bool,
    pub splat_loss_weight: f32,
    pub color_loss_weight: f32,
    pub density_loss_weight: f32,
    pub displacement_regularizer_weight: f32,
    pub overflow_regularizer_weight: f32,
    pub bound_regularizer_weight: f32,
}

impl Default for Target2dLossConfig {
    fn default() -> Self {
        Self {
            image_size: 256,
            sigma: 1.0,
            lo: -1.0,
            hi: 1.0,
            center: true,
            splat_loss_weight: 2.0,
            color_loss_weight: 5.0,
            density_loss_weight: 1.0,
            displacement_regularizer_weight: 0.01,
            overflow_regularizer_weight: 100.0,
            bound_regularizer_weight: 100.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Target2dLossReport {
    pub total_loss: f32,
    pub splat_loss: f32,
    pub color_loss: f32,
    pub density_loss: f32,
    pub displacement_regularizer: f32,
    pub overflow_regularizer: f32,
    pub bound_regularizer: f32,
    pub target_points: usize,
    pub particle_count: usize,
    pub batch_size: usize,
    pub image_size: usize,
}

#[derive(Clone, Debug)]
pub struct Target2dLossOutput {
    pub report: Target2dLossReport,
    pub position_gradients: Vec<[f32; 4]>,
    pub state_gradients: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Target2dTrainingConfig {
    pub epochs: usize,
    pub repetitions: usize,
    pub report_interval: usize,
    pub batch_size: usize,
    pub pool_size: usize,
    pub particle_count: usize,
    pub step_min: usize,
    pub step_max: usize,
    pub inject_seed_interval: usize,
    pub update_prob: f32,
    pub seed: u64,
    pub seed_scale: f32,
    pub seed_mode: ParticleSeed,
    pub brush_size: f32,
    pub per_parameter_grad_normalization: bool,
    pub optimizer: AdamWConfig,
    pub scheduler_milestones: Vec<usize>,
    pub scheduler_gamma: f32,
}

impl Default for Target2dTrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 10_000,
            repetitions: 3,
            report_interval: 100,
            batch_size: 8,
            pool_size: 512,
            particle_count: 4096,
            step_min: 32,
            step_max: 96,
            inject_seed_interval: 16,
            update_prob: 0.5,
            seed: 42,
            seed_scale: 0.2,
            seed_mode: ParticleSeed::UniformCircle,
            brush_size: 0.1,
            per_parameter_grad_normalization: true,
            optimizer: AdamWConfig {
                learning_rate: 5.0e-4,
                weight_decay: 0.0,
                grad_clip_norm: 0.0,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1.0e-8,
            },
            scheduler_milestones: vec![2000, 4000, 6000, 8000],
            scheduler_gamma: 0.3,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Target2dTrainingHistoryEntry {
    pub epoch: usize,
    pub rollout_steps: usize,
    pub loss: Target2dLossReport,
    pub eval_loss: Option<Target2dLossReport>,
    pub grad_norm: f32,
    pub grad_scale: f32,
    pub elapsed_ms: f64,
    pub particle_steps_per_sec: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Target2dTrainingReport {
    pub objective: &'static str,
    pub gradient_mode: &'static str,
    pub initializer: &'static str,
    pub config: Target2dTrainingConfig,
    pub loss_config: Target2dLossConfig,
    pub initial_loss: Target2dLossReport,
    pub initial_eval_loss: Target2dLossReport,
    pub final_loss: Target2dLossReport,
    pub best_loss: Target2dLossReport,
    pub best_epoch: usize,
    pub best_eval_loss: Target2dLossReport,
    pub best_eval_epoch: usize,
    pub epochs_completed: usize,
    pub repetitions_completed: usize,
    pub total_elapsed_ms: f64,
    pub median_particle_steps_per_sec: f64,
    pub history: Vec<Target2dTrainingHistoryEntry>,
}

#[derive(Clone, Debug)]
struct RolloutSnapshot {
    positions: Vec<[f32; 4]>,
    states: Vec<f32>,
    mask: Vec<f32>,
}

#[derive(Clone, Debug)]
struct RolloutForTraining {
    snapshots: Vec<RolloutSnapshot>,
    final_positions: Vec<[f32; 4]>,
    final_states: Vec<f32>,
    mean_dx_norm_sum: f32,
    steps: usize,
    batch_size: usize,
    particle_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Target2dRenderedSplat {
    pub rgb: Vec<f32>,
    pub density: Vec<f32>,
}

type RenderedSplat2d = Target2dRenderedSplat;

#[derive(Clone, Debug)]
struct ParticleSplatSample {
    pixel: usize,
    g: f32,
    dx: f32,
    dy: f32,
}

pub fn upstream_growing_2d_hashgrid() -> HashGridConfig {
    HashGridConfig {
        dim: 2,
        boundary: burn_automata_kernels::Boundary::Clamped,
        mode: burn_automata_kernels::HashGridMode::Grid,
        grid_size: [16, 16, 1],
        eps: 0.1,
        max_particles_per_block: 32,
    }
}

pub fn upstream_growing_2d_model(seed: u64) -> NpaModel {
    NpaModel::upstream_seeded(NpaConfig::growing_2d(), seed)
}

pub fn render_target_2d_splat(
    target: &TargetImage2d,
    cfg: Target2dLossConfig,
) -> AutomataResult<Target2dRenderedSplat> {
    render_target_splat(target, cfg)
}

pub fn render_rollout_2d_splat(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    pixel_size: f32,
    cfg: Target2dLossConfig,
    center: Option<[f32; 2]>,
    output_scale: f32,
) -> AutomataResult<Target2dRenderedSplat> {
    if state_dims < 3 {
        return Err(AutomataError::InvalidArgument(
            "2D rollout splat rendering requires at least three state channels".to_string(),
        ));
    }
    if states.len() != positions.len() * state_dims {
        return Err(AutomataError::InvalidArgument(format!(
            "state len {} != positions {} * state dims {}",
            states.len(),
            positions.len(),
            state_dims
        )));
    }
    if !pixel_size.is_finite() || pixel_size <= 0.0 {
        return Err(AutomataError::InvalidArgument(
            "2D rollout splat rendering requires a positive finite pixel size".to_string(),
        ));
    }
    if !output_scale.is_finite() || output_scale < 0.0 {
        return Err(AutomataError::InvalidArgument(
            "2D rollout splat rendering requires a finite non-negative output scale".to_string(),
        ));
    }
    let positions_2d = if let Some(target_mean) = center {
        centered_batch_positions(positions, target_mean, true)
    } else {
        positions
            .iter()
            .map(|position| [position[0], position[1]])
            .collect()
    };
    let colors = tail_colors(states, state_dims);
    splat_render(&positions_2d, &colors, pixel_size, cfg, output_scale)
}

pub fn target_2d_loss(
    positions: &[[f32; 4]],
    states: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    target: &TargetImage2d,
    cfg: Target2dLossConfig,
) -> AutomataResult<Target2dLossReport> {
    Ok(target_2d_loss_with_adjoint(
        positions,
        states,
        batch_size,
        particle_count,
        state_dims,
        target,
        cfg,
        0.0,
        0,
    )?
    .report)
}

#[allow(clippy::too_many_arguments)]
pub fn target_2d_loss_with_adjoint(
    positions: &[[f32; 4]],
    states: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    target: &TargetImage2d,
    cfg: Target2dLossConfig,
    mean_dx_norm_sum: f32,
    _rollout_steps: usize,
) -> AutomataResult<Target2dLossOutput> {
    validate_target_loss_inputs(
        positions,
        states,
        batch_size,
        particle_count,
        state_dims,
        target,
        cfg,
    )?;
    let target_render = render_target_splat(target, cfg)?;
    let pixels = cfg.image_size * cfg.image_size;
    let color_denom = (batch_size * pixels * 3).max(1) as f32;
    let density_denom = (batch_size * pixels).max(1) as f32;
    let mut color_loss = 0.0_f32;
    let mut density_loss = 0.0_f32;
    let mut position_gradients = vec![[0.0_f32; 4]; positions.len()];
    let mut state_gradients = vec![0.0_f32; states.len()];
    let target_mean = target.mean_position();

    for batch in 0..batch_size {
        let range = batch * particle_count..(batch + 1) * particle_count;
        let centered = centered_batch_positions(&positions[range.clone()], target_mean, cfg.center);
        let colors = tail_colors(
            &states[range.start * state_dims..range.end * state_dims],
            state_dims,
        );
        let point_scale = target.point_count() as f32 / particle_count.max(1) as f32;
        let rendered = splat_render(&centered, &colors, target.pixel_size, cfg, point_scale)?;
        let mut rgb_adjoint = vec![0.0_f32; pixels * 3];
        let mut density_adjoint = vec![0.0_f32; pixels];

        for (pixel, density_adj) in density_adjoint.iter_mut().enumerate().take(pixels) {
            let density_diff = rendered.density[pixel] - target_render.density[pixel];
            let density_term = l1l2(density_diff);
            density_loss += density_term / density_denom;
            *density_adj =
                cfg.splat_loss_weight * cfg.density_loss_weight * l1l2_grad(density_diff)
                    / density_denom;
            let color_gate = (-density_term).exp();
            for channel in 0..3 {
                let idx = pixel * 3 + channel;
                let color_diff = rendered.rgb[idx] - target_render.rgb[idx];
                color_loss += l1l2(color_diff) * color_gate / color_denom;
                rgb_adjoint[idx] = cfg.splat_loss_weight
                    * cfg.color_loss_weight
                    * color_gate
                    * l1l2_grad(color_diff)
                    / color_denom;
            }
        }

        let adjoint = splat_adjoint(
            &centered,
            &colors,
            &rgb_adjoint,
            &density_adjoint,
            target.pixel_size,
            cfg,
            point_scale,
        )?;
        let mut mean_position_adjoint = [0.0_f32; 2];
        if cfg.center {
            for gradient in &adjoint.positions {
                mean_position_adjoint[0] += gradient[0] / particle_count.max(1) as f32;
                mean_position_adjoint[1] += gradient[1] / particle_count.max(1) as f32;
            }
        }
        for local in 0..particle_count {
            let row = range.start + local;
            position_gradients[row][0] += adjoint.positions[local][0] - mean_position_adjoint[0];
            position_gradients[row][1] += adjoint.positions[local][1] - mean_position_adjoint[1];
            let state_base = row * state_dims + state_dims - 3;
            state_gradients[state_base] += adjoint.colors[local][0];
            state_gradients[state_base + 1] += adjoint.colors[local][1];
            state_gradients[state_base + 2] += adjoint.colors[local][2];
        }
    }

    let overflow_regularizer = overflow_regularizer_and_adjoint(
        states,
        &mut state_gradients,
        cfg.overflow_regularizer_weight,
    );
    let bound_regularizer = bound_regularizer_and_adjoint(
        positions,
        &mut position_gradients,
        SPATIAL_DIMS_2D,
        cfg.bound_regularizer_weight,
    );
    let displacement_regularizer = mean_dx_norm_sum;
    let splat_loss = cfg.color_loss_weight * color_loss + cfg.density_loss_weight * density_loss;
    let total_loss = cfg.splat_loss_weight * splat_loss
        + cfg.displacement_regularizer_weight * displacement_regularizer
        + cfg.overflow_regularizer_weight * overflow_regularizer
        + cfg.bound_regularizer_weight * bound_regularizer;

    Ok(Target2dLossOutput {
        report: Target2dLossReport {
            total_loss,
            splat_loss,
            color_loss,
            density_loss,
            displacement_regularizer,
            overflow_regularizer,
            bound_regularizer,
            target_points: target.point_count(),
            particle_count,
            batch_size,
            image_size: cfg.image_size,
        },
        position_gradients,
        state_gradients,
    })
}

pub fn train_target_2d(
    model: &mut NpaModel,
    grid: &HashGridConfig,
    target: &TargetImage2d,
    cfg: Target2dTrainingConfig,
    loss_cfg: Target2dLossConfig,
) -> AutomataResult<Target2dTrainingReport> {
    validate_training_config(&cfg)?;
    model.validate()?;
    if model.config.spatial_dims != SPATIAL_DIMS_2D || grid.dim != SPATIAL_DIMS_2D {
        return Err(AutomataError::InvalidArgument(
            "target 2D training requires a 2D model and hashgrid".to_string(),
        ));
    }

    let mut rng = StdRng::seed_from_u64(cfg.seed ^ 0x7a27_2d91);
    let mut best_loss = Target2dLossReport {
        total_loss: f32::INFINITY,
        ..Target2dLossReport::default()
    };
    let mut best_epoch = 0usize;
    let initial_eval_loss = evaluate_seed_rollout_loss(model, grid, target, &cfg, loss_cfg)?;
    let mut best_eval_loss = initial_eval_loss;
    let mut best_eval_model = model.clone();
    let mut best_eval_epoch = 0usize;
    let mut initial_loss = None;
    let mut final_loss = Target2dLossReport::default();
    let mut history = Vec::new();
    let mut throughputs = Vec::new();
    let total_start = std::time::Instant::now();

    let updates_per_repetition = cfg.epochs + 1;
    let total_updates = updates_per_repetition * cfg.repetitions;
    for repetition in 0..cfg.repetitions {
        let mut pool = ParticlePool2d::new(model, cfg.pool_size, cfg.particle_count, &cfg);
        let mut adamw_state = AdamWState::for_model(model);
        for local_epoch in 0..=cfg.epochs {
            let epoch = repetition * cfg.epochs + local_epoch;
            let update_index = repetition * updates_per_repetition + local_epoch;
            let epoch_start = std::time::Instant::now();
            let replace_seed = epoch.is_multiple_of(cfg.inject_seed_interval.max(1));
            let sample = pool.sample_batch(&mut rng, replace_seed, model, &cfg);
            let rollout_steps = sample_rollout_steps(&mut rng, cfg.step_min, cfg.step_max);
            let rollout = rollout_for_training(
                model,
                grid,
                sample.positions,
                sample.states,
                cfg.batch_size,
                cfg.particle_count,
                rollout_steps,
                cfg.update_prob,
                cfg.seed ^ epoch as u64,
            )?;
            let loss = target_2d_loss_with_adjoint(
                &rollout.final_positions,
                &rollout.final_states,
                cfg.batch_size,
                cfg.particle_count,
                model.config.state_dims,
                target,
                loss_cfg,
                rollout.mean_dx_norm_sum,
                rollout.steps,
            )?;
            initial_loss.get_or_insert(loss.report);
            final_loss = loss.report;
            if final_loss.total_loss < best_loss.total_loss {
                best_loss = final_loss;
                best_epoch = epoch;
            }
            let mut gradients = bptt_gradients(model, grid, &rollout, &loss, loss_cfg)?;
            if cfg.per_parameter_grad_normalization {
                normalize_gradient_tensors(&mut gradients);
            }
            let optimizer = optimizer_for_epoch(cfg.optimizer, &cfg, local_epoch);
            let step_report = apply_adamw_gradients(model, gradients, &mut adamw_state, optimizer)?;
            pool.update_batch(
                &sample.indices,
                &rollout.final_positions,
                &rollout.final_states,
                cfg.particle_count,
            );

            let elapsed_ms = epoch_start.elapsed().as_secs_f64() * 1000.0;
            let particle_steps =
                cfg.batch_size as f64 * cfg.particle_count as f64 * rollout_steps as f64;
            let throughput =
                particle_steps / epoch_start.elapsed().as_secs_f64().max(f64::MIN_POSITIVE);
            throughputs.push(throughput);
            let should_report = update_index + 1 == total_updates
                || epoch.is_multiple_of(cfg.report_interval.max(1));
            let eval_loss = if should_report {
                Some(evaluate_seed_rollout_loss(
                    model, grid, target, &cfg, loss_cfg,
                )?)
            } else {
                None
            };
            if let Some(eval_loss) = eval_loss {
                if eval_loss.total_loss < best_eval_loss.total_loss {
                    best_eval_loss = eval_loss;
                    best_eval_model = model.clone();
                    best_eval_epoch = epoch;
                }
                history.push(Target2dTrainingHistoryEntry {
                    epoch,
                    rollout_steps,
                    loss: final_loss,
                    eval_loss: Some(eval_loss),
                    grad_norm: step_report.grad_norm,
                    grad_scale: step_report.grad_scale,
                    elapsed_ms,
                    particle_steps_per_sec: throughput,
                });
            }
        }
    }

    throughputs.sort_by(|a, b| a.total_cmp(b));
    let median_particle_steps_per_sec = throughputs
        .get(throughputs.len() / 2)
        .copied()
        .unwrap_or_default();
    let epochs_completed = total_updates;
    let repetitions_completed = cfg.repetitions;
    if best_eval_loss.total_loss.is_finite() {
        *model = best_eval_model;
        final_loss = best_eval_loss;
    }
    Ok(Target2dTrainingReport {
        objective: "upstream_target_image_splat_loss",
        gradient_mode: "exact_bptt_reference_stopgrad_pos",
        initializer: "upstream_xavier_uniform_gain_0_1_pytorch_linear_bias",
        config: cfg,
        loss_config: loss_cfg,
        initial_loss: initial_loss.unwrap_or_default(),
        initial_eval_loss,
        final_loss,
        best_loss,
        best_epoch,
        best_eval_loss,
        best_eval_epoch,
        epochs_completed,
        repetitions_completed,
        total_elapsed_ms: total_start.elapsed().as_secs_f64() * 1000.0,
        median_particle_steps_per_sec,
        history,
    })
}

fn evaluate_seed_rollout_loss(
    model: &NpaModel,
    grid: &HashGridConfig,
    target: &TargetImage2d,
    cfg: &Target2dTrainingConfig,
    loss_cfg: Target2dLossConfig,
) -> AutomataResult<Target2dLossReport> {
    let trace = run_rollout(
        model,
        grid,
        &RolloutConfig {
            batch_size: 1,
            particle_count: cfg.particle_count,
            steps: cfg.step_max,
            update_prob: cfg.update_prob,
            seed: cfg.seed,
            seed_scale: cfg.seed_scale,
            ..RolloutConfig::default()
        },
        cfg.seed_mode,
    )?;
    Ok(target_2d_loss_with_adjoint(
        &trace.positions,
        &trace.states,
        trace.batch_size,
        trace.particle_count,
        trace.state_dims,
        target,
        loss_cfg,
        trace.mean_dx.iter().copied().sum(),
        trace.steps,
    )?
    .report)
}

pub fn target_2d_rollout_loss_with_gradients(
    model: &NpaModel,
    grid: &HashGridConfig,
    target: &TargetImage2d,
    cfg: RolloutConfig,
    seed_mode: ParticleSeed,
    loss_cfg: Target2dLossConfig,
    per_parameter_grad_normalization: bool,
) -> AutomataResult<(Target2dLossReport, SupervisedGradients)> {
    model.validate()?;
    if model.config.spatial_dims != SPATIAL_DIMS_2D || grid.dim != SPATIAL_DIMS_2D {
        return Err(AutomataError::InvalidArgument(
            "target 2D rollout gradients require a 2D model and hashgrid".to_string(),
        ));
    }
    if cfg.batch_size == 0 || cfg.particle_count == 0 || cfg.steps == 0 {
        return Err(AutomataError::InvalidArgument(
            "target 2D rollout gradients require non-zero batch, particles, and steps".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&cfg.update_prob) || !cfg.update_prob.is_finite() {
        return Err(AutomataError::InvalidArgument(
            "target 2D rollout gradients require finite update_prob in [0, 1]".to_string(),
        ));
    }
    let (positions, states) = seed_particles_scaled(
        cfg.batch_size,
        cfg.particle_count,
        model.config.state_dims,
        model.config.spatial_dims,
        cfg.seed,
        seed_mode,
        cfg.seed_scale,
    );
    let rollout = rollout_for_training(
        model,
        grid,
        positions,
        states,
        cfg.batch_size,
        cfg.particle_count,
        cfg.steps,
        cfg.update_prob,
        cfg.seed,
    )?;
    let loss = target_2d_loss_with_adjoint(
        &rollout.final_positions,
        &rollout.final_states,
        cfg.batch_size,
        cfg.particle_count,
        model.config.state_dims,
        target,
        loss_cfg,
        rollout.mean_dx_norm_sum,
        rollout.steps,
    )?;
    let report = loss.report;
    let mut gradients = bptt_gradients(model, grid, &rollout, &loss, loss_cfg)?;
    if per_parameter_grad_normalization {
        normalize_gradient_tensors(&mut gradients);
    }
    Ok((report, gradients))
}

fn validate_target_loss_inputs(
    positions: &[[f32; 4]],
    states: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    target: &TargetImage2d,
    cfg: Target2dLossConfig,
) -> AutomataResult<()> {
    if batch_size == 0 || particle_count == 0 || state_dims < 3 {
        return Err(AutomataError::InvalidArgument(
            "target 2D loss requires non-zero batch/particles and at least three state channels"
                .to_string(),
        ));
    }
    if positions.len() != batch_size * particle_count {
        return Err(AutomataError::InvalidArgument(format!(
            "position len {} != {}",
            positions.len(),
            batch_size * particle_count
        )));
    }
    if states.len() != positions.len() * state_dims {
        return Err(AutomataError::InvalidArgument(format!(
            "state len {} != {}",
            states.len(),
            positions.len() * state_dims
        )));
    }
    if target.positions.len() != target.colors.len() || target.positions.is_empty() {
        return Err(AutomataError::InvalidArgument(
            "target image must contain matching non-empty positions and colors".to_string(),
        ));
    }
    if cfg.image_size == 0 || !cfg.sigma.is_finite() || cfg.sigma <= 0.0 {
        return Err(AutomataError::InvalidArgument(
            "target 2D loss requires positive image_size and sigma".to_string(),
        ));
    }
    if !cfg.lo.is_finite() || !cfg.hi.is_finite() || cfg.hi <= cfg.lo {
        return Err(AutomataError::InvalidArgument(
            "target 2D loss requires finite lo < hi".to_string(),
        ));
    }
    Ok(())
}

fn validate_training_config(cfg: &Target2dTrainingConfig) -> AutomataResult<()> {
    if cfg.epochs == 0
        || cfg.repetitions == 0
        || cfg.batch_size == 0
        || cfg.pool_size == 0
        || cfg.particle_count == 0
        || cfg.step_min == 0
        || cfg.step_max < cfg.step_min
    {
        return Err(AutomataError::InvalidArgument(
            "target 2D training requires positive epochs/batch/pool/particles/steps".to_string(),
        ));
    }
    if cfg.batch_size > cfg.pool_size {
        return Err(AutomataError::InvalidArgument(
            "target 2D training batch_size must not exceed pool_size".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&cfg.update_prob) || !cfg.update_prob.is_finite() {
        return Err(AutomataError::InvalidArgument(
            "target 2D training update_prob must be finite and in [0, 1]".to_string(),
        ));
    }
    if !cfg.seed_scale.is_finite() || cfg.seed_scale <= 0.0 {
        return Err(AutomataError::InvalidArgument(
            "target 2D training seed_scale must be finite and positive".to_string(),
        ));
    }
    if !cfg.brush_size.is_finite() || cfg.brush_size < 0.0 {
        return Err(AutomataError::InvalidArgument(
            "target 2D training brush_size must be finite and non-negative".to_string(),
        ));
    }
    if !cfg.scheduler_gamma.is_finite() || cfg.scheduler_gamma <= 0.0 {
        return Err(AutomataError::InvalidArgument(
            "target 2D training scheduler_gamma must be finite and positive".to_string(),
        ));
    }
    Ok(())
}

fn optimizer_for_epoch(
    mut optimizer: AdamWConfig,
    cfg: &Target2dTrainingConfig,
    local_epoch: usize,
) -> AdamWConfig {
    let milestones_passed = cfg
        .scheduler_milestones
        .iter()
        .filter(|milestone| **milestone <= local_epoch)
        .count();
    optimizer.learning_rate *= cfg.scheduler_gamma.powi(milestones_passed as i32);
    optimizer
}

fn render_target_splat(
    target: &TargetImage2d,
    cfg: Target2dLossConfig,
) -> AutomataResult<RenderedSplat2d> {
    let positions = target
        .positions
        .iter()
        .map(|position| [position[0], position[1]])
        .collect::<Vec<_>>();
    splat_render(&positions, &target.colors, target.pixel_size, cfg, 1.0)
}

fn centered_batch_positions(
    positions: &[[f32; 4]],
    target_mean: [f32; 2],
    center: bool,
) -> Vec<[f32; 2]> {
    let mut mean = [0.0_f32; 2];
    if center {
        for position in positions {
            mean[0] += position[0];
            mean[1] += position[1];
        }
        let denom = positions.len().max(1) as f32;
        mean[0] /= denom;
        mean[1] /= denom;
    }
    positions
        .iter()
        .map(|position| {
            if center {
                [
                    position[0] - mean[0] + target_mean[0],
                    position[1] - mean[1] + target_mean[1],
                ]
            } else {
                [position[0], position[1]]
            }
        })
        .collect()
}

fn tail_colors(states: &[f32], state_dims: usize) -> Vec<[f32; 3]> {
    states
        .chunks_exact(state_dims)
        .map(|state| {
            [
                state[state_dims - 3] + 0.5,
                state[state_dims - 2] + 0.5,
                state[state_dims - 1] + 0.5,
            ]
        })
        .collect()
}

fn splat_render(
    positions: &[[f32; 2]],
    colors: &[[f32; 3]],
    pixel_size: f32,
    cfg: Target2dLossConfig,
    output_scale: f32,
) -> AutomataResult<RenderedSplat2d> {
    if positions.len() != colors.len() {
        return Err(AutomataError::InvalidArgument(
            "splat positions and colors lengths differ".to_string(),
        ));
    }
    let size = cfg.image_size;
    let pixels = size * size;
    let mut rgb = vec![0.0_f32; pixels * 3];
    let mut density = vec![0.0_f32; pixels];
    for (position, color) in positions.iter().zip(colors) {
        let samples = particle_splat_samples(*position, pixel_size, cfg);
        let denom = samples.iter().map(|sample| sample.g).sum::<f32>() + IMAGE_EPSILON;
        let norm_scale = splat_norm_scale(pixel_size, cfg);
        for sample in samples {
            let weight = output_scale * norm_scale * sample.g / denom;
            density[sample.pixel] += weight;
            let rgb_base = sample.pixel * 3;
            rgb[rgb_base] += color[0] * weight;
            rgb[rgb_base + 1] += color[1] * weight;
            rgb[rgb_base + 2] += color[2] * weight;
        }
    }
    Ok(RenderedSplat2d { rgb, density })
}

#[derive(Clone, Debug)]
struct SplatAdjoint {
    positions: Vec<[f32; 2]>,
    colors: Vec<[f32; 3]>,
}

fn splat_adjoint(
    positions: &[[f32; 2]],
    colors: &[[f32; 3]],
    rgb_adjoint: &[f32],
    density_adjoint: &[f32],
    pixel_size: f32,
    cfg: Target2dLossConfig,
    output_scale: f32,
) -> AutomataResult<SplatAdjoint> {
    let pixels = cfg.image_size * cfg.image_size;
    if rgb_adjoint.len() != pixels * 3 || density_adjoint.len() != pixels {
        return Err(AutomataError::InvalidArgument(
            "splat image adjoint sizes do not match image_size".to_string(),
        ));
    }
    let mut position_adjoint = vec![[0.0_f32; 2]; positions.len()];
    let mut color_adjoint = vec![[0.0_f32; 3]; colors.len()];
    let sigma = splat_sigma_pixels(pixel_size, cfg);
    let inv_sigma2 = (sigma * sigma).recip();
    let pixel_to_world = (cfg.image_size as f32 - 1.0) / (cfg.hi - cfg.lo);
    let norm_scale = splat_norm_scale(pixel_size, cfg) * output_scale;

    for (particle, (position, color)) in positions.iter().zip(colors).enumerate() {
        let samples = particle_splat_samples(*position, pixel_size, cfg);
        let denom = samples.iter().map(|sample| sample.g).sum::<f32>() + IMAGE_EPSILON;
        let mut weighted_adjoint_sum = 0.0_f32;
        let mut sample_weight_adjoint = Vec::with_capacity(samples.len());
        for sample in &samples {
            let rgb_base = sample.pixel * 3;
            let mut weight_adjoint = density_adjoint[sample.pixel];
            for channel in 0..3 {
                color_adjoint[particle][channel] +=
                    rgb_adjoint[rgb_base + channel] * norm_scale * sample.g / denom;
                weight_adjoint += rgb_adjoint[rgb_base + channel] * color[channel];
            }
            sample_weight_adjoint.push(weight_adjoint);
            weighted_adjoint_sum += weight_adjoint * sample.g;
        }

        let mut pix_adjoint = [0.0_f32; 2];
        for (sample, weight_adjoint) in samples.iter().zip(sample_weight_adjoint.iter()) {
            let g_adjoint =
                norm_scale * (*weight_adjoint / denom - weighted_adjoint_sum / (denom * denom));
            let g_pos = g_adjoint * sample.g * inv_sigma2;
            pix_adjoint[0] += g_pos * sample.dx;
            pix_adjoint[1] += g_pos * sample.dy;
        }
        position_adjoint[particle][0] += pix_adjoint[0] * pixel_to_world;
        position_adjoint[particle][1] -= pix_adjoint[1] * pixel_to_world;
    }

    Ok(SplatAdjoint {
        positions: position_adjoint,
        colors: color_adjoint,
    })
}

fn particle_splat_samples(
    position: [f32; 2],
    pixel_size: f32,
    cfg: Target2dLossConfig,
) -> Vec<ParticleSplatSample> {
    let size = cfg.image_size;
    let sigma = splat_sigma_pixels(pixel_size, cfg);
    let radius = (5.0 * sigma).ceil().max(1.0) as isize;
    let px = (position[0] - cfg.lo) / (cfg.hi - cfg.lo) * (size as f32 - 1.0);
    let py_unflipped = (position[1] - cfg.lo) / (cfg.hi - cfg.lo) * (size as f32 - 1.0);
    let py = (size as f32 - 1.0) - py_unflipped;
    let base_x = px.floor() as isize;
    let base_y = py.floor() as isize;
    let frac_x = px - base_x as f32;
    let frac_y = py - base_y as f32;
    let inv_two_sigma2 = (2.0 * sigma * sigma).recip();
    let mut samples = Vec::new();

    for oy in -radius..=radius {
        for ox in -radius..=radius {
            let x = base_x + ox;
            let y = base_y + oy;
            if x < 0 || y < 0 || x >= size as isize || y >= size as isize {
                continue;
            }
            let dx = ox as f32 - frac_x;
            let dy = oy as f32 - frac_y;
            let g = (-(dx * dx + dy * dy) * inv_two_sigma2).exp();
            samples.push(ParticleSplatSample {
                pixel: y as usize * size + x as usize,
                g,
                dx,
                dy,
            });
        }
    }
    samples
}

fn splat_sigma_pixels(pixel_size: f32, cfg: Target2dLossConfig) -> f32 {
    cfg.sigma * cfg.image_size as f32 * pixel_size / (cfg.hi - cfg.lo)
}

fn splat_norm_scale(pixel_size: f32, cfg: Target2dLossConfig) -> f32 {
    (cfg.image_size as f32 * pixel_size / (cfg.hi - cfg.lo)).powi(2)
}

fn l1l2(value: f32) -> f32 {
    value.abs() + value * value
}

fn l1l2_grad(value: f32) -> f32 {
    value.signum() + 2.0 * value
}

fn overflow_regularizer_and_adjoint(states: &[f32], adjoint: &mut [f32], weight: f32) -> f32 {
    if states.is_empty() || weight == 0.0 {
        return 0.0;
    }
    let denom = states.len() as f32;
    let mut loss = 0.0_f32;
    for (value, gradient) in states.iter().zip(adjoint.iter_mut()) {
        let clipped = value.clamp(-1.0, 1.0);
        let diff = value - clipped;
        loss += diff.abs() / denom;
        if diff != 0.0 {
            *gradient += weight * diff.signum() / denom;
        }
    }
    loss
}

fn bound_regularizer_and_adjoint(
    positions: &[[f32; 4]],
    adjoint: &mut [[f32; 4]],
    spatial_dims: usize,
    weight: f32,
) -> f32 {
    if positions.is_empty() || weight == 0.0 {
        return 0.0;
    }
    let denom = (positions.len() * spatial_dims).max(1) as f32;
    let mut loss = 0.0_f32;
    for (position, gradient) in positions.iter().zip(adjoint.iter_mut()) {
        for axis in 0..spatial_dims {
            let clipped = position[axis].clamp(-1.0, 1.0);
            let diff = position[axis] - clipped;
            loss += diff.abs() / denom;
            if diff != 0.0 {
                gradient[axis] += weight * diff.signum() / denom;
            }
        }
    }
    loss
}

struct ParticlePool2d {
    positions: Vec<[f32; 4]>,
    states: Vec<f32>,
    pool_size: usize,
    particle_count: usize,
    state_dims: usize,
}

struct PoolBatch {
    indices: Vec<usize>,
    positions: Vec<[f32; 4]>,
    states: Vec<f32>,
}

impl ParticlePool2d {
    fn new(
        model: &NpaModel,
        pool_size: usize,
        particle_count: usize,
        cfg: &Target2dTrainingConfig,
    ) -> Self {
        let (positions, states) = seed_particles_scaled(
            pool_size,
            particle_count,
            model.config.state_dims,
            model.config.spatial_dims,
            cfg.seed,
            cfg.seed_mode,
            cfg.seed_scale,
        );
        Self {
            positions,
            states,
            pool_size,
            particle_count,
            state_dims: model.config.state_dims,
        }
    }

    fn sample_batch(
        &self,
        rng: &mut StdRng,
        replace_seed: bool,
        model: &NpaModel,
        cfg: &Target2dTrainingConfig,
    ) -> PoolBatch {
        let mut indices = (0..self.pool_size).collect::<Vec<_>>();
        indices.shuffle(rng);
        indices.truncate(cfg.batch_size);
        let mut positions = Vec::with_capacity(cfg.batch_size * self.particle_count);
        let mut states = Vec::with_capacity(cfg.batch_size * self.particle_count * self.state_dims);
        for index in &indices {
            let row_start = index * self.particle_count;
            let row_end = row_start + self.particle_count;
            positions.extend_from_slice(&self.positions[row_start..row_end]);
            let state_start = row_start * self.state_dims;
            let state_end = row_end * self.state_dims;
            states.extend_from_slice(&self.states[state_start..state_end]);
        }
        if replace_seed {
            let seed = cfg.seed ^ rng.random::<u64>();
            let (seed_positions, seed_states) = seed_particles_scaled(
                1,
                self.particle_count,
                model.config.state_dims,
                model.config.spatial_dims,
                seed,
                cfg.seed_mode,
                cfg.seed_scale,
            );
            positions[..self.particle_count].copy_from_slice(&seed_positions);
            states[..self.particle_count * self.state_dims].copy_from_slice(&seed_states);
        }
        if cfg.brush_size > 0.0 {
            apply_brush_damage(
                &positions,
                &mut states,
                cfg.batch_size,
                self.particle_count,
                self.state_dims,
                cfg.brush_size,
                rng,
            );
        }
        PoolBatch {
            indices,
            positions,
            states,
        }
    }

    fn update_batch(
        &mut self,
        indices: &[usize],
        positions: &[[f32; 4]],
        states: &[f32],
        particle_count: usize,
    ) {
        for (batch, index) in indices.iter().copied().enumerate() {
            let dst_row = index * self.particle_count;
            let src_row = batch * particle_count;
            self.positions[dst_row..dst_row + self.particle_count]
                .copy_from_slice(&positions[src_row..src_row + particle_count]);
            let dst_state = dst_row * self.state_dims;
            let src_state = src_row * self.state_dims;
            self.states[dst_state..dst_state + self.particle_count * self.state_dims]
                .copy_from_slice(&states[src_state..src_state + particle_count * self.state_dims]);
        }
    }
}

fn apply_brush_damage(
    positions: &[[f32; 4]],
    states: &mut [f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    brush_size: f32,
    rng: &mut StdRng,
) {
    for batch in 0..batch_size {
        let center_idx = batch * particle_count + rng.random_range(0..particle_count);
        let center = positions[center_idx];
        let brush2 = brush_size * brush_size;
        for particle in 0..particle_count {
            let row = batch * particle_count + particle;
            let dx = positions[row][0] - center[0];
            let dy = positions[row][1] - center[1];
            if dx * dx + dy * dy < brush2 {
                let state_base = row * state_dims;
                states[state_base..state_base + state_dims].fill(0.0);
            }
        }
    }
}

fn sample_rollout_steps(rng: &mut StdRng, step_min: usize, step_max: usize) -> usize {
    if step_min == step_max {
        step_min
    } else {
        rng.random_range(step_min..step_max)
    }
}

#[allow(clippy::too_many_arguments)]
fn rollout_for_training(
    model: &NpaModel,
    grid: &HashGridConfig,
    mut positions: Vec<[f32; 4]>,
    mut states: Vec<f32>,
    batch_size: usize,
    particle_count: usize,
    steps: usize,
    update_prob: f32,
    seed: u64,
) -> AutomataResult<RolloutForTraining> {
    let mut rng = StdRng::seed_from_u64(seed ^ 0x005e_ed2d);
    let mut snapshots = Vec::with_capacity(steps);
    let mut mean_dx_norm_sum = 0.0_f32;

    for _ in 0..steps {
        let mask = stochastic_mask(batch_size * particle_count, update_prob, &mut rng);
        let perception =
            perceive_for_model(model, &positions, &states, batch_size, particle_count, grid)?;
        let raw_update = model.forward_update_from_features(&perception.features)?;
        let (dx, ds) = update_to_dx_ds(model, &raw_update, grid.eps)?;
        mean_dx_norm_sum += mean_dx_norm(&dx, model.config.spatial_dims);
        snapshots.push(RolloutSnapshot {
            positions: positions.clone(),
            states: states.clone(),
            mask: mask.clone(),
        });
        let stepped = euler_step(
            &positions,
            &states,
            &dx,
            &ds,
            batch_size,
            particle_count,
            model.config.state_dims,
            grid,
            1.0,
            Some(&mask),
        )?;
        positions = stepped.0;
        states = stepped.1;
    }

    Ok(RolloutForTraining {
        snapshots,
        final_positions: positions,
        final_states: states,
        mean_dx_norm_sum,
        steps,
        batch_size,
        particle_count,
    })
}

fn perceive_for_model(
    model: &NpaModel,
    positions: &[[f32; 4]],
    states: &[f32],
    batch_size: usize,
    particle_count: usize,
    grid: &HashGridConfig,
) -> AutomataResult<burn_automata_kernels::PerceptionOutput> {
    Ok(crate::kernels::perceive_with_options(
        positions,
        states,
        batch_size,
        particle_count,
        model.config.state_dims,
        grid,
        perception_options(model, grid),
    )?)
}

fn perception_options(model: &NpaModel, _grid: &HashGridConfig) -> PerceptionOptions {
    PerceptionOptions {
        state_grad: model.config.state_grad,
        density_grad: model.config.density_grad,
        eps0: model.config.eps0,
        scale_equivariance: model.config.scale_equivariant(),
        particle_density_equivariance: model.config.particle_density_equivariant(),
        log_norm_grad: model.config.log_norm_grad,
        log_norm_density_grad: model.config.log_norm_density_grad,
        hybrid_state_gradient: true,
        position_features: model.config.position_features,
    }
}

fn update_to_dx_ds(
    model: &NpaModel,
    raw_update: &[f32],
    grid_eps: f32,
) -> AutomataResult<(Vec<[f32; 4]>, Vec<f32>)> {
    let output_dims = model.config.update_dims();
    if !raw_update.len().is_multiple_of(output_dims) {
        return Err(AutomataError::InvalidArgument(
            "raw update length is not divisible by output dims".to_string(),
        ));
    }
    let rows = raw_update.len() / output_dims;
    let mut dx = vec![[0.0_f32; 4]; rows];
    let mut ds = vec![0.0_f32; rows * model.config.state_dims];
    let motion_eps = model.config.motion_eps(grid_eps);
    for (row, dx_row) in dx.iter_mut().enumerate().take(rows) {
        let base = row * output_dims;
        let mut norm = 0.0_f32;
        for axis in 0..model.config.spatial_dims {
            let value = raw_update[base + axis];
            norm += value * value;
        }
        norm = norm.sqrt();
        for axis in 0..model.config.spatial_dims {
            dx_row[axis] = model.config.alpha * raw_update[base + axis] * motion_eps / (1.0 + norm);
        }
        let state_base = row * model.config.state_dims;
        ds[state_base..state_base + model.config.state_dims].copy_from_slice(
            &raw_update[base + model.config.spatial_dims
                ..base + model.config.spatial_dims + model.config.state_dims],
        );
    }
    Ok((dx, ds))
}

fn mean_dx_norm(dx: &[[f32; 4]], spatial_dims: usize) -> f32 {
    dx.iter()
        .map(|delta| {
            delta
                .iter()
                .take(spatial_dims)
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt()
        })
        .sum::<f32>()
        / dx.len().max(1) as f32
}

fn bptt_gradients(
    model: &NpaModel,
    grid: &HashGridConfig,
    rollout: &RolloutForTraining,
    loss: &Target2dLossOutput,
    loss_cfg: Target2dLossConfig,
) -> AutomataResult<SupervisedGradients> {
    let input_dims = model.config.perception_dims();
    let output_dims = model.config.update_dims();
    let rows = rollout.final_positions.len();
    let mut position_adjoint = loss.position_gradients.clone();
    let mut state_adjoint = loss.state_gradients.clone();
    let mut total = SupervisedGradients {
        w1: vec![0.0; model.weights.w1.len()],
        b1: vec![0.0; model.weights.b1.len()],
        w2: vec![0.0; model.weights.w2.len()],
        b2: vec![0.0; model.weights.b2.len()],
        features: vec![0.0; rows * input_dims],
    };
    for snapshot in rollout.snapshots.iter().rev() {
        let perception = perceive_for_model(
            model,
            &snapshot.positions,
            &snapshot.states,
            rollout.batch_size,
            rollout.particle_count,
            grid,
        )?;
        let raw_update = model.forward_update_from_features(&perception.features)?;
        let mut output_adjoint = vec![0.0_f32; rows * output_dims];
        let displacement_scale = if loss_cfg.displacement_regularizer_weight == 0.0 {
            0.0
        } else {
            loss_cfg.displacement_regularizer_weight / rows.max(1) as f32
        };
        for row in 0..rows {
            let mask = snapshot.mask[row];
            let mut dx_adjoint = [0.0_f32; 4];
            for axis in 0..model.config.spatial_dims {
                dx_adjoint[axis] = position_adjoint[row][axis] * mask;
            }
            add_dx_norm_regularizer_adjoint(
                &raw_update[row * output_dims..row * output_dims + model.config.spatial_dims],
                model,
                grid.eps,
                displacement_scale,
                &mut output_adjoint
                    [row * output_dims..row * output_dims + model.config.spatial_dims],
            );
            raw_motion_adjoint(
                &raw_update[row * output_dims..row * output_dims + model.config.spatial_dims],
                &dx_adjoint,
                model,
                grid.eps,
                &mut output_adjoint
                    [row * output_dims..row * output_dims + model.config.spatial_dims],
            );
            let state_base = row * model.config.state_dims;
            let output_state_base = row * output_dims + model.config.spatial_dims;
            for channel in 0..model.config.state_dims {
                output_adjoint[output_state_base + channel] +=
                    state_adjoint[state_base + channel] * mask;
            }
        }

        let step_grads =
            mlp_backward_from_output_gradients(model, &perception.features, &output_adjoint)?;
        add_gradients(&mut total, &step_grads);
        let perception_adjoint = perceive_adjoint_with_options(
            &snapshot.positions,
            &snapshot.states,
            rollout.batch_size,
            rollout.particle_count,
            model.config.state_dims,
            grid,
            perception_options(model, grid),
            &step_grads.features,
        )?;
        let mut prev_position_adjoint = position_adjoint;
        let mut prev_state_adjoint = state_adjoint;
        if !model.config.stopgrad_pos {
            for (left, right) in prev_position_adjoint
                .iter_mut()
                .zip(perception_adjoint.position.iter())
            {
                for axis in 0..model.config.spatial_dims {
                    left[axis] += right[axis];
                }
            }
        }
        if !model.config.stopgrad_state {
            for (left, right) in prev_state_adjoint
                .iter_mut()
                .zip(perception_adjoint.state.iter())
            {
                *left += *right;
            }
        }
        position_adjoint = prev_position_adjoint;
        state_adjoint = prev_state_adjoint;
    }

    Ok(total)
}

fn raw_motion_adjoint(
    raw_motion: &[f32],
    dx_adjoint: &[f32; 4],
    model: &NpaModel,
    grid_eps: f32,
    out: &mut [f32],
) {
    let dims = model.config.spatial_dims;
    let norm = raw_motion
        .iter()
        .take(dims)
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    let denom = 1.0 + norm;
    let scale = model.config.alpha * model.config.motion_eps(grid_eps);
    let dot = raw_motion
        .iter()
        .zip(dx_adjoint.iter())
        .take(dims)
        .map(|(raw, adjoint)| raw * adjoint)
        .sum::<f32>();
    for axis in 0..dims {
        let mut grad = dx_adjoint[axis] / denom;
        if norm > 0.0 {
            grad -= raw_motion[axis] * dot / (norm * denom * denom);
        }
        out[axis] += scale * grad;
    }
}

fn add_dx_norm_regularizer_adjoint(
    raw_motion: &[f32],
    model: &NpaModel,
    grid_eps: f32,
    weight: f32,
    out: &mut [f32],
) {
    if weight == 0.0 {
        return;
    }
    let mut dx = [0.0_f32; 4];
    let dims = model.config.spatial_dims;
    let norm = raw_motion
        .iter()
        .take(dims)
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    let scale = model.config.alpha * model.config.motion_eps(grid_eps);
    for axis in 0..dims {
        dx[axis] = scale * raw_motion[axis] / (1.0 + norm);
    }
    let dx_norm = dx
        .iter()
        .take(dims)
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if dx_norm == 0.0 {
        return;
    }
    let mut dx_adjoint = [0.0_f32; 4];
    for axis in 0..dims {
        dx_adjoint[axis] = weight * dx[axis] / dx_norm;
    }
    raw_motion_adjoint(raw_motion, &dx_adjoint, model, grid_eps, out);
}

fn add_gradients(left: &mut SupervisedGradients, right: &SupervisedGradients) {
    add_slice(&mut left.w1, &right.w1);
    add_slice(&mut left.b1, &right.b1);
    add_slice(&mut left.w2, &right.w2);
    add_slice(&mut left.b2, &right.b2);
}

fn add_slice(left: &mut [f32], right: &[f32]) {
    for (left_value, right_value) in left.iter_mut().zip(right) {
        *left_value += *right_value;
    }
}

fn normalize_gradient_tensors(grads: &mut SupervisedGradients) {
    normalize_gradient_tensor(&mut grads.w1);
    normalize_gradient_tensor(&mut grads.b1);
    normalize_gradient_tensor(&mut grads.w2);
    normalize_gradient_tensor(&mut grads.b2);
}

fn normalize_gradient_tensor(values: &mut [f32]) {
    for value in values.iter_mut() {
        if !value.is_finite() {
            *value = 0.0;
        }
    }
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    let scale = (norm + 1.0e-8).recip();
    for value in values {
        *value *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_image_extracts_foreground_with_y_up_coordinates() {
        let rgba = vec![
            1.0, 0.0, 0.0, 1.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 1.0,
        ];
        let target =
            TargetImage2d::from_rgba_pixels(2, 2, &rgba, TargetImage2dExtractConfig::default())
                .unwrap();

        assert_eq!(target.point_count(), 2);
        assert_eq!(target.positions[0], [-0.5, 0.5]);
        assert_eq!(target.positions[1], [0.5, -0.5]);
        assert_eq!(target.colors[0], [1.0, 0.0, 0.0]);
        assert_eq!(target.colors[1], [0.0, 1.0, 0.0]);
    }

    #[test]
    fn upstream_growing_hashgrid_matches_growing_yaml() {
        let grid = upstream_growing_2d_hashgrid();
        let (_, preset_grid) = NpaConfig::for_preset(crate::AutomataPreset::Growing2d);

        assert_eq!(grid.boundary, burn_automata_kernels::Boundary::Clamped);
        assert_eq!(grid.mode, burn_automata_kernels::HashGridMode::Grid);
        assert_eq!(grid.grid_size, [16, 16, 1]);
        assert_eq!(grid.eps, 0.1);
        assert_eq!(grid.max_particles_per_block, 32);
        assert_eq!(preset_grid, grid);
    }

    #[test]
    fn public_target_splat_render_matches_loss_image_shape() {
        let target = single_point_target();
        let cfg = Target2dLossConfig {
            image_size: 8,
            ..finite_difference_loss_config()
        };
        let render = render_target_2d_splat(&target, cfg).unwrap();

        assert_eq!(render.rgb.len(), cfg.image_size * cfg.image_size * 3);
        assert_eq!(render.density.len(), cfg.image_size * cfg.image_size);
        assert!(render.rgb.iter().all(|value| value.is_finite()));
        assert!(render.density.iter().all(|value| value.is_finite()));
        assert!(render.density.iter().copied().sum::<f32>() > 0.0);
    }

    #[test]
    fn target_splat_adjoint_matches_position_finite_difference() {
        let target = single_point_target();
        let cfg = finite_difference_loss_config();
        let mut positions = vec![[0.13, -0.07, 0.0, 0.0]];
        let states = vec![0.20, -0.30, 0.10];
        let loss = target_2d_loss_with_adjoint(&positions, &states, 1, 1, 3, &target, cfg, 0.0, 0)
            .unwrap();
        let eps = 1.0e-3;
        positions[0][0] += eps;
        let plus = target_2d_loss(&positions, &states, 1, 1, 3, &target, cfg)
            .unwrap()
            .total_loss;
        positions[0][0] -= 2.0 * eps;
        let minus = target_2d_loss(&positions, &states, 1, 1, 3, &target, cfg)
            .unwrap()
            .total_loss;
        let finite = (plus - minus) / (2.0 * eps);

        assert!(
            (loss.position_gradients[0][0] - finite).abs() < 2.0e-2,
            "analytic={} finite={finite}",
            loss.position_gradients[0][0]
        );
    }

    #[test]
    fn target_splat_adjoint_matches_tail_color_finite_difference() {
        let target = single_point_target();
        let cfg = finite_difference_loss_config();
        let positions = vec![[0.13, -0.07, 0.0, 0.0]];
        let mut states = vec![0.20, -0.30, 0.10];
        let loss = target_2d_loss_with_adjoint(&positions, &states, 1, 1, 3, &target, cfg, 0.0, 0)
            .unwrap();
        let eps = 1.0e-3;
        states[0] += eps;
        let plus = target_2d_loss(&positions, &states, 1, 1, 3, &target, cfg)
            .unwrap()
            .total_loss;
        states[0] -= 2.0 * eps;
        let minus = target_2d_loss(&positions, &states, 1, 1, 3, &target, cfg)
            .unwrap()
            .total_loss;
        let finite = (plus - minus) / (2.0 * eps);

        assert!(
            (loss.state_gradients[0] - finite).abs() < 2.0e-2,
            "analytic={} finite={finite}",
            loss.state_gradients[0]
        );
    }

    #[test]
    fn bptt_weight_gradients_match_rollout_finite_difference() {
        let target = single_point_target();
        let grid = upstream_growing_2d_hashgrid();
        let model = upstream_growing_2d_model(11);
        let loss_cfg = finite_difference_loss_config();
        let (positions, states) = seed_particles_scaled(
            1,
            2,
            model.config.state_dims,
            model.config.spatial_dims,
            37,
            ParticleSeed::UniformCircle,
            0.2,
        );
        let rollout = rollout_for_training(
            &model,
            &grid,
            positions.clone(),
            states.clone(),
            1,
            2,
            2,
            1.0,
            91,
        )
        .unwrap();
        let loss = target_2d_loss_with_adjoint(
            &rollout.final_positions,
            &rollout.final_states,
            1,
            2,
            model.config.state_dims,
            &target,
            loss_cfg,
            rollout.mean_dx_norm_sum,
            rollout.steps,
        )
        .unwrap();
        let grads = bptt_gradients(&model, &grid, &rollout, &loss, loss_cfg).unwrap();

        assert_rollout_gradient_matches_finite_difference(
            &model,
            &grid,
            &target,
            loss_cfg,
            &positions,
            &states,
            GradientParam::B2(largest_abs_index(&grads.b2)),
            &grads,
        );
        assert_rollout_gradient_matches_finite_difference(
            &model,
            &grid,
            &target,
            loss_cfg,
            &positions,
            &states,
            GradientParam::W2(largest_abs_index(&grads.w2)),
            &grads,
        );
        assert_rollout_gradient_matches_finite_difference(
            &model,
            &grid,
            &target,
            loss_cfg,
            &positions,
            &states,
            GradientParam::W1(largest_abs_index(&grads.w1)),
            &grads,
        );
    }

    #[test]
    fn target_rollout_loss_gradient_wrapper_returns_finite_gradients() {
        let target = single_point_target();
        let grid = upstream_growing_2d_hashgrid();
        let model = upstream_growing_2d_model(13);
        let (loss, grads) = target_2d_rollout_loss_with_gradients(
            &model,
            &grid,
            &target,
            RolloutConfig {
                batch_size: 1,
                particle_count: 2,
                steps: 2,
                update_prob: 1.0,
                seed: 37,
                seed_scale: 0.2,
                ..RolloutConfig::default()
            },
            ParticleSeed::UniformCircle,
            finite_difference_loss_config(),
            false,
        )
        .unwrap();

        let grad_norm = grads
            .w1
            .iter()
            .chain(grads.b1.iter())
            .chain(grads.w2.iter())
            .chain(grads.b2.iter())
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!(loss.total_loss.is_finite());
        assert_eq!(grads.w1.len(), model.weights.w1.len());
        assert_eq!(grads.b1.len(), model.weights.b1.len());
        assert_eq!(grads.w2.len(), model.weights.w2.len());
        assert_eq!(grads.b2.len(), model.weights.b2.len());
        assert!(grad_norm.is_finite() && grad_norm > 0.0);
    }

    #[test]
    fn target_training_uses_upstream_inclusive_epochs_and_restores_best() {
        let target = single_point_target();
        let mut model = upstream_growing_2d_model(7);
        let cfg = Target2dTrainingConfig {
            epochs: 1,
            repetitions: 2,
            report_interval: 1,
            batch_size: 1,
            pool_size: 2,
            particle_count: 2,
            step_min: 1,
            step_max: 1,
            inject_seed_interval: 1,
            brush_size: 0.0,
            optimizer: AdamWConfig {
                learning_rate: 0.0,
                grad_clip_norm: 0.0,
                ..AdamWConfig::default()
            },
            ..Target2dTrainingConfig::default()
        };
        let loss_cfg = Target2dLossConfig {
            image_size: 8,
            displacement_regularizer_weight: 0.0,
            overflow_regularizer_weight: 0.0,
            bound_regularizer_weight: 0.0,
            ..Target2dLossConfig::default()
        };

        let report = train_target_2d(
            &mut model,
            &upstream_growing_2d_hashgrid(),
            &target,
            cfg.clone(),
            loss_cfg,
        )
        .unwrap();

        assert_eq!(report.epochs_completed, 4);
        assert_eq!(report.repetitions_completed, 2);
        assert_eq!(report.history.len(), 4);
        assert!(report.best_loss.total_loss.is_finite());
        assert!(report.best_eval_loss.total_loss.is_finite());
        assert_eq!(
            report.final_loss.total_loss,
            report.best_eval_loss.total_loss
        );
        assert!(report.history.iter().all(|entry| entry.eval_loss.is_some()));
        let restored_eval = evaluate_seed_rollout_loss(
            &model,
            &upstream_growing_2d_hashgrid(),
            &target,
            &cfg,
            loss_cfg,
        )
        .unwrap();
        assert_eq!(report.final_loss.total_loss, restored_eval.total_loss);
    }

    fn single_point_target() -> TargetImage2d {
        TargetImage2d {
            source_width: 16,
            source_height: 16,
            positions: vec![[0.0, 0.0]],
            colors: vec![[0.7, 0.2, 0.6]],
            pixel_size: 2.0 / 16.0,
            threshold: 0.05,
            aabb: DEFAULT_AABB,
        }
    }

    fn finite_difference_loss_config() -> Target2dLossConfig {
        Target2dLossConfig {
            image_size: 16,
            sigma: 1.0,
            center: false,
            displacement_regularizer_weight: 0.0,
            overflow_regularizer_weight: 0.0,
            bound_regularizer_weight: 0.0,
            ..Target2dLossConfig::default()
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum GradientParam {
        W1(usize),
        W2(usize),
        B2(usize),
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_rollout_gradient_matches_finite_difference(
        model: &NpaModel,
        grid: &HashGridConfig,
        target: &TargetImage2d,
        loss_cfg: Target2dLossConfig,
        positions: &[[f32; 4]],
        states: &[f32],
        param: GradientParam,
        grads: &SupervisedGradients,
    ) {
        let analytic = gradient_value(grads, param);
        let eps = 1.0e-3;
        let mut plus = model.clone();
        perturb_param(&mut plus, param, eps);
        let plus_loss = rollout_total_loss(
            &plus,
            grid,
            target,
            loss_cfg,
            positions.to_vec(),
            states.to_vec(),
        );
        let mut minus = model.clone();
        perturb_param(&mut minus, param, -eps);
        let minus_loss = rollout_total_loss(
            &minus,
            grid,
            target,
            loss_cfg,
            positions.to_vec(),
            states.to_vec(),
        );
        let finite = (plus_loss - minus_loss) / (2.0 * eps);
        let tolerance = 5.0e-2_f32.max(0.08 * finite.abs());

        assert!(
            (analytic - finite).abs() <= tolerance,
            "param={param:?} analytic={analytic} finite={finite} tolerance={tolerance}"
        );
    }

    fn rollout_total_loss(
        model: &NpaModel,
        grid: &HashGridConfig,
        target: &TargetImage2d,
        loss_cfg: Target2dLossConfig,
        positions: Vec<[f32; 4]>,
        states: Vec<f32>,
    ) -> f32 {
        let rollout =
            rollout_for_training(model, grid, positions, states, 1, 2, 2, 1.0, 91).unwrap();
        target_2d_loss_with_adjoint(
            &rollout.final_positions,
            &rollout.final_states,
            1,
            2,
            model.config.state_dims,
            target,
            loss_cfg,
            rollout.mean_dx_norm_sum,
            rollout.steps,
        )
        .unwrap()
        .report
        .total_loss
    }

    fn largest_abs_index(values: &[f32]) -> usize {
        values
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
            .map(|(index, _)| index)
            .unwrap()
    }

    fn gradient_value(grads: &SupervisedGradients, param: GradientParam) -> f32 {
        match param {
            GradientParam::W1(index) => grads.w1[index],
            GradientParam::W2(index) => grads.w2[index],
            GradientParam::B2(index) => grads.b2[index],
        }
    }

    fn perturb_param(model: &mut NpaModel, param: GradientParam, delta: f32) {
        match param {
            GradientParam::W1(index) => model.weights.w1[index] += delta,
            GradientParam::W2(index) => model.weights.w2[index] += delta,
            GradientParam::B2(index) => model.weights.b2[index] += delta,
        }
    }
}
