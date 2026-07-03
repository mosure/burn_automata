use crate::cli::prelude::*;

#[derive(Clone, Debug)]
pub(super) struct Hyper2dSourceDescriptor {
    pub(super) slug: String,
    pub(super) title: Option<String>,
    pub(super) group: Option<String>,
    pub(super) condition_path: PathBuf,
    pub(super) target_path: PathBuf,
    pub(super) particles: Option<usize>,
    pub(super) seed_scale: Option<f32>,
    pub(super) update_prob: Option<f32>,
}

#[derive(Clone, Debug)]
pub(super) struct Hyper2dLoadedExample {
    pub(super) descriptor: Hyper2dSourceDescriptor,
    pub(super) condition: ConditionImage2d,
    pub(super) batch: SupervisedBatch,
    pub(super) rows: usize,
    pub(super) particle_count: usize,
    pub(super) rollout_steps: usize,
    pub(super) rollouts: usize,
    pub(super) update_prob: f32,
    pub(super) seed_scale: f32,
    pub(super) seed_mode: ParticleSeed,
    pub(super) seed: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Hyper2dImageMetricConfig {
    pub(super) image_size: usize,
    pub(super) rollout_steps: usize,
    pub(super) particle_count: Option<usize>,
    pub(super) update_prob: Option<f32>,
    pub(super) sigma: f32,
    pub(super) threshold: f32,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Hyper2dDynamicsMetricConfig {
    pub(super) particle_count: usize,
    pub(super) rollout_steps: usize,
    pub(super) update_prob: Option<f32>,
    pub(super) image_size: usize,
    pub(super) sigma: f32,
}

pub(super) struct Hyper2dAdapterBootstrapResult {
    pub(super) examples: Vec<HyperAdapterExample2d>,
    pub(super) reports: Vec<CliHyper2dAdapterBootstrapReport>,
}

#[derive(Clone, Debug, serde::Deserialize)]
struct SelfOrgCatalogEntry {
    slug: String,
    title: Option<String>,
    group: String,
    preset: String,
    output: PathBuf,
    particles: Option<usize>,
    seed_scale: Option<f32>,
    update_prob: Option<f32>,
}

pub(super) struct ResolveHyper2dSourcesConfig<'a> {
    pub(super) preset: PresetArg,
    pub(super) condition: Option<&'a PathBuf>,
    pub(super) target_model: Option<&'a PathBuf>,
    pub(super) catalog: Option<&'a PathBuf>,
    pub(super) catalog_thumbnail_dir: &'a Path,
    pub(super) catalog_group: Option<Hyper2dCatalogGroupArg>,
    pub(super) catalog_targets: &'a [String],
    pub(super) catalog_limit: usize,
}

pub(super) fn resolve_hyper2d_sources(
    config: ResolveHyper2dSourcesConfig<'_>,
) -> Result<Vec<Hyper2dSourceDescriptor>, Box<dyn std::error::Error>> {
    if let Some(catalog_path) = config.catalog {
        if config.condition.is_some() || config.target_model.is_some() {
            return Err(std::io::Error::other(
                "--catalog cannot be combined with --condition or --target-model",
            )
            .into());
        }
        let text = std::fs::read_to_string(catalog_path)?;
        let entries: Vec<SelfOrgCatalogEntry> = serde_json::from_str(&text)?;
        let mut descriptors = Vec::new();
        for entry in entries {
            if !config.catalog_targets.is_empty()
                && !config
                    .catalog_targets
                    .iter()
                    .any(|target| target == &entry.slug)
            {
                continue;
            }
            if !config.catalog_targets.is_empty()
                || catalog_entry_matches(config.preset, config.catalog_group, &entry)
            {
                let condition_path = config
                    .catalog_thumbnail_dir
                    .join(format!("{}.png", entry.slug));
                descriptors.push(Hyper2dSourceDescriptor {
                    slug: entry.slug,
                    title: entry.title,
                    group: Some(entry.group),
                    condition_path,
                    target_path: resolve_data_path(&entry.output, Some(catalog_path)),
                    particles: entry.particles,
                    seed_scale: entry.seed_scale,
                    update_prob: entry.update_prob,
                });
            }
            if config.catalog_limit > 0 && descriptors.len() >= config.catalog_limit {
                break;
            }
        }
        return Ok(descriptors);
    }

    let condition = config.condition.ok_or_else(|| {
        std::io::Error::other("--condition is required when --catalog is not provided")
    })?;
    let target_model = config.target_model.ok_or_else(|| {
        std::io::Error::other("--target-model is required when --catalog is not provided")
    })?;
    Ok(vec![Hyper2dSourceDescriptor {
        slug: path_slug(condition),
        title: None,
        group: None,
        condition_path: condition.clone(),
        target_path: target_model.clone(),
        particles: None,
        seed_scale: None,
        update_prob: None,
    }])
}

#[allow(clippy::too_many_arguments)]
pub(super) fn load_hyper2d_examples(
    base: &NpaModel,
    base_manifest: &BpkModelManifest,
    descriptors: &[Hyper2dSourceDescriptor],
    rows: usize,
    rollout_particles: Option<usize>,
    rollout_steps: usize,
    rollouts: usize,
    rollout_update_prob: Option<f32>,
    seed_scale: Option<f32>,
    preset: AutomataPreset,
    seed_mode: ParticleSeed,
    seed: u64,
) -> Result<Vec<Hyper2dLoadedExample>, Box<dyn std::error::Error>> {
    let mut examples = Vec::with_capacity(descriptors.len());
    for (idx, descriptor) in descriptors.iter().enumerate() {
        let target_manifest = crate::import::load_manifest(&descriptor.target_path)?;
        if target_manifest.config != base.config {
            return Err(std::io::Error::other(format!(
                "target {} config does not match base model config",
                descriptor.target_path.display()
            ))
            .into());
        }
        if target_manifest.hashgrid != base_manifest.hashgrid {
            return Err(std::io::Error::other(format!(
                "target {} hashgrid does not match base model hashgrid",
                descriptor.target_path.display()
            ))
            .into());
        }
        let target = target_manifest.into_model();
        let actual_seed_scale = seed_scale
            .or(descriptor.seed_scale)
            .unwrap_or_else(|| NpaConfig::seed_scale_for_preset(preset));
        let actual_update_prob = rollout_update_prob
            .or(descriptor.update_prob)
            .unwrap_or(1.0);
        let actual_particles = rollout_particles.or(descriptor.particles).unwrap_or(1024);
        let actual_seed = seed.wrapping_add(idx as u64);
        let condition = load_condition_image_2d(&descriptor.condition_path)?;
        let batch = rollout_supervised_batch_from_model(
            base,
            &target,
            &base_manifest.hashgrid,
            SupervisedTarget::Teacher(&target),
            RolloutSupervisionConfig {
                max_rows: rows,
                particle_count: actual_particles,
                rollout_steps,
                rollouts,
                update_prob: actual_update_prob,
                seed: actual_seed,
                seed_scale: actual_seed_scale,
                seed_mode,
                ..RolloutSupervisionConfig::default()
            },
        )?;
        examples.push(Hyper2dLoadedExample {
            descriptor: descriptor.clone(),
            condition,
            batch,
            rows,
            particle_count: actual_particles,
            rollout_steps,
            rollouts,
            update_prob: actual_update_prob,
            seed_scale: actual_seed_scale,
            seed_mode,
            seed: actual_seed,
        });
    }
    Ok(examples)
}

pub(super) fn split_hyper2d_examples(
    examples: Vec<Hyper2dLoadedExample>,
    holdout_stride: usize,
    holdout_offset: usize,
) -> Result<(Vec<Hyper2dLoadedExample>, Vec<Hyper2dLoadedExample>), Box<dyn std::error::Error>> {
    let mut train = Vec::new();
    let mut holdout = Vec::new();
    for (idx, example) in examples.into_iter().enumerate() {
        if holdout_stride > 0 && idx % holdout_stride == holdout_offset % holdout_stride {
            holdout.push(example);
        } else {
            train.push(example);
        }
    }
    if train.is_empty() {
        return Err(
            std::io::Error::other("train-hyper2d split produced no training examples").into(),
        );
    }
    Ok((train, holdout))
}

pub(super) fn flow_examples(examples: &[Hyper2dLoadedExample]) -> Vec<HyperFlowExample2d> {
    examples
        .iter()
        .map(|example| HyperFlowExample2d {
            condition: example.condition.clone(),
            batch: example.batch.clone(),
        })
        .collect()
}

pub(super) fn bootstrap_hyper2d_adapters(
    base: &NpaModel,
    examples: &[Hyper2dLoadedExample],
    adapter_rank: usize,
    adapter_alpha: f32,
    seed: u64,
    config: TrainingRunConfig,
) -> Result<Hyper2dAdapterBootstrapResult, Box<dyn std::error::Error>> {
    if examples.is_empty() {
        return Err(std::io::Error::other("adapter bootstrap requires training examples").into());
    }
    let mut adapter_examples = Vec::with_capacity(examples.len());
    let mut reports = Vec::with_capacity(examples.len());
    let exact_rank = base.config.perception_dims().max(base.config.update_dims());
    for (idx, example) in examples.iter().enumerate() {
        let initial_adapter = NpaLowRankAdapter::zeros(&base.config, adapter_rank, adapter_alpha);
        let initial_loss = supervised_adapter_loss(base, &initial_adapter, &example.batch)?;
        if adapter_rank >= exact_rank {
            let target_manifest = crate::import::load_manifest(&example.descriptor.target_path)?;
            let target_model = target_manifest.into_model();
            let adapter = NpaLowRankAdapter::exact_model_delta(
                base,
                &target_model,
                adapter_rank,
                adapter_alpha,
            )?;
            let final_loss = supervised_adapter_loss(base, &adapter, &example.batch)?;
            reports.push(CliHyper2dAdapterBootstrapReport {
                slug: example.descriptor.slug.clone(),
                method: "exact-weight-delta",
                steps: 0,
                rows: example.rows,
                initial_loss,
                final_loss,
                best_loss: final_loss,
                adapter_parameter_count: adapter.parameter_count(),
            });
            adapter_examples.push(HyperAdapterExample2d {
                condition: example.condition.clone(),
                target_adapter: adapter,
            });
            continue;
        }

        let mut adapter = NpaLowRankAdapter::seeded(
            &base.config,
            adapter_rank,
            adapter_alpha,
            seed.wrapping_add((idx as u64).wrapping_mul(0x517c_c1b7)),
        );
        let report = run_supervised_adapter_training(base, &mut adapter, &example.batch, config)?;
        reports.push(CliHyper2dAdapterBootstrapReport {
            slug: example.descriptor.slug.clone(),
            method: "sgd",
            steps: report.steps,
            rows: report.rows,
            initial_loss,
            final_loss: report.final_loss,
            best_loss: report.best_loss,
            adapter_parameter_count: adapter.parameter_count(),
        });
        adapter_examples.push(HyperAdapterExample2d {
            condition: example.condition.clone(),
            target_adapter: adapter,
        });
    }
    Ok(Hyper2dAdapterBootstrapResult {
        examples: adapter_examples,
        reports,
    })
}

pub(super) fn initialize_hyper_adapter_residual_fit(
    hyper: &mut HyperNpa2d,
    examples: &[HyperAdapterExample2d],
) -> Result<usize, Box<dyn std::error::Error>> {
    let Some(anchor_input) = hyper.anchor_input.clone() else {
        return Ok(0);
    };
    hyper.validate()?;
    let input_dims = hyper.config.condition_feature_dims;
    let hidden_dims = hyper.config.hidden_dims;
    if anchor_input.len() != input_dims {
        return Err(std::io::Error::other("hyper anchor input dimensions are invalid").into());
    }

    hyper.weights.w1.fill(0.0);
    hyper.weights.b1.fill(0.0);
    hyper.weights.w2.fill(0.0);
    hyper.weights.b2.fill(0.0);

    let mut fitted = 0_usize;
    for example in examples {
        let target = example.target_adapter.to_parameter_vector();
        let max_abs_target = target
            .iter()
            .fold(0.0_f32, |max_value, value| max_value.max(value.abs()));
        if max_abs_target <= 1.0e-8 {
            continue;
        }
        if fitted >= hidden_dims {
            return Err(std::io::Error::other(format!(
                "not enough hyper hidden dims for analytic adapter fit: need more than {hidden_dims}"
            ))
            .into());
        }

        let input = example.condition.feature_vector_with_tokens(
            hyper.config.condition_token_grid_width,
            hyper.config.condition_token_grid_height,
        )?;
        if input.len() != input_dims {
            return Err(std::io::Error::other("condition feature dimensions changed").into());
        }
        let delta = input
            .iter()
            .zip(anchor_input.iter())
            .map(|(input_value, anchor_value)| input_value - anchor_value)
            .collect::<Vec<_>>();
        let norm_sq = delta.iter().map(|value| value * value).sum::<f32>();
        if norm_sq <= f32::EPSILON {
            continue;
        }

        let w1_base = fitted * input_dims;
        let mut anchor_projection = 0.0_f32;
        for (idx, delta_value) in delta.iter().copied().enumerate() {
            let weight = delta_value / norm_sq;
            hyper.weights.w1[w1_base + idx] = weight;
            anchor_projection += weight * anchor_input[idx];
        }
        hyper.weights.b1[fitted] = -anchor_projection;

        for (output_idx, target_value) in target.iter().copied().enumerate() {
            let normalized =
                (target_value / hyper.config.output_scale).clamp(-0.999_999, 0.999_999);
            hyper.weights.w2[output_idx * hidden_dims + fitted] = normalized.atanh();
        }
        fitted += 1;
    }

    hyper.validate()?;
    Ok(fitted)
}

pub(super) fn example_reports(
    base: &NpaModel,
    base_manifest: &BpkModelManifest,
    hyper: &HyperNpa2d,
    examples: &[Hyper2dLoadedExample],
    initial_losses: &[f32],
    image_metric_config: Option<Hyper2dImageMetricConfig>,
    dynamics_metric_config: Option<Hyper2dDynamicsMetricConfig>,
) -> Result<Vec<CliHyper2dExampleReport>, Box<dyn std::error::Error>> {
    if initial_losses.len() != examples.len() {
        return Err(std::io::Error::other("example initial losses do not match examples").into());
    }
    let mut reports = Vec::with_capacity(examples.len());
    for (example, initial_loss) in examples.iter().zip(initial_losses.iter().copied()) {
        let flow = vec![HyperFlowExample2d {
            condition: example.condition.clone(),
            batch: example.batch.clone(),
        }];
        let final_loss = hyper_rectified_flow_loss(base, hyper, &flow)?;
        let summary = example.condition.summary()?;
        let prior =
            ParticlePrior2d::from_summary(&base.config, &summary, ParticlePriorConfig::default())?;
        let image_metrics = image_metric_config
            .map(|config| image_metrics_for_example(base, base_manifest, hyper, example, config))
            .transpose()?;
        let dynamics_metrics = dynamics_metric_config
            .map(|config| dynamics_metrics_for_example(base, base_manifest, hyper, example, config))
            .transpose()?;
        reports.push(CliHyper2dExampleReport {
            slug: example.descriptor.slug.clone(),
            title: example.descriptor.title.clone(),
            group: example.descriptor.group.clone(),
            condition: example.descriptor.condition_path.display().to_string(),
            target_model: example.descriptor.target_path.display().to_string(),
            initial_loss,
            final_loss,
            rows: example.rows,
            particle_count: example.particle_count,
            rollout_steps: example.rollout_steps,
            rollouts: example.rollouts,
            update_prob: example.update_prob,
            seed_scale: example.seed_scale,
            condition_summary: summary,
            prior,
            image_metrics,
            dynamics_metrics,
        });
    }
    Ok(reports)
}

fn dynamics_metrics_for_example(
    base: &NpaModel,
    base_manifest: &BpkModelManifest,
    hyper: &HyperNpa2d,
    example: &Hyper2dLoadedExample,
    config: Hyper2dDynamicsMetricConfig,
) -> Result<CliHyper2dDynamicsMetricsReport, Box<dyn std::error::Error>> {
    if config.particle_count == 0 {
        return Err(
            std::io::Error::other("--dynamics-metric-particles must be greater than zero").into(),
        );
    }
    if config.rollout_steps == 0 {
        return Err(
            std::io::Error::other("--dynamics-metric-steps must be greater than zero").into(),
        );
    }
    if config.image_size == 0 {
        return Err(std::io::Error::other(
            "--dynamics-metric-image-size must be greater than zero",
        )
        .into());
    }
    if !config.sigma.is_finite() || config.sigma <= 0.0 {
        return Err(std::io::Error::other(
            "--dynamics-metric-sigma must be finite and greater than zero",
        )
        .into());
    }
    let update_prob = config.update_prob.unwrap_or(example.update_prob);
    if !(0.0..=1.0).contains(&update_prob) || !update_prob.is_finite() {
        return Err(std::io::Error::other(
            "--dynamics-metric-update-prob must be finite and in [0, 1]",
        )
        .into());
    }

    let target_manifest = crate::import::load_manifest(&example.descriptor.target_path)?;
    if target_manifest.config != base.config {
        return Err(std::io::Error::other(format!(
            "target {} config does not match base model config",
            example.descriptor.target_path.display()
        ))
        .into());
    }
    if target_manifest.hashgrid != base_manifest.hashgrid {
        return Err(std::io::Error::other(format!(
            "target {} hashgrid does not match base model hashgrid",
            example.descriptor.target_path.display()
        ))
        .into());
    }
    let target_model = target_manifest.into_model();
    let conditioned = generate_conditioned_npa_2d(
        base,
        hyper,
        &example.condition,
        ParticlePriorConfig::default(),
    )?;
    let rollout_cfg = RolloutConfig {
        steps: config.rollout_steps,
        particle_count: config.particle_count,
        update_prob,
        seed: example.seed,
        seed_scale: example.seed_scale,
        ..RolloutConfig::default()
    };
    let target_trace = run_rollout(
        &target_model,
        &base_manifest.hashgrid,
        &rollout_cfg,
        example.seed_mode,
    )?;
    let generated_trace = run_rollout(
        &conditioned.model,
        &base_manifest.hashgrid,
        &rollout_cfg,
        example.seed_mode,
    )?;

    let target_positions = flatten_positions(&target_trace.positions);
    let generated_positions = flatten_positions(&generated_trace.positions);
    let position_stats = compare_dynamic_signal(&generated_positions, &target_positions)?;
    let state_stats = compare_dynamic_signal(&generated_trace.states, &target_trace.states)?;
    let target_tail = tail_rgb_values(&target_trace.states, target_trace.state_dims)?;
    let generated_tail = tail_rgb_values(&generated_trace.states, generated_trace.state_dims)?;
    let tail_stats = compare_unit_signal(&generated_tail, &target_tail)?;
    let target_render = rasterize_tail_rgb_gaussian(
        &target_trace.positions,
        &target_trace.states,
        target_trace.state_dims,
        &base_manifest.hashgrid,
        config.image_size,
        config.sigma,
    )?;
    let generated_render = rasterize_tail_rgb_gaussian(
        &generated_trace.positions,
        &generated_trace.states,
        generated_trace.state_dims,
        &base_manifest.hashgrid,
        config.image_size,
        config.sigma,
    )?;
    let render_stats = compare_rgb_images(&generated_render.rgb, &target_render.rgb)?;
    let (mean_dx_mse, mean_dx_mae) =
        compare_mean_dx(&generated_trace.mean_dx, &target_trace.mean_dx)?;

    Ok(CliHyper2dDynamicsMetricsReport {
        particle_count: config.particle_count,
        rollout_steps: config.rollout_steps,
        update_prob,
        seed: example.seed,
        seed_scale: example.seed_scale,
        seed_mode: example.seed_mode,
        image_size: config.image_size,
        render_sigma_px: config.sigma,
        position_mse: position_stats.mse,
        position_psnr_db: position_stats.psnr_db,
        state_mse: state_stats.mse,
        state_psnr_db: state_stats.psnr_db,
        tail_rgb_mse: tail_stats.mse,
        tail_rgb_psnr_db: tail_stats.psnr_db,
        render_rgb_mse: render_stats.mse,
        render_rgb_psnr_db: render_stats.psnr_db,
        mean_dx_mse,
        mean_dx_mae,
        target_final_mean_dx: target_trace.mean_dx.last().copied().unwrap_or_default(),
        generated_final_mean_dx: generated_trace.mean_dx.last().copied().unwrap_or_default(),
    })
}

fn image_metrics_for_example(
    base: &NpaModel,
    base_manifest: &BpkModelManifest,
    hyper: &HyperNpa2d,
    example: &Hyper2dLoadedExample,
    config: Hyper2dImageMetricConfig,
) -> Result<CliHyper2dImageMetricsReport, Box<dyn std::error::Error>> {
    if config.image_size == 0 {
        return Err(std::io::Error::other("--image-metric-size must be greater than zero").into());
    }
    if config.threshold < 0.0 || !config.threshold.is_finite() {
        return Err(
            std::io::Error::other("--image-metric-threshold must be finite and >= 0").into(),
        );
    }
    let particle_count = config.particle_count.unwrap_or(example.particle_count);
    if particle_count == 0 {
        return Err(
            std::io::Error::other("--image-metric-particles must be greater than zero").into(),
        );
    }
    let update_prob = config.update_prob.unwrap_or(example.update_prob);
    if !(0.0..=1.0).contains(&update_prob) || !update_prob.is_finite() {
        return Err(std::io::Error::other(
            "--image-metric-update-prob must be finite and in [0, 1]",
        )
        .into());
    }
    if !config.sigma.is_finite() || config.sigma <= 0.0 {
        return Err(std::io::Error::other(
            "--image-metric-sigma must be finite and greater than zero",
        )
        .into());
    }

    let conditioned = generate_conditioned_npa_2d(
        base,
        hyper,
        &example.condition,
        ParticlePriorConfig::default(),
    )?;
    let rollout = run_rollout(
        &conditioned.model,
        &base_manifest.hashgrid,
        &RolloutConfig {
            steps: config.rollout_steps,
            particle_count,
            update_prob,
            seed: example.seed,
            seed_scale: example.seed_scale,
            ..RolloutConfig::default()
        },
        example.seed_mode,
    )?;
    let rendered = rasterize_tail_rgb_gaussian(
        &rollout.positions,
        &rollout.states,
        rollout.state_dims,
        &base_manifest.hashgrid,
        config.image_size,
        config.sigma,
    )?;
    let target_rgb = condition_rgb_resampled(&example.condition, config.image_size)?;
    let color_stats = compare_rgb_images(&rendered.rgb, &target_rgb)?;
    let rendered_luma = rgb_to_luma_image(&rendered.rgb);
    let target_luma = rgb_to_luma_image(&target_rgb);
    let luma_stats = compare_scalar_images(&rendered_luma, &target_luma)?;
    let generated_occupancy = rasterize_grid_occupancy(
        &rollout.positions,
        &base_manifest.hashgrid,
        config.image_size,
    );
    let occupancy_stats =
        compare_occupancy_images(&generated_occupancy, &target_luma, config.threshold)?;
    let domain_radius = grid_domain_radius(&base_manifest.hashgrid);

    Ok(CliHyper2dImageMetricsReport {
        image_size: config.image_size,
        rollout_steps: config.rollout_steps,
        particle_count,
        update_prob,
        seed: example.seed,
        seed_scale: example.seed_scale,
        seed_mode: example.seed_mode,
        decoder: "tail-rgb-plus-half-gaussian-splat",
        render_sigma_px: config.sigma,
        domain_radius,
        mse: color_stats.mse,
        psnr_db: color_stats.psnr_db,
        luma_mse: luma_stats.mse,
        luma_psnr_db: luma_stats.psnr_db,
        occupancy_mse: occupancy_stats.mse,
        occupancy_psnr_db: occupancy_stats.psnr_db,
        foreground_iou: occupancy_stats.foreground_iou,
        generated_occupancy: occupancy_stats.generated_occupancy,
        target_occupancy: occupancy_stats.target_occupancy,
    })
}

#[derive(Clone, Copy, Debug)]
struct Hyper2dImageMetricStats {
    mse: f32,
    psnr_db: f32,
    foreground_iou: f32,
    generated_occupancy: f32,
    target_occupancy: f32,
}

#[derive(Debug)]
struct Hyper2dRenderedImage {
    rgb: Vec<f32>,
}

fn grid_domain_radius(grid: &burn_automata_kernels::HashGridConfig) -> f32 {
    let extent_x = grid.grid_size[0] as f32 * grid.eps;
    let extent_y = grid.grid_size[1] as f32 * grid.eps;
    (extent_x.max(extent_y) * 0.5).max(1.0e-6)
}

fn rasterize_tail_rgb_gaussian(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    grid: &burn_automata_kernels::HashGridConfig,
    image_size: usize,
    sigma: f32,
) -> Result<Hyper2dRenderedImage, Box<dyn std::error::Error>> {
    if state_dims < 3 {
        return Err(std::io::Error::other(
            "tail-rgb image metrics require at least three state channels",
        )
        .into());
    }
    if states.len() < positions.len().saturating_mul(state_dims) {
        return Err(std::io::Error::other("rollout state buffer is shorter than positions").into());
    }
    if image_size == 0 {
        return Err(std::io::Error::other("image metric size must be greater than zero").into());
    }
    let (extent_x, extent_y) = grid_extents(grid)?;
    let half_x = extent_x * 0.5;
    let half_y = extent_y * 0.5;
    let radius = (sigma * 3.0).ceil().max(1.0) as isize;
    let inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma);
    let max_pixel = image_size.saturating_sub(1) as f32;
    let mut rgb = vec![0.0_f32; image_size * image_size * 3];
    let mut weights = vec![0.0_f32; image_size * image_size];

    for (row, position) in positions.iter().enumerate() {
        let px = (position[0] + half_x) / extent_x * max_pixel;
        let py = (position[1] + half_y) / extent_y * max_pixel;
        if !px.is_finite() || !py.is_finite() {
            continue;
        }
        let state_base = row * state_dims;
        let color = [
            (states[state_base + state_dims - 3] + 0.5).clamp(0.0, 1.0),
            (states[state_base + state_dims - 2] + 0.5).clamp(0.0, 1.0),
            (states[state_base + state_dims - 1] + 0.5).clamp(0.0, 1.0),
        ];
        let raw_min_x = px.floor() as isize - radius;
        let raw_max_x = px.ceil() as isize + radius;
        let raw_min_y = py.floor() as isize - radius;
        let raw_max_y = py.ceil() as isize + radius;
        let image_limit = image_size as isize - 1;
        if raw_max_x < 0 || raw_max_y < 0 || raw_min_x > image_limit || raw_min_y > image_limit {
            continue;
        }
        let min_x = raw_min_x.max(0) as usize;
        let max_x = raw_max_x.min(image_limit) as usize;
        let min_y = raw_min_y.max(0) as usize;
        let max_y = raw_max_y.min(image_limit) as usize;
        for y in min_y..=max_y {
            let dy = y as f32 - py;
            for x in min_x..=max_x {
                let dx = x as f32 - px;
                let weight = (-(dx * dx + dy * dy) * inv_two_sigma2).exp();
                let pixel = y * image_size + x;
                weights[pixel] += weight;
                let rgb_base = pixel * 3;
                rgb[rgb_base] += color[0] * weight;
                rgb[rgb_base + 1] += color[1] * weight;
                rgb[rgb_base + 2] += color[2] * weight;
            }
        }
    }
    for (pixel, weight) in weights.iter().enumerate() {
        if *weight <= 0.0 {
            continue;
        }
        let rgb_base = pixel * 3;
        rgb[rgb_base] /= *weight;
        rgb[rgb_base + 1] /= *weight;
        rgb[rgb_base + 2] /= *weight;
    }
    Ok(Hyper2dRenderedImage { rgb })
}

fn rasterize_grid_occupancy(
    positions: &[[f32; 4]],
    grid: &burn_automata_kernels::HashGridConfig,
    image_size: usize,
) -> Vec<f32> {
    let mut image = vec![0.0_f32; image_size * image_size];
    if image_size == 0 {
        return image;
    }
    let Ok((extent_x, extent_y)) = grid_extents(grid) else {
        return image;
    };
    let half_x = extent_x * 0.5;
    let half_y = extent_y * 0.5;
    let max_pixel = image_size.saturating_sub(1) as f32;
    for position in positions {
        let px = (position[0] + half_x) / extent_x * max_pixel;
        let py = (position[1] + half_y) / extent_y * max_pixel;
        if !px.is_finite() || !py.is_finite() {
            continue;
        }
        let x = px.clamp(0.0, max_pixel).round() as usize;
        let y = py.clamp(0.0, max_pixel).round() as usize;
        image[y * image_size + x] = 1.0;
    }
    image
}

fn grid_extents(
    grid: &burn_automata_kernels::HashGridConfig,
) -> Result<(f32, f32), Box<dyn std::error::Error>> {
    let extent_x = grid.grid_size[0] as f32 * grid.eps;
    let extent_y = grid.grid_size[1] as f32 * grid.eps;
    if !extent_x.is_finite() || !extent_y.is_finite() || extent_x <= 0.0 || extent_y <= 0.0 {
        return Err(std::io::Error::other(
            "image metric hashgrid extent must be finite and positive",
        )
        .into());
    }
    Ok((extent_x, extent_y))
}

fn condition_rgb_resampled(
    condition: &ConditionImage2d,
    image_size: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if image_size == 0 {
        return Err(std::io::Error::other("image metric size must be greater than zero").into());
    }
    condition.validate()?;
    let mut rgb = Vec::with_capacity(image_size * image_size * 3);
    for y in 0..image_size {
        let source_y = y * condition.height / image_size;
        for x in 0..image_size {
            let source_x = x * condition.width / image_size;
            rgb.extend_from_slice(&condition_rgb_at(condition, source_x, source_y));
        }
    }
    Ok(rgb)
}

fn condition_rgb_at(condition: &ConditionImage2d, x: usize, y: usize) -> [f32; 3] {
    let offset = (y * condition.width + x) * condition.channels;
    match condition.channels {
        1 => [condition.values[offset]; 3],
        _ => [
            condition.values[offset],
            condition.values[offset + 1],
            condition.values[offset + 2],
        ],
    }
}

fn rgb_to_luma_image(rgb: &[f32]) -> Vec<f32> {
    rgb.chunks_exact(3)
        .map(|pixel| 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2])
        .collect()
}

fn compare_rgb_images(
    generated: &[f32],
    target: &[f32],
) -> Result<Hyper2dImageMetricStats, Box<dyn std::error::Error>> {
    compare_scalar_images(generated, target)
}

fn compare_unit_signal(
    generated: &[f32],
    target: &[f32],
) -> Result<Hyper2dImageMetricStats, Box<dyn std::error::Error>> {
    compare_signal_with_peak(generated, target, 1.0)
}

fn compare_dynamic_signal(
    generated: &[f32],
    target: &[f32],
) -> Result<Hyper2dImageMetricStats, Box<dyn std::error::Error>> {
    if generated.len() != target.len() {
        return Err(std::io::Error::other("generated and target signal sizes differ").into());
    }
    let peak = generated
        .iter()
        .chain(target)
        .fold(1.0_f32, |peak, value| peak.max(value.abs()));
    compare_signal_with_peak(generated, target, peak)
}

fn compare_scalar_images(
    generated: &[f32],
    target: &[f32],
) -> Result<Hyper2dImageMetricStats, Box<dyn std::error::Error>> {
    compare_signal_with_peak(generated, target, 1.0)
}

fn compare_signal_with_peak(
    generated: &[f32],
    target: &[f32],
    peak: f32,
) -> Result<Hyper2dImageMetricStats, Box<dyn std::error::Error>> {
    if generated.len() != target.len() {
        return Err(std::io::Error::other("generated and target image metric sizes differ").into());
    }
    if generated.is_empty() {
        return Err(std::io::Error::other("image metric buffers must not be empty").into());
    }
    let mse = generated
        .iter()
        .zip(target)
        .map(|(&generated_value, &target_value)| {
            let diff = generated_value - target_value;
            diff * diff
        })
        .sum::<f32>()
        / generated.len() as f32;
    Ok(Hyper2dImageMetricStats {
        mse,
        psnr_db: psnr(mse, peak),
        foreground_iou: 0.0,
        generated_occupancy: 0.0,
        target_occupancy: 0.0,
    })
}

fn compare_occupancy_images(
    generated: &[f32],
    target: &[f32],
    threshold: f32,
) -> Result<Hyper2dImageMetricStats, Box<dyn std::error::Error>> {
    if generated.len() != target.len() {
        return Err(std::io::Error::other("generated and target image metric sizes differ").into());
    }
    if generated.is_empty() {
        return Err(std::io::Error::other("image metric buffers must not be empty").into());
    }
    let mut mse = 0.0_f32;
    let mut intersection = 0_usize;
    let mut union = 0_usize;
    let mut generated_foreground = 0_usize;
    let mut target_foreground = 0_usize;
    for (&generated_value, &target_value) in generated.iter().zip(target) {
        let diff = generated_value - target_value;
        mse += diff * diff;
        let generated_hit = generated_value > threshold;
        let target_hit = target_value > threshold;
        generated_foreground += usize::from(generated_hit);
        target_foreground += usize::from(target_hit);
        intersection += usize::from(generated_hit && target_hit);
        union += usize::from(generated_hit || target_hit);
    }
    let pixels = generated.len() as f32;
    mse /= pixels;
    let foreground_iou = if union == 0 {
        1.0
    } else {
        intersection as f32 / union as f32
    };
    Ok(Hyper2dImageMetricStats {
        mse,
        psnr_db: psnr_unit(mse),
        foreground_iou,
        generated_occupancy: generated_foreground as f32 / pixels,
        target_occupancy: target_foreground as f32 / pixels,
    })
}

fn psnr_unit(mse: f32) -> f32 {
    psnr(mse, 1.0)
}

fn psnr(mse: f32, peak: f32) -> f32 {
    if mse <= f32::EPSILON {
        99.0
    } else {
        20.0 * (peak.max(f32::MIN_POSITIVE) / mse.sqrt()).log10()
    }
}

fn flatten_positions(positions: &[[f32; 4]]) -> Vec<f32> {
    positions
        .iter()
        .flat_map(|position| [position[0], position[1], position[2], position[3]])
        .collect()
}

fn tail_rgb_values(
    states: &[f32],
    state_dims: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if state_dims < 3 {
        return Err(std::io::Error::other(
            "tail RGB dynamics metrics require at least three state channels",
        )
        .into());
    }
    Ok(states
        .chunks_exact(state_dims)
        .flat_map(|state| {
            let tail = state_dims - 3;
            [
                (state[tail] + 0.5).clamp(0.0, 1.0),
                (state[tail + 1] + 0.5).clamp(0.0, 1.0),
                (state[tail + 2] + 0.5).clamp(0.0, 1.0),
            ]
        })
        .collect())
}

fn compare_mean_dx(
    generated: &[f32],
    target: &[f32],
) -> Result<(f32, f32), Box<dyn std::error::Error>> {
    if generated.len() != target.len() {
        return Err(std::io::Error::other("generated and target mean_dx sizes differ").into());
    }
    if generated.is_empty() {
        return Err(std::io::Error::other("mean_dx buffers must not be empty").into());
    }
    let mut mse = 0.0_f32;
    let mut mae = 0.0_f32;
    for (&generated_value, &target_value) in generated.iter().zip(target) {
        let diff = generated_value - target_value;
        mse += diff * diff;
        mae += diff.abs();
    }
    let len = generated.len() as f32;
    Ok((mse / len, mae / len))
}

pub(super) fn example_losses(
    base: &NpaModel,
    hyper: &HyperNpa2d,
    examples: &[Hyper2dLoadedExample],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    examples
        .iter()
        .map(|example| {
            let flow = vec![HyperFlowExample2d {
                condition: example.condition.clone(),
                batch: example.batch.clone(),
            }];
            Ok(hyper_rectified_flow_loss(base, hyper, &flow)?)
        })
        .collect()
}

pub(super) fn save_generated_examples(
    base: &NpaModel,
    base_manifest: &BpkModelManifest,
    base_model_path: Option<&PathBuf>,
    hyper: &HyperNpa2d,
    examples: &[Hyper2dLoadedExample],
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    for example in examples {
        let conditioned = generate_conditioned_npa_2d(
            base,
            hyper,
            &example.condition,
            ParticlePriorConfig::default(),
        )?;
        let slug = sanitize_slug(&example.descriptor.slug);
        save_conditioned_outputs(
            base_manifest,
            base_model_path,
            &example.descriptor.condition_path,
            &conditioned.adapter,
            &conditioned.model,
            Some(&output_dir.join(format!("{slug}.adapter.json"))),
            Some(&output_dir.join(format!("{slug}.bpk"))),
        )?;
    }
    Ok(())
}

fn catalog_entry_matches(
    preset: PresetArg,
    group: Option<Hyper2dCatalogGroupArg>,
    entry: &SelfOrgCatalogEntry,
) -> bool {
    match group {
        Some(Hyper2dCatalogGroupArg::Growing) => entry.group == "growing",
        Some(Hyper2dCatalogGroupArg::Texture) => entry.group == "texture",
        Some(Hyper2dCatalogGroupArg::All) => true,
        None => entry.preset == preset_name(preset),
    }
}

fn preset_name(preset: PresetArg) -> &'static str {
    match preset {
        PresetArg::Growing2d => "growing-2d",
        PresetArg::Texture2d => "texture-2d",
        PresetArg::Growing3dgs => "growing-3d-gs",
        PresetArg::PointMnist => "point-mnist",
    }
}

fn resolve_data_path(path: &Path, catalog_path: Option<&Path>) -> PathBuf {
    if path.is_absolute() || path.exists() {
        return path.to_path_buf();
    }
    if let Some(parent) = catalog_path.and_then(Path::parent) {
        let joined = parent.join(path);
        if joined.exists() {
            return joined;
        }
    }
    path.to_path_buf()
}

pub(super) fn load_condition_image_2d(
    path: &Path,
) -> Result<ConditionImage2d, Box<dyn std::error::Error>> {
    let image = image::ImageReader::open(path)?.decode()?.to_rgb8();
    let (width, height) = image.dimensions();
    let values = image
        .as_raw()
        .iter()
        .map(|value| *value as f32 / 255.0)
        .collect::<Vec<_>>();
    Ok(ConditionImage2d::from_rgb(
        width as usize,
        height as usize,
        values,
    )?)
}

pub(super) fn load_hyper_2d(path: &Path) -> Result<HyperNpa2d, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let hyper: HyperNpa2d = serde_json::from_slice(&bytes)?;
    hyper.validate()?;
    Ok(hyper)
}

pub(super) fn save_hyper_2d(
    path: &Path,
    hyper: &HyperNpa2d,
) -> Result<(), Box<dyn std::error::Error>> {
    hyper.validate()?;
    write_pretty_json(path, hyper)
}

pub(super) fn save_conditioned_outputs(
    base_manifest: &BpkModelManifest,
    base_model_path: Option<&PathBuf>,
    condition_path: &Path,
    adapter: &NpaLowRankAdapter,
    model: &NpaModel,
    adapter_output: Option<&PathBuf>,
    materialized_output: Option<&PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(path) = adapter_output {
        let adapter_manifest = BpkAdapterManifest::from_adapter(
            base_manifest,
            base_model_path.map(|path| path.display().to_string()),
            adapter.clone(),
            Some(format!("hyper2d-adapter:{}", condition_path.display())),
        )?;
        crate::import::save_adapter_manifest(path, &adapter_manifest)?;
    }
    if let Some(path) = materialized_output {
        let manifest = BpkModelManifest::from_model(
            model,
            base_manifest.hashgrid.clone(),
            Some(format!("hyper2d-materialized:{}", condition_path.display())),
        );
        crate::import::save_manifest(path, &manifest)?;
    }
    Ok(())
}

pub(super) fn write_pretty_json<T: Serialize>(
    path: &Path,
    value: &T,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(value)?)?;
    Ok(())
}

fn path_slug(path: &Path) -> String {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .map(sanitize_slug)
        .unwrap_or_else(|| "condition".to_string())
}

fn sanitize_slug(value: &str) -> String {
    let slug = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    if slug.is_empty() {
        "example".to_string()
    } else {
        slug
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn condition_rgb_resample_preserves_nearest_neighbor_quadrants() {
        let condition = ConditionImage2d::from_luma(2, 2, vec![0.0, 1.0, 0.5, 0.25]).unwrap();
        let rgb = condition_rgb_resampled(&condition, 4).unwrap();

        assert_eq!(rgb[0], 0.0);
        assert_eq!(rgb[3], 0.0);
        assert_eq!(rgb[6], 1.0);
        assert_eq!(rgb[(4 * 2) * 3], 0.5);
        assert_eq!(rgb[(4 * 2 + 2) * 3], 0.25);
    }

    #[test]
    fn occupancy_comparison_reports_overlap_and_error() {
        let generated = vec![1.0, 0.0, 1.0, 0.0];
        let target = vec![1.0, 1.0, 0.0, 0.0];
        let stats = compare_occupancy_images(&generated, &target, 0.05).unwrap();

        assert!((stats.mse - 0.5).abs() < 1.0e-6);
        assert!((stats.psnr_db - 3.0103).abs() < 1.0e-3);
        assert!((stats.foreground_iou - (1.0 / 3.0)).abs() < 1.0e-6);
        assert!((stats.generated_occupancy - 0.5).abs() < 1.0e-6);
        assert!((stats.target_occupancy - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn rasterize_grid_occupancy_maps_center_and_corners() {
        let grid = burn_automata_kernels::HashGridConfig {
            grid_size: [2, 2, 1],
            eps: 1.0,
            ..burn_automata_kernels::HashGridConfig::growing_2d()
        };
        let positions = vec![[0.0, 0.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]];
        let image = rasterize_grid_occupancy(&positions, &grid, 3);

        assert_eq!(image[4], 1.0);
        assert_eq!(image[6], 1.0);
        assert_eq!(image.iter().filter(|value| **value > 0.0).count(), 2);
    }

    #[test]
    fn tail_rgb_gaussian_render_uses_last_three_state_channels() {
        let grid = burn_automata_kernels::HashGridConfig {
            grid_size: [2, 2, 1],
            eps: 1.0,
            ..burn_automata_kernels::HashGridConfig::growing_2d()
        };
        let positions = vec![[0.0, 0.0, 0.0, 0.0]];
        let states = vec![9.0, -0.5, 0.0, 0.5];
        let rendered = rasterize_tail_rgb_gaussian(&positions, &states, 4, &grid, 3, 1.0).unwrap();

        let center = 4 * 3;
        assert!((rendered.rgb[center] - 0.0).abs() < 1.0e-6);
        assert!((rendered.rgb[center + 1] - 0.5).abs() < 1.0e-6);
        assert!((rendered.rgb[center + 2] - 1.0).abs() < 1.0e-6);
    }
}
