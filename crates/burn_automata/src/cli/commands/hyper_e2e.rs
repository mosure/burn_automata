use crate::cli::prelude::*;

use super::hyper_support::{
    Hyper2dLoadedExample, Hyper2dSourceDescriptor, bootstrap_hyper2d_adapters, flow_examples,
    load_condition_image_2d, load_hyper2d_examples, save_generated_examples, save_hyper_2d,
    write_pretty_json,
};
use sources::{Hyper2dScratchSource, preset_name, resolve_scratch_sources, sanitize_slug};

mod sources;

use crate::supervised_backward;

#[derive(Clone, Debug)]
struct Hyper2dTrainedSource {
    source: Hyper2dScratchSource,
    target: TargetImage2d,
    condition: ConditionImage2d,
    target_path: PathBuf,
    target_model: NpaModel,
    training: Target2dTrainingReport,
}

#[derive(Clone, Debug, Serialize)]
struct ScratchCatalogEntry {
    slug: String,
    title: Option<String>,
    group: String,
    preset: &'static str,
    output: PathBuf,
    particles: Option<usize>,
    seed_scale: Option<f32>,
    update_prob: Option<f32>,
}

#[allow(clippy::too_many_lines)]
pub(crate) fn run_train_hyper_2d_e2e(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::TrainHyper2dE2e {
        preset,
        target_images,
        catalog,
        catalog_thumbnail_dir,
        catalog_group,
        catalog_targets,
        catalog_limit,
        output_dir,
        report_output,
        scratch_catalog_output,
        shared_base_output,
        hyper_output,
        generated_output_dir,
        target_epochs,
        target_repetitions,
        target_report_interval,
        target_batch_size,
        target_pool_size,
        target_particles,
        target_step_min,
        target_step_max,
        target_inject_seed_interval,
        target_update_prob,
        target_seed,
        student_seed,
        seed_scale,
        seed_mode,
        target_brush_size,
        target_learning_rate,
        target_weight_decay,
        target_grad_clip_norm,
        target_adam_beta1,
        target_adam_beta2,
        target_adam_epsilon,
        target_scheduler_milestones,
        target_scheduler_gamma,
        target_per_parameter_grad_normalization,
        target_points,
        target_image_size,
        target_threshold,
        target_loss_image_size,
        target_splat_sigma,
        target_splat_loss_weight,
        target_color_loss_weight,
        target_density_loss_weight,
        target_displacement_regularizer_weight,
        target_overflow_regularizer_weight,
        target_bound_regularizer_weight,
        adapter_rank,
        adapter_alpha,
        adapter_rows,
        adapter_train_steps,
        adapter_learning_rate,
        adapter_grad_clip_norm,
        adapter_rollout_particles,
        adapter_rollout_steps,
        adapter_rollouts,
        shared_fit_steps,
        shared_fit_report_interval,
        shared_fit_base_learning_rate,
        shared_fit_base_weight_decay,
        shared_fit_base_grad_clip_norm,
        shared_fit_adapter_learning_rate,
        shared_fit_adapter_weight_decay,
        shared_fit_adapter_grad_clip_norm,
        hyper_steps,
        hyper_learning_rate,
        hyper_grad_clip_norm,
        hyper_weight_decay,
        hyper_hidden,
        hyper_output_scale,
        condition_token_grid_width,
        condition_token_grid_height,
        hyper_seed,
        flow_steps,
        flow_rows,
        flow_rollout_particles,
        flow_rollout_steps,
        flow_rollouts,
        eval_particles,
        eval_steps,
        eval_seed,
    } = command
    else {
        unreachable!("run_train_hyper_2d_e2e called with the wrong command variant");
    };

    let preset_arg = preset;
    let preset: AutomataPreset = preset.into();
    if preset != AutomataPreset::Growing2d {
        return Err(std::io::Error::other(
            "train-hyper2d-e2e currently supports the upstream growing-2d target objective",
        )
        .into());
    }

    let seed_mode: ParticleSeed = seed_mode.into();
    let seed_scale = seed_scale.unwrap_or_else(|| NpaConfig::seed_scale_for_preset(preset));
    let report_output = report_output.unwrap_or_else(|| output_dir.join("report.json"));
    let scratch_catalog_output =
        scratch_catalog_output.unwrap_or_else(|| output_dir.join("scratch_catalog.json"));
    let shared_base_output =
        shared_base_output.unwrap_or_else(|| output_dir.join("shared_base.bpk"));
    let hyper_output = hyper_output.unwrap_or_else(|| output_dir.join("hyper_2d.json"));
    let generated_output_dir = generated_output_dir.unwrap_or_else(|| output_dir.join("generated"));
    let target_dir = output_dir.join("targets");
    let adapter_dir = output_dir.join("static_adapters");
    let static_model_dir = output_dir.join("static_materialized");

    let sources = resolve_scratch_sources(
        preset_arg,
        &target_images,
        catalog.as_ref(),
        &catalog_thumbnail_dir,
        catalog_group,
        &catalog_targets,
        catalog_limit,
    )?;
    if sources.is_empty() {
        return Err(std::io::Error::other("no train-hyper2d-e2e sources matched").into());
    }
    if adapter_rows == 0 || adapter_rollout_steps == 0 || adapter_rollouts == 0 {
        return Err(std::io::Error::other(
            "adapter rows, rollout steps, and rollouts must be greater than zero",
        )
        .into());
    }
    if flow_steps > 0 && (flow_rows == 0 || flow_rollout_steps == 0 || flow_rollouts == 0) {
        return Err(std::io::Error::other(
            "flow rows, rollout steps, and rollouts must be greater than zero when --flow-steps is used",
        )
        .into());
    }

    let hashgrid = upstream_growing_2d_hashgrid();
    let loss_config = super::target2d::target2d_loss_config(
        target_loss_image_size,
        target_splat_sigma,
        target_splat_loss_weight,
        target_color_loss_weight,
        target_density_loss_weight,
        target_displacement_regularizer_weight,
        target_overflow_regularizer_weight,
        target_bound_regularizer_weight,
    )?;
    let target_training_config = Target2dTrainingConfig {
        epochs: target_epochs,
        repetitions: target_repetitions,
        report_interval: target_report_interval,
        batch_size: target_batch_size,
        pool_size: target_pool_size,
        particle_count: target_particles,
        step_min: target_step_min,
        step_max: target_step_max,
        inject_seed_interval: target_inject_seed_interval,
        update_prob: target_update_prob,
        seed: target_seed,
        seed_scale,
        seed_mode,
        brush_size: target_brush_size,
        per_parameter_grad_normalization: target_per_parameter_grad_normalization,
        optimizer: AdamWConfig {
            learning_rate: target_learning_rate,
            weight_decay: target_weight_decay,
            grad_clip_norm: target_grad_clip_norm,
            beta1: target_adam_beta1,
            beta2: target_adam_beta2,
            epsilon: target_adam_epsilon,
        },
        scheduler_milestones: if target_scheduler_milestones.is_empty() {
            Target2dTrainingConfig::default().scheduler_milestones
        } else {
            target_scheduler_milestones
        },
        scheduler_gamma: target_scheduler_gamma,
    };

    let trained = train_scratch_targets(
        &sources,
        &target_dir,
        &hashgrid,
        target_training_config.clone(),
        loss_config,
        target_threshold,
        target_points,
        target_image_size,
        student_seed,
    )?;
    let mut base = shared_mean_model(&trained)?;
    let mut base_manifest = BpkModelManifest::from_model(
        &base,
        hashgrid.clone(),
        Some(format!(
            "trained-rust:hyper2d-e2e-shared-mean:{}",
            trained.len()
        )),
    );

    let descriptors = trained
        .iter()
        .map(|example| Hyper2dSourceDescriptor {
            slug: example.source.slug.clone(),
            title: example.source.title.clone(),
            group: example.source.group.clone(),
            condition_path: example.source.condition_path.clone(),
            target_path: example.target_path.clone(),
            particles: example.source.particles.or(Some(target_particles)),
            seed_scale: example.source.seed_scale.or(Some(seed_scale)),
            update_prob: example.source.update_prob.or(Some(target_update_prob)),
        })
        .collect::<Vec<_>>();
    write_scratch_catalog(
        &scratch_catalog_output,
        &trained,
        preset_name(preset_arg),
        target_particles,
        seed_scale,
        target_update_prob,
    )?;

    let adapter_loaded = load_hyper2d_examples(
        &base,
        &base_manifest,
        &descriptors,
        adapter_rows,
        adapter_rollout_particles,
        adapter_rollout_steps,
        adapter_rollouts,
        None,
        Some(seed_scale),
        preset,
        seed_mode,
        hyper_seed,
    )?;
    let exact_adapter_required_rank = base.config.perception_dims().max(base.config.update_dims());
    if adapter_rank < exact_adapter_required_rank && adapter_train_steps == 0 {
        return Err(std::io::Error::other(format!(
            "adapter rank {adapter_rank} is below exact delta rank {exact_adapter_required_rank}; use --adapter-train-steps > 0"
        ))
        .into());
    }
    let adapter_bootstrap = bootstrap_hyper2d_adapters(
        &base,
        &adapter_loaded,
        adapter_rank,
        adapter_alpha,
        hyper_seed ^ 0x0a_da_70_2d,
        TrainingRunConfig {
            steps: adapter_train_steps,
            report_interval: adapter_train_steps.max(1),
            sgd: SgdConfig {
                learning_rate: adapter_learning_rate,
                weight_decay: 0.0,
                grad_clip_norm: adapter_grad_clip_norm,
            },
        },
    )?;
    let mut adapter_examples = adapter_bootstrap.examples;
    let shared_basis_fit = fit_shared_basis_and_adapters(
        &mut base,
        &mut adapter_examples,
        &adapter_loaded,
        SharedBasisFitConfig {
            steps: shared_fit_steps,
            report_interval: shared_fit_report_interval,
            base_sgd: SgdConfig {
                learning_rate: shared_fit_base_learning_rate,
                weight_decay: shared_fit_base_weight_decay,
                grad_clip_norm: shared_fit_base_grad_clip_norm,
            },
            adapter_sgd: SgdConfig {
                learning_rate: shared_fit_adapter_learning_rate,
                weight_decay: shared_fit_adapter_weight_decay,
                grad_clip_norm: shared_fit_adapter_grad_clip_norm,
            },
        },
    )?;
    base_manifest = BpkModelManifest::from_model(
        &base,
        hashgrid.clone(),
        Some(format!(
            "trained-rust:hyper2d-e2e-shared-basis-fit:{}:steps={}",
            trained.len(),
            shared_fit_steps
        )),
    );
    crate::import::save_manifest(&shared_base_output, &base_manifest)?;
    let static_adapter_reports = save_static_adapter_outputs(
        StaticAdapterSaveContext {
            base: &base,
            base_manifest: &base_manifest,
            base_model_path: &shared_base_output,
            adapter_dir: &adapter_dir,
            static_model_dir: &static_model_dir,
        },
        &adapter_loaded,
        &adapter_examples,
        &adapter_bootstrap.reports,
        (shared_fit_steps > 0).then_some("joint-shared-basis-fit"),
    )?;

    let hyper_config = HyperNpa2dConfig {
        condition_feature_dims: condition_feature_dims_for_token_grid(
            condition_token_grid_width,
            condition_token_grid_height,
        )?,
        condition_token_grid_width,
        condition_token_grid_height,
        hidden_dims: hyper_hidden,
        adapter_rank,
        adapter_alpha,
        output_scale: hyper_output_scale,
    };
    let mut hyper = HyperNpa2d::seeded(base.config.clone(), hyper_config, hyper_seed)?;
    let hyper_sgd = SgdConfig {
        learning_rate: hyper_learning_rate,
        weight_decay: hyper_weight_decay,
        grad_clip_norm: hyper_grad_clip_norm,
    };
    let initial_adapter_loss = hyper_adapter_regression_loss(&hyper, &adapter_examples)?;
    let (best_adapter_loss, best_adapter_step, adapter_history) = train_adapter_regression(
        &mut hyper,
        &adapter_examples,
        hyper_steps,
        hyper_sgd,
        initial_adapter_loss,
    )?;

    let flow_sgd = (flow_steps > 0).then_some(hyper_sgd);
    let mut initial_flow_loss = None;
    let mut best_flow_loss = None;
    let mut best_flow_step = None;
    let mut flow_history = Vec::new();
    if flow_steps > 0 {
        let flow_loaded = load_hyper2d_examples(
            &base,
            &base_manifest,
            &descriptors,
            flow_rows,
            flow_rollout_particles,
            flow_rollout_steps,
            flow_rollouts,
            None,
            Some(seed_scale),
            preset,
            seed_mode,
            hyper_seed ^ 0x5f_10_2d,
        )?;
        let flow_examples = flow_examples(&flow_loaded);
        let initial_loss = hyper_rectified_flow_loss(&base, &hyper, &flow_examples)?;
        initial_flow_loss = Some(initial_loss);
        let (loss, step, history) = train_flow_refinement(
            &base,
            &mut hyper,
            &flow_examples,
            flow_steps,
            hyper_sgd,
            initial_loss,
        )?;
        best_flow_loss = Some(loss);
        best_flow_step = Some(step);
        flow_history = history;
    }

    let final_adapter_loss = hyper_adapter_regression_loss(&hyper, &adapter_examples)?;
    let final_flow_loss = if flow_steps > 0 {
        let flow_loaded = load_hyper2d_examples(
            &base,
            &base_manifest,
            &descriptors,
            flow_rows,
            flow_rollout_particles,
            flow_rollout_steps,
            flow_rollouts,
            None,
            Some(seed_scale),
            preset,
            seed_mode,
            hyper_seed ^ 0x5f_10_2d,
        )?;
        Some(hyper_rectified_flow_loss(
            &base,
            &hyper,
            &flow_examples(&flow_loaded),
        )?)
    } else {
        None
    };

    save_hyper_2d(&hyper_output, &hyper)?;
    save_generated_examples(
        &base,
        &base_manifest,
        Some(&shared_base_output),
        &hyper,
        &adapter_loaded,
        &generated_output_dir,
    )?;

    let eval = evaluate_e2e_models(
        &trained,
        &base,
        &hyper,
        &adapter_examples,
        &hashgrid,
        loss_config,
        EvalConfig {
            particle_count: eval_particles.unwrap_or(target_particles),
            rollout_steps: eval_steps.unwrap_or(target_step_max),
            update_prob: target_update_prob,
            seed: eval_seed,
            seed_scale,
            seed_mode,
        },
    )?;
    let target_training = trained
        .iter()
        .map(|example| CliHyper2dE2eTargetReport {
            slug: example.source.slug.clone(),
            title: example.source.title.clone(),
            group: example.source.group.clone(),
            condition: example.source.condition_path.display().to_string(),
            target_model: example.target_path.display().to_string(),
            target_source_width: example.target.source_width,
            target_source_height: example.target.source_height,
            target_points: example.target.point_count(),
            training: example.training.clone(),
        })
        .collect::<Vec<_>>();
    let report = CliHyper2dE2eTrainingReport {
        preset,
        target_images: trained
            .iter()
            .map(|example| example.source.condition_path.display().to_string())
            .collect(),
        catalog: catalog.as_ref().map(|path| path.display().to_string()),
        catalog_group,
        catalog_targets,
        output_dir: output_dir.display().to_string(),
        report_output: report_output.display().to_string(),
        scratch_catalog_output: scratch_catalog_output.display().to_string(),
        shared_base_output: shared_base_output.display().to_string(),
        hyper_output: hyper_output.display().to_string(),
        generated_output_dir: generated_output_dir.display().to_string(),
        condition_encoder: "summary-pooled-token-grid-v1",
        shared_base_strategy: if shared_fit_steps > 0 {
            "mean-initialized-then-joint-shared-basis-fit"
        } else {
            "elementwise-mean-of-scratch-trained-target-weights"
        },
        static_adapter_strategy: if shared_fit_steps > 0 {
            "joint-shared-basis-fit"
        } else if adapter_rank >= exact_adapter_required_rank {
            "exact-weight-delta"
        } else {
            "supervised-low-rank-dynamics-distillation"
        },
        npa_config: base.config.clone(),
        hashgrid,
        target_loss_config: loss_config,
        target_training_config,
        hyper_config,
        hyper_sgd,
        flow_sgd,
        adapter_parameter_count: hyper.adapter_parameter_count(),
        materialized_parameter_count: crate::import::parameter_count(&base_manifest),
        exact_adapter_required_rank,
        target_training,
        shared_basis_fit,
        static_adapters: static_adapter_reports,
        initial_adapter_loss,
        final_adapter_loss,
        best_adapter_loss,
        best_adapter_step,
        initial_flow_loss,
        final_flow_loss,
        best_flow_loss,
        best_flow_step,
        adapter_history,
        flow_history,
        eval,
    };
    write_pretty_json(&report_output, &report)?;
    println!(
        "wrote {} examples={} targets={} hyper={} shared_base={}",
        report_output.display(),
        report.target_training.len(),
        target_dir.display(),
        hyper_output.display(),
        shared_base_output.display()
    );

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn train_scratch_targets(
    sources: &[Hyper2dScratchSource],
    target_dir: &Path,
    hashgrid: &burn_automata_kernels::HashGridConfig,
    base_config: Target2dTrainingConfig,
    loss_config: Target2dLossConfig,
    target_threshold: f32,
    target_points: usize,
    target_image_size: Option<usize>,
    student_seed: u64,
) -> Result<Vec<Hyper2dTrainedSource>, Box<dyn std::error::Error>> {
    let mut trained = Vec::with_capacity(sources.len());
    for (idx, source) in sources.iter().enumerate() {
        let slug = sanitize_slug(&source.slug);
        let target = super::target2d::load_target_image_2d_adaptive(
            &source.condition_path,
            target_threshold,
            target_points,
            target_image_size,
        )?;
        let condition = load_condition_image_2d(&source.condition_path)?;
        let mut model = NpaModel::upstream_seeded(
            NpaConfig::growing_2d(),
            student_seed.wrapping_add(idx as u64),
        );
        let mut config = base_config.clone();
        config.seed = config.seed.wrapping_add(idx as u64);
        let training = train_target_2d(&mut model, hashgrid, &target, config, loss_config)?;
        let target_path = target_dir.join(format!("{slug}.bpk"));
        let manifest = BpkModelManifest::from_model(
            &model,
            hashgrid.clone(),
            Some(format!(
                "trained-rust:hyper2d-e2e-target:{}",
                source.condition_path.display()
            )),
        );
        crate::import::save_manifest(&target_path, &manifest)?;
        trained.push(Hyper2dTrainedSource {
            source: source.clone(),
            target,
            condition,
            target_path,
            target_model: model,
            training,
        });
    }
    Ok(trained)
}

fn shared_mean_model(
    trained: &[Hyper2dTrainedSource],
) -> Result<NpaModel, Box<dyn std::error::Error>> {
    let models = trained
        .iter()
        .map(|example| &example.target_model)
        .collect::<Vec<_>>();
    shared_mean_model_from_refs(&models)
}

fn shared_mean_model_from_refs(
    models: &[&NpaModel],
) -> Result<NpaModel, Box<dyn std::error::Error>> {
    let first = models
        .first()
        .ok_or_else(|| std::io::Error::other("shared base requires at least one trained target"))?;
    let config = first.config.clone();
    let mut weights = NpaWeights::zeros(&config);
    for model in models {
        if model.config != config {
            return Err(
                std::io::Error::other("trained target configs differ; cannot average").into(),
            );
        }
        add_assign(&mut weights.w1, &model.weights.w1);
        add_assign(&mut weights.b1, &model.weights.b1);
        add_assign(&mut weights.w2, &model.weights.w2);
        add_assign(&mut weights.b2, &model.weights.b2);
    }
    let scale = 1.0 / models.len() as f32;
    scale_slice(&mut weights.w1, scale);
    scale_slice(&mut weights.b1, scale);
    scale_slice(&mut weights.w2, scale);
    scale_slice(&mut weights.b2, scale);
    let model = NpaModel { config, weights };
    model.validate()?;
    Ok(model)
}

fn add_assign(dst: &mut [f32], src: &[f32]) {
    for (dst, src) in dst.iter_mut().zip(src) {
        *dst += src;
    }
}

fn scale_slice(values: &mut [f32], scale: f32) {
    for value in values {
        *value *= scale;
    }
}

fn train_adapter_regression(
    hyper: &mut HyperNpa2d,
    examples: &[HyperAdapterExample2d],
    steps: usize,
    sgd: SgdConfig,
    initial_loss: f32,
) -> Result<(f32, usize, Vec<CliHyper2dE2eHyperHistoryEntry>), Box<dyn std::error::Error>> {
    let mut best_loss = initial_loss;
    let mut best_step = 0usize;
    let mut best_hyper = hyper.clone();
    let mut history = Vec::with_capacity(steps);
    for step in 1..=steps {
        let step_report = hyper_adapter_regression_train_step(hyper, examples, sgd)?;
        let loss = hyper_adapter_regression_loss(hyper, examples)?;
        if loss < best_loss {
            best_loss = loss;
            best_step = step;
            best_hyper = hyper.clone();
        }
        history.push(CliHyper2dE2eHyperHistoryEntry {
            step,
            loss,
            grad_norm: step_report.grad_norm,
            grad_scale: step_report.grad_scale,
        });
    }
    if best_loss < hyper_adapter_regression_loss(hyper, examples)? {
        *hyper = best_hyper;
    }
    Ok((best_loss, best_step, history))
}

fn train_flow_refinement(
    base: &NpaModel,
    hyper: &mut HyperNpa2d,
    examples: &[HyperFlowExample2d],
    steps: usize,
    sgd: SgdConfig,
    initial_loss: f32,
) -> Result<(f32, usize, Vec<CliHyper2dE2eHyperHistoryEntry>), Box<dyn std::error::Error>> {
    let mut best_loss = initial_loss;
    let mut best_step = 0usize;
    let mut best_hyper = hyper.clone();
    let mut history = Vec::with_capacity(steps);
    for step in 1..=steps {
        let step_report = hyper_rectified_flow_train_step(base, hyper, examples, sgd)?;
        let loss = hyper_rectified_flow_loss(base, hyper, examples)?;
        if loss < best_loss {
            best_loss = loss;
            best_step = step;
            best_hyper = hyper.clone();
        }
        history.push(CliHyper2dE2eHyperHistoryEntry {
            step,
            loss,
            grad_norm: step_report.grad_norm,
            grad_scale: step_report.grad_scale,
        });
    }
    if best_loss < hyper_rectified_flow_loss(base, hyper, examples)? {
        *hyper = best_hyper;
    }
    Ok((best_loss, best_step, history))
}

#[derive(Clone, Copy)]
struct SharedBasisFitConfig {
    steps: usize,
    report_interval: usize,
    base_sgd: SgdConfig,
    adapter_sgd: SgdConfig,
}

#[derive(Clone, Copy)]
struct SharedBasisStepStats {
    base_grad_norm: f32,
    base_grad_scale: f32,
    mean_adapter_grad_norm: f32,
    max_adapter_grad_norm: f32,
}

fn fit_shared_basis_and_adapters(
    base: &mut NpaModel,
    examples: &mut [HyperAdapterExample2d],
    loaded: &[Hyper2dLoadedExample],
    config: SharedBasisFitConfig,
) -> Result<CliHyper2dE2eSharedBasisFitReport, Box<dyn std::error::Error>> {
    validate_basis_examples(examples, loaded)?;
    let rows = shared_basis_rows(base, loaded);
    let initial_loss = shared_basis_loss(base, examples, loaded)?;
    if config.steps == 0 {
        return Ok(CliHyper2dE2eSharedBasisFitReport {
            enabled: false,
            steps: 0,
            report_interval: config.report_interval.max(1),
            rows,
            base_sgd: config.base_sgd,
            adapter_sgd: config.adapter_sgd,
            initial_loss,
            final_loss: initial_loss,
            best_loss: initial_loss,
            best_step: 0,
            history: Vec::new(),
        });
    }

    let report_interval = config.report_interval.max(1);
    let mut final_loss = initial_loss;
    let mut best_loss = initial_loss;
    let mut best_step = 0usize;
    let mut best_base = base.clone();
    let mut best_examples = examples.to_vec();
    let mut history = Vec::new();
    for step in 1..=config.steps {
        let step_stats =
            shared_basis_train_step(base, examples, loaded, config.base_sgd, config.adapter_sgd)?;
        if step == config.steps || step.is_multiple_of(report_interval) {
            final_loss = shared_basis_loss(base, examples, loaded)?;
            if final_loss < best_loss {
                best_loss = final_loss;
                best_step = step;
                best_base = base.clone();
                best_examples = examples.to_vec();
            }
            history.push(CliHyper2dE2eSharedBasisHistoryEntry {
                step,
                loss: final_loss,
                base_grad_norm: step_stats.base_grad_norm,
                base_grad_scale: step_stats.base_grad_scale,
                mean_adapter_grad_norm: step_stats.mean_adapter_grad_norm,
                max_adapter_grad_norm: step_stats.max_adapter_grad_norm,
            });
        }
    }
    if best_loss < final_loss {
        *base = best_base;
        examples.clone_from_slice(&best_examples);
        final_loss = best_loss;
    }

    Ok(CliHyper2dE2eSharedBasisFitReport {
        enabled: true,
        steps: config.steps,
        report_interval,
        rows,
        base_sgd: config.base_sgd,
        adapter_sgd: config.adapter_sgd,
        initial_loss,
        final_loss,
        best_loss,
        best_step,
        history,
    })
}

fn shared_basis_train_step(
    base: &mut NpaModel,
    examples: &mut [HyperAdapterExample2d],
    loaded: &[Hyper2dLoadedExample],
    base_sgd: SgdConfig,
    adapter_sgd: SgdConfig,
) -> Result<SharedBasisStepStats, Box<dyn std::error::Error>> {
    validate_basis_examples(examples, loaded)?;
    let mut base_grads = zero_model_gradients(base);
    let example_scale = 1.0 / examples.len() as f32;
    let mut adapter_grad_sum = 0.0_f32;
    let mut adapter_grad_max = 0.0_f32;

    for (example, loaded) in examples.iter_mut().zip(loaded) {
        let adapted = example.target_adapter.apply_to_model(base)?;
        let (full_grads, _) = supervised_backward(&adapted, &loaded.batch)?;
        let adapter_grads =
            project_low_rank_adapter_gradients(base, &example.target_adapter, &full_grads)?;
        let adapter_step =
            apply_sgd_adapter_gradients(&mut example.target_adapter, &adapter_grads, adapter_sgd)?;
        add_scaled_model_gradients(&mut base_grads, &full_grads, example_scale);
        adapter_grad_sum += adapter_step.grad_norm;
        adapter_grad_max = adapter_grad_max.max(adapter_step.grad_norm);
    }

    let base_step = apply_sgd_gradients(base, &base_grads, base_sgd)?;
    Ok(SharedBasisStepStats {
        base_grad_norm: base_step.grad_norm,
        base_grad_scale: base_step.grad_scale,
        mean_adapter_grad_norm: adapter_grad_sum / examples.len() as f32,
        max_adapter_grad_norm: adapter_grad_max,
    })
}

fn shared_basis_loss(
    base: &NpaModel,
    examples: &[HyperAdapterExample2d],
    loaded: &[Hyper2dLoadedExample],
) -> Result<f32, Box<dyn std::error::Error>> {
    validate_basis_examples(examples, loaded)?;
    let mut loss = 0.0_f32;
    for (example, loaded) in examples.iter().zip(loaded) {
        loss += supervised_adapter_loss(base, &example.target_adapter, &loaded.batch)?;
    }
    Ok(loss / examples.len() as f32)
}

fn shared_basis_rows(base: &NpaModel, loaded: &[Hyper2dLoadedExample]) -> usize {
    let input_dims = base.config.perception_dims().max(1);
    loaded
        .iter()
        .map(|example| example.batch.features.len() / input_dims)
        .sum()
}

fn validate_basis_examples(
    examples: &[HyperAdapterExample2d],
    loaded: &[Hyper2dLoadedExample],
) -> Result<(), Box<dyn std::error::Error>> {
    if examples.is_empty() {
        return Err(std::io::Error::other("shared basis fitting requires examples").into());
    }
    if examples.len() != loaded.len() {
        return Err(
            std::io::Error::other("shared basis examples do not match loaded batches").into(),
        );
    }
    Ok(())
}

fn zero_model_gradients(model: &NpaModel) -> SupervisedGradients {
    SupervisedGradients {
        w1: vec![0.0; model.weights.w1.len()],
        b1: vec![0.0; model.weights.b1.len()],
        w2: vec![0.0; model.weights.w2.len()],
        b2: vec![0.0; model.weights.b2.len()],
        features: Vec::new(),
    }
}

fn add_scaled_model_gradients(
    dst: &mut SupervisedGradients,
    src: &SupervisedGradients,
    scale: f32,
) {
    add_scaled_slice(&mut dst.w1, &src.w1, scale);
    add_scaled_slice(&mut dst.b1, &src.b1, scale);
    add_scaled_slice(&mut dst.w2, &src.w2, scale);
    add_scaled_slice(&mut dst.b2, &src.b2, scale);
}

fn add_scaled_slice(dst: &mut [f32], src: &[f32], scale: f32) {
    for (dst, src) in dst.iter_mut().zip(src) {
        *dst += src * scale;
    }
}

struct StaticAdapterSaveContext<'a> {
    base: &'a NpaModel,
    base_manifest: &'a BpkModelManifest,
    base_model_path: &'a Path,
    adapter_dir: &'a Path,
    static_model_dir: &'a Path,
}

fn save_static_adapter_outputs(
    context: StaticAdapterSaveContext<'_>,
    loaded: &[Hyper2dLoadedExample],
    examples: &[HyperAdapterExample2d],
    bootstrap_reports: &[CliHyper2dAdapterBootstrapReport],
    method_override: Option<&'static str>,
) -> Result<Vec<CliHyper2dE2eAdapterReport>, Box<dyn std::error::Error>> {
    if loaded.len() != examples.len() || loaded.len() != bootstrap_reports.len() {
        return Err(
            std::io::Error::other("static adapter reports do not match loaded examples").into(),
        );
    }
    let mut reports = Vec::with_capacity(examples.len());
    for ((loaded, example), bootstrap_report) in loaded.iter().zip(examples).zip(bootstrap_reports)
    {
        let slug = sanitize_slug(&loaded.descriptor.slug);
        let adapter_path = context.adapter_dir.join(format!("{slug}.adapter.json"));
        let materialized_path = context.static_model_dir.join(format!("{slug}.bpk"));
        let adapter_manifest = BpkAdapterManifest::from_adapter(
            context.base_manifest,
            Some(context.base_model_path.display().to_string()),
            example.target_adapter.clone(),
            Some(format!(
                "hyper2d-e2e-static-adapter:{}",
                loaded.descriptor.condition_path.display()
            )),
        )?;
        crate::import::save_adapter_manifest(&adapter_path, &adapter_manifest)?;
        let model = example.target_adapter.apply_to_model(context.base)?;
        let materialized_manifest = BpkModelManifest::from_model(
            &model,
            context.base_manifest.hashgrid.clone(),
            Some(format!(
                "hyper2d-e2e-static-materialized:{}",
                loaded.descriptor.condition_path.display()
            )),
        );
        crate::import::save_manifest(&materialized_path, &materialized_manifest)?;
        let final_loss =
            supervised_adapter_loss(context.base, &example.target_adapter, &loaded.batch)?;
        reports.push(CliHyper2dE2eAdapterReport {
            slug: loaded.descriptor.slug.clone(),
            adapter_output: adapter_path.display().to_string(),
            materialized_output: materialized_path.display().to_string(),
            method: method_override.unwrap_or(bootstrap_report.method),
            steps: bootstrap_report.steps,
            rows: bootstrap_report.rows,
            initial_loss: bootstrap_report.initial_loss,
            final_loss,
            best_loss: bootstrap_report.best_loss.min(final_loss),
            adapter_parameter_count: bootstrap_report.adapter_parameter_count,
        });
    }
    Ok(reports)
}

#[derive(Clone, Copy)]
struct EvalConfig {
    particle_count: usize,
    rollout_steps: usize,
    update_prob: f32,
    seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
}

fn evaluate_e2e_models(
    trained: &[Hyper2dTrainedSource],
    base: &NpaModel,
    hyper: &HyperNpa2d,
    adapter_examples: &[HyperAdapterExample2d],
    hashgrid: &burn_automata_kernels::HashGridConfig,
    loss_config: Target2dLossConfig,
    config: EvalConfig,
) -> Result<Vec<CliHyper2dE2eEvalReport>, Box<dyn std::error::Error>> {
    if trained.len() != adapter_examples.len() {
        return Err(std::io::Error::other("eval examples do not match adapters").into());
    }
    let mut reports = Vec::with_capacity(trained.len());
    for (idx, (example, adapter_example)) in trained.iter().zip(adapter_examples).enumerate() {
        let seed = config.seed.wrapping_add(idx as u64);
        let update_prob = example.source.update_prob.unwrap_or(config.update_prob);
        let seed_scale = example.source.seed_scale.unwrap_or(config.seed_scale);
        let particle_count = example.source.particles.unwrap_or(config.particle_count);
        let trained_target_loss = evaluate_model_target_loss(
            &example.target_model,
            hashgrid,
            &example.target,
            loss_config,
            EvalConfig {
                particle_count,
                rollout_steps: config.rollout_steps,
                update_prob,
                seed,
                seed_scale,
                seed_mode: config.seed_mode,
            },
        )?;
        let static_model = adapter_example.target_adapter.apply_to_model(base)?;
        let static_adapter_loss = evaluate_model_target_loss(
            &static_model,
            hashgrid,
            &example.target,
            loss_config,
            EvalConfig {
                particle_count,
                rollout_steps: config.rollout_steps,
                update_prob,
                seed,
                seed_scale,
                seed_mode: config.seed_mode,
            },
        )?;
        let hyper_adapter = hyper.predict_adapter(&example.condition)?;
        let hyper_model = hyper_adapter.apply_to_model(base)?;
        let hyper_loss = evaluate_model_target_loss(
            &hyper_model,
            hashgrid,
            &example.target,
            loss_config,
            EvalConfig {
                particle_count,
                rollout_steps: config.rollout_steps,
                update_prob,
                seed,
                seed_scale,
                seed_mode: config.seed_mode,
            },
        )?;
        reports.push(CliHyper2dE2eEvalReport {
            slug: example.source.slug.clone(),
            condition: example.source.condition_path.display().to_string(),
            particle_count,
            rollout_steps: config.rollout_steps,
            update_prob,
            seed,
            seed_scale,
            seed_mode: config.seed_mode,
            trained_target_loss,
            static_adapter_loss,
            hyper_loss,
            static_adapter_gap_to_trained_target: static_adapter_loss.total_loss
                - trained_target_loss.total_loss,
            hyper_gap_to_trained_target: hyper_loss.total_loss - trained_target_loss.total_loss,
            hyper_gap_to_static_adapter: hyper_loss.total_loss - static_adapter_loss.total_loss,
            static_adapter_ratio_to_trained_target: ratio(
                static_adapter_loss.total_loss,
                trained_target_loss.total_loss,
            ),
            hyper_ratio_to_trained_target: ratio(
                hyper_loss.total_loss,
                trained_target_loss.total_loss,
            ),
            hyper_ratio_to_static_adapter: ratio(
                hyper_loss.total_loss,
                static_adapter_loss.total_loss,
            ),
        });
    }
    Ok(reports)
}

fn evaluate_model_target_loss(
    model: &NpaModel,
    hashgrid: &burn_automata_kernels::HashGridConfig,
    target: &TargetImage2d,
    loss_config: Target2dLossConfig,
    config: EvalConfig,
) -> Result<Target2dLossReport, Box<dyn std::error::Error>> {
    let trace = run_rollout(
        model,
        hashgrid,
        &RolloutConfig {
            particle_count: config.particle_count,
            steps: config.rollout_steps,
            update_prob: config.update_prob,
            seed: config.seed,
            seed_scale: config.seed_scale,
            ..RolloutConfig::default()
        },
        config.seed_mode,
    )?;
    Ok(target_2d_loss_with_adjoint(
        &trace.positions,
        &trace.states,
        trace.batch_size,
        trace.particle_count,
        trace.state_dims,
        target,
        loss_config,
        trace.mean_dx.iter().copied().sum(),
        trace.steps,
    )?
    .report)
}

fn ratio(value: f32, reference: f32) -> Option<f32> {
    (reference.abs() > f32::MIN_POSITIVE).then_some(value / reference)
}

fn write_scratch_catalog(
    path: &Path,
    trained: &[Hyper2dTrainedSource],
    preset: &'static str,
    default_particles: usize,
    default_seed_scale: f32,
    default_update_prob: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let entries = trained
        .iter()
        .map(|example| ScratchCatalogEntry {
            slug: example.source.slug.clone(),
            title: example.source.title.clone(),
            group: example
                .source
                .group
                .clone()
                .unwrap_or_else(|| "scratch".to_string()),
            preset,
            output: example.target_path.clone(),
            particles: example.source.particles.or(Some(default_particles)),
            seed_scale: example.source.seed_scale.or(Some(default_seed_scale)),
            update_prob: example.source.update_prob.or(Some(default_update_prob)),
        })
        .collect::<Vec<_>>();
    write_pretty_json(path, &entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_mean_model_averages_all_weight_tensors() {
        let config = NpaConfig::growing_2d();
        let mut first = NpaModel {
            config: config.clone(),
            weights: NpaWeights::zeros(&config),
        };
        let mut second = first.clone();
        first.weights.w1.fill(2.0);
        first.weights.b1.fill(4.0);
        first.weights.w2.fill(6.0);
        first.weights.b2.fill(8.0);
        second.weights.w1.fill(4.0);
        second.weights.b1.fill(8.0);
        second.weights.w2.fill(10.0);
        second.weights.b2.fill(12.0);

        let mean = shared_mean_model_from_refs(&[&first, &second]).unwrap();

        assert!(mean.weights.w1.iter().all(|value| *value == 3.0));
        assert!(mean.weights.b1.iter().all(|value| *value == 6.0));
        assert!(mean.weights.w2.iter().all(|value| *value == 8.0));
        assert!(mean.weights.b2.iter().all(|value| *value == 10.0));
    }

    #[test]
    fn shared_basis_step_updates_base_and_adapter() {
        let config = NpaConfig::growing_2d();
        let mut base = NpaModel::upstream_seeded(config.clone(), 1);
        let target = NpaModel::upstream_seeded(config.clone(), 2);
        let condition = ConditionImage2d::from_rgb(1, 1, vec![1.0, 0.0, 0.0]).unwrap();
        let batch = feature_supervised_batch(
            &base,
            SupervisedTarget::Teacher(&target),
            FeatureBatchConfig {
                rows: 8,
                seed: 3,
                amplitude: 0.25,
            },
        )
        .unwrap();
        let mut examples = vec![HyperAdapterExample2d {
            condition: condition.clone(),
            target_adapter: NpaLowRankAdapter::seeded(&config, 4, 4.0, 4),
        }];
        let loaded = vec![Hyper2dLoadedExample {
            descriptor: Hyper2dSourceDescriptor {
                slug: "sample".to_string(),
                title: None,
                group: None,
                condition_path: PathBuf::from("sample.png"),
                target_path: PathBuf::from("sample.bpk"),
                particles: None,
                seed_scale: None,
                update_prob: None,
            },
            condition,
            batch,
            rows: 8,
            particle_count: 8,
            rollout_steps: 1,
            rollouts: 1,
            update_prob: 1.0,
            seed_scale: 0.2,
            seed_mode: ParticleSeed::UniformCircle,
            seed: 5,
        }];
        let before_base_w1 = base.weights.w1.clone();
        let before_adapter = examples[0].target_adapter.to_parameter_vector();

        let report = shared_basis_train_step(
            &mut base,
            &mut examples,
            &loaded,
            SgdConfig {
                learning_rate: 1.0e-3,
                weight_decay: 0.0,
                grad_clip_norm: 1.0,
            },
            SgdConfig {
                learning_rate: 1.0e-3,
                weight_decay: 0.0,
                grad_clip_norm: 1.0,
            },
        )
        .unwrap();

        assert!(report.base_grad_norm.is_finite());
        assert!(report.mean_adapter_grad_norm.is_finite());
        assert_ne!(base.weights.w1, before_base_w1);
        assert_ne!(
            examples[0].target_adapter.to_parameter_vector(),
            before_adapter
        );
    }
}
