use crate::cli::prelude::*;
use crate::hyper::{
    hypernet::HyperNpa2dGradients,
    training::{adapter_gradient_vector, apply_hyper_sgd},
};

use super::hyper_support::{
    Hyper2dAdapterBootstrapResult, Hyper2dConditionFeatureCache, Hyper2dLoadedExample,
    Hyper2dSourceDescriptor, attach_condition_features, bootstrap_hyper2d_adapters, flow_examples,
    load_condition_image_2d, load_hyper2d_examples, save_generated_examples, save_hyper_2d,
    write_pretty_json,
};
use shared_basis::{SharedBasisFitConfig, fit_shared_basis_and_adapters};
use sources::{
    Hyper2dScratchSource, OmniSvgSourceConfig, ScratchSourceResolveConfig, preset_name,
    resolve_scratch_sources, sanitize_slug,
};

#[cfg(feature = "dino")]
mod dino;
mod direct_basis;
mod shared_basis;
mod sources;

pub(crate) use direct_basis::run_train_hyper_2d_direct_basis;

#[derive(Clone, Debug)]
struct Hyper2dTrainedSource {
    source: Hyper2dScratchSource,
    split: Hyper2dE2eSplit,
    target: TargetImage2d,
    condition: ConditionImage2d,
    target_path: PathBuf,
    target_model: NpaModel,
    training: Target2dTrainingReport,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Hyper2dE2eSplit {
    Train,
    Holdout,
}

impl Hyper2dE2eSplit {
    fn label(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Holdout => "holdout",
        }
    }

    fn is_train(self) -> bool {
        self == Self::Train
    }
}

fn omnisvg_source_report(
    config: Option<OmniSvgSourceConfig<'_>>,
) -> Option<CliOmniSvgSourceReport> {
    config.map(|config| CliOmniSvgSourceReport {
        dataset: config.dataset,
        dataset_id: config.dataset.dataset_id().to_string(),
        split: config.split.to_string(),
        cache_dir: config.cache_dir.display().to_string(),
        offset: config.offset,
        limit: config.limit,
        page_size: config.page_size,
        download: config.download,
        refresh: config.refresh,
        token_env: config.token_env.to_string(),
    })
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
        omnisvg_dataset,
        omnisvg_split,
        omnisvg_cache_dir,
        omnisvg_offset,
        omnisvg_limit,
        omnisvg_page_size,
        omnisvg_download,
        omnisvg_refresh,
        omnisvg_token_env,
        holdout_targets,
        holdout_stride,
        holdout_offset,
        fit_holdout_static_oracles,
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
        condition_encoder,
        dino_model,
        dino_image_size,
        shared_fit_steps,
        shared_fit_report_interval,
        shared_fit_example_batch_size,
        shared_fit_adapter_l2,
        shared_fit_seed,
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
        direct_finetune_steps,
        direct_finetune_report_interval,
        direct_finetune_rollout_particles,
        direct_finetune_rollout_steps,
        direct_finetune_seed,
        direct_finetune_hyper_learning_rate,
        direct_finetune_hyper_weight_decay,
        direct_finetune_hyper_grad_clip_norm,
        direct_finetune_adapter_l2,
        eval_particles,
        eval_steps,
        eval_seed,
        quality_max_static_ratio,
        quality_max_hyper_static_ratio,
        quality_max_hyper_target_ratio,
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
    let holdout_adapter_dir = output_dir.join("holdout_static_adapters");
    let holdout_static_model_dir = output_dir.join("holdout_static_materialized");
    let omnisvg_source = omnisvg_dataset.map(|dataset| OmniSvgSourceConfig {
        dataset,
        split: &omnisvg_split,
        cache_dir: &omnisvg_cache_dir,
        offset: omnisvg_offset,
        limit: omnisvg_limit,
        page_size: omnisvg_page_size,
        download: omnisvg_download,
        refresh: omnisvg_refresh,
        token_env: &omnisvg_token_env,
    });

    let sources = resolve_scratch_sources(ScratchSourceResolveConfig {
        preset: preset_arg,
        target_images: &target_images,
        target_image_dirs: &[],
        target_image_recursive: false,
        image_extensions: &[],
        catalog: catalog.as_ref(),
        catalog_thumbnail_dir: &catalog_thumbnail_dir,
        catalog_group,
        catalog_targets: &catalog_targets,
        catalog_limit,
        omnisvg: omnisvg_source,
    })?;
    if sources.is_empty() {
        return Err(std::io::Error::other("no train-hyper2d-e2e sources matched").into());
    }
    let splits = resolve_e2e_splits(&sources, &holdout_targets, holdout_stride, holdout_offset)?;
    let condition_encoder: ConditionEncoder2d = condition_encoder.into();
    let condition_features = build_condition_feature_cache(
        &sources,
        condition_encoder,
        dino_model.as_ref(),
        dino_image_size,
    )?;
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
    if direct_finetune_steps > 0 && direct_finetune_rollout_steps == 0 {
        return Err(std::io::Error::other(
            "--direct-finetune-rollout-steps must be greater than zero when --direct-finetune-steps is used",
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
        &splits,
        &target_dir,
        &hashgrid,
        target_training_config.clone(),
        loss_config,
        target_threshold,
        target_points,
        target_image_size,
        student_seed,
        &condition_features,
    )?;
    let train_trained = trained
        .iter()
        .filter(|example| example.split.is_train())
        .cloned()
        .collect::<Vec<_>>();
    let holdout_trained = trained
        .iter()
        .filter(|example| !example.split.is_train())
        .cloned()
        .collect::<Vec<_>>();
    let mut base = initialize_shared_base(&train_trained, student_seed)?;
    let mut base_manifest = BpkModelManifest::from_model(
        &base,
        hashgrid.clone(),
        Some(shared_base_source(train_trained.len(), 0)),
    );

    let train_descriptors = descriptors_for_trained(
        &train_trained,
        target_particles,
        seed_scale,
        target_update_prob,
    );
    let holdout_descriptors = descriptors_for_trained(
        &holdout_trained,
        target_particles,
        seed_scale,
        target_update_prob,
    );
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
        &train_descriptors,
        Some(&condition_features),
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
            example_batch_size: shared_fit_example_batch_size,
            adapter_l2_weight: shared_fit_adapter_l2,
            seed: shared_fit_seed,
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
        Some(shared_base_source(train_trained.len(), shared_fit_steps)),
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
        Hyper2dE2eSplit::Train,
        (shared_fit_steps > 0).then_some("joint-shared-basis-low-rank-adapters"),
    )?;
    let holdout_adapter_loaded = if holdout_descriptors.is_empty() {
        Vec::new()
    } else {
        load_hyper2d_examples(
            &base,
            &base_manifest,
            &holdout_descriptors,
            Some(&condition_features),
            adapter_rows,
            adapter_rollout_particles,
            adapter_rollout_steps,
            adapter_rollouts,
            None,
            Some(seed_scale),
            preset,
            seed_mode,
            hyper_seed ^ 0x0a_da_70_2d ^ 0x90_1d_00_00,
        )?
    };
    let holdout_adapter_bootstrap = if holdout_adapter_loaded.is_empty() {
        empty_adapter_bootstrap()
    } else if fit_holdout_static_oracles {
        bootstrap_hyper2d_adapters(
            &base,
            &holdout_adapter_loaded,
            adapter_rank,
            adapter_alpha,
            hyper_seed ^ 0x0a_da_70_2d ^ 0x90_1d_00_00,
            TrainingRunConfig {
                steps: adapter_train_steps,
                report_interval: adapter_train_steps.max(1),
                sgd: SgdConfig {
                    learning_rate: adapter_learning_rate,
                    weight_decay: 0.0,
                    grad_clip_norm: adapter_grad_clip_norm,
                },
            },
        )?
    } else {
        zero_adapter_bootstrap(&base, &holdout_adapter_loaded, adapter_rank, adapter_alpha)?
    };
    let holdout_adapter_reports = holdout_adapter_bootstrap.reports;
    let holdout_adapter_examples = holdout_adapter_bootstrap.examples;
    let holdout_static_adapter_reports = save_static_adapter_outputs(
        StaticAdapterSaveContext {
            base: &base,
            base_manifest: &base_manifest,
            base_model_path: &shared_base_output,
            adapter_dir: &holdout_adapter_dir,
            static_model_dir: &holdout_static_model_dir,
        },
        &holdout_adapter_loaded,
        &holdout_adapter_examples,
        &holdout_adapter_reports,
        Hyper2dE2eSplit::Holdout,
        (shared_fit_steps > 0).then_some("holdout-oracle-low-rank-adapters-for-final-shared-base"),
    )?;

    let hyper_config = HyperNpa2dConfig {
        condition_encoder,
        condition_feature_dims: condition_feature_dims_for_encoder(
            condition_encoder,
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
            &train_descriptors,
            Some(&condition_features),
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
            &train_descriptors,
            Some(&condition_features),
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
    let direct_finetune = if direct_finetune_steps > 0 {
        Some(train_direct_image_finetune(
            &base,
            &mut hyper,
            &train_trained,
            &hashgrid,
            DirectFinetuneConfig {
                steps: direct_finetune_steps,
                report_interval: direct_finetune_report_interval,
                particle_count: direct_finetune_rollout_particles.unwrap_or(target_particles),
                rollout_steps: direct_finetune_rollout_steps,
                update_prob: target_update_prob,
                seed: direct_finetune_seed,
                seed_scale,
                seed_mode,
                loss_config,
                per_parameter_grad_normalization: target_per_parameter_grad_normalization,
                hyper_sgd: SgdConfig {
                    learning_rate: direct_finetune_hyper_learning_rate,
                    weight_decay: direct_finetune_hyper_weight_decay,
                    grad_clip_norm: direct_finetune_hyper_grad_clip_norm,
                },
                adapter_l2_weight: direct_finetune_adapter_l2,
            },
        )?)
    } else {
        None
    };

    save_hyper_2d(&hyper_output, &hyper)?;
    let generated_loaded = adapter_loaded
        .iter()
        .chain(holdout_adapter_loaded.iter())
        .cloned()
        .collect::<Vec<_>>();
    save_generated_examples(
        &base,
        &base_manifest,
        Some(&shared_base_output),
        &hyper,
        &generated_loaded,
        &generated_output_dir,
    )?;

    let train_eval = evaluate_e2e_models(
        &train_trained,
        &base,
        &hyper,
        &adapter_examples,
        &hashgrid,
        loss_config,
        E2eEvalConfig {
            split: Hyper2dE2eSplit::Train,
            rollout: EvalConfig {
                particle_count: eval_particles.unwrap_or(target_particles),
                rollout_steps: eval_steps.unwrap_or(target_step_max),
                update_prob: target_update_prob,
                seed: eval_seed,
                seed_scale,
                seed_mode,
            },
        },
    )?;
    let holdout_eval = if holdout_trained.is_empty() {
        Vec::new()
    } else {
        evaluate_e2e_models(
            &holdout_trained,
            &base,
            &hyper,
            &holdout_adapter_examples,
            &hashgrid,
            loss_config,
            E2eEvalConfig {
                split: Hyper2dE2eSplit::Holdout,
                rollout: EvalConfig {
                    particle_count: eval_particles.unwrap_or(target_particles),
                    rollout_steps: eval_steps.unwrap_or(target_step_max),
                    update_prob: target_update_prob,
                    seed: eval_seed ^ 0x90_1d_00_00,
                    seed_scale,
                    seed_mode,
                },
            },
        )?
    };
    let quality_gates = QualityGateConfig {
        max_static_ratio: quality_max_static_ratio,
        max_hyper_static_ratio: quality_max_hyper_static_ratio,
        max_hyper_target_ratio: quality_max_hyper_target_ratio,
    };
    let train_quality = summarize_e2e_quality(&train_eval, quality_gates);
    let holdout_quality =
        (!holdout_eval.is_empty()).then(|| summarize_e2e_quality(&holdout_eval, quality_gates));
    let mut eval = train_eval;
    eval.extend(holdout_eval);
    let quality = summarize_e2e_quality(&eval, quality_gates);
    let target_training = trained
        .iter()
        .map(|example| CliHyper2dE2eTargetReport {
            slug: example.source.slug.clone(),
            split: example.split.label(),
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
        omnisvg: omnisvg_source_report(omnisvg_source),
        holdout_targets,
        holdout_stride,
        holdout_offset,
        fit_holdout_static_oracles,
        output_dir: output_dir.display().to_string(),
        report_output: report_output.display().to_string(),
        scratch_catalog_output: scratch_catalog_output.display().to_string(),
        shared_base_output: shared_base_output.display().to_string(),
        hyper_output: hyper_output.display().to_string(),
        generated_output_dir: generated_output_dir.display().to_string(),
        condition_encoder: condition_encoder_label(condition_encoder),
        shared_base_strategy: shared_base_strategy(shared_fit_steps),
        static_adapter_strategy: static_adapter_strategy(
            shared_fit_steps,
            adapter_rank,
            exact_adapter_required_rank,
        ),
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
        static_adapters: static_adapter_reports
            .into_iter()
            .chain(holdout_static_adapter_reports)
            .collect(),
        initial_adapter_loss,
        final_adapter_loss,
        best_adapter_loss,
        best_adapter_step,
        initial_flow_loss,
        final_flow_loss,
        best_flow_loss,
        best_flow_step,
        direct_finetune,
        adapter_history,
        flow_history,
        quality,
        train_quality,
        holdout_quality,
        eval,
    };
    write_pretty_json(&report_output, &report)?;
    if let Some(message) = quality_failure_message(&report.quality) {
        return Err(std::io::Error::other(message).into());
    }
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
    splits: &[Hyper2dE2eSplit],
    target_dir: &Path,
    hashgrid: &burn_automata_kernels::HashGridConfig,
    base_config: Target2dTrainingConfig,
    loss_config: Target2dLossConfig,
    target_threshold: f32,
    target_points: usize,
    target_image_size: Option<usize>,
    student_seed: u64,
    condition_features: &Hyper2dConditionFeatureCache,
) -> Result<Vec<Hyper2dTrainedSource>, Box<dyn std::error::Error>> {
    if sources.len() != splits.len() {
        return Err(std::io::Error::other("source split count does not match sources").into());
    }
    let mut trained = Vec::with_capacity(sources.len());
    for (idx, (source, split)) in sources.iter().zip(splits).enumerate() {
        let slug = sanitize_slug(&source.slug);
        let target = super::target2d::load_target_image_2d_adaptive(
            &source.condition_path,
            target_threshold,
            target_points,
            target_image_size,
        )?;
        let condition = attach_condition_features(
            load_condition_image_2d(&source.condition_path)?,
            &source.condition_path,
            Some(condition_features),
        )?;
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
            split: *split,
            target,
            condition,
            target_path,
            target_model: model,
            training,
        });
    }
    Ok(trained)
}

fn build_condition_feature_cache(
    sources: &[Hyper2dScratchSource],
    encoder: ConditionEncoder2d,
    dino_model: Option<&PathBuf>,
    dino_image_size: usize,
) -> Result<Hyper2dConditionFeatureCache, Box<dyn std::error::Error>> {
    match encoder {
        ConditionEncoder2d::SummaryTokens => Ok(Hyper2dConditionFeatureCache::new()),
        ConditionEncoder2d::DinoVitsClsPatchMean => {
            build_dino_condition_feature_cache(sources, dino_model, dino_image_size)
        }
    }
}

fn resolve_e2e_splits(
    sources: &[Hyper2dScratchSource],
    holdout_targets: &[String],
    holdout_stride: usize,
    holdout_offset: usize,
) -> Result<Vec<Hyper2dE2eSplit>, Box<dyn std::error::Error>> {
    if holdout_stride == 0 && holdout_offset != 0 {
        return Err(std::io::Error::other("--holdout-offset requires --holdout-stride > 0").into());
    }
    let requested = holdout_targets
        .iter()
        .map(|target| sanitize_slug(target))
        .collect::<std::collections::BTreeSet<_>>();
    let matched = sources
        .iter()
        .filter(|source| requested.contains(&sanitize_slug(&source.slug)))
        .map(|source| sanitize_slug(&source.slug))
        .collect::<std::collections::BTreeSet<_>>();
    if let Some(missing) = requested.iter().find(|target| !matched.contains(*target)) {
        return Err(std::io::Error::other(format!(
            "--holdout-target {missing} did not match any selected source"
        ))
        .into());
    }

    let splits = sources
        .iter()
        .enumerate()
        .map(|(idx, source)| {
            let explicit_holdout = requested.contains(&sanitize_slug(&source.slug));
            let strided_holdout =
                holdout_stride > 0 && idx % holdout_stride == holdout_offset % holdout_stride;
            if explicit_holdout || strided_holdout {
                Hyper2dE2eSplit::Holdout
            } else {
                Hyper2dE2eSplit::Train
            }
        })
        .collect::<Vec<_>>();
    if !splits.iter().any(|split| split.is_train()) {
        return Err(
            std::io::Error::other("train-hyper2d-e2e split produced no training targets").into(),
        );
    }
    Ok(splits)
}

fn descriptors_for_trained(
    trained: &[Hyper2dTrainedSource],
    default_particles: usize,
    default_seed_scale: f32,
    default_update_prob: f32,
) -> Vec<Hyper2dSourceDescriptor> {
    trained
        .iter()
        .map(|example| Hyper2dSourceDescriptor {
            slug: example.source.slug.clone(),
            title: example.source.title.clone(),
            group: example.source.group.clone(),
            condition_path: example.source.condition_path.clone(),
            target_path: example.target_path.clone(),
            particles: example.source.particles.or(Some(default_particles)),
            seed_scale: example.source.seed_scale.or(Some(default_seed_scale)),
            update_prob: example.source.update_prob.or(Some(default_update_prob)),
        })
        .collect()
}

fn build_dino_condition_feature_cache(
    sources: &[Hyper2dScratchSource],
    dino_model: Option<&PathBuf>,
    dino_image_size: usize,
) -> Result<Hyper2dConditionFeatureCache, Box<dyn std::error::Error>> {
    #[cfg(feature = "dino")]
    {
        let model_path = dino_model.ok_or_else(|| {
            std::io::Error::other("--dino-model is required for --condition-encoder dino")
        })?;
        let encoder = dino::DinoVitsConditionEncoder::load(model_path, dino_image_size)?;
        let mut cache = Hyper2dConditionFeatureCache::new();
        for source in sources {
            if cache.contains_key(&source.condition_path) {
                continue;
            }
            let condition = load_condition_image_2d(&source.condition_path)?;
            cache.insert(source.condition_path.clone(), encoder.encode(&condition)?);
        }
        Ok(cache)
    }
    #[cfg(not(feature = "dino"))]
    {
        let _ = (sources, dino_model, dino_image_size);
        Err(std::io::Error::other(
            "--condition-encoder dino requires building burn_automata with --features dino",
        )
        .into())
    }
}

fn condition_encoder_label(encoder: ConditionEncoder2d) -> &'static str {
    match encoder {
        ConditionEncoder2d::SummaryTokens => "summary-pooled-token-grid-v1",
        ConditionEncoder2d::DinoVitsClsPatchMean => "dino-vits-cls-patch-mean-v1",
    }
}

fn initialize_shared_base(
    trained: &[Hyper2dTrainedSource],
    seed: u64,
) -> Result<NpaModel, Box<dyn std::error::Error>> {
    let config = shared_base_config(trained)?;
    let model = NpaModel::upstream_seeded(config, seed ^ 0x5e_ed_ba_5e);
    model.validate()?;
    Ok(model)
}

fn shared_base_config(
    trained: &[Hyper2dTrainedSource],
) -> Result<NpaConfig, Box<dyn std::error::Error>> {
    let first = trained
        .first()
        .ok_or_else(|| std::io::Error::other("shared base requires at least one trained target"))?;
    let config = first.target_model.config.clone();
    if trained
        .iter()
        .any(|example| example.target_model.config != config)
    {
        return Err(
            std::io::Error::other("trained target configs differ; cannot share base").into(),
        );
    }
    Ok(config)
}

fn shared_base_source(examples: usize, shared_fit_steps: usize) -> String {
    format!(
        "trained-rust:hyper2d-e2e-shared-base:init=seeded:examples={examples}:shared-fit-steps={shared_fit_steps}",
    )
}

fn shared_base_strategy(shared_fit_steps: usize) -> &'static str {
    if shared_fit_steps > 0 {
        "seeded-shared-base-then-joint-shared-basis-fit"
    } else {
        "seeded-shared-base-no-joint-fit"
    }
}

fn static_adapter_strategy(
    shared_fit_steps: usize,
    adapter_rank: usize,
    exact_adapter_required_rank: usize,
) -> &'static str {
    if shared_fit_steps > 0 {
        "joint-shared-basis-low-rank-adapters"
    } else if adapter_rank >= exact_adapter_required_rank {
        "exact-weight-delta"
    } else {
        "supervised-low-rank-dynamics-distillation"
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
struct DirectFinetuneConfig {
    steps: usize,
    report_interval: usize,
    particle_count: usize,
    rollout_steps: usize,
    update_prob: f32,
    seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
    loss_config: Target2dLossConfig,
    per_parameter_grad_normalization: bool,
    hyper_sgd: SgdConfig,
    adapter_l2_weight: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct DirectFinetuneLoss {
    total: f32,
    image: f32,
    adapter_l2: f32,
}

fn train_direct_image_finetune(
    base: &NpaModel,
    hyper: &mut HyperNpa2d,
    trained: &[Hyper2dTrainedSource],
    hashgrid: &burn_automata_kernels::HashGridConfig,
    config: DirectFinetuneConfig,
) -> Result<CliHyper2dE2eDirectFinetuneReport, Box<dyn std::error::Error>> {
    if trained.is_empty() {
        return Err(
            std::io::Error::other("direct HyperNPA fine-tune requires train examples").into(),
        );
    }
    if config.particle_count == 0 || config.rollout_steps == 0 {
        return Err(std::io::Error::other(
            "direct HyperNPA fine-tune requires non-zero particles and rollout steps",
        )
        .into());
    }
    if !config.adapter_l2_weight.is_finite() || config.adapter_l2_weight < 0.0 {
        return Err(std::io::Error::other(
            "--direct-finetune-adapter-l2 must be finite and non-negative",
        )
        .into());
    }

    let initial = direct_image_pass(base, hyper, trained, hashgrid, config, 0, None)?;
    let mut best_loss = initial.total;
    let mut best_step = 0usize;
    let mut best_hyper = hyper.clone();
    let mut last_loss = initial;
    let mut history = Vec::new();
    for step in 1..=config.steps {
        let mut grads = HyperNpa2dGradients::zeros_like(hyper);
        let loss = direct_image_pass(
            base,
            hyper,
            trained,
            hashgrid,
            config,
            step,
            Some(&mut grads),
        )?;
        let (grad_norm, grad_scale, _) = apply_hyper_sgd(hyper, &grads, config.hyper_sgd)?;
        last_loss = loss;
        if loss.total < best_loss {
            best_loss = loss.total;
            best_step = step;
            best_hyper = hyper.clone();
        }
        if step == config.steps || step.is_multiple_of(config.report_interval.max(1)) {
            history.push(CliHyper2dE2eDirectFinetuneHistoryEntry {
                step,
                loss: loss.total,
                image_loss: loss.image,
                adapter_l2_loss: loss.adapter_l2,
                grad_norm,
                grad_scale,
            });
        }
    }
    if best_loss < last_loss.total {
        *hyper = best_hyper;
    }
    let final_loss = direct_image_pass(
        base,
        hyper,
        trained,
        hashgrid,
        config,
        config.steps + 1,
        None,
    )?;
    Ok(CliHyper2dE2eDirectFinetuneReport {
        objective: "target2d_image_loss_exact_bptt",
        updates: "conditioned_lora_hypernet_only_shared_base_fixed",
        steps: config.steps,
        report_interval: config.report_interval,
        examples: trained.len(),
        particle_count: config.particle_count,
        rollout_steps: config.rollout_steps,
        seed: config.seed,
        seed_scale: config.seed_scale,
        seed_mode: config.seed_mode,
        adapter_l2_weight: config.adapter_l2_weight,
        hyper_sgd: config.hyper_sgd,
        initial_loss: initial.total,
        final_loss: final_loss.total,
        best_loss,
        best_step,
        history,
    })
}

fn direct_image_pass(
    base: &NpaModel,
    hyper: &HyperNpa2d,
    trained: &[Hyper2dTrainedSource],
    hashgrid: &burn_automata_kernels::HashGridConfig,
    config: DirectFinetuneConfig,
    step: usize,
    mut hyper_grads: Option<&mut HyperNpa2dGradients>,
) -> Result<DirectFinetuneLoss, Box<dyn std::error::Error>> {
    let example_scale = 1.0 / trained.len() as f32;
    let mut losses = DirectFinetuneLoss::default();
    for (idx, example) in trained.iter().enumerate() {
        let cache = hyper.forward_cache(&example.condition)?;
        let adapter = NpaLowRankAdapter::from_parameter_vector(
            &hyper.npa_config,
            hyper.config.adapter_rank,
            hyper.config.adapter_alpha,
            cache.output.clone(),
        )?;
        let model = adapter.apply_to_model(base)?;
        let particle_count = example.source.particles.unwrap_or(config.particle_count);
        let update_prob = example.source.update_prob.unwrap_or(config.update_prob);
        let seed_scale = example.source.seed_scale.unwrap_or(config.seed_scale);
        let (loss, full_grads) = target_2d_rollout_loss_with_gradients(
            &model,
            hashgrid,
            &example.target,
            RolloutConfig {
                batch_size: 1,
                particle_count,
                steps: config.rollout_steps,
                update_prob,
                seed: config
                    .seed
                    .wrapping_add((step as u64).wrapping_mul(0x9e37_79b9))
                    .wrapping_add(idx as u64),
                seed_scale,
                ..RolloutConfig::default()
            },
            config.seed_mode,
            config.loss_config,
            config.per_parameter_grad_normalization,
        )?;
        let adapter_l2 = adapter_l2_loss(&cache.output, config.adapter_l2_weight);
        losses.image += loss.total_loss * example_scale;
        losses.adapter_l2 += adapter_l2 * example_scale;
        if let Some(grads) = hyper_grads.as_deref_mut() {
            let adapter_grads = project_low_rank_adapter_gradients(base, &adapter, &full_grads)?;
            let mut output_gradients = adapter_gradient_vector(&adapter_grads);
            add_adapter_l2_gradient(
                &mut output_gradients,
                &cache.output,
                config.adapter_l2_weight,
            );
            hyper.accumulate_output_gradients(&cache, &output_gradients, example_scale, grads)?;
        }
    }
    losses.total = losses.image + losses.adapter_l2;
    Ok(losses)
}

fn adapter_l2_loss(values: &[f32], weight: f32) -> f32 {
    if weight == 0.0 || values.is_empty() {
        return 0.0;
    }
    weight * values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32
}

fn add_adapter_l2_gradient(output_gradients: &mut [f32], values: &[f32], weight: f32) {
    if weight == 0.0 || values.is_empty() {
        return;
    }
    let scale = 2.0 * weight / values.len() as f32;
    for (gradient, value) in output_gradients.iter_mut().zip(values.iter().copied()) {
        *gradient += scale * value;
    }
}

fn empty_adapter_bootstrap() -> Hyper2dAdapterBootstrapResult {
    Hyper2dAdapterBootstrapResult {
        examples: Vec::new(),
        reports: Vec::new(),
    }
}

fn zero_adapter_bootstrap(
    base: &NpaModel,
    loaded: &[Hyper2dLoadedExample],
    adapter_rank: usize,
    adapter_alpha: f32,
) -> Result<Hyper2dAdapterBootstrapResult, Box<dyn std::error::Error>> {
    let mut examples = Vec::with_capacity(loaded.len());
    let mut reports = Vec::with_capacity(loaded.len());
    for example in loaded {
        let adapter = NpaLowRankAdapter::zeros(&base.config, adapter_rank, adapter_alpha);
        let loss = supervised_adapter_loss(base, &adapter, &example.batch)?;
        reports.push(CliHyper2dAdapterBootstrapReport {
            slug: example.descriptor.slug.clone(),
            method: "zero-adapter-no-holdout-oracle",
            steps: 0,
            rows: example.rows,
            initial_loss: loss,
            final_loss: loss,
            best_loss: loss,
            adapter_parameter_count: adapter.parameter_count(),
        });
        examples.push(HyperAdapterExample2d {
            condition: example.condition.clone(),
            target_adapter: adapter,
        });
    }
    Ok(Hyper2dAdapterBootstrapResult { examples, reports })
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
    split: Hyper2dE2eSplit,
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
            split: split.label(),
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

#[derive(Clone, Copy)]
struct E2eEvalConfig {
    split: Hyper2dE2eSplit,
    rollout: EvalConfig,
}

fn evaluate_e2e_models(
    trained: &[Hyper2dTrainedSource],
    base: &NpaModel,
    hyper: &HyperNpa2d,
    adapter_examples: &[HyperAdapterExample2d],
    hashgrid: &burn_automata_kernels::HashGridConfig,
    loss_config: Target2dLossConfig,
    config: E2eEvalConfig,
) -> Result<Vec<CliHyper2dE2eEvalReport>, Box<dyn std::error::Error>> {
    if trained.len() != adapter_examples.len() {
        return Err(std::io::Error::other("eval examples do not match adapters").into());
    }
    let mut reports = Vec::with_capacity(trained.len());
    for (idx, (example, adapter_example)) in trained.iter().zip(adapter_examples).enumerate() {
        let rollout_config = config.rollout;
        let seed = rollout_config.seed.wrapping_add(idx as u64);
        let update_prob = example
            .source
            .update_prob
            .unwrap_or(rollout_config.update_prob);
        let seed_scale = example
            .source
            .seed_scale
            .unwrap_or(rollout_config.seed_scale);
        let particle_count = example
            .source
            .particles
            .unwrap_or(rollout_config.particle_count);
        let trained_target_loss = evaluate_model_target_loss(
            &example.target_model,
            hashgrid,
            &example.target,
            loss_config,
            EvalConfig {
                particle_count,
                rollout_steps: rollout_config.rollout_steps,
                update_prob,
                seed,
                seed_scale,
                seed_mode: rollout_config.seed_mode,
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
                rollout_steps: rollout_config.rollout_steps,
                update_prob,
                seed,
                seed_scale,
                seed_mode: rollout_config.seed_mode,
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
                rollout_steps: rollout_config.rollout_steps,
                update_prob,
                seed,
                seed_scale,
                seed_mode: rollout_config.seed_mode,
            },
        )?;
        reports.push(CliHyper2dE2eEvalReport {
            slug: example.source.slug.clone(),
            split: config.split.label(),
            condition: example.source.condition_path.display().to_string(),
            particle_count,
            rollout_steps: rollout_config.rollout_steps,
            update_prob,
            seed,
            seed_scale,
            seed_mode: rollout_config.seed_mode,
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

#[derive(Clone, Copy)]
struct QualityGateConfig {
    max_static_ratio: Option<f32>,
    max_hyper_static_ratio: Option<f32>,
    max_hyper_target_ratio: Option<f32>,
}

fn summarize_e2e_quality(
    eval: &[CliHyper2dE2eEvalReport],
    gates: QualityGateConfig,
) -> CliHyper2dE2eQualityReport {
    let static_ratios = eval
        .iter()
        .filter_map(|report| report.static_adapter_ratio_to_trained_target)
        .collect::<Vec<_>>();
    let hyper_static_ratios = eval
        .iter()
        .filter_map(|report| report.hyper_ratio_to_static_adapter)
        .collect::<Vec<_>>();
    let hyper_target_ratios = eval
        .iter()
        .filter_map(|report| report.hyper_ratio_to_trained_target)
        .collect::<Vec<_>>();
    let max_static_adapter_gap_to_trained_target = max_metric(
        eval.iter()
            .map(|report| report.static_adapter_gap_to_trained_target),
    );
    let max_hyper_gap_to_static_adapter =
        max_metric(eval.iter().map(|report| report.hyper_gap_to_static_adapter));
    let max_hyper_gap_to_trained_target =
        max_metric(eval.iter().map(|report| report.hyper_gap_to_trained_target));
    let max_static_adapter_ratio_to_trained_target = max_metric(static_ratios.iter().copied());
    let max_hyper_ratio_to_static_adapter = max_metric(hyper_static_ratios.iter().copied());
    let max_hyper_ratio_to_trained_target = max_metric(hyper_target_ratios.iter().copied());
    let passed = threshold_passed(
        max_static_adapter_ratio_to_trained_target,
        gates.max_static_ratio,
    ) && threshold_passed(
        max_hyper_ratio_to_static_adapter,
        gates.max_hyper_static_ratio,
    ) && threshold_passed(
        max_hyper_ratio_to_trained_target,
        gates.max_hyper_target_ratio,
    );
    CliHyper2dE2eQualityReport {
        examples: eval.len(),
        mean_static_adapter_ratio_to_trained_target: mean_metric(static_ratios.iter().copied()),
        max_static_adapter_ratio_to_trained_target,
        mean_hyper_ratio_to_static_adapter: mean_metric(hyper_static_ratios.iter().copied()),
        max_hyper_ratio_to_static_adapter,
        mean_hyper_ratio_to_trained_target: mean_metric(hyper_target_ratios.iter().copied()),
        max_hyper_ratio_to_trained_target,
        max_static_adapter_gap_to_trained_target,
        max_hyper_gap_to_static_adapter,
        max_hyper_gap_to_trained_target,
        max_static_ratio_threshold: gates.max_static_ratio,
        max_hyper_static_ratio_threshold: gates.max_hyper_static_ratio,
        max_hyper_target_ratio_threshold: gates.max_hyper_target_ratio,
        passed,
    }
}

fn threshold_passed(value: Option<f32>, threshold: Option<f32>) -> bool {
    match (value, threshold) {
        (_, None) => true,
        (Some(value), Some(threshold)) => value <= threshold,
        (None, Some(_)) => false,
    }
}

fn quality_failure_message(quality: &CliHyper2dE2eQualityReport) -> Option<String> {
    if quality.passed {
        return None;
    }
    Some(format!(
        "HyperNPA e2e quality gates failed: max_static_ratio={:?}/{:?} max_hyper_static_ratio={:?}/{:?} max_hyper_target_ratio={:?}/{:?}",
        quality.max_static_adapter_ratio_to_trained_target,
        quality.max_static_ratio_threshold,
        quality.max_hyper_ratio_to_static_adapter,
        quality.max_hyper_static_ratio_threshold,
        quality.max_hyper_ratio_to_trained_target,
        quality.max_hyper_target_ratio_threshold,
    ))
}

fn mean_metric(values: impl Iterator<Item = f32>) -> Option<f32> {
    let mut sum = 0.0_f32;
    let mut count = 0_usize;
    for value in values {
        if value.is_finite() {
            sum += value;
            count += 1;
        }
    }
    (count > 0).then_some(sum / count as f32)
}

fn max_metric(values: impl Iterator<Item = f32>) -> Option<f32> {
    let mut max_value = None::<f32>;
    for value in values {
        if value.is_finite() {
            max_value = Some(max_value.map_or(value, |current| current.max(value)));
        }
    }
    max_value
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

    fn source(slug: &str) -> Hyper2dScratchSource {
        Hyper2dScratchSource {
            slug: slug.to_string(),
            title: None,
            group: Some("growing".to_string()),
            condition_path: PathBuf::from(format!("{slug}.png")),
            particles: None,
            seed_scale: None,
            update_prob: None,
        }
    }

    #[test]
    fn e2e_split_respects_explicit_holdout_targets() {
        let sources = vec![source("lizard"), source("ghost"), source("frog_face")];
        let splits = resolve_e2e_splits(&sources, &["ghost".to_string()], 0, 0).unwrap();

        assert_eq!(
            splits,
            vec![
                Hyper2dE2eSplit::Train,
                Hyper2dE2eSplit::Holdout,
                Hyper2dE2eSplit::Train
            ]
        );
    }

    #[test]
    fn e2e_split_respects_stride_and_rejects_all_holdout() {
        let sources = vec![source("lizard"), source("ghost"), source("frog_face")];
        let splits = resolve_e2e_splits(&sources, &[], 2, 1).unwrap();

        assert_eq!(
            splits,
            vec![
                Hyper2dE2eSplit::Train,
                Hyper2dE2eSplit::Holdout,
                Hyper2dE2eSplit::Train
            ]
        );
        assert!(resolve_e2e_splits(&sources, &[], 1, 0).is_err());
    }
}
