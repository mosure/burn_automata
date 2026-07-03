use crate::cli::prelude::*;

use super::shared_basis::{
    add_adapter_l2_gradients, add_scaled_model_gradients, normalized_example_batch_size,
    sample_example_indices, zero_model_gradients,
};
use super::sources::{
    OmniSvgSourceConfig, ScratchSourceResolveConfig, resolve_scratch_sources, sanitize_slug,
};
use super::{Hyper2dE2eSplit, resolve_e2e_splits};
use crate::cli::commands::hyper_support::write_pretty_json;

mod burn_wgpu;

#[derive(Clone)]
struct DirectBasisExample {
    source: super::sources::Hyper2dScratchSource,
    split: Hyper2dE2eSplit,
    target: TargetImage2d,
    adapter: NpaLowRankAdapter,
    last_train_loss: Option<f32>,
}

#[derive(Clone, Copy)]
struct DirectBasisTrainConfig {
    steps: usize,
    report_interval: usize,
    example_batch_size: usize,
    rollout_particles: usize,
    rollout_steps: usize,
    update_prob: f32,
    seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
    loss_config: Target2dLossConfig,
    per_parameter_grad_normalization: bool,
    base_sgd: SgdConfig,
    adapter_sgd: SgdConfig,
    adapter_l2_weight: f32,
    update_base: bool,
    eval_examples: usize,
    eval_seed: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct DirectBasisStepStats {
    loss: f32,
    base_grad_norm: f32,
    base_grad_scale: f32,
    mean_adapter_grad_norm: f32,
    max_adapter_grad_norm: f32,
    examples_seen: usize,
    particle_steps_per_sec: f64,
    elapsed_ms: f64,
}

struct DirectBasisPhaseReport {
    history: Vec<CliHyper2dDirectBasisHistoryEntry>,
    best_loss: Option<f32>,
    best_step: usize,
}

#[derive(Serialize)]
struct DirectBasisAdapterBankManifest {
    base_model: String,
    adapter_rank: usize,
    adapter_alpha: f32,
    entries: Vec<CliHyper2dDirectBasisAdapterReport>,
}

#[derive(Serialize)]
struct DirectBasisSourceManifest {
    sources: Vec<DirectBasisSourceEntry>,
}

#[derive(Serialize)]
struct DirectBasisSourceEntry {
    slug: String,
    split: &'static str,
    title: Option<String>,
    group: Option<String>,
    path: String,
}

#[derive(Deserialize)]
struct GpuDirectBasisPayload {
    backend: String,
    upstream_root: Option<String>,
    device: String,
    torch_version: Option<String>,
    cuda_version: Option<String>,
    gpu_name: Option<String>,
    train_examples: usize,
    holdout_examples: usize,
    initial_train_loss: Option<CliHyper2dDirectBasisLossSummary>,
    final_train_loss: Option<CliHyper2dDirectBasisLossSummary>,
    initial_holdout_loss: Option<CliHyper2dDirectBasisLossSummary>,
    final_holdout_loss: Option<CliHyper2dDirectBasisLossSummary>,
    best_train_loss: Option<f32>,
    best_train_step: usize,
    history: Vec<CliHyper2dDirectBasisHistoryEntry>,
    holdout_history: Vec<CliHyper2dDirectBasisHistoryEntry>,
    base: GpuDirectBasisBaseWeights,
    adapters: Vec<GpuDirectBasisAdapter>,
}

#[derive(Deserialize)]
struct GpuDirectBasisBaseWeights {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

#[derive(Deserialize)]
struct GpuDirectBasisAdapter {
    slug: String,
    split: String,
    title: Option<String>,
    group: Option<String>,
    condition: String,
    #[serde(default)]
    target_source_width: usize,
    #[serde(default)]
    target_source_height: usize,
    target_points: usize,
    last_train_loss: Option<f32>,
    adapter: Vec<f32>,
}

#[allow(clippy::too_many_lines)]
pub(crate) fn run_train_hyper_2d_direct_basis(
    command: Command,
) -> Result<(), Box<dyn std::error::Error>> {
    let Command::TrainHyper2dDirectBasis {
        preset,
        target_images,
        target_image_dirs,
        target_image_recursive,
        image_extensions,
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
        source_limit,
        holdout_targets,
        holdout_stride,
        holdout_offset,
        output_dir,
        report_output,
        shared_base_output,
        adapter_bank_output,
        adapter_output_dir,
        training_device,
        gpu_backend,
        python,
        gpu_upstream_root,
        gpu_device,
        gpu_payload_output,
        adapter_rank,
        adapter_alpha,
        steps,
        report_interval,
        example_batch_size,
        rollout_particles,
        rollout_steps,
        update_prob,
        seed,
        base_seed,
        seed_scale,
        seed_mode,
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
        per_parameter_grad_normalization,
        base_learning_rate,
        base_weight_decay,
        base_grad_clip_norm,
        adapter_learning_rate,
        adapter_weight_decay,
        adapter_grad_clip_norm,
        adapter_l2,
        holdout_adapter_steps,
        holdout_adapter_batch_size,
        eval_examples,
        eval_seed,
    } = command
    else {
        unreachable!("run_train_hyper_2d_direct_basis called with the wrong command variant");
    };

    let preset_arg = preset;
    let preset: AutomataPreset = preset.into();
    if preset != AutomataPreset::Growing2d {
        return Err(std::io::Error::other(
            "train-hyper2d-direct-basis currently supports the growing-2d target image objective",
        )
        .into());
    }
    validate_direct_basis_args(DirectBasisArgCheck {
        adapter_rank,
        adapter_alpha,
        rollout_particles,
        rollout_steps,
        update_prob,
        base_learning_rate,
        adapter_learning_rate,
        adapter_l2,
    })?;

    let seed_mode: ParticleSeed = seed_mode.into();
    let seed_scale = seed_scale.unwrap_or_else(|| NpaConfig::seed_scale_for_preset(preset));
    let report_output = report_output.unwrap_or_else(|| output_dir.join("report.json"));
    let shared_base_output =
        shared_base_output.unwrap_or_else(|| output_dir.join("shared_base.bpk"));
    let adapter_bank_output =
        adapter_bank_output.unwrap_or_else(|| output_dir.join("adapter_bank.json"));
    let adapter_output_dir = adapter_output_dir.unwrap_or_else(|| output_dir.join("adapters"));
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

    let mut sources = resolve_scratch_sources(ScratchSourceResolveConfig {
        preset: preset_arg,
        target_images: &target_images,
        target_image_dirs: &target_image_dirs,
        target_image_recursive,
        image_extensions: &image_extensions,
        catalog: catalog.as_ref(),
        catalog_thumbnail_dir: &catalog_thumbnail_dir,
        catalog_group,
        catalog_targets: &catalog_targets,
        catalog_limit,
        omnisvg: omnisvg_source,
    })?;
    if source_limit > 0 && sources.len() > source_limit {
        sources.truncate(source_limit);
    }
    let splits = resolve_e2e_splits(&sources, &holdout_targets, holdout_stride, holdout_offset)?;

    let hashgrid = upstream_growing_2d_hashgrid();
    let loss_config = super::super::target2d::target2d_loss_config(
        target_loss_image_size,
        target_splat_sigma,
        target_splat_loss_weight,
        target_color_loss_weight,
        target_density_loss_weight,
        target_displacement_regularizer_weight,
        target_overflow_regularizer_weight,
        target_bound_regularizer_weight,
    )?;
    if training_device != TrainingDeviceArg::Cpu {
        let request = GpuDirectBasisRunRequest {
            preset,
            requested_training_device: training_device,
            target_images: &target_images,
            target_image_dirs: &target_image_dirs,
            target_image_recursive,
            image_extensions,
            catalog: catalog.as_ref(),
            catalog_group,
            catalog_targets,
            omnisvg: super::omnisvg_source_report(omnisvg_source),
            source_limit,
            holdout_targets,
            holdout_stride,
            holdout_offset,
            output_dir: &output_dir,
            report_output: &report_output,
            shared_base_output: &shared_base_output,
            adapter_bank_output: &adapter_bank_output,
            adapter_output_dir: &adapter_output_dir,
            python: &python,
            gpu_upstream_root: gpu_upstream_root.as_ref(),
            gpu_device,
            gpu_payload_output: gpu_payload_output
                .unwrap_or_else(|| output_dir.join("gpu_direct_basis_payload.json")),
            sources: &sources,
            splits: &splits,
            hashgrid,
            loss_config,
            adapter_rank,
            adapter_alpha,
            steps,
            report_interval,
            example_batch_size,
            rollout_particles,
            rollout_steps,
            update_prob,
            seed,
            base_seed,
            seed_scale,
            seed_mode,
            target_points,
            target_image_size,
            target_threshold,
            per_parameter_grad_normalization,
            base_sgd: SgdConfig {
                learning_rate: base_learning_rate,
                weight_decay: base_weight_decay,
                grad_clip_norm: base_grad_clip_norm,
            },
            adapter_sgd: SgdConfig {
                learning_rate: adapter_learning_rate,
                weight_decay: adapter_weight_decay,
                grad_clip_norm: adapter_grad_clip_norm,
            },
            adapter_l2,
            holdout_adapter_steps,
            holdout_adapter_batch_size,
            eval_examples,
            eval_seed,
        };
        return match gpu_backend {
            Hyper2dDirectBasisGpuBackendArg::BurnWgpu => run_burn_wgpu_direct_basis(request),
            Hyper2dDirectBasisGpuBackendArg::UpstreamPython => run_python_gpu_direct_basis(request),
        };
    }
    let mut base = NpaModel::upstream_seeded(NpaConfig::growing_2d(), base_seed);
    base.validate()?;
    let examples = load_direct_basis_examples(
        &sources,
        &splits,
        &base.config,
        adapter_rank,
        adapter_alpha,
        seed,
        DirectBasisTargetConfig {
            threshold: target_threshold,
            points: target_points,
            image_size: target_image_size,
        },
    )?;
    let (mut train_examples, mut holdout_examples): (Vec<_>, Vec<_>) = examples
        .into_iter()
        .partition(|example| example.split == Hyper2dE2eSplit::Train);
    if train_examples.is_empty() {
        return Err(
            std::io::Error::other("train-hyper2d-direct-basis requires train examples").into(),
        );
    }

    let base_sgd = SgdConfig {
        learning_rate: base_learning_rate,
        weight_decay: base_weight_decay,
        grad_clip_norm: base_grad_clip_norm,
    };
    let adapter_sgd = SgdConfig {
        learning_rate: adapter_learning_rate,
        weight_decay: adapter_weight_decay,
        grad_clip_norm: adapter_grad_clip_norm,
    };
    let train_config = DirectBasisTrainConfig {
        steps,
        report_interval,
        example_batch_size,
        rollout_particles,
        rollout_steps,
        update_prob,
        seed,
        seed_scale,
        seed_mode,
        loss_config,
        per_parameter_grad_normalization,
        base_sgd,
        adapter_sgd,
        adapter_l2_weight: adapter_l2,
        update_base: true,
        eval_examples,
        eval_seed,
    };
    let initial_train_loss = evaluate_direct_basis_examples(
        &base,
        &train_examples,
        &hashgrid,
        train_config,
        eval_examples,
        eval_seed,
    )?;
    let initial_holdout_loss = evaluate_direct_basis_examples(
        &base,
        &holdout_examples,
        &hashgrid,
        train_config,
        eval_examples,
        eval_seed ^ 0x90_1d_2d,
    )?;
    let train_phase =
        train_direct_basis_phase(&mut base, &mut train_examples, &hashgrid, train_config)?;
    let holdout_config = DirectBasisTrainConfig {
        steps: holdout_adapter_steps,
        example_batch_size: holdout_adapter_batch_size,
        update_base: false,
        seed: seed ^ 0x90_1d_2d,
        eval_seed: eval_seed ^ 0x90_1d_2d,
        ..train_config
    };
    let holdout_phase =
        train_direct_basis_phase(&mut base, &mut holdout_examples, &hashgrid, holdout_config)?;
    let final_train_loss = evaluate_direct_basis_examples(
        &base,
        &train_examples,
        &hashgrid,
        train_config,
        eval_examples,
        eval_seed,
    )?;
    let final_holdout_loss = evaluate_direct_basis_examples(
        &base,
        &holdout_examples,
        &hashgrid,
        train_config,
        eval_examples,
        eval_seed ^ 0x90_1d_2d,
    )?;

    let base_manifest = BpkModelManifest::from_model(
        &base,
        hashgrid.clone(),
        Some(format!(
            "trained-rust:hyper2d-direct-basis:sources={}:steps={steps}",
            train_examples.len()
        )),
    );
    crate::import::save_manifest(&shared_base_output, &base_manifest)?;
    let adapter_reports = save_direct_basis_adapters(
        &base_manifest,
        &shared_base_output,
        &adapter_output_dir,
        train_examples
            .iter()
            .chain(holdout_examples.iter())
            .collect::<Vec<_>>()
            .as_slice(),
    )?;
    let adapter_bank = DirectBasisAdapterBankManifest {
        base_model: shared_base_output.display().to_string(),
        adapter_rank,
        adapter_alpha,
        entries: adapter_reports.clone(),
    };
    write_pretty_json(&adapter_bank_output, &adapter_bank)?;

    let report = CliHyper2dDirectBasisTrainingReport {
        preset,
        target_images: target_images
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        target_image_dirs: target_image_dirs
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        target_image_recursive,
        image_extensions,
        catalog: catalog.as_ref().map(|path| path.display().to_string()),
        catalog_group,
        catalog_targets,
        omnisvg: super::omnisvg_source_report(omnisvg_source),
        source_limit,
        holdout_targets,
        holdout_stride,
        holdout_offset,
        output_dir: output_dir.display().to_string(),
        report_output: report_output.display().to_string(),
        shared_base_output: shared_base_output.display().to_string(),
        adapter_bank_output: adapter_bank_output.display().to_string(),
        adapter_output_dir: adapter_output_dir.display().to_string(),
        requested_training_device: training_device,
        training_device: TrainingDeviceArg::Cpu,
        gpu_training: None,
        npa_config: base.config.clone(),
        hashgrid,
        target_loss_config: loss_config,
        adapter_rank,
        adapter_alpha,
        train_examples: train_examples.len(),
        holdout_examples: holdout_examples.len(),
        steps,
        report_interval,
        example_batch_size: normalized_example_batch_size(example_batch_size, train_examples.len()),
        rollout_particles,
        rollout_steps,
        update_prob,
        seed,
        seed_scale,
        seed_mode,
        per_parameter_grad_normalization,
        base_sgd,
        adapter_sgd,
        adapter_l2_weight: adapter_l2,
        holdout_adapter_steps,
        holdout_adapter_batch_size: normalized_example_batch_size(
            holdout_adapter_batch_size,
            holdout_examples.len().max(1),
        ),
        eval_examples,
        initial_train_loss,
        final_train_loss,
        initial_holdout_loss,
        final_holdout_loss,
        best_train_loss: train_phase.best_loss,
        best_train_step: train_phase.best_step,
        history: train_phase.history,
        holdout_history: holdout_phase.history,
        adapters: adapter_reports,
    };
    write_pretty_json(&report_output, &report)?;
    println!(
        "wrote {} train={} holdout={} shared_base={} adapter_bank={}",
        report_output.display(),
        report.train_examples,
        report.holdout_examples,
        shared_base_output.display(),
        adapter_bank_output.display()
    );
    Ok(())
}

struct GpuDirectBasisRunRequest<'a> {
    preset: AutomataPreset,
    requested_training_device: TrainingDeviceArg,
    target_images: &'a [PathBuf],
    target_image_dirs: &'a [PathBuf],
    target_image_recursive: bool,
    image_extensions: Vec<String>,
    catalog: Option<&'a PathBuf>,
    catalog_group: Option<Hyper2dCatalogGroupArg>,
    catalog_targets: Vec<String>,
    omnisvg: Option<CliOmniSvgSourceReport>,
    source_limit: usize,
    holdout_targets: Vec<String>,
    holdout_stride: usize,
    holdout_offset: usize,
    output_dir: &'a Path,
    report_output: &'a Path,
    shared_base_output: &'a Path,
    adapter_bank_output: &'a Path,
    adapter_output_dir: &'a Path,
    python: &'a Path,
    gpu_upstream_root: Option<&'a PathBuf>,
    gpu_device: String,
    gpu_payload_output: PathBuf,
    sources: &'a [super::sources::Hyper2dScratchSource],
    splits: &'a [Hyper2dE2eSplit],
    hashgrid: burn_automata_kernels::HashGridConfig,
    loss_config: Target2dLossConfig,
    adapter_rank: usize,
    adapter_alpha: f32,
    steps: usize,
    report_interval: usize,
    example_batch_size: usize,
    rollout_particles: usize,
    rollout_steps: usize,
    update_prob: f32,
    seed: u64,
    base_seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
    target_points: usize,
    target_image_size: Option<usize>,
    target_threshold: f32,
    per_parameter_grad_normalization: bool,
    base_sgd: SgdConfig,
    adapter_sgd: SgdConfig,
    adapter_l2: f32,
    holdout_adapter_steps: usize,
    holdout_adapter_batch_size: usize,
    eval_examples: usize,
    eval_seed: u64,
}

fn run_burn_wgpu_direct_basis(
    request: GpuDirectBasisRunRequest<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut base = NpaModel::upstream_seeded(NpaConfig::growing_2d(), request.base_seed);
    base.validate()?;
    let examples = load_direct_basis_examples(
        request.sources,
        request.splits,
        &base.config,
        request.adapter_rank,
        request.adapter_alpha,
        request.seed,
        DirectBasisTargetConfig {
            threshold: request.target_threshold,
            points: request.target_points,
            image_size: request.target_image_size,
        },
    )?;
    let (mut train_examples, mut holdout_examples): (Vec<_>, Vec<_>) = examples
        .into_iter()
        .partition(|example| example.split == Hyper2dE2eSplit::Train);
    if train_examples.is_empty() {
        return Err(
            std::io::Error::other("train-hyper2d-direct-basis requires train examples").into(),
        );
    }

    let train_config = DirectBasisTrainConfig {
        steps: request.steps,
        report_interval: request.report_interval,
        example_batch_size: request.example_batch_size,
        rollout_particles: request.rollout_particles,
        rollout_steps: request.rollout_steps,
        update_prob: request.update_prob,
        seed: request.seed,
        seed_scale: request.seed_scale,
        seed_mode: request.seed_mode,
        loss_config: request.loss_config,
        per_parameter_grad_normalization: request.per_parameter_grad_normalization,
        base_sgd: request.base_sgd,
        adapter_sgd: request.adapter_sgd,
        adapter_l2_weight: request.adapter_l2,
        update_base: true,
        eval_examples: request.eval_examples,
        eval_seed: request.eval_seed,
    };
    let initial_train_loss = evaluate_direct_basis_examples(
        &base,
        &train_examples,
        &request.hashgrid,
        train_config,
        request.eval_examples,
        request.eval_seed,
    )?;
    let initial_holdout_loss = evaluate_direct_basis_examples(
        &base,
        &holdout_examples,
        &request.hashgrid,
        train_config,
        request.eval_examples,
        request.eval_seed ^ 0x90_1d_2d,
    )?;
    let holdout_config = DirectBasisTrainConfig {
        steps: request.holdout_adapter_steps,
        example_batch_size: request.holdout_adapter_batch_size,
        update_base: false,
        seed: request.seed ^ 0x90_1d_2d,
        eval_seed: request.eval_seed ^ 0x90_1d_2d,
        ..train_config
    };
    let burn_report = burn_wgpu::train_direct_basis_burn_wgpu(
        &mut base,
        &mut train_examples,
        &mut holdout_examples,
        train_config,
        holdout_config,
    )?;
    let final_train_loss = evaluate_direct_basis_examples(
        &base,
        &train_examples,
        &request.hashgrid,
        train_config,
        request.eval_examples,
        request.eval_seed,
    )?;
    let final_holdout_loss = evaluate_direct_basis_examples(
        &base,
        &holdout_examples,
        &request.hashgrid,
        train_config,
        request.eval_examples,
        request.eval_seed ^ 0x90_1d_2d,
    )?;

    let base_manifest = BpkModelManifest::from_model(
        &base,
        request.hashgrid.clone(),
        Some(format!(
            "trained-rust:hyper2d-direct-basis:burn-wgpu:sources={}:steps={}",
            train_examples.len(),
            request.steps
        )),
    );
    crate::import::save_manifest(request.shared_base_output, &base_manifest)?;
    let adapter_reports = save_direct_basis_adapters(
        &base_manifest,
        request.shared_base_output,
        request.adapter_output_dir,
        train_examples
            .iter()
            .chain(holdout_examples.iter())
            .collect::<Vec<_>>()
            .as_slice(),
    )?;
    let adapter_bank = DirectBasisAdapterBankManifest {
        base_model: request.shared_base_output.display().to_string(),
        adapter_rank: request.adapter_rank,
        adapter_alpha: request.adapter_alpha,
        entries: adapter_reports.clone(),
    };
    write_pretty_json(request.adapter_bank_output, &adapter_bank)?;
    let report = CliHyper2dDirectBasisTrainingReport {
        preset: request.preset,
        target_images: request
            .target_images
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        target_image_dirs: request
            .target_image_dirs
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        target_image_recursive: request.target_image_recursive,
        image_extensions: request.image_extensions,
        catalog: request.catalog.map(|path| path.display().to_string()),
        catalog_group: request.catalog_group,
        catalog_targets: request.catalog_targets,
        omnisvg: request.omnisvg,
        source_limit: request.source_limit,
        holdout_targets: request.holdout_targets,
        holdout_stride: request.holdout_stride,
        holdout_offset: request.holdout_offset,
        output_dir: request.output_dir.display().to_string(),
        report_output: request.report_output.display().to_string(),
        shared_base_output: request.shared_base_output.display().to_string(),
        adapter_bank_output: request.adapter_bank_output.display().to_string(),
        adapter_output_dir: request.adapter_output_dir.display().to_string(),
        requested_training_device: request.requested_training_device,
        training_device: TrainingDeviceArg::Gpu,
        gpu_training: Some(CliHyper2dDirectBasisGpuTrainingReport {
            backend: burn_report.backend.to_string(),
            python: None,
            device: burn_report.device,
            upstream_root: None,
            payload_output: None,
            gpu_name: None,
            torch_version: None,
            cuda_version: None,
            metrics: burn_report.metrics,
        }),
        npa_config: base.config.clone(),
        hashgrid: request.hashgrid,
        target_loss_config: request.loss_config,
        adapter_rank: request.adapter_rank,
        adapter_alpha: request.adapter_alpha,
        train_examples: train_examples.len(),
        holdout_examples: holdout_examples.len(),
        steps: request.steps,
        report_interval: request.report_interval,
        example_batch_size: normalized_example_batch_size(
            request.example_batch_size,
            train_examples.len(),
        ),
        rollout_particles: request.rollout_particles,
        rollout_steps: request.rollout_steps,
        update_prob: request.update_prob,
        seed: request.seed,
        seed_scale: request.seed_scale,
        seed_mode: request.seed_mode,
        per_parameter_grad_normalization: request.per_parameter_grad_normalization,
        base_sgd: request.base_sgd,
        adapter_sgd: request.adapter_sgd,
        adapter_l2_weight: request.adapter_l2,
        holdout_adapter_steps: request.holdout_adapter_steps,
        holdout_adapter_batch_size: normalized_example_batch_size(
            request.holdout_adapter_batch_size,
            holdout_examples.len().max(1),
        ),
        eval_examples: request.eval_examples,
        initial_train_loss,
        final_train_loss,
        initial_holdout_loss,
        final_holdout_loss,
        best_train_loss: burn_report.best_train_loss,
        best_train_step: burn_report.best_train_step,
        history: burn_report.history,
        holdout_history: burn_report.holdout_history,
        adapters: adapter_reports,
    };
    write_pretty_json(request.report_output, &report)?;
    println!(
        "wrote {} train={} holdout={} shared_base={} adapter_bank={} backend=burn-wgpu",
        request.report_output.display(),
        report.train_examples,
        report.holdout_examples,
        request.shared_base_output.display(),
        request.adapter_bank_output.display()
    );
    Ok(())
}

fn run_python_gpu_direct_basis(
    request: GpuDirectBasisRunRequest<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/train_hyper2d_direct_basis_gpu.py");
    let sources_output = request.output_dir.join("gpu_direct_basis_sources.json");
    write_gpu_sources(&sources_output, request.sources, request.splits)?;
    let mut command = std::process::Command::new(request.python);
    command
        .arg(&script)
        .arg("--sources-json")
        .arg(&sources_output)
        .arg("--payload-output")
        .arg(&request.gpu_payload_output)
        .arg("--device")
        .arg(&request.gpu_device)
        .arg("--steps")
        .arg(request.steps.to_string())
        .arg("--report-interval")
        .arg(request.report_interval.to_string())
        .arg("--example-batch-size")
        .arg(request.example_batch_size.to_string())
        .arg("--rollout-particles")
        .arg(request.rollout_particles.to_string())
        .arg("--rollout-steps")
        .arg(request.rollout_steps.to_string())
        .arg("--update-prob")
        .arg(request.update_prob.to_string())
        .arg("--seed")
        .arg(request.seed.to_string())
        .arg("--base-seed")
        .arg(request.base_seed.to_string())
        .arg("--seed-scale")
        .arg(request.seed_scale.to_string())
        .arg("--seed-mode")
        .arg(upstream_seed_mode(request.seed_mode)?)
        .arg("--adapter-rank")
        .arg(request.adapter_rank.to_string())
        .arg("--adapter-alpha")
        .arg(request.adapter_alpha.to_string())
        .arg("--target-points")
        .arg(request.target_points.to_string())
        .arg("--target-threshold")
        .arg(request.target_threshold.to_string())
        .arg("--image-size")
        .arg(request.loss_config.image_size.to_string())
        .arg("--splat-sigma")
        .arg(request.loss_config.sigma.to_string())
        .arg("--splat-loss-weight")
        .arg(request.loss_config.splat_loss_weight.to_string())
        .arg("--color-loss-weight")
        .arg(request.loss_config.color_loss_weight.to_string())
        .arg("--density-loss-weight")
        .arg(request.loss_config.density_loss_weight.to_string())
        .arg("--displacement-regularizer-weight")
        .arg(
            request
                .loss_config
                .displacement_regularizer_weight
                .to_string(),
        )
        .arg("--overflow-regularizer-weight")
        .arg(request.loss_config.overflow_regularizer_weight.to_string())
        .arg("--bound-regularizer-weight")
        .arg(request.loss_config.bound_regularizer_weight.to_string())
        .arg("--base-learning-rate")
        .arg(request.base_sgd.learning_rate.to_string())
        .arg("--base-weight-decay")
        .arg(request.base_sgd.weight_decay.to_string())
        .arg("--base-grad-clip-norm")
        .arg(request.base_sgd.grad_clip_norm.to_string())
        .arg("--adapter-learning-rate")
        .arg(request.adapter_sgd.learning_rate.to_string())
        .arg("--adapter-weight-decay")
        .arg(request.adapter_sgd.weight_decay.to_string())
        .arg("--adapter-grad-clip-norm")
        .arg(request.adapter_sgd.grad_clip_norm.to_string())
        .arg("--adapter-l2")
        .arg(request.adapter_l2.to_string())
        .arg("--holdout-adapter-steps")
        .arg(request.holdout_adapter_steps.to_string())
        .arg("--holdout-adapter-batch-size")
        .arg(request.holdout_adapter_batch_size.to_string())
        .arg("--eval-examples")
        .arg(request.eval_examples.to_string())
        .arg("--eval-seed")
        .arg(request.eval_seed.to_string());
    if let Some(upstream_root) = request.gpu_upstream_root {
        command.arg("--upstream-root").arg(upstream_root);
    }
    if let Some(target_image_size) = request.target_image_size {
        command
            .arg("--target-image-size")
            .arg(target_image_size.to_string());
    }
    if request.per_parameter_grad_normalization {
        command.arg("--normalize-grads");
    } else {
        command.arg("--no-normalize-grads");
    }
    let output = command.output()?;
    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "GPU direct-basis training failed with status {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ))
        .into());
    }
    let payload_text = std::fs::read_to_string(&request.gpu_payload_output)?;
    let metrics: serde_json::Value = serde_json::from_str(&payload_text)?;
    let payload: GpuDirectBasisPayload = serde_json::from_value(metrics.clone())?;
    let config = NpaConfig::growing_2d();
    let base = NpaModel {
        config: config.clone(),
        weights: NpaWeights {
            w1: payload.base.w1,
            b1: payload.base.b1,
            w2: payload.base.w2,
            b2: payload.base.b2,
        },
    };
    base.validate()?;
    let base_manifest = BpkModelManifest::from_model(
        &base,
        request.hashgrid.clone(),
        Some(format!(
            "trained-rust:hyper2d-direct-basis:gpu:sources={}:steps={}",
            payload.train_examples, request.steps
        )),
    );
    crate::import::save_manifest(request.shared_base_output, &base_manifest)?;
    let adapter_reports = save_gpu_direct_basis_adapters(
        &payload.adapters,
        &base_manifest,
        request.shared_base_output,
        request.adapter_output_dir,
        request.adapter_rank,
        request.adapter_alpha,
    )?;
    let adapter_bank = DirectBasisAdapterBankManifest {
        base_model: request.shared_base_output.display().to_string(),
        adapter_rank: request.adapter_rank,
        adapter_alpha: request.adapter_alpha,
        entries: adapter_reports.clone(),
    };
    write_pretty_json(request.adapter_bank_output, &adapter_bank)?;
    let report = CliHyper2dDirectBasisTrainingReport {
        preset: request.preset,
        target_images: request
            .target_images
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        target_image_dirs: request
            .target_image_dirs
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        target_image_recursive: request.target_image_recursive,
        image_extensions: request.image_extensions,
        catalog: request.catalog.map(|path| path.display().to_string()),
        catalog_group: request.catalog_group,
        catalog_targets: request.catalog_targets,
        omnisvg: request.omnisvg,
        source_limit: request.source_limit,
        holdout_targets: request.holdout_targets,
        holdout_stride: request.holdout_stride,
        holdout_offset: request.holdout_offset,
        output_dir: request.output_dir.display().to_string(),
        report_output: request.report_output.display().to_string(),
        shared_base_output: request.shared_base_output.display().to_string(),
        adapter_bank_output: request.adapter_bank_output.display().to_string(),
        adapter_output_dir: request.adapter_output_dir.display().to_string(),
        requested_training_device: request.requested_training_device,
        training_device: TrainingDeviceArg::Gpu,
        gpu_training: Some(CliHyper2dDirectBasisGpuTrainingReport {
            backend: payload.backend,
            python: Some(request.python.display().to_string()),
            device: payload.device,
            upstream_root: payload.upstream_root,
            payload_output: Some(request.gpu_payload_output.display().to_string()),
            gpu_name: payload.gpu_name,
            torch_version: payload.torch_version,
            cuda_version: payload.cuda_version,
            metrics,
        }),
        npa_config: config,
        hashgrid: request.hashgrid,
        target_loss_config: request.loss_config,
        adapter_rank: request.adapter_rank,
        adapter_alpha: request.adapter_alpha,
        train_examples: payload.train_examples,
        holdout_examples: payload.holdout_examples,
        steps: request.steps,
        report_interval: request.report_interval,
        example_batch_size: request.example_batch_size,
        rollout_particles: request.rollout_particles,
        rollout_steps: request.rollout_steps,
        update_prob: request.update_prob,
        seed: request.seed,
        seed_scale: request.seed_scale,
        seed_mode: request.seed_mode,
        per_parameter_grad_normalization: request.per_parameter_grad_normalization,
        base_sgd: request.base_sgd,
        adapter_sgd: request.adapter_sgd,
        adapter_l2_weight: request.adapter_l2,
        holdout_adapter_steps: request.holdout_adapter_steps,
        holdout_adapter_batch_size: request.holdout_adapter_batch_size,
        eval_examples: request.eval_examples,
        initial_train_loss: payload.initial_train_loss,
        final_train_loss: payload.final_train_loss,
        initial_holdout_loss: payload.initial_holdout_loss,
        final_holdout_loss: payload.final_holdout_loss,
        best_train_loss: payload.best_train_loss,
        best_train_step: payload.best_train_step,
        history: payload.history,
        holdout_history: payload.holdout_history,
        adapters: adapter_reports,
    };
    write_pretty_json(request.report_output, &report)?;
    println!(
        "wrote {} train={} holdout={} shared_base={} adapter_bank={} backend=gpu",
        request.report_output.display(),
        report.train_examples,
        report.holdout_examples,
        request.shared_base_output.display(),
        request.adapter_bank_output.display()
    );
    Ok(())
}

fn write_gpu_sources(
    path: &Path,
    sources: &[super::sources::Hyper2dScratchSource],
    splits: &[Hyper2dE2eSplit],
) -> Result<(), Box<dyn std::error::Error>> {
    if sources.len() != splits.len() {
        return Err(std::io::Error::other("source split count does not match sources").into());
    }
    let manifest = DirectBasisSourceManifest {
        sources: sources
            .iter()
            .zip(splits)
            .map(|(source, split)| {
                let path = std::fs::canonicalize(&source.condition_path)
                    .unwrap_or_else(|_| source.condition_path.clone());
                DirectBasisSourceEntry {
                    slug: source.slug.clone(),
                    split: split.label(),
                    title: source.title.clone(),
                    group: source.group.clone(),
                    path: path.display().to_string(),
                }
            })
            .collect(),
    };
    write_pretty_json(path, &manifest)
}

fn save_gpu_direct_basis_adapters(
    adapters: &[GpuDirectBasisAdapter],
    base_manifest: &BpkModelManifest,
    base_model_path: &Path,
    adapter_dir: &Path,
    adapter_rank: usize,
    adapter_alpha: f32,
) -> Result<Vec<CliHyper2dDirectBasisAdapterReport>, Box<dyn std::error::Error>> {
    let mut reports = Vec::with_capacity(adapters.len());
    for adapter_payload in adapters {
        let adapter = NpaLowRankAdapter::from_parameter_vector(
            &base_manifest.config,
            adapter_rank,
            adapter_alpha,
            adapter_payload.adapter.clone(),
        )?;
        let slug = sanitize_slug(&adapter_payload.slug);
        let adapter_path = adapter_dir.join(format!("{slug}.adapter.json"));
        let adapter_manifest = BpkAdapterManifest::from_adapter(
            base_manifest,
            Some(base_model_path.display().to_string()),
            adapter.clone(),
            Some(format!(
                "hyper2d-direct-basis-gpu:{}",
                adapter_payload.condition
            )),
        )?;
        crate::import::save_adapter_manifest(&adapter_path, &adapter_manifest)?;
        reports.push(CliHyper2dDirectBasisAdapterReport {
            slug: adapter_payload.slug.clone(),
            split: match adapter_payload.split.as_str() {
                "holdout" => "holdout",
                _ => "train",
            },
            title: adapter_payload.title.clone(),
            group: adapter_payload.group.clone(),
            condition: adapter_payload.condition.clone(),
            adapter_output: adapter_path.display().to_string(),
            target_source_width: adapter_payload.target_source_width,
            target_source_height: adapter_payload.target_source_height,
            target_points: adapter_payload.target_points,
            last_train_loss: adapter_payload.last_train_loss,
            adapter_parameter_count: adapter.parameter_count(),
        });
    }
    Ok(reports)
}

fn upstream_seed_mode(seed_mode: ParticleSeed) -> Result<&'static str, Box<dyn std::error::Error>> {
    match seed_mode {
        ParticleSeed::Gaussian => Ok("gaussian"),
        ParticleSeed::Uniform => Ok("uniform"),
        ParticleSeed::UniformCircle => Ok("uniform_circle"),
        other => Err(std::io::Error::other(format!(
            "direct-basis GPU training supports gaussian, uniform, and uniform-circle seeds, got {other:?}"
        ))
        .into()),
    }
}

struct DirectBasisArgCheck {
    adapter_rank: usize,
    adapter_alpha: f32,
    rollout_particles: usize,
    rollout_steps: usize,
    update_prob: f32,
    base_learning_rate: f32,
    adapter_learning_rate: f32,
    adapter_l2: f32,
}

fn validate_direct_basis_args(
    config: DirectBasisArgCheck,
) -> Result<(), Box<dyn std::error::Error>> {
    if config.adapter_rank == 0 || !config.adapter_alpha.is_finite() || config.adapter_alpha <= 0.0
    {
        return Err(std::io::Error::other(
            "adapter rank must be non-zero and adapter alpha must be finite and positive",
        )
        .into());
    }
    if config.rollout_particles == 0 || config.rollout_steps == 0 {
        return Err(std::io::Error::other(
            "rollout particles and rollout steps must be greater than zero",
        )
        .into());
    }
    if !(0.0..=1.0).contains(&config.update_prob) || !config.update_prob.is_finite() {
        return Err(
            std::io::Error::other("update probability must be finite and in [0, 1]").into(),
        );
    }
    if !config.base_learning_rate.is_finite()
        || config.base_learning_rate < 0.0
        || !config.adapter_learning_rate.is_finite()
        || config.adapter_learning_rate < 0.0
    {
        return Err(std::io::Error::other("learning rates must be finite and non-negative").into());
    }
    if !config.adapter_l2.is_finite() || config.adapter_l2 < 0.0 {
        return Err(
            std::io::Error::other("adapter L2 weight must be finite and non-negative").into(),
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct DirectBasisTargetConfig {
    threshold: f32,
    points: usize,
    image_size: Option<usize>,
}

fn load_direct_basis_examples(
    sources: &[super::sources::Hyper2dScratchSource],
    splits: &[Hyper2dE2eSplit],
    npa_config: &NpaConfig,
    adapter_rank: usize,
    adapter_alpha: f32,
    seed: u64,
    target_config: DirectBasisTargetConfig,
) -> Result<Vec<DirectBasisExample>, Box<dyn std::error::Error>> {
    if sources.len() != splits.len() {
        return Err(std::io::Error::other("source split count does not match sources").into());
    }
    let mut examples = Vec::with_capacity(sources.len());
    for (idx, (source, split)) in sources.iter().zip(splits).enumerate() {
        let target = super::super::target2d::load_target_image_2d_adaptive(
            &source.condition_path,
            target_config.threshold,
            target_config.points,
            target_config.image_size,
        )?;
        let adapter = NpaLowRankAdapter::seeded(
            npa_config,
            adapter_rank,
            adapter_alpha,
            seed.wrapping_add((idx as u64).wrapping_mul(0x517c_c1b7)),
        );
        examples.push(DirectBasisExample {
            source: source.clone(),
            split: *split,
            target,
            adapter,
            last_train_loss: None,
        });
    }
    Ok(examples)
}

fn train_direct_basis_phase(
    base: &mut NpaModel,
    examples: &mut [DirectBasisExample],
    hashgrid: &burn_automata_kernels::HashGridConfig,
    config: DirectBasisTrainConfig,
) -> Result<DirectBasisPhaseReport, Box<dyn std::error::Error>> {
    if examples.is_empty() || config.steps == 0 {
        return Ok(DirectBasisPhaseReport {
            history: Vec::new(),
            best_loss: None,
            best_step: 0,
        });
    }
    let mut rng = StdRng::seed_from_u64(config.seed);
    let batch_size = normalized_example_batch_size(config.example_batch_size, examples.len());
    let report_interval = config.report_interval.max(1);
    let mut best_loss = None::<f32>;
    let mut best_step = 0usize;
    let mut history = Vec::new();
    for step in 1..=config.steps {
        let indices = sample_example_indices(examples.len(), batch_size, &mut rng);
        let stats = direct_basis_train_step(base, examples, hashgrid, &indices, config, step)?;
        if step == config.steps || step.is_multiple_of(report_interval) {
            let eval_loss = evaluate_direct_basis_examples(
                base,
                examples,
                hashgrid,
                config,
                config.eval_examples,
                config.eval_seed,
            )?;
            if let Some(summary) = eval_loss
                && best_loss.is_none_or(|loss| summary.mean_total_loss < loss)
            {
                best_loss = Some(summary.mean_total_loss);
                best_step = step;
            }
            history.push(CliHyper2dDirectBasisHistoryEntry {
                step,
                loss: stats.loss,
                eval_loss,
                base_grad_norm: stats.base_grad_norm,
                base_grad_scale: stats.base_grad_scale,
                mean_adapter_grad_norm: stats.mean_adapter_grad_norm,
                max_adapter_grad_norm: stats.max_adapter_grad_norm,
                examples_seen: stats.examples_seen,
                particle_steps_per_sec: stats.particle_steps_per_sec,
                elapsed_ms: stats.elapsed_ms,
            });
        }
    }
    Ok(DirectBasisPhaseReport {
        history,
        best_loss,
        best_step,
    })
}

fn direct_basis_train_step(
    base: &mut NpaModel,
    examples: &mut [DirectBasisExample],
    hashgrid: &burn_automata_kernels::HashGridConfig,
    indices: &[usize],
    config: DirectBasisTrainConfig,
    step: usize,
) -> Result<DirectBasisStepStats, Box<dyn std::error::Error>> {
    if indices.is_empty() {
        return Err(std::io::Error::other("direct basis step requires examples").into());
    }
    let start = Instant::now();
    let mut base_grads = zero_model_gradients(base);
    let example_scale = 1.0 / indices.len() as f32;
    let mut loss_sum = 0.0_f32;
    let mut adapter_grad_sum = 0.0_f32;
    let mut adapter_grad_max = 0.0_f32;
    let mut particle_steps = 0.0_f64;
    for &idx in indices {
        let example = examples
            .get_mut(idx)
            .ok_or_else(|| std::io::Error::other("direct basis index is out of range"))?;
        let adapted = example.adapter.apply_to_model(base)?;
        let particle_count = example.source.particles.unwrap_or(config.rollout_particles);
        let update_prob = example.source.update_prob.unwrap_or(config.update_prob);
        let seed_scale = example.source.seed_scale.unwrap_or(config.seed_scale);
        let (loss, full_grads) = target_2d_rollout_loss_with_gradients(
            &adapted,
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
        loss_sum += loss.total_loss;
        example.last_train_loss = Some(loss.total_loss);
        let mut adapter_grads =
            project_low_rank_adapter_gradients(base, &example.adapter, &full_grads)?;
        add_adapter_l2_gradients(
            &example.adapter,
            &mut adapter_grads,
            config.adapter_l2_weight,
        );
        let adapter_step =
            apply_sgd_adapter_gradients(&mut example.adapter, &adapter_grads, config.adapter_sgd)?;
        adapter_grad_sum += adapter_step.grad_norm;
        adapter_grad_max = adapter_grad_max.max(adapter_step.grad_norm);
        if config.update_base {
            add_scaled_model_gradients(&mut base_grads, &full_grads, example_scale);
        }
        particle_steps += particle_count as f64 * config.rollout_steps as f64;
    }
    let (base_grad_norm, base_grad_scale) = if config.update_base {
        let step_report = apply_sgd_gradients(base, &base_grads, config.base_sgd)?;
        (step_report.grad_norm, step_report.grad_scale)
    } else {
        (0.0, 1.0)
    };
    let elapsed = start.elapsed();
    Ok(DirectBasisStepStats {
        loss: loss_sum / indices.len() as f32,
        base_grad_norm,
        base_grad_scale,
        mean_adapter_grad_norm: adapter_grad_sum / indices.len() as f32,
        max_adapter_grad_norm: adapter_grad_max,
        examples_seen: indices.len(),
        particle_steps_per_sec: particle_steps / elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
    })
}

fn evaluate_direct_basis_examples(
    base: &NpaModel,
    examples: &[DirectBasisExample],
    hashgrid: &burn_automata_kernels::HashGridConfig,
    config: DirectBasisTrainConfig,
    requested_examples: usize,
    seed: u64,
) -> Result<Option<CliHyper2dDirectBasisLossSummary>, Box<dyn std::error::Error>> {
    if examples.is_empty() {
        return Ok(None);
    }
    let indices = eval_indices(examples.len(), requested_examples, seed);
    let mut total = CliHyper2dDirectBasisLossSummary {
        examples: indices.len(),
        mean_total_loss: 0.0,
        max_total_loss: 0.0,
        mean_splat_loss: 0.0,
        mean_color_loss: 0.0,
        mean_density_loss: 0.0,
    };
    for &idx in &indices {
        let example = &examples[idx];
        let loss = evaluate_direct_basis_example(
            base,
            example,
            hashgrid,
            EvalConfig {
                particle_count: example.source.particles.unwrap_or(config.rollout_particles),
                rollout_steps: config.rollout_steps,
                update_prob: example.source.update_prob.unwrap_or(config.update_prob),
                seed: seed.wrapping_add(idx as u64),
                seed_scale: example.source.seed_scale.unwrap_or(config.seed_scale),
                seed_mode: config.seed_mode,
            },
            config.loss_config,
        )?;
        total.mean_total_loss += loss.total_loss;
        total.max_total_loss = total.max_total_loss.max(loss.total_loss);
        total.mean_splat_loss += loss.splat_loss;
        total.mean_color_loss += loss.color_loss;
        total.mean_density_loss += loss.density_loss;
    }
    let scale = 1.0 / indices.len() as f32;
    total.mean_total_loss *= scale;
    total.mean_splat_loss *= scale;
    total.mean_color_loss *= scale;
    total.mean_density_loss *= scale;
    Ok(Some(total))
}

fn eval_indices(examples_len: usize, requested_examples: usize, seed: u64) -> Vec<usize> {
    let mut indices = (0..examples_len).collect::<Vec<_>>();
    if requested_examples == 0 || requested_examples >= examples_len {
        return indices;
    }
    let mut rng = StdRng::seed_from_u64(seed);
    indices.shuffle(&mut rng);
    indices.truncate(requested_examples);
    indices.sort_unstable();
    indices
}

fn evaluate_direct_basis_example(
    base: &NpaModel,
    example: &DirectBasisExample,
    hashgrid: &burn_automata_kernels::HashGridConfig,
    config: EvalConfig,
    loss_config: Target2dLossConfig,
) -> Result<Target2dLossReport, Box<dyn std::error::Error>> {
    let model = example.adapter.apply_to_model(base)?;
    let trace = run_rollout(
        &model,
        hashgrid,
        &RolloutConfig {
            batch_size: 1,
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
        &example.target,
        loss_config,
        trace.mean_dx.iter().copied().sum(),
        trace.steps,
    )?
    .report)
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

fn save_direct_basis_adapters(
    base_manifest: &BpkModelManifest,
    base_model_path: &Path,
    adapter_dir: &Path,
    examples: &[&DirectBasisExample],
) -> Result<Vec<CliHyper2dDirectBasisAdapterReport>, Box<dyn std::error::Error>> {
    let mut reports = Vec::with_capacity(examples.len());
    for example in examples {
        let slug = sanitize_slug(&example.source.slug);
        let adapter_path = adapter_dir.join(format!("{slug}.adapter.json"));
        let adapter_manifest = BpkAdapterManifest::from_adapter(
            base_manifest,
            Some(base_model_path.display().to_string()),
            example.adapter.clone(),
            Some(format!(
                "hyper2d-direct-basis:{}",
                example.source.condition_path.display()
            )),
        )?;
        crate::import::save_adapter_manifest(&adapter_path, &adapter_manifest)?;
        reports.push(CliHyper2dDirectBasisAdapterReport {
            slug: example.source.slug.clone(),
            split: example.split.label(),
            title: example.source.title.clone(),
            group: example.source.group.clone(),
            condition: example.source.condition_path.display().to_string(),
            adapter_output: adapter_path.display().to_string(),
            target_source_width: example.target.source_width,
            target_source_height: example.target.source_height,
            target_points: example.target.point_count(),
            last_train_loss: example.last_train_loss,
            adapter_parameter_count: example.adapter.parameter_count(),
        });
    }
    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_indices_are_sorted_and_bounded() {
        assert_eq!(eval_indices(3, 0, 1), vec![0, 1, 2]);
        let indices = eval_indices(10, 4, 9);

        assert_eq!(indices.len(), 4);
        assert!(indices.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(indices.iter().all(|idx| *idx < 10));
    }

    #[test]
    fn direct_basis_arg_validation_rejects_empty_rollouts() {
        let err = validate_direct_basis_args(DirectBasisArgCheck {
            adapter_rank: 1,
            adapter_alpha: 1.0,
            rollout_particles: 0,
            rollout_steps: 1,
            update_prob: 0.5,
            base_learning_rate: 1.0e-4,
            adapter_learning_rate: 1.0e-3,
            adapter_l2: 0.0,
        })
        .unwrap_err()
        .to_string();

        assert!(err.contains("rollout particles"));
    }
}
