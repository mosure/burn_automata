use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::cli::commands::hyper_support::{
    attach_condition_features, load_condition_image_2d, load_hyper_2d, write_pretty_json,
};
use crate::cli::prelude::*;

use super::super::sources::{Hyper2dScratchSource, sanitize_slug};
use super::super::{
    DinoConditionFeatureCacheConfig, build_condition_feature_cache,
    default_dino_cache_write_interval_batches, default_dino_feature_batch_size,
};
use super::{
    DirectBasisAdapterBankLoadEntry, config_value_enum, load_direct_basis_adapter_bank,
    resolve_direct_basis_artifact_path,
};

#[derive(Deserialize)]
struct PsnrGateOracleReportLoad {
    oracle_validation: Option<PsnrGateOracleValidationLoad>,
}

#[derive(Deserialize)]
struct PsnrGateOracleValidationLoad {
    entries: Vec<PsnrGateOracleEntryLoad>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PsnrGateExperimentConfig {
    preset: Option<String>,
    input: PsnrGateInputExperimentConfig,
    condition: PsnrGateConditionExperimentConfig,
    output: PsnrGateOutputExperimentConfig,
    eval: PsnrGateEvalExperimentConfig,
    gate: PsnrGateThresholdExperimentConfig,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PsnrGateInputExperimentConfig {
    base_model: Option<PathBuf>,
    adapter_bank: Option<PathBuf>,
    oracle_report: Option<PathBuf>,
    hyper: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PsnrGateConditionExperimentConfig {
    dino_model: Option<PathBuf>,
    dino_image_size: Option<usize>,
    dino_batch_size: Option<usize>,
    dino_cache_write_interval_batches: Option<usize>,
    feature_cache: Option<PathBuf>,
    token_grid_width: Option<usize>,
    token_grid_height: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PsnrGateOutputExperimentConfig {
    output: Option<PathBuf>,
    generated_dir: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PsnrGateEvalExperimentConfig {
    limit: Option<usize>,
    particles: Option<usize>,
    steps: Option<Vec<usize>>,
    update_prob: Option<f32>,
    seed: Option<u64>,
    seed_scale: Option<f32>,
    seed_mode: Option<String>,
    image_size: Option<usize>,
    render_sigma_px: Option<f32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PsnrGateThresholdExperimentConfig {
    min_render_rgb_psnr_db: Option<f32>,
    fail_on_threshold: Option<bool>,
}

#[derive(Clone, Deserialize)]
struct PsnrGateOracleEntryLoad {
    slug: String,
    split: String,
    condition: String,
    oracle_model_output: Option<String>,
}

#[derive(Serialize)]
struct Hyper2dPsnrGateReport {
    preset: AutomataPreset,
    base_model: String,
    adapter_bank: String,
    oracle_report: String,
    hyper: Option<String>,
    output: String,
    generated_dir: String,
    adapter_bank_base_model: String,
    adapter_rank: usize,
    adapter_alpha: f32,
    examples: usize,
    particle_count: usize,
    rollout_steps: Vec<usize>,
    update_prob: f32,
    seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
    image_size: usize,
    render_sigma_px: f32,
    min_render_rgb_psnr_db: f32,
    passed: bool,
    summaries: Vec<Hyper2dPsnrGateSummary>,
    entries: Vec<Hyper2dPsnrGateEntry>,
}

#[derive(Serialize)]
struct Hyper2dPsnrGateSummary {
    kind: &'static str,
    rollout_steps: usize,
    examples: usize,
    mean_render_rgb_psnr_db: f32,
    median_render_rgb_psnr_db: f32,
    min_render_rgb_psnr_db: f32,
    max_render_rgb_psnr_db: f32,
    below_threshold: usize,
    passed: bool,
}

#[derive(Serialize)]
struct Hyper2dPsnrGateEntry {
    slug: String,
    split: String,
    oracle_split: String,
    condition: String,
    oracle_condition: String,
    kind: &'static str,
    model: String,
    target_model: String,
    rollout_steps: usize,
    render_rgb_psnr_db: f32,
    passed: bool,
    metrics: CliHyper2dDynamicsMetricsReport,
}

pub(crate) fn run_validate_hyper_2d_psnr_gate(
    command: Command,
) -> Result<(), Box<dyn std::error::Error>> {
    let Command::ValidateHyper2dPsnrGate {
        config,
        preset,
        base_model,
        adapter_bank,
        oracle_report,
        hyper,
        output,
        generated_dir,
        limit,
        particles,
        steps,
        update_prob,
        seed,
        seed_scale,
        seed_mode,
        image_size,
        render_sigma_px,
        min_render_rgb_psnr_db,
        fail_on_threshold,
    } = command
    else {
        unreachable!("run_validate_hyper_2d_psnr_gate called with the wrong command variant");
    };

    let experiment_config = load_psnr_gate_experiment_config(config.as_deref())?;
    let PsnrGateExperimentConfig {
        preset: config_preset,
        input: config_input,
        condition: config_condition,
        output: config_output,
        eval: config_eval,
        gate: config_gate,
    } = experiment_config;
    let PsnrGateInputExperimentConfig {
        base_model: config_base_model,
        adapter_bank: config_adapter_bank,
        oracle_report: config_oracle_report,
        hyper: config_hyper,
    } = config_input;
    let PsnrGateConditionExperimentConfig {
        dino_model: config_dino_model,
        dino_image_size: config_dino_image_size,
        dino_batch_size: config_dino_batch_size,
        dino_cache_write_interval_batches: config_dino_cache_write_interval_batches,
        feature_cache: config_condition_feature_cache,
        token_grid_width: config_condition_token_grid_width,
        token_grid_height: config_condition_token_grid_height,
    } = config_condition;
    let PsnrGateOutputExperimentConfig {
        output: config_output_path,
        generated_dir: config_generated_dir,
    } = config_output;
    let PsnrGateEvalExperimentConfig {
        limit: config_limit,
        particles: config_particles,
        steps: config_steps,
        update_prob: config_update_prob,
        seed: config_seed,
        seed_scale: config_seed_scale,
        seed_mode: config_seed_mode,
        image_size: config_image_size,
        render_sigma_px: config_render_sigma_px,
    } = config_eval;
    let PsnrGateThresholdExperimentConfig {
        min_render_rgb_psnr_db: config_min_render_rgb_psnr_db,
        fail_on_threshold: config_fail_on_threshold,
    } = config_gate;

    let preset = config_value_enum("preset", config_preset, preset)?;
    let base_model = config_base_model.or(base_model).ok_or_else(|| {
        std::io::Error::other(
            "validate-hyper2d-psnr-gate requires --base-model or input.base_model",
        )
    })?;
    let adapter_bank = config_adapter_bank.or(adapter_bank).ok_or_else(|| {
        std::io::Error::other(
            "validate-hyper2d-psnr-gate requires --adapter-bank or input.adapter_bank",
        )
    })?;
    let oracle_report = config_oracle_report.or(oracle_report).ok_or_else(|| {
        std::io::Error::other(
            "validate-hyper2d-psnr-gate requires --oracle-report or input.oracle_report",
        )
    })?;
    let hyper = config_hyper.or(hyper);
    let dino_image_size = config_dino_image_size.unwrap_or(518);
    let dino_batch_size = config_dino_batch_size.unwrap_or_else(default_dino_feature_batch_size);
    let dino_cache_write_interval_batches = config_dino_cache_write_interval_batches
        .unwrap_or(default_dino_cache_write_interval_batches());
    let output = config_output_path.unwrap_or(output);
    let generated_dir = config_generated_dir.unwrap_or(generated_dir);
    let limit = config_limit.unwrap_or(limit);
    let particles = config_particles.unwrap_or(particles);
    let steps = config_steps.unwrap_or(steps);
    let update_prob = config_update_prob.unwrap_or(update_prob);
    let seed = config_seed.unwrap_or(seed);
    let seed_scale = config_seed_scale.or(seed_scale);
    let seed_mode = config_value_enum("eval.seed_mode", config_seed_mode, seed_mode)?;
    let image_size = config_image_size.unwrap_or(image_size);
    let render_sigma_px = config_render_sigma_px.unwrap_or(render_sigma_px);
    let min_render_rgb_psnr_db = config_min_render_rgb_psnr_db.unwrap_or(min_render_rgb_psnr_db);
    let fail_on_threshold = config_fail_on_threshold.unwrap_or(fail_on_threshold);

    if particles == 0 {
        return Err(std::io::Error::other("--particles must be greater than zero").into());
    }
    if steps.is_empty() || steps.contains(&0) {
        return Err(std::io::Error::other("--step values must be greater than zero").into());
    }
    if !(0.0..=1.0).contains(&update_prob) || !update_prob.is_finite() {
        return Err(std::io::Error::other("--update-prob must be finite and in [0, 1]").into());
    }
    if image_size == 0 {
        return Err(std::io::Error::other("--image-size must be greater than zero").into());
    }
    if !render_sigma_px.is_finite() || render_sigma_px <= 0.0 {
        return Err(std::io::Error::other(
            "--render-sigma-px must be finite and greater than zero",
        )
        .into());
    }
    if !min_render_rgb_psnr_db.is_finite() {
        return Err(std::io::Error::other("--min-render-rgb-psnr-db must be finite").into());
    }

    let preset: AutomataPreset = preset.into();
    let seed_mode: ParticleSeed = seed_mode.into();
    let seed_scale = seed_scale.unwrap_or_else(|| NpaConfig::seed_scale_for_preset(preset));
    let base_manifest = crate::import::load_manifest(&base_model)?;
    if base_manifest.config.spatial_dims != 2 {
        return Err(
            std::io::Error::other("validate-hyper2d-psnr-gate requires a 2D base model").into(),
        );
    }
    let base = base_manifest.clone().into_model();
    let hyper_model = hyper.as_ref().map(|path| load_hyper_2d(path)).transpose()?;
    if let Some(hyper_model) = &hyper_model
        && hyper_model.npa_config != base.config
    {
        return Err(std::io::Error::other(
            "hyper checkpoint NPA config must match base model config",
        )
        .into());
    }
    let bank = load_direct_basis_adapter_bank(&adapter_bank)?;
    if bank.entries.is_empty() {
        return Err(std::io::Error::other("adapter bank has no entries").into());
    }
    if let Some(hyper_model) = &hyper_model
        && (bank.adapter_rank != hyper_model.config.adapter_rank
            || (bank.adapter_alpha - hyper_model.config.adapter_alpha).abs() > f32::EPSILON)
    {
        return Err(std::io::Error::other(format!(
            "adapter bank rank/alpha ({}/{}) does not match hyper checkpoint ({}/{})",
            bank.adapter_rank,
            bank.adapter_alpha,
            hyper_model.config.adapter_rank,
            hyper_model.config.adapter_alpha
        ))
        .into());
    }

    let oracle_anchor = oracle_report.parent().unwrap_or_else(|| Path::new(""));
    let oracle_entries = load_oracle_entries(&oracle_report)?;
    let oracle_by_key = oracle_entries
        .iter()
        .map(|entry| (oracle_key(&entry.split, &entry.slug), entry))
        .collect::<HashMap<_, _>>();
    let oracle_by_slug = oracle_entries
        .iter()
        .map(|entry| (entry.slug.clone(), entry))
        .collect::<HashMap<_, _>>();
    let mut oracle_slug_counts = HashMap::<String, usize>::new();
    for entry in &oracle_entries {
        *oracle_slug_counts.entry(entry.slug.clone()).or_default() += 1;
    }
    let bank_anchor = adapter_bank.parent().unwrap_or_else(|| Path::new(""));
    let oracle_backed_entries = bank
        .entries
        .iter()
        .filter(|entry| {
            oracle_entry_for_bank_entry(entry, &oracle_by_key, &oracle_by_slug, &oracle_slug_counts)
                .is_some()
        })
        .collect::<Vec<_>>();
    let selected_entries = oracle_backed_entries
        .into_iter()
        .take(if limit == 0 { usize::MAX } else { limit })
        .collect::<Vec<_>>();
    if selected_entries.is_empty() {
        return Err(std::io::Error::other(
            "no adapter-bank entries with oracle-model coverage selected",
        )
        .into());
    }
    let condition_features = if let Some(hyper_model) = &hyper_model
        && requires_dino_condition_features(hyper_model.config.condition_encoder)
    {
        let selected_sources = selected_entries
            .iter()
            .map(|entry| Hyper2dScratchSource {
                slug: entry.slug.clone(),
                title: entry.title.clone(),
                group: entry.group.clone(),
                condition_path: resolve_direct_basis_artifact_path(bank_anchor, &entry.condition),
                particles: None,
                seed_scale: None,
                update_prob: None,
            })
            .collect::<Vec<_>>();
        Some(build_condition_feature_cache(
            &selected_sources,
            hyper_model.config.condition_encoder,
            DinoConditionFeatureCacheConfig {
                model: config_dino_model.as_ref(),
                image_size: dino_image_size,
                batch_size: dino_batch_size,
                cache_write_interval_batches: dino_cache_write_interval_batches,
                token_grid_width: config_condition_token_grid_width
                    .unwrap_or(hyper_model.config.condition_token_grid_width),
                token_grid_height: config_condition_token_grid_height
                    .unwrap_or(hyper_model.config.condition_token_grid_height),
                cache_path: config_condition_feature_cache.as_deref(),
            },
        )?)
    } else {
        None
    };

    let mut report_entries = Vec::with_capacity(selected_entries.len() * steps.len() * 2);
    for bank_entry in selected_entries {
        let oracle_entry = oracle_entry_for_bank_entry(
            bank_entry,
            &oracle_by_key,
            &oracle_by_slug,
            &oracle_slug_counts,
        );
        let Some(oracle_entry) = oracle_entry else {
            return Err(std::io::Error::other(format!(
                "oracle report has no entry for {}:{}",
                bank_entry.split, bank_entry.slug
            ))
            .into());
        };
        let Some(oracle_model_output) = oracle_entry.oracle_model_output.as_deref() else {
            return Err(std::io::Error::other(format!(
                "oracle report entry {}:{} has no oracle_model_output",
                oracle_entry.split, oracle_entry.slug
            ))
            .into());
        };
        let condition_path = resolve_direct_basis_artifact_path(bank_anchor, &bank_entry.condition);
        let oracle_model_path =
            resolve_direct_basis_artifact_path(oracle_anchor, oracle_model_output);
        let adapter_path =
            resolve_direct_basis_artifact_path(bank_anchor, &bank_entry.adapter_output);
        let slug = sanitize_slug(&bank_entry.slug);
        let direct_model_path = generated_dir
            .join("direct")
            .join(&bank_entry.split)
            .join(format!("{slug}.bpk"));
        let hyper_model_path = generated_dir
            .join("hyper")
            .join(&bank_entry.split)
            .join(format!("{slug}.bpk"));

        let direct_adapter_manifest = crate::import::load_adapter_manifest(&adapter_path)?;
        let direct_manifest = direct_adapter_manifest.materialize(&base_manifest)?;
        crate::import::save_manifest(&direct_model_path, &direct_manifest)?;
        let direct_model = direct_manifest.into_model();

        let hyper_materialized = if let Some(hyper_model) = &hyper_model {
            let condition = attach_condition_features(
                load_condition_image_2d(&condition_path)?,
                &condition_path,
                condition_features.as_ref(),
            )?;
            let hyper_adapter = hyper_model.predict_adapter(&condition)?;
            let hyper_materialized = hyper_adapter.apply_to_model(&base)?;
            let hyper_manifest = BpkModelManifest::from_model(
                &hyper_materialized,
                base_manifest.hashgrid.clone(),
                Some(format!("hyper2d-psnr-gate:{}", bank_entry.slug)),
            );
            crate::import::save_manifest(&hyper_model_path, &hyper_manifest)?;
            Some(hyper_materialized)
        } else {
            None
        };

        let target_manifest = crate::import::load_manifest(&oracle_model_path)?;
        if target_manifest.config != base_manifest.config {
            return Err(std::io::Error::other(format!(
                "oracle model config differs for {}:{}",
                bank_entry.split, bank_entry.slug
            ))
            .into());
        }
        if target_manifest.hashgrid != base_manifest.hashgrid {
            return Err(std::io::Error::other(format!(
                "oracle model hashgrid differs for {}:{}",
                bank_entry.split, bank_entry.slug
            ))
            .into());
        }
        let target_model = target_manifest.into_model();
        for &rollout_steps in &steps {
            let direct_metrics = evaluate_gate_model(
                &direct_model,
                &target_model,
                &base_manifest.hashgrid,
                particles,
                rollout_steps,
                update_prob,
                seed,
                seed_scale,
                seed_mode,
                image_size,
                render_sigma_px,
            )?;
            push_gate_entry(
                &mut report_entries,
                bank_entry,
                oracle_entry,
                "direct",
                &direct_model_path,
                &oracle_model_path,
                rollout_steps,
                min_render_rgb_psnr_db,
                direct_metrics,
            );

            if let Some(hyper_materialized) = &hyper_materialized {
                let hyper_metrics = evaluate_gate_model(
                    hyper_materialized,
                    &target_model,
                    &base_manifest.hashgrid,
                    particles,
                    rollout_steps,
                    update_prob,
                    seed,
                    seed_scale,
                    seed_mode,
                    image_size,
                    render_sigma_px,
                )?;
                push_gate_entry(
                    &mut report_entries,
                    bank_entry,
                    oracle_entry,
                    "hyper",
                    &hyper_model_path,
                    &oracle_model_path,
                    rollout_steps,
                    min_render_rgb_psnr_db,
                    hyper_metrics,
                );
            }
        }
    }

    let summaries = summarize_gate_entries(&report_entries, min_render_rgb_psnr_db);
    let passed = summaries.iter().all(|summary| summary.passed);
    let report = Hyper2dPsnrGateReport {
        preset,
        base_model: base_model.display().to_string(),
        adapter_bank: adapter_bank.display().to_string(),
        oracle_report: oracle_report.display().to_string(),
        hyper: hyper.as_ref().map(|path| path.display().to_string()),
        output: output.display().to_string(),
        generated_dir: generated_dir.display().to_string(),
        adapter_bank_base_model: bank.base_model,
        adapter_rank: bank.adapter_rank,
        adapter_alpha: bank.adapter_alpha,
        examples: selected_entries_len(&report_entries, steps.len()),
        particle_count: particles,
        rollout_steps: steps,
        update_prob,
        seed,
        seed_scale,
        seed_mode,
        image_size,
        render_sigma_px,
        min_render_rgb_psnr_db,
        passed,
        summaries,
        entries: report_entries,
    };
    write_pretty_json(&output, &report)?;
    println!(
        "wrote {} passed={} examples={} threshold={:.3}dB",
        output.display(),
        report.passed,
        report.examples,
        min_render_rgb_psnr_db
    );

    if fail_on_threshold && !report.passed {
        return Err(std::io::Error::other(format!(
            "HyperNPA PSNR gate failed below {:.3} dB; see {}",
            min_render_rgb_psnr_db,
            output.display()
        ))
        .into());
    }
    Ok(())
}

fn requires_dino_condition_features(encoder: ConditionEncoder2d) -> bool {
    matches!(
        encoder,
        ConditionEncoder2d::DinoVitsClsPatchMean
            | ConditionEncoder2d::DinoVitsPatchStats
            | ConditionEncoder2d::DinoVitsTokenGrid
    )
}

fn load_psnr_gate_experiment_config(
    path: Option<&Path>,
) -> Result<PsnrGateExperimentConfig, Box<dyn std::error::Error>> {
    let Some(path) = path else {
        return Ok(PsnrGateExperimentConfig::default());
    };
    let text = std::fs::read_to_string(path)?;
    toml::from_str(&text).map_err(|err| {
        std::io::Error::other(format!(
            "failed to parse Hyper2D PSNR gate config {}: {err}",
            path.display()
        ))
        .into()
    })
}

fn load_oracle_entries(
    oracle_report: &Path,
) -> Result<Vec<PsnrGateOracleEntryLoad>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(oracle_report)?;
    let report: PsnrGateOracleReportLoad = serde_json::from_str(&text)?;
    let Some(oracle_validation) = report.oracle_validation else {
        return Err(std::io::Error::other("oracle report has no oracle_validation section").into());
    };
    if oracle_validation.entries.is_empty() {
        return Err(std::io::Error::other("oracle report has no oracle entries").into());
    }
    Ok(oracle_validation.entries)
}

fn oracle_key(split: &str, slug: &str) -> String {
    format!("{split}\0{slug}")
}

fn oracle_entry_for_bank_entry<'a>(
    bank_entry: &DirectBasisAdapterBankLoadEntry,
    oracle_by_key: &HashMap<String, &'a PsnrGateOracleEntryLoad>,
    oracle_by_slug: &HashMap<String, &'a PsnrGateOracleEntryLoad>,
    oracle_slug_counts: &HashMap<String, usize>,
) -> Option<&'a PsnrGateOracleEntryLoad> {
    oracle_by_key
        .get(&oracle_key(&bank_entry.split, &bank_entry.slug))
        .copied()
        .or_else(|| {
            if oracle_slug_counts
                .get(&bank_entry.slug)
                .is_some_and(|count| *count == 1)
            {
                oracle_by_slug.get(&bank_entry.slug).copied()
            } else {
                None
            }
        })
}

#[allow(clippy::too_many_arguments)]
fn evaluate_gate_model(
    generated_model: &NpaModel,
    target_model: &NpaModel,
    hashgrid: &burn_automata_kernels::HashGridConfig,
    particles: usize,
    steps: usize,
    update_prob: f32,
    seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
    image_size: usize,
    render_sigma_px: f32,
) -> Result<CliHyper2dDynamicsMetricsReport, Box<dyn std::error::Error>> {
    super::super::super::dynamics2d::evaluate_2d_dynamics_models(
        generated_model,
        target_model,
        hashgrid,
        super::super::super::dynamics2d::Dynamics2dEvalConfig {
            particles,
            steps,
            update_prob,
            seed,
            seed_scale,
            seed_mode,
            image_size,
            render_sigma_px,
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn push_gate_entry(
    entries: &mut Vec<Hyper2dPsnrGateEntry>,
    bank_entry: &super::DirectBasisAdapterBankLoadEntry,
    oracle_entry: &PsnrGateOracleEntryLoad,
    kind: &'static str,
    model_path: &Path,
    target_model_path: &Path,
    rollout_steps: usize,
    min_render_rgb_psnr_db: f32,
    metrics: CliHyper2dDynamicsMetricsReport,
) {
    let render_rgb_psnr_db = metrics.render_rgb_psnr_db;
    entries.push(Hyper2dPsnrGateEntry {
        slug: bank_entry.slug.clone(),
        split: bank_entry.split.clone(),
        oracle_split: oracle_entry.split.clone(),
        condition: bank_entry.condition.clone(),
        oracle_condition: oracle_entry.condition.clone(),
        kind,
        model: model_path.display().to_string(),
        target_model: target_model_path.display().to_string(),
        rollout_steps,
        render_rgb_psnr_db,
        passed: render_rgb_psnr_db >= min_render_rgb_psnr_db,
        metrics,
    });
}

fn summarize_gate_entries(
    entries: &[Hyper2dPsnrGateEntry],
    min_render_rgb_psnr_db: f32,
) -> Vec<Hyper2dPsnrGateSummary> {
    let mut by_kind_step = BTreeMap::<(&'static str, usize), Vec<f32>>::new();
    for entry in entries {
        by_kind_step
            .entry((entry.kind, entry.rollout_steps))
            .or_default()
            .push(entry.render_rgb_psnr_db);
    }
    by_kind_step
        .into_iter()
        .map(|((kind, rollout_steps), mut values)| {
            values.sort_by(|left, right| left.total_cmp(right));
            let examples = values.len();
            let sum = values.iter().copied().sum::<f32>();
            let median = if examples % 2 == 0 {
                (values[examples / 2 - 1] + values[examples / 2]) * 0.5
            } else {
                values[examples / 2]
            };
            let below_threshold = values
                .iter()
                .filter(|value| **value < min_render_rgb_psnr_db)
                .count();
            Hyper2dPsnrGateSummary {
                kind,
                rollout_steps,
                examples,
                mean_render_rgb_psnr_db: sum / examples as f32,
                median_render_rgb_psnr_db: median,
                min_render_rgb_psnr_db: values[0],
                max_render_rgb_psnr_db: values[examples - 1],
                below_threshold,
                passed: below_threshold == 0,
            }
        })
        .collect()
}

fn selected_entries_len(entries: &[Hyper2dPsnrGateEntry], step_count: usize) -> usize {
    let kinds = entries
        .iter()
        .map(|entry| entry.kind)
        .collect::<BTreeSet<_>>()
        .len()
        .max(1);
    let divisor = step_count.saturating_mul(kinds).max(1);
    entries.len() / divisor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_psnr_gate_config_parses() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let path = repo_root
            .join("configs/hyper2d_adapter_bank")
            .join("psnr_gate_1k_dino_token_grid_flow_h512_rms_noise.toml");

        let config = load_psnr_gate_experiment_config(Some(&path)).unwrap();

        assert_eq!(config.preset.as_deref(), Some("growing-2d"));
        assert!(config.input.base_model.is_some());
        assert!(config.input.adapter_bank.is_some());
        assert!(config.input.oracle_report.is_some());
        assert!(config.input.hyper.is_some());
        assert!(config.condition.dino_model.is_some());
        assert_eq!(config.condition.dino_image_size, Some(518));
        assert_eq!(config.condition.dino_batch_size, Some(4));
        assert_eq!(config.condition.token_grid_width, Some(8));
        assert_eq!(config.condition.token_grid_height, Some(8));
        assert_eq!(config.eval.particles, Some(2048));
        assert_eq!(config.eval.steps.as_deref(), Some(&[32, 64][..]));
        assert_eq!(config.gate.min_render_rgb_psnr_db, Some(26.0));
        assert_eq!(config.gate.fail_on_threshold, Some(true));

        for file_name in [
            "psnr_gate_1k_dino_token_grid_flow_h512_rms_noise_oracle8x8.toml",
            "psnr_gate_1k_dino_token_grid_flow_h512_sampled_refine_oracle8x8.toml",
            "psnr_gate_10k_dino_canonical_h1024_valselect_oracle8x8.toml",
        ] {
            let path = repo_root
                .join("configs/hyper2d_adapter_bank")
                .join(file_name);
            let config = load_psnr_gate_experiment_config(Some(&path)).unwrap();
            assert_eq!(config.preset.as_deref(), Some("growing-2d"));
            assert!(config.input.base_model.is_some());
            assert!(config.input.adapter_bank.is_some());
            assert!(config.input.oracle_report.is_some());
            assert!(config.input.hyper.is_some());
            assert!(config.condition.feature_cache.is_some());
            if file_name.contains("token_grid") {
                assert_eq!(config.condition.token_grid_width, Some(8));
                assert_eq!(config.condition.token_grid_height, Some(8));
            }
            assert_eq!(config.eval.limit, Some(16));
            assert_eq!(config.eval.particles, Some(2048));
            assert_eq!(config.eval.steps.as_deref(), Some(&[32][..]));
            assert_eq!(config.gate.min_render_rgb_psnr_db, Some(26.0));
            assert_eq!(config.gate.fail_on_threshold, Some(false));
        }

        let direct_only_path = repo_root
            .join("configs/hyper2d_adapter_bank")
            .join("psnr_gate_exact_oracle_10k8x8_2048_rank132_direct.toml");
        let direct_only = load_psnr_gate_experiment_config(Some(&direct_only_path)).unwrap();
        assert!(direct_only.input.base_model.is_some());
        assert!(direct_only.input.adapter_bank.is_some());
        assert!(direct_only.input.oracle_report.is_some());
        assert!(direct_only.input.hyper.is_none());
        assert_eq!(direct_only.eval.particles, Some(2048));
        assert_eq!(direct_only.eval.steps.as_deref(), Some(&[32][..]));
        assert_eq!(direct_only.gate.fail_on_threshold, Some(false));

        let dino_flow_path = repo_root
            .join("configs/hyper2d_adapter_bank")
            .join("psnr_gate_exact_oracle_10k8x8_2048_rank132_dino_flow_overfit.toml");
        let dino_flow = load_psnr_gate_experiment_config(Some(&dino_flow_path)).unwrap();
        assert!(dino_flow.input.base_model.is_some());
        assert!(dino_flow.input.adapter_bank.is_some());
        assert!(dino_flow.input.oracle_report.is_some());
        assert!(dino_flow.input.hyper.is_some());
        assert!(dino_flow.condition.feature_cache.is_some());
        assert_eq!(dino_flow.condition.token_grid_width, Some(8));
        assert_eq!(dino_flow.condition.token_grid_height, Some(8));
        assert_eq!(dino_flow.eval.particles, Some(2048));
        assert_eq!(dino_flow.eval.steps.as_deref(), Some(&[32][..]));
        assert_eq!(dino_flow.gate.fail_on_threshold, Some(false));

        let dino_flow_linear_path = repo_root
            .join("configs/hyper2d_adapter_bank")
            .join("psnr_gate_exact_oracle_10k8x8_2048_rank132_dino_flow_linear_solve_overfit.toml");
        let dino_flow_linear =
            load_psnr_gate_experiment_config(Some(&dino_flow_linear_path)).unwrap();
        assert!(dino_flow_linear.input.hyper.is_some());
        assert!(dino_flow_linear.condition.feature_cache.is_some());
        assert_eq!(dino_flow_linear.condition.token_grid_width, Some(8));
        assert_eq!(dino_flow_linear.condition.token_grid_height, Some(8));
        assert_eq!(dino_flow_linear.eval.particles, Some(2048));
        assert_eq!(dino_flow_linear.eval.steps.as_deref(), Some(&[32][..]));
        assert_eq!(dino_flow_linear.gate.fail_on_threshold, Some(false));

        let dino_flow_warmstart_path = repo_root.join("configs/hyper2d_adapter_bank").join(
            "psnr_gate_exact_oracle_10k8x8_2048_rank132_dino_flow_zero_source_warmstart.toml",
        );
        let dino_flow_warmstart =
            load_psnr_gate_experiment_config(Some(&dino_flow_warmstart_path)).unwrap();
        assert!(dino_flow_warmstart.input.hyper.is_some());
        assert!(dino_flow_warmstart.condition.feature_cache.is_some());
        assert_eq!(dino_flow_warmstart.condition.token_grid_width, Some(8));
        assert_eq!(dino_flow_warmstart.condition.token_grid_height, Some(8));
        assert_eq!(dino_flow_warmstart.eval.particles, Some(2048));
        assert_eq!(dino_flow_warmstart.eval.steps.as_deref(), Some(&[32][..]));
        assert_eq!(dino_flow_warmstart.gate.fail_on_threshold, Some(false));

        for file_name in [
            "psnr_gate_exact_oracle_10k8x8_2048_rank132_dino_flow_zero_source_overfit.toml",
            "psnr_gate_exact_oracle_10k8x8_2048_rank132_dino_flow_zero_source_h384_lr2e3.toml",
            "psnr_gate_exact_oracle_10k8x8_2048_rank132_dino_flow_zero_source_h384_lr2e4_refine.toml",
            "psnr_gate_exact_oracle_10k8x8_2048_rank132_dino_flow_zero_source_h384_lr2e5_refine2.toml",
            "psnr_gate_exact_oracle_10k8x8_2048_rank132_dino_flow_zero_source_h384_sampled_refine.toml",
            "psnr_gate_exact_oracle_10k8x8_2048_rank132_dino_flow_zero_source_h384_sampled_refine2.toml",
            "psnr_gate_exact_oracle_10k8x8_2048_rank132_dino_flow_zero_source_h384_sampled_weighted_refine.toml",
            "psnr_gate_exact_oracle_10k8x8_2048_rank132_dino_flow_zero_source_h384_sampled_weighted_margin_refine.toml",
            "psnr_gate_exact_oracle_10k8x8_2048_rank132_dino_flow_zero_source_h384_sampled_weighted_floor_refine.toml",
        ] {
            let path = repo_root
                .join("configs/hyper2d_adapter_bank")
                .join(file_name);
            let config = load_psnr_gate_experiment_config(Some(&path)).unwrap();
            assert!(config.input.hyper.is_some());
            assert!(config.condition.feature_cache.is_some());
            assert_eq!(config.condition.token_grid_width, Some(8));
            assert_eq!(config.condition.token_grid_height, Some(8));
            assert_eq!(config.eval.particles, Some(2048));
            assert_eq!(config.eval.steps.as_deref(), Some(&[32][..]));
            assert_eq!(config.gate.min_render_rgb_psnr_db, Some(26.0));
            assert_eq!(config.gate.fail_on_threshold, Some(false));
        }
    }

    #[test]
    fn selected_entries_len_supports_direct_only_reports() {
        let mut entries = Vec::new();
        for slug in ["a", "b"] {
            for step in [32, 64] {
                entries.push(Hyper2dPsnrGateEntry {
                    slug: slug.to_string(),
                    split: "train".to_string(),
                    oracle_split: "train".to_string(),
                    condition: format!("{slug}.png"),
                    oracle_condition: format!("{slug}.png"),
                    kind: "direct",
                    model: format!("{slug}.bpk"),
                    target_model: format!("{slug}_oracle.bpk"),
                    rollout_steps: step,
                    render_rgb_psnr_db: 99.0,
                    passed: true,
                    metrics: test_metrics(step),
                });
            }
        }

        assert_eq!(selected_entries_len(&entries, 2), 2);
    }

    fn test_metrics(rollout_steps: usize) -> CliHyper2dDynamicsMetricsReport {
        CliHyper2dDynamicsMetricsReport {
            particle_count: 2048,
            rollout_steps,
            update_prob: 0.5,
            seed: 42,
            seed_scale: 0.5,
            seed_mode: ParticleSeed::UniformCircle,
            image_size: 128,
            render_sigma_px: 1.0,
            position_mse: 0.0,
            position_psnr_db: 99.0,
            state_mse: 0.0,
            state_psnr_db: 99.0,
            tail_rgb_mse: 0.0,
            tail_rgb_psnr_db: 99.0,
            render_rgb_mse: 0.0,
            render_rgb_psnr_db: 99.0,
            render_density_mse: 0.0,
            render_density_psnr_db: 99.0,
            mean_dx_mse: 0.0,
            mean_dx_mae: 0.0,
            target_final_mean_dx: 0.0,
            generated_final_mean_dx: 0.0,
        }
    }
}
