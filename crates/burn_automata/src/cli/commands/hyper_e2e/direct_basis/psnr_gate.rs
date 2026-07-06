use std::collections::{BTreeMap, HashMap};

use crate::cli::commands::hyper_support::{
    load_condition_image_2d, load_hyper_2d, write_pretty_json,
};
use crate::cli::prelude::*;

use super::super::sources::sanitize_slug;
use super::{load_direct_basis_adapter_bank, resolve_direct_basis_artifact_path};

#[derive(Deserialize)]
struct PsnrGateOracleReportLoad {
    oracle_validation: Option<PsnrGateOracleValidationLoad>,
}

#[derive(Deserialize)]
struct PsnrGateOracleValidationLoad {
    entries: Vec<PsnrGateOracleEntryLoad>,
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
    hyper: String,
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
    let hyper_model = load_hyper_2d(&hyper)?;
    if hyper_model.npa_config != base.config {
        return Err(std::io::Error::other(
            "hyper checkpoint NPA config must match base model config",
        )
        .into());
    }
    let bank = load_direct_basis_adapter_bank(&adapter_bank)?;
    if bank.entries.is_empty() {
        return Err(std::io::Error::other("adapter bank has no entries").into());
    }
    if bank.adapter_rank != hyper_model.config.adapter_rank
        || (bank.adapter_alpha - hyper_model.config.adapter_alpha).abs() > f32::EPSILON
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
    let selected_entries = bank
        .entries
        .iter()
        .take(if limit == 0 {
            bank.entries.len()
        } else {
            limit.min(bank.entries.len())
        })
        .collect::<Vec<_>>();
    if selected_entries.is_empty() {
        return Err(std::io::Error::other("no adapter-bank entries selected").into());
    }

    let mut report_entries = Vec::with_capacity(selected_entries.len() * steps.len() * 2);
    for bank_entry in selected_entries {
        let oracle_entry = oracle_by_key
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
            });
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

        let condition = load_condition_image_2d(&condition_path)?;
        let hyper_adapter = hyper_model.predict_adapter(&condition)?;
        let hyper_materialized = hyper_adapter.apply_to_model(&base)?;
        let hyper_manifest = BpkModelManifest::from_model(
            &hyper_materialized,
            base_manifest.hashgrid.clone(),
            Some(format!("hyper2d-psnr-gate:{}", bank_entry.slug)),
        );
        crate::import::save_manifest(&hyper_model_path, &hyper_manifest)?;

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

            let hyper_metrics = evaluate_gate_model(
                &hyper_materialized,
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

    let summaries = summarize_gate_entries(&report_entries, min_render_rgb_psnr_db);
    let passed = summaries.iter().all(|summary| summary.passed);
    let report = Hyper2dPsnrGateReport {
        preset,
        base_model: base_model.display().to_string(),
        adapter_bank: adapter_bank.display().to_string(),
        oracle_report: oracle_report.display().to_string(),
        hyper: hyper.display().to_string(),
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
    let divisor = step_count.saturating_mul(2).max(1);
    entries.len() / divisor
}
