use std::fmt::Write as _;

use serde_json::Value;

use super::hyper_support::write_pretty_json;
use crate::cli::prelude::*;

mod hyper2d_latex;

const DIRECT_ORACLE_READY_MAX_RATIO: f64 = 1.20;
const ADAPTER_VECTOR_READY_MAX_NRMSE: f64 = 0.35;
const ADAPTER_VECTOR_READY_MIN_COSINE: f64 = 0.80;
const ADAPTER_ROLLOUT_READY_MAX_RATIO: f64 = 1.15;
const MIN_QUALITY_ROLLOUT_PARTICLES: usize = 2048;
const MIN_QUALITY_TARGET_POINTS: usize = 2048;
const MIN_QUALITY_ORACLE_EXAMPLES_PER_SPLIT: usize = 8;
const MIN_CONDITIONING_TRAIN_EXAMPLES: usize = 900;
const MIN_CONDITIONING_HOLDOUT_EXAMPLES: usize = 100;
const MIN_CONDITIONING_ROLLOUT_EXAMPLES_PER_SPLIT: usize = 8;

pub(crate) fn run_report_hyper_2d(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::ReportHyper2d {
        report,
        oracle_report,
        output_dir,
        summary_output,
        markdown_output,
        latex_output,
        require_quality_ready,
    } = command
    else {
        return Err(std::io::Error::other("expected report-hyper2d command").into());
    };

    let report_value = read_json_value(&report)?;
    let oracle_value = oracle_report
        .as_ref()
        .map(|path| read_json_value(path))
        .transpose()?;
    let summary = summarize_hyper2d_report(
        &report,
        &report_value,
        oracle_report.as_deref(),
        oracle_value.as_ref(),
    )?;

    let summary_output =
        summary_output.unwrap_or_else(|| output_dir.join("validation_summary.json"));
    let markdown_output =
        markdown_output.unwrap_or_else(|| output_dir.join("validation_report.md"));
    let latex_output = latex_output.unwrap_or_else(|| output_dir.join("validation_report.tex"));
    write_pretty_json(&summary_output, &summary)?;
    write_text(&markdown_output, &markdown_for_hyper2d_summary(&summary))?;
    write_text(
        &latex_output,
        &hyper2d_latex::latex_for_hyper2d_summary(&summary),
    )?;

    println!(
        "wrote summary={} markdown={} latex={} status={}",
        summary_output.display(),
        markdown_output.display(),
        latex_output.display(),
        summary.quality_status
    );
    if require_quality_ready && !summary.quality_ready {
        return Err(std::io::Error::other(format!(
            "Hyper2D quality gate failed: status={}",
            summary.quality_status
        ))
        .into());
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Hyper2dReportKind {
    DirectBasis,
    AdapterBankConditioning,
}

#[derive(Clone, Debug, Serialize)]
struct Hyper2dValidationSummary {
    report_kind: Hyper2dReportKind,
    source_report: String,
    oracle_report: Option<String>,
    experiment_config: Option<String>,
    preset: Option<String>,
    output_dir: Option<String>,
    train_examples: Option<usize>,
    holdout_examples: Option<usize>,
    evaluated_system: &'static str,
    burn_first_status: &'static str,
    quality_status: &'static str,
    quality_ready: bool,
    quality_gates: QualityGateSummary,
    hypernet_generalization_validated: bool,
    direct_basis: Option<DirectBasisReportSummary>,
    adapter_bank_conditioning: Option<AdapterBankConditioningSummary>,
    interpretation: Vec<String>,
    next_steps: Vec<String>,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct QualityGateSummary {
    direct_oracle_ready_max_ratio: f64,
    adapter_vector_ready_max_nrmse: f64,
    adapter_vector_ready_min_cosine: f64,
    adapter_rollout_ready_max_ratio: f64,
    min_quality_rollout_particles: usize,
    min_quality_target_points: usize,
    min_quality_oracle_examples_per_split: usize,
    min_conditioning_train_examples: usize,
    min_conditioning_holdout_examples: usize,
    min_conditioning_rollout_examples_per_split: usize,
}

#[derive(Clone, Debug, Serialize)]
struct DirectBasisReportSummary {
    backend: Option<String>,
    shared_base_output: Option<String>,
    adapter_bank_output: Option<String>,
    adapter_rank: Option<usize>,
    adapter_alpha: Option<f64>,
    train_loss_initial: Option<f64>,
    train_loss_final: Option<f64>,
    train_loss_reduction_fraction: Option<f64>,
    holdout_loss_initial: Option<f64>,
    holdout_loss_final: Option<f64>,
    holdout_loss_reduction_fraction: Option<f64>,
    rollout_particles: Option<usize>,
    rollout_steps: Option<usize>,
    min_target_points: Option<usize>,
    target_loss_image_size: Option<usize>,
    oracle_train: Option<OracleSplitSummary>,
    oracle_holdout: Option<OracleSplitSummary>,
    particle_steps_per_sec: Option<ThroughputSummary>,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct OracleSplitSummary {
    examples: Option<usize>,
    mean_shared_loss: Option<f64>,
    mean_oracle_loss: Option<f64>,
    mean_ratio_to_oracle: Option<f64>,
    max_ratio_to_oracle: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
struct AdapterBankConditioningSummary {
    backend: Option<String>,
    training_backend: Option<String>,
    generator_architecture: Option<String>,
    generator_objective: Option<String>,
    adapter_target_canonicalization: Option<String>,
    hyper_output: Option<String>,
    adapter_bank: Option<String>,
    shared_base: Option<String>,
    condition_encoder: Option<String>,
    adapter_rank: Option<usize>,
    adapter_alpha: Option<f64>,
    adapter_parameter_count: Option<usize>,
    target_output_scale: Option<f64>,
    target_outside_output_scale_fraction: Option<f64>,
    train_loss_initial: Option<f64>,
    train_loss_final: Option<f64>,
    train_loss_reduction_fraction: Option<f64>,
    best_loss: Option<f64>,
    best_step: Option<usize>,
    elapsed_ms: Option<f64>,
    rollout_particles: Option<usize>,
    rollout_steps: Option<usize>,
    target_points: Option<usize>,
    target_loss_image_size: Option<usize>,
    train_vector_metrics: Option<AdapterVectorMetricSummary>,
    holdout_vector_metrics: Option<AdapterVectorMetricSummary>,
    train_rollout: Option<AdapterRolloutSplitSummary>,
    holdout_rollout: Option<AdapterRolloutSplitSummary>,
    adapter_values_per_sec: Option<ThroughputSummary>,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct ThroughputSummary {
    samples: usize,
    median: f64,
    max: f64,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct AdapterVectorMetricSummary {
    examples: Option<usize>,
    normalized_rmse_to_target_rms: Option<f64>,
    mean_cosine_similarity: Option<f64>,
    target_rms: Option<f64>,
    prediction_rms: Option<f64>,
    max_abs_error: Option<f64>,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct AdapterRolloutSplitSummary {
    examples: Option<usize>,
    mean_static_loss: Option<f64>,
    mean_hyper_loss: Option<f64>,
    mean_ratio_to_static: Option<f64>,
    max_ratio_to_static: Option<f64>,
}

fn summarize_hyper2d_report(
    report_path: &Path,
    report: &Value,
    oracle_path: Option<&Path>,
    oracle_report: Option<&Value>,
) -> Result<Hyper2dValidationSummary, Box<dyn std::error::Error>> {
    if report.get("train_vector_metrics").is_some() && report.get("hyper_output").is_some() {
        return Ok(summarize_adapter_bank_conditioning(report_path, report));
    }
    if report.get("adapter_bank_output").is_some() || report.get("gpu_training").is_some() {
        return Ok(summarize_direct_basis(
            report_path,
            report,
            oracle_path,
            oracle_report,
        ));
    }

    Err(std::io::Error::other(
        "unsupported Hyper2D report shape: expected direct-basis or adapter-bank-conditioning report",
    )
    .into())
}

fn summarize_direct_basis(
    report_path: &Path,
    report: &Value,
    oracle_path: Option<&Path>,
    oracle_report: Option<&Value>,
) -> Hyper2dValidationSummary {
    let oracle_block = oracle_report
        .and_then(|report| report.get("oracle_validation").or(Some(report)))
        .or_else(|| report.get("oracle_validation"));
    let direct = DirectBasisReportSummary {
        backend: path_string(report, &["gpu_training", "backend"])
            .or_else(|| path_string(report, &["training_device"])),
        shared_base_output: path_string(report, &["shared_base_output"]),
        adapter_bank_output: path_string(report, &["adapter_bank_output"]),
        adapter_rank: path_usize(report, &["adapter_rank"]),
        adapter_alpha: path_f64(report, &["adapter_alpha"]),
        train_loss_initial: path_f64(report, &["initial_train_loss", "mean_total_loss"]),
        train_loss_final: path_f64(report, &["final_train_loss", "mean_total_loss"]),
        train_loss_reduction_fraction: reduction_fraction(
            path_f64(report, &["initial_train_loss", "mean_total_loss"]),
            path_f64(report, &["final_train_loss", "mean_total_loss"]),
        ),
        holdout_loss_initial: path_f64(report, &["initial_holdout_loss", "mean_total_loss"]),
        holdout_loss_final: path_f64(report, &["final_holdout_loss", "mean_total_loss"]),
        holdout_loss_reduction_fraction: reduction_fraction(
            path_f64(report, &["initial_holdout_loss", "mean_total_loss"]),
            path_f64(report, &["final_holdout_loss", "mean_total_loss"]),
        ),
        rollout_particles: path_usize(report, &["rollout_particles"]),
        rollout_steps: path_usize(report, &["rollout_steps"]),
        min_target_points: min_adapter_target_points(report),
        target_loss_image_size: path_usize(report, &["target_loss_config", "image_size"]),
        oracle_train: oracle_block.and_then(|block| oracle_split_summary(block, "train_summary")),
        oracle_holdout: oracle_block
            .and_then(|block| oracle_split_summary(block, "holdout_summary")),
        particle_steps_per_sec: throughput_summary(report, &["history"], "particle_steps_per_sec"),
    };

    let oracle_ratio_ready = split_oracle_ratio_ready(direct.oracle_train)
        && split_oracle_ratio_ready(direct.oracle_holdout)
        && (direct.oracle_train.is_some() || direct.oracle_holdout.is_some());
    let oracle_example_ready = split_oracle_examples_ready(direct.oracle_train)
        && split_oracle_examples_ready(direct.oracle_holdout);
    let oracle_ready = oracle_ratio_ready && oracle_example_ready;
    let quality_scale_ready = particle_count_ready(direct.rollout_particles)
        && target_points_ready(direct.min_target_points);
    let quality_status = if oracle_ready && quality_scale_ready {
        "direct_basis_oracle_ready"
    } else if oracle_ratio_ready && !oracle_example_ready {
        "insufficient_oracle_examples"
    } else if oracle_ready {
        "quality_particle_count_too_low"
    } else if direct.oracle_train.is_none() && direct.oracle_holdout.is_none() {
        "needs_oracle_validation"
    } else {
        "direct_basis_oracle_gap"
    };

    let mut interpretation = vec![
        "This report evaluates a shared NPA base plus directly optimized stored per-sample LoRA adapters.".to_string(),
        "It does not validate image-to-LoRA hypernet generalization by itself.".to_string(),
    ];
    if oracle_ready {
        interpretation.push(format!(
            "Sampled oracle ratios are within the {:.2}x direct-basis gate.",
            DIRECT_ORACLE_READY_MAX_RATIO
        ));
    } else if oracle_ratio_ready && !oracle_example_ready {
        interpretation.push(format!(
            "Oracle ratios are within threshold, but each split needs at least {MIN_QUALITY_ORACLE_EXAMPLES_PER_SPLIT} oracle examples for a quality claim."
        ));
    } else if direct.oracle_train.is_some() || direct.oracle_holdout.is_some() {
        interpretation.push(format!(
            "At least one sampled oracle ratio exceeds the {:.2}x direct-basis gate.",
            DIRECT_ORACLE_READY_MAX_RATIO
        ));
    } else {
        interpretation.push(
            "No oracle validation block was found; run validate-hyper2d-direct-basis-oracles or pass --oracle-report.".to_string(),
        );
    }
    if !quality_scale_ready {
        interpretation.push(quality_scale_interpretation(
            direct.rollout_particles,
            direct.min_target_points,
        ));
    }

    Hyper2dValidationSummary {
        report_kind: Hyper2dReportKind::DirectBasis,
        source_report: report_path.display().to_string(),
        oracle_report: oracle_path.map(|path| path.display().to_string()),
        experiment_config: path_string(report, &["experiment_config"]),
        preset: path_string(report, &["preset"]),
        output_dir: path_string(report, &["output_dir"]),
        train_examples: path_usize(report, &["train_examples"]),
        holdout_examples: path_usize(report, &["holdout_examples"]),
        evaluated_system: "shared NPA base plus directly optimized stored per-sample LoRA adapters",
        burn_first_status: "burn_wgpu_primary",
        quality_status,
        quality_ready: oracle_ready && quality_scale_ready,
        quality_gates: quality_gate_summary(),
        hypernet_generalization_validated: false,
        direct_basis: Some(direct),
        adapter_bank_conditioning: None,
        interpretation,
        next_steps: vec![
            "Run report-hyper2d over the matching adapter-bank conditioning report to validate condition-to-LoRA generalization.".to_string(),
            format!("Rerun direct-basis oracle validation with at least {MIN_QUALITY_ROLLOUT_PARTICLES} rollout particles and {MIN_QUALITY_TARGET_POINTS} target samples before making paper-quality claims."),
            "Expand oracle validation examples before claiming direct-basis parity for a larger slice.".to_string(),
        ],
    }
}

fn summarize_adapter_bank_conditioning(
    report_path: &Path,
    report: &Value,
) -> Hyper2dValidationSummary {
    let adapter = AdapterBankConditioningSummary {
        backend: path_string(report, &["backend"]),
        training_backend: path_string(report, &["training", "backend"]),
        generator_architecture: path_string(report, &["generator_architecture"]),
        generator_objective: path_string(report, &["generator_objective"]),
        adapter_target_canonicalization: path_string(report, &["adapter_target_canonicalization"]),
        hyper_output: path_string(report, &["hyper_output"]),
        adapter_bank: path_string(report, &["adapter_bank"]),
        shared_base: path_string(report, &["shared_base"]),
        condition_encoder: path_string(report, &["condition_encoder"]),
        adapter_rank: path_usize(report, &["adapter_rank"]),
        adapter_alpha: path_f64(report, &["adapter_alpha"]),
        adapter_parameter_count: path_usize(report, &["adapter_parameter_count"]),
        target_output_scale: path_f64(report, &["target_stats", "output_scale"]),
        target_outside_output_scale_fraction: path_f64(
            report,
            &[
                "target_stats",
                "target_values_outside_output_scale_fraction",
            ],
        ),
        train_loss_initial: path_f64(report, &["training", "initial_loss"]),
        train_loss_final: path_f64(report, &["training", "final_loss"]),
        train_loss_reduction_fraction: reduction_fraction(
            path_f64(report, &["training", "initial_loss"]),
            path_f64(report, &["training", "final_loss"]),
        ),
        best_loss: path_f64(report, &["training", "best_loss"]),
        best_step: path_usize(report, &["training", "best_step"]),
        elapsed_ms: path_f64(report, &["training", "elapsed_ms"]),
        rollout_particles: path_usize(report, &["rollout_particles"]),
        rollout_steps: path_usize(report, &["rollout_steps"]),
        target_points: path_usize(report, &["target_points"]),
        target_loss_image_size: path_usize(report, &["target_loss_config", "image_size"]),
        train_vector_metrics: adapter_vector_summary(report, "train_vector_metrics"),
        holdout_vector_metrics: adapter_vector_summary(report, "holdout_vector_metrics"),
        train_rollout: adapter_rollout_summary(report, &["rollout_eval", "train_summary"]),
        holdout_rollout: adapter_rollout_summary(report, &["rollout_eval", "holdout_summary"]),
        adapter_values_per_sec: throughput_summary(
            report,
            &["training", "history"],
            "adapter_values_per_sec",
        ),
    };

    let vector_ready = split_vector_ready(adapter.train_vector_metrics)
        && split_vector_ready(adapter.holdout_vector_metrics);
    let rollout_ready =
        split_rollout_ready(adapter.train_rollout) && split_rollout_ready(adapter.holdout_rollout);
    let rollout_example_ready = split_rollout_examples_ready(adapter.train_rollout)
        && split_rollout_examples_ready(adapter.holdout_rollout);
    let quality_scale_ready = particle_count_ready(adapter.rollout_particles)
        && target_points_ready(adapter.target_points);
    let broad_example_ready = conditioning_example_count_ready(
        path_usize(report, &["train_examples"]),
        path_usize(report, &["holdout_examples"]),
    );
    let dino_ready = adapter
        .condition_encoder
        .as_deref()
        .is_some_and(|encoder| encoder.contains("dino"));
    let flow_ready = adapter
        .generator_objective
        .as_deref()
        .is_some_and(|objective| objective.contains("rectified-flow"));
    let quality_status = if !dino_ready {
        "conditioning_not_dino"
    } else if !flow_ready {
        "conditioning_not_rectified_flow"
    } else if !broad_example_ready {
        "insufficient_conditioning_examples"
    } else if !rollout_example_ready {
        "insufficient_conditioning_rollout_examples"
    } else if vector_ready && rollout_ready && quality_scale_ready {
        "conditioning_quality_ready"
    } else if vector_ready && rollout_ready {
        "quality_particle_count_too_low"
    } else if !vector_ready && rollout_ready {
        "conditioning_vector_underfit"
    } else {
        "conditioning_not_quality_ready"
    };

    let mut interpretation = vec![
        "This report evaluates an image-condition model that predicts LoRA adapter weights from a stored adapter bank.".to_string(),
        "It validates the condition-to-LoRA stage, but only against the feature encoder and adapter bank used in this run.".to_string(),
    ];
    if !dino_ready {
        interpretation.push(
            "This is not a DINO-conditioned run; it cannot prove DINO HyperNPA generalization."
                .to_string(),
        );
    }
    if !flow_ready {
        interpretation.push(
            "The generator objective is not rectified flow; this report is a static LoRA vector regression baseline.".to_string(),
        );
    }
    if !broad_example_ready {
        interpretation.push(format!(
            "The broad-generalization gate requires at least {MIN_CONDITIONING_TRAIN_EXAMPLES} train and {MIN_CONDITIONING_HOLDOUT_EXAMPLES} holdout examples."
        ));
    }
    if !rollout_example_ready {
        interpretation.push(format!(
            "The rollout gate requires at least {MIN_CONDITIONING_ROLLOUT_EXAMPLES_PER_SPLIT} generated-vs-static rollout examples per split."
        ));
    }
    if !vector_ready {
        interpretation.push(format!(
            "Adapter-vector metrics miss the quality gate: NRMSE must be <= {:.2} and cosine must be >= {:.2} on train and holdout splits.",
            ADAPTER_VECTOR_READY_MAX_NRMSE, ADAPTER_VECTOR_READY_MIN_COSINE
        ));
    }
    if !rollout_ready {
        interpretation.push(format!(
            "Rollout-vs-static-adapter ratios miss the {:.2}x gate on at least one split.",
            ADAPTER_ROLLOUT_READY_MAX_RATIO
        ));
    }
    if vector_ready && rollout_ready {
        interpretation.push(
            "Generated LoRAs are close to stored direct LoRAs under both vector and rollout gates."
                .to_string(),
        );
    }
    if !quality_scale_ready {
        interpretation.push(quality_scale_interpretation(
            adapter.rollout_particles,
            adapter.target_points,
        ));
    }

    let quality_ready = dino_ready
        && flow_ready
        && broad_example_ready
        && rollout_example_ready
        && vector_ready
        && rollout_ready
        && quality_scale_ready;
    let mut next_steps = Vec::new();
    if !dino_ready {
        next_steps.push(
            "Run the DINO/Burn-DINO condition encoder path and rerun the same report gates."
                .to_string(),
        );
    }
    if !flow_ready {
        next_steps.push(
            "Replace the static vector-regression adapter head with a rectified-flow LoRA generator or add a flow refinement stage over generated adapters.".to_string(),
        );
    }
    if !broad_example_ready {
        next_steps.push(format!(
            "Scale the conditioning run to at least {MIN_CONDITIONING_TRAIN_EXAMPLES}+{MIN_CONDITIONING_HOLDOUT_EXAMPLES} train/holdout examples before making a broad 1k claim."
        ));
    }
    if !rollout_example_ready {
        next_steps.push(format!(
            "Evaluate generated-vs-direct LoRA rollout quality on at least {MIN_CONDITIONING_ROLLOUT_EXAMPLES_PER_SPLIT} examples per split."
        ));
    }
    next_steps.extend([
        format!("Rerun adapter-bank rollout evaluation with at least {MIN_QUALITY_ROLLOUT_PARTICLES} particles and {MIN_QUALITY_TARGET_POINTS} target samples before comparing generated LoRAs in the paper."),
        "If vector metrics remain poor, train a stronger adapter decoder or a two-stage residual/flow head before scaling dataset size.".to_string(),
        "Compare generated LoRAs against direct LoRAs and single-sample overfit oracles on the same held-out examples.".to_string(),
    ]);

    Hyper2dValidationSummary {
        report_kind: Hyper2dReportKind::AdapterBankConditioning,
        source_report: report_path.display().to_string(),
        oracle_report: None,
        experiment_config: path_string(report, &["experiment_config"]),
        preset: path_string(report, &["preset"]),
        output_dir: path_string(report, &["output_dir"]),
        train_examples: path_usize(report, &["train_examples"]),
        holdout_examples: path_usize(report, &["holdout_examples"]),
        evaluated_system: "image condition to HyperNPA LoRA generator over a stored shared-base adapter bank",
        burn_first_status: "burn_wgpu_primary",
        quality_status,
        quality_ready,
        quality_gates: quality_gate_summary(),
        hypernet_generalization_validated: quality_ready,
        direct_basis: None,
        adapter_bank_conditioning: Some(adapter),
        interpretation,
        next_steps,
    }
}

fn quality_gate_summary() -> QualityGateSummary {
    QualityGateSummary {
        direct_oracle_ready_max_ratio: DIRECT_ORACLE_READY_MAX_RATIO,
        adapter_vector_ready_max_nrmse: ADAPTER_VECTOR_READY_MAX_NRMSE,
        adapter_vector_ready_min_cosine: ADAPTER_VECTOR_READY_MIN_COSINE,
        adapter_rollout_ready_max_ratio: ADAPTER_ROLLOUT_READY_MAX_RATIO,
        min_quality_rollout_particles: MIN_QUALITY_ROLLOUT_PARTICLES,
        min_quality_target_points: MIN_QUALITY_TARGET_POINTS,
        min_quality_oracle_examples_per_split: MIN_QUALITY_ORACLE_EXAMPLES_PER_SPLIT,
        min_conditioning_train_examples: MIN_CONDITIONING_TRAIN_EXAMPLES,
        min_conditioning_holdout_examples: MIN_CONDITIONING_HOLDOUT_EXAMPLES,
        min_conditioning_rollout_examples_per_split: MIN_CONDITIONING_ROLLOUT_EXAMPLES_PER_SPLIT,
    }
}

fn read_json_value(path: &Path) -> Result<Value, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

fn write_text(path: &Path, text: &str) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, text)?;
    Ok(())
}

fn path_value<'a>(root: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut value = root;
    for key in path {
        value = value.get(*key)?;
    }
    Some(value)
}

fn path_f64(root: &Value, path: &[&str]) -> Option<f64> {
    path_value(root, path).and_then(Value::as_f64)
}

fn path_usize(root: &Value, path: &[&str]) -> Option<usize> {
    path_value(root, path)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

fn path_string(root: &Value, path: &[&str]) -> Option<String> {
    path_value(root, path)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn reduction_fraction(initial: Option<f64>, final_loss: Option<f64>) -> Option<f64> {
    let initial = initial?;
    let final_loss = final_loss?;
    (initial > 0.0).then_some((initial - final_loss) / initial)
}

fn oracle_split_summary(root: &Value, split: &str) -> Option<OracleSplitSummary> {
    let split = root.get(split)?;
    Some(OracleSplitSummary {
        examples: path_usize(split, &["examples"]),
        mean_shared_loss: path_f64(split, &["mean_shared_loss"]),
        mean_oracle_loss: path_f64(split, &["mean_oracle_loss"]),
        mean_ratio_to_oracle: path_f64(split, &["mean_ratio_to_oracle"]),
        max_ratio_to_oracle: path_f64(split, &["max_ratio_to_oracle"]),
    })
}

fn adapter_vector_summary(root: &Value, key: &str) -> Option<AdapterVectorMetricSummary> {
    let split = root.get(key)?;
    Some(AdapterVectorMetricSummary {
        examples: path_usize(split, &["examples"]),
        normalized_rmse_to_target_rms: path_f64(split, &["normalized_rmse_to_target_rms"]),
        mean_cosine_similarity: path_f64(split, &["mean_cosine_similarity"]),
        target_rms: path_f64(split, &["target_rms"]),
        prediction_rms: path_f64(split, &["prediction_rms"]),
        max_abs_error: path_f64(split, &["max_abs_error"]),
    })
}

fn adapter_rollout_summary(root: &Value, path: &[&str]) -> Option<AdapterRolloutSplitSummary> {
    let split = path_value(root, path)?;
    Some(AdapterRolloutSplitSummary {
        examples: path_usize(split, &["examples"]),
        mean_static_loss: path_f64(split, &["mean_static_loss"]),
        mean_hyper_loss: path_f64(split, &["mean_hyper_loss"]),
        mean_ratio_to_static: path_f64(split, &["mean_ratio_to_static"]),
        max_ratio_to_static: path_f64(split, &["max_ratio_to_static"]),
    })
}

fn min_adapter_target_points(root: &Value) -> Option<usize> {
    root.get("adapters")?
        .as_array()?
        .iter()
        .filter_map(|entry| path_usize(entry, &["target_points"]))
        .min()
}

fn throughput_summary(root: &Value, path: &[&str], metric: &str) -> Option<ThroughputSummary> {
    let mut values = path_value(root, path)?
        .as_array()?
        .iter()
        .filter_map(|entry| entry.get(metric).and_then(Value::as_f64))
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    let median = if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    };
    Some(ThroughputSummary {
        samples: values.len(),
        median,
        max: *values.last().expect("non-empty throughput values"),
    })
}

fn split_oracle_ratio_ready(split: Option<OracleSplitSummary>) -> bool {
    split
        .and_then(|split| split.max_ratio_to_oracle.or(split.mean_ratio_to_oracle))
        .is_some_and(|ratio| ratio <= DIRECT_ORACLE_READY_MAX_RATIO)
}

fn split_oracle_examples_ready(split: Option<OracleSplitSummary>) -> bool {
    split
        .and_then(|split| split.examples)
        .is_some_and(|examples| examples >= MIN_QUALITY_ORACLE_EXAMPLES_PER_SPLIT)
}

fn split_vector_ready(split: Option<AdapterVectorMetricSummary>) -> bool {
    split.is_some_and(|split| {
        split
            .normalized_rmse_to_target_rms
            .is_some_and(|value| value <= ADAPTER_VECTOR_READY_MAX_NRMSE)
            && split
                .mean_cosine_similarity
                .is_some_and(|value| value >= ADAPTER_VECTOR_READY_MIN_COSINE)
    })
}

fn split_rollout_ready(split: Option<AdapterRolloutSplitSummary>) -> bool {
    split
        .and_then(|split| split.max_ratio_to_static.or(split.mean_ratio_to_static))
        .is_some_and(|ratio| ratio <= ADAPTER_ROLLOUT_READY_MAX_RATIO)
}

fn split_rollout_examples_ready(split: Option<AdapterRolloutSplitSummary>) -> bool {
    split
        .and_then(|split| split.examples)
        .is_some_and(|examples| examples >= MIN_CONDITIONING_ROLLOUT_EXAMPLES_PER_SPLIT)
}

fn conditioning_example_count_ready(train: Option<usize>, holdout: Option<usize>) -> bool {
    train.is_some_and(|examples| examples >= MIN_CONDITIONING_TRAIN_EXAMPLES)
        && holdout.is_some_and(|examples| examples >= MIN_CONDITIONING_HOLDOUT_EXAMPLES)
}

fn particle_count_ready(value: Option<usize>) -> bool {
    value.is_some_and(|value| value >= MIN_QUALITY_ROLLOUT_PARTICLES)
}

fn target_points_ready(value: Option<usize>) -> bool {
    value.is_some_and(|value| value >= MIN_QUALITY_TARGET_POINTS)
}

fn quality_scale_interpretation(
    rollout_particles: Option<usize>,
    target_points: Option<usize>,
) -> String {
    format!(
        "Quality-scale validation is incomplete: rollout_particles={} must be >= {}, and target_points={} must be >= {}.",
        display_opt_usize(rollout_particles),
        MIN_QUALITY_ROLLOUT_PARTICLES,
        display_opt_usize(target_points),
        MIN_QUALITY_TARGET_POINTS
    )
}

fn markdown_for_hyper2d_summary(summary: &Hyper2dValidationSummary) -> String {
    let mut text = String::new();
    writeln!(text, "# Hyper2D Validation Report").unwrap();
    writeln!(text).unwrap();
    writeln!(text, "## Summary").unwrap();
    writeln!(text).unwrap();
    writeln!(text, "| Field | Value |").unwrap();
    writeln!(text, "| --- | --- |").unwrap();
    writeln!(text, "| Report kind | {} |", summary.report_kind.label()).unwrap();
    writeln!(text, "| Source report | `{}` |", summary.source_report).unwrap();
    writeln!(
        text,
        "| Experiment | {} |",
        display_opt_str(summary.experiment_config.as_deref())
    )
    .unwrap();
    writeln!(
        text,
        "| Preset | {} |",
        display_opt_str(summary.preset.as_deref())
    )
    .unwrap();
    writeln!(
        text,
        "| Train examples | {} |",
        display_opt_usize(summary.train_examples)
    )
    .unwrap();
    writeln!(
        text,
        "| Holdout examples | {} |",
        display_opt_usize(summary.holdout_examples)
    )
    .unwrap();
    writeln!(text, "| Evaluated system | {} |", summary.evaluated_system).unwrap();
    writeln!(text, "| Quality status | `{}` |", summary.quality_status).unwrap();
    writeln!(text, "| Quality ready | {} |", summary.quality_ready).unwrap();
    writeln!(
        text,
        "| Hypernet generalization validated | {} |",
        summary.hypernet_generalization_validated
    )
    .unwrap();
    writeln!(text).unwrap();
    writeln!(text, "## Quality Gates").unwrap();
    writeln!(text).unwrap();
    writeln!(text, "| Gate | Threshold |").unwrap();
    writeln!(text, "| --- | --- |").unwrap();
    writeln!(
        text,
        "| Direct-basis oracle max ratio | <= {:.2}x |",
        summary.quality_gates.direct_oracle_ready_max_ratio
    )
    .unwrap();
    writeln!(
        text,
        "| Adapter-vector normalized RMSE | <= {:.2} |",
        summary.quality_gates.adapter_vector_ready_max_nrmse
    )
    .unwrap();
    writeln!(
        text,
        "| Adapter-vector mean cosine | >= {:.2} |",
        summary.quality_gates.adapter_vector_ready_min_cosine
    )
    .unwrap();
    writeln!(
        text,
        "| Adapter rollout max ratio to static LoRA | <= {:.2}x |",
        summary.quality_gates.adapter_rollout_ready_max_ratio
    )
    .unwrap();
    writeln!(
        text,
        "| Quality rollout particles | >= {} |",
        summary.quality_gates.min_quality_rollout_particles
    )
    .unwrap();
    writeln!(
        text,
        "| Quality target samples | >= {} |",
        summary.quality_gates.min_quality_target_points
    )
    .unwrap();
    writeln!(
        text,
        "| Oracle examples per split | >= {} |",
        summary.quality_gates.min_quality_oracle_examples_per_split
    )
    .unwrap();

    if let Some(direct) = &summary.direct_basis {
        writeln!(text).unwrap();
        writeln!(text, "## Direct-Basis Metrics").unwrap();
        writeln!(text).unwrap();
        writeln!(text, "| Metric | Value |").unwrap();
        writeln!(text, "| --- | --- |").unwrap();
        writeln!(
            text,
            "| Backend | {} |",
            display_opt_str(direct.backend.as_deref())
        )
        .unwrap();
        writeln!(
            text,
            "| Adapter rank | {} |",
            display_opt_usize(direct.adapter_rank)
        )
        .unwrap();
        writeln!(
            text,
            "| Rollout particles | {} |",
            display_opt_usize(direct.rollout_particles)
        )
        .unwrap();
        writeln!(
            text,
            "| Rollout steps | {} |",
            display_opt_usize(direct.rollout_steps)
        )
        .unwrap();
        writeln!(
            text,
            "| Min target samples | {} |",
            display_opt_usize(direct.min_target_points)
        )
        .unwrap();
        writeln!(
            text,
            "| Target loss image size | {} |",
            display_opt_usize(direct.target_loss_image_size)
        )
        .unwrap();
        writeln!(
            text,
            "| Train loss initial | {} |",
            display_opt_f64(direct.train_loss_initial)
        )
        .unwrap();
        writeln!(
            text,
            "| Train loss final | {} |",
            display_opt_f64(direct.train_loss_final)
        )
        .unwrap();
        writeln!(
            text,
            "| Train loss reduction | {} |",
            display_opt_percent(direct.train_loss_reduction_fraction)
        )
        .unwrap();
        writeln!(
            text,
            "| Holdout loss initial | {} |",
            display_opt_f64(direct.holdout_loss_initial)
        )
        .unwrap();
        writeln!(
            text,
            "| Holdout loss final | {} |",
            display_opt_f64(direct.holdout_loss_final)
        )
        .unwrap();
        writeln!(
            text,
            "| Train oracle mean ratio | {} |",
            display_opt_f64(
                direct
                    .oracle_train
                    .and_then(|split| split.mean_ratio_to_oracle)
            )
        )
        .unwrap();
        writeln!(
            text,
            "| Holdout oracle mean ratio | {} |",
            display_opt_f64(
                direct
                    .oracle_holdout
                    .and_then(|split| split.mean_ratio_to_oracle)
            )
        )
        .unwrap();
        write_throughput_row(
            &mut text,
            "Particle steps/sec",
            direct.particle_steps_per_sec,
        );
    }

    if let Some(adapter) = &summary.adapter_bank_conditioning {
        writeln!(text).unwrap();
        writeln!(text, "## Adapter-Bank Conditioning Metrics").unwrap();
        writeln!(text).unwrap();
        writeln!(text, "| Metric | Value |").unwrap();
        writeln!(text, "| --- | --- |").unwrap();
        writeln!(
            text,
            "| Backend | {} |",
            display_opt_str(adapter.backend.as_deref())
        )
        .unwrap();
        writeln!(
            text,
            "| Training backend | {} |",
            display_opt_str(adapter.training_backend.as_deref())
        )
        .unwrap();
        writeln!(
            text,
            "| Condition encoder | {} |",
            display_opt_str(adapter.condition_encoder.as_deref())
        )
        .unwrap();
        writeln!(
            text,
            "| Adapter target canonicalization | {} |",
            display_opt_str(adapter.adapter_target_canonicalization.as_deref())
        )
        .unwrap();
        writeln!(
            text,
            "| Adapter rank | {} |",
            display_opt_usize(adapter.adapter_rank)
        )
        .unwrap();
        writeln!(
            text,
            "| Adapter parameters | {} |",
            display_opt_usize(adapter.adapter_parameter_count)
        )
        .unwrap();
        writeln!(
            text,
            "| Rollout particles | {} |",
            display_opt_usize(adapter.rollout_particles)
        )
        .unwrap();
        writeln!(
            text,
            "| Rollout steps | {} |",
            display_opt_usize(adapter.rollout_steps)
        )
        .unwrap();
        writeln!(
            text,
            "| Target samples | {} |",
            display_opt_usize(adapter.target_points)
        )
        .unwrap();
        writeln!(
            text,
            "| Target loss image size | {} |",
            display_opt_usize(adapter.target_loss_image_size)
        )
        .unwrap();
        writeln!(
            text,
            "| Train loss initial | {} |",
            display_opt_f64(adapter.train_loss_initial)
        )
        .unwrap();
        writeln!(
            text,
            "| Train loss final | {} |",
            display_opt_f64(adapter.train_loss_final)
        )
        .unwrap();
        writeln!(
            text,
            "| Train loss reduction | {} |",
            display_opt_percent(adapter.train_loss_reduction_fraction)
        )
        .unwrap();
        write_vector_row(&mut text, "Train vector", adapter.train_vector_metrics);
        write_vector_row(&mut text, "Holdout vector", adapter.holdout_vector_metrics);
        write_rollout_row(&mut text, "Train rollout", adapter.train_rollout);
        write_rollout_row(&mut text, "Holdout rollout", adapter.holdout_rollout);
        write_throughput_row(
            &mut text,
            "Adapter values/sec",
            adapter.adapter_values_per_sec,
        );
    }

    writeln!(text).unwrap();
    writeln!(text, "## Interpretation").unwrap();
    writeln!(text).unwrap();
    for item in &summary.interpretation {
        writeln!(text, "- {item}").unwrap();
    }
    writeln!(text).unwrap();
    writeln!(text, "## Next Steps").unwrap();
    writeln!(text).unwrap();
    for item in &summary.next_steps {
        writeln!(text, "- {item}").unwrap();
    }
    text
}

impl Hyper2dReportKind {
    fn label(&self) -> &'static str {
        match self {
            Self::DirectBasis => "direct-basis",
            Self::AdapterBankConditioning => "adapter-bank-conditioning",
        }
    }
}

fn write_vector_row(text: &mut String, label: &str, metrics: Option<AdapterVectorMetricSummary>) {
    let Some(metrics) = metrics else {
        writeln!(text, "| {label} | n/a |").unwrap();
        return;
    };
    writeln!(
        text,
        "| {label} | nrmse={} cosine={} target_rms={} pred_rms={} |",
        display_opt_f64(metrics.normalized_rmse_to_target_rms),
        display_opt_f64(metrics.mean_cosine_similarity),
        display_opt_f64(metrics.target_rms),
        display_opt_f64(metrics.prediction_rms)
    )
    .unwrap();
}

fn write_rollout_row(text: &mut String, label: &str, metrics: Option<AdapterRolloutSplitSummary>) {
    let Some(metrics) = metrics else {
        writeln!(text, "| {label} | n/a |").unwrap();
        return;
    };
    writeln!(
        text,
        "| {label} | mean_ratio={} max_ratio={} static_loss={} hyper_loss={} |",
        display_opt_f64(metrics.mean_ratio_to_static),
        display_opt_f64(metrics.max_ratio_to_static),
        display_opt_f64(metrics.mean_static_loss),
        display_opt_f64(metrics.mean_hyper_loss)
    )
    .unwrap();
}

fn write_throughput_row(text: &mut String, label: &str, metrics: Option<ThroughputSummary>) {
    let Some(metrics) = metrics else {
        writeln!(text, "| {label} | n/a |").unwrap();
        return;
    };
    writeln!(
        text,
        "| {label} | median={} max={} samples={} |",
        display_opt_f64(Some(metrics.median)),
        display_opt_f64(Some(metrics.max)),
        metrics.samples
    )
    .unwrap();
}

fn display_opt_str(value: Option<&str>) -> String {
    value
        .map(|value| format!("`{value}`"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn display_opt_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "n/a".to_string())
}

fn display_opt_f64(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn display_opt_percent(value: Option<f64>) -> String {
    value
        .map(|value| format!("{:.2}%", value * 100.0))
        .unwrap_or_else(|| "n/a".to_string())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn adapter_bank_summary_flags_vector_underfit_even_when_rollout_ratio_is_close() {
        let report = json!({
            "experiment_config": "configs/hyper2d_adapter_bank/smoke.toml",
            "preset": "Growing2d",
            "output_dir": "artifacts/hyper2d_adapter_bank",
            "shared_base": "shared_base.bpk",
            "adapter_bank": "adapter_bank.json",
            "hyper_output": "hyper_2d.json",
            "backend": "BurnWgpu",
            "condition_encoder": "dino-vits-cls-patch-mean-v1",
            "generator_objective": "rectified-flow-lora-vector",
            "adapter_rank": 16,
            "adapter_alpha": 16.0,
            "adapter_parameter_count": 5586,
            "train_examples": 900,
            "holdout_examples": 100,
            "target_stats": {
                "output_scale": 0.0788,
                "target_values_outside_output_scale_fraction": 0.0
            },
            "training": {
                "backend": "burn_wgpu_manual_mlp_adapter_regression",
                "initial_loss": 0.00019,
                "final_loss": 0.00018,
                "best_loss": 0.00018,
                "best_step": 300,
                "elapsed_ms": 1000.0,
                "history": [
                    {"adapter_values_per_sec": 100.0},
                    {"adapter_values_per_sec": 300.0},
                    {"adapter_values_per_sec": 200.0}
                ]
            },
            "train_vector_metrics": {
                "examples": 256,
                "normalized_rmse_to_target_rms": 0.98,
                "mean_cosine_similarity": 0.19,
                "target_rms": 0.014,
                "prediction_rms": 0.002,
                "max_abs_error": 0.06
            },
            "holdout_vector_metrics": {
                "examples": 100,
                "normalized_rmse_to_target_rms": 1.01,
                "mean_cosine_similarity": 0.05,
                "target_rms": 0.010,
                "prediction_rms": 0.002,
                "max_abs_error": 0.04
            },
            "rollout_eval": {
                "train_summary": {
                    "examples": 8,
                    "mean_static_loss": 10.5,
                    "mean_hyper_loss": 10.6,
                    "mean_ratio_to_static": 1.01,
                    "max_ratio_to_static": 1.02
                },
                "holdout_summary": {
                    "examples": 8,
                    "mean_static_loss": 3.8,
                    "mean_hyper_loss": 3.9,
                    "mean_ratio_to_static": 1.04,
                    "max_ratio_to_static": 1.08
                }
            }
        });

        let summary =
            summarize_hyper2d_report(Path::new("report.json"), &report, None, None).unwrap();

        assert_eq!(
            summary.report_kind,
            Hyper2dReportKind::AdapterBankConditioning
        );
        assert_eq!(summary.quality_status, "conditioning_vector_underfit");
        assert!(!summary.quality_ready);
        assert_eq!(
            summary.quality_gates.adapter_vector_ready_min_cosine,
            ADAPTER_VECTOR_READY_MIN_COSINE
        );
        assert!(!summary.hypernet_generalization_validated);
        let adapter = summary.adapter_bank_conditioning.unwrap();
        let throughput = adapter.adapter_values_per_sec.unwrap();
        assert_eq!(throughput.samples, 3);
        assert!((throughput.median - 200.0).abs() < f64::EPSILON);
        assert!((throughput.max - 300.0).abs() < f64::EPSILON);
    }

    #[test]
    fn direct_basis_summary_uses_external_oracle_report() {
        let report = json!({
            "experiment_config": "configs/hyper2d_direct_basis/omnisvg_10k.toml",
            "preset": "Growing2d",
            "output_dir": "artifacts/hyper2d_direct_basis",
            "shared_base_output": "shared_base.bpk",
            "adapter_bank_output": "adapter_bank.json",
            "gpu_training": {"backend": "burn_wgpu_autodiff_dense_direct_basis"},
            "adapter_rank": 16,
            "adapter_alpha": 16.0,
            "train_examples": 9000,
            "holdout_examples": 1000,
            "rollout_particles": 2048,
            "rollout_steps": 32,
            "target_loss_config": {"image_size": 128},
            "adapters": [
                {"target_points": 2048},
                {"target_points": 4096}
            ],
            "initial_train_loss": {"mean_total_loss": 12.0},
            "final_train_loss": {"mean_total_loss": 6.0},
            "history": [
                {"particle_steps_per_sec": 10.0},
                {"particle_steps_per_sec": 20.0}
            ]
        });
        let oracle_report = json!({
            "oracle_validation": {
                "train_summary": {
                    "examples": 8,
                    "mean_shared_loss": 7.8,
                    "mean_oracle_loss": 7.2,
                    "mean_ratio_to_oracle": 1.08,
                    "max_ratio_to_oracle": 1.14
                },
                "holdout_summary": {
                    "examples": 8,
                    "mean_shared_loss": 7.2,
                    "mean_oracle_loss": 6.7,
                    "mean_ratio_to_oracle": 1.07,
                    "max_ratio_to_oracle": 1.13
                }
            }
        });

        let summary = summarize_hyper2d_report(
            Path::new("report.json"),
            &report,
            Some(Path::new("oracle.json")),
            Some(&oracle_report),
        )
        .unwrap();

        assert_eq!(summary.report_kind, Hyper2dReportKind::DirectBasis);
        assert_eq!(summary.quality_status, "direct_basis_oracle_ready");
        assert!(summary.quality_ready);
        assert!(!summary.hypernet_generalization_validated);
        let direct = summary.direct_basis.unwrap();
        assert_eq!(direct.rollout_particles, Some(2048));
        assert_eq!(direct.min_target_points, Some(2048));
        assert_eq!(direct.train_loss_reduction_fraction, Some(0.5));
        let throughput = direct.particle_steps_per_sec.unwrap();
        assert_eq!(throughput.samples, 2);
        assert!((throughput.median - 15.0).abs() < f64::EPSILON);
        assert!((throughput.max - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn direct_basis_summary_rejects_low_particle_quality_claims() {
        let report = json!({
            "preset": "Growing2d",
            "shared_base_output": "shared_base.bpk",
            "adapter_bank_output": "adapter_bank.json",
            "gpu_training": {"backend": "burn_wgpu_autodiff_dense_direct_basis"},
            "rollout_particles": 64,
            "rollout_steps": 8,
            "adapters": [{"target_points": 256}],
            "oracle_validation": {
                "train_summary": {
                    "examples": 8,
                    "mean_ratio_to_oracle": 1.02,
                    "max_ratio_to_oracle": 1.03
                },
                "holdout_summary": {
                    "examples": 8,
                    "mean_ratio_to_oracle": 1.04,
                    "max_ratio_to_oracle": 1.05
                }
            }
        });

        let summary =
            summarize_hyper2d_report(Path::new("report.json"), &report, None, None).unwrap();

        assert_eq!(summary.report_kind, Hyper2dReportKind::DirectBasis);
        assert_eq!(summary.quality_status, "quality_particle_count_too_low");
        assert!(!summary.quality_ready);
        assert!(
            summary
                .interpretation
                .iter()
                .any(|item| item.contains("64"))
        );
    }
}
