use std::fmt::Write as _;

use super::{
    AdapterBankConditioningSummary, AdapterRolloutSplitSummary, AdapterVectorMetricSummary,
    DirectBasisReportSummary, Hyper2dValidationSummary, PsnrKindStepSummary, PsnrValidationSummary,
    ThroughputSummary,
};

pub(super) fn latex_for_hyper2d_summary(summary: &Hyper2dValidationSummary) -> String {
    let mut text = String::new();
    writeln!(text, "\\documentclass[11pt]{{article}}").unwrap();
    writeln!(text, "\\usepackage[margin=1in]{{geometry}}").unwrap();
    writeln!(text, "\\usepackage{{booktabs}}").unwrap();
    writeln!(text, "\\usepackage{{tabularx}}").unwrap();
    writeln!(text, "\\usepackage{{hyperref}}").unwrap();
    writeln!(text, "\\title{{Hyper2D Validation Report}}").unwrap();
    writeln!(text, "\\date{{}}").unwrap();
    writeln!(text, "\\begin{{document}}").unwrap();
    writeln!(text, "\\emergencystretch=2em").unwrap();
    writeln!(text, "\\maketitle").unwrap();
    write_summary_table(&mut text, summary);
    write_gate_table(&mut text, summary);
    if let Some(direct) = &summary.direct_basis {
        write_direct_basis_table(&mut text, direct);
    }
    if let Some(adapter) = &summary.adapter_bank_conditioning {
        write_adapter_bank_table(&mut text, adapter);
    }
    if let Some(psnr) = &summary.oracle_dynamics_psnr {
        write_psnr_table(&mut text, psnr);
    }
    write_list_section(&mut text, "Interpretation", &summary.interpretation);
    write_list_section(&mut text, "Next Steps", &summary.next_steps);
    writeln!(text, "\\end{{document}}").unwrap();
    text
}

fn write_summary_table(text: &mut String, summary: &Hyper2dValidationSummary) {
    writeln!(text, "\\section{{Summary}}").unwrap();
    begin_table(text);
    row(text, "Report kind", summary.report_kind.label());
    row_path(text, "Source report", &summary.source_report);
    row_path_opt(text, "PSNR report", summary.psnr_report.as_deref());
    row_path_opt(text, "Experiment", summary.experiment_config.as_deref());
    row(text, "Preset", &display_opt_str(summary.preset.as_deref()));
    row(
        text,
        "Train examples",
        &display_opt_usize(summary.train_examples),
    );
    row(
        text,
        "Holdout examples",
        &display_opt_usize(summary.holdout_examples),
    );
    row(text, "Evaluated system", summary.evaluated_system);
    row(text, "Quality status", summary.quality_status);
    row(text, "Quality ready", &summary.quality_ready.to_string());
    row(
        text,
        "Hypernet generalization validated",
        &summary.hypernet_generalization_validated.to_string(),
    );
    end_table(text);
}

fn write_gate_table(text: &mut String, summary: &Hyper2dValidationSummary) {
    writeln!(text, "\\section{{Quality Gates}}").unwrap();
    begin_table(text);
    row(
        text,
        "Direct-basis oracle max ratio",
        &format!(
            "at most {:.2}x",
            summary.quality_gates.direct_oracle_ready_max_ratio
        ),
    );
    row(
        text,
        "Direct-basis shared/zero max ratio",
        &format!(
            "at most {:.2}x",
            summary.quality_gates.direct_zero_ready_max_ratio
        ),
    );
    row(
        text,
        "Adapter-vector normalized RMSE",
        &format!(
            "at most {:.2}",
            summary.quality_gates.adapter_vector_ready_max_nrmse
        ),
    );
    row(
        text,
        "Adapter-vector mean cosine",
        &format!(
            "at least {:.2}",
            summary.quality_gates.adapter_vector_ready_min_cosine
        ),
    );
    row(
        text,
        "Adapter rollout max ratio to static LoRA",
        &format!(
            "at most {:.2}x",
            summary.quality_gates.adapter_rollout_ready_max_ratio
        ),
    );
    row(
        text,
        "Adapter rollout max ratio to zero adapter",
        &format!(
            "at most {:.2}x",
            summary.quality_gates.adapter_rollout_zero_ready_max_ratio
        ),
    );
    row(
        text,
        "Oracle render RGB PSNR",
        &format!(
            "at least {:.1} dB",
            summary.quality_gates.oracle_render_rgb_psnr_ready_db
        ),
    );
    row(
        text,
        "Quality rollout particles",
        &format!(
            "at least {}",
            summary.quality_gates.min_quality_rollout_particles
        ),
    );
    row(
        text,
        "Quality target samples",
        &format!(
            "at least {}",
            summary.quality_gates.min_quality_target_points
        ),
    );
    row(
        text,
        "Oracle examples per split",
        &format!(
            "at least {}",
            summary.quality_gates.min_quality_oracle_examples_per_split
        ),
    );
    end_table(text);
}

fn write_psnr_table(text: &mut String, psnr: &PsnrValidationSummary) {
    writeln!(text, "\\section{{Oracle Dynamics PSNR}}").unwrap();
    begin_table(text);
    row(text, "Particles", &display_opt_usize(psnr.particle_count));
    row(
        text,
        "Rollout steps",
        &display_usize_list(&psnr.rollout_steps),
    );
    row(
        text,
        "Threshold",
        &format!("{} dB", display_opt_f64(psnr.min_render_rgb_psnr_db)),
    );
    row(text, "Direct passed", &display_opt_bool(psnr.direct_passed));
    row(text, "Hyper passed", &display_opt_bool(psnr.hyper_passed));
    row(
        text,
        "Direct min render RGB PSNR",
        &display_opt_f64(psnr.direct_min_render_rgb_psnr_db),
    );
    row(
        text,
        "Hyper min render RGB PSNR",
        &display_opt_f64(psnr.hyper_min_render_rgb_psnr_db),
    );
    for item in &psnr.summaries {
        row(text, &psnr_row_label(item), &psnr_row_summary(item));
    }
    end_table(text);
}

fn write_direct_basis_table(text: &mut String, direct: &DirectBasisReportSummary) {
    writeln!(text, "\\section{{Direct-Basis Metrics}}").unwrap();
    begin_table(text);
    row(text, "Backend", &display_opt_str(direct.backend.as_deref()));
    row(
        text,
        "Adapter rank",
        &display_opt_usize(direct.adapter_rank),
    );
    row(
        text,
        "Rollout particles",
        &display_opt_usize(direct.rollout_particles),
    );
    row(
        text,
        "Rollout steps",
        &display_opt_usize(direct.rollout_steps),
    );
    row(
        text,
        "Min target samples",
        &display_opt_usize(direct.min_target_points),
    );
    row(
        text,
        "Target loss image size",
        &display_opt_usize(direct.target_loss_image_size),
    );
    row(
        text,
        "Train loss initial",
        &display_opt_f64(direct.train_loss_initial),
    );
    row(
        text,
        "Train loss final",
        &display_opt_f64(direct.train_loss_final),
    );
    row(
        text,
        "Train loss reduction",
        &display_opt_percent(direct.train_loss_reduction_fraction),
    );
    row(
        text,
        "Holdout loss initial",
        &display_opt_f64(direct.holdout_loss_initial),
    );
    row(
        text,
        "Holdout loss final",
        &display_opt_f64(direct.holdout_loss_final),
    );
    row(
        text,
        "Train oracle mean ratio",
        &display_opt_f64(
            direct
                .oracle_train
                .and_then(|split| split.mean_ratio_to_oracle),
        ),
    );
    row(
        text,
        "Train oracle max ratio",
        &display_opt_f64(
            direct
                .oracle_train
                .and_then(|split| split.max_ratio_to_oracle),
        ),
    );
    row(
        text,
        "Train shared/zero mean ratio",
        &display_opt_f64(
            direct
                .oracle_train
                .and_then(|split| split.mean_ratio_to_zero),
        ),
    );
    row(
        text,
        "Train shared/zero max ratio",
        &display_opt_f64(
            direct
                .oracle_train
                .and_then(|split| split.max_ratio_to_zero),
        ),
    );
    row(
        text,
        "Holdout oracle mean ratio",
        &display_opt_f64(
            direct
                .oracle_holdout
                .and_then(|split| split.mean_ratio_to_oracle),
        ),
    );
    row(
        text,
        "Holdout oracle max ratio",
        &display_opt_f64(
            direct
                .oracle_holdout
                .and_then(|split| split.max_ratio_to_oracle),
        ),
    );
    row(
        text,
        "Holdout shared/zero mean ratio",
        &display_opt_f64(
            direct
                .oracle_holdout
                .and_then(|split| split.mean_ratio_to_zero),
        ),
    );
    row(
        text,
        "Holdout shared/zero max ratio",
        &display_opt_f64(
            direct
                .oracle_holdout
                .and_then(|split| split.max_ratio_to_zero),
        ),
    );
    row(
        text,
        "Particle steps/sec",
        &throughput_summary(direct.particle_steps_per_sec),
    );
    end_table(text);
}

fn write_adapter_bank_table(text: &mut String, adapter: &AdapterBankConditioningSummary) {
    writeln!(text, "\\section{{Adapter-Bank Conditioning Metrics}}").unwrap();
    begin_table(text);
    row(
        text,
        "Backend",
        &display_opt_str(adapter.backend.as_deref()),
    );
    row(
        text,
        "Training backend",
        &display_opt_str(adapter.training_backend.as_deref()),
    );
    row(
        text,
        "Condition encoder",
        &display_opt_str(adapter.condition_encoder.as_deref()),
    );
    row(
        text,
        "Adapter target canonicalization",
        &display_opt_str(adapter.adapter_target_canonicalization.as_deref()),
    );
    row(
        text,
        "Adapter rank",
        &display_opt_usize(adapter.adapter_rank),
    );
    row(
        text,
        "Adapter parameters",
        &display_opt_usize(adapter.adapter_parameter_count),
    );
    row(
        text,
        "Rollout particles",
        &display_opt_usize(adapter.rollout_particles),
    );
    row(
        text,
        "Rollout steps",
        &display_opt_usize(adapter.rollout_steps),
    );
    row(
        text,
        "Target samples",
        &display_opt_usize(adapter.target_points),
    );
    row(
        text,
        "Target loss image size",
        &display_opt_usize(adapter.target_loss_image_size),
    );
    row(
        text,
        "Train loss initial",
        &display_opt_f64(adapter.train_loss_initial),
    );
    row(
        text,
        "Train loss final",
        &display_opt_f64(adapter.train_loss_final),
    );
    row(
        text,
        "Train loss reduction",
        &display_opt_percent(adapter.train_loss_reduction_fraction),
    );
    row(
        text,
        "Train vector",
        &vector_summary(adapter.train_vector_metrics),
    );
    row(
        text,
        "Holdout vector",
        &vector_summary(adapter.holdout_vector_metrics),
    );
    row(
        text,
        "Train rollout",
        &rollout_summary(adapter.train_rollout),
    );
    row(
        text,
        "Holdout rollout",
        &rollout_summary(adapter.holdout_rollout),
    );
    row(
        text,
        "Adapter values/sec",
        &throughput_summary(adapter.adapter_values_per_sec),
    );
    end_table(text);
}

fn write_list_section(text: &mut String, title: &str, entries: &[String]) {
    writeln!(text, "\\section{{{}}}", latex_escape(title)).unwrap();
    writeln!(text, "\\begin{{itemize}}").unwrap();
    for entry in entries {
        writeln!(text, "\\item {}", latex_escape(entry)).unwrap();
    }
    writeln!(text, "\\end{{itemize}}").unwrap();
}

fn begin_table(text: &mut String) {
    writeln!(
        text,
        "\\begin{{tabularx}}{{\\linewidth}}{{@{{}}p{{0.30\\linewidth}}X@{{}}}}"
    )
    .unwrap();
    writeln!(text, "\\toprule").unwrap();
    writeln!(text, "Field & Value \\\\").unwrap();
    writeln!(text, "\\midrule").unwrap();
}

fn row(text: &mut String, key: &str, value: &str) {
    writeln!(text, "{} & {} \\\\", latex_escape(key), latex_escape(value)).unwrap();
}

fn row_path(text: &mut String, key: &str, value: &str) {
    writeln!(text, "{} & {} \\\\", latex_escape(key), latex_path(value)).unwrap();
}

fn row_path_opt(text: &mut String, key: &str, value: Option<&str>) {
    if let Some(value) = value {
        row_path(text, key, value);
    } else {
        row(text, key, "n/a");
    }
}

fn end_table(text: &mut String) {
    writeln!(text, "\\bottomrule").unwrap();
    writeln!(text, "\\end{{tabularx}}").unwrap();
}

fn vector_summary(metrics: Option<AdapterVectorMetricSummary>) -> String {
    metrics.map_or_else(
        || "n/a".to_string(),
        |metrics| {
            format!(
                "NRMSE={}, cosine={}, target RMS={}, pred RMS={}",
                display_opt_f64(metrics.normalized_rmse_to_target_rms),
                display_opt_f64(metrics.mean_cosine_similarity),
                display_opt_f64(metrics.target_rms),
                display_opt_f64(metrics.prediction_rms)
            )
        },
    )
}

fn rollout_summary(metrics: Option<AdapterRolloutSplitSummary>) -> String {
    metrics.map_or_else(
        || "n/a".to_string(),
        |metrics| {
            format!(
                "static mean ratio={}, static max ratio={}, zero mean ratio={}, zero max ratio={}, zero loss={}, static loss={}, hyper loss={}",
                display_opt_f64(metrics.mean_ratio_to_static),
                display_opt_f64(metrics.max_ratio_to_static),
                display_opt_f64(metrics.mean_ratio_to_zero),
                display_opt_f64(metrics.max_ratio_to_zero),
                display_opt_f64(metrics.mean_zero_loss),
                display_opt_f64(metrics.mean_static_loss),
                display_opt_f64(metrics.mean_hyper_loss)
            )
        },
    )
}

fn throughput_summary(metrics: Option<ThroughputSummary>) -> String {
    metrics.map_or_else(
        || "n/a".to_string(),
        |metrics| {
            format!(
                "median={}, max={}, samples={}",
                display_opt_f64(Some(metrics.median)),
                display_opt_f64(Some(metrics.max)),
                metrics.samples
            )
        },
    )
}

fn display_opt_str(value: Option<&str>) -> String {
    value.map_or_else(|| "n/a".to_string(), ToOwned::to_owned)
}

fn display_opt_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "n/a".to_string(), |value| value.to_string())
}

fn display_opt_f64(value: Option<f64>) -> String {
    value.map_or_else(|| "n/a".to_string(), |value| format!("{value:.6}"))
}

fn display_opt_percent(value: Option<f64>) -> String {
    value.map_or_else(
        || "n/a".to_string(),
        |value| format!("{:.2}%", value * 100.0),
    )
}

fn display_opt_bool(value: Option<bool>) -> String {
    value.map_or_else(|| "n/a".to_string(), |value| value.to_string())
}

fn display_usize_list(values: &[usize]) -> String {
    if values.is_empty() {
        "n/a".to_string()
    } else {
        values
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn psnr_row_label(metrics: &PsnrKindStepSummary) -> String {
    format!(
        "PSNR {} step {}",
        metrics.kind,
        display_opt_usize(metrics.rollout_steps)
    )
}

fn psnr_row_summary(metrics: &PsnrKindStepSummary) -> String {
    format!(
        "examples={}, mean={}, median={}, min={}, max={}, passed={}",
        display_opt_usize(metrics.examples),
        display_opt_f64(metrics.mean_render_rgb_psnr_db),
        display_opt_f64(metrics.median_render_rgb_psnr_db),
        display_opt_f64(metrics.min_render_rgb_psnr_db),
        display_opt_f64(metrics.max_render_rgb_psnr_db),
        display_opt_bool(metrics.passed)
    )
}

fn latex_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '&' => escaped.push_str("\\&"),
            '%' => escaped.push_str("\\%"),
            '$' => escaped.push_str("\\$"),
            '#' => escaped.push_str("\\#"),
            '_' => escaped.push_str("\\_"),
            '{' => escaped.push_str("\\{"),
            '}' => escaped.push_str("\\}"),
            '~' => escaped.push_str("\\textasciitilde{}"),
            '^' => escaped.push_str("\\textasciicircum{}"),
            '\\' => escaped.push_str("\\textbackslash{}"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn latex_path(value: &str) -> String {
    for delimiter in ['|', '!', '+', ':'] {
        if !value.contains(delimiter) {
            return format!("\\path{delimiter}{value}{delimiter}");
        }
    }
    latex_escape(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latex_escape_handles_report_paths_and_thresholds() {
        assert_eq!(latex_escape("a_b/c&d% <= 1.20x"), "a\\_b/c\\&d\\% <= 1.20x");
        assert_eq!(latex_path("a_b/c"), "\\path|a_b/c|");
    }
}
