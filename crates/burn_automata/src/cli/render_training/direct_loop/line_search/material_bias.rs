use super::*;

pub(super) const DIRECT_LINE_SEARCH_KIND_MATERIAL_BIAS: &str = "material-opacity-bias";
pub(super) const DIRECT_LINE_SEARCH_KIND_SGD_MATERIAL_BIAS: &str =
    "sgd-scale-material-opacity-bias";

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_material_opacity_bias_line_search_candidate(
    candidate_kind: &'static str,
    base_model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: &RenderProxyTrainingConfig,
    gradient: &RenderProxyGradientRows,
    render_cfg: RenderLossConfig,
    selection_baseline: &[RenderSelectionBaselineCase],
    source_scale: f32,
    material_opacity_bias: f32,
) -> Result<Option<DirectLineSearchCandidate>, Box<dyn std::error::Error>> {
    if !material_opacity_bias.is_finite() || material_opacity_bias <= 0.0 {
        return Ok(None);
    }
    let mut candidate = base_model.clone();
    if add_growth_3d_material_opacity_update_bias(&mut candidate, material_opacity_bias).is_err() {
        return Ok(None);
    }
    let selection = render_selection_metrics(
        &candidate,
        grid,
        target,
        cfg,
        render_cfg,
        Some(selection_baseline),
    )?;
    let report = render_direct_rollout_noop_report(selection.render_loss, gradient);
    let candidate_report = direct_line_search_candidate_report(
        candidate_kind,
        source_scale,
        material_opacity_bias,
        &selection,
        &report,
    );
    Ok(Some(DirectLineSearchCandidate {
        model: candidate,
        report,
        selection,
        candidate_report,
    }))
}

pub(crate) fn material_opacity_bias_line_search_candidates(
    selection: &RenderSelectionMetrics,
    cfg: &RenderProxyTrainingConfig,
) -> Vec<f32> {
    const BIAS_FRACTIONS: [f32; 3] = [0.25, 0.5, 1.0];
    const MAX_ABSOLUTE_BIAS: f32 = 0.25;
    const MATERIAL_CAP_FRACTION: f32 = 0.25;
    if selection.strict_surface_active_count == 0
        || selection.strict_surface_materialized_fraction >= 1.0
        || !selection.strict_surface_material_visible_margin.is_finite()
        || selection.strict_surface_material_visible_margin <= 0.0
        || cfg.rollout_steps == 0
    {
        return Vec::new();
    }
    let cap =
        if cfg.material_max_opacity_update.is_finite() && cfg.material_max_opacity_update > 0.0 {
            (cfg.material_max_opacity_update * MATERIAL_CAP_FRACTION).min(MAX_ABSOLUTE_BIAS)
        } else {
            MAX_ABSOLUTE_BIAS
        };
    if cap <= 0.0 {
        return Vec::new();
    }
    let base = (selection.strict_surface_material_visible_margin / cfg.rollout_steps as f32)
        .clamp(1.0e-5, cap);
    let mut candidates = Vec::new();
    for fraction in BIAS_FRACTIONS {
        let bias = (base * fraction).clamp(1.0e-5, cap);
        if bias.is_finite()
            && bias > 0.0
            && !candidates.iter().any(|existing: &f32| {
                ((*existing - bias).abs() / existing.abs().max(bias.abs()).max(1.0e-6)) < 1.0e-5
            })
        {
            candidates.push(bias);
        }
    }
    candidates
}
