use super::prelude::*;

pub(crate) const DIRECT_LOCAL_FRONT_EXPANSION_GAIN_FRACTION: f32 = 0.20;
const TEMPORAL_FRONT_CANDIDATE_ROW_FRACTION: usize = 4;
const TEMPORAL_FRONT_CANDIDATE_CAP_ROW_FRACTION: usize = 8;
const TEMPORAL_FRONT_CANDIDATE_MIN_CAP: usize = 16;
const TEMPORAL_FRONT_CANDIDATE_MAX_CAP: usize = 512;
const TEMPORAL_NONLOCAL_LIVENESS_SUPPRESSION_GAIN_FRACTION: f32 = 0.35;

mod adjoints;
mod direct_loop;
mod geometry_updates;
mod gradients;
mod objective_diagnostics;
mod output_objectives;
mod selection;

pub(crate) use adjoints::*;
pub(crate) use direct_loop::*;
pub(crate) use geometry_updates::*;
pub(crate) use gradients::*;
pub(crate) use objective_diagnostics::*;
pub(crate) use output_objectives::*;
pub(crate) use selection::*;

#[derive(Clone)]
pub(crate) struct RenderLossEvalConfig {
    pub(crate) particle_count: usize,
    pub(crate) steps: usize,
    pub(crate) seed: u64,
    pub(crate) extra_seeds: Vec<u64>,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) render: RenderLossConfig,
}

pub(crate) fn default_render_loss_config(seed_scale: f32) -> RenderLossConfig {
    RenderLossConfig {
        world_scale: seed_scale.max(1.0e-4) * 2.0,
        target_samples: 0,
        ..RenderLossConfig::default()
    }
}

pub(crate) fn mesh_render_loss_for_model(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: RenderLossEvalConfig,
) -> Result<MultiViewRenderLossReport, Box<dyn std::error::Error>> {
    let mut render_cfg = cfg.render;
    if render_cfg.target_samples == 0 {
        render_cfg.target_samples = cfg.particle_count;
    }
    let seeds = eval_seed_list(cfg.seed, &cfg.extra_seeds);
    let mut reports = Vec::with_capacity(seeds.len());
    for seed in seeds {
        let trace = run_rollout(
            model,
            grid,
            &RolloutConfig {
                particle_count: cfg.particle_count,
                steps: cfg.steps,
                update_prob: 1.0,
                seed,
                seed_scale: cfg.seed_scale,
                ..RolloutConfig::default()
            },
            cfg.seed_mode,
        )?;
        reports.push(mesh_multiview_render_loss_from_trace(
            &trace, target, render_cfg,
        )?);
    }
    Ok(average_render_loss_reports(reports, render_cfg))
}

pub(crate) fn eval_seed_list(seed: u64, extra_seeds: &[u64]) -> Vec<u64> {
    let mut seeds = Vec::with_capacity(extra_seeds.len() + 1);
    seeds.push(seed);
    for &extra_seed in extra_seeds {
        if !seeds.contains(&extra_seed) {
            seeds.push(extra_seed);
        }
    }
    seeds
}

pub(crate) fn average_render_loss_reports(
    reports: Vec<MultiViewRenderLossReport>,
    cfg: RenderLossConfig,
) -> MultiViewRenderLossReport {
    let count = reports.len().max(1) as f32;
    let first = reports
        .first()
        .cloned()
        .unwrap_or_else(|| empty_render_loss_report(cfg));
    if reports.len() <= 1 {
        return first;
    }

    let views = (0..first.views.len())
        .map(|view_idx| {
            let view = first.views[view_idx].view;
            let view_reports: Vec<&RenderViewLossReport> = reports
                .iter()
                .filter_map(|report| report.views.get(view_idx))
                .collect();
            let view_count = view_reports.len().max(1) as f32;
            let density_mse = view_reports
                .iter()
                .map(|report| report.density_mse)
                .sum::<f32>()
                / view_count;
            let color_mse = view_reports
                .iter()
                .map(|report| report.color_mse)
                .sum::<f32>()
                / view_count;
            let depth_mse = view_reports
                .iter()
                .map(|report| report.depth_mse)
                .sum::<f32>()
                / view_count;
            let nonzero_target_alpha_fraction = view_reports
                .iter()
                .map(|report| report.nonzero_target_alpha_fraction)
                .sum::<f32>()
                / view_count;
            let nonzero_particle_alpha_fraction = view_reports
                .iter()
                .map(|report| report.nonzero_particle_alpha_fraction)
                .sum::<f32>()
                / view_count;
            RenderViewLossReport {
                view,
                total_loss: cfg.density_weight * density_mse
                    + cfg.color_weight * color_mse
                    + cfg.depth_weight * depth_mse,
                density_mse,
                color_mse,
                depth_mse,
                density_psnr_db: render_psnr_db(density_mse, 1.0),
                color_psnr_db: render_psnr_db(color_mse, 1.0),
                depth_psnr_db: render_psnr_db(depth_mse, 1.0),
                nonzero_target_alpha_fraction,
                nonzero_particle_alpha_fraction,
            }
        })
        .collect::<Vec<_>>();

    let density_mse = reports.iter().map(|report| report.density_mse).sum::<f32>() / count;
    let color_mse = reports.iter().map(|report| report.color_mse).sum::<f32>() / count;
    let depth_mse = reports.iter().map(|report| report.depth_mse).sum::<f32>() / count;
    let density_psnr_db = render_psnr_db(density_mse, 1.0);
    let color_psnr_db = render_psnr_db(color_mse, 1.0);
    let depth_psnr_db = render_psnr_db(depth_mse, 1.0);
    let nonzero_target_alpha_fraction = reports
        .iter()
        .map(|report| report.nonzero_target_alpha_fraction)
        .sum::<f32>()
        / count;
    let nonzero_particle_alpha_fraction = reports
        .iter()
        .map(|report| report.nonzero_particle_alpha_fraction)
        .sum::<f32>()
        / count;
    let finite = reports.iter().all(|report| {
        report.total_loss.is_finite()
            && report.density_mse.is_finite()
            && report.color_mse.is_finite()
            && report.depth_mse.is_finite()
            && report.nonzero_particle_alpha_fraction > 0.0
    });
    MultiViewRenderLossReport {
        passed: finite
            && reports.iter().all(|report| report.passed)
            && density_psnr_db >= 10.0
            && color_psnr_db >= 12.0
            && depth_psnr_db >= 14.0,
        image_size: cfg.image_size,
        target_samples: cfg.target_samples,
        total_loss: cfg.density_weight * density_mse
            + cfg.color_weight * color_mse
            + cfg.depth_weight * depth_mse,
        density_mse,
        color_mse,
        depth_mse,
        density_psnr_db,
        color_psnr_db,
        depth_psnr_db,
        nonzero_target_alpha_fraction,
        nonzero_particle_alpha_fraction,
        views,
    }
}

pub(crate) fn render_psnr_db(mse: f32, max_value: f32) -> f32 {
    if mse <= 0.0 {
        99.0
    } else {
        10.0 * ((max_value * max_value) / mse.max(1.0e-8)).log10()
    }
}

pub(crate) fn empty_render_loss_report(cfg: RenderLossConfig) -> MultiViewRenderLossReport {
    MultiViewRenderLossReport {
        passed: false,
        image_size: cfg.image_size,
        target_samples: cfg.target_samples,
        total_loss: f32::INFINITY,
        density_mse: f32::INFINITY,
        color_mse: f32::INFINITY,
        depth_mse: f32::INFINITY,
        density_psnr_db: f32::NEG_INFINITY,
        color_psnr_db: f32::NEG_INFINITY,
        depth_psnr_db: f32::NEG_INFINITY,
        nonzero_target_alpha_fraction: 0.0,
        nonzero_particle_alpha_fraction: 0.0,
        views: Vec::new(),
    }
}

#[derive(Clone)]
pub(crate) struct RenderProxyTrainingConfig {
    pub(crate) target: MeshTargetArg,
    pub(crate) rounds: usize,
    pub(crate) supervised_steps_per_round: usize,
    pub(crate) particles: usize,
    pub(crate) rollout_steps: usize,
    pub(crate) gradient_particles: usize,
    pub(crate) gradient_mode: RenderGradientModeArg,
    pub(crate) finite_diff_eps: f32,
    pub(crate) motion_gain: f32,
    pub(crate) perception_position_gain: f32,
    pub(crate) max_update_norm: f32,
    pub(crate) trajectory_supervision: bool,
    pub(crate) trajectory_render_gain: f32,
    pub(crate) trajectory_mesh_gain: f32,
    pub(crate) trajectory_render_samples: usize,
    pub(crate) liveness_gain: f32,
    pub(crate) liveness_front_radius: f32,
    pub(crate) liveness_update_multiplier: f32,
    pub(crate) coverage_gain: f32,
    pub(crate) coverage_samples: usize,
    pub(crate) coverage_mode: CoverageUpdateModeArg,
    pub(crate) coverage_softness: f32,
    pub(crate) coverage_repulsion_gain: f32,
    pub(crate) coverage_gap_gain: f32,
    pub(crate) coverage_repulsion_radius: f32,
    pub(crate) coverage_normal_weight: f32,
    pub(crate) extent_gain: f32,
    pub(crate) full_coverage_adjoint: bool,
    pub(crate) surface_gain: f32,
    pub(crate) surface_escape_gain: f32,
    pub(crate) opacity_gain: f32,
    pub(crate) material_liveness_gain: f32,
    pub(crate) material_tail_gain: f32,
    pub(crate) material_suppression_update_multiplier: f32,
    pub(crate) material_max_opacity_update: f32,
    pub(crate) scale_gain: f32,
    pub(crate) scale_budget_weight: f32,
    pub(crate) max_opacity_update: f32,
    pub(crate) direct_output_gradient_rms_cap: f32,
    pub(crate) direct_line_search: bool,
    pub(crate) direct_line_search_scales: Vec<f32>,
    pub(crate) direct_material_output_only: bool,
    pub(crate) training_backend: RenderTrainingBackendArg,
    pub(crate) direct_selection_seed_training: bool,
    pub(crate) seed: u64,
    pub(crate) selection_seed: Option<u64>,
    pub(crate) selection_seeds: Vec<u64>,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) render: RenderLossConfig,
    pub(crate) sgd: SgdConfig,
}

pub(crate) fn render_training_default_seed_mode(target: MeshTargetArg) -> ParticleSeed {
    conditionless_local_seed_mode(target)
}

pub(crate) fn render_training_base_model(
    target: MeshTargetArg,
    target_mesh: &TriangleMeshTarget,
    seed_mode: ParticleSeed,
) -> Result<(NpaModel, String), Box<dyn std::error::Error>> {
    if !target_local_growth_seed(target, seed_mode) {
        return Err(std::io::Error::other(format!(
            "default render training base requires a target local growth seed; got seed_mode={seed_mode:?}"
        ))
        .into());
    }
    let model = local_growth_student_model_with_axis_gains(
        NpaConfig::growing_3dgs(),
        0x005a_173d,
        0.0,
        mesh_axis_expansion_gains(target_mesh, LOCAL_GROWTH_EXPANSION_GAIN),
    )?;
    let source = format!(
        "ablation-rust:{}",
        mesh_conditionless_local_target_source_for_seed(target, seed_mode)
    );
    Ok((model, source))
}

pub(crate) fn render_training_seed_mode(target: MeshTargetArg) -> ParticleSeed {
    match target {
        MeshTargetArg::Torus => ParticleSeed::TorusFieldDense3d,
        MeshTargetArg::Teapot => ParticleSeed::TeapotFieldDense3d,
    }
}

pub(crate) fn default_render_training_seed_mode(
    target: MeshTargetArg,
    model: &NpaModel,
) -> ParticleSeed {
    if model.config.position_features {
        render_training_seed_mode(target)
    } else {
        conditionless_local_seed_mode(target)
    }
}

pub(crate) fn render_proxy_selection_seeds(cfg: &RenderProxyTrainingConfig) -> Vec<u64> {
    let mut seeds = Vec::with_capacity(cfg.selection_seeds.len() + 2);
    seeds.push(cfg.seed);
    if let Some(selection_seed) = cfg.selection_seed
        && !seeds.contains(&selection_seed)
    {
        seeds.push(selection_seed);
    }
    for &selection_seed in &cfg.selection_seeds {
        if !seeds.contains(&selection_seed) {
            seeds.push(selection_seed);
        }
    }
    seeds
}

pub(crate) fn render_training_validation_extra_seeds(
    selection_seed: u64,
    extra_selection_seeds: &[u64],
) -> Vec<u64> {
    let mut seeds = Vec::with_capacity(extra_selection_seeds.len() + 1);
    seeds.push(selection_seed);
    for &extra_seed in extra_selection_seeds {
        if !seeds.contains(&extra_seed) {
            seeds.push(extra_seed);
        }
    }
    seeds
}

pub(crate) fn catalog_promotion_validation_extra_seeds(
    selection_seed: u64,
    extra_selection_seeds: &[u64],
) -> Vec<u64> {
    let mut seeds =
        Vec::with_capacity(CATALOG_3D_HELD_OUT_SEEDS.len() + 1 + extra_selection_seeds.len());
    for seed in CATALOG_3D_HELD_OUT_SEEDS {
        push_catalog_extra_seed(&mut seeds, seed);
    }
    push_catalog_extra_seed(&mut seeds, selection_seed);
    for &extra_seed in extra_selection_seeds {
        push_catalog_extra_seed(&mut seeds, extra_seed);
    }
    seeds
}

pub(crate) fn push_catalog_extra_seed(seeds: &mut Vec<u64>, seed: u64) {
    if seed != CATALOG_3D_APP_EVAL_SEED && !seeds.contains(&seed) {
        seeds.push(seed);
    }
}

pub(crate) fn catalog_promotion_render_config(mut render: RenderLossConfig) -> RenderLossConfig {
    render.image_size = render.image_size.max(CATALOG_3D_VALIDATION_IMAGE_SIZE);
    render.target_samples = render
        .target_samples
        .max(CATALOG_3D_VALIDATION_TARGET_SAMPLES);
    render
}

pub(crate) fn catalog_promotion_validation_configs(
    selection_seed: u64,
    extra_selection_seeds: &[u64],
    seed_scale: f32,
    seed_mode: ParticleSeed,
    render: RenderLossConfig,
) -> Vec<Growth3dValidationConfig> {
    let extra_seeds =
        catalog_promotion_validation_extra_seeds(selection_seed, extra_selection_seeds);
    let render = catalog_promotion_render_config(render);
    CATALOG_3D_PROMOTION_STEPS
        .into_iter()
        .map(|steps| Growth3dValidationConfig {
            particle_count: CATALOG_3D_VALIDATION_PARTICLES,
            steps,
            seed: CATALOG_3D_APP_EVAL_SEED,
            extra_seeds: extra_seeds.clone(),
            seed_scale,
            seed_mode,
            gate: Growth3dValidationGateArg::Strict,
            render,
        })
        .collect()
}

pub(crate) fn require_catalog_promotion_validations_pass(
    validations: &[CliGrowth3dValidationReport],
    model_output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut failures = Vec::new();
    for validation in validations {
        if !growth_3d_fail_on_validation_passed(validation) {
            failures.push(format!(
                "{}p/{}s score={:.6} failures={:?}",
                validation.particle_count,
                validation.steps,
                validation.strict_score.score,
                validation.strict_checks.failure_reasons
            ));
        }
    }
    if failures.is_empty() {
        return Ok(());
    }
    Err(std::io::Error::other(format!(
        "catalog-bound 3D render training candidate failed app-scale strict growth validation ({}); refusing to overwrite {}",
        failures.join("; "),
        model_output.display()
    ))
    .into())
}

pub(crate) fn render_training_source(
    target: MeshTargetArg,
    base_source: Option<&str>,
    seed_mode: ParticleSeed,
) -> String {
    let local_growth_seed = matches!(
        seed_mode,
        ParticleSeed::TorusGrowth3d
            | ParticleSeed::TeapotGrowth3d
            | ParticleSeed::TorusSubstrateGrowth3d
            | ParticleSeed::TeapotSubstrateGrowth3d
            | ParticleSeed::TorusLocalGrowth3d
            | ParticleSeed::TeapotLocalGrowth3d
            | ParticleSeed::TorusLocalSubstrateGrowth3d
            | ParticleSeed::TeapotLocalSubstrateGrowth3d
    );
    if let Some(source) = base_source {
        if source.starts_with("render-refined-rust:") && local_growth_seed {
            return source.to_string();
        }
        if source.contains("conditionless-local") && local_growth_seed {
            return format!("render-refined-rust:{source}");
        }
        return format!("render-proxy-rust:{target:?}:base={source}:seed={seed_mode:?}");
    }
    format!("render-proxy-rust:{target:?}:field-baseline")
}

pub(crate) fn is_catalog_model_output_path(path: &Path) -> bool {
    let parts = path
        .components()
        .filter_map(|component| match component {
            std::path::Component::Normal(part) => part.to_str(),
            _ => None,
        })
        .collect::<Vec<_>>();
    parts
        .windows(2)
        .any(|window| window[0] == "assets" && window[1] == "models")
}

pub(crate) fn catalog_bound_candidate_path(target: MeshTargetArg, process_id: u32) -> PathBuf {
    let target_label = match target {
        MeshTargetArg::Torus => "torus",
        MeshTargetArg::Teapot => "teapot",
    };
    PathBuf::from("target").join(format!(
        "catalog_{target_label}_render3d_candidate_{process_id}.bpk"
    ))
}

pub(crate) fn target_local_growth_seed(target: MeshTargetArg, seed_mode: ParticleSeed) -> bool {
    matches!(
        (target, seed_mode),
        (MeshTargetArg::Torus, ParticleSeed::TorusGrowth3d)
            | (MeshTargetArg::Teapot, ParticleSeed::TeapotGrowth3d)
            | (MeshTargetArg::Torus, ParticleSeed::TorusSubstrateGrowth3d)
            | (MeshTargetArg::Teapot, ParticleSeed::TeapotSubstrateGrowth3d)
            | (MeshTargetArg::Torus, ParticleSeed::TorusLocalGrowth3d)
            | (MeshTargetArg::Teapot, ParticleSeed::TeapotLocalGrowth3d)
            | (
                MeshTargetArg::Torus,
                ParticleSeed::TorusLocalSubstrateGrowth3d
            )
            | (
                MeshTargetArg::Teapot,
                ParticleSeed::TeapotLocalSubstrateGrowth3d
            )
    )
}

pub(crate) fn validate_diagnostic_3d_output_not_catalog(
    model_output: &Path,
    command: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if is_catalog_model_output_path(model_output) {
        return Err(std::io::Error::other(format!(
            "{command} writes diagnostic 3D artifacts and refuses catalog-bound output {}; write to target/ or artifacts/ and promote only after validate_3d_catalog.py passes",
            model_output.display()
        ))
        .into());
    }
    Ok(())
}

pub(crate) fn validate_catalog_bound_render_training_output(
    model_output: &Path,
    target: MeshTargetArg,
    seed_mode: ParticleSeed,
    base_source: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    if !is_catalog_model_output_path(model_output) {
        return Ok(());
    }
    if !target_local_growth_seed(target, seed_mode) {
        return Err(std::io::Error::other(format!(
            "catalog-bound 3D render training output {} requires the target local growth seed; got seed_mode={seed_mode:?}",
            model_output.display()
        ))
        .into());
    }
    let source = base_source.unwrap_or_default();
    if !local_conditionless_lineage(source) {
        return Err(std::io::Error::other(format!(
            "catalog-bound 3D render training output {} requires a conditionless-local base model; source={source:?}",
            model_output.display()
        ))
        .into());
    }
    Ok(())
}

pub(crate) fn local_conditionless_lineage(source: &str) -> bool {
    source.contains("conditionless-local")
        && !source.contains("position-field")
        && !source.contains("seed-frame")
        && !source.contains("render-proxy-rust")
}

pub(crate) fn load_conditionless_local_base_model(
    path: &Path,
    target_source: &str,
) -> Result<(NpaModel, crate::kernels::HashGridConfig, String), Box<dyn std::error::Error>> {
    let manifest = crate::import::load_manifest(path)?;
    if manifest.config.spatial_dims != 3 || manifest.config.state_dims <= 3 {
        return Err(std::io::Error::other(format!(
            "local 3D continuation requires spatial_dims=3 and state_dims>3; got spatial_dims={} state_dims={}",
            manifest.config.spatial_dims, manifest.config.state_dims
        ))
        .into());
    }
    if manifest.config.position_features {
        return Err(std::io::Error::other(format!(
            "local 3D continuation rejects position-feature base model {}",
            path.display()
        ))
        .into());
    }
    let source_text = manifest.source.as_deref().unwrap_or_default();
    if !local_conditionless_lineage(source_text) {
        return Err(std::io::Error::other(format!(
            "local 3D continuation rejects shortcut lineage for {}: source={source_text:?}",
            path.display()
        ))
        .into());
    }
    let source = format!("ablation-rust:{target_source}:continued-from={source_text}");
    let hashgrid = manifest.hashgrid.clone();
    Ok((manifest.into_model(), hashgrid, source))
}

pub(crate) fn run_render_proxy_training(
    model: &mut NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: RenderProxyTrainingConfig,
) -> Result<RenderProxyTrainingReport, Box<dyn std::error::Error>> {
    if cfg.rounds == 0 || cfg.supervised_steps_per_round == 0 {
        return Err(std::io::Error::other(
            "render-proxy training requires non-zero rounds and supervised steps",
        )
        .into());
    }
    if !cfg.finite_diff_eps.is_finite() || cfg.finite_diff_eps <= 0.0 {
        return Err(std::io::Error::other("finite_diff_eps must be positive and finite").into());
    }
    let mut render_cfg = cfg.render;
    if render_cfg.target_samples == 0 {
        render_cfg.target_samples = cfg.particles;
    }
    let initial_trace = render_training_trace(model, grid, &cfg, 0)?;
    let initial_render_loss =
        mesh_multiview_render_loss_from_trace(&initial_trace, target, render_cfg)?;
    let initial_gaussian_volume = gaussian_volume_stats_for_trace(&initial_trace, render_cfg);
    let selection_baseline = render_selection_baseline(model, grid, target, &cfg, render_cfg)?;
    let initial_selection = render_selection_metrics(
        model,
        grid,
        target,
        &cfg,
        render_cfg,
        Some(&selection_baseline),
    )?;
    let mut best_model = model.clone();
    let mut best_render_loss = initial_render_loss.clone();
    let mut best_selection = initial_selection.clone();
    let mut selected_round = None;
    let mut history = Vec::with_capacity(cfg.rounds);

    for round in 0..cfg.rounds {
        let needs_trajectory = cfg.trajectory_supervision
            || cfg.training_backend == RenderTrainingBackendArg::DirectRollout;
        let (trace, trajectory) = if needs_trajectory {
            render_training_trajectory(model, grid, &cfg, round)?
        } else {
            (render_training_trace(model, grid, &cfg, round)?, Vec::new())
        };
        let before = mesh_multiview_render_loss_from_trace(&trace, target, render_cfg)?;
        let before_selection = render_selection_metrics(
            model,
            grid,
            target,
            &cfg,
            render_cfg,
            Some(&selection_baseline),
        )?;
        let gradient = render_position_gradient(&trace, target, render_cfg, &cfg)?;
        let gradient_rms = (gradient
            .gradients
            .iter()
            .map(|g| g[0] * g[0] + g[1] * g[1] + g[2] * g[2])
            .sum::<f32>()
            / gradient.gradients.len().max(1) as f32)
            .sqrt();
        let opacity_gradient_rms = (gradient
            .opacity_gradients
            .iter()
            .map(|gradient| gradient * gradient)
            .sum::<f32>()
            / gradient.opacity_gradients.len().max(1) as f32)
            .sqrt();
        let scale_gradient_rms = (gradient
            .scale_gradients
            .iter()
            .map(|gradient| gradient * gradient)
            .sum::<f32>()
            / gradient.scale_gradients.len().max(1) as f32)
            .sqrt();
        let direct_objective_diagnostics =
            if cfg.training_backend == RenderTrainingBackendArg::DirectRollout {
                direct_rollout_objective_diagnostics(model, target, &trajectory, &cfg)?
            } else {
                DirectRolloutObjectiveDiagnostics::default()
            };
        let before_training_weights = model.weights.clone();
        let (train_report, train_step_scale) = match cfg.training_backend {
            RenderTrainingBackendArg::Proxy => {
                let batch = render_proxy_supervised_batch(
                    model,
                    grid,
                    target,
                    &trace,
                    &trajectory,
                    &gradient,
                    &cfg,
                )?;
                (
                    run_supervised_training(
                        model,
                        &batch,
                        TrainingRunConfig {
                            steps: cfg.supervised_steps_per_round,
                            report_interval: cfg.supervised_steps_per_round,
                            sgd: cfg.sgd,
                        },
                    )?,
                    1.0,
                )
            }
            RenderTrainingBackendArg::DirectRollout => render_direct_rollout_training_steps(
                model,
                grid,
                target,
                &cfg,
                round,
                render_cfg,
                &selection_baseline,
            )?,
        };
        let train_liveness_output_delta_norm = output_channel_delta_norm(
            &before_training_weights,
            &model.weights,
            model.config.hidden_dims,
            model.config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL,
        );
        let train_phase_output_delta_norm = growth_3d_phase_channel(model.config.state_dims)
            .map(|channel| {
                output_channel_delta_norm(
                    &before_training_weights,
                    &model.weights,
                    model.config.hidden_dims,
                    model.config.spatial_dims + channel,
                )
            })
            .unwrap_or(0.0);
        let train_motion_output_delta_norm = spatial_output_delta_norm(
            &before_training_weights,
            &model.weights,
            model.config.hidden_dims,
            model.config.spatial_dims,
        );
        let train_motion_memory_output_delta_norm =
            growth_3d_velocity_channels(model.config.state_dims)
                .map(|channels| {
                    channels
                        .map(|channel| {
                            output_channel_delta_norm(
                                &before_training_weights,
                                &model.weights,
                                model.config.hidden_dims,
                                model.config.spatial_dims + channel,
                            )
                            .powi(2)
                        })
                        .sum::<f32>()
                        .sqrt()
                })
                .unwrap_or(0.0);
        let train_material_output_delta_norm =
            growth_3d_material_opacity_channel(model.config.state_dims)
                .map(|channel| {
                    output_channel_delta_norm(
                        &before_training_weights,
                        &model.weights,
                        model.config.hidden_dims,
                        model.config.spatial_dims + channel,
                    )
                })
                .unwrap_or(0.0);
        let after_trace = render_training_trace(model, grid, &cfg, round)?;
        let after = mesh_multiview_render_loss_from_trace(&after_trace, target, render_cfg)?;
        let selection = render_selection_metrics(
            model,
            grid,
            target,
            &cfg,
            render_cfg,
            Some(&selection_baseline),
        )?;
        let selected_checkpoint =
            render_selection_candidate_metrics_beats(&selection, &best_selection);
        if selected_checkpoint {
            best_model = model.clone();
            best_render_loss = selection.base_report.clone();
            best_selection = selection.clone();
            selected_round = Some(round);
        }
        let continue_training_checkpoint = selected_checkpoint
            || render_selection_training_progress_beats(&selection, &before_selection);
        let rolled_back_to_best_checkpoint = !continue_training_checkpoint;
        let reported_train_step_scale = if rolled_back_to_best_checkpoint {
            0.0
        } else {
            train_step_scale
        };
        history.push(RenderProxyTrainingHistoryEntry {
            round,
            before_loss: before.total_loss,
            after_loss: after.total_loss,
            before_selection_loss: before_selection.render_loss,
            before_selection_score: before_selection.score,
            before_selection_density_psnr_db: before_selection.density_psnr_db,
            before_selection_min_active_extent_bbox_ratio: before_selection
                .min_active_extent_bbox_ratio,
            before_selection_min_active_extent_min_axis_ratio: before_selection
                .min_active_extent_min_axis_ratio,
            selection_loss: selection.render_loss,
            selection_score: selection.score,
            before_density_psnr_db: before.density_psnr_db,
            after_density_psnr_db: after.density_psnr_db,
            selection_density_psnr_db: selection.density_psnr_db,
            selection_active_surface_max: selection.active_surface_max,
            selection_target_coverage_fraction: selection.target_coverage_fraction,
            selection_material_visible_target_mean_distance: selection
                .material_visible_target_mean_distance,
            selection_material_visible_target_max_distance: selection
                .material_visible_target_max_distance,
            selection_material_visible_target_coverage_fraction: selection
                .material_visible_target_coverage_fraction,
            selection_material_visible_inactive_fraction: selection
                .material_visible_inactive_fraction,
            selection_material_visible_max_inactive_opacity: selection
                .material_visible_max_inactive_opacity,
            selection_material_active_mean_opacity: selection.material_active_mean_opacity,
            selection_material_visible_count: selection.material_visible_count,
            selection_surface_covered_bin_fraction: selection.surface_covered_bin_fraction,
            selection_surface_mean_bin_covered_fraction: selection
                .surface_mean_bin_covered_fraction,
            selection_material_visible_surface_covered_bin_fraction: selection
                .material_visible_surface_covered_bin_fraction,
            selection_material_visible_surface_mean_bin_covered_fraction: selection
                .material_visible_surface_mean_bin_covered_fraction,
            selection_surface_normal_covered_bin_fraction: selection
                .surface_normal_covered_bin_fraction,
            selection_surface_normal_mean_bin_covered_fraction: selection
                .surface_normal_mean_bin_covered_fraction,
            selection_material_visible_surface_normal_covered_bin_fraction: selection
                .material_visible_surface_normal_covered_bin_fraction,
            selection_material_visible_surface_normal_mean_bin_covered_fraction: selection
                .material_visible_surface_normal_mean_bin_covered_fraction,
            selection_material_visible_surface_tail_p99_distance: selection
                .material_visible_surface_tail_p99_distance,
            selection_material_visible_surface_tail_over_threshold_fraction: selection
                .material_visible_surface_tail_over_threshold_fraction,
            selection_min_active_extent_bbox_ratio: selection.min_active_extent_bbox_ratio,
            selection_min_active_extent_min_axis_ratio: selection.min_active_extent_min_axis_ratio,
            selection_min_final_active_count: selection.min_final_active_count,
            selection_min_newly_activated_fraction: selection.min_newly_activated_fraction,
            selection_min_front_local_newly_activated_fraction: selection
                .min_front_local_newly_activated_fraction,
            selection_max_front_liveness_margin: selection.max_front_liveness_margin,
            selection_min_front_liveness_candidate_count: selection
                .min_front_liveness_candidate_count,
            selection_max_extent_front_liveness_margin: selection.max_extent_front_liveness_margin,
            selection_min_extent_front_liveness_candidate_count: selection
                .min_extent_front_liveness_candidate_count,
            selection_max_temporal_front_liveness_margin: selection
                .max_temporal_front_liveness_margin,
            selection_min_temporal_front_liveness_candidate_count: selection
                .min_temporal_front_liveness_candidate_count,
            selection_max_temporal_extent_front_liveness_margin: selection
                .max_temporal_extent_front_liveness_margin,
            selection_min_temporal_extent_front_liveness_candidate_count: selection
                .min_temporal_extent_front_liveness_candidate_count,
            selection_max_temporal_activation_schedule_error: selection
                .max_temporal_activation_schedule_error,
            selection_all_temporal_activation_progressive: selection
                .all_temporal_activation_progressive,
            selection_all_temporal_geometry_progressive: selection
                .all_temporal_geometry_progressive,
            selection_morphology_non_regressed: selection.morphology_non_regressed,
            selected_checkpoint,
            rolled_back_to_best_checkpoint,
            selection_worst_seed: selection.worst_seed,
            selection_worst_failure_reasons: selection.worst_failure_reasons,
            before_color_psnr_db: before.color_psnr_db,
            after_color_psnr_db: after.color_psnr_db,
            before_depth_psnr_db: before.depth_psnr_db,
            after_depth_psnr_db: after.depth_psnr_db,
            train_initial_loss: train_report.initial_loss,
            train_final_loss: train_report.final_loss,
            train_best_loss: train_report.best_loss,
            supervised_loss: train_report.final_loss,
            train_step_count: train_report.steps,
            train_loss_history: train_report
                .history
                .iter()
                .map(|entry| entry.loss)
                .collect(),
            train_grad_norm: train_report
                .history
                .last()
                .map(|entry| entry.grad_norm)
                .unwrap_or(0.0),
            train_grad_norm_history: train_report
                .history
                .iter()
                .map(|entry| entry.grad_norm)
                .collect(),
            train_grad_scale: train_report
                .history
                .last()
                .map(|entry| entry.grad_scale)
                .unwrap_or(1.0),
            train_grad_scale_history: train_report
                .history
                .iter()
                .map(|entry| entry.grad_scale)
                .collect(),
            train_step_scale: reported_train_step_scale,
            train_motion_output_delta_norm,
            train_motion_memory_output_delta_norm,
            train_liveness_output_delta_norm,
            train_phase_output_delta_norm,
            train_material_output_delta_norm,
            direct_objective_diagnostics,
            gradient_rms,
            opacity_gradient_rms,
            scale_gradient_rms,
        });
        if rolled_back_to_best_checkpoint {
            *model = best_model.clone();
        }
    }
    let final_render_loss = if selected_round.is_some() {
        *model = best_model;
        best_render_loss
    } else {
        mesh_multiview_render_loss_from_trace(
            &render_training_trace(model, grid, &cfg, 0)?,
            target,
            render_cfg,
        )?
    };
    let final_trace = render_training_trace(model, grid, &cfg, 0)?;
    let final_gaussian_volume = gaussian_volume_stats_for_trace(&final_trace, render_cfg);

    Ok(RenderProxyTrainingReport {
        rounds: cfg.rounds,
        supervised_steps_per_round: cfg.supervised_steps_per_round,
        objective: render_training_objective_config(&cfg, render_cfg),
        gradient_particles: cfg.gradient_particles,
        gradient_mode: cfg.gradient_mode,
        finite_diff_eps: cfg.finite_diff_eps,
        motion_gain: cfg.motion_gain,
        perception_position_gain: cfg.perception_position_gain,
        max_update_norm: cfg.max_update_norm,
        trajectory_supervision: cfg.trajectory_supervision,
        trajectory_render_gain: cfg.trajectory_render_gain,
        trajectory_mesh_gain: cfg.trajectory_mesh_gain,
        trajectory_render_samples: cfg.trajectory_render_samples,
        liveness_gain: cfg.liveness_gain,
        liveness_front_radius: cfg.liveness_front_radius,
        liveness_update_multiplier: cfg.liveness_update_multiplier,
        coverage_gain: cfg.coverage_gain,
        coverage_samples: cfg.coverage_samples,
        coverage_mode: cfg.coverage_mode,
        coverage_softness: cfg.coverage_softness,
        coverage_repulsion_gain: cfg.coverage_repulsion_gain,
        coverage_gap_gain: cfg.coverage_gap_gain,
        coverage_repulsion_radius: cfg.coverage_repulsion_radius,
        coverage_normal_weight: cfg.coverage_normal_weight,
        extent_gain: cfg.extent_gain,
        full_coverage_adjoint: cfg.full_coverage_adjoint,
        surface_gain: cfg.surface_gain,
        surface_escape_gain: cfg.surface_escape_gain,
        opacity_gain: cfg.opacity_gain,
        material_liveness_gain: cfg.material_liveness_gain,
        material_tail_gain: cfg.material_tail_gain,
        material_suppression_update_multiplier: cfg.material_suppression_update_multiplier,
        material_max_opacity_update: cfg.material_max_opacity_update,
        scale_gain: cfg.scale_gain,
        scale_budget_weight: cfg.scale_budget_weight,
        max_opacity_update: cfg.max_opacity_update,
        direct_output_gradient_rms_cap: cfg.direct_output_gradient_rms_cap,
        direct_line_search: cfg.direct_line_search,
        direct_line_search_scales: sanitized_direct_line_search_scales(&cfg),
        direct_material_output_only: cfg.direct_material_output_only,
        training_backend: cfg.training_backend,
        direct_selection_seed_training: cfg.direct_selection_seed_training,
        selection_seed: cfg.selection_seed,
        selection_seeds: render_proxy_selection_seeds(&cfg),
        initial_gaussian_volume,
        final_gaussian_volume,
        initial_render_loss,
        final_render_loss,
        selected_round,
        history,
    })
}
