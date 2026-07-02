use super::*;

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
    mesh_target_training_profile(target).field_seed_mode
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

pub(crate) fn render_training_default_extra_selection_seeds(
    selection_seed: u64,
    user_extra_selection_seeds: &[u64],
) -> Vec<u64> {
    let mut seeds =
        Vec::with_capacity(CATALOG_3D_HELD_OUT_SEEDS.len() + user_extra_selection_seeds.len());
    for heldout_seed in CATALOG_3D_HELD_OUT_SEEDS {
        push_extra_training_selection_seed(&mut seeds, selection_seed, heldout_seed);
    }
    for &extra_seed in user_extra_selection_seeds {
        push_extra_training_selection_seed(&mut seeds, selection_seed, extra_seed);
    }
    seeds
}

fn push_extra_training_selection_seed(seeds: &mut Vec<u64>, selection_seed: u64, seed: u64) {
    if seed != selection_seed && !seeds.contains(&seed) {
        seeds.push(seed);
    }
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

pub(crate) fn render_training_particle_count_for_output(
    model_output: &Path,
    particle_count: usize,
) -> usize {
    if is_catalog_model_output_path(model_output) {
        particle_count.max(CATALOG_3D_VALIDATION_PARTICLES)
    } else {
        particle_count
    }
}

pub(crate) fn render_training_rollout_steps_for_output(
    model_output: &Path,
    rollout_steps: usize,
) -> usize {
    if is_catalog_model_output_path(model_output) {
        rollout_steps.max(catalog_promotion_max_steps())
    } else {
        rollout_steps
    }
}

fn catalog_promotion_max_steps() -> usize {
    CATALOG_3D_PROMOTION_STEPS
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
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
        if source.starts_with("render-refined-rust:")
            && local_growth_seed
            && target_conditionless_lineage(target, source)
        {
            return source.to_string();
        }
        if target_conditionless_lineage(target, source) && local_growth_seed {
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

pub(crate) fn save_render_training_manifest_for_validation(
    model_output: &Path,
    manifest: &BpkModelManifest,
    target: MeshTargetArg,
) -> Result<Option<PathBuf>, Box<dyn std::error::Error>> {
    if is_catalog_model_output_path(model_output) {
        let candidate_path = catalog_bound_candidate_path(target, std::process::id());
        if let Some(parent) = candidate_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        crate::import::save_manifest(&candidate_path, manifest)?;
        Ok(Some(candidate_path))
    } else {
        crate::import::save_manifest(model_output, manifest)?;
        Ok(None)
    }
}

pub(crate) fn finalize_render_training_manifest_promotion(
    model_output: &Path,
    manifest: &BpkModelManifest,
    candidate_path: Option<&Path>,
    promotion_error: Option<Box<dyn std::error::Error>>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(candidate_path) = candidate_path {
        if promotion_error.is_none() {
            crate::import::save_manifest(model_output, manifest)?;
        }
        std::fs::remove_file(candidate_path).ok();
    }
    if let Some(error) = promotion_error {
        return Err(error);
    }
    Ok(())
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
    if growth_3d_seed_has_coordinate_scaffold(seed_mode) {
        return Err(std::io::Error::other(format!(
            "catalog-bound 3D render training output {} requires a strict no-scaffold local growth seed; got seed_mode={seed_mode:?}",
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
    if !target_conditionless_lineage(target, source) {
        return Err(std::io::Error::other(format!(
            "catalog-bound 3D render training output {} requires a conditionless-local base model for target {target:?}; source={source:?}",
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

pub(crate) fn target_conditionless_lineage(target: MeshTargetArg, source: &str) -> bool {
    local_conditionless_lineage(source) && source.contains(mesh_target_lineage_marker(target))
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
