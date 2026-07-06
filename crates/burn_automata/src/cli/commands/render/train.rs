use crate::cli::prelude::*;

use super::super::{resolve_direct_selection_seed_training, resolve_full_coverage_adjoint};

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Render3dExperimentConfig {
    target: Option<String>,
    input: Render3dInputConfig,
    output: Render3dOutputConfig,
    training: Render3dTrainingConfig,
    objective: Render3dObjectiveConfig,
    optimizer: Render3dOptimizerConfig,
    adapter: Render3dAdapterConfig,
    seed: Render3dSeedConfig,
    render: Render3dRenderConfig,
    validation: Render3dValidationConfig,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Render3dInputConfig {
    base_model: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Render3dOutputConfig {
    model_output: Option<PathBuf>,
    report_output: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Render3dTrainingConfig {
    rounds: Option<usize>,
    supervised_steps_per_round: Option<usize>,
    particles: Option<usize>,
    rollout_steps: Option<usize>,
    gradient_particles: Option<usize>,
    gradient_mode: Option<String>,
    finite_diff_eps: Option<f32>,
    motion_gain: Option<f32>,
    perception_position_gain: Option<f32>,
    max_update_norm: Option<f32>,
    trajectory_supervision: Option<bool>,
    backend: Option<String>,
    weight_update_mode: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Render3dObjectiveConfig {
    trajectory_render_gain: Option<f32>,
    trajectory_mesh_gain: Option<f32>,
    trajectory_render_samples: Option<usize>,
    liveness_gain: Option<f32>,
    liveness_front_radius: Option<f32>,
    liveness_update_multiplier: Option<f32>,
    coverage_gain: Option<f32>,
    coverage_samples: Option<usize>,
    coverage_mode: Option<String>,
    coverage_softness: Option<f32>,
    coverage_repulsion_gain: Option<f32>,
    coverage_gap_gain: Option<f32>,
    coverage_repulsion_radius: Option<f32>,
    coverage_normal_weight: Option<f32>,
    extent_gain: Option<f32>,
    full_coverage_adjoint: Option<bool>,
    surface_gain: Option<f32>,
    surface_escape_gain: Option<f32>,
    opacity_gain: Option<f32>,
    material_liveness_gain: Option<f32>,
    material_tail_gain: Option<f32>,
    material_suppression_update_multiplier: Option<f32>,
    material_max_opacity_update: Option<f32>,
    scale_gain: Option<f32>,
    scale_budget_weight: Option<f32>,
    max_opacity_update: Option<f32>,
    direct_output_gradient_rms_cap: Option<f32>,
    direct_line_search: Option<bool>,
    direct_line_search_scales: Option<Vec<f32>>,
    direct_material_output_only: Option<bool>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Render3dOptimizerConfig {
    learning_rate: Option<f32>,
    grad_clip_norm: Option<f32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Render3dAdapterConfig {
    rank: Option<usize>,
    alpha: Option<f32>,
    seed: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Render3dSeedConfig {
    seed_scale: Option<f32>,
    seed_mode: Option<String>,
    selection_seed: Option<u64>,
    extra_selection_seeds: Option<Vec<u64>>,
    direct_selection_seed_training: Option<bool>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Render3dRenderConfig {
    image_size: Option<usize>,
    target_samples: Option<usize>,
    sigma: Option<f32>,
    min_sigma: Option<f32>,
    max_sigma: Option<f32>,
    gaussian_decode_mode: Option<String>,
    world_scale: Option<f32>,
    opacity_logit_bias: Option<f32>,
    density_weight: Option<f32>,
    color_weight: Option<f32>,
    depth_weight: Option<f32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Render3dValidationConfig {
    fail_on_validation: Option<bool>,
}

fn load_render3d_experiment_config(
    path: Option<&Path>,
) -> Result<Render3dExperimentConfig, Box<dyn std::error::Error>> {
    let Some(path) = path else {
        return Ok(Render3dExperimentConfig::default());
    };
    let text = std::fs::read_to_string(path)?;
    toml::from_str(&text).map_err(|err| {
        std::io::Error::other(format!(
            "failed to parse render3d experiment config {}: {err}",
            path.display()
        ))
        .into()
    })
}

fn render3d_config_value_enum<T: ValueEnum>(
    field: &str,
    value: Option<String>,
    fallback: T,
) -> Result<T, Box<dyn std::error::Error>> {
    match value {
        Some(value) => T::from_str(&value, true).map_err(|err| {
            std::io::Error::other(format!(
                "invalid {field} `{value}` in render3d TOML config: {err}"
            ))
            .into()
        }),
        None => Ok(fallback),
    }
}

fn render3d_config_value_enum_option<T: ValueEnum>(
    field: &str,
    value: Option<String>,
    fallback: Option<T>,
) -> Result<Option<T>, Box<dyn std::error::Error>> {
    match value {
        Some(value) => Ok(Some(T::from_str(&value, true).map_err(|err| {
            std::io::Error::other(format!(
                "invalid {field} `{value}` in render3d TOML config: {err}"
            ))
        })?)),
        None => Ok(fallback),
    }
}

fn override_bool_switch(value: Option<bool>, positive: bool, negative: bool) -> (bool, bool) {
    match value {
        Some(true) => (true, false),
        Some(false) => (false, true),
        None => (positive, negative),
    }
}

pub(crate) fn run_train_render_3d(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::TrainRender3d {
        config,
        target,
        base_model,
        model_output,
        report_output,
        rounds,
        supervised_steps_per_round,
        particles,
        rollout_steps,
        gradient_particles,
        gradient_mode,
        finite_diff_eps,
        motion_gain,
        perception_position_gain,
        max_update_norm,
        trajectory_supervision,
        trajectory_render_gain,
        trajectory_mesh_gain,
        trajectory_render_samples,
        liveness_gain,
        liveness_front_radius,
        liveness_update_multiplier,
        coverage_gain,
        coverage_samples,
        coverage_mode,
        coverage_softness,
        coverage_repulsion_gain,
        coverage_gap_gain,
        coverage_repulsion_radius,
        coverage_normal_weight,
        extent_gain,
        full_coverage_adjoint,
        no_full_coverage_adjoint,
        surface_gain,
        surface_escape_gain,
        opacity_gain,
        material_liveness_gain,
        material_tail_gain,
        material_suppression_update_multiplier,
        material_max_opacity_update,
        scale_gain,
        scale_budget_weight,
        max_opacity_update,
        learning_rate,
        grad_clip_norm,
        direct_output_gradient_rms_cap,
        direct_line_search,
        direct_line_search_scales,
        direct_material_output_only,
        training_backend,
        weight_update_mode,
        adapter_rank,
        adapter_alpha,
        adapter_seed,
        direct_selection_seed_training,
        no_direct_selection_seed_training,
        seed_scale,
        seed_mode,
        selection_seed,
        extra_selection_seeds,
        image_size,
        target_samples,
        sigma,
        min_sigma,
        max_sigma,
        gaussian_decode_mode,
        world_scale,
        render_opacity_logit_bias,
        density_weight,
        color_weight,
        depth_weight,
        fail_on_validation,
    } = command
    else {
        unreachable!("run_train_render_3d called with the wrong command variant");
    };

    let config = load_render3d_experiment_config(config.as_deref())?;
    let Render3dExperimentConfig {
        target: config_target,
        input: config_input,
        output: config_output,
        training: config_training,
        objective: config_objective,
        optimizer: config_optimizer,
        adapter: config_adapter,
        seed: config_seed,
        render: config_render,
        validation: config_validation,
    } = config;
    let Render3dInputConfig {
        base_model: config_base_model,
    } = config_input;
    let Render3dOutputConfig {
        model_output: config_model_output,
        report_output: config_report_output,
    } = config_output;
    let Render3dTrainingConfig {
        rounds: config_rounds,
        supervised_steps_per_round: config_supervised_steps_per_round,
        particles: config_particles,
        rollout_steps: config_rollout_steps,
        gradient_particles: config_gradient_particles,
        gradient_mode: config_gradient_mode,
        finite_diff_eps: config_finite_diff_eps,
        motion_gain: config_motion_gain,
        perception_position_gain: config_perception_position_gain,
        max_update_norm: config_max_update_norm,
        trajectory_supervision: config_trajectory_supervision,
        backend: config_training_backend,
        weight_update_mode: config_weight_update_mode,
    } = config_training;
    let Render3dObjectiveConfig {
        trajectory_render_gain: config_trajectory_render_gain,
        trajectory_mesh_gain: config_trajectory_mesh_gain,
        trajectory_render_samples: config_trajectory_render_samples,
        liveness_gain: config_liveness_gain,
        liveness_front_radius: config_liveness_front_radius,
        liveness_update_multiplier: config_liveness_update_multiplier,
        coverage_gain: config_coverage_gain,
        coverage_samples: config_coverage_samples,
        coverage_mode: config_coverage_mode,
        coverage_softness: config_coverage_softness,
        coverage_repulsion_gain: config_coverage_repulsion_gain,
        coverage_gap_gain: config_coverage_gap_gain,
        coverage_repulsion_radius: config_coverage_repulsion_radius,
        coverage_normal_weight: config_coverage_normal_weight,
        extent_gain: config_extent_gain,
        full_coverage_adjoint: config_full_coverage_adjoint,
        surface_gain: config_surface_gain,
        surface_escape_gain: config_surface_escape_gain,
        opacity_gain: config_opacity_gain,
        material_liveness_gain: config_material_liveness_gain,
        material_tail_gain: config_material_tail_gain,
        material_suppression_update_multiplier: config_material_suppression_update_multiplier,
        material_max_opacity_update: config_material_max_opacity_update,
        scale_gain: config_scale_gain,
        scale_budget_weight: config_scale_budget_weight,
        max_opacity_update: config_max_opacity_update,
        direct_output_gradient_rms_cap: config_direct_output_gradient_rms_cap,
        direct_line_search: config_direct_line_search,
        direct_line_search_scales: config_direct_line_search_scales,
        direct_material_output_only: config_direct_material_output_only,
    } = config_objective;
    let Render3dOptimizerConfig {
        learning_rate: config_learning_rate,
        grad_clip_norm: config_grad_clip_norm,
    } = config_optimizer;
    let Render3dAdapterConfig {
        rank: config_adapter_rank,
        alpha: config_adapter_alpha,
        seed: config_adapter_seed,
    } = config_adapter;
    let Render3dSeedConfig {
        seed_scale: config_seed_scale,
        seed_mode: config_seed_mode,
        selection_seed: config_selection_seed,
        extra_selection_seeds: config_extra_selection_seeds,
        direct_selection_seed_training: config_direct_selection_seed_training,
    } = config_seed;
    let Render3dRenderConfig {
        image_size: config_image_size,
        target_samples: config_target_samples,
        sigma: config_sigma,
        min_sigma: config_min_sigma,
        max_sigma: config_max_sigma,
        gaussian_decode_mode: config_gaussian_decode_mode,
        world_scale: config_world_scale,
        opacity_logit_bias: config_render_opacity_logit_bias,
        density_weight: config_density_weight,
        color_weight: config_color_weight,
        depth_weight: config_depth_weight,
    } = config_render;
    let Render3dValidationConfig {
        fail_on_validation: config_fail_on_validation,
    } = config_validation;

    let target = render3d_config_value_enum("target", config_target, target)?;
    let base_model = config_base_model.or(base_model);
    let model_output = config_model_output.unwrap_or(model_output);
    let report_output = config_report_output.unwrap_or(report_output);
    let rounds = config_rounds.unwrap_or(rounds);
    let supervised_steps_per_round =
        config_supervised_steps_per_round.unwrap_or(supervised_steps_per_round);
    let particles = config_particles.unwrap_or(particles);
    let rollout_steps = config_rollout_steps.unwrap_or(rollout_steps);
    let gradient_particles = config_gradient_particles.unwrap_or(gradient_particles);
    let gradient_mode = render3d_config_value_enum(
        "training.gradient_mode",
        config_gradient_mode,
        gradient_mode,
    )?;
    let finite_diff_eps = config_finite_diff_eps.unwrap_or(finite_diff_eps);
    let motion_gain = config_motion_gain.unwrap_or(motion_gain);
    let perception_position_gain =
        config_perception_position_gain.unwrap_or(perception_position_gain);
    let max_update_norm = config_max_update_norm.unwrap_or(max_update_norm);
    let trajectory_supervision = config_trajectory_supervision.unwrap_or(trajectory_supervision);
    let trajectory_render_gain = config_trajectory_render_gain.unwrap_or(trajectory_render_gain);
    let trajectory_mesh_gain = config_trajectory_mesh_gain.unwrap_or(trajectory_mesh_gain);
    let trajectory_render_samples =
        config_trajectory_render_samples.unwrap_or(trajectory_render_samples);
    let liveness_gain = config_liveness_gain.unwrap_or(liveness_gain);
    let liveness_front_radius = config_liveness_front_radius.unwrap_or(liveness_front_radius);
    let liveness_update_multiplier =
        config_liveness_update_multiplier.unwrap_or(liveness_update_multiplier);
    let coverage_gain = config_coverage_gain.unwrap_or(coverage_gain);
    let coverage_samples = config_coverage_samples.unwrap_or(coverage_samples);
    let coverage_mode = render3d_config_value_enum(
        "objective.coverage_mode",
        config_coverage_mode,
        coverage_mode,
    )?;
    let coverage_softness = config_coverage_softness.unwrap_or(coverage_softness);
    let coverage_repulsion_gain = config_coverage_repulsion_gain.unwrap_or(coverage_repulsion_gain);
    let coverage_gap_gain = config_coverage_gap_gain.or(coverage_gap_gain);
    let coverage_repulsion_radius =
        config_coverage_repulsion_radius.unwrap_or(coverage_repulsion_radius);
    let coverage_normal_weight = config_coverage_normal_weight.unwrap_or(coverage_normal_weight);
    let extent_gain = config_extent_gain.unwrap_or(extent_gain);
    let (full_coverage_adjoint, no_full_coverage_adjoint) = override_bool_switch(
        config_full_coverage_adjoint,
        full_coverage_adjoint,
        no_full_coverage_adjoint,
    );
    let surface_gain = config_surface_gain.unwrap_or(surface_gain);
    let surface_escape_gain = config_surface_escape_gain.unwrap_or(surface_escape_gain);
    let opacity_gain = config_opacity_gain.unwrap_or(opacity_gain);
    let material_liveness_gain = config_material_liveness_gain.unwrap_or(material_liveness_gain);
    let material_tail_gain = config_material_tail_gain.unwrap_or(material_tail_gain);
    let material_suppression_update_multiplier = config_material_suppression_update_multiplier
        .unwrap_or(material_suppression_update_multiplier);
    let material_max_opacity_update =
        config_material_max_opacity_update.unwrap_or(material_max_opacity_update);
    let scale_gain = config_scale_gain.unwrap_or(scale_gain);
    let scale_budget_weight = config_scale_budget_weight.unwrap_or(scale_budget_weight);
    let max_opacity_update = config_max_opacity_update.unwrap_or(max_opacity_update);
    let learning_rate = config_learning_rate.unwrap_or(learning_rate);
    let grad_clip_norm = config_grad_clip_norm.unwrap_or(grad_clip_norm);
    let direct_output_gradient_rms_cap =
        config_direct_output_gradient_rms_cap.unwrap_or(direct_output_gradient_rms_cap);
    let direct_line_search = config_direct_line_search.unwrap_or(direct_line_search);
    let direct_line_search_scales =
        config_direct_line_search_scales.unwrap_or(direct_line_search_scales);
    let direct_material_output_only =
        config_direct_material_output_only.unwrap_or(direct_material_output_only);
    let training_backend = render3d_config_value_enum(
        "training.backend",
        config_training_backend,
        training_backend,
    )?;
    let weight_update_mode = render3d_config_value_enum(
        "training.weight_update_mode",
        config_weight_update_mode,
        weight_update_mode,
    )?;
    let adapter_rank = config_adapter_rank.unwrap_or(adapter_rank);
    let adapter_alpha = config_adapter_alpha.unwrap_or(adapter_alpha);
    let adapter_seed = config_adapter_seed.unwrap_or(adapter_seed);
    let (direct_selection_seed_training, no_direct_selection_seed_training) = override_bool_switch(
        config_direct_selection_seed_training,
        direct_selection_seed_training,
        no_direct_selection_seed_training,
    );
    let seed_scale = config_seed_scale.or(seed_scale);
    let seed_mode =
        render3d_config_value_enum_option("seed.seed_mode", config_seed_mode, seed_mode)?;
    let selection_seed = config_selection_seed.unwrap_or(selection_seed);
    let extra_selection_seeds = config_extra_selection_seeds.unwrap_or(extra_selection_seeds);
    let image_size = config_image_size.unwrap_or(image_size);
    let target_samples = config_target_samples.unwrap_or(target_samples);
    let sigma = config_sigma.unwrap_or(sigma);
    let min_sigma = config_min_sigma.unwrap_or(min_sigma);
    let max_sigma = config_max_sigma.unwrap_or(max_sigma);
    let gaussian_decode_mode = render3d_config_value_enum(
        "render.gaussian_decode_mode",
        config_gaussian_decode_mode,
        gaussian_decode_mode,
    )?;
    let world_scale = config_world_scale.or(world_scale);
    let render_opacity_logit_bias =
        config_render_opacity_logit_bias.unwrap_or(render_opacity_logit_bias);
    let density_weight = config_density_weight.unwrap_or(density_weight);
    let color_weight = config_color_weight.unwrap_or(color_weight);
    let depth_weight = config_depth_weight.unwrap_or(depth_weight);
    let fail_on_validation = config_fail_on_validation.unwrap_or(fail_on_validation);

    let full_coverage_adjoint =
        resolve_full_coverage_adjoint(full_coverage_adjoint, no_full_coverage_adjoint)?;
    let direct_selection_seed_training = resolve_direct_selection_seed_training(
        direct_selection_seed_training,
        no_direct_selection_seed_training,
    )?;
    let hashgrid = crate::kernels::HashGridConfig::growing_3dgs();
    let seed_scale = seed_scale.unwrap_or_else(|| mesh_target_render_training_seed_scale(target));
    let requested_seed_mode = seed_mode.map(ParticleSeed::from);
    let target_mesh = mesh_target_for_arg(target, seed_scale);
    let (mut model, base_source, default_seed_mode) = if let Some(path) = base_model.as_ref() {
        let manifest = crate::import::load_manifest(path)?;
        let base_source = manifest.source.clone();
        let model = manifest.into_model();
        let default_seed_mode = default_render_training_seed_mode(target, &model);
        (model, base_source, default_seed_mode)
    } else {
        let default_seed_mode = render_training_default_seed_mode(target);
        let seed_mode = requested_seed_mode.unwrap_or(default_seed_mode);
        if !target_strict_conditionless_local_growth_seed(target, seed_mode) {
            return Err(std::io::Error::other(format!(
                "train-render3d without --base-model defaults to conditionless-local growth and requires the target strict conditionless-local growth seed; got seed_mode={seed_mode:?}"
            ))
            .into());
        }
        let (model, source) = render_training_base_model(target, &target_mesh, seed_mode)?;
        (model, Some(source), default_seed_mode)
    };
    let seed_mode = requested_seed_mode.unwrap_or(default_seed_mode);
    let catalog_bound_output = is_catalog_model_output_path(&model_output);
    validate_catalog_bound_render_training_output(
        &model_output,
        target,
        seed_mode,
        base_source.as_deref(),
    )?;
    let training_particles = render_training_particle_count_for_output(&model_output, particles);
    let training_rollout_steps =
        render_training_rollout_steps_for_output(&model_output, rollout_steps);
    let coverage_gap_gain = coverage_gap_gain.unwrap_or(coverage_repulsion_gain);
    let render = RenderLossConfig {
        image_size,
        sigma,
        min_sigma,
        max_sigma,
        gaussian_decode_mode: gaussian_decode_mode.into(),
        world_scale: world_scale.unwrap_or(seed_scale * 2.0),
        target_samples,
        opacity_logit_bias: render_opacity_logit_bias,
        density_weight,
        color_weight,
        depth_weight,
    };
    let training_selection_seeds =
        render_training_default_extra_selection_seeds(selection_seed, &extra_selection_seeds);
    let sgd = SgdConfig {
        learning_rate,
        grad_clip_norm,
        weight_decay: 0.0,
    };
    let report = run_render_proxy_training(
        &mut model,
        &hashgrid,
        &target_mesh,
        RenderProxyTrainingConfig {
            target,
            rounds,
            supervised_steps_per_round,
            particles: training_particles,
            rollout_steps: training_rollout_steps,
            gradient_particles,
            gradient_mode,
            finite_diff_eps,
            motion_gain,
            perception_position_gain,
            max_update_norm,
            trajectory_supervision,
            trajectory_render_gain,
            trajectory_mesh_gain,
            trajectory_render_samples,
            liveness_gain,
            liveness_front_radius,
            liveness_update_multiplier,
            coverage_gain,
            coverage_samples,
            coverage_mode,
            coverage_softness,
            coverage_repulsion_gain,
            coverage_gap_gain,
            coverage_repulsion_radius,
            coverage_normal_weight,
            extent_gain,
            full_coverage_adjoint,
            surface_gain,
            surface_escape_gain,
            opacity_gain,
            material_liveness_gain,
            material_tail_gain,
            material_suppression_update_multiplier,
            material_max_opacity_update,
            scale_gain,
            scale_budget_weight,
            max_opacity_update,
            direct_output_gradient_rms_cap,
            direct_line_search,
            direct_line_search_scales: direct_line_search_scales.clone(),
            direct_material_output_only,
            training_backend,
            weight_update_mode,
            adapter_rank,
            adapter_alpha,
            adapter_seed,
            direct_selection_seed_training,
            seed: 0x005a_173d,
            selection_seed: Some(selection_seed),
            selection_seeds: training_selection_seeds.clone(),
            seed_scale,
            seed_mode,
            render,
            sgd,
        },
    )?;
    if let Some(parent) = model_output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if let Some(parent) = report_output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let manifest = BpkModelManifest::from_model(
        &model,
        hashgrid.clone(),
        Some(render_training_source(
            target,
            base_source.as_deref(),
            seed_mode,
        )),
    );
    let validation_extra_seeds =
        render_training_validation_extra_seeds(selection_seed, &training_selection_seeds);
    let candidate_path =
        save_render_training_manifest_for_validation(&model_output, &manifest, target)?;
    let validation_model_path = candidate_path.as_ref().unwrap_or(&model_output);
    let mut catalog_promotion_validations = Vec::new();
    if catalog_bound_output {
        for validation_cfg in catalog_promotion_validation_configs(
            selection_seed,
            &training_selection_seeds,
            seed_scale,
            seed_mode,
            render,
        ) {
            catalog_promotion_validations.push(growth_3d_validation_report(
                validation_model_path,
                target,
                validation_cfg,
            )?);
        }
    }
    let loaded = crate::import::load_manifest(validation_model_path)?;
    let loaded_hashgrid = loaded.hashgrid.clone();
    let loaded_model = loaded.into_model();
    let growth_validation = growth_3d_validation_report(
        validation_model_path,
        target,
        Growth3dValidationConfig {
            particle_count: training_particles,
            steps: training_rollout_steps,
            seed: 0x005a_173d,
            extra_seeds: validation_extra_seeds,
            seed_scale,
            seed_mode,
            gate: Growth3dValidationGateArg::Strict,
            render,
        },
    )?;
    let final_render_loss = mesh_render_loss_for_model(
        &loaded_model,
        &loaded_hashgrid,
        &target_mesh,
        RenderLossEvalConfig {
            particle_count: training_particles,
            steps: training_rollout_steps,
            seed: 0x005a_173d,
            extra_seeds: Vec::new(),
            seed_scale,
            seed_mode,
            render,
        },
    )?;
    let strict_gate_summary = CliRenderTrainingGateSummary::from_validation(&growth_validation);
    let missing_train_signal_rounds = render_proxy_missing_signal_rounds(&report);
    let mut promotion_rejection_reasons = Vec::new();
    if catalog_bound_output
        && let Err(error) = require_catalog_promotion_validations_pass(
            &catalog_promotion_validations,
            &model_output,
        )
    {
        promotion_rejection_reasons.push(error.to_string());
    }
    if !missing_train_signal_rounds.is_empty() {
        promotion_rejection_reasons.push(format!(
            "direct rollout training signal missing for rounds {:?}",
            missing_train_signal_rounds
        ));
    }
    let promotion_error: Option<Box<dyn std::error::Error>> =
        if catalog_bound_output && !promotion_rejection_reasons.is_empty() {
            Some(std::io::Error::other(promotion_rejection_reasons.join("; ")).into())
        } else {
            None
        };
    let validation_error = if !missing_train_signal_rounds.is_empty() {
        Some(std::io::Error::other(format!(
            "direct rollout training signal missing for rounds {:?}; see {}",
            missing_train_signal_rounds,
            report_output.display()
        )))
    } else if fail_on_validation && !growth_3d_fail_on_validation_passed(&growth_validation) {
        Some(std::io::Error::other(format!(
            "render-proxy training failed strict growth validation (score={:.6}, failures={:?}); see {}",
            growth_validation.strict_score.score,
            growth_validation.strict_checks.failure_reasons,
            report_output.display(),
        )))
    } else {
        None
    };
    let promotion_rejection_reason = if catalog_bound_output {
        promotion_error.as_ref().map(|error| error.to_string())
    } else {
        validation_error.as_ref().map(|error| error.to_string())
    };
    let catalog_promotion = CliCatalogPromotionSummary::from_validation_and_training_result(
        catalog_bound_output,
        catalog_promotion_validations.len(),
        missing_train_signal_rounds.clone(),
        promotion_rejection_reason,
    );
    let output_report = CliRenderTrainingReport {
        schema_version: CLI_RENDER_TRAINING_REPORT_SCHEMA_VERSION,
        target,
        base_model: base_model.as_ref().map(|path| path.display().to_string()),
        model_output: model_output.display().to_string(),
        particle_count: training_particles,
        rollout_steps: training_rollout_steps,
        seed_scale,
        seed_mode,
        sgd,
        report,
        final_render_loss,
        strict_gate_summary,
        growth_validation,
        catalog_promotion,
        catalog_promotion_validations,
    };
    std::fs::write(
        &report_output,
        serde_json::to_string_pretty(&output_report)?,
    )?;
    if let Some(error) = promotion_error {
        println!(
            "wrote {} for rejected catalog candidate targeting {} render_loss={:.6} density_psnr={:.3} color_psnr={:.3} depth_psnr={:.3} passed={} strict_passed={} strict_score={:.3}",
            report_output.display(),
            model_output.display(),
            output_report.final_render_loss.total_loss,
            output_report.final_render_loss.density_psnr_db,
            output_report.final_render_loss.color_psnr_db,
            output_report.final_render_loss.depth_psnr_db,
            output_report.final_render_loss.passed,
            output_report.growth_validation.strict_passed,
            output_report.growth_validation.strict_score.score
        );
        return finalize_render_training_manifest_promotion(
            &model_output,
            &manifest,
            candidate_path.as_deref(),
            Some(error),
        );
    }
    finalize_render_training_manifest_promotion(
        &model_output,
        &manifest,
        candidate_path.as_deref(),
        None,
    )?;
    println!(
        "wrote {} and {} render_loss={:.6} density_psnr={:.3} color_psnr={:.3} depth_psnr={:.3} passed={} strict_passed={} strict_score={:.3}",
        model_output.display(),
        report_output.display(),
        output_report.final_render_loss.total_loss,
        output_report.final_render_loss.density_psnr_db,
        output_report.final_render_loss.color_psnr_db,
        output_report.final_render_loss.depth_psnr_db,
        output_report.final_render_loss.passed,
        output_report.growth_validation.strict_passed,
        output_report.growth_validation.strict_score.score
    );
    if let Some(error) = validation_error
        && (fail_on_validation || output_report.catalog_promotion.requested)
    {
        return Err(error.into());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render3d_experiment_config_accepts_nested_toml() {
        let config: Render3dExperimentConfig = toml::from_str(
            r#"
            target = "torus"

            [output]
            model_output = "artifacts/render3d_torus/model.bpk"
            report_output = "artifacts/render3d_torus/report.json"

            [training]
            rounds = 2
            supervised_steps_per_round = 3
            particles = 64
            rollout_steps = 8
            gradient_particles = 16
            gradient_mode = "analytic"
            backend = "direct-rollout"
            weight_update_mode = "adapter"

            [objective]
            coverage_mode = "sliced-ot"
            full_coverage_adjoint = false
            direct_line_search_scales = [0.25, 0.5, 1.0]

            [adapter]
            rank = 8
            alpha = 8.0

            [seed]
            seed_mode = "uniform-circle"
            extra_selection_seeds = [42, 99]

            [render]
            gaussian_decode_mode = "fixed-sh0"
            image_size = 32

            [validation]
            fail_on_validation = false
            "#,
        )
        .unwrap();

        assert!(matches!(
            render3d_config_value_enum("target", config.target, MeshTargetArg::Teapot).unwrap(),
            MeshTargetArg::Torus
        ));
        assert_eq!(config.training.rounds, Some(2));
        assert_eq!(
            config.objective.direct_line_search_scales,
            Some(vec![0.25, 0.5, 1.0])
        );
        assert!(matches!(
            render3d_config_value_enum_option::<SeedModeArg>(
                "seed.seed_mode",
                config.seed.seed_mode,
                None,
            )
            .unwrap(),
            Some(SeedModeArg::UniformCircle)
        ));
    }

    #[test]
    fn render3d_bool_switch_override_is_authoritative() {
        assert_eq!(override_bool_switch(Some(true), false, true), (true, false));
        assert_eq!(
            override_bool_switch(Some(false), true, false),
            (false, true)
        );
        assert_eq!(override_bool_switch(None, true, false), (true, false));
    }
}
