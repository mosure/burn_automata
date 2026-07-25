use std::{fs, path::PathBuf, time::Duration};

use serde::{Deserialize, Serialize};

use crate::adaptive::{
    AdaptiveTarget2dGpuTrainingReport, AdaptiveTarget2dTrainingConfig,
    AdaptiveTarget2dValidationConfig, AdaptiveTarget2dValidationReport,
    train_adaptive_target_2d_gpu,
};
use crate::{
    AdaptiveModelArtifact, AdaptiveNpaConfig, AdaptiveNpaModel, NpaConfig, NpaModel,
    Target2dGpuBackend, Target2dGpuCheckpointConfig, Target2dLossConfig, load_adaptive_model,
    save_adaptive_model,
};
use burn_automata_kernels::HashGridConfig;

use super::super::args::Command;

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AdaptiveTarget2dExperiment {
    backend: Target2dGpuBackend,
    source: AdaptiveTarget2dSource,
    output: AdaptiveTarget2dOutput,
    model: AdaptiveTarget2dModelInit,
    target: AdaptiveTarget2dTarget,
    hashgrid: HashGridConfig,
    training: AdaptiveTarget2dTrainingConfig,
    loss: Target2dLossConfig,
    checkpoint: Option<AdaptiveTarget2dCheckpoint>,
    validation: Option<AdaptiveTarget2dValidationConfig>,
}

impl Default for AdaptiveTarget2dExperiment {
    fn default() -> Self {
        Self {
            backend: Target2dGpuBackend::Wgpu,
            source: AdaptiveTarget2dSource::default(),
            output: AdaptiveTarget2dOutput::default(),
            model: AdaptiveTarget2dModelInit::default(),
            target: AdaptiveTarget2dTarget::default(),
            hashgrid: HashGridConfig::growing_2d(),
            training: AdaptiveTarget2dTrainingConfig::default(),
            loss: Target2dLossConfig::default(),
            checkpoint: None,
            validation: None,
        }
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AdaptiveTarget2dSource {
    target_image: Option<PathBuf>,
    adaptive_model: Option<PathBuf>,
    rule_model: Option<PathBuf>,
    /// Optional trained local residual checkpoint emitted by this command's
    /// checkpoint writer. This permits reproducible refinement without
    /// pretending the residual is a complete adaptive artifact.
    residual_model: Option<PathBuf>,
    oracle_model: Option<PathBuf>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AdaptiveTarget2dOutput {
    model: Option<PathBuf>,
    report: Option<PathBuf>,
    evaluation_report: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AdaptiveTarget2dModelInit {
    rule_seed: u64,
    controller_seed: u64,
}

impl Default for AdaptiveTarget2dModelInit {
    fn default() -> Self {
        Self {
            rule_seed: 42,
            controller_seed: 0xada2_7a2d,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AdaptiveTarget2dTarget {
    points: usize,
    threshold: f32,
    image_size: Option<usize>,
}

impl Default for AdaptiveTarget2dTarget {
    fn default() -> Self {
        Self {
            points: 4_096,
            threshold: 0.05,
            image_size: None,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AdaptiveTarget2dCheckpoint {
    current_rule: PathBuf,
    best_rule: PathBuf,
    report: PathBuf,
    current_training_state: Option<PathBuf>,
    resume_training_state: Option<PathBuf>,
    curriculum_resume: bool,
    include_particle_pool: bool,
    interval_steps: usize,
    interval_seconds: Option<u64>,
}

impl Default for AdaptiveTarget2dCheckpoint {
    fn default() -> Self {
        Self {
            current_rule: PathBuf::new(),
            best_rule: PathBuf::new(),
            report: PathBuf::new(),
            current_training_state: None,
            resume_training_state: None,
            curriculum_resume: false,
            include_particle_pool: false,
            interval_steps: 0,
            interval_seconds: Some(900),
        }
    }
}

#[derive(Serialize)]
struct AdaptiveTarget2dExperimentReport {
    objective: &'static str,
    config: String,
    target_image: String,
    initial_model: String,
    output_model: String,
    model_sha256: String,
    no_hidden_fine_state: bool,
    training: AdaptiveTarget2dGpuTrainingReport,
    validation: Option<AdaptiveTarget2dValidationReport>,
}

#[derive(Serialize)]
struct AdaptiveTarget2dEvaluationReport {
    objective: &'static str,
    config: String,
    target_image: String,
    evaluated_model: String,
    output_model: Option<String>,
    model_sha256: Option<String>,
    no_hidden_fine_state: bool,
    validation: AdaptiveTarget2dValidationReport,
}

pub(super) fn run_train_adaptive_target2d(
    command: Command,
) -> Result<(), Box<dyn std::error::Error>> {
    let Command::TrainAdaptiveTarget2d { config } = command else {
        unreachable!("adaptive Target2D dispatcher passed a different command")
    };
    let source = fs::read_to_string(&config)?;
    let experiment: AdaptiveTarget2dExperiment = toml::from_str(&source)?;
    let target_image = experiment.source.target_image.as_ref().ok_or_else(|| {
        std::io::Error::other("train-adaptive-target2d requires source.target_image")
    })?;
    let output_model =
        experiment.output.model.as_ref().ok_or_else(|| {
            std::io::Error::other("train-adaptive-target2d requires output.model")
        })?;
    let output_report =
        experiment.output.report.as_ref().ok_or_else(|| {
            std::io::Error::other("train-adaptive-target2d requires output.report")
        })?;
    if experiment.target.points == 0
        || !experiment.target.threshold.is_finite()
        || !(0.0..1.0).contains(&experiment.target.threshold)
    {
        return Err(std::io::Error::other(
            "adaptive Target2D target points and threshold are invalid",
        )
        .into());
    }
    experiment.hashgrid.validate()?;

    let (mut model, initial_model, fresh) = load_or_initialize_model(&experiment)?;
    let target = super::target2d::load_target_image_2d_adaptive(
        target_image,
        experiment.target.threshold,
        experiment.target.points,
        experiment.target.image_size,
    )?;
    let checkpoint = experiment
        .checkpoint
        .as_ref()
        .map(|checkpoint| {
            if checkpoint.current_rule.as_os_str().is_empty()
                || checkpoint.best_rule.as_os_str().is_empty()
                || checkpoint.report.as_os_str().is_empty()
            {
                return Err(std::io::Error::other(
                    "adaptive Target2D checkpoint paths must all be non-empty",
                ));
            }
            if checkpoint.resume_training_state.is_some()
                && checkpoint.current_training_state.is_none()
            {
                return Err(std::io::Error::other(
                    "adaptive Target2D resume requires checkpoint.current_training_state",
                ));
            }
            if checkpoint.resume_training_state.is_none()
                && checkpoint
                    .current_training_state
                    .as_ref()
                    .is_some_and(|path| path.exists())
            {
                let path = checkpoint
                    .current_training_state
                    .as_ref()
                    .expect("checked as present");
                return Err(std::io::Error::other(format!(
                    "adaptive Target2D training state {} already exists; use checkpoint.resume_training_state for an explicit resume or choose new checkpoint outputs",
                    path.display(),
                )));
            }
            if checkpoint.curriculum_resume && checkpoint.include_particle_pool {
                return Err(std::io::Error::other(
                    "adaptive Target2D curriculum resume must reset the particle pool",
                ));
            }
            let resume_model_sha256 = checkpoint
                .resume_training_state
                .as_ref()
                .map(|_| {
                    let model_path = match experiment.training.rule_training {
                        crate::adaptive::AdaptiveTarget2dRuleTraining::FrozenBaseCompatibleResidual
                        | crate::adaptive::AdaptiveTarget2dRuleTraining::FrozenBaseMaterialConditionedResidual
                        | crate::adaptive::AdaptiveTarget2dRuleTraining::FrozenBaseNormalizedAdaptiveResidual => {
                            experiment.source.residual_model.as_ref()
                        }
                        _ => experiment.source.rule_model.as_ref(),
                    }
                    .ok_or_else(|| {
                        std::io::Error::other(
                            "adaptive Target2D resume requires the trained rule checkpoint as source.rule_model or source.residual_model",
                        )
                    })?;
                    crate::import::bpk_payload_sha256(&fs::read(model_path)?)
                        .map_err(std::io::Error::other)
                })
                .transpose()?;
            Ok(Target2dGpuCheckpointConfig {
                current_model_output: checkpoint.current_rule.clone(),
                best_model_output: checkpoint.best_rule.clone(),
                metadata_output: checkpoint.report.clone(),
                training_state_output: checkpoint.current_training_state.clone(),
                resume_training_state: checkpoint.resume_training_state.clone(),
                resume_model_sha256,
                curriculum_resume: checkpoint.curriculum_resume,
                include_particle_pool: checkpoint.include_particle_pool,
                source: format!("train-adaptive-target2d:{}", config.display()),
                interval_steps: checkpoint.interval_steps,
                interval_duration: checkpoint.interval_seconds.map(Duration::from_secs),
            })
        })
        .transpose()?;
    let validation_target = target.clone();
    let validation_material = experiment.training.material;
    let training_config = experiment.training.clone();
    let report = train_adaptive_target_2d_gpu(
        experiment.backend,
        &mut model,
        &experiment.hashgrid,
        target,
        training_config,
        experiment.loss,
        checkpoint.as_ref(),
    )?;
    validate_deployable_model(&model)?;
    let validation =
        run_validation_if_requested(&experiment, &model, &validation_target, validation_material)?;
    let provenance = Some(format!(
        "train-adaptive-target2d:{}:{}",
        config.display(),
        target_image.display()
    ));
    let artifact = if fresh {
        AdaptiveModelArtifact::fresh_task_trained(model, provenance)?
    } else {
        AdaptiveModelArtifact::task_trained(model, provenance)?
    };
    let model_sha256 = save_adaptive_model(output_model, &artifact)?;
    let experiment_report = AdaptiveTarget2dExperimentReport {
        objective: "recurrent_adaptive_target2d_active_material",
        config: config.display().to_string(),
        target_image: target_image.display().to_string(),
        initial_model,
        output_model: output_model.display().to_string(),
        model_sha256,
        no_hidden_fine_state: true,
        training: report,
        validation,
    };
    if let Some(parent) = output_report.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        output_report,
        serde_json::to_vec_pretty(&experiment_report)?,
    )?;
    println!(
        "adaptive Target2D model={} report={} active={} reference={} best_psnr_db={} validation={}",
        output_model.display(),
        output_report.display(),
        experiment_report.training.active_particle_count,
        experiment_report.training.reference_particle_count,
        experiment_report
            .training
            .training
            .best_fresh_seed_render_rgb_psnr_db
            .map_or_else(|| "none".to_owned(), |value| format!("{value:.3}")),
        experiment_report
            .validation
            .as_ref()
            .map_or("not-requested", |report| if report.passed {
                "passed"
            } else {
                "failed"
            }),
    );
    if let Some(validation) = &experiment_report.validation {
        validation.require_pass()?;
    }
    Ok(())
}

pub(super) fn run_eval_adaptive_target2d(
    command: Command,
) -> Result<(), Box<dyn std::error::Error>> {
    let Command::EvalAdaptiveTarget2d { config } = command else {
        unreachable!("adaptive Target2D evaluator dispatcher passed a different command")
    };
    let source = fs::read_to_string(&config)?;
    let experiment: AdaptiveTarget2dExperiment = toml::from_str(&source)?;
    let target_image = experiment.source.target_image.as_ref().ok_or_else(|| {
        std::io::Error::other("eval-adaptive-target2d requires source.target_image")
    })?;
    let output_report = experiment
        .output
        .evaluation_report
        .as_ref()
        .ok_or_else(|| {
            std::io::Error::other("eval-adaptive-target2d requires output.evaluation_report")
        })?;
    if experiment.source.adaptive_model.is_none()
        && experiment.source.rule_model.is_none()
        && experiment.source.residual_model.is_none()
    {
        return Err(std::io::Error::other(
            "eval-adaptive-target2d requires an existing source model",
        )
        .into());
    }
    if experiment.target.points == 0
        || !experiment.target.threshold.is_finite()
        || !(0.0..1.0).contains(&experiment.target.threshold)
    {
        return Err(std::io::Error::other(
            "adaptive Target2D target points and threshold are invalid",
        )
        .into());
    }
    experiment.hashgrid.validate()?;
    let (mut model, evaluated_model, _) = load_or_initialize_model(&experiment)?;
    crate::hyper::e2e_training::prepare_adaptive_target2d_model(&mut model, &experiment.training)?;
    validate_deployable_model(&model)?;
    let target = super::target2d::load_target_image_2d_adaptive(
        target_image,
        experiment.target.threshold,
        experiment.target.points,
        experiment.target.image_size,
    )?;
    let validation =
        run_validation_if_requested(&experiment, &model, &target, experiment.training.material)?
            .ok_or_else(|| {
                std::io::Error::other("eval-adaptive-target2d requires a [validation] section")
            })?;
    validation.require_pass()?;
    let (output_model, model_sha256) = if let Some(output_model) = experiment.output.model.as_ref()
    {
        let provenance = Some(format!(
            "{} promoted by eval-adaptive-target2d:{}",
            evaluated_model,
            config.display(),
        ));
        let artifact = if let Some(source_model) = experiment.source.adaptive_model.as_ref() {
            let mut artifact = load_adaptive_model(source_model)?;
            artifact.model = model.clone();
            artifact.source = provenance;
            artifact.validate()?;
            artifact
        } else {
            AdaptiveModelArtifact::task_trained(model.clone(), provenance)?
        };
        let digest = save_adaptive_model(output_model, &artifact)?;
        (Some(output_model.display().to_string()), Some(digest))
    } else {
        (None, None)
    };
    let report = AdaptiveTarget2dEvaluationReport {
        objective: "recurrent_adaptive_target2d_active_material_evaluation",
        config: config.display().to_string(),
        target_image: target_image.display().to_string(),
        evaluated_model,
        output_model,
        model_sha256,
        no_hidden_fine_state: true,
        validation,
    };
    if let Some(parent) = output_report.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output_report, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "adaptive Target2D evaluation report={} active={} reference={} mean_psnr_db={:.3} worst_psnr_db={:.3} validation={}",
        output_report.display(),
        report.validation.active_particle_count,
        report.validation.reference_particle_count,
        report.validation.mean_adaptive_psnr_db,
        report.validation.worst_adaptive_psnr_db,
        if report.validation.passed {
            "passed"
        } else {
            "failed"
        },
    );
    Ok(())
}

fn validate_deployable_model(model: &AdaptiveNpaModel) -> Result<(), Box<dyn std::error::Error>> {
    let canonical_rule_path = model.local_residual_rule.is_none()
        || model.uses_canonical_compatible_residual()
        || model.uses_canonical_normalized_residual();
    if model.config.retain_bootstrap_templates
        || model.config.coarse_dynamics
            != crate::adaptive::AdaptiveCoarseDynamics::RepresentedMeasure
        || !canonical_rule_path
        || model.proxy_rule.is_some()
        || model.deployment_rule.is_some()
        || model.deployment_local_rule.is_some()
        || model.closure_mode_rule.is_some()
    {
        return Err(std::io::Error::other(
            "adaptive Target2D model contains a forbidden hidden fine-state or noncanonical auxiliary rule path",
        )
        .into());
    }
    Ok(())
}

fn run_validation_if_requested(
    experiment: &AdaptiveTarget2dExperiment,
    model: &AdaptiveNpaModel,
    target: &crate::TargetImage2d,
    material: crate::adaptive::AdaptiveTarget2dMaterialConfig,
) -> Result<Option<AdaptiveTarget2dValidationReport>, Box<dyn std::error::Error>> {
    let Some(validation) = &experiment.validation else {
        return Ok(None);
    };
    let oracle_path = experiment.source.oracle_model.as_ref().ok_or_else(|| {
        std::io::Error::other("adaptive Target2D validation requires source.oracle_model")
    })?;
    let oracle = crate::import::load_manifest(oracle_path)?.into_model();
    #[cfg(feature = "gpu_wgpu")]
    {
        crate::adaptive::validate_adaptive_target2d_wgpu(
            model,
            &oracle,
            &experiment.hashgrid,
            target,
            material,
            experiment.loss,
            validation,
        )
        .map(Some)
        .map_err(Into::into)
    }
    #[cfg(not(feature = "gpu_wgpu"))]
    {
        let _ = (model, target, material, oracle, validation);
        Err(
            std::io::Error::other("adaptive Target2D validation requires the gpu_wgpu feature")
                .into(),
        )
    }
}

fn load_or_initialize_model(
    experiment: &AdaptiveTarget2dExperiment,
) -> Result<(AdaptiveNpaModel, String, bool), Box<dyn std::error::Error>> {
    if experiment.source.adaptive_model.is_some()
        && (experiment.source.rule_model.is_some() || experiment.source.residual_model.is_some())
    {
        return Err(std::io::Error::other(
            "source.adaptive_model cannot be combined with source.rule_model or source.residual_model",
        )
        .into());
    }
    if let Some(path) = &experiment.source.adaptive_model {
        let artifact = load_adaptive_model(path)?;
        return Ok((artifact.model, path.display().to_string(), false));
    }

    let rule = if let Some(path) = &experiment.source.rule_model {
        crate::import::load_manifest(path)?.into_model()
    } else {
        NpaModel::upstream_seeded(NpaConfig::growing_2d(), experiment.model.rule_seed)
    };
    let mut adaptive = AdaptiveNpaConfig::growing_2d();
    adaptive.target_leaves = experiment.training.target2d.particle_count;
    adaptive.min_leaves = adaptive
        .min_leaves
        .min(experiment.training.target2d.particle_count);
    adaptive.max_leaves = adaptive
        .max_leaves
        .max(experiment.training.material.reference_particle_count);
    adaptive.bootstrap_fine_leaves = experiment.training.material.reference_particle_count;
    if experiment.training.rule_training
        == crate::adaptive::AdaptiveTarget2dRuleTraining::NormalizedAdaptiveRule
    {
        adaptive.rule_perception = crate::adaptive::AdaptiveRulePerception::NormalizedAdaptive;
    }
    if rule.config.auxiliary_input_dims == 1
        && matches!(
            experiment.training.rule_training,
            crate::adaptive::AdaptiveTarget2dRuleTraining::SharedScaleConditionedRule
                | crate::adaptive::AdaptiveTarget2dRuleTraining::NormalizedAdaptiveRule
        )
    {
        adaptive.material_scale_conditioning = true;
    }
    let mut model = AdaptiveNpaModel::seeded(rule, adaptive, experiment.model.controller_seed)?;
    if let Some(path) = &experiment.source.residual_model {
        use crate::adaptive::{
            AdaptiveLocalRuleSemantics, AdaptiveResidualGateReference, AdaptiveTarget2dRuleTraining,
        };

        let (material_conditioned, normalized_adaptive) = match experiment.training.rule_training {
            AdaptiveTarget2dRuleTraining::FrozenBaseMaterialConditionedResidual => (true, false),
            AdaptiveTarget2dRuleTraining::FrozenBaseCompatibleResidual => (false, false),
            AdaptiveTarget2dRuleTraining::FrozenBaseNormalizedAdaptiveResidual => (false, true),
            _ => {
                return Err(std::io::Error::other(
                    "source.residual_model requires a frozen-base residual rule_training mode",
                )
                .into());
            }
        };
        model.config.local_rule_semantics = if normalized_adaptive {
            AdaptiveLocalRuleSemantics::NormalizedExposureResidual
        } else {
            AdaptiveLocalRuleSemantics::CompatibleResidual
        };
        model.config.residual_gate_reference = AdaptiveResidualGateReference::BaseRule;
        model.config.local_residual_scale = 1.0;
        model.config.local_residual_motion_scale = 1.0;
        model.config.local_residual_state_scale = 1.0;
        model.config.closure_moment_features = false;
        model.config.closure_recurrent_mode = false;
        if normalized_adaptive {
            model.enable_material_conditioned_normalized_residual_rule()?;
        } else if material_conditioned {
            model.enable_material_conditioned_compatible_residual_rule()?;
        } else {
            model.enable_zero_local_residual_rule()?;
        }
        let expected = model
            .local_residual_rule
            .as_ref()
            .expect("residual initializer must install a rule")
            .config
            .clone();
        let residual = crate::import::load_manifest(path)?.into_model();
        if residual.config != expected {
            return Err(std::io::Error::other(format!(
                "residual checkpoint {} has an incompatible model configuration",
                path.display()
            ))
            .into());
        }
        model.local_residual_rule = Some(residual);
        model.validate()?;
    }
    let mut label = experiment.source.rule_model.as_ref().map_or_else(
        || "fresh-upstream-initialization".to_owned(),
        |path| path.display().to_string(),
    );
    if let Some(path) = &experiment.source.residual_model {
        label.push_str(&format!(" + residual:{}", path.display()));
    }
    Ok((model, label, experiment.source.rule_model.is_none()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const VERIFIED_RECIPES: [(&str, &str); 4] = [
        (
            "stage1",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../configs/verified/adaptive/recurrent_target2d_lizard_stage1_scale_age1024_2d_cuda.toml"
            )),
        ),
        (
            "stage2",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../configs/verified/adaptive/recurrent_target2d_lizard_stage2_scale_tail_age4096_2d_cuda.toml"
            )),
        ),
        (
            "stage3",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../configs/verified/adaptive/recurrent_target2d_lizard_stage3_fullrule_tail_age4096_2d_cuda.toml"
            )),
        ),
        (
            "promotion",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../configs/verified/adaptive/recurrent_target2d_lizard_events1_eval_3070_2d_wgpu.toml"
            )),
        ),
    ];

    #[test]
    fn verified_recurrent_lizard_recipes_preserve_the_upstream_objective() {
        for (name, source) in VERIFIED_RECIPES {
            let experiment: AdaptiveTarget2dExperiment =
                toml::from_str(source).unwrap_or_else(|error| {
                    panic!("verified adaptive recipe {name} did not parse: {error}")
                });
            assert_eq!(experiment.target.points, 4_096, "{name}");
            assert_eq!(experiment.training.target2d.particle_count, 3_070, "{name}");
            assert_eq!(
                experiment.training.material.reference_particle_count, 4_096,
                "{name}"
            );
            assert!(experiment.loss.center, "{name}");
            assert_eq!(experiment.loss.splat_loss_weight, 2.0, "{name}");
            assert_eq!(experiment.loss.color_loss_weight, 5.0, "{name}");
            assert_eq!(experiment.loss.density_loss_weight, 1.0, "{name}");
            assert_eq!(
                experiment.loss.displacement_regularizer_weight, 0.01,
                "{name}"
            );
            assert_eq!(experiment.loss.overflow_regularizer_weight, 100.0, "{name}");
            assert_eq!(experiment.loss.bound_regularizer_weight, 100.0, "{name}");
            assert_eq!(
                experiment.loss.background_density_loss_weight, 0.0,
                "{name}"
            );
            assert_eq!(experiment.loss.render_rgb_loss_weight, 0.0, "{name}");
            assert_eq!(experiment.loss.shape_chamfer_loss_weight, 0.0, "{name}");
        }
    }

    #[test]
    fn verified_recurrent_lizard_promotion_uses_the_robust_event_budget() {
        let experiment: AdaptiveTarget2dExperiment =
            toml::from_str(VERIFIED_RECIPES[3].1).expect("promotion recipe should parse");
        let validation = experiment
            .validation
            .expect("promotion recipe should include strict validation");
        assert_eq!(experiment.backend, Target2dGpuBackend::Wgpu);
        assert_eq!(experiment.training.topology.events_per_interval, 1);
        assert_eq!(experiment.training.topology.interval_steps, 95);
        assert_eq!(validation.quality_horizon_min_steps, 512);
        assert_eq!(validation.min_quality_mean_adaptive_psnr_db, 26.0);
        assert_eq!(validation.min_quality_worst_adaptive_psnr_db, 24.0);
        assert_eq!(validation.max_interaction_work_ratio, 0.8);
        assert_eq!(validation.max_wall_time_ratio, 1.1);
        assert!(experiment.output.model.is_some());
        assert!(experiment.output.evaluation_report.is_some());
    }
}
