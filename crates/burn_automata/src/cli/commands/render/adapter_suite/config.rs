use crate::cli::prelude::*;

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct AdapterSuiteExperimentConfig {
    pub(super) input: AdapterSuiteInputConfig,
    pub(super) shared_base: AdapterSuiteSharedBaseConfig,
    pub(super) targets: AdapterSuiteTargetsConfig,
    pub(super) output: AdapterSuiteOutputConfig,
    pub(super) training: AdapterSuiteTrainingConfig,
    pub(super) objective: AdapterSuiteObjectiveConfig,
    pub(super) optimizer: AdapterSuiteOptimizerConfig,
    pub(super) adapter: AdapterSuiteAdapterConfig,
    pub(super) seed: AdapterSuiteSeedConfig,
    pub(super) render: AdapterSuiteRenderConfig,
    pub(super) validation: AdapterSuiteValidationConfig,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct AdapterSuiteInputConfig {
    pub(super) base_model: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct AdapterSuiteSharedBaseConfig {
    pub(super) output: Option<PathBuf>,
    pub(super) cycles: Option<usize>,
    pub(super) seed: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct AdapterSuiteTargetsConfig {
    pub(super) target_set: Option<String>,
    pub(super) targets: Option<Vec<String>>,
    pub(super) holdout_targets: Option<Vec<String>>,
    pub(super) auto_holdout_stride: Option<usize>,
    pub(super) auto_holdout_offset: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct AdapterSuiteOutputConfig {
    pub(super) output_dir: Option<PathBuf>,
    pub(super) report_output: Option<PathBuf>,
    pub(super) adapter_bank_output: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct AdapterSuiteTrainingConfig {
    pub(super) skip_shared_base_eval: Option<bool>,
    pub(super) rounds: Option<usize>,
    pub(super) supervised_steps_per_round: Option<usize>,
    pub(super) particles: Option<usize>,
    pub(super) rollout_steps: Option<usize>,
    pub(super) gradient_particles: Option<usize>,
    pub(super) gradient_mode: Option<String>,
    pub(super) finite_diff_eps: Option<f32>,
    pub(super) motion_gain: Option<f32>,
    pub(super) perception_position_gain: Option<f32>,
    pub(super) max_update_norm: Option<f32>,
    pub(super) trajectory_supervision: Option<bool>,
    pub(super) backend: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct AdapterSuiteObjectiveConfig {
    pub(super) direct_output_gradient_rms_cap: Option<f32>,
    pub(super) direct_line_search: Option<bool>,
    pub(super) direct_line_search_scales: Option<Vec<f32>>,
    pub(super) direct_material_output_only: Option<bool>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct AdapterSuiteOptimizerConfig {
    pub(super) learning_rate: Option<f32>,
    pub(super) grad_clip_norm: Option<f32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct AdapterSuiteAdapterConfig {
    pub(super) rank: Option<usize>,
    pub(super) alpha: Option<f32>,
    pub(super) seed: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct AdapterSuiteSeedConfig {
    pub(super) seed_scale: Option<f32>,
    pub(super) seed_mode: Option<String>,
    pub(super) selection_seed: Option<u64>,
    pub(super) extra_selection_seeds: Option<Vec<u64>>,
    pub(super) direct_selection_seed_training: Option<bool>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct AdapterSuiteRenderConfig {
    pub(super) image_size: Option<usize>,
    pub(super) target_samples: Option<usize>,
    pub(super) sigma: Option<f32>,
    pub(super) min_sigma: Option<f32>,
    pub(super) max_sigma: Option<f32>,
    pub(super) gaussian_decode_mode: Option<String>,
    pub(super) world_scale: Option<f32>,
    pub(super) opacity_logit_bias: Option<f32>,
    pub(super) density_weight: Option<f32>,
    pub(super) color_weight: Option<f32>,
    pub(super) depth_weight: Option<f32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct AdapterSuiteValidationConfig {
    pub(super) fail_on_validation: Option<bool>,
}

pub(super) fn load_adapter_suite_experiment_config(
    path: Option<&Path>,
) -> Result<AdapterSuiteExperimentConfig, Box<dyn std::error::Error>> {
    let Some(path) = path else {
        return Ok(AdapterSuiteExperimentConfig::default());
    };
    let text = std::fs::read_to_string(path)?;
    toml::from_str(&text).map_err(|err| {
        std::io::Error::other(format!(
            "failed to parse render3d adapter-suite config {}: {err}",
            path.display()
        ))
        .into()
    })
}

pub(super) fn adapter_suite_config_value_enum<T: ValueEnum>(
    field: &str,
    value: Option<String>,
    fallback: T,
) -> Result<T, Box<dyn std::error::Error>> {
    match value {
        Some(value) => T::from_str(&value, true).map_err(|err| {
            std::io::Error::other(format!(
                "invalid {field} `{value}` in render3d adapter-suite TOML config: {err}"
            ))
            .into()
        }),
        None => Ok(fallback),
    }
}

pub(super) fn adapter_suite_config_value_enum_option<T: ValueEnum>(
    field: &str,
    value: Option<String>,
    fallback: Option<T>,
) -> Result<Option<T>, Box<dyn std::error::Error>> {
    match value {
        Some(value) => Ok(Some(T::from_str(&value, true).map_err(|err| {
            std::io::Error::other(format!(
                "invalid {field} `{value}` in render3d adapter-suite TOML config: {err}"
            ))
        })?)),
        None => Ok(fallback),
    }
}

pub(super) fn adapter_suite_config_value_enum_vec<T: ValueEnum>(
    field: &str,
    values: Option<Vec<String>>,
) -> Result<Option<Vec<T>>, Box<dyn std::error::Error>> {
    values
        .map(|values| {
            values
                .into_iter()
                .map(|value| {
                    T::from_str(&value, true).map_err(|err| {
                        std::io::Error::other(format!(
                            "invalid {field} value `{value}` in render3d adapter-suite TOML config: {err}"
                        ))
                        .into()
                    })
                })
                .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()
        })
        .transpose()
}

pub(super) fn adapter_suite_override_bool_switch(
    value: Option<bool>,
    positive: bool,
    negative: bool,
) -> (bool, bool) {
    match value {
        Some(true) => (true, false),
        Some(false) => (false, true),
        None => (positive, negative),
    }
}

#[derive(Clone, Copy)]
pub(super) struct AdapterSuiteRenderSettings {
    pub(super) image_size: usize,
    pub(super) target_samples: usize,
    pub(super) sigma: f32,
    pub(super) min_sigma: f32,
    pub(super) max_sigma: f32,
    pub(super) gaussian_decode_mode: RenderGaussianDecodeModeArg,
    pub(super) world_scale: Option<f32>,
    pub(super) render_opacity_logit_bias: f32,
    pub(super) density_weight: f32,
    pub(super) color_weight: f32,
    pub(super) depth_weight: f32,
}

impl AdapterSuiteRenderSettings {
    pub(super) fn loss_config(self, seed_scale: f32) -> RenderLossConfig {
        RenderLossConfig {
            image_size: self.image_size,
            sigma: self.sigma,
            min_sigma: self.min_sigma,
            max_sigma: self.max_sigma,
            gaussian_decode_mode: self.gaussian_decode_mode.into(),
            world_scale: self.world_scale.unwrap_or(seed_scale * 2.0),
            target_samples: self.target_samples,
            opacity_logit_bias: self.render_opacity_logit_bias,
            density_weight: self.density_weight,
            color_weight: self.color_weight,
            depth_weight: self.depth_weight,
        }
    }
}

#[derive(Clone)]
pub(super) struct AdapterSuiteTrainingSettings {
    pub(super) supervised_steps_per_round: usize,
    pub(super) particles: usize,
    pub(super) rollout_steps: usize,
    pub(super) gradient_particles: usize,
    pub(super) gradient_mode: RenderGradientModeArg,
    pub(super) finite_diff_eps: f32,
    pub(super) motion_gain: f32,
    pub(super) perception_position_gain: f32,
    pub(super) max_update_norm: f32,
    pub(super) trajectory_supervision: bool,
    pub(super) training_backend: RenderTrainingBackendArg,
    pub(super) direct_output_gradient_rms_cap: f32,
    pub(super) direct_line_search: bool,
    pub(super) direct_line_search_scales: Vec<f32>,
    pub(super) direct_material_output_only: bool,
    pub(super) direct_selection_seed_training: bool,
    pub(super) selection_seed: u64,
    pub(super) selection_seeds: Vec<u64>,
    pub(super) sgd: SgdConfig,
    pub(super) adapter_rank: usize,
    pub(super) adapter_alpha: f32,
}

impl AdapterSuiteTrainingSettings {
    pub(super) fn render_proxy_config(
        &self,
        phase: AdapterSuiteTrainingPhaseConfig,
        render: RenderLossConfig,
    ) -> RenderProxyTrainingConfig {
        RenderProxyTrainingConfig {
            target: phase.target,
            rounds: phase.rounds,
            supervised_steps_per_round: self.supervised_steps_per_round,
            particles: self.particles,
            rollout_steps: self.rollout_steps,
            gradient_particles: self.gradient_particles,
            gradient_mode: self.gradient_mode,
            finite_diff_eps: self.finite_diff_eps,
            motion_gain: self.motion_gain,
            perception_position_gain: self.perception_position_gain,
            max_update_norm: self.max_update_norm,
            trajectory_supervision: self.trajectory_supervision,
            trajectory_render_gain: ROBUST_3D_TRAJECTORY_RENDER_GAIN,
            trajectory_mesh_gain: ROBUST_3D_TRAJECTORY_MESH_GAIN,
            trajectory_render_samples: ROBUST_3D_TRAJECTORY_RENDER_SAMPLES,
            liveness_gain: ROBUST_3D_LIVENESS_GAIN,
            liveness_front_radius: ROBUST_3D_LIVENESS_FRONT_RADIUS,
            liveness_update_multiplier: ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER,
            coverage_gain: ROBUST_3D_COVERAGE_GAIN,
            coverage_samples: ROBUST_3D_COVERAGE_SAMPLES,
            coverage_mode: CoverageUpdateModeArg::SlicedOt,
            coverage_softness: 0.0,
            coverage_repulsion_gain: ROBUST_3D_COVERAGE_REPULSION_GAIN,
            coverage_gap_gain: ROBUST_3D_COVERAGE_REPULSION_GAIN,
            coverage_repulsion_radius: 0.0,
            coverage_normal_weight: ROBUST_3D_COVERAGE_NORMAL_WEIGHT,
            extent_gain: ROBUST_3D_EXTENT_GAIN,
            full_coverage_adjoint: true,
            surface_gain: ROBUST_3D_SURFACE_GAIN,
            surface_escape_gain: ROBUST_3D_SURFACE_ESCAPE_GAIN,
            opacity_gain: ROBUST_3D_OPACITY_GAIN,
            material_liveness_gain: ROBUST_3D_MATERIAL_LIVENESS_GAIN,
            material_tail_gain: ROBUST_3D_MATERIAL_TAIL_GAIN,
            material_suppression_update_multiplier:
                ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
            material_max_opacity_update: ROBUST_3D_MATERIAL_MAX_OPACITY_UPDATE,
            scale_gain: ROBUST_3D_SCALE_GAIN,
            scale_budget_weight: ROBUST_3D_SCALE_BUDGET_WEIGHT,
            max_opacity_update: 0.05,
            direct_output_gradient_rms_cap: self.direct_output_gradient_rms_cap,
            direct_line_search: self.direct_line_search,
            direct_line_search_scales: self.direct_line_search_scales.clone(),
            direct_material_output_only: self.direct_material_output_only,
            training_backend: self.training_backend,
            weight_update_mode: phase.weight_update_mode,
            adapter_rank: self.adapter_rank,
            adapter_alpha: self.adapter_alpha,
            adapter_seed: phase.adapter_seed,
            direct_selection_seed_training: self.direct_selection_seed_training,
            seed: phase.seed,
            selection_seed: Some(self.selection_seed),
            selection_seeds: self.selection_seeds.clone(),
            seed_scale: phase.seed_scale,
            seed_mode: phase.seed_mode,
            render,
            sgd: self.sgd,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct AdapterSuiteTrainingPhaseConfig {
    pub(super) target: MeshTargetArg,
    pub(super) rounds: usize,
    pub(super) weight_update_mode: RenderWeightUpdateModeArg,
    pub(super) adapter_seed: u64,
    pub(super) seed: u64,
    pub(super) seed_scale: f32,
    pub(super) seed_mode: ParticleSeed,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render3d_adapter_suite_config_accepts_nested_toml() {
        let config: AdapterSuiteExperimentConfig = toml::from_str(
            r#"
            [input]
            base_model = "artifacts/render3d/shared_base.bpk"

            [shared_base]
            output = "artifacts/render3d_adapters/shared_base.bpk"
            cycles = 2
            seed = 5904189

            [targets]
            target_set = "many"
            targets = ["torus", "teapot"]
            holdout_targets = ["capsule"]
            auto_holdout_stride = 4
            auto_holdout_offset = 1

            [output]
            output_dir = "artifacts/render3d_adapters"
            report_output = "artifacts/render3d_adapters/report.json"
            adapter_bank_output = "artifacts/render3d_adapters/adapter_bank.json"

            [training]
            skip_shared_base_eval = true
            rounds = 3
            supervised_steps_per_round = 4
            particles = 96
            rollout_steps = 12
            gradient_particles = 24
            gradient_mode = "analytic"
            backend = "direct-rollout"

            [objective]
            direct_line_search = false
            direct_line_search_scales = [0.25, 0.5, 1.0]

            [optimizer]
            learning_rate = 0.001
            grad_clip_norm = 0.5

            [adapter]
            rank = 8
            alpha = 8.0
            seed = 11381043

            [seed]
            seed_mode = "uniform-circle"
            extra_selection_seeds = [42, 99]
            direct_selection_seed_training = false

            [render]
            gaussian_decode_mode = "fixed-sh0"
            image_size = 32

            [validation]
            fail_on_validation = false
            "#,
        )
        .unwrap();

        assert_eq!(config.shared_base.cycles, Some(2));
        assert!(config.training.skip_shared_base_eval.unwrap());
        assert_eq!(
            adapter_suite_config_value_enum_vec::<MeshTargetArg>(
                "targets.targets",
                config.targets.targets
            )
            .unwrap()
            .unwrap(),
            vec![MeshTargetArg::Torus, MeshTargetArg::Teapot]
        );
        assert_eq!(
            adapter_suite_config_value_enum(
                "targets.target_set",
                config.targets.target_set,
                MeshTargetSetArg::Core
            )
            .unwrap(),
            MeshTargetSetArg::Many
        );
        assert!(matches!(
            adapter_suite_config_value_enum_option::<SeedModeArg>(
                "seed.seed_mode",
                config.seed.seed_mode,
                None,
            )
            .unwrap(),
            Some(SeedModeArg::UniformCircle)
        ));
    }

    #[test]
    fn render3d_adapter_suite_bool_switch_override_is_authoritative() {
        assert_eq!(
            adapter_suite_override_bool_switch(Some(true), false, true),
            (true, false)
        );
        assert_eq!(
            adapter_suite_override_bool_switch(Some(false), true, false),
            (false, true)
        );
        assert_eq!(
            adapter_suite_override_bool_switch(None, true, false),
            (true, false)
        );
    }
}
