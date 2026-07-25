use serde::{Deserialize, Serialize};

use super::{
    AdaptiveController, AdaptiveHierarchyRestrictionPolicy, AdaptiveLocalRuleSemantics,
    AdaptiveNpaConfig, AdaptiveResidualGateReference,
};
#[cfg(feature = "gpu_wgpu")]
use crate::gpu::WgpuAdaptiveLocalRuleMode;
use crate::{AutomataError, AutomataResult, NpaModel, NpaWeights};

#[cfg(feature = "gpu_wgpu")]
pub(crate) struct AdaptiveGpuInferenceRule {
    pub rule: NpaModel,
    pub local_rule_mode: WgpuAdaptiveLocalRuleMode,
    pub local_hidden_start: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveNpaModel {
    pub config: AdaptiveNpaConfig,
    /// Shared task NPA. Residual curricula keep it frozen; full normalized
    /// curricula optimize it directly under unequal-measure perception.
    pub rule: NpaModel,
    /// Local coarse-graining correction. Zero initialization preserves the
    /// exact fixed-NPA limit before task multiscale training.
    #[serde(default)]
    pub local_residual_rule: Option<NpaModel>,
    /// Nonmaterial hierarchy context branch. Zero initialization preserves the
    /// local rule exactly before multiscale training.
    #[serde(default)]
    pub proxy_rule: Option<NpaModel>,
    /// Functionally distilled rule used by the resident represented-measure GPU
    /// executor. It consumes NPA-compatible features and predicts the complete
    /// local/proxy update in one wider MLP.
    #[serde(default)]
    pub deployment_rule: Option<NpaModel>,
    /// Deployable residual that consumes the normalized adaptive perception
    /// already produced by the fused local WGPU kernel. Unlike the flat rule,
    /// this preserves multiscale information instead of regressing the full
    /// policy from NPA-compatible features alone.
    #[serde(default)]
    pub deployment_local_rule: Option<NpaModel>,
    /// Recurrent affine-null closure head. Motion outputs evolve the unit
    /// geometry phase while state outputs evolve `AdaptiveParticleSet::closure_mode`
    /// independently from physical NPA state.
    #[serde(default)]
    pub closure_mode_rule: Option<NpaModel>,
    /// Recurrent geometry head for the four-child affine-null basis. The
    /// basis carries two compact shape degrees of freedom that are independent
    /// from covariance and phase. Legacy checkpoints omit this head and retain
    /// their original frozen-basis behavior until explicitly migrated.
    #[serde(default)]
    pub closure_basis_rule: Option<NpaModel>,
    pub controller: AdaptiveController,
    /// Dedicated target-independent scorer for scheduled hierarchy cuts. It is
    /// separate from the steady split/merge controller so restriction
    /// distillation cannot alter ordinary topology behavior.
    #[serde(default)]
    pub restriction_controller: Option<AdaptiveController>,
}

impl AdaptiveNpaModel {
    pub fn seeded(
        rule: NpaModel,
        config: AdaptiveNpaConfig,
        controller_seed: u64,
    ) -> AutomataResult<Self> {
        let controller = AdaptiveController::seeded(config.controller_hidden_dims, controller_seed);
        let restriction_controller = (config.hierarchical_restriction_policy
            == AdaptiveHierarchyRestrictionPolicy::LearnedController)
            .then(|| {
                AdaptiveController::seeded(
                    config.controller_hidden_dims,
                    controller_seed ^ 0x7265_7374_7269_6374,
                )
            });
        let proxy_rule = config.proxy.enabled.then(|| NpaModel {
            config: rule.config.clone(),
            weights: NpaWeights::zero_output_seeded(
                &rule.config,
                controller_seed ^ 0x7072_6f78_795f_6e70,
            ),
        });
        let closure_mode_rule = config.closure_recurrent_mode.then(|| {
            let mut closure_config = rule.config.clone();
            closure_config.auxiliary_input_dims =
                super::features::closure_recurrent_auxiliary_dims(&config, rule.config.state_dims);
            NpaModel {
                config: closure_config.clone(),
                weights: NpaWeights::zero_output_seeded(
                    &closure_config,
                    controller_seed ^ 0x636c_6f73_7572_655f,
                ),
            }
        });
        let closure_basis_rule = config.closure_recurrent_mode.then(|| {
            let mut closure_config = rule.config.clone();
            closure_config.auxiliary_input_dims =
                super::features::closure_recurrent_auxiliary_dims(&config, rule.config.state_dims);
            NpaModel {
                config: closure_config.clone(),
                weights: NpaWeights::zero_output_seeded(
                    &closure_config,
                    controller_seed ^ 0x6261_7369_735f_6e70,
                ),
            }
        });
        let local_residual_rule = config.compatible_residual_material_features.then(|| {
            let mut local_config = rule.config.clone();
            local_config.auxiliary_input_dims =
                super::features::local_residual_auxiliary_dims(&config, rule.config.state_dims);
            NpaModel {
                config: local_config.clone(),
                weights: NpaWeights::zero_output_seeded(
                    &local_config,
                    controller_seed ^ 0x6d61_7465_7269_616c,
                ),
            }
        });
        let model = Self {
            config,
            rule,
            local_residual_rule,
            proxy_rule,
            deployment_rule: None,
            deployment_local_rule: None,
            closure_mode_rule,
            closure_basis_rule,
            controller,
            restriction_controller,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn enable_zero_proxy_rule(&mut self) -> AutomataResult<()> {
        self.config.proxy.enabled = true;
        self.proxy_rule = Some(NpaModel {
            config: self.rule.config.clone(),
            weights: NpaWeights::zero_output_seeded(&self.rule.config, 0x7072_6f78_795f_6e70),
        });
        self.validate()
    }

    /// Inserts compact recurrent memory immediately before the RGB tail while
    /// preserving every source-rule input and physical output coordinate.
    pub fn enable_compact_recurrent_memory(&mut self, memory_dims: usize) -> AutomataResult<()> {
        if memory_dims == 0 {
            return self.validate();
        }
        if self.config.compact_recurrent_memory_dims == memory_dims {
            return self.validate();
        }
        if self.config.compact_recurrent_memory_dims != 0
            || self.local_residual_rule.is_some()
            || self.proxy_rule.is_some()
            || self.deployment_rule.is_some()
            || self.deployment_local_rule.is_some()
            || self.closure_mode_rule.is_some()
            || self.closure_basis_rule.is_some()
        {
            return Err(AutomataError::InvalidArgument(
                "compact recurrent memory must be inserted before auxiliary adaptive rules"
                    .to_owned(),
            ));
        }
        if memory_dims > 8 || self.rule.config.state_dims + memory_dims > 24 {
            return Err(AutomataError::InvalidArgument(format!(
                "compact recurrent memory requires 1..=8 channels and at most 24 total state channels, got {} + {memory_dims}",
                self.rule.config.state_dims,
            )));
        }
        self.rule = state_expanded_compatible_rule(&self.rule, memory_dims)?;
        self.config.compact_recurrent_memory_dims = memory_dims;
        self.validate()
    }

    /// State channels reserved for unresolved sub-leaf memory. The three RGB
    /// channels remain the final state channels for renderer compatibility.
    pub fn compact_recurrent_memory_range(&self) -> Option<std::ops::Range<usize>> {
        let memory_dims = self.config.compact_recurrent_memory_dims;
        let memory_end = self.rule.config.state_dims.checked_sub(3)?;
        let memory_start = memory_end.checked_sub(memory_dims)?;
        (memory_dims > 0).then_some(memory_start..memory_end)
    }

    pub fn enable_zero_local_residual_rule(&mut self) -> AutomataResult<()> {
        self.config.compatible_residual_material_features = false;
        self.enable_zero_local_residual_rule_with_current_features()
    }

    /// Adds a zero-output normalized residual with explicit local material
    /// scale and coarse-source exposure inputs.
    pub fn enable_material_conditioned_normalized_residual_rule(&mut self) -> AutomataResult<()> {
        self.config.local_rule_semantics = AdaptiveLocalRuleSemantics::NormalizedExposureResidual;
        self.config.compatible_residual_material_features = true;
        self.enable_zero_local_residual_rule_with_current_features()
    }

    fn enable_zero_local_residual_rule_with_current_features(&mut self) -> AutomataResult<()> {
        let mut local_config = self.rule.config.clone();
        local_config.auxiliary_input_dims = super::features::local_residual_auxiliary_dims(
            &self.config,
            self.rule.config.state_dims,
        );
        self.local_residual_rule = Some(NpaModel {
            config: local_config.clone(),
            weights: NpaWeights::zero_output_seeded(&local_config, 0x6c6f_6361_6c5f_6e70),
        });
        self.validate()
    }

    /// Adds a zero-output compatible residual with explicit local material
    /// scale and coarse-source exposure inputs.
    pub fn enable_material_conditioned_compatible_residual_rule(&mut self) -> AutomataResult<()> {
        self.config.local_rule_semantics = AdaptiveLocalRuleSemantics::CompatibleResidual;
        self.config.compatible_residual_material_features = true;
        let auxiliary_input_dims = super::features::local_residual_auxiliary_dims(
            &self.config,
            self.rule.config.state_dims,
        );
        self.local_residual_rule = Some(match self.local_residual_rule.take() {
            Some(rule) => input_expanded_compatible_rule(&rule, auxiliary_input_dims)?,
            None => {
                let mut local_config = self.rule.config.clone();
                local_config.auxiliary_input_dims = auxiliary_input_dims;
                NpaModel {
                    config: local_config.clone(),
                    weights: NpaWeights::zero_output_seeded(&local_config, 0x6d61_7465_7269_616c),
                }
            }
        });
        self.validate()
    }

    /// Widens the primary shared rule with one zero-initialized material-scale
    /// input. This is function preserving until the widened rule is trained.
    pub fn enable_material_scale_conditioning(&mut self) -> AutomataResult<()> {
        const MATERIAL_SCALE_DIMS: usize = 1;
        match self.rule.config.auxiliary_input_dims {
            0 => {
                self.rule = input_expanded_compatible_rule(&self.rule, MATERIAL_SCALE_DIMS)?;
            }
            MATERIAL_SCALE_DIMS => {}
            actual => {
                return Err(AutomataError::InvalidModel(format!(
                    "material-scale conditioning requires zero or one primary auxiliary input, got {actual}",
                )));
            }
        }
        self.config.material_scale_conditioning = true;
        self.validate()
    }

    pub(crate) fn uses_canonical_compatible_residual(&self) -> bool {
        self.local_residual_rule.is_some()
            && self.config.local_rule_semantics == AdaptiveLocalRuleSemantics::CompatibleResidual
            && self.config.residual_gate_reference == AdaptiveResidualGateReference::BaseRule
            && (self.config.local_residual_scale - 1.0).abs() <= 1.0e-6
            && (self.config.local_residual_motion_scale - 1.0).abs() <= 1.0e-6
            && (self.config.local_residual_state_scale - 1.0).abs() <= 1.0e-6
    }

    pub(crate) fn uses_canonical_normalized_residual(&self) -> bool {
        self.local_residual_rule.is_some()
            && self.config.rule_perception == super::AdaptiveRulePerception::NpaCompatible
            && self.config.local_rule_semantics
                == AdaptiveLocalRuleSemantics::NormalizedExposureResidual
            && self.config.residual_gate_reference == AdaptiveResidualGateReference::BaseRule
            && (self.config.local_residual_scale - 1.0).abs() <= 1.0e-6
            && (self.config.local_residual_motion_scale - 1.0).abs() <= 1.0e-6
            && (self.config.local_residual_state_scale - 1.0).abs() <= 1.0e-6
            && self.config.compatible_residual_material_features
    }

    /// Initializes the normalized local branch to the base NPA function. Any
    /// closure-only inputs begin with zero weights, so native-scale behavior is
    /// a well-defined starting point before coarse replacement training.
    pub(crate) fn enable_base_initialized_local_rule(&mut self) -> AutomataResult<()> {
        let auxiliary_input_dims = super::features::local_residual_auxiliary_dims(
            &self.config,
            self.rule.config.state_dims,
        );
        self.local_residual_rule = Some(input_expanded_compatible_rule(
            &self.rule,
            auxiliary_input_dims,
        )?);
        self.validate()
    }

    pub fn enable_seeded_restriction_controller(&mut self, seed: u64) -> AutomataResult<()> {
        self.restriction_controller = Some(AdaptiveController::seeded(
            self.config.controller_hidden_dims,
            seed,
        ));
        self.validate()
    }

    pub fn enable_zero_closure_mode_rule(&mut self) -> AutomataResult<()> {
        self.config.closure_moment_features = true;
        self.config.closure_recurrent_mode = true;
        if self.config.compatible_residual_material_features {
            let auxiliary_input_dims = super::features::local_residual_auxiliary_dims(
                &self.config,
                self.rule.config.state_dims,
            );
            self.local_residual_rule = Some(match self.local_residual_rule.take() {
                Some(rule) => input_expanded_compatible_rule(&rule, auxiliary_input_dims)?,
                None => {
                    let mut local_config = self.rule.config.clone();
                    local_config.auxiliary_input_dims = auxiliary_input_dims;
                    NpaModel {
                        config: local_config.clone(),
                        weights: NpaWeights::zero_output_seeded(
                            &local_config,
                            0x6d61_7465_7269_616c,
                        ),
                    }
                }
            });
        }
        let mut config = self.rule.config.clone();
        config.auxiliary_input_dims = super::features::closure_recurrent_auxiliary_dims(
            &self.config,
            self.rule.config.state_dims,
        );
        self.closure_mode_rule = Some(match self.closure_mode_rule.take() {
            Some(rule) => input_expanded_compatible_rule(&rule, config.auxiliary_input_dims)?,
            None => NpaModel {
                config: config.clone(),
                weights: NpaWeights::zero_output_seeded(&config, 0x636c_6f73_7572_655f),
            },
        });
        self.closure_basis_rule = Some(match self.closure_basis_rule.take() {
            Some(rule) => input_expanded_compatible_rule(&rule, config.auxiliary_input_dims)?,
            None => NpaModel {
                config: config.clone(),
                weights: NpaWeights::zero_output_seeded(&config, 0x6261_7369_735f_6e70),
            },
        });
        self.validate()
    }

    pub fn validate(&self) -> AutomataResult<()> {
        self.config.validate()?;
        self.rule.validate()?;
        let expected_primary_auxiliary_dims = usize::from(self.config.material_scale_conditioning);
        if self.rule.config.auxiliary_input_dims != expected_primary_auxiliary_dims {
            return Err(AutomataError::InvalidModel(format!(
                "adaptive primary rule has {} auxiliary inputs but material-scale conditioning requires {expected_primary_auxiliary_dims}",
                self.rule.config.auxiliary_input_dims,
            )));
        }
        if self.config.material_scale_conditioning
            && self.config.coarse_dynamics != super::AdaptiveCoarseDynamics::RepresentedMeasure
        {
            return Err(AutomataError::InvalidModel(
                "material-scale-conditioned shared rules require represented-measure coarse dynamics"
                .to_owned(),
            ));
        }
        if self.config.compatible_residual_material_features
            && (!matches!(
                self.config.local_rule_semantics,
                AdaptiveLocalRuleSemantics::CompatibleResidual
                    | AdaptiveLocalRuleSemantics::NormalizedExposureResidual
            ))
        {
            return Err(AutomataError::InvalidModel(
                "material-conditioned residuals require compatible or normalized-exposure semantics"
                    .to_owned(),
            ));
        }
        if let Some(local_residual_rule) = &self.local_residual_rule {
            local_residual_rule.validate()?;
            let mut normalized = local_residual_rule.config.clone();
            normalized.hidden_dims = self.rule.config.hidden_dims;
            normalized.auxiliary_input_dims = self.rule.config.auxiliary_input_dims;
            if normalized != self.rule.config {
                return Err(AutomataError::InvalidModel(
                    "adaptive local residual differs from the frozen base outside hidden width"
                        .to_string(),
                ));
            }
            let expected_local_auxiliary_dims = super::features::local_residual_auxiliary_dims(
                &self.config,
                self.rule.config.state_dims,
            );
            if local_residual_rule.config.auxiliary_input_dims != expected_local_auxiliary_dims {
                return Err(AutomataError::InvalidModel(format!(
                    "adaptive local residual has {} auxiliary inputs, expected {expected_local_auxiliary_dims}",
                    local_residual_rule.config.auxiliary_input_dims,
                )));
            }
        } else if self.config.compatible_residual_material_features {
            return Err(AutomataError::InvalidModel(
                "material-conditioned residual has no residual rule".to_owned(),
            ));
        }
        if let Some(proxy_rule) = &self.proxy_rule {
            proxy_rule.validate()?;
            if proxy_rule.config != self.rule.config {
                return Err(AutomataError::InvalidModel(
                    "adaptive proxy rule config differs from local rule".to_string(),
                ));
            }
        }
        if let Some(deployment_rule) = &self.deployment_rule {
            deployment_rule.validate()?;
            let mut normalized = deployment_rule.config.clone();
            normalized.hidden_dims = self.rule.config.hidden_dims;
            if normalized != self.rule.config {
                return Err(AutomataError::InvalidModel(
                    "adaptive deployment rule differs from the base rule outside hidden width"
                        .to_string(),
                ));
            }
        }
        if let Some(deployment_local_rule) = &self.deployment_local_rule {
            deployment_local_rule.validate()?;
            let mut normalized = deployment_local_rule.config.clone();
            normalized.hidden_dims = self.rule.config.hidden_dims;
            normalized.auxiliary_input_dims = self.rule.config.auxiliary_input_dims;
            if normalized != self.rule.config {
                return Err(AutomataError::InvalidModel(
                    "adaptive local deployment rule differs from the base rule outside hidden width"
                        .to_string(),
                ));
            }
        }
        if self.deployment_rule.is_some() && self.deployment_local_rule.is_some() {
            return Err(AutomataError::InvalidModel(
                "adaptive model cannot enable flat and fused-local deployment together".to_string(),
            ));
        }
        if self.config.closure_recurrent_mode != self.closure_mode_rule.is_some() {
            return Err(AutomataError::InvalidModel(
                "recurrent closure mode and closure rule must be enabled together".to_owned(),
            ));
        }
        if let Some(closure) = &self.closure_mode_rule {
            closure.validate()?;
            let mut normalized = closure.config.clone();
            normalized.hidden_dims = self.rule.config.hidden_dims;
            normalized.auxiliary_input_dims = self.rule.config.auxiliary_input_dims;
            if normalized != self.rule.config {
                return Err(AutomataError::InvalidModel(
                    "adaptive closure rule differs from the base outside hidden width and auxiliary inputs"
                        .to_owned(),
                ));
            }
        }
        if let Some(closure) = &self.closure_basis_rule {
            closure.validate()?;
            let mut normalized = closure.config.clone();
            normalized.hidden_dims = self.rule.config.hidden_dims;
            normalized.auxiliary_input_dims = self.rule.config.auxiliary_input_dims;
            if normalized != self.rule.config {
                return Err(AutomataError::InvalidModel(
                    "adaptive closure-basis rule differs from the base outside hidden width and auxiliary inputs"
                        .to_owned(),
                ));
            }
            if !self.config.closure_recurrent_mode {
                return Err(AutomataError::InvalidModel(
                    "adaptive closure-basis rule requires recurrent closure mode".to_owned(),
                ));
            }
        }
        if self.config.proxy.enabled != self.proxy_rule.is_some() {
            return Err(AutomataError::InvalidModel(
                "adaptive proxy execution and proxy rule must be enabled together".to_string(),
            ));
        }
        self.controller.validate()?;
        if let Some(controller) = &self.restriction_controller {
            controller.validate()?;
            if controller.hidden_dims != self.config.controller_hidden_dims {
                return Err(AutomataError::InvalidModel(format!(
                    "adaptive restriction controller hidden dims {} != configured {}",
                    controller.hidden_dims, self.config.controller_hidden_dims,
                )));
            }
        }
        if self.config.compact_recurrent_memory_dims > self.rule.config.state_dims.saturating_sub(3)
        {
            return Err(AutomataError::InvalidModel(format!(
                "compact recurrent memory has {} channels but the rule has only {} state channels",
                self.config.compact_recurrent_memory_dims, self.rule.config.state_dims,
            )));
        }
        if self.config.hierarchical_restriction_policy
            == AdaptiveHierarchyRestrictionPolicy::LearnedController
            && self.restriction_controller.is_none()
        {
            return Err(AutomataError::InvalidModel(
                "learned hierarchy restriction requires restriction_controller".to_string(),
            ));
        }
        if self.rule.config.spatial_dims != self.config.spatial_dims {
            return Err(AutomataError::InvalidModel(format!(
                "adaptive rule spatial dims {} != adaptive config {}",
                self.rule.config.spatial_dims, self.config.spatial_dims
            )));
        }
        let adaptive_features = self
            .config
            .perception
            .feature_dims(self.rule.config.state_dims)
            + usize::from(self.config.material_scale_conditioning);
        if adaptive_features != self.rule.config.perception_dims() {
            return Err(AutomataError::InvalidModel(format!(
                "adaptive perception dims {adaptive_features} != rule perception dims {}",
                self.rule.config.perception_dims()
            )));
        }
        if !self.rule.config.state_grad || !self.rule.config.density_grad {
            return Err(AutomataError::InvalidModel(
                "adaptive rule currently requires state_grad and density_grad".to_string(),
            ));
        }
        if self.config.perception.include_position_features != self.rule.config.position_features {
            return Err(AutomataError::InvalidModel(
                "adaptive and rule position-feature settings must match".to_string(),
            ));
        }
        Ok(())
    }

    pub(crate) fn expand_local_residual_rule(
        &mut self,
        hidden_dims: usize,
        seed: u64,
    ) -> AutomataResult<()> {
        let current = self.local_residual_rule.as_ref().ok_or_else(|| {
            AutomataError::InvalidModel(
                "adaptive local residual must be initialized before expansion".to_string(),
            )
        })?;
        if hidden_dims < current.config.hidden_dims {
            return Err(AutomataError::InvalidArgument(format!(
                "adaptive local residual width cannot shrink from {} to {hidden_dims}",
                current.config.hidden_dims,
            )));
        }
        if hidden_dims == current.config.hidden_dims {
            return Ok(());
        }
        self.local_residual_rule = Some(expanded_compatible_rule(current, hidden_dims, seed)?);
        self.validate()
    }

    pub fn inference_rule(&self) -> &NpaModel {
        self.deployment_rule.as_ref().unwrap_or(&self.rule)
    }

    pub(crate) fn uses_flat_deployment_rule(&self) -> bool {
        self.deployment_rule.is_some()
    }

    pub(crate) fn uses_local_deployment_rule(&self) -> bool {
        self.deployment_local_rule.is_some()
    }

    pub(crate) fn uses_deployment_rule(&self) -> bool {
        self.uses_flat_deployment_rule() || self.uses_local_deployment_rule()
    }

    /// Builds one packed MLP buffer for the GPU while retaining an exact
    /// per-particle residual gate in the shader. The trailing constant hidden
    /// unit carries the local branch output bias through that same gate.
    #[cfg(feature = "gpu_wgpu")]
    pub(crate) fn gpu_inference_rule(&self) -> AutomataResult<AdaptiveGpuInferenceRule> {
        if let Some(deployment) = &self.deployment_rule {
            if self.config.closure_recurrent_mode {
                return Err(AutomataError::InvalidModel(
                    "recurrent closure mode is incompatible with a flat deployment rule".to_owned(),
                ));
            }
            return Ok(AdaptiveGpuInferenceRule {
                rule: deployment.clone(),
                local_rule_mode: WgpuAdaptiveLocalRuleMode::Disabled,
                local_hidden_start: None,
            });
        }
        if self.config.rule_perception == super::AdaptiveRulePerception::NormalizedAdaptive {
            if self.config.local_residual_scale > f32::MIN_POSITIVE
                && (self.local_residual_rule.is_some() || self.deployment_local_rule.is_some())
            {
                return Err(AutomataError::InvalidModel(
                    "normalized-primary WGPU inference does not support an additional local residual"
                        .to_string(),
                ));
            }
            let rule = if self.config.closure_recurrent_mode {
                input_expanded_compatible_rule(
                    &self.rule,
                    super::features::local_residual_auxiliary_dims(
                        &self.config,
                        self.rule.config.state_dims,
                    ),
                )?
            } else {
                self.rule.clone()
            };
            return Ok(AdaptiveGpuInferenceRule {
                rule,
                local_rule_mode: WgpuAdaptiveLocalRuleMode::NormalizedPrimary,
                local_hidden_start: Some(0),
            });
        }
        // Match the CPU dynamics boundary: a serialized training branch with
        // zero deployment gain is inert and must not widen the base MLP or
        // trigger the extra normalized-perception pass.
        let closure_only = self.config.closure_recurrent_mode
            && self.config.local_residual_scale <= f32::MIN_POSITIVE;
        if self.config.local_residual_scale <= f32::MIN_POSITIVE && !closure_only {
            return Ok(AdaptiveGpuInferenceRule {
                rule: self.rule.clone(),
                local_rule_mode: WgpuAdaptiveLocalRuleMode::Disabled,
                local_hidden_start: None,
            });
        }
        let local = self
            .deployment_local_rule
            .as_ref()
            .or(self.local_residual_rule.as_ref())
            .or_else(|| {
                closure_only
                    .then_some(self.closure_mode_rule.as_ref())
                    .flatten()
            });
        let Some(local) = local else {
            return Ok(AdaptiveGpuInferenceRule {
                rule: self.inference_rule().clone(),
                local_rule_mode: WgpuAdaptiveLocalRuleMode::Disabled,
                local_hidden_start: None,
            });
        };
        let mut normalized_local = local.config.clone();
        normalized_local.hidden_dims = self.rule.config.hidden_dims;
        normalized_local.auxiliary_input_dims = self.rule.config.auxiliary_input_dims;
        if normalized_local != self.rule.config {
            return Err(AutomataError::InvalidModel(
                "adaptive local residual config differs from base rule".to_string(),
            ));
        }
        let base_input_dims = self.rule.config.perception_dims();
        let input_dims = local.config.perception_dims();
        let output_dims = self.rule.config.update_dims();
        let base_hidden = self.rule.config.hidden_dims;
        let local_hidden = local.config.hidden_dims;
        let packed_hidden = base_hidden + local_hidden + 1;
        let mut config = self.rule.config.clone();
        config.hidden_dims = packed_hidden;
        config.auxiliary_input_dims = local.config.auxiliary_input_dims;

        let mut weights = NpaWeights::zeros(&config);
        for hidden in 0..base_hidden {
            weights.w1[hidden * input_dims..hidden * input_dims + base_input_dims].copy_from_slice(
                &self.rule.weights.w1[hidden * base_input_dims..(hidden + 1) * base_input_dims],
            );
        }
        weights.w1[base_hidden * input_dims..(base_hidden + local_hidden) * input_dims]
            .copy_from_slice(&local.weights.w1);
        weights.b1[..base_hidden].copy_from_slice(&self.rule.weights.b1);
        weights.b1[base_hidden..base_hidden + local_hidden].copy_from_slice(&local.weights.b1);
        weights.b1[packed_hidden - 1] = 1.0;
        for output in 0..output_dims {
            let packed = output * packed_hidden;
            let base = output * base_hidden;
            let residual = output * local_hidden;
            weights.w2[packed..packed + base_hidden]
                .copy_from_slice(&self.rule.weights.w2[base..base + base_hidden]);
            let residual_output_scale = self.config.local_residual_output_scale(output);
            for hidden in 0..local_hidden {
                weights.w2[packed + base_hidden + hidden] =
                    residual_output_scale * local.weights.w2[residual + hidden];
            }
            weights.w2[packed + packed_hidden - 1] =
                residual_output_scale * local.weights.b2[output];
        }
        weights.b2.copy_from_slice(&self.rule.weights.b2);
        let rule = NpaModel { config, weights };
        rule.validate()?;
        Ok(AdaptiveGpuInferenceRule {
            rule,
            local_rule_mode: if closure_only {
                WgpuAdaptiveLocalRuleMode::Residual
            } else {
                match self.config.local_rule_semantics {
                    AdaptiveLocalRuleSemantics::Residual => WgpuAdaptiveLocalRuleMode::Residual,
                    AdaptiveLocalRuleSemantics::NormalizedExposureResidual => {
                        WgpuAdaptiveLocalRuleMode::NormalizedExposureResidual
                    }
                    AdaptiveLocalRuleSemantics::CompatibleResidual => {
                        WgpuAdaptiveLocalRuleMode::CompatibleResidual
                    }
                    AdaptiveLocalRuleSemantics::CoarseReplacement => {
                        WgpuAdaptiveLocalRuleMode::CoarseReplacement
                    }
                }
            },
            local_hidden_start: Some(base_hidden),
        })
    }
}

pub(crate) fn input_expanded_compatible_rule(
    base: &NpaModel,
    auxiliary_input_dims: usize,
) -> AutomataResult<NpaModel> {
    if auxiliary_input_dims < base.config.auxiliary_input_dims {
        return Err(AutomataError::InvalidArgument(format!(
            "compatible NPA input expansion cannot shrink auxiliary inputs from {} to {auxiliary_input_dims}",
            base.config.auxiliary_input_dims,
        )));
    }
    if auxiliary_input_dims == base.config.auxiliary_input_dims {
        return Ok(base.clone());
    }
    let mut config = base.config.clone();
    config.auxiliary_input_dims = auxiliary_input_dims;
    let base_input_dims = base.config.perception_dims();
    let input_dims = config.perception_dims();
    let hidden_dims = config.hidden_dims;
    let mut weights = NpaWeights::zeros(&config);
    for hidden in 0..hidden_dims {
        weights.w1[hidden * input_dims..hidden * input_dims + base_input_dims].copy_from_slice(
            &base.weights.w1[hidden * base_input_dims..(hidden + 1) * base_input_dims],
        );
    }
    weights.b1.copy_from_slice(&base.weights.b1);
    weights.w2.copy_from_slice(&base.weights.w2);
    weights.b2.copy_from_slice(&base.weights.b2);
    let rule = NpaModel { config, weights };
    rule.validate()?;
    Ok(rule)
}

fn expanded_state_channel(channel: usize, source_state_dims: usize, memory_dims: usize) -> usize {
    let rgb_start = source_state_dims - 3;
    if channel < rgb_start {
        channel
    } else {
        channel + memory_dims
    }
}

fn expanded_state_feature_index(
    config: &crate::NpaConfig,
    memory_dims: usize,
    source_index: usize,
) -> usize {
    let source_state_dims = config.state_dims;
    let target_state_dims = source_state_dims + memory_dims;
    if source_index < source_state_dims {
        return expanded_state_channel(source_index, source_state_dims, memory_dims);
    }
    if source_index < 2 * source_state_dims {
        let channel = source_index - source_state_dims;
        return target_state_dims + expanded_state_channel(channel, source_state_dims, memory_dims);
    }
    let source_gradient_start = 2 * source_state_dims;
    let source_gradient_dims =
        usize::from(config.state_grad) * source_state_dims * config.spatial_dims;
    if source_index < source_gradient_start + source_gradient_dims {
        let local = source_index - source_gradient_start;
        let channel = local / config.spatial_dims;
        let axis = local % config.spatial_dims;
        return 2 * target_state_dims
            + expanded_state_channel(channel, source_state_dims, memory_dims)
                * config.spatial_dims
            + axis;
    }
    let target_gradient_dims =
        usize::from(config.state_grad) * target_state_dims * config.spatial_dims;
    2 * target_state_dims + target_gradient_dims + source_index
        - source_gradient_start
        - source_gradient_dims
}

pub(crate) fn state_expanded_compatible_rule(
    base: &NpaModel,
    memory_dims: usize,
) -> AutomataResult<NpaModel> {
    if memory_dims == 0 {
        return Ok(base.clone());
    }
    if base.config.state_dims < 3 {
        return Err(AutomataError::InvalidArgument(
            "compact recurrent memory requires a three-channel RGB tail".to_owned(),
        ));
    }
    let source_state_dims = base.config.state_dims;
    let mut config = base.config.clone();
    config.state_dims += memory_dims;
    let source_input_dims = base.config.perception_dims();
    let input_dims = config.perception_dims();
    let hidden_dims = config.hidden_dims;
    let mut weights = NpaWeights::zeros(&config);
    for hidden in 0..hidden_dims {
        for source in 0..source_input_dims {
            let target = expanded_state_feature_index(&base.config, memory_dims, source);
            weights.w1[hidden * input_dims + target] =
                base.weights.w1[hidden * source_input_dims + source];
        }
    }
    weights.b1.copy_from_slice(&base.weights.b1);
    for axis in 0..config.spatial_dims {
        let source = axis * hidden_dims;
        let target = axis * hidden_dims;
        weights.w2[target..target + hidden_dims]
            .copy_from_slice(&base.weights.w2[source..source + hidden_dims]);
        weights.b2[axis] = base.weights.b2[axis];
    }
    for channel in 0..source_state_dims {
        let target_channel = expanded_state_channel(channel, source_state_dims, memory_dims);
        let source = (base.config.spatial_dims + channel) * hidden_dims;
        let target = (config.spatial_dims + target_channel) * hidden_dims;
        weights.w2[target..target + hidden_dims]
            .copy_from_slice(&base.weights.w2[source..source + hidden_dims]);
        weights.b2[config.spatial_dims + target_channel] =
            base.weights.b2[base.config.spatial_dims + channel];
    }
    let rule = NpaModel { config, weights };
    rule.validate()?;
    Ok(rule)
}

pub(crate) fn expanded_compatible_rule(
    base: &NpaModel,
    hidden_dims: usize,
    seed: u64,
) -> AutomataResult<NpaModel> {
    if hidden_dims < base.config.hidden_dims {
        return Err(AutomataError::InvalidArgument(format!(
            "compatible NPA expansion width {hidden_dims} is smaller than source width {}",
            base.config.hidden_dims,
        )));
    }
    let mut config = base.config.clone();
    config.hidden_dims = hidden_dims;
    let mut weights = NpaWeights::zero_output_seeded(&config, seed);
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let base_hidden_dims = base.config.hidden_dims;
    for hidden in 0..base_hidden_dims {
        let source = hidden * input_dims;
        let target = hidden * input_dims;
        weights.w1[target..target + input_dims]
            .copy_from_slice(&base.weights.w1[source..source + input_dims]);
        weights.b1[hidden] = base.weights.b1[hidden];
    }
    for output in 0..output_dims {
        let source = output * base_hidden_dims;
        let target = output * hidden_dims;
        weights.w2[target..target + base_hidden_dims]
            .copy_from_slice(&base.weights.w2[source..source + base_hidden_dims]);
        weights.b2[output] = base.weights.b2[output];
    }
    let rule = NpaModel { config, weights };
    rule.validate()?;
    Ok(rule)
}

#[cfg(test)]
mod expansion_tests {
    use super::*;
    use crate::NpaConfig;

    #[test]
    fn compact_recurrent_memory_preserves_source_function_and_rgb_tail() {
        let base = NpaModel::seeded(NpaConfig::growing_2d(), 19);
        let memory_dims = 8;
        let expanded = state_expanded_compatible_rule(&base, memory_dims).unwrap();
        assert_eq!(
            expanded.config.state_dims,
            base.config.state_dims + memory_dims
        );

        let rows = 3;
        let source_input_dims = base.config.perception_dims();
        let target_input_dims = expanded.config.perception_dims();
        let source_features = (0..rows * source_input_dims)
            .map(|index| index as f32 * 1.0e-4 - 0.2)
            .collect::<Vec<_>>();
        let mut target_features = vec![0.0; rows * target_input_dims];
        for row in 0..rows {
            for source in 0..source_input_dims {
                let target = expanded_state_feature_index(&base.config, memory_dims, source);
                target_features[row * target_input_dims + target] =
                    source_features[row * source_input_dims + source];
            }
        }
        let source_update = base.forward_update_from_features(&source_features).unwrap();
        let target_update = expanded
            .forward_update_from_features(&target_features)
            .unwrap();
        let source_output_dims = base.config.update_dims();
        let target_output_dims = expanded.config.update_dims();
        let memory_start = base.config.state_dims - 3;
        for row in 0..rows {
            for axis in 0..base.config.spatial_dims {
                assert!(
                    (source_update[row * source_output_dims + axis]
                        - target_update[row * target_output_dims + axis])
                        .abs()
                        < 1.0e-6
                );
            }
            for channel in 0..base.config.state_dims {
                let target_channel =
                    expanded_state_channel(channel, base.config.state_dims, memory_dims);
                assert!(
                    (source_update[row * source_output_dims + base.config.spatial_dims + channel]
                        - target_update[row * target_output_dims
                            + expanded.config.spatial_dims
                            + target_channel])
                        .abs()
                        < 1.0e-6
                );
            }
            let memory = &target_update[row * target_output_dims
                + expanded.config.spatial_dims
                + memory_start
                ..row * target_output_dims
                    + expanded.config.spatial_dims
                    + memory_start
                    + memory_dims];
            assert!(memory.iter().all(|value| *value == 0.0));
        }
    }

    #[test]
    fn local_residual_expansion_preserves_the_existing_function() {
        let base = NpaModel::seeded(NpaConfig::growing_2d(), 23);
        let mut model =
            AdaptiveNpaModel::seeded(base, super::super::AdaptiveNpaConfig::growing_2d(), 29)
                .unwrap();
        model.enable_zero_local_residual_rule().unwrap();
        let input_dims = model.rule.config.perception_dims();
        let features = (0..3 * input_dims)
            .map(|index| index as f32 * 1.0e-3 - 0.1)
            .collect::<Vec<_>>();
        let before = model
            .local_residual_rule
            .as_ref()
            .unwrap()
            .forward_update_from_features(&features)
            .unwrap();

        model.expand_local_residual_rule(191, 31).unwrap();
        let expanded = model.local_residual_rule.as_ref().unwrap();
        let after = expanded.forward_update_from_features(&features).unwrap();

        assert_eq!(expanded.config.hidden_dims, 191);
        assert_eq!(before, after);
    }

    #[test]
    fn material_feature_expansion_preserves_a_trained_compatible_residual() {
        let base = NpaModel::seeded(NpaConfig::growing_2d(), 33);
        let mut adaptive = super::super::AdaptiveNpaConfig::growing_2d();
        adaptive.local_rule_semantics = AdaptiveLocalRuleSemantics::CompatibleResidual;
        adaptive.proxy.enabled = false;
        let mut model = AdaptiveNpaModel::seeded(base, adaptive, 35).unwrap();
        model.enable_zero_local_residual_rule().unwrap();
        let local = model.local_residual_rule.as_mut().unwrap();
        for (index, bias) in local.weights.b2.iter_mut().enumerate() {
            *bias = index as f32 * 0.01 - 0.05;
        }
        let old_input_dims = local.config.perception_dims();
        let old_features = (0..3 * old_input_dims)
            .map(|index| (index as f32 * 0.013).cos())
            .collect::<Vec<_>>();
        let before = local.forward_update_from_features(&old_features).unwrap();

        model
            .enable_material_conditioned_compatible_residual_rule()
            .unwrap();
        let expanded = model.local_residual_rule.as_ref().unwrap();
        let new_input_dims = expanded.config.perception_dims();
        let mut new_features = vec![0.0; 3 * new_input_dims];
        for row in 0..3 {
            new_features[row * new_input_dims..row * new_input_dims + old_input_dims]
                .copy_from_slice(&old_features[row * old_input_dims..(row + 1) * old_input_dims]);
            new_features[row * new_input_dims + old_input_dims] = row as f32 - 1.0;
            new_features[row * new_input_dims + old_input_dims + 1] = 0.25 * row as f32;
        }
        let after = expanded
            .forward_update_from_features(&new_features)
            .unwrap();

        assert_eq!(new_input_dims, old_input_dims + 2);
        assert_eq!(before, after);
    }

    #[test]
    fn recurrent_closure_migration_preserves_the_static_residual_function() {
        let base = NpaModel::seeded(NpaConfig::growing_2d(), 37);
        let mut adaptive = super::super::AdaptiveNpaConfig::growing_2d();
        adaptive.local_rule_semantics = AdaptiveLocalRuleSemantics::CompatibleResidual;
        adaptive.compatible_residual_material_features = true;
        adaptive.closure_moment_features = true;
        adaptive.proxy.enabled = false;
        let mut model = AdaptiveNpaModel::seeded(base, adaptive, 39).unwrap();
        let local = model.local_residual_rule.as_mut().unwrap();
        for (index, bias) in local.weights.b2.iter_mut().enumerate() {
            *bias = index as f32 * 0.007 - 0.03;
        }
        let old_input_dims = local.config.perception_dims();
        let old_features = (0..3 * old_input_dims)
            .map(|index| (index as f32 * 0.011).sin())
            .collect::<Vec<_>>();
        let before = local.forward_update_from_features(&old_features).unwrap();

        model.config.closure_recurrent_mode = true;
        model.enable_zero_closure_mode_rule().unwrap();
        let expanded = model.local_residual_rule.as_ref().unwrap();
        let new_input_dims = expanded.config.perception_dims();
        let mut new_features = vec![0.0; 3 * new_input_dims];
        for row in 0..3 {
            new_features[row * new_input_dims..row * new_input_dims + old_input_dims]
                .copy_from_slice(&old_features[row * old_input_dims..(row + 1) * old_input_dims]);
            for column in old_input_dims..new_input_dims {
                new_features[row * new_input_dims + column] =
                    ((row * new_input_dims + column) as f32 * 0.019).cos();
            }
        }
        let after = expanded
            .forward_update_from_features(&new_features)
            .unwrap();
        let closure = model
            .closure_mode_rule
            .as_ref()
            .unwrap()
            .forward_update_from_features(&new_features)
            .unwrap();
        let closure_basis = model
            .closure_basis_rule
            .as_ref()
            .unwrap()
            .forward_update_from_features(&new_features)
            .unwrap();

        assert_eq!(
            new_input_dims,
            old_input_dims + 2 * (model.rule.config.state_dims + 6)
        );
        assert_eq!(before, after);
        assert!(closure.iter().all(|value| *value == 0.0));
        assert!(closure_basis.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn recurrent_closure_context_migration_preserves_trained_heads() {
        let base = NpaModel::seeded(NpaConfig::growing_2d(), 83);
        let mut adaptive = super::super::AdaptiveNpaConfig::growing_2d();
        adaptive.closure_moment_features = true;
        adaptive.closure_recurrent_mode = true;
        adaptive.proxy.enabled = false;
        let mut model = AdaptiveNpaModel::seeded(base, adaptive, 89).unwrap();
        let context_dims = model.rule.config.state_dims + 6;
        let target_auxiliary_dims = super::super::features::closure_recurrent_auxiliary_dims(
            &model.config,
            model.rule.config.state_dims,
        );
        let legacy_auxiliary_dims = target_auxiliary_dims - context_dims;
        let mut legacy_config = model.rule.config.clone();
        legacy_config.auxiliary_input_dims = legacy_auxiliary_dims;
        let mut mode = NpaModel::seeded(legacy_config.clone(), 97);
        let mut basis = NpaModel::seeded(legacy_config, 101);
        for (index, value) in mode.weights.b2.iter_mut().enumerate() {
            *value += 0.01 * index as f32;
        }
        for (index, value) in basis.weights.b2.iter_mut().enumerate() {
            *value -= 0.008 * index as f32;
        }
        let old_input_dims = mode.config.perception_dims();
        let old_features = (0..3 * old_input_dims)
            .map(|index| (index as f32 * 0.017).sin())
            .collect::<Vec<_>>();
        let expected_mode = mode.forward_update_from_features(&old_features).unwrap();
        let expected_basis = basis.forward_update_from_features(&old_features).unwrap();
        model.closure_mode_rule = Some(mode);
        model.closure_basis_rule = Some(basis);

        model.enable_zero_closure_mode_rule().unwrap();
        let new_input_dims = model
            .closure_mode_rule
            .as_ref()
            .unwrap()
            .config
            .perception_dims();
        let mut new_features = vec![0.0; 3 * new_input_dims];
        for row in 0..3 {
            new_features[row * new_input_dims..row * new_input_dims + old_input_dims]
                .copy_from_slice(&old_features[row * old_input_dims..(row + 1) * old_input_dims]);
            for column in old_input_dims..new_input_dims {
                new_features[row * new_input_dims + column] =
                    ((row * new_input_dims + column) as f32 * 0.023).cos();
            }
        }

        assert_eq!(new_input_dims, old_input_dims + context_dims);
        assert_eq!(
            model
                .closure_mode_rule
                .as_ref()
                .unwrap()
                .forward_update_from_features(&new_features)
                .unwrap(),
            expected_mode,
        );
        assert_eq!(
            model
                .closure_basis_rule
                .as_ref()
                .unwrap()
                .forward_update_from_features(&new_features)
                .unwrap(),
            expected_basis,
        );
    }

    #[test]
    fn seeded_material_conditioned_compatible_model_is_immediately_valid() {
        let base = NpaModel::seeded(NpaConfig::growing_2d(), 41);
        let mut adaptive = super::super::AdaptiveNpaConfig::growing_2d();
        adaptive.local_rule_semantics = AdaptiveLocalRuleSemantics::CompatibleResidual;
        adaptive.compatible_residual_material_features = true;
        adaptive.proxy.enabled = false;
        let model = AdaptiveNpaModel::seeded(base.clone(), adaptive, 43).unwrap();

        assert_eq!(
            model
                .local_residual_rule
                .as_ref()
                .unwrap()
                .config
                .perception_dims(),
            base.config.perception_dims() + 2,
        );
    }

    #[test]
    fn base_initialized_local_rule_preserves_the_base_function() {
        let base = NpaModel::seeded(NpaConfig::growing_2d(), 41);
        let mut adaptive = super::super::AdaptiveNpaConfig::growing_2d();
        adaptive.closure_moment_features = true;
        let mut model = AdaptiveNpaModel::seeded(base.clone(), adaptive, 43).unwrap();
        model.enable_base_initialized_local_rule().unwrap();
        let base_input_dims = base.config.perception_dims();
        let local_input_dims = model
            .local_residual_rule
            .as_ref()
            .unwrap()
            .config
            .perception_dims();
        let base_features = (0..5 * base_input_dims)
            .map(|index| (index as f32 * 0.017).sin())
            .collect::<Vec<_>>();
        let mut local_features = vec![0.0; 5 * local_input_dims];
        for row in 0..5 {
            local_features[row * local_input_dims..row * local_input_dims + base_input_dims]
                .copy_from_slice(
                    &base_features[row * base_input_dims..(row + 1) * base_input_dims],
                );
        }

        assert_eq!(
            base.forward_update_from_features(&base_features).unwrap(),
            model
                .local_residual_rule
                .as_ref()
                .unwrap()
                .forward_update_from_features(&local_features)
                .unwrap(),
        );
    }

    #[test]
    fn material_scale_conditioning_preserves_the_native_rule_at_initialization() {
        let base = NpaModel::seeded(NpaConfig::growing_2d(), 47);
        let mut model = AdaptiveNpaModel::seeded(
            base.clone(),
            super::super::AdaptiveNpaConfig::growing_2d(),
            53,
        )
        .unwrap();
        let rows = 5;
        let base_input_dims = base.config.perception_dims();
        let base_features = (0..rows * base_input_dims)
            .map(|index| (index as f32 * 0.019).cos())
            .collect::<Vec<_>>();
        let expected = base.forward_update_from_features(&base_features).unwrap();

        model.enable_material_scale_conditioning().unwrap();
        let conditioned_input_dims = model.rule.config.perception_dims();
        let mut conditioned_features = Vec::with_capacity(rows * conditioned_input_dims);
        for row in 0..rows {
            conditioned_features.extend_from_slice(
                &base_features[row * base_input_dims..(row + 1) * base_input_dims],
            );
            conditioned_features.push(0.0);
        }

        assert!(model.config.material_scale_conditioning);
        assert_eq!(
            model.rule.config.auxiliary_input_dims,
            base.config.auxiliary_input_dims + 1,
        );
        assert_eq!(
            expected,
            model
                .rule
                .forward_update_from_features(&conditioned_features)
                .unwrap(),
        );
    }
}

#[cfg(all(test, feature = "gpu_wgpu"))]
mod tests {
    use super::*;
    use crate::NpaConfig;

    #[test]
    fn gpu_prefers_distilled_deployment_over_exact_training_branches() {
        let base = NpaModel::seeded(NpaConfig::growing_2d(), 3);
        let mut model = AdaptiveNpaModel::seeded(
            base.clone(),
            super::super::AdaptiveNpaConfig::growing_2d(),
            5,
        )
        .unwrap();
        model.enable_zero_local_residual_rule().unwrap();
        let deployment = NpaModel::seeded(base.config.clone(), 11);
        model.deployment_rule = Some(deployment.clone());
        model.validate().unwrap();

        let gpu = model.gpu_inference_rule().unwrap();
        assert_eq!(gpu.rule.weights.w1, deployment.weights.w1);
        assert_eq!(gpu.rule.weights.b1, deployment.weights.b1);
        assert_eq!(gpu.rule.weights.w2, deployment.weights.w2);
        assert_eq!(gpu.rule.weights.b2, deployment.weights.b2);
        assert_eq!(gpu.local_rule_mode, WgpuAdaptiveLocalRuleMode::Disabled);
        assert_eq!(gpu.local_hidden_start, None);
    }

    #[test]
    fn gpu_ignores_serialized_local_branch_when_its_runtime_gain_is_zero() {
        let base = NpaModel::seeded(NpaConfig::growing_2d(), 7);
        let mut adaptive = super::super::AdaptiveNpaConfig::growing_2d();
        adaptive.local_residual_scale = 0.0;
        let mut model = AdaptiveNpaModel::seeded(base.clone(), adaptive, 9).unwrap();
        model.enable_zero_local_residual_rule().unwrap();
        model.expand_local_residual_rule(191, 11).unwrap();

        let gpu = model.gpu_inference_rule().unwrap();
        assert_eq!(gpu.local_rule_mode, WgpuAdaptiveLocalRuleMode::Disabled);
        assert_eq!(gpu.local_hidden_start, None);
        assert_eq!(gpu.rule.config.hidden_dims, base.config.hidden_dims);
        assert_eq!(gpu.rule.weights.w1, base.weights.w1);
        assert_eq!(gpu.rule.weights.b1, base.weights.b1);
        assert_eq!(gpu.rule.weights.w2, base.weights.w2);
        assert_eq!(gpu.rule.weights.b2, base.weights.b2);
    }

    #[test]
    fn gpu_packs_fused_local_deployment_at_the_normalized_feature_boundary() {
        let base = NpaModel::seeded(NpaConfig::growing_2d(), 13);
        let mut model = AdaptiveNpaModel::seeded(
            base.clone(),
            super::super::AdaptiveNpaConfig::growing_2d(),
            17,
        )
        .unwrap();
        model.enable_zero_local_residual_rule().unwrap();
        let mut local_config = base.config.clone();
        local_config.hidden_dims = 191;
        model.deployment_local_rule = Some(NpaModel::seeded(local_config, 19));
        model.validate().unwrap();

        let gpu = model.gpu_inference_rule().unwrap();
        assert_eq!(gpu.local_rule_mode, WgpuAdaptiveLocalRuleMode::Residual);
        assert_eq!(gpu.local_hidden_start, Some(base.config.hidden_dims));
        assert_eq!(gpu.rule.config.hidden_dims, 320);
        assert!(model.uses_deployment_rule());
        assert!(!model.uses_flat_deployment_rule());
    }

    #[test]
    fn gpu_executes_normalized_primary_rule_at_the_normalized_feature_boundary() {
        let base = NpaModel::seeded(NpaConfig::growing_2d(), 23);
        let mut adaptive = super::super::AdaptiveNpaConfig::growing_2d();
        adaptive.rule_perception = super::super::AdaptiveRulePerception::NormalizedAdaptive;
        adaptive.local_residual_scale = 0.0;
        let model = AdaptiveNpaModel::seeded(base.clone(), adaptive, 29).unwrap();

        let gpu = model.gpu_inference_rule().unwrap();
        assert_eq!(
            gpu.local_rule_mode,
            WgpuAdaptiveLocalRuleMode::NormalizedPrimary
        );
        assert_eq!(gpu.local_hidden_start, Some(0));
        assert_eq!(gpu.rule.weights.w1, base.weights.w1);
        assert_eq!(gpu.rule.weights.b1, base.weights.b1);
        assert_eq!(gpu.rule.weights.w2, base.weights.w2);
        assert_eq!(gpu.rule.weights.b2, base.weights.b2);
    }

    #[test]
    fn gpu_packs_coarse_replacement_without_changing_the_base_branch() {
        let base = NpaModel::seeded(NpaConfig::growing_2d(), 31);
        let mut adaptive = super::super::AdaptiveNpaConfig::growing_2d();
        adaptive.local_rule_semantics = AdaptiveLocalRuleSemantics::CoarseReplacement;
        adaptive.proxy.enabled = false;
        let mut model = AdaptiveNpaModel::seeded(base.clone(), adaptive, 37).unwrap();
        model.enable_base_initialized_local_rule().unwrap();
        model.expand_local_residual_rule(191, 39).unwrap();

        let gpu = model.gpu_inference_rule().unwrap();
        assert_eq!(
            gpu.local_rule_mode,
            WgpuAdaptiveLocalRuleMode::CoarseReplacement
        );
        assert_eq!(gpu.local_hidden_start, Some(base.config.hidden_dims));
        assert_eq!(gpu.rule.config.hidden_dims, 320);
        for hidden in 0..base.config.hidden_dims {
            let packed = hidden * gpu.rule.config.perception_dims();
            let source = hidden * base.config.perception_dims();
            assert_eq!(
                &gpu.rule.weights.w1[packed..packed + base.config.perception_dims()],
                &base.weights.w1[source..source + base.config.perception_dims()],
            );
        }
    }

    #[test]
    fn gpu_packs_compatible_residual_into_the_native_perception_rule() {
        let base = NpaModel::seeded(NpaConfig::growing_2d(), 47);
        let mut adaptive = super::super::AdaptiveNpaConfig::growing_2d();
        adaptive.local_rule_semantics = AdaptiveLocalRuleSemantics::CompatibleResidual;
        adaptive.proxy.enabled = false;
        let mut model = AdaptiveNpaModel::seeded(base.clone(), adaptive, 53).unwrap();
        model.enable_zero_local_residual_rule().unwrap();

        assert!(model.uses_canonical_compatible_residual());
        let gpu = model.gpu_inference_rule().unwrap();
        assert_eq!(
            gpu.local_rule_mode,
            WgpuAdaptiveLocalRuleMode::CompatibleResidual
        );
        assert_eq!(gpu.local_hidden_start, Some(base.config.hidden_dims));
        assert_eq!(
            gpu.rule.config.perception_dims(),
            base.config.perception_dims()
        );
        assert_eq!(gpu.rule.config.hidden_dims, 2 * base.config.hidden_dims + 1);
    }

    #[test]
    fn gpu_packs_material_conditioned_compatible_residual_inputs() {
        let base = NpaModel::seeded(NpaConfig::growing_2d(), 59);
        let mut adaptive = super::super::AdaptiveNpaConfig::growing_2d();
        adaptive.proxy.enabled = false;
        let mut model = AdaptiveNpaModel::seeded(base.clone(), adaptive, 61).unwrap();
        model
            .enable_material_conditioned_compatible_residual_rule()
            .unwrap();

        let gpu = model.gpu_inference_rule().unwrap();
        assert_eq!(
            gpu.local_rule_mode,
            WgpuAdaptiveLocalRuleMode::CompatibleResidual
        );
        assert_eq!(gpu.local_hidden_start, Some(base.config.hidden_dims));
        assert_eq!(
            gpu.rule.config.perception_dims(),
            base.config.perception_dims() + 2,
        );
        assert_eq!(gpu.rule.config.hidden_dims, 2 * base.config.hidden_dims + 1);
    }

    #[test]
    fn gpu_packs_material_conditioned_static_closure_moments() {
        let base = NpaModel::seeded(NpaConfig::growing_2d(), 63);
        let mut adaptive = super::super::AdaptiveNpaConfig::growing_2d();
        adaptive.proxy.enabled = false;
        adaptive.local_rule_semantics = AdaptiveLocalRuleSemantics::CompatibleResidual;
        adaptive.compatible_residual_material_features = true;
        adaptive.closure_moment_features = true;
        let model = AdaptiveNpaModel::seeded(base.clone(), adaptive, 65).unwrap();

        assert!(model.uses_canonical_compatible_residual());
        let gpu = model.gpu_inference_rule().unwrap();
        let static_moment_dims = 1 + 3 + base.config.state_dims * 2;
        assert_eq!(
            gpu.rule.config.perception_dims(),
            base.config.perception_dims() + 2 + static_moment_dims,
        );
        assert_eq!(gpu.local_hidden_start, Some(base.config.hidden_dims));
    }

    #[test]
    fn gpu_packs_material_conditioned_normalized_residual_inputs() {
        let base = NpaModel::seeded(NpaConfig::growing_2d(), 67);
        let mut adaptive = super::super::AdaptiveNpaConfig::growing_2d();
        adaptive.proxy.enabled = false;
        let mut model = AdaptiveNpaModel::seeded(base.clone(), adaptive, 71).unwrap();
        model
            .enable_material_conditioned_normalized_residual_rule()
            .unwrap();

        assert!(model.uses_canonical_normalized_residual());
        let gpu = model.gpu_inference_rule().unwrap();
        assert_eq!(
            gpu.local_rule_mode,
            WgpuAdaptiveLocalRuleMode::NormalizedExposureResidual
        );
        assert_eq!(gpu.local_hidden_start, Some(base.config.hidden_dims));
        assert_eq!(
            gpu.rule.config.perception_dims(),
            base.config.perception_dims() + 2,
        );
        assert_eq!(gpu.rule.config.hidden_dims, 2 * base.config.hidden_dims + 1);
    }
}
