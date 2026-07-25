use serde::{Deserialize, Serialize};

use super::{
    ADAPTIVE_CONTROLLER_INPUT_DIMS, ADAPTIVE_CONTROLLER_OUTPUT_DIMS, AdaptiveNpaConfig,
    AdaptiveNpaModel, AdaptiveParticleSet, AdaptiveRenderDecoder, AdaptiveTopologyControl,
};
use crate::{AdamWConfig, AutomataError, AutomataResult};

mod closure_audit;
mod closure_mode_rule;
mod dataset;
mod deployment_on_policy;
mod deployment_rule;
mod dual_mlp;
mod mlp;
mod multiscale_dataset;
mod multiscale_on_policy;
mod multiscale_rule;
mod recurrent_replay;
mod restriction_dataset;
mod rule;
mod rule_dataset;
mod target2d;
mod tensor;

pub use closure_audit::{
    AdaptiveClosureAuditBackend, AdaptiveClosureIdentifiabilityConfig,
    AdaptiveClosureIdentifiabilityReport, audit_adaptive_closure_identifiability,
    audit_adaptive_closure_identifiability_wgpu,
};
pub use closure_mode_rule::{
    adaptive_closure_mode_validation, train_adaptive_closure_mode_rule_cuda,
    train_adaptive_closure_mode_rule_ndarray, train_adaptive_closure_mode_rule_wgpu,
};
pub use dataset::adaptive_oracle_training_batch;
pub use deployment_on_policy::adaptive_deployment_on_policy_batch_wgpu;
pub use deployment_rule::{
    adaptive_deployment_rule_validation, train_adaptive_deployment_rule_cuda,
    train_adaptive_deployment_rule_ndarray, train_adaptive_deployment_rule_wgpu,
};
pub use multiscale_dataset::adaptive_multiscale_training_batch;
#[cfg(feature = "gpu_wgpu")]
pub use multiscale_dataset::adaptive_multiscale_training_batch_wgpu_with_executor;
pub use multiscale_on_policy::adaptive_multiscale_on_policy_batch;
#[cfg(feature = "gpu_wgpu")]
pub use multiscale_on_policy::adaptive_multiscale_on_policy_batch_wgpu_with_executor;
pub use multiscale_rule::{
    adaptive_multiscale_rule_validation, adaptive_multiscale_rule_validation_cuda,
    adaptive_multiscale_rule_validation_ndarray, adaptive_multiscale_rule_validation_wgpu,
    train_adaptive_multiscale_rule_cuda, train_adaptive_multiscale_rule_ndarray,
    train_adaptive_multiscale_rule_wgpu,
};
#[cfg(any(feature = "backend_cuda", feature = "backend_wgpu"))]
pub use restriction_dataset::adaptive_restriction_training_batch_burn;
#[cfg(all(
    feature = "gpu_wgpu",
    any(feature = "backend_cuda", feature = "backend_wgpu")
))]
pub use restriction_dataset::adaptive_restriction_training_batch_burn_with_executor;
pub use restriction_dataset::{
    adaptive_restriction_training_batch, validate_adaptive_restriction_selection,
};
#[cfg(feature = "backend_cuda")]
pub use rule::train_adaptive_rule_cuda;
#[cfg(feature = "backend_ndarray")]
pub use rule::train_adaptive_rule_ndarray;
#[cfg(feature = "backend_wgpu")]
pub use rule::train_adaptive_rule_wgpu;
pub use rule_dataset::{adaptive_rule_distillation_batch, adaptive_rule_on_policy_batch};
#[cfg(feature = "gpu_wgpu")]
pub(crate) use target2d::adaptive_target2d_seed_particles;
pub use target2d::{
    AdaptiveTarget2dGpuTrainingReport, AdaptiveTarget2dMaterialConfig,
    AdaptiveTarget2dMaterialLayout, AdaptiveTarget2dRuleTraining, AdaptiveTarget2dTopologyConfig,
    AdaptiveTarget2dTrainingConfig, train_adaptive_target_2d_gpu,
};
pub(crate) use target2d::{
    AdaptiveTarget2dSeedBank, AdaptiveTarget2dUpdateMask, build_adaptive_target2d_seed_bank,
};
#[cfg(feature = "backend_cuda")]
pub use tensor::{
    train_adaptive_controller_cuda, train_adaptive_restriction_controller_cuda,
    validate_adaptive_controller_cuda,
};

/// Applies the same represented-measure bandwidth law used by the adaptive
/// Target2D trainer. A material row carrying `k` fine units has a footprint and
/// interaction bandwidth scaled by `sqrt(k)` in 2D.
fn apply_multiscale_material_bandwidth(
    adaptive: &AdaptiveNpaConfig,
    config: &AdaptiveMultiscaleTrainingConfig,
    particles: &mut AdaptiveParticleSet,
) -> AutomataResult<()> {
    let reference = adaptive.reference_footprint.max(f32::MIN_POSITIVE);
    for row in 0..particles.len() {
        particles.bandwidth[row] = (config.bandwidth * particles.footprint(row) / reference).clamp(
            adaptive.perception.min_bandwidth,
            adaptive.perception.max_bandwidth,
        );
    }
    particles.validate()
}

fn normalize_positive_weights(weights: &mut [f32], label: &str) -> AutomataResult<()> {
    if weights.is_empty()
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(AutomataError::InvalidArgument(format!(
            "{label} weights must be non-empty, finite, and non-negative"
        )));
    }
    let sum = weights.iter().sum::<f32>();
    if !sum.is_finite() || sum <= f32::MIN_POSITIVE {
        return Err(AutomataError::InvalidArgument(format!(
            "{label} weights must have a positive finite sum"
        )));
    }
    let mean = sum / weights.len() as f32;
    for weight in weights {
        *weight /= mean;
    }
    Ok(())
}
#[cfg(feature = "backend_ndarray")]
pub use tensor::{
    train_adaptive_controller_ndarray, train_adaptive_restriction_controller_ndarray,
    validate_adaptive_controller_ndarray,
};
#[cfg(feature = "backend_wgpu")]
pub use tensor::{
    train_adaptive_controller_wgpu, train_adaptive_restriction_controller_wgpu,
    validate_adaptive_controller_wgpu,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveControllerTrainingBatch {
    pub features: Vec<f32>,
    pub targets: Vec<f32>,
    pub rows: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveRestrictionDatasetConfig {
    pub seeds: Vec<u64>,
    pub cut_steps: Vec<usize>,
    pub update_prob: f32,
    pub seed_scale: f32,
    pub total_measure: f32,
    pub bandwidth: f32,
    #[serde(default)]
    pub bandwidth_adaptation_enabled: bool,
    /// Decoder whose composited image loss defines merge-oracle labels. The
    /// default is the deployable one-isotropic-Gaussian-per-leaf contract.
    #[serde(default)]
    pub render_decoder: AdaptiveRenderDecoder,
    #[serde(default = "default_restriction_render_compactness")]
    pub render_compactness: f32,
    /// Image used by the counterfactual merge oracle. Fine-teacher
    /// preservation is deployable without privileged target access and is the
    /// canonical parity objective; target-image remains an explicit ablation.
    #[serde(default)]
    pub label_target: AdaptiveRestrictionLabelTarget,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdaptiveRestrictionLabelTarget {
    #[default]
    FineTeacher,
    TargetImage,
}

impl Default for AdaptiveRestrictionDatasetConfig {
    fn default() -> Self {
        Self {
            seeds: (42..50).collect(),
            cut_steps: vec![128, 160, 192, 224, 256],
            update_prob: 0.5,
            seed_scale: 0.2,
            total_measure: std::f32::consts::PI * 0.2 * 0.2,
            bandwidth: 0.1,
            bandwidth_adaptation_enabled: false,
            render_decoder: AdaptiveRenderDecoder::default(),
            render_compactness: default_restriction_render_compactness(),
            label_target: AdaptiveRestrictionLabelTarget::default(),
        }
    }
}

const fn default_restriction_render_compactness() -> f32 {
    1.0
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveRestrictionTrainingBatch {
    pub controller: AdaptiveControllerTrainingBatch,
    /// Signed within-snapshot oracle rank in [-1, 1], where larger values are
    /// less destructive merges. Unlike raw merge costs this is invariant to
    /// image-energy scale and preserves the complete budget ordering.
    pub oracle_rank_targets: Vec<f32>,
    /// Robustly normalized target-render utility in [-1, 1]. Larger values
    /// identify less destructive merges while preserving the relative cost
    /// gaps that signed ranks intentionally discard.
    pub oracle_cost_utility_targets: Vec<f32>,
    pub snapshots: usize,
    pub groups_per_snapshot: usize,
    pub merges_per_snapshot: usize,
}

impl AdaptiveRestrictionTrainingBatch {
    pub fn validate(&self) -> AutomataResult<()> {
        self.controller.validate()?;
        if self.snapshots == 0
            || self.groups_per_snapshot == 0
            || self.controller.rows != self.snapshots * self.groups_per_snapshot
            || self.oracle_rank_targets.len() != self.controller.rows
            || self.oracle_cost_utility_targets.len() != self.controller.rows
            || self.merges_per_snapshot > self.groups_per_snapshot
            || self
                .oracle_rank_targets
                .iter()
                .chain(&self.oracle_cost_utility_targets)
                .any(|target| !target.is_finite() || !(-1.0..=1.0).contains(target))
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive restriction training batch shape mismatch".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveRestrictionDatasetReport {
    pub label_backend: String,
    pub seeds: usize,
    pub snapshots: usize,
    pub rows: usize,
    pub groups_per_snapshot: usize,
    pub merges_per_snapshot: usize,
    pub positive_fraction: f32,
    pub generation_ms: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveRestrictionSelectionReport {
    pub snapshots: usize,
    pub rows: usize,
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub intersection_over_union: f32,
    pub exact_cut_fraction: f32,
    /// Mean excess oracle merge cost of the deployed top-k cut, normalized by
    /// the full [-1, 1] utility range and selected row count.
    #[serde(default)]
    pub mean_normalized_cost_regret: f32,
    /// Worst per-snapshot normalized excess merge cost.
    #[serde(default)]
    pub worst_normalized_cost_regret: f32,
}

impl AdaptiveControllerTrainingBatch {
    pub fn validate(&self) -> AutomataResult<()> {
        if self.rows == 0
            || self.features.len() != self.rows * ADAPTIVE_CONTROLLER_INPUT_DIMS
            || self.targets.len() != self.rows * ADAPTIVE_CONTROLLER_OUTPUT_DIMS
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive controller training batch shape mismatch".to_string(),
            ));
        }
        if self
            .features
            .iter()
            .chain(&self.targets)
            .any(|value| !value.is_finite())
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive controller training batch contains non-finite values".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AdaptiveOracleDatasetConfig {
    pub rows: usize,
    pub spatial_dims: usize,
    pub seed: u64,
    pub reference_footprint: f32,
    pub min_footprint: f32,
    pub max_footprint: f32,
    pub total_measure: f32,
    pub target_leaf_count: usize,
    pub boundary_epsilon: f32,
    pub boundary_slope: f32,
}

impl Default for AdaptiveOracleDatasetConfig {
    fn default() -> Self {
        Self {
            rows: 65_536,
            spatial_dims: 2,
            seed: 42,
            reference_footprint: 0.025,
            min_footprint: 0.0015625,
            max_footprint: 0.1,
            total_measure: std::f32::consts::PI * 0.2 * 0.2,
            target_leaf_count: 4_096,
            boundary_epsilon: 0.00625,
            boundary_slope: 0.35,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AdaptiveControllerTrainConfig {
    #[serde(default = "default_controller_training_enabled")]
    pub enabled: bool,
    pub steps: usize,
    pub report_interval: usize,
    #[serde(default = "default_gradient_reduction_chunk_rows")]
    pub gradient_reduction_chunk_rows: usize,
    /// Rows materialized by one optimizer step. Zero uses the complete dataset.
    /// Non-zero values must divide the dataset row count so deterministic
    /// cycling never splits or drops a training row.
    #[serde(default)]
    pub optimizer_batch_rows: usize,
    /// Additional row weight centered on the deployed top-k restriction
    /// boundary. This is used only by the hierarchy restriction ranker; zero
    /// preserves uniform signed-rank regression.
    #[serde(default)]
    pub restriction_rank_boundary_emphasis: f32,
    /// Standard deviation of the Gaussian boundary emphasis in signed-rank
    /// coordinates. Signed oracle ranks span [-1, 1].
    #[serde(default = "default_restriction_rank_boundary_width")]
    pub restriction_rank_boundary_width: f32,
    /// Weight of the deployed top-k boundary classification term. The
    /// restriction controller still regresses the complete signed rank, while
    /// this term directly penalizes inversions across the exact merge cut.
    #[serde(default)]
    pub restriction_topk_loss_weight: f32,
    /// Logistic temperature in signed-rank coordinates for the top-k term.
    #[serde(default = "default_restriction_topk_temperature")]
    pub restriction_topk_temperature: f32,
    /// Blend from signed-rank supervision (0) to robust target-render cost
    /// utility (1). This preserves exact ordering while retaining the severity
    /// of bad merge decisions.
    #[serde(default)]
    pub restriction_cost_utility_weight: f32,
    pub optimizer: AdamWConfig,
}

impl Default for AdaptiveControllerTrainConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            steps: 2_000,
            report_interval: 100,
            gradient_reduction_chunk_rows: default_gradient_reduction_chunk_rows(),
            optimizer_batch_rows: 0,
            restriction_rank_boundary_emphasis: 0.0,
            restriction_rank_boundary_width: default_restriction_rank_boundary_width(),
            restriction_topk_loss_weight: 0.0,
            restriction_topk_temperature: default_restriction_topk_temperature(),
            restriction_cost_utility_weight: 0.0,
            optimizer: AdamWConfig {
                learning_rate: 2.0e-3,
                weight_decay: 1.0e-5,
                grad_clip_norm: 5.0,
                ..AdamWConfig::default()
            },
        }
    }
}

const fn default_restriction_rank_boundary_width() -> f32 {
    0.125
}

const fn default_restriction_topk_temperature() -> f32 {
    0.25
}

pub(crate) fn validate_adaptive_restriction_training_memory_plan(
    model: &AdaptiveNpaModel,
    dataset: &AdaptiveRestrictionDatasetConfig,
    training: AdaptiveControllerTrainConfig,
) -> AutomataResult<()> {
    let groups_per_snapshot = model.config.bootstrap_fine_leaf_count() / 4;
    let snapshots = dataset
        .seeds
        .len()
        .checked_mul(dataset.cut_steps.len())
        .ok_or_else(|| {
            AutomataError::InvalidArgument(
                "adaptive restriction training snapshot count overflow".to_owned(),
            )
        })?;
    let rows = snapshots.checked_mul(groups_per_snapshot).ok_or_else(|| {
        AutomataError::InvalidArgument(
            "adaptive restriction training row count overflow".to_owned(),
        )
    })?;
    mlp::validate_mlp_buffer_plan(
        rows,
        mlp::MlpShape {
            input_dims: ADAPTIVE_CONTROLLER_INPUT_DIMS,
            hidden_dims: model.config.controller_hidden_dims,
            output_dims: ADAPTIVE_CONTROLLER_OUTPUT_DIMS,
        },
        training.optimizer_batch_rows,
        "adaptive restriction ranker",
    )?;
    Ok(())
}

const fn default_controller_training_enabled() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveControllerTrainingHistory {
    pub step: usize,
    pub loss: f32,
    pub gradient_norm: f32,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveControllerTrainingReport {
    pub backend: String,
    pub rows: usize,
    pub steps: usize,
    pub initial_loss: f32,
    pub final_loss: f32,
    pub best_loss: f32,
    #[serde(default)]
    pub best_step: usize,
    /// Checkpoint selected by heldout top-k cost regret. This is the final
    /// training step when no heldout selector batch is supplied.
    #[serde(default)]
    pub selected_step: usize,
    #[serde(default)]
    pub selected_heldout_normalized_cost_regret: Option<f32>,
    #[serde(default)]
    pub selected_heldout_worst_normalized_cost_regret: Option<f32>,
    pub elapsed_ms: f64,
    pub rows_per_second: f64,
    #[serde(default)]
    pub optimizer_batch_rows: usize,
    pub event_positive_weights: [f32; 2],
    #[serde(default)]
    pub restriction_rank_boundary_emphasis: f32,
    #[serde(default)]
    pub restriction_rank_boundary_width: f32,
    #[serde(default)]
    pub restriction_topk_loss_weight: f32,
    #[serde(default)]
    pub restriction_topk_temperature: f32,
    #[serde(default)]
    pub restriction_cost_utility_weight: f32,
    pub history: Vec<AdaptiveControllerTrainingHistory>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct AdaptiveRuleDistillationConfig {
    pub enabled: bool,
    pub particle_count: usize,
    pub rollout_steps: usize,
    pub rollouts: usize,
    pub temporal_samples: usize,
    pub rows_per_snapshot: usize,
    pub validation_rollouts: usize,
    pub dt: f32,
    pub update_prob: f32,
    pub seed: u64,
    pub seed_scale: f32,
    pub total_measure: f32,
    pub bandwidth: f32,
    pub steps: usize,
    pub report_interval: usize,
    #[serde(default)]
    pub on_policy_rounds: usize,
    #[serde(default = "default_on_policy_steps")]
    pub on_policy_steps: usize,
    pub optimizer: AdamWConfig,
}

impl Default for AdaptiveRuleDistillationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            particle_count: 1_024,
            rollout_steps: 64,
            rollouts: 8,
            temporal_samples: 9,
            rows_per_snapshot: 256,
            validation_rollouts: 2,
            dt: 1.0,
            update_prob: 0.5,
            seed: 42,
            seed_scale: 0.2,
            total_measure: std::f32::consts::PI * 0.2 * 0.2,
            bandwidth: 0.1,
            steps: 1_000,
            report_interval: 50,
            on_policy_rounds: 0,
            on_policy_steps: default_on_policy_steps(),
            optimizer: AdamWConfig {
                learning_rate: 1.0e-3,
                weight_decay: 1.0e-6,
                grad_clip_norm: 5.0,
                ..AdamWConfig::default()
            },
        }
    }
}

const fn default_on_policy_steps() -> usize {
    250
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdaptiveReplayBackend {
    #[default]
    CpuReference,
    WgpuResident,
}

/// Oracle used to label states visited by the recurrent adaptive policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdaptiveReplayTeacher {
    /// Evaluate the fixed-resolution rule directly on the coarse student state.
    /// This is inexpensive, but cannot supervise the coarse-graining defect.
    #[default]
    SameState,
    /// Prolong the current active leaves into stateless affine quadrature,
    /// evaluate the frozen fine rule, and restrict its action back to the
    /// active state. Unlike `coupled-fine`, this is a deterministic Markov
    /// target that contains no inaccessible persistent child residuals.
    MarkovQuadrature,
    /// Preserve a fine material realization behind every coarse leaf and
    /// restrict the fine teacher's action onto the recurrent student state.
    CoupledFine,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdaptiveDeploymentStrategy {
    #[default]
    Flat,
    FusedLocal,
}

/// Functional target used to train the single recurrent deployment rule.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdaptiveDeploymentTarget {
    /// Compress the current exact base/local/proxy policy into one MLP.
    #[default]
    Policy,
    /// Learn the measure-restricted fine teacher action directly. This is the
    /// recurrent coarse-graining path; it must use replay carrying a real fine
    /// teacher label rather than resident-policy self-distillation.
    RestrictedFineTeacher,
}

/// Parameterization optimized by the multiscale dynamics curriculum.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdaptiveMultiscaleRuleStrategy {
    /// Preserve a frozen fixed-resolution NPA and learn a gated correction.
    #[default]
    Residual,
    /// Keep the pretrained NPA-compatible rule exact at its native footprint
    /// and train a complete normalized-perception update for coarse leaves.
    /// This isolates coarse closure learning from the known-good fine-scale
    /// attractor.
    CoarseReplacement,
    /// Optimize the shared NPA rule itself on normalized unequal-measure
    /// perception and measure-restricted fine-teacher updates.
    FullNormalized,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AdaptiveMultiscaleTrainingConfig {
    #[serde(default)]
    pub enabled: bool,
    pub fine_particle_count: usize,
    pub cut_leaf_counts: Vec<usize>,
    pub rollout_steps: usize,
    pub rollouts: usize,
    pub temporal_samples: usize,
    pub rows_per_cut: usize,
    pub validation_rollouts: usize,
    pub dt: f32,
    pub update_prob: f32,
    pub seed: u64,
    pub seed_scale: f32,
    pub total_measure: f32,
    pub bandwidth: f32,
    pub steps: usize,
    pub report_interval: usize,
    pub controller_steps: usize,
    /// Select whether multiscale supervision trains a correction around a
    /// frozen regular rule or the complete normalized-adaptive NPA rule.
    #[serde(default)]
    pub rule_strategy: AdaptiveMultiscaleRuleStrategy,
    /// Preserve a warm checkpoint's exact local, proxy, and controller policy
    /// while fitting only its deployable recurrent surrogate.
    #[serde(default)]
    pub freeze_training_policy: bool,
    /// Preserve the exact local/proxy multiscale rule while allowing the
    /// topology controller and deployable surrogate to be retrained.
    #[serde(default)]
    pub freeze_multiscale_rule: bool,
    /// Preserve the learned local/proxy correction while fitting the
    /// recurrent closure rule. This isolates recurrent-state learning from a
    /// known-good static closure checkpoint.
    #[serde(default)]
    pub freeze_local_residual_rule: bool,
    /// Preserve the recurrent closure transition while fitting the local
    /// dynamics against a longer on-policy horizon.
    #[serde(default)]
    pub freeze_closure_rule: bool,
    /// Preserve the topology controller while allowing the exact multiscale
    /// rule and deployable surrogate to be retrained.
    #[serde(default)]
    pub freeze_controller: bool,
    /// Width of the exact leaf-local multiscale correction. Zero preserves the
    /// base NPA width. The packed WGPU path permits base + local + bias through
    /// 320 hidden units.
    #[serde(default)]
    pub local_residual_hidden_dims: usize,
    /// Hidden width of the single deployable NPA rule. The resident GPU path
    /// supports widths through 256; zero falls back to twice the base width,
    /// capped at 256.
    #[serde(default)]
    pub deployment_enabled: bool,
    #[serde(default)]
    pub deployment_hidden_dims: usize,
    #[serde(default)]
    pub deployment_strategy: AdaptiveDeploymentStrategy,
    #[serde(default)]
    pub deployment_target: AdaptiveDeploymentTarget,
    /// Functional-distillation steps for the deployable represented-measure
    /// rule. Zero reuses `steps`.
    #[serde(default)]
    pub deployment_steps: usize,
    /// DAgger rounds that visit states under the deployment rule and relabel
    /// them with the frozen exact policy.
    #[serde(default)]
    pub deployment_on_policy_rounds: usize,
    /// Discard offline/static rows when the first deployment DAgger batch is
    /// collected. This prevents recurrent teacher labels from being diluted
    /// by a much larger one-step material-cut dataset.
    #[serde(default)]
    pub deployment_on_policy_only_replay: bool,
    /// Optimizer steps per deployment DAgger round. Zero reuses
    /// `deployment_steps`.
    #[serde(default)]
    pub deployment_on_policy_steps: usize,
    /// Dynamics backend used to collect deployment DAgger states. This is
    /// independent from the Burn optimizer backend.
    #[serde(default)]
    pub deployment_replay_backend: AdaptiveReplayBackend,
    /// Rows per partial GPU weight-gradient reduction. The default exposes
    /// enough independent work for skinny adaptive MLPs; zero is the slower
    /// direct-reduction reference used by parity benchmarks.
    #[serde(default = "default_gradient_reduction_chunk_rows")]
    pub gradient_reduction_chunk_rows: usize,
    /// Optional high-confidence event labels. These are independent from the
    /// runtime hysteresis used after the controller predicts desired scale.
    #[serde(default)]
    pub controller_split_label_ratio: Option<f32>,
    #[serde(default)]
    pub controller_merge_label_ratio: Option<f32>,
    /// Coordinate-space regularization added to the exact gate-squared
    /// functional weighting. This constrains the residual near the native
    /// zero-gate scale and prevents coarse cuts from monopolizing the fit.
    #[serde(default = "default_residual_coordinate_weight")]
    pub residual_coordinate_weight: f32,
    /// Coordinate-space gain used while fitting the leaf-local residual. This
    /// is intentionally independent from the deployed gain so a smaller
    /// deployment gain cannot be cancelled by proportionally larger weights.
    #[serde(default = "default_residual_training_scale")]
    pub local_residual_training_scale: f32,
    /// Coordinate-space gain used while fitting the hub/proxy residual.
    #[serde(default = "default_residual_training_scale")]
    pub proxy_residual_training_scale: f32,
    #[serde(default)]
    pub on_policy_rounds: usize,
    /// After offline controller pretraining, discard static material-cut rows
    /// from the DAgger replay and fit only accumulated deployment-state rows.
    /// This prevents a much larger synthetic cut dataset from diluting exact
    /// counterfactual labels at quality-scale leaf counts.
    #[serde(default)]
    pub controller_on_policy_only_replay: bool,
    /// After static-cut initialization, fit the multiscale dynamics rule only
    /// on accumulated recurrent coupled-teacher rows. This is the recurrent
    /// analogue of `controller_on_policy_only_replay` and prevents a much
    /// larger one-step dataset from diluting long-horizon correction labels.
    #[serde(default)]
    pub multiscale_on_policy_only_replay: bool,
    /// Topology policy used to generate exact on-policy curriculum states.
    /// Pure oracle modes are diagnostic controls. Deployable inference may be
    /// fully learned or use learned event gates with deterministic
    /// refinement-defect allocation.
    #[serde(default)]
    pub on_policy_topology_control: AdaptiveTopologyControl,
    /// Dynamics backend used to collect exact-policy curriculum states.
    #[serde(default)]
    pub on_policy_replay_backend: AdaptiveReplayBackend,
    /// Label source for recurrent multiscale replay. `markov-quadrature` is the
    /// active-leaf coarse-graining path, `coupled-fine` is a persistent-detail
    /// teacher ceiling, and `same-state` is the inexpensive ablation.
    #[serde(default)]
    pub on_policy_teacher: AdaptiveReplayTeacher,
    /// Absolute fine-teacher rollout steps at which material cuts are formed.
    ///
    /// A zero cut trains seed-state closure. Positive cuts train the same
    /// temporal restriction boundary exercised by deployment. The coupled
    /// fine teacher and active student continue from the same stochastic step
    /// index after every cut.
    #[serde(default = "default_multiscale_on_policy_cut_steps")]
    pub on_policy_cut_steps: Vec<usize>,
    #[serde(default = "default_multiscale_on_policy_rollout_steps")]
    pub on_policy_rollout_steps: usize,
    #[serde(default = "default_multiscale_on_policy_rollouts")]
    pub on_policy_rollouts: usize,
    #[serde(default = "default_multiscale_on_policy_snapshot_interval")]
    pub on_policy_snapshot_interval: usize,
    #[serde(default = "default_multiscale_on_policy_rows")]
    pub on_policy_rows_per_snapshot: usize,
    #[serde(default = "default_on_policy_steps")]
    pub on_policy_steps: usize,
    pub optimizer: AdamWConfig,
    /// Optional optimizer override for the recurrent closure rule. When
    /// omitted, `optimizer` remains the shared default.
    #[serde(default)]
    pub closure_optimizer: Option<AdamWConfig>,
    pub controller_optimizer: AdamWConfig,
}

impl Default for AdaptiveMultiscaleTrainingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            fine_particle_count: 1_024,
            cut_leaf_counts: vec![256, 512, 1_024],
            rollout_steps: 64,
            rollouts: 4,
            temporal_samples: 9,
            rows_per_cut: 256,
            validation_rollouts: 2,
            dt: 1.0,
            update_prob: 0.5,
            seed: 42,
            seed_scale: 0.2,
            total_measure: std::f32::consts::PI * 0.2 * 0.2,
            bandwidth: 0.1,
            steps: 1_000,
            report_interval: 50,
            controller_steps: 1_000,
            rule_strategy: AdaptiveMultiscaleRuleStrategy::default(),
            freeze_training_policy: false,
            freeze_multiscale_rule: false,
            freeze_local_residual_rule: false,
            freeze_closure_rule: false,
            freeze_controller: false,
            local_residual_hidden_dims: 0,
            deployment_hidden_dims: 0,
            deployment_strategy: AdaptiveDeploymentStrategy::default(),
            deployment_target: AdaptiveDeploymentTarget::default(),
            deployment_enabled: false,
            deployment_steps: 0,
            deployment_on_policy_rounds: 0,
            deployment_on_policy_only_replay: false,
            deployment_on_policy_steps: 0,
            deployment_replay_backend: AdaptiveReplayBackend::default(),
            gradient_reduction_chunk_rows: default_gradient_reduction_chunk_rows(),
            controller_split_label_ratio: None,
            controller_merge_label_ratio: None,
            residual_coordinate_weight: default_residual_coordinate_weight(),
            local_residual_training_scale: default_residual_training_scale(),
            proxy_residual_training_scale: default_residual_training_scale(),
            on_policy_rounds: 0,
            controller_on_policy_only_replay: false,
            multiscale_on_policy_only_replay: false,
            on_policy_topology_control: AdaptiveTopologyControl::Learned,
            on_policy_replay_backend: AdaptiveReplayBackend::default(),
            on_policy_teacher: AdaptiveReplayTeacher::default(),
            on_policy_cut_steps: default_multiscale_on_policy_cut_steps(),
            on_policy_rollout_steps: default_multiscale_on_policy_rollout_steps(),
            on_policy_rollouts: default_multiscale_on_policy_rollouts(),
            on_policy_snapshot_interval: default_multiscale_on_policy_snapshot_interval(),
            on_policy_rows_per_snapshot: default_multiscale_on_policy_rows(),
            on_policy_steps: default_on_policy_steps(),
            optimizer: AdamWConfig {
                learning_rate: 5.0e-4,
                weight_decay: 1.0e-6,
                grad_clip_norm: 5.0,
                ..AdamWConfig::default()
            },
            closure_optimizer: None,
            controller_optimizer: AdamWConfig {
                learning_rate: 1.0e-3,
                weight_decay: 1.0e-5,
                grad_clip_norm: 5.0,
                ..AdamWConfig::default()
            },
        }
    }
}

impl AdaptiveMultiscaleTrainingConfig {
    pub fn resolved_local_residual_hidden_dims(&self, base_hidden_dims: usize) -> usize {
        if self.local_residual_hidden_dims == 0 {
            base_hidden_dims
        } else {
            self.local_residual_hidden_dims
        }
    }

    pub fn resolved_deployment_hidden_dims(&self, base_hidden_dims: usize) -> usize {
        if self.deployment_hidden_dims == 0 {
            base_hidden_dims.saturating_mul(2).min(256)
        } else {
            self.deployment_hidden_dims
        }
    }

    pub fn resolved_deployment_steps(&self) -> usize {
        if self.deployment_steps == 0 {
            self.steps
        } else {
            self.deployment_steps
        }
    }

    pub fn resolved_deployment_on_policy_steps(&self) -> usize {
        if self.deployment_on_policy_steps == 0 {
            self.resolved_deployment_steps()
        } else {
            self.deployment_on_policy_steps
        }
    }

    pub(super) fn controller_label_ratios(
        &self,
        adaptive: &super::AdaptiveNpaConfig,
    ) -> AutomataResult<(f32, f32)> {
        let split = self
            .controller_split_label_ratio
            .unwrap_or(adaptive.split_ratio);
        let merge = self
            .controller_merge_label_ratio
            .unwrap_or(adaptive.merge_ratio);
        if !split.is_finite() || !merge.is_finite() || split <= 0.0 || split >= 1.0 || merge <= 1.0
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive controller split labels require a ratio in (0,1) and merge labels require a ratio > 1"
                    .to_string(),
            ));
        }
        Ok((split, merge))
    }
}

const fn default_residual_coordinate_weight() -> f32 {
    0.0
}

const fn default_gradient_reduction_chunk_rows() -> usize {
    mlp::DEFAULT_GRADIENT_REDUCTION_CHUNK_ROWS
}

const fn default_residual_training_scale() -> f32 {
    1.0
}

const fn default_multiscale_on_policy_rollout_steps() -> usize {
    64
}

fn default_multiscale_on_policy_cut_steps() -> Vec<usize> {
    vec![0]
}

const fn default_multiscale_on_policy_rollouts() -> usize {
    2
}

const fn default_multiscale_on_policy_snapshot_interval() -> usize {
    4
}

const fn default_multiscale_on_policy_rows() -> usize {
    256
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveMultiscaleDatasetReport {
    pub rollouts: usize,
    pub snapshots: usize,
    pub cuts: usize,
    pub rows: usize,
    pub minimum_material_leaves: usize,
    pub maximum_material_leaves: usize,
    pub minimum_footprint: f32,
    pub maximum_footprint: f32,
    pub footprint_coefficient_of_variation: f32,
    pub mean_proxy_nodes: f32,
    pub mean_counterfactual_error: f32,
    pub mean_teacher_update_error: f32,
    /// Largest absolute material-state value observed while collecting replay.
    #[serde(default)]
    pub maximum_particle_state_absolute: f32,
    /// Largest absolute recurrent closure coefficient observed in replay.
    #[serde(default)]
    pub maximum_closure_mode_absolute: f32,
    /// Tail diagnostics for the physical teacher update selected into the batch.
    #[serde(default)]
    pub teacher_update_p99_absolute: f32,
    #[serde(default)]
    pub maximum_teacher_update_absolute: f32,
    /// Tail diagnostics for the recurrent closure target selected into the batch.
    #[serde(default)]
    pub closure_target_p99_absolute: f32,
    #[serde(default)]
    pub maximum_closure_target_absolute: f32,
    #[serde(default)]
    pub closure_basis_target_p99_absolute: f32,
    #[serde(default)]
    pub maximum_closure_basis_target_absolute: f32,
    /// Per-rollout maxima preserve catastrophic tails hidden by row averages.
    #[serde(default)]
    pub rollout_maximum_particle_state_absolute: Vec<f32>,
    #[serde(default)]
    pub rollout_peak_particle_state_step: Vec<usize>,
    #[serde(default)]
    pub rollout_maximum_teacher_update_absolute: Vec<f32>,
    #[serde(default)]
    pub rollout_peak_teacher_update_step: Vec<usize>,
    #[serde(default)]
    pub rollout_maximum_closure_target_absolute: Vec<f32>,
    #[serde(default)]
    pub rollout_peak_closure_target_step: Vec<usize>,
    pub generation_elapsed_ms: f64,
}

pub(super) fn absolute_percentile(values: &[f32], percentile: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut absolute = values.iter().map(|value| value.abs()).collect::<Vec<_>>();
    absolute.sort_unstable_by(f32::total_cmp);
    let rank = ((absolute.len() - 1) as f32 * percentile.clamp(0.0, 1.0)).round() as usize;
    absolute[rank]
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveMultiscaleTrainingBatch {
    pub local_features: Vec<f32>,
    /// Closure-specific perception over neighboring phase/null-mode state.
    /// Empty when recurrent closure is disabled.
    pub closure_features: Vec<f32>,
    pub proxy_features: Vec<f32>,
    pub target_update: Vec<f32>,
    /// Teacher derivative for the compact affine-null closure state. This is
    /// laid out like an NPA update: spatial channels are zero and state
    /// channels contain `d(closure_mode) / dt`. It is empty when recurrent
    /// closure is disabled.
    pub closure_mode_target_update: Vec<f32>,
    /// Teacher derivative for the compact four-child affine-null basis. The
    /// first four coordinates of each NPA-shaped row contain `d(basis) / dt`;
    /// remaining coordinates are zero.
    pub closure_basis_target_update: Vec<f32>,
    /// Per-row closure loss weight. Native leaves have zero weight because
    /// they do not discard an affine-null state mode.
    pub closure_mode_row_weights: Vec<f32>,
    pub deployment_features: Vec<f32>,
    pub deployment_target_update: Vec<f32>,
    pub deployment_row_weights: Vec<f32>,
    pub deployment_residual_gate: Vec<f32>,
    pub controller_features: Vec<f32>,
    pub controller_targets: Vec<f32>,
    pub row_weights: Vec<f32>,
    pub rows: usize,
    pub report: AdaptiveMultiscaleDatasetReport,
}

impl AdaptiveMultiscaleTrainingBatch {
    pub fn validate(&self, input_dims: usize, output_dims: usize) -> AutomataResult<()> {
        if self.rows == 0
            || !self.local_features.len().is_multiple_of(self.rows)
            || self.local_features.len() / self.rows < input_dims
            || !(self.closure_features.is_empty()
                || self.closure_features.len().is_multiple_of(self.rows))
            || !(self.proxy_features.is_empty()
                || self.proxy_features.len() == self.rows * input_dims)
            || self.target_update.len() != self.rows * output_dims
            || !((self.closure_mode_target_update.is_empty()
                && self.closure_basis_target_update.is_empty()
                && self.closure_mode_row_weights.is_empty())
                || (self.closure_mode_target_update.len() == self.rows * output_dims
                    && self.closure_basis_target_update.len() == self.rows * output_dims
                    && self.closure_mode_row_weights.len() == self.rows))
            || self.deployment_features.len() != self.rows * input_dims
            || self.deployment_target_update.len() != self.rows * output_dims
            || self.deployment_row_weights.len() != self.rows
            || self.deployment_residual_gate.len() != self.rows
            || self.controller_features.len() != self.rows * ADAPTIVE_CONTROLLER_INPUT_DIMS
            || self.controller_targets.len() != self.rows * ADAPTIVE_CONTROLLER_OUTPUT_DIMS
            || self.row_weights.len() != self.rows
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive multiscale training batch shape mismatch".to_string(),
            ));
        }
        if self
            .local_features
            .iter()
            .chain(&self.closure_features)
            .chain(&self.proxy_features)
            .chain(&self.target_update)
            .chain(&self.closure_mode_target_update)
            .chain(&self.closure_basis_target_update)
            .chain(&self.closure_mode_row_weights)
            .chain(&self.deployment_features)
            .chain(&self.deployment_target_update)
            .chain(&self.deployment_row_weights)
            .chain(&self.deployment_residual_gate)
            .chain(&self.controller_features)
            .chain(&self.controller_targets)
            .chain(&self.row_weights)
            .any(|value| !value.is_finite())
            || self.row_weights.iter().any(|value| *value < 0.0)
            || self
                .closure_mode_row_weights
                .iter()
                .any(|value| *value < 0.0)
            || self.deployment_row_weights.iter().any(|value| *value < 0.0)
            || self.row_weights.iter().sum::<f32>() <= f32::MIN_POSITIVE
            || self.deployment_row_weights.iter().sum::<f32>() <= f32::MIN_POSITIVE
            || (!self.closure_mode_row_weights.is_empty()
                && self.closure_mode_row_weights.iter().sum::<f32>() <= f32::MIN_POSITIVE)
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive multiscale training batch contains invalid values".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod weight_tests {
    use super::*;

    #[test]
    fn positive_weight_normalization_is_checked_and_unit_mean() {
        let mut weights = vec![1.0, 3.0];
        normalize_positive_weights(&mut weights, "test").unwrap();
        assert_eq!(weights, vec![0.5, 1.5]);

        let error = normalize_positive_weights(&mut [0.0, 0.0], "zero").unwrap_err();
        assert!(error.to_string().contains("positive finite sum"));
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveMultiscaleRuleValidationReport {
    pub rows: usize,
    pub local_only_mean_squared_error: f32,
    pub combined_mean_squared_error: f32,
    pub normalized_mean_squared_error: f32,
    pub update_correlation: f32,
    pub proxy_update_root_mean_square: f32,
    pub proxy_relative_mse_gain: f32,
    /// Fraction of the zero-residual functional MSE removed by both branches.
    #[serde(default)]
    pub functional_relative_mse_gain: f32,
    /// Per-update-channel normalized RMSE under the runtime gate weighting.
    #[serde(default)]
    pub channel_normalized_root_mean_squared_error: Vec<f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveMultiscaleRuleTrainingReport {
    #[serde(default)]
    pub strategy: AdaptiveMultiscaleRuleStrategy,
    pub backend: String,
    pub rows: usize,
    pub steps: usize,
    pub initial_validation: AdaptiveMultiscaleRuleValidationReport,
    pub trained_validation: AdaptiveMultiscaleRuleValidationReport,
    pub initial_mean_squared_error: f32,
    pub final_mean_squared_error: f32,
    pub best_mean_squared_error: f32,
    pub training_elapsed_ms: f64,
    pub rows_per_second: f64,
    pub history: Vec<AdaptiveRuleTrainingHistory>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveClosureModeValidationReport {
    pub active_rows: usize,
    pub weighted_mean_squared_error: f32,
    pub target_root_mean_square: f32,
    pub normalized_root_mean_squared_error: f32,
    pub update_correlation: f32,
    pub maximum_absolute_error: f32,
    #[serde(default)]
    pub phase_normalized_root_mean_squared_error: f32,
    #[serde(default)]
    pub phase_update_correlation: f32,
    #[serde(default)]
    pub state_normalized_root_mean_squared_error: f32,
    #[serde(default)]
    pub state_update_correlation: f32,
    #[serde(default)]
    pub basis_normalized_root_mean_squared_error: f32,
    #[serde(default)]
    pub basis_update_correlation: f32,
    #[serde(default)]
    pub basis_maximum_absolute_error: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveClosureModeTrainingReport {
    pub backend: String,
    pub rows: usize,
    pub active_rows: usize,
    pub steps: usize,
    pub initial_validation: AdaptiveClosureModeValidationReport,
    pub trained_validation: AdaptiveClosureModeValidationReport,
    pub initial_mean_squared_error: f32,
    pub final_mean_squared_error: f32,
    pub best_mean_squared_error: f32,
    pub training_elapsed_ms: f64,
    pub rows_per_second: f64,
    #[serde(default)]
    pub target_channel_root_mean_square: Vec<f32>,
    #[serde(default)]
    pub target_channel_standardization: Vec<f32>,
    pub history: Vec<AdaptiveRuleTrainingHistory>,
    #[serde(default)]
    pub basis_initial_mean_squared_error: f32,
    #[serde(default)]
    pub basis_final_mean_squared_error: f32,
    #[serde(default)]
    pub basis_best_mean_squared_error: f32,
    #[serde(default)]
    pub basis_training_elapsed_ms: f64,
    #[serde(default)]
    pub basis_rows_per_second: f64,
    #[serde(default)]
    pub basis_target_channel_root_mean_square: Vec<f32>,
    #[serde(default)]
    pub basis_target_channel_standardization: Vec<f32>,
    #[serde(default)]
    pub basis_history: Vec<AdaptiveRuleTrainingHistory>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveDeploymentRuleValidationReport {
    pub rows: usize,
    pub mean_squared_error: f32,
    pub normalized_mean_squared_error: f32,
    pub update_correlation: f32,
    pub target_root_mean_square: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveDeploymentRuleTrainingReport {
    pub backend: String,
    pub rows: usize,
    pub steps: usize,
    pub hidden_dims: usize,
    #[serde(default)]
    pub target: AdaptiveDeploymentTarget,
    pub initial_validation: AdaptiveDeploymentRuleValidationReport,
    pub trained_validation: AdaptiveDeploymentRuleValidationReport,
    pub initial_mean_squared_error: f32,
    pub final_mean_squared_error: f32,
    pub best_mean_squared_error: f32,
    pub training_elapsed_ms: f64,
    pub rows_per_second: f64,
    pub history: Vec<AdaptiveRuleTrainingHistory>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveRuleTrainingBatch {
    pub features: Vec<f32>,
    pub target_update: Vec<f32>,
    pub rows: usize,
    pub generation_elapsed_ms: f64,
}

impl AdaptiveRuleTrainingBatch {
    pub fn validate(&self, input_dims: usize, output_dims: usize) -> AutomataResult<()> {
        if self.rows == 0
            || self.features.len() != self.rows * input_dims
            || self.target_update.len() != self.rows * output_dims
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive rule training batch shape mismatch".to_string(),
            ));
        }
        if self
            .features
            .iter()
            .chain(&self.target_update)
            .any(|value| !value.is_finite())
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive rule training batch contains non-finite values".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveRuleTrainingHistory {
    pub step: usize,
    pub mean_squared_error: f32,
    pub gradient_norm: f32,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveRuleTrainingReport {
    pub backend: String,
    pub rows: usize,
    pub steps: usize,
    pub initial_mean_squared_error: f32,
    pub final_mean_squared_error: f32,
    pub best_mean_squared_error: f32,
    pub dataset_generation_ms: f64,
    pub training_elapsed_ms: f64,
    pub rows_per_second: f64,
    pub history: Vec<AdaptiveRuleTrainingHistory>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveRuleValidationReport {
    pub rows: usize,
    pub mean_squared_error: f32,
    pub normalized_mean_squared_error: f32,
    pub update_correlation: f32,
    pub target_root_mean_square: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveRuleDistillationReport {
    pub source_validation: AdaptiveRuleValidationReport,
    pub offline_validation: AdaptiveRuleValidationReport,
    pub trained_validation: AdaptiveRuleValidationReport,
    pub training: AdaptiveRuleTrainingReport,
    pub on_policy_training: Vec<AdaptiveRuleTrainingReport>,
}
