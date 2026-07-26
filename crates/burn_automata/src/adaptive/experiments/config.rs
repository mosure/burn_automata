use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::adaptive::{
    AdaptiveClosureAuditBackend, AdaptiveClosureIdentifiabilityConfig,
    AdaptiveClosureIdentifiabilityReport, AdaptiveControllerTrainConfig,
    AdaptiveMultiscaleTrainingConfig, AdaptiveNpaConfig, AdaptiveOracleDatasetConfig,
    AdaptiveRenderDecoder, AdaptiveRestrictionDatasetConfig, AdaptiveRestrictionDatasetReport,
    AdaptiveRestrictionSelectionReport, AdaptiveRolloutConfig, AdaptiveRuleDistillationConfig,
    AdaptiveTopologyControl,
};
use crate::{Target2dGpuTrainingReport, Target2dLossConfig, Target2dTrainingConfig};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdaptiveTrainingBackend {
    NdArray,
    #[default]
    Wgpu,
    Cuda,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveExperimentConfig {
    pub report_output: PathBuf,
    pub model_output: PathBuf,
    pub base_model: Option<PathBuf>,
    /// Optional task-trained adaptive checkpoint used to resume multiscale
    /// training. Its frozen base rule must exactly match `base_model`.
    #[serde(default)]
    pub adaptive_checkpoint: Option<PathBuf>,
    #[serde(default)]
    pub base_training: AdaptiveBaseTrainingConfig,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default)]
    pub backend: AdaptiveTrainingBackend,
    #[serde(default = "AdaptiveNpaConfig::growing_2d")]
    pub adaptive: AdaptiveNpaConfig,
    #[serde(default)]
    pub operator: AdaptiveOperatorExperimentConfig,
    #[serde(default)]
    pub topology: AdaptiveTopologyExperimentConfig,
    #[serde(default)]
    pub graph: AdaptiveGraphExperimentConfig,
    #[serde(default)]
    pub scaling: AdaptiveScalingExperimentConfig,
    #[serde(default)]
    pub training_data: AdaptiveOracleDatasetConfig,
    #[serde(default)]
    pub training: AdaptiveControllerTrainConfig,
    #[serde(default)]
    pub rule_distillation: AdaptiveRuleDistillationConfig,
    #[serde(default)]
    pub multiscale_training: AdaptiveMultiscaleTrainingConfig,
    #[serde(default)]
    pub closure_identifiability: AdaptiveClosureIdentifiabilityConfig,
    #[serde(default)]
    pub restriction_training: AdaptiveRestrictionExperimentConfig,
    #[serde(default)]
    pub task_quality: AdaptiveTaskQualityConfig,
    #[serde(default)]
    pub rollout: AdaptiveRolloutExperimentConfig,
    #[serde(default)]
    pub gates: AdaptiveExperimentGates,
}

/// Standalone, bounded audit bundle for testing whether a deterministic
/// memoryless coarse closure can represent the restricted fine dynamics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveClosureAuditConfig {
    pub report_output: PathBuf,
    pub base_model: PathBuf,
    #[serde(default)]
    pub backend: AdaptiveClosureAuditBackend,
    #[serde(default = "AdaptiveNpaConfig::growing_2d")]
    pub adaptive: AdaptiveNpaConfig,
    #[serde(default)]
    pub audit: AdaptiveClosureIdentifiabilityConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveBaseTrainingConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub target_image: PathBuf,
    #[serde(default)]
    pub reference_model: Option<PathBuf>,
    #[serde(default)]
    pub model_output: Option<PathBuf>,
    #[serde(default)]
    pub checkpoint_root: Option<PathBuf>,
    #[serde(default)]
    pub checkpoint_interval_steps: usize,
    #[serde(default)]
    pub checkpoint_interval_seconds: u64,
    #[serde(default = "default_target_points")]
    pub target_points: usize,
    #[serde(default)]
    pub target_image_size: Option<usize>,
    #[serde(default = "default_target_threshold")]
    pub target_threshold: f32,
    #[serde(default)]
    pub loss: Target2dLossConfig,
    #[serde(default)]
    pub phases: Vec<AdaptiveBaseTrainingPhase>,
}

impl Default for AdaptiveBaseTrainingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            target_image: PathBuf::new(),
            reference_model: None,
            model_output: None,
            checkpoint_root: None,
            checkpoint_interval_steps: 0,
            checkpoint_interval_seconds: 0,
            target_points: default_target_points(),
            target_image_size: None,
            target_threshold: default_target_threshold(),
            loss: Target2dLossConfig::default(),
            phases: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveBaseTrainingPhase {
    pub name: String,
    #[serde(default)]
    pub loss: Option<Target2dLossConfig>,
    #[serde(flatten)]
    pub training: Target2dTrainingConfig,
}

const fn default_target_points() -> usize {
    4096
}

const fn default_target_threshold() -> f32 {
    0.05
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AdaptiveExperimentGates {
    #[serde(default)]
    pub require_fresh_base_training: bool,
    pub max_operator_constant_error: f32,
    pub max_operator_affine_error: f32,
    pub max_operator_fallback_fraction: f32,
    pub max_topology_relative_error: f64,
    pub max_topology_spd_failures: usize,
    pub min_topology_events_per_second: f64,
    pub max_validation_mse: f32,
    pub min_desired_scale_correlation: f32,
    pub min_controller_rows_per_second: f64,
    #[serde(default = "infinite_f32")]
    pub max_rule_normalized_mean_squared_error: f32,
    #[serde(default = "negative_one_f32")]
    pub min_rule_update_correlation: f32,
    #[serde(default = "infinite_f32")]
    pub max_multiscale_normalized_mean_squared_error: f32,
    #[serde(default = "negative_one_f32")]
    pub min_multiscale_update_correlation: f32,
    #[serde(default = "infinite_f32")]
    pub max_recurrent_closure_normalized_root_mean_squared_error: f32,
    #[serde(default = "negative_one_f32")]
    pub min_recurrent_closure_update_correlation: f32,
    #[serde(default = "negative_one_f32")]
    pub min_proxy_relative_mse_gain: f32,
    #[serde(default)]
    pub min_multiscale_dataset_footprint_coefficient_of_variation: f32,
    #[serde(default)]
    pub min_restriction_heldout_accuracy: f32,
    #[serde(default)]
    pub min_restriction_heldout_intersection_over_union: f32,
    #[serde(default = "negative_infinity_f32")]
    pub min_adaptive_target_psnr_db: f32,
    #[serde(default = "infinite_f32")]
    pub max_adaptive_teacher_psnr_gap_db: f32,
    #[serde(default = "infinite_f32")]
    pub max_fine_fixed_teacher_psnr_gap_db: f32,
    #[serde(default = "negative_infinity_f32")]
    pub min_adaptive_over_budget_fixed_psnr_gain_db: f32,
    #[serde(default = "negative_infinity_f32")]
    pub min_bandwidth_adaptation_psnr_gain_db: f32,
    #[serde(default = "negative_infinity_f32")]
    pub min_adaptive_over_regular_base_psnr_gain_db: f32,
    #[serde(default = "negative_infinity_f32")]
    pub min_adaptive_over_regular_matched_budget_psnr_gain_db: f32,
    #[serde(default = "negative_infinity_f32")]
    pub min_adaptive_over_regular_material_matched_budget_psnr_gain_db: f32,
    #[serde(default = "negative_infinity_f32")]
    pub min_deployment_over_training_policy_psnr_gain_db: f32,
    #[serde(default = "one_f64")]
    pub max_task_quality_leaf_relative_error: f64,
    #[serde(default)]
    pub min_task_quality_footprint_coefficient_of_variation: f32,
    #[serde(default = "one_f32")]
    pub min_task_quality_footprint_ratio: f32,
    #[serde(default)]
    pub min_task_quality_occupied_material_scale_bins: usize,
    #[serde(default)]
    pub min_task_quality_fractional_material_scale_fraction: f32,
    #[serde(default)]
    pub min_task_quality_topology_events: usize,
    #[serde(default)]
    pub require_task_quality_bandwidth_adaptation_active: bool,
    #[serde(default = "negative_one_f32")]
    pub min_task_quality_detail_density_correlation: f32,
    #[serde(default = "infinite_f32")]
    pub max_task_quality_high_to_low_detail_footprint_ratio: f32,
    #[serde(default = "negative_one_f32")]
    pub min_task_quality_refinement_defect_density_correlation: f32,
    #[serde(default)]
    pub min_task_quality_low_to_high_refinement_defect_footprint_ratio: f32,
    #[serde(default = "negative_infinity_f32")]
    pub min_task_quality_refinement_defect_relative_gain: f32,
    #[serde(default = "negative_one_f32")]
    pub min_task_quality_controller_oracle_refinement_scale_correlation: f32,
    #[serde(default)]
    pub min_task_quality_validation_seeds: usize,
    /// Minimum mean PSNR delta between the adaptive rollout and the explicit
    /// task-quality reference model across validation seeds.
    #[serde(default = "negative_infinity_f32")]
    pub min_validation_mean_adaptive_over_teacher_psnr_gain_db: f32,
    /// Minimum per-seed PSNR delta between the adaptive rollout and the
    /// explicit task-quality reference model.
    #[serde(default = "negative_infinity_f32")]
    pub min_validation_worst_adaptive_over_teacher_psnr_gain_db: f32,
    #[serde(default = "negative_infinity_f32")]
    pub min_validation_mean_adaptive_over_regular_base_psnr_gain_db: f32,
    #[serde(default = "negative_infinity_f32")]
    pub min_validation_worst_adaptive_over_regular_base_psnr_gain_db: f32,
    #[serde(default = "negative_infinity_f32")]
    pub min_validation_mean_adaptive_over_regular_matched_budget_psnr_gain_db: f32,
    #[serde(default = "negative_infinity_f32")]
    pub min_validation_worst_adaptive_over_regular_matched_budget_psnr_gain_db: f32,
    #[serde(default = "negative_infinity_f32")]
    pub min_validation_mean_adaptive_over_regular_material_matched_budget_psnr_gain_db: f32,
    #[serde(default = "negative_infinity_f32")]
    pub min_validation_worst_adaptive_over_regular_material_matched_budget_psnr_gain_db: f32,
    #[serde(default = "negative_infinity_f32")]
    pub min_validation_mean_adaptive_over_budget_fixed_psnr_gain_db: f32,
    #[serde(default = "negative_infinity_f32")]
    pub min_validation_worst_adaptive_over_budget_fixed_psnr_gain_db: f32,
    #[serde(default = "negative_one_f32")]
    pub min_validation_controller_oracle_refinement_scale_correlation: f32,
    #[serde(default = "infinite_f64")]
    pub max_validation_measure_relative_drift: f64,
    #[serde(default = "one_f64")]
    pub max_validation_leaf_relative_error: f64,
    /// Reject recurrent implementations that retain hidden fine state behind
    /// the visible material leaves. Temporary stateless quadrature remains an
    /// active-leaf method and is accounted separately by interaction rows.
    #[serde(default)]
    pub require_active_leaf_dynamics: bool,
    /// Maximum rule-evaluation rows divided by visible material leaves for
    /// every validation trajectory.
    #[serde(default = "infinite_f64")]
    pub max_validation_interaction_particle_ratio: f64,
    #[serde(default = "infinite_f64")]
    pub max_validation_mean_adaptive_rollout_elapsed_ms: f64,
    #[serde(default = "infinite_f64")]
    pub max_validation_mean_adaptive_topology_elapsed_ms: f64,
    #[serde(default = "infinite_f64")]
    pub max_validation_topology_update_elapsed_ms: f64,
    /// Minimum full-rollout selected-mode quality relative to the matched
    /// uncut regular NPA at the final gap-decomposition horizon.
    #[serde(default = "negative_infinity_f32")]
    pub min_gap_final_selected_mode_vs_fine_control_db: f32,
    /// Minimum change from the immediate post-cut gap to the final recurrent
    /// gap. This rejects partitions that look acceptable only at cut time.
    #[serde(default = "negative_infinity_f32")]
    pub min_gap_post_cut_recurrent_change_vs_fine_control_db: f32,
    /// Maximum final-horizon quality left between the deployed learned cut and
    /// the target-render-ranked oracle over the same action family.
    #[serde(default = "infinite_f32")]
    pub max_gap_controller_target_render_regret_db: f32,
    pub max_rollout_target_relative_error: f64,
    pub max_rollout_measure_relative_drift: f64,
    pub min_rollout_footprint_coefficient_of_variation: f32,
    #[serde(default)]
    pub min_rollout_occupied_material_scale_bins: usize,
    #[serde(default)]
    pub min_rollout_fractional_material_scale_fraction: f32,
    pub min_rollout_particle_steps_per_second: f64,
    pub require_graph_cap_and_search_parity: bool,
    #[serde(default = "default_sparse_conservation_error")]
    pub max_sparse_conservation_error: f64,
    #[serde(default = "default_sparse_protected_band_nrmse")]
    pub max_sparse_protected_band_nrmse: f64,
    #[serde(default)]
    pub min_sparse_largest_cap_count_reduction: f64,
    #[serde(default = "infinite_f64")]
    pub max_sparse_boundary_hd95_voxels: f64,
}

impl Default for AdaptiveExperimentGates {
    fn default() -> Self {
        Self {
            require_fresh_base_training: false,
            max_operator_constant_error: 1.0e-3,
            max_operator_affine_error: 1.0e-3,
            max_operator_fallback_fraction: 1.0e-3,
            max_topology_relative_error: 1.0e-12,
            max_topology_spd_failures: 0,
            min_topology_events_per_second: 0.0,
            max_validation_mse: f32::INFINITY,
            min_desired_scale_correlation: -1.0,
            min_controller_rows_per_second: 0.0,
            max_rule_normalized_mean_squared_error: f32::INFINITY,
            min_rule_update_correlation: -1.0,
            max_multiscale_normalized_mean_squared_error: f32::INFINITY,
            min_multiscale_update_correlation: -1.0,
            max_recurrent_closure_normalized_root_mean_squared_error: f32::INFINITY,
            min_recurrent_closure_update_correlation: -1.0,
            min_proxy_relative_mse_gain: -1.0,
            min_multiscale_dataset_footprint_coefficient_of_variation: 0.0,
            min_restriction_heldout_accuracy: 0.0,
            min_restriction_heldout_intersection_over_union: 0.0,
            min_adaptive_target_psnr_db: f32::NEG_INFINITY,
            max_adaptive_teacher_psnr_gap_db: f32::INFINITY,
            max_fine_fixed_teacher_psnr_gap_db: f32::INFINITY,
            min_adaptive_over_budget_fixed_psnr_gain_db: f32::NEG_INFINITY,
            min_bandwidth_adaptation_psnr_gain_db: f32::NEG_INFINITY,
            min_adaptive_over_regular_base_psnr_gain_db: f32::NEG_INFINITY,
            min_adaptive_over_regular_matched_budget_psnr_gain_db: f32::NEG_INFINITY,
            min_adaptive_over_regular_material_matched_budget_psnr_gain_db: f32::NEG_INFINITY,
            min_deployment_over_training_policy_psnr_gain_db: f32::NEG_INFINITY,
            max_task_quality_leaf_relative_error: 1.0,
            min_task_quality_footprint_coefficient_of_variation: 0.0,
            min_task_quality_footprint_ratio: 1.0,
            min_task_quality_occupied_material_scale_bins: 0,
            min_task_quality_fractional_material_scale_fraction: 0.0,
            min_task_quality_topology_events: 0,
            require_task_quality_bandwidth_adaptation_active: false,
            min_task_quality_detail_density_correlation: -1.0,
            max_task_quality_high_to_low_detail_footprint_ratio: f32::INFINITY,
            min_task_quality_refinement_defect_density_correlation: -1.0,
            min_task_quality_low_to_high_refinement_defect_footprint_ratio: 0.0,
            min_task_quality_refinement_defect_relative_gain: f32::NEG_INFINITY,
            min_task_quality_controller_oracle_refinement_scale_correlation: -1.0,
            min_task_quality_validation_seeds: 0,
            min_validation_mean_adaptive_over_teacher_psnr_gain_db: f32::NEG_INFINITY,
            min_validation_worst_adaptive_over_teacher_psnr_gain_db: f32::NEG_INFINITY,
            min_validation_mean_adaptive_over_regular_base_psnr_gain_db: f32::NEG_INFINITY,
            min_validation_worst_adaptive_over_regular_base_psnr_gain_db: f32::NEG_INFINITY,
            min_validation_mean_adaptive_over_regular_matched_budget_psnr_gain_db:
                f32::NEG_INFINITY,
            min_validation_worst_adaptive_over_regular_matched_budget_psnr_gain_db:
                f32::NEG_INFINITY,
            min_validation_mean_adaptive_over_regular_material_matched_budget_psnr_gain_db:
                f32::NEG_INFINITY,
            min_validation_worst_adaptive_over_regular_material_matched_budget_psnr_gain_db:
                f32::NEG_INFINITY,
            min_validation_mean_adaptive_over_budget_fixed_psnr_gain_db: f32::NEG_INFINITY,
            min_validation_worst_adaptive_over_budget_fixed_psnr_gain_db: f32::NEG_INFINITY,
            min_validation_controller_oracle_refinement_scale_correlation: -1.0,
            max_validation_measure_relative_drift: f64::INFINITY,
            max_validation_leaf_relative_error: 1.0,
            require_active_leaf_dynamics: false,
            max_validation_interaction_particle_ratio: f64::INFINITY,
            max_validation_mean_adaptive_rollout_elapsed_ms: f64::INFINITY,
            max_validation_mean_adaptive_topology_elapsed_ms: f64::INFINITY,
            max_validation_topology_update_elapsed_ms: f64::INFINITY,
            min_gap_final_selected_mode_vs_fine_control_db: f32::NEG_INFINITY,
            min_gap_post_cut_recurrent_change_vs_fine_control_db: f32::NEG_INFINITY,
            max_gap_controller_target_render_regret_db: f32::INFINITY,
            max_rollout_target_relative_error: 1.0,
            max_rollout_measure_relative_drift: 1.0e-6,
            min_rollout_footprint_coefficient_of_variation: 0.0,
            min_rollout_occupied_material_scale_bins: 0,
            min_rollout_fractional_material_scale_fraction: 0.0,
            min_rollout_particle_steps_per_second: 0.0,
            require_graph_cap_and_search_parity: true,
            max_sparse_conservation_error: default_sparse_conservation_error(),
            max_sparse_protected_band_nrmse: default_sparse_protected_band_nrmse(),
            min_sparse_largest_cap_count_reduction: 0.0,
            max_sparse_boundary_hd95_voxels: f64::INFINITY,
        }
    }
}

const fn default_sparse_conservation_error() -> f64 {
    1.0e-12
}

const fn one_f64() -> f64 {
    1.0
}

const fn default_sparse_protected_band_nrmse() -> f64 {
    1.0e-12
}

const fn infinite_f64() -> f64 {
    f64::INFINITY
}

const fn infinite_f32() -> f32 {
    f32::INFINITY
}

const fn negative_one_f32() -> f32 {
    -1.0
}

const fn negative_infinity_f32() -> f32 {
    f32::NEG_INFINITY
}

const fn one_f32() -> f32 {
    1.0
}

const fn default_true() -> bool {
    true
}

const fn default_structural_audit_seeds() -> usize {
    1
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AdaptiveTaskQualityConfig {
    #[serde(default)]
    pub enabled: bool,
    pub target_image: PathBuf,
    pub image_size: usize,
    pub rollout_steps: usize,
    pub update_prob: f32,
    pub seed: u64,
    /// Matched stochastic-mask seeds used for broad adaptive/regular parity.
    /// The primary `seed` remains the detailed task-quality report.
    #[serde(default)]
    pub validation_seeds: Vec<u64>,
    /// Number of aggregate-validation seeds that run the expensive canonical
    /// refinement/controller audit. Image and rollout quality use every seed.
    #[serde(default = "default_structural_audit_seeds")]
    pub structural_audit_seeds: usize,
    #[serde(default = "default_true")]
    pub bandwidth_adaptation_enabled: bool,
    /// Released/reference model used only for matched evaluation. It is never
    /// used to initialize a fresh adaptive training run.
    #[serde(default)]
    pub reference_model: Option<PathBuf>,
    #[serde(default)]
    pub topology_control: AdaptiveTopologyControl,
    /// Decoder used only for task-quality rendering. The canonical quadrature
    /// path preserves the active material budget while allowing a coarse leaf
    /// to emit bounded render samples from its covariance and state Jacobian.
    #[serde(default)]
    pub render_decoder: AdaptiveRenderDecoder,
    /// Mass-preserving contraction strength for `compact-moment-gaussian`.
    /// Zero is the covariance moment Gaussian and one is the unit-opacity
    /// equal-measure ellipse.
    #[serde(default = "one_f32")]
    pub render_compactness: f32,
    /// Selection rule used only when the model schedules a hierarchical
    /// fine-to-coarse restriction during task-quality evaluation.
    #[serde(default)]
    pub restriction_policy: AdaptiveTaskRestrictionPolicy,
    /// Bounded matched-seed audit that separates immediate render/restriction
    /// loss from recurrent post-cut drift and persistent-mode compression.
    #[serde(default)]
    pub gap_decomposition: AdaptiveGapDecompositionConfig,
}

impl Default for AdaptiveTaskQualityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            target_image: PathBuf::new(),
            image_size: 128,
            rollout_steps: 120,
            update_prob: 0.5,
            seed: 42,
            validation_seeds: Vec::new(),
            structural_audit_seeds: default_structural_audit_seeds(),
            bandwidth_adaptation_enabled: true,
            reference_model: None,
            topology_control: AdaptiveTopologyControl::Learned,
            render_decoder: AdaptiveRenderDecoder::default(),
            render_compactness: 1.0,
            restriction_policy: AdaptiveTaskRestrictionPolicy::default(),
            gap_decomposition: AdaptiveGapDecompositionConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AdaptiveGapDecompositionConfig {
    pub enabled: bool,
    /// Absolute rollout horizons. Empty selects a bounded cut-centered set.
    pub horizons: Vec<usize>,
    /// Persistent modes retained per coarse leaf. The selected deployment
    /// count and full canonical child count are always included.
    pub mode_counts: Vec<usize>,
    /// Limits this expensive diagnostic independently of broad parity seeds.
    pub max_seeds: usize,
    /// Also evaluate the former covariance-shaped compact decoder as a
    /// diagnostic control. It is never the promoted render result.
    pub covariance_decoder_control: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdaptiveTaskRestrictionPolicy {
    /// Deployable target-independent hierarchy score derived from NPA state and
    /// update variation.
    #[default]
    DynamicsDetail,
    /// Target-independent scorer distilled from target-render merge decisions.
    LearnedController,
    /// Evaluation-only ceiling that ranks canonical 4-to-1 merges by their
    /// exact single-merge composited image loss against the known target.
    TargetRenderOracle,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveRestrictionExperimentConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub training_data: AdaptiveRestrictionDatasetConfig,
    #[serde(default = "default_restriction_validation_data")]
    pub validation_data: AdaptiveRestrictionDatasetConfig,
    #[serde(default)]
    pub training: AdaptiveControllerTrainConfig,
}

impl Default for AdaptiveRestrictionExperimentConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            training_data: AdaptiveRestrictionDatasetConfig::default(),
            validation_data: default_restriction_validation_data(),
            training: AdaptiveControllerTrainConfig::default(),
        }
    }
}

fn default_restriction_validation_data() -> AdaptiveRestrictionDatasetConfig {
    AdaptiveRestrictionDatasetConfig {
        seeds: (50..58).collect(),
        ..AdaptiveRestrictionDatasetConfig::default()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AdaptiveScalingExperimentConfig {
    pub resolutions: Vec<usize>,
    pub interior_spacing_cap: f32,
    pub protected_band_voxels: f32,
    pub transition_divisor: f32,
    pub retention_samples: usize,
    #[serde(default = "default_quality_resolution")]
    pub quality_resolution: usize,
    #[serde(default = "default_quality_spacing_cap_ratios")]
    pub quality_spacing_cap_ratios: Vec<f32>,
    #[serde(default = "default_sphere_outer_radius")]
    pub sphere_outer_radius: f32,
    #[serde(default = "default_sphere_cavity_center")]
    pub sphere_cavity_center: [f32; 3],
    #[serde(default = "default_sphere_cavity_radius")]
    pub sphere_cavity_radius: f32,
    #[serde(default = "default_torus_major_radius")]
    pub torus_major_radius: f32,
    #[serde(default = "default_torus_minor_radius")]
    pub torus_minor_radius: f32,
}

impl Default for AdaptiveScalingExperimentConfig {
    fn default() -> Self {
        Self {
            resolutions: vec![32, 48, 64, 96],
            interior_spacing_cap: 0.24,
            protected_band_voxels: 2.25,
            transition_divisor: 1.65,
            retention_samples: 3,
            quality_resolution: default_quality_resolution(),
            quality_spacing_cap_ratios: default_quality_spacing_cap_ratios(),
            sphere_outer_radius: default_sphere_outer_radius(),
            sphere_cavity_center: default_sphere_cavity_center(),
            sphere_cavity_radius: default_sphere_cavity_radius(),
            torus_major_radius: default_torus_major_radius(),
            torus_minor_radius: default_torus_minor_radius(),
        }
    }
}

const fn default_quality_resolution() -> usize {
    64
}

fn default_quality_spacing_cap_ratios() -> Vec<f32> {
    vec![1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
}

const fn default_sphere_outer_radius() -> f32 {
    0.85
}

const fn default_sphere_cavity_center() -> [f32; 3] {
    [0.22, -0.12, 0.10]
}

const fn default_sphere_cavity_radius() -> f32 {
    0.22
}

const fn default_torus_major_radius() -> f32 {
    0.60
}

const fn default_torus_minor_radius() -> f32 {
    318.0 / 1_000.0
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AdaptiveOperatorExperimentConfig {
    pub side: usize,
    #[serde(default = "default_operator_side_3d")]
    pub side_3d: usize,
    pub jitter: f32,
    pub sparse_side_stride: usize,
}

impl Default for AdaptiveOperatorExperimentConfig {
    fn default() -> Self {
        Self {
            side: 33,
            side_3d: default_operator_side_3d(),
            jitter: 5.0e-4,
            sparse_side_stride: 2,
        }
    }
}

const fn default_operator_side_3d() -> usize {
    17
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AdaptiveTopologyExperimentConfig {
    pub samples: usize,
    /// Largest child-measure ratio sampled by the constrained unequal-event
    /// audit. One reduces the audit to the canonical equal event.
    #[serde(default = "default_topology_unequal_measure_ratio")]
    pub max_unequal_measure_ratio: f64,
}

/// Reproducible, topology-only audit bundle. This deliberately avoids model
/// loading, rollout, and training so conservative event algebra can be gated
/// independently from task quality and device throughput.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveTopologyAuditConfig {
    pub report_output: PathBuf,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default)]
    pub topology: AdaptiveTopologyExperimentConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveTopologyAuditReport {
    pub schema_version: u32,
    pub seed: u64,
    pub topology_config: AdaptiveTopologyExperimentConfig,
    pub topology: AdaptiveTopologyExperimentReport,
}

impl Default for AdaptiveTopologyExperimentConfig {
    fn default() -> Self {
        Self {
            samples: 100_000,
            max_unequal_measure_ratio: default_topology_unequal_measure_ratio(),
        }
    }
}

const fn default_topology_unequal_measure_ratio() -> f64 {
    8.0
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AdaptiveGraphExperimentConfig {
    #[serde(default = "default_graph_spatial_dims")]
    pub spatial_dims: Vec<usize>,
    pub particle_counts: Vec<usize>,
    pub neighbor_caps: Vec<usize>,
    #[serde(default = "default_all_pairs_baseline_max_particles")]
    pub all_pairs_baseline_max_particles: usize,
    pub coarse_fraction: f32,
    pub fine_bandwidth: f32,
    pub coarse_bandwidth: f32,
    #[serde(default = "default_graph_target_fine_degree")]
    pub target_fine_degree: f32,
    #[serde(default = "default_graph_warmup_runs")]
    pub warmup_runs: usize,
    #[serde(default = "default_graph_timed_runs")]
    pub timed_runs: usize,
}

impl Default for AdaptiveGraphExperimentConfig {
    fn default() -> Self {
        Self {
            spatial_dims: default_graph_spatial_dims(),
            particle_counts: vec![1_024, 4_096],
            neighbor_caps: vec![16, 32, 64],
            all_pairs_baseline_max_particles: default_all_pairs_baseline_max_particles(),
            coarse_fraction: 0.25,
            fine_bandwidth: 0.035,
            coarse_bandwidth: 0.14,
            target_fine_degree: default_graph_target_fine_degree(),
            warmup_runs: default_graph_warmup_runs(),
            timed_runs: default_graph_timed_runs(),
        }
    }
}

fn default_graph_spatial_dims() -> Vec<usize> {
    vec![2, 3]
}

const fn default_all_pairs_baseline_max_particles() -> usize {
    1_024
}

const fn default_graph_target_fine_degree() -> f32 {
    48.0
}

const fn default_graph_warmup_runs() -> usize {
    2
}

const fn default_graph_timed_runs() -> usize {
    10
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AdaptiveRolloutExperimentConfig {
    pub enabled: bool,
    pub particles: usize,
    pub seed_scale: f32,
    pub total_measure: f32,
    #[serde(default)]
    pub rollout: AdaptiveRolloutConfig,
}

impl Default for AdaptiveRolloutExperimentConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            particles: 1_024,
            seed_scale: 0.2,
            total_measure: std::f32::consts::PI * 0.2 * 0.2,
            rollout: AdaptiveRolloutConfig {
                steps: 32,
                snapshot_interval: 8,
                ..AdaptiveRolloutConfig::default()
            },
        }
    }
}

const fn default_seed() -> u64 {
    42
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveExperimentReport {
    pub schema_version: u32,
    pub gates_passed: bool,
    pub gate_failures: Vec<String>,
    pub paper_scope: String,
    pub base_model_source: String,
    pub base_training: Option<AdaptiveBaseTrainingReport>,
    pub rule_perception: crate::adaptive::AdaptiveRulePerception,
    pub operator: AdaptiveOperatorExperimentReport,
    pub operator_3d: AdaptiveOperatorExperimentReport,
    pub topology: AdaptiveTopologyExperimentReport,
    pub graph: Vec<AdaptiveGraphExperimentRow>,
    pub scaling: AdaptiveScalingExperimentReport,
    pub training: crate::adaptive::AdaptiveControllerTrainingReport,
    pub validation: AdaptiveControllerValidationReport,
    pub rule_distillation: Option<crate::adaptive::AdaptiveRuleDistillationReport>,
    pub multiscale_training: Option<AdaptiveMultiscaleExperimentReport>,
    #[serde(default)]
    pub closure_identifiability: Option<AdaptiveClosureIdentifiabilityReport>,
    #[serde(default)]
    pub restriction_training: Option<AdaptiveRestrictionExperimentReport>,
    pub task_quality: Option<AdaptiveTaskQualityReport>,
    #[serde(default)]
    pub task_quality_validation: Option<AdaptiveTaskQualityValidationReport>,
    pub rollout: Option<AdaptiveRolloutExperimentReport>,
    pub model_output: String,
    pub model_sha256: String,
    pub total_elapsed_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveRestrictionExperimentReport {
    pub training_dataset: AdaptiveRestrictionDatasetReport,
    pub validation_dataset: AdaptiveRestrictionDatasetReport,
    pub training: crate::adaptive::AdaptiveControllerTrainingReport,
    pub training_selection: AdaptiveRestrictionSelectionReport,
    pub heldout_selection: AdaptiveRestrictionSelectionReport,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveTaskQualityValidationReport {
    pub rows: Vec<AdaptiveTaskQualityReport>,
    #[serde(default)]
    pub structural_audit_seeds: usize,
    pub mean_adaptive_target_composited_psnr_db: f32,
    /// Mean target PSNR of the explicit task-quality reference model. For the
    /// maintained lizard evaluation this is the released upstream NPA.
    #[serde(default)]
    pub mean_teacher_target_composited_psnr_db: f32,
    #[serde(default)]
    pub mean_adaptive_over_teacher_psnr_gain_db: f32,
    #[serde(default)]
    pub worst_adaptive_over_teacher_psnr_gain_db: f32,
    /// Mean target PSNR of the rule embedded in the adaptive artifact. This is
    /// intentionally reported separately because it need not equal `teacher`.
    pub mean_regular_base_target_composited_psnr_db: f32,
    pub mean_adaptive_over_regular_base_psnr_gain_db: f32,
    pub worst_adaptive_over_regular_base_psnr_gain_db: f32,
    #[serde(default)]
    pub mean_regular_matched_budget_target_composited_psnr_db: f32,
    #[serde(default)]
    pub mean_adaptive_over_regular_matched_budget_psnr_gain_db: f32,
    #[serde(default)]
    pub worst_adaptive_over_regular_matched_budget_psnr_gain_db: f32,
    #[serde(default)]
    pub mean_regular_material_matched_budget_target_composited_psnr_db: f32,
    #[serde(default)]
    pub mean_adaptive_over_regular_material_matched_budget_psnr_gain_db: f32,
    #[serde(default)]
    pub worst_adaptive_over_regular_material_matched_budget_psnr_gain_db: f32,
    pub mean_adaptive_over_budget_fixed_psnr_gain_db: f32,
    pub worst_adaptive_over_budget_fixed_psnr_gain_db: f32,
    pub minimum_controller_oracle_refinement_scale_correlation: f32,
    pub maximum_measure_relative_drift: f64,
    pub maximum_leaf_relative_error: f64,
    #[serde(default)]
    pub minimum_final_occupied_material_scale_bins: usize,
    #[serde(default)]
    pub minimum_final_fractional_material_scale_fraction: f32,
    #[serde(default)]
    pub minimum_final_dyadic_scale_quantization_rmse_octaves: f32,
    #[serde(default)]
    pub mean_adaptive_rollout_elapsed_ms: f64,
    #[serde(default)]
    pub mean_adaptive_topology_elapsed_ms: f64,
    #[serde(default)]
    pub maximum_topology_update_elapsed_ms: f64,
    #[serde(default)]
    pub gap_decomposition: Option<AdaptiveGapDecompositionReport>,
}

/// Recurrent state carried between adaptive rollout steps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdaptiveDynamicsSemantics {
    /// Only active material leaves persist between steps.
    #[default]
    ActiveLeaves,
    /// Fine sub-leaf modes persist between steps and are restricted only for
    /// material output. This is a teacher/renderer ceiling, not active compute.
    PersistentHiddenFineModes,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveGapDecompositionReport {
    pub cut_step: usize,
    pub final_horizon: usize,
    pub regular_fine_particles: usize,
    pub selected_mode_count: usize,
    pub full_mode_count: usize,
    pub render_contract: String,
    pub rows: Vec<AdaptiveGapDecompositionRow>,
    pub mean_cut_only_full_mode_gap_db: Option<f32>,
    pub mean_final_full_mode_gap_db: Option<f32>,
    pub mean_final_selected_mode_gap_db: Option<f32>,
    pub mean_post_cut_recurrent_gap_change_db: Option<f32>,
    /// Final selected-mode PSNR minus the matched uncut adaptive fine control.
    /// This is the topology-only quality gap used by promotion gates.
    #[serde(default)]
    pub mean_final_selected_mode_gap_vs_fine_control_db: Option<f32>,
    /// Change in full-mode topology-only gap from cut time to final horizon.
    #[serde(default)]
    pub mean_post_cut_recurrent_gap_change_vs_fine_control_db: Option<f32>,
    pub mean_selected_mode_compression_penalty_db: Option<f32>,
    pub mean_covariance_decoder_advantage_db: Option<f32>,
    /// Numerical drift introduced by the adaptive resident path before any
    /// hierarchy cut. This should stay near zero and is not a coarse-model gap.
    #[serde(default)]
    pub mean_final_uncut_fine_control_gap_db: Option<f32>,
    /// Internal full-quadrature PSNR minus the matched uncut adaptive fine
    /// control. This isolates recurrent dynamics from visible-leaf decoding.
    #[serde(default)]
    pub mean_final_full_mode_internal_gap_db: Option<f32>,
    /// Visible isotropic-leaf PSNR minus internal full-quadrature PSNR. A
    /// negative value is information lost only when modes are restricted for
    /// rendering, not by the recurrent dynamics themselves.
    #[serde(default)]
    pub mean_final_full_mode_visible_decode_penalty_db: Option<f32>,
    /// Final-horizon PSNR from applying a fresh target-independent dynamics
    /// detail cut to the exact uncut fine control, minus that fine control.
    #[serde(default)]
    pub mean_final_late_dynamics_cut_gap_db: Option<f32>,
    /// Final-horizon PSNR from applying the trained restriction controller to
    /// the exact uncut fine control, minus that fine control.
    #[serde(default)]
    pub mean_final_late_learned_cut_gap_db: Option<f32>,
    /// Final-horizon PSNR from ranking the deployment cut with the known target
    /// render, minus the exact uncut fine control. This is an evaluation-only
    /// ceiling for the current mixed-arity action family.
    #[serde(default)]
    pub mean_final_late_target_render_cut_gap_db: Option<f32>,
    /// Target-render-ranked late-cut PSNR minus learned late-cut PSNR at the
    /// final horizon. Both cuts use the same topology action family.
    #[serde(default)]
    pub mean_final_controller_target_render_regret_db: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveGapDecompositionRow {
    pub seed: u64,
    pub horizon: usize,
    pub post_cut_steps: usize,
    pub mode_count: usize,
    pub visible_particles: usize,
    pub dynamics_particles: usize,
    pub regular_fine_psnr_db: f32,
    #[serde(default)]
    pub adaptive_fine_control_psnr_db: f32,
    #[serde(default)]
    pub adaptive_fine_control_gap_db: f32,
    #[serde(default)]
    pub internal_mode_psnr_db: f32,
    #[serde(default)]
    pub internal_mode_gap_vs_fine_control_db: f32,
    pub adaptive_isotropic_psnr_db: f32,
    pub adaptive_isotropic_gap_db: f32,
    #[serde(default)]
    pub visible_decode_penalty_db: f32,
    #[serde(default)]
    pub late_dynamics_cut_psnr_db: Option<f32>,
    #[serde(default)]
    pub late_dynamics_cut_gap_db: Option<f32>,
    #[serde(default)]
    pub late_learned_cut_psnr_db: Option<f32>,
    #[serde(default)]
    pub late_learned_cut_gap_db: Option<f32>,
    #[serde(default)]
    pub late_target_render_cut_psnr_db: Option<f32>,
    #[serde(default)]
    pub late_target_render_cut_gap_db: Option<f32>,
    pub covariance_control_psnr_db: Option<f32>,
    pub covariance_control_advantage_db: Option<f32>,
    pub maximum_covariance_axis_ratio: f32,
    pub represented_measure_relative_drift: f64,
    pub rollout_elapsed_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveBaseTrainingReport {
    pub initializer: String,
    pub target_image: String,
    pub reference_model: Option<String>,
    pub target_points: usize,
    pub phases: Vec<AdaptiveBaseTrainingPhaseReport>,
    pub model_output: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveBaseTrainingPhaseReport {
    pub name: String,
    pub particle_count: usize,
    pub training: Target2dGpuTrainingReport,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveTaskQualityReport {
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub structural_audit_performed: bool,
    #[serde(default)]
    pub topology_control: AdaptiveTopologyControl,
    pub rollout_steps: usize,
    #[serde(default)]
    pub bandwidth_adaptation_enabled: bool,
    #[serde(default)]
    pub bandwidth_adaptation_active: bool,
    pub teacher_particles: usize,
    pub adaptive_initial_particles: usize,
    #[serde(default)]
    pub adaptive_target_particles: usize,
    pub adaptive_final_particles: usize,
    pub adaptive_min_particles: usize,
    pub adaptive_max_particles: usize,
    #[serde(default)]
    pub render_decoder: AdaptiveRenderDecoder,
    #[serde(default)]
    pub restriction_policy: AdaptiveTaskRestrictionPolicy,
    #[serde(default)]
    pub adaptive_render_primitives: usize,
    /// Persistent internal rows advanced by the NPA rule. This can exceed the
    /// visible material/render budget when coarse leaves retain sub-leaf modes.
    #[serde(default)]
    pub adaptive_dynamics_particles: usize,
    /// Classifies whether `adaptive_dynamics_particles` is the actual active
    /// material state or only a visible projection of hidden recurrent modes.
    #[serde(default)]
    pub dynamics_semantics: AdaptiveDynamicsSemantics,
    /// Rows evaluated by the NPA rule per dynamics step. Stateless quadrature
    /// can exceed the recurrent active-leaf count without retaining hidden
    /// state across steps.
    #[serde(default)]
    pub adaptive_interaction_particles: usize,
    pub teacher_target_composited_psnr_db: f32,
    #[serde(default)]
    pub regular_base_target_composited_psnr_db: f32,
    #[serde(default)]
    pub regular_matched_budget_target_composited_psnr_db: f32,
    #[serde(default)]
    pub regular_material_matched_budget_target_composited_psnr_db: f32,
    pub adaptive_fine_fixed_target_composited_psnr_db: f32,
    pub adaptive_fine_fixed_teacher_composited_psnr_db: f32,
    pub adaptive_fine_fixed_teacher_psnr_gap_db: f32,
    #[serde(default)]
    pub adaptive_budget_frozen_base_target_composited_psnr_db: f32,
    #[serde(default)]
    pub adaptive_budget_local_only_target_composited_psnr_db: f32,
    pub adaptive_budget_fixed_target_composited_psnr_db: f32,
    #[serde(default)]
    pub adaptive_budget_fixed_no_bandwidth_target_composited_psnr_db: f32,
    #[serde(default)]
    pub bandwidth_adaptation_target_psnr_gain_db: f32,
    pub adaptive_budget_fixed_teacher_composited_psnr_db: f32,
    #[serde(default)]
    pub local_residual_target_psnr_gain_db: f32,
    #[serde(default)]
    pub proxy_residual_target_psnr_gain_db: f32,
    pub adaptive_target_composited_psnr_db: f32,
    #[serde(default)]
    pub adaptive_training_policy_target_composited_psnr_db: f32,
    #[serde(default)]
    pub adaptive_over_regular_base_psnr_gain_db: f32,
    #[serde(default)]
    pub adaptive_over_regular_matched_budget_psnr_gain_db: f32,
    #[serde(default)]
    pub adaptive_over_regular_material_matched_budget_psnr_gain_db: f32,
    #[serde(default)]
    pub deployment_over_training_policy_psnr_gain_db: f32,
    pub adaptive_teacher_composited_psnr_db: f32,
    pub adaptive_teacher_psnr_gap_db: f32,
    pub teacher_target_density_psnr_db: f32,
    pub adaptive_target_density_psnr_db: f32,
    pub final_min_footprint: f32,
    pub final_max_footprint: f32,
    #[serde(default)]
    pub final_min_render_footprint: f32,
    #[serde(default)]
    pub final_max_render_footprint: f32,
    #[serde(default)]
    pub final_mean_render_to_material_footprint_ratio: f32,
    #[serde(default)]
    pub final_max_render_target_relative_error: f32,
    pub final_footprint_coefficient_of_variation: f32,
    pub maximum_footprint_coefficient_of_variation: f32,
    /// Occupied represented-material scales at 1/64-octave audit resolution.
    /// This is independent from renderer interpolation and support bins.
    #[serde(default)]
    pub final_occupied_material_scale_bins: usize,
    /// Fraction of final material leaves off integer octaves relative to the
    /// model's reference footprint.
    #[serde(default)]
    pub final_fractional_material_scale_fraction: f32,
    #[serde(default)]
    pub final_dyadic_scale_quantization_rmse_octaves: f32,
    #[serde(default)]
    pub fine_leaf_count: usize,
    #[serde(default)]
    pub reference_leaf_count: usize,
    #[serde(default)]
    pub coarse_leaf_count: usize,
    #[serde(default)]
    pub fine_represented_measure_fraction: f32,
    #[serde(default)]
    pub reference_represented_measure_fraction: f32,
    #[serde(default)]
    pub coarse_represented_measure_fraction: f32,
    pub total_split_events: usize,
    pub total_merge_events: usize,
    #[serde(default)]
    pub bootstrap_split_events: usize,
    /// Merge-equivalent events from the one-shot scheduled hierarchy cut.
    /// These are not learned steady topology reallocation.
    #[serde(default)]
    pub restriction_merge_events: usize,
    #[serde(default)]
    pub steady_split_events: usize,
    #[serde(default)]
    pub steady_merge_events: usize,
    #[serde(default)]
    pub mean_event_state_transfer_rms: f32,
    #[serde(default)]
    pub max_event_state_transfer_rms: f32,
    #[serde(default)]
    pub max_split_probability: f32,
    #[serde(default)]
    pub max_merge_probability: f32,
    #[serde(default)]
    pub max_compatible_merge_probability: f32,
    #[serde(default)]
    pub max_eligible_split_candidates: usize,
    #[serde(default)]
    pub max_eligible_merge_clusters: usize,
    pub mean_proxy_messages: f32,
    pub measure_relative_drift: f64,
    /// Correlation between local state-gradient detail and inverse footprint.
    #[serde(default)]
    pub detail_density_correlation: f32,
    /// Mean footprint in the highest-detail quartile divided by the lowest.
    #[serde(default)]
    pub high_to_low_detail_footprint_ratio: f32,
    /// Correlation between the frozen base rule's canonical refinement defect
    /// and inverse material footprint.
    #[serde(default)]
    pub refinement_defect_density_correlation: f32,
    /// Mean footprint of the lowest-defect quartile divided by the highest.
    #[serde(default)]
    pub low_to_high_refinement_defect_footprint_ratio: f32,
    #[serde(default)]
    pub budget_fixed_mean_refinement_defect: f32,
    #[serde(default)]
    pub adaptive_mean_refinement_defect: f32,
    #[serde(default)]
    pub adaptive_refinement_defect_relative_gain: f32,
    #[serde(default)]
    pub controller_oracle_refinement_scale_correlation: f32,
    #[serde(default)]
    pub oracle_min_desired_footprint_ratio: f32,
    #[serde(default)]
    pub oracle_max_desired_footprint_ratio: f32,
    #[serde(default)]
    pub controller_min_desired_footprint_ratio: f32,
    #[serde(default)]
    pub controller_max_desired_footprint_ratio: f32,
    #[serde(default)]
    pub minimum_desired_footprint_ratio: f32,
    #[serde(default)]
    pub maximum_desired_footprint_ratio: f32,
    /// Resident WGPU state construction, rollout, topology, and final readback.
    #[serde(default)]
    pub adaptive_rollout_elapsed_ms: f64,
    /// Sum of bounded topology update latency within the adaptive rollout.
    #[serde(default)]
    pub adaptive_topology_elapsed_ms: f64,
    #[serde(default)]
    pub maximum_topology_update_elapsed_ms: f64,
    #[serde(default)]
    pub adaptive_topology_updates: Vec<AdaptiveTopologyTimingRow>,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveTopologyTimingRow {
    pub step: usize,
    pub split_events: usize,
    pub merge_events: usize,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveMultiscaleExperimentReport {
    pub training_dataset: crate::adaptive::AdaptiveMultiscaleDatasetReport,
    pub validation_dataset: crate::adaptive::AdaptiveMultiscaleDatasetReport,
    pub training: crate::adaptive::AdaptiveMultiscaleRuleTrainingReport,
    #[serde(default)]
    pub closure_training: crate::adaptive::AdaptiveClosureModeTrainingReport,
    pub on_policy_datasets: Vec<crate::adaptive::AdaptiveMultiscaleDatasetReport>,
    pub on_policy_training: Vec<crate::adaptive::AdaptiveMultiscaleRuleTrainingReport>,
    #[serde(default)]
    pub on_policy_closure_training: Vec<crate::adaptive::AdaptiveClosureModeTrainingReport>,
    pub on_policy_controller_training: Vec<crate::adaptive::AdaptiveControllerTrainingReport>,
    pub heldout_validation: crate::adaptive::AdaptiveMultiscaleRuleValidationReport,
    #[serde(default)]
    pub heldout_closure_validation: crate::adaptive::AdaptiveClosureModeValidationReport,
    #[serde(default)]
    pub heldout_on_policy_dataset: Option<crate::adaptive::AdaptiveMultiscaleDatasetReport>,
    #[serde(default)]
    pub heldout_on_policy_validation:
        Option<crate::adaptive::AdaptiveMultiscaleRuleValidationReport>,
    #[serde(default)]
    pub heldout_on_policy_closure_validation:
        Option<crate::adaptive::AdaptiveClosureModeValidationReport>,
    #[serde(default)]
    pub deployment_training: crate::adaptive::AdaptiveDeploymentRuleTrainingReport,
    #[serde(default)]
    pub deployment_on_policy_datasets: Vec<crate::adaptive::AdaptiveMultiscaleDatasetReport>,
    #[serde(default)]
    pub deployment_on_policy_training: Vec<crate::adaptive::AdaptiveDeploymentRuleTrainingReport>,
    #[serde(default)]
    pub heldout_deployment_validation: crate::adaptive::AdaptiveDeploymentRuleValidationReport,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AdaptiveScalingExperimentReport {
    pub rows: Vec<AdaptiveScalingExperimentRow>,
    pub fits: Vec<AdaptiveScalingFit>,
    pub quality_rows: Vec<AdaptiveSparseQualityRow>,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AdaptiveSparseQualityRow {
    pub solid: String,
    pub allocation: String,
    pub resolution: usize,
    pub spacing_cap_ratio: f32,
    pub sample: usize,
    pub fine_leaves: usize,
    pub retained_leaves: usize,
    pub protected_leaves: usize,
    pub count_reduction: f64,
    pub field_nrmse: f64,
    pub protected_band_nrmse: f64,
    pub measure_relative_error: f64,
    pub centroid_l2_error: f64,
    pub field_integral_relative_error: f64,
    pub quadratic_integral_loss_fraction: f64,
    pub median_deep_footprint_ratio: f64,
    pub median_boundary_footprint_ratio: f64,
    pub boundary_hd95_voxels: f64,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AdaptiveScalingExperimentRow {
    pub solid: String,
    pub resolution: usize,
    pub fine_leaves: usize,
    pub protected_leaves: usize,
    pub expected_retained_leaves: f64,
    pub retained_mean: f64,
    pub retained_stddev: f64,
    pub count_reduction: f64,
    pub mean_spacing_ratio: f64,
    pub max_spacing_ratio: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AdaptiveScalingFit {
    pub solid: String,
    pub full_count_exponent: f64,
    pub tail_count_exponent: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AdaptiveOperatorExperimentReport {
    pub spatial_dims: usize,
    pub particles: usize,
    pub fixed_constant_max_error: f32,
    pub adaptive_constant_max_error: f32,
    pub adaptive_affine_gradient_mean_error: f32,
    pub interface_affine_gradient_mean_error: f32,
    pub moment_fallback_fraction: f32,
    pub partition_min: f32,
    pub partition_max: f32,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AdaptiveTopologyExperimentReport {
    pub samples: usize,
    #[serde(default)]
    pub canonical_events: usize,
    #[serde(default)]
    pub unequal_events: usize,
    #[serde(default)]
    pub maximum_sampled_child_measure_ratio: f64,
    pub events_per_second: f64,
    pub max_measure_relative_error: f64,
    pub max_centroid_l2_error: f64,
    pub max_second_moment_relative_error: f64,
    pub max_extensive_relative_error: f64,
    pub max_determinant_scale_relative_error: f64,
    pub spd_failures: usize,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AdaptiveGraphExperimentRow {
    pub spatial_dims: usize,
    pub particles: usize,
    pub search: String,
    pub policy: String,
    pub neighbor_cap: usize,
    #[serde(default)]
    pub candidate_visits: usize,
    pub raw_messages: usize,
    pub accepted_messages: usize,
    pub degree_mean: f32,
    pub degree_p95: usize,
    pub degree_max: usize,
    pub isolated_particles: usize,
    pub cross_scale_fraction: f32,
    pub elapsed_ms: f64,
    pub elapsed_ms_stddev: f64,
    pub elapsed_ms_min: f64,
    pub timed_runs: usize,
    pub messages_per_second: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AdaptiveControllerValidationReport {
    pub rows: usize,
    pub mean_squared_error: f32,
    pub channel_mean_squared_error: [f32; 4],
    pub desired_scale_correlation: f32,
    pub event_positive_fraction: [f32; 2],
    pub event_precision: [f32; 2],
    pub event_recall: [f32; 2],
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AdaptiveRolloutExperimentReport {
    pub bandwidth_adaptation_requested: bool,
    pub bandwidth_adaptation_active: bool,
    pub topology_enabled: bool,
    pub initial_leaves: usize,
    pub final_leaves: usize,
    pub target_leaves: usize,
    pub target_leaf_relative_error: f64,
    pub minimum_leaves: usize,
    pub maximum_leaves: usize,
    pub final_min_footprint: f32,
    pub final_max_footprint: f32,
    pub final_mean_footprint: f32,
    pub final_footprint_coefficient_of_variation: f32,
    #[serde(default)]
    pub final_occupied_material_scale_bins: usize,
    #[serde(default)]
    pub final_fractional_material_scale_fraction: f32,
    #[serde(default)]
    pub final_dyadic_scale_quantization_rmse_octaves: f32,
    pub final_min_generation: u16,
    pub final_max_generation: u16,
    pub initial_measure: f64,
    pub final_measure: f64,
    pub measure_relative_drift: f64,
    pub total_split_events: usize,
    pub total_merge_events: usize,
    #[serde(default)]
    pub mean_event_state_transfer_rms: f32,
    #[serde(default)]
    pub max_event_state_transfer_rms: f32,
    #[serde(default)]
    pub max_split_probability: f32,
    #[serde(default)]
    pub max_merge_probability: f32,
    #[serde(default)]
    pub max_compatible_merge_probability: f32,
    #[serde(default)]
    pub max_eligible_split_candidates: usize,
    #[serde(default)]
    pub max_eligible_merge_clusters: usize,
    #[serde(default)]
    pub minimum_desired_footprint_ratio: f32,
    #[serde(default)]
    pub maximum_desired_footprint_ratio: f32,
    pub mean_accepted_messages: f64,
    pub moment_fallback_fraction: f64,
    pub elapsed_ms: f64,
    pub particle_steps_per_second: f64,
    #[serde(default)]
    pub mean_perception_ms: f64,
    #[serde(default)]
    pub mean_controller_ms: f64,
    #[serde(default)]
    pub mean_local_rule_ms: f64,
    #[serde(default)]
    pub mean_proxy_rule_ms: f64,
    #[serde(default)]
    pub mean_integration_ms: f64,
    #[serde(default)]
    pub mean_topology_ms: f64,
    #[serde(default)]
    pub mean_total_step_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive::AdaptiveRestrictionArity;

    #[test]
    fn controller_training_can_be_frozen_without_repeating_defaults() {
        let frozen: AdaptiveControllerTrainConfig = toml::from_str("enabled = false").unwrap();
        assert!(!frozen.enabled);
        assert_eq!(frozen.steps, AdaptiveControllerTrainConfig::default().steps);

        let legacy: AdaptiveControllerTrainConfig = toml::from_str("").unwrap();
        assert!(legacy.enabled);
    }

    #[test]
    fn adaptive_quality_defaults_to_isotropic_material_rendering() {
        let quality = AdaptiveTaskQualityConfig::default();
        assert_eq!(
            quality.render_decoder,
            AdaptiveRenderDecoder::IsotropicMaterialGaussian
        );
        assert!(!quality.gap_decomposition.enabled);
        assert_eq!(quality.gap_decomposition.max_seeds, 0);
    }

    #[test]
    fn fresh_base_phase_inherits_canonical_target2d_defaults() {
        let config: AdaptiveExperimentConfig = toml::from_str(
            r#"
report_output = "artifacts/report.json"
model_output = "artifacts/model.bpk"
backend = "cuda"

[base_training]
enabled = true
target_image = "assets/lizard.png"
reference_model = "models/lizard.bpk"

[[base_training.phases]]
name = "fine"
particle_count = 3072
epochs = 99

[task_quality]
enabled = false
target_image = ""
image_size = 128
rollout_steps = 256
update_prob = 0.5
seed = 42
reference_model = "models/lizard.bpk"
"#,
        )
        .unwrap();

        assert!(config.base_training.enabled);
        assert!(config.base_model.is_none());
        assert_eq!(config.base_training.phases.len(), 1);
        let phase = &config.base_training.phases[0];
        assert_eq!(phase.training.particle_count, 3072);
        assert_eq!(phase.training.epochs, 99);
        assert_eq!(phase.training.batch_size, 8);
        assert_eq!(phase.training.step_min, 32);
        assert_eq!(phase.training.step_max, 96);
        assert_eq!(phase.training.repetitions, 3);
        assert!(phase.loss.is_none());
        assert_eq!(
            config.multiscale_training.local_residual_training_scale,
            1.0
        );
        assert_eq!(
            config.multiscale_training.proxy_residual_training_scale,
            1.0
        );
    }

    #[test]
    fn verified_lizard_configs_encode_controller_only_exact_bootstrap() {
        for (name, target_leaves, initial_leaves, validation_seeds) in [
            ("task_multiscale_lizard_smoke_2d_wgpu.toml", 128, 32, 1),
            ("task_multiscale_lizard_full_2d_cuda.toml", 4_096, 1_024, 8),
        ] {
            let path = format!(
                "{}/../../configs/verified/2d/adaptive/training/{name}",
                env!("CARGO_MANIFEST_DIR")
            );
            let source = std::fs::read_to_string(path).unwrap();
            let config: AdaptiveExperimentConfig = toml::from_str(&source).unwrap();

            assert_eq!(config.adaptive.target_leaves, target_leaves);
            assert_eq!(config.adaptive.initial_leaves, initial_leaves);
            assert!(config.adaptive.hierarchical_bootstrap_seed);
            assert_eq!(config.adaptive.bootstrap_end_step, 1);
            assert_eq!(
                config.adaptive.runtime_topology_control,
                AdaptiveTopologyControl::LearnedRefinementDefect
            );
            assert_eq!(config.adaptive.local_residual_scale, 0.0);
            assert!(config.multiscale_training.freeze_multiscale_rule);
            assert!(config.multiscale_training.controller_on_policy_only_replay);
            assert_eq!(
                config.multiscale_training.on_policy_topology_control,
                AdaptiveTopologyControl::RefinementDefectOracle
            );
            assert_eq!(config.task_quality.validation_seeds.len(), validation_seeds);
            assert!(!config.task_quality.bandwidth_adaptation_enabled);
            assert_eq!(
                config.task_quality.topology_control,
                AdaptiveTopologyControl::LearnedRefinementDefect
            );
        }
    }

    #[test]
    fn progressive_lod_config_separates_bootstrap_and_steady_budgets() {
        for relative in [
            "configs/verified/2d/adaptive/evaluation/task_lod_lizard_smoke_3070_2d_wgpu.toml",
            "configs/verified/2d/adaptive/evaluation/task_lod_lizard_eval_3070_2d_wgpu.toml",
        ] {
            let path = format!("{}/../../{relative}", env!("CARGO_MANIFEST_DIR"));
            let source = std::fs::read_to_string(path).unwrap();
            let config: AdaptiveExperimentConfig = toml::from_str(&source).unwrap();

            assert_eq!(config.adaptive.initial_leaf_count(), 1_024);
            assert_eq!(config.adaptive.target_leaves, 3_070);
            assert_eq!(config.adaptive.bootstrap_target_leaf_count(), 4_096);
            assert_eq!(config.adaptive.bootstrap_fine_leaf_count(), 4_096);
            assert_eq!(config.adaptive.bootstrap_end_step, 4);
            assert_eq!(config.adaptive.topology_end_step, 1_024);
            assert_eq!(config.adaptive.bootstrap_events_per_interval, 256);
            assert_eq!(config.adaptive.hierarchical_restriction_step, 230);
            assert_eq!(
                config
                    .adaptive
                    .hierarchical_restriction_leaf_delta_per_interval,
                0
            );
            assert_eq!(
                config.adaptive.hierarchical_restriction_arity,
                AdaptiveRestrictionArity::Canonical
            );
            assert_eq!(
                config.adaptive.hierarchical_restriction_schedule,
                crate::adaptive::AdaptiveRestrictionSchedule::RollingRecompute
            );
            assert_eq!(config.adaptive.bootstrap_quadrature_point_count(), 4);
            assert_eq!(config.adaptive.coarse_quadrature_points, 4);
            assert_eq!(config.adaptive.steady_topology_interval(), 256);
            assert_eq!(config.task_quality.rollout_steps, 1_024);
            assert!(config.task_quality.gap_decomposition.enabled);
            assert!(
                config
                    .gates
                    .min_validation_mean_adaptive_over_teacher_psnr_gain_db
                    .is_finite()
            );
            assert!(
                config
                    .gates
                    .min_validation_worst_adaptive_over_teacher_psnr_gain_db
                    .is_finite()
            );
            assert!(
                config
                    .gates
                    .min_gap_final_selected_mode_vs_fine_control_db
                    .is_finite()
            );
            config.adaptive.validate().unwrap();
        }
    }

    #[test]
    fn resident_direct_active_config_encodes_canonical_mixed_scale_progression() {
        let path = format!(
            "{}/../../configs/verified/2d/adaptive/evaluation/task_resident_lizard_smoke_3070_2d_wgpu.toml",
            env!("CARGO_MANIFEST_DIR")
        );
        let source = std::fs::read_to_string(path).unwrap();
        let config: AdaptiveExperimentConfig = toml::from_str(&source).unwrap();

        assert_eq!(config.adaptive.initial_leaf_count(), 1_024);
        assert_eq!(config.adaptive.target_leaves, 3_070);
        assert_eq!(config.adaptive.bootstrap_fine_leaf_count(), 4_096);
        assert_eq!(config.adaptive.bootstrap_end_step, 3);
        assert_eq!(config.adaptive.bootstrap_events_per_interval, 256);
        assert!(!config.adaptive.retain_bootstrap_templates);
        assert!(config.adaptive.expected_coarse_update_mask);
        assert!(!config.adaptive.material_scale_conditioning);
        assert_eq!(
            config.adaptive.runtime_topology_control,
            AdaptiveTopologyControl::PairedLocalDetail
        );
        assert_eq!(
            config.adaptive.material_seed_layout,
            crate::adaptive::AdaptiveMaterialSeedLayout::CanonicalGrouped
        );
        assert_eq!(config.task_quality.rollout_steps, 256);
        assert_eq!(config.task_quality.validation_seeds.len(), 2);
    }
}
