use crate::cli::prelude::*;

use super::super::super::{Hyper2dE2eSplit, sources::Hyper2dScratchSource};
use super::super::{DirectBasisTargetConfig, EvalConfig};

#[derive(Clone)]
pub(super) struct AdapterBankConditionedExample {
    pub(super) source: Hyper2dScratchSource,
    pub(super) split: Hyper2dE2eSplit,
    pub(super) condition: ConditionImage2d,
    pub(super) target_vector: Vec<f32>,
    pub(super) target_has_bias_correction: bool,
    pub(super) target_source_width: usize,
    pub(super) target_source_height: usize,
    pub(super) target_points: usize,
    pub(super) last_train_loss: Option<f32>,
    pub(super) sample_weight: f32,
}

#[derive(Clone, Copy)]
pub(super) struct AdapterBankRolloutEvalConfig {
    pub(super) target: DirectBasisTargetConfig,
    pub(super) rollout: EvalConfig,
    pub(super) loss: Target2dLossConfig,
    pub(super) requested_examples_per_split: usize,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub(super) struct AdapterBankTrainConfig {
    pub(super) objective: AdapterBankTrainingObjective,
    pub(super) steps: usize,
    pub(super) report_interval: usize,
    pub(super) example_batch_size: usize,
    pub(super) diagnostic_vector_examples: usize,
    pub(super) loss_eval_batch_size: usize,
    pub(super) system_memory_budget_gb: Option<f32>,
    pub(super) seed: u64,
    pub(super) optimizer: AdamWConfig,
    pub(super) flow: AdapterBankFlowTrainConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AdapterBankTrainingObjective {
    StaticVectorMse,
    RectifiedFlow,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub(super) struct AdapterBankFlowTrainConfig {
    pub(super) hidden_dims: usize,
    pub(super) sample_steps: usize,
    pub(super) source_scale: f32,
    pub(super) sample_seed: u64,
    pub(super) hidden_activation: HyperNpa2dFlowActivation,
    pub(super) init: AdapterBankFlowInit,
    pub(super) loss: AdapterBankFlowLoss,
    pub(super) sample_weights: AdapterBankSampleWeights,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AdapterBankFlowInit {
    Random,
    LinearSolveConditionWarmstart,
    FromHyper,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AdapterBankFlowLoss {
    VelocityMse,
    SampledAdapterMse,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub(super) struct AdapterBankSampleWeights {
    pub(super) enabled: bool,
    pub(super) hard_weight: f32,
    pub(super) psnr_threshold_db: f32,
    pub(super) hard_examples: usize,
}

impl Default for AdapterBankSampleWeights {
    fn default() -> Self {
        Self {
            enabled: false,
            hard_weight: 1.0,
            psnr_threshold_db: 26.0,
            hard_examples: 0,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct AdapterBankExperimentConfig {
    pub(super) preset: Option<String>,
    pub(super) input: AdapterBankInputExperimentConfig,
    pub(super) selection: AdapterBankSelectionExperimentConfig,
    pub(super) output: AdapterBankOutputExperimentConfig,
    pub(super) condition: AdapterBankConditionExperimentConfig,
    pub(super) training: AdapterBankTrainingExperimentConfig,
    pub(super) eval: AdapterBankEvalExperimentConfig,
    pub(super) target: AdapterBankTargetExperimentConfig,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct AdapterBankInputExperimentConfig {
    pub(super) shared_base: Option<PathBuf>,
    pub(super) adapter_bank: Option<PathBuf>,
    pub(super) initial_hyper: Option<PathBuf>,
    pub(super) psnr_gate_report: Option<PathBuf>,
    pub(super) source_limit: Option<usize>,
    pub(super) train_limit: Option<usize>,
    pub(super) holdout_limit: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct AdapterBankSelectionExperimentConfig {
    pub(super) selection_seed: Option<u64>,
    pub(super) selection_manifest: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct AdapterBankOutputExperimentConfig {
    pub(super) output_dir: Option<PathBuf>,
    pub(super) report_output: Option<PathBuf>,
    pub(super) hyper_output: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct AdapterBankConditionExperimentConfig {
    pub(super) encoder: Option<String>,
    pub(super) dino_model: Option<PathBuf>,
    pub(super) dino_image_size: Option<usize>,
    pub(super) dino_batch_size: Option<usize>,
    pub(super) dino_cache_write_interval_batches: Option<usize>,
    pub(super) feature_cache: Option<PathBuf>,
    pub(super) token_grid_width: Option<usize>,
    pub(super) token_grid_height: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct AdapterBankTrainingExperimentConfig {
    pub(super) backend: Option<String>,
    pub(super) objective: Option<String>,
    pub(super) hidden: Option<usize>,
    pub(super) output_scale: Option<f32>,
    pub(super) linear_output: Option<bool>,
    pub(super) canonicalize_adapters: Option<bool>,
    pub(super) flow_hidden: Option<usize>,
    pub(super) flow_sample_steps: Option<usize>,
    pub(super) flow_source_scale: Option<f32>,
    pub(super) flow_sample_seed: Option<u64>,
    pub(super) flow_hidden_activation: Option<String>,
    pub(super) flow_init: Option<String>,
    pub(super) flow_loss: Option<String>,
    pub(super) flow_hard_sample_weight: Option<f32>,
    pub(super) flow_hard_sample_psnr_threshold_db: Option<f32>,
    pub(super) diagnostic_vector_examples: Option<usize>,
    pub(super) loss_eval_batch_size: Option<usize>,
    pub(super) system_memory_budget_gb: Option<f32>,
    pub(super) seed: Option<u64>,
    pub(super) steps: Option<usize>,
    pub(super) report_interval: Option<usize>,
    pub(super) example_batch_size: Option<usize>,
    pub(super) learning_rate: Option<f32>,
    pub(super) weight_decay: Option<f32>,
    pub(super) grad_clip_norm: Option<f32>,
    pub(super) adam_beta1: Option<f32>,
    pub(super) adam_beta2: Option<f32>,
    pub(super) adam_epsilon: Option<f32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct AdapterBankEvalExperimentConfig {
    pub(super) vector_examples: Option<usize>,
    pub(super) rollout_examples: Option<usize>,
    pub(super) particles: Option<usize>,
    pub(super) steps: Option<usize>,
    pub(super) update_prob: Option<f32>,
    pub(super) seed: Option<u64>,
    pub(super) seed_scale: Option<f32>,
    pub(super) seed_mode: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct AdapterBankTargetExperimentConfig {
    pub(super) points: Option<usize>,
    pub(super) image_size: Option<usize>,
    pub(super) threshold: Option<f32>,
    pub(super) loss_image_size: Option<usize>,
    pub(super) splat_sigma: Option<f32>,
    pub(super) splat_loss_weight: Option<f32>,
    pub(super) color_loss_weight: Option<f32>,
    pub(super) density_loss_weight: Option<f32>,
    pub(super) displacement_regularizer_weight: Option<f32>,
    pub(super) overflow_regularizer_weight: Option<f32>,
    pub(super) bound_regularizer_weight: Option<f32>,
}

#[derive(Serialize)]
pub(super) struct AdapterBankConditionedTrainingReport {
    pub(super) experiment_config: Option<String>,
    pub(super) preset: AutomataPreset,
    pub(super) shared_base: String,
    pub(super) adapter_bank: String,
    pub(super) adapter_bank_base_model: String,
    pub(super) output_dir: String,
    pub(super) report_output: String,
    pub(super) hyper_output: String,
    pub(super) backend: Hyper2dAdapterBankBackendArg,
    pub(super) npa_config: NpaConfig,
    pub(super) hashgrid: burn_automata_kernels::HashGridConfig,
    pub(super) hyper_config: HyperNpa2dConfig,
    pub(super) generator_architecture: &'static str,
    pub(super) generator_objective: &'static str,
    pub(super) adapter_rank: usize,
    pub(super) adapter_alpha: f32,
    pub(super) adapter_parameter_count: usize,
    pub(super) condition_encoder: String,
    pub(super) train_examples: usize,
    pub(super) holdout_examples: usize,
    pub(super) source_limit: usize,
    pub(super) train_limit: usize,
    pub(super) holdout_limit: usize,
    pub(super) selection: AdapterBankSelectionReport,
    pub(super) target_stats: AdapterBankTargetVectorStats,
    pub(super) requested_training: AdapterBankTrainingSettingsReport,
    pub(super) adapter_target_canonicalization: &'static str,
    pub(super) memory: Vec<AdapterBankMemorySnapshot>,
    pub(super) training: AdapterBankTrainingPhaseReport,
    pub(super) train_vector_metrics: AdapterBankVectorMetricsReport,
    pub(super) holdout_vector_metrics: Option<AdapterBankVectorMetricsReport>,
    pub(super) rollout_particles: usize,
    pub(super) rollout_steps: usize,
    pub(super) target_points: usize,
    pub(super) target_loss_config: Target2dLossConfig,
    pub(super) rollout_eval: AdapterBankRolloutEvalReport,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct AdapterBankSelectionReport {
    pub(super) selection_seed: Option<u64>,
    pub(super) selection_manifest: Option<String>,
    pub(super) replayed_manifest: bool,
    pub(super) train_selected: usize,
    pub(super) holdout_selected: usize,
}

#[derive(Serialize)]
pub(super) struct AdapterBankTrainingSettingsReport {
    pub(super) objective: &'static str,
    pub(super) steps: usize,
    pub(super) report_interval: usize,
    pub(super) example_batch_size: usize,
    pub(super) diagnostic_vector_examples: usize,
    pub(super) loss_eval_batch_size: usize,
    pub(super) system_memory_budget_gb: Option<f32>,
    pub(super) seed: u64,
    pub(super) optimizer: AdamWConfig,
    pub(super) flow: Option<AdapterBankFlowTrainingSettingsReport>,
}

#[derive(Serialize)]
pub(super) struct AdapterBankFlowTrainingSettingsReport {
    pub(super) hidden_dims: usize,
    pub(super) sample_steps: usize,
    pub(super) source_scale: f32,
    pub(super) sample_seed: u64,
    pub(super) hidden_activation: HyperNpa2dFlowActivation,
    pub(super) init: &'static str,
    pub(super) loss: &'static str,
    pub(super) sample_weights: AdapterBankSampleWeights,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) initial_hyper: Option<String>,
}

#[derive(Clone, Serialize)]
pub(super) struct AdapterBankTrainingPhaseReport {
    pub(super) backend: String,
    pub(super) device: String,
    pub(super) selection_metric: String,
    pub(super) initial_loss: f32,
    pub(super) initial_validation_loss: Option<f32>,
    pub(super) final_loss: f32,
    pub(super) final_validation_loss: Option<f32>,
    pub(super) best_loss: f32,
    pub(super) best_validation_loss: Option<f32>,
    pub(super) best_step: usize,
    pub(super) history: Vec<AdapterBankTrainingHistoryEntry>,
    pub(super) memory: Vec<AdapterBankMemorySnapshot>,
    pub(super) elapsed_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) vector_selection: Option<AdapterBankTrainingVectorSelectionReport>,
}

#[derive(Clone, Serialize)]
pub(super) struct AdapterBankTrainingVectorSelectionReport {
    pub(super) requested_examples: usize,
    pub(super) initial_train: AdapterBankVectorMetricsReport,
    pub(super) initial_validation: Option<AdapterBankVectorMetricsReport>,
    pub(super) final_train: AdapterBankVectorMetricsReport,
    pub(super) final_validation: Option<AdapterBankVectorMetricsReport>,
    pub(super) best_train: AdapterBankVectorMetricsReport,
    pub(super) best_validation: Option<AdapterBankVectorMetricsReport>,
}

#[derive(Clone, Serialize)]
pub(super) struct AdapterBankTrainingHistoryEntry {
    pub(super) step: usize,
    pub(super) loss: f32,
    pub(super) grad_norm: f32,
    pub(super) grad_scale: f32,
    pub(super) examples_seen: usize,
    pub(super) adapter_values_per_sec: f64,
    pub(super) validation_loss: Option<f32>,
    pub(super) memory: AdapterBankMemorySnapshot,
    pub(super) elapsed_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) train_vector_metrics: Option<AdapterBankVectorMetricsReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) validation_vector_metrics: Option<AdapterBankVectorMetricsReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) flow_optimizer: Option<AdapterBankFlowOptimizerDiagnosticsReport>,
}

#[derive(Clone, Copy, Serialize)]
pub(super) struct AdapterBankFlowOptimizerDiagnosticsReport {
    pub(super) prediction_rms: f32,
    pub(super) velocity_rms: f32,
    pub(super) residual_rms: f32,
    pub(super) pre_hidden_rms: f32,
    pub(super) hidden_rms: f32,
    pub(super) hidden_zero_fraction: f32,
    pub(super) grad_w1_norm: f32,
    pub(super) grad_b1_norm: f32,
    pub(super) grad_w2_norm: f32,
    pub(super) grad_b2_norm: f32,
}

#[derive(Clone, Serialize)]
pub(super) struct AdapterBankMemorySnapshot {
    pub(super) label: String,
    pub(super) rss_bytes: Option<u64>,
    pub(super) peak_rss_bytes: Option<u64>,
    pub(super) swap_bytes: Option<u64>,
}

#[derive(Clone, Copy, Serialize)]
pub(super) struct AdapterBankTargetVectorStats {
    pub(super) examples: usize,
    pub(super) parameters_per_adapter: usize,
    pub(super) mean_rms: f32,
    pub(super) mean_abs: f32,
    pub(super) max_abs: f32,
    pub(super) output_scale: f32,
    pub(super) target_values_outside_output_scale_fraction: f32,
}

#[derive(Clone, Copy, Serialize)]
pub(super) struct AdapterBankVectorMetricsReport {
    pub(super) examples: usize,
    pub(super) parameters_per_adapter: usize,
    pub(super) mse: f32,
    pub(super) rmse: f32,
    pub(super) normalized_rmse_to_target_rms: Option<f32>,
    pub(super) mean_abs_error: f32,
    pub(super) max_abs_error: f32,
    pub(super) target_rms: f32,
    pub(super) prediction_rms: f32,
    pub(super) target_max_abs: f32,
    pub(super) prediction_max_abs: f32,
    pub(super) mean_cosine_similarity: f32,
    pub(super) prediction_values_near_output_scale_fraction: f32,
    pub(super) target_values_outside_output_scale_fraction: f32,
}

#[derive(Serialize)]
pub(super) struct AdapterBankRolloutEvalReport {
    pub(super) requested_examples_per_split: usize,
    pub(super) train_summary: Option<AdapterBankRolloutSummary>,
    pub(super) holdout_summary: Option<AdapterBankRolloutSummary>,
    pub(super) entries: Vec<AdapterBankRolloutEntry>,
}

#[derive(Clone, Copy, Serialize)]
pub(super) struct AdapterBankRolloutSummary {
    pub(super) examples: usize,
    pub(super) mean_zero_loss: f32,
    pub(super) mean_static_loss: f32,
    pub(super) mean_hyper_loss: f32,
    pub(super) mean_gap_to_static: f32,
    pub(super) mean_ratio_to_static: f32,
    pub(super) max_ratio_to_static: f32,
    pub(super) mean_gap_to_zero: f32,
    pub(super) mean_ratio_to_zero: f32,
    pub(super) max_ratio_to_zero: f32,
}

#[derive(Serialize)]
pub(super) struct AdapterBankRolloutEntry {
    pub(super) slug: String,
    pub(super) split: &'static str,
    pub(super) condition: String,
    pub(super) target_source_width: usize,
    pub(super) target_source_height: usize,
    pub(super) target_points: usize,
    pub(super) zero_adapter_loss: Target2dLossReport,
    pub(super) static_adapter_loss: Target2dLossReport,
    pub(super) hyper_adapter_loss: Target2dLossReport,
    pub(super) hyper_gap_to_static: f32,
    pub(super) hyper_ratio_to_static: f32,
    pub(super) hyper_gap_to_zero: f32,
    pub(super) hyper_ratio_to_zero: f32,
    pub(super) adapter_vector_mse: f32,
    pub(super) adapter_vector_cosine_similarity: f32,
}
