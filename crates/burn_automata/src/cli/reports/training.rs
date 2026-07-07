use crate::cli::prelude::*;

use super::mesh_rollout::MeshRolloutReport;

#[derive(Serialize)]
pub(crate) struct CliTrainingReport {
    pub(crate) preset: AutomataPreset,
    pub(crate) objective: &'static str,
    pub(crate) target_source: String,
    pub(crate) student_seed: u64,
    pub(crate) sgd: SgdConfig,
    pub(crate) optimizer: SupervisedOptimizerConfig,
    pub(crate) training_device: TrainingDeviceArg,
    pub(crate) rounds: usize,
    pub(crate) total_rows_seen: usize,
    pub(crate) report: TrainingRunReport,
    pub(crate) model_output: Option<String>,
    pub(crate) batch_source: TrainingBatchArg,
    pub(crate) rollout_supervision: Option<CliRolloutSupervisionReport>,
    pub(crate) mesh_rollout: Option<MeshRolloutReport>,
    pub(crate) render_loss: Option<MultiViewRenderLossReport>,
}

#[derive(Serialize)]
pub(crate) struct CliTarget2dEvalReport {
    pub(crate) preset: AutomataPreset,
    pub(crate) model: String,
    pub(crate) target_image: String,
    pub(crate) reference_model: Option<String>,
    pub(crate) particle_count: usize,
    pub(crate) rollout_steps: usize,
    pub(crate) update_prob: f32,
    pub(crate) seed: u64,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) loss_config: Target2dLossConfig,
    pub(crate) target_source_width: usize,
    pub(crate) target_source_height: usize,
    pub(crate) target_points: usize,
    pub(crate) loss: Target2dLossReport,
    pub(crate) reference_loss: Option<Target2dLossReport>,
    pub(crate) total_loss_gap_to_reference: Option<f32>,
    pub(crate) total_loss_ratio_to_reference: Option<f32>,
    pub(crate) render_diagnostics: Option<CliTarget2dRenderDiagnosticsReport>,
}

#[derive(Serialize)]
pub(crate) struct CliTarget2dRenderDiagnosticsReport {
    pub(crate) output_dir: String,
    pub(crate) image_size: usize,
    pub(crate) density_lit_threshold: f32,
    pub(crate) target: CliTarget2dRenderImageReport,
    pub(crate) model_cpu: CliTarget2dRenderImageReport,
    pub(crate) reference_cpu: Option<CliTarget2dRenderImageReport>,
    pub(crate) model_wgpu: Option<CliTarget2dRenderImageReport>,
    pub(crate) reference_wgpu: Option<CliTarget2dRenderImageReport>,
}

#[derive(Serialize)]
pub(crate) struct CliTarget2dRenderImageReport {
    pub(crate) label: &'static str,
    pub(crate) rgb_png: String,
    pub(crate) density_png: String,
    pub(crate) rgb_mse_to_target: Option<f32>,
    pub(crate) rgb_psnr_db_to_target: Option<f32>,
    pub(crate) density_mse_to_target: Option<f32>,
    pub(crate) density_psnr_db_to_target: Option<f32>,
    pub(crate) density_total: f32,
    pub(crate) density_max: f32,
    pub(crate) lit_pixels: usize,
    pub(crate) lit_bbox_xyxy: Option<[usize; 4]>,
    pub(crate) geometry_to_target: Option<CliTarget2dRenderGeometryReport>,
    pub(crate) particle_stats: Option<CliTarget2dParticleStatsReport>,
}

#[derive(Serialize)]
pub(crate) struct CliTarget2dRenderGeometryReport {
    pub(crate) lit_pixel_ratio: f32,
    pub(crate) foreground_iou: f32,
    pub(crate) target_recall: f32,
    pub(crate) generated_precision: f32,
    pub(crate) bbox_iou: Option<f32>,
    pub(crate) bbox_width_ratio: Option<f32>,
    pub(crate) bbox_height_ratio: Option<f32>,
    pub(crate) bbox_area_ratio: Option<f32>,
    pub(crate) particle_rms_radius_ratio: Option<f32>,
}

#[derive(Clone, Serialize)]
pub(crate) struct CliTarget2dParticleStatsReport {
    pub(crate) count: usize,
    pub(crate) mean_xy: [f32; 2],
    pub(crate) rms_radius: f32,
    pub(crate) bounds_min_xy: [f32; 2],
    pub(crate) bounds_max_xy: [f32; 2],
    pub(crate) out_of_domain_fraction: f32,
}

#[derive(Serialize)]
pub(crate) struct CliTarget2dTrainingReport {
    pub(crate) preset: AutomataPreset,
    pub(crate) requested_training_device: TrainingDeviceArg,
    pub(crate) training_device: TrainingDeviceArg,
    pub(crate) gpu_backend: Option<DirectBasisOracleBackendArg>,
    pub(crate) gpu_training: Option<CliHyper2dDirectBasisGpuTrainingReport>,
    pub(crate) target_image: String,
    pub(crate) target_source_width: usize,
    pub(crate) target_source_height: usize,
    pub(crate) target_points: usize,
    pub(crate) model_output: Option<String>,
    pub(crate) model_eval_loss: Option<Target2dLossReport>,
    pub(crate) reference_model: Option<String>,
    pub(crate) reference_loss: Option<Target2dLossReport>,
    pub(crate) final_loss_gap_to_reference: Option<f32>,
    pub(crate) final_loss_ratio_to_reference: Option<f32>,
    pub(crate) hashgrid: burn_automata_kernels::HashGridConfig,
    pub(crate) training: Option<Target2dTrainingReport>,
}

#[derive(Serialize)]
pub(crate) struct CliTrainingBenchReport {
    pub(crate) preset: AutomataPreset,
    pub(crate) requested_training_device: TrainingDeviceArg,
    pub(crate) training_device: TrainingDeviceArg,
    pub(crate) batch_source: TrainingBatchArg,
    pub(crate) optimizer: SupervisedOptimizerConfig,
    pub(crate) rows: usize,
    pub(crate) steps: usize,
    pub(crate) repeats: usize,
    pub(crate) warmup_steps: usize,
    pub(crate) report_interval: usize,
    pub(crate) target_model: Option<String>,
    pub(crate) runs: Vec<CliTrainingBenchRunReport>,
    pub(crate) min_row_steps_per_sec: f64,
    pub(crate) median_row_steps_per_sec: f64,
    pub(crate) max_row_steps_per_sec: f64,
}

#[derive(Serialize)]
pub(crate) struct CliTrainingBenchRunReport {
    pub(crate) repeat: usize,
    pub(crate) elapsed_ms: f64,
    pub(crate) row_steps_per_sec: f64,
    pub(crate) initial_loss: f32,
    pub(crate) final_loss: f32,
    pub(crate) best_loss: f32,
    pub(crate) history_points: usize,
}

#[derive(Serialize)]
pub(crate) struct CliRenderLossEvalReport {
    pub(crate) target: MeshTargetArg,
    pub(crate) model: String,
    pub(crate) particle_count: usize,
    pub(crate) steps: usize,
    pub(crate) seed: u64,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) render_loss: MultiViewRenderLossReport,
}

#[derive(Serialize)]
pub(crate) struct CliDynamics2dEvalReport {
    pub(crate) preset: AutomataPreset,
    pub(crate) model: String,
    pub(crate) target_model: String,
    pub(crate) particle_count: usize,
    pub(crate) rollout_steps: usize,
    pub(crate) update_prob: f32,
    pub(crate) seed: u64,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) image_size: usize,
    pub(crate) render_sigma_px: f32,
    pub(crate) metrics: CliHyper2dDynamicsMetricsReport,
}

#[derive(Serialize)]
pub(crate) struct CliTorusTrainingReport {
    pub(crate) preset: AutomataPreset,
    pub(crate) target_source: String,
    pub(crate) student_seed: u64,
    pub(crate) sgd: SgdConfig,
    pub(crate) report: TrainingRunReport,
    pub(crate) model_output: Option<String>,
    pub(crate) robustness: TorusRobustnessReport,
    pub(crate) batch_source: TrainingBatchArg,
    pub(crate) training_mode: MeshTrainingModeArg,
    pub(crate) rollout_supervision: Option<CliRolloutSupervisionReport>,
}

#[derive(Serialize)]
pub(crate) struct CliHyper2dTrainingReport {
    pub(crate) preset: AutomataPreset,
    pub(crate) condition: Option<String>,
    pub(crate) catalog: Option<String>,
    pub(crate) catalog_group: Option<Hyper2dCatalogGroupArg>,
    pub(crate) catalog_targets: Vec<String>,
    pub(crate) base_model: Option<String>,
    pub(crate) target_model: Option<String>,
    pub(crate) hyper_input: Option<String>,
    pub(crate) hyper_output: String,
    pub(crate) adapter_output: Option<String>,
    pub(crate) materialized_output: Option<String>,
    pub(crate) generated_output_dir: Option<String>,
    pub(crate) npa_config: NpaConfig,
    pub(crate) hyper_config: HyperNpa2dConfig,
    pub(crate) sgd: SgdConfig,
    pub(crate) rollout_supervision: CliRolloutSupervisionReport,
    pub(crate) initial_loss: f32,
    pub(crate) holdout_initial_loss: Option<f32>,
    pub(crate) final_loss: f32,
    pub(crate) holdout_final_loss: Option<f32>,
    pub(crate) best_loss: f32,
    pub(crate) best_step: usize,
    pub(crate) history: Vec<CliHyper2dHistoryEntry>,
    pub(crate) adapter_bootstrap: Vec<CliHyper2dAdapterBootstrapReport>,
    pub(crate) train_examples: Vec<CliHyper2dExampleReport>,
    pub(crate) holdout_examples: Vec<CliHyper2dExampleReport>,
    pub(crate) adapter_parameter_count: usize,
    pub(crate) materialized_parameter_count: usize,
}

#[derive(Clone, Serialize)]
pub(crate) struct CliOmniSvgSourceReport {
    pub(crate) dataset: OmniSvgDatasetArg,
    pub(crate) dataset_id: String,
    pub(crate) split: String,
    pub(crate) cache_dir: String,
    pub(crate) offset: usize,
    pub(crate) limit: usize,
    pub(crate) page_size: usize,
    pub(crate) download: bool,
    pub(crate) refresh: bool,
    pub(crate) token_env: String,
}

#[derive(Serialize)]
pub(crate) struct CliHyper2dE2eTrainingReport {
    pub(crate) preset: AutomataPreset,
    pub(crate) target_images: Vec<String>,
    pub(crate) catalog: Option<String>,
    pub(crate) catalog_group: Option<Hyper2dCatalogGroupArg>,
    pub(crate) catalog_targets: Vec<String>,
    pub(crate) omnisvg: Option<CliOmniSvgSourceReport>,
    pub(crate) holdout_targets: Vec<String>,
    pub(crate) holdout_stride: usize,
    pub(crate) holdout_offset: usize,
    pub(crate) fit_holdout_static_oracles: bool,
    pub(crate) output_dir: String,
    pub(crate) report_output: String,
    pub(crate) scratch_catalog_output: String,
    pub(crate) shared_base_output: String,
    pub(crate) hyper_output: String,
    pub(crate) generated_output_dir: String,
    pub(crate) condition_encoder: &'static str,
    pub(crate) shared_base_strategy: &'static str,
    pub(crate) static_adapter_strategy: &'static str,
    pub(crate) npa_config: NpaConfig,
    pub(crate) hashgrid: burn_automata_kernels::HashGridConfig,
    pub(crate) target_loss_config: Target2dLossConfig,
    pub(crate) target_training_config: Target2dTrainingConfig,
    pub(crate) hyper_config: HyperNpa2dConfig,
    pub(crate) hyper_sgd: SgdConfig,
    pub(crate) flow_sgd: Option<SgdConfig>,
    pub(crate) adapter_parameter_count: usize,
    pub(crate) materialized_parameter_count: usize,
    pub(crate) exact_adapter_required_rank: usize,
    pub(crate) target_training: Vec<CliHyper2dE2eTargetReport>,
    pub(crate) shared_basis_fit: CliHyper2dE2eSharedBasisFitReport,
    pub(crate) static_adapters: Vec<CliHyper2dE2eAdapterReport>,
    pub(crate) initial_adapter_loss: f32,
    pub(crate) final_adapter_loss: f32,
    pub(crate) best_adapter_loss: f32,
    pub(crate) best_adapter_step: usize,
    pub(crate) initial_flow_loss: Option<f32>,
    pub(crate) final_flow_loss: Option<f32>,
    pub(crate) best_flow_loss: Option<f32>,
    pub(crate) best_flow_step: Option<usize>,
    pub(crate) direct_finetune: Option<CliHyper2dE2eDirectFinetuneReport>,
    pub(crate) adapter_history: Vec<CliHyper2dE2eHyperHistoryEntry>,
    pub(crate) flow_history: Vec<CliHyper2dE2eHyperHistoryEntry>,
    pub(crate) quality: CliHyper2dE2eQualityReport,
    pub(crate) train_quality: CliHyper2dE2eQualityReport,
    pub(crate) holdout_quality: Option<CliHyper2dE2eQualityReport>,
    pub(crate) eval: Vec<CliHyper2dE2eEvalReport>,
}

#[derive(Serialize)]
pub(crate) struct CliHyper2dE2eSharedBasisFitReport {
    pub(crate) enabled: bool,
    pub(crate) steps: usize,
    pub(crate) report_interval: usize,
    pub(crate) rows: usize,
    pub(crate) example_batch_size: usize,
    pub(crate) adapter_l2_weight: f32,
    pub(crate) seed: u64,
    pub(crate) base_sgd: SgdConfig,
    pub(crate) adapter_sgd: SgdConfig,
    pub(crate) initial_loss: f32,
    pub(crate) final_loss: f32,
    pub(crate) best_loss: f32,
    pub(crate) best_step: usize,
    pub(crate) history: Vec<CliHyper2dE2eSharedBasisHistoryEntry>,
}

#[derive(Serialize)]
pub(crate) struct CliHyper2dE2eSharedBasisHistoryEntry {
    pub(crate) step: usize,
    pub(crate) loss: f32,
    pub(crate) base_grad_norm: f32,
    pub(crate) base_grad_scale: f32,
    pub(crate) mean_adapter_grad_norm: f32,
    pub(crate) max_adapter_grad_norm: f32,
    pub(crate) examples_seen: usize,
}

#[derive(Serialize)]
pub(crate) struct CliHyper2dE2eTargetReport {
    pub(crate) slug: String,
    pub(crate) split: &'static str,
    pub(crate) title: Option<String>,
    pub(crate) group: Option<String>,
    pub(crate) condition: String,
    pub(crate) target_model: String,
    pub(crate) target_source_width: usize,
    pub(crate) target_source_height: usize,
    pub(crate) target_points: usize,
    pub(crate) training: Target2dTrainingReport,
}

#[derive(Serialize)]
pub(crate) struct CliHyper2dE2eAdapterReport {
    pub(crate) slug: String,
    pub(crate) split: &'static str,
    pub(crate) adapter_output: String,
    pub(crate) materialized_output: String,
    pub(crate) method: &'static str,
    pub(crate) steps: usize,
    pub(crate) rows: usize,
    pub(crate) initial_loss: f32,
    pub(crate) final_loss: f32,
    pub(crate) best_loss: f32,
    pub(crate) adapter_parameter_count: usize,
}

#[derive(Serialize)]
pub(crate) struct CliHyper2dE2eHyperHistoryEntry {
    pub(crate) step: usize,
    pub(crate) loss: f32,
    pub(crate) grad_norm: f32,
    pub(crate) grad_scale: f32,
}

#[derive(Serialize)]
pub(crate) struct CliHyper2dE2eDirectFinetuneReport {
    pub(crate) objective: &'static str,
    pub(crate) updates: &'static str,
    pub(crate) steps: usize,
    pub(crate) report_interval: usize,
    pub(crate) examples: usize,
    pub(crate) particle_count: usize,
    pub(crate) rollout_steps: usize,
    pub(crate) seed: u64,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) adapter_l2_weight: f32,
    pub(crate) hyper_sgd: SgdConfig,
    pub(crate) initial_loss: f32,
    pub(crate) final_loss: f32,
    pub(crate) best_loss: f32,
    pub(crate) best_step: usize,
    pub(crate) history: Vec<CliHyper2dE2eDirectFinetuneHistoryEntry>,
}

#[derive(Serialize)]
pub(crate) struct CliHyper2dE2eDirectFinetuneHistoryEntry {
    pub(crate) step: usize,
    pub(crate) loss: f32,
    pub(crate) image_loss: f32,
    pub(crate) adapter_l2_loss: f32,
    pub(crate) grad_norm: f32,
    pub(crate) grad_scale: f32,
}

#[derive(Serialize)]
pub(crate) struct CliHyper2dE2eEvalReport {
    pub(crate) slug: String,
    pub(crate) split: &'static str,
    pub(crate) condition: String,
    pub(crate) particle_count: usize,
    pub(crate) rollout_steps: usize,
    pub(crate) update_prob: f32,
    pub(crate) seed: u64,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) trained_target_loss: Target2dLossReport,
    pub(crate) static_adapter_loss: Target2dLossReport,
    pub(crate) hyper_loss: Target2dLossReport,
    pub(crate) static_adapter_gap_to_trained_target: f32,
    pub(crate) hyper_gap_to_trained_target: f32,
    pub(crate) hyper_gap_to_static_adapter: f32,
    pub(crate) static_adapter_ratio_to_trained_target: Option<f32>,
    pub(crate) hyper_ratio_to_trained_target: Option<f32>,
    pub(crate) hyper_ratio_to_static_adapter: Option<f32>,
}

#[derive(Serialize)]
pub(crate) struct CliHyper2dE2eQualityReport {
    pub(crate) examples: usize,
    pub(crate) mean_static_adapter_ratio_to_trained_target: Option<f32>,
    pub(crate) max_static_adapter_ratio_to_trained_target: Option<f32>,
    pub(crate) mean_hyper_ratio_to_static_adapter: Option<f32>,
    pub(crate) max_hyper_ratio_to_static_adapter: Option<f32>,
    pub(crate) mean_hyper_ratio_to_trained_target: Option<f32>,
    pub(crate) max_hyper_ratio_to_trained_target: Option<f32>,
    pub(crate) max_static_adapter_gap_to_trained_target: Option<f32>,
    pub(crate) max_hyper_gap_to_static_adapter: Option<f32>,
    pub(crate) max_hyper_gap_to_trained_target: Option<f32>,
    pub(crate) max_static_ratio_threshold: Option<f32>,
    pub(crate) max_hyper_static_ratio_threshold: Option<f32>,
    pub(crate) max_hyper_target_ratio_threshold: Option<f32>,
    pub(crate) passed: bool,
}

#[derive(Serialize)]
pub(crate) struct CliHyper2dDirectBasisTrainingReport {
    pub(crate) experiment_config: Option<String>,
    pub(crate) preset: AutomataPreset,
    pub(crate) target_images: Vec<String>,
    pub(crate) target_image_dirs: Vec<String>,
    pub(crate) target_image_recursive: bool,
    pub(crate) image_extensions: Vec<String>,
    pub(crate) catalog: Option<String>,
    pub(crate) catalog_group: Option<Hyper2dCatalogGroupArg>,
    pub(crate) catalog_targets: Vec<String>,
    pub(crate) omnisvg: Option<CliOmniSvgSourceReport>,
    pub(crate) source_limit: usize,
    pub(crate) holdout_targets: Vec<String>,
    pub(crate) holdout_stride: usize,
    pub(crate) holdout_offset: usize,
    pub(crate) output_dir: String,
    pub(crate) report_output: String,
    pub(crate) shared_base_output: String,
    pub(crate) adapter_bank_output: String,
    pub(crate) adapter_output_dir: String,
    pub(crate) requested_training_device: TrainingDeviceArg,
    pub(crate) training_device: TrainingDeviceArg,
    pub(crate) gpu_training: Option<CliHyper2dDirectBasisGpuTrainingReport>,
    pub(crate) npa_config: NpaConfig,
    pub(crate) hashgrid: burn_automata_kernels::HashGridConfig,
    pub(crate) target_loss_config: Target2dLossConfig,
    pub(crate) adapter_rank: usize,
    pub(crate) adapter_alpha: f32,
    pub(crate) train_examples: usize,
    pub(crate) holdout_examples: usize,
    pub(crate) steps: usize,
    pub(crate) report_interval: usize,
    pub(crate) example_batch_size: usize,
    pub(crate) tbptt_chunk_steps: usize,
    pub(crate) rollout_particles: usize,
    pub(crate) rollout_steps: usize,
    pub(crate) update_prob: f32,
    pub(crate) seed: u64,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) per_parameter_grad_normalization: bool,
    pub(crate) base_sgd: SgdConfig,
    pub(crate) adapter_sgd: SgdConfig,
    pub(crate) train_refine_adapter_sgd: SgdConfig,
    pub(crate) holdout_adapter_sgd: SgdConfig,
    pub(crate) adapter_l2_weight: f32,
    pub(crate) train_adapter_refine_steps: usize,
    pub(crate) train_adapter_refine_batch_size: usize,
    pub(crate) holdout_adapter_steps: usize,
    pub(crate) holdout_adapter_batch_size: usize,
    pub(crate) eval_examples: usize,
    pub(crate) eval_interval: usize,
    pub(crate) eval_batch_size: usize,
    pub(crate) system_memory_budget_gb: Option<f32>,
    pub(crate) gpu_memory_budget_gb: Option<f32>,
    pub(crate) max_dense_train_particles: usize,
    pub(crate) max_dense_chunk_floats: usize,
    pub(crate) max_splat_chunk_floats: usize,
    pub(crate) initial_train_loss: Option<CliHyper2dDirectBasisLossSummary>,
    pub(crate) final_train_loss: Option<CliHyper2dDirectBasisLossSummary>,
    pub(crate) initial_holdout_loss: Option<CliHyper2dDirectBasisLossSummary>,
    pub(crate) final_holdout_loss: Option<CliHyper2dDirectBasisLossSummary>,
    pub(crate) best_train_loss: Option<f32>,
    pub(crate) best_train_step: usize,
    pub(crate) history: Vec<CliHyper2dDirectBasisHistoryEntry>,
    pub(crate) train_refine_history: Vec<CliHyper2dDirectBasisHistoryEntry>,
    pub(crate) holdout_history: Vec<CliHyper2dDirectBasisHistoryEntry>,
    pub(crate) oracle_validation: Option<CliHyper2dDirectBasisOracleReport>,
    pub(crate) adapters: Vec<CliHyper2dDirectBasisAdapterReport>,
}

#[derive(Serialize)]
pub(crate) struct CliHyper2dDirectBasisGpuTrainingReport {
    pub(crate) backend: String,
    pub(crate) device: String,
    pub(crate) metrics: serde_json::Value,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub(crate) struct CliHyper2dDirectBasisLossSummary {
    pub(crate) examples: usize,
    pub(crate) mean_total_loss: f32,
    pub(crate) max_total_loss: f32,
    pub(crate) mean_splat_loss: f32,
    pub(crate) mean_color_loss: f32,
    pub(crate) mean_density_loss: f32,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct CliHyper2dDirectBasisHistoryEntry {
    pub(crate) step: usize,
    pub(crate) loss: f32,
    pub(crate) eval_loss: Option<CliHyper2dDirectBasisLossSummary>,
    pub(crate) base_grad_norm: f32,
    pub(crate) base_grad_scale: f32,
    pub(crate) mean_adapter_grad_norm: f32,
    pub(crate) max_adapter_grad_norm: f32,
    pub(crate) examples_seen: usize,
    pub(crate) particle_steps_per_sec: f64,
    pub(crate) elapsed_ms: f64,
}

#[derive(Clone, Serialize)]
pub(crate) struct CliHyper2dDirectBasisOracleReport {
    pub(crate) backend: DirectBasisOracleBackendArg,
    pub(crate) gpu_device: Option<String>,
    pub(crate) resume_existing: bool,
    pub(crate) gpu_parallel_jobs: usize,
    pub(crate) train_examples_requested: usize,
    pub(crate) holdout_examples_requested: usize,
    pub(crate) train_examples: usize,
    pub(crate) holdout_examples: usize,
    pub(crate) epochs: usize,
    pub(crate) repetitions: usize,
    pub(crate) batch_size: usize,
    pub(crate) pool_size: usize,
    pub(crate) learning_rate: f32,
    pub(crate) weight_decay: f32,
    pub(crate) grad_clip_norm: f32,
    pub(crate) seed: u64,
    pub(crate) effective_particle_steps_per_sec: f64,
    pub(crate) mean_reported_particle_steps_per_sec: f64,
    pub(crate) train_summary: Option<CliHyper2dDirectBasisOracleSummary>,
    pub(crate) holdout_summary: Option<CliHyper2dDirectBasisOracleSummary>,
    pub(crate) entries: Vec<CliHyper2dDirectBasisOracleEntry>,
    pub(crate) elapsed_ms: f64,
}

#[derive(Clone, Copy, Serialize)]
pub(crate) struct CliHyper2dDirectBasisOracleSummary {
    pub(crate) examples: usize,
    pub(crate) mean_shared_loss: f32,
    pub(crate) mean_zero_loss: f32,
    pub(crate) mean_oracle_loss: f32,
    pub(crate) mean_gap_to_oracle: f32,
    pub(crate) mean_ratio_to_oracle: f32,
    pub(crate) max_ratio_to_oracle: f32,
    pub(crate) mean_gap_to_zero: f32,
    pub(crate) mean_ratio_to_zero: f32,
    pub(crate) max_ratio_to_zero: f32,
    pub(crate) mean_zero_ratio_to_oracle: f32,
}

#[derive(Clone, Serialize)]
pub(crate) struct CliHyper2dDirectBasisOracleEntry {
    pub(crate) slug: String,
    pub(crate) split: &'static str,
    pub(crate) condition: String,
    pub(crate) oracle_backend: DirectBasisOracleBackendArg,
    pub(crate) oracle_model_output: Option<String>,
    pub(crate) oracle_checkpoint_output: Option<String>,
    pub(crate) oracle_metrics_output: Option<String>,
    pub(crate) shared_loss: Target2dLossReport,
    pub(crate) zero_adapter_loss: Target2dLossReport,
    pub(crate) oracle_initial_eval_loss: Target2dLossReport,
    pub(crate) oracle_final_loss: Target2dLossReport,
    pub(crate) oracle_best_eval_loss: Target2dLossReport,
    pub(crate) oracle_epochs_completed: usize,
    pub(crate) oracle_median_particle_steps_per_sec: f64,
    pub(crate) loss_gap_to_oracle: f32,
    pub(crate) loss_ratio_to_oracle: f32,
    pub(crate) loss_gap_to_zero: f32,
    pub(crate) loss_ratio_to_zero: f32,
    pub(crate) zero_ratio_to_oracle: f32,
}

#[derive(Clone, Serialize)]
pub(crate) struct CliHyper2dDirectBasisAdapterReport {
    pub(crate) slug: String,
    pub(crate) split: &'static str,
    pub(crate) title: Option<String>,
    pub(crate) group: Option<String>,
    pub(crate) condition: String,
    pub(crate) adapter_output: String,
    pub(crate) target_source_width: usize,
    pub(crate) target_source_height: usize,
    pub(crate) target_points: usize,
    pub(crate) last_train_loss: Option<f32>,
    pub(crate) adapter_parameter_count: usize,
}

#[derive(Serialize)]
pub(crate) struct CliHyper2dHistoryEntry {
    pub(crate) step: usize,
    pub(crate) loss: f32,
    pub(crate) holdout_loss: Option<f32>,
    pub(crate) grad_norm: f32,
    pub(crate) grad_scale: f32,
}

#[derive(Clone, Serialize)]
pub(crate) struct CliHyper2dAdapterBootstrapReport {
    pub(crate) slug: String,
    pub(crate) method: &'static str,
    pub(crate) steps: usize,
    pub(crate) rows: usize,
    pub(crate) initial_loss: f32,
    pub(crate) final_loss: f32,
    pub(crate) best_loss: f32,
    pub(crate) adapter_parameter_count: usize,
}

#[derive(Serialize)]
pub(crate) struct CliHyper2dExampleReport {
    pub(crate) slug: String,
    pub(crate) title: Option<String>,
    pub(crate) group: Option<String>,
    pub(crate) condition: String,
    pub(crate) target_model: String,
    pub(crate) initial_loss: f32,
    pub(crate) final_loss: f32,
    pub(crate) rows: usize,
    pub(crate) particle_count: usize,
    pub(crate) rollout_steps: usize,
    pub(crate) rollouts: usize,
    pub(crate) update_prob: f32,
    pub(crate) seed_scale: f32,
    pub(crate) condition_summary: ConditionSummary2d,
    pub(crate) prior: ParticlePrior2d,
    pub(crate) image_metrics: Option<CliHyper2dImageMetricsReport>,
    pub(crate) dynamics_metrics: Option<CliHyper2dDynamicsMetricsReport>,
}

#[derive(Serialize)]
pub(crate) struct CliHyper2dImageMetricsReport {
    pub(crate) image_size: usize,
    pub(crate) rollout_steps: usize,
    pub(crate) particle_count: usize,
    pub(crate) update_prob: f32,
    pub(crate) seed: u64,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) decoder: &'static str,
    pub(crate) render_sigma_px: f32,
    pub(crate) domain_radius: f32,
    pub(crate) mse: f32,
    pub(crate) psnr_db: f32,
    pub(crate) luma_mse: f32,
    pub(crate) luma_psnr_db: f32,
    pub(crate) occupancy_mse: f32,
    pub(crate) occupancy_psnr_db: f32,
    pub(crate) foreground_iou: f32,
    pub(crate) generated_occupancy: f32,
    pub(crate) target_occupancy: f32,
}

#[derive(Serialize)]
pub(crate) struct CliHyper2dDynamicsMetricsReport {
    pub(crate) particle_count: usize,
    pub(crate) rollout_steps: usize,
    pub(crate) update_prob: f32,
    pub(crate) seed: u64,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) image_size: usize,
    pub(crate) render_sigma_px: f32,
    pub(crate) position_mse: f32,
    pub(crate) position_psnr_db: f32,
    pub(crate) state_mse: f32,
    pub(crate) state_psnr_db: f32,
    pub(crate) tail_rgb_mse: f32,
    pub(crate) tail_rgb_psnr_db: f32,
    pub(crate) render_rgb_mse: f32,
    pub(crate) render_rgb_psnr_db: f32,
    pub(crate) render_density_mse: f32,
    pub(crate) render_density_psnr_db: f32,
    pub(crate) mean_dx_mse: f32,
    pub(crate) mean_dx_mae: f32,
    pub(crate) target_final_mean_dx: f32,
    pub(crate) generated_final_mean_dx: f32,
}

#[derive(Serialize)]
pub(crate) struct CliHyper2dInferReport {
    pub(crate) preset: AutomataPreset,
    pub(crate) condition: String,
    pub(crate) base_model: Option<String>,
    pub(crate) hyper: String,
    pub(crate) adapter_output: Option<String>,
    pub(crate) materialized_output: Option<String>,
    pub(crate) rollout_output: Option<String>,
    pub(crate) npa_config: NpaConfig,
    pub(crate) hyper_config: HyperNpa2dConfig,
    pub(crate) condition_summary: ConditionSummary2d,
    pub(crate) prior: ParticlePrior2d,
    pub(crate) rollout_particles: Option<usize>,
    pub(crate) rollout_steps: Option<usize>,
    pub(crate) seed: Option<u64>,
    pub(crate) seed_scale: Option<f32>,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) adapter_parameter_count: usize,
    pub(crate) materialized_parameter_count: usize,
}

#[derive(Serialize)]
pub(crate) struct CliHyper2dEvalReport {
    pub(crate) preset: AutomataPreset,
    pub(crate) condition: Option<String>,
    pub(crate) catalog: Option<String>,
    pub(crate) catalog_group: Option<Hyper2dCatalogGroupArg>,
    pub(crate) catalog_targets: Vec<String>,
    pub(crate) base_model: Option<String>,
    pub(crate) hyper: String,
    pub(crate) report_output: String,
    pub(crate) generated_output_dir: Option<String>,
    pub(crate) npa_config: NpaConfig,
    pub(crate) hyper_config: HyperNpa2dConfig,
    pub(crate) rollout_supervision: CliRolloutSupervisionReport,
    pub(crate) train_loss: f32,
    pub(crate) holdout_loss: Option<f32>,
    pub(crate) train_examples: Vec<CliHyper2dExampleReport>,
    pub(crate) holdout_examples: Vec<CliHyper2dExampleReport>,
    pub(crate) adapter_parameter_count: usize,
    pub(crate) materialized_parameter_count: usize,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct CliRolloutSupervisionReport {
    pub(crate) particle_count: usize,
    pub(crate) rollout_steps: usize,
    pub(crate) rollouts: usize,
    pub(crate) temporal_samples: usize,
    pub(crate) update_prob: f32,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) motion_gain: Option<f32>,
    pub(crate) max_update_norm: Option<f32>,
    pub(crate) density_gain: Option<f32>,
    pub(crate) expansion_gain: Option<f32>,
    pub(crate) coverage_gain: Option<f32>,
    pub(crate) coverage_samples: Option<usize>,
    pub(crate) coverage_mode: Option<CoverageUpdateModeArg>,
    pub(crate) coverage_softness: Option<f32>,
    pub(crate) coverage_repulsion_gain: Option<f32>,
    pub(crate) coverage_gap_gain: Option<f32>,
    pub(crate) coverage_repulsion_radius: Option<f32>,
    pub(crate) coverage_normal_weight: Option<f32>,
    pub(crate) extent_gain: Option<f32>,
    pub(crate) color_gain: Option<f32>,
    pub(crate) aux_state_gain: Option<f32>,
    pub(crate) opacity_gain: Option<f32>,
    pub(crate) front_opacity_gain: Option<f32>,
    pub(crate) front_radius: Option<f32>,
    pub(crate) front_max_opacity_update: Option<f32>,
    pub(crate) front_motion_gate: Option<bool>,
    pub(crate) preserve_opacity_update: Option<bool>,
}
