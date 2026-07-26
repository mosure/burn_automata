use super::super::{
    AdaptiveTarget2dBurnConfig, BurnDenseOracleBatchOutput, BurnE2eRolloutExample,
    BurnE2eRolloutOutput, BurnE2eRolloutTrainConfig, BurnWgpuDirectBasisOutput,
    DirectBasisTrainConfig, DirectBasisTrainingExample, Target2dBurnCheckpointConfig,
    Target2dOracleTrainPlan,
};

#[cfg(feature = "backend_wgpu")]
pub(crate) fn train_adaptive_target2d_burn_wgpu(
    model: &mut crate::NpaModel,
    example: &DirectBasisTrainingExample,
    plan: Target2dOracleTrainPlan,
    adaptive: AdaptiveTarget2dBurnConfig,
    checkpoint: Option<&Target2dBurnCheckpointConfig>,
    observer: Option<&mut dyn crate::Target2dGpuTrainingObserver>,
) -> Result<BurnDenseOracleBatchOutput, Box<dyn std::error::Error>> {
    super::wgpu_imp::adaptive_training::train_adaptive_target2d_burn_dense(
        model, example, plan, adaptive, checkpoint, observer,
    )
}

#[cfg(not(feature = "backend_wgpu"))]
pub(crate) fn train_adaptive_target2d_burn_wgpu(
    _model: &mut crate::NpaModel,
    _example: &DirectBasisTrainingExample,
    _plan: Target2dOracleTrainPlan,
    _adaptive: AdaptiveTarget2dBurnConfig,
    _checkpoint: Option<&Target2dBurnCheckpointConfig>,
    _observer: Option<&mut dyn crate::Target2dGpuTrainingObserver>,
) -> Result<BurnDenseOracleBatchOutput, Box<dyn std::error::Error>> {
    Err(
        std::io::Error::other("adaptive Target2D training requires the backend_wgpu feature")
            .into(),
    )
}

#[cfg(feature = "backend_cuda")]
pub(crate) fn train_adaptive_target2d_burn_cuda(
    model: &mut crate::NpaModel,
    example: &DirectBasisTrainingExample,
    plan: Target2dOracleTrainPlan,
    adaptive: AdaptiveTarget2dBurnConfig,
    checkpoint: Option<&Target2dBurnCheckpointConfig>,
    observer: Option<&mut dyn crate::Target2dGpuTrainingObserver>,
) -> Result<BurnDenseOracleBatchOutput, Box<dyn std::error::Error>> {
    super::cuda_imp::adaptive_training::train_adaptive_target2d_burn_dense(
        model, example, plan, adaptive, checkpoint, observer,
    )
}

#[cfg(not(feature = "backend_cuda"))]
pub(crate) fn train_adaptive_target2d_burn_cuda(
    _model: &mut crate::NpaModel,
    _example: &DirectBasisTrainingExample,
    _plan: Target2dOracleTrainPlan,
    _adaptive: AdaptiveTarget2dBurnConfig,
    _checkpoint: Option<&Target2dBurnCheckpointConfig>,
    _observer: Option<&mut dyn crate::Target2dGpuTrainingObserver>,
) -> Result<BurnDenseOracleBatchOutput, Box<dyn std::error::Error>> {
    Err(
        std::io::Error::other("adaptive Target2D training requires the backend_cuda feature")
            .into(),
    )
}

#[cfg(feature = "backend_wgpu")]
pub(crate) fn predict_conditional_row_flow_adapter_wgpu(
    hyper: &crate::E2eHyperNpa2d,
    config: &crate::NpaConfig,
    condition: &[f32],
) -> crate::AutomataResult<crate::NpaLowRankAdapter> {
    super::wgpu_imp::entrypoints::predict_conditional_row_flow_adapter(hyper, config, condition)
}

#[cfg(feature = "backend_cuda")]
pub(crate) fn predict_conditional_row_flow_adapter_cuda(
    hyper: &crate::E2eHyperNpa2d,
    config: &crate::NpaConfig,
    condition: &[f32],
) -> crate::AutomataResult<crate::NpaLowRankAdapter> {
    super::cuda_imp::entrypoints::predict_conditional_row_flow_adapter(hyper, config, condition)
}

#[cfg(feature = "backend_wgpu")]
pub(crate) fn train_direct_basis_burn_wgpu(
    base: &mut crate::NpaModel,
    train_examples: &mut [DirectBasisTrainingExample],
    holdout_examples: &mut [DirectBasisTrainingExample],
    train_config: DirectBasisTrainConfig,
    train_refine_config: DirectBasisTrainConfig,
    holdout_config: DirectBasisTrainConfig,
    checkpoint: Option<&Target2dBurnCheckpointConfig>,
) -> Result<BurnWgpuDirectBasisOutput, Box<dyn std::error::Error>> {
    super::wgpu_imp::entrypoints::train_direct_basis_burn_dense(
        base,
        train_examples,
        holdout_examples,
        train_config,
        train_refine_config,
        holdout_config,
        checkpoint,
    )
}

#[cfg(feature = "backend_wgpu")]
pub(crate) fn train_e2e_rollout_burn_wgpu(
    base: &mut crate::NpaModel,
    train_examples: &mut [BurnE2eRolloutExample],
    holdout_examples: &mut [BurnE2eRolloutExample],
    train_config: BurnE2eRolloutTrainConfig,
    initial_generator: Option<&crate::E2eHyperNpa2d>,
) -> Result<BurnE2eRolloutOutput, Box<dyn std::error::Error>> {
    super::wgpu_imp::entrypoints::train_e2e_rollout_burn_dense(
        base,
        train_examples,
        holdout_examples,
        train_config,
        initial_generator,
    )
}

#[cfg(not(feature = "backend_wgpu"))]
pub(crate) fn train_e2e_rollout_burn_wgpu(
    _base: &mut crate::NpaModel,
    _train_examples: &mut [BurnE2eRolloutExample],
    _holdout_examples: &mut [BurnE2eRolloutExample],
    _train_config: BurnE2eRolloutTrainConfig,
    _initial_generator: Option<&crate::E2eHyperNpa2d>,
) -> Result<BurnE2eRolloutOutput, Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "Burn/WGPU HyperNPA e2e rollout training requires the backend_wgpu feature; rebuild with --features cli,backend_wgpu",
    )
    .into())
}

#[cfg(not(feature = "backend_wgpu"))]
pub(crate) fn train_direct_basis_burn_wgpu(
    _base: &mut crate::NpaModel,
    _train_examples: &mut [DirectBasisTrainingExample],
    _holdout_examples: &mut [DirectBasisTrainingExample],
    _train_config: DirectBasisTrainConfig,
    _train_refine_config: DirectBasisTrainConfig,
    _holdout_config: DirectBasisTrainConfig,
    _checkpoint: Option<&Target2dBurnCheckpointConfig>,
) -> Result<BurnWgpuDirectBasisOutput, Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "Burn/WGPU direct-basis training requires the backend_wgpu feature; rebuild with --features cli,backend_wgpu or choose the Burn/CUDA backend in a CUDA build",
    )
    .into())
}

#[cfg(feature = "backend_wgpu")]
pub(crate) fn train_oracle_models_burn_wgpu(
    models: &mut [crate::NpaModel],
    examples: &[DirectBasisTrainingExample],
    train_config: DirectBasisTrainConfig,
) -> Result<BurnDenseOracleBatchOutput, Box<dyn std::error::Error>> {
    super::wgpu_imp::entrypoints::train_oracle_models_burn_dense(models, examples, train_config)
}

#[cfg(feature = "backend_wgpu")]
pub(crate) fn train_target2d_oracle_burn_wgpu(
    model: &mut crate::NpaModel,
    example: &DirectBasisTrainingExample,
    plan: Target2dOracleTrainPlan,
    checkpoint: Option<&Target2dBurnCheckpointConfig>,
    observer: Option<&mut dyn crate::Target2dGpuTrainingObserver>,
) -> Result<BurnDenseOracleBatchOutput, Box<dyn std::error::Error>> {
    super::wgpu_imp::entrypoints::train_target2d_oracle_burn_dense(
        model, example, plan, checkpoint, observer,
    )
}

#[cfg(not(feature = "backend_wgpu"))]
pub(crate) fn train_target2d_oracle_burn_wgpu(
    _model: &mut crate::NpaModel,
    _example: &DirectBasisTrainingExample,
    _plan: Target2dOracleTrainPlan,
    _checkpoint: Option<&Target2dBurnCheckpointConfig>,
    _observer: Option<&mut dyn crate::Target2dGpuTrainingObserver>,
) -> Result<BurnDenseOracleBatchOutput, Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "Burn/WGPU target2d oracle training requires the backend_wgpu feature; rebuild with --features cli,backend_wgpu",
    )
    .into())
}

#[cfg(not(feature = "backend_wgpu"))]
pub(crate) fn train_oracle_models_burn_wgpu(
    _models: &mut [crate::NpaModel],
    _examples: &[DirectBasisTrainingExample],
    _train_config: DirectBasisTrainConfig,
) -> Result<BurnDenseOracleBatchOutput, Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "Burn/WGPU vectorized oracle training requires the backend_wgpu feature; rebuild with --features cli,backend_wgpu",
    )
    .into())
}

#[cfg(feature = "backend_cuda")]
pub(crate) fn train_direct_basis_burn_cuda(
    base: &mut crate::NpaModel,
    train_examples: &mut [DirectBasisTrainingExample],
    holdout_examples: &mut [DirectBasisTrainingExample],
    train_config: DirectBasisTrainConfig,
    train_refine_config: DirectBasisTrainConfig,
    holdout_config: DirectBasisTrainConfig,
    checkpoint: Option<&Target2dBurnCheckpointConfig>,
) -> Result<BurnWgpuDirectBasisOutput, Box<dyn std::error::Error>> {
    super::cuda_imp::entrypoints::train_direct_basis_burn_dense(
        base,
        train_examples,
        holdout_examples,
        train_config,
        train_refine_config,
        holdout_config,
        checkpoint,
    )
}

#[cfg(feature = "backend_cuda")]
pub(crate) fn train_e2e_rollout_burn_cuda(
    base: &mut crate::NpaModel,
    train_examples: &mut [BurnE2eRolloutExample],
    holdout_examples: &mut [BurnE2eRolloutExample],
    train_config: BurnE2eRolloutTrainConfig,
    initial_generator: Option<&crate::E2eHyperNpa2d>,
) -> Result<BurnE2eRolloutOutput, Box<dyn std::error::Error>> {
    super::cuda_imp::entrypoints::train_e2e_rollout_burn_dense(
        base,
        train_examples,
        holdout_examples,
        train_config,
        initial_generator,
    )
}

#[cfg(not(feature = "backend_cuda"))]
pub(crate) fn train_e2e_rollout_burn_cuda(
    _base: &mut crate::NpaModel,
    _train_examples: &mut [BurnE2eRolloutExample],
    _holdout_examples: &mut [BurnE2eRolloutExample],
    _train_config: BurnE2eRolloutTrainConfig,
    _initial_generator: Option<&crate::E2eHyperNpa2d>,
) -> Result<BurnE2eRolloutOutput, Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "Burn/CUDA HyperNPA e2e rollout training requires the backend_cuda feature; rebuild with --features cli,backend_cuda or use backend = \"burn-wgpu\"",
    )
    .into())
}

#[cfg(not(feature = "backend_cuda"))]
pub(crate) fn train_direct_basis_burn_cuda(
    _base: &mut crate::NpaModel,
    _train_examples: &mut [DirectBasisTrainingExample],
    _holdout_examples: &mut [DirectBasisTrainingExample],
    _train_config: DirectBasisTrainConfig,
    _train_refine_config: DirectBasisTrainConfig,
    _holdout_config: DirectBasisTrainConfig,
    _checkpoint: Option<&Target2dBurnCheckpointConfig>,
) -> Result<BurnWgpuDirectBasisOutput, Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "Burn/CUDA dense direct-basis training requires the backend_cuda feature; rebuild with --features cli,backend_cuda or use backend = \"burn-wgpu\"",
    )
    .into())
}

#[cfg(feature = "backend_cuda")]
pub(crate) fn train_oracle_models_burn_cuda(
    models: &mut [crate::NpaModel],
    examples: &[DirectBasisTrainingExample],
    train_config: DirectBasisTrainConfig,
) -> Result<BurnDenseOracleBatchOutput, Box<dyn std::error::Error>> {
    super::cuda_imp::entrypoints::train_oracle_models_burn_dense(models, examples, train_config)
}

#[cfg(feature = "backend_cuda")]
pub(crate) fn train_target2d_oracle_burn_cuda(
    model: &mut crate::NpaModel,
    example: &DirectBasisTrainingExample,
    plan: Target2dOracleTrainPlan,
    checkpoint: Option<&Target2dBurnCheckpointConfig>,
    observer: Option<&mut dyn crate::Target2dGpuTrainingObserver>,
) -> Result<BurnDenseOracleBatchOutput, Box<dyn std::error::Error>> {
    super::cuda_imp::entrypoints::train_target2d_oracle_burn_dense(
        model, example, plan, checkpoint, observer,
    )
}

#[cfg(not(feature = "backend_cuda"))]
pub(crate) fn train_target2d_oracle_burn_cuda(
    _model: &mut crate::NpaModel,
    _example: &DirectBasisTrainingExample,
    _plan: Target2dOracleTrainPlan,
    _checkpoint: Option<&Target2dBurnCheckpointConfig>,
    _observer: Option<&mut dyn crate::Target2dGpuTrainingObserver>,
) -> Result<BurnDenseOracleBatchOutput, Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "Burn/CUDA target2d oracle training requires the backend_cuda feature; rebuild with --features cli,backend_cuda",
    )
    .into())
}

#[cfg(not(feature = "backend_cuda"))]
pub(crate) fn train_oracle_models_burn_cuda(
    _models: &mut [crate::NpaModel],
    _examples: &[DirectBasisTrainingExample],
    _train_config: DirectBasisTrainConfig,
) -> Result<BurnDenseOracleBatchOutput, Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "Burn/CUDA vectorized oracle training requires the backend_cuda feature; rebuild with --features cli,backend_cuda or use backend = \"burn-wgpu\"",
    )
    .into())
}
