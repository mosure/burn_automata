use super::{
    BurnDenseOracleBatchOutput, BurnE2eRolloutExample, BurnE2eRolloutOutput,
    BurnE2eRolloutTrainConfig, BurnWgpuDirectBasisOutput,
};

#[cfg(feature = "backend_wgpu")]
pub(in crate::cli::commands::hyper_e2e::direct_basis) fn train_direct_basis_burn_wgpu(
    base: &mut crate::NpaModel,
    train_examples: &mut [super::super::DirectBasisExample],
    holdout_examples: &mut [super::super::DirectBasisExample],
    train_config: super::super::DirectBasisTrainConfig,
    train_refine_config: super::super::DirectBasisTrainConfig,
    holdout_config: super::super::DirectBasisTrainConfig,
    checkpoint: Option<&super::super::Target2dBurnCheckpointConfig>,
) -> Result<BurnWgpuDirectBasisOutput, Box<dyn std::error::Error>> {
    super::wgpu_imp::train_direct_basis_burn_dense(
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
pub(in crate::cli::commands::hyper_e2e) fn train_e2e_rollout_burn_wgpu(
    base: &mut crate::NpaModel,
    train_examples: &mut [BurnE2eRolloutExample],
    holdout_examples: &mut [BurnE2eRolloutExample],
    train_config: BurnE2eRolloutTrainConfig,
) -> Result<BurnE2eRolloutOutput, Box<dyn std::error::Error>> {
    super::wgpu_imp::train_e2e_rollout_burn_dense(
        base,
        train_examples,
        holdout_examples,
        train_config,
    )
}

#[cfg(not(feature = "backend_wgpu"))]
pub(in crate::cli::commands::hyper_e2e) fn train_e2e_rollout_burn_wgpu(
    _base: &mut crate::NpaModel,
    _train_examples: &mut [BurnE2eRolloutExample],
    _holdout_examples: &mut [BurnE2eRolloutExample],
    _train_config: BurnE2eRolloutTrainConfig,
) -> Result<BurnE2eRolloutOutput, Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "Burn/WGPU HyperNPA e2e rollout training requires the backend_wgpu feature; rebuild with --features cli,backend_wgpu",
    )
    .into())
}

#[cfg(not(feature = "backend_wgpu"))]
pub(in crate::cli::commands::hyper_e2e::direct_basis) fn train_direct_basis_burn_wgpu(
    _base: &mut crate::NpaModel,
    _train_examples: &mut [super::super::DirectBasisExample],
    _holdout_examples: &mut [super::super::DirectBasisExample],
    _train_config: super::super::DirectBasisTrainConfig,
    _train_refine_config: super::super::DirectBasisTrainConfig,
    _holdout_config: super::super::DirectBasisTrainConfig,
    _checkpoint: Option<&super::super::Target2dBurnCheckpointConfig>,
) -> Result<BurnWgpuDirectBasisOutput, Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "Burn/WGPU direct-basis training requires the backend_wgpu feature; rebuild with --features cli,backend_wgpu or choose the Burn/CUDA backend in a CUDA build",
    )
    .into())
}

#[cfg(feature = "backend_wgpu")]
pub(in crate::cli::commands::hyper_e2e::direct_basis) fn train_oracle_models_burn_wgpu(
    models: &mut [crate::NpaModel],
    examples: &[super::super::DirectBasisExample],
    train_config: super::super::DirectBasisTrainConfig,
) -> Result<BurnDenseOracleBatchOutput, Box<dyn std::error::Error>> {
    super::wgpu_imp::train_oracle_models_burn_dense(models, examples, train_config)
}

#[cfg(not(feature = "backend_wgpu"))]
pub(in crate::cli::commands::hyper_e2e::direct_basis) fn train_oracle_models_burn_wgpu(
    _models: &mut [crate::NpaModel],
    _examples: &[super::super::DirectBasisExample],
    _train_config: super::super::DirectBasisTrainConfig,
) -> Result<BurnDenseOracleBatchOutput, Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "Burn/WGPU vectorized oracle training requires the backend_wgpu feature; rebuild with --features cli,backend_wgpu",
    )
    .into())
}

#[cfg(feature = "backend_cuda")]
pub(in crate::cli::commands::hyper_e2e::direct_basis) fn train_direct_basis_burn_cuda(
    base: &mut crate::NpaModel,
    train_examples: &mut [super::super::DirectBasisExample],
    holdout_examples: &mut [super::super::DirectBasisExample],
    train_config: super::super::DirectBasisTrainConfig,
    train_refine_config: super::super::DirectBasisTrainConfig,
    holdout_config: super::super::DirectBasisTrainConfig,
    checkpoint: Option<&super::super::Target2dBurnCheckpointConfig>,
) -> Result<BurnWgpuDirectBasisOutput, Box<dyn std::error::Error>> {
    super::cuda_imp::train_direct_basis_burn_dense(
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
pub(in crate::cli::commands::hyper_e2e) fn train_e2e_rollout_burn_cuda(
    base: &mut crate::NpaModel,
    train_examples: &mut [BurnE2eRolloutExample],
    holdout_examples: &mut [BurnE2eRolloutExample],
    train_config: BurnE2eRolloutTrainConfig,
) -> Result<BurnE2eRolloutOutput, Box<dyn std::error::Error>> {
    super::cuda_imp::train_e2e_rollout_burn_dense(
        base,
        train_examples,
        holdout_examples,
        train_config,
    )
}

#[cfg(not(feature = "backend_cuda"))]
pub(in crate::cli::commands::hyper_e2e) fn train_e2e_rollout_burn_cuda(
    _base: &mut crate::NpaModel,
    _train_examples: &mut [BurnE2eRolloutExample],
    _holdout_examples: &mut [BurnE2eRolloutExample],
    _train_config: BurnE2eRolloutTrainConfig,
) -> Result<BurnE2eRolloutOutput, Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "Burn/CUDA HyperNPA e2e rollout training requires the backend_cuda feature; rebuild with --features cli,backend_cuda or use backend = \"burn-wgpu\"",
    )
    .into())
}

#[cfg(not(feature = "backend_cuda"))]
pub(in crate::cli::commands::hyper_e2e::direct_basis) fn train_direct_basis_burn_cuda(
    _base: &mut crate::NpaModel,
    _train_examples: &mut [super::super::DirectBasisExample],
    _holdout_examples: &mut [super::super::DirectBasisExample],
    _train_config: super::super::DirectBasisTrainConfig,
    _train_refine_config: super::super::DirectBasisTrainConfig,
    _holdout_config: super::super::DirectBasisTrainConfig,
    _checkpoint: Option<&super::super::Target2dBurnCheckpointConfig>,
) -> Result<BurnWgpuDirectBasisOutput, Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "Burn/CUDA dense direct-basis training requires the backend_cuda feature; rebuild with --features cli,backend_cuda or use backend = \"burn-wgpu\"",
    )
    .into())
}

#[cfg(feature = "backend_cuda")]
pub(in crate::cli::commands::hyper_e2e::direct_basis) fn train_oracle_models_burn_cuda(
    models: &mut [crate::NpaModel],
    examples: &[super::super::DirectBasisExample],
    train_config: super::super::DirectBasisTrainConfig,
) -> Result<BurnDenseOracleBatchOutput, Box<dyn std::error::Error>> {
    super::cuda_imp::train_oracle_models_burn_dense(models, examples, train_config)
}

#[cfg(not(feature = "backend_cuda"))]
pub(in crate::cli::commands::hyper_e2e::direct_basis) fn train_oracle_models_burn_cuda(
    _models: &mut [crate::NpaModel],
    _examples: &[super::super::DirectBasisExample],
    _train_config: super::super::DirectBasisTrainConfig,
) -> Result<BurnDenseOracleBatchOutput, Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "Burn/CUDA vectorized oracle training requires the backend_cuda feature; rebuild with --features cli,backend_cuda or use backend = \"burn-wgpu\"",
    )
    .into())
}
