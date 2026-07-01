#[derive(Debug, thiserror::Error)]
pub enum AutomataError {
    #[error(transparent)]
    Kernel(#[from] burn_automata_kernels::KernelError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("invalid model: {0}")]
    InvalidModel(String),
    #[error("invalid model format: {0}")]
    InvalidFormat(String),
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

pub type AutomataResult<T> = Result<T, AutomataError>;
