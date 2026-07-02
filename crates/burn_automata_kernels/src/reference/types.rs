pub(crate) const GROWTH_3D_MIN_OPACITY_LOGIT: f32 = -8.0;
pub(crate) const GROWTH_3D_MAX_OPACITY_LOGIT: f32 = 24.0;

#[derive(Clone, Copy, Debug)]
pub struct PerceptionOptions {
    pub state_grad: bool,
    pub density_grad: bool,
    pub eps0: f32,
    pub scale_equivariance: bool,
    pub particle_density_equivariance: bool,
    pub log_norm_grad: bool,
    pub log_norm_density_grad: bool,
    pub hybrid_state_gradient: bool,
    pub position_features: bool,
}

impl PerceptionOptions {
    pub fn new(state_grad: bool, density_grad: bool, eps0: f32) -> Self {
        Self {
            state_grad,
            density_grad,
            eps0,
            scale_equivariance: true,
            particle_density_equivariance: true,
            log_norm_grad: false,
            log_norm_density_grad: false,
            hybrid_state_gradient: true,
            position_features: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PerceptionOutput {
    pub features: Vec<f32>,
    pub density: Vec<f32>,
    pub blurred_state: Vec<f32>,
    pub state_gradient: Vec<f32>,
    pub density_gradient: Vec<f32>,
    pub feature_dims: usize,
}

#[derive(Clone, Debug)]
pub struct PerceptionAdjointOutput {
    pub state: Vec<f32>,
    pub position: Vec<[f32; 4]>,
}
