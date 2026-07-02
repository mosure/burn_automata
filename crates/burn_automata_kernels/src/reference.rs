use rayon::prelude::*;

use crate::{
    HashGridConfig, HashGridMode, KernelError, KernelResult,
    hashgrid::{
        HashGridSnapshot, build_hashgrid, cell_coords_for_position, cell_index_from_coords,
        neighbor_delta, wrap_position,
    },
};

mod adjoint;
mod forward;
mod kernels;
mod neighbor;
mod normalize;
mod shape;
mod step;
mod types;

pub use adjoint::{perceive_adjoint_with_options, perceive_state_adjoint_with_options};
pub use forward::{perceive, perceive_with_options};
pub use step::euler_step;
pub use types::{PerceptionAdjointOutput, PerceptionOptions, PerceptionOutput};

pub(crate) use forward::density;
pub(crate) use kernels::{
    RecipFinite, safe_inverse_symmetric, smoothing_poly6_delta_adjoint, smoothing_poly6_kernel,
    spiky_gradient_delta_adjoint, spiky_gradient_kernel,
};
pub(crate) use neighbor::for_each_neighbor;
pub(crate) use normalize::{
    apply_moment_correction, log_normalize_adjoint, normalize_density_gradient,
    normalize_state_gradient,
};
pub(crate) use shape::check_shapes;
pub(crate) use types::{GROWTH_3D_MAX_OPACITY_LOGIT, GROWTH_3D_MIN_OPACITY_LOGIT};
