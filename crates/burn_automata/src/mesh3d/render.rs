use crate::rollout::UV_TORUS_NORMAL_STATE_OFFSET;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mesh3dGaussianGeometry {
    /// Unit quaternion in scalar-first `[w, x, y, z]` order.
    pub rotation: [f32; 4],
    /// Two tangent radii followed by the surface-normal thickness.
    pub scale: [f32; 3],
}

/// Decodes the canonical mesh3d surface-state contract into an oriented splat.
///
/// The footprint tracks the expected two-dimensional surface spacing rather
/// than three-dimensional volume spacing. This keeps 4k and 16k renderings
/// visually comparable while retaining a thin normal axis.
pub fn mesh3d_gaussian_geometry(
    state: &[f32],
    interaction_radius: f32,
    particle_count: usize,
) -> Mesh3dGaussianGeometry {
    let normal = if state.len() >= UV_TORUS_NORMAL_STATE_OFFSET + 3 {
        [
            state[UV_TORUS_NORMAL_STATE_OFFSET],
            state[UV_TORUS_NORMAL_STATE_OFFSET + 1],
            state[UV_TORUS_NORMAL_STATE_OFFSET + 2],
        ]
    } else {
        [0.0, 0.0, 1.0]
    };
    let density_scale = (4096.0 / particle_count.max(1) as f32)
        .sqrt()
        .clamp(0.4, 2.0);
    let tangent = (interaction_radius * 0.32 * density_scale)
        .clamp(interaction_radius * 0.08, interaction_radius * 0.64)
        .max(0.00008);
    Mesh3dGaussianGeometry {
        rotation: quaternion_from_positive_z(normal),
        scale: [tangent, tangent, (tangent * 0.18).max(0.00008)],
    }
}

fn quaternion_from_positive_z(normal: [f32; 3]) -> [f32; 4] {
    let norm = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if !norm.is_finite() || norm <= 1.0e-6 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let normal = [normal[0] / norm, normal[1] / norm, normal[2] / norm];
    if normal[2] <= -0.999_999 {
        return [0.0, 1.0, 0.0, 0.0];
    }
    let mut quaternion = [1.0 + normal[2], -normal[1], normal[0], 0.0];
    let quaternion_norm = quaternion
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if !quaternion_norm.is_finite() || quaternion_norm <= 1.0e-6 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    for value in &mut quaternion {
        *value /= quaternion_norm;
    }
    quaternion
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rotate_positive_z(quaternion: [f32; 4]) -> [f32; 3] {
        let [w, x, y, z] = quaternion;
        [
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ]
    }

    #[test]
    fn surface_rotation_maps_positive_z_to_normal() {
        for normal in [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]] {
            let actual = rotate_positive_z(quaternion_from_positive_z(normal));
            for axis in 0..3 {
                assert!((actual[axis] - normal[axis]).abs() <= 1.0e-5);
            }
        }
    }

    #[test]
    fn surface_footprint_scales_with_inverse_sqrt_particle_count() {
        let state = [0.0; 24];
        let coarse = mesh3d_gaussian_geometry(&state, 0.1, 4096);
        let fine = mesh3d_gaussian_geometry(&state, 0.1, 16_384);
        assert!((coarse.scale[0] / fine.scale[0] - 2.0).abs() <= 1.0e-5);
        assert!(coarse.scale[2] < coarse.scale[0]);
    }
}
