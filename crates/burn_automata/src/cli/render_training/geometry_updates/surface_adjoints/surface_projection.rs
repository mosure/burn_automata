#![allow(
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use super::*;

pub(crate) fn add_surface_position_adjoint(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    surface_gain: f32,
    surface_escape_gain: f32,
    position_adjoint: &mut [[f32; 4]],
) {
    if surface_gain <= 0.0 || !surface_gain.is_finite() {
        return;
    }
    let updates = render_proxy_surface_projection_updates(
        config,
        target,
        positions,
        states,
        surface_gain,
        surface_escape_gain,
        f32::INFINITY,
    );
    for (row, update) in updates.iter().enumerate() {
        if row >= position_adjoint.len() {
            break;
        }
        for axis in 0..config.spatial_dims {
            position_adjoint[row][axis] -= update[axis];
        }
        clamp_position_adjoint_row(&mut position_adjoint[row], config.spatial_dims);
    }
}

pub(crate) fn add_material_visible_surface_position_adjoint(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    surface_gain: f32,
    surface_escape_gain: f32,
    seed_scale: f32,
    front_radius: f32,
    position_adjoint: &mut [[f32; 4]],
) {
    if surface_gain <= 0.0 || !surface_gain.is_finite() {
        return;
    }
    let updates = material_visible_surface_approach_updates(
        config,
        target,
        positions,
        states,
        None,
        surface_gain,
        surface_escape_gain,
        f32::INFINITY,
        seed_scale,
        front_radius,
        None,
    );
    for (row, update) in updates.iter().enumerate() {
        if row >= position_adjoint.len() {
            break;
        }
        for axis in 0..config.spatial_dims {
            position_adjoint[row][axis] -= update[axis];
        }
        clamp_position_adjoint_row(&mut position_adjoint[row], config.spatial_dims);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_material_visible_surface_coverage_position_adjoint(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    coverage_gain: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    coverage_mode: CoverageUpdateModeArg,
    coverage_softness: f32,
    coverage_repulsion_gain: f32,
    coverage_gap_gain: f32,
    coverage_repulsion_radius: f32,
    coverage_normal_weight: f32,
    seed_scale: f32,
    front_radius: f32,
    position_adjoint: &mut [[f32; 4]],
) {
    let updates = material_visible_surface_coverage_updates(
        config,
        target,
        positions,
        states,
        None,
        coverage_gain,
        coverage_samples,
        max_update_norm,
        coverage_mode,
        coverage_softness,
        coverage_repulsion_gain,
        coverage_gap_gain,
        coverage_repulsion_radius,
        coverage_normal_weight,
        seed_scale,
        front_radius,
        None,
    );
    for (row, update) in updates.iter().enumerate() {
        if row >= position_adjoint.len() {
            break;
        }
        for axis in 0..config.spatial_dims {
            position_adjoint[row][axis] -= update[axis];
        }
        clamp_position_adjoint_row(&mut position_adjoint[row], config.spatial_dims);
    }
}

pub(crate) fn render_proxy_surface_projection_updates(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    surface_gain: f32,
    surface_escape_gain: f32,
    max_update_norm: f32,
) -> Vec<[f32; 3]> {
    let mut updates = vec![[0.0; 3]; positions.len()];
    if surface_gain <= 0.0 || !surface_gain.is_finite() {
        return updates;
    }
    for (row, position) in positions.iter().enumerate() {
        if config.state_dims > 3
            && states
                .get(row * config.state_dims + 3)
                .is_some_and(|opacity| *opacity <= -1.0)
        {
            continue;
        }
        let projection = target.project(position3(*position));
        let weight = surface_escape_weight(
            projection.distance,
            GROWTH_3D_SURFACE_MAX_DISTANCE,
            surface_escape_gain,
        );
        updates[row] = [
            surface_gain * weight * projection.residual[0],
            surface_gain * weight * projection.residual[1],
            surface_gain * weight * projection.residual[2],
        ];
        clamp_update_row(&mut updates, row, max_update_norm);
    }
    updates
}
