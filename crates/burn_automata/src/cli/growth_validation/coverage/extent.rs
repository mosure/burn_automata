use super::*;

pub(crate) fn growth_3d_extent_report(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
) -> Growth3dExtentReport {
    let (target_bounds_min, target_bounds_max) = target.bounds();
    let target_extent = [
        target_bounds_max[0] - target_bounds_min[0],
        target_bounds_max[1] - target_bounds_min[1],
        target_bounds_max[2] - target_bounds_min[2],
    ];
    let target_max_radius = target
        .vertices
        .iter()
        .map(|position| {
            (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                .sqrt()
        })
        .fold(0.0_f32, f32::max);

    let mut active_bounds_min = [f32::MAX; 3];
    let mut active_bounds_max = [f32::MIN; 3];
    let mut active_count = 0usize;
    let mut final_active_max_radius = 0.0_f32;
    for (idx, position) in positions.iter().enumerate() {
        let opacity = states[idx * state_dims + 3];
        if opacity <= -1.0 {
            continue;
        }
        active_count += 1;
        for axis in 0..3 {
            active_bounds_min[axis] = active_bounds_min[axis].min(position[axis]);
            active_bounds_max[axis] = active_bounds_max[axis].max(position[axis]);
        }
        final_active_max_radius = final_active_max_radius.max(
            (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                .sqrt(),
        );
    }

    if active_count == 0 {
        active_bounds_min = [0.0; 3];
        active_bounds_max = [0.0; 3];
    }
    let final_active_extent = [
        active_bounds_max[0] - active_bounds_min[0],
        active_bounds_max[1] - active_bounds_min[1],
        active_bounds_max[2] - active_bounds_min[2],
    ];
    let axis_extent_ratio = [
        final_active_extent[0] / target_extent[0].max(1.0e-6),
        final_active_extent[1] / target_extent[1].max(1.0e-6),
        final_active_extent[2] / target_extent[2].max(1.0e-6),
    ];
    let min_axis_extent_ratio = axis_extent_ratio
        .iter()
        .copied()
        .fold(f32::MAX, f32::min)
        .min(1.0e6);
    let target_diag = (target_extent[0] * target_extent[0]
        + target_extent[1] * target_extent[1]
        + target_extent[2] * target_extent[2])
        .sqrt();
    let active_diag = (final_active_extent[0] * final_active_extent[0]
        + final_active_extent[1] * final_active_extent[1]
        + final_active_extent[2] * final_active_extent[2])
        .sqrt();

    Growth3dExtentReport {
        target_bounds_min,
        target_bounds_max,
        final_active_bounds_min: active_bounds_min,
        final_active_bounds_max: active_bounds_max,
        target_extent,
        final_active_extent,
        axis_extent_ratio,
        min_axis_extent_ratio,
        bbox_diagonal_ratio: active_diag / target_diag.max(1.0e-6),
        target_max_radius,
        final_active_max_radius,
        max_radius_ratio: final_active_max_radius / target_max_radius.max(1.0e-6),
    }
}
