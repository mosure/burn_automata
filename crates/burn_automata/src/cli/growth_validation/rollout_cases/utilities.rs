#![allow(clippy::too_many_arguments)]

use super::*;

pub(crate) fn run_rollout_from_state(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    mut positions: Vec<[f32; 4]>,
    mut states: Vec<f32>,
    batch_size: usize,
    particle_count: usize,
    steps: usize,
    dt: f32,
) -> Result<crate::RolloutTrace, Box<dyn std::error::Error>> {
    let mut mean_dx = Vec::with_capacity(steps);
    for _ in 0..steps {
        let step = model.step_cpu(
            &positions,
            &states,
            batch_size,
            particle_count,
            grid,
            dt,
            None,
        )?;
        let dx_norm = step
            .dx
            .iter()
            .map(|delta| (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt())
            .sum::<f32>()
            / step.dx.len().max(1) as f32;
        mean_dx.push(dx_norm);
        positions = step.next_positions;
        states = step.next_states;
    }

    Ok(crate::RolloutTrace {
        positions,
        states,
        batch_size,
        particle_count,
        state_dims: model.config.state_dims,
        steps,
        mean_dx,
    })
}

pub(crate) fn growth_3d_surface_stats(
    positions: &[[f32; 4]],
    target: &TriangleMeshTarget,
) -> Growth3dSurfaceStats {
    let mut max_distance = 0.0_f32;
    let mut sum_distance = 0.0_f32;
    for position in positions {
        let projection = target.project([position[0], position[1], position[2]]);
        max_distance = max_distance.max(projection.distance);
        sum_distance += projection.distance;
    }
    Growth3dSurfaceStats {
        mean_distance: sum_distance / positions.len().max(1) as f32,
        max_distance,
    }
}

pub(crate) fn growth_3d_active_surface_stats(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
) -> Growth3dSurfaceStats {
    let mut max_distance = 0.0_f32;
    let mut sum_distance = 0.0_f32;
    let mut count = 0usize;
    for (idx, position) in positions.iter().enumerate() {
        if state_dims <= 3 || states[idx * state_dims + 3] <= -1.0 {
            continue;
        }
        let projection = target.project([position[0], position[1], position[2]]);
        max_distance = max_distance.max(projection.distance);
        sum_distance += projection.distance;
        count += 1;
    }
    Growth3dSurfaceStats {
        mean_distance: if count > 0 {
            sum_distance / count as f32
        } else {
            f32::INFINITY
        },
        max_distance: if count > 0 {
            max_distance
        } else {
            f32::INFINITY
        },
    }
}

pub(crate) fn growth_3d_active_surface_tail_report(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
    threshold: f32,
) -> Growth3dSurfaceTailReport {
    let mut distances = Vec::new();
    let mut max_distance = 0.0_f32;
    let mut over_threshold_count = 0usize;
    let mut weighted_sum = 0.0_f32;
    let mut weight_sum = 0.0_f32;
    let mut weighted_over_threshold_sum = 0.0_f32;

    if state_dims > 3 {
        for (idx, position) in positions.iter().enumerate() {
            let opacity_logit = states[idx * state_dims + 3];
            if opacity_logit <= -1.0 {
                continue;
            }
            let projection = target.project([position[0], position[1], position[2]]);
            let distance = projection.distance;
            let weight = sigmoid_unit(opacity_logit);
            max_distance = max_distance.max(distance);
            if distance >= threshold {
                over_threshold_count += 1;
                weighted_over_threshold_sum += weight;
            }
            weighted_sum += distance * weight;
            weight_sum += weight;
            distances.push(distance);
        }
    }

    if distances.is_empty() {
        return empty_growth_3d_surface_tail_report(threshold);
    }

    distances.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let count = distances.len();
    Growth3dSurfaceTailReport {
        count,
        threshold,
        p95_distance: percentile_from_sorted(&distances, 0.95),
        p99_distance: percentile_from_sorted(&distances, 0.99),
        max_distance,
        over_threshold_count,
        over_threshold_fraction: over_threshold_count as f32 / count as f32,
        opacity_weighted_mean_distance: weighted_sum / weight_sum.max(1.0e-8),
        opacity_weighted_over_threshold_fraction: weighted_over_threshold_sum
            / weight_sum.max(1.0e-8),
    }
}

pub(crate) fn growth_3d_material_visible_surface_tail_report(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
    threshold: f32,
) -> Growth3dSurfaceTailReport {
    let Some(material_channel) = growth_3d_material_opacity_channel(state_dims) else {
        return empty_growth_3d_surface_tail_report(threshold);
    };
    let mut distances = Vec::new();
    let mut max_distance = 0.0_f32;
    let mut over_threshold_count = 0usize;
    let mut weighted_sum = 0.0_f32;
    let mut weight_sum = 0.0_f32;
    let mut weighted_over_threshold_sum = 0.0_f32;

    for (idx, position) in positions.iter().enumerate() {
        let material_logit = states[idx * state_dims + material_channel];
        if material_logit <= -1.0 {
            continue;
        }
        let projection = target.project([position[0], position[1], position[2]]);
        let distance = projection.distance;
        let weight = sigmoid_unit(material_logit);
        max_distance = max_distance.max(distance);
        if distance >= threshold {
            over_threshold_count += 1;
            weighted_over_threshold_sum += weight;
        }
        weighted_sum += distance * weight;
        weight_sum += weight;
        distances.push(distance);
    }

    if distances.is_empty() {
        return empty_growth_3d_surface_tail_report(threshold);
    }

    distances.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let count = distances.len();
    Growth3dSurfaceTailReport {
        count,
        threshold,
        p95_distance: percentile_from_sorted(&distances, 0.95),
        p99_distance: percentile_from_sorted(&distances, 0.99),
        max_distance,
        over_threshold_count,
        over_threshold_fraction: over_threshold_count as f32 / count as f32,
        opacity_weighted_mean_distance: weighted_sum / weight_sum.max(1.0e-8),
        opacity_weighted_over_threshold_fraction: weighted_over_threshold_sum
            / weight_sum.max(1.0e-8),
    }
}

pub(crate) fn empty_growth_3d_surface_tail_report(threshold: f32) -> Growth3dSurfaceTailReport {
    Growth3dSurfaceTailReport {
        count: 0,
        threshold,
        p95_distance: f32::INFINITY,
        p99_distance: f32::INFINITY,
        max_distance: f32::INFINITY,
        over_threshold_count: 0,
        over_threshold_fraction: 0.0,
        opacity_weighted_mean_distance: f32::INFINITY,
        opacity_weighted_over_threshold_fraction: 0.0,
    }
}

pub(crate) fn percentile_from_sorted(values: &[f32], percentile: f32) -> f32 {
    if values.is_empty() {
        return f32::INFINITY;
    }
    let clamped = percentile.clamp(0.0, 1.0);
    let idx = ((values.len() as f32 * clamped).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1);
    values[idx]
}

pub(crate) fn sigmoid_unit(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

pub(crate) fn sigmoid_unit_derivative(value: f32) -> f32 {
    let sigmoid = sigmoid_unit(value);
    sigmoid * (1.0 - sigmoid)
}

pub(crate) fn growth_3d_mean_displacement(
    initial: &[[f32; 4]],
    final_positions: &[[f32; 4]],
) -> f32 {
    initial
        .iter()
        .zip(final_positions.iter())
        .map(|(a, b)| {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
        })
        .sum::<f32>()
        / initial.len().max(1) as f32
}
