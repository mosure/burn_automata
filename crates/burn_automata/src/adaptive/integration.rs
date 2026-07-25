use burn_automata_kernels::adaptive_perceive_without_spacing;
use std::collections::BTreeMap;

use super::{AdaptiveNpaModel, AdaptiveParticleSet};
use crate::{AutomataError, AutomataResult, rollout::stable_material_uniform};

pub(crate) fn integration_masks(
    model: &AdaptiveNpaModel,
    particles: &AdaptiveParticleSet,
    rollout_seed: u64,
    absolute_step: usize,
    update_prob: f32,
) -> Vec<f32> {
    let templates = particles
        .bootstrap_templates
        .iter()
        .map(|template| (template.parent_id, template))
        .collect::<BTreeMap<_, _>>();
    (0..particles.len())
        .map(|row| {
            if model.config.expected_coarse_update_mask && is_coarse_material(model, particles, row)
            {
                update_prob
            } else if let Some(template) = templates.get(&particles.particle_id[row]) {
                let total = template
                    .children
                    .iter()
                    .map(|child| child.represented_measure)
                    .sum::<f32>()
                    .max(f32::MIN_POSITIVE);
                template
                    .children
                    .iter()
                    .map(|child| {
                        child.represented_measure / total
                            * f32::from(
                                stable_material_uniform(
                                    rollout_seed,
                                    absolute_step,
                                    child.particle_id,
                                ) < update_prob,
                            )
                    })
                    .sum()
            } else {
                f32::from(
                    stable_material_uniform(
                        rollout_seed,
                        absolute_step,
                        particles.particle_id[row],
                    ) < update_prob,
                )
            }
        })
        .collect()
}

pub(crate) fn integrate_represented_measure_update(
    model: &AdaptiveNpaModel,
    particles: &mut AdaptiveParticleSet,
    update: &[f32],
    mask: &[f32],
    dt: f32,
) -> AutomataResult<f32> {
    let spatial_dims = particles.spatial_dims;
    let state_dims = particles.state_dims;
    let output_dims = model.rule.config.update_dims();
    if update.len() != particles.len() * output_dims || mask.len() != particles.len() {
        return Err(AutomataError::InvalidArgument(
            "adaptive represented-measure integration shape mismatch".to_string(),
        ));
    }

    let physical_update = physical_update_field(model, particles, update);
    let has_coarse = (0..particles.len()).any(|row| is_coarse_material(model, particles, row));
    let update_gradient = if model.config.transport_coarse_moments && has_coarse {
        let mut perception = model.config.perception;
        perception.log_normalize_gradients = false;
        perception.include_position_features = false;
        Some(adaptive_perceive_without_spacing(
            &particles.positions,
            &physical_update,
            &particles.represented_measure,
            &particles.bandwidth,
            1,
            particles.len(),
            output_dims,
            perception,
        )?)
    } else {
        None
    };

    let mut displacement_sum = 0.0;
    for row in 0..particles.len() {
        if is_coarse_material(model, particles, row)
            && let Some(gradient) = &update_gradient
            && !gradient.moment_fallback[row]
        {
            let row_gradient = &gradient.state_gradient
                [row * output_dims * spatial_dims..(row + 1) * output_dims * spatial_dims];
            transport_row_moments(particles, row, row_gradient, mask[row] * dt)?;
        }

        let physical = &physical_update[row * output_dims..(row + 1) * output_dims];
        let mut displacement2 = 0.0;
        for (axis, &update) in physical.iter().enumerate().take(spatial_dims) {
            let displacement = mask[row] * dt * update;
            particles.positions[row][axis] = (particles.positions[row][axis] + displacement)
                .clamp(model.config.domain_min[axis], model.config.domain_max[axis]);
            displacement2 += displacement * displacement;
        }
        displacement_sum += displacement2.sqrt();
        for channel in 0..state_dims {
            particles.states[row * state_dims + channel] +=
                mask[row] * dt * physical[spatial_dims + channel];
        }
    }
    Ok(displacement_sum)
}

pub(crate) fn integrate_closure_mode_update(
    model: &AdaptiveNpaModel,
    particles: &mut AdaptiveParticleSet,
    update: &[f32],
    mask: &[f32],
    dt: f32,
) -> AutomataResult<()> {
    if !model.config.closure_recurrent_mode {
        return Ok(());
    }
    let output_dims = particles.spatial_dims + particles.state_dims;
    if update.len() != particles.len() * output_dims
        || mask.len() != particles.len()
        || !dt.is_finite()
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive closure-mode integration shape mismatch".to_owned(),
        ));
    }
    if particles.closure_mode.is_empty() {
        particles
            .closure_mode
            .resize(particles.len() * particles.state_dims, 0.0);
    }
    if particles.closure_phase.is_empty() {
        particles.closure_phase.resize(particles.len() * 2, 0.0);
    }
    for row in 0..particles.len() {
        if !is_coarse_material(model, particles, row) {
            continue;
        }
        for axis in 0..particles.spatial_dims {
            particles.closure_phase[row * 2 + axis] +=
                mask[row] * dt * update[row * output_dims + axis];
        }
        let phase = &mut particles.closure_phase[row * 2..(row + 1) * 2];
        let phase_norm = phase.iter().map(|value| value * value).sum::<f32>().sqrt();
        if phase_norm > 1.0e-6 {
            phase.iter_mut().for_each(|value| *value /= phase_norm);
        } else {
            phase.copy_from_slice(&[1.0, 0.0]);
        }
        for channel in 0..particles.state_dims {
            let index = row * particles.state_dims + channel;
            particles.closure_mode[index] +=
                mask[row] * dt * update[row * output_dims + particles.spatial_dims + channel];
        }
    }
    Ok(())
}

pub(crate) fn integrate_closure_basis_update(
    model: &AdaptiveNpaModel,
    particles: &mut AdaptiveParticleSet,
    update: &[f32],
    mask: &[f32],
    dt: f32,
) -> AutomataResult<()> {
    if !model.config.closure_recurrent_mode || model.closure_basis_rule.is_none() {
        return Ok(());
    }
    const BASIS_DIMS: usize = 4;
    let output_dims = particles.spatial_dims + particles.state_dims;
    if update.len() != particles.len() * output_dims
        || mask.len() != particles.len()
        || output_dims < BASIS_DIMS
        || !dt.is_finite()
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive closure-basis integration shape mismatch".to_owned(),
        ));
    }
    if particles.closure_basis.is_empty() {
        particles
            .closure_basis
            .resize(particles.len() * BASIS_DIMS, 0.0);
    }
    for row in 0..particles.len() {
        if !is_coarse_material(model, particles, row) {
            continue;
        }
        let basis = &mut particles.closure_basis[row * BASIS_DIMS..(row + 1) * BASIS_DIMS];
        let previous = [basis[0], basis[1], basis[2], basis[3]];
        for component in 0..BASIS_DIMS {
            basis[component] += mask[row] * dt * update[row * output_dims + component];
        }

        // First-level adaptive aggregates consist of four equal-measure
        // siblings. Their affine-null direction is orthogonal to the constant
        // weight vector and has unit norm. Project after every Euler update so
        // numerical drift cannot create a hidden fourth degree of freedom.
        let mean = basis.iter().sum::<f32>() / BASIS_DIMS as f32;
        basis.iter_mut().for_each(|value| *value -= mean);
        let norm = basis.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > 1.0e-6 {
            basis.iter_mut().for_each(|value| *value /= norm);
            let alignment = basis
                .iter()
                .zip(previous)
                .map(|(next, previous)| next * previous)
                .sum::<f32>();
            if alignment < 0.0 {
                basis.iter_mut().for_each(|value| *value = -*value);
            }
        } else {
            basis.copy_from_slice(&previous);
        }
    }
    Ok(())
}

fn is_coarse_material(
    model: &AdaptiveNpaModel,
    particles: &AdaptiveParticleSet,
    row: usize,
) -> bool {
    particles.footprint(row) > model.config.base_rule_footprint() * (1.0 + 32.0 * f32::EPSILON)
}

fn physical_update_field(
    model: &AdaptiveNpaModel,
    particles: &AdaptiveParticleSet,
    update: &[f32],
) -> Vec<f32> {
    let output_dims = model.rule.config.update_dims();
    let mut physical = update.to_vec();
    for row in 0..particles.len() {
        let output = &update[row * output_dims..(row + 1) * output_dims];
        let motion_norm = output[..particles.spatial_dims]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let motion_scale = model.rule.config.alpha
            * model.rule.config.motion_eps(particles.bandwidth[row])
            / (1.0 + motion_norm);
        for axis in 0..particles.spatial_dims {
            physical[row * output_dims + axis] *= motion_scale;
        }
    }
    physical
}

fn transport_row_moments(
    particles: &mut AdaptiveParticleSet,
    row: usize,
    update_gradient: &[f32],
    dt: f32,
) -> AutomataResult<()> {
    let dim = particles.spatial_dims;
    let state_dims = particles.state_dims;
    let output_dims = dim + state_dims;
    if update_gradient.len() != output_dims * dim {
        return Err(AutomataError::InvalidArgument(
            "adaptive moment transport gradient shape mismatch".to_string(),
        ));
    }

    let mut deformation = [0.0_f32; 9];
    for out_axis in 0..dim {
        for in_axis in 0..dim {
            deformation[out_axis * 3 + in_axis] =
                f32::from(out_axis == in_axis) + dt * update_gradient[out_axis * dim + in_axis];
        }
    }
    let Some(inverse) = invert_matrix(deformation, dim) else {
        return Ok(());
    };

    let previous_covariance = particles.covariance[row];
    let mut covariance = [0.0_f32; 9];
    for lhs in 0..dim {
        for rhs in 0..dim {
            for inner_lhs in 0..dim {
                for inner_rhs in 0..dim {
                    covariance[lhs * 3 + rhs] += deformation[lhs * 3 + inner_lhs]
                        * previous_covariance[inner_lhs * 3 + inner_rhs]
                        * deformation[rhs * 3 + inner_rhs];
                }
            }
        }
    }
    regularize_covariance(&mut covariance, dim);

    let jacobian_dims = state_dims * dim;
    let previous_jacobian =
        particles.state_jacobian[row * jacobian_dims..(row + 1) * jacobian_dims].to_vec();
    let mut jacobian = vec![0.0_f32; jacobian_dims];
    for channel in 0..state_dims {
        for out_axis in 0..dim {
            for inner in 0..dim {
                let material_gradient = previous_jacobian[channel * dim + inner]
                    + dt * update_gradient[(dim + channel) * dim + inner];
                jacobian[channel * dim + out_axis] +=
                    material_gradient * inverse[inner * 3 + out_axis];
            }
        }
    }
    if covariance.iter().all(|value| value.is_finite())
        && jacobian.iter().all(|value| value.is_finite())
    {
        particles.covariance[row] = covariance;
        particles.state_jacobian[row * jacobian_dims..(row + 1) * jacobian_dims]
            .copy_from_slice(&jacobian);
    }
    Ok(())
}

fn invert_matrix(matrix: [f32; 9], dim: usize) -> Option<[f32; 9]> {
    let mut augmented = [[0.0_f32; 6]; 3];
    for row in 0..dim {
        for col in 0..dim {
            augmented[row][col] = matrix[row * 3 + col];
        }
        augmented[row][dim + row] = 1.0;
    }
    for pivot in 0..dim {
        let selected = (pivot..dim).max_by(|lhs, rhs| {
            augmented[*lhs][pivot]
                .abs()
                .total_cmp(&augmented[*rhs][pivot].abs())
        })?;
        if augmented[selected][pivot].abs() <= 1.0e-6 {
            return None;
        }
        augmented.swap(pivot, selected);
        let inverse = augmented[pivot][pivot].recip();
        for value in augmented[pivot].iter_mut().take(2 * dim) {
            *value *= inverse;
        }
        let pivot_row = augmented[pivot];
        for (row, values) in augmented.iter_mut().enumerate().take(dim) {
            if row == pivot {
                continue;
            }
            let scale = values[pivot];
            for (value, &pivot_value) in values.iter_mut().zip(&pivot_row).take(2 * dim) {
                *value -= scale * pivot_value;
            }
        }
    }
    let mut inverse = [0.0_f32; 9];
    for row in 0..dim {
        for col in 0..dim {
            inverse[row * 3 + col] = augmented[row][dim + col];
        }
    }
    Some(inverse)
}

fn regularize_covariance(covariance: &mut [f32; 9], dim: usize) {
    for row in 0..dim {
        for col in row + 1..dim {
            let symmetric = 0.5 * (covariance[row * 3 + col] + covariance[col * 3 + row]);
            covariance[row * 3 + col] = symmetric;
            covariance[col * 3 + row] = symmetric;
        }
        covariance[row * 3 + row] = covariance[row * 3 + row].max(1.0e-10);
    }
    if dim == 2 {
        let determinant = covariance[0] * covariance[4] - covariance[1].powi(2);
        if determinant <= 1.0e-12 {
            let correction = (1.0e-12 - determinant).max(0.0).sqrt();
            covariance[0] += correction;
            covariance[4] += correction;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NpaConfig, NpaModel};

    fn particles() -> AdaptiveParticleSet {
        let mut particles = AdaptiveParticleSet::from_equal_measure(
            vec![[0.0; 4]],
            vec![0.0],
            2,
            1,
            std::f32::consts::PI,
            0.1,
        )
        .unwrap();
        particles.covariance[0] = [4.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0];
        particles.state_jacobian = vec![2.0, 4.0];
        particles
    }

    #[test]
    fn affine_transport_evolves_covariance_and_state_jacobian() {
        let mut particles = particles();
        // v = [x, -0.5y], ds/dt = 2x - y. Therefore F=diag(2, 0.5).
        transport_row_moments(&mut particles, 0, &[1.0, 0.0, 0.0, -0.5, 2.0, -1.0], 1.0).unwrap();
        assert!((particles.covariance[0][0] - 16.0).abs() <= 1.0e-6);
        assert!((particles.covariance[0][4] - 2.25).abs() <= 1.0e-6);
        assert!((particles.state_jacobian[0] - 2.0).abs() <= 1.0e-6);
        assert!((particles.state_jacobian[1] - 6.0).abs() <= 1.0e-6);
    }

    #[test]
    fn pair_merge_uses_expected_update_gate() {
        let rule = NpaModel::seeded(NpaConfig::growing_2d(), 3);
        let mut config = super::super::AdaptiveNpaConfig::growing_2d();
        config.reference_footprint = 1.0;
        config.base_rule_footprint = 1.0;
        config.min_footprint = 0.5;
        config.max_footprint = 2.0;
        config.min_leaves = 1;
        config.target_leaves = 1;
        config.max_leaves = 1;
        config.expected_coarse_update_mask = true;
        config.proxy.enabled = false;
        let model = AdaptiveNpaModel::seeded(rule, config, 5).unwrap();
        let mut particles = AdaptiveParticleSet::from_equal_measure(
            vec![[0.0; 4]],
            vec![0.0; model.rule.config.state_dims],
            2,
            model.rule.config.state_dims,
            2.0 * std::f32::consts::PI,
            0.1,
        )
        .unwrap();
        particles.render_footprint[0] = particles.footprint(0);

        assert!(is_coarse_material(&model, &particles, 0));
        assert_eq!(
            integration_masks(&model, &particles, 7, 11, 0.25),
            vec![0.25]
        );
    }
}
