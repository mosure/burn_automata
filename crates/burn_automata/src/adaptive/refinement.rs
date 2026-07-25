use super::{
    AdaptiveNpaModel, AdaptiveParticleSet, CanonicalMaterial, canonical_split,
    dynamics::adaptive_raw_update, material_footprint_radius,
};
use crate::AutomataResult;

/// Measures the one-level restriction/evolution defect of the adaptive NPA.
/// Each material leaf is canonically refined, evolved at child resolution, and
/// measure-restricted back before comparison with the parent-resolution update.
pub(super) fn adaptive_refinement_defect(
    model: &AdaptiveNpaModel,
    particles: &AdaptiveParticleSet,
) -> AutomataResult<Vec<f32>> {
    let coarse_update = adaptive_raw_update(model, particles)?;
    let (refined, child_parent) = canonical_refinement(particles)?;
    let refined_update = adaptive_raw_update(model, &refined)?;
    let output_dims = model.rule.config.update_dims();
    let spatial_dims = particles.spatial_dims;
    let mut restricted_dx = vec![0.0; particles.len() * spatial_dims];
    let mut restricted_ds = vec![0.0; particles.len() * particles.state_dims];
    for (child, parent) in child_parent.into_iter().enumerate() {
        let weight = refined.represented_measure[child]
            / particles.represented_measure[parent].max(f32::MIN_POSITIVE);
        let child_update = &refined_update[child * output_dims..(child + 1) * output_dims];
        let motion_norm = child_update[..spatial_dims]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let motion_scale = model.rule.config.alpha
            * model.rule.config.motion_eps(refined.bandwidth[child])
            / (1.0 + motion_norm);
        for axis in 0..spatial_dims {
            restricted_dx[parent * spatial_dims + axis] +=
                weight * motion_scale * child_update[axis];
        }
        for channel in 0..particles.state_dims {
            restricted_ds[parent * particles.state_dims + channel] +=
                weight * child_update[spatial_dims + channel];
        }
    }
    let restricted_update = raw_update_from_restricted_step(
        &restricted_dx,
        &restricted_ds,
        &particles.bandwidth,
        model,
    );
    Ok((0..particles.len())
        .map(|row| {
            ((0..output_dims)
                .map(|channel| {
                    let index = row * output_dims + channel;
                    (restricted_update[index] - coarse_update[index]).powi(2)
                })
                .sum::<f32>()
                / output_dims as f32)
                .sqrt()
        })
        .collect())
}

fn canonical_refinement(
    particles: &AdaptiveParticleSet,
) -> AutomataResult<(AdaptiveParticleSet, Vec<usize>)> {
    let children_per_parent = 2 * particles.spatial_dims;
    let refined_count = particles.len() * children_per_parent;
    let mut refined = AdaptiveParticleSet {
        spatial_dims: particles.spatial_dims,
        state_dims: particles.state_dims,
        positions: Vec::with_capacity(refined_count),
        states: Vec::with_capacity(refined_count * particles.state_dims),
        state_jacobian: Vec::with_capacity(
            refined_count * particles.state_dims * particles.spatial_dims,
        ),
        closure_mode: Vec::with_capacity(refined_count * particles.state_dims),
        closure_basis: Vec::with_capacity(refined_count * 4),
        closure_phase: Vec::with_capacity(refined_count * 2),
        represented_measure: Vec::with_capacity(refined_count),
        render_footprint: Vec::with_capacity(refined_count),
        bandwidth: Vec::with_capacity(refined_count),
        covariance: Vec::with_capacity(refined_count),
        particle_id: Vec::with_capacity(refined_count),
        sibling_group: Vec::with_capacity(refined_count),
        generation: Vec::with_capacity(refined_count),
        cooldown: vec![0; refined_count],
        next_id: refined_count as u64,
        next_sibling_group: particles.len() as u64 + 1,
        bootstrap_templates: Vec::new(),
    };
    let mut child_parent = Vec::with_capacity(refined_count);
    for parent in 0..particles.len() {
        let material = material_at(particles, parent);
        let state =
            &particles.states[parent * particles.state_dims..(parent + 1) * particles.state_dims];
        for child in canonical_split(&material)? {
            let mut position = [0.0; 4];
            for (axis, value) in child.position.iter().enumerate() {
                position[axis] = *value as f32;
            }
            let mut covariance = [0.0; 9];
            for row in 0..particles.spatial_dims {
                for col in 0..particles.spatial_dims {
                    covariance[row * 3 + col] =
                        child.covariance[row * particles.spatial_dims + col] as f32;
                }
            }
            refined.positions.push(position);
            refined.states.extend_from_slice(state);
            let jacobian_dims = particles.state_dims * particles.spatial_dims;
            refined.state_jacobian.extend_from_slice(
                &particles.state_jacobian[parent * jacobian_dims..(parent + 1) * jacobian_dims],
            );
            refined
                .closure_mode
                .extend(std::iter::repeat_n(0.0, particles.state_dims));
            refined.closure_basis.extend(std::iter::repeat_n(0.0, 4));
            refined.closure_phase.extend(std::iter::repeat_n(0.0, 2));
            refined
                .represented_measure
                .push(child.represented_measure as f32);
            refined.render_footprint.push(material_footprint_radius(
                child.represented_measure as f32,
                particles.spatial_dims,
            ));
            refined.bandwidth.push(particles.bandwidth[parent]);
            refined.covariance.push(covariance);
            refined.particle_id.push(refined.particle_id.len() as u64);
            refined.sibling_group.push(parent as u64 + 1);
            refined
                .generation
                .push(particles.generation[parent].saturating_add(1));
            child_parent.push(parent);
        }
    }
    refined.validate()?;
    Ok((refined, child_parent))
}

fn material_at(particles: &AdaptiveParticleSet, index: usize) -> CanonicalMaterial {
    let dim = particles.spatial_dims;
    CanonicalMaterial {
        represented_measure: particles.represented_measure[index] as f64,
        position: particles.positions[index][..dim]
            .iter()
            .map(|value| *value as f64)
            .collect(),
        covariance: (0..dim)
            .flat_map(|row| {
                (0..dim).map(move |col| particles.covariance[index][row * 3 + col] as f64)
            })
            .collect(),
        extensive: Vec::new(),
    }
}

fn raw_update_from_restricted_step(
    dx: &[f32],
    ds: &[f32],
    bandwidth: &[f32],
    model: &AdaptiveNpaModel,
) -> Vec<f32> {
    let spatial_dims = model.rule.config.spatial_dims;
    let state_dims = model.rule.config.state_dims;
    let output_dims = model.rule.config.update_dims();
    let mut output = vec![0.0; bandwidth.len() * output_dims];
    for row in 0..bandwidth.len() {
        let spatial = &dx[row * spatial_dims..(row + 1) * spatial_dims];
        let norm = spatial
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let scale = (model.rule.config.alpha * model.rule.config.motion_eps(bandwidth[row]))
            .max(f32::MIN_POSITIVE);
        let normalized = (norm / scale).clamp(0.0, 0.999);
        let raw_norm = normalized / (1.0 - normalized).max(1.0e-4);
        for axis in 0..spatial_dims {
            output[row * output_dims + axis] = if norm > 1.0e-12 {
                spatial[axis] * raw_norm / norm
            } else {
                0.0
            };
        }
        output[row * output_dims + spatial_dims..(row + 1) * output_dims]
            .copy_from_slice(&ds[row * state_dims..(row + 1) * state_dims]);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NpaConfig, NpaModel};

    #[test]
    fn canonical_refinement_preserves_material_and_parent_state() {
        let particles = AdaptiveParticleSet::from_equal_measure(
            vec![[0.0; 4], [0.2, 0.0, 0.0, 0.0]],
            (0..32).map(|value| value as f32).collect(),
            2,
            16,
            0.2,
            0.1,
        )
        .unwrap();
        let (refined, parent) = canonical_refinement(&particles).unwrap();
        assert_eq!(refined.len(), 8);
        assert_eq!(parent, vec![0, 0, 0, 0, 1, 1, 1, 1]);
        assert!((refined.total_measure() - particles.total_measure()).abs() < 1.0e-9);
        assert_eq!(&refined.states[..16], &particles.states[..16]);
        assert_eq!(&refined.states[64..80], &particles.states[16..32]);
    }

    #[test]
    fn refinement_defect_is_finite() {
        let base = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 7);
        let mut config = super::super::AdaptiveNpaConfig::growing_2d();
        config.min_leaves = 1;
        config.target_leaves = 8;
        config.max_leaves = 64;
        let model = AdaptiveNpaModel::seeded(base, config, 9).unwrap();
        let particles = AdaptiveParticleSet::from_equal_measure(
            (0..8)
                .map(|index| [index as f32 * 0.02 - 0.07, 0.0, 0.0, 0.0])
                .collect(),
            vec![0.1; 8 * 16],
            2,
            16,
            0.2,
            0.1,
        )
        .unwrap();
        let defect = adaptive_refinement_defect(&model, &particles).unwrap();
        assert_eq!(defect.len(), particles.len());
        assert!(
            defect
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        );
    }
}
