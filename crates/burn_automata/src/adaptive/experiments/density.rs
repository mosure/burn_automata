use crate::{AutomataResult, adaptive::AdaptiveParticleSet};

use super::super::{
    AdaptiveNpaModel, allocate_resolution_budget, features::controller_features,
    normalize_footprint_budget, refinement::adaptive_refinement_defect,
};

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct DensityAlignment {
    pub state_detail_correlation: f32,
    pub high_to_low_state_detail_footprint_ratio: f32,
    pub refinement_defect_correlation: f32,
    pub low_to_high_refinement_defect_footprint_ratio: f32,
    pub mean_refinement_defect: f32,
    pub controller_oracle_scale_correlation: f32,
    pub oracle_min_desired_ratio: f32,
    pub oracle_max_desired_ratio: f32,
    pub controller_min_desired_ratio: f32,
    pub controller_max_desired_ratio: f32,
}

pub(super) fn task_density_alignment(
    model: &AdaptiveNpaModel,
    particles: &AdaptiveParticleSet,
) -> AutomataResult<DensityAlignment> {
    let perception_pair =
        super::super::perception::rule_perception_pair(&model.config, &model.rule, particles)?;
    let perception = &perception_pair.normalized;
    let state_detail = super::super::features::state_variation(particles, perception);
    let refinement_defect = adaptive_refinement_defect(model, particles)?;
    let inverse_footprint = (0..particles.len())
        .map(|index| {
            (model.config.reference_footprint / particles.footprint(index))
                .max(f32::MIN_POSITIVE)
                .ln()
        })
        .collect::<Vec<_>>();
    let state_footprint_ratio = detail_footprint_ratio(particles, &state_detail, false);
    let defect_footprint_ratio = detail_footprint_ratio(particles, &refinement_defect, true);
    let oracle = allocate_resolution_budget(
        &refinement_defect,
        &particles.represented_measure,
        particles.spatial_dims,
        2.0,
        model.config.reference_footprint,
        model.config.min_footprint,
        model.config.max_footprint,
        model.config.target_leaves,
    )?;
    let primary_features = super::super::dynamics::primary_rule_features(
        model,
        particles,
        &perception_pair.npa_compatible.features,
    )?;
    let base_update = model
        .rule
        .forward_update_from_features(primary_features.as_ref())?;
    let controller = model.controller.forward(&controller_features(
        &model.config,
        particles,
        perception,
        &base_update,
    ));
    let proposed = controller
        .iter()
        .map(|output| {
            (model.config.reference_footprint * output.desired_log_footprint.exp())
                .clamp(model.config.min_footprint, model.config.max_footprint)
        })
        .collect::<Vec<_>>();
    let controller_desired = normalize_footprint_budget(
        &proposed,
        &particles.represented_measure,
        particles.spatial_dims,
        model.config.min_footprint,
        model.config.max_footprint,
        model.config.target_leaves,
    )?
    .desired_footprint;
    let oracle_log_scale = oracle
        .desired_footprint
        .iter()
        .map(|value| (value / model.config.reference_footprint).ln())
        .collect::<Vec<_>>();
    let controller_log_scale = controller_desired
        .iter()
        .map(|value| (value / model.config.reference_footprint).ln())
        .collect::<Vec<_>>();
    let ratio_range = |desired: &[f32]| {
        desired
            .iter()
            .enumerate()
            .map(|(index, desired)| desired / particles.footprint(index))
            .fold(
                (f32::INFINITY, f32::NEG_INFINITY),
                |(minimum, maximum), ratio| (minimum.min(ratio), maximum.max(ratio)),
            )
    };
    let (oracle_min_desired_ratio, oracle_max_desired_ratio) =
        ratio_range(&oracle.desired_footprint);
    let (controller_min_desired_ratio, controller_max_desired_ratio) =
        ratio_range(&controller_desired);
    let total_measure = particles
        .represented_measure
        .iter()
        .sum::<f32>()
        .max(f32::MIN_POSITIVE);
    let mean_refinement_defect = refinement_defect
        .iter()
        .zip(&particles.represented_measure)
        .map(|(defect, measure)| defect * measure)
        .sum::<f32>()
        / total_measure;
    Ok(DensityAlignment {
        state_detail_correlation: pearson_correlation(&state_detail, &inverse_footprint),
        high_to_low_state_detail_footprint_ratio: state_footprint_ratio,
        refinement_defect_correlation: pearson_correlation(&refinement_defect, &inverse_footprint),
        low_to_high_refinement_defect_footprint_ratio: defect_footprint_ratio,
        mean_refinement_defect,
        controller_oracle_scale_correlation: pearson_correlation(
            &oracle_log_scale,
            &controller_log_scale,
        ),
        oracle_min_desired_ratio,
        oracle_max_desired_ratio,
        controller_min_desired_ratio,
        controller_max_desired_ratio,
    })
}

fn detail_footprint_ratio(
    particles: &AdaptiveParticleSet,
    detail: &[f32],
    low_over_high: bool,
) -> f32 {
    let mut order = (0..particles.len()).collect::<Vec<_>>();
    order.sort_unstable_by(|lhs, rhs| detail[*lhs].total_cmp(&detail[*rhs]));
    let quartile = (order.len() / 4).max(1);
    let low = order[..quartile]
        .iter()
        .map(|index| particles.footprint(*index))
        .sum::<f32>()
        / quartile as f32;
    let high = order[order.len() - quartile..]
        .iter()
        .map(|index| particles.footprint(*index))
        .sum::<f32>()
        / quartile as f32;
    if low_over_high {
        low / high.max(f32::MIN_POSITIVE)
    } else {
        high / low.max(f32::MIN_POSITIVE)
    }
}

fn pearson_correlation(lhs: &[f32], rhs: &[f32]) -> f32 {
    if lhs.len() != rhs.len() || lhs.is_empty() {
        return 0.0;
    }
    let lhs_mean = lhs.iter().sum::<f32>() / lhs.len() as f32;
    let rhs_mean = rhs.iter().sum::<f32>() / rhs.len() as f32;
    let (covariance, lhs_variance, rhs_variance) = lhs.iter().zip(rhs).fold(
        (0.0_f32, 0.0_f32, 0.0_f32),
        |(covariance, lhs_variance, rhs_variance), (lhs, rhs)| {
            let lhs_delta = *lhs - lhs_mean;
            let rhs_delta = *rhs - rhs_mean;
            (
                covariance + lhs_delta * rhs_delta,
                lhs_variance + lhs_delta.powi(2),
                rhs_variance + rhs_delta.powi(2),
            )
        },
    );
    covariance / (lhs_variance * rhs_variance).sqrt().max(f32::MIN_POSITIVE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NpaConfig, NpaModel};

    #[test]
    fn refinement_defect_audit_is_finite() {
        let base = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 7);
        let mut config = super::super::super::AdaptiveNpaConfig::growing_2d();
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
        let report = task_density_alignment(&model, &particles).unwrap();
        assert!(report.mean_refinement_defect.is_finite());
        assert!(report.mean_refinement_defect >= 0.0);
    }

    #[test]
    fn refinement_defect_audit_supports_material_scale_conditioning() {
        let base = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 7);
        let mut config = super::super::super::AdaptiveNpaConfig::growing_2d();
        config.coarse_dynamics = super::super::super::AdaptiveCoarseDynamics::RepresentedMeasure;
        config.min_leaves = 1;
        config.target_leaves = 8;
        config.max_leaves = 64;
        let mut model = AdaptiveNpaModel::seeded(base, config, 9).unwrap();
        model.enable_material_scale_conditioning().unwrap();
        let mut particles = AdaptiveParticleSet::from_equal_measure(
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
        particles.represented_measure[0] *= 4.0;

        let report = task_density_alignment(&model, &particles).unwrap();

        assert!(report.mean_refinement_defect.is_finite());
        assert!(report.controller_oracle_scale_correlation.is_finite());
    }
}
