use rayon::prelude::*;

#[cfg(all(
    any(feature = "backend_cuda", feature = "backend_wgpu"),
    any(test, feature = "gpu_wgpu")
))]
use burn::tensor::{Tensor, TensorData, backend::Backend};
#[cfg(all(
    any(feature = "backend_cuda", feature = "backend_wgpu"),
    any(test, feature = "gpu_wgpu")
))]
use burn_automata_kernels::AdaptiveMergeCostCubeBackend;

#[cfg(all(
    any(feature = "backend_cuda", feature = "backend_wgpu"),
    any(test, feature = "gpu_wgpu")
))]
use super::material_footprint_radius;
use super::{
    AdaptiveHierarchyMember, AdaptiveParticleSet, AdaptiveProxyHierarchy, AdaptiveRenderDecoder,
    AdaptiveRestrictionLabelTarget,
};
#[cfg(all(
    any(feature = "backend_cuda", feature = "backend_wgpu"),
    any(test, feature = "gpu_wgpu")
))]
use crate::target2d::{
    AdaptiveCompactSplatPrimitive, adaptive_compact_splat_primitive,
    adaptive_isotropic_splat_primitive,
};
use crate::{
    AutomataError, AutomataResult,
    target2d::{
        Target2dLossConfig, Target2dRenderedSplat, TargetImage2d,
        render_adaptive_material_2d_compact_splat, render_adaptive_material_2d_isotropic_splat,
        render_target_2d_splat,
    },
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn target_render_merge_costs(
    fine: &AdaptiveParticleSet,
    target_leaves: usize,
    target: &TargetImage2d,
    render_config: Target2dLossConfig,
    fine_measure: f32,
    render_decoder: AdaptiveRenderDecoder,
    compactness: f32,
    label_target: AdaptiveRestrictionLabelTarget,
) -> AutomataResult<Vec<f32>> {
    fine.validate()?;
    if fine.spatial_dims != 2 {
        return Err(AutomataError::InvalidArgument(
            "target-render merge oracle currently supports 2D material".to_string(),
        ));
    }
    let hierarchy = AdaptiveProxyHierarchy::build(fine, 4)?;
    let level = hierarchy.levels.first().ok_or_else(|| {
        AutomataError::InvalidArgument(
            "target-render merge oracle requires first-level hierarchy nodes".to_string(),
        )
    })?;
    let reduction_per_merge = hierarchy.branch_factor - 1;
    let merge_count = fine
        .len()
        .checked_sub(target_leaves)
        .filter(|reduction| reduction.is_multiple_of(reduction_per_merge))
        .map(|reduction| reduction / reduction_per_merge)
        .ok_or_else(|| {
            AutomataError::InvalidArgument(format!(
                "target-render merge oracle cannot reach {target_leaves} leaves from {}",
                fine.len(),
            ))
        })?;
    if merge_count == level.len() {
        return Ok(vec![0.0; level.len()]);
    }

    let target_center = target.mean_position();
    let centered_fine = centered_positions(fine, target_center);
    let split_count = level.len().saturating_sub(merge_count);
    let rank_from_fine = merge_count <= split_count;
    let center_offset = [
        centered_fine[0][0] - fine.positions[0][0],
        centered_fine[0][1] - fine.positions[0][1],
    ];
    let baseline = if rank_from_fine {
        render_material_for_decoder(
            render_decoder,
            &centered_fine,
            &fine.states,
            fine.state_dims,
            &fine.represented_measure,
            &fine.covariance,
            fine_measure,
            target.pixel_size,
            render_config,
            None,
            compactness,
        )?
    } else {
        let parent_positions = level
            .iter()
            .map(|node| {
                let mut position = hierarchy.nodes[*node].position;
                position[0] += center_offset[0];
                position[1] += center_offset[1];
                position
            })
            .collect::<Vec<_>>();
        let parent_states = level
            .iter()
            .flat_map(|node| hierarchy.nodes[*node].state.iter().copied())
            .collect::<Vec<_>>();
        let parent_measure = level
            .iter()
            .map(|node| hierarchy.nodes[*node].represented_measure)
            .collect::<Vec<_>>();
        let parent_covariance = level
            .iter()
            .map(|node| hierarchy.nodes[*node].covariance)
            .collect::<Vec<_>>();
        render_material_for_decoder(
            render_decoder,
            &parent_positions,
            &parent_states,
            fine.state_dims,
            &parent_measure,
            &parent_covariance,
            fine_measure,
            target.pixel_size,
            render_config,
            None,
            compactness,
        )?
    };
    let comparison_render = match label_target {
        AdaptiveRestrictionLabelTarget::TargetImage => {
            render_target_2d_splat(target, render_config)?
        }
        AdaptiveRestrictionLabelTarget::FineTeacher => render_material_for_decoder(
            render_decoder,
            &centered_fine,
            &fine.states,
            fine.state_dims,
            &fine.represented_measure,
            &fine.covariance,
            fine_measure,
            target.pixel_size,
            render_config,
            None,
            compactness,
        )?,
    };

    level
        .par_iter()
        .map(|node_index| {
            let node = &hierarchy.nodes[*node_index];
            let children =
                hierarchy.member_leaf_indices(AdaptiveHierarchyMember::Proxy(*node_index));
            let child_positions = children
                .iter()
                .map(|index| {
                    let mut position = fine.positions[*index];
                    position[0] += center_offset[0];
                    position[1] += center_offset[1];
                    position
                })
                .collect::<Vec<_>>();
            let child_states = children
                .iter()
                .flat_map(|index| {
                    fine.states[*index * fine.state_dims..(*index + 1) * fine.state_dims]
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>();
            let child_measure = children
                .iter()
                .map(|index| fine.represented_measure[*index])
                .collect::<Vec<_>>();
            let child_covariance = children
                .iter()
                .map(|index| fine.covariance[*index])
                .collect::<Vec<_>>();
            let children_render = render_material_for_decoder(
                render_decoder,
                &child_positions,
                &child_states,
                fine.state_dims,
                &child_measure,
                &child_covariance,
                fine_measure,
                target.pixel_size,
                render_config,
                None,
                compactness,
            )?;
            let mut parent_position = node.position;
            parent_position[0] += center_offset[0];
            parent_position[1] += center_offset[1];
            let parent_render = render_material_for_decoder(
                render_decoder,
                &[parent_position],
                &node.state,
                fine.state_dims,
                &[node.represented_measure],
                &[node.covariance],
                fine_measure,
                target.pixel_size,
                render_config,
                None,
                compactness,
            )?;
            let candidate_mse = if rank_from_fine {
                replacement_composited_mse(
                    &baseline,
                    &children_render,
                    &parent_render,
                    &comparison_render,
                )
            } else {
                replacement_composited_mse(
                    &baseline,
                    &parent_render,
                    &children_render,
                    &comparison_render,
                )
            };
            // The hierarchy merges the lowest costs. From the coarse endpoint,
            // invert the split score so the best candidate splits remain fine.
            Ok(if rank_from_fine {
                candidate_mse
            } else {
                -candidate_mse
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn render_material_for_decoder(
    decoder: AdaptiveRenderDecoder,
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    represented_measure: &[f32],
    covariance: &[[f32; 9]],
    fine_measure: f32,
    base_pixel_size: f32,
    render_config: Target2dLossConfig,
    center: Option<[f32; 2]>,
    compactness: f32,
) -> AutomataResult<Target2dRenderedSplat> {
    match decoder {
        AdaptiveRenderDecoder::IsotropicMaterialGaussian => {
            render_adaptive_material_2d_isotropic_splat(
                positions,
                states,
                state_dims,
                represented_measure,
                fine_measure,
                base_pixel_size,
                render_config,
                center,
            )
        }
        AdaptiveRenderDecoder::CompactMomentGaussian => render_adaptive_material_2d_compact_splat(
            positions,
            states,
            state_dims,
            represented_measure,
            covariance,
            fine_measure,
            base_pixel_size,
            render_config,
            center,
            compactness,
        ),
        _ => Err(AutomataError::InvalidArgument(format!(
            "restriction labels do not support {decoder:?}; use isotropic-material-gaussian or the diagnostic compact-moment-gaussian control",
        ))),
    }
}

#[cfg(all(
    any(feature = "backend_cuda", feature = "backend_wgpu"),
    any(test, feature = "gpu_wgpu")
))]
#[allow(clippy::too_many_arguments)]
fn merge_splat_primitive(
    decoder: AdaptiveRenderDecoder,
    position: [f32; 2],
    color: [f32; 3],
    represented_measure: f32,
    covariance: [f32; 9],
    fine_footprint: f32,
    base_pixel_size: f32,
    render_config: Target2dLossConfig,
    compactness: f32,
) -> AutomataResult<AdaptiveCompactSplatPrimitive> {
    match decoder {
        AdaptiveRenderDecoder::IsotropicMaterialGaussian => adaptive_isotropic_splat_primitive(
            position,
            color,
            represented_measure,
            fine_footprint,
            base_pixel_size,
            render_config,
        ),
        AdaptiveRenderDecoder::CompactMomentGaussian => adaptive_compact_splat_primitive(
            position,
            color,
            represented_measure,
            covariance,
            fine_footprint,
            base_pixel_size,
            render_config,
            compactness,
        ),
        _ => Err(AutomataError::InvalidArgument(format!(
            "restriction merge primitive does not support {decoder:?}",
        ))),
    }
}

#[cfg(all(test, any(feature = "backend_cuda", feature = "backend_wgpu")))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn target_render_merge_costs_burn<B: Backend + AdaptiveMergeCostCubeBackend>(
    fine: &AdaptiveParticleSet,
    target_leaves: usize,
    target: &TargetImage2d,
    render_config: Target2dLossConfig,
    fine_measure: f32,
    render_decoder: AdaptiveRenderDecoder,
    compactness: f32,
    label_target: AdaptiveRestrictionLabelTarget,
    device: &B::Device,
) -> AutomataResult<Vec<f32>> {
    let hierarchy = AdaptiveProxyHierarchy::build(fine, 4)?;
    target_render_merge_costs_burn_with_hierarchy::<B>(
        fine,
        &hierarchy,
        target_leaves,
        target,
        render_config,
        fine_measure,
        render_decoder,
        compactness,
        label_target,
        device,
    )
}

#[cfg(all(test, any(feature = "backend_cuda", feature = "backend_wgpu")))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn target_render_merge_costs_burn_with_hierarchy<
    B: Backend + AdaptiveMergeCostCubeBackend,
>(
    fine: &AdaptiveParticleSet,
    hierarchy: &AdaptiveProxyHierarchy,
    target_leaves: usize,
    target: &TargetImage2d,
    render_config: Target2dLossConfig,
    fine_measure: f32,
    render_decoder: AdaptiveRenderDecoder,
    compactness: f32,
    label_target: AdaptiveRestrictionLabelTarget,
    device: &B::Device,
) -> AutomataResult<Vec<f32>> {
    let mut rows = target_render_merge_costs_burn_batch_with_hierarchies::<B>(
        &[(fine, hierarchy)],
        target_leaves,
        target,
        render_config,
        fine_measure,
        render_decoder,
        compactness,
        label_target,
        device,
    )?;
    rows.pop().ok_or_else(|| {
        AutomataError::InvalidModel("Burn merge-cost batch returned no rows".to_string())
    })
}

#[cfg(all(
    any(feature = "backend_cuda", feature = "backend_wgpu"),
    any(test, feature = "gpu_wgpu")
))]
struct PreparedMergeCostOracle {
    baseline: Target2dRenderedSplat,
    fine_teacher: Option<Target2dRenderedSplat>,
    primitives: Vec<AdaptiveCompactSplatPrimitive>,
    signs: Vec<f32>,
    groups: usize,
    rank_from_fine: bool,
}

#[cfg(all(
    any(feature = "backend_cuda", feature = "backend_wgpu"),
    any(test, feature = "gpu_wgpu")
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn target_render_merge_costs_burn_batch_with_hierarchies<
    B: Backend + AdaptiveMergeCostCubeBackend,
>(
    snapshots: &[(&AdaptiveParticleSet, &AdaptiveProxyHierarchy)],
    target_leaves: usize,
    target: &TargetImage2d,
    render_config: Target2dLossConfig,
    fine_measure: f32,
    render_decoder: AdaptiveRenderDecoder,
    compactness: f32,
    label_target: AdaptiveRestrictionLabelTarget,
    device: &B::Device,
) -> AutomataResult<Vec<Vec<f32>>> {
    if snapshots.is_empty() {
        return Err(AutomataError::InvalidArgument(
            "Burn target-render merge oracle requires at least one snapshot".to_string(),
        ));
    }
    let prepared = snapshots
        .iter()
        .map(|(fine, hierarchy)| {
            prepare_target_render_merge_cost(
                fine,
                hierarchy,
                target_leaves,
                target,
                render_config,
                fine_measure,
                render_decoder,
                compactness,
                label_target,
            )
        })
        .collect::<AutomataResult<Vec<_>>>()?;
    if prepared.iter().all(Option::is_none) {
        return snapshots
            .iter()
            .map(|(_, hierarchy)| {
                hierarchy
                    .levels
                    .first()
                    .map(|level| vec![0.0; level.len()])
                    .ok_or_else(|| {
                        AutomataError::InvalidArgument(
                            "Burn target-render merge oracle requires first-level hierarchy nodes"
                                .to_string(),
                        )
                    })
            })
            .collect();
    }
    if prepared.iter().any(Option::is_none) {
        return Err(AutomataError::InvalidArgument(
            "Burn merge-cost batch cannot mix trivial and scored hierarchy cuts".to_string(),
        ));
    }
    let prepared = prepared.into_iter().map(Option::unwrap).collect::<Vec<_>>();
    let target_render = render_target_2d_splat(target, render_config)?;
    let mut costs = merge_replacement_mse_burn_batch::<B>(
        &prepared,
        &target_render,
        label_target,
        render_config.image_size,
        device,
    )?;
    for (row, snapshot) in costs.iter_mut().zip(&prepared) {
        if !snapshot.rank_from_fine {
            row.iter_mut().for_each(|cost| *cost = -*cost);
        }
    }
    Ok(costs)
}

#[cfg(all(
    any(feature = "backend_cuda", feature = "backend_wgpu"),
    any(test, feature = "gpu_wgpu")
))]
#[allow(clippy::too_many_arguments)]
fn prepare_target_render_merge_cost(
    fine: &AdaptiveParticleSet,
    hierarchy: &AdaptiveProxyHierarchy,
    target_leaves: usize,
    target: &TargetImage2d,
    render_config: Target2dLossConfig,
    fine_measure: f32,
    render_decoder: AdaptiveRenderDecoder,
    compactness: f32,
    label_target: AdaptiveRestrictionLabelTarget,
) -> AutomataResult<Option<PreparedMergeCostOracle>> {
    fine.validate()?;
    if fine.spatial_dims != 2 {
        return Err(AutomataError::InvalidArgument(
            "Burn target-render merge oracle currently supports 2D material".to_string(),
        ));
    }
    let level = hierarchy.levels.first().ok_or_else(|| {
        AutomataError::InvalidArgument(
            "Burn target-render merge oracle requires first-level hierarchy nodes".to_string(),
        )
    })?;
    let reduction_per_merge = hierarchy.branch_factor - 1;
    let merge_count = fine
        .len()
        .checked_sub(target_leaves)
        .filter(|reduction| reduction.is_multiple_of(reduction_per_merge))
        .map(|reduction| reduction / reduction_per_merge)
        .ok_or_else(|| {
            AutomataError::InvalidArgument(format!(
                "Burn target-render merge oracle cannot reach {target_leaves} leaves from {}",
                fine.len(),
            ))
        })?;
    if merge_count == level.len() {
        return Ok(None);
    }

    let target_center = target.mean_position();
    let centered_fine = centered_positions(fine, target_center);
    let split_count = level.len().saturating_sub(merge_count);
    let rank_from_fine = merge_count <= split_count;
    let center_offset = [
        centered_fine[0][0] - fine.positions[0][0],
        centered_fine[0][1] - fine.positions[0][1],
    ];
    let baseline = if rank_from_fine {
        render_material_for_decoder(
            render_decoder,
            &centered_fine,
            &fine.states,
            fine.state_dims,
            &fine.represented_measure,
            &fine.covariance,
            fine_measure,
            target.pixel_size,
            render_config,
            None,
            compactness,
        )?
    } else {
        let parent_positions = level
            .iter()
            .map(|node| {
                let mut position = hierarchy.nodes[*node].position;
                position[0] += center_offset[0];
                position[1] += center_offset[1];
                position
            })
            .collect::<Vec<_>>();
        let parent_states = level
            .iter()
            .flat_map(|node| hierarchy.nodes[*node].state.iter().copied())
            .collect::<Vec<_>>();
        let parent_measure = level
            .iter()
            .map(|node| hierarchy.nodes[*node].represented_measure)
            .collect::<Vec<_>>();
        let parent_covariance = level
            .iter()
            .map(|node| hierarchy.nodes[*node].covariance)
            .collect::<Vec<_>>();
        render_material_for_decoder(
            render_decoder,
            &parent_positions,
            &parent_states,
            fine.state_dims,
            &parent_measure,
            &parent_covariance,
            fine_measure,
            target.pixel_size,
            render_config,
            None,
            compactness,
        )?
    };
    let fine_teacher =
        if label_target == AdaptiveRestrictionLabelTarget::FineTeacher && !rank_from_fine {
            Some(render_material_for_decoder(
                render_decoder,
                &centered_fine,
                &fine.states,
                fine.state_dims,
                &fine.represented_measure,
                &fine.covariance,
                fine_measure,
                target.pixel_size,
                render_config,
                None,
                compactness,
            )?)
        } else {
            None
        };
    let fine_footprint = material_footprint_radius(fine_measure, 2);
    let mut primitives = Vec::with_capacity(level.len() * 5);
    let mut signs = Vec::with_capacity(level.len() * 5);
    for node_index in level.iter().copied() {
        let node = &hierarchy.nodes[node_index];
        let children = hierarchy.member_leaf_indices(AdaptiveHierarchyMember::Proxy(node_index));
        if children.len() != 4 {
            return Err(AutomataError::InvalidModel(
                "Burn target-render merge oracle requires complete four-child groups".to_string(),
            ));
        }
        let child_sign = if rank_from_fine { -1.0 } else { 1.0 };
        let parent_sign = -child_sign;
        for &child in children {
            primitives.push(merge_splat_primitive(
                render_decoder,
                [
                    fine.positions[child][0] + center_offset[0],
                    fine.positions[child][1] + center_offset[1],
                ],
                tail_color(&fine.states, fine.state_dims, child),
                fine.represented_measure[child],
                fine.covariance[child],
                fine_footprint,
                target.pixel_size,
                render_config,
                compactness,
            )?);
            signs.push(child_sign);
        }
        primitives.push(merge_splat_primitive(
            render_decoder,
            [
                node.position[0] + center_offset[0],
                node.position[1] + center_offset[1],
            ],
            tail_color_row(&node.state),
            node.represented_measure,
            node.covariance,
            fine_footprint,
            target.pixel_size,
            render_config,
            compactness,
        )?);
        signs.push(parent_sign);
    }
    Ok(Some(PreparedMergeCostOracle {
        baseline,
        fine_teacher,
        primitives,
        signs,
        groups: level.len(),
        rank_from_fine,
    }))
}

#[allow(clippy::too_many_arguments)]
#[cfg(all(
    any(feature = "backend_cuda", feature = "backend_wgpu"),
    any(test, feature = "gpu_wgpu")
))]
fn merge_replacement_mse_burn_batch<B: Backend + AdaptiveMergeCostCubeBackend>(
    prepared: &[PreparedMergeCostOracle],
    target: &Target2dRenderedSplat,
    label_target: AdaptiveRestrictionLabelTarget,
    image_size: usize,
    device: &B::Device,
) -> AutomataResult<Vec<Vec<f32>>> {
    const PRIMITIVES: usize = 5;
    let batch = prepared.len();
    let groups = prepared.first().map_or(0, |snapshot| snapshot.groups);
    if batch == 0
        || groups == 0
        || prepared.iter().any(|snapshot| {
            snapshot.groups != groups
                || snapshot.primitives.len() != groups * PRIMITIVES
                || snapshot.signs.len() != snapshot.primitives.len()
        })
    {
        return Err(AutomataError::InvalidArgument(
            "Burn batched merge replacement primitive shape mismatch".to_string(),
        ));
    }
    let pixels = image_size * image_size;
    if prepared.iter().any(|snapshot| {
        snapshot.baseline.density.len() != pixels || snapshot.baseline.rgb.len() != pixels * 3
    }) || target.density.len() != pixels
        || target.rgb.len() != pixels * 3
    {
        return Err(AutomataError::InvalidArgument(
            "Burn merge replacement image shape mismatch".to_string(),
        ));
    }
    let pack_render = |render: &Target2dRenderedSplat| {
        let mut packed = Vec::with_capacity(pixels * 4);
        for pixel in 0..pixels {
            packed.push(render.density[pixel]);
            packed.extend_from_slice(&render.rgb[pixel * 3..pixel * 3 + 3]);
        }
        packed
    };
    let mut packed_baselines = Vec::with_capacity(batch * pixels * 4);
    let mut packed_targets = Vec::with_capacity(batch * pixels * 4);
    let mut packed_primitives = Vec::with_capacity(batch * groups * PRIMITIVES * 10);
    for snapshot in prepared {
        packed_baselines.extend(pack_render(&snapshot.baseline));
        packed_targets.extend(pack_render(match label_target {
            AdaptiveRestrictionLabelTarget::TargetImage => target,
            AdaptiveRestrictionLabelTarget::FineTeacher if snapshot.rank_from_fine => {
                &snapshot.baseline
            }
            AdaptiveRestrictionLabelTarget::FineTeacher => snapshot
                .fine_teacher
                .as_ref()
                .expect("coarse-endpoint teacher render prepared"),
        }));
        for (primitive, sign) in snapshot.primitives.iter().zip(&snapshot.signs) {
            packed_primitives.extend_from_slice(&[
                primitive.center_pixel[0],
                primitive.center_pixel[1],
                primitive.inverse_sigma_squared[0],
                primitive.inverse_sigma_squared[1],
                primitive.sin,
                primitive.cos,
                primitive.weight_scale * sign,
                primitive.color[0],
                primitive.color[1],
                primitive.color[2],
            ]);
        }
    }
    let baseline = Tensor::<B, 3>::from_data(
        TensorData::new(packed_baselines, [batch, pixels, 4]),
        device,
    );
    let target =
        Tensor::<B, 3>::from_data(TensorData::new(packed_targets, [batch, pixels, 4]), device);
    let primitives = Tensor::<B, 4>::from_data(
        TensorData::new(packed_primitives, [batch, groups, PRIMITIVES, 10]),
        device,
    );
    let flat = B::adaptive_merge_cost_cube(baseline, target, primitives, image_size)
        .ok_or_else(|| {
            AutomataError::InvalidArgument(
                "adaptive merge-cost CubeCL kernel is unavailable for this backend".to_string(),
            )
        })?
        .map_err(|error| AutomataError::InvalidArgument(error.to_string()))?
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| AutomataError::InvalidArgument(error.to_string()))?;
    if flat.len() != batch * groups {
        return Err(AutomataError::InvalidModel(format!(
            "Burn merge-cost kernel returned {} values, expected {}",
            flat.len(),
            batch * groups,
        )));
    }
    Ok(flat.chunks_exact(groups).map(<[f32]>::to_vec).collect())
}

#[cfg(all(
    any(feature = "backend_cuda", feature = "backend_wgpu"),
    any(test, feature = "gpu_wgpu")
))]
fn tail_color(states: &[f32], state_dims: usize, row: usize) -> [f32; 3] {
    tail_color_row(&states[row * state_dims..(row + 1) * state_dims])
}

#[cfg(all(
    any(feature = "backend_cuda", feature = "backend_wgpu"),
    any(test, feature = "gpu_wgpu")
))]
fn tail_color_row(state: &[f32]) -> [f32; 3] {
    let dims = state.len();
    [
        state[dims - 3] + 0.5,
        state[dims - 2] + 0.5,
        state[dims - 1] + 0.5,
    ]
}

fn centered_positions(particles: &AdaptiveParticleSet, target: [f32; 2]) -> Vec<[f32; 4]> {
    let total = particles
        .represented_measure
        .iter()
        .sum::<f32>()
        .max(f32::MIN_POSITIVE);
    let mut mean = [0.0_f32; 2];
    for (position, measure) in particles
        .positions
        .iter()
        .zip(&particles.represented_measure)
    {
        mean[0] += position[0] * measure / total;
        mean[1] += position[1] * measure / total;
    }
    particles
        .positions
        .iter()
        .map(|position| {
            let mut centered = *position;
            centered[0] += target[0] - mean[0];
            centered[1] += target[1] - mean[1];
            centered
        })
        .collect()
}

fn replacement_composited_mse(
    baseline: &Target2dRenderedSplat,
    removed: &Target2dRenderedSplat,
    added: &Target2dRenderedSplat,
    target: &Target2dRenderedSplat,
) -> f32 {
    let pixels = baseline
        .density
        .len()
        .min(removed.density.len())
        .min(added.density.len())
        .min(target.density.len());
    let mut squared_error = 0.0_f32;
    for pixel in 0..pixels {
        let density = (baseline.density[pixel] - removed.density[pixel] + added.density[pixel])
            .clamp(0.0, 1.0);
        let target_density = target.density[pixel].clamp(0.0, 1.0);
        for channel in 0..3 {
            let index = pixel * 3 + channel;
            let value = (baseline.rgb[index] - removed.rgb[index] + added.rgb[index] + 1.0
                - density)
                .clamp(0.0, 1.0);
            let target_value = (target.rgb[index] + 1.0 - target_density).clamp(0.0, 1.0);
            squared_error += (value - target_value).powi(2);
        }
    }
    squared_error / (pixels * 3).max(1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacement_mse_is_zero_for_an_exact_replacement() {
        let baseline = Target2dRenderedSplat {
            rgb: vec![0.2, 0.3, 0.4],
            density: vec![0.5],
        };
        let removed = Target2dRenderedSplat {
            rgb: vec![0.1, 0.1, 0.1],
            density: vec![0.2],
        };
        let target = baseline.clone();

        assert_eq!(
            replacement_composited_mse(&baseline, &removed, &removed, &target),
            0.0
        );
    }

    #[cfg(any(feature = "backend_cuda", feature = "backend_wgpu"))]
    fn assert_cube_merge_costs_match_cpu_reference<B>()
    where
        B: Backend + AdaptiveMergeCostCubeBackend,
        B::Device: Default,
    {
        use crate::{NpaConfig, NpaModel, ParticleSeed, TargetImage2dExtractConfig};

        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let fine_count = 64;
        let fine_measure = total_measure / fine_count as f32;
        let footprint = material_footprint_radius(fine_measure, 2);
        let mut adaptive = crate::adaptive::AdaptiveNpaConfig::growing_2d();
        adaptive.reference_footprint = footprint;
        adaptive.base_rule_footprint = footprint;
        adaptive.min_footprint = 0.5 * footprint;
        adaptive.max_footprint = 2.0 * footprint;
        adaptive.min_leaves = 16;
        adaptive.target_leaves = 58;
        adaptive.max_leaves = fine_count;
        adaptive.initial_leaves = fine_count;
        adaptive.bootstrap_fine_leaves = fine_count;
        let model = crate::adaptive::AdaptiveNpaModel::seeded(
            NpaModel::seeded(NpaConfig::growing_2d(), 7),
            adaptive,
            11,
        )
        .unwrap();
        let mut particles = crate::adaptive::seed_adaptive_particles_scaled(
            &model,
            fine_count,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            0.1,
        )
        .unwrap();
        for row in 0..particles.len() {
            let base = row * particles.state_dims + particles.state_dims - 3;
            particles.states[base] = 0.4 * particles.positions[row][0];
            particles.states[base + 1] = 0.4 * particles.positions[row][1];
            particles.states[base + 2] = 0.2 * (row as f32 / fine_count as f32 - 0.5);
        }
        let mut rgba = vec![0.0_f32; 8 * 8 * 4];
        for y in 1..7 {
            for x in 1..7 {
                let base = (y * 8 + x) * 4;
                rgba[base] = x as f32 / 7.0;
                rgba[base + 1] = y as f32 / 7.0;
                rgba[base + 2] = 0.5;
                rgba[base + 3] = 1.0;
            }
        }
        let target =
            TargetImage2d::from_rgba_pixels(8, 8, &rgba, TargetImage2dExtractConfig::default())
                .unwrap();
        let render_config = Target2dLossConfig {
            image_size: 16,
            ..Target2dLossConfig::default()
        };
        let hierarchy = AdaptiveProxyHierarchy::build(&particles, 4).unwrap();
        let mut particles_b = particles.clone();
        for row in 0..particles_b.len() {
            particles_b.positions[row][0] += 0.015 * (row as f32 * 0.7).sin();
            particles_b.positions[row][1] += 0.01 * (row as f32 * 0.4).cos();
            let base = row * particles_b.state_dims + particles_b.state_dims - 3;
            particles_b.states[base] += 0.1;
            particles_b.states[base + 2] -= 0.05;
        }
        let hierarchy_b = AdaptiveProxyHierarchy::build(&particles_b, 4).unwrap();
        for decoder in [
            AdaptiveRenderDecoder::IsotropicMaterialGaussian,
            AdaptiveRenderDecoder::CompactMomentGaussian,
        ] {
            for label_target in [
                AdaptiveRestrictionLabelTarget::FineTeacher,
                AdaptiveRestrictionLabelTarget::TargetImage,
            ] {
                let expected = target_render_merge_costs(
                    &particles,
                    58,
                    &target,
                    render_config,
                    fine_measure,
                    decoder,
                    1.0,
                    label_target,
                )
                .unwrap();
                let actual = target_render_merge_costs_burn::<B>(
                    &particles,
                    58,
                    &target,
                    render_config,
                    fine_measure,
                    decoder,
                    1.0,
                    label_target,
                    &Default::default(),
                )
                .unwrap();
                let max_error = expected
                    .iter()
                    .zip(&actual)
                    .map(|(expected, actual)| (expected - actual).abs())
                    .fold(0.0_f32, f32::max);
                assert!(
                    max_error < 2.0e-6,
                    "{decoder:?}/{label_target:?} max merge-cost error {max_error}"
                );
                assert_eq!(
                    hierarchy
                        .level_one_merge_mask(&particles, 58, &expected)
                        .unwrap(),
                    hierarchy
                        .level_one_merge_mask(&particles, 58, &actual)
                        .unwrap(),
                );

                let expected_b = target_render_merge_costs(
                    &particles_b,
                    58,
                    &target,
                    render_config,
                    fine_measure,
                    decoder,
                    1.0,
                    label_target,
                )
                .unwrap();
                let rows = target_render_merge_costs_burn_batch_with_hierarchies::<B>(
                    &[(&particles, &hierarchy), (&particles_b, &hierarchy_b)],
                    58,
                    &target,
                    render_config,
                    fine_measure,
                    decoder,
                    1.0,
                    label_target,
                    &Default::default(),
                )
                .unwrap();
                assert_eq!(rows.len(), 2);
                for (row, expected) in rows.iter().zip([&expected, &expected_b]) {
                    let max_error = expected
                        .iter()
                        .zip(row)
                        .map(|(expected, actual)| (expected - actual).abs())
                        .fold(0.0_f32, f32::max);
                    assert!(
                        max_error < 2.0e-6,
                        "{decoder:?}/{label_target:?} batched merge-cost error {max_error}"
                    );
                }
            }
        }
    }

    #[cfg(feature = "backend_cuda")]
    #[test]
    fn cuda_merge_costs_match_cpu_reference() {
        assert_cube_merge_costs_match_cpu_reference::<burn::backend::Cuda<f32>>();
    }

    #[cfg(feature = "backend_wgpu")]
    #[test]
    #[ignore = "requires a WGPU device"]
    fn wgpu_merge_costs_match_cpu_reference() {
        assert_cube_merge_costs_match_cpu_reference::<burn::backend::Wgpu<f32>>();
    }
}
