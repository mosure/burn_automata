use serde::{Deserialize, Serialize};

use super::{AdaptiveMaterialView, AdaptiveParticleSet, AdaptiveProxyHierarchy};
use crate::{AutomataError, AutomataResult};

/// Compact state left after weighted mean and affine state-position moments.
/// The current 2D first-level hierarchy has at most one such mode per material
/// row. Values are laid out `[material, state_channel]`; inactive rows are
/// exactly zero.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveClosureModeField {
    pub values: Vec<f32>,
    #[serde(default)]
    pub basis: Vec<f32>,
    #[serde(default)]
    pub phase: Vec<f32>,
    pub active: Vec<bool>,
    pub state_dims: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveClosureReconstructionMetrics {
    pub active_rows: usize,
    pub unresolved_modes: usize,
    pub maximum_unresolved_modes_per_row: usize,
    pub reconstructed_state_values: usize,
    pub affine_root_mean_square_error: f32,
    pub augmented_root_mean_square_error: f32,
    pub maximum_augmented_absolute_error: f32,
}

pub(crate) fn restrict_first_closure_mode(
    fine: &AdaptiveParticleSet,
    hierarchy: &AdaptiveProxyHierarchy,
    view: &AdaptiveMaterialView,
) -> AutomataResult<(
    AdaptiveClosureModeField,
    AdaptiveClosureReconstructionMetrics,
)> {
    if view.members.len() != view.particles.len()
        || fine.spatial_dims != view.particles.spatial_dims
        || fine.state_dims != view.particles.state_dims
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive closure restriction shape mismatch".to_owned(),
        ));
    }
    restrict_first_closure_mode_for_members(fine, hierarchy, &view.members)
}

pub(crate) fn restrict_first_closure_mode_for_members(
    fine: &AdaptiveParticleSet,
    hierarchy: &AdaptiveProxyHierarchy,
    members: &[super::AdaptiveHierarchyMember],
) -> AutomataResult<(
    AdaptiveClosureModeField,
    AdaptiveClosureReconstructionMetrics,
)> {
    restrict_first_closure_mode_for_members_oriented(fine, hierarchy, members, None)
}

pub(crate) fn restrict_first_closure_mode_for_members_oriented(
    fine: &AdaptiveParticleSet,
    hierarchy: &AdaptiveProxyHierarchy,
    members: &[super::AdaptiveHierarchyMember],
    basis_anchors: Option<&[f32]>,
) -> AutomataResult<(
    AdaptiveClosureModeField,
    AdaptiveClosureReconstructionMetrics,
)> {
    if basis_anchors.is_some_and(|anchors| anchors.len() != members.len() * 4) {
        return Err(AutomataError::InvalidArgument(
            "adaptive closure basis-anchor shape mismatch".to_owned(),
        ));
    }
    let mut field = AdaptiveClosureModeField {
        values: vec![0.0; members.len() * fine.state_dims],
        basis: vec![0.0; members.len() * 4],
        phase: vec![0.0; members.len() * 2],
        active: vec![false; members.len()],
        state_dims: fine.state_dims,
    };
    let mut affine_square_sum = 0.0_f64;
    let mut augmented_square_sum = 0.0_f64;
    let mut maximum_augmented_error = 0.0_f32;
    let mut reconstructed_values = 0;
    let mut unresolved_modes = 0;
    let mut maximum_unresolved_modes = 0;

    for (material, member) in members.iter().copied().enumerate() {
        let leaves = hierarchy.member_leaf_indices(member);
        let anchor = basis_anchors.map(|anchors| &anchors[material * 4..(material + 1) * 4]);
        let Some(system) = AffineClosureSystem::new(fine, leaves, anchor) else {
            continue;
        };
        let remaining_modes = leaves.len().saturating_sub(system.rank);
        if remaining_modes == 0 {
            continue;
        }
        if remaining_modes != 1 {
            return Err(AutomataError::InvalidArgument(format!(
                "compact closure supports one unresolved mode per row, found {remaining_modes} for {} leaves",
                leaves.len(),
            )));
        }
        field.active[material] = true;
        let phase = closure_geometry_phase(fine, leaves, &system).ok_or_else(|| {
            AutomataError::InvalidModel("adaptive closure geometry phase is degenerate".to_owned())
        })?;
        field.phase[material * 2..(material + 1) * 2].copy_from_slice(&phase);
        for (output, value) in field.basis[material * 4..(material + 1) * 4]
            .iter_mut()
            .zip(&system.null_direction)
        {
            *output = *value as f32;
        }
        unresolved_modes += remaining_modes;
        maximum_unresolved_modes = maximum_unresolved_modes.max(remaining_modes);
        for channel in 0..fine.state_dims {
            let actual = leaves
                .iter()
                .map(|leaf| fine.states[*leaf * fine.state_dims + channel] as f64)
                .collect::<Vec<_>>();
            let constraints = system.constraint_targets(&actual)?;
            let affine = system.minimum_norm_solution(&constraints)?;
            let coefficient = actual
                .iter()
                .zip(&affine)
                .zip(&system.null_direction)
                .map(|((actual, affine), direction)| (actual - affine) * direction)
                .sum::<f64>();
            field.values[material * fine.state_dims + channel] = coefficient as f32;
            for ((actual, affine), direction) in
                actual.iter().zip(&affine).zip(&system.null_direction)
            {
                let affine_error = actual - affine;
                let augmented_error = actual - (affine + coefficient * direction);
                affine_square_sum += affine_error * affine_error;
                augmented_square_sum += augmented_error * augmented_error;
                maximum_augmented_error = maximum_augmented_error.max(augmented_error.abs() as f32);
                reconstructed_values += 1;
            }
        }
    }

    let divisor = reconstructed_values.max(1) as f64;
    Ok((
        field,
        AdaptiveClosureReconstructionMetrics {
            active_rows: unresolved_modes,
            unresolved_modes,
            maximum_unresolved_modes_per_row: maximum_unresolved_modes,
            reconstructed_state_values: reconstructed_values,
            affine_root_mean_square_error: (affine_square_sum / divisor).sqrt() as f32,
            augmented_root_mean_square_error: (augmented_square_sum / divisor).sqrt() as f32,
            maximum_augmented_absolute_error: maximum_augmented_error,
        },
    ))
}

pub(crate) fn attach_first_closure_mode(
    fine: &AdaptiveParticleSet,
    hierarchy: &AdaptiveProxyHierarchy,
    view: &mut AdaptiveMaterialView,
) -> AutomataResult<AdaptiveClosureReconstructionMetrics> {
    let (field, metrics) = restrict_first_closure_mode(fine, hierarchy, view)?;
    view.particles.closure_mode = field.values;
    view.particles.closure_basis = field.basis;
    view.particles.closure_phase = field.phase;
    view.particles.validate()?;
    Ok(metrics)
}

/// Reconstructs first-level fine child state from the material mean, affine
/// state-position moment, and compact affine-null coefficient. This makes the
/// recurrent closure state causally control coupled-fine oracle queries.
pub(crate) fn reconstruct_first_closure_state_for_members(
    fine: &mut AdaptiveParticleSet,
    hierarchy: &AdaptiveProxyHierarchy,
    members: &[super::AdaptiveHierarchyMember],
    material: &AdaptiveParticleSet,
) -> AutomataResult<()> {
    if members.len() != material.len()
        || fine.spatial_dims != material.spatial_dims
        || fine.state_dims != material.state_dims
        || material.closure_mode.len() != material.len() * material.state_dims
        || material.closure_basis.len() != material.len() * 4
        || material.closure_phase.len() != material.len() * 2
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive closure reconstruction shape mismatch".to_owned(),
        ));
    }
    reconstruct_first_closure_geometry_for_members(fine, hierarchy, members, material)?;
    let jacobian_dims = material.state_dims * material.spatial_dims;
    for (row, member) in members.iter().copied().enumerate() {
        let leaves = hierarchy.member_leaf_indices(member);
        if leaves.len() == 1 {
            let leaf = leaves[0];
            fine.states[leaf * fine.state_dims..(leaf + 1) * fine.state_dims].copy_from_slice(
                &material.states[row * material.state_dims..(row + 1) * material.state_dims],
            );
            continue;
        }
        let anchor = (!material.closure_basis.is_empty())
            .then(|| &material.closure_basis[row * 4..(row + 1) * 4]);
        let system = AffineClosureSystem::new(fine, leaves, anchor).ok_or_else(|| {
            AutomataError::InvalidArgument(
                "adaptive closure reconstruction has a rank-deficient aggregate".to_owned(),
            )
        })?;
        let remaining_modes = leaves.len().saturating_sub(system.rank);
        if remaining_modes != 1 {
            return Err(AutomataError::InvalidArgument(format!(
                "compact closure reconstruction supports one unresolved mode per row, found {remaining_modes}",
            )));
        }
        let covariance = material.covariance[row];
        let jacobian = &material.state_jacobian[row * jacobian_dims..(row + 1) * jacobian_dims];
        for channel in 0..fine.state_dims {
            let mut constraints = vec![0.0_f64; system.rank];
            constraints[0] = material.states[row * material.state_dims + channel] as f64;
            for axis in 0..fine.spatial_dims {
                constraints[axis + 1] = (0..fine.spatial_dims)
                    .map(|inner| {
                        covariance[axis * 3 + inner] as f64
                            * jacobian[channel * fine.spatial_dims + inner] as f64
                    })
                    .sum();
            }
            let affine = system.minimum_norm_solution(&constraints)?;
            let mode = material.closure_mode[row * material.state_dims + channel] as f64;
            for ((leaf, affine), direction) in leaves.iter().zip(affine).zip(&system.null_direction)
            {
                fine.states[*leaf * fine.state_dims + channel] = (affine + mode * direction) as f32;
            }
        }
    }
    fine.validate()
}

/// Reconstructs the four-child 2D geometry represented by a material row's
/// centroid, covariance, affine-null basis, and in-plane phase. The basis
/// orientation carries the discrete reflection and the phase carries the
/// continuous rotation, making both serialized closure fields causal.
pub(crate) fn reconstruct_first_closure_geometry_for_members(
    fine: &mut AdaptiveParticleSet,
    hierarchy: &AdaptiveProxyHierarchy,
    members: &[super::AdaptiveHierarchyMember],
    material: &AdaptiveParticleSet,
) -> AutomataResult<()> {
    if members.len() != material.len()
        || fine.spatial_dims != 2
        || material.spatial_dims != 2
        || material.closure_basis.len() != material.len() * 4
        || material.closure_phase.len() != material.len() * 2
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive closure geometry reconstruction shape mismatch".to_owned(),
        ));
    }

    for (row, member) in members.iter().copied().enumerate() {
        let leaves = hierarchy.member_leaf_indices(member);
        if leaves.len() == 1 {
            fine.positions[leaves[0]][..2].copy_from_slice(&material.positions[row][..2]);
            continue;
        }
        if leaves.len() != 4 {
            return Err(AutomataError::InvalidArgument(format!(
                "compact closure geometry supports one or four leaves, found {}",
                leaves.len(),
            )));
        }

        let total = leaves
            .iter()
            .map(|leaf| f64::from(fine.represented_measure[*leaf]))
            .sum::<f64>()
            .max(f64::MIN_POSITIVE);
        let weights = leaves
            .iter()
            .map(|leaf| f64::from(fine.represented_measure[*leaf]) / total)
            .collect::<Vec<_>>();
        let weight_norm = l2_norm(&weights);
        let weight_direction = weights
            .iter()
            .map(|weight| weight / weight_norm)
            .collect::<Vec<_>>();
        let mut null_direction = material.closure_basis[row * 4..(row + 1) * 4]
            .iter()
            .map(|value| f64::from(*value))
            .collect::<Vec<_>>();
        orthogonalize(&mut null_direction, std::slice::from_ref(&weight_direction));
        let null_norm = l2_norm(&null_direction);
        if null_norm <= 1.0e-10 {
            return Err(AutomataError::InvalidModel(
                "adaptive closure geometry has a degenerate null basis".to_owned(),
            ));
        }
        null_direction
            .iter_mut()
            .for_each(|value| *value /= null_norm);
        let plane =
            closure_geometry_plane(&weight_direction, &null_direction).ok_or_else(|| {
                AutomataError::InvalidModel(
                    "adaptive closure geometry has a degenerate tangent plane".to_owned(),
                )
            })?;

        let mut metric = [[0.0_f64; 2]; 2];
        for lhs in 0..2 {
            for rhs in 0..2 {
                metric[lhs][rhs] = (0..4)
                    .map(|column| weights[column] * plane[lhs][column] * plane[rhs][column])
                    .sum();
            }
        }
        let metric_sqrt = symmetric_sqrt_2x2(metric).ok_or_else(|| {
            AutomataError::InvalidModel(
                "adaptive closure geometry has a non-positive weighted metric".to_owned(),
            )
        })?;
        let metric_inverse_sqrt = invert_2x2(metric_sqrt).ok_or_else(|| {
            AutomataError::InvalidModel(
                "adaptive closure geometry has a singular weighted metric".to_owned(),
            )
        })?;

        let mut offset_covariance = [[0.0_f64; 2]; 2];
        for (lhs, covariance_row) in offset_covariance.iter_mut().enumerate() {
            for (rhs, covariance_value) in covariance_row.iter_mut().enumerate() {
                let intrinsic = leaves
                    .iter()
                    .enumerate()
                    .map(|(column, leaf)| {
                        weights[column] * f64::from(fine.covariance[*leaf][lhs * 3 + rhs])
                    })
                    .sum::<f64>();
                *covariance_value = f64::from(material.covariance[row][lhs * 3 + rhs]) - intrinsic;
            }
        }
        let off_diagonal = 0.5 * (offset_covariance[0][1] + offset_covariance[1][0]);
        offset_covariance[0][1] = off_diagonal;
        offset_covariance[1][0] = off_diagonal;
        let covariance_sqrt = symmetric_sqrt_2x2(offset_covariance).ok_or_else(|| {
            AutomataError::InvalidModel(
                "adaptive closure geometry has a non-positive offset covariance".to_owned(),
            )
        })?;

        let phase = &material.closure_phase[row * 2..(row + 1) * 2];
        let phase_norm = f64::from(phase[0]).hypot(f64::from(phase[1]));
        if !phase_norm.is_finite() || phase_norm <= 1.0e-10 {
            return Err(AutomataError::InvalidModel(
                "adaptive closure geometry has a degenerate phase".to_owned(),
            ));
        }
        let cosine = f64::from(phase[0]) / phase_norm;
        let sine = f64::from(phase[1]) / phase_norm;
        let canonical_normal =
            cofactor_null_direction(&[weights.clone(), plane[0].clone(), plane[1].clone()])
                .ok_or_else(|| {
                    AutomataError::InvalidModel(
                        "adaptive closure geometry orientation is degenerate".to_owned(),
                    )
                })?;
        let orientation = canonical_normal
            .iter()
            .zip(&null_direction)
            .map(|(canonical, actual)| canonical * actual)
            .sum::<f64>()
            .signum();
        let rotation = [[cosine, -orientation * sine], [sine, orientation * cosine]];
        let coordinates =
            multiply_2x2(multiply_2x2(covariance_sqrt, rotation), metric_inverse_sqrt);
        for (column, leaf) in leaves.iter().copied().enumerate() {
            for (axis, coordinate) in coordinates.iter().enumerate() {
                fine.positions[leaf][axis] = (f64::from(material.positions[row][axis])
                    + coordinate[0] * plane[0][column]
                    + coordinate[1] * plane[1][column])
                    as f32;
            }
        }
    }
    Ok(())
}

struct AffineClosureSystem {
    constraints: Vec<Vec<f64>>,
    inverse_gram: Vec<f64>,
    null_direction: Vec<f64>,
    rank: usize,
}

fn closure_geometry_phase(
    fine: &AdaptiveParticleSet,
    leaves: &[usize],
    system: &AffineClosureSystem,
) -> Option<[f32; 2]> {
    if fine.spatial_dims != 2 || leaves.len() != 4 {
        return None;
    }
    let weights = &system.constraints[0];
    let weight_norm = l2_norm(weights);
    if weight_norm <= 1.0e-12 {
        return None;
    }
    let weight_direction = weights
        .iter()
        .map(|weight| weight / weight_norm)
        .collect::<Vec<_>>();
    let mut plane = closure_geometry_plane(&weight_direction, &system.null_direction)?;

    let total = leaves
        .iter()
        .map(|leaf| f64::from(fine.represented_measure[*leaf]))
        .sum::<f64>()
        .max(f64::MIN_POSITIVE);
    let mut center = [0.0_f64; 2];
    for (column, leaf) in leaves.iter().copied().enumerate() {
        let weight = weights[column];
        for (axis, value) in center.iter_mut().enumerate() {
            *value += weight * f64::from(fine.positions[leaf][axis]);
        }
    }
    let mut centered = [[0.0_f64; 4]; 2];
    for (column, leaf) in leaves.iter().copied().enumerate() {
        for axis in 0..2 {
            centered[axis][column] = f64::from(fine.positions[leaf][axis]) - center[axis];
        }
    }
    let mut coordinates = [[0.0_f64; 2]; 2];
    for axis in 0..2 {
        for basis in 0..2 {
            coordinates[axis][basis] = centered[axis]
                .iter()
                .zip(&plane[basis])
                .map(|(value, direction)| value * direction)
                .sum();
        }
    }
    if determinant_2x2(coordinates) < 0.0 {
        plane[1].iter_mut().for_each(|value| *value = -*value);
        for coordinate in &mut coordinates {
            coordinate[1] = -coordinate[1];
        }
    }

    let mut metric = [[0.0_f64; 2]; 2];
    for lhs in 0..2 {
        for rhs in 0..2 {
            metric[lhs][rhs] = (0..4)
                .map(|column| weights[column] * plane[lhs][column] * plane[rhs][column])
                .sum();
        }
    }
    let metric_sqrt = symmetric_sqrt_2x2(metric)?;
    let weighted_coordinates = multiply_2x2(coordinates, metric_sqrt);
    let covariance = multiply_2x2(weighted_coordinates, transpose_2x2(weighted_coordinates));
    let covariance_inverse_sqrt = invert_2x2(symmetric_sqrt_2x2(covariance)?)?;
    let rotation = multiply_2x2(covariance_inverse_sqrt, weighted_coordinates);
    let mut phase = [rotation[0][0], rotation[1][0]];
    let norm = phase[0].hypot(phase[1]);
    if !norm.is_finite() || norm <= 1.0e-10 || !total.is_finite() {
        return None;
    }
    phase.iter_mut().for_each(|value| *value /= norm);
    Some([phase[0] as f32, phase[1] as f32])
}

fn closure_geometry_plane(
    weight_direction: &[f64],
    null_direction: &[f64],
) -> Option<Vec<Vec<f64>>> {
    if weight_direction.len() != 4 || null_direction.len() != 4 {
        return None;
    }
    let mut plane = Vec::<Vec<f64>>::with_capacity(2);
    for seed_axis in 0..4 {
        let mut candidate = vec![0.0_f64; 4];
        candidate[seed_axis] = 1.0;
        orthogonalize(
            &mut candidate,
            &[weight_direction.to_vec(), null_direction.to_vec()],
        );
        orthogonalize(&mut candidate, &plane);
        let norm = l2_norm(&candidate);
        if norm > 1.0e-10 {
            candidate.iter_mut().for_each(|value| *value /= norm);
            plane.push(candidate);
            if plane.len() == 2 {
                return Some(plane);
            }
        }
    }
    None
}

fn determinant_2x2(matrix: [[f64; 2]; 2]) -> f64 {
    matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
}

fn transpose_2x2(matrix: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [[matrix[0][0], matrix[1][0]], [matrix[0][1], matrix[1][1]]]
}

fn multiply_2x2(lhs: [[f64; 2]; 2], rhs: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let mut output = [[0.0_f64; 2]; 2];
    for row in 0..2 {
        for col in 0..2 {
            output[row][col] = (0..2).map(|inner| lhs[row][inner] * rhs[inner][col]).sum();
        }
    }
    output
}

fn symmetric_sqrt_2x2(matrix: [[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
    let determinant_sqrt = determinant_2x2(matrix).max(0.0).sqrt();
    let denominator = (matrix[0][0] + matrix[1][1] + 2.0 * determinant_sqrt).sqrt();
    if !denominator.is_finite() || denominator <= 1.0e-12 {
        return None;
    }
    Some([
        [
            (matrix[0][0] + determinant_sqrt) / denominator,
            matrix[0][1] / denominator,
        ],
        [
            matrix[1][0] / denominator,
            (matrix[1][1] + determinant_sqrt) / denominator,
        ],
    ])
}

fn invert_2x2(matrix: [[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
    let determinant = determinant_2x2(matrix);
    if !determinant.is_finite() || determinant.abs() <= 1.0e-12 {
        return None;
    }
    let inverse = determinant.recip();
    Some([
        [matrix[1][1] * inverse, -matrix[0][1] * inverse],
        [-matrix[1][0] * inverse, matrix[0][0] * inverse],
    ])
}

impl AffineClosureSystem {
    fn new(
        fine: &AdaptiveParticleSet,
        leaves: &[usize],
        basis_anchor: Option<&[f32]>,
    ) -> Option<Self> {
        let rank = fine.spatial_dims + 1;
        if leaves.len() <= rank {
            return None;
        }
        let total = leaves
            .iter()
            .map(|leaf| fine.represented_measure[*leaf] as f64)
            .sum::<f64>()
            .max(f64::MIN_POSITIVE);
        let mut center = vec![0.0_f64; fine.spatial_dims];
        for leaf in leaves {
            let weight = fine.represented_measure[*leaf] as f64 / total;
            for (axis, value) in center.iter_mut().enumerate() {
                *value += weight * fine.positions[*leaf][axis] as f64;
            }
        }
        let constraints = (0..rank)
            .map(|constraint| {
                leaves
                    .iter()
                    .map(|leaf| {
                        let weight = fine.represented_measure[*leaf] as f64 / total;
                        if constraint == 0 {
                            weight
                        } else {
                            weight
                                * (fine.positions[*leaf][constraint - 1]
                                    - center[constraint - 1] as f32)
                                    as f64
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut gram = vec![0.0_f64; rank * rank];
        for row in 0..rank {
            for col in 0..rank {
                gram[row * rank + col] = constraints[row]
                    .iter()
                    .zip(&constraints[col])
                    .map(|(lhs, rhs)| lhs * rhs)
                    .sum();
            }
        }
        let inverse_gram = invert_square(gram, rank)?;
        let mut basis = Vec::with_capacity(rank);
        for row in &constraints {
            let mut direction = row.clone();
            orthogonalize(&mut direction, &basis);
            let norm = l2_norm(&direction);
            if norm <= 1.0e-12 {
                return None;
            }
            direction.iter_mut().for_each(|value| *value /= norm);
            basis.push(direction);
        }
        let null_direction = normalized_cofactor_null_direction(&constraints, basis_anchor)?;
        Some(Self {
            constraints,
            inverse_gram,
            null_direction,
            rank,
        })
    }

    fn minimum_norm_solution(&self, targets: &[f64]) -> AutomataResult<Vec<f64>> {
        if targets.len() != self.rank {
            return Err(AutomataError::InvalidArgument(
                "adaptive closure constraint shape mismatch".to_owned(),
            ));
        }
        let coefficients = (0..self.rank)
            .map(|row| {
                (0..self.rank)
                    .map(|col| self.inverse_gram[row * self.rank + col] * targets[col])
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        Ok((0..self.constraints[0].len())
            .map(|leaf| {
                (0..self.rank)
                    .map(|row| self.constraints[row][leaf] * coefficients[row])
                    .sum()
            })
            .collect())
    }

    fn constraint_targets(&self, values: &[f64]) -> AutomataResult<Vec<f64>> {
        if values.len() != self.constraints[0].len() {
            return Err(AutomataError::InvalidArgument(
                "adaptive closure value shape mismatch".to_owned(),
            ));
        }
        Ok(self
            .constraints
            .iter()
            .map(|row| row.iter().zip(values).map(|(lhs, rhs)| lhs * rhs).sum())
            .collect())
    }
}

fn cofactor_null_direction(constraints: &[Vec<f64>]) -> Option<Vec<f64>> {
    if constraints.len() != 3 || constraints.iter().any(|row| row.len() != 4) {
        return None;
    }
    let mut direction = Vec::with_capacity(4);
    for removed in 0..4 {
        let columns = (0..4)
            .filter(|column| *column != removed)
            .collect::<Vec<_>>();
        let determinant = determinant_3x3([
            [
                constraints[0][columns[0]],
                constraints[0][columns[1]],
                constraints[0][columns[2]],
            ],
            [
                constraints[1][columns[0]],
                constraints[1][columns[1]],
                constraints[1][columns[2]],
            ],
            [
                constraints[2][columns[0]],
                constraints[2][columns[1]],
                constraints[2][columns[2]],
            ],
        ]);
        direction.push(if removed % 2 == 0 {
            determinant
        } else {
            -determinant
        });
    }
    Some(direction)
}

fn normalized_cofactor_null_direction(
    constraints: &[Vec<f64>],
    basis_anchor: Option<&[f32]>,
) -> Option<Vec<f64>> {
    let mut direction = cofactor_null_direction(constraints)?;
    let norm = l2_norm(&direction);
    if norm <= 1.0e-12 {
        return None;
    }
    direction.iter_mut().for_each(|value| *value /= norm);
    if let Some(anchor) = basis_anchor {
        if anchor.len() != direction.len() {
            return None;
        }
        let alignment = direction
            .iter()
            .zip(anchor)
            .map(|(direction, anchor)| direction * f64::from(*anchor))
            .sum::<f64>();
        if alignment < 0.0 {
            direction.iter_mut().for_each(|value| *value = -*value);
        }
    }
    Some(direction)
}

fn determinant_3x3(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

fn orthogonalize(vector: &mut [f64], basis: &[Vec<f64>]) {
    for direction in basis {
        let projection = vector
            .iter()
            .zip(direction)
            .map(|(lhs, rhs)| lhs * rhs)
            .sum::<f64>();
        for (value, direction) in vector.iter_mut().zip(direction) {
            *value -= projection * direction;
        }
    }
}

fn l2_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn invert_square(mut matrix: Vec<f64>, dimension: usize) -> Option<Vec<f64>> {
    let mut inverse = vec![0.0_f64; dimension * dimension];
    for row in 0..dimension {
        inverse[row * dimension + row] = 1.0;
    }
    for pivot in 0..dimension {
        let selected = (pivot..dimension).max_by(|lhs, rhs| {
            matrix[*lhs * dimension + pivot]
                .abs()
                .total_cmp(&matrix[*rhs * dimension + pivot].abs())
        })?;
        if matrix[selected * dimension + pivot].abs() <= 1.0e-14 {
            return None;
        }
        for col in 0..dimension {
            matrix.swap(pivot * dimension + col, selected * dimension + col);
            inverse.swap(pivot * dimension + col, selected * dimension + col);
        }
        let scale = matrix[pivot * dimension + pivot].recip();
        for col in 0..dimension {
            matrix[pivot * dimension + col] *= scale;
            inverse[pivot * dimension + col] *= scale;
        }
        for row in 0..dimension {
            if row == pivot {
                continue;
            }
            let factor = matrix[row * dimension + pivot];
            for col in 0..dimension {
                matrix[row * dimension + col] -= factor * matrix[pivot * dimension + col];
                inverse[row * dimension + col] -= factor * inverse[pivot * dimension + col];
            }
        }
    }
    Some(inverse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ParticleSeed, adaptive::AdaptiveProxyHierarchy, rollout::seed_particles_scaled};

    #[test]
    fn cofactor_mode_is_orthogonal_and_temporally_oriented() {
        let constraints = vec![
            vec![0.25, 0.25, 0.25, 0.25],
            vec![-0.03, 0.01, 0.04, -0.02],
            vec![0.02, -0.05, 0.01, 0.02],
        ];
        let perturbed = vec![
            constraints[0].clone(),
            vec![-0.0301, 0.0102, 0.0399, -0.02],
            vec![0.0201, -0.0502, 0.0101, 0.02],
        ];
        let direction = cofactor_null_direction(&constraints).unwrap();
        let next_direction = cofactor_null_direction(&perturbed).unwrap();
        for row in &constraints {
            let residual = row
                .iter()
                .zip(&direction)
                .map(|(lhs, rhs)| lhs * rhs)
                .sum::<f64>();
            assert!(residual.abs() < 1.0e-12, "null residual {residual}");
        }
        let orientation = direction
            .iter()
            .zip(&next_direction)
            .map(|(lhs, rhs)| lhs * rhs)
            .sum::<f64>()
            / (l2_norm(&direction) * l2_norm(&next_direction));
        assert!(orientation > 0.999, "mode orientation {orientation}");
    }

    #[test]
    fn basis_anchor_removes_odd_child_permutation_gauge() {
        let constraints = vec![
            vec![0.25, 0.25, 0.25, 0.25],
            vec![-0.03, 0.01, 0.04, -0.02],
            vec![0.02, -0.05, 0.01, 0.02],
        ];
        let direction = normalized_cofactor_null_direction(&constraints, None).unwrap();
        let permutation = [1, 0, 2, 3];
        let permuted = constraints
            .iter()
            .map(|row| permutation.map(|index| row[index]).to_vec())
            .collect::<Vec<_>>();
        let anchor = permutation.map(|index| direction[index] as f32);
        let oriented = normalized_cofactor_null_direction(&permuted, Some(&anchor)).unwrap();
        let error = oriented
            .iter()
            .zip(anchor)
            .map(|(actual, expected)| (actual - f64::from(expected)).abs())
            .fold(0.0_f64, f64::max);
        assert!(error < 1.0e-7, "permuted basis error {error}");
    }

    #[test]
    fn one_mode_reconstructs_four_child_state_exactly() {
        let (positions, mut states) =
            seed_particles_scaled(1, 16, 4, 2, 17, ParticleSeed::UniformCircle, 0.2);
        for (index, value) in states.iter_mut().enumerate() {
            *value += 0.2 * ((index * 7 + index / 4) as f32 * 0.37).sin();
        }
        let fine = AdaptiveParticleSet::from_equal_measure(
            positions,
            states,
            2,
            4,
            std::f32::consts::PI * 0.2 * 0.2,
            0.1,
        )
        .unwrap();
        let hierarchy = AdaptiveProxyHierarchy::build(&fine, 4).unwrap();
        let detail = (0..fine.len() * 4)
            .map(|index| (index as f32 * 0.37).sin())
            .collect::<Vec<_>>();
        let view = hierarchy.material_cut(&fine, 4, &detail, 4).unwrap();
        let (modes, metrics) = restrict_first_closure_mode(&fine, &hierarchy, &view).unwrap();

        assert_eq!(metrics.active_rows, 4);
        assert_eq!(metrics.unresolved_modes, 4);
        assert_eq!(metrics.maximum_unresolved_modes_per_row, 1);
        assert_eq!(modes.values.len(), view.particles.len() * fine.state_dims);
        assert!(metrics.affine_root_mean_square_error > 1.0e-4);
        assert!(metrics.augmented_root_mean_square_error < 2.0e-5);
        assert!(metrics.maximum_augmented_absolute_error < 1.0e-4);
    }

    #[test]
    fn geometry_phase_is_unit_length_and_temporally_continuous() {
        let (positions, states) =
            seed_particles_scaled(1, 16, 4, 2, 41, ParticleSeed::UniformCircle, 0.2);
        let fine = AdaptiveParticleSet::from_equal_measure(
            positions,
            states,
            2,
            4,
            std::f32::consts::PI * 0.2 * 0.2,
            0.1,
        )
        .unwrap();
        let hierarchy = AdaptiveProxyHierarchy::build(&fine, 4).unwrap();
        let detail = (0..fine.len() * 4)
            .map(|index| (index as f32 * 0.13).sin())
            .collect::<Vec<_>>();
        let view = hierarchy.material_cut(&fine, 4, &detail, 4).unwrap();
        let (current, _) = restrict_first_closure_mode(&fine, &hierarchy, &view).unwrap();
        let mut perturbed = fine.clone();
        for (row, position) in perturbed.positions.iter_mut().enumerate() {
            position[0] += 1.0e-4 * (row as f32 * 0.7).sin();
            position[1] += 1.0e-4 * (row as f32 * 0.9).cos();
        }
        let (next, _) = restrict_first_closure_mode_for_members_oriented(
            &perturbed,
            &hierarchy,
            &view.members,
            Some(&current.basis),
        )
        .unwrap();
        for row in 0..view.particles.len() {
            if !current.active[row] {
                continue;
            }
            let phase = &current.phase[row * 2..(row + 1) * 2];
            let next_phase = &next.phase[row * 2..(row + 1) * 2];
            let norm = phase.iter().map(|value| value * value).sum::<f32>().sqrt();
            let alignment = phase
                .iter()
                .zip(next_phase)
                .map(|(lhs, rhs)| lhs * rhs)
                .sum::<f32>();
            assert!((norm - 1.0).abs() < 1.0e-5, "phase norm {norm}");
            assert!(alignment > 0.99, "phase alignment {alignment}");
        }
    }

    #[test]
    fn compact_geometry_round_trips_four_child_positions() {
        let (positions, states) =
            seed_particles_scaled(1, 16, 4, 2, 43, ParticleSeed::UniformCircle, 0.2);
        let fine = AdaptiveParticleSet::from_equal_measure(
            positions,
            states,
            2,
            4,
            std::f32::consts::PI * 0.2 * 0.2,
            0.1,
        )
        .unwrap();
        let hierarchy = AdaptiveProxyHierarchy::build(&fine, 4).unwrap();
        let detail = (0..fine.len() * 4)
            .map(|index| (index as f32 * 0.23).sin())
            .collect::<Vec<_>>();
        let mut view = hierarchy.material_cut(&fine, 4, &detail, 4).unwrap();
        attach_first_closure_mode(&fine, &hierarchy, &mut view).unwrap();

        let mut reconstructed = fine.clone();
        for (row, position) in reconstructed.positions.iter_mut().enumerate() {
            position[0] = 1.0 + row as f32 * 0.01;
            position[1] = -1.0 - row as f32 * 0.02;
        }
        reconstruct_first_closure_geometry_for_members(
            &mut reconstructed,
            &hierarchy,
            &view.members,
            &view.particles,
        )
        .unwrap();

        let maximum_error = reconstructed
            .positions
            .iter()
            .zip(&fine.positions)
            .flat_map(|(actual, expected)| {
                actual[..2]
                    .iter()
                    .zip(&expected[..2])
                    .map(|(actual, expected)| (actual - expected).abs())
            })
            .fold(0.0_f32, f32::max);
        assert!(
            maximum_error < 2.0e-4,
            "compact geometry round-trip error {maximum_error}"
        );
    }

    #[test]
    fn geometry_phase_changes_shape_orientation_without_changing_moments() {
        let (positions, states) =
            seed_particles_scaled(1, 16, 4, 2, 47, ParticleSeed::UniformCircle, 0.2);
        let fine = AdaptiveParticleSet::from_equal_measure(
            positions,
            states,
            2,
            4,
            std::f32::consts::PI * 0.2 * 0.2,
            0.1,
        )
        .unwrap();
        let hierarchy = AdaptiveProxyHierarchy::build(&fine, 4).unwrap();
        let detail = (0..fine.len() * 4)
            .map(|index| (index as f32 * 0.31).cos())
            .collect::<Vec<_>>();
        let mut view = hierarchy.material_cut(&fine, 4, &detail, 4).unwrap();
        attach_first_closure_mode(&fine, &hierarchy, &mut view).unwrap();
        let coarse_row = view
            .members
            .iter()
            .position(|member| hierarchy.member_leaf_indices(*member).len() == 4)
            .unwrap();
        let leaves = hierarchy.member_leaf_indices(view.members[coarse_row]);

        let mut baseline = fine.clone();
        reconstruct_first_closure_geometry_for_members(
            &mut baseline,
            &hierarchy,
            &view.members,
            &view.particles,
        )
        .unwrap();
        let mut rotated_material = view.particles.clone();
        let phase = &mut rotated_material.closure_phase[coarse_row * 2..(coarse_row + 1) * 2];
        let [cosine, sine] = [0.7_f32.cos(), 0.7_f32.sin()];
        [phase[0], phase[1]] = [
            cosine * phase[0] - sine * phase[1],
            sine * phase[0] + cosine * phase[1],
        ];
        let mut rotated = fine.clone();
        reconstruct_first_closure_geometry_for_members(
            &mut rotated,
            &hierarchy,
            &view.members,
            &rotated_material,
        )
        .unwrap();

        let maximum_change = leaves
            .iter()
            .flat_map(|leaf| {
                baseline.positions[*leaf][..2]
                    .iter()
                    .zip(&rotated.positions[*leaf][..2])
                    .map(|(baseline, rotated)| (baseline - rotated).abs())
            })
            .fold(0.0_f32, f32::max);
        assert!(maximum_change > 1.0e-3, "phase change {maximum_change}");

        let (center, covariance) = weighted_geometry(&rotated, leaves);
        for (axis, covariance_row) in covariance.iter().enumerate() {
            assert!(
                (center[axis] - f64::from(view.particles.positions[coarse_row][axis])).abs()
                    < 2.0e-5
            );
            for (inner, covariance_value) in covariance_row.iter().enumerate() {
                assert!(
                    (*covariance_value
                        - f64::from(view.particles.covariance[coarse_row][axis * 3 + inner]))
                    .abs()
                        < 2.0e-5
                );
            }
        }
    }

    #[test]
    fn four_child_geometry_requires_the_affine_null_basis_to_evolve() {
        let (positions, states) =
            seed_particles_scaled(1, 16, 4, 2, 53, ParticleSeed::UniformCircle, 0.2);
        let fine = AdaptiveParticleSet::from_equal_measure(
            positions,
            states,
            2,
            4,
            std::f32::consts::PI * 0.2 * 0.2,
            0.1,
        )
        .unwrap();
        let hierarchy = AdaptiveProxyHierarchy::build(&fine, 4).unwrap();
        let detail = (0..fine.len() * 4)
            .map(|index| (index as f32 * 0.37).sin())
            .collect::<Vec<_>>();
        let mut view = hierarchy.material_cut(&fine, 4, &detail, 4).unwrap();
        attach_first_closure_mode(&fine, &hierarchy, &mut view).unwrap();
        let coarse_row = view
            .members
            .iter()
            .position(|member| hierarchy.member_leaf_indices(*member).len() == 4)
            .unwrap();
        let leaves = hierarchy.member_leaf_indices(view.members[coarse_row]);

        let mut next = fine.clone();
        let displacements = [
            [0.018, -0.007],
            [-0.011, 0.016],
            [0.006, 0.013],
            [-0.013, -0.022],
        ];
        for (leaf, displacement) in leaves.iter().zip(displacements) {
            next.positions[*leaf][0] += displacement[0];
            next.positions[*leaf][1] += displacement[1];
        }
        let (next_field, _) = restrict_first_closure_mode_for_members_oriented(
            &next,
            &hierarchy,
            &view.members,
            Some(&view.particles.closure_basis),
        )
        .unwrap();
        let (center, covariance) = weighted_geometry(&next, leaves);
        let mut next_material = view.particles.clone();
        next_material.positions[coarse_row][0] = center[0] as f32;
        next_material.positions[coarse_row][1] = center[1] as f32;
        for (lhs, covariance_row) in covariance.iter().enumerate() {
            for (rhs, value) in covariance_row.iter().enumerate() {
                next_material.covariance[coarse_row][lhs * 3 + rhs] = *value as f32;
            }
        }
        next_material.closure_phase[coarse_row * 2..(coarse_row + 1) * 2]
            .copy_from_slice(&next_field.phase[coarse_row * 2..(coarse_row + 1) * 2]);

        let mut frozen_basis = next.clone();
        reconstruct_first_closure_geometry_for_members(
            &mut frozen_basis,
            &hierarchy,
            &view.members,
            &next_material,
        )
        .unwrap();
        let frozen_error = leaves
            .iter()
            .flat_map(|leaf| {
                frozen_basis.positions[*leaf][..2]
                    .iter()
                    .zip(&next.positions[*leaf][..2])
                    .map(|(actual, expected)| (actual - expected).abs())
            })
            .fold(0.0_f32, f32::max);

        next_material.closure_basis[coarse_row * 4..(coarse_row + 1) * 4]
            .copy_from_slice(&next_field.basis[coarse_row * 4..(coarse_row + 1) * 4]);
        let mut evolved_basis = next.clone();
        reconstruct_first_closure_geometry_for_members(
            &mut evolved_basis,
            &hierarchy,
            &view.members,
            &next_material,
        )
        .unwrap();
        let evolved_error = leaves
            .iter()
            .flat_map(|leaf| {
                evolved_basis.positions[*leaf][..2]
                    .iter()
                    .zip(&next.positions[*leaf][..2])
                    .map(|(actual, expected)| (actual - expected).abs())
            })
            .fold(0.0_f32, f32::max);

        assert!(frozen_error > 1.0e-3, "frozen basis error {frozen_error}");
        assert!(
            evolved_error < 2.0e-4,
            "evolved basis round-trip error {evolved_error}"
        );
    }

    #[test]
    fn material_moments_and_mode_reconstruct_restricted_child_state() {
        let (positions, mut states) =
            seed_particles_scaled(1, 16, 4, 2, 31, ParticleSeed::UniformCircle, 0.2);
        for (index, value) in states.iter_mut().enumerate() {
            *value += 0.3 * ((index * 11 + 5) as f32 * 0.21).cos();
        }
        let fine = AdaptiveParticleSet::from_equal_measure(
            positions,
            states,
            2,
            4,
            std::f32::consts::PI * 0.2 * 0.2,
            0.1,
        )
        .unwrap();
        let hierarchy = AdaptiveProxyHierarchy::build(&fine, 4).unwrap();
        let detail = (0..fine.len() * 4)
            .map(|index| (index as f32 * 0.19).cos())
            .collect::<Vec<_>>();
        let mut view = hierarchy.material_cut(&fine, 4, &detail, 4).unwrap();
        attach_first_closure_mode(&fine, &hierarchy, &mut view).unwrap();
        let mut reconstructed = fine.clone();
        reconstructed.states.fill(0.0);
        reconstruct_first_closure_state_for_members(
            &mut reconstructed,
            &hierarchy,
            &view.members,
            &view.particles,
        )
        .unwrap();
        let max_error = reconstructed
            .states
            .iter()
            .zip(&fine.states)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_error < 1.0e-4,
            "closure reconstruction error {max_error}"
        );
    }

    fn weighted_geometry(
        particles: &AdaptiveParticleSet,
        leaves: &[usize],
    ) -> ([f64; 2], [[f64; 2]; 2]) {
        let total = leaves
            .iter()
            .map(|leaf| f64::from(particles.represented_measure[*leaf]))
            .sum::<f64>();
        let mut center = [0.0_f64; 2];
        for leaf in leaves {
            let weight = f64::from(particles.represented_measure[*leaf]) / total;
            for (axis, center) in center.iter_mut().enumerate() {
                *center += weight * f64::from(particles.positions[*leaf][axis]);
            }
        }
        let mut covariance = [[0.0_f64; 2]; 2];
        for leaf in leaves {
            let weight = f64::from(particles.represented_measure[*leaf]) / total;
            for lhs in 0..2 {
                for rhs in 0..2 {
                    covariance[lhs][rhs] += weight
                        * (f64::from(particles.covariance[*leaf][lhs * 3 + rhs])
                            + (f64::from(particles.positions[*leaf][lhs]) - center[lhs])
                                * (f64::from(particles.positions[*leaf][rhs]) - center[rhs]));
                }
            }
        }
        (center, covariance)
    }
}
