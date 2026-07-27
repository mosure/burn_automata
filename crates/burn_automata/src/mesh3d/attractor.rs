use crate::{
    AutomataError, AutomataResult, NpaModel, TriangleMeshTarget,
    rollout::{GROWTH_3D_LIVENESS_CHANNEL, GROWTH_3D_RENDER_OPACITY_CHANNEL},
};

const TARGET_OPACITY_LOGIT: f32 = 4.0;
const ATTRACTOR_HIDDEN_DIMS: usize = 16;

/// Seeds stable material-state residuals while leaving the rest of the NPA trainable.
///
/// Imported OBJ colors are represented by their best affine fit over mesh
/// vertices. The canonical teapot's normalized-position colors are exactly
/// affine, so erased color, liveness, and opacity recover without introducing
/// a poorly conditioned recurrent attractor into supervised optimization.
pub(crate) fn seed_mesh3d_state_attractor(
    model: &mut NpaModel,
    target: &TriangleMeshTarget,
    color_gain: f32,
    opacity_gain: f32,
) -> AutomataResult<()> {
    if !model.config.position_features
        || model.config.spatial_dims != 3
        || model.config.state_dims <= GROWTH_3D_RENDER_OPACITY_CHANNEL
        || model.config.state_dims < 3
        || model.config.hidden_dims < ATTRACTOR_HIDDEN_DIMS
    {
        return Err(AutomataError::InvalidModel(
            "mesh3d state attractor requires a position-conditioned 3D model with at least 16 hidden dimensions"
                .to_string(),
        ));
    }
    if !color_gain.is_finite()
        || color_gain < 0.0
        || !opacity_gain.is_finite()
        || opacity_gain < 0.0
    {
        return Err(AutomataError::InvalidArgument(
            "mesh3d state-attractor gains must be finite and non-negative".to_string(),
        ));
    }

    let input_dims = model.config.perception_dims();
    let hidden_dims = model.config.hidden_dims;
    let output_dims = model.config.update_dims();
    let position_offset =
        input_dims - model.config.auxiliary_input_dims - model.config.spatial_dims;
    let color_tail = model.config.state_dims - 3;
    let controlled_channels = [
        GROWTH_3D_LIVENESS_CHANNEL,
        GROWTH_3D_RENDER_OPACITY_CHANNEL,
        color_tail,
        color_tail + 1,
        color_tail + 2,
    ];

    for hidden in 0..ATTRACTOR_HIDDEN_DIMS {
        model.weights.w1[hidden * input_dims..(hidden + 1) * input_dims].fill(0.0);
        model.weights.b1[hidden] = 0.0;
        for output in 0..output_dims {
            model.weights.w2[output * hidden_dims + hidden] = 0.0;
        }
    }
    for channel in controlled_channels {
        let output = model.config.spatial_dims + channel;
        model.weights.w2[output * hidden_dims..(output + 1) * hidden_dims].fill(0.0);
        model.weights.b2[output] = 0.0;
    }

    let position_pairs = [
        install_identity_pair(model, 0, position_offset),
        install_identity_pair(model, 1, position_offset + 1),
        install_identity_pair(model, 2, position_offset + 2),
    ];
    let color_pairs = [
        install_identity_pair(model, 3, color_tail),
        install_identity_pair(model, 4, color_tail + 1),
        install_identity_pair(model, 5, color_tail + 2),
    ];
    let liveness_pair = install_identity_pair(model, 6, GROWTH_3D_LIVENESS_CHANNEL);
    let opacity_pair = install_identity_pair(model, 7, GROWTH_3D_RENDER_OPACITY_CHANNEL);
    let affine_color = fit_affine_vertex_color(target);

    for channel in 0..3 {
        let output = model.config.spatial_dims + color_tail + channel;
        for axis in 0..3 {
            add_pair_weight(
                model,
                output,
                position_pairs[axis],
                color_gain * affine_color[channel][axis],
            );
        }
        add_pair_weight(model, output, color_pairs[channel], -color_gain);
        model.weights.b2[output] = color_gain * (affine_color[channel][3] - 0.5);
    }

    for (channel, pair) in [
        (GROWTH_3D_LIVENESS_CHANNEL, liveness_pair),
        (GROWTH_3D_RENDER_OPACITY_CHANNEL, opacity_pair),
    ] {
        let output = model.config.spatial_dims + channel;
        add_pair_weight(model, output, pair, -opacity_gain);
        model.weights.b2[output] = opacity_gain * TARGET_OPACITY_LOGIT;
    }
    Ok(())
}

fn install_identity_pair(model: &mut NpaModel, pair: usize, input: usize) -> [usize; 2] {
    let hidden = [pair * 2, pair * 2 + 1];
    let input_dims = model.config.perception_dims();
    model.weights.w1[hidden[0] * input_dims + input] = 1.0;
    model.weights.w1[hidden[1] * input_dims + input] = -1.0;
    hidden
}

fn add_pair_weight(model: &mut NpaModel, output: usize, hidden: [usize; 2], coefficient: f32) {
    let base = output * model.config.hidden_dims;
    model.weights.w2[base + hidden[0]] += coefficient;
    model.weights.w2[base + hidden[1]] -= coefficient;
}

fn fit_affine_vertex_color(target: &TriangleMeshTarget) -> [[f32; 4]; 3] {
    let mut gram = [[0.0_f64; 4]; 4];
    let mut rhs = [[0.0_f64; 4]; 3];
    for (index, position) in target.vertices.iter().copied().enumerate() {
        let basis = [
            f64::from(position[0]),
            f64::from(position[1]),
            f64::from(position[2]),
            1.0,
        ];
        let color = target
            .colors
            .as_ref()
            .map_or_else(|| target.project(position).color, |colors| colors[index]);
        for row in 0..4 {
            for column in 0..4 {
                gram[row][column] += basis[row] * basis[column];
            }
            for channel in 0..3 {
                rhs[channel][row] += basis[row] * f64::from(color[channel]);
            }
        }
    }
    for (axis, row) in gram.iter_mut().enumerate() {
        row[axis] += 1.0e-8;
    }

    let mut fit = [[0.0_f32; 4]; 3];
    for channel in 0..3 {
        fit[channel] = solve_4x4(gram, rhs[channel]).unwrap_or([0.0, 0.0, 0.0, 0.5]);
    }
    fit
}

fn solve_4x4(matrix: [[f64; 4]; 4], rhs: [f64; 4]) -> Option<[f32; 4]> {
    let mut augmented = [[0.0_f64; 5]; 4];
    for row in 0..4 {
        augmented[row][..4].copy_from_slice(&matrix[row]);
        augmented[row][4] = rhs[row];
    }
    for pivot in 0..4 {
        let best = (pivot..4).max_by(|left, right| {
            augmented[*left][pivot]
                .abs()
                .total_cmp(&augmented[*right][pivot].abs())
        })?;
        if augmented[best][pivot].abs() <= 1.0e-12 {
            return None;
        }
        augmented.swap(pivot, best);
        let divisor = augmented[pivot][pivot];
        for value in &mut augmented[pivot][pivot..] {
            *value /= divisor;
        }
        let pivot_row = augmented[pivot];
        for (row, augmented_row) in augmented.iter_mut().enumerate() {
            if row == pivot {
                continue;
            }
            let factor = augmented_row[pivot];
            for (value, pivot_value) in augmented_row[pivot..].iter_mut().zip(&pivot_row[pivot..]) {
                *value -= factor * pivot_value;
            }
        }
    }
    Some([
        augmented[0][4] as f32,
        augmented[1][4] as f32,
        augmented[2][4] as f32,
        augmented[3][4] as f32,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh3d::mesh3d_model_config;

    #[test]
    fn canonical_teapot_attractor_recovers_color_and_material_residuals() {
        let target = TriangleMeshTarget::utah_teapot(0.72).unwrap();
        let mut model = NpaModel::upstream_seeded(mesh3d_model_config(64), 42);
        seed_mesh3d_state_attractor(&mut model, &target, 0.35, 0.25).unwrap();
        let input_dims = model.config.perception_dims();
        let position_offset =
            input_dims - model.config.auxiliary_input_dims - model.config.spatial_dims;
        let color_tail = model.config.state_dims - 3;
        let mut features = vec![0.0_f32; input_dims * 16];
        for row in 0..16 {
            let sample = target.surface_sample(row * 97);
            let feature = &mut features[row * input_dims..(row + 1) * input_dims];
            feature[position_offset..position_offset + 3].copy_from_slice(&sample.position);
            feature[GROWTH_3D_RENDER_OPACITY_CHANNEL] = -4.0;
        }
        let updates = model.forward_update_from_features(&features).unwrap();
        for row in 0..16 {
            let sample = target.surface_sample(row * 97);
            let update =
                &updates[row * model.config.update_dims()..(row + 1) * model.config.update_dims()];
            let state_update = &update[model.config.spatial_dims..];
            assert!((state_update[GROWTH_3D_LIVENESS_CHANNEL] - 1.0).abs() <= 1.0e-5);
            assert!((state_update[GROWTH_3D_RENDER_OPACITY_CHANNEL] - 2.0).abs() <= 1.0e-5);
            for channel in 0..3 {
                let expected = 0.35 * (sample.color[channel] - 0.5);
                assert!(
                    (state_update[color_tail + channel] - expected).abs() <= 2.0e-5,
                    "row {row} channel {channel}: {} != {expected}",
                    state_update[color_tail + channel]
                );
            }
        }
    }
}
