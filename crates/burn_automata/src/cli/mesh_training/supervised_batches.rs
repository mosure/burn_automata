#![allow(clippy::too_many_arguments)]

use super::*;

#[allow(dead_code)]
pub(crate) fn torus_growth_supervised_batch(config: &NpaConfig, rows: usize) -> SupervisedBatch {
    let rows = rows.max(1);
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let mut features = vec![0.0; rows * input_dims];
    let mut target_update = vec![0.0; rows * output_dims];
    let scales = [0.56_f32, 0.72, 0.88];
    let mut rng = StdRng::seed_from_u64(0x703d_5eed);

    for row in 0..rows {
        let scale = scales[row % scales.len()];
        let sample = uv_torus_sample(row, rows, scale);
        let structured_position = [
            sample.position[0] * UV_TORUS_INITIAL_SCALE,
            sample.position[1] * UV_TORUS_INITIAL_SCALE,
            sample.position[2] * UV_TORUS_INITIAL_SCALE,
        ];
        let dense_position = uv_torus_dense_seed_position(&mut rng, scale);
        let initial_position = if row % 2 == 0 {
            structured_position
        } else {
            dense_position
        };
        let feature_base = row * input_dims;
        let update_base = row * output_dims;
        for axis in 0..3 {
            let residual = sample.position[axis] - initial_position[axis];
            features[feature_base + axis] = residual;
            target_update[update_base + axis] = UV_TORUS_MOTION_GAIN * residual;
            target_update[update_base + config.spatial_dims + axis] =
                -UV_TORUS_RESIDUAL_DECAY * residual;
        }
        features[feature_base + 3] = UV_TORUS_INITIAL_OPACITY_LOGIT;
        target_update[update_base + config.spatial_dims + 3] = UV_TORUS_OPACITY_GROWTH_DELTA;
        if config.state_dims >= 6 {
            let tail = config.state_dims - 3;
            let tail_color = uv_torus_tail_state_color(sample.position, scale);
            features[feature_base + tail] = tail_color[0];
            features[feature_base + tail + 1] = tail_color[1];
            features[feature_base + tail + 2] = tail_color[2];
        }
    }

    SupervisedBatch {
        features,
        target_update,
    }
}

pub(crate) fn torus_morphogen_supervised_batch(config: &NpaConfig, rows: usize) -> SupervisedBatch {
    let rows = rows.max(1);
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let mut features = vec![0.0; rows * input_dims];
    let mut target_update = vec![0.0; rows * output_dims];
    let mut rng = StdRng::seed_from_u64(0x703d_6d0f);
    let scales = [0.56_f32, UV_TORUS_FIELD_SCALE, 0.88];
    let targets = [
        uv_torus_mesh_target(scales[0]),
        uv_torus_mesh_target(scales[1]),
        uv_torus_mesh_target(scales[2]),
    ];

    for row in 0..rows {
        let scale_idx = row % scales.len();
        let scale = scales[scale_idx];
        let position = torus_implicit_training_position(row, scale, &mut rng);
        let projection = targets[scale_idx].project(position);
        let target = projection.closest;
        let residual = projection.residual;
        let feature_base = row * input_dims;
        let update_base = row * output_dims;

        for axis in 0..3 {
            features[feature_base + axis] = residual[axis];
            target_update[update_base + axis] = UV_TORUS_MOTION_GAIN * residual[axis];
            target_update[update_base + config.spatial_dims + axis] =
                -UV_TORUS_RESIDUAL_DECAY * residual[axis];
        }
        if config.state_dims > 3 {
            features[feature_base + 3] = UV_TORUS_INITIAL_OPACITY_LOGIT;
            target_update[update_base + config.spatial_dims + 3] = UV_TORUS_OPACITY_GROWTH_DELTA;
        }
        if uv_torus_orientation_state_available(config.state_dims) {
            features[feature_base + UV_TORUS_NORMAL_STATE_OFFSET] = projection.normal[0];
            features[feature_base + UV_TORUS_NORMAL_STATE_OFFSET + 1] = projection.normal[1];
            features[feature_base + UV_TORUS_NORMAL_STATE_OFFSET + 2] = projection.normal[2];
            features[feature_base + UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET] =
                projection.signed_distance;
        }
        if config.state_dims >= 6 {
            let tail = config.state_dims - 3;
            let tail_color = uv_torus_tail_state_color(target, scale);
            features[feature_base + tail] = tail_color[0];
            features[feature_base + tail + 1] = tail_color[1];
            features[feature_base + tail + 2] = tail_color[2];
        }
        let blur_offset = config.state_dims;
        for channel in 0..config.state_dims {
            features[feature_base + blur_offset + channel] = features[feature_base + channel];
        }
    }

    SupervisedBatch {
        features,
        target_update,
    }
}

pub(crate) fn teapot_morphogen_supervised_batch(
    config: &NpaConfig,
    rows: usize,
) -> SupervisedBatch {
    let rows = rows.max(1);
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let mut features = vec![0.0; rows * input_dims];
    let mut target_update = vec![0.0; rows * output_dims];
    let mut rng = StdRng::seed_from_u64(0x7ea9_07d0);
    let scales = [0.56_f32, UV_TORUS_FIELD_SCALE, 0.88];
    let targets = [
        utah_teapot_mesh_target(scales[0]),
        utah_teapot_mesh_target(scales[1]),
        utah_teapot_mesh_target(scales[2]),
    ];

    for row in 0..rows {
        let scale_idx = row % scales.len();
        let scale = scales[scale_idx];
        let position = utah_teapot_training_position(row, scale, &targets[scale_idx], &mut rng);
        let projection = targets[scale_idx].project(position);
        let target = projection.closest;
        let residual = projection.residual;
        let feature_base = row * input_dims;
        let update_base = row * output_dims;

        for axis in 0..3 {
            features[feature_base + axis] = residual[axis];
            target_update[update_base + axis] = UV_TORUS_MOTION_GAIN * residual[axis];
            target_update[update_base + config.spatial_dims + axis] =
                -UV_TORUS_RESIDUAL_DECAY * residual[axis];
        }
        if config.state_dims > 3 {
            features[feature_base + 3] = UV_TORUS_INITIAL_OPACITY_LOGIT;
            target_update[update_base + config.spatial_dims + 3] = UV_TORUS_OPACITY_GROWTH_DELTA;
        }
        if uv_torus_orientation_state_available(config.state_dims) {
            features[feature_base + UV_TORUS_NORMAL_STATE_OFFSET] = projection.normal[0];
            features[feature_base + UV_TORUS_NORMAL_STATE_OFFSET + 1] = projection.normal[1];
            features[feature_base + UV_TORUS_NORMAL_STATE_OFFSET + 2] = projection.normal[2];
            features[feature_base + UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET] =
                projection.signed_distance;
        }
        if config.state_dims >= 6 {
            let tail = config.state_dims - 3;
            let tail_color = utah_teapot_tail_state_color(target, &targets[scale_idx]);
            features[feature_base + tail] = tail_color[0];
            features[feature_base + tail + 1] = tail_color[1];
            features[feature_base + tail + 2] = tail_color[2];
        }
        let blur_offset = config.state_dims;
        for channel in 0..config.state_dims {
            features[feature_base + blur_offset + channel] = features[feature_base + channel];
        }
    }

    SupervisedBatch {
        features,
        target_update,
    }
}

pub(crate) fn torus_field_supervised_batch(config: &NpaConfig, rows: usize) -> SupervisedBatch {
    assert!(
        config.position_features,
        "torus field training requires position features"
    );
    let rows = rows.max(1);
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let mut features = vec![0.0; rows * input_dims];
    let mut target_update = vec![0.0; rows * output_dims];
    let mut rng = StdRng::seed_from_u64(0x703d_f13d);
    let scales = [UV_TORUS_FIELD_SCALE];
    let targets = [uv_torus_mesh_target(scales[0])];
    let position_offset = input_dims - config.spatial_dims;

    for row in 0..rows {
        let scale_idx = row % scales.len();
        let scale = scales[scale_idx];
        let position = torus_implicit_training_position(row, scale, &mut rng);
        let projection = targets[scale_idx].project(position);
        let target = projection.closest;
        let residual = projection.residual;

        let feature_base = row * input_dims;
        let update_base = row * output_dims;
        let mut current_tail = [0.0_f32; 3];
        if config.state_dims > 3 {
            features[feature_base + 3] =
                rng.random_range(UV_TORUS_INITIAL_OPACITY_LOGIT..UV_TORUS_FIELD_OPACITY_TARGET);
        }
        if config.state_dims >= 6 {
            let tail = config.state_dims - 3;
            for channel in 0..3 {
                current_tail[channel] = rng.random_range(-0.35..0.35);
                features[feature_base + tail + channel] = current_tail[channel];
            }
        }

        let blur_offset = config.state_dims;
        for channel in 0..config.state_dims {
            features[feature_base + blur_offset + channel] = features[feature_base + channel];
        }
        for axis in 0..3 {
            features[feature_base + position_offset + axis] = position[axis];
            target_update[update_base + axis] = UV_TORUS_FIELD_MOTION_GAIN * residual[axis];
        }

        if config.state_dims > 3 {
            let current_opacity = features[feature_base + 3];
            target_update[update_base + config.spatial_dims + 3] =
                UV_TORUS_FIELD_OPACITY_GAIN * (UV_TORUS_FIELD_OPACITY_TARGET - current_opacity);
        }
        if config.state_dims >= 6 {
            let tail = config.state_dims - 3;
            let target_tail = uv_torus_tail_state_color(target, scale);
            for channel in 0..3 {
                target_update[update_base + config.spatial_dims + tail + channel] =
                    UV_TORUS_FIELD_COLOR_GAIN * (target_tail[channel] - current_tail[channel]);
            }
        }
    }

    SupervisedBatch {
        features,
        target_update,
    }
}

pub(crate) fn teapot_field_supervised_batch(config: &NpaConfig, rows: usize) -> SupervisedBatch {
    assert!(
        config.position_features,
        "teapot field training requires position features"
    );
    let rows = rows.max(1);
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let mut features = vec![0.0; rows * input_dims];
    let mut target_update = vec![0.0; rows * output_dims];
    let mut rng = StdRng::seed_from_u64(0x7ea9_f13d);
    let scale = UV_TORUS_FIELD_SCALE;
    let target_mesh = utah_teapot_mesh_target(scale);
    let position_offset = input_dims - config.spatial_dims;

    for row in 0..rows {
        let position = utah_teapot_training_position(row, scale, &target_mesh, &mut rng);
        let projection = target_mesh.project(position);
        let target = projection.closest;
        let residual = projection.residual;

        let feature_base = row * input_dims;
        let update_base = row * output_dims;
        let mut current_tail = [0.0_f32; 3];
        if config.state_dims > 3 {
            features[feature_base + 3] =
                rng.random_range(UV_TORUS_INITIAL_OPACITY_LOGIT..UV_TORUS_FIELD_OPACITY_TARGET);
        }
        if config.state_dims >= 6 {
            let tail = config.state_dims - 3;
            for channel in 0..3 {
                current_tail[channel] = rng.random_range(-0.35..0.35);
                features[feature_base + tail + channel] = current_tail[channel];
            }
        }

        let blur_offset = config.state_dims;
        for channel in 0..config.state_dims {
            features[feature_base + blur_offset + channel] = features[feature_base + channel];
        }
        for axis in 0..3 {
            features[feature_base + position_offset + axis] = position[axis];
            target_update[update_base + axis] = TEAPOT_FIELD_MOTION_GAIN * residual[axis];
        }

        if config.state_dims > 3 {
            let current_opacity = features[feature_base + 3];
            target_update[update_base + config.spatial_dims + 3] =
                UV_TORUS_FIELD_OPACITY_GAIN * (UV_TORUS_FIELD_OPACITY_TARGET - current_opacity);
        }
        if config.state_dims >= 6 {
            let tail = config.state_dims - 3;
            let target_tail = utah_teapot_tail_state_color(target, &target_mesh);
            for channel in 0..3 {
                target_update[update_base + config.spatial_dims + tail + channel] =
                    TEAPOT_FIELD_COLOR_GAIN * (target_tail[channel] - current_tail[channel]);
            }
        }
    }

    SupervisedBatch {
        features,
        target_update,
    }
}
