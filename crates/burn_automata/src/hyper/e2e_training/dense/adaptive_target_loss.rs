//! Variable-footprint Target2D loss for active adaptive material.
//!
//! Render footprint, material output scale, and centering weights are detached
//! material metadata. Gradients flow only to particle position and recurrent
//! state, matching the adaptive perception boundary.

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn adaptive_target_splat_loss_batch_vector_base_only_selected(
    x: &Tensor3,
    s: &Tensor3,
    targets: &[BurnTargetExample],
    indices: &[usize],
    config: DirectBasisTrainConfig,
    represented_measure: Tensor2,
    particle_pixel_size: Tensor2,
    particle_output_scale: Tensor2,
    displacement: Tensor1,
) -> AutomataResult<BurnLossBatchTensors> {
    let [batches, particle_count, _] = x.shape().dims::<3>();
    let expected = [batches, particle_count];
    if represented_measure.shape().dims::<2>() != expected
        || particle_pixel_size.shape().dims::<2>() != expected
        || particle_output_scale.shape().dims::<2>() != expected
    {
        return Err(AutomataError::InvalidArgument(format!(
            "adaptive Target2D material tensors must have shape {expected:?}"
        )));
    }
    if batches != indices.len() || indices.is_empty() {
        return Err(AutomataError::InvalidArgument(
            "adaptive Target2D batch and target indices differ".to_string(),
        ));
    }

    let target_mean = stack_target_mean(targets, indices);
    let centered = represented_measure_centered(
        x,
        represented_measure.clone(),
        target_mean,
        config.loss_config.center,
    );

    #[cfg(any(feature = "backend_wgpu", feature = "backend_cuda"))]
    if target2d_loss_backend_effective(config) == Target2dLossBackend::TiledAdjoint
        && config.loss_config.shape_chamfer_loss_weight == 0.0
    {
        let device_loss = InnerBackend::target2d_cube_adjoint_variable(
            x.clone().inner(),
            centered.clone().inner(),
            s.clone().inner(),
            stack_target_rgb(targets, indices).inner(),
            stack_target_density(targets, indices).inner(),
            stack_target_foreground(targets, indices).inner(),
            stack_target_foreground_scales(targets, indices).inner(),
            particle_pixel_size.clone().inner(),
            particle_output_scale.clone().inner(),
            represented_measure.clone().inner(),
            target2d_cube_loss_config(config.loss_config),
        );
        if let Some(device_loss) = device_loss {
            TARGET2D_CUBE_ADJOINT_DEVICE_HITS.fetch_add(1, Ordering::Relaxed);
            let device_loss = device_loss?;
            return Ok(base_only_loss_batch_vector_from_device_adjoint(
                x,
                s,
                device_loss,
                config,
                None,
                displacement,
            ));
        }
    }

    TARGET2D_CUBE_ADJOINT_FALLBACK_HITS.fetch_add(1, Ordering::Relaxed);
    let state_dims = s.shape().dims::<3>()[2];
    let colors = s.clone().narrow(2, state_dims - 3, 3).add_scalar(0.5);
    let (rgb, density) = adaptive_splat_render_batch(
        &centered,
        &colors,
        targets,
        indices,
        config,
        particle_pixel_size,
        particle_output_scale,
    );
    Ok(base_only_loss_batch_vector_from_render(
        x,
        s,
        &centered,
        rgb,
        density,
        targets,
        indices,
        config,
        displacement,
    ))
}

pub(super) fn represented_measure_centered(
    x: &Tensor3,
    represented_measure: Tensor2,
    target_mean: Tensor3,
    center: bool,
) -> Tensor3 {
    if !center {
        return x.clone();
    }
    let [batches, particles, _] = x.shape().dims::<3>();
    let weight = represented_measure.unsqueeze_dim::<3>(2);
    let total = weight
        .clone()
        .sum_dim(1)
        .clamp_min(EPSILON)
        .expand([batches, 1, 2]);
    let mean = x
        .clone()
        .mul(weight.expand([batches, particles, 2]))
        .sum_dim(1)
        .div(total);
    x.clone() - mean.expand([batches, particles, 2])
        + target_mean.expand([batches, particles, 2])
}

#[allow(clippy::too_many_arguments)]
pub(super) fn adaptive_splat_render_batch(
    x: &Tensor3,
    colors: &Tensor3,
    targets: &[BurnTargetExample],
    indices: &[usize],
    config: DirectBasisTrainConfig,
    particle_pixel_size: Tensor2,
    particle_output_scale: Tensor2,
) -> (Tensor3, Tensor3) {
    let batches = indices.len();
    let particle_count = x.shape().dims::<3>()[1];
    let pixels = config.loss_config.image_size * config.loss_config.image_size;
    let particle_pixels = particle_pixel_positions_batch(x, config);
    let sigma = particle_pixel_size
        .clone()
        .unsqueeze_dim::<3>(1)
        .mul_scalar(config.loss_config.sigma * config.loss_config.image_size as f32)
        .div_scalar(config.loss_config.hi - config.loss_config.lo)
        .clamp_min(EPSILON);
    let denom = splat_particle_denominator_batch(
        &particle_pixels,
        particle_count,
        sigma.clone(),
        config,
    );
    let norm_scale = particle_pixel_size
        .mul_scalar(config.loss_config.image_size as f32)
        .div_scalar(config.loss_config.hi - config.loss_config.lo);
    let norm_scale = norm_scale
        .clone()
        .mul(norm_scale)
        .mul(particle_output_scale)
        .unsqueeze_dim::<3>(1);
    let chunk_size = splat_pixel_chunk_size(
        batches,
        particle_count,
        pixels,
        config.max_splat_chunk_floats,
    );
    let mut rgbs = Vec::new();
    let mut densities = Vec::new();
    for (start, len) in chunks_for(pixels, chunk_size) {
        let g = splat_gaussian_batch_chunk(
            &particle_pixels,
            targets,
            indices,
            particle_count,
            sigma.clone(),
            start,
            len,
        );
        let weights = g
            .div(denom.clone().expand([batches, len, particle_count]))
            .mul(norm_scale.clone().expand([batches, len, particle_count]));
        densities.push(weights.clone().sum_dim(2));
        rgbs.push(weights.matmul(colors.clone()));
    }
    (Tensor::cat(rgbs, 1), Tensor::cat(densities, 1))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn adaptive_uncentered_render_rgb_mse_batch(
    x: &Tensor3,
    s: &Tensor3,
    targets: &[BurnTargetExample],
    indices: &[usize],
    config: DirectBasisTrainConfig,
    particle_pixel_size: Tensor2,
    particle_output_scale: Tensor2,
) -> Tensor1 {
    let [batches, _, _] = x.shape().dims::<3>();
    let state_dims = s.shape().dims::<3>()[2];
    let colors = s.clone().narrow(2, state_dims - 3, 3).add_scalar(0.5);
    let (rgb, _) = adaptive_splat_render_batch(
        x,
        &colors,
        targets,
        indices,
        config,
        particle_pixel_size,
        particle_output_scale,
    );
    let pixels = config.loss_config.image_size * config.loss_config.image_size;
    let diff = rgb - stack_target_rgb(targets, indices);
    diff.clone()
        .mul(diff)
        .reshape([batches, pixels * 3])
        .mean_dim(1)
        .squeeze_dim::<1>(1)
}
