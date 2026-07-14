//! Target2D rendering, tiled adjoint dispatch, and differentiable loss terms.

use super::*;

    pub(super) fn background_density_term(density: Tensor2, foreground: Tensor2) -> Tensor2 {
        let background = foreground.mul_scalar(-1.0).add_scalar(1.0);
        let leak = density.mul(background);
        leak.clone().mul(leak)
    }

    pub(super) fn background_density_term_batch(density: Tensor3, foreground: Tensor3) -> Tensor3 {
        let background = foreground.mul_scalar(-1.0).add_scalar(1.0);
        let leak = density.mul(background);
        leak.clone().mul(leak)
    }

    pub(super) fn foreground_density_term(
        density: Tensor2,
        target_density: Tensor2,
        foreground: Tensor2,
        foreground_scale: f32,
    ) -> Tensor1 {
        l1l2_tensor((density - target_density).mul(foreground))
            .mean()
            .mul_scalar(foreground_scale)
    }

    pub(super) fn foreground_density_term_batch(
        density: Tensor3,
        target_density: Tensor3,
        foreground: Tensor3,
        foreground_scales: Tensor3,
    ) -> Tensor3 {
        l1l2_tensor3((density - target_density).mul(foreground))
            .mul(foreground_scales)
    }

    pub(super) fn target_splat_loss(
        x: &Tensor2,
        s: &Tensor2,
        target: &BurnTargetExample,
        config: DirectBasisTrainConfig,
        adapter: &BurnAdapterParams,
        displacement: Tensor1,
    ) -> BurnLossTensors {
        let particle_count = x.shape().dims::<2>()[0];
        let state_dims = s.shape().dims::<2>()[1];
        let centered = if config.loss_config.center {
            x.clone() - x.clone().mean_dim(0).expand([particle_count, 2])
                + target.target_mean.clone().expand([particle_count, 2])
        } else {
            x.clone()
        };
        let colors = s.clone().narrow(1, state_dims - 3, 3).add_scalar(0.5);
        let (rgb, density) = splat_render(&centered, &colors, target, config, particle_count);
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let predicted_alpha = density.clone().clamp_min(0.0).clamp_max(1.0);
        let target_alpha = target
            .target_density
            .clone()
            .clamp_min(0.0)
            .clamp_max(1.0);
        let predicted_composited_rgb = (rgb.clone()
            + predicted_alpha
                .mul_scalar(-1.0)
                .add_scalar(1.0)
                .expand([pixels, 3]))
        .clamp_min(0.0)
        .clamp_max(1.0);
        let target_composited_rgb = (target.target_rgb.clone()
            + target_alpha
                .mul_scalar(-1.0)
                .add_scalar(1.0)
                .expand([pixels, 3]))
        .clamp_min(0.0)
        .clamp_max(1.0);
        let composited_diff = predicted_composited_rgb - target_composited_rgb;
        let composited_rgb_loss = composited_diff.clone().mul(composited_diff).mean();
        let background_density_loss =
            background_density_term(density.clone(), target.target_foreground.clone()).mean();
        let foreground_density_loss = foreground_density_term(
            density.clone(),
            target.target_density.clone(),
            target.target_foreground.clone(),
            target.target_foreground_scale,
        );
        let density_diff = density - target.target_density.clone();
        let density_term = l1l2_tensor(density_diff);
        let density_loss = density_term.clone().mean();
        let color_gate = target_2d_detached_color_gate2(density_term).expand([
            config.loss_config.image_size * config.loss_config.image_size,
            3,
        ]);
        let color_loss = l1l2_tensor(rgb - target.target_rgb.clone())
            .mul(color_gate)
            .mean();
        let shape_chamfer_loss = target_shape_chamfer_loss(&centered, target, config);
        let splat = color_loss
            .clone()
            .mul_scalar(config.loss_config.color_loss_weight)
            + density_loss
                .clone()
                .mul_scalar(config.loss_config.density_loss_weight)
            + background_density_loss
                .clone()
                .mul_scalar(config.loss_config.background_density_loss_weight)
            + foreground_density_loss
                .clone()
                .mul_scalar(config.loss_config.foreground_density_loss_weight)
            + composited_rgb_loss.mul_scalar(config.loss_config.composited_rgb_loss_weight);
        let bound = relu(x.clone().abs().add_scalar(-1.0));
        let bound_loss = bound.mean();
        let overflow = relu(s.clone().abs().add_scalar(-1.0));
        let overflow_loss = overflow.mean();
        let mut total = splat
            .clone()
            .mul_scalar(config.loss_config.splat_loss_weight)
            + shape_chamfer_loss.mul_scalar(config.loss_config.shape_chamfer_loss_weight)
            + displacement.mul_scalar(config.loss_config.displacement_regularizer_weight)
            + bound_loss.mul_scalar(config.loss_config.bound_regularizer_weight)
            + overflow_loss.mul_scalar(config.loss_config.overflow_regularizer_weight);
        if config.adapter_l2_weight > 0.0 {
            total = total + adapter.l2_loss().mul_scalar(config.adapter_l2_weight);
        }
        BurnLossTensors {
            total,
            splat,
            color: color_loss,
            density: density_loss,
        }
    }

    pub(super) fn target_splat_loss_batch(
        x: &Tensor3,
        s: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        adapter: &BurnAdapterBatch,
        displacement: Tensor1,
    ) -> AutomataResult<BurnLossTensors> {
        let per_example = target_splat_loss_batch_vector_selected(
            x,
            s,
            targets,
            indices,
            config,
            adapter,
            displacement,
        )?;
        Ok(BurnLossTensors {
            total: per_example.total.mean(),
            splat: per_example.splat.mean(),
            color: per_example.color.mean(),
            density: per_example.density.mean(),
        })
    }

    pub(super) fn target_splat_loss_batch_vector(
        x: &Tensor3,
        s: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        adapter: &BurnAdapterBatch,
        displacement: Tensor1,
    ) -> BurnLossBatchTensors {
        let dims = x.shape().dims::<3>();
        let batches = dims[0];
        let particle_count = dims[1];
        let state_dims = s.shape().dims::<3>()[2];
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let target_mean = stack_target_mean(targets, indices);
        let target_rgb = stack_target_rgb(targets, indices);
        let target_density = stack_target_density(targets, indices);
        let target_foreground = stack_target_foreground(targets, indices);
        let target_foreground_scales = stack_target_foreground_scales(targets, indices);
        let centered = if config.loss_config.center {
            x.clone() - x.clone().mean_dim(1).expand([batches, particle_count, 2])
                + target_mean.expand([batches, particle_count, 2])
        } else {
            x.clone()
        };
        let colors = s.clone().narrow(2, state_dims - 3, 3).add_scalar(0.5);
        let (rgb, density) =
            splat_render_batch(&centered, &colors, targets, indices, config, particle_count);
        let predicted_alpha = density.clone().clamp_min(0.0).clamp_max(1.0);
        let target_alpha = target_density.clone().clamp_min(0.0).clamp_max(1.0);
        let predicted_composited_rgb = (rgb.clone()
            + predicted_alpha
                .mul_scalar(-1.0)
                .add_scalar(1.0)
                .expand([batches, pixels, 3]))
        .clamp_min(0.0)
        .clamp_max(1.0);
        let target_composited_rgb = (target_rgb.clone()
            + target_alpha
                .mul_scalar(-1.0)
                .add_scalar(1.0)
                .expand([batches, pixels, 3]))
        .clamp_min(0.0)
        .clamp_max(1.0);
        let composited_diff = predicted_composited_rgb - target_composited_rgb;
        let composited_rgb_loss = composited_diff
            .clone()
            .mul(composited_diff)
            .reshape([batches, pixels * 3])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let background_density_loss = background_density_term_batch(
            density.clone(),
            target_foreground.clone(),
        )
        .reshape([batches, pixels])
        .mean_dim(1)
        .squeeze_dim::<1>(1);
        let foreground_density_loss = foreground_density_term_batch(
            density.clone(),
            target_density.clone(),
            target_foreground,
            target_foreground_scales,
        )
        .reshape([batches, pixels])
        .mean_dim(1)
        .squeeze_dim::<1>(1);
        let density_diff = density - target_density;
        let density_term = l1l2_tensor3(density_diff);
        let density_loss = density_term
            .clone()
            .reshape([batches, pixels])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let color_gate = target_2d_detached_color_gate3(density_term).expand([batches, pixels, 3]);
        let color_loss = l1l2_tensor3(rgb - target_rgb)
            .mul(color_gate)
            .reshape([batches, pixels * 3])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let shape_chamfer_loss =
            target_shape_chamfer_loss_batch_vector(&centered, targets, indices, config);
        let splat = color_loss
            .clone()
            .mul_scalar(config.loss_config.color_loss_weight)
            + density_loss
                .clone()
                .mul_scalar(config.loss_config.density_loss_weight)
            + background_density_loss
                .clone()
                .mul_scalar(config.loss_config.background_density_loss_weight)
            + foreground_density_loss
                .clone()
                .mul_scalar(config.loss_config.foreground_density_loss_weight)
            + composited_rgb_loss.mul_scalar(config.loss_config.composited_rgb_loss_weight);
        let bound_loss = relu(x.clone().abs().add_scalar(-1.0))
            .reshape([batches, particle_count * 2])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let overflow_loss = relu(s.clone().abs().add_scalar(-1.0))
            .reshape([batches, particle_count * state_dims])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let mut total = splat
            .clone()
            .mul_scalar(config.loss_config.splat_loss_weight)
            + shape_chamfer_loss.mul_scalar(config.loss_config.shape_chamfer_loss_weight)
            + displacement.mul_scalar(config.loss_config.displacement_regularizer_weight)
            + bound_loss.mul_scalar(config.loss_config.bound_regularizer_weight)
            + overflow_loss.mul_scalar(config.loss_config.overflow_regularizer_weight);
        if config.adapter_l2_weight > 0.0 {
            total = total
                + adapter
                    .l2_loss_vector()
                    .mul_scalar(config.adapter_l2_weight);
        }
        BurnLossBatchTensors {
            total,
            splat,
            color: color_loss,
            density: density_loss,
        }
    }

    pub(super) fn target_splat_loss_batch_vector_selected(
        x: &Tensor3,
        s: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        adapter: &BurnAdapterBatch,
        displacement: Tensor1,
    ) -> AutomataResult<BurnLossBatchTensors> {
        match target2d_loss_backend_effective(config) {
            Target2dLossBackend::Dense => Ok(target_splat_loss_batch_vector(
                x,
                s,
                targets,
                indices,
                config,
                adapter,
                displacement,
            )),
            Target2dLossBackend::TiledAdjoint => target_splat_loss_batch_vector_tiled_adjoint(
                x,
                s,
                targets,
                indices,
                config,
                Some(adapter),
                displacement,
            ),
            Target2dLossBackend::Auto => unreachable!("auto target2d backend must be resolved"),
        }
    }

    pub(super) fn target2d_loss_backend_effective(config: DirectBasisTrainConfig) -> Target2dLossBackend {
        match config.target2d_loss_backend {
            Target2dLossBackend::Auto => target2d_loss_backend_auto(config),
            Target2dLossBackend::Dense => Target2dLossBackend::Dense,
            Target2dLossBackend::TiledAdjoint => Target2dLossBackend::TiledAdjoint,
        }
    }

    pub(super) fn target2d_loss_backend_auto(config: DirectBasisTrainConfig) -> Target2dLossBackend {
        if PERCEPTION_CUBE_ENABLED {
            if config.rollout_particles >= 128 {
                Target2dLossBackend::TiledAdjoint
            } else {
                Target2dLossBackend::Dense
            }
        } else {
            Target2dLossBackend::Dense
        }
    }

    #[cfg(any(feature = "backend_wgpu", feature = "backend_cuda"))]
    pub(super) fn target2d_cube_loss_config(value: crate::Target2dLossConfig) -> Target2dCubeLossConfig {
        Target2dCubeLossConfig {
            image_size: value.image_size,
            sigma: value.sigma,
            lo: value.lo,
            hi: value.hi,
            splat_loss_weight: value.splat_loss_weight,
            color_loss_weight: value.color_loss_weight,
            density_loss_weight: value.density_loss_weight,
            background_density_loss_weight: value.background_density_loss_weight,
            foreground_density_loss_weight: value.foreground_density_loss_weight,
            composited_rgb_loss_weight: value.composited_rgb_loss_weight,
            center: value.center,
        }
    }

    pub(super) fn target_splat_loss_batch_vector_tiled_adjoint(
        x: &Tensor3,
        s: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        adapter: Option<&BurnAdapterBatch>,
        displacement: Tensor1,
    ) -> AutomataResult<BurnLossBatchTensors> {
        if indices.is_empty() {
            return Err(AutomataError::InvalidArgument(
                "tiled target2d loss requires at least one target index".to_string(),
            ));
        }
        let dims = x.shape().dims::<3>();
        let batches = dims[0];
        let particle_count = dims[1];
        let state_dims = s.shape().dims::<3>()[2];
        if batches != indices.len() {
            return Err(AutomataError::InvalidArgument(format!(
                "tiled target2d batch mismatch: tensor batch={batches} indices={}",
                indices.len()
            )));
        }
        let device = &x.device();
        #[cfg(any(feature = "backend_wgpu", feature = "backend_cuda"))]
        {
          if config.loss_config.shape_chamfer_loss_weight == 0.0 {
            let target_mean = config
                .loss_config
                .center
                .then(|| stack_target_mean(targets, indices).inner());
            let target_rgb = stack_target_rgb(targets, indices).inner();
            let target_density = stack_target_density(targets, indices).inner();
            let target_foreground = stack_target_foreground(targets, indices).inner();
            let target_foreground_scales = stack_target_foreground_scales(targets, indices).inner();
            let pixel_sizes = stack_pixel_sizes(targets, indices).inner();
            let target_point_counts = stack_target_point_counts(targets, indices).inner();
            if let Some(device_loss) = InnerBackend::target2d_cube_adjoint(
                x.clone().inner(),
                {
                    let x_inner = x.clone().inner();
                    if let Some(target_mean) = target_mean {
                        x_inner.clone()
                            - x_inner.clone().mean_dim(1).expand([batches, particle_count, 2])
                            + target_mean.expand([batches, particle_count, 2])
                    } else {
                        x_inner
                    }
                },
                s.clone().inner(),
                target_rgb,
                target_density,
                target_foreground,
                target_foreground_scales,
                pixel_sizes,
                target_point_counts,
                target2d_cube_loss_config(config.loss_config),
            ) {
                TARGET2D_CUBE_ADJOINT_DEVICE_HITS.fetch_add(1, Ordering::Relaxed);
                let device_loss = device_loss?;
                let position_grad = Tensor::<BurnBackend, 3>::from_inner(device_loss.position_grad);
                let state_grad = Tensor::<BurnBackend, 3>::from_inner(device_loss.state_grad);
                let position_term = x
                    .clone()
                    .mul(position_grad)
                    .reshape([batches, particle_count * 2])
                    .sum_dim(1)
                    .squeeze_dim::<1>(1);
                let state_term = s
                    .clone()
                    .mul(state_grad)
                    .reshape([batches, particle_count * state_dims])
                    .sum_dim(1)
                    .squeeze_dim::<1>(1);
                let mut total = position_term
                    + state_term
                    + Tensor::<BurnBackend, 1>::from_inner(device_loss.constant);
                if config.loss_config.bound_regularizer_weight > 0.0 {
                    let bound_loss = relu(x.clone().abs().add_scalar(-1.0))
                        .reshape([batches, particle_count * 2])
                        .mean_dim(1)
                        .squeeze_dim::<1>(1);
                    total = total
                        + bound_loss.mul_scalar(config.loss_config.bound_regularizer_weight);
                }
                if config.loss_config.overflow_regularizer_weight > 0.0 {
                    let overflow_loss = relu(s.clone().abs().add_scalar(-1.0))
                        .reshape([batches, particle_count * state_dims])
                        .mean_dim(1)
                        .squeeze_dim::<1>(1);
                    total = total
                        + overflow_loss.mul_scalar(config.loss_config.overflow_regularizer_weight);
                }
                if config.loss_config.displacement_regularizer_weight > 0.0 {
                    total = total
                        + displacement.mul_scalar(config.loss_config.displacement_regularizer_weight);
                }
                if config.adapter_l2_weight > 0.0 {
                    let adapter = adapter.expect(
                        "adapter L2 regularization requires adapter parameters",
                    );
                    total = total
                        + adapter
                            .l2_loss_vector()
                            .mul_scalar(config.adapter_l2_weight);
                }
                return Ok(BurnLossBatchTensors {
                    total,
                    splat: Tensor::<BurnBackend, 1>::from_inner(device_loss.splat),
                    color: Tensor::<BurnBackend, 1>::from_inner(device_loss.color),
                    density: Tensor::<BurnBackend, 1>::from_inner(device_loss.density),
                });
            }
          }
        }
        TARGET2D_CUBE_ADJOINT_FALLBACK_HITS.fetch_add(1, Ordering::Relaxed);
        let x_values = tensor3_vec(x.clone().inner())?;
        let s_values = tensor3_vec(s.clone().inner())?;
        let displacement_values = tensor1_vec(displacement.clone().inner())?;
        if displacement_values.len() != batches {
            return Err(AutomataError::InvalidArgument(format!(
                "tiled target2d displacement batch mismatch: displacement={} batches={batches}",
                displacement_values.len()
            )));
        }

        let mut position_grad_values = vec![0.0_f32; batches * particle_count * 2];
        let mut state_grad_values = vec![0.0_f32; batches * particle_count * state_dims];
        let mut total_values = Vec::with_capacity(batches);
        let mut splat_values = Vec::with_capacity(batches);
        let mut color_values = Vec::with_capacity(batches);
        let mut density_values = Vec::with_capacity(batches);
        let mut constant_values = Vec::with_capacity(batches);

        for (batch, target_idx) in indices.iter().copied().enumerate() {
            let target = targets.get(target_idx).ok_or_else(|| {
                AutomataError::InvalidArgument(format!(
                    "tiled target2d target index {target_idx} is out of bounds"
                ))
            })?;
            let x_offset = batch * particle_count * 2;
            let s_offset = batch * particle_count * state_dims;
            let mut positions = Vec::with_capacity(particle_count);
            for particle in 0..particle_count {
                let base = x_offset + particle * 2;
                positions.push([x_values[base], x_values[base + 1], 0.0, 0.0]);
            }
            let states =
                &s_values[s_offset..s_offset + particle_count.saturating_mul(state_dims)];
            let reference = crate::target_2d_loss_with_adjoint(
                &positions,
                states,
                1,
                particle_count,
                state_dims,
                &target.target_cpu,
                config.loss_config,
                0.0,
                0,
            )?;

            let total = reference.report.total_loss;
            total_values.push(total);
            splat_values.push(reference.report.splat_loss);
            color_values.push(reference.report.color_loss);
            density_values.push(reference.report.density_loss);

            let mut dot = 0.0_f32;
            for particle in 0..particle_count {
                let pos_base = x_offset + particle * 2;
                let reference_position = reference.position_gradients[particle];
                position_grad_values[pos_base] = reference_position[0];
                position_grad_values[pos_base + 1] = reference_position[1];
                dot += x_values[pos_base] * reference_position[0]
                    + x_values[pos_base + 1] * reference_position[1];

                let state_base = s_offset + particle * state_dims;
                let reference_state = &reference.state_gradients
                    [particle * state_dims..(particle + 1) * state_dims];
                for dim in 0..state_dims {
                    state_grad_values[state_base + dim] = reference_state[dim];
                    dot += s_values[state_base + dim] * reference_state[dim];
                }
            }
            constant_values.push(total - dot);
        }

        let position_grad = tensor3(position_grad_values, [batches, particle_count, 2], device);
        let state_grad = tensor3(
            state_grad_values,
            [batches, particle_count, state_dims],
            device,
        );
        let position_term = x
            .clone()
            .mul(position_grad)
            .reshape([batches, particle_count * 2])
            .sum_dim(1)
            .squeeze_dim::<1>(1);
        let state_term = s
            .clone()
            .mul(state_grad)
            .reshape([batches, particle_count * state_dims])
            .sum_dim(1)
            .squeeze_dim::<1>(1);
        let mut total = position_term + state_term + tensor1(constant_values, [batches], device);
        if config.loss_config.displacement_regularizer_weight > 0.0 {
            total = total
                + displacement.mul_scalar(config.loss_config.displacement_regularizer_weight);
        }
        if config.adapter_l2_weight > 0.0 {
            let adapter = adapter.expect("adapter L2 regularization requires adapter parameters");
            total = total
                + adapter
                    .l2_loss_vector()
                    .mul_scalar(config.adapter_l2_weight);
        }

        Ok(BurnLossBatchTensors {
            total,
            splat: tensor1(splat_values, [batches], device),
            color: tensor1(color_values, [batches], device),
            density: tensor1(density_values, [batches], device),
        })
    }

    pub(super) fn target_splat_quality_batch_vector(
        x: &Tensor3,
        s: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        adapter: &BurnAdapterBatch,
        displacement: Tensor1,
    ) -> BurnE2eQualityBatchTensors {
        let dims = x.shape().dims::<3>();
        let batches = dims[0];
        let particle_count = dims[1];
        let state_dims = s.shape().dims::<3>()[2];
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let target_mean = stack_target_mean(targets, indices);
        let target_rgb = stack_target_rgb(targets, indices);
        let target_density = stack_target_density(targets, indices);
        let target_foreground = stack_target_foreground(targets, indices);
        let target_foreground_scales = stack_target_foreground_scales(targets, indices);
        let centered = if config.loss_config.center {
            x.clone() - x.clone().mean_dim(1).expand([batches, particle_count, 2])
                + target_mean.expand([batches, particle_count, 2])
        } else {
            x.clone()
        };
        let colors = s.clone().narrow(2, state_dims - 3, 3).add_scalar(0.5);
        let (rgb, density) =
            splat_render_batch(&centered, &colors, targets, indices, config, particle_count);
        let rgb_diff = rgb.clone() - target_rgb.clone();
        let render_rgb_mse = rgb_diff
            .clone()
            .mul(rgb_diff.clone())
            .reshape([batches, pixels * 3])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let predicted_alpha = density.clone().clamp_min(0.0).clamp_max(1.0);
        let target_alpha = target_density.clone().clamp_min(0.0).clamp_max(1.0);
        let render_occupancy = predicted_alpha
            .clone()
            .reshape([batches, pixels])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let position_overflow_fraction = x
            .clone()
            .abs()
            .greater_elem(1.0)
            .float()
            .sum_dim(2)
            .greater_elem(0.0)
            .float()
            .reshape([batches, particle_count])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let state_overflow_fraction = s
            .clone()
            .abs()
            .greater_elem(1.0)
            .float()
            .reshape([batches, particle_count * state_dims])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let density_diff_for_metrics = predicted_alpha.clone() - target_alpha.clone();
        let density_mse = density_diff_for_metrics
            .clone()
            .mul(density_diff_for_metrics)
            .reshape([batches, pixels])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let predicted_straight_rgb = rgb
            .clone()
            .div(
                density
                    .clone()
                    .clamp_min(EPSILON)
                    .expand([batches, pixels, 3]),
            )
            .clamp_min(0.0)
            .clamp_max(1.0);
        let target_straight_rgb = target_rgb
            .clone()
            .div(
                target_density
                    .clone()
                    .clamp_min(EPSILON)
                    .expand([batches, pixels, 3]),
            )
            .clamp_min(0.0)
            .clamp_max(1.0);
        let foreground_rgb_diff = predicted_straight_rgb - target_straight_rgb;
        let foreground_rgb_squared = foreground_rgb_diff
            .clone()
            .mul(foreground_rgb_diff)
            .mul(target_foreground.clone().expand([batches, pixels, 3]))
            .reshape([batches, pixels * 3])
            .sum_dim(1)
            .squeeze_dim::<1>(1);
        let foreground_rgb_denominator = target_foreground
            .clone()
            .reshape([batches, pixels])
            .sum_dim(1)
            .squeeze_dim::<1>(1)
            .mul_scalar(3.0)
            .clamp_min(EPSILON);
        let foreground_rgb_mse = foreground_rgb_squared.div(foreground_rgb_denominator);
        let predicted_composited_rgb = (rgb.clone()
            + predicted_alpha
                .clone()
                .mul_scalar(-1.0)
                .add_scalar(1.0)
                .expand([batches, pixels, 3]))
        .clamp_min(0.0)
        .clamp_max(1.0);
        let target_composited_rgb = (target_rgb.clone()
            + target_alpha
                .clone()
                .mul_scalar(-1.0)
                .add_scalar(1.0)
                .expand([batches, pixels, 3]))
        .clamp_min(0.0)
        .clamp_max(1.0);
        let composited_rgb_diff = predicted_composited_rgb - target_composited_rgb;
        let composited_rgb_mse = composited_rgb_diff
            .clone()
            .mul(composited_rgb_diff)
            .reshape([batches, pixels * 3])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let density_intersection = predicted_alpha
            .clone()
            .min_pair(target_alpha.clone())
            .reshape([batches, pixels])
            .sum_dim(1)
            .squeeze_dim::<1>(1);
        let density_union = predicted_alpha
            .max_pair(target_alpha)
            .reshape([batches, pixels])
            .sum_dim(1)
            .squeeze_dim::<1>(1)
            .clamp_min(EPSILON);
        let density_soft_iou = density_intersection.div(density_union);
        let background_density_loss = background_density_term_batch(
            density.clone(),
            target_foreground.clone(),
        )
        .reshape([batches, pixels])
        .mean_dim(1)
        .squeeze_dim::<1>(1);
        let foreground_density_loss = foreground_density_term_batch(
            density.clone(),
            target_density.clone(),
            target_foreground,
            target_foreground_scales,
        )
        .reshape([batches, pixels])
        .mean_dim(1)
        .squeeze_dim::<1>(1);
        let density_diff = density - target_density;
        let density_term = l1l2_tensor3(density_diff);
        let density_loss = density_term
            .clone()
            .reshape([batches, pixels])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let color_gate = target_2d_detached_color_gate3(density_term).expand([batches, pixels, 3]);
        let color_loss = l1l2_tensor3(rgb_diff)
            .mul(color_gate)
            .reshape([batches, pixels * 3])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let shape_chamfer_loss =
            target_shape_chamfer_loss_batch_vector(&centered, targets, indices, config);
        let splat = color_loss
            .clone()
            .mul_scalar(config.loss_config.color_loss_weight)
            + density_loss
                .clone()
                .mul_scalar(config.loss_config.density_loss_weight)
            + background_density_loss
                .clone()
                .mul_scalar(config.loss_config.background_density_loss_weight)
            + foreground_density_loss
                .clone()
                .mul_scalar(config.loss_config.foreground_density_loss_weight);
        let bound_loss = relu(x.clone().abs().add_scalar(-1.0))
            .reshape([batches, particle_count * 2])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let overflow_loss = relu(s.clone().abs().add_scalar(-1.0))
            .reshape([batches, particle_count * state_dims])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let mut total = splat
            .clone()
            .mul_scalar(config.loss_config.splat_loss_weight)
            + shape_chamfer_loss.mul_scalar(config.loss_config.shape_chamfer_loss_weight)
            + displacement.mul_scalar(config.loss_config.displacement_regularizer_weight)
            + bound_loss.mul_scalar(config.loss_config.bound_regularizer_weight)
            + overflow_loss.mul_scalar(config.loss_config.overflow_regularizer_weight);
        if config.adapter_l2_weight > 0.0 {
            total = total
                + adapter
                    .l2_loss_vector()
                    .mul_scalar(config.adapter_l2_weight);
        }
        BurnE2eQualityBatchTensors {
            loss: BurnLossBatchTensors {
                total,
                splat,
                color: color_loss,
                density: density_loss,
            },
            adapter_vector: None,
            render_rgb_mse,
            composited_rgb_mse,
            foreground_rgb_mse,
            density_mse,
            density_soft_iou,
            render_occupancy,
            position_overflow_fraction,
            state_overflow_fraction,
        }
    }

    pub(super) fn target_splat_loss_batch_vector_base_only_selected(
        x: &Tensor3,
        s: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        displacement: Tensor1,
    ) -> AutomataResult<BurnLossBatchTensors> {
        match target2d_loss_backend_effective(config) {
            Target2dLossBackend::Dense => Ok(target_splat_loss_batch_vector_base_only_dense(
                x,
                s,
                targets,
                indices,
                config,
                displacement,
            )),
            Target2dLossBackend::TiledAdjoint => target_splat_loss_batch_vector_tiled_adjoint(
                x,
                s,
                targets,
                indices,
                config,
                None,
                displacement,
            ),
            Target2dLossBackend::Auto => unreachable!("auto target2d backend must be resolved"),
        }
    }

    fn target_splat_loss_batch_vector_base_only_dense(
        x: &Tensor3,
        s: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        displacement: Tensor1,
    ) -> BurnLossBatchTensors {
        let dims = x.shape().dims::<3>();
        let batches = dims[0];
        let particle_count = dims[1];
        let state_dims = s.shape().dims::<3>()[2];
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let target_mean = stack_target_mean(targets, indices);
        let target_rgb = stack_target_rgb(targets, indices);
        let target_density = stack_target_density(targets, indices);
        let target_foreground = stack_target_foreground(targets, indices);
        let target_foreground_scales = stack_target_foreground_scales(targets, indices);
        let centered = if config.loss_config.center {
            x.clone() - x.clone().mean_dim(1).expand([batches, particle_count, 2])
                + target_mean.expand([batches, particle_count, 2])
        } else {
            x.clone()
        };
        let colors = s.clone().narrow(2, state_dims - 3, 3).add_scalar(0.5);
        let (rgb, density) =
            splat_render_batch(&centered, &colors, targets, indices, config, particle_count);
        let background_density_loss = background_density_term_batch(
            density.clone(),
            target_foreground.clone(),
        )
        .reshape([batches, pixels])
        .mean_dim(1)
        .squeeze_dim::<1>(1);
        let foreground_density_loss = foreground_density_term_batch(
            density.clone(),
            target_density.clone(),
            target_foreground,
            target_foreground_scales,
        )
        .reshape([batches, pixels])
        .mean_dim(1)
        .squeeze_dim::<1>(1);
        let density_diff = density - target_density;
        let density_term = l1l2_tensor3(density_diff);
        let density_loss = density_term
            .clone()
            .reshape([batches, pixels])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let color_gate = target_2d_detached_color_gate3(density_term).expand([batches, pixels, 3]);
        let color_loss = l1l2_tensor3(rgb - target_rgb)
            .mul(color_gate)
            .reshape([batches, pixels * 3])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let shape_chamfer_loss =
            target_shape_chamfer_loss_batch_vector(&centered, targets, indices, config);
        let splat = color_loss
            .clone()
            .mul_scalar(config.loss_config.color_loss_weight)
            + density_loss
                .clone()
                .mul_scalar(config.loss_config.density_loss_weight)
            + background_density_loss
                .clone()
                .mul_scalar(config.loss_config.background_density_loss_weight)
            + foreground_density_loss
                .clone()
                .mul_scalar(config.loss_config.foreground_density_loss_weight);
        let bound_loss = relu(x.clone().abs().add_scalar(-1.0))
            .reshape([batches, particle_count * 2])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let overflow_loss = relu(s.clone().abs().add_scalar(-1.0))
            .reshape([batches, particle_count * state_dims])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let total = splat
            .clone()
            .mul_scalar(config.loss_config.splat_loss_weight)
            + shape_chamfer_loss.mul_scalar(config.loss_config.shape_chamfer_loss_weight)
            + displacement.mul_scalar(config.loss_config.displacement_regularizer_weight)
            + bound_loss.mul_scalar(config.loss_config.bound_regularizer_weight)
            + overflow_loss.mul_scalar(config.loss_config.overflow_regularizer_weight);
        BurnLossBatchTensors {
            total,
            splat,
            color: color_loss,
            density: density_loss,
        }
    }

    pub(super) fn target_shape_chamfer_loss(
        x: &Tensor2,
        target: &BurnTargetExample,
        config: DirectBasisTrainConfig,
    ) -> Tensor1 {
        if config.loss_config.shape_chamfer_loss_weight <= 0.0 {
            return Tensor::<BurnBackend, 1>::zeros([1], &target.target_rgb.device());
        }
        let particle_count = x.shape().dims::<2>()[0];
        let target_count = target.target_positions.shape().dims::<2>()[0];
        if particle_count == 0 || target_count == 0 {
            return Tensor::<BurnBackend, 1>::zeros([1], &target.target_rgb.device());
        }
        let particle_i = x
            .clone()
            .unsqueeze_dim::<3>(1)
            .expand([particle_count, target_count, 2]);
        let target_j = target
            .target_positions
            .clone()
            .unsqueeze_dim::<3>(0)
            .expand([particle_count, target_count, 2]);
        let diff = particle_i - target_j;
        let dist2 = diff.clone().mul(diff).sum_dim(2).squeeze_dim::<2>(2);
        let particle_to_target = dist2.clone().min_dim(1).mean();
        let target_to_particle = dist2.min_dim(0).mean();
        particle_to_target + target_to_particle
    }

    pub(super) fn target_shape_chamfer_loss_batch_vector(
        x: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
    ) -> Tensor1 {
        if config.loss_config.shape_chamfer_loss_weight <= 0.0 {
            return Tensor::<BurnBackend, 1>::zeros(
                [indices.len()],
                &targets[indices[0]].target_rgb.device(),
            );
        }
        Tensor::cat(
            indices
                .iter()
                .enumerate()
                .map(|(local, idx)| {
                    let x_local = x.clone().narrow(0, local, 1).squeeze_dim::<2>(0);
                    target_shape_chamfer_loss(&x_local, &targets[*idx], config)
                })
                .collect::<Vec<_>>(),
            0,
        )
    }

    pub(super) fn splat_render(
        x: &Tensor2,
        colors: &Tensor2,
        target: &BurnTargetExample,
        config: DirectBasisTrainConfig,
        particle_count: usize,
    ) -> (Tensor2, Tensor2) {
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let particle_pixels = particle_pixel_positions(x, config);
        let sigma =
            (config.loss_config.sigma * config.loss_config.image_size as f32 * target.pixel_size
                / (config.loss_config.hi - config.loss_config.lo))
                .max(EPSILON);
        let denom =
            splat_particle_denominator(&particle_pixels, target, particle_count, sigma, config);
        let norm_scale = (config.loss_config.image_size as f32 * target.pixel_size
            / (config.loss_config.hi - config.loss_config.lo))
            .powi(2);
        let output_scale = target.target_points as f32 / particle_count.max(1) as f32;
        let chunk_size =
            splat_pixel_chunk_size(1, particle_count, pixels, config.max_splat_chunk_floats);
        let mut rgbs = Vec::new();
        let mut densities = Vec::new();
        for (start, len) in chunks_for(pixels, chunk_size) {
            let g =
                splat_gaussian_chunk(&particle_pixels, target, particle_count, sigma, start, len);
            let weights = g
                .div(denom.clone().expand([len, particle_count]))
                .mul_scalar(output_scale * norm_scale);
            densities.push(weights.clone().sum_dim(1));
            rgbs.push(weights.matmul(colors.clone()));
        }
        (Tensor::cat(rgbs, 0), Tensor::cat(densities, 0))
    }

    pub(super) fn splat_render_batch(
        x: &Tensor3,
        colors: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        particle_count: usize,
    ) -> (Tensor3, Tensor3) {
        let batches = indices.len();
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let particle_pixels = particle_pixel_positions_batch(x, config);
        let pixel_sizes = stack_pixel_sizes(targets, indices);
        let sigma = pixel_sizes
            .clone()
            .mul_scalar(config.loss_config.sigma * config.loss_config.image_size as f32)
            .div_scalar(config.loss_config.hi - config.loss_config.lo)
            .clamp_min(EPSILON);
        let denom = splat_particle_denominator_batch(
            &particle_pixels,
            targets,
            indices,
            particle_count,
            sigma.clone(),
            config,
        );
        let norm_scale = pixel_sizes
            .mul_scalar(config.loss_config.image_size as f32)
            .div_scalar(config.loss_config.hi - config.loss_config.lo);
        let norm_scale = norm_scale.clone().mul(norm_scale);
        let output_scale =
            stack_target_point_counts(targets, indices).div_scalar(particle_count.max(1) as f32);
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
                .mul(norm_scale.clone().expand([batches, len, particle_count]))
                .mul(output_scale.clone().expand([batches, len, particle_count]));
            densities.push(weights.clone().sum_dim(2));
            rgbs.push(weights.matmul(colors.clone()));
        }
        (Tensor::cat(rgbs, 1), Tensor::cat(densities, 1))
    }

    pub(super) fn target_point_splat_composited_mses(
        targets: &[BurnTargetExample],
        config: DirectBasisTrainConfig,
        requested_batch_size: usize,
        device: &BurnDevice,
    ) -> AutomataResult<Vec<f32>> {
        let particle_count = config.rollout_particles.max(1);
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let batch_size = requested_batch_size.max(1).min(targets.len().max(1));
        let mut output = Vec::with_capacity(targets.len());
        for start in (0..targets.len()).step_by(batch_size) {
            let end = (start + batch_size).min(targets.len());
            let indices = (start..end).collect::<Vec<_>>();
            let mut positions = Vec::with_capacity(indices.len() * particle_count * 2);
            let mut colors = Vec::with_capacity(indices.len() * particle_count * 3);
            for &target_index in &indices {
                let target = &targets[target_index].target_cpu;
                let target_points = target.point_count();
                debug_assert!(target_points > 0);
                for particle in 0..particle_count {
                    let point = if particle_count <= target_points {
                        particle.saturating_mul(target_points) / particle_count
                    } else {
                        particle % target_points
                    }
                    .min(target_points - 1);
                    positions.extend_from_slice(&target.positions[point]);
                    colors.extend_from_slice(&target.colors[point]);
                }
            }
            let positions = tensor3(
                positions,
                [indices.len(), particle_count, 2],
                device,
            );
            let colors = tensor3(colors, [indices.len(), particle_count, 3], device);
            let (rgb, density) = splat_render_batch(
                &positions,
                &colors,
                targets,
                &indices,
                config,
                particle_count,
            );
            let predicted_alpha = density.clamp_min(0.0).clamp_max(1.0);
            let target_alpha = stack_target_density(targets, &indices)
                .clamp_min(0.0)
                .clamp_max(1.0);
            let predicted_composited_rgb = (rgb
                + predicted_alpha
                    .mul_scalar(-1.0)
                    .add_scalar(1.0)
                    .expand([indices.len(), pixels, 3]))
            .clamp_min(0.0)
            .clamp_max(1.0);
            let target_composited_rgb = (stack_target_rgb(targets, &indices)
                + target_alpha
                    .mul_scalar(-1.0)
                    .add_scalar(1.0)
                    .expand([indices.len(), pixels, 3]))
            .clamp_min(0.0)
            .clamp_max(1.0);
            let diff = predicted_composited_rgb - target_composited_rgb;
            let mses = diff
                .clone()
                .mul(diff)
                .reshape([indices.len(), pixels * 3])
                .mean_dim(1)
                .squeeze_dim::<1>(1);
            for mse in tensor1_vec(mses.inner())? {
                output.push(finite_scalar(
                    "HyperNPA target-point splat composited RGB MSE",
                    mse,
                )?);
            }
        }
        Ok(output)
    }

    pub(super) fn dense_particle_density(x: &Tensor2, config: DirectBasisTrainConfig) -> Tensor2 {
        let rows = x.shape().dims::<2>()[0];
        let chunk_size = dense_query_chunk_size(1, rows, 1, config.max_dense_chunk_floats);
        let mut chunks = Vec::new();
        for (start, len) in chunks_for(rows, chunk_size) {
            let xi = x
                .clone()
                .narrow(0, start, len)
                .unsqueeze_dim::<3>(1)
                .expand([len, rows, 2]);
            let xj = x.clone().unsqueeze_dim::<3>(0).expand([len, rows, 2]);
            let diff = xj - xi;
            let dist2 = diff.clone().mul(diff).sum_dim(2).squeeze_dim::<2>(2);
            let eps = config.grid_eps.max(EPSILON);
            let compact = relu(dist2.mul_scalar(-1.0).add_scalar(eps * eps));
            let compact2 = compact.clone().mul(compact.clone());
            chunks.push(
                compact2
                    .mul(compact)
                    .mul_scalar(4.0 / (std::f32::consts::PI * eps.powi(8)))
                    .sum_dim(1)
                    .clamp_min(EPSILON),
            );
        }
        Tensor::cat(chunks, 0)
    }

    pub(super) fn dense_particle_density_batch(x: &Tensor3, config: DirectBasisTrainConfig) -> Tensor3 {
        let dims = x.shape().dims::<3>();
        let batches = dims[0];
        let rows = dims[1];
        let chunk_size = dense_query_chunk_size(batches, rows, 1, config.max_dense_chunk_floats);
        let mut chunks = Vec::new();
        for (start, len) in chunks_for(rows, chunk_size) {
            let xi = x
                .clone()
                .narrow(1, start, len)
                .unsqueeze_dim::<4>(2)
                .expand([batches, len, rows, 2]);
            let xj = x
                .clone()
                .unsqueeze_dim::<4>(1)
                .expand([batches, len, rows, 2]);
            let diff = xj - xi;
            let dist2 = diff.clone().mul(diff).sum_dim(3).squeeze_dim::<3>(3);
            let eps = config.grid_eps.max(EPSILON);
            let compact = relu(dist2.mul_scalar(-1.0).add_scalar(eps * eps));
            let compact2 = compact.clone().mul(compact.clone());
            chunks.push(
                compact2
                    .mul(compact)
                    .mul_scalar(4.0 / (std::f32::consts::PI * eps.powi(8)))
                    .sum_dim(2)
                    .clamp_min(EPSILON),
            );
        }
        Tensor::cat(chunks, 1)
    }

    pub(super) fn dense_particle_density_batch_generic<B: burn::tensor::backend::Backend>(
        x: &Tensor<B, 3>,
        config: DirectBasisTrainConfig,
    ) -> Tensor<B, 3> {
        let dims = x.shape().dims::<3>();
        let batches = dims[0];
        let rows = dims[1];
        let chunk_size = dense_query_chunk_size(batches, rows, 1, config.max_dense_chunk_floats);
        let mut chunks = Vec::new();
        for (start, len) in chunks_for(rows, chunk_size) {
            let xi = x
                .clone()
                .narrow(1, start, len)
                .unsqueeze_dim::<4>(2)
                .expand([batches, len, rows, 2]);
            let xj = x
                .clone()
                .unsqueeze_dim::<4>(1)
                .expand([batches, len, rows, 2]);
            let diff = xj - xi;
            let dist2 = diff.clone().mul(diff).sum_dim(3).squeeze_dim::<3>(3);
            let eps = config.grid_eps.max(EPSILON);
            let compact = relu(dist2.mul_scalar(-1.0).add_scalar(eps * eps));
            let compact2 = compact.clone().mul(compact.clone());
            chunks.push(
                compact2
                    .mul(compact)
                    .mul_scalar(4.0 / (std::f32::consts::PI * eps.powi(8)))
                    .sum_dim(2)
                    .clamp_min(EPSILON),
            );
        }
        Tensor::cat(chunks, 1)
    }

    pub(super) fn splat_particle_denominator(
        particle_pixels: &Tensor2,
        target: &BurnTargetExample,
        particle_count: usize,
        sigma: f32,
        config: DirectBasisTrainConfig,
    ) -> Tensor2 {
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let chunk_size =
            splat_pixel_chunk_size(1, particle_count, pixels, config.max_splat_chunk_floats);
        let mut denom = None::<Tensor2>;
        for (start, len) in chunks_for(pixels, chunk_size) {
            let g =
                splat_gaussian_chunk(particle_pixels, target, particle_count, sigma, start, len);
            let contribution = g.sum_dim(0);
            denom = Some(match denom {
                Some(value) => value + contribution,
                None => contribution,
            });
        }
        denom
            .unwrap_or_else(|| {
                Tensor::<BurnBackend, 2>::zeros([1, particle_count], &target.target_rgb.device())
            })
            .add_scalar(EPSILON)
    }

    pub(super) fn splat_particle_denominator_batch(
        particle_pixels: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        sigma: Tensor3,
        config: DirectBasisTrainConfig,
    ) -> Tensor3 {
        let batches = indices.len();
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let chunk_size = splat_pixel_chunk_size(
            batches,
            particle_count,
            pixels,
            config.max_splat_chunk_floats,
        );
        let mut denom = None::<Tensor3>;
        for (start, len) in chunks_for(pixels, chunk_size) {
            let g = splat_gaussian_batch_chunk(
                particle_pixels,
                targets,
                indices,
                particle_count,
                sigma.clone(),
                start,
                len,
            );
            let contribution = g.sum_dim(1);
            denom = Some(match denom {
                Some(value) => value + contribution,
                None => contribution,
            });
        }
        denom
            .unwrap_or_else(|| {
                Tensor::<BurnBackend, 3>::zeros(
                    [batches, 1, particle_count],
                    &targets[indices[0]].target_rgb.device(),
                )
            })
            .add_scalar(EPSILON)
    }

    pub(super) fn splat_gaussian_chunk(
        particle_pixels: &Tensor2,
        target: &BurnTargetExample,
        particle_count: usize,
        sigma: f32,
        start: usize,
        len: usize,
    ) -> Tensor2 {
        let pixel_i = target
            .pixel_xy
            .clone()
            .narrow(0, start, len)
            .unsqueeze_dim::<3>(1)
            .expand([len, particle_count, 2]);
        let particle_j =
            particle_pixels
                .clone()
                .unsqueeze_dim::<3>(0)
                .expand([len, particle_count, 2]);
        let diff = pixel_i - particle_j;
        let dist2 = diff.clone().mul(diff).sum_dim(2).squeeze_dim::<2>(2);
        dist2.mul_scalar(-0.5 / (sigma * sigma)).exp()
    }

    pub(super) fn splat_gaussian_batch_chunk(
        particle_pixels: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        sigma: Tensor3,
        start: usize,
        len: usize,
    ) -> Tensor3 {
        let batches = indices.len();
        let pixel_i = targets[indices[0]]
            .pixel_xy
            .clone()
            .narrow(0, start, len)
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(2)
            .expand([batches, len, particle_count, 2]);
        let particle_j =
            particle_pixels
                .clone()
                .unsqueeze_dim::<4>(1)
                .expand([batches, len, particle_count, 2]);
        let diff = pixel_i - particle_j;
        let dist2 = diff.clone().mul(diff).sum_dim(3).squeeze_dim::<3>(3);
        let sigma2 = sigma
            .clone()
            .mul(sigma)
            .expand([batches, len, particle_count]);
        dist2.mul_scalar(-0.5).div(sigma2).exp()
    }

    pub(super) fn chunks_for(total: usize, chunk_size: usize) -> impl Iterator<Item = (usize, usize)> {
        let chunk_size = chunk_size.max(1);
        (0..total)
            .step_by(chunk_size)
            .map(move |start| (start, (total - start).min(chunk_size)))
    }

    pub(super) fn dense_query_chunk_size(
        batches: usize,
        rows: usize,
        state_dims: usize,
        max_floats: usize,
    ) -> usize {
        let denominator = batches
            .max(1)
            .saturating_mul(rows.max(1))
            .saturating_mul(state_dims.max(1))
            .saturating_mul(2)
            .max(1);
        (max_floats / denominator).max(1).min(rows.max(1))
    }

    pub(super) fn splat_pixel_chunk_size(
        batches: usize,
        particle_count: usize,
        pixels: usize,
        max_floats: usize,
    ) -> usize {
        let denominator = batches
            .max(1)
            .saturating_mul(particle_count.max(1))
            .saturating_mul(2)
            .max(1);
        (max_floats / denominator).max(1).min(pixels.max(1))
    }

    pub(super) fn particle_pixel_positions(x: &Tensor2, config: DirectBasisTrainConfig) -> Tensor2 {
        let size = config.loss_config.image_size as f32;
        let world_scale = (size - 1.0) / (config.loss_config.hi - config.loss_config.lo);
        let px = x
            .clone()
            .narrow(1, 0, 1)
            .add_scalar(-config.loss_config.lo)
            .mul_scalar(world_scale);
        let py = x
            .clone()
            .narrow(1, 1, 1)
            .add_scalar(-config.loss_config.lo)
            .mul_scalar(-world_scale)
            .add_scalar(size - 1.0);
        Tensor::cat(vec![px, py], 1)
    }

    pub(super) fn particle_pixel_positions_batch(x: &Tensor3, config: DirectBasisTrainConfig) -> Tensor3 {
        let size = config.loss_config.image_size as f32;
        let world_scale = (size - 1.0) / (config.loss_config.hi - config.loss_config.lo);
        let px = x
            .clone()
            .narrow(2, 0, 1)
            .add_scalar(-config.loss_config.lo)
            .mul_scalar(world_scale);
        let py = x
            .clone()
            .narrow(2, 1, 1)
            .add_scalar(-config.loss_config.lo)
            .mul_scalar(-world_scale)
            .add_scalar(size - 1.0);
        Tensor::cat(vec![px, py], 2)
    }

    pub(super) fn l1l2_tensor(value: Tensor2) -> Tensor2 {
        value.clone().abs() + value.clone().mul(value)
    }

    pub(super) fn l1l2_tensor3(value: Tensor3) -> Tensor3 {
        value.clone().abs() + value.clone().mul(value)
    }

    fn stable_log1p_over_norm<B: burn::tensor::backend::Backend, const D: usize>(
        norm: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let small = norm.clone().lower_elem(1.0e-3);
        let norm2 = norm.clone().mul(norm.clone());
        let norm3 = norm2.clone().mul(norm.clone());
        let norm4 = norm3.clone().mul(norm.clone());
        let norm5 = norm4.clone().mul(norm.clone());
        let series = norm
            .clone()
            .mul_scalar(-0.5)
            .add_scalar(1.0)
            + norm2.mul_scalar(1.0 / 3.0)
            + norm3.mul_scalar(-0.25)
            + norm4.mul_scalar(0.2)
            + norm5.mul_scalar(-1.0 / 6.0);
        norm.clone().log1p().div(norm).mask_where(small, series)
    }

    pub(super) fn log_normalize_vectors(values: Tensor2) -> Tensor2 {
        let dims = values.shape().dims::<2>();
        let norm = values
            .clone()
            .mul(values.clone())
            .sum_dim(1)
            .add_scalar(EPSILON * EPSILON)
            .sqrt()
            .clamp_min(EPSILON);
        values * stable_log1p_over_norm(norm).expand([dims[0], dims[1]])
    }

    pub(super) fn log_normalize_vectors_batch(values: Tensor3) -> Tensor3 {
        let dims = values.shape().dims::<3>();
        let norm = values
            .clone()
            .mul(values.clone())
            .sum_dim(2)
            .add_scalar(EPSILON * EPSILON)
            .sqrt()
            .clamp_min(EPSILON);
        values * stable_log1p_over_norm(norm).expand([dims[0], dims[1], dims[2]])
    }

    pub(super) fn log_normalize_vectors_batch_generic<B: burn::tensor::backend::Backend>(
        values: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let dims = values.shape().dims::<3>();
        let norm = values
            .clone()
            .mul(values.clone())
            .sum_dim(2)
            .add_scalar(EPSILON * EPSILON)
            .sqrt()
            .clamp_min(EPSILON);
        values * stable_log1p_over_norm(norm).expand([dims[0], dims[1], dims[2]])
    }

    pub(super) fn log_normalize_state_gradient(values: Tensor3) -> Tensor2 {
        let dims = values.shape().dims::<3>();
        let norm = values
            .clone()
            .mul(values.clone())
            .sum_dim(2)
            .add_scalar(EPSILON * EPSILON)
            .sqrt()
            .clamp_min(EPSILON);
        (values * stable_log1p_over_norm(norm).expand([dims[0], dims[1], dims[2]]))
        .reshape([dims[0], dims[1] * dims[2]])
    }

    pub(super) fn log_normalize_state_gradient_batch(values: Tensor4) -> Tensor3 {
        let dims = values.shape().dims::<4>();
        let norm = values
            .clone()
            .mul(values.clone())
            .sum_dim(3)
            .add_scalar(EPSILON * EPSILON)
            .sqrt()
            .clamp_min(EPSILON);
        (values * stable_log1p_over_norm(norm).expand([dims[0], dims[1], dims[2], dims[3]]))
        .reshape([dims[0], dims[1], dims[2] * dims[3]])
    }

    pub(super) fn log_normalize_state_gradient_batch_generic<B: burn::tensor::backend::Backend>(
        values: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let dims = values.shape().dims::<4>();
        let norm = values
            .clone()
            .mul(values.clone())
            .sum_dim(3)
            .add_scalar(EPSILON * EPSILON)
            .sqrt()
            .clamp_min(EPSILON);
        (values * stable_log1p_over_norm(norm).expand([dims[0], dims[1], dims[2], dims[3]]))
        .reshape([dims[0], dims[1], dims[2] * dims[3]])
    }

    pub(super) fn apply_moment_correction_2d(
        state_gradient: Tensor3,
        diff: Tensor3,
        volume_grad: Tensor3,
    ) -> Tensor3 {
        let dims = state_gradient.shape().dims::<3>();
        let query_rows = dims[0];
        let state_dims = dims[1];
        let neighbor_rows = diff.shape().dims::<3>()[1];
        let moment = diff
            .unsqueeze_dim::<4>(3)
            .expand([query_rows, neighbor_rows, 2, 2])
            .mul(
                volume_grad
                    .unsqueeze_dim::<4>(2)
                    .expand([query_rows, neighbor_rows, 2, 2]),
            )
            .sum_dim(1)
            .squeeze_dim::<3>(1);
        let a = moment
            .clone()
            .narrow(1, 0, 1)
            .narrow(2, 0, 1)
            .reshape([query_rows, 1]);
        let b = moment
            .clone()
            .narrow(1, 0, 1)
            .narrow(2, 1, 1)
            .reshape([query_rows, 1]);
        let d = moment
            .narrow(1, 1, 1)
            .narrow(2, 1, 1)
            .reshape([query_rows, 1]);
        let det = a.clone().mul(d.clone()) - b.clone().mul(b.clone());
        let near_singular = det.clone().abs().lower_elem(1.0e-3);
        let ones = Tensor::<BurnBackend, 2>::ones([query_rows, 1], &state_gradient.device());
        let zeros = Tensor::<BurnBackend, 2>::zeros([query_rows, 1], &state_gradient.device());
        let inv_det = det.mask_where(near_singular.clone(), ones.clone()).recip();
        let inv00 = d
            .mul(inv_det.clone())
            .mask_where(near_singular.clone(), ones);
        let inv01 = b
            .mul_scalar(-1.0)
            .mul(inv_det.clone())
            .mask_where(near_singular.clone(), zeros.clone());
        let inv11 = a.mul(inv_det).mask_where(
            near_singular,
            Tensor::<BurnBackend, 2>::ones([query_rows, 1], &state_gradient.device()),
        );
        let gx = state_gradient.clone().narrow(2, 0, 1);
        let gy = state_gradient.narrow(2, 1, 1);
        let inv00 = inv00
            .unsqueeze_dim::<3>(1)
            .expand([query_rows, state_dims, 1]);
        let inv01 = inv01
            .unsqueeze_dim::<3>(1)
            .expand([query_rows, state_dims, 1]);
        let inv11 = inv11
            .unsqueeze_dim::<3>(1)
            .expand([query_rows, state_dims, 1]);
        let corrected_x = gx.clone().mul(inv00) + gy.clone().mul(inv01.clone());
        let corrected_y = gx.mul(inv01) + gy.mul(inv11);
        Tensor::cat(vec![corrected_x, corrected_y], 2)
    }

    pub(super) fn apply_moment_correction_2d_batch(
        state_gradient: Tensor4,
        diff: Tensor4,
        volume_grad: Tensor4,
    ) -> Tensor4 {
        let dims = state_gradient.shape().dims::<4>();
        let batches = dims[0];
        let query_rows = dims[1];
        let state_dims = dims[2];
        let neighbor_rows = diff.shape().dims::<4>()[2];
        let moment = diff
            .unsqueeze_dim::<5>(4)
            .expand([batches, query_rows, neighbor_rows, 2, 2])
            .mul(volume_grad.unsqueeze_dim::<5>(3).expand([
                batches,
                query_rows,
                neighbor_rows,
                2,
                2,
            ]))
            .sum_dim(2)
            .squeeze_dim::<4>(2);
        let a = moment
            .clone()
            .narrow(2, 0, 1)
            .narrow(3, 0, 1)
            .reshape([batches, query_rows, 1]);
        let b = moment
            .clone()
            .narrow(2, 0, 1)
            .narrow(3, 1, 1)
            .reshape([batches, query_rows, 1]);
        let d = moment
            .narrow(2, 1, 1)
            .narrow(3, 1, 1)
            .reshape([batches, query_rows, 1]);
        let det = a.clone().mul(d.clone()) - b.clone().mul(b.clone());
        let near_singular = det.clone().abs().lower_elem(1.0e-3);
        let ones =
            Tensor::<BurnBackend, 3>::ones([batches, query_rows, 1], &state_gradient.device());
        let zeros =
            Tensor::<BurnBackend, 3>::zeros([batches, query_rows, 1], &state_gradient.device());
        let inv_det = det.mask_where(near_singular.clone(), ones.clone()).recip();
        let inv00 = d
            .mul(inv_det.clone())
            .mask_where(near_singular.clone(), ones);
        let inv01 = b
            .mul_scalar(-1.0)
            .mul(inv_det.clone())
            .mask_where(near_singular.clone(), zeros);
        let inv11 = a.mul(inv_det).mask_where(
            near_singular,
            Tensor::<BurnBackend, 3>::ones([batches, query_rows, 1], &state_gradient.device()),
        );
        let gx = state_gradient.clone().narrow(3, 0, 1);
        let gy = state_gradient.narrow(3, 1, 1);
        let inv00 = inv00
            .unsqueeze_dim::<4>(2)
            .expand([batches, query_rows, state_dims, 1]);
        let inv01 = inv01
            .unsqueeze_dim::<4>(2)
            .expand([batches, query_rows, state_dims, 1]);
        let inv11 = inv11
            .unsqueeze_dim::<4>(2)
            .expand([batches, query_rows, state_dims, 1]);
        let corrected_x = gx.clone().mul(inv00) + gy.clone().mul(inv01.clone());
        let corrected_y = gx.mul(inv01) + gy.mul(inv11);
        Tensor::cat(vec![corrected_x, corrected_y], 3)
    }

    pub(super) fn apply_moment_correction_2d_batch_generic<B: burn::tensor::backend::Backend>(
        state_gradient: Tensor<B, 4>,
        diff: Tensor<B, 4>,
        volume_grad: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let dims = state_gradient.shape().dims::<4>();
        let batches = dims[0];
        let query_rows = dims[1];
        let state_dims = dims[2];
        let neighbor_rows = diff.shape().dims::<4>()[2];
        let moment = diff
            .unsqueeze_dim::<5>(4)
            .expand([batches, query_rows, neighbor_rows, 2, 2])
            .mul(volume_grad.unsqueeze_dim::<5>(3).expand([
                batches,
                query_rows,
                neighbor_rows,
                2,
                2,
            ]))
            .sum_dim(2)
            .squeeze_dim::<4>(2);
        let a = moment
            .clone()
            .narrow(2, 0, 1)
            .narrow(3, 0, 1)
            .reshape([batches, query_rows, 1]);
        let b = moment
            .clone()
            .narrow(2, 0, 1)
            .narrow(3, 1, 1)
            .reshape([batches, query_rows, 1]);
        let d = moment
            .narrow(2, 1, 1)
            .narrow(3, 1, 1)
            .reshape([batches, query_rows, 1]);
        let det = a.clone().mul(d.clone()) - b.clone().mul(b.clone());
        let near_singular = det.clone().abs().lower_elem(1.0e-3);
        let ones = Tensor::<B, 3>::ones([batches, query_rows, 1], &state_gradient.device());
        let zeros = Tensor::<B, 3>::zeros([batches, query_rows, 1], &state_gradient.device());
        let inv_det = det.mask_where(near_singular.clone(), ones.clone()).recip();
        let inv00 = d
            .mul(inv_det.clone())
            .mask_where(near_singular.clone(), ones);
        let inv01 = b
            .mul_scalar(-1.0)
            .mul(inv_det.clone())
            .mask_where(near_singular.clone(), zeros);
        let inv11 = a.mul(inv_det).mask_where(
            near_singular,
            Tensor::<B, 3>::ones([batches, query_rows, 1], &state_gradient.device()),
        );
        let gx = state_gradient.clone().narrow(3, 0, 1);
        let gy = state_gradient.narrow(3, 1, 1);
        let inv00 = inv00
            .unsqueeze_dim::<4>(2)
            .expand([batches, query_rows, state_dims, 1]);
        let inv01 = inv01
            .unsqueeze_dim::<4>(2)
            .expand([batches, query_rows, state_dims, 1]);
        let inv11 = inv11
            .unsqueeze_dim::<4>(2)
            .expand([batches, query_rows, state_dims, 1]);
        let corrected_x = gx.clone().mul(inv00) + gy.clone().mul(inv01.clone());
        let corrected_y = gx.mul(inv01) + gy.mul(inv11);
        Tensor::cat(vec![corrected_x, corrected_y], 3)
    }
