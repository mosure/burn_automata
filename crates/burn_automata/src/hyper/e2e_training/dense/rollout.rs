//! NPA rollout chunks and geometry extraction.

use super::*;

pub(super) type DensityLitStats = (usize, Option<[usize; 4]>);

fn normalized_motion_2d(
    dx_raw: Tensor2,
    motion_scale: f32,
    rows: usize,
) -> (Tensor2, Tensor2, Tensor2) {
    let dx_squared = dx_raw.clone().mul(dx_raw.clone()).sum_dim(1);
    let denominator = dx_squared
        .clone()
        .add_scalar(EPSILON * EPSILON)
        .sqrt()
        .add_scalar(1.0);
    let dx = dx_raw
        .mul_scalar(motion_scale)
        .div(denominator.clone().expand([rows, 2]));
    (dx, dx_squared, denominator)
}

fn displacement_magnitude_2d(
    dx_squared: Tensor2,
    denominator: Tensor2,
    motion_scale: f32,
) -> Tensor2 {
    dx_squared
        .mul_scalar(motion_scale * motion_scale)
        .div(denominator.clone().mul(denominator))
        .add_scalar(EPSILON * EPSILON)
        .sqrt()
}

pub(super) fn normalized_motion_batch(
    dx_raw: Tensor3,
    motion_scale: f32,
    batches: usize,
    particles: usize,
) -> (Tensor3, Tensor3, Tensor3) {
    let dx_squared = dx_raw.clone().mul(dx_raw.clone()).sum_dim(2);
    let denominator = dx_squared
        .clone()
        .add_scalar(EPSILON * EPSILON)
        .sqrt()
        .add_scalar(1.0);
    let dx = dx_raw
        .mul_scalar(motion_scale)
        .div(denominator.clone().expand([batches, particles, 2]));
    (dx, dx_squared, denominator)
}

pub(super) fn displacement_magnitude_batch(
    dx_squared: Tensor3,
    denominator: Tensor3,
    motion_scale: f32,
) -> Tensor3 {
    dx_squared
        .mul_scalar(motion_scale * motion_scale)
        .div(denominator.clone().mul(denominator))
        .add_scalar(EPSILON * EPSILON)
        .sqrt()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn rollout_single_chunk(
        params: &BurnBaseParams,
        adapter: &BurnAdapterParams,
        target: &BurnTargetExample,
        mut x: Tensor2,
        mut s: Tensor2,
        config: DirectBasisTrainConfig,
        rng: &mut StdRng,
        steps: usize,
        mut displacement: Tensor1,
    ) -> (Tensor2, Tensor2, Tensor1) {
        let mask_stack = if target.update_prob >= 1.0 {
            None
        } else {
            Some(host_single_mask_stack(target, steps, rng))
        };
        for step in 0..steps {
            let features = rollout_dense_perception(&x, &s, config);
            let update = params.forward_adapter(features, adapter, config);
            let dx_raw = update.clone().narrow(1, 0, 2);
            let ds = update.narrow(1, 2, s.shape().dims::<2>()[1]);
            let (dx, dx_squared, denominator) =
                normalized_motion_2d(dx_raw, config.motion_scale, target.particle_count);
            if config.loss_config.displacement_regularizer_weight > 0.0 {
                displacement = displacement
                    + displacement_magnitude_2d(
                        dx_squared,
                        denominator,
                        config.motion_scale,
                    )
                    .mean();
            }
            let state_dims = s.shape().dims::<2>()[1];
            if target.update_prob >= 1.0 {
                x = x + dx;
                s = s + ds;
            } else {
                let mask = mask_stack
                    .as_ref()
                    .expect("non-unit update_prob should have a mask stack")
                    .clone()
                    .narrow(0, step, 1)
                    .squeeze_dim::<2>(0);
                x = x + dx.mul(mask.clone().expand([target.particle_count, 2]));
                s = s + ds.mul(mask.expand([target.particle_count, state_dims]));
            }
        }
        (x, s, displacement)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn rollout_batch_chunk(
        params: &BurnBaseParams,
        adapter_batch: &BurnAdapterBatch,
        targets: &[BurnTargetExample],
        indices: &[usize],
        mut x: Tensor3,
        mut s: Tensor3,
        config: DirectBasisTrainConfig,
        particle_count: usize,
        _rng: &mut StdRng,
        steps: usize,
        mut displacement: Tensor1,
        condition_control: Option<&BurnE2eConditionControlBatch>,
    ) -> (Tensor3, Tensor3, Tensor1) {
        let unit_update = batch_update_prob_is_one(targets, indices);
        let mask_stack = if unit_update {
            None
        } else {
            Some(device_batch_mask_stack(
                targets,
                indices,
                particle_count,
                steps,
            ))
        };
        for step in 0..steps {
            let features = rollout_dense_perception_batch(&x, &s, config);
            let mut update = params.forward_adapter_batch(features, adapter_batch);
            if let Some(condition_control) = condition_control {
                update = update + condition_control.update_for_particles(&x, &s);
            }
            let state_dims = s.shape().dims::<3>()[2];
            let dx_raw = update.clone().narrow(2, 0, 2);
            let ds = update.narrow(2, 2, state_dims);
            let (dx, dx_squared, denominator) = normalized_motion_batch(
                dx_raw,
                config.motion_scale,
                indices.len(),
                particle_count,
            );
            if config.loss_config.displacement_regularizer_weight > 0.0 {
                displacement = displacement
                    + displacement_magnitude_batch(
                        dx_squared,
                        denominator,
                        config.motion_scale,
                    )
                    .mean();
            }
            if unit_update {
                x = x + dx;
                s = s + ds;
            } else {
                let mask = mask_stack
                    .as_ref()
                    .expect("non-unit update_prob should have a mask stack")
                    .clone()
                    .narrow(0, step, 1)
                    .squeeze_dim::<3>(0);
                x = x + dx.mul(mask.clone().expand([indices.len(), particle_count, 2]));
                s = s + ds.mul(mask.expand([indices.len(), particle_count, state_dims]));
            }
        }
        (x, s, displacement)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn rollout_batch_eval_chunk(
        params: &BurnBaseParams,
        adapter_batch: &BurnAdapterBatch,
        targets: &[BurnTargetExample],
        indices: &[usize],
        mut x: Tensor3,
        mut s: Tensor3,
        config: DirectBasisTrainConfig,
        particle_count: usize,
        rngs: &mut [StdRng],
        steps: usize,
        mut displacement: Tensor1,
        condition_control: Option<&BurnE2eConditionControlBatch>,
    ) -> (Tensor3, Tensor3, Tensor1) {
        let unit_update = batch_update_prob_is_one(targets, indices);
        let mask_stack = if unit_update {
            None
        } else {
            Some(host_batch_mask_stack_with_rngs(
                targets,
                indices,
                particle_count,
                steps,
                rngs,
            ))
        };
        for step in 0..steps {
            let features = rollout_dense_perception_batch(&x, &s, config);
            let mut update = params.forward_adapter_batch(features, adapter_batch);
            if let Some(condition_control) = condition_control {
                update = update + condition_control.update_for_particles(&x, &s);
            }
            let state_dims = s.shape().dims::<3>()[2];
            let dx_raw = update.clone().narrow(2, 0, 2);
            let ds = update.narrow(2, 2, state_dims);
            let (dx, dx_squared, denominator) = normalized_motion_batch(
                dx_raw,
                config.motion_scale,
                indices.len(),
                particle_count,
            );
            if config.loss_config.displacement_regularizer_weight > 0.0 {
                let dx_norm = displacement_magnitude_batch(
                    dx_squared,
                    denominator,
                    config.motion_scale,
                )
                .reshape([indices.len(), particle_count])
                .mean_dim(1)
                .squeeze_dim::<1>(1);
                displacement = displacement + dx_norm;
            }
            if unit_update {
                x = x + dx;
                s = s + ds;
            } else {
                let mask = mask_stack
                    .as_ref()
                    .expect("non-unit update_prob should have a mask stack")
                    .clone()
                    .narrow(0, step, 1)
                    .squeeze_dim::<3>(0);
                x = x + dx.mul(mask.clone().expand([indices.len(), particle_count, 2]));
                s = s + ds.mul(mask.expand([indices.len(), particle_count, state_dims]));
            }
        }
        (x, s, displacement)
}

    #[allow(clippy::too_many_arguments)]
    pub(super) fn rollout_oracle_model_batch_chunk(
        params: &BurnBaseBatch,
        targets: &[BurnTargetExample],
        indices: &[usize],
        mut x: Tensor3,
        mut s: Tensor3,
        config: DirectBasisTrainConfig,
        particle_count: usize,
        rngs: &mut [StdRng],
        steps: usize,
        mut displacement: Tensor1,
    ) -> (Tensor3, Tensor3, Tensor1) {
        let unit_update = batch_update_prob_is_one(targets, indices);
        let mask_stack = if unit_update {
            None
        } else {
            Some(host_batch_mask_stack_with_rngs(
                targets,
                indices,
                particle_count,
                steps,
                rngs,
            ))
        };
        for step in 0..steps {
            let features = rollout_dense_perception_batch(&x, &s, config);
            let update = params.forward(features);
            let state_dims = s.shape().dims::<3>()[2];
            let dx_raw = update.clone().narrow(2, 0, 2);
            let ds = update.narrow(2, 2, state_dims);
            let (dx, dx_squared, denominator) = normalized_motion_batch(
                dx_raw,
                config.motion_scale,
                indices.len(),
                particle_count,
            );
            if config.loss_config.displacement_regularizer_weight > 0.0 {
                let dx_norm = displacement_magnitude_batch(
                    dx_squared,
                    denominator,
                    config.motion_scale,
                )
                    .reshape([indices.len(), particle_count])
                    .mean_dim(1)
                    .squeeze_dim::<1>(1);
                displacement = displacement + dx_norm;
            }
            if unit_update {
                x = x + dx;
                s = s + ds;
            } else {
                let mask = mask_stack
                    .as_ref()
                    .expect("non-unit update_prob should have a mask stack")
                    .clone()
                    .narrow(0, step, 1)
                    .squeeze_dim::<3>(0);
                x = x + dx.mul(mask.clone().expand([indices.len(), particle_count, 2]));
                s = s + ds.mul(mask.expand([indices.len(), particle_count, state_dims]));
            }
        }
        (x, s, displacement)
    }

    pub(super) fn example_eval_loss_bounded(
        params: &BurnBaseParams,
        adapter: &BurnAdapterParams,
        target: &BurnTargetExample,
        config: DirectBasisTrainConfig,
        seed: u64,
    ) -> BurnLossTensors {
        let device = &target.target_rgb.device();
        let (mut x, mut s) = seed_tensors(
            target.particle_count,
            config,
            target.seed_scale,
            seed,
            device,
        );
        let mut rng = StdRng::seed_from_u64(seed ^ 0x005e_ed2d);
        let mut displacement = Tensor::<BurnBackend, 1>::zeros([1], device);
        let chunk_steps = tbptt_chunk_steps(config);
        let mut remaining_steps = config.rollout_steps;
        while remaining_steps > 0 {
            let steps = remaining_steps.min(chunk_steps);
            (x, s, displacement) = rollout_single_chunk(
                params,
                adapter,
                target,
                x,
                s,
                config,
                &mut rng,
                steps,
                displacement,
            );
            remaining_steps -= steps;
            if remaining_steps > 0 {
                x = detach2(x);
                s = detach2(s);
                displacement = detach1(displacement);
            }
        }
        target_splat_loss(&x, &s, target, config, adapter, displacement)
    }

    pub(super) fn batch_example_eval_loss(
        params: &BurnBaseParams,
        adapters: &[BurnAdapterParams],
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        seed: u64,
    ) -> Result<BurnLossBatchTensors, Box<dyn std::error::Error>> {
        let Some(particle_count) = homogeneous_particle_count(targets, indices) else {
            return Err(std::io::Error::other(
                "Burn eval batch path requires homogeneous particle counts",
            )
            .into());
        };
        let device = &targets[indices[0]].target_rgb.device();
        let adapter_batch = BurnAdapterBatch::from_indices(adapters, indices);
        let (mut x, mut s) =
            seed_batch_tensors(targets, indices, particle_count, config, seed, device);
        let mut rngs = indices
            .iter()
            .map(|idx| StdRng::seed_from_u64(seed.wrapping_add(*idx as u64) ^ 0x005e_ed2d))
            .collect::<Vec<_>>();
        let mut displacement = Tensor::<BurnBackend, 1>::zeros([indices.len()], device);
        let chunk_steps = tbptt_chunk_steps(config);
        let mut remaining_steps = config.rollout_steps;
        while remaining_steps > 0 {
            let steps = remaining_steps.min(chunk_steps);
            (x, s, displacement) = rollout_batch_eval_chunk(
                params,
                &adapter_batch,
                targets,
                indices,
                x,
                s,
                config,
                particle_count,
                &mut rngs,
                steps,
                displacement,
                None,
            );
            remaining_steps -= steps;
            if remaining_steps > 0 {
                x = detach3(x);
                s = detach3(s);
                displacement = detach1(displacement);
            }
        }
        Ok(target_splat_loss_batch_vector_selected(
            &x,
            &s,
            targets,
            indices,
            config,
            &adapter_batch,
            displacement,
        )?)
    }

    pub(super) fn batch_example_geometry(
        params: &BurnBaseParams,
        adapters: &[BurnAdapterParams],
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        seed: u64,
    ) -> Result<Option<BurnGeometrySummary>, Box<dyn std::error::Error>> {
        let Some(particle_count) = homogeneous_particle_count(targets, indices) else {
            return Ok(None);
        };
        let device = &targets[indices[0]].target_rgb.device();
        let adapter_batch = BurnAdapterBatch::from_indices(adapters, indices);
        let (mut x, mut s) =
            seed_batch_tensors(targets, indices, particle_count, config, seed, device);
        let mut rngs = indices
            .iter()
            .map(|idx| StdRng::seed_from_u64(seed.wrapping_add(*idx as u64) ^ 0x005e_ed2d))
            .collect::<Vec<_>>();
        let mut displacement = Tensor::<BurnBackend, 1>::zeros([indices.len()], device);
        let chunk_steps = tbptt_chunk_steps(config);
        let mut remaining_steps = config.rollout_steps;
        while remaining_steps > 0 {
            let steps = remaining_steps.min(chunk_steps);
            (x, s, displacement) = rollout_batch_eval_chunk(
                params,
                &adapter_batch,
                targets,
                indices,
                x,
                s,
                config,
                particle_count,
                &mut rngs,
                steps,
                displacement,
                None,
            );
            remaining_steps -= steps;
            if remaining_steps > 0 {
                x = detach3(x);
                s = detach3(s);
                displacement = detach1(displacement);
            }
        }

        let centered = if config.loss_config.center {
            let target_mean = stack_target_mean(targets, indices);
            x.clone() - x.clone().mean_dim(1).expand([indices.len(), particle_count, 2])
                + target_mean.expand([indices.len(), particle_count, 2])
        } else {
            x.clone()
        };
        let state_dims = s.shape().dims::<3>()[2];
        let colors = s.narrow(2, state_dims - 3, 3).add_scalar(0.5);
        let (_, density) =
            splat_render_batch(&centered, &colors, targets, indices, config, particle_count);
        let target_density = stack_target_density(targets, indices);
        geometry_summary_from_density(
            tensor3_vec(density.inner())?,
            tensor3_vec(target_density.inner())?,
            indices.len(),
            config.loss_config.image_size,
        )
    }

    pub(super) fn homogeneous_particle_count(
        targets: &[BurnTargetExample],
        indices: &[usize],
    ) -> Option<usize> {
        let mut iter = indices.iter().map(|idx| targets[*idx].particle_count);
        let first = iter.next()?;
        iter.all(|count| count == first).then_some(first)
    }

    pub(super) fn geometry_summary_from_density(
        density: Vec<f32>,
        target_density: Vec<f32>,
        batches: usize,
        image_size: usize,
    ) -> Result<Option<BurnGeometrySummary>, Box<dyn std::error::Error>> {
        let pixels = image_size * image_size;
        if batches == 0 {
            return Ok(None);
        }
        if density.len() != batches * pixels || target_density.len() != batches * pixels {
            return Err(std::io::Error::other(format!(
                "Burn geometry density shape mismatch: density={} target={} expected={}",
                density.len(),
                target_density.len(),
                batches * pixels
            ))
            .into());
        }
        let mut summary = BurnGeometrySummary {
            examples: batches,
            mean_score: 0.0,
            mean_foreground_iou: 0.0,
            mean_target_recall: 0.0,
            mean_generated_precision: 0.0,
            mean_bbox_iou: 0.0,
            mean_lit_pixel_ratio: 0.0,
            mean_bbox_width_ratio: 0.0,
            mean_bbox_area_ratio: 0.0,
        };
        for batch in 0..batches {
            let start = batch * pixels;
            let end = start + pixels;
            let generated = &density[start..end];
            let target = &target_density[start..end];
            let threshold = target
                .iter()
                .copied()
                .fold(0.0_f32, |max_value, value| max_value.max(value))
                .mul_add(0.05, 0.0)
                .max(1.0e-6);
            let (lit_pixels, bbox) = density_lit_stats(generated, image_size, threshold)?;
            let (target_lit_pixels, target_bbox) =
                density_lit_stats(target, image_size, threshold)?;
            let lit_ratio = lit_pixels as f32 / target_lit_pixels.max(1) as f32;
            let iou = bbox_iou(bbox, target_bbox).unwrap_or(0.0);
            let width_ratio = bbox_width_ratio(bbox, target_bbox).unwrap_or(0.0);
            let area_ratio = bbox_area_ratio(bbox, target_bbox).unwrap_or(0.0);
            let overlap = density_overlap_stats(generated, target, threshold)?;
            let score = 1.5 * overlap.iou
                + 0.5 * overlap.target_recall
                + 0.25 * overlap.generated_precision
                + 0.25 * iou
                - 0.25 * (lit_ratio - 1.0).abs()
                - 0.35 * (width_ratio - 1.0).abs()
                - 0.15 * (area_ratio - 1.0).abs();
            summary.mean_score += score;
            summary.mean_foreground_iou += overlap.iou;
            summary.mean_target_recall += overlap.target_recall;
            summary.mean_generated_precision += overlap.generated_precision;
            summary.mean_bbox_iou += iou;
            summary.mean_lit_pixel_ratio += lit_ratio;
            summary.mean_bbox_width_ratio += width_ratio;
            summary.mean_bbox_area_ratio += area_ratio;
        }
        let scale = 1.0 / batches as f32;
        summary.mean_score *= scale;
        summary.mean_foreground_iou *= scale;
        summary.mean_target_recall *= scale;
        summary.mean_generated_precision *= scale;
        summary.mean_bbox_iou *= scale;
        summary.mean_lit_pixel_ratio *= scale;
        summary.mean_bbox_width_ratio *= scale;
        summary.mean_bbox_area_ratio *= scale;
        Ok(Some(summary))
    }

    #[derive(Clone, Copy)]
    pub(super) struct BurnDensityOverlapStats {
        iou: f32,
        target_recall: f32,
        generated_precision: f32,
    }

    pub(super) fn density_overlap_stats(
        generated: &[f32],
        target: &[f32],
        threshold: f32,
    ) -> Result<BurnDensityOverlapStats, Box<dyn std::error::Error>> {
        if generated.len() != target.len() {
            return Err(std::io::Error::other("Burn geometry density overlap shape mismatch").into());
        }
        let mut generated_count = 0usize;
        let mut target_count = 0usize;
        let mut intersection = 0usize;
        let mut union = 0usize;
        for (&generated_density, &target_density) in generated.iter().zip(target) {
            let generated_hit = generated_density >= threshold;
            let target_hit = target_density >= threshold;
            generated_count += usize::from(generated_hit);
            target_count += usize::from(target_hit);
            intersection += usize::from(generated_hit && target_hit);
            union += usize::from(generated_hit || target_hit);
        }
        Ok(BurnDensityOverlapStats {
            iou: intersection as f32 / union.max(1) as f32,
            target_recall: intersection as f32 / target_count.max(1) as f32,
            generated_precision: intersection as f32 / generated_count.max(1) as f32,
        })
    }

    pub(super) fn density_lit_stats(
        density: &[f32],
        image_size: usize,
        threshold: f32,
    ) -> Result<DensityLitStats, Box<dyn std::error::Error>> {
        if density.len() != image_size * image_size {
            return Err(std::io::Error::other("Burn geometry density shape mismatch").into());
        }
        let mut lit_pixels = 0usize;
        let mut min_x = image_size;
        let mut min_y = image_size;
        let mut max_x = 0usize;
        let mut max_y = 0usize;
        for y in 0..image_size {
            for x in 0..image_size {
                if density[y * image_size + x] < threshold {
                    continue;
                }
                lit_pixels += 1;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
        Ok((
            lit_pixels,
            (lit_pixels > 0).then_some([min_x, min_y, max_x, max_y]),
        ))
    }

    pub(super) fn bbox_iou(left: Option<[usize; 4]>, right: Option<[usize; 4]>) -> Option<f32> {
        let left = left?;
        let right = right?;
        let x0 = left[0].max(right[0]);
        let y0 = left[1].max(right[1]);
        let x1 = left[2].min(right[2]);
        let y1 = left[3].min(right[3]);
        let intersection = if x1 >= x0 && y1 >= y0 {
            bbox_area([x0, y0, x1, y1])
        } else {
            0.0
        };
        let union = bbox_area(left) + bbox_area(right) - intersection;
        Some(intersection / union.max(f32::MIN_POSITIVE))
    }

    pub(super) fn bbox_width_ratio(left: Option<[usize; 4]>, right: Option<[usize; 4]>) -> Option<f32> {
        Some(bbox_width(left?) / bbox_width(right?).max(f32::MIN_POSITIVE))
    }

    pub(super) fn bbox_area_ratio(left: Option<[usize; 4]>, right: Option<[usize; 4]>) -> Option<f32> {
        Some(bbox_area(left?) / bbox_area(right?).max(f32::MIN_POSITIVE))
    }

    pub(super) fn bbox_width(bbox: [usize; 4]) -> f32 {
        bbox[2].saturating_sub(bbox[0]).saturating_add(1) as f32
    }

    pub(super) fn bbox_height(bbox: [usize; 4]) -> f32 {
        bbox[3].saturating_sub(bbox[1]).saturating_add(1) as f32
    }

    pub(super) fn bbox_area(bbox: [usize; 4]) -> f32 {
        bbox_width(bbox) * bbox_height(bbox)
    }
