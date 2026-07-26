//! Gradient, tensor snapshot, synchronization, and memory-budget utilities.

use super::*;

    pub(super) fn sample_update_stats(counts: &[usize]) -> SampleUpdateStats {
        if counts.is_empty() {
            return SampleUpdateStats {
                examples: 0,
                total_updates: 0,
                min_updates: 0,
                max_updates: 0,
                mean_updates: 0.0,
                zero_update_examples: 0,
            };
        }
        let total_updates = counts.iter().sum::<usize>();
        SampleUpdateStats {
            examples: counts.len(),
            total_updates,
            min_updates: counts.iter().copied().min().unwrap_or(0),
            max_updates: counts.iter().copied().max().unwrap_or(0),
            mean_updates: total_updates as f32 / counts.len() as f32,
            zero_update_examples: counts.iter().filter(|updates| **updates == 0).count(),
        }
    }

    pub(super) fn loss_scalars(loss: &BurnLossTensors) -> AutomataResult<BurnLossScalars> {
        Ok(BurnLossScalars {
            total: finite_scalar(
                "Burn direct total loss",
                loss.total.clone().inner().into_scalar(),
            )?,
            splat: finite_scalar(
                "Burn direct splat loss",
                loss.splat.clone().inner().into_scalar(),
            )?,
            color: finite_scalar(
                "Burn direct color loss",
                loss.color.clone().inner().into_scalar(),
            )?,
            density: finite_scalar(
                "Burn direct density loss",
                loss.density.clone().inner().into_scalar(),
            )?,
        })
    }

    pub(super) fn loss_vector_scalars(loss: BurnLossBatchTensors) -> AutomataResult<Vec<BurnLossScalars>> {
        let total = tensor1_vec(loss.total.inner())?;
        let splat = tensor1_vec(loss.splat.inner())?;
        let color = tensor1_vec(loss.color.inner())?;
        let density = tensor1_vec(loss.density.inner())?;
        loss_vector_scalars_from_parts(total, splat, color, density)
    }

    pub(super) async fn loss_vector_scalars_async(
        loss: BurnLossBatchTensors,
    ) -> AutomataResult<Vec<BurnLossScalars>> {
        let (total, splat, color, density) = (
            tensor1_vec_async(loss.total.inner()).await?,
            tensor1_vec_async(loss.splat.inner()).await?,
            tensor1_vec_async(loss.color.inner()).await?,
            tensor1_vec_async(loss.density.inner()).await?,
        );
        loss_vector_scalars_from_parts(total, splat, color, density)
    }

    fn loss_vector_scalars_from_parts(
        total: Vec<f32>,
        splat: Vec<f32>,
        color: Vec<f32>,
        density: Vec<f32>,
    ) -> AutomataResult<Vec<BurnLossScalars>> {
        if total.len() != splat.len() || total.len() != color.len() || total.len() != density.len()
        {
            return Err(AutomataError::InvalidArgument(
                "Burn direct vector loss readback length mismatch".to_string(),
            ));
        }
        total
            .into_iter()
            .zip(splat)
            .zip(color)
            .zip(density)
            .enumerate()
            .map(|(idx, (((total, splat), color), density))| {
                if !total.is_finite()
                    || !splat.is_finite()
                    || !color.is_finite()
                    || !density.is_finite()
                {
                    return Err(AutomataError::InvalidArgument(format!(
                        "Burn direct loss row {idx} is not finite: total={total:?} splat={splat:?} color={color:?} density={density:?}"
                    )));
                }
                Ok(BurnLossScalars {
                    total,
                    splat,
                    color,
                    density,
                })
            })
            .collect()
    }

    pub(super) fn prepare_grad_group(
        tensors: &mut [Tensor2Inner],
        clip_norm: f32,
        normalize: bool,
        collect_metrics: bool,
    ) -> AutomataResult<(f32, f32, Tensor1Inner)> {
        sanitize_nonfinite_gradients(tensors);
        let original_norm_tensor = group_norm_tensor(tensors);
        let original_norm = if collect_metrics {
            finite_scalar(
                "Burn direct grad norm",
                original_norm_tensor.clone().into_scalar(),
            )?
        } else {
            0.0
        };
        if normalize {
            for tensor in tensors.iter_mut() {
                let dims = tensor.shape().dims::<2>();
                let norm = tensor_l2_norm_tensor(tensor).add_scalar(1.0e-8);
                *tensor = tensor.clone().div(norm.expand(dims));
            }
        }
        let clip_norm_source = if normalize {
            group_norm_tensor(tensors)
        } else {
            original_norm_tensor
        };
        let scale_tensor = if clip_norm > 0.0 {
            clip_norm_source
                .clone()
                .clamp_min(clip_norm)
                .recip()
                .mul_scalar(clip_norm)
        } else {
            clip_norm_source.zeros_like().add_scalar(1.0)
        };
        let scale = if collect_metrics {
            finite_scalar("Burn direct grad scale", scale_tensor.clone().into_scalar())?
        } else {
            1.0
        };
        Ok((original_norm, scale, scale_tensor))
    }

    pub(super) fn sanitize_nonfinite_gradients(tensors: &mut [Tensor2Inner]) {
        for tensor in tensors {
            let nonfinite = tensor.clone().is_finite().bool_not();
            *tensor = tensor.clone().mask_fill(nonfinite, 0.0);
        }
    }

    pub(super) fn sanitize_nonfinite_model_batch_gradients(tensors: &mut [Tensor3Inner]) {
        for tensor in tensors {
            let nonfinite = tensor.clone().is_finite().bool_not();
            *tensor = tensor.clone().mask_fill(nonfinite, 0.0);
        }
    }

    pub(super) fn accumulate_gradient_group(
        accumulated: &mut Option<Vec<Tensor2Inner>>,
        gradients: Vec<Tensor2Inner>,
    ) {
        if let Some(accumulated) = accumulated {
            assert_eq!(
                accumulated.len(),
                gradients.len(),
                "gradient group shape changed between TBPTT chunks"
            );
            for (total, gradient) in accumulated.iter_mut().zip(gradients) {
                *total = total.clone() + gradient;
            }
        } else {
            *accumulated = Some(gradients);
        }
    }

    pub(super) fn scale_gradient_group(gradients: &mut [Tensor2Inner], scale: f32) {
        if scale != 1.0 {
            for gradient in gradients {
                *gradient = gradient.clone().mul_scalar(scale);
            }
        }
    }

    pub(super) fn prepare_model_batch_grad_group(
        tensors: &mut [Tensor3Inner],
        clip_norm: f32,
        normalize: bool,
        collect_metrics: bool,
    ) -> AutomataResult<(Vec<f32>, Vec<f32>, Tensor1Inner)> {
        sanitize_nonfinite_model_batch_gradients(tensors);
        let model_count = tensors
            .first()
            .map(|tensor| tensor.shape().dims::<3>()[0])
            .unwrap_or(0);
        let original_norm_tensor = model_batch_group_norm_tensor(tensors);
        let original_norms = if collect_metrics {
            tensor1_vec(original_norm_tensor.clone())?
                .into_iter()
                .enumerate()
                .map(|(model, value)| {
                    finite_scalar(&format!("Burn oracle model batch grad norm[{model}]"), value)
                })
                .collect::<AutomataResult<Vec<_>>>()?
        } else {
            vec![0.0; model_count]
        };
        if normalize {
            for tensor in tensors.iter_mut() {
                *tensor = normalize_model_batch_tensor(tensor);
            }
        }
        let clip_norm_source = if normalize {
            model_batch_group_norm_tensor(tensors)
        } else {
            original_norm_tensor
        };
        let scale_tensor = if clip_norm > 0.0 {
            clip_norm_source
                .clone()
                .clamp_min(clip_norm)
                .recip()
                .mul_scalar(clip_norm)
        } else {
            clip_norm_source.zeros_like().add_scalar(1.0)
        };
        let scales = if collect_metrics {
            tensor1_vec(scale_tensor.clone())?
                .into_iter()
                .enumerate()
                .map(|(model, value)| {
                    finite_scalar(&format!("Burn oracle model batch grad scale[{model}]"), value)
                })
                .collect::<AutomataResult<Vec<_>>>()?
        } else {
            vec![1.0; model_count]
        };
        Ok((original_norms, scales, scale_tensor))
    }

    pub(super) async fn prepare_model_batch_grad_group_async(
        tensors: &mut [Tensor3Inner],
        clip_norm: f32,
        normalize: bool,
        collect_metrics: bool,
    ) -> AutomataResult<(Vec<f32>, Vec<f32>, Tensor1Inner)> {
        sanitize_nonfinite_model_batch_gradients(tensors);
        let model_count = tensors
            .first()
            .map(|tensor| tensor.shape().dims::<3>()[0])
            .unwrap_or(0);
        let original_norm_tensor = model_batch_group_norm_tensor(tensors);
        let original_norms = if collect_metrics {
            tensor1_vec_async(original_norm_tensor.clone())
                .await?
                .into_iter()
                .enumerate()
                .map(|(model, value)| {
                    finite_scalar(&format!("Burn oracle model batch grad norm[{model}]"), value)
                })
                .collect::<AutomataResult<Vec<_>>>()?
        } else {
            vec![0.0; model_count]
        };
        if normalize {
            for tensor in tensors.iter_mut() {
                *tensor = normalize_model_batch_tensor(tensor);
            }
        }
        let clip_norm_source = if normalize {
            model_batch_group_norm_tensor(tensors)
        } else {
            original_norm_tensor
        };
        let scale_tensor = if clip_norm > 0.0 {
            clip_norm_source
                .clone()
                .clamp_min(clip_norm)
                .recip()
                .mul_scalar(clip_norm)
        } else {
            clip_norm_source.zeros_like().add_scalar(1.0)
        };
        let scales = if collect_metrics {
            tensor1_vec_async(scale_tensor.clone())
                .await?
                .into_iter()
                .enumerate()
                .map(|(model, value)| {
                    finite_scalar(&format!("Burn oracle model batch grad scale[{model}]"), value)
                })
                .collect::<AutomataResult<Vec<_>>>()?
        } else {
            vec![1.0; model_count]
        };
        Ok((original_norms, scales, scale_tensor))
    }

    pub(super) fn model_batch_group_norm_tensor(tensors: &[Tensor3Inner]) -> Tensor1Inner {
        let mut maximum = None::<Tensor1Inner>;
        for tensor in tensors {
            let value = model_batch_tensor_max_abs_tensor(tensor);
            maximum = Some(match maximum {
                Some(maximum) => maximum.max_pair(value),
                None => value,
            });
        }
        let maximum = maximum
            .expect("model batch gradient group has tensors")
            .clamp_min(1.0e-30);
        let mut total = None::<Tensor1Inner>;
        for tensor in tensors {
            let value = model_batch_tensor_scaled_squared_norm_tensor(tensor, maximum.clone());
            total = Some(match total {
                Some(total) => total + value,
                None => value,
            });
        }
        total
            .expect("model batch gradient group has tensors")
            .sqrt()
            .mul(maximum)
            .clamp_max(f32::MAX)
    }

    /// Select small top-k index sets without Burn's host sorting fallback.
    ///
    /// Adaptive topology uses very small `k` values. Repeated device argmax is
    /// cheaper than a full sort at that shape and remains valid in browser
    /// workers, where synchronous tensor readback is unavailable.
    pub(super) fn device_topk_indices(scores: Tensor2, count: usize) -> Tensor2Int {
        let [batch_size, width] = scores.shape().dims::<2>();
        let count = count.min(width);
        assert!(count > 0, "device top-k requires at least one selected row");
        let columns = Tensor::<BurnBackend, 1, Int>::arange(0..width as i64, &scores.device())
            .reshape([1, width])
            .expand([batch_size, width]);
        let mut remaining = scores;
        let mut selected = Vec::with_capacity(count);
        for _ in 0..count {
            let index = remaining.clone().argmax(1);
            let selected_mask = columns
                .clone()
                .equal(index.clone().expand([batch_size, width]));
            remaining = remaining.mask_fill(selected_mask, f32::NEG_INFINITY);
            selected.push(index);
        }
        Tensor::cat(selected, 1)
    }

    pub(super) fn model_batch_tensor_l2_norm_tensor(tensor: &Tensor3Inner) -> Tensor1Inner {
        let maximum = model_batch_tensor_max_abs_tensor(tensor).clamp_min(1.0e-30);
        model_batch_tensor_scaled_squared_norm_tensor(tensor, maximum.clone())
            .sqrt()
            .mul(maximum)
            .clamp_max(f32::MAX)
    }

    fn normalize_model_batch_tensor(tensor: &Tensor3Inner) -> Tensor3Inner {
        let dims = tensor.shape().dims::<3>();
        let maximum = model_batch_tensor_max_abs_tensor(tensor)
            .clamp_min(1.0e-30)
            .reshape([dims[0], 1, 1]);
        let scaled = tensor.clone().div(maximum.expand(dims));
        let norm = model_batch_tensor_squared_norm_tensor(&scaled)
            .sqrt()
            .add_scalar(1.0e-8)
            .reshape([dims[0], 1, 1]);
        scaled.div(norm.expand(dims))
    }

    fn model_batch_tensor_scaled_squared_norm_tensor(
        tensor: &Tensor3Inner,
        scale: Tensor1Inner,
    ) -> Tensor1Inner {
        let dims = tensor.shape().dims::<3>();
        let scaled = tensor
            .clone()
            .div(scale.reshape([dims[0], 1, 1]).expand(dims));
        model_batch_tensor_squared_norm_tensor(&scaled)
    }

    fn model_batch_tensor_squared_norm_tensor(tensor: &Tensor3Inner) -> Tensor1Inner {
        let dims = tensor.shape().dims::<3>();
        tensor
            .clone()
            .mul(tensor.clone())
            .reshape([dims[0], dims[1] * dims[2]])
            .sum_dim(1)
            .squeeze_dim::<1>(1)
    }

    fn model_batch_tensor_max_abs_tensor(tensor: &Tensor3Inner) -> Tensor1Inner {
        let dims = tensor.shape().dims::<3>();
        tensor
            .clone()
            .abs()
            .reshape([dims[0], dims[1] * dims[2]])
            .max_dim(1)
            .squeeze_dim::<1>(1)
    }

    pub(super) fn group_norm_tensor(tensors: &[Tensor2Inner]) -> Tensor1Inner {
        let mut total = None::<Tensor1Inner>;
        for tensor in tensors {
            let value = tensor.clone().mul(tensor.clone()).sum();
            total = Some(match total {
                Some(total) => total + value,
                None => value,
            });
        }
        total.expect("gradient group has tensors").sqrt()
    }

    pub(super) fn normalize_sample_id_table_gradient(
        gradient: Tensor2Inner,
        segments: &[(usize, usize)],
    ) -> Tensor2Inner {
        let dims = gradient.shape().dims::<2>();
        debug_assert_eq!(
            segments.iter().map(|(_, len)| *len).sum::<usize>(),
            dims[0]
        );
        Tensor::cat(
            segments
                .iter()
                .map(|&(offset, len)| {
                    let segment = gradient.clone().narrow(0, offset, len);
                    let per_identity_norm = segment
                        .clone()
                        .mul(segment.clone())
                        .sum_dim(0)
                        .sqrt()
                        .add_scalar(1.0e-8);
                    segment.div(per_identity_norm.expand([len, dims[1]]))
                })
                .collect(),
            0,
        )
    }

    impl PackedNpaGradientLayout {
        pub(super) fn new(config: &NpaConfig, output_bias: bool) -> Self {
            Self {
                hidden_dims: config.hidden_dims,
                perception_dims: config.perception_dims(),
                update_dims: config.update_dims(),
                max_row_dims: NpaParameterRowLayout2d::new(config).max_row_dims(),
                output_bias,
            }
        }

        fn packed_values(self) -> usize {
            (self.hidden_dims + self.update_dims) * self.max_row_dims
        }
    }

    pub(super) fn normalize_packed_npa_table_gradient(
        gradient: Tensor2Inner,
        layout: PackedNpaGradientLayout,
    ) -> Tensor2Inner {
        let [packed_values, identities] = gradient.shape().dims::<2>();
        debug_assert_eq!(packed_values, layout.packed_values());
        let rows = gradient.reshape([
            layout.hidden_dims + layout.update_dims,
            layout.max_row_dims,
            identities,
        ]);

        let w1_rows = rows.clone().narrow(0, 0, layout.hidden_dims);
        let w1 = w1_rows.clone().narrow(1, 0, layout.perception_dims);
        let b1 = w1_rows.clone().narrow(1, layout.perception_dims, 1);
        let w1_norm = w1
            .clone()
            .mul(w1.clone())
            .sum_dim(0)
            .sum_dim(1)
            .sqrt()
            .add_scalar(1.0e-8);
        let b1_norm = b1
            .clone()
            .mul(b1.clone())
            .sum_dim(0)
            .sum_dim(1)
            .sqrt()
            .add_scalar(1.0e-8);
        let mut normalized_w1 = vec![
            w1.div(w1_norm.expand([
                layout.hidden_dims,
                layout.perception_dims,
                identities,
            ])),
            b1.div(b1_norm.expand([layout.hidden_dims, 1, identities])),
        ];
        let w1_padding = layout
            .max_row_dims
            .saturating_sub(layout.perception_dims + 1);
        if w1_padding > 0 {
            normalized_w1.push(
                w1_rows
                    .narrow(1, layout.perception_dims + 1, w1_padding)
                    .zeros_like(),
            );
        }
        let normalized_w1 = Tensor::cat(normalized_w1, 1);

        let w2_rows = rows.narrow(0, layout.hidden_dims, layout.update_dims);
        let w2 = w2_rows.clone().narrow(1, 0, layout.hidden_dims);
        let b2 = w2_rows.clone().narrow(1, layout.hidden_dims, 1);
        let w2_norm = w2
            .clone()
            .mul(w2.clone())
            .sum_dim(0)
            .sum_dim(1)
            .sqrt()
            .add_scalar(1.0e-8);
        let normalized_b2 = if layout.output_bias {
            let b2_norm = b2
                .clone()
                .mul(b2.clone())
                .sum_dim(0)
                .sum_dim(1)
                .sqrt()
                .add_scalar(1.0e-8);
            b2.div(b2_norm.expand([layout.update_dims, 1, identities]))
        } else {
            b2.zeros_like()
        };
        let mut normalized_w2 = vec![
            w2.div(w2_norm.expand([
                layout.update_dims,
                layout.hidden_dims,
                identities,
            ])),
            normalized_b2,
        ];
        let w2_padding = layout
            .max_row_dims
            .saturating_sub(layout.hidden_dims + 1);
        if w2_padding > 0 {
            normalized_w2.push(
                w2_rows
                    .narrow(1, layout.hidden_dims + 1, w2_padding)
                    .zeros_like(),
            );
        }
        Tensor::cat(vec![normalized_w1, Tensor::cat(normalized_w2, 1)], 0)
            .reshape([packed_values, identities])
    }

    pub(super) fn tensor_l2_norm_tensor(tensor: &Tensor2Inner) -> Tensor1Inner {
        tensor.clone().mul(tensor.clone()).sum().sqrt()
    }

    pub(super) fn tensor_l2_norm(tensor: &Tensor2Inner) -> AutomataResult<f32> {
        finite_scalar(
            "Burn direct tensor norm",
            tensor_l2_norm_tensor(tensor).into_scalar(),
        )
    }

    pub(super) fn adamw_from_sgd(cfg: SgdConfig) -> AdamWConfig {
        AdamWConfig {
            learning_rate: cfg.learning_rate,
            weight_decay: cfg.weight_decay,
            grad_clip_norm: cfg.grad_clip_norm,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1.0e-8,
        }
    }

    pub(super) fn milestone_lr_scale(
        phase_step: usize,
        milestones: &[usize],
        gamma: f32,
    ) -> f32 {
        let passed = milestones
            .iter()
            .filter(|milestone| phase_step > **milestone)
            .count();
        gamma.powi(passed.min(i32::MAX as usize) as i32)
    }

    pub(super) fn oracle_repetition_position(
        global_step: usize,
        steps_per_repetition: usize,
    ) -> (usize, usize, usize) {
        let steps_per_repetition = steps_per_repetition.max(1);
        let zero_based_step = global_step.max(1) - 1;
        let repetition = zero_based_step / steps_per_repetition;
        let phase_step = zero_based_step % steps_per_repetition + 1;
        let upstream_epoch = (phase_step - 1)
            .saturating_add(repetition.saturating_mul(steps_per_repetition - 1));
        (repetition, phase_step, upstream_epoch)
    }

    pub(super) fn apply_adamw_tensor(
        param: Tensor2Inner,
        grad: Tensor2Inner,
        moment: &mut Tensor2Inner,
        velocity: &mut Tensor2Inner,
        cfg: AdamWConfig,
        scale: Tensor1Inner,
        bias: AdamWBiasCorrection,
    ) -> Tensor2Inner {
        let dims = param.shape().dims::<2>();
        let grad = grad.mul(scale.expand(dims));
        let decayed = if cfg.weight_decay > 0.0 {
            param
                .clone()
                .mul_scalar(1.0 - cfg.learning_rate * cfg.weight_decay)
        } else {
            param.clone()
        };
        *moment = moment.clone().mul_scalar(cfg.beta1) + grad.clone().mul_scalar(1.0 - cfg.beta1);
        *velocity = velocity.clone().mul_scalar(cfg.beta2)
            + grad.clone().mul(grad).mul_scalar(1.0 - cfg.beta2);
        let normalized_step = moment
            .clone()
            .div_scalar(bias.beta1.max(f32::MIN_POSITIVE))
            .div(
                velocity
                    .clone()
                    .div_scalar(bias.beta2.max(f32::MIN_POSITIVE))
                    .sqrt()
                    .add_scalar(cfg.epsilon),
            );
        decayed - normalized_step.mul_scalar(cfg.learning_rate)
    }

    pub(super) struct SparseIdentityAdamW<'a> {
        pub(super) identity_steps: &'a mut [usize],
        pub(super) active_identities: &'a [usize],
        pub(super) upstream_growing_min_lr_scale: Option<f32>,
    }

    pub(super) fn apply_sparse_column_adamw_tensor(
        param: Tensor2Inner,
        grad: Tensor2Inner,
        moment: &mut Tensor2Inner,
        velocity: &mut Tensor2Inner,
        cfg: AdamWConfig,
        scale: Tensor1Inner,
        sparse: SparseIdentityAdamW<'_>,
    ) -> AutomataResult<Tensor2Inner> {
        let SparseIdentityAdamW {
            identity_steps,
            active_identities,
            upstream_growing_min_lr_scale,
        } = sparse;
        let [rows, identities] = param.shape().dims::<2>();
        if identity_steps.len() != identities {
            return Err(AutomataError::InvalidArgument(format!(
                "sparse AdamW identity state has {} entries for {identities} parameter columns",
                identity_steps.len()
            )));
        }
        let mut active = vec![0.0_f32; identities];
        for &identity in active_identities {
            let Some(active) = active.get_mut(identity) else {
                return Err(AutomataError::InvalidArgument(format!(
                    "sparse AdamW identity {identity} exceeds {identities} parameter columns"
                )));
            };
            *active = 1.0;
        }
        let mut beta1_bias = vec![1.0_f32; identities];
        let mut beta2_bias = vec![1.0_f32; identities];
        let mut learning_rate_scale = vec![1.0_f32; identities];
        let mut reset_moments = vec![0.0_f32; identities];
        for (identity, is_active) in active.iter().enumerate() {
            if *is_active == 0.0 {
                continue;
            }
            identity_steps[identity] = identity_steps[identity].saturating_add(1);
            let (step, lr_scale, reset) = upstream_growing_min_lr_scale.map_or_else(
                || {
                    (
                        identity_steps[identity].min(i32::MAX as usize),
                        1.0,
                        false,
                    )
                },
                |min_lr_scale| {
                    upstream_growing_identity_schedule(
                        identity_steps[identity],
                        min_lr_scale,
                    )
                },
            );
            let step = step as i32;
            beta1_bias[identity] = (1.0 - cfg.beta1.powi(step)).max(f32::MIN_POSITIVE);
            beta2_bias[identity] = (1.0 - cfg.beta2.powi(step)).max(f32::MIN_POSITIVE);
            learning_rate_scale[identity] = lr_scale;
            reset_moments[identity] = if reset { 1.0 } else { 0.0 };
        }

        let device = param.device();
        let active = Tensor::<InnerBackend, 2>::from_data(
            TensorData::new(active, [1, identities]),
            &device,
        )
        .expand([rows, identities]);
        let inactive = active.clone().neg().add_scalar(1.0);
        let beta1_bias = Tensor::<InnerBackend, 2>::from_data(
            TensorData::new(beta1_bias, [1, identities]),
            &device,
        )
        .expand([rows, identities]);
        let beta2_bias = Tensor::<InnerBackend, 2>::from_data(
            TensorData::new(beta2_bias, [1, identities]),
            &device,
        )
        .expand([rows, identities]);
        let learning_rate_scale = Tensor::<InnerBackend, 2>::from_data(
            TensorData::new(learning_rate_scale, [1, identities]),
            &device,
        )
        .expand([rows, identities]);
        let retain_moments = Tensor::<InnerBackend, 2>::from_data(
            TensorData::new(reset_moments, [1, identities]),
            &device,
        )
        .expand([rows, identities])
        .neg()
        .add_scalar(1.0);
        let grad = grad
            .mul(scale.expand([rows, identities]))
            .mul(active.clone());
        let prior_moment = moment.clone().mul(retain_moments.clone());
        let prior_velocity = velocity.clone().mul(retain_moments);
        let next_moment = prior_moment.mul_scalar(cfg.beta1)
            + grad.clone().mul_scalar(1.0 - cfg.beta1);
        let next_velocity = prior_velocity.mul_scalar(cfg.beta2)
            + grad.clone().mul(grad).mul_scalar(1.0 - cfg.beta2);
        *moment = moment.clone().mul(inactive.clone()) + next_moment.mul(active.clone());
        *velocity = velocity.clone().mul(inactive) + next_velocity.mul(active.clone());

        let normalized_step = moment
            .clone()
            .div(beta1_bias)
            .div(velocity.clone().div(beta2_bias).sqrt().add_scalar(cfg.epsilon));
        let update = normalized_step
            .mul(active.clone())
            .mul(learning_rate_scale.clone())
            .mul_scalar(cfg.learning_rate);
        let decayed = if cfg.weight_decay > 0.0 {
            param.clone()
                - param
                    .clone()
                    .mul(active)
                    .mul(learning_rate_scale)
                    .mul_scalar(cfg.learning_rate * cfg.weight_decay)
        } else {
            param
        };
        Ok(decayed - update)
    }

    pub(super) fn upstream_growing_identity_schedule(
        identity_step: usize,
        min_lr_scale: f32,
    ) -> (usize, f32, bool) {
        let phase_step = identity_step.saturating_sub(1) % 10_000 + 1;
        let milestones_passed = phase_step.saturating_sub(1).div_euclid(2_000).min(4);
        let raw_scale = 0.3_f32.powi(milestones_passed as i32);
        let min_lr_scale = min_lr_scale.clamp(0.0, 1.0);
        let lr_scale = min_lr_scale + (1.0 - min_lr_scale) * raw_scale;
        let reset_moments = identity_step > 1 && phase_step == 1;
        (phase_step, lr_scale, reset_moments)
    }

    pub(super) fn mean_upstream_growing_identity_lr_scale(
        identity_steps: &[usize],
        active_identities: &[usize],
        min_lr_scale: f32,
    ) -> f32 {
        let mut identities = active_identities.to_vec();
        identities.sort_unstable();
        identities.dedup();
        if identities.is_empty() {
            return 1.0;
        }
        identities
            .iter()
            .map(|identity| {
                let next_step = identity_steps
                    .get(*identity)
                    .copied()
                    .unwrap_or_default()
                    .saturating_add(1);
                upstream_growing_identity_schedule(next_step, min_lr_scale).1
            })
            .sum::<f32>()
            / identities.len() as f32
    }

    pub(super) fn apply_adamw_tensor3(
        param: Tensor3Inner,
        grad: Tensor3Inner,
        moment: &mut Tensor3Inner,
        velocity: &mut Tensor3Inner,
        cfg: AdamWConfig,
        scale: Tensor1Inner,
        bias: AdamWBiasCorrection,
    ) -> Tensor3Inner {
        let dims = param.shape().dims::<3>();
        let grad = grad.mul(scale.reshape([dims[0], 1, 1]).expand(dims));
        let decayed = if cfg.weight_decay > 0.0 {
            param
                .clone()
                .mul_scalar(1.0 - cfg.learning_rate * cfg.weight_decay)
        } else {
            param.clone()
        };
        *moment = moment.clone().mul_scalar(cfg.beta1) + grad.clone().mul_scalar(1.0 - cfg.beta1);
        *velocity = velocity.clone().mul_scalar(cfg.beta2)
            + grad.clone().mul(grad).mul_scalar(1.0 - cfg.beta2);
        let normalized_step = moment
            .clone()
            .div_scalar(bias.beta1.max(f32::MIN_POSITIVE))
            .div(
                velocity
                    .clone()
                    .div_scalar(bias.beta2.max(f32::MIN_POSITIVE))
                    .sqrt()
                    .add_scalar(cfg.epsilon),
            );
        decayed - normalized_step.mul_scalar(cfg.learning_rate)
    }

    pub(super) fn tracked_tensor(values: Vec<f32>, shape: [usize; 2], device: &BurnDevice) -> Tensor2 {
        tensor(values, shape, device).require_grad()
    }

    pub(super) fn tensor(values: Vec<f32>, shape: [usize; 2], device: &BurnDevice) -> Tensor2 {
        Tensor::<BurnBackend, 2>::from_data(TensorData::new(values, shape), device)
    }

    pub(super) fn tensor3(values: Vec<f32>, shape: [usize; 3], device: &BurnDevice) -> Tensor3 {
        Tensor::<BurnBackend, 3>::from_data(TensorData::new(values, shape), device)
    }

    pub(super) fn tensor4(values: Vec<f32>, shape: [usize; 4], device: &BurnDevice) -> Tensor4 {
        Tensor::<BurnBackend, 4>::from_data(TensorData::new(values, shape), device)
    }

    pub(super) fn tensor1(values: Vec<f32>, shape: [usize; 1], device: &BurnDevice) -> Tensor1 {
        Tensor::<BurnBackend, 1>::from_data(TensorData::new(values, shape), device)
    }

    pub(super) fn detach1(tensor: Tensor1) -> Tensor1 {
        Tensor::<BurnBackend, 1>::from_inner(tensor.inner())
    }

    pub(super) fn detach2(tensor: Tensor2) -> Tensor2 {
        Tensor::<BurnBackend, 2>::from_inner(tensor.inner())
    }

    pub(super) fn detach3(tensor: Tensor3) -> Tensor3 {
        Tensor::<BurnBackend, 3>::from_inner(tensor.inner())
    }

    pub(super) fn sync_training_device(device: &BurnDevice) -> Result<(), Box<dyn std::error::Error>> {
        let inner_device: Device<InnerBackend> = device.clone();
        <InnerBackend as Backend>::sync(&inner_device)?;
        Ok(())
    }

    pub(super) fn target_2d_detached_color_gate2(density_term: Tensor2) -> Tensor2 {
        debug_assert_eq!(
            crate::TARGET_2D_COLOR_GATE_GRADIENT,
            crate::Target2dColorGateGradient::DetachedDensity
        );
        detach2(density_term.mul_scalar(-1.0).exp())
    }

    pub(super) fn target_2d_detached_color_gate3(density_term: Tensor3) -> Tensor3 {
        debug_assert_eq!(
            crate::TARGET_2D_COLOR_GATE_GRADIENT,
            crate::Target2dColorGateGradient::DetachedDensity
        );
        detach3(density_term.mul_scalar(-1.0).exp())
    }

    pub(super) fn track(tensor: Tensor2Inner) -> Tensor2 {
        Tensor::<BurnBackend, 2>::from_inner(tensor).require_grad()
    }

    pub(super) fn track3(tensor: Tensor3Inner) -> Tensor3 {
        Tensor::<BurnBackend, 3>::from_inner(tensor).require_grad()
    }

    pub(super) fn tensor_vec(tensor: Tensor2Inner) -> AutomataResult<Vec<f32>> {
        tensor.into_data().to_vec::<f32>().map_err(|err| {
            AutomataError::InvalidArgument(format!("Burn dense tensor readback failed: {err}"))
        })
    }

    pub(super) async fn tensor_vec_async(
        tensor: Tensor2Inner,
    ) -> AutomataResult<Vec<f32>> {
        tensor
            .into_data_async()
            .await
            .map_err(|err| {
                AutomataError::InvalidArgument(format!(
                    "Burn dense tensor readback failed: {err}"
                ))
            })?
            .to_vec::<f32>()
            .map_err(|err| {
                AutomataError::InvalidArgument(format!(
                    "Burn dense tensor conversion failed: {err}"
                ))
            })
    }

    pub(super) fn tensor2_snapshot(name: &str, tensor: Tensor2Inner) -> AutomataResult<E2eTensorSnapshot> {
        let shape = tensor.shape().dims::<2>();
        Ok(E2eTensorSnapshot {
            name: name.to_string(),
            shape: shape.to_vec(),
            values: tensor_vec(tensor)?,
        })
    }

    pub(super) fn tensor2_from_snapshot(
        snapshot: &E2eTensorSnapshot,
        device: &BurnDevice,
    ) -> AutomataResult<Tensor2Inner> {
        let shape: [usize; 2] = snapshot.shape.clone().try_into().map_err(|_| {
            AutomataError::InvalidArgument(format!(
                "checkpoint tensor {} is not rank two",
                snapshot.name
            ))
        })?;
        if shape[0].saturating_mul(shape[1]) != snapshot.values.len() {
            return Err(AutomataError::InvalidArgument(format!(
                "checkpoint tensor {} shape {:?} does not match {} values",
                snapshot.name,
                shape,
                snapshot.values.len()
            )));
        }
        let inner_device: Device<InnerBackend> = device.clone();
        Ok(Tensor::<InnerBackend, 2>::from_data(
            TensorData::new(snapshot.values.clone(), shape),
            &inner_device,
        ))
    }

    pub(super) fn tensor3_vec(tensor: Tensor3Inner) -> AutomataResult<Vec<f32>> {
        tensor.into_data().to_vec::<f32>().map_err(|err| {
            AutomataError::InvalidArgument(format!("Burn dense tensor readback failed: {err}"))
        })
    }

    pub(super) async fn tensor3_vec_async(tensor: Tensor3Inner) -> AutomataResult<Vec<f32>> {
        tensor
            .into_data_async()
            .await
            .map_err(|err| {
                AutomataError::InvalidArgument(format!(
                    "Burn dense tensor readback failed: {err}"
                ))
            })?
            .to_vec::<f32>()
            .map_err(|err| {
                AutomataError::InvalidArgument(format!(
                    "Burn dense tensor conversion failed: {err}"
                ))
            })
    }

    pub(super) fn tensor3_snapshot(name: &str, tensor: Tensor3Inner) -> AutomataResult<E2eTensorSnapshot> {
        let shape = tensor.shape().dims::<3>();
        Ok(E2eTensorSnapshot {
            name: name.to_string(),
            shape: shape.to_vec(),
            values: tensor3_vec(tensor)?,
        })
    }

    pub(super) fn tensor3_from_snapshot(
        snapshot: &E2eTensorSnapshot,
        device: &BurnDevice,
    ) -> AutomataResult<Tensor3Inner> {
        let shape: [usize; 3] = snapshot.shape.clone().try_into().map_err(|_| {
            AutomataError::InvalidArgument(format!(
                "checkpoint tensor {} is not rank three",
                snapshot.name
            ))
        })?;
        if shape.iter().product::<usize>() != snapshot.values.len() {
            return Err(AutomataError::InvalidArgument(format!(
                "checkpoint tensor {} shape {:?} does not match {} values",
                snapshot.name,
                shape,
                snapshot.values.len()
            )));
        }
        let inner_device: Device<InnerBackend> = device.clone();
        Ok(Tensor::<InnerBackend, 3>::from_data(
            TensorData::new(snapshot.values.clone(), shape),
            &inner_device,
        ))
    }

    pub(super) fn tensor1_vec(tensor: Tensor1Inner) -> AutomataResult<Vec<f32>> {
        tensor.into_data().to_vec::<f32>().map_err(|err| {
            AutomataError::InvalidArgument(format!("Burn dense tensor readback failed: {err}"))
        })
    }

    pub(super) async fn tensor1_vec_async(tensor: Tensor1Inner) -> AutomataResult<Vec<f32>> {
        tensor
            .into_data_async()
            .await
            .map_err(|err| {
                AutomataError::InvalidArgument(format!(
                    "Burn dense tensor readback failed: {err}"
                ))
            })?
            .to_vec::<f32>()
            .map_err(|err| {
                AutomataError::InvalidArgument(format!(
                    "Burn dense tensor conversion failed: {err}"
                ))
            })
    }

    pub(super) fn finite_scalar(name: &str, value: f32) -> AutomataResult<f32> {
        if value.is_finite() {
            Ok(value)
        } else {
            Err(AutomataError::InvalidArgument(format!(
                "{name} is not finite"
            )))
        }
    }

    pub(super) fn finite_values_summary(name: &str, values: &[f32]) -> String {
        let mut finite = 0usize;
        let mut nan = 0usize;
        let mut positive_infinite = 0usize;
        let mut negative_infinite = 0usize;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for value in values.iter().copied() {
            if value.is_nan() {
                nan += 1;
            } else if value == f32::INFINITY {
                positive_infinite += 1;
            } else if value == f32::NEG_INFINITY {
                negative_infinite += 1;
            } else {
                finite += 1;
                min = min.min(value);
                max = max.max(value);
            }
        }
        if finite == 0 {
            min = f32::NAN;
            max = f32::NAN;
        }
        format!(
            "{name}[len={} finite={} nan={} +inf={} -inf={} min={min:.6e} max={max:.6e}]",
            values.len(),
            finite,
            nan,
            positive_infinite,
            negative_infinite,
        )
    }

    pub(super) fn check_process_memory_budget(
        label: &str,
        config: DirectBasisTrainConfig,
    ) -> Result<ProcessMemorySnapshot, Box<dyn std::error::Error>> {
        let budget_bytes = config
            .system_memory_budget_gb
            .map(memory_budget_gb_to_bytes);
        let snapshot = ProcessMemorySnapshot {
            label: label.to_string(),
            rss_bytes: current_process_rss_bytes(),
            budget_bytes,
        };
        if let (Some(rss_bytes), Some(budget_bytes)) = (snapshot.rss_bytes, snapshot.budget_bytes)
            && rss_bytes > budget_bytes
        {
            return Err(std::io::Error::other(format!(
                "Burn dense direct-basis memory budget exceeded at {label}: rss={:.2} GiB budget={:.2} GiB",
                bytes_to_gib(rss_bytes),
                bytes_to_gib(budget_bytes)
            ))
            .into());
        }
        Ok(snapshot)
    }

    pub(super) fn current_process_rss_bytes() -> Option<u64> {
        let status = fs::read_to_string("/proc/self/status").ok()?;
        status.lines().find_map(|line| {
            let rest = line.strip_prefix("VmRSS:")?;
            let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            Some(kb.saturating_mul(1024))
        })
    }

    pub(super) fn check_gpu_memory_budget(
        label: &str,
        config: DirectBasisTrainConfig,
    ) -> Result<GpuMemorySnapshot, Box<dyn std::error::Error>> {
        let budget_bytes = config.gpu_memory_budget_gb.map(memory_budget_gb_to_bytes);
        let (used_bytes, total_bytes) = current_nvidia_gpu_memory_bytes();
        let snapshot = GpuMemorySnapshot {
            label: label.to_string(),
            used_bytes,
            total_bytes,
            budget_bytes,
        };
        if let (Some(used_bytes), Some(budget_bytes)) = (snapshot.used_bytes, snapshot.budget_bytes)
            && used_bytes > budget_bytes
        {
            return Err(std::io::Error::other(format!(
                "Burn dense direct-basis GPU memory budget exceeded at {label}: used={:.2} GiB budget={:.2} GiB",
                bytes_to_gib(used_bytes),
                bytes_to_gib(budget_bytes)
            ))
            .into());
        }
        Ok(snapshot)
    }

    pub(super) fn current_nvidia_gpu_memory_bytes() -> (Option<u64>, Option<u64>) {
        let output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok();
        let Some(output) = output else {
            return (None, None);
        };
        if !output.status.success() {
            return (None, None);
        }
        let text = String::from_utf8_lossy(&output.stdout);
        let Some(line) = text.lines().next() else {
            return (None, None);
        };
        let mut fields = line.split(',').map(str::trim);
        let used_mib = fields.next().and_then(|value| value.parse::<u64>().ok());
        let total_mib = fields.next().and_then(|value| value.parse::<u64>().ok());
        (
            used_mib.map(|mib| mib.saturating_mul(1024 * 1024)),
            total_mib.map(|mib| mib.saturating_mul(1024 * 1024)),
        )
    }

    pub(super) fn memory_budget_gb_to_bytes(gb: f32) -> u64 {
        (gb as f64 * 1024.0 * 1024.0 * 1024.0).round() as u64
    }

    pub(super) fn bytes_to_gib(bytes: u64) -> f64 {
        bytes as f64 / 1024.0 / 1024.0 / 1024.0
    }
