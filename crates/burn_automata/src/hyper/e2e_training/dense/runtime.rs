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
                Ok(BurnLossScalars {
                    total: finite_scalar(&format!("Burn direct total loss[{idx}]"), total)?,
                    splat: finite_scalar(&format!("Burn direct splat loss[{idx}]"), splat)?,
                    color: finite_scalar(&format!("Burn direct color loss[{idx}]"), color)?,
                    density: finite_scalar(&format!("Burn direct density loss[{idx}]"), density)?,
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

    pub(super) fn prepare_model_batch_grad_group(
        tensors: &mut [Tensor3Inner],
        clip_norm: f32,
        normalize: bool,
        collect_metrics: bool,
    ) -> AutomataResult<(Vec<f32>, Vec<f32>, Tensor1Inner)> {
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
                let dims = tensor.shape().dims::<3>();
                let norm = model_batch_tensor_l2_norm_tensor(tensor)
                    .add_scalar(1.0e-8)
                    .reshape([dims[0], 1, 1]);
                *tensor = tensor.clone().div(norm.expand(dims));
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

    pub(super) fn model_batch_group_norm_tensor(tensors: &[Tensor3Inner]) -> Tensor1Inner {
        let mut total = None::<Tensor1Inner>;
        for tensor in tensors {
            let value = model_batch_tensor_squared_norm_tensor(tensor);
            total = Some(match total {
                Some(total) => total + value,
                None => value,
            });
        }
        total.expect("model batch gradient group has tensors").sqrt()
    }

    pub(super) fn model_batch_tensor_l2_norm_tensor(tensor: &Tensor3Inner) -> Tensor1Inner {
        model_batch_tensor_squared_norm_tensor(tensor).sqrt()
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

    pub(super) fn finite_scalar(name: &str, value: f32) -> AutomataResult<f32> {
        if value.is_finite() {
            Ok(value)
        } else {
            Err(AutomataError::InvalidArgument(format!(
                "{name} is not finite"
            )))
        }
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
