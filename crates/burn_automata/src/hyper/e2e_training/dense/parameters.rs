//! Burn parameter containers, generator execution, and optimizer updates.

use super::*;

    pub(super) fn validate_warm_start_output_bias_contract(
        artifact_output_bias: Option<bool>,
        configured_output_bias: bool,
    ) -> AutomataResult<()> {
        let artifact_output_bias = artifact_output_bias.unwrap_or(true);
        if artifact_output_bias != configured_output_bias {
            return Err(AutomataError::InvalidModel(format!(
                "warm-start HyperNPA output-bias contract {artifact_output_bias} does not match configured {configured_output_bias}; legacy artifacts without adapter_output_bias are bias-enabled and cannot warm-start the upstream-aligned zero-bias trainer"
            )));
        }
        Ok(())
    }

    #[derive(Clone, Copy)]
    pub(super) struct AdamWBiasCorrection {
        pub(super) beta1: f32,
        pub(super) beta2: f32,
    }

    #[derive(Clone, Copy)]
    pub(super) struct GeneratorAdamWOptions<'a> {
        pub(super) normalize: bool,
        pub(super) collect_metrics: bool,
        pub(super) active_identities: &'a [usize],
        pub(super) upstream_growing_min_lr_scale: Option<f32>,
    }

    impl BurnBaseAdamWState {
        pub(super) fn zeros_like(params: &BurnBaseParams) -> Self {
            Self {
                step: 0,
                w1_m: params.w1.clone().inner().zeros_like(),
                w1_v: params.w1.clone().inner().zeros_like(),
                b1_m: params.b1.clone().inner().zeros_like(),
                b1_v: params.b1.clone().inner().zeros_like(),
                w2_m: params.w2.clone().inner().zeros_like(),
                w2_v: params.w2.clone().inner().zeros_like(),
                b2_m: params.b2.clone().inner().zeros_like(),
                b2_v: params.b2.clone().inner().zeros_like(),
            }
        }

        pub(super) fn next_bias_correction(&mut self, cfg: AdamWConfig) -> AdamWBiasCorrection {
            next_adamw_bias_correction(&mut self.step, cfg)
        }

        pub(super) fn snapshots(&self) -> AutomataResult<Vec<E2eTensorSnapshot>> {
            Ok(vec![
                tensor2_snapshot("base.w1.m", self.w1_m.clone())?,
                tensor2_snapshot("base.w1.v", self.w1_v.clone())?,
                tensor2_snapshot("base.b1.m", self.b1_m.clone())?,
                tensor2_snapshot("base.b1.v", self.b1_v.clone())?,
                tensor2_snapshot("base.w2.m", self.w2_m.clone())?,
                tensor2_snapshot("base.w2.v", self.w2_v.clone())?,
                tensor2_snapshot("base.b2.m", self.b2_m.clone())?,
                tensor2_snapshot("base.b2.v", self.b2_v.clone())?,
            ])
        }

        pub(super) fn restore(
            checkpoint: &E2eTrainingCheckpoint,
            device: &BurnDevice,
        ) -> AutomataResult<Self> {
            Ok(Self {
                step: checkpoint.base_optimizer_step,
                w1_m: tensor2_from_snapshot(checkpoint.tensor("base.w1.m")?, device)?,
                w1_v: tensor2_from_snapshot(checkpoint.tensor("base.w1.v")?, device)?,
                b1_m: tensor2_from_snapshot(checkpoint.tensor("base.b1.m")?, device)?,
                b1_v: tensor2_from_snapshot(checkpoint.tensor("base.b1.v")?, device)?,
                w2_m: tensor2_from_snapshot(checkpoint.tensor("base.w2.m")?, device)?,
                w2_v: tensor2_from_snapshot(checkpoint.tensor("base.w2.v")?, device)?,
                b2_m: tensor2_from_snapshot(checkpoint.tensor("base.b2.m")?, device)?,
                b2_v: tensor2_from_snapshot(checkpoint.tensor("base.b2.v")?, device)?,
            })
        }
    }

    impl BurnBaseBatchAdamWState {
        pub(super) fn zeros_like(params: &BurnBaseBatch) -> Self {
            Self {
                step: 0,
                w1_m: params.w1.clone().inner().zeros_like(),
                w1_v: params.w1.clone().inner().zeros_like(),
                b1_m: params.b1.clone().inner().zeros_like(),
                b1_v: params.b1.clone().inner().zeros_like(),
                w2_m: params.w2.clone().inner().zeros_like(),
                w2_v: params.w2.clone().inner().zeros_like(),
                b2_m: params.b2.clone().inner().zeros_like(),
                b2_v: params.b2.clone().inner().zeros_like(),
            }
        }

        pub(super) fn next_bias_correction(
            &mut self,
            cfg: AdamWConfig,
        ) -> AdamWBiasCorrection {
            next_adamw_bias_correction(&mut self.step, cfg)
        }

        pub(super) fn target2d_snapshots(&self) -> AutomataResult<Vec<E2eTensorSnapshot>> {
            Ok(vec![
                tensor3_snapshot("target2d.w1.m", self.w1_m.clone())?,
                tensor3_snapshot("target2d.w1.v", self.w1_v.clone())?,
                tensor3_snapshot("target2d.b1.m", self.b1_m.clone())?,
                tensor3_snapshot("target2d.b1.v", self.b1_v.clone())?,
                tensor3_snapshot("target2d.w2.m", self.w2_m.clone())?,
                tensor3_snapshot("target2d.w2.v", self.w2_v.clone())?,
                tensor3_snapshot("target2d.b2.m", self.b2_m.clone())?,
                tensor3_snapshot("target2d.b2.v", self.b2_v.clone())?,
            ])
        }

        pub(super) fn restore_target2d(
            checkpoint: &Target2dTrainingCheckpoint,
            params: &BurnBaseBatch,
            device: &BurnDevice,
        ) -> AutomataResult<Self> {
            fn restore_tensor(
                checkpoint: &Target2dTrainingCheckpoint,
                name: &str,
                expected_shape: [usize; 3],
                device: &BurnDevice,
            ) -> AutomataResult<Tensor3Inner> {
                let snapshot = checkpoint.tensor(name)?;
                if snapshot.shape != expected_shape {
                    return Err(AutomataError::InvalidArgument(format!(
                        "Target2D checkpoint tensor {name} shape {:?} != expected {:?}",
                        snapshot.shape, expected_shape
                    )));
                }
                tensor3_from_snapshot(snapshot, device)
            }

            Ok(Self {
                step: checkpoint.optimizer_step,
                w1_m: restore_tensor(
                    checkpoint,
                    "target2d.w1.m",
                    params.w1.shape().dims::<3>(),
                    device,
                )?,
                w1_v: restore_tensor(
                    checkpoint,
                    "target2d.w1.v",
                    params.w1.shape().dims::<3>(),
                    device,
                )?,
                b1_m: restore_tensor(
                    checkpoint,
                    "target2d.b1.m",
                    params.b1.shape().dims::<3>(),
                    device,
                )?,
                b1_v: restore_tensor(
                    checkpoint,
                    "target2d.b1.v",
                    params.b1.shape().dims::<3>(),
                    device,
                )?,
                w2_m: restore_tensor(
                    checkpoint,
                    "target2d.w2.m",
                    params.w2.shape().dims::<3>(),
                    device,
                )?,
                w2_v: restore_tensor(
                    checkpoint,
                    "target2d.w2.v",
                    params.w2.shape().dims::<3>(),
                    device,
                )?,
                b2_m: restore_tensor(
                    checkpoint,
                    "target2d.b2.m",
                    params.b2.shape().dims::<3>(),
                    device,
                )?,
                b2_v: restore_tensor(
                    checkpoint,
                    "target2d.b2.v",
                    params.b2.shape().dims::<3>(),
                    device,
                )?,
            })
        }
    }

    impl BurnAdapterAdamWState {
        pub(super) fn zeros_like(params: &BurnAdapterParams) -> Self {
            Self {
                step: 0,
                w1_down_m: params.w1_down.clone().inner().zeros_like(),
                w1_down_v: params.w1_down.clone().inner().zeros_like(),
                w1_up_m: params.w1_up.clone().inner().zeros_like(),
                w1_up_v: params.w1_up.clone().inner().zeros_like(),
                w2_down_m: params.w2_down.clone().inner().zeros_like(),
                w2_down_v: params.w2_down.clone().inner().zeros_like(),
                w2_up_m: params.w2_up.clone().inner().zeros_like(),
                w2_up_v: params.w2_up.clone().inner().zeros_like(),
                b1_delta_m: params.b1_delta.clone().inner().zeros_like(),
                b1_delta_v: params.b1_delta.clone().inner().zeros_like(),
                b2_delta_m: params.b2_delta.clone().inner().zeros_like(),
                b2_delta_v: params.b2_delta.clone().inner().zeros_like(),
            }
        }

        pub(super) fn next_bias_correction(&mut self, cfg: AdamWConfig) -> AdamWBiasCorrection {
            next_adamw_bias_correction(&mut self.step, cfg)
        }
    }

    impl BurnE2eGeneratorAdamWState {
        pub(super) fn new(params: &BurnE2eGeneratorParams) -> Self {
            Self {
                step: 0,
                row_flow_step: 0,
                amortization_step: 0,
                sample_id_steps: if params.kind == E2eHyperGeneratorKind::SampleIdTable {
                    vec![0; params.token_w.shape().dims::<2>()[1]]
                } else {
                    Vec::new()
                },
                amortization_identity_steps: params
                    .amortization_residual_table
                    .as_ref()
                    .map_or_else(Vec::new, |table| {
                        vec![0; table.shape().dims::<2>()[1]]
                    }),
            token_w_m: params.token_w.clone().inner().zeros_like(),
            token_w_v: params.token_w.clone().inner().zeros_like(),
            token_b_m: params.token_b.clone().inner().zeros_like(),
            token_b_v: params.token_b.clone().inner().zeros_like(),
            token_gate_w_m: params.token_gate_w.clone().inner().zeros_like(),
            token_gate_w_v: params.token_gate_w.clone().inner().zeros_like(),
            token_gate_b_m: params.token_gate_b.clone().inner().zeros_like(),
            token_gate_b_v: params.token_gate_b.clone().inner().zeros_like(),
            state_w_m: params.state_w.clone().inner().zeros_like(),
            state_w_v: params.state_w.clone().inner().zeros_like(),
                time_w_m: params.time_w.clone().inner().zeros_like(),
                time_w_v: params.time_w.clone().inner().zeros_like(),
                output_w_m: params.output_w.clone().inner().zeros_like(),
                output_w_v: params.output_w.clone().inner().zeros_like(),
                output_b_m: params.output_b.clone().inner().zeros_like(),
                output_b_v: params.output_b.clone().inner().zeros_like(),
                condition_control_w_m: params
                    .condition_control_w
                    .clone()
                    .inner()
                    .zeros_like(),
                condition_control_w_v: params
                    .condition_control_w
                    .clone()
                    .inner()
                    .zeros_like(),
                condition_control_b_m: params
                    .condition_control_b
                    .clone()
                    .inner()
                    .zeros_like(),
                condition_control_b_v: params
                    .condition_control_b
                    .clone()
                    .inner()
                    .zeros_like(),
                condition_control_state_w_m: params
                    .condition_control_state_w
                    .clone()
                    .inner()
                    .zeros_like(),
                condition_control_state_w_v: params
                    .condition_control_state_w
                    .clone()
                    .inner()
                    .zeros_like(),
                // The row flow is much larger than the endpoint table. Allocate
                // its moments only when the flow first receives gradients so a
                // substrate warm-up does not reserve or checkpoint idle state.
                row_flow_m: Vec::new(),
                row_flow_v: Vec::new(),
                amortization_residual_m: params
                    .amortization_residual_table
                    .as_ref()
                    .map(|table| table.clone().inner().zeros_like()),
                amortization_residual_v: params
                    .amortization_residual_table
                    .as_ref()
                    .map(|table| table.clone().inner().zeros_like()),
            }
        }

        pub(super) fn next_bias_correction(&mut self, cfg: AdamWConfig) -> AdamWBiasCorrection {
            next_adamw_bias_correction(&mut self.step, cfg)
        }

        fn next_row_flow_bias_correction(&mut self, cfg: AdamWConfig) -> AdamWBiasCorrection {
            next_adamw_bias_correction(&mut self.row_flow_step, cfg)
        }

        pub(super) fn snapshots(&self) -> AutomataResult<Vec<E2eTensorSnapshot>> {
            let tensors = [
                ("generator.token_w.m", self.token_w_m.clone()),
                ("generator.token_w.v", self.token_w_v.clone()),
                ("generator.token_b.m", self.token_b_m.clone()),
                ("generator.token_b.v", self.token_b_v.clone()),
                ("generator.token_gate_w.m", self.token_gate_w_m.clone()),
                ("generator.token_gate_w.v", self.token_gate_w_v.clone()),
                ("generator.token_gate_b.m", self.token_gate_b_m.clone()),
                ("generator.token_gate_b.v", self.token_gate_b_v.clone()),
                ("generator.state_w.m", self.state_w_m.clone()),
                ("generator.state_w.v", self.state_w_v.clone()),
                ("generator.time_w.m", self.time_w_m.clone()),
                ("generator.time_w.v", self.time_w_v.clone()),
                ("generator.output_w.m", self.output_w_m.clone()),
                ("generator.output_w.v", self.output_w_v.clone()),
                ("generator.output_b.m", self.output_b_m.clone()),
                ("generator.output_b.v", self.output_b_v.clone()),
                ("generator.condition_control_w.m", self.condition_control_w_m.clone()),
                ("generator.condition_control_w.v", self.condition_control_w_v.clone()),
                ("generator.condition_control_b.m", self.condition_control_b_m.clone()),
                ("generator.condition_control_b.v", self.condition_control_b_v.clone()),
                (
                    "generator.condition_control_state_w.m",
                    self.condition_control_state_w_m.clone(),
                ),
                (
                    "generator.condition_control_state_w.v",
                    self.condition_control_state_w_v.clone(),
                ),
            ];
            let mut snapshots = tensors
                .into_iter()
                .map(|(name, tensor)| tensor2_snapshot(name, tensor))
                .collect::<AutomataResult<Vec<_>>>()?;
            for (index, tensor) in self.row_flow_m.iter().cloned().enumerate() {
                snapshots.push(tensor2_snapshot(
                    &format!("generator.row_flow.{index}.m"),
                    tensor,
                )?);
            }
            for (index, tensor) in self.row_flow_v.iter().cloned().enumerate() {
                snapshots.push(tensor2_snapshot(
                    &format!("generator.row_flow.{index}.v"),
                    tensor,
                )?);
            }
            if let Some(tensor) = self.amortization_residual_m.clone() {
                snapshots.push(tensor2_snapshot("generator.amortization_residual.m", tensor)?);
            }
            if let Some(tensor) = self.amortization_residual_v.clone() {
                snapshots.push(tensor2_snapshot("generator.amortization_residual.v", tensor)?);
            }
            Ok(snapshots)
        }

        pub(super) fn restore(
            checkpoint: &E2eTrainingCheckpoint,
            params: &BurnE2eGeneratorParams,
            device: &BurnDevice,
            restore_row_flow: bool,
            allow_missing_amortization: bool,
        ) -> AutomataResult<Self> {
            let tensor = |name| tensor2_from_snapshot(checkpoint.tensor(name)?, device);
            let amortization_tensor = |name: &str, table: &Tensor2| {
                checkpoint.tensor_optional(name).map_or_else(
                    || {
                        if allow_missing_amortization {
                            Ok(table.clone().inner().zeros_like())
                        } else {
                            Err(AutomataError::InvalidArgument(format!(
                                "training checkpoint is missing tensor {name}"
                            )))
                        }
                    },
                    |snapshot| tensor2_from_snapshot(snapshot, device),
                )
            };
            let row_flow_count = params
                .row_flow
                .as_ref()
                .map_or(0, |flow| flow.tensors.len());
            let mut row_flow_m = Vec::with_capacity(row_flow_count);
            let mut row_flow_v = Vec::with_capacity(row_flow_count);
            for index in 0..usize::from(restore_row_flow).saturating_mul(row_flow_count) {
                let m_name = format!("generator.row_flow.{index}.m");
                let v_name = format!("generator.row_flow.{index}.v");
                match (
                    checkpoint.tensor_optional(&m_name),
                    checkpoint.tensor_optional(&v_name),
                ) {
                    (Some(m), Some(v)) => {
                        row_flow_m.push(tensor2_from_snapshot(m, device)?);
                        row_flow_v.push(tensor2_from_snapshot(v, device)?);
                    }
                    (None, None) if index == 0 => break,
                    (None, None) => {
                        return Err(AutomataError::InvalidArgument(
                            "training checkpoint contains a partial row-flow optimizer state"
                                .to_string(),
                        ));
                    }
                    _ => {
                        return Err(AutomataError::InvalidArgument(format!(
                            "training checkpoint must contain both {m_name} and {v_name}"
                        )));
                    }
                }
            }
            Ok(Self {
                step: checkpoint.generator_optimizer_step,
                row_flow_step: checkpoint
                    .row_flow_optimizer_step
                    .unwrap_or(checkpoint.generator_optimizer_step)
                    * usize::from(restore_row_flow),
                amortization_step: checkpoint
                    .amortization_optimizer_step
                    .unwrap_or(checkpoint.generator_optimizer_step),
                sample_id_steps: if params.kind == E2eHyperGeneratorKind::SampleIdTable {
                    let identities = params.token_w.shape().dims::<2>()[1];
                    if checkpoint.sample_id_optimizer_steps.is_empty() {
                        vec![checkpoint.generator_optimizer_step; identities]
                    } else if checkpoint.sample_id_optimizer_steps.len() == identities {
                        checkpoint.sample_id_optimizer_steps.clone()
                    } else {
                        return Err(AutomataError::InvalidArgument(format!(
                            "training checkpoint has {} sample-ID optimizer steps for {identities} identities",
                            checkpoint.sample_id_optimizer_steps.len()
                        )));
                    }
                } else {
                    Vec::new()
                },
                amortization_identity_steps: params
                    .amortization_residual_table
                    .as_ref()
                    .map(|table| {
                        let identities = table.shape().dims::<2>()[1];
                        if checkpoint.amortization_identity_optimizer_steps.is_empty() {
                            Ok(vec![
                                checkpoint
                                    .amortization_optimizer_step
                                    .unwrap_or(checkpoint.generator_optimizer_step);
                                identities
                            ])
                        } else if checkpoint.amortization_identity_optimizer_steps.len()
                            == identities
                        {
                            Ok(checkpoint.amortization_identity_optimizer_steps.clone())
                        } else {
                            Err(AutomataError::InvalidArgument(format!(
                                "training checkpoint has {} amortization optimizer steps for {identities} identities",
                                checkpoint.amortization_identity_optimizer_steps.len()
                            )))
                        }
                    })
                    .transpose()?
                    .unwrap_or_default(),
                token_w_m: tensor("generator.token_w.m")?,
                token_w_v: tensor("generator.token_w.v")?,
                token_b_m: tensor("generator.token_b.m")?,
                token_b_v: tensor("generator.token_b.v")?,
                token_gate_w_m: tensor("generator.token_gate_w.m")?,
                token_gate_w_v: tensor("generator.token_gate_w.v")?,
                token_gate_b_m: tensor("generator.token_gate_b.m")?,
                token_gate_b_v: tensor("generator.token_gate_b.v")?,
                state_w_m: tensor("generator.state_w.m")?,
                state_w_v: tensor("generator.state_w.v")?,
                time_w_m: tensor("generator.time_w.m")?,
                time_w_v: tensor("generator.time_w.v")?,
                output_w_m: tensor("generator.output_w.m")?,
                output_w_v: tensor("generator.output_w.v")?,
                output_b_m: tensor("generator.output_b.m")?,
                output_b_v: tensor("generator.output_b.v")?,
                condition_control_w_m: tensor("generator.condition_control_w.m")?,
                condition_control_w_v: tensor("generator.condition_control_w.v")?,
                condition_control_b_m: tensor("generator.condition_control_b.m")?,
                condition_control_b_v: tensor("generator.condition_control_b.v")?,
                condition_control_state_w_m: tensor(
                    "generator.condition_control_state_w.m",
                )?,
                condition_control_state_w_v: tensor(
                    "generator.condition_control_state_w.v",
                )?,
                row_flow_m,
                row_flow_v,
                amortization_residual_m: params
                    .amortization_residual_table
                    .as_ref()
                    .map(|table| {
                        amortization_tensor("generator.amortization_residual.m", table)
                    })
                    .transpose()?,
                amortization_residual_v: params
                    .amortization_residual_table
                    .as_ref()
                    .map(|table| {
                        amortization_tensor("generator.amortization_residual.v", table)
                    })
                    .transpose()?,
            })
        }
    }

    impl BurnE2eGeneratorParams {
        pub(super) fn from_seed_or_artifact(
            base: &NpaModel,
            examples: &[BurnE2eRolloutExample],
            config: BurnE2eRolloutTrainConfig,
            initial: Option<&E2eHyperNpa2d>,
            device: &BurnDevice,
        ) -> AutomataResult<Self> {
            match initial {
                Some(initial) => Self::from_artifact(base, examples, config, initial, device),
                None => Self::seeded(base, examples, config, device),
            }
        }

        fn with_row_flow(
            base: &NpaModel,
            examples: &[BurnE2eRolloutExample],
            config: BurnE2eRolloutTrainConfig,
            row_flow: BurnRowFlowParams,
            device: &BurnDevice,
        ) -> AutomataResult<Self> {
            let layout = NpaParameterRowLayout2d::new(&base.config);
            layout.validate_flow_config(&row_flow.config)?;
            let canonical_rank = layout.canonical_rank();
            if !config.canonical_full_rank_lora
                || config.adapter_rank != canonical_rank
                || (config.adapter_alpha - canonical_rank as f32).abs() > f32::EPSILON
            {
                return Err(AutomataError::InvalidArgument(format!(
                    "conditional row flow requires canonical-full-rank adapters with rank/alpha {canonical_rank}/{canonical_rank}, got {}/{}, canonical={}",
                    config.adapter_rank, config.adapter_alpha, config.canonical_full_rank_lora
                )));
            }
            let placeholder = || tracked_tensor(vec![0.0], [1, 1], device);
            let amortization_residual_table = if config.amortization_enabled {
                if config.generator_optimizer.learning_rate <= 0.0 {
                    return Err(AutomataError::InvalidArgument(
                        "amortization requires a positive generator learning rate".to_string(),
                    ));
                }
                let row_values = row_flow.config.row_count * row_flow.config.max_row_dims;
                let mut values = vec![0.0; row_values * examples.len()];
                if config.amortization_initialize_from_teacher {
                    for (identity, example) in examples.iter().enumerate() {
                        let teacher = example.teacher_adapter.as_ref().ok_or_else(|| {
                            AutomataError::InvalidArgument(
                                "teacher-initialized amortization requires every training example to provide an exact adapter"
                                    .to_string(),
                            )
                        })?;
                        let adapter = NpaLowRankAdapter::from_parameter_vector(
                            &base.config,
                            config.adapter_rank,
                            config.adapter_alpha,
                            teacher.clone(),
                        )?;
                        let packed = layout.adapter_to_packed(&adapter)?;
                        for (row, value) in packed.into_iter().enumerate() {
                            values[row * examples.len() + identity] = value;
                        }
                    }
                }
                Some(tracked_tensor(
                    values,
                    [row_values, examples.len()],
                    device,
                ))
            } else {
                None
            };
            let amortization_gradient_layout = Some(PackedNpaGradientLayout::new(
                &base.config,
                config.adapter_output_bias,
            ));
            Ok(Self {
                kind: E2eHyperGeneratorKind::ConditionalRowFlow,
                token_w: placeholder(),
                token_b: placeholder(),
                token_gate_w: placeholder(),
                token_gate_b: placeholder(),
                state_w: placeholder(),
                time_w: placeholder(),
                output_w: placeholder(),
                output_b: placeholder(),
                condition_control_w: placeholder(),
                condition_control_b: placeholder(),
                condition_control_state_w: placeholder(),
                hidden_dims: row_flow.config.width,
                token_attention_heads: row_flow.config.heads,
                softmax_token_attention: true,
                canonical_full_rank_lora: true,
                adapter_constants: placeholder(),
                adapter_trainable_mask: placeholder(),
                adapter_parameter_segments: Vec::new(),
                output_dims: layout.parameter_count(),
                output_scale: row_flow.config.source_scale,
                sample_steps: row_flow.config.sample_steps,
                adapter_chunk_size: row_flow.config.max_row_dims,
                output_chunks: row_flow.config.row_count,
                row_flow: Some(row_flow),
                amortization_residual_table,
                amortization_gradient_layout,
                amortization_learning_rate_scale: config.amortization_learning_rate
                    / config.generator_optimizer.learning_rate,
                amortization_grad_normalization: config.amortization_grad_normalization,
            })
        }

        pub(super) fn adapter_parameterization_tensors(
            base: &NpaModel,
            config: BurnE2eRolloutTrainConfig,
            device: &BurnDevice,
        ) -> AutomataResult<(Tensor2, Tensor2)> {
            let output_dims =
                NpaLowRankAdapter::parameter_count_for_config(&base.config, config.adapter_rank);
            let (constants, mask) = if config.canonical_full_rank_lora {
                let canonical = crate::hyper::adapter_layout::CanonicalFullRankLora2d::new_with_output_bias(
                    &base.config,
                    config.adapter_rank,
                    config.adapter_alpha,
                    config.adapter_output_bias,
                )?;
                (canonical.constants, canonical.trainable_mask)
            } else {
                let mut mask = vec![1.0; output_dims];
                if !config.adapter_output_bias {
                    mask[output_dims - base.config.update_dims()..].fill(0.0);
                }
                (vec![0.0; output_dims], mask)
            };
            Ok((
                tensor(constants, [1, output_dims], device),
                tensor(mask, [1, output_dims], device),
            ))
        }

        pub(super) fn adapter_parameter_segments(config: &NpaConfig, rank: usize) -> Vec<(usize, usize)> {
            let lengths = [
                rank * config.perception_dims(),
                config.hidden_dims * rank,
                rank * config.hidden_dims,
                config.update_dims() * rank,
                config.hidden_dims,
                config.update_dims(),
            ];
            let mut offset = 0usize;
            lengths
                .into_iter()
                .map(|len| {
                    let segment = (offset, len);
                    offset += len;
                    segment
                })
                .collect()
        }

        pub(super) fn from_artifact(
            base: &NpaModel,
            examples: &[BurnE2eRolloutExample],
            config: BurnE2eRolloutTrainConfig,
            initial: &E2eHyperNpa2d,
            device: &BurnDevice,
        ) -> AutomataResult<Self> {
            initial.validate()?;
            let first = examples.first().ok_or_else(|| {
                AutomataError::InvalidArgument("HyperNPA e2e generator requires examples".into())
            })?;
            if examples.iter().any(|example| {
                example.embed_dims != first.embed_dims
                    || example.token_count != first.token_count
            }) {
                return Err(AutomataError::InvalidArgument(
                    "HyperNPA e2e examples must have homogeneous condition token shapes"
                        .to_string(),
                ));
            }
            let expected_kind = config.generator_kind;
            if initial.architecture != expected_kind.artifact_architecture() {
                return Err(AutomataError::InvalidModel(format!(
                    "warm-start HyperNPA architecture {:?} does not match configured {:?}",
                    initial.architecture,
                    expected_kind.artifact_architecture()
                )));
            }
            validate_warm_start_output_bias_contract(
                initial.adapter_output_bias,
                config.adapter_output_bias,
            )?;
            if expected_kind == E2eHyperGeneratorKind::ConditionalRowFlow {
                let flow = BurnRowFlowParams::from_artifact_with_output_bias(
                    initial,
                    &base.config,
                    config.adapter_output_bias,
                    device,
                )?;
                if flow.config.condition_tokens != first.token_count
                    || flow.config.condition_dims != first.embed_dims
                    || flow.config.layers != config.generator_layers
                    || flow.config.width != config.generator_hidden_dims
                    || flow.config.heads != config.token_attention_heads.max(1)
                    || flow.config.ffn_dims != config.generator_ffn_dims
                    || flow.config.sample_steps != config.generator_sample_steps.max(1)
                {
                    return Err(AutomataError::InvalidModel(
                        "warm-start conditional row flow contract does not match training config"
                            .to_string(),
                    ));
                }
                return Self::with_row_flow(base, examples, config, flow, device);
            }
            if initial.uses_canonical_full_rank_lora() != config.canonical_full_rank_lora {
                return Err(AutomataError::InvalidModel(format!(
                    "warm-start HyperNPA adapter parameterization {:?} does not match configured {:?}",
                    initial
                        .adapter_parameterization
                        .as_deref()
                        .unwrap_or(E2E_HYPER_ADAPTER_FACTORIZED),
                    if config.canonical_full_rank_lora {
                        E2E_HYPER_ADAPTER_CANONICAL_FULL_RANK
                    } else {
                        E2E_HYPER_ADAPTER_FACTORIZED
                    }
                )));
            }
            let (adapter_constants, adapter_trainable_mask) =
                Self::adapter_parameterization_tensors(base, config, device)?;
            let initial_embed_dims = initial.embed_dims()?;
            let adding_rgb_channels = expected_kind != E2eHyperGeneratorKind::SampleIdTable
                && !initial.condition_rgb_channels.unwrap_or(false)
                && config.dino_rgb_channels
                && first.embed_dims == initial_embed_dims + 3;
            if initial.condition_token_count.is_some_and(|value| value != first.token_count)
                || (initial_embed_dims != first.embed_dims && !adding_rgb_channels)
            {
                return Err(AutomataError::InvalidModel(format!(
                    "warm-start HyperNPA condition shape {:?}x{} does not match training {}x{}",
                    initial.condition_token_count,
                    initial.embed_dims()?,
                    first.token_count,
                    first.embed_dims
                )));
            }
            let adapter = initial.adapter_spec(&base.config)?;
            if adapter.rank != config.adapter_rank
                || (adapter.alpha - config.adapter_alpha).abs() > f32::EPSILON
            {
                return Err(AutomataError::InvalidModel(format!(
                    "warm-start HyperNPA adapter rank/alpha {}/{} does not match configured {}/{}",
                    adapter.rank, adapter.alpha, config.adapter_rank, config.adapter_alpha
                )));
            }
            if expected_kind != E2eHyperGeneratorKind::SampleIdTable
                && (initial.hidden_dims != config.generator_hidden_dims
                    || initial.token_attention_heads != config.token_attention_heads.max(1)
                    || initial.sample_steps != config.generator_sample_steps.max(1)
                    || (initial.output_scale - config.generator_output_scale).abs()
                        > f32::EPSILON)
            {
                return Err(AutomataError::InvalidModel(
                    "warm-start HyperNPA hidden/sample/output-scale contract does not match the configured generator"
                        .to_string(),
                ));
            }
            if expected_kind != E2eHyperGeneratorKind::SampleIdTable
                && ((!adding_rgb_channels
                    && (initial.condition_rgb_channels.unwrap_or(false)
                        != config.dino_rgb_channels
                        || (initial.condition_rgb_channel_scale.unwrap_or(1.0)
                            - config.dino_rgb_channel_scale)
                            .abs()
                            > f32::EPSILON))
                    || initial.condition_alpha_channel.unwrap_or(false)
                        != config.dino_alpha_channel
                    || (initial.condition_alpha_channel_scale.unwrap_or(1.0)
                        - config.dino_alpha_channel_scale)
                        .abs()
                        > f32::EPSILON
                    || initial.condition_patch_pixels.unwrap_or(false)
                        != config.dino_patch_pixels
                    || initial.condition_l2_normalize_features.unwrap_or(true)
                        != config.dino_l2_normalize_features)
            {
                return Err(AutomataError::InvalidModel(
                    "warm-start HyperNPA DINO RGB/alpha/patch-pixel/normalization contract does not match the configured condition pipeline"
                        .to_string(),
                ));
            }
            let output_dims =
                NpaLowRankAdapter::parameter_count_for_config(&base.config, config.adapter_rank);
            if initial.output_dims != output_dims {
                return Err(AutomataError::InvalidModel(format!(
                    "warm-start HyperNPA output dims {} do not match model adapter dims {output_dims}",
                    initial.output_dims
                )));
            }
            if expected_kind == E2eHyperGeneratorKind::SampleIdTable {
                if first.token_count != 1 || initial.hidden_dims != 1 {
                    return Err(AutomataError::InvalidModel(
                        "sample-ID adapter table requires one token and hidden_dims=1".to_string(),
                    ));
                }
                let placeholder = |values: Vec<f32>| tracked_tensor(values, [1, 1], device);
                return Ok(Self {
                    kind: expected_kind,
                    token_w: tracked_tensor(
                        initial.weights.token_w.clone(),
                        [output_dims, first.embed_dims],
                        device,
                    ),
                    token_b: placeholder(initial.weights.token_b.clone()),
                    token_gate_w: placeholder(initial.weights.token_gate_w.clone()),
                    token_gate_b: placeholder(initial.weights.token_gate_b.clone()),
                    state_w: placeholder(initial.weights.state_w.clone()),
                    time_w: placeholder(initial.weights.time_w.clone()),
                    output_w: placeholder(initial.weights.output_w.clone()),
                    output_b: placeholder(initial.weights.output_b.clone()),
                    condition_control_w: tracked_tensor(
                        vec![0.0; base.config.update_dims()],
                        [base.config.update_dims(), 1],
                        device,
                    ),
                    condition_control_b: tracked_tensor(
                        vec![0.0; base.config.update_dims()],
                        [1, base.config.update_dims()],
                        device,
                    ),
                    condition_control_state_w: tracked_tensor(
                        vec![0.0; base.config.state_dims],
                        [1, base.config.state_dims],
                        device,
                    ),
                    hidden_dims: 1,
                    token_attention_heads: 1,
                    softmax_token_attention: false,
                    canonical_full_rank_lora: config.canonical_full_rank_lora,
                    adapter_constants,
                    adapter_trainable_mask,
                    adapter_parameter_segments: Self::adapter_parameter_segments(
                        &base.config,
                        config.adapter_rank,
                    ),
                    output_dims,
                    output_scale: 1.0,
                    sample_steps: 1,
                    adapter_chunk_size: output_dims,
                    output_chunks: 1,
                    row_flow: None,
                    amortization_residual_table: None,
                    amortization_gradient_layout: None,
                    amortization_learning_rate_scale: 1.0,
                    amortization_grad_normalization: false,
                });
            }
            let adapter_chunk_size = if expected_kind == E2eHyperGeneratorKind::PooledFlow {
                output_dims
            } else {
                initial.adapter_chunk_size_value()
            };
            if adapter_chunk_size != config.adapter_chunk_size.max(1).min(output_dims).max(1)
                && expected_kind != E2eHyperGeneratorKind::PooledFlow
            {
                return Err(AutomataError::InvalidModel(format!(
                    "warm-start HyperNPA adapter chunk size {adapter_chunk_size} does not match configured {}",
                    config.adapter_chunk_size
                )));
            }
            let output_chunks = initial.weights.output_b.len() / adapter_chunk_size;
            let expected_control = config.spatial_condition_control;
            let initial_has_control = initial.has_spatial_condition_control();
            if initial_has_control && !expected_control {
                return Err(AutomataError::InvalidModel(
                    "cannot warm-start a per-step condition-field HyperNPA as a static-adapter model"
                        .to_string(),
                ));
            }
            let initial_has_state_control =
                initial.spatial_condition_state_control.unwrap_or(false);
            if initial_has_state_control && !config.spatial_condition_state_control {
                return Err(AutomataError::InvalidModel(
                    "cannot warm-start a state-conditioned field as a state-independent field"
                        .to_string(),
                ));
            }
            let update_dims = base.config.update_dims();
            let control_w = if initial_has_control {
                initial.weights.condition_control_w.clone()
            } else if expected_control {
                let mut rng = StdRng::seed_from_u64(config.seed ^ 0xc01d_f1e1_d2d0);
                seeded_values(
                    update_dims * initial.hidden_dims,
                    config.generator_output_init_scale
                        / (initial.hidden_dims as f32).sqrt().max(1.0),
                    &mut rng,
                )
            } else {
                vec![0.0; update_dims * initial.hidden_dims]
            };
            let control_b = if initial_has_control {
                initial.weights.condition_control_b.clone()
            } else {
                vec![0.0; update_dims]
            };
            let control_state_w = if initial_has_state_control {
                if initial.weights.condition_control_state_w.len()
                    != initial.hidden_dims * base.config.state_dims
                {
                    return Err(AutomataError::InvalidModel(format!(
                        "warm-start condition state projection has {} values, expected {}",
                        initial.weights.condition_control_state_w.len(),
                        initial.hidden_dims * base.config.state_dims,
                    )));
                }
                initial.weights.condition_control_state_w.clone()
            } else {
                vec![0.0; initial.hidden_dims * base.config.state_dims]
            };
            let token_w = if adding_rgb_channels {
                let has_alpha = initial.condition_alpha_channel.unwrap_or(false);
                let semantic_dims = initial_embed_dims - usize::from(has_alpha);
                let mut expanded = vec![0.0; initial.hidden_dims * first.embed_dims];
                for hidden in 0..initial.hidden_dims {
                    let old = &initial.weights.token_w
                        [hidden * initial_embed_dims..(hidden + 1) * initial_embed_dims];
                    let new = &mut expanded
                        [hidden * first.embed_dims..(hidden + 1) * first.embed_dims];
                    new[..semantic_dims].copy_from_slice(&old[..semantic_dims]);
                    if has_alpha {
                        new[first.embed_dims - 1] = old[initial_embed_dims - 1];
                    }
                }
                eprintln!(
                    "warm-starting HyperNPA with RGB token channels: condition projection {} -> {} dimensions",
                    initial_embed_dims, first.embed_dims,
                );
                expanded
            } else {
                initial.weights.token_w.clone()
            };
            Ok(Self {
                kind: expected_kind,
                token_w: tracked_tensor(token_w, [initial.hidden_dims, first.embed_dims], device),
                token_b: tracked_tensor(
                    initial.weights.token_b.clone(),
                    [1, initial.hidden_dims],
                    device,
                ),
                token_gate_w: tracked_tensor(
                    initial.weights.token_gate_w.clone(),
                    if expected_kind == E2eHyperGeneratorKind::PooledFlow {
                        [initial.token_attention_heads, initial.hidden_dims]
                    } else {
                        [output_chunks, initial.hidden_dims]
                    },
                    device,
                ),
                token_gate_b: tracked_tensor(
                    initial.weights.token_gate_b.clone(),
                    if expected_kind == E2eHyperGeneratorKind::PooledFlow {
                        [1, initial.token_attention_heads]
                    } else {
                        [output_chunks, initial.hidden_dims]
                    },
                    device,
                ),
                state_w: tracked_tensor(
                    initial.weights.state_w.clone(),
                    [initial.hidden_dims, adapter_chunk_size],
                    device,
                ),
                time_w: tracked_tensor(
                    initial.weights.time_w.clone(),
                    [initial.hidden_dims, 1],
                    device,
                ),
                output_w: tracked_tensor(
                    initial.weights.output_w.clone(),
                    [adapter_chunk_size, initial.hidden_dims],
                    device,
                ),
                output_b: tracked_tensor(
                    initial.weights.output_b.clone(),
                    [output_chunks, adapter_chunk_size],
                    device,
                ),
                condition_control_w: tracked_tensor(
                    control_w,
                    [update_dims, initial.hidden_dims],
                    device,
                ),
                condition_control_b: tracked_tensor(control_b, [1, update_dims], device),
                condition_control_state_w: tracked_tensor(
                    control_state_w,
                    [initial.hidden_dims, base.config.state_dims],
                    device,
                ),
                hidden_dims: initial.hidden_dims,
                token_attention_heads: initial.token_attention_heads,
                softmax_token_attention: config.softmax_token_attention,
                canonical_full_rank_lora: config.canonical_full_rank_lora,
                adapter_constants,
                adapter_trainable_mask,
                adapter_parameter_segments: Self::adapter_parameter_segments(
                    &base.config,
                    config.adapter_rank,
                ),
                output_dims,
                output_scale: initial.output_scale,
                sample_steps: initial.sample_steps,
                adapter_chunk_size,
                output_chunks,
                row_flow: None,
                amortization_residual_table: None,
                amortization_gradient_layout: None,
                amortization_learning_rate_scale: 1.0,
                amortization_grad_normalization: false,
            })
        }

        pub(super) fn seeded(
            base: &NpaModel,
            examples: &[BurnE2eRolloutExample],
            config: BurnE2eRolloutTrainConfig,
            device: &BurnDevice,
        ) -> AutomataResult<Self> {
            let first = examples.first().ok_or_else(|| {
                AutomataError::InvalidArgument("HyperNPA e2e generator requires examples".into())
            })?;
            let embed_dims = first.embed_dims;
            let token_count = first.token_count;
            if embed_dims == 0 || token_count == 0 {
                return Err(AutomataError::InvalidArgument(
                    "condition token dimensions must be positive".to_string(),
                ));
            }
            if examples.iter().any(|example| {
                example.embed_dims != embed_dims || example.token_count != token_count
            }) {
                return Err(AutomataError::InvalidArgument(
                    "HyperNPA e2e examples must have homogeneous condition token shapes"
                        .to_string(),
                ));
            }
            let output_dims =
                NpaLowRankAdapter::parameter_count_for_config(&base.config, config.adapter_rank);
            let (adapter_constants, adapter_trainable_mask) =
                Self::adapter_parameterization_tensors(base, config, device)?;
            let kind = config.generator_kind;
            if kind == E2eHyperGeneratorKind::ConditionalRowFlow {
                let flow = BurnRowFlowParams::seeded(
                    &base.config,
                    examples,
                    config,
                    token_count,
                    embed_dims,
                    device,
                )?;
                return Self::with_row_flow(base, examples, config, flow, device);
            }
            if kind == E2eHyperGeneratorKind::SampleIdTable {
                if token_count != 1 {
                    return Err(AutomataError::InvalidArgument(
                        "sample-ID adapter table requires exactly one condition token".to_string(),
                    ));
                }
                let initial_adapter = if config.canonical_full_rank_lora {
                    vec![0.0; output_dims]
                } else {
                    NpaLowRankAdapter::seeded_zero_delta(
                        &base.config,
                        config.adapter_rank,
                        config.adapter_alpha,
                        config.seed ^ 0x5eed_10da,
                    )
                    .to_parameter_vector()
                };
                let mut table = vec![0.0; output_dims * embed_dims];
                for (output, value) in initial_adapter.into_iter().enumerate() {
                    table[output * embed_dims..(output + 1) * embed_dims].fill(value);
                }
                let placeholder = || tracked_tensor(vec![0.0], [1, 1], device);
                return Ok(Self {
                    kind,
                    token_w: tracked_tensor(table, [output_dims, embed_dims], device),
                    token_b: placeholder(),
                    token_gate_w: placeholder(),
                    token_gate_b: placeholder(),
                    state_w: placeholder(),
                    time_w: placeholder(),
                    output_w: placeholder(),
                    output_b: placeholder(),
                    condition_control_w: tracked_tensor(
                        vec![0.0; base.config.update_dims()],
                        [base.config.update_dims(), 1],
                        device,
                    ),
                    condition_control_b: tracked_tensor(
                        vec![0.0; base.config.update_dims()],
                        [1, base.config.update_dims()],
                        device,
                    ),
                    condition_control_state_w: tracked_tensor(
                        vec![0.0; base.config.state_dims],
                        [1, base.config.state_dims],
                        device,
                    ),
                    hidden_dims: 1,
                    token_attention_heads: 1,
                    softmax_token_attention: false,
                    canonical_full_rank_lora: config.canonical_full_rank_lora,
                    adapter_constants,
                    adapter_trainable_mask,
                    adapter_parameter_segments: Self::adapter_parameter_segments(
                        &base.config,
                        config.adapter_rank,
                    ),
                    output_dims,
                    output_scale: 1.0,
                    sample_steps: 1,
                    adapter_chunk_size: output_dims,
                    output_chunks: 1,
                    row_flow: None,
                    amortization_residual_table: None,
                    amortization_gradient_layout: None,
                    amortization_learning_rate_scale: 1.0,
                    amortization_grad_normalization: false,
                });
            }
            let hidden_dims = config.generator_hidden_dims.max(1);
            let token_attention_heads = config.token_attention_heads.max(1);
            if kind == E2eHyperGeneratorKind::ModuleTokenDecoder
                && !hidden_dims.is_multiple_of(token_attention_heads)
            {
                return Err(AutomataError::InvalidArgument(format!(
                    "module-token-decoder hidden_dims={hidden_dims} must be divisible by token_attention_heads={token_attention_heads}"
                )));
            }
            let adapter_chunk_size = match kind {
                E2eHyperGeneratorKind::PooledFlow => output_dims,
                E2eHyperGeneratorKind::SpatialTokenFlow
                | E2eHyperGeneratorKind::ModuleTokenDecoderV2
                | E2eHyperGeneratorKind::ModuleTokenDecoder => config
                    .adapter_chunk_size
                    .max(1)
                    .min(output_dims)
                    .max(1),
                E2eHyperGeneratorKind::SampleIdTable => unreachable!(),
                E2eHyperGeneratorKind::ConditionalRowFlow => unreachable!(),
            };
            let module_layout = kind
                .is_module_token_decoder()
                .then(|| {
                    crate::hyper::adapter_layout::AdapterParameterLayout2d::new(
                        &base.config,
                        config.adapter_rank,
                        adapter_chunk_size,
                    )
                })
                .transpose()?;
            let output_chunks = module_layout
                .as_ref()
                .map_or_else(|| output_dims.div_ceil(adapter_chunk_size), |layout| layout.chunk_count);
            let mut rng = StdRng::seed_from_u64(config.seed ^ 0xa11c_e2e0_7a5e);
            let token_w = tracked_tensor(
                seeded_values(
                    hidden_dims * embed_dims,
                    config.generator_condition_init_scale / (embed_dims as f32).sqrt().max(1.0),
                    &mut rng,
                ),
                [hidden_dims, embed_dims],
                device,
            );
            let token_b = tracked_tensor(vec![0.0; hidden_dims], [1, hidden_dims], device);
            let (token_gate_w, token_gate_b, state_w) = match kind {
                E2eHyperGeneratorKind::PooledFlow => (
                    tracked_tensor(
                        seeded_values(
                            token_attention_heads * hidden_dims,
                            config.generator_condition_init_scale
                                / (hidden_dims as f32).sqrt().max(1.0),
                            &mut rng,
                        ),
                        [token_attention_heads, hidden_dims],
                        device,
                    ),
                    tracked_tensor(
                        vec![0.0; token_attention_heads],
                        [1, token_attention_heads],
                        device,
                    ),
                    tracked_tensor(
                        seeded_values(
                            hidden_dims * output_dims,
                            config.generator_condition_init_scale
                                / (output_dims as f32).sqrt().max(1.0),
                            &mut rng,
                        ),
                        [hidden_dims, output_dims],
                        device,
                    ),
                ),
                E2eHyperGeneratorKind::SpatialTokenFlow
                | E2eHyperGeneratorKind::ModuleTokenDecoderV2
                | E2eHyperGeneratorKind::ModuleTokenDecoder => (
                    tracked_tensor(
                        seeded_values(
                            output_chunks * hidden_dims,
                            config.generator_condition_init_scale
                                / (hidden_dims as f32).sqrt().max(1.0),
                            &mut rng,
                        ),
                        [output_chunks, hidden_dims],
                        device,
                    ),
                    tracked_tensor(
                        module_layout.as_ref().map_or_else(
                            || vec![0.0; output_chunks * hidden_dims],
                            |layout| {
                                layout.structured_query_initialization(
                                    hidden_dims,
                                    config.generator_condition_init_scale,
                                )
                            },
                        ),
                        [output_chunks, hidden_dims],
                        device,
                    ),
                    tracked_tensor(
                        seeded_values(
                            hidden_dims * adapter_chunk_size,
                            config.generator_condition_init_scale
                                / (adapter_chunk_size as f32).sqrt().max(1.0),
                            &mut rng,
                        ),
                        [hidden_dims, adapter_chunk_size],
                        device,
                    ),
                ),
                E2eHyperGeneratorKind::SampleIdTable => unreachable!(),
                E2eHyperGeneratorKind::ConditionalRowFlow => unreachable!(),
            };
            let time_w = tracked_tensor(
                seeded_values(
                    hidden_dims,
                    config.generator_condition_init_scale,
                    &mut rng,
                ),
                [hidden_dims, 1],
                device,
            );
            let (output_w, output_b) = match kind {
                E2eHyperGeneratorKind::PooledFlow => (
                    tracked_tensor(
                        seeded_values(
                            output_dims * hidden_dims,
                            config.generator_output_init_scale
                                / (hidden_dims as f32).sqrt().max(1.0),
                            &mut rng,
                        ),
                        [output_dims, hidden_dims],
                        device,
                    ),
                    tracked_tensor(
                        if config.canonical_full_rank_lora {
                            vec![0.0; output_dims]
                        } else {
                            seeded_zero_delta_output_bias(
                                &base.config,
                                config.adapter_rank,
                                config.adapter_alpha,
                                config.seed ^ 0x5eed_10da,
                                config.generator_output_scale,
                            )
                        },
                        [1, output_dims],
                        device,
                    ),
                ),
                E2eHyperGeneratorKind::SpatialTokenFlow
                | E2eHyperGeneratorKind::ModuleTokenDecoderV2
                | E2eHyperGeneratorKind::ModuleTokenDecoder => (
                    tracked_tensor(
                        seeded_values(
                            adapter_chunk_size * hidden_dims,
                            config.generator_output_init_scale
                                / (hidden_dims as f32).sqrt().max(1.0),
                            &mut rng,
                        ),
                        [adapter_chunk_size, hidden_dims],
                        device,
                    ),
                    tracked_tensor(
                        if config.canonical_full_rank_lora {
                            vec![0.0; output_chunks * adapter_chunk_size]
                        } else {
                            seeded_zero_delta_chunk_output_bias(
                                &base.config,
                                config.adapter_rank,
                                config.adapter_alpha,
                                config.seed ^ 0x5eed_10da,
                                adapter_chunk_size,
                                output_chunks,
                                module_layout.as_ref(),
                            )
                        },
                        [output_chunks, adapter_chunk_size],
                        device,
                    ),
                ),
                E2eHyperGeneratorKind::SampleIdTable => unreachable!(),
                E2eHyperGeneratorKind::ConditionalRowFlow => unreachable!(),
            };
            let condition_control_w = tracked_tensor(
                seeded_values(
                    base.config.update_dims() * hidden_dims,
                    config.generator_output_init_scale
                        / (hidden_dims as f32).sqrt().max(1.0),
                    &mut rng,
                ),
                [base.config.update_dims(), hidden_dims],
                device,
            );
            let condition_control_b = tracked_tensor(
                vec![0.0; base.config.update_dims()],
                [1, base.config.update_dims()],
                device,
            );
            let condition_control_state_w = tracked_tensor(
                vec![0.0; hidden_dims * base.config.state_dims],
                [hidden_dims, base.config.state_dims],
                device,
            );
            Ok(Self {
                kind,
                token_w,
                token_b,
                token_gate_w,
                token_gate_b,
                state_w,
                time_w,
                output_w,
                output_b,
                condition_control_w,
                condition_control_b,
                condition_control_state_w,
                hidden_dims,
                token_attention_heads,
                softmax_token_attention: config.softmax_token_attention,
                canonical_full_rank_lora: config.canonical_full_rank_lora,
                adapter_constants,
                adapter_trainable_mask,
                adapter_parameter_segments: Self::adapter_parameter_segments(
                    &base.config,
                    config.adapter_rank,
                ),
                output_dims,
                output_scale: config.generator_output_scale,
                sample_steps: config.generator_sample_steps.max(1),
                adapter_chunk_size,
                output_chunks,
                row_flow: None,
                amortization_residual_table: None,
                amortization_gradient_layout: None,
                amortization_learning_rate_scale: 1.0,
                amortization_grad_normalization: false,
            })
        }

        pub(super) fn token_hidden_batch(&self, condition: Tensor3) -> Tensor3 {
            let dims = condition.shape().dims::<3>();
            let batches = dims[0];
            let tokens = dims[1];
            let embed_dims = dims[2];
            let token_w = self
                .token_w
                .clone()
                .transpose()
                .unsqueeze_dim::<3>(0)
                .expand([batches, embed_dims, self.hidden_dims]);
            let token_b = self
                .token_b
                .clone()
                .unsqueeze_dim::<3>(0)
                .expand([batches, tokens, self.hidden_dims]);
            relu(condition.matmul(token_w) + token_b)
        }

        pub(super) fn condition_control_batch(
            &self,
            condition: Tensor3,
            config: BurnE2eRolloutTrainConfig,
        ) -> Option<BurnE2eConditionControlBatch> {
            if !config.spatial_condition_control {
                return None;
            }
            let dims = condition.shape().dims::<3>();
            let tokens = dims[1];
            let grid_width = config.dino_token_grid_width.max(1);
            let grid_height = config.dino_token_grid_height.max(1);
            let patch_tokens = grid_width.saturating_mul(grid_height);
            if tokens <= 1 || patch_tokens == 0 || tokens < patch_tokens.saturating_add(1) {
                return None;
            }
            let token_hidden = self.token_hidden_batch(condition);
            Some(BurnE2eConditionControlBatch {
                patch_hidden: token_hidden.narrow(1, 1, patch_tokens),
                update_w: self.condition_control_w.clone(),
                update_b: self.condition_control_b.clone(),
                state_w: config
                    .spatial_condition_state_control
                    .then(|| self.condition_control_state_w.clone()),
                grid_width,
                grid_height,
                sigma: config.spatial_condition_control_sigma.max(1.0e-4),
                scale: config.spatial_condition_control_scale,
            })
        }

        pub(super) fn apply_adapter_parameterization(&self, vector: Tensor2) -> Tensor2 {
            let batches = vector.shape().dims::<2>()[0];
            vector.mul(
                self.adapter_trainable_mask
                    .clone()
                    .expand([batches, self.output_dims]),
            ) + self
                .adapter_constants
                .clone()
                .expand([batches, self.output_dims])
        }

        pub(super) fn adapter_batch(
            &self,
            condition: Tensor3,
            npa_config: &NpaConfig,
            config: BurnE2eRolloutTrainConfig,
        ) -> BurnAdapterBatch {
            if let Some(flow) = &self.row_flow {
                return flow.sample_adapter_batch(condition, npa_config);
            }
            if self.kind == E2eHyperGeneratorKind::SampleIdTable {
                let dims = condition.shape().dims::<3>();
                debug_assert_eq!(dims[1], 1);
                let vector = condition
                    .squeeze_dim::<2>(1)
                    .matmul(self.token_w.clone().transpose());
                return BurnAdapterBatch::from_parameter_vector(
                    self.apply_adapter_parameterization(vector),
                    npa_config,
                    config.adapter_rank,
                    config.adapter_alpha,
                );
            }
            if matches!(
                self.kind,
                E2eHyperGeneratorKind::SpatialTokenFlow
                    | E2eHyperGeneratorKind::ModuleTokenDecoderV2
                    | E2eHyperGeneratorKind::ModuleTokenDecoder
            ) {
                let vector = self.spatial_token_adapter_vector_batch(
                    condition,
                    npa_config,
                    config.adapter_rank,
                );
                return BurnAdapterBatch::from_parameter_vector(
                    self.apply_adapter_parameterization(vector),
                    npa_config,
                    config.adapter_rank,
                    config.adapter_alpha,
                );
            }
            let dims = condition.shape().dims::<3>();
            let batches = dims[0];
            let tokens = dims[1];
            let device = condition.device();
            let token_hidden = self.token_hidden_batch(condition);
            let mean_pooled = token_hidden
                .clone()
                .sum_dim(1)
                .squeeze_dim::<2>(1)
                .div_scalar(tokens.max(1) as f32);
            let heads = self.token_attention_heads.max(1);
            let gate_w = self
                .token_gate_w
                .clone()
                .transpose()
                .unsqueeze_dim::<3>(0)
                .expand([batches, self.hidden_dims, heads]);
            let gate_b = self
                .token_gate_b
                .clone()
                .unsqueeze_dim::<3>(0)
                .expand([batches, tokens, heads]);
            let attention_logits = token_hidden.clone().matmul(gate_w) + gate_b;
            let attention_weights = if self.softmax_token_attention {
                softmax(attention_logits, 1)
            } else {
                let weights = attention_logits.tanh().exp();
                let denominator = weights
                    .clone()
                    .sum_dim(1)
                    .add_scalar(EPSILON)
                    .expand([batches, tokens, heads]);
                weights.div(denominator)
            };
            let attended = attention_weights
                .swap_dims(1, 2)
                .matmul(token_hidden)
                .sum_dim(1)
                .squeeze_dim::<2>(1)
                .div_scalar(heads as f32);
            let pooled = (mean_pooled + attended).div_scalar(2.0);
            let mut vector = Tensor::<BurnBackend, 2>::zeros([batches, self.output_dims], &device);
            for step in 0..self.sample_steps {
                let t = if self.sample_steps <= 1 {
                    0.0
                } else {
                    step as f32 / (self.sample_steps - 1) as f32
                };
                let state_hidden = vector.clone().matmul(self.state_w.clone().transpose());
                let time_hidden = self
                    .time_w
                    .clone()
                    .transpose()
                    .mul_scalar(t)
                    .expand([batches, self.hidden_dims]);
                let hidden = relu(
                    pooled.clone()
                        + state_hidden
                        + time_hidden
                        + self.token_b.clone().expand([batches, self.hidden_dims]),
                );
                let velocity = hidden.matmul(self.output_w.clone().transpose())
                    + self.output_b.clone().expand([batches, self.output_dims]);
                vector = vector + velocity.div_scalar(self.sample_steps as f32);
            }
            let vector = vector.tanh().mul_scalar(self.output_scale);
            BurnAdapterBatch::from_parameter_vector(
                self.apply_adapter_parameterization(vector),
                npa_config,
                config.adapter_rank,
                config.adapter_alpha,
            )
        }

        pub(super) fn amortization_residual_rows(
            &self,
            indices: &[usize],
        ) -> Option<Tensor3> {
            let table = self.amortization_residual_table.as_ref()?;
            let flow = self
                .row_flow
                .as_ref()
                .expect("amortization is restricted to conditional row flow");
            let device = table.device();
            let selected = table.clone().transpose().select(
                0,
                Tensor::<BurnBackend, 1, Int>::from_data(
                    TensorData::new(
                        indices.iter().map(|index| *index as i64).collect::<Vec<_>>(),
                        [indices.len()],
                    ),
                    &device,
                ),
            );
            Some(selected.reshape([
                indices.len(),
                flow.config.row_count,
                flow.config.max_row_dims,
            ]))
        }

        pub(super) fn amortization_snapshot(
            &self,
        ) -> AutomataResult<Option<E2eTensorSnapshot>> {
            self.amortization_residual_table
                .as_ref()
                .map(|table| {
                    tensor2_snapshot(
                        "generator.amortization_residual.parameter",
                        table.clone().inner(),
                    )
                })
                .transpose()
        }

        pub(super) fn restore_amortization(
            &mut self,
            checkpoint: &E2eTrainingCheckpoint,
            device: &BurnDevice,
            allow_missing: bool,
        ) -> AutomataResult<bool> {
            let Some(current) = self.amortization_residual_table.as_ref() else {
                return Ok(false);
            };
            let expected = current.shape().dims::<2>();
            let Some(snapshot) = checkpoint
                .tensor_optional("generator.amortization_residual.parameter")
            else {
                if allow_missing {
                    return Ok(false);
                }
                return Err(AutomataError::InvalidArgument(
                    "training checkpoint is missing tensor generator.amortization_residual.parameter"
                        .to_string(),
                ));
            };
            let restored = tensor2_from_snapshot(snapshot, device)?;
            if restored.shape().dims::<2>() != expected {
                return Err(AutomataError::InvalidArgument(format!(
                    "amortization checkpoint shape {:?} does not match {:?}",
                    restored.shape().dims::<2>(),
                    expected,
                )));
            }
            self.amortization_residual_table = Some(track(restored));
            Ok(true)
        }

        pub(super) fn initialize_amortization_from_rows(
            &mut self,
            rows: Tensor3,
        ) -> AutomataResult<()> {
            let table = self.amortization_residual_table.as_ref().ok_or_else(|| {
                AutomataError::InvalidArgument(
                    "generator endpoint initialization requires an amortization table"
                        .to_string(),
                )
            })?;
            let [identities, row_count, row_dims] = rows.shape().dims::<3>();
            let expected = table.shape().dims::<2>();
            if [row_count * row_dims, identities] != expected {
                return Err(AutomataError::InvalidArgument(format!(
                    "generated endpoint table shape [{}, {}] does not match {:?}",
                    row_count * row_dims,
                    identities,
                    expected,
                )));
            }
            let initialized = rows
                .reshape([identities, row_count * row_dims])
                .transpose();
            self.amortization_residual_table = Some(track(initialized.inner()));
            Ok(())
        }

        pub(super) fn adapter_batch_with_dense_rows(
            &self,
            condition: Tensor3,
            npa_config: &NpaConfig,
            config: BurnE2eRolloutTrainConfig,
        ) -> (
            BurnAdapterBatch,
            Option<Tensor3>,
            Option<BurnRowFlowCondition>,
        ) {
            if let Some(flow) = &self.row_flow {
                let (rows, prepared) = flow.sample_rows_with_prepared_steps(
                    condition,
                    npa_config,
                    config.generator_train_sample_steps,
                );
                return (
                    BurnAdapterBatch::from_dense_residual_rows(rows.clone(), npa_config),
                    Some(rows),
                    Some(prepared),
                );
            }
            (
                self.adapter_batch(condition, npa_config, config),
                None,
                None,
            )
        }

        pub(super) fn spatial_token_adapter_vector_batch(
            &self,
            condition: Tensor3,
            npa_config: &NpaConfig,
            adapter_rank: usize,
        ) -> Tensor2 {
            let dims = condition.shape().dims::<3>();
            let batches = dims[0];
            let device = condition.device();
            let token_hidden = self.token_hidden_batch(condition);
            let mut chunks = Tensor::<BurnBackend, 3>::zeros(
                [batches, self.output_chunks, self.adapter_chunk_size],
                &device,
            );
            let query_base = self
                .token_gate_w
                .clone()
                .unsqueeze_dim::<3>(0)
                .expand([batches, self.output_chunks, self.hidden_dims])
                + self
                    .token_gate_b
                    .clone()
                    .unsqueeze_dim::<3>(0)
                    .expand([batches, self.output_chunks, self.hidden_dims]);
            let shared_bias = self
                .token_b
                .clone()
                .unsqueeze_dim::<3>(0)
                .expand([batches, self.output_chunks, self.hidden_dims]);
            let state_w = self
                .state_w
                .clone()
                .transpose()
                .unsqueeze_dim::<3>(0)
                .expand([batches, self.adapter_chunk_size, self.hidden_dims]);
            let output_w = self
                .output_w
                .clone()
                .transpose()
                .unsqueeze_dim::<3>(0)
                .expand([batches, self.hidden_dims, self.adapter_chunk_size]);
            let output_b = self
                .output_b
                .clone()
                .unsqueeze_dim::<3>(0)
                .expand([batches, self.output_chunks, self.adapter_chunk_size]);
            let attention_heads = if self.kind == E2eHyperGeneratorKind::ModuleTokenDecoder {
                self.token_attention_heads
            } else {
                1
            };
            let head_dims = self.hidden_dims / attention_heads;
            let attention_scale = 1.0 / (head_dims as f32).sqrt().max(1.0);
            for step in 0..self.sample_steps {
                let t = if self.sample_steps <= 1 {
                    0.0
                } else {
                    step as f32 / (self.sample_steps - 1) as f32
                };
                let time_hidden = self
                    .time_w
                    .clone()
                    .transpose()
                    .mul_scalar(t)
                    .unsqueeze_dim::<3>(0)
                    .expand([batches, self.output_chunks, self.hidden_dims]);
                let state_hidden = chunks.clone().matmul(state_w.clone());
                let query_hidden = relu(
                    query_base.clone() + shared_bias.clone() + state_hidden + time_hidden,
                );
                let attend = |query: Tensor3, tokens: Tensor3, hidden: usize| {
                    let attention_logits = query.matmul(tokens.clone().swap_dims(1, 2));
                    let attention_logits = attention_logits.mul_scalar(attention_scale);
                    let attention_weights = if self.softmax_token_attention {
                        softmax(attention_logits, 2)
                    } else {
                        let weights = attention_logits.tanh().exp();
                        let denominator = weights
                            .clone()
                            .sum_dim(2)
                            .add_scalar(EPSILON)
                            .expand([batches, self.output_chunks, dims[1]]);
                        weights.div(denominator)
                    };
                    debug_assert_eq!(tokens.shape().dims::<3>()[2], hidden);
                    attention_weights.matmul(tokens)
                };
                let attended = if attention_heads == 1 {
                    attend(query_hidden.clone(), token_hidden.clone(), self.hidden_dims)
                } else {
                    Tensor::cat(
                        (0..attention_heads)
                            .map(|head| {
                                let start = head * head_dims;
                                attend(
                                    query_hidden.clone().narrow(2, start, head_dims),
                                    token_hidden.clone().narrow(2, start, head_dims),
                                    head_dims,
                                )
                            })
                            .collect(),
                        2,
                    )
                };
                let hidden = relu(query_hidden + attended);
                let velocity = hidden.matmul(output_w.clone()) + output_b.clone();
                chunks = chunks
                    + velocity
                        .mul_scalar(self.output_scale)
                        .div_scalar(self.sample_steps as f32);
            }
            let padded = chunks.reshape([
                batches,
                self.output_chunks * self.adapter_chunk_size,
            ]);
            if self.kind.is_module_token_decoder() {
                let layout = crate::hyper::adapter_layout::AdapterParameterLayout2d::new(
                    npa_config,
                    adapter_rank,
                    self.adapter_chunk_size,
                )
                .expect("module adapter layout validated during generator construction");
                assert_eq!(layout.chunk_count, self.output_chunks);
                Tensor::cat(
                    layout
                        .segments
                        .iter()
                        .map(|segment| {
                            padded
                                .clone()
                                .narrow(1, segment.chunk_offset * self.adapter_chunk_size, segment.len)
                        })
                        .collect(),
                    1,
                )
            } else {
                padded.narrow(1, 0, self.output_dims)
            }
        }

        pub(super) fn apply_adamw(
            &mut self,
            grads: &mut <BurnBackend as burn::tensor::backend::AutodiffBackend>::Gradients,
            state: &mut BurnE2eGeneratorAdamWState,
            cfg: AdamWConfig,
            options: GeneratorAdamWOptions<'_>,
        ) -> AutomataResult<(f32, f32, f32)> {
            let tensors = self.take_gradients(grads);
            self.apply_adamw_gradients(tensors, state, cfg, options)
        }

        pub(super) fn apply_row_flow_adamw(
            &mut self,
            grads: &mut <BurnBackend as burn::tensor::backend::AutodiffBackend>::Gradients,
            state: &mut BurnE2eGeneratorAdamWState,
            cfg: AdamWConfig,
            normalize: bool,
            collect_metrics: bool,
        ) -> AutomataResult<(f32, f32)> {
            let flow = self.row_flow.as_mut().ok_or_else(|| {
                AutomataError::InvalidArgument(
                    "row-flow-only optimizer update requires conditional-row-flow".to_string(),
                )
            })?;
            let tensors = flow
                .tensors
                .iter()
                .map(|tensor| {
                    tensor
                        .grad_remove(grads)
                        .unwrap_or_else(|| tensor.clone().inner().zeros_like())
                })
                .collect();
            Self::apply_row_flow_gradients(
                flow,
                tensors,
                state,
                cfg,
                normalize,
                collect_metrics,
            )
        }

        fn apply_row_flow_gradients(
            flow: &mut BurnRowFlowParams,
            mut tensors: Vec<Tensor2Inner>,
            state: &mut BurnE2eGeneratorAdamWState,
            cfg: AdamWConfig,
            normalize: bool,
            collect_metrics: bool,
        ) -> AutomataResult<(f32, f32)> {
            if state.row_flow_m.is_empty() && state.row_flow_v.is_empty() {
                state.row_flow_m = flow
                    .tensors
                    .iter()
                    .map(|tensor| tensor.clone().inner().zeros_like())
                    .collect();
                state.row_flow_v = flow
                    .tensors
                    .iter()
                    .map(|tensor| tensor.clone().inner().zeros_like())
                    .collect();
            }
            if state.row_flow_m.len() != flow.tensors.len()
                || state.row_flow_v.len() != flow.tensors.len()
                || tensors.len() != flow.tensors.len()
            {
                return Err(AutomataError::InvalidArgument(
                    "conditional row flow optimizer state does not match model tensors"
                        .to_string(),
                ));
            }
            let (norm, scale, scale_tensor) = prepare_grad_group(
                &mut tensors,
                cfg.grad_clip_norm,
                normalize,
                collect_metrics,
            )?;
            let bias = state.next_row_flow_bias_correction(cfg);
            for (index, gradient) in tensors.into_iter().enumerate() {
                flow.tensors[index] = track(apply_adamw_tensor(
                    flow.tensors[index].clone().inner(),
                    gradient,
                    &mut state.row_flow_m[index],
                    &mut state.row_flow_v[index],
                    cfg,
                    scale_tensor.clone(),
                    bias,
                ));
            }
            state.step = state.step.max(state.row_flow_step);
            Ok((norm, scale))
        }

        pub(super) fn apply_amortization_adamw(
            &mut self,
            grads: &mut <BurnBackend as burn::tensor::backend::AutodiffBackend>::Gradients,
            state: &mut BurnE2eGeneratorAdamWState,
            cfg: AdamWConfig,
            collect_metrics: bool,
            active_identities: &[usize],
            upstream_growing_min_lr_scale: Option<f32>,
        ) -> AutomataResult<f32> {
            let table = self.amortization_residual_table.as_mut().ok_or_else(|| {
                AutomataError::InvalidArgument(
                    "amortization-only update requires an endpoint table".to_string(),
                )
            })?;
            let mut gradient = table
                .grad_remove(grads)
                .unwrap_or_else(|| table.clone().inner().zeros_like());
            if self.amortization_grad_normalization {
                gradient = normalize_packed_npa_table_gradient(
                    gradient,
                    self.amortization_gradient_layout
                        .expect("amortization table has an NPA gradient layout"),
                );
            }
            let table_cfg = AdamWConfig {
                learning_rate: cfg.learning_rate * self.amortization_learning_rate_scale,
                weight_decay: 0.0,
                ..cfg
            };
            let (norm, _, scale) = prepare_grad_group(
                std::slice::from_mut(&mut gradient),
                table_cfg.grad_clip_norm,
                false,
                collect_metrics,
            )?;
            let moment = state.amortization_residual_m.as_mut().ok_or_else(|| {
                AutomataError::InvalidArgument(
                    "amortization optimizer is missing first moments".to_string(),
                )
            })?;
            let velocity = state.amortization_residual_v.as_mut().ok_or_else(|| {
                AutomataError::InvalidArgument(
                    "amortization optimizer is missing second moments".to_string(),
                )
            })?;
            *table = track(apply_sparse_column_adamw_tensor(
                table.clone().inner(),
                gradient,
                moment,
                velocity,
                table_cfg,
                scale,
                SparseIdentityAdamW {
                    identity_steps: &mut state.amortization_identity_steps,
                    active_identities,
                    upstream_growing_min_lr_scale,
                },
            )?);
            state.amortization_step = state.amortization_step.saturating_add(1);
            state.step = state.step.max(state.amortization_step);
            Ok(norm)
        }

        pub(super) fn take_gradients(
            &self,
            grads: &mut <BurnBackend as burn::tensor::backend::AutodiffBackend>::Gradients,
        ) -> Vec<Tensor2Inner> {
            if let Some(flow) = &self.row_flow {
                let mut tensors = flow
                    .tensors
                    .iter()
                    .map(|tensor| {
                        tensor
                            .grad_remove(grads)
                            .unwrap_or_else(|| tensor.clone().inner().zeros_like())
                    })
                    .collect::<Vec<_>>();
                if let Some(table) = &self.amortization_residual_table {
                    tensors.push(
                        table
                            .grad_remove(grads)
                            .unwrap_or_else(|| table.clone().inner().zeros_like()),
                    );
                }
                return tensors;
            }
            vec![
                self.token_w
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.token_w.clone().inner().zeros_like()),
                self.token_b
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.token_b.clone().inner().zeros_like()),
                self.token_gate_w
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.token_gate_w.clone().inner().zeros_like()),
                self.token_gate_b
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.token_gate_b.clone().inner().zeros_like()),
                self.state_w
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.state_w.clone().inner().zeros_like()),
                self.time_w
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.time_w.clone().inner().zeros_like()),
                self.output_w
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.output_w.clone().inner().zeros_like()),
                self.output_b
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.output_b.clone().inner().zeros_like()),
                self.condition_control_w
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.condition_control_w.clone().inner().zeros_like()),
                self.condition_control_b
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.condition_control_b.clone().inner().zeros_like()),
                self.condition_control_state_w
                    .grad_remove(grads)
                    .unwrap_or_else(|| {
                        self.condition_control_state_w.clone().inner().zeros_like()
                    }),
            ]
        }

        pub(super) fn apply_adamw_gradients(
            &mut self,
            mut tensors: Vec<Tensor2Inner>,
            state: &mut BurnE2eGeneratorAdamWState,
            cfg: AdamWConfig,
            options: GeneratorAdamWOptions<'_>,
        ) -> AutomataResult<(f32, f32, f32)> {
            let GeneratorAdamWOptions {
                normalize,
                collect_metrics,
                active_identities,
                upstream_growing_min_lr_scale,
            } = options;
            if let Some(flow) = &mut self.row_flow {
                let table_gradient = self
                    .amortization_residual_table
                    .as_ref()
                    .map(|_| tensors.pop().expect("amortization gradient is present"));
                let (norm, scale) = Self::apply_row_flow_gradients(
                    flow,
                    tensors,
                    state,
                    cfg,
                    normalize,
                    collect_metrics,
                )?;
                let amortization_grad_norm = if let Some(mut gradient) = table_gradient {
                    let table = self
                        .amortization_residual_table
                        .as_mut()
                        .expect("amortization table matches its gradient");
                    let table_cfg = AdamWConfig {
                        learning_rate: cfg.learning_rate
                            * self.amortization_learning_rate_scale,
                        weight_decay: 0.0,
                        ..cfg
                    };
                    let moment = state.amortization_residual_m.as_mut().ok_or_else(|| {
                        AutomataError::InvalidArgument(
                            "amortization optimizer is missing first moments".to_string(),
                        )
                    })?;
                    let velocity = state.amortization_residual_v.as_mut().ok_or_else(|| {
                        AutomataError::InvalidArgument(
                            "amortization optimizer is missing second moments".to_string(),
                        )
                    })?;
                    if self.amortization_grad_normalization {
                        gradient = normalize_packed_npa_table_gradient(
                            gradient,
                            self.amortization_gradient_layout
                                .expect("amortization table has an NPA gradient layout"),
                        );
                    }
                    let (table_norm, _, table_scale) = prepare_grad_group(
                        std::slice::from_mut(&mut gradient),
                        table_cfg.grad_clip_norm,
                        false,
                        collect_metrics,
                    )?;
                    *table = track(apply_sparse_column_adamw_tensor(
                        table.clone().inner(),
                        gradient,
                        moment,
                        velocity,
                        table_cfg,
                        table_scale,
                        SparseIdentityAdamW {
                            identity_steps: &mut state.amortization_identity_steps,
                            active_identities,
                            upstream_growing_min_lr_scale,
                        },
                    )?);
                    state.amortization_step = state.amortization_step.saturating_add(1);
                    table_norm
                } else if state.amortization_residual_m.is_some()
                    || state.amortization_residual_v.is_some()
                {
                    return Err(AutomataError::InvalidArgument(
                        "amortization optimizer state exists without a residual table".to_string(),
                    ));
                } else {
                    0.0
                };
                state.step = state
                    .step
                    .max(state.row_flow_step)
                    .max(state.amortization_step);
                return Ok((norm, scale, amortization_grad_norm));
            }
            if tensors.len() != 11 {
                return Err(AutomataError::InvalidArgument(format!(
                    "HyperNPA generator expected 11 gradient tensors, got {}",
                    tensors.len()
                )));
            }
            let normalize_sample_table = normalize && self.kind == E2eHyperGeneratorKind::SampleIdTable;
            let original_table_norm = (normalize_sample_table && collect_metrics)
                .then(|| group_norm_tensor(&tensors));
            if normalize_sample_table {
                tensors[0] = normalize_sample_id_table_gradient(
                    tensors[0].clone(),
                    &self.adapter_parameter_segments,
                );
            }
            let (prepared_norm, scale, scale_tensor) = prepare_grad_group(
                &mut tensors,
                cfg.grad_clip_norm,
                normalize && !normalize_sample_table,
                collect_metrics,
            )?;
            let norm = if let Some(original) = original_table_norm {
                finite_scalar(
                    "Burn sample-ID adapter table grad norm",
                    original.into_scalar(),
                )?
            } else {
                prepared_norm
            };
            let bias = state.next_bias_correction(cfg);
            self.token_w = track(if self.kind == E2eHyperGeneratorKind::SampleIdTable {
                apply_sparse_column_adamw_tensor(
                    self.token_w.clone().inner(),
                    tensors.remove(0),
                    &mut state.token_w_m,
                    &mut state.token_w_v,
                    cfg,
                    scale_tensor.clone(),
                    SparseIdentityAdamW {
                        identity_steps: &mut state.sample_id_steps,
                        active_identities,
                        upstream_growing_min_lr_scale,
                    },
                )?
            } else {
                apply_adamw_tensor(
                    self.token_w.clone().inner(),
                    tensors.remove(0),
                    &mut state.token_w_m,
                    &mut state.token_w_v,
                    cfg,
                    scale_tensor.clone(),
                    bias,
                )
            });
            self.token_b = track(apply_adamw_tensor(
                self.token_b.clone().inner(),
                tensors.remove(0),
                &mut state.token_b_m,
                &mut state.token_b_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.token_gate_w = track(apply_adamw_tensor(
                self.token_gate_w.clone().inner(),
                tensors.remove(0),
                &mut state.token_gate_w_m,
                &mut state.token_gate_w_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.token_gate_b = track(apply_adamw_tensor(
                self.token_gate_b.clone().inner(),
                tensors.remove(0),
                &mut state.token_gate_b_m,
                &mut state.token_gate_b_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.state_w = track(apply_adamw_tensor(
                self.state_w.clone().inner(),
                tensors.remove(0),
                &mut state.state_w_m,
                &mut state.state_w_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.time_w = track(apply_adamw_tensor(
                self.time_w.clone().inner(),
                tensors.remove(0),
                &mut state.time_w_m,
                &mut state.time_w_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.output_w = track(apply_adamw_tensor(
                self.output_w.clone().inner(),
                tensors.remove(0),
                &mut state.output_w_m,
                &mut state.output_w_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.output_b = track(apply_adamw_tensor(
                self.output_b.clone().inner(),
                tensors.remove(0),
                &mut state.output_b_m,
                &mut state.output_b_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.condition_control_w = track(apply_adamw_tensor(
                self.condition_control_w.clone().inner(),
                tensors.remove(0),
                &mut state.condition_control_w_m,
                &mut state.condition_control_w_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.condition_control_b = track(apply_adamw_tensor(
                self.condition_control_b.clone().inner(),
                tensors.remove(0),
                &mut state.condition_control_b_m,
                &mut state.condition_control_b_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.condition_control_state_w = track(apply_adamw_tensor(
                self.condition_control_state_w.clone().inner(),
                tensors.remove(0),
                &mut state.condition_control_state_w_m,
                &mut state.condition_control_state_w_v,
                cfg,
                scale_tensor,
                bias,
            ));
            Ok((norm, scale, 0.0))
        }

        pub(super) fn to_hyper(&self, config: BurnE2eRolloutTrainConfig) -> AutomataResult<E2eHyperNpa2d> {
            if let Some(flow) = &self.row_flow {
                return Ok(E2eHyperNpa2d {
                    version: if config.dino_patch_pixels { 3 } else { 2 },
                    architecture: E2E_HYPER_ARCH_CONDITIONAL_ROW_FLOW.to_string(),
                    backend: Some(format!("{BACKEND}_e2e_rollout")),
                    condition_encoder: None,
                    condition_token_count: Some(flow.config.condition_tokens),
                    condition_embed_dims: Some(flow.config.condition_dims),
                    condition_token_grid_width: None,
                    condition_token_grid_height: None,
                    condition_image_size: Some(config.dino_image_size),
                    condition_alpha_mode: Some("composite-white".to_string()),
                    condition_rgb_channels: Some(config.dino_rgb_channels),
                    condition_rgb_channel_scale: Some(config.dino_rgb_channel_scale),
                    condition_alpha_channel: Some(config.dino_alpha_channel),
                    condition_alpha_channel_scale: Some(config.dino_alpha_channel_scale),
                    condition_patch_pixels: Some(config.dino_patch_pixels),
                    condition_l2_normalize_features: Some(config.dino_l2_normalize_features),
                    condition_resize_mode: Some("stretch".to_string()),
                    condition_application: Some("static-adapter".to_string()),
                    shared_base_sha256: None,
                    hidden_dims: flow.config.width,
                    token_attention_heads: flow.config.heads,
                    attention_normalization: Some(E2E_HYPER_ATTENTION_SOFTMAX.to_string()),
                    output_dims: self.output_dims,
                    sample_steps: flow.config.sample_steps,
                    output_scale: flow.config.source_scale,
                    adapter_rank: Some(config.adapter_rank),
                    adapter_alpha: Some(config.adapter_alpha),
                    adapter_parameterization: Some(
                        E2E_HYPER_ADAPTER_DENSE_ROW_RESIDUAL.to_string(),
                    ),
                    adapter_output_bias: Some(config.adapter_output_bias),
                    adapter_chunk_size: None,
                    spatial_condition_control: None,
                    spatial_condition_control_scale: None,
                    spatial_condition_control_sigma: None,
                    spatial_condition_state_control: None,
                    row_flow: Some(flow.config.clone()),
                    weights: E2eHyperNpa2dWeights {
                        token_w: Vec::new(),
                        token_b: Vec::new(),
                        token_gate_w: Vec::new(),
                        token_gate_b: Vec::new(),
                        state_w: Vec::new(),
                        time_w: Vec::new(),
                        output_w: Vec::new(),
                        output_b: Vec::new(),
                        condition_control_w: Vec::new(),
                        condition_control_b: Vec::new(),
                        condition_control_state_w: Vec::new(),
                        row_flow: flow.weight_values()?,
                    },
                });
            }
            let image_conditioned = self.kind != E2eHyperGeneratorKind::SampleIdTable;
            Ok(E2eHyperNpa2d {
                version: 1,
                architecture: self.kind.artifact_architecture().to_string(),
                backend: Some(format!("{BACKEND}_e2e_rollout")),
                condition_encoder: None,
                condition_token_count: None,
                condition_embed_dims: None,
                condition_token_grid_width: None,
                condition_token_grid_height: None,
                condition_image_size: image_conditioned.then_some(config.dino_image_size),
                condition_alpha_mode: image_conditioned.then(|| "composite-white".to_string()),
                condition_rgb_channels: image_conditioned.then_some(config.dino_rgb_channels),
                condition_rgb_channel_scale: image_conditioned
                    .then_some(config.dino_rgb_channel_scale),
                condition_alpha_channel: image_conditioned.then_some(config.dino_alpha_channel),
                condition_alpha_channel_scale: image_conditioned
                    .then_some(config.dino_alpha_channel_scale),
                condition_patch_pixels: image_conditioned.then_some(config.dino_patch_pixels),
                condition_l2_normalize_features: image_conditioned
                    .then_some(config.dino_l2_normalize_features),
                condition_resize_mode: image_conditioned.then(|| "stretch".to_string()),
                condition_application: Some(if config.spatial_condition_control {
                    "per-step-field"
                } else {
                    "static-adapter"
                }.to_string()),
                shared_base_sha256: None,
                hidden_dims: self.hidden_dims,
                token_attention_heads: self.token_attention_heads,
                attention_normalization: image_conditioned.then(|| {
                    if self.softmax_token_attention {
                        crate::hyper::e2e::E2E_HYPER_ATTENTION_SOFTMAX
                    } else {
                        crate::hyper::e2e::E2E_HYPER_ATTENTION_TANH_EXP
                    }
                    .to_string()
                }),
                output_dims: self.output_dims,
                sample_steps: self.sample_steps,
                output_scale: self.output_scale,
                adapter_rank: Some(config.adapter_rank),
                adapter_alpha: Some(config.adapter_alpha),
                adapter_parameterization: Some(
                    if self.canonical_full_rank_lora {
                        E2E_HYPER_ADAPTER_CANONICAL_FULL_RANK
                    } else {
                        E2E_HYPER_ADAPTER_FACTORIZED
                    }
                    .to_string(),
                ),
                adapter_output_bias: Some(config.adapter_output_bias),
                adapter_chunk_size: matches!(
                    self.kind,
                    E2eHyperGeneratorKind::SpatialTokenFlow
                        | E2eHyperGeneratorKind::ModuleTokenDecoderV2
                        | E2eHyperGeneratorKind::ModuleTokenDecoder
                )
                .then_some(self.adapter_chunk_size),
                spatial_condition_control: config.spatial_condition_control.then_some(true),
                spatial_condition_control_scale: config
                    .spatial_condition_control
                    .then_some(config.spatial_condition_control_scale),
                spatial_condition_control_sigma: config
                    .spatial_condition_control
                    .then_some(config.spatial_condition_control_sigma),
                spatial_condition_state_control: config
                    .spatial_condition_state_control
                    .then_some(true),
                row_flow: None,
                weights: E2eHyperNpa2dWeights {
                    token_w: tensor_vec(self.token_w.clone().inner())?,
                    token_b: tensor_vec(self.token_b.clone().inner())?,
                    token_gate_w: tensor_vec(self.token_gate_w.clone().inner())?,
                    token_gate_b: tensor_vec(self.token_gate_b.clone().inner())?,
                    state_w: tensor_vec(self.state_w.clone().inner())?,
                    time_w: tensor_vec(self.time_w.clone().inner())?,
                    output_w: tensor_vec(self.output_w.clone().inner())?,
                    output_b: tensor_vec(self.output_b.clone().inner())?,
                    condition_control_w: if config.spatial_condition_control {
                        tensor_vec(self.condition_control_w.clone().inner())?
                    } else {
                        Vec::new()
                    },
                    condition_control_b: if config.spatial_condition_control {
                        tensor_vec(self.condition_control_b.clone().inner())?
                    } else {
                        Vec::new()
                    },
                    condition_control_state_w: if config.spatial_condition_state_control {
                        tensor_vec(self.condition_control_state_w.clone().inner())?
                    } else {
                        Vec::new()
                    },
                    row_flow: Vec::new(),
                },
            })
        }

        pub(super) fn detached(&self) -> Self {
            Self {
                kind: self.kind,
                token_w: detach2(self.token_w.clone()),
                token_b: detach2(self.token_b.clone()),
                token_gate_w: detach2(self.token_gate_w.clone()),
                token_gate_b: detach2(self.token_gate_b.clone()),
                state_w: detach2(self.state_w.clone()),
                time_w: detach2(self.time_w.clone()),
                output_w: detach2(self.output_w.clone()),
                output_b: detach2(self.output_b.clone()),
                condition_control_w: detach2(self.condition_control_w.clone()),
                condition_control_b: detach2(self.condition_control_b.clone()),
                condition_control_state_w: detach2(self.condition_control_state_w.clone()),
                hidden_dims: self.hidden_dims,
                token_attention_heads: self.token_attention_heads,
                softmax_token_attention: self.softmax_token_attention,
                canonical_full_rank_lora: self.canonical_full_rank_lora,
                adapter_constants: detach2(self.adapter_constants.clone()),
                adapter_trainable_mask: detach2(self.adapter_trainable_mask.clone()),
                adapter_parameter_segments: self.adapter_parameter_segments.clone(),
                output_dims: self.output_dims,
                output_scale: self.output_scale,
                sample_steps: self.sample_steps,
                adapter_chunk_size: self.adapter_chunk_size,
                output_chunks: self.output_chunks,
                row_flow: self.row_flow.as_ref().map(BurnRowFlowParams::detached),
                amortization_residual_table: self
                    .amortization_residual_table
                    .as_ref()
                    .map(|table| detach2(table.clone())),
                amortization_gradient_layout: self.amortization_gradient_layout,
                amortization_learning_rate_scale: self.amortization_learning_rate_scale,
                amortization_grad_normalization: self.amortization_grad_normalization,
            }
        }
    }

    pub(super) fn next_adamw_bias_correction(step: &mut usize, cfg: AdamWConfig) -> AdamWBiasCorrection {
        *step = step.saturating_add(1);
        let step_i32 = (*step).min(i32::MAX as usize) as i32;
        AdamWBiasCorrection {
            beta1: 1.0 - cfg.beta1.powi(step_i32),
            beta2: 1.0 - cfg.beta2.powi(step_i32),
        }
    }

    impl BurnBaseParams {
        pub(super) fn from_model(model: &NpaModel, device: &BurnDevice) -> AutomataResult<Self> {
            let config = &model.config;
            Ok(Self {
                w1: tracked_tensor(
                    model.weights.w1.clone(),
                    [config.hidden_dims, config.perception_dims()],
                    device,
                ),
                b1: tracked_tensor(model.weights.b1.clone(), [1, config.hidden_dims], device),
                w2: tracked_tensor(
                    model.weights.w2.clone(),
                    [config.update_dims(), config.hidden_dims],
                    device,
                ),
                b2: tracked_tensor(
                    vec![0.0; config.update_dims()],
                    [1, config.update_dims()],
                    device,
                ),
            })
        }

        pub(super) fn forward_adapter(
            &self,
            features: Tensor2,
            adapter: &BurnAdapterParams,
            _config: DirectBasisTrainConfig,
        ) -> Tensor2 {
            let rows = features.shape().dims::<2>()[0];
            let scale = adapter.alpha / adapter.rank.max(1) as f32;
            let w1 = self.w1.clone()
                + adapter
                    .w1_up
                    .clone()
                    .matmul(adapter.w1_down.clone())
                    .mul_scalar(scale);
            let w2 = self.w2.clone()
                + adapter
                    .w2_up
                    .clone()
                    .matmul(adapter.w2_down.clone())
                    .mul_scalar(scale);
            let b1 = self.b1.clone() + adapter.b1_delta.clone();
            let hidden_dims = b1.shape().dims::<2>()[1];
            relu(features.matmul(w1.transpose()) + b1.expand([rows, hidden_dims]))
                .matmul(w2.transpose())
                + adapter
                    .b2_delta
                    .clone()
                    .expand([rows, self.w2.shape().dims::<2>()[0]])
        }

        pub(super) fn forward_adapter_batch(
            &self,
            features: Tensor3,
            adapter: &BurnAdapterBatch,
        ) -> Tensor3 {
            let dims = features.shape().dims::<3>();
            let batches = dims[0];
            let rows = dims[1];
            let input_dims = dims[2];
            let adapter_batches = adapter.w1_down.shape().dims::<3>()[0];
            assert_eq!(
                adapter_batches, batches,
                "adapter batch must cover every rollout row"
            );
            let scale = adapter.alpha / adapter.rank.max(1) as f32;
            let w1 = self.w1.clone().unsqueeze_dim::<3>(0).expand([
                adapter_batches,
                self.w1.shape().dims::<2>()[0],
                self.w1.shape().dims::<2>()[1],
            ]) + if adapter.canonical_dense_residual {
                adapter.w1_up.clone().narrow(2, 0, input_dims)
            } else {
                adapter
                    .w1_up
                    .clone()
                    .matmul(adapter.w1_down.clone())
                    .mul_scalar(scale)
            };
            let w2 = self.w2.clone().unsqueeze_dim::<3>(0).expand([
                adapter_batches,
                self.w2.shape().dims::<2>()[0],
                self.w2.shape().dims::<2>()[1],
            ]) + if adapter.canonical_dense_residual {
                adapter.w2_down.clone().narrow(
                    1,
                    0,
                    self.w2.shape().dims::<2>()[0],
                )
            } else {
                adapter
                    .w2_up
                    .clone()
                    .matmul(adapter.w2_down.clone())
                    .mul_scalar(scale)
            };
            let hidden_dims = self.b1.shape().dims::<2>()[1];
            let b1 = self
                .b1
                .clone()
                .unsqueeze_dim::<3>(0)
                .expand([adapter_batches, rows, hidden_dims])
                + adapter
                    .b1_delta
                    .clone()
                    .expand([adapter_batches, rows, hidden_dims]);
            let output_dims = self.w2.shape().dims::<2>()[0];
            relu(features.matmul(w1.swap_dims(1, 2)) + b1).matmul(w2.swap_dims(1, 2))
                + adapter
                    .b2_delta
                    .clone()
                    .expand([adapter_batches, rows, output_dims])
        }

        pub(super) fn apply_adamw(
            &mut self,
            grads: &mut <BurnBackend as burn::tensor::backend::AutodiffBackend>::Gradients,
            state: &mut BurnBaseAdamWState,
            cfg: AdamWConfig,
            normalize: bool,
            collect_metrics: bool,
        ) -> AutomataResult<(f32, f32)> {
            let tensors = self.take_gradients(grads);
            self.apply_adamw_gradients(tensors, state, cfg, normalize, collect_metrics)
        }

        pub(super) fn take_gradients(
            &self,
            grads: &mut <BurnBackend as burn::tensor::backend::AutodiffBackend>::Gradients,
        ) -> Vec<Tensor2Inner> {
            vec![
                self.w1
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.w1.clone().inner().zeros_like()),
                self.b1
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.b1.clone().inner().zeros_like()),
                self.w2
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.w2.clone().inner().zeros_like()),
                self.b2
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.b2.clone().inner().zeros_like()),
            ]
        }

        pub(super) fn apply_adamw_gradients(
            &mut self,
            mut tensors: Vec<Tensor2Inner>,
            state: &mut BurnBaseAdamWState,
            cfg: AdamWConfig,
            normalize: bool,
            collect_metrics: bool,
        ) -> AutomataResult<(f32, f32)> {
            if tensors.len() != 4 {
                return Err(AutomataError::InvalidArgument(format!(
                    "NPA base expected 4 gradient tensors, got {}",
                    tensors.len()
                )));
            }
            let (norm, scale, scale_tensor) =
                prepare_grad_group(&mut tensors, cfg.grad_clip_norm, normalize, collect_metrics)?;
            let bias = state.next_bias_correction(cfg);
            self.w1 = track(apply_adamw_tensor(
                self.w1.clone().inner(),
                tensors.remove(0),
                &mut state.w1_m,
                &mut state.w1_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.b1 = track(apply_adamw_tensor(
                self.b1.clone().inner(),
                tensors.remove(0),
                &mut state.b1_m,
                &mut state.b1_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.w2 = track(apply_adamw_tensor(
                self.w2.clone().inner(),
                tensors.remove(0),
                &mut state.w2_m,
                &mut state.w2_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            let _unused_output_bias_gradient = tensors.remove(0);
            self.b2 = track(self.b2.clone().inner().zeros_like());
            state.b2_m = state.b2_m.clone().zeros_like();
            state.b2_v = state.b2_v.clone().zeros_like();
            Ok((norm, scale))
        }

        pub(super) fn write_to_model(&self, model: &mut NpaModel) -> AutomataResult<()> {
            model.weights = NpaWeights {
                w1: tensor_vec(self.w1.clone().inner())?,
                b1: tensor_vec(self.b1.clone().inner())?,
                w2: tensor_vec(self.w2.clone().inner())?,
                b2: vec![0.0; model.config.update_dims()],
            };
            model.validate()
        }

        pub(super) fn detached(&self) -> Self {
            Self {
                w1: detach2(self.w1.clone()),
                b1: detach2(self.b1.clone()),
                w2: detach2(self.w2.clone()),
                b2: detach2(self.b2.clone()),
            }
        }
    }

    impl BurnBaseBatch {
        pub(super) fn from_models(models: &[NpaModel], device: &BurnDevice) -> AutomataResult<Self> {
            let first = models.first().ok_or_else(|| {
                AutomataError::InvalidArgument(
                    "Burn oracle model batch requires at least one model".to_string(),
                )
            })?;
            let model_count = models.len();
            let input_dims = first.config.perception_dims();
            let hidden_dims = first.config.hidden_dims;
            let output_dims = first.config.update_dims();
            Ok(Self {
                w1: tensor3(
                    models
                        .iter()
                        .flat_map(|model| model.weights.w1.iter().copied())
                        .collect(),
                    [model_count, hidden_dims, input_dims],
                    device,
                )
                .require_grad(),
                b1: tensor3(
                    models
                        .iter()
                        .flat_map(|model| model.weights.b1.iter().copied())
                        .collect(),
                    [model_count, 1, hidden_dims],
                    device,
                )
                .require_grad(),
                w2: tensor3(
                    models
                        .iter()
                        .flat_map(|model| model.weights.w2.iter().copied())
                        .collect(),
                    [model_count, output_dims, hidden_dims],
                    device,
                )
                .require_grad(),
                b2: tensor3(
                    vec![0.0; model_count * output_dims],
                    [model_count, 1, output_dims],
                    device,
                )
                .require_grad(),
            })
        }

        pub(super) fn model_count(&self) -> usize {
            self.w1.shape().dims::<3>()[0]
        }

        pub(super) fn model(&self, index: usize) -> BurnBaseParams {
            let [model_count, hidden_dims, input_dims] = self.w1.shape().dims::<3>();
            assert!(index < model_count, "oracle model index out of bounds");
            let output_dims = self.w2.shape().dims::<3>()[1];
            BurnBaseParams {
                w1: self
                    .w1
                    .clone()
                    .narrow(0, index, 1)
                    .reshape([hidden_dims, input_dims]),
                b1: self
                    .b1
                    .clone()
                    .narrow(0, index, 1)
                    .reshape([1, hidden_dims]),
                w2: self
                    .w2
                    .clone()
                    .narrow(0, index, 1)
                    .reshape([output_dims, hidden_dims]),
                b2: self
                    .b2
                    .clone()
                    .narrow(0, index, 1)
                    .reshape([1, output_dims]),
            }
        }

        pub(super) fn repeated(&self, repeats_per_model: usize) -> Self {
            let repeats_per_model = repeats_per_model.max(1);
            Self {
                w1: repeat_model_batch_tensor(self.w1.clone(), repeats_per_model),
                b1: repeat_model_batch_tensor(self.b1.clone(), repeats_per_model),
                w2: repeat_model_batch_tensor(self.w2.clone(), repeats_per_model),
                b2: repeat_model_batch_tensor(self.b2.clone(), repeats_per_model),
            }
        }

        pub(super) fn detached(&self) -> Self {
            Self {
                w1: detach3(self.w1.clone()),
                b1: detach3(self.b1.clone()),
                w2: detach3(self.w2.clone()),
                b2: detach3(self.b2.clone()),
            }
        }

        pub(super) fn forward(&self, features: Tensor3) -> Tensor3 {
            let [trajectory_count, particles, input_dims] = features.shape().dims::<3>();
            let model_count = self.model_count();
            assert!(
                trajectory_count.is_multiple_of(model_count),
                "oracle trajectory batch {trajectory_count} must be divisible by model count {model_count}"
            );
            let trajectories_per_model = trajectory_count / model_count;
            let rows_per_model = trajectories_per_model * particles;
            let hidden_dims = self.w1.shape().dims::<3>()[1];
            let output_dims = self.w2.shape().dims::<3>()[1];

            if model_count == 1 {
                let w1 = self
                    .w1
                    .clone()
                    .expand([trajectory_count, hidden_dims, input_dims]);
                let b1 = self
                    .b1
                    .clone()
                    .expand([trajectory_count, particles, hidden_dims]);
                let w2 = self
                    .w2
                    .clone()
                    .expand([trajectory_count, output_dims, hidden_dims]);
                return relu(features.matmul(w1.swap_dims(1, 2)) + b1)
                    .matmul(w2.swap_dims(1, 2));
            }

            // Keep each model's parameters once and expose its trajectories as the
            // row dimension of a strided batched GEMM. This maps directly to the
            // backend's optimized matrix-multiplication path and accumulates one
            // independent parameter gradient per model.
            let features = features.reshape([model_count, rows_per_model, input_dims]);
            let hidden = relu(
                features.matmul(self.w1.clone().swap_dims(1, 2))
                    + self
                        .b1
                        .clone()
                        .expand([model_count, rows_per_model, hidden_dims]),
            );
            hidden
                .matmul(self.w2.clone().swap_dims(1, 2))
                .reshape([trajectory_count, particles, output_dims])
        }

        pub(super) fn apply_adamw(
            &mut self,
            grads: &mut <BurnBackend as burn::tensor::backend::AutodiffBackend>::Gradients,
            state: &mut BurnBaseBatchAdamWState,
            cfg: AdamWConfig,
            normalize: bool,
            collect_metrics: bool,
        ) -> AutomataResult<(Vec<f32>, Vec<f32>)> {
            let tensors = self.take_gradients(grads);
            self.apply_adamw_gradients(tensors, state, cfg, normalize, collect_metrics)
        }

        pub(super) fn apply_adamw_last_input_column(
            &mut self,
            grads: &mut <BurnBackend as burn::tensor::backend::AutodiffBackend>::Gradients,
            state: &mut BurnBaseBatchAdamWState,
            cfg: AdamWConfig,
            normalize: bool,
            collect_metrics: bool,
        ) -> AutomataResult<(Vec<f32>, Vec<f32>)> {
            if cfg.weight_decay != 0.0 {
                return Err(AutomataError::InvalidArgument(
                    "last-input-column optimization requires zero weight decay".to_owned(),
                ));
            }
            let tensors = self.take_gradients(grads);
            self.apply_adamw_last_input_column_gradients(
                tensors,
                state,
                cfg,
                normalize,
                collect_metrics,
            )
        }

        pub(super) fn take_gradients(
            &self,
            grads: &mut <BurnBackend as burn::tensor::backend::AutodiffBackend>::Gradients,
        ) -> Vec<Tensor3Inner> {
            vec![
                self.w1
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.w1.clone().inner().zeros_like()),
                self.b1
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.b1.clone().inner().zeros_like()),
                self.w2
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.w2.clone().inner().zeros_like()),
                self.b2
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.b2.clone().inner().zeros_like()),
            ]
        }

        pub(super) fn apply_adamw_gradients(
            &mut self,
            tensors: Vec<Tensor3Inner>,
            state: &mut BurnBaseBatchAdamWState,
            cfg: AdamWConfig,
            normalize: bool,
            collect_metrics: bool,
        ) -> AutomataResult<(Vec<f32>, Vec<f32>)> {
            self.apply_adamw_masked_gradients(
                tensors,
                state,
                cfg,
                normalize,
                collect_metrics,
                false,
            )
        }

        pub(super) fn apply_adamw_last_input_column_gradients(
            &mut self,
            tensors: Vec<Tensor3Inner>,
            state: &mut BurnBaseBatchAdamWState,
            cfg: AdamWConfig,
            normalize: bool,
            collect_metrics: bool,
        ) -> AutomataResult<(Vec<f32>, Vec<f32>)> {
            if cfg.weight_decay != 0.0 {
                return Err(AutomataError::InvalidArgument(
                    "last-input-column optimization requires zero weight decay".to_owned(),
                ));
            }
            self.apply_adamw_masked_gradients(
                tensors,
                state,
                cfg,
                normalize,
                collect_metrics,
                true,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn apply_adamw_masked_gradients(
            &mut self,
            mut tensors: Vec<Tensor3Inner>,
            state: &mut BurnBaseBatchAdamWState,
            cfg: AdamWConfig,
            normalize: bool,
            collect_metrics: bool,
            last_input_column_only: bool,
        ) -> AutomataResult<(Vec<f32>, Vec<f32>)> {
            if tensors.len() != 4 {
                return Err(AutomataError::InvalidArgument(format!(
                    "NPA model batch expected 4 gradient tensors, got {}",
                    tensors.len()
                )));
            }
            if last_input_column_only {
                tensors[0] = retain_last_model_batch_input_column(tensors[0].clone());
                for tensor in &mut tensors[1..] {
                    *tensor = tensor.clone().zeros_like();
                }
                state.w1_m = retain_last_model_batch_input_column(state.w1_m.clone());
                state.w1_v = retain_last_model_batch_input_column(state.w1_v.clone());
                state.b1_m = state.b1_m.clone().zeros_like();
                state.b1_v = state.b1_v.clone().zeros_like();
                state.w2_m = state.w2_m.clone().zeros_like();
                state.w2_v = state.w2_v.clone().zeros_like();
                state.b2_m = state.b2_m.clone().zeros_like();
                state.b2_v = state.b2_v.clone().zeros_like();
            }
            let (norms, scales, scale_tensor) = prepare_model_batch_grad_group(
                &mut tensors,
                cfg.grad_clip_norm,
                normalize,
                collect_metrics,
            )?;
            let bias = state.next_bias_correction(cfg);
            self.w1 = track3(apply_adamw_tensor3(
                self.w1.clone().inner(),
                tensors.remove(0),
                &mut state.w1_m,
                &mut state.w1_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.b1 = track3(apply_adamw_tensor3(
                self.b1.clone().inner(),
                tensors.remove(0),
                &mut state.b1_m,
                &mut state.b1_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.w2 = track3(apply_adamw_tensor3(
                self.w2.clone().inner(),
                tensors.remove(0),
                &mut state.w2_m,
                &mut state.w2_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            let _unused_output_bias_gradient = tensors.remove(0);
            self.b2 = track3(self.b2.clone().inner().zeros_like());
            state.b2_m = state.b2_m.clone().zeros_like();
            state.b2_v = state.b2_v.clone().zeros_like();
            Ok((norms, scales))
        }

        pub(super) fn write_to_models(&self, models: &mut [NpaModel]) -> AutomataResult<()> {
            if models.len() != self.model_count() {
                return Err(AutomataError::InvalidArgument(format!(
                    "Burn oracle model writeback mismatch: models={} batch={}",
                    models.len(),
                    self.model_count(),
                )));
            }
            let w1 = tensor3_vec(self.w1.clone().inner())?;
            let b1 = tensor3_vec(self.b1.clone().inner())?;
            let w2 = tensor3_vec(self.w2.clone().inner())?;
            let w1_len = models[0].weights.w1.len();
            let b1_len = models[0].weights.b1.len();
            let w2_len = models[0].weights.w2.len();
            for (index, model) in models.iter_mut().enumerate() {
                model.weights.w1 = w1[index * w1_len..(index + 1) * w1_len].to_vec();
                model.weights.b1 = b1[index * b1_len..(index + 1) * b1_len].to_vec();
                model.weights.w2 = w2[index * w2_len..(index + 1) * w2_len].to_vec();
                model.weights.b2.fill(0.0);
                model.validate()?;
            }
            Ok(())
        }
    }

    fn retain_last_model_batch_input_column(tensor: Tensor3Inner) -> Tensor3Inner {
        let dims = tensor.shape().dims::<3>();
        assert!(dims[2] > 0, "model input tensor must have at least one column");
        if dims[2] == 1 {
            return tensor;
        }
        Tensor::cat(
            vec![
                tensor.clone().narrow(2, 0, dims[2] - 1).zeros_like(),
                tensor.narrow(2, dims[2] - 1, 1),
            ],
            2,
        )
    }

    pub(super) fn repeat_model_batch_tensor(tensor: Tensor3, repeats_per_model: usize) -> Tensor3 {
        if repeats_per_model <= 1 {
            return tensor;
        }
        let dims = tensor.shape().dims::<3>();
        Tensor::cat(
            (0..dims[0])
                .flat_map(|model| {
                    let model_tensor = tensor.clone().narrow(0, model, 1);
                    (0..repeats_per_model).map(move |_| model_tensor.clone())
                })
                .collect::<Vec<_>>(),
            0,
        )
    }

    impl BurnAdapterParams {
        pub(super) fn from_adapter(
            adapter: &NpaLowRankAdapter,
            model: &NpaModel,
            device: &BurnDevice,
        ) -> AutomataResult<Self> {
            let config = &model.config;
            Ok(Self {
                rank: adapter.rank,
                alpha: adapter.alpha,
                w1_down: tracked_tensor(
                    adapter.w1_down.clone(),
                    [adapter.rank, config.perception_dims()],
                    device,
                ),
                w1_up: tracked_tensor(
                    adapter.w1_up.clone(),
                    [config.hidden_dims, adapter.rank],
                    device,
                ),
                w2_down: tracked_tensor(
                    adapter.w2_down.clone(),
                    [adapter.rank, config.hidden_dims],
                    device,
                ),
                w2_up: tracked_tensor(
                    adapter.w2_up.clone(),
                    [config.update_dims(), adapter.rank],
                    device,
                ),
                b1_delta: tracked_tensor(adapter.b1_delta.clone(), [1, config.hidden_dims], device),
                b2_delta: tracked_tensor(
                    vec![0.0; config.update_dims()],
                    [1, config.update_dims()],
                    device,
                ),
            })
        }

        pub(super) fn to_adapter(&self) -> AutomataResult<NpaLowRankAdapter> {
            Ok(NpaLowRankAdapter {
                rank: self.rank,
                alpha: self.alpha,
                w1_down: tensor_vec(self.w1_down.clone().inner())?,
                w1_up: tensor_vec(self.w1_up.clone().inner())?,
                w2_down: tensor_vec(self.w2_down.clone().inner())?,
                w2_up: tensor_vec(self.w2_up.clone().inner())?,
                b1_delta: tensor_vec(self.b1_delta.clone().inner())?,
                b2_delta: vec![0.0; self.b2_delta.shape().dims::<2>()[1]],
                b1_delta_correction: Vec::new(),
                b2_delta_correction: Vec::new(),
            })
        }

        pub(super) fn l2_loss(&self) -> Tensor1 {
            let terms = vec![
                self.w1_down.clone(),
                self.w1_up.clone(),
                self.w2_down.clone(),
                self.w2_up.clone(),
                self.b1_delta.clone(),
            ];
            let mut total = None::<Tensor1>;
            for tensor in terms {
                let value = tensor.clone().mul(tensor).mean();
                total = Some(match total {
                    Some(total) => total + value,
                    None => value,
                });
            }
            total.expect("adapter has parameters").div_scalar(5.0)
        }

        pub(super) fn apply_adamw(
            &mut self,
            grads: &mut <BurnBackend as burn::tensor::backend::AutodiffBackend>::Gradients,
            state: &mut BurnAdapterAdamWState,
            cfg: AdamWConfig,
            normalize: bool,
            gradient_scale: f32,
            collect_metrics: bool,
        ) -> AutomataResult<(f32, f32)> {
            let mut tensors = vec![
                self.w1_down
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.w1_down.clone().inner().zeros_like()),
                self.w1_up
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.w1_up.clone().inner().zeros_like()),
                self.w2_down
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.w2_down.clone().inner().zeros_like()),
                self.w2_up
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.w2_up.clone().inner().zeros_like()),
                self.b1_delta
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.b1_delta.clone().inner().zeros_like()),
                self.b2_delta
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.b2_delta.clone().inner().zeros_like()),
            ];
            if gradient_scale != 1.0 {
                for tensor in &mut tensors {
                    *tensor = tensor.clone().mul_scalar(gradient_scale);
                }
            }
            let (norm, scale, scale_tensor) =
                prepare_grad_group(&mut tensors, cfg.grad_clip_norm, normalize, collect_metrics)?;
            let bias = state.next_bias_correction(cfg);
            self.w1_down = track(apply_adamw_tensor(
                self.w1_down.clone().inner(),
                tensors.remove(0),
                &mut state.w1_down_m,
                &mut state.w1_down_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.w1_up = track(apply_adamw_tensor(
                self.w1_up.clone().inner(),
                tensors.remove(0),
                &mut state.w1_up_m,
                &mut state.w1_up_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.w2_down = track(apply_adamw_tensor(
                self.w2_down.clone().inner(),
                tensors.remove(0),
                &mut state.w2_down_m,
                &mut state.w2_down_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.w2_up = track(apply_adamw_tensor(
                self.w2_up.clone().inner(),
                tensors.remove(0),
                &mut state.w2_up_m,
                &mut state.w2_up_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.b1_delta = track(apply_adamw_tensor(
                self.b1_delta.clone().inner(),
                tensors.remove(0),
                &mut state.b1_delta_m,
                &mut state.b1_delta_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            let _unused_output_bias_gradient = tensors.remove(0);
            self.b2_delta = track(self.b2_delta.clone().inner().zeros_like());
            state.b2_delta_m = state.b2_delta_m.clone().zeros_like();
            state.b2_delta_v = state.b2_delta_v.clone().zeros_like();
            Ok((norm, scale))
        }

        pub(super) fn detached(&self) -> Self {
            Self {
                rank: self.rank,
                alpha: self.alpha,
                w1_down: detach2(self.w1_down.clone()),
                w1_up: detach2(self.w1_up.clone()),
                w2_down: detach2(self.w2_down.clone()),
                w2_up: detach2(self.w2_up.clone()),
                b1_delta: detach2(self.b1_delta.clone()),
                b2_delta: detach2(self.b2_delta.clone()),
            }
        }
    }

    impl BurnAdapterBatch {
        pub(super) fn from_indices(adapters: &[BurnAdapterParams], indices: &[usize]) -> Self {
            let first = &adapters[indices[0]];
            Self {
                rank: first.rank,
                alpha: first.alpha,
                canonical_dense_residual: false,
                w1_down: stack_adapter_tensor(adapters, indices, |adapter| &adapter.w1_down),
                w1_up: stack_adapter_tensor(adapters, indices, |adapter| &adapter.w1_up),
                w2_down: stack_adapter_tensor(adapters, indices, |adapter| &adapter.w2_down),
                w2_up: stack_adapter_tensor(adapters, indices, |adapter| &adapter.w2_up),
                b1_delta: stack_adapter_tensor(adapters, indices, |adapter| &adapter.b1_delta),
                b2_delta: stack_adapter_tensor(adapters, indices, |adapter| &adapter.b2_delta),
            }
        }

        pub(super) fn select_rows(self, rows: &[usize]) -> Self {
            if rows.is_empty() {
                return self;
            }
            let device = self.w1_down.device();
            let indices = Tensor::<BurnBackend, 1, Int>::from_data(
                TensorData::new(
                    rows.iter().map(|row| *row as i64).collect::<Vec<_>>(),
                    [rows.len()],
                ),
                &device,
            );
            Self {
                rank: self.rank,
                alpha: self.alpha,
                canonical_dense_residual: self.canonical_dense_residual,
                w1_down: self.w1_down.select(0, indices.clone()),
                w1_up: self.w1_up.select(0, indices.clone()),
                w2_down: self.w2_down.select(0, indices.clone()),
                w2_up: self.w2_up.select(0, indices.clone()),
                b1_delta: self.b1_delta.select(0, indices.clone()),
                b2_delta: self.b2_delta.select(0, indices),
            }
        }

        pub(super) fn select_rows_or_identity(self, rows: Option<&[usize]>) -> Self {
            match rows {
                Some(rows) => self.select_rows(rows),
                None => self,
            }
        }

        pub(super) fn detached(&self) -> Self {
            Self {
                rank: self.rank,
                alpha: self.alpha,
                canonical_dense_residual: self.canonical_dense_residual,
                w1_down: detach3(self.w1_down.clone()),
                w1_up: detach3(self.w1_up.clone()),
                w2_down: detach3(self.w2_down.clone()),
                w2_up: detach3(self.w2_up.clone()),
                b1_delta: detach3(self.b1_delta.clone()),
                b2_delta: detach3(self.b2_delta.clone()),
            }
        }

        pub(super) fn from_parameter_vector(
            vector: Tensor2,
            config: &NpaConfig,
            rank: usize,
            alpha: f32,
        ) -> Self {
            let batches = vector.shape().dims::<2>()[0];
            let input_dims = config.perception_dims();
            let hidden_dims = config.hidden_dims;
            let output_dims = config.update_dims();
            let mut offset = 0usize;
            let mut take = |len: usize| {
                let out = vector.clone().narrow(1, offset, len);
                offset += len;
                out
            };
            Self {
                rank,
                alpha,
                canonical_dense_residual: false,
                w1_down: take(rank * input_dims).reshape([batches, rank, input_dims]),
                w1_up: take(hidden_dims * rank).reshape([batches, hidden_dims, rank]),
                w2_down: take(rank * hidden_dims).reshape([batches, rank, hidden_dims]),
                w2_up: take(output_dims * rank).reshape([batches, output_dims, rank]),
                b1_delta: take(hidden_dims).reshape([batches, 1, hidden_dims]),
                b2_delta: take(output_dims).reshape([batches, 1, output_dims]),
            }
        }

        pub(super) fn l2_loss(&self) -> Tensor1 {
            let terms = vec![
                self.w1_down.clone(),
                self.w1_up.clone(),
                self.w2_down.clone(),
                self.w2_up.clone(),
                self.b1_delta.clone(),
                self.b2_delta.clone(),
            ];
            let mut total = None::<Tensor1>;
            for tensor in terms {
                let value = tensor.clone().mul(tensor).mean();
                total = Some(match total {
                    Some(total) => total + value,
                    None => value,
                });
            }
            total.expect("adapter batch has parameters").div_scalar(6.0)
        }

        pub(super) fn to_parameter_vector(&self) -> Tensor2 {
            let batches = self.w1_down.shape().dims::<3>()[0];
            Tensor::cat(
                vec![
                    self.w1_down.clone().reshape([batches, self.w1_down.shape().num_elements() / batches]),
                    self.w1_up.clone().reshape([batches, self.w1_up.shape().num_elements() / batches]),
                    self.w2_down.clone().reshape([batches, self.w2_down.shape().num_elements() / batches]),
                    self.w2_up.clone().reshape([batches, self.w2_up.shape().num_elements() / batches]),
                    self.b1_delta.clone().reshape([batches, self.b1_delta.shape().num_elements() / batches]),
                    self.b2_delta.clone().reshape([batches, self.b2_delta.shape().num_elements() / batches]),
                ],
                1,
            )
        }

        pub(super) fn l2_loss_vector(&self) -> Tensor1 {
            let terms = vec![
                self.w1_down.clone(),
                self.w1_up.clone(),
                self.w2_down.clone(),
                self.w2_up.clone(),
                self.b1_delta.clone(),
                self.b2_delta.clone(),
            ];
            let mut total = None::<Tensor1>;
            for tensor in terms {
                let dims = tensor.shape().dims::<3>();
                let value = tensor
                    .clone()
                    .mul(tensor)
                    .reshape([dims[0], dims[1] * dims[2]])
                    .mean_dim(1)
                    .squeeze_dim::<1>(1);
                total = Some(match total {
                    Some(total) => total + value,
                    None => value,
                });
            }
            total
                .expect("adapter batch has parameters")
                .div_scalar(6.0)
        }
    }

    pub(super) fn stack_adapter_tensor(
        adapters: &[BurnAdapterParams],
        indices: &[usize],
        select: impl Fn(&BurnAdapterParams) -> &Tensor2,
    ) -> Tensor3 {
        Tensor::cat(
            indices
                .iter()
                .map(|idx| select(&adapters[*idx]).clone().unsqueeze_dim::<3>(0))
                .collect::<Vec<_>>(),
            0,
        )
    }

    pub(super) fn stack_base_tensor(
        params: &[BurnBaseParams],
        select: impl Fn(&BurnBaseParams) -> &Tensor2,
    ) -> Tensor3 {
        stack_base_tensor_repeated(params, 1, select)
    }

    pub(super) fn stack_base_tensor_repeated(
        params: &[BurnBaseParams],
        repeats_per_model: usize,
        select: impl Fn(&BurnBaseParams) -> &Tensor2,
    ) -> Tensor3 {
        let repeats_per_model = repeats_per_model.max(1);
        Tensor::cat(
            params
                .iter()
                .flat_map(|param| {
                    (0..repeats_per_model)
                        .map(|_| select(param).clone().unsqueeze_dim::<3>(0))
                })
                .collect::<Vec<_>>(),
            0,
        )
    }
