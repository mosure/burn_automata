//! Burn autodiff implementation of the structured conditional row flow.

use super::*;

impl BurnRowFlowParams {
    pub(super) fn seeded(
        npa: &NpaConfig,
        examples: &[BurnE2eRolloutExample],
        config: BurnE2eRolloutTrainConfig,
        condition_tokens: usize,
        condition_dims: usize,
        device: &BurnDevice,
    ) -> AutomataResult<Self> {
        let layout = NpaParameterRowLayout2d::new(npa);
        let row_rms = endpoint_row_rms(npa, examples, config, &layout)?;
        let flow = ConditionalRowFlowConfig {
            layers: config.generator_layers.max(1),
            width: config.generator_hidden_dims.max(1),
            heads: config.token_attention_heads.max(1),
            ffn_dims: config.generator_ffn_dims.max(config.generator_hidden_dims),
            condition_dims,
            condition_tokens,
            row_count: layout.row_count(),
            max_row_dims: layout.max_row_dims(),
            row_value_dims: layout.rows().iter().map(|row| row.value_dims).collect(),
            sample_steps: config.generator_sample_steps.max(1),
            source_seed: config.generator_source_seed,
            source_scale: config.generator_output_scale,
            solver: crate::hyper::row_flow::CONDITIONAL_ROW_FLOW_SOLVER_HEUN.to_string(),
            row_rms,
        };
        let weights = ConditionalRowFlowWeights::seeded_with_initialization(
            &flow,
            config.seed ^ 0x726f_7766_6c6f_7734,
            config.generator_cross_gate_init,
            config.generator_output_init_scale,
        )?;
        Self::from_values_with_output_bias(
            flow,
            &weights.values,
            npa,
            config.adapter_output_bias,
            device,
        )
    }

    pub(super) fn from_artifact(
        artifact: &E2eHyperNpa2d,
        npa: &NpaConfig,
        device: &BurnDevice,
    ) -> AutomataResult<Self> {
        if !artifact.is_conditional_row_flow() {
            return Err(AutomataError::InvalidModel(
                "expected a conditional row flow artifact".to_string(),
            ));
        }
        let flow = artifact
            .row_flow
            .clone()
            .ok_or_else(|| AutomataError::InvalidModel("missing row flow contract".to_string()))?;
        NpaParameterRowLayout2d::new(npa).validate_flow_config(&flow)?;
        ConditionalRowFlowWeights {
            values: artifact.weights.row_flow.clone(),
        }
        .validate(&flow)?;
        Self::from_values_with_output_bias(
            flow,
            &artifact.weights.row_flow,
            npa,
            artifact.adapter_output_bias_enabled(),
            device,
        )
    }

    pub(super) fn from_artifact_with_output_bias(
        artifact: &E2eHyperNpa2d,
        npa: &NpaConfig,
        output_bias: bool,
        device: &BurnDevice,
    ) -> AutomataResult<Self> {
        let flow = artifact
            .row_flow
            .clone()
            .ok_or_else(|| AutomataError::InvalidModel("missing row flow contract".to_string()))?;
        ConditionalRowFlowWeights {
            values: artifact.weights.row_flow.clone(),
        }
        .validate(&flow)?;
        Self::from_values_with_output_bias(
            flow,
            &artifact.weights.row_flow,
            npa,
            output_bias,
            device,
        )
    }

    pub(super) fn from_values(
        config: ConditionalRowFlowConfig,
        values: &[f32],
        npa: &NpaConfig,
        device: &BurnDevice,
    ) -> AutomataResult<Self> {
        Self::from_values_with_output_bias(config, values, npa, true, device)
    }

    pub(super) fn from_values_with_output_bias(
        config: ConditionalRowFlowConfig,
        values: &[f32],
        npa: &NpaConfig,
        output_bias: bool,
        device: &BurnDevice,
    ) -> AutomataResult<Self> {
        let layout = NpaParameterRowLayout2d::new(npa);
        layout.validate_flow_config(&config)?;
        let specs = config.tensor_specs()?;
        let mut offset = 0usize;
        let mut tensors = Vec::with_capacity(specs.len());
        for spec in specs {
            let len = spec.rows * spec.cols;
            tensors.push(tracked_tensor(
                values[offset..offset + len].to_vec(),
                [spec.rows, spec.cols],
                device,
            ));
            offset += len;
        }
        if offset != values.len() {
            return Err(AutomataError::InvalidModel(
                "row flow tensor layout did not consume all artifact weights".to_string(),
            ));
        }
        let row_scale = static_row_scale(&config, device);
        let row_mask = static_row_mask(&layout, output_bias, device);
        let source_rows = static_source_rows(&layout, &config, device).mul(row_mask.clone());
        let trainable_value_count = layout.trainable_parameter_count(output_bias);
        let time_frequencies = static_time_frequencies(config.width, device);
        Ok(Self {
            config,
            tensors,
            source_rows,
            row_scale,
            row_mask,
            trainable_value_count,
            time_frequencies,
        })
    }

    pub(super) fn detached(&self) -> Self {
        Self {
            config: self.config.clone(),
            tensors: self.tensors.iter().cloned().map(detach2).collect(),
            source_rows: detach3(self.source_rows.clone()),
            row_scale: detach3(self.row_scale.clone()),
            row_mask: detach3(self.row_mask.clone()),
            trainable_value_count: self.trainable_value_count,
            time_frequencies: detach2(self.time_frequencies.clone()),
        }
    }

    pub(super) fn recalibrate_from_endpoint_table(
        &mut self,
        endpoint_table: Tensor2,
        npa: &NpaConfig,
    ) -> AutomataResult<Vec<f32>> {
        let rows = self.config.row_count;
        let row_dims = self.config.max_row_dims;
        let [packed_rows, examples] = endpoint_table.shape().dims::<2>();
        if examples == 0 || packed_rows != rows.saturating_mul(row_dims) {
            return Err(AutomataError::InvalidArgument(format!(
                "row-flow endpoint scale calibration expected [{}, examples], got [{packed_rows}, {examples}]",
                rows.saturating_mul(row_dims),
            )));
        }
        let device = endpoint_table.device();
        let endpoints = detach2(endpoint_table).reshape([rows, row_dims, examples]);
        let squared = endpoints.clone().mul(endpoints);
        let sum_squares = squared
            .sum_dim(2)
            .sum_dim(1)
            .reshape([rows]);
        let denominators = self
            .config
            .row_value_dims
            .iter()
            .map(|dims| dims.saturating_mul(examples) as f32)
            .collect::<Vec<_>>();
        let row_rms = sum_squares
            .div(Tensor::<BurnBackend, 1>::from_data(
                TensorData::new(denominators, [rows]),
                &device,
            ))
            .sqrt()
            .clamp_min(1.0e-3);
        let values = tensor1_vec(row_rms.inner())?;
        if values.len() != rows
            || values
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(AutomataError::InvalidArgument(
                "row-flow endpoint scale calibration produced invalid row RMS values".to_string(),
            ));
        }
        let layout = NpaParameterRowLayout2d::new(npa);
        layout.validate_flow_config(&self.config)?;
        self.config.row_rms.clone_from(&values);
        self.row_scale = static_row_scale(&self.config, &device);
        self.source_rows =
            static_source_rows(&layout, &self.config, &device).mul(self.row_mask.clone());
        Ok(values)
    }

    pub(super) fn weight_values(&self) -> AutomataResult<Vec<f32>> {
        let mut values = Vec::with_capacity(self.config.parameter_count()?);
        for tensor in &self.tensors {
            values.extend(tensor_vec(tensor.clone().inner())?);
        }
        Ok(values)
    }

    pub(super) fn sample_adapter_batch(
        &self,
        condition: Tensor3,
        npa: &NpaConfig,
    ) -> BurnAdapterBatch {
        BurnAdapterBatch::from_dense_residual_rows(self.sample_rows(condition, npa), npa)
    }

    pub(super) fn sample_rows(&self, condition: Tensor3, npa: &NpaConfig) -> Tensor3 {
        self.sample_rows_with_prepared(condition, npa).0
    }

    pub(super) fn sample_rows_with_prepared(
        &self,
        condition: Tensor3,
        npa: &NpaConfig,
    ) -> (Tensor3, BurnRowFlowCondition) {
        self.sample_rows_with_prepared_steps(condition, npa, self.config.sample_steps)
    }

    pub(super) fn sample_rows_with_prepared_steps(
        &self,
        condition: Tensor3,
        npa: &NpaConfig,
        sample_steps: usize,
    ) -> (Tensor3, BurnRowFlowCondition) {
        let batches = condition.shape().dims::<3>()[0];
        let device = condition.device();
        let layout = NpaParameterRowLayout2d::new(npa);
        debug_assert_eq!(layout.row_count(), self.config.row_count);
        debug_assert_eq!(layout.max_row_dims(), self.config.max_row_dims);
        let condition = self.prepare_condition(condition);
        let rows = self.sample_rows_prepared_steps(&condition, batches, &device, sample_steps);
        (rows, condition)
    }

    fn source_rows(&self, batches: usize) -> Tensor3 {
        self.source_rows.clone().expand([
            batches,
            self.config.row_count,
            self.config.max_row_dims,
        ])
    }

    pub(super) fn sample_rows_prepared(
        &self,
        condition: &BurnRowFlowCondition,
        batches: usize,
        device: &BurnDevice,
    ) -> Tensor3 {
        self.sample_rows_prepared_steps(condition, batches, device, self.config.sample_steps)
    }

    pub(super) fn sample_rows_prepared_steps(
        &self,
        condition: &BurnRowFlowCondition,
        batches: usize,
        device: &BurnDevice,
        sample_steps: usize,
    ) -> Tensor3 {
        let mut state = self.source_rows(batches);
        let sample_steps = sample_steps.max(1);
        let dt = 1.0 / sample_steps as f32;
        for step in 0..sample_steps {
            let t0 = Tensor::<BurnBackend, 2>::full(
                [batches, 1],
                step as f32 * dt,
                device,
            );
            let v0 = self.velocity(condition, state.clone(), t0);
            let predictor = state.clone() + v0.clone().mul_scalar(dt);
            let t1 = Tensor::<BurnBackend, 2>::full(
                [batches, 1],
                (step + 1) as f32 * dt,
                device,
            );
            let v1 = self.velocity(condition, predictor, t1);
            state = state + (v0 + v1).mul_scalar(0.5 * dt);
        }
        state
    }

    pub(super) fn self_rectification_loss_to_endpoint(
        &self,
        condition: Tensor3,
        endpoint: Tensor3,
        npa: &NpaConfig,
        seed: u64,
    ) -> Tensor1 {
        let prepared = self.prepare_condition(condition);
        self.self_rectification_loss_to_endpoint_prepared(&prepared, endpoint, npa, seed)
    }

    pub(super) fn self_rectification_loss_to_endpoint_prepared(
        &self,
        condition: &BurnRowFlowCondition,
        endpoint: Tensor3,
        _npa: &NpaConfig,
        seed: u64,
    ) -> Tensor1 {
        let batches = endpoint.shape().dims::<3>()[0];
        let device = endpoint.device();
        let source = self.source_rows(batches);
        let endpoint_dims = endpoint.shape().dims::<3>();
        debug_assert_eq!(
            endpoint_dims,
            [batches, self.config.row_count, self.config.max_row_dims]
        );
        let endpoint = detach3(endpoint);
        let mut rng = StdRng::seed_from_u64(seed);
        let times = (0..batches)
            .map(|_| rng.random_range(1.0e-4_f32..1.0 - 1.0e-4_f32))
            .collect::<Vec<_>>();
        let time = Tensor::<BurnBackend, 2>::from_data(
            TensorData::new(times, [batches, 1]),
            &device,
        );
        let time3 = time.clone().unsqueeze_dim::<3>(1).expand([
            batches,
            self.config.row_count,
            self.config.max_row_dims,
        ]);
        let state = source.clone().mul(time3.clone().neg().add_scalar(1.0))
            + endpoint.clone().mul(time3);
        let expected_velocity = endpoint - source;
        let predicted_velocity = self.velocity(condition, state, time);
        let scale = self.row_scale.clone().expand(endpoint_dims);
        let mask = self.row_mask.clone().expand(endpoint_dims);
        let residual = (predicted_velocity - expected_velocity)
            .div(scale)
            .mul(mask.clone());
        residual
            .clone()
            .mul(residual)
            .sum()
            .div_scalar((batches * self.trainable_value_count) as f32)
    }

    pub(super) fn flow_matching_loss(
        &self,
        condition: Tensor3,
        teacher: Tensor2,
        npa: &NpaConfig,
        adapter_rank: usize,
        adapter_alpha: f32,
        match_inference_source: bool,
    ) -> Tensor1 {
        let prepared = self.prepare_condition(condition);
        self.flow_matching_loss_prepared(
            &prepared,
            teacher,
            npa,
            adapter_rank,
            adapter_alpha,
            match_inference_source,
        )
    }

    pub(super) fn flow_matching_loss_prepared(
        &self,
        condition: &BurnRowFlowCondition,
        teacher: Tensor2,
        npa: &NpaConfig,
        adapter_rank: usize,
        adapter_alpha: f32,
        match_inference_source: bool,
    ) -> Tensor1 {
        let target = BurnAdapterBatch::from_parameter_vector(
            teacher,
            npa,
            adapter_rank,
            adapter_alpha,
        )
        .dense_residual_rows(npa);
        let batches = target.shape().dims::<3>()[0];
        let device = target.device();
        let row_dims = [batches, self.config.row_count, self.config.max_row_dims];
        let scale = self.row_scale.clone().expand(row_dims);
        let mask = self.row_mask.clone().expand(row_dims);
        let source = if match_inference_source {
            self.source_rows(batches)
        } else {
            Tensor::<BurnBackend, 3>::random(
                [batches, self.config.row_count, self.config.max_row_dims],
                Distribution::Normal(0.0, 1.0),
                &device,
            )
            .mul(scale.clone())
            .mul_scalar(self.config.source_scale)
            .mul(mask.clone())
        };
        let time = Tensor::<BurnBackend, 2>::random(
            [batches, 1],
            Distribution::Uniform(1.0e-4, 1.0 - 1.0e-4),
            &device,
        );
        let time3 = time
            .clone()
            .unsqueeze_dim::<3>(1)
            .expand([
                batches,
                self.config.row_count,
                self.config.max_row_dims,
            ]);
        let state = source.clone().mul(time3.clone().neg().add_scalar(1.0))
            + target.clone().mul(time3);
        let expected_velocity = target - source;
        let predicted_velocity = self.velocity(condition, state, time);
        let residual = (predicted_velocity - expected_velocity).div(scale).mul(mask.clone());
        residual
            .clone()
            .mul(residual)
            .sum()
            .div_scalar((batches * self.trainable_value_count) as f32)
    }

    pub(super) fn endpoint_reconstruction_loss(
        &self,
        sampled_rows: Tensor3,
        teacher: Tensor2,
        npa: &NpaConfig,
        adapter_rank: usize,
        adapter_alpha: f32,
    ) -> Tensor1 {
        let target = BurnAdapterBatch::from_parameter_vector(
            teacher,
            npa,
            adapter_rank,
            adapter_alpha,
        )
        .dense_residual_rows(npa);
        let batches = target.shape().dims::<3>()[0];
        let row_dims = [batches, self.config.row_count, self.config.max_row_dims];
        let scale = self.row_scale.clone().expand(row_dims);
        let mask = self.row_mask.clone().expand(row_dims);
        let residual = (sampled_rows - target).div(scale).mul(mask);
        residual
            .clone()
            .mul(residual)
            .sum()
            .div_scalar((batches * self.trainable_value_count) as f32)
    }

    pub(super) fn amortization_distillation_loss(
        &self,
        generated_rows: Tensor3,
        teacher_rows: Tensor3,
        _npa: &NpaConfig,
    ) -> Tensor1 {
        let batches = generated_rows.shape().dims::<3>()[0];
        let row_dims = [batches, self.config.row_count, self.config.max_row_dims];
        debug_assert_eq!(teacher_rows.shape().dims::<3>(), row_dims);
        let scale = self.row_scale.clone().expand(row_dims);
        let mask = self.row_mask.clone().expand(row_dims);
        let residual = (generated_rows - detach3(teacher_rows)).div(scale).mul(mask);
        residual
            .clone()
            .mul(residual)
            .sum()
            .div_scalar((batches * self.trainable_value_count) as f32)
    }

    pub(super) fn endpoint_rms(&self, rows: Tensor3, _npa: &NpaConfig) -> Tensor1 {
        let batches = rows.shape().dims::<3>()[0];
        let row_dims = [batches, self.config.row_count, self.config.max_row_dims];
        let mask = self.row_mask.clone().expand(row_dims);
        let rows = rows.mul(mask);
        rows.clone()
            .mul(rows)
            .sum()
            .div_scalar((batches * self.trainable_value_count) as f32)
            .sqrt()
    }

    pub(super) fn prepare_condition(&self, condition: Tensor3) -> BurnRowFlowCondition {
        let hidden = linear3(
            condition,
            self.tensors[0].clone(),
            self.tensors[1].clone(),
        );
        let key_values = (0..self.config.layers)
            .map(|layer| {
                let offset = 9 + layer * 16;
                linear3(
                    hidden.clone(),
                    self.tensors[offset + 6].clone(),
                    self.tensors[offset + 7].clone(),
                )
            })
            .collect();
        BurnRowFlowCondition { key_values }
    }

    fn velocity(
        &self,
        condition: &BurnRowFlowCondition,
        state: Tensor3,
        time: Tensor2,
    ) -> Tensor3 {
        let row_w = self.tensors[2].clone();
        let row_b = self.tensors[3].clone();
        let row_embedding = self.tensors[4].clone();
        let time_w0 = self.tensors[5].clone();
        let time_b0 = self.tensors[6].clone();
        let time_w1 = self.tensors[7].clone();
        let time_b1 = self.tensors[8].clone();
        let batches = state.shape().dims::<3>()[0];
        let rows = self.config.row_count;
        let width = self.config.width;
        let row_dims = [batches, self.config.row_count, self.config.max_row_dims];
        let scale = self.row_scale.clone().expand(row_dims);
        let mask = self.row_mask.clone().expand(row_dims);
        let normalized_state = state.div(scale.clone()).mul(mask.clone());
        let mut hidden = linear3(normalized_state, row_w.clone(), row_b.clone())
            + row_embedding
                .clone()
                .unsqueeze_dim::<3>(0)
                .expand([batches, rows, width]);
        let phase = time.matmul(self.time_frequencies.clone());
        let time_features = Tensor::cat(vec![phase.clone().sin(), phase.cos()], 1)
            .narrow(1, 0, width);
        let time_hidden = gelu(linear2(time_features, time_w0.clone(), time_b0.clone()));
        let time_hidden = linear2(time_hidden, time_w1.clone(), time_b1.clone());
        let mut tensor_offset = 9usize;
        for layer in 0..self.config.layers {
            let block = &self.tensors[tensor_offset..tensor_offset + 16];
            tensor_offset += 16;
            let modulation = linear2(time_hidden.clone(), block[14].clone(), block[15].clone());

            let self_input = modulated_layer_norm3(
                hidden.clone(),
                modulation.clone().narrow(1, 0, width),
                modulation.clone().narrow(1, width, width),
            );
            let qkv = linear3(self_input, block[0].clone(), block[1].clone());
            let attended = self_attention3(qkv, self.config.heads);
            let self_output = linear3(attended, block[2].clone(), block[3].clone());
            hidden = gated_residual3(
                hidden,
                self_output,
                modulation.clone().narrow(1, 2 * width, width),
            );

            let cross_input = modulated_layer_norm3(
                hidden.clone(),
                modulation.clone().narrow(1, 3 * width, width),
                modulation.clone().narrow(1, 4 * width, width),
            );
            let query = linear3(cross_input, block[4].clone(), block[5].clone());
            let key_value = condition.key_values[layer].clone();
            let attended = cross_attention3(query, key_value, self.config.heads);
            let cross_output = linear3(attended, block[8].clone(), block[9].clone());
            hidden = gated_residual3(
                hidden,
                cross_output,
                modulation.clone().narrow(1, 5 * width, width),
            );

            let ffn_input = modulated_layer_norm3(
                hidden.clone(),
                modulation.clone().narrow(1, 6 * width, width),
                modulation.clone().narrow(1, 7 * width, width),
            );
            let ffn_hidden = gelu(linear3(ffn_input, block[10].clone(), block[11].clone()));
            let ffn_output = linear3(ffn_hidden, block[12].clone(), block[13].clone());
            hidden = gated_residual3(
                hidden,
                ffn_output,
                modulation.narrow(1, 8 * width, width),
            );
        }
        let output_w = self.tensors[tensor_offset].clone();
        let output_b = self.tensors[tensor_offset + 1].clone();
        linear3(layer_norm3(hidden), output_w, output_b)
            .mul(scale)
            .mul(mask)
    }
}

impl BurnAdapterBatch {
    pub(super) fn dense_residual_vector(&self, npa: &NpaConfig) -> Tensor2 {
        let batches = self.w1_down.shape().dims::<3>()[0];
        let p = npa.perception_dims();
        let h = npa.hidden_dims;
        let u = npa.update_dims();
        let scale = self.alpha / self.rank as f32;
        let w1 = self
            .w1_up
            .clone()
            .matmul(self.w1_down.clone())
            .mul_scalar(scale)
            .reshape([batches, h * p]);
        let b1 = self.b1_delta.clone().reshape([batches, h]);
        let w2 = self
            .w2_up
            .clone()
            .matmul(self.w2_down.clone())
            .mul_scalar(scale)
            .reshape([batches, u * h]);
        let b2 = self.b2_delta.clone().reshape([batches, u]);
        Tensor::cat(vec![w1, b1, w2, b2], 1)
    }

    pub(super) fn dense_residual_rows(&self, npa: &NpaConfig) -> Tensor3 {
        let batches = self.w1_down.shape().dims::<3>()[0];
        let p = npa.perception_dims();
        let h = npa.hidden_dims;
        let u = npa.update_dims();
        let m = (p + 1).max(h + 1);
        let scale = self.alpha / self.rank as f32;
        let w1 = self.w1_up.clone().matmul(self.w1_down.clone()).mul_scalar(scale);
        let b1 = self.b1_delta.clone().swap_dims(1, 2);
        let mut rows1 = Tensor::cat(vec![w1, b1], 2);
        if p + 1 < m {
            rows1 = Tensor::cat(
                vec![
                    rows1,
                    Tensor::<BurnBackend, 3>::zeros([batches, h, m - p - 1], &self.w1_down.device()),
                ],
                2,
            );
        }
        let w2 = self.w2_up.clone().matmul(self.w2_down.clone()).mul_scalar(scale);
        let b2 = self.b2_delta.clone().swap_dims(1, 2);
        let mut rows2 = Tensor::cat(vec![w2, b2], 2);
        if h + 1 < m {
            rows2 = Tensor::cat(
                vec![
                    rows2,
                    Tensor::<BurnBackend, 3>::zeros([batches, u, m - h - 1], &self.w1_down.device()),
                ],
                2,
            );
        }
        Tensor::cat(vec![rows1, rows2], 1)
    }

    pub(super) fn from_dense_residual_rows(rows: Tensor3, npa: &NpaConfig) -> Self {
        let batches = rows.shape().dims::<3>()[0];
        let device = rows.device();
        let p = npa.perception_dims();
        let h = npa.hidden_dims;
        let u = npa.update_dims();
        let rank = p.min(h).max(h.min(u));
        let w1 = rows.clone().narrow(1, 0, h).narrow(2, 0, p);
        let b1 = rows
            .clone()
            .narrow(1, 0, h)
            .narrow(2, p, 1)
            .swap_dims(1, 2);
        let w2 = rows.clone().narrow(1, h, u).narrow(2, 0, h);
        let b2 = rows.narrow(1, h, u).narrow(2, h, 1).swap_dims(1, 2);
        let (w1_down, w1_up) = if p <= rank {
            let fixed = canonical_identity(rank, p, p, &device)
                .expand([batches, rank, p]);
            let padding = Tensor::<BurnBackend, 3>::zeros([batches, h, rank - p], &device);
            (fixed, Tensor::cat(vec![w1, padding], 2))
        } else {
            let fixed = canonical_identity(h, rank, h, &device)
                .expand([batches, h, rank]);
            let padding = Tensor::<BurnBackend, 3>::zeros([batches, rank - h, p], &device);
            (Tensor::cat(vec![w1, padding], 1), fixed)
        };
        let (w2_down, w2_up) = if u <= rank {
            let fixed = canonical_identity(u, rank, u, &device)
                .expand([batches, u, rank]);
            let padding = Tensor::<BurnBackend, 3>::zeros([batches, rank - u, h], &device);
            (Tensor::cat(vec![w2, padding], 1), fixed)
        } else {
            let fixed = canonical_identity(rank, h, h, &device)
                .expand([batches, rank, h]);
            let padding = Tensor::<BurnBackend, 3>::zeros([batches, u, rank - h], &device);
            (fixed, Tensor::cat(vec![w2, padding], 2))
        };
        Self {
            rank,
            alpha: rank as f32,
            canonical_dense_residual: true,
            w1_down,
            w1_up,
            w2_down,
            w2_up,
            b1_delta: b1,
            b2_delta: b2,
        }
    }
}

fn endpoint_row_rms(
    npa: &NpaConfig,
    examples: &[BurnE2eRolloutExample],
    config: BurnE2eRolloutTrainConfig,
    layout: &NpaParameterRowLayout2d,
) -> AutomataResult<Vec<f32>> {
    let endpoints = examples
        .iter()
        .filter_map(|example| example.teacher_adapter.as_ref())
        .map(|values| {
            let adapter = NpaLowRankAdapter::from_parameter_vector(
                npa,
                config.adapter_rank,
                config.adapter_alpha,
                values.clone(),
            )?;
            layout.adapter_to_packed(&adapter)
        })
        .collect::<AutomataResult<Vec<_>>>()?;
    if endpoints.is_empty() {
        return Ok(vec![config.generator_default_endpoint_rms; layout.row_count()]);
    }
    Ok(layout
        .rows()
        .iter()
        .enumerate()
        .map(|(row_idx, row)| {
            let start = row_idx * layout.max_row_dims();
            let sum = endpoints
                .iter()
                .flat_map(|endpoint| endpoint[start..start + row.value_dims].iter())
                .map(|value| value * value)
                .sum::<f32>();
            (sum / (endpoints.len() * row.value_dims) as f32)
                .sqrt()
                .max(1.0e-3)
        })
        .collect())
}

fn linear2(input: Tensor2, weight: Tensor2, bias: Tensor2) -> Tensor2 {
    let rows = input.shape().dims::<2>()[0];
    let output = weight.shape().dims::<2>()[1];
    input.matmul(weight) + bias.expand([rows, output])
}

fn linear3(input: Tensor3, weight: Tensor2, bias: Tensor2) -> Tensor3 {
    let [batches, rows, _] = input.shape().dims::<3>();
    let output = weight.shape().dims::<2>()[1];
    input
        .flatten(0, 1)
        .matmul(weight)
        .reshape([batches, rows, output])
        + bias
            .unsqueeze_dim::<3>(0)
            .expand([batches, rows, output])
}

pub(super) fn layer_norm3(input: Tensor3) -> Tensor3 {
    let [batches, rows, dims] = input.shape().dims::<3>();
    let mean = input.clone().mean_dim(2);
    let centered = input - mean.expand([batches, rows, dims]);
    let variance = centered.clone().mul(centered.clone()).mean_dim(2);
    centered.div(
        variance
            .add_scalar(1.0e-6)
            .sqrt()
            .expand([batches, rows, dims]),
    )
}

fn gated_residual3(input: Tensor3, update: Tensor3, gate: Tensor2) -> Tensor3 {
    let [batches, rows, dims] = input.shape().dims::<3>();
    input + update.mul(gate.unsqueeze_dim::<3>(1).expand([batches, rows, dims]))
}

fn self_attention3(qkv: Tensor3, heads: usize) -> Tensor3 {
    let [batches, tokens, triple_width] = qkv.shape().dims::<3>();
    let width = triple_width / 3;
    let head_dims = width / heads;
    let q = qkv
        .clone()
        .narrow(2, 0, width)
        .reshape([batches, tokens, heads, head_dims])
        .swap_dims(1, 2);
    let k = qkv
        .clone()
        .narrow(2, width, width)
        .reshape([batches, tokens, heads, head_dims])
        .swap_dims(1, 2);
    let v = qkv
        .narrow(2, 2 * width, width)
        .reshape([batches, tokens, heads, head_dims])
        .swap_dims(1, 2);
    tiled_attention_adjoint(q, k, v)
        .swap_dims(1, 2)
        .reshape([batches, tokens, width])
}

fn cross_attention3(query: Tensor3, key_value: Tensor3, heads: usize) -> Tensor3 {
    let [batches, query_tokens, width] = query.shape().dims::<3>();
    let condition_tokens = key_value.shape().dims::<3>()[1];
    let head_dims = width / heads;
    let q = query
        .reshape([batches, query_tokens, heads, head_dims])
        .swap_dims(1, 2);
    let k = key_value
        .clone()
        .narrow(2, 0, width)
        .reshape([batches, condition_tokens, heads, head_dims])
        .swap_dims(1, 2);
    let v = key_value
        .narrow(2, width, width)
        .reshape([batches, condition_tokens, heads, head_dims])
        .swap_dims(1, 2);
    tiled_attention_adjoint(q, k, v)
        .swap_dims(1, 2)
        .reshape([batches, query_tokens, width])
}

fn static_time_frequencies(width: usize, device: &BurnDevice) -> Tensor2 {
    let half = width.div_ceil(2);
    let values = (0..half)
        .map(|index| 10_000.0_f32.powf(-(index as f32 / half as f32)))
        .collect::<Vec<_>>();
    Tensor::<BurnBackend, 2>::from_data(TensorData::new(values, [1, half]), device)
}

fn static_row_scale(flow: &ConditionalRowFlowConfig, device: &BurnDevice) -> Tensor3 {
    let values = flow
        .row_rms
        .iter()
        .flat_map(|scale| std::iter::repeat_n(*scale, flow.max_row_dims))
        .collect::<Vec<_>>();
    Tensor::<BurnBackend, 3>::from_data(
        TensorData::new(values, [1, flow.row_count, flow.max_row_dims]),
        device,
    )
}

fn static_row_mask(
    layout: &NpaParameterRowLayout2d,
    output_bias: bool,
    device: &BurnDevice,
) -> Tensor3 {
    let values = layout.trainable_mask(output_bias);
    Tensor::<BurnBackend, 3>::from_data(
        TensorData::new(values, [1, layout.row_count(), layout.max_row_dims()]),
        device,
    )
}

fn static_source_rows(
    layout: &NpaParameterRowLayout2d,
    flow: &ConditionalRowFlowConfig,
    device: &BurnDevice,
) -> Tensor3 {
    Tensor::<BurnBackend, 3>::from_data(
        TensorData::new(
            layout.deterministic_source(flow),
            [1, layout.row_count(), layout.max_row_dims()],
        ),
        device,
    )
}

fn canonical_identity(
    rows: usize,
    cols: usize,
    diagonal: usize,
    device: &BurnDevice,
) -> Tensor3 {
    let mut values = vec![0.0; rows * cols];
    for index in 0..diagonal {
        values[index * cols + index] = 1.0;
    }
    Tensor::<BurnBackend, 3>::from_data(TensorData::new(values, [1, rows, cols]), device)
}
