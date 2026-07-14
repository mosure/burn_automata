//! Structured controller rows and the conditional rectified-flow artifact contract.

use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{AutomataError, AutomataResult, NpaConfig, NpaLowRankAdapter, NpaWeights};

pub const CONDITIONAL_ROW_FLOW_ARCHITECTURE_LEGACY: &str = "conditional-row-rectified-flow-v4";
pub const CONDITIONAL_ROW_FLOW_ARCHITECTURE: &str = "conditional-row-rectified-flow-v5";
pub const CONDITIONAL_ROW_FLOW_SOLVER_HEUN: &str = "heun";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConditionalRowFlowConfig {
    pub layers: usize,
    pub width: usize,
    pub heads: usize,
    pub ffn_dims: usize,
    pub condition_dims: usize,
    pub condition_tokens: usize,
    pub row_count: usize,
    pub max_row_dims: usize,
    /// Number of non-padding values in each controller row.
    pub row_value_dims: Vec<usize>,
    pub sample_steps: usize,
    pub source_seed: u64,
    pub source_scale: f32,
    pub solver: String,
    /// One positive scale per controller row. No elementwise centering is used.
    pub row_rms: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConditionalRowFlowWeights {
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConditionalRowFlowTensorSpec {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
}

impl ConditionalRowFlowConfig {
    pub fn flow_s(npa: &NpaConfig, condition_tokens: usize, condition_dims: usize) -> Self {
        let rows = NpaParameterRowLayout2d::new(npa);
        Self {
            layers: 12,
            width: 768,
            heads: 12,
            ffn_dims: 3072,
            condition_dims,
            condition_tokens,
            row_count: rows.row_count(),
            max_row_dims: rows.max_row_dims(),
            row_value_dims: rows.rows().iter().map(|row| row.value_dims).collect(),
            sample_steps: 8,
            source_seed: 42,
            source_scale: 1.0e-3,
            solver: CONDITIONAL_ROW_FLOW_SOLVER_HEUN.to_string(),
            row_rms: vec![1.0; rows.row_count()],
        }
    }

    pub fn validate(&self) -> AutomataResult<()> {
        if self.layers == 0
            || self.width == 0
            || self.heads == 0
            || self.ffn_dims == 0
            || self.condition_dims == 0
            || self.condition_tokens == 0
            || self.row_count == 0
            || self.max_row_dims == 0
            || self.sample_steps == 0
        {
            return Err(AutomataError::InvalidModel(
                "conditional row flow dimensions and sample steps must be positive".to_string(),
            ));
        }
        if !self.width.is_multiple_of(self.heads) {
            return Err(AutomataError::InvalidModel(format!(
                "conditional row flow width {} must be divisible by {} heads",
                self.width, self.heads
            )));
        }
        if !self.width.is_multiple_of(2) {
            return Err(AutomataError::InvalidModel(
                "conditional row flow width must be even for sinusoidal time features".to_string(),
            ));
        }
        if self.solver != CONDITIONAL_ROW_FLOW_SOLVER_HEUN {
            return Err(AutomataError::InvalidModel(format!(
                "unsupported conditional row flow solver {:?}",
                self.solver
            )));
        }
        if !self.source_scale.is_finite() || self.source_scale <= 0.0 {
            return Err(AutomataError::InvalidModel(
                "conditional row flow source_scale must be positive and finite".to_string(),
            ));
        }
        if self.row_rms.len() != self.row_count
            || self
                .row_rms
                .iter()
                .any(|scale| !scale.is_finite() || *scale <= 0.0)
        {
            return Err(AutomataError::InvalidModel(format!(
                "conditional row flow requires {} positive finite row RMS values",
                self.row_count
            )));
        }
        if self.row_value_dims.len() != self.row_count
            || self
                .row_value_dims
                .iter()
                .any(|dims| *dims == 0 || *dims > self.max_row_dims)
        {
            return Err(AutomataError::InvalidModel(format!(
                "conditional row flow requires {} valid row widths in 1..={}",
                self.row_count, self.max_row_dims
            )));
        }
        Ok(())
    }

    pub fn parameter_count(&self) -> AutomataResult<usize> {
        self.tensor_specs()?
            .into_iter()
            .try_fold(0usize, |sum, spec| {
                sum.checked_add(spec.rows.checked_mul(spec.cols)?)
            })
            .ok_or_else(|| {
                AutomataError::InvalidModel("row flow parameter count overflowed".to_string())
            })
    }

    pub fn tensor_specs(&self) -> AutomataResult<Vec<ConditionalRowFlowTensorSpec>> {
        self.validate()?;
        let d = self.width;
        let m = self.max_row_dims;
        let f = self.ffn_dims;
        let mut specs = vec![
            spec("condition.weight", self.condition_dims, d),
            spec("condition.bias", 1, d),
            spec("row_input.weight", m, d),
            spec("row_input.bias", 1, d),
            spec("row_embedding", self.row_count, d),
            spec("time.0.weight", d, d),
            spec("time.0.bias", 1, d),
            spec("time.1.weight", d, d),
            spec("time.1.bias", 1, d),
        ];
        for layer in 0..self.layers {
            let prefix = format!("blocks.{layer}");
            specs.extend([
                spec(format!("{prefix}.self_qkv.weight"), d, 3 * d),
                spec(format!("{prefix}.self_qkv.bias"), 1, 3 * d),
                spec(format!("{prefix}.self_out.weight"), d, d),
                spec(format!("{prefix}.self_out.bias"), 1, d),
                spec(format!("{prefix}.cross_q.weight"), d, d),
                spec(format!("{prefix}.cross_q.bias"), 1, d),
                spec(format!("{prefix}.cross_kv.weight"), d, 2 * d),
                spec(format!("{prefix}.cross_kv.bias"), 1, 2 * d),
                spec(format!("{prefix}.cross_out.weight"), d, d),
                spec(format!("{prefix}.cross_out.bias"), 1, d),
                spec(format!("{prefix}.ffn.0.weight"), d, f),
                spec(format!("{prefix}.ffn.0.bias"), 1, f),
                spec(format!("{prefix}.ffn.1.weight"), f, d),
                spec(format!("{prefix}.ffn.1.bias"), 1, d),
                spec(format!("{prefix}.modulation.weight"), d, 9 * d),
                spec(format!("{prefix}.modulation.bias"), 1, 9 * d),
            ]);
        }
        specs.extend([spec("output.weight", d, m), spec("output.bias", 1, m)]);
        Ok(specs)
    }
}

impl ConditionalRowFlowWeights {
    pub fn seeded(config: &ConditionalRowFlowConfig, seed: u64) -> AutomataResult<Self> {
        let specs = config.tensor_specs()?;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut values = Vec::with_capacity(config.parameter_count()?);
        for tensor in specs {
            let is_bias = tensor.name.ends_with("bias");
            let is_modulation = tensor.name.contains("modulation");
            let is_output = tensor.name == "output.weight";
            let scale = if is_output {
                1.0e-3
            } else {
                (tensor.rows as f32).sqrt().recip()
            };
            for _ in 0..tensor.rows * tensor.cols {
                values.push(if is_bias || is_modulation {
                    0.0
                } else {
                    rng.random_range(-scale..=scale)
                });
            }
        }
        let result = Self { values };
        result.validate(config)?;
        Ok(result)
    }

    pub fn validate(&self, config: &ConditionalRowFlowConfig) -> AutomataResult<()> {
        let expected = config.parameter_count()?;
        if self.values.len() != expected || self.values.iter().any(|value| !value.is_finite()) {
            return Err(AutomataError::InvalidModel(format!(
                "conditional row flow weights must contain {expected} finite values, got {}",
                self.values.len()
            )));
        }
        Ok(())
    }

    pub fn predict_packed(
        &self,
        flow: &ConditionalRowFlowConfig,
        npa: &NpaConfig,
        condition: &[f32],
    ) -> AutomataResult<Vec<f32>> {
        self.validate(flow)?;
        let layout = NpaParameterRowLayout2d::new(npa);
        layout.validate_flow_config(flow)?;
        let expected_condition = flow.condition_tokens * flow.condition_dims;
        if condition.len() != expected_condition || condition.iter().any(|value| !value.is_finite())
        {
            return Err(AutomataError::InvalidArgument(format!(
                "conditional row flow expected {expected_condition} finite condition values, got {}",
                condition.len()
            )));
        }
        let tensors = FlowTensorViews::new(flow, &self.values)?;
        let condition_hidden = linear(
            condition,
            flow.condition_tokens,
            flow.condition_dims,
            tensors.get("condition.weight"),
            tensors.get("condition.bias"),
            flow.width,
        );
        let mut state = layout.deterministic_source(flow);
        let dt = 1.0 / flow.sample_steps as f32;
        for step in 0..flow.sample_steps {
            let t0 = step as f32 * dt;
            let v0 = flow_velocity(flow, &layout, &tensors, &condition_hidden, &state, t0);
            let predictor = state
                .iter()
                .zip(&v0)
                .map(|(state, velocity)| state + velocity * dt)
                .collect::<Vec<_>>();
            let v1 = flow_velocity(
                flow,
                &layout,
                &tensors,
                &condition_hidden,
                &predictor,
                t0 + dt,
            );
            for ((state, first), second) in state.iter_mut().zip(v0).zip(v1) {
                *state += 0.5 * (first + second) * dt;
            }
        }
        for (row_idx, row) in layout.rows().iter().enumerate() {
            let start = row_idx * flow.max_row_dims + row.value_dims;
            let end = (row_idx + 1) * flow.max_row_dims;
            state[start..end].fill(0.0);
        }
        Ok(state)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NpaParameterRowModule2d {
    W1,
    W2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NpaParameterRow2d {
    pub module: NpaParameterRowModule2d,
    pub module_row: usize,
    pub value_offset: usize,
    pub value_dims: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NpaParameterRowLayout2d {
    config: NpaConfig,
    rows: Vec<NpaParameterRow2d>,
}

impl NpaParameterRowLayout2d {
    pub fn new(config: &NpaConfig) -> Self {
        let perception_dims = config.perception_dims();
        let hidden_dims = config.hidden_dims;
        let update_dims = config.update_dims();
        let mut rows = Vec::with_capacity(hidden_dims + update_dims);
        let mut value_offset = 0;
        for module_row in 0..hidden_dims {
            let value_dims = perception_dims + 1;
            rows.push(NpaParameterRow2d {
                module: NpaParameterRowModule2d::W1,
                module_row,
                value_offset,
                value_dims,
            });
            value_offset += value_dims;
        }
        for module_row in 0..update_dims {
            let value_dims = hidden_dims + 1;
            rows.push(NpaParameterRow2d {
                module: NpaParameterRowModule2d::W2,
                module_row,
                value_offset,
                value_dims,
            });
            value_offset += value_dims;
        }
        Self {
            config: config.clone(),
            rows,
        }
    }

    pub fn rows(&self) -> &[NpaParameterRow2d] {
        &self.rows
    }

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    pub fn max_row_dims(&self) -> usize {
        (self.config.perception_dims() + 1).max(self.config.hidden_dims + 1)
    }

    pub fn parameter_count(&self) -> usize {
        self.rows.iter().map(|row| row.value_dims).sum()
    }

    pub fn canonical_rank(&self) -> usize {
        let perception_dims = self.config.perception_dims();
        let hidden_dims = self.config.hidden_dims;
        let update_dims = self.config.update_dims();
        perception_dims
            .min(hidden_dims)
            .max(hidden_dims.min(update_dims))
    }

    pub fn validate_flow_config(&self, flow: &ConditionalRowFlowConfig) -> AutomataResult<()> {
        flow.validate()?;
        if flow.row_count != self.row_count() || flow.max_row_dims != self.max_row_dims() {
            return Err(AutomataError::InvalidModel(format!(
                "row flow layout {}x{} does not match NPA layout {}x{}",
                flow.row_count,
                flow.max_row_dims,
                self.row_count(),
                self.max_row_dims()
            )));
        }
        let expected_row_dims = self
            .rows
            .iter()
            .map(|row| row.value_dims)
            .collect::<Vec<_>>();
        if flow.row_value_dims != expected_row_dims {
            return Err(AutomataError::InvalidModel(
                "row flow valid row widths do not match the NPA parameter layout".to_string(),
            ));
        }
        Ok(())
    }

    pub fn pack_weights(&self, weights: &NpaWeights) -> AutomataResult<Vec<f32>> {
        weights.validate(&self.config)?;
        let p = self.config.perception_dims();
        let h = self.config.hidden_dims;
        let mut packed = vec![0.0; self.row_count() * self.max_row_dims()];
        for (global_row, row) in self.rows.iter().enumerate() {
            let dst = global_row * self.max_row_dims();
            match row.module {
                NpaParameterRowModule2d::W1 => {
                    let source = row.module_row * p;
                    packed[dst..dst + p].copy_from_slice(&weights.w1[source..source + p]);
                    packed[dst + p] = weights.b1[row.module_row];
                }
                NpaParameterRowModule2d::W2 => {
                    let source = row.module_row * h;
                    packed[dst..dst + h].copy_from_slice(&weights.w2[source..source + h]);
                    packed[dst + h] = weights.b2[row.module_row];
                }
            }
        }
        Ok(packed)
    }

    pub fn unpack_weights(&self, packed: &[f32]) -> AutomataResult<NpaWeights> {
        self.validate_packed(packed)?;
        let p = self.config.perception_dims();
        let h = self.config.hidden_dims;
        let u = self.config.update_dims();
        let mut weights = NpaWeights {
            w1: vec![0.0; h * p],
            b1: vec![0.0; h],
            w2: vec![0.0; u * h],
            b2: vec![0.0; u],
        };
        for (global_row, row) in self.rows.iter().enumerate() {
            let src = global_row * self.max_row_dims();
            match row.module {
                NpaParameterRowModule2d::W1 => {
                    let dst = row.module_row * p;
                    weights.w1[dst..dst + p].copy_from_slice(&packed[src..src + p]);
                    weights.b1[row.module_row] = packed[src + p];
                }
                NpaParameterRowModule2d::W2 => {
                    let dst = row.module_row * h;
                    weights.w2[dst..dst + h].copy_from_slice(&packed[src..src + h]);
                    weights.b2[row.module_row] = packed[src + h];
                }
            }
        }
        Ok(weights)
    }

    pub fn adapter_to_packed(&self, adapter: &NpaLowRankAdapter) -> AutomataResult<Vec<f32>> {
        adapter.validate(&self.config)?;
        let h = self.config.hidden_dims;
        let p = self.config.perception_dims();
        let u = self.config.update_dims();
        let scale = adapter.alpha / adapter.rank as f32;
        let mut dense = NpaWeights {
            w1: matmul_up_down(&adapter.w1_up, &adapter.w1_down, h, p, adapter.rank, scale),
            b1: adapter.b1_delta.clone(),
            w2: matmul_up_down(&adapter.w2_up, &adapter.w2_down, u, h, adapter.rank, scale),
            b2: adapter.b2_delta.clone(),
        };
        for (value, correction) in dense.b1.iter_mut().zip(&adapter.b1_delta_correction) {
            *value += correction;
        }
        for (value, correction) in dense.b2.iter_mut().zip(&adapter.b2_delta_correction) {
            *value += correction;
        }
        self.pack_weights(&dense)
    }

    pub fn packed_to_canonical_adapter(&self, packed: &[f32]) -> AutomataResult<NpaLowRankAdapter> {
        let dense = self.unpack_weights(packed)?;
        let p = self.config.perception_dims();
        let h = self.config.hidden_dims;
        let u = self.config.update_dims();
        let rank = self.canonical_rank();
        let alpha = rank as f32;
        let mut adapter = NpaLowRankAdapter::zeros(&self.config, rank, alpha);
        if p <= rank {
            for dim in 0..p {
                adapter.w1_down[dim * p + dim] = 1.0;
            }
            for row in 0..h {
                let src = row * p;
                let dst = row * rank;
                adapter.w1_up[dst..dst + p].copy_from_slice(&dense.w1[src..src + p]);
            }
        } else {
            for dim in 0..h {
                adapter.w1_up[dim * rank + dim] = 1.0;
                let src = dim * p;
                adapter.w1_down[src..src + p].copy_from_slice(&dense.w1[src..src + p]);
            }
        }
        if u <= rank {
            for dim in 0..u {
                adapter.w2_up[dim * rank + dim] = 1.0;
                let src = dim * h;
                adapter.w2_down[src..src + h].copy_from_slice(&dense.w2[src..src + h]);
            }
        } else {
            for dim in 0..h {
                adapter.w2_down[dim * h + dim] = 1.0;
            }
            for row in 0..u {
                let src = row * h;
                let dst = row * rank;
                adapter.w2_up[dst..dst + h].copy_from_slice(&dense.w2[src..src + h]);
            }
        }
        adapter.b1_delta = dense.b1;
        adapter.b2_delta = dense.b2;
        adapter.validate(&self.config)?;
        Ok(adapter)
    }

    pub fn deterministic_source(&self, flow: &ConditionalRowFlowConfig) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(flow.source_seed);
        let mut source = vec![0.0; self.row_count() * self.max_row_dims()];
        for (row_idx, row) in self.rows.iter().enumerate() {
            let scale = flow.row_rms[row_idx] * flow.source_scale;
            let start = row_idx * self.max_row_dims();
            for value in &mut source[start..start + row.value_dims] {
                let u1 = rng.random_range(f32::MIN_POSITIVE..1.0);
                let u2 = rng.random_range(0.0..1.0);
                let normal = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
                *value = normal * scale;
            }
        }
        source
    }

    fn validate_packed(&self, packed: &[f32]) -> AutomataResult<()> {
        let expected = self.row_count() * self.max_row_dims();
        if packed.len() != expected || packed.iter().any(|value| !value.is_finite()) {
            return Err(AutomataError::InvalidModel(format!(
                "packed NPA rows must contain {expected} finite values, got {}",
                packed.len()
            )));
        }
        Ok(())
    }
}

fn matmul_up_down(
    up: &[f32],
    down: &[f32],
    output_dims: usize,
    input_dims: usize,
    rank: usize,
    scale: f32,
) -> Vec<f32> {
    let mut output = vec![0.0; output_dims * input_dims];
    for row in 0..output_dims {
        for col in 0..input_dims {
            let mut sum = 0.0;
            for inner in 0..rank {
                sum += up[row * rank + inner] * down[inner * input_dims + col];
            }
            output[row * input_dims + col] = sum * scale;
        }
    }
    output
}

fn spec(name: impl Into<String>, rows: usize, cols: usize) -> ConditionalRowFlowTensorSpec {
    ConditionalRowFlowTensorSpec {
        name: name.into(),
        rows,
        cols,
    }
}

struct FlowTensorView<'a> {
    name: String,
    rows: usize,
    cols: usize,
    values: &'a [f32],
}

struct FlowTensorViews<'a> {
    tensors: Vec<FlowTensorView<'a>>,
}

impl<'a> FlowTensorViews<'a> {
    fn new(config: &ConditionalRowFlowConfig, values: &'a [f32]) -> AutomataResult<Self> {
        let specs = config.tensor_specs()?;
        let mut offset = 0usize;
        let mut tensors = Vec::with_capacity(specs.len());
        for spec in specs {
            let len = spec.rows * spec.cols;
            let end = offset + len;
            tensors.push(FlowTensorView {
                name: spec.name,
                rows: spec.rows,
                cols: spec.cols,
                values: &values[offset..end],
            });
            offset = end;
        }
        if offset != values.len() {
            return Err(AutomataError::InvalidModel(
                "conditional row flow tensor layout does not consume all weights".to_string(),
            ));
        }
        Ok(Self { tensors })
    }

    fn get(&self, name: &str) -> &FlowTensorView<'a> {
        self.tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap_or_else(|| panic!("row flow tensor layout is missing {name}"))
    }
}

fn flow_velocity(
    flow: &ConditionalRowFlowConfig,
    layout: &NpaParameterRowLayout2d,
    tensors: &FlowTensorViews<'_>,
    condition_hidden: &[f32],
    state: &[f32],
    t: f32,
) -> Vec<f32> {
    let d = flow.width;
    let rows = flow.row_count;
    let m = flow.max_row_dims;
    let mut normalized_state = state.to_vec();
    for (row_idx, row) in layout.rows().iter().enumerate() {
        let scale = flow.row_rms[row_idx];
        let start = row_idx * m;
        for value in &mut normalized_state[start..start + row.value_dims] {
            *value /= scale;
        }
        normalized_state[start + row.value_dims..start + m].fill(0.0);
    }
    let mut hidden = linear(
        &normalized_state,
        rows,
        m,
        tensors.get("row_input.weight"),
        tensors.get("row_input.bias"),
        d,
    );
    add_rows_in_place(&mut hidden, tensors.get("row_embedding").values, rows, d);
    let time_features = sinusoidal_time(t, d);
    let time_hidden = linear(
        &time_features,
        1,
        d,
        tensors.get("time.0.weight"),
        tensors.get("time.0.bias"),
        d,
    )
    .into_iter()
    .map(gelu)
    .collect::<Vec<_>>();
    let time_hidden = linear(
        &time_hidden,
        1,
        d,
        tensors.get("time.1.weight"),
        tensors.get("time.1.bias"),
        d,
    );
    for layer in 0..flow.layers {
        let prefix = format!("blocks.{layer}");
        let modulation = linear(
            &time_hidden,
            1,
            d,
            tensors.get(&format!("{prefix}.modulation.weight")),
            tensors.get(&format!("{prefix}.modulation.bias")),
            9 * d,
        );
        let self_input =
            modulated_layer_norm(&hidden, rows, d, &modulation[0..d], &modulation[d..2 * d]);
        let self_qkv = linear(
            &self_input,
            rows,
            d,
            tensors.get(&format!("{prefix}.self_qkv.weight")),
            tensors.get(&format!("{prefix}.self_qkv.bias")),
            3 * d,
        );
        let self_attended = self_attention(&self_qkv, rows, d, flow.heads);
        let self_output = linear(
            &self_attended,
            rows,
            d,
            tensors.get(&format!("{prefix}.self_out.weight")),
            tensors.get(&format!("{prefix}.self_out.bias")),
            d,
        );
        gated_residual(
            &mut hidden,
            &self_output,
            rows,
            d,
            &modulation[2 * d..3 * d],
        );

        let cross_input = modulated_layer_norm(
            &hidden,
            rows,
            d,
            &modulation[3 * d..4 * d],
            &modulation[4 * d..5 * d],
        );
        let query = linear(
            &cross_input,
            rows,
            d,
            tensors.get(&format!("{prefix}.cross_q.weight")),
            tensors.get(&format!("{prefix}.cross_q.bias")),
            d,
        );
        let key_value = linear(
            condition_hidden,
            flow.condition_tokens,
            d,
            tensors.get(&format!("{prefix}.cross_kv.weight")),
            tensors.get(&format!("{prefix}.cross_kv.bias")),
            2 * d,
        );
        let cross_attended = cross_attention(
            &query,
            &key_value,
            rows,
            flow.condition_tokens,
            d,
            flow.heads,
        );
        let cross_output = linear(
            &cross_attended,
            rows,
            d,
            tensors.get(&format!("{prefix}.cross_out.weight")),
            tensors.get(&format!("{prefix}.cross_out.bias")),
            d,
        );
        gated_residual(
            &mut hidden,
            &cross_output,
            rows,
            d,
            &modulation[5 * d..6 * d],
        );

        let ffn_input = modulated_layer_norm(
            &hidden,
            rows,
            d,
            &modulation[6 * d..7 * d],
            &modulation[7 * d..8 * d],
        );
        let ffn_hidden = linear(
            &ffn_input,
            rows,
            d,
            tensors.get(&format!("{prefix}.ffn.0.weight")),
            tensors.get(&format!("{prefix}.ffn.0.bias")),
            flow.ffn_dims,
        )
        .into_iter()
        .map(gelu)
        .collect::<Vec<_>>();
        let ffn_output = linear(
            &ffn_hidden,
            rows,
            flow.ffn_dims,
            tensors.get(&format!("{prefix}.ffn.1.weight")),
            tensors.get(&format!("{prefix}.ffn.1.bias")),
            d,
        );
        gated_residual(&mut hidden, &ffn_output, rows, d, &modulation[8 * d..9 * d]);
    }
    let hidden = layer_norm(&hidden, rows, d);
    let mut velocity = linear(
        &hidden,
        rows,
        d,
        tensors.get("output.weight"),
        tensors.get("output.bias"),
        m,
    );
    for (row_idx, row) in layout.rows().iter().enumerate() {
        let scale = flow.row_rms[row_idx];
        let start = row_idx * m;
        for value in &mut velocity[start..start + row.value_dims] {
            *value *= scale;
        }
        velocity[start + row.value_dims..start + m].fill(0.0);
    }
    velocity
}

fn linear(
    input: &[f32],
    input_rows: usize,
    input_dims: usize,
    weight: &FlowTensorView<'_>,
    bias: &FlowTensorView<'_>,
    output_dims: usize,
) -> Vec<f32> {
    debug_assert_eq!((weight.rows, weight.cols), (input_dims, output_dims));
    debug_assert_eq!((bias.rows, bias.cols), (1, output_dims));
    let mut output = vec![0.0; input_rows * output_dims];
    output
        .par_chunks_mut(output_dims)
        .enumerate()
        .for_each(|(row, output)| {
            let input = &input[row * input_dims..(row + 1) * input_dims];
            for (col, output_value) in output.iter_mut().enumerate() {
                let mut sum = bias.values[col];
                for (inner, input_value) in input.iter().enumerate() {
                    sum += input_value * weight.values[inner * output_dims + col];
                }
                *output_value = sum;
            }
        });
    output
}

fn layer_norm(input: &[f32], rows: usize, dims: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    output
        .par_chunks_mut(dims)
        .zip(input.par_chunks(dims))
        .for_each(|(output, input)| {
            let mean = input.iter().sum::<f32>() / dims as f32;
            let variance = input
                .iter()
                .map(|value| {
                    let delta = value - mean;
                    delta * delta
                })
                .sum::<f32>()
                / dims as f32;
            let inverse_std = (variance + 1.0e-6).sqrt().recip();
            for (output, input) in output.iter_mut().zip(input) {
                *output = (*input - mean) * inverse_std;
            }
        });
    debug_assert_eq!(output.len(), rows * dims);
    output
}

fn modulated_layer_norm(
    input: &[f32],
    rows: usize,
    dims: usize,
    shift: &[f32],
    scale: &[f32],
) -> Vec<f32> {
    let mut output = layer_norm(input, rows, dims);
    for row in output.chunks_mut(dims) {
        for dim in 0..dims {
            row[dim] = row[dim] * (1.0 + scale[dim]) + shift[dim];
        }
    }
    output
}

fn self_attention(qkv: &[f32], tokens: usize, width: usize, heads: usize) -> Vec<f32> {
    let mut output = vec![0.0; tokens * width];
    let head_dims = width / heads;
    let scale = (head_dims as f32).sqrt().recip();
    for head in 0..heads {
        for query_token in 0..tokens {
            let mut logits = vec![0.0; tokens];
            for key_token in 0..tokens {
                let mut dot = 0.0;
                for dim in 0..head_dims {
                    let channel = head * head_dims + dim;
                    dot += qkv[query_token * 3 * width + channel]
                        * qkv[key_token * 3 * width + width + channel];
                }
                logits[key_token] = dot * scale;
            }
            softmax_in_place(&mut logits);
            for dim in 0..head_dims {
                let channel = head * head_dims + dim;
                output[query_token * width + channel] = logits
                    .iter()
                    .enumerate()
                    .map(|(token, weight)| weight * qkv[token * 3 * width + 2 * width + channel])
                    .sum();
            }
        }
    }
    output
}

fn cross_attention(
    query: &[f32],
    key_value: &[f32],
    query_tokens: usize,
    condition_tokens: usize,
    width: usize,
    heads: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; query_tokens * width];
    let head_dims = width / heads;
    let scale = (head_dims as f32).sqrt().recip();
    for head in 0..heads {
        for query_token in 0..query_tokens {
            let mut logits = vec![0.0; condition_tokens];
            for key_token in 0..condition_tokens {
                let mut dot = 0.0;
                for dim in 0..head_dims {
                    let channel = head * head_dims + dim;
                    dot += query[query_token * width + channel]
                        * key_value[key_token * 2 * width + channel];
                }
                logits[key_token] = dot * scale;
            }
            softmax_in_place(&mut logits);
            for dim in 0..head_dims {
                let channel = head * head_dims + dim;
                output[query_token * width + channel] = logits
                    .iter()
                    .enumerate()
                    .map(|(token, weight)| weight * key_value[token * 2 * width + width + channel])
                    .sum();
            }
        }
    }
    output
}

fn softmax_in_place(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    let inverse = sum.max(f32::MIN_POSITIVE).recip();
    for value in values {
        *value *= inverse;
    }
}

fn gated_residual(state: &mut [f32], update: &[f32], rows: usize, dims: usize, gate: &[f32]) {
    debug_assert_eq!(state.len(), rows * dims);
    for row in 0..rows {
        for dim in 0..dims {
            state[row * dims + dim] += update[row * dims + dim] * gate[dim];
        }
    }
}

fn add_rows_in_place(output: &mut [f32], rows: &[f32], row_count: usize, dims: usize) {
    debug_assert_eq!(output.len(), row_count * dims);
    debug_assert_eq!(rows.len(), row_count * dims);
    for (output, row) in output.iter_mut().zip(rows) {
        *output += row;
    }
}

fn sinusoidal_time(t: f32, dims: usize) -> Vec<f32> {
    let half = dims / 2;
    let mut output = vec![0.0; dims];
    for index in 0..half {
        let exponent = index as f32 / half.max(1) as f32;
        let frequency = 10_000.0_f32.powf(-exponent);
        output[index] = (t * frequency).sin();
        output[half + index] = (t * frequency).cos();
    }
    output
}

fn gelu(value: f32) -> f32 {
    0.5 * value
        * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI * (value + 0.044_715 * value.powi(3))).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_rows_round_trip_through_canonical_adapter() {
        let (config, _) = NpaConfig::for_preset(crate::AutomataPreset::Growing2d);
        let layout = NpaParameterRowLayout2d::new(&config);
        let mut values = (0..layout.row_count() * layout.max_row_dims())
            .map(|index| index as f32 * 1.0e-4)
            .collect::<Vec<_>>();
        for (row_idx, row) in layout.rows().iter().enumerate() {
            let start = row_idx * layout.max_row_dims() + row.value_dims;
            let end = (row_idx + 1) * layout.max_row_dims();
            values[start..end].fill(0.0);
        }
        let adapter = layout.packed_to_canonical_adapter(&values).unwrap();
        let restored = layout.adapter_to_packed(&adapter).unwrap();
        assert_eq!(restored, values);
    }

    #[test]
    fn flow_s_contract_matches_npa_rows() {
        let (config, _) = NpaConfig::for_preset(crate::AutomataPreset::Growing2d);
        let flow = ConditionalRowFlowConfig::flow_s(&config, 257, 388);
        let layout = NpaParameterRowLayout2d::new(&config);
        layout.validate_flow_config(&flow).unwrap();
        assert!(flow.parameter_count().unwrap() > 10_000_000);
        assert_eq!(flow.source_scale, 1.0e-3);
    }
}
