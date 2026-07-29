//! Device-resident VJP bridge between truncated rollouts and row-flow endpoints.

use super::*;

/// Keeps one conditional-flow graph alive while rollout chunks backpropagate
/// through a small endpoint leaf. The accumulated endpoint VJP is contracted
/// through the flow once after all chunks complete.
pub(super) struct BurnRowFlowEndpointBridge {
    endpoint_rows: Tensor3,
    generated_rows: Tensor3,
    prepared_condition: Option<BurnRowFlowCondition>,
    rollout_rows: Tensor3,
    rollout_rows_expanded: bool,
    accumulated_gradient: Option<Tensor3Inner>,
}

impl BurnRowFlowEndpointBridge {
    pub(super) fn new(generated_rows: Tensor3) -> Self {
        let rollout_rows = track3(generated_rows.clone().inner());
        Self {
            endpoint_rows: generated_rows.clone(),
            generated_rows,
            prepared_condition: None,
            rollout_rows,
            rollout_rows_expanded: false,
            accumulated_gradient: None,
        }
    }

    pub(super) fn with_prepared_condition(
        generated_rows: Tensor3,
        prepared_condition: BurnRowFlowCondition,
    ) -> Self {
        let mut bridge = Self::new(generated_rows);
        bridge.prepared_condition = Some(prepared_condition);
        bridge
    }

    pub(super) fn with_mixed_endpoint(
        generated_rows: Tensor3,
        endpoint_rows: Tensor3,
        prepared_condition: BurnRowFlowCondition,
    ) -> Self {
        let rollout_rows = track3(endpoint_rows.clone().inner());
        Self {
            endpoint_rows,
            generated_rows,
            prepared_condition: Some(prepared_condition),
            rollout_rows,
            rollout_rows_expanded: true,
            accumulated_gradient: None,
        }
    }

    pub(super) fn generated_rows(&self) -> Tensor3 {
        self.generated_rows.clone()
    }

    pub(super) fn prepared_condition(&self) -> Option<BurnRowFlowCondition> {
        self.prepared_condition.clone()
    }

    pub(super) fn adapter_batch(
        &self,
        npa_config: &NpaConfig,
        expansion: Option<&[usize]>,
    ) -> BurnAdapterBatch {
        BurnAdapterBatch::from_dense_residual_rows(self.rollout_rows.clone(), npa_config)
            .select_rows_or_identity((!self.rollout_rows_expanded).then_some(expansion).flatten())
    }

    pub(super) fn detached_adapter_batch(
        &self,
        npa_config: &NpaConfig,
        expansion: Option<&[usize]>,
    ) -> BurnAdapterBatch {
        BurnAdapterBatch::from_dense_residual_rows(
            detach3(self.rollout_rows.clone()),
            npa_config,
        )
        .select_rows_or_identity((!self.rollout_rows_expanded).then_some(expansion).flatten())
    }

    pub(super) fn accumulate(
        &mut self,
        grads: &mut <BurnBackend as burn::tensor::backend::AutodiffBackend>::Gradients,
    ) {
        let gradient = self
            .rollout_rows
            .grad_remove(grads)
            .unwrap_or_else(|| self.rollout_rows.clone().inner().zeros_like());
        self.accumulated_gradient = Some(match self.accumulated_gradient.take() {
            Some(accumulated) => accumulated + gradient,
            None => gradient,
        });
    }

    pub(super) fn objective(
        &self,
        gradient_scale: f32,
        normalization: Option<PackedNpaGradientLayout>,
    ) -> Option<Tensor1> {
        let gradient = self.accumulated_gradient.clone()?;
        let gradient = gradient.mul_scalar(gradient_scale);
        let gradient = normalization.map_or(gradient.clone(), |layout| {
            normalize_packed_npa_endpoint_gradient(gradient, layout)
        });
        let gradient = Tensor::<BurnBackend, 3>::from_inner(gradient);
        Some(self.endpoint_rows.clone().mul(gradient).sum())
    }
}

pub(super) fn row_flow_endpoint_bridge_enabled(
    generator: &BurnE2eGeneratorParams,
    config: BurnE2eRolloutTrainConfig,
) -> bool {
    generator.row_flow.is_some()
        && !config.spatial_condition_control
        && !config.amortization_substrate_only
}

pub(super) const fn endpoint_bridge_normalizes_gradient(
    per_parameter_grad_normalization: bool,
    amortization_substrate_only: bool,
) -> bool {
    per_parameter_grad_normalization && !amortization_substrate_only
}
