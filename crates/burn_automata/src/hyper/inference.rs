use serde::{Deserialize, Serialize};

use crate::{AutomataResult, NpaLowRankAdapter, NpaModel};

use super::{
    condition::{ConditionImage2d, ConditionSummary2d},
    hypernet::HyperNpa2d,
    prior::{ParticlePrior2d, ParticlePriorConfig},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConditionedNpa2d {
    pub summary: ConditionSummary2d,
    pub prior: ParticlePrior2d,
    pub adapter: NpaLowRankAdapter,
    pub model: NpaModel,
}

pub fn generate_conditioned_npa_2d(
    base_model: &NpaModel,
    hyper: &HyperNpa2d,
    condition: &ConditionImage2d,
    prior_config: ParticlePriorConfig,
) -> AutomataResult<ConditionedNpa2d> {
    base_model.validate()?;
    hyper.validate()?;
    if base_model.config != hyper.npa_config {
        return Err(crate::AutomataError::InvalidArgument(
            "hyper NPA config must match base model config".to_string(),
        ));
    }
    let summary = condition.summary()?;
    let prior = ParticlePrior2d::from_summary(&base_model.config, &summary, prior_config)?;
    let adapter = hyper.predict_adapter(condition)?;
    let model = adapter.apply_to_model(base_model)?;
    Ok(ConditionedNpa2d {
        summary,
        prior,
        adapter,
        model,
    })
}
