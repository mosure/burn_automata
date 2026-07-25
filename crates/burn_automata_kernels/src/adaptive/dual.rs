use rayon::prelude::*;

use super::{
    AdaptiveGraphPolicy, AdaptiveNpaPerceptionOptions, AdaptivePerceptionConfig,
    AdaptivePerceptionPair,
    compatible::perceive_from_neighborhoods as perceive_npa_from_neighborhoods,
    perception::{
        ParticleNeighbors, apply_graph_policy, build_neighborhoods,
        perceive_from_neighborhoods as perceive_normalized_from_neighborhoods, validate_inputs,
    },
};
use crate::KernelResult;

/// Computes normalized-paper and NPA-compatible features from one spatial
/// search. Each consumer may apply a different hard graph policy without
/// rebuilding the candidate cells.
#[allow(clippy::too_many_arguments)]
pub fn adaptive_perceive_pair(
    positions: &[[f32; 4]],
    states: &[f32],
    represented_measure: &[f32],
    bandwidth: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    normalized_config: AdaptivePerceptionConfig,
    npa_graph_policy: AdaptiveGraphPolicy,
    npa_options: AdaptiveNpaPerceptionOptions,
) -> KernelResult<AdaptivePerceptionPair> {
    validate_inputs(
        positions,
        states,
        represented_measure,
        bandwidth,
        batch_size,
        particle_count,
        state_dims,
        normalized_config,
    )?;
    npa_options.validate()?;
    let mut raw_config = normalized_config;
    raw_config.graph_policy = AdaptiveGraphPolicy::RawSupport;
    let raw_neighborhoods =
        build_neighborhoods(positions, bandwidth, particle_count, raw_config, true);
    let mut normalized_neighborhoods = raw_neighborhoods
        .par_iter()
        .map(|neighbors| ParticleNeighbors {
            candidates: if normalized_config.graph_policy == AdaptiveGraphPolicy::RawSupport {
                neighbors.candidates.clone()
            } else {
                neighbors
                    .candidates
                    .iter()
                    .take(normalized_config.max_neighbors)
                    .copied()
                    .collect()
            },
            candidate_visits: neighbors.candidate_visits,
            raw_count: neighbors.raw_count,
            observed_spacing: neighbors.observed_spacing,
        })
        .collect::<Vec<_>>();
    apply_graph_policy(&mut normalized_neighborhoods, normalized_config);
    let mut npa_neighborhoods = raw_neighborhoods;
    let mut npa_config = normalized_config;
    npa_config.graph_policy = npa_graph_policy;
    apply_graph_policy(&mut npa_neighborhoods, npa_config);
    Ok(AdaptivePerceptionPair {
        normalized: perceive_normalized_from_neighborhoods(
            positions,
            states,
            represented_measure,
            bandwidth,
            state_dims,
            normalized_config,
            &normalized_neighborhoods,
        ),
        npa_compatible: perceive_npa_from_neighborhoods(
            positions,
            states,
            represented_measure,
            bandwidth,
            particle_count,
            state_dims,
            npa_config,
            npa_options,
            &npa_neighborhoods,
        ),
    })
}
