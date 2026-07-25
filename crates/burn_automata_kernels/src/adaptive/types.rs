#[derive(Clone, Debug, Default, PartialEq)]
pub struct AdaptiveGraphMetrics {
    /// Broad-phase source rows whose exact support was evaluated.
    pub candidate_visits: usize,
    pub raw_messages: usize,
    pub accepted_messages: usize,
    pub degree_mean: f32,
    pub degree_p95: usize,
    pub degree_max: usize,
    pub isolated_particles: usize,
    pub cross_scale_fraction: f32,
}

#[derive(Clone, Debug)]
pub struct AdaptivePerceptionOutput {
    pub features: Vec<f32>,
    pub normalized_state: Vec<f32>,
    pub state_gradient: Vec<f32>,
    pub occupancy_gradient: Vec<f32>,
    pub partition: Vec<f32>,
    /// Fraction of each row's compatible SPH density contributed by sources
    /// whose represented measure exceeds one native-resolution particle. A
    /// legacy configuration without `reference_measure` falls back to support.
    pub coarse_exposure: Vec<f32>,
    pub observed_spacing: Vec<f32>,
    pub moment_condition: Vec<f32>,
    pub moment_fallback: Vec<bool>,
    pub accepted_degree: Vec<usize>,
    pub graph: AdaptiveGraphMetrics,
    pub feature_dims: usize,
}

#[derive(Clone, Debug)]
pub struct AdaptivePerceptionPair {
    pub normalized: AdaptivePerceptionOutput,
    pub npa_compatible: AdaptivePerceptionOutput,
}
