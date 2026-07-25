use std::{
    cmp::Ordering,
    collections::HashMap,
    hash::{BuildHasherDefault, Hasher},
};

use rayon::prelude::*;

use super::{
    AdaptiveGraphMetrics, AdaptiveGraphPolicy, AdaptivePerceptionConfig, AdaptivePerceptionOutput,
    AdaptiveSupportBins,
};
use crate::{KernelError, KernelResult};

#[derive(Clone, Copy, Debug)]
pub(super) struct Candidate {
    pub(super) index: usize,
    pub(super) delta: [f32; 3],
    pub(super) distance2: f32,
    pub(super) pair_bandwidth: f32,
    normalized_distance2: f32,
}

#[derive(Clone, Debug)]
pub(super) struct ParticleNeighbors {
    pub(super) candidates: Vec<Candidate>,
    pub(super) candidate_visits: usize,
    pub(super) raw_count: usize,
    pub(super) observed_spacing: f32,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct CellKey {
    batch: usize,
    coordinates: [i64; 3],
}

#[derive(Debug)]
struct CellKeyHasher(u64);

impl Default for CellKeyHasher {
    fn default() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
}

impl Hasher for CellKeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

type CellMap = HashMap<CellKey, Vec<usize>, BuildHasherDefault<CellKeyHasher>>;

#[derive(Debug)]
struct SpatialIndex {
    cells: CellMap,
    cell_size: f32,
    dim: usize,
    particle_count: usize,
}

impl SpatialIndex {
    fn empty(particle_count: usize, dim: usize, cell_size: f32) -> Self {
        Self {
            cells: HashMap::with_capacity_and_hasher(particle_count, BuildHasherDefault::default()),
            cell_size,
            dim,
            particle_count,
        }
    }

    fn insert(&mut self, index: usize, position: &[f32; 4]) {
        self.cells
            .entry(Self::key(
                index / self.particle_count,
                position,
                self.dim,
                self.cell_size,
            ))
            .or_default()
            .push(index);
    }

    fn build(
        positions: &[[f32; 4]],
        particle_count: usize,
        dim: usize,
        cell_size: f32,
        include: impl Fn(usize) -> bool,
    ) -> Self {
        let mut spatial_index = Self::empty(particle_count, dim, cell_size);
        for (particle_index, position) in positions.iter().enumerate() {
            if !include(particle_index) {
                continue;
            }
            spatial_index.insert(particle_index, position);
        }
        spatial_index
    }

    fn candidates(&self, index: usize, positions: &[[f32; 4]], search_radius: f32) -> Vec<usize> {
        let center = Self::key(
            index / self.particle_count,
            &positions[index],
            self.dim,
            self.cell_size,
        );
        let mut candidates = Vec::new();
        let cell_radius = (search_radius / self.cell_size).ceil() as i64;
        let z_range = if self.dim == 3 {
            -cell_radius..=cell_radius
        } else {
            0..=0
        };
        for z in z_range {
            for y in -cell_radius..=cell_radius {
                for x in -cell_radius..=cell_radius {
                    let key = CellKey {
                        batch: center.batch,
                        coordinates: [
                            center.coordinates[0] + x,
                            center.coordinates[1] + y,
                            center.coordinates[2] + z,
                        ],
                    };
                    if let Some(cell) = self.cells.get(&key) {
                        candidates.extend_from_slice(cell);
                    }
                }
            }
        }
        // Preserve the all-pairs oracle's accumulation order and deterministic ties.
        candidates.sort_unstable();
        candidates
    }

    fn key(batch: usize, position: &[f32; 4], dim: usize, cell_size: f32) -> CellKey {
        let mut coordinates = [0_i64; 3];
        for axis in 0..dim {
            coordinates[axis] = (position[axis] / cell_size).floor() as i64;
        }
        CellKey { batch, coordinates }
    }
}

#[derive(Debug)]
struct SupportLevel {
    upper_bandwidth: f32,
    index: SpatialIndex,
}

#[derive(Debug)]
struct AdaptiveSpatialSearch {
    support_levels: Vec<SupportLevel>,
    spacing_index: Option<SpatialIndex>,
}

impl AdaptiveSpatialSearch {
    fn build(
        positions: &[[f32; 4]],
        bandwidth: &[f32],
        particle_count: usize,
        cfg: AdaptivePerceptionConfig,
        compute_spacing: bool,
    ) -> Self {
        let bins =
            AdaptiveSupportBins::new(cfg.min_bandwidth, cfg.max_bandwidth, cfg.support_bin_ratio)
                .expect("validated perception bandwidth bounds produce support bins");
        let mut support_levels = bins
            .upper_bounds()
            .iter()
            .copied()
            .map(|upper_bandwidth| SupportLevel {
                upper_bandwidth,
                index: SpatialIndex::empty(particle_count, cfg.dim, upper_bandwidth),
            })
            .collect::<Vec<_>>();
        for (index, position) in positions.iter().enumerate() {
            let level = support_levels
                .iter_mut()
                .find(|level| bandwidth[index] <= level.upper_bandwidth)
                .expect("validated bandwidth fits support levels");
            level.index.insert(index, position);
        }
        Self {
            support_levels,
            spacing_index: compute_spacing.then(|| {
                let spacing_cell_size = (cfg.max_bandwidth * 0.25).max(cfg.min_bandwidth);
                SpatialIndex::build(
                    positions,
                    particle_count,
                    cfg.dim,
                    spacing_cell_size,
                    |_| true,
                )
            }),
        }
    }

    fn candidates(
        &self,
        target: usize,
        positions: &[[f32; 4]],
        target_bandwidth: f32,
        cfg: AdaptivePerceptionConfig,
        compute_spacing: bool,
    ) -> (Vec<usize>, f32) {
        let mut candidates = Vec::new();
        for level in &self.support_levels {
            let radius = pair_bandwidth(
                target_bandwidth,
                level.upper_bandwidth,
                cfg.pair_scale_power,
            );
            candidates.extend(level.index.candidates(target, positions, radius));
        }

        if !compute_spacing {
            candidates.sort_unstable();
            candidates.dedup();
            return (candidates, target_bandwidth);
        }

        let spacing_index = self
            .spacing_index
            .as_ref()
            .expect("spacing index is built when spacing is requested");
        let mut spacing_radius = cfg.min_bandwidth;
        loop {
            let spacing_candidates = spacing_index.candidates(target, positions, spacing_radius);
            if smooth_occupancy(
                target,
                positions,
                &spacing_candidates,
                spacing_radius,
                cfg.dim,
            ) >= cfg.spacing_target_neighbors
                || spacing_radius >= cfg.max_bandwidth
            {
                candidates.extend(spacing_candidates);
                break;
            }
            spacing_radius = (spacing_radius * 2.0).min(cfg.max_bandwidth);
        }
        candidates.sort_unstable();
        candidates.dedup();
        (candidates, spacing_radius)
    }
}

#[derive(Clone, Copy)]
enum CandidateSearch<'a> {
    AllPairs,
    SpatialHash(&'a AdaptiveSpatialSearch),
}

#[allow(clippy::too_many_arguments)]
pub fn adaptive_perceive(
    positions: &[[f32; 4]],
    states: &[f32],
    represented_measure: &[f32],
    bandwidth: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    cfg: AdaptivePerceptionConfig,
) -> KernelResult<AdaptivePerceptionOutput> {
    adaptive_perceive_impl(
        positions,
        states,
        represented_measure,
        bandwidth,
        batch_size,
        particle_count,
        state_dims,
        cfg,
        true,
    )
}

/// Computes normalized adaptive perception without the auxiliary spacing root.
///
/// This is intended for transported fields whose spatial gradient is the only
/// required output. It reuses the production spatial search and graph policy,
/// while avoiding the independent spacing-neighborhood construction.
#[allow(clippy::too_many_arguments)]
pub fn adaptive_perceive_without_spacing(
    positions: &[[f32; 4]],
    states: &[f32],
    represented_measure: &[f32],
    bandwidth: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    cfg: AdaptivePerceptionConfig,
) -> KernelResult<AdaptivePerceptionOutput> {
    validate_inputs(
        positions,
        states,
        represented_measure,
        bandwidth,
        batch_size,
        particle_count,
        state_dims,
        cfg,
    )?;
    let neighborhoods =
        build_neighborhoods_without_spacing(positions, bandwidth, particle_count, cfg);
    Ok(perceive_from_neighborhoods(
        positions,
        states,
        represented_measure,
        bandwidth,
        state_dims,
        cfg,
        &neighborhoods,
    ))
}

/// Deterministic all-pairs oracle for parity tests and performance comparisons.
/// Production callers should use [`adaptive_perceive`].
#[allow(clippy::too_many_arguments)]
pub fn adaptive_perceive_all_pairs(
    positions: &[[f32; 4]],
    states: &[f32],
    represented_measure: &[f32],
    bandwidth: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    cfg: AdaptivePerceptionConfig,
) -> KernelResult<AdaptivePerceptionOutput> {
    adaptive_perceive_impl(
        positions,
        states,
        represented_measure,
        bandwidth,
        batch_size,
        particle_count,
        state_dims,
        cfg,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn adaptive_perceive_impl(
    positions: &[[f32; 4]],
    states: &[f32],
    represented_measure: &[f32],
    bandwidth: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    cfg: AdaptivePerceptionConfig,
    spatial_hash: bool,
) -> KernelResult<AdaptivePerceptionOutput> {
    validate_inputs(
        positions,
        states,
        represented_measure,
        bandwidth,
        batch_size,
        particle_count,
        state_dims,
        cfg,
    )?;
    let neighborhoods =
        build_neighborhoods(positions, bandwidth, particle_count, cfg, spatial_hash);
    Ok(perceive_from_neighborhoods(
        positions,
        states,
        represented_measure,
        bandwidth,
        state_dims,
        cfg,
        &neighborhoods,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn perceive_from_neighborhoods(
    positions: &[[f32; 4]],
    states: &[f32],
    represented_measure: &[f32],
    bandwidth: &[f32],
    state_dims: usize,
    cfg: AdaptivePerceptionConfig,
    neighborhoods: &[ParticleNeighbors],
) -> AdaptivePerceptionOutput {
    let total = positions.len();
    let feature_dims = cfg.feature_dims(state_dims);
    let mut features = vec![0.0; total * feature_dims];
    let mut normalized_state = vec![0.0; total * state_dims];
    let mut state_gradient = vec![0.0; total * state_dims * cfg.dim];
    let mut occupancy_gradient = vec![0.0; total * cfg.dim];
    let mut partition = vec![0.0; total];
    let mut observed_spacing = Vec::with_capacity(total);
    let mut moment_condition = vec![0.0; total];
    let mut moment_fallback = vec![false; total];
    let mut accepted_degree = Vec::with_capacity(total);
    features
        .par_chunks_mut(feature_dims)
        .zip(normalized_state.par_chunks_mut(state_dims))
        .zip(state_gradient.par_chunks_mut(state_dims * cfg.dim))
        .zip(occupancy_gradient.par_chunks_mut(cfg.dim))
        .zip(partition.par_iter_mut())
        .zip(moment_condition.par_iter_mut())
        .zip(moment_fallback.par_iter_mut())
        .enumerate()
        .for_each(
            |(
                index,
                (
                    (((((features, normalized), gradient), occupancy), partition), condition),
                    fallback,
                ),
            )| {
                perceive_particle_into(
                    index,
                    positions,
                    states,
                    represented_measure,
                    bandwidth,
                    state_dims,
                    &neighborhoods[index],
                    cfg,
                    features,
                    normalized,
                    gradient,
                    occupancy,
                    partition,
                    condition,
                    fallback,
                );
            },
        );
    for neighborhood in neighborhoods {
        observed_spacing.push(neighborhood.observed_spacing);
        accepted_degree.push(neighborhood.candidates.len());
    }

    let candidate_visits = neighborhoods.iter().map(|row| row.candidate_visits).sum();
    let raw_messages = neighborhoods.iter().map(|row| row.raw_count).sum();
    let accepted_messages = accepted_degree.iter().sum();
    let mut sorted_degree = accepted_degree.clone();
    sorted_degree.sort_unstable();
    let degree_p95 = percentile_usize(&sorted_degree, 0.95);
    let degree_max = sorted_degree.last().copied().unwrap_or_default();
    let isolated_particles = accepted_degree
        .iter()
        .filter(|degree| **degree == 0)
        .count();
    let cross_scale_messages = neighborhoods
        .iter()
        .enumerate()
        .map(|(index, row)| {
            row.candidates
                .iter()
                .filter(|candidate| {
                    let ratio = bandwidth[index] / bandwidth[candidate.index];
                    !(0.5..=2.0).contains(&ratio)
                })
                .count()
        })
        .sum::<usize>();
    let graph = AdaptiveGraphMetrics {
        candidate_visits,
        raw_messages,
        accepted_messages,
        degree_mean: accepted_messages as f32 / total.max(1) as f32,
        degree_p95,
        degree_max,
        isolated_particles,
        cross_scale_fraction: cross_scale_messages as f32 / accepted_messages.max(1) as f32,
    };

    AdaptivePerceptionOutput {
        features,
        normalized_state,
        state_gradient,
        occupancy_gradient,
        partition,
        coarse_exposure: vec![0.0; total],
        observed_spacing,
        moment_condition,
        moment_fallback,
        accepted_degree,
        graph,
        feature_dims,
    }
}

pub(super) fn build_neighborhoods(
    positions: &[[f32; 4]],
    bandwidth: &[f32],
    particle_count: usize,
    cfg: AdaptivePerceptionConfig,
    spatial_hash: bool,
) -> Vec<ParticleNeighbors> {
    build_neighborhoods_impl(
        positions,
        bandwidth,
        particle_count,
        cfg,
        spatial_hash,
        true,
    )
}

pub(super) fn build_neighborhoods_without_spacing(
    positions: &[[f32; 4]],
    bandwidth: &[f32],
    particle_count: usize,
    cfg: AdaptivePerceptionConfig,
) -> Vec<ParticleNeighbors> {
    build_neighborhoods_impl(positions, bandwidth, particle_count, cfg, true, false)
}

fn build_neighborhoods_impl(
    positions: &[[f32; 4]],
    bandwidth: &[f32],
    particle_count: usize,
    cfg: AdaptivePerceptionConfig,
    spatial_hash: bool,
    compute_spacing: bool,
) -> Vec<ParticleNeighbors> {
    let mut raw_config = cfg;
    raw_config.graph_policy = AdaptiveGraphPolicy::RawSupport;
    let spatial_index = spatial_hash.then(|| {
        AdaptiveSpatialSearch::build(
            positions,
            bandwidth,
            particle_count,
            raw_config,
            compute_spacing,
        )
    });
    let search = spatial_index
        .as_ref()
        .map_or(CandidateSearch::AllPairs, CandidateSearch::SpatialHash);
    let mut neighborhoods = (0..positions.len())
        .into_par_iter()
        .map(|index| {
            build_particle_neighbors(
                index,
                positions,
                bandwidth,
                particle_count,
                raw_config,
                search,
                compute_spacing,
            )
        })
        .collect::<Vec<_>>();
    apply_graph_policy(&mut neighborhoods, cfg);
    neighborhoods
}

pub(super) fn apply_graph_policy(
    neighborhoods: &mut [ParticleNeighbors],
    cfg: AdaptivePerceptionConfig,
) {
    if cfg.graph_policy != AdaptiveGraphPolicy::RawSupport {
        neighborhoods.par_iter_mut().for_each(|neighbors| {
            neighbors.candidates.truncate(cfg.max_neighbors);
            if cfg.graph_policy == AdaptiveGraphPolicy::MutualTopK {
                neighbors
                    .candidates
                    .sort_unstable_by_key(|candidate| candidate.index);
            }
        });
    }
    if cfg.graph_policy == AdaptiveGraphPolicy::MutualTopK {
        let directed = neighborhoods
            .iter()
            .map(|neighbors| {
                neighbors
                    .candidates
                    .iter()
                    .map(|candidate| candidate.index)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        neighborhoods
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, neighbors)| {
                neighbors
                    .candidates
                    .retain(|candidate| directed[candidate.index].binary_search(&index).is_ok());
            });
    }
}

#[allow(clippy::too_many_arguments)]
fn perceive_particle_into(
    index: usize,
    positions: &[[f32; 4]],
    states: &[f32],
    represented_measure: &[f32],
    bandwidth: &[f32],
    state_dims: usize,
    neighbors: &ParticleNeighbors,
    cfg: AdaptivePerceptionConfig,
    features: &mut [f32],
    normalized_state: &mut [f32],
    state_gradient: &mut [f32],
    occupancy_gradient: &mut [f32],
    partition: &mut f32,
    moment_condition: &mut f32,
    moment_fallback: &mut bool,
) {
    let state_base = index * state_dims;
    let state_i = &states[state_base..state_base + state_dims];
    let self_h = bandwidth[index];
    let self_kernel = kernel_value(0.0, self_h, cfg.dim);
    let mut denominator = represented_measure[index] * self_kernel + cfg.shepard_epsilon;
    for (output, value) in normalized_state.iter_mut().zip(state_i) {
        *output = cfg.shepard_epsilon * value + represented_measure[index] * self_kernel * value;
    }
    state_gradient.fill(0.0);
    occupancy_gradient.fill(0.0);
    let mut moment = [0.0_f32; 9];

    for candidate in &neighbors.candidates {
        let j = candidate.index;
        let weight = represented_measure[j]
            * kernel_value(candidate.distance2, candidate.pair_bandwidth, cfg.dim);
        denominator += weight;
        let source = j * state_dims;
        for channel in 0..state_dims {
            normalized_state[channel] += weight * states[source + channel];
        }

        let kernel_gradient = kernel_gradient(
            candidate.delta,
            candidate.distance2,
            candidate.pair_bandwidth,
            cfg.dim,
        );
        for axis in 0..cfg.dim {
            let weighted_gradient = represented_measure[j] * kernel_gradient[axis];
            occupancy_gradient[axis] += weighted_gradient;
            for col in 0..cfg.dim {
                moment[axis * cfg.dim + col] += weighted_gradient * candidate.delta[col];
            }
            for channel in 0..state_dims {
                let difference = states[source + channel] - state_i[channel];
                state_gradient[channel * cfg.dim + axis] += difference * weighted_gradient;
            }
        }
    }

    *partition = denominator;
    let denominator_inverse = denominator.recip();
    normalized_state.iter_mut().for_each(|value| {
        *value *= denominator_inverse;
        if !value.is_finite() {
            *value = 0.0;
        }
    });
    for value in occupancy_gradient.iter_mut() {
        *value *= denominator_inverse;
    }

    let (inverse, condition, fallback) = regularized_inverse(moment, cfg);
    *moment_condition = condition;
    *moment_fallback = fallback;
    for channel in 0..state_dims {
        let mut rhs = [0.0_f32; 3];
        rhs[..cfg.dim].copy_from_slice(&state_gradient[channel * cfg.dim..(channel + 1) * cfg.dim]);
        for out_axis in 0..cfg.dim {
            state_gradient[channel * cfg.dim + out_axis] = 0.0;
            for in_axis in 0..cfg.dim {
                state_gradient[channel * cfg.dim + out_axis] +=
                    inverse[out_axis * cfg.dim + in_axis] * rhs[in_axis];
            }
        }
    }

    for channel in 0..state_dims {
        let row = &mut state_gradient[channel * cfg.dim..(channel + 1) * cfg.dim];
        for value in row.iter_mut() {
            *value *= self_h;
        }
        if cfg.log_normalize_gradients {
            log_normalize(row);
        }
    }
    for value in occupancy_gradient.iter_mut() {
        *value *= self_h;
    }
    if cfg.log_normalize_gradients {
        log_normalize(occupancy_gradient);
    }

    let mut cursor = 0;
    features[cursor..cursor + state_dims].copy_from_slice(state_i);
    cursor += state_dims;
    features[cursor..cursor + state_dims].copy_from_slice(normalized_state);
    cursor += state_dims;
    features[cursor..cursor + state_gradient.len()].copy_from_slice(state_gradient);
    cursor += state_gradient.len();
    features[cursor..cursor + occupancy_gradient.len()].copy_from_slice(occupancy_gradient);
    cursor += occupancy_gradient.len();
    if cfg.include_position_features {
        features[cursor..cursor + cfg.dim].copy_from_slice(&positions[index][..cfg.dim]);
    }
}

fn build_particle_neighbors(
    index: usize,
    positions: &[[f32; 4]],
    bandwidth: &[f32],
    particle_count: usize,
    cfg: AdaptivePerceptionConfig,
    search: CandidateSearch<'_>,
    compute_spacing: bool,
) -> ParticleNeighbors {
    let batch = index / particle_count;
    let batch_start = batch * particle_count;
    let batch_end = batch_start + particle_count;
    let mut support_candidates = Vec::new();
    let mut spacing_distances = Vec::new();
    let mut candidate_visits = 0;
    let (spatial_candidates, spacing_radius) = match search {
        CandidateSearch::AllPairs => (None, cfg.max_bandwidth),
        CandidateSearch::SpatialHash(spatial_index) => {
            let (candidates, radius) =
                spatial_index.candidates(index, positions, bandwidth[index], cfg, compute_spacing);
            (Some(candidates), radius)
        }
    };
    let mut visit = |j: usize| {
        candidate_visits += 1;
        if j == index {
            return;
        }
        let mut delta = [0.0_f32; 3];
        let mut distance2 = 0.0;
        for axis in 0..cfg.dim {
            delta[axis] = positions[j][axis] - positions[index][axis];
            distance2 += delta[axis] * delta[axis];
        }
        if compute_spacing && distance2 < spacing_radius * spacing_radius {
            spacing_distances.push(distance2.sqrt());
        }
        let pair_bandwidth = pair_bandwidth(bandwidth[index], bandwidth[j], cfg.pair_scale_power);
        if distance2 < pair_bandwidth * pair_bandwidth {
            support_candidates.push(Candidate {
                index: j,
                delta,
                distance2,
                pair_bandwidth,
                normalized_distance2: distance2 / (pair_bandwidth * pair_bandwidth),
            });
        }
    };
    match search {
        CandidateSearch::AllPairs => {
            for j in batch_start..batch_end {
                visit(j);
            }
        }
        CandidateSearch::SpatialHash(_) => {
            for j in spatial_candidates.expect("spatial candidates built") {
                visit(j);
            }
        }
    }
    let raw_count = support_candidates.len();
    if cfg.graph_policy != AdaptiveGraphPolicy::RawSupport {
        if support_candidates.len() > cfg.max_neighbors {
            support_candidates.select_nth_unstable_by(cfg.max_neighbors, candidate_order);
            support_candidates.truncate(cfg.max_neighbors);
        }
        support_candidates.sort_by(candidate_order);
        // Mutual filtering uses binary search on index lists after this function.
        if cfg.graph_policy == AdaptiveGraphPolicy::MutualTopK {
            support_candidates.sort_unstable_by_key(|candidate| candidate.index);
        }
    } else {
        support_candidates.sort_by(candidate_order);
    }
    let observed_spacing = if compute_spacing {
        spacing_root(&spacing_distances, cfg)
    } else {
        bandwidth[index]
    };
    ParticleNeighbors {
        candidates: support_candidates,
        candidate_visits,
        raw_count,
        observed_spacing,
    }
}

fn smooth_occupancy(
    target: usize,
    positions: &[[f32; 4]],
    candidates: &[usize],
    radius: f32,
    dim: usize,
) -> f32 {
    let radius2 = radius * radius;
    candidates
        .iter()
        .filter_map(|index| {
            if *index == target {
                return None;
            }
            let distance2 = (0..dim)
                .map(|axis| {
                    let delta = positions[*index][axis] - positions[target][axis];
                    delta * delta
                })
                .sum::<f32>();
            if distance2 >= radius2 {
                return None;
            }
            let shoulder = 1.0 - distance2 / radius2;
            Some(shoulder * shoulder * shoulder)
        })
        .sum()
}

fn candidate_order(lhs: &Candidate, rhs: &Candidate) -> Ordering {
    lhs.normalized_distance2
        .total_cmp(&rhs.normalized_distance2)
        .then_with(|| lhs.index.cmp(&rhs.index))
}

pub(super) fn pair_bandwidth(lhs: f32, rhs: f32, power: f32) -> f32 {
    ((lhs.powf(power) + rhs.powf(power)) * 0.5).powf(power.recip())
}

pub(super) fn kernel_value(distance2: f32, bandwidth: f32, dim: usize) -> f32 {
    let q2 = distance2 / (bandwidth * bandwidth);
    if q2 >= 1.0 {
        return 0.0;
    }
    let shoulder = 1.0 - q2;
    shoulder * shoulder * shoulder / bandwidth.powi(dim as i32)
}

pub(super) fn kernel_gradient(
    delta: [f32; 3],
    distance2: f32,
    bandwidth: f32,
    dim: usize,
) -> [f32; 3] {
    let q2 = distance2 / (bandwidth * bandwidth);
    if q2 <= 0.0 || q2 >= 1.0 {
        return [0.0; 3];
    }
    let shoulder = 1.0 - q2;
    let scale = -6.0 * shoulder * shoulder / bandwidth.powi(dim as i32 + 2);
    let mut gradient = [0.0; 3];
    for axis in 0..dim {
        gradient[axis] = scale * delta[axis];
    }
    gradient
}

fn spacing_root(distances: &[f32], cfg: AdaptivePerceptionConfig) -> f32 {
    let occupancy = |radius: f32| {
        distances
            .iter()
            .filter(|distance| **distance < radius)
            .map(|distance| {
                let q = distance / radius;
                let shoulder = 1.0 - q * q;
                shoulder * shoulder * shoulder
            })
            .sum::<f32>()
    };
    if occupancy(cfg.max_bandwidth) < cfg.spacing_target_neighbors {
        return cfg.max_bandwidth;
    }
    let mut lo = cfg.min_bandwidth;
    let mut hi = cfg.max_bandwidth;
    for _ in 0..cfg.spacing_root_iterations {
        let mid = 0.5 * (lo + hi);
        if occupancy(mid) < cfg.spacing_target_neighbors {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

pub(super) fn regularized_inverse(
    mut matrix: [f32; 9],
    cfg: AdaptivePerceptionConfig,
) -> ([f32; 9], f32, bool) {
    let trace = (0..cfg.dim)
        .map(|axis| matrix[axis * cfg.dim + axis].abs())
        .sum::<f32>();
    let diagonal = cfg.moment_regularization * (trace / cfg.dim as f32).max(1.0e-8);
    for axis in 0..cfg.dim {
        matrix[axis * cfg.dim + axis] += diagonal.copysign(matrix[axis * cfg.dim + axis]);
    }
    let inverse = inverse_matrix(matrix, cfg.dim);
    let condition = inverse.map_or(f32::INFINITY, |value| {
        frobenius_norm(&matrix, cfg.dim) * frobenius_norm(&value, cfg.dim)
    });
    if let Some(inverse) = inverse
        && condition.is_finite()
        && condition <= cfg.moment_condition_limit
    {
        return (inverse, condition, false);
    }

    let mut fallback = [0.0; 9];
    let scale = (trace / cfg.dim as f32).max(1.0e-6).recip();
    for axis in 0..cfg.dim {
        fallback[axis * cfg.dim + axis] = scale;
    }
    (fallback, condition, true)
}

fn inverse_matrix(matrix: [f32; 9], dim: usize) -> Option<[f32; 9]> {
    let mut out = [0.0; 9];
    if dim == 2 {
        let determinant = matrix[0] * matrix[3] - matrix[1] * matrix[2];
        if !determinant.is_finite() || determinant.abs() < 1.0e-12 {
            return None;
        }
        let reciprocal = determinant.recip();
        out[0] = matrix[3] * reciprocal;
        out[1] = -matrix[1] * reciprocal;
        out[2] = -matrix[2] * reciprocal;
        out[3] = matrix[0] * reciprocal;
        return Some(out);
    }

    let determinant = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
        - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
        + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
    if !determinant.is_finite() || determinant.abs() < 1.0e-12 {
        return None;
    }
    let reciprocal = determinant.recip();
    out[0] = (matrix[4] * matrix[8] - matrix[5] * matrix[7]) * reciprocal;
    out[1] = (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * reciprocal;
    out[2] = (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * reciprocal;
    out[3] = (matrix[5] * matrix[6] - matrix[3] * matrix[8]) * reciprocal;
    out[4] = (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * reciprocal;
    out[5] = (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * reciprocal;
    out[6] = (matrix[3] * matrix[7] - matrix[4] * matrix[6]) * reciprocal;
    out[7] = (matrix[1] * matrix[6] - matrix[0] * matrix[7]) * reciprocal;
    out[8] = (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * reciprocal;
    Some(out)
}

fn frobenius_norm(matrix: &[f32; 9], dim: usize) -> f32 {
    let mut sum = 0.0;
    for row in 0..dim {
        for col in 0..dim {
            let value = matrix[row * dim + col];
            sum += value * value;
        }
    }
    sum.sqrt()
}

pub(super) fn log_normalize(values: &mut [f32]) {
    let norm = (values.iter().map(|value| value * value).sum::<f32>() + 1.0e-12).sqrt();
    let scale = norm.ln_1p() / norm.max(1.0e-6);
    for value in values {
        *value *= scale;
    }
}

pub(super) fn percentile_usize(sorted: &[usize], quantile: f32) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let index = ((sorted.len() - 1) as f32 * quantile).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_inputs(
    positions: &[[f32; 4]],
    states: &[f32],
    represented_measure: &[f32],
    bandwidth: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    cfg: AdaptivePerceptionConfig,
) -> KernelResult<()> {
    cfg.validate()?;
    if batch_size == 0 || particle_count == 0 || state_dims == 0 {
        return Err(KernelError::InvalidArgument(
            "adaptive perception dimensions must be non-zero".to_string(),
        ));
    }
    let total = batch_size.checked_mul(particle_count).ok_or_else(|| {
        KernelError::InvalidArgument("adaptive particle count overflow".to_string())
    })?;
    if positions.len() != total {
        return Err(KernelError::PositionShape {
            positions: positions.len(),
            expected: total,
        });
    }
    if states.len() != total * state_dims {
        return Err(KernelError::StateShape {
            states: states.len(),
            expected: total * state_dims,
        });
    }
    if represented_measure.len() != total || bandwidth.len() != total {
        return Err(KernelError::InvalidArgument(format!(
            "adaptive measure/bandwidth lengths must equal {total}, got {}/{}",
            represented_measure.len(),
            bandwidth.len()
        )));
    }
    if positions
        .iter()
        .flat_map(|position| position.iter().take(cfg.dim))
        .any(|value| !value.is_finite())
        || states.iter().any(|value| !value.is_finite())
        || represented_measure
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || bandwidth.iter().any(|value| {
            !value.is_finite() || *value < cfg.min_bandwidth || *value > cfg.max_bandwidth
        })
    {
        return Err(KernelError::InvalidArgument(
            "adaptive inputs must be finite with positive measure and in-range bandwidth"
                .to_string(),
        ));
    }
    Ok(())
}
