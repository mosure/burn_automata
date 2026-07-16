use rand::{Rng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct E2eIdentitySampler {
    len: usize,
    batch_size: usize,
    uniform_fraction: f32,
    priority_ema_beta: f32,
    priority_min_weight: f32,
    priority_max_weight: f32,
    uniform_order: Vec<usize>,
    uniform_cursor: usize,
    trajectory_counts: Vec<u64>,
    loss_ema: Vec<f32>,
    loss_observed: Vec<bool>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct E2eExposureStats {
    pub(crate) total: u64,
    pub(crate) min: u64,
    pub(crate) p10: u64,
    pub(crate) median: u64,
    pub(crate) p90: u64,
    pub(crate) max: u64,
    pub(crate) mean: f64,
}

impl E2eIdentitySampler {
    pub(crate) fn new<R: Rng + ?Sized>(
        len: usize,
        requested_batch_size: usize,
        uniform_fraction: f32,
        priority_ema_beta: f32,
        priority_min_weight: f32,
        priority_max_weight: f32,
        rng: &mut R,
    ) -> Self {
        let batch_size = requested_batch_size.max(1).min(len.max(1));
        let mut uniform_order = (0..len).collect::<Vec<_>>();
        uniform_order.shuffle(rng);
        Self {
            len,
            batch_size,
            uniform_fraction: uniform_fraction.clamp(0.0, 1.0),
            priority_ema_beta: priority_ema_beta.clamp(0.0, 0.999_999),
            priority_min_weight: priority_min_weight.max(f32::MIN_POSITIVE),
            priority_max_weight: priority_max_weight.max(priority_min_weight),
            uniform_order,
            uniform_cursor: 0,
            trajectory_counts: vec![0; len],
            loss_ema: vec![0.0; len],
            loss_observed: vec![false; len],
        }
    }

    pub(crate) fn next_batch<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Vec<usize> {
        if self.len == 0 {
            return Vec::new();
        }
        if self.batch_size >= self.len {
            let mut all = (0..self.len).collect::<Vec<_>>();
            all.shuffle(rng);
            return all;
        }

        let uniform_count = ((self.batch_size as f32 * self.uniform_fraction).ceil() as usize)
            .clamp(1, self.batch_size);
        let mut selected = Vec::with_capacity(self.batch_size);
        while selected.len() < uniform_count {
            if self.uniform_cursor >= self.uniform_order.len() {
                self.uniform_order = (0..self.len).collect();
                self.uniform_order.shuffle(rng);
                self.uniform_cursor = 0;
            }
            let index = self.uniform_order[self.uniform_cursor];
            self.uniform_cursor += 1;
            if !selected.contains(&index) {
                selected.push(index);
            }
        }

        let observed_count = self
            .loss_observed
            .iter()
            .filter(|observed| **observed)
            .count();
        let observed_mean = self
            .loss_ema
            .iter()
            .zip(&self.loss_observed)
            .filter_map(|(loss, observed)| observed.then_some(*loss))
            .sum::<f32>()
            / observed_count.max(1) as f32;
        let min_exposure = self.trajectory_counts.iter().copied().min().unwrap_or(0);
        while selected.len() < self.batch_size {
            let candidates = (0..self.len)
                .filter(|index| !selected.contains(index))
                .collect::<Vec<_>>();
            let weights = candidates
                .iter()
                .map(|index| self.priority_weight(*index, observed_mean, min_exposure))
                .collect::<Vec<_>>();
            let total = weights.iter().sum::<f32>();
            let mut draw = rng.random::<f32>() * total.max(f32::MIN_POSITIVE);
            let mut chosen = *candidates.last().expect("non-empty priority candidates");
            for (index, weight) in candidates.into_iter().zip(weights) {
                draw -= weight;
                if draw <= 0.0 {
                    chosen = index;
                    break;
                }
            }
            selected.push(chosen);
        }
        selected
    }

    pub(crate) fn record_trajectories(&mut self, identities: &[usize], replicas: usize) {
        let replicas = replicas.max(1) as u64;
        for &identity in identities {
            if let Some(count) = self.trajectory_counts.get_mut(identity) {
                *count = count.saturating_add(replicas);
            }
        }
    }

    pub(crate) fn update_losses(&mut self, identities: &[usize], rollout_losses: &[f32]) {
        if identities.len() != rollout_losses.len() {
            return;
        }
        let mut sums = vec![0.0_f32; self.len];
        let mut counts = vec![0usize; self.len];
        for (&identity, &loss) in identities.iter().zip(rollout_losses) {
            if identity < self.len && loss.is_finite() {
                sums[identity] += loss;
                counts[identity] += 1;
            }
        }
        for identity in 0..self.len {
            if counts[identity] == 0 {
                continue;
            }
            let value = sums[identity] / counts[identity] as f32;
            if self.loss_observed[identity] {
                self.loss_ema[identity] = self.priority_ema_beta * self.loss_ema[identity]
                    + (1.0 - self.priority_ema_beta) * value;
            } else {
                self.loss_ema[identity] = value;
                self.loss_observed[identity] = true;
            }
        }
    }

    pub(crate) fn exposure_stats(&self) -> E2eExposureStats {
        if self.trajectory_counts.is_empty() {
            return E2eExposureStats::default();
        }
        let mut sorted = self.trajectory_counts.clone();
        sorted.sort_unstable();
        let total = sorted.iter().copied().sum::<u64>();
        let percentile = |numerator: usize| {
            let index = (sorted.len().saturating_sub(1) * numerator) / 10;
            sorted[index]
        };
        E2eExposureStats {
            total,
            min: sorted[0],
            p10: percentile(1),
            median: percentile(5),
            p90: percentile(9),
            max: *sorted.last().expect("non-empty exposure counts"),
            mean: total as f64 / sorted.len() as f64,
        }
    }

    pub(crate) fn is_compatible(
        &self,
        len: usize,
        requested_batch_size: usize,
        uniform_fraction: f32,
        priority_ema_beta: f32,
        priority_min_weight: f32,
        priority_max_weight: f32,
    ) -> bool {
        let batch_size = requested_batch_size.max(1).min(len.max(1));
        self.len == len
            && self.batch_size == batch_size
            && (self.uniform_fraction - uniform_fraction.clamp(0.0, 1.0)).abs() <= f32::EPSILON
            && (self.priority_ema_beta - priority_ema_beta.clamp(0.0, 0.999_999)).abs()
                <= f32::EPSILON
            && (self.priority_min_weight - priority_min_weight.max(f32::MIN_POSITIVE)).abs()
                <= f32::EPSILON
            && (self.priority_max_weight - priority_max_weight.max(priority_min_weight)).abs()
                <= f32::EPSILON
    }

    pub(crate) fn ensure_minimum_trajectory_counts(
        &mut self,
        identity_optimizer_steps: &[usize],
        replicas: usize,
    ) {
        let replicas = replicas.max(1) as u64;
        for (identity, count) in self.trajectory_counts.iter_mut().enumerate() {
            let minimum = (identity_optimizer_steps
                .get(identity)
                .copied()
                .unwrap_or_default() as u64)
                .saturating_mul(replicas);
            *count = (*count).max(minimum);
        }
    }

    #[cfg(test)]
    pub(crate) fn trajectory_counts(&self) -> &[u64] {
        &self.trajectory_counts
    }

    fn priority_weight(&self, identity: usize, observed_mean: f32, min_exposure: u64) -> f32 {
        let loss_weight = if self.loss_observed[identity] && observed_mean > f32::MIN_POSITIVE {
            (self.loss_ema[identity] / observed_mean).max(0.0).sqrt()
        } else {
            1.0
        };
        let coverage_weight = ((min_exposure + 1) as f32
            / (self.trajectory_counts[identity] + 1) as f32)
            .sqrt()
            .max(0.5);
        (loss_weight * coverage_weight).clamp(self.priority_min_weight, self.priority_max_weight)
    }
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;

    #[test]
    fn mixed_sampler_preserves_uniform_coverage_and_unique_batches() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut sampler = E2eIdentitySampler::new(100, 20, 0.75, 0.95, 0.5, 4.0, &mut rng);
        for _ in 0..100 {
            let batch = sampler.next_batch(&mut rng);
            assert_eq!(batch.len(), 20);
            let mut unique = batch.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), batch.len());
            sampler.record_trajectories(&batch, 4);
        }
        let stats = sampler.exposure_stats();
        assert_eq!(stats.total, 8_000);
        assert!(stats.min as f64 >= stats.mean * 0.70, "{stats:?}");
    }

    #[test]
    fn hard_examples_receive_priority_without_starving_coverage() {
        let mut rng = StdRng::seed_from_u64(11);
        let mut sampler = E2eIdentitySampler::new(32, 8, 0.75, 0.9, 0.5, 4.0, &mut rng);
        sampler.update_losses(&(0..32).collect::<Vec<_>>(), &[1.0; 32]);
        sampler.update_losses(&[31], &[100.0]);
        for _ in 0..200 {
            let batch = sampler.next_batch(&mut rng);
            sampler.record_trajectories(&batch, 1);
        }
        let counts = sampler.trajectory_counts();
        assert!(counts[31] > counts.iter().copied().sum::<u64>() / counts.len() as u64);
        let stats = sampler.exposure_stats();
        assert!(stats.min as f64 >= stats.mean * 0.70, "{stats:?}");
    }

    #[test]
    fn sampler_state_round_trips_losslessly() {
        let mut rng = StdRng::seed_from_u64(17);
        let mut sampler = E2eIdentitySampler::new(16, 4, 0.75, 0.95, 0.5, 4.0, &mut rng);
        let batch = sampler.next_batch(&mut rng);
        sampler.record_trajectories(&batch, 8);
        sampler.update_losses(&batch, &[1.0, 2.0, 3.0, 4.0]);
        let encoded = serde_json::to_vec(&sampler).unwrap();
        let restored: E2eIdentitySampler = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(restored.exposure_stats(), sampler.exposure_stats());
        assert_eq!(restored.trajectory_counts(), sampler.trajectory_counts());
    }

    #[test]
    fn compatible_resume_repairs_historical_exposure_from_identity_steps() {
        let mut rng = StdRng::seed_from_u64(19);
        let mut sampler = E2eIdentitySampler::new(4, 4, 0.75, 0.95, 0.5, 4.0, &mut rng);
        sampler.record_trajectories(&[0, 1, 2, 3], 800);
        assert!(sampler.is_compatible(4, 4, 0.75, 0.95, 0.5, 4.0));
        assert!(!sampler.is_compatible(4, 2, 0.75, 0.95, 0.5, 4.0));

        sampler.ensure_minimum_trajectory_counts(&[2_000, 1_500, 100, 0], 8);
        assert_eq!(sampler.trajectory_counts(), [16_000, 12_000, 800, 800]);
    }
}
