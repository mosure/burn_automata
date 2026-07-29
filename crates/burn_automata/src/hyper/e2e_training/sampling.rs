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
    #[serde(default)]
    active_window_size: usize,
    #[serde(default)]
    active_window_steps: usize,
    #[serde(default)]
    active_window_refresh_size: usize,
    #[serde(default)]
    active_window_batches: usize,
    #[serde(default)]
    active_window: Vec<usize>,
    #[serde(default)]
    window_order: Vec<usize>,
    #[serde(default)]
    window_cursor: usize,
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
    #[cfg(test)]
    pub(crate) fn new<R: Rng + ?Sized>(
        len: usize,
        requested_batch_size: usize,
        uniform_fraction: f32,
        priority_ema_beta: f32,
        priority_min_weight: f32,
        priority_max_weight: f32,
        rng: &mut R,
    ) -> Self {
        Self::new_with_active_window(
            len,
            requested_batch_size,
            uniform_fraction,
            priority_ema_beta,
            priority_min_weight,
            priority_max_weight,
            0,
            0,
            rng,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_active_window<R: Rng + ?Sized>(
        len: usize,
        requested_batch_size: usize,
        uniform_fraction: f32,
        priority_ema_beta: f32,
        priority_min_weight: f32,
        priority_max_weight: f32,
        requested_active_window_size: usize,
        active_window_steps: usize,
        rng: &mut R,
    ) -> Self {
        let batch_size = requested_batch_size.max(1).min(len.max(1));
        let mut uniform_order = (0..len).collect::<Vec<_>>();
        uniform_order.shuffle(rng);
        let active_window_size = requested_active_window_size.max(batch_size).min(len);
        let active_window_size = if active_window_steps == 0 || active_window_size >= len {
            0
        } else {
            active_window_size
        };
        let active_window_refresh_size = active_window_size.div_ceil(4);
        let mut window_order = (0..len).collect::<Vec<_>>();
        window_order.shuffle(rng);
        let mut sampler = Self {
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
            active_window_size,
            active_window_steps: active_window_steps * usize::from(active_window_size > 0),
            active_window_refresh_size,
            active_window_batches: 0,
            active_window: Vec::new(),
            window_order,
            window_cursor: 0,
        };
        if sampler.active_window_enabled() {
            sampler.activate_next_window(rng);
        }
        sampler
    }

    pub(crate) fn next_batch<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Vec<usize> {
        if self.len == 0 {
            return Vec::new();
        }
        if self.active_window_enabled() {
            return self.next_active_window_batch(rng);
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

    #[cfg(test)]
    pub(crate) fn is_compatible(
        &self,
        len: usize,
        requested_batch_size: usize,
        uniform_fraction: f32,
        priority_ema_beta: f32,
        priority_min_weight: f32,
        priority_max_weight: f32,
    ) -> bool {
        self.is_compatible_with_active_window(
            len,
            requested_batch_size,
            uniform_fraction,
            priority_ema_beta,
            priority_min_weight,
            priority_max_weight,
            0,
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn is_compatible_with_active_window(
        &self,
        len: usize,
        requested_batch_size: usize,
        uniform_fraction: f32,
        priority_ema_beta: f32,
        priority_min_weight: f32,
        priority_max_weight: f32,
        requested_active_window_size: usize,
        active_window_steps: usize,
    ) -> bool {
        let batch_size = requested_batch_size.max(1).min(len.max(1));
        let active_window_size = requested_active_window_size.max(batch_size).min(len);
        let active_window_size = if active_window_steps == 0 || active_window_size >= len {
            0
        } else {
            active_window_size
        };
        self.len == len
            && self.batch_size == batch_size
            && (self.uniform_fraction - uniform_fraction.clamp(0.0, 1.0)).abs() <= f32::EPSILON
            && (self.priority_ema_beta - priority_ema_beta.clamp(0.0, 0.999_999)).abs()
                <= f32::EPSILON
            && (self.priority_min_weight - priority_min_weight.max(f32::MIN_POSITIVE)).abs()
                <= f32::EPSILON
            && (self.priority_max_weight - priority_max_weight.max(priority_min_weight)).abs()
                <= f32::EPSILON
            && self.active_window_size == active_window_size
            && self.active_window_steps == active_window_steps * usize::from(active_window_size > 0)
            && self.active_window_refresh_size == active_window_size.div_ceil(4)
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

    #[cfg(test)]
    pub(crate) fn active_window(&self) -> &[usize] {
        &self.active_window
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

    fn active_window_enabled(&self) -> bool {
        self.active_window_size > 0 && self.active_window_steps > 0
    }

    fn activate_next_window<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        if !self.active_window.is_empty() {
            let refresh = self
                .active_window_refresh_size
                .max(1)
                .min(self.active_window.len());
            self.active_window.drain(..refresh);
        }
        while self.active_window.len() < self.active_window_size {
            if self.window_cursor >= self.window_order.len() {
                self.window_order = (0..self.len).collect();
                self.window_order.shuffle(rng);
                self.window_cursor = 0;
            }
            let identity = self.window_order[self.window_cursor];
            self.window_cursor += 1;
            if !self.active_window.contains(&identity) {
                self.active_window.push(identity);
            }
        }
        self.uniform_order = self.active_window.clone();
        self.uniform_order.shuffle(rng);
        self.uniform_cursor = 0;
        self.active_window_batches = 0;
    }

    fn next_active_window_batch<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Vec<usize> {
        if self.active_window.is_empty() || self.active_window_batches >= self.active_window_steps {
            self.activate_next_window(rng);
        }
        let batch_size = self.batch_size.min(self.active_window.len());
        if batch_size >= self.active_window.len() {
            let mut all = self.active_window.clone();
            all.shuffle(rng);
            self.active_window_batches += 1;
            return all;
        }

        let uniform_count =
            ((batch_size as f32 * self.uniform_fraction).ceil() as usize).clamp(1, batch_size);
        let mut selected = Vec::with_capacity(batch_size);
        while selected.len() < uniform_count {
            if self.uniform_cursor >= self.uniform_order.len() {
                self.uniform_order = self.active_window.clone();
                self.uniform_order.shuffle(rng);
                self.uniform_cursor = 0;
            }
            let identity = self.uniform_order[self.uniform_cursor];
            self.uniform_cursor += 1;
            if !selected.contains(&identity) {
                selected.push(identity);
            }
        }

        let observed = self
            .active_window
            .iter()
            .copied()
            .filter(|identity| self.loss_observed[*identity])
            .collect::<Vec<_>>();
        let observed_mean = observed
            .iter()
            .map(|identity| self.loss_ema[*identity])
            .sum::<f32>()
            / observed.len().max(1) as f32;
        let min_exposure = self
            .active_window
            .iter()
            .map(|identity| self.trajectory_counts[*identity])
            .min()
            .unwrap_or(0);
        while selected.len() < batch_size {
            let candidates = self
                .active_window
                .iter()
                .copied()
                .filter(|identity| !selected.contains(identity))
                .collect::<Vec<_>>();
            let weights = candidates
                .iter()
                .map(|identity| self.priority_weight(*identity, observed_mean, min_exposure))
                .collect::<Vec<_>>();
            let total = weights.iter().sum::<f32>();
            let mut draw = rng.random::<f32>() * total.max(f32::MIN_POSITIVE);
            let mut chosen = *candidates.last().expect("non-empty priority candidates");
            for (identity, weight) in candidates.into_iter().zip(weights) {
                draw -= weight;
                if draw <= 0.0 {
                    chosen = identity;
                    break;
                }
            }
            selected.push(chosen);
        }
        self.active_window_batches += 1;
        selected
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

    #[test]
    fn active_windows_preserve_local_revisits_and_rotate_across_the_dataset() {
        let mut rng = StdRng::seed_from_u64(23);
        let mut sampler = E2eIdentitySampler::new_with_active_window(
            100, 8, 1.0, 0.95, 0.5, 4.0, 20, 10, &mut rng,
        );
        let first_window = sampler.active_window().to_vec();
        let mut windows = Vec::new();
        for window in 0..17 {
            let mut identities = Vec::new();
            for _ in 0..10 {
                let batch = sampler.next_batch(&mut rng);
                assert_eq!(batch.len(), 8);
                identities.extend(batch.iter().copied());
                sampler.record_trajectories(&batch, 4);
            }
            identities.sort_unstable();
            identities.dedup();
            assert!(
                identities.len() <= 20,
                "window {window} escaped its active identity set: {identities:?}"
            );
            windows.extend(identities);
            if window == 1 {
                let retained = first_window
                    .iter()
                    .filter(|identity| sampler.active_window().contains(identity))
                    .count();
                assert_eq!(retained, 15);
            }
        }
        windows.sort_unstable();
        windows.dedup();
        assert_eq!(windows.len(), 100);
        assert!(sampler.is_compatible_with_active_window(100, 8, 1.0, 0.95, 0.5, 4.0, 20, 10,));
        assert!(!sampler.is_compatible_with_active_window(100, 8, 1.0, 0.95, 0.5, 4.0, 25, 10,));
    }

    #[test]
    fn active_window_checkpoint_round_trip_preserves_rotation_state() {
        let mut rng = StdRng::seed_from_u64(29);
        let mut sampler =
            E2eIdentitySampler::new_with_active_window(32, 4, 0.75, 0.95, 0.5, 4.0, 8, 3, &mut rng);
        for _ in 0..5 {
            sampler.next_batch(&mut rng);
        }
        let encoded = serde_json::to_vec(&sampler).unwrap();
        let mut restored: E2eIdentitySampler = serde_json::from_slice(&encoded).unwrap();
        let mut expected_rng = rng.clone();
        assert_eq!(
            restored.next_batch(&mut rng),
            sampler.next_batch(&mut expected_rng)
        );
    }
}
