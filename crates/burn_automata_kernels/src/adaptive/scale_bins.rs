use crate::{KernelError, KernelResult};

/// Hard limit on geometric support bins used by the adaptive broad phase.
///
/// Ratios arbitrarily close to one otherwise turn a bounded bandwidth range
/// into an effectively unbounded allocation and device-dispatch surface.
pub const MAX_ADAPTIVE_SUPPORT_BINS: usize = 64;

/// Logarithmic source-bandwidth bins used only to accelerate neighbor search.
///
/// A source belongs to one bin whose upper bound is at least its exact
/// continuous bandwidth. Queries use that upper bound to form a conservative
/// search radius, then evaluate the exact pair bandwidth before accepting an
/// interaction. Bins therefore change candidate work, never NPA semantics.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveSupportBins {
    minimum: f32,
    maximum: f32,
    ratio: f32,
    upper_bounds: Vec<f32>,
}

impl AdaptiveSupportBins {
    pub fn new(minimum: f32, maximum: f32, ratio: f32) -> KernelResult<Self> {
        if !minimum.is_finite()
            || !maximum.is_finite()
            || !ratio.is_finite()
            || minimum <= 0.0
            || maximum < minimum
            || ratio <= 1.0
        {
            return Err(KernelError::InvalidArgument(
                "adaptive support bins require finite 0 < min <= max and ratio > 1".to_string(),
            ));
        }
        let mut upper_bounds = Vec::new();
        let mut upper = (minimum * ratio).min(maximum);
        loop {
            if upper_bounds.len() == MAX_ADAPTIVE_SUPPORT_BINS {
                return Err(KernelError::InvalidArgument(format!(
                    "adaptive support bins exceed the {MAX_ADAPTIVE_SUPPORT_BINS}-bin limit; increase support_bin_ratio or narrow the bandwidth range"
                )));
            }
            upper_bounds.push(upper);
            if upper >= maximum {
                break;
            }
            let next = (upper * ratio).min(maximum);
            if next <= upper {
                return Err(KernelError::InvalidArgument(
                    "adaptive support-bin progression did not advance".to_string(),
                ));
            }
            upper = next;
        }
        Ok(Self {
            minimum,
            maximum,
            ratio,
            upper_bounds,
        })
    }

    pub fn dyadic(minimum: f32, maximum: f32) -> KernelResult<Self> {
        Self::new(minimum, maximum, 2.0)
    }

    pub fn len(&self) -> usize {
        self.upper_bounds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.upper_bounds.is_empty()
    }

    pub fn minimum(&self) -> f32 {
        self.minimum
    }

    pub fn maximum(&self) -> f32 {
        self.maximum
    }

    pub fn ratio(&self) -> f32 {
        self.ratio
    }

    pub fn upper_bounds(&self) -> &[f32] {
        &self.upper_bounds
    }

    pub fn bin_index(&self, bandwidth: f32) -> KernelResult<usize> {
        if !bandwidth.is_finite() || bandwidth < self.minimum || bandwidth > self.maximum {
            return Err(KernelError::InvalidArgument(format!(
                "bandwidth {bandwidth} is outside adaptive support bins {}..{}",
                self.minimum, self.maximum,
            )));
        }
        Ok(self
            .upper_bounds
            .partition_point(|upper| *upper < bandwidth)
            .min(self.upper_bounds.len() - 1))
    }

    pub fn conservative_pair_radius(
        &self,
        target_bandwidth: f32,
        source_bin: usize,
        pair_scale_power: f32,
    ) -> KernelResult<f32> {
        if !target_bandwidth.is_finite()
            || target_bandwidth < self.minimum
            || target_bandwidth > self.maximum
            || source_bin >= self.upper_bounds.len()
            || !pair_scale_power.is_finite()
            || pair_scale_power < 1.0
        {
            return Err(KernelError::InvalidArgument(
                "invalid adaptive support-bin query".to_string(),
            ));
        }
        Ok(pair_bandwidth(
            target_bandwidth,
            self.upper_bounds[source_bin],
            pair_scale_power,
        ))
    }
}

fn pair_bandwidth(lhs: f32, rhs: f32, power: f32) -> f32 {
    ((lhs.powf(power) + rhs.powf(power)) * 0.5).powf(power.recip())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dyadic_bins_cover_continuous_bandwidths_without_quantizing_pair_support() {
        let bins = AdaptiveSupportBins::dyadic(0.025, 0.4).unwrap();
        assert_eq!(bins.upper_bounds(), &[0.05, 0.1, 0.2, 0.4]);
        for target_step in 0..=128 {
            let target = 0.025 + (0.4 - 0.025) * target_step as f32 / 128.0;
            for source_step in 0..=128 {
                let source = 0.025 + (0.4 - 0.025) * source_step as f32 / 128.0;
                let source_bin = bins.bin_index(source).unwrap();
                let conservative = bins
                    .conservative_pair_radius(target, source_bin, 8.0)
                    .unwrap();
                let exact = pair_bandwidth(target, source, 8.0);
                assert!(conservative + 1.0e-7 >= exact);
            }
        }
    }

    #[test]
    fn near_unit_ratio_is_rejected_before_unbounded_bin_growth() {
        let error = AdaptiveSupportBins::new(0.025, 0.4, 1.000_001).unwrap_err();
        assert!(error.to_string().contains("64-bin limit"));
    }
}
