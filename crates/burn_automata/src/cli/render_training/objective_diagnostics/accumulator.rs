#[derive(Default)]
pub(super) struct OutputGradientAccumulator {
    sum_sq: f32,
    samples: usize,
    nonzero: usize,
}

impl OutputGradientAccumulator {
    pub(super) fn add_value(&mut self, value: f32) {
        if !value.is_finite() {
            return;
        }
        self.sum_sq += value * value;
        self.samples += 1;
        if value.abs() > 1.0e-8 {
            self.nonzero += 1;
        }
    }

    pub(super) fn rms(&self) -> f32 {
        if self.samples == 0 {
            0.0
        } else {
            (self.sum_sq / self.samples as f32).sqrt()
        }
    }

    pub(super) fn nonzero_fraction(&self) -> f32 {
        if self.samples == 0 {
            0.0
        } else {
            self.nonzero as f32 / self.samples as f32
        }
    }
}

pub(super) fn accumulate_output_channels<I>(
    accumulator: &mut OutputGradientAccumulator,
    gradients: &[f32],
    rows: usize,
    output_dims: usize,
    channels: I,
) where
    I: IntoIterator<Item = usize> + Clone,
{
    if output_dims == 0 || gradients.len() < rows.saturating_mul(output_dims) {
        return;
    }
    for row in 0..rows {
        let base = row * output_dims;
        for channel in channels.clone() {
            if channel < output_dims {
                accumulator.add_value(gradients[base + channel]);
            }
        }
    }
}
