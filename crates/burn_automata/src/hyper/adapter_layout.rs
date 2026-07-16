use crate::{AutomataError, AutomataResult, NpaConfig, NpaLowRankAdapter};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdapterParameterGroup2d {
    W1Down,
    W1Up,
    W2Down,
    W2Up,
    B1,
    B2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdapterParameterSegment2d {
    pub group: AdapterParameterGroup2d,
    pub vector_offset: usize,
    pub len: usize,
    pub chunk_offset: usize,
    pub chunk_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterParameterLayout2d {
    pub rank: usize,
    pub chunk_size: usize,
    pub parameter_count: usize,
    pub chunk_count: usize,
    pub segments: Vec<AdapterParameterSegment2d>,
}

/// A gauge-fixed full-rank LoRA parameterization.
///
/// One factor of each adapted matrix is a fixed identity embedding. The other
/// factor therefore represents the dense weight delta directly, without the
/// non-identifiability of jointly generated LoRA factors.
#[derive(Clone, Debug, PartialEq)]
pub struct CanonicalFullRankLora2d {
    pub constants: Vec<f32>,
    pub trainable_mask: Vec<f32>,
    pub trainable_parameters: usize,
}

impl CanonicalFullRankLora2d {
    pub fn new(config: &NpaConfig, rank: usize, alpha: f32) -> AutomataResult<Self> {
        Self::new_with_output_bias(config, rank, alpha, true)
    }

    pub fn new_with_output_bias(
        config: &NpaConfig,
        rank: usize,
        alpha: f32,
        output_bias: bool,
    ) -> AutomataResult<Self> {
        if rank == 0 || !alpha.is_finite() || alpha.abs() <= f32::EPSILON {
            return Err(AutomataError::InvalidArgument(format!(
                "canonical full-rank LoRA requires positive rank and non-zero finite alpha, got rank={rank} alpha={alpha}"
            )));
        }
        let layout = AdapterParameterLayout2d::new(config, rank, 1)?;
        let mut result = Self {
            constants: vec![0.0; layout.parameter_count],
            trainable_mask: vec![0.0; layout.parameter_count],
            trainable_parameters: 0,
        };
        let fixed_identity_scale = rank as f32 / alpha;
        result.configure_matrix(
            &layout,
            AdapterParameterGroup2d::W1Down,
            AdapterParameterGroup2d::W1Up,
            config.perception_dims(),
            config.hidden_dims,
            rank,
            fixed_identity_scale,
        )?;
        result.configure_matrix(
            &layout,
            AdapterParameterGroup2d::W2Down,
            AdapterParameterGroup2d::W2Up,
            config.hidden_dims,
            config.update_dims(),
            rank,
            fixed_identity_scale,
        )?;
        result.enable_group(&layout, AdapterParameterGroup2d::B1);
        if output_bias {
            result.enable_group(&layout, AdapterParameterGroup2d::B2);
        }
        result.trainable_parameters = result
            .trainable_mask
            .iter()
            .filter(|value| **value != 0.0)
            .count();
        Ok(result)
    }

    pub fn apply(&self, generated: &[f32]) -> AutomataResult<Vec<f32>> {
        if generated.len() != self.constants.len() {
            return Err(AutomataError::InvalidArgument(format!(
                "canonical LoRA generated vector len {} does not match {}",
                generated.len(),
                self.constants.len()
            )));
        }
        Ok(generated
            .iter()
            .zip(&self.trainable_mask)
            .zip(&self.constants)
            .map(|((value, mask), constant)| value * mask + constant)
            .collect())
    }

    #[allow(clippy::too_many_arguments)]
    fn configure_matrix(
        &mut self,
        layout: &AdapterParameterLayout2d,
        down_group: AdapterParameterGroup2d,
        up_group: AdapterParameterGroup2d,
        input_dims: usize,
        output_dims: usize,
        rank: usize,
        fixed_identity_scale: f32,
    ) -> AutomataResult<()> {
        let down = layout.segment(down_group);
        let up = layout.segment(up_group);
        if input_dims <= rank {
            for dim in 0..input_dims {
                self.constants[down.vector_offset + dim * input_dims + dim] = fixed_identity_scale;
            }
            for output in 0..output_dims {
                let row = up.vector_offset + output * rank;
                self.trainable_mask[row..row + input_dims].fill(1.0);
            }
            return Ok(());
        }
        if output_dims <= rank {
            for dim in 0..output_dims {
                self.constants[up.vector_offset + dim * rank + dim] = fixed_identity_scale;
            }
            for row in 0..output_dims {
                let start = down.vector_offset + row * input_dims;
                self.trainable_mask[start..start + input_dims].fill(1.0);
            }
            return Ok(());
        }
        Err(AutomataError::InvalidArgument(format!(
            "rank {rank} cannot canonically represent a full {output_dims}x{input_dims} matrix delta"
        )))
    }

    fn enable_group(&mut self, layout: &AdapterParameterLayout2d, group: AdapterParameterGroup2d) {
        let segment = layout.segment(group);
        self.trainable_mask[segment.vector_offset..segment.vector_offset + segment.len].fill(1.0);
    }
}

impl AdapterParameterLayout2d {
    pub fn new(config: &NpaConfig, rank: usize, chunk_size: usize) -> AutomataResult<Self> {
        if rank == 0 || chunk_size == 0 {
            return Err(AutomataError::InvalidArgument(
                "adapter layout rank and chunk size must be positive".to_string(),
            ));
        }
        let lengths = [
            (
                AdapterParameterGroup2d::W1Down,
                rank * config.perception_dims(),
            ),
            (AdapterParameterGroup2d::W1Up, config.hidden_dims * rank),
            (AdapterParameterGroup2d::W2Down, rank * config.hidden_dims),
            (AdapterParameterGroup2d::W2Up, config.update_dims() * rank),
            (AdapterParameterGroup2d::B1, config.hidden_dims),
            (AdapterParameterGroup2d::B2, config.update_dims()),
        ];
        let mut vector_offset = 0usize;
        let mut chunk_offset = 0usize;
        let mut segments = Vec::with_capacity(lengths.len());
        for (group, len) in lengths {
            let chunk_count = len.div_ceil(chunk_size);
            segments.push(AdapterParameterSegment2d {
                group,
                vector_offset,
                len,
                chunk_offset,
                chunk_count,
            });
            vector_offset = vector_offset.checked_add(len).ok_or_else(|| {
                AutomataError::InvalidArgument("adapter parameter layout overflowed".to_string())
            })?;
            chunk_offset = chunk_offset.checked_add(chunk_count).ok_or_else(|| {
                AutomataError::InvalidArgument("adapter chunk layout overflowed".to_string())
            })?;
        }
        let expected = NpaLowRankAdapter::parameter_count_for_config(config, rank);
        if vector_offset != expected {
            return Err(AutomataError::InvalidModel(format!(
                "adapter layout parameter count {vector_offset} does not match adapter count {expected}"
            )));
        }
        Ok(Self {
            rank,
            chunk_size,
            parameter_count: vector_offset,
            chunk_count: chunk_offset,
            segments,
        })
    }

    pub fn padded_parameter_count(&self) -> usize {
        self.chunk_count * self.chunk_size
    }

    pub fn segment(&self, group: AdapterParameterGroup2d) -> &AdapterParameterSegment2d {
        self.segments
            .iter()
            .find(|segment| segment.group == group)
            .expect("2D adapter layout contains every parameter group")
    }

    pub fn pack(&self, parameters: &[f32]) -> AutomataResult<Vec<f32>> {
        if parameters.len() != self.parameter_count {
            return Err(AutomataError::InvalidArgument(format!(
                "adapter vector len {} does not match layout {}",
                parameters.len(),
                self.parameter_count
            )));
        }
        let mut padded = vec![0.0; self.padded_parameter_count()];
        for segment in &self.segments {
            let source = &parameters[segment.vector_offset..segment.vector_offset + segment.len];
            let destination = segment.chunk_offset * self.chunk_size;
            padded[destination..destination + segment.len].copy_from_slice(source);
        }
        Ok(padded)
    }

    pub fn unpack(&self, padded: &[f32]) -> AutomataResult<Vec<f32>> {
        if padded.len() != self.padded_parameter_count() {
            return Err(AutomataError::InvalidArgument(format!(
                "padded adapter vector len {} does not match layout {}",
                padded.len(),
                self.padded_parameter_count()
            )));
        }
        let mut parameters = vec![0.0; self.parameter_count];
        for segment in &self.segments {
            let source = segment.chunk_offset * self.chunk_size;
            parameters[segment.vector_offset..segment.vector_offset + segment.len]
                .copy_from_slice(&padded[source..source + segment.len]);
        }
        Ok(parameters)
    }

    pub fn structured_query_initialization(&self, hidden_dims: usize, scale: f32) -> Vec<f32> {
        let mut values = vec![0.0; self.chunk_count * hidden_dims];
        for (segment_index, segment) in self.segments.iter().enumerate() {
            for local_chunk in 0..segment.chunk_count {
                let chunk = segment.chunk_offset + local_chunk;
                let progress = (local_chunk as f32 + 0.5) / segment.chunk_count as f32;
                for hidden in 0..hidden_dims {
                    let frequency = 1.0 + (hidden % 16) as f32;
                    let phase = segment_index as f32 * 0.73 + progress * frequency;
                    values[chunk * hidden_dims + hidden] = if hidden % 2 == 0 {
                        phase.sin() * scale
                    } else {
                        phase.cos() * scale
                    };
                }
            }
        }
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_layout_round_trips_without_crossing_parameter_groups() {
        for rank in [16, 32, 64] {
            let config = NpaConfig::growing_2d();
            let layout = AdapterParameterLayout2d::new(&config, rank, 64).unwrap();
            let parameters = (0..layout.parameter_count)
                .map(|index| index as f32)
                .collect::<Vec<_>>();
            assert_eq!(
                layout.unpack(&layout.pack(&parameters).unwrap()).unwrap(),
                parameters
            );
            for segment in &layout.segments {
                assert!(segment.chunk_count * layout.chunk_size >= segment.len);
                assert!(segment.chunk_count.saturating_sub(1) * layout.chunk_size < segment.len);
            }
        }
    }

    #[test]
    fn canonical_full_rank_lora_has_exact_dense_delta_coordinates() {
        let config = NpaConfig::growing_2d();
        let rank = config.perception_dims().max(config.update_dims());
        let canonical = CanonicalFullRankLora2d::new(&config, rank, rank as f32).unwrap();
        let generated = (0..canonical.constants.len())
            .map(|index| index as f32 * 1.0e-5)
            .collect::<Vec<_>>();
        let values = canonical.apply(&generated).unwrap();
        let adapter =
            NpaLowRankAdapter::from_parameter_vector(&config, rank, rank as f32, values).unwrap();

        let w1_scale = adapter.alpha / adapter.rank as f32;
        for output in 0..config.hidden_dims {
            for input in 0..config.perception_dims() {
                let delta = (0..rank)
                    .map(|inner| {
                        adapter.w1_up[output * rank + inner]
                            * adapter.w1_down[inner * config.perception_dims() + input]
                    })
                    .sum::<f32>()
                    * w1_scale;
                let offset = config.perception_dims() * rank + output * rank + input;
                assert!((delta - generated[offset]).abs() < 1.0e-6);
            }
        }

        let layout = AdapterParameterLayout2d::new(&config, rank, 1).unwrap();
        let w2_down = layout.segment(AdapterParameterGroup2d::W2Down);
        for output in 0..config.update_dims() {
            for input in 0..config.hidden_dims {
                let delta = (0..rank)
                    .map(|inner| {
                        adapter.w2_up[output * rank + inner]
                            * adapter.w2_down[inner * config.hidden_dims + input]
                    })
                    .sum::<f32>()
                    * w1_scale;
                let offset = w2_down.vector_offset + output * config.hidden_dims + input;
                assert!((delta - generated[offset]).abs() < 1.0e-6);
            }
        }
        assert_eq!(
            canonical.trainable_parameters,
            config.hidden_dims * config.perception_dims()
                + config.update_dims() * config.hidden_dims
                + config.hidden_dims
                + config.update_dims()
        );
    }

    #[test]
    fn canonical_full_rank_lora_rejects_insufficient_rank() {
        let config = NpaConfig::growing_2d();
        let error = CanonicalFullRankLora2d::new(&config, 16, 16.0).unwrap_err();
        assert!(error.to_string().contains("cannot canonically represent"));
    }

    #[test]
    fn upstream_aligned_canonical_lora_excludes_output_bias() {
        let config = NpaConfig::growing_2d();
        let rank = config.perception_dims().max(config.update_dims());
        let canonical =
            CanonicalFullRankLora2d::new_with_output_bias(&config, rank, rank as f32, false)
                .unwrap();
        let layout = AdapterParameterLayout2d::new(&config, rank, 1).unwrap();
        let b2 = layout.segment(AdapterParameterGroup2d::B2);
        assert!(
            canonical.trainable_mask[b2.vector_offset..b2.vector_offset + b2.len]
                .iter()
                .all(|value| *value == 0.0)
        );
        assert_eq!(
            canonical.trainable_parameters,
            config.hidden_dims * config.perception_dims()
                + config.update_dims() * config.hidden_dims
                + config.hidden_dims
        );
    }
}
