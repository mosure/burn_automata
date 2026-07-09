use std::path::Path;

#[cfg(feature = "backend_cuda")]
use burn::backend::Cuda;
#[cfg(all(not(feature = "backend_cuda"), not(feature = "backend_wgpu")))]
use burn::backend::NdArray;
#[cfg(all(not(feature = "backend_cuda"), feature = "backend_wgpu"))]
use burn::backend::Wgpu;
use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::Tensor,
};
use burn_dino::model::dino::{DinoVisionTransformer, DinoVisionTransformerConfig};
use image::{DynamicImage, RgbImage};

use super::{ConditionEncoder2d, ConditionImage2d};

#[cfg(feature = "backend_cuda")]
type DinoBackend = Cuda<f32>;
#[cfg(all(not(feature = "backend_cuda"), feature = "backend_wgpu"))]
type DinoBackend = Wgpu<f32>;
#[cfg(all(not(feature = "backend_cuda"), not(feature = "backend_wgpu")))]
type DinoBackend = NdArray<f32>;

pub struct DinoVitsConditionEncoder {
    config: DinoVisionTransformerConfig,
    device: burn::tensor::Device<DinoBackend>,
    model: DinoVisionTransformer<DinoBackend>,
}

impl DinoVitsConditionEncoder {
    pub fn load(model_path: &Path, image_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        if image_size == 0 {
            return Err(std::io::Error::other("DINO image size must be greater than zero").into());
        }
        let device = burn::tensor::Device::<DinoBackend>::default();
        let config = DinoVisionTransformerConfig {
            register_token_count: 0,
            use_register_tokens: false,
            normalize_intermediate_tokens: false,
            ..DinoVisionTransformerConfig::vits(Some(image_size), None)
        };
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let model = DinoVisionTransformer::new(&device, config.clone())
            .load_file(model_path, &recorder, &device)?;
        Ok(Self {
            config,
            device,
            model,
        })
    }

    pub fn encode_batch(
        &self,
        conditions: &[ConditionImage2d],
        encoder: ConditionEncoder2d,
        token_grid_width: usize,
        token_grid_height: usize,
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        self.encode_batch_with_options(
            conditions,
            encoder,
            token_grid_width,
            token_grid_height,
            true,
        )
    }

    pub fn encode_batch_with_options(
        &self,
        conditions: &[ConditionImage2d],
        encoder: ConditionEncoder2d,
        token_grid_width: usize,
        token_grid_height: usize,
        l2_normalize_features: bool,
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        if conditions.is_empty() {
            return Ok(Vec::new());
        }
        let input = preprocess_conditions(conditions, &self.config, &self.device)?;
        let output = self.model.forward(input, None);
        let cls_dims = output.x_norm_clstoken.dims();
        let patch_dims = output.x_norm_patchtokens.dims();
        let batch = cls_dims[0];
        let embed_dims = cls_dims[1];
        let patch_count = patch_dims[1];
        if batch != conditions.len() || patch_dims[0] != batch || patch_dims[2] != embed_dims {
            return Err(std::io::Error::other("DINO output dimensions are inconsistent").into());
        }
        let cls = output.x_norm_clstoken.into_data().to_vec::<f32>()?;
        let patch = output.x_norm_patchtokens.into_data().to_vec::<f32>()?;
        if cls.len() != batch * embed_dims || patch.len() != batch * patch_count * embed_dims {
            return Err(std::io::Error::other("DINO output dimensions are inconsistent").into());
        }

        let mut encoded = Vec::with_capacity(batch);
        for row in 0..batch {
            let cls_base = row * embed_dims;
            let patch_base = row * patch_count * embed_dims;
            let mut features = match encoder {
                ConditionEncoder2d::DinoVitsClsPatchMean => encode_cls_patch_mean(
                    &cls[cls_base..cls_base + embed_dims],
                    &patch,
                    patch_base,
                    patch_count,
                    embed_dims,
                ),
                ConditionEncoder2d::DinoVitsPatchStats => encode_cls_patch_stats(
                    &cls[cls_base..cls_base + embed_dims],
                    &patch,
                    patch_base,
                    patch_count,
                    embed_dims,
                ),
                ConditionEncoder2d::DinoVitsTokenGrid => encode_cls_patch_token_grid(
                    &cls[cls_base..cls_base + embed_dims],
                    &patch,
                    patch_base,
                    patch_count,
                    embed_dims,
                    token_grid_width,
                    token_grid_height,
                )?,
                ConditionEncoder2d::SummaryTokens => {
                    return Err(
                        std::io::Error::other("summary-token encoder does not use DINO").into(),
                    );
                }
            };
            if l2_normalize_features {
                l2_normalize(&mut features);
            }
            encoded.push(features);
        }
        Ok(encoded)
    }
}

fn encode_cls_patch_token_grid(
    cls: &[f32],
    patch: &[f32],
    patch_base: usize,
    patch_count: usize,
    embed_dims: usize,
    token_grid_width: usize,
    token_grid_height: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let source_grid = square_patch_grid(patch_count)?;
    let grid_width = token_grid_width.max(1);
    let grid_height = token_grid_height.max(1);
    let mut features = Vec::with_capacity((1 + grid_width * grid_height) * embed_dims);
    features.extend_from_slice(cls);
    for tile_y in 0..grid_height {
        for tile_x in 0..grid_width {
            let x0 = tile_x * source_grid / grid_width;
            let mut x1 = (tile_x + 1) * source_grid / grid_width;
            let y0 = tile_y * source_grid / grid_height;
            let mut y1 = (tile_y + 1) * source_grid / grid_height;
            x1 = x1.max(x0 + 1).min(source_grid);
            y1 = y1.max(y0 + 1).min(source_grid);
            let count = ((x1 - x0) * (y1 - y0)).max(1) as f32;
            for dim in 0..embed_dims {
                let mut sum = 0.0_f32;
                for py in y0..y1 {
                    for px in x0..x1 {
                        let patch_idx = py * source_grid + px;
                        sum += patch[patch_base + patch_idx * embed_dims + dim];
                    }
                }
                features.push(sum / count);
            }
        }
    }
    Ok(features)
}

fn square_patch_grid(patch_count: usize) -> Result<usize, Box<dyn std::error::Error>> {
    let grid = (patch_count as f64).sqrt().round() as usize;
    if grid * grid != patch_count {
        return Err(std::io::Error::other(format!(
            "DINO patch token count {patch_count} is not a square grid"
        ))
        .into());
    }
    Ok(grid)
}

fn encode_cls_patch_mean(
    cls: &[f32],
    patch: &[f32],
    patch_base: usize,
    patch_count: usize,
    embed_dims: usize,
) -> Vec<f32> {
    let mut features = Vec::with_capacity(embed_dims * 2);
    features.extend_from_slice(cls);
    for dim in 0..embed_dims {
        let mut sum = 0.0_f32;
        for patch_idx in 0..patch_count {
            sum += patch[patch_base + patch_idx * embed_dims + dim];
        }
        features.push(sum / patch_count.max(1) as f32);
    }
    features
}

fn encode_cls_patch_stats(
    cls: &[f32],
    patch: &[f32],
    patch_base: usize,
    patch_count: usize,
    embed_dims: usize,
) -> Vec<f32> {
    let mut mean = vec![0.0_f32; embed_dims];
    let mut sq_mean = vec![0.0_f32; embed_dims];
    let mut min = vec![f32::INFINITY; embed_dims];
    let mut max = vec![f32::NEG_INFINITY; embed_dims];
    for patch_idx in 0..patch_count {
        let base = patch_base + patch_idx * embed_dims;
        for dim in 0..embed_dims {
            let value = patch[base + dim];
            mean[dim] += value;
            sq_mean[dim] += value * value;
            min[dim] = min[dim].min(value);
            max[dim] = max[dim].max(value);
        }
    }
    let inv_count = 1.0 / patch_count.max(1) as f32;
    let mut std = vec![0.0_f32; embed_dims];
    for dim in 0..embed_dims {
        mean[dim] *= inv_count;
        sq_mean[dim] *= inv_count;
        std[dim] = (sq_mean[dim] - mean[dim] * mean[dim]).max(0.0).sqrt();
    }
    let mut features = Vec::with_capacity(embed_dims * 5);
    features.extend_from_slice(cls);
    features.extend_from_slice(&mean);
    features.extend_from_slice(&std);
    features.extend_from_slice(&min);
    features.extend_from_slice(&max);
    features
}

fn preprocess_conditions(
    conditions: &[ConditionImage2d],
    config: &DinoVisionTransformerConfig,
    device: &burn::tensor::Device<DinoBackend>,
) -> Result<Tensor<DinoBackend, 4>, Box<dyn std::error::Error>> {
    let image_values = conditions
        .iter()
        .map(|condition| preprocess_condition_values(condition, config))
        .collect::<Result<Vec<_>, _>>()?;
    let values = image_values.into_iter().flatten().collect::<Vec<_>>();
    let batch = conditions.len();
    let input = Tensor::<DinoBackend, 1>::from_floats(values.as_slice(), device)
        .reshape([
            batch,
            config.image_size,
            config.image_size,
            config.input_channels,
        ])
        .permute([0, 3, 1, 2]);
    Ok(normalize(input, device))
}

fn preprocess_condition_values(
    condition: &ConditionImage2d,
    config: &DinoVisionTransformerConfig,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    condition.validate()?;
    let mut raw = Vec::with_capacity(condition.width * condition.height * 3);
    for pixel in 0..condition.width * condition.height {
        let base = pixel * condition.channels;
        let rgb = match condition.channels {
            1 => [condition.values[base]; 3],
            _ => [
                condition.values[base],
                condition.values[base + 1],
                condition.values[base + 2],
            ],
        };
        raw.extend(rgb.map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8));
    }
    let image = RgbImage::from_raw(condition.width as u32, condition.height as u32, raw)
        .ok_or_else(|| std::io::Error::other("failed to build DINO condition image buffer"))?;
    let resized = DynamicImage::ImageRgb8(image)
        .resize_exact(
            config.image_size as u32,
            config.image_size as u32,
            image::imageops::FilterType::Triangle,
        )
        .to_rgb32f();
    Ok(resized.as_flat_samples().as_slice().to_vec())
}

fn normalize(
    input: Tensor<DinoBackend, 4>,
    device: &burn::tensor::Device<DinoBackend>,
) -> Tensor<DinoBackend, 4> {
    let mean: Tensor<DinoBackend, 1> = Tensor::from_floats([0.485, 0.456, 0.406], device);
    let std: Tensor<DinoBackend, 1> = Tensor::from_floats([0.229, 0.224, 0.225], device);
    input
        .permute([0, 2, 3, 1])
        .sub(mean.unsqueeze())
        .div(std.unsqueeze())
        .permute([0, 3, 1, 2])
}

fn l2_normalize(values: &mut [f32]) {
    let norm_sq = values.iter().map(|value| value * value).sum::<f32>();
    let inv_norm = norm_sq.max(1.0e-12).sqrt().recip();
    for value in values {
        *value *= inv_norm;
    }
}
