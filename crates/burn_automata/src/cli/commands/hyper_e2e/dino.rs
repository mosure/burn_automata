use burn::{
    backend::NdArray,
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::Tensor,
};
use burn_dino::model::dino::{DinoVisionTransformer, DinoVisionTransformerConfig};
use image::{DynamicImage, RgbImage};

use crate::cli::prelude::*;

type DinoBackend = NdArray<f32>;

pub(super) struct DinoVitsConditionEncoder {
    config: DinoVisionTransformerConfig,
    device: burn::tensor::Device<DinoBackend>,
    model: DinoVisionTransformer<DinoBackend>,
}

impl DinoVitsConditionEncoder {
    pub(super) fn load(
        model_path: &Path,
        image_size: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if image_size == 0 {
            return Err(
                std::io::Error::other("--dino-image-size must be greater than zero").into(),
            );
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

    pub(super) fn encode(
        &self,
        condition: &ConditionImage2d,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let input = preprocess_condition(condition, &self.config, &self.device)?;
        let output = self.model.forward(input, None);
        let patch_dims = output.x_norm_patchtokens.dims();
        let patch_count = patch_dims[1];
        let embed_dims = patch_dims[2];
        let cls = output.x_norm_clstoken.into_data().to_vec::<f32>()?;
        let patch = output.x_norm_patchtokens.into_data().to_vec::<f32>()?;
        if cls.len() != embed_dims || patch.len() != patch_count * embed_dims {
            return Err(std::io::Error::other("DINO output dimensions are inconsistent").into());
        }

        let mut features = Vec::with_capacity(embed_dims * 2);
        features.extend_from_slice(&cls);
        for dim in 0..embed_dims {
            let mut sum = 0.0_f32;
            for patch_idx in 0..patch_count {
                sum += patch[patch_idx * embed_dims + dim];
            }
            features.push(sum / patch_count.max(1) as f32);
        }
        l2_normalize(&mut features);
        Ok(features)
    }
}

fn preprocess_condition(
    condition: &ConditionImage2d,
    config: &DinoVisionTransformerConfig,
    device: &burn::tensor::Device<DinoBackend>,
) -> Result<Tensor<DinoBackend, 4>, Box<dyn std::error::Error>> {
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
    let samples = resized.as_flat_samples();
    let floats = samples.as_slice();
    let input = Tensor::<DinoBackend, 1>::from_floats(floats, device)
        .reshape([
            1,
            config.image_size,
            config.image_size,
            config.input_channels,
        ])
        .permute([0, 3, 1, 2]);
    Ok(normalize(input, device))
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
