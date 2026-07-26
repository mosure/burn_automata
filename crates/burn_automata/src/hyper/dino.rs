use std::path::Path;

#[cfg(feature = "backend_cuda")]
use burn::backend::Cuda;
#[cfg(all(not(feature = "backend_cuda"), not(feature = "backend_wgpu")))]
use burn::backend::NdArray;
#[cfg(all(not(feature = "backend_cuda"), feature = "backend_wgpu"))]
use burn::backend::Wgpu;
use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, NamedMpkFileRecorder, Recorder},
    tensor::{Tensor, backend::Backend, module::adaptive_avg_pool2d},
};
use burn_dino::model::dino::{DinoVisionTransformer, DinoVisionTransformerConfig};
use image::{DynamicImage, GrayImage, RgbImage};

use super::{ConditionEncoder2d, ConditionImage2d};

#[cfg(feature = "backend_cuda")]
type DinoBackend = Cuda<f32>;
#[cfg(all(not(feature = "backend_cuda"), feature = "backend_wgpu"))]
type DinoBackend = Wgpu<f32>;
#[cfg(all(not(feature = "backend_cuda"), not(feature = "backend_wgpu")))]
type DinoBackend = NdArray<f32>;

/// Transparent condition images are interpreted as artwork on a white canvas.
/// This matches normal SVG thumbnail presentation and prevents transparent
/// black line art from becoming an all-black DINO input.
pub const DINO_CONDITION_BACKGROUND_RGB: [f32; 3] = [1.0, 1.0, 1.0];

pub fn decode_condition_image(
    bytes: &[u8],
) -> Result<ConditionImage2d, Box<dyn std::error::Error>> {
    let image = image::load_from_memory(bytes)?.to_rgba8();
    let (width, height) = image.dimensions();
    let values = image
        .as_raw()
        .iter()
        .map(|value| *value as f32 / 255.0)
        .collect();
    Ok(ConditionImage2d::from_rgba(
        width as usize,
        height as usize,
        values,
    )?)
}

pub fn load_condition_image(path: &Path) -> Result<ConditionImage2d, Box<dyn std::error::Error>> {
    decode_condition_image(&std::fs::read(path)?)
}

pub type DinoVitsConditionEncoder = DinoVitsConditionEncoderBackend<DinoBackend>;

pub struct DinoVitsConditionEncoderBackend<B: Backend> {
    config: DinoVisionTransformerConfig,
    device: burn::tensor::Device<B>,
    model: DinoVisionTransformer<B>,
}

#[derive(Clone, Copy, Debug)]
pub struct DinoVitsConditionContract {
    pub encoder: ConditionEncoder2d,
    pub token_grid_width: usize,
    pub token_grid_height: usize,
    pub l2_normalize_features: bool,
    pub append_rgb_channels: bool,
    pub rgb_channel_scale: f32,
    pub append_alpha_channel: bool,
    pub alpha_channel_scale: f32,
    pub append_patch_pixels: bool,
}

impl DinoVitsConditionContract {
    pub const fn token_grid(
        token_grid_width: usize,
        token_grid_height: usize,
        l2_normalize_features: bool,
        append_rgb_channels: bool,
        rgb_channel_scale: f32,
        append_alpha_channel: bool,
        alpha_channel_scale: f32,
    ) -> Self {
        Self {
            encoder: ConditionEncoder2d::DinoVitsTokenGrid,
            token_grid_width,
            token_grid_height,
            l2_normalize_features,
            append_rgb_channels,
            rgb_channel_scale,
            append_alpha_channel,
            alpha_channel_scale,
            append_patch_pixels: false,
        }
    }

    pub const fn with_patch_pixels(mut self, append_patch_pixels: bool) -> Self {
        self.append_patch_pixels = append_patch_pixels;
        self
    }
}

#[derive(Clone, Debug)]
pub struct DinoVitsPreparedConditionBatch {
    values: Vec<f32>,
    alpha_values: Vec<f32>,
    batch: usize,
    image_size: usize,
    input_channels: usize,
}

impl DinoVitsPreparedConditionBatch {
    pub fn from_conditions(
        conditions: &[ConditionImage2d],
        image_size: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut values = Vec::with_capacity(
            conditions
                .len()
                .saturating_mul(image_size)
                .saturating_mul(image_size)
                .saturating_mul(3),
        );
        let mut alpha_values = Vec::with_capacity(
            conditions
                .len()
                .saturating_mul(image_size)
                .saturating_mul(image_size),
        );
        for condition in conditions {
            values.extend(preprocess_condition_values(condition, image_size)?);
            alpha_values.extend(preprocess_condition_alpha_values(condition, image_size)?);
        }
        Ok(Self {
            values,
            alpha_values,
            batch: conditions.len(),
            image_size,
            input_channels: 3,
        })
    }

    fn validate_for(
        &self,
        config: &DinoVisionTransformerConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.batch == 0 {
            return Err(std::io::Error::other("prepared DINO batch is empty").into());
        }
        if self.image_size != config.image_size || self.input_channels != config.input_channels {
            return Err(std::io::Error::other(format!(
                "prepared DINO batch shape {}x{}x{} does not match model {}x{}x{}",
                self.batch,
                self.image_size,
                self.input_channels,
                config.image_size,
                config.image_size,
                config.input_channels
            ))
            .into());
        }
        let expected = self
            .batch
            .saturating_mul(self.image_size)
            .saturating_mul(self.image_size)
            .saturating_mul(self.input_channels);
        if self.values.len() != expected {
            return Err(std::io::Error::other(format!(
                "prepared DINO batch has {} values, expected {expected}",
                self.values.len()
            ))
            .into());
        }
        let expected_alpha = self
            .batch
            .saturating_mul(self.image_size)
            .saturating_mul(self.image_size);
        if self.alpha_values.len() != expected_alpha {
            return Err(std::io::Error::other(format!(
                "prepared DINO alpha batch has {} values, expected {expected_alpha}",
                self.alpha_values.len()
            ))
            .into());
        }
        Ok(())
    }
}

impl<B: Backend> DinoVitsConditionEncoderBackend<B> {
    pub fn load(model_path: &Path, image_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        if image_size == 0 {
            return Err(std::io::Error::other("DINO image size must be greater than zero").into());
        }
        let device = burn::tensor::Device::<B>::default();
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

    pub fn load_bytes(
        bytes: Vec<u8>,
        image_size: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if image_size == 0 {
            return Err(std::io::Error::other("DINO image size must be greater than zero").into());
        }
        let device = burn::tensor::Device::<B>::default();
        let config = DinoVisionTransformerConfig {
            register_token_count: 0,
            use_register_tokens: false,
            normalize_intermediate_tokens: false,
            ..DinoVisionTransformerConfig::vits(Some(image_size), None)
        };
        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
        let record = recorder.load(bytes, &device)?;
        let model = DinoVisionTransformer::new(&device, config.clone()).load_record(record);
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

    pub fn encode_batch_with_alpha_channel(
        &self,
        conditions: &[ConditionImage2d],
        encoder: ConditionEncoder2d,
        token_grid_width: usize,
        token_grid_height: usize,
        l2_normalize_features: bool,
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        self.encode_batch_with_scaled_alpha_channel(
            conditions,
            encoder,
            token_grid_width,
            token_grid_height,
            l2_normalize_features,
            1.0,
        )
    }

    pub fn encode_batch_with_scaled_alpha_channel(
        &self,
        conditions: &[ConditionImage2d],
        encoder: ConditionEncoder2d,
        token_grid_width: usize,
        token_grid_height: usize,
        l2_normalize_features: bool,
        alpha_channel_scale: f32,
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        if conditions.is_empty() {
            return Ok(Vec::new());
        }
        self.encode_batch_with_contract(
            conditions,
            DinoVitsConditionContract {
                encoder,
                token_grid_width,
                token_grid_height,
                l2_normalize_features,
                append_rgb_channels: false,
                rgb_channel_scale: 1.0,
                append_alpha_channel: true,
                alpha_channel_scale,
                append_patch_pixels: false,
            },
        )
    }

    pub fn encode_batch_with_contract(
        &self,
        conditions: &[ConditionImage2d],
        contract: DinoVitsConditionContract,
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        if conditions.is_empty() {
            return Ok(Vec::new());
        }
        let encoded = self.encode_batch_tensor_with_contract(conditions, contract)?;
        let dims = encoded.dims();
        let values = encoded.into_data().to_vec::<f32>()?;
        condition_tensor_rows(dims, values)
    }

    pub async fn encode_batch_with_contract_async(
        &self,
        conditions: &[ConditionImage2d],
        contract: DinoVitsConditionContract,
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        if conditions.is_empty() {
            return Ok(Vec::new());
        }
        let encoded = self.encode_batch_tensor_with_contract(conditions, contract)?;
        let dims = encoded.dims();
        let values = encoded.into_data_async().await?.to_vec::<f32>()?;
        condition_tensor_rows(dims, values)
    }

    pub fn encode_batch_tensor_with_options(
        &self,
        conditions: &[ConditionImage2d],
        encoder: ConditionEncoder2d,
        token_grid_width: usize,
        token_grid_height: usize,
        l2_normalize_features: bool,
    ) -> Result<Tensor<B, 3>, Box<dyn std::error::Error>> {
        self.encode_batch_tensor_with_contract(
            conditions,
            DinoVitsConditionContract {
                encoder,
                token_grid_width,
                token_grid_height,
                l2_normalize_features,
                append_rgb_channels: false,
                rgb_channel_scale: 1.0,
                append_alpha_channel: false,
                alpha_channel_scale: 1.0,
                append_patch_pixels: false,
            },
        )
    }

    pub fn encode_batch_tensor_with_contract(
        &self,
        conditions: &[ConditionImage2d],
        contract: DinoVitsConditionContract,
    ) -> Result<Tensor<B, 3>, Box<dyn std::error::Error>> {
        if conditions.is_empty() {
            return Err(
                std::io::Error::other("DINO tensor encoding requires a non-empty batch").into(),
            );
        }
        let prepared =
            DinoVitsPreparedConditionBatch::from_conditions(conditions, self.config.image_size)?;
        self.encode_preprocessed_batch_tensor_with_contract(&prepared, contract)
    }

    pub fn encode_preprocessed_batch_tensor_with_options(
        &self,
        prepared: &DinoVitsPreparedConditionBatch,
        encoder: ConditionEncoder2d,
        token_grid_width: usize,
        token_grid_height: usize,
        l2_normalize_features: bool,
    ) -> Result<Tensor<B, 3>, Box<dyn std::error::Error>> {
        self.encode_preprocessed_batch_tensor_with_contract(
            prepared,
            DinoVitsConditionContract {
                encoder,
                token_grid_width,
                token_grid_height,
                l2_normalize_features,
                append_rgb_channels: false,
                rgb_channel_scale: 1.0,
                append_alpha_channel: false,
                alpha_channel_scale: 1.0,
                append_patch_pixels: false,
            },
        )
    }

    pub fn encode_preprocessed_batch_tensor_with_contract(
        &self,
        prepared: &DinoVitsPreparedConditionBatch,
        contract: DinoVitsConditionContract,
    ) -> Result<Tensor<B, 3>, Box<dyn std::error::Error>> {
        let DinoVitsConditionContract {
            encoder,
            token_grid_width,
            token_grid_height,
            l2_normalize_features,
            append_rgb_channels,
            rgb_channel_scale,
            append_alpha_channel,
            alpha_channel_scale,
            append_patch_pixels,
        } = contract;
        if !rgb_channel_scale.is_finite() || rgb_channel_scale <= 0.0 {
            return Err(std::io::Error::other(
                "DINO RGB channel scale must be positive and finite",
            )
            .into());
        }
        if !alpha_channel_scale.is_finite() || alpha_channel_scale <= 0.0 {
            return Err(std::io::Error::other(
                "DINO alpha channel scale must be positive and finite",
            )
            .into());
        }
        prepared.validate_for(&self.config)?;
        let input = preprocessed_conditions_tensor(prepared, &self.config, &self.device)?;
        let output = self.model.forward(input, None);
        let cls_dims = output.x_norm_clstoken.dims();
        let patch_dims = output.x_norm_patchtokens.dims();
        let batch = cls_dims[0];
        let embed_dims = cls_dims[1];
        let patch_count = patch_dims[1];
        if batch != prepared.batch || patch_dims[0] != batch || patch_dims[2] != embed_dims {
            return Err(std::io::Error::other("DINO output dimensions are inconsistent").into());
        }
        let mut encoded = match encoder {
            ConditionEncoder2d::DinoVitsTokenGrid => encode_cls_patch_token_grid_tensor(
                output.x_norm_clstoken,
                output.x_norm_patchtokens,
                patch_count,
                embed_dims,
                token_grid_width,
                token_grid_height,
            )?,
            ConditionEncoder2d::DinoVitsClsPatchMean => Tensor::cat(
                vec![
                    output.x_norm_clstoken.unsqueeze_dim::<3>(1),
                    output.x_norm_patchtokens.mean_dim(1),
                ],
                1,
            ),
            ConditionEncoder2d::DinoVitsPatchStats => {
                let patch = output.x_norm_patchtokens;
                let mean = patch.clone().mean_dim(1);
                let std = patch.clone().var_bias(1).sqrt();
                let min = patch.clone().min_dim(1);
                let max = patch.max_dim(1);
                Tensor::cat(
                    vec![
                        output.x_norm_clstoken.unsqueeze_dim::<3>(1),
                        mean,
                        std,
                        min,
                        max,
                    ],
                    1,
                )
            }
            ConditionEncoder2d::SummaryTokens => {
                return Err(
                    std::io::Error::other("summary-token encoder does not use DINO").into(),
                );
            }
        };
        if (append_rgb_channels || append_alpha_channel)
            && encoder != ConditionEncoder2d::DinoVitsTokenGrid
        {
            return Err(std::io::Error::other(
                "DINO RGB/alpha channels currently require the token-grid encoder",
            )
            .into());
        }
        if append_patch_pixels {
            let pixels = patch_pixel_token_grid_tensor(
                prepared,
                token_grid_width,
                token_grid_height,
                PatchPixelChannels {
                    rgb_scale: append_rgb_channels.then_some(rgb_channel_scale),
                    alpha_scale: append_alpha_channel.then_some(alpha_channel_scale),
                },
                &self.device,
            )?;
            encoded = Tensor::cat(vec![encoded, pixels], 2);
        } else {
            if append_rgb_channels {
                let rgb = rgb_token_grid_tensor(
                    prepared,
                    token_grid_width,
                    token_grid_height,
                    &self.device,
                )?
                .mul_scalar(rgb_channel_scale);
                encoded = Tensor::cat(vec![encoded, rgb], 2);
            }
            if append_alpha_channel {
                let alpha = alpha_token_grid_tensor(
                    prepared,
                    token_grid_width,
                    token_grid_height,
                    &self.device,
                )?
                .mul_scalar(alpha_channel_scale);
                encoded = Tensor::cat(vec![encoded, alpha], 2);
            }
        }
        if l2_normalize_features {
            Ok(l2_normalize_tensor(encoded))
        } else {
            Ok(encoded)
        }
    }
}

fn condition_tensor_rows(
    dims: [usize; 3],
    values: Vec<f32>,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let row_len = dims[1].saturating_mul(dims[2]);
    let expected = dims[0].saturating_mul(row_len);
    if row_len == 0 || values.len() != expected {
        return Err(std::io::Error::other(format!(
            "DINO condition tensor shape {dims:?} does not match {} values",
            values.len()
        ))
        .into());
    }
    Ok(values.chunks_exact(row_len).map(<[f32]>::to_vec).collect())
}

#[derive(Clone, Copy)]
struct PatchPixelChannels {
    rgb_scale: Option<f32>,
    alpha_scale: Option<f32>,
}

fn patch_pixel_token_grid_tensor<B: Backend>(
    prepared: &DinoVitsPreparedConditionBatch,
    token_grid_width: usize,
    token_grid_height: usize,
    channel_config: PatchPixelChannels,
    device: &burn::tensor::Device<B>,
) -> Result<Tensor<B, 3>, Box<dyn std::error::Error>> {
    if channel_config.rgb_scale.is_none() && channel_config.alpha_scale.is_none() {
        return Err(std::io::Error::other(
            "DINO patch-pixel conditioning requires RGB and/or alpha channels",
        )
        .into());
    }
    let grid_width = token_grid_width.max(1);
    let grid_height = token_grid_height.max(1);
    if !prepared.image_size.is_multiple_of(grid_width)
        || !prepared.image_size.is_multiple_of(grid_height)
    {
        return Err(std::io::Error::other(format!(
            "DINO image size {} must be divisible by patch grid {}x{} for lossless patch pixels",
            prepared.image_size, grid_width, grid_height
        ))
        .into());
    }
    let patch_width = prepared.image_size / grid_width;
    let patch_height = prepared.image_size / grid_height;
    let mut channels = Vec::with_capacity(2);
    if let Some(rgb_scale) = channel_config.rgb_scale {
        channels.push(
            Tensor::<B, 1>::from_floats(prepared.values.as_slice(), device)
                .reshape([prepared.batch, prepared.image_size, prepared.image_size, 3])
                .mul_scalar(rgb_scale),
        );
    }
    if let Some(alpha_scale) = channel_config.alpha_scale {
        channels.push(
            Tensor::<B, 1>::from_floats(prepared.alpha_values.as_slice(), device)
                .reshape([prepared.batch, prepared.image_size, prepared.image_size, 1])
                .mul_scalar(alpha_scale),
        );
    }
    let rgba = Tensor::cat(channels, 3);
    let channel_count = rgba.dims()[3];
    let global = adaptive_avg_pool2d(
        rgba.clone().permute([0, 3, 1, 2]),
        [patch_height, patch_width],
    )
    .permute([0, 2, 3, 1])
    .reshape([
        prepared.batch,
        1,
        patch_height * patch_width * channel_count,
    ]);
    let patches = rgba
        .reshape([
            prepared.batch,
            grid_height,
            patch_height,
            grid_width,
            patch_width,
            channel_count,
        ])
        .permute([0, 1, 3, 2, 4, 5])
        .reshape([
            prepared.batch,
            grid_height * grid_width,
            patch_height * patch_width * channel_count,
        ]);
    Ok(Tensor::cat(vec![global, patches], 1))
}

fn rgb_token_grid_tensor<B: Backend>(
    prepared: &DinoVitsPreparedConditionBatch,
    token_grid_width: usize,
    token_grid_height: usize,
    device: &burn::tensor::Device<B>,
) -> Result<Tensor<B, 3>, Box<dyn std::error::Error>> {
    let grid_width = token_grid_width.max(1);
    let grid_height = token_grid_height.max(1);
    let rgb = Tensor::<B, 1>::from_floats(prepared.values.as_slice(), device)
        .reshape([prepared.batch, prepared.image_size, prepared.image_size, 3])
        .permute([0, 3, 1, 2]);
    let cls = rgb
        .clone()
        .reshape([prepared.batch, 3, prepared.image_size * prepared.image_size])
        .mean_dim(2)
        .permute([0, 2, 1]);
    let patches = adaptive_avg_pool2d(rgb, [grid_height, grid_width])
        .permute([0, 2, 3, 1])
        .reshape([prepared.batch, grid_width * grid_height, 3]);
    Ok(Tensor::cat(vec![cls, patches], 1))
}

fn encode_cls_patch_token_grid_tensor<B: Backend>(
    cls: Tensor<B, 2>,
    patch: Tensor<B, 3>,
    patch_count: usize,
    embed_dims: usize,
    token_grid_width: usize,
    token_grid_height: usize,
) -> Result<Tensor<B, 3>, Box<dyn std::error::Error>> {
    let source_grid = square_patch_grid(patch_count)?;
    let grid_width = token_grid_width.max(1);
    let grid_height = token_grid_height.max(1);
    let patch_tokens = if grid_width == source_grid && grid_height == source_grid {
        patch
    } else {
        adaptive_avg_pool2d(
            patch
                .reshape([cls.dims()[0], source_grid, source_grid, embed_dims])
                .permute([0, 3, 1, 2]),
            [grid_height, grid_width],
        )
        .permute([0, 2, 3, 1])
        .reshape([cls.dims()[0], grid_width * grid_height, embed_dims])
    };
    Ok(Tensor::cat(
        vec![cls.unsqueeze_dim::<3>(1), patch_tokens],
        1,
    ))
}

fn alpha_token_grid_tensor<B: Backend>(
    prepared: &DinoVitsPreparedConditionBatch,
    token_grid_width: usize,
    token_grid_height: usize,
    device: &burn::tensor::Device<B>,
) -> Result<Tensor<B, 3>, Box<dyn std::error::Error>> {
    let grid_width = token_grid_width.max(1);
    let grid_height = token_grid_height.max(1);
    let alpha = Tensor::<B, 1>::from_floats(prepared.alpha_values.as_slice(), device).reshape([
        prepared.batch,
        1,
        prepared.image_size,
        prepared.image_size,
    ]);
    let cls = alpha
        .clone()
        .reshape([prepared.batch, prepared.image_size * prepared.image_size])
        .mean_dim(1)
        .unsqueeze_dim::<3>(1);
    let patches = adaptive_avg_pool2d(alpha, [grid_height, grid_width])
        .permute([0, 2, 3, 1])
        .reshape([prepared.batch, grid_width * grid_height, 1]);
    Ok(Tensor::cat(vec![cls, patches], 1))
}

fn l2_normalize_tensor<B: Backend>(values: Tensor<B, 3>) -> Tensor<B, 3> {
    let dims = values.dims();
    let batch = dims[0];
    let tokens = dims[1];
    let embed_dims = dims[2];
    let norm = values
        .clone()
        .reshape([batch, tokens * embed_dims])
        .powf_scalar(2.0)
        .sum_dim(1)
        .sqrt()
        .add_scalar(1.0e-12)
        .reshape([batch, 1, 1])
        .expand([batch, tokens, embed_dims]);
    values.div(norm)
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

fn preprocess_conditions<B: Backend>(
    conditions: &[ConditionImage2d],
    config: &DinoVisionTransformerConfig,
    device: &burn::tensor::Device<B>,
) -> Result<Tensor<B, 4>, Box<dyn std::error::Error>> {
    let prepared = DinoVitsPreparedConditionBatch::from_conditions(conditions, config.image_size)?;
    preprocessed_conditions_tensor(&prepared, config, device)
}

fn preprocessed_conditions_tensor<B: Backend>(
    prepared: &DinoVitsPreparedConditionBatch,
    config: &DinoVisionTransformerConfig,
    device: &burn::tensor::Device<B>,
) -> Result<Tensor<B, 4>, Box<dyn std::error::Error>> {
    prepared.validate_for(config)?;
    let input = Tensor::<B, 1>::from_floats(prepared.values.as_slice(), device)
        .reshape([
            prepared.batch,
            config.image_size,
            config.image_size,
            config.input_channels,
        ])
        .permute([0, 3, 1, 2]);
    Ok(normalize(input, device))
}

fn preprocess_condition_values(
    condition: &ConditionImage2d,
    image_size: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    condition.validate()?;
    let raw = condition
        .composited_rgb_values(DINO_CONDITION_BACKGROUND_RGB)?
        .into_iter()
        .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    let image = RgbImage::from_raw(condition.width as u32, condition.height as u32, raw)
        .ok_or_else(|| std::io::Error::other("failed to build DINO condition image buffer"))?;
    let resized = DynamicImage::ImageRgb8(image)
        .resize_exact(
            image_size as u32,
            image_size as u32,
            image::imageops::FilterType::Triangle,
        )
        .to_rgb32f();
    Ok(resized.as_flat_samples().as_slice().to_vec())
}

fn preprocess_condition_alpha_values(
    condition: &ConditionImage2d,
    image_size: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let raw = condition
        .alpha_values()?
        .into_iter()
        .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    let image = GrayImage::from_raw(condition.width as u32, condition.height as u32, raw)
        .ok_or_else(|| std::io::Error::other("failed to build DINO alpha image buffer"))?;
    let resized = DynamicImage::ImageLuma8(image)
        .resize_exact(
            image_size as u32,
            image_size as u32,
            image::imageops::FilterType::Triangle,
        )
        .to_luma32f();
    Ok(resized.as_flat_samples().as_slice().to_vec())
}

fn normalize<B: Backend>(input: Tensor<B, 4>, device: &burn::tensor::Device<B>) -> Tensor<B, 4> {
    let mean: Tensor<B, 1> = Tensor::from_floats([0.485, 0.456, 0.406], device);
    let std: Tensor<B, 1> = Tensor::from_floats([0.229, 0.224, 0.225], device);
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
    use std::io::Cursor;

    #[test]
    fn transparent_black_artwork_survives_decode_and_preprocess() {
        let image = RgbaImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                Rgba([0, 0, 0, 0])
            } else {
                Rgba([0, 0, 0, 255])
            }
        });
        let mut bytes = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(image)
            .write_to(&mut bytes, ImageFormat::Png)
            .unwrap();
        let condition = decode_condition_image(bytes.get_ref()).unwrap();
        assert_eq!(condition.channels, 4);
        let values = preprocess_condition_values(&condition, 2).unwrap();
        assert!(values[0] > 0.95);
        assert!(values[3] < 0.05);
    }

    #[test]
    fn alpha_channel_preserves_patch_occupancy() {
        let condition = ConditionImage2d::from_rgba(
            2,
            2,
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        )
        .unwrap();
        let prepared = DinoVitsPreparedConditionBatch::from_conditions(&[condition], 2).unwrap();
        let device = burn::tensor::Device::<NdArray<f32>>::default();
        let alpha = alpha_token_grid_tensor::<NdArray<f32>>(&prepared, 2, 2, &device).unwrap();
        assert_eq!(alpha.dims(), [1, 5, 1]);
        let values = alpha.into_data().to_vec::<f32>().unwrap();
        assert!((values[0] - 0.5).abs() < 1.0e-6);
        assert_eq!(&values[1..], &[0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn rgb_channels_preserve_patch_color_and_layout() {
        let condition = ConditionImage2d::from_rgba(
            2,
            2,
            vec![
                1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ],
        )
        .unwrap();
        let prepared = DinoVitsPreparedConditionBatch::from_conditions(&[condition], 2).unwrap();
        let device = burn::tensor::Device::<NdArray<f32>>::default();
        let rgb = rgb_token_grid_tensor::<NdArray<f32>>(&prepared, 2, 2, &device).unwrap();
        assert_eq!(rgb.dims(), [1, 5, 3]);
        let values = rgb.into_data().to_vec::<f32>().unwrap();
        assert_eq!(&values[0..3], &[0.5, 0.5, 0.5]);
        assert_eq!(
            &values[3..],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        );
    }

    #[test]
    fn patch_pixels_preserve_native_pixels_in_dino_token_order() {
        let mut values = Vec::new();
        let mut alpha_values = Vec::new();
        for pixel in 0..16 {
            values.extend([pixel as f32, 0.0, 0.0]);
            alpha_values.push(1.0);
        }
        let prepared = DinoVitsPreparedConditionBatch {
            values,
            alpha_values,
            batch: 1,
            image_size: 4,
            input_channels: 3,
        };
        let device = burn::tensor::Device::<NdArray<f32>>::default();
        let pixels = patch_pixel_token_grid_tensor::<NdArray<f32>>(
            &prepared,
            2,
            2,
            PatchPixelChannels {
                rgb_scale: Some(1.0),
                alpha_scale: Some(1.0),
            },
            &device,
        )
        .unwrap();
        assert_eq!(pixels.dims(), [1, 5, 16]);
        let values = pixels.into_data().to_vec::<f32>().unwrap();
        let first_patch = &values[16..32];
        assert_eq!(
            first_patch,
            &[
                0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 4.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0, 1.0,
            ]
        );
        let last_patch = &values[64..80];
        assert_eq!(
            last_patch,
            &[
                10.0, 0.0, 0.0, 1.0, 11.0, 0.0, 0.0, 1.0, 14.0, 0.0, 0.0, 1.0, 15.0, 0.0, 0.0, 1.0,
            ]
        );
    }
}
