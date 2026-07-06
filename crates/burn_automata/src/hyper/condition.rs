use serde::{Deserialize, Serialize};

use crate::{AutomataError, AutomataResult};

pub const CONDITION_FEATURE_DIMS: usize = 17;
pub const CONDITION_TOKEN_FEATURE_DIMS: usize = CONDITION_FEATURE_DIMS + 4;
pub const DEFAULT_CONDITION_TOKEN_GRID_WIDTH: usize = 4;
pub const DEFAULT_CONDITION_TOKEN_GRID_HEIGHT: usize = 4;
pub const DINO_VITS_EMBED_DIMS: usize = 384;
pub const DINO_VITS_CLS_PATCH_MEAN_FEATURE_DIMS: usize = 768;
pub const DINO_VITS_PATCH_STATS_FEATURE_DIMS: usize = DINO_VITS_EMBED_DIMS * 5;
pub const DEFAULT_DINO_VITS_TOKEN_GRID_WIDTH: usize = 8;
pub const DEFAULT_DINO_VITS_TOKEN_GRID_HEIGHT: usize = 8;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ConditionEncoder2d {
    #[default]
    SummaryTokens,
    DinoVitsClsPatchMean,
    DinoVitsPatchStats,
    DinoVitsTokenGrid,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConditionImage2d {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub values: Vec<f32>,
    #[serde(
        default,
        alias = "dino_vits_cls_patch_mean",
        skip_serializing_if = "Option::is_none"
    )]
    pub dino_vits_features: Option<Vec<f32>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConditionSummary2d {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub mean_luma: f32,
    pub variance_luma: f32,
    pub min_luma: f32,
    pub max_luma: f32,
    pub mean_rgb: [f32; 3],
    pub variance_rgb: [f32; 3],
    pub center_of_mass: [f32; 2],
    pub occupancy: f32,
    pub edge_energy: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConditionToken2d {
    pub center: [f32; 2],
    pub size: [f32; 2],
    pub features: Vec<f32>,
}

impl ConditionImage2d {
    pub fn from_luma(width: usize, height: usize, values: Vec<f32>) -> AutomataResult<Self> {
        Self::new(width, height, 1, values)
    }

    pub fn from_rgb(width: usize, height: usize, values: Vec<f32>) -> AutomataResult<Self> {
        Self::new(width, height, 3, values)
    }

    pub fn new(
        width: usize,
        height: usize,
        channels: usize,
        values: Vec<f32>,
    ) -> AutomataResult<Self> {
        let image = Self {
            width,
            height,
            channels,
            values,
            dino_vits_features: None,
        };
        image.validate()?;
        Ok(image)
    }

    pub fn validate(&self) -> AutomataResult<()> {
        if self.width == 0 || self.height == 0 {
            return Err(AutomataError::InvalidArgument(format!(
                "condition image dimensions must be positive, got {}x{}",
                self.width, self.height
            )));
        }
        if !matches!(self.channels, 1 | 3 | 4) {
            return Err(AutomataError::InvalidArgument(format!(
                "condition image channels must be 1, 3, or 4, got {}",
                self.channels
            )));
        }
        let expected = self.width * self.height * self.channels;
        if self.values.len() != expected {
            return Err(AutomataError::InvalidArgument(format!(
                "condition image values len {} != {expected}",
                self.values.len()
            )));
        }
        if !self.values.iter().all(|value| value.is_finite()) {
            return Err(AutomataError::InvalidArgument(
                "condition image contains non-finite values".to_string(),
            ));
        }
        if let Some(features) = &self.dino_vits_features
            && (!features.iter().all(|value| value.is_finite())
                || !matches!(
                    features.len(),
                    DINO_VITS_CLS_PATCH_MEAN_FEATURE_DIMS | DINO_VITS_PATCH_STATS_FEATURE_DIMS
                ) && !is_valid_dino_token_grid_feature_len(features.len()))
        {
            return Err(AutomataError::InvalidArgument(format!(
                "DINO condition feature vector len {} must be {DINO_VITS_CLS_PATCH_MEAN_FEATURE_DIMS}, {DINO_VITS_PATCH_STATS_FEATURE_DIMS}, or CLS plus an integer DINO token grid with finite values",
                features.len()
            )));
        }
        Ok(())
    }

    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }

    pub fn summary(&self) -> AutomataResult<ConditionSummary2d> {
        self.validate()?;
        let mut sum_luma = 0.0_f32;
        let mut sum_luma_sq = 0.0_f32;
        let mut min_luma = f32::INFINITY;
        let mut max_luma = f32::NEG_INFINITY;
        let mut sum_rgb = [0.0_f32; 3];
        let mut sum_rgb_sq = [0.0_f32; 3];
        let mut weighted_x = 0.0_f32;
        let mut weighted_y = 0.0_f32;
        let mut weight_sum = 0.0_f32;
        let mut occupied = 0_usize;
        let mut edge_energy = 0.0_f32;

        for y in 0..self.height {
            for x in 0..self.width {
                let rgb = self.rgb_at(x, y);
                let luma = luma(rgb);
                sum_luma += luma;
                sum_luma_sq += luma * luma;
                min_luma = min_luma.min(luma);
                max_luma = max_luma.max(luma);
                for c in 0..3 {
                    sum_rgb[c] += rgb[c];
                    sum_rgb_sq[c] += rgb[c] * rgb[c];
                }
                if luma > 0.05 {
                    occupied += 1;
                }
                let positive_luma = luma.max(0.0);
                weighted_x += normalized_coord(x, self.width) * positive_luma;
                weighted_y += normalized_coord(y, self.height) * positive_luma;
                weight_sum += positive_luma;
                if x + 1 < self.width {
                    edge_energy += (luma - self.luma_at(x + 1, y)).abs();
                }
                if y + 1 < self.height {
                    edge_energy += (luma - self.luma_at(x, y + 1)).abs();
                }
            }
        }

        let pixels = self.pixel_count() as f32;
        let mean_luma = sum_luma / pixels;
        let variance_luma = (sum_luma_sq / pixels - mean_luma * mean_luma).max(0.0);
        let mut mean_rgb = [0.0; 3];
        let mut variance_rgb = [0.0; 3];
        for c in 0..3 {
            mean_rgb[c] = sum_rgb[c] / pixels;
            variance_rgb[c] = (sum_rgb_sq[c] / pixels - mean_rgb[c] * mean_rgb[c]).max(0.0);
        }
        let center_of_mass = if weight_sum > 0.0 {
            [weighted_x / weight_sum, weighted_y / weight_sum]
        } else {
            [0.0, 0.0]
        };
        let edge_denominator = ((self.width.saturating_sub(1) * self.height)
            + (self.height.saturating_sub(1) * self.width))
            .max(1) as f32;

        Ok(ConditionSummary2d {
            width: self.width,
            height: self.height,
            channels: self.channels,
            mean_luma,
            variance_luma,
            min_luma,
            max_luma,
            mean_rgb,
            variance_rgb,
            center_of_mass,
            occupancy: occupied as f32 / pixels,
            edge_energy: edge_energy / edge_denominator,
        })
    }

    pub fn feature_vector(&self) -> AutomataResult<Vec<f32>> {
        self.summary().map(|summary| summary.feature_vector())
    }

    pub fn feature_vector_with_tokens(
        &self,
        grid_width: usize,
        grid_height: usize,
    ) -> AutomataResult<Vec<f32>> {
        if grid_width == 0 && grid_height == 0 {
            return self.feature_vector();
        }
        if grid_width == 0 || grid_height == 0 {
            return Err(AutomataError::InvalidArgument(format!(
                "condition token grid must be either disabled or positive in both dimensions, got {grid_width}x{grid_height}"
            )));
        }

        let mut features = self.feature_vector()?;
        let tokens = self.pooled_tokens(grid_width, grid_height)?;
        features.reserve(tokens.len() * CONDITION_TOKEN_FEATURE_DIMS);
        for token in tokens {
            features.extend_from_slice(&token.center);
            features.extend_from_slice(&token.size);
            features.extend_from_slice(&token.features);
        }
        Ok(features)
    }

    pub fn feature_vector_for_encoder(
        &self,
        encoder: ConditionEncoder2d,
        grid_width: usize,
        grid_height: usize,
    ) -> AutomataResult<Vec<f32>> {
        match encoder {
            ConditionEncoder2d::SummaryTokens => {
                self.feature_vector_with_tokens(grid_width, grid_height)
            }
            ConditionEncoder2d::DinoVitsClsPatchMean
            | ConditionEncoder2d::DinoVitsPatchStats
            | ConditionEncoder2d::DinoVitsTokenGrid => {
                self.dino_vits_features.clone().ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "condition is missing DINO vits features".to_string(),
                    )
                })
            }
        }
    }

    pub fn with_dino_vits_features(mut self, features: Vec<f32>) -> AutomataResult<Self> {
        self.dino_vits_features = Some(features);
        self.validate()?;
        Ok(self)
    }

    pub fn with_dino_vits_cls_patch_mean(self, features: Vec<f32>) -> AutomataResult<Self> {
        self.with_dino_vits_features(features)
    }

    pub fn pooled_tokens(
        &self,
        grid_width: usize,
        grid_height: usize,
    ) -> AutomataResult<Vec<ConditionToken2d>> {
        self.validate()?;
        if grid_width == 0 || grid_height == 0 {
            return Err(AutomataError::InvalidArgument(format!(
                "condition token grid must be positive, got {grid_width}x{grid_height}"
            )));
        }
        let mut tokens = Vec::with_capacity(grid_width * grid_height);
        for tile_y in 0..grid_height {
            for tile_x in 0..grid_width {
                let x0 = tile_x * self.width / grid_width;
                let x1 = ((tile_x + 1) * self.width / grid_width).max(x0 + 1);
                let y0 = tile_y * self.height / grid_height;
                let y1 = ((tile_y + 1) * self.height / grid_height).max(y0 + 1);
                let tile = self.crop_summary(x0, x1.min(self.width), y0, y1.min(self.height))?;
                tokens.push(ConditionToken2d {
                    center: [
                        ((tile_x as f32 + 0.5) / grid_width as f32) * 2.0 - 1.0,
                        ((tile_y as f32 + 0.5) / grid_height as f32) * 2.0 - 1.0,
                    ],
                    size: [2.0 / grid_width as f32, 2.0 / grid_height as f32],
                    features: tile.feature_vector(),
                });
            }
        }
        Ok(tokens)
    }

    fn crop_summary(
        &self,
        x0: usize,
        x1: usize,
        y0: usize,
        y1: usize,
    ) -> AutomataResult<ConditionSummary2d> {
        let width = x1.saturating_sub(x0);
        let height = y1.saturating_sub(y0);
        let mut values = Vec::with_capacity(width * height * self.channels);
        for y in y0..y1 {
            let start = (y * self.width + x0) * self.channels;
            let end = (y * self.width + x1) * self.channels;
            values.extend_from_slice(&self.values[start..end]);
        }
        Self::new(width, height, self.channels, values)?.summary()
    }

    fn rgb_at(&self, x: usize, y: usize) -> [f32; 3] {
        let offset = (y * self.width + x) * self.channels;
        match self.channels {
            1 => [self.values[offset]; 3],
            _ => [
                self.values[offset],
                self.values[offset + 1],
                self.values[offset + 2],
            ],
        }
    }

    fn luma_at(&self, x: usize, y: usize) -> f32 {
        luma(self.rgb_at(x, y))
    }
}

impl ConditionSummary2d {
    pub fn feature_vector(&self) -> Vec<f32> {
        let aspect = self.width as f32 / self.height as f32;
        let width_log = (self.width as f32).log2() / 16.0;
        let height_log = (self.height as f32).log2() / 16.0;
        vec![
            aspect,
            width_log,
            height_log,
            self.mean_luma,
            self.variance_luma,
            self.min_luma,
            self.max_luma,
            self.mean_rgb[0],
            self.mean_rgb[1],
            self.mean_rgb[2],
            self.variance_rgb[0],
            self.variance_rgb[1],
            self.variance_rgb[2],
            self.center_of_mass[0],
            self.center_of_mass[1],
            self.occupancy,
            self.edge_energy,
        ]
    }
}

fn normalized_coord(index: usize, len: usize) -> f32 {
    if len <= 1 {
        0.0
    } else {
        (index as f32 / (len - 1) as f32) * 2.0 - 1.0
    }
}

fn luma(rgb: [f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

pub fn condition_feature_dims_for_token_grid(
    grid_width: usize,
    grid_height: usize,
) -> AutomataResult<usize> {
    if grid_width == 0 && grid_height == 0 {
        return Ok(CONDITION_FEATURE_DIMS);
    }
    if grid_width == 0 || grid_height == 0 {
        return Err(AutomataError::InvalidArgument(format!(
            "condition token grid must be either disabled or positive in both dimensions, got {grid_width}x{grid_height}"
        )));
    }
    let token_count = grid_width.checked_mul(grid_height).ok_or_else(|| {
        AutomataError::InvalidArgument(format!(
            "condition token grid {grid_width}x{grid_height} overflows"
        ))
    })?;
    let token_dims = token_count
        .checked_mul(CONDITION_TOKEN_FEATURE_DIMS)
        .ok_or_else(|| {
            AutomataError::InvalidArgument(format!(
                "condition token grid {grid_width}x{grid_height} feature dims overflow"
            ))
        })?;
    CONDITION_FEATURE_DIMS
        .checked_add(token_dims)
        .ok_or_else(|| {
            AutomataError::InvalidArgument(format!(
                "condition token grid {grid_width}x{grid_height} feature dims overflow"
            ))
        })
}

pub fn condition_feature_dims_for_encoder(
    encoder: ConditionEncoder2d,
    grid_width: usize,
    grid_height: usize,
) -> AutomataResult<usize> {
    match encoder {
        ConditionEncoder2d::SummaryTokens => {
            condition_feature_dims_for_token_grid(grid_width, grid_height)
        }
        ConditionEncoder2d::DinoVitsClsPatchMean => Ok(DINO_VITS_CLS_PATCH_MEAN_FEATURE_DIMS),
        ConditionEncoder2d::DinoVitsPatchStats => Ok(DINO_VITS_PATCH_STATS_FEATURE_DIMS),
        ConditionEncoder2d::DinoVitsTokenGrid => {
            let grid_width = if grid_width == 0 {
                DEFAULT_DINO_VITS_TOKEN_GRID_WIDTH
            } else {
                grid_width
            };
            let grid_height = if grid_height == 0 {
                DEFAULT_DINO_VITS_TOKEN_GRID_HEIGHT
            } else {
                grid_height
            };
            let token_count = grid_width.checked_mul(grid_height).ok_or_else(|| {
                AutomataError::InvalidArgument(format!(
                    "DINO token grid {grid_width}x{grid_height} overflows"
                ))
            })?;
            let token_dims = token_count
                .checked_add(1)
                .and_then(|tokens| tokens.checked_mul(DINO_VITS_EMBED_DIMS))
                .ok_or_else(|| {
                    AutomataError::InvalidArgument(format!(
                        "DINO token grid {grid_width}x{grid_height} feature dims overflow"
                    ))
                })?;
            Ok(token_dims)
        }
    }
}

fn is_valid_dino_token_grid_feature_len(len: usize) -> bool {
    len > DINO_VITS_EMBED_DIMS
        && len.is_multiple_of(DINO_VITS_EMBED_DIMS)
        && (len / DINO_VITS_EMBED_DIMS).saturating_sub(1) > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dino_token_grid_dims_include_cls_token() {
        assert_eq!(
            condition_feature_dims_for_encoder(ConditionEncoder2d::DinoVitsTokenGrid, 8, 8)
                .unwrap(),
            (1 + 8 * 8) * DINO_VITS_EMBED_DIMS
        );
        assert_eq!(
            condition_feature_dims_for_encoder(ConditionEncoder2d::DinoVitsTokenGrid, 37, 37)
                .unwrap(),
            (1 + 37 * 37) * DINO_VITS_EMBED_DIMS
        );
    }

    #[test]
    fn dino_token_grid_feature_vectors_validate() {
        let dims = (1 + 2 * 3) * DINO_VITS_EMBED_DIMS;
        let image = ConditionImage2d::from_rgb(1, 1, vec![0.0, 0.0, 0.0])
            .unwrap()
            .with_dino_vits_features(vec![0.25; dims])
            .unwrap();
        let features = image
            .feature_vector_for_encoder(ConditionEncoder2d::DinoVitsTokenGrid, 2, 3)
            .unwrap();
        assert_eq!(features.len(), dims);
    }
}
