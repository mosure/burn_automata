use crate::{AutomataError, AutomataResult};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlphaAwareImageMetrics {
    pub raw_rgb_mse: f32,
    pub composited_rgb_mse: f32,
    pub density_mse: f32,
    pub density_soft_iou: f32,
}

/// CPU reference for the alpha-aware validation metric used by HyperNPA.
///
/// RGB buffers are premultiplied by their corresponding density buffers. The
/// reference is intentionally small and backend-independent so GPU validation
/// implementations can be parity-tested against it.
pub fn alpha_aware_image_metrics(
    predicted_rgb: &[f32],
    predicted_density: &[f32],
    target_rgb: &[f32],
    target_density: &[f32],
    background: [f32; 3],
) -> AutomataResult<AlphaAwareImageMetrics> {
    let pixels = predicted_density.len();
    if pixels == 0
        || target_density.len() != pixels
        || predicted_rgb.len() != pixels * 3
        || target_rgb.len() != pixels * 3
    {
        return Err(AutomataError::InvalidArgument(
            "alpha-aware image metric dimensions are inconsistent".to_string(),
        ));
    }
    if !predicted_rgb
        .iter()
        .chain(predicted_density)
        .chain(target_rgb)
        .chain(target_density)
        .chain(background.iter())
        .all(|value| value.is_finite())
    {
        return Err(AutomataError::InvalidArgument(
            "alpha-aware image metric input contains non-finite values".to_string(),
        ));
    }

    let mut raw_rgb_squared = 0.0_f32;
    let mut composited_rgb_squared = 0.0_f32;
    let mut density_squared = 0.0_f32;
    let mut intersection = 0.0_f32;
    let mut union = 0.0_f32;
    for pixel in 0..pixels {
        let predicted_alpha = predicted_density[pixel].clamp(0.0, 1.0);
        let target_alpha = target_density[pixel].clamp(0.0, 1.0);
        intersection += predicted_alpha.min(target_alpha);
        union += predicted_alpha.max(target_alpha);
        let density_diff = predicted_alpha - target_alpha;
        density_squared += density_diff * density_diff;
        for (channel, background_value) in background.iter().copied().enumerate() {
            let offset = pixel * 3 + channel;
            let raw_diff = predicted_rgb[offset] - target_rgb[offset];
            raw_rgb_squared += raw_diff * raw_diff;
            let predicted_composited = (predicted_rgb[offset]
                + background_value * (1.0 - predicted_alpha))
                .clamp(0.0, 1.0);
            let target_composited =
                (target_rgb[offset] + background_value * (1.0 - target_alpha)).clamp(0.0, 1.0);
            let composited_diff = predicted_composited - target_composited;
            composited_rgb_squared += composited_diff * composited_diff;
        }
    }
    Ok(AlphaAwareImageMetrics {
        raw_rgb_mse: raw_rgb_squared / (pixels * 3) as f32,
        composited_rgb_mse: composited_rgb_squared / (pixels * 3) as f32,
        density_mse: density_squared / pixels as f32,
        density_soft_iou: intersection / union.max(1.0e-8),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_black_prediction_cannot_match_opaque_black_target() {
        let metrics =
            alpha_aware_image_metrics(&[0.0, 0.0, 0.0], &[0.0], &[0.0, 0.0, 0.0], &[1.0], [1.0; 3])
                .unwrap();
        assert_eq!(metrics.raw_rgb_mse, 0.0);
        assert_eq!(metrics.composited_rgb_mse, 1.0);
        assert_eq!(metrics.density_mse, 1.0);
        assert_eq!(metrics.density_soft_iou, 0.0);
    }

    #[test]
    fn identical_premultiplied_images_match() {
        let metrics =
            alpha_aware_image_metrics(&[0.1, 0.2, 0.3], &[0.5], &[0.1, 0.2, 0.3], &[0.5], [1.0; 3])
                .unwrap();
        assert_eq!(metrics.raw_rgb_mse, 0.0);
        assert_eq!(metrics.composited_rgb_mse, 0.0);
        assert_eq!(metrics.density_mse, 0.0);
        assert_eq!(metrics.density_soft_iou, 1.0);
    }
}
