use super::{EPS, RenderLossConfig, RenderViewLossReport, RenderViewPreset, math::psnr};
use crate::render_loss::projection::{ProjectedPoint, view_basis};

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct RenderedPixel {
    color: [f32; 3],
    alpha: f32,
    depth_sum: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct RenderedPixelAdjoint {
    color: [f32; 3],
    alpha: f32,
    depth_sum: f32,
}

pub(super) fn splat_projected(
    positions: &[ProjectedPoint],
    colors: &[[f32; 3]],
    opacities: &[f32],
    sigmas: &[f32],
    cfg: RenderLossConfig,
) -> Vec<RenderedPixel> {
    debug_assert_eq!(positions.len(), colors.len());
    debug_assert_eq!(positions.len(), opacities.len());
    debug_assert_eq!(positions.len(), sigmas.len());
    let size = cfg.image_size;
    let mut out = vec![RenderedPixel::default(); size * size];
    let norm_scale = 1.0 / positions.len().max(1) as f32;

    for (((pos, color), opacity), sigma) in positions
        .iter()
        .zip(colors.iter())
        .zip(opacities.iter())
        .zip(sigmas.iter())
    {
        let sigma = sigma.clamp(cfg.min_sigma, cfg.max_sigma).max(EPS);
        let sigma2 = sigma * sigma;
        let radius = (5.0 * sigma).ceil().max(1.0) as isize;
        let px = (pos.x + cfg.world_scale) / (2.0 * cfg.world_scale) * (size as f32 - 1.0);
        let py = (cfg.world_scale - pos.y) / (2.0 * cfg.world_scale) * (size as f32 - 1.0);
        let base_x = px.floor() as isize;
        let base_y = py.floor() as isize;
        let frac_x = px - base_x as f32;
        let frac_y = py - base_y as f32;

        let mut weights = Vec::new();
        let mut weight_sum = 0.0_f32;
        for oy in -radius..=radius {
            for ox in -radius..=radius {
                let x = base_x + ox;
                let y = base_y + oy;
                if x < 0 || y < 0 || x >= size as isize || y >= size as isize {
                    continue;
                }
                let dx = ox as f32 - frac_x;
                let dy = oy as f32 - frac_y;
                let w = (-(dx * dx + dy * dy) / (2.0 * sigma2)).exp();
                weights.push((x as usize, y as usize, w));
                weight_sum += w;
            }
        }

        let denom = weight_sum.max(EPS);
        let opacity = opacity.clamp(0.0, 1.0);
        for (x, y, w) in weights {
            let w = w / denom * norm_scale * opacity;
            let pixel = &mut out[y * size + x];
            pixel.color[0] += color[0] * w;
            pixel.color[1] += color[1] * w;
            pixel.color[2] += color[2] * w;
            pixel.alpha += w;
            pixel.depth_sum += pos.depth * w;
        }
    }

    out
}

pub(super) fn image_loss(
    view: RenderViewPreset,
    particle_image: &[RenderedPixel],
    target_image: &[RenderedPixel],
    cfg: RenderLossConfig,
) -> RenderViewLossReport {
    image_loss_with_adjoint(view, particle_image, target_image, cfg, 0.0).0
}

pub(super) fn image_loss_with_adjoint(
    view: RenderViewPreset,
    particle_image: &[RenderedPixel],
    target_image: &[RenderedPixel],
    cfg: RenderLossConfig,
    loss_scale: f32,
) -> (RenderViewLossReport, Vec<RenderedPixelAdjoint>) {
    debug_assert_eq!(particle_image.len(), target_image.len());
    let mut density_loss = 0.0_f32;
    let mut target_density_energy = 0.0_f32;
    let mut color_loss = 0.0_f32;
    let mut depth_loss = 0.0_f32;
    let mut gated_color_weight = 0.0_f32;
    let mut gated_depth_weight = 0.0_f32;
    let mut nonzero_target = 0usize;
    let mut nonzero_particle = 0usize;

    for (actual, target) in particle_image.iter().zip(target_image.iter()) {
        let actual_alpha = actual.alpha.max(0.0);
        let target_alpha = target.alpha.max(0.0);
        if target_alpha > 1.0e-5 {
            nonzero_target += 1;
        }
        if actual_alpha > 1.0e-5 {
            nonzero_particle += 1;
        }

        let density_diff = actual_alpha - target_alpha;
        density_loss += density_diff * density_diff;
        target_density_energy += target_alpha * target_alpha;

        let density_match = 1.0
            - (actual_alpha - target_alpha).abs() / (actual_alpha + target_alpha + EPS).max(EPS);
        let color_gate = density_match.clamp(0.0, 1.0);
        gated_color_weight += color_gate;
        for channel in 0..3 {
            let actual_color = (actual.color[channel] / actual_alpha.max(EPS)).clamp(0.0, 1.0);
            let target_color = (target.color[channel] / target_alpha.max(EPS)).clamp(0.0, 1.0);
            let color_diff = actual_color - target_color;
            color_loss += color_gate * color_diff * color_diff;
        }
        if actual_alpha > 1.0e-5 && target_alpha > 1.0e-5 {
            let actual_depth = (actual.depth_sum / actual_alpha.max(EPS)).clamp(0.0, 1.0);
            let target_depth = (target.depth_sum / target_alpha.max(EPS)).clamp(0.0, 1.0);
            let depth_diff = actual_depth - target_depth;
            depth_loss += color_gate * depth_diff * depth_diff;
            gated_depth_weight += color_gate;
        }
    }

    let pixels = particle_image.len().max(1) as f32;
    let density_mse = density_loss / target_density_energy.max(EPS);
    let color_mse = color_loss / (gated_color_weight.max(EPS) * 3.0);
    let depth_mse = depth_loss / gated_depth_weight.max(EPS);
    let total_loss = cfg.density_weight * density_mse
        + cfg.color_weight * color_mse
        + cfg.depth_weight * depth_mse;

    let mut adjoints = vec![RenderedPixelAdjoint::default(); particle_image.len()];
    if loss_scale != 0.0 {
        let density_den = target_density_energy.max(EPS);
        let color_den = gated_color_weight.max(EPS) * 3.0;
        let depth_den = gated_depth_weight.max(EPS);
        let color_loss_per_gate = color_loss / gated_color_weight.max(EPS);
        let depth_loss_per_gate = depth_loss / gated_depth_weight.max(EPS);
        for ((actual, target), adjoint) in particle_image
            .iter()
            .zip(target_image.iter())
            .zip(adjoints.iter_mut())
        {
            let actual_alpha = actual.alpha.max(0.0);
            let target_alpha = target.alpha.max(0.0);
            let alpha_den = actual_alpha.max(EPS);
            let target_alpha_den = target_alpha.max(EPS);
            let density_diff = actual_alpha - target_alpha;
            adjoint.alpha += loss_scale * cfg.density_weight * 2.0 * density_diff / density_den;

            let density_match = 1.0
                - (actual_alpha - target_alpha).abs()
                    / (actual_alpha + target_alpha + EPS).max(EPS);
            let color_gate = density_match.clamp(0.0, 1.0);
            let gate_alpha_gradient =
                density_gate_alpha_gradient(actual_alpha, target_alpha, density_match);
            let mut color_error_sum = 0.0_f32;
            for channel in 0..3 {
                let raw_actual_color = actual.color[channel] / alpha_den;
                let raw_target_color = target.color[channel] / target_alpha_den;
                let actual_color = raw_actual_color.clamp(0.0, 1.0);
                let target_color = raw_target_color.clamp(0.0, 1.0);
                let color_diff = actual_color - target_color;
                color_error_sum += color_diff * color_diff;
                if raw_actual_color > 0.0 && raw_actual_color < 1.0 && actual_alpha > EPS {
                    let coeff =
                        loss_scale * cfg.color_weight * color_gate * 2.0 * color_diff / color_den;
                    adjoint.color[channel] += coeff / alpha_den;
                    adjoint.alpha -= coeff * actual.color[channel] / (alpha_den * alpha_den);
                }
            }
            if gate_alpha_gradient != 0.0 && gated_color_weight > EPS {
                let color_gate_coeff = cfg.color_weight * (color_error_sum - color_loss_per_gate)
                    / gated_color_weight.max(EPS)
                    / 3.0;
                adjoint.alpha += loss_scale * color_gate_coeff * gate_alpha_gradient;
            }

            if actual_alpha > 1.0e-5 && target_alpha > 1.0e-5 {
                let raw_actual_depth = actual.depth_sum / alpha_den;
                let raw_target_depth = target.depth_sum / target_alpha_den;
                let actual_depth = raw_actual_depth.clamp(0.0, 1.0);
                let target_depth = raw_target_depth.clamp(0.0, 1.0);
                let depth_diff = actual_depth - target_depth;
                let depth_error = depth_diff * depth_diff;
                if raw_actual_depth > 0.0 && raw_actual_depth < 1.0 {
                    let coeff =
                        loss_scale * cfg.depth_weight * color_gate * 2.0 * depth_diff / depth_den;
                    adjoint.depth_sum += coeff / alpha_den;
                    adjoint.alpha -= coeff * actual.depth_sum / (alpha_den * alpha_den);
                }
                if gate_alpha_gradient != 0.0 && gated_depth_weight > EPS {
                    let depth_gate_coeff = cfg.depth_weight * (depth_error - depth_loss_per_gate)
                        / gated_depth_weight.max(EPS);
                    adjoint.alpha += loss_scale * depth_gate_coeff * gate_alpha_gradient;
                }
            }
        }
    }

    (
        RenderViewLossReport {
            view,
            total_loss,
            density_mse,
            color_mse,
            depth_mse,
            density_psnr_db: psnr(density_mse, 1.0),
            color_psnr_db: psnr(color_mse, 1.0),
            depth_psnr_db: psnr(depth_mse, 1.0),
            nonzero_target_alpha_fraction: nonzero_target as f32 / pixels,
            nonzero_particle_alpha_fraction: nonzero_particle as f32 / pixels,
        },
        adjoints,
    )
}

fn density_gate_alpha_gradient(
    actual_alpha: f32,
    target_alpha: f32,
    unclamped_density_match: f32,
) -> f32 {
    if unclamped_density_match <= 0.0
        || unclamped_density_match >= 1.0
        || actual_alpha <= 0.0
        || target_alpha <= 0.0
    {
        return 0.0;
    }
    let den = (actual_alpha + target_alpha + EPS).max(EPS);
    let delta = actual_alpha - target_alpha;
    if delta > 0.0 {
        -(2.0 * target_alpha + EPS) / (den * den)
    } else if delta < 0.0 {
        (2.0 * target_alpha + EPS) / (den * den)
    } else {
        0.0
    }
}

pub(super) fn accumulate_projected_position_gradients(
    view: RenderViewPreset,
    projected: &[ProjectedPoint],
    colors: &[[f32; 3]],
    opacities: &[f32],
    sigmas: &[f32],
    scale_logit_derivatives: &[f32],
    row_indices: &[usize],
    pixel_adjoints: &[RenderedPixelAdjoint],
    cfg: RenderLossConfig,
    gradients: &mut [[f32; 3]],
    opacity_gradients: &mut [f32],
    scale_gradients: &mut [f32],
    color_gradients: &mut [[f32; 3]],
) {
    debug_assert_eq!(projected.len(), colors.len());
    debug_assert_eq!(projected.len(), opacities.len());
    debug_assert_eq!(projected.len(), sigmas.len());
    debug_assert_eq!(projected.len(), scale_logit_derivatives.len());
    debug_assert_eq!(row_indices.len(), gradients.len());
    debug_assert_eq!(row_indices.len(), opacity_gradients.len());
    debug_assert_eq!(row_indices.len(), scale_gradients.len());
    debug_assert_eq!(row_indices.len(), color_gradients.len());
    let size = cfg.image_size;
    let norm_scale = 1.0 / projected.len().max(1) as f32;
    let pixel_scale = (size as f32 - 1.0) / (2.0 * cfg.world_scale.max(EPS));
    let depth_scale = 0.5 / cfg.world_scale.max(EPS);
    let (right, up, forward) = view_basis(view);

    for (gradient_idx, &row) in row_indices.iter().enumerate() {
        let gradient = &mut gradients[gradient_idx];
        let opacity_gradient = &mut opacity_gradients[gradient_idx];
        let scale_gradient = &mut scale_gradients[gradient_idx];
        let color_gradient = &mut color_gradients[gradient_idx];
        let pos = projected[row];
        let color = colors[row];
        let opacity = opacities[row].clamp(0.0, 1.0);
        let sigma = sigmas[row].clamp(cfg.min_sigma, cfg.max_sigma).max(EPS);
        let sigma2 = sigma * sigma;
        let sigma3 = (sigma2 * sigma).max(EPS);
        let radius = (5.0 * sigma).ceil().max(1.0) as isize;
        let px = (pos.x + cfg.world_scale) / (2.0 * cfg.world_scale) * (size as f32 - 1.0);
        let py = (cfg.world_scale - pos.y) / (2.0 * cfg.world_scale) * (size as f32 - 1.0);
        let base_x = px.floor() as isize;
        let base_y = py.floor() as isize;
        let frac_x = px - base_x as f32;
        let frac_y = py - base_y as f32;

        let mut taps = Vec::new();
        let mut weight_sum = 0.0_f32;
        let mut dsum_dpx = 0.0_f32;
        let mut dsum_dpy = 0.0_f32;
        let mut dsum_dsigma = 0.0_f32;
        for oy in -radius..=radius {
            for ox in -radius..=radius {
                let x = base_x + ox;
                let y = base_y + oy;
                if x < 0 || y < 0 || x >= size as isize || y >= size as isize {
                    continue;
                }
                let dx = ox as f32 - frac_x;
                let dy = oy as f32 - frac_y;
                let d2 = dx * dx + dy * dy;
                let raw = (-(dx * dx + dy * dy) / (2.0 * sigma2)).exp();
                let draw_dpx = raw * dx / sigma2;
                let draw_dpy = raw * dy / sigma2;
                let draw_dsigma = raw * d2 / sigma3;
                taps.push((
                    y as usize * size + x as usize,
                    raw,
                    draw_dpx,
                    draw_dpy,
                    draw_dsigma,
                ));
                weight_sum += raw;
                dsum_dpx += draw_dpx;
                dsum_dpy += draw_dpy;
                dsum_dsigma += draw_dsigma;
            }
        }

        let denom = weight_sum.max(EPS);
        let denom2 = denom * denom;
        let mut dloss_dpx = 0.0_f32;
        let mut dloss_dpy = 0.0_f32;
        let mut dloss_ddepth = 0.0_f32;
        let mut dloss_dopacity = 0.0_f32;
        let mut dloss_dsigma = 0.0_f32;
        for (pixel_idx, raw, draw_dpx, draw_dpy, draw_dsigma) in taps {
            let w = raw / denom * norm_scale * opacity;
            let adjoint = pixel_adjoints[pixel_idx];
            let dloss_dw = adjoint.alpha
                + adjoint.color[0] * color[0]
                + adjoint.color[1] * color[1]
                + adjoint.color[2] * color[2]
                + adjoint.depth_sum * pos.depth;
            dloss_ddepth += adjoint.depth_sum * w;
            dloss_dopacity += dloss_dw * raw / denom * norm_scale;
            let dw_dpx = norm_scale * opacity * (draw_dpx * denom - raw * dsum_dpx) / denom2;
            let dw_dpy = norm_scale * opacity * (draw_dpy * denom - raw * dsum_dpy) / denom2;
            let dw_dsigma =
                norm_scale * opacity * (draw_dsigma * denom - raw * dsum_dsigma) / denom2;
            dloss_dpx += dloss_dw * dw_dpx;
            dloss_dpy += dloss_dw * dw_dpy;
            dloss_dsigma += dloss_dw * dw_dsigma;
            for (channel, color_gradient) in color_gradient.iter_mut().enumerate() {
                *color_gradient += adjoint.color[channel] * w;
            }
        }

        let dloss_dproj_x = dloss_dpx * pixel_scale;
        let dloss_dproj_y = -dloss_dpy * pixel_scale;
        let dloss_ddepth_unclamped = if pos.depth > 0.0 && pos.depth < 1.0 {
            dloss_ddepth * depth_scale
        } else {
            0.0
        };
        for axis in 0..3 {
            gradient[axis] += dloss_dproj_x * right[axis]
                + dloss_dproj_y * up[axis]
                + dloss_ddepth_unclamped * forward[axis];
        }
        *opacity_gradient += dloss_dopacity;
        *scale_gradient += dloss_dsigma * scale_logit_derivatives[row];
    }
}
