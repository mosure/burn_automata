#![allow(clippy::too_many_arguments)]

use serde::{Deserialize, Serialize};

use crate::{
    AutomataError, AutomataResult, RolloutTrace, TriangleMeshTarget,
    rollout::growth_3d_material_opacity_channel,
};
use burn_automata_kernels::GaussianDecodeMode;

const EPS: f32 = 1.0e-8;
const RENDER_MIN_OPACITY: f32 = 0.001;
const RENDER_MAX_OPACITY: f32 = 0.95;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderViewPreset {
    Xy,
    Xz,
    Yz,
    Iso,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct RenderLossConfig {
    pub image_size: usize,
    pub sigma: f32,
    pub min_sigma: f32,
    pub max_sigma: f32,
    pub gaussian_decode_mode: GaussianDecodeMode,
    pub world_scale: f32,
    pub target_samples: usize,
    pub opacity_logit_bias: f32,
    pub density_weight: f32,
    pub color_weight: f32,
    pub depth_weight: f32,
}

impl Default for RenderLossConfig {
    fn default() -> Self {
        Self {
            image_size: 64,
            sigma: 2.5,
            min_sigma: 0.75,
            max_sigma: 5.0,
            gaussian_decode_mode: GaussianDecodeMode::GaussianSh0FixedScale,
            world_scale: 1.25,
            target_samples: 8192,
            opacity_logit_bias: 0.0,
            density_weight: 1.0,
            color_weight: 1.0,
            depth_weight: 1.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiViewRenderLossReport {
    pub passed: bool,
    pub image_size: usize,
    pub target_samples: usize,
    pub total_loss: f32,
    pub density_mse: f32,
    pub color_mse: f32,
    pub depth_mse: f32,
    pub density_psnr_db: f32,
    pub color_psnr_db: f32,
    pub depth_psnr_db: f32,
    pub nonzero_target_alpha_fraction: f32,
    pub nonzero_particle_alpha_fraction: f32,
    pub views: Vec<RenderViewLossReport>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RenderViewLossReport {
    pub view: RenderViewPreset,
    pub total_loss: f32,
    pub density_mse: f32,
    pub color_mse: f32,
    pub depth_mse: f32,
    pub density_psnr_db: f32,
    pub color_psnr_db: f32,
    pub depth_psnr_db: f32,
    pub nonzero_target_alpha_fraction: f32,
    pub nonzero_particle_alpha_fraction: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiViewRenderPositionGradient {
    pub loss: MultiViewRenderLossReport,
    pub row_indices: Vec<usize>,
    pub gradients: Vec<[f32; 3]>,
    pub opacity_gradients: Vec<f32>,
    pub scale_gradients: Vec<f32>,
    pub color_gradients: Vec<[f32; 3]>,
}

pub fn mesh_multiview_render_loss_from_trace(
    trace: &RolloutTrace,
    target: &TriangleMeshTarget,
    cfg: RenderLossConfig,
) -> AutomataResult<MultiViewRenderLossReport> {
    validate_render_trace(trace, cfg)?;

    let particle_attrs = state_tail_render_attributes(&trace.states, trace.state_dims, cfg)?;
    let target_samples = mesh_surface_render_samples(target, cfg.target_samples);
    let target_opacities = vec![RENDER_MAX_OPACITY; target_samples.positions.len()];
    let target_sigmas = vec![cfg.sigma; target_samples.positions.len()];
    let views = [
        RenderViewPreset::Xy,
        RenderViewPreset::Xz,
        RenderViewPreset::Yz,
        RenderViewPreset::Iso,
    ];
    let mut reports = Vec::with_capacity(views.len());
    for view in views {
        reports.push(render_view_loss(
            view,
            &trace.positions,
            &particle_attrs.colors,
            &particle_attrs.opacities,
            &particle_attrs.sigmas,
            &target_samples.positions,
            &target_samples.colors,
            &target_opacities,
            &target_sigmas,
            cfg,
        )?);
    }

    Ok(combine_view_reports(reports, cfg))
}

pub fn mesh_multiview_render_position_gradient_from_trace(
    trace: &RolloutTrace,
    target: &TriangleMeshTarget,
    cfg: RenderLossConfig,
    max_particles: usize,
) -> AutomataResult<MultiViewRenderPositionGradient> {
    let rows = trace.particle_count.min(max_particles).max(1);
    let row_indices = (0..rows).collect::<Vec<_>>();
    mesh_multiview_render_position_gradient_for_rows_from_trace(trace, target, cfg, &row_indices)
}

pub fn mesh_multiview_render_position_gradient_for_rows_from_trace(
    trace: &RolloutTrace,
    target: &TriangleMeshTarget,
    cfg: RenderLossConfig,
    row_indices: &[usize],
) -> AutomataResult<MultiViewRenderPositionGradient> {
    validate_render_trace(trace, cfg)?;
    if row_indices.is_empty() {
        return Err(AutomataError::InvalidArgument(
            "render position gradient needs at least one row".to_string(),
        ));
    }
    if let Some(row) = row_indices
        .iter()
        .copied()
        .find(|&row| row >= trace.particle_count)
    {
        return Err(AutomataError::InvalidArgument(format!(
            "render position gradient row {row} out of range for {} particles",
            trace.particle_count
        )));
    }
    let particle_attrs = state_tail_render_attributes(&trace.states, trace.state_dims, cfg)?;
    let target_samples = mesh_surface_render_samples(target, cfg.target_samples);
    let target_opacities = vec![RENDER_MAX_OPACITY; target_samples.positions.len()];
    let target_sigmas = vec![cfg.sigma; target_samples.positions.len()];
    let views = [
        RenderViewPreset::Xy,
        RenderViewPreset::Xz,
        RenderViewPreset::Yz,
        RenderViewPreset::Iso,
    ];
    let mut gradients = vec![[0.0; 3]; row_indices.len()];
    let mut opacity_gradients = vec![0.0; row_indices.len()];
    let mut scale_gradients = vec![0.0; row_indices.len()];
    let mut color_gradients = vec![[0.0; 3]; row_indices.len()];
    let mut reports = Vec::with_capacity(views.len());
    let loss_scale = 1.0 / views.len() as f32;

    for view in views {
        let particle_projected = project_positions(view, &trace.positions, cfg.world_scale);
        let target_projected = project_positions(view, &target_samples.positions, cfg.world_scale);
        let particle_image = splat_projected(
            &particle_projected,
            &particle_attrs.colors,
            &particle_attrs.opacities,
            &particle_attrs.sigmas,
            cfg,
        );
        let target_image = splat_projected(
            &target_projected,
            &target_samples.colors,
            &target_opacities,
            &target_sigmas,
            cfg,
        );
        let (report, pixel_adjoints) =
            image_loss_with_adjoint(view, &particle_image, &target_image, cfg, loss_scale);
        accumulate_projected_position_gradients(
            view,
            &particle_projected,
            &particle_attrs.colors,
            &particle_attrs.opacities,
            &particle_attrs.sigmas,
            &particle_attrs.scale_logit_derivatives,
            row_indices,
            &pixel_adjoints,
            cfg,
            &mut gradients,
            &mut opacity_gradients,
            &mut scale_gradients,
            &mut color_gradients,
        );
        reports.push(report);
    }

    Ok(MultiViewRenderPositionGradient {
        loss: combine_view_reports(reports, cfg),
        row_indices: row_indices.to_vec(),
        gradients,
        opacity_gradients,
        scale_gradients,
        color_gradients,
    })
}

fn combine_view_reports(
    reports: Vec<RenderViewLossReport>,
    cfg: RenderLossConfig,
) -> MultiViewRenderLossReport {
    let count = reports.len().max(1) as f32;
    let density_mse = reports.iter().map(|report| report.density_mse).sum::<f32>() / count;
    let color_mse = reports.iter().map(|report| report.color_mse).sum::<f32>() / count;
    let depth_mse = reports.iter().map(|report| report.depth_mse).sum::<f32>() / count;
    let total_loss = cfg.density_weight * density_mse
        + cfg.color_weight * color_mse
        + cfg.depth_weight * depth_mse;
    let nonzero_target_alpha_fraction = reports
        .iter()
        .map(|report| report.nonzero_target_alpha_fraction)
        .sum::<f32>()
        / count;
    let nonzero_particle_alpha_fraction = reports
        .iter()
        .map(|report| report.nonzero_particle_alpha_fraction)
        .sum::<f32>()
        / count;
    let density_psnr_db = psnr(density_mse, 1.0);
    let color_psnr_db = psnr(color_mse, 1.0);
    let depth_psnr_db = psnr(depth_mse, 1.0);
    let finite = reports.iter().all(|report| {
        report.total_loss.is_finite()
            && report.density_mse.is_finite()
            && report.color_mse.is_finite()
            && report.depth_mse.is_finite()
            && report.nonzero_particle_alpha_fraction > 0.0
    });

    MultiViewRenderLossReport {
        passed: finite && density_psnr_db >= 10.0 && color_psnr_db >= 12.0 && depth_psnr_db >= 14.0,
        image_size: cfg.image_size,
        target_samples: cfg.target_samples,
        total_loss,
        density_mse,
        color_mse,
        depth_mse,
        density_psnr_db,
        color_psnr_db,
        depth_psnr_db,
        nonzero_target_alpha_fraction,
        nonzero_particle_alpha_fraction,
        views: reports,
    }
}

pub fn mesh_surface_render_samples(
    target: &TriangleMeshTarget,
    samples: usize,
) -> MeshRenderSamples {
    let samples = samples.max(1);
    let mut positions = Vec::with_capacity(samples);
    let mut colors = Vec::with_capacity(samples);
    for sample_idx in 0..samples {
        let sample = target.surface_sample(sample_idx);
        positions.push([
            sample.position[0],
            sample.position[1],
            sample.position[2],
            0.0,
        ]);
        colors.push(sample.color);
    }
    MeshRenderSamples { positions, colors }
}

#[derive(Clone, Debug)]
pub struct MeshRenderSamples {
    pub positions: Vec<[f32; 4]>,
    pub colors: Vec<[f32; 3]>,
}

fn render_view_loss(
    view: RenderViewPreset,
    particle_positions: &[[f32; 4]],
    particle_colors: &[[f32; 3]],
    particle_opacities: &[f32],
    particle_sigmas: &[f32],
    target_positions: &[[f32; 4]],
    target_colors: &[[f32; 3]],
    target_opacities: &[f32],
    target_sigmas: &[f32],
    cfg: RenderLossConfig,
) -> AutomataResult<RenderViewLossReport> {
    let particle_projected = project_positions(view, particle_positions, cfg.world_scale);
    let target_projected = project_positions(view, target_positions, cfg.world_scale);
    let particle_image = splat_projected(
        &particle_projected,
        particle_colors,
        particle_opacities,
        particle_sigmas,
        cfg,
    );
    let target_image = splat_projected(
        &target_projected,
        target_colors,
        target_opacities,
        target_sigmas,
        cfg,
    );
    Ok(image_loss(view, &particle_image, &target_image, cfg))
}

#[derive(Clone, Copy, Debug, Default)]
struct ProjectedPoint {
    x: f32,
    y: f32,
    depth: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct RenderedPixel {
    color: [f32; 3],
    alpha: f32,
    depth_sum: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct RenderedPixelAdjoint {
    color: [f32; 3],
    alpha: f32,
    depth_sum: f32,
}

fn splat_projected(
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

fn image_loss(
    view: RenderViewPreset,
    particle_image: &[RenderedPixel],
    target_image: &[RenderedPixel],
    cfg: RenderLossConfig,
) -> RenderViewLossReport {
    image_loss_with_adjoint(view, particle_image, target_image, cfg, 0.0).0
}

fn image_loss_with_adjoint(
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

fn accumulate_projected_position_gradients(
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

fn project_positions(
    view: RenderViewPreset,
    positions: &[[f32; 4]],
    world_scale: f32,
) -> Vec<ProjectedPoint> {
    let (right, up, forward) = view_basis(view);
    positions
        .iter()
        .map(|position| {
            let depth =
                (0.5 + 0.5 * dot3(*position, forward) / world_scale.max(EPS)).clamp(0.0, 1.0);
            ProjectedPoint {
                x: dot3(*position, right),
                y: dot3(*position, up),
                depth,
            }
        })
        .collect()
}

fn view_basis(view: RenderViewPreset) -> ([f32; 3], [f32; 3], [f32; 3]) {
    match view {
        RenderViewPreset::Xy => ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        RenderViewPreset::Xz => ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        RenderViewPreset::Yz => ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        RenderViewPreset::Iso => {
            let right = normalize3([1.0, -1.0, 0.0]);
            let up = normalize3([1.0, 1.0, 2.0]);
            let forward = normalize3(cross3(right, up));
            (right, up, forward)
        }
    }
}

#[derive(Clone, Debug)]
struct RenderParticleAttributes {
    colors: Vec<[f32; 3]>,
    opacities: Vec<f32>,
    sigmas: Vec<f32>,
    scale_logit_derivatives: Vec<f32>,
}

fn state_tail_render_attributes(
    states: &[f32],
    state_dims: usize,
    cfg: RenderLossConfig,
) -> AutomataResult<RenderParticleAttributes> {
    if state_dims < 3 {
        return Err(AutomataError::InvalidArgument(format!(
            "render loss needs at least 3 state channels for color, got {state_dims}"
        )));
    }
    if cfg.gaussian_decode_mode == GaussianDecodeMode::GaussianSh0LearnedScale && state_dims < 5 {
        return Err(AutomataError::InvalidArgument(format!(
            "learned-scale render decode needs at least 5 state channels, got {state_dims}"
        )));
    }
    if !states.len().is_multiple_of(state_dims) {
        return Err(AutomataError::InvalidArgument(format!(
            "state len {} is not divisible by state_dims {state_dims}",
            states.len()
        )));
    }
    let count = states.len() / state_dims;
    let tail = state_dims - 3;
    let mut colors = Vec::with_capacity(count);
    let mut opacities = Vec::with_capacity(count);
    let mut sigmas = Vec::with_capacity(count);
    let mut scale_logit_derivatives = Vec::with_capacity(count);
    for idx in 0..count {
        let base = idx * state_dims + tail;
        colors.push([
            (0.5 + 0.5 * states[base]).clamp(0.0, 1.0),
            (0.5 + 0.5 * states[base + 1]).clamp(0.0, 1.0),
            (0.5 + 0.5 * states[base + 2]).clamp(0.0, 1.0),
        ]);
        let opacity = if let Some(channel) = growth_3d_material_opacity_channel(state_dims) {
            sigmoid(states[idx * state_dims + channel] + cfg.opacity_logit_bias)
                .clamp(RENDER_MIN_OPACITY, RENDER_MAX_OPACITY)
        } else {
            RENDER_MAX_OPACITY
        };
        opacities.push(opacity);
        let (sigma, derivative) = match cfg.gaussian_decode_mode {
            GaussianDecodeMode::ParticlePoint => (cfg.min_sigma.max(EPS), 0.0),
            GaussianDecodeMode::GaussianSh0FixedScale | GaussianDecodeMode::GaussianSh0Oriented => {
                (cfg.sigma.clamp(cfg.min_sigma, cfg.max_sigma), 0.0)
            }
            GaussianDecodeMode::GaussianSh0LearnedScale => {
                let channel = state_dims - 5;
                let logit = states[idx * state_dims + channel].clamp(-8.0, 8.0);
                let raw_sigma = cfg.sigma * logit.exp();
                let sigma = raw_sigma.clamp(cfg.min_sigma, cfg.max_sigma);
                let derivative = if raw_sigma > cfg.min_sigma && raw_sigma < cfg.max_sigma {
                    raw_sigma
                } else {
                    0.0
                };
                (sigma, derivative)
            }
        };
        sigmas.push(sigma);
        scale_logit_derivatives.push(derivative);
    }
    Ok(RenderParticleAttributes {
        colors,
        opacities,
        sigmas,
        scale_logit_derivatives,
    })
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn validate_render_trace(trace: &RolloutTrace, cfg: RenderLossConfig) -> AutomataResult<()> {
    validate_render_loss_config(cfg)?;
    if trace.batch_size != 1 {
        return Err(AutomataError::InvalidArgument(format!(
            "render loss expects batch_size=1, got {}",
            trace.batch_size
        )));
    }
    if trace.positions.len() != trace.particle_count {
        return Err(AutomataError::InvalidArgument(format!(
            "render loss expects final trace positions for one batch, got {} for {} particles",
            trace.positions.len(),
            trace.particle_count
        )));
    }
    if trace.states.len() != trace.particle_count * trace.state_dims {
        return Err(AutomataError::InvalidArgument(format!(
            "render loss state len {} does not match particles {} * state_dims {}",
            trace.states.len(),
            trace.particle_count,
            trace.state_dims
        )));
    }
    Ok(())
}

fn validate_render_loss_config(cfg: RenderLossConfig) -> AutomataResult<()> {
    if cfg.image_size == 0 {
        return Err(AutomataError::InvalidArgument(
            "render loss image_size must be non-zero".to_string(),
        ));
    }
    if cfg.target_samples == 0 {
        return Err(AutomataError::InvalidArgument(
            "render loss target_samples must be non-zero".to_string(),
        ));
    }
    if !cfg.sigma.is_finite() || cfg.sigma <= 0.0 {
        return Err(AutomataError::InvalidArgument(format!(
            "render loss sigma must be finite and positive, got {}",
            cfg.sigma
        )));
    }
    if !cfg.min_sigma.is_finite() || cfg.min_sigma <= 0.0 {
        return Err(AutomataError::InvalidArgument(format!(
            "render loss min_sigma must be finite and positive, got {}",
            cfg.min_sigma
        )));
    }
    if !cfg.max_sigma.is_finite() || cfg.max_sigma < cfg.min_sigma {
        return Err(AutomataError::InvalidArgument(format!(
            "render loss max_sigma must be finite and >= min_sigma, got max={} min={}",
            cfg.max_sigma, cfg.min_sigma
        )));
    }
    if !cfg.world_scale.is_finite() || cfg.world_scale <= 0.0 {
        return Err(AutomataError::InvalidArgument(format!(
            "render loss world_scale must be finite and positive, got {}",
            cfg.world_scale
        )));
    }
    if !cfg.opacity_logit_bias.is_finite() {
        return Err(AutomataError::InvalidArgument(format!(
            "render loss opacity_logit_bias must be finite, got {}",
            cfg.opacity_logit_bias
        )));
    }
    if !cfg.density_weight.is_finite() || cfg.density_weight < 0.0 {
        return Err(AutomataError::InvalidArgument(format!(
            "render loss density_weight must be finite and non-negative, got {}",
            cfg.density_weight
        )));
    }
    if !cfg.color_weight.is_finite() || cfg.color_weight < 0.0 {
        return Err(AutomataError::InvalidArgument(format!(
            "render loss color_weight must be finite and non-negative, got {}",
            cfg.color_weight
        )));
    }
    if !cfg.depth_weight.is_finite() || cfg.depth_weight < 0.0 {
        return Err(AutomataError::InvalidArgument(format!(
            "render loss depth_weight must be finite and non-negative, got {}",
            cfg.depth_weight
        )));
    }
    Ok(())
}

fn psnr(mse: f32, peak: f32) -> f32 {
    if mse <= EPS {
        f32::INFINITY
    } else {
        10.0 * ((peak * peak) / mse.max(EPS)).log10()
    }
}

fn dot3(lhs: [f32; 4], rhs: [f32; 3]) -> f32 {
    lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}

fn cross3(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    ]
}

fn normalize3(value: [f32; 3]) -> [f32; 3] {
    let norm = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    if norm <= EPS {
        [0.0, 0.0, 1.0]
    } else {
        [value[0] / norm, value[1] / norm, value[2] / norm]
    }
}

#[cfg(test)]
mod tests;
