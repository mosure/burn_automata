#![allow(clippy::too_many_arguments)]

mod attributes;
mod image;
mod math;
mod projection;

use serde::{Deserialize, Serialize};

use crate::{AutomataError, AutomataResult, RolloutTrace, TriangleMeshTarget};
use attributes::{state_tail_render_attributes, validate_render_trace};
use burn_automata_kernels::GaussianDecodeMode;
use image::{
    accumulate_projected_position_gradients, image_loss, image_loss_with_adjoint, splat_projected,
};
use math::psnr;
#[cfg(test)]
use math::sigmoid;
use projection::project_positions;

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

#[cfg(test)]
mod tests;
