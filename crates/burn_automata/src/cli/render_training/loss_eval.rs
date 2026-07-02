use super::*;

pub(crate) fn default_render_loss_config(seed_scale: f32) -> RenderLossConfig {
    RenderLossConfig {
        world_scale: seed_scale.max(1.0e-4) * 2.0,
        target_samples: 0,
        ..RenderLossConfig::default()
    }
}

pub(crate) fn mesh_render_loss_for_model(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: RenderLossEvalConfig,
) -> Result<MultiViewRenderLossReport, Box<dyn std::error::Error>> {
    let mut render_cfg = cfg.render;
    if render_cfg.target_samples == 0 {
        render_cfg.target_samples = cfg.particle_count;
    }
    let seeds = eval_seed_list(cfg.seed, &cfg.extra_seeds);
    let mut reports = Vec::with_capacity(seeds.len());
    for seed in seeds {
        let trace = run_rollout(
            model,
            grid,
            &RolloutConfig {
                particle_count: cfg.particle_count,
                steps: cfg.steps,
                update_prob: 1.0,
                seed,
                seed_scale: cfg.seed_scale,
                ..RolloutConfig::default()
            },
            cfg.seed_mode,
        )?;
        reports.push(mesh_multiview_render_loss_from_trace(
            &trace, target, render_cfg,
        )?);
    }
    Ok(average_render_loss_reports(reports, render_cfg))
}

pub(crate) fn eval_seed_list(seed: u64, extra_seeds: &[u64]) -> Vec<u64> {
    let mut seeds = Vec::with_capacity(extra_seeds.len() + 1);
    seeds.push(seed);
    for &extra_seed in extra_seeds {
        if !seeds.contains(&extra_seed) {
            seeds.push(extra_seed);
        }
    }
    seeds
}

pub(crate) fn average_render_loss_reports(
    reports: Vec<MultiViewRenderLossReport>,
    cfg: RenderLossConfig,
) -> MultiViewRenderLossReport {
    let count = reports.len().max(1) as f32;
    let first = reports
        .first()
        .cloned()
        .unwrap_or_else(|| empty_render_loss_report(cfg));
    if reports.len() <= 1 {
        return first;
    }

    let views = (0..first.views.len())
        .map(|view_idx| {
            let view = first.views[view_idx].view;
            let view_reports: Vec<&RenderViewLossReport> = reports
                .iter()
                .filter_map(|report| report.views.get(view_idx))
                .collect();
            let view_count = view_reports.len().max(1) as f32;
            let density_mse = view_reports
                .iter()
                .map(|report| report.density_mse)
                .sum::<f32>()
                / view_count;
            let color_mse = view_reports
                .iter()
                .map(|report| report.color_mse)
                .sum::<f32>()
                / view_count;
            let depth_mse = view_reports
                .iter()
                .map(|report| report.depth_mse)
                .sum::<f32>()
                / view_count;
            let nonzero_target_alpha_fraction = view_reports
                .iter()
                .map(|report| report.nonzero_target_alpha_fraction)
                .sum::<f32>()
                / view_count;
            let nonzero_particle_alpha_fraction = view_reports
                .iter()
                .map(|report| report.nonzero_particle_alpha_fraction)
                .sum::<f32>()
                / view_count;
            RenderViewLossReport {
                view,
                total_loss: cfg.density_weight * density_mse
                    + cfg.color_weight * color_mse
                    + cfg.depth_weight * depth_mse,
                density_mse,
                color_mse,
                depth_mse,
                density_psnr_db: render_psnr_db(density_mse, 1.0),
                color_psnr_db: render_psnr_db(color_mse, 1.0),
                depth_psnr_db: render_psnr_db(depth_mse, 1.0),
                nonzero_target_alpha_fraction,
                nonzero_particle_alpha_fraction,
            }
        })
        .collect::<Vec<_>>();

    let density_mse = reports.iter().map(|report| report.density_mse).sum::<f32>() / count;
    let color_mse = reports.iter().map(|report| report.color_mse).sum::<f32>() / count;
    let depth_mse = reports.iter().map(|report| report.depth_mse).sum::<f32>() / count;
    let density_psnr_db = render_psnr_db(density_mse, 1.0);
    let color_psnr_db = render_psnr_db(color_mse, 1.0);
    let depth_psnr_db = render_psnr_db(depth_mse, 1.0);
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
    let finite = reports.iter().all(|report| {
        report.total_loss.is_finite()
            && report.density_mse.is_finite()
            && report.color_mse.is_finite()
            && report.depth_mse.is_finite()
            && report.nonzero_particle_alpha_fraction > 0.0
    });
    MultiViewRenderLossReport {
        passed: finite
            && reports.iter().all(|report| report.passed)
            && density_psnr_db >= 10.0
            && color_psnr_db >= 12.0
            && depth_psnr_db >= 14.0,
        image_size: cfg.image_size,
        target_samples: cfg.target_samples,
        total_loss: cfg.density_weight * density_mse
            + cfg.color_weight * color_mse
            + cfg.depth_weight * depth_mse,
        density_mse,
        color_mse,
        depth_mse,
        density_psnr_db,
        color_psnr_db,
        depth_psnr_db,
        nonzero_target_alpha_fraction,
        nonzero_particle_alpha_fraction,
        views,
    }
}

pub(crate) fn render_psnr_db(mse: f32, max_value: f32) -> f32 {
    if mse <= 0.0 {
        99.0
    } else {
        10.0 * ((max_value * max_value) / mse.max(1.0e-8)).log10()
    }
}

pub(crate) fn empty_render_loss_report(cfg: RenderLossConfig) -> MultiViewRenderLossReport {
    MultiViewRenderLossReport {
        passed: false,
        image_size: cfg.image_size,
        target_samples: cfg.target_samples,
        total_loss: f32::INFINITY,
        density_mse: f32::INFINITY,
        color_mse: f32::INFINITY,
        depth_mse: f32::INFINITY,
        density_psnr_db: f32::NEG_INFINITY,
        color_psnr_db: f32::NEG_INFINITY,
        depth_psnr_db: f32::NEG_INFINITY,
        nonzero_target_alpha_fraction: 0.0,
        nonzero_particle_alpha_fraction: 0.0,
        views: Vec::new(),
    }
}
