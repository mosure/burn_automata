use super::*;
use crate::{
    NpaConfig,
    rollout::{ParticleSeed, growth_3d_material_opacity_channel, seed_particles_scaled},
};

#[test]
fn multiview_render_loss_accepts_mesh_samples_against_themselves() {
    let target = TriangleMeshTarget::torus(0.72, 0.72 * 0.72, 24, 16).unwrap();
    let samples = mesh_surface_render_samples(&target, 512);
    let state_dims = 16;
    let opacity_channel = growth_3d_material_opacity_channel(state_dims).unwrap();
    let mut states = vec![0.0; samples.positions.len() * state_dims];
    for (idx, color) in samples.colors.iter().enumerate() {
        states[idx * state_dims + opacity_channel] = 8.0;
        let base = idx * state_dims + state_dims - 3;
        states[base] = 2.0 * color[0] - 1.0;
        states[base + 1] = 2.0 * color[1] - 1.0;
        states[base + 2] = 2.0 * color[2] - 1.0;
    }
    let trace = RolloutTrace {
        batch_size: 1,
        particle_count: samples.positions.len(),
        state_dims,
        steps: 0,
        positions: samples.positions,
        states,
        mean_dx: vec![0.0],
    };

    let report = mesh_multiview_render_loss_from_trace(
        &trace,
        &target,
        RenderLossConfig {
            image_size: 32,
            target_samples: 512,
            world_scale: 1.4,
            ..RenderLossConfig::default()
        },
    )
    .unwrap();

    assert!(
        report.passed,
        "density={} color={} density_psnr={} color_psnr={}",
        report.density_mse, report.color_mse, report.density_psnr_db, report.color_psnr_db
    );
    assert!(report.density_mse <= 1.0e-6);
    assert!(report.color_mse <= 1.0e-6);
    assert_eq!(report.views.len(), 4);
}

#[test]
fn multiview_render_loss_rejects_random_ball_against_torus() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::torus(0.72, 0.72 * 0.72, 24, 16).unwrap();
    let (positions, states) = seed_particles_scaled(
        1,
        512,
        config.state_dims,
        config.spatial_dims,
        42,
        ParticleSeed::UniformCircle,
        0.72,
    );
    let trace = RolloutTrace {
        batch_size: 1,
        particle_count: 512,
        state_dims: config.state_dims,
        steps: 0,
        positions,
        states,
        mean_dx: vec![0.0],
    };

    let report = mesh_multiview_render_loss_from_trace(
        &trace,
        &target,
        RenderLossConfig {
            image_size: 32,
            target_samples: 512,
            world_scale: 1.4,
            ..RenderLossConfig::default()
        },
    )
    .unwrap();

    assert!(
        !report.passed,
        "density_psnr={} color_psnr={} depth_psnr={} density={} color={} depth={}",
        report.density_psnr_db,
        report.color_psnr_db,
        report.depth_psnr_db,
        report.density_mse,
        report.color_mse,
        report.depth_mse
    );
    assert!(report.depth_mse > 1.0e-4);
    assert!(report.nonzero_particle_alpha_fraction > 0.0);
}

#[test]
fn render_loss_weights_particles_by_opacity_state() {
    let target = TriangleMeshTarget::torus(0.72, 0.72 * 0.72, 24, 16).unwrap();
    let samples = mesh_surface_render_samples(&target, 128);
    let state_dims = 16;
    let opacity_channel = growth_3d_material_opacity_channel(state_dims).unwrap();
    let mut visible_states = vec![0.0; samples.positions.len() * state_dims];
    let mut hidden_states = vec![0.0; samples.positions.len() * state_dims];
    for (idx, color) in samples.colors.iter().enumerate() {
        visible_states[idx * state_dims + opacity_channel] = 8.0;
        hidden_states[idx * state_dims + opacity_channel] = -8.0;
        let base = idx * state_dims + state_dims - 3;
        for states in [&mut visible_states, &mut hidden_states] {
            states[base] = 2.0 * color[0] - 1.0;
            states[base + 1] = 2.0 * color[1] - 1.0;
            states[base + 2] = 2.0 * color[2] - 1.0;
        }
    }
    let visible = RolloutTrace {
        batch_size: 1,
        particle_count: samples.positions.len(),
        state_dims,
        steps: 0,
        positions: samples.positions.clone(),
        states: visible_states,
        mean_dx: vec![0.0],
    };
    let hidden = RolloutTrace {
        states: hidden_states,
        ..visible.clone()
    };
    let cfg = RenderLossConfig {
        image_size: 32,
        target_samples: 128,
        world_scale: 1.4,
        color_weight: 0.0,
        depth_weight: 0.0,
        ..RenderLossConfig::default()
    };

    let visible_report = mesh_multiview_render_loss_from_trace(&visible, &target, cfg).unwrap();
    let hidden_report = mesh_multiview_render_loss_from_trace(&hidden, &target, cfg).unwrap();

    assert!(
        visible_report.density_mse < hidden_report.density_mse * 0.01,
        "visible density={} hidden density={}",
        visible_report.density_mse,
        hidden_report.density_mse
    );
    assert!(visible_report.density_psnr_db > hidden_report.density_psnr_db);
}

#[test]
fn render_opacity_logit_bias_calibrates_material_decoder_without_state_mutation() {
    let target = TriangleMeshTarget::torus(0.72, 0.72 * 0.72, 24, 16).unwrap();
    let samples = mesh_surface_render_samples(&target, 128);
    let state_dims = 16;
    let mut states = vec![0.0; samples.positions.len() * state_dims];
    for (idx, color) in samples.colors.iter().enumerate() {
        let base = idx * state_dims + state_dims - 3;
        states[base] = 2.0 * color[0] - 1.0;
        states[base + 1] = 2.0 * color[1] - 1.0;
        states[base + 2] = 2.0 * color[2] - 1.0;
    }
    let trace = RolloutTrace {
        batch_size: 1,
        particle_count: samples.positions.len(),
        state_dims,
        steps: 0,
        positions: samples.positions,
        states: states.clone(),
        mean_dx: vec![0.0],
    };
    let cfg = RenderLossConfig {
        image_size: 32,
        target_samples: 128,
        world_scale: 1.4,
        color_weight: 0.0,
        depth_weight: 0.0,
        ..RenderLossConfig::default()
    };

    let uncalibrated = mesh_multiview_render_loss_from_trace(&trace, &target, cfg).unwrap();
    let calibrated = mesh_multiview_render_loss_from_trace(
        &trace,
        &target,
        RenderLossConfig {
            opacity_logit_bias: 8.0,
            ..cfg
        },
    )
    .unwrap();

    assert_eq!(trace.states, states);
    assert!(
        calibrated.density_mse < uncalibrated.density_mse,
        "positive render-only material bias should improve under-opaque surface samples: uncalibrated={} calibrated={}",
        uncalibrated.density_mse,
        calibrated.density_mse
    );
}

#[test]
fn learned_scale_decode_changes_render_footprint_without_moving_particles() {
    let target = TriangleMeshTarget::torus(0.72, 0.72 * 0.72, 24, 16).unwrap();
    let samples = mesh_surface_render_samples(&target, 256);
    let state_dims = 16;
    let opacity_channel = growth_3d_material_opacity_channel(state_dims).unwrap();
    let scale_channel = state_dims - 5;
    let mut states = vec![0.0; samples.positions.len() * state_dims];
    for (idx, color) in samples.colors.iter().enumerate() {
        let state_base = idx * state_dims;
        states[state_base + opacity_channel] = 8.0;
        let color_base = state_base + state_dims - 3;
        states[color_base] = 2.0 * color[0] - 1.0;
        states[color_base + 1] = 2.0 * color[1] - 1.0;
        states[color_base + 2] = 2.0 * color[2] - 1.0;
    }
    let matching = RolloutTrace {
        batch_size: 1,
        particle_count: samples.positions.len(),
        state_dims,
        steps: 0,
        positions: samples.positions.clone(),
        states: states.clone(),
        mean_dx: vec![0.0],
    };
    let mut large_scale = matching.clone();
    for idx in 0..large_scale.particle_count {
        large_scale.states[idx * state_dims + scale_channel] = 1.25;
    }
    let cfg = RenderLossConfig {
        image_size: 32,
        target_samples: 256,
        sigma: 2.0,
        min_sigma: 0.5,
        max_sigma: 6.0,
        world_scale: 1.4,
        gaussian_decode_mode: GaussianDecodeMode::GaussianSh0LearnedScale,
        ..RenderLossConfig::default()
    };

    let matching_report = mesh_multiview_render_loss_from_trace(&matching, &target, cfg).unwrap();
    let large_report = mesh_multiview_render_loss_from_trace(&large_scale, &target, cfg).unwrap();

    assert!(
        matching_report.density_mse < large_report.density_mse,
        "oversized learned scales should be measured as a density regression: matching={} large={}",
        matching_report.density_mse,
        large_report.density_mse
    );
}

#[test]
fn render_position_gradient_matches_density_finite_difference() {
    let target = TriangleMeshTarget::torus(0.72, 0.72 * 0.72, 24, 16).unwrap();
    let state_dims = 16;
    let positions = vec![
        [0.13, -0.17, 0.09, 0.0],
        [-0.24, 0.31, -0.18, 0.0],
        [0.42, 0.07, 0.21, 0.0],
        [-0.36, -0.28, 0.14, 0.0],
        [0.18, 0.44, -0.33, 0.0],
        [-0.08, -0.41, -0.23, 0.0],
        [0.51, -0.11, 0.04, 0.0],
        [-0.47, 0.16, 0.37, 0.0],
    ];
    let states = vec![0.0; positions.len() * state_dims];
    let trace = RolloutTrace {
        batch_size: 1,
        particle_count: positions.len(),
        state_dims,
        steps: 0,
        positions,
        states,
        mean_dx: vec![0.0],
    };
    let cfg = RenderLossConfig {
        image_size: 24,
        target_samples: 64,
        world_scale: 1.4,
        color_weight: 0.0,
        depth_weight: 0.0,
        ..RenderLossConfig::default()
    };

    let gradient =
        mesh_multiview_render_position_gradient_from_trace(&trace, &target, cfg, 1).unwrap();
    assert!(gradient.loss.total_loss.is_finite());
    for axis in 0..3 {
        let eps = 1.0e-3;
        let mut plus = trace.clone();
        let mut minus = trace.clone();
        plus.positions[0][axis] += eps;
        minus.positions[0][axis] -= eps;
        let plus_loss = mesh_multiview_render_loss_from_trace(&plus, &target, cfg)
            .unwrap()
            .total_loss;
        let minus_loss = mesh_multiview_render_loss_from_trace(&minus, &target, cfg)
            .unwrap()
            .total_loss;
        let finite_difference = (plus_loss - minus_loss) / (2.0 * eps);
        let analytic = gradient.gradients[0][axis];
        assert!(
            (analytic - finite_difference).abs() <= 2.5e-2 + finite_difference.abs() * 0.1,
            "axis={axis} analytic={analytic} finite_difference={finite_difference}"
        );
    }

    let eps = 1.0e-3;
    let mut plus = trace.clone();
    let mut minus = trace.clone();
    let opacity_channel = growth_3d_material_opacity_channel(state_dims).unwrap();
    plus.states[opacity_channel] += eps;
    minus.states[opacity_channel] -= eps;
    let plus_loss = mesh_multiview_render_loss_from_trace(&plus, &target, cfg)
        .unwrap()
        .total_loss;
    let minus_loss = mesh_multiview_render_loss_from_trace(&minus, &target, cfg)
        .unwrap()
        .total_loss;
    let finite_difference = (plus_loss - minus_loss) / (2.0 * eps);
    let analytic = gradient.opacity_gradients[0] * 0.25;
    assert!(
        (analytic - finite_difference).abs() <= 2.5e-2 + finite_difference.abs() * 0.1,
        "opacity analytic={analytic} finite_difference={finite_difference}"
    );
}

#[test]
fn render_scale_gradient_matches_density_finite_difference() {
    let target = TriangleMeshTarget::torus(0.72, 0.72 * 0.72, 24, 16).unwrap();
    let state_dims = 16;
    let positions = vec![
        [0.13, -0.17, 0.09, 0.0],
        [-0.24, 0.31, -0.18, 0.0],
        [0.42, 0.07, 0.21, 0.0],
        [-0.36, -0.28, 0.14, 0.0],
        [0.18, 0.44, -0.33, 0.0],
        [-0.08, -0.41, -0.23, 0.0],
        [0.51, -0.11, 0.04, 0.0],
        [-0.47, 0.16, 0.37, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * state_dims];
    let opacity_channel = growth_3d_material_opacity_channel(state_dims).unwrap();
    for idx in 0..positions.len() {
        states[idx * state_dims + opacity_channel] = 6.0;
    }
    let trace = RolloutTrace {
        batch_size: 1,
        particle_count: positions.len(),
        state_dims,
        steps: 0,
        positions,
        states,
        mean_dx: vec![0.0],
    };
    let cfg = RenderLossConfig {
        image_size: 24,
        target_samples: 64,
        sigma: 2.0,
        min_sigma: 0.5,
        max_sigma: 6.0,
        world_scale: 1.4,
        color_weight: 0.0,
        depth_weight: 0.0,
        gaussian_decode_mode: GaussianDecodeMode::GaussianSh0LearnedScale,
        ..RenderLossConfig::default()
    };

    let gradient =
        mesh_multiview_render_position_gradient_from_trace(&trace, &target, cfg, 1).unwrap();
    let eps = 1.0e-3;
    let scale_channel = state_dims - 5;
    let mut plus = trace.clone();
    let mut minus = trace.clone();
    plus.states[scale_channel] += eps;
    minus.states[scale_channel] -= eps;
    let plus_loss = mesh_multiview_render_loss_from_trace(&plus, &target, cfg)
        .unwrap()
        .total_loss;
    let minus_loss = mesh_multiview_render_loss_from_trace(&minus, &target, cfg)
        .unwrap()
        .total_loss;
    let finite_difference = (plus_loss - minus_loss) / (2.0 * eps);
    let analytic = gradient.scale_gradients[0];

    assert!(
        (analytic - finite_difference).abs() <= 2.5e-2 + finite_difference.abs() * 0.15,
        "scale analytic={analytic} finite_difference={finite_difference}"
    );
}

#[test]
fn render_gradient_matches_full_loss_finite_difference() {
    let target = TriangleMeshTarget::torus(0.72, 0.72 * 0.72, 24, 16).unwrap();
    let state_dims = 16;
    let positions = vec![
        [0.16, -0.19, 0.11, 0.0],
        [-0.29, 0.27, -0.21, 0.0],
        [0.39, 0.09, 0.24, 0.0],
        [-0.33, -0.25, 0.17, 0.0],
        [0.21, 0.41, -0.29, 0.0],
        [-0.11, -0.38, -0.26, 0.0],
        [0.48, -0.14, 0.07, 0.0],
        [-0.44, 0.19, 0.34, 0.0],
    ];
    let opacity_channel = growth_3d_material_opacity_channel(state_dims).unwrap();
    let scale_channel = state_dims - 5;
    let color_base = state_dims - 3;
    let mut states = vec![0.0; positions.len() * state_dims];
    for idx in 0..positions.len() {
        let base = idx * state_dims;
        states[base + opacity_channel] = 0.75;
        states[base + scale_channel] = -0.1;
        states[base + color_base] = -0.25 + idx as f32 * 0.03;
        states[base + color_base + 1] = 0.35 - idx as f32 * 0.02;
        states[base + color_base + 2] = -0.15 + idx as f32 * 0.01;
    }
    let trace = RolloutTrace {
        batch_size: 1,
        particle_count: positions.len(),
        state_dims,
        steps: 0,
        positions,
        states,
        mean_dx: vec![0.0],
    };
    let cfg = RenderLossConfig {
        image_size: 24,
        target_samples: 64,
        sigma: 2.0,
        min_sigma: 0.5,
        max_sigma: 6.0,
        world_scale: 1.4,
        gaussian_decode_mode: GaussianDecodeMode::GaussianSh0LearnedScale,
        density_weight: 1.0,
        color_weight: 0.7,
        depth_weight: 0.5,
        ..RenderLossConfig::default()
    };

    let gradient =
        mesh_multiview_render_position_gradient_from_trace(&trace, &target, cfg, 1).unwrap();
    assert!(gradient.loss.total_loss.is_finite());

    let eps = 1.0e-3;
    for axis in 0..3 {
        let mut plus = trace.clone();
        let mut minus = trace.clone();
        plus.positions[0][axis] += eps;
        minus.positions[0][axis] -= eps;
        let finite_difference = finite_difference_render_loss(&plus, &minus, &target, cfg, eps);
        let analytic = gradient.gradients[0][axis];
        assert_gradient_close("position", axis, analytic, finite_difference, 0.05, 0.20);
    }

    let mut plus = trace.clone();
    let mut minus = trace.clone();
    plus.states[opacity_channel] += eps;
    minus.states[opacity_channel] -= eps;
    let finite_difference = finite_difference_render_loss(&plus, &minus, &target, cfg, eps);
    let opacity = sigmoid(trace.states[opacity_channel] + cfg.opacity_logit_bias);
    let analytic = gradient.opacity_gradients[0] * opacity * (1.0 - opacity);
    assert_gradient_close("opacity", 0, analytic, finite_difference, 0.05, 0.20);

    let mut plus = trace.clone();
    let mut minus = trace.clone();
    plus.states[color_base] += eps;
    minus.states[color_base] -= eps;
    let finite_difference = finite_difference_render_loss(&plus, &minus, &target, cfg, eps);
    let analytic = 0.5 * gradient.color_gradients[0][0];
    assert_gradient_close("color", 0, analytic, finite_difference, 0.05, 0.20);

    let mut plus = trace.clone();
    let mut minus = trace.clone();
    plus.states[scale_channel] += eps;
    minus.states[scale_channel] -= eps;
    let finite_difference = finite_difference_render_loss(&plus, &minus, &target, cfg, eps);
    let analytic = gradient.scale_gradients[0];
    assert_gradient_close("scale", 0, analytic, finite_difference, 0.05, 0.20);
}

fn finite_difference_render_loss(
    plus: &RolloutTrace,
    minus: &RolloutTrace,
    target: &TriangleMeshTarget,
    cfg: RenderLossConfig,
    eps: f32,
) -> f32 {
    let plus_loss = mesh_multiview_render_loss_from_trace(plus, target, cfg)
        .unwrap()
        .total_loss;
    let minus_loss = mesh_multiview_render_loss_from_trace(minus, target, cfg)
        .unwrap()
        .total_loss;
    (plus_loss - minus_loss) / (2.0 * eps)
}

fn assert_gradient_close(
    name: &str,
    axis: usize,
    analytic: f32,
    finite_difference: f32,
    abs_tol: f32,
    rel_tol: f32,
) {
    assert!(
        (analytic - finite_difference).abs() <= abs_tol + finite_difference.abs() * rel_tol,
        "{name} axis={axis} analytic={analytic} finite_difference={finite_difference}"
    );
}
